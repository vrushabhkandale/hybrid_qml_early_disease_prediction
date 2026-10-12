"""
calibration.py — Post-Hoc Probability Calibration & Threshold Optimization
==========================================================================
Modularized calibration routines for Quantum-Classical Neural Networks:
  • _collect_val_logits: Gathers unscaled model logits on validation/test sets
  • optimize_temperature_scaling: Strictly positive latent temperature parameterization
  • QuantumBetaCalibrator: Beta calibration for bounded quantum output confidences
  • QuantumMultiClassCalibrator: Multi-class Dirichlet probability calibration
  • optimize_multiclass_cost_sensitive_thresholds: Clinical cost matrix formulation
  • predict_with_cost_matrix: Bayes-risk minimal classification decisions
  • optimize_decision_threshold: Multi-criteria threshold tuner (Youden, Spec, Cost)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def _collect_val_logits(
    model: nn.Module,
    val_loader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect all validation logits and labels for calibration with single host-device transfer."""
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.squeeze().long().to(device, non_blocking=True)
        try:
            logits = model(images, use_cached_features=True)
        except TypeError:
            logits = model(images)
        all_logits.append(logits)
        all_labels.append(labels)
    # Consolidated device-to-host memory transfer to avoid per-batch CUDA pipeline flushes
    return torch.cat(all_logits, dim=0).cpu(), torch.cat(all_labels, dim=0).cpu()


def optimize_temperature_scaling(
    model_or_logits,
    val_loader_or_labels=None,
    device: Optional[torch.device] = None,
    lr: float = 0.01,
    max_iter: int = 100,
    t_min: float = 0.10,
) -> Union[float, Tuple[float, torch.Tensor]]:
    r"""Platt Temperature Scaling with Strictly Positive Latent Parameterization.

    Supports two call signatures:
      1. Tensor-based (Phase 5 OOF calibration)::

             T_opt, cal_probs = optimize_temperature_scaling(
                 logits=oof_logits,   # torch.Tensor [N, C]
                 labels=oof_labels,   # torch.Tensor [N]  (long)
                 lr=0.01, max_iter=100, t_min=0.10
             )

      2. Legacy model+loader form (preserved for backward compat)::

             T_opt = optimize_temperature_scaling(model, val_loader, device)

    Returns float T_opt (legacy mode) or (float T_opt, Tensor calibrated_probs) in tensor mode.

    T(tau) = t_min + softplus(tau) >= t_min for all tau in R.
    Init: tau_0 = ln(exp(1 - t_min) - 1)  =>  T(tau_0) = 1.0
    """
    # ── Detect call signature ─────────────────────────────────────────────
    tensor_mode = isinstance(model_or_logits, torch.Tensor)
    if tensor_mode:
        logits: torch.Tensor = model_or_logits
        labels: torch.Tensor = val_loader_or_labels.long().to(logits.device)
    else:
        # Legacy: collect logits from model + val_loader
        logits, labels = _collect_val_logits(model_or_logits, val_loader_or_labels, device)
        labels = labels.long().to(logits.device)
        t_min = 0.10          # enforce plan invariant in legacy mode too
        max_iter = max(max_iter, 50)

    # ── Softplus-bounded optimization ─────────────────────────────────────
    tau_init = float(math.log(math.exp(1.0 - t_min) - 1.0))   # T(tau_0) = 1.0
    tau = nn.Parameter(torch.tensor([tau_init], device=logits.device))
    optimizer = torch.optim.LBFGS([tau], lr=lr, max_iter=max_iter)

    def eval_nll():
        optimizer.zero_grad()
        T = t_min + F.softplus(tau)
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(eval_nll)
    T_opt = float((t_min + F.softplus(tau)).item())
    assert T_opt >= t_min, f"Temperature invariant violated: T* = {T_opt:.6f} < {t_min}"
    print(f"[calibration] \u2713 Strictly positive temperature: T* = {T_opt:.4f} (tau={float(tau.item()):.4f}, t_min={t_min})")

    if tensor_mode:
        calibrated_probs = F.softmax(logits / T_opt, dim=-1).detach()
        return T_opt, calibrated_probs
    return T_opt


class QuantumBetaCalibrator(nn.Module):
    """Beta Calibration for Bounded Quantum Neural Network Outputs (Kull et al., AISTATS 2017).

    Models the log-odds ratio of two Beta distributions:
        ln P(Y=1|s) / P(Y=0|s) = a·ln(s) - b·ln(1-s) + c
    where s ∈ (0,1) is the raw model confidence score.

    Exact Identity Invariant:
        When a_pos = b_pos = 1.0 and c = 0.0, Calibrate(s) = s.
        Since a_pos = softplus(a), solving softplus(a_0) = 1.0 yields:
        a_0 = ln(exp(1) - 1) ≈ 0.5413248546129181.
    """
    def __init__(self, input_is_pauli_expectation: bool = False):
        super().__init__()
        self.input_is_pauli_expectation = input_is_pauli_expectation
        # Exact inverse softplus initialization: softplus(0.54132485) == 1.0
        inv_softplus_one = float(math.log(math.exp(1.0) - 1.0))
        self.a = nn.Parameter(torch.full((1,), inv_softplus_one, dtype=torch.float32))
        self.b = nn.Parameter(torch.full((1,), inv_softplus_one, dtype=torch.float32))
        self.c = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, probs: torch.Tensor, is_pauli_expectation: Optional[bool] = None) -> torch.Tensor:
        """Calibrate raw probabilities or Pauli expectation values using Beta calibration transform."""
        is_pauli = self.input_is_pauli_expectation if is_pauli_expectation is None else is_pauli_expectation
        p1 = probs[:, 1] if (probs.ndim == 2 and probs.shape[1] > 1) else probs
        # Quantum Born-rule affine projection for raw Pauli expectation values in [-1, 1]
        if is_pauli:
            p1 = (p1 + 1.0) / 2.0
        eps = 1e-7
        p1_clamped = torch.clamp(p1, eps, 1.0 - eps)
        a_pos = F.softplus(self.a)
        b_pos = F.softplus(self.b)
        cal_logits = a_pos * torch.log(p1_clamped) - b_pos * torch.log(1.0 - p1_clamped) + self.c
        cal_p1 = torch.sigmoid(cal_logits)
        if probs.ndim == 2 and probs.shape[1] > 1:
            return torch.stack([1.0 - cal_p1, cal_p1], dim=-1)
        return cal_p1

    def fit(self, val_probs: torch.Tensor, val_labels: torch.Tensor):
        """Fit parameters a, b, c on validation set via L-BFGS with identity-anchored regularization."""
        self.to(val_probs.device)
        optimizer = torch.optim.LBFGS(self.parameters(), lr=0.01, max_iter=100)
        labels_f = val_labels.float().view(-1).to(val_probs.device)

        def closure():
            optimizer.zero_grad()
            cal_p = self.forward(val_probs)
            cal_p1 = cal_p[:, 1] if (cal_p.ndim == 2 and cal_p.shape[1] > 1) else cal_p
            cal_p1 = cal_p1.view(-1)
            eps = 1e-7
            nll = -torch.mean(labels_f * torch.log(cal_p1 + eps) + (1.0 - labels_f) * torch.log(1.0 - cal_p1 + eps))
            # Regularize effective parameters towards identity (a_pos=1.0, b_pos=1.0, c=0.0)
            a_pos = F.softplus(self.a)
            b_pos = F.softplus(self.b)
            reg = 1e-3 * ((a_pos - 1.0) ** 2 + (b_pos - 1.0) ** 2 + self.c ** 2)
            loss = nll + reg
            loss.backward()
            return loss

        optimizer.step(closure)
        a_val = float(F.softplus(self.a).item())
        b_val = float(F.softplus(self.b).item())
        c_val = float(self.c.item())
        print(f"[beta-calibration] Fitted: a_pos={a_val:.4f}, b_pos={b_val:.4f}, c={c_val:.4f}")


class QuantumMultiClassCalibrator(nn.Module):
    r"""
    Multi-Class Dirichlet Probability Calibrator (Kull et al., NeurIPS 2019; Fix F-01 & Solution 1.5).
    Calibrates multi-class bounded quantum neural network outputs via L2-regularized multinomial
    logistic regression over log-probabilities:
        ln p_cal = softmax(W_cal * ln p_raw + b_cal)
    Parameters:
        W_cal in R^{K x K}, b_cal in R^K
    Loss:
        L_Dir = NLL + lambda_diag * ||diag(W) - I||^2 + lambda_off * ||off_diag(W)||^2
    """
    def __init__(self, n_classes: int, lambda_diag: float = 1e-3, lambda_off: float = 1e-2):
        super().__init__()
        self.n_classes = n_classes
        self.lambda_diag = lambda_diag
        self.lambda_off = lambda_off
        self.W = nn.Parameter(torch.eye(n_classes))
        self.b = nn.Parameter(torch.zeros(n_classes))

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        p_clamped = torch.clamp(probs, eps, 1.0)
        ln_p = torch.log(p_clamped)
        cal_logits = F.linear(ln_p, self.W, self.b)
        return F.softmax(cal_logits, dim=-1)

    def fit(self, val_probs: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 150):
        optimizer = torch.optim.LBFGS(self.parameters(), lr=0.05, max_iter=max_iter)
        labels_idx = val_labels.long().to(val_probs.device)
        K = self.n_classes
        eye_K = torch.eye(K, device=val_probs.device)

        def closure():
            optimizer.zero_grad()
            cal_p = self.forward(val_probs)
            eps = 1e-12
            nll = F.nll_loss(torch.log(torch.clamp(cal_p, eps, 1.0)), labels_idx)
            diag_diff = torch.diag(self.W) - 1.0
            off_diag = self.W * (1.0 - eye_K)
            reg = self.lambda_diag * torch.sum(diag_diff ** 2) + self.lambda_off * torch.sum(off_diag ** 2)
            loss = nll + reg
            loss.backward()
            return loss

        optimizer.step(closure)
        print(f"[dirichlet-calibration] Fitted Multi-Class Calibrator for {K} classes.")


def optimize_multiclass_cost_sensitive_thresholds(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    n_classes: int = 7,
    target_specificity: float = 0.90,
    cost_matrix: Optional[np.ndarray] = None,
    n_steps: int = 200,
) -> np.ndarray:
    r"""Vectorized Cost-Sensitive Multi-Class Threshold Optimization.

    If cost_matrix is supplied:
      Finds the per-class threshold vector tau* in [0.005, 0.995]^K that minimizes
      the empirical clinical Bayes-risk while resolving multi-label collisions by
      margin ratio (p_c / tau_c).

    If cost_matrix is None:
      Vectorized OVR threshold grid sweep selecting the threshold meeting
      Specificity_c >= target_specificity with maximal Sensitivity_c.
    """
    val_labels = np.asarray(val_labels).squeeze().astype(int)
    val_probs = np.asarray(val_probs)
    taus = np.linspace(0.005, 0.995, n_steps)
    thresholds = np.full(n_classes, 0.5)

    # ── Fully Vectorized Threshold Confusion Matrix Computations ──────────
    # val_probs: (N, K), taus: (S,) -> preds: (N, K, S)
    preds_all = val_probs[:, :, None] >= taus[None, None, :]
    y_true_all = (val_labels[:, None, None] == np.arange(n_classes)[None, :, None])

    tp = np.sum(preds_all & y_true_all, axis=0)  # (K, S)
    fp = np.sum(preds_all & (~y_true_all), axis=0)
    tn = np.sum((~preds_all) & (~y_true_all), axis=0)
    fn = np.sum((~preds_all) & y_true_all, axis=0)

    sens = tp / np.maximum(tp + fn, 1)
    spec = tn / np.maximum(tn + fp, 1)

    if cost_matrix is not None:
        cost_matrix = np.asarray(cost_matrix)
        for c in range(n_classes):
            other = [j for j in range(n_classes) if j != c]
            # C[action, true]: FN for true c predicted as another action;
            # FP for another true class predicted as action c.
            cost_fn = float(np.mean(cost_matrix[other, c]))
            cost_fp = float(np.mean(cost_matrix[c, other]))
            class_risk = cost_fn * fn[c] + cost_fp * fp[c]  # (S,)
            valid_spec = np.where(spec[c] >= target_specificity)[0]
            if len(valid_spec) > 0:
                best_step = valid_spec[np.argmin(class_risk[valid_spec])]
            else:
                best_step = int(np.argmin(class_risk))
            thresholds[c] = float(taus[best_step])
            print(f"[threshold/cost-sensitive/class-{c}] ✓ τ*={thresholds[c]:.3f} (Sens={sens[c, best_step]:.4f}, Spec={spec[c, best_step]:.4f}, Risk={class_risk[best_step]:.1f})")
    else:
        for c in range(n_classes):
            valid_spec = np.where(spec[c] >= target_specificity)[0]
            if len(valid_spec) > 0:
                best_step = valid_spec[np.argmax(sens[c, valid_spec])]
            else:
                best_step = int(np.argmax(spec[c]))
                print(f"[threshold/class-{c}] ⚠ No τ meets Spec≥{target_specificity:.2f}; using τ={taus[best_step]:.3f}")
            thresholds[c] = float(taus[best_step])
            print(f"[threshold/class-{c}] ✓ τ*={thresholds[c]:.3f} (Sens={sens[c, best_step]:.4f}, Spec={spec[c, best_step]:.4f})")

    return thresholds


def predict_with_cost_matrix(probs: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
    """Compute risk-minimal prediction vector given posterior probabilities and clinical cost matrix.

    ``cost_matrix[action, true_class]`` is the cost convention used by the
    training engine. Therefore ``R(action | x) = C @ p``; multiplying on the
    other side silently transposes the clinical policy.
    """
    probs = np.asarray(probs)
    cost_matrix = np.asarray(cost_matrix)
    if probs.ndim != 2 or cost_matrix.ndim != 2:
        raise ValueError("probs and cost_matrix must both be rank-2 arrays")
    if probs.shape[1] != cost_matrix.shape[1] or cost_matrix.shape[0] != cost_matrix.shape[1]:
        raise ValueError(
            f"Expected probs shape (N,K) and square cost matrix (K,K), got "
            f"{probs.shape} and {cost_matrix.shape}"
        )
    expected_risk = np.matmul(probs, cost_matrix.T)
    return np.argmin(expected_risk, axis=1)


def build_mondrian_conformal_artifact(
    probabilities: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Build finite-sample class-conditional split-conformal artifacts.

    Scores and quantiles are kept per true class.  This avoids allowing a
    large/easy class to determine the prediction-set threshold for a rare or
    clinically ambiguous class.
    """
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1).astype(int)
    if probs.ndim != 2 or probs.shape[0] != labels.size:
        raise ValueError(f"Expected probabilities (N,K) and labels (N,), got {probs.shape}, {labels.shape}")
    if not np.isfinite(probs).all():
        raise ValueError("probabilities contain NaN/Inf values")
    n_classes = probs.shape[1]
    if np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError("labels must be valid probability-column indices")
    if np.any(probs < 0) or np.any(probs > 1) or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("probability rows must be nonnegative and sum to one")
    scores_by_class: Dict[str, list[float]] = {}
    quantiles: Dict[str, float] = {}
    for class_idx in range(n_classes):
        class_scores = (1.0 - probs[labels == class_idx, class_idx]).astype(np.float64)
        class_scores = np.clip(class_scores, 0.0, 1.0)
        key = str(class_idx)
        scores_by_class[key] = class_scores.tolist()
        if class_scores.size == 0:
            quantiles[key] = 1.0
            continue
        quantiles[key] = conformal_quantile(class_scores, alpha)
    return {
        "method": "mondrian_split_conformal",
        "alpha": float(alpha),
        "n_classes": int(n_classes),
        "scores_by_class": scores_by_class,
        "quantiles": quantiles,
    }


def mondrian_prediction_set(
    probabilities: np.ndarray,
    artifact: Optional[Dict[str, Any]],
    alpha: Optional[float] = None,
) -> np.ndarray:
    """Apply a persisted Mondrian conformal artifact to one probability vector."""
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if probs.size == 0 or not np.isfinite(probs).all():
        raise ValueError("probabilities must be a non-empty finite vector")
    if not artifact or "quantiles" not in artifact:
        return np.asarray([int(np.argmax(probs))], dtype=int)
    if alpha is not None and abs(float(artifact.get("alpha", alpha)) - float(alpha)) > 1e-9:
        raise ValueError("Requested alpha does not match persisted conformal artifact")
    quantiles = artifact["quantiles"]
    selected = [
        class_idx for class_idx, probability in enumerate(probs)
        if float(probability) >= 1.0 - float(quantiles.get(str(class_idx), 1.0)) - 1e-12
    ]
    return np.asarray(selected, dtype=int)


def conformal_quantile(scores: np.ndarray, alpha: float = 0.05) -> float:
    """Exact split-conformal order statistic, with conservative sparse-class handling."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    values = np.asarray(scores, dtype=float).reshape(-1)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("conformal scores must be finite and in [0, 1]")
    rank = int(np.ceil((len(values) + 1) * (1 - alpha)))
    if rank > len(values) or not len(values):
        return 1.0  # Equivalent to +infinity for bounded 1-p scores.
    return float(np.partition(values, rank - 1)[rank - 1])


def split_calibration_indices(labels: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Separate probability-calibrator fitting from conformal-score estimation."""
    labels = np.asarray(labels).reshape(-1)
    rng = np.random.default_rng(seed)
    fit_indices, score_indices = [], []
    for class_idx in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == class_idx))
        if len(indices) < 2:
            fit_indices.extend(indices.tolist())
        else:
            split = max(1, len(indices) // 2)
            fit_indices.extend(indices[:split].tolist())
            score_indices.extend(indices[split:].tolist())
    return np.asarray(sorted(fit_indices), dtype=int), np.asarray(sorted(score_indices), dtype=int)


def optimize_decision_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    n_steps: int = 100,
    criterion: str = "youden",
    target_specificity: float = 0.90,
    cost_fp: float = 1.5,
    cost_fn: float = 1.0,
    beta: float = 0.5,
) -> Tuple[float, float]:
    r"""Multi-criteria decision threshold optimization for calibrated binary classification.

    Sweeps tau across [0.005, 0.995] to support rare malignancy screening operating regimes.
    """
    p_pos = probs[:, 1] if (probs.ndim == 2 and probs.shape[1] > 1) else probs.squeeze()
    flat_labels = labels.squeeze().astype(int)

    # Expanded search interval [0.005, 0.995] eliminating artificial sensitivity floor
    taus = np.linspace(0.005, 0.995, n_steps)
    sens_list, spec_list, prec_list, f_beta_list = [], [], [], []
    risk_list, j_list = [], []

    for tau in taus:
        preds = (p_pos >= tau).astype(int)
        tp = float(np.sum((preds == 1) & (flat_labels == 1)))
        tn = float(np.sum((preds == 0) & (flat_labels == 0)))
        fp = float(np.sum((preds == 1) & (flat_labels == 0)))
        fn = float(np.sum((preds == 0) & (flat_labels == 1)))

        sens = tp / max(tp + fn, 1.0)
        spec = tn / max(tn + fp, 1.0)
        prec = tp / max(tp + fp, 1.0)
        j = sens + spec - 1.0

        b2 = beta ** 2
        f_b = (1.0 + b2) * prec * sens / max(b2 * prec + sens, 1e-8)
        risk = cost_fp * fp + cost_fn * fn

        sens_list.append(sens)
        spec_list.append(spec)
        prec_list.append(prec)
        j_list.append(j)
        f_beta_list.append(f_b)
        risk_list.append(risk)

    sens_arr = np.array(sens_list)
    spec_arr = np.array(spec_list)
    j_arr = np.array(j_list)
    f_beta_arr = np.array(f_beta_list)
    risk_arr = np.array(risk_list)

    # 1. Youden's J
    idx_youden = int(np.argmax(j_arr))

    # 2. Target Specificity: maximize sensitivity among all thresholds meeting target_specificity
    valid_spec_indices = np.where(spec_arr >= target_specificity)[0]
    if len(valid_spec_indices) > 0:
        best_in_valid = np.argmax(sens_arr[valid_spec_indices])
        idx_target_spec = int(valid_spec_indices[best_in_valid])
    else:
        idx_target_spec = int(np.argmax(spec_arr))

    # 3. F_beta, Bayesian Cost, and Equal Error Rate
    idx_f_beta = int(np.argmax(f_beta_arr))
    idx_cost = int(np.argmin(risk_arr))
    idx_eer = int(np.argmin(np.abs(sens_arr - spec_arr)))

    criterion_map = {
        "youden": idx_youden,
        "target_specificity": idx_target_spec,
        "f_beta": idx_f_beta,
        "cost_sensitive": idx_cost,
        "balanced_eer": idx_eer,
    }
    selected_idx = criterion_map.get(criterion, idx_youden)
    best_tau = float(taus[selected_idx])
    best_score = float(j_arr[selected_idx]) if criterion == "youden" else float(spec_arr[selected_idx])

    print(f"[calibration] Decision Threshold Optimization (Criterion: '{criterion}'):")
    print(f"       • Youden τ*              = {taus[idx_youden]:.4f}  (Sens={sens_arr[idx_youden]:.4f}, Spec={spec_arr[idx_youden]:.4f}, J={j_arr[idx_youden]:.4f})")
    print(f"       • Target Spec (≥{target_specificity:.2f}) τ* = {taus[idx_target_spec]:.4f}  (Sens={sens_arr[idx_target_spec]:.4f}, Spec={spec_arr[idx_target_spec]:.4f})")
    print(f"       • F_{beta:.1f} Score τ*        = {taus[idx_f_beta]:.4f}  (Sens={sens_arr[idx_f_beta]:.4f}, Spec={spec_arr[idx_f_beta]:.4f}, F_{beta}={f_beta_arr[idx_f_beta]:.4f})")
    print(f"       • Cost-Sensitive τ*      = {taus[idx_cost]:.4f}  (Sens={sens_arr[idx_cost]:.4f}, Spec={spec_arr[idx_cost]:.4f}, Risk={risk_arr[idx_cost]:.1f})")
    print(f"       • Balanced EER τ*        = {taus[idx_eer]:.4f}  (Sens={sens_arr[idx_eer]:.4f}, Spec={spec_arr[idx_eer]:.4f})")
    print(f"       → APPLIED DECISION THRESHOLD: τ* = {best_tau:.4f}  (Sens={sens_arr[selected_idx]:.4f}, Spec={spec_arr[selected_idx]:.4f})")

    return best_tau, best_score


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE) with exact boundary inclusion for edge bins."""
    labels = labels.view(-1)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    confidences, predictions = torch.max(probs, dim=-1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=probs.device)
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])):
        # Include lower edge for first bin so exact 0.0 confidence predictions are never excluded
        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = confidences.gt(bin_lower.item()) & confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece.item())


def compute_classwise_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Class-Wise Expected Calibration Error (ECE_CW) preventing minority-pathology blindness."""
    labels = labels.view(-1)
    K = probs.shape[-1]
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    cw_ece_sum = torch.zeros(1, device=probs.device)

    for k in range(K):
        p_k = probs[:, k]
        y_k = (labels == k).float()
        for i, (bin_lower, bin_upper) in enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])):
            if i == 0:
                in_bin = (p_k >= bin_lower) & (p_k <= bin_upper)
            else:
                in_bin = p_k.gt(bin_lower.item()) & p_k.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                acc_in_bin = y_k[in_bin].mean()
                conf_in_bin = p_k[in_bin].mean()
                cw_ece_sum += torch.abs(acc_in_bin - conf_in_bin) * prop_in_bin

    return float((cw_ece_sum / float(K)).item())


@torch.no_grad()
def evaluate_shortcut_calibration(
    model: nn.Module,
    val_loader,
    device: torch.device,
    T_opt: float = 1.0,
    border_fraction: float = 0.20,
) -> Dict[str, float]:
    """
    Evaluate whether the calibrated model is over-confident on shortcut-only inputs.
    Applies neutral value (0.0 in standardized space) to center anatomical region.
    """
    model.eval()
    orig_probs_list, border_probs_list, labels_list = [], [], []

    for batch in val_loader:
        if isinstance(batch, (tuple, list)):
            imgs, lbls = batch[0], batch[1]
        elif isinstance(batch, dict):
            imgs = batch.get("image", batch.get("images"))
            lbls = batch.get("label", batch.get("labels"))
        else:
            continue

        imgs = imgs.to(device, non_blocking=True).float()
        lbls = lbls.to(device, non_blocking=True).long()
        if lbls.ndim > 1:
            lbls = lbls.squeeze()

        # Original predictions
        logits_orig = model(imgs)
        probs_orig = F.softmax(logits_orig / max(T_opt, 1e-6), dim=-1)

        # Border-only inputs (mask center anatomical region with neutral 0.0 in standardized space)
        B, C, H, W = imgs.shape
        bh, bw = int(H * border_fraction), int(W * border_fraction)
        border_imgs = imgs.clone()
        border_imgs[:, :, bh:-bh, bw:-bw] = 0.0

        logits_border = model(border_imgs)
        probs_border = F.softmax(logits_border / max(T_opt, 1e-6), dim=-1)

        orig_probs_list.append(probs_orig)
        border_probs_list.append(probs_border)
        labels_list.append(lbls)

    if not orig_probs_list:
        return {
            "ece_original": 0.0,
            "ece_border_only": 0.0,
            "confidence_border": 0.5,
            "shortcut_calibration_gap": 0.0,
        }

    # Consolidated device-to-host transfer
    all_orig_probs = torch.cat(orig_probs_list, dim=0).cpu()
    all_border_probs = torch.cat(border_probs_list, dim=0).cpu()
    all_labels = torch.cat(labels_list, dim=0).cpu()

    ece_orig = compute_ece(all_orig_probs, all_labels)
    ece_border = compute_ece(all_border_probs, all_labels)
    max_conf_border, _ = torch.max(all_border_probs, dim=-1)
    conf_border = float(max_conf_border.mean().item())
    gap = conf_border - 0.5

    result = {
        "ece_original": float(ece_orig),
        "ece_border_only": float(ece_border),
        "confidence_border": float(conf_border),
        "shortcut_calibration_gap": float(gap),
        "classwise_ece_orig": compute_classwise_ece(all_orig_probs, all_labels),
    }

    if conf_border > 0.60:
        print(f"[calibration] WARNING: High shortcut overconfidence detected on border-only images: {conf_border:.4f} > 0.60 (Gap={gap:+.4f})")
    else:
        print(f"[calibration] Shortcut calibration check passed: border confidence = {conf_border:.4f} (Gap={gap:+.4f})")

    return result
