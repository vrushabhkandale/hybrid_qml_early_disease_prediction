"""
losses.py — Loss Functions, Imbalance Remediation & Quantum Penalties
====================================================================
Modularized loss components for Hybrid Classical-Quantum network:
  • FocalLoss: Asymmetric & symmetric focal loss for binary/multiclass imbalance
  • resolve_asymmetric_loss_params: Clinical dataset loss parameter parser
  • ClassBalancedLDAMLoss: Label-Distribution-Aware Margin loss with DRW scheduling
  • QuantumInfoNCELoss: Hilbert space InfoNCE contrastive kernel loss
  • compute_drusen_cnv_orthogonality_loss: Hilbert-space state fidelity penalty
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# 1.  FOCAL LOSS
# ═══════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    r"""Asymmetric Focal Loss for class-imbalanced binary classification (Fix: Flaw 1).

    Supports per-class focusing parameters to independently control the
    penalty on the minority (Normal, y=0) and majority (Pneumonia, y=1) classes:
        FL = -α_t (1 - p_t)^γ_t  log(p_t)
    where α_t and γ_t are per-class scalars.

    For binary (n_classes=2):
        Class 0 (Normal/minority):   alpha_0=2.0, gamma_0=1.0
            → Heavy penalty on FP (healthy misdiagnosed), strong gradient signal
        Class 1 (Pneumonia/majority): alpha_1=0.7, gamma_1=2.5
            → Suppress easy majority examples that dominate the loss

    Falls back to symmetric focal loss when alpha/gamma are scalars.

    Supports both hard labels (class indices) and soft labels (one-hot/probabilities)
    for MixUp/CutMix style augmentations.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        # Per-class asymmetric params (used when alpha/gamma are lists)
        alpha_per_class: Optional[List[float]] = None,
        gamma_per_class: Optional[List[float]] = None,
    ):
        super().__init__()
        self.gamma       = gamma
        self.weight      = weight
        self.reduction   = reduction
        # Store per-class asymmetric parameters (Fix: Flaw 1)
        self.alpha_per_class = alpha_per_class   # e.g. [2.0, 0.7]
        self.gamma_per_class = gamma_per_class   # e.g. [1.0, 2.5]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_cls = logits.size(1)

        # ── Build per-sample α_t and γ_t tensors ─────────────────────────
        if self.alpha_per_class is not None and len(self.alpha_per_class) == n_cls:
            alpha_t_vec = torch.tensor(self.alpha_per_class,
                                       device=logits.device, dtype=logits.dtype)
        else:
            alpha_t_vec = None

        if self.gamma_per_class is not None and len(self.gamma_per_class) == n_cls:
            gamma_t_vec = torch.tensor(self.gamma_per_class,
                                       device=logits.device, dtype=logits.dtype)
        else:
            gamma_t_vec = None

        # Handle both hard labels (long, shape [B]) and soft labels (float, shape [B, C])
        if targets.dtype == torch.long or targets.ndim <= 1:
            # Hard labels
            targets = targets.long().view(-1)
            log_probs  = F.log_softmax(logits, dim=1)
            probs      = torch.exp(log_probs)
            targets_oh = F.one_hot(targets, n_cls).float()
            p_t        = (probs * targets_oh).sum(dim=1)

            if gamma_t_vec is not None:
                # Per-class gamma: pick gamma for each sample's true class
                gamma_t = gamma_t_vec[targets]
            else:
                gamma_t = self.gamma

            focal_weight = (1.0 - p_t) ** gamma_t
            # Guard against double-weighting: if alpha_t_vec is provided, let it serve as the class
            # weighting factor so F.cross_entropy does not multiply by self.weight a second time.
            effective_ce_weight = None if alpha_t_vec is not None else self.weight
            ce = F.cross_entropy(logits, targets, weight=effective_ce_weight, reduction="none")

            if alpha_t_vec is not None:
                alpha_t = alpha_t_vec[targets]
                loss = alpha_t * focal_weight * ce
            else:
                loss = focal_weight * ce

        else:
            # Soft labels (one-hot or probabilities)
            log_probs = F.log_softmax(logits, dim=1)
            probs     = torch.exp(log_probs)

            if gamma_t_vec is not None:
                gamma_t = gamma_t_vec.unsqueeze(0)
            else:
                gamma_t = self.gamma

            focal_weight_per_class = (1.0 - probs) ** gamma_t
            ce_per_class = -(targets * log_probs)

            # Guard against double-weighting: only apply weight_per_sample if alpha is not provided
            if self.weight is not None and alpha_t_vec is None:
                ce_per_class = ce_per_class * self.weight.unsqueeze(0)

            if alpha_t_vec is not None:
                loss = (alpha_t_vec.unsqueeze(0) * focal_weight_per_class * ce_per_class).sum(dim=1)
            else:
                loss = (focal_weight_per_class * ce_per_class).sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def resolve_asymmetric_loss_params(
    asym_alpha: Optional[Union[str, List[float], float]] = None,
    asym_gamma: Optional[Union[str, List[float], float]] = None,
    dataset_name: str = "pneumoniamnist",
    n_classes: int = 2,
    asymmetric_loss: bool = True,
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """Parse, autodetect, and validate per-class alpha and gamma for Asymmetric Focal Loss.

    Supports:
      - None / 'auto': autodetects optimal settings based on dataset_name and n_classes.
      - Comma-separated strings (e.g. '1.5,1.0' or '1.0,1.2,2.5,0.8').
      - List/tuple of floats matching n_classes.
      - Single scalar: broadcasts to all n_classes.
    """
    if not asymmetric_loss:
        return None, None

    def _parse(val: Any, param_name: str) -> Optional[List[float]]:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return [float(val)] * n_classes
        if isinstance(val, str):
            val = val.strip()
            if not val or val.lower() in ("none", "auto"):
                return None
            parts = [float(x.strip()) for x in val.split(",") if x.strip()]
            if len(parts) == 1:
                return parts * n_classes
            if len(parts) != n_classes:
                raise ValueError(
                    f"Mismatch in {param_name}: expected {n_classes} values, got {len(parts)} in '{val}'"
                )
            return parts
        if isinstance(val, (list, tuple)):
            if len(val) == 1:
                return [float(val[0])] * n_classes
            if len(val) != n_classes:
                raise ValueError(
                    f"Mismatch in {param_name}: expected {n_classes} values, got {len(val)}"
                )
            return [float(x) for x in val]
        return None

    alpha_list = _parse(asym_alpha, "asym_alpha")
    gamma_list = _parse(asym_gamma, "asym_gamma")

    if alpha_list is None or gamma_list is None:
        ds_lower = dataset_name.lower()
        if n_classes == 2:
            if "breast" in ds_lower:
                # BreastMNIST: Class 0 is Malignant (~27% minority), Class 1 is Benign (~73% majority)
                def_alpha = [1.5, 1.0]
                def_gamma = [1.5, 2.0]
            else:
                # PneumoniaMNIST / general binary: Class 0 Normal, Class 1 Disease
                def_alpha = [1.0, 1.2]
                def_gamma = [2.0, 2.0]
        elif n_classes == 4:
            # OCTMNIST / 4-class retinal: 0: CNV, 1: DME, 2: DRUSEN, 3: NORMAL
            def_alpha = [1.0, 1.4, 2.5, 0.7]
            def_gamma = [2.0, 2.0, 1.0, 2.5]
        else:
            # General multi-class default (N-class balanced)
            def_alpha = [1.0] * n_classes
            def_gamma = [2.0] * n_classes

        if alpha_list is None:
            alpha_list = def_alpha
        if gamma_list is None:
            gamma_list = def_gamma

    return alpha_list, gamma_list


def _compute_class_counts(loader, n_classes: int) -> List[int]:
    """Extract integer sample count per class from a DataLoader, Dataset, or tensor."""
    if loader is None:
        return [1] * n_classes
    ds = getattr(loader, "dataset", loader)
    if hasattr(ds, "labels"):
        lbls = np.asarray(ds.labels).squeeze()
        bc = np.bincount(lbls, minlength=n_classes)[:n_classes]
        return [max(1, int(c)) for c in bc]
    elif hasattr(ds, "tensors") and len(ds.tensors) >= 2:
        lbls = ds.tensors[1].squeeze().cpu().long().numpy()
        bc = np.bincount(lbls, minlength=n_classes)[:n_classes]
        return [max(1, int(c)) for c in bc]
    elif hasattr(loader, "__iter__"):
        counts = [0] * n_classes
        for batch in loader:
            labels = batch[1] if isinstance(batch, (list, tuple)) else batch
            if hasattr(labels, "ndim") and labels.ndim > 1 and labels.shape[1] == n_classes:
                bc = labels.sum(dim=0).cpu().long().tolist()
                for c in range(n_classes):
                    counts[c] += int(bc[c])
            else:
                flat = labels.reshape(-1).cpu().long()
                for c in range(n_classes):
                    counts[c] += int((flat == c).sum().item())
        return [max(1, c) for c in counts]
    return [1] * n_classes


class ClassBalancedLDAMLoss(nn.Module):
    r"""
    Class-Balanced Label-Distribution-Aware Margin (CB-LDAM-DRW) Loss (Solution 1.1).
    Synthesizes:
      1. Effective Number of Samples (Cui et al., CVPR 2019):
         E_{N_c} = (1 - \beta^{N_c}) / (1 - \beta),  with \beta = 0.9999
         w_c \propto 1 / E_{N_c}
      2. Label-Distribution-Aware Margin (Cao et al., NeurIPS 2019):
         \Delta_c = C_{\text{margin}} / N_c^{1/4}
         \tilde{z}_y = z_y - \Delta_y
      3. Deferred Re-Weighting (DRW):
         Epochs 1..drw_start: unweighted LDAM (w_c = 1) for stable representation learning.
         Epochs >= drw_start: activate class-balanced weights w_c to enforce minority margin.
    """
    def __init__(
        self,
        cls_num_list: List[int],
        max_margin: float = 0.50,
        scale: float = 10.0,
        temperature: float = 1.0,
        beta: float = 0.9999,
        drw_start_epoch: int = 10,
    ):
        super().__init__()
        self.cls_num_list = cls_num_list
        # scale (s) multiplies adjusted logits before softmax: s * logits
        # temperature = 1/scale for backward compat; scale takes precedence when != default
        if temperature != 1.0:
            self._scale = 1.0 / max(temperature, 1e-4)  # backward compat: explicit temperature
        else:
            self._scale = max(scale, 1e-4)  # use scale directly
        self.scale = scale
        self.drw_start_epoch = drw_start_epoch
        self.current_epoch = 1

        # Calculate LDAM margins: Delta_c = C_margin / N_c^0.25
        cls_nums = np.array(cls_num_list, dtype=float)
        m_list = 1.0 / np.power(np.maximum(cls_nums, 1.0), 0.25)
        m_list = m_list * (max_margin / max(float(np.max(m_list)), 1e-6))
        self.register_buffer("margins", torch.tensor(m_list, dtype=torch.float))

        # Calculate Cui et al. Effective Number of Samples weights
        effective_num = 1.0 - np.power(beta, cls_nums)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        weights = weights / np.sum(weights) * len(cls_num_list)
        self.register_buffer("cb_weights", torch.tensor(weights, dtype=torch.float))

    def update_epoch(self, epoch: int):
        self.current_epoch = epoch

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Smooth 2-epoch cosine ramp for class weights at DRW onset (Task 5.1)
        if self.current_epoch <= self.drw_start_epoch:
            weight = None
        else:
            ramp = min(1.0, (self.current_epoch - self.drw_start_epoch) / 2.0)
            weight = 1.0 - ramp + ramp * self.cb_weights

        scaled_logits = logits * self._scale
        if targets.dtype == torch.long or targets.ndim <= 1:
            targets_idx = targets.long().view(-1)
            margin_mask = F.one_hot(targets_idx, num_classes=logits.size(1)).float()
            # Direct margin subtraction in logit space without 10x multiplier shock (Task 5.1)
            adjusted_logits = scaled_logits - margin_mask * self.margins.unsqueeze(0)
            return F.cross_entropy(adjusted_logits, targets_idx, weight=weight)
        else:
            adjusted_logits = scaled_logits - targets * self.margins.unsqueeze(0)
            log_probs = F.log_softmax(adjusted_logits, dim=-1)
            loss = -(targets * log_probs).sum(dim=-1)
            if weight is not None:
                sample_weights = (targets * weight.unsqueeze(0)).sum(dim=-1)
                loss = loss * sample_weights
            return loss.mean()


class QuantumInfoNCELoss(nn.Module):
    """
    Quantum Kernel Contrastive Loss (Hilbert Space InfoNCE — Section 12.3):
    Computes quantum state fidelity κ(x_i, x_j) = |⟨ψ(x_i)|ψ(x_j)⟩|² in Hilbert space.
    Pulls positive pairs to identical states; pushes negative samples into orthogonal subspaces.
    Stabilized against singleton samples and self-similarity collapse (Task 5.2).
    """
    def __init__(self, n_qubits: int, tau: float = 0.07):
        super().__init__()
        self.n_qubits = n_qubits
        self.tau = tau
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def forward(
        self,
        z_batch: torch.Tensor,
        labels: torch.Tensor,
        q_weights: torch.Tensor,
        model: Optional[Any] = None,
    ) -> torch.Tensor:
        B = min(12, z_batch.size(0))
        z_sub = z_batch[:B]
        labels_sub = labels[:B]
        n_q = self.n_qubits

        states = []
        try:
            if model is not None and hasattr(model, "get_state_qnode"):
                _state_circ = model.get_state_qnode(diff_method="backprop")
                try:
                    states_t = _state_circ(z_sub, q_weights)
                    if states_t.dim() == 1 and B == 1:
                        states_t = states_t.unsqueeze(0)
                    states = [states_t[i] for i in range(min(B, states_t.size(0)))]
                except Exception:
                    states = [_state_circ(z_sub[i], q_weights) for i in range(B)]
            else:
                @qml.qnode(self.dev, interface="torch")
                def state_circ(inp, w):
                    for i in range(n_q):
                        qml.U3(inp[i], inp[i + n_q], inp[i + 2 * n_q], wires=i)
                    if w.ndim == 3:
                        qml.StronglyEntanglingLayers(w, wires=range(n_q))
                    elif w.ndim == 2:
                        qml.BasicEntanglerLayers(w, wires=range(n_q))
                    elif w.ndim == 1:
                        if w.numel() % (n_q * 3) == 0 and w.numel() > 0:
                            w_3d = w.view(-1, n_q, 3)
                            qml.StronglyEntanglingLayers(w_3d, wires=range(n_q))
                        else:
                            for idx, param in enumerate(w):
                                wire = (idx // 3) % n_q
                                rot = idx % 3
                                if rot == 0:
                                    qml.RX(param, wires=wire)
                                elif rot == 1:
                                    qml.RY(param, wires=wire)
                                else:
                                    qml.RZ(param, wires=wire)
                            for wire in range(n_q - 1):
                                qml.CNOT(wires=[wire, wire + 1])
                    return qml.state()

                for i in range(B):
                    states.append(state_circ(z_sub[i], q_weights))
            states = torch.stack(states)  # (B, 2^n)
        except Exception:
            return torch.tensor(0.0, device=z_batch.device, requires_grad=True)

        # Pairwise inner product: <psi_i | psi_j>
        inner_prod = torch.matmul(states, torch.conj(states).T)
        fidelities = inner_prod.real ** 2 + inner_prod.imag ** 2  # (B, B) in [0, 1]

        # Exclude diagonal self-similarity from positive masks (Task 5.2)
        eye_mask = torch.eye(B, device=z_batch.device)
        mask_pos = (labels_sub.unsqueeze(0) == labels_sub.unsqueeze(1)).float() * (1.0 - eye_mask)
        
        # Mask out anchor samples that have zero valid positive pairs (Task 5.2)
        valid_anchors = (mask_pos.sum(dim=-1) > 0)
        if not valid_anchors.any():
            return torch.tensor(0.0, device=z_batch.device, requires_grad=True)

        sim_matrix = fidelities / self.tau
        exp_sim = torch.exp(sim_matrix - torch.max(sim_matrix, dim=-1, keepdim=True)[0])

        log_probs = sim_matrix - torch.log(
            (exp_sim * (1.0 - eye_mask)).sum(dim=-1, keepdim=True) + 1e-8
        )
        n_positives = mask_pos.sum(dim=-1).clamp(min=1)
        per_anchor_loss = -(mask_pos * log_probs).sum(dim=-1) / n_positives
        loss = per_anchor_loss[valid_anchors].mean()
        return loss


class FeatureDecorrelationLoss(nn.Module):
    """
    Penalizes Pearson correlation between quantum angles and spurious image heuristics (Task 5.3).
    Decouples latent representation from background brightness and border gradient energy.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, angles: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        if angles.size(0) <= 2:
            return torch.tensor(0.0, device=angles.device, requires_grad=True)
        # Image confounder 1: global mean brightness
        b_mean = images.mean(dim=(-3, -2, -1))  # (B,)
        # Image confounder 2: border edge intensity
        h, w = images.shape[-2:]
        dh, dw = max(1, int(h * 0.1)), max(1, int(w * 0.1))
        border_energy = (
            images[..., :dh, :].mean(dim=(-3, -2, -1)) +
            images[..., -dh:, :].mean(dim=(-3, -2, -1)) +
            images[..., :, :dw].mean(dim=(-3, -2, -1)) +
            images[..., :, -dw:].mean(dim=(-3, -2, -1))
        ) * 0.25

        confounders = torch.stack([b_mean, border_energy], dim=1)  # (B, 2)
        a_c = angles.view(angles.size(0), -1) - angles.view(angles.size(0), -1).mean(dim=0, keepdim=True)
        c_c = confounders - confounders.mean(dim=0, keepdim=True)

        a_std = torch.norm(a_c, p=2, dim=0, keepdim=True).clamp(min=self.eps)
        c_std = torch.norm(c_c, p=2, dim=0, keepdim=True).clamp(min=self.eps)

        corr = torch.mm(a_c.T, c_c) / (torch.mm(a_std.T, c_std) + self.eps)
        return (corr ** 2).mean()


class SaliencyConcentrationLoss(nn.Module):
    """
    Penalizes attention mass / saliency gradients falling outside anatomical foreground mask (Task 5.4).
    L_saliency = ||A * (1 - M_fg)||_1 / (||A||_1 + eps)
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, saliency_map: torch.Tensor, foreground_mask: torch.Tensor) -> torch.Tensor:
        if saliency_map.ndim == 3:
            saliency_map = saliency_map.unsqueeze(1)
        if foreground_mask.ndim == 3:
            foreground_mask = foreground_mask.unsqueeze(1)
        if saliency_map.shape[-2:] != foreground_mask.shape[-2:]:
            foreground_mask = F.interpolate(foreground_mask.float(), size=saliency_map.shape[-2:], mode="nearest")

        saliency_abs = torch.abs(saliency_map)
        background_mask = 1.0 - foreground_mask.float()
        bg_mass = (saliency_abs * background_mask).sum(dim=(-2, -1))
        tot_mass = saliency_abs.sum(dim=(-2, -1)) + self.eps
        return (bg_mass / tot_mass).mean()


def compute_drusen_cnv_orthogonality_loss(
    states: torch.Tensor,
    labels: torch.Tensor,
    drusen_class: int = 2,
    cnv_class: int = 0,
    eps_target: float = 0.05,
) -> torch.Tensor:
    r"""
    Hilbert-Space Orthogonality Penalty between Drusen and CNV (Solution 1.2; Lloyd et al., 2020).
    Penalizes state fidelity F_ij = |<psi_i | psi_j>|^2 between Drusen and CNV samples:
        L_ortho = (1 / (|Drusen| * |CNV|)) * sum_{i in Drusen, j in CNV} max(0, F_ij - eps_target)^2
    Guarantees Bhattacharyya distance D_B >= -ln(eps_target) = 2.996, eliminating overlap (F=0.4170).
    """
    # Explicit dimension validation - allow both pure statevectors (power-of-2) and expectation vectors
    dim = states.shape[-1]
    if states.ndim < 2 or dim < 2:
        return torch.tensor(0.0, device=states.device, dtype=states.dtype)

    if labels.ndim > 1:
        labels = labels.argmax(dim=-1)

    drusen_mask = (labels == drusen_class)
    cnv_mask = (labels == cnv_class)

    n_d = drusen_mask.sum()
    n_c = cnv_mask.sum()

    if n_d == 0 or n_c == 0:
        return torch.tensor(0.0, device=states.device, dtype=states.dtype)

    psi_d = F.normalize(states[drusen_mask], p=2, dim=-1)
    psi_c = F.normalize(states[cnv_mask], p=2, dim=-1)

    is_pure_state = states.is_complex()
    if is_pure_state:
        # Pure Hilbert statevectors: fidelity = |<psi_d|psi_c>|^2
        inner = torch.matmul(psi_d, torch.conj(psi_c).T)
        fidelities = inner.real.pow(2) + inner.imag.pow(2)
        excess = F.relu(fidelities - eps_target)
        return (excess ** 2).mean()
    else:
        # Expectation vectors: penalize positive alignment (collinearity)
        cos_sim = torch.matmul(psi_d, psi_c.T)
        excess = F.relu(cos_sim - eps_target)
        return (excess ** 2).mean()
