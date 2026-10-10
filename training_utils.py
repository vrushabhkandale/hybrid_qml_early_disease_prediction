"""
training_utils.py — Optimizers, SWA, Checkpointing & Evaluation Utilities
========================================================================
Modularized optimization and training infrastructure:
  • _is_better_checkpoint: Multi-criteria model selection (AUROC + Loss tie-breaker)
  • QSAM: Quantum Sharpness-Aware Minimization optimizer wrapper
  • QSWAHandler: Quantum Stochastic Weight Averaging manager
  • HybridQMLOptimizer: Coordinated Classical (AdamW) + Quantum Natural Gradient optimizer
  • _build_hybrid_optimizer: Optimizer factory with momentum buffer preservation
  • compute_clinical_metrics: Sensitivity, Specificity, AUROC, AUPRC, Macro F1
  • McNemarResult & mcnemars_test: Statistical classifier comparison
  • mc_dropout_predict: Concurrent MC-Dropout inference for predictive uncertainty
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pennylane as qml
import scipy.stats as sp_stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def _is_better_checkpoint(
    new_auc: float,
    new_loss: float,
    best_auc: float,
    best_loss: float,
    tolerance: float = 1e-4,
) -> bool:
    """Compound clinical criterion: highest val_auc, with val_loss as tie-breaker."""
    if new_auc > best_auc + tolerance:
        return True
    elif abs(new_auc - best_auc) <= tolerance and new_loss < best_loss:
        return True
    return False


class QSAM:
    """
    Quantum Sharpness-Aware Minimization (Q-SAM — Section 12.4):
    Simultaneously minimizes loss value and loss sharpness by seeking parameters
    whose entire neighborhood has low loss.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, model: nn.Module, rho: float = 0.05):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.state: Dict[nn.Parameter, torch.Tensor] = {}

    def zero_grad(self):
        self.optimizer.zero_grad()

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = 0.0
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = math.sqrt(grad_norm) + 1e-12

        scale = self.rho / grad_norm
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                e_w = p.grad.data * scale
                self.state[p] = e_w.clone()
                p.data.add_(e_w)  # ascent step

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False, latent_batch: Optional[torch.Tensor] = None):
        for p in self.model.parameters():
            if p.requires_grad and p in self.state:
                p.data.sub_(self.state[p])
        if hasattr(self.optimizer, "step_with_latent"):
            self.optimizer.step_with_latent(latent_batch=latent_batch)
        else:
            self.optimizer.step()
        self.state.clear()
        if zero_grad:
            self.zero_grad()


class QSWAHandler:
    """
    Quantum Stochastic Weight Averaging (Q-SWA — Section 12.8):
    Averages parameters during cyclical learning rate schedule in final 30% of epochs
    to settle parameters into the geometric center of flat loss basins.
    """
    def __init__(self, model: nn.Module, start_epoch: int, total_epochs: int):
        self.model = model
        self.start_epoch = start_epoch
        self.total_epochs = total_epochs
        self.collected_weights: List[Dict[str, torch.Tensor]] = []
        self.n_models = 0

    def should_collect(self, epoch: int) -> bool:
        return epoch >= self.start_epoch

    @torch.no_grad()
    def collect(self):
        snapshot = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        self.collected_weights.append(snapshot)
        self.n_models += 1

    @torch.no_grad()
    def apply_swa_weights(self, train_loader: Optional[Any] = None):
        if not self.collected_weights:
            return
        print(f"\n[Q-SWA] Averaging parameters across {self.n_models} collected checkpoints...")
        avg_weights: Dict[str, torch.Tensor] = {}
        for k in self.collected_weights[0].keys():
            stack = torch.stack([snap[k] for snap in self.collected_weights], dim=0)
            if "qlayer" in k:
                # Circular Fréchet mean for periodic quantum angles:
                avg_weights[k] = torch.atan2(
                    torch.sin(stack).mean(dim=0),
                    torch.cos(stack).mean(dim=0),
                )
            else:
                avg_weights[k] = stack.mean(dim=0)

        for n, p in self.model.named_parameters():
            if p.requires_grad and n in avg_weights:
                p.data.copy_(avg_weights[n])
        print("[Q-SWA] SWA weights successfully loaded into model.")

        # Recalibrate BatchNorm statistics if loader provided
        if train_loader is not None:
            self.model.train()
            dev = next(self.model.parameters()).device
            for batch in train_loader:
                imgs = batch[0] if isinstance(batch, (list, tuple)) else batch.get("image", batch.get("images"))
                if imgs is not None:
                    is_cached = (imgs.ndim == 2 and imgs.size(1) == 512)
                    self.model(imgs.to(dev), use_cached_features=is_cached)
            self.model.eval()

    def update(self, epoch: int):
        """Conditionally collect model weights if epoch >= start_epoch."""
        if self.should_collect(epoch):
            self.collect()

    def apply_swa(self):
        """Convenience alias for apply_swa_weights()."""
        self.apply_swa_weights()


class HybridQMLOptimizer:
    """
    Coordinated Hybrid Quantum-Classical Optimizer (Section 4.3):
    - Classical parameters updated via AdamW
    - Quantum parameters updated via Quantum Natural Gradient (QNG):
      θ_{t+1} = θ_t - η_q · (g(θ) + damping · I)⁺ · ∇_θ L
    """
    def __init__(
        self,
        classical_params,
        model: nn.Module,
        lr_classical: float = 1e-3,
        lr_quantum: float = 1e-3,
        damping: float = 1e-4,
    ):
        self.opt_classical = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, classical_params),
            lr=lr_classical, weight_decay=1e-4
        )
        self.model = model
        self.lr_quantum = lr_quantum
        self.damping = damping

        self.q_param = None
        for n, p in model.named_parameters():
            if "qlayer" in n and "weights" in n:
                self.q_param = p
                break
        self._metric_fn = None

    def _get_metric_fn(self):
        """Build metric tensor function compatible with both QNAS 1D DAGs
        and StronglyEntanglingLayers (Fix E-06)."""
        if self._metric_fn is None:
            dev = qml.device("default.qubit", wires=self.model.n_qubits)
            n_q = self.model.n_qubits
            n_l = self.model.n_layers
            reupload = getattr(self.model, "reupload", True)
            genome = getattr(self.model, "genome", None)

            @qml.qnode(dev)
            def circ(inp, w):
                # Fix E-06: Dynamic circuit construction from QNAS genome
                if genome is not None:
                    for i in range(n_q):
                        qml.Hadamard(wires=i)
                    w_idx = 0
                    for l, layer in enumerate(genome.layers):
                        if reupload or l == 0:
                            for i in range(n_q):
                                qml.U3(inp[i], inp[i + n_q], inp[i + 2 * n_q], wires=i)
                        for gene in layer:
                            if gene.gate_type == "Rx":
                                qml.RX(w[w_idx], wires=gene.target); w_idx += 1
                            elif gene.gate_type == "Ry":
                                qml.RY(w[w_idx], wires=gene.target); w_idx += 1
                            elif gene.gate_type == "Rz":
                                qml.RZ(w[w_idx], wires=gene.target); w_idx += 1
                            elif gene.gate_type == "U3":
                                qml.U3(w[w_idx], w[w_idx+1], w[w_idx+2], wires=gene.target); w_idx += 3
                            elif gene.gate_type == "CNOT":
                                qml.CNOT(wires=[gene.control, gene.target])
                            elif gene.gate_type == "CZ":
                                qml.CZ(wires=[gene.control, gene.target])
                else:
                    # Default StronglyEntanglingLayers path
                    if reupload:
                        for l in range(n_l):
                            for i in range(n_q):
                                qml.U3(inp[i], inp[i + n_q], inp[i + 2 * n_q], wires=i)
                            qml.StronglyEntanglingLayers(w[l:l+1], wires=range(n_q))
                    else:
                        for i in range(n_q):
                            qml.U3(inp[i], inp[i + n_q], inp[i + 2 * n_q], wires=i)
                        qml.StronglyEntanglingLayers(w, wires=range(n_q))
                return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]

            self._metric_fn = qml.metric_tensor(circ, approx="diag")
        return self._metric_fn

    def zero_grad(self):
        self.opt_classical.zero_grad()
        if self.q_param is not None and self.q_param.grad is not None:
            self.q_param.grad.zero_()

    def step(self):
        self.opt_classical.step()

    def step_with_latent(self, latent_batch: Optional[torch.Tensor] = None):
        self.opt_classical.step()

        if self.q_param is not None and self.q_param.grad is not None:
            from pennylane import numpy as pnp
            n_q = self.model.n_qubits
            w_shape = self.q_param.shape
            P = int(np.prod(w_shape))
            w_np = pnp.array(self.q_param.detach().cpu().numpy(), requires_grad=True)

            if latent_batch is not None:
                K = min(4, latent_batch.shape[0])
                g_diags = []
                for k in range(K):
                    z_k = pnp.array(latent_batch[k].detach().cpu().numpy(), requires_grad=False)
                    try:
                        metric_fn = self._get_metric_fn()
                        g_raw_k = np.asarray(metric_fn(z_k, w_np))
                        g_diag_k = g_raw_k.flatten() if g_raw_k.size == P else np.diag(g_raw_k.reshape(P, P))
                        g_diags.append(g_diag_k)
                    except Exception as e:
                        logger.warning("QNG metric_tensor failed for sample %d: %s", k, e)
                g_diag = np.mean(g_diags, axis=0) if len(g_diags) > 0 else None
            else:
                z_sample = pnp.zeros(3 * n_q, requires_grad=False)
                try:
                    metric_fn = self._get_metric_fn()
                    g_raw = np.asarray(metric_fn(z_sample, w_np))
                    g_diag = g_raw.flatten() if g_raw.size == P else np.diag(g_raw.reshape(P, P))
                except Exception as e:
                    logger.warning("QNG metric_tensor failed for zero vector: %s", e)
                    g_diag = None

            if g_diag is not None:
                g_inv = 1.0 / (g_diag + self.damping)
                euc_grad = self.q_param.grad.detach().cpu().numpy().flatten()
                qng_step = g_inv * euc_grad
                qng_update = torch.from_numpy(qng_step.reshape(w_shape)).to(self.q_param.device).float()
                with torch.no_grad():
                    self.q_param.sub_(self.lr_quantum * qng_update)
            else:
                logger.error("QNG: All metric tensor evaluations failed. Falling back to SGD.")
                with torch.no_grad():
                    self.q_param.sub_(self.lr_quantum * self.q_param.grad)


def _build_hybrid_optimizer(
    model: nn.Module,
    use_qng: bool = False,
    lr_classical: float = 2e-4,
    lr_quantum: float = 5e-3,
    lr_head: float = 1e-3,
    lr_backbone: float = 0.0,
    old_optimizer: Optional[torch.optim.Optimizer] = None,
):
    """Build or rebuild optimizer for HybridCNNVQC with classical state preservation.

    Preserves Adam momentum and variance states for classical adapter and head parameters
    across progressive layer growth transitions, while providing fresh optimizer states
    for newly appended quantum parameters (Skolik et al., 2021).

    Parameters
    ----------
    lr_backbone : float
        Learning rate for finetuned backbone (vision encoder) parameters.
        Default 0.0 means backbone params are excluded from the optimizer.
    """
    trainable_classical = list(filter(lambda p: p.requires_grad, model.backbone.cq_adapter.parameters()))
    if hasattr(model.backbone, "skip"):
        trainable_classical += list(filter(lambda p: p.requires_grad, model.backbone.skip.parameters()))

    # Collect backbone (vision encoder) trainable params if lr_backbone > 0
    backbone_vit_params = []
    if lr_backbone > 0 and hasattr(model.backbone, "vision_encoder") and model.backbone.vision_encoder is not None:
        backbone_vit_params = list(filter(lambda p: p.requires_grad, model.backbone.vision_encoder.parameters()))

    if use_qng:
        new_opt = HybridQMLOptimizer(
            trainable_classical + list(model.head.parameters()),
            model=model,
            lr_classical=lr_classical,
            lr_quantum=lr_quantum,
        )
    else:
        param_groups = [
            {"params": trainable_classical, "lr": lr_classical, "weight_decay": 1e-4},
            {"params": filter(lambda p: p.requires_grad, model.qlayer.parameters()), "lr": lr_quantum, "weight_decay": 0.0},
            {"params": filter(lambda p: p.requires_grad, model.head.parameters()), "lr": lr_head, "weight_decay": 1e-4},
        ]
        if backbone_vit_params:
            param_groups.append({"params": backbone_vit_params, "lr": lr_backbone, "weight_decay": 1e-5})
        new_opt = torch.optim.AdamW(param_groups)

    # Migrate Adam momentum/variance buffers for classical parameters if available
    if old_optimizer is not None and not use_qng and hasattr(old_optimizer, "state"):
        classical_params = trainable_classical + list(model.head.parameters())
        for p in classical_params:
            if p in old_optimizer.state:
                new_opt.state[p] = old_optimizer.state[p]

    return new_opt



def compute_clinical_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    n_classes: int,
) -> Dict[str, Union[float, int]]:
    """Compute sensitivity, specificity, F1, AUROC, AUPRC for binary and multi-class."""
    from sklearn.metrics import confusion_matrix, recall_score

    metrics: Dict[str, Union[float, int]] = {}
    metrics["accuracy"] = float(accuracy_score(labels, preds))
    metrics["f1_macro"] = float(f1_score(labels, preds, average="macro", zero_division=0))
    metrics["macro_f1"] = metrics["f1_macro"]  # canonical alias
    metrics["f1_weighted"] = float(f1_score(labels, preds, average="weighted", zero_division=0))

    if n_classes == 2:
        # Binary metrics
        tn = float(np.sum((preds == 0) & (labels == 0)))
        fp = float(np.sum((preds == 1) & (labels == 0)))
        fn = float(np.sum((preds == 0) & (labels == 1)))
        tp = float(np.sum((preds == 1) & (labels == 1)))
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
        metrics["sensitivity"] = sens
        metrics["specificity"] = spec
        # Aliases expected by kfold_summary.json acceptance assertions
        metrics["sensitivity_macro"] = sens
        metrics["specificity_macro"] = spec
        p1 = probs[:, 1] if (probs.ndim == 2 and probs.shape[1] > 1) else probs.squeeze()
        try:
            metrics["auroc"] = float(roc_auc_score(labels, p1))
        except ValueError:
            metrics["auroc"] = float("nan") if len(np.unique(labels)) < 2 else 0.0
        try:
            metrics["auprc"] = float(average_precision_score(labels, p1))
        except ValueError:
            metrics["auprc"] = float("nan") if len(np.unique(labels)) < 2 else 0.0
        # Aliases
        metrics["auroc_macro"] = metrics["auroc"]
        metrics["pr_auc_macro"] = metrics["auprc"]
    else:
        # Multi-class
        try:
            metrics["auroc"] = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
        except ValueError:
            metrics["auroc"] = 0.0
        metrics["auroc_macro"] = metrics["auroc"]

        try:
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(labels, classes=range(n_classes))
            metrics["auprc"] = float(average_precision_score(y_bin, probs, average="macro"))
        except (ValueError, ImportError):
            metrics["auprc"] = 0.0
        metrics["pr_auc_macro"] = metrics["auprc"]

        cm = confusion_matrix(labels, preds, labels=range(n_classes))

        # Per-class sensitivity (recall) averaged as macro
        per_class_sens = []
        per_class_specs = []
        for c in range(n_classes):
            tp_c = float(cm[c, c])
            fn_c = float(cm[c, :].sum() - cm[c, c])
            tn_c = float(cm.sum() - (cm[c, :].sum() + cm[:, c].sum() - cm[c, c]))
            fp_c = float(cm[:, c].sum() - cm[c, c])
            per_class_sens.append(tp_c / max(tp_c + fn_c, 1.0))
            per_class_specs.append(tn_c / max(tn_c + fp_c, 1.0))

        sens_macro = float(np.mean(per_class_sens))
        spec_macro = float(np.mean(per_class_specs))
        metrics["sensitivity"] = sens_macro
        metrics["specificity"] = spec_macro
        # Aliases expected by kfold_summary.json acceptance assertions
        metrics["sensitivity_macro"] = sens_macro
        metrics["specificity_macro"] = spec_macro

    return metrics


class McNemarResult(dict):
    """Container for McNemar's test results that supports:
    1. Dictionary semantics: res["p_value"], res["significant"], res.get(...)
    2. Tuple unpacking: p_val, b_stat, c_stat = res
    3. JSON serialization via standard dict inheritance
    """
    def __iter__(self):
        # Enables tuple unpacking: p_val, b_stat, c_stat = mcnemars_test(...)
        yield self["p_value"]
        yield self["b_only_correct"]
        yield self["c_only_correct"]


def mcnemars_test(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    labels: np.ndarray,
) -> McNemarResult:
    """McNemar's test for statistical significance between two classifiers."""
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)
    # b = A correct & B wrong,  c = A wrong & B correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    # McNemar statistic with continuity correction
    if b + c == 0:
        return McNemarResult({
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "b_only_correct": 0,
            "c_only_correct": 0,
        })
    diff = abs(b - c)
    chi2 = float(max(0.0, diff - 1.0) ** 2 / (b + c))
    p_value = float(1.0 - sp_stats.chi2.cdf(chi2, df=1))
    return McNemarResult({
        "statistic": chi2,
        "p_value": p_value,
        "significant": bool(p_value < 0.05),
        "b_only_correct": int(b),
        "c_only_correct": int(c),
    })


def mc_dropout_predict(
    model: nn.Module,
    loader,
    device: torch.device,
    n_samples: int = 20,
    noise_std: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantum MC Dropout: evaluate full loader with noise T times concurrently."""
    all_probs = []
    labels_list = []
    orig_params = {n: p.data.clone() for n, p in model.named_parameters() if "qlayer" in n}

    for _ in range(n_samples):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in orig_params:
                    p.data = orig_params[n] + torch.randn_like(p) * noise_std

        probs_list = []
        labels_list = []
        with torch.no_grad():
            for images, labels in loader:
                is_cached = (images.ndim == 2 and images.size(1) == 512)
                logits = model(images.to(device), use_cached_features=is_cached)
                probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
                labels_list.append(labels.detach().cpu().squeeze().numpy())

        all_probs.append(np.concatenate(probs_list, axis=0))

    # Restore original params
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in orig_params:
                p.data = orig_params[n]

    all_probs_arr = np.stack(all_probs, axis=0)
    labels_arr = np.concatenate(labels_list, axis=0)

    mean_probs = all_probs_arr.mean(axis=0)
    std_probs = all_probs_arr.std(axis=0)
    return mean_probs, std_probs, labels_arr
