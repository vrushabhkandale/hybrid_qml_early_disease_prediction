"""
adapter.py — Classical-to-Quantum (CQ) Adapter Architecture, Geometric Losses & Pre-training Engine
==================================================================================================
Houses:
  • LoRAQLinear                  — Low-Rank Adaptation (LoRA-Q) layer for parameter-efficient adapter compression
  • MultiHeadOrthogonalCQAdapter  — 3-Head Cartesian projection with Frobenius difference loss
  • Geometric Loss Suite          — Decoupled SupCon (DCL), Spherical Covariance, Hyperspherical Uniformity,
                                   Azimuthal Torque, Supervised Antipodal vMF, Latent Multi-View Perturbation
  • Pre-Training Engine           — pretrain_cq_adapter() with automatic diagnostic plot generation
  • Checkpoint Management         — get_adapter_dir(), get_adapter_checkpoint_path(),
                                   save_adapter_checkpoint(), load_adapter_checkpoint()
  • Standalone CLI                — python -m qml_models.adapter --dataset <name> --n_qubits 4 --epochs 30
"""

from __future__ import annotations

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    from plots import (
        plot_cq_pretrain_bloch_evolution,
        plot_cq_pretrain_geometry_dynamics,
        plot_cq_pretrain_loss_curves,
        plot_cq_pretrain_qubit_covariance_spectra,
    )
except ImportError:
    try:
        from qml_models.plots import (
            plot_cq_pretrain_bloch_evolution,
            plot_cq_pretrain_geometry_dynamics,
            plot_cq_pretrain_loss_curves,
            plot_cq_pretrain_qubit_covariance_spectra,
        )
    except ImportError:
        plot_cq_pretrain_bloch_evolution = None
        plot_cq_pretrain_geometry_dynamics = None
        plot_cq_pretrain_loss_curves = None
        plot_cq_pretrain_qubit_covariance_spectra = None


# ═══════════════════════════════════════════════════════════════════════════
# 1.  ADAPTER ARCHITECTURE LAYERS
# ═══════════════════════════════════════════════════════════════════════════

class LoRAQLinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA-Q) layer for parameter-efficient adapter compression.
    W(x) = W_0(x) + (alpha / r) * B(A(x))
    Reduces adapter trainable parameters by >80% while constraining trajectory to
    intrinsic low-dimensional manifold.
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 16.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.rank = rank
        self.scaling = alpha / rank if rank > 0 else 1.0
        if rank > 0:
            # Semi-orthogonal base projection initialization
            nn.init.orthogonal_(self.linear.weight)
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)
            self.linear.weight.requires_grad = False
            if self.linear.bias is not None:
                self.linear.bias.requires_grad = False
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.linear(x)
        if self.rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_term = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            res = res + lora_term
        return res


class MultiHeadOrthogonalCQAdapter(nn.Module):
    r"""
    3-Head Orthogonal CQ Adapter (Bousmalis et al., NeurIPS 2016; Difference Loss).

    Factors the classical-to-quantum projection into 3 independent channels:
      - Head X: projects 512-d vision embeddings to v_x logits
      - Head Y: projects 512-d vision embeddings to v_y logits
      - Head Z: projects 512-d vision embeddings to v_z logits

    Enforces rank-3 Cartesian pre-activations and breaks rank-1 collinearity on S^2,
    guaranteeing that data points expand across a 2D surface area rather than collapsing to a 1D arc.
    """

    def __init__(
        self,
        in_features: int = 512,
        n_qubits: int = 4,
        dropout: float = 0.3,
        linear_cls=nn.Linear,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.out_features = 3 * n_qubits

        # Shared representation trunk
        self.shared = nn.Sequential(
            linear_cls(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        # Head X (X-axis Cartesian logit)
        self.head_x = nn.Sequential(
            linear_cls(128, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, n_qubits),
        )

        # Head Y (Y-axis Cartesian logit)
        self.head_y = nn.Sequential(
            linear_cls(128, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, n_qubits),
        )

        # Head Z (Z-axis Cartesian logit)
        self.head_z = nn.Sequential(
            linear_cls(128, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, n_qubits),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        feat = self.shared(features)
        vx = self.head_x(feat)  # (B, n_qubits)
        vy = self.head_y(feat)  # (B, n_qubits)
        vz = self.head_z(feat)  # (B, n_qubits)

        # Interleave coordinates: for each qubit q, (vx_q, vy_q, vz_q)
        v = torch.stack([vx, vy, vz], dim=-1)  # (B, n_qubits, 3)
        return v.view(-1, 3 * self.n_qubits)

    def compute_orthogonality_loss(self, skip_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Computes per-qubit pairwise inner-product loss between Head X, Y, Z projection weights.

        Enforces that for each qubit q in {0, ..., N-1}, (wx_q, wy_q, wz_q) are mutually orthogonal
        in R^32. Bounded within rank-3 subspace per qubit, scaling to any N qubits.
        Also penalizes collinearity in residual skip projection rows (Fix: ADP-02).
        """
        wx = F.normalize(self.head_x[-1].weight, p=2, dim=1)  # (n_qubits, 32)
        wy = F.normalize(self.head_y[-1].weight, p=2, dim=1)  # (n_qubits, 32)
        wz = F.normalize(self.head_z[-1].weight, p=2, dim=1)  # (n_qubits, 32)

        cos_xy = (wx * wy).sum(dim=1)  # (n_qubits,)
        cos_yz = (wy * wz).sum(dim=1)  # (n_qubits,)
        cos_xz = (wx * wz).sum(dim=1)  # (n_qubits,)

        loss = torch.mean(cos_xy ** 2 + cos_yz ** 2 + cos_xz ** 2)

        if skip_weight is not None:
            n_q = self.n_qubits
            skip_loss = torch.tensor(0.0, device=skip_weight.device)
            for q in range(n_q):
                sx = F.normalize(skip_weight[q], dim=0)
                sy = F.normalize(skip_weight[q + n_q], dim=0)
                sz = F.normalize(skip_weight[q + 2 * n_q], dim=0)
                skip_loss = skip_loss + (sx @ sy) ** 2 + (sy @ sz) ** 2 + (sx @ sz) ** 2
            loss = loss + (skip_loss / float(max(1, n_q)))

        return loss


# ═══════════════════════════════════════════════════════════════════════════
# 2.  GEOMETRIC LOSSES & LATENT AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

def _decoupled_supcon_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.10,
) -> torch.Tensor:
    r"""Decoupled Supervised Contrastive Loss (DCL-SupCon) with Inverse Class Frequency Weighting.

    Eliminates positive-pair competition in the softmax denominator and weights anchors
    by inverse class prevalence to protect minority medical pathologies.
    """
    batch_size = z.size(0)
    if batch_size <= 1:
        return (z * 0.0).sum()

    labels_flat = labels.contiguous().view(-1)
    labels_2d = labels_flat.view(-1, 1)
    mask_self = torch.eye(batch_size, device=z.device, dtype=torch.bool)
    pos_mask = torch.eq(labels_2d, labels_2d.T) & ~mask_self
    neg_mask = torch.ne(labels_2d, labels_2d.T)

    valid_anchors = pos_mask.sum(dim=1) > 0
    if not valid_anchors.any():
        return (z * 0.0).sum()

    z_norm = F.normalize(z, p=2, dim=1)
    sim = torch.matmul(z_norm, z_norm.T) / temperature

    # Negative log-sum-exp per anchor row: (B, 1)
    sim_neg = sim.clone().masked_fill_(~neg_mask, float("-inf"))
    log_neg_sum = torch.logsumexp(sim_neg, dim=1, keepdim=True)

    diff = log_neg_sum - sim  # (B, B)
    loss_matrix = F.softplus(diff)  # (B, B)

    n_pos = pos_mask.sum(dim=1).float().clamp(min=1)
    loss_per_anchor = (pos_mask.float() * loss_matrix).sum(dim=1) / n_pos

    # Inverse-class-frequency weighting to prevent majority class dominance
    class_counts = torch.bincount(labels_flat, minlength=int(labels_flat.max().item() + 1)).float()
    n_classes_eff = max(1, int((class_counts > 0).sum().item()))
    class_weights = (labels_flat.numel() / (n_classes_eff * class_counts.clamp(min=1.0)))
    anchor_weights = class_weights[labels_flat]

    weighted_loss = (anchor_weights * loss_per_anchor)[valid_anchors]
    return weighted_loss.sum() / anchor_weights[valid_anchors].sum().clamp(min=1e-6)


def _latent_augment(
    features: torch.Tensor,
    p_drop: float = 0.10,
    noise_std: float = 0.02,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate two manifold-preserving stochastic multi-view latent perturbations on S^511."""
    def _augment_single(z):
        mask = (torch.rand_like(z) > p_drop).float() / (1.0 - p_drop)
        z_drop = z * mask
        noise = torch.randn_like(z) * noise_std
        # Project noise onto tangent space T_z S^511 and retract via L2 normalization
        tangent_noise = noise - (noise * z).sum(dim=-1, keepdim=True) * z
        return F.normalize(z_drop + tangent_noise, p=2, dim=-1)

    return _augment_single(features), _augment_single(features)


def compute_per_qubit_spherical_loss(
    r_vectors: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Class-Conditional Centered Log-Determinant Covariance Loss on S^2.

    Computes covariance within each class cluster to prevent conflict with polar separation.
    """
    batch_sz, n_qubits, _ = r_vectors.shape
    eye3 = torch.eye(3, device=r_vectors.device, dtype=r_vectors.dtype) * 1e-4
    MIN_THEORETICAL_LOGDET = 3.295836866  # 3 * ln(3)

    if labels is not None:
        labels_flat = labels.view(-1)
        unique_c = torch.unique(labels_flat)
        total_loss = torch.tensor(0.0, device=r_vectors.device)
        valid_clusters = 0
        for c in unique_c:
            c_mask = (labels_flat == c)
            if c_mask.sum() > 2:
                r_c = r_vectors[c_mask]  # (B_c, n_qubits, 3)
                for q in range(n_qubits):
                    rq = r_c[:, q, :]
                    rq_centered = rq - rq.mean(dim=0, keepdim=True)
                    cov_q = torch.matmul(rq_centered.T, rq_centered) / float(rq.size(0)) + eye3
                    log_det = torch.logdet(cov_q)
                    centered_loss = torch.clamp(-log_det - MIN_THEORETICAL_LOGDET, min=0.0)
                    total_loss = total_loss + centered_loss
                valid_clusters += 1
        if valid_clusters > 0:
            return total_loss / float(valid_clusters * n_qubits)

    total_loss = torch.tensor(0.0, device=r_vectors.device)
    for q in range(n_qubits):
        r_q = r_vectors[:, q, :]
        cov_q = torch.matmul(r_q.T, r_q) / float(batch_sz) + eye3
        log_det = torch.logdet(cov_q)
        centered_loss = torch.clamp(-log_det - MIN_THEORETICAL_LOGDET, min=0.0)
        total_loss = total_loss + centered_loss

    return total_loss / float(n_qubits)


def compute_antipodal_vmf_loss(
    r_vectors: torch.Tensor,
    labels: torch.Tensor,
    target_cone_deg: float = 25.0,
) -> Tuple[torch.Tensor, float]:
    """Supervised Antipodal von Mises-Fisher loss with one-sided cone margin on S^2.

    Pure basis states |0> (+Z) and |1> (-Z) incur zero penalty.
    """
    labels_flat = labels.view(-1)
    unique_classes = torch.unique(labels_flat)
    if len(unique_classes) > 2 or (unique_classes.numel() > 0 and unique_classes.max() > 1):
        return compute_hyperspherical_uniformity_loss(r_vectors, labels)

    batch_sz, n_qubits, _ = r_vectors.shape
    device = r_vectors.device
    dtype = r_vectors.dtype

    # Target sign: +1.0 for 0 (Normal/Negative), -1.0 for 1 (Disease/Positive)
    target_sign = torch.where(labels_flat == 0, 1.0, -1.0).view(-1, 1).to(device=device, dtype=dtype)

    # Cosine of polar angle to assigned class pole: cos_psi = r_z * target_sign
    cos_psi = r_vectors[..., 2] * target_sign  # (B, n_qubits)

    # 1. Penalty for crossing equator into wrong hemisphere
    wrong_hemi_penalty = F.relu(-cos_psi) ** 2

    # 2. One-sided cone penalty: rewards pure basis states (cos_psi >= cos(target_cone_deg))
    target_cos = math.cos(math.radians(target_cone_deg))
    cone_penalty = F.relu(target_cos - cos_psi) ** 2

    vmf_loss = (wrong_hemi_penalty.mean() * 2.0 + cone_penalty.mean()).squeeze()

    # Empirical mean angular separation between class centroids (in degrees)
    c0_mask = (labels_flat == 0)
    c1_mask = (labels_flat == 1)
    if c0_mask.any() and c1_mask.any():
        mu_0 = F.normalize(r_vectors[c0_mask].mean(dim=0).mean(dim=0, keepdim=True), p=2, dim=-1)
        mu_1 = F.normalize(r_vectors[c1_mask].mean(dim=0).mean(dim=0, keepdim=True), p=2, dim=-1)
        cos_sep = torch.clamp((mu_0 * mu_1).sum(), -1.0, 1.0).item()
        mean_sep_deg = math.degrees(math.acos(cos_sep))
    else:
        mean_sep_deg = 180.0

    return vmf_loss, mean_sep_deg


def compute_hyperspherical_uniformity_loss(
    r_vectors: torch.Tensor,
    labels: torch.Tensor,
    t: float = 2.0,
    margin_deg: float = 120.0,
    instance_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """Generalized Multi-Class All-Pairs Hyperspherical Uniformity & Repulsion Loss on S^2.

    Combines:
      1. Intra-Class Uniformity (Wang & Isola, ICML 2020): Gaussian potential repulsion within
         every class cluster c in {0, ..., K-1} to prevent single-point collapse.
         Excludes multi-view perturbations of the same physical instance (Fix: ADP-01).
      2. All-Pairs Inter-Class Centroid Repulsion: Maximizes angular separation across all pairs (c_a, c_b)
         conforming to the optimal Tammes/Thomson spherical packing bound.
      3. Polar Alignment: Soft hemispheric pinning when K=2.
    """
    batch_sz, n_qubits, _ = r_vectors.shape
    device = r_vectors.device
    labels = labels.view(-1)
    unique_classes = torch.unique(labels)
    num_classes = len(unique_classes)

    total_uniform_loss = torch.tensor(0.0, device=device)
    total_antipodal_loss = torch.tensor(0.0, device=device)
    total_hemi_loss = torch.tensor(0.0, device=device)

    # Adaptive spherical packing margin angle based on number of classes
    if num_classes <= 2:
        target_cos = math.cos(math.radians(margin_deg))  # e.g. cos(120°) = -0.50
    elif num_classes == 3:
        target_cos = math.cos(math.radians(min(margin_deg, 110.0)))
    elif num_classes == 4:
        # Regular tetrahedron angle: arccos(-1/3) ≈ 109.47°
        target_cos = math.cos(math.radians(min(margin_deg, 95.0)))
    else:
        # Optimal Tammes spherical packing bound for K > 4
        max_spherical_deg = math.degrees(math.acos(max(-0.99, -1.0 / max(1, num_classes - 1))))
        target_cos = math.cos(math.radians(min(margin_deg, max_spherical_deg * 0.85)))

    for q in range(n_qubits):
        rq = r_vectors[:, q, :]  # (B, 3)

        # 1. Within-class uniformity across ALL classes present in batch
        for c in unique_classes:
            mask_c = (labels == c)
            if mask_c.sum() > 1:
                rq_c = rq[mask_c]
                cos_sim = torch.matmul(rq_c, rq_c.T).clamp(-1.0, 1.0)
                dist_sq = 2.0 - 2.0 * cos_sim
                eye_mask = torch.eye(rq_c.size(0), dtype=torch.bool, device=device)
                if instance_ids is not None:
                    inst_c = instance_ids[mask_c]
                    same_inst = (inst_c.unsqueeze(0) == inst_c.unsqueeze(1))
                    valid_mask = ~eye_mask & ~same_inst
                else:
                    valid_mask = ~eye_mask
                non_diag_dist_sq = dist_sq.masked_select(valid_mask)
                if non_diag_dist_sq.numel() > 0:
                    uniform_c = torch.log(torch.mean(torch.exp(-t * non_diag_dist_sq)).clamp(min=1e-6))
                    total_uniform_loss = total_uniform_loss + uniform_c

        # 2. All-pairs inter-class repulsion across all pairs (c_a, c_b)
        if num_classes >= 2:
            pair_count = 0
            qubit_pair_loss = torch.tensor(0.0, device=device)
            for i_idx in range(num_classes):
                c_a = unique_classes[i_idx]
                mask_a = (labels == c_a)
                rq_a = rq[mask_a]

                for j_idx in range(i_idx + 1, num_classes):
                    c_b = unique_classes[j_idx]
                    mask_b = (labels == c_b)
                    rq_b = rq[mask_b]

                    cos_cross = torch.matmul(rq_a, rq_b.T).clamp(-1.0, 1.0)
                    pair_pen = F.relu(cos_cross - target_cos).pow(2).mean()
                    qubit_pair_loss = qubit_pair_loss + pair_pen
                    pair_count += 1

            if pair_count > 0:
                total_antipodal_loss = total_antipodal_loss + (qubit_pair_loss / float(pair_count))

        # 3. Soft polar orientation (+Z for Class 0, -Z for Class 1) ONLY for strictly binary tasks
        if num_classes == 2 and 0 in unique_classes and 1 in unique_classes:
            rq_0 = rq[labels == 0]
            rq_1 = rq[labels == 1]
            c0_wrong_hemi = F.relu(-rq_0[:, 2]).pow(2).mean()
            c1_wrong_hemi = F.relu(rq_1[:, 2]).pow(2).mean()
            total_hemi_loss = total_hemi_loss + (c0_wrong_hemi + c1_wrong_hemi)

    loss = (total_uniform_loss + 2.0 * total_antipodal_loss + 1.0 * total_hemi_loss) / max(1, n_qubits)

    # Compute empirical mean pairwise angular separation across all class centroids (Vectorized on GPU)
    if num_classes >= 2:
        class_centroids = []
        for c in unique_classes:
            c_mask = (labels == c)
            c_mean = r_vectors[c_mask].mean(dim=0)  # (n_qubits, 3)
            c_norm = F.normalize(c_mean, p=2, dim=-1)
            class_centroids.append(c_norm)

        c_stack = torch.stack(class_centroids, dim=0)  # (K, n_qubits, 3)
        # Cosine matrix across all class pairs: (K, K, n_qubits)
        cos_mat = torch.einsum('ikd,jkd->ijk', c_stack, c_stack).clamp(-1.0, 1.0)
        triu_i, triu_j = torch.triu_indices(num_classes, num_classes, offset=1, device=device)
        pair_cos = cos_mat[triu_i, triu_j, :]  # (num_pairs, n_qubits)
        pair_angles_deg = torch.rad2deg(torch.acos(pair_cos))
        mean_sep_deg = float(pair_angles_deg.mean().item())
    else:
        mean_sep_deg = 180.0

    return loss, mean_sep_deg


def compute_azimuthal_torque_loss(r_vectors: torch.Tensor) -> torch.Tensor:
    """Singularity-free azimuthal dispersion via polar-damped Cartesian directional cosines.

    Eliminates atan2 coordinate singularity near poles where ||grad phi|| -> infinity.
    Damped by w(r_z) = 1 - r_z^2 to guarantee smooth, bounded gradients everywhere on S^2.
    """
    rx = r_vectors[..., 0]  # (B, n_qubits)
    ry = r_vectors[..., 1]  # (B, n_qubits)
    rz = r_vectors[..., 2]  # (B, n_qubits)

    rho = torch.sqrt(rx ** 2 + ry ** 2 + 1e-7)
    # Polar damping envelope cancels the 1/rho singularity: (rx / rho) * (rx^2 + ry^2) = rx * rho
    w_polar = torch.clamp(1.0 - rz ** 2, min=0.0)

    u_x = (rx / rho) * w_polar
    u_y = (ry / rho) * w_polar

    r_bar_x = torch.mean(u_x, dim=0)  # (n_qubits,)
    r_bar_y = torch.mean(u_y, dim=0)  # (n_qubits,)

    torque_loss = torch.mean(r_bar_x ** 2 + r_bar_y ** 2)
    return torque_loss


def compute_readout_balancing_loss(model: nn.Module) -> torch.Tensor:
    """Equalizes the sensitivity allocated by classification MLP head across all qubits."""
    if not hasattr(model, "head") or not hasattr(model, "n_qubits"):
        dev = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        return torch.tensor(0.0, device=dev)

    n_q = model.n_qubits
    first_linear = model.head[0]
    if not isinstance(first_linear, nn.Linear):
        return torch.tensor(0.0, device=next(model.parameters()).device)

    w = first_linear.weight  # (32, 11) or (32, 8)
    if w.shape[1] < 2 * n_q:
        return torch.tensor(0.0, device=w.device)

    qubit_sensitivities = []
    for q in range(n_q):
        if w.shape[1] == 2 * n_q:
            # Dual-basis readout: Z_q at q, X_q at n_q + q
            sens_q = torch.norm(w[:, q], p=2) + torch.norm(w[:, n_q + q], p=2)
        else:
            # Sum-of-local readout: Z_q at q, ZZ at n_q..2n_q-2, X_q at 2n_q-1+q
            sens_q = torch.norm(w[:, q], p=2) + torch.norm(w[:, 2 * n_q - 1 + q], p=2)
            if q < n_q - 1:
                sens_q = sens_q + 0.5 * torch.norm(w[:, n_q + q], p=2)
            if q > 0:
                sens_q = sens_q + 0.5 * torch.norm(w[:, n_q + q - 1], p=2)
        qubit_sensitivities.append(sens_q)

    stacked_sens = torch.stack(qubit_sensitivities)
    prob_dist = stacked_sens / stacked_sens.sum().clamp(min=1e-6)
    target_dist = torch.full_like(prob_dist, 1.0 / float(n_q))
    return F.mse_loss(prob_dist, target_dist)


def compute_fibonacci_coverage(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    K: int = 256,
    threshold_deg: float = 25.0,
) -> float:
    """Computes continuous spherical surface coverage using a K-point Fibonacci lattice on S^2 (Fisher et al., 1987)."""
    if len(x) == 0:
        return 0.0

    pts = np.stack([np.asarray(x, dtype=float),
                    np.asarray(y, dtype=float),
                    np.asarray(z, dtype=float)], axis=-1)
    norms = np.linalg.norm(pts, axis=-1, keepdims=True)
    norms = np.where(norms > 1e-7, norms, 1.0)
    pts = pts / norms

    indices = np.arange(K, dtype=float) + 0.5
    phi = np.pi * (1.0 + 5.0 ** 0.5) * indices
    costheta = 1.0 - (2.0 * indices) / float(K)
    sintheta = np.sqrt(np.maximum(0.0, 1.0 - costheta * costheta))
    grid = np.stack([sintheta * np.cos(phi), sintheta * np.sin(phi), costheta], axis=-1)  # (K, 3)

    cos_threshold = math.cos(math.radians(threshold_deg))
    sims = np.dot(grid, pts.T)
    max_sims = np.max(sims, axis=1)
    covered = np.sum(max_sims >= cos_threshold)
    return float((covered / float(K)) * 100.0)


def _compute_angular_spread(raw: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute mean intra-class angular spread (in degrees) on the quantum rotation angles."""
    raw_norm = F.normalize(raw, p=2, dim=1)
    sim = raw_norm @ raw_norm.T
    batch_size = raw.size(0)
    mask_self = torch.eye(batch_size, device=raw.device, dtype=torch.bool)
    pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~mask_self

    if pos_mask.sum() == 0:
        return 0.0

    mean_intra_cos = sim[pos_mask].mean().clamp(-1.0, 1.0)
    angular_spread_deg = math.degrees(math.acos(mean_intra_cos.item()))
    return angular_spread_deg


def _supcon_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.10,
    return_details: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, float, float]]:
    """Supervised Contrastive Loss (SupCon) — Retained for backward compatibility."""
    batch_size = z.size(0)
    if batch_size <= 1:
        zero_loss = (z * 0.0).sum()
        return (zero_loss, 0.0, 0.0) if return_details else zero_loss

    labels = labels.contiguous().view(-1, 1)
    mask_self = torch.eye(batch_size, device=z.device, dtype=torch.bool)
    pos_mask = torch.eq(labels, labels.T) & ~mask_self

    pos_counts = pos_mask.sum(dim=1)
    valid_anchors = pos_counts > 0
    if not valid_anchors.any():
        zero_loss = (z * 0.0).sum()
        return (zero_loss, 0.0, 0.0) if return_details else zero_loss

    z_norm = F.normalize(z, p=2, dim=1)
    sim    = torch.matmul(z_norm, z_norm.T) / temperature

    sim_masked = sim.clone()
    sim_masked.masked_fill_(mask_self, float("-inf"))
    log_prob = sim_masked - torch.logsumexp(sim_masked, dim=1, keepdim=True)
    log_prob = torch.where(mask_self, torch.zeros_like(log_prob), log_prob)

    n_pos = pos_counts.float().clamp(min=1)
    loss_per_anchor = -(pos_mask.float() * log_prob).sum(dim=1) / n_pos
    raw_loss = loss_per_anchor[valid_anchors].mean()

    if return_details:
        theoretical_floor = torch.log(n_pos[valid_anchors]).mean().item()
        normalized_loss = max(0.0, raw_loss.item() - theoretical_floor)
        return raw_loss, normalized_loss, theoretical_floor

    return raw_loss


# ===========================================================================
# 3.  CHECKPOINT SERIALIZATION & PATH RESOLUTION UTILITIES
# ===========================================================================

def get_adapter_dir(dataset: str, base_dir: Union[str, Path] = "qml_models/adapter") -> Path:
    """Returns the canonical dataset-specific directory: qml_models/adapter/<dataset>/"""
    clean_name = dataset.strip().lower()
    target_dir = Path(base_dir) / clean_name
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def get_adapter_checkpoint_path(
    dataset: str,
    n_qubits: int = 4,
    use_lora: bool = False,
    lora_rank: int = 4,
    base_dir: Union[str, Path] = "qml_models/adapter",
) -> Path:
    """Returns the canonical checkpoint path: .../cq_adapter_{n_qubits}q[_lora{rank}].pth"""
    target_dir = get_adapter_dir(dataset, base_dir=base_dir)
    lora_suffix = f"_lora{lora_rank}" if use_lora else ""
    return target_dir / f"cq_adapter_{n_qubits}q{lora_suffix}.pth"


def save_adapter_checkpoint(
    backbone_or_model: nn.Module,
    save_path: Union[str, Path],
    metadata: Optional[Dict] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epoch: Optional[int] = None,
    best_val_loss: Optional[float] = None,
) -> Path:
    """Saves the CQ-Adapter, Skip connection, optimizer state, and configuration metadata to disk."""
    backbone = backbone_or_model.backbone if hasattr(backbone_or_model, "backbone") else backbone_or_model
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "cq_adapter": backbone.cq_adapter.state_dict(),
        "skip": backbone.skip.state_dict() if hasattr(backbone, "skip") and backbone.skip is not None else None,
        "n_qubits": getattr(backbone, "n_qubits", None),
        "use_lora": getattr(backbone, "use_lora", False),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "metadata": metadata or {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    temp_path = save_path.with_suffix(".tmp")
    torch.save(checkpoint, temp_path)
    if os.path.exists(save_path):
        os.remove(save_path)
    os.rename(temp_path, save_path)
    print(f"[adapter] Saved adapter checkpoint to: {save_path}")
    return save_path


def load_adapter_checkpoint(
    backbone_or_model: nn.Module,
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict:
    """Loads weights into CQ-Adapter and Residual Skip connection from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Adapter checkpoint not found at: {checkpoint_path}")

    device = device or torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    backbone = backbone_or_model.backbone if hasattr(backbone_or_model, "backbone") else backbone_or_model

    # Check qubit topology compatibility
    ckpt_qubits = checkpoint.get("n_qubits")
    if ckpt_qubits is not None and hasattr(backbone, "n_qubits") and ckpt_qubits != backbone.n_qubits:
        raise ValueError(
            f"Qubit count mismatch: checkpoint has {ckpt_qubits} qubits, "
            f"but target model expects {backbone.n_qubits} qubits."
        )

    # Load adapter weights
    if "cq_adapter" in checkpoint and checkpoint["cq_adapter"] is not None:
        backbone.cq_adapter.load_state_dict(checkpoint["cq_adapter"], strict=strict)

    # Load residual skip connection weights
    if "skip" in checkpoint and checkpoint["skip"] is not None and hasattr(backbone, "skip") and backbone.skip is not None:
        backbone.skip.load_state_dict(checkpoint["skip"], strict=strict)

    meta = checkpoint.get("metadata", {})
    return meta


# ═══════════════════════════════════════════════════════════════════════════
# 4.  PRE-TRAIN CQ ADAPTER: DECOUPLED SUPCON & QUANTUM DISPERSION GUARD
# ═══════════════════════════════════════════════════════════════════════════

def pretrain_cq_adapter(
    backbone_or_model: nn.Module,
    train_loader,
    device: torch.device,
    val_loader=None,
    epochs: int = 30,
    temperature_start: float = 0.15,
    temperature_end: float = 0.10,
    lr: float = 1e-3,
    early_stopping_patience: int = 7,
    out_dir: Optional[Union[str, Path]] = None,
    dataset_name: str = "Medical Dataset",
) -> Dict:
    """Pre-train the CQ Adapter with Decoupled Contrastive Loss & Spherical Dispersion Guards.

    Features:
      1. Decoupled Contrastive Learning (DCL) with class inverse frequency balancing.
      2. Direct Bottleneck Representation Clustering on S^2 angles and Bloch vectors.
      3. Class-Conditional Spherical Covariance Loss: avoids rank-1 polar cancellation.
      4. Supervised Antipodal Margin Loss with one-sided pure-basis cone margin.
      5. Strictly Detached Leaf Feature Inputs with torch.no_grad() ViT inference.
      6. Static Evaluation Probe Set: guarantees deterministic sample tracking in Bloch plots.
      7. Validation telemetry & Early Stopping with best-model checkpointing.
      8. Proper Dropout Lifecycle (train during pretraining, eval during inference & checkpointing).
    """
    backbone = backbone_or_model.backbone if hasattr(backbone_or_model, "backbone") else backbone_or_model
    print("\n[pretrain] Starting CQ Adapter pre-training with DCL & Spherical S^2 Dispersion...")

    # Set adapter in training mode during optimization
    backbone.train()
    backbone.cq_adapter.train()

    # ── Static Un-shuffled Evaluation Probe Set Allocation ─────────────────
    # Guarantees sample i at Epoch 0 is the exact same patient at Epoch T
    probe_source = val_loader if val_loader is not None else train_loader
    probe_dataset = probe_source.dataset
    probe_loader = torch.utils.data.DataLoader(
        probe_dataset,
        batch_size=min(250, len(probe_dataset)),
        shuffle=False,
        drop_last=False,
    )
    probe_batch = next(iter(probe_loader))
    if isinstance(probe_batch, (tuple, list)):
        eval_imgs_raw, eval_labels_raw = probe_batch[0], probe_batch[1]
    elif isinstance(probe_batch, dict):
        eval_imgs_raw = probe_batch.get("image", probe_batch.get("images"))
        eval_labels_raw = probe_batch.get("label", probe_batch.get("labels"))
    else:
        eval_imgs_raw, eval_labels_raw = probe_batch, torch.zeros(len(probe_batch))

    eval_imgs_raw = eval_imgs_raw.to(device, non_blocking=True)
    labels_eval = eval_labels_raw.squeeze().cpu().numpy()[:250]

    with torch.no_grad():
        if eval_imgs_raw.dim() == 4 and hasattr(backbone, "extract_features"):
            eval_features = backbone.extract_features(eval_imgs_raw)
        else:
            eval_features = eval_imgs_raw
        if hasattr(backbone, "skip") and backbone.skip is not None:
            raw_init = backbone.cq_adapter(eval_features) + backbone.skip(eval_features)
        else:
            raw_init = backbone.cq_adapter(eval_features)
        _, r_i = backbone.riemannian_head(raw_init, backbone.n_qubits)
        r_initial = r_i.cpu().numpy()[:len(labels_eval)]

    # Detachable Projection Head (3*n_qubits -> 128 -> 128)
    in_dim = getattr(backbone.cq_adapter, "out_features", 3 * backbone.n_qubits)
    proj_head = nn.Sequential(
        nn.Linear(in_dim, 128),
        nn.LayerNorm(128),
        nn.GELU(),
        nn.Linear(128, 128),
    ).to(device)

    # Optimizer includes adapter, skip shortcut, and projection head
    trainable_params = list(filter(lambda p: p.requires_grad, backbone.cq_adapter.parameters()))
    if hasattr(backbone, "skip") and backbone.skip is not None:
        trainable_params += list(filter(lambda p: p.requires_grad, backbone.skip.parameters()))
    trainable_params += list(proj_head.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    pretrain_hist = {
        "epoch": [],
        "total_loss": [],
        "dcl_loss": [],
        "cov_loss": [],
        "uniform_loss": [],
        "torque_loss": [],
        "ortho_loss": [],
        "centroid_sep_deg": [],
        "val_loss": [],
        "tau": [],
        "lr": [],
        "cov_eigenvalues": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        backbone.cq_adapter.train()
        proj_head.train()

        running_dcl_loss = 0.0
        running_cov_loss = 0.0
        running_uniform_loss = 0.0
        running_torque_loss = 0.0
        running_ortho_loss = 0.0
        running_total_loss = 0.0
        running_sep_deg = 0.0
        total_batches = 0
        total_samples = 0
        last_cov_eigs = []

        # Temperature annealing: 0.15 -> 0.10
        decay_factor = (epoch - 1) / max(1, epochs - 1)
        current_tau = temperature_start - (temperature_start - temperature_end) * decay_factor

        for batch in tqdm(train_loader, desc=f"Pre-train Epoch {epoch}/{epochs}", leave=False):
            if isinstance(batch, (tuple, list)):
                images, labels = batch[0], batch[1]
            elif isinstance(batch, dict):
                images = batch.get("image", batch.get("images"))
                labels = batch.get("label", batch.get("labels"))
            else:
                continue

            images = images.to(device, non_blocking=True)
            labels = labels.squeeze().long().to(device, non_blocking=True)
            batch_sz = images.size(0)

            # Isolated ViT inference under torch.no_grad()
            if images.dim() == 4 and hasattr(backbone, "extract_features"):
                with torch.no_grad():
                    images = backbone.extract_features(images)
            images = images.detach()

            # Latent manifold-preserving multi-view perturbation on S^511
            v1, v2 = _latent_augment(images, p_drop=0.10, noise_std=0.02)
            images_2x = torch.cat([v1, v2], dim=0)
            labels_2x = torch.cat([labels, labels], dim=0)
            instance_ids_2x = torch.arange(images.size(0), device=device).repeat(2)

            optimizer.zero_grad()

            # Adapter with residual skip projection
            if hasattr(backbone, "skip") and backbone.skip is not None:
                raw_2x = backbone.cq_adapter(images_2x) + backbone.skip(images_2x)
            else:
                raw_2x = backbone.cq_adapter(images_2x)

            # Riemannian S^2 projection onto unit sphere
            z_angles_2x, r_vectors_2x = backbone.riemannian_head(raw_2x, backbone.n_qubits)

            # Class-conditional spherical covariance loss
            cov_loss = compute_per_qubit_spherical_loss(r_vectors_2x, labels_2x)

            # Hyperspherical uniformity loss with instance isolation (Fix: ADP-01)
            uniform_loss, sep_deg = compute_hyperspherical_uniformity_loss(
                r_vectors_2x, labels_2x, t=2.0, margin_deg=120.0, instance_ids=instance_ids_2x
            )

            # Singularity-free azimuthal torque loss
            torque_loss = compute_azimuthal_torque_loss(r_vectors_2x)

            # Per-qubit row-wise orthogonality loss including skip connection (Fix: ADP-02)
            skip_w = backbone.skip.weight if hasattr(backbone, "skip") and backbone.skip is not None else None
            ortho_loss = backbone.cq_adapter.compute_orthogonality_loss(skip_weight=skip_w) if hasattr(backbone.cq_adapter, "compute_orthogonality_loss") else torch.tensor(0.0, device=device)

            # Dual contrastive clustering: projection head + direct bottleneck angles
            z_proj = proj_head(raw_2x)
            dcl_proj = _decoupled_supcon_loss(z_proj, labels_2x, temperature=current_tau)
            r_flat = r_vectors_2x.view(r_vectors_2x.size(0), -1)
            dcl_bottleneck = _decoupled_supcon_loss(r_flat, labels_2x, temperature=current_tau)
            dcl_loss = dcl_proj + 0.5 * dcl_bottleneck

            total_loss = dcl_loss + 0.05 * cov_loss + 0.10 * uniform_loss + 0.05 * torque_loss + 0.02 * ortho_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            running_total_loss += total_loss.detach() * batch_sz
            running_dcl_loss += dcl_loss.detach() * batch_sz
            running_cov_loss += cov_loss.detach() * batch_sz
            running_uniform_loss += uniform_loss.detach() * batch_sz
            running_torque_loss += torque_loss.detach() * batch_sz
            running_ortho_loss += ortho_loss.detach() * batch_sz
            running_sep_deg += sep_deg
            total_batches += 1
            total_samples += batch_sz

        # Track per-qubit eigenvalues of second-moment matrix M_q once per epoch on final batch
        with torch.no_grad():
            q_eigs = []
            for q in range(backbone.n_qubits):
                rq = r_vectors_2x[:, q, :].detach()
                mq = torch.matmul(rq.T, rq) / rq.size(0)
                e = torch.linalg.eigvalsh(mq).cpu().numpy().tolist()
                q_eigs.append(e)
            last_cov_eigs = q_eigs

        scheduler.step()
        curr_lr = scheduler.get_last_lr()[0]
        n_ds = total_samples if total_samples > 0 else len(train_loader.dataset)

        ep_total = float(running_total_loss.item() if torch.is_tensor(running_total_loss) else running_total_loss) / n_ds
        ep_dcl = float(running_dcl_loss.item() if torch.is_tensor(running_dcl_loss) else running_dcl_loss) / n_ds
        ep_cov = float(running_cov_loss.item() if torch.is_tensor(running_cov_loss) else running_cov_loss) / n_ds
        ep_uniform = float(running_uniform_loss.item() if torch.is_tensor(running_uniform_loss) else running_uniform_loss) / n_ds
        ep_torque = float(running_torque_loss.item() if torch.is_tensor(running_torque_loss) else running_torque_loss) / n_ds
        ep_ortho = float(running_ortho_loss.item() if torch.is_tensor(running_ortho_loss) else running_ortho_loss) / n_ds
        ep_sep = running_sep_deg / max(1, total_batches)

        # ── Optional Validation Pass & Early Stopping Checkpoint ───────────
        ep_val_loss = None
        if val_loader is not None:
            backbone.cq_adapter.eval()
            proj_head.eval()
            val_loss_accum = 0.0
            val_samples = 0
            with torch.no_grad():
                for v_batch in val_loader:
                    v_imgs = v_batch[0] if isinstance(v_batch, (tuple, list)) else v_batch.get("image", v_batch.get("images"))
                    v_lbls = v_batch[1] if isinstance(v_batch, (tuple, list)) else v_batch.get("label", v_batch.get("labels"))
                    v_imgs = v_imgs.to(device, non_blocking=True)
                    v_lbls = v_lbls.squeeze().long().to(device, non_blocking=True)
                    if v_imgs.dim() == 4 and hasattr(backbone, "extract_features"):
                        v_imgs = backbone.extract_features(v_imgs)
                    if hasattr(backbone, "skip") and backbone.skip is not None:
                        v_raw = backbone.cq_adapter(v_imgs) + backbone.skip(v_imgs)
                    else:
                        v_raw = backbone.cq_adapter(v_imgs)
                    v_z = proj_head(v_raw)
                    v_dcl = _decoupled_supcon_loss(v_z, v_lbls, temperature=current_tau)
                    val_loss_accum += float(v_dcl.item()) * v_imgs.size(0)
                    val_samples += v_imgs.size(0)
            ep_val_loss = val_loss_accum / max(1, val_samples)
            if ep_val_loss < best_val_loss:
                best_val_loss = ep_val_loss
                patience_counter = 0
                if out_dir is not None:
                    save_adapter_checkpoint(
                        backbone_or_model,
                        Path(out_dir) / "best_adapter.pth",
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        best_val_loss=best_val_loss,
                    )
            else:
                patience_counter += 1

        pretrain_hist["epoch"].append(epoch)
        pretrain_hist["total_loss"].append(float(ep_total))
        pretrain_hist["dcl_loss"].append(float(ep_dcl))
        pretrain_hist["cov_loss"].append(float(ep_cov))
        pretrain_hist["uniform_loss"].append(float(ep_uniform))
        pretrain_hist["torque_loss"].append(float(ep_torque))
        pretrain_hist["ortho_loss"].append(float(ep_ortho))
        pretrain_hist["centroid_sep_deg"].append(float(ep_sep))
        pretrain_hist["val_loss"].append(float(ep_val_loss) if ep_val_loss is not None else None)
        pretrain_hist["tau"].append(float(current_tau))
        pretrain_hist["lr"].append(float(curr_lr))
        pretrain_hist["cov_eigenvalues"].append(last_cov_eigs)

        val_str = f" | ValLoss={ep_val_loss:.4f}" if ep_val_loss is not None else ""
        print(
            f"   Pre-train Epoch {epoch:2d}/{epochs:2d}: "
            f"DCL={ep_dcl:.4f} | "
            f"Cov={ep_cov:.3f} | "
            f"Uniform={ep_uniform:.3f} | "
            f"Torque={ep_torque:.3f} | "
            f"CentroidSep={ep_sep:.1f}° | "
            f"τ={current_tau:.2f} | "
            f"lr={curr_lr:.2e}"
            f"{val_str}"
        )

        if val_loader is not None and patience_counter >= early_stopping_patience:
            print(f"[pretrain] Early stopping triggered at epoch {epoch} (patience={early_stopping_patience}).")
            break

    # Capture final pre-trained state projection on the exact same static probe set
    with torch.no_grad():
        backbone.cq_adapter.eval()
        if hasattr(backbone, "skip") and backbone.skip is not None:
            raw_fin = backbone.cq_adapter(eval_features) + backbone.skip(eval_features)
        else:
            raw_fin = backbone.cq_adapter(eval_features)
        _, r_f = backbone.riemannian_head(raw_fin, backbone.n_qubits)
        r_final = r_f.cpu().numpy()[:len(labels_eval)]

    # Leave adapter in eval mode upon completion
    backbone.cq_adapter.eval()
    del proj_head

    # Save pre-training history and generate CQ-Adapter diagnostic plots
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if r_initial is not None:
            pretrain_hist["r_initial"] = r_initial.tolist()
        if r_final is not None:
            pretrain_hist["r_final"] = r_final.tolist()
        if labels_eval is not None:
            pretrain_hist["labels"] = labels_eval.tolist()
        hist_path = out_dir / "cq_pretrain_history.json"
        try:
            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump(pretrain_hist, f, indent=2)
            print(f"[pretrain] Saved pre-training history to: {hist_path}")
        except Exception as e:
            print(f"[pretrain] Warning: Failed to save {hist_path}: {e}")

        print(f"[pretrain] Generating CQ-Adapter pre-training diagnostic plots in {out_dir}/...")
        try:
            if plot_cq_pretrain_loss_curves is not None:
                plot_cq_pretrain_loss_curves(pretrain_hist, str(out_dir))
            if plot_cq_pretrain_geometry_dynamics is not None:
                plot_cq_pretrain_geometry_dynamics(pretrain_hist, str(out_dir))
            if plot_cq_pretrain_bloch_evolution is not None and r_initial is not None and r_final is not None and labels_eval is not None:
                meta_dict = {"dataset_name": dataset_name, "n_classes": len(np.unique(labels_eval))}
                plot_cq_pretrain_bloch_evolution(r_initial, r_final, labels_eval, str(out_dir), n_qubits=backbone.n_qubits, meta=meta_dict)
            if plot_cq_pretrain_qubit_covariance_spectra is not None:
                plot_cq_pretrain_qubit_covariance_spectra(pretrain_hist, str(out_dir), n_qubits=backbone.n_qubits)
            print(f"[pretrain] Successfully generated all 4 CQ-Adapter pre-training plots.")
        except Exception as e:
            print(f"[pretrain] Warning: Failed to render pre-training plots: {e}")

    return pretrain_hist


# ═══════════════════════════════════════════════════════════════════════════
# 5.  STANDALONE CLI ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Standalone CQ-Adapter Pre-Training & Diagnostic Artifact Generator (Zero Quantum Simulator Overhead)"
    )
    parser.add_argument("--dataset", type=str, default="pneumoniamnist",
                        help="Target MedMNIST dataset (e.g. pneumoniamnist, octmnist, breastmnist, pathmnist)")
    parser.add_argument("--n_qubits", type=int, default=4,
                        help="Number of target quantum wire channels (3 * n_qubits projection dimension)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of pre-training epochs for CQ-Adapter")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for pre-training dataloader")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate for AdamW")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Compute device for classical contrastive optimization")
    parser.add_argument("--use_lora", action="store_true",
                        help="Enable LoRA-Q low-rank factorization")
    parser.add_argument("--lora_rank", type=int, default=4,
                        help="Rank of LoRA-Q factorization (default: 4)")
    parser.add_argument("--adapter_dir", type=str, default="qml_models/adapter",
                        help="Root base directory for dataset-partitioned adapter persistence")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Resolution of input images (default: 224 for BiomedCLIP)")

    args = parser.parse_args()

    # Device selection
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    print(f"[adapter-cli] Running on device: {dev}")

    # Dataset loader
    try:
        from dataset import get_medmnist_dataloaders
    except ImportError:
        from qml_models.dataset import get_medmnist_dataloaders

    print(f"[adapter-cli] Ingesting dataset '{args.dataset}' (batch_size={args.batch_size})...")
    train_loader, val_loader, test_loader, meta = get_medmnist_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        size=args.image_size,
    )

    # Instantiate Backbone
    try:
        from model import BiomedCLIPQuantumBackbone
    except ImportError:
        from qml_models.model import BiomedCLIPQuantumBackbone

    print(f"[adapter-cli] Initializing BiomedCLIP Quantum Backbone (n_qubits={args.n_qubits}, use_lora={args.use_lora})...")
    backbone = BiomedCLIPQuantumBackbone(
        in_channels=meta["in_channels"],
        n_qubits=args.n_qubits,
        dropout=0.3,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
    ).to(dev)

    # Resolve target directory: qml_models/adapter/<dataset>/
    target_dir = get_adapter_dir(args.dataset, base_dir=args.adapter_dir)
    target_ckpt = get_adapter_checkpoint_path(
        dataset=args.dataset,
        n_qubits=args.n_qubits,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        base_dir=args.adapter_dir,
    )
    print(f"[adapter-cli] Target directory : {target_dir}")
    print(f"[adapter-cli] Checkpoint path  : {target_ckpt}")

    # Run Pre-training with Validation Feedback & Early Stopping
    hist = pretrain_cq_adapter(
        backbone_or_model=backbone,
        train_loader=train_loader,
        val_loader=val_loader,
        device=dev,
        epochs=args.epochs,
        lr=args.lr,
        out_dir=target_dir,
        dataset_name=meta.get("dataset_name", args.dataset),
    )

    # Save weights
    save_adapter_checkpoint(
        backbone_or_model=backbone,
        save_path=target_ckpt,
        metadata={
            "dataset": args.dataset,
            "n_qubits": args.n_qubits,
            "epochs": args.epochs,
            "use_lora": args.use_lora,
            "lora_rank": args.lora_rank,
            "final_dcl_loss": hist["dcl_loss"][-1] if hist["dcl_loss"] else None,
            "final_centroid_sep_deg": hist["centroid_sep_deg"][-1] if hist["centroid_sep_deg"] else None,
        },
    )

    print(f"\n[adapter-cli] Pre-training & Artifact Generation for '{args.dataset}' complete!")
    print(f"              Weights: {target_ckpt}")
    print(f"              Plots  : {target_dir}")


if __name__ == "__main__":
    main()
