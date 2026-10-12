"""
engine.py — Training Loop, QEWC Continual Learning & Comprehensive Benchmarking
=================================================================================
CLI usage:
  python engine.py --dataset pneumoniamnist --epochs 15 --n_qubits 4 --n_layers 3

Generates 20+ diagnostic plots in  outputs/<dataset>/
"""

from __future__ import annotations

import sys
from pathlib import Path

# Force UTF-8 output on Windows to prevent cp1252 UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Ensure engine is registered in sys.modules if invoked directly as __main__
if __name__ == "__main__":
    sys.modules.setdefault("engine", sys.modules["__main__"])

# Ensure project root and package directory are in sys.path
_pkg_root = str(Path(__file__).resolve().parent.parent)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
_module_dir = str(Path(__file__).resolve().parent)
if _module_dir not in sys.path:
    sys.path.insert(0, _module_dir)

print("[engine] Initializing Hybrid CNN-VQC Engine (loading dependencies)...", flush=True)

import argparse
import copy
import hashlib
import json
import math
import os
# Cap OpenMP and MKL threads to prevent WSL2 Hyper-V thread thrashing and memory exhaustion
_num_threads = str(min(4, os.cpu_count() or 4))
os.environ["OMP_NUM_THREADS"] = _num_threads
os.environ["MKL_NUM_THREADS"] = _num_threads
os.environ["OPENBLAS_NUM_THREADS"] = _num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = _num_threads
os.environ["NUMEXPR_NUM_THREADS"] = _num_threads

import textwrap
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import seaborn as sns
import torch
torch.set_num_threads(int(_num_threads))
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sp_stats
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from dataset import (
    DATASET_CONFIG,
    compute_class_weights,
    expectation_quantum_mixup,
    get_kfold_loaders,
    get_medmnist_dataloaders,
    phase_perturbation,
    plot_class_distribution,
    plot_pixel_intensity_histogram,
    plot_sample_grid,
    quantum_mixup,
    su2_phase_perturbation,
)
from model import (
    ClassicalBaseline,
    HybridCNNVQC,
    QuantumDropoutTorchLayer,
    build_models,
    create_vqc_qnode,
    get_torch_device,
    get_pennylane_device,
)
try:
    from qnas import (
        CircuitGenome,
        GateGene,
        QNASOptimizer,
        compute_latent_mutual_information,
        load_genome,
        save_genome,
    )
except ImportError:
    from qml_models.qnas import (
        CircuitGenome,
        GateGene,
        QNASOptimizer,
        compute_latent_mutual_information,
        load_genome,
        save_genome,
    )

try:
    from adapter import (
        get_adapter_dir,
        get_adapter_checkpoint_path,
        load_adapter_checkpoint,
        save_adapter_checkpoint,
        pretrain_cq_adapter,
        compute_per_qubit_spherical_loss,
        compute_azimuthal_torque_loss,
        compute_hyperspherical_uniformity_loss,
        compute_antipodal_vmf_loss,
        compute_readout_balancing_loss,
        compute_fibonacci_coverage,
        _decoupled_supcon_loss,
        _latent_augment,
    )
except ImportError:
    from qml_models.adapter import (
        get_adapter_dir,
        get_adapter_checkpoint_path,
        load_adapter_checkpoint,
        save_adapter_checkpoint,
        pretrain_cq_adapter,
        compute_per_qubit_spherical_loss,
        compute_azimuthal_torque_loss,
        compute_hyperspherical_uniformity_loss,
        compute_antipodal_vmf_loss,
        compute_readout_balancing_loss,
        compute_fibonacci_coverage,
        _decoupled_supcon_loss,
        _latent_augment,
    )

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)

# ═══════════════════════════════════════════════════════════════════════════
# 1.  LOSS FUNCTIONS & REGULARIZERS (Decoupled to qml_models.losses)
# ═══════════════════════════════════════════════════════════════════════════
try:
    from losses import (
        FocalLoss,
        resolve_asymmetric_loss_params,
        _compute_class_counts,
        ClassBalancedLDAMLoss,
        QuantumInfoNCELoss,
        compute_drusen_cnv_orthogonality_loss,
        FeatureDecorrelationLoss,
        SaliencyConcentrationLoss,
    )
except ImportError:
    from qml_models.losses import (
        FocalLoss,
        resolve_asymmetric_loss_params,
        _compute_class_counts,
        ClassBalancedLDAMLoss,
        QuantumInfoNCELoss,
        compute_drusen_cnv_orthogonality_loss,
        FeatureDecorrelationLoss,
        SaliencyConcentrationLoss,
    )

# ═══════════════════════════════════════════════════════════════════════════
# ANTI-SHORTCUT TELEMETRY & MULTI-RATE OPTIMIZER UTILITIES (Tasks 1.3 & 3.6)
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_shortcut_score(
    model: torch.nn.Module,
    val_loader: Any,
    device: torch.device,
    max_batches: int = 5,
) -> float:
    """Computes mean shortcut score: confidence(background_only) / confidence(original) (Task 1.3)."""
    model.eval()
    try:
        from preprocessing import compute_foreground_mask
    except ImportError:
        try:
            from qml_models.preprocessing import compute_foreground_mask
        except ImportError:
            def compute_foreground_mask(img):
                return (img > img.mean(dim=(-2, -1), keepdim=True)).float()

    scores = []
    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if b_idx >= max_batches:
                break
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch.get("image")
            if imgs is None:
                continue
            imgs = imgs.to(device)
            mask = compute_foreground_mask(imgs)
            bg_imgs = imgs * (1.0 - mask)

            out_orig = model(imgs)
            out_bg = model(bg_imgs)

            conf_orig = torch.softmax(out_orig, dim=-1).max(dim=-1)[0]
            conf_bg = torch.softmax(out_bg, dim=-1).max(dim=-1)[0]

            batch_scores = conf_bg / (conf_orig + 1e-4)
            scores.extend(batch_scores.cpu().tolist())

    return float(np.mean(scores)) if scores else 0.0


def build_multi_rate_optimizer(
    model: torch.nn.Module,
    lr_backbone: float = 1e-5,
    lr_adapter: float = 1e-3,
    lr_quantum: float = 1e-2,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Builds multi-rate optimizer with distinct parameter group learning rates (Task 3.6)."""
    backbone_params = []
    adapter_params = []
    quantum_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "vision_encoder" in name or "backbone.vision_encoder" in name:
            backbone_params.append(param)
        elif "cq_adapter" in name or "skip" in name:
            adapter_params.append(param)
        elif "qlayer" in name:
            quantum_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone, "name": "backbone"})
    if adapter_params:
        param_groups.append({"params": adapter_params, "lr": lr_adapter, "name": "adapter"})
    if quantum_params:
        param_groups.append({"params": quantum_params, "lr": lr_quantum, "name": "quantum"})
    if other_params:
        param_groups.append({"params": other_params, "lr": lr_adapter, "name": "head"})

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# ═══════════════════════════════════════════════════════════════════════════
# 2.5 FEATURE CACHING
# ═══════════════════════════════════════════════════════════════════════════
FEATURE_CACHE_SCHEMA_VERSION = "3.0"


def _feature_cache_contract(model, loader, extra_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a reproducibility contract for an extracted feature cache.

    Cache files are training inputs, not disposable speedups.  The contract
    binds them to the dataset ordering/labels, preprocessing owner, encoder
    identity, and requested split so a stale cache cannot silently change the
    clinical experiment.
    """
    dataset = getattr(loader, "dataset", None)
    labels = getattr(dataset, "labels", None)
    if labels is None and hasattr(dataset, "tensors") and len(dataset.tensors) >= 2:
        labels = dataset.tensors[1]
    if torch.is_tensor(labels):
        labels_np = labels.detach().cpu().numpy()
    elif labels is not None:
        labels_np = np.asarray(labels)
    else:
        labels_np = np.empty((0,), dtype=np.int64)
    labels_np = np.ascontiguousarray(labels_np)

    backbone = getattr(model, "backbone", None)
    preprocess_mode = str(getattr(backbone, "preprocess_mode", "unknown"))
    encoder_id = getattr(backbone, "encoder_id", "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    encoder_source = getattr(backbone, "encoder_source", "pretrained")
    contract = {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "dataset_class": (
            f"{dataset.__class__.__module__}.{dataset.__class__.__qualname__}"
            if dataset is not None else "unknown"
        ),
        "n_samples": int(len(dataset)) if dataset is not None else int(len(labels_np)),
        "label_sha256": hashlib.sha256(labels_np.tobytes()).hexdigest(),
        "feature_dim": 512,
        "feature_dtype": "torch.float32",
        "preprocess_mode": preprocess_mode,
        "encoder_id": encoder_id,
        "encoder_source": encoder_source,
    }
    if extra_metadata:
        contract.update({str(k): v for k, v in extra_metadata.items()})
    return contract


def extract_and_cache_features(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cache_path: str,
    cache_in_vram: bool = False,
    cache_metadata: Optional[Dict[str, Any]] = None,
) -> torch.utils.data.DataLoader:
    import os
    import gc
    expected_contract = _feature_cache_contract(model, loader, cache_metadata)
    use_existing = False
    if os.path.exists(cache_path):
        try:
            data = torch.load(cache_path, map_location="cpu", weights_only=True)
            use_existing = data.get("cache_contract") == expected_contract
            if use_existing:
                print(f"[cache] Loading verified feature cache from {cache_path}")
                all_features = data["features"]
                all_labels = data["labels"]
            else:
                print(f"[cache] Rejecting stale or legacy feature cache: {cache_path}")
        except Exception as exc:
            print(f"[cache] Rejecting unreadable feature cache ({exc}): {cache_path}")

    if not use_existing:
        print(f"[cache] Extracting features to {cache_path}...")
        model.eval()
        all_features = []
        all_labels = []

        # Never extract through a WeightedRandomSampler.  Sampling with
        # replacement would cache duplicates and omit patients, permanently
        # changing the training cohort.  Cache the underlying dataset once in
        # deterministic order, then recreate the requested sampler below.
        source_loader = torch.utils.data.DataLoader(
            loader.dataset,
            batch_size=loader.batch_size or 32,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        extract_chunk_size = min(source_loader.batch_size or 32, 32)
        with torch.inference_mode():
            for batch_i, (images, labels) in enumerate(tqdm(source_loader, desc="Caching features", leave=False)):
                labels = labels.squeeze().long()
                b_sz = images.size(0)
                sub_feats = []
                for start_idx in range(0, b_sz, extract_chunk_size):
                    end_idx = min(start_idx + extract_chunk_size, b_sz)
                    chunk_imgs = images[start_idx:end_idx].to(device)
                    f_chunk = model.backbone.extract_features(chunk_imgs)
                    sub_feats.append(f_chunk.cpu())
                    del chunk_imgs, f_chunk

                all_features.append(torch.cat(sub_feats, dim=0))
                all_labels.append(labels.cpu())
                del images, labels, sub_feats

                if (batch_i + 1) % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        all_features = torch.cat(all_features, dim=0).float()
        all_labels = torch.cat(all_labels, dim=0)
        expected_contract["n_samples"] = int(all_features.shape[0])
        expected_contract["feature_dim"] = int(all_features.shape[-1])
        expected_contract["feature_dtype"] = str(all_features.dtype)
        torch.save({
            "cache_contract": expected_contract,
            "features": all_features,
            "labels": all_labels,
        }, cache_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    pin_mem = torch.cuda.is_available() and not cache_in_vram
    if cache_in_vram and device.type == "cuda":
        mem_mb = (all_features.element_size() * all_features.nelement() + all_labels.element_size() * all_labels.nelement()) / 1e6
        print(f"[cache] Preloading {all_features.size(0)} features ({mem_mb:.2f} MB) directly into GPU VRAM ({device})...")
        feat_tensor = all_features.to(device)
        label_tensor = all_labels.to(device)
    else:
        feat_tensor = all_features
        label_tensor = all_labels

    dataset = torch.utils.data.TensorDataset(feat_tensor, label_tensor)
    sampler = None
    shuffle = False
    if hasattr(loader, "sampler") and loader.sampler is not None:
        from torch.utils.data import WeightedRandomSampler, RandomSampler
        if isinstance(loader.sampler, WeightedRandomSampler):
            labels_np = all_labels.numpy().squeeze()
            class_counts = np.bincount(labels_np)
            class_weights = 1.0 / np.maximum(class_counts, 1)
            sample_weights = class_weights[labels_np]
            sampler = WeightedRandomSampler(weights=torch.as_tensor(sample_weights, dtype=torch.double), num_samples=len(sample_weights), replacement=True)
        elif isinstance(loader.sampler, RandomSampler):
            shuffle = True
    else:
        shuffle = True

    return torch.utils.data.DataLoader(
        dataset, batch_size=loader.batch_size, 
        shuffle=shuffle, sampler=sampler,
        num_workers=0, pin_memory=pin_mem, drop_last=loader.drop_last
    )

def _apply_ansatz(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    n_qubits: int,
    n_layers: int,
    reupload: bool = True,
    genome: Optional[object] = None,
):
    """Execute dynamic QNAS genome gates or StronglyEntanglingLayers."""
    if genome is not None:
        if isinstance(genome, dict):
            try:
                from qnas import CircuitGenome
            except ImportError:
                from qml_models.qnas import CircuitGenome
            genome = CircuitGenome.from_dict(genome)
        w_idx = 0
        for l, layer in enumerate(genome.layers):
            if reupload or l == 0:
                for i in range(n_qubits):
                    qml.U3(inputs[..., i], inputs[..., i + n_qubits], inputs[..., i + 2 * n_qubits], wires=i)
            for gene in layer:
                if gene.gate_type == "Rx":
                    qml.RX(weights[w_idx], wires=gene.target); w_idx += 1
                elif gene.gate_type == "Ry":
                    qml.RY(weights[w_idx], wires=gene.target); w_idx += 1
                elif gene.gate_type == "Rz":
                    qml.RZ(weights[w_idx], wires=gene.target); w_idx += 1
                elif gene.gate_type == "U3":
                    qml.U3(weights[w_idx], weights[w_idx + 1], weights[w_idx + 2], wires=gene.target); w_idx += 3
                elif gene.gate_type == "CNOT":
                    qml.CNOT(wires=[gene.control, gene.target])
                elif gene.gate_type == "CZ":
                    qml.CZ(wires=[gene.control, gene.target])
                elif gene.gate_type == "Identity":
                    qml.Identity(wires=gene.target)
    elif reupload:
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.U3(inputs[..., i], inputs[..., i + n_qubits], inputs[..., i + 2 * n_qubits], wires=i)
            w_l = weights[l:l+1] if hasattr(weights, "ndim") and weights.ndim >= 3 else weights
            qml.StronglyEntanglingLayers(w_l, wires=range(n_qubits))
    else:
        for i in range(n_qubits):
            qml.U3(inputs[..., i], inputs[..., i + n_qubits], inputs[..., i + 2 * n_qubits], wires=i)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))


# 2.  TRAINING  &  EVALUATION  LOOPS
# ═══════════════════════════════════════════════════════════════════════════
def _compute_entanglement_entropy_batch(
    model: "HybridCNNVQC",
    z_batch: torch.Tensor,
    n_qubits: Optional[int] = None,
) -> torch.Tensor:
    """Differentiable Bipartite Von Neumann Entanglement Entropy (Fix: Defect 3).

    Returns a torch.Tensor WITH gradient, not a detached float.
    Uses PyTorch-native SVD to maintain autograd graph: ∂S/∂θ ≠ 0.

    Previous implementation used np.linalg.svd on NumPy arrays, producing a Python
    float that contributed exactly 0.0 gradient to the quantum parameters.
    """
    n_qubits = getattr(model, "n_qubits", getattr(getattr(model, "genome", None), "n_qubits", n_qubits or 4))
    if hasattr(model, "get_state_qnode"):
        _state_circ = model.get_state_qnode(diff_method="backprop")
    else:
        dev_bp = qml.device("default.qubit", wires=n_qubits)
        reupload = getattr(model, "reupload", True)
        n_layers = getattr(model, "n_layers", 3)
        genome = getattr(model, "genome", None)

        @qml.qnode(dev_bp, interface="torch", diff_method="backprop")
        def _state_circ(inputs, weights):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            _apply_ansatz(inputs, weights, n_qubits, n_layers, reupload=reupload, genome=genome)
    q_weights = None
    for n_p, p in model.named_parameters():
        if "qlayer" in n_p and "weights" in n_p:
            q_weights = p
            break
    if q_weights is None:
        return torch.tensor(0.0, device=z_batch.device)

    n_a = n_qubits // 2
    dim_a = 2 ** n_a
    dim_b = 2 ** (n_qubits - n_a)

    entropies = []
    # Sample up to 2 items from the batch for speed (default.qubit backprop is slower)
    n_sample = min(2, z_batch.size(0))
    for i in range(n_sample):
        state = _state_circ(z_batch[i], q_weights)
        psi = state.reshape(dim_a, dim_b)
        svd_vals = torch.linalg.svdvals(psi)
        probs = svd_vals ** 2
        probs = probs.clamp(min=1e-12)
        s_cont = -torch.sum(probs * torch.log2(probs + 1e-30))

        # Differentiable Brennen-Meyer-Wallach linear entropy for arbitrary QNAS DAGs
        q_val = torch.tensor(0.0, device=z_batch.device, dtype=torch.float32)
        for k in range(n_qubits):
            state_tensor = state.view(*([2] * n_qubits))
            moved = state_tensor.transpose(0, k).reshape(2, -1)
            rho_k = torch.matmul(moved, moved.conj().T)
            purity = torch.real(torch.trace(torch.matmul(rho_k, rho_k)))
            q_val = q_val + (1.0 - purity)

        q_mw = (2.0 / float(n_qubits)) * q_val
        # Meyer-Wallach and bipartite entropy are different measures. A maximum
        # between them is not von Neumann entropy and can create a false floor.
        entropies.append(s_cont)

    if entropies:
        return torch.stack(entropies).mean()
    return torch.tensor(0.0, device=z_batch.device)


def _get_component_grad_norm(params_iter) -> torch.Tensor:
    """Compute L2 gradient norm for a parameter iterator (before clipping).

    Uses detached GPU tensor accumulation to avoid CUDA sync stalls during training.
    """
    norms = []
    for p in params_iter:
        if p.grad is not None:
            norms.append(p.grad.detach().norm(2))
    if not norms:
        return torch.tensor(0.0)
    return torch.norm(torch.stack(norms), 2).detach()


def _compute_differentiable_kta(
    model: "HybridCNNVQC",
    z_sub: torch.Tensor,
    labels_sub: torch.Tensor,
    q_weights: torch.Tensor,
    n_classes: int,
) -> torch.Tensor:
    r"""Differentiable Kernel-Target Alignment for training-time regularization (Fix: FL-09).

    Computes the Frobenius alignment between the quantum state fidelity Gram matrix
    K_{ij} = |<psi(z_i)|psi(z_j)>|^2 and a categorical Kronecker delta target kernel
    Y_{ij} = delta(y_i, y_j) - 1/C (Cortes et al., JMLR 2012):

        KTA(K, Y) = Tr(K Y) / (||K||_F * ||Y||_F)

    Returns a differentiable scalar KTA value with gradient flow through q_weights.
    Gradients are connected via PyTorch autograd through the state circuit.

    Note: For efficiency, this is computed on a small subsample (typically 32 points).
    """
    n_qubits = getattr(model, "n_qubits", 4)
    M = z_sub.size(0)
    if M < 2:
        return torch.tensor(0.0, device=z_sub.device)

    # Get or build state circuit
    if hasattr(model, "get_state_qnode"):
        _state_circ = model.get_state_qnode(diff_method="backprop")
    else:
        dev_bp = qml.device("default.qubit", wires=n_qubits)
        reupload = getattr(model, "reupload", True)
        n_layers = getattr(model, "n_layers", 3)
        genome = getattr(model, "genome", None)

        @qml.qnode(dev_bp, interface="torch", diff_method="backprop")
        def _state_circ(inputs, weights):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            _apply_ansatz(inputs, weights, n_qubits, n_layers, reupload=reupload, genome=genome)
            return qml.state()

    # Compute state vectors (autograd-connected through q_weights)
    try:
        states = _state_circ(z_sub, q_weights)
        if states.dim() == 1 and M == 1:
            states = states.unsqueeze(0)
    except Exception:
        states = torch.stack([_state_circ(z_sub[i], q_weights) for i in range(M)])

    # Quantum state fidelity Gram matrix: K_ij = |<psi_i|psi_j>|^2 (NaN-safe, autograd-connected)
    inner_prods = torch.matmul(states, states.conj().T)
    K = inner_prods.real.pow(2) + inner_prods.imag.pow(2)  # (M, M)

    # Target kernel: Kronecker delta with centering (Cortes et al., JMLR 2012)
    y_np = labels_sub.detach().cpu().numpy()
    labels_eq = (y_np[:, None] == y_np[None, :]).astype(np.float32)
    n_c = max(len(np.unique(y_np)), 2)
    if n_c > 2:
        Y_np = labels_eq - (1.0 - labels_eq) / (n_c - 1)
    else:
        Y_np = labels_eq * 2.0 - 1.0
    Y = torch.tensor(Y_np, device=z_sub.device, dtype=torch.float32)

    # Centered Frobenius alignment (KTA)
    N = K.shape[0]
    H = torch.eye(N, device=K.device) - torch.ones(N, N, device=K.device) / N
    K_c = H @ K @ H
    Y_c = H @ Y @ H
    tr_KY = torch.sum(K_c * Y_c)
    tr_K2 = torch.sum(K_c * K_c)
    tr_Y2 = torch.sum(Y_c * Y_c)
    denom = torch.sqrt(torch.clamp(tr_K2 * tr_Y2, min=1e-12))
    kta = tr_KY / denom

    return kta


def _scheduled_mixup_alpha(
    epoch: int,
    alpha_target: float = 0.15,
    warmup_end: int = 5,
    ramp_end: int = 15,
) -> float:
    """Curriculum mixup schedule (Fix: Flaw 4 — Scheduled Augmentation Warmup).

    Epochs 1-warmup_end : alpha = 0.0 (clean data, sharp boundary learning)
    Epochs warmup_end+1 - ramp_end : linear ramp to alpha_target
    Epochs > ramp_end   : alpha = alpha_target
    """
    if epoch <= warmup_end:
        return 0.0
    elif epoch <= ramp_end:
        return alpha_target * (epoch - warmup_end) / (ramp_end - warmup_end)
    return alpha_target


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_classes: int,
    use_quantum_aug: bool = True,
    mixup_alpha: float = 0.15,
    phase_eps: float = 0.05,
    clip_grad: float = 5.0,
    epoch: int = 1,
    lambda_ent: float = 0.05,
    s_target_frac: float = 0.70,
    qsam: Optional[QSAM] = None,
    lambda_infonce: float = 0.0,
    infonce_fn: Optional[QuantumInfoNCELoss] = None,
    grad_accum_steps: int = 1,
    qsam_freq: int = 1,
    use_amp: bool = False,
    post_quantum_mixup: bool = True,
    lambda_kta: float = 0.0,
) -> Dict[str, float]:
    """Train for one epoch; return dict of metrics.

    Args:
        epoch: Current epoch (1-indexed) for scheduled mixup warmup (Fix: Flaw 4).
        lambda_ent: Entanglement regularization coefficient (Section 12.5).
        s_target_frac: Fraction of theoretical max entropy for Goldilocks zone (default 0.70).
        qsam: Optional QSAM handler for flatness minimization.
        lambda_infonce: Optional Hilbert Space InfoNCE contrastive loss weight.
        grad_accum_steps: Number of micro-batches to accumulate gradients across.
        qsam_freq: Frequency interval for Q-SAM sharpness perturbation step.
        use_amp: Enable mixed-precision operations for classical tensor ops.
        post_quantum_mixup: If True, mix in expectation space AFTER the quantum circuit
            (density-operator convexity, Fix: FL-07). If False, mix pre-quantum rotation
            angles in Euclidean space (fast fallback, physically approximate).
        lambda_kta: Kernel-Target Alignment regularization weight (Fix: FL-09).
            Set > 0 to penalize KTA de-alignment during training.
    """
    model.train()
    running_task_loss = 0.0
    running_total_loss = 0.0
    all_preds, all_labels = [], []
    grad_norms_vqc:     List[float] = []
    grad_norms_adapter: List[float] = []
    grad_norms_head:    List[float] = []

    effective_alpha = _scheduled_mixup_alpha(epoch, alpha_target=mixup_alpha)

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Training", leave=False)):
        images = images.to(device, non_blocking=True)
        labels = labels.view(-1).long().to(device, non_blocking=True)

        if batch_idx % grad_accum_steps == 0:
            optimizer.zero_grad()

        # Apply quantum augmentations in latent space if model is hybrid
        if use_quantum_aug and isinstance(model, HybridCNNVQC):
            z = model.backbone(images, use_cached_features=True)
            # Fix E-02: True SU(2) Lie algebra phase perturbation
            # (record delta_phase for deterministic Q-SAM pass 2 synchronization)
            if phase_eps > 0.0:
                n_q = model.n_qubits
                z_perturbed = su2_phase_perturbation(z, n_qubits=n_q, epsilon=phase_eps)
                delta_phase = (z_perturbed - z).detach()  # Record the applied delta for Q-SAM replay (detached to prevent backward graph retention)
                z = z_perturbed
            else:
                delta_phase = None

            # Quantum Mixup with scheduled alpha (Fix: Flaw 4, Solution 5.3, FL-07)
            if effective_alpha > 0.0:
                idx = torch.randperm(z.size(0), device=device)
                # Solution 5.3: Protect minority Drusen (Class 2) from being diluted by majority CNV (Class 0)
                is_drusen = (labels == 2)
                if is_drusen.any():
                    drusen_indices = torch.where(is_drusen)[0]
                    if len(drusen_indices) > 1:
                        perm = torch.randperm(len(drusen_indices), device=device)
                        idx[is_drusen] = drusen_indices[perm]
                    else:
                        idx[is_drusen] = drusen_indices

                lam = float(np.random.beta(effective_alpha, effective_alpha))

                # Density matrix convexity violation: force post_quantum_mixup to prevent pre-quantum angle interpolation
                if not post_quantum_mixup:
                    post_quantum_mixup = True

                if post_quantum_mixup:
                    # FL-07 Fix: Post-Quantum Expectation-Space Mixup (density operator convexity)
                    # Mix AFTER the quantum circuit: exact for incoherent statistical mixtures.
                    # ρ_mix = λ ρ_A + (1-λ) ρ_B  ⟹  ⟨O⟩_mix = λ ⟨O⟩_A + (1-λ) ⟨O⟩_B
                    z_mix = z  # keep clean latents for entanglement/ortho regularization
                    z_float = z.float() if z.dtype != torch.float32 else z
                    exp_a = model.qlayer(z_float)
                    if isinstance(exp_a, (list, tuple)):
                        exp_a = torch.stack(exp_a, dim=-1)
                    exp_b = exp_a[idx]  # O(1) index slice avoiding redundant quantum re-execution (Fix: ENG-02)
                    exp, y_mix = expectation_quantum_mixup(
                        exp_a, exp_b, labels, labels[idx], n_classes, lam=lam
                    )
                    y_mix = y_mix.to(device)
                else:
                    # Legacy: Pre-quantum Euclidean angle mixup (fast, physically approximate)
                    z_mix, y_mix = quantum_mixup(
                        z, z[idx], labels, labels[idx], n_classes, lam=lam
                    )
                    y_mix = y_mix.to(device)
                    z_float = z_mix.float() if z_mix.dtype != torch.float32 else z_mix
                    exp = model.qlayer(z_float)
                    if isinstance(exp, (list, tuple)):
                        exp = torch.stack(exp, dim=-1)
            else:
                idx = None
                lam = 1.0
                # Clean data pass-through for warmup epochs
                z_mix = z
                y_mix = F.one_hot(labels, n_classes).float()
                z_float = z_mix.float() if z_mix.dtype != torch.float32 else z_mix
                exp = model.qlayer(z_float)
                if isinstance(exp, (list, tuple)):
                    exp = torch.stack(exp, dim=-1)

            logits = model.head(exp)
            task_loss = criterion(logits, y_mix)
            loss = task_loss

            # Solution 5.2: Active Hilbert Space Drusen-CNV Orthogonality Regularization
            ds_name = getattr(loader.dataset, "dataset_name", "")
            if n_classes == 4 and "oct" in ds_name.lower() and isinstance(model, HybridCNNVQC):
                is_d = (labels == 2)
                is_c = (labels == 0)
                if is_d.any() and is_c.any():
                    q_w = None
                    for n_p, p in model.named_parameters():
                        if "qlayer" in n_p and "weights" in n_p:
                            q_w = p
                            break
                    if q_w is not None:
                        d_idx = torch.where(is_d)[0][:8]
                        c_idx = torch.where(is_c)[0][:8]
                        sub_idx = torch.cat([d_idx, c_idx])
                        z_sub = z_mix[sub_idx]
                        labels_sub = labels[sub_idx]
                        try:
                            state_fn = model.get_state_qnode() if hasattr(model, "get_state_qnode") else None
                            if state_fn is not None:
                                states_tensor = state_fn(z_sub, q_w)
                                if states_tensor.dim() == 1:
                                    states_tensor = states_tensor.unsqueeze(0)
                            else:
                                n_wires = model.n_qubits
                                dev_sub = qml.device("default.qubit", wires=n_wires)
                                @qml.qnode(dev_sub, interface="torch")
                                def _get_state(inp, w):
                                    for q_i in range(n_wires):
                                        qml.U3(inp[q_i], inp[q_i + n_wires], inp[q_i + 2 * n_wires], wires=q_i)
                                    if getattr(model, "genome", None) is not None:
                                        _apply_ansatz(inp, w, n_wires, model.n_layers, reupload=model.reupload, genome=model.genome)
                                    else:
                                        qml.StronglyEntanglingLayers(w, wires=range(n_wires))
                                    return qml.state()
                                states_list = [_get_state(z_sub[k], q_w) for k in range(len(sub_idx))]
                                states_tensor = torch.stack(states_list)

                            ortho_loss = compute_drusen_cnv_orthogonality_loss(
                                states_tensor, labels_sub, drusen_class=2, cnv_class=0, eps_target=0.05
                            )
                            loss = loss + 0.15 * ortho_loss
                        except Exception:
                            pass

            # Solution 4: Retained Spherical Regularization & Readout Sensitivity Equalization
            # Re-scaled weights (0.02, 0.02, 0.01) with centered spherical log-det loss
            r_curr = getattr(model.backbone, "_current_r", None)
            if r_curr is not None:
                downstream_cov = compute_per_qubit_spherical_loss(r_curr)
                downstream_torque = compute_azimuthal_torque_loss(r_curr)
                readout_bal = compute_readout_balancing_loss(model)
                loss = loss + 0.02 * downstream_cov + 0.02 * downstream_torque + 0.01 * readout_bal

            # Entanglement regularization: Quadratic Goldilocks penalty (Section 12.5)
            # Penalize deviation from optimal intermediate entanglement S_target
            if lambda_ent > 0.0 and batch_idx == 0:
                n_q = model.n_qubits
                s_max = float(n_q // 2)
                s_target = s_target_frac * s_max
                s_current = _compute_entanglement_entropy_batch(model, z_mix, n_q)
                ent_penalty = lambda_ent * ((s_current - s_target) ** 2)
                loss = loss + ent_penalty

            # KTA Regularization: Penalize Hilbert-space de-alignment from labels (Fix: FL-09)
            # Computed on first batch only to limit computational overhead.
            if lambda_kta > 0.0 and batch_idx == 0 and isinstance(model, HybridCNNVQC):
                try:
                    n_q = model.n_qubits
                    q_w = None
                    for n_p, p in model.named_parameters():
                        if "qlayer" in n_p and "weights" in n_p:
                            q_w = p
                            break
                    if q_w is not None:
                        kta_sub_size = min(32, z_mix.size(0))
                        z_kta = z_mix[:kta_sub_size].detach()
                        y_kta = labels[:kta_sub_size]
                        kta_loss = _compute_differentiable_kta(
                            model, z_kta, y_kta, q_w, n_classes
                        )
                        # Maximize KTA → minimize negative KTA
                        loss = loss - lambda_kta * kta_loss
                except Exception:
                    pass  # Graceful fallback if state circuit unavailable

            # Add Hilbert Space InfoNCE contrastive regularisation if enabled
            if infonce_fn is not None and lambda_infonce > 0:
                try:
                    q_w = None
                    for n_p, p in model.named_parameters():
                        if "qlayer" in n_p and ("weights" in n_p or "params" in n_p):
                            q_w = p
                            break
                    if q_w is not None:
                        loss_infonce = infonce_fn(z_mix, labels, q_w, model=model)
                        loss = loss + lambda_infonce * loss_infonce
                except Exception:
                    pass  # Graceful fallback if ansatz circuit is incompatible

            preds = logits.argmax(dim=1)
        else:
            delta_phase = None
            idx = None
            lam = 1.0
            z_mix = None
            logits = model(images, use_cached_features=True)
            task_loss = criterion(logits, labels)
            loss = task_loss
            preds = logits.argmax(dim=1)

        if grad_accum_steps > 1:
            loss = loss / grad_accum_steps

        loss.backward()

        # Track per-component gradient norms BEFORE clipping (Fix: Flaw 8)
        if isinstance(model, HybridCNNVQC):
            vqc_norm = _get_component_grad_norm(model.qlayer.parameters())
            adapter_norm = _get_component_grad_norm(model.backbone.cq_adapter.parameters())
            head_norm = _get_component_grad_norm(model.head.parameters())
            grad_norms_vqc.append(vqc_norm)
            grad_norms_adapter.append(adapter_norm)
            grad_norms_head.append(head_norm)

            # Real-Time Barren Plateau Watchdog (Skolik et al., 2021; Grant et al., 2019)
            # Only synchronize with host CPU on watchdog inspection batches (batch_idx % 25 == 0) to eliminate 96% of CUDA stalls
            if batch_idx % 25 == 0:
                vqc_val = float(vqc_norm.item()) if torch.is_tensor(vqc_norm) else float(vqc_norm)
                if vqc_val > 0.0:
                    log_vqc = math.log10(vqc_val)
                    if log_vqc < -4.0:
                        print(f"\n[WATCHDOG WARNING] Batch {batch_idx}: VQC Gradient Norm log10||∇_θ L||_2 = {log_vqc:.2f} < -4.0 (Warning threshold)")
                    if log_vqc < -5.0:
                        print(f"\n[CRITICAL WATCHDOG] Batch {batch_idx}: Barren Plateau collapse detected (log10 < -5.0). Gradients vanishing exponentially.")
        else:
            grad_norms_vqc.append(
                _get_component_grad_norm(model.parameters())
            )

        is_accum_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(loader))

        if is_accum_step:
            should_qsam = (qsam is not None) and (batch_idx % qsam_freq == 0)
            if should_qsam:
                # Ascend to worst-case neighborhood point w + e_w and clear ascent gradients
                qsam.first_step(zero_grad=True)

                # Sharpness descent step: re-evaluate with fresh computation graph
                if use_quantum_aug and isinstance(model, HybridCNNVQC):
                    z2 = model.backbone(images, use_cached_features=True)
                    if delta_phase is not None:
                        z2 = su2_phase_perturbation(z2, n_qubits=model.n_qubits, delta=delta_phase)
                    if effective_alpha > 0.0 and idx is not None:
                        if post_quantum_mixup:
                            # FL-07: Post-quantum mixup consistency for QSAM pass 2
                            z2_float = z2.float() if z2.dtype != torch.float32 else z2
                            exp2_a = model.qlayer(z2_float)
                            if isinstance(exp2_a, (list, tuple)):
                                exp2_a = torch.stack(exp2_a, dim=-1)
                            exp2_b = model.qlayer(z2_float[idx])
                            if isinstance(exp2_b, (list, tuple)):
                                exp2_b = torch.stack(exp2_b, dim=-1)
                            exp2, _ = expectation_quantum_mixup(
                                exp2_a, exp2_b, labels, labels[idx], n_classes, lam=lam
                            )
                            z_mix2 = z2  # for QSAM step_with_latent
                        else:
                            z_mix2, _ = quantum_mixup(
                                z2, z2[idx], labels, labels[idx], n_classes, lam=lam
                            )
                            exp2 = model.qlayer(z_mix2.float() if z_mix2.dtype != torch.float32 else z_mix2)
                            if isinstance(exp2, (list, tuple)):
                                exp2 = torch.stack(exp2, dim=-1)
                    else:
                        z_mix2 = z2
                        exp2 = model.qlayer(z_mix2.float() if z_mix2.dtype != torch.float32 else z_mix2)
                        if isinstance(exp2, (list, tuple)):
                            exp2 = torch.stack(exp2, dim=-1)
                    logits2 = model.head(exp2)
                    loss2 = criterion(logits2, y_mix)
                else:
                    z_mix2 = None
                    logits2 = model(images, use_cached_features=True)
                    loss2 = criterion(logits2, labels)

                if grad_accum_steps > 1:
                    loss2 = loss2 / grad_accum_steps
                loss2.backward()
                trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
                if clip_grad > 0:
                    nn.utils.clip_grad_norm_(trainable_params, clip_grad)
                qsam.second_step(zero_grad=True, latent_batch=z_mix2 if use_quantum_aug and isinstance(model, HybridCNNVQC) else None)
            else:
                trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
                if clip_grad > 0:
                    nn.utils.clip_grad_norm_(trainable_params, clip_grad)
                if hasattr(optimizer, "step_with_latent"):
                    optimizer.step_with_latent(latent_batch=z_mix if use_quantum_aug and isinstance(model, HybridCNNVQC) else None)
                else:
                    optimizer.step()

        raw_loss_det = (loss.detach() * grad_accum_steps) if grad_accum_steps > 1 else loss.detach()
        running_task_loss += task_loss.detach() * images.size(0)
        running_total_loss += raw_loss_det * images.size(0)
        all_preds.append(preds.cpu())
        # Dominant-target label accounting during mixup (Fix: Defect 5 — Epoch 6 accuracy crash)
        # When mixup is active, evaluate argmax predictions against the dominant (majority weight) label
        # This prevents the artificial ~22.5% accuracy drop caused by evaluating mixed predictions
        # against unmixed hard labels (Zhang et al., ICLR 2018: "mixup: Beyond Empirical Risk Minimization")
        if effective_alpha > 0.0 and idx is not None and isinstance(model, HybridCNNVQC):
            dominant_labels = torch.where(
                torch.tensor(lam >= 0.5, device=labels.device).expand_as(labels),
                labels,
                labels[idx],
            )
            all_labels.append(dominant_labels.cpu())
        else:
            all_labels.append(labels.cpu())

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    epoch_task_loss = float(running_task_loss.item() if torch.is_tensor(running_task_loss) else running_task_loss) / len(all_labels)
    epoch_total_loss = float(running_total_loss.item() if torch.is_tensor(running_total_loss) else running_total_loss) / len(all_labels)
    epoch_acc  = accuracy_score(all_labels, all_preds)

    # Convert grad norms to floats asynchronously at epoch boundary
    grad_norms_vqc = [float(gn.item()) if torch.is_tensor(gn) else float(gn) for gn in grad_norms_vqc]
    grad_norms_adapter = [float(gn.item()) if torch.is_tensor(gn) else float(gn) for gn in grad_norms_adapter]
    grad_norms_head = [float(gn.item()) if torch.is_tensor(gn) else float(gn) for gn in grad_norms_head]

    return {
        "loss":               epoch_task_loss,
        "total_loss":         epoch_total_loss,
        "accuracy":           epoch_acc,
        "grad_norms":         grad_norms_vqc,
        "grad_norms_vqc":     grad_norms_vqc,
        "grad_norms_adapter": grad_norms_adapter,
        "grad_norms_head":    grad_norms_head,
        "effective_mixup_alpha": effective_alpha,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    n_classes: int,
    max_samples: int = 1000,
    cost_matrix: Optional[torch.Tensor] = None,
) -> Dict:
    """Evaluate model; return dict with loss, accuracy, per-sample info.

    Args:
        max_samples: Maximum number of latent/expectation samples to store
                     to prevent RAM bloat. Default 1000.
        cost_matrix: Optional clinical loss matrix C (shape [K, K]) for Bayes risk minimization.
    """
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs, all_logits_list = [], [], [], []
    all_latents, all_exps = [], []
    total_samples = 0

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.view(-1).long().to(device, non_blocking=True)

        if hasattr(model, "forward_with_latent"):
            logits, z, exp = model.forward_with_latent(images, use_cached_features=True)
            # Only store latents/expectations up to max_samples
            if total_samples < max_samples:
                remaining = max_samples - total_samples
                batch_size = min(z.size(0), remaining)
                all_latents.append(z[:batch_size].cpu())
                all_exps.append(exp[:batch_size].cpu())
        else:
            try:
                logits = model(images, use_cached_features=True)
            except TypeError:
                logits = model(images)

        loss = criterion(logits, labels)
        probs = F.softmax(logits, dim=1)

        running_loss += loss.detach() * images.size(0)

        if cost_matrix is not None and logits.size(1) == cost_matrix.size(0):
            cm = cost_matrix.to(logits.device)
            # cm[action, true_class] -> expected risk = C @ p.
            risk = torch.matmul(probs, cm.transpose(0, 1))
            preds = risk.argmin(dim=1).cpu()
        else:
            preds = logits.argmax(dim=1).cpu()

        all_preds.append(preds)
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())
        all_logits_list.append(logits.detach().cpu())
        total_samples += images.size(0)

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs  = torch.cat(all_probs).numpy()
    # Compute validation AUROC for best checkpoint tracking
    val_auc = float(accuracy_score(all_labels, all_preds))
    try:
        unique_labels = np.unique(all_labels)
        if len(unique_labels) >= 2:
            if n_classes == 2:
                p_pos = all_probs[:, 1] if (all_probs.ndim == 2 and all_probs.shape[1] > 1) else all_probs.squeeze()
                val_auc = float(roc_auc_score(all_labels, p_pos))
            else:
                val_auc = float(roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro"))
    except Exception:
        pass

    epoch_loss = float(running_loss.item() if torch.is_tensor(running_loss) else running_loss) / max(total_samples, 1)

    all_logits_cat = torch.cat(all_logits_list)  # (N, C) — raw unnormalized logits
    result = {
        "loss":     epoch_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "auroc":    val_auc,
        "preds":    all_preds,
        "labels":   all_labels,
        "probs":    all_probs,
        "logits":   all_logits_cat,  # raw logits for temperature scaling (no double-log)
    }
    result["balanced_accuracy"] = float(balanced_accuracy_score(all_labels, all_preds))
    result["macro_f1"] = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    try:
        if n_classes == 2:
            result["auprc"] = float(average_precision_score(all_labels, all_probs[:, 1]))
        else:
            result["auprc"] = float(
                average_precision_score(
                    label_binarize(all_labels, classes=np.arange(n_classes)),
                    all_probs,
                    average="macro",
                )
            )
    except ValueError:
        result["auprc"] = float("nan")
    if all_latents:
        result["latents"] = torch.cat(all_latents).numpy()
        result["latent_labels"] = all_labels[:len(result["latents"])]
    if all_exps:
        result["expectations"] = torch.cat(all_exps).numpy()

    # Store trained quantum circuit weights if present (Fix: Flaw F-02)
    if hasattr(model, "named_parameters"):
        for name, param in model.named_parameters():
            if "qlayer" in name and "weights" in name:
                result["trained_q_weights"] = param.detach().cpu().numpy()
                break

    return result


def evaluate_training_epoch_metrics(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    n_classes: int,
    max_samples: int = 1000,
) -> Dict[str, float]:
    """Evaluate task metrics at an epoch boundary with no train-time augmentation.

    This is intentionally separate from ``train_one_epoch``: online mixup,
    quantum perturbations, dropout, and balanced sampling are training
    procedures, not comparable measurements of the learned task function.
    """
    result = evaluate(model, loader, criterion, device, n_classes, max_samples=max_samples)
    return {
        "loss": float(result["loss"]),
        "accuracy": float(result["accuracy"]),
        "balanced_accuracy": float(result["balanced_accuracy"]),
        "macro_f1": float(result["macro_f1"]),
        "auprc": float(result["auprc"]),
    }
# ═══════════════════════════════════════════════════════════════════════════
# 3.  ADVANCED QML OPTIMIZATION & CONTINUAL LEARNING (Decoupled modules)
# ═══════════════════════════════════════════════════════════════════════════
try:
    from training_utils import (
        _is_better_checkpoint,
        QSAM,
        QSWAHandler,
        HybridQMLOptimizer,
        _build_hybrid_optimizer,
        compute_clinical_metrics,
        McNemarResult,
        mcnemars_test,
        mc_dropout_predict,
    )
    from continual import (
        compute_diagonal_qfim,
        compute_fisher_diagonal,
        qewc_penalty,
        compute_expressibility,
        compute_meyer_wallach,
    )
except ImportError:
    from qml_models.training_utils import (
        _is_better_checkpoint,
        QSAM,
        QSWAHandler,
        HybridQMLOptimizer,
        _build_hybrid_optimizer,
        compute_clinical_metrics,
        McNemarResult,
        mcnemars_test,
        mc_dropout_predict,
    )
    from qml_models.continual import (
        compute_diagonal_qfim,
        compute_fisher_diagonal,
        qewc_penalty,
        compute_expressibility,
        compute_meyer_wallach,
    )



# ═══════════════════════════════════════════════════════════════════════════
# 7.  DIAGNOSTIC VISUALIZATION SUITE (Decoupled to qml_models/plots.py)
# ═══════════════════════════════════════════════════════════════════════════
try:
    from plots import (
        _save,
        compute_entanglement_entropy_epoch,
        compute_kernel_target_alignment,
        compute_qfim_spectrum,
        evaluate_noise_robustness,
        generate_all_diagnostic_plots,
        plot_ansatz_capacity,
        plot_bloch_sphere,
        plot_calibration_curve,
        plot_class_conditional_reliability,
        plot_conformal_prediction_efficiency,
        plot_confusion_matrices,
        plot_convergence_curves,
        plot_cost_utility_curve,
        plot_cq_pretrain_bloch_evolution,
        plot_cq_pretrain_geometry_dynamics,
        plot_cq_pretrain_loss_curves,
        plot_cq_pretrain_qubit_covariance_spectra,
        plot_decision_boundary_comparison,
        plot_effective_quantum_dimension,
        plot_entanglement_entropy,
        plot_epoch_time_breakdown,
        plot_feature_importance_heatmap,
        plot_gradient_norms,
        plot_haar_fidelity_expressibility,
        plot_kta_evolution,
        plot_latent_tsne,
        plot_learning_rate_sensitivity,
        plot_loss_landscape,
        plot_mollweide_bloch_density,
        plot_noise_channel_robustness,
        plot_parameter_distribution,
        plot_pauli_z_expectations,
        plot_per_class_metrics,
        plot_pr_curves,
        plot_qewc_retention,
        plot_qfim_spectrum,
        plot_qnas_admissibility,
        plot_quantum_bhattacharyya_distance,
        plot_quantum_classical_gradient_ratio,
        plot_quantum_kernel_gram,
        plot_quantum_state_purity_profile,
        plot_roc_curves,
        plot_uncertainty_histogram,
        plot_xai_gradcam_quantum,
        plot_zne_curves,
        plot_progressive_learning_trajectory,
        plot_kfold_metric_distributions,
    )
except ImportError:
    from qml_models.plots import (
        _save,
        compute_entanglement_entropy_epoch,
        compute_kernel_target_alignment,
        compute_qfim_spectrum,
        evaluate_noise_robustness,
        generate_all_diagnostic_plots,
        plot_ansatz_capacity,
        plot_bloch_sphere,
        plot_calibration_curve,
        plot_class_conditional_reliability,
        plot_conformal_prediction_efficiency,
        plot_confusion_matrices,
        plot_convergence_curves,
        plot_cost_utility_curve,
        plot_cq_pretrain_bloch_evolution,
        plot_cq_pretrain_geometry_dynamics,
        plot_cq_pretrain_loss_curves,
        plot_cq_pretrain_qubit_covariance_spectra,
        plot_decision_boundary_comparison,
        plot_effective_quantum_dimension,
        plot_entanglement_entropy,
        plot_epoch_time_breakdown,
        plot_feature_importance_heatmap,
        plot_gradient_norms,
        plot_haar_fidelity_expressibility,
        plot_kta_evolution,
        plot_latent_tsne,
        plot_learning_rate_sensitivity,
        plot_loss_landscape,
        plot_mollweide_bloch_density,
        plot_pauli_z_expectations,
        plot_per_class_metrics,
        plot_pr_curves,
        plot_qewc_retention,
        plot_qfim_spectrum,
        plot_qnas_admissibility,
        plot_quantum_bhattacharyya_distance,
        plot_quantum_classical_gradient_ratio,
        plot_quantum_kernel_gram,
        plot_quantum_state_purity_profile,
        plot_roc_curves,
        plot_uncertainty_histogram,
        plot_xai_gradcam_quantum,
        plot_zne_curves,
        plot_progressive_learning_trajectory,
        plot_kfold_metric_distributions,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 7.5 PRE-TRAIN CQ ADAPTER & GEOMETRIC LOSSES (Imported from adapter.py)
# ═══════════════════════════════════════════════════════════════════════════
# Note: pretrain_cq_adapter, _decoupled_supcon_loss, compute_per_qubit_spherical_loss,
# compute_hyperspherical_uniformity_loss, compute_azimuthal_torque_loss,
# compute_readout_balancing_loss, and compute_fibonacci_coverage are imported from adapter.py


# ═══════════════════════════════════════════════════════════════════════════
# 7.6  CALIBRATION & THRESHOLD OPTIMIZATION (Decoupled to qml_models.calibration)
# ═══════════════════════════════════════════════════════════════════════════
try:
    from calibration import (
        _collect_val_logits,
        optimize_temperature_scaling,
        QuantumBetaCalibrator,
        QuantumMultiClassCalibrator,
        optimize_multiclass_cost_sensitive_thresholds,
        predict_with_cost_matrix,
        build_mondrian_conformal_artifact,
        split_calibration_indices,
        optimize_decision_threshold,
    )
except ImportError:
    from qml_models.calibration import (
        _collect_val_logits,
        optimize_temperature_scaling,
        QuantumBetaCalibrator,
        QuantumMultiClassCalibrator,
        optimize_multiclass_cost_sensitive_thresholds,
        predict_with_cost_matrix,
        build_mondrian_conformal_artifact,
        split_calibration_indices,
        optimize_decision_threshold,
    )



# ═══════════════════════════════════════════════════════════════════════════
# 7.7 PROGRESSIVE LAYERWISE TRAINING ORCHESTRATOR (Skolik et al., 2021)
# ═══════════════════════════════════════════════════════════════════════════
def train_progressive_hybrid(
    hybrid: "HybridCNNVQC",
    baseline: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
    n_classes: int,
    depth_schedule: List[int] = [2, 4, 6],
    epochs_per_stage: int = 10,
    finetune_epochs: int = 5,
    q_init_std: float = 0.01,
    freeze_mode: str = "hard",
    use_qng: bool = False,
    mixup_alpha: float = 0.4,
    phase_eps: float = 0.05,
    lambda_ent: float = 0.05,
    s_target_frac: float = 0.70,
    use_qsam: bool = False,
    use_qswa: bool = False,
    lambda_infonce: float = 0.0,
    lr_baseline: float = 1e-4,
    n_qubits: int = 4,
    out_dir: str = "outputs",
    entropy_every_epoch: bool = True,
    entropy_interval: int = 1,
    entropy_samples: int = 32,
    grad_accum_steps: int = 1,
    qsam_freq: int = 1,
    use_amp: bool = False,
    post_quantum_mixup: bool = True,
    lambda_kta: float = 0.0,
) -> Dict:
    """Progressive Layerwise Training Orchestrator (Skolik et al., 2021; Grant et al., 2019).

    Executes multi-stage depth growth (e.g. L=2 -> 4 -> 6) to prevent unitary 2-design collapse.
    Inherits trained parameters and initializes new layers near identity (sigma=q_init_std).
    """
    total_total_epochs = len(depth_schedule) * epochs_per_stage + finetune_epochs
    print(f"\n{'='*70}")
    print(f"  PROGRESSIVE LAYERWISE TRAINING PIPELINE (Skolik et al., 2021)")
    print(f"  Depth Schedule       : {depth_schedule}")
    print(f"  Epochs Per Stage     : {epochs_per_stage} (Total stages={len(depth_schedule)})")
    print(f"  Harmonization Epochs : {finetune_epochs}")
    print(f"  Total Scheduled Eps  : {total_total_epochs}")
    print(f"  Freeze Mode          : {freeze_mode.upper()} (Zero-grad hook on previous layers)")
    print(f"  Identity Init Std    : {q_init_std} (Grant et al., 2019)")
    print(f"  Use QNG              : {use_qng}")
    print(f"{'='*70}\n")

    # 1. Ensure initial depth matches first stage
    init_depth = depth_schedule[0]
    if hybrid.n_layers != init_depth:
        print(f"[progressive] Setting initial hybrid depth to L={init_depth} (was L={hybrid.n_layers})")
        hybrid.n_layers = init_depth
        qnode, weight_shapes, hybrid.qdev, hybrid.qdev_name, hybrid.n_meas = create_vqc_qnode(
            hybrid.n_qubits, init_depth, diff_method=hybrid.diff_method, reupload=hybrid.reupload,
            genome=hybrid.genome, qdevice=hybrid.qdevice,
            dual_basis_only=getattr(hybrid, "dual_basis_only", False),
            use_qaf_attention=getattr(hybrid, "use_qaf_attention", False),
        )
        init_method = {"weights": partial(torch.nn.init.normal_, mean=0.0, std=q_init_std)}
        hybrid.qlayer = QuantumDropoutTorchLayer(qnode, weight_shapes, init_method=init_method, p_drop=hybrid.p_drop).to(device)

    # Initial optimizer setup
    opt_hybrid = _build_hybrid_optimizer(hybrid, use_qng=use_qng, lr_classical=2e-4, lr_quantum=5e-3)
    opt_baseline = torch.optim.Adam(filter(lambda p: p.requires_grad, baseline.parameters()), lr=lr_baseline)

    scheduler_h = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_hybrid if not use_qng else opt_hybrid.opt_classical, T_0=epochs_per_stage, T_mult=1, eta_min=1e-6
    )
    scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_baseline, T_max=max(1, total_total_epochs))

    qsam_h = QSAM(opt_hybrid, hybrid, rho=0.05) if use_qsam else None
    infonce_fn = QuantumInfoNCELoss(n_qubits, tau=0.07).to(device) if lambda_infonce > 0 else None

    hybrid_hist = {"train_loss": [], "train_total_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_balanced_accuracy": [], "val_balanced_accuracy": [], "train_macro_f1": [], "val_macro_f1": [], "train_auprc": [], "val_auprc": []}
    baseline_hist = {"train_loss": [], "train_total_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_balanced_accuracy": [], "val_balanced_accuracy": [], "train_macro_f1": [], "val_macro_f1": [], "train_auprc": [], "val_auprc": []}
    all_grad_norms_vqc = []
    all_grad_norms_adapter = []
    all_grad_norms_head = []
    entropy_history = []
    kta_history = []
    epoch_times_hybrid = []
    epoch_times_baseline = []
    stage_transitions = []
    global_epoch = 0

    best_h_auc = -1.0
    best_h_loss = float("inf")
    best_h_epoch = 0
    best_h_state = None

    best_b_auc = -1.0
    best_b_loss = float("inf")
    best_b_epoch = 0
    best_b_state = None

    best_kta = -1.0
    best_kta_epoch = 0
    best_kta_state = None
    kta_decay_count = 0

    for stage_idx, target_depth in enumerate(depth_schedule):
        if stage_idx > 0:
            prev_depth = depth_schedule[stage_idx - 1]
            freeze_prev = (freeze_mode == "hard")
            print(f"\n{'#'*70}")
            print(f"  PROGRESSIVE STAGE TRANSITION: Expanding Depth L={prev_depth} ──► L={target_depth}")
            print(f"  Identity Initialization: θ_new ~ N(0, {q_init_std}^2) (Grant et al., 2019)")
            print(f"  Previous Layer Freezing: {freeze_mode.upper()} (Skolik et al., 2021)")
            print(f"{'#'*70}\n")

            hybrid.grow_layers(new_depth=target_depth, init_std=q_init_std, freeze_previous=freeze_prev)
            # Reset KTA peak tracking for the new depth stage
            best_kta = -1.0
            best_kta_epoch = 0
            best_kta_state = None
            kta_decay_count = 0

            opt_hybrid = _build_hybrid_optimizer(
                hybrid, use_qng=use_qng, lr_classical=2e-4, lr_quantum=5e-3, old_optimizer=opt_hybrid
            )
            scheduler_h = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt_hybrid if not use_qng else opt_hybrid.opt_classical, T_0=epochs_per_stage, T_mult=1, eta_min=1e-6
            )
            qsam_h = QSAM(opt_hybrid, hybrid, rho=0.05) if use_qsam else None
            stage_transitions.append({
                "epoch": global_epoch + 1,
                "from_depth": prev_depth,
                "to_depth": target_depth,
                "description": f"Stage {stage_idx+1}: L={target_depth} ({freeze_mode})",
            })
        else:
            stage_transitions.append({
                "epoch": 1,
                "from_depth": 0,
                "to_depth": target_depth,
                "description": f"Stage 1: Base L={target_depth}",
            })

        # Train epochs for this stage
        for ep in range(1, epochs_per_stage + 1):
            global_epoch += 1
            if hasattr(criterion, "update_epoch"):
                criterion.update_epoch(global_epoch)
            print(f"\n── [Stage {stage_idx+1}/{len(depth_schedule)}: L={target_depth}] Epoch {global_epoch}/{total_total_epochs} (Stage Ep {ep}/{epochs_per_stage}) {'─'*25}")

            # --- Hybrid ---
            t0 = time.time()
            train_res_h = train_one_epoch(
                hybrid, train_loader, criterion, opt_hybrid, device, n_classes,
                use_quantum_aug=True, mixup_alpha=mixup_alpha, phase_eps=phase_eps,
                epoch=global_epoch,
                lambda_ent=lambda_ent,
                s_target_frac=s_target_frac,
                qsam=qsam_h,
                lambda_infonce=lambda_infonce,
                infonce_fn=infonce_fn,
                grad_accum_steps=grad_accum_steps,
                qsam_freq=qsam_freq,
                use_amp=use_amp,
                post_quantum_mixup=post_quantum_mixup,
                lambda_kta=lambda_kta,
            )
            t_hybrid = time.time() - t0
            epoch_times_hybrid.append(t_hybrid)
            val_res_h = evaluate(hybrid, val_loader, criterion, device, n_classes, max_samples=1000)
            train_eval_h = evaluate_training_epoch_metrics(hybrid, train_loader, criterion, device, n_classes, max_samples=1000)

            hybrid_hist["train_loss"].append(train_res_h["loss"])
            hybrid_hist["train_total_loss"].append(train_res_h.get("total_loss", train_res_h["loss"]))
            hybrid_hist["val_loss"].append(val_res_h["loss"])
            hybrid_hist["train_acc"].append(train_eval_h["accuracy"])
            hybrid_hist["val_acc"].append(val_res_h["accuracy"])
            for key in ("balanced_accuracy", "macro_f1", "auprc"):
                hybrid_hist[f"train_{key}"].append(train_eval_h[key])
                hybrid_hist[f"val_{key}"].append(val_res_h[key])
            all_grad_norms_vqc.extend(train_res_h.get("grad_norms_vqc", []))
            all_grad_norms_adapter.extend(train_res_h.get("grad_norms_adapter", []))
            all_grad_norms_head.extend(train_res_h.get("grad_norms_head", []))

            # --- Baseline ---
            t0 = time.time()
            train_res_b = train_one_epoch(
                baseline, train_loader, criterion, opt_baseline, device, n_classes,
                use_quantum_aug=False, epoch=global_epoch,
                grad_accum_steps=grad_accum_steps,
                use_amp=use_amp,
            )
            t_baseline = time.time() - t0
            epoch_times_baseline.append(t_baseline)
            val_res_b = evaluate(baseline, val_loader, criterion, device, n_classes, max_samples=1000)
            train_eval_b = evaluate_training_epoch_metrics(baseline, train_loader, criterion, device, n_classes, max_samples=1000)

            # Track best baseline model across all stages
            val_b_auc = val_res_b.get("auroc", val_res_b["accuracy"])
            val_b_loss = val_res_b["loss"]
            if _is_better_checkpoint(val_b_auc, val_b_loss, best_b_auc, best_b_loss):
                best_b_auc = val_b_auc
                best_b_loss = val_b_loss
                best_b_epoch = global_epoch
                best_b_state = copy.deepcopy(baseline.state_dict())
                print(f"      ★ New best Baseline checkpoint (Val AUC: {best_b_auc:.4f}, Val Loss: {best_b_loss:.4f})")

            # Track best hybrid model on final target depth stage
            if target_depth == depth_schedule[-1]:
                val_h_auc = val_res_h.get("auroc", val_res_h["accuracy"])
                val_h_loss = val_res_h["loss"]
                if _is_better_checkpoint(val_h_auc, val_h_loss, best_h_auc, best_h_loss):
                    best_h_auc = val_h_auc
                    best_h_loss = val_h_loss
                    best_h_epoch = global_epoch
                    best_h_state = copy.deepcopy(hybrid.state_dict())
                    print(f"      ★ New best Hybrid checkpoint (Val AUC: {best_h_auc:.4f}, Val Loss: {best_h_loss:.4f})")

            baseline_hist["train_loss"].append(train_res_b["loss"])
            baseline_hist["train_total_loss"].append(train_res_b.get("total_loss", train_res_b["loss"]))
            baseline_hist["val_loss"].append(val_res_b["loss"])
            baseline_hist["train_acc"].append(train_eval_b["accuracy"])
            baseline_hist["val_acc"].append(val_res_b["accuracy"])
            for key in ("balanced_accuracy", "macro_f1", "auprc"):
                baseline_hist[f"train_{key}"].append(train_eval_b[key])
                baseline_hist[f"val_{key}"].append(val_res_b[key])

            # Entanglement entropy & KTA (vectorized GPU state evaluation)
            active_nq = getattr(hybrid, "actual_n_qubits", getattr(hybrid, "n_qubits", n_qubits))
            should_compute_entropy = (
                entropy_every_epoch
                or (ep % entropy_interval == 0)
                or ep == 1
                or ep == epochs_per_stage
                or len(entropy_history) == 0
            )
            if should_compute_entropy:
                ee = compute_entanglement_entropy_epoch(
                    hybrid, val_loader, device, active_nq, n_samples=entropy_samples
                )
                kta = compute_kernel_target_alignment(
                    hybrid, val_loader, device, active_nq, n_samples=min(entropy_samples * 2, 64)
                )
            else:
                ee = entropy_history[-1] if entropy_history else 0.0
                kta = kta_history[-1] if kta_history else 0.0
            entropy_history.append(ee)
            kta_history.append(kta)

            # KTA Peak Tracking (Fix: FL-09)
            if kta > best_kta:
                best_kta = kta
                best_kta_epoch = global_epoch
                best_kta_state = copy.deepcopy(hybrid.state_dict())

            print(f"   Hybrid  → loss={train_res_h['loss']:.4f} (tot={train_res_h.get('total_loss', train_res_h['loss']):.4f})"
                  f"  acc={train_res_h['accuracy']:.4f}"
                  f"  val_loss={val_res_h['loss']:.4f}  val_acc={val_res_h['accuracy']:.4f}"
                  f"  [{t_hybrid:.1f}s]")
            print(f"   Baseline→ loss={train_res_b['loss']:.4f}  acc={train_res_b['accuracy']:.4f}"
                  f"  val_loss={val_res_b['loss']:.4f}  val_acc={val_res_b['accuracy']:.4f}"
                  f"  [{t_baseline:.1f}s]")
            print(f"   Entanglement Entropy: S(ρ_A)={ee:.4f}  |  KTA: S(K,Y)={kta:.4f}")

            if not use_qng:
                scheduler_h.step(ep)
            scheduler_b.step()

    # Stage 4: Global Harmonization Phase
    if finetune_epochs > 0:
        final_d = depth_schedule[-1]
        print(f"\n{'#'*70}")
        print(f"  PROGRESSIVE HARMONIZATION PHASE: Unfreezing All L={final_d} Layers")
        print(f"  Joint Co-adaptation with Low Learning Rate (lr_q = 5e-4, lr_c = 5e-5)")
        print(f"{'#'*70}\n")

        hybrid.unfreeze_all_quantum_layers()
        # Reset KTA peak tracking for harmonization phase
        best_kta = -1.0
        best_kta_epoch = 0
        best_kta_state = None
        kta_decay_count = 0

        opt_hybrid = _build_hybrid_optimizer(
            hybrid, use_qng=use_qng, lr_classical=5e-5, lr_quantum=5e-4, lr_head=2e-4, old_optimizer=opt_hybrid
        )
        scheduler_h = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_hybrid if not use_qng else opt_hybrid.opt_classical, T_max=finetune_epochs, eta_min=1e-6
        )
        qsam_h = QSAM(opt_hybrid, hybrid, rho=0.05) if use_qsam else None
        stage_transitions.append({
            "epoch": global_epoch + 1,
            "from_depth": final_d,
            "to_depth": final_d,
            "description": f"Harmonization (All L={final_d} Unfrozen)",
        })

        for ep in range(1, finetune_epochs + 1):
            global_epoch += 1
            if hasattr(criterion, "update_epoch"):
                criterion.update_epoch(global_epoch)
            print(f"\n── [Harmonization: All L={final_d} Unfrozen] Epoch {global_epoch}/{total_total_epochs} (Harmonize Ep {ep}/{finetune_epochs}) {'─'*20}")

            t0 = time.time()
            train_res_h = train_one_epoch(
                hybrid, train_loader, criterion, opt_hybrid, device, n_classes,
                use_quantum_aug=True, mixup_alpha=mixup_alpha, phase_eps=phase_eps,
                epoch=global_epoch,
                lambda_ent=lambda_ent,
                s_target_frac=s_target_frac,
                qsam=qsam_h,
                lambda_infonce=lambda_infonce,
                infonce_fn=infonce_fn,
                grad_accum_steps=grad_accum_steps,
                qsam_freq=qsam_freq,
                use_amp=use_amp,
                post_quantum_mixup=post_quantum_mixup,
                lambda_kta=lambda_kta,
            )
            t_hybrid = time.time() - t0
            epoch_times_hybrid.append(t_hybrid)
            val_res_h = evaluate(hybrid, val_loader, criterion, device, n_classes, max_samples=1000)

            hybrid_hist["train_loss"].append(train_res_h["loss"])
            hybrid_hist["train_total_loss"].append(train_res_h.get("total_loss", train_res_h["loss"]))
            hybrid_hist["val_loss"].append(val_res_h["loss"])
            hybrid_hist["train_acc"].append(train_res_h["accuracy"])
            hybrid_hist["val_acc"].append(val_res_h["accuracy"])
            all_grad_norms_vqc.extend(train_res_h.get("grad_norms_vqc", []))
            all_grad_norms_adapter.extend(train_res_h.get("grad_norms_adapter", []))
            all_grad_norms_head.extend(train_res_h.get("grad_norms_head", []))

            # Baseline
            t0 = time.time()
            train_res_b = train_one_epoch(
                baseline, train_loader, criterion, opt_baseline, device, n_classes,
                use_quantum_aug=False, epoch=global_epoch,
                grad_accum_steps=grad_accum_steps,
                use_amp=use_amp,
            )
            t_baseline = time.time() - t0
            epoch_times_baseline.append(t_baseline)
            val_res_b = evaluate(baseline, val_loader, criterion, device, n_classes, max_samples=1000)

            # Track best models during Harmonization
            val_h_auc = val_res_h.get("auroc", val_res_h["accuracy"])
            val_h_loss = val_res_h["loss"]
            if _is_better_checkpoint(val_h_auc, val_h_loss, best_h_auc, best_h_loss):
                best_h_auc = val_h_auc
                best_h_loss = val_h_loss
                best_h_epoch = global_epoch
                best_h_state = copy.deepcopy(hybrid.state_dict())
                print(f"      ★ New best Hybrid checkpoint (Val AUC: {best_h_auc:.4f}, Val Loss: {best_h_loss:.4f})")

            val_b_auc = val_res_b.get("auroc", val_res_b["accuracy"])
            val_b_loss = val_res_b["loss"]
            if _is_better_checkpoint(val_b_auc, val_b_loss, best_b_auc, best_b_loss):
                best_b_auc = val_b_auc
                best_b_loss = val_b_loss
                best_b_epoch = global_epoch
                best_b_state = copy.deepcopy(baseline.state_dict())
                print(f"      ★ New best Baseline checkpoint (Val AUC: {best_b_auc:.4f}, Val Loss: {best_b_loss:.4f})")

            baseline_hist["train_loss"].append(train_res_b["loss"])
            baseline_hist["train_total_loss"].append(train_res_b.get("total_loss", train_res_b["loss"]))
            baseline_hist["val_loss"].append(val_res_b["loss"])
            baseline_hist["train_acc"].append(train_res_b["accuracy"])
            baseline_hist["val_acc"].append(val_res_b["accuracy"])

            active_nq = getattr(hybrid, "actual_n_qubits", getattr(hybrid, "n_qubits", n_qubits))
            ee = compute_entanglement_entropy_epoch(hybrid, val_loader, device, active_nq, n_samples=entropy_samples)
            entropy_history.append(ee)
            kta = compute_kernel_target_alignment(hybrid, val_loader, device, active_nq, n_samples=min(entropy_samples * 2, 64))
            kta_history.append(kta)

            # KTA Peak & Decay Guard (Fix: FL-09)
            if kta > best_kta:
                best_kta = kta
                best_kta_epoch = global_epoch
                best_kta_state = copy.deepcopy(hybrid.state_dict())
                kta_decay_count = 0
            elif best_kta > 0.0 and (best_kta - kta) / best_kta > 0.15:
                kta_decay_count += 1
                if kta_decay_count >= 5 and best_kta_state is not None:
                    # Verify shape compatibility to prevent cross-stage size mismatch
                    can_restore = True
                    if "qlayer.weights" in best_kta_state:
                        cur_w = getattr(getattr(hybrid, "qlayer", None), "weights", None)
                        if cur_w is not None and cur_w.shape != best_kta_state["qlayer.weights"].shape:
                            can_restore = False
                    if can_restore:
                        print(f"\n   [KTA Guard] KTA decayed >15% from peak ({best_kta:.4f} -> {kta:.4f}) for 5 epochs.")
                        print(f"               Restoring peak-KTA checkpoint from Epoch {best_kta_epoch} and freezing quantum backbone.")
                        hybrid.load_state_dict(best_kta_state)
                        for p in hybrid.backbone.parameters():
                            p.requires_grad = False
                        for p in hybrid.qlayer.parameters():
                            p.requires_grad = False
                        kta_decay_count = 0
                    else:
                        # State is from an earlier depth stage: re-anchor to current stage
                        best_kta = kta
                        best_kta_epoch = global_epoch
                        best_kta_state = copy.deepcopy(hybrid.state_dict())
                        kta_decay_count = 0

            print(f"   Hybrid  → loss={train_res_h['loss']:.4f} (tot={train_res_h.get('total_loss', train_res_h['loss']):.4f})"
                  f"  acc={train_res_h['accuracy']:.4f}"
                  f"  val_loss={val_res_h['loss']:.4f}  val_acc={val_res_h['accuracy']:.4f}"
                  f"  [{t_hybrid:.1f}s]")
            print(f"   Baseline→ loss={train_res_b['loss']:.4f}  acc={train_res_b['accuracy']:.4f}"
                  f"  val_loss={val_res_b['loss']:.4f}  val_acc={val_res_b['accuracy']:.4f}"
                  f"  [{t_baseline:.1f}s]")
            print(f"   Entanglement Entropy: S(ρ_A)={ee:.4f}  |  KTA: S(K,Y)={kta:.4f}")

            if not use_qng:
                scheduler_h.step()
            scheduler_b.step()

    # Capture final models and restore best models
    last_h_state = copy.deepcopy(hybrid.state_dict())
    last_b_state = copy.deepcopy(baseline.state_dict())

    if best_h_state is not None:
        hybrid.load_state_dict(best_h_state)
        print(f"\n[checkpoint] Restored best Hybrid model from Epoch {best_h_epoch} (Val AUC: {best_h_auc:.4f}, Val Loss: {best_h_loss:.4f})")
    if best_b_state is not None:
        baseline.load_state_dict(best_b_state)
        print(f"[checkpoint] Restored best Baseline model from Epoch {best_b_epoch} (Val AUC: {best_b_auc:.4f}, Val Loss: {best_b_loss:.4f})")

    return {
        "hybrid_hist": hybrid_hist,
        "baseline_hist": baseline_hist,
        "all_grad_norms_vqc": all_grad_norms_vqc,
        "all_grad_norms_adapter": all_grad_norms_adapter,
        "all_grad_norms_head": all_grad_norms_head,
        "entropy_history": entropy_history,
        "kta_history": kta_history,
        "stage_transitions": stage_transitions,
        "epoch_times_hybrid": epoch_times_hybrid,
        "epoch_times_baseline": epoch_times_baseline,
        "criterion": criterion,
        "best_hybrid_state": best_h_state,
        "best_baseline_state": best_b_state,
        "last_hybrid_state": last_h_state,
        "last_baseline_state": last_b_state,
        "best_epoch_hybrid": best_h_epoch,
        "best_val_auc_hybrid": best_h_auc,
        "best_val_loss_hybrid": best_h_loss,
        "best_epoch_baseline": best_b_epoch,
        "best_val_auc_baseline": best_b_auc,
        "best_val_loss_baseline": best_b_loss,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 8.  MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
def run_training(
    dataset_name: str = "pneumoniamnist",
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-4,
    n_qubits: int = 4,
    n_layers: int = 3,
    mixup_alpha: float = 0.4,
    phase_eps: float = 0.05,
    output_dir: str = "outputs",
    lambda_ewc: float = 400.0,
    dropout: float = 0.3,
    diff_method: str = "auto",
    use_lora: bool = False,
    use_qng: bool = False,
    use_qsam: bool = False,
    use_qswa: bool = False,
    lambda_infonce: float = 0.0,
    image_size: int = 224,
    pretrain_epochs: int = 30,
    ansatz_mode: str = "standard_static",
    use_qnas: bool = False,
    qnas_generations: int = 6,
    qnas_pop_size: int = 16,
    qnas_min_cnot: int = 2,
    qnas_max_cnot: int = 6,
    qnas_target_mw: float = 0.45,
    qnas_lambda_cnot: float = 0.01,
    qnas_genome_path: Optional[str] = None,
    qdevice: str = "auto",
    force_pretrain: bool = False,
    adapter_checkpoint: Optional[str] = None,
    progressive: bool = False,
    progressive_depths: Optional[List[int]] = None,
    epochs_per_stage: Optional[int] = None,
    finetune_epochs: int = 5,
    freeze_mode: str = "hard",
    q_init_std: float = 0.01,
    balanced_sampler: bool = True,
    asymmetric_loss: bool = True,
    asym_alpha: Optional[Union[str, List[float], float]] = None,
    asym_gamma: Optional[Union[str, List[float], float]] = None,
    threshold_criterion: str = "youden",
    target_specificity: float = 0.90,
    cost_fp: float = 1.5,
    entropy_every_epoch: bool = True,
    entropy_interval: int = 1,
    entropy_samples: int = 32,
    fast_eval: bool = False,
    skip_plots: bool = False,
    plot_workers: int = 4,
    use_amp: bool = True,
    cache_in_vram: bool = False,
    qsam_freq: int = 1,
    grad_accum_steps: int = 1,
    benchmark_mode: bool = False,
    profile_runtime: bool = False,
    post_quantum_mixup: bool = True,
    lambda_kta: float = 0.05,
    use_cost_matrix: bool = False,
    loss_type: str = "focal",
    ldam_max_margin: float = 0.50,
    drw_start_epoch: int = 10,
    dual_basis_readout: bool = False,
    use_qaf_attention: bool = False,
    clinical_hard: bool = False,
    enable_vit_finetuning: bool = False,
    vit_unfreeze_layers: int = 2,
    vit_finetune_lr: float = 1e-5,
    vit_epochs: int = 5,
    biomedclip_dir: str = "biomedclip",
    lambda_decorr: float = 0.01,
    lambda_saliency: float = 0.05,
) -> Dict:
    """Full training pipeline: train both models, evaluate, generate all plots."""
    if ansatz_mode == "qnas_static":
        use_qnas = True
        progressive = False
    elif ansatz_mode == "progressive_dense":
        progressive = True
        use_qnas = False
    elif ansatz_mode == "qnas_progressive":
        use_qnas = True
        progressive = True

    device = get_torch_device()
    out_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "eda"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  HYBRID CNN-VQC TRAINING PIPELINE")
    print(f"  Dataset      : {dataset_name}")
    print(f"  Image Size   : {image_size}x{image_size} (Native High-Res ViT Input)")
    print(f"  Epochs       : {epochs}")
    print(f"  Pre-train Eps: {pretrain_epochs}")
    print(f"  Qubits       : {n_qubits}")
    print(f"  VQC Layers   : {n_layers}")
    print(f"  Device       : {device}")
    print(f"  Quantum Dev  : {qdevice}")
    print(f"  Diff Method  : {diff_method}")
    print(f"  Output       : {out_dir}")
    print(f"  Batch Size   : {batch_size}")
    print(f"  Learning Rate: {lr}")
    print(f"  Dropout      : {dropout}")
    print(f"  Mixup Alpha  : {mixup_alpha}")
    print(f"  Phase Eps    : {phase_eps}")
    print(f"  Lambda EWC   : {lambda_ewc}")
    print(f"  Use LoRA-Q   : {use_lora}")
    print(f"  Use QNG      : {use_qng}")
    print(f"  Use Q-SAM    : {use_qsam}")
    print(f"  Use Q-SWA    : {use_qswa}")
    print(f"  InfoNCE Wt   : {lambda_infonce}")
    print(f"  Ansatz Mode  : {ansatz_mode}")
    print(f"  Use QNAS     : {use_qnas}")
    if use_qnas:
        print(f"  QNAS Config  : Gens={qnas_generations}, Pop={qnas_pop_size}, CNOT Budget=[{qnas_min_cnot}, {qnas_max_cnot}], Target MW={qnas_target_mw}, Lambda CNOT={qnas_lambda_cnot}")
    print(f"  Progressive  : {progressive}")
    if progressive:
        actual_pdepths = progressive_depths if progressive_depths is not None else [2, 4, 6]
        print(f"  Prog Depths  : {actual_pdepths}")
        print(f"  Freeze Mode  : {freeze_mode}")
        print(f"  Init Std     : {q_init_std}")
    print(f"{'='*70}\n")

    # ── Load data ─────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, meta = get_medmnist_dataloaders(
        dataset_name, batch_size=batch_size, download_root="./data", size=image_size,
        balanced_sampling=balanced_sampler
    )
    raw_test_loader = test_loader
    n_classes = meta["n_classes"]
    in_channels = meta["in_channels"]
    meta["dataset_name"] = dataset_name
    if "label_map" not in meta or not meta["label_map"]:
        info_label = meta.get("info", {}).get("label", {})
        if info_label:
            meta["label_map"] = {int(k): v for k, v in info_label.items()}
        elif dataset_name in DATASET_CONFIG and "label_map" in DATASET_CONFIG[dataset_name]:
            meta["label_map"] = DATASET_CONFIG[dataset_name]["label_map"]
    print(f"[data] Resolution={meta.get('image_size', image_size)}x{meta.get('image_size', image_size)}, "
          f"Train={meta['train_size']}, Val={meta['val_size']}, "
          f"Test={meta['test_size']}, Classes={n_classes}")

    # EDA plots
    plot_class_distribution(train_loader, meta,
                            save_path=os.path.join(out_dir, "eda", "class_dist.png"))
    plot_sample_grid(train_loader, meta,
                     save_path=os.path.join(out_dir, "eda", "samples.png"))
    plot_pixel_intensity_histogram(train_loader, meta,
                                   save_path=os.path.join(out_dir, "eda", "pixel_hist.png"))


    # ── Pre-flight safety assertions ────────────────────────────────────────
    if dual_basis_readout:
        assert n_qubits >= 4, f"dual_basis_readout requires n_qubits >= 4, got {n_qubits}"
    if use_qaf_attention:
        assert n_qubits >= 4, f"QAF ansatz requires n_qubits >= 4, got {n_qubits}"
    if loss_type == "cb_ldam":
        assert n_classes >= 2, f"CB-LDAM requires n_classes >= 2, got {n_classes}"

    # ── BiomedCLIP ViT Fine-Tuning (Isolated Checkpoint) ───────────────────
    if enable_vit_finetuning:
        try:
            from finetuning import finetune_biomedclip_vit, BiomedCLIPFineTuneConfig
        except ImportError:
            from qml_models.finetuning import finetune_biomedclip_vit, BiomedCLIPFineTuneConfig
        ft_cfg = BiomedCLIPFineTuneConfig(
            enable_finetuning=True,
            dataset_name=dataset_name,
            n_classes=n_classes,
            unfreeze_layers=vit_unfreeze_layers,
            finetune_lr=vit_finetune_lr,
            epochs=vit_epochs,
            save_dir=biomedclip_dir,
            device=str(device),
        )
        print(f"\n[vit_finetune] Initiating BiomedCLIP ViT fine-tuning ({vit_unfreeze_layers} layers, lr={vit_finetune_lr}, epochs={vit_epochs}, classes={n_classes})...")
        finetune_biomedclip_vit(
            ft_cfg,
            train_loader,
            val_loader=val_loader,
            device=device,
            n_classes=n_classes,
        )
        print(f"[vit_finetune] Fine-tuned ViT weights saved to: {biomedclip_dir}/{dataset_name}_biomedclip_vit.pt\n")

    # ── Build models ──────────────────────────────────────────────────────
    init_vqc_layers = (progressive_depths if progressive_depths is not None else [2, 4, 6])[0] if progressive else n_layers
    hybrid, baseline = build_models(
        in_channels, n_classes, n_qubits, init_vqc_layers, device, dropout,
        diff_method=diff_method, use_lora=use_lora, qdevice=qdevice,
        q_init_std=q_init_std,
        dual_basis_only=dual_basis_readout,
        use_qaf_attention=use_qaf_attention,
        clinical_hard=clinical_hard,
        finetune_backbone=enable_vit_finetuning,
        ft_dataset_name=dataset_name,
        ft_n_unfreeze=vit_unfreeze_layers,
        ft_base_dir=biomedclip_dir,
    )

    # ── Offline Feature Extraction (Resolution-Aware Cache) ───────────────
    cache_dir = os.path.join(out_dir, f"cache_{image_size}")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[cache] Starting offline feature extraction (Cache: {cache_dir})...")
    cache_common = {
        "dataset_name": dataset_name,
        "image_size": int(image_size),
        "encoder_mode": "finetuned" if enable_vit_finetuning else "frozen",
        "finetune_layers": int(vit_unfreeze_layers) if enable_vit_finetuning else 0,
        "finetune_dir": str(biomedclip_dir),
    }
    train_loader = extract_and_cache_features(
        hybrid, train_loader, device, os.path.join(cache_dir, "train.pt"),
        cache_in_vram=cache_in_vram, cache_metadata={**cache_common, "split": "train"},
    )
    val_loader = extract_and_cache_features(
        hybrid, val_loader, device, os.path.join(cache_dir, "val.pt"),
        cache_in_vram=cache_in_vram, cache_metadata={**cache_common, "split": "val"},
    )
    test_loader = extract_and_cache_features(
        hybrid, test_loader, device, os.path.join(cache_dir, "test.pt"),
        cache_in_vram=cache_in_vram, cache_metadata={**cache_common, "split": "test"},
    )

    # Offload ViT vision encoder to CPU to free GPU VRAM for downstream VQC training
    hybrid.offload_vision_encoder()
    baseline.offload_vision_encoder()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[cache] Offloaded ViT backbone to CPU to conserve GPU VRAM for quantum layers.")

    # ── Dataset-Specific CQ Adapter Cache Lookup ─────────────────────────
    adapter_dir = get_adapter_dir(dataset_name)
    adapter_ckpt = Path(adapter_checkpoint) if adapter_checkpoint else get_adapter_checkpoint_path(
        dataset=dataset_name,
        n_qubits=n_qubits,
        use_lora=use_lora,
        lora_rank=4,
    )

    if adapter_ckpt.exists() and not force_pretrain:
        print(f"\n[adapter] Found pre-trained CQ-Adapter checkpoint:")
        print(f"          Path: {adapter_ckpt}")
        meta = load_adapter_checkpoint(hybrid.backbone, adapter_ckpt, device=device)
        print(f"[adapter] Successfully loaded pre-trained weights (dataset: {meta.get('dataset', dataset_name)}).")
        print(f"[adapter] Skipping {pretrain_epochs} pre-training epochs — instant cache hit!\n")
    else:
        print(f"\n[adapter] Pre-training CQ-Adapter for dataset '{dataset_name}' ({n_qubits} qubits, {pretrain_epochs} epochs)...")
        pretrain_cq_adapter(
            hybrid.backbone,
            train_loader,
            device,
            epochs=pretrain_epochs,
            out_dir=str(adapter_dir),
            dataset_name=dataset_name,
        )
        save_adapter_checkpoint(
            hybrid.backbone,
            adapter_ckpt,
            metadata={"dataset": dataset_name, "n_qubits": n_qubits, "epochs": pretrain_epochs, "use_lora": use_lora}
        )
        print(f"[adapter] Saved pre-trained weights and diagnostic plots to: {adapter_dir}\n")

    # The baseline is a classical control, not a different representation
    # learning experiment.  Start it from the same pretrained CQ projection
    # and then let its MLP optimize independently from the hybrid head.
    baseline.backbone.cq_adapter.load_state_dict(copy.deepcopy(hybrid.backbone.cq_adapter.state_dict()))
    if hasattr(hybrid.backbone, "skip") and hasattr(baseline.backbone, "skip"):
        baseline.backbone.skip.load_state_dict(copy.deepcopy(hybrid.backbone.skip.state_dict()))

    print("[pretrain] CQ Adapter & Residual Skip remain trainable — will be fine-tuned with lr=2e-4.")

    best_genome = None
    if use_qnas:
        resolved_genome_path = qnas_genome_path
        if resolved_genome_path and os.path.exists(resolved_genome_path):
            print(f"[QNAS] Loading pre-computed QNAS genome from: {resolved_genome_path}")
            best_genome = load_genome(resolved_genome_path)
        else:
            print(f"\n{'='*70}")
            print(f"  QUANTUM NEURAL ARCHITECTURE SEARCH (QNAS) ACTIVATED")
            print(f"  Generations : {qnas_generations}")
            print(f"  Population  : {qnas_pop_size}")
            print(f"  CNOT Budget : [{qnas_min_cnot}, {qnas_max_cnot}] (enforces minimum entanglement >= {qnas_min_cnot} CNOTs)")
            print(f"  Target MW   : {qnas_target_mw}")
            print(f"  Lambda CNOT : {qnas_lambda_cnot}")
            print(f"{'='*70}\n")

            # 1. Extract latent features from trained CQ adapter
            hybrid.eval()
            train_z_list, train_y_list = [], []
            with torch.no_grad():
                for bx, by in train_loader:
                    bx = bx.to(device)
                    bz = hybrid.backbone(bx, use_cached_features=True)
                    train_z_list.append(bz.cpu())
                    train_y_list.append(by.squeeze().cpu())
                    if sum(len(x) for x in train_z_list) >= 400:
                        break
            train_z_sub = torch.cat(train_z_list, dim=0)[:400]
            train_y_sub = torch.cat(train_y_list, dim=0)[:400]

            val_z_list, val_y_list = [], []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(device)
                    bz = hybrid.backbone(bx, use_cached_features=True)
                    val_z_list.append(bz.cpu())
                    val_y_list.append(by.squeeze().cpu())
                    if sum(len(x) for x in val_z_list) >= 150:
                        break
            val_z_sub = torch.cat(val_z_list, dim=0)[:150]
            val_y_sub = torch.cat(val_y_list, dim=0)[:150]

            # 2. Initialize QNAS optimizer & compute Mutual Information Admissibility Graph
            qnas_opt = QNASOptimizer(
                n_qubits=n_qubits,
                min_cnot=qnas_min_cnot,
                max_cnot=qnas_max_cnot,
                max_depth=init_vqc_layers if progressive else n_layers,
                population_size=qnas_pop_size,
                generations=qnas_generations,
                lambda_cnot=qnas_lambda_cnot,
                target_mw=qnas_target_mw,
                output_dir=out_dir,
                qdevice=qdevice,
            )
            mi_matrix = qnas_opt.setup_admissibility_graph(train_z_sub)
            print(f"[QNAS] Admissible entangling pairs (|E|={len(qnas_opt.admissible_pairs)}): {qnas_opt.admissible_pairs}")

            # Plot admissibility graph
            qnas_plot_path = os.path.join(out_dir, "qnas_admissibility_graph.png")
            plot_qnas_admissibility(mi_matrix, qnas_opt.admissible_pairs, n_qubits, qnas_plot_path)

            # 3. Evolutionary search
            best_genome = qnas_opt.evolve(
                train_z_sub, train_y_sub, val_z_sub, val_y_sub, n_classes=n_classes
            )
            print(f"\n[QNAS] Optimal Genome Found:")
            print(f"       CNOT count  : {best_genome.cnot_count} (vs. baseline {n_layers * n_qubits})")
            print(f"       Param count : {best_genome.total_params} (vs. baseline {n_layers * n_qubits * 3})")
            print(f"       Proxy AUC   : {best_genome.val_auc:.4f}")
            print(f"       MW Entangle : {best_genome.mw_entanglement:.4f}")

            # 4. Save best genome artifact
            genome_path = os.path.join(out_dir, "checkpoints", "qnas_genome.json")
            save_genome(best_genome, genome_path)
            print(f"[QNAS] Saved Pareto-optimal genome to {genome_path}")

        # 5. Re-instantiate HybridCNNVQC with discovered sparse genome while preserving pre-trained adapter weights
        old_adapter_state = copy.deepcopy(hybrid.backbone.cq_adapter.state_dict())
        old_skip_state = copy.deepcopy(hybrid.backbone.skip.state_dict()) if hasattr(hybrid.backbone, "skip") else None

        hybrid = HybridCNNVQC(
            in_channels=in_channels,
            n_classes=n_classes,
            n_qubits=n_qubits,
            n_layers=init_vqc_layers if progressive else n_layers,
            dropout=dropout,
            diff_method=diff_method,
            use_lora=use_lora,
            genome=best_genome,
            qdevice=qdevice,
            dual_basis_only=dual_basis_readout,
            use_qaf_attention=use_qaf_attention,
            clinical_hard=clinical_hard,
        ).to(device)

        # Restore pre-trained adapter and skip weights
        hybrid.backbone.cq_adapter.load_state_dict(old_adapter_state)
        if old_skip_state is not None and hasattr(hybrid.backbone, "skip"):
            hybrid.backbone.skip.load_state_dict(old_skip_state)
        print("[QNAS] Rebuilt HybridCNNVQC with discovered sparse ansatz and restored pre-trained adapter weights.")

    # ── Loss & Optimiser (Fix: Flaws F-01, F-04, F-12, and Drusen Remediation) ────────
    _effective_loss_type = loss_type
    # Auto-detect CB-LDAM for OCTMNIST if not explicitly overridden
    if dataset_name == "octmnist" and n_classes == 4 and _effective_loss_type == "focal":
        _effective_loss_type = "cb_ldam"

    if _effective_loss_type == "cb_ldam":
        # Dynamically compute class counts from training data
        _cls_counts = _compute_class_counts(train_loader, n_classes)
        criterion = ClassBalancedLDAMLoss(
            cls_num_list=_cls_counts,
            max_margin=ldam_max_margin,
            beta=0.9999,
            drw_start_epoch=drw_start_epoch,
        ).to(device)
        print(f"[loss] Initialized ClassBalancedLDAMLoss (CB-LDAM-DRW) for {dataset_name}:")
        print(f"       Class counts : {_cls_counts}")
        print(f"       Margins      : {criterion.margins.cpu().tolist()}")
        print(f"       CB Weights   : {criterion.cb_weights.cpu().tolist()}")
        print(f"       DRW Start Ep : {criterion.drw_start_epoch}")
    elif _effective_loss_type == "ce":
        class_weights = compute_class_weights(train_loader, n_classes, device, method="effective")
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        print(f"[loss] CrossEntropyLoss with effective weights")
    else:
        class_weights = compute_class_weights(train_loader, n_classes, device, method="effective")
        print(f"[loss] Class-balanced effective weights: {class_weights.tolist()}")
        alpha_per_class, gamma_per_class = resolve_asymmetric_loss_params(
            asym_alpha=asym_alpha,
            asym_gamma=asym_gamma,
            dataset_name=dataset_name,
            n_classes=n_classes,
            asymmetric_loss=asymmetric_loss,
        )
        if alpha_per_class is not None and gamma_per_class is not None:
            print(f"[loss] Asymmetric Focal Loss enabled: α={alpha_per_class}, γ={gamma_per_class}")
        else:
            print(f"[loss] Symmetric Focal Loss enabled (gamma=2.0)")

        criterion = FocalLoss(
            gamma=2.0,
            weight=class_weights,
            alpha_per_class=alpha_per_class,
            gamma_per_class=gamma_per_class,
        ).to(device)

    stage_transitions = []
    best_h_auc = -1.0
    best_h_loss = float("inf")
    best_h_epoch = 0
    best_b_auc = -1.0
    best_b_loss = float("inf")
    best_b_epoch = 0
    last_hybrid_state = None
    last_baseline_state = None

    if progressive:
        actual_pdepths = progressive_depths if progressive_depths is not None else [2, 4, 6]
        prog_eps = epochs_per_stage if epochs_per_stage is not None else max(1, epochs // len(actual_pdepths))
        prog_results = train_progressive_hybrid(
            hybrid=hybrid,
            baseline=baseline,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            depth_schedule=actual_pdepths,
            epochs_per_stage=prog_eps,
            finetune_epochs=finetune_epochs,
            q_init_std=q_init_std,
            freeze_mode=freeze_mode,
            use_qng=use_qng,
            mixup_alpha=mixup_alpha,
            phase_eps=phase_eps,
            use_qsam=use_qsam,
            use_qswa=use_qswa,
            lambda_infonce=lambda_infonce,
            lr_baseline=lr,
            n_qubits=n_qubits,
            out_dir=out_dir,
            entropy_every_epoch=entropy_every_epoch,
            entropy_interval=entropy_interval,
            entropy_samples=entropy_samples,
            grad_accum_steps=grad_accum_steps,
            qsam_freq=qsam_freq,
            use_amp=use_amp,
            post_quantum_mixup=post_quantum_mixup,
            lambda_kta=lambda_kta,
        )
        hybrid_hist = prog_results["hybrid_hist"]
        baseline_hist = prog_results["baseline_hist"]
        all_grad_norms_vqc = prog_results["all_grad_norms_vqc"]
        all_grad_norms_adapter = prog_results["all_grad_norms_adapter"]
        all_grad_norms_head = prog_results["all_grad_norms_head"]
        entropy_history = prog_results["entropy_history"]
        kta_history = prog_results["kta_history"]
        stage_transitions = prog_results["stage_transitions"]
        epoch_times_hybrid = prog_results["epoch_times_hybrid"]
        epoch_times_baseline = prog_results["epoch_times_baseline"]
        n_layers = actual_pdepths[-1]
        best_h_epoch = prog_results.get("best_epoch_hybrid", 0)
        best_h_auc = prog_results.get("best_val_auc_hybrid", 0.0)
        best_h_loss = prog_results.get("best_val_loss_hybrid", 0.0)
        best_b_epoch = prog_results.get("best_epoch_baseline", 0)
        best_b_auc = prog_results.get("best_val_auc_baseline", 0.0)
        best_b_loss = prog_results.get("best_val_loss_baseline", 0.0)
        last_hybrid_state = prog_results.get("last_hybrid_state", None)
        last_baseline_state = prog_results.get("last_baseline_state", None)
    else:
        # Optimizer filtering: only pass trainable parameters (include residual skip connection)
        trainable_classical = list(filter(lambda p: p.requires_grad, hybrid.backbone.cq_adapter.parameters()))
        if hasattr(hybrid.backbone, "skip"):
            trainable_classical += list(filter(lambda p: p.requires_grad, hybrid.backbone.skip.parameters()))

        if use_qng:
            print("[optimizer] Initializing HybridQMLOptimizer with Quantum Natural Gradient (QNG)...")
            opt_hybrid = HybridQMLOptimizer(
                trainable_classical + list(hybrid.head.parameters()),
                model=hybrid,
                lr_classical=2e-4,
                lr_quantum=5e-3,
            )
        else:
            # Decoupled learning rates: 5e-3 for VQC circuit, 2e-4 for adapter, 1e-3 for head (Fix: Flaw F-04 & F-12)
            opt_hybrid = torch.optim.AdamW([
                {"params": trainable_classical,                                         "lr": 2e-4, "weight_decay": 1e-4},
                {"params": filter(lambda p: p.requires_grad, hybrid.qlayer.parameters()), "lr": 5e-3, "weight_decay": 0.0},
                {"params": filter(lambda p: p.requires_grad, hybrid.head.parameters()),   "lr": 1e-3, "weight_decay": 1e-4},
            ])

        opt_baseline = torch.optim.Adam(
            filter(lambda p: p.requires_grad, baseline.parameters()), lr=lr
        )

        qsam_h = QSAM(opt_hybrid, hybrid, rho=0.05) if use_qsam else None
        swa_handler = QSWAHandler(hybrid, start_epoch=max(1, int(epochs * 0.70)), total_epochs=epochs) if use_qswa else None
        infonce_fn = QuantumInfoNCELoss(n_qubits, tau=0.07).to(device) if lambda_infonce > 0 else None

        # Cosine Annealing Warm Restarts (SGDR) to escape saddle points
        scheduler_h = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt_hybrid if not use_qng else opt_hybrid.opt_classical, T_0=10, T_mult=2, eta_min=1e-6
        )
        scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_baseline, T_max=epochs)

        # ── Training loop ─────────────────────────────────────────────────────
        hybrid_hist = {"train_loss": [], "train_total_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        baseline_hist = {"train_loss": [], "train_total_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        all_grad_norms_vqc     = []
        all_grad_norms_adapter = []
        all_grad_norms_head    = []
        entropy_history = []
        kta_history = []
        epoch_times_hybrid = []
        epoch_times_baseline = []
        best_h_state = None
        best_b_state = None
        best_kta = -1.0
        best_kta_epoch = 0
        best_kta_state = None
        kta_decay_count = 0

        for epoch in range(1, epochs + 1):
            if hasattr(criterion, "update_epoch"):
                criterion.update_epoch(epoch)
            print(f"\n── Epoch {epoch}/{epochs} {'─'*50}")

            # --- Hybrid ---
            t0 = time.time()
            train_res_h = train_one_epoch(
                hybrid, train_loader, criterion, opt_hybrid, device, n_classes,
                use_quantum_aug=True, mixup_alpha=mixup_alpha, phase_eps=phase_eps,
                epoch=epoch,
                lambda_ent=0.05,
                s_target_frac=0.70,
                qsam=qsam_h,
                lambda_infonce=lambda_infonce,
                infonce_fn=infonce_fn,
                grad_accum_steps=grad_accum_steps,
                qsam_freq=qsam_freq,
                use_amp=use_amp,
                post_quantum_mixup=post_quantum_mixup,
                lambda_kta=lambda_kta,
            )
            t_hybrid = time.time() - t0
            epoch_times_hybrid.append(t_hybrid)
            val_res_h = evaluate(hybrid, val_loader, criterion, device, n_classes, max_samples=1000)

            hybrid_hist["train_loss"].append(train_res_h["loss"])
            hybrid_hist["train_total_loss"].append(train_res_h.get("total_loss", train_res_h["loss"]))
            hybrid_hist["val_loss"].append(val_res_h["loss"])
            hybrid_hist["train_acc"].append(train_res_h["accuracy"])
            hybrid_hist["val_acc"].append(val_res_h["accuracy"])
            all_grad_norms_vqc.extend(train_res_h.get("grad_norms_vqc", []))
            all_grad_norms_adapter.extend(train_res_h.get("grad_norms_adapter", []))
            all_grad_norms_head.extend(train_res_h.get("grad_norms_head", []))

            # Collect SWA weights in final 30% of epochs
            if use_qswa and swa_handler is not None and swa_handler.should_collect(epoch):
                swa_handler.collect()

            # --- Baseline ---
            t0 = time.time()
            train_res_b = train_one_epoch(
                baseline, train_loader, criterion, opt_baseline, device, n_classes,
                use_quantum_aug=False, epoch=epoch,
                grad_accum_steps=grad_accum_steps,
                use_amp=use_amp,
            )
            t_baseline = time.time() - t0
            epoch_times_baseline.append(t_baseline)
            val_res_b = evaluate(baseline, val_loader, criterion, device, n_classes, max_samples=1000)

            baseline_hist["train_loss"].append(train_res_b["loss"])
            baseline_hist["train_total_loss"].append(train_res_b.get("total_loss", train_res_b["loss"]))
            baseline_hist["val_loss"].append(val_res_b["loss"])
            baseline_hist["train_acc"].append(train_res_b["accuracy"])
            baseline_hist["val_acc"].append(val_res_b["accuracy"])

            # Track best models (Compound criterion: highest val_auc, tie-breaker val_loss)
            val_h_auc = val_res_h.get("auroc", val_res_h["accuracy"])
            val_h_loss = val_res_h["loss"]
            if _is_better_checkpoint(val_h_auc, val_h_loss, best_h_auc, best_h_loss):
                best_h_auc = val_h_auc
                best_h_loss = val_h_loss
                best_h_epoch = epoch
                best_h_state = copy.deepcopy(hybrid.state_dict())
                print(f"      ★ New best Hybrid checkpoint (Val AUC: {best_h_auc:.4f}, Val Loss: {best_h_loss:.4f})")

            val_b_auc = val_res_b.get("auroc", val_res_b["accuracy"])
            val_b_loss = val_res_b["loss"]
            if _is_better_checkpoint(val_b_auc, val_b_loss, best_b_auc, best_b_loss):
                best_b_auc = val_b_auc
                best_b_loss = val_b_loss
                best_b_epoch = epoch
                best_b_state = copy.deepcopy(baseline.state_dict())
                print(f"      ★ New best Baseline checkpoint (Val AUC: {best_b_auc:.4f}, Val Loss: {best_b_loss:.4f})")

            # Entanglement entropy & KTA (vectorized GPU state evaluation)
            should_compute_entropy = (
                entropy_every_epoch
                or (epoch % entropy_interval == 0)
                or epoch == 1
                or epoch == epochs
                or len(entropy_history) == 0
            )
            if should_compute_entropy:
                ee = compute_entanglement_entropy_epoch(
                    hybrid, val_loader, device, n_qubits, n_samples=entropy_samples
                )
                kta = compute_kernel_target_alignment(
                    hybrid, val_loader, device, n_qubits, n_samples=min(entropy_samples * 2, 64)
                )
            else:
                ee = entropy_history[-1] if entropy_history else 0.0
                kta = kta_history[-1] if kta_history else 0.0
            entropy_history.append(ee)
            kta_history.append(kta)

            # KTA Peak & Decay Guard (Fix: FL-09)
            if kta > best_kta:
                best_kta = kta
                best_kta_epoch = epoch
                best_kta_state = copy.deepcopy(hybrid.state_dict())
                kta_decay_count = 0
            elif best_kta > 0.0 and (best_kta - kta) / best_kta > 0.15:
                kta_decay_count += 1
                if kta_decay_count >= 5 and best_kta_state is not None:
                    print(f"\n   [KTA Guard] KTA decayed >15% from peak ({best_kta:.4f} -> {kta:.4f}) for 5 epochs.")
                    print(f"               Restoring peak-KTA checkpoint from Epoch {best_kta_epoch} and freezing quantum backbone.")
                    hybrid.load_state_dict(best_kta_state)
                    for p in hybrid.backbone.parameters():
                        p.requires_grad = False
                    for p in hybrid.qlayer.parameters():
                        p.requires_grad = False
                    kta_decay_count = 0

            print(f"   Hybrid  → loss={train_res_h['loss']:.4f} (tot={train_res_h.get('total_loss', train_res_h['loss']):.4f})"
                  f"  acc={train_res_h['accuracy']:.4f}"
                  f"  val_loss={val_res_h['loss']:.4f}  val_acc={val_res_h['accuracy']:.4f}"
                  f"  [{t_hybrid:.1f}s]")
            print(f"   Baseline→ loss={train_res_b['loss']:.4f}  acc={train_res_b['accuracy']:.4f}"
                  f"  val_loss={val_res_b['loss']:.4f}  val_acc={val_res_b['accuracy']:.4f}"
                  f"  [{t_baseline:.1f}s]")
            print(f"   Entanglement Entropy: S(ρ_A)={ee:.4f}  |  KTA: S(K,Y)={kta:.4f}")

            if not use_qng:
                scheduler_h.step(epoch)
            scheduler_b.step()

        # Capture final epoch models before restoration
        last_hybrid_state = copy.deepcopy(hybrid.state_dict())
        last_baseline_state = copy.deepcopy(baseline.state_dict())

        # Restore best weights if not applying SWA
        if not use_qswa:
            if best_h_state is not None:
                hybrid.load_state_dict(best_h_state)
                print(f"\n[checkpoint] Restored best Hybrid model from Epoch {best_h_epoch} (Val AUC: {best_h_auc:.4f}, Val Loss: {best_h_loss:.4f})")
            if best_b_state is not None:
                baseline.load_state_dict(best_b_state)
                print(f"[checkpoint] Restored best Baseline model from Epoch {best_b_epoch} (Val AUC: {best_b_auc:.4f}, Val Loss: {best_b_loss:.4f})")
        else:
            # Apply SWA averaged weights if enabled
            if swa_handler is not None:
                swa_handler.apply_swa_weights()

    # ── Final Test Evaluation ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  FINAL TEST EVALUATION")
    print(f"{'='*70}")

    # Define OCTMNIST asymmetric clinical cost matrix for Bayes risk minimization
    oct_cost_matrix = None
    if dataset_name == "octmnist" and n_classes == 4:
        # C[action, true_class]; transpose the established true-by-action costs.
        # Classes: 0: CNV, 1: DME, 2: DRUSEN, 3: NORMAL
        # C(Drusen->CNV) = 1.5 avoids total suppression of CNV predictions
        oct_cost_matrix = torch.tensor([
            [0.0, 1.0, 1.0, 2.0],   # True CNV
            [1.0, 0.0, 1.0, 2.0],   # True DME
            [1.5, 1.5, 0.0, 2.0],   # True Drusen
            [2.0, 1.5, 1.0, 0.0],   # True Normal: false alarm penalty
        ], device=device).T.contiguous()
    decision_policy = {
        "type": "cost_matrix" if oct_cost_matrix is not None else "argmax",
        "cost_matrix": oct_cost_matrix.detach().cpu().tolist() if oct_cost_matrix is not None else None,
        "cost_matrix_orientation": "action_by_true_class",
    }

    hybrid_test  = evaluate(hybrid,   test_loader, criterion, device, n_classes, max_samples=1000, cost_matrix=oct_cost_matrix)
    baseline_test = evaluate(baseline, test_loader, criterion, device, n_classes, max_samples=1000, cost_matrix=oct_cost_matrix)

    # Temperature scaling + Youden's J threshold calibration
    print("\n[calibration] Optimizing temperature scaling on validation set...")
    T_opt = optimize_temperature_scaling(hybrid, val_loader, device)

    hybrid_test_logits, _ = _collect_val_logits(hybrid, test_loader, device)
    hybrid_test_probs_temp = F.softmax(hybrid_test_logits / max(T_opt, 1e-6), dim=1)

    val_logits, val_labels_arr = _collect_val_logits(hybrid, val_loader, device)
    val_probs_temp = F.softmax(val_logits / max(T_opt, 1e-6), dim=1)
    val_labels_np = val_labels_arr.numpy()
    cal_fit_idx, conformal_idx = split_calibration_indices(val_labels_np, seed=42)
    fit_idx_tensor = torch.as_tensor(cal_fit_idx, dtype=torch.long)
    fit_probs_temp = val_probs_temp[fit_idx_tensor]
    fit_labels_tensor = val_labels_arr[fit_idx_tensor]

    # Calibration for bounded quantum outputs (Fix: Defect F-01 & Solution 1.5)
    if n_classes == 2:
        print("[calibration] Fitting Beta calibration on validation set...")
        beta_cal = QuantumBetaCalibrator()
        beta_cal.fit(fit_probs_temp.detach(), fit_labels_tensor)
        with torch.no_grad():
            hybrid_test_probs_cal = beta_cal(hybrid_test_probs_temp).numpy()
            val_probs_cal = beta_cal(val_probs_temp).detach().numpy()
        tau_star, _ = optimize_decision_threshold(
            val_probs_cal[cal_fit_idx], val_labels_np[cal_fit_idx],
            criterion=threshold_criterion,
            target_specificity=target_specificity,
            cost_fp=cost_fp,
        )
        p_pos_test = hybrid_test_probs_cal[:, 1]
        hybrid_test_preds_cal = (p_pos_test >= tau_star).astype(int)
    else:
        print(f"[calibration] Fitting Multi-Class Dirichlet calibration on validation set ({n_classes} classes)...")
        multi_cal = QuantumMultiClassCalibrator(n_classes=n_classes)
        multi_cal.fit(fit_probs_temp.detach(), fit_labels_tensor)
        with torch.no_grad():
            hybrid_test_probs_cal = multi_cal(hybrid_test_probs_temp).numpy()
            val_probs_cal = multi_cal(val_probs_temp).detach().numpy()
        
        # Apply Cost-Sensitive Bayes Risk Minimization if cost matrix defined
        if oct_cost_matrix is not None:
            hybrid_test_preds_cal = predict_with_cost_matrix(
                hybrid_test_probs_cal, oct_cost_matrix.cpu().numpy()
            )
            print("[calibration] Applied Cost-Sensitive Bayes Risk Minimization to calibrated multi-class probabilities.")
        else:
            hybrid_test_preds_cal = np.argmax(hybrid_test_probs_cal, axis=1)
        tau_star = 0.5

    # P0-Fix: Unified calibrator reference for checkpoint serialization
    _cal_ckpt = beta_cal if n_classes == 2 else multi_cal
    _cal_type = "beta" if n_classes == 2 else "multiclass"
    conformal_scores = build_mondrian_conformal_artifact(
        val_probs_cal[conformal_idx], val_labels_np[conformal_idx], alpha=0.05
    )

    hybrid_test_cal = dict(hybrid_test)
    hybrid_test_cal["probs"] = hybrid_test_probs_cal
    hybrid_test_cal["preds"] = hybrid_test_preds_cal
    hybrid_test_cal["model"] = hybrid  # Fix E-03/E-04: Pass model for decision boundary & kernel gram

    # Baseline threshold calibration on validation set (Fix: prevent uncalibrated argmax collapse to all-Normal)
    print("[calibration] Optimizing baseline temperature scaling on validation set...")
    T_opt_b = optimize_temperature_scaling(baseline, val_loader, device)
    val_logits_b, _ = _collect_val_logits(baseline, val_loader, device)
    val_probs_b_temp = F.softmax(val_logits_b / max(T_opt_b, 1e-6), dim=1)
    fit_probs_b_temp = val_probs_b_temp[fit_idx_tensor]
    baseline_test_logits, _ = _collect_val_logits(baseline, test_loader, device)
    baseline_test_probs_temp = F.softmax(baseline_test_logits / max(T_opt_b, 1e-6), dim=1)

    if n_classes == 2:
        print("[calibration] Fitting baseline Beta calibration on validation set...")
        baseline_calibrator = QuantumBetaCalibrator()
        baseline_calibrator.fit(fit_probs_b_temp.detach(), fit_labels_tensor)
        with torch.no_grad():
            val_probs_b = baseline_calibrator(val_probs_b_temp).numpy()
            baseline_test_probs = baseline_calibrator(baseline_test_probs_temp).numpy()
        baseline_conformal_scores = build_mondrian_conformal_artifact(
            val_probs_b[conformal_idx], val_labels_np[conformal_idx], alpha=0.05
        )
        baseline_calibrator_type = "beta"
    else:
        print(f"[calibration] Fitting baseline Dirichlet calibration on validation set ({n_classes} classes)...")
        baseline_calibrator = QuantumMultiClassCalibrator(n_classes=n_classes)
        baseline_calibrator.fit(fit_probs_b_temp.detach(), fit_labels_tensor)
        with torch.no_grad():
            val_probs_b = baseline_calibrator(val_probs_b_temp).numpy()
            baseline_test_probs = baseline_calibrator(baseline_test_probs_temp).numpy()
        baseline_conformal_scores = build_mondrian_conformal_artifact(
            val_probs_b[conformal_idx], val_labels_np[conformal_idx], alpha=0.05
        )
        baseline_calibrator_type = "multiclass"

    if n_classes == 2:
        tau_star_b, _ = optimize_decision_threshold(
            val_probs_b[cal_fit_idx], val_labels_np[cal_fit_idx],
            criterion=threshold_criterion,
            target_specificity=target_specificity,
            cost_fp=cost_fp,
        )
        baseline_test_preds = (baseline_test_probs[:, 1] >= tau_star_b).astype(int)
        print(f"[calibration] Optimized Baseline decision threshold: τ*_b = {tau_star_b:.4f}")
    else:
        baseline_test_preds = predict_with_cost_matrix(baseline_test_probs, oct_cost_matrix.cpu().numpy()) if oct_cost_matrix is not None else np.argmax(baseline_test_probs, axis=1)
        tau_star_b = 0.5

    baseline_test = dict(baseline_test)
    baseline_test["probs"] = baseline_test_probs
    baseline_test["preds"] = baseline_test_preds

    # Clinical metrics (calibrated hybrid)
    h_metrics = compute_clinical_metrics(
        hybrid_test_cal["labels"], hybrid_test_cal["preds"],
        hybrid_test_cal["probs"], n_classes,
    )
    b_metrics = compute_clinical_metrics(
        baseline_test["labels"], baseline_test["preds"],
        baseline_test["probs"], n_classes,
    )

    print("\n── Calibrated Hybrid Clinical Metrics ──")
    for k, v in h_metrics.items():
        if isinstance(v, float):
            print(f"   {k:<22}: {v:.4f}")
    print("\n── Baseline Clinical Metrics ──")
    for k, v in b_metrics.items():
        if isinstance(v, float):
            print(f"   {k:<22}: {v:.4f}")

    # Statistical significance: McNemar's test
    mcnemar_res = mcnemars_test(
        hybrid_test_cal["preds"], baseline_test["preds"], hybrid_test_cal["labels"]
    )
    p_val = mcnemar_res["p_value"]
    b_stat = mcnemar_res["b_only_correct"]
    c_stat = mcnemar_res["c_only_correct"]

    print(f"\n[stat] McNemar's Test: p-value={p_val:.4e}  "
          f"(Hybrid correct/Baseline wrong: {b_stat}, Baseline correct/Hybrid wrong: {c_stat})")
    if mcnemar_res["significant"]:
        print("       → The difference is STATISTICALLY SIGNIFICANT (p < 0.05).")
    else:
        print("       → The difference is NOT statistically significant at p = 0.05.")

    # ── Save Checkpoints (Fix: Defect — Checkpoint Contract Synchronization) ──
    # app.py L184-193 expects ckpt["config"], ckpt["meta"], ckpt["model_state_dict"]
    # Previously: torch.save(hybrid.state_dict(), ...) caused KeyError: 'config' in app.py
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    def _checkpoint_safe(v):
        if isinstance(v, (np.floating, float)):
            return float(v)
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, np.ndarray):
            return [_checkpoint_safe(x) for x in v.tolist()]
        if isinstance(v, (list, tuple)):
            return [_checkpoint_safe(x) for x in v]
        if isinstance(v, dict):
            return {str(k): _checkpoint_safe(x) for k, x in v.items()}
        return v
    hybrid_ckpt = {
        "config": {
            "schema_version": "3.0",
            "in_channels": in_channels,
            "n_classes": n_classes,
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "dropout": dropout,
            "use_lora": use_lora,
            "diff_method": diff_method,
            "qdevice": qdevice,
            "image_size": image_size,
            "use_qnas": use_qnas,
            "qnas_genome": best_genome.to_dict() if best_genome is not None else None,
            "reupload": True,
            "p_drop": 0.10,
            "q_init_std": q_init_std,
            "dual_basis_only": dual_basis_readout,
            "dual_basis_readout": dual_basis_readout,
            "use_qaf_attention": use_qaf_attention,
            "clinical_hard": clinical_hard,
            "preprocess_mode": "biomedclip_normalized",
            "class_order": list(range(n_classes)),
            "decision_policy": decision_policy,
            "finetune_backbone": enable_vit_finetuning,
            "finetune_dataset": dataset_name if enable_vit_finetuning else None,
            "finetune_unfreeze_layers": vit_unfreeze_layers if enable_vit_finetuning else 0,
            "finetune_dir": biomedclip_dir if enable_vit_finetuning else None,
        },
        "meta": {
            "dataset_name": dataset_name,
            "epochs": epochs,
            "pretrain_epochs": pretrain_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "temperature": float(T_opt),
            "threshold": float(tau_star) if tau_star is not None else 0.5,
            "best_epoch": int(best_h_epoch),
            "best_val_auc": float(best_h_auc),
            "best_val_loss": float(best_h_loss),
            "selection_criterion": "highest_val_auc_tiebreak_loss",
            "is_best": True,
        },
        "model_state_dict": hybrid.state_dict(),
        "calibrator_state_dict": _cal_ckpt.state_dict(),
        "calibrator_type": _cal_type,
        "conformal_scores": conformal_scores,
        "metrics": _checkpoint_safe(dict(h_metrics)),
    }
    baseline_ckpt = {
        "config": {
            "schema_version": "3.0",
            "in_channels": in_channels,
            "n_classes": n_classes,
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "dropout": dropout,
            "use_lora": use_lora,
            "image_size": image_size,
            "dual_basis_only": dual_basis_readout,
            "dual_basis_readout": dual_basis_readout,
            "use_qaf_attention": use_qaf_attention,
            "clinical_hard": clinical_hard,
            "use_qnas": use_qnas,
            "qnas_genome": best_genome.to_dict() if best_genome is not None else None,
            "preprocess_mode": "biomedclip_normalized",
            "class_order": list(range(n_classes)),
            "decision_policy": decision_policy,
            "finetune_backbone": enable_vit_finetuning,
            "finetune_dataset": dataset_name if enable_vit_finetuning else None,
            "finetune_unfreeze_layers": vit_unfreeze_layers if enable_vit_finetuning else 0,
            "finetune_dir": biomedclip_dir if enable_vit_finetuning else None,
        },
        "meta": {
            "dataset_name": dataset_name,
            "epochs": epochs,
            "best_epoch": int(best_b_epoch),
            "best_val_auc": float(best_b_auc),
            "best_val_loss": float(best_b_loss),
            "selection_criterion": "highest_val_auc_tiebreak_loss",
            "threshold": float(tau_star_b) if tau_star_b is not None else 0.5,
            "temperature": float(T_opt_b),
            "is_best": True,
        },
        "model_state_dict": baseline.state_dict(),
        "calibrator_state_dict": baseline_calibrator.state_dict(),
        "calibrator_type": baseline_calibrator_type,
        "conformal_scores": baseline_conformal_scores,
        "metrics": _checkpoint_safe(dict(b_metrics)),
    }
    torch.save(hybrid_ckpt, os.path.join(ckpt_dir, "hybrid_model.pt"))
    torch.save(baseline_ckpt, os.path.join(ckpt_dir, "baseline_model.pt"))

    # Also archive the final epoch checkpoint
    if last_hybrid_state is not None:
        hybrid_last_ckpt = copy.deepcopy(hybrid_ckpt)
        hybrid_last_ckpt["meta"]["is_best"] = False
        hybrid_last_ckpt["model_state_dict"] = last_hybrid_state
        torch.save(hybrid_last_ckpt, os.path.join(ckpt_dir, "hybrid_model_last.pt"))

    if last_baseline_state is not None:
        baseline_last_ckpt = copy.deepcopy(baseline_ckpt)
        baseline_last_ckpt["meta"]["is_best"] = False
        baseline_last_ckpt["model_state_dict"] = last_baseline_state
        torch.save(baseline_last_ckpt, os.path.join(ckpt_dir, "baseline_model_last.pt"))

    print(f"\n[checkpoints] Saved structured checkpoints to {ckpt_dir}/")
    print(f"             → hybrid_model.pt (BEST: Epoch {best_h_epoch}, Val AUC: {best_h_auc:.4f})")
    print(f"             → hybrid_model_last.pt (FINAL)")
    print(f"             → baseline_model.pt (BEST: Epoch {best_b_epoch}, Val AUC: {best_b_auc:.4f})")
    print(f"             → baseline_model_last.pt (FINAL)")

    # ── Save Training History ─────────────────────────────────────────────
    hist_path = os.path.join(out_dir, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump({
            "hybrid": hybrid_hist,
            "baseline": baseline_hist,
            "stage_transitions": stage_transitions,
            "grad_norms_vqc": [float(x.item() if torch.is_tensor(x) else x) for x in all_grad_norms_vqc],
            "entropy_history": [float(x) for x in entropy_history],
            "kta_history": [float(x) for x in kta_history],
            "mcnemar": dict(mcnemar_res),
            "calibrated_threshold": float(tau_star) if tau_star is not None else 0.5,
            "temperature": float(T_opt),
        }, f, indent=2)
    print(f"[history] Saved to {hist_path}")

    # ── Save Summary Metrics (Fix: Flaw F-04) ──────────────────────────────
    def _clean_json_val(v):
        if isinstance(v, (np.floating, float)):
            return float(v)
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, (np.ndarray, list)):
            return [_clean_json_val(x) for x in v]
        return v

    summary_path = os.path.join(out_dir, "summary.json")
    summary_data = {
        "schema_version": "3.0",
        "config": hybrid_ckpt["config"],
        "meta": hybrid_ckpt["meta"],
        "hybrid_metrics": {k: _clean_json_val(v) for k, v in h_metrics.items()},
        "baseline_metrics": {k: _clean_json_val(v) for k, v in b_metrics.items()},
        "calibration": {
            "temperature": float(T_opt),
            "threshold": _clean_json_val(tau_star),
            "calibrator_type": _cal_type,
        },
        "baseline_calibration": {
            "temperature": float(T_opt_b),
            "threshold": _clean_json_val(tau_star_b),
            "calibrator_type": baseline_calibrator_type,
        },
        "hybrid": {k: _clean_json_val(v) for k, v in h_metrics.items()},
        "baseline": {k: _clean_json_val(v) for k, v in b_metrics.items()},
        "best_epoch_hybrid": int(best_h_epoch),
        "best_val_auc_hybrid": float(best_h_auc),
        "best_val_loss_hybrid": float(best_h_loss),
        "best_epoch_baseline": int(best_b_epoch),
        "best_val_auc_baseline": float(best_b_auc),
        "best_val_loss_baseline": float(best_b_loss),
    }
    if progressive:
        actual_pdepths = progressive_depths if progressive_depths is not None else [2, 4, 6]
        prog_eps = epochs_per_stage if epochs_per_stage is not None else max(1, epochs // len(actual_pdepths))
        summary_data["progressive"] = {
            "depth_schedule": actual_pdepths,
            "freeze_mode": freeze_mode,
            "q_init_std": q_init_std,
            "epochs_per_stage": prog_eps,
            "finetune_epochs": finetune_epochs,
            "stage_transitions": stage_transitions,
        }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
    print(f"[summary] Saved metrics to {summary_path}")

    # ── Generate Diagnostic Plots (Decoupled High-Throughput Suite) ────────
    if not skip_plots:
        print(f"\n{'='*70}\n  GENERATING DIAGNOSTIC PLOTS ({out_dir}/)\n{'='*70}")
        generate_all_diagnostic_plots(
            hybrid_test_cal=hybrid_test_cal,
            baseline_test=baseline_test,
            meta=meta,
            out_dir=out_dir,
            hybrid_hist=hybrid_hist,
            baseline_hist=baseline_hist,
            hybrid_model=hybrid,
            baseline_model=baseline,
            test_loader=test_loader,
            raw_test_loader=raw_test_loader,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            n_qubits=n_qubits,
            n_layers=n_layers,
            criterion=criterion,
            all_grad_norms_vqc=all_grad_norms_vqc,
            all_grad_norms_adapter=all_grad_norms_adapter,
            all_grad_norms_head=all_grad_norms_head,
            kta_history=kta_history,
            entropy_history=entropy_history,
            epoch_times_hybrid=epoch_times_hybrid,
            epoch_times_baseline=epoch_times_baseline,
            parallel=True,
            plot_workers=plot_workers,
            fast_eval=fast_eval,
            skip_plots=skip_plots,
        )
    else:
        print(f"\n[plots] Skipping diagnostic plots (--skip_plots enabled). Metrics saved in {out_dir}/summary.json")
    if progressive and stage_transitions:
        prog_plot_path = os.path.join(out_dir, "progressive_learning_trajectory.png")
        try:
            plot_progressive_learning_trajectory(
                hybrid_hist=hybrid_hist,
                baseline_hist=baseline_hist,
                all_grad_norms_vqc=all_grad_norms_vqc,
                entropy_history=entropy_history,
                stage_transitions=stage_transitions,
                n_qubits=n_qubits,
                save_path=prog_plot_path,
            )
            print(f"[plots] Saved Progressive Learning Trajectory to {prog_plot_path}")
        except Exception as e:
            print(f"[warn] Failed to generate progressive learning trajectory plot: {e}")
    if oct_cost_matrix is not None:
        np.save(os.path.join(out_dir, "cost_sensitive_preds.npy"), hybrid_test_preds_cal)

    if benchmark_mode:
        print(f"\n{'='*70}")
        print(f"  PERFORMANCE BENCHMARK PROFILING REPORT")
        print(f"{'='*70}")
        avg_h_time = float(np.mean(epoch_times_hybrid)) if epoch_times_hybrid else 0.0
        avg_b_time = float(np.mean(epoch_times_baseline)) if epoch_times_baseline else 0.0
        n_train_samples = len(train_loader.dataset) if hasattr(train_loader, "dataset") else 1
        h_throughput = n_train_samples / max(avg_h_time, 1e-6)
        b_throughput = n_train_samples / max(avg_b_time, 1e-6)
        print(f"  Hybrid Mean Epoch Time   : {avg_h_time:.3f} s ({h_throughput:.1f} samples/s)")
        print(f"  Baseline Mean Epoch Time : {avg_b_time:.3f} s ({b_throughput:.1f} samples/s)")
        if torch.cuda.is_available():
            vram_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
            vram_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
            print(f"  Peak GPU VRAM Allocated  : {vram_allocated:.1f} MB (Reserved: {vram_reserved:.1f} MB)")
        print(f"  Entanglement Tracking    : {len(entropy_history)} evaluations completed ({entropy_samples} samples/eval)")
        print(f"{'='*70}\n")

    print(f"\n[done] All training, evaluation, and diagnostic outputs saved to: {out_dir}/\n")
    return {
        "hybrid_metrics": h_metrics,
        "baseline_metrics": b_metrics,
        "mcnemar_p": p_val,
        "out_dir": out_dir,
    }


# ═══════════════════════════════════════════════════════════════════════════
# K-FOLD STRATIFIED CROSS-VALIDATION ENGINE (Extracted to kfold.py)
# ═══════════════════════════════════════════════════════════════════════════
def run_kfold_training(*args, **kwargs):
    try:
        from kfold import run_kfold_training as _rkt
    except ImportError:
        from qml_models.kfold import run_kfold_training as _rkt
    return _rkt(*args, **kwargs)


def run_continual_learning(
    dataset_sequence: List[str] = ["pneumoniamnist", "breastmnist"],
    epochs_per_task: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    n_qubits: int = 4,
    n_layers: int = 3,
    lambda_ewc: float = 400.0,
    output_dir: str = "outputs/continual",
    diff_method: str = "adjoint",
    use_lora: bool = False,
    image_size: int = 224,
    pretrain_epochs: int = 30,
    qdevice: str = "auto",
):
    """Sequential training across MedMNIST subsets with Geometric QEWC."""
    device = get_torch_device()
    os.makedirs(output_dir, exist_ok=True)

    all_n_classes = [DATASET_CONFIG[d]["n_classes"] for d in dataset_sequence]
    max_classes = max(all_n_classes)
    in_channels = DATASET_CONFIG[dataset_sequence[0]]["in_channels"]

    # A shared classifier can only support continual learning when class index
    # semantics are identical across tasks.  PneumoniaMNIST and BreastMNIST
    # are both binary, but class 1 means different diseases; comparing their
    # retention accuracies through one head is clinically meaningless.
    class_orders = [
        tuple(DATASET_CONFIG[d].get("label_map", {}).get(i, str(i)) for i in range(DATASET_CONFIG[d]["n_classes"]))
        for d in dataset_sequence
    ]
    if any(order != class_orders[0] for order in class_orders[1:]):
        raise ValueError(
            "Continual learning requires identical class-order semantics across tasks. "
            f"Received: {dict(zip(dataset_sequence, class_orders))}. "
            "Use task-specific heads for unrelated disease taxonomies."
        )

    hybrid = HybridCNNVQC(
        in_channels, max_classes, n_qubits, n_layers,
        diff_method=diff_method, use_lora=use_lora, qdevice=qdevice
    ).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, hybrid.parameters()), lr=lr)

    fisher = None
    prev_params = None
    retention_matrix = np.zeros((len(dataset_sequence), len(dataset_sequence)))
    test_loaders = {}

    for task_idx, dataset_name in enumerate(dataset_sequence):
        print(f"\n{'='*70}")
        print(f"  CONTINUAL TASK {task_idx + 1}/{len(dataset_sequence)}: {dataset_name} ({image_size}x{image_size})")
        print(f"{'='*70}")

        train_loader, val_loader, test_loader, meta = get_medmnist_dataloaders(
            dataset_name, batch_size=batch_size, balanced_sampling=True, size=image_size
        )
        n_classes = meta["n_classes"]
        test_loaders[dataset_name] = (test_loader, meta)

        # Continual tasks may be resumed after a memory-saving offload.  Raw
        # image extraction always requires the encoder; cached feature
        # training below does not.  Make the lifecycle explicit rather than
        # allowing a later task to fail inside extract_features().
        if getattr(hybrid.backbone, "vision_encoder", None) is None:
            hybrid.load_vision_encoder()

        cache_dir = os.path.join(output_dir, f"cache_{image_size}", dataset_name)
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[cache] Extracting continual features for {dataset_name} ({cache_dir})...")
        train_loader = extract_and_cache_features(
            hybrid, train_loader, device, os.path.join(cache_dir, "train.pt"),
            cache_metadata={"dataset_name": dataset_name, "image_size": int(image_size), "split": "train", "continual_task": task_idx},
        )
        test_loader = extract_and_cache_features(
            hybrid, test_loader, device, os.path.join(cache_dir, "test.pt"),
            cache_metadata={"dataset_name": dataset_name, "image_size": int(image_size), "split": "test", "continual_task": task_idx},
        )
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Pre-train adapter on new dataset (with disk cache lookup)
        task_adapter_dir = get_adapter_dir(dataset_name)
        task_adapter_ckpt = get_adapter_checkpoint_path(
            dataset=dataset_name,
            n_qubits=n_qubits,
            use_lora=use_lora,
            lora_rank=4,
        )
        if task_adapter_ckpt.exists():
            print(f"\n[adapter] Found pre-trained CQ-Adapter checkpoint for task '{dataset_name}': {task_adapter_ckpt}")
            load_adapter_checkpoint(hybrid.backbone, task_adapter_ckpt, device=device)
            print(f"[adapter] Loaded weights into CQ-Adapter (skipping pre-training).\n")
        else:
            print(f"\n[adapter] Pre-training adapter on task dataset '{dataset_name}'...")
            pretrain_cq_adapter(hybrid.backbone, train_loader, device, epochs=pretrain_epochs, out_dir=str(task_adapter_dir))
            save_adapter_checkpoint(
                hybrid.backbone,
                task_adapter_ckpt,
                metadata={"dataset": dataset_name, "n_qubits": n_qubits, "epochs": pretrain_epochs, "use_lora": use_lora}
            )
            print(f"[adapter] Saved pre-trained weights and diagnostic plots to: {task_adapter_dir}\n")

        class_weights = compute_class_weights(train_loader, n_classes, device)
        criterion = FocalLoss(gamma=2.0, weight=class_weights)

        # Training loop with QEWC penalty
        for epoch in range(1, epochs_per_task + 1):
            hybrid.train()
            running_loss = 0.0
            n_total = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.squeeze().long().to(device)

                optimizer.zero_grad()
                logits = hybrid(images)
                # Use only first n_classes outputs
                logits = logits[:, :n_classes]
                loss = criterion(logits, labels)

                # QEWC penalty
                if fisher is not None and prev_params is not None:
                    ewc_loss = qewc_penalty(hybrid, fisher, prev_params, lambda_ewc)
                    loss = loss + ewc_loss

                loss.backward()
                nn.utils.clip_grad_norm_(hybrid.parameters(), 5.0)
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                n_total += images.size(0)

            if epoch % 5 == 0 or epoch == epochs_per_task:
                print(f"   Epoch {epoch}/{epochs_per_task}: loss={running_loss/n_total:.4f}")

        # Compute Fisher for this task
        fisher = compute_fisher_diagonal(hybrid, train_loader, criterion, device,
                                          n_classes, n_samples=200)
        prev_params = {n: p.data.clone() for n, p in hybrid.named_parameters()
                       if p.requires_grad}

        # Evaluate on ALL previous tasks
        for eval_idx, eval_name in enumerate(dataset_sequence[:task_idx + 1]):
            eval_loader, eval_meta = test_loaders[eval_name]
            # Use cached version for eval as well
            cache_dir = os.path.join(output_dir, f"cache_{image_size}", eval_name)
            eval_loader = extract_and_cache_features(
                hybrid, eval_loader, device, os.path.join(cache_dir, "test.pt"),
                cache_metadata={"dataset_name": eval_name, "image_size": int(image_size), "split": "test", "continual_task": eval_idx},
            )
            eval_n_classes = eval_meta["n_classes"]
            eval_criterion = FocalLoss(gamma=2.0)
            res = evaluate(hybrid, eval_loader, eval_criterion, device, eval_n_classes, max_samples=1000)
            retention_matrix[task_idx, eval_idx] = res["accuracy"]
            print(f"   Retention on {eval_name}: {res['accuracy']:.4f}")

    # Fix E-05: Wire plot_qewc_retention into continual learning
    try:
        plot_qewc_retention(retention_matrix, dataset_sequence, output_dir)
    except Exception as e:
        print(f"[warn] plot_qewc_retention skipped: {e}")

    hybrid.offload_vision_encoder()

    print(f"\n[done] Continual learning completed across {len(dataset_sequence)} tasks.")
    print(f"  Retention matrix saved to: {output_dir}/")
    return retention_matrix


def run_eval_only(
    dataset_name: str = "pneumoniamnist",
    output_dir: str = "outputs",
    batch_size: int = 128,
    image_size: int = 224,
    qdevice: str = "auto",
    threshold_criterion: str = "youden",
    target_specificity: float = 0.90,
    cost_fp: float = 1.5,
    recalibrate_threshold: bool = False,
    fast_eval: bool = False,
    skip_plots: bool = False,
    plot_workers: int = 4,
):
    """Load existing checkpoints and training history to generate all diagnostic plots without retraining."""
    device = get_torch_device()
    out_dir = os.path.join(output_dir, dataset_name)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    h_path = os.path.join(ckpt_dir, "hybrid_model.pt")
    b_path = os.path.join(ckpt_dir, "baseline_model.pt")

    if not os.path.exists(h_path) or not os.path.exists(b_path):
        print(f"[eval_only] Error: Checkpoints not found in {ckpt_dir}")
        return

    print(f"\n{'='*70}")
    print(f"  EVAL-ONLY & DIAGNOSTIC PLOT GENERATION ({out_dir}/)")
    print(f"{'='*70}")

    hybrid_ckpt = torch.load(h_path, map_location=device, weights_only=True)
    baseline_ckpt = torch.load(b_path, map_location=device, weights_only=True)

    h_cfg = hybrid_ckpt.get("config", {})
    b_cfg = baseline_ckpt.get("config", {})
    in_channels = h_cfg.get("in_channels", 3)
    n_classes = h_cfg.get("n_classes", 2)
    n_qubits = h_cfg.get("n_qubits", 4)
    n_layers = h_cfg.get("n_layers", 3)
    dropout = h_cfg.get("dropout", 0.3)
    use_lora = h_cfg.get("use_lora", False)
    diff_method = h_cfg.get("diff_method", "backprop")
    chosen_qdevice = qdevice if qdevice != "auto" else h_cfg.get("qdevice", "auto")
    dual_basis_only = h_cfg.get("dual_basis_only", h_cfg.get("dual_basis_readout", False))
    use_qaf_attention = h_cfg.get("use_qaf_attention", False)
    clinical_hard = h_cfg.get("clinical_hard", False)
    preprocess_mode = h_cfg.get("preprocess_mode", "biomedclip_normalized")
    p_drop = h_cfg.get("p_drop", 0.10)
    q_init_std = h_cfg.get("q_init_std", 0.01)

    qw = hybrid_ckpt.get("model_state_dict", {}).get("qlayer.weights", None)
    if qw is not None and qw.ndim == 3:
        genome = None
        n_layers = qw.shape[0]
        n_qubits = qw.shape[1]
    else:
        qnas_genome_dict = h_cfg.get("qnas_genome", None)
        genome = None
        if qnas_genome_dict is not None:
            try:
                from qnas import CircuitGenome
                genome = CircuitGenome.from_dict(qnas_genome_dict)
            except Exception:
                try:
                    from qml_models.qnas import CircuitGenome
                    genome = CircuitGenome.from_dict(qnas_genome_dict)
                except Exception:
                    pass
        else:
            qnas_file = os.path.join(ckpt_dir, "qnas_genome.json")
            if os.path.exists(qnas_file):
                try:
                    from qnas import load_genome
                    genome = load_genome(qnas_file)
                except Exception:
                    try:
                        from qml_models.qnas import load_genome
                        genome = load_genome(qnas_file)
                    except Exception:
                        pass

    init_depth = len(genome.layers) if genome is not None and hasattr(genome, "layers") else n_layers
    hybrid = HybridCNNVQC(in_channels, n_classes, n_qubits, init_depth, dropout=dropout,
                          diff_method=diff_method, p_drop=p_drop,
                          use_lora=use_lora, genome=genome, q_init_std=q_init_std,
                          dual_basis_only=dual_basis_only,
                          use_qaf_attention=use_qaf_attention,
                          clinical_hard=clinical_hard,
                          preprocess_mode=preprocess_mode,
                          qdevice=chosen_qdevice).to(device)

    # Grow layers if checkpoint weights indicate progressive expansion (e.g. L=2 -> 4 -> 6)
    h_sd = hybrid_ckpt.get("model_state_dict", {})
    if "qlayer.weights" in h_sd and h_sd["qlayer.weights"].ndim == 1:
        target_numel = h_sd["qlayer.weights"].numel()
        while sum(p.numel() for n, p in hybrid.qlayer.named_parameters() if "weights" in n) < target_numel:
            hybrid.grow_layers(new_depth=hybrid.n_layers + 2)

    b_weights = baseline_ckpt.get("model_state_dict", {})
    b_mlp_h = b_weights["mlp.0.weight"].shape[0] if "mlp.0.weight" in b_weights else None
    b_mlp_in = b_weights["mlp.0.weight"].shape[1] if "mlp.0.weight" in b_weights else None
    baseline = ClassicalBaseline(
        in_channels, n_classes, n_qubits, n_layers=n_layers, dropout=dropout, use_lora=use_lora,
        mlp_hidden_dim=b_mlp_h, mlp_in_dim=b_mlp_in,
        dual_basis_only=b_cfg.get("dual_basis_only", dual_basis_only),
        clinical_hard=b_cfg.get("clinical_hard", clinical_hard),
        genome=genome,
        use_qaf_attention=b_cfg.get("use_qaf_attention", use_qaf_attention),
        preprocess_mode=b_cfg.get("preprocess_mode", preprocess_mode),
    ).to(device)

    h_missing, h_unexpected = hybrid.load_state_dict(hybrid_ckpt["model_state_dict"], strict=False)
    b_missing, b_unexpected = baseline.load_state_dict(baseline_ckpt["model_state_dict"], strict=False)
    h_critical = [k for k in h_missing if not k.startswith("backbone.vision_encoder.")]
    b_critical = [k for k in b_missing if not k.startswith("backbone.vision_encoder.")]
    if h_critical or b_critical or h_unexpected or b_unexpected:
        raise RuntimeError(
            "Eval-only checkpoint architecture mismatch: "
            f"hybrid_missing={h_critical[:5]}, hybrid_unexpected={h_unexpected[:5]}, "
            f"baseline_missing={b_critical[:5]}, baseline_unexpected={b_unexpected[:5]}"
        )
    hybrid.to(device)
    baseline.to(device)

    train_loader, val_loader, test_loader, meta = get_medmnist_dataloaders(
        dataset_name, batch_size=batch_size, download_root="./data", size=image_size
    )
    meta["dataset_name"] = dataset_name
    meta["n_classes"] = n_classes
    if "label_map" not in meta or not meta["label_map"]:
        info_label = meta.get("info", {}).get("label", {})
        if info_label:
            meta["label_map"] = {int(k): v for k, v in info_label.items()}
        elif dataset_name in DATASET_CONFIG and "label_map" in DATASET_CONFIG[dataset_name]:
            meta["label_map"] = DATASET_CONFIG[dataset_name]["label_map"]
    raw_test_loader = test_loader
    cache_dir = os.path.join(out_dir, f"cache_{image_size}")
    if not os.path.exists(cache_dir):
        cache_dir = os.path.join(out_dir, "cache")
    eval_cache_common = {
        "dataset_name": dataset_name,
        "image_size": int(image_size),
        "split_source": "eval_only",
    }
    train_loader = extract_and_cache_features(
        hybrid, train_loader, device, os.path.join(cache_dir, "train.pt"),
        cache_metadata={**eval_cache_common, "split": "train"},
    )
    val_loader = extract_and_cache_features(
        hybrid, val_loader, device, os.path.join(cache_dir, "val.pt"),
        cache_metadata={**eval_cache_common, "split": "val"},
    )
    test_loader = extract_and_cache_features(
        hybrid, test_loader, device, os.path.join(cache_dir, "test.pt"),
        cache_metadata={**eval_cache_common, "split": "test"},
    )
    hybrid.offload_vision_encoder()
    baseline.offload_vision_encoder()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    criterion = nn.CrossEntropyLoss()
    print("[eval_only] Evaluating models on test set...")
    hybrid_test = evaluate(hybrid, test_loader, criterion, device, n_classes, max_samples=1000)
    baseline_test = evaluate(baseline, test_loader, criterion, device, n_classes, max_samples=1000)

    h_meta = hybrid_ckpt.get("meta", {})
    T_opt = float(h_meta.get("temperature", 1.0))
    tau_star = float(h_meta.get("threshold", 0.5))

    h_logits_test = hybrid_test["logits"].float()
    probs_temp_t = F.softmax(h_logits_test / max(T_opt, 1e-6), dim=1)
    h_calibration = hybrid_ckpt.get("calibration") or {}
    h_cal_state = hybrid_ckpt.get("calibrator_state_dict")
    if h_cal_state:
        h_calibrator = QuantumBetaCalibrator() if hybrid_ckpt.get("calibrator_type", h_calibration.get("calibrator_type")) == "beta" else QuantumMultiClassCalibrator(n_classes=n_classes)
        h_calibrator.load_state_dict(h_cal_state, strict=True)
        with torch.no_grad():
            probs_temp_t = h_calibrator(probs_temp_t)
    probs_temp = probs_temp_t.numpy()

    if n_classes == 2 and (recalibrate_threshold or threshold_criterion != "youden" or target_specificity != 0.90):
        print(f"\n[eval_only] Optimizing decision threshold on validation set (Criterion: '{threshold_criterion}')...")
        val_logits, val_labels_arr = _collect_val_logits(hybrid, val_loader, device)
        val_probs_temp = F.softmax(val_logits / max(T_opt, 1e-6), dim=1)
        if h_cal_state:
            with torch.no_grad():
                val_probs_temp = h_calibrator(val_probs_temp)
        val_labels_np = val_labels_arr.numpy()
        tau_star, _ = optimize_decision_threshold(
            val_probs_temp.numpy(), val_labels_np,
            criterion=threshold_criterion,
            target_specificity=target_specificity,
            cost_fp=cost_fp,
        )

    if n_classes == 2:
        p_pos = probs_temp[:, 1]
        preds_cal = (p_pos >= tau_star).astype(int)
    else:
        preds_cal = np.argmax(probs_temp, axis=1)
        policy = h_cfg.get("decision_policy", {})
        if policy.get("type") == "cost_matrix":
            preds_cal = predict_with_cost_matrix(probs_temp, policy["cost_matrix"])

    hybrid_test_cal = dict(hybrid_test)
    hybrid_test_cal["probs"] = probs_temp
    hybrid_test_cal["preds"] = preds_cal
    hybrid_test_cal["model"] = hybrid  # Fix E-03/E-04: Pass model for decision boundary & kernel gram
    if "trained_q_weights" in hybrid_test:
        hybrid_test_cal["trained_q_weights"] = hybrid_test["trained_q_weights"]

    # Calibrate / threshold baseline for fair comparison
    val_logits_b, val_labels_arr_b = _collect_val_logits(baseline, val_loader, device)
    b_meta = baseline_ckpt.get("meta", {})
    tau_star_b = float(b_meta.get("threshold", 0.5))
    T_opt_b = float(b_meta.get("temperature", 1.0))
    val_probs_b_t = F.softmax(val_logits_b / max(T_opt_b, 1e-6), dim=1)
    b_calibration = baseline_ckpt.get("calibration") or {}
    b_cal_state = baseline_ckpt.get("calibrator_state_dict")
    if b_cal_state:
        b_calibrator = QuantumBetaCalibrator() if baseline_ckpt.get("calibrator_type", b_calibration.get("calibrator_type")) == "beta" else QuantumMultiClassCalibrator(n_classes=n_classes)
        b_calibrator.load_state_dict(b_cal_state, strict=True)
        with torch.no_grad():
            val_probs_b_t = b_calibrator(val_probs_b_t)
    val_probs_b = val_probs_b_t.numpy()

    if n_classes == 2 and (recalibrate_threshold or threshold_criterion != "youden" or target_specificity != 0.90 or "threshold" not in b_meta):
        tau_star_b, _ = optimize_decision_threshold(
            val_probs_b, val_labels_arr_b.numpy(),
            criterion=threshold_criterion,
            target_specificity=target_specificity,
            cost_fp=cost_fp,
        )
        print(f"[eval_only] Optimized Baseline decision threshold on validation set: τ*_b = {tau_star_b:.4f}")

    raw_probs_b = torch.tensor(baseline_test["probs"]).float()
    b_logits_test = baseline_test["logits"].float()
    probs_temp_b_t = F.softmax(b_logits_test / max(T_opt_b, 1e-6), dim=1)
    if b_cal_state:
        with torch.no_grad():
            probs_temp_b_t = b_calibrator(probs_temp_b_t)
    probs_temp_b = probs_temp_b_t.numpy()
    if n_classes == 2:
        preds_cal_b = (probs_temp_b[:, 1] >= tau_star_b).astype(int)
    else:
        preds_cal_b = np.argmax(probs_temp_b, axis=1)
        policy_b = b_cfg.get("decision_policy", {})
        if policy_b.get("type") == "cost_matrix":
            preds_cal_b = predict_with_cost_matrix(probs_temp_b, policy_b["cost_matrix"])

    baseline_test = dict(baseline_test)
    baseline_test["probs"] = probs_temp_b
    baseline_test["preds"] = preds_cal_b

    hist_path = os.path.join(out_dir, "training_history.json")
    if os.path.exists(hist_path):
        with open(hist_path, "r") as f:
            hist = json.load(f)
        hybrid_hist = hist.get("hybrid", {})
        baseline_hist = hist.get("baseline", {})
    else:
        hybrid_hist, baseline_hist = {}, {}

    epochs = h_meta.get("epochs", len(hybrid_hist.get("train_loss", [50])))
    epoch_times_hybrid = hybrid_hist.get("epoch_times", [1.0] * epochs)
    epoch_times_baseline = baseline_hist.get("epoch_times", [0.5] * epochs)
    all_grad_norms_vqc = hybrid_hist.get("grad_norms_vqc", [0.01] * epochs)
    all_grad_norms_adapter = hybrid_hist.get("grad_norms_adapter", [0.01] * epochs)
    all_grad_norms_head = hybrid_hist.get("grad_norms_head", [0.01] * epochs)
    kta_history = hybrid_hist.get("kta_history", [0.1] * (epochs // 5 + 1))
    entropy_history = hybrid_hist.get("entropy_history", [1.0] * (epochs // 5 + 1))

    if not skip_plots:
        print(f"\n[eval_only] Generating all diagnostic plots in {out_dir}/...")
        generate_all_diagnostic_plots(
            hybrid_test_cal=hybrid_test_cal,
            baseline_test=baseline_test,
            meta=meta,
            out_dir=out_dir,
            hybrid_hist=hybrid_hist,
            baseline_hist=baseline_hist,
            hybrid_model=hybrid,
            baseline_model=baseline,
            test_loader=test_loader,
            raw_test_loader=raw_test_loader,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            n_qubits=n_qubits,
            n_layers=n_layers,
            criterion=criterion,
            all_grad_norms_vqc=all_grad_norms_vqc,
            all_grad_norms_adapter=all_grad_norms_adapter,
            all_grad_norms_head=all_grad_norms_head,
            kta_history=kta_history,
            entropy_history=entropy_history,
            epoch_times_hybrid=epoch_times_hybrid,
            epoch_times_baseline=epoch_times_baseline,
            parallel=True,
            plot_workers=plot_workers,
            fast_eval=fast_eval,
            skip_plots=skip_plots,
        )
    else:
        print(f"\n[eval_only] Skipping diagnostic plots (--skip_plots enabled). Metrics saved in {out_dir}/summary.json")
    policy = h_cfg.get("decision_policy", {})
    if policy.get("type") == "cost_matrix":
        np.save(os.path.join(out_dir, "cost_sensitive_preds.npy"), hybrid_test_cal["preds"])

    h_metrics = compute_clinical_metrics(
        hybrid_test_cal["labels"], hybrid_test_cal["preds"], hybrid_test_cal["probs"], n_classes
    )
    b_metrics = compute_clinical_metrics(
        baseline_test["labels"], baseline_test["preds"], baseline_test["probs"], n_classes
    )

    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY & CLINICAL METRICS (MedMNIST v2)")
    print("=" * 70)
    print(f"  Hybrid Accuracy:     {h_metrics['accuracy']:.4f}  |  Baseline: {b_metrics['accuracy']:.4f}")
    print(f"  Hybrid Macro Spec:   {h_metrics['specificity']:.4f}  |  Baseline: {b_metrics['specificity']:.4f}")
    print(f"  Hybrid Macro Sens:   {h_metrics['sensitivity']:.4f}  |  Baseline: {b_metrics['sensitivity']:.4f}")
    print(f"  Hybrid Macro F1:     {h_metrics['f1_macro']:.4f}  |  Baseline: {b_metrics['f1_macro']:.4f}")
    print(f"  Hybrid AUROC:        {h_metrics['auroc']:.4f}  |  Baseline: {b_metrics['auroc']:.4f}")
    print(f"  Hybrid AUPRC:        {h_metrics['auprc']:.4f}  |  Baseline: {b_metrics['auprc']:.4f}")
    print("=" * 70 + "\n")

    def _clean_json_val(v):
        if isinstance(v, (np.floating, float)):
            return float(v)
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, (np.ndarray, list)):
            return [_clean_json_val(x) for x in v]
        return v

    summary_data = {
        "hybrid": {k: _clean_json_val(v) for k, v in h_metrics.items()},
        "baseline": {k: _clean_json_val(v) for k, v in b_metrics.items()},
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    print(f"\n[done] All diagnostic plots successfully generated in {out_dir}/\n")


# ═══════════════════════════════════════════════════════════════════════════
# 10.  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Hybrid CNN-VQC Training Engine for MedMNIST v2"
    )
    parser.add_argument("--dataset", type=str, default="pneumoniamnist",
                        choices=list(DATASET_CONFIG.keys()),
                        help="MedMNIST v2 dataset name")
    parser.add_argument("--eval_only", action="store_true",
                        help="Load existing checkpoints and generate all diagnostic plots without retraining")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate for Adam")
    parser.add_argument("--n_qubits", type=int, default=4,
                        help="Number of qubits in VQC")
    parser.add_argument("--n_layers", type=int, default=3,
                        help="Number of StronglyEntanglingLayers")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate for fully connected layers")
    parser.add_argument("--diff_method", type=str, default="auto",
                        help="PennyLane differentiation method: 'auto' (recommended: backprop for N<=10, adjoint for N>10), 'backprop', or 'adjoint'")
    parser.add_argument("--mixup_alpha", type=float, default=0.4,
                        help="Beta distribution alpha for quantum mixup")
    parser.add_argument("--pre_quantum_mixup", action="store_true",
                        help="Use legacy Euclidean pre-quantum mixup instead of expectation-space quantum mixup (Fix: FL-07)")
    parser.add_argument("--phase_eps", type=float, default=0.05,
                        help="Phase perturbation epsilon")
    parser.add_argument("--lambda_kta", type=float, default=0.05,
                        help="Kernel-Target Alignment (KTA) regularization weight (Fix: FL-09, default: 0.05)")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for plots and checkpoints")
    parser.add_argument("--lambda_ewc", type=float, default=400.0,
                        help="QEWC regularisation strength")
    parser.add_argument("--use_lora", action="store_true",
                        help="Enable LoRA-Q low-rank adapter (rank-4 decomposition, 86.7 percent parameter reduction)")
    parser.add_argument("--use_qng", action="store_true",
                        help="Enable Riemannian Quantum Natural Gradient (QNG) optimization on quantum weights")
    parser.add_argument("--use_qsam", action="store_true",
                        help="Enable Quantum Sharpness-Aware Minimization (Q-SAM) for flat-basin generalization")
    parser.add_argument("--use_qswa", action="store_true",
                        help="Enable Quantum Stochastic Weight Averaging (Q-SWA) in final 30 percent of training")
    parser.add_argument("--lambda_infonce", type=float, default=0.0,
                        help="Weight for Hilbert space InfoNCE quantum fidelity contrastive loss (default: 0.0)")
    parser.add_argument("--continual", action="store_true",
                        help="Run continual learning across multiple datasets")
    parser.add_argument("--continual_datasets", type=str, nargs="+",
                        default=["pneumoniamnist", "breastmnist"],
                        help="Dataset sequence for continual learning")
    parser.add_argument("--image_size", type=int, default=224,
                        choices=[28, 64, 128, 224],
                        help="Input image resolution for MedMNIST v2 (default: 224 for native high-res ViT input)")
    parser.add_argument("--pretrain_epochs", type=int, default=30,
                        help="Number of pre-training epochs for CQ adapter (default: 30)")
    parser.add_argument("--ansatz_mode", type=str, default="auto",
                        choices=["auto", "qnas_static", "progressive_dense", "standard_static", "qnas_progressive"],
                        help="Quantum ansatz design strategy: 'qnas_static' (Route A: sparse QNAS + static CV), "
                             "'progressive_dense' (Route B: dense StronglyEntangling + progressive CV), "
                             "'qnas_progressive' (combined: search base sparse ansatz and progressively grow), "
                             "'standard_static' (fixed depth standard ansatz), or 'auto' (inferred from --use_qnas or --progressive).")
    parser.add_argument("--allow_qnas_progressive", action="store_true",
                        help="Allow running QNAS and Progressive Layerwise Training together without mutual-exclusion error.")
    parser.add_argument("--use_qnas", action="store_true",
                        help="Enable Quantum Neural Architecture Search (QNAS) for automated sparse ansatz discovery")
    parser.add_argument("--qnas_generations", type=int, default=6,
                        help="Number of evolutionary search generations for QNAS (default: 6)")
    parser.add_argument("--qnas_pop_size", type=int, default=16,
                        help="Population size for QNAS evolutionary pool (default: 16)")
    parser.add_argument("--qnas_min_cnot", type=int, default=2,
                        help="Minimum CNOT gate count constraint for QNAS genome to enforce non-zero entanglement (default: 2)")
    parser.add_argument("--qnas_max_cnot", type=int, default=6,
                        help="Maximum CNOT gate budget constraint for QNAS genome (default: 6, >=40% reduction)")
    parser.add_argument("--qnas_target_mw", "--target_mw", type=float, default=0.45,
                        help="Target Meyer-Wallach entanglement Q_MW for QNAS fitness (default: 0.45)")
    parser.add_argument("--qnas_lambda_cnot", "--lambda_cnot", type=float, default=0.01,
                        help="CNOT gate penalty weight lambda_cnot for QNAS fitness (default: 0.01)")
    parser.add_argument("--qnas_genome_path", "--genome_path", type=str, default=None,
                        help="Path to pre-computed QNAS genome JSON to bypass evolutionary search (default: None)")
    parser.add_argument("--qdevice", "--quantum_device", type=str, default="auto",
                        choices=["auto", "default.qubit", "lightning.qubit", "lightning.gpu"],
                        help="PennyLane quantum simulator device: 'default.qubit', 'lightning.qubit', 'lightning.gpu', or 'auto' (default: auto)")
    parser.add_argument("--force_pretrain", action="store_true",
                        help="Force re-running CQ-Adapter pre-training even if cached weights exist in qml_models/adapter/<dataset>/")
    parser.add_argument("--adapter_checkpoint", type=str, default=None,
                        help="Explicit path to pre-trained CQ-Adapter checkpoint (overrides auto-detection)")
    parser.add_argument("--progressive", action="store_true",
                        help="Enable Progressive Layerwise Training (Skolik et al., 2021) to mitigate barren plateaus")
    parser.add_argument("--progressive_depths", type=int, nargs="+", default=[2, 4, 6],
                        help="Depth schedule for progressive layerwise training (default: 2 4 6)")
    parser.add_argument("--epochs_per_stage", type=int, default=None,
                        help="Number of training epochs per depth stage in progressive training (default: auto)")
    parser.add_argument("--finetune_epochs", type=int, default=5,
                        help="Global harmonization epochs with all layers unfrozen (default: 5)")
    parser.add_argument("--freeze_mode", type=str, default="hard", choices=["hard", "none"],
                        help="Freezing mode for previous layers during progressive growth: 'hard' or 'none' (default: hard)")
    parser.add_argument("--q_init_std", type=float, default=0.01,
                        help="Standard deviation for Gaussian initialization near identity for newly added quantum layers (default: 0.01)")
    parser.add_argument("--balanced_sampler", action="store_true",
                        help="Enable WeightedRandomSampler for 50/50 class-balanced mini-batches (elevates minority gradients)")
    parser.add_argument("--threshold_criterion", type=str, default="youden",
                        choices=["youden", "target_specificity", "f_beta", "cost_sensitive", "balanced_eer"],
                        help="Criterion for post-hoc threshold optimization: 'youden', 'target_specificity', 'f_beta', 'cost_sensitive', or 'balanced_eer'")
    parser.add_argument("--target_specificity", type=float, default=0.90,
                        help="Target specificity floor for constrained optimization (default: 0.90)")
    parser.add_argument("--cost_fp", type=float, default=1.5,
                        help="Cost ratio for False Positives relative to False Negatives (default: 1.5)")
    parser.add_argument("--no_asymmetric_loss", action="store_true",
                        help="Disable asymmetric focal loss and use standard symmetric focal loss")
    parser.add_argument("--asym_alpha", "--alpha_per_class", type=str, default=None, dest="asym_alpha",
                        help="Per-class weight alpha for Asymmetric Focal Loss. Pass comma-separated floats (e.g. '1.5,1.0' or '1.0,1.2,2.5,0.8') or single float. Autodetects by dataset/n_classes if omitted.")
    parser.add_argument("--asym_gamma", "--gamma_per_class", type=str, default=None, dest="asym_gamma",
                        help="Per-class focusing parameter gamma for Asymmetric Focal Loss. Pass comma-separated floats (e.g. '1.5,2.0' or '2.0,2.0,1.0,2.5') or single float. Autodetects by dataset/n_classes if omitted.")
    parser.add_argument("--loss_type", type=str, default="focal",
                        choices=["ce", "focal", "asymmetric", "cb_ldam"],
                        help="Loss function type: 'ce' (CrossEntropy), 'focal' (Focal Loss), 'asymmetric' (Asymmetric Focal), 'cb_ldam' (Class-Balanced LDAM with DRW). Default: focal.")
    parser.add_argument("--ldam_max_margin", type=float, default=0.50,
                        help="Maximum margin C_margin for LDAM loss: Delta_c = C / N_c^0.25 (default: 0.50)")
    parser.add_argument("--drw_start_epoch", type=int, default=10,
                        help="Epoch to activate class-balanced Deferred Re-Weighting in CB-LDAM (default: 10)")
    parser.add_argument("--dual_basis_readout", action="store_true",
                        help="Enable dual-basis Pauli [Z, X] readout (2n features) instead of default [Z, ZZ, X] readout (3n-1 features). Required for HQCNN 8-qubit DermaMNIST.")
    parser.add_argument("--use_qaf_attention", action="store_true",
                        help="Enable Quantum Attention-Fourier (QAF) circuit layers: Superpositional Entanglement + Toffoli Attention + Fourier CPhase.")
    parser.add_argument("--recalibrate_threshold", action="store_true",
                        help="Re-optimize decision threshold on validation set during eval_only mode")
    parser.add_argument("--entropy_every_epoch", dest="entropy_every_epoch", action="store_true", default=True,
                        help="Compute Von Neumann entanglement entropy and KTA every single epoch (default: True)")
    parser.add_argument("--no_entropy_every_epoch", dest="entropy_every_epoch", action="store_false",
                        help="Disable per-epoch entropy computation and use periodic intervals")
    parser.add_argument("--entropy_samples", type=int, default=32,
                        help="Number of validation samples for vectorized entanglement entropy and KTA evaluations (default: 32)")
    parser.add_argument("--entropy_interval", type=int, default=1,
                        help="Interval (in epochs) between entanglement entropy evaluations if per-epoch is disabled (default: 1)")
    parser.add_argument("--fast_eval", action="store_true",
                        help="Bypass 4,240 heavy quantum noise simulations (ZNE/noise robustness) and 2D loss landscapes in diagnostic plots to save 5-15 min runtime")
    parser.add_argument("--skip_plots", action="store_true",
                        help="Skip post-training diagnostic plot generation entirely for fast experimentation and sweeps")
    parser.add_argument("--plot_workers", type=int, default=4,
                        help="Number of parallel worker processes for diagnostic plot generation (default: 4)")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True,
                        help="Enable automatic mixed precision (AMP) for classical operations (default: True)")
    parser.add_argument("--no_amp", dest="amp", action="store_false",
                        help="Disable automatic mixed precision")
    parser.add_argument("--cache_in_vram", action="store_true",
                        help="Preload extracted feature tensors directly into GPU VRAM to eliminate host-device PCIe overhead (recommended on GPUs with >=8GB VRAM)")
    parser.add_argument("--qsam_freq", type=int, default=1,
                        help="Frequency of Q-SAM sharpness perturbation steps (default: 1)")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Number of micro-batches to accumulate gradients over (default: 1)")
    parser.add_argument("--benchmark_mode", action="store_true",
                        help="Run benchmark profiling mode and display throughput and VRAM report")
    parser.add_argument("--profile_runtime", action="store_true",
                        help="Profile per-component training and evaluation runtimes")
    parser.add_argument("--kfold", action="store_true",
                        help="Run K-Fold Stratified Cross-Validation with OOF ensembling")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of folds for K-fold cross validation (default: 5)")
    parser.add_argument("--kfold_no_ensemble", action="store_true",
                        help="Disable OOF model ensembling in K-fold evaluation")
    parser.add_argument("--use_cost_matrix", action="store_true",
                        help="Apply clinical cost matrix for Bayes risk minimization (default: False, uses argmax)")
    parser.add_argument("--clinical_hard", action="store_true",
                        help="Enable clinical hardening: restrict CQ-adapter angle range to [-pi/2, pi/2] to eliminate phase-wrapping decision boundary islands.")
    parser.add_argument("--enable_vit_finetuning", action="store_true",
                        help="Enable fine-tuning of top layers of BiomedCLIP ViT backbone before VQC training")
    parser.add_argument("--vit_unfreeze_layers", type=int, default=2,
                        help="Number of top transformer layers to unfreeze in BiomedCLIP ViT (default: 2)")
    parser.add_argument("--vit_finetune_lr", type=float, default=1e-5,
                        help="Learning rate for fine-tuning BiomedCLIP ViT backbone (default: 1e-5)")
    parser.add_argument("--vit_epochs", type=int, default=5,
                        help="Number of epochs for fine-tuning BiomedCLIP ViT backbone (default: 5)")
    parser.add_argument("--biomedclip_dir", type=str, default="biomedclip",
                        help="Directory storing fine-tuned BiomedCLIP ViT checkpoints (default: biomedclip)")
    parser.add_argument("--lambda_decorr", type=float, default=0.01,
                        help="Weight for anti-shortcut feature decorrelation loss (default: 0.01)")
    parser.add_argument("--lambda_saliency", type=float, default=0.05,
                        help="Weight for anti-shortcut saliency concentration loss (default: 0.05)")

    args = parser.parse_args()

    # Resolve and validate ansatz_mode
    if args.allow_qnas_progressive or args.ansatz_mode == "qnas_progressive":
        args.ansatz_mode = "qnas_progressive"
        args.use_qnas = True
        args.progressive = True
        print("[ansatz_mode] Activated combined 'qnas_progressive' mode: QNAS searches base ansatz, then progressively scales depth.")
    elif args.ansatz_mode == "auto":
        if args.use_qnas and args.progressive:
            args.ansatz_mode = "qnas_progressive"
            print("[ansatz_mode] Both --use_qnas and --progressive detected: automatically activating combined 'qnas_progressive' mode.")
        elif args.use_qnas:
            args.ansatz_mode = "qnas_static"
        elif args.progressive:
            args.ansatz_mode = "progressive_dense"
        else:
            args.ansatz_mode = "standard_static"
    elif args.ansatz_mode == "qnas_static":
        args.use_qnas = True
        if args.progressive:
            args.ansatz_mode = "qnas_progressive"
            print("[ansatz_mode] --progressive passed with --ansatz_mode qnas_static: auto-upgrading to 'qnas_progressive'.")
    elif args.ansatz_mode == "progressive_dense":
        args.progressive = True
        if args.use_qnas:
            args.ansatz_mode = "qnas_progressive"
            print("[ansatz_mode] --use_qnas passed with --ansatz_mode progressive_dense: auto-upgrading to 'qnas_progressive'.")

    if args.eval_only:
        run_eval_only(
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            qdevice=args.qdevice,
            threshold_criterion=args.threshold_criterion,
            target_specificity=args.target_specificity,
            cost_fp=args.cost_fp,
            recalibrate_threshold=args.recalibrate_threshold,
            fast_eval=args.fast_eval,
            skip_plots=args.skip_plots,
            plot_workers=args.plot_workers,
        )
    elif args.continual:
        run_continual_learning(
            dataset_sequence=args.continual_datasets,
            epochs_per_task=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            lambda_ewc=args.lambda_ewc,
            output_dir=args.output_dir,
            diff_method=args.diff_method,
            use_lora=args.use_lora,
            image_size=args.image_size,
            pretrain_epochs=args.pretrain_epochs,
            qdevice=args.qdevice,
        )
    elif args.kfold:
        run_kfold_training(
            dataset_name=args.dataset,
            n_folds=args.n_folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            mixup_alpha=args.mixup_alpha,
            phase_eps=args.phase_eps,
            output_dir=args.output_dir,
            lambda_ewc=args.lambda_ewc,
            dropout=args.dropout,
            diff_method=args.diff_method,
            use_lora=args.use_lora,
            use_qng=args.use_qng,
            use_qsam=args.use_qsam,
            use_qswa=args.use_qswa,
            lambda_infonce=args.lambda_infonce,
            image_size=args.image_size,
            pretrain_epochs=args.pretrain_epochs,
            use_qnas=args.use_qnas,
            qnas_generations=args.qnas_generations,
            qnas_pop_size=args.qnas_pop_size,
            qnas_min_cnot=args.qnas_min_cnot,
            qnas_max_cnot=args.qnas_max_cnot,
            qnas_target_mw=args.qnas_target_mw,
            qnas_lambda_cnot=args.qnas_lambda_cnot,
            qnas_genome_path=args.qnas_genome_path,
            qdevice=args.qdevice,
            force_pretrain=args.force_pretrain,
            balanced_sampler=args.balanced_sampler,
            asymmetric_loss=not args.no_asymmetric_loss,
            asym_alpha=args.asym_alpha,
            asym_gamma=args.asym_gamma,
            threshold_criterion=args.threshold_criterion,
            target_specificity=args.target_specificity,
            cost_fp=args.cost_fp,
            fast_eval=args.fast_eval,
            skip_plots=args.skip_plots,
            plot_workers=args.plot_workers,
            use_amp=args.amp,
            cache_in_vram=args.cache_in_vram,
            grad_accum_steps=args.grad_accum_steps,
            q_init_std=args.q_init_std,
            kfold_no_ensemble=args.kfold_no_ensemble,
            ansatz_mode=args.ansatz_mode,
            progressive_depths=args.progressive_depths,
            epochs_per_stage=args.epochs_per_stage,
            finetune_epochs=args.finetune_epochs,
            freeze_mode=args.freeze_mode,
            loss_type=args.loss_type,
            ldam_max_margin=args.ldam_max_margin,
            drw_start_epoch=args.drw_start_epoch,
            dual_basis_readout=args.dual_basis_readout,
            use_qaf_attention=args.use_qaf_attention,
            clinical_hard=args.clinical_hard,
            enable_vit_finetuning=args.enable_vit_finetuning,
            vit_unfreeze_layers=args.vit_unfreeze_layers,
            vit_finetune_lr=args.vit_finetune_lr,
            vit_epochs=args.vit_epochs,
            biomedclip_dir=args.biomedclip_dir,
            lambda_decorr=args.lambda_decorr,
            lambda_saliency=args.lambda_saliency,
        )
    else:
        run_training(
            dataset_name=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            mixup_alpha=args.mixup_alpha,
            phase_eps=args.phase_eps,
            output_dir=args.output_dir,
            lambda_ewc=args.lambda_ewc,
            dropout=args.dropout,
            diff_method=args.diff_method,
            use_lora=args.use_lora,
            use_qng=args.use_qng,
            use_qsam=args.use_qsam,
            use_qswa=args.use_qswa,
            lambda_infonce=args.lambda_infonce,
            image_size=args.image_size,
            pretrain_epochs=args.pretrain_epochs,
            use_qnas=args.use_qnas,
            qnas_generations=args.qnas_generations,
            qnas_pop_size=args.qnas_pop_size,
            qnas_min_cnot=args.qnas_min_cnot,
            qnas_max_cnot=args.qnas_max_cnot,
            qnas_target_mw=args.qnas_target_mw,
            qnas_lambda_cnot=args.qnas_lambda_cnot,
            qnas_genome_path=args.qnas_genome_path,
            qdevice=args.qdevice,
            force_pretrain=args.force_pretrain,
            adapter_checkpoint=args.adapter_checkpoint,
            progressive=args.progressive,
            progressive_depths=args.progressive_depths,
            epochs_per_stage=args.epochs_per_stage,
            finetune_epochs=args.finetune_epochs,
            freeze_mode=args.freeze_mode,
            q_init_std=args.q_init_std,
            balanced_sampler=args.balanced_sampler,
            asymmetric_loss=not args.no_asymmetric_loss,
            asym_alpha=args.asym_alpha,
            asym_gamma=args.asym_gamma,
            threshold_criterion=args.threshold_criterion,
            target_specificity=args.target_specificity,
            cost_fp=args.cost_fp,
            entropy_every_epoch=args.entropy_every_epoch,
            entropy_interval=args.entropy_interval,
            entropy_samples=args.entropy_samples,
            fast_eval=args.fast_eval,
            skip_plots=args.skip_plots,
            plot_workers=args.plot_workers,
            use_amp=args.amp,
            cache_in_vram=args.cache_in_vram,
            qsam_freq=args.qsam_freq,
            grad_accum_steps=args.grad_accum_steps,
            benchmark_mode=args.benchmark_mode,
            profile_runtime=args.profile_runtime,
            post_quantum_mixup=not args.pre_quantum_mixup,
            lambda_kta=args.lambda_kta,
            use_cost_matrix=args.use_cost_matrix,
            ansatz_mode=args.ansatz_mode,
            loss_type=args.loss_type,
            ldam_max_margin=args.ldam_max_margin,
            drw_start_epoch=args.drw_start_epoch,
            dual_basis_readout=args.dual_basis_readout,
            use_qaf_attention=args.use_qaf_attention,
            clinical_hard=args.clinical_hard,
            enable_vit_finetuning=args.enable_vit_finetuning,
            vit_unfreeze_layers=args.vit_unfreeze_layers,
            vit_finetune_lr=args.vit_finetune_lr,
            vit_epochs=args.vit_epochs,
            biomedclip_dir=args.biomedclip_dir,
            lambda_decorr=args.lambda_decorr,
            lambda_saliency=args.lambda_saliency,
        )


if __name__ == "__main__":
    main()
