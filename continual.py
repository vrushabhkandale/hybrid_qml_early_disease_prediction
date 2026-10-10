"""
continual.py — Quantum Elastic Weight Consolidation (QEWC) & Information Metrics
================================================================================
Lifelong learning and quantum information theoretic metrics:
  • compute_diagonal_qfim: True geometric Fubini-Study Quantum Fisher Information Matrix
  • compute_fisher_diagonal: Diagonal empirical Fisher for classical & QFIM for quantum
  • qewc_penalty: Geometric quadratic parameter regularizer
  • compute_expressibility: Haar-random state fidelity KL divergence
  • compute_meyer_wallach: Multi-qubit bipartite entanglement measure
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model import get_pennylane_device, HybridCNNVQC
except ImportError:
    from qml_models.model import get_pennylane_device, HybridCNNVQC


def compute_diagonal_qfim(
    model: HybridCNNVQC,
    loader,
    device: torch.device,
    n_samples: int = 16,
) -> torch.Tensor:
    """Compute exact diagonal of Quantum Fisher Information Matrix (QFIM).
    F_Q,kk = 4 * g_kk(theta)
    where g is the Fubini-Study metric tensor of the state generator.
    (Section 3.2 & 3.3 in overfitting_fixes.md)

    Fix E-06: Dynamically constructs the metric circuit from model.genome
    when available (QNAS 1D DAG), falling back to StronglyEntanglingLayers
    for standard models. Eliminates dimension crashes on 1D weight tensors.
    """
    n_qubits = model.n_qubits
    n_layers = model.n_layers
    dev = qml.device("default.qubit", wires=n_qubits)
    reupload = getattr(model, "reupload", True)
    genome = getattr(model, "genome", None)

    @qml.qnode(dev)
    def circ(inputs, weights):
        # Fix E-06: Dynamic circuit construction from QNAS genome
        if genome is not None:
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            w_idx = 0
            for l, layer in enumerate(genome.layers):
                if reupload or l == 0:
                    for i in range(n_qubits):
                        qml.U3(inputs[i], inputs[i + n_qubits], inputs[i + 2 * n_qubits], wires=i)
                for gene in layer:
                    if gene.gate_type == "Rx":
                        qml.RX(weights[w_idx], wires=gene.target); w_idx += 1
                    elif gene.gate_type == "Ry":
                        qml.RY(weights[w_idx], wires=gene.target); w_idx += 1
                    elif gene.gate_type == "Rz":
                        qml.RZ(weights[w_idx], wires=gene.target); w_idx += 1
                    elif gene.gate_type == "U3":
                        qml.U3(weights[w_idx], weights[w_idx+1], weights[w_idx+2], wires=gene.target); w_idx += 3
                    elif gene.gate_type == "CNOT":
                        qml.CNOT(wires=[gene.control, gene.target])
                    elif gene.gate_type == "CZ":
                        qml.CZ(wires=[gene.control, gene.target])
        else:
            # Default StronglyEntanglingLayers path
            if reupload:
                for l in range(n_layers):
                    for i in range(n_qubits):
                        qml.U3(inputs[i], inputs[i + n_qubits], inputs[i + 2 * n_qubits], wires=i)
                    qml.StronglyEntanglingLayers(weights[l:l+1], wires=range(n_qubits))
            else:
                for i in range(n_qubits):
                    qml.U3(inputs[i], inputs[i + n_qubits], inputs[i + 2 * n_qubits], wires=i)
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    metric_fn = qml.metric_tensor(circ, approx="diag")

    q_param = None
    for n, p in model.named_parameters():
        if "qlayer" in n and "weights" in n:
            q_param = p
            break
    if q_param is None:
        return torch.ones((n_layers, n_qubits, 3), device=device)

    w_shape = q_param.shape
    P = int(np.prod(w_shape))
    w_np = pnp.array(q_param.detach().cpu().numpy(), requires_grad=True)

    # Extract classical feature batch
    latent_samples = []
    ds = getattr(loader, "dataset", None)
    if ds is not None and hasattr(ds, "tensors") and len(ds.tensors) >= 1 and ds.tensors[0].ndim == 2:
        feats = ds.tensors[0][:n_samples]
        if feats.size(1) == 3 * n_qubits:
            latent_samples = [feats]
        elif feats.size(1) == 512:
            with torch.no_grad():
                z = model.backbone(feats.to(device), use_cached_features=True)
                latent_samples = [z.detach().cpu()]
    else:
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                images = images.to(device)
                if images.ndim == 2 and images.size(1) == 3 * n_qubits:
                    z = images
                elif images.ndim == 2 and images.size(1) == 512:
                    z = model.backbone(images, use_cached_features=True)
                else:
                    z = model.backbone(images, use_cached_features=False)
                latent_samples.append(z.detach().cpu())
                if sum(s.size(0) for s in latent_samples) >= n_samples:
                    break

    if not latent_samples:
        return torch.ones(w_shape, device=device)

    latent_cat = torch.cat(latent_samples, dim=0)[:n_samples]

    qfim_accum = np.zeros(P)
    count = 0
    for s in range(latent_cat.size(0)):
        z_s = pnp.array(latent_cat[s].numpy(), requires_grad=False)
        try:
            g_raw = np.asarray(metric_fn(z_s, w_np))
            if g_raw.size == P:
                g_diag = g_raw.flatten()
            else:
                g_diag = np.diag(g_raw.reshape(P, P))
            qfim_accum += 4.0 * np.maximum(g_diag, 1e-6)
            count += 1
        except Exception:
            pass

    if count > 0:
        qfim_accum /= count
    else:
        qfim_accum = np.ones(P)

    qfim_tensor = torch.from_numpy(qfim_accum.reshape(w_shape)).float().to(device)
    return torch.clamp(qfim_tensor, min=1e-5, max=100.0)


def compute_fisher_diagonal(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    n_classes: int,
    n_samples: int = 200,
) -> Dict[str, torch.Tensor]:
    """Compute diagonal Fisher information for classical parameters and true
    geometric Quantum Fisher Information Matrix (QFIM) for quantum parameters.
    (Section 3 in overfitting_fixes.md)
    """
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}

    # True geometric Fubini-Study QFIM for the quantum parameters
    if isinstance(model, HybridCNNVQC):
        for n, p in model.named_parameters():
            if "qlayer" in n and "weights" in n and p.requires_grad:
                print("[QEWC] Computing true geometric Quantum Fisher Information Matrix (F_Q = 4·g)...")
                fisher[n] = compute_diagonal_qfim(model, loader, device, n_samples=min(16, n_samples))

    # Classical parameters empirical Fisher
    class_samples = {c: [] for c in range(n_classes)}
    class_counts = {c: 0 for c in range(n_classes)}
    target_per_class = max(1, n_samples // n_classes)

    for images, labels in loader:
        labels = labels.squeeze().long()
        for i, label in enumerate(labels):
            c = label.item()
            if class_counts[c] < target_per_class:
                class_samples[c].append((images[i:i+1], label.unsqueeze(0)))
                class_counts[c] += 1
        if all(v >= target_per_class for v in class_counts.values()):
            break

    balanced_samples = []
    for c in range(n_classes):
        balanced_samples.extend(class_samples[c])

    if not balanced_samples:
        return fisher

    rng = np.random.RandomState(42)
    rng.shuffle(balanced_samples)

    count: int = 0
    for sample in balanced_samples:
        b_image = sample[0].to(device)
        b_label = sample[1].to(device)
        model.zero_grad()
        is_cached = (b_image.ndim == 2 and b_image.size(1) == 512)
        logits = model(b_image, use_cached_features=is_cached)
        loss = criterion(logits, b_label)
        loss.backward()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None and "qlayer" not in n:
                    fisher[n] += p.grad.detach() ** 2
        model.zero_grad()
        count += 1
        if count >= n_samples:
            break

    # Normalise classical parameters
    for n in fisher:
        if "qlayer" not in n:
            fisher[n] /= max(count, 1)

    return fisher


def qewc_penalty(
    model: nn.Module,
    fisher: Dict[str, torch.Tensor],
    prev_params: Dict[str, torch.Tensor],
    lambda_ewc: float = 400.0,
) -> torch.Tensor:
    r"""Geometric QEWC quadratic penalty:  Σ_j  (λ/2) F_Q,jj (θ_j - θ_j,prev)²."""
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for n, p in model.named_parameters():
        if n in fisher and n in prev_params:
            f_diag = fisher[n].to(p.device)
            p_prev = prev_params[n].to(p.device)
            if "qlayer" in n:
                # Geodesic distance on S^1 — wraps difference into [-pi, pi]
                delta = torch.atan2(torch.sin(p - p_prev), torch.cos(p - p_prev))
                penalty += (f_diag * delta ** 2).sum()
            else:
                penalty += (f_diag * (p - p_prev) ** 2).sum()
    return (lambda_ewc / 2.0) * penalty


def compute_expressibility(
    n_qubits: int = 4,
    n_layers: int = 3,
    n_samples: int = 500,
    n_bins: int = 75,
) -> float:
    r"""Expressibility via KL divergence between ansatz fidelity distribution
    and Haar-random uniform distribution.

    Expr = D_KL(P_ansatz || P_Haar)
    Lower is better (closer to Haar-random).
    """
    dev, _ = get_pennylane_device(n_qubits)

    @qml.qnode(dev, interface="numpy")
    def circuit(params):
        qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
        return qml.state()

    shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    def compute_fid(_):
        p1 = np.random.uniform(0, 2 * np.pi, size=shape)
        p2 = np.random.uniform(0, 2 * np.pi, size=shape)
        s1 = circuit(p1)
        s2 = circuit(p2)
        return np.abs(np.dot(np.conj(s1), s2)) ** 2

    fidelities = [compute_fid(i) for i in range(n_samples)]
    fidelities = np.array(fidelities)

    # Haar distribution for N-qubit pure states: P_Haar(F) = (2^n - 1)(1-F)^(2^n - 2)
    dim = 2 ** n_qubits
    hist_ansatz, bin_edges = np.histogram(fidelities, bins=n_bins, range=(0, 1), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    haar_pdf = (dim - 1) * (1 - bin_centers) ** (dim - 2)

    # Normalise to proper PMFs
    eps = 1e-10
    p = hist_ansatz / (hist_ansatz.sum() + eps)
    q = haar_pdf / (haar_pdf.sum() + eps)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)

    kl = float(np.sum(p * np.log(p / q)))
    return kl


def compute_meyer_wallach(
    latent_vectors: np.ndarray,
    n_qubits: int = 4,
    n_layers: int = 3,
    n_samples: int = 100,
) -> float:
    r"""Meyer-Wallach entanglement measure Q averaged over medical latent vectors.

    Q(|ψ⟩) = (2/n) Σ_k (1 - Tr(ρ_k²))
    where ρ_k is the reduced density matrix of qubit k.
    """
    dev, _ = get_pennylane_device(n_qubits)
    shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    rng = np.random.RandomState(42)
    weights = rng.normal(0, 0.01, size=shape)

    @qml.qnode(dev, interface="numpy")
    def circuit(inputs, w):
        for i in range(n_qubits):
            qml.U3(inputs[..., i], inputs[..., i + n_qubits], inputs[..., i + 2 * n_qubits], wires=i)
        qml.StronglyEntanglingLayers(w, wires=range(n_qubits))
        return qml.state()

    indices = rng.choice(len(latent_vectors), size=min(n_samples, len(latent_vectors)), replace=False)

    def compute_q_val(idx):
        z = latent_vectors[idx]
        state = circuit(z, weights)
        n = n_qubits
        state_vec = np.array(state).reshape(-1)
        
        q_val = 0.0
        for k in range(n):
            state_reshaped = state_vec.reshape([2] * n)
            state_moved = np.moveaxis(state_reshaped, k, 0)
            state_2d = state_moved.reshape(2, -1)
            rho_k = state_2d @ state_2d.conj().T
            purity = np.real(np.trace(rho_k @ rho_k))
            q_val += (1.0 - purity)
            
        return q_val * (2.0 / n)

    q_values = [compute_q_val(idx) for idx in indices]
    return float(np.mean(q_values))
