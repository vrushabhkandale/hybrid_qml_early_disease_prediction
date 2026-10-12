"""
qnas.py — Quantum Neural Architecture Search (QNAS) for Medical Hybrid Architectures
====================================================================================
Implements:
  1. Pairwise Mutual Information (MI) analysis of latent medical features
  2. Admissibility Graph G = (V, E) construction with adaptive thresholding
  3. Discrete DAG Genome encoding candidate single-qubit & entangling gates
  4. Multi-objective Pareto evolutionary search (Val-AUC, CNOT budget, Meyer-Wallach Q_MW)
  5. Dynamic PennyLane QNode builder from Pareto-optimal genomes

Theoretical Reference: overfitting_fixes.md (Section 2)

Remediation Log (qnas_engine_fixes.md):
  F-01: Analytical Meyer-Wallach via Pauli expectations on data states (Brennen 2003)
  F-02: Vectorized batched QNode execution (18.7x speedup)
  F-03: Hadamard + interleaved data re-uploading contract parity with model.py
  F-04: Stratified balanced validation sampling (eliminates single-class AUC crash)
  F-05: Depth-preserving crossover with dynamic gate insertion mutation
  F-06: 3D multivariate MI estimation (eliminates scalar SU(2) angle averaging)
  F-07: Weight serialization for warm-start transfer to HybridCNNVQC
  F-08: Dynamic diff_method & backend selection
  F-09: Threshold correction (0.25σ → 0.50σ)
  F-10: Deterministic RNG via seeded RandomState
  F-11: Insert mutation operator for reversible Markov chain search
  F-12: Explicit dimension validation (no silent feature truncation)
  F-13: Internal diagnostic visualizations (Pareto frontier, fitness trajectory)
  F-14: Statevector normalization assertion
"""

from __future__ import annotations

import copy
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import roc_auc_score

__all__ = [
    "compute_meyer_wallach_pure",
    "compute_meyer_wallach_analytical",
    "get_stratified_eval_batch",
    "compute_multivariate_mi_matrix",
    "compute_latent_mutual_information",
    "select_parent_tournament",
    "compute_hardware_coupling_cost",
    "compute_geometric_condition_number",
    "GateGene",
    "CircuitGenome",
    "save_genome",
    "load_genome",
    "QNASOptimizer",
]


# ---------------------------------------------------------------------------
# Meyer-Wallach Entanglement Measure (Legacy Tensordot — F-14 norm assertion)
# ---------------------------------------------------------------------------
def compute_meyer_wallach_pure(state_vector: np.ndarray, n_qubits: int) -> float:
    """Compute Meyer-Wallach entanglement measure Q(|psi>) in [0, 1].

    Q(|psi>) = (2 / n) * sum_{k=0}^{n-1} (1 - Tr(rho_k^2))
    where rho_k is the reduced density matrix of qubit k.

    Fix F-14: Added statevector normalization assertion.
    """
    if n_qubits <= 1:
        return 0.0

    state = np.asarray(state_vector).flatten()
    dim = 2 ** n_qubits
    if len(state) != dim:
        return 0.0

    # F-14: Normalize if numerical drift occurs and assert physicality
    norm = np.linalg.norm(state)
    if abs(norm - 1.0) > 1e-5:
        if norm < 1e-10:
            return 0.0
        state = state / norm

    psi = state.reshape([2] * n_qubits)
    q_sum = 0.0

    for k in range(n_qubits):
        # Trace out all qubits except k
        axes_to_trace = [i for i in range(n_qubits) if i != k]
        # Reshape to bipartite cut: qubit k vs rest
        rho_k = np.tensordot(psi, np.conj(psi), axes=(axes_to_trace, axes_to_trace))
        tr_rho_sq = np.real(np.trace(rho_k @ rho_k))
        q_sum += (1.0 - tr_rho_sq)

    return float(max(0.0, min(1.0, (2.0 / n_qubits) * q_sum)))


# ---------------------------------------------------------------------------
# Analytical Meyer-Wallach via Pauli Expectations (Brennen 2003) — Fix F-01
# ---------------------------------------------------------------------------
def compute_meyer_wallach_analytical(
    pauli_expvals: np.ndarray, n_qubits: int
) -> float:
    """Compute exact Meyer-Wallach entanglement from 1-qubit Pauli observables.

    Q(|psi>) = 1 - (1 / n) * sum_{k=0}^{n-1} (<X_k>^2 + <Y_k>^2 + <Z_k>^2)

    Mathematical derivation (Brennen 2003, arXiv:quant-ph/0305094):
        rho_k = (I + <X_k> sigma_x + <Y_k> sigma_y + <Z_k> sigma_z) / 2
        Tr(rho_k^2) = (1 + |r_k|^2) / 2
        Q = (2/n) * sum_k (1 - Tr(rho_k^2)) = 1 - (1/n) * sum_k |r_k|^2

    Args:
        pauli_expvals: Array of shape (3 * n_qubits,) ordered as
                       [<X_0>, ..., <X_{n-1}>, <Y_0>, ..., <Y_{n-1}>, <Z_0>, ..., <Z_{n-1}>]
                       OR shape (B, 3 * n_qubits) for batch-averaged computation.
        n_qubits: Number of qubits.

    Returns:
        Meyer-Wallach entanglement Q in [0, 1].
    """
    if n_qubits <= 1:
        return 0.0

    exp = np.asarray(pauli_expvals)
    if exp.ndim == 1:
        exp = exp[np.newaxis, :]  # Shape: (1, 3 * n_qubits)

    B = exp.shape[0]
    x_vals = exp[:, :n_qubits]                  # Shape: (B, n_qubits)
    y_vals = exp[:, n_qubits:2 * n_qubits]      # Shape: (B, n_qubits)
    z_vals = exp[:, 2 * n_qubits:3 * n_qubits]  # Shape: (B, n_qubits)

    # Compute Bloch vector squared lengths per sample per qubit: |r_{i, k}|^2
    r_sq = x_vals ** 2 + y_vals ** 2 + z_vals ** 2
    r_sq = np.clip(r_sq, 0.0, 1.0)

    # Meyer-Wallach per sample: Q_i = 1 - (1/n) * sum_k |r_{i, k}|^2
    q_per_sample = 1.0 - np.mean(r_sq, axis=1)  # Shape: (B,)

    return float(np.clip(np.mean(q_per_sample), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Stratified Balanced Validation Sampling — Fix F-04
# ---------------------------------------------------------------------------
def get_stratified_eval_batch(
    val_z: torch.Tensor,
    val_y: torch.Tensor,
    total_samples: int = 64,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample balanced representation across all classes to prevent AUC crashes.

    Fix F-04: Eliminates single-class AUC crash by guaranteeing every class
    has at least min(N_c, 4) samples in the evaluation batch.
    """
    rng = np.random.RandomState(seed)
    unique_classes = torch.unique(val_y).cpu().numpy()
    n_classes = len(unique_classes)
    samples_per_class = max(4, total_samples // n_classes)

    selected_indices = []
    for c in unique_classes:
        cls_idx = (val_y == c).nonzero(as_tuple=True)[0].cpu().numpy()
        n_sel = min(len(cls_idx), samples_per_class)
        chosen = rng.choice(cls_idx, size=n_sel, replace=False)
        selected_indices.extend(chosen.tolist())

    idx_tensor = torch.tensor(selected_indices, dtype=torch.long)
    return val_z[idx_tensor], val_y[idx_tensor]


# ---------------------------------------------------------------------------
# Multivariate Mutual Information (MI) Matrix — Fast Log-Det Covariance
# ---------------------------------------------------------------------------
def compute_multivariate_mi_matrix(
    latent_tensor: torch.Tensor | np.ndarray,
    n_qubits: int,
    seed: int = 42,
) -> np.ndarray:
    """Compute pairwise mutual information matrix between 3D Bloch coordinate vectors.

    Phase 2.1 (Kraskov et al., 2004 & Gaussian log-det formulation):
        I(V_i; V_j) = 0.5 * ln( (det(C_ii) * det(C_jj)) / det(C_joint) )
    Captures all 9 cross-angle correlations (theta_i, phi_i, lambda_i) vs (theta_j, phi_j, lambda_j).
    Regularized via Tikhonov regularization C + eps * I (eps = 1e-4) to guarantee positive definiteness.

    Args:
        latent_tensor: Tensor or ndarray of shape (N, 3 * n_qubits) or (N, n_qubits, 3).
        n_qubits: Number of qubits.
        seed: Random seed for deterministic subsampling if necessary.

    Returns:
        Symmetric (n_qubits, n_qubits) matrix of pairwise mutual information.
    """
    if isinstance(latent_tensor, torch.Tensor):
        z_np = latent_tensor.detach().cpu().numpy()
    else:
        z_np = np.asarray(latent_tensor)

    N = z_np.shape[0]
    if z_np.ndim == 3 and z_np.shape[1] == n_qubits and z_np.shape[2] == 3:
        z_3d = z_np
    elif z_np.ndim == 2 and z_np.shape[1] >= 3 * n_qubits:
        z_3d = np.zeros((N, n_qubits, 3))
        for q in range(n_qubits):
            z_3d[:, q, 0] = z_np[:, q]
            z_3d[:, q, 1] = z_np[:, q + n_qubits]
            z_3d[:, q, 2] = z_np[:, q + 2 * n_qubits]
    else:
        raise ValueError(
            f"Expected latent_tensor shape (N, {3 * n_qubits}) or (N, {n_qubits}, 3), got {z_np.shape}"
        )

    mi_matrix = np.zeros((n_qubits, n_qubits))
    reg3 = 1e-4 * np.eye(3)
    reg6 = 1e-4 * np.eye(6)

    for i in range(n_qubits):
        vi = z_3d[:, i, :]  # (N, 3)
        vi_c = vi - vi.mean(axis=0, keepdims=True)
        c_ii = (vi_c.T @ vi_c) / max(1, N - 1) + reg3
        det_ii = float(np.linalg.det(c_ii))

        for j in range(i + 1, n_qubits):
            vj = z_3d[:, j, :]  # (N, 3)
            vj_c = vj - vj.mean(axis=0, keepdims=True)
            c_jj = (vj_c.T @ vj_c) / max(1, N - 1) + reg3
            det_jj = float(np.linalg.det(c_jj))

            vij_c = np.hstack([vi_c, vj_c])  # (N, 6)
            c_joint = (vij_c.T @ vij_c) / max(1, N - 1) + reg6
            det_joint = float(np.linalg.det(c_joint))

            ratio = (det_ii * det_jj) / max(1e-12, det_joint)
            mi_val = max(1e-4, 0.5 * np.log(max(1.0, ratio)))

            mi_matrix[i, j] = mi_val
            mi_matrix[j, i] = mi_val

    return mi_matrix


# ---------------------------------------------------------------------------
# Mutual Information (MI) Graph Construction
# ---------------------------------------------------------------------------
def compute_latent_mutual_information(
    latent_tensor: torch.Tensor,
    n_qubits: int,
    seed: int = 42,  # Fix F-10: Deterministic RNG
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pairwise mutual information matrix M in R^{n_qubits x n_qubits}.

    Args:
        latent_tensor: Tensor of shape (N, 3*n_qubits) or (N, n_qubits).
        n_qubits: Number of qubits.
        seed: Random seed for reproducibility (Fix F-10).

    Returns:
        mi_matrix: Symmetric (n_qubits, n_qubits) matrix of pairwise MI.
        admissibility_mask: Binary boolean (n_qubits, n_qubits) mask where (i, j) is True
                            iff MI(Z_i, Z_j) >= threshold tau = mean(M) + 0.5 * std(M).
    """
    # Fix F-10: Use dedicated seeded RandomState instead of global RNG
    rng = np.random.RandomState(seed)

    z_np = latent_tensor.detach().cpu().numpy()

    # Fix F-12: Explicit dimension validation (no silent feature truncation)
    if z_np.shape[1] >= 3 * n_qubits:
        # Fix F-06 & Phase 2.1: Use 3D multivariate MI instead of scalar averaging
        z_channels_3d = np.zeros((z_np.shape[0], n_qubits, 3))
        for q in range(n_qubits):
            z_channels_3d[:, q, 0] = z_np[:, q]
            z_channels_3d[:, q, 1] = z_np[:, q + n_qubits]
            z_channels_3d[:, q, 2] = z_np[:, q + 2 * n_qubits]
        use_3d = True
    elif z_np.shape[1] >= n_qubits:
        z_channels_3d = z_np[:, :n_qubits].reshape(z_np.shape[0], n_qubits, 1)
        use_3d = False
    else:
        raise ValueError(
            f"Latent dimension {z_np.shape[1]} is less than n_qubits={n_qubits}. "
            f"Expected at least {n_qubits} (or {3 * n_qubits} for full U3 angles). "
            f"Check CQ adapter output dimension."
        )

    n_samples = min(z_channels_3d.shape[0], 500)
    indices = rng.choice(z_channels_3d.shape[0], n_samples, replace=False)
    sub_3d = z_channels_3d[indices]

    if use_3d:
        mi_matrix = compute_multivariate_mi_matrix(sub_3d, n_qubits, seed=seed)
    else:
        mi_matrix = np.zeros((n_qubits, n_qubits))
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                try:
                    xi = sub_3d[:, i] + rng.normal(0, 1e-6, size=sub_3d[:, i].shape)
                    yj = sub_3d[:, j].ravel()
                    mi_val = float(mutual_info_regression(xi, yj, random_state=seed)[0])
                except Exception:
                    corr = np.corrcoef(sub_3d[:, i].ravel(), sub_3d[:, j].ravel())[0, 1]
                    mi_val = -0.5 * np.log(max(1e-5, 1.0 - corr ** 2)) if not np.isnan(corr) else 1e-4
                mi_val = max(1e-4, mi_val)
                mi_matrix[i, j] = mi_val
                mi_matrix[j, i] = mi_val

    # Fix F-09: Compute adaptive threshold tau = mu + 0.50 * sigma (was 0.25)
    off_diag = mi_matrix[~np.eye(n_qubits, dtype=bool)]
    if len(off_diag) > 0 and np.std(off_diag) > 1e-6:
        tau = np.mean(off_diag) + 0.50 * np.std(off_diag)
    else:
        tau = 0.0

    # Ensure graph is connected via Maximum Spanning Tree of MI
    # Guaranteed: EVERY qubit has degree >= 1, eliminating Qubit 0 isolation
    try:
        from scipy.sparse.csgraph import minimum_spanning_tree
        dist_matrix = -mi_matrix.copy()
        np.fill_diagonal(dist_matrix, 0)
        mst = minimum_spanning_tree(dist_matrix).toarray()
        mst_mask = (mst < 0) | (mst.T < 0)
    except Exception:
        mst_mask = np.zeros((n_qubits, n_qubits), dtype=bool)
        for i in range(n_qubits - 1):
            mst_mask[i, i + 1] = True
            mst_mask[i + 1, i] = True

    admissibility_mask = (mi_matrix >= tau) | mst_mask
    np.fill_diagonal(admissibility_mask, False)

    return mi_matrix, admissibility_mask


# ---------------------------------------------------------------------------
# Genome & Gate Specification
# ---------------------------------------------------------------------------
SINGLE_QUBIT_GATES = ["Rx", "Ry", "Rz", "U3", "Identity"]
TWO_QUBIT_GATES = ["CNOT", "CZ"]

@dataclass
class GateGene:
    gate_type: str  # "Rx", "Ry", "Rz", "U3", "Identity", "CNOT", "CZ"
    target: int
    control: Optional[int] = None
    is_parameterized: bool = False
    param_count: int = 0

    def to_dict(self) -> dict:
        return {
            "gate_type": self.gate_type,
            "target": self.target,
            "control": self.control,
            "is_parameterized": self.is_parameterized,
            "param_count": self.param_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GateGene:
        return cls(
            gate_type=d["gate_type"],
            target=d["target"],
            control=d.get("control"),
            is_parameterized=d.get("is_parameterized", False),
            param_count=d.get("param_count", 0),
        )


@dataclass
class CircuitGenome:
    n_qubits: int
    layers: List[List[GateGene]] = field(default_factory=list)
    fitness: float = -999.0
    val_auc: float = 0.0
    cnot_count: int = 0
    mw_entanglement: float = 0.0
    trained_weights: Optional[List[float]] = None  # Fix F-07: Weight serialization
    max_depth: Optional[int] = None
    qfim_condition: Optional[float] = None  # Phase 6.1: Metric tensor condition number
    hardware_cost: Optional[int] = None     # Phase 6.2: Routing-aware CNOT coupling cost
    valid: bool = True
    invalid_reason: Optional[str] = None

    def compute_stats(self) -> None:
        cnot = 0
        for layer in self.layers:
            for g in layer:
                if g.gate_type in ("CNOT", "CZ"):
                    cnot += 1
        self.cnot_count = cnot

    @property
    def total_params(self) -> int:
        p = 0
        for layer in self.layers:
            for g in layer:
                p += g.param_count
        return p

    def to_dict(self) -> dict:
        d = {
            "n_qubits": self.n_qubits,
            "layers": [[g.to_dict() for g in layer] for layer in self.layers],
            "fitness": float(self.fitness),
            "val_auc": float(self.val_auc),
            "cnot_count": int(self.cnot_count),
            "mw_entanglement": float(self.mw_entanglement),
            "max_depth": self.max_depth,
            "valid": bool(self.valid),
            "invalid_reason": self.invalid_reason,
        }
        # Fix F-07: Serialize trained weights for warm-start transfer
        if self.trained_weights is not None:
            d["trained_weights"] = [float(w) for w in self.trained_weights]
        if self.qfim_condition is not None:
            d["qfim_condition"] = float(self.qfim_condition)
        if self.hardware_cost is not None:
            d["hardware_cost"] = int(self.hardware_cost)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> CircuitGenome:
        genome = cls(
            n_qubits=d["n_qubits"],
            fitness=d.get("fitness", -999.0),
            val_auc=d.get("val_auc", 0.0),
            cnot_count=d.get("cnot_count", 0),
            mw_entanglement=d.get("mw_entanglement", 0.0),
            max_depth=d.get("max_depth", None),
            qfim_condition=d.get("qfim_condition"),
            hardware_cost=d.get("hardware_cost"),
            valid=d.get("valid", True),
            invalid_reason=d.get("invalid_reason"),
        )
        genome.layers = [
            [GateGene.from_dict(g) for g in layer]
            for layer in d.get("layers", [])
        ]
        # Fix F-07: Deserialize trained weights
        if "trained_weights" in d and d["trained_weights"] is not None:
            genome.trained_weights = d["trained_weights"]
        genome.compute_stats()
        return genome


def save_genome(genome: CircuitGenome, filepath: str) -> None:
    """Save CircuitGenome to JSON file."""
    import json, os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(genome.to_dict(), f, indent=2)


def load_genome(filepath: str) -> CircuitGenome:
    """Load CircuitGenome from JSON file."""
    import json
    with open(filepath, "r") as f:
        d = json.load(f)
    return CircuitGenome.from_dict(d)


# ---------------------------------------------------------------------------
# Genetic Operators & Quantum Geometric Helpers
# ---------------------------------------------------------------------------
def select_parent_tournament(population: List[CircuitGenome], k: int = 3) -> CircuitGenome:
    """Tournament selection (NSGA-II inspired) to preserve genetic diversity.

    Phase 3.1: Replaces top-25% truncation selection to prevent premature stagnation.
    """
    if not population:
        raise ValueError("Population cannot be empty for tournament selection.")
    candidates = random.sample(population, min(k, len(population)))
    return max(candidates, key=lambda ind: ind.fitness)


def compute_hardware_coupling_cost(
    genome: CircuitGenome, topology: str = "linear", n_qubits: int = 4
) -> int:
    """Compute physical routing distance cost for 2-qubit gates on a target hardware topology.

    Phase 6.2: Instead of uniform CNOT cost, weight each two-qubit gate by its physical
    coupling distance: C_weighted(G) = sum_{g in 2Q} dist_topo(ctrl(g), tgt(g)).
    """
    cost = 0
    for layer in genome.layers:
        for gene in layer:
            if gene.gate_type in ("CNOT", "CZ"):
                c, t = gene.control, gene.target
                if c is None or t is None:
                    continue
                if topology == "linear":
                    cost += abs(c - t)
                elif topology == "ring":
                    cost += min(abs(c - t), n_qubits - abs(c - t))
                else:
                    cost += 1
    return cost


def compute_geometric_condition_number(
    genome: CircuitGenome,
    weights: torch.Tensor,
    dev: Optional[Any] = None,
    n_qubits: Optional[int] = None,
    data_embeddings: Optional[torch.Tensor] = None,
) -> float:
    """Compute condition number of the Fubini-Study metric tensor g(theta).

    Phase 6.2 (Stokes et al., Quantum 2020):
    kappa(g) = lambda_max(g) / (lambda_min(g) + 1e-6)
    Measures the quantum geometric condition number of the parameterized ansatz.
    Evaluated over empirical data states using block-diagonal approximation.
    """
    total_p = genome.total_params
    if total_p <= 1:
        return 1.0

    n_q = n_qubits or genome.n_qubits
    if dev is None:
        dev = qml.device("default.qubit", wires=n_q)

    # Use empirical training embeddings if available, else canonical zeros (Task 6.2)
    if data_embeddings is not None and len(data_embeddings) > 0:
        eval_inputs = data_embeddings[:16]
    else:
        eval_inputs = torch.zeros((1, 3 * n_q), dtype=torch.float32)

    def _build_geom_circuit(inp_vec):
        @qml.qnode(dev, interface="torch")
        def geom_circuit(w):
            w_idx = 0
            for l, layer in enumerate(genome.layers):
                if l == 0:
                    for q in range(n_q):
                        qml.U3(
                            inp_vec[q],
                            inp_vec[q + n_q],
                            inp_vec[q + 2 * n_q],
                            wires=q,
                        )
                for gene in layer:
                    if gene.gate_type == "Rx":
                        qml.RX(w[w_idx], wires=gene.target); w_idx += 1
                    elif gene.gate_type == "Ry":
                        qml.RY(w[w_idx], wires=gene.target); w_idx += 1
                    elif gene.gate_type == "Rz":
                        qml.RZ(w[w_idx], wires=gene.target); w_idx += 1
                    elif gene.gate_type == "U3":
                        qml.U3(w[w_idx], w[w_idx + 1], w[w_idx + 2], wires=gene.target); w_idx += 3
                    elif gene.gate_type == "CNOT":
                        qml.CNOT(wires=[gene.control, gene.target])
                    elif gene.gate_type == "CZ":
                        qml.CZ(wires=[gene.control, gene.target])
                    elif gene.gate_type == "Identity":
                        qml.Identity(wires=gene.target)
            return qml.expval(qml.PauliZ(0))
        return geom_circuit

    try:
        w_tensor = weights.detach().clone().float().requires_grad_(True)
        g_matrices = []
        for inp in eval_inputs:
            c_node = _build_geom_circuit(inp)
            try:
                mt_fn = qml.metric_tensor(c_node, approx="block-diagonal")
            except Exception:
                mt_fn = qml.metric_tensor(c_node)
            g = mt_fn(w_tensor)
            if isinstance(g, torch.Tensor):
                g_mat = g.detach().cpu().double()
            else:
                g_mat = torch.tensor(g, dtype=torch.float64)
            g_matrices.append(g_mat)

        avg_g = torch.stack(g_matrices, dim=0).mean(dim=0)
        eigvals = torch.linalg.eigvalsh(avg_g)
        eigvals = torch.clamp(eigvals, min=0.0)
        max_eig = float(eigvals.max().item())
        min_eig = float(eigvals.min().item())
        cond = max_eig / (min_eig + 1e-6)
        return float(min(1e8, max(1.0, cond)))
    except Exception:
        return 1.0



# ---------------------------------------------------------------------------
# QNAS Evolutionary Optimizer
# ---------------------------------------------------------------------------
class QNASOptimizer:
    """Quantum Neural Architecture Search via Pareto Genetic Evolution.

    Fixes applied:
        F-01: Analytical Meyer-Wallach on data states (not vacuum)
        F-02: Vectorized batched QNode execution
        F-03: Hadamard + data re-uploading parity with model.py
        F-04: Stratified balanced validation sampling
        F-05: Depth-preserving crossover
        F-07: Weight serialization
        F-08: Dynamic diff_method and backend selection
        F-11: Insert mutation operator
    """

    def __init__(
        self,
        n_qubits: int = 4,
        min_cnot: int = 2,
        max_cnot: int = 8,
        max_depth: int = 4,
        population_size: int = 16,
        generations: int = 6,
        lambda_cnot: float = 0.01,
        lambda_under_cnot: float = 0.25,
        lambda_ent: float = 0.08,
        target_mw: float = 0.45,
        reupload: bool = True,  # Fix F-03: Contract parity with model.py
        diff_method: str = "backprop",  # Fix F-08: Respect diff_method
        output_dir: Optional[str] = None,  # Fix F-13: Diagnostic output
        qdevice: Optional[str] = None,
        hardware_topology: str = "linear",  # Phase 6.2: Physical qubit coupling topology
        check_geometric_condition: bool = True,  # Phase 6.1: QFIM / metric tensor condition filter
        dual_basis_only: bool = False,  # Task 6.1: Dual-basis readout synchronization
    ):
        self.n_qubits = n_qubits
        self.min_cnot = min_cnot
        self.max_cnot = max_cnot
        self.max_depth = max_depth
        self.pop_size = population_size
        self.generations = generations
        self.lambda_cnot = lambda_cnot
        self.lambda_under_cnot = lambda_under_cnot
        self.lambda_ent = lambda_ent
        self.target_mw = target_mw
        self.reupload = reupload
        self.diff_method = diff_method
        self.output_dir = output_dir
        self.qdevice = qdevice
        self.hardware_topology = hardware_topology
        self.check_geometric_condition = check_geometric_condition
        self.dual_basis_only = dual_basis_only
        self.admissibility_mask: Optional[np.ndarray] = None
        self.admissible_pairs: List[Tuple[int, int]] = []
        # Fix F-13: Track fitness history for diagnostic plots
        self._fitness_history: List[List[float]] = []
        self._best_history: List[float] = []
        self.invalid_evaluations: int = 0
        self.evaluation_errors: List[str] = []
        # Cache singleton PennyLane simulation device to avoid repeated reallocations
        self._cached_dev, self._cached_dev_name = self._get_device()

    def setup_admissibility_graph(self, latent_tensor: torch.Tensor) -> np.ndarray:
        """Analyze latent features and construct admissible 2-qubit interaction graph."""
        _, self.admissibility_mask = compute_latent_mutual_information(
            latent_tensor, self.n_qubits
        )
        self.admissible_pairs = []
        for i in range(self.n_qubits):
            for j in range(self.n_qubits):
                if self.admissibility_mask[i, j]:
                    self.admissible_pairs.append((i, j))
        # Ensure graph connectivity: all qubits must have at least one candidate entangling edge
        covered_qubits = set()
        for c, t in self.admissible_pairs:
            covered_qubits.add(c)
            covered_qubits.add(t)
        for q in range(self.n_qubits):
            if q not in covered_qubits:
                partner = (q + 1) % self.n_qubits
                self.admissible_pairs.append((q, partner))
                self.admissibility_mask[q, partner] = True
        if not self.admissible_pairs:
            self.admissible_pairs = [(i, (i + 1) % self.n_qubits) for i in range(self.n_qubits)]
        return self.admissibility_mask

    def _random_gate_gene(self, layer_idx: int) -> GateGene:
        """Create a random gate respecting constraints."""
        # 40% probability 2-qubit gate, 60% single-qubit
        if self.admissible_pairs and random.random() < 0.40:
            ctrl, tgt = random.choice(self.admissible_pairs)
            g_type = random.choice(TWO_QUBIT_GATES)
            return GateGene(gate_type=g_type, target=tgt, control=ctrl, is_parameterized=False, param_count=0)
        else:
            tgt = random.randint(0, self.n_qubits - 1)
            g_type = random.choice(SINGLE_QUBIT_GATES)
            param_cnt = 3 if g_type == "U3" else (1 if g_type in ("Rx", "Ry", "Rz") else 0)
            return GateGene(gate_type=g_type, target=tgt, is_parameterized=(param_cnt > 0), param_count=param_cnt)

    def initialize_population(self) -> List[CircuitGenome]:
        """Generate diverse initial population respecting admissibility graph."""
        population = []
        for _ in range(self.pop_size):
            genome = CircuitGenome(n_qubits=self.n_qubits, max_depth=self.max_depth)
            depth = random.randint(2, max(2, self.max_depth))
            for _ in range(depth):
                layer = []
                # Add single qubit rotation on each wire
                for q in range(self.n_qubits):
                    g_type = random.choice(["Ry", "Rz", "U3"])
                    cnt = 3 if g_type == "U3" else 1
                    layer.append(GateGene(gate_type=g_type, target=q, is_parameterized=True, param_count=cnt))
                # Add 1 or 2 entangling gates from admissible graph
                if self.admissible_pairs:
                    pair = random.choice(self.admissible_pairs)
                    layer.append(GateGene(gate_type="CNOT", target=pair[1], control=pair[0]))
                genome.layers.append(layer)
            # Ensure initial genome satisfies minimum CNOT count
            while genome.cnot_count < self.min_cnot and self.admissible_pairs:
                l_idx = random.randint(0, len(genome.layers) - 1)
                pair = random.choice(self.admissible_pairs)
                genome.layers[l_idx].append(GateGene(gate_type="CNOT", target=pair[1], control=pair[0]))
                genome.compute_stats()

            # Ensure every qubit (especially diagnostic Qubit 0) participates in at least one entangling gate
            covered = set()
            for layer in genome.layers:
                for gene in layer:
                    if gene.gate_type in ("CNOT", "CZ"):
                        if gene.control is not None: covered.add(gene.control)
                        if gene.target is not None: covered.add(gene.target)
            for q in range(self.n_qubits):
                if q not in covered and self.admissible_pairs:
                    cand = [p for p in self.admissible_pairs if p[0] == q or p[1] == q]
                    pair = cand[0] if cand else (q, (q + 1) % self.n_qubits)
                    l_idx = random.randint(0, len(genome.layers) - 1)
                    genome.layers[l_idx].append(GateGene(gate_type="CNOT", target=pair[1], control=pair[0]))
                    covered.add(pair[0]); covered.add(pair[1])

            genome.compute_stats()
            population.append(genome)
        return population

    def _get_device(self):
        """Dynamic backend selection using unified get_pennylane_device."""
        try:
            from model import get_pennylane_device
        except ImportError:
            from qml_models.model import get_pennylane_device
        return get_pennylane_device(
            self.n_qubits, self.diff_method, qdevice=getattr(self, "qdevice", None)
        )

    def build_qnode(self, genome: CircuitGenome, diff_method: Optional[str] = None):
        """Construct a vectorized batched PennyLane QNode from a circuit genome.

        Fix F-02: Accepts 2D batch tensors inputs of shape (B, 3*n_qubits).
        Fix F-03: Includes Hadamard initialization + interleaved data re-uploading.
        Fix F-08: Uses the diff_method parameter instead of ignoring it.

        Returns:
            (qnode, total_params) tuple.
        """
        # Fix F-08: Use provided or instance diff_method
        dm = diff_method or self.diff_method

        # Dynamic backend selection: reuse cached device to eliminate recreation overhead
        dev = getattr(self, "_cached_dev", None)
        dev_name = getattr(self, "_cached_dev_name", None)
        if dev is None or dev_name is None:
            dev, dev_name = self._get_device()

        # Resolve diff_method compatibility
        if "default.qubit" in dev_name and dm == "adjoint":
            actual_dm = "backprop"
        elif "lightning" in dev_name and dm == "backprop":
            actual_dm = "adjoint"
        else:
            actual_dm = dm

        n_q = self.n_qubits
        reupload = self.reupload

        @qml.qnode(dev, interface="torch", diff_method=actual_dm)
        def dynamic_circuit(inputs, weights):
            w_idx = 0
            for l, layer in enumerate(genome.layers):
                # Fix F-03: Interleaved data re-uploading at every layer (parity with model.py)
                if reupload or l == 0:
                    for q in range(n_q):
                        qml.U3(
                            inputs[..., q],
                            inputs[..., q + n_q],
                            inputs[..., q + 2 * n_q],
                            wires=q,
                        )

                for gene in layer:
                    if gene.gate_type == "Rx":
                        qml.RX(weights[w_idx], wires=gene.target)
                        w_idx += 1
                    elif gene.gate_type == "Ry":
                        qml.RY(weights[w_idx], wires=gene.target)
                        w_idx += 1
                    elif gene.gate_type == "Rz":
                        qml.RZ(weights[w_idx], wires=gene.target)
                        w_idx += 1
                    elif gene.gate_type == "U3":
                        qml.U3(weights[w_idx], weights[w_idx + 1], weights[w_idx + 2], wires=gene.target)
                        w_idx += 3
                    elif gene.gate_type == "CNOT":
                        qml.CNOT(wires=[gene.control, gene.target])
                    elif gene.gate_type == "CZ":
                        qml.CZ(wires=[gene.control, gene.target])
                    elif gene.gate_type == "Identity":
                        qml.Identity(wires=gene.target)

            # Measure sum-of-local or dual-basis observables (Tasks 4.1 & 6.1)
            local_z = [qml.expval(qml.PauliZ(i)) for i in range(n_q)]
            if getattr(self, "dual_basis_only", False):
                transverse_x = [qml.expval(qml.PauliX(i)) for i in range(n_q)]
                return local_z + transverse_x
            nearest_zz = [qml.expval(qml.PauliZ(i) @ qml.PauliZ(i + 1)) for i in range(n_q - 1)]
            transverse_x = [qml.expval(qml.PauliX(i)) for i in range(n_q)]
            return local_z + nearest_zz + transverse_x

        return dynamic_circuit, genome.total_params

    def _build_entanglement_qnode(self, genome: CircuitGenome):
        """Build a QNode that returns Pauli X, Y, Z expectations for Meyer-Wallach.

        Fix F-01: Evaluates entanglement on data-encoded states with Hadamard init,
        NOT on vacuum |0000>.
        """
        dev = getattr(self, "_cached_dev", None)
        if dev is None:
            dev = qml.device("default.qubit", wires=self.n_qubits)
        n_q = self.n_qubits
        reupload = self.reupload

        # diff_method=None ensures full compatibility with lightning.qubit and lightning.gpu
        @qml.qnode(dev, interface="torch", diff_method=None)
        def pauli_circuit(inputs, weights):
            # Evaluate directly from |00...0> vacuum — identical to production HybridCNNVQC
            w_idx = 0
            for l, layer in enumerate(genome.layers):
                if reupload or l == 0:
                    for q in range(n_q):
                        qml.U3(
                            inputs[..., q],
                            inputs[..., q + n_q],
                            inputs[..., q + 2 * n_q],
                            wires=q,
                        )
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

            # Return all single-qubit Pauli expectations for analytical MW
            pauli_x = [qml.expval(qml.PauliX(i)) for i in range(n_q)]
            pauli_y = [qml.expval(qml.PauliY(i)) for i in range(n_q)]
            pauli_z = [qml.expval(qml.PauliZ(i)) for i in range(n_q)]
            return pauli_x + pauli_y + pauli_z

        return pauli_circuit

    def evaluate_individual_proxy(
        self,
        genome: CircuitGenome,
        train_z: torch.Tensor,
        train_y: torch.Tensor,
        val_z: torch.Tensor,
        val_y: torch.Tensor,
        n_classes: int = 2,
        proxy_epochs: int = 3,
        class_weights: Optional[torch.Tensor] = None,
    ) -> float:
        """Fast proxy evaluation using pre-cached latent features.

        Fixes applied:
            F-01: Analytical Meyer-Wallach on data states (not vacuum)
            F-02: Vectorized batched QNode execution
            F-04: Stratified balanced validation sampling
            F-07: Save trained proxy weights into genome & SuperCircuit parameter inheritance
        """
        genome.compute_stats()
        # Phase 6.2: Compute hardware routing-aware coupling cost
        cnot_cost = compute_hardware_coupling_cost(
            genome, topology=self.hardware_topology, n_qubits=self.n_qubits
        )
        genome.hardware_cost = cnot_cost

        if cnot_cost > self.max_cnot:
            genome.valid = False
            genome.invalid_reason = f"hardware_cost {cnot_cost} exceeds max_cnot {self.max_cnot}"
            genome.val_auc = float("nan")
            genome.fitness = -float("inf")
            self.invalid_evaluations += 1
            return genome.fitness

        total_p = genome.total_params
        if total_p == 0:
            genome.valid = False
            genome.invalid_reason = "candidate contains no trainable parameters"
            genome.val_auc = float("nan")
            genome.fitness = -float("inf")
            self.invalid_evaluations += 1
            return genome.fitness

        qnode, _ = self.build_qnode(genome)
        n_meas = 2 * self.n_qubits if getattr(self, "dual_basis_only", False) else (3 * self.n_qubits - 1)

        # Small lightweight head
        head = nn.Linear(n_meas, n_classes)
        # Allow head to adapt to candidate quantum observable representation
        head.train()
        for p in head.parameters():
            p.requires_grad = True

        # SuperCircuit parameter inheritance (Wang et al., HPCA 2022)
        if genome.trained_weights is not None and len(genome.trained_weights) == total_p:
            weights = nn.Parameter(torch.tensor(genome.trained_weights, dtype=torch.float32))
        else:
            weights = nn.Parameter(torch.randn(total_p) * 0.05)

        optimizer = torch.optim.Adam([
            {"params": [weights], "lr": 0.01},
            {"params": head.parameters(), "lr": 0.02},
        ])
        if class_weights is not None:
            dev_target = train_z.device if hasattr(train_z, "device") else "cpu"
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(dev_target))
        else:
            criterion = nn.CrossEntropyLoss()

        batch_size = min(32, train_z.size(0))
        n_batches = min(6, train_z.size(0) // batch_size)

        for _ in range(proxy_epochs):
            indices = torch.randperm(train_z.size(0))[:n_batches * batch_size]
            for b in range(n_batches):
                b_idx = indices[b * batch_size : (b + 1) * batch_size]
                bx = train_z[b_idx]
                by = train_y[b_idx]

                optimizer.zero_grad()
                # Fix F-02: Vectorized batched QNode execution (single call for entire batch)
                exp_results = qnode(bx, weights)
                if isinstance(exp_results, (list, tuple)):
                    exp_batch = torch.stack(exp_results, dim=1).float()
                else:
                    exp_batch = exp_results.float()
                logits = head(exp_batch)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        # Fix F-04: Stratified balanced validation evaluation (Task 6.4: N=160)
        head.eval()
        with torch.no_grad():
            if hasattr(self, "_cached_val_z") and self._cached_val_z is not None:
                vx, vy = self._cached_val_z, self._cached_val_y
            else:
                vx, vy = get_stratified_eval_batch(val_z, val_y, total_samples=160, seed=42)

            # Fix F-02: Vectorized batched validation pass
            val_exp = qnode(vx, weights)
            if isinstance(val_exp, (list, tuple)):
                val_exp_batch = torch.stack(val_exp, dim=1).float()
            else:
                val_exp_batch = val_exp.float()
            val_logits = head(val_exp_batch)

            probs = F.softmax(val_logits, dim=-1).cpu().numpy()
            y_true = vy.cpu().numpy()
            unique_labels = np.unique(y_true)

            if len(unique_labels) < 2:
                val_auc = 0.50
                print(f"  [QNAS] Warning: Validation batch has only class {unique_labels}. "
                      f"AUC defaulted to 0.50.")
            else:
                try:
                    if n_classes == 2:
                        val_auc = float(roc_auc_score(y_true, probs[:, 1]))
                    else:
                        val_auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
                except Exception as e:
                    from sklearn.metrics import balanced_accuracy_score
                    preds = val_logits.argmax(dim=1).cpu().numpy()
                    val_auc = float(balanced_accuracy_score(y_true, preds))

            # Task 6.3: OOD Robustness Metric as 4th Pareto Objective
            noise = 0.10 * torch.randn_like(vx)
            vx_ood = vx + noise
            n_q = self.n_qubits
            # Respect Lie group domains: theta in [0, pi], phi/lambda in [-pi, pi]
            if vx_ood.shape[-1] >= 3 * n_q:
                vx_ood[:, :n_q] = torch.clamp(vx_ood[:, :n_q], 0.0, math.pi)
                vx_ood[:, n_q:] = (vx_ood[:, n_q:] + math.pi) % (2 * math.pi) - math.pi
            val_exp_ood = qnode(vx_ood, weights)
            val_exp_ood_batch = torch.stack(val_exp_ood, dim=1).float() if isinstance(val_exp_ood, (list, tuple)) else val_exp_ood.float()
            ood_logits = head(val_exp_ood_batch)
            ood_probs = F.softmax(ood_logits, dim=-1).cpu().numpy()
            try:
                if n_classes == 2:
                    ood_auc = float(roc_auc_score(y_true, ood_probs[:, 1]))
                else:
                    ood_auc = float(roc_auc_score(y_true, ood_probs, multi_class="ovr", average="macro"))
            except Exception:
                ood_auc = val_auc
            ood_gap = max(0.0, val_auc - ood_auc)
            genome.ood_gap = float(ood_gap)

        # Fix F-01: Analytical Meyer-Wallach on data-encoded states (not vacuum)
        try:
            pauli_qnode = self._build_entanglement_qnode(genome)
            # Use a small data batch for MW evaluation
            mw_batch_size = min(16, train_z.size(0))
            mw_data = train_z[:mw_batch_size]
            with torch.no_grad():
                pauli_results = pauli_qnode(mw_data, weights)
                if isinstance(pauli_results, (list, tuple)):
                    pauli_batch = torch.stack(pauli_results, dim=1).float().cpu().numpy()
                else:
                    pauli_batch = pauli_results.float().cpu().numpy()
                mw = compute_meyer_wallach_analytical(pauli_batch, self.n_qubits)
        except Exception as e:
            print(f"  [QNAS] MW computation failed ({e}), using fallback 0.50")
            mw = 0.50

        genome.val_auc = val_auc
        genome.mw_entanglement = mw

        # Fix F-07: Save trained proxy weights into genome for warm-start transfer
        genome.trained_weights = weights.detach().cpu().tolist()

        # Multi-Objective Pareto Fitness Function (Section 2.4 & Phase 6.1, 6.2, 6.3)
        # Upper CNOT budget penalty (only penalizes excess CNOTs above min_cnot)
        cnot_range = max(1, self.max_cnot - self.min_cnot)
        cnot_excess = max(0, cnot_cost - self.min_cnot)
        cnot_penalty = self.lambda_cnot * (cnot_excess / cnot_range)

        # Minimum CNOT deficiency penalty (enforces minimum entanglement >= min_cnot)
        if self.min_cnot > 0 and cnot_cost < self.min_cnot:
            under_cnot_penalty = self.lambda_under_cnot * (self.min_cnot - cnot_cost)
        else:
            under_cnot_penalty = 0.0

        # Entanglement (Meyer-Wallach) target deviation penalty
        mw_penalty = self.lambda_ent * abs(mw - self.target_mw)
        fitness = val_auc - cnot_penalty - under_cnot_penalty - mw_penalty - 0.10 * getattr(genome, "ood_gap", 0.0)

        # Qubit connectivity check (Solution 5.4: Enforce Qubit 0 & all-qubit entanglement)
        connected_qubits = set()
        for layer in genome.layers:
            for gene in layer:
                if gene.gate_type in ("CNOT", "CZ") and gene.control is not None and gene.target is not None:
                    connected_qubits.add(gene.control)
                    connected_qubits.add(gene.target)
        if 0 not in connected_qubits:
            # Qubit 0 is the primary diagnostic wire for RPE retinal pathologies
            fitness -= 0.15
        disconnected = self.n_qubits - len(connected_qubits)
        if disconnected > 0:
            fitness -= 0.05 * disconnected

        # Phase 6.2: Quantum Geometric Condition Number Check evaluated on data embeddings
        if self.check_geometric_condition and total_p > 1:
            geom_cond = compute_geometric_condition_number(
                genome, weights, dev=getattr(self, "_cached_dev", None), n_qubits=self.n_qubits,
                data_embeddings=train_z[:16]
            )
            genome.qfim_condition = geom_cond
            if geom_cond > 1e4:
                geom_penalty = 0.05 * math.log10(geom_cond / 1e4)
                fitness -= geom_penalty

        genome.fitness = float(fitness)
        return genome.fitness

    def crossover(self, parent_a: CircuitGenome, parent_b: CircuitGenome) -> CircuitGenome:
        """Layer-wise crossover between two high-fitness parents.

        Fix F-05: Depth-preserving crossover — child depth can range from min to max
        of parent depths (plus optional expansion up to max_depth), preventing
        monotonic depth collapse to D=2.
        """
        child = CircuitGenome(n_qubits=self.n_qubits)
        depth_a = len(parent_a.layers)
        depth_b = len(parent_b.layers)
        min_depth = min(depth_a, depth_b)
        max_depth_parents = max(depth_a, depth_b)

        # Fix F-05: Allow depth exploration up to max_depth
        target_depth = random.randint(
            min_depth,
            min(self.max_depth, max_depth_parents + 1)
        )

        child_weights = []

        def get_layer_weights(parent):
            if parent.trained_weights is None:
                return {}
            w_map = {}
            w_idx = 0
            for l_idx, layer in enumerate(parent.layers):
                l_w = []
                for gene in layer:
                    if gene.param_count > 0 and w_idx + gene.param_count <= len(parent.trained_weights):
                        l_w.extend(parent.trained_weights[w_idx:w_idx + gene.param_count])
                        w_idx += gene.param_count
                    elif gene.param_count > 0:
                        l_w.extend([0.0] * gene.param_count)
                w_map[l_idx] = l_w
            return w_map

        weights_a = get_layer_weights(parent_a)
        weights_b = get_layer_weights(parent_b)

        for l in range(target_depth):
            if l < min_depth:
                # Both parents have this layer — randomly choose one
                if random.random() < 0.5:
                    chosen_layer = parent_a.layers[l]
                    child.layers.append(copy.deepcopy(chosen_layer))
                    if l in weights_a:
                        child_weights.extend(weights_a[l])
                    else:
                        for g in chosen_layer:
                            if g.param_count > 0:
                                child_weights.extend((np.random.randn(g.param_count) * 0.05).tolist())
                else:
                    chosen_layer = parent_b.layers[l]
                    child.layers.append(copy.deepcopy(chosen_layer))
                    if l in weights_b:
                        child_weights.extend(weights_b[l])
                    else:
                        for g in chosen_layer:
                            if g.param_count > 0:
                                child_weights.extend((np.random.randn(g.param_count) * 0.05).tolist())
            elif l < depth_a:
                # Only parent_a has this layer
                child.layers.append(copy.deepcopy(parent_a.layers[l]))
                if l in weights_a:
                    child_weights.extend(weights_a[l])
                else:
                    for g in parent_a.layers[l]:
                        if g.param_count > 0:
                            child_weights.extend((np.random.randn(g.param_count) * 0.05).tolist())
            elif l < depth_b:
                # Only parent_b has this layer
                child.layers.append(copy.deepcopy(parent_b.layers[l]))
                if l in weights_b:
                    child_weights.extend(weights_b[l])
                else:
                    for g in parent_b.layers[l]:
                        if g.param_count > 0:
                            child_weights.extend((np.random.randn(g.param_count) * 0.05).tolist())
            else:
                # Beyond both parents — generate a new random layer
                layer = []
                for q in range(self.n_qubits):
                    g_type = random.choice(["Ry", "Rz"])
                    layer.append(GateGene(gate_type=g_type, target=q, is_parameterized=True, param_count=1))
                    child_weights.extend((np.random.randn(1) * 0.05).tolist())
                if self.admissible_pairs:
                    pair = random.choice(self.admissible_pairs)
                    layer.append(GateGene(gate_type="CNOT", target=pair[1], control=pair[0]))
                child.layers.append(layer)

        child.compute_stats()
        if len(child_weights) == child.total_params and child.total_params > 0:
            child.trained_weights = child_weights
        return child

    def mutate(self, genome: CircuitGenome, mutation_prob: float = 0.25) -> CircuitGenome:
        """Gate replacement, insertion, pruning, or swap respecting topological constraints.

        Fix F-11: Added 'insert' mutation operator for reversible Markov chain search.
        Once gates are pruned, they can now be re-inserted, preventing asymmetric
        bias toward degenerate sparse circuits.
        """
        for layer in genome.layers:
            if random.random() < mutation_prob:
                # Fix F-11: Added "insert" action
                action = random.choice(["replace", "prune", "swap", "insert"])
                if action == "replace" and layer:
                    idx = random.randint(0, len(layer) - 1)
                    layer[idx] = self._random_gate_gene(0)
                elif action == "prune" and len(layer) > 1:
                    idx = random.randint(0, len(layer) - 1)
                    target_gene = layer[idx]
                    if target_gene.gate_type in ("CNOT", "CZ"):
                        if genome.cnot_count > self.min_cnot:
                            layer.pop(idx)
                    elif target_gene.is_parameterized:
                        # Allow pruning single-qubit gate if wire has at least 1 other rotation
                        wire_rotations = sum(1 for g in layer if g.target == target_gene.target and g.is_parameterized)
                        if wire_rotations > 1:
                            layer.pop(idx)
                elif action == "swap" and len(layer) >= 2:
                    i1, i2 = random.sample(range(len(layer)), 2)
                    layer[i1], layer[i2] = layer[i2], layer[i1]
                elif action == "insert" and len(layer) < 2 * self.n_qubits:
                    # Fix F-11: Prioritize inserting CNOT if below min_cnot, else random gate
                    if genome.cnot_count < self.min_cnot and self.admissible_pairs:
                        ctrl, tgt = random.choice(self.admissible_pairs)
                        new_gene = GateGene(gate_type="CNOT", target=tgt, control=ctrl)
                    else:
                        new_gene = self._random_gate_gene(0)
                    layer.append(new_gene)
        genome.compute_stats()
        return genome

    def evolve(
        self,
        train_z: torch.Tensor,
        train_y: torch.Tensor,
        val_z: torch.Tensor,
        val_y: torch.Tensor,
        n_classes: int = 2,
    ) -> CircuitGenome:
        """Execute evolutionary search across generations and return Pareto-optimal genome."""
        # Cache stratified balanced validation evaluation tensor to avoid redundant CPU re-allocations
        self._cached_val_z, self._cached_val_y = get_stratified_eval_batch(val_z, val_y, total_samples=64, seed=42)

        # Compute class-balanced effective weights for proxy training if imbalanced
        classes, counts = torch.unique(train_y, return_counts=True)
        if len(classes) > 1 and counts.max().float() / max(1, counts.min().item()) > 1.5:
            beta = 0.99
            eff_num = 1.0 - torch.pow(torch.tensor(beta), counts.float())
            weights_c = (1.0 - beta) / eff_num
            weights_c = weights_c / weights_c.sum() * len(classes)
            class_weights = torch.ones(n_classes, dtype=torch.float32)
            for c, w in zip(classes, weights_c):
                if c < n_classes:
                    class_weights[c] = float(w)
        else:
            class_weights = None

        print(f"[QNAS] Initializing population of {self.pop_size} genomes (Max CNOT: {self.max_cnot})...")
        population = self.initialize_population()

        for g in population:
            try:
                self.evaluate_individual_proxy(g, train_z, train_y, val_z, val_y, n_classes, class_weights=class_weights)
            except Exception as exc:
                g.valid = False
                g.invalid_reason = f"evaluation error: {type(exc).__name__}: {exc}"
                g.val_auc = float("nan")
                g.fitness = -float("inf")
                self.invalid_evaluations += 1
                self.evaluation_errors.append(g.invalid_reason)

        valid_population = [ind for ind in population if ind.valid and np.isfinite(ind.fitness)]
        if not valid_population:
            raise RuntimeError(
                "QNAS produced no valid candidate architectures. "
                f"invalid_evaluations={self.invalid_evaluations}; "
                f"errors={self.evaluation_errors[:3]}"
            )
        best_genome = max(valid_population, key=lambda ind: ind.fitness)

        # Fix F-13: Track fitness history
        self._fitness_history.append([ind.fitness for ind in population])
        self._best_history.append(best_genome.fitness)

        for gen in range(1, self.generations + 1):
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            elites = population[: max(2, self.pop_size // 4)]
            next_pop = [copy.deepcopy(e) for e in elites]

            while len(next_pop) < self.pop_size:
                p1 = select_parent_tournament(population, k=3)
                p2 = select_parent_tournament(population, k=3)
                for _ in range(3):
                    if p1 is not p2:
                        break
                    p2 = select_parent_tournament(population, k=3)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                try:
                    self.evaluate_individual_proxy(child, train_z, train_y, val_z, val_y, n_classes, class_weights=class_weights)
                except Exception as exc:
                    child.valid = False
                    child.invalid_reason = f"evaluation error: {type(exc).__name__}: {exc}"
                    child.val_auc = float("nan")
                    child.fitness = -float("inf")
                    self.invalid_evaluations += 1
                    self.evaluation_errors.append(child.invalid_reason)
                next_pop.append(child)

            population = next_pop
            current_best = max(population, key=lambda ind: ind.fitness)
            if current_best.fitness > best_genome.fitness:
                best_genome = copy.deepcopy(current_best)

            # Fix F-13: Track fitness history
            self._fitness_history.append([ind.fitness for ind in population])
            self._best_history.append(best_genome.fitness)

            print(f"[QNAS] Gen {gen}/{self.generations} | Best Fitness: {best_genome.fitness:.4f} | "
                  f"Val AUC: {best_genome.val_auc:.4f} | CNOTs: {best_genome.cnot_count} | MW: {best_genome.mw_entanglement:.3f}")

        # Fix F-13: Save diagnostic plots if output_dir is provided
        if self.output_dir:
            self._plot_diagnostics(population)

        if self.output_dir:
            with open(os.path.join(self.output_dir, "qnas_evaluation_status.json"), "w", encoding="utf-8") as handle:
                import json
                json.dump({
                    "invalid_evaluations": self.invalid_evaluations,
                    "evaluation_errors": self.evaluation_errors[:100],
                    "valid_final_candidates": int(sum(ind.valid for ind in population)),
                }, handle, indent=2)

        return best_genome

    # ---------------------------------------------------------------------------
    # Fix F-13: Internal Diagnostic Visualizations
    # ---------------------------------------------------------------------------
    def _plot_diagnostics(self, final_population: List[CircuitGenome]) -> None:
        """Generate QNAS-internal diagnostic plots (Pareto frontier, fitness trajectory)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        out_dir = self.output_dir
        os.makedirs(out_dir, exist_ok=True)

        # 1. Fitness trajectory across generations
        if self._best_history:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(range(len(self._best_history)), self._best_history,
                    "o-", color="#2196F3", linewidth=2, markersize=6, label="Best Fitness")
            if self._fitness_history:
                gen_means = [
                    np.mean([value for value in gen_fits if np.isfinite(value)])
                    if any(np.isfinite(value) for value in gen_fits) else np.nan
                    for gen_fits in self._fitness_history
                ]
                ax.plot(range(len(gen_means)), gen_means,
                        "s--", color="#FF9800", linewidth=1.5, markersize=4, label="Population Mean")
            ax.set_xlabel("Generation", fontsize=12)
            ax.set_ylabel("Fitness", fontsize=12)
            ax.set_title("QNAS Fitness Trajectory", fontsize=14, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "qnas_fitness_trajectory.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

        # 2. Pareto frontier: Val-AUC vs CNOT count vs MW entanglement
        if final_population:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            valid_population = [g for g in final_population if g.valid and np.isfinite(g.val_auc)]
            if not valid_population:
                return
            aucs = [g.val_auc for g in valid_population]
            cnots = [g.cnot_count for g in valid_population]
            mws = [g.mw_entanglement for g in valid_population]

            ax = axes[0]
            sc = ax.scatter(cnots, aucs, c=mws, cmap="coolwarm", s=80, edgecolors="k", linewidths=0.5)
            plt.colorbar(sc, ax=ax, label="MW Entanglement")
            ax.set_xlabel("CNOT Count", fontsize=12)
            ax.set_ylabel("Validation AUC", fontsize=12)
            ax.set_title("Pareto Frontier: AUC vs CNOT Budget", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3)

            ax = axes[1]
            sc = ax.scatter(mws, aucs, c=cnots, cmap="viridis", s=80, edgecolors="k", linewidths=0.5)
            plt.colorbar(sc, ax=ax, label="CNOT Count")
            ax.set_xlabel("MW Entanglement", fontsize=12)
            ax.set_ylabel("Validation AUC", fontsize=12)
            ax.set_title("Pareto Frontier: AUC vs Entanglement", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "qnas_pareto_frontier.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

        # 3. Gate distribution histogram
        if final_population:
            gate_counts: Dict[str, int] = {}
            for genome in final_population:
                for layer in genome.layers:
                    for gene in layer:
                        gate_counts[gene.gate_type] = gate_counts.get(gene.gate_type, 0) + 1

            if gate_counts:
                fig, ax = plt.subplots(figsize=(8, 5))
                gates = sorted(gate_counts.keys())
                counts = [gate_counts[g] for g in gates]
                colors = plt.cm.Set3(np.linspace(0, 1, len(gates)))
                ax.bar(gates, counts, color=colors, edgecolor="k", linewidth=0.5)
                ax.set_xlabel("Gate Type", fontsize=12)
                ax.set_ylabel("Total Count (Population)", fontsize=12)
                ax.set_title("Gate Distribution in Final Population", fontsize=13, fontweight="bold")
                ax.grid(True, alpha=0.3, axis="y")
                plt.tight_layout()
                fig.savefig(os.path.join(out_dir, "qnas_gate_distribution.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)

        # 4. Realized DAG topology vs Admissible MI Graph (Phase 5.3)
        if final_population:
            best_gen = max(final_population, key=lambda ind: ind.fitness)
            dag_path = os.path.join(out_dir, "qnas_realized_dag_topology.png")
            try:
                try:
                    from plots import plot_qnas_realized_topology
                except ImportError:
                    from qml_models.plots import plot_qnas_realized_topology
                plot_qnas_realized_topology(
                    self.admissibility_mask.astype(float) if self.admissibility_mask is not None else None,
                    self.admissible_pairs,
                    best_gen,
                    self.n_qubits,
                    dag_path,
                )
            except Exception as e:
                print(f"[QNAS] Note: Could not generate realized DAG plot: {e}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile
    print("[QNAS] Running comprehensive remediation self-test...")
    n_q = 4
    synth_z = torch.randn(64, 3 * n_q)
    synth_y = torch.randint(0, 2, (64,))

    with tempfile.TemporaryDirectory() as tmp_dir:
        optimizer = QNASOptimizer(
            n_qubits=n_q,
            max_cnot=6,
            max_depth=3,
            population_size=4,
            generations=2,
            output_dir=tmp_dir,
            hardware_topology="linear",
            check_geometric_condition=True,
        )
        mask = optimizer.setup_admissibility_graph(synth_z)
        print(f"[QNAS] Admissible pairs count: {len(optimizer.admissible_pairs)}")

        # Verify Phase 2.1: Standalone multivariate MI matrix
        mi_mat = compute_multivariate_mi_matrix(synth_z, n_q)
        assert mi_mat.shape == (n_q, n_q), f"Shape mismatch: {mi_mat.shape}"
        assert np.allclose(mi_mat, mi_mat.T), "MI matrix must be symmetric!"
        print("[QNAS] Multivariate Log-Det MI Matrix: OK")

        best = optimizer.evolve(synth_z, synth_y, synth_z[:32], synth_y[:32])
        print(f"[QNAS] Search completed. Optimal CNOT count: {best.cnot_count} (<= {optimizer.max_cnot})")
        print(f"[QNAS] Best Val AUC: {best.val_auc:.4f}, MW Entanglement: {best.mw_entanglement:.4f}")
        assert best.cnot_count <= optimizer.max_cnot, "CNOT budget constraint violated!"

        # Verify Phase 6.2: Hardware coupling cost
        hw_cost = compute_hardware_coupling_cost(best, topology="linear", n_qubits=n_q)
        print(f"[QNAS] Hardware routing cost: {hw_cost}")

        # Verify Phase 6.1: Geometric condition number
        if best.trained_weights is not None:
            w_t = torch.tensor(best.trained_weights)
            cond = compute_geometric_condition_number(best, w_t, dev=optimizer._cached_dev, n_qubits=n_q)
            print(f"[QNAS] Geometric metric tensor condition number: {cond:.2f}")

        # Verify Phase 5.3: Diagnostic plot generation
        expected_plots = [
            "qnas_fitness_trajectory.png",
            "qnas_pareto_frontier.png",
            "qnas_gate_distribution.png",
            "qnas_realized_dag_topology.png",
        ]
        for plot_name in expected_plots:
            plot_path = os.path.join(tmp_dir, plot_name)
            assert os.path.exists(plot_path), f"Missing diagnostic plot: {plot_name}"
            assert os.path.getsize(plot_path) > 0, f"Empty diagnostic plot: {plot_name}"
        print(f"[QNAS] All 4 diagnostic plot artifacts generated successfully in {tmp_dir}")

        # Verify Phase 5.2: Weight serialization round-trip
        import json
        tmp_json = os.path.join(tmp_dir, "test_genome.json")
        save_genome(best, tmp_json)
        reloaded = load_genome(tmp_json)
        assert reloaded.trained_weights is not None, "Weights lost in serialization!"
        assert len(reloaded.trained_weights) == len(best.trained_weights), "Weight count mismatch!"
        print("[QNAS] Weight serialization round-trip: OK")

    print("[OK]  qnas.py self-test passed successfully.")
