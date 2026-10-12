"""
qaf.py — Quantum Attention-Fourier (QAF) Circuit Architecture (Remediated)
===========================================================================
Implements the remediated 3-stage HQCNN circuit with:
  Stage 1: Full U3(θ, φ, λ) data encoding per qubit (replaces RY-only slicing)
  Stage 2: Interleaved data re-uploading + hardware-efficient nearest-neighbor entanglers
  Stage 3: Per-qubit Rot variational rotations with decoupled CRZ weights

Key Remediations (from short_learning_fix.md and qml_architecture_remediation_plan.md):
  - Fix B6: Full 3-angle U3 encoding eliminates 66.7% feature discarding
  - Fix B7: Weight tensor expanded to (n_layers, n_qubits, 4) — CRZ and Rot use independent params
  - NISQ pruning: Dyadic all-to-all CRZ mesh replaced with nearest-neighbor linear entanglers
  - Toffoli gates replaced with parameterized CZ for hardware efficiency
  - Universal data re-uploading at every layer for expanded Fourier expressivity

Dual-Basis Readout: [⟨Z_0⟩..⟨Z_{n-1}⟩, ⟨X_0⟩..⟨X_{n-1}⟩] ∈ ℝ^{2n}
  → 16 dimensions for n=8, guaranteeing full-rank Linear(16→7) head.
"""

from __future__ import annotations

import math
import pennylane as qml


def build_qaf_circuit(
    features,
    weights,
    n_qubits: int = 8,
    n_layers: int = 3,
):
    r"""Execute the remediated QAF circuit body inside an existing QNode context.

    Remediated architecture per layer:
      1. Universal Data Re-uploading: U3(θ_q, φ_q, λ_q) on each qubit
         — ingests ALL 3*n_qubits coordinates from the CQ adapter
      2. Nearest-neighbor linear CNOT entanglers (pruned from all-to-all)
      3. Parameterized CRZ nearest-neighbor phases (decoupled weight index 0)
      4. Per-qubit Rot(w[l,q,1], w[l,q,2], w[l,q,3]) variational rotations

    Parameters
    ----------
    features : tensor [..., 3*n_qubits or n_qubits]
        Quantum angle inputs. If shape[-1] >= 3*n_qubits, slices into (θ, φ, λ).
        If shape[-1] == n_qubits, uses as θ with φ=λ=0 (backward compat).
    weights  : tensor (n_layers, n_qubits, 4) or (n_layers, n_qubits, 3)
        Variational parameters. Shape (..., 4) uses decoupled CRZ+Rot.
        Shape (..., 3) falls back to legacy shared-weight mode.
    n_qubits : int
    n_layers : int
    """
    # ── Parse features into (θ, φ, λ) per qubit ─────────────────────────
    feat_dim = features.shape[-1] if hasattr(features, "shape") else n_qubits
    if feat_dim >= 3 * n_qubits:
        theta = features[..., :n_qubits]
        phi   = features[..., n_qubits : 2 * n_qubits]
        lam   = features[..., 2 * n_qubits : 3 * n_qubits]
    else:
        # Backward compatibility: single-angle input (theta only)
        theta = features[..., :n_qubits] if feat_dim > n_qubits else features
        phi   = theta * 0.0  # zeros
        lam   = theta * 0.0  # zeros

    # Determine weight format: (n_layers, n_qubits, 4) or legacy (n_layers, n_qubits, 3)
    w_dim = weights.shape[-1] if hasattr(weights, "shape") and weights.ndim >= 3 else 3
    use_decoupled = (w_dim >= 4)

    # ── Layered circuit ──────────────────────────────────────────────────
    for layer in range(n_layers):

        # 1. Universal Data Re-uploading: U3(θ, φ, λ) per qubit at EVERY layer
        for q in range(n_qubits):
            qml.U3(theta[..., q], phi[..., q], lam[..., q], wires=q)

        # 2. Hardware-efficient nearest-neighbor CNOT entanglement
        #    (pruned from cyclic all-to-all: reduces CNOT count dramatically)
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])
        # Ring closure: connect last qubit to first
        if n_qubits > 2:
            qml.CNOT(wires=[n_qubits - 1, 0])

        # 3. Parameterized nearest-neighbor IsingZZ phases (decoupled from Rot, single-frequency)
        for q in range(n_qubits - 1):
            if use_decoupled:
                qml.IsingZZ(weights[layer, q, 0], wires=[q, q + 1])
            else:
                phase = math.pi / (2 ** (1))
                qml.IsingZZ(phase * weights[layer, q, 0], wires=[q, q + 1])
        # Ring closure: wire final decoupled parameter to close entanglement loop
        if use_decoupled and n_qubits > 2:
            qml.IsingZZ(weights[layer, n_qubits - 1, 0], wires=[n_qubits - 1, 0])

        # 4. Per-qubit variational rotations.  In decoupled mode weight 0 is
        # reserved for the IsingZZ phase, while weights 1-3 form a complete
        # Rot gate.  Keeping all three Euler angles active avoids silently
        # allocating an untrainable parameter.
        for q in range(n_qubits):
            if use_decoupled:
                qml.Rot(
                    weights[layer, q, 1],
                    weights[layer, q, 2],
                    weights[layer, q, 3],
                    wires=q,
                )
            else:
                # Legacy weight layout (n_layers, n_qubits, 3)
                qml.RY(weights[layer, q, 1], wires=q)
                qml.RZ(weights[layer, q, 2], wires=q)



def build_qaf_qnode(
    n_qubits: int = 8,
    n_layers: int = 3,
    diff_method: str = "backprop",
    qdevice: str = "default.qubit",
    dual_basis: bool = True,
):
    """Build a standalone PennyLane QNode executing the remediated QAF circuit.

    Returns
    -------
    qnode        : qml.QNode
    weight_shapes: dict — shapes for TorchLayer (now uses 4-param weights)
    n_meas       : int  — total measurement output dimension (2*n_qubits if dual_basis)
    """
    dev = qml.device(qdevice, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        """
        inputs  : (3*n_qubits,)                — U3 angles from CQ-Adapter S² projection
        weights : (n_layers, n_qubits, 4)      — decoupled variational parameters
        """
        build_qaf_circuit(inputs, weights, n_qubits=n_qubits, n_layers=n_layers)
        observables = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        if dual_basis:
            # Dual-Basis Readout: append ⟨X_i⟩ (transverse phase coherence)
            observables += [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
        return observables

    # Decoupled weight shape: index 0 = CRZ, indices 1-3 = Rot
    weight_shapes = {"weights": (n_layers, n_qubits, 4)}
    n_meas = 2 * n_qubits if dual_basis else n_qubits
    return circuit, weight_shapes, n_meas


class HierarchicalQAFCircuit:
    """Descriptor for the remediated Hierarchical Quantum Attention-Fourier (QAF) circuit.

    Usage
    -----
    >>> circ = HierarchicalQAFCircuit(n_qubits=8, n_layers=3)
    >>> qnode, weight_shapes, n_meas = circ.build()
    >>> # weight_shapes == {"weights": (3, 8, 4)}   # 4-param decoupled
    >>> # n_meas == 16  (dual-basis: [Z,X] × 8 qubits)
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 3,
        diff_method: str = "backprop",
        qdevice: str = "default.qubit",
        dual_basis: bool = True,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.diff_method = diff_method
        self.qdevice = qdevice
        self.dual_basis = dual_basis
        self.n_params = n_layers * n_qubits * 4  # 4-param decoupled weights
        self.n_meas = 2 * n_qubits if dual_basis else n_qubits

    def build(self):
        """Return (qnode, weight_shapes, n_meas) for TorchLayer construction."""
        return build_qaf_qnode(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            diff_method=self.diff_method,
            qdevice=self.qdevice,
            dual_basis=self.dual_basis,
        )

    def __repr__(self) -> str:
        return (
            f"HierarchicalQAFCircuit(n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"dual_basis={self.dual_basis}, n_meas={self.n_meas}, "
            f"n_params={self.n_params})"
        )
