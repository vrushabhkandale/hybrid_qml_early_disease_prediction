"""
model.py — Classical CNN Backbone, PennyLane VQC TorchLayer & Parameter-Matched Baseline
=========================================================================================
Architecture (Overfitting & Expressivity Hardened):
  BiomedCLIP ViT-B/16 (Frozen)
    → Residual CQ Adapter (Optional LoRA-Q rank-4 factorization): 512→128→32→3*n_qubits
    → Symmetric Tanh Latent Scaling: θ, φ, λ = π · tanh(z) ∈ [-π, π] centered at 0
    → Interleaved Data Re-Uploading VQC with Adjoint Differentiation (lightning.qubit)
    → Quantum Angle Dropout (Q-Dropout, p=0.10) for co-adaptation suppression
    → Local Observable Feature Vector: [<Z_i>, <Z_i Z_{i+1}>, <X_i>] (3n - 1 polynomial features)
    → Nonlinear Head: Linear(3n-1, 32) → GELU → Dropout → Linear(32, n_classes)

Exports:
  • HybridCNNVQC          — full hybrid quantum model
  • ClassicalBaseline      — parameter-matched classical control
  • LoRAQLinear           — low-rank adapter projection
  • QuantumDropout        — stochastic angle deactivation
  • QuantumDropoutTorchLayer — PennyLane TorchLayer with Q-Dropout
  • compute_n_meas        — observable count calculator
  • compute_local_observables — local Hamiltonian measurement suite
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from functools import partial
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module configurations & offline execution helper (Fix: M-01)
# ---------------------------------------------------------------------------
def configure_offline_mode(hf_offline: bool = True, fix_stdout_encoding: bool = False) -> None:
    """Call from application entry-point (train.py), NOT from module import time."""
    if hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if fix_stdout_encoding:
        import sys
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class ImagePreprocessMode(Enum):
    """Explicit preprocessing contract for BiomedCLIP input normalization (Fix: H-04).

    Exactly ONE pipeline stage owns normalization (Fix: Problems-#1):

    RAW_UINT8       : tensor values are integers in [0, 255];
                      model divides by 255 then applies BiomedCLIP mean/std.
    RAW_FLOAT01     : tensor values are floats in [0.0, 1.0];
                      model applies BiomedCLIP mean/std.
    BIOMEDCLIP_NORM : tensor is ALREADY normalized with BiomedCLIP mean/std;
                      model applies NO further normalization.
                      This is the DEFAULT because the workspace data loaders
                      (qml_models/dataset.py) emit already-normalized tensors.
    """
    RAW_UINT8       = "raw_uint8"
    RAW_FLOAT01     = "raw_float01"
    BIOMEDCLIP_NORM = "biomedclip_normalized"


# BiomedCLIP open-ai CLIP normalization constants (must match qml_models/dataset.py)
BIOMEDCLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
BIOMEDCLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _validate_image_batch(x: torch.Tensor, mode: Optional[ImagePreprocessMode] = None) -> None:
    """Validate an image tensor against its declared preprocessing contract (Fix: Problems-#1).

    Raises on non-finite values, wrong rank/dtype, and out-of-contract ranges.
    """
    if x.ndim != 4:
        raise ValueError(
            f"Problems-#1: expected rank-4 image batch (B, C, H, W), got shape {tuple(x.shape)}"
        )
    if not x.is_floating_point():
        raise TypeError(
            f"Problems-#1: expected floating-point image tensor, got dtype={x.dtype}. "
            "Cast uint8 tensors explicitly before calling extract_features()."
        )
    if not torch.isfinite(x).all():
        raise ValueError("Problems-#1: image batch contains non-finite (NaN/Inf) values.")
    m = mode or ImagePreprocessMode.BIOMEDCLIP_NORM
    if m == ImagePreprocessMode.RAW_UINT8:
        if x.min().item() < 0.0 or x.max().item() > 255.0:
            raise ValueError(
                f"Problems-#1: RAW_UINT8 contract violated — values in "
                f"[{x.min().item():.3f}, {x.max().item():.3f}], expected [0, 255]."
            )
    elif m == ImagePreprocessMode.RAW_FLOAT01:
        if x.min().item() < 0.0 or x.max().item() > 1.0:
            raise ValueError(
                f"Problems-#1: RAW_FLOAT01 contract violated — values in "
                f"[{x.min().item():.3f}, {x.max().item():.3f}], expected [0, 1]."
            )
    else:  # BIOMEDCLIP_NORM
        # Channels should be roughly mean-0 / unit-var. A clearly unnormalized
        # tensor (e.g. [0,1] floats or double-normalized inputs) is a pipeline bug.
        if x.abs().max().item() <= 1.0001:
            import warnings
            warnings.warn(
                "Problems-#1: BIOMEDCLIP_NORM contract expected but input range looks "
                "unnormalized (|x| <= 1). Double-check the owner of normalization "
                "in the data pipeline.",
                stacklevel=3,
            )
        if x.abs().max().item() > 100.0:
            raise ValueError(
                f"Problems-#1: BIOMEDCLIP_NORM contract violated — |x|max="
                f"{x.abs().max().item():.2f} is far outside normalized range "
                "(double normalization suspected)."
            )


# ---------------------------------------------------------------------------
# Checkpoint schema & feature cache helpers (Fix: H-06, L-07)
# ---------------------------------------------------------------------------
CHECKPOINT_SCHEMA_VERSION = "2.0"

# Keys of `model` that define the predictive function; all must be identical on load.
_PREDICTION_DEFINING_KEYS = (
    "n_qubits", "n_layers", "n_classes", "diff_method",
    "reupload", "p_drop", "dual_basis_only", "use_qaf_attention",
    "clinical_hard", "q_init_std",
)


def _encoder_weight_fingerprint() -> str:
    """Stable short fingerprint of the BiomedCLIP encoder loading config (Fix: Problems-#5).

    Combines the encoder identity and the fine-tuning applicability so cached
    features / checkpoints cannot silently transfer between encoder versions.
    """
    h = hashlib.sha256()
    h.update(b"BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    h.update(str(_FINETUNING_AVAILABLE).encode())
    return h.hexdigest()[:16]


def save_feature_cache(
    features: torch.Tensor,
    path: Union[str, Path],
    encoder_id: str = "BiomedCLIP-v1",
    preprocess_mode: Optional[ImagePreprocessMode] = None,
    sample_ids: Optional[List[str]] = None,
) -> None:
    """Save pre-extracted features with MANDATORY provenance metadata (Fix: H-06, Problems-#8).

    The sidecar binds the cache to encoder identity, encoder weight fingerprint,
    preprocessing contract, sample ordering, shape and dtype. Loaders refuse
    caches without this metadata.
    """
    features = torch.as_tensor(features)
    p = Path(path)
    torch.save(features, str(p))
    meta_path = p.with_suffix(p.suffix + ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "encoder_id": encoder_id,
            "encoder_fingerprint": _encoder_weight_fingerprint(),
            "preprocess_mode": str(preprocess_mode or ImagePreprocessMode.BIOMEDCLIP_NORM),
            "feature_dim": int(features.shape[-1]),
            "n_samples": int(features.shape[0]),
            "dtype": str(features.dtype),
            "sample_ids_sha256": (
                hashlib.sha256(json.dumps(sample_ids).encode()).hexdigest()[:16]
                if sample_ids is not None else None
            ),
            "finite": bool(torch.isfinite(features).all().item()),
            "schema_version": "2.0",
        }, f, indent=2)


def load_feature_cache(
    path: Union[str, Path],
    expected_encoder_id: str = "BiomedCLIP-v1",
    expected_feature_dim: Optional[int] = None,
    expected_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Load pre-extracted features validating MANDATORY metadata & finiteness (Fix: Problems-#8).

    Rejects metadata-free legacy caches, encoder/fingerprint mismatches, dim/dtype
    mismatches, and non-finite features.
    """
    p = Path(path)
    meta_path = p.with_suffix(p.suffix + ".meta.json")
    if not meta_path.exists():
        raise ValueError(
            f"Problems-#8: feature cache '{p.name}' has no provenance metadata sidecar "
            f"({meta_path.name}). Caches without provenance cannot be verified as "
            "compatible with the current encoder/preprocessing — re-extract features."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    problems = []
    if meta.get("encoder_id") != expected_encoder_id:
        problems.append(
            f"Cache encoder '{meta.get('encoder_id')}' != expected '{expected_encoder_id}'."
        )
    if meta.get("schema_version") != "2.0":
        problems.append(f"Cache schema '{meta.get("schema_version")}' is not '2.0'.")
    if meta.get("encoder_fingerprint") and meta["encoder_fingerprint"] != _encoder_weight_fingerprint():
        problems.append(
            f"Encoder weight fingerprint mismatch: cache={meta['encoder_fingerprint']}, "
            f"current={_encoder_weight_fingerprint()}."
        )
    if meta.get("finite") is not True:
        problems.append("Cache metadata records non-finite features.")
    features = torch.load(str(p), map_location="cpu", weights_only=True)
    if not torch.isfinite(features).all():
        problems.append("Feature tensor contains NaN/Inf values.")
    if expected_feature_dim is not None and features.shape[-1] != expected_feature_dim:
        problems.append(
            f"Feature dim {features.shape[-1]} != expected {expected_feature_dim}."
        )
    if expected_dtype is not None and features.dtype != expected_dtype:
        problems.append(f"Feature dtype {features.dtype} != expected {expected_dtype}.")
    if "n_samples" in meta and features.shape[0] != meta["n_samples"]:
        problems.append(
            f"Sample count mismatch: tensor={features.shape[0]}, meta={meta['n_samples']}."
        )
    if problems:
        raise ValueError(
            "Problems-#8: feature cache rejected — " + " | ".join(problems)
        )
    return features


def save_model_checkpoint(
    model: "HybridCNNVQC",
    path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model checkpoint with the FULL prediction-defining config (Fix: Problems-#5, L-07)."""
    bb = getattr(model, "backbone", None)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "architecture": {
            "n_qubits": model.n_qubits,
            "n_layers": model.n_layers,
            "n_classes": model.n_classes,
            "diff_method": model.diff_method,
            "reupload": model.reupload,
            "p_drop": model.p_drop,
            "q_init_std": model.q_init_std,
            "dual_basis_only": model.dual_basis_only,
            "use_qaf_attention": model.use_qaf_attention,
            "clinical_hard": model.clinical_hard,
            # Structured genome identity (gate order matters for the same weights)
            "genome_repr": str(model.genome) if model.genome else None,
            "genome_n_params": (
                int(getattr(model.genome, "total_params", -1)) if model.genome else None
            ),
        },
        "encoder_id": "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "encoder_fingerprint": _encoder_weight_fingerprint(),
        "feature_dim": 512,
        "pennylane_dev": getattr(model, "qdev_name", "unknown"),
        # Preprocessing contract — part of the predictive function (Fix: Problems-#1/#5)
        "preprocess_mode": str(getattr(bb, "preprocess_mode", ImagePreprocessMode.BIOMEDCLIP_NORM)),
        # Class ordering contract
        "class_order": list(range(model.n_classes)),
    }
    if extra_metadata:
        payload.update(extra_metadata)
    torch.save({
        "metadata": payload,
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
    }, str(path))
    logger.info("L-07: Saved versioned checkpoint to %s (schema v%s)", path, CHECKPOINT_SCHEMA_VERSION)


def load_model_checkpoint(
    path: Union[str, Path],
    model: "HybridCNNVQC",
    strict: bool = True,
    allow_config_mismatch: bool = False,
) -> Dict[str, Any]:
    """Load model checkpoint enforcing FULL prediction-equivalent config (Fix: Problems-#5).

    Every prediction-defining field must match: architecture, reupload/clinical_hard,
    p_drop, q_init_std, preprocessing contract, genome structure, encoder identity
    fingerprint, and class ordering. A mismatch raises ValueError unless
    ``allow_config_mismatch=True`` is explicitly passed (and then only with a warning).
    """
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    meta = ckpt.get("metadata", {})
    if meta.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"L-07/Problems-#5: Checkpoint schema '{meta.get('schema_version')}' != current "
            f"'{CHECKPOINT_SCHEMA_VERSION}'. Silently loading weights across schema "
            "versions changes the predictive function."
        )
    arch = meta.get("architecture", {})
    mismatches = [
        f"{k}: ckpt={arch.get(k)}, model={getattr(model, k, None)}"
        for k in _PREDICTION_DEFINING_KEYS
        if arch.get(k) is not None and arch.get(k) != getattr(model, k, None)
    ]

    # Missing mandatory metadata must not pass compatibility checks (Fix: Problems-#5)
    missing = [
        k for k in ("schema_version", "architecture", "preprocess_mode", "encoder_id")
        if k not in meta or meta.get(k) is None
    ]
    if missing:
        raise ValueError(f"Problems-#5: checkpoint metadata missing mandatory keys: {missing}")

    # Genome structural identity
    ckpt_genome = arch.get("genome_repr")
    model_genome = str(model.genome) if model.genome else None
    if ckpt_genome != model_genome:
        mismatches.append(f"genome: ckpt={ckpt_genome}, model={model_genome}")

    # Preprocessing contract identity
    ckpt_pm = meta.get("preprocess_mode")
    model_pm = str(getattr(getattr(model, "backbone", None), "preprocess_mode", ImagePreprocessMode.BIOMEDCLIP_NORM))
    if ckpt_pm != model_pm:
        mismatches.append(f"preprocess_mode: ckpt={ckpt_pm}, model={model_pm}")

    # Encoder identity + weight-config fingerprint
    if meta.get("encoder_fingerprint") and meta["encoder_fingerprint"] != _encoder_weight_fingerprint():
        mismatches.append(
            f"encoder_fingerprint: ckpt={meta['encoder_fingerprint']}, "
            f"model={_encoder_weight_fingerprint()}"
        )

    # Class ordering
    if "class_order" in meta and meta["class_order"] != list(range(model.n_classes)):
        mismatches.append(f"class_order: ckpt={meta['class_order']}, model=identity")

    # The projection head itself must carry the same clinical_hard mode as the
    # saved function (a silent projection-mode change alters the encoding).
    bb_proj_hard = getattr(getattr(getattr(model, "backbone", None), "riemannian_head", None), "clinical_hard", None)
    if arch.get("clinical_hard") is not None and bb_proj_hard is not None and bb_proj_hard != arch["clinical_hard"]:
        mismatches.append(f"riemannian_head.clinical_hard: ckpt={arch['clinical_hard']}, model={bb_proj_hard}")

    if mismatches:
        msg = f"Problems-#5: checkpoint/config mismatch — {'; '.join(mismatches)}"
        if not allow_config_mismatch:
            raise ValueError(msg)
        logger.warning("%s (explicitly allowed by caller)", msg)

    model.load_state_dict(ckpt["state_dict"], strict=strict)
    return ckpt


# ---------------------------------------------------------------------------
# Backend configuration dataclass (Fix: H-07)
# ---------------------------------------------------------------------------
@dataclass
class BackendConfig:
    """Unified backend configuration — created once, passed through the model (Fix: H-07)."""
    torch_device: torch.device
    pennylane_dev: qml.Device
    pennylane_name: str
    diff_method: str
    transfers_to_cpu: bool


try:
    from adapter import (
        LoRAQLinear,
        MultiHeadOrthogonalCQAdapter,
        load_adapter_checkpoint,
        save_adapter_checkpoint,
    )
except ImportError:
    from qml_models.adapter import (
        LoRAQLinear,
        MultiHeadOrthogonalCQAdapter,
        load_adapter_checkpoint,
        save_adapter_checkpoint,
    )

_FINETUNING_AVAILABLE = False
has_finetuned_checkpoint = None
load_finetuned_backbone = None
apply_finetuned_backbone = None
get_finetuned_checkpoint_path = None
load_finetuned_biomedclip_vit = None
save_finetuned_biomedclip_vit = None

for _mod in ("finetunning", "qml_models.finetunning", "finetuning", "qml_models.finetuning"):
    try:
        import importlib
        _ft = importlib.import_module(_mod)
        has_finetuned_checkpoint = getattr(_ft, "has_finetuned_checkpoint", None)
        load_finetuned_backbone = getattr(_ft, "load_finetuned_backbone", None)
        apply_finetuned_backbone = getattr(_ft, "apply_finetuned_backbone", None)
        get_finetuned_checkpoint_path = getattr(_ft, "get_finetuned_checkpoint_path", None)
        load_finetuned_biomedclip_vit = getattr(_ft, "load_finetuned_biomedclip_vit", None)
        save_finetuned_biomedclip_vit = getattr(_ft, "save_finetuned_biomedclip_vit", None)
        _FINETUNING_AVAILABLE = all([
            has_finetuned_checkpoint is not None,
            load_finetuned_backbone is not None,
            apply_finetuned_backbone is not None,
        ])
        if _FINETUNING_AVAILABLE:
            break
    except ImportError:
        continue

def _require_finetuning(caller: str) -> None:
    """Enforce presence of finetuning module when fine-tuning features are requested (Fix: H-15)."""
    if not _FINETUNING_AVAILABLE:
        raise RuntimeError(
            f"H-15: {caller} requires the finetuning module which could not be imported. "
            "Ensure finetuning.py is on the Python path."
        )

# ---------------------------------------------------------------------------
# Utility: device selection
# ---------------------------------------------------------------------------
def get_torch_device() -> torch.device:
    """Return the best available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_pennylane_device(
    n_qubits: int,
    diff_method: str = "adjoint",
    qdevice: Optional[str] = None,
) -> Tuple[qml.Device, str]:
    """Return high-performance PennyLane device configured for GPU or multi-threaded CPU.

    Supports explicit device selection ('default.qubit', 'lightning.qubit', 'lightning.gpu')
    or automatic resolution ('auto' / None). (Fix: H-08, M-04, L-01).
    """
    _VALID_EXPLICIT = {
        "default", "default.qubit",
        "lightning", "lightning.qubit",
        "gpu", "lightning.gpu",
    }
    _AUTO = {"auto", "none", ""}
    dev_key = (qdevice or "auto").strip().lower()

    if dev_key in _AUTO:
        dev_name = "auto"
    elif dev_key in _VALID_EXPLICIT:
        dev_name = {
            "default": "default.qubit", "default.qubit": "default.qubit",
            "lightning": "lightning.qubit", "lightning.qubit": "lightning.qubit",
            "gpu": "lightning.gpu", "lightning.gpu": "lightning.gpu",
        }[dev_key]
    else:
        raise ValueError(
            f"H-08: Unknown quantum device '{qdevice}'. "
            f"Valid options: {sorted(_VALID_EXPLICIT | _AUTO)}. "
            "Use 'auto' for automatic selection."
        )

    # Explicit device selections
    if dev_name == "lightning.gpu":
        try:
            dev = qml.device("lightning.gpu", wires=n_qubits)
            return dev, "lightning.gpu (cuquantum-gpu)"
        except (ImportError, RuntimeError) as e:
            if dev_key != "auto":
                raise RuntimeError(
                    f"Requested quantum device 'lightning.gpu' failed to initialize: {e}. "
                    f"Ensure PennyLane-Lightning-GPU and cuQuantum are installed, "
                    f"or select 'lightning.qubit' or 'default.qubit'."
                ) from e
    elif dev_name == "lightning.qubit":
        try:
            dev = qml.device("lightning.qubit", wires=n_qubits)
            tag = "adjoint-c++" if diff_method == "adjoint" else "cpu"
            return dev, f"lightning.qubit ({tag})"
        except (ImportError, RuntimeError) as e:
            if dev_key != "auto":
                raise RuntimeError(
                    f"Requested quantum device 'lightning.qubit' failed to initialize: {e}. "
                    f"Ensure PennyLane-Lightning is installed, or select 'default.qubit'."
                ) from e
    elif dev_name == "default.qubit":
        dev = qml.device("default.qubit", wires=n_qubits)
        tag = "backprop" if diff_method in ("adjoint", "backprop") else "cpu"
        return dev, f"default.qubit ({tag})"

    # Automatic fallback / auto-resolution
    if diff_method == "backprop":
        dev = qml.device("default.qubit", wires=n_qubits)
        return dev, "default.qubit (backprop-vectorized)"

    if dev_name == "auto":
        # 1. GPU cuQuantum acceleration if available
        if torch.cuda.is_available():
            try:
                dev = qml.device("lightning.gpu", wires=n_qubits)
                return dev, "lightning.gpu (cuquantum-gpu)"
            except (ImportError, RuntimeError) as _err:
                logger.debug("M-04: Auto-selection lightning.gpu unavailable (%s), trying lightning.qubit.", _err)
        # 2. Prefer lightning.qubit with adjoint differentiation
        try:
            dev = qml.device("lightning.qubit", wires=n_qubits)
            return dev, "lightning.qubit (adjoint-c++)"
        except (ImportError, RuntimeError) as _err:
            logger.warning("M-04: lightning.qubit unavailable (%s); falling back to default.qubit.", _err)
            dev = qml.device("default.qubit", wires=n_qubits)
            return dev, "default.qubit (backprop-vectorized)"

    # Adjoint fallback
    if diff_method in ("adjoint", "auto"):
        try:
            dev = qml.device("lightning.qubit", wires=n_qubits)
            return dev, "lightning.qubit (adjoint-c++)"
        except (ImportError, RuntimeError) as _err:
            logger.warning("M-04: lightning.qubit unavailable (%s); falling back to default.qubit.", _err)
            dev = qml.device("default.qubit", wires=n_qubits)
            return dev, "default.qubit (backprop-fallback)"
    else:
        try:
            dev = qml.device("lightning.qubit", wires=n_qubits)
            return dev, "lightning.qubit"
        except (ImportError, RuntimeError):
            dev = qml.device("default.qubit", wires=n_qubits)
            return dev, "default.qubit"


def make_backend_config(
    torch_device: torch.device,
    qdevice: Optional[str],
    diff_method: str,
    n_qubits: int,
) -> BackendConfig:
    """Build unified backend configuration with device migration detection (Fix: H-07)."""
    dev, dev_name = get_pennylane_device(n_qubits, diff_method, qdevice=qdevice)
    is_torch_cuda = torch_device.type == "cuda"
    is_pl_gpu = "gpu" in dev_name.lower() or "cuda" in dev_name.lower()
    transfers_to_cpu = is_torch_cuda and not is_pl_gpu
    if transfers_to_cpu:
        logger.warning(
            "H-07: PyTorch device is CUDA but PennyLane device '%s' runs on CPU. "
            "Every quantum pass incurs CPU<->GPU transfers. "
            "Install pennylane-lightning-gpu to eliminate this.", dev_name,
        )
    if "default.qubit" in dev_name and diff_method == "adjoint":
        diff_method = "backprop"
    return BackendConfig(torch_device, dev, dev_name, diff_method, transfers_to_cpu)


# ---------------------------------------------------------------------------
# Note: LoRAQLinear & MultiHeadOrthogonalCQAdapter are imported from adapter.py
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Riemannian S^2 Bloch Sphere Projection (Change 1: Full Surface Utilization)
# ---------------------------------------------------------------------------
class RiemannianBlochProjection(nn.Module):
    r"""Projects unconstrained 3-D Cartesian logits per qubit onto the unit 2-sphere S^2.

    Eliminates tanh saturation plateaus, vanishing gradients, and coordinate fold singularities.
    Converts Cartesian unit vectors r = (rx, ry, rz) \u2208 S^2 into canonical spherical angles:
        θ = atan2(sqrt(rx^2+ry^2), rz) ∈ [0, π]   (polar colatitude, smooth everywhere, Fix: H-02)
        φ = atan2(ry, rx)              ∈ [-π, π]  (azimuthal equatorial longitude)
        λ ≡ 0                          (Fix: H-01, Problems-#3)

    Fix (Problems-#3 — discontinuous λ): the previous third angle
    λ = atan2(rz, ry) is a singular coordinate chart (it diverges near the ±X
    directions where both ry, rz → 0, so two almost-identical inputs can encode
    to nearly orthogonal states). The projection is a PURE-STATE preparation:
    a Bloch direction has exactly two degrees of freedom, so the U3 encoding
    U3(θ, φ, 0) = RZ(φ+0) · RY(θ) prepares |R(θ, φ) |0⟩ with no third
    independent coordinate. λ is therefore FIXED to 0 (the latent keeps the same
    (B, 3*n_qubits) layout for checkpoint/algorithm compatibility, with the
    third slot constant). State continuity is guaranteed: θ and φ are smooth on
    S^2 except at the negative-pole azimuth chart, which is handled by the
    identical-limit behavior of atan2 at the branch cut (the prepared STATES at
    ±π differ only by a global phase, so any physical observable is identical).

    When clinical_hard=True (Fix: Problems-#4 — periodicity-preserving):
        θ' = 0.90·θ + 0.05π ∈ [0.05π, 0.95π]        (not periodic: safe to scale)
        φ' = φ - κ·sin(φ), κ = 0.2                       (smooth, odd, periodic —
                                                          ±π remain equivalent; no
                                                          φ·0.95 branch-cut discontinuity)
    """
    def __init__(self, eps: float = 1e-6, clinical_hard: bool = False, azimuth_compression: float = 0.2):
        super().__init__()
        self.eps = eps
        self.clinical_hard = clinical_hard
        # Problems-#4: periodicity-preserving azimuth compression strength κ ∈ [0, 1)
        if not (0.0 <= azimuth_compression < 1.0):
            raise ValueError(
                f"Problems-#4: azimuth_compression κ must be in [0, 1), got {azimuth_compression}"
            )
        self.azimuth_compression = azimuth_compression

    def forward(self, raw: torch.Tensor, n_qubits: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # raw: (B, 3 * n_qubits) -> validate shape and reshape (Fix: M-08)
        if raw.ndim != 2:
            raise ValueError(f"M-08: Expected rank-2 tensor (B, 3*n_qubits), got shape {raw.shape}")
        if raw.shape[-1] != 3 * n_qubits:
            raise ValueError(
                f"M-08: Last dim must be {3 * n_qubits}, got {raw.shape[-1]}. "
                "Check adapter output dimension."
            )
        v = raw.reshape(-1, n_qubits, 3)

        # Safe normalization with north-pole fallback for zero/near-zero vectors (Fix: H-03)
        norms = v.norm(p=2, dim=-1, keepdim=True)
        north_pole = torch.zeros_like(v)
        north_pole[..., 2] = 1.0  # |0> state safe canonical fallback
        r = torch.where(norms > self.eps, v / norms.clamp_min(self.eps), north_pole)  # (B, n_qubits, 3) on S^2

        rx = r[..., 0]
        ry = r[..., 1]
        rz = r[..., 2].clamp(-1.0 + self.eps, 1.0 - self.eps)

        # Smooth polar angle avoiding acos boundary singularity (Fix: H-02)
        rxy_norm = torch.sqrt(rx**2 + ry**2 + self.eps**2)
        theta = torch.atan2(rxy_norm, rz)                # [0, pi], smooth gradient everywhere
        phi   = torch.atan2(ry, rx)                      # [-pi, pi]

        # Fix: Problems-#3 — the third U3 angle is FIXED to 0 (pure-state
        # preparation has only two degrees of freedom; deriving it from a
        # singular atan2(rz, ry) chart created encoding discontinuities).
        lam = torch.zeros_like(theta)

        if self.clinical_hard:
            # Smooth polar expansion allowing both Northern and Southern hemispheres (Task 3.5, C-03)
            theta = theta * 0.90 + 0.05 * math.pi        # [0.05*pi, 0.95*pi] differentiable polar range
            # Fix: Problems-#4 — periodicity-preserving azimuth compression.
            # Old code scaled phi and lam by 0.95, mapping ±pi to ±0.95pi and
            # breaking the ±pi equivalence (branch-cut discontinuity ~0.156).
            # phi - kappa*sin(phi) is smooth, odd, 2pi-periodic, monotone on
            # [-pi, pi] (for kappa < 1), and maps ±pi → ±pi exactly.
            kappa = self.azimuth_compression
            phi = phi - kappa * torch.sin(phi)           # periodic-shrunk toward equator
            # lam is identically 0 — compression is a no-op for it.

        z = torch.cat([theta, phi, lam], dim=-1)         # (B, 3 * n_qubits)
        return z, r


# ---------------------------------------------------------------------------
# BiomedCLIP Foundation Model + CQ Adapter (Riemannian S^2 Latent Projection)
# ---------------------------------------------------------------------------
class BiomedCLIPQuantumBackbone(nn.Module):
    """
    BiomedCLIP vision encoder → Trainable 3-Head Orthogonal CQ Adapter (Fix: H-04, H-05, M-02, H-13).

    Features:
      - 3-Head Orthogonal CQ Adapter: separates projection into X, Y, Z channels with
        Frobenius difference loss, breaking rank-1 collinearity on S^2.
      - Optional LoRA-Q low-rank factorization for data-scarce domains.
      - Riemannian S^2 Bloch Sphere Projection: maps Cartesian logits to unit sphere S^2.
      - Explicit image preprocessing contract with ImagePreprocessMode.
      - Lazy vision encoder weight loading and offloading.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_qubits: int = 4,
        dropout: float = 0.3,
        use_lora: bool = False,
        lora_rank: int = 4,
        clinical_hard: bool = False,
        backbone_unfreeze_layers: int = 0,
        dataset_name: Optional[str] = None,
        enable_finetuning: bool = False,
        preprocess_mode: Optional[ImagePreprocessMode] = None,
        feature_only_mode: bool = False,
    ):
        super().__init__()
        self._in_channels = in_channels  # Stored explicitly (Fix: H-05)
        self.n_qubits = n_qubits
        self.use_lora = use_lora
        self.clinical_hard = clinical_hard
        self.backbone_unfreeze_layers = backbone_unfreeze_layers
        self.dataset_name = dataset_name
        self.enable_finetuning = enable_finetuning
        # Fix: Problems-#1 — default to BIOMEDCLIP_NORM because the workspace data
        # loaders (qml_models/dataset.py) already apply BiomedCLIP normalization;
        # the old RAW_FLOAT01 default double-normalized every input.
        self.preprocess_mode = preprocess_mode or ImagePreprocessMode.BIOMEDCLIP_NORM  # (Fix: H-04)
        self.feature_only_mode = feature_only_mode  # (Fix: M-02)

        # Lazy loading of vision encoder (Fix: M-02, L-04)
        self.vision_encoder = None
        if not feature_only_mode:
            self._load_vision_encoder_weights()

        LinearLayer = partial(LoRAQLinear, rank=lora_rank) if use_lora else nn.Linear

        # 3-Head Orthogonal CQ Adapter — guarantees rank-3 Cartesian pre-activations
        self.cq_adapter = MultiHeadOrthogonalCQAdapter(
            in_features=512,
            n_qubits=n_qubits,
            dropout=dropout,
            linear_cls=LinearLayer,
        )

        # Residual projection shortcut (512 → 3*n_qubits) with small nonzero weight init (Fix: H-03)
        self.skip = LinearLayer(512, 3 * n_qubits)
        if isinstance(self.skip, nn.Linear):
            nn.init.normal_(self.skip.weight, mean=0.0, std=1e-3)
            if self.skip.bias is not None:
                nn.init.zeros_(self.skip.bias)

        # Riemannian S^2 projection head
        self.riemannian_head = RiemannianBlochProjection(clinical_hard=clinical_hard)
        self._latent_z: Optional[torch.Tensor] = None
        self._r_vectors: Optional[torch.Tensor] = None
        self._current_r: Optional[torch.Tensor] = None

    def _load_vision_encoder_weights(self) -> None:
        """Internal helper to load BiomedCLIP weights on demand (Fix: M-02, L-04)."""
        if self.vision_encoder is not None:
            return
        # pyrefly: ignore [missing-import]
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.vision_encoder = clip_model.visual

        # Freeze encoder by default
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        # Load dataset-specific fine-tuned weights if available (Fix: H-15)
        if self.dataset_name and _FINETUNING_AVAILABLE and has_finetuned_checkpoint:
            if has_finetuned_checkpoint(self.dataset_name, self.backbone_unfreeze_layers):
                load_finetuned_backbone(self.vision_encoder, self.dataset_name, self.backbone_unfreeze_layers)

        # Unfreeze top N layers if requested
        if self.backbone_unfreeze_layers > 0:
            resblocks = None
            if hasattr(self.vision_encoder, "transformer") and hasattr(self.vision_encoder.transformer, "resblocks"):
                resblocks = self.vision_encoder.transformer.resblocks
            elif hasattr(self.vision_encoder, "trunk") and hasattr(self.vision_encoder.trunk, "blocks"):
                resblocks = self.vision_encoder.trunk.blocks
            elif hasattr(self.vision_encoder, "blocks"):
                resblocks = self.vision_encoder.blocks

            if resblocks is not None:
                tot = len(resblocks)
                unfreeze_count = min(self.backbone_unfreeze_layers, tot)
                for i in range(tot - unfreeze_count, tot):
                    for p in resblocks[i].parameters():
                        p.requires_grad = True

    def extract_features(self, x: torch.Tensor, needs_input_grad: bool = False) -> torch.Tensor:
        """Offline pass helper: returns 512-d feature vectors with explicit normalization (Fix: H-04, H-05, H-14, M-15).

        needs_input_grad=True (Fix: Problems-#10): keeps gradients flowing to the
        INPUT image — the encoder's weights stay frozen via requires_grad=False,
        so freezing parameters does not require blocking input gradients. Enable
        for input-gradient attribution / saliency; the default False preserves
        the memory-efficient inference path.
        """
        if self.vision_encoder is None:
            raise RuntimeError(
                "H-14: Vision encoder has been offloaded or initialized in feature_only_mode. "
                "Call load_vision_encoder() before passing raw images, "
                "or pass pre-extracted features with use_cached_features=True."
            )

        target_device = next(self.vision_encoder.parameters()).device
        x = x.to(target_device).float()
        if x.ndim == 3:
            x = x.unsqueeze(0)

        # Strict channel validation (Fix: H-05, Problems-#1 validation)
        declared_ch = getattr(self, "_in_channels", None)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 2:
            raise ValueError(
                "H-05: 2-channel input is not supported. Supply 1-channel or 3-channel input."
            )
        elif x.shape[1] > 3:
            raise ValueError(
                f"H-05: Input has {x.shape[1]} channels; only 1 or 3 are supported. "
                "Truncating medical imaging channels discards diagnostic signal."
            )
        if declared_ch is not None and declared_ch in (1, 3) and x.shape[1] != 3 and x.shape[1] != declared_ch:
            raise ValueError(
                f"H-05: Model configured for {declared_ch} channels but input has {x.shape[1]}."
            )

        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # Deterministic preprocessing mode (Fix: H-04; Problems-#1 single owner)
        mode = getattr(self, "preprocess_mode", ImagePreprocessMode.BIOMEDCLIP_NORM)
        _validate_image_batch(x, mode)
        if mode != ImagePreprocessMode.BIOMEDCLIP_NORM:
            if mode == ImagePreprocessMode.RAW_UINT8:
                x = x / 255.0
            mean = torch.tensor(BIOMEDCLIP_MEAN, device=x.device, dtype=x.dtype
            ).view(1, 3, 1, 1)
            std = torch.tensor(BIOMEDCLIP_STD, device=x.device, dtype=x.dtype
            ).view(1, 3, 1, 1)
            x = (x - mean) / std

        # Train/eval mode alignment (Fix: M-15)
        is_fine_tuning = self.training and (self.backbone_unfreeze_layers > 0)
        if is_fine_tuning:
            self.vision_encoder.train()
        else:
            self.vision_encoder.eval()

        # Problems-#10: attribution path — do not disable input gradients when
        # the caller requests them (encoder weights remain frozen independently).
        grad_context = (
            torch.enable_grad
            if (is_fine_tuning or needs_input_grad)
            else torch.no_grad
        )
        with grad_context():
            if x.is_cuda and torch.cuda.is_bf16_supported():
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    features = self.vision_encoder(x)
            elif x.is_cuda:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    features = self.vision_encoder(x)
            else:
                features = self.vision_encoder(x)
        return features.float()

    def forward(
        self,
        x: torch.Tensor,
        use_cached_features: bool = False,
        return_regularizer: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Return canonical latent z for U3 angles via Riemannian S^2 projection (Fix: H-06, H-13)."""
        if use_cached_features:
            # Strict cached feature validation (Fix: H-06)
            if x.ndim != 2:
                raise ValueError(
                    f"H-06: Cached features must be rank-2 (B, 512), got shape {x.shape}"
                )
            if x.shape[-1] != 512:
                raise ValueError(
                    f"H-06: Cached features must have last dim=512, got {x.shape[-1]}. "
                    "Re-extract with the same BiomedCLIP model version."
                )
            if not x.is_floating_point():
                raise TypeError(
                    f"H-06: Cached features must be floating-point, got dtype={x.dtype}"
                )
            features = x.float()
        else:
            features = self.extract_features(x)

        # Raw Cartesian logits from adapter + residual skip connection
        raw = self.cq_adapter(features) + self.skip(features)  # (B, 3*n_qubits)

        # Riemannian S^2 projection
        z, r = self.riemannian_head(raw, self.n_qubits)

        # Explicit graph detachment — prevent memory leaks (Fix: H-13)
        self._latent_z = z.detach()
        self._r_vectors = r.detach()
        self._current_r = None

        if return_regularizer:
            return z, r
        return z

    def get_r_vectors(self) -> Optional[torch.Tensor]:
        """Returns the most recent detached 3D unit Bloch vectors (B, n_qubits, 3) on S^2."""
        return self._r_vectors

    def get_current_r(self) -> None:
        """Removed to prevent accidental autograd graph retention (Fix: H-13)."""
        raise AttributeError(
            "H-13: get_current_r() removed. Call forward(..., return_regularizer=True) instead."
        )

    def offload_vision_encoder(self):
        """Release frozen ViT vision encoder weights from memory once offline features are cached."""
        if hasattr(self, "vision_encoder") and self.vision_encoder is not None:
            del self.vision_encoder
            self.vision_encoder = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_vision_encoder(self):
        """Reload frozen ViT vision encoder on demand for XAI attribution maps or live inference."""
        if getattr(self, "vision_encoder", None) is None:
            self._load_vision_encoder_weights()
            dev = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
            if self.vision_encoder is not None:
                self.vision_encoder.to(dev)

    def load_adapter(self, checkpoint_path: Union[str, os.PathLike], device: Optional[torch.device] = None) -> Dict:
        """Load pre-trained CQ-Adapter and residual skip weights from disk checkpoint."""
        return load_adapter_checkpoint(self, checkpoint_path, device=device)

    def save_adapter(self, checkpoint_path: Union[str, os.PathLike], metadata: Optional[Dict] = None) -> Path:
        """Save CQ-Adapter and residual skip weights to disk checkpoint."""
        return save_adapter_checkpoint(self, checkpoint_path, metadata=metadata)

    def load_finetuned_vit(
        self,
        dataset_name: str,
        n_unfreeze: int = 2,
        base_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> bool:
        """Load finetuned BiomedCLIP ViT weights for a specific dataset (Fix: H-15, L-03)."""
        _require_finetuning("load_finetuned_vit")
        if self.vision_encoder is None:
            self.load_vision_encoder()
        if not has_finetuned_checkpoint(dataset_name, n_unfreeze, base_dir):
            return False
        load_finetuned_backbone(
            self.vision_encoder, dataset_name, n_unfreeze, base_dir,
            device=device, freeze_after_load=True,
        )
        return True

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Refuse silent architecture mutation on legacy checkpoints (Fix: M-13)."""
        legacy_key = prefix + "cq_adapter.0.linear.weight"
        legacy_key2 = prefix + "cq_adapter.0.weight"
        if legacy_key in state_dict or legacy_key2 in state_dict:
            raise RuntimeError(
                "M-13: Detected legacy sequential cq_adapter checkpoint. "
                "Run `python -m qml_models.migrate_checkpoint --input <ckpt> --output <new>` "
                "to convert before loading."
            )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


# ---------------------------------------------------------------------------
# Quantum Dropout Module (Q-Dropout — Fix: Section 12.2)
# ---------------------------------------------------------------------------
class QuantumDropout(nn.Module):
    """
    Stochastic Angle Dropout & Phase-Symmetric Jitter (Q-Dropout — Fix: M-05;
    noise semantics — Fix: Problems-#7):

    - When use_phase_jitter=True: applies bounded zero-mean Gaussian phase jitter
      during training whose STRENGTH is controlled by p_drop —
        theta_tilde = theta + delta, delta ~ N(0, (p_drop * jitter_std)^2)
        delta is clamped to ±3·jitter_std (bounded, unlike the old unbounded noise).
      p_drop = 0 disables noise entirely; p_drop = 1 uses the full jitter_std.
      (The old semantics ignored p_drop's magnitude: 0.01, 0.1, and 1.0 produced
      IDENTICAL noise, so tuning "p_drop" wasn't tuning anything.)
    - When use_phase_jitter=False: deactivates rotation angles with probability
      p_drop without rescaling (scaling angles breaks the S^1 phase topology).
    During evaluation: leaves angles untouched.
    """
    MAX_JITTER_SIGMAS = 3.0  # clamp bound; keeps phases within ±3·jitter_std

    def __init__(self, p_drop: float = 0.10, use_phase_jitter: bool = True, jitter_std: float = 0.05):
        super().__init__()
        if not (0.0 <= p_drop <= 1.0):
            raise ValueError(f"M-05: p_drop must be in [0, 1], got {p_drop}")
        if jitter_std < 0.0:
            raise ValueError(f"M-05: jitter_std must be >= 0, got {jitter_std}")
        self.p_drop = p_drop
        self.use_phase_jitter = use_phase_jitter
        self.jitter_std = jitter_std

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p_drop <= 0.0:
            return weights
        if self.use_phase_jitter:
            # Problems-#7: noise STRENGTH scales with p_drop; bound to ±3σ_base
            effective_std = self.p_drop * self.jitter_std
            delta = torch.randn_like(weights) * effective_std
            return weights + delta.clamp(
                -self.MAX_JITTER_SIGMAS * self.jitter_std,
                 self.MAX_JITTER_SIGMAS * self.jitter_std,
            )
        else:
            # Stochastic angle deactivation without inverted-dropout rescaling
            # Scaling angles breaks the S^1 topology, causing phase-inversion.
            mask = (torch.rand_like(weights) > self.p_drop).float()
            return weights * mask


class QuantumDropoutTorchLayer(qml.qnn.TorchLayer):
    """PennyLane TorchLayer with stochastic angle deactivation (Q-Dropout — Fix: H-07, M-06).

    NOTE (M-06): q_dropout is applied once per forward call in weight-space,
    meaning all batch samples share the same perturbation for that pass.
    This acts as circuit-level regularization rather than per-example noise.
    """
    def __init__(
        self,
        qnode,
        weight_shapes,
        init_method=None,
        p_drop: float = 0.10,
        transfers_to_cpu: bool = False,
        jitter_std: float = 0.05,
    ):
        super().__init__(qnode, weight_shapes, init_method=init_method)
        self.q_dropout = QuantumDropout(p_drop=p_drop, jitter_std=jitter_std)
        self._transfers_to_cpu = transfers_to_cpu  # Configured explicitly (Fix: H-07)

    def _evaluate_qnode(self, x):
        orig_device = x.device
        orig_dtype = x.dtype
        # Device transfer coordinated by BackendConfig flag, not string heuristic (Fix: H-07)
        needs_cpu = getattr(self, "_transfers_to_cpu", False) and x.is_cuda

        x_in = x.cpu() if needs_cpu else x
        kwargs = {
            **{self.input_arg: x_in},
            **{arg: (self.q_dropout(weight) if self.training else weight).to(x_in.device)
               for arg, weight in self.qnode_weights.items()},
        }
        res = self.qnode(**kwargs)
        if isinstance(res, torch.Tensor):
            return res.to(device=orig_device, dtype=orig_dtype)
        if isinstance(res, (list, tuple)):
            if len(x.shape) > 1:
                res = [torch.reshape(r, (x.shape[0], -1)) for r in res]
            return torch.hstack(res).to(device=orig_device, dtype=orig_dtype)
        return res


# ---------------------------------------------------------------------------
# Strict Sum-of-Local Observables (Fix: Section 5)
# ---------------------------------------------------------------------------
def compute_n_meas(n_qubits: int, dual_basis_only: bool = False) -> int:
    """Compute total measurement output dimension (Fix: M-09).

    When dual_basis_only=False (default):
        Total = n (local Z) + (n-1) (nearest-neighbor ZZ) + n (transverse X) = 3n - 1.
        Guarantees polynomial gradient scaling (Cerezo et al., 2021).

    When dual_basis_only=True (HQCNN paradigm, arXiv:2509.14277v1):
        Total = n (local Z) + n (transverse X) = 2n.
        Drops ZZ correlators; Z captures pigment density, X captures phase coherence.
    """
    if dual_basis_only:
        return 2 * n_qubits
    return 3 * n_qubits - 1


def compute_local_observables(n_qubits: int, dual_basis_only: bool = False):
    """Build PennyLane observable feature vector for VQC measurement layer (Fix: M-09).

    Returns a feature vector of individual local observables:
    When dual_basis_only=False: [<Z_i>, <Z_i Z_{i+1}>, <X_i>] (3n-1 features).
    When dual_basis_only=True:  [<Z_i>, <X_i>] only (2n features, HQCNN paradigm).
    """
    local_z = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    if dual_basis_only:
        transverse_x = [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
        return local_z + transverse_x
    nearest_zz = [qml.expval(qml.PauliZ(i) @ qml.PauliZ(i + 1)) for i in range(n_qubits - 1)]
    transverse_x = [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
    return local_z + nearest_zz + transverse_x


def validate_genome(genome: Optional[object]) -> None:
    """Validate CircuitGenome integrity before QNode construction (Fix: M-11)."""
    if genome is None:
        return
    if not hasattr(genome, "layers"):
        raise ValueError("M-11: Genome has no 'layers' attribute.")
    n_q = getattr(genome, "n_qubits", None)
    for l_idx, layer in enumerate(genome.layers):
        for gene in layer:
            t = getattr(gene, "target", None)
            c = getattr(gene, "control", None)
            if t is not None and n_q is not None and not (0 <= t < n_q):
                raise ValueError(f"M-11: Layer {l_idx} target={t} out of range [0,{n_q-1}].")
            if c is not None and t is not None and c == t:
                raise ValueError(f"M-11: Control==target={c} in layer {l_idx}.")


def _apply_ansatz(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    n_qubits: int,
    n_layers: int,
    reupload: bool = True,
    genome: Optional[object] = None,
):
    """Execute dynamic QNAS genome gates or StronglyEntanglingLayers (Fix: M-11)."""
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
        if hasattr(genome, "total_params"):
            expected = max(1, genome.total_params)
            if w_idx != expected:
                raise RuntimeError(
                    f"M-11: Ansatz consumed {w_idx} params but genome.total_params={expected}."
                )
    elif reupload:
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.U3(inputs[..., i], inputs[..., i + n_qubits], inputs[..., i + 2 * n_qubits], wires=i)
            w_l = weights[l:l+1] if hasattr(weights, "ndim") and weights.ndim >= 3 else weights
            range_l = [(l % max(n_qubits - 1, 1)) + 1]
            qml.StronglyEntanglingLayers(w_l, wires=range(n_qubits), ranges=range_l)
    else:
        for i in range(n_qubits):
            qml.U3(inputs[..., i], inputs[..., i + n_qubits], inputs[..., i + 2 * n_qubits], wires=i)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))


# ---------------------------------------------------------------------------
# PennyLane Variational Quantum Circuit (QNode + TorchLayer)
# ---------------------------------------------------------------------------
def create_vqc_qnode(
    n_qubits: int = 4,
    n_layers: int = 3,
    diff_method: str = "adjoint",
    reupload: bool = True,
    genome: Optional[object] = None,
    qdevice: Optional[str] = None,
    dual_basis_only: bool = False,
    use_qaf_attention: bool = False,
):
    """Build a PennyLane QNode with Interleaved Data Re-Uploading + Sum-of-Local Observables (Fix: H-08, H-09, M-11).

    Features:
      - Interleaved data re-uploading S_l(z) at each layer for universal Fourier expressivity
      - Adjoint differentiation on lightning.qubit for O(1) single-pass backward scaling
      - Sum-of-local observables with polynomial gradient scaling
      - Dynamic QNAS genome support for minimal CNOT sparse topologies
      - dual_basis_only=True: HQCNN 2n readout (Z+X only)
      - use_qaf_attention=True: QAF circuit (Superpositional + Toffoli + Fourier)
    """
    dev, dev_name = get_pennylane_device(n_qubits, diff_method, qdevice=qdevice)
    if "default.qubit" in dev_name:
        actual_diff = "backprop" if diff_method in ("adjoint", "backprop", "auto") else diff_method
    elif "lightning" in dev_name:
        if diff_method == "backprop":
            logger.info("Device '%s' does not support 'backprop'; using 'adjoint' differentiation.", dev_name)
        actual_diff = "adjoint" if diff_method in ("backprop", "auto", "adjoint") else diff_method
    else:
        actual_diff = diff_method

    # Determine if QAF ansatz should be used
    _use_qaf = use_qaf_attention
    if _use_qaf:
        try:
            from ansatz.qaf import build_qaf_circuit
        except ImportError:
            try:
                from qml_models.ansatz.qaf import build_qaf_circuit
            except ImportError:
                logger.warning("QAF ansatz requested but qml_models/ansatz/qaf.py not found. Falling back to standard ansatz.")
                _use_qaf = False

    is_dual = dual_basis_only or _use_qaf
    n_meas = compute_n_meas(n_qubits, dual_basis_only=is_dual)

    # Convert dict to CircuitGenome if needed and validate (Fix: H-09, M-11)
    if isinstance(genome, dict):
        try:
            from qnas import CircuitGenome
        except ImportError:
            from qml_models.qnas import CircuitGenome
        try:
            genome = CircuitGenome.from_dict(genome)
        except Exception as exc:
            raise ValueError(f"H-09: Failed to parse genome dict into CircuitGenome: {exc}") from exc

    validate_genome(genome)

    @qml.qnode(dev, interface="torch", diff_method=actual_diff)
    def circuit(inputs, weights):
        if _use_qaf:
            build_qaf_circuit(inputs, weights, n_qubits=n_qubits, n_layers=n_layers)
        else:
            _apply_ansatz(inputs, weights, n_qubits, n_layers, reupload=reupload, genome=genome)

        # Local observables
        return compute_local_observables(n_qubits, dual_basis_only=is_dual)

    if _use_qaf:
        # QAF decoupled variational parameters: index 0 = CRZ, indices 1-3 = Rot
        weight_shapes = {"weights": (n_layers, n_qubits, 4)}
    elif genome is not None:
        weight_shapes = {"weights": (max(1, getattr(genome, "total_params", 1)),)}
    else:
        weight_shapes = {
            "weights": qml.StronglyEntanglingLayers.shape(
                n_layers=n_layers, n_wires=n_qubits
            )
        }

    return circuit, weight_shapes, dev, dev_name, n_meas


# ---------------------------------------------------------------------------
# Full Hybrid CNN-VQC Model
# ---------------------------------------------------------------------------
class HybridCNNVQC(nn.Module):
    """End-to-end Hybrid BiomedCLIP → Residual CQ Adapter → VQC → Nonlinear Head.

    THREAD-SAFETY WARNING (L-06):
    The activation caches (`_latent_z`, `_expectations`) are mutable module attributes
    updated on each forward pass. For concurrent serving across threads, use
    `forward_with_latent()` which returns activations locally.

    Attributes
    ----------
    backbone     : BiomedCLIPQuantumBackbone
    qlayer       : QuantumDropoutTorchLayer
    head         : nn.Sequential
    n_meas       : int
    """

    @staticmethod
    def _validate_constructor_args(
        n_qubits: int, n_layers: int, n_classes: int, dropout: float, p_drop: float, q_init_std: float
    ) -> None:
        """Validate hyperparameter domains on initialization (Fix: M-07)."""
        if n_qubits < 1:
            raise ValueError(f"M-07: n_qubits must be >= 1, got {n_qubits}")
        if n_layers < 1:
            raise ValueError(f"M-07: n_layers must be >= 1, got {n_layers}")
        if n_classes < 2:
            raise ValueError(f"M-07: n_classes must be >= 2, got {n_classes}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"M-07: dropout must be in [0, 1), got {dropout}")
        if not (0.0 <= p_drop <= 1.0):
            raise ValueError(f"M-07: p_drop must be in [0, 1], got {p_drop}")
        if q_init_std <= 0.0:
            raise ValueError(f"M-07: q_init_std must be > 0, got {q_init_std}")

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 2,
        n_qubits: int = 4,
        n_layers: int = 3,
        diff_method: str = "adjoint",
        dropout: float = 0.3,
        reupload: bool = True,
        p_drop: float = 0.10,
        use_lora: bool = False,
        genome: Optional[object] = None,
        q_init_std: float = 0.01,
        qdevice: Optional[str] = None,
        dual_basis_only: bool = False,
        use_qaf_attention: bool = False,
        clinical_hard: bool = False,
        backbone: Optional[BiomedCLIPQuantumBackbone] = None,
        preprocess_mode: Optional[ImagePreprocessMode] = None,
        feature_only_mode: bool = False,
        jitter_std: float = 0.05,
    ):
        super().__init__()
        self._validate_constructor_args(n_qubits, n_layers, n_classes, dropout, p_drop, q_init_std)

        self.n_qubits    = n_qubits
        self.n_layers    = n_layers
        self.n_classes   = n_classes
        self.diff_method = diff_method
        self.reupload    = reupload
        self.p_drop      = p_drop
        self.jitter_std  = jitter_std
        self.genome      = genome
        self.q_init_std  = q_init_std
        self.qdevice     = qdevice
        self.dual_basis_only = dual_basis_only
        self.use_qaf_attention = use_qaf_attention
        self.clinical_hard = clinical_hard
        self._freeze_hook_handle = None
        self._cached_state_qnode = None
        self._state_qnode_cache: Dict[Tuple, Any] = {}

        # Canonicalize genome dict to CircuitGenome (Fix: H-09)
        if isinstance(self.genome, dict):
            try:
                from qnas import CircuitGenome
            except ImportError:
                from qml_models.qnas import CircuitGenome
            try:
                self.genome = CircuitGenome.from_dict(self.genome)
            except Exception as exc:
                raise ValueError(f"H-09: Failed to parse genome dict into CircuitGenome: {exc}") from exc

        # Classical feature extractor + residual adapter (Fix: M-03 shared backbone)
        if backbone is not None:
            self.backbone = backbone
            self._owns_full_backbone = False
        else:
            self.backbone = BiomedCLIPQuantumBackbone(
                in_channels=in_channels,
                n_qubits=n_qubits,
                dropout=dropout,
                use_lora=use_lora,
                clinical_hard=clinical_hard,
                preprocess_mode=preprocess_mode,
                feature_only_mode=feature_only_mode,
            )
            self._owns_full_backbone = True

        # Quantum layer (supports dynamic QNAS genome or static StronglyEntanglingLayers)
        qnode, weight_shapes, self.qdev, self.qdev_name, self.n_meas = create_vqc_qnode(
            n_qubits, n_layers, diff_method=diff_method, reupload=reupload, genome=self.genome,
            qdevice=qdevice, dual_basis_only=dual_basis_only,
            use_qaf_attention=use_qaf_attention,
        )

        # Device coordination flag (Fix: H-07)
        is_pl_gpu = "gpu" in self.qdev_name.lower() or "cuda" in self.qdev_name.lower()
        transfers_to_cpu = (torch.cuda.is_available() and not is_pl_gpu)

        # Non-barren Gaussian init: θ ~ N(0, q_init_std)
        init_method = {
            "weights": partial(torch.nn.init.normal_, mean=0.0, std=q_init_std)
        }
        self.qlayer = QuantumDropoutTorchLayer(
            qnode, weight_shapes, init_method=init_method, p_drop=p_drop,
            transfers_to_cpu=transfers_to_cpu,
            jitter_std=jitter_std,
        )

        # SuperCircuit Weight Inheritance: Size mismatch raises ValueError (Fix: M-12)
        trained_w = None
        if self.genome is not None:
            trained_w = getattr(self.genome, "trained_weights", None)

        if trained_w is not None and len(trained_w) > 0:
            inherited = False
            for n, p in self.qlayer.named_parameters():
                if "weights" in n:
                    if p.data.numel() == len(trained_w):
                        with torch.no_grad():
                            p.data.copy_(torch.tensor(trained_w, dtype=p.dtype, device=p.device))
                        logger.info("M-12: Successfully inherited %d weights from QNAS SuperCircuit.", len(trained_w))
                        inherited = True
                    else:
                        raise ValueError(
                            f"M-12: Weight inheritance failed: genome has {len(trained_w)} params "
                            f"but qlayer '{n}' has {p.data.numel()}. Sizes must match exactly."
                        )
                    break

        # Nonlinear classification head
        self.head = nn.Sequential(
            nn.Linear(self.n_meas, 32),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, n_classes),
        )
        if isinstance(self.head[-1], nn.Linear):
            # Problems-#9: deterministic, seed-INDEPENDENT identical final-layer
            # init for hybrid AND baseline: same fan-in-averaged scale, drawn from
            # a dedicated local generator so neither model's head depends on the
            # other's RNG consumption order.
            g = torch.Generator(device="cpu")
            g.manual_seed(20240218)  # fixed init seed shared by both model classes
            with torch.no_grad():
                self.head[-1].weight.copy_(torch.randn(
                    self.head[-1].weight.shape, generator=g, dtype=torch.float32,
                ))
            self.head[-1].weight.data.mul_(0.01)  # target std ~= 0.01 per-fan-in neutral
            if self.head[-1].bias is not None:
                nn.init.zeros_(self.head[-1].bias)

        # Hooks for XAI
        self._latent_z: Optional[torch.Tensor] = None
        self._expectations: Optional[torch.Tensor] = None

    def to(self, *args, **kwargs):
        """Override to() to invalidate QNode caches upon device migration (Fix: M-14)."""
        result = super().to(*args, **kwargs)
        self._cached_state_qnode = None
        if hasattr(self, "_state_qnode_cache"):
            self._state_qnode_cache.clear()
        logger.debug("M-14: Model moved — state QNode caches invalidated.")
        return result

    @property
    def actual_n_qubits(self) -> int:
        """Active wire count from the canonical CircuitGenome (Fix: H-09)."""
        if self.genome is not None:
            g_qubits = getattr(self.genome, "n_qubits", None)
            if g_qubits is not None:
                return int(g_qubits)
        return int(self.n_qubits)

    @property
    def actual_n_layers(self) -> int:
        """Active circuit depth from the canonical CircuitGenome (Fix: H-09)."""
        if self.genome is not None:
            g_layers = getattr(self.genome, "n_layers", None)
            if g_layers is None and hasattr(self.genome, "layers"):
                g_layers = len(self.genome.layers)
            if g_layers is not None:
                return int(g_layers)
        return int(self.n_layers)

    @property
    def is_qnas_model(self) -> bool:
        """True if the model uses a discovered QNAS DAG architecture."""
        return self.genome is not None

    def get_state_qnode(self, diff_method: str = "backprop"):
        """Cached PennyLane state QNode keyed by (diff_method, n_qubits, n_layers, genome_id, qaf) (Fix: H-16)."""
        genome_id = (
            hashlib.md5(str(self.genome).encode()).hexdigest()[:8]
            if self.genome is not None else "none"
        )
        cache_key = (
            diff_method, self.actual_n_qubits, self.actual_n_layers,
            genome_id, getattr(self, "use_qaf_attention", False),
        )
        if not hasattr(self, "_state_qnode_cache"):
            self._state_qnode_cache = {}

        if cache_key not in self._state_qnode_cache:
            n_q = self.actual_n_qubits
            n_lay = self.actual_n_layers
            dev = qml.device("default.qubit", wires=n_q)
            reupload = getattr(self, "reupload", True)
            genome = getattr(self, "genome", None)
            use_qaf = getattr(self, "use_qaf_attention", False)

            build_qaf_circuit = None
            if use_qaf:
                try:
                    from ansatz.qaf import build_qaf_circuit
                except ImportError:
                    try:
                        from qml_models.ansatz.qaf import build_qaf_circuit
                    except ImportError:
                        pass

            @qml.qnode(dev, interface="torch", diff_method=diff_method)
            def _raw_state_circuit(inputs, weights):
                if use_qaf and build_qaf_circuit is not None:
                    build_qaf_circuit(inputs, weights, n_qubits=n_q, n_layers=n_lay)
                else:
                    _apply_ansatz(inputs, weights, n_q, n_lay, reupload=reupload, genome=genome)
                return qml.state()

            torch_dev = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")

            def _state_circuit(inputs, weights):
                needs_cpu = torch_dev.type == "cuda"
                in_c = inputs.cpu() if needs_cpu else inputs
                w_c = weights.cpu() if (needs_cpu and hasattr(weights, "cpu")) else weights
                st = _raw_state_circuit(in_c, w_c)
                if needs_cpu and isinstance(st, torch.Tensor):
                    return st.to(device=torch_dev)
                return st

            self._state_qnode_cache[cache_key] = _state_circuit
            logger.info("H-16: Built state QNode for cache key %s", cache_key)

        return self._state_qnode_cache[cache_key]

    def grow_layers(
        self,
        new_depth: int,
        init_std: float = 0.01,
        freeze_previous: bool = True,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Optional[torch.optim.Optimizer]:
        """Dynamically expand VQC ansatz depth and rebuild optimizer parameter groups (Fix: C-01, C-02).

        Returns
        -------
        optimizer : Optional[torch.optim.Optimizer]
            The optimizer with rebuilt param-groups referencing the new qlayer parameters.
            CALLER MUST REASSIGN: ``optimizer = model.grow_layers(..., optimizer=optimizer)``
        """
        old_depth = self.n_layers
        if new_depth <= old_depth:
            return optimizer

        # 1. Extract and clone existing trained quantum weights
        old_weights = None
        for n, p in self.qlayer.named_parameters():
            if "weights" in n:
                old_weights = p.data.clone()
                break

        # Remove previous freeze hook if one exists
        self.unfreeze_all_quantum_layers()

        # If discrete QNAS genome, expand genome layers
        if self.genome is not None and hasattr(self.genome, "layers"):
            self.genome = copy.deepcopy(self.genome)
            curr_genome_layers = len(self.genome.layers)
            if new_depth > curr_genome_layers:
                layers_to_add = new_depth - curr_genome_layers
                try:
                    from qnas import GateGene
                except ImportError:
                    try:
                        from qml_models.qnas import GateGene
                    except ImportError:
                        GateGene = None

                if GateGene is not None:
                    for l_idx in range(layers_to_add):
                        new_layer = []
                        for q in range(self.n_qubits):
                            new_layer.append(GateGene(gate_type="Ry", target=q, is_parameterized=True, param_count=1))
                        if curr_genome_layers > 0:
                            template = self.genome.layers[l_idx % curr_genome_layers]
                            for g in template:
                                if g.gate_type in ("CNOT", "CZ"):
                                    new_layer.append(copy.deepcopy(g))
                        else:
                            new_layer.append(GateGene(gate_type="CNOT", target=(l_idx + 1) % self.n_qubits, control=l_idx % self.n_qubits))
                        self.genome.layers.append(new_layer)
                    self.genome.compute_stats()

        old_qlayer = self.qlayer
        # 2. Build new QNode for expanded depth
        self.n_layers = new_depth
        qnode, weight_shapes, self.qdev, self.qdev_name, self.n_meas = create_vqc_qnode(
            self.n_qubits, new_depth, diff_method=self.diff_method, reupload=self.reupload,
            genome=self.genome, qdevice=self.qdevice,
            dual_basis_only=getattr(self, 'dual_basis_only', False),
            use_qaf_attention=getattr(self, 'use_qaf_attention', False),
        )

        # 3. Create new TorchLayer
        init_method = {
            "weights": partial(torch.nn.init.normal_, mean=0.0, std=init_std)
        }
        is_pl_gpu = "gpu" in self.qdev_name.lower() or "cuda" in self.qdev_name.lower()
        transfers_to_cpu = (torch.cuda.is_available() and not is_pl_gpu)

        new_qlayer = QuantumDropoutTorchLayer(
            qnode, weight_shapes, init_method=init_method, p_drop=self.p_drop,
            transfers_to_cpu=transfers_to_cpu,
        )

        param_device = next(self.parameters()).device
        new_qlayer = new_qlayer.to(param_device)

        # 5. Inherits parent module training mode (Fix: Problems-#6C)
        #    Zero-variational ROY gates + data-dependent U3 + entanglers mean a
        #    near-zero initialized extension is NOT identity, so a newly attached
        #    layer must not silently stay in training mode while the model is
        #    evaluating (would make inference stochastically noisy).
        new_qlayer.train(self.training)

        # 6. Copy trained weights and calculate exact freeze boundary (Fix: C-02)
        freeze_len = 0
        is_flat = False
        with torch.no_grad():
            for n, p in new_qlayer.named_parameters():
                if "weights" not in n:
                    continue

                if (p.data.dim() > 1 and old_weights is not None and
                        p.data.dim() == old_weights.dim() and
                        p.data.shape[1:] == old_weights.shape[1:]):
                    # Multi-dim: StronglyEntanglingLayers — safe layer-slice copy
                    copy_len = min(p.data.shape[0], old_weights.shape[0])
                    p.data[:copy_len] = old_weights[:copy_len].to(p.device)
                    if copy_len < p.data.shape[0]:
                        p.data[copy_len:] = torch.randn_like(p.data[copy_len:]) * init_std
                    freeze_len = copy_len
                    is_flat = False

                elif p.data.dim() == 1 and old_weights is not None and old_weights.dim() == 1:
                    # 1-D QNAS: compute freeze boundary from genome layer structure (Fix: C-02)
                    new_g = getattr(self, "genome", None)
                    old_gate_scalar_end = 0
                    if new_g is not None and hasattr(new_g, "layers"):
                        offset = 0
                        for l_idx, layer in enumerate(new_g.layers):
                            for gene in layer:
                                pc = getattr(gene, "param_count", 0)
                                if pc > 0:
                                    if l_idx < old_depth:
                                        old_gate_scalar_end = offset + pc
                                    offset += pc
                        freeze_len = min(old_gate_scalar_end, old_weights.numel(), p.data.numel())
                    else:
                        freeze_len = min(p.data.numel(), old_weights.numel())

                    copy_len = min(p.data.numel(), old_weights.numel())
                    p.data[:copy_len] = old_weights[:copy_len].to(p.device)
                    if copy_len < p.data.numel():
                        p.data[copy_len:] = torch.randn_like(p.data[copy_len:]) * init_std
                    is_flat = True

                else:
                    # Dimension mismatch fallback
                    copy_len = min(p.data.numel(), old_weights.numel()) if old_weights is not None else 0
                    p_flat = p.data.view(-1)
                    if copy_len > 0:
                        p_flat[:copy_len] = old_weights.view(-1)[:copy_len].to(p.device)
                    if copy_len < p.data.numel():
                        p_flat[copy_len:] = torch.randn_like(p_flat[copy_len:]) * init_std
                    freeze_len = copy_len
                    is_flat = True
                break

        # Validate prefix invariant before registering hook (Fix: C-02)
        if is_flat and freeze_len > 0 and old_weights is not None:
            for n, p in new_qlayer.named_parameters():
                if "weights" in n:
                    assert torch.allclose(
                        p.data.view(-1)[:old_weights.numel()],
                        old_weights.to(p.device),
                        atol=1e-6,
                    ), "C-02: Old gate weights were not copied to correct prefix positions."
                    break

        # 5. Attach gradient freeze hook
        if freeze_previous and freeze_len > 0:
            def _make_freeze_hook(n_frozen: int, flat_mode: bool):
                def hook(grad):
                    if grad is None:
                        return None
                    grad_masked = grad.clone()
                    if flat_mode:
                        grad_masked.view(-1)[:n_frozen] = 0.0
                    else:
                        grad_masked[:n_frozen] = 0.0
                    return grad_masked
                return hook

            for n, p in new_qlayer.named_parameters():
                if "weights" in n:
                    self._freeze_hook_handle = p.register_hook(_make_freeze_hook(freeze_len, is_flat))
                    break

        self.qlayer = new_qlayer
        self._cached_state_qnode = None
        if hasattr(self, "_state_qnode_cache"):
            self._state_qnode_cache.clear()

        # 7. Rebuild optimizer param-groups & flush momentum (Fix: C-01, Problems-#6A)
        if optimizer is not None:
            # Use optimizer defaults to render COMPLETE hyperparameter dicts (Problems-#6A:
            # the old rebuild produced {params, lr}-only groups, so AdamW's step crashed
            # with KeyError: 'betas')
            defaults = dict(getattr(optimizer, "defaults", {}))
            old_qlayer_param_ids = {id(p) for p in old_qlayer.parameters()}
            model_param_ids = {id(p) for p in self.parameters()}

            new_param_groups = []
            for group in optimizer.param_groups:
                live_params = [p for p in group["params"] if id(p) in model_param_ids]
                new_group = {k: v for k, v in group.items() if k != "params"}
                for dk, dv in defaults.items():
                    new_group.setdefault(dk, dv)
                new_group["params"] = live_params
                new_param_groups.append(new_group)

            # Add new qlayer params as a fresh group preserving the replaced quantum
            # group's lr (old code installed the FIRST group's lr — typically the
            # 1e-5 backbone rate — on quantum params)
            old_quantum_lrs = [
                g["lr"] for g in optimizer.param_groups
                if g.get("params") and any(id(p) in old_qlayer_param_ids for p in g["params"])
            ]
            quantum_lr = old_quantum_lrs[0] if old_quantum_lrs else (
                optimizer.param_groups[0]["lr"] if optimizer.param_groups else 1e-3
            )
            quantum_group = dict(defaults)
            quantum_group.update({
                "params": list(self.qlayer.parameters()),
                "lr": quantum_lr,
            })
            new_param_groups.append(quantum_group)

            # Flush momentum buffers only for replaced quantum parameters (FLAW-MOD-04)
            for p in list(optimizer.state.keys()):
                if id(p) in old_qlayer_param_ids:
                    del optimizer.state[p]

            optimizer.param_groups = new_param_groups

            # Problems-#6B: enforce the freeze at parameter-update level (gradient
            # masking alone cannot stop weight decay / momentum from mutating the
            # frozen prefix). Wrap optimizer.step to zero frozen grads pre-update.
            if freeze_previous and freeze_len > 0:
                frozen_param = None
                for n, p in self.qlayer.named_parameters():
                    if "weights" in n:
                        frozen_param = p
                        break
                if frozen_param is not None and callable(getattr(optimizer, "step", None)):
                    _n_frozen, _flat = freeze_len, is_flat
                    _orig_step = optimizer.step

                    def _freeze_guard_step(closure=None):
                        if frozen_param.grad is not None:
                            g = frozen_param.grad
                            if _flat:
                                g.view(-1)[:_n_frozen] = 0.0
                            else:
                                g[:_n_frozen] = 0.0
                        return _orig_step(closure)

                    optimizer.step = _freeze_guard_step
                    self._freeze_step_guard = (frozen_param, _orig_step)

            # Assert C-01 invariant: every trainable parameter is registered in the optimizer
            opt_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
            for p in self.parameters():
                if p.requires_grad:
                    assert id(p) in opt_param_ids, (
                        f"C-01: Trainable parameter missing from optimizer after grow_layers. Shape={p.shape}"
                    )

        return optimizer

    def unfreeze_all_quantum_layers(self) -> None:
        """Remove any active gradient masking hooks on quantum weights."""
        if self._freeze_hook_handle is not None:
            self._freeze_hook_handle.remove()
            self._freeze_hook_handle = None

    def _stack_expectations(self, exp, expected_batch_size: Optional[int] = None) -> torch.Tensor:
        """Stack list/tuple of expectation values returned by TorchLayer with shape verification (Fix: M-10)."""
        if isinstance(exp, (list, tuple)):
            exp = torch.stack(exp, dim=-1)
        if exp.ndim == 1:
            exp = exp.unsqueeze(0)
        if exp.shape[-1] != self.n_meas:
            raise RuntimeError(
                f"M-10: QNode returned {exp.shape[-1]} values but n_meas={self.n_meas}. "
                "PennyLane version change or QAF mismatch."
            )
        if expected_batch_size is not None and exp.shape[0] != expected_batch_size:
            raise RuntimeError(
                f"M-10: Expected batch={expected_batch_size}, got {exp.shape[0]}."
            )
        return exp

    def _forward_core(
        self, x: torch.Tensor, use_cached_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared forward logic. Atomically updates all activation caches (Fix: H-12)."""
        z = self.backbone(x, use_cached_features)
        exp = self._stack_expectations(self.qlayer(z), expected_batch_size=z.shape[0])
        logits = self.head(exp)
        self._latent_z = z.detach()
        self._expectations = exp.detach()
        return logits, z, exp

    def forward(self, x: torch.Tensor, use_cached_features: bool = False) -> torch.Tensor:
        logits, _, _ = self._forward_core(x, use_cached_features)
        return logits

    def forward_with_latent(
        self, x: torch.Tensor, use_cached_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (logits, z_latent, quantum_expectations). Also updates get_latent/get_expectations (Fix: H-12)."""
        return self._forward_core(x, use_cached_features)

    def get_latent(self) -> Optional[torch.Tensor]:
        return self._latent_z

    def get_r_vectors(self) -> Optional[torch.Tensor]:
        """Return 3D unit Bloch vectors (B, n_qubits, 3) on S^2 from backbone."""
        return self.backbone.get_r_vectors()

    def get_expectations(self) -> Optional[torch.Tensor]:
        return self._expectations

    def offload_vision_encoder(self):
        """Offload the frozen ViT backbone to free GPU and RAM memory."""
        if hasattr(self, "backbone") and hasattr(self.backbone, "offload_vision_encoder"):
            self.backbone.offload_vision_encoder()

    def load_vision_encoder(self):
        """Reload the frozen ViT backbone on demand."""
        if hasattr(self, "backbone") and hasattr(self.backbone, "load_vision_encoder"):
            self.backbone.load_vision_encoder()

    def count_parameters(self) -> Dict[str, int]:
        """Return parameter counts broken down by component."""
        backbone_p = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        quantum_p  = sum(p.numel() for p in self.qlayer.parameters() if p.requires_grad)
        head_p     = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        return {
            "backbone": backbone_p,
            "quantum":  quantum_p,
            "head":     head_p,
            "total":    backbone_p + quantum_p + head_p,
        }


# ---------------------------------------------------------------------------
# Parameter-Matched Classical Baseline
# ---------------------------------------------------------------------------
class ClassicalBaseline(nn.Module):
    """Same backbone + matched-parameter MLP replacing the quantum layer (Fix: H-10, H-11, H-12, M-03).

    The MLP is sized via exact integer parameter search to match the trainable parameter count
    of the quantum circuit for a scientifically rigorous comparison.
    """

    @staticmethod
    def _validate_constructor_args(n_qubits: int, n_layers: int, n_classes: int, dropout: float) -> None:
        if n_qubits < 1:
            raise ValueError(f"M-07: n_qubits must be >= 1, got {n_qubits}")
        if n_layers < 1:
            raise ValueError(f"M-07: n_layers must be >= 1, got {n_layers}")
        if n_classes < 2:
            raise ValueError(f"M-07: n_classes must be >= 2, got {n_classes}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"M-07: dropout must be in [0, 1), got {dropout}")

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 2,
        n_qubits: int = 4,
        n_layers: int = 3,
        dropout: float = 0.3,
        use_lora: bool = False,
        mlp_hidden_dim: Optional[int] = None,
        mlp_in_dim: Optional[int] = None,
        dual_basis_only: bool = False,
        clinical_hard: bool = False,
        genome: Optional[object] = None,
        use_qaf_attention: bool = False,
        backbone: Optional[BiomedCLIPQuantumBackbone] = None,
        preprocess_mode: Optional[ImagePreprocessMode] = None,
        feature_only_mode: bool = False,
    ):
        super().__init__()
        self._validate_constructor_args(n_qubits, n_layers, n_classes, dropout)

        self.n_qubits  = n_qubits
        self.n_classes = n_classes
        self.clinical_hard = clinical_hard
        self.genome = genome
        self.use_qaf_attention = use_qaf_attention

        # Same backbone (BiomedCLIP + residual adapter; may share the frozen encoder, Fix: M-03)
        if backbone is not None:
            self.backbone = backbone
            self._owns_full_backbone = False
        else:
            self.backbone = BiomedCLIPQuantumBackbone(
                in_channels=in_channels,
                n_qubits=n_qubits,
                dropout=dropout,
                use_lora=use_lora,
                clinical_hard=clinical_hard,
                preprocess_mode=preprocess_mode,
                feature_only_mode=feature_only_mode,
            )
            self._owns_full_backbone = True

        # Compute quantum parameter count to match (Fix: H-10, H-11)
        if use_qaf_attention:
            q_params = n_layers * n_qubits * 4
        elif genome is not None:
            _g = genome
            if isinstance(_g, dict):
                try:
                    from qnas import CircuitGenome
                except ImportError:
                    from qml_models.qnas import CircuitGenome
                _g = CircuitGenome.from_dict(_g)
            q_params = max(1, getattr(_g, "total_params", 1))
        else:
            q_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
            q_params = int(torch.tensor(list(q_shape)).prod().item())

        n_meas_val = compute_n_meas(n_qubits, dual_basis_only=(dual_basis_only or use_qaf_attention))
        in_f_val = mlp_in_dim if mlp_in_dim is not None else 3 * n_qubits
        self.n_meas = n_meas_val

        # Exact integer search for optimal MLP hidden width (Fix: H-11)
        if mlp_hidden_dim is not None:
            h = mlp_hidden_dim
        else:
            best_h, best_delta = 2, float("inf")
            for candidate_h in range(2, max(q_params * 2 + 10, 20)):
                mlp_p = (in_f_val * candidate_h + candidate_h) + (candidate_h * n_meas_val + n_meas_val)
                delta = abs(mlp_p - q_params)
                if delta < best_delta:
                    best_delta, best_h = delta, candidate_h
                if delta == 0:
                    break
            h = best_h
            logger.info(
                "H-11: Baseline MLP hidden=%d, mlp_params≈%d, quantum_target=%d, delta=%d",
                h, (in_f_val * h + h) + (h * n_meas_val + n_meas_val), q_params, best_delta,
            )

        # Symmetrical Classical Control Baseline
        self.mlp = nn.Sequential(
            nn.Linear(in_f_val, h),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(h, n_meas_val),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(n_meas_val, 32),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, n_classes),
        )

        # Problems-#9: IDENTICAL head initialization procedure to HybridCNNVQC —
        # deterministic, seed-independent (dedicated fixed-seed generator).
        if isinstance(self.head[-1], nn.Linear):
            g = torch.Generator(device="cpu")
            g.manual_seed(20240218)
            with torch.no_grad():
                self.head[-1].weight.copy_(torch.randn(
                    self.head[-1].weight.shape, generator=g, dtype=torch.float32,
                ))
            self.head[-1].weight.data.mul_(0.01)
            if self.head[-1].bias is not None:
                nn.init.zeros_(self.head[-1].bias)

        # Hook for latent
        self._latent_z: Optional[torch.Tensor] = None
        self._expectations: Optional[torch.Tensor] = None

    def _forward_core(
        self, x: torch.Tensor, use_cached_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared forward logic for ClassicalBaseline (Fix: H-12)."""
        z = self.backbone(x, use_cached_features)
        h = self.mlp(z)
        logits = self.head(h)
        self._latent_z = z.detach()
        self._expectations = h.detach()
        return logits, z, h

    def forward(self, x: torch.Tensor, use_cached_features: bool = False) -> torch.Tensor:
        logits, _, _ = self._forward_core(x, use_cached_features)
        return logits

    def forward_with_latent(
        self, x: torch.Tensor, use_cached_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._forward_core(x, use_cached_features)

    def get_latent(self) -> Optional[torch.Tensor]:
        return self._latent_z

    def get_expectations(self) -> Optional[torch.Tensor]:
        return self._expectations

    def get_r_vectors(self) -> Optional[torch.Tensor]:
        """Return 3D unit Bloch vectors (B, n_qubits, 3) on S^2 from backbone."""
        return self.backbone.get_r_vectors()

    def offload_vision_encoder(self):
        """Offload the frozen ViT backbone to free GPU and RAM memory."""
        if hasattr(self, "backbone") and hasattr(self.backbone, "offload_vision_encoder"):
            self.backbone.offload_vision_encoder()

    def load_vision_encoder(self):
        """Reload the frozen ViT backbone on demand."""
        if hasattr(self, "backbone") and hasattr(self.backbone, "load_vision_encoder"):
            self.backbone.load_vision_encoder()

    def count_parameters(self) -> Dict[str, int]:
        backbone_p = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        mlp_p      = sum(p.numel() for p in self.mlp.parameters() if p.requires_grad)
        head_p     = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        return {
            "backbone": backbone_p,
            "mlp":      mlp_p,
            "head":     head_p,
            "total":    backbone_p + mlp_p + head_p,
        }


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_models(
    in_channels: int = 1,
    n_classes: int = 2,
    n_qubits: int = 4,
    n_layers: int = 3,
    device: Optional[torch.device] = None,
    dropout: float = 0.3,
    diff_method: str = "adjoint",
    reupload: bool = True,
    p_drop: float = 0.10,
    use_lora: bool = False,
    genome: Optional[object] = None,
    qdevice: Optional[str] = None,
    q_init_std: float = 0.01,
    dual_basis_only: bool = False,
    use_qaf_attention: bool = False,
    clinical_hard: bool = False,
    finetune_backbone: bool = False,
    ft_dataset_name: Optional[str] = None,
    ft_n_unfreeze: int = 2,
    ft_base_dir: Optional[str] = None,
    preprocess_mode: Optional[ImagePreprocessMode] = None,
    feature_only_mode: bool = False,
    shared_backbone: bool = True,
    jitter_std: float = 0.05,
) -> Tuple[HybridCNNVQC, ClassicalBaseline]:
    """Instantiate both hybrid and classical baseline models, moving to device (Fix: M-03, H-10, L-03).

    When shared_backbone=True, the frozen BiomedCLIP vision encoder is shared between the
    hybrid model and classical baseline, but each model receives its OWN trainable
    CQ adapter + residual skip (deep-copied to identical initial weights for a fair
    comparison). The old sharing of the entire backbone made the two models
    interfere: baseline optimization mutated the hybrid representation (Fix: Problems-#2).
    """
    if device is None:
        device = get_torch_device()

    # Create the anchor backbone when requested (Fix: Problems-#2)
    backbone_inst = None
    if shared_backbone:
        backbone_inst = BiomedCLIPQuantumBackbone(
            in_channels=in_channels,
            n_qubits=n_qubits,
            dropout=dropout,
            use_lora=use_lora,
            clinical_hard=clinical_hard,
            preprocess_mode=preprocess_mode,
            feature_only_mode=feature_only_mode,
        ).to(device)

    hybrid = HybridCNNVQC(
        in_channels=in_channels, n_classes=n_classes, n_qubits=n_qubits, n_layers=n_layers,
        diff_method=diff_method, dropout=dropout,
        reupload=reupload, p_drop=p_drop, use_lora=use_lora,
        genome=genome, qdevice=qdevice, q_init_std=q_init_std,
        dual_basis_only=dual_basis_only,
        use_qaf_attention=use_qaf_attention,
        clinical_hard=clinical_hard,
        backbone=backbone_inst,
        preprocess_mode=preprocess_mode,
        feature_only_mode=feature_only_mode,
        jitter_std=jitter_std,
    ).to(device)

    # Apply finetuned BiomedCLIP ViT weights if available
    if finetune_backbone and ft_dataset_name and _FINETUNING_AVAILABLE:
        loaded = hybrid.backbone.load_finetuned_vit(
            dataset_name=ft_dataset_name,
            n_unfreeze=ft_n_unfreeze,
            base_dir=ft_base_dir,
            device=device,
        )
        if loaded:
            logger.info("Loaded finetuned BiomedCLIP for '%s' (%d layers)", ft_dataset_name, ft_n_unfreeze)
        else:
            logger.info("No finetuned checkpoint for '%s', using original BiomedCLIP", ft_dataset_name)

    baseline = ClassicalBaseline(
        in_channels=in_channels, n_classes=n_classes, n_qubits=n_qubits, n_layers=n_layers,
        dropout=dropout, use_lora=use_lora,
        dual_basis_only=dual_basis_only,
        clinical_hard=clinical_hard,
        genome=genome,
        use_qaf_attention=use_qaf_attention,
        backbone=backbone_inst,
        preprocess_mode=preprocess_mode,
        feature_only_mode=feature_only_mode,
    ).to(device)

    # Fix: Problems-#2 — the baseline must NOT share the hybrid's trainable adapter.
    # Copy the anchor backbone's weights into a fresh backbone (IDENTICAL initial
    # weights, frozen vision encoder shared only via its Parameters since it is
    # frozen) so both models start from exactly the same representation but
    # optimize independently. (We rebuild + state_dict-copy instead of
    # copy.deepcopy: nn.Module.__deepcopy__ keeps parameter *tensor* references
    # and shared mutable attribute objects, which silently re-couples optimization.)
    if shared_backbone and backbone_inst is not None:
        baseline_backbone = BiomedCLIPQuantumBackbone(
            in_channels=in_channels,
            n_qubits=n_qubits,
            dropout=dropout,
            use_lora=use_lora,
            clinical_hard=clinical_hard,
            preprocess_mode=preprocess_mode,
            feature_only_mode=feature_only_mode,
        ).to(device)
        with torch.no_grad():
            baseline_backbone.load_state_dict(backbone_inst.state_dict(), strict=False)
        if feature_only_mode or backbone_inst.vision_encoder is None:
            baseline_backbone.vision_encoder = backbone_inst.vision_encoder
        baseline.backbone = baseline_backbone
        baseline._owns_full_backbone = True
    # Also mirror finetuned ViT weights if they were just applied to the hybrid
    if shared_backbone and backbone_inst is not None and finetune_backbone and ft_dataset_name and _FINETUNING_AVAILABLE:
        with torch.no_grad():
            if hybrid.backbone.vision_encoder is not None and baseline_backbone.vision_encoder is not None:
                baseline_backbone.vision_encoder.load_state_dict(hybrid.backbone.vision_encoder.state_dict())

    logger.info("Hybrid   trainable params : %d", hybrid.count_parameters()["total"])
    logger.info("Baseline trainable params: %d", baseline.count_parameters()["total"])
    # Problems-#9: honest parameter-match accounting — the baseline MLP only
    # approximately matches the quantum trainable-parameter count (integer
    # grid search finds the CLOSEST width). Do not claim exact matching.
    h_q = hybrid.count_parameters()
    b_c = baseline.count_parameters()
    h_q_vp = h_q["quantum"]
    b_mlp = b_c["mlp"]
    logger.info(
        "Problems-#9 baseline matching: quantum_trainable=%d vs baseline_mlp=%d "
        "(delta=%d, approximately matched — NOT exactly; extras: extra GELU+dropout "
        "layers and different init for hidden layer)",
        h_q_vp, b_mlp, abs(b_mlp - h_q_vp),
    )
    logger.info("PennyLane dev  : %s", hybrid.qdev_name)
    logger.info("PyTorch dev    : %s", device)
    logger.info("n_meas (local observables): %d", hybrid.n_meas)
    logger.info("dual_basis_only: %s | use_qaf: %s", dual_basis_only, use_qaf_attention)
    logger.info("clinical_hard  : %s", clinical_hard)
    logger.info("finetune_backbone: %s", finetune_backbone)
    return hybrid, baseline


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = get_torch_device()
    logger.info("Device: %s", device)

    n_qubits = 4
    n_meas   = compute_n_meas(n_qubits)
    logger.info(
        "Expected n_meas for %d qubits: %d (<Z>*%d + <ZZ>*%d + <X>*%d)",
        n_qubits, n_meas, n_qubits, n_qubits - 1, n_qubits
    )

    hybrid, baseline = build_models(
        in_channels=1, n_classes=2, n_qubits=n_qubits, n_layers=3, device=device,
        diff_method="adjoint", reupload=True, feature_only_mode=True,
    )

    # Test forward pass with a small synthetic batch (cached features = 512-d)
    x = torch.randn(2, 512, device=device)
    logits_h = hybrid(x, use_cached_features=True)
    logits_b = baseline(x, use_cached_features=True)
    logger.info("Hybrid   output shape: %s", logits_h.shape)
    logger.info("Baseline output shape: %s", logits_b.shape)
    assert logits_h.shape == (2, 2), f"Expected (2,2), got {logits_h.shape}"
    assert logits_b.shape == (2, 2), f"Expected (2,2), got {logits_b.shape}"

    # Verify expectations shape
    exp = hybrid.get_expectations()
    logger.info("Hybrid expectations: %s (should be [2, %d])", exp.shape, n_meas)
    assert exp.shape == (2, n_meas), f"Expected (2,{n_meas}), got {exp.shape}"

    # Verify latent z range is symmetric and bounded in [-pi, pi]
    z = hybrid.get_latent()
    logger.info("Latent z range       : [%.4f, %.4f] (bounded in [-pi, pi])", z.min(), z.max())
    assert z.min() >= -math.pi - 1e-4 and z.max() <= math.pi + 1e-4, \
        f"Latent z out of bounds: [{z.min():.4f}, {z.max():.4f}]"

    # Verify clinical_hard mode produces ranges: theta in [0.05*pi, 0.95*pi], phi, lam in [-0.95*pi, 0.95*pi] (Fix: C-03)
    hybrid_hard, _ = build_models(
        in_channels=1, n_classes=2, n_qubits=n_qubits, n_layers=2, device=device,
        diff_method="adjoint", reupload=True, clinical_hard=True, feature_only_mode=True,
    )
    _ = hybrid_hard(x, use_cached_features=True)
    z_hard = hybrid_hard.get_latent()
    n_q = n_qubits
    theta_hard = z_hard[:, :n_q]
    phi_hard   = z_hard[:, n_q:2*n_q]
    lam_hard   = z_hard[:, 2*n_q:]
    assert theta_hard.min() >= 0.05 * math.pi - 1e-4, \
        f"clinical_hard theta below 0.05*pi: {theta_hard.min():.4f}"
    assert theta_hard.max() <= 0.95 * math.pi + 1e-4, \
        f"clinical_hard theta above 0.95*pi: {theta_hard.max():.4f}"
    # Problems-#4: azimuthal compression is periodicity-preserving: phi stays in
    # [-pi, pi] with ±pi EXACTLY equivalent (the old phi*0.95 bound is obsolete).
    assert phi_hard.abs().max() <= math.pi + 1e-4, \
        f"clinical_hard phi out of [-pi, pi]: {phi_hard.abs().max():.4f}"
    assert lam_hard.abs().max() <= 1e-7, \
        f"clinical_hard lam must be identically zero (pure-state prep): {lam_hard.abs().max():.4f}"
    logger.info(
        "clinical_hard range — theta: [%.4f, %.4f] phi: [%.4f, %.4f] lam: [%.4f, %.4f]",
        theta_hard.min(), theta_hard.max(),
        phi_hard.min(), phi_hard.max(),
        lam_hard.min(), lam_hard.max(),
    )

    # Verify identity mapping for quiescent input
    z_zero = hybrid.backbone(torch.zeros(1, 512, device=device), use_cached_features=True)
    logger.info("Zero-feature latent magnitude: %.6f", z_zero.abs().max().item())

    # Backward pass verification
    loss = logits_h.sum()
    loss.backward()
    logger.info("Backward pass verified successfully!")

    # Verify grow_layers returns rebuilt optimizer with non-None gradients (Fix: C-01)
    opt = torch.optim.AdamW(hybrid.parameters(), lr=1e-3)
    opt = hybrid.grow_layers(new_depth=4, optimizer=opt)
    assert opt is not None, "C-01: grow_layers must return rebuilt optimizer"
    logits_grown = hybrid(x, use_cached_features=True)
    opt.zero_grad()
    loss_grown = logits_grown.sum()
    loss_grown.backward()
    for n, p in hybrid.qlayer.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"C-01: qlayer parameter {n} has no gradient after grow_layers"
    opt.step()
    logger.info("C-01 grow_layers optimizer update verified successfully!")

    logger.info("\n[OK] model.py self-test passed.")
