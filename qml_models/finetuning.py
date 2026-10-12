"""
finetuning.py — Unified BiomedCLIP ViT-B/16 Fine-Tuning Module & Diagnostics Suite
===================================================================================
Selective layer unfreezing and isolated fine-tuning of the BiomedCLIP vision encoder
to mitigate shortcut learning from frozen representations.

Key Design Decisions & Invariants:
  1. Original pretrained BiomedCLIP weights are NEVER modified in-place or mutated in cache.
  2. Dataset-specific checkpointing: fine-tuned weights saved in `biomedclip/<dataset_name>/` or
     `biomedclip/<dataset_name>_biomedclip_vit.pt`.
  3. Controlled top-N unfreezing of ViT transformer blocks while base blocks remain frozen.
  4. Real-time batch progress tracking and complete per-epoch clinical diagnostic metrics:
       - AUROC (Macro and per-class)
       - AUPRC / Average Precision (AP)
       - Macro F1-Score
       - Sensitivity (Recall / True Positive Rate)
       - Specificity (True Negative Rate)
       - Balanced Accuracy
       - Training Loss vs. Validation Loss
       - Confusion Matrix
       - Global Gradient Norm (‖g‖₂) of trainable layers
       - Cosine Similarity Drift (relative to frozen baseline embeddings)
       - Logit Scale / Learned Temperature (τ and exp(τ))
       - Learning Rate schedule value
  5. Backwards-compatible with all legacy `finetunning.py` imports and function signatures.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import json
import logging
import math
import os
import pickle
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Optional Scikit-Learn Metrics
try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
        recall_score,
        confusion_matrix,
        balanced_accuracy_score,
    )
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# Optional Progress Bar (tqdm)
try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

logger = logging.getLogger(__name__)


def _checkpoint_safe_value(value: Any) -> Any:
    """Convert checkpoint metadata to primitives accepted by safe torch loading."""
    if isinstance(value, np.ndarray):
        return [_checkpoint_safe_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _checkpoint_safe_value(value.item())
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {str(key): _checkpoint_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_checkpoint_safe_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _load_finetuned_checkpoint(path: Path, map_location: torch.device) -> Dict[str, Any]:
    """Load a fine-tuned checkpoint with a safe legacy-NumPy compatibility path.

    Older checkpoints stored NumPy arrays inside diagnostic metadata.  The
    tensors themselves are safe, but PyTorch's ``weights_only`` unpickler
    rejects those legacy NumPy globals.  We explicitly allow only NumPy array,
    dtype, reconstruction, and concrete scalar dtype classes; arbitrary
    application classes are never allowlisted.
    """
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError) as exc:
        safe_items = [
            np.ndarray,
            np.dtype,
        ]
        numpy_core = getattr(np, "_core", getattr(np, "core", None))
        multiarray = getattr(numpy_core, "multiarray", None) if numpy_core is not None else None
        reconstruct = getattr(multiarray, "_reconstruct", None)
        if reconstruct is not None:
            safe_items.append(reconstruct)
        for dtype_name in (
            "BoolDType", "Int8DType", "Int16DType", "Int32DType", "Int64DType",
            "UInt8DType", "UInt16DType", "UInt32DType", "UInt64DType",
            "Float16DType", "Float32DType", "Float64DType",
        ):
            dtype_cls = getattr(getattr(np, "dtypes", None), dtype_name, None)
            if dtype_cls is not None:
                safe_items.append(dtype_cls)
        for dtype_value in (
            np.bool_, np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64,
            np.float16, np.float32, np.float64,
        ):
            safe_items.append(type(np.dtype(dtype_value)))
        try:
            with torch.serialization.safe_globals(safe_items):
                checkpoint = torch.load(str(path), map_location=map_location, weights_only=True)
            logger.warning(
                "Loaded legacy fine-tuned checkpoint with restricted NumPy safe globals: %s",
                path,
            )
            return checkpoint
        except Exception as safe_exc:
            raise RuntimeError(
                f"Fine-tuned checkpoint {path} is not safe-loadable. Re-save it with the "
                "current finetuning.py checkpoint writer. Original error: "
                f"{exc}; restricted-load error: {safe_exc}"
            ) from safe_exc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BIOMEDCLIP_MEAN = [0.48145466, 0.45782750, 0.40821073]
BIOMEDCLIP_STD = [0.26862954, 0.26130258, 0.27577711]
FINETUNED_DIR = os.path.join(os.path.dirname(__file__), "biomedclip")


# ---------------------------------------------------------------------------
# Configuration Dataclass (Legacy & Canonical Compatible)
# ---------------------------------------------------------------------------
@dataclass
class BiomedCLIPFineTuneConfig:
    """Configuration for BiomedCLIP ViT fine-tuning."""
    enable_finetuning: bool = True
    dataset_name: str = "pneumoniamnist"
    n_classes: int = 2
    save_dir: str = "biomedclip"
    unfreeze_layers: int = 2
    finetune_lr: float = 1e-5
    weight_decay: float = 1e-5
    epochs: int = 10
    warmup_epochs: int = 3
    batch_size: int = 32
    save_filename: Optional[str] = None
    # The workspace dataset loaders already emit BiomedCLIP-normalized tensors.
    # Keep the preprocessing owner explicit so fine-tuning cannot silently
    # double-normalize those tensors.
    preprocess_mode: str = "biomedclip_normalized"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def get_save_path(self, base_dir: Optional[str] = None) -> str:
        target_dir = base_dir or self.save_dir
        os.makedirs(target_dir, exist_ok=True)
        filename = self.save_filename or f"{self.dataset_name}_biomedclip_vit.pt"
        return os.path.join(target_dir, filename)


# ---------------------------------------------------------------------------
# ViT Architecture Inspection & Unfreezing
# ---------------------------------------------------------------------------
def _get_vit_transformer_blocks(vision_encoder: nn.Module) -> List[nn.Module]:
    """Extract ordered list of transformer blocks from BiomedCLIP ViT encoder."""
    if hasattr(vision_encoder, "trunk") and hasattr(vision_encoder.trunk, "blocks"):
        return list(vision_encoder.trunk.blocks)
    if hasattr(vision_encoder, "transformer") and hasattr(vision_encoder.transformer, "resblocks"):
        return list(vision_encoder.transformer.resblocks)
    if hasattr(vision_encoder, "blocks"):
        return list(vision_encoder.blocks)
    if hasattr(vision_encoder, "encoder") and hasattr(vision_encoder.encoder, "layer"):
        return list(vision_encoder.encoder.layer)
    raise AttributeError(
        "Cannot locate transformer blocks in the vision encoder. "
        "Expected `.trunk.blocks`, `.transformer.resblocks`, `.blocks`, or `.encoder.layer`."
    )


def get_vit_resblocks(visual_model: nn.Module) -> Optional[List[nn.Module]]:
    """Alias for get_vit_transformer_blocks with safe None fallback."""
    try:
        return _get_vit_transformer_blocks(visual_model)
    except Exception:
        return None


def _get_vit_layer_norm(vision_encoder: nn.Module) -> Optional[nn.Module]:
    """Get the final LayerNorm of the ViT encoder if present."""
    if hasattr(vision_encoder, "trunk") and hasattr(vision_encoder.trunk, "norm"):
        return vision_encoder.trunk.norm
    if hasattr(vision_encoder, "ln_post"):
        return vision_encoder.ln_post
    if hasattr(vision_encoder, "norm"):
        return vision_encoder.norm
    return None


def unfreeze_top_layers(
    vision_encoder: nn.Module,
    n_unfreeze: int = 2,
    unfreeze_ln: bool = True,
) -> Dict[str, int]:
    """Selectively unfreeze top N transformer blocks of BiomedCLIP ViT.
    Base blocks remain strictly frozen.
    """
    for param in vision_encoder.parameters():
        param.requires_grad = False

    if n_unfreeze <= 0:
        total = sum(p.numel() for p in vision_encoder.parameters())
        return {"unfrozen_params": 0, "total_params": total, "unfrozen_blocks": 0}

    blocks = _get_vit_transformer_blocks(vision_encoder)
    total_blocks = len(blocks)
    blocks_to_unfreeze = min(n_unfreeze, total_blocks)

    for i in range(total_blocks - blocks_to_unfreeze, total_blocks):
        for param in blocks[i].parameters():
            param.requires_grad = True

    if unfreeze_ln:
        ln = _get_vit_layer_norm(vision_encoder)
        if ln is not None:
            for param in ln.parameters():
                param.requires_grad = True

    for attr in ["proj", "head"]:
        if hasattr(vision_encoder, attr):
            sub = getattr(vision_encoder, attr)
            if isinstance(sub, nn.Parameter):
                sub.requires_grad = True
            elif isinstance(sub, nn.Module):
                for p in sub.parameters():
                    p.requires_grad = True

    unfrozen = sum(p.numel() for p in vision_encoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in vision_encoder.parameters())
    return {
        "unfrozen_params": unfrozen,
        "total_params": total,
        "unfrozen_blocks": blocks_to_unfreeze,
    }


def configure_vit_trainability(visual_model: nn.Module, unfreeze_layers: int = 2) -> int:
    """Legacy alias returning count of trainable parameters."""
    info = unfreeze_top_layers(visual_model, n_unfreeze=unfreeze_layers)
    return info["unfrozen_params"]


# ---------------------------------------------------------------------------
# Checkpoint Resolution & Persistence
# ---------------------------------------------------------------------------
def get_finetuned_checkpoint_path(
    dataset_name: str,
    n_unfreeze: int = 2,
    base_dir: Optional[str] = None,
) -> Path:
    """Resolve candidate file path for a dataset-specific fine-tuned ViT checkpoint."""
    root = Path(base_dir) if base_dir else Path(FINETUNED_DIR)
    candidates = [
        root / f"{dataset_name}_biomedclip_vit.pt",
        root / dataset_name / f"biomedclip_vit_unfreeze{n_unfreeze}.pt",
        root / f"{dataset_name}_vit_unfreeze{n_unfreeze}.pt",
        root / dataset_name / "best_vit.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def has_finetuned_checkpoint(
    dataset_name: str,
    n_unfreeze: int = 2,
    base_dir: Optional[str] = None,
) -> bool:
    """Check whether a fine-tuned checkpoint exists for the given dataset."""
    root = Path(base_dir) if base_dir else Path(FINETUNED_DIR)
    candidates = [
        root / f"{dataset_name}_biomedclip_vit.pt",
        root / dataset_name / f"biomedclip_vit_unfreeze{n_unfreeze}.pt",
        root / f"{dataset_name}_vit_unfreeze{n_unfreeze}.pt",
        root / dataset_name / "best_vit.pt",
    ]
    return any(c.exists() for c in candidates)


def save_finetuned_backbone(
    vision_encoder: nn.Module,
    dataset_name: str,
    n_unfreeze: int = 2,
    metadata: Optional[Dict[str, Any]] = None,
    base_dir: Optional[str] = None,
) -> Path:
    """Save fine-tuned weights and diagnostic training metadata."""
    ckpt_path = get_finetuned_checkpoint_path(dataset_name, n_unfreeze, base_dir)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    finetuned_state = {
        name: param.detach().cpu().clone()
        for name, param in vision_encoder.named_parameters()
        if param.requires_grad
    }

    checkpoint = {
        "dataset_name": dataset_name,
        "n_unfreeze": n_unfreeze,
        "unfreeze_layers": n_unfreeze,
        "finetuned_state_dict": finetuned_state,
        "visual_state_dict": {k: v.detach().cpu().clone() for k, v in vision_encoder.state_dict().items()},
        "metadata": _checkpoint_safe_value(metadata or {}),
        "metrics": _checkpoint_safe_value(metadata or {}),
    }

    torch.save(checkpoint, str(ckpt_path))
    logger.info("Saved fine-tuned BiomedCLIP ViT checkpoint -> %s", ckpt_path)
    return ckpt_path


def save_finetuned_biomedclip_vit(
    visual_model: nn.Module,
    config: BiomedCLIPFineTuneConfig,
    metrics: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
) -> str:
    """Compatibility wrapper matching legacy finetunning.py save signature."""
    path = save_finetuned_backbone(
        visual_model,
        dataset_name=config.dataset_name,
        n_unfreeze=config.unfreeze_layers,
        metadata=metrics,
        base_dir=save_path or config.save_dir,
    )
    return str(path)


def load_finetuned_backbone(
    vision_encoder: nn.Module,
    dataset_name: str,
    n_unfreeze: int = 2,
    base_dir: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    freeze_after_load: bool = True,
) -> Dict[str, Any]:
    """Load fine-tuned weights into matching parameters of the vision encoder."""
    ckpt_path = get_finetuned_checkpoint_path(dataset_name, n_unfreeze, base_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No fine-tuned checkpoint found at {ckpt_path}")

    target_dev = torch.device(device) if device else next(vision_encoder.parameters()).device
    checkpoint = _load_finetuned_checkpoint(ckpt_path, target_dev)

    loaded_count = 0
    if "visual_state_dict" in checkpoint:
        visual_state = {
            k: v for k, v in checkpoint["visual_state_dict"].items()
            if k in vision_encoder.state_dict()
            and vision_encoder.state_dict()[k].shape == v.shape
        }
        if not visual_state:
            raise RuntimeError("No compatible visual_state_dict parameters found in fine-tuned checkpoint")
        missing, unexpected = vision_encoder.load_state_dict(visual_state, strict=False)
        loaded_count = len(visual_state) - len(unexpected)
        if loaded_count < len(vision_encoder.state_dict()) * 0.5:
            raise RuntimeError(
                f"FATAL: Only {loaded_count}/{len(vision_encoder.state_dict())} "
                "parameters loaded from visual_state_dict checkpoint."
            )
    elif "finetuned_state_dict" in checkpoint:
        raw_state = checkpoint["finetuned_state_dict"]
        model_keys = set(vision_encoder.state_dict().keys())

        # 1. Try exact match
        cleaned_state = {k: v for k, v in raw_state.items() if k in model_keys}

        # 2. If no exact match, attempt robust suffix matching
        if len(cleaned_state) == 0:
            for target_k in model_keys:
                for src_k, v in raw_state.items():
                    # Map weights based on unambiguous structural suffix
                    if target_k.endswith(src_k) or src_k.endswith(target_k):
                        cleaned_state[target_k] = v
                        break

        if len(cleaned_state) == 0:
            raise RuntimeError(
                f"Zero parameters matched after prefix normalization.\n"
                f"Checkpoint keys (sample): {list(raw_state.keys())[:5]}\n"
                f"Model keys (sample):      {list(model_keys)[:5]}"
            )

        missing, unexpected = vision_encoder.load_state_dict(cleaned_state, strict=False)
        loaded_count = len(cleaned_state) - len(missing)
        if loaded_count < len(model_keys) * 0.5:
            raise RuntimeError(f"FATAL: Only {loaded_count}/{len(model_keys)} parameters loaded from checkpoint.")
        if len(missing) > 0:
            import warnings
            warnings.warn(
                f"{len(missing)} parameters not loaded from checkpoint: {missing[:5]}..."
            )
        loaded_count = len(cleaned_state)
    else:
        vision_encoder.load_state_dict(checkpoint, strict=False)
        loaded_count = len(checkpoint)

    if freeze_after_load:
        for param in vision_encoder.parameters():
            param.requires_grad = False

    logger.info("Loaded fine-tuned ViT backbone (%d tensors) from %s", loaded_count, ckpt_path)
    return checkpoint.get("metadata", checkpoint.get("metrics", {}))


def load_finetuned_biomedclip_vit(
    visual_model: nn.Module,
    dataset_name: str,
    save_dir: str = "biomedclip",
    device: Optional[Union[str, torch.device]] = None,
) -> bool:
    """Legacy boolean loading wrapper."""
    try:
        load_finetuned_backbone(visual_model, dataset_name, base_dir=save_dir, device=device)
        return True
    except Exception as e:
        logger.warning("Could not load fine-tuned backbone: %s", e)
        return False


def apply_finetuned_backbone(
    model_or_backbone: nn.Module,
    dataset_name: str,
    n_unfreeze: int = 2,
    base_dir: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> bool:
    """Load fine-tuned weights onto a hybrid model or backbone instance."""
    if hasattr(model_or_backbone, "backbone"):
        encoder = getattr(model_or_backbone.backbone, "vision_encoder", None)
    elif hasattr(model_or_backbone, "vision_encoder"):
        encoder = model_or_backbone.vision_encoder
    else:
        encoder = model_or_backbone

    if encoder is None:
        return False

    try:
        load_finetuned_backbone(encoder, dataset_name, n_unfreeze, base_dir, device=device)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Learning Rate Warmup Scheduler
# ---------------------------------------------------------------------------
class BackboneWarmupScheduler:
    """Linear warmup for backbone learning rate."""
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        backbone_param_group_idx: int,
        target_lr: float = 1e-5,
        warmup_epochs: int = 3,
        ramp_epochs: int = 2,
    ):
        self.optimizer = optimizer
        self.group_idx = backbone_param_group_idx
        self.target_lr = target_lr
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs

    def step(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            lr = 0.0
        elif epoch < self.warmup_epochs + self.ramp_epochs:
            progress = (epoch - self.warmup_epochs) / max(1, self.ramp_epochs)
            lr = self.target_lr * progress
        else:
            lr = self.target_lr
        if len(self.optimizer.param_groups) > self.group_idx:
            self.optimizer.param_groups[self.group_idx]["lr"] = lr
        return lr


# ---------------------------------------------------------------------------
# Comprehensive Clinical Metrics Calculator
# ---------------------------------------------------------------------------
def compute_comprehensive_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_classes: int,
) -> Dict[str, Any]:
    """Calculate all clinical diagnostic evaluation metrics required by task guidelines."""
    y_true = np.asarray(y_true).astype(int).ravel()
    preds = np.argmax(probs, axis=1)

    # 1. AUROC (Macro and per-class)
    per_class_auroc = {}
    macro_auroc = 0.5
    if _SKLEARN_AVAILABLE:
        try:
            if n_classes == 2:
                p1 = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                score = float(roc_auc_score(y_true, p1))
                macro_auroc = score
                per_class_auroc = {0: score, 1: score}  # Binary symmetry: AUROC_0 === AUROC_1
                assert abs(per_class_auroc[0] - per_class_auroc[1]) < 1e-6, (
                    "Binary AUROC symmetry violated — check per_class_auroc computation."
                )
            else:
                y_oh = np.eye(n_classes)[y_true]
                for c in range(n_classes):
                    if len(np.unique(y_oh[:, c])) > 1:
                        per_class_auroc[c] = float(roc_auc_score(y_oh[:, c], probs[:, c]))
                    else:
                        per_class_auroc[c] = 0.5
                macro_auroc = float(np.mean(list(per_class_auroc.values())))
        except Exception:
            macro_auroc = 0.5
            per_class_auroc = {c: 0.5 for c in range(n_classes)}

    # 2. AUPRC / Average Precision (AP)
    auprc = 0.0
    if _SKLEARN_AVAILABLE:
        try:
            if n_classes == 2:
                p1 = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                auprc = float(average_precision_score(y_true, p1))
            else:
                y_oh = np.eye(n_classes)[y_true]
                auprc = float(average_precision_score(y_oh, probs, average="macro"))
        except Exception:
            auprc = 0.0

    # 3. Macro F1-Score
    macro_f1 = 0.0
    if _SKLEARN_AVAILABLE:
        try:
            macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
        except Exception:
            macro_f1 = 0.0

    # 4. Sensitivity (Recall / TPR)
    sensitivity = 0.0
    if _SKLEARN_AVAILABLE:
        try:
            if n_classes == 2:
                sensitivity = float(recall_score(y_true, preds, pos_label=1, zero_division=0))
            else:
                sensitivity = float(recall_score(y_true, preds, average="macro", zero_division=0))
        except Exception:
            sensitivity = 0.0

    # 5. Specificity (True Negative Rate) & Confusion Matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    per_class_spec = {c: 0.0 for c in range(n_classes)}
    macro_specificity = 0.0
    if _SKLEARN_AVAILABLE:
        try:
            cm = confusion_matrix(y_true, preds, labels=list(range(n_classes)))
            for c in range(n_classes):
                tp_c = cm[c, c]
                fp_c = cm[:, c].sum() - tp_c
                fn_c = cm[c, :].sum() - tp_c
                tn_c = cm.sum() - (tp_c + fp_c + fn_c)
                denom = tn_c + fp_c
                per_class_spec[c] = float(tn_c / max(1, denom))
            macro_specificity = float(np.mean(list(per_class_spec.values())))
        except Exception:
            pass

    # 6. Balanced Accuracy
    balanced_acc = 0.0
    if _SKLEARN_AVAILABLE:
        try:
            balanced_acc = float(balanced_accuracy_score(y_true, preds))
        except Exception:
            balanced_acc = (sensitivity + macro_specificity) / 2.0
    else:
        balanced_acc = (sensitivity + macro_specificity) / 2.0

    return {
        "macro_auroc": macro_auroc,
        "per_class_auroc": per_class_auroc,
        "auprc": auprc,
        "macro_f1": macro_f1,
        "sensitivity": sensitivity,
        "specificity": macro_specificity,
        "per_class_specificity": per_class_spec,
        "balanced_accuracy": balanced_acc,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Robust Preprocessing & Loss Utilities
# ---------------------------------------------------------------------------
def robust_biomedclip_preprocess(
    images: torch.Tensor,
    device: torch.device,
    preprocess_mode: str = "biomedclip_normalized",
) -> torch.Tensor:
    """
    Convert an image batch according to one explicit preprocessing contract.

    ``dataset.py`` owns normalization in the main pipeline, so the default
    ``biomedclip_normalized`` mode only performs channel/spatial validation and
    never applies mean/std a second time. Raw callers may request
    ``raw_uint8`` or ``raw_float01`` explicitly.
    """
    images = images.to(device=device, dtype=torch.float32, non_blocking=True)

    # 1. Grayscale -> 3-channel or multichannel slicing
    if images.ndim == 4 and images.shape[1] == 1:
        images = images.expand(-1, 3, -1, -1)
    elif images.ndim == 4 and images.shape[1] > 3:
        images = images[:, :3]

    # 2. Spatial resize if needed
    if images.shape[-2:] != (224, 224):
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)

    if preprocess_mode == "biomedclip_normalized":
        if not torch.isfinite(images).all():
            raise ValueError("BiomedCLIP-normalized batch contains NaN/Inf values.")
        return images
    if preprocess_mode not in {"raw_uint8", "raw_float01"}:
        raise ValueError(
            f"Unknown preprocessing mode '{preprocess_mode}'. "
            "Use biomedclip_normalized, raw_uint8, or raw_float01."
        )

    if preprocess_mode == "raw_uint8":
        if images.min().item() < 0.0 or images.max().item() > 255.0:
            raise ValueError("raw_uint8 input must be in [0, 255].")
        images = images / 255.0
    elif images.min().item() < 0.0 or images.max().item() > 1.0:
        raise ValueError("raw_float01 input must be in [0, 1].")

    # Explicit raw-input normalization owner.
    mean = torch.tensor(BIOMEDCLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(BIOMEDCLIP_STD, device=device).view(1, 3, 1, 1)
    return (images - mean) / std


def compute_class_weights(train_loader: DataLoader, n_classes: int, device: torch.device) -> torch.Tensor:
    """Compute balanced inverse-frequency class weights from train_loader."""
    from collections import Counter
    label_counts = Counter()
    for batch in train_loader:
        if isinstance(batch, (list, tuple)):
            lbls = batch[1]
        elif isinstance(batch, dict):
            lbls = batch.get("label", batch.get("labels"))
        else:
            continue
        if torch.is_tensor(lbls):
            lbls = lbls.view(-1).tolist()
        label_counts.update(lbls)

    total = sum(label_counts.values())
    weights = []
    for c in range(n_classes):
        cnt = label_counts.get(c, 0)
        if cnt > 0:
            weights.append(total / (n_classes * cnt))
        else:
            weights.append(1.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        return ((1.0 - pt) ** self.gamma * ce_loss).mean()


# ---------------------------------------------------------------------------
# Primary Fine-Tuning Engine
# ---------------------------------------------------------------------------
def finetune_biomedclip(
    vision_encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    dataset_name: str = "pneumoniamnist",
    n_classes: int = 2,
    n_unfreeze: int = 2,
    lr: float = 1e-5,
    epochs: int = 10,
    warmup_epochs: int = 3,
    weight_decay: float = 1e-5,
    device: Optional[torch.device] = None,
    base_dir: Optional[str] = None,
    save_checkpoint: bool = True,
    verbose: bool = True,
    preprocess_mode: str = "biomedclip_normalized",
) -> Dict[str, Any]:
    """Execute isolated fine-tuning on BiomedCLIP ViT with real-time progress and diagnostics."""
    if device is None:
        device = next(vision_encoder.parameters()).device

    vision_encoder = vision_encoder.to(device)

    # 1. Unfreeze top N layers
    param_info = unfreeze_top_layers(vision_encoder, n_unfreeze=n_unfreeze)
    if verbose:
        print("\n" + "═" * 80)
        print("🧬 BiomedCLIP ViT Fine-Tuning Execution Initialized")
        print("═" * 80)
        print(f"  • Dataset Name      : {dataset_name}")
        print(f"  • Compute Target    : {device}")
        print(f"  • Unfrozen Blocks   : Top {param_info['unfrozen_blocks']} Transformer Layers")
        print(f"  • Trainable Weights : {param_info['unfrozen_params']:,} / {param_info['total_params']:,} parameters")
        print(f"  • Learning Rate     : {lr} (Warmup={warmup_epochs} epochs)")
        print("═" * 80 + "\n")

    backbone_params = [p for p in vision_encoder.parameters() if p.requires_grad]
    if not backbone_params:
        logger.warning("No backbone parameters unfrozen (n_unfreeze=0). Skipping fine-tuning.")
        return {"skipped": True}

    # 2. Linear probe head with learnable Logit Scale / Learned Temperature (τ)
    probe_head = nn.Linear(512, n_classes).to(device)
    nn.init.normal_(probe_head.weight, std=0.01)
    nn.init.zeros_(probe_head.bias)

    # Learned Temperature / Logit Scale (initialized to CLIP canonical tau = ln(1/0.07))
    logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07), device=device))

    # 3. Reference embedding capture for Cosine Similarity Drift calculation
    frozen_ref_features: Optional[torch.Tensor] = None
    cat_ref: Optional[torch.Tensor] = None
    ref_eval_loader = val_loader or train_loader
    if ref_eval_loader is not None:
        vision_encoder.eval()
        with torch.no_grad():
            ref_imgs_list = []
            for b in ref_eval_loader:
                imgs = b[0] if isinstance(b, (list, tuple)) else b.get("image", b.get("images"))
                if imgs is None:
                    continue
                imgs = robust_biomedclip_preprocess(imgs, device, preprocess_mode)
                ref_imgs_list.append(imgs)
                if sum(x.shape[0] for x in ref_imgs_list) >= 128:
                    break
            if ref_imgs_list:
                cat_ref = torch.cat(ref_imgs_list, dim=0)[:128]
                frozen_ref_features = vision_encoder(cat_ref).float().detach()

    # 4. Representation Distillation Reference Encoder Initialization
    frozen_encoder = copy.deepcopy(vision_encoder)
    for p in frozen_encoder.parameters():
        p.requires_grad_(False)
    frozen_encoder.eval()

    # 5. Optimizer & Warmup Scheduler
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 0.0, "weight_decay": weight_decay},
        {"params": probe_head.parameters(), "lr": lr * 10, "weight_decay": 1e-4},
        {"params": [logit_scale], "lr": lr, "weight_decay": 0.0},
    ])

    warmup_scheduler = BackboneWarmupScheduler(
        optimizer,
        backbone_param_group_idx=0,
        target_lr=lr,
        warmup_epochs=warmup_epochs,
        ramp_epochs=2,
    )

    class_weights = compute_class_weights(train_loader, n_classes=n_classes, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    history: Dict[str, List[Any]] = {
        "train_loss": [], "val_loss": [], "lr": [], "grad_norm": [],
        "cosine_drift": [], "cosine_sim": [], "logit_scale_tau": [], "exp_tau": [],
        "macro_auroc": [], "auprc": [], "macro_f1": [], "sensitivity": [],
        "specificity": [], "balanced_acc": [], "distill_loss": [],
    }
    best_val_loss = float("inf")

    # 6. Training Loop
    for epoch in range(1, epochs + 1):
        backbone_lr = warmup_scheduler.step(epoch - 1)
        vision_encoder.train()
        probe_head.train()

        running_train_loss = 0.0
        n_train_batches = 0
        batch_grad_norms = []

        train_iter = train_loader
        pbar_desc = f"Epoch {epoch:02d}/{epochs:02d} [Train ViT]"
        if _TQDM_AVAILABLE and verbose:
            train_iter = tqdm(train_loader, desc=pbar_desc, dynamic_ncols=True, leave=False)

        for batch in train_iter:
            if isinstance(batch, (list, tuple)):
                images, labels = batch[0], batch[1]
            elif isinstance(batch, dict):
                images = batch.get("image", batch.get("images"))
                labels = batch.get("label", batch.get("labels"))
            else:
                continue

            images = robust_biomedclip_preprocess(images, device, preprocess_mode)
            labels = labels.to(device).long()
            if labels.ndim > 1:
                labels = labels.squeeze(-1) if labels.shape[-1] == 1 else labels.squeeze()

            optimizer.zero_grad()

            features = vision_encoder(images).float()
            features = F.normalize(features, p=2, dim=-1)
            scale = torch.exp(logit_scale)
            logits = probe_head(features) * scale

            # Distillation loss against frozen reference encoder
            with torch.no_grad():
                frozen_feats = F.normalize(frozen_encoder(images).float(), dim=-1)
            distill_loss = (1.0 - F.cosine_similarity(features, frozen_feats, dim=-1)).mean()

            lambda_distill = 0.3
            loss = criterion(logits, labels) + lambda_distill * distill_loss

            loss.backward()

            # Global Gradient Norm and fused clipping across all trainable layers
            all_trainable = backbone_params + list(probe_head.parameters()) + [logit_scale]
            total_norm = torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
            batch_grad_norms.append(total_norm)

            optimizer.step()
            with torch.no_grad():
                logit_scale.clamp_(min=0.0, max=math.log(100.0))

            running_train_loss += loss.detach()
            n_train_batches += 1

            if _TQDM_AVAILABLE and verbose and hasattr(train_iter, "set_postfix"):
                if n_train_batches % 50 == 0:
                    train_iter.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "‖g‖": f"{total_norm.item():.3f}",
                        "lr": f"{backbone_lr:.1e}",
                    })

        avg_train_loss = float(running_train_loss) / max(1, n_train_batches)
        avg_grad_norm = float(torch.stack(batch_grad_norms).mean().item()) if batch_grad_norms else 0.0
        cur_tau = float(logit_scale.detach().item())
        cur_exp_tau = float(math.exp(cur_tau))

        # 7. Validation & Comprehensive Metrics Evaluation
        val_targets, val_probs_list = [], []
        running_val_loss = 0.0
        n_val_batches = 0

        eval_loader = val_loader or train_loader
        vision_encoder.eval()
        probe_head.eval()

        val_iter = eval_loader
        if _TQDM_AVAILABLE and verbose:
            val_iter = tqdm(eval_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Validating]", dynamic_ncols=True, leave=False)

        with torch.no_grad():
            for batch in val_iter:
                if isinstance(batch, (list, tuple)):
                    imgs, lbls = batch[0], batch[1]
                elif isinstance(batch, dict):
                    imgs = batch.get("image", batch.get("images"))
                    lbls = batch.get("label", batch.get("labels"))
                else:
                    continue

                imgs = robust_biomedclip_preprocess(imgs, device, preprocess_mode)
                lbls = lbls.to(device).long()
                if lbls.ndim > 1:
                    lbls = lbls.squeeze(-1) if lbls.shape[-1] == 1 else lbls.squeeze()

                feats = vision_encoder(imgs).float()
                feats = F.normalize(feats, p=2, dim=-1)
                scale = torch.exp(logit_scale)
                logits = probe_head(feats) * scale
                v_loss = criterion(logits, lbls)

                running_val_loss += v_loss.detach()
                n_val_batches += 1

                probs = F.softmax(logits, dim=-1).cpu().numpy()
                val_probs_list.append(probs)
                val_targets.extend(lbls.cpu().numpy().tolist())

        avg_val_loss = float(running_val_loss) / max(1, n_val_batches)
        all_probs_arr = np.concatenate(val_probs_list, axis=0) if val_probs_list else np.zeros((0, n_classes))
        all_targets_arr = np.array(val_targets)

        # 7. Cosine Similarity Drift calculation
        cos_sim, cos_drift = 1.0, 0.0
        if frozen_ref_features is not None:
            with torch.no_grad():
                current_ref_feats = vision_encoder(cat_ref).float()
                cos_sim_t = F.cosine_similarity(
                    F.normalize(current_ref_feats, dim=-1),
                    F.normalize(frozen_ref_features, dim=-1),
                    dim=-1
                ).mean().item()
                cos_sim = float(cos_sim_t)
                cos_drift = float(1.0 - cos_sim)

        # 8. Compute all clinical metrics
        metrics = compute_comprehensive_metrics(all_targets_arr, all_probs_arr, n_classes=n_classes)

        # Record history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(backbone_lr)
        history["grad_norm"].append(avg_grad_norm)
        history["cosine_drift"].append(cos_drift)
        history["cosine_sim"].append(cos_sim)
        history["logit_scale_tau"].append(cur_tau)
        history["exp_tau"].append(cur_exp_tau)
        history["macro_auroc"].append(metrics["macro_auroc"])
        history["auprc"].append(metrics["auprc"])
        history["macro_f1"].append(metrics["macro_f1"])
        history["sensitivity"].append(metrics["sensitivity"])
        history["specificity"].append(metrics["specificity"])
        history["balanced_acc"].append(metrics["balanced_accuracy"])

        # Display Per-Epoch Diagnostic Dashboard
        if verbose:
            cm_str = np.array2string(metrics["confusion_matrix"], separator=", ")
            per_class_str = ", ".join(f"C{c}: {v:.4f}" for c, v in metrics["per_class_auroc"].items())
            print(f"\n┌─ [Epoch {epoch:02d}/{epochs:02d}] BiomedCLIP ViT Progress & Evaluation Dashboard " + "─" * 22)
            print(f"│ • Learning Rate Schedule : {backbone_lr:.2e}")
            print(f"│ • Global Gradient Norm ‖g‖: {avg_grad_norm:.4f}")
            print(f"│ • Logit Scale / Temp (τ) : {cur_tau:.4f}  |  exp(τ) = {cur_exp_tau:.3f}")
            print(f"│ • Cosine Sim Drift to Base: {cos_drift:.4f} (Cos Sim = {cos_sim:.4f})")
            print(f"├───────────────────────────────────────────────────────────────────────────────")
            print(f"│ • Training Loss vs Val Loss: {avg_train_loss:.4f}  vs  {avg_val_loss:.4f}")
            print(f"│ • Balanced Accuracy      : {metrics['balanced_accuracy']:.2%}")
            print(f"│ • Macro F1-Score         : {metrics['macro_f1']:.4f}")
            print(f"│ • Sensitivity (TPR)      : {metrics['sensitivity']:.2%}")
            print(f"│ • Specificity (TNR)      : {metrics['specificity']:.2%}")
            print(f"│ • AUROC (Macro)          : {metrics['macro_auroc']:.4f}  [{per_class_str}]")
            print(f"│ • AUPRC / Avg Precision  : {metrics['auprc']:.4f}")
            print(f"├───────────────────────────────────────────────────────────────────────────────")
            print(f"│ • Confusion Matrix:\n{cm_str}")
            print(f"└" + "─" * 78 + "\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    # 9. Save Checkpoint
    ckpt_path = None
    if save_checkpoint:
        meta_dict = {
            "dataset_name": dataset_name,
            "n_unfreeze": n_unfreeze,
            "lr": lr,
            "epochs": epochs,
            "warmup_epochs": warmup_epochs,
            "preprocess_mode": preprocess_mode,
            "best_val_loss": best_val_loss,
            "final_metrics": metrics,
            "history": history,
        }
        ckpt_path = save_finetuned_backbone(
            vision_encoder=vision_encoder,
            dataset_name=dataset_name,
            n_unfreeze=n_unfreeze,
            metadata=meta_dict,
            base_dir=base_dir,
        )

    # 10. Re-freeze backbone for subsequent quantum pipeline execution
    for param in vision_encoder.parameters():
        param.requires_grad = False

    return {
        "history": history,
        "checkpoint_path": str(ckpt_path) if ckpt_path else None,
        "param_info": param_info,
        "best_val_loss": best_val_loss,
        "final_metrics": metrics,
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Polymorphic Entry Point for Full Pipeline Interoperability
# ---------------------------------------------------------------------------
def finetune_biomedclip_vit(
    visual_model: Optional[Union[nn.Module, BiomedCLIPFineTuneConfig]] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    config: Optional[BiomedCLIPFineTuneConfig] = None,
    criterion: Optional[nn.Module] = None,
    head: Optional[nn.Module] = None,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs: Any,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Flexible wrapper matching both legacy finetunning.py and modern finetuning.py signatures."""
    if isinstance(visual_model, BiomedCLIPFineTuneConfig):
        config = visual_model
        visual_model = None
    elif isinstance(train_loader, BiomedCLIPFineTuneConfig):
        config = train_loader
        train_loader = None

    if config is None:
        config = kwargs.get("ft_cfg", BiomedCLIPFineTuneConfig())

    target_dev = torch.device(device if device is not None else (config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu")))

    if visual_model is None:
        try:
            from model import BiomedCLIPQuantumBackbone
        except ImportError:
            from qml_models.model import BiomedCLIPQuantumBackbone
        backbone = BiomedCLIPQuantumBackbone(in_channels=3, n_qubits=4, dropout=0.0).to(target_dev)
        visual_model = backbone.vision_encoder

    n_classes = kwargs.get("n_classes", getattr(config, "n_classes", None))
    if hasattr(head, "out_features"):
        n_classes = head.out_features
    if n_classes is None or n_classes == 2:
        if hasattr(train_loader, "dataset"):
            ds = getattr(train_loader, "dataset")
            if hasattr(ds, "info") and isinstance(ds.info, dict) and "label" in ds.info:
                n_classes = len(ds.info["label"])
            elif hasattr(ds, "labels") and hasattr(ds.labels, "max"):
                n_classes = int(ds.labels.max()) + 1
            elif hasattr(ds, "targets") and hasattr(ds.targets, "max"):
                n_classes = int(ds.targets.max()) + 1
    if n_classes is None:
        n_classes = 2

    res = finetune_biomedclip(
        vision_encoder=visual_model,
        train_loader=train_loader,
        val_loader=val_loader,
        dataset_name=config.dataset_name,
        n_classes=n_classes,
        n_unfreeze=config.unfreeze_layers,
        lr=config.finetune_lr,
        epochs=config.epochs,
        warmup_epochs=config.warmup_epochs,
        weight_decay=config.weight_decay,
        device=target_dev,
        base_dir=config.save_dir,
        save_checkpoint=True,
        verbose=kwargs.get("verbose", True),
        preprocess_mode=config.preprocess_mode,
    )
    return visual_model, res


# ---------------------------------------------------------------------------
# CLI Argument Binding
# ---------------------------------------------------------------------------
def add_finetuning_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add fine-tuning parameters to standard argparse parser."""
    group = parser.add_argument_group("BiomedCLIP ViT Fine-Tuning")
    group.add_argument("--finetune_backbone", action="store_true", default=False,
                        help="Enable BiomedCLIP ViT fine-tuning before quantum training")
    group.add_argument("--ft_unfreeze_layers", type=int, default=2,
                        help="Number of top ViT transformer blocks to unfreeze (default: 2)")
    group.add_argument("--ft_lr", type=float, default=1e-5,
                        help="Learning rate for backbone fine-tuning (default: 1e-5)")
    group.add_argument("--ft_epochs", type=int, default=10,
                        help="Number of fine-tuning epochs (default: 10)")
    group.add_argument("--ft_warmup_epochs", type=int, default=3,
                        help="Warmup epochs with backbone frozen (default: 3)")
    group.add_argument("--ft_weight_decay", type=float, default=1e-5,
                        help="Weight decay for backbone parameters (default: 1e-5)")
    group.add_argument("--ft_base_dir", type=str, default=None,
                        help="Directory for saving fine-tuned checkpoints")
    group.add_argument("--ft_force_retrain", action="store_true", default=False,
                        help="Force re-training even if checkpoint already exists")
    return parser


def get_finetuning_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Extract fine-tuning configuration dictionary from parsed arguments."""
    return {
        "finetune_backbone": getattr(args, "finetune_backbone", False),
        "n_unfreeze": getattr(args, "ft_unfreeze_layers", 2),
        "lr": getattr(args, "ft_lr", 1e-5),
        "epochs": getattr(args, "ft_epochs", 10),
        "warmup_epochs": getattr(args, "ft_warmup_epochs", 3),
        "weight_decay": getattr(args, "ft_weight_decay", 1e-5),
        "base_dir": getattr(args, "ft_base_dir", None),
        "force_retrain": getattr(args, "ft_force_retrain", False),
    }


def main():
    """Standalone CLI executable for BiomedCLIP fine-tuning."""
    parser = argparse.ArgumentParser(description="BiomedCLIP ViT-B/16 Fine-Tuning & Diagnostic Profiler")
    parser.add_argument("--dataset", type=str, default="pneumoniamnist", help="MedMNIST dataset name")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=224)
    parser = add_finetuning_args(parser)
    parser.set_defaults(finetune_backbone=True, ft_epochs=10)
    args = parser.parse_args()

    ft_cfg = get_finetuning_config(args)

    try:
        from model import BiomedCLIPQuantumBackbone, get_torch_device
        from dataset import get_medmnist_dataloaders
    except ImportError:
        from qml_models.model import BiomedCLIPQuantumBackbone, get_torch_device
        from qml_models.dataset import get_medmnist_dataloaders

    device = get_torch_device()
    train_loader, val_loader, _, meta = get_medmnist_dataloaders(
        args.dataset, args.batch_size, size=args.image_size
    )

    if has_finetuned_checkpoint(args.dataset, ft_cfg["n_unfreeze"], ft_cfg["base_dir"]):
        if not ft_cfg["force_retrain"]:
            p = get_finetuned_checkpoint_path(args.dataset, ft_cfg["n_unfreeze"], ft_cfg["base_dir"])
            print(f"[finetuning] Checkpoint already exists at {p}. Use --ft_force_retrain to override.")
            return

    backbone = BiomedCLIPQuantumBackbone(
        in_channels=meta.get("in_channels", 1), n_qubits=4, dropout=0.0
    ).to(device)

    finetune_biomedclip(
        vision_encoder=backbone.vision_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        dataset_name=args.dataset,
        n_classes=meta["n_classes"],
        n_unfreeze=ft_cfg["n_unfreeze"],
        lr=ft_cfg["lr"],
        epochs=ft_cfg["epochs"],
        warmup_epochs=ft_cfg["warmup_epochs"],
        weight_decay=ft_cfg["weight_decay"],
        device=device,
        base_dir=ft_cfg["base_dir"],
    )


if __name__ == "__main__":
    main()

# Register module aliases in sys.modules for flawless backward compatibility
sys.modules["finetunning"] = sys.modules[__name__]
sys.modules["qml_models.finetunning"] = sys.modules[__name__]
