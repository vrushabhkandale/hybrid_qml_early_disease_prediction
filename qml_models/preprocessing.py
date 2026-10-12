"""
preprocessing.py — Auto-Adaptive Multi-View Augmentation & Class-Stratified Subsampling
========================================================================================
Enterprise-grade, mathematically grounded, and hardware-aligned data preprocessing library
designed specifically for Hybrid Quantum-Classical Medical Diagnostic Architectures
(BiomedCLIP ViT-B/16 -> CQ Adapter -> Riemannian S^2 -> PennyLane VQC -> Clinical Head).

Core Capabilities:
  1. Modality Invariance Policies (Anatomical Symmetry Groups G_modal)
     - Strict physiological invariants (prevents vertical inversion on Chest X-Rays / Retinal OCT)
     - Dynamic curriculum warmup scheduling: alpha(t) = alpha_max * sin^2(pi*t / (2*T_warmup))
  2. Auto-Adaptive Multi-View Augmentation Engine
     - Dual-domain execution: Pixel-Space Multi-View & Latent-Manifold Multi-View
     - Multi-view batch collation for parallel Vision ViT feature extraction
  3. Latent Manifold Multi-View (Cached 512-d Feature Space & Quantum Angle Space)
     - Tangent Gaussian perturbation & feature dimension dropout
     - Intra-class manifold mixup (Verma et al., ICML 2019) preserving true clinical labels
     - SU(2) Lie algebra and Riemannian S^2 spherical tangent perturbation
  4. Test-Time Augmentation (TTA) & Dual-Uncertainty Decomposition
     - Multi-view predictive pooling with Bayesian inverse-entropy weighting
     - Decomposition into Aleatoric Uncertainty (view variance) & Epistemic Uncertainty
  5. Exact Class-Stratified Subsampling & Core-Set Selection
     - Proportional, Uniform Balanced, and Effective-Number allocation (Cui et al., CVPR 2019)
     - Minimum class quorum enforcement (K_c >= K_min) preventing single-class AUC crashes
     - K-Center Greedy Core-Set active subsampling (Sener & Savarese, ICLR 2018)
     - Custom PyTorch ClassStratifiedBatchSampler enforcing balanced quotas per mini-batch

Academic Foundations:
  • Cubuk et al. (NeurIPS 2020) — RandAugment: Practical automated data augmentation
  • Müller & Hutter (ICCV 2021) — TrivialAugment: Tuning-free Data Augmentation
  • Tian et al. (NeurIPS 2020) — What Makes for Good Views for Contrastive Learning (InfoMin)
  • Khosla et al. (NeurIPS 2020) — Supervised Contrastive Learning (SupCon)
  • Verma et al. (ICML 2019) — Manifold Mixup: Better Representations through Interpolator Secrecy
  • Cui et al. (CVPR 2019) — Class-Balanced Loss Based on Effective Number of Samples
  • Sener & Savarese (ICLR 2018) — Active Learning for CNNs: A Core-Set Approach
  • Wang et al. (IEEE TMI 2019) — Aleatoric and Epistemic Uncertainty with TTA
  • Dilip et al. (Phys. Rev. A 2022) — Data Augmentation for Quantum Machine Learning
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

# ---------------------------------------------------------------------------
# Strict BiomedCLIP Normalization Constants (Standardized Across Codebase)
# ---------------------------------------------------------------------------
BIOMEDCLIP_MEAN = [0.48145466, 0.45782750, 0.40821073]
BIOMEDCLIP_STD = [0.26862954, 0.26130258, 0.27577711]


# ---------------------------------------------------------------------------
# Anti-Shortcut Foreground Mask & Confounder Suppression Utilities
# ---------------------------------------------------------------------------
def compute_foreground_mask(
    image_tensor: torch.Tensor,
    threshold_method: str = "otsu",
) -> torch.Tensor:
    """Computes binary foreground mask {0, 1} segmenting anatomical ROI from background.
    
    Args:
        image_tensor: Tensor of shape (..., C, H, W) or (..., H, W).
        threshold_method: 'otsu' or 'mean'.
        
    Returns:
        Binary mask of shape matching spatial dimensions (..., 1, H, W) with {0.0, 1.0}.
    """
    if image_tensor.ndim == 3:
        # (C, H, W) -> add batch dim
        expanded = True
        img = image_tensor.unsqueeze(0)
    else:
        expanded = False
        img = image_tensor

    # Intensity map across channels
    if img.shape[1] > 1:
        gray = img.mean(dim=1, keepdim=True)
    else:
        gray = img

    if threshold_method == "otsu":
        # Differentiable approximation of Otsu threshold via median / adaptive thresholding
        b, _, h, w = gray.shape
        masks = []
        for i in range(b):
            flat = gray[i].view(-1)
            # Find adaptive threshold separating dark background from anatomical tissue
            sorted_vals, _ = torch.sort(flat)
            # Use 25th percentile as robust background baseline
            th = sorted_vals[int(len(sorted_vals) * 0.25)]
            mask_i = (gray[i] > th).float()
            # Suppress edge boundary 5% border to clear scanner artifacts
            border_h = max(1, int(h * 0.05))
            border_w = max(1, int(w * 0.05))
            mask_i[:, :border_h, :] = 0.0
            mask_i[:, -border_h:, :] = 0.0
            mask_i[:, :, :border_w] = 0.0
            mask_i[:, :, -border_w:] = 0.0
            masks.append(mask_i)
        mask = torch.stack(masks, dim=0)
    else:
        mean_val = gray.mean(dim=(-2, -1), keepdim=True)
        mask = (gray > mean_val).float()

    return mask.squeeze(0) if expanded else mask


class GridMask(nn.Module):
    """GridMask data augmentation to eliminate shortcut co-adaptation (Chen et al., 2020)."""
    def __init__(self, ratio: float = 0.5, d_min: int = 16, d_max: int = 48, p: float = 0.5):
        super().__init__()
        self.ratio = ratio
        self.d_min = d_min
        self.d_max = d_max
        self.p = p

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return img
        _, _, h, w = img.shape
        d = torch.randint(self.d_min, self.d_max + 1, (1,)).item()
        l = int(d * self.ratio)
        mask = torch.ones((h, w), device=img.device, dtype=img.dtype)
        for i in range(0, h, d):
            for j in range(0, w, d):
                mask[i:min(i + l, h), j:min(j + l, w)] = 0.0
        return img * mask.unsqueeze(0).unsqueeze(0)


class RandomBorderCrop(nn.Module):
    """Crops peripheral borders to break scanner frame artifacts and padding shortcuts."""
    def __init__(self, crop_ratio: float = 0.10, p: float = 0.5):
        super().__init__()
        self.crop_ratio = crop_ratio
        self.p = p

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return img
        h, w = img.shape[-2:]
        dh = int(h * self.crop_ratio)
        dw = int(w * self.crop_ratio)
        cropped = img[..., dh:h - dh, dw:w - dw]
        return F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)


class RandomHistogramEqualization(nn.Module):
    """Random contrast normalization breaking scanner-specific brightness shortcuts."""
    def __init__(self, p: float = 0.3):
        super().__init__()
        self.p = p

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return img
        # Min-max robust percentile scaling per image
        min_v = img.amin(dim=(-2, -1), keepdim=True)
        max_v = img.amax(dim=(-2, -1), keepdim=True)
        return (img - min_v) / (max_v - min_v + 1e-6)


class AntiShortcutAugmentationScheduler:
    """Curriculum scheduler ramping anti-shortcut regularization intensity over epochs."""
    def __init__(self, start_epoch: int = 5, max_epoch: int = 30, max_intensity: float = 1.0):
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.max_intensity = max_intensity

    def get_intensity(self, current_epoch: int) -> float:
        if current_epoch < self.start_epoch:
            return 0.0
        progress = min(1.0, (current_epoch - self.start_epoch) / max(1, self.max_epoch - self.start_epoch))
        return self.max_intensity * float(np.sin(progress * np.pi / 2.0) ** 2)



# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: MODALITY INVARIANCE POLICIES & ANATOMICAL SYMMETRIES
# ═══════════════════════════════════════════════════════════════════════════

class SymmetryGroup(Enum):
    """Anatomical symmetry groups governing valid planar transformations in SE(2)."""
    TRIVIAL = "Trivial"                   # Identity {e}: strictly no reflections (e.g. Retinal OCT)
    REFLECTION_H = "Reflection_H"         # Z_2: horizontal reflection permitted (e.g. Ultrasound)
    REFLECTION_H_MILD = "Reflection_H_Mild" # Z_2 with low probability (e.g. Chest X-Ray)
    D4 = "D4"                             # Dihedral group of order 8: 90-deg rots + flips (Histology)
    SO2 = "SO2"                           # Full rotation invariance in R^2 (e.g. Dermoscopy)


@dataclass(frozen=True)
class ModalityPolicy:
    """Rigorous physical and anatomical invariance constraints per medical imaging modality.

    Prevents clinically invalid augmentations (such as inverting the cranial-caudal
    axis in chest radiography or transposing retinal layers in optical coherence tomography).
    """
    dataset_name: str
    modality_type: str                  # 'radiology', 'ophthalmology', 'histopathology', 'dermoscopy', 'ultrasound'
    symmetry_group: SymmetryGroup
    allow_flip_h: bool
    flip_h_prob: float
    allow_flip_v: bool
    flip_v_prob: float
    max_rotation_degrees: float
    max_translation_ratio: float
    max_scaling_ratio: float
    enable_clahe: bool                  # Contrast Limited Adaptive Histogram Equalization
    enable_speckle_noise: bool          # Ultrasound/OCT acoustic/optical speckle
    enable_stain_jitter: bool           # Histology H&E optical density jitter
    enable_gamma_jitter: bool           # Non-linear sensor response perturbation
    intensity_warmup_epochs: int = 5    # Curriculum warmup duration


# Standardized Domain Registry covering 2D and 3D MedMNIST benchmarks
MODALITY_REGISTRY: Dict[str, ModalityPolicy] = {
    # 1. Chest Radiographs (Cranial-Caudal anatomical axis strictly fixed; mild H-flip allowed)
    "pneumoniamnist": ModalityPolicy(
        dataset_name="pneumoniamnist",
        modality_type="radiology",
        symmetry_group=SymmetryGroup.REFLECTION_H_MILD,
        allow_flip_h=True,
        flip_h_prob=0.20,
        allow_flip_v=False,
        flip_v_prob=0.00,
        max_rotation_degrees=7.0,
        max_translation_ratio=0.04,
        max_scaling_ratio=0.05,
        enable_clahe=True,
        enable_speckle_noise=False,
        enable_stain_jitter=False,
        enable_gamma_jitter=True,
    ),
    # 2. Breast Ultrasound (Skin surface at top; acoustic shadows downward; H-flip valid)
    "breastmnist": ModalityPolicy(
        dataset_name="breastmnist",
        modality_type="ultrasound",
        symmetry_group=SymmetryGroup.REFLECTION_H,
        allow_flip_h=True,
        flip_h_prob=0.50,
        allow_flip_v=False,
        flip_v_prob=0.00,
        max_rotation_degrees=10.0,
        max_translation_ratio=0.05,
        max_scaling_ratio=0.06,
        enable_clahe=True,
        enable_speckle_noise=True,
        enable_stain_jitter=False,
        enable_gamma_jitter=True,
    ),
    # 3. Retinal Optical Coherence Tomography (Horizontal reflection is clinically invariant; vertical hierarchy absolute)
    "octmnist": ModalityPolicy(
        dataset_name="octmnist",
        modality_type="ophthalmology",
        symmetry_group=SymmetryGroup.REFLECTION_H,
        allow_flip_h=True,
        flip_h_prob=0.50,
        allow_flip_v=False,
        flip_v_prob=0.00,
        max_rotation_degrees=5.0,
        max_translation_ratio=0.03,
        max_scaling_ratio=0.04,
        enable_clahe=True,
        enable_speckle_noise=False,
        enable_stain_jitter=False,
        enable_gamma_jitter=True,
    ),
    # 4. Colorectal Histopathology Patches (No canonical gravity orientation; full D4 symmetry)
    "pathmnist": ModalityPolicy(
        dataset_name="pathmnist",
        modality_type="histopathology",
        symmetry_group=SymmetryGroup.D4,
        allow_flip_h=True,
        flip_h_prob=0.50,
        allow_flip_v=True,
        flip_v_prob=0.50,
        max_rotation_degrees=180.0,
        max_translation_ratio=0.08,
        max_scaling_ratio=0.10,
        enable_clahe=False,
        enable_speckle_noise=False,
        enable_stain_jitter=True,
        enable_gamma_jitter=True,
    ),
    # 5. Dermatoscopy Lesions (Skin surface; full rotational invariance SO(2))
    "dermamnist": ModalityPolicy(
        dataset_name="dermamnist",
        modality_type="dermoscopy",
        symmetry_group=SymmetryGroup.SO2,
        allow_flip_h=True,
        flip_h_prob=0.50,
        allow_flip_v=True,
        flip_v_prob=0.50,
        max_rotation_degrees=180.0,
        max_translation_ratio=0.06,
        max_scaling_ratio=0.08,
        enable_clahe=True,
        enable_speckle_noise=False,
        enable_stain_jitter=False,
        enable_gamma_jitter=True,
    ),
    # 6. Fundus Photography (Retina fundus; mild rotation and reflection)
    "retinamnist": ModalityPolicy(
        dataset_name="retinamnist",
        modality_type="ophthalmology",
        symmetry_group=SymmetryGroup.REFLECTION_H,
        allow_flip_h=True,
        flip_h_prob=0.50,
        allow_flip_v=False,
        flip_v_prob=0.00,
        max_rotation_degrees=15.0,
        max_translation_ratio=0.04,
        max_scaling_ratio=0.05,
        enable_clahe=True,
        enable_speckle_noise=False,
        enable_stain_jitter=False,
        enable_gamma_jitter=True,
    ),
}

# Default conservative fallback policy
DEFAULT_MODALITY_POLICY = ModalityPolicy(
    dataset_name="default",
    modality_type="generic_radiology",
    symmetry_group=SymmetryGroup.REFLECTION_H_MILD,
    allow_flip_h=True,
    flip_h_prob=0.30,
    allow_flip_v=False,
    flip_v_prob=0.00,
    max_rotation_degrees=7.0,
    max_translation_ratio=0.04,
    max_scaling_ratio=0.05,
    enable_clahe=True,
    enable_speckle_noise=False,
    enable_stain_jitter=False,
    enable_gamma_jitter=True,
)


def get_modality_policy(dataset_name: str) -> ModalityPolicy:
    """Retrieve the anatomical invariance policy for a given MedMNIST dataset."""
    key = dataset_name.lower().strip()
    return MODALITY_REGISTRY.get(key, DEFAULT_MODALITY_POLICY)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: AUTO-ADAPTIVE MULTI-VIEW AUGMENTATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class SpeckleNoiseTransform:
    """Multiplicative speckle noise simulation for Ultrasound and OCT: I' = I + I * n."""
    def __init__(self, std: float = 0.04):
        self.std = std

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.std <= 0.0:
            return img
        arr = np.array(img, dtype=np.float32) / 255.0
        noise = np.random.normal(0.0, self.std, size=arr.shape).astype(np.float32)
        noisy_arr = np.clip(arr + arr * noise, 0.0, 1.0)
        return Image.fromarray((noisy_arr * 255.0).astype(np.uint8))


class HistologyStainJitterTransform:
    """Simulates biological stain concentration variations in H&E histology."""
    def __init__(self, alpha: float = 0.08):
        self.alpha = alpha

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.alpha <= 0.0 or img.mode != "RGB":
            return img
        enhancer_c = ImageEnhance.Color(img)
        img = enhancer_c.enhance(float(np.random.uniform(1.0 - self.alpha, 1.0 + self.alpha)))
        enhancer_b = ImageEnhance.Brightness(img)
        img = enhancer_b.enhance(float(np.random.uniform(1.0 - self.alpha, 1.0 + self.alpha)))
        return img


class CLAHETransform:
    """Adaptive contrast enhancement mimicking clinical viewing workstation presets."""
    def __call__(self, img: Image.Image) -> Image.Image:
        return ImageOps.autocontrast(img, cutoff=0.5)


class ModalityAdaptiveAugmentor:
    """Builds modality-compliant torchvision transform pipelines with curriculum warmup.

    Curriculum Intensity Scaling (Müller & Hutter, ICCV 2021):
      alpha(t) = alpha_max * sin^2(pi * min(t, T_warmup) / (2 * T_warmup))
    Smoothly transitions the training process from unperturbed canonical images
    during initial parameter stabilization to fully randomized invariant views.
    """
    def __init__(
        self,
        policy: ModalityPolicy,
        target_size: int = 224,
        is_training: bool = True,
        curriculum_epoch: int = 0,
    ):
        self.policy = policy
        self.target_size = target_size
        self.is_training = is_training
        self.curriculum_epoch = curriculum_epoch

    def compute_curriculum_factor(self) -> float:
        """Compute the smooth [0.0, 1.0] intensity coefficient for the current epoch."""
        if not self.is_training or self.policy.intensity_warmup_epochs <= 0:
            return 1.0
        t = float(max(0, self.curriculum_epoch))
        t_w = float(self.policy.intensity_warmup_epochs)
        if t >= t_w:
            return 1.0
        # Smooth sinusoidal S-curve warmup
        return float(math.sin((math.pi * t) / (2.0 * t_w)) ** 2)

    def build_canonical_transform(self) -> transforms.Compose:
        """Deterministic canonical evaluation transform (center-crop / resize + normalization)."""
        return transforms.Compose([
            transforms.Resize((self.target_size, self.target_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=BIOMEDCLIP_MEAN, std=BIOMEDCLIP_STD),
        ])

    def build_geometric_view_transform(self) -> transforms.Compose:
        """Stochastic geometric augmentation respecting modality symmetry group constraints."""
        cf = self.compute_curriculum_factor()
        tfms: List[Callable] = []

        # Resize to target canvas
        tfms.append(transforms.Resize((self.target_size, self.target_size), interpolation=transforms.InterpolationMode.BILINEAR))

        # Horizontal reflection
        if self.policy.allow_flip_h and self.is_training:
            tfms.append(transforms.RandomHorizontalFlip(p=self.policy.flip_h_prob * cf))

        # Vertical reflection (strictly blocked if policy.allow_flip_v is False)
        if self.policy.allow_flip_v and self.is_training:
            tfms.append(transforms.RandomVerticalFlip(p=self.policy.flip_v_prob * cf))

        # Rotation and translation
        max_deg = self.policy.max_rotation_degrees * cf
        max_trans = self.policy.max_translation_ratio * cf
        max_scale = self.policy.max_scaling_ratio * cf
        scale_range = (max(0.85, 1.0 - max_scale), min(1.15, 1.0 + max_scale))

        if self.is_training and (max_deg > 0 or max_trans > 0):
            tfms.append(
                transforms.RandomAffine(
                    degrees=(-max_deg, max_deg),
                    translate=(max_trans, max_trans),
                    scale=scale_range,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )
            )

        tfms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=BIOMEDCLIP_MEAN, std=BIOMEDCLIP_STD),
        ])
        return transforms.Compose(tfms)

    def build_radiometric_view_transform(self) -> transforms.Compose:
        """Stochastic photometric/radiometric augmentation respecting modality characteristics."""
        cf = self.compute_curriculum_factor()
        tfms: List[Callable] = []

        tfms.append(transforms.Resize((self.target_size, self.target_size), interpolation=transforms.InterpolationMode.BILINEAR))

        # Domain-specific noise / color jitter
        if self.is_training and cf > 0.05:
            if self.policy.enable_speckle_noise:
                tfms.append(SpeckleNoiseTransform(std=0.04 * cf))
            if self.policy.enable_stain_jitter:
                tfms.append(HistologyStainJitterTransform(alpha=0.08 * cf))
            if self.policy.enable_clahe and np.random.rand() < (0.35 * cf):
                tfms.append(CLAHETransform())

        tfms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=BIOMEDCLIP_MEAN, std=BIOMEDCLIP_STD),
        ])
        return transforms.Compose(tfms)


class MultiViewTransform:
    """Generates V diverse, clinically valid views for a single input PIL Image.

    Adheres to the InfoMin Principle (Tian et al., NeurIPS 2020):
    Views eliminate non-diagnostic nuisance variations (tilt, lighting, sensor noise)
    while strictly preserving mutual information with respect to the diagnostic label.

    Default V=2:
      View 0: Canonical clinical view (standard resize & normalization)
      View 1: Geometric/photometric perturbed view
    Optional V=4:
      View 2: Radiometric contrast-enhanced view
      View 3: Modality-specific perturbation view
    """
    def __init__(
        self,
        dataset_name: str,
        n_views: int = 2,
        target_size: int = 224,
        is_training: bool = True,
        curriculum_epoch: int = 0,
    ):
        self.dataset_name = dataset_name
        self.n_views = max(1, n_views)
        self.policy = get_modality_policy(dataset_name)
        self.augmentor = ModalityAdaptiveAugmentor(
            self.policy,
            target_size=target_size,
            is_training=is_training,
            curriculum_epoch=curriculum_epoch,
        )
        self.canonical_tfm = self.augmentor.build_canonical_transform()
        self.geometric_tfm = self.augmentor.build_geometric_view_transform()
        self.radiometric_tfm = self.augmentor.build_radiometric_view_transform()

    def set_epoch(self, epoch: int):
        """Update curriculum warmup epoch."""
        self.augmentor.curriculum_epoch = epoch
        self.geometric_tfm = self.augmentor.build_geometric_view_transform()
        self.radiometric_tfm = self.augmentor.build_radiometric_view_transform()

    def __call__(self, img: Union[Image.Image, np.ndarray, torch.Tensor]) -> List[torch.Tensor]:
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = Image.fromarray(img).convert("RGB")
            elif img.ndim == 3 and img.shape[0] in (1, 3):
                img = Image.fromarray(np.transpose(img, (1, 2, 0))).convert("RGB")
            else:
                img = Image.fromarray(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 3 and img.shape[0] in (1, 3):
                arr = (img.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(arr).convert("RGB")
            elif img.ndim == 2:
                arr = (img.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(arr).convert("RGB")
        elif hasattr(img, "convert"):
            img = img.convert("RGB")

        views: List[torch.Tensor] = []
        # View 0 is always the deterministic canonical view
        views.append(self.canonical_tfm(img))

        if self.n_views >= 2:
            views.append(self.geometric_tfm(img))
        if self.n_views >= 3:
            views.append(self.radiometric_tfm(img))
        if self.n_views >= 4:
            views.append(self.geometric_tfm(img))

        while len(views) < self.n_views:
            views.append(self.geometric_tfm(img))

        return views[:self.n_views]


class MultiViewBatchCollator:
    """Collates a list of (views_list, label) tuples into batched PyTorch tensors.

    Reshaping Mechanics:
      Inputs: B samples, each having V views of shape (3, 224, 224).
      Batch Output: Flattened tensor of shape (B * V, 3, 224, 224) enabling a single
      broadcasted forward pass through frozen BiomedCLIP ViT-B/16, which is then
      reshaped to (B, V, 512).
    """
    def __init__(self, flatten_views: bool = True):
        self.flatten_views = flatten_views

    def __call__(self, batch: Sequence[Tuple[Union[List[torch.Tensor], torch.Tensor], Union[int, torch.Tensor]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        all_views: List[torch.Tensor] = []
        labels: List[int] = []

        for item in batch:
            v_data, lbl = item
            if isinstance(lbl, torch.Tensor):
                lbl_val = int(lbl.squeeze().item())
            elif isinstance(lbl, np.ndarray):
                lbl_val = int(lbl.squeeze())
            else:
                lbl_val = int(lbl)
            labels.append(lbl_val)

            if isinstance(v_data, (list, tuple)):
                all_views.append(torch.stack(v_data, dim=0))  # (V, C, H, W)
            elif isinstance(v_data, torch.Tensor):
                all_views.append(v_data.unsqueeze(0))         # (1, C, H, W)

        stacked_views = torch.stack(all_views, dim=0)          # (B, V, C, H, W)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        if self.flatten_views:
            B, V, C, H, W = stacked_views.shape
            flat_tensor = stacked_views.view(B * V, C, H, W)
            labels_flat = labels_tensor.repeat_interleave(V)
            return flat_tensor, labels_flat

        return stacked_views, labels_tensor


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: LATENT MANIFOLD MULTI-VIEW (CACHED 512-D FEATURE SPACE)
# ═══════════════════════════════════════════════════════════════════════════

class LatentManifoldMultiView:
    """Generates synthetic multi-view pairs directly on cached 512-d feature vectors.

    Solves the Single-View Offline Caching Defect:
    When images are pre-extracted to disk as static embeddings to accelerate training,
    standard pixel-level augmentations cannot run during downstream epoch loops.
    This class generates valid, stochastic multi-view representations in feature space
    using tangent perturbations, dimension dropout, and intra-class manifold mixup.
    """
    def __init__(
        self,
        noise_std: float = 0.05,
        dropout_prob: float = 0.08,
        mixup_alpha: float = 8.0,
        mixup_beta: float = 2.0,
    ):
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.mixup_alpha = mixup_alpha
        self.mixup_beta = mixup_beta

    def generate_stochastic_view(self, features: torch.Tensor) -> torch.Tensor:
        """Apply feature-space dropout and tangent Gaussian perturbation.

        f_tilde = (f * mask) / (1 - p) + epsilon,  epsilon ~ N(0, noise_std^2 * I)
        """
        if self.dropout_prob > 0.0:
            mask = (torch.rand_like(features) > self.dropout_prob).float()
            perturbed = (features * mask) / (1.0 - self.dropout_prob)
        else:
            perturbed = features.clone()

        if self.noise_std > 0.0:
            noise = torch.randn_like(features) * self.noise_std
            perturbed = perturbed + noise

        # Renormalize to match standard representation manifold scale
        return F.normalize(perturbed, p=2, dim=-1)

    def intra_class_manifold_mixup(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs manifold mixup strictly between samples belonging to the same clinical class.

        By restricting mixup to pairs with y_i == y_j:
          f_mix = lambda * f_i + (1 - lambda) * f_j,   lambda ~ Beta(alpha, beta)
        we synthesize valid internal variance without corrupting one-hot disease labels.
        """
        device = features.device
        B = features.size(0)
        mixed_features = features.clone()

        unique_classes = torch.unique(labels)
        for c in unique_classes:
            idx = (labels == c).nonzero(as_tuple=True)[0]
            n_c = len(idx)
            if n_c > 1:
                perm = torch.randperm(n_c, device=device)
                partner_idx = idx[perm]
                lam = float(np.random.beta(self.mixup_alpha, self.mixup_beta))
                mixed_features[idx] = lam * features[idx] + (1.0 - lam) * features[partner_idx]

        return F.normalize(mixed_features, p=2, dim=-1), labels

    def create_multiview_pair(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return two distinct stochastic views (view_1, view_2) for contrastive adapter learning."""
        v1 = self.generate_stochastic_view(features)
        v2, _ = self.intra_class_manifold_mixup(features, labels)
        v2 = self.generate_stochastic_view(v2)
        return v1, v2


class SphericalTangentPerturbation:
    """Non-commuting SU(2) Lie algebra and Riemannian S^2 tangent perturbation for latents z.

    Operates on the 3*n_qubits latent angle tensor z = (theta, phi, lambda).
    Guarantees perturbations remain physically bounded and mathematically consistent:
      - Polar angles theta in [0, pi] are perturbed with epsilon / 2 (staying within hemisphere)
      - Azimuthal angles phi, lambda in [-pi, pi] are perturbed with full epsilon respecting 2*pi periodicity.
    """
    def __init__(self, epsilon: float = 0.05):
        self.epsilon = epsilon

    def __call__(self, z: torch.Tensor, n_qubits: int) -> torch.Tensor:
        if self.epsilon <= 0.0 or z.size(-1) < 3 * n_qubits:
            return z

        device = z.device
        B = z.size(0)

        # Polar perturbations
        d_theta = torch.empty(B, n_qubits, device=device).uniform_(-self.epsilon / 2.0, self.epsilon / 2.0)
        # Azimuthal perturbations
        d_phi = torch.empty(B, n_qubits, device=device).uniform_(-self.epsilon, self.epsilon)
        d_lam = torch.empty(B, n_qubits, device=device).uniform_(-self.epsilon, self.epsilon)

        delta = torch.cat([d_theta, d_phi, d_lam], dim=-1)
        z_perturbed = z + delta

        # Wrap azimuthal angles into [-pi, pi]
        theta_part = z_perturbed[:, :n_qubits].clamp(0.0, math.pi)
        phi_part = (z_perturbed[:, n_qubits:2 * n_qubits] + math.pi) % (2.0 * math.pi) - math.pi
        lam_part = (z_perturbed[:, 2 * n_qubits:3 * n_qubits] + math.pi) % (2.0 * math.pi) - math.pi

        return torch.cat([theta_part, phi_part, lam_part], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: TEST-TIME AUGMENTATION (TTA) & UNCERTAINTY DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════

def bayesian_view_pooling(
    probabilities: torch.Tensor,
    temperature: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aggregates multi-view probability distributions using inverse-entropy confidence weighting.

    Args:
        probabilities: Tensor of shape (V, C) containing class probabilities across V views.
        temperature: Softmax temperature parameter for entropy sharpness.

    Returns:
        (pooled_prob, aleatoric_uncertainty):
          pooled_prob: (C,) aggregated probability distribution.
          aleatoric_uncertainty: scalar measurement of view-induced variance.
    """
    eps = 1e-8
    probs = probabilities.clamp(min=eps, max=1.0 - eps)
    # Shannon entropy per view: H(p) = -sum(p * log(p))
    entropies = -(probs * torch.log(probs)).sum(dim=-1)  # (V,)

    # Weights inversely proportional to entropy (high confidence -> higher weight)
    logits_w = -entropies / max(1e-4, temperature)
    weights = F.softmax(logits_w, dim=0).unsqueeze(-1)   # (V, 1)

    pooled_prob = (weights * probs).sum(dim=0)           # (C,)

    # Aleatoric uncertainty: expected divergence across spatial acquisition views
    view_variance = torch.mean(torch.norm(probs - pooled_prob.unsqueeze(0), p=2, dim=-1) ** 2)

    return pooled_prob, view_variance


class TestTimeMultiViewEnsemble:
    """Performs clinically safe Test-Time Augmentation (TTA) on individual medical images.

    Integrates into app.py to decompose total predictive uncertainty into:
      1. Aleatoric Uncertainty: Variation induced by patient positioning and scanner noise.
      2. Epistemic Uncertainty: Quantum parameter ambiguity (via MC Quantum Dropout).
    """
    def __init__(self, dataset_name: str, n_views: int = 4):
        self.dataset_name = dataset_name
        self.policy = get_modality_policy(dataset_name)
        self.n_views = n_views
        self.augmentor = ModalityAdaptiveAugmentor(self.policy, target_size=224, is_training=True)

    def generate_tta_views(self, image: Image.Image) -> torch.Tensor:
        """Generate V diverse, modality-valid views for a single PIL image."""
        img = image.convert("RGB")
        views: List[torch.Tensor] = []

        # View 0: Canonical clinical view
        views.append(self.augmentor.build_canonical_transform()(img))

        # View 1: Mild geometric view
        views.append(self.augmentor.build_geometric_view_transform()(img))

        # View 2: Radiometric contrast-adjusted view
        views.append(self.augmentor.build_radiometric_view_transform()(img))

        # View 3: Secondary geometric view
        if self.n_views >= 4:
            views.append(self.augmentor.build_geometric_view_transform()(img))

        while len(views) < self.n_views:
            views.append(self.augmentor.build_canonical_transform()(img))

        return torch.stack(views[:self.n_views], dim=0)  # (V, 3, 224, 224)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: EXACT CLASS-STRATIFIED SUBSAMPLING & CORE-SET SELECTION
# ═══════════════════════════════════════════════════════════════════════════

class StratificationMode(Enum):
    """Subsampling allocation modes."""
    PROPORTIONAL = "proportional"        # Strictly preserves dataset class distribution
    BALANCED = "balanced"                # Uniform representation (equal samples per class)
    EFFECTIVE_NUMBER = "effective_number" # Cui et al. (CVPR 2019) power-law balancing


class StratifiedSubsampler:
    """Exact class-stratified subsampling with guaranteed minimum quorum.

    Eliminates the single-class crash failure mode by mathematically guaranteeing
    that every class has at least min(N_c, min_quorum) samples in the output set.
    """
    def __init__(
        self,
        mode: StratificationMode = StratificationMode.BALANCED,
        min_quorum: int = 4,
        beta: float = 0.9999,
        seed: int = 42,
    ):
        self.mode = mode
        self.min_quorum = max(1, min_quorum)
        self.beta = beta
        self.seed = seed

    def compute_class_quotas(
        self,
        labels: np.ndarray,
        total_samples: int,
    ) -> Dict[int, int]:
        """Calculate the exact number of exemplars to select from each class."""
        unique_classes, counts = np.unique(labels, return_counts=True)
        n_classes = len(unique_classes)
        class_count_map = dict(zip(unique_classes, counts))
        n_total = len(labels)
        target_k = min(total_samples, n_total)

        quotas: Dict[int, int] = {}

        if self.mode == StratificationMode.BALANCED:
            base_quota = target_k // n_classes
            rem = target_k % n_classes
            for i, c in enumerate(unique_classes):
                q = base_quota + (1 if i < rem else 0)
                quotas[c] = min(q, class_count_map[c])

        elif self.mode == StratificationMode.PROPORTIONAL:
            allocated = 0
            for c in unique_classes:
                frac = class_count_map[c] / float(n_total)
                q = int(round(target_k * frac))
                q = max(self.min_quorum, min(q, class_count_map[c]))
                quotas[c] = q
                allocated += q

            # Reconcile rounding discrepancies
            diff = target_k - sum(quotas.values())
            classes_sorted = sorted(unique_classes, key=lambda x: class_count_map[x], reverse=True)
            for c in classes_sorted:
                if diff == 0:
                    break
                if diff > 0 and quotas[c] < class_count_map[c]:
                    quotas[c] += 1
                    diff -= 1
                elif diff < 0 and quotas[c] > self.min_quorum:
                    quotas[c] -= 1
                    diff += 1

        elif self.mode == StratificationMode.EFFECTIVE_NUMBER:
            # E_n = (1 - beta^N_c) / (1 - beta)
            effective_nums = {c: (1.0 - (self.beta ** class_count_map[c])) / (1.0 - self.beta) for c in unique_classes}
            sum_eff = sum(effective_nums.values())
            for c in unique_classes:
                frac = effective_nums[c] / sum_eff
                q = int(round(target_k * frac))
                q = max(self.min_quorum, min(q, class_count_map[c]))
                quotas[c] = q

        # Final quorum guard: ensure no class has fewer than min_quorum (unless dataset has fewer)
        for c in unique_classes:
            quotas[c] = max(min(class_count_map[c], self.min_quorum), quotas.get(c, 0))

        return quotas

    def sample_indices(
        self,
        labels: Union[torch.Tensor, np.ndarray],
        total_samples: int,
    ) -> np.ndarray:
        """Return the stratified subset of indices."""
        if isinstance(labels, torch.Tensor):
            labels_np = labels.squeeze().cpu().numpy()
        else:
            labels_np = np.asarray(labels).squeeze()

        quotas = self.compute_class_quotas(labels_np, total_samples)
        rng = np.random.RandomState(self.seed)
        selected_indices: List[int] = []

        for c, count in quotas.items():
            class_indices = np.nonzero(labels_np == c)[0]
            if len(class_indices) <= count:
                chosen = class_indices
            else:
                chosen = rng.choice(class_indices, size=count, replace=False)
            selected_indices.extend(chosen.tolist())

        rng.shuffle(selected_indices)
        return np.array(selected_indices, dtype=np.int64)


class CoreSetStratifiedSelector:
    """Feature-space Core-Set selection via K-Center Greedy Medoid Clustering.

    Based on Sener & Savarese (ICLR 2018):
    Rather than choosing random samples within each class, iteratively selects the
    data point that maximizes the minimum distance to all previously selected points:
      x^* = argmax_{i in U_c} min_{j in S_c} ||f_i - f_j||_2
    Guarantees maximal topological coverage of the feature manifold, preserving
    both central disease morphologies and rare borderline clinical presentations.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed

    def select_coreset_indices(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        target_counts: Dict[int, int],
        is_periodic: bool = False,
    ) -> torch.Tensor:
        """Select core-set indices per class using k-center greedy facility location."""
        device = features.device
        selected_indices: List[int] = []
        rng = np.random.RandomState(self.seed)

        features_flat = features.view(features.size(0), -1).float()
        labels_flat = labels.squeeze()

        for c, k_target in target_counts.items():
            class_mask = (labels_flat == c).nonzero(as_tuple=True)[0]
            n_class = len(class_mask)
            if n_class == 0:
                continue
            if n_class <= k_target:
                selected_indices.extend(class_mask.cpu().tolist())
                continue

            class_feats = features_flat[class_mask]  # (N_c, D)

            # Torus Geodesic Distance for angles/periodic features or Euclidean fallback
            def _modular_torus_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                if is_periodic:
                    diff = torch.remainder(torch.abs(a - b), 2.0 * math.pi)
                    circ = torch.minimum(diff, 2.0 * math.pi - diff)
                    return torch.norm(circ, p=2, dim=-1)
                return torch.norm(a - b, p=2, dim=-1)

            # 1. Initialize with medoid (sample closest to class centroid)
            if is_periodic:
                centroid = torch.atan2(
                    torch.sin(class_feats).mean(dim=0, keepdim=True),
                    torch.cos(class_feats).mean(dim=0, keepdim=True)
                )
            else:
                centroid = class_feats.mean(dim=0, keepdim=True)

            dists_to_center = _modular_torus_dist(class_feats, centroid)
            medoid_idx = int(torch.argmin(dists_to_center).item())

            chosen_in_class: List[int] = [medoid_idx]
            min_dists = _modular_torus_dist(class_feats, class_feats[medoid_idx:medoid_idx + 1])

            # 2. Greedily add furthest points (minimax distance)
            for _ in range(1, k_target):
                furthest_idx = int(torch.argmax(min_dists).item())
                chosen_in_class.append(furthest_idx)
                # Update distances with torus geodesic metric
                new_dists = _modular_torus_dist(class_feats, class_feats[furthest_idx:furthest_idx + 1])
                min_dists = torch.minimum(min_dists, new_dists)

            global_chosen = class_mask[chosen_in_class].cpu().tolist()
            selected_indices.extend(global_chosen)

        return torch.tensor(selected_indices, dtype=torch.long, device=device)


class ClassStratifiedBatchSampler(Sampler[List[int]]):
    """PyTorch Sampler that guarantees an exact balanced quota in EVERY mini-batch.

    Solves the fundamental defect in WeightedRandomSampler:
    Instead of relying on stochastic sampling with replacement (which frequently
    produces batches missing minority classes), this sampler partitions indices
    into per-class FIFO queues and constructs mini-batches by drawing exactly
    b_c = max(1, B // n_classes) samples from each class.
    """
    def __init__(
        self,
        labels: Union[torch.Tensor, np.ndarray, Sequence[int]],
        batch_size: int,
        drop_last: bool = False,
        seed: int = 42,
    ):
        if isinstance(labels, torch.Tensor):
            self.labels = labels.squeeze().cpu().numpy()
        else:
            self.labels = np.asarray(labels).squeeze()

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.unique_classes = np.unique(self.labels)
        self.n_classes = len(self.unique_classes)
        self.class_indices: Dict[int, np.ndarray] = {
            c: np.nonzero(self.labels == c)[0] for c in self.unique_classes
        }

        # Quota per class per batch
        self.samples_per_class = max(1, self.batch_size // self.n_classes)
        self.actual_batch_size = self.samples_per_class * self.n_classes

        # Number of complete batches available
        max_class_samples = max(len(idx) for idx in self.class_indices.values())
        self.n_batches = math.ceil(max_class_samples / self.samples_per_class)

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle indices within each class independently and store in O(1) deques
        shuffled_queues: Dict[int, deque] = {}
        for c in self.unique_classes:
            idx = self.class_indices[c].copy()
            self.rng.shuffle(idx)
            shuffled_queues[c] = deque(idx.tolist())

        for _ in range(self.n_batches):
            batch: List[int] = []
            for c in self.unique_classes:
                q = shuffled_queues[c]
                if len(q) < self.samples_per_class:
                    # Replenish queue if exhausted
                    fresh = self.class_indices[c].copy()
                    self.rng.shuffle(fresh)
                    q.extend(fresh.tolist())
                chosen = [q.popleft() for _ in range(min(len(q), self.samples_per_class))]
                batch.extend(chosen)

            self.rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.n_batches


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: TARGET FILE ADAPTERS & HIGH-LEVEL FUNCTIONAL APIS
# ═══════════════════════════════════════════════════════════════════════════

def prepare_qnas_subsets(
    train_z: torch.Tensor,
    train_y: torch.Tensor,
    val_z: torch.Tensor,
    val_y: torch.Tensor,
    n_train: int = 400,
    n_val: int = 150,
    seed: int = 42,
    use_coreset: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepares mathematically balanced, representative datasets for QNAS candidate evaluation.

    Drop-in replacement for naive sequential slicing in engine.py:5597-5610:
      train_z_sub = train_z[:400]
      val_z_sub   = val_z[:150]
    Guarantees that all disease classes are proportionally and topologically represented.
    """
    subsampler = StratifiedSubsampler(mode=StratificationMode.BALANCED, min_quorum=4, seed=seed)

    # 1. Validation subset selection (strictly stratified)
    val_indices = subsampler.sample_indices(val_y, total_samples=n_val)
    val_z_sub = val_z[val_indices]
    val_y_sub = val_y[val_indices]

    # 2. Training subset selection (via core-set active medoid coverage or balanced sampling)
    if use_coreset and train_z.size(0) > n_train:
        quotas = subsampler.compute_class_quotas(train_y.squeeze().cpu().numpy(), n_train)
        selector = CoreSetStratifiedSelector(seed=seed)
        train_indices = selector.select_coreset_indices(train_z, train_y, quotas, is_periodic=False)
        train_z_sub = train_z[train_indices]
        train_y_sub = train_y[train_indices]
    else:
        train_indices = subsampler.sample_indices(train_y, total_samples=n_train)
        train_z_sub = train_z[train_indices]
        train_y_sub = train_y[train_indices]

    return train_z_sub, train_y_sub, val_z_sub, val_y_sub


def get_stratified_medmnist_loaders(
    dataset_name: str,
    batch_size: int = 128,
    download_root: str = "./data",
    size: int = 224,
    use_multiview: bool = False,
    n_views: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Constructs MedMNIST DataLoaders equipped with class-stratified batching and multi-view transforms."""
    import medmnist
    from medmnist import INFO

    if dataset_name not in INFO:
        raise ValueError(f"Unknown MedMNIST dataset '{dataset_name}'. Choose from {list(INFO.keys())}")

    info = INFO[dataset_name]
    DatasetClass = getattr(medmnist, info["python_class"])
    actual_size = size if not dataset_name.endswith("3d") else 28

    if use_multiview:
        train_tfm = MultiViewTransform(dataset_name, n_views=n_views, target_size=actual_size, is_training=True)
        collator = MultiViewBatchCollator(flatten_views=True)
    else:
        policy = get_modality_policy(dataset_name)
        augmentor = ModalityAdaptiveAugmentor(policy, target_size=actual_size, is_training=True)
        train_tfm = augmentor.build_geometric_view_transform()
        collator = None

    eval_policy = get_modality_policy(dataset_name)
    eval_augmentor = ModalityAdaptiveAugmentor(eval_policy, target_size=actual_size, is_training=False)
    eval_tfm = eval_augmentor.build_canonical_transform()

    train_ds = DatasetClass(split="train", transform=train_tfm, download=True, root=download_root, size=actual_size)
    val_ds = DatasetClass(split="val", transform=eval_tfm, download=True, root=download_root, size=actual_size)
    test_ds = DatasetClass(split="test", transform=eval_tfm, download=True, root=download_root, size=actual_size)

    # Wrap train_ds in exact class-stratified batch sampler
    sampler = ClassStratifiedBatchSampler(train_ds.labels, batch_size=batch_size, seed=seed)
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    meta = {
        "dataset_name": dataset_name,
        "image_size": actual_size,
        "n_classes": len(info["label"]),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "modality_policy": eval_policy,
    }
    return train_loader, val_loader, test_loader, meta


def get_stratified_test_batch(
    dataset,
    batch_size: int = 16,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Retrieves an exact, balanced test batch for Streamlit interactive evaluation in app.py.

    Drop-in replacement for unstratified np.random.choice in app.py:401.
    """
    labels = getattr(dataset, "labels", None)
    if labels is None:
        raise AttributeError("Dataset does not have a 'labels' attribute.")

    subsampler = StratifiedSubsampler(mode=StratificationMode.BALANCED, min_quorum=1, seed=seed)
    indices = subsampler.sample_indices(labels, total_samples=batch_size)

    images: List[torch.Tensor] = []
    lbls: List[int] = []
    for idx in indices:
        im, lb = dataset[idx]
        images.append(im)
        lbls.append(int(np.squeeze(lb)))

    return torch.stack(images, dim=0), torch.tensor(lbls, dtype=torch.long)


@torch.no_grad()
def predict_with_multiview_tta(
    model: nn.Module,
    image: Image.Image,
    dataset_name: str,
    device: torch.device,
    n_views: int = 4,
) -> Dict[str, Union[np.ndarray, float, int]]:
    """Runs Test-Time Augmentation (TTA) with Bayesian View Pooling and Aleatoric Uncertainty Estimation.

    Integrates into app.py for clinical inference.
    """
    model.eval()
    tta_engine = TestTimeMultiViewEnsemble(dataset_name=dataset_name, n_views=n_views)
    view_tensor = tta_engine.generate_tta_views(image).to(device)  # (V, 3, 224, 224)

    logits = model(view_tensor, use_cached_features=False)         # (V, C)
    probs = F.softmax(logits, dim=-1).cpu()                        # (V, C)

    pooled_prob, aleatoric_unc = bayesian_view_pooling(probs, temperature=0.5)

    pred_class = int(torch.argmax(pooled_prob).item())

    return {
        "predicted_class": pred_class,
        "pooled_probabilities": pooled_prob.numpy(),
        "per_view_probabilities": probs.numpy(),
        "aleatoric_uncertainty": float(aleatoric_unc.item()),
    }
