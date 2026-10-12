"""
dataset.py — MedMNIST v2 Data Ingestion, Normalisation & State-Level Quantum Augmentation
==========================================================================================
Supports:  pneumoniamnist (1-ch, 2-cls)
           breastmnist   (1-ch, 2-cls)
           pathmnist     (3-ch, 9-cls)

Key quantum-aware augmentation primitives:
  • Quantum Mixup   – Hilbert-space interpolation of latent states
  • Phase Perturbation – small Rx/Rz rotations to prevent fragile interference
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import medmnist
import numpy as np
import seaborn as sns
import torch
from medmnist import INFO
from torch.utils.data import DataLoader
from torchvision import transforms

# ---------------------------------------------------------------------------
# Strict BiomedCLIP Normalization Constants (Fix: Flaws 1 & 7)
# ---------------------------------------------------------------------------
BIOMEDCLIP_MEAN = [0.48145466, 0.45782750, 0.40821073]
BIOMEDCLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASET_CONFIG: Dict[str, Dict] = {}
for name, info in INFO.items():
    DatasetClass = getattr(medmnist, info['python_class'])
    DATASET_CONFIG[name] = {
        "class": DatasetClass,
        "in_channels": info["n_channels"],
        "n_classes": len(info["label"]),
        "task": info["task"],
        "label_map": {int(k): v.title() for k, v in info["label"].items()},
        "is_3d": name.endswith("3d")
    }

# Overrides for previous custom naming
DATASET_CONFIG["pneumoniamnist"]["label_map"] = {0: "Normal", 1: "Pneumonia"}
DATASET_CONFIG["breastmnist"]["label_map"] = {0: "Malignant", 1: "Benign/Normal"}
DATASET_CONFIG["pathmnist"]["label_map"] = {
    0: "Adipose", 1: "Background", 2: "Debris",
    3: "Lymphocytes", 4: "Mucus", 5: "Smooth Muscle",
    6: "Normal Mucosa", 7: "Cancer Stroma", 8: "Tumour Epithelium",
}

# ---------------------------------------------------------------------------
# Custom NPZ Dataset Support (e.g. covid_224.npz)
# ---------------------------------------------------------------------------
class NPZDataset(torch.utils.data.Dataset):
    """Loads pre-processed .npz datasets matching the MedMNIST array convention."""
    def __init__(self, npz_path: str, split: str = "train", transform=None, **kwargs):
        self.split = split
        self.transform = transform
        data = np.load(npz_path)
        self.imgs = data[f"{split}_images"]
        self.labels = data[f"{split}_labels"]
        pid_key = f"{split}_patient_ids"
        self.patient_ids = data[pid_key] if pid_key in data.files else None

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        from PIL import Image
        img = self.imgs[idx]
        if img.ndim == 2:
            img = Image.fromarray(img, mode="L")
        else:
            img = Image.fromarray(img)
        target = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class Covid224Dataset(NPZDataset):
    """Top-level dataset class for covid_224.npz (pickleable for PyTorch multiprocessing)."""
    def __init__(self, split: str = "train", transform=None, root: str = "./data", size: int = 224, **kwargs):
        candidates = [
            os.path.join(root, "covid_224.npz"),
            os.path.join("./data", "covid_224.npz"),
            os.path.join(os.path.dirname(__file__), "..", "data", "covid_224.npz"),
            "data/covid_224.npz",
            "covid_224.npz",
        ]
        chosen = next((p for p in candidates if os.path.exists(p)), candidates[0])
        super().__init__(npz_path=chosen, split=split, transform=transform, **kwargs)


DATASET_CONFIG["covid_224"] = {
    "class": Covid224Dataset,
    "in_channels": 1,
    "n_classes": 2,
    "task": "binary-class",
    "label_map": {0: "Normal", 1: "COVID"},
    "is_3d": False,
}
DATASET_CONFIG["covid"] = DATASET_CONFIG["covid_224"]


class Oasis224Dataset(NPZDataset):
    """Top-level dataset class for oasis_224.npz (pickleable for PyTorch multiprocessing)."""
    def __init__(self, split: str = "train", transform=None, root: str = "./data", size: int = 224, **kwargs):
        candidates = [
            os.path.join(root, "oasis_224.npz"),
            os.path.join("./data", "oasis_224.npz"),
            os.path.join(os.path.dirname(__file__), "..", "data", "oasis_224.npz"),
            "data/oasis_224.npz",
            "oasis_224.npz",
        ]
        chosen = next((p for p in candidates if os.path.exists(p)), candidates[0])
        super().__init__(npz_path=chosen, split=split, transform=transform, **kwargs)


DATASET_CONFIG["oasis_224"] = {
    "class": Oasis224Dataset,
    "in_channels": 1,
    "n_classes": 2,
    "task": "binary-class",
    "label_map": {0: "Non-Demented", 1: "Demented"},
    "is_3d": False,
}
DATASET_CONFIG["oasis"] = DATASET_CONFIG["oasis_224"]




# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def get_medmnist_dataloaders(
    dataset_name: str,
    batch_size: int = 128,
    download_root: str = "./data",
    num_workers: int = 4,
    balanced_sampling: bool = False,
    size: int = 224,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Return (train_loader, val_loader, test_loader, meta) for a MedMNIST v2 subset.

    Parameters:
      size: Input image resolution (default: 224 for native high-resolution ViT input,
            supports 28, 64, 128, 224).
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from {list(DATASET_CONFIG.keys())}"
        )

    cfg = DATASET_CONFIG[dataset_name]
    # BiomedCLIP expects 3 channels
    n_ch = 3

    # Ensure download root exists
    os.makedirs(download_root, exist_ok=True)

    # --- transforms --------------------------------------------------------
    mean = BIOMEDCLIP_MEAN
    std  = BIOMEDCLIP_STD
    
    class MedMNIST3DTransform:
        def __call__(self, img):
            # img is typically a (1, D, H, W) numpy array for 3D medmnist
            tensor = torch.from_numpy(np.array(img)).float()
            if tensor.max() > 2.0:
                tensor = tensor / 255.0
            # Broadcast normalize
            m = torch.tensor(mean).view(-1, 1, 1, 1)
            s = torch.tensor(std).view(-1, 1, 1, 1)
            return (tensor - m) / s

    if cfg.get("is_3d", False):
        tfm = MedMNIST3DTransform()
    else:
        if cfg["in_channels"] == 1:
            gray_mean = [0.48145466, 0.48145466, 0.48145466]
            gray_std  = [0.26862954, 0.26862954, 0.26862954]
            tfm = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=gray_mean, std=gray_std),
            ])
        else:
            tfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

    DatasetClass = cfg["class"]

    # 3D datasets only support 28 or 64 voxel grids
    actual_size = size if not cfg.get("is_3d", False) else 28

    # Check if local dataset file already exists to avoid redundant remote MD5/Zenodo requests
    expected_filename = f"{dataset_name}_{actual_size}.npz" if actual_size and actual_size != 28 else f"{dataset_name}.npz"
    file_exists = os.path.exists(os.path.join(download_root, expected_filename))
    need_download = not file_exists

    train_ds = DatasetClass(split="train", transform=tfm,
                            download=need_download, root=download_root, size=actual_size)
    val_ds   = DatasetClass(split="val",   transform=tfm,
                            download=need_download, root=download_root, size=actual_size)
    test_ds  = DatasetClass(split="test",  transform=tfm,
                            download=need_download, root=download_root, size=actual_size)

    # Memory safety for WSL2 and resource-constrained environments
    if num_workers > 2:
        num_workers = min(2, os.cpu_count() or 1)
    if actual_size == 224 and dataset_name in ["octmnist", "dermamnist", "pathmnist"]:
        num_workers = 0  # Avoid multi-process forking overhead for multi-GB arrays
    use_pin = torch.cuda.is_available() and (num_workers == 0 or actual_size <= 64)

    should_balance = balanced_sampling or (dataset_name == "octmnist")
    if should_balance:
        from torch.utils.data import WeightedRandomSampler
        labels = train_ds.labels
        if labels.ndim > 1 and labels.shape[1] > 1:
            sample_weights = torch.ones(len(labels), dtype=torch.double)
        else:
            labels_sq = np.asarray(labels).squeeze()
            class_counts = np.bincount(labels_sq)
            class_counts = np.maximum(class_counts, 1)
            class_weights = 1.0 / class_counts.astype(np.float64)
            sample_weights = torch.from_numpy(class_weights[labels_sq]).double()
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        drop_last = len(train_ds) >= batch_size
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=use_pin, drop_last=drop_last)
    else:
        drop_last = len(train_ds) >= batch_size
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=use_pin, drop_last=drop_last)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=use_pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=use_pin)

    meta = {
        "dataset_name": dataset_name,
        "image_size":   actual_size,
        "in_channels":  n_ch,
        "n_classes":    cfg["n_classes"],
        "task":         cfg["task"],
        "label_map":    cfg["label_map"],
        "is_3d":        cfg.get("is_3d", False),
        "train_size":   len(train_ds),
        "val_size":     len(val_ds),
        "test_size":    len(test_ds),
    }
    if hasattr(train_ds, "patient_ids") and train_ds.patient_ids is not None:
        meta["train_patient_ids"] = train_ds.patient_ids
        meta["val_patient_ids"] = val_ds.patient_ids
        meta["test_patient_ids"] = test_ds.patient_ids

    return train_loader, val_loader, test_loader, meta


# ---------------------------------------------------------------------------
# Class-weight computation (for Focal Loss / Weighted CE)
# ---------------------------------------------------------------------------
def compute_effective_class_weights(
    loader: DataLoader,
    n_classes: int,
    beta: float = 0.9999,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    r"""Compute class-balanced weights based on effective number of samples (Cui et al., CVPR 2019).

    E_n = (1 - \beta^{N_c}) / (1 - \beta)
    w_c = 1 / E_n \propto (1 - \beta) / (1 - \beta^{N_c})
    Normalized so sum(w) = n_classes (mean = 1.0).
    """
    counts = torch.zeros(n_classes, dtype=torch.float64)
    ds = getattr(loader, "dataset", None)
    if ds is not None and hasattr(ds, "labels"):
        lbls = np.asarray(ds.labels).squeeze()
        bc = np.bincount(lbls, minlength=n_classes)[:n_classes]
        counts = torch.from_numpy(bc).double()
    elif ds is not None and hasattr(ds, "tensors") and len(ds.tensors) >= 2:
        lbls = ds.tensors[1].squeeze().cpu().long()
        bc = torch.bincount(lbls, minlength=n_classes)[:n_classes]
        counts = bc.double()
    else:
        for _, labels in loader:
            if labels.ndim > 1 and labels.shape[1] == n_classes:
                counts += labels.sum(dim=0).cpu().double()
            else:
                labels = labels.reshape(-1).long()
                for c in range(n_classes):
                    counts[c] += (labels == c).sum().item()

    counts = counts.clamp(min=1.0)
    effective_num = 1.0 - torch.pow(beta, counts)
    weights = (1.0 - beta) / (effective_num + 1e-8)
    # Normalize weights so mean is 1.0 (sum = n_classes)
    weights = weights / weights.mean()
    return weights.float().to(device)


def compute_class_weights(
    loader: DataLoader,
    n_classes: int,
    device: torch.device = torch.device("cuda"),
    method: str = "effective",
    beta: float = 0.9999,
) -> torch.Tensor:
    """Compute class weights from a DataLoader. Supports 'effective' (Cui et al., 2019) or 'inverse'."""
    if method == "effective":
        return compute_effective_class_weights(loader, n_classes, beta=beta, device=device)
    counts = torch.zeros(n_classes, dtype=torch.float64)
    ds = getattr(loader, "dataset", None)
    if ds is not None and hasattr(ds, "labels"):
        lbls = np.asarray(ds.labels).squeeze()
        bc = np.bincount(lbls, minlength=n_classes)[:n_classes]
        counts = torch.from_numpy(bc).double()
    elif ds is not None and hasattr(ds, "tensors") and len(ds.tensors) >= 2:
        lbls = ds.tensors[1].squeeze().cpu().long()
        bc = torch.bincount(lbls, minlength=n_classes)[:n_classes]
        counts = bc.double()
    else:
        for _, labels in loader:
            if labels.ndim > 1 and labels.shape[1] == n_classes:
                counts += labels.sum(dim=0).cpu().double()
            else:
                labels = labels.reshape(-1).long()
                for c in range(n_classes):
                    counts[c] += (labels == c).sum().item()
    weights = counts.sum() / (n_classes * counts.clamp(min=1))
    return weights.float().to(device)


# ---------------------------------------------------------------------------
# Quantum Mixup — Hilbert-space interpolation of latent vectors
# ---------------------------------------------------------------------------
def quantum_mixup(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    n_classes: int,
    alpha: float = 0.15,
    lam: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Affine convex interpolation of latent angle vectors (Fix: Flaw 4).

    Replaces the broken sqrt-scaling formulation which caused up to 41.4%
    norm inflation. The correct convex homotopy is:
        z_mix = λ * z_a + (1 - λ) * z_b,   λ ~ Beta(α, α)
    This guarantees the mixed angles remain within the valid bounded convex
    hull without any artificial norm distortion.

    Returns mixed latent tensor and soft label tensor.
    """
    if lam is None:
        lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0

    # Affine convex combination — no sqrt, no norm inflation (Fix: Flaw 4)
    z_mix = lam * z_a + (1.0 - lam) * z_b

    # Soft one-hot labels (consistent linear interpolation)
    y_a = torch.nn.functional.one_hot(labels_a.long(), n_classes).float()
    y_b = torch.nn.functional.one_hot(labels_b.long(), n_classes).float()
    y_mix = lam * y_a + (1.0 - lam) * y_b

    return z_mix, y_mix


# ---------------------------------------------------------------------------
# Phase Perturbation — small Rx/Rz rotations on latent angles
# ---------------------------------------------------------------------------
def phase_perturbation(
    z: torch.Tensor,
    epsilon: float = 0.05,
    delta: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Apply small orthogonal phase perturbations to the latent vector.

    For each latent dimension i the effective rotation becomes:
       z'_i  =  z_i  +  δφ_x  +  δφ_z
    with  δφ_x, δφ_z  ~  U(-ε, ε).

    This prevents the variational circuit from overfitting to fragile
    interference states while preserving the overall Bloch-sphere coverage.
    """
    if delta is not None:
        return z + delta
    if epsilon <= 0.0:
        return z
    delta_x = torch.empty_like(z).uniform_(-epsilon, epsilon)
    delta_z = torch.empty_like(z).uniform_(-epsilon, epsilon)
    return z + delta_x + delta_z


# ---------------------------------------------------------------------------
# Expectation-Space Quantum Mixup (Density Operator Convexity) — Fix E-01
# ---------------------------------------------------------------------------
def expectation_quantum_mixup(
    exp_a: torch.Tensor,
    exp_b: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    n_classes: int,
    alpha: float = 0.2,
    lam: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Convex statistical mixture in quantum expectation space (Fix E-01).

    Unlike ``quantum_mixup`` which interpolates pre-quantum rotation angles
    in Euclidean :math:`\mathbb{R}^{12}` — violating unitary dynamics because
    :math:`U(\lambda z_A + (1-\lambda)z_B) \neq \lambda U(z_A) + (1-\lambda) U(z_B)` —
    this function interpolates *post-quantum observable expectation vectors*.

    By the linearity of the quantum trace operation (Dilip et al., Phys. Rev. A, 2022):

    .. math::
        \langle \hat{O}_k \rangle_{\rho_{\text{mix}}}
        = \text{Tr}(\hat{O}_k (\lambda \rho_A + (1-\lambda) \rho_B))
        = \lambda \langle \hat{O}_k \rangle_A + (1-\lambda) \langle \hat{O}_k \rangle_B

    This is **rigorously exact** for incoherent statistical mixtures and preserves
    the convex structure of the density operator space :math:`\mathcal{S}(\mathcal{H})`.

    Args:
        exp_a: Post-quantum expectation vectors of shape ``(B, D_obs)`` for sample batch A,
               where ``D_obs`` is the number of observables (e.g., 11 for 4-qubit readout).
        exp_b: Post-quantum expectation vectors of shape ``(B, D_obs)`` for sample batch B.
        labels_a: Integer class labels of shape ``(B,)`` for batch A.
        labels_b: Integer class labels of shape ``(B,)`` for batch B.
        n_classes: Total number of classes in the classification task.
        alpha: Beta distribution parameter controlling mixup strength.
               Smaller values → weaker interpolation (closer to original samples).
        lam: If provided, override the sampled mixing coefficient.

    Returns:
        (mixed_exp, mixed_labels): Tuple of mixed expectation vectors and
        soft one-hot label vectors.
    """
    if lam is None:
        lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0

    # Convex combination in observable expectation space — exact by trace linearity
    exp_mix = lam * exp_a + (1.0 - lam) * exp_b

    # Soft one-hot labels (consistent linear interpolation)
    y_a = torch.nn.functional.one_hot(labels_a.long(), n_classes).float()
    y_b = torch.nn.functional.one_hot(labels_b.long(), n_classes).float()
    y_mix = lam * y_a + (1.0 - lam) * y_b

    return exp_mix, y_mix


# ---------------------------------------------------------------------------
# True SU(2) Lie Algebra Phase Perturbation — Fix E-02
# ---------------------------------------------------------------------------
def su2_phase_perturbation(
    z: torch.Tensor,
    n_qubits: int,
    epsilon: float = 0.05,
    delta: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Apply true non-commuting :math:`SU(2)` Lie group perturbations (Fix E-02).

    Unlike ``phase_perturbation`` which adds two scalar uniform deltas (yielding a
    single symmetric triangular shift that fails to respect the non-commutative
    structure of :math:`SU(2)`), this function generates independent perturbation
    angles :math:`\delta_x` and :math:`\delta_z` for each qubit coordinate, preserving
    the non-commutative Lie algebra structure :math:`[J_x, J_z] \neq 0`.

    The perturbed unitary for each qubit is:

    .. math::
        \hat{U}_{\text{perturbed}}(z_q) = R_z(\delta_{z,q}) \, R_x(\delta_{x,q}) \, R_y(z_q)

    In the latent angle representation, this manifests as independent per-coordinate
    perturbations that respect the different geometric roles of polar (:math:`\theta`)
    and azimuthal (:math:`\phi, \lambda`) angles on the Bloch sphere:

    - Polar angle :math:`\theta \in [0, \pi]`: perturbed with :math:`\epsilon/2`
      to avoid escaping the hemisphere.
    - Azimuthal angles :math:`\phi, \lambda \in [-\pi, \pi]`: perturbed with
      full :math:`\epsilon` respecting their :math:`2\pi`-periodic nature.

    Args:
        z: Latent angle tensor of shape ``(B, 3*n_qubits)``.
            Columns ``0..n-1`` are polar (:math:`\theta`),
            ``n..2n-1`` and ``2n..3n-1`` are azimuthal (:math:`\phi, \lambda`).
        n_qubits: Number of qubits.
        epsilon: Maximum perturbation magnitude (default: 0.05 radians).
        delta: Optional pre-computed perturbation tensor for Q-SAM replay.

    Returns:
        Perturbed latent angle tensor of the same shape.
    """
    if delta is not None:
        return z + delta
    if epsilon <= 0.0 or z.size(-1) < 3 * n_qubits:
        return z

    # Polar angles (theta ∈ [0, π]): smaller perturbation to stay in valid range
    delta_theta = torch.empty(z.size(0), n_qubits, device=z.device).uniform_(
        -epsilon / 2.0, epsilon / 2.0
    )
    # Azimuthal angles (phi ∈ [-π, π]): independent Rz-like perturbation
    delta_phi = torch.empty(z.size(0), n_qubits, device=z.device).uniform_(
        -epsilon, epsilon
    )
    # Lambda angles (lambda ∈ [-π, π]): independent Rx-like perturbation
    delta_lam = torch.empty(z.size(0), n_qubits, device=z.device).uniform_(
        -epsilon, epsilon
    )

    delta_phase = torch.cat([delta_theta, delta_phi, delta_lam], dim=-1)
    return z + delta_phase


# ---------------------------------------------------------------------------
# K-Fold Cross-Validation Data Partitioning (Varma & Simon 2006; Cawley & Talbot 2010)
# ---------------------------------------------------------------------------
def get_kfold_loaders(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_folds: int = 5,
    batch_size: int = 128,
    patient_ids: Optional[np.ndarray] = None,
    seed: int = 42,
    cache_in_vram: bool = False,
    device: Optional[torch.device] = None,
) -> List[Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]]:
    r"""Generate K stratified (or stratified-group) fold train/val DataLoaders.

    Operates directly on pre-cached (N, 512) BiomedCLIP embeddings in memory —
    no re-extraction needed.  Each fold's WeightedRandomSampler is recomputed
    strictly within its training partition (zero data leakage).

    Args:
        features: (N, D) cached embedding tensor (typically D=512).
        labels:   (N,) integer class labels.
        n_folds:  Number of cross-validation folds (default: 5).
        batch_size: Mini-batch size for DataLoaders.
        patient_ids: Optional (N,) array of patient/group identifiers.
                     When provided, uses StratifiedGroupKFold to prevent
                     intra-patient anatomical leakage (Wiens et al., 2019).
        seed: Random seed for reproducible fold partitions.
        cache_in_vram: If True and device is CUDA, preload fold tensors
                       directly into GPU VRAM.
        device: Target torch device (used only when cache_in_vram=True).

    Returns:
        List of (train_loader, val_loader, train_indices, val_indices) tuples,
        one per fold.  Indices are numpy arrays into the original dataset.

    References:
        - Varma, S. & Simon, R. (2006). BMC Bioinformatics.
        - Cawley, G. & Talbot, T. (2010). JMLR.
        - Wiens, J. et al. (2019). Nature Medicine.
    """
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    labels_np = labels.cpu().numpy().squeeze() if torch.is_tensor(labels) else np.asarray(labels).squeeze()
    n_classes = int(labels_np.max()) + 1

    # Select splitter based on whether patient group IDs are available
    if patient_ids is not None:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(len(labels_np)), labels_np, groups=patient_ids)
    else:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(len(labels_np)), labels_np)

    fold_loaders: List[Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
        train_idx = np.asarray(train_idx)
        val_idx = np.asarray(val_idx)

        # Slice features and labels for this fold
        feat_train = features[train_idx]
        lbl_train = labels[train_idx] if torch.is_tensor(labels) else torch.from_numpy(labels_np[train_idx]).long()
        feat_val = features[val_idx]
        lbl_val = labels[val_idx] if torch.is_tensor(labels) else torch.from_numpy(labels_np[val_idx]).long()

        # Optionally preload into GPU VRAM
        if cache_in_vram and device is not None and device.type == "cuda":
            feat_train = feat_train.to(device)
            lbl_train = lbl_train.to(device)
            feat_val = feat_val.to(device)
            lbl_val = lbl_val.to(device)

        # --- Per-fold class-balanced WeightedRandomSampler (zero leakage) ---
        fold_labels_np = labels_np[train_idx]
        class_counts = np.bincount(fold_labels_np.astype(int), minlength=n_classes)
        class_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float64)
        sample_weights = torch.from_numpy(class_weights[fold_labels_np.astype(int)]).double()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        pin_mem = torch.cuda.is_available() and not cache_in_vram
        drop_last_train = len(train_idx) >= batch_size

        train_ds = TensorDataset(feat_train, lbl_train)
        val_ds = TensorDataset(feat_val, lbl_val)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=pin_mem,
            drop_last=drop_last_train,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_mem,
        )

        fold_loaders.append((train_loader, val_loader, train_idx, val_idx))

    return fold_loaders


# ---------------------------------------------------------------------------
# EDA / Visualisation Utilities
# ---------------------------------------------------------------------------
def plot_class_distribution(loader: DataLoader, meta: Dict,
                            save_path: Optional[str] = None) -> plt.Figure:
    """Bar chart of class frequencies."""
    all_labels = None
    # Fast O(1) retrieval directly from dataset if labels are cached
    if hasattr(loader, "dataset") and hasattr(loader.dataset, "labels") and loader.dataset.labels is not None:
        raw_lbls = loader.dataset.labels
        if isinstance(raw_lbls, np.ndarray):
            all_labels = raw_lbls
        elif torch.is_tensor(raw_lbls):
            all_labels = raw_lbls.detach().cpu().numpy()
        else:
            all_labels = np.asarray(raw_lbls)

    # Fallback to loader iteration if dataset.labels is absent
    if all_labels is None or len(all_labels) == 0:
        collected = []
        for _, labels in loader:
            if torch.is_tensor(labels):
                collected.append(labels.detach().cpu().numpy())
            elif isinstance(labels, np.ndarray):
                collected.append(labels)
            else:
                collected.append(np.asarray(labels))
        if len(collected) > 0:
            all_labels = np.concatenate(collected, axis=0)
        else:
            all_labels = np.zeros((0, 1), dtype=int)

    label_map = meta.get("label_map", {})
    classes = [label_map.get(i, str(i)) for i in range(meta["n_classes"])]
    
    if "multi-label" in meta["task"]:
        counts = np.sum(all_labels, axis=0).astype(int).flatten().tolist()
    else:
        flat_labels = all_labels.flatten()
        counts = [np.sum(flat_labels == i) for i in range(meta["n_classes"])]

    fig, ax = plt.subplots(figsize=(8, 4))
    palette = sns.color_palette("viridis", len(classes))
    bars = ax.bar(classes, counts, color=palette, edgecolor="white", linewidth=0.8)
    ax.set_title(f"Class Distribution — {meta['dataset_name']}", fontsize=13,
                 fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(cnt), ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_sample_grid(loader: DataLoader, meta: Dict, n: int = 16,
                     save_path: Optional[str] = None) -> plt.Figure:
    """Display a balanced stratified n-image grid from the loader (Fix: Flaw F-12)."""
    n_classes = meta.get("n_classes", len(meta.get("label_map", {})))
    per_class = max(1, n // max(1, n_classes))
    is_multi_label = "multi-label" in meta.get("task", "")

    collected_images = []
    collected_labels = []

    if not is_multi_label and n_classes > 1:
        counts = {c: 0 for c in range(n_classes)}
        for batch_imgs, batch_lbls in loader:
            b_lbls = batch_lbls.squeeze().long()
            for idx in range(len(b_lbls)):
                c = int(b_lbls[idx].item()) if b_lbls.dim() > 0 else int(b_lbls.item())
                if c in counts and counts[c] < per_class:
                    collected_images.append(batch_imgs[idx])
                    collected_labels.append(b_lbls[idx] if b_lbls.dim() > 0 else b_lbls)
                    counts[c] += 1
            if len(collected_images) >= n or all(cnt >= per_class for cnt in counts.values()):
                break

    # Fallback to standard loader slicing if stratified collection didn't fill n items
    if len(collected_images) < n:
        try:
            batch_imgs, batch_lbls = next(iter(loader))
            rem = n - len(collected_images)
            for idx in range(min(rem, len(batch_imgs))):
                collected_images.append(batch_imgs[idx])
                b_l = batch_lbls[idx].squeeze()
                collected_labels.append(b_l)
        except Exception:
            pass

    images = torch.stack(collected_images[:n]) if collected_images else torch.zeros(n, 1, 28, 28)
    labels = torch.stack(collected_labels[:n]).squeeze() if collected_labels else torch.zeros(n, dtype=torch.long)

    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))
    axes = np.asarray(axes).flatten()
    label_map = meta["label_map"]

    for i in range(n):
        img = images[i].clone()
        # Undo normalisation for display
        if img.shape[0] == 3:
            m = torch.tensor(BIOMEDCLIP_MEAN, dtype=img.dtype).view(3, 1, 1)
            s = torch.tensor(BIOMEDCLIP_STD, dtype=img.dtype).view(3, 1, 1)
            img = (img * s + m).clamp(0.0, 1.0)
        else:
            img = (img * BIOMEDCLIP_STD[0] + BIOMEDCLIP_MEAN[0]).clamp(0.0, 1.0)
        if meta.get("is_3d", False):
            # Take middle slice of the depth dimension
            d_mid = img.shape[1] // 2
            img = img[:, d_mid, :, :]
            
        if img.shape[0] == 1:
            axes[i].imshow(img.squeeze(0).numpy(), cmap="gray")
        else:
            axes[i].imshow(img.permute(1, 2, 0).numpy())
        if "multi-label" in meta["task"]:
            # Handle multi-label by joining all active classes
            active_classes = [label_map.get(idx, str(idx)) for idx, val in enumerate(labels[i]) if val == 1]
            title = ", ".join(active_classes) if active_classes else "None"
            axes[i].set_title(title, fontsize=6)
        else:
            lbl = int(labels[i].item()) if labels.dim() > 0 else int(labels.item())
            axes[i].set_title(label_map.get(lbl, str(lbl)), fontsize=8)
            
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")

    img_res = meta.get("image_size", 224)
    fig.suptitle(f"Sample Images — {meta['dataset_name']} ({img_res}×{img_res})", fontsize=12,
                 fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_pixel_intensity_histogram(loader: DataLoader, meta: Dict,
                                   save_path: Optional[str] = None) -> plt.Figure:
    """Histogram of normalised pixel intensities (first batch)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    try:
        images, _ = next(iter(loader))
        pixel_vals = images.detach().cpu().numpy().flatten()
    except Exception:
        pixel_vals = np.array([0.0])
    ax.hist(pixel_vals, bins=100, color="#4c72b0", alpha=0.8, edgecolor="white",
            linewidth=0.4)
    ax.set_title(f"Pixel Intensity Distribution (normalised) — {meta['dataset_name']}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    for name in DATASET_CONFIG:
        print(f"\n{'='*60}")
        print(f"Loading {name} …")
        tr, va, te, m = get_medmnist_dataloaders(name, batch_size=32,
                                                  download_root="./data")
        print(f"  Train: {m['train_size']},  Val: {m['val_size']},  "
              f"Test: {m['test_size']},  Classes: {m['n_classes']},  "
              f"Channels: {m['in_channels']}")

        # quick shape check
        imgs, lbls = next(iter(tr))
        print(f"  Batch images shape : {imgs.shape}")
        print(f"  Batch labels shape : {lbls.shape}")
        print(f"  Pixel range        : [{imgs.min():.2f}, {imgs.max():.2f}]")

        # class weights
        cw = compute_class_weights(tr, m["n_classes"])
        print(f"  Class weights      : {cw.tolist()}")

        # Save EDA plots
        os.makedirs("outputs/eda", exist_ok=True)
        plot_class_distribution(tr, m, save_path=f"outputs/eda/{name}_class_dist.png")
        plot_sample_grid(tr, m, save_path=f"outputs/eda/{name}_samples.png")
        plot_pixel_intensity_histogram(tr, m,
                                       save_path=f"outputs/eda/{name}_pixel_hist.png")

    print("\n[OK]  dataset.py self-test passed.")
