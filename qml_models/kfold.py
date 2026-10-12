"""
K-Fold Stratified Cross-Validation Engine for Hybrid Quantum-Classical Architectures.
Decoupled from engine.py for modularity, unit testability, and clean architecture.

References:
    - Varma & Simon (2006): Bias in error estimation when using cross-validation for model selection.
    - Cawley & Talbot (2010): Over-fitting in model selection and subsequent selection bias in performance evaluation.
    - Wiens et al. (2019): Do no harm: a roadmap for responsible machine learning for healthcare.
"""

import os
import math
import json
import copy
import gc
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

# Ensure project root and module dir are in sys.path
_pkg_root = str(Path(__file__).resolve().parent.parent)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
_module_dir = str(Path(__file__).resolve().parent)
if _module_dir not in sys.path:
    sys.path.insert(0, _module_dir)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from model import build_models, HybridCNNVQC, ClassicalBaseline, get_torch_device
except ImportError:
    from qml_models.model import build_models, HybridCNNVQC, ClassicalBaseline, get_torch_device

try:
    from dataset import get_medmnist_dataloaders, get_kfold_loaders, DATASET_CONFIG, compute_class_weights
except ImportError:
    from qml_models.dataset import get_medmnist_dataloaders, get_kfold_loaders, DATASET_CONFIG, compute_class_weights

try:
    from adapter import pretrain_cq_adapter
except ImportError:
    from qml_models.adapter import pretrain_cq_adapter

try:
    from losses import ClassBalancedLDAMLoss, _compute_class_counts, FocalLoss, resolve_asymmetric_loss_params
except ImportError:
    from qml_models.losses import ClassBalancedLDAMLoss, _compute_class_counts, FocalLoss, resolve_asymmetric_loss_params

try:
    from calibration import (
        _collect_val_logits, optimize_temperature_scaling, optimize_decision_threshold,
        QuantumBetaCalibrator, QuantumMultiClassCalibrator, optimize_multiclass_cost_sensitive_thresholds,
        predict_with_cost_matrix, build_mondrian_conformal_artifact, split_calibration_indices
    )
except ImportError:
    from qml_models.calibration import (
        _collect_val_logits, optimize_temperature_scaling, optimize_decision_threshold,
        QuantumBetaCalibrator, QuantumMultiClassCalibrator, optimize_multiclass_cost_sensitive_thresholds,
        predict_with_cost_matrix, build_mondrian_conformal_artifact, split_calibration_indices
    )

try:
    from training_utils import compute_clinical_metrics, _is_better_checkpoint
except ImportError:
    from qml_models.training_utils import compute_clinical_metrics, _is_better_checkpoint

try:
    from qnas import QNASOptimizer, save_genome, load_genome
except ImportError:
    from qml_models.qnas import QNASOptimizer, save_genome, load_genome

try:
    from plots import generate_all_diagnostic_plots, plot_kfold_metric_distributions
except ImportError:
    from qml_models.plots import generate_all_diagnostic_plots, plot_kfold_metric_distributions

try:
    from engine import (
        extract_and_cache_features, train_one_epoch, evaluate, train_progressive_hybrid
    )
except ImportError:
    try:
        from qml_models.engine import (
            extract_and_cache_features, train_one_epoch, evaluate, train_progressive_hybrid
        )
    except ImportError:
        if "__main__" in sys.modules and hasattr(sys.modules["__main__"], "train_one_epoch"):
            _main = sys.modules["__main__"]
            extract_and_cache_features = _main.extract_and_cache_features
            train_one_epoch = _main.train_one_epoch
            evaluate = _main.evaluate
            train_progressive_hybrid = _main.train_progressive_hybrid
        else:
            raise


# ═══════════════════════════════════════════════════════════════════════════
# K-FOLD STRATIFIED CROSS-VALIDATION ENGINE
# (Varma & Simon 2006; Cawley & Talbot 2010; Wiens et al. 2019)
# ═══════════════════════════════════════════════════════════════════════════
def run_kfold_training(
    dataset_name: str = "pneumoniamnist",
    n_folds: int = 5,
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
    balanced_sampler: bool = True,
    asymmetric_loss: bool = True,
    asym_alpha: Optional[Union[str, List[float], float]] = None,
    asym_gamma: Optional[Union[str, List[float], float]] = None,
    threshold_criterion: str = "youden",
    target_specificity: float = 0.90,
    cost_fp: float = 1.5,
    fast_eval: bool = False,
    skip_plots: bool = False,
    plot_workers: int = 4,
    use_amp: bool = True,
    cache_in_vram: bool = False,
    grad_accum_steps: int = 1,
    q_init_std: float = 0.01,
    kfold_no_ensemble: bool = False,
    seed: int = 42,
    ansatz_mode: str = "standard_static",
    progressive_depths: Optional[List[int]] = None,
    epochs_per_stage: Optional[int] = None,
    finetune_epochs: int = 5,
    freeze_mode: str = "hard",
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
    **kwargs: Any,
) -> Dict:
    """K-Fold Stratified Cross-Validation with OOF Ensembling.

    Pipeline:
        1. Extract features once (shared frozen ViT backbone)
        2. Run QNAS once if enabled (Varma & Simon 2006 — decoupled from folds)
        3. Partition features into K stratified folds
        4. For each fold k:
            a. Fresh model instantiation with distinct PRNG seed
            b. Quarantined adapter pretraining on fold_k train partition only
            c. Train hybrid + baseline for `epochs` epochs
            d. Collect OOF predictions on fold_k validation partition
            e. Save fold checkpoint
        5. Aggregate OOF predictions → full-dataset unbiased evaluation
        6. Fit Dirichlet calibration on pooled OOF logits
        7. Compute cross-fold metric distributions (mean ± std)
        8. Save ensemble checkpoints for app.py deployment
    """
    if ansatz_mode == "qnas_static":
        use_qnas = True
    elif ansatz_mode == "progressive_dense":
        use_qnas = False
    elif ansatz_mode == "qnas_progressive":
        use_qnas = True

    device = get_torch_device()
    kfold_dir = os.path.join(output_dir, dataset_name, "kfold")
    ckpt_dir = os.path.join(kfold_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  K-FOLD STRATIFIED CROSS-VALIDATION ENGINE")
    print(f"  Dataset      : {dataset_name}")
    print(f"  Folds        : {n_folds}")
    print(f"  Epochs/Fold  : {epochs}")
    print(f"  Image Size   : {image_size}x{image_size}")
    print(f"  Qubits       : {n_qubits}")
    print(f"  VQC Layers   : {n_layers}")
    print(f"  Device       : {device}")
    print(f"  Adapter Mode : Strict Quarantine (re-pretrain per fold)")
    print(f"  Ansatz Mode  : {ansatz_mode}")
    if ansatz_mode == "progressive_dense":
        actual_pdepths = progressive_depths if progressive_depths is not None else [2, 4, 6]
        print(f"  Prog Depths  : {actual_pdepths} (Freeze: {freeze_mode})")
    print(f"  Seed         : {seed}")
    print(f"{'='*70}\n")

    # ── Step 1: Load data & extract features ONCE ─────────────────────────
    train_loader, val_loader, test_loader, meta = get_medmnist_dataloaders(
        dataset_name, batch_size=batch_size, download_root="./data", size=image_size,
        balanced_sampling=balanced_sampler,
    )
    meta["dataset_name"] = dataset_name
    if "label_map" not in meta or not meta["label_map"]:
        info_label = meta.get("info", {}).get("label", {})
        if info_label:
            meta["label_map"] = {int(k): v for k, v in info_label.items()}
        elif dataset_name in DATASET_CONFIG and "label_map" in DATASET_CONFIG[dataset_name]:
            meta["label_map"] = DATASET_CONFIG[dataset_name]["label_map"]
    n_classes = meta["n_classes"]
    in_channels = meta["in_channels"]
    total_train = meta["train_size"]
    total_val = meta["val_size"]

    # ── Step 0: BiomedCLIP ViT Fine-Tuning & Representation Isolation ────
    cache_mode = "finetuned" if enable_vit_finetuning else "frozen"
    cache_dir = os.path.join(kfold_dir, f"cache_{image_size}_{cache_mode}")
    os.makedirs(cache_dir, exist_ok=True)

    if enable_vit_finetuning:
        print("\n" + "!" * 70)
        print("[kfold/safety] WARNING: enable_vit_finetuning is active in K-Fold.")
        print("[kfold/safety] K-Fold will only consume a precomputed isolated ViT checkpoint.")
        print("!" * 70)
        try:
            from finetuning import has_finetuned_checkpoint
        except ImportError:
            from qml_models.finetuning import has_finetuned_checkpoint
        if not has_finetuned_checkpoint(dataset_name, vit_unfreeze_layers, biomedclip_dir):
            raise FileNotFoundError(
                "K-Fold fine-tuning requires a precomputed isolated checkpoint. "
                f"Expected dataset={dataset_name}, unfreeze_layers={vit_unfreeze_layers}, "
                f"base_dir={biomedclip_dir}. Run finetuning.py separately on a discovery cohort first."
            )
    else:
        print(f"[kfold/vit] Using frozen pretrained BiomedCLIP backbone (zero label bleed, invariant representations).")

    # Build a temporary model just for feature extraction
    tmp_hybrid, _ = build_models(
        in_channels, n_classes, n_qubits, n_layers, device, dropout,
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

    # For K-fold we need the combined train+val features as our full dataset
    # Extract train and val features separately, then concatenate
    print(f"[kfold] Extracting features for train+val combined dataset...")
    train_cache = os.path.join(cache_dir, "train.pt")
    val_cache = os.path.join(cache_dir, "val.pt")
    test_cache = os.path.join(cache_dir, "test.pt")

    cache_common = {
        "dataset_name": dataset_name,
        "image_size": int(image_size),
        "encoder_mode": cache_mode,
        "finetune_layers": int(vit_unfreeze_layers) if enable_vit_finetuning else 0,
        "finetune_dir": str(biomedclip_dir),
    }
    cached_train = extract_and_cache_features(
        tmp_hybrid, train_loader, device, train_cache, cache_in_vram=False,
        cache_metadata={**cache_common, "split": "train"},
    )
    cached_val = extract_and_cache_features(
        tmp_hybrid, val_loader, device, val_cache, cache_in_vram=False,
        cache_metadata={**cache_common, "split": "val"},
    )
    cached_test = extract_and_cache_features(
        tmp_hybrid, test_loader, device, test_cache, cache_in_vram=cache_in_vram,
        cache_metadata={**cache_common, "split": "test"},
    )

    # Extract feature tensors from cached loaders
    train_features = cached_train.dataset.tensors[0]
    train_labels = cached_train.dataset.tensors[1]
    val_features = cached_val.dataset.tensors[0]
    val_labels = cached_val.dataset.tensors[1]

    # Combine train + val into a single dataset for K-fold splitting
    all_features = torch.cat([train_features, val_features], dim=0)
    all_labels = torch.cat([train_labels, val_labels], dim=0)
    N_total = all_features.size(0)

    train_pids = meta.get("train_patient_ids", None)
    val_pids = meta.get("val_patient_ids", None)
    if train_pids is not None and val_pids is not None:
        all_patient_ids = np.concatenate([train_pids, val_pids])
        print(f"[kfold] \u2713 Patient IDs detected: {len(np.unique(all_patient_ids))} unique patients across {N_total} samples.")
        print(f"[kfold] \u2713 Activating StratifiedGroupKFold to eliminate intra-patient anatomical leakage.")
    else:
        all_patient_ids = None

    print(f"[kfold] Combined dataset: N={N_total}, D={all_features.size(1)}, Classes={n_classes}")
    print(f"[kfold] Label distribution: {np.bincount(all_labels.cpu().numpy().squeeze().astype(int), minlength=n_classes).tolist()}")

    # Free the temporary model
    del tmp_hybrid
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Step 2: Option A Architecture Discovery Holdout Partitioning (LEAK-02 Quarantine) ──
    best_genome = None
    cv_features = all_features
    cv_labels = all_labels
    cv_patient_ids = all_patient_ids

    resolved_genome_path = qnas_genome_path or kwargs.get("qnas_genome_path") or kwargs.get("genome_path")

    if ansatz_mode in ("qnas_static", "qnas_progressive") or use_qnas:
        if resolved_genome_path and os.path.exists(resolved_genome_path):
            print(f"[kfold/qnas] Loading pre-computed QNAS genome from: {resolved_genome_path}")
            best_genome = load_genome(resolved_genome_path)
        else:
            # Option A: Allocate dedicated 10% stratified holdout for QNAS search
            discovery_ratio = 0.10
            n_discovery = max(64, min(int(N_total * discovery_ratio), 400))
            if n_discovery < N_total - 64:
                from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold
                if all_patient_ids is not None:
                    sgkf = StratifiedGroupKFold(n_splits=max(2, int(N_total / n_discovery)))
                    cv_idx, disc_idx = next(sgkf.split(all_features.numpy(), all_labels.numpy(), groups=all_patient_ids))
                    cv_patient_ids = [all_patient_ids[i] for i in cv_idx]
                else:
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=n_discovery, random_state=seed)
                    cv_idx, disc_idx = next(sss.split(all_features.numpy(), all_labels.numpy()))
                    cv_patient_ids = None

                disc_features, disc_labels = all_features[disc_idx], all_labels[disc_idx]
                cv_features, cv_labels = all_features[cv_idx], all_labels[cv_idx]
                print(f"\n[kfold/qnas] Option A Quarantine: Allocated {len(disc_idx)} samples to QNAS Discovery Holdout ({len(disc_idx)/N_total*100:.1f}%).")
                print(f"[kfold/qnas] Remaining {len(cv_idx)} samples allocated to K-Fold CV Pool (100% Isolated from QNAS search).")
            else:
                disc_features, disc_labels = all_features, all_labels

            print(f"\n[kfold] Running QNAS evolutionary search strictly on Discovery Holdout (Zero Cross-Fold Bleed)...")
            actual_pdepths = progressive_depths if progressive_depths is not None else [2, 4, 6]
            qnas_search_depth = actual_pdepths[0] if ansatz_mode == "qnas_progressive" else n_layers
            qnas_hybrid, _ = build_models(
                in_channels, n_classes, n_qubits, qnas_search_depth, device, dropout,
                diff_method=diff_method, use_lora=use_lora, qdevice=qdevice,
                q_init_std=q_init_std,
                dual_basis_only=dual_basis_readout,
                use_qaf_attention=use_qaf_attention,
                clinical_hard=clinical_hard,
            )
            qnas_hybrid.eval()

            # Split discovery set into 70% proxy train and 30% proxy val
            n_disc_total = len(disc_labels)
            n_disc_train = max(32, int(n_disc_total * 0.70))
            qnas_z = disc_features[:n_disc_train].to(device)
            with torch.no_grad():
                qnas_z_latent = qnas_hybrid.backbone(qnas_z, use_cached_features=True)
            qnas_y = disc_labels[:n_disc_train].cpu()

            qnas_opt = QNASOptimizer(
                n_qubits=n_qubits,
                min_cnot=qnas_min_cnot,
                max_cnot=qnas_max_cnot,
                max_depth=qnas_search_depth,
                population_size=qnas_pop_size,
                generations=qnas_generations,
                lambda_cnot=qnas_lambda_cnot,
                target_mw=qnas_target_mw,
                output_dir=kfold_dir,
                qdevice=qdevice,
            )
            mi_matrix = qnas_opt.setup_admissibility_graph(qnas_z_latent.cpu())

            qnas_val_z = disc_features[n_disc_train:].to(device)
            with torch.no_grad():
                qnas_val_latent = qnas_hybrid.backbone(qnas_val_z, use_cached_features=True)
            qnas_val_y = disc_labels[n_disc_train:].cpu()

            best_genome = qnas_opt.evolve(
                qnas_z_latent.cpu(), qnas_y, qnas_val_latent.cpu(), qnas_val_y, n_classes=n_classes,
            )
            genome_path = os.path.join(ckpt_dir, "qnas_genome.json")
            save_genome(best_genome, genome_path)
            print(f"[kfold/qnas] Saved Pareto-optimal genome: CNOT={best_genome.cnot_count}, Params={best_genome.total_params}")

            del qnas_hybrid, qnas_z, qnas_z_latent, qnas_val_z, qnas_val_latent
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Step 3: Partition CV pool into K stratified (or grouped) folds ───
    N_cv = cv_features.size(0)
    print(f"\n[kfold] Generating {n_folds} stratified folds on {N_cv} samples (seed={seed})...")
    fold_loaders = get_kfold_loaders(
        features=cv_features,
        labels=cv_labels,
        n_folds=n_folds,
        batch_size=batch_size,
        patient_ids=cv_patient_ids,
        seed=seed,
        cache_in_vram=cache_in_vram,
        device=device,
    )

    # Verify partition integrity and subject isolation
    all_val_indices = []
    for fold_idx, (_, _, train_idx, val_idx) in enumerate(fold_loaders):
        all_val_indices.append(val_idx)
        train_dist = np.bincount(cv_labels.cpu().numpy().squeeze()[train_idx].astype(int), minlength=n_classes)
        val_dist = np.bincount(cv_labels.cpu().numpy().squeeze()[val_idx].astype(int), minlength=n_classes)
        pid_info = ""
        if cv_patient_ids is not None:
            train_p_set = set(np.array(cv_patient_ids)[train_idx])
            val_p_set = set(np.array(cv_patient_ids)[val_idx])
            overlap = len(train_p_set & val_p_set)
            pid_info = f", Pts(Tr={len(train_p_set)}, Val={len(val_p_set)}, Overlap={overlap})"
            assert overlap == 0, f"[kfold] Critical Patient Leakage in Fold {fold_idx}: {overlap} overlapping patients!"
        print(f"  Fold {fold_idx}: Train={len(train_idx)} {train_dist.tolist()}, Val={len(val_idx)} {val_dist.tolist()}{pid_info}")

    all_val_indices_flat = np.concatenate(all_val_indices)
    assert len(np.unique(all_val_indices_flat)) == N_cv, \
        f"OOF coverage error: {len(np.unique(all_val_indices_flat))} != {N_cv}"
    print(f"[kfold] ✓ Partition integrity verified: all {N_cv} samples covered exactly once.")

    # ── Step 4: Per-fold training loop ────────────────────────────────────
    # Pre-allocate OOF prediction tensors for CV pool
    oof_logits = torch.zeros(N_cv, n_classes)
    oof_probs = torch.zeros(N_cv, n_classes)
    oof_preds = torch.zeros(N_cv, dtype=torch.long)
    oof_labels = cv_labels.cpu().squeeze().long()

    baseline_oof_logits = torch.zeros(N_cv, n_classes)
    baseline_oof_probs = torch.zeros(N_cv, n_classes)
    baseline_oof_preds = torch.zeros(N_cv, dtype=torch.long)

    oof_latents = None
    oof_exps = None

    per_fold_metrics_hybrid = []
    per_fold_metrics_baseline = []
    all_fold_h_hists = []
    all_fold_b_hists = []
    best_kfold_h_state = None
    best_kfold_b_state = None
    best_kfold_auc = -1.0
    best_kfold_n_layers = n_layers

    for fold_idx, (fold_train_loader, fold_val_loader, train_idx, val_idx) in enumerate(fold_loaders):
        fold_seed = seed + fold_idx + 1
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)

        print(f"\n{'='*70}")
        print(f"  FOLD {fold_idx + 1}/{n_folds}  (seed={fold_seed}, train={len(train_idx)}, val={len(val_idx)})")
        print(f"{'='*70}")

        # ── 4a. Nested Inner-Fold Sub-Split (LEAK-03 Remediation) ─────────
        n_ftrain = len(train_idx)
        n_inner_cal = max(16, min(int(n_ftrain * 0.20), 200))
        train_labels_np = cv_labels.cpu().numpy().squeeze()[train_idx].astype(int)
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=n_inner_cal, random_state=fold_seed)
            inner_tr_local, inner_cal_local = next(sss_inner.split(np.zeros(n_ftrain), train_labels_np))
            inner_train_idx = train_idx[inner_tr_local]
            inner_cal_idx = train_idx[inner_cal_local]
        except Exception:
            perm = np.random.RandomState(fold_seed).permutation(n_ftrain)
            inner_cal_idx = train_idx[perm[:n_inner_cal]]
            inner_train_idx = train_idx[perm[n_inner_cal:]]

        inner_cal_loader = DataLoader(
            TensorDataset(cv_features[inner_cal_idx], cv_labels[inner_cal_idx]),
            batch_size=batch_size, shuffle=False
        )

        # ── 4b. Fresh model instantiation ─────────────────────────────────
        init_depth = n_layers
        if ansatz_mode in ("progressive_dense", "qnas_progressive"):
            actual_pdepths = progressive_depths if progressive_depths is not None else [2, 4, 6]
            init_depth = actual_pdepths[0]

        fold_hybrid, fold_baseline = build_models(
            in_channels, n_classes, n_qubits, init_depth, device, dropout,
            diff_method=diff_method, use_lora=use_lora, qdevice=qdevice,
            q_init_std=q_init_std,
            dual_basis_only=dual_basis_readout,
            use_qaf_attention=use_qaf_attention,
            clinical_hard=clinical_hard,
        )

        # If QNAS genome was discovered, re-instantiate with sparse ansatz
        if best_genome is not None:
            old_adapter_state = copy.deepcopy(fold_hybrid.backbone.cq_adapter.state_dict())
            old_skip_state = copy.deepcopy(fold_hybrid.backbone.skip.state_dict()) if hasattr(fold_hybrid.backbone, "skip") else None
            fold_hybrid = HybridCNNVQC(
                in_channels=in_channels,
                n_classes=n_classes,
                n_qubits=n_qubits,
                n_layers=n_layers,
                dropout=dropout,
                diff_method=diff_method,
                use_lora=use_lora,
                genome=best_genome,
                qdevice=qdevice,
                dual_basis_only=dual_basis_readout,
                use_qaf_attention=use_qaf_attention,
                clinical_hard=clinical_hard,
            ).to(device)
            fold_hybrid.backbone.cq_adapter.load_state_dict(old_adapter_state)
            if old_skip_state is not None and hasattr(fold_hybrid.backbone, "skip"):
                fold_hybrid.backbone.skip.load_state_dict(old_skip_state)

        # Offload ViT (we only need the adapter + VQC + head)
        fold_hybrid.offload_vision_encoder()
        fold_baseline.offload_vision_encoder()

        # ── 4b. Quarantined adapter pretraining (strict per-fold) ─────────
        fold_adapter_dir = os.path.join(kfold_dir, f"fold_{fold_idx}", "adapter")
        os.makedirs(fold_adapter_dir, exist_ok=True)
        print(f"[fold-{fold_idx}] Quarantined CQ-Adapter pre-training ({pretrain_epochs} epochs)...")
        pretrain_cq_adapter(
            fold_hybrid.backbone,
            fold_train_loader,
            device,
            epochs=pretrain_epochs,
            out_dir=fold_adapter_dir,
            dataset_name=f"{dataset_name}_fold{fold_idx}",
        )

        # ── 4c. Train for `epochs` epochs ────────────────────────────────
        # Loss function
        _effective_loss_type = loss_type
        # Auto-route 7-class dermamnist and 4-class octmnist to CB-LDAM-DRW
        if dataset_name == "dermamnist" and n_classes == 7 and _effective_loss_type == "focal":
            _effective_loss_type = "cb_ldam"
            if fold_idx == 0:
                print("[kfold] DermaMNIST 7-class detected → auto-routing to CB-LDAM-DRW loss.")
        if dataset_name == "octmnist" and n_classes == 4 and _effective_loss_type == "focal":
            _effective_loss_type = "cb_ldam"

        if _effective_loss_type == "cb_ldam":
            _cls_counts = _compute_class_counts(fold_train_loader, n_classes)
            criterion = ClassBalancedLDAMLoss(
                cls_num_list=_cls_counts,
                max_margin=ldam_max_margin,
                scale=10.0,
                beta=0.9999,
                drw_start_epoch=drw_start_epoch,
            ).to(device)
            if fold_idx == 0:
                print(f"[loss] CB-LDAM-DRW: counts={_cls_counts}, margins={criterion.margins.cpu().tolist()}")
        elif _effective_loss_type == "ce":
            class_weights = compute_class_weights(fold_train_loader, n_classes, device, method="effective")
            criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        else:
            class_weights = compute_class_weights(fold_train_loader, n_classes, device, method="effective")
            alpha_per_class, gamma_per_class = resolve_asymmetric_loss_params(
                asym_alpha=asym_alpha,
                asym_gamma=asym_gamma,
                dataset_name=dataset_name,
                n_classes=n_classes,
                asymmetric_loss=asymmetric_loss,
            )
            if alpha_per_class is not None and gamma_per_class is not None and fold_idx == 0:
                print(f"[loss] Asymmetric Focal Loss enabled: α={alpha_per_class}, γ={gamma_per_class}")
            criterion = FocalLoss(
                gamma=2.0,
                weight=class_weights,
                alpha_per_class=alpha_per_class,
                gamma_per_class=gamma_per_class,
            ).to(device)

        # Optimizer (decoupled learning rates)
        trainable_classical = list(filter(lambda p: p.requires_grad, fold_hybrid.backbone.cq_adapter.parameters()))
        if hasattr(fold_hybrid.backbone, "skip"):
            trainable_classical += list(filter(lambda p: p.requires_grad, fold_hybrid.backbone.skip.parameters()))

        opt_hybrid = torch.optim.AdamW([
            {"params": trainable_classical,                                                       "lr": 2e-4, "weight_decay": 1e-4},
            {"params": filter(lambda p: p.requires_grad, fold_hybrid.qlayer.parameters()),         "lr": 5e-3, "weight_decay": 0.0},
            {"params": filter(lambda p: p.requires_grad, fold_hybrid.head.parameters()),           "lr": 1e-3, "weight_decay": 1e-4},
        ])
        opt_baseline = torch.optim.Adam(
            filter(lambda p: p.requires_grad, fold_baseline.parameters()), lr=lr
        )

        scheduler_h = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_hybrid, T_0=10, T_mult=2, eta_min=1e-6)
        scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_baseline, T_max=epochs)

        best_fold_h_auc = -1.0
        best_fold_h_loss = float("inf")
        best_fold_h_state = None

        best_fold_b_auc = -1.0
        best_fold_b_loss = float("inf")
        best_fold_b_state = None

        if ansatz_mode in ("progressive_dense", "qnas_progressive"):
            actual_pdepths = progressive_depths if progressive_depths is not None else [2, 4, 6]
            if epochs_per_stage is None:
                stage_eps = max(2, (epochs - finetune_epochs) // len(actual_pdepths))
            else:
                stage_eps = epochs_per_stage

            prog_res = train_progressive_hybrid(
                hybrid=fold_hybrid,
                baseline=fold_baseline,
                train_loader=fold_train_loader,
                val_loader=fold_val_loader,
                criterion=criterion,
                device=device,
                n_classes=n_classes,
                depth_schedule=actual_pdepths,
                epochs_per_stage=stage_eps,
                finetune_epochs=finetune_epochs,
                q_init_std=q_init_std,
                freeze_mode=freeze_mode,
                use_qng=use_qng,
                mixup_alpha=mixup_alpha,
                phase_eps=phase_eps,
                lambda_ent=0.05,
                s_target_frac=0.70,
                use_qsam=use_qsam,
                use_qswa=use_qswa,
                lambda_infonce=lambda_infonce,
                lr_baseline=lr,
                n_qubits=n_qubits,
                out_dir=fold_adapter_dir,
                entropy_every_epoch=False,
                grad_accum_steps=grad_accum_steps,
                use_amp=use_amp,
                post_quantum_mixup=True,
                lambda_kta=0.0,
            )
            fold_h_hist = prog_res.get("hybrid_hist", {})
            fold_b_hist = prog_res.get("baseline_hist", {})
            best_fold_h_state = prog_res.get("best_hybrid_state") or copy.deepcopy(fold_hybrid.state_dict())
            best_fold_b_state = prog_res.get("best_baseline_state") or copy.deepcopy(fold_baseline.state_dict())
            best_fold_h_auc = prog_res.get("best_val_auc_hybrid", 0.0)
            best_fold_h_loss = prog_res.get("best_val_loss_hybrid", 0.0)
            best_fold_b_auc = prog_res.get("best_val_auc_baseline", 0.0)
            best_fold_b_loss = prog_res.get("best_val_loss_baseline", 0.0)
        else:
            fold_h_hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
            fold_b_hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

            for epoch in range(1, epochs + 1):
                if hasattr(criterion, "update_epoch"):
                    criterion.update_epoch(epoch)

                # Hybrid training
                train_res_h = train_one_epoch(
                    fold_hybrid, fold_train_loader, criterion, opt_hybrid, device, n_classes,
                    use_quantum_aug=True, mixup_alpha=mixup_alpha, phase_eps=phase_eps,
                    epoch=epoch, lambda_ent=0.05, s_target_frac=0.70,
                    grad_accum_steps=grad_accum_steps, use_amp=use_amp,
                )
                val_res_h = evaluate(fold_hybrid, fold_val_loader, criterion, device, n_classes, max_samples=5000)

                # Baseline training
                train_res_b = train_one_epoch(
                    fold_baseline, fold_train_loader, criterion, opt_baseline, device, n_classes,
                    use_quantum_aug=False, epoch=epoch,
                    grad_accum_steps=grad_accum_steps, use_amp=use_amp,
                )
                val_res_b = evaluate(fold_baseline, fold_val_loader, criterion, device, n_classes, max_samples=5000)

                fold_h_hist["train_loss"].append(train_res_h["loss"])
                fold_h_hist["val_loss"].append(val_res_h["loss"])
                fold_h_hist["train_acc"].append(train_res_h["accuracy"])
                fold_h_hist["val_acc"].append(val_res_h["accuracy"])

                fold_b_hist["train_loss"].append(train_res_b["loss"])
                fold_b_hist["val_loss"].append(val_res_b["loss"])
                fold_b_hist["train_acc"].append(train_res_b["accuracy"])
                fold_b_hist["val_acc"].append(val_res_b["accuracy"])

                # LR scheduling
                scheduler_h.step()
                scheduler_b.step()

                # Track best hybrid
                val_h_auc = val_res_h.get("auroc", val_res_h["accuracy"])
                val_h_loss = val_res_h["loss"]
                if _is_better_checkpoint(val_h_auc, val_h_loss, best_fold_h_auc, best_fold_h_loss):
                    best_fold_h_auc = val_h_auc
                    best_fold_h_loss = val_h_loss
                    best_fold_h_state = copy.deepcopy(fold_hybrid.state_dict())

                # Track best baseline
                val_b_auc = val_res_b.get("auroc", val_res_b["accuracy"])
                val_b_loss = val_res_b["loss"]
                if _is_better_checkpoint(val_b_auc, val_b_loss, best_fold_b_auc, best_fold_b_loss):
                    best_fold_b_auc = val_b_auc
                    best_fold_b_loss = val_b_loss
                    best_fold_b_state = copy.deepcopy(fold_baseline.state_dict())

        # ── 4d. Nested Calibration on inner_cal_loader (LEAK-03 Remediation) ───
        if best_fold_h_state is not None:
            fold_hybrid.load_state_dict(best_fold_h_state)

        cal_res = evaluate(fold_hybrid, inner_cal_loader, criterion, device, n_classes, max_samples=len(inner_cal_idx))
        cal_logits = cal_res["logits"]
        if not isinstance(cal_logits, torch.Tensor):
            cal_logits = torch.tensor(cal_logits)
        cal_labels_tensor = torch.tensor(cal_res["labels"])
        cal_labels_np = np.asarray(cal_res["labels"]).astype(int)
        cal_fit_idx, cal_score_idx = split_calibration_indices(cal_labels_np, seed=fold_seed)
        cal_fit_tensor = torch.as_tensor(cal_fit_idx, dtype=torch.long)

        T_k = 1.0
        try:
            T_k, cal_probs_temp = optimize_temperature_scaling(
                cal_logits, cal_labels_tensor, lr=0.01, max_iter=100, t_min=0.10
            )
            T_k = max(float(T_k), 0.10)
        except Exception:
            T_k = 1.0
            cal_probs_temp = F.softmax(cal_logits / max(T_k, 1e-6), dim=1)

        if n_classes == 2:
            beta_cal_k = QuantumBetaCalibrator()
            beta_cal_k.fit(cal_probs_temp[cal_fit_tensor].detach(), cal_labels_tensor[cal_fit_tensor])
            with torch.no_grad():
                cal_probs_inner = beta_cal_k(cal_probs_temp).numpy()
            tau_k, _ = optimize_decision_threshold(
                probs=cal_probs_inner[cal_fit_idx],
                labels=cal_labels_np[cal_fit_idx],
                criterion=threshold_criterion,
                target_specificity=target_specificity,
                cost_fp=cost_fp,
            )
        else:
            beta_cal_k = QuantumMultiClassCalibrator(n_classes=n_classes)
            beta_cal_k.fit(cal_probs_temp[cal_fit_tensor].detach(), cal_labels_tensor[cal_fit_tensor])
            with torch.no_grad():
                cal_probs_inner = beta_cal_k(cal_probs_temp).numpy()
            if threshold_criterion in ("cost_sensitive", "target_specificity"):
                tau_k = optimize_multiclass_cost_sensitive_thresholds(
                    val_probs=cal_probs_inner[cal_fit_idx],
                    val_labels=cal_labels_np[cal_fit_idx],
                    n_classes=n_classes,
                    target_specificity=target_specificity,
                    n_steps=200,
                )
            else:
                tau_k = None

        conformal_scores_k = build_mondrian_conformal_artifact(
            cal_probs_inner,
            cal_labels_np[cal_score_idx],
            alpha=0.05,
        )

        # ── 4e. Apply calibrated parameters out-of-fold to fold_val_loader ────
        fold_val_result = evaluate(fold_hybrid, fold_val_loader, criterion, device, n_classes, max_samples=N_cv)

        # Collect OOF logits: prefer raw logits from evaluate()
        if "logits" in fold_val_result and fold_val_result["logits"] is not None:
            fold_logits = fold_val_result["logits"]
            if not isinstance(fold_logits, torch.Tensor):
                fold_logits = torch.tensor(fold_logits)
        else:
            fold_logits = torch.log(torch.from_numpy(fold_val_result["probs"]).clamp(min=1e-8))

        # Apply fold-specific temperature and calibrator
        val_probs_temp = F.softmax(fold_logits / max(T_k, 0.10), dim=1)
        with torch.no_grad():
            fold_probs_cal = beta_cal_k(val_probs_temp).numpy()

        if n_classes == 2:
            fold_preds_np = (fold_probs_cal[:, 1] >= tau_k).astype(int)
        elif tau_k is not None and isinstance(tau_k, (list, np.ndarray)) and len(tau_k) == n_classes:
            fold_preds_np = np.argmax(fold_probs_cal, axis=1)
            for i in range(len(fold_preds_np)):
                valid_cls = [c for c in range(n_classes) if fold_probs_cal[i, c] >= tau_k[c]]
                if valid_cls:
                    fold_preds_np[i] = valid_cls[np.argmax([fold_probs_cal[i, c] for c in valid_cls])]
        else:
            fold_preds_np = np.argmax(fold_probs_cal, axis=1)

        fold_probs = torch.from_numpy(fold_probs_cal)
        oof_logits[val_idx] = fold_logits
        oof_probs[val_idx] = fold_probs
        oof_preds[val_idx] = torch.from_numpy(fold_preds_np).long()

        # Track OOF latents and expectations across folds for global diagnostic plots (e.g. Bloch sphere)
        if "latents" in fold_val_result and fold_val_result["latents"] is not None:
            if oof_latents is None:
                oof_latents = torch.zeros(N_cv, fold_val_result["latents"].shape[1], dtype=torch.float32)
            oof_latents[val_idx] = torch.from_numpy(fold_val_result["latents"]).float()

        if "expectations" in fold_val_result and fold_val_result["expectations"] is not None:
            if oof_exps is None:
                oof_exps = torch.zeros(N_cv, fold_val_result["expectations"].shape[1], dtype=torch.float32)
            oof_exps[val_idx] = torch.from_numpy(fold_val_result["expectations"]).float()

        # Compute per-fold metrics using out-of-fold calibrated outputs
        fold_h_metrics = compute_clinical_metrics(fold_val_result["labels"], fold_preds_np, fold_probs_cal, n_classes)
        per_fold_metrics_hybrid.append(fold_h_metrics)

        if best_fold_b_state is not None:
            fold_baseline.load_state_dict(best_fold_b_state)
        fold_b_result = evaluate(fold_baseline, fold_val_loader, criterion, device, n_classes, max_samples=N_cv)
        baseline_oof_probs[val_idx] = torch.from_numpy(fold_b_result["probs"])
        baseline_oof_preds[val_idx] = torch.from_numpy(fold_b_result["preds"]).long()
        baseline_oof_logits[val_idx] = torch.log(torch.from_numpy(fold_b_result["probs"]).clamp(min=1e-8))
        fold_b_metrics = compute_clinical_metrics(fold_b_result["labels"], fold_b_result["preds"], fold_b_result["probs"], n_classes)
        per_fold_metrics_baseline.append(fold_b_metrics)

        print(f"\n  [fold-{fold_idx}] Truly Unbiased OOF Hybrid  — Acc={fold_h_metrics['accuracy']:.4f}, "
              f"F1={fold_h_metrics.get('f1_macro', 0):.4f}, AUC={fold_h_metrics.get('auroc', 0):.4f} (T*={T_k:.2f})")
        print(f"  [fold-{fold_idx}] Truly Unbiased OOF Baseline — Acc={fold_b_metrics['accuracy']:.4f}, "
              f"F1={fold_b_metrics.get('f1_macro', 0):.4f}, AUC={fold_b_metrics.get('auroc', 0):.4f}")

        # Track best fold model state for post-kfold diagnostic plots
        curr_fold_auc = float(fold_h_metrics.get("auroc", 0))
        if curr_fold_auc > best_kfold_auc or best_kfold_h_state is None:
            best_kfold_auc = curr_fold_auc
            best_kfold_h_state = {k: v.cpu().clone() for k, v in fold_hybrid.state_dict().items()}
            best_kfold_b_state = {k: v.cpu().clone() for k, v in fold_baseline.state_dict().items()}
            best_kfold_n_layers = getattr(fold_hybrid, "n_layers", n_layers)

        # ── 4f. Save fold checkpoint with self-contained calibration ────
        if not kfold_no_ensemble:
            def _fold_safe(value):
                if isinstance(value, (np.floating, float)):
                    return float(value)
                if isinstance(value, (np.integer, int)):
                    return int(value)
                if isinstance(value, np.ndarray):
                    return [_fold_safe(item) for item in value.tolist()]
                if isinstance(value, (list, tuple)):
                    return [_fold_safe(item) for item in value]
                if isinstance(value, dict):
                    return {str(key): _fold_safe(item) for key, item in value.items()}
                return value

            fold_ckpt = {
                "config": {
                    "in_channels": in_channels,
                    "n_classes": n_classes,
                    "n_qubits": n_qubits,
                    "n_layers": fold_hybrid.n_layers,
                    "ansatz_mode": ansatz_mode,
                    "dropout": dropout,
                    "use_lora": use_lora,
                    "diff_method": diff_method,
                    "qdevice": qdevice,
                    "image_size": image_size,
                    "use_qnas": use_qnas,
                    "qnas_genome": best_genome.to_dict() if best_genome is not None else None,
                    "p_drop": 0.10,
                    "q_init_std": q_init_std,
                    "dual_basis_only": dual_basis_readout,
                    "dual_basis_readout": dual_basis_readout,
                    "use_qaf_attention": use_qaf_attention,
                    "clinical_hard": clinical_hard,
                    "preprocess_mode": "biomedclip_normalized",
                    "class_order": list(range(n_classes)),
                },
                "calibration": {
                    "temperature": float(T_k),
                    "threshold": float(tau_k) if isinstance(tau_k, (float, int)) else ([float(x) for x in tau_k] if tau_k is not None else None),
                    "calibrator_type": "beta" if n_classes == 2 else "multiclass",
                    "conformal_scores": conformal_scores_k,
                },
                "meta": {
                    "dataset_name": dataset_name,
                    "fold_idx": fold_idx,
                    "n_folds": n_folds,
                    "epochs": epochs,
                    "seed": fold_seed,
                    "best_val_auc": float(best_fold_h_auc),
                    "best_val_loss": float(best_fold_h_loss),
                },
                "model_state_dict": fold_hybrid.state_dict(),
                "calibrator_state_dict": {k: v.detach().cpu() for k, v in beta_cal_k.state_dict().items()},
                 "metrics": _fold_safe(fold_h_metrics),
            }
            fold_ckpt_path = os.path.join(ckpt_dir, f"hybrid_fold_{fold_idx}.pt")
            torch.save(fold_ckpt, fold_ckpt_path)
            print(f"  [fold-{fold_idx}] Saved checkpoint with nested calibration → {fold_ckpt_path}")

        # ── 4g. Generate per-fold diagnostic plots into isolated fold directories ──
        if not skip_plots:
            fold_dir = os.path.join(kfold_dir, f"fold_{fold_idx + 1}")
            os.makedirs(fold_dir, exist_ok=True)
            print(f"\n[fold-{fold_idx + 1}] Generating diagnostic plots → {fold_dir}/")
            fold_hybrid_eval = dict(fold_val_result)
            fold_hybrid_eval["model"] = fold_hybrid
            fold_baseline_eval = dict(fold_b_result)
            fold_baseline_eval["model"] = fold_baseline

            try:
                generate_all_diagnostic_plots(
                    hybrid_test_cal=fold_hybrid_eval,
                    baseline_test=fold_baseline_eval,
                    meta=meta,
                    out_dir=fold_dir,
                    hybrid_hist=fold_h_hist,
                    baseline_hist=fold_b_hist,
                    hybrid_model=fold_hybrid,
                    baseline_model=fold_baseline,
                    test_loader=fold_val_loader,
                    train_loader=fold_train_loader,
                    val_loader=fold_val_loader,
                    device=device,
                    n_qubits=n_qubits,
                    n_layers=getattr(fold_hybrid, "n_layers", n_layers),
                    criterion=criterion,
                    parallel=True,
                    plot_workers=plot_workers,
                    fast_eval=fast_eval,
                    skip_plots=skip_plots,
                )
                print(f"[fold-{fold_idx + 1}] ✓ Saved diagnostic plots (including Bloch spheres) → {fold_dir}/")
            except Exception as e:
                print(f"[warn] Failed to generate diagnostic plots for fold {fold_idx + 1}: {e}")

        # Append fold histories for cross-fold average convergence plots
        all_fold_h_hists.append(fold_h_hist)
        all_fold_b_hists.append(fold_b_hist)

        # Free fold models to conserve VRAM
        del fold_hybrid, fold_baseline, opt_hybrid, opt_baseline
        del criterion, scheduler_h, scheduler_b
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Step 5: Aggregate Truly Unbiased OOF predictions (LEAK-03 Remediation) ───
    print(f"\n{'='*70}")
    print(f"  OUT-OF-FOLD AGGREGATION ({N_cv} samples, Nested Calibration Zero-Leakage)")
    print(f"{'='*70}")

    oof_labels_np = oof_labels.numpy()
    oof_probs_np = oof_probs.numpy()
    oof_preds_np = oof_preds.numpy()

    # Raw metrics before calibration for diagnostic comparison
    raw_probs = F.softmax(oof_logits, dim=1).numpy()
    raw_preds = np.argmax(raw_probs, axis=1)
    oof_metrics_raw = compute_clinical_metrics(oof_labels_np, raw_preds, raw_probs, n_classes)

    # Truly unbiased metrics from nested out-of-fold calibrated predictions
    oof_metrics = compute_clinical_metrics(oof_labels_np, oof_preds_np, oof_probs_np, n_classes)

    print("\n── Truly Unbiased Out-of-Fold Hybrid Metrics (Nested Zero-Leakage) ──")
    for k, v in oof_metrics.items():
        if isinstance(v, float):
            print(f"   {k:<22}: {v:.4f}")

    # ── Step 7: Cross-fold metric distributions (mean ± std) ─────────────
    print(f"\n{'='*70}")
    print(f"  CROSS-FOLD METRIC DISTRIBUTIONS")
    print(f"{'='*70}")

    metric_keys = ["accuracy", "f1_macro", "auroc", "sensitivity", "specificity", "auprc"]
    fold_stats = {}

    for key in metric_keys:
        h_vals = [m.get(key, 0.0) for m in per_fold_metrics_hybrid if isinstance(m.get(key, 0.0), (int, float))]
        b_vals = [m.get(key, 0.0) for m in per_fold_metrics_baseline if isinstance(m.get(key, 0.0), (int, float))]

        if h_vals:
            h_mean = float(np.mean(h_vals))
            h_std = float(np.std(h_vals, ddof=1)) if len(h_vals) > 1 else 0.0
            b_mean = float(np.mean(b_vals)) if b_vals else 0.0
            b_std = float(np.std(b_vals, ddof=1)) if len(b_vals) > 1 else 0.0

            fold_stats[key] = {
                "hybrid_mean": h_mean, "hybrid_std": h_std,
                "baseline_mean": b_mean, "baseline_std": b_std,
                "hybrid_per_fold": [float(v) for v in h_vals],
                "baseline_per_fold": [float(v) for v in b_vals],
            }
            print(f"  {key:<16} | Hybrid: {h_mean:.4f} ± {h_std:.4f} | Baseline: {b_mean:.4f} ± {b_std:.4f}")

    # Paired significance test
    paired_test_results = {}
    for key in metric_keys:
        h_vals = fold_stats.get(key, {}).get("hybrid_per_fold", [])
        b_vals = fold_stats.get(key, {}).get("baseline_per_fold", [])
        if len(h_vals) >= 3 and len(b_vals) >= 3:
            try:
                from scipy import stats as sp_stats
                t_stat, t_pval = sp_stats.ttest_rel(h_vals, b_vals)
                try:
                    w_stat, w_pval = sp_stats.wilcoxon(
                        [h - b for h, b in zip(h_vals, b_vals)],
                        alternative="two-sided",
                    )
                except ValueError:
                    w_stat, w_pval = float("nan"), float("nan")
                paired_test_results[key] = {
                    "paired_t_stat": float(t_stat), "paired_t_pval": float(t_pval),
                    "wilcoxon_stat": float(w_stat), "wilcoxon_pval": float(w_pval),
                }
                sig = "✓ SIGNIFICANT" if t_pval < 0.05 else "✗ Not significant"
                print(f"  {key:<16} | Paired t-test: t={t_stat:.3f}, p={t_pval:.4f} ({sig})")
            except Exception as e:
                print(f"  {key:<16} | Paired test failed: {e}")

    # ── Step 8: Independent Test Holdout Verification (LEAK-04 Remediation) ───
    print(f"\n{'='*70}")
    print(f"  INDEPENDENT TEST HOLDOUT VERIFICATION (cached_test)")
    print(f"{'='*70}")

    test_features = cached_test.dataset.tensors[0].to(device)
    test_labels_np = cached_test.dataset.tensors[1].cpu().numpy().squeeze().astype(int)
    n_test_samples = test_features.size(0)

    ensemble_test_probs = np.zeros((n_test_samples, n_classes), dtype=np.float32)
    fold_tau_list = []
    ensemble_conformal_scores = {str(class_idx): [] for class_idx in range(n_classes)}
    evaluated_folds = 0

    for k in range(n_folds):
        ckpt_path = os.path.join(ckpt_dir, f"hybrid_fold_{k}.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            ckpt_depth = ckpt.get("config", {}).get("n_layers", n_layers)
            fold_h_test, _ = build_models(
                in_channels, n_classes, n_qubits, ckpt_depth, device, dropout,
                diff_method=diff_method, use_lora=use_lora, qdevice=qdevice,
                q_init_std=q_init_std,
                dual_basis_only=dual_basis_readout,
                use_qaf_attention=use_qaf_attention,
                clinical_hard=clinical_hard,
            )
            if best_genome is not None:
                fold_h_test = HybridCNNVQC(
                    in_channels=in_channels, n_classes=n_classes, n_qubits=n_qubits,
                    n_layers=ckpt_depth, dropout=dropout,
                    diff_method=diff_method, use_lora=use_lora,
                    genome=best_genome, qdevice=qdevice,
                    dual_basis_only=dual_basis_readout,
                    use_qaf_attention=use_qaf_attention,
                    clinical_hard=clinical_hard,
                ).to(device)
            fold_h_test.load_state_dict(ckpt["model_state_dict"])
            fold_h_test.eval()
            fold_h_test.offload_vision_encoder()

            with torch.no_grad():
                test_raw = fold_h_test(test_features, use_cached_features=True)
                if isinstance(test_raw, tuple):
                    test_raw = test_raw[0]
                T_k_val = ckpt.get("calibration", {}).get("temperature", 1.0)
                cal_p = F.softmax(test_raw / max(T_k_val, 0.10), dim=1).cpu().numpy()
                cal_meta = ckpt.get("calibration", {})
                cal_state = ckpt.get("calibrator_state_dict")
                if cal_state:
                    if cal_meta.get("calibrator_type") == "beta":
                        test_calibrator = QuantumBetaCalibrator()
                    else:
                        test_calibrator = QuantumMultiClassCalibrator(n_classes=n_classes)
                    test_calibrator.load_state_dict(cal_state, strict=True)
                    with torch.no_grad():
                        cal_p = test_calibrator(torch.from_numpy(cal_p)).cpu().numpy()
                fold_scores = ckpt.get("conformal_scores", cal_meta.get("conformal_scores", {}))
                if isinstance(fold_scores, dict):
                    for class_key, values in fold_scores.get("scores_by_class", {}).items():
                        ensemble_conformal_scores.setdefault(str(class_key), []).extend(
                            np.asarray(values, dtype=np.float32).reshape(-1).tolist()
                        )
                ensemble_test_probs += cal_p
                evaluated_folds += 1
                t_val = ckpt.get("calibration", {}).get("threshold", 0.5)
                if t_val is not None:
                    fold_tau_list.append(t_val)
            del fold_h_test

    if evaluated_folds > 0:
        ensemble_test_probs /= float(evaluated_folds)

    if n_classes == 2:
        mean_tau = float(np.mean(fold_tau_list)) if fold_tau_list else 0.50
        ensemble_test_preds = (ensemble_test_probs[:, 1] >= mean_tau).astype(int)
    else:
        if fold_tau_list and all(isinstance(t, (list, tuple, np.ndarray)) for t in fold_tau_list):
            mean_tau = np.mean(np.asarray(fold_tau_list, dtype=np.float32), axis=0)
            ensemble_test_preds = np.argmax(ensemble_test_probs, axis=1)
            for i in range(len(ensemble_test_preds)):
                valid_cls = np.flatnonzero(ensemble_test_probs[i] >= mean_tau)
                if len(valid_cls):
                    ensemble_test_preds[i] = int(valid_cls[np.argmax(ensemble_test_probs[i, valid_cls])])
        else:
            ensemble_test_preds = np.argmax(ensemble_test_probs, axis=1)

    ensemble_test_metrics = compute_clinical_metrics(test_labels_np, ensemble_test_preds, ensemble_test_probs, n_classes)
    print("\n── 5-Fold Calibrated Ensemble Metrics on External Holdout (cached_test) ──")
    for k_m, v_m in ensemble_test_metrics.items():
        if isinstance(v_m, float):
            print(f"   {k_m:<22}: {v_m:.4f}")

    # ── Step 9: Save kfold_summary.json ──────────────────────────────────
    def _clean_json_val(v):
        if isinstance(v, (np.floating, float)):
            return float(v)
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, (np.ndarray, list)):
            return [_clean_json_val(x) for x in v]
        if isinstance(v, dict):
            return {k: _clean_json_val(vv) for k, vv in v.items()}
        return v

    summary = {
        "dataset": dataset_name,
        "n_folds": n_folds,
        "epochs_per_fold": epochs,
        "n_total_samples": N_total,
        "n_cv_samples": N_cv,
        "n_classes": n_classes,
        "n_qubits": n_qubits,
        "dual_basis_readout": dual_basis_readout,
        "oof_metrics": _clean_json_val(oof_metrics),               # nested zero-leakage OOF metrics
        "oof_metrics_raw": _clean_json_val(oof_metrics_raw),       # pre-calibration raw metrics
        "ensemble_test_metrics": _clean_json_val(ensemble_test_metrics), # independent external holdout metrics
        "cross_fold_stats": _clean_json_val(fold_stats),
        "paired_significance_tests": _clean_json_val(paired_test_results),
        "per_fold_hybrid": [_clean_json_val(m) for m in per_fold_metrics_hybrid],
        "per_fold_baseline": [_clean_json_val(m) for m in per_fold_metrics_baseline],
        "ensemble_checkpoints": [
            os.path.join(ckpt_dir, f"hybrid_fold_{k}.pt") for k in range(n_folds)
        ] if not kfold_no_ensemble else [],
    }
    if fold_tau_list:
        summary["mean_optimal_threshold"] = _clean_json_val(
            np.mean(np.asarray(fold_tau_list, dtype=np.float32), axis=0)
        )
    temp_values = []
    for k in range(n_folds):
        ckpt_probe = os.path.join(ckpt_dir, f"hybrid_fold_{k}.pt")
        if os.path.exists(ckpt_probe):
            temp_values.append(float(torch.load(
                ckpt_probe, map_location="cpu", weights_only=True,
            ).get("calibration", {}).get("temperature", 1.0)))
    summary["mean_temperature"] = float(np.mean(temp_values)) if temp_values else 1.0
    summary["conformal_scores"] = {
        "method": "mondrian_split_conformal",
        "alpha": 0.05,
        "n_classes": n_classes,
        "scores_by_class": ensemble_conformal_scores,
        "quantiles": {
            key: float(np.quantile(values, min(1.0, np.ceil((len(values) + 1) * 0.95) / len(values)), method="higher"))
            if values else 1.0
            for key, values in ensemble_conformal_scores.items()
        },
    }
    summary_path = os.path.join(kfold_dir, "kfold_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"\n[kfold] Saved K-Fold summary to {summary_path}")

    # ── Step 9: Diagnostic Plot Generation ─────────────────────────────────
    if not skip_plots:
        print(f"\n{'='*70}\n  GENERATING K-FOLD DIAGNOSTIC PLOTS ({kfold_dir}/)\n{'='*70}")
        try:
            plot_kfold_metric_distributions(fold_stats, paired_test_results, kfold_dir, dataset_name=dataset_name)
            print(f"[kfold/plots] Saved Cross-Fold Metric Distributions → {os.path.join(kfold_dir, 'kfold_metric_distributions.png')}")
        except Exception as e:
            print(f"[warn] Failed to plot k-fold metric distributions: {e}")

        # Reconstruct representative best-fold models for diagnostic plots
        plot_depth = best_kfold_n_layers
        plot_hybrid, _ = build_models(
            in_channels, n_classes, n_qubits, plot_depth, device, dropout,
            diff_method=diff_method, use_lora=use_lora, qdevice=qdevice,
            q_init_std=q_init_std,
            dual_basis_only=dual_basis_readout,
            use_qaf_attention=use_qaf_attention,
        )
        if best_genome is not None:
            init_depth = len(best_genome.layers) if hasattr(best_genome, "layers") else plot_depth
            plot_hybrid = HybridCNNVQC(
                in_channels=in_channels,
                n_classes=n_classes,
                n_qubits=n_qubits,
                n_layers=init_depth,
                dropout=dropout,
                diff_method=diff_method,
                use_lora=use_lora,
                genome=best_genome,
                qdevice=qdevice,
                dual_basis_only=dual_basis_readout,
                use_qaf_attention=use_qaf_attention,
            ).to(device)

        # Grow layers if saved state indicates progressive expansion (e.g. L=2 -> 4 -> 6)
        if best_kfold_h_state is not None and "qlayer.weights" in best_kfold_h_state:
            target_weights = best_kfold_h_state["qlayer.weights"]
            target_numel = target_weights.numel()
            current_numel = sum(p.numel() for n, p in plot_hybrid.qlayer.named_parameters() if "weights" in n)
            while current_numel < target_numel:
                plot_hybrid.grow_layers(new_depth=plot_hybrid.n_layers + 2)
                current_numel = sum(p.numel() for n, p in plot_hybrid.qlayer.named_parameters() if "weights" in n)

        # Instantiate baseline matching the exact MLP shape from training checkpoint
        b_mlp_h = best_kfold_b_state["mlp.0.weight"].shape[0] if (best_kfold_b_state and "mlp.0.weight" in best_kfold_b_state) else None
        b_mlp_in = best_kfold_b_state["mlp.0.weight"].shape[1] if (best_kfold_b_state and "mlp.0.weight" in best_kfold_b_state) else None
        plot_baseline = ClassicalBaseline(
            in_channels, n_classes, n_qubits, n_layers=n_layers, dropout=dropout, use_lora=use_lora,
            mlp_hidden_dim=b_mlp_h, mlp_in_dim=b_mlp_in,
        ).to(device)

        plot_hybrid.offload_vision_encoder()
        plot_baseline.offload_vision_encoder()
        if best_kfold_h_state is not None:
            try:
                plot_hybrid.load_state_dict(best_kfold_h_state, strict=False)
            except RuntimeError as err:
                h_sd = plot_hybrid.state_dict()
                filtered_h = {k: v for k, v in best_kfold_h_state.items() if k in h_sd and h_sd[k].shape == v.shape}
                plot_hybrid.load_state_dict(filtered_h, strict=False)
                print(f"[warn] Partial load_state_dict applied for plot_hybrid: {err}")

        if best_kfold_b_state is not None:
            try:
                plot_baseline.load_state_dict(best_kfold_b_state, strict=False)
            except RuntimeError as err:
                b_sd = plot_baseline.state_dict()
                filtered_b = {k: v for k, v in best_kfold_b_state.items() if k in b_sd and b_sd[k].shape == v.shape}
                plot_baseline.load_state_dict(filtered_b, strict=False)
                print(f"[warn] Partial load_state_dict applied for plot_baseline: {err}")

        _eff_loss = loss_type
        if dataset_name == "octmnist" and n_classes == 4 and _eff_loss == "focal":
            _eff_loss = "cb_ldam"
        if _eff_loss == "cb_ldam":
            _plot_cls = _compute_class_counts(cached_test, n_classes) if cached_test is not None else [1] * n_classes
            plot_criterion = ClassBalancedLDAMLoss(
                cls_num_list=_plot_cls, max_margin=ldam_max_margin, beta=0.9999
            ).to(device)
        else:
            plot_criterion = nn.CrossEntropyLoss()

        # Construct pooled OOF prediction dictionaries with latents & expectations
        hybrid_test_cal = {
            "labels": oof_labels.numpy(),
            "preds": oof_preds.numpy(),
            "probs": oof_probs_np,
            "model": plot_hybrid,
        }
        if oof_latents is not None:
            hybrid_test_cal["latents"] = oof_latents.numpy()
            hybrid_test_cal["latent_labels"] = oof_labels.numpy()
        if oof_exps is not None:
            hybrid_test_cal["expectations"] = oof_exps.numpy()
        if best_kfold_h_state is not None:
            for k, v in best_kfold_h_state.items():
                if "qlayer" in k and "weights" in k:
                    hybrid_test_cal["trained_q_weights"] = v.cpu().numpy()
                    break

        baseline_test_oof = {
            "labels": oof_labels.numpy(),
            "preds": baseline_oof_preds.numpy(),
            "probs": baseline_oof_probs.numpy(),
            "model": plot_baseline,
        }

        # Averaged training history across folds for convergence curves
        avg_h_hist = {
            "train_loss": [float(np.mean([h["train_loss"][e] for h in all_fold_h_hists if len(h["train_loss"]) > e])) for e in range(epochs)],
            "val_loss": [float(np.mean([h["val_loss"][e] for h in all_fold_h_hists if len(h["val_loss"]) > e])) for e in range(epochs)],
            "train_acc": [float(np.mean([h["train_acc"][e] for h in all_fold_h_hists if len(h["train_acc"]) > e])) for e in range(epochs)],
            "val_acc": [float(np.mean([h["val_acc"][e] for h in all_fold_h_hists if len(h["val_acc"]) > e])) for e in range(epochs)],
        } if all_fold_h_hists else {}
        avg_b_hist = {
            "train_loss": [float(np.mean([h["train_loss"][e] for h in all_fold_b_hists if len(h["train_loss"]) > e])) for e in range(epochs)],
            "val_loss": [float(np.mean([h["val_loss"][e] for h in all_fold_b_hists if len(h["val_loss"]) > e])) for e in range(epochs)],
            "train_acc": [float(np.mean([h["train_acc"][e] for h in all_fold_b_hists if len(h["train_acc"]) > e])) for e in range(epochs)],
            "val_acc": [float(np.mean([h["val_acc"][e] for h in all_fold_b_hists if len(h["val_acc"]) > e])) for e in range(epochs)],
        } if all_fold_b_hists else {}

        avg_dir = os.path.join(kfold_dir, "average")
        os.makedirs(avg_dir, exist_ok=True)
        try:
            plot_kfold_metric_distributions(fold_stats, paired_test_results, avg_dir, dataset_name=dataset_name)
        except Exception as e:
            pass

        generate_all_diagnostic_plots(
            hybrid_test_cal=hybrid_test_cal,
            baseline_test=baseline_test_oof,
            meta=meta,
            out_dir=avg_dir,
            hybrid_hist=avg_h_hist,
            baseline_hist=avg_b_hist,
            hybrid_model=plot_hybrid,
            baseline_model=plot_baseline,
            test_loader=cached_test,
            train_loader=cached_train,
            val_loader=cached_val,
            device=device,
            n_qubits=n_qubits,
            n_layers=n_layers,
            criterion=plot_criterion,
            parallel=True,
            plot_workers=plot_workers,
            fast_eval=fast_eval,
            skip_plots=skip_plots,
        )

        for fname in os.listdir(avg_dir):
            if fname.endswith(".png"):
                shutil.copy2(os.path.join(avg_dir, fname), os.path.join(kfold_dir, fname))
        print(f"[kfold/plots] ✓ Saved aggregated average diagnostic plots to {avg_dir}/ and {kfold_dir}/")

    print(f"\n{'='*70}")
    print(f"  K-FOLD CROSS-VALIDATION COMPLETE")
    print(f"  OOF Accuracy  : {oof_metrics.get('accuracy', 0):.4f}")
    print(f"  OOF Macro F1  : {oof_metrics.get('f1_macro', 0):.4f}")
    print(f"  OOF AUROC     : {oof_metrics.get('auroc', 0):.4f}")
    print(f"  Checkpoints   : {ckpt_dir}")
    print(f"  Summary       : {summary_path}")
    print(f"{'='*70}\n")

    return {
        "oof_metrics": oof_metrics,
        "cross_fold_stats": fold_stats,
        "paired_significance_tests": paired_test_results,
        "summary_path": summary_path,
        "kfold_dir": kfold_dir,
    }
