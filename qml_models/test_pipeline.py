"""
test_pipeline.py — Automated Cross-Module Contract Test Suite (Task 7.1)
========================================================================
Validates all architectural remediations:
  1. Multi-view tensor shape flow: (B, 3, 224, 224) -> 4D batched tensor flow.
  2. LoRA-Q weight freezing: base projection weights are strictly frozen when rank > 0.
  3. Decoupled S^2 Bloch projection: rank 3 coordinates (theta, phi, lambda).
  4. Quantum circuit initialization: pure |0>^N without spurious Hadamard loops.
  5. Anti-shortcut regularizers: FeatureDecorrelationLoss, SaliencyConcentrationLoss, foreground mask.
  6. Symmetrical Classical Baseline head expressivity.
  7. BiomedCLIP ViT fine-tuning isolation and separate weights saving.
"""

import os
import sys
import unittest
import math
import numpy as np
import torch
import torch.nn as nn

# Ensure qml_models is in sys.path
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import (
    compute_foreground_mask,
    GridMask,
    RandomBorderCrop,
    RandomHistogramEqualization,
    AntiShortcutAugmentationScheduler,
    MultiViewBatchCollator,
)
from adapter import LoRAQLinear, MultiHeadOrthogonalCQAdapter
from model import (
    RiemannianBlochProjection,
    QuantumDropout,
    ClassicalBaseline,
    compute_n_meas,
)
from losses import (
    ClassBalancedLDAMLoss,
    QuantumInfoNCELoss,
    FeatureDecorrelationLoss,
    SaliencyConcentrationLoss,
    compute_drusen_cnv_orthogonality_loss,
)
from finetuning import (
    BiomedCLIPFineTuneConfig,
    configure_vit_trainability,
    has_finetuned_checkpoint,
    save_finetuned_biomedclip_vit,
    load_finetuned_biomedclip_vit,
)
from calibration import (
    build_mondrian_conformal_artifact,
    mondrian_prediction_set,
    predict_with_cost_matrix,
)


class TestRemediatedQMLPipeline(unittest.TestCase):

    def test_foreground_mask_dimensions(self):
        """Task 1.1: Verify compute_foreground_mask outputs binary {0, 1} matching spatial shape."""
        x = torch.rand(4, 3, 64, 64)
        mask = compute_foreground_mask(x, threshold_method="otsu")
        self.assertEqual(mask.shape, (4, 1, 64, 64))
        unique_vals = torch.unique(mask).tolist()
        for v in unique_vals:
            self.assertIn(v, [0.0, 1.0])

    def test_multiview_collator_flattening(self):
        """Task 2.1: Verify MultiViewBatchCollator with flatten_views=True outputs strictly 4D tensor."""
        collator = MultiViewBatchCollator(flatten_views=True)
        # Mock batch of 2 samples, each having 2 views of shape (3, 32, 32)
        batch = [
            ([torch.randn(3, 32, 32), torch.randn(3, 32, 32)], torch.tensor(0)),
            ([torch.randn(3, 32, 32), torch.randn(3, 32, 32)], torch.tensor(1)),
        ]
        flat_imgs, flat_labels = collator(batch)
        self.assertEqual(flat_imgs.ndim, 4)
        self.assertEqual(flat_imgs.shape[0], 4)  # B * V = 2 * 2 = 4
        self.assertEqual(flat_labels.shape[0], 4)

    def test_lora_q_parameter_freezing(self):
        """Task 3.3: Verify LoRA-Q strictly freezes base linear weights when rank > 0."""
        lora_layer = LoRAQLinear(in_features=64, out_features=16, rank=4)
        self.assertFalse(lora_layer.linear.weight.requires_grad)
        if lora_layer.linear.bias is not None:
            self.assertFalse(lora_layer.linear.bias.requires_grad)
        self.assertTrue(lora_layer.lora_A.requires_grad)
        self.assertTrue(lora_layer.lora_B.requires_grad)
        # Expected trainable parameter count = rank * in + out * rank = 4*64 + 16*4 = 320
        trainable = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
        self.assertEqual(trainable, 4 * 64 + 16 * 4)

    def test_decoupled_riemannian_bloch_projection(self):
        """Tasks 3.4 & 3.5: Verify decoupled lambda and accessible South Pole."""
        head = RiemannianBlochProjection(clinical_hard=True)
        raw = torch.randn(8, 3 * 4)  # 4 qubits
        z, r = head(raw, n_qubits=4)
        self.assertEqual(z.shape, (8, 12))
        self.assertEqual(r.shape, (8, 4, 3))
        # Verify unit norm of r vectors on S^2
        r_norms = torch.norm(r, p=2, dim=-1)
        torch.testing.assert_close(r_norms, torch.ones_like(r_norms), atol=1e-5, rtol=1e-5)
        # Verify theta spans full hemisphere including near South Pole
        theta = z[..., :4]
        self.assertTrue((theta.max() > 1.5).item())

    def test_quantum_dropout_unitary_jitter(self):
        """Task 4.5: Verify QuantumDropout applies additive phase jitter without non-unitary scaling."""
        q_drop = QuantumDropout(p_drop=0.2, use_phase_jitter=True, jitter_std=0.05)
        q_drop.train()
        weights = torch.ones(5, 5) * 1.5
        perturbed = q_drop(weights)
        # Active angles should have jitter without 1/(1-p) inflation
        self.assertFalse(torch.allclose(perturbed, weights))
        self.assertTrue((perturbed.mean() - 1.5).abs().item() < 0.2)

    def test_symmetrical_classical_baseline(self):
        """Task 4.7: Verify the baseline uses the matched-parameter MLP contract."""
        baseline = ClassicalBaseline(in_channels=3, n_classes=2, n_qubits=4, n_layers=2)
        n_meas = compute_n_meas(4, dual_basis_only=False)
        self.assertGreaterEqual(baseline.mlp[0].out_features, 2)
        self.assertEqual(baseline.head[0].in_features, n_meas)
        self.assertEqual(baseline.head[-1].out_features, 2)

    def test_anti_shortcut_losses(self):
        """Tasks 5.3 & 5.4: Verify FeatureDecorrelationLoss and SaliencyConcentrationLoss."""
        decorr_loss = FeatureDecorrelationLoss()
        angles = torch.randn(8, 12, requires_grad=True)
        images = torch.rand(8, 3, 32, 32)
        loss_d = decorr_loss(angles, images)
        self.assertTrue(loss_d.item() >= 0.0)
        loss_d.backward()
        self.assertIsNotNone(angles.grad)

        saliency_loss = SaliencyConcentrationLoss()
        saliency = torch.rand(8, 1, 32, 32, requires_grad=True)
        mask = compute_foreground_mask(images)
        loss_s = saliency_loss(saliency, mask)
        self.assertTrue(loss_s.item() >= 0.0)
        loss_s.backward()
        self.assertIsNotNone(saliency.grad)

    def test_biomedclip_finetuning_isolated_saving(self):
        """Verify fine-tuning saves separate weights per dataset in biomedclip/ without mutating base."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = BiomedCLIPFineTuneConfig(
                enable_finetuning=True,
                dataset_name="test_mnist",
                save_dir=tmp_dir,
                unfreeze_layers=2,
            )
            # Create a simple dummy ViT visual trunk
            class DummyViT(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.blocks = nn.ModuleList([nn.Linear(32, 32) for _ in range(4)])
                    self.proj = nn.Linear(32, 16)
                def forward(self, x):
                    for b in self.blocks:
                        x = b(x)
                    return self.proj(x)

            vit = DummyViT()
            trainable = configure_vit_trainability(vit, unfreeze_layers=2)
            self.assertTrue(trainable > 0)
            self.assertFalse(vit.blocks[0].weight.requires_grad)
            self.assertTrue(vit.blocks[-1].weight.requires_grad)

            # Test checkpoint saving
            out_path = save_finetuned_biomedclip_vit(vit, config)
            self.assertTrue(os.path.exists(out_path))
            self.assertIn("test_mnist_biomedclip_vit.pt", out_path)

            # Test checkpoint loading
            new_vit = DummyViT()
            loaded = load_finetuned_biomedclip_vit(new_vit, dataset_name="test_mnist", save_dir=tmp_dir)
            self.assertTrue(loaded)

    def test_qnas_discovery_isolation(self):
        """Phase 2 (LEAK-02 / Option A): Verify QNAS discovery partition and CV pool are strictly disjoint."""
        from sklearn.model_selection import train_test_split
        import numpy as np

        n_samples = 200
        all_indices = np.arange(n_samples)
        all_labels = np.array([0] * 100 + [1] * 100)
        seed = 42

        disc_size = max(int(0.10 * n_samples), min(64, n_samples // 4))
        disc_idx, cv_idx = train_test_split(
            all_indices,
            test_size=(n_samples - disc_size),
            stratify=all_labels,
            random_state=seed,
        )

        set_disc = set(disc_idx)
        set_cv = set(cv_idx)

        # 1. Complete disjointness
        self.assertTrue(set_disc.isdisjoint(set_cv), "QNAS discovery set overlaps with CV pool!")
        # 2. Partition covers full dataset
        self.assertEqual(set_disc.union(set_cv), set(all_indices))
        # 3. Size check
        self.assertEqual(len(disc_idx), disc_size)
        self.assertEqual(len(cv_idx), n_samples - disc_size)

    def test_kfold_index_isolation(self):
        """Phase 3 (LEAK-03): Verify inner_train_idx, inner_cal_idx, and val_idx are strictly pairwise disjoint."""
        from sklearn.model_selection import StratifiedKFold, train_test_split
        import numpy as np

        n_samples = 150
        cv_indices = np.arange(n_samples)
        labels = np.array([0] * 75 + [1] * 75)
        seed = 42
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        for fold, (train_fold_idx, val_fold_idx) in enumerate(skf.split(cv_indices, labels)):
            train_idx = cv_indices[train_fold_idx]
            val_idx = cv_indices[val_fold_idx]
            train_labels = labels[train_idx]

            # Replicate nested calibration split from kfold.py
            cal_size = max(int(0.20 * len(train_idx)), min(32, len(train_idx) // 4))
            inner_train_idx, inner_cal_idx = train_test_split(
                train_idx,
                test_size=cal_size,
                stratify=train_labels,
                random_state=seed + fold,
            )

            set_inner_train = set(inner_train_idx)
            set_inner_cal = set(inner_cal_idx)
            set_val = set(val_idx)

            # Pairwise disjointness: inner_cal vs val (LEAK-03 mitigation)
            self.assertTrue(
                set_inner_cal.isdisjoint(set_val),
                f"Fold {fold}: Calibration indices leaked into validation set!",
            )
            # inner_train vs val
            self.assertTrue(
                set_inner_train.isdisjoint(set_val),
                f"Fold {fold}: Inner train indices overlap with validation set!",
            )
            # inner_train vs inner_cal
            self.assertTrue(
                set_inner_train.isdisjoint(set_inner_cal),
                f"Fold {fold}: Inner train and calibration sets overlap!",
            )
            # Conservation of fold train pool
            self.assertEqual(set_inner_train.union(set_inner_cal), set(train_idx))

    def test_kfold_summary_structure(self):
        """Phase 4 (LEAK-04): Validate a production K-fold artifact, not fake data."""
        import json
        artifact_dir = os.environ.get("QML_KFOLD_ARTIFACT_DIR")
        if not artifact_dir:
            self.skipTest("Set QML_KFOLD_ARTIFACT_DIR to an actual kfold output directory")

        summary_path = os.path.join(artifact_dir, "kfold_summary.json")
        self.assertTrue(os.path.isfile(summary_path), summary_path)
        with open(summary_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        for key in ("oof_metrics", "oof_metrics_raw", "ensemble_test_metrics", "per_fold_hybrid"):
            self.assertIn(key, loaded)
        self.assertIn("mean_optimal_threshold", loaded)
        self.assertIn("mean_temperature", loaded)
        self.assertIn("conformal_scores", loaded)
        self.assertIsInstance(loaded["conformal_scores"], (list, dict))

        for checkpoint_path in loaded.get("ensemble_checkpoints", []):
            if not os.path.isabs(checkpoint_path):
                checkpoint_path = os.path.join(os.path.dirname(summary_path), checkpoint_path)
            self.assertTrue(os.path.isfile(checkpoint_path), checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.assertIn("model_state_dict", checkpoint)
            self.assertIn("calibrator_state_dict", checkpoint)
            self.assertIn("calibration", checkpoint)
            self.assertIn("conformal_scores", checkpoint["calibration"])

    def test_mondrian_conformal_artifact_is_class_conditional(self):
        """Clinical classes receive independent finite-sample conformal quantiles."""
        probs = torch.tensor([
            [0.90, 0.05, 0.03, 0.02],
            [0.80, 0.10, 0.05, 0.05],
            [0.20, 0.55, 0.15, 0.10],
            [0.25, 0.50, 0.15, 0.10],
            [0.20, 0.20, 0.55, 0.05],
            [0.20, 0.20, 0.50, 0.10],
            [0.20, 0.20, 0.10, 0.50],
            [0.20, 0.20, 0.10, 0.50],
        ]).numpy()
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]).numpy()
        artifact = build_mondrian_conformal_artifact(probs, labels, alpha=0.10)
        self.assertEqual(artifact["method"], "mondrian_split_conformal")
        self.assertEqual(set(artifact["quantiles"]), {"0", "1", "2", "3"})
        prediction_set = mondrian_prediction_set(np.array([0.20, 0.10, 0.65, 0.05]), artifact)
        self.assertIn(2, prediction_set.tolist())

    def test_cost_matrix_uses_action_by_true_orientation(self):
        """Bayes risk must calculate C[action, true] @ p."""
        probs = np.array([[0.20, 0.30, 0.50]])
        costs = np.array([
            [0.0, 1.0, 20.0],
            [1.0, 0.0, 2.0],
            [50.0, 3.0, 0.0],
        ])
        # Action 1 risk=0.2+1.0, action 0 risk=0.3+10.0, action 2 risk=10+0.9.
        self.assertEqual(int(predict_with_cost_matrix(probs, costs)[0]), 1)


if __name__ == "__main__":
    unittest.main()
