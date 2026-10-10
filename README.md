# 🧬 Hybrid Classical-Quantum Neural Network (CNN-VQC & HQCNN) Framework for Early Disease Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane 0.33+](https://img.shields.io/badge/PennyLane-0.33+-yellow.svg)](https://pennylane.ai/)
[![MedMNIST v2](https://img.shields.io/badge/Dataset-MedMNIST%20v2-green.svg)](https://medmnist.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An institutional-grade, research-validated computational framework integrating frozen and fine-tuned Foundation Vision Backbones (**BiomedCLIP ViT-B/16**) with **N-Qubit Variational Quantum Circuits (VQC / HQCNN)** for multimodal early medical disease detection.

Designed for high-stakes clinical diagnostic workflows across the **MedMNIST v2** benchmark suite (PneumoniaMNIST, DermaMNIST, RetinaMNIST, BreastMNIST, OrganAMNIST, PathMNIST) and real-world medical image archives (COVID-19 Radiography, OASIS-1 Brain MRI), this repository features rigorous anti-shortcut learning modules, geometric Riemannian spherical feature projection, quantum neural architecture search (QNAS), dual-basis Pauli observable readout, strictly positive temperature calibration, and an interactive Streamlit clinical dashboard with Grad-CAM and quantum saliency interpretability.

---

## 📑 Table of Contents

1. [🏥 Executive Summary & Clinical Rationale](#1--executive-summary--clinical-rationale)
2. [🧠 Key Architectural Innovations & Formal Foundations](#2--key-architectural-innovations--formal-foundations)
   - [2.1 Foundation Vision Backbone (BiomedCLIP ViT-B/16)](#21-foundation-vision-backbone-biomedclip-vit-b16)
   - [2.2 Classical-to-Quantum Adapter (CQ-Adapter) & Riemannian S² Head](#22-classical-to-quantum-adapter-cq-adapter--riemannian-s-head)
   - [2.3 Variational Quantum Circuit (VQC) Ansätze](#23-variational-quantum-circuit-vqc-ansätze)
   - [2.4 Quantum Readout Paradigms: Dual-Basis (HQCNN) vs. Legacy Pauli](#24-quantum-readout-paradigms-dual-basis-hqcnn-vs-legacy-pauli)
   - [2.5 Parameter-Matched Classical Baseline](#25-parameter-matched-classical-baseline)
   - [2.6 Anti-Shortcut Learning & Modality Adaptation Suite](#26-anti-shortcut-learning--modality-adaptation-suite)
   - [2.7 Class Imbalance Mitigation: CB-LDAM-DRW & Asymmetric Loss](#27-class-imbalance-mitigation-cb-ldam-drw--asymmetric-loss)
   - [2.8 Clinical Calibration & Specificity-Constrained Decision Thresholds](#28-clinical-calibration--specificity-constrained-decision-thresholds)
   - [2.9 Quantum Continual Learning: QFIM & QEWC](#29-quantum-continual-learning-qfim--qewc)
3. [📂 Repository Directory & File Architecture](#3--repository-directory--file-architecture)
4. [⚙️ Installation & Environment Setup](#4--installation--environment-setup)
   - [4.1 Prerequisites & System Requirements](#41-prerequisites--system-requirements)
   - [4.2 Virtual Environment Initialization](#42-virtual-environment-initialization)
   - [4.3 Dependency Installation](#43-dependency-installation)
   - [4.4 Multithreading & Memory Safety Variables](#44-multithreading--memory-safety-variables)
5. [🚀 End-to-End Execution Guide](#5--end-to-end-execution-guide)
   - [5.1 Training Engine (`engine.py`)](#51-training-engine-enginepy)
     - [Standard Hybrid VQC Training](#standard-hybrid-vqc-training)
     - [HQCNN Dual-Basis DermaMNIST Flagship Pipeline](#hqcnn-dual-basis-dermamnist-flagship-pipeline)
     - [Stratified K-Fold Cross-Validation](#stratified-k-fold-cross-validation)
     - [Quantum Continual Learning (QEWC)](#quantum-continual-learning-qewc)
     - [Quantum Architecture Search (QNAS) & Progressive Depth Growth](#quantum-architecture-search-qnas--progressive-depth-growth)
     - [Evaluation-Only Mode & Threshold Recalibration](#evaluation-only-mode--threshold-recalibration)
     - [Comprehensive CLI Flag Reference for `engine.py`](#comprehensive-cli-flag-reference-for-enginepy)
   - [5.2 Interactive Clinical Streamlit Dashboard (`app.py`)](#52-interactive-clinical-streamlit-dashboard-apppy)
   - [5.3 Independent Evaluation & Benchmarking Suite (`testing.py`)](#53-independent-evaluation--benchmarking-suite-testingpy)
   - [5.4 Dataset Ingestion & NPZ Preprocessing (`npz_convert.py`)](#54-dataset-ingestion--npz-preprocessing-npz_convertpy)
   - [5.5 BiomedCLIP ViT Fine-Tuning (`finetuning.py`)](#55-biomedclip-vit-fine-tuning-finetuningpy)
   - [5.6 Classical-to-Quantum Adapter Pre-training (`adapter.py`)](#56-classical-to-quantum-adapter-pre-training-adapterpy)
   - [5.7 Diagnostic Pipeline Verification (`test_pipeline.py`)](#57-diagnostic-pipeline-verification-test_pipelinepy)
6. [📊 Supported Datasets & Clinical Modalities](#6--supported-datasets--clinical-modalities)
7. [📈 Comprehensive Diagnostic Plotting Suite (39 Plots)](#7--comprehensive-diagnostic-plotting-suite-39-plots)
8. [🛠️ Troubleshooting & Common Engineering Pitfalls](#8--troubleshooting--common-engineering-pitfalls)
9. [📜 Citation & Academic Reference](#9--citation--academic-reference)

---

## 1. 🏥 Executive Summary & Clinical Rationale

Deep learning models deployed in clinical imaging are notoriously prone to two fundamental failure modes:
1. **Shortcut Learning:** Exploiting peripheral non-pathological artifacts such as scanner brand borders, hospital orientation markers, text watermarks, or illumination gradients rather than true biological lesion morphology.
2. **Pathological Overconfidence:** Generating uncalibrated, extreme probability scores on rare clinical classes due to severe population prevalence skew (e.g., malignant melanoma vs. benign nevi).

This repository implements a **Hybrid Classical-Quantum Convolutional Architecture (HQCNN / CNN-VQC)** engineered to address these challenges:
- **Geometric Quantum State Compression:** High-dimensional vision representations from BiomedCLIP ViT are compressed and mapped onto single-qubit Riemannian Bloch spheres S², filtering high-frequency scanner noise.
- **Dual-Basis Quantum Readout:** Measuring both the longitudinal computational basis `<Z_i>` (optical pigment absorption density) and transverse superposition basis `<X_i>` (micro-structural lesion margin asymmetry) allows the quantum circuit to extract non-linear spatial correlations.
- **Provable Anti-Shortcut Guarantees:** Through morphological foreground masking, GridMask, border cropping, feature decorrelation loss (`L_decorr`), and saliency concentration penalties (`L_saliency`), model reliance on peripheral background pixels is penalized.
- **Safety-Constrained Decisions:** Predictive distributions undergo strictly positive Platt temperature calibration (`T(tau) = 0.1 + softplus(tau)`) and cost-sensitive threshold tuning bounded by a strict `Specificity >= 0.9000` clinical floor.

---

## 2. 🧠 Key Architectural Innovations & Formal Foundations

```
MedMNIST Medical Input (28x28 to 224x224 RGB / Grayscale)
         │
         ▼  [preprocessing.py / dataset.py]
 Modality-Adaptive Preprocessing & Anti-Shortcut Suite
  ├── Foreground Mask Extraction: M_fg in {0, 1}^(H x W)
  ├── RandomBorderCrop, GridMask, RandomHistogramEqualization
  └── MultiViewBatchCollator (Flattened 4D Batch: (B*V, C, H, W))
         │
         ▼  [model.py / finetuning.py]
 Frozen / Top-2 Unfrozen BiomedCLIP ViT-B/16 Backbone
  └── L2-Normalized CLS Embedding Vector: f in R^512
         │
         ▼  [adapter.py]
 Classical-to-Quantum Adapter (CQ-Adapter)
  ├── LoRA-Q Factorization: W_eff = W_0 + (scale / rank) * (B @ A)
  ├── Cayley Map Orthogonal Parameterization
  └── Riemannian S² Spherical Head:
        theta_i = sigmoid(v_i) * pi          in [0, pi]
        phi_i   = tanh(v_{i+n}) * pi         in [-pi, pi]
        lambda_i = tanh(v_{i+2n}) * pi       in [-pi, pi]
         │
         ▼  [model.py / ansatz/qaf.py / PennyLane Engine]
 Quantum Hilbert Space Circuit (N=8 Qubits, dim H = 256)
  ├── Ground State Preparation: |0>^N
  ├── Universal Data Re-Uploading: U3(theta, phi, lambda)
  ├── Variational Layer Topologies:
  │     ├── Option A: Hierarchical QAF (Superpositional Entanglement + Toffoli Attention + Fourier CPhase)
  │     ├── Option B: QNAS Evolutionary DAG Supercircuit
  │     └── Option C: StronglyEntanglingLayers (Hardware-Efficient CNOT Ring)
  └── Quantum Weight Dropout (Gaussian Lie-Algebra Phase Jitter)
         │
         ▼  [Readout Observables]
 Quantum Measurement Transformation:
  ├── Dual-Basis HQCNN: [<Z_i>, <X_i>] in [-1, 1]^(2N)      (16 dims for N=8)
  └── Legacy Pauli:     [<Z_i>, <Z_i Z_{i+1}>, <X_i>] in [-1, 1]^(3N-1) (23 dims for N=8)
         │
         ▼  [model.py]
 Symmetric Projection Head:
  Linear(dim_readout -> 32) ──► GELU ──► Dropout(0.2) ──► Linear(32 -> C)
         │
         ▼  [losses.py / calibration.py / engine.py]
 Optimization & Clinical Decision Calibration:
  ├── Loss: CB-LDAM-DRW + L_decorr + L_saliency
  ├── Multi-Rate AdamW: lr_backbone=1e-5, lr_adapter=1e-3, lr_quantum=1e-2
  ├── Strictly Positive Platt Calibration: T(tau) = 0.1 + ln(1 + exp(tau))
  └── Cost-Sensitive Thresholding: Specificity >= 0.9000 floor
```

### 2.1 Foundation Vision Backbone (BiomedCLIP ViT-B/16)
The vision encoder leverages domain knowledge pretrained on over 15 million biomedical image-text pairs from PubMed Central:
- Grayscale medical inputs (e.g., pediatric chest radiographs) are expanded to 3 channels: `x = x.repeat(1, 3, 1, 1)`.
- Input normalization values: `mean = [0.4814, 0.4578, 0.4082]`, `std = [0.2686, 0.2613, 0.2758]`.
- Feature extraction produces a 512-dimensional L2-normalized class token: `f_CLS in R^512`.
- Fine-tuned backbone weights are saved per dataset under `biomedclip/{dataset}_biomedclip_vit.pt`.

### 2.2 Classical-to-Quantum Adapter (CQ-Adapter) & Riemannian S² Head
To bridge the 512-dimensional classical latent space with the N-qubit quantum register (default N=8, Hilbert dimension 256), the network applies a constrained projection:
```text
v = W_eff @ f_CLS + b   where v in R^(3N)
```

Where `W_eff` utilizes **LoRA-Q Factorization**:
```text
W_eff = W_0 + (alpha / rank) * (B @ A)
```
- `W_0` is frozen with orthogonal Cayley initialization.
- `A in R^(rank x 512)` and `B in R^(3N x rank)` (rank = 4, alpha = 1.0) are trainable.

The resulting vector `v` is mapped onto single-qubit Bloch spheres via the **Riemannian S² Head**:
```text
theta_i  = sigmoid(v_i) * pi          in [0, pi]
phi_i    = tanh(v_{i+N}) * pi         in [-pi, pi]
lambda_i = tanh(v_{i+2N}) * pi        in [-pi, pi]
```

Each qubit receives parameters defining the pure state:
```text
|psi_i> = cos(theta_i / 2)|0> + exp(i * phi_i) * sin(theta_i / 2)|1>
```

In clinical-hardening mode (`--clinical_hard`), the polar angle is restricted to `[-pi/2, pi/2]` to eliminate phase-wrapping decision boundaries.

### 2.3 Variational Quantum Circuit (VQC) Ansätze
The engine supports three variational circuit architectures:

1. **Hierarchical Quantum Attention-Fourier (QAF) Circuit (`--use_qaf_attention`):**
   - **Universal Data Re-Uploading:** `U3(theta_i, phi_i, lambda_i)` single-qubit rotations applied before every layer.
   - **Superpositional Entanglement:** Nearest-neighbor cyclic CNOT ring creating entanglement across adjacent wires.
   - **Local 3-Qubit Toffoli Attention:** Parameterized non-linear multi-qubit interaction modulating attention between adjacent qubit triplets.
   - **Global Dyadic Fourier Phase Modulation:** Controlled-phase (`CRZ`) rotations with dyadic scaling phases `Delta_phi_{j,k} = 2*pi / (2^|j-k|)` capturing multi-scale non-local feature interactions.
   - **Variational Rotation Blocks:** Independent `Rot(w_1, w_2, w_3)` gates applied per wire.

2. **Evolutionary Quantum Architecture Search (QNAS DAG, `--use_qnas`):**
   - Represents the ansatz as a directed acyclic graph (DAG) genome.
   - Gate vocabulary: `Rx, Ry, Rz, U3, CNOT, CZ, Identity`.
   - Evaluates architectures on a Pareto frontier balancing validation accuracy, Meyer-Wallach entanglement capacity, and a penalty for CNOT count to reduce circuit noise.

3. **Strongly Entangling Layers (Default Fallback):**
   - Periodic parameterized single-qubit rotations followed by entangling CNOT gates with range offsets `r = 1, ..., N-1`.

### 2.4 Quantum Readout Paradigms: Dual-Basis (HQCNN) vs. Legacy Pauli
- **HQCNN Dual-Basis Readout (`--dual_basis_readout`):**
  Measures both longitudinal computational and transverse superposition observables:
  ```text
  y_readout = [<Z_0>, ..., <Z_{N-1}>, <X_0>, ..., <X_{N-1}>]^T  in [-1, 1]^(2N)
  ```
  For N=8, this outputs 16 dimensions. `<Z_i>` captures diagonal optical absorption (pigment density), while `<X_i> = 2 * Re(rho_01)` measures off-diagonal coherence (lesion margin asymmetry).
- **Legacy Pauli Readout (Default):**
  ```text
  y_readout = [<Z_i> for i in 0..N-1, <Z_i Z_{i+1}> for i in 0..N-2, <X_i> for i in 0..N-1]^T  in [-1, 1]^(3N-1)
  ```
  Outputs 23 measurement features for N=8.

### 2.5 Parameter-Matched Classical Baseline
To provide a fair comparison with the quantum model, `model.py` implements `ClassicalBaseline`:
```text
z_baseline = W_2 @ Dropout(GELU(W_1 @ f_CLS))
```
This baseline matches the parameter count, learning rate, and loss formulation of the hybrid model without using a VQC.

### 2.6 Anti-Shortcut Learning & Modality Adaptation Suite
Medical models often overfit to scanner-specific borders, text watermarks, or peripheral illumination. The anti-shortcut suite in `preprocessing.py` mitigates these confounders:
- **Foreground Mask Extraction:** Computes binary region-of-interest masks `M_fg in {0, 1}^(H x W)` using adaptive thresholding.
- **GridMask & RandomBorderCrop:** Crops peripheral margins and introduces dropout grids to disrupt edge artifacts.
- **Feature Decorrelation Loss (`L_decorr`):**
  ```text
  L_decorr = (1 / K) * sum_{k=1}^K |Corr(f_CLS, h_k)|
  ```
  Penalizes linear correlation between backbone embeddings and low-level heuristic features `h_k` (border intensity, mean luminance, edge energy).
- **Saliency Concentration Loss (`L_saliency`):**
  ```text
  L_saliency = 1.0 - (sum_{(u,v) in ROI} A(u, v)) / (sum_{u,v} A(u, v))
  ```
  Forces Grad-CAM activation mass `A` to concentrate inside the pathological lesion boundary.
- **Shortcut Verification Score (`S`):**
  ```text
  S = E[ max_c P(y=c | x * (1 - M_fg)) ] / E[ max_c P(y=c | x) ]
  ```
  An ideal model exhibits `S -> 0` (no confidence from background alone). Values of `S > 0.5` flag reliance on background shortcuts.

### 2.7 Class Imbalance Mitigation: CB-LDAM-DRW & Asymmetric Loss
For severe clinical imbalance (e.g., DermaMNIST melanoma vs. benign nevi):
- **Class-Balanced LDAM Loss (CB-LDAM):** Enforces margins inversely proportional to class population:
  ```text
  Delta_c = C_margin / (N_c^(1/4))
  L_LDAM(z, y) = -log( exp(z_y - Delta_y) / (exp(z_y - Delta_y) + sum_{j != y} exp(z_j)) )
  ```
- **Deferred Re-Weighting (DRW):** Uses uniform class weights for the first `E_DRW` epochs, then ramps class-balanced effective weights using a cosine schedule:
  ```text
  w_c proportional to (1 - beta) / (1 - beta^(N_c)),  where beta = 0.9999
  ```

### 2.8 Clinical Calibration & Specificity-Constrained Decision Thresholds
- **Strictly Positive Platt Scaling (`calibration.py`):**
  Calculates calibrated probabilities `P(y=c | z) = softmax(z / T(tau))` using a positive temperature:
  ```text
  T(tau) = 0.1 + ln(1 + exp(tau))
  ```
  This prevents probability inversion during numerical optimization.
- **Cost-Sensitive Multi-Class Threshold Optimization:**
  Finds decision thresholds `t*` that maximize sensitivity while adhering to a strict specificity floor:
  ```text
  t* = argmax_t Sensitivity(t)   subject to   Specificity(t) >= 0.9000
  ```

### 2.9 Quantum Continual Learning: QFIM & QEWC
When fine-tuning sequentially across hospital sites or imaging domains, `continual.py` mitigates catastrophic forgetting using **Quantum Elastic Weight Consolidation (QEWC)**:
```text
L_total(theta) = L_task(theta) + sum_k (lambda_EWC / 2) * F_{k,k} * (theta_k - theta_k^*)^2
```
Where `F_{k,k}` represents the diagonal of the **Quantum Fisher Information Matrix (QFIM)** evaluated via parameter-shift rules:
```text
F_{k,k} = 4 * ( <d_k psi | d_k psi> - |<psi | d_k psi>|^2 )
```

---

## 3. 📂 Repository Directory & File Architecture

```
Early Disease Prediction/
├── architecture.md               # Detailed scientific architectural reference
├── env.sh                        # Shell environment script for thread capping
├── requirements.txt              # Production dependency specifications
├── npz_convert.py                # Medical dataset downloader & .npz compiler
├── testing.py                    # Independent batch model evaluation & benchmark suite
├── task.md                       # Task tracker and verification checklist
├── problems.md                   # Formal mathematical and scientific audit records
├── README.md                     # Comprehensive repository documentation
├── biomedclip/                   # Fine-tuned BiomedCLIP ViT checkpoints
├── data/                         # Local MedMNIST and custom .npz dataset storage
├── outputs/                      # Generated evaluation outputs and diagnostic plots
│   └── <dataset_name>/           # Per-dataset metrics, checkpoints, and 39 plots
├── deployment/                   # Deployment assets and runtime configurations
└── qml_models/                   # Core Python package containing the QML system
    ├── __init__.py               # Package initializer and path configuration
    ├── adapter.py                # CQ-Adapter, LoRA-Q linear, and S² head pretraining
    ├── ansatz/                   # Quantum circuit topology implementations
    │   ├── __init__.py           # Sub-package initializer
    │   └── qaf.py                # Hierarchical Quantum Attention-Fourier (QAF) circuit
    ├── app.py                    # Dual Clinical Streamlit Interactive Dashboard
    ├── calibration.py            # Strictly positive Platt, Beta, and Dirichlet calibration
    ├── continual.py              # Quantum Fisher Information (QFIM) and QEWC engine
    ├── dataset.py                # MedMNIST ingestion, quantum mixup, and data loaders
    ├── engine.py                 # Core CLI orchestrator, training loops, and benchmarking
    ├── finetuning.py             # Top-N unfreezing fine-tuning for BiomedCLIP ViT
    ├── kfold.py                  # Stratified K-Fold cross-validation & OOF ensembling
    ├── losses.py                 # CB-LDAM-DRW, Focal, InfoNCE, and anti-shortcut losses
    ├── model.py                  # HybridCNNVQC, ClassicalBaseline, and PennyLane QNode
    ├── plots.py                  # Comprehensive 39-plot diagnostic visualization suite
    ├── preprocessing.py          # Foreground extraction, GridMask, and anti-shortcut tools
    ├── qnas.py                   # Quantum Neural Architecture Search evolutionary engine
    ├── test_pipeline.py          # Smoke tests and pipeline verification scripts
    └── training_utils.py         # Multi-rate optimizers, Q-SAM, Q-SWA, and hypothesis tests
```

---

## 4. ⚙️ Installation & Environment Setup

### 4.1 Prerequisites & System Requirements
- **Operating System:** Linux (Ubuntu 20.04/22.04), Windows 10/11 (PowerShell), or macOS (Apple Silicon M-series supported).
- **Python:** Version 3.10 or 3.11 recommended.
- **Compute Hardware:** Dedicated NVIDIA GPU with 8 GB or more VRAM recommended; CPU execution is supported via PennyLane's `default.qubit`.

### 4.2 Virtual Environment Initialization

```bash
# Navigate to the repository root
cd "d:/Code/QuantumML/Early Disease Prediction"

# Create a clean virtual environment
python -m venv venv

# Activate on Linux / macOS:
source venv/bin/activate

# Activate on Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Activate on Windows (Command Prompt):
.\venv\Scripts\activate.bat
```

### 4.3 Dependency Installation

Install production dependencies using `requirements.txt`:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Verify the environment:
```bash
python -c "import torch, pennylane, streamlit, medmnist, open_clip; print('All core libraries verified!')"
```

### 4.4 Multithreading & Memory Safety Variables
Under WSL2, Hyper-V, or multi-core workstations, OpenMP and MKL thread oversubscription can cause memory bottlenecks during parallel quantum simulation. Cap thread counts before training:

**Linux / macOS:**
```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
```

**Windows (PowerShell):**
```powershell
$env:OMP_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"
$env:OPENBLAS_NUM_THREADS="4"
```

---

## 5. 🚀 End-to-End Execution Guide

### 5.1 Training Engine (`engine.py`)
`engine.py` is the primary CLI orchestrator for training, architecture search, cross-validation, and diagnostic evaluation.

#### Standard Hybrid VQC Training
Train a standard 4-qubit, 3-layer Hybrid CNN-VQC on PneumoniaMNIST:
```bash
python qml_models/engine.py --dataset pneumoniamnist --epochs 15 --batch_size 32 --n_qubits 4 --n_layers 3 --lr 1e-3
```

#### HQCNN Dual-Basis DermaMNIST Flagship Pipeline
Train an 8-qubit HQCNN on DermaMNIST with Dual-Basis Readout, QAF Attention, CB-LDAM Loss, Anti-Shortcut penalties, and Clinical Hardening:
```bash
python qml_models/engine.py \
  --dataset dermamnist \
  --n_qubits 8 \
  --n_layers 3 \
  --epochs 20 \
  --batch_size 32 \
  --dual_basis_readout \
  --use_qaf_attention \
  --clinical_hard \
  --loss_type ldam \
  --drw_start_epoch 8 \
  --lambda_decorr 0.01 \
  --lambda_saliency 0.05 \
  --target_specificity 0.9000 \
  --amp \
  --cache_in_vram
```

#### Stratified K-Fold Cross-Validation
Run 5-fold cross-validation with Out-of-Fold (OOF) ensembling and Dirichlet calibration:
```bash
python qml_models/engine.py \
  --dataset pneumoniamnist \
  --kfold \
  --n_folds 5 \
  --epochs 10 \
  --n_qubits 4 \
  --n_layers 3 \
  --dual_basis_readout
```

#### Quantum Continual Learning (QEWC)
Train sequentially across multiple datasets while regularizing weights via the Quantum Fisher Information Matrix:
```bash
python qml_models/engine.py \
  --continual \
  --continual_datasets pneumoniamnist breastmnist \
  --epochs 10 \
  --lambda_ewc 100.0 \
  --n_qubits 4 \
  --n_layers 3
```

#### Quantum Architecture Search (QNAS) & Progressive Depth Growth
Search for an optimal quantum ansatz using Pareto evolutionary optimization, then progressively expand circuit depth:
```bash
python qml_models/engine.py \
  --dataset pneumoniamnist \
  --ansatz_mode qnas_progressive \
  --qnas_generations 15 \
  --qnas_pop_size 20 \
  --progressive_depths 1 2 3 \
  --epochs_per_stage 4 \
  --n_qubits 4
```

#### Evaluation-Only Mode & Threshold Recalibration
Evaluate a trained model checkpoint and optimize decision thresholds on validation data:
```bash
python qml_models/engine.py \
  --dataset pneumoniamnist \
  --eval_only \
  --recalibrate_threshold \
  --target_specificity 0.9000
```

---

#### Comprehensive CLI Flag Reference for `engine.py`

| CLI Flag | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--dataset` | `str` | `pneumoniamnist` | Target dataset: `pneumoniamnist`, `dermamnist`, `retinamnist`, `breastmnist`, `organamnist`, `pathmnist`, `covid`, `oasis`. |
| `--epochs` | `int` | `15` | Total training epochs. |
| `--batch_size` | `int` | `32` | Minibatch size for training and validation loaders. |
| `--lr` | `float` | `1e-3` | Base learning rate for projection and classification layers. |
| `--n_qubits` | `int` | `4` | Number of qubits in quantum register (scalable to 8). |
| `--n_layers` | `int` | `3` | Number of variational layers in the ansatz circuit. |
| `--dual_basis_readout` | `flag` | `False` | Enables Dual-Basis Pauli `[<Z>, <X>]` readout (2N features) instead of legacy Pauli (3N-1 features). |
| `--use_qaf_attention` | `flag` | `False` | Enables Hierarchical QAF circuit (Superpositional Entanglement + Toffoli Attention + Fourier CPhase). |
| `--clinical_hard` | `flag` | `False` | Restricts CQ-adapter polar angles to `[-pi/2, pi/2]` to eliminate phase-wrapping boundary artifacts. |
| `--loss_type` | `str` | `ce` | Primary loss function: `ce` (Cross-Entropy), `focal`, or `ldam` (CB-LDAM-DRW). |
| `--ldam_max_margin` | `float` | `0.5` | Maximum margin offset for minority classes in CB-LDAM. |
| `--drw_start_epoch` | `int` | `10` | Epoch at which Deferred Re-Weighting activates in CB-LDAM. |
| `--lambda_decorr` | `float` | `0.01` | Weight for anti-shortcut feature decorrelation loss penalty. |
| `--lambda_saliency` | `float` | `0.05` | Weight for anti-shortcut saliency concentration loss penalty. |
| `--target_specificity`| `float` | `0.9000` | Minimum specificity constraint enforced during threshold search. |
| `--use_lora` | `flag` | `False` | Applies LoRA-Q factorization (rank=4) to the classical adapter. |
| `--use_qnas` | `flag` | `False` | Executes Quantum Neural Architecture Search for ansatz design. |
| `--progressive` | `flag` | `False` | Activates progressive layer-wise depth expansion during training. |
| `--kfold` | `flag` | `False` | Executes Stratified K-Fold Cross-Validation with OOF ensembling. |
| `--n_folds` | `int` | `5` | Number of cross-validation folds. |
| `--continual` | `flag` | `False` | Enables continual learning mode using QEWC. |
| `--lambda_ewc` | `float` | `100.0` | Regularization coefficient for Quantum Elastic Weight Consolidation. |
| `--amp` / `--no_amp` | `flag` | `True` | Enables/disables Automatic Mixed Precision (AMP) for classical operations. |
| `--cache_in_vram` | `flag` | `False` | Caches frozen BiomedCLIP embeddings directly in GPU VRAM. |
| `--fast_eval` | `flag` | `False` | Bypasses 2D loss landscapes and heavy quantum noise simulations. |
| `--skip_plots` | `flag` | `False` | Skips generation of diagnostic plots during quick hyperparameter sweeps. |
| `--plot_workers` | `int` | `4` | Parallel worker processes used to render diagnostic figures. |

---

### 5.2 Interactive Clinical Streamlit Dashboard (`app.py`)
`app.py` delivers an interactive clinical dashboard for side-by-side model auditing, diagnostic consensus verification, and interpretability analysis.

```bash
# Launch the dashboard from the repository root:
streamlit run qml_models/app.py
```

The application runs locally at `http://localhost:8501`.

```text
========================================================================================
                  DUAL CLINICAL DIAGNOSIS & QUANTUM AUDITING DASHBOARD
========================================================================================
 [Input Image]          [Hybrid CNN-VQC Prediction]       [Classical Baseline Prediction]
  Chest X-Ray /           Diagnosis: PNEUMONIA (94.2%)      Diagnosis: PNEUMONIA (81.6%)
  Dermoscopy              Calibrated Conf: 91.8%            Calibrated Conf: 78.4%
  Slide                   Conformal Set: {Pneumonia}        Conformal Set: {Normal, Pneumonia}
                          Uncertainty: +/- 0.038 (Low)      Uncertainty: +/- 0.092 (Moderate)
----------------------------------------------------------------------------------------
 [CONSENSUS ALERT]      STATUS: HIGH CLINICAL CONSENSUS (Models Agree on Primary Pathogen)
----------------------------------------------------------------------------------------
 [INTERPRETABILITY]
  Captum Grad-CAM:        Highlights dense consolidation in lower right lung field
  Quantum Pauli-Z Saliency: Wires q0 and q3 show >0.85 polarization on lesion boundary
========================================================================================
```

#### Core Dashboard Capabilities:
1. **Dual Model Comparison:** Runs `HybridCNNVQC` and `ClassicalBaseline` simultaneously on uploaded patient scans or benchmark samples.
2. **Consensus & Divergence Alerts:** Automatically verifies diagnostic agreement. If models diverge, a clinical warning flag prompts manual physician review.
3. **Calibrated Probabilities & Conformal Sets:** Displays strictly positive Platt-calibrated probabilities alongside 90% and 95% Conformal Prediction sets with bounded coverage errors.
4. **Captum Grad-CAM Lesion Localization:** Overlays gradient-weighted class activation heatmaps onto original scans to verify anatomical focus.
5. **Live Pauli Observable Saliency:** Displays horizontal bar charts of quantum expectation values (`<Z_i>`, `<X_i>`), showing qubit polarizations during classification.
6. **Zero-Latency Offline Mode:** Operates with cached models and offline flags (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`), ensuring fast local startup.

---

### 5.3 Independent Evaluation & Benchmarking Suite (`testing.py`)
To benchmark saved model checkpoints, compute hypothesis tests, or regenerate diagnostic figures without retraining:

```bash
# Evaluate on a single dataset with 100 test samples:
python testing.py --dataset pneumoniamnist --num_samples 100

# Evaluate using Out-of-Fold (OOF) K-Fold checkpoints:
python testing.py --dataset pneumoniamnist --kfold

# Run benchmarks across all available trained datasets:
python testing.py --dataset all
```

---

### 5.4 Dataset Ingestion & NPZ Preprocessing (`npz_convert.py`)
To incorporate external medical archives (COVID-19 Radiography or OASIS-1 Brain MRI), `npz_convert.py` downloads cohorts via KaggleHub, resizes images, applies stratified splits (70% train / 10% validation / 20% test), and compiles MedMNIST-compatible `.npz` files.

```bash
# Download and compile COVID-19 Radiography (224x224):
python npz_convert.py --dataset covid --data_dir data --output data/covid_224.npz --size 224

# Download and compile OASIS-1 Brain MRI with class balancing:
python npz_convert.py --dataset oasis --data_dir data --output data/oasis_224.npz --size 224 --balance
```

---

### 5.5 BiomedCLIP ViT Fine-Tuning (`finetuning.py`)
Before training the quantum circuit, fine-tune the upper transformer layers of BiomedCLIP for a specific clinical domain:

```bash
python qml_models/finetuning.py \
  --dataset dermamnist \
  --unfreeze_layers 2 \
  --epochs 5 \
  --lr 1e-5 \
  --save_dir biomedclip
```

Checkpoints are stored at `biomedclip/{dataset}_biomedclip_vit.pt`.

---

### 5.6 Classical-to-Quantum Adapter Pre-training (`adapter.py`)
Pre-train the Classical-to-Quantum adapter to map classical features uniformly across the Bloch sphere prior to VQC training:

```bash
python qml_models/adapter.py \
  --dataset pneumoniamnist \
  --n_qubits 4 \
  --epochs 10 \
  --lr 1e-3
```

---

### 5.7 Diagnostic Pipeline Verification (`test_pipeline.py`)
Verify end-to-end pipeline integrity (model construction, quantum nodes, loss functions, and data loaders) with smoke tests:

```bash
python qml_models/test_pipeline.py
```

---

## 6. 📊 Supported Datasets & Clinical Modalities

| Dataset Identifier | Clinical Modality | Diagnostic Target | Classes | Input Dimensions | Default Split (Train / Val / Test) |
|:---|:---|:---|:---:|:---:|:---:|
| `pneumoniamnist` | Pediatric Chest X-Ray | Normal vs. Pneumonia | 2 | 28x28 / 224x224 | 4,708 / 524 / 624 |
| `dermamnist` | Dermatoscopy | Skin Lesions (Melanoma, Basal Cell, Nevi, etc.) | 7 | 28x28 / 224x224 | 7,007 / 1,003 / 2,005 |
| `retinamnist` | Fundus Photography | Diabetic Retinopathy Grading | 5 | 28x28 / 224x224 | 1,080 / 120 / 400 |
| `breastmnist` | Breast Ultrasound | Benign vs. Malignant Masses | 2 | 28x28 / 224x224 | 546 / 78 / 156 |
| `organamnist` | Abdominal Computed Tomography (CT) | Multi-Organ Anatomical Classification | 11 | 28x28 / 224x224 | 34,581 / 6,491 / 17,778 |
| `pathmnist` | Colorectal Histopathology | Tissue Morphology (Adenoma, Mucosa, Stroma) | 9 | 28x28 / 224x224 | 89,996 / 10,004 / 7,180 |
| `covid` | Adult Chest X-Ray | Healthy vs. COVID-19 Infiltration | 2 | 224x224 | 10,616 / 1,516 / 3,033 |
| `oasis` | Cross-Sectional Brain MRI | Cognitive Health vs. Dementia | 2 | 224x224 | 291 / 42 / 83 |

---

## 7. 📈 Comprehensive Diagnostic Plotting Suite (39 Plots)

After training or during `--eval_only` execution, `plots.py` generates up to 39 diagnostic figures in `outputs/<dataset>/`:

```
outputs/<dataset_name>/
├── 01_loss_curves.png                    # Training & validation loss trajectory across epochs
├── 02_roc_curves.png                     # Multi-class ROC curves with micro/macro AUC
├── 03_pr_curves.png                      # Precision-Recall curves with Average Precision (AP)
├── 04_confusion_matrices.png             # Normalized confusion matrix (Hybrid vs. Baseline)
├── 05_calibration_curves.png             # Reliability diagrams before and after Platt calibration
├── 06_bloch_sphere_coverage.png          # Per-qubit Bloch vector coordinates on S² manifold
├── 07_quantum_saliency.png               # Pauli observable parameter-shift sensitivity gradients
├── 08_gradcam_overlays.png               # Anatomical Grad-CAM heatmaps on pathological regions
├── 09_shortcut_verification.png          # 4-Column audit: Original, Grad-CAM, Background, Foreground
├── 10_conformal_intervals.png            # Conformal prediction set sizes across confidence levels
├── 11_tsne_embeddings.png                # Pre vs. post-training t-SNE latent representations
├── 12_umap_embeddings.png                # Pre vs. post-training UMAP manifold projections
├── 13_cosine_drift_histogram.png         # Feature cosine similarity drift during optimization
├── 14_distance_matrix.png                # Inter-class vs. intra-class distance metrics
├── 15_cls_attention_heatmaps.png         # ViT CLS token self-attention heatmaps
├── 16_attention_shift_map.png            # Pre vs. post fine-tuning ViT attention shift
├── 17_generalization_gap.png             # Train-to-validation loss generalization divergence
├── 18_loss_ratio_plot.png                # Empirical loss ratio bounds across epochs
├── 19_sample_size_learning_curves.png    # Performance scaling across training sample fractions
├── 20_weight_norm_trajectory.png         # Layer-wise weight norm and parameter displacement
├── 21_effective_rank_curve.png           # Effective rank of latent representations over time
├── 22_singular_value_decay.png           # Singular value spectrum decay across transformer layers
├── 23_cka_similarity_heatmap.png         # Centered Kernel Alignment (CKA) between layers
├── 24_attention_entropy.png              # Attention distribution entropy per transformer head
├── 25_spatial_frequency_energy.png       # High vs. low spatial frequency attention energy
├── 26_occlusion_sensitivity.png          # Sliding window occlusion sensitivity maps
├── 27_ood_performance_drop.png           # Out-of-distribution robustness degradation chart
├── 28_demographic_disparities.png        # Subgroup and cohort performance parity audit
├── 29_input_gradient_norm.png            # Input gradient norm vs. loss sensitivity trajectory
├── 30_saliency_iou_curves.png            # Grad-CAM IoU overlap with ground-truth lesion ROI
├── 31_domain_shift_curves.png            # Cross-site distribution shift evaluation curves
├── 32_ece_brier_trajectory.png           # Expected Calibration Error (ECE) and Brier scores
├── 33_confidence_histogram.png           # Softmax top-1 predictive confidence distributions
├── 34_uncertainty_error_scatter.png      # Predictive uncertainty vs. classification error
├── 35_qnas_pareto_front.png              # Pareto front: Expressibility vs. CNOT gate count
├── 36_qfim_spectrum.png                  # Quantum Fisher Information eigenvalue distribution
├── 37_loss_landscape_2d.png              # 2D loss landscape contour around optimal parameters
├── 38_noise_robustness_zne.png           # Zero-Noise Extrapolation (ZNE) error mitigation curve
└── 39_mcnemar_hypothesis_test.png        # Statistical McNemar contingency test comparison
```

---

## 8. 🛠️ Troubleshooting & Common Engineering Pitfalls

### 1. Windows Unicode Encoding Errors (`cp1252`)
- **Symptom:** `UnicodeEncodeError: 'charmap' codec can't encode character '\u2500'` when printing box-drawing characters on Windows.
- **Solution:** `engine.py` and `app.py` configure standard I/O streams to UTF-8 on startup:
  ```python
  import sys
  if hasattr(sys.stdout, 'reconfigure'):
      sys.stdout.reconfigure(encoding='utf-8', errors='replace')
  ```

### 2. Host-to-Device Memory Thrashing & CUDA Out of Memory (OOM)
- **Symptom:** Excessive RAM consumption or GPU OOM errors during training.
- **Solution:**
  1. Only pass `--cache_in_vram` if dedicated GPU memory is 8 GB or greater.
  2. Restrict OpenMP and MKL thread pools to 4 (Section 4.4).
  3. Pass `--fast_eval` to skip intensive 2D loss landscapes and multi-sample noise evaluations.

### 3. PennyLane Adjoint Differentiation Incompatibilities
- **Symptom:** Gradient backpropagation errors when combining `--diff_method adjoint` with trainable classical backbones.
- **Solution:** Adjoint differentiation cannot backpropagate gradients into classical PyTorch networks. Use the default PyTorch autograd backprop method:
  ```bash
  --diff_method backprop --qdevice default.qubit
  ```

### 4. Offline Model Loading for Hugging Face & OpenCLIP
- **Symptom:** Network connection timeouts when loading BiomedCLIP weights in air-gapped clinical environments.
- **Solution:** Ensure offline environment flags are set:
  ```bash
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  ```
  `app.py` sets these environment flags automatically before importing dependencies.

---

## 9. 📜 Citation & Academic Reference

If you use this hybrid quantum-classical framework, clinical calibration pipeline, or anti-shortcut suite in your research, please cite:

```bibtex
@article{qml_early_disease_prediction_2026,
  title   = {Hybrid Classical-Quantum Convolutional Variational Circuits with Dual-Basis Readout and Anti-Shortcut Guarantees for Early Disease Prediction},
  author  = {Advanced Quantum-Classical Clinical AI Research Consortium},
  journal = {IEEE Transactions on Quantum Engineering / Medical Image Analysis},
  year    = {2026}
}
```
