# Hybrid Classical-Quantum Neural Network (CNN-VQC & HQCNN) Framework for Early Disease Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane 0.33+](https://img.shields.io/badge/PennyLane-0.33+-yellow.svg)](https://pennylane.ai/)
[![MedMNIST v2](https://img.shields.io/badge/Dataset-MedMNIST%20v2-green.svg)](https://medmnist.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An institutional-grade, research-validated computational framework integrating frozen and fine-tuned Foundation Vision Backbones (**BiomedCLIP ViT-B/16**) with **$N$-Qubit Variational Quantum Circuits (VQC / HQCNN)** for multimodal early medical disease detection. 

Designed for high-stakes clinical diagnostic workflows across the **MedMNIST v2** benchmark suite (e.g., PneumoniaMNIST, DermaMNIST, RetinaMNIST, BreastMNIST, OrganAMNIST) and real-world medical image archives (COVID-19 Radiography, OASIS-1 Brain MRI), this repository features rigorous anti-shortcut learning modules, geometric Riemannian spherical feature projection, quantum neural architecture search (QNAS), dual-basis Pauli observable readout, strictly positive temperature calibration, and an interactive Streamlit clinical dashboard with Grad-CAM and quantum saliency interpretability.

---

## Table of Contents

1. [Executive Summary & Clinical Rationale](#1-executive-summary--clinical-rationale)
2. [Key Architectural Innovations & Mathematical Foundations](#2-key-architectural-innovations--mathematical-foundations)
   - [2.1 Foundation Vision Backbone (BiomedCLIP ViT-B/16)](#21-foundation-vision-backbone-biomedclip-vit-b16)
   - [2.2 Classical-to-Quantum Adapter (CQ-Adapter) & Riemannian S² Head](#22-classical-to-quantum-adapter-cq-adapter--riemannian-s-head)
   - [2.3 Variational Quantum Circuit (VQC) Ansätze](#23-variational-quantum-circuit-vqc-ansätze)
   - [2.4 Quantum Readout Paradigms: Dual-Basis (HQCNN) vs. Legacy Pauli](#24-quantum-readout-paradigms-dual-basis-hqcnn-vs-legacy-pauli)
   - [2.5 Parameter-Matched Classical Baseline](#25-parameter-matched-classical-baseline)
   - [2.6 Anti-Shortcut Learning & Modality Adaptation Suite](#26-anti-shortcut-learning--modality-adaptation-suite)
   - [2.7 Class Imbalance Mitigation: CB-LDAM-DRW & Asymmetric Loss](#27-class-imbalance-mitigation-cb-ldam-drw--asymmetric-loss)
   - [2.8 Clinical Calibration & Specificity-Constrained Decision Thresholds](#28-clinical-calibration--specificity-constrained-decision-thresholds)
   - [2.9 Quantum Continual Learning: QFIM & QEWC](#29-quantum-continual-learning-qfim--qewc)
3. [Repository Directory & File Architecture](#3-repository-directory--file-architecture)
4. [Installation & Environment Setup](#4-installation--environment-setup)
   - [4.1 Prerequisites](#41-prerequisites)
   - [4.2 Virtual Environment Initialization](#42-virtual-environment-initialization)
   - [4.3 Dependency Installation](#43-dependency-installation)
   - [4.4 Multithreading & Memory Safety Variables](#44-multithreading--memory-safety-variables)
5. [End-to-End Execution Guide](#5-end-to-end-execution-guide)
   - [5.1 Training Engine (`engine.py`)](#51-training-engine-enginepy)
     - [Standard Hybrid VQC Training](#standard-hybrid-vqc-training)
     - [HQCNN Dual-Basis DermaMNIST Pipeline](#hqcnn-dual-basis-dermamnist-pipeline)
     - [Stratified K-Fold Cross-Validation](#stratified-k-fold-cross-validation)
     - [Quantum Continual Learning (QEWC)](#quantum-continual-learning-qewc)
     - [Quantum Architecture Search (QNAS) & Progressive Depth Growth](#quantum-architecture-search-qnas--progressive-depth-growth)
     - [Evaluation-Only Mode](#evaluation-only-mode)
     - [Comprehensive CLI Flag Reference for `engine.py`](#comprehensive-cli-flag-reference-for-enginepy)
   - [5.2 Interactive Clinical Streamlit Dashboard (`app.py`)](#52-interactive-clinical-streamlit-dashboard-apppy)
   - [5.3 Independent Evaluation & Benchmarking Suite (`testing.py`)](#53-independent-evaluation--benchmarking-suite-testingpy)
   - [5.4 Dataset Ingestion & NPZ Preprocessing (`npz_convert.py`)](#54-dataset-ingestion--npz-preprocessing-npz_convertpy)
   - [5.5 BiomedCLIP ViT Fine-Tuning (`finetuning.py`)](#55-biomedclip-vit-fine-tuning-finetuningpy)
   - [5.6 Classical-to-Quantum Adapter Pre-training (`adapter.py`)](#56-classical-to-quantum-adapter-pre-training-adapterpy)
   - [5.7 Diagnostic Pipeline Verification (`test_pipeline.py`)](#57-diagnostic-pipeline-verification-test_pipelinepy)
6. [Supported Datasets & Clinical Modalities](#6-supported-datasets--clinical-modalities)
7. [Comprehensive Diagnostic Plotting Suite (39 Plots)](#7-comprehensive-diagnostic-plotting-suite-39-plots)
8. [Troubleshooting & Common Engineering Pitfalls](#8-troubleshooting--common-engineering-pitfalls)

---

## 1. Executive Summary & Clinical Rationale

Deep learning models deployed in clinical imaging are prone to shortcut learning (relying on peripheral scanner markers, border illumination artifacts, or background noise) and overconfident miscalibrations on underrepresented pathological classes. 

This repository implements a **Hybrid Classical-Quantum Architecture (HQCNN / CNN-VQC)** designed to overcome these fundamental failure modes:
1. **Geometric Quantum State Encoding:** High-dimensional vision representations from BiomedCLIP ViT are compressed and projected onto a compact Riemannian Bloch sphere $\mathcal{S}^2$, preventing the network from overfitting to non-biological high-frequency noise.
2. **Dual-Basis Readout Observables:** Measuring both longitudinal computational basis $\langle Z_i \rangle$ (optical density/absorption) and transverse superposition basis $\langle X_i \rangle$ (structural boundary asymmetry) grants the quantum layer high expressive sensitivity to micro-architectural lesion features.
3. **Provable Anti-Shortcut Guarantees:** Through morphological foreground masking, GridMask, border cropping, feature-decorrelation loss ($\mathcal{L}_{\text{decorr}}$), and saliency-concentration penalties ($\mathcal{L}_{\text{saliency}}$), the model is penalized whenever predictions correlate with peripheral background pixels.
4. **Safety-Constrained Clinical Decisions:** Multi-class probability distributions undergo strictly positive Platt temperature calibration ($T(\tau) = 0.1 + \text{softplus}(\tau)$) and cost-sensitive threshold shifting enforced with a strict $\text{Specificity} \ge 0.9000$ floor.

---

## 2. Key Architectural Innovations & Mathematical Foundations

```
MedMNIST Medical Input (28x28 to 224x224 RGB/Grayscale)
         │
         ▼  [preprocessing.py / dataset.py]
 Modality-Adaptive Preprocessing & Anti-Shortcut Suite
  ├── Foreground Mask Extraction: M_fg ∈ {0, 1}^{H×W}
  ├── RandomBorderCrop, GridMask, RandomHistogramEqualization
  └── MultiViewBatchCollator (Flattened 4D Batch)
         │
         ▼  [model.py / finetuning.py]
 Frozen / Top-2 Unfrozen BiomedCLIP ViT-B/16 Backbone
  └── L2-Normalized CLS Embedding Vector: f ∈ ℝ⁵¹²
         │
         ▼  [adapter.py]
 Classical-to-Quantum Adapter (CQ-Adapter)
  ├── LoRA-Q Factorization: W_eff = W₀ + (scale/rank) · (B @ A)
  ├── Cayley Map Orthogonal Parameterization
  └── Riemannian S² Spherical Head:
        θ_i = σ(v_i) · π  ∈ [0, π]
        φ_i = tanh(v_{i+n}) · π ∈ [-π, π]
        λ_i = tanh(v_{i+2n}) · π ∈ [-π, π]
         │
         ▼  [model.py / ansatz/qaf.py / PennyLane Engine]
 Quantum Hilbert Space Circuit (N=8 Qubits, dim ℋ = 256)
  ├── Ground State Preparation: |0⟩^⊗N
  ├── Universal Data Re-Uploading: U3(θ, φ, λ)
  ├── Variational Layer Topologies:
  │     ├── Option A: Hierarchical QAF (Superpositional Entanglement + Toffoli Attention + Fourier CPhase)
  │     ├── Option B: QNAS Evolutionary DAG Supercircuit
  │     └── Option C: StronglyEntanglingLayers (Hardware-Efficient CNOT Ring)
  └── Quantum Weight Dropout (Gaussian Lie-Algebra Phase Jitter)
         │
         ▼  [Readout Observables]
 Quantum Measurement Transformation:
  ├── Dual-Basis HQCNN: [⟨Z_i⟩, ⟨X_i⟩] ∈ [-1, 1]^{2N}  (16 dims for N=8)
  └── Legacy Pauli:     [⟨Z_i⟩, ⟨Z_i Z_{i+1}⟩, ⟨X_i⟩] ∈ [-1, 1]^{3N-1} (23 dims for N=8)
         │
         ▼  [model.py]
 Symmetric Projection Head:
  Linear(dim_readout → 32) ──► GELU ──► Dropout(0.2) ──► Linear(32 → C)
         │
         ▼  [losses.py / calibration.py / engine.py]
 Optimization & Clinical Decision Calibration:
  ├── Loss: CB-LDAM-DRW + L_decorr + L_saliency
  ├── Multi-Rate AdamW: lr_backbone=1e-5, lr_adapter=1e-3, lr_quantum=1e-2
  ├── Strictly Positive Platt Calibration: T(τ) = 0.1 + ln(1 + e^τ)
  └── Cost-Sensitive Thresholding: Specificity ≥ 0.9000 floor
```

### 2.1 Foundation Vision Backbone (BiomedCLIP ViT-B/16)
The vision encoder leverages domain-specific knowledge pretrained on over 15 million biomedical image-text pairs (PubMed Central).
- Grayscale medical inputs (e.g., Pneumonia chest radiographs) are expanded to 3 channels via channel replication: $x \leftarrow x.\text{repeat}(1, 3, 1, 1)$.
- Input normalization: $\mu = [0.4814, 0.4578, 0.4082]$, $\sigma = [0.2686, 0.2613, 0.2758]$.
- Feature extraction yields a 512-dimensional $\ell_2$-normalized class token: $f_{\text{CLS}} \in \mathbb{R}^{512}$.
- To eliminate data leakage and domain conflict, fine-tuned backbone weights are isolated per dataset under `biomedclip/{dataset}_biomedclip_vit.pt`.

### 2.2 Classical-to-Quantum Adapter (CQ-Adapter) & Riemannian S² Head
To bridge the 512-dimensional classical latent space with the $N$-qubit quantum register (default $N=8$, $\dim \mathcal{H} = 256$), the network implements a constrained linear projection:
$$\mathbf{v} = \mathbf{W}_{\text{eff}} f_{\text{CLS}} + \mathbf{b} \in \mathbb{R}^{3N}$$

Where $\mathbf{W}_{\text{eff}}$ uses **LoRA-Q Factorization**:
$$\mathbf{W}_{\text{eff}} = \mathbf{W}_0 + \frac{\alpha}{r} (\mathbf{B} \mathbf{A})$$
Here, $\mathbf{W}_0$ is frozen with orthogonal Cayley initialization, while $\mathbf{A} \in \mathbb{R}^{r \times 512}$ and $\mathbf{B} \in \mathbb{R}^{3N \times r}$ ($r=4$) are trained.

The resulting vector $\mathbf{v}$ is mapped onto individual single-qubit Bloch spheres via the **Riemannian $\mathcal{S}^2$ Head**:
$$\theta_i = \sigma(v_i) \cdot \pi \in [0, \pi]$$
$$\phi_i = \tanh(v_{i+N}) \cdot \pi \in [-\pi, \pi]$$
$$\lambda_i = \tanh(v_{i+2N}) \cdot \pi \in [-\pi, \pi]$$

Each qubit receives a full polar parameterization defining the state $|\psi_i\rangle = \cos(\theta_i/2)|0\rangle + e^{i\phi_i}\sin(\theta_i/2)|1\rangle$. In clinical-hardening mode (`--clinical_hard`), the polar angle is bounded to $[-\pi/2, \pi/2]$ to prevent phase-wrapping boundary singularities.

### 2.3 Variational Quantum Circuit (VQC) Ansätze
The engine supports three variational quantum circuit configurations:

1. **Hierarchical Quantum Attention-Fourier (QAF) Circuit (`--use_qaf_attention`):**
   - **Universal Data Re-Uploading:** $U_3(\theta_i, \phi_i, \lambda_i)$ parameterized rotation gates inserted before *every* variational layer.
   - **Superpositional Entanglement:** Nearest-neighbor cyclic CNOT ring creating entanglement across adjacent wires.
   - **Local 3-Qubit Toffoli Attention:** Parameterized non-linear multi-qubit interaction modulating attention between adjacent qubit triplets.
   - **Global Dyadic Fourier Phase Modulation:** Controlled-phase ($CR_Z$) rotations with dyadic scaling phases $\Delta \phi_{j,k} = \frac{2\pi}{2^{|j-k|}}$ capturing multi-scale non-local feature interactions.
   - **Variational Rotation Blocks:** Independent $R_{\text{ot}}(\omega_1, \omega_2, \omega_3)$ gates applied per qubit wire.

2. **Evolutionary Quantum Architecture Search (QNAS DAG, `--use_qnas`):**
   - Represents the ansatz as a directed acyclic graph (DAG) genome.
   - Candidate gate vocabulary: $\{R_X, R_Y, R_Z, U_3, \text{CNOT}, \text{CZ}, \mathbb{I}\}$.
   - Evaluates architectures on a Pareto frontier balancing validation accuracy, Meyer-Wallach entanglement capability, and a penalization term for CNOT gate count to minimize physical decoherence.

3. **Strongly Entangling Layers (Default Fallback):**
   - Periodic parameterized single-qubit rotations followed by entangling CNOT gates with range offsets $r = 1, \dots, N-1$.

### 2.4 Quantum Readout Paradigms: Dual-Basis (HQCNN) vs. Legacy Pauli
- **HQCNN Dual-Basis Readout (`--dual_basis_readout`):**
  Measures both computational basis and transversal superposition observables:
  $$\mathbf{y}_{\text{readout}} = \left[ \langle Z_0 \rangle, \dots, \langle Z_{N-1} \rangle, \langle X_0 \rangle, \dots, \langle X_{N-1} \rangle \right]^T \in [-1, 1]^{2N}$$
  For $N=8$, this outputs a 16-dimensional vector. Clinically, $\langle Z_i \rangle$ measures diagonal state absorption (e.g., optical pigment concentration), while $\langle X_i \rangle = 2\operatorname{Re}(\rho_{01})$ captures off-diagonal coherence (structural lesion margin asymmetry).
- **Legacy Pauli Readout (Default):**
  $$\mathbf{y}_{\text{readout}} = \left[ \{\langle Z_i \rangle\}_{i=0}^{N-1}, \{\langle Z_i Z_{i+1} \rangle\}_{i=0}^{N-2}, \{\langle X_i \rangle\}_{i=0}^{N-1} \right]^T \in [-1, 1]^{3N-1}$$
  Outputs 23 measurement features for $N=8$.

### 2.5 Parameter-Matched Classical Baseline
To prove whether quantum advantage exists, `model.py` provides `ClassicalBaseline`:
$$\mathbf{z}_{\text{baseline}} = \mathbf{W}_2 \cdot \text{Dropout}\left(\text{GELU}\left(\mathbf{W}_1 f_{\text{CLS}}\right)\right)$$
The baseline replaces the VQC with an identically dimensioned classical MLP having matched parameters, identical learning rates, and identical loss formulations.

### 2.6 Anti-Shortcut Learning & Modality Adaptation Suite
Medical models often overfit to scanner-specific borders, text watermarks, or peripheral illumination. The anti-shortcut suite in `preprocessing.py` eliminates these confounders:
- **Otsu / Adaptive Foreground Mask Extraction:** Generates binary region-of-interest masks $M_{\text{fg}} \in \{0, 1\}^{H \times W}$.
- **GridMask & RandomBorderCrop:** Zeroes peripheral margins and applies structured dropout to eliminate edge artifacts.
- **Feature Decorrelation Loss ($\mathcal{L}_{\text{decorr}}$):**
  $$\mathcal{L}_{\text{decorr}} = \frac{1}{K} \sum_{k=1}^K \left| \text{Corr}(f_{\text{CLS}}, \mathbf{h}_k) \right|$$
  Penalizes linear correlation between backbone embeddings and low-level heuristic features $\mathbf{h}_k$ (image border intensity, global mean luminance, high-frequency boundary energy).
- **Saliency Concentration Loss ($\mathcal{L}_{\text{saliency}}$):**
  $$\mathcal{L}_{\text{saliency}} = 1.0 - \frac{\sum_{(u, v) \in \text{ROI}} \mathcal{A}(u, v)}{\sum_{u, v} \mathcal{A}(u, v)}$$
  Forces Grad-CAM attention maps $\mathcal{A}$ to lie within the clinical lesion boundary.
- **Shortcut Verification Score ($S$):**
  $$S = \frac{\mathbb{E}_{x \sim \mathcal{D}} \left[ \max_c P(y=c \mid x \cdot (1 - M_{\text{fg}})) \right]}{\mathbb{E}_{x \sim \mathcal{D}} \left[ \max_c P(y=c \mid x) \right]}$$
  A model with $S \to 0$ makes decisions purely based on pathological anatomy; $S > 0.5$ warns of shortcut reliance.

### 2.7 Class Imbalance Mitigation: CB-LDAM-DRW & Asymmetric Loss
For severe clinical imbalance (e.g., DermaMNIST melanoma vs. benign nevi):
- **Class-Balanced LDAM Loss (CB-LDAM):** Enforces margins scaled inversely by class population:
  $$\Delta_c = \frac{C_{\text{margin}}}{N_c^{1/4}}$$
  $$\mathcal{L}_{\text{LDAM}}(z, y) = -\log \frac{e^{z_y - \Delta_y}}{e^{z_y - \Delta_y} + \sum_{j \neq y} e^{z_j}}$$
- **Deferred Re-Weighting (DRW):** Operates with uniform class weights for the first $E_{\text{DRW}}$ epochs, then ramps class-balanced effective number weights via a cosine schedule:
  $$w_c \propto \frac{1 - \beta}{1 - \beta^{N_c}}, \quad \beta = 0.9999$$

### 2.8 Clinical Calibration & Specificity-Constrained Decision Thresholds
- **Strictly Positive Platt Scaling (`calibration.py`):**
  Uncalibrated logits $z$ produce calibrated probabilities $P(y=c \mid z) = \text{softmax}(z / T(\tau))$ using a guaranteed positive temperature:
  $$T(\tau) = 0.1 + \ln(1 + e^\tau)$$
  This eliminates catastrophic probability inversion when optimizing over parameter $\tau$.
- **Cost-Sensitive Multi-Class Threshold Optimization:**
  Selects probability thresholds $\mathbf{t}^*$ to minimize expected false-negative penalties while satisfying a regulatory specificity ceiling:
  $$\mathbf{t}^* = \arg\max_{\mathbf{t}} \text{Sensitivity}(\mathbf{t}) \quad \text{subject to} \quad \text{Specificity}(\mathbf{t}) \ge 0.9000$$

### 2.9 Quantum Continual Learning: QFIM & QEWC
When fine-tuning sequentially across hospital sites or imaging domains, `continual.py` prevents catastrophic forgetting using **Quantum Elastic Weight Consolidation (QEWC)**:
$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{task}}(\theta) + \sum_{k} \frac{\lambda_{\text{EWC}}}{2} F_{k, k} (\theta_k - \theta_k^*)^2$$
Where $F_{k, k}$ is the diagonal of the **Quantum Fisher Information Matrix (QFIM)** evaluated via parameter-shift rules:
$$F_{k, k} = 4 \left( \langle \partial_k \psi \mid \partial_k \psi \rangle - \left| \langle \psi \mid \partial_k \psi \rangle \right|^2 \right)$$

---

## 3. Repository Directory & File Architecture

```
Early Disease Prediction/
├── architecture.md               # Detailed scientific architectural reference
├── env.sh                        # Shell environment export script for thread capping
├── requirements.txt              # Production dependency specifications
├── npz_convert.py                # Standalone medical dataset downloader & .npz compiler
├── testing.py                    # Independent batch model evaluation & benchmark suite
├── task.md                       # Task tracker and verification checklist
├── problems.md                   # Formal mathematical and scientific audit records
├── biomedclip/                   # Isolated fine-tuned BiomedCLIP ViT checkpoints
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

## 4. Installation & Environment Setup

### 4.1 Prerequisites
- **Operating System:** Linux (Ubuntu 20.04/22.04), Windows 10/11, or macOS (Apple Silicon M1/M2/M3 supported).
- **Python:** Version 3.10 or 3.11 recommended.
- **CUDA / Compute Hardware:** NVIDIA GPU with $\ge 8\text{ GB}$ VRAM recommended for hardware acceleration; CPU execution is fully supported via PennyLane's `default.qubit`.

### 4.2 Virtual Environment Initialization

```bash
# Clone or navigate to the repository root
cd "d:/Code/QuantumML/Early Disease Prediction"

# Create a clean virtual environment
python -m venv venv

# Activate on Linux/macOS:
source venv/bin/activate

# Activate on Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Activate on Windows (Command Prompt):
.\venv\Scripts\activate.bat
```

### 4.3 Dependency Installation

Install all required dependencies using `requirements.txt`:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Verify that the core packages are correctly recognized:
```bash
python -c "import torch, pennylane, streamlit, medmnist, open_clip; print('All core libraries imported successfully!')"
```

### 4.4 Multithreading & Memory Safety Variables
Under WSL2, Hyper-V, or multi-core workstations, OpenMP and MKL thread oversubscription can exhaust RAM during parallel quantum circuit simulation. Before running large training jobs, cap the worker thread count:

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

## 5. End-to-End Execution Guide

### 5.1 Training Engine (`engine.py`)
`engine.py` is the central command-line interface orchestrator for model training, architecture search, cross-validation, and metric generation.

#### Standard Hybrid VQC Training
Train a standard 4-qubit, 3-layer Hybrid CNN-VQC on PneumoniaMNIST:
```bash
python qml_models/engine.py --dataset pneumoniamnist --epochs 15 --batch_size 32 --n_qubits 4 --n_layers 3 --lr 1e-3
```

#### HQCNN Dual-Basis DermaMNIST Pipeline
Train the flagship 8-qubit HQCNN with Dual-Basis Readout, Hierarchical QAF Attention, CB-LDAM Loss, Anti-Shortcut penalties, and Clinical Hardening:
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
Perform 5-fold cross-validation with Out-of-Fold (OOF) ensembling and Dirichlet probability calibration:
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
Train sequentially across multiple datasets while preserving previously acquired representations:
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
Search for an optimal quantum ansatz using Pareto evolutionary optimization, then progressively grow depth from 1 to 3 layers:
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

#### Evaluation-Only Mode
Evaluate a previously trained checkpoint and re-calibrate decision thresholds on validation data:
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
| `--dataset` | `str` | `pneumoniamnist` | Target dataset name from MedMNIST or converted local archives (`dermamnist`, `retinamnist`, `breastmnist`, `covid`, `oasis`). |
| `--epochs` | `int` | `15` | Total training epochs. |
| `--batch_size` | `int` | `32` | Minibatch size for training and validation loaders. |
| `--lr` | `float` | `1e-3` | Base learning rate for classical projection and classification layers. |
| `--n_qubits` | `int` | `4` | Number of qubits in quantum register (scalable to 8). |
| `--n_layers` | `int` | `3` | Number of variational layers in the ansatz circuit. |
| `--dual_basis_readout` | `flag` | `False` | Enables Dual-Basis Pauli $[\langle Z \rangle, \langle X \rangle]$ readout ($2N$ features) instead of legacy Pauli ($3N-1$ features). |
| `--use_qaf_attention` | `flag` | `False` | Enables Hierarchical QAF circuit (Superpositional Entanglement + Toffoli Attention + Fourier CPhase). |
| `--clinical_hard` | `flag` | `False` | Hardens adapter: constrains polar angles to $[-\pi/2, \pi/2]$ to prevent phase-wrapping decision boundaries. |
| `--loss_type` | `str` | `ce` | Primary loss function: `ce` (Cross-Entropy), `focal`, or `ldam` (CB-LDAM-DRW). |
| `--ldam_max_margin` | `float` | `0.5` | Maximum margin offset $\Delta_{\max}$ for minority classes in CB-LDAM. |
| `--drw_start_epoch` | `int` | `10` | Epoch at which Deferred Re-Weighting activates in CB-LDAM. |
| `--lambda_decorr` | `float` | `0.01` | Weight for anti-shortcut feature decorrelation loss penalty. |
| `--lambda_saliency` | `float` | `0.05` | Weight for anti-shortcut saliency concentration loss penalty. |
| `--target_specificity`| `float` | `0.9000` | Minimum specificity constraint enforced during threshold search. |
| `--use_lora` | `flag` | `False` | Applies LoRA-Q factorization ($r=4$) to the classical adapter. |
| `--use_qnas` | `flag` | `False` | Executes Quantum Neural Architecture Search for ansatz structure. |
| `--progressive` | `flag` | `False` | Activates progressive layer-wise depth expansion during training. |
| `--kfold` | `flag` | `False` | Executes Stratified K-Fold Cross-Validation with OOF ensembling. |
| `--n_folds` | `int` | `5` | Number of cross-validation folds. |
| `--continual` | `flag` | `False` | Enables continual learning mode using QEWC. |
| `--lambda_ewc` | `float` | `100.0` | Regularization coefficient for Quantum Elastic Weight Consolidation. |
| `--amp` / `--no_amp` | `flag` | `True` | Enables/disables Automatic Mixed Precision (AMP) for classical operations. |
| `--cache_in_vram` | `flag` | `False` | Caches frozen BiomedCLIP embeddings directly in GPU VRAM for maximum throughput. |
| `--fast_eval` | `flag` | `False` | Bypasses heavy quantum noise simulations and 2D loss landscapes to save 10-15 minutes. |
| `--skip_plots` | `flag` | `False` | Completely skips generation of diagnostic plots during quick sweeps. |
| `--plot_workers` | `int` | `4` | Parallel worker processes used to render diagnostic figures. |

---

### 5.2 Interactive Clinical Streamlit Dashboard (`app.py`)
`app.py` delivers an institutional-grade, clinician-facing visual interface designed for real-time diagnostic evaluation, side-by-side model auditing, and interpretability analysis.

```bash
# Launch the dashboard from the repository root:
streamlit run qml_models/app.py
```

The application opens automatically in your browser at `http://localhost:8501`.

```
========================================================================================
                  DUAL CLINICAL DIAGNOSIS & QUANTUM AUDITING DASHBOARD
========================================================================================
 [Input Image]          [Hybrid CNN-VQC Prediction]       [Classical Baseline Prediction]
  Chest X-Ray /           Diagnosis: PNEUMONIA (94.2%)      Diagnosis: PNEUMONIA (81.6%)
  Dermoscopy              Calibrated Conf: 91.8%            Calibrated Conf: 78.4%
  Slide                   Conformal Set: {Pneumonia}        Conformal Set: {Normal, Pneumonia}
                          Uncertainty: ±0.038 (Low)         Uncertainty: ±0.092 (Moderate)
----------------------------------------------------------------------------------------
 [CONSENSUS ALERT]      STATUS: HIGH CLINICAL CONSENSUS (Models Agree on Primary Pathogen)
----------------------------------------------------------------------------------------
 [INTERPRETABILITY]
  Captum Grad-CAM:        Highlights dense pulmonary consolidation in lower right lobe
  Quantum Pauli-Z Saliency: Wires q0 and q3 show >0.85 polarization on lesion boundary
========================================================================================
```

#### Core Dashboard Capabilities:
1. **Dual Model Comparison:** Runs the `HybridCNNVQC` model and the parameter-matched `ClassicalBaseline` simultaneously on uploaded patient scans or sample datasets.
2. **Consensus & Divergence Alerts:** Automatically flags whether both architectures agree. When models diverge, an amber alert prompts the clinician for secondary specialist review.
3. **Calibrated Probabilities & Conformal Prediction:** Displays strictly positive Platt-calibrated probability distributions alongside 90% and 95% Conformal Prediction sets with bounded error rates.
4. **Captum Grad-CAM Lesion Localization:** Computes gradient-weighted class activation heatmaps overlaid onto the original medical image to localize biological pathology.
5. **Live Pauli Observable Saliency:** Generates bar charts showing the expectation values of quantum observables ($\langle Z_i \rangle$, $\langle X_i \rangle$), showing which qubits drove the clinical classification.
6. **Zero-Latency Offline Mode:** Operates with cached models and offline flags (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`), ensuring instant sub-second startup with no network dependencies.

---

### 5.3 Independent Evaluation & Benchmarking Suite (`testing.py`)
To benchmark trained checkpoints, compute statistical hypothesis tests, or regenerate diagnostic figures without running a training loop:

```bash
# Evaluate on a single dataset with 100 test samples:
python testing.py --dataset pneumoniamnist --num_samples 100

# Evaluate using Out-of-Fold (OOF) K-Fold checkpoints:
python testing.py --dataset pneumoniamnist --kfold

# Run across all available trained datasets:
python testing.py --dataset all
```

---

### 5.4 Dataset Ingestion & NPZ Preprocessing (`npz_convert.py`)
To incorporate real-world external medical archives (COVID-19 Radiography or OASIS-1 Brain MRI), `npz_convert.py` downloads images via KaggleHub, extracts binary diagnostic cohorts, applies stratified splits (70% train / 10% validation / 20% test), and compiles them into MedMNIST-compatible `.npz` archives.

```bash
# Download and compile COVID-19 Radiography (224x224):
python npz_convert.py --dataset covid --data_dir data --output data/covid_224.npz --size 224

# Download and compile OASIS-1 Brain MRI with class balancing:
python npz_convert.py --dataset oasis --data_dir data --output data/oasis_224.npz --size 224 --balance
```

---

### 5.5 BiomedCLIP ViT Fine-Tuning (`finetuning.py`)
Before training the quantum layer, you can fine-tune the upper transformer layers of the BiomedCLIP backbone for a specific dataset:

```bash
python qml_models/finetuning.py \
  --dataset dermamnist \
  --unfreeze_layers 2 \
  --epochs 5 \
  --lr 1e-5 \
  --save_dir biomedclip
```

This saves an isolated checkpoint at `biomedclip/dermamnist_biomedclip_vit.pt`.

---

### 5.6 Classical-to-Quantum Adapter Pre-training (`adapter.py`)
Pre-train the Classical-to-Quantum adapter to map classical features onto a uniform, isotropic distribution across the Bloch sphere prior to VQC optimization:

```bash
python qml_models/adapter.py \
  --dataset pneumoniamnist \
  --n_qubits 4 \
  --epochs 10 \
  --lr 1e-3
```

---

### 5.7 Diagnostic Pipeline Verification (`test_pipeline.py`)
Run the test pipeline to verify that all modules (model instantiations, adapters, quantum nodes, loss functions, and dataset loaders) execute without runtime errors:

```bash
python qml_models/test_pipeline.py
```

---

## 6. Supported Datasets & Clinical Modalities

| Dataset Identifier | Clinical Modality | Diagnostic Target | Classes | Input Dimensions | Default Split (Train/Val/Test) |
|:---|:---|:---|:---:|:---:|:---:|
| `pneumoniamnist` | Pediatric Chest X-Ray | Normal vs. Pneumonia | 2 | 28×28 / 224×224 | 4,708 / 524 / 624 |
| `dermamnist` | Dermatoscopy | Skin Lesions (Melanoma, Basal Cell, Nevi, etc.) | 7 | 28×28 / 224×224 | 7,007 / 1,003 / 2,005 |
| `retinamnist` | Fundus Photography | Diabetic Retinopathy Grading | 5 | 28×28 / 224×224 | 1,080 / 120 / 400 |
| `breastmnist` | Breast Ultrasound | Benign vs. Malignant Masses | 2 | 28×28 / 224×224 | 546 / 78 / 156 |
| `organamnist` | Abdominal Computed Tomography (CT) | Multi-Organ Anatomical Classification | 11 | 28×28 / 224×224 | 34,581 / 6,491 / 17,778 |
| `pathmnist` | Colorectal Histopathology | Tissue Morphology (Adenoma, Mucosa, Stroma) | 9 | 28×28 / 224×224 | 89,996 / 10,004 / 7,180 |
| `covid` | Adult Chest X-Ray | Healthy vs. COVID-19 Infiltration | 2 | 224×224 | 10,616 / 1,516 / 3,033 |
| `oasis` | Cross-Sectional Brain MRI | Cognitive Health vs. Dementia | 2 | 224×224 | 291 / 42 / 83 |

---

## 7. Comprehensive Diagnostic Plotting Suite (39 Plots)

After training or during `--eval_only` execution, `plots.py` renders up to 39 publication-grade diagnostic plots in `outputs/<dataset>/`:

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
├── 10_conformal_intervals.png            # Conformal prediction set sizes across confidence intervals
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

## 8. Troubleshooting & Common Engineering Pitfalls

### 1. Windows Unicode Encoding Errors (`cp1252`)
- **Symptom:** `UnicodeEncodeError: 'charmap' codec can't encode character '\u2500'` when printing CLI status trees or box drawings on Windows.
- **Solution:** Both `engine.py` and `app.py` force standard output streams to UTF-8 at startup:
  ```python
  import sys
  if hasattr(sys.stdout, 'reconfigure'):
      sys.stdout.reconfigure(encoding='utf-8', errors='replace')
  ```

### 2. Host-to-Device Memory Thrashing & CUDA Out of Memory (OOM)
- **Symptom:** System RAM spikes or training halts during evaluation.
- **Solution:** 
  1. Use `--cache_in_vram` only if GPU memory $\ge 8\text{ GB}$.
  2. Cap OpenMP/MKL threads to 4 using environment variables (Section 4.4).
  3. Pass `--fast_eval` to bypass 2D loss landscapes and heavy quantum noise simulations.

### 3. PennyLane Adjoint Differentiation Incompatibilities
- **Symptom:** Gradient backpropagation errors when using `--diff_method adjoint` with trainable classical backbones.
- **Solution:** Adjoint differentiation cannot backpropagate gradients into classical PyTorch modules. Always use the default backprop method:
  ```bash
  --diff_method backprop --qdevice default.qubit
  ```

### 4. Offline Model Loading for Hugging Face & OpenCLIP
- **Symptom:** Network connection timeouts when attempting to load BiomedCLIP weights in restricted or isolated clinical environments.
- **Solution:** Ensure offline environment flags are set:
  ```bash
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  ```
  `app.py` sets these environment flags automatically before importing dependencies.

---

## Citation & Academic Reference

If you use this hybrid quantum-classical framework, clinical calibration pipeline, or anti-shortcut suite in your research, please cite:

```bibtex
@article{qml_early_disease_prediction_2026,
  title   = {Hybrid Classical-Quantum Convolutional Variational Circuits with Dual-Basis Readout and Anti-Shortcut Guarantees for Early Disease Prediction},
  author  = {Advanced Quantum-Classical Clinical AI Research Consortium},
  journal = {IEEE Transactions on Quantum Engineering / Medical Image Analysis},
  year    = {2026}
}
```
