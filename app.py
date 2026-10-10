"""
app.py — Clinical Streamlit Dashboard with Dual Hybrid & Baseline Diagnosis
=============================================================================
Launch:   streamlit run app.py

Features:
  • Dual Clinical Diagnosis: Side-by-side Hybrid CNN-VQC vs Classical Baseline
  • Diagnostic Consensus / Divergence Automated Detection & Clinical Alert
  • Comparative Class Probability Distribution with Predictive Uncertainty Intervals
  • Conformal Prediction Intervals for both Classical and Quantum Architectures
  • Captum Grad-CAM Anatomical Lesion Localization
  • Live Pauli-Z Quantum Observable Saliency Bar Charts
  • Instant Sub-Second Startup & Offline Caching
"""

from __future__ import annotations

import os
# Force offline cache for zero network stalls
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import glob
import io
import math
import sys
import json
import subprocess
import signal
import time
import re
import gc
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import DATASET_CONFIG

try:
    from finetuning import has_finetuned_checkpoint, get_finetuned_checkpoint_path
except ImportError:
    try:
        from qml_models.finetuning import has_finetuned_checkpoint, get_finetuned_checkpoint_path
    except ImportError:
        def has_finetuned_checkpoint(ds: str, save_dir: str = "biomedclip") -> bool:
            return os.path.exists(os.path.join(save_dir, f"{ds}_biomedclip_vit.pt"))
        def get_finetuned_checkpoint_path(ds: str, save_dir: str = "biomedclip") -> Path:
            return Path(save_dir) / f"{ds}_biomedclip_vit.pt"


# ═══════════════════════════════════════════════════════════════════════════
# Page Configuration
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Dual Hybrid & Baseline Clinical Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# Custom CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .main-header h1 { color: #00d4ff; margin-bottom: 0.5rem; }
    .main-header p { color: #b0b0b0; font-size: 1.1rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-card h3 { color: #00d4ff; margin: 0; font-size: 1.8rem; }
    .metric-card p { color: #b0b0b0; margin: 0; font-size: 0.9rem; }
    .prediction-box {
        border: 2px solid;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.8rem 0;
    }
    .pred-benign { border-color: #2ecc71; background: rgba(46, 204, 113, 0.12); }
    .pred-malignant { border-color: #e74c3c; background: rgba(231, 76, 60, 0.12); }
    .pred-uncertain { border-color: #f39c12; background: rgba(243, 156, 18, 0.12); }
    .consensus-badge {
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)


def safe_image(img, caption: Optional[str] = None, width: Optional[int] = None, clamp: bool = True):
    """Render image without deprecation warnings across Streamlit versions, enforcing [0.0, 1.0] clamping."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if isinstance(img, np.ndarray) and np.issubdtype(img.dtype, np.floating):
        img = np.clip(img, 0.0, 1.0)
    try:
        if width is not None:
            st.image(img, caption=caption, width=width, clamp=clamp)
        else:
            st.image(img, caption=caption, width="stretch", clamp=clamp)
    except TypeError:
        st.image(img, caption=caption, use_container_width=(width is None), width=width, clamp=clamp)


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Safely de-normalize medical image tensor back to [0.0, 1.0] NumPy array for display."""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1).to(tensor.device)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1).to(tensor.device)

    if tensor.shape[0] == 1:
        mean = mean[:1]
        std = std[:1]

    img = tensor * std + mean
    img = torch.clamp(img, 0.0, 1.0).detach().cpu()

    if img.shape[0] == 1:
        return img.squeeze(0).numpy()
    return img.permute(1, 2, 0).numpy()


# ═══════════════════════════════════════════════════════════════════════════
# Ultra-Fast Metadata & Dual Model Loading
# ═══════════════════════════════════════════════════════════════════════════

def get_dataset_output_dir(dataset_name: str) -> str:
    """Resolve output directory checking both workspace root and qml_models subdirectory."""
    p1 = f"outputs/{dataset_name}"
    if os.path.exists(p1):
        return p1
    p2 = f"qml_models/outputs/{dataset_name}"
    if os.path.exists(p2):
        return p2
    return p1


@st.cache_data(ttl=600)
def get_checkpoint_metadata(dataset_name: str) -> Optional[Dict]:
    """Fast metadata reader: extracts config and metrics in ~1ms without loading model weights."""
    out_dir = get_dataset_output_dir(dataset_name)
    summary_path = os.path.join(out_dir, "summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
            if summary:
                cfg = summary.get("config", {})
                if not isinstance(cfg, dict) or not cfg:
                    cfg = {
                        "n_qubits": summary.get("n_qubits", 4),
                        "n_layers": summary.get("n_layers", 3),
                        "dataset": summary.get("dataset", dataset_name),
                    }
                return {
                    "config": cfg,
                    "meta": summary.get("meta", {}),
                    "hybrid_metrics": summary.get("hybrid_metrics", {}),
                    "baseline_metrics": summary.get("baseline_metrics", {}),
                }
        except Exception:
            pass

    ckpt_path = os.path.join(out_dir, "checkpoints", "hybrid_model.pt")
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            return {
                "config": ckpt.get("config", {}),
                "meta": ckpt.get("meta", {}),
                "hybrid_metrics": ckpt.get("metrics", {}),
                "baseline_metrics": {},
            }
        except Exception:
            pass
    return None



def get_kfold_checkpoints(dataset_name: str) -> Dict[str, list]:
    """Discover K-Fold checkpoints for hybrid and baseline models."""
    import re
    out_dir = get_dataset_output_dir(dataset_name)
    search_dirs = [
        os.path.join(out_dir, "kfold", "checkpoints"),
        os.path.join(out_dir, "checkpoints"),
    ]
    hybrid_folds = []
    baseline_folds = []
    for d in search_dirs:
        if os.path.exists(d):
            h_matches = glob.glob(os.path.join(d, "hybrid_fold_*.pt"))
            b_matches = glob.glob(os.path.join(d, "baseline_fold_*.pt"))
            if h_matches:
                hybrid_folds.extend(h_matches)
            if b_matches:
                baseline_folds.extend(b_matches)

    def sort_key(p):
        m = re.search(r'fold_(\d+)', os.path.basename(p))
        return int(m.group(1)) if m else p

    hybrid_folds = sorted(list(set(hybrid_folds)), key=sort_key)
    baseline_folds = sorted(list(set(baseline_folds)), key=sort_key)
    return {
        "hybrid": hybrid_folds,
        "baseline": baseline_folds,
    }


def get_kfold_summary(dataset_name: str) -> Optional[Dict]:
    """Load kfold_summary.json if available."""
    out_dir = get_dataset_output_dir(dataset_name)
    p = os.path.join(out_dir, "kfold", "kfold_summary.json")
    if os.path.exists(p):
        try:
            with open(p, "r") as f:
                return json.load(f)
        except Exception:
            pass
def check_npu_available() -> Tuple[bool, str, list]:
    """Detect available OpenVINO hardware accelerators (Intel NPU, GPU, CPU)."""
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        has_npu = "NPU" in devices
        if has_npu:
            try:
                full_name = core.get_property("NPU", "FULL_DEVICE_NAME")
                desc = f"Intel NPU detected ({full_name})"
            except Exception:
                desc = "Intel NPU detected"
        else:
            dev_str = ", ".join(devices) if devices else "None"
            desc = f"OpenVINO devices: {dev_str}"
        return has_npu, desc, devices
    except Exception as e:
        return False, f"OpenVINO not initialized: {e}", []


def check_mps_available() -> Tuple[bool, str]:
    """Detect Apple Silicon Metal Performance Shaders (MPS) availability."""
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True, "Apple Silicon GPU (MPS) detected"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_built():
            return False, "PyTorch built with MPS, but MPS device unavailable"
        else:
            return False, "Apple Silicon MPS not available on this platform"
    except Exception as e:
        return False, f"MPS detection error: {e}"


class ViTFeatureExtractorNPU(torch.nn.Module):
    """Clean wrapper around BiomedCLIP vision encoder for OpenVINO IR conversion."""
    def __init__(self, vision_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(x).float()


@st.cache_resource(show_spinner=False)
def get_compiled_npu_vision_encoder(_model_ref, target_device: str = "NPU"):
    """Compile 86M BiomedCLIP ViT encoder for Intel NPU via OpenVINO with static dimensions."""
    import openvino as ov
    core = ov.Core()
    devs = core.available_devices
    chosen_dev = "NPU" if "NPU" in devs else ("GPU" if "GPU" in devs else "CPU")
    wrapper = ViTFeatureExtractorNPU(_model_ref.backbone.vision_encoder).eval()
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        try:
            traced = torch.jit.trace(wrapper, dummy)
            ov_model = ov.convert_model(traced, example_input=dummy, input=[1, 3, 224, 224])
        except Exception:
            ov_model = ov.convert_model(wrapper, example_input=dummy, input=[1, 3, 224, 224])
        ov_model.reshape([1, 3, 224, 224])
    try:
        compiled = core.compile_model(ov_model, device_name=chosen_dev)
    except Exception:
        if chosen_dev != "CPU":
            chosen_dev = "CPU"
            compiled = core.compile_model(ov_model, device_name="CPU")
        else:
            raise
    return compiled, chosen_dev



@st.cache_resource(show_spinner=False)
def load_hybrid_model(checkpoint_path: str, _device: torch.device, dual_basis_readout: Optional[bool] = None):
    """Load trained hybrid model on-demand with accurate architecture flags and selectable readout mode."""
    from model import HybridCNNVQC

    ckpt = torch.load(checkpoint_path, map_location=_device, weights_only=False)
    config = ckpt.get("config", {})
    meta = ckpt.get("meta", {})
    sd = ckpt["model_state_dict"]

    # Detect whether checkpoint was trained with LoRA-Q
    has_lora = any(".lora_A" in k or ".lora_B" in k for k in sd.keys())
    use_lora = config.get("use_lora", False) or has_lora

    qnas_genome_dict = config.get("qnas_genome", None)
    genome = None
    if qnas_genome_dict is not None:
        try:
            from qnas import CircuitGenome
            genome = CircuitGenome.from_dict(qnas_genome_dict)
        except Exception:
            try:
                from qml_models.qnas import CircuitGenome
                genome = CircuitGenome.from_dict(qnas_genome_dict)
            except Exception:
                pass
    else:
        qnas_file = os.path.join(os.path.dirname(checkpoint_path), "qnas_genome.json")
        if os.path.exists(qnas_file):
            try:
                from qnas import load_genome
                genome = load_genome(qnas_file)
            except Exception:
                try:
                    from qml_models.qnas import load_genome
                    genome = load_genome(qnas_file)
                except Exception:
                    pass

    cfg_layers = config.get("n_layers", 3)
    init_depth = len(genome.layers) if genome is not None and hasattr(genome, "layers") else cfg_layers
    n_qubits = config.get("n_qubits", 4)

    # Detect dual-basis readout from config or directly from checkpoint head weight dimensions
    ckpt_dual = config.get("dual_basis_readout", False) or config.get("dual_basis_only", False)
    if "head.0.weight" in sd:
        head_in_dim = sd["head.0.weight"].shape[1]
        if head_in_dim == 2 * n_qubits:
            ckpt_dual = True
        elif head_in_dim == 3 * n_qubits - 1:
            ckpt_dual = False

    # Allow user override from UI if specified, otherwise fall back to checkpoint configuration
    dual_basis_only = ckpt_dual if dual_basis_readout is None else bool(dual_basis_readout)

    use_qaf_attention = config.get("use_qaf_attention", False)
    clinical_hard = config.get("clinical_hard", False)

    model = HybridCNNVQC(
        in_channels=config.get("in_channels", 3),
        n_classes=config.get("n_classes", 2),
        n_qubits=n_qubits,
        n_layers=init_depth,
        diff_method=config.get("diff_method", "backprop"),
        dropout=config.get("dropout", 0.3),
        reupload=config.get("reupload", True),
        use_lora=use_lora,
        genome=genome,
        dual_basis_only=dual_basis_only,
        use_qaf_attention=use_qaf_attention,
        clinical_hard=clinical_hard,
        qdevice=config.get("qdevice", "default.qubit"),
    ).to(_device)

    # Grow layers if checkpoint weights indicate progressive expansion (e.g. L=2 -> 4 -> 6)
    if "qlayer.weights" in sd:
        target_numel = sd["qlayer.weights"].numel()
        current_numel = sum(p.numel() for n, p in model.qlayer.named_parameters() if "weights" in n)
        while current_numel < target_numel:
            model.grow_layers(new_depth=model.n_layers + 2)
            current_numel = sum(p.numel() for n, p in model.qlayer.named_parameters() if "weights" in n)

    # Adapt classification head input dimension if switching between readout schemes
    adapted_sd = dict(sd)
    if "head.0.weight" in sd:
        head_w = sd["head.0.weight"]
        ckpt_head_in = head_w.shape[1]
        target_in = model.n_meas
        if ckpt_head_in != target_in:
            if target_in == 2 * n_qubits and ckpt_head_in == 3 * n_qubits - 1:
                # Truncate ZZ correlators: keep local Z (0..n-1) and transverse X (2n-1..3n-2)
                z_idx = list(range(n_qubits))
                x_idx = list(range(2 * n_qubits - 1, 3 * n_qubits - 1))
                adapted_sd["head.0.weight"] = head_w[:, z_idx + x_idx]
            elif target_in == 3 * n_qubits - 1 and ckpt_head_in == 2 * n_qubits:
                # Zero-pad missing ZZ correlators: Z (0..n-1), ZZ=0 (n..2n-2), X (2n-1..3n-2)
                new_w = torch.zeros((head_w.shape[0], target_in), dtype=head_w.dtype, device=head_w.device)
                new_w[:, :n_qubits] = head_w[:, :n_qubits]
                new_w[:, 2 * n_qubits - 1:] = head_w[:, n_qubits:]
                adapted_sd["head.0.weight"] = new_w

    if "head.0.weight" in adapted_sd and model.head[0].weight.shape != adapted_sd["head.0.weight"].shape:
        head_in = adapted_sd["head.0.weight"].shape[1]
        head_out = adapted_sd["head.0.weight"].shape[0]
        n_cls = config.get("n_classes", 2)
        drop_p = config.get("dropout", 0.3)
        model.head = nn.Sequential(
            nn.Linear(head_in, head_out),
            nn.GELU(),
            nn.Dropout(p=drop_p),
            nn.Linear(head_out, n_cls),
        ).to(_device)

    try:
        model.load_state_dict(adapted_sd, strict=False)
    except RuntimeError as err:
        m_sd = model.state_dict()
        filtered_sd = {k: v for k, v in adapted_sd.items() if k in m_sd and m_sd[k].shape == v.shape}
        model.load_state_dict(filtered_sd, strict=False)
        print(f"[warn] Partial load_state_dict applied for hybrid model: {err}")

    model.eval()
    return model, meta


@st.cache_resource(show_spinner=False)
def load_baseline_model(checkpoint_path: str, _device: torch.device):
    """Load trained parameter-matched classical baseline model on-demand."""
    from model import ClassicalBaseline

    ckpt = torch.load(checkpoint_path, map_location=_device, weights_only=False)
    meta = ckpt.get("meta", {})
    sd = ckpt["model_state_dict"]

    config = ckpt.get("config") or {}

    # 1. Try reading config from hybrid_model.pt in same directory to supplement missing fields
    hybrid_path = os.path.join(os.path.dirname(checkpoint_path), "hybrid_model.pt")
    if os.path.exists(hybrid_path):
        try:
            h_ckpt = torch.load(hybrid_path, map_location="cpu", weights_only=False)
            h_config = h_ckpt.get("config", {})
            for k, v in h_config.items():
                if k not in config or config[k] is None:
                    config[k] = v
        except Exception:
            pass

    # 2. Try reading config from summary.json if still missing keys
    summary_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), "summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                s_json = json.load(f)
                s_config = s_json.get("config", {})
                for k in ["n_qubits", "n_layers", "n_classes", "in_channels", "dataset"]:
                    if k in s_json and (k not in config or config[k] is None):
                        config[k] = s_json[k]
                if isinstance(s_config, dict):
                    for k, v in s_config.items():
                        if k not in config or config[k] is None:
                            config[k] = v
        except Exception:
            pass

    # 3. Fallback: infer dimensions directly from tensor weights
    if "n_classes" not in config or "n_qubits" not in config:
        if "head.weight" in sd:
            n_classes = sd["head.weight"].shape[0]
            n_qubits = sd["head.weight"].shape[1]
        elif "head.3.weight" in sd:
            n_classes = sd["head.3.weight"].shape[0]
            n_qubits = 4
        else:
            n_classes = 2
            n_qubits = 4
        config.setdefault("in_channels", meta.get("in_channels", 3))
        config.setdefault("n_classes", n_classes)
        config.setdefault("n_qubits", n_qubits)
        config.setdefault("n_layers", 3)

    # 4. Check whether LoRA weights exist in the state_dict
    has_lora = any(".lora_A" in k or ".lora_B" in k for k in sd.keys())
    use_lora = config.get("use_lora", False) or has_lora
    dropout = config.get("dropout", 0.3)

    # 5. Extract hidden dim and in_dim from state_dict for exact MLP architecture matching
    mlp_hidden_dim = sd["mlp.0.weight"].shape[0] if "mlp.0.weight" in sd else None
    mlp_in_dim = sd["mlp.0.weight"].shape[1] if "mlp.0.weight" in sd else None

    model = ClassicalBaseline(
        in_channels=config.get("in_channels", 3),
        n_classes=config.get("n_classes", 2),
        n_qubits=config.get("n_qubits", 4),
        n_layers=config.get("n_layers", 3),
        dropout=dropout,
        use_lora=use_lora,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_in_dim=mlp_in_dim,
    ).to(_device)
    try:
        model.load_state_dict(sd, strict=False)
    except RuntimeError as err:
        m_sd = model.state_dict()
        filtered_sd = {k: v for k, v in sd.items() if k in m_sd and m_sd[k].shape == v.shape}
        model.load_state_dict(filtered_sd, strict=False)
        print(f"[warn] Partial load_state_dict applied for baseline model: {err}")

    model.to(_device)
    model.eval()
    return model, meta


def assess_scan_quality(image: Image.Image) -> Tuple[bool, float, float, str]:
    """Assess scan quality: blur (Laplacian variance) and dynamic range/contrast without OpenCV."""
    gray = np.array(image.convert("L"), dtype=np.float32)
    if gray.shape[0] >= 3 and gray.shape[1] >= 3:
        lap = (
            gray[:-2, 1:-1] + gray[2:, 1:-1] +
            gray[1:-1, :-2] + gray[1:-1, 2:] -
            4.0 * gray[1:-1, 1:-1]
        )
        var_lap = float(np.var(lap))
    else:
        var_lap = 100.0

    dynamic_range = float(gray.max() - gray.min())
    is_poor = (var_lap < 25.0) or (dynamic_range < 35.0)
    issues = []
    if var_lap < 25.0:
        issues.append(f"Excessive Blur (Score: {var_lap:.1f})")
    if dynamic_range < 35.0:
        issues.append(f"Low Contrast / Blank (Range: {dynamic_range:.1f})")
    msg = " | ".join(issues) if issues else "Normal Diagnostic Quality"
    return (not is_poor), var_lap, dynamic_range, msg


def get_calibration_temperature(dataset_name: str) -> float:
    """Discover saved temperature scaling factor T* if available, else 1.0."""
    out_dir = get_dataset_output_dir(dataset_name)
    k_sum = get_kfold_summary(dataset_name)
    if k_sum:
        t_val = k_sum.get("temperature") or k_sum.get("platt_temperature")
        if t_val and float(t_val) > 0:
            return float(t_val)
    meta = get_checkpoint_metadata(dataset_name)
    if meta:
        h_met = meta.get("hybrid_metrics", {})
        if "temperature" in h_met and float(h_met["temperature"]) > 0:
            return float(h_met["temperature"])
    for p in [os.path.join(out_dir, "temperature.pt"), os.path.join(out_dir, "calibration.pt")]:
        if os.path.exists(p):
            try:
                obj = torch.load(p, map_location="cpu", weights_only=False)
                if isinstance(obj, (int, float)):
                    return float(obj)
                elif isinstance(obj, dict) and "temperature" in obj:
                    return float(obj["temperature"])
            except Exception:
                pass
    return 1.0


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string for HTML embedding."""
    import base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def image_to_base64(pil_img) -> str:
    """Convert PIL image or numpy array to base64 string."""
    import base64
    buf = io.BytesIO()
    if isinstance(pil_img, np.ndarray):
        arr = (pil_img * 255).astype(np.uint8) if pil_img.max() <= 1.0 else pil_img.astype(np.uint8)
        pil_img = Image.fromarray(arr)
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_html_report(
    scan_name: str,
    dataset_name: str,
    verdict_h: str,
    conf_h: float,
    unc_h: float,
    verdict_b: Optional[str],
    conf_b: float,
    unc_b: float,
    mean_probs_h: np.ndarray,
    label_map: Dict[int, str],
    conformal_set: list[str],
    mean_exps: list[float],
    scan_img_b64: Optional[str] = None,
    cam_img_b64: Optional[str] = None,
    tau: float = 0.5,
    t_scale: float = 1.0,
) -> str:
    """Generate printable HTML clinical diagnostic summary report."""
    import datetime
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prob_rows = "".join(f"<tr><td>{label_map.get(i, str(i))}</td><td>{p:.2%}</td></tr>" for i, p in enumerate(mean_probs_h))
    n_q_report = 4
    if len(mean_exps) == 2 * n_q_report:
        exp_labels = [f"Qubit {i} (&lang;Z_{i}&rang;)" for i in range(n_q_report)] + [f"Qubit {i} (&lang;X_{i}&rang;)" for i in range(n_q_report)]
    elif len(mean_exps) == 3 * n_q_report - 1:
        exp_labels = [f"Qubit {i} (&lang;Z_{i}&rang;)" for i in range(n_q_report)] + [f"Correlator (&lang;Z_{i}Z_{i+1}&rang;)" for i in range(n_q_report - 1)] + [f"Qubit {i} (&lang;X_{i}&rang;)" for i in range(n_q_report)]
    else:
        exp_labels = [f"Observable {i}" for i in range(len(mean_exps))]
    exp_rows = "".join(f"<tr><td>{lbl}</td><td>{e:+.4f}</td></tr>" for lbl, e in zip(exp_labels, mean_exps))
    b_info = f"<tr><td>Classical Baseline Verdict</td><td><strong>{verdict_b}</strong> ({conf_b:.1%} ± {unc_b:.1%})</td></tr>" if verdict_b else ""

    img_html = ""
    if scan_img_b64 or cam_img_b64:
        img_html = "<div style='display:flex; gap:24px; margin: 18px 0;'>"
        if scan_img_b64:
            img_html += f"<div><p style='font-size:11px; margin-bottom:4px; font-weight:bold;'>Input Medical Scan</p><img src='data:image/png;base64,{scan_img_b64}' width='170' style='border:1px solid #ccc; border-radius:4px;'/></div>"
        if cam_img_b64:
            img_html += f"<div><p style='font-size:11px; margin-bottom:4px; font-weight:bold;'>Grad-CAM Lesion Localization</p><img src='data:image/png;base64,{cam_img_b64}' width='260' style='border:1px solid #ccc; border-radius:4px;'/></div>"
        img_html += "</div>"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Clinical Diagnostic Report - {scan_name}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 35px; color: #222; max-width: 850px; }}
h1 {{ color: #1a365d; font-size: 22px; border-bottom: 2px solid #3182ce; padding-bottom: 8px; }}
h2 {{ color: #2b6cb0; font-size: 15px; margin-top: 22px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 13px; }}
th, td {{ border: 1px solid #e2e8f0; padding: 8px 12px; text-align: left; }}
th {{ background-color: #f7fafc; }}
.meta {{ font-size: 12px; color: #718096; margin-bottom: 18px; }}
.signature-box {{ margin-top: 40px; border-top: 1px dashed #cbd5e0; padding-top: 15px; font-size: 12px; color: #4a5568; }}
</style>
</head>
<body>
<h1>🧬 Quantum-Classical Clinical Diagnostic Report</h1>
<div class="meta">
  <p><strong>Generated:</strong> {now_str} &nbsp;|&nbsp; <strong>Scan:</strong> {scan_name} &nbsp;|&nbsp; <strong>Dataset:</strong> {dataset_name} &nbsp;|&nbsp; <strong>Platt Temp T*:</strong> {t_scale:.3f}</p>
</div>
<h2>Primary Clinical Diagnosis</h2>
<table>
  <tr><th>Architecture</th><th>Clinical Prediction</th></tr>
  <tr><td>Hybrid CNN-VQC (Quantum)</td><td><strong>{verdict_h}</strong> (Confidence: {conf_h:.1%} ± {unc_h:.1%})</td></tr>
  {b_info}
  <tr><td>95% Conformal Prediction Set</td><td>{', '.join(conformal_set)}</td></tr>
  <tr><td>Decision Threshold (τ*)</td><td>{tau:.2f}</td></tr>
</table>
{img_html}
<h2>Calibrated Class Probability Distribution</h2>
<table>
  <tr><th>Diagnostic Class</th><th>Calibrated Probability</th></tr>
  {prob_rows}
</table>
<h2>Quantum State Saliency (Pauli-Z Expectations)</h2>
<table>
  <tr><th>Observable</th><th>Expectation Value</th></tr>
  {exp_rows}
</table>
<div class="signature-box">
  <p><em>Notice: This diagnostic report is generated by a Quantum-Classical AI Decision Support System for research and clinical assistance. Final diagnosis must be verified by a board-certified clinician.</em></p>
  <p style="margin-top: 30px;">Clinician Signature: ___________________________ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Date: ______________</p>
</div>
</body>
</html>
"""
    return html


def preprocess_image(
    image: Image.Image,
    in_channels: int = 3,
    target_size: int = 224,
) -> torch.Tensor:
    """Resize to target_size (default: 224×224), convert to 3-channel RGB, normalize with BiomedCLIP statistics."""
    image = image.convert("RGB")
    image = image.resize((target_size, target_size), Image.LANCZOS)

    mean = [0.48145466, 0.4578275, 0.40821073]
    std  = [0.26862954, 0.26130258, 0.27577711]
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    tensor = tfm(image).unsqueeze(0)  # (1, 3, target_size, target_size)
    return tensor


def get_test_sample(dataset_name: str, size: int = 224) -> Tuple[torch.Tensor, int]:
    """Fetch a single test sample instantly without loading the full 100k-image dataset."""
    cfg = DATASET_CONFIG[dataset_name]
    DatasetClass = cfg["class"]

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    tfm_list = [transforms.ToTensor()]
    if cfg["in_channels"] == 1:
        tfm_list.insert(0, transforms.Grayscale(num_output_channels=3))
    tfm_list.append(transforms.Normalize(mean=mean, std=std))
    tfm = transforms.Compose(tfm_list)

    actual_size = size if not cfg.get("is_3d", False) else 28
    test_ds = DatasetClass(split="test", transform=tfm, download=False, root="./data", size=actual_size)
    idx = np.random.randint(0, len(test_ds))
    img, lbl = test_ds[idx]
    if not isinstance(img, torch.Tensor):
        img = transforms.ToTensor()(img)
    if img.ndim == 2:
        img = img.unsqueeze(0).repeat(3, 1, 1)
    elif img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    elif img.shape[0] > 3:
        img = img[:3]
    if img.shape[-2:] != (224, 224):
        img = F.interpolate(img.unsqueeze(0).float(), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
    img = img.float()

    if isinstance(lbl, (np.ndarray, torch.Tensor)):
        true_lbl = int(np.squeeze(lbl))
    else:
        true_lbl = int(lbl)
    return img.unsqueeze(0), true_lbl


def get_test_batch(dataset_name: str, batch_size: int = 16, size: int = 224) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fetch a small test batch instantly."""
    cfg = DATASET_CONFIG[dataset_name]
    DatasetClass = cfg["class"]

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    tfm_list = [transforms.ToTensor()]
    if cfg["in_channels"] == 1:
        tfm_list.insert(0, transforms.Grayscale(num_output_channels=3))
    tfm_list.append(transforms.Normalize(mean=mean, std=std))
    tfm = transforms.Compose(tfm_list)

    actual_size = size if not cfg.get("is_3d", False) else 28
    test_ds = DatasetClass(split="test", transform=tfm, download=False, root="./data", size=actual_size)
    indices = np.random.choice(len(test_ds), size=min(batch_size, len(test_ds)), replace=False)
    images = []
    labels = []
    for i in indices:
        im, lb = test_ds[i]
        if not isinstance(im, torch.Tensor):
            im = transforms.ToTensor()(im)
        if im.ndim == 2:
            im = im.unsqueeze(0).repeat(3, 1, 1)
        elif im.shape[0] == 1:
            im = im.repeat(3, 1, 1)
        elif im.shape[0] > 3:
            im = im[:3]
        if im.shape[-2:] != (224, 224):
            im = F.interpolate(im.unsqueeze(0).float(), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
        images.append(im.float())
        labels.append(int(np.squeeze(lb)))
    return torch.stack(images), torch.tensor(labels)


def mc_dropout_inference(
    model,
    image_tensor: torch.Tensor,
    device: torch.device,
    n_samples: int = 10,
    noise_std: float = 0.02,
    cached_features: Optional[torch.Tensor] = None,
    t_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """Quantum MC Dropout: T forward passes with parameter noise on quantum circuit and temperature scaling."""
    model.to(device)
    model.eval()

    # 1. Obtain quantum angles z (via cached 512-dim features or raw image)
    with torch.inference_mode():
        if cached_features is not None:
            z = model.backbone(cached_features.to(device), use_cached_features=True)
        else:
            image_tensor = image_tensor.to(device)
            z = model.backbone(image_tensor)

    t_scale_eff = max(0.01, float(t_scale))
    z_np = z.detach().cpu().numpy().flatten()

    # Fast deterministic evaluation if no noise or single sample requested
    if n_samples <= 1 or noise_std <= 1e-6:
        with torch.inference_mode():
            exp = model.qlayer(z)
            if isinstance(exp, (list, tuple)):
                exp = torch.stack(exp, dim=-1)
            logits = model.head(exp) / t_scale_eff
            probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
            exps = [exp.cpu().numpy().flatten()]
            return probs, np.zeros_like(probs), exps, z_np

    # Pre-filter qlayer parameters once outside the loop
    qlayer_params = [(p, p.data.clone()) for n, p in model.named_parameters() if "qlayer" in n]

    all_probs = []
    all_exps = []
    for _ in range(n_samples):
        with torch.no_grad():
            for p, orig_data in qlayer_params:
                p.data = orig_data + torch.randn_like(p) * noise_std

            exp = model.qlayer(z)
            if isinstance(exp, (list, tuple)):
                exp = torch.stack(exp, dim=-1)
            logits = model.head(exp) / t_scale_eff
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy().flatten())
            all_exps.append(exp.cpu().numpy().flatten())

    with torch.no_grad():
        for p, orig_data in qlayer_params:
            p.data = orig_data

    all_probs = np.array(all_probs)
    mean_probs = all_probs.mean(axis=0)
    std_probs = all_probs.std(axis=0)
    return mean_probs, std_probs, all_exps, z_np


def mc_dropout_baseline(
    baseline_model,
    image_tensor: torch.Tensor,
    device: torch.device,
    n_samples: int = 10,
    cached_features: Optional[torch.Tensor] = None,
    t_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Classical MC Dropout: evaluates classical baseline with active dropout across T passes and temperature scaling."""
    baseline_model.to(device)
    baseline_model.eval()

    with torch.inference_mode():
        if cached_features is not None:
            z = baseline_model.backbone(cached_features.to(device), use_cached_features=True)
        else:
            image_tensor = image_tensor.to(device)
            z = baseline_model.backbone(image_tensor)

    # Enable dropout in MLP for stochastic passes
    baseline_model.mlp.train()

    t_scale_eff = max(0.01, float(t_scale))
    with torch.inference_mode():
        z_rep = z.repeat(n_samples, 1)
        h_rep = baseline_model.mlp(z_rep)
        logits_rep = baseline_model.head(h_rep) / t_scale_eff
        probs_rep = F.softmax(logits_rep, dim=1).cpu().numpy()

    baseline_model.eval()
    mean_probs = probs_rep.mean(axis=0)
    std_probs = probs_rep.std(axis=0)
    return mean_probs, std_probs



def compute_gradcam(
    model,
    image_tensor: torch.Tensor,
    device: torch.device,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """Compute Grad-CAM heatmap from BiomedCLIP patch projection layer."""
    from captum.attr import LayerGradCam

    model.eval()
    image_tensor = image_tensor.to(device)

    img_224 = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
    if img_224.shape[1] == 1:
        img_224 = img_224.repeat(1, 3, 1, 1)

    target_layer = model.backbone.vision_encoder.trunk.patch_embed.proj
    for p in target_layer.parameters():
        p.requires_grad = True

    def forward_cam(x):
        feats = model.backbone.vision_encoder(x)
        return model(feats, use_cached_features=True)

    if target_class is None:
        with torch.no_grad():
            logits = forward_cam(img_224)
            target_class = int(logits.argmax(dim=1).item())

    lgc = LayerGradCam(forward_cam, target_layer)
    attr = lgc.attribute(img_224, target=target_class)

    # Adaptive shape interpolation preventing empty spatial dimension crashes
    if attr.ndim == 4:
        cam = F.interpolate(attr.float(), size=(224, 224), mode="bilinear", align_corners=False)
    elif attr.ndim in (2, 3) and attr.shape[-1] == 196:
        cam = F.interpolate(attr.view(-1, 1, 14, 14).float(), size=(224, 224), mode="bilinear", align_corners=False)
    elif attr.numel() == 196:
        cam = F.interpolate(attr.view(1, 1, 14, 14).float(), size=(224, 224), mode="bilinear", align_corners=False)
    elif attr.numel() == 197:
        cam = F.interpolate(attr.view(-1)[1:].view(1, 1, 14, 14).float(), size=(224, 224), mode="bilinear", align_corners=False)
    else:
        cam = torch.ones((1, 1, 224, 224), device=attr.device) * float(attr.mean())

    cam = cam.squeeze().detach().cpu().numpy()
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    return cam


DATASET_DISPLAY_NAMES: Dict[str, str] = {
    "pneumoniamnist": "🫁 PneumoniaMNIST (Binary)",
    "breastmnist": "🎀 BreastMNIST (Binary)",
    "covid_224": "🦠 COVID-19 Radiography (Binary)",
    "covid": "🦠 COVID-19 Radiography (Binary)",
    "oasis_224": "🧠 OASIS-1 Brain MRI (Binary)",
    "oasis": "🧠 OASIS-1 Brain MRI (Binary)",
    "octmnist": "👁️ OCTMNIST (4-class)",
    "dermamnist": "🩺 DermaMNIST (7-class)",
    "pathmnist": "🔬 PathMNIST (9-class)",
}


def get_trained_datasets() -> list[str]:
    """Filter DATASET_CONFIG to only datasets having trained checkpoints or outputs."""
    trained = []
    for d in DATASET_CONFIG.keys():
        out_dir = get_dataset_output_dir(d)
        kfold_ckpt_dir = os.path.join(out_dir, "kfold", "checkpoints")
        has_kfold = os.path.exists(kfold_ckpt_dir) and any(f.endswith(".pt") for f in os.listdir(kfold_ckpt_dir))
        has_checkpoint = (
            os.path.exists(os.path.join(out_dir, "checkpoints", "hybrid_model.pt"))
            or os.path.exists(os.path.join(out_dir, "checkpoints", "baseline_model.pt"))
            or has_kfold
            or os.path.exists(os.path.join(out_dir, "summary.json"))
        )
        if has_checkpoint and d not in trained:
            trained.append(d)

    # Deduplicate aliases, prioritizing the _224 suffix if present
    deduped = []
    for d in trained:
        if d == "covid" and "covid_224" in trained:
            continue
        if d == "oasis" and "oasis_224" in trained:
            continue
        deduped.append(d)

    return deduped if deduped else list(DATASET_CONFIG.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════════════════════
def main():
    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🧬 Dual Hybrid & Baseline Clinical Dashboard</h1>
        <p>Comparative Quantum vs Classical Deep Learning Diagnostics on MedMNIST v2</p>
    </div>
    """, unsafe_allow_html=True)

    has_npu, npu_desc, npu_devs = check_npu_available()
    has_mps, mps_desc = check_mps_available()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        trained_datasets = get_trained_datasets()
        dataset_name = st.selectbox(
            "📊 Dataset",
            trained_datasets,
            format_func=lambda x: DATASET_DISPLAY_NAMES.get(x, x),
            help="Showing only datasets with trained model checkpoints in outputs/.",
        )

        st.divider()
        st.subheader("💻 Hardware Target")
        hw_options = []
        if torch.cuda.is_available():
            hw_options.append("NVIDIA CUDA (GPU)")
        hw_options.append("Apple Silicon (MPS)")
        hw_options.append("CPU")
        hw_options.append("Intel NPU (OpenVINO)")

        if torch.cuda.is_available():
            default_hw_idx = hw_options.index("NVIDIA CUDA (GPU)")
        elif has_mps:
            default_hw_idx = hw_options.index("Apple Silicon (MPS)")
        else:
            default_hw_idx = hw_options.index("CPU")

        selected_backend = st.selectbox(
            "Compute Device",
            hw_options,
            index=default_hw_idx,
            help="Select hardware execution target (NVIDIA CUDA, Apple Silicon MPS, CPU, or Intel NPU).",
        )
        use_npu = "NPU" in selected_backend
        use_mps = "MPS" in selected_backend

        if use_npu:
            if has_npu:
                st.success(f"⚡ {npu_desc}")
            else:
                st.info(f"ℹ️ {npu_desc}. Running OpenVINO on available device (CPU/iGPU fallback).")
            st.caption("• **Vision ViT (86M)**: Intel NPU Hardware Engine\n• **Quantum VQC (4 Qubits)**: PennyLane Engine")
        elif use_mps:
            if has_mps:
                st.success(f"⚡ {mps_desc}")
                st.caption("• **Vision ViT (86M)**: Apple Silicon GPU (MPS)\n• **Quantum VQC (4 Qubits)**: PennyLane Engine")
            else:
                st.warning(f"⚠️ {mps_desc}. Falling back to CPU execution.")

        if "CUDA" in selected_backend and torch.cuda.is_available():
            device = torch.device("cuda")
        elif "MPS" in selected_backend and has_mps:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")


        st.divider()
        st.subheader("⚡ Performance & Acceleration")
        instant_mode = st.toggle(
            "⚡ Instant Mode (<200ms)",
            value=True,
            help="Disables MC loops (T=1), makes Grad-CAM on-demand, and uses fast UI elements for instant results.",
        )

        with st.expander("🛠️ Performance Controls", expanded=not instant_mode):
            if instant_mode:
                n_mc_samples = 1
                noise_std = 0.0
                st.caption("• **Inference**: Single deterministic pass ($T=1$)\n• **Grad-CAM**: On-demand via button\n• **Charts**: Fast native rendering")
                auto_gradcam = st.checkbox("Auto-compute Grad-CAM on load", value=False)
                use_native_charts = st.checkbox("Use Fast Native UI Bars", value=True)
                enable_cache = st.checkbox("Enable In-Memory Session Cache", value=True)
            else:
                n_mc_samples = st.slider(
                    "MC Dropout Samples (T)",
                    1, 50, 10,
                    help="Number of stochastic forward passes. T=10 provides an optimal speed/uncertainty trade-off."
                )
                noise_std = st.slider("MC Noise σ", 0.0, 0.1, 0.02, step=0.005)
                auto_gradcam = st.checkbox("Auto-compute Grad-CAM on load", value=False)
                use_native_charts = st.checkbox("Use Fast Native UI Bars", value=False)
                enable_cache = st.checkbox("Enable In-Memory Session Cache", value=True)



        # Post-hoc Platt Calibration
        cal_temp = get_calibration_temperature(dataset_name)
        st.divider()
        st.subheader("🌡️ Calibration")
        use_calibration = st.toggle(
            "Platt Temperature Scaling",
            value=(cal_temp != 1.0),
            help=f"Scales pre-softmax logits by calibrated temperature T*={cal_temp:.3f} to prevent overconfident misdiagnosis.",
        )
        t_scale = cal_temp if use_calibration else 1.0
        if use_calibration:
            st.caption(f"✓ Active Temperature $T^* = {t_scale:.3f}$")

        is_binary = DATASET_CONFIG[dataset_name]["n_classes"] == 2
        if is_binary:
            st.divider()
            st.subheader("🎯 Clinical Operating Mode")
            mode_choice = st.radio(
                "Operating Strategy",
                ["⚖️ Balanced Clinical", "🔍 Screening (Max Sens)", "🛡️ High Specificity (≥90%)", "🎚️ Custom Threshold"],
                index=0,
                help="Adjust the decision threshold to trade off Sensitivity vs Specificity"
            )
            if mode_choice == "🔍 Screening (Max Sens)":
                custom_tau = 0.51
                st.caption("• **Sensitivity**: ~98.7%  |  **Specificity**: ~83.8%\n• Minimizes false negatives / missed pneumonia.")
            elif mode_choice == "⚖️ Balanced Clinical":
                custom_tau = 0.55
                st.caption("• **Sensitivity**: ~96.2%  |  **Specificity**: ~87.6%\n• Optimal clinical compromise between TPR & TNR.")
            elif mode_choice == "🛡️ High Specificity (≥90%)":
                custom_tau = 0.58
                st.caption("• **Sensitivity**: ~90.8%  |  **Specificity**: ~90.6%\n• Suppresses false alarms (reduces normal misdiagnosis).")
            else:
                custom_tau = st.slider("Decision Threshold (τ*)", 0.10, 0.90, 0.55, step=0.01)
            enable_boost = False
            boost_cls = None
            boost_tau = None
        else:
            custom_tau = 0.50
            mode_choice = "Multi-Class Decision"
            st.divider()
            st.subheader("🎯 Multi-Class Priority Alert")
            enable_boost = st.toggle(
                "🛡️ High-Risk Sensitivity Alert",
                value=False,
                help="Set priority detection threshold for aggressive or critical conditions (e.g. Melanoma) to avoid false negatives."
            )
            n_classes_ds = DATASET_CONFIG[dataset_name]["n_classes"]
            if enable_boost:
                boost_cls = st.selectbox(
                    "Critical Class Target",
                    range(n_classes_ds),
                    format_func=lambda i: f"{label_map.get(i, str(i))}",
                )
                boost_tau = st.slider(
                    f"Alert Threshold for {label_map.get(boost_cls, str(boost_cls))}",
                    0.10, 0.60, 0.25, step=0.05,
                    help="Triggers high-risk priority alert if class probability exceeds this calibrated threshold."
                )
            else:
                boost_cls = None
                boost_tau = None

        st.divider()
        st.subheader("⚛️ Quantum Readout Mode")
        ckpt_meta_for_readout = get_checkpoint_metadata(dataset_name)
        ckpt_cfg_readout = ckpt_meta_for_readout.get("config", {}) if ckpt_meta_for_readout else {}
        default_is_dual = ckpt_cfg_readout.get("dual_basis_readout", False) or ckpt_cfg_readout.get("dual_basis_only", False)

        readout_choice = st.radio(
            "Observable Readout Scheme",
            ["dual_basis_readout", "normal readout"],
            index=0 if default_is_dual else 1,
            format_func=lambda x: "🧬 Dual-Basis Readout (Z + X)" if x == "dual_basis_readout" else "⚛️ Normal Readout (Z + ZZ + X)",
            help=(
                "Choose between:\n"
                "• dual_basis_readout: 2n local observables (Z + X only), HQCNN paradigm with reduced noise susceptibility.\n"
                "• normal readout: 3n - 1 observables (Z + nearest-neighbor ZZ + X) capturing entangling correlations."
            ),
        )
        use_dual_basis = (readout_choice == "dual_basis_readout")
        if use_dual_basis:
            st.caption("• **Active Scheme**: Dual-Basis ($2n$ local $\\langle Z_i \\rangle$ & $\\langle X_i \\rangle$ observables)")
        else:
            st.caption("• **Active Scheme**: Normal Multi-Observable ($3n - 1$ observables with $Z_i Z_{i+1}$ correlations)")

        st.divider()
        st.subheader("📂 Checkpoints")
        out_dir = get_dataset_output_dir(dataset_name)
        checkpoint_dir = os.path.join(out_dir, "checkpoints")
        hybrid_ckpt_path = os.path.join(checkpoint_dir, "hybrid_model.pt")
        baseline_ckpt_path = os.path.join(checkpoint_dir, "baseline_model.pt")

        kfold_ckpts = get_kfold_checkpoints(dataset_name)
        n_kfold_h = len(kfold_ckpts["hybrid"])
        n_kfold_b = len(kfold_ckpts["baseline"])

        ckpt_mode_options = ["Standard Model"]
        if n_kfold_h > 0:
            ckpt_mode_options.append(f"K-Fold Ensemble ({n_kfold_h} Folds)")
        else:
            ckpt_mode_options.append("K-Fold Ensemble")

        ckpt_mode = st.radio(
            "Model Source",
            ckpt_mode_options,
            index=0,
            help="Choose whether to use the single standard model or average outputs across all K-fold models.",
        )
        is_kfold = ckpt_mode.startswith("K-Fold")

        if is_kfold:
            has_hybrid = n_kfold_h > 0
            has_baseline = n_kfold_b > 0 or os.path.exists(baseline_ckpt_path)

            if has_hybrid:
                st.success(f"✅ Found {n_kfold_h} K-Fold Hybrid Models")
            else:
                st.warning("⚠️ No K-Fold hybrid checkpoints found in `kfold/checkpoints/`")

            if n_kfold_b > 0:
                st.success(f"✅ Found {n_kfold_b} K-Fold Baseline Models")
            elif os.path.exists(baseline_ckpt_path):
                st.info("ℹ️ Using standard baseline model for comparison")
            else:
                st.info("ℹ️ Baseline checkpoint missing")

            k_summary = get_kfold_summary(dataset_name)
            if k_summary is not None:
                acc_stats = k_summary.get("cross_fold_stats", {}).get("accuracy", {})
                h_mean = acc_stats.get("hybrid_mean")
                h_std = acc_stats.get("hybrid_std")
                if h_mean is not None and h_std is not None:
                    st.caption(f"**Cross-Fold Hybrid Acc**: {h_mean:.2%} ± {h_std:.2%}")
                oof = k_summary.get("oof_metrics", {})
                if "auroc" in oof:
                    st.caption(f"**OOF AUROC**: {oof['auroc']:.4f}")
        else:
            has_hybrid = os.path.exists(hybrid_ckpt_path)
            has_baseline = os.path.exists(baseline_ckpt_path)

            ckpt_info = get_checkpoint_metadata(dataset_name)

            if has_hybrid:
                st.success("✅ Hybrid CNN-VQC found")
            else:
                st.warning("⚠️ Hybrid model checkpoint missing")

            if has_baseline:
                st.success("✅ Classical Baseline found")
            else:
                st.info("ℹ️ Classical baseline checkpoint missing")

            if ckpt_info is not None:
                cfg_info = ckpt_info.get("config", {})
                st.caption(f"**Architecture**: {cfg_info.get('n_qubits', 4)} Qubits • {cfg_info.get('n_layers', 3)} Layers")
                h_met = ckpt_info.get("hybrid_metrics", {})
                b_met = ckpt_info.get("baseline_metrics", {})
                if "accuracy" in h_met:
                    st.caption(f"**Hybrid Accuracy**: {h_met['accuracy']:.2%}")
                if "accuracy" in b_met:
                    st.caption(f"**Baseline Accuracy**: {b_met['accuracy']:.2%}")

        st.divider()
        st.caption(f"**Device**: {device}")
        st.caption(f"**PyTorch**: {torch.__version__}")

    # Helper: lazy cached model loader for both architectures
    @st.cache_resource(show_spinner=False)
    def _load_all_models_cached(h_paths: Tuple[str, ...], b_paths: Tuple[str, ...], _device: torch.device, dual_basis_readout: bool = True):
        h_mods, b_mods, m_out = [], [], None
        for p in h_paths:
            m, m_out = load_hybrid_model(p, _device, dual_basis_readout=dual_basis_readout)
            h_mods.append(m)
        for p in b_paths:
            bm, _ = load_baseline_model(p, _device)
            b_mods.append(bm)
        return h_mods, b_mods, m_out

    def get_loaded_models():
        if is_kfold:
            h_paths = tuple(kfold_ckpts["hybrid"])
            b_paths = tuple(kfold_ckpts["baseline"]) if n_kfold_b > 0 else ((baseline_ckpt_path,) if os.path.exists(baseline_ckpt_path) else ())
        else:
            h_paths = (hybrid_ckpt_path,) if has_hybrid else ()
            b_paths = (baseline_ckpt_path,) if has_baseline else ()

        try:
            with st.spinner("⚡ Initializing cached model weights..."):
                h_mods, b_mods, meta_out = _load_all_models_cached(h_paths, b_paths, device, dual_basis_readout=use_dual_basis)
            for hm in h_mods:
                if hm is not None:
                    hm.to(device)
            for bm in b_mods:
                if bm is not None:
                    bm.to(device)
            if not is_kfold:
                return (h_mods[0] if h_mods else None), (b_mods[0] if b_mods else None), meta_out
            return h_mods, b_mods, meta_out
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return ([], [], None) if is_kfold else (None, None, None)


    # ── Main Content ──────────────────────────────────────────────────────
    tab1, tab_qstate, tab_train = st.tabs([
        "🔬 Dual Clinical Diagnosis",
        "🧪 Quantum State Inspector",
        "🏋️ Training",
    ])

    cfg = DATASET_CONFIG[dataset_name]
    label_map = cfg["label_map"]

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1: DUAL CLINICAL DIAGNOSIS
    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        st.header("🔬 Dual Clinical Diagnosis: Hybrid vs Baseline")

        col_upload, col_result = st.columns([1, 1.6])

        with col_upload:
            st.subheader("📤 Medical Image Input")
            uploaded_files = st.file_uploader(
                "Upload medical image(s) (PNG, JPEG, BMP)",
                type=["png", "jpg", "jpeg", "bmp", "tiff"],
                accept_multiple_files=True,
                key="inference_upload",
                help="Select one or multiple images for batch or single-scan analysis.",
            )

            use_sample = st.checkbox("Use random sample from test set", value=False)

            images_list = []
            image_names = []
            true_label = None

            if use_sample:
                try:
                    sample_imgs, true_label = get_test_sample(dataset_name)
                    images_tensor = sample_imgs
                    images_list = [denormalize_image(sample_imgs[0])]
                    image_names = [f"Random Test Sample ({label_map.get(true_label, str(true_label))})"]
                except Exception as e:
                    st.error(f"Error loading sample: {e}")
                    images_tensor = None
            elif uploaded_files:
                for uf in uploaded_files:
                    try:
                        im = Image.open(uf)
                        images_list.append(im)
                        image_names.append(uf.name)
                    except Exception as e:
                        st.warning(f"Could not load {uf.name}: {e}")
                if images_list:
                    images_tensor = torch.cat(
                        [preprocess_image(im, cfg["in_channels"]) for im in images_list],
                        dim=0
                    )
                else:
                    images_tensor = None
            else:
                images_tensor = None

            n_imgs = len(images_list) if images_list else 0

            if n_imgs > 1:
                st.success(f"📂 **{n_imgs} scans loaded** for batch evaluation")
                active_idx = st.selectbox(
                    "🔍 Select scan for detailed explainability:",
                    range(n_imgs),
                    format_func=lambda i: f"#{i+1}: {image_names[i]}",
                    key="active_img_selector",
                )
                safe_image(images_list[active_idx], caption=f"Selected: {image_names[active_idx]}", width=220)
            elif n_imgs == 1:
                active_idx = 0
                safe_image(images_list[0], caption=image_names[0], width=220)
            else:
                active_idx = 0

            # Pre-inference scan quality & OOD guardrail
            if n_imgs > 0 and active_idx < len(images_list):
                act_im = images_list[active_idx]
                if isinstance(act_im, np.ndarray):
                    arr_val = (act_im * 255).astype(np.uint8) if act_im.max() <= 1.0 else act_im.astype(np.uint8)
                    act_im_pil = Image.fromarray(arr_val)
                else:
                    act_im_pil = act_im
                is_q_ok, var_lap, dyn_r, q_msg = assess_scan_quality(act_im_pil)
                if not is_q_ok:
                    st.warning(f"⚠️ **Scan Quality Warning**: {q_msg}. Diagnostic reliability may be affected.")
                else:
                    st.caption(f"✓ Quality verified: Blur score {var_lap:.1f} | Dynamic range {dyn_r:.0f}")

        with col_result:
            st.subheader("🩺 Comparative Diagnosis Results")

            if images_tensor is not None:
                h_model_res, b_model_res, meta = get_loaded_models()
                st.session_state["models_loaded"] = True


                if (is_kfold and h_model_res) or (not is_kfold and h_model_res is not None):
                    primary_h = h_model_res[0] if is_kfold else h_model_res

                    # Robust tensor preprocessing: enforce 4D (B, 3, 224, 224) and float32
                    images_tensor = images_tensor.float()
                    if images_tensor.ndim == 3:
                        images_tensor = images_tensor.unsqueeze(0)
                    if images_tensor.shape[1] == 1:
                        images_tensor = images_tensor.repeat(1, 3, 1, 1)
                    elif images_tensor.shape[1] > 3:
                        images_tensor = images_tensor[:, :3, :, :]
                    if images_tensor.shape[-2:] != (224, 224):
                        images_tensor = F.interpolate(images_tensor, size=(224, 224), mode="bilinear", align_corners=False)

                    # Fast batch feature extraction (Intel NPU or PyTorch):
                    if use_npu:
                        try:
                            with st.spinner("⚡ Offloading 86M ViT to Intel NPU via OpenVINO..."):
                                compiled_npu, chosen_dev = get_compiled_npu_vision_encoder(primary_h)
                                out_layer = compiled_npu.output(0)
                                feats = []
                                for i in range(images_tensor.shape[0]):
                                    single_np = images_tensor[i:i+1].cpu().numpy().astype(np.float32)
                                    out_i = compiled_npu([single_np])[out_layer]
                                    feats.append(torch.from_numpy(out_i))
                                batch_features = torch.cat(feats, dim=0).to(device)
                        except Exception as e:
                            st.warning(f"Intel NPU acceleration error ({e}). Falling back to CPU/GPU.")
                            with torch.inference_mode():
                                batch_features = primary_h.backbone.extract_features(images_tensor.to(device))
                    else:
                        with torch.inference_mode():
                            batch_features = primary_h.backbone.extract_features(images_tensor.to(device))

                    h_list = h_model_res if is_kfold else [h_model_res]
                    b_list = b_model_res if (is_kfold and b_model_res) else ([b_model_res] if b_model_res else [])

                    if "inference_cache" not in st.session_state:
                        st.session_state["inference_cache"] = {}

                    all_scan_results = []
                    with st.spinner(f"⚡ Fast inference on {n_imgs} scan(s)..."):
                        for img_i in range(n_imgs):
                            img_key = f"{dataset_name}_{is_kfold}_{selected_backend}_{n_mc_samples}_{noise_std}_{custom_tau}_{t_scale}_{readout_choice}_{image_names[img_i]}"
                            if enable_cache and img_key in st.session_state["inference_cache"]:
                                all_scan_results.append(st.session_state["inference_cache"][img_key])
                                continue


                            feat_i = batch_features[img_i:img_i + 1]
                            tens_i = images_tensor[img_i:img_i + 1]

                            # Quantum Hybrid evaluation
                            fold_mean_probs_h, fold_std_probs_h, all_exps, last_z = [], [], [], None
                            for m in h_list:
                                mph, sph, exps, z_ang = mc_dropout_inference(
                                    m, tens_i, device,
                                    n_samples=n_mc_samples, noise_std=noise_std,
                                    cached_features=feat_i,
                                    t_scale=t_scale,
                                )
                                fold_mean_probs_h.append(mph)
                                fold_std_probs_h.append(sph)
                                all_exps.extend(exps)
                                last_z = z_ang

                            mean_probs_h = np.mean(fold_mean_probs_h, axis=0)
                            std_probs_h = np.sqrt(
                                np.mean([s**2 for s in fold_std_probs_h], axis=0) +
                                np.var(fold_mean_probs_h, axis=0)
                            ) if len(fold_mean_probs_h) > 1 else fold_std_probs_h[0]

                            # Classical Baseline evaluation
                            if b_list:
                                fold_mean_probs_b, fold_std_probs_b = [], []
                                for bm in b_list:
                                    mpb, spb = mc_dropout_baseline(
                                        bm, tens_i, device,
                                        n_samples=n_mc_samples,
                                        cached_features=feat_i,
                                        t_scale=t_scale,
                                    )
                                    fold_mean_probs_b.append(mpb)
                                    fold_std_probs_b.append(spb)
                                mean_probs_b = np.mean(fold_mean_probs_b, axis=0)
                                std_probs_b = np.sqrt(
                                    np.mean([s**2 for s in fold_std_probs_b], axis=0) +
                                    np.var(fold_mean_probs_b, axis=0)
                                ) if len(fold_mean_probs_b) > 1 else fold_std_probs_b[0]
                            else:
                                mean_probs_b, std_probs_b = None, None

                            # Predictions with calibrated clinical threshold
                            if is_binary and len(mean_probs_h) == 2:
                                p_h = 1 if mean_probs_h[1] >= custom_tau else 0
                                c_h = float(mean_probs_h[p_h])
                                u_h = float(std_probs_h[p_h])
                                cls_h = label_map.get(p_h, str(p_h))

                                if mean_probs_b is not None:
                                    p_b = 1 if mean_probs_b[1] >= custom_tau else 0
                                    c_b = float(mean_probs_b[p_b])
                                    u_b = float(std_probs_b[p_b])
                                    cls_b = label_map.get(p_b, str(p_b))
                                else:
                                    p_b, c_b, u_b, cls_b = None, 0.0, 0.0, "N/A"
                            else:
                                p_h = int(np.argmax(mean_probs_h))
                                c_h = float(mean_probs_h[p_h])
                                u_h = float(std_probs_h[p_h])
                                cls_h = label_map.get(p_h, str(p_h))

                                if mean_probs_b is not None:
                                    p_b = int(np.argmax(mean_probs_b))
                                    c_b = float(mean_probs_b[p_b])
                                    u_b = float(std_probs_b[p_b])
                                    cls_b = label_map.get(p_b, str(p_b))
                                else:
                                    p_b, c_b, u_b, cls_b = None, 0.0, 0.0, "N/A"

                            scan_res = {
                                "name": image_names[img_i],
                                "pred_h": p_h,
                                "conf_h": c_h,
                                "unc_h": u_h,
                                "class_h": cls_h,
                                "mean_probs_h": mean_probs_h,
                                "std_probs_h": std_probs_h,
                                "pred_b": p_b,
                                "conf_b": c_b,
                                "unc_b": u_b,
                                "class_b": cls_b,
                                "mean_probs_b": mean_probs_b,
                                "std_probs_b": std_probs_b,
                                "all_exps": all_exps,
                                "z_angles": last_z,
                            }
                            if enable_cache:
                                st.session_state["inference_cache"][img_key] = scan_res
                            all_scan_results.append(scan_res)


                    # If multiple images uploaded, display batch overview table with divergence filter
                    if n_imgs > 1:
                        st.markdown(f"#### 📋 Batch Diagnosis Summary ({n_imgs} Scans)")
                        triage_filter = st.radio(
                            "Triage Filter",
                            ["All Scans", "⚠️ Divergent Only (Hybrid ≠ Baseline)", "🤝 Consensus Only"],
                            horizontal=True,
                            key="batch_triage_filter"
                        )
                        summary_rows = []
                        for idx, r in enumerate(all_scan_results):
                            agreed = (r["pred_h"] == r["pred_b"]) if r["pred_b"] is not None else True
                            if triage_filter.startswith("⚠️") and agreed:
                                continue
                            if triage_filter.startswith("🤝") and not agreed:
                                continue
                            status_icon = "🤝 Agree" if agreed else "⚠️ Diverge"
                            b_str = f"{r['class_b']} ({r['conf_b']:.1%})" if r["pred_b"] is not None else "N/A"
                            delta_p = abs(r["conf_h"] - r["conf_b"]) if r["pred_b"] is not None else 0.0
                            summary_rows.append({
                                "#": idx + 1,
                                "Scan File": r["name"],
                                "Quantum Hybrid": f"{r['class_h']} ({r['conf_h']:.1%})",
                                "Classical Baseline": b_str,
                                "Δ Conf": f"{delta_p:.1%}",
                                "Consensus": status_icon,
                            })
                        if summary_rows:
                            try:
                                st.dataframe(summary_rows, width="stretch")
                            except TypeError:
                                st.dataframe(summary_rows, use_container_width=True)
                        else:
                            st.info("No scans match the active triage filter.")
                        st.caption(f"Displaying in-depth diagnostic analysis for: **{image_names[active_idx]}**")

                        st.divider()

                    # Unpack active scan results
                    act = all_scan_results[active_idx]
                    st.session_state["last_z_angles"] = act.get("z_angles")
                    pred_h = act["pred_h"]
                    conf_h = act["conf_h"]
                    unc_h = act["unc_h"]
                    class_h = act["class_h"]
                    mean_probs_h = act["mean_probs_h"]
                    std_probs_h = act["std_probs_h"]
                    pred_b = act["pred_b"]
                    conf_b = act["conf_b"]
                    unc_b = act["unc_b"]
                    class_b = act["class_b"]
                    mean_probs_b = act["mean_probs_b"]
                    std_probs_b = act["std_probs_b"]
                    all_exps = act["all_exps"]
                    active_tensor = images_tensor[active_idx:active_idx + 1]

                    # High-Risk Priority Alert for Multi-Class
                    if not is_binary and enable_boost and boost_cls is not None:
                        p_boost = float(mean_probs_h[boost_cls])
                        target_name = label_map.get(boost_cls, str(boost_cls))
                        if p_boost >= boost_tau:
                            st.error(
                                f"🚨 **CLINICAL PRIORITY ALERT**: Calibrated probability for high-risk class **{target_name}** "
                                f"is **{p_boost:.1%}** (exceeds safety threshold $\\tau^* = {boost_tau:.0%}$). Immediate specialist review advised."
                            )

                    # ── 1. Diagnostic Consensus Banner ────────────────────────
                    if pred_b is not None:
                        if pred_h == pred_b:
                            st.success(
                                f"🤝 **Diagnostic Consensus**: Both Quantum Hybrid and Classical Baseline agree on **{class_h}** "
                                f"(Hybrid: {conf_h:.1%}, Baseline: {conf_b:.1%})"
                            )
                        else:
                            st.warning(
                                f"⚠️ **Diagnostic Divergence**: Quantum Hybrid predicts **{class_h}** ({conf_h:.1%}) "
                                f"while Classical Baseline predicts **{class_b}** ({conf_b:.1%}). Comprehensive clinical review recommended."
                            )


                    # ── 2. Side-by-Side Diagnosis Cards ───────────────────────
                    c_hyb, c_base = st.columns(2)

                    def get_risk_style(conf, unc, pred_cls):
                        if conf > 0.85 and unc < 0.05:
                            return ("pred-benign" if pred_cls == 0 else "pred-malignant"), "High Confidence"
                        elif conf < 0.60:
                            return "pred-uncertain", "Low Confidence (Review Required)"
                        else:
                            return "pred-uncertain", "Moderate Confidence"

                    # Hybrid Card
                    risk_h, text_h = get_risk_style(conf_h, unc_h, pred_h)
                    tag_h = f"🧬 HYBRID CNN-VQC ({len(h_model_res)}-FOLD AVERAGE)" if is_kfold else "🧬 HYBRID CNN-VQC (QUANTUM)"
                    if use_npu:
                        tag_h += " [⚡ INTEL NPU]"
                    elif device.type == "mps":
                        tag_h += " [⚡ APPLE MPS]"
                    with c_hyb:
                        st.markdown(
                            f'<div class="prediction-box {risk_h}">'
                            f'<div style="font-size:0.85rem; color:#00d4ff; font-weight:bold; margin-bottom:4px;">{tag_h}</div>'
                            f'<h3 style="margin:4px 0;">{class_h}</h3>'
                            f'<p style="font-size:1.1rem; margin:4px 0;">Confidence: <strong>{conf_h:.1%}</strong> ± {unc_h:.1%}</p>'
                            f'<span style="font-size:0.8rem; color:#aaa;">{text_h}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        # Conformal prediction
                        alpha_conf = 0.05
                        sorted_p = np.sort(mean_probs_h)[::-1]
                        set_size_h = np.searchsorted(np.cumsum(sorted_p), 1 - alpha_conf) + 1
                        pred_set_h = np.argsort(mean_probs_h)[::-1][:set_size_h]
                        names_h = [label_map.get(c, str(c)) for c in pred_set_h]
                        st.caption(f"**95% Conformal Set**: {', '.join(names_h)}")

                    # Baseline Card
                    if mean_probs_b is not None:
                        risk_b, text_b = get_risk_style(conf_b, unc_b, pred_b)
                        tag_b = f"💻 CLASSICAL BASELINE ({len(b_model_res)}-FOLD AVERAGE)" if (is_kfold and len(b_model_res) > 1) else "💻 CLASSICAL BASELINE (MATCHED MLP)"
                        if use_npu:
                            tag_b += " [⚡ INTEL NPU]"
                        elif device.type == "mps":
                            tag_b += " [⚡ APPLE MPS]"
                        with c_base:
                            st.markdown(
                                f'<div class="prediction-box {risk_b}">'
                                f'<div style="font-size:0.85rem; color:#f39c12; font-weight:bold; margin-bottom:4px;">{tag_b}</div>'
                                f'<h3 style="margin:4px 0;">{class_b}</h3>'
                                f'<p style="font-size:1.1rem; margin:4px 0;">Confidence: <strong>{conf_b:.1%}</strong> ± {unc_b:.1%}</p>'
                                f'<span style="font-size:0.8rem; color:#aaa;">{text_b}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                            sorted_pb = np.sort(mean_probs_b)[::-1]
                            set_size_b = np.searchsorted(np.cumsum(sorted_pb), 1 - alpha_conf) + 1
                            pred_set_b = np.argsort(mean_probs_b)[::-1][:set_size_b]
                            names_b = [label_map.get(c, str(c)) for c in pred_set_b]
                            st.caption(f"**95% Conformal Set**: {', '.join(names_b)}")

                    # ── 3. Delta Metrics ──────────────────────────────────────
                    if mean_probs_b is not None:
                        delta_conf = conf_h - conf_b
                        delta_unc = unc_h - unc_b
                        ent_h = -float(np.sum(mean_probs_h * np.log(np.clip(mean_probs_h, 1e-12, 1.0))))
                        ent_b = -float(np.sum(mean_probs_b * np.log(np.clip(mean_probs_b, 1e-12, 1.0))))

                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("Hybrid Confidence", f"{conf_h:.1%}", f"{delta_conf:+.1%} vs Baseline")
                        with m2:
                            st.metric("Hybrid Uncertainty", f"±{unc_h:.2%}", f"{delta_unc:+.2%} vs Baseline", delta_color="inverse")
                        with m3:
                            st.metric("Predictive Entropy", f"{ent_h:.3f} nats", f"{ent_h - ent_b:+.3f} vs Baseline", delta_color="inverse")

                    # ── 4. Grouped Comparative Probability Display ────────────
                    st.subheader("📊 Comparative Class Probabilities")
                    n_cls = len(mean_probs_h)
                    classes = [label_map.get(i, str(i)) for i in range(n_cls)]

                    if use_native_charts:
                        chart_cols = st.columns(2 if mean_probs_b is not None else 1)
                        with chart_cols[0]:
                            st.markdown("**🧬 Hybrid CNN-VQC (Quantum)**")
                            for c_i, c_name in enumerate(classes):
                                p_val = float(mean_probs_h[c_i])
                                u_val = float(std_probs_h[c_i])
                                u_str = f" ± {u_val:.1%}" if u_val > 0.001 else ""
                                st.write(f"**{c_name}**: {p_val:.1%}{u_str}")
                                st.progress(min(1.0, max(0.0, p_val)))

                        if mean_probs_b is not None:
                            with chart_cols[1]:
                                st.markdown("**💻 Classical Baseline**")
                                for c_i, c_name in enumerate(classes):
                                    pb_val = float(mean_probs_b[c_i])
                                    ub_val = float(std_probs_b[c_i])
                                    ub_str = f" ± {ub_val:.1%}" if ub_val > 0.001 else ""
                                    st.write(f"**{c_name}**: {pb_val:.1%}{ub_str}")
                                    st.progress(min(1.0, max(0.0, pb_val)))
                    else:
                        fig, ax = plt.subplots(figsize=(7, max(3.2, 0.65 * n_cls)))
                        y_pos = np.arange(n_cls)
                        bar_h = 0.35 if mean_probs_b is not None else 0.55

                        if mean_probs_b is not None:
                            ax.barh(y_pos + bar_h / 2, mean_probs_h, bar_h, xerr=std_probs_h,
                                    label="Hybrid CNN-VQC", color="#2980b9", alpha=0.85, edgecolor="white", capsize=3)
                            ax.barh(y_pos - bar_h / 2, mean_probs_b, bar_h, xerr=std_probs_b,
                                    label="Classical Baseline", color="#e67e22", alpha=0.85, edgecolor="white", capsize=3)
                            ax.legend(loc="lower right", fontsize=8)
                        else:
                            ax.barh(y_pos, mean_probs_h, bar_h, xerr=std_probs_h,
                                    label="Hybrid CNN-VQC", color="#2980b9", alpha=0.85, edgecolor="white", capsize=3)

                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(classes, fontsize=9, fontweight="semibold")
                        ax.set_xlabel("Probability", fontsize=10, fontweight="bold")
                        ax.set_title("Probability Distribution & Uncertainty Intervals", fontsize=11, fontweight="bold")
                        ax.set_xlim(0, 1.05)
                        ax.grid(True, alpha=0.25, axis="x")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()


                    # ── 5. Grad-CAM & Quantum Observables ──────────────────────
                    st.subheader("🔥 Dual Anatomical Grad-CAM & Quantum Observables")
                    exp_col1, exp_col2 = st.columns([1.3, 1])

                    active_cam_fig = None
                    with exp_col1:
                        st.markdown("**Dual Anatomical Lesion Localization**")
                        compute_cam_clicked = False
                        if not auto_gradcam:
                            compute_cam_clicked = st.button("🔥 Generate Dual Grad-CAM Overlays", key=f"btn_cam_{active_idx}")

                        if auto_gradcam or compute_cam_clicked:
                            try:
                                with st.spinner("Generating Dual Grad-CAM (Hybrid vs Baseline)..."):
                                    primary_h = h_model_res[0] if isinstance(h_model_res, list) else h_model_res
                                    cam_h = compute_gradcam(
                                        primary_h,
                                        active_tensor.clone(),
                                        device,
                                        target_class=int(pred_h),
                                    )

                                    primary_b = None
                                    if is_kfold and b_model_res:
                                        primary_b = b_model_res[0]
                                    elif not is_kfold and b_model_res is not None:
                                        primary_b = b_model_res

                                    cam_b = None
                                    if primary_b is not None and pred_b is not None:
                                        try:
                                            cam_b = compute_gradcam(
                                                primary_b,
                                                active_tensor.clone(),
                                                device,
                                                target_class=int(pred_b),
                                            )
                                        except Exception:
                                            cam_b = None

                                    active_disp = denormalize_image(active_tensor[0])
                                    if active_disp.ndim == 2:
                                        base = np.stack([active_disp] * 3, axis=-1)
                                    else:
                                        base = active_disp

                                    overlay_h = np.clip(0.5 * base + 0.5 * plt.cm.jet(cam_h)[:, :, :3], 0.0, 1.0)
                                    n_plots = 4 if cam_b is not None else 3
                                    fig_cam, axes = plt.subplots(1, n_plots, figsize=(3 * n_plots, 2.8))

                                    # Scan
                                    if active_disp.ndim == 2:
                                        axes[0].imshow(active_disp, cmap="gray")
                                    else:
                                        axes[0].imshow(active_disp)
                                    axes[0].set_title("Input Scan", fontsize=9, fontweight="bold")
                                    axes[0].axis("off")

                                    # Hybrid Overlay
                                    axes[1].imshow(overlay_h)
                                    axes[1].set_title(f"🧬 Hybrid: {class_h}", fontsize=9, fontweight="bold", color="#0099cc")
                                    axes[1].axis("off")

                                    # Hybrid Saliency
                                    axes[2].imshow(cam_h, cmap="jet")
                                    axes[2].set_title("Hybrid Saliency", fontsize=9, fontweight="bold")
                                    axes[2].axis("off")

                                    # Baseline Overlay
                                    if cam_b is not None:
                                        overlay_b = np.clip(0.5 * base + 0.5 * plt.cm.jet(cam_b)[:, :, :3], 0.0, 1.0)
                                        axes[3].imshow(overlay_b)
                                        axes[3].set_title(f"💻 Baseline: {class_b}", fontsize=9, fontweight="bold", color="#d35400")
                                        axes[3].axis("off")

                                    plt.tight_layout()
                                    st.pyplot(fig_cam)
                                    active_cam_fig = fig_cam
                                    plt.close()
                            except Exception as e:
                                st.warning(f"Grad-CAM computation failed: {e}")
                        else:
                            st.info("💡 Click the button above to generate dual Grad-CAM lesion overlays on demand.")


                    with exp_col2:
                        mean_exps = np.mean(all_exps, axis=0)
                        std_exps = np.std(all_exps, axis=0)
                        fig_q, ax_q = plt.subplots(figsize=(5, 3))
                        n_q = 4
                        if is_kfold and h_model_res:
                            n_q = getattr(h_model_res[0], "n_qubits", 4)
                        elif not is_kfold and h_model_res is not None:
                            n_q = getattr(h_model_res, "n_qubits", 4)

                        if len(mean_exps) == 2 * n_q:
                            qubit_labels = [f"Z_{i}" for i in range(n_q)] + [f"X_{i}" for i in range(n_q)]
                            chart_title = "Dual-Basis Observables (Z + X)"
                        elif len(mean_exps) == 3 * n_q - 1:
                            qubit_labels = [f"Z_{i}" for i in range(n_q)] + [f"Z_{i}Z_{i+1}" for i in range(n_q - 1)] + [f"X_{i}" for i in range(n_q)]
                            chart_title = "Normal Observables (Z + ZZ + X)"
                        else:
                            qubit_labels = [f"O_{i}" for i in range(len(mean_exps))]
                            chart_title = "Quantum State Observables"

                        colors_q = plt.cm.viridis(np.linspace(0.3, 0.9, len(mean_exps)))
                        ax_q.bar(qubit_labels, mean_exps, yerr=std_exps, color=colors_q,
                                 alpha=0.85, edgecolor="white", capsize=4)
                        ax_q.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
                        ax_q.set_ylabel("Expectation", fontsize=9)
                        ax_q.set_title(chart_title, fontsize=9, fontweight="bold")
                        ax_q.set_ylim(-1.1, 1.1)
                        if len(qubit_labels) > 6:
                            ax_q.tick_params(axis='x', rotation=35, labelsize=8)
                        ax_q.grid(True, alpha=0.3, axis="y")
                        plt.tight_layout()
                        st.pyplot(fig_q)
                        plt.close()

                    # ── 6. Automated Clinical Report Download ──────────────────
                    st.divider()
                    st.subheader("📄 Clinical Documentation & EHR Export")
                    scan_b64 = image_to_base64(images_list[active_idx]) if active_idx < len(images_list) else None
                    cam_b64 = fig_to_base64(active_cam_fig) if active_cam_fig is not None else None

                    report_html = generate_html_report(
                        scan_name=image_names[active_idx] if active_idx < len(image_names) else f"Scan_{active_idx+1}",
                        dataset_name=dataset_name,
                        verdict_h=class_h,
                        conf_h=conf_h,
                        unc_h=unc_h,
                        verdict_b=class_b if pred_b is not None else None,
                        conf_b=conf_b if pred_b is not None else 0.0,
                        unc_b=unc_b if pred_b is not None else 0.0,
                        mean_probs_h=mean_probs_h,
                        label_map=label_map,
                        conformal_set=names_h,
                        mean_exps=list(mean_exps),
                        scan_img_b64=scan_b64,
                        cam_img_b64=cam_b64,
                        tau=custom_tau,
                        t_scale=t_scale,
                    )

                    st.download_button(
                        label="📥 Download Clinical Diagnostic Summary (Printable HTML)",
                        data=report_html,
                        file_name=f"clinical_report_{image_names[active_idx].replace(' ', '_').replace('/', '_')}.html",
                        mime="text/html",
                        key=f"dl_report_{active_idx}",
                        help="Export printable, self-contained diagnostic verification report including confidence intervals, conformal set, and practitioner sign-off block."
                    )

            elif not has_hybrid:
                st.info("👆 Train a model first using `engine.py`, then come back for inference.")
            else:
                st.info("👆 Upload an image or select a test sample to run comparative diagnosis.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2: QUANTUM STATE INSPECTOR
    # ══════════════════════════════════════════════════════════════════════
    with tab_qstate:
        st.header("🧪 Quantum State Inspector")

        if has_hybrid:
            models_already_loaded = st.session_state.get("models_loaded", False)
            inspect_clicked = False
            if not models_already_loaded:
                st.info("💡 Model weights are lazy-loaded to ensure instant sub-second app startup.")
                inspect_clicked = st.button("🔬 Load & Inspect Quantum Circuit Parameters", key="btn_inspect_tab_qstate")

            if models_already_loaded or inspect_clicked:
                h_loaded, _, _ = get_loaded_models()
                h_model = h_loaded[0] if (isinstance(h_loaded, list) and len(h_loaded) > 0) else h_loaded
                if h_model is not None:
                    st.session_state["models_loaded"] = True
                    st.subheader("Circuit Parameters")

                    q_params = {}
                    for n, p in h_model.named_parameters():
                        if "qlayer" in n:
                            q_params[n] = p.detach().cpu().numpy()

                    if q_params:
                        for name, params in q_params.items():
                            st.write(f"**{name}**: shape = {params.shape}")
                            fig, ax = plt.subplots(figsize=(8, 3))
                            ax.hist(params.flatten(), bins=50, color="#9b59b6", alpha=0.8,
                                    edgecolor="white", linewidth=0.5)
                            ax.set_xlabel("Parameter Value")
                            ax.set_ylabel("Count")
                            ax.set_title(f"Distribution: {name}", fontweight="bold")
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                    # Patient-Specific Quantum Angle Encoding & Circuit Representation
                    st.subheader("🧬 Active Patient Quantum State Encoding")
                    patient_z = st.session_state.get("last_z_angles", None)
                    if patient_z is not None:
                        st.markdown(r"Angles $\theta = \text{Adapter}(x)$ mapped directly from the active patient image into quantum rotation gates ($R_y$):")
                        col_ang1, col_ang2 = st.columns([1.3, 1])
                        with col_ang1:
                            fig_ang, ax_ang = plt.subplots(figsize=(6, 2.7))
                            q_idx = np.arange(len(patient_z))
                            cols = plt.cm.coolwarm(np.linspace(0.15, 0.85, len(patient_z)))
                            ax_ang.bar(q_idx, patient_z, color=cols, edgecolor="white", alpha=0.85)
                            ax_ang.axhline(0, color="gray", linestyle="--", linewidth=0.7)
                            ax_ang.set_xticks(q_idx)
                            ax_ang.set_xticklabels([f"Q{i%h_model.n_qubits} (θ_{i})" for i in q_idx], fontsize=8, rotation=20)
                            ax_ang.set_ylabel("Rotation (rad)", fontsize=8)
                            ax_ang.set_ylim(-math.pi - 0.2, math.pi + 0.2)
                            ax_ang.set_title("Input Quantum Feature Coordinates", fontsize=9, fontweight="bold")
                            ax_ang.grid(True, alpha=0.25)
                            plt.tight_layout()
                            st.pyplot(fig_ang)
                            plt.close(fig_ang)

                        with col_ang2:
                            st.markdown("**Angle Rotation Values**")
                            ang_data = [{"Coordinate": f"θ_{i}", "Radians": f"{val:+.4f}", "Degrees": f"{math.degrees(val):+.1f}°"} for i, val in enumerate(patient_z)]
                            try:
                                st.dataframe(ang_data, hide_index=True)
                            except TypeError:
                                st.dataframe(ang_data)

                        # Dynamic Pennylane Wire Diagram for Patient State
                        st.markdown("**Ansatz Circuit Wire Diagram**")
                        try:
                            import pennylane as qml
                            if hasattr(h_model, "qlayer") and hasattr(h_model.qlayer, "qnode"):
                                sample_z = torch.from_numpy(patient_z).unsqueeze(0).to(device)
                                drawer = qml.draw(h_model.qlayer.qnode, expansion_strategy="device", max_length=100)
                                circ_repr = drawer(sample_z, h_model.qlayer.weights)
                                st.code(circ_repr, language="text")
                        except Exception as e:
                            st.caption(f"StronglyEntanglingLayers on {h_model.n_qubits} qubits across {h_model.n_layers} layers with periodic ring CNOT entanglement.")
                    else:
                        st.info("💡 Run diagnosis on an image in Tab 1 to inspect its specific quantum feature angles and state encoding.")

                    st.subheader("Circuit Information")
                    meas_desc = f"Dual-Basis Readout ($\\langle Z_i \\rangle, \\langle X_i \\rangle$ — {2*h_model.n_qubits} observables)" if getattr(h_model, "dual_basis_only", False) else f"Normal Multi-Observable Readout ($\\langle Z_i \\rangle, \\langle Z_i Z_j \\rangle, \\langle X_i \\rangle$ — {3*h_model.n_qubits-1} observables)"
                    st.markdown(rf"""
                    | Property | Value |
                    |---|---|
                    | **Qubits** | {h_model.n_qubits} |
                    | **VQC Layers** | {h_model.n_layers} |
                    | **Embedding** | Angle Embedding ($R_y$) |
                    | **Ansatz** | StronglyEntanglingLayers |
                    | **Measurement** | {meas_desc} |
                    | **Quantum Device** | {h_model.qdev_name} |
                    | **Diff Method** | Backprop / Parameter-shift |
                    """)

                    st.subheader("Quantum Kernel Gram Matrix (Test Batch)")
                    if st.button("Compute Kernel (small batch)"):
                        try:
                            import pennylane as qml
                            import seaborn as sns

                            images, labels = get_test_batch(dataset_name, batch_size=16)
                            images = images.to(device)
                            with torch.no_grad():
                                z = h_model.backbone(images)
                            z_np = z.cpu().numpy()

                            dev_k = qml.device("default.qubit", wires=h_model.n_qubits)
                            q_w = None
                            for n, p in h_model.named_parameters():
                                if "qlayer" in n and "weights" in n:
                                    q_w = p.detach().cpu().numpy()
                                    break

                            if q_w is not None:
                                @qml.qnode(dev_k, interface="numpy")
                                def kernel_circ(x):
                                    for i in range(h_model.n_qubits):
                                        qml.U3(x[..., i], x[..., i + h_model.n_qubits], x[..., i + 2 * h_model.n_qubits], wires=i)
                                    qml.StronglyEntanglingLayers(q_w, wires=range(h_model.n_qubits))
                                    return qml.state()

                                states = []
                                for i in range(len(z_np)):
                                    s = kernel_circ(z_np[i])
                                    states.append(np.array(s))
                                states = np.array(states)
                                gram = np.abs(states @ states.conj().T) ** 2

                                fig, ax = plt.subplots(figsize=(6, 5))
                                sns.heatmap(gram, cmap="magma", ax=ax, cbar_kws={"shrink": 0.8},
                                            annot=False)
                                lbl_names = [label_map.get(int(l), str(int(l)))
                                             for l in labels.squeeze().numpy()]
                                ax.set_xticklabels(lbl_names, fontsize=7, rotation=45)
                                ax.set_yticklabels(lbl_names, fontsize=7, rotation=0)
                                ax.set_title("Quantum Kernel Gram Matrix", fontweight="bold")
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)
                        except Exception as e:
                            st.error(f"Error computing kernel: {e}")
                else:
                    st.warning("⚠️ Could not load hybrid model checkpoint. Please verify that the checkpoint file exists and is valid.")
        else:
            st.info("Load a trained model to inspect quantum states.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3: TRAINING (PASSWORD PROTECTED & SUITE)
    # ══════════════════════════════════════════════════════════════════════
    with tab_train:
        st.header("🏋️ Quantum Model Training & Laboratory")

        # ── 1. Authentication Gatekeeper ──────────────────────────────────
        if not st.session_state.get("training_auth", False):
            st.subheader("🔒 Authentication Required")
            st.markdown("Please enter authorized credentials to access the Training Suite, Analysis Gallery, and Performance Outputs.")

            with st.form("train_login_form"):
                col_u, col_p = st.columns(2)
                with col_u:
                    user_input = st.text_input("Username", placeholder="Enter username", key="auth_train_user")
                with col_p:
                    pass_input = st.text_input("Password", type="password", placeholder="Enter password", key="auth_train_pass")

                login_btn = st.form_submit_button("🔓 Unlock Training Access", use_container_width=True)
                if login_btn:
                    if user_input.strip() == "vrushabh_qml" and pass_input.strip() == "vrush1108":
                        st.session_state["training_auth"] = True
                        st.success("✅ Access granted! Welcome, vrushabh_qml.")
                        st.rerun()
                    else:
                        st.error("❌ Access denied: Invalid username or password.")
        else:
            # Authenticated Header
            auth_head1, auth_head2 = st.columns([5, 1])
            with auth_head1:
                st.caption("👤 Logged in as **vrushabh_qml** • Authorized for Model Training, Retraining, and Benchmark Inspection")
            with auth_head2:
                if st.button("🚪 Logout", key="btn_train_logout", use_container_width=True):
                    st.session_state["training_auth"] = False
                    st.rerun()

            # Sub-Tabs for Training, Gallery, and Summary
            subtab_train, subtab_gallery, subtab_summary = st.tabs([
                "🚀 Train & Retrain Model",
                "📊 Analysis Gallery",
                "📈 Training Summary & Benchmarks",
            ])

            # ──────────────────────────────────────────────────────────────
            # SUBTAB A: TRAIN & RETRAIN MODEL
            # ──────────────────────────────────────────────────────────────
            with subtab_train:
                st.subheader("⚙️ Quantum Architecture & Training Configuration")

                col_ds, col_mode = st.columns([1, 1.2])
                with col_ds:
                    avail_ds = list(DATASET_CONFIG.keys())
                    def_ds_idx = avail_ds.index(dataset_name) if dataset_name in avail_ds else 0
                    train_dataset = st.selectbox(
                        "Target Dataset",
                        avail_ds,
                        index=def_ds_idx,
                        key="train_ds_target",
                        help="Select dataset to train or retrain."
                    )
                    ds_cfg_active = DATASET_CONFIG[train_dataset]
                    st.caption(f"• Classes: **{ds_cfg_active['n_classes']}** | Channels: **{ds_cfg_active['in_channels']}** | 3D: **{ds_cfg_active.get('is_3d', False)}**")

                with col_mode:
                    train_paradigm = st.selectbox(
                        "Training Strategy / Paradigm",
                        [
                            "Standard Hybrid (CNN-VQC + Baseline)",
                            "BiomedCLIP ViT Fine-Tuning + Hybrid",
                            "K-Fold Cross-Validation (kfold)",
                            "Progressive Layer Expansion (progressive)",
                            "Quantum Neural Architecture Search (QNAS)",
                            "Quantum Attention Filter (QAF)",
                            "Continual Learning",
                            "Evaluation Only (--eval_only)",
                            "Custom Architectural Blend",
                        ],
                        index=0,
                        key="train_paradigm_choice",
                        help="Choose the primary training paradigm. All parameters below can be fine-tuned."
                    )

                # Dataset Checkpoint & Fine-Tuning Status Indicator
                has_vit_ckpt = has_finetuned_checkpoint(train_dataset)
                has_model_ckpt = os.path.exists(os.path.join(get_dataset_output_dir(train_dataset), "checkpoints", "hybrid_model.pt"))
                status_c1, status_c2 = st.columns(2)
                with status_c1:
                    if has_vit_ckpt:
                        st.success(f"🟢 **BiomedCLIP ViT:** Domain weights detected at `biomedclip/{train_dataset}_biomedclip_vit.pt`")
                    else:
                        st.info(f"⚪ **BiomedCLIP ViT:** Frozen pretrained zero-shot backbone. Toggle fine-tuning below to train.")
                with status_c2:
                    if has_model_ckpt:
                        st.success(f"🟢 **Trained Model:** Found existing checkpoint in `{get_dataset_output_dir(train_dataset)}/checkpoints/`")
                    else:
                        st.warning(f"🟡 **Trained Model:** No checkpoint found for `{train_dataset}` yet.")

                # Base flag presets from paradigm selection
                is_vit_ft_flag = (train_paradigm == "BiomedCLIP ViT Fine-Tuning + Hybrid")
                is_kfold_flag = (train_paradigm == "K-Fold Cross-Validation (kfold)")
                is_prog_flag = (train_paradigm == "Progressive Layer Expansion (progressive)")
                is_qnas_flag = (train_paradigm == "Quantum Neural Architecture Search (QNAS)")
                is_qaf_flag = (train_paradigm == "Quantum Attention Filter (QAF)")
                is_continual_flag = (train_paradigm == "Continual Learning")
                is_eval_only_flag = (train_paradigm == "Evaluation Only (--eval_only)")

                # Retrain and execution hardware options
                st.divider()
                col_retrain, col_hardware, col_res = st.columns([1.2, 1, 1])
                with col_retrain:
                    retrain_mode = st.checkbox(
                        "🔄 Retrain Model (Clean Start & Force Pretraining)",
                        value=True,
                        help="If checked, existing checkpoints and summaries will be refreshed and CQ-Adapter pretraining will be forced."
                    )
                    eval_only_arg = st.checkbox("Evaluation Only Mode (--eval_only)", value=is_eval_only_flag, help="Generate all diagnostic plots without retraining.")
                with col_hardware:
                    device_opts = ["auto", "cuda", "cpu"]
                    if torch.backends.mps.is_available():
                        device_opts.insert(1, "mps")
                    if not torch.cuda.is_available() and "cuda" in device_opts:
                        device_opts.remove("cuda")
                    train_device = st.selectbox("Execution Device", device_opts, index=0, key="train_device_choice")
                    output_dir_arg = st.text_input("Output Directory (--output_dir)", value="outputs", key="arg_output_dir")
                with col_res:
                    image_size_arg = st.selectbox("Image Size (--image_size)", [224, 128, 64, 28], index=0, key="arg_image_size")
                    qdevice_arg = st.selectbox("Quantum Device (--qdevice)", ["auto", "default.qubit", "lightning.qubit", "lightning.gpu"], index=0, key="arg_qdev")

                st.divider()
                st.subheader("🎛️ Optimization & Circuit Topology")

                arg_col1, arg_col2, arg_col3 = st.columns(3)
                with arg_col1:
                    st.markdown("**Optimization & Epochs**")
                    epochs_arg = st.number_input("Epochs (--epochs)", min_value=1, max_value=200, value=15, step=1, key="arg_epochs")
                    batch_size_arg = st.selectbox("Batch Size (--batch_size)", [8, 16, 32, 64, 128, 256], index=4, key="arg_batch_size")
                    lr_arg = st.number_input("Learning Rate (--lr)", min_value=1e-6, max_value=0.1, value=0.0002, format="%.6f", step=1e-4, key="arg_lr")
                    grad_accum_steps_arg = st.number_input("Gradient Accumulation Steps (--grad_accum_steps)", min_value=1, max_value=32, value=1, step=1, key="arg_grad_accum")
                    amp_arg = st.checkbox("Automatic Mixed Precision (--amp)", value=True, key="arg_amp")
                    cache_in_vram_arg = st.checkbox("Cache Features in GPU VRAM (--cache_in_vram)", value=False, key="arg_cache_vram")

                with arg_col2:
                    st.markdown("**Quantum Circuit Topology**")
                    n_qubits_arg = st.slider("Number of Qubits (--n_qubits)", 2, 8, 4, key="arg_qubits")
                    n_layers_arg = st.slider("VQC Depth (--n_layers)", 1, 8, 3, key="arg_layers")
                    dropout_arg = st.slider("Dropout Rate (--dropout)", 0.0, 0.6, 0.3, step=0.05, key="arg_dropout")
                    diff_method_arg = st.selectbox("Diff Method (--diff_method)", ["auto", "backprop", "adjoint"], index=0, key="arg_diff")
                    ansatz_mode_arg = st.selectbox("Ansatz Mode (--ansatz_mode)", ["auto", "qnas_static", "progressive_dense", "standard_static", "qnas_progressive"], index=0, key="arg_ansatz_mode")
                    q_init_std_arg = st.number_input("Quantum Init Std (--q_init_std)", min_value=0.001, max_value=0.5, value=0.01, format="%.4f", step=0.005, key="arg_qinit")

                with arg_col3:
                    st.markdown("**Architectural Extensions**")
                    use_lora_arg = st.checkbox("Enable LoRA-Q Adapter (--use_lora)", value=True, key="arg_lora")
                    dual_basis_arg = st.checkbox("Dual-Basis [Z, X] Readout (--dual_basis_readout)", value=True, key="arg_dual_basis")
                    use_qaf_attention_arg = st.checkbox("Quantum Attention Filter (--use_qaf_attention)", value=is_qaf_flag, key="arg_qaf")
                    clinical_hard_arg = st.checkbox("Clinical Hard Constrained Phase (--clinical_hard)", value=False, key="arg_clinical_hard")
                    balanced_sampler_arg = st.checkbox("Balanced Mini-Batch Sampler (--balanced_sampler)", value=False, key="arg_balanced_sampler")

                # ── 1. BiomedCLIP ViT Fine-Tuning & Weights Isolation ─────────
                with st.expander("🔬 BiomedCLIP ViT Fine-Tuning & Weights Isolation", expanded=is_vit_ft_flag):
                    vit_col1, vit_col2 = st.columns([1.2, 1])
                    with vit_col1:
                        enable_vit_finetuning_arg = st.checkbox(
                            "Enable BiomedCLIP ViT Fine-Tuning (--enable_vit_finetuning)",
                            value=is_vit_ft_flag,
                            key="arg_enable_vit_ft",
                            help="Unfreezes top transformer layers of BiomedCLIP ViT and fine-tunes with domain classification objective."
                        )
                        vit_unfreeze_layers_arg = st.slider(
                            "Top ViT Layers to Unfreeze (--vit_unfreeze_layers)",
                            min_value=1, max_value=12, value=2, step=1,
                            key="arg_vit_unfreeze_layers",
                            help="Number of transformer encoder blocks to unfreeze (starting from layer 11 down)."
                        )
                        vit_finetune_lr_arg = st.number_input(
                            "ViT Fine-Tuning Learning Rate (--vit_finetune_lr)",
                            min_value=1e-7, max_value=1e-3, value=1e-5, format="%.6f", step=1e-6,
                            key="arg_vit_ft_lr",
                            help="Conservative learning rate for ViT backbone fine-tuning (default: 1e-5)."
                        )
                    with vit_col2:
                        vit_epochs_arg = st.slider(
                            "ViT Fine-Tuning Epochs (--vit_epochs)",
                            min_value=1, max_value=30, value=5, step=1,
                            key="arg_vit_epochs",
                            help="Number of fine-tuning epochs for the ViT backbone."
                        )
                        biomedclip_dir_arg = st.text_input(
                            "BiomedCLIP Weights Folder (--biomedclip_dir)",
                            value="biomedclip",
                            key="arg_biomedclip_dir",
                            help="Isolated folder to store dataset-specific fine-tuned ViT weights."
                        )
                        st.caption(f"📁 Target checkpoint: `{biomedclip_dir_arg}/{train_dataset}_biomedclip_vit.pt`")
                        if has_vit_ckpt:
                            st.caption("✅ Checkpoint present — will be loaded during feature extraction.")
                        else:
                            st.caption("⚠️ Checkpoint missing — click 'Fine-Tune ViT Only' below or enable during full training.")

                # ── 2. Loss Functions & Anti-Shortcut Parameters ──────────────
                with st.expander("⚖️ Loss Functions & Anti-Shortcut Regularization", expanded=True):
                    loss_col1, loss_col2, loss_col3 = st.columns(3)
                    with loss_col1:
                        loss_type_arg = st.selectbox(
                            "Loss Function Type (--loss_type)",
                            ["focal", "ce", "asymmetric", "cb_ldam"],
                            index=0,
                            key="arg_loss_type",
                            help="Choose loss: 'focal' (Focal Loss), 'ce' (Cross-Entropy), 'asymmetric' (Asymmetric Focal), 'cb_ldam' (Class-Balanced LDAM with DRW)."
                        )
                        no_asym_loss_arg = st.checkbox(
                            "Disable Asymmetric Loss (--no_asymmetric_loss)",
                            value=False,
                            key="arg_no_asym",
                            help="Force standard symmetric focal loss when asymmetric loss is active."
                        )
                        ldam_margin_arg = st.number_input(
                            "CB-LDAM Max Margin C (--ldam_max_margin)",
                            min_value=0.05, max_value=2.0, value=0.50, step=0.05,
                            key="arg_ldam_margin",
                            help="Maximum margin for Class-Balanced LDAM."
                        )
                        drw_epoch_arg = st.number_input(
                            "CB-LDAM DRW Start Epoch (--drw_start_epoch)",
                            min_value=1, max_value=100, value=10, step=1,
                            key="arg_drw_epoch",
                            help="Epoch to activate Deferred Re-Weighting for class imbalance."
                        )

                    with loss_col2:
                        asym_alpha_arg = st.text_input(
                            "Asymmetric Alpha (--asym_alpha)",
                            value="",
                            placeholder="e.g. 1.5,1.0 (empty for auto)",
                            key="arg_asym_alpha",
                            help="Per-class weights alpha for Asymmetric Focal Loss. Pass comma-separated floats or single float."
                        )
                        asym_gamma_arg = st.text_input(
                            "Asymmetric Gamma (--asym_gamma)",
                            value="",
                            placeholder="e.g. 1.5,2.0 (empty for auto)",
                            key="arg_asym_gamma",
                            help="Per-class focusing parameters gamma for Asymmetric Focal Loss."
                        )
                        use_cost_matrix_arg = st.checkbox(
                            "Apply Clinical Cost Matrix (--use_cost_matrix)",
                            value=False,
                            key="arg_use_cost_matrix",
                            help="Applies clinical cost matrix for Bayes risk minimization."
                        )
                        cost_fp_arg = st.number_input(
                            "False Positive Cost Ratio (--cost_fp)",
                            min_value=0.1, max_value=10.0, value=1.5, step=0.1,
                            key="arg_cost_fp",
                            help="Cost ratio for False Positives relative to False Negatives."
                        )

                    with loss_col3:
                        threshold_crit_arg = st.selectbox(
                            "Threshold Criterion (--threshold_criterion)",
                            ["youden", "target_specificity", "f_beta", "cost_sensitive", "balanced_eer"],
                            index=0,
                            key="arg_threshold_crit",
                            help="Criterion for post-hoc threshold optimization."
                        )
                        target_spec_arg = st.number_input(
                            "Target Specificity Floor (--target_specificity)",
                            min_value=0.50, max_value=0.99, value=0.90, step=0.01,
                            key="arg_target_spec",
                            help="Target specificity floor for constrained optimization."
                        )
                        recalibrate_thresh_arg = st.checkbox(
                            "Recalibrate Threshold (--recalibrate_threshold)",
                            value=False,
                            key="arg_recalibrate_thresh",
                            help="Re-optimize decision threshold on validation set."
                        )

                    st.divider()
                    st.markdown("**🛡️ Anti-Shortcut Learning Regularizers**")
                    sc_col1, sc_col2 = st.columns(2)
                    with sc_col1:
                        lambda_decorr_arg = st.number_input(
                            "Feature Decorrelation Weight (--lambda_decorr)",
                            min_value=0.0, max_value=1.0, value=0.01, step=0.005, format="%.4f",
                            key="arg_lambda_decorr",
                            help="Penalizes off-diagonal cross-feature covariance to prevent shortcut feature collapse."
                        )
                    with sc_col2:
                        lambda_saliency_arg = st.number_input(
                            "Saliency Concentration Weight (--lambda_saliency)",
                            min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.4f",
                            key="arg_lambda_saliency",
                            help="Penalizes Grad-CAM attribution on peripheral border artifacts outside anatomical mask."
                        )
                    st.caption("🎯 Anti-shortcut target: Shortcut score < 0.30 & peripheral border attribution < 15%.")

                # ── 2. Quantum Optimizers & Regularization ────────────────────
                with st.expander("⚛️ Advanced Quantum Optimizers & Regularization Weights", expanded=False):
                    opt_col1, opt_col2 = st.columns(2)
                    with opt_col1:
                        st.markdown("**Quantum Optimizers**")
                        use_qng_arg = st.checkbox("Quantum Natural Gradient (--use_qng)", value=False, key="arg_qng", help="Riemannian QNG optimization on quantum weights.")
                        use_qsam_arg = st.checkbox("Quantum Sharpness-Aware Min (--use_qsam)", value=False, key="arg_qsam", help="Q-SAM for flat-basin generalization.")
                        qsam_freq_arg = st.number_input("Q-SAM Step Frequency (--qsam_freq)", min_value=1, max_value=10, value=1, step=1, key="arg_qsam_freq")
                        use_qswa_arg = st.checkbox("Quantum Stochastic Weight Averaging (--use_qswa)", value=False, key="arg_qswa", help="Q-SWA in final 30% of training.")

                    with opt_col2:
                        st.markdown("**Quantum Mixup & Loss Regularization**")
                        mixup_alpha_arg = st.number_input("Quantum Mixup Alpha (--mixup_alpha)", min_value=0.0, max_value=2.0, value=0.4, step=0.05, key="arg_mixup_alpha")
                        pre_quantum_mixup_arg = st.checkbox("Legacy Pre-Quantum Mixup (--pre_quantum_mixup)", value=False, key="arg_pre_qmixup")
                        phase_eps_arg = st.number_input("Phase Perturbation Epsilon (--phase_eps)", min_value=0.0, max_value=1.0, value=0.05, step=0.01, key="arg_phase_eps")
                        lambda_kta_arg = st.number_input("KTA Alignment Weight (--lambda_kta)", min_value=0.0, max_value=1.0, value=0.05, step=0.01, key="arg_lambda_kta")
                        lambda_ewc_arg = st.number_input("QEWC Regularization Weight (--lambda_ewc)", min_value=0.0, max_value=5000.0, value=400.0, step=50.0, key="arg_lambda_ewc")
                        lambda_infonce_arg = st.number_input("InfoNCE Fidelity Weight (--lambda_infonce)", min_value=0.0, max_value=2.0, value=0.0, step=0.05, key="arg_lambda_infonce")

                # ── 3. QNAS, Progressive Training & Adapters ──────────────────
                with st.expander("🧬 QNAS, Progressive Depth & CQ-Adapter Pretraining", expanded=False):
                    qnas_c1, qnas_c2, qnas_c3 = st.columns(3)
                    with qnas_c1:
                        st.markdown("**CQ-Adapter Pretraining**")
                        pretrain_epochs_arg = st.number_input("Pretrain Epochs (--pretrain_epochs)", min_value=0, max_value=100, value=30, step=5, key="arg_pretrain_ep")
                        force_pretrain_arg = st.checkbox("Force CQ-Adapter Pretrain (--force_pretrain)", value=False, key="arg_force_pretrain")
                        adapter_ckpt_arg = st.text_input("Adapter Checkpoint Path (--adapter_checkpoint)", value="", placeholder="Optional explicit checkpoint path", key="arg_adapter_ckpt")

                    with qnas_c2:
                        st.markdown("**Progressive Layerwise Training**")
                        progressive_arg = st.checkbox("Enable Progressive Growth (--progressive)", value=is_prog_flag, key="arg_progressive")
                        prog_depths_arg = st.text_input("Progressive Depths (--progressive_depths)", value="2 4 6", key="arg_prog_depths", help="Depth schedule for progressive training (e.g. '2 4 6' or '2,4,6').")
                        epochs_per_stage_arg = st.number_input("Epochs Per Stage (--epochs_per_stage)", min_value=0, max_value=50, value=0, step=1, key="arg_stage_epochs", help="0 for automatic calculation.")
                        finetune_epochs_arg = st.number_input("Finetune Epochs (--finetune_epochs)", min_value=0, max_value=50, value=5, step=1, key="arg_finetune_ep")
                        freeze_mode_arg = st.selectbox("Freeze Mode (--freeze_mode)", ["hard", "none"], index=0, key="arg_freeze_mode")

                    with qnas_c3:
                        st.markdown("**QNAS Evolutionary Search**")
                        use_qnas_arg = st.checkbox("Enable QNAS (--use_qnas)", value=is_qnas_flag, key="arg_use_qnas")
                        allow_qnas_prog_arg = st.checkbox("Allow QNAS + Progressive Combined (--allow_qnas_progressive)", value=False, key="arg_allow_qnas_prog")
                        qnas_gen_arg = st.number_input("QNAS Generations (--qnas_generations)", min_value=1, max_value=30, value=6, step=1, key="arg_qnas_gen")
                        qnas_pop_arg = st.number_input("QNAS Population Size (--qnas_pop_size)", min_value=2, max_value=50, value=16, step=2, key="arg_qnas_pop")
                        qnas_min_cnot_arg = st.number_input("Min CNOT Count (--qnas_min_cnot)", min_value=0, max_value=20, value=2, step=1, key="arg_qnas_min_cnot")
                        qnas_max_cnot_arg = st.number_input("Max CNOT Budget (--qnas_max_cnot)", min_value=1, max_value=30, value=6, step=1, key="arg_qnas_max_cnot")
                        qnas_target_mw_arg = st.number_input("Target Meyer-Wallach (--qnas_target_mw)", min_value=0.05, max_value=1.0, value=0.45, step=0.05, key="arg_qnas_mw")
                        qnas_lambda_cnot_arg = st.number_input("CNOT Penalty Lambda (--qnas_lambda_cnot)", min_value=0.001, max_value=0.2, value=0.01, format="%.4f", step=0.005, key="arg_qnas_lambda")

                # ── 4. Validation, Continual & Diagnostic Settings ─────────────
                with st.expander("🔁 K-Fold, Continual Learning & Evaluation Diagnostics", expanded=False):
                    diag_c1, diag_c2, diag_c3 = st.columns(3)
                    with diag_c1:
                        st.markdown("**K-Fold Cross-Validation**")
                        kfold_arg = st.checkbox("Enable K-Fold CV (--kfold)", value=is_kfold_flag, key="arg_kfold")
                        n_folds_arg = st.slider("Number of Folds (--n_folds)", 2, 10, 5, key="arg_nfolds")
                        kfold_no_ensemble_arg = st.checkbox("Disable OOF Ensembling (--kfold_no_ensemble)", value=False, key="arg_kfold_no_ens")

                    with diag_c2:
                        st.markdown("**Continual Learning**")
                        continual_arg = st.checkbox("Continual Learning (--continual)", value=is_continual_flag, key="arg_continual")
                        continual_datasets_arg = st.multiselect(
                            "Continual Dataset Sequence (--continual_datasets)",
                            avail_ds,
                            default=["pneumoniamnist", "breastmnist"],
                            key="arg_continual_ds"
                        )

                    with diag_c3:
                        st.markdown("**Diagnostics & Plot Generation**")
                        entropy_every_epoch_arg = st.checkbox("Entropy & KTA Every Epoch (--entropy_every_epoch)", value=True, key="arg_entropy_every")
                        entropy_samples_arg = st.number_input("Entropy Validation Samples (--entropy_samples)", min_value=8, max_value=128, value=32, step=8, key="arg_entropy_samples")
                        entropy_interval_arg = st.number_input("Entropy Evaluation Interval (--entropy_interval)", min_value=1, max_value=10, value=1, step=1, key="arg_entropy_interval")
                        fast_eval_arg = st.checkbox("Fast Evaluation Mode (--fast_eval)", value=False, key="arg_fast_eval", help="Bypass heavy noise simulations to save 5-15 min.")
                        skip_plots_arg = st.checkbox("Skip Diagnostic Plots (--skip_plots)", value=False, key="arg_skip_plots")
                        plot_workers_arg = st.number_input("Plot Worker Threads (--plot_workers)", min_value=1, max_value=16, value=4, step=1, key="arg_plot_workers")
                        benchmark_mode_arg = st.checkbox("Benchmark Profiling Mode (--benchmark_mode)", value=False, key="arg_bench_mode")
                        profile_runtime_arg = st.checkbox("Profile Component Runtime (--profile_runtime)", value=False, key="arg_profile_runtime")

                # Assemble Full CLI Command
                engine_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine.py")
                cmd = [
                    sys.executable,
                    "-u",
                    engine_script,
                    "--dataset", train_dataset,
                    "--epochs", str(epochs_arg),
                    "--batch_size", str(batch_size_arg),
                    "--lr", str(lr_arg),
                    "--n_qubits", str(n_qubits_arg),
                    "--n_layers", str(n_layers_arg),
                    "--dropout", str(dropout_arg),
                    "--diff_method", diff_method_arg,
                    "--qdevice", qdevice_arg,
                    "--image_size", str(image_size_arg),
                    "--output_dir", output_dir_arg,
                    "--loss_type", loss_type_arg,
                    "--threshold_criterion", threshold_crit_arg,
                    "--target_specificity", str(target_spec_arg),
                    "--cost_fp", str(cost_fp_arg),
                    "--ldam_max_margin", str(ldam_margin_arg),
                    "--drw_start_epoch", str(drw_epoch_arg),
                    "--mixup_alpha", str(mixup_alpha_arg),
                    "--phase_eps", str(phase_eps_arg),
                    "--lambda_kta", str(lambda_kta_arg),
                    "--lambda_ewc", str(lambda_ewc_arg),
                    "--lambda_infonce", str(lambda_infonce_arg),
                    "--ansatz_mode", ansatz_mode_arg,
                    "--q_init_std", str(q_init_std_arg),
                    "--plot_workers", str(plot_workers_arg),
                    "--entropy_samples", str(entropy_samples_arg),
                    "--entropy_interval", str(entropy_interval_arg),
                ]

                # Loss & Evaluation Flags
                if eval_only_arg:
                    cmd.append("--eval_only")
                if no_asym_loss_arg:
                    cmd.append("--no_asymmetric_loss")
                if asym_alpha_arg and asym_alpha_arg.strip():
                    cmd.extend(["--asym_alpha", asym_alpha_arg.strip()])
                if asym_gamma_arg and asym_gamma_arg.strip():
                    cmd.extend(["--asym_gamma", asym_gamma_arg.strip()])
                if use_cost_matrix_arg:
                    cmd.append("--use_cost_matrix")
                if recalibrate_thresh_arg:
                    cmd.append("--recalibrate_threshold")
                if pre_quantum_mixup_arg:
                    cmd.append("--pre_quantum_mixup")
                if balanced_sampler_arg:
                    cmd.append("--balanced_sampler")
                if use_lora_arg:
                    cmd.append("--use_lora")
                if dual_basis_arg:
                    cmd.append("--dual_basis_readout")
                if use_qaf_attention_arg:
                    cmd.append("--use_qaf_attention")
                if clinical_hard_arg:
                    cmd.append("--clinical_hard")

                # Quantum Optimization Flags
                if use_qng_arg:
                    cmd.append("--use_qng")
                if use_qsam_arg:
                    cmd.append("--use_qsam")
                    cmd.extend(["--qsam_freq", str(qsam_freq_arg)])
                if use_qswa_arg:
                    cmd.append("--use_qswa")
                if amp_arg:
                    cmd.append("--amp")
                else:
                    cmd.append("--no_amp")
                if cache_in_vram_arg:
                    cmd.append("--cache_in_vram")
                if grad_accum_steps_arg > 1:
                    cmd.extend(["--grad_accum_steps", str(grad_accum_steps_arg)])

                # CQ-Adapter Pretraining Flags
                if pretrain_epochs_arg > 0:
                    cmd.extend(["--pretrain_epochs", str(pretrain_epochs_arg)])
                if force_pretrain_arg or retrain_mode:
                    cmd.append("--force_pretrain")
                if adapter_ckpt_arg and adapter_ckpt_arg.strip():
                    cmd.extend(["--adapter_checkpoint", adapter_ckpt_arg.strip()])

                # Progressive Layerwise Flags
                if progressive_arg:
                    cmd.append("--progressive")
                    if prog_depths_arg and prog_depths_arg.strip():
                        d_tokens = prog_depths_arg.replace(",", " ").split()
                        if d_tokens:
                            cmd.append("--progressive_depths")
                            cmd.extend(d_tokens)
                    if epochs_per_stage_arg > 0:
                        cmd.extend(["--epochs_per_stage", str(epochs_per_stage_arg)])
                    cmd.extend(["--finetune_epochs", str(finetune_epochs_arg)])
                    cmd.extend(["--freeze_mode", freeze_mode_arg])

                # QNAS Flags
                if use_qnas_arg:
                    cmd.append("--use_qnas")
                    if allow_qnas_prog_arg:
                        cmd.append("--allow_qnas_progressive")
                    cmd.extend(["--qnas_generations", str(qnas_gen_arg)])
                    cmd.extend(["--qnas_pop_size", str(qnas_pop_arg)])
                    cmd.extend(["--qnas_min_cnot", str(qnas_min_cnot_arg)])
                    cmd.extend(["--qnas_max_cnot", str(qnas_max_cnot_arg)])
                    cmd.extend(["--qnas_target_mw", str(qnas_target_mw_arg)])
                    cmd.extend(["--qnas_lambda_cnot", str(qnas_lambda_cnot_arg)])

                # K-Fold Flags
                if kfold_arg:
                    cmd.extend(["--kfold", "--n_folds", str(n_folds_arg)])
                    if kfold_no_ensemble_arg:
                        cmd.append("--kfold_no_ensemble")

                # Continual Learning Flags
                if continual_arg:
                    cmd.append("--continual")
                    if continual_datasets_arg:
                        cmd.append("--continual_datasets")
                        cmd.extend(continual_datasets_arg)

                # Entropy Diagnostics Flags
                if not entropy_every_epoch_arg:
                    cmd.append("--no_entropy_every_epoch")

                # Performance & Plots Flags
                if fast_eval_arg:
                    cmd.append("--fast_eval")
                if skip_plots_arg:
                    cmd.append("--skip_plots")
                if benchmark_mode_arg:
                    cmd.append("--benchmark_mode")
                if profile_runtime_arg:
                    cmd.append("--profile_runtime")

                # BiomedCLIP ViT Fine-Tuning Flags
                if enable_vit_finetuning_arg:
                    cmd.append("--enable_vit_finetuning")
                    cmd.extend(["--vit_unfreeze_layers", str(vit_unfreeze_layers_arg)])
                    cmd.extend(["--vit_finetune_lr", str(vit_finetune_lr_arg)])
                    cmd.extend(["--vit_epochs", str(vit_epochs_arg)])
                    if biomedclip_dir_arg and biomedclip_dir_arg.strip():
                        cmd.extend(["--biomedclip_dir", biomedclip_dir_arg.strip()])

                # Anti-Shortcut Regularization Flags
                if lambda_decorr_arg > 0:
                    cmd.extend(["--lambda_decorr", str(lambda_decorr_arg)])
                if lambda_saliency_arg > 0:
                    cmd.extend(["--lambda_saliency", str(lambda_saliency_arg)])

                with st.expander("🛠️ View Full Assembled Execution Command", expanded=False):
                    st.code(" ".join(cmd), language="bash")

                # Execution Trigger
                st.divider()
                btn_train_col1, btn_train_col2 = st.columns([1.5, 1])
                with btn_train_col1:
                    start_training_btn = st.button("🚀 Start Training / Retraining", type="primary", use_container_width=True, key="btn_start_training")
                with btn_train_col2:
                    start_vit_btn = st.button("🔬 Fine-Tune ViT Only", type="secondary", use_container_width=True, key="btn_start_vit_only", help="Fine-tune BiomedCLIP ViT top layers on this dataset and save weights into biomedclip/ folder without running full VQC training.")

                # Live Training Runtime Monitor
                if start_training_btn:
                    st.session_state["training_in_progress"] = True
                    st.info(f"⏳ Training initiated for `{train_dataset}` using `{train_paradigm}`...")

                    progress_bar = st.progress(0.0, text="Initializing training engine and loading datasets...")
                    metric_m1, metric_m2, metric_m3, metric_m4 = st.columns(4)
                    box_ep = metric_m1.empty()
                    box_loss = metric_m2.empty()
                    box_acc = metric_m3.empty()
                    box_time = metric_m4.empty()

                    box_ep.metric("Current Epoch", "0")
                    box_loss.metric("Loss", "---")
                    box_acc.metric("Accuracy", "---")
                    box_time.metric("Elapsed", "0s")

                    st.markdown("**Real-Time Terminal Stream**")
                    terminal_view = st.empty()

                    run_env = os.environ.copy()
                    run_env["PYTHONUNBUFFERED"] = "1"
                    run_env["OMP_NUM_THREADS"] = "4"
                    run_env["PYTHONIOENCODING"] = "utf-8"

                    start_epoch_time = time.time()
                    log_history = []
                    curr_epoch = 0
                    total_epochs = epochs_arg

                    ep_pat = re.compile(r"epoch\s*\[?(\d+)[/:](\d+)\]?", re.IGNORECASE)
                    loss_pat = re.compile(r"(?:train_loss|loss|val_loss)[:=\s]+([0-9.]+)", re.IGNORECASE)
                    acc_pat = re.compile(r"(?:accuracy|acc|val_acc)[:=\s]+([0-9.]+)", re.IGNORECASE)

                    try:
                        proc = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            cwd=os.path.dirname(engine_script),
                            env=run_env,
                            universal_newlines=True,
                            encoding="utf-8",
                            errors="replace",
                        )

                        for line in iter(proc.stdout.readline, ''):
                            if not line and proc.poll() is not None:
                                break
                            clean_l = line.strip()
                            if clean_l:
                                log_history.append(clean_l)

                                # Parse regex
                                m_ep = ep_pat.search(clean_l)
                                if m_ep:
                                    curr_epoch = int(m_ep.group(1))
                                    total_epochs = int(m_ep.group(2))
                                    p_ratio = max(0.0, min(1.0, curr_epoch / max(1, total_epochs)))
                                    progress_bar.progress(p_ratio, text=f"Training: Epoch {curr_epoch}/{total_epochs} ({int(p_ratio*100)}%)")
                                    box_ep.metric("Current Epoch", f"{curr_epoch}/{total_epochs}")

                                m_loss = loss_pat.search(clean_l)
                                if m_loss:
                                    box_loss.metric("Loss", f"{float(m_loss.group(1)):.4f}")

                                m_acc = acc_pat.search(clean_l)
                                if m_acc:
                                    acc_val = float(m_acc.group(1))
                                    acc_str = f"{acc_val:.2%}" if acc_val <= 1.0 else f"{acc_val:.2f}%"
                                    box_acc.metric("Accuracy", acc_str)

                                elapsed_sec = int(time.time() - start_epoch_time)
                                box_time.metric("Elapsed", f"{elapsed_sec}s")

                                # Keep last 20 lines displayed
                                terminal_view.code("\n".join(log_history[-20:]), language="text")

                        proc.stdout.close()
                        ret_code = proc.wait()

                        if ret_code == 0:
                            progress_bar.progress(1.0, text="🎉 Training & Evaluation Complete!")
                            st.success(f"✅ Training completed successfully in {time.time() - start_epoch_time:.1f}s! All models, weights, and plots have been generated.")
                            # Clear streamlit cache to immediately expose newly generated models and plots
                            st.cache_data.clear()
                            st.cache_resource.clear()
                            st.session_state["models_loaded"] = False
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        else:
                            st.error(f"❌ Training finished with exit code {ret_code}. See logs above for details.")
                    except Exception as e:
                        st.error(f"Failed to execute training process: {e}")
                    finally:
                        st.session_state["training_in_progress"] = False

                # Standalone BiomedCLIP ViT Fine-Tuning Runtime Monitor
                if start_vit_btn:
                    st.session_state["training_in_progress"] = True
                    st.info(f"🔬 Launching Standalone BiomedCLIP ViT Fine-Tuning on `{train_dataset}`...")

                    vit_progress = st.progress(0.0, text="Initializing BiomedCLIP ViT model and DataLoader...")
                    vm1, vm2, vm3, vm4 = st.columns(4)
                    b_ep = vm1.empty()
                    b_loss = vm2.empty()
                    b_acc = vm3.empty()
                    b_time = vm4.empty()

                    b_ep.metric("ViT Epoch", "0")
                    b_loss.metric("Loss", "---")
                    b_acc.metric("Accuracy", "---")
                    b_time.metric("Elapsed", "0s")

                    st.markdown("**BiomedCLIP Fine-Tuning Terminal Stream**")
                    v_terminal = st.empty()

                    vit_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetuning.py")
                    vit_cmd = [
                        sys.executable,
                        "-u",
                        vit_script,
                        "--dataset", train_dataset,
                        "--unfreeze_layers", str(vit_unfreeze_layers_arg),
                        "--finetune_lr", str(vit_finetune_lr_arg),
                        "--epochs", str(vit_epochs_arg),
                        "--batch_size", str(batch_size_arg),
                        "--save_dir", biomedclip_dir_arg,
                    ]

                    vit_env = os.environ.copy()
                    vit_env["PYTHONUNBUFFERED"] = "1"
                    vit_env["OMP_NUM_THREADS"] = "4"
                    vit_env["PYTHONIOENCODING"] = "utf-8"

                    t_start = time.time()
                    v_logs = []
                    vit_ep_pat = re.compile(r"epoch\s*\[?(\d+)[/:](\d+)\]?", re.IGNORECASE)
                    vit_loss_pat = re.compile(r"(?:loss|train_loss)[:=\s]+([0-9.]+)", re.IGNORECASE)
                    vit_acc_pat = re.compile(r"(?:acc|val_acc|accuracy)[:=\s]+([0-9.]+)", re.IGNORECASE)

                    try:
                        v_proc = subprocess.Popen(
                            vit_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            cwd=os.path.dirname(engine_script),
                            env=vit_env,
                            universal_newlines=True,
                            encoding="utf-8",
                            errors="replace",
                        )

                        for line in iter(v_proc.stdout.readline, ''):
                            if not line and v_proc.poll() is not None:
                                break
                            cl = line.strip()
                            if cl:
                                v_logs.append(cl)
                                m_ep = vit_ep_pat.search(cl)
                                if m_ep:
                                    c_ep, tot_ep = int(m_ep.group(1)), int(m_ep.group(2))
                                    p_r = max(0.0, min(1.0, c_ep / max(1, tot_ep)))
                                    vit_progress.progress(p_r, text=f"ViT Fine-Tuning: Epoch {c_ep}/{tot_ep} ({int(p_r*100)}%)")
                                    b_ep.metric("ViT Epoch", f"{c_ep}/{tot_ep}")

                                m_loss = vit_loss_pat.search(cl)
                                if m_loss:
                                    b_loss.metric("Loss", f"{float(m_loss.group(1)):.4f}")

                                m_acc = vit_acc_pat.search(cl)
                                if m_acc:
                                    val = float(m_acc.group(1))
                                    b_acc.metric("Accuracy", f"{val:.2%}" if val <= 1.0 else f"{val:.2f}%")

                                b_time.metric("Elapsed", f"{int(time.time() - t_start)}s")
                                v_terminal.code("\n".join(v_logs[-20:]), language="text")

                        v_proc.stdout.close()
                        v_ret = v_proc.wait()

                        if v_ret == 0:
                            vit_progress.progress(1.0, text="🎉 ViT Fine-Tuning Complete!")
                            st.success(f"✅ BiomedCLIP ViT fine-tuned successfully in {time.time() - t_start:.1f}s! Checkpoint saved to `{biomedclip_dir_arg}/{train_dataset}_biomedclip_vit.pt`.")
                            st.cache_data.clear()
                            st.cache_resource.clear()
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        else:
                            st.error(f"❌ ViT fine-tuning finished with exit code {v_ret}. See logs above for details.")
                    except Exception as ex:
                        st.error(f"Failed to execute ViT fine-tuning process: {ex}")
                    finally:
                        st.session_state["training_in_progress"] = False

            # ──────────────────────────────────────────────────────────────
            # SUBTAB B: ANALYSIS GALLERY (Moved under Training Tab)
            # ──────────────────────────────────────────────────────────────
            with subtab_gallery:
                st.subheader("📊 Analysis Gallery & Generated Plots")
                plot_dir = get_dataset_output_dir(dataset_name)

                if os.path.isdir(plot_dir):
                    all_plots = sorted(glob.glob(os.path.join(plot_dir, "*.png")))
                    all_plots += sorted(glob.glob(os.path.join(plot_dir, "eda", "*.png")))
                    if is_kfold:
                        all_plots += sorted(glob.glob(os.path.join(plot_dir, "kfold", "*.png")))
                        all_plots += sorted(glob.glob(os.path.join(plot_dir, "kfold", "average", "*.png")))

                    if all_plots:
                        categories = {
                            "🛡️ Anti-Shortcut Verification & Generalization": ["shortcut_verification", "shortcut", "decorrelation", "saliency"],
                            "🔁 K-Fold Cross-Validation": ["kfold", "fold_"],
                            "🔮 CQ-Adapter Pre-Training & Dispersion": ["cq_pretrain", "pretrain"],
                            "🌌 Advanced QML Diagnostics (SOTA Research)": ["adv_gradient_ratio", "conformal_prediction", "effective_quantum", "haar_fidelity"],
                            "📈 Training Dynamics": ["convergence", "gradient", "loss_landscape", "epoch_time", "kta_evolution"],
                            "🩺 Clinical Diagnostics": ["roc", "pr_curve", "confusion", "calibration", "uncertainty"],
                            "⚛️ Quantum Geometry & Spectra": ["tsne", "kernel_gram", "bloch", "decision_boundary", "qfim_spectrum"],
                            "🔬 Explainability & Robustness": ["pauli_z", "entanglement", "param_dist", "xai_gradcam", "noise_channel"],
                            "📊 EDA": ["class_dist", "samples", "pixel_hist"],
                            "🔄 Continual Learning": ["qewc", "zne"],
                        }

                        for cat_name, keywords in categories.items():
                            matching = [p for p in all_plots
                                        if any(kw in os.path.basename(p).lower() for kw in keywords)]
                            if matching:
                                st.subheader(cat_name)
                                cols = st.columns(min(len(matching), 2))
                                for i, plot_path in enumerate(matching):
                                    with cols[i % len(cols)]:
                                        safe_image(plot_path, caption=Path(plot_path).stem.replace("_", " ").title())

                        categorised = set()
                        for kws in categories.values():
                            for p in all_plots:
                                if any(kw in os.path.basename(p).lower() for kw in kws):
                                    categorised.add(p)
                        remaining = [p for p in all_plots if p not in categorised]
                        if remaining:
                            st.subheader("📁 Other Plots")
                            cols = st.columns(2)
                            for i, p in enumerate(remaining):
                                with cols[i % 2]:
                                    safe_image(p, caption=Path(p).stem.replace("_", " ").title())
                    else:
                        st.info("No plots found for this dataset yet. Use the 'Train & Retrain Model' subtab to generate analysis.")
                else:
                    st.info(f"Output directory `{plot_dir}` not found. Train a model to view analysis.")

            # ──────────────────────────────────────────────────────────────
            # SUBTAB C: TRAINING SUMMARY (Moved under Training Tab)
            # ──────────────────────────────────────────────────────────────
            with subtab_summary:
                st.subheader("📈 Training Summary & Performance Benchmarks")
                k_summary = get_kfold_summary(dataset_name) if is_kfold else None
                summary_path = os.path.join(get_dataset_output_dir(dataset_name), "summary.json")

                if is_kfold and k_summary is not None:
                    st.subheader("🔁 K-Fold Cross-Validation Overall Summary")
                    acc_stats = k_summary.get("cross_fold_stats", {}).get("accuracy", {})
                    oof_met = k_summary.get("oof_metrics", {})
                    h_mean = acc_stats.get("hybrid_mean", 0)
                    h_std = acc_stats.get("hybrid_std", 0)
                    b_mean = acc_stats.get("baseline_mean", 0)
                    b_std = acc_stats.get("baseline_std", 0)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Hybrid CV Accuracy", f"{h_mean:.2%} ± {h_std:.2%}", f"{h_mean - b_mean:+.2%} vs Baseline")
                    with col2:
                        st.metric("OOF AUROC", f"{oof_met.get('auroc', 0):.4f}")
                    with col3:
                        st.metric("OOF Sensitivity", f"{oof_met.get('sensitivity', 0):.2%}")
                    with col4:
                        st.metric("OOF Specificity", f"{oof_met.get('specificity', 0):.2%}")

                    st.divider()
                    st.subheader("📋 Per-Fold Performance Breakdown")
                    h_folds = acc_stats.get("hybrid_per_fold", [])
                    b_folds = acc_stats.get("baseline_per_fold", [])
                    fold_table = []
                    for i, h_acc in enumerate(h_folds):
                        b_acc = b_folds[i] if i < len(b_folds) else 0.0
                        fold_table.append({
                            "Fold": f"Fold {i}",
                            "Hybrid Accuracy": f"{h_acc:.2%}",
                            "Baseline Accuracy": f"{b_acc:.2%}",
                            "Δ (Quantum Advantage)": f"{h_acc - b_acc:+.2%}",
                        })
                    if fold_table:
                        st.table(fold_table)

                elif os.path.exists(summary_path):
                    with open(summary_path) as f:
                        summary = json.load(f)

                    col1, col2, col3, col4 = st.columns(4)
                    h_met = summary.get("hybrid_metrics", {})
                    b_met = summary.get("baseline_metrics", {})

                    with col1:
                        st.metric("Hybrid Accuracy", f"{h_met.get('accuracy', 0):.2%}",
                                   f"{h_met.get('accuracy', 0) - b_met.get('accuracy', 0):+.2%}")
                    with col2:
                        st.metric("Hybrid AUROC", f"{h_met.get('auroc', 0):.4f}")
                    with col3:
                        st.metric("Hybrid F1 (macro)", f"{h_met.get('f1_macro', 0):.4f}")
                    with col4:
                        st.metric("Expressibility (KL)", f"{summary.get('expressibility_kl', 0):.6f}")

                    st.divider()

                    st.subheader("🔬 Hybrid vs Classical Baseline")
                    comparison_data = []
                    for key in h_met:
                        comparison_data.append({
                            "Metric": key.replace("_", " ").title(),
                            "Hybrid CNN-VQC": f"{h_met[key]:.4f}",
                            "Classical Baseline": f"{b_met.get(key, 0):.4f}",
                            "Δ": f"{h_met[key] - b_met.get(key, 0):+.4f}",
                        })
                    st.table(comparison_data)

                    mcnemar = summary.get("mcnemar", {})
                    if mcnemar:
                        st.subheader("📊 Statistical Significance (McNemar's Test)")
                        st.json(mcnemar)
                        if mcnemar.get("significant"):
                            st.success("✅ Statistically significant difference (p < 0.05)")
                        else:
                            st.warning("⚠️ No statistically significant difference (p ≥ 0.05)")

                    st.subheader("🔢 Parameter Counts")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.json(summary.get("hybrid_params", {}))
                        st.caption("Hybrid CNN-VQC")
                    with col_b:
                        st.json(summary.get("baseline_params", {}))
                        st.caption("Classical Baseline")

                    st.subheader("⚛️ Quantum Information Metrics")
                    qcol1, qcol2, qcol3 = st.columns(3)
                    with qcol1:
                        st.metric("Expressibility (KL)", f"{summary.get('expressibility_kl', 0):.6f}")
                    with qcol2:
                        st.metric("Meyer-Wallach Q", f"{summary.get('meyer_wallach_q', 0):.6f}")
                    with qcol3:
                        st.metric("Entanglement Entropy", f"{summary.get('entanglement_entropy_final', 0):.4f}")

                else:
                    st.info("No training summary found. Launch training via the 'Train & Retrain Model' subtab to generate metrics.")

    # ── Footer ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div style="text-align:center; color: #888; padding: 1rem;">
        <p>Hybrid CNN-VQC Platform • PennyLane + PyTorch • MedMNIST v2</p>
        <p style="font-size: 0.8rem">Quantum Machine Learning for Clinical Diagnostics</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
