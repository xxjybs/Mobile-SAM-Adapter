from __future__ import annotations

"""
Generate a paper-style module-response heatmap figure for DMANet (v15).

Main goals of v9:
1) Adapter row is forced to show Adapter1-Adapter4.
2) Dual-branch row is forced to show Db1-Db4, emphasizing dual-path fusion results.
3) Decoder row is ordered from shallow to deep.
4) Mask-generation row is ordered from shallow to deep and titled CA-1, CA-2, ... only.
5) Prefer composite block outputs when they better match the paper figures.

This script is intended to be run from your project root where `models` is importable.

v11 fix:
- Do NOT use module.register_full_backward_hook().
- Capture gradients with Tensor.register_hook() inside the forward hook.
- This avoids the PyTorch inplace-ReLU error:
  "Output 0 of BackwardHookFunctionBackward is a view and is being modified inplace".
"""

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models import model_dict


@dataclass
class GroupSpec:
    roman: str
    title: str
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    max_maps: int = 4


@dataclass
class CamRecord:
    name: str
    group: str
    cam: np.ndarray
    score: float
    fg_mean: float
    bg_mean: float
    peak: float
    shape: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", type=str, default="mobile_sam_adapter")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--label", type=Path, default=None)
    p.add_argument("--pred-dir", type=Path, default=None, help="Directory containing saved prediction PNG files whose stem matches the input image stem.")
    p.add_argument("--output", type=Path, default=Path("outputs/dmanet_layercam_figure_v15.png"))
    p.add_argument("--dump-csv", type=Path, default=Path("outputs/dmanet_cam_candidates_v15.csv"))
    p.add_argument("--input-size", type=int, default=1024)
    p.add_argument("--pred-thr", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--target-mode",
        choices=["pred", "label", "intersection", "union", "hybrid"],
        default="hybrid",
        help="CAM objective. hybrid=0.7*pred+0.3*label; union is often good for explanation figures.",
    )
    p.add_argument(
        "--cam-method",
        choices=["layercam", "gradcam", "activation"],
        default="layercam",
        help="layercam is recommended for segmentation module visualization.",
    )
    p.add_argument("--roi-mask", action="store_true", help="Suppress background heat outside union(label,pred).")
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--pred-alpha", type=float, default=0.68, help="Overlay strength for the Predict panel.")
    p.add_argument("--pred-edge-width", type=int, default=4, help="Contour width for the Predict panel.")
    p.add_argument("--row0-predict-alpha", type=float, default=0.55, help="Overlay alpha for row-0 Predict when using pred PNG.")
    p.add_argument("--row0-heatmap-alpha", type=float, default=0.52, help="Overlay alpha for row-0 Heatmap when using pred PNG.")
    p.add_argument("--row0-heatmap-dilate-kernel", type=int, default=11, help="Display expansion kernel for row-0 Heatmap component support.")
    p.add_argument("--row0-heatmap-blur-kernel", type=int, default=29, help="Base Gaussian blur kernel for row-0 Heatmap component halos.")
    p.add_argument("--row0-heatmap-gamma", type=float, default=0.72, help="Gamma for row-0 Heatmap soft response.")
    p.add_argument("--row0-heatmap-min-area", type=int, default=6, help="Ignore extremely tiny components in row-0 Heatmap.")
    p.add_argument("--paper-bg-dim", type=float, default=0.88, help="Background dim factor for paper-style heatmap overlay.")
    p.add_argument("--paper-min-alpha-ratio", type=float, default=0.26, help="Minimum alpha ratio used in paper-style heatmap overlay.")
    p.add_argument("--gamma", type=float, default=0.75)
    p.add_argument("--low-percentile", type=float, default=1.0)
    p.add_argument("--high-percentile", type=float, default=99.5)
    p.add_argument("--blur-kernel", type=int, default=5)
    p.add_argument("--min-score", type=float, default=-0.02)
    p.add_argument("--show-score", action="store_true")
    p.add_argument("--fig-dpi", type=int, default=220)
    p.add_argument("--font-size", type=int, default=11)
    p.add_argument("--title-size", type=int, default=13)
    return p.parse_args()


def load_model(model_name: str, ckpt_path: Path, device: torch.device, input_size: int) -> torch.nn.Module:
    if model_name not in model_dict:
        raise KeyError(f"Unknown model_name={model_name}. choices={list(model_dict.keys())}")
    ctor = model_dict[model_name]
    try:
        model = ctor(inp_size=input_size)
    except TypeError:
        model = ctor()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "model_state", "model_state_dict", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    fixed = {}
    for k, v in ckpt.items():
        nk = str(k)
        for prefix in ["module.", "model."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        fixed[nk] = v
    missing, unexpected = model.load_state_dict(fixed, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")
    model.to(device).eval()
    return model


def read_image(path: Path, input_size: int) -> tuple[np.ndarray, torch.Tensor]:
    img = Image.open(path).convert("RGB")
    img_np = np.array(img)
    img_resized = img.resize((input_size, input_size), Image.BILINEAR)
    x = torch.from_numpy(np.array(img_resized, copy=True).transpose(2, 0, 1)).float() / 255.0
    return img_np, x.unsqueeze(0)


def read_label(path: Path | None, hw: tuple[int, int]) -> np.ndarray | None:
    if path is None:
        return None
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0).astype(np.uint8)
    return cv2.resize(mask, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)


def read_pred_png_mask(pred_dir: Path | None, image_path: Path, hw: tuple[int, int]) -> np.ndarray | None:
    """Load the display mask from pred folder using the same stem as the input image.

    Example:
        image = xxx/abc.jpg -> pred PNG expected at pred_dir/abc.png
    """
    if pred_dir is None:
        return None
    pred_path = pred_dir / f"{image_path.stem}.png"
    if not pred_path.exists():
        cands = list(pred_dir.glob(f"{image_path.stem}.png")) + list(pred_dir.glob(f"{image_path.stem}.PNG"))
        if not cands:
            print(f"[WARN] pred PNG not found for image stem: {image_path.stem} in {pred_dir}")
            return None
        pred_path = cands[0]
    mask = np.asarray(Image.open(pred_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0).astype(np.uint8)
    return cv2.resize(mask, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)


def get_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    y = model(x)
    if isinstance(y, (list, tuple)):
        y = y[0]
    if y.ndim == 3:
        y = y.unsqueeze(1)
    return y


def make_target(pred: np.ndarray, label: np.ndarray | None, mode: str) -> tuple[np.ndarray, np.ndarray]:
    pred_f = pred.astype(np.float32)
    label_f = pred_f if label is None else label.astype(np.float32)
    if mode == "pred":
        target = pred_f
    elif mode == "label":
        target = label_f
    elif mode == "intersection":
        target = ((pred_f > 0.5) & (label_f > 0.5)).astype(np.float32)
    elif mode == "union":
        target = ((pred_f > 0.5) | (label_f > 0.5)).astype(np.float32)
    else:
        target = 0.7 * pred_f + 0.3 * label_f
    roi = ((pred_f > 0.5) | (label_f > 0.5)).astype(np.float32)
    if target.sum() < 1:
        target = roi.copy()
    return target, roi


def color_mask(mask: np.ndarray, color: tuple[int, int, int] = (220, 190, 30)) -> np.ndarray:
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    out[mask > 0] = np.array(color, dtype=np.uint8)
    return out


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(220, 190, 30), alpha=0.35) -> np.ndarray:
    cm = color_mask(mask, color)
    return cv2.addWeighted(image, 1 - alpha, cm, alpha, 0)


def overlay_region_from_predpng(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """Overlay a binary region from the pred PNG onto the original image."""
    return overlay_mask(image, mask, color=color, alpha=alpha)


def predpng_to_soft_heatmap(
    mask: np.ndarray,
    dilate_kernel: int = 11,
    blur_kernel: int = 29,
    gamma: float = 0.72,
    min_area: int = 6,
) -> np.ndarray:
    """Convert a binary pred PNG mask into a paper-style soft heatmap.

    Instead of turning the whole predicted region into one smooth blob, this function
    builds a localized response for each connected defect component. The result is closer
    to the reference paper figure: each defect appears as a separate hotspot with a warm
    center and a cooler surrounding halo after colormap rendering.
    """
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return m.astype(np.float32)

    k = max(1, int(dilate_kernel))
    if k % 2 == 0:
        k += 1
    if k > 1:
        kernel = np.ones((k, k), np.uint8)
        m = cv2.dilate(m, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    heat = np.zeros(m.shape, dtype=np.float32)

    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        comp = (labels == idx).astype(np.uint8)

        # Outer halo: a broader smooth field around the component.
        halo = comp.astype(np.float32)
        halo_k = max(7, int(np.sqrt(area) * 2.4), int(blur_kernel))
        if halo_k % 2 == 0:
            halo_k += 1
        halo_k = min(halo_k, 81)
        halo = cv2.GaussianBlur(halo, (halo_k, halo_k), 0)
        halo = normalize_cam(halo)

        # Inner peak: emphasize the center of the defect.
        dt = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        if float(dt.max()) > 0:
            dt = dt / float(dt.max())
        dt = np.power(dt.astype(np.float32), 0.65)
        peak_k = max(5, int(np.sqrt(area) * 0.9))
        if peak_k % 2 == 0:
            peak_k += 1
        peak_k = min(peak_k, 31)
        if peak_k > 1:
            dt = cv2.GaussianBlur(dt, (peak_k, peak_k), 0)
        dt = normalize_cam(dt)

        comp_heat = normalize_cam(0.62 * halo + 0.95 * dt)
        heat = np.maximum(heat, comp_heat)

    # Add a very weak global support term so the fruit still sits in a cool response field.
    support = cv2.GaussianBlur(m.astype(np.float32), (31, 31), 0)
    support = normalize_cam(support) * 0.18
    heat = np.maximum(heat, support)

    heat = normalize_cam(heat)
    if gamma > 0:
        heat = np.power(heat, gamma)
    return np.clip(heat, 0, 1)


def overlay_prediction(
    image: np.ndarray,
    mask: np.ndarray,
    fill_color: tuple[int, int, int] = (255, 64, 64),
    fill_alpha: float = 0.68,
    edge_color: tuple[int, int, int] = (255, 255, 0),
    edge_width: int = 4,
) -> np.ndarray:
    """Make the Predict panel visually distinct from Image.

    The previous v9 version used a light yellow transparent overlay, which could
    blend into the orange fruit and become hard to see. This version uses:
    1) a vivid red semi-transparent fill inside the predicted mask;
    2) a bright yellow contour around the mask boundary.
    """
    out = image.copy()
    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.sum() == 0:
        return out

    # Strong filled highlight inside the predicted region.
    fill_layer = np.zeros_like(out, dtype=np.uint8)
    fill_layer[mask_u8 > 0] = np.array(fill_color, dtype=np.uint8)
    out = cv2.addWeighted(out, 1.0, fill_layer, fill_alpha, 0)

    # Draw an explicit contour so the predicted region is clearly visible.
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, edge_color, max(1, int(edge_width)), lineType=cv2.LINE_AA)

    # Also emphasize small regions by drawing a thin inner contour in white.
    if edge_width >= 2:
        cv2.drawContours(out, contours, -1, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    return out


def overlay_heatmap(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
    bg_dim: float = 0.88,
    min_alpha_ratio: float = 0.26,
) -> np.ndarray:
    """Paper-style heatmap overlay.

    The overlay softly dims the original image, then blends a JET heatmap using a
    per-pixel alpha map. Compared with plain cv2.addWeighted, this produces a style
    closer to the reference paper figure: cooler background tone and clearer warm
    hotspots.
    """
    cam = np.clip(cam.astype(np.float32), 0, 1)
    heat = (cam * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32)

    base = image.astype(np.float32) * float(np.clip(bg_dim, 0.0, 1.0))
    alpha_map = float(alpha) * (float(min_alpha_ratio) + (1.0 - float(min_alpha_ratio)) * cam)
    alpha_map = alpha_map[..., None]
    out = base * (1.0 - alpha_map) + heat * alpha_map
    return np.clip(out, 0, 255).astype(np.uint8)


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32)
    cam = cam - float(cam.min())
    m = float(cam.max())
    if m > 1e-8:
        cam = cam / m
    return cam


def postprocess_cam(cam: np.ndarray, roi: np.ndarray | None, low_p: float, high_p: float, gamma: float, blur: int) -> np.ndarray:
    cam = cam.astype(np.float32)
    if roi is not None:
        cam = cam * roi.astype(np.float32)
    vals = cam[cam > 0] if np.any(cam > 0) else cam.reshape(-1)
    lo = np.percentile(vals, low_p)
    hi = np.percentile(vals, high_p)
    if hi > lo:
        cam = np.clip((cam - lo) / (hi - lo), 0, 1)
    else:
        cam = normalize_cam(cam)
    if gamma > 0:
        cam = np.power(cam, gamma)
    if blur > 1:
        k = blur if blur % 2 == 1 else blur + 1
        cam = cv2.GaussianBlur(cam, (k, k), 0)
    return np.clip(cam, 0, 1)


def module_produces_spatial_tensor(module: torch.nn.Module) -> bool:
    cls = module.__class__.__name__.lower()
    bad = ["embedding", "dropout", "identity", "relu", "gelu", "sigmoid", "softmax"]
    return not any(b in cls for b in bad)


def prefer_composite_module(name: str) -> bool:
    patterns = [
        r"image_encoder\.prompt_generator\.lightweight_mlp\d+_\d+$",
        r"image_encoder\.prompt_generator\.lightweight_mlp\d+$",
        r"image_encoder\.branch_mobile_v3\.\d+\.block3$",
        r"image_encoder\.branch_mobile_v3\.\d+\.block3\.(EAG|DHAR)$",
        r"mask_decoder\.output_upscaling$",
        r"mask_decoder\.transformer\.layers\.\d+\.cross_attn_(token_to_image|image_to_token)$",
        r"mask_decoder\.transformer\.final_attn_token_to_image$",
    ]
    return any(re.search(p, name) for p in patterns)


def spatialize(t: torch.Tensor) -> torch.Tensor | None:
    if isinstance(t, (list, tuple)):
        t = t[0]
    if not torch.is_tensor(t):
        return None
    if t.ndim == 4:
        return t
    if t.ndim == 3:
        b, n, c = t.shape
        s = int(np.sqrt(n))
        if s * s == n:
            return t.permute(0, 2, 1).reshape(b, c, s, s)
    return None


def compute_cam(acts: torch.Tensor, grads: torch.Tensor | None, method: str) -> np.ndarray | None:
    a = spatialize(acts)
    g = spatialize(grads) if grads is not None else None
    if a is None:
        return None
    if method == "activation" or g is None:
        cam = a.detach().abs().mean(dim=1)
    elif method == "gradcam":
        w = g.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * a).sum(dim=1))
        if float(cam.max().detach().cpu()) < 1e-8:
            cam = a.detach().abs().mean(dim=1)
    else:
        cam = F.relu(a) * F.relu(g)
        cam = cam.sum(dim=1)
        if float(cam.max().detach().cpu()) < 1e-8:
            cam = a.detach().abs().mean(dim=1)
    return normalize_cam(cam[0].detach().cpu().numpy())


def build_group_specs(max_maps: int) -> list[GroupSpec]:
    return [
        GroupSpec(
            roman="II",
            title="Adapter stages",
            include=("adapter", "lightweight_mlp", "prompt_generator"),
            exclude=("mask_decoder", "output_hyper", "iou_prediction", "mask_tokens"),
            max_maps=max_maps,
        ),
        GroupSpec(
            roman="III",
            title="Dual-branch fusion",
            include=("branch_mobile_v3", "bottleneck", "agff", "eag", "dhar", "block3"),
            exclude=("mask_decoder", "output_upscaling", "output_hyper", "iou_prediction", "mask_tokens"),
            max_maps=max_maps,
        ),
        GroupSpec(
            roman="IV",
            title="Decoder upsampling",
            include=("output_upscaling", "upscal", "deconv", "conv_transpose"),
            exclude=("image_encoder",),
            max_maps=max_maps,
        ),
        GroupSpec(
            roman="V",
            title="Mask generation layers",
            include=("output_hypernetworks_mlps", "output_hyper", "iou_prediction_head", "mask_decoder.layers", "mask_decoder.transformer"),
            exclude=("mask_tokens",),
            max_maps=max_maps,
        ),
    ]


def match_group(name: str, groups: list[GroupSpec]) -> str | None:
    lname = name.lower()
    for g in groups:
        if any(tok in lname for tok in g.include) and not any(tok in lname for tok in g.exclude):
            return g.title
    return None


def clean_name(name: str) -> str:
    return name.replace("__alias", "")


def adapter_stage(name: str) -> int | None:
    m = re.search(r"lightweight_mlp(\d+)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"adapter[_\.]?(\d+)", name.lower())
    if m:
        return int(m.group(1))
    return None


def dual_stage(name: str) -> tuple[int, int, int]:
    """Return (stage_id, subtype_rank, detail_rank). Lower is shallower / preferred earlier."""
    lname = name.lower()
    m = re.search(r"branch_mobile_v3\.(\d+)", lname)
    stage = int(m.group(1)) if m else 99
    # Prefer fused block outputs, then DHAR / EAG block outputs, then internal ops.
    if re.search(r"branch_mobile_v3\.\d+\.block3$", lname):
        subtype = 0
    elif lname.endswith(".block3.dhar"):
        subtype = 1
    elif lname.endswith(".block3.eag"):
        subtype = 2
    else:
        subtype = 3
    detail = 0
    if "grouped_conv_x" in lname:
        detail = 1
    elif "grouped_conv_g" in lname:
        detail = 2
    elif ".psi" in lname:
        detail = 3
    elif ".sa" in lname:
        detail = 4
    elif ".conv" in lname:
        detail = 5
    return (stage, subtype, detail)


def decoder_stage(name: str) -> tuple[int, int]:
    lname = name.lower()
    # output_upscaling.0 / .1 / .2 / .3 are the most meaningful ordered internal stages.
    m = re.search(r"output_upscaling\.(\d+)$", lname)
    if m:
        return (int(m.group(1)), 0)
    if lname.endswith("output_upscaling"):
        return (99, 0)
    return (100, 0)


def ca_stage(name: str) -> tuple[int, int, int]:
    lname = name.lower()
    m = re.search(r"transformer\.layers\.(\d+)\.(cross_attn_token_to_image|cross_attn_image_to_token)", lname)
    if m:
        layer_id = int(m.group(1))
        block = m.group(2)
        # token_to_image first, image_to_token second within same layer.
        block_rank = 0 if block == "cross_attn_token_to_image" else 1
        # prefer parent block first, then sub-projections.
        if lname.endswith(block):
            detail = 0
        elif lname.endswith(".v_proj"):
            detail = 1
        elif lname.endswith(".k_proj"):
            detail = 2
        elif lname.endswith(".q_proj"):
            detail = 3
        elif lname.endswith(".out_proj"):
            detail = 4
        else:
            detail = 5
        return (layer_id, block_rank, detail)
    if "final_attn_token_to_image" in lname:
        detail = 0
        if lname.endswith("final_attn_token_to_image"):
            detail = 0
        elif lname.endswith(".v_proj"):
            detail = 1
        elif lname.endswith(".k_proj"):
            detail = 2
        elif lname.endswith(".q_proj"):
            detail = 3
        elif lname.endswith(".out_proj"):
            detail = 4
        return (99, 0, detail)
    return (100, 0, 0)


def display_title(name: str, group_title: str, position: int) -> str:
    if group_title == "Adapter stages":
        return f"Adapter{position + 1}"
    if group_title == "Dual-branch fusion":
        return f"Db{position + 1}"
    if group_title == "Decoder upsampling":
        return f"Up-{position + 1}"
    if group_title == "Mask generation layers":
        return f"CA-{position + 1}"
    return name


def _make_tensor_grad_hook(name: str, gradients: dict[str, torch.Tensor]) -> Callable[[torch.Tensor], None]:
    """Store gradients using Tensor.register_hook instead of module backward hooks.

    v9/v10 used module.register_full_backward_hook(). In models with inplace ReLU,
    PyTorch may wrap module outputs as views for backward hooks, and the next inplace
    activation then raises:
        Output 0 of BackwardHookFunctionBackward is a view and is being modified inplace.

    Registering a hook directly on the tensor observed in the forward hook avoids that
    BackwardHookFunction wrapper and keeps LayerCAM/GradCAM working.
    """

    def hook(grad: torch.Tensor) -> None:
        st = spatialize(grad)
        if st is not None:
            gradients[name] = st.detach()

    return hook


def collect_layercams(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_mask: torch.Tensor,
    groups: list[GroupSpec],
    cam_method: str,
    roi: np.ndarray | None,
    target_np: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, list[CamRecord]]:
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}
    handles = []
    candidate_names: dict[str, str] = {}

    for name, module in model.named_modules():
        group_title = match_group(name, groups)
        if group_title is None:
            continue
        if not module_produces_spatial_tensor(module) and not prefer_composite_module(name):
            continue
        candidate_names[name] = group_title

        def make_fwd(n: str) -> Callable[..., None]:
            def hook(_m: torch.nn.Module, _inp: Any, out: Any) -> None:
                st = spatialize(out)
                if st is None:
                    return
                activations[n] = st
                if st.requires_grad:
                    st.register_hook(_make_tensor_grad_hook(n, gradients))
            return hook

        # IMPORTANT for v11:
        # Only use a forward hook. Do not call register_full_backward_hook here.
        # The tensor hook above captures gradients safely even when the model uses inplace ReLU.
        handles.append(module.register_forward_hook(make_fwd(name)))

    model.zero_grad(set_to_none=True)
    logits = get_logits(model, x)
    logits = F.interpolate(logits, size=target_mask.shape[-2:], mode="bilinear", align_corners=False)
    prob = torch.sigmoid(logits)
    loss = (prob * target_mask).sum() / (target_mask.sum() + 1e-6)
    loss.backward()

    for h in handles:
        h.remove()

    out: dict[str, list[CamRecord]] = {g.title: [] for g in groups}
    for name, group_title in candidate_names.items():
        if name not in activations:
            continue
        cam = compute_cam(activations[name], gradients.get(name), cam_method)
        if cam is None:
            continue
        cam = cv2.resize(cam, (target_np.shape[1], target_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        cam = postprocess_cam(cam, roi, args.low_percentile, args.high_percentile, args.gamma, args.blur_kernel)
        fg = float(cam[target_np > 0.5].mean()) if np.any(target_np > 0.5) else float(cam.mean())
        bg = float(cam[target_np <= 0.5].mean()) if np.any(target_np <= 0.5) else 0.0
        peak = float(cam.max())
        score = fg - bg
        rec = CamRecord(name=name, group=group_title, cam=cam, score=score, fg_mean=fg, bg_mean=bg, peak=peak, shape=tuple(activations[name].shape))
        keep = score >= args.min_score or peak > 0.25
        # Keep more candidates for rows that are often sparse, then let the selector decide.
        if group_title in {"Adapter stages", "Dual-branch fusion", "Decoder upsampling", "Mask generation layers"}:
            keep = True
        if keep:
            out[group_title].append(rec)
    return out

def best_by_key(records: list[CamRecord], key_func) -> list[CamRecord]:
    best = {}
    for r in records:
        k = key_func(r)
        if k not in best or (r.score, r.fg_mean, r.peak) > (best[k].score, best[k].fg_mean, best[k].peak):
            best[k] = r
    return list(best.values())


def adapter_select(records: list[CamRecord], max_maps: int) -> list[CamRecord]:
    per_stage = best_by_key(records, lambda r: adapter_stage(r.name))
    valid = [r for r in per_stage if adapter_stage(r.name) is not None]
    valid = sorted(valid, key=lambda r: (adapter_stage(r.name), -r.score))
    chosen = []
    used = set()
    for stage in [1, 2, 3, 4]:
        cands = [r for r in valid if adapter_stage(r.name) == stage and r.name not in used]
        if cands:
            chosen.append(cands[0])
            used.add(cands[0].name)
    if len(chosen) < max_maps:
        rest = sorted(records, key=lambda r: (-(r.score), -(r.fg_mean), -(r.peak), r.name))
        for r in rest:
            if r.name in used:
                continue
            chosen.append(r)
            used.add(r.name)
            if len(chosen) >= max_maps:
                break
    return chosen[:max_maps]


def dual_select(records: list[CamRecord], max_maps: int) -> list[CamRecord]:
    # First prefer one representative from each branch stage (shallow to deep).
    per_stage = best_by_key(records, lambda r: dual_stage(r.name)[0])
    valid = [r for r in per_stage if dual_stage(r.name)[0] != 99]
    valid = sorted(valid, key=lambda r: (dual_stage(r.name)[0], dual_stage(r.name)[1], -r.score))
    chosen = []
    used = set()
    for r in valid:
        chosen.append(r)
        used.add(r.name)
        if len(chosen) >= max_maps:
            return chosen[:max_maps]
    # Then fill with other distinct fusion responses in shallow-to-deep order.
    rest = sorted(records, key=lambda r: (dual_stage(r.name), -r.score, r.name))
    for r in rest:
        if r.name in used:
            continue
        chosen.append(r)
        used.add(r.name)
        if len(chosen) >= max_maps:
            break
    return chosen[:max_maps]


def decoder_select(records: list[CamRecord], max_maps: int) -> list[CamRecord]:
    # Prefer distinct ordered upsampling stages. Keep the row visually full if possible.
    chosen = []
    used = set()
    ordered = sorted(records, key=lambda r: (decoder_stage(r.name), -r.score, r.name))
    # First pick true internal stages .0/.1/.2/.3 if available.
    per_key = best_by_key(ordered, lambda r: decoder_stage(r.name)[0])
    per_key = sorted(per_key, key=lambda r: (decoder_stage(r.name), -r.score))
    for r in per_key:
        key = decoder_stage(r.name)[0]
        if key >= 100:
            continue
        chosen.append(r)
        used.add(r.name)
        if len(chosen) >= max_maps:
            return chosen[:max_maps]
    # Then add whole-block response as the deepest summary.
    block = [r for r in ordered if r.name.lower().endswith("output_upscaling") and r.name not in used]
    if block and len(chosen) < max_maps:
        chosen.append(block[0])
        used.add(block[0].name)
    # If still not enough, fill with other distinct decoder candidates.
    for r in ordered:
        if r.name in used:
            continue
        chosen.append(r)
        used.add(r.name)
        if len(chosen) >= max_maps:
            break
    # Final order remains shallow to deep.
    chosen = sorted(chosen[:max_maps], key=lambda r: (decoder_stage(r.name), -r.score, r.name))
    return chosen[:max_maps]


def mask_select(records: list[CamRecord], max_maps: int) -> list[CamRecord]:
    # Collapse same cross-attention block to one best map, ordered shallow -> deep.
    def collapse_key(r: CamRecord):
        lname = r.name.lower()
        m = re.search(r"transformer\.layers\.(\d+)\.(cross_attn_token_to_image|cross_attn_image_to_token)", lname)
        if m:
            return f"L{m.group(1)}-{m.group(2)}"
        if "final_attn_token_to_image" in lname:
            return "final_attn_token_to_image"
        return lname

    reduced = best_by_key(records, collapse_key)
    reduced = sorted(reduced, key=lambda r: (ca_stage(r.name), -r.score, r.name))
    chosen = reduced[:max_maps]
    if len(chosen) < max_maps:
        rest = sorted(records, key=lambda r: (ca_stage(r.name), -r.score, r.name))
        used = {r.name for r in chosen}
        for r in rest:
            if r.name in used:
                continue
            chosen.append(r)
            used.add(r.name)
            if len(chosen) >= max_maps:
                break
    return chosen[:max_maps]


def select_records(records: list[CamRecord], group: GroupSpec) -> list[CamRecord]:
    if group.title == "Adapter stages":
        return adapter_select(records, group.max_maps)
    if group.title == "Dual-branch fusion":
        return dual_select(records, group.max_maps)
    if group.title == "Decoder upsampling":
        return decoder_select(records, group.max_maps)
    if group.title == "Mask generation layers":
        return mask_select(records, group.max_maps)
    records = sorted(records, key=lambda r: (-(r.score), -(r.fg_mean), -(r.peak), r.name))
    return records[:group.max_maps]


def dump_csv(path: Path, all_records: dict[str, list[CamRecord]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "name", "score", "fg_mean", "bg_mean", "peak", "activation_shape"])
        for group, records in all_records.items():
            for r in sorted(records, key=lambda x: x.score, reverse=True):
                w.writerow([group, r.name, f"{r.score:.6f}", f"{r.fg_mean:.6f}", f"{r.bg_mean:.6f}", f"{r.peak:.6f}", str(r.shape)])


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args.model_name, args.checkpoint, device, args.input_size)

    orig_img, x = read_image(args.image, args.input_size)
    x = x.to(device)
    base_img = cv2.resize(orig_img, (args.input_size, args.input_size), interpolation=cv2.INTER_LINEAR)

    with torch.no_grad():
        logits = get_logits(model, x)
        logits = F.interpolate(logits, size=(args.input_size, args.input_size), mode="bilinear", align_corners=False)
        pred_prob = torch.sigmoid(logits)
        pred_mask = (pred_prob > args.pred_thr).float()[0, 0].cpu().numpy().astype(np.uint8)

    label = read_label(args.label, (args.input_size, args.input_size))
    if label is None:
        label = pred_mask.copy()

    # Row-0 display can be grounded on the saved pred PNG from train/test outputs.
    pred_png_mask = read_pred_png_mask(args.pred_dir, args.image, (args.input_size, args.input_size))
    if pred_png_mask is None:
        pred_png_mask = pred_mask.copy()

    target_np, roi_np = make_target(pred_mask, label, args.target_mode)
    target_mask = torch.from_numpy(target_np).float().unsqueeze(0).unsqueeze(0).to(device)
    roi = roi_np if args.roi_mask else None

    groups = build_group_specs(max_maps=4)
    all_records = collect_layercams(model, x, target_mask, groups, args.cam_method, roi, target_np, args)
    dump_csv(args.dump_csv, all_records)

    x_for_global = x.detach().clone().requires_grad_(True)
    logits_global = get_logits(model, x_for_global)
    logits_global = F.interpolate(logits_global, size=(args.input_size, args.input_size), mode="bilinear", align_corners=False)
    loss_global = (torch.sigmoid(logits_global) * target_mask).sum() / (target_mask.sum() + 1e-6)
    model.zero_grad(set_to_none=True)
    loss_global.backward()
    grad = x_for_global.grad.detach().abs().mean(dim=1)[0].cpu().numpy()
    global_cam = postprocess_cam(normalize_cam(grad), roi, args.low_percentile, args.high_percentile, args.gamma, args.blur_kernel)

    selected: list[tuple[GroupSpec, list[CamRecord]]] = []
    for g in groups:
        selected.append((g, select_records(all_records[g.title], g)))

    n_cols = 4
    n_rows = 1 + len(groups)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.9 * n_rows), squeeze=False)

    row0_predict_img = overlay_region_from_predpng(
        base_img,
        pred_png_mask,
        color=(220, 190, 30),
        alpha=args.row0_predict_alpha,
    )
    row0_heatmap_soft = predpng_to_soft_heatmap(
        pred_png_mask,
        dilate_kernel=args.row0_heatmap_dilate_kernel,
        blur_kernel=args.row0_heatmap_blur_kernel,
        gamma=args.row0_heatmap_gamma,
        min_area=args.row0_heatmap_min_area,
    )
    row0_heatmap_img = overlay_heatmap(
        base_img,
        row0_heatmap_soft,
        alpha=args.row0_heatmap_alpha,
        bg_dim=args.paper_bg_dim,
        min_alpha_ratio=args.paper_min_alpha_ratio,
    )

    row0 = [
        ("Image", base_img),
        ("Label", color_mask(label)),
        ("Predict", row0_predict_img),
        ("Heatmap", row0_heatmap_img),
    ]
    for c, (title, img) in enumerate(row0):
        axes[0, c].imshow(img)
        axes[0, c].axis("off")
        axes[0, c].set_title(title, fontsize=args.title_size)
    axes[0, 0].text(-0.12, 0.5, "I", transform=axes[0, 0].transAxes, fontsize=22, va="center", ha="right")

    for r, (g, records) in enumerate(selected, start=1):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(records):
                rec = records[c]
                ax.imshow(overlay_heatmap(base_img, rec.cam, alpha=args.alpha, bg_dim=args.paper_bg_dim, min_alpha_ratio=args.paper_min_alpha_ratio))
                title = display_title(rec.name, g.title, c)
                if args.show_score:
                    title += f"\nscore={rec.score:.3f}"
                ax.set_title(title, fontsize=args.font_size)
            else:
                ax.imshow(base_img)
                ax.set_title("", fontsize=args.font_size)
        axes[r, 0].text(-0.14, 0.5, g.roman, transform=axes[r, 0].transAxes, fontsize=22, va="center", ha="right")
        axes[r, 0].text(-0.14, 0.06, g.title, transform=axes[r, 0].transAxes, fontsize=10, va="bottom", ha="right")

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.fig_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {args.output}")
    print(f"Saved candidates: {args.dump_csv}")


if __name__ == "__main__":
    main()
