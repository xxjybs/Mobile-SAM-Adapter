# #!/usr/bin/env python3
# """
# Compare multiple binary segmentation models on Orange test set.
#
# Features:
# 1) Evaluate multiple `.pth` models on same `test.txt` split.
# 2) Robustly normalize GT masks regardless of {0,1} or {0,255} encoding.
# 3) Save side-by-side qualitative figure similar to paper-style grid.
# 4) Export per-model metrics to CSV/JSON.
#
# Usage:
#   python scripts/compare_testset_models.py \
#     --config scripts/compare_models_config.example.json \
#     --data-root data/orange \
#     --test-list data/orange/imageset/test.txt \
#     --output-dir outputs/model_compare
# """
#
# from __future__ import annotations
#
# import argparse
# import csv
# import os
# import pickle
# import importlib
# import importlib.util
# import inspect
# import types
# import sys
# import json
# import warnings
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any, Dict, Iterable, List, Optional, Tuple
#
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn.functional as F
# from PIL import Image
# from tqdm import tqdm
#
#
# IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
#
#
# @dataclass
# class ModelConfig:
#     name: str
#     module: str
#     class_name: str
#     checkpoint: str
#     module_file: Optional[str]
#     init_kwargs: Dict[str, Any]
#     ckpt_state_dict_key: Optional[str]
#     strict: bool
#     input_size: Optional[List[int]]
#     normalize_mean: List[float]
#     normalize_std: List[float]
#     output_type: str
#     threshold: float
#     foreground_index: int
#     invert_mask: bool
#     max_skipped_keys: int
#     device: str
#     config_dir: str
#     pythonpath: List[str]
#
#
# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser()
#     p.add_argument("--config", type=Path, required=True, help="json config containing model definitions")
#     p.add_argument("--data-root", type=Path, default=Path("data/orange"))
#     p.add_argument("--images-dir", type=str, default="images")
#     p.add_argument("--masks-dir", type=str, default="masks")
#     p.add_argument("--test-list", type=Path, default=Path("data/orange/imageset/test1.txt"))
#     p.add_argument("--output-dir", type=Path, default=Path("outputs/model_compare"))
#     p.add_argument("--max-vis", type=int, default=12, help="max samples in qualitative panel")
#     p.add_argument("--vis-cols", type=int, default=8, help="max rows in qualitative panel (legacy arg name)")
#     p.add_argument("--vis-mode", type=str, choices=["color", "bw"], default="bw", help="visualization mode: color TP/FP/FN or black-white masks")
#     return p.parse_args()
#
#
# def read_config(path: Path) -> List[ModelConfig]:
#     data = json.loads(path.read_text(encoding="utf-8"))
#     models = []
#     for item in data["models"]:
#         models.append(
#             ModelConfig(
#                 name=item["name"],
#                 module=item["module"],
#                 class_name=item["class_name"],
#                 checkpoint=item["checkpoint"],
#                 module_file=item.get("module_file"),
#                 init_kwargs=item.get("init_kwargs", {}),
#                 ckpt_state_dict_key=item.get("ckpt_state_dict_key"),
#                 strict=bool(item.get("strict", True)),
#                 input_size=item.get("input_size"),
#                 normalize_mean=item.get("normalize_mean", [0.485, 0.456, 0.406]),
#                 normalize_std=item.get("normalize_std", [0.229, 0.224, 0.225]),
#                 output_type=item.get("output_type", "logits"),
#                 threshold=float(item.get("threshold", 0.5)),
#                 foreground_index=int(item.get("foreground_index", 1)),
#                 invert_mask=bool(item.get("invert_mask", False)),
#                 max_skipped_keys=int(item.get("max_skipped_keys", 0)),
#                 device=item.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
#                 config_dir=str(path.parent.resolve()),
#                 pythonpath=item.get("pythonpath", []),
#             )
#         )
#     return models
#
#
#
#
# def _resolve_module_file(module_file: str, config_dir: str) -> Path:
#     raw = Path(os.path.expandvars(module_file)).expanduser()
#     candidates: List[Path] = []
#
#     if raw.is_absolute():
#         candidates.append(raw)
#     else:
#         candidates.append((Path(config_dir) / raw).resolve())
#         candidates.append((Path.cwd() / raw).resolve())
#         # Common typo: use "home/..." instead of "/home/..."
#         if str(raw).startswith("home/"):
#             candidates.append((Path("/") / raw).resolve())
#
#     for c in candidates:
#         if c.exists():
#             return c
#
#     raise FileNotFoundError(
#         "module_file not found. tried: " + ", ".join(str(x) for x in candidates)
#     )
#
#
# def _resolve_extra_paths(extra_paths: List[str], config_dir: str) -> List[Path]:
#     resolved: List[Path] = []
#     for p in extra_paths:
#         raw = Path(os.path.expandvars(p)).expanduser()
#         cand = raw if raw.is_absolute() else (Path(config_dir) / raw)
#         cand = cand.resolve()
#         if cand.exists():
#             resolved.append(cand)
#     return resolved
#
#
# def _prepend_sys_path(paths: List[Path]) -> None:
#     for p in paths:
#         sp = str(p)
#         if sp not in sys.path:
#             sys.path.insert(0, sp)
#
#
# def _infer_module_name_from_file(module_path: Path, search_roots: List[Path]) -> Optional[str]:
#     for root in search_roots:
#         try:
#             rel = module_path.resolve().relative_to(root.resolve())
#         except ValueError:
#             continue
#         if rel.suffix != ".py":
#             continue
#         parts = list(rel.with_suffix("").parts)
#         if not parts:
#             continue
#         if parts[-1] == "__init__":
#             parts = parts[:-1]
#         if not parts:
#             continue
#         return ".".join(parts)
#     return None
#
#
# def import_model_module(module_name: str, module_file: Optional[str], config_dir: str, extra_paths: List[str]):
#     # try import with user-provided pythonpath first
#     resolved_extra_paths = _resolve_extra_paths(extra_paths, config_dir)
#     local_roots = [Path.cwd(), Path(__file__).resolve().parent.parent]
#     _prepend_sys_path(local_roots + resolved_extra_paths)
#     try:
#         return importlib.import_module(module_name)
#     except ModuleNotFoundError as e:
#         if not module_file:
#             raise ModuleNotFoundError(
#                 f"Cannot import module '{module_name}'. "
#                 f"Either install it / add to PYTHONPATH, or set 'module_file' in config."
#             ) from e
#
#         module_path = _resolve_module_file(module_file, config_dir)
#         # help relative imports inside external repos, e.g. "from nets.xxx import ..."
#         ancestor_paths = [module_path.parent]
#         for _ in range(4):
#             ancestor_paths.append(ancestor_paths[-1].parent)
#         _prepend_sys_path(ancestor_paths)
#
#         # Try importing with inferred package module name first.
#         search_roots = resolved_extra_paths + ancestor_paths
#         inferred_name = _infer_module_name_from_file(module_path, search_roots)
#         if inferred_name:
#             try:
#                 return importlib.import_module(inferred_name)
#             except Exception:
#                 pass
#
#         # Fallback 1: package-context execution for relative imports like `from .utils import ...`.
#         pkg_name = module_path.parent.name
#         pkg_module_name = f"{pkg_name}.{module_path.stem}"
#         if pkg_name not in sys.modules:
#             pkg_spec = importlib.util.spec_from_loader(pkg_name, loader=None)
#             if pkg_spec is not None:
#                 pkg_mod = importlib.util.module_from_spec(pkg_spec)
#                 pkg_mod.__path__ = [str(module_path.parent)]  # type: ignore[attr-defined]
#                 sys.modules[pkg_name] = pkg_mod
#         spec = importlib.util.spec_from_file_location(pkg_module_name, module_path)
#         if spec is not None and spec.loader is not None:
#             module = importlib.util.module_from_spec(spec)
#             sys.modules[pkg_module_name] = module
#             try:
#                 spec.loader.exec_module(module)
#                 return module
#             except Exception:
#                 pass
#
#         # Fallback 2: unique-name file execution.
#         unique_name = f"dynamic_model_{module_path.stem}_{abs(hash(str(module_path)))}"
#         spec = importlib.util.spec_from_file_location(unique_name, module_path)
#         if spec is None or spec.loader is None:
#             raise ImportError(f"Failed to create import spec from {module_path}")
#         module = importlib.util.module_from_spec(spec)
#         sys.modules[unique_name] = module
#         spec.loader.exec_module(module)
#         return module
#
#
#
#
# def _looks_like_state_dict(obj: Any) -> bool:
#     if not isinstance(obj, dict) or len(obj) == 0:
#         return False
#     sample_keys = list(obj.keys())[:20]
#     # typical state-dict values are tensors; keys often contain dots
#     tensor_like = all(torch.is_tensor(obj[k]) for k in sample_keys)
#     dotted_keys = any("." in str(k) for k in sample_keys)
#     return tensor_like or dotted_keys
#
#
# def _select_state_dict(checkpoint: Any, cfg: ModelConfig) -> Dict[str, Any]:
#     if _looks_like_state_dict(checkpoint):
#         return checkpoint
#
#     if not isinstance(checkpoint, dict):
#         raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")
#
#     if cfg.ckpt_state_dict_key:
#         if cfg.ckpt_state_dict_key in checkpoint:
#             candidate = checkpoint[cfg.ckpt_state_dict_key]
#             if isinstance(candidate, dict):
#                 return candidate
#             raise TypeError(
#                 f"checkpoint['{cfg.ckpt_state_dict_key}'] is not a dict, got {type(candidate)}"
#             )
#         else:
#             print(
#                 f"[Warning][{cfg.name}] ckpt_state_dict_key='{cfg.ckpt_state_dict_key}' not found. "
#                 f"Will try common keys automatically."
#             )
#
#     for key in ["state_dict", "model", "model_state", "model_state_dict", "net", "weights"]:
#         if key in checkpoint and isinstance(checkpoint[key], dict):
#             return checkpoint[key]
#
#     if _looks_like_state_dict(checkpoint):
#         return checkpoint
#
#     available = list(checkpoint.keys())
#     raise KeyError(
#         f"Cannot locate state_dict for model '{cfg.name}'. Available top-level keys: {available[:20]}"
#     )
#
#
#
# def _filter_incompatible_keys(
#     model: torch.nn.Module, state_dict: Dict[str, Any], model_name: str
# ) -> Tuple[Dict[str, Any], List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]]]:
#     model_state = model.state_dict()
#     filtered: Dict[str, Any] = {}
#     skipped = []
#
#     for k, v in state_dict.items():
#         if k not in model_state:
#             continue
#         if hasattr(v, "shape") and hasattr(model_state[k], "shape") and tuple(v.shape) != tuple(model_state[k].shape):
#             skipped.append((k, tuple(v.shape), tuple(model_state[k].shape)))
#             continue
#         filtered[k] = v
#
#     if skipped:
#         print(f"[Warning][{model_name}] skipped {len(skipped)} incompatible keys (shape mismatch).")
#         for name, ckpt_shape, model_shape in skipped[:8]:
#             print(f"  - {name}: ckpt{ckpt_shape} != model{model_shape}")
#         if len(skipped) > 8:
#             print(f"  ... and {len(skipped) - 8} more")
#
#     return filtered, skipped
#
#
# def _torch_load_checkpoint(path: str) -> Any:
#     """
#     Robust checkpoint loader across PyTorch versions.
#
#     - PyTorch >= 2.6 defaults `weights_only=True`, which can fail for checkpoints
#       that contain numpy/python objects.
#     - We first try safe load mode, then fallback to `weights_only=False` if needed.
#     """
#     try:
#         return torch.load(path, map_location="cpu", weights_only=True)
#     except TypeError:
#         # Older PyTorch may not support weights_only argument.
#         return torch.load(path, map_location="cpu")
#     except pickle.UnpicklingError:
#         warnings.warn(
#             f"Safe checkpoint loading failed for '{path}'. "
#             "Falling back to `weights_only=False`. "
#             "Only do this for trusted checkpoints.",
#             RuntimeWarning,
#         )
#         return torch.load(path, map_location="cpu", weights_only=False)
#
#
# def _is_bad_typing_symbol(obj: Any) -> bool:
#     return obj is Any or getattr(obj, "__module__", "") == "typing"
#
#
# def _is_constructor_candidate(obj: Any) -> bool:
#     if isinstance(obj, types.ModuleType):
#         return False
#     if _is_bad_typing_symbol(obj):
#         return False
#     return inspect.isclass(obj) or inspect.isfunction(obj) or callable(obj)
#
#
# def load_model(cfg: ModelConfig) -> torch.nn.Module:
#     module = import_model_module(cfg.module, cfg.module_file, cfg.config_dir, cfg.pythonpath)
#     # Prefer model_dict mapping when available (more explicit and less error-prone).
#     if hasattr(module, "model_dict") and cfg.class_name in module.model_dict:
#         cls = module.model_dict[cfg.class_name]
#     elif hasattr(module, cfg.class_name):
#         cls = getattr(module, cfg.class_name)
#     else:
#         raise AttributeError(f"Cannot find class '{cfg.class_name}' in module {cfg.module}")
#
#     # Some projects expose submodules in __init__.py, so cfg.class_name may resolve to a module.
#     # Also guard against typing placeholders like typing.Any.
#     if isinstance(cls, types.ModuleType) or _is_bad_typing_symbol(cls):
#         module_obj = cls if isinstance(cls, types.ModuleType) else module
#         resolved = False
#         fallback_names = [
#             cfg.class_name.split(".")[-1],
#             "build_model",
#             "get_model",
#             "create_model",
#             "Model",
#             "model",
#         ]
#         for name in fallback_names:
#             cand = getattr(module_obj, name, None)
#             if _is_constructor_candidate(cand):
#                 cls = cand
#                 resolved = True
#                 break
#         if (not resolved) and _is_bad_typing_symbol(cls) and hasattr(module, "model_dict") and cfg.class_name in module.model_dict:
#             md_cand = module.model_dict[cfg.class_name]
#             if _is_constructor_candidate(md_cand):
#                 cls = md_cand
#                 resolved = True
#         if not resolved:
#             # last fallback: first public callable in submodule
#             public_callables = [
#                 getattr(module_obj, n)
#                 for n in dir(module_obj)
#                 if not n.startswith("_") and _is_constructor_candidate(getattr(module_obj, n))
#             ]
#             if public_callables:
#                 cls = public_callables[0]
#             else:
#                 raise TypeError(
#                     f"Resolved '{cfg.class_name}' to submodule '{module_obj.__name__}', "
#                     "but no callable constructor was found."
#                 )
#
#     if not callable(cls):
#         raise TypeError(
#             f"Resolved constructor for '{cfg.name}' is not callable: {type(cls)}. "
#             "Please check class_name/module configuration."
#         )
#     model = cls(**cfg.init_kwargs)
#
#     checkpoint = _torch_load_checkpoint(cfg.checkpoint)
#     state_dict = _select_state_dict(checkpoint, cfg)
#
#     fixed = {}
#     for k, v in state_dict.items():
#         nk = str(k)
#         if nk.startswith("module."):
#             nk = nk.replace("module.", "", 1)
#         if nk.startswith("model."):
#             nk = nk.replace("model.", "", 1)
#         fixed[nk] = v
#     state_dict = fixed
#     state_dict, skipped = _filter_incompatible_keys(model, state_dict, cfg.name)
#     if len(skipped) > cfg.max_skipped_keys:
#         raise RuntimeError(
#             f"[{cfg.name}] skipped {len(skipped)} incompatible keys, exceeding "
#             f"max_skipped_keys={cfg.max_skipped_keys}. This usually means the checkpoint "
#             "does not match the model architecture/version. Please use a matching trained checkpoint."
#         )
#
#     missing, unexpected = model.load_state_dict(state_dict, strict=cfg.strict)
#     if missing:
#         print(f"[Warning][{cfg.name}] missing keys: {len(missing)}")
#     if unexpected:
#         print(f"[Warning][{cfg.name}] unexpected keys: {len(unexpected)}")
#
#     model.to(cfg.device)
#     model.eval()
#     return model
#
#
# def find_image(images_dir: Path, stem: str) -> Path:
#     for ext in IMG_EXTS:
#         p = images_dir / f"{stem}{ext}"
#         if p.exists():
#             return p
#     raise FileNotFoundError(f"Image not found for stem '{stem}' under {images_dir}")
#
#
# def to_tensor(img: Image.Image, mean: List[float], std: List[float]) -> torch.Tensor:
#     arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
#     arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
#     arr = np.transpose(arr, (2, 0, 1))
#     return torch.from_numpy(arr)
#
#
# def normalize_binary_mask(mask: np.ndarray) -> np.ndarray:
#     """Normalize binary mask to {0,1}, robust to {0,1} or {0,255} or probability maps."""
#     if mask.ndim == 3:
#         mask = mask[..., 0]
#     mask = mask.astype(np.float32)
#     unique_vals = np.unique(mask)
#     if unique_vals.size <= 2 and set(np.round(unique_vals).astype(int).tolist()) <= {0, 1}:
#         return mask.astype(np.uint8)
#     if unique_vals.size <= 2 and set(np.round(unique_vals).astype(int).tolist()) <= {0, 255}:
#         return (mask > 127).astype(np.uint8)
#     return (mask > 0.5).astype(np.uint8)
#
#
# def infer_binary_mask(model: torch.nn.Module, cfg: ModelConfig, image: Image.Image) -> np.ndarray:
#     orig_w, orig_h = image.size
#     proc = image
#     if cfg.input_size is not None:
#         proc = image.resize((cfg.input_size[1], cfg.input_size[0]), Image.BILINEAR)
#
#     x = to_tensor(proc, cfg.normalize_mean, cfg.normalize_std).unsqueeze(0).to(cfg.device)
#     with torch.no_grad():
#         y = model(x)
#         if isinstance(y, (list, tuple)):
#             y = y[0]
#
#         if y.ndim == 4 and y.shape[1] > 1:
#             # multi-class logits/probabilities => class map
#             pred = (torch.argmax(y, dim=1) == int(cfg.foreground_index)).float()
#         else:
#             if y.ndim == 4:
#                 y = y[:, 0]
#             if cfg.output_type == "logits":
#                 y = torch.sigmoid(y)
#             pred = (y > cfg.threshold).float()
#
#     pred = pred[0].cpu().numpy().astype(np.uint8)
#     if cfg.input_size is not None and (pred.shape[1] != orig_w or pred.shape[0] != orig_h):
#         pred = np.array(Image.fromarray(pred).resize((orig_w, orig_h), Image.NEAREST), dtype=np.uint8)
#     pred = normalize_binary_mask(pred)
#     if cfg.invert_mask:
#         pred = (1 - pred).astype(np.uint8)
#     return pred
#
#
# def calc_stats(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
#     pred = pred.astype(bool)
#     gt = gt.astype(bool)
#     tp = np.logical_and(pred, gt).sum()
#     tn = np.logical_and(~pred, ~gt).sum()
#     fp = np.logical_and(pred, ~gt).sum()
#     fn = np.logical_and(~pred, gt).sum()
#
#     eps = 1e-8
#     iou_fg = tp / (tp + fp + fn + eps)
#     iou_bg = tn / (tn + fp + fn + eps)
#     miou = (iou_fg + iou_bg) / 2.0
#     dice = (2 * tp) / (2 * tp + fp + fn + eps)
#     precision = tp / (tp + fp + eps)
#     recall = tp / (tp + fn + eps)
#     f1 = (2 * precision * recall) / (precision + recall + eps)
#
#     return {
#         "iou_fg": float(iou_fg),
#         "iou_bg": float(iou_bg),
#         "miou": float(miou),
#         "dice": float(dice),
#         "precision": float(precision),
#         "recall": float(recall),
#         "f1": float(f1),
#     }
#
#
# def overlay_mask(mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
#     h, w = mask.shape
#     out = np.zeros((h, w, 3), dtype=np.uint8)
#     out[mask == 1] = np.asarray(color, dtype=np.uint8)
#     return out
#
#
# def save_qualitative_grid(
#     output_path: Path,
#     image_stems: List[str],
#     image_dir: Path,
#     gt_masks: Dict[str, np.ndarray],
#     pred_masks: Dict[str, Dict[str, np.ndarray]],
#     model_names: List[str],
#     rows: int,
#     vis_mode: str,
# ) -> None:
#     n = len(image_stems)
#     if n == 0:
#         return
#     if rows > 0 and n > rows:
#         # keep order but downsample evenly to avoid an over-long figure
#         idx = np.linspace(0, n - 1, rows, dtype=int).tolist()
#         image_stems = [image_stems[i] for i in idx]
#         n = len(image_stems)
#     cols = 2 + len(model_names)
#
#     fig, axes = plt.subplots(nrows=n, ncols=cols, figsize=(2.5 * cols, 2.2 * n), squeeze=False)
#     column_labels = ["(a) Image", "(b) Label"] + [f"({chr(97 + i)}) {name}" for i, name in enumerate(model_names, start=2)]
#     for r, stem in enumerate(image_stems):
#         img = np.asarray(Image.open(find_image(image_dir, stem)).convert("RGB"), dtype=np.uint8)
#         gt = gt_masks[stem]
#
#         axes[r, 0].imshow(img)
#         axes[r, 0].axis("off")
#
#         if vis_mode == "bw":
#             axes[r, 1].imshow((gt * 255).astype(np.uint8), cmap="gray", vmin=0, vmax=255)
#         else:
#             axes[r, 1].imshow(overlay_mask(gt, (178, 160, 0)))
#         axes[r, 1].axis("off")
#
#         for c, mname in enumerate(model_names, start=2):
#             pmask = pred_masks[mname][stem]
#             if vis_mode == "bw":
#                 axes[r, c].imshow((pmask * 255).astype(np.uint8), cmap="gray", vmin=0, vmax=255)
#             else:
#                 canvas = np.zeros((pmask.shape[0], pmask.shape[1], 3), dtype=np.uint8)
#                 canvas[(pmask == 1) & (gt == 1)] = (0, 170, 0)
#                 canvas[(pmask == 1) & (gt == 0)] = (220, 0, 0)
#                 canvas[(pmask == 0) & (gt == 1)] = (190, 150, 0)
#                 axes[r, c].imshow(canvas)
#             axes[r, c].axis("off")
#
#     # Put captions at the very bottom like paper figures.
#     for c, text in enumerate(column_labels):
#         x = (c + 0.5) / cols
#         fig.text(x, 0.008, text, ha="center", va="bottom", fontsize=11)
#
#     plt.tight_layout(rect=[0, 0.03, 1, 1])
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(output_path, dpi=200, bbox_inches="tight")
#     plt.close(fig)
#
#
# def write_metrics_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
#     rows = list(rows)
#     keys = ["model", "miou", "iou_fg", "iou_bg", "dice", "precision", "recall", "f1"]
#     with path.open("w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=keys)
#         writer.writeheader()
#         writer.writerows(rows)
#
#
# def main() -> None:
#     args = parse_args()
#     model_cfgs = read_config(args.config)
#
#     image_dir = args.data_root / args.images_dir
#     mask_dir = args.data_root / args.masks_dir
#     stems = [x.strip() for x in args.test_list.read_text(encoding="utf-8").splitlines() if x.strip()]
#
#     gt_masks: Dict[str, np.ndarray] = {}
#     for s in stems:
#         gt_path = mask_dir / f"{s}.png"
#         gt = np.asarray(Image.open(gt_path))
#         gt_masks[s] = normalize_binary_mask(gt)
#
#     loaded_models = {cfg.name: load_model(cfg) for cfg in model_cfgs}
#
#     pred_masks: Dict[str, Dict[str, np.ndarray]] = {cfg.name: {} for cfg in model_cfgs}
#     metric_acc: Dict[str, List[Dict[str, float]]] = {cfg.name: [] for cfg in model_cfgs}
#
#     for stem in tqdm(stems, desc="Evaluating samples"):
#         image = Image.open(find_image(image_dir, stem)).convert("RGB")
#         gt = gt_masks[stem]
#
#         for cfg in model_cfgs:
#             pred = infer_binary_mask(loaded_models[cfg.name], cfg, image)
#             pred_masks[cfg.name][stem] = pred
#             metric_acc[cfg.name].append(calc_stats(pred, gt))
#
#     summary_rows = []
#     for cfg in model_cfgs:
#         arr = metric_acc[cfg.name]
#         row = {
#             "model": cfg.name,
#             "miou": float(np.mean([x["miou"] for x in arr])),
#             "iou_fg": float(np.mean([x["iou_fg"] for x in arr])),
#             "iou_bg": float(np.mean([x["iou_bg"] for x in arr])),
#             "dice": float(np.mean([x["dice"] for x in arr])),
#             "precision": float(np.mean([x["precision"] for x in arr])),
#             "recall": float(np.mean([x["recall"] for x in arr])),
#             "f1": float(np.mean([x["f1"] for x in arr])),
#         }
#         summary_rows.append(row)
#
#     args.output_dir.mkdir(parents=True, exist_ok=True)
#     write_metrics_csv(args.output_dir / "metrics_summary.csv", summary_rows)
#     (args.output_dir / "metrics_summary.json").write_text(
#         json.dumps(summary_rows, indent=2, ensure_ascii=False),
#         encoding="utf-8",
#     )
#
#     vis_stems = stems[: max(0, args.max_vis)]
#     save_qualitative_grid(
#         output_path=args.output_dir / "qualitative_grid.png",
#         image_stems=vis_stems,
#         image_dir=image_dir,
#         gt_masks=gt_masks,
#         pred_masks=pred_masks,
#         model_names=[cfg.name for cfg in model_cfgs],
#         rows=args.vis_cols,
#         vis_mode=args.vis_mode,
#     )
#
#     print("\n=== Done ===")
#     print(f"Saved metrics: {args.output_dir / 'metrics_summary.csv'}")
#     print(f"Saved figure : {args.output_dir / 'qualitative_grid.png'}")
#
#
# if __name__ == "__main__":
#     main()

from __future__ import annotations

import argparse
import csv
import os
import pickle
import importlib
import importlib.util
import inspect
import types
import sys
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


@dataclass
class ModelConfig:
    name: str
    module: str
    class_name: str
    checkpoint: str
    module_file: Optional[str]
    init_kwargs: Dict[str, Any]
    ckpt_state_dict_key: Optional[str]
    strict: bool
    input_size: Optional[List[int]]
    normalize_mean: List[float]
    normalize_std: List[float]
    output_type: str
    threshold: float
    foreground_index: int
    invert_mask: bool
    max_skipped_keys: int
    device: str
    config_dir: str
    pythonpath: List[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True, help="json config containing model definitions")
    p.add_argument("--data-root", type=Path, default=Path("data/orange"))
    p.add_argument("--images-dir", type=str, default="images")
    p.add_argument("--masks-dir", type=str, default="masks")
    p.add_argument("--test-list", type=Path, default=Path("data/orange/imageset/test.txt"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/model_compare"))
    p.add_argument("--max-vis", type=int, default=12, help="max samples in qualitative panel")
    p.add_argument("--vis-cols", type=int, default=8, help="max rows in qualitative panel (legacy arg name)")
    p.add_argument(
        "--vis-mode",
        type=str,
        choices=["color", "bw"],
        default="color",
        help="visualization mode: color TP/FP/FN (default) or black-white masks",
    )
    return p.parse_args()


def read_config(path: Path) -> List[ModelConfig]:
    data = json.loads(path.read_text(encoding="utf-8"))
    models = []
    for item in data["models"]:
        models.append(
            ModelConfig(
                name=item["name"],
                module=item["module"],
                class_name=item["class_name"],
                checkpoint=item["checkpoint"],
                module_file=item.get("module_file"),
                init_kwargs=item.get("init_kwargs", {}),
                ckpt_state_dict_key=item.get("ckpt_state_dict_key"),
                strict=bool(item.get("strict", True)),
                input_size=item.get("input_size"),
                normalize_mean=item.get("normalize_mean", [0.485, 0.456, 0.406]),
                normalize_std=item.get("normalize_std", [0.229, 0.224, 0.225]),
                output_type=item.get("output_type", "logits"),
                threshold=float(item.get("threshold", 0.5)),
                foreground_index=int(item.get("foreground_index", 1)),
                invert_mask=bool(item.get("invert_mask", False)),
                max_skipped_keys=int(item.get("max_skipped_keys", 0)),
                device=item.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
                config_dir=str(path.parent.resolve()),
                pythonpath=item.get("pythonpath", []),
            )
        )
    return models




def _resolve_module_file(module_file: str, config_dir: str) -> Path:
    raw = Path(os.path.expandvars(module_file)).expanduser()
    candidates: List[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((Path(config_dir) / raw).resolve())
        candidates.append((Path.cwd() / raw).resolve())
        # Common typo: use "home/..." instead of "/home/..."
        if str(raw).startswith("home/"):
            candidates.append((Path("/") / raw).resolve())

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        "module_file not found. tried: " + ", ".join(str(x) for x in candidates)
    )


def _resolve_extra_paths(extra_paths: List[str], config_dir: str) -> List[Path]:
    resolved: List[Path] = []
    for p in extra_paths:
        raw = Path(os.path.expandvars(p)).expanduser()
        cand = raw if raw.is_absolute() else (Path(config_dir) / raw)
        cand = cand.resolve()
        if cand.exists():
            resolved.append(cand)
    return resolved


def _prepend_sys_path(paths: List[Path]) -> None:
    for p in paths:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def _infer_module_name_from_file(module_path: Path, search_roots: List[Path]) -> Optional[str]:
    for root in search_roots:
        try:
            rel = module_path.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        if rel.suffix != ".py":
            continue
        parts = list(rel.with_suffix("").parts)
        if not parts:
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts:
            continue
        return ".".join(parts)
    return None


def import_model_module(module_name: str, module_file: Optional[str], config_dir: str, extra_paths: List[str]):
    # try import with user-provided pythonpath first
    resolved_extra_paths = _resolve_extra_paths(extra_paths, config_dir)
    local_roots = [Path.cwd(), Path(__file__).resolve().parent.parent]
    _prepend_sys_path(local_roots + resolved_extra_paths)
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if not module_file:
            raise ModuleNotFoundError(
                f"Cannot import module '{module_name}'. "
                f"Either install it / add to PYTHONPATH, or set 'module_file' in config."
            ) from e

        module_path = _resolve_module_file(module_file, config_dir)
        # help relative imports inside external repos, e.g. "from nets.xxx import ..."
        ancestor_paths = [module_path.parent]
        for _ in range(4):
            ancestor_paths.append(ancestor_paths[-1].parent)
        _prepend_sys_path(ancestor_paths)

        # Try importing with inferred package module name first.
        search_roots = resolved_extra_paths + ancestor_paths
        inferred_name = _infer_module_name_from_file(module_path, search_roots)
        if inferred_name:
            try:
                return importlib.import_module(inferred_name)
            except Exception:
                pass

        # Fallback 1: package-context execution for relative imports like `from .utils import ...`.
        pkg_name = module_path.parent.name
        pkg_module_name = f"{pkg_name}.{module_path.stem}"
        if pkg_name not in sys.modules:
            pkg_spec = importlib.util.spec_from_loader(pkg_name, loader=None)
            if pkg_spec is not None:
                pkg_mod = importlib.util.module_from_spec(pkg_spec)
                pkg_mod.__path__ = [str(module_path.parent)]  # type: ignore[attr-defined]
                sys.modules[pkg_name] = pkg_mod
        spec = importlib.util.spec_from_file_location(pkg_module_name, module_path)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            sys.modules[pkg_module_name] = module
            try:
                spec.loader.exec_module(module)
                return module
            except Exception:
                pass

        # Fallback 2: unique-name file execution.
        unique_name = f"dynamic_model_{module_path.stem}_{abs(hash(str(module_path)))}"
        spec = importlib.util.spec_from_file_location(unique_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create import spec from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = module
        spec.loader.exec_module(module)
        return module




def _looks_like_state_dict(obj: Any) -> bool:
    if not isinstance(obj, dict) or len(obj) == 0:
        return False
    sample_keys = list(obj.keys())[:20]
    # typical state-dict values are tensors; keys often contain dots
    tensor_like = all(torch.is_tensor(obj[k]) for k in sample_keys)
    dotted_keys = any("." in str(k) for k in sample_keys)
    return tensor_like or dotted_keys


def _select_state_dict(checkpoint: Any, cfg: ModelConfig) -> Dict[str, Any]:
    if _looks_like_state_dict(checkpoint):
        return checkpoint

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    if cfg.ckpt_state_dict_key:
        if cfg.ckpt_state_dict_key in checkpoint:
            candidate = checkpoint[cfg.ckpt_state_dict_key]
            if isinstance(candidate, dict):
                return candidate
            raise TypeError(
                f"checkpoint['{cfg.ckpt_state_dict_key}'] is not a dict, got {type(candidate)}"
            )
        else:
            print(
                f"[Warning][{cfg.name}] ckpt_state_dict_key='{cfg.ckpt_state_dict_key}' not found. "
                f"Will try common keys automatically."
            )

    for key in ["state_dict", "model", "model_state", "model_state_dict", "net", "weights"]:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    if _looks_like_state_dict(checkpoint):
        return checkpoint

    available = list(checkpoint.keys())
    raise KeyError(
        f"Cannot locate state_dict for model '{cfg.name}'. Available top-level keys: {available[:20]}"
    )



def _filter_incompatible_keys(
    model: torch.nn.Module, state_dict: Dict[str, Any], model_name: str
) -> Tuple[Dict[str, Any], List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]]]:
    model_state = model.state_dict()
    filtered: Dict[str, Any] = {}
    skipped = []

    for k, v in state_dict.items():
        if k not in model_state:
            continue
        if hasattr(v, "shape") and hasattr(model_state[k], "shape") and tuple(v.shape) != tuple(model_state[k].shape):
            skipped.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        filtered[k] = v

    if skipped:
        print(f"[Warning][{model_name}] skipped {len(skipped)} incompatible keys (shape mismatch).")
        for name, ckpt_shape, model_shape in skipped[:8]:
            print(f"  - {name}: ckpt{ckpt_shape} != model{model_shape}")
        if len(skipped) > 8:
            print(f"  ... and {len(skipped) - 8} more")

    return filtered, skipped


def _torch_load_checkpoint(path: str) -> Any:
    """
    Robust checkpoint loader across PyTorch versions.

    - PyTorch >= 2.6 defaults `weights_only=True`, which can fail for checkpoints
      that contain numpy/python objects.
    - We first try safe load mode, then fallback to `weights_only=False` if needed.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older PyTorch may not support weights_only argument.
        return torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        warnings.warn(
            f"Safe checkpoint loading failed for '{path}'. "
            "Falling back to `weights_only=False`. "
            "Only do this for trusted checkpoints.",
            RuntimeWarning,
        )
        return torch.load(path, map_location="cpu", weights_only=False)


def _is_bad_typing_symbol(obj: Any) -> bool:
    return obj is Any or getattr(obj, "__module__", "") == "typing"


def _is_constructor_candidate(obj: Any) -> bool:
    if isinstance(obj, types.ModuleType):
        return False
    if _is_bad_typing_symbol(obj):
        return False
    return inspect.isclass(obj) or inspect.isfunction(obj) or callable(obj)


def load_model(cfg: ModelConfig) -> torch.nn.Module:
    module = import_model_module(cfg.module, cfg.module_file, cfg.config_dir, cfg.pythonpath)
    # Prefer model_dict mapping when available (more explicit and less error-prone).
    if hasattr(module, "model_dict") and cfg.class_name in module.model_dict:
        cls = module.model_dict[cfg.class_name]
    elif hasattr(module, cfg.class_name):
        cls = getattr(module, cfg.class_name)
    else:
        raise AttributeError(f"Cannot find class '{cfg.class_name}' in module {cfg.module}")

    # Some projects expose submodules in __init__.py, so cfg.class_name may resolve to a module.
    # Also guard against typing placeholders like typing.Any.
    if isinstance(cls, types.ModuleType) or _is_bad_typing_symbol(cls):
        module_obj = cls if isinstance(cls, types.ModuleType) else module
        resolved = False
        fallback_names = [
            cfg.class_name.split(".")[-1],
            "build_model",
            "get_model",
            "create_model",
            "Model",
            "model",
        ]
        for name in fallback_names:
            cand = getattr(module_obj, name, None)
            if _is_constructor_candidate(cand):
                cls = cand
                resolved = True
                break
        if (not resolved) and _is_bad_typing_symbol(cls) and hasattr(module, "model_dict") and cfg.class_name in module.model_dict:
            md_cand = module.model_dict[cfg.class_name]
            if _is_constructor_candidate(md_cand):
                cls = md_cand
                resolved = True
        if not resolved:
            # last fallback: first public callable in submodule
            public_callables = [
                getattr(module_obj, n)
                for n in dir(module_obj)
                if not n.startswith("_") and _is_constructor_candidate(getattr(module_obj, n))
            ]
            if public_callables:
                cls = public_callables[0]
            else:
                raise TypeError(
                    f"Resolved '{cfg.class_name}' to submodule '{module_obj.__name__}', "
                    "but no callable constructor was found."
                )

    if not callable(cls):
        raise TypeError(
            f"Resolved constructor for '{cfg.name}' is not callable: {type(cls)}. "
            "Please check class_name/module configuration."
        )
    model = cls(**cfg.init_kwargs)

    checkpoint = _torch_load_checkpoint(cfg.checkpoint)
    state_dict = _select_state_dict(checkpoint, cfg)

    fixed = {}
    for k, v in state_dict.items():
        nk = str(k)
        if nk.startswith("module."):
            nk = nk.replace("module.", "", 1)
        if nk.startswith("model."):
            nk = nk.replace("model.", "", 1)
        fixed[nk] = v
    state_dict = fixed
    state_dict, skipped = _filter_incompatible_keys(model, state_dict, cfg.name)
    if len(skipped) > cfg.max_skipped_keys:
        raise RuntimeError(
            f"[{cfg.name}] skipped {len(skipped)} incompatible keys, exceeding "
            f"max_skipped_keys={cfg.max_skipped_keys}. This usually means the checkpoint "
            "does not match the model architecture/version. Please use a matching trained checkpoint."
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=cfg.strict)
    if missing:
        print(f"[Warning][{cfg.name}] missing keys: {len(missing)}")
    if unexpected:
        print(f"[Warning][{cfg.name}] unexpected keys: {len(unexpected)}")

    model.to(cfg.device)
    model.eval()
    return model


def find_image(images_dir: Path, stem: str) -> Path:
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Image not found for stem '{stem}' under {images_dir}")


def to_tensor(img: Image.Image, mean: List[float], std: List[float]) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def normalize_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Normalize binary mask to {0,1}, robust to {0,1} or {0,255} or probability maps."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask.astype(np.float32)
    unique_vals = np.unique(mask)
    if unique_vals.size <= 2 and set(np.round(unique_vals).astype(int).tolist()) <= {0, 1}:
        return mask.astype(np.uint8)
    if unique_vals.size <= 2 and set(np.round(unique_vals).astype(int).tolist()) <= {0, 255}:
        return (mask > 127).astype(np.uint8)
    return (mask > 0.5).astype(np.uint8)


def infer_binary_mask(model: torch.nn.Module, cfg: ModelConfig, image: Image.Image) -> np.ndarray:
    orig_w, orig_h = image.size
    proc = image
    if cfg.input_size is not None:
        proc = image.resize((cfg.input_size[1], cfg.input_size[0]), Image.BILINEAR)

    x = to_tensor(proc, cfg.normalize_mean, cfg.normalize_std).unsqueeze(0).to(cfg.device)
    with torch.no_grad():
        y = model(x)
        if isinstance(y, (list, tuple)):
            y = y[0]

        if y.ndim == 4 and y.shape[1] > 1:
            # multi-class logits/probabilities => class map
            pred = (torch.argmax(y, dim=1) == int(cfg.foreground_index)).float()
        else:
            if y.ndim == 4:
                y = y[:, 0]
            if cfg.output_type == "logits":
                y = torch.sigmoid(y)
            pred = (y > cfg.threshold).float()

    pred = pred[0].cpu().numpy().astype(np.uint8)
    if cfg.input_size is not None and (pred.shape[1] != orig_w or pred.shape[0] != orig_h):
        pred = np.array(Image.fromarray(pred).resize((orig_w, orig_h), Image.NEAREST), dtype=np.uint8)
    pred = normalize_binary_mask(pred)
    if cfg.invert_mask:
        pred = (1 - pred).astype(np.uint8)
    return pred


def calc_stats(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    eps = 1e-8
    iou_fg = tp / (tp + fp + fn + eps)
    iou_bg = tn / (tn + fp + fn + eps)
    miou = (iou_fg + iou_bg) / 2.0
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)

    return {
        "iou_fg": float(iou_fg),
        "iou_bg": float(iou_bg),
        "miou": float(miou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def overlay_mask(mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[mask == 1] = np.asarray(color, dtype=np.uint8)
    return out


def save_qualitative_grid(
    output_path: Path,
    image_stems: List[str],
    image_dir: Path,
    gt_masks: Dict[str, np.ndarray],
    pred_masks: Dict[str, Dict[str, np.ndarray]],
    model_names: List[str],
    rows: int,
    vis_mode: str,
) -> None:
    n = len(image_stems)
    if n == 0:
        return
    if rows > 0 and n > rows:
        # keep order but downsample evenly to avoid an over-long figure
        idx = np.linspace(0, n - 1, rows, dtype=int).tolist()
        image_stems = [image_stems[i] for i in idx]
        n = len(image_stems)
    cols = 2 + len(model_names)

    fig, axes = plt.subplots(nrows=n, ncols=cols, figsize=(2.5 * cols, 2.2 * n), squeeze=False)
    column_labels = ["(a) Image", "(b) Label"] + [f"({chr(97 + i)}) {name}" for i, name in enumerate(model_names, start=2)]
    for r, stem in enumerate(image_stems):
        img = np.asarray(Image.open(find_image(image_dir, stem)).convert("RGB"), dtype=np.uint8)
        gt = gt_masks[stem]

        axes[r, 0].imshow(img)
        axes[r, 0].axis("off")

        if vis_mode == "bw":
            axes[r, 1].imshow((gt * 255).astype(np.uint8), cmap="gray", vmin=0, vmax=255)
        else:
            axes[r, 1].imshow(overlay_mask(gt, (178, 160, 0)))
        axes[r, 1].axis("off")

        for c, mname in enumerate(model_names, start=2):
            pmask = pred_masks[mname][stem]
            if vis_mode == "bw":
                axes[r, c].imshow((pmask * 255).astype(np.uint8), cmap="gray", vmin=0, vmax=255)
            else:
                canvas = np.zeros((pmask.shape[0], pmask.shape[1], 3), dtype=np.uint8)
                canvas[(pmask == 1) & (gt == 1)] = (0, 170, 0)
                canvas[(pmask == 1) & (gt == 0)] = (220, 0, 0)
                canvas[(pmask == 0) & (gt == 1)] = (190, 150, 0)
                axes[r, c].imshow(canvas)
            axes[r, c].axis("off")

    # Put captions at the very bottom like paper figures.
    for c, text in enumerate(column_labels):
        x = (c + 0.5) / cols
        fig.text(x, 0.008, text, ha="center", va="bottom", fontsize=11)

    # if vis_mode == "color":
    #     # TP/FP/FN legend for colored comparison columns.
    #     fig.text(0.02, 0.008, "Green=TP  Red=FP  Yellow=FN", ha="left", va="bottom", fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_metrics_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    keys = ["model", "miou", "iou_fg", "iou_bg", "dice", "precision", "recall", "f1"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    model_cfgs = read_config(args.config)

    image_dir = args.data_root / args.images_dir
    mask_dir = args.data_root / args.masks_dir
    stems = [x.strip() for x in args.test_list.read_text(encoding="utf-8").splitlines() if x.strip()]

    gt_masks: Dict[str, np.ndarray] = {}
    for s in stems:
        gt_path = mask_dir / f"{s}.png"
        gt = np.asarray(Image.open(gt_path))
        gt_masks[s] = normalize_binary_mask(gt)

    loaded_models = {cfg.name: load_model(cfg) for cfg in model_cfgs}

    pred_masks: Dict[str, Dict[str, np.ndarray]] = {cfg.name: {} for cfg in model_cfgs}
    metric_acc: Dict[str, List[Dict[str, float]]] = {cfg.name: [] for cfg in model_cfgs}

    for stem in tqdm(stems, desc="Evaluating samples"):
        image = Image.open(find_image(image_dir, stem)).convert("RGB")
        gt = gt_masks[stem]

        for cfg in model_cfgs:
            pred = infer_binary_mask(loaded_models[cfg.name], cfg, image)
            pred_masks[cfg.name][stem] = pred
            metric_acc[cfg.name].append(calc_stats(pred, gt))

    summary_rows = []
    for cfg in model_cfgs:
        arr = metric_acc[cfg.name]
        row = {
            "model": cfg.name,
            "miou": float(np.mean([x["miou"] for x in arr])),
            "iou_fg": float(np.mean([x["iou_fg"] for x in arr])),
            "iou_bg": float(np.mean([x["iou_bg"] for x in arr])),
            "dice": float(np.mean([x["dice"] for x in arr])),
            "precision": float(np.mean([x["precision"] for x in arr])),
            "recall": float(np.mean([x["recall"] for x in arr])),
            "f1": float(np.mean([x["f1"] for x in arr])),
        }
        summary_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(args.output_dir / "metrics_summary.csv", summary_rows)
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    vis_stems = stems[: max(0, args.max_vis)]
    save_qualitative_grid(
        output_path=args.output_dir / "qualitative_grid.png",
        image_stems=vis_stems,
        image_dir=image_dir,
        gt_masks=gt_masks,
        pred_masks=pred_masks,
        model_names=[cfg.name for cfg in model_cfgs],
        rows=args.vis_cols,
        vis_mode=args.vis_mode,
    )

    print("\n=== Done ===")
    print(f"Saved metrics: {args.output_dir / 'metrics_summary.csv'}")
    print(f"Saved figure : {args.output_dir / 'qualitative_grid.png'}")


if __name__ == "__main__":
    main()