from __future__ import annotations

import argparse
import json
from pathlib import Path
from copy import deepcopy
from typing import Any

import torch

from compare_testset_models import ModelConfig, import_model_module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--search-root", type=Path, required=True)
    p.add_argument("--glob", type=str, default="**/*.pth")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument(
        "--probe-fastsegformer",
        action="store_true",
        help="Probe common FastSegFormer kwargs (Pyramid/cnn_branch) when no exact match is found.",
    )
    return p.parse_args()


def read_model_cfg(config_path: Path, model_name: str) -> ModelConfig:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    for item in data["models"]:
        if item["name"] != model_name:
            continue
        return ModelConfig(
            name=item["name"],
            module=item["module"],
            class_name=item["class_name"],
            checkpoint=item.get("checkpoint", ""),
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
            device=item.get("device", "cpu"),
            config_dir=str(config_path.parent.resolve()),
            pythonpath=item.get("pythonpath", []),
        )
    raise ValueError(f"model '{model_name}' not found in {config_path}")


def extract_state_dict(ckpt: Any, preferred: str | None) -> dict[str, Any] | None:
    if isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in list(ckpt.values())[:20]):
        return ckpt
    if not isinstance(ckpt, dict):
        return None
    if preferred and preferred in ckpt and isinstance(ckpt[preferred], dict):
        return ckpt[preferred]
    for k in ["state_dict", "model", "model_state", "model_state_dict", "net", "weights"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    return None


def normalize_keys(sd: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in sd.items():
        nk = str(k)
        if nk.startswith("module."):
            nk = nk.replace("module.", "", 1)
        if nk.startswith("model."):
            nk = nk.replace("model.", "", 1)
        out[nk] = v
    return out


def resolve_ctor(cfg: ModelConfig):
    module = import_model_module(cfg.module, cfg.module_file, cfg.config_dir, cfg.pythonpath)
    if hasattr(module, "model_dict") and cfg.class_name in module.model_dict:
        return module.model_dict[cfg.class_name]
    if hasattr(module, cfg.class_name):
        return getattr(module, cfg.class_name)
    raise AttributeError(f"Cannot resolve constructor for class_name={cfg.class_name}")


def main() -> None:
    args = parse_args()
    cfg = read_model_cfg(args.config, args.model_name)

    ctor = resolve_ctor(cfg)
    model = ctor(**cfg.init_kwargs)
    model_sd = model.state_dict()

    ckpts = sorted(args.search_root.glob(args.glob))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found under {args.search_root} with glob {args.glob}")

    scored = []
    exact = []
    for path in ckpts:
        try:
            raw = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        sd = extract_state_dict(raw, cfg.ckpt_state_dict_key)
        if not sd:
            continue
        sd = normalize_keys(sd)

        overlap = 0
        shape_ok = 0
        shape_mismatch = 0
        for k, v in sd.items():
            if k not in model_sd:
                continue
            overlap += 1
            if tuple(v.shape) == tuple(model_sd[k].shape):
                shape_ok += 1
            else:
                shape_mismatch += 1

        score = shape_ok / max(overlap, 1)
        scored.append((score, shape_ok, overlap, shape_mismatch, str(path)))
        if overlap > 0 and shape_mismatch == 0:
            exact.append(str(path))

    print("\n=== Exact architecture matches (no shape mismatch) ===")
    if exact:
        for p in exact:
            print(p)
    else:
        print("None")

    scored.sort(key=lambda x: (x[0], x[1], -x[3]), reverse=True)
    print(f"\n=== Top {args.topk} closest checkpoints ===")
    for row in scored[: args.topk]:
        score, ok, overlap, bad, p = row
        print(f"score={score:.4f} ok={ok} overlap={overlap} mismatch={bad} :: {p}")

    if not exact:
        print("\nNo exact match found. This usually means checkpoint and current code version differ.")
        print("Use one of the top checkpoints with mismatch=0, or switch to the original training code version.")
        if args.probe_fastsegformer and "fastsegformer" in cfg.class_name.lower():
            print("\n=== FastSegFormer kwarg probe ===")
            probe_fastsegformer(cfg, ckpts)


def compare_state_dict_shapes(model_sd: dict[str, Any], sd: dict[str, Any]) -> tuple[int, int, int]:
    overlap = 0
    shape_ok = 0
    shape_mismatch = 0
    for k, v in sd.items():
        if k not in model_sd:
            continue
        overlap += 1
        if tuple(v.shape) == tuple(model_sd[k].shape):
            shape_ok += 1
        else:
            shape_mismatch += 1
    return overlap, shape_ok, shape_mismatch


def probe_fastsegformer(cfg: ModelConfig, ckpts: list[Path]) -> None:
    variants = []
    base = deepcopy(cfg.init_kwargs)
    for pyramid in ["multiscale", "pooling"]:
        for cnn_branch in [True, False]:
            kwargs = deepcopy(base)
            kwargs["Pyramid"] = pyramid
            kwargs["cnn_branch"] = cnn_branch
            variants.append(kwargs)

    rows: list[tuple[float, int, int, int, dict[str, Any], str]] = []
    for kwargs in variants:
        probe_cfg = deepcopy(cfg)
        probe_cfg.init_kwargs = kwargs
        try:
            ctor = resolve_ctor(probe_cfg)
            model = ctor(**probe_cfg.init_kwargs)
            model_sd = model.state_dict()
        except Exception as e:
            rows.append((0.0, 0, 0, 10**9, kwargs, f"build_error={e}"))
            continue

        best = (0.0, 0, 0, 10**9, "")
        for path in ckpts:
            try:
                raw = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                continue
            sd = extract_state_dict(raw, probe_cfg.ckpt_state_dict_key)
            if not sd:
                continue
            sd = normalize_keys(sd)
            overlap, shape_ok, shape_mismatch = compare_state_dict_shapes(model_sd, sd)
            score = shape_ok / max(overlap, 1)
            if (score, shape_ok, -shape_mismatch) > (best[0], best[1], -best[3]):
                best = (score, shape_ok, overlap, shape_mismatch, str(path))
        rows.append((best[0], best[1], best[2], best[3], kwargs, best[4]))

    rows.sort(key=lambda x: (x[0], x[1], -x[3]), reverse=True)
    for score, ok, overlap, bad, kwargs, path in rows:
        print(
            f"score={score:.4f} ok={ok} overlap={overlap} mismatch={bad} "
            f"kwargs={kwargs} :: {path}"
        )


if __name__ == "__main__":
    main()