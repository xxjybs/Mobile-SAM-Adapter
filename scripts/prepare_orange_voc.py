#!/usr/bin/env python3
"""
Convert data/orange to VOC-style layout for FastSegFormer.

Expected source layout (default):
  data/orange/
    images/
    masks/
    imageset/(train.txt|val.txt|test.txt optional)

Output layout (default):
  data/orange_voc/
    VOC2007/
      JPEGImages/
      SegmentationClass/
      ImageSets/Segmentation/{train,val,test}.txt
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("data/orange"), help="source orange dataset root")
    p.add_argument("--dst", type=Path, default=Path("data/orange_voc"), help="output VOC root (contains VOC2007)")
    p.add_argument("--images-dir", type=str, default="images", help="relative dir for source images")
    p.add_argument("--masks-dir", type=str, default="masks", help="relative dir for source masks")
    p.add_argument("--imageset-dir", type=str, default="imageset", help="relative dir containing split txt files")
    p.add_argument("--copy", action="store_true", help="copy files instead of symlink")
    p.add_argument("--seed", type=int, default=42, help="seed for random split fallback")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--force-binary-mask", action="store_true", help="force mask values to {0,1}")
    return p.parse_args()


def read_split_file(path: Path) -> List[str]:
    if not path.exists():
        return []
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        x = line.strip()
        if x:
            names.append(Path(x).stem)
    return names


def find_image_path(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def ensure_clean_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, do_copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if do_copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def normalize_mask(mask_path: Path, out_path: Path, force_binary: bool = False) -> Tuple[int, int]:
    """Save mask as single-channel PNG with class ids.

    Returns: (min_val, max_val)
    """
    m = np.array(Image.open(mask_path))
    if m.ndim == 3:
        # Use first channel for RGB masks.
        m = m[..., 0]

    m = m.astype(np.int64)
    min_v, max_v = int(m.min()), int(m.max())

    # Common binary mask format 0/255 -> 0/1.
    if force_binary or (min_v >= 0 and max_v == 255 and set(np.unique(m).tolist()) <= {0, 255}):
        m = (m > 127).astype(np.uint8)
    else:
        m = m.astype(np.uint8)

    Image.fromarray(m, mode="L").save(out_path)
    return int(m.min()), int(m.max())


def fallback_split(stems: List[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    s = train_ratio + val_ratio + test_ratio
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {s}")
    rnd = random.Random(seed)
    items = stems[:]
    rnd.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def write_split(path: Path, names: Iterable[str]) -> None:
    content = "\n".join(names)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()

    src = args.src
    images_dir = src / args.images_dir
    masks_dir = src / args.masks_dir
    imageset_dir = src / args.imageset_dir

    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(f"images/masks dir missing under {src}")

    voc_root = args.dst / "VOC2007"
    jpg_dir = voc_root / "JPEGImages"
    seg_dir = voc_root / "SegmentationClass"
    split_dir = voc_root / "ImageSets" / "Segmentation"
    ensure_clean_dir(jpg_dir)
    ensure_clean_dir(seg_dir)
    ensure_clean_dir(split_dir)

    # Prefer existing split files if present.
    train_names = read_split_file(imageset_dir / "train.txt")
    val_names = read_split_file(imageset_dir / "val.txt")
    test_names = read_split_file(imageset_dir / "test.txt")

    all_stems_from_masks = sorted([p.stem for p in masks_dir.glob("*.png")])
    if not all_stems_from_masks:
        raise RuntimeError(f"No png masks found in {masks_dir}")

    if not (train_names or val_names or test_names):
        train_names, val_names, test_names = fallback_split(
            all_stems_from_masks,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed,
        )
        print("[Info] No split txt found. Generated random train/val/test split.")

    # Keep unique order and only include samples that have both image+mask.
    merged = []
    seen = set()
    for n in train_names + val_names + test_names:
        if n not in seen:
            merged.append(n)
            seen.add(n)

    valid_names: List[str] = []
    dropped = 0
    min_vals, max_vals = [], []

    for stem in merged:
        img_path = find_image_path(images_dir, stem)
        mask_path = masks_dir / f"{stem}.png"
        if img_path is None or not mask_path.exists():
            dropped += 1
            continue

        img_dst = jpg_dir / f"{stem}.jpg"
        if img_path.suffix.lower() != ".jpg":
            # Always store jpg in JPEGImages for compatibility.
            img = Image.open(img_path).convert("RGB")
            img.save(img_dst, quality=95)
        else:
            link_or_copy(img_path, img_dst, do_copy=args.copy)

        mask_dst = seg_dir / f"{stem}.png"
        mn, mx = normalize_mask(mask_path, mask_dst, force_binary=args.force_binary_mask)
        min_vals.append(mn)
        max_vals.append(mx)

        valid_names.append(stem)

    valid_set = set(valid_names)
    train_names = [x for x in train_names if x in valid_set]
    val_names = [x for x in val_names if x in valid_set]
    test_names = [x for x in test_names if x in valid_set]

    # If split txt existed but became empty due filtering, fallback from valid names.
    if not train_names and not val_names and not test_names:
        train_names, val_names, test_names = fallback_split(
            valid_names,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed,
        )

    write_split(split_dir / "train.txt", train_names)
    write_split(split_dir / "val.txt", val_names)
    write_split(split_dir / "test.txt", test_names)

    if not valid_names:
        raise RuntimeError("No valid image-mask pairs were found.")

    print("\n=== Conversion done ===")
    print(f"Source        : {src}")
    print(f"VOC root      : {voc_root}")
    print(f"Total valid   : {len(valid_names)}")
    print(f"Dropped pairs : {dropped}")
    print(f"train/val/test: {len(train_names)}/{len(val_names)}/{len(test_names)}")
    print(f"Mask value min/max (global): {min(min_vals)}/{max(max_vals)}")
    print("Next: set FastSegFormer train.py VOCdevkit_path to this --dst path.")


if __name__ == "__main__":
    main()