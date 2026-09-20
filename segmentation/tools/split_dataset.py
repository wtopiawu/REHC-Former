from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("data/gripper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split paired images and masks into train/val/test folders.")
    parser.add_argument("--images", type=str, required=True, help="Raw image directory.")
    parser.add_argument("--masks", type=str, required=True, help="Raw mask directory.")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help=f"Output dataset root. Default: {DEFAULT_OUT}")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["copy", "link"], default="copy", help="copy files or create symlinks.")
    return parser.parse_args()


def list_files(root: Path, exts: set[str]) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def build_pairs(image_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path, Path]]:
    masks = list_files(mask_dir, MASK_EXTS)
    mask_map: Dict[str, Path] = {str(m.relative_to(mask_dir).with_suffix("")): m for m in masks}
    pairs: List[Tuple[Path, Path, Path]] = []
    missing_masks: List[str] = []
    for img in list_files(image_dir, IMAGE_EXTS):
        rel_no_ext = img.relative_to(image_dir).with_suffix("")
        mask = mask_map.get(str(rel_no_ext))
        if mask is not None:
            pairs.append((img, mask, rel_no_ext))
        else:
            missing_masks.append(str(rel_no_ext))
    if not pairs:
        image_count = len(list_files(image_dir, IMAGE_EXTS))
        mask_count = len(masks)
        image_examples = [p.name for p in list_files(image_dir, IMAGE_EXTS)[:5]]
        mask_examples = [p.name for p in masks[:5]]
        raise RuntimeError(
            "No paired files found. Image and mask relative stems must match.\n"
            f"image_dir={image_dir} ({image_count} image files)\n"
            f"mask_dir={mask_dir} ({mask_count} mask files)\n"
            f"image examples={image_examples}\n"
            f"mask examples={mask_examples}"
        )
    if missing_masks:
        print(f"[WARN] {len(missing_masks)} image(s) have no matching mask. First missing: {missing_masks[:10]}")
    return pairs


def put_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main() -> None:
    args = parse_args()
    ratios = [args.train_ratio, args.val_ratio, args.test_ratio]
    if any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must be non-negative and sum to 1.0")

    image_dir = Path(args.images)
    mask_dir = Path(args.masks)
    out_root = Path(args.out)
    if out_root.exists() and any(out_root.iterdir()):
        raise ValueError("Output must be new or empty; existing splits are never deleted or mixed.")
    pairs = build_pairs(image_dir, mask_dir)

    random.seed(args.seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }

    for split, items in splits.items():
        for img, mask, rel_no_ext in items:
            img_dst = out_root / "images" / split / rel_no_ext.with_suffix(img.suffix)
            mask_dst = out_root / "masks" / split / rel_no_ext.with_suffix(mask.suffix)
            put_file(img, img_dst, args.mode)
            put_file(mask, mask_dst, args.mode)
        print(f"{split}: {len(items)} pairs")

    print(f"Dataset written to: {out_root}")
    print("Expected training config paths:")
    print(f"  train_images: {out_root / 'images' / 'train'}")
    print(f"  train_masks:  {out_root / 'masks' / 'train'}")
    print(f"  val_images:   {out_root / 'images' / 'val'}")
    print(f"  val_masks:    {out_root / 'masks' / 'val'}")


if __name__ == "__main__":
    main()
