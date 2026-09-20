from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("data/gripper")
SPLITS = ("train", "val", "test")


def list_files(root: Path, exts: set[str]):
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def check_pair_dirs(img_root: Path, mask_root: Path, binary: bool, label: str) -> bool:
    masks = {str(p.relative_to(mask_root).with_suffix("")): p for p in list_files(mask_root, MASK_EXTS)}
    images = list_files(img_root, IMAGE_EXTS)

    paired = 0
    missing = []
    fg_ratios = []
    shape_mismatch = []
    bad_masks = []

    for img_path in images:
        rel = str(img_path.relative_to(img_root).with_suffix(""))
        mask_path = masks.get(rel)
        if mask_path is None:
            missing.append(rel)
            continue
        paired += 1
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            bad_masks.append(rel)
            continue
        if img.shape[:2] != mask.shape[:2]:
            shape_mismatch.append((rel, img.shape[:2], mask.shape[:2]))
        if binary:
            fg = mask > 0
        else:
            vals = np.unique(mask)
            if vals.max(initial=0) > 1:
                bad_masks.append(f"{rel}: unique={vals[:10]}")
            fg = mask == 1
        fg_ratios.append(float(fg.mean()))

    print(f"[{label}]")
    print(f"Images: {len(images)}")
    print(f"Masks: {len(masks)}")
    print(f"Paired: {paired}")
    print(f"Missing masks: {len(missing)}")
    print(f"Shape mismatches: {len(shape_mismatch)}")
    print(f"Potential bad masks: {len(bad_masks)}")
    if fg_ratios:
        arr = np.array(fg_ratios)
        print(f"Foreground ratio: mean={arr.mean():.6f}, min={arr.min():.6f}, max={arr.max():.6f}")
    if missing[:10]:
        print("First missing:", missing[:10])
    if shape_mismatch[:5]:
        print("First shape mismatches:", shape_mismatch[:5])
    if bad_masks[:5]:
        print("First bad masks:", bad_masks[:5])
    return not missing and not shape_mismatch and not bad_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check image/mask pairing and basic mask statistics.")
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT), help=f"Dataset root. Default: {DEFAULT_ROOT}")
    parser.add_argument("--split", choices=[*SPLITS, "all"], default="all", help="Split to check. Default: all")
    parser.add_argument("--images", type=str, default="", help="Optional image directory override.")
    parser.add_argument("--masks", type=str, default="", help="Optional mask directory override.")
    parser.add_argument("--binary", dest="binary", action="store_true", default=True, help="Treat mask > 0 as foreground. Default.")
    parser.add_argument("--multi-class", dest="binary", action="store_false", help="Check masks as class-id masks instead of binary masks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if bool(args.images) != bool(args.masks):
        raise ValueError("--images and --masks must be used together.")

    ok = True
    if args.images and args.masks:
        ok = check_pair_dirs(Path(args.images), Path(args.masks), args.binary, "custom")
    else:
        root = Path(args.root)
        splits = SPLITS if args.split == "all" else (args.split,)
        for i, split in enumerate(splits):
            if i:
                print()
            img_root = root / "images" / split
            mask_root = root / "masks" / split
            ok = check_pair_dirs(img_root, mask_root, args.binary, split) and ok

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
