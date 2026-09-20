#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
NAMED_GROUP_RE = re.compile(r"(?:^|[_-])g(?P<group>\d+)(?:[_-]s\d+)?", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Augment image/mask pairs by adding noise to the original background "
            "and applying synchronized group-level geometric transforms."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N matched images.")
    parser.add_argument(
        "--no-write-masks",
        action="store_true",
        help="Only write images, do not write augmented masks.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--background-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-rotate-bg", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-copy-masks", dest="no_write_masks", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument(
        "--group-size",
        type=int,
        default=15,
        help=(
            "Number of consecutive files treated as one group when filenames do "
            "not contain a g### group id. Use 1 for per-image transforms."
        ),
    )
    parser.add_argument(
        "--max-translate",
        type=float,
        default=20.0,
        help="Max absolute x/y translation in pixels.",
    )
    parser.add_argument("--max-rotate", type=float, default=7.0, help="Max absolute rotation in degrees.")
    parser.add_argument(
        "--scale-range",
        type=float,
        nargs=2,
        default=(0.95, 1.05),
        metavar=("MIN", "MAX"),
        help="Random scale range for synchronized geometric transforms.",
    )
    parser.add_argument(
        "--perspective-jitter",
        type=float,
        default=0.025,
        help="Max corner jitter as a fraction of min(image width, image height).",
    )

    parser.add_argument(
        "--bg-noise-std",
        type=float,
        default=10.0,
        help="Gaussian noise std applied to background pixels.",
    )
    parser.add_argument(
        "--bg-brightness",
        type=float,
        default=8.0,
        help="Max absolute brightness shift applied to background pixels.",
    )
    parser.add_argument(
        "--bg-contrast",
        type=float,
        default=0.08,
        help="Max absolute contrast jitter around 1.0 applied to background pixels.",
    )
    return parser.parse_args()


def list_image_files(root):
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def find_mask(mask_dir, image_dir, image_path):
    try:
        relative = image_path.relative_to(image_dir)
        search_dir = mask_dir / relative.parent
    except ValueError:
        search_dir = mask_dir

    candidates = [
        search_dir / f"{image_path.stem}.png",
        search_dir / f"{image_path.stem}.jpg",
        search_dir / f"{image_path.stem}.jpeg",
        search_dir / f"{image_path.stem}.bmp",
        search_dir / f"{image_path.stem}.webp",
        search_dir / f"{image_path.stem}.tif",
        search_dir / f"{image_path.stem}.tiff",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def read_mask(path, size, threshold):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    width, height = size
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > threshold


def write_image(path, image, jpeg_quality):
    path.parent.mkdir(parents=True, exist_ok=True)
    params = []
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    ok = cv2.imwrite(str(path), image, params)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def get_group_id(image_path, sorted_index, group_size):
    match = NAMED_GROUP_RE.search(image_path.stem)
    if match:
        return f"g{int(match.group('group')):03d}"

    if group_size <= 0:
        raise ValueError("--group-size must be positive")
    return f"seq_{sorted_index // group_size:04d}"


def sample_group_transform(rng, args):
    min_scale, max_scale = args.scale_range
    if min_scale <= 0 or max_scale <= 0 or min_scale > max_scale:
        raise ValueError("--scale-range must be two positive numbers in MIN MAX order")

    return {
        "translate_xy": [
            rng.uniform(-args.max_translate, args.max_translate),
            rng.uniform(-args.max_translate, args.max_translate),
        ],
        "rotation_deg": rng.uniform(-args.max_rotate, args.max_rotate),
        "scale": rng.uniform(min_scale, max_scale),
        "corner_jitter": [
            [rng.uniform(-args.perspective_jitter, args.perspective_jitter),
             rng.uniform(-args.perspective_jitter, args.perspective_jitter)]
            for _ in range(4)
        ],
    }


def build_transform_matrix(width, height, transform):
    tx, ty = transform["translate_xy"]
    angle = transform["rotation_deg"]
    scale = transform["scale"]

    center = np.array([width * 0.5, height * 0.5], dtype=np.float32)
    corners = np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=np.float32,
    )

    theta = np.deg2rad(angle)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    dst = (corners - center) @ rotation.T * scale + center
    dst += np.array([tx, ty], dtype=np.float32)

    jitter_scale = float(min(width, height))
    jitter = np.array(transform["corner_jitter"], dtype=np.float32) * jitter_scale
    dst += jitter

    return cv2.getPerspectiveTransform(corners, dst)


def add_background_noise(image, mask, np_rng, args):
    background = ~mask
    if not np.any(background):
        return image.copy(), {
            "bg_noise_std": 0.0,
            "bg_brightness": 0.0,
            "bg_contrast": 1.0,
        }

    noise_std = max(0.0, float(args.bg_noise_std))
    brightness = np_rng.uniform(-args.bg_brightness, args.bg_brightness)
    contrast = np_rng.uniform(1.0 - args.bg_contrast, 1.0 + args.bg_contrast)

    out = image.astype(np.float32)
    bg_pixels = out[background]
    if noise_std > 0:
        bg_pixels = bg_pixels + np_rng.normal(0.0, noise_std, size=bg_pixels.shape)
    bg_pixels = (bg_pixels - 127.5) * contrast + 127.5 + brightness
    out[background] = bg_pixels

    return np.clip(out, 0, 255).astype(np.uint8), {
        "bg_noise_std": noise_std,
        "bg_brightness": float(brightness),
        "bg_contrast": float(contrast),
    }


def warp_pair(image, mask, matrix):
    height, width = image.shape[:2]
    warped_image = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    warped_mask = cv2.warpPerspective(
        mask.astype(np.uint8) * 255,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped_image, warped_mask


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    image_dir = args.input_root / "images"
    mask_dir = args.input_root / "masks"
    out_image_dir = args.output_root / "images"
    out_mask_dir = args.output_root / "masks"

    image_paths = list_image_files(image_dir)
    matched = []
    missing_masks = []
    for sorted_index, image_path in enumerate(image_paths):
        mask_path = find_mask(mask_dir, image_dir, image_path)
        if mask_path is None:
            missing_masks.append(str(image_path))
            continue
        group_id = get_group_id(image_path, sorted_index, args.group_size)
        matched.append((sorted_index, group_id, image_path, mask_path))

    if args.limit is not None:
        matched = matched[: args.limit]

    args.output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_root / "augmentation_assignments.jsonl"
    group_transforms = {}

    with metadata_path.open("w", encoding="utf-8") as meta:
        for index, (_, group_id, image_path, mask_path) in enumerate(matched, start=1):
            image = read_image(image_path)
            height, width = image.shape[:2]
            mask = read_mask(mask_path, (width, height), args.mask_threshold)

            if group_id not in group_transforms:
                group_transforms[group_id] = sample_group_transform(rng, args)
            transform = group_transforms[group_id]

            matrix = build_transform_matrix(width, height, transform)
            warped_image, out_mask = warp_pair(image, mask, matrix)
            out_image, noise_info = add_background_noise(
                warped_image,
                out_mask > args.mask_threshold,
                np_rng,
                args,
            )

            rel_path = image_path.relative_to(image_dir)
            out_image_path = out_image_dir / rel_path
            write_image(out_image_path, out_image, args.jpeg_quality)

            out_mask_path = None
            if not args.no_write_masks:
                try:
                    mask_rel_path = mask_path.relative_to(mask_dir)
                except ValueError:
                    mask_rel_path = Path(mask_path.name)
                out_mask_path = out_mask_dir / mask_rel_path
                write_image(out_mask_path, out_mask, args.jpeg_quality)

            record = {
                "image": str(out_image_path),
                "mask": str(out_mask_path) if out_mask_path is not None else str(mask_path),
                "source_image": str(image_path),
                "source_mask": str(mask_path),
                "group_id": group_id,
                "transform": transform,
                "noise": noise_info,
            }
            meta.write(json.dumps(record, ensure_ascii=False) + "\n")

            if index % 100 == 0 or index == len(matched):
                print(f"processed {index}/{len(matched)}")

    print(f"done: {len(matched)} images written to {out_image_dir}")
    if not args.no_write_masks:
        print(f"done: {len(matched)} masks written to {out_mask_dir}")
    if missing_masks:
        print(f"warning: skipped {len(missing_masks)} images without masks")
        for path in missing_masks[:20]:
            print(f"  missing mask: {path}")
        if len(missing_masks) > 20:
            print("  ...")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
