#!/usr/bin/env python3
"""Sanity check RGB-mask-JSON dataset pairing before training."""

import argparse
import json
from pathlib import Path

from PIL import Image

from rehc_former.data import discover_samples, infer_image_size_from_samples


def parse_args():
    parser = argparse.ArgumentParser(description="检查 RGB + mask + JSON 数据集是否能被训练脚本读取")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--label_mode", type=str, default="tec", choices=["tec", "tce", "auto"])
    parser.add_argument("--show_first", type=int, default=5)
    parser.add_argument("--allow_missing_mask", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    samples = discover_samples(
        data_dir=args.data_dir,
        label_mode=args.label_mode,
        cache_path=None,
        require_mask=not args.allow_missing_mask,
    )
    image_size = infer_image_size_from_samples(samples)
    num_with_mask = sum(1 for s in samples if s.mask_path is not None)
    phase_counts = {}
    unit_counts = {}
    pose_counts = {}
    for s in samples:
        phase_counts[s.phase] = phase_counts.get(s.phase, 0) + 1
        unit_counts[s.translation_unit] = unit_counts.get(s.translation_unit, 0) + 1
        pose_counts[s.pose_name] = pose_counts.get(s.pose_name, 0) + 1

    summary = {
        "num_samples": len(samples),
        "num_with_mask": num_with_mask,
        "inferred_image_size_h_w": list(image_size),
        "phase_counts": phase_counts,
        "translation_unit_counts": unit_counts,
        "pose_name_counts": pose_counts,
        "first_samples": [],
    }

    for s in samples[: max(0, args.show_first)]:
        item = {
            "image": str(s.image_path),
            "mask": None if s.mask_path is None else str(s.mask_path),
            "json": str(s.json_path),
            "translation": s.translation.tolist(),
            "quaternion_xyzw": s.quaternion_xyzw.tolist(),
            "pose_name": s.pose_name,
            "translation_unit": s.translation_unit,
            "phase": s.phase,
        }
        try:
            with Image.open(s.image_path) as im:
                item["image_size_w_h"] = list(im.size)
        except OSError:
            item["image_size_w_h"] = None
        if s.mask_path is not None:
            try:
                with Image.open(s.mask_path) as im:
                    item["mask_size_w_h"] = list(im.size)
            except OSError:
                item["mask_size_w_h"] = None
        summary["first_samples"].append(item)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
