"""Evaluate labeled simulated RGB/mask pairs with one frozen checkpoint."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rehc_former.data import discover_samples, EyeInHandPoseMaskDataset
from rehc_former.factory import load_model, mask_options
from rehc_former.geometry import convert_length_array, pose_to_T_EC_transform


def metrics(translation_mm, rotation_deg):
    t, r = np.asarray(translation_mm), np.asarray(rotation_deg)
    def stats(values):
        return {"mean": float(values.mean()), "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95, method="linear"))}
    return {"translation_mm": stats(t), "rotation_deg": stats(r),
            "joint_success_5mm_2deg_pct": float(100 * ((t <= 5) & (r <= 2)).mean()),
            "joint_success_10mm_5deg_pct": float(100 * ((t <= 10) & (r <= 5)).mean())}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--points", type=Path, help="Optional N x 3 .npy evaluation points in camera frame C, in mm")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    mode = checkpoint.get("label_mode", "tec")
    samples = discover_samples(str(args.data_dir), label_mode=mode, require_mask=True)
    if len({s.pose_name for s in samples}) != 1:
        raise ValueError("Mixed pose directions in evaluation data")
    unit = checkpoint["translation_unit"]
    for sample in samples:
        sample.translation = convert_length_array(sample.translation, sample.translation_unit, unit)
        sample.translation_unit = unit
    mean = np.asarray(checkpoint["translation_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["translation_std"], dtype=np.float32)
    options = mask_options(checkpoint)
    dataset = EyeInHandPoseMaskDataset(samples, mean, std,
        image_size=tuple(checkpoint.get("image_size", checkpoint["model_config"]["image_size"])),
        mask_threshold=options["threshold"], mask_invert=options["invert"], mask_binary=options["binary"])
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    points = np.load(args.points, allow_pickle=False) if args.points else None
    if points is not None and (points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all()):
        raise ValueError("points must contain a finite, nonempty N x 3 array")
    mean_t, std_t = torch.tensor(mean, device=device), torch.tensor(std, device=device)
    rows = []
    for batch in tqdm(loader, desc="Evaluate"):
        out = model(batch["image"].to(device), batch["mask"].to(device))
        pred_t = (out["pred_translation_norm"] * std_t + mean_t).cpu().numpy()
        pred_r = out["pred_rotation_matrix"].cpu().numpy()
        for i in range(len(pred_t)):
            sample = samples[len(rows)]
            pred, gt = np.eye(4), np.eye(4)
            pred[:3, :3], pred[:3, 3] = pred_r[i], pred_t[i]
            gt[:3, :3], gt[:3, 3] = sample.rotation_matrix, sample.translation
            pred = pose_to_T_EC_transform(pred, mode, sample.pose_name).astype(np.float64)
            gt = pose_to_T_EC_transform(gt, mode, sample.pose_name).astype(np.float64)
            pred[:3, 3] = convert_length_array(pred[:3, 3], unit, "mm")
            gt[:3, 3] = convert_length_array(gt[:3, 3], unit, "mm")
            angle = np.degrees(np.arccos(np.clip((np.trace(pred[:3, :3] @ gt[:3, :3].T) - 1) / 2, -1, 1)))
            row = {"sample": sample.json_path.relative_to(args.data_dir.resolve()).as_posix(),
                   "extrinsic_group_id": sample.extrinsic_group_id,
                   "translation_mm": float(np.linalg.norm(pred[:3, 3] - gt[:3, 3])),
                   "rotation_deg": float(angle), "pred_T_EC_mm": json.dumps(pred.tolist())}
            if points is not None:
                delta = (points @ pred[:3, :3].T + pred[:3, 3]) - (points @ gt[:3, :3].T + gt[:3, 3])
                row["ead_mm"] = float(np.linalg.norm(delta, axis=1).mean())
            rows.append(row)
    groups = {}
    for row in rows:
        groups.setdefault(row["extrinsic_group_id"], []).append(row)
    group_rows = [{"extrinsic_group_id": group, "num_images": len(items),
                   "mean_translation_mm": float(np.mean([x["translation_mm"] for x in items])),
                   "mean_rotation_deg": float(np.mean([x["rotation_deg"] for x in items]))}
                  for group, items in sorted(groups.items())]
    summary = {"num_images": len(rows), "num_extrinsic_groups": len(groups),
               "aggregation": "per-image; group means are separately exported",
               **metrics([x["translation_mm"] for x in rows], [x["rotation_deg"] for x in rows])}
    summary["ead_mm_mean"] = float(np.mean([x["ead_mm"] for x in rows])) if points is not None else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in [("predictions.csv", rows), ("groups.csv", group_rows)]:
        with (args.output_dir/name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
