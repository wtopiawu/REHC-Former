#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import build_datasets
from .losses import PoseLoss
from .factory import VARIANTS, build_model
from .geometry import geodesic_distance_from_two_matrices


SCRIPT_DIR = Path.cwd()
PLOT_SIZE = (1200, 700)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


def log_stage(stage: str, start_time: float) -> None:
    elapsed = time.time() - start_time
    print(f"[TIME] {stage}: {elapsed:.2f}s")


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool, prefetch_factor: int) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    epoch: int,
    epochs: int,
    amp: bool,
    grad_clip: float,
):
    model.train()
    meters = {k: AverageMeter() for k in ["loss", "loss_t", "loss_r", "loss_r_geo"]}
    pbar = tqdm(loader, desc=f"Train {epoch}/{epochs}", ncols=120)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        gt_t_norm = batch["translation_norm"].to(device, non_blocking=True)
        gt_R = batch["rotation_matrix"].to(device, non_blocking=True)
        gt_rotation_9d = batch["rotation_9d"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            outputs = model(images, masks)
            loss_dict = criterion(
                pred_translation_norm=outputs["pred_translation_norm"],
                gt_translation_norm=gt_t_norm,
                pred_rotation_matrix=outputs["pred_rotation_matrix"],
                gt_rotation_matrix=gt_R,
                pred_rotation_repr=outputs.get("pred_rotation_9d"),
                gt_rotation_repr=gt_rotation_9d if "pred_rotation_9d" in outputs else None,
            )
            loss = loss_dict["loss"]

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        for k in meters:
            meters[k].update(loss_dict[k].detach().item(), bs)
        pbar.set_postfix(
            loss=f"{meters['loss'].avg:.4f}",
            loss_t=f"{meters['loss_t'].avg:.4f}",
            loss_r=f"{meters['loss_r'].avg:.4f}",
            loss_r_geo=f"{meters['loss_r_geo'].avg:.4f}",
        )

    return {k: v.avg for k, v in meters.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, amp: bool, translation_mean: torch.Tensor, translation_std: torch.Tensor):
    model.eval()
    meters = {
        k: AverageMeter()
        for k in ["loss", "loss_t", "loss_r", "loss_r_geo", "mae_tx", "mae_ty", "mae_tz", "trans_l2", "rot_deg"]
    }

    for batch in tqdm(loader, desc="Val", ncols=120):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        gt_t = batch["translation"].to(device, non_blocking=True)
        gt_t_norm = batch["translation_norm"].to(device, non_blocking=True)
        gt_R = batch["rotation_matrix"].to(device, non_blocking=True)
        gt_rotation_9d = batch["rotation_9d"].to(device, non_blocking=True)

        with autocast(enabled=amp):
            outputs = model(images, masks)
            loss_dict = criterion(
                pred_translation_norm=outputs["pred_translation_norm"],
                gt_translation_norm=gt_t_norm,
                pred_rotation_matrix=outputs["pred_rotation_matrix"],
                gt_rotation_matrix=gt_R,
                pred_rotation_repr=outputs.get("pred_rotation_9d"),
                gt_rotation_repr=gt_rotation_9d if "pred_rotation_9d" in outputs else None,
            )

        pred_t = outputs["pred_translation_norm"] * translation_std + translation_mean
        trans_abs_per_sample = (pred_t - gt_t).abs()
        trans_abs = trans_abs_per_sample.mean(dim=0)
        trans_l2 = torch.linalg.norm(pred_t - gt_t, dim=1).mean()
        rot_deg = torch.rad2deg(geodesic_distance_from_two_matrices(outputs["pred_rotation_matrix"], gt_R)).mean()

        bs = images.size(0)
        for k in ["loss", "loss_t", "loss_r", "loss_r_geo"]:
            meters[k].update(loss_dict[k].item(), bs)
        meters["mae_tx"].update(trans_abs[0].item(), bs)
        meters["mae_ty"].update(trans_abs[1].item(), bs)
        meters["mae_tz"].update(trans_abs[2].item(), bs)
        meters["trans_l2"].update(trans_l2.item(), bs)
        meters["rot_deg"].update(rot_deg.item(), bs)

    return {k: v.avg for k, v in meters.items()}


def save_checkpoint(save_path: Path, payload: Dict) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, save_path)


def save_split_manifest(train_dataset, val_dataset, output_path: Path) -> None:
    """Persist the exact group-aware split used by this run."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "extrinsic_group_id", "global_index", "image_path", "mask_path", "json_path"],
        )
        writer.writeheader()
        for split_name, dataset in (("train", train_dataset), ("val", val_dataset)):
            for sample in dataset.samples:
                writer.writerow({
                    "split": split_name,
                    "extrinsic_group_id": sample.extrinsic_group_id,
                    "global_index": sample.global_index,
                    "image_path": str(sample.image_path),
                    "mask_path": "" if sample.mask_path is None else str(sample.mask_path),
                    "json_path": str(sample.json_path),
                })


def _collect_series(records, key: str):
    xs = []
    ys = []
    for item in records:
        value = item.get(key)
        if value is None:
            continue
        try:
            y = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(y) or math.isinf(y):
            continue
        xs.append(int(item["epoch"]))
        ys.append(y)
    return xs, ys


def _draw_line_chart(records, series_keys, colors, title: str, y_label: str, output_path: Path) -> None:
    from PIL import Image, ImageDraw

    width, height = PLOT_SIZE
    margin_left, margin_right = 90, 30
    margin_top, margin_bottom = 70, 80
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    xs_all = []
    ys_all = []
    series_data = []
    for label, key in series_keys:
        xs, ys = _collect_series(records, key)
        series_data.append((label, xs, ys))
        xs_all.extend(xs)
        ys_all.extend(ys)

    if not xs_all or not ys_all:
        draw.text((plot_left, plot_top), "No valid metrics to plot yet.", fill="black")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return

    x_min, x_max = min(xs_all), max(xs_all)
    y_min, y_max = min(ys_all), max(ys_all)
    if math.isclose(y_min, y_max):
        pad = max(1e-6, abs(y_min) * 0.1 + 1e-3)
        y_min -= pad
        y_max += pad

    def map_x(epoch: int) -> float:
        if x_max == x_min:
            return plot_left + plot_w / 2
        return plot_left + (epoch - x_min) / (x_max - x_min) * plot_w

    def map_y(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * plot_h

    grid_steps = 5
    axis_color = (40, 40, 40)
    grid_color = (225, 225, 225)
    for i in range(grid_steps + 1):
        y = plot_top + plot_h * i / grid_steps
        draw.line((plot_left, y, plot_right, y), fill=grid_color, width=1)
        value = y_max - (y_max - y_min) * i / grid_steps
        draw.text((10, y - 8), f"{value:.4f}", fill=axis_color)

    for i in range(grid_steps + 1):
        x = plot_left + plot_w * i / grid_steps
        draw.line((x, plot_top, x, plot_bottom), fill=grid_color, width=1)
        epoch = x_min + (x_max - x_min) * i / grid_steps if x_max != x_min else x_min
        draw.text((x - 10, plot_bottom + 10), str(int(round(epoch))), fill=axis_color)

    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=axis_color, width=2)
    draw.text((plot_left, 20), title, fill=axis_color)
    draw.text((plot_left, height - 45), "epoch", fill=axis_color)
    draw.text((15, plot_top - 20), y_label, fill=axis_color)

    legend_x = plot_right - 260
    legend_y = 25
    for idx, (label, xs, ys) in enumerate(series_data):
        color = colors[idx % len(colors)]
        if len(xs) >= 2:
            points = [(map_x(x), map_y(y)) for x, y in zip(xs, ys)]
            draw.line(points, fill=color, width=3)
        for x, y in zip(xs, ys):
            draw.ellipse((map_x(x) - 3, map_y(y) - 3, map_x(x) + 3, map_y(y) + 3), fill=color, outline=color)
        draw.rectangle((legend_x, legend_y + idx * 24, legend_x + 14, legend_y + 14 + idx * 24), fill=color, outline=color)
        draw.text((legend_x + 20, legend_y + idx * 24), label, fill=axis_color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def export_training_artifacts(records, output_dir: Path) -> None:
    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for record in records for k in record.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    _draw_line_chart(
        records,
        series_keys=[("train loss", "train_loss"), ("val loss", "val_loss")],
        colors=[(31, 119, 180), (214, 39, 40)],
        title="Training / Validation Loss",
        y_label="loss",
        output_path=output_dir / "loss_curve.png",
    )

    _draw_line_chart(
        records,
        series_keys=[
            ("train loss_t", "train_loss_t"),
            ("val loss_t", "val_loss_t"),
            ("train loss_r", "train_loss_r"),
            ("val loss_r", "val_loss_r"),
        ],
        colors=[(44, 160, 44), (148, 103, 189), (255, 127, 14), (23, 190, 207)],
        title="Translation / Rotation Loss",
        y_label="loss",
        output_path=output_dir / "loss_components.png",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train REHC-Former or a controlled RGB/Mask ablation")
    parser.add_argument("--variant", default="rehc_former", choices=VARIANTS, help="Full model or input/fusion ablation")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="数据集根目录；建议结构为 data/rgb/*.png, data/mask/*.png, data/anno/*.json，或 JSON image 字段指向 rgb。",
    )
    parser.add_argument("--output_dir", type=str, default="", help="留空时输出到 runs/<variant>_seed<seed>")
    parser.add_argument(
        "--label_mode",
        type=str,
        default="tec",
        choices=["tec", "t_ec", "eye_in_hand", "camera_to_ee", "tce", "t_ce", "ee_to_camera", "auto"],
        help="默认 tec，即训练 T_EC: camera frame -> end-effector frame。",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader 每个 worker 预取批次数")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_h", type=int, default=0, help="目标图像高；设为 0 时从数据集中自动推断")
    parser.add_argument("--img_w", type=int, default=0, help="目标图像宽；设为 0 时从数据集中自动推断")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2, help="self-attention 层数；应与完整模型 depth 一致")
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--patch_stride", type=int, default=2, help="token 下采样步长；越大 token 越少、显存越省")
    parser.add_argument("--mlp_ratio", type=float, default=2.0, help="attention block 内 MLP 扩展比例")
    parser.add_argument("--stem_width", type=int, default=32, help="lite_cnn 和 mask stem 的基础通道数")
    parser.add_argument(
        "--rotation_repr",
        type=str,
        default="9d",
        choices=["6d", "9d"],
        help="旋转头输出和监督表示；默认 9d 直接监督 3x3 旋转矩阵展平标签",
    )
    parser.add_argument(
        "--backbone_name",
        type=str,
        default="lite_cnn",
        choices=["lite_cnn", "resnet18", "convnext_tiny"],
        help="Paper backbone: lite_cnn. Other backbones are optional extensions.",
    )
    parser.add_argument("--backbone_pretrained", dest="backbone_pretrained", action="store_true")
    parser.add_argument("--no_backbone_pretrained", dest="backbone_pretrained", action="store_false")
    parser.set_defaults(backbone_pretrained=False)
    parser.add_argument("--translation_weight", type=float, default=1.0)
    parser.add_argument("--rotation_weight", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="开启混合精度训练")
    parser.add_argument("--dataset_cache", type=str, default="", help="样本缓存文件路径，留空则写入 data 目录")
    parser.add_argument(
        "--allow_missing_mask",
        action="store_true",
        help="允许缺失 mask；正式消融不要开启，因为四个模型必须使用完全相同的样本集合",
    )
    parser.add_argument("--mask_threshold", type=float, default=0.5, help="mask 二值化阈值，输入范围 0~1")
    parser.add_argument("--mask_invert", action="store_true", help="如果你的 mask 是背景白、夹爪黑，打开此项")
    parser.add_argument("--soft_mask", action="store_true", help="不做二值化，保留 0~1 soft mask")
    return parser.parse_args(argv)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def main(argv=None, fixed_variant: str = None):
    argv = list(argv) if argv is not None else None
    if fixed_variant is not None:
        import sys
        active_argv = list(sys.argv[1:] if argv is None else argv)
        if "--variant" in active_argv:
            raise ValueError("分类训练入口已经固定 variant，请不要再次传 --variant")
        argv = ["--variant", fixed_variant, *active_argv]
    args = parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    set_seed(args.seed)
    overall_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir or f"runs/{args.variant}_seed{args.seed}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if (args.img_h <= 0) != (args.img_w <= 0):
        raise ValueError("img_h 和 img_w 要么同时设置为正数，要么同时设为 0 以自动推断")
    requested_image_size = None if args.img_h <= 0 else (args.img_h, args.img_w)

    # Use identical paired samples for the full model and all ablations.
    require_mask = not args.allow_missing_mask

    data_build_start = time.time()
    train_dataset, val_dataset, metadata = build_datasets(
        data_dir=str(data_dir),
        label_mode=args.label_mode,
        image_size=requested_image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        cache_path=args.dataset_cache or None,
        require_mask=require_mask,
        mask_threshold=args.mask_threshold,
        mask_invert=args.mask_invert,
        mask_binary=not args.soft_mask,
    )
    log_stage("build_datasets", data_build_start)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    resolved_image_size = tuple(metadata["image_size"])
    save_split_manifest(train_dataset, val_dataset, output_dir / "split_manifest.csv")

    loader_build_start = time.time()
    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers, True, args.prefetch_factor)
    val_loader = make_loader(val_dataset, args.batch_size, args.num_workers, False, args.prefetch_factor)
    log_stage("build_dataloader", loader_build_start)

    model_build_start = time.time()
    model = build_model(
        variant=args.variant,
        image_size=resolved_image_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        dropout=args.dropout,
        patch_stride=args.patch_stride,
        backbone_name=args.backbone_name,
        backbone_pretrained=args.backbone_pretrained,
        mlp_ratio=args.mlp_ratio,
        stem_width=args.stem_width,
        rotation_repr=args.rotation_repr,
    ).to(device)
    log_stage("build_model", model_build_start)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] trainable_params: {trainable_params:,} ({trainable_params / 1e6:.3f} M)")

    criterion = PoseLoss(translation_weight=args.translation_weight, rotation_weight=args.rotation_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    translation_mean = torch.tensor(metadata["translation_mean"], dtype=torch.float32, device=device).view(1, 3)
    translation_std = torch.tensor(metadata["translation_std"], dtype=torch.float32, device=device).view(1, 3)

    train_log_path = output_dir / "train_log.jsonl"
    config_path = output_dir / "config.json"
    config_payload = vars(args).copy()
    config_payload.update(metadata)
    config_payload["trainable_params"] = trainable_params
    config_payload["comparison_contract"] = {
        "same_dataset_required": True,
        "same_split_seed_required": True,
        "only_changed_factor": "input_and_fusion_mechanism",
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False)

    best_val_loss = float("inf")
    best_epoch = -1
    start_time = time.time()
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            epoch,
            args.epochs,
            args.amp and device.type == "cuda",
            args.grad_clip,
        )
        if len(val_dataset) > 0:
            val_metrics = validate(
                model,
                val_loader,
                criterion,
                device,
                args.amp and device.type == "cuda",
                translation_mean,
                translation_std,
            )
        else:
            val_metrics = {
                "loss": float("nan"),
                "loss_t": float("nan"),
                "loss_r": float("nan"),
                "loss_r_geo": float("nan"),
                "mae_tx": float("nan"),
                "mae_ty": float("nan"),
                "mae_tz": float("nan"),
                "trans_l2": float("nan"),
                "rot_deg": float("nan"),
            }

        scheduler.step()

        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        print(json.dumps(record, ensure_ascii=False))
        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        history.append(record)
        export_training_artifacts(history, output_dir)

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "model_config": model.get_init_config(),
            "label_mode": metadata["label_mode"],
            "pose_name": metadata["pose_name"],
            "translation_mean": metadata["translation_mean"],
            "translation_std": metadata["translation_std"],
            "translation_unit": metadata["translation_unit"],
            "image_size": metadata["image_size"],
            "phase_counts": metadata.get("phase_counts", {}),
            "train_args": vars(args),
            "model_variant": args.variant,
            "trainable_params": trainable_params,
        }
        infer_checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "best_val_loss": best_val_loss,
            "model_config": model.get_init_config(),
            "label_mode": metadata["label_mode"],
            "pose_name": metadata["pose_name"],
            "translation_mean": metadata["translation_mean"],
            "translation_std": metadata["translation_std"],
            "translation_unit": metadata["translation_unit"],
            "image_size": metadata["image_size"],
            "phase_counts": metadata.get("phase_counts", {}),
            "train_args": vars(args),
            "model_variant": args.variant,
            "trainable_params": trainable_params,
        }
        save_checkpoint(output_dir / "last_model.pt", checkpoint)
        save_checkpoint(output_dir / "last_infer_only.pt", infer_checkpoint)

        current_val = val_metrics["loss"]
        if not math.isnan(current_val) and current_val < best_val_loss:
            best_val_loss = current_val
            best_epoch = epoch
            checkpoint["best_val_loss"] = best_val_loss
            infer_checkpoint["best_val_loss"] = best_val_loss
            save_checkpoint(output_dir / "best_model.pt", checkpoint)
            save_checkpoint(output_dir / "best_infer_only.pt", infer_checkpoint)

    elapsed = time.time() - start_time
    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "elapsed_seconds": elapsed,
        "startup_seconds": start_time - overall_start,
        "output_dir": str(output_dir),
        "target_pose": metadata["pose_name"],
        "translation_unit": metadata["translation_unit"],
        "input_mode": args.variant,
        "model_variant": args.variant,
        "trainable_params": trainable_params,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    export_training_artifacts(history, output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
