from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from segmentation.src.dataset import SegmentationDataset, build_train_augment
from segmentation.src.losses import SegmentationLoss
from segmentation.src.metrics import SegmentationMetrics
from segmentation.src.model import build_segformer
from segmentation.src.utils import (
    AverageMeter,
    build_warmup_cosine_scheduler,
    copy_file,
    ensure_dir,
    get_device,
    load_checkpoint,
    load_yaml,
    save_checkpoint,
    save_json,
    set_seed,
    worker_init_fn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SegFormer-B1 for binary gripper mask segmentation.")
    parser.add_argument("--config", type=str, default="segmentation/configs/segformer_b1.yaml", help="Path to YAML config.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--dry-run", action="store_true", help="Build model/dataloader and exit without training.")
    return parser.parse_args()


def make_loaders(cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    input_cfg = cfg["input"]
    paths = cfg["paths"]
    train_cfg = cfg["train"]

    train_aug = build_train_augment(cfg)
    train_ds = SegmentationDataset(
        paths["train_images"],
        paths["train_masks"],
        image_size=input_cfg["image_size"],
        mean=input_cfg["normalize_mean"],
        std=input_cfg["normalize_std"],
        augment=train_aug,
        binary_mask=bool(input_cfg.get("binary_mask", True)),
    )
    val_ds = SegmentationDataset(
        paths["val_images"],
        paths["val_masks"],
        image_size=input_cfg["image_size"],
        mean=input_cfg["normalize_mean"],
        std=input_cfg["normalize_std"],
        augment=None,
        binary_mask=bool(input_cfg.get("binary_mask", True)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    return train_loader, val_loader


def forward_logits(model: torch.nn.Module, pixel_values: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    out = model(pixel_values=pixel_values)
    logits = out.logits
    if logits.shape[-2:] != target_size:
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
    return logits


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    cfg: Dict,
) -> float:
    model.train()
    meter = AverageMeter()
    accum_steps = int(cfg["train"].get("gradient_accumulation_steps", 1))
    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    grad_clip = float(cfg["train"].get("grad_clip_norm", 0.0))

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Train epoch {epoch}", dynamic_ncols=True)
    for step, batch in enumerate(pbar, start=1):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with autocast(enabled=amp_enabled):
            logits = forward_logits(model, pixel_values, target_size=labels.shape[-2:])
            loss = criterion(logits, labels) / accum_steps

        scaler.scale(loss).backward()

        if step % accum_steps == 0 or step == len(loader):
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        batch_loss = float(loss.item() * accum_steps)
        meter.update(batch_loss, pixel_values.size(0))
        pbar.set_postfix(loss=f"{meter.avg:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return meter.avg


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    cfg: Dict,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    meter = AverageMeter()
    metrics = SegmentationMetrics(num_classes=int(cfg["model"]["num_classes"]), ignore_index=int(cfg["loss"].get("ignore_index", 255)))
    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"

    pbar = tqdm(loader, desc="Validate", dynamic_ncols=True)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with autocast(enabled=amp_enabled):
            logits = forward_logits(model, pixel_values, target_size=labels.shape[-2:])
            loss = criterion(logits, labels)
        meter.update(float(loss.item()), pixel_values.size(0))
        metrics.update(logits, labels)
        pbar.set_postfix(loss=f"{meter.avg:.4f}")

    return meter.avg, metrics.compute()


def append_history(csv_path: Path, row: Dict[str, float | int]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(args.device)
    print(f"Using device: {device}")

    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    ensure_dir(output_dir / "checkpoints")
    copy_file(args.config, output_dir / "config.yaml")

    train_loader, val_loader = make_loaders(cfg)
    model = build_segformer(cfg["model"]).to(device)

    criterion = SegmentationLoss(
        num_classes=int(cfg["model"]["num_classes"]),
        ce_weight=float(cfg["loss"].get("ce_weight", 0.5)),
        dice_weight=float(cfg["loss"].get("dice_weight", 0.5)),
        class_weights=cfg["loss"].get("class_weights"),
        ignore_index=int(cfg["loss"].get("ignore_index", 255)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.01)),
    )

    epochs = int(cfg["train"]["epochs"])
    steps_per_epoch = max(1, math.ceil(len(train_loader) / int(cfg["train"].get("gradient_accumulation_steps", 1))))
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(cfg["train"].get("warmup_epochs", 0)) * steps_per_epoch
    base_lr = float(cfg["train"]["learning_rate"])
    min_lr = float(cfg["train"].get("min_lr", 1e-6))
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=min_lr / base_lr)
    scaler = GradScaler(enabled=bool(cfg["train"].get("amp", True)) and device.type == "cuda")

    start_epoch = 1
    best_fg_iou = -1.0
    best_epoch = 0
    resume_path = str(cfg["train"].get("resume", "")).strip()
    if resume_path:
        ckpt = load_checkpoint(resume_path, device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_fg_iou = float(ckpt.get("best_fg_iou", best_fg_iou))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    if args.dry_run:
        batch = next(iter(train_loader))
        with torch.no_grad():
            logits = forward_logits(model, batch["pixel_values"].to(device), target_size=batch["labels"].shape[-2:])
        print(f"Dry run OK. Logits shape: {tuple(logits.shape)}")
        return

    history_csv = output_dir / "history.csv"
    patience = int(cfg["train"].get("early_stop_patience", 0))
    save_every = int(cfg["train"].get("save_every", 10))

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch, cfg)
        val_loss, val_metrics = validate(model, val_loader, criterion, device, cfg)
        elapsed = time.time() - t0

        fg_iou = float(val_metrics.get("foreground_iou", val_metrics.get("mIoU", 0.0)))
        is_best = fg_iou > best_fg_iou
        if is_best:
            best_fg_iou = fg_iou
            best_epoch = epoch

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "foreground_iou": fg_iou,
            "foreground_dice": float(val_metrics.get("foreground_dice", 0.0)),
            "mIoU": float(val_metrics.get("mIoU", 0.0)),
            "mDice": float(val_metrics.get("mDice", 0.0)),
            "pixel_acc": float(val_metrics.get("pixel_acc", 0.0)),
            "lr": float(scheduler.get_last_lr()[0]),
            "seconds": round(elapsed, 2),
        }
        append_history(history_csv, row)
        save_json({"best_epoch": best_epoch, "best_fg_iou": best_fg_iou, "last_metrics": row}, output_dir / "summary.json")

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "cfg": cfg,
            "best_fg_iou": best_fg_iou,
            "best_epoch": best_epoch,
        }
        save_checkpoint(state, output_dir / "checkpoints" / "last.pth")
        if is_best:
            save_checkpoint(state, output_dir / "checkpoints" / "best.pth")
            model.save_pretrained(output_dir / "hf_best")
        if save_every > 0 and epoch % save_every == 0:
            save_checkpoint(state, output_dir / "checkpoints" / f"epoch_{epoch:03d}.pth")

        print(
            f"Epoch {epoch:03d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"fg_iou={fg_iou:.4f} | fg_dice={row['foreground_dice']:.4f} | best={best_fg_iou:.4f}@{best_epoch}"
        )

        if patience > 0 and (epoch - best_epoch) >= patience:
            print(f"Early stopping: no foreground IoU improvement for {patience} epochs.")
            break

    print(f"Training done. Best foreground IoU: {best_fg_iou:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
