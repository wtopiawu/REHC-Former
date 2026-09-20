from __future__ import annotations

from typing import Dict

import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = torch.argmax(logits, dim=1).detach().cpu().long()
        targets = targets.detach().cpu().long()
        valid = targets != self.ignore_index
        preds = preds[valid]
        targets = targets[valid]
        if targets.numel() == 0:
            return
        idx = targets * self.num_classes + preds
        conf = torch.bincount(idx, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        self.confusion += conf.to(torch.float64)

    def compute(self) -> Dict[str, float]:
        cm = self.confusion
        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        denom_iou = tp + fp + fn
        denom_dice = 2 * tp + fp + fn

        iou = torch.where(denom_iou > 0, tp / denom_iou.clamp_min(1.0), torch.full_like(tp, float("nan")))
        dice = torch.where(denom_dice > 0, 2 * tp / denom_dice.clamp_min(1.0), torch.full_like(tp, float("nan")))
        pixel_acc = tp.sum() / cm.sum().clamp_min(1.0)

        out = {
            "pixel_acc": float(pixel_acc.item()),
            "mIoU": float(torch.nanmean(iou).item()),
            "mDice": float(torch.nanmean(dice).item()),
        }
        for c in range(self.num_classes):
            out[f"iou_class_{c}"] = float(iou[c].item()) if not torch.isnan(iou[c]) else float("nan")
            out[f"dice_class_{c}"] = float(dice[c].item()) if not torch.isnan(dice[c]) else float("nan")
        if self.num_classes > 1:
            out["foreground_iou"] = out["iou_class_1"]
            out["foreground_dice"] = out["dice_class_1"]
        return out

    def reset(self) -> None:
        self.confusion.zero_()
