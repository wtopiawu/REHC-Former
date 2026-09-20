from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    """Cross entropy + soft Dice loss for semantic segmentation."""

    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: Optional[Sequence[float]] = None,
        ignore_index: int = 255,
        dice_include_background: bool = False,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.ignore_index = int(ignore_index)
        self.dice_include_background = bool(dice_include_background)
        self.smooth = float(smooth)
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32)
            if len(weights) != self.num_classes:
                raise ValueError(f"class_weights length {len(weights)} != num_classes {self.num_classes}")
            self.register_buffer("class_weights", weights)
        else:
            self.class_weights = None  # type: ignore[assignment]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
        )
        dice = self._dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice

    def _dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        valid = targets != self.ignore_index
        safe_targets = targets.clone()
        safe_targets[~valid] = 0

        one_hot = F.one_hot(safe_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1).float()
        probs = probs * valid
        one_hot = one_hot * valid

        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dims)
        cardinality = torch.sum(probs + one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        if not self.dice_include_background and self.num_classes > 1:
            dice = dice[1:]
        return 1.0 - dice.mean()
