from typing import Dict

import torch
import torch.nn as nn

from .geometry import geodesic_distance_from_two_matrices


class PoseLoss(nn.Module):
    def __init__(self, translation_weight: float = 1.0, rotation_weight: float = 1.0) -> None:
        super().__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.translation_loss = nn.SmoothL1Loss(beta=1.0)
        self.rotation_repr_loss = nn.SmoothL1Loss(beta=1.0)

    def forward(
        self,
        pred_translation_norm: torch.Tensor,
        gt_translation_norm: torch.Tensor,
        pred_rotation_matrix: torch.Tensor,
        gt_rotation_matrix: torch.Tensor,
        pred_rotation_repr: torch.Tensor = None,
        gt_rotation_repr: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        loss_t = self.translation_loss(pred_translation_norm, gt_translation_norm)
        loss_r_geo = geodesic_distance_from_two_matrices(pred_rotation_matrix, gt_rotation_matrix).mean()
        if pred_rotation_repr is not None and gt_rotation_repr is not None:
            loss_r = self.rotation_repr_loss(pred_rotation_repr, gt_rotation_repr)
        else:
            loss_r = loss_r_geo
        total = self.translation_weight * loss_t + self.rotation_weight * loss_r
        return {
            "loss": total,
            "loss_t": loss_t,
            "loss_r": loss_r,
            "loss_r_geo": loss_r_geo,
        }
