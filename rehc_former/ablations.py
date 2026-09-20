"""Four controlled ablation models for RGB/Mask eye-in-hand pose regression.

All variants share the same output contract and pose heads.  The only intended
difference is the input/fusion mechanism selected by ``variant``.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import rotation_6d_to_matrix, rotation_9d_to_matrix

try:
    from torchvision.models import convnext_tiny, resnet18
except Exception:  # pragma: no cover - torchvision is optional for lite_cnn
    convnext_tiny = None
    resnet18 = None


VARIANTS = ("rgb_only", "mask_only", "early_fusion", "late_fusion")


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 2.0, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = residual + x
        return x + self.mlp(self.norm2(x))


class LiteBackbone(nn.Module):
    """The same lightweight backbone topology as the supplied full model."""

    def __init__(self, in_channels: int = 3, stem_width: int = 32, out_channels: int = 128) -> None:
        super().__init__()
        w1, w2, w3 = stem_width, stem_width * 2, stem_width * 3
        channels = [(in_channels, w1, 2), (w1, w1, 1), (w1, w2, 2), (w2, w2, 1),
                    (w2, w3, 2), (w3, w3, 1), (w3, out_channels, 2)]
        layers = []
        for cin, cout, stride in channels:
            layers.extend([
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.GELU(),
            ])
        self.net = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FallbackBackbone(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        widths = [64, 128, 256, out_channels]
        layers = []
        cin = in_channels
        for index, width in enumerate(widths):
            stride = 2 if index < 3 else 1
            layers.extend([
                nn.Conv2d(cin, width, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.GELU(),
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.GELU(),
            ])
            cin = width
        self.net = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _replace_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Adapt a pretrained 3-channel stem to four-channel early fusion."""
    if conv.in_channels == in_channels:
        return conv
    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        copy_channels = min(conv.in_channels, in_channels)
        new_conv.weight[:, :copy_channels].copy_(conv.weight[:, :copy_channels])
        if in_channels > conv.in_channels:
            mean_weight = conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, conv.in_channels:].copy_(mean_weight.expand(-1, in_channels - conv.in_channels, -1, -1))
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


class BackboneFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone_name: str = "lite_cnn",
        pretrained: bool = False,
        stem_width: int = 32,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.in_channels = int(in_channels)
        if self.backbone_name in {"lite_cnn", "lite", "tiny_cnn"}:
            self.model = LiteBackbone(in_channels=in_channels, stem_width=stem_width, out_channels=128)
            self.out_channels = 128
        elif self.backbone_name == "resnet18":
            if resnet18 is None:
                raise ImportError("resnet18 requires a working torchvision installation")
            else:
                try:
                    from torchvision.models import ResNet18_Weights
                    self.model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
                except Exception:
                    self.model = resnet18(pretrained=pretrained)
                self.model.conv1 = _replace_first_conv(self.model.conv1, in_channels)
            self.out_channels = 512
        elif self.backbone_name in {"convnext_tiny", "convnext"}:
            if convnext_tiny is None:
                raise ImportError("convnext_tiny requires a working torchvision installation")
            else:
                try:
                    from torchvision.models import ConvNeXt_Tiny_Weights
                    self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
                except Exception:
                    self.model = convnext_tiny(pretrained=pretrained)
                self.model.features[0][0] = _replace_first_conv(self.model.features[0][0], in_channels)
            self.out_channels = 768
        else:
            raise ValueError(f"不支持的 backbone: {backbone_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_name in {"lite_cnn", "lite", "tiny_cnn"} or isinstance(self.model, FallbackBackbone):
            return self.model(x)
        if self.backbone_name == "resnet18":
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            return self.model.layer4(x)
        return self.model.features(x)


class MaskFeatureStem(nn.Module):
    """The same mask stem used by the supplied full cross-attention model."""

    def __init__(self, embed_dim: int, stem_width: int = 32) -> None:
        super().__init__()
        hidden = max(32, stem_width * 2)
        self.net = nn.Sequential(
            nn.Conv2d(1, stem_width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_width), nn.GELU(),
            nn.Conv2d(stem_width, hidden, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, embed_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim), nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim), nn.GELU(),
        )

    def forward(self, mask: torch.Tensor, target_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        if mask.ndim != 4:
            raise ValueError(f"mask 必须为 [B,1,H,W]，当前 shape={tuple(mask.shape)}")
        if mask.shape[1] != 1:
            mask = mask.mean(dim=1, keepdim=True)
        feat = self.net(mask.float().clamp(0.0, 1.0))
        if target_hw is not None and feat.shape[-2:] != target_hw:
            feat = F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
        return feat


class PoseOutputHead(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, dropout: float, rotation_repr: str) -> None:
        super().__init__()
        self.rotation_repr = rotation_repr
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, embed_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.translation_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim, 3)
        )
        self.rotation_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, 9 if rotation_repr == "9d" else 6),
        )

    def forward(self, pooled: torch.Tensor) -> Dict[str, torch.Tensor]:
        fused = self.fusion(pooled)
        pred_t_norm = self.translation_head(fused)
        pred_repr = self.rotation_head(fused)
        pred_R = rotation_9d_to_matrix(pred_repr) if self.rotation_repr == "9d" else rotation_6d_to_matrix(pred_repr)
        outputs = {
            "pred_translation_norm": pred_t_norm,
            "pred_rotation_repr": pred_repr,
            "pred_rotation_matrix": pred_R,
        }
        outputs[f"pred_rotation_{self.rotation_repr}"] = pred_repr
        return outputs


class AblationModelBase(nn.Module):
    variant = "base"

    def __init__(
        self,
        image_size: Tuple[int, int] = (480, 640),
        embed_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.05,
        patch_stride: int = 2,
        backbone_name: str = "lite_cnn",
        backbone_pretrained: bool = False,
        mlp_ratio: float = 2.0,
        stem_width: int = 32,
        rotation_repr: str = "9d",
    ) -> None:
        super().__init__()
        if embed_dim % 4 != 0:
            raise ValueError("embed_dim 必须能被 4 整除")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除")
        if patch_stride < 1:
            raise ValueError("patch_stride 必须 >= 1")
        rotation_repr = str(rotation_repr).lower()
        if rotation_repr not in {"6d", "9d"}:
            raise ValueError("rotation_repr 必须是 6d 或 9d")
        self.image_size = tuple(image_size)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.patch_stride = patch_stride
        self.backbone_name = backbone_name
        self.backbone_pretrained = backbone_pretrained
        self.mlp_ratio = mlp_ratio
        self.stem_width = stem_width
        self.rotation_repr = rotation_repr
        self.token_dropout = nn.Dropout(dropout)

    @staticmethod
    def _pos_embed(h: int, w: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y = torch.arange(h, device=device, dtype=torch.float32)
        x = torch.arange(w, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        omega = torch.arange(dim // 4, device=device, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / max(1, dim // 4)))
        py = yy.reshape(-1, 1) * omega.reshape(1, -1)
        px = xx.reshape(-1, 1) * omega.reshape(1, -1)
        pos = torch.cat([py.sin(), py.cos(), px.sin(), px.cos()], dim=1)
        return pos.unsqueeze(0).to(dtype=dtype)

    def _to_tokens(self, feat: torch.Tensor, cls_token: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feat.shape
        spatial = feat.flatten(2).transpose(1, 2)
        spatial = spatial + self._pos_embed(height, width, channels, feat.device, feat.dtype)
        cls = cls_token.expand(batch, -1, -1)
        return self.token_dropout(torch.cat([cls, spatial], dim=1))

    @staticmethod
    def _pool(tokens: torch.Tensor) -> torch.Tensor:
        return torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1)], dim=-1)

    def _reset_parameters(self) -> None:
        for name, parameter in self.named_parameters():
            if "cls_token" in name:
                nn.init.trunc_normal_(parameter, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def get_init_config(self) -> Dict:
        return {
            "variant": self.variant,
            "image_size": list(self.image_size),
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
            "patch_stride": self.patch_stride,
            "backbone_name": self.backbone_name,
            "backbone_pretrained": self.backbone_pretrained,
            "mlp_ratio": self.mlp_ratio,
            "stem_width": self.stem_width,
            "rotation_repr": self.rotation_repr,
        }

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class RGBOnlyPoseTransformer(AblationModelBase):
    """RGB backbone + self-attention only; the mask tensor is ignored."""
    variant = "rgb_only"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.backbone = BackboneFeatureExtractor(self.backbone_name, self.backbone_pretrained, self.stem_width, 3)
        self.proj = nn.Conv2d(self.backbone.out_channels, self.embed_dim, 1)
        self.patch_proj = nn.Conv2d(self.embed_dim, self.embed_dim, self.patch_stride, stride=self.patch_stride)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(self.embed_dim, self.num_heads, self.mlp_ratio, self.dropout_rate)
            for _ in range(self.depth)
        ])
        self.norm = nn.LayerNorm(self.embed_dim)
        self.pose_head = PoseOutputHead(self.embed_dim * 2, self.embed_dim, self.dropout_rate, self.rotation_repr)
        self._reset_parameters()

    def forward(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        feat = self.patch_proj(F.gelu(self.proj(self.backbone(image))))
        tokens = self._to_tokens(feat, self.cls_token)
        for block in self.blocks:
            tokens = block(tokens)
        outputs = self.pose_head(self._pool(self.norm(tokens)))
        outputs["input_mode"] = self.variant
        return outputs


class MaskOnlyPoseTransformer(AblationModelBase):
    """Mask geometry stem + self-attention only; RGB appearance is not used."""
    variant = "mask_only"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mask_stem = MaskFeatureStem(self.embed_dim, self.stem_width)
        self.patch_proj = nn.Conv2d(self.embed_dim, self.embed_dim, self.patch_stride, stride=self.patch_stride)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(self.embed_dim, self.num_heads, self.mlp_ratio, self.dropout_rate)
            for _ in range(self.depth)
        ])
        self.norm = nn.LayerNorm(self.embed_dim)
        self.pose_head = PoseOutputHead(self.embed_dim * 2, self.embed_dim, self.dropout_rate, self.rotation_repr)
        self._reset_parameters()

    def forward(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if mask is None:
            raise ValueError("Mask-only 模型必须调用 model(image, mask)")
        # The full supplied lite_cnn model aligns the mask stream to an RGB feature map of stride 16.
        target_hw = ((mask.shape[-2] + 15) // 16, (mask.shape[-1] + 15) // 16)
        feat = self.patch_proj(self.mask_stem(mask, target_hw=target_hw))
        tokens = self._to_tokens(feat, self.cls_token)
        for block in self.blocks:
            tokens = block(tokens)
        outputs = self.pose_head(self._pool(self.norm(tokens)))
        outputs["input_mode"] = self.variant
        return outputs


class EarlyFusionPoseTransformer(AblationModelBase):
    """Concatenate RGB and mask into a four-channel tensor before the backbone."""
    variant = "early_fusion"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.backbone = BackboneFeatureExtractor(self.backbone_name, self.backbone_pretrained, self.stem_width, 4)
        self.proj = nn.Conv2d(self.backbone.out_channels, self.embed_dim, 1)
        self.patch_proj = nn.Conv2d(self.embed_dim, self.embed_dim, self.patch_stride, stride=self.patch_stride)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(self.embed_dim, self.num_heads, self.mlp_ratio, self.dropout_rate)
            for _ in range(self.depth)
        ])
        self.norm = nn.LayerNorm(self.embed_dim)
        self.pose_head = PoseOutputHead(self.embed_dim * 2, self.embed_dim, self.dropout_rate, self.rotation_repr)
        self._reset_parameters()

    def forward(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if mask is None:
            raise ValueError("Early-fusion 模型必须调用 model(image, mask)")
        if mask.shape[-2:] != image.shape[-2:]:
            mask = F.interpolate(mask, size=image.shape[-2:], mode="nearest")
        if mask.shape[1] != 1:
            mask = mask.mean(dim=1, keepdim=True)
        fused_input = torch.cat([image, mask.float().clamp(0.0, 1.0)], dim=1)
        feat = self.patch_proj(F.gelu(self.proj(self.backbone(fused_input))))
        tokens = self._to_tokens(feat, self.cls_token)
        for block in self.blocks:
            tokens = block(tokens)
        outputs = self.pose_head(self._pool(self.norm(tokens)))
        outputs["input_mode"] = self.variant
        return outputs


class LateFusionPoseTransformer(AblationModelBase):
    """Independent RGB/Mask self-attention streams, concatenated only before the head."""
    variant = "late_fusion"

    def __init__(self, legacy_mask_alignment: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.legacy_mask_alignment = legacy_mask_alignment
        self.rgb_backbone = BackboneFeatureExtractor(self.backbone_name, self.backbone_pretrained, self.stem_width, 3)
        self.rgb_proj = nn.Conv2d(self.rgb_backbone.out_channels, self.embed_dim, 1)
        self.rgb_patch_proj = nn.Conv2d(self.embed_dim, self.embed_dim, self.patch_stride, stride=self.patch_stride)
        self.mask_stem = MaskFeatureStem(self.embed_dim, self.stem_width)
        self.mask_patch_proj = nn.Conv2d(self.embed_dim, self.embed_dim, self.patch_stride, stride=self.patch_stride)
        self.rgb_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.mask_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.rgb_blocks = nn.ModuleList([
            SelfAttentionBlock(self.embed_dim, self.num_heads, self.mlp_ratio, self.dropout_rate)
            for _ in range(self.depth)
        ])
        self.mask_blocks = nn.ModuleList([
            SelfAttentionBlock(self.embed_dim, self.num_heads, self.mlp_ratio, self.dropout_rate)
            for _ in range(self.depth)
        ])
        self.rgb_norm = nn.LayerNorm(self.embed_dim)
        self.mask_norm = nn.LayerNorm(self.embed_dim)
        self.pose_head = PoseOutputHead(self.embed_dim * 4, self.embed_dim, self.dropout_rate, self.rotation_repr)
        self._reset_parameters()

    def forward(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if mask is None:
            raise ValueError("Late-fusion 模型必须调用 model(image, mask)")
        rgb_features = F.gelu(self.rgb_proj(self.rgb_backbone(image)))
        rgb_feat = self.rgb_patch_proj(rgb_features)
        target_hw = rgb_feat.shape[-2:] if self.legacy_mask_alignment else rgb_features.shape[-2:]
        mask_feat = self.mask_patch_proj(self.mask_stem(mask, target_hw=target_hw))
        rgb_tokens = self._to_tokens(rgb_feat, self.rgb_cls_token)
        mask_tokens = self._to_tokens(mask_feat, self.mask_cls_token)
        for rgb_block, mask_block in zip(self.rgb_blocks, self.mask_blocks):
            rgb_tokens = rgb_block(rgb_tokens)
            mask_tokens = mask_block(mask_tokens)
        pooled = torch.cat([
            self._pool(self.rgb_norm(rgb_tokens)),
            self._pool(self.mask_norm(mask_tokens)),
        ], dim=-1)
        outputs = self.pose_head(pooled)
        outputs["input_mode"] = self.variant
        return outputs

    def get_init_config(self) -> Dict:
        cfg = super().get_init_config()
        cfg["legacy_mask_alignment"] = self.legacy_mask_alignment
        return cfg


MODEL_REGISTRY = {
    "rgb_only": RGBOnlyPoseTransformer,
    "mask_only": MaskOnlyPoseTransformer,
    "early_fusion": EarlyFusionPoseTransformer,
    "late_fusion": LateFusionPoseTransformer,
}


def build_model(variant: str, **kwargs) -> AblationModelBase:
    variant = str(variant).lower().strip()
    if variant not in MODEL_REGISTRY:
        raise ValueError(f"未知 variant={variant}; 可选: {', '.join(VARIANTS)}")
    return MODEL_REGISTRY[variant](**kwargs)
