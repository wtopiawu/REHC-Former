from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import rotation_6d_to_matrix, rotation_9d_to_matrix

try:
    from torchvision.models import convnext_tiny, resnet18
except Exception:  # pragma: no cover
    convnext_tiny = None
    resnet18 = None


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
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        residual = q_tokens
        q = self.q_norm(q_tokens)
        kv = self.kv_norm(kv_tokens)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        x = residual + out
        x = x + self.mlp(self.out_norm(x))
        return x


class DualStreamFusionBlock(nn.Module):
    """RGB stream and mask-geometry stream: self-attention first, then bidirectional cross-attention."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.rgb_self = SelfAttentionBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
        self.mask_self = SelfAttentionBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
        self.rgb_cross = CrossAttentionBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
        self.mask_cross = CrossAttentionBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, rgb_tokens: torch.Tensor, mask_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_tokens = self.rgb_self(rgb_tokens)
        mask_tokens = self.mask_self(mask_tokens)
        rgb_tokens = self.rgb_cross(rgb_tokens, mask_tokens)
        mask_tokens = self.mask_cross(mask_tokens, rgb_tokens)
        return rgb_tokens, mask_tokens


class LiteRGBBackbone(nn.Module):
    """Small RGB backbone. Output stride is approximately 16."""

    def __init__(self, stem_width: int = 32, out_channels: int = 128) -> None:
        super().__init__()
        w1 = stem_width
        w2 = stem_width * 2
        w3 = stem_width * 3
        self.net = nn.Sequential(
            nn.Conv2d(3, w1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w1),
            nn.GELU(),
            nn.Conv2d(w1, w1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(w1),
            nn.GELU(),
            nn.Conv2d(w1, w2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w2),
            nn.GELU(),
            nn.Conv2d(w2, w2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(w2),
            nn.GELU(),
            nn.Conv2d(w2, w3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w3),
            nn.GELU(),
            nn.Conv2d(w3, w3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(w3),
            nn.GELU(),
            nn.Conv2d(w3, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FallbackBackbone(nn.Module):
    def __init__(self, out_channels: int = 512) -> None:
        super().__init__()
        widths = [64, 128, 256, out_channels]
        layers = []
        in_ch = 3
        for idx, width in enumerate(widths):
            stride = 2 if idx < 3 else 1
            layers.extend([
                nn.Conv2d(in_ch, width, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.GELU(),
                nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.GELU(),
            ])
            in_ch = width
        self.net = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BackboneFeatureExtractor(nn.Module):
    def __init__(self, backbone_name: str = "lite_cnn", pretrained: bool = False, stem_width: int = 32) -> None:
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.pretrained = pretrained
        self.stem_width = stem_width

        if self.backbone_name in {"lite_cnn", "lite", "tiny_cnn"}:
            self.model = LiteRGBBackbone(stem_width=stem_width, out_channels=128)
            self.out_channels = 128
        elif self.backbone_name == "resnet18":
            if resnet18 is None:
                raise ImportError("resnet18 requires a working torchvision installation")
            else:
                self.model = self._build_resnet18(pretrained=pretrained)
            self.out_channels = 512
        elif self.backbone_name in {"convnext_tiny", "convnext"}:
            if convnext_tiny is None:
                raise ImportError("convnext_tiny requires a working torchvision installation")
            else:
                self.model = self._build_convnext_tiny(pretrained=pretrained)
            self.out_channels = 768
        else:
            raise ValueError(f"不支持的 backbone: {backbone_name}. 可选: lite_cnn, resnet18, convnext_tiny")

    @staticmethod
    def _build_resnet18(pretrained: bool):
        try:
            from torchvision.models import ResNet18_Weights
            return resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        except Exception:
            return resnet18(pretrained=pretrained)

    @staticmethod
    def _build_convnext_tiny(pretrained: bool):
        try:
            from torchvision.models import ConvNeXt_Tiny_Weights
            return convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        except Exception:
            return convnext_tiny(pretrained=pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_name in {"lite_cnn", "lite", "tiny_cnn"}:
            return self.model(x)

        if self.backbone_name == "resnet18":
            if isinstance(self.model, FallbackBackbone):
                return self.model(x)
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            return x

        if self.backbone_name in {"convnext_tiny", "convnext"}:
            if isinstance(self.model, FallbackBackbone):
                return self.model(x)
            return self.model.features(x)

        raise RuntimeError(f"未知 backbone: {self.backbone_name}")


class MaskFeatureStem(nn.Module):
    """Mask branch stem. Input is [B, 1, H, W], output is aligned to the RGB feature map size."""

    def __init__(self, embed_dim: int, stem_width: int = 32) -> None:
        super().__init__()
        hidden = max(32, stem_width * 2)
        self.net = nn.Sequential(
            nn.Conv2d(1, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.GELU(),
            nn.Conv2d(stem_width, hidden, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, mask: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        if mask.ndim != 4:
            raise ValueError(f"mask 必须是 [B,1,H,W] 或 [B,C,H,W]，当前 shape={tuple(mask.shape)}")
        if mask.shape[1] != 1:
            mask = mask.mean(dim=1, keepdim=True)
        mask = mask.float().clamp(0.0, 1.0)
        feat = self.net(mask)
        if feat.shape[-2:] != target_hw:
            feat = F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
        return feat


class CrossAttentionTECMaskTransformer(nn.Module):
    """RGB + mask dual-stream cross-attention model for eye-in-hand T_EC regression.

    Design:
    - RGB stream keeps texture, color, illumination and local appearance cues.
    - Mask stream encodes foreground silhouette and geometry.
    - Bidirectional cross-attention lets RGB tokens query mask geometry, and mask tokens query RGB appearance.
    """

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
        processor_type: str = "mask",
        mlp_ratio: float = 2.0,
        stem_width: int = 32,
        rotation_repr: str = "9d",
        allow_missing_mask: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % 4 != 0:
            raise ValueError("embed_dim 必须能被 4 整除，以构造 2D sin-cos 位置编码")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除")
        if patch_stride < 1:
            raise ValueError("patch_stride 必须 >= 1")

        self.image_size = image_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.patch_stride = patch_stride
        self.backbone_name = backbone_name
        self.backbone_pretrained = backbone_pretrained
        self.processor_type = processor_type
        self.mlp_ratio = mlp_ratio
        self.stem_width = stem_width
        self.rotation_repr = str(rotation_repr).lower()
        self.allow_missing_mask = bool(allow_missing_mask)
        if self.rotation_repr not in {"6d", "9d"}:
            raise ValueError("rotation_repr 必须是 6d 或 9d")
        if processor_type not in {"mask", "binary_mask"}:
            raise ValueError("本版本的第二路固定为 mask/binary_mask")

        self.rgb_backbone = BackboneFeatureExtractor(
            backbone_name=backbone_name,
            pretrained=backbone_pretrained,
            stem_width=stem_width,
        )
        self.rgb_proj = nn.Conv2d(self.rgb_backbone.out_channels, embed_dim, kernel_size=1, stride=1)
        self.rgb_patch_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_stride, stride=patch_stride)

        self.mask_stem = MaskFeatureStem(embed_dim=embed_dim, stem_width=stem_width)
        self.mask_patch_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_stride, stride=patch_stride)

        self.rgb_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.token_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [DualStreamFusionBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)]
        )
        self.rgb_norm = nn.LayerNorm(embed_dim)
        self.mask_norm = nn.LayerNorm(embed_dim)

        fused_dim = embed_dim * 4
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.translation_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 3),
        )
        self.rotation_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 9 if self.rotation_repr == "9d" else 6),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.rgb_cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _build_2d_sincos_pos_embed(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
        if dim % 4 != 0:
            raise ValueError("embed_dim 必须能被 4 整除，以构造 2D sin-cos 位置编码")
        grid_y = torch.arange(h, device=device, dtype=torch.float32)
        grid_x = torch.arange(w, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        omega = torch.arange(dim // 4, device=device, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / max(1, dim // 4)))
        out_y = yy.reshape(-1, 1) * omega.reshape(1, -1)
        out_x = xx.reshape(-1, 1) * omega.reshape(1, -1)
        pos = torch.cat([torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)], dim=1)
        return pos.unsqueeze(0)

    def _to_tokens(self, feat: torch.Tensor, cls_token: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        pos = self._build_2d_sincos_pos_embed(h, w, c, feat.device)
        cls = cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens + pos], dim=1)
        return self.token_dropout(tokens)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        if mask is None:
            if not self.allow_missing_mask:
                raise ValueError("当前模型需要 mask 输入: model(rgb, mask)。训练和推理必须保持一致。")
            mask = torch.ones((x.shape[0], 1, x.shape[-2], x.shape[-1]), dtype=x.dtype, device=x.device)

        rgb_feat = self.rgb_backbone(x)
        rgb_feat = F.gelu(self.rgb_proj(rgb_feat))

        mask_feat = self.mask_stem(mask, target_hw=rgb_feat.shape[-2:])

        rgb_feat = self.rgb_patch_proj(rgb_feat)
        mask_feat = self.mask_patch_proj(mask_feat)

        rgb_tokens = self._to_tokens(rgb_feat, self.rgb_cls_token)
        mask_tokens = self._to_tokens(mask_feat, self.mask_cls_token)

        for block in self.blocks:
            rgb_tokens, mask_tokens = block(rgb_tokens, mask_tokens)

        rgb_tokens = self.rgb_norm(rgb_tokens)
        mask_tokens = self.mask_norm(mask_tokens)

        rgb_cls = rgb_tokens[:, 0]
        rgb_mean = rgb_tokens[:, 1:].mean(dim=1)
        mask_cls = mask_tokens[:, 0]
        mask_mean = mask_tokens[:, 1:].mean(dim=1)

        fused = self.fusion(torch.cat([rgb_cls, rgb_mean, mask_cls, mask_mean], dim=-1))
        pred_t_norm = self.translation_head(fused)
        pred_rotation_repr = self.rotation_head(fused)
        if self.rotation_repr == "9d":
            pred_rotmat = rotation_9d_to_matrix(pred_rotation_repr)
        else:
            pred_rotmat = rotation_6d_to_matrix(pred_rotation_repr)

        outputs = {
            "pred_translation_norm": pred_t_norm,
            "pred_rotation_repr": pred_rotation_repr,
            "pred_rotation_matrix": pred_rotmat,
            "mask_input": mask,
        }
        if self.rotation_repr == "9d":
            outputs["pred_rotation_9d"] = pred_rotation_repr
        else:
            outputs["pred_rotation_6d"] = pred_rotation_repr
        return outputs

    def get_init_config(self) -> Dict:
        return {
            "image_size": list(self.image_size),
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
            "patch_stride": self.patch_stride,
            "backbone_name": self.backbone_name,
            "backbone_pretrained": self.backbone_pretrained,
            "processor_type": self.processor_type,
            "mlp_ratio": self.mlp_ratio,
            "stem_width": self.stem_width,
            "rotation_repr": self.rotation_repr,
            "allow_missing_mask": self.allow_missing_mask,
            "input_mode": "rgb_mask_dual_stream",
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Compatibility aliases. Existing train/infer code can import the old class name after switching this file to model.py.
CrossAttentionTECTransformer = CrossAttentionTECMaskTransformer
CrossAttentionTCBTransformer = CrossAttentionTECMaskTransformer
