"""Construct models without changing checkpoint parameter names."""
import warnings

import numpy as np
import torch

from .ablations import VARIANTS as ABLATIONS, build_model as build_ablation
from .model import CrossAttentionTECMaskTransformer

VARIANTS = ("rehc_former", *ABLATIONS)


def build_model(variant="rehc_former", **config):
    config = dict(config)
    config.pop("input_mode", None)
    if variant == "rehc_former":
        return CrossAttentionTECMaskTransformer(**config)
    return build_ablation(variant, **config)


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model_state", "model_config", "translation_mean", "translation_std", "translation_unit"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"Checkpoint is missing: {sorted(missing)}")
    config = dict(checkpoint["model_config"])
    variant = checkpoint.get("model_variant", config.pop("variant", "rehc_former"))
    config.pop("variant", None)
    config["backbone_pretrained"] = False
    if variant == "late_fusion" and "legacy_mask_alignment" not in config:
        config["legacy_mask_alignment"] = True
        warnings.warn("Legacy late-fusion checkpoint: preserving its original mask downsampling. Retrain for aligned tokens.")
    mean = np.asarray(checkpoint["translation_mean"])
    std = np.asarray(checkpoint["translation_std"])
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Checkpoint translation statistics must be finite 3-vectors with positive std")
    if checkpoint["translation_unit"] not in {"m", "cm", "mm"}:
        raise ValueError("Unsupported checkpoint translation unit")
    model = build_model(variant, **config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), checkpoint


def mask_options(checkpoint, threshold=None, invert=None, soft=None):
    args = checkpoint.get("train_args", {})
    return {
        "threshold": args.get("mask_threshold", 0.5) if threshold is None else threshold,
        "invert": args.get("mask_invert", False) if invert is None else invert,
        "binary": not (args.get("soft_mask", False) if soft is None else soft),
    }
