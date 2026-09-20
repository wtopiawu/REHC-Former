from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

from segmentation.src.model import build_segformer
from segmentation.src.utils import ensure_dir, get_device, load_checkpoint, load_yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SegFormer-B1 gripper mask inference.")
    parser.add_argument("--config", type=str, default="outputs/segformer_b1_gripper/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/segformer_b1_gripper/checkpoints/best.pth")
    parser.add_argument("--hf-model-dir", type=str, default="", help="Optional saved Hugging Face model dir, e.g. outputs/.../hf_best")
    parser.add_argument("--input", type=str, required=True, help="Image file or directory.")
    parser.add_argument("--output", type=str, default="outputs/segmentation", help="Output directory.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--save-prob", action="store_true", help="Save foreground probability map as uint8 PNG.")
    parser.add_argument("--no-overlay", action="store_true", help="Do not save overlay visualization.")
    return parser.parse_args()


def iter_images(path: str | Path) -> List[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted([x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in IMAGE_EXTS])
    raise FileNotFoundError(f"Input not found: {p}")


def preprocess(image_bgr: np.ndarray, cfg: Dict) -> Tuple[torch.Tensor, Tuple[int, int]]:
    h0, w0 = image_bgr.shape[:2]
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_size = cfg["input"]["image_size"]
    tf = A.Compose(
        [
            A.Resize(height=int(image_size[0]), width=int(image_size[1]), interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=cfg["input"]["normalize_mean"], std=cfg["input"]["normalize_std"]),
        ]
    )
    image = tf(image=image)["image"]
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, (h0, w0)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1, constant_values=0)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 2)
    return np.logical_or(padded, flood == 0)[1:-1, 1:-1].astype(np.uint8)


def postprocess(mask: np.ndarray, cfg: Dict) -> np.ndarray:
    infer_cfg = cfg.get("infer", {})
    if bool(infer_cfg.get("fill_holes", True)):
        mask = fill_holes(mask)

    min_area = int(infer_cfg.get("min_area", 0))
    if min_area > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        keep = np.zeros_like(mask, dtype=np.uint8)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == i] = 1
        mask = keep
    return mask


def build_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    color = np.zeros_like(image_bgr)
    color[:, :, 1] = 255
    alpha = 0.45
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = cv2.addWeighted(image_bgr, 1 - alpha, color, alpha, 0)[mask_bool]
    return overlay


def load_model(cfg: Dict, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    hf_model_dir = Path(args.hf_model_dir) if args.hf_model_dir else None
    if hf_model_dir and hf_model_dir.exists():
        model = SegformerForSemanticSegmentation.from_pretrained(args.hf_model_dir)
    else:
        if hf_model_dir:
            print(f"[WARN] Hugging Face model dir not found: {hf_model_dir}. Falling back to checkpoint.")
        model = build_segformer(cfg["model"])
        ckpt = load_checkpoint(args.checkpoint, device)
        model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_one(model: torch.nn.Module, image_bgr: np.ndarray, cfg: Dict, device: torch.device, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    tensor, orig_size = preprocess(image_bgr, cfg)
    tensor = tensor.to(device)
    out = model(pixel_values=tensor)
    logits = F.interpolate(out.logits, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
    prob = torch.softmax(logits, dim=1)[0, 1].detach().cpu().numpy()

    h0, w0 = orig_size
    prob_orig = cv2.resize(prob, (w0, h0), interpolation=cv2.INTER_LINEAR)
    mask = (prob_orig >= threshold).astype(np.uint8)
    mask = postprocess(mask, cfg)
    return mask, prob_orig


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.threshold is not None:
        cfg.setdefault("infer", {})["threshold"] = float(args.threshold)
    threshold = float(cfg.get("infer", {}).get("threshold", 0.5))

    device = get_device(args.device)
    model = load_model(cfg, args, device)
    out_dir = ensure_dir(args.output)
    mask_dir = ensure_dir(out_dir / "masks")
    overlay_dir = ensure_dir(out_dir / "overlays")
    prob_dir = ensure_dir(out_dir / "probabilities")

    images = iter_images(args.input)
    if not images:
        raise RuntimeError(f"No images found in {args.input}")

    for img_path in tqdm(images, desc="Infer", dynamic_ncols=True):
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue
        mask, prob = predict_one(model, image_bgr, cfg, device, threshold)
        stem = img_path.stem
        cv2.imwrite(str(mask_dir / f"{stem}_mask.png"), (mask * 255).astype(np.uint8))
        if args.save_prob:
            cv2.imwrite(str(prob_dir / f"{stem}_prob.png"), np.clip(prob * 255, 0, 255).astype(np.uint8))
        if not args.no_overlay:
            overlay = build_overlay(image_bgr, mask)
            cv2.imwrite(str(overlay_dir / f"{stem}_overlay.jpg"), overlay)

    print(f"Saved predictions to: {out_dir}")


if __name__ == "__main__":
    main()
