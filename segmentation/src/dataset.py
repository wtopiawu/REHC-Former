from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}


def _list_files(root: str | Path, exts: set[str]) -> List[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def build_pairs(image_dir: str | Path, mask_dir: str | Path) -> List[Tuple[Path, Path]]:
    """Pair image and mask files by relative stem.

    Supported layouts:
      images/train/a.jpg      masks/train/a.png
      images/train/sub/a.jpg  masks/train/sub/a.png

    The extension can differ, but the relative path stem must match.
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    images = _list_files(image_dir, IMAGE_EXTS)
    masks = _list_files(mask_dir, MASK_EXTS)

    mask_map: Dict[str, Path] = {}
    for m in masks:
        rel = m.relative_to(mask_dir).with_suffix("")
        mask_map[str(rel)] = m

    pairs: List[Tuple[Path, Path]] = []
    missing: List[str] = []
    for img in images:
        rel = img.relative_to(image_dir).with_suffix("")
        key = str(rel)
        mask = mask_map.get(key)
        if mask is None:
            missing.append(key)
        else:
            pairs.append((img, mask))

    if not pairs:
        raise RuntimeError(
            f"No image/mask pairs found. image_dir={image_dir}, mask_dir={mask_dir}. "
            "Ensure files share the same relative filename stem."
        )
    if missing:
        preview = ", ".join(missing[:10])
        print(f"[WARN] {len(missing)} image(s) have no matching mask. First missing: {preview}")
    return pairs


class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        image_size: Sequence[int] = (480, 640),
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        augment: Optional[Callable] = None,
        binary_mask: bool = True,
    ) -> None:
        self.pairs = build_pairs(image_dir, mask_dir)
        self.image_size = tuple(int(x) for x in image_size)  # h, w
        self.mean = tuple(float(x) for x in mean)
        self.std = tuple(float(x) for x in std)
        self.augment = augment
        self.binary_mask = binary_mask
        self.base_tf = A.Compose(
            [
                A.Resize(height=self.image_size[0], width=self.image_size[1], interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        img_path, mask_path = self.pairs[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        if self.binary_mask:
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        if self.augment is not None:
            transformed = self.augment(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        transformed = self.base_tf(image=image, mask=mask)
        image, mask = transformed["image"], transformed["mask"]

        # HWC -> CHW, float32. Mask remains long class indices [H, W].
        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask).long()

        return {
            "pixel_values": image_t,
            "labels": mask_t,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
        }


def build_train_augment(cfg: Dict) -> Optional[A.Compose]:
    aug_cfg = cfg.get("augment", {})
    if not aug_cfg.get("enabled", True):
        return None

    return A.Compose(
        [
            A.HorizontalFlip(p=float(aug_cfg.get("horizontal_flip_p", 0.5))),
            A.ShiftScaleRotate(
                shift_limit=float(aug_cfg.get("shift_limit", 0.05)),
                scale_limit=float(aug_cfg.get("scale_limit", 0.15)),
                rotate_limit=int(aug_cfg.get("rotate_limit", 15)),
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                interpolation=cv2.INTER_LINEAR,
                p=0.7,
            ),
            A.RandomBrightnessContrast(p=float(aug_cfg.get("brightness_contrast_p", 0.4))),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ],
                p=float(aug_cfg.get("blur_p", 0.15)),
            ),
            A.GaussNoise(var_limit=(5.0, 30.0), p=float(aug_cfg.get("noise_p", 0.15))),
            A.CoarseDropout(
                max_holes=8,
                max_height=48,
                max_width=48,
                min_holes=1,
                min_height=12,
                min_width=12,
                fill_value=0,
                mask_fill_value=0,
                p=float(aug_cfg.get("coarse_dropout_p", 0.2)),
            ),
        ]
    )
