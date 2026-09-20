import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class BasicImageTransform:
    """RGB transform. Only photometric augmentation is used, so it does not desynchronize RGB and mask geometry."""

    def __init__(self, image_size: Tuple[int, int], train: bool = False) -> None:
        self.image_height, self.image_width = image_size
        self.train = train

    def _augment(self, image: Image.Image) -> Image.Image:
        brightness = 1.0 + random.uniform(-0.12, 0.12)
        contrast = 1.0 + random.uniform(-0.12, 0.12)
        color = 1.0 + random.uniform(-0.10, 0.10)
        sharpness = 1.0 + random.uniform(-0.08, 0.08)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Color(image).enhance(color)
        image = ImageEnhance.Sharpness(image).enhance(sharpness)
        if random.random() < 0.15:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.8)))
        return image

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.image_width, self.image_height), resample=Image.BILINEAR)
        if self.train:
            image = self._augment(image)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr.astype(np.float32))


class MaskTransform:
    """Binary/soft mask transform. Output shape: [1, H, W], value range: [0, 1]."""

    def __init__(self, image_size: Tuple[int, int], threshold: float = 0.5, invert: bool = False, binary: bool = True) -> None:
        self.image_height, self.image_width = image_size
        self.threshold = float(threshold)
        if not 0 <= self.threshold <= 1:
            raise ValueError("mask threshold must be in [0, 1]")
        self.invert = bool(invert)
        self.binary = bool(binary)

    def __call__(self, mask: Image.Image) -> torch.Tensor:
        mask = mask.convert("L")
        mask = mask.resize((self.image_width, self.image_height), resample=Image.NEAREST)
        arr = np.asarray(mask, dtype=np.float32) / 255.0
        if self.invert:
            arr = 1.0 - arr
        arr = np.clip(arr, 0.0, 1.0)
        if self.binary:
            arr = (arr >= self.threshold).astype(np.float32)
        arr = arr[None, :, :]
        return torch.from_numpy(arr.astype(np.float32))
