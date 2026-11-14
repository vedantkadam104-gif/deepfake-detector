"""Shared preprocessing, label mapping, and TTA transforms for training and inference.

Keeping all normalization, RGB handling, and label indices in one module prevents
the most common causes of deepfake model failure:
- Alphabetical ImageFolder label inversion (fake=0, real=1)
- Training vs. inference normalization mismatch
- BGR vs. RGB channel flips
"""

from __future__ import annotations

import io
import random
from typing import List, Tuple

from PIL import Image
from torchvision import transforms

IMG_SIZE: int = 224
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Explicit label mapping (NEVER rely on alphabetical folder ordering)
CLASS_TO_IDX = {"real": 0, "fake": 1}
IDX_TO_CLASS = {0: "Real", 1: "Fake"}


class RandomJPEGCompression:
    """Simulates social media / WhatsApp JPEG recompression during training."""

    def __init__(self, quality_range: Tuple[int, int] = (55, 95), p: float = 0.4) -> None:
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def get_train_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Augmentations tailored for 140k+ face datasets with robustness to compression."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        RandomJPEGCompression(quality_range=(55, 95), p=0.4),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
    ])


def get_eval_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Deterministic evaluation / single-crop inference transform."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_tta_transforms(img_size: int = IMG_SIZE) -> List[transforms.Compose]:
    """4-view Test-Time Augmentation transforms for higher accuracy & stability."""
    tight_size = int(round(img_size / 0.92))
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return [
        # 1. Standard resized face
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]),
        # 2. Horizontal flip
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            normalize,
        ]),
        # 3. 92% tight center crop
        transforms.Compose([
            transforms.Resize((tight_size, tight_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]),
        # 4. 92% tight center crop + horizontal flip
        transforms.Compose([
            transforms.Resize((tight_size, tight_size)),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            normalize,
        ]),
    ]
