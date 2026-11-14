"""DeepfakeEfficientNet: Spatial CNN backbone + Radial 2D-FFT Frequency Fusion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import timm
import torch
import torch.nn as nn

from .transforms import CLASS_TO_IDX, IMG_SIZE


class RadialFFTExtractor(nn.Module):
    """Extracts a 16-bin normalized radial 2D-FFT log-magnitude profile on the GPU."""

    def __init__(self, num_bins: int = 16, img_size: int = IMG_SIZE) -> None:
        super().__init__()
        self.num_bins = num_bins
        cy, cx = img_size // 2, img_size // 2
        yy, xx = torch.meshgrid(
            torch.arange(img_size, dtype=torch.float32),
            torch.arange(img_size, dtype=torch.float32),
            indexing="ij",
        )
        radius = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        max_r = float(radius.max().item()) + 1e-6
        bin_idx = torch.clamp((radius / max_r * num_bins).long(), 0, num_bins - 1)

        # Precompute one-hot bin mask [num_bins, H, W] and bin counts
        masks = torch.zeros(num_bins, img_size, img_size, dtype=torch.float32)
        for b in range(num_bins):
            masks[b] = (bin_idx == b).float()
        counts = masks.sum(dim=(1, 2)).clamp(min=1.0)
        self.register_buffer("bin_masks", masks / counts.view(num_bins, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Force float32 for cuFFT stability under AMP fp16
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_f32 = x.float()
            # Convert normalized RGB tensor to luminance [B, H, W]
            gray = 0.299 * x_f32[:, 0] + 0.587 * x_f32[:, 1] + 0.114 * x_f32[:, 2]
            h, w = gray.shape[-2], gray.shape[-1]
            if h != self.bin_masks.shape[-2] or w != self.bin_masks.shape[-1]:
                gray = torch.nn.functional.interpolate(
                    gray.unsqueeze(1),
                    size=(self.bin_masks.shape[-2], self.bin_masks.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

            fft = torch.fft.fft2(gray, norm="ortho")
            fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
            mag = torch.log1p(torch.abs(fft_shift))  # [B, H, W]

            # Radial averaging via einsum -> [B, num_bins]
            radial = torch.einsum("bhw,khw->bk", mag, self.bin_masks)
            # Per-sample z-score normalization so it captures spectral shape, not brightness
            mean = radial.mean(dim=1, keepdim=True)
            std = radial.std(dim=1, keepdim=True).clamp(min=1e-5)
            return (radial - mean) / std


class DeepfakeEfficientNet(nn.Module):
    """EfficientNet backbone fused with 16-bin radial FFT frequency profile."""

    def __init__(
        self,
        arch: str = "efficientnet_b0",
        pretrained: bool = True,
        num_classes: int = 2,
        fft_bins: int = 16,
        img_size: int = IMG_SIZE,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.img_size = img_size
        self.fft_bins = fft_bins

        self.backbone = timm.create_model(
            arch,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=0.3,
            drop_path_rate=0.2,
        )
        backbone_dim = getattr(self.backbone, "num_features", 1280)
        self.fft_extractor = RadialFFTExtractor(num_bins=fft_bins, img_size=img_size)

        fused_dim = backbone_dim + fft_bins
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(0.3),
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_feats = self.backbone(x)
        freq_feats = self.fft_extractor(x).to(dtype=spatial_feats.dtype)
        fused = torch.cat([spatial_feats, freq_feats], dim=1)
        return self.head(fused)


def save_checkpoint(
    path: Path,
    model: DeepfakeEfficientNet,
    epoch: int,
    metrics: Dict[str, float],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "arch": model.arch,
        "img_size": model.img_size,
        "fft_bins": model.fft_bins,
        "class_to_idx": CLASS_TO_IDX,
        "epoch": epoch,
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)

    mapping_path = path.parent / "class_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(CLASS_TO_IDX, f, indent=2)


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> Tuple[DeepfakeEfficientNet, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "efficientnet_b0")
    img_size = ckpt.get("img_size", IMG_SIZE)
    fft_bins = ckpt.get("fft_bins", 16)

    model = DeepfakeEfficientNet(
        arch=arch,
        pretrained=False,
        num_classes=2,
        fft_bins=fft_bins,
        img_size=img_size,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, ckpt
