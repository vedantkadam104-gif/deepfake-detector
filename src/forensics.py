"""Multi-signal visual forensic analysis: Error Level Analysis (ELA), 2D FFT, and Seam Consistency."""

from __future__ import annotations

import io
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image

from .face_pipeline import pil_to_base64


def compute_ela_map(pil_img: Image.Image, quality: int = 90) -> Tuple[float, str]:
    """Computes JPEG Error Level Analysis (ELA) anomaly score and base64 visualization."""
    rgb = pil_img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    orig_np = np.asarray(rgb, dtype=np.float32)
    recomp_np = np.asarray(recompressed, dtype=np.float32)
    diff = np.abs(orig_np - recomp_np)

    # Spatial variance of ELA highlights spliced / synthesized regions vs uniform natural noise
    gray_diff = diff.mean(axis=2)
    ela_std = float(np.std(gray_diff))
    ela_p95 = float(np.percentile(gray_diff, 95))
    raw_score = 0.5 * min(ela_std / 8.0, 1.0) + 0.5 * min(ela_p95 / 22.0, 1.0)
    ela_score = float(np.clip(raw_score, 0.0, 1.0))

    # Amplify 15x for clear visual inspection
    amplified = np.clip(diff * 15.0, 0, 255).astype(np.uint8)
    # Apply magma/inferno colormap on luminance diff for crisp forensic rendering
    amp_gray = cv2.cvtColor(amplified, cv2.COLOR_RGB2GRAY)
    colored_bgr = cv2.applyColorMap(amp_gray, cv2.COLORMAP_INFERNO)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    ela_b64 = pil_to_base64(Image.fromarray(colored_rgb))
    return ela_score, ela_b64


def compute_fft_spectrum(pil_img: Image.Image) -> Tuple[float, str]:
    """Computes 2D FFT log-magnitude spectrum and detects GAN/diffusion high-freq anomalies."""
    gray = np.asarray(pil_img.convert("L").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    # Measure ratio of high-frequency energy / azimuthal peakiness vs smooth natural 1/f falloff
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    mid_mask = (dist >= 18) & (dist < 55)
    high_mask = dist >= 55
    mid_energy = float(np.mean(magnitude[mid_mask])) + 1e-6
    high_energy = float(np.mean(magnitude[high_mask]))
    high_std = float(np.std(magnitude[high_mask]))

    ratio = high_energy / mid_energy
    # Natural photos decay smoothly; GAN checkerboard/diffusion artifacts elevate high_std & ratio
    spectral_score = float(np.clip((ratio - 0.55) * 1.8 + max(0.0, high_std - 0.85) * 0.4, 0.0, 1.0))

    # Normalize magnitude to 0..255 for display
    mag_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-6)
    mag_u8 = np.clip(mag_norm * 255.0, 0, 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(mag_u8, cv2.COLORMAP_VIRIDIS)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    fft_b64 = pil_to_base64(Image.fromarray(colored_rgb))
    return spectral_score, fft_b64


def compute_seam_consistency(pil_img: Image.Image) -> float:
    """Compares Laplacian sharpness/noise variance between inner face core and outer ring."""
    gray = np.asarray(pil_img.convert("L").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)

    h, w = lap.shape
    inner = lap[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)]
    mask = np.ones_like(lap, dtype=bool)
    mask[int(h * 0.20):int(h * 0.80), int(w * 0.20):int(w * 0.80)] = False
    outer = lap[mask]

    inner_var = float(np.var(inner)) + 1e-6
    outer_var = float(np.var(outer)) + 1e-6
    disparity = abs(np.log1p(inner_var) - np.log1p(outer_var)) / max(np.log1p(inner_var), np.log1p(outer_var), 1.0)
    return float(np.clip(disparity * 1.5, 0.0, 1.0))


def run_forensic_suite(pil_img: Image.Image) -> Dict[str, object]:
    """Runs ELA, 2D FFT, and Seam Consistency analysis on a cropped face image."""
    ela_score, ela_b64 = compute_ela_map(pil_img)
    fft_score, fft_b64 = compute_fft_spectrum(pil_img)
    seam_score = compute_seam_consistency(pil_img)
    return {
        "ela_score": round(ela_score, 4),
        "fft_score": round(fft_score, 4),
        "seam_score": round(seam_score, 4),
        "ela_map_b64": ela_b64,
        "fft_spectrum_b64": fft_b64,
    }
