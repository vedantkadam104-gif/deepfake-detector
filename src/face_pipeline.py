"""Automatic face detection and 22% margin cropping for consistent inference.

Ensures full-frame photos and video frames are cropped to match the tightly
aligned faces in the 140k Real & Fake Faces training distribution, while
preserving images that are already tightly cropped portraits.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@dataclass
class FaceExtractionResult:
    cropped_pil: Image.Image
    bbox: Optional[Tuple[int, int, int, int]]  # (x1, y1, x2, y2) in original image coords
    face_detected: bool
    already_tight_crop: bool
    annotated_preview_b64: str
    cropped_face_b64: str


class FaceExtractor:
    """Detects primary face using OpenCV Haar cascades and crops with blending margin."""

    def __init__(self, margin: float = 0.22) -> None:
        self.margin = margin
        cascade_dir = cv2.data.haarcascades
        self.frontal_cascade = cv2.CascadeClassifier(
            cascade_dir + "haarcascade_frontalface_default.xml"
        )
        self.alt_cascade = cv2.CascadeClassifier(
            cascade_dir + "haarcascade_frontalface_alt2.xml"
        )

    def _detect_largest_face(self, rgb_np: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        h, w = gray.shape
        min_dim = max(32, int(min(h, w) * 0.10))

        faces = self.frontal_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_dim, min_dim)
        )
        if len(faces) == 0:
            faces = self.alt_cascade.detectMultiScale(
                gray, scaleFactor=1.08, minNeighbors=4, minSize=(min_dim, min_dim)
            )
        if len(faces) == 0:
            return None

        # Pick largest detected face by area
        x, y, fw, fh = max(faces, key=lambda b: int(b[2]) * int(b[3]))
        return int(x), int(y), int(x + fw), int(y + fh)

    def extract(self, pil_img: Image.Image) -> FaceExtractionResult:
        rgb_pil = pil_img.convert("RGB")
        rgb_np = np.array(rgb_pil)
        h, w = rgb_np.shape[:2]
        img_area = float(h * w)

        bbox = self._detect_largest_face(rgb_np)
        already_tight_crop = False
        face_detected = bbox is not None

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            bw = x2 - x1
            bh = y2 - y1
            face_area_ratio = (bw * bh) / max(img_area, 1.0)

            # Check if image is already a tightly cropped portrait (e.g., 140k dataset 256x256)
            if face_area_ratio > 0.55 or (max(h, w) <= 320 and face_area_ratio > 0.35):
                already_tight_crop = True
                cropped_pil = rgb_pil
                crop_box = (0, 0, w, h)
            else:
                # Expand bbox by margin (default 22%) and make roughly square
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                side = max(bw, bh) * (1.0 + 2.0 * self.margin)
                nx1 = max(0, int(round(cx - side / 2.0)))
                ny1 = max(0, int(round(cy - side / 2.0)))
                nx2 = min(w, int(round(cx + side / 2.0)))
                ny2 = min(h, int(round(cy + side / 2.0)))
                crop_box = (nx1, ny1, nx2, ny2)
                cropped_pil = rgb_pil.crop(crop_box)
        else:
            # Heuristic fallback: if small square portrait crop, preserve full image; else center square crop
            aspect = max(w, h) / max(min(w, h), 1)
            if max(h, w) <= 320 and aspect <= 1.25:
                already_tight_crop = True
                cropped_pil = rgb_pil
                crop_box = (0, 0, w, h)
            else:
                side = min(w, h)
                cx, cy = w // 2, h // 2
                nx1 = max(0, cx - side // 2)
                ny1 = max(0, cy - side // 2)
                crop_box = (nx1, ny1, nx1 + side, ny1 + side)
                cropped_pil = rgb_pil.crop(crop_box)

        # Draw annotated preview
        annotated = cv2.cvtColor(rgb_np.copy(), cv2.COLOR_RGB2BGR)
        cx1, cy1, cx2, cy2 = crop_box
        cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (16, 185, 129), max(2, min(w, h) // 180))
        if bbox is not None:
            bx1, by1, bx2, by2 = bbox
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (56, 189, 248), max(2, min(w, h) // 180))

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        annotated_pil = Image.fromarray(annotated_rgb)

        return FaceExtractionResult(
            cropped_pil=cropped_pil,
            bbox=bbox,
            face_detected=face_detected,
            already_tight_crop=already_tight_crop,
            annotated_preview_b64=pil_to_base64(annotated_pil),
            cropped_face_b64=pil_to_base64(cropped_pil),
        )
