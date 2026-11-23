"""FastAPI Backend for DeepShield AI — Deepfake Forensic Studio."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFilter

from src.detector import DeepfakeDetectionEngine

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SAMPLES_DIR = BASE_DIR / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="DeepShield AI — Deepfake Forensic Studio", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

engine: DeepfakeDetectionEngine | None = None


def get_engine() -> DeepfakeDetectionEngine:
    global engine
    if engine is None:
        engine = DeepfakeDetectionEngine()
    return engine


def _ensure_demo_samples() -> Dict[str, Dict[str, str]]:
    """Generates clean local demo images if samples/ is empty so one-click demo works offline too."""
    real_path = SAMPLES_DIR / "sample_authentic_face.png"
    fake_path = SAMPLES_DIR / "sample_synthetic_splice.png"

    if not real_path.exists():
        # Create a natural textured portrait-like image with smooth gradients & organic noise
        rng = np.random.default_rng(42)
        h, w = 256, 256
        yy, xx = np.mgrid[0:h, 0:w]
        bg = np.zeros((h, w, 3), dtype=np.float32)
        bg[:, :, 0] = 45 + 30 * np.sin(xx / 60.0)
        bg[:, :, 1] = 55 + 25 * np.cos(yy / 70.0)
        bg[:, :, 2] = 70 + 20 * np.sin((xx + yy) / 90.0)
        img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        # Head silhouette & facial structure
        draw.ellipse([58, 36, 198, 214], fill=(214, 168, 138))
        draw.ellipse([86, 98, 112, 114], fill=(42, 34, 30))
        draw.ellipse([144, 98, 170, 114], fill=(42, 34, 30))
        draw.polygon([(128, 114), (118, 148), (138, 148)], fill=(195, 145, 115))
        draw.arc([102, 158, 154, 182], start=10, end=170, fill=(160, 75, 75), width=4)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
        arr = np.asarray(img, dtype=np.float32) + rng.normal(0, 2.2, (h, w, 3))
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(real_path)

    if not fake_path.exists():
        # Create a sample with GAN checkerboard high-frequency grid + sharp splice seam + JPEG double compression
        base = Image.open(real_path).convert("RGB")
        arr = np.asarray(base, dtype=np.float32).copy()
        # Inject subtle 4px GAN upsampling checkerboard inside inner face box
        yy, xx = np.mgrid[0:256, 0:256]
        checker = (((xx % 4 == 0) ^ (yy % 4 == 0)).astype(np.float32) - 0.5) * 14.0
        mask = (xx >= 72) & (xx <= 184) & (yy >= 64) & (yy <= 192)
        for c in range(3):
            arr[:, :, c] = np.where(mask, arr[:, :, c] + checker, arr[:, :, c])
        # Simulate mismatched spliced inner face compression
        spliced = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        buf = io.BytesIO()
        spliced.crop((72, 64, 184, 192)).save(buf, format="JPEG", quality=45)
        buf.seek(0)
        patch = Image.open(buf).convert("RGB")
        spliced.paste(patch, (72, 64))
        spliced.save(fake_path)

    return {
        "authentic": {
            "id": "authentic",
            "label": "Sample Portrait (Natural)",
            "path": str(real_path),
        },
        "synthetic": {
            "id": "synthetic",
            "label": "Sample Splice + GAN Grid",
            "path": str(fake_path),
        },
    }


@app.on_event("startup")
def startup_event() -> None:
    _ensure_demo_samples()
    get_engine()


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
def api_status() -> JSONResponse:
    det = get_engine()
    return JSONResponse(det.get_status())


@app.post("/api/reload")
def api_reload() -> JSONResponse:
    det = get_engine()
    det.reload_models()
    return JSONResponse(det.get_status())


@app.get("/api/samples")
def api_samples() -> JSONResponse:
    samples = _ensure_demo_samples()
    return JSONResponse([{"id": v["id"], "label": v["label"]} for v in samples.values()])


@app.post("/api/analyze/sample/{sample_id}")
def api_analyze_sample(sample_id: str, mode: str = "ensemble") -> JSONResponse:
    samples = _ensure_demo_samples()
    if sample_id not in samples:
        raise HTTPException(status_code=404, detail=f"Unknown sample '{sample_id}'")
    det = get_engine()
    with Image.open(samples[sample_id]["path"]) as img:
        result = det.analyze_image(img, mode=mode)
    result["filename"] = Path(samples[sample_id]["path"]).name
    result["media_type"] = "image"
    return JSONResponse(result)


@app.post("/api/analyze/image")
async def api_analyze_image(
    file: UploadFile = File(...),
    mode: str = Form("ensemble"),
) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing image file.")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")
    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {exc}") from exc

    det = get_engine()
    result = det.analyze_image(pil_img, mode=mode)
    result["filename"] = file.filename
    result["media_type"] = "image"
    return JSONResponse(result)


@app.post("/api/analyze/video")
async def api_analyze_video(
    file: UploadFile = File(...),
    mode: str = Form("ensemble"),
) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing video file.")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded video file is empty.")

    suffix = Path(file.filename).suffix or ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        det = get_engine()
        result = det.analyze_video(tmp_path, max_frames=16, mode=mode)
        result["filename"] = file.filename
        return JSONResponse(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
