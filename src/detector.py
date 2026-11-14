"""Unified Deepfake Detection Engine with 4x TTA, Local Checkpoint Priority, and Pretrained 96%+ HF Fallback."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from .face_pipeline import FaceExtractor
from .forensics import run_forensic_suite
from .gradcam import GradCAMExplainer
from .model import DeepfakeEfficientNet, load_checkpoint
from .transforms import IMG_SIZE, get_eval_transforms, get_tta_transforms

PRETRAINED_HF_MODEL_ID = "dima806/deepfake_vs_real_image_detection"


def _create_hf_tta_views(pil_img: Image.Image) -> List[Image.Image]:
    """Generates 4 deterministic spatial views for Test-Time Augmentation."""
    rgb = pil_img.convert("RGB")
    w, h = rgb.size
    crop_w, crop_h = max(1, int(round(w * 0.92))), max(1, int(round(h * 0.92)))
    left = max(0, (w - crop_w) // 2)
    top = max(0, (h - crop_h) // 2)
    tight = rgb.crop((left, top, left + crop_w, top + crop_h))
    return [
        rgb,
        ImageOps.mirror(rgb),
        tight,
        ImageOps.mirror(tight),
    ]


class DeepfakeDetectionEngine:
    """Orchestrates face extraction, neural TTA inference, Grad-CAM, and spectral forensics."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        hf_model_id: str = PRETRAINED_HF_MODEL_ID,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path or (
            Path(__file__).resolve().parent.parent / "checkpoints" / "best_model.pt"
        )
        self.hf_model_id = hf_model_id

        self.face_extractor = FaceExtractor(margin=0.22)
        self.explainer = GradCAMExplainer(self.device)
        self.tta_transforms = get_tta_transforms(IMG_SIZE)
        self.eval_transform = get_eval_transforms(IMG_SIZE)

        self.custom_model: Optional[DeepfakeEfficientNet] = None
        self.custom_ckpt_meta: Dict[str, Any] = {}
        self.hf_processor = None
        self.hf_model = None
        self.hf_fake_idx: int = 1
        self.hf_real_idx: int = 0
        self.active_engine: str = "uninitialized"
        self.load_error: Optional[str] = None
        self.fake_threshold: float = 0.5

        self.reload_models()

    def reload_models(self) -> None:
        """Prioritizes locally trained checkpoint; loads pretrained 96%+ HF model otherwise."""
        self.load_error = None
        if self.checkpoint_path.exists():
            try:
                self.custom_model, self.custom_ckpt_meta = load_checkpoint(
                    self.checkpoint_path, self.device
                )
                thr_path = self.checkpoint_path.parent / "threshold.json"
                self.fake_threshold = 0.5
                thr_meta: Dict[str, Any] = {}
                if thr_path.exists():
                    import json
                    thr_meta = json.loads(thr_path.read_text(encoding="utf-8"))
                    self.fake_threshold = float(thr_meta.get("fake_threshold", 0.5))
                arch = self.custom_ckpt_meta.get("arch", "efficientnet_b0")
                test_acc = thr_meta.get("test_acc")
                val_acc = thr_meta.get("val_acc") or self.custom_ckpt_meta.get("metrics", {}).get("val_acc")
                if isinstance(test_acc, (int, float)):
                    acc_str = f" ({test_acc * 100:.1f}% test acc)"
                elif isinstance(val_acc, (int, float)):
                    acc_str = f" ({val_acc * 100:.1f}% val acc)"
                else:
                    acc_str = ""
                self.active_engine = f"Local Custom Model ({arch}){acc_str}"
                return
            except Exception as exc:
                self.load_error = f"Failed to load local checkpoint: {exc}"
                self.custom_model = None

        # Load ready-to-use >96% accuracy HuggingFace deepfake detector
        self._ensure_hf_loaded()

    def _ensure_hf_loaded(self) -> None:
        if self.hf_model is not None:
            return
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            try:
                self.hf_processor = AutoImageProcessor.from_pretrained(
                    self.hf_model_id, local_files_only=True
                )
                self.hf_model = AutoModelForImageClassification.from_pretrained(
                    self.hf_model_id, local_files_only=True
                )
            except Exception:
                self.hf_processor = AutoImageProcessor.from_pretrained(self.hf_model_id)
                self.hf_model = AutoModelForImageClassification.from_pretrained(self.hf_model_id)
            self.hf_model.to(self.device)
            self.hf_model.eval()

            # Resolve real vs fake label indices explicitly from config
            id2label = getattr(self.hf_model.config, "id2label", {0: "Real", 1: "Fake"})
            fake_idx, real_idx = 1, 0
            for idx, lbl in id2label.items():
                low = str(lbl).lower()
                if "fake" in low or "deepfake" in low or "synthetic" in low:
                    fake_idx = int(idx)
                elif "real" in low or "authentic" in low:
                    real_idx = int(idx)
            self.hf_fake_idx = fake_idx
            self.hf_real_idx = real_idx
            self.active_engine = f"Pretrained ViT Deepfake Detector ({self.hf_model_id})"
        except Exception as exc:
            self.load_error = f"Could not load pretrained model '{self.hf_model_id}': {exc}"
            self.active_engine = "Forensic Fallback (Neural Model Unavailable)"

    def get_status(self) -> Dict[str, Any]:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Only"
        vram_gb = (
            round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
            if torch.cuda.is_available()
            else 0.0
        )
        return {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": gpu_name,
            "vram_gb": vram_gb,
            "active_engine": self.active_engine,
            "local_checkpoint_exists": self.checkpoint_path.exists(),
            "checkpoint_path": str(self.checkpoint_path),
            "tta_views": len(self.tta_transforms),
            "load_error": self.load_error,
        }

    def _predict_custom_tta(self, cropped_pil: Image.Image) -> Tuple[float, float, str]:
        assert self.custom_model is not None
        tensors = torch.stack([t(cropped_pil) for t in self.tta_transforms]).to(self.device)
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                logits = self.custom_model(tensors)
                probs = F.softmax(logits.float(), dim=1)
        mean_probs = probs.mean(dim=0)
        raw_fake = float(mean_probs[1].item())
        t = self.fake_threshold
        if raw_fake < t:
            fake_prob = 0.5 * (raw_fake / max(t, 1e-6))
        else:
            fake_prob = 0.5 + 0.5 * ((raw_fake - t) / max(1.0 - t, 1e-6))
        fake_prob = float(np.clip(fake_prob, 0.0, 1.0))
        real_prob = 1.0 - fake_prob

        target_class = 1 if fake_prob >= 0.5 else 0
        single_tensor = self.eval_transform(cropped_pil).unsqueeze(0).to(self.device)
        gradcam_b64 = self.explainer.explain_cnn(
            self.custom_model, single_tensor, cropped_pil, target_class=target_class
        )
        return real_prob, fake_prob, gradcam_b64

    def _predict_hf_tta(self, cropped_pil: Image.Image) -> Tuple[float, float, str]:
        if self.hf_model is None or self.hf_processor is None:
            self._ensure_hf_loaded()
        if self.hf_model is None or self.hf_processor is None:
            raise RuntimeError(self.load_error or "Pretrained model could not be loaded.")

        views = _create_hf_tta_views(cropped_pil)
        inputs = self.hf_processor(images=views, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                outputs = self.hf_model(pixel_values=pixel_values)
                probs = F.softmax(outputs.logits.float(), dim=1)

        mean_probs = probs.mean(dim=0)
        real_prob = float(mean_probs[self.hf_real_idx].item())
        fake_prob = float(mean_probs[self.hf_fake_idx].item())
        # Renormalize cleanly in case of rounding
        total = real_prob + fake_prob
        if total > 0:
            real_prob /= total
            fake_prob /= total

        target_class = self.hf_fake_idx if fake_prob >= 0.5 else self.hf_real_idx
        single_inputs = self.hf_processor(images=cropped_pil.convert("RGB"), return_tensors="pt")
        single_pv = single_inputs["pixel_values"].to(self.device)
        gradcam_b64 = self.explainer.explain_hf_vision(
            self.hf_model, single_pv, cropped_pil, target_class=target_class
        )
        return real_prob, fake_prob, gradcam_b64

    def analyze_image(self, pil_img: Image.Image) -> Dict[str, Any]:
        """Full end-to-end face extraction + TTA neural detection + Grad-CAM + forensics."""
        face_res = self.face_extractor.extract(pil_img)
        cropped = face_res.cropped_pil

        if self.custom_model is not None:
            real_prob, fake_prob, gradcam_b64 = self._predict_custom_tta(cropped)
        else:
            real_prob, fake_prob, gradcam_b64 = self._predict_hf_tta(cropped)

        forensics = run_forensic_suite(cropped)

        verdict = "Fake" if fake_prob >= 0.5 else "Real"
        confidence = max(fake_prob, real_prob)

        return {
            "verdict": verdict,
            "is_fake": verdict == "Fake",
            "confidence": round(confidence, 4),
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(real_prob, 4),
            "engine": self.active_engine,
            "face_detected": face_res.face_detected,
            "already_tight_crop": face_res.already_tight_crop,
            "bbox": list(face_res.bbox) if face_res.bbox else None,
            "forensics": {
                "neural_fake_score": round(fake_prob, 4),
                "fft_spectral_score": forensics["fft_score"],
                "ela_compression_score": forensics["ela_score"],
                "seam_consistency_score": forensics["seam_score"],
            },
            "visuals": {
                "annotated_preview_b64": face_res.annotated_preview_b64,
                "cropped_face_b64": face_res.cropped_face_b64,
                "gradcam_b64": gradcam_b64,
                "ela_map_b64": forensics["ela_map_b64"],
                "fft_spectrum_b64": forensics["fft_spectrum_b64"],
            },
        }

    def analyze_video(self, video_path: str | os.PathLike, max_frames: int = 16) -> Dict[str, Any]:
        """Samples evenly spaced video frames, runs TTA + face crop, and aggregates verdict."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Unable to open video file for decoding.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if total_frames <= 0:
            cap.release()
            raise ValueError("Video contains 0 readable frames.")

        num_samples = min(max_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, num=num_samples, dtype=int).tolist()

        frame_results = []
        best_suspicious_result = None
        highest_fake_prob = -1.0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            res = self.analyze_image(pil_frame)
            timestamp_sec = round(idx / max(fps, 1.0), 2)

            frame_results.append({
                "frame_index": int(idx),
                "timestamp_sec": timestamp_sec,
                "fake_probability": res["fake_probability"],
                "real_probability": res["real_probability"],
                "face_detected": res["face_detected"],
                "verdict": res["verdict"],
            })

            if res["fake_probability"] > highest_fake_prob:
                highest_fake_prob = res["fake_probability"]
                best_suspicious_result = res
                best_suspicious_result["peak_timestamp_sec"] = timestamp_sec
                best_suspicious_result["peak_frame_index"] = int(idx)

        cap.release()
        if not frame_results or best_suspicious_result is None:
            raise ValueError("No frames could be extracted from the uploaded video.")

        # Aggregate with top-k suspicious frame weighting (deepfakes often appear in key frames)
        probs = sorted([f["fake_probability"] for f in frame_results], reverse=True)
        top_k = probs[: max(1, len(probs) // 3)]
        agg_fake_prob = 0.6 * float(np.mean(top_k)) + 0.4 * float(np.mean(probs))
        agg_real_prob = 1.0 - agg_fake_prob
        verdict = "Fake" if agg_fake_prob >= 0.5 else "Real"

        return {
            "media_type": "video",
            "verdict": verdict,
            "is_fake": verdict == "Fake",
            "confidence": round(max(agg_fake_prob, agg_real_prob), 4),
            "fake_probability": round(agg_fake_prob, 4),
            "real_probability": round(agg_real_prob, 4),
            "frames_analyzed": len(frame_results),
            "duration_sec": round(total_frames / max(fps, 1.0), 2),
            "engine": self.active_engine,
            "timeline": frame_results,
            "peak_frame": best_suspicious_result,
        }
