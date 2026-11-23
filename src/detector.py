"""Unified Multi-Domain Deepfake Detection Engine.

Combines 3 complementary specialists on CUDA (RTX 3050):
1. CommunityForensics-DeepfakeDet-ViT (2.7M samples, 4,800+ AI generators, 95% zero-shot acc)
2. prithivMLmods/deepfake-detector-model-v1 (SigLIP v1, ultra-low 2.5% Real FPR anchor)
3. Local Custom DeepfakeEfficientNet + Radial 2D-FFT (calibrated at 96.28% on local dataset)

Plus an on-demand dedicated Face-Swap specialist mode:
4. Wvolf/ViT_Deepfake_Detection (190k Face-Swap ViT on tight face crop)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
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

COMMUNITY_HF_MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
SIGLIP_HF_MODEL_ID = "prithivMLmods/deepfake-detector-model-v1"
WVOLF_HF_MODEL_ID = "Wvolf/ViT_Deepfake_Detection"


def _resolve_hf_indices(id2label: Dict[Any, Any]) -> Tuple[int, int]:
    """Resolves real vs fake label indices explicitly from HuggingFace config."""
    fake_idx, real_idx = 1, 0
    for idx, lbl in id2label.items():
        low = str(lbl).lower()
        if any(w in low for w in ("fake", "deepfake", "synthetic", "artificial", "ai")):
            fake_idx = int(idx)
        elif any(w in low for w in ("real", "realism", "authentic", "human")):
            real_idx = int(idx)
    return fake_idx, real_idx


@dataclass
class HFSpecialist:
    key: str
    display_name: str
    model_id: str
    processor: Any
    model: Any
    fake_idx: int
    real_idx: int
    is_single_logit: bool


class DeepfakeDetectionEngine:
    """Orchestrates face extraction, 3-model ensemble, Face-Swap specialist mode, Grad-CAM, and forensics."""

    def __init__(self, checkpoint_path: Optional[Path] = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path or (
            Path(__file__).resolve().parent.parent / "checkpoints" / "best_model.pt"
        )

        self.face_extractor = FaceExtractor(margin=0.22)
        self.explainer = GradCAMExplainer(self.device)
        self.tta_transforms = get_tta_transforms(IMG_SIZE)
        self.eval_transform = get_eval_transforms(IMG_SIZE)

        self.custom_model: Optional[DeepfakeEfficientNet] = None
        self.custom_ckpt_meta: Dict[str, Any] = {}
        self.fake_threshold: float = 0.5
        self.local_display_name: str = "Local EfficientNet-B0 + FFT"

        self.community_spec: Optional[HFSpecialist] = None
        self.siglip_spec: Optional[HFSpecialist] = None
        self.wvolf_spec: Optional[HFSpecialist] = None

        self.active_engine: str = "uninitialized"
        self.load_error: Optional[str] = None

        self.reload_models()

    def _load_hf_specialist(self, key: str, display_name: str, model_id: str) -> Optional[HFSpecialist]:
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            try:
                proc = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
                model = AutoModelForImageClassification.from_pretrained(model_id, local_files_only=True)
            except Exception:
                proc = AutoImageProcessor.from_pretrained(model_id)
                model = AutoModelForImageClassification.from_pretrained(model_id)

            model = model.to(self.device)
            if self.device.type == "cuda":
                model = model.half()
            model.eval()

            num_labels = int(getattr(model.config, "num_labels", 2))
            is_single_logit = num_labels == 1
            id2label = getattr(model.config, "id2label", {0: "Real", 1: "Fake"})
            fake_idx, real_idx = (1, 0) if is_single_logit else _resolve_hf_indices(id2label)
            return HFSpecialist(
                key=key,
                display_name=display_name,
                model_id=model_id,
                processor=proc,
                model=model,
                fake_idx=fake_idx,
                real_idx=real_idx,
                is_single_logit=is_single_logit,
            )
        except Exception as exc:
            msg = f"Failed to load {model_id}: {exc}"
            self.load_error = f"{self.load_error}; {msg}" if self.load_error else msg
            return None

    def _ensure_wvolf_loaded(self) -> Optional[HFSpecialist]:
        if self.wvolf_spec is None:
            self.wvolf_spec = self._load_hf_specialist(
                key="wvolf",
                display_name="Face-Swap Specialist (Wvolf 190k ViT)",
                model_id=WVOLF_HF_MODEL_ID,
            )
        return self.wvolf_spec

    def reload_models(self) -> None:
        """Loads CommunityForensics 2.7M, SigLIP v1, and Local EfficientNet; keeps Wvolf lazy."""
        self.load_error = None
        self.custom_model = None
        self.custom_ckpt_meta = {}
        self.fake_threshold = 0.5

        if self.checkpoint_path.exists():
            try:
                self.custom_model, self.custom_ckpt_meta = load_checkpoint(
                    self.checkpoint_path, self.device
                )
                thr_path = self.checkpoint_path.parent / "threshold.json"
                if thr_path.exists():
                    thr_meta = json.loads(thr_path.read_text(encoding="utf-8"))
                    self.fake_threshold = float(thr_meta.get("fake_threshold", 0.5))
            except Exception as exc:
                self.load_error = f"Failed to load local checkpoint: {exc}"
                self.custom_model = None

        if self.community_spec is None:
            self.community_spec = self._load_hf_specialist(
                key="community",
                display_name="CommunityForensics ViT (2.7M Generators)",
                model_id=COMMUNITY_HF_MODEL_ID,
            )
        if self.siglip_spec is None:
            self.siglip_spec = self._load_hf_specialist(
                key="siglip",
                display_name="SigLIP v1 Low-FPR Detector",
                model_id=SIGLIP_HF_MODEL_ID,
            )

        has_comm = self.community_spec is not None
        has_siglip = self.siglip_spec is not None
        has_local = self.custom_model is not None

        if has_comm and has_siglip and has_local:
            self.active_engine = "3-Model Ensemble (CommunityForensics 2.7M + SigLIP v1 + Local EfficientNet)"
        elif has_comm and has_local:
            self.active_engine = "Hybrid Ensemble (CommunityForensics 2.7M + Local EfficientNet)"
        elif has_comm and has_siglip:
            self.active_engine = "Pretrained Ensemble (CommunityForensics 2.7M + SigLIP v1)"
        elif has_comm:
            self.active_engine = "CommunityForensics ViT (2.7M)"
        elif has_local:
            arch = self.custom_ckpt_meta.get("arch", "efficientnet_b0")
            self.active_engine = f"Local Custom Model ({arch})"
        else:
            self.active_engine = "Forensic Fallback (Neural Models Unavailable)"

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
            "specialists": {
                "community": self.community_spec is not None,
                "siglip": self.siglip_spec is not None,
                "local": self.custom_model is not None,
                "wvolf_cached": self.wvolf_spec is not None,
            },
            "load_error": self.load_error,
        }

    def _predict_custom_tta_score(self, cropped_pil: Image.Image) -> float:
        assert self.custom_model is not None
        tensors = torch.stack([t(cropped_pil) for t in self.tta_transforms]).to(self.device)
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                logits = self.custom_model(tensors)
                probs = F.softmax(logits.float(), dim=1)
        raw_fake = float(probs.mean(dim=0)[1].item())
        t = self.fake_threshold
        if raw_fake < t:
            fake_prob = 0.5 * (raw_fake / max(t, 1e-6))
        else:
            fake_prob = 0.5 + 0.5 * ((raw_fake - t) / max(1.0 - t, 1e-6))
        return float(np.clip(fake_prob, 0.0, 1.0))

    def _predict_hf_2tta_score(self, spec: HFSpecialist, pil_crop: Image.Image) -> float:
        rgb = pil_crop.convert("RGB")
        views = [rgb, ImageOps.mirror(rgb)]
        inputs = spec.processor(images=views, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        if self.device.type == "cuda":
            pixel_values = pixel_values.half()

        with torch.no_grad():
            outputs = spec.model(pixel_values=pixel_values)
            logits = outputs.logits.float()
            if spec.is_single_logit:
                fake_probs = torch.sigmoid(logits[:, 0])
                fake_prob = float(fake_probs.mean().item())
            else:
                probs = F.softmax(logits, dim=1)
                mean_probs = probs.mean(dim=0)
                real_p = float(mean_probs[spec.real_idx].item())
                fake_p = float(mean_probs[spec.fake_idx].item())
                total = real_p + fake_p
                fake_prob = fake_p / total if total > 0 else fake_p
        return float(np.clip(fake_prob, 0.0, 1.0))

    def _explain_hf_specialist(self, spec: HFSpecialist, pil_crop: Image.Image, is_fake: bool) -> str:
        target_class = 1 if spec.is_single_logit and is_fake else (0 if spec.is_single_logit else (spec.fake_idx if is_fake else spec.real_idx))
        inputs = spec.processor(images=pil_crop.convert("RGB"), return_tensors="pt")
        pv = inputs["pixel_values"].to(self.device)
        return self.explainer.explain_hf_vision(spec.model, pv, pil_crop, target_class=target_class)

    def _explain_local(self, cropped_pil: Image.Image, is_fake: bool) -> str:
        assert self.custom_model is not None
        target_class = 1 if is_fake else 0
        single_tensor = self.eval_transform(cropped_pil).unsqueeze(0).to(self.device)
        return self.explainer.explain_cnn(
            self.custom_model, single_tensor, cropped_pil, target_class=target_class
        )

    def analyze_image(
        self,
        pil_img: Image.Image,
        mode: str = "ensemble",
        include_visuals: bool = True,
    ) -> Dict[str, Any]:
        """Full dual-crop face extraction + specialist ensemble + smart Grad-CAM + forensics."""
        norm_mode = (mode or "ensemble").strip().lower()
        if norm_mode not in {"ensemble", "faceswap", "community", "local"}:
            norm_mode = "ensemble"

        face_res = self.face_extractor.extract(pil_img)
        cropped = face_res.cropped_pil
        tight = face_res.tight_face_pil

        p_comm: Optional[float] = None
        p_siglip: Optional[float] = None
        p_local: Optional[float] = None
        p_wvolf: Optional[float] = None
        model_votes: List[Dict[str, Any]] = []

        if norm_mode == "faceswap":
            wvolf = self._ensure_wvolf_loaded()
            if wvolf is None:
                raise RuntimeError(self.load_error or "Face-Swap specialist (Wvolf 190k ViT) could not be loaded.")
            p_wvolf = self._predict_hf_2tta_score(wvolf, tight)
            model_votes.append({
                "name": wvolf.display_name,
                "fake_probability": round(p_wvolf, 4),
            })
            fake_prob = p_wvolf
            engine_label = wvolf.display_name
        elif norm_mode == "community":
            if self.community_spec is None:
                raise RuntimeError("CommunityForensics ViT model is not available.")
            p_comm = self._predict_hf_2tta_score(self.community_spec, cropped)
            model_votes.append({
                "name": self.community_spec.display_name,
                "fake_probability": round(p_comm, 4),
            })
            fake_prob = p_comm
            engine_label = self.community_spec.display_name
        elif norm_mode == "local":
            if self.custom_model is None:
                raise RuntimeError("Local EfficientNet checkpoint is not available.")
            p_local = self._predict_custom_tta_score(cropped)
            model_votes.append({
                "name": self.local_display_name,
                "fake_probability": round(p_local, 4),
            })
            fake_prob = p_local
            engine_label = f"Local Only ({self.local_display_name})"
        else:
            # Default: 3-Model Ensemble (CommunityForensics 2.7M + SigLIP v1 + Local EfficientNet)
            if self.community_spec is not None:
                p_comm = self._predict_hf_2tta_score(self.community_spec, cropped)
                model_votes.append({
                    "name": self.community_spec.display_name,
                    "fake_probability": round(p_comm, 4),
                })
            if self.siglip_spec is not None:
                p_siglip = self._predict_hf_2tta_score(self.siglip_spec, cropped)
                model_votes.append({
                    "name": self.siglip_spec.display_name,
                    "fake_probability": round(p_siglip, 4),
                })
            if self.custom_model is not None:
                p_local = self._predict_custom_tta_score(cropped)
                model_votes.append({
                    "name": self.local_display_name,
                    "fake_probability": round(p_local, 4),
                })

            if p_comm is not None and p_siglip is not None and p_local is not None:
                weighted = 0.50 * p_comm + 0.35 * p_local + 0.15 * p_siglip
                if p_comm >= 0.85:
                    fake_prob = max(weighted, 0.80 * p_comm + 0.20 * p_siglip)
                elif p_local >= 0.85 and p_comm >= 0.40:
                    fake_prob = max(weighted, 0.70 * p_local + 0.30 * p_comm)
                else:
                    fake_prob = weighted
                engine_label = self.active_engine
            elif p_comm is not None and p_local is not None:
                fake_prob = 0.55 * p_comm + 0.45 * p_local
                engine_label = self.active_engine
            elif p_comm is not None and p_siglip is not None:
                fake_prob = 0.75 * p_comm + 0.25 * p_siglip
                engine_label = self.active_engine
            elif p_comm is not None:
                fake_prob = p_comm
                engine_label = self.active_engine
            elif p_local is not None:
                fake_prob = p_local
                engine_label = self.active_engine
            else:
                raise RuntimeError(self.load_error or "No deepfake detection models are loaded.")

        fake_prob = float(np.clip(fake_prob, 0.0, 1.0))
        real_prob = 1.0 - fake_prob
        verdict = "Fake" if fake_prob >= 0.5 else "Real"
        confidence = max(fake_prob, real_prob)

        visuals: Dict[str, str] = {}
        forensics_scores: Dict[str, float] = {}
        if include_visuals:
            is_fake = verdict == "Fake"
            if norm_mode == "faceswap" and self.wvolf_spec is not None:
                gradcam_b64 = self._explain_hf_specialist(self.wvolf_spec, tight, is_fake=is_fake)
            elif (
                norm_mode == "community"
                or p_local is None
                or (p_comm is not None and p_comm >= p_local)
            ) and self.community_spec is not None:
                gradcam_b64 = self._explain_hf_specialist(self.community_spec, cropped, is_fake=is_fake)
            elif self.custom_model is not None:
                gradcam_b64 = self._explain_local(cropped, is_fake=is_fake)
            else:
                gradcam_b64 = face_res.cropped_face_b64

            forensics = run_forensic_suite(cropped)
            forensics_scores = {
                "neural_fake_score": round(fake_prob, 4),
                "fft_spectral_score": forensics["fft_score"],
                "ela_compression_score": forensics["ela_score"],
                "seam_consistency_score": forensics["seam_score"],
            }
            visuals = {
                "annotated_preview_b64": face_res.annotated_preview_b64,
                "cropped_face_b64": face_res.cropped_face_b64,
                "gradcam_b64": gradcam_b64,
                "ela_map_b64": forensics["ela_map_b64"],
                "fft_spectrum_b64": forensics["fft_spectrum_b64"],
            }

        return {
            "verdict": verdict,
            "is_fake": verdict == "Fake",
            "confidence": round(confidence, 4),
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(real_prob, 4),
            "mode": norm_mode,
            "engine": engine_label,
            "model_votes": model_votes,
            "face_detected": face_res.face_detected,
            "already_tight_crop": face_res.already_tight_crop,
            "bbox": list(face_res.bbox) if face_res.bbox else None,
            "forensics": forensics_scores,
            "visuals": visuals,
        }

    def analyze_video(
        self,
        video_path: str | os.PathLike,
        max_frames: int = 16,
        mode: str = "ensemble",
    ) -> Dict[str, Any]:
        """Samples evenly spaced video frames, runs specialist ensemble + face crop, and aggregates verdict."""
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
        best_frame_pil: Optional[Image.Image] = None
        best_timestamp_sec = 0.0
        best_frame_idx = 0
        highest_fake_prob = -1.0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            res = self.analyze_image(pil_frame, mode=mode, include_visuals=False)
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
                best_frame_pil = pil_frame
                best_timestamp_sec = timestamp_sec
                best_frame_idx = int(idx)

        cap.release()
        if not frame_results or best_frame_pil is None:
            raise ValueError("No frames could be extracted from the uploaded video.")

        best_suspicious_result = self.analyze_image(best_frame_pil, mode=mode, include_visuals=True)
        best_suspicious_result["peak_timestamp_sec"] = best_timestamp_sec
        best_suspicious_result["peak_frame_index"] = best_frame_idx

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
            "mode": best_suspicious_result["mode"],
            "frames_analyzed": len(frame_results),
            "duration_sec": round(total_frames / max(fps, 1.0), 2),
            "engine": best_suspicious_result["engine"],
            "model_votes": best_suspicious_result["model_votes"],
            "timeline": frame_results,
            "peak_frame": best_suspicious_result,
        }
