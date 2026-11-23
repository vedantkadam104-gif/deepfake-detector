"""Grad-CAM & Vision Transformer Activation Gradient Heatmap Explainer."""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .face_pipeline import pil_to_base64


class GradCAMExplainer:
    """Generates manipulation heatmaps for both CNN (DeepfakeEfficientNet) and ViT models."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @staticmethod
    def _extract_target_score(logits: torch.Tensor, target_class: int) -> torch.Tensor:
        if logits.shape[-1] == 1:
            return logits[0, 0].float() if target_class == 1 else -logits[0, 0].float()
        return logits[0, target_class].float()

    @staticmethod
    def _find_last_conv_layer(module: nn.Module) -> Optional[nn.Module]:
        last_conv = None
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        return last_conv

    @staticmethod
    def _overlay_heatmap(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> str:
        rgb = np.asarray(pil_img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
        cam_resized = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
        cam_resized = np.clip(cam_resized, 0.0, 1.0)
        heatmap_u8 = np.uint8(255 * cam_resized)
        heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        blended = np.uint8((1.0 - alpha) * rgb.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32))
        return pil_to_base64(Image.fromarray(blended))

    def explain_cnn(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        pil_img: Image.Image,
        target_class: int = 1,
        is_hf: bool = False,
    ) -> str:
        """Computes Grad-CAM on the final Conv2d layer of DeepfakeEfficientNet or HF ConvNet."""
        model.eval()
        target_layer = self._find_last_conv_layer(model.backbone if hasattr(model, "backbone") else model)
        if target_layer is None:
            return self._explain_input_saliency(model, input_tensor, pil_img, target_class, is_hf=is_hf)

        activations = []
        gradients = []

        def fwd_hook(_mod, _inp, out):
            activations.append(out.detach())

        def bwd_hook(_mod, _grad_in, grad_out):
            gradients.append(grad_out[0].detach())

        h_fwd = target_layer.register_forward_hook(fwd_hook)
        h_bwd = target_layer.register_full_backward_hook(bwd_hook)

        try:
            model_dtype = next(model.parameters()).dtype
            x = input_tensor.clone().detach().to(device=self.device, dtype=model_dtype).requires_grad_(True)
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                out = model(pixel_values=x) if is_hf else model(x)
                logits = out.logits if hasattr(out, "logits") else out
                score = self._extract_target_score(logits, target_class)
                score.backward()

            if not activations or not gradients:
                return self._explain_input_saliency(model, input_tensor, pil_img, target_class, is_hf=is_hf)

            act = activations[0][0].float()  # [C, H, W]
            grad = gradients[0][0].float()   # [C, H, W]
            if act.ndim != 3:
                return self._explain_input_saliency(model, input_tensor, pil_img, target_class, is_hf=is_hf)
            weights = grad.mean(dim=(1, 2), keepdim=True)
            cam = torch.relu((weights * act).sum(dim=0)).cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
            return self._overlay_heatmap(pil_img, cam)
        finally:
            h_fwd.remove()
            h_bwd.remove()
            model.zero_grad(set_to_none=True)

    def explain_hf_vision(
        self,
        hf_model: nn.Module,
        pixel_values: torch.Tensor,
        pil_img: Image.Image,
        target_class: int,
    ) -> str:
        """Computes token-gradient or conv Grad-CAM heatmap for HuggingFace vision models."""
        hf_model.eval()
        model_dtype = next(hf_model.parameters()).dtype
        # Check if ViT-style encoder exists
        if hasattr(hf_model, "vit") and hasattr(hf_model.vit, "encoder"):
            target_layer = getattr(hf_model.vit, "layernorm", None) or hf_model.vit.encoder.layer[-1].layernorm_before
            activations = []
            gradients = []

            def fwd_hook(_mod, _inp, out):
                activations.append(out.detach())

            def bwd_hook(_mod, _grad_in, grad_out):
                if grad_out and grad_out[0] is not None:
                    gradients.append(grad_out[0].detach())

            h_fwd = target_layer.register_forward_hook(fwd_hook)
            h_bwd = target_layer.register_full_backward_hook(bwd_hook)
            try:
                x = pixel_values.clone().detach().to(device=self.device, dtype=model_dtype).requires_grad_(True)
                hf_model.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    out = hf_model(pixel_values=x)
                    logits = out.logits if hasattr(out, "logits") else out
                    score = self._extract_target_score(logits, target_class)
                    score.backward()

                if activations and gradients:
                    # [1, 1 + num_patches, hidden] -> drop CLS token
                    act = activations[0][0, 1:, :].float()
                    grad = gradients[0][0, 1:, :].float()
                    weights = torch.relu((act * grad).sum(dim=-1))
                    n_patches = weights.shape[0]
                    grid = int(round(math.sqrt(n_patches)))
                    if grid * grid == n_patches:
                        cam = weights.reshape(grid, grid).cpu().numpy()
                        cam = cv2.GaussianBlur(cam, (3, 3), 0)
                        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
                        return self._overlay_heatmap(pil_img, cam)
            finally:
                h_fwd.remove()
                h_bwd.remove()
                hf_model.zero_grad(set_to_none=True)

        # Fallback to last Conv2d or smooth input gradient saliency
        conv = self._find_last_conv_layer(hf_model)
        if conv is not None:
            return self.explain_cnn(hf_model, pixel_values, pil_img, target_class, is_hf=True)
        return self._explain_input_saliency(hf_model, pixel_values, pil_img, target_class, is_hf=True)

    def _explain_input_saliency(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        pil_img: Image.Image,
        target_class: int,
        is_hf: bool = False,
    ) -> str:
        model_dtype = next(model.parameters()).dtype
        x = input_tensor.clone().detach().to(device=self.device, dtype=model_dtype).requires_grad_(True)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            out = model(pixel_values=x).logits if is_hf else model(x)
            score = self._extract_target_score(out, target_class)
            score.backward()
        sal = x.grad[0].float().abs().max(dim=0)[0].cpu().numpy()
        sal = cv2.GaussianBlur(sal, (15, 15), 0)
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-6)
        model.zero_grad(set_to_none=True)
        return self._overlay_heatmap(pil_img, sal)
