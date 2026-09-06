"""Grad-CAM Implementation for DocGuard

Generates gradient-weighted class activation heatmaps on the final convolutional
layers of the RGB backbone to explain deep model attribution.
"""

from typing import Tuple
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Computes Grad-CAM heatmap overlays for DocGuard."""

    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        self.model = model
        self.target_layer = target_layer or model.rgb_stream.layer4
        self.gradients = None
        self.activations = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(
        self, rgb_tensor: torch.Tensor, dct_tensor: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes Grad-CAM activation map for the binary forgery classifier.

        Args:
            rgb_tensor: (1, 3, 512, 512)
            dct_tensor: (1, 21, 512, 512)

        Returns:
            tuple: (heatmap_norm (512, 512) float in [0, 1], colormap_rgb (512, 512, 3) uint8)
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        outputs = self.model(rgb_tensor, dct_tensor)
        binary_logit = outputs["binary_logits"]

        # Backward pass on binary classification score
        binary_logit.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback uniform map if hooks didn't capture
            heatmap = np.zeros((512, 512), dtype=np.float32)
            return heatmap, np.zeros((512, 512, 3), dtype=np.uint8)

        # Global average pooling of gradients: weights alpha_k
        alpha = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of forward activation maps
        cam = torch.sum(alpha * self.activations, dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Upsample to 512x512
        cam = F.interpolate(cam, size=(512, 512), mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = np.min(cam_np), np.max(cam_np)
        if cam_max - cam_min > 1e-6:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        # Generate Jet colormap overlay
        cam_uint8 = (cam_norm * 255).astype(np.uint8)
        cam_color = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        cam_color_rgb = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)

        return cam_norm, cam_color_rgb

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
