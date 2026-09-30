"""Swin Transformer Backbone for Document Image Forgery Analysis

Implements a Hierarchical Vision Transformer (Swin-T) with Shifted Windows,
producing multi-scale spatial representations at strides [4, 8, 16, 32].
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import timm


class SwinRGBBackbone(nn.Module):
    """Hierarchical Swin Transformer backbone with multi-scale skip stages."""

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = False,
        img_size: int = 512,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            img_size=img_size,
        )
        # Stage channels for swin_tiny: [96, 192, 384, 768]
        self.stage_channels = [96, 192, 384, 768]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Extracts hierarchical features from RGB document.

        Args:
            x: Input RGB tensor (B, 3, 512, 512)

        Returns:
            bottleneck: (B, 768, 16, 16)
            skips: dict with 'c0' (B, 96, 128, 128),
                             'c1' (B, 192, 64, 64),
                             'c2' (B, 384, 32, 32)
        """
        feats = self.encoder(x)
        # timm Swin outputs in channels-last: (B, H, W, C) -> permute to (B, C, H, W)
        c0 = feats[0].permute(0, 3, 1, 2).contiguous()  # (B, 96, 128, 128)
        c1 = feats[1].permute(0, 3, 1, 2).contiguous()  # (B, 192, 64, 64)
        c2 = feats[2].permute(0, 3, 1, 2).contiguous()  # (B, 384, 32, 32)
        c3 = feats[3].permute(0, 3, 1, 2).contiguous()  # (B, 768, 16, 16)

        skips = {
            "c0": c0,
            "c1": c1,
            "c2": c2,
        }
        return c3, skips
