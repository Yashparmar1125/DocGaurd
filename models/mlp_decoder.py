"""All-MLP Segmentation Decoder (SegFormer Design)

Replaces heavy deconvolutional decoders with an efficient All-MLP decoder:
1. Multi-scale feature projection to a uniform embedding dimension (C = 256)
2. Bilinear upsampling of all stages to 1/4th input resolution (128x128)
3. Feature concatenation and MLP fusion
4. Linear classification head upsampled to 512x512
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AllMLPLocalizationDecoder(nn.Module):
    """Efficient multi-scale MLP decoder for pixel-accurate forgery segmentation."""

    def __init__(
        self,
        in_channels: list = [96, 192, 384, 512],
        embed_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 4 Projection layers for the 4 hierarchical stages
        self.proj0 = MLPBlock(in_channels[0], embed_dim)  # c0: 96 -> 256
        self.proj1 = MLPBlock(in_channels[1], embed_dim)  # c1: 192 -> 256
        self.proj2 = MLPBlock(in_channels[2], embed_dim)  # c2: 384 -> 256
        self.proj3 = MLPBlock(in_channels[3], embed_dim)  # fused bottleneck: 512 -> 256

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

        # Final 1x1 mask classifier
        self.classifier = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(
        self, fused_bottleneck: torch.Tensor, skips: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Decodes multi-scale features to 512x512 mask logits.

        Args:
            fused_bottleneck: (B, 512, 16, 16)
            skips: dict with 'c0' (B, 96, 128, 128),
                             'c1' (B, 192, 64, 64),
                             'c2' (B, 384, 32, 32)

        Returns:
            torch.Tensor: (B, 1, 512, 512) logits
        """
        c0 = skips["c0"]  # (B, 96, 128, 128)
        c1 = skips["c1"]  # (B, 192, 64, 64)
        c2 = skips["c2"]  # (B, 384, 32, 32)
        c3 = fused_bottleneck  # (B, 512, 16, 16)

        target_size = c0.shape[2:]  # (128, 128)

        # 1. Project all stages to embed_dim
        p0 = self.proj0(c0)
        p1 = F.interpolate(self.proj1(c1), size=target_size, mode="bilinear", align_corners=False)
        p2 = F.interpolate(self.proj2(c2), size=target_size, mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.proj3(c3), size=target_size, mode="bilinear", align_corners=False)

        # 2. Concatenate and fuse
        fused = self.fusion(torch.cat([p0, p1, p2, p3], dim=1))  # (B, 256, 128, 128)

        # 3. Classify and upsample to 512x512
        mask_128 = self.classifier(fused)  # (B, 1, 128, 128)
        mask_logits = F.interpolate(mask_128, size=(512, 512), mode="bilinear", align_corners=False)

        return mask_logits
