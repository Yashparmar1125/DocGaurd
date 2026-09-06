"""U-Net Segmentation Decoder for Pixel-Level Forgery Localization

Upsamples bottleneck fused representations through 5 progressive deconvolution/interpolation
stages with skip connections from RGB hierarchical features, generating a full 512x512
probability mask.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetLocalizationDecoder(nn.Module):
    """Progressive decoder producing 512x512 pixel-level forgery probability maps."""

    def __init__(self, bottleneck_ch: int = 512):
        super().__init__()
        # Stage 1: 16x16 -> 32x32, skip c3 (1024 channels)
        self.up1 = DecoderBlock(in_ch=bottleneck_ch, skip_ch=1024, out_ch=256)
        # Stage 2: 32x32 -> 64x64, skip c2 (512 channels)
        self.up2 = DecoderBlock(in_ch=256, skip_ch=512, out_ch=128)
        # Stage 3: 64x64 -> 128x128, skip c1 (256 channels)
        self.up3 = DecoderBlock(in_ch=128, skip_ch=256, out_ch=64)
        # Stage 4: 128x128 -> 256x256, skip c0 (64 channels)
        self.up4 = DecoderBlock(in_ch=64, skip_ch=64, out_ch=32)

        # Stage 5: 256x256 -> 512x512 (no skip at raw input scale)
        self.final_up = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, bottleneck: torch.Tensor, skips: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Upsamples features to full-resolution tampering mask logits.

        Args:
            bottleneck: (B, 512, 16, 16)
            skips: dict with 'c3', 'c2', 'c1', 'c0'

        Returns:
            torch.Tensor: Mask logits of shape (B, 1, 512, 512)
        """
        x = self.up1(bottleneck, skips["c3"])  # (B, 256, 32, 32)
        x = self.up2(x, skips["c2"])           # (B, 128, 64, 64)
        x = self.up3(x, skips["c1"])           # (B, 64, 128, 128)
        x = self.up4(x, skips["c0"])           # (B, 32, 256, 256)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # (B, 32, 512, 512)
        logits = self.final_up(x)               # (B, 1, 512, 512)
        return logits
