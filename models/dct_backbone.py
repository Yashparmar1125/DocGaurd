"""Forensic Stream Backbone (DCT Frequency Grid Encoder)

Processes 8x8 block DCT coefficient maps on the luminance channel through convolutional
and pooling blocks to capture compression grid discontinuities and double-JPEG traces.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DCTForensicBackbone(nn.Module):
    """Convolutional encoder extracting double-compression features from 21-ch DCT maps."""

    def __init__(self, in_channels: int = 21, out_channels: int = 512):
        super().__init__()
        # Input: (B, 21, 512, 512)
        # Stage 1: 512 -> 256
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        # Stage 2: 256 -> 128
        self.block1 = ConvBlock(64, 128, stride=2)
        # Stage 3: 128 -> 64
        self.block2 = ConvBlock(128, 256, stride=2)
        # Stage 4: 64 -> 32
        self.block3 = ConvBlock(256, 384, stride=2)
        # Stage 5: 32 -> 16
        self.block4 = ConvBlock(384, out_channels, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes DCT map to bottleneck dimensions.

        Args:
            x: DCT tensor (B, 21, 512, 512)

        Returns:
            torch.Tensor: (B, 512, 16, 16)
        """
        x = self.stem(x)    # (B, 64, 256, 256)
        x = self.block1(x)  # (B, 128, 128, 128)
        x = self.block2(x)  # (B, 256, 64, 64)
        x = self.block3(x)  # (B, 384, 32, 32)
        x = self.block4(x)  # (B, 512, 16, 16)
        return x
