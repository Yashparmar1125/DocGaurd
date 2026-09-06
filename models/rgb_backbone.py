"""RGB Stream Backbone (ResNet-50 with Multi-Scale Skip Connections)

Extracts hierarchical visual and semantic feature maps for downstream cross-attention
fusion and U-Net skip-connection decoders.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class RGBResNetBackbone(nn.Module):
    """ResNet-50 feature extractor exposing multi-scale intermediate activations."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        try:
            base = resnet50(weights=weights)
        except Exception:
            # Offline fallback
            base = resnet50(weights=None)

        # Stage 0: Stem (stride 4, 64 channels)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        # Stage 1: layer1 (stride 4, 256 channels)
        self.layer1 = base.layer1
        # Stage 2: layer2 (stride 8, 512 channels)
        self.layer2 = base.layer2
        # Stage 3: layer3 (stride 16, 1024 channels)
        self.layer3 = base.layer3
        # Stage 4: layer4 (stride 32, 2048 channels)
        self.layer4 = base.layer4

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Extracts deep features and skip connection maps.

        Args:
            x: RGB tensor (B, 3, 512, 512)

        Returns:
            bottleneck: (B, 2048, 16, 16)
            skips: dict with 'c0' (64, 256, 256), 'c1' (256, 128, 128),
                             'c2' (512, 64, 64), 'c3' (1024, 32, 32)
        """
        # Stem
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        p0 = self.maxpool(x0)

        # Layers
        c1 = self.layer1(p0)  # (B, 256, 128, 128)
        c2 = self.layer2(c1)  # (B, 512, 64, 64)
        c3 = self.layer3(c2)  # (B, 1024, 32, 32)
        c4 = self.layer4(c3)  # (B, 2048, 16, 16)

        skips = {
            "c0": x0,
            "c1": c1,
            "c2": c2,
            "c3": c3,
        }
        return c4, skips
