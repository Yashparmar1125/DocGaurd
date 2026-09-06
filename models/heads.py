"""Multi-Task Classification Heads

Computes:
1. Authentic vs. Forged binary prediction (logit)
2. 5-Way Forgery-Type classification logits
"""

from typing import Tuple
import torch
import torch.nn as nn


class ClassificationHeads(nn.Module):
    """Dual classification heads operating on pooled fused features."""

    def __init__(self, in_channels: int = 512, num_forgery_types: int = 5, dropout: float = 0.25):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Binary authentic vs forged classifier
        self.binary_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        # 5-way forgery type classifier:
        # [authentic, copy_move, splicing, text_tamper, erasure]
        self.type_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_forgery_types),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs dual classification heads.

        Args:
            x: Fused feature map (B, in_channels, H, W)

        Returns:
            binary_logits: (B, 1)
            type_logits: (B, num_forgery_types)
        """
        pooled = self.pool(x)
        binary_logits = self.binary_classifier(pooled)
        type_logits = self.type_classifier(pooled)
        return binary_logits, type_logits
