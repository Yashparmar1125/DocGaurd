"""Frequency Patch Tokenizer & Transformer Stream

Converts block-wise 8x8 DCT coefficients into frequency tokens and processes them
via Multi-Head Self-Attention layers to capture double-JPEG grid periodicity.
"""

from typing import Tuple
import torch
import torch.nn as nn


class FrequencyTransformerStream(nn.Module):
    """Tokenizes DCT frequency maps and models block periodicity via self-attention."""

    def __init__(
        self,
        in_channels: int = 21,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        target_grid: Tuple[int, int] = (16, 16),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.target_grid = target_grid

        # Spatial patch projection: downsamples 512x512 DCT map to 16x16 grid
        # stride 32 with 32x32 patch
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=4, stride=4, padding=0),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=8, stride=8, padding=0),
        )

        num_patches = target_grid[0] * target_grid[1]  # 16 * 16 = 256
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, dct_map: torch.Tensor) -> torch.Tensor:
        """Forward pass converting DCT map to bottleneck frequency tokens.

        Args:
            dct_map: (B, 21, 512, 512)

        Returns:
            torch.Tensor: (B, embed_dim, 16, 16)
        """
        B = dct_map.size(0)
        # Patch embed: (B, 21, 512, 512) -> (B, embed_dim, 16, 16)
        x = self.patch_embed(dct_map)
        H, W = x.shape[2], x.shape[3]

        # Reshape to token sequence: (B, H*W, embed_dim)
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]

        # Self-attention over frequency tokens
        tokens = self.transformer_encoder(tokens)
        tokens = self.norm(tokens)

        # Reshape back to spatial grid: (B, embed_dim, H, W)
        out = tokens.transpose(1, 2).view(B, self.embed_dim, H, W)
        return out
