"""Cross-Attention Fusion Module

Fuses spatial RGB features with frequency-domain DCT features using multi-head
cross-attention, allowing spatial regions to query frequency compression traces.
"""

from typing import Tuple
import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Multi-Head Cross-Attention fusing RGB visual representations with DCT frequency traces."""

    def __init__(
        self,
        rgb_channels: int = 2048,
        dct_channels: int = 512,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Projections to common embedding space
        self.proj_rgb = nn.Conv2d(rgb_channels, embed_dim, kernel_size=1)
        self.proj_dct = nn.Conv2d(dct_channels, embed_dim, kernel_size=1)

        # Multi-Head Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Normalization and Feed-Forward Network
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

        # Output projection back to 1024 / bottleneck channels
        self.out_conv = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(
        self, feat_rgb: torch.Tensor, feat_dct: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Executes cross-attention between RGB queries and DCT keys/values.

        Args:
            feat_rgb: (B, 2048, H, W)
            feat_dct: (B, 512, H, W)

        Returns:
            fused_feature_map: (B, 512, H, W)
            attn_weights: (B, num_heads, H*W, H*W)
        """
        B, _, H, W = feat_rgb.shape

        # Project and reshape: (B, embed_dim, H, W) -> (B, H*W, embed_dim)
        q = self.proj_rgb(feat_rgb).flatten(2).permute(0, 2, 1)
        kv = self.proj_dct(feat_dct).flatten(2).permute(0, 2, 1)

        # Cross attention: Queries = RGB, Keys/Values = DCT
        attn_out, attn_weights = self.cross_attn(query=q, key=kv, value=kv)
        x = self.norm1(q + attn_out)
        x = self.norm2(x + self.mlp(x))

        # Reshape back to spatial grid: (B, embed_dim, H, W)
        spatial_fused = x.permute(0, 2, 1).view(B, self.embed_dim, H, W)

        # Concatenate projected DCT for dual identity and project
        dct_proj = self.proj_dct(feat_dct)
        concat = torch.cat([spatial_fused, dct_proj], dim=1)
        output = self.out_conv(concat)

        return output, attn_weights
