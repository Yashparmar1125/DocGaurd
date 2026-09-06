"""DocGuard: Hybrid CNN-Transformer Multi-Task Forgery Detection Architecture

Unifies RGB spatial stream (ResNet-50), DCT frequency stream (CNN),
Cross-Attention Fusion, U-Net localization decoder, and dual classification heads.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from .rgb_backbone import RGBResNetBackbone
from .dct_backbone import DCTForensicBackbone
from .cross_attention import CrossAttentionFusion
from .unet_decoder import UNetLocalizationDecoder
from .heads import ClassificationHeads


class DocGuardModel(nn.Module):
    """Complete DocGuard Multi-Task Architecture."""

    def __init__(
        self,
        pretrained_backbone: bool = True,
        num_forgery_types: int = 5,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.rgb_stream = RGBResNetBackbone(pretrained=pretrained_backbone)
        self.dct_stream = DCTForensicBackbone(in_channels=21, out_channels=embed_dim)
        self.fusion = CrossAttentionFusion(
            rgb_channels=2048, dct_channels=embed_dim, embed_dim=embed_dim
        )
        self.decoder = UNetLocalizationDecoder(bottleneck_ch=embed_dim)
        self.heads = ClassificationHeads(
            in_channels=embed_dim, num_forgery_types=num_forgery_types
        )

    def forward(
        self,
        rgb: torch.Tensor,
        dct: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Dict[str, Any]:
        """Multi-task forward pass.

        Args:
            rgb: RGB tensor (B, 3, 512, 512)
            dct: 8x8 block DCT coefficient tensor (B, 21, 512, 512)
            return_intermediates: Whether to include raw features for XAI/Grad-CAM

        Returns:
            dict containing:
                - 'mask_logits': (B, 1, 512, 512)
                - 'mask_prob': (B, 1, 512, 512) in [0, 1]
                - 'binary_logits': (B, 1)
                - 'forgery_prob': (B, 1) in [0, 1]
                - 'type_logits': (B, 5)
                - 'type_prob': (B, 5)
                - 'attn_weights': cross-attention weights
        """
        # Step 1: Feature Extraction
        rgb_bottleneck, skips = self.rgb_stream(rgb)  # (B, 2048, 16, 16)
        dct_bottleneck = self.dct_stream(dct)        # (B, 512, 16, 16)

        # Step 2: Cross-Attention Fusion
        fused, attn_weights = self.fusion(rgb_bottleneck, dct_bottleneck)  # (B, 512, 16, 16)

        # Step 3: Localization Decoder Head
        mask_logits = self.decoder(fused, skips)  # (B, 1, 512, 512)
        mask_prob = torch.sigmoid(mask_logits)

        # Step 4: Classification Heads
        binary_logits, type_logits = self.heads(fused)
        forgery_prob = torch.sigmoid(binary_logits)
        type_prob = torch.softmax(type_logits, dim=-1)

        results = {
            "mask_logits": mask_logits,
            "mask_prob": mask_prob,
            "binary_logits": binary_logits,
            "forgery_prob": forgery_prob,
            "type_logits": type_logits,
            "type_prob": type_prob,
            "attn_weights": attn_weights,
        }

        if return_intermediates:
            results["fused_feature"] = fused
            results["rgb_bottleneck"] = rgb_bottleneck
            results["skips"] = skips

        return results
