"""DocGuard-ViT: Vision Transformer Architecture for Document Forgery Forensics

Unifies:
1. SwinRGBBackbone: Hierarchical Swin Transformer RGB spatial stream
2. FrequencyTransformerStream: 8x8 block DCT patch tokenizer and self-attention
3. CrossAttentionFusion: Multi-head cross-attention between spatial and frequency tokens
4. AllMLPLocalizationDecoder: SegFormer-style All-MLP segmentation decoder
5. ClassificationHeads: Dual authentic/forged & 5-way forgery-type classifiers
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from .transformer_backbone import SwinRGBBackbone
from .dct_tokenizer import FrequencyTransformerStream
from .cross_attention import CrossAttentionFusion
from .mlp_decoder import AllMLPLocalizationDecoder
from .heads import ClassificationHeads


class DocGuardTransformerModel(nn.Module):
    """Full Vision Transformer architecture for multi-task document forensics."""

    def __init__(
        self,
        pretrained_backbone: bool = False,
        num_forgery_types: int = 5,
        embed_dim: int = 512,
    ):
        super().__init__()
        # RGB Stream: Swin-T (outputs stages with channels [96, 192, 384, 768])
        self.rgb_stream = SwinRGBBackbone(pretrained=pretrained_backbone, img_size=512)

        # Frequency Stream: DCT Tokenizer + Self-Attention (outputs 256 channels at 16x16)
        self.dct_stream = FrequencyTransformerStream(
            in_channels=21, embed_dim=256, num_heads=8, num_layers=2
        )

        # Cross-Attention Fusion: Query = RGB (768), Key/Value = DCT (256) -> Fused (512)
        self.fusion = CrossAttentionFusion(
            rgb_channels=768, dct_channels=256, embed_dim=embed_dim, num_heads=8
        )

        # All-MLP Decoder: takes c0 (96), c1 (192), c2 (384), and fused (512)
        self.decoder = AllMLPLocalizationDecoder(
            in_channels=[96, 192, 384, embed_dim], embed_dim=256
        )

        # Classification Heads: pooled from fused bottleneck (512)
        self.heads = ClassificationHeads(
            in_channels=embed_dim, num_forgery_types=num_forgery_types
        )

    def forward(
        self,
        rgb: torch.Tensor,
        dct: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Dict[str, Any]:
        """Multi-task transformer forward pass.

        Args:
            rgb: RGB tensor (B, 3, 512, 512)
            dct: DCT frequency tensor (B, 21, 512, 512)

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
        # 1. Hierarchical RGB feature extraction
        rgb_bottleneck, skips = self.rgb_stream(rgb)  # (B, 768, 16, 16)

        # 2. Tokenized frequency feature extraction
        dct_bottleneck = self.dct_stream(dct)        # (B, 256, 16, 16)

        # 3. Cross-attention fusion
        fused, attn_weights = self.fusion(rgb_bottleneck, dct_bottleneck)  # (B, 512, 16, 16)

        # 4. All-MLP localization decoding
        mask_logits = self.decoder(fused, skips)      # (B, 1, 512, 512)
        mask_prob = torch.sigmoid(mask_logits)

        # 5. Dual classification heads
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
