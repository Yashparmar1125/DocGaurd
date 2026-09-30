"""Unit Tests for DocGuard-ViT (Vision Transformer Architecture)"""

import torch
import pytest

from models.docguard_transformer import DocGuardTransformerModel
from models.transformer_backbone import SwinRGBBackbone
from models.dct_tokenizer import FrequencyTransformerStream
from models.mlp_decoder import AllMLPLocalizationDecoder
from training.losses import MultiTaskDocGuardLoss


def test_swin_backbone_shapes():
    backbone = SwinRGBBackbone(pretrained=False, img_size=512)
    backbone.eval()

    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        bottleneck, skips = backbone(x)

    assert bottleneck.shape == (1, 768, 16, 16)
    assert skips["c0"].shape == (1, 96, 128, 128)
    assert skips["c1"].shape == (1, 192, 64, 64)
    assert skips["c2"].shape == (1, 384, 32, 32)


def test_frequency_transformer_stream():
    stream = FrequencyTransformerStream(in_channels=21, embed_dim=256)
    stream.eval()

    dct = torch.randn(1, 21, 512, 512)
    with torch.no_grad():
        out = stream(dct)

    assert out.shape == (1, 256, 16, 16)


def test_all_mlp_decoder():
    decoder = AllMLPLocalizationDecoder(in_channels=[96, 192, 384, 512], embed_dim=256)
    decoder.eval()

    bottleneck = torch.randn(1, 512, 16, 16)
    skips = {
        "c0": torch.randn(1, 96, 128, 128),
        "c1": torch.randn(1, 192, 64, 64),
        "c2": torch.randn(1, 384, 32, 32),
    }

    with torch.no_grad():
        logits = decoder(bottleneck, skips)

    assert logits.shape == (1, 1, 512, 512)


def test_docguard_transformer_end_to_end():
    model = DocGuardTransformerModel(pretrained_backbone=False)
    model.eval()

    rgb = torch.randn(1, 3, 512, 512)
    dct = torch.randn(1, 21, 512, 512)

    with torch.no_grad():
        out = model(rgb, dct)

    assert "mask_logits" in out
    assert "mask_prob" in out
    assert "binary_logits" in out
    assert "type_logits" in out

    assert out["mask_logits"].shape == (1, 1, 512, 512)
    assert out["mask_prob"].shape == (1, 1, 512, 512)
    assert out["binary_logits"].shape == (1, 1)
    assert out["type_logits"].shape == (1, 5)

    # Test loss backward pass
    loss_fn = MultiTaskDocGuardLoss()
    targets = {
        "mask": torch.zeros(1, 1, 512, 512),
        "is_forged": torch.tensor([1.0]),
        "forgery_type": torch.tensor([2]),
    }
    loss_dict = loss_fn(out, targets)
    assert not torch.isnan(loss_dict["loss_total"])
