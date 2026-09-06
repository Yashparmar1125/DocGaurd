"""Unit Tests for DocGuard Architecture and Loss Functions"""

import torch
import pytest

from models.docguard import DocGuardModel
from models.baselines import PlainResNetBaseline
from training.losses import MultiTaskDocGuardLoss


def test_docguard_forward_shapes():
    model = DocGuardModel(pretrained_backbone=False, embed_dim=256)
    model.eval()

    B = 2
    rgb = torch.randn(B, 3, 512, 512)
    dct = torch.randn(B, 21, 512, 512)

    with torch.no_grad():
        out = model(rgb, dct)

    assert "mask_logits" in out
    assert "mask_prob" in out
    assert "binary_logits" in out
    assert "forgery_prob" in out
    assert "type_logits" in out

    # Check shapes
    assert out["mask_logits"].shape == (B, 1, 512, 512)
    assert out["mask_prob"].shape == (B, 1, 512, 512)
    assert out["binary_logits"].shape == (B, 1)
    assert out["type_logits"].shape == (B, 5)

    # Check ranges
    assert (out["mask_prob"] >= 0.0).all() and (out["mask_prob"] <= 1.0).all()
    assert (out["forgery_prob"] >= 0.0).all() and (out["forgery_prob"] <= 1.0).all()


def test_multitask_loss_computation():
    loss_fn = MultiTaskDocGuardLoss()

    B = 2
    outputs = {
        "mask_logits": torch.randn(B, 1, 512, 512, requires_grad=True),
        "mask_prob": torch.sigmoid(torch.randn(B, 1, 512, 512)),
        "binary_logits": torch.randn(B, 1, requires_grad=True),
        "type_logits": torch.randn(B, 5, requires_grad=True),
    }

    targets = {
        "mask": torch.zeros(B, 1, 512, 512),
        "is_forged": torch.tensor([1.0, 0.0]),
        "forgery_type": torch.tensor([1, 0]),
    }

    loss_dict = loss_fn(outputs, targets)
    assert "loss_total" in loss_dict
    assert "loss_seg" in loss_dict
    assert "loss_cls" in loss_dict
    assert "loss_type" in loss_dict

    total = loss_dict["loss_total"]
    assert not torch.isnan(total)
    total.backward()
    assert outputs["mask_logits"].grad is not None
