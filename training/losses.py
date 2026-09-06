"""Multi-Task Loss Functions for DocGuard

Implements:
1. Dice Loss for handling extreme class imbalance in document forgery masks
2. Balanced Binary Cross-Entropy
3. Categorical Cross-Entropy with label smoothing (0.1) for 5-way forgery type
4. Weighted Multi-Task Combined Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss with Laplace smoothing."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_probs: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """Computes Dice loss between predicted probabilities and ground truth masks.

        Args:
            pred_probs: (B, 1, H, W) in [0, 1]
            target_masks: (B, 1, H, W) in {0, 1}
        """
        p_flat = pred_probs.view(pred_probs.size(0), -1)
        t_flat = target_masks.view(target_masks.size(0), -1)

        intersection = (p_flat * t_flat).sum(dim=1)
        cardinality = p_flat.sum(dim=1) + t_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class MultiTaskDocGuardLoss(nn.Module):
    """Weighted Multi-Task Loss as defined in Project Specification Section 20.

    L_total = lambda_cls * L_cls + lambda_type * L_type + lambda_seg * (alpha * L_Dice + (1 - alpha) * L_BCE)
    """

    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_type: float = 0.5,
        lambda_seg: float = 1.0,
        alpha_dice: float = 0.5,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_type = lambda_type
        self.lambda_seg = lambda_seg
        self.alpha_dice = alpha_dice

        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.type_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        outputs: dict,
        targets: dict,
    ) -> dict:
        """Calculates all component losses and weighted sum.

        Args:
            outputs: Dictionary from DocGuardModel.forward
            targets: Dictionary from DocumentForgeryDataset.__getitem__
        """
        mask_logits = outputs["mask_logits"]
        mask_prob = outputs["mask_prob"]
        binary_logits = outputs["binary_logits"].squeeze(-1)
        type_logits = outputs["type_logits"]

        target_mask = targets["mask"]
        target_is_forged = targets["is_forged"]
        target_type = targets["forgery_type"]

        # 1. Segmentation loss (Dice + BCE)
        l_dice = self.dice_loss(mask_prob, target_mask)
        l_mask_bce = self.bce_loss(mask_logits, target_mask)
        l_seg = self.alpha_dice * l_dice + (1.0 - self.alpha_dice) * l_mask_bce

        # 2. Binary classification loss (Authentic vs Forged)
        l_cls = self.bce_loss(binary_logits, target_is_forged)

        # 3. Forgery-type loss (computed only on forged samples)
        forged_indices = torch.where(target_is_forged > 0.5)[0]
        if len(forged_indices) > 0:
            l_type = self.type_loss_fn(
                type_logits[forged_indices], target_type[forged_indices]
            )
        else:
            l_type = torch.tensor(0.0, device=binary_logits.device)

        # 4. Total multi-task loss
        l_total = (
            self.lambda_cls * l_cls
            + self.lambda_type * l_type
            + self.lambda_seg * l_seg
        )

        return {
            "loss_total": l_total,
            "loss_cls": l_cls,
            "loss_type": l_type,
            "loss_seg": l_seg,
            "loss_dice": l_dice,
            "loss_mask_bce": l_mask_bce,
        }
