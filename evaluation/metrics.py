"""Forensic Evaluation Metrics

Implements standard benchmark metrics for classification and pixel-level localization:
- Classification: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Localization: Pixel-level IoU, Dice coefficient
"""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc
)


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Computes all standard classification performance metrics."""
    y_pred = (y_prob >= 0.5).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    try:
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(r_curve, p_curve))
    except Exception:
        pr_auc = 0.5

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def compute_localization_metrics(
    pred_masks: np.ndarray, gt_masks: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Computes pixel-level IoU and Dice across evaluation samples.

    Args:
        pred_masks: (N, H, W) float in [0, 1]
        gt_masks: (N, H, W) float/binary in {0, 1}
    """
    binary_preds = (pred_masks >= threshold).astype(np.float32)
    binary_gts = (gt_masks >= 0.5).astype(np.float32)

    ious = []
    dices = []

    for i in range(len(binary_preds)):
        p = binary_preds[i]
        g = binary_gts[i]

        inter = np.sum(p * g)
        union = np.sum((p + g) > 0)
        card = np.sum(p) + np.sum(g)

        if union == 0:
            # Both empty: perfect match on authentic document
            iou = 1.0
            dice = 1.0
        else:
            iou = float(inter / (union + 1e-6))
            dice = float((2.0 * inter) / (card + 1e-6))

        ious.append(iou)
        dices.append(dice)

    return {
        "mean_iou": float(np.mean(ious)),
        "mean_dice": float(np.mean(dices)),
    }
