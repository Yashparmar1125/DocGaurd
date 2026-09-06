"""Baseline Forgery Detection Models

Implements:
1. Baseline 1: Classical Error Level Analysis (ELA) + Noise Variance + Random Forest Classifier
2. Baseline 2: Plain Single-Stream ResNet-50 Image Classifier (no DCT, no localization decoder)
"""

from typing import Tuple, Dict, Any
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.ensemble import RandomForestClassifier


class ELARandomForestBaseline:
    """Baseline 1: Classical forensic ELA feature extractor + Random Forest."""

    def __init__(self, ela_quality: int = 90, n_estimators: int = 100):
        self.ela_quality = ela_quality
        self.clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        self.is_fitted = False

    def compute_ela_features(self, image: np.ndarray) -> np.ndarray:
        """Extracts Error Level Analysis (ELA) and noise statistics from an RGB image."""
        # Step 1: Re-encode at known JPEG quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.ela_quality]
        _, encoded = cv2.imencode(".jpg", image, encode_param)
        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        # Absolute difference (scale by 10 for standard ELA visualization)
        diff = np.abs(image.astype(np.float32) - recompressed.astype(np.float32))
        gray_diff = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Step 2: Extract statistical summary features
        mean_diff = float(np.mean(gray_diff))
        std_diff = float(np.std(gray_diff))
        max_diff = float(np.max(gray_diff))
        p95_diff = float(np.percentile(gray_diff, 95))
        p99_diff = float(np.percentile(gray_diff, 99))

        # Local block variance of difference
        h, w = gray_diff.shape
        b_size = 16
        h_b, w_b = h // b_size, w // b_size
        blocks = gray_diff[: h_b * b_size, : w_b * b_size].reshape(h_b, b_size, w_b, b_size)
        block_vars = np.var(blocks, axis=(1, 3))
        max_block_var = float(np.max(block_vars)) if block_vars.size > 0 else 0.0
        var_of_vars = float(np.var(block_vars)) if block_vars.size > 0 else 0.0

        # Noise estimation via Laplacian variance
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        return np.array([
            mean_diff, std_diff, max_diff, p95_diff, p99_diff,
            max_block_var, var_of_vars, laplacian_var
        ], dtype=np.float32)

    def fit(self, images: list, labels: np.ndarray):
        X = np.array([self.compute_ela_features(img) for img in images])
        self.clf.fit(X, labels)
        self.is_fitted = True

    def predict_proba(self, image: np.ndarray) -> float:
        if not self.is_fitted:
            return 0.5
        feats = self.compute_ela_features(image).reshape(1, -1)
        return float(self.clf.predict_proba(feats)[0, 1])


class PlainResNetBaseline(nn.Module):
    """Baseline 2: Plain Single-Stream ResNet-50 Classifier (No DCT, No U-Net Decoder)."""

    def __init__(self, pretrained: bool = True, num_classes: int = 1):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        try:
            self.backbone = resnet50(weights=weights)
        except Exception:
            self.backbone = resnet50(weights=None)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting classification logits (B, 1)."""
        return self.backbone(x)
