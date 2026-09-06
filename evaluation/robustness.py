"""Robustness and Distortion Stress Suite

Evaluates model resilience against real-world degradation:
1. JPEG Re-compression across quality factors Q in [40, 55, 70, 85, 95]
2. Additive Gaussian noise (sigma in [2, 5, 10, 15])
3. Spatial downscaling and interpolation
"""

from typing import Dict, List, Any
import numpy as np
import cv2
import torch

from .metrics import compute_classification_metrics
from models.docguard import DocGuardModel
from preprocessing.pipeline import PreprocessingPipeline
from data.generator import DocumentForgeryGenerator


class RobustnessSuite:
    """Evaluates DocGuard performance under systematic image distortions."""

    def __init__(self, model: DocGuardModel, device: str = "cpu"):
        self.model = model.eval().to(torch.device(device))
        self.device = torch.device(device)
        self.preprocessor = PreprocessingPipeline()
        self.generator = DocumentForgeryGenerator()

    def evaluate_jpeg_compression(
        self, quality_factors: List[int] = [40, 55, 70, 85, 95], samples_per_q: int = 20
    ) -> Dict[int, Dict[str, float]]:
        """Measures F1 and accuracy across decreasing JPEG quality factors."""
        results = {}
        for q in quality_factors:
            y_true = []
            y_probs = []

            for _ in range(samples_per_q):
                s = self.generator.generate_sample(apply_double_jpeg=False)
                # Re-compress
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
                _, enc = cv2.imencode(".jpg", s["image"], encode_param)
                distorted = cv2.imdecode(enc, cv2.IMREAD_COLOR)

                p = self.preprocessor.process_image(distorted)
                with torch.no_grad():
                    out = self.model(
                        p["rgb_tensor"].to(self.device), p["dct_tensor"].to(self.device)
                    )
                prob = float(out["forgery_prob"].item())

                y_true.append(int(s["is_forged"]))
                y_probs.append(prob)

            metrics = compute_classification_metrics(np.array(y_true), np.array(y_probs))
            results[q] = metrics

        return results

    def evaluate_gaussian_noise(
        self, sigmas: List[float] = [2.0, 5.0, 10.0, 15.0], samples_per_sigma: int = 20
    ) -> Dict[float, Dict[str, float]]:
        """Measures degradation under additive Gaussian sensor noise."""
        results = {}
        for sig in sigmas:
            y_true = []
            y_probs = []

            for _ in range(samples_per_sigma):
                s = self.generator.generate_sample()
                noise = np.random.normal(0, sig, s["image"].shape)
                noisy = np.clip(s["image"].astype(np.float32) + noise, 0, 255).astype(np.uint8)

                p = self.preprocessor.process_image(noisy)
                with torch.no_grad():
                    out = self.model(
                        p["rgb_tensor"].to(self.device), p["dct_tensor"].to(self.device)
                    )
                prob = float(out["forgery_prob"].item())

                y_true.append(int(s["is_forged"]))
                y_probs.append(prob)

            metrics = compute_classification_metrics(np.array(y_true), np.array(y_probs))
            results[sig] = metrics

        return results
