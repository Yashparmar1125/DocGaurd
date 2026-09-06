"""Confidence Calibration Module (Temperature Scaling)

Calibrates raw deep neural network logits post-hoc using Temperature Scaling
to ensure reported confidence percentages correspond to empirical accuracy probabilities.
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize


class TemperatureScalingCalibrator:
    """Post-hoc temperature scaling for binary classification logits."""

    def __init__(self, initial_temperature: float = 1.2):
        self.temperature = initial_temperature
        self.is_fitted = False

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        """Finds optimal scalar temperature T > 0 by minimizing Negative Log-Likelihood (NLL).

        Args:
            logits: Array of 1D uncalibrated logits.
            labels: Array of binary ground truth labels in {0, 1}.
        """
        def nll_obj(t):
            temp = max(t[0], 0.05)
            scaled = logits / temp
            # Stable BCE with logits
            loss = np.mean(np.maximum(scaled, 0) - scaled * labels + np.log(1.0 + np.exp(-np.abs(scaled))))
            return loss

        res = minimize(nll_obj, [self.temperature], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        if res.success:
            self.temperature = float(res.x[0])
            self.is_fitted = True

    def calibrate(self, logit: float) -> Tuple[float, float]:
        """Applies temperature scaling to a logit.

        Args:
            logit: Raw network output logit.

        Returns:
            tuple: (calibrated_probability in [0, 1], confidence_score in [0.5, 1.0])
        """
        scaled_logit = logit / max(self.temperature, 0.05)
        prob = 1.0 / (1.0 + np.exp(-scaled_logit))
        conf = max(prob, 1.0 - prob)
        return float(prob), float(conf)
