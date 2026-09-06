"""DocGuard Explainability Package

Contains:
- GradCAM: Visual attribution heatmaps on deep convolutional features
- TemperatureScalingCalibrator: Post-hoc confidence calibration
- ForensicReasoningEngine: Natural language rationale generation
"""

from .gradcam import GradCAM
from .calibration import TemperatureScalingCalibrator
from .reasoning import ForensicReasoningEngine

__all__ = [
    "GradCAM",
    "TemperatureScalingCalibrator",
    "ForensicReasoningEngine",
]
