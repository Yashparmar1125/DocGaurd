"""DocGuard Evaluation and Benchmark Package

Contains:
- Metrics: Pixel IoU, Dice, Precision, Recall, F1, PR-AUC, ROC-AUC
- BenchmarkRunner: Compares Baseline 1, Baseline 2, and DocGuard
- RobustnessSuite: Stress-tests against JPEG compression, noise, and resizing
"""

from .metrics import compute_classification_metrics, compute_localization_metrics
from .benchmark import BenchmarkRunner
from .robustness import RobustnessSuite

__all__ = [
    "compute_classification_metrics",
    "compute_localization_metrics",
    "BenchmarkRunner",
    "RobustnessSuite",
]
