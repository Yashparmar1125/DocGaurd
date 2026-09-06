"""DocGuard OCR and Layout Anomaly Package

Provides OCR text box extraction, text-baseline alignment checking,
font height anomaly detection, and late fusion corroboration.
"""

from .engine import OCREngine, TextBox
from .anomaly_detector import OCRAnomalyDetector

__all__ = [
    "OCREngine",
    "TextBox",
    "OCRAnomalyDetector",
]
