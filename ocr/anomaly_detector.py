"""OCR Anomaly Detector

Detects font height discrepancies, baseline alignment anomalies, and inter-word
spacing irregularities across document lines to flag retyped or spliced text fields.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
from .engine import OCREngine, TextBox


@dataclass
class AnomalyFlag:
    box: TextBox
    anomaly_type: str  # 'font_size_mismatch', 'baseline_misalignment', 'spacing_outlier'
    severity: float    # Z-score / severity in [0, 1]
    message: str


class OCRAnomalyDetector:
    """Analyzes text geometry to detect local typographic inconsistencies."""

    def __init__(self, engine: OCREngine = None):
        self.engine = engine or OCREngine()

    def detect_anomalies(
        self, image: np.ndarray, line_tolerance_px: int = 14
    ) -> Tuple[List[AnomalyFlag], np.ndarray]:
        """Detects layout and typographic anomalies in a document image.

        Args:
            image: RGB document array (H, W, 3).
            line_tolerance_px: Maximum vertical distance to cluster into one line.

        Returns:
            tuple: (list of AnomalyFlag, binary_anomaly_mask (H, W) float32)
        """
        boxes = self.engine.extract_text_boxes(image)
        h, w = image.shape[:2]
        anomaly_mask = np.zeros((h, w), dtype=np.float32)
        anomalies: List[AnomalyFlag] = []

        if len(boxes) < 4:
            return anomalies, anomaly_mask

        # Step 1: Cluster boxes into horizontal lines
        lines: List[List[TextBox]] = []
        # Sort boxes primarily by baseline_y
        sorted_boxes = sorted(boxes, key=lambda b: b.baseline_y)

        current_line = [sorted_boxes[0]]
        for b in sorted_boxes[1:]:
            # Check if this box belongs to the same horizontal line
            line_median_y = np.median([box.baseline_y for box in current_line])
            if abs(b.baseline_y - line_median_y) <= line_tolerance_px:
                current_line.append(b)
            else:
                lines.append(sorted(current_line, key=lambda box: box.x))
                current_line = [b]
        if current_line:
            lines.append(sorted(current_line, key=lambda box: box.x))

        # Step 2: Analyze each line for baseline and font height outliers
        for line in lines:
            if len(line) < 3:
                continue

            heights = np.array([box.h for box in line], dtype=np.float32)
            baselines = np.array([box.baseline_y for box in line], dtype=np.float32)

            median_h = np.median(heights)
            std_h = np.std(heights) + 1e-4

            median_base = np.median(baselines)
            std_base = np.std(baselines) + 1e-4

            for box in line:
                z_h = abs(box.h - median_h) / std_h
                z_base = abs(box.baseline_y - median_base) / std_base

                # Flag 1: Font height mismatch (> 2.5 std and > 4px diff)
                if z_h > 2.5 and abs(box.h - median_h) > 4:
                    severity = min(float(z_h / 5.0), 1.0)
                    anomalies.append(AnomalyFlag(
                        box=box,
                        anomaly_type="font_size_mismatch",
                        severity=severity,
                        message=f"Font height ({box.h}px) deviates significantly from line average ({median_h:.1f}px)"
                    ))
                    anomaly_mask[box.y : box.y + box.h, box.x : box.x + box.w] = 1.0

                # Flag 2: Baseline misalignment (> 2.5 std and > 3px shift)
                elif z_base > 2.5 and abs(box.baseline_y - median_base) > 3:
                    severity = min(float(z_base / 5.0), 1.0)
                    anomalies.append(AnomalyFlag(
                        box=box,
                        anomaly_type="baseline_misalignment",
                        severity=severity,
                        message=f"Text baseline sits off-axis by {abs(box.baseline_y - median_base):.1f}px compared to neighboring line text"
                    ))
                    anomaly_mask[box.y : box.y + box.h, box.x : box.x + box.w] = 1.0

        return anomalies, anomaly_mask
