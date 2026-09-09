"""OCR Anomaly Detector

Detects font height discrepancies, baseline alignment anomalies, and ink color
inconsistencies across document text lines to flag retyped or spliced fields.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from .engine import OCREngine, TextBox


@dataclass
class AnomalyFlag:
    box: TextBox
    anomaly_type: str  # 'font_size_mismatch', 'baseline_misalignment', 'ink_color_inconsistency'
    severity: float    # Severity in [0, 1]
    message: str


class OCRAnomalyDetector:
    """Analyzes text geometry and ink characteristics to detect local typographic tampering."""

    def __init__(self, engine: OCREngine = None):
        self.engine = engine or OCREngine()

    def detect_anomalies(
        self, image: np.ndarray, line_tolerance_px: int = 14
    ) -> Tuple[List[AnomalyFlag], np.ndarray]:
        """Detects layout, baseline, and typographic anomalies in a document image.

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

        if len(boxes) < 2:
            return anomalies, anomaly_mask

        # Step 1: Cluster boxes into horizontal lines
        lines: List[List[TextBox]] = []
        sorted_boxes = sorted(boxes, key=lambda b: b.baseline_y)

        current_line = [sorted_boxes[0]]
        for b in sorted_boxes[1:]:
            line_median_y = np.median([box.baseline_y for box in current_line])
            if abs(b.baseline_y - line_median_y) <= line_tolerance_px:
                current_line.append(b)
            else:
                lines.append(sorted(current_line, key=lambda box: box.x))
                current_line = [b]
        if current_line:
            lines.append(sorted(current_line, key=lambda box: box.x))

        # Document-wide text height statistics
        doc_heights = np.array([b.h for b in boxes], dtype=np.float32)
        doc_median_h = float(np.median(doc_heights))

        # Step 2: Line-level analysis (baseline & font size)
        for line in lines:
            if len(line) >= 2:
                heights = np.array([box.h for box in line], dtype=np.float32)
                baselines = np.array([box.baseline_y for box in line], dtype=np.float32)

                line_med_h = np.median(heights)
                line_med_base = np.median(baselines)

                for box in line:
                    h_diff = abs(box.h - line_med_h)
                    base_diff = abs(box.baseline_y - line_med_base)

                    # Baseline misalignment
                    if base_diff >= 4.0:
                        severity = min(float(base_diff / 10.0), 1.0)
                        anomalies.append(AnomalyFlag(
                            box=box,
                            anomaly_type="baseline_misalignment",
                            severity=severity,
                            message=f"Text baseline sits {base_diff:.1f}px off-axis from line neighbor text",
                        ))
                        anomaly_mask[box.y : box.y + box.h, box.x : box.x + box.w] = 1.0

                    # Font height outlier within line
                    elif h_diff >= 4.0:
                        severity = min(float(h_diff / 8.0), 1.0)
                        anomalies.append(AnomalyFlag(
                            box=box,
                            anomaly_type="font_size_mismatch",
                            severity=severity,
                            message=f"Font height ({box.h}px) deviates from line average ({line_med_h:.1f}px)",
                        ))
                        anomaly_mask[box.y : box.y + box.h, box.x : box.x + box.w] = 1.0

        # Step 3: Ink Chromatic / Color Inconsistency Check
        # Sample text pixels across boxes to detect re-rendered text in mismatched ink (e.g. red text in black document)
        for box in boxes:
            patch = image[box.y : box.y + box.h, box.x : box.x + box.w]
            if patch.size == 0 or patch.shape[0] < 3 or patch.shape[1] < 3:
                continue

            gray_p = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            p_thresh = np.percentile(gray_p, 35)
            text_pixels = patch[gray_p <= p_thresh]

            if len(text_pixels) > 10:
                mean_rgb = np.mean(text_pixels, axis=0)
                # Chromatic divergence: max channel minus min channel
                chroma = np.max(mean_rgb) - np.min(mean_rgb)
                # Red or blue saturation
                red_excess = mean_rgb[0] - (mean_rgb[1] + mean_rgb[2]) / 2.0

                if chroma > 28.0 and red_excess > 20.0:
                    severity = min(float(chroma / 60.0), 1.0)
                    anomalies.append(AnomalyFlag(
                        box=box,
                        anomaly_type="ink_color_inconsistency",
                        severity=severity,
                        message=f"Text ink chromatic profile (RGB {int(mean_rgb[0])},{int(mean_rgb[1])},{int(mean_rgb[2])}) deviates from document ink",
                    ))
                    anomaly_mask[box.y : box.y + box.h, box.x : box.x + box.w] = 1.0

        return anomalies, anomaly_mask
