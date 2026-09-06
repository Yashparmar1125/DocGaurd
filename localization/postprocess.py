"""Mask Post-Processing and Region Extraction

Converts raw continuous sigmoid probability maps into cleaned binary masks,
extracts connected component bounding boxes, and labels spatial zones (e.g., top-right).
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import cv2


@dataclass
class TamperedRegion:
    box: Tuple[int, int, int, int]  # (x, y, w, h)
    area_px: int
    mean_confidence: float
    location_label: str  # e.g., "top-right", "center", "bottom-left"


class PostProcessor:
    """Post-processes predicted forgery masks for reporting and visual overlay."""

    def __init__(self, threshold: float = 0.5, min_region_area: int = 40):
        self.threshold = threshold
        self.min_region_area = min_region_area

    def process(self, prob_map: np.ndarray) -> Tuple[np.ndarray, List[TamperedRegion]]:
        """Cleans probability map and extracts discrete tampered regions.

        Args:
            prob_map: Float array (H, W) in range [0, 1].

        Returns:
            tuple: (cleaned_binary_mask uint8 {0, 255}, list of TamperedRegion)
        """
        # Threshold to binary
        binary = (prob_map >= self.threshold).astype(np.uint8) * 255

        # Morphological opening to remove tiny isolated specks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        # Morphological closing to bridge small intra-text fissures
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        # Find connected components / contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = prob_map.shape[:2]
        regions: List[TamperedRegion] = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_region_area:
                continue

            x, y, rw, rh = cv2.boundingRect(c)
            region_prob = float(np.mean(prob_map[y : y + rh, x : x + rw]))
            loc_desc = self._describe_location(x, y, rw, rh, w, h)

            regions.append(TamperedRegion(
                box=(x, y, rw, rh),
                area_px=int(area),
                mean_confidence=region_prob,
                location_label=loc_desc,
            ))

        # Sort largest region first
        regions.sort(key=lambda r: r.area_px, reverse=True)
        return closed, regions

    def _describe_location(self, x: int, y: int, rw: int, rh: int, w: int, h: int) -> str:
        """Determines intuitive location phrase like 'top-right' or 'center'."""
        cx = x + rw / 2.0
        cy = y + rh / 2.0

        v_pos = "top" if cy < h * 0.35 else ("bottom" if cy > h * 0.65 else "middle")
        h_pos = "left" if cx < w * 0.35 else ("right" if cx > w * 0.65 else "center")

        if v_pos == "middle" and h_pos == "center":
            return "center"
        return f"{v_pos}-{h_pos}"
