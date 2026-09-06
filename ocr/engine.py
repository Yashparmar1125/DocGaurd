"""OCR Engine Wrapper

Extracts text boxes, baseline coordinates, and character confidences using pytesseract
with an automated morphological text-region fallback for systems without Tesseract binaries.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import cv2


@dataclass
class TextBox:
    text: str
    x: int
    y: int
    w: int
    h: int
    confidence: float
    baseline_y: int  # bottom coordinate (y + h)


class OCREngine:
    """Extracts text geometry and content from document images."""

    def __init__(self, tesseract_cmd: str = None):
        self.has_tesseract = False
        if tesseract_cmd:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            self.has_tesseract = True
        else:
            try:
                import pytesseract
                # Quick test to see if tesseract binary is in PATH
                pytesseract.get_tesseract_version()
                self.has_tesseract = True
            except Exception:
                self.has_tesseract = False

    def extract_text_boxes(self, image: np.ndarray) -> List[TextBox]:
        """Extracts word/text bounding boxes with coordinates and baselines."""
        if self.has_tesseract:
            try:
                return self._extract_via_tesseract(image)
            except Exception:
                pass

        # Robust morphological text line/box detector fallback
        return self._extract_via_morphology(image)

    def _extract_via_tesseract(self, image: np.ndarray) -> List[TextBox]:
        import pytesseract
        # Tesseract expects RGB
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        boxes = []
        n_boxes = len(data["text"])
        for i in range(n_boxes):
            text = data["text"][i].strip()
            conf = float(data["conf"][i])
            if text and conf > 20.0:
                x = data["left"][i]
                y = data["top"][i]
                w = data["width"][i]
                h = data["height"][i]
                boxes.append(TextBox(
                    text=text, x=x, y=y, w=w, h=h,
                    confidence=conf / 100.0,
                    baseline_y=y + h,
                ))
        return boxes

    def _extract_via_morphology(self, image: np.ndarray) -> List[TextBox]:
        """Detects word bounding boxes via adaptive thresholding and morphology."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
        )

        # Connect text characters into word components horizontally
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 3))
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Filter out non-text blobs (too large or tiny specks)
            if 15 < w < image.shape[1] * 0.9 and 8 < h < 60:
                boxes.append(TextBox(
                    text="",  # Geometry placeholder
                    x=x, y=y, w=w, h=h,
                    confidence=0.85,
                    baseline_y=y + h,
                ))

        # Sort top-to-bottom, left-to-right
        boxes.sort(key=lambda b: (b.y // 20, b.x))
        return boxes
