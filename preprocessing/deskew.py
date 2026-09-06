"""Document Deskew Module

Detects document text line rotation angle via Hough Line Transform and corrects
residual skew using affine rotation with boundary padding.
"""

from typing import Tuple
import cv2
import numpy as np


def detect_skew_angle(image: np.ndarray, max_angle_deg: float = 15.0) -> float:
    """Estimates skew angle in degrees using Hough Line Transform on text edges.

    Args:
        image: RGB or grayscale image.
        max_angle_deg: Maximum credible skew angle (default 15.0 deg).

    Returns:
        float: Estimated skew angle in degrees (positive = counter-clockwise).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Invert binary threshold to get white text on black background for edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
    )

    # Use horizontal morphological kernel to connect text characters into lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    edges = cv2.Canny(dilated, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=15)

    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        coords = line.ravel()
        if len(coords) < 4:
            continue
        x1, y1, x2, y2 = coords[:4]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        if abs(angle) <= max_angle_deg:
            angles.append(angle)

    if not angles:
        return 0.0

    # Return median angle to reject outliers
    return float(np.median(angles))


def deskew_document(image: np.ndarray, max_angle_deg: float = 15.0) -> Tuple[np.ndarray, float]:
    """Detects and corrects residual document rotation.

    Args:
        image: Input RGB numpy array (H, W, 3).
        max_angle_deg: Maximum rotation correction permitted.

    Returns:
        tuple: (deskewed_image, angle_corrected_degrees)
    """
    angle = detect_skew_angle(image, max_angle_deg=max_angle_deg)
    if abs(angle) < 0.2:  # Negligible skew
        return image, 0.0

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    # Negative angle because cv2.getRotationMatrix2D rotates counter-clockwise for positive angles
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Sample background color from image corners for clean border fill
    corners = np.array([image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]])
    bg_color = tuple(int(c) for c in np.median(corners, axis=0))

    deskewed = cv2.warpAffine(
        image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color
    )

    return deskewed, angle
