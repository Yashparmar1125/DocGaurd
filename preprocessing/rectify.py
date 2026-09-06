"""Document Rectification Module

Detects document boundaries using morphological edge contours and computes
a 4-point perspective transform to de-warp photographed documents to a flat frontal view.
"""

from typing import Tuple, Optional
import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 points as: [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right has smallest diff
    rect[3] = pts[np.argmax(diff)]  # Bottom-left has largest diff
    return rect


def find_document_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """Finds the largest 4-point polygon contour representing a document boundary."""
    h, w = image.shape[:2]
    image_area = h * w

    # Downscale for stable edge detection if very large
    scale = 1.0
    max_dim = 1200
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = image

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    # Dilate edges slightly to close small gaps in document border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    small_area = small.shape[0] * small.shape[1]
    for c in contours:
        area = cv2.contourArea(c)
        # Require document to occupy at least 25% of the frame
        if area < 0.25 * small_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            # Rescale points back to original image coordinates
            pts = approx.reshape(4, 2).astype("float32")
            if scale != 1.0:
                pts = pts / scale
            return pts

    return None


def rectify_document_boundary(image: np.ndarray) -> Tuple[np.ndarray, bool, Optional[np.ndarray]]:
    """Detects document boundary and applies perspective warp if valid.

    Args:
        image: Input RGB numpy array (H, W, 3).

    Returns:
        tuple: (rectified_image, was_rectified_bool, corner_points_or_none)
    """
    pts = find_document_contour(image)
    if pts is None:
        return image, False, None

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate width of new image
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    # Calculate height of new image
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    if max_width < 100 or max_height < 100:
        return image, False, None

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transform_matrix, (max_width, max_height), flags=cv2.INTER_LINEAR)

    return warped, True, rect
