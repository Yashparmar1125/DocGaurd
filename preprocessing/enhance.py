"""Forensic-Preserving Contrast Enhancement Module

Applies conservative CLAHE to the Luminance (Y) channel in YCrCb color space,
normalizing document readability while strictly preserving micro-noise and compression
artifacts critical for deep forensic and DCT analysis.
"""

from typing import Tuple
import cv2
import numpy as np


def apply_forensic_clahe(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Enhance document contrast mildly on the Luminance channel.

    Args:
        image: RGB numpy array (H, W, 3).
        clip_limit: Conservative clip threshold (default 2.0 to avoid noise clipping).
        tile_grid_size: Grid size for local histogram equalization (default 8x8).

    Returns:
        np.ndarray: Contrast-normalized RGB image (H, W, 3).
    """
    # Convert RGB to YCrCb
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_y = clahe.apply(y_channel)

    merged_ycrcb = cv2.merge([enhanced_y, cr_channel, cb_channel])
    enhanced_rgb = cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2RGB)

    return enhanced_rgb
