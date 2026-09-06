"""Discrete Cosine Transform (DCT) Forensic Feature Extractor

Computes block-wise 8x8 2D-DCT coefficients on the Luminance channel to expose
double-compression periodicity and JPEG grid misalignment traces.
"""

from typing import Tuple
import cv2
import numpy as np


def extract_block_dct(
    image: np.ndarray, block_size: int = 8, num_ac_channels: int = 21
) -> np.ndarray:
    """Computes block-wise 8x8 2D-DCT coefficients on the luminance channel.

    Args:
        image: RGB numpy array (H, W, 3) or Luminance array (H, W).
        block_size: JPEG standard block size (default 8).
        num_ac_channels: Number of zigzag-ordered AC coefficient channels to extract (default 21).

    Returns:
        np.ndarray: Block DCT feature map of shape (H // block_size, W // block_size, num_ac_channels),
                    float32 normalized.
    """
    if len(image.shape) == 3:
        # Standard ITU-R BT.601 conversion to Luminance
        y_channel = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)
    else:
        y_channel = image.astype(np.float32)

    h, w = y_channel.shape
    # Pad to multiple of block_size if necessary
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size
    if pad_h > 0 or pad_w > 0:
        y_channel = np.pad(y_channel, ((0, pad_h), (0, pad_w)), mode="reflect")

    h_blocks = y_channel.shape[0] // block_size
    w_blocks = y_channel.shape[1] // block_size

    # Reshape into 8x8 blocks: (h_blocks, 8, w_blocks, 8) -> (h_blocks, w_blocks, 8, 8)
    blocks = y_channel.reshape(h_blocks, block_size, w_blocks, block_size).transpose(0, 2, 1, 3)

    # Standard 8x8 zigzag index order for frequency coefficients (first num_ac_channels)
    zigzag_indices = [
        (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1),
        (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5), (1, 4),
        (2, 3), (3, 2), (4, 1), (5, 0), (6, 0)
    ][:num_ac_channels]

    # Pre-allocate output feature map
    dct_features = np.zeros((h_blocks, w_blocks, num_ac_channels), dtype=np.float32)

    # Center luminance around 0 (JPEG standard: -128)
    blocks_centered = blocks - 128.0

    # Vectorized / looped 2D DCT over blocks
    # Using cv2.dct per block row for high performance
    for i in range(h_blocks):
        for j in range(w_blocks):
            dct_blk = cv2.dct(blocks_centered[i, j])
            for ch_idx, (r, c) in enumerate(zigzag_indices):
                dct_features[i, j, ch_idx] = dct_blk[r, c]

    # Log-scale compression artifacts to stabilize gradient flow: log(1 + |x|) * sign(x)
    dct_normalized = np.sign(dct_features) * np.log1p(np.abs(dct_features))

    return dct_normalized


def compute_dct_features(
    image: np.ndarray, target_spatial_shape: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    """Extracts DCT features and resamples to target spatial dimensions.

    Args:
        image: RGB input image (H, W, 3).
        target_spatial_shape: (H, W) for downstream tensor alignment (default (512, 512)).

    Returns:
        np.ndarray: Array of shape (Channels, target_H, target_W), float32.
    """
    # Resize image to target shape first to ensure consistent 8x8 block grid
    if image.shape[:2] != target_spatial_shape:
        resized = cv2.resize(image, (target_spatial_shape[1], target_spatial_shape[0]), interpolation=cv2.INTER_AREA)
    else:
        resized = image

    raw_dct = extract_block_dct(resized, block_size=8, num_ac_channels=21)
    # raw_dct is shape (64, 64, 21) for 512x512
    # Transpose to (21, 64, 64)
    ch_first = raw_dct.transpose(2, 0, 1)

    # Upsample DCT frequency energy back to 512x512 for pixel-aligned fusion
    upsampled = cv2.resize(
        raw_dct, (target_spatial_shape[1], target_spatial_shape[0]), interpolation=cv2.INTER_NEAREST
    )
    # Return (Channels, H, W)
    return upsampled.transpose(2, 0, 1)
