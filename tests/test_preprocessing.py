"""Unit Tests for Preprocessing Pipeline"""

import numpy as np
import pytest
import cv2

from preprocessing.rectify import rectify_document_boundary
from preprocessing.deskew import deskew_document, detect_skew_angle
from preprocessing.enhance import apply_forensic_clahe
from preprocessing.dct import extract_block_dct, compute_dct_features
from preprocessing.pipeline import PreprocessingPipeline


def test_dct_shape_and_range():
    # Synthetic test image 512x512
    img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    # 1. Block DCT
    raw_dct = extract_block_dct(img, block_size=8, num_ac_channels=21)
    assert raw_dct.shape == (64, 64, 21)
    assert not np.isnan(raw_dct).any()

    # 2. Resampled DCT for neural tensor
    spatial_dct = compute_dct_features(img, target_spatial_shape=(512, 512))
    assert spatial_dct.shape == (21, 512, 512)
    assert spatial_dct.dtype == np.float32


def test_clahe_preserves_dimensions():
    img = np.random.randint(200, 255, (256, 256, 3), dtype=np.uint8)
    enhanced = apply_forensic_clahe(img, clip_limit=2.0)
    assert enhanced.shape == img.shape
    assert enhanced.dtype == np.uint8


def test_deskew_correction():
    # Synthetic image with text line rotated by 4 degrees
    img = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.putText(img, "TEST DOCUMENT FORENSIC LINE", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    center = (150, 150)
    rot_mat = cv2.getRotationMatrix2D(center, 4.0, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, (300, 300), borderValue=(255, 255, 255))

    deskewed, angle = deskew_document(rotated, max_angle_deg=15.0)
    assert isinstance(angle, float)
    assert deskewed.shape == rotated.shape


def test_end_to_end_preprocessing_pipeline():
    pipeline = PreprocessingPipeline(target_size=(512, 512))
    sample_img = np.full((400, 600, 3), 240, dtype=np.uint8)
    cv2.putText(sample_img, "COMMERCIAL INVOICE", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)

    out = pipeline.process_image(sample_img)
    assert "rgb_tensor" in out
    assert "dct_tensor" in out
    assert out["rgb_tensor"].shape == (1, 3, 512, 512)
    assert out["dct_tensor"].shape == (1, 21, 512, 512)
    assert out["resized_512"].shape == (512, 512, 3)
