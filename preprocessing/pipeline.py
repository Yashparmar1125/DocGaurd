"""Preprocessing Pipeline Orchestrator

Chains document loading, perspective rectification, deskewing, conservative CLAHE,
and dual-stream (RGB + DCT) feature extraction into a unified inference/training preprocessor.
"""

from typing import Tuple, Dict, Any, Union
from pathlib import Path
import cv2
import numpy as np
import torch

from .document_loader import load_document
from .rectify import rectify_document_boundary
from .deskew import deskew_document
from .enhance import apply_forensic_clahe
from .dct import compute_dct_features


# Standard ImageNet normalization parameters for RGB stream
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PreprocessingPipeline:
    """End-to-end preprocessing pipeline for document forensic analysis."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        apply_rectify: bool = True,
        apply_deskew: bool = True,
        apply_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
    ):
        self.target_size = target_size
        self.apply_rectify = apply_rectify
        self.apply_deskew = apply_deskew
        self.apply_clahe = apply_clahe
        self.clahe_clip_limit = clahe_clip_limit

    def process_image(
        self, image: np.ndarray, metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Runs geometric and contrast preprocessing on an RGB image array.

        Args:
            image: Raw RGB numpy image (H, W, 3).
            metadata: Optional dictionary with document metadata.

        Returns:
            dict containing:
                - 'original': Original RGB image
                - 'rectified': Image after perspective rectification
                - 'deskewed': Image after skew correction
                - 'enhanced': Image after Luminance CLAHE
                - 'resized_512': 512x512 RGB uint8 image
                - 'rgb_tensor': Torch tensor (1, 3, 512, 512) normalized
                - 'dct_tensor': Torch tensor (1, 21, 512, 512) normalized
                - 'transform_info': Recorded transformation metrics
        """
        if metadata is None:
            metadata = {}

        transform_info = {
            "was_rectified": False,
            "skew_angle_deg": 0.0,
            "original_shape": image.shape[:2],
        }

        current = image.copy()

        # Step 1: Document boundary detection & perspective rectification
        if self.apply_rectify:
            rectified, was_rectified, _ = rectify_document_boundary(current)
            transform_info["was_rectified"] = was_rectified
            current = rectified
        rectified_view = current.copy()

        # Step 2: Deskew residual rotation
        if self.apply_deskew:
            deskewed, angle = deskew_document(current)
            transform_info["skew_angle_deg"] = angle
            current = deskewed
        deskewed_view = current.copy()

        # Step 3: Forensic-preserving Luminance CLAHE
        if self.apply_clahe:
            enhanced = apply_forensic_clahe(current, clip_limit=self.clahe_clip_limit)
            current = enhanced
        enhanced_view = current.copy()

        # Step 4: Fixed 512x512 sizing for deep backbone
        resized_512 = cv2.resize(
            current, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_AREA
        )

        # Step 5: Dual-Stream feature representations
        # RGB stream normalization (ImageNet mean/std)
        rgb_norm = (resized_512.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        rgb_tensor = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).float().unsqueeze(0)

        # Forensic stream: 8x8 block DCT features
        dct_features = compute_dct_features(resized_512, target_spatial_shape=self.target_size)
        dct_tensor = torch.from_numpy(dct_features).float().unsqueeze(0)

        return {
            "original": image,
            "rectified": rectified_view,
            "deskewed": deskewed_view,
            "enhanced": enhanced_view,
            "resized_512": resized_512,
            "rgb_tensor": rgb_tensor,
            "dct_tensor": dct_tensor,
            "transform_info": transform_info,
            "metadata": metadata,
        }

    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Loads document from disk and executes complete preprocessing pipeline."""
        raw_image, metadata = load_document(file_path)
        return self.process_image(raw_image, metadata)
