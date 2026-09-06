"""DocGuard Localization Package

Handles mask thresholding, morphological cleanup, contour extraction,
and bounding-box geographic labeling for forensic reporting.
"""

from .postprocess import PostProcessor, TamperedRegion

__all__ = ["PostProcessor", "TamperedRegion"]
