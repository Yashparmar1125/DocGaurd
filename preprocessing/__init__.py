"""DocGuard Preprocessing Package

Provides forensic-signal-preserving document loading, rectification, deskewing,
contrast enhancement, and 8x8 block Discrete Cosine Transform (DCT) extraction.
"""

from .document_loader import load_document, DocumentLoadingError
from .rectify import rectify_document_boundary
from .deskew import deskew_document
from .enhance import apply_forensic_clahe
from .dct import extract_block_dct, compute_dct_features
from .pipeline import PreprocessingPipeline

__all__ = [
    "load_document",
    "DocumentLoadingError",
    "rectify_document_boundary",
    "deskew_document",
    "apply_forensic_clahe",
    "extract_block_dct",
    "compute_dct_features",
    "PreprocessingPipeline",
]
