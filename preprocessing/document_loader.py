"""Document Loader Module

Validates, inspects, and converts input document formats (scans, smartphone photos,
and single-page PDFs) into standardized RGB numpy images at forensic resolution (300 DPI).
"""

from pathlib import Path
from typing import Union, Tuple
import numpy as np
from PIL import Image, ImageOps
import pymupdf as fitz  # PyMuPDF


class DocumentLoadingError(Exception):
    """Raised when an input document fails validation or conversion."""
    pass


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
MIN_FORENSIC_DIMENSION = 256  # Minimum dimension for forensic feature validity


def load_document(file_path: Union[str, Path], target_dpi: int = 300) -> Tuple[np.ndarray, dict]:
    """Load a document image or PDF page into an RGB numpy array.

    Args:
        file_path: Path to the document image or PDF.
        target_dpi: DPI to rasterize PDFs at (default 300 DPI).

    Returns:
        tuple: (rgb_image_array uint8, metadata_dict)
    """
    path = Path(file_path)
    if not path.exists():
        raise DocumentLoadingError(f"File not found: {path}")

    suffix = path.suffix.lower()
    metadata = {
        "filename": path.name,
        "extension": suffix,
        "source_path": str(path.resolve()),
        "original_type": "pdf" if suffix == ".pdf" else "image"
    }

    if suffix == ".pdf":
        return _load_pdf(path, target_dpi, metadata)
    elif suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return _load_image(path, metadata)
    else:
        raise DocumentLoadingError(
            f"Unsupported document format '{suffix}'. Supported: PDF, "
            f"{', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
        )


def _load_image(path: Path, metadata: dict) -> Tuple[np.ndarray, dict]:
    try:
        with Image.open(path) as img:
            # Respect EXIF orientation tag if present
            img = ImageOps.exif_transpose(img)
            # Ensure RGB mode
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.array(img, dtype=np.uint8)

        h, w = arr.shape[:2]
        if min(h, w) < MIN_FORENSIC_DIMENSION:
            raise DocumentLoadingError(
                f"Resolution ({w}x{h}) is too low for forensic tamper detection. "
                f"Minimum shortest side is {MIN_FORENSIC_DIMENSION}px."
            )

        metadata.update({"width": w, "height": h, "channels": 3, "pages": 1})
        return arr, metadata
    except Exception as e:
        if isinstance(e, DocumentLoadingError):
            raise
        raise DocumentLoadingError(f"Failed to read image '{path.name}': {str(e)}") from e


def _load_pdf(path: Path, target_dpi: int, metadata: dict) -> Tuple[np.ndarray, dict]:
    try:
        doc = fitz.open(str(path))
        if len(doc) == 0:
            raise DocumentLoadingError(f"PDF '{path.name}' contains 0 pages.")

        page = doc[0]  # Single-page MVP scope
        # Standard PDF points are 72 per inch
        zoom = target_dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert pixmap buffer to numpy RGB array
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
        metadata.update({
            "width": pix.width,
            "height": pix.height,
            "channels": 3,
            "pages": len(doc),
            "rendered_dpi": target_dpi
        })
        doc.close()
        return arr, metadata
    except Exception as e:
        if isinstance(e, DocumentLoadingError):
            raise
        raise DocumentLoadingError(f"Failed to rasterize PDF '{path.name}': {str(e)}") from e
