"""DocGuard Data Package

Provides synthetic document generation, controlled tamper simulation
(copy-move, splicing, text tampering, erasure, double JPEG compression),
and PyTorch Dataset/DataLoader implementations.
"""

from .generator import DocumentForgeryGenerator, ForgeryType
from .dataset import DocumentForgeryDataset, create_dataloaders

__all__ = [
    "DocumentForgeryGenerator",
    "ForgeryType",
    "DocumentForgeryDataset",
    "create_dataloaders",
]
