"""DocGuard Data Package

Provides synthetic document generation, controlled tamper simulation
(copy-move, splicing, text tampering, erasure, double JPEG compression),
and PyTorch Dataset/DataLoader implementations.
"""

from .generator import DocumentForgeryGenerator, ForgeryType
from .dataset import DocumentForgeryDataset, create_dataloaders
from .benchmark_dataset import BenchmarkDocumentDataset, create_benchmark_dataloaders
from .doctamper_dataset import DocTamperLMDBDataset, create_doctamper_dataloaders

__all__ = [
    "DocumentForgeryGenerator",
    "ForgeryType",
    "DocumentForgeryDataset",
    "create_dataloaders",
    "BenchmarkDocumentDataset",
    "create_benchmark_dataloaders",
    "DocTamperLMDBDataset",
    "create_doctamper_dataloaders",
]
