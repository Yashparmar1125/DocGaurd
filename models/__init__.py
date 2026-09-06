"""DocGuard Deep Forensic Models Package

Contains:
- DocGuardModel: Dual-stream CNN-Transformer multi-task architecture
- Baselines: Baseline 1 (ELA + Random Forest) & Baseline 2 (Single-stream ResNet-50)
- Components: RGB backbone, DCT backbone, Cross-attention fusion, U-Net decoder, Heads
"""

from .docguard import DocGuardModel
from .baselines import ELARandomForestBaseline, PlainResNetBaseline

__all__ = [
    "DocGuardModel",
    "ELARandomForestBaseline",
    "PlainResNetBaseline",
]
