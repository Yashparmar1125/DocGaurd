"""DocGuard Deep Forensic Models Package

Contains:
- DocGuardModel: Dual-stream CNN-Transformer multi-task architecture
- Baselines: Baseline 1 (ELA + Random Forest) & Baseline 2 (Single-stream ResNet-50)
- Components: RGB backbone, DCT backbone, Cross-attention fusion, U-Net decoder, Heads
"""

from .docguard import DocGuardModel
from .docguard_transformer import DocGuardTransformerModel
try:
    from .baselines import ELARandomForestBaseline, PlainResNetBaseline
except (ImportError, OSError):
    ELARandomForestBaseline = None
    PlainResNetBaseline = None

__all__ = [
    "DocGuardModel",
    "DocGuardTransformerModel",
    "ELARandomForestBaseline",
    "PlainResNetBaseline",
]
