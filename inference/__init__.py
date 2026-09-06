"""DocGuard Inference Package

Orchestrates end-to-end document forensics, visual overlay generation,
and downloadable audit PDF report production.
"""

from .pipeline import DocGuardInferencePipeline
from .report_generator import generate_pdf_report

__all__ = [
    "DocGuardInferencePipeline",
    "generate_pdf_report",
]
