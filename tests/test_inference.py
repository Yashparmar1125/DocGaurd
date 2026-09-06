"""End-to-End Integration Tests for DocGuard Inference and PDF Generation"""

from pathlib import Path
import numpy as np
import pytest

from inference.pipeline import DocGuardInferencePipeline
from inference.report_generator import generate_pdf_report
from data.generator import DocumentForgeryGenerator, ForgeryType


def test_end_to_end_inference_and_pdf(tmp_path):
    # Instantiate inference pipeline (using CPU for reliable test execution)
    pipeline = DocGuardInferencePipeline(device="cpu")
    generator = DocumentForgeryGenerator()

    # 1. Test authentic sample
    auth_sample = generator.generate_sample(forgery_type=ForgeryType.AUTHENTIC)
    res_auth = pipeline.analyze(auth_sample["image"])

    assert "verdict" in res_auth
    assert "confidence_pct" in res_auth
    assert "forgery_type" in res_auth
    assert "rationale" in res_auth
    assert "images" in res_auth
    assert res_auth["images"]["mask_overlay"].shape == (512, 512, 3)

    # 2. Test forged sample
    forged_sample = generator.generate_sample(forgery_type=ForgeryType.COPY_MOVE)
    res_forged = pipeline.analyze(forged_sample["image"])
    assert "verdict" in res_forged
    assert isinstance(res_forged["regions"], list)

    # 3. Test PDF generation
    pdf_dest = tmp_path / "test_report.pdf"
    out_file = generate_pdf_report(res_forged, str(pdf_dest))
    assert Path(out_file).exists()
    assert Path(out_file).stat().st_size > 1000  # Non-trivial PDF size
