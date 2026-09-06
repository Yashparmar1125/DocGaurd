"""Unit Tests for OCR Anomaly Detector and Engine"""

import numpy as np
import cv2
import pytest

from ocr.engine import OCREngine
from ocr.anomaly_detector import OCRAnomalyDetector
from data.generator import DocumentForgeryGenerator, ForgeryType


def test_ocr_engine_extracts_boxes():
    engine = OCREngine()
    gen = DocumentForgeryGenerator()
    doc, _ = gen.create_authentic_document(doc_type="invoice")

    boxes = engine.extract_text_boxes(doc)
    assert isinstance(boxes, list)
    assert len(boxes) > 0
    for b in boxes:
        assert b.w > 0 and b.h > 0
        assert b.baseline_y >= b.y


def test_ocr_anomaly_detection_on_tampered_text():
    gen = DocumentForgeryGenerator()
    detector = OCRAnomalyDetector()

    # Generate an authentic sample and tamper with text
    auth, meta = gen.create_authentic_document(doc_type="receipt")
    tampered, mask, f_type = gen.apply_text_tamper(auth, meta)

    anomalies, anom_mask = detector.detect_anomalies(tampered)
    assert isinstance(anomalies, list)
    assert anom_mask.shape == tampered.shape[:2]
