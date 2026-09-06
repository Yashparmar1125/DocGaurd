# DocGuard: Hybrid CNN-Transformer Framework for Document Forgery Detection, Localization, and Explanation

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**DocGuard** is an end-to-end deep document forensics system engineered to detect, localize, and explain digital document manipulation across financial, legal, and identity records (receipts, invoices, forms, certificates, and ID cards).

---

## Key Features & MVP Scope

* **Multi-Format Ingestion**: Scanned documents (`JPEG`, `PNG`, `TIFF`), mobile photos, and single-page `PDF`s (rasterized at 300 DPI via `PyMuPDF`).
* **Forensic Preprocessing**:
  * 4-point document boundary contour detection and perspective rectification (`cv2.warpPerspective`).
  * Hough-transform-based text deskewing.
  * Conservative Luminance CLAHE (clip limit 2.0) preserving double-compression micro-artifacts.
* **Dual-Stream Multi-Task Architecture**:
  * **RGB Stream**: Pretrained ResNet-50 feature extractor with multi-scale skip connections.
  * **Forensic Stream**: 8×8 block Discrete Cosine Transform (DCT) on Luminance (Y-channel) processed via a dedicated convolutional encoder.
  * **Fusion Block**: Multi-head cross-attention mechanism bridging spatial textures and frequency-domain compression discontinuities.
  * **U-Net Localization Decoder**: Full-resolution (512×512) pixel-level tampering segmentation mask.
  * **Dual Classification Heads**: Binary authentic vs. forged prediction + 5-way forgery-type classifier (`authentic`, `copy_move`, `splicing`, `text_tamper`, `erasure`).
* **Corroborating OCR Layout Engine**:
  * Word bounding box and baseline geometry extraction.
  * Statistical anomaly detector flagging font height mismatches and off-axis text baselines.
* **Explainable AI (XAI)**:
  * Grad-CAM attribution heatmaps from deep convolutional layers.
  * Temperature-scaled confidence calibration.
  * Natural language forensic audit rationale generator.
* **Downloadable PDF Forensic Audit Reports**:
  * Formal audit certificates featuring side-by-side visual evidence grids, OCR findings, and technical reasoning.
* **Interactive Web Workstation**:
  * Modern FastAPI backend with responsive dark-mode dashboard for real-time document analysis.

---

## System Architecture

```text
Input Document (Scan / Photo / PDF)
                │
        Preprocessing & Rectification
        (Deskew + CLAHE + Block DCT)
                │
       ┌────────┴────────┐
       ▼                 ▼
   RGB Stream     Forensic Stream
   (ResNet-50)     (8×8 Block DCT)
       └────────┬────────┘
                ▼
      Cross-Attention Fusion
                │
       ┌────────┴────────┐
       ▼                 ▼
Classification Heads   U-Net Localization Decoder
(Authentic/Forged +    (512×512 Pixel Mask)
 5-Way Forgery Type)
       └────────┬────────┘
                ▼
       OCR Anomaly Cross-Check
                ▼
      Explainable AI (Grad-CAM)
                ▼
      Confidence Calibration
                ▼
  Final Audit Report (JSON + PDF)
```

---

## Directory Structure

```text
AML_PRJT/
├── backend/            # FastAPI app, REST routes, schemas
├── data/               # Synthetic document generator, dataset loaders
├── explainability/     # Grad-CAM, temperature calibration, forensic reasoning
├── frontend/           # Interactive forensic dashboard UI
├── inference/          # Pipeline orchestrator, PDF report generator
├── localization/       # Mask post-processing, bounding box extraction
├── models/             # Dual-stream backbones, cross-attention, U-Net, baselines
├── ocr/                # OCR wrapper and typographical anomaly detector
├── preprocessing/      # PDF rasterizer, rectification, deskew, CLAHE, DCT
├── training/           # Multi-task loss functions, trainer, configs
├── tests/              # Pytest test suite
├── train.py            # Training entrypoint
├── run_benchmark.py    # Benchmark reproduction script
└── requirements.txt    # Dependencies
```

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Yashparmar1125/DocGaurd.git
cd DocGaurd

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Test Suite

```bash
pytest tests/ -v
```

### 3. Launch Web Application

```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
Open **http://localhost:8000** in your browser to inspect documents and generate audit reports.

### 4. Run Benchmark Evaluation

```bash
python run_benchmark.py --test-samples 50 --train-samples 100
```

### 5. Train / Fine-Tune Model

```bash
python train.py --epochs 10 --batch-size 8 --train-samples 300 --val-samples 60
```
