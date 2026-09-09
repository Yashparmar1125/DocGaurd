"""FastAPI Application Server for DocGuard

Provides endpoints for document analysis, forensic PDF report generation,
and static frontend dashboard serving.
"""

from pathlib import Path
import io
import uuid
import base64
from typing import Dict, Any, Optional
import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from inference.pipeline import DocGuardInferencePipeline
from inference.report_generator import generate_pdf_report
from data.generator import DocumentForgeryGenerator, ForgeryType
from .schemas import ForensicAnalysisResponse, HealthResponse

# Application Setup
app = FastAPI(
    title="DocGuard Forensic API",
    description="Deep Multi-Task Framework for Document Forgery Detection, Localization, and Explanation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary directory for generated PDF audit certificates
REPORTS_DIR = Path("reports_cache")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline Singleton
_pipeline: DocGuardInferencePipeline = None


def get_pipeline() -> DocGuardInferencePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DocGuardInferencePipeline()
    return _pipeline


def ndarray_to_base64_jpg(img: np.ndarray, quality: int = 90) -> str:
    """Encodes a uint8 RGB or grayscale numpy array into base64 JPEG string."""
    if len(img.shape) == 3:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        bgr = img
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, enc = cv2.imencode(".jpg", bgr, encode_param)
    if not success:
        return ""
    return base64.b64encode(enc).decode("utf-8")


@app.get("/api/health", response_model=HealthResponse)
def health():
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    return HealthResponse(
        status="healthy",
        device="cuda" if gpu_available else "cpu",
        gpu_name=gpu_name,
        model_loaded=get_pipeline() is not None,
    )


@app.post("/api/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """Uploads document image or PDF, runs full inference, and returns forensic findings."""
    pipeline = get_pipeline()
    contents = await file.read()
    suffix = Path(file.filename).suffix.lower()

    # Save to a temporary file for PyMuPDF or OpenCV loading
    temp_file_path = REPORTS_DIR / f"upload_{uuid.uuid4().hex}{suffix}"
    try:
        with open(temp_file_path, "wb") as f:
            f.write(contents)

        result = pipeline.analyze(temp_file_path)

        # Generate unique report ID and compile PDF
        report_id = uuid.uuid4().hex
        pdf_path = REPORTS_DIR / f"DocGuard_Report_{report_id}.pdf"
        generate_pdf_report(result, str(pdf_path))

        # Encode images to base64 for direct browser rendering
        imgs = result["images"]
        images_base64 = {
            "original": ndarray_to_base64_jpg(imgs["original"]),
            "rectified": ndarray_to_base64_jpg(imgs["rectified"]),
            "mask_overlay": ndarray_to_base64_jpg(imgs["mask_overlay"]),
            "gradcam_overlay": ndarray_to_base64_jpg(imgs["gradcam_overlay"]),
            "clean_mask": ndarray_to_base64_jpg(imgs["clean_mask"]),
        }

        response_data = {
            "report_id": report_id,
            "verdict": result["verdict"],
            "is_forged": result["is_forged"],
            "confidence_pct": result["confidence_pct"],
            "forgery_type": result["forgery_type"],
            "summary": result["summary"],
            "rationale": result["rationale"],
            "regions_summary": result["regions_summary"],
            "ocr_summary": result["ocr_summary"],
            "regions": result["regions"],
            "ocr_anomalies": result["ocr_anomalies"],
            "transform_info": result["transform_info"],
            "images_base64": images_base64,
        }
        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forensic analysis failed: {str(e)}")
    finally:
        if temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception:
                pass


@app.post("/api/sample")
def analyze_synthetic_sample(forgery_type: Optional[str] = None):
    """Generates a controlled synthetic document (authentic or forged) and analyzes it."""
    generator = DocumentForgeryGenerator()
    type_map = {
        "authentic": ForgeryType.AUTHENTIC,
        "copy_move": ForgeryType.COPY_MOVE,
        "splicing": ForgeryType.SPLICING,
        "text_tamper": ForgeryType.TEXT_TAMPER,
        "erasure": ForgeryType.ERASURE,
    }
    selected_type = type_map.get(forgery_type, None)
    sample = generator.generate_sample(forgery_type=selected_type)

    pipeline = get_pipeline()
    result = pipeline.analyze(sample["image"])

    report_id = uuid.uuid4().hex
    pdf_path = REPORTS_DIR / f"DocGuard_Report_{report_id}.pdf"
    generate_pdf_report(result, str(pdf_path))

    imgs = result["images"]
    images_base64 = {
        "original": ndarray_to_base64_jpg(imgs["original"]),
        "rectified": ndarray_to_base64_jpg(imgs["rectified"]),
        "mask_overlay": ndarray_to_base64_jpg(imgs["mask_overlay"]),
        "gradcam_overlay": ndarray_to_base64_jpg(imgs["gradcam_overlay"]),
        "clean_mask": ndarray_to_base64_jpg(imgs["clean_mask"]),
    }

    return JSONResponse(content={
        "report_id": report_id,
        "verdict": result["verdict"],
        "is_forged": result["is_forged"],
        "confidence_pct": result["confidence_pct"],
        "forgery_type": result["forgery_type"],
        "summary": result["summary"],
        "rationale": result["rationale"],
        "regions_summary": result["regions_summary"],
        "ocr_summary": result["ocr_summary"],
        "regions": result["regions"],
        "ocr_anomalies": result["ocr_anomalies"],
        "transform_info": result["transform_info"],
        "images_base64": images_base64,
        "ground_truth_type": sample["forgery_type_name"],
    })


@app.get("/api/report/{report_id}")
def download_report(report_id: str):
    """Downloads the generated forensic PDF certificate."""
    pdf_path = REPORTS_DIR / f"DocGuard_Report_{report_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Forensic report not found.")
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=f"DocGuard_Forensic_Report_{report_id}.pdf",
    )


# Mount Static Frontend
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
