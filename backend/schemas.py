"""Pydantic API Schemas for DocGuard Backend"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class RegionSchema(BaseModel):
    box: List[int]
    area: int
    confidence: float
    label: str


class OCRAnomalySchema(BaseModel):
    type: str
    message: str
    severity: float


class TransformInfoSchema(BaseModel):
    was_rectified: bool
    skew_angle_deg: float
    original_shape: List[int]


class ForensicAnalysisResponse(BaseModel):
    report_id: str
    verdict: str
    is_forged: bool
    confidence_pct: float
    forgery_type: str
    summary: str
    rationale: str
    regions_summary: str
    ocr_summary: str
    regions: List[RegionSchema]
    ocr_anomalies: List[OCRAnomalySchema]
    transform_info: TransformInfoSchema
    images_base64: Dict[str, str]  # 'original', 'rectified', 'mask_overlay', 'gradcam_overlay', 'clean_mask'


class HealthResponse(BaseModel):
    status: str
    device: str
    gpu_name: Optional[str] = None
    model_loaded: bool
