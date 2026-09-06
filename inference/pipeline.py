"""End-to-End DocGuard Inference Pipeline

Executes the unified document forensics analysis flow:
1. Document ingestion (Image / PDF)
2. Forensic-preserving preprocessing & DCT extraction
3. Multi-task deep inference (DocGuard)
4. Mask cleanup & region bounding-box extraction
5. OCR anomaly detection
6. Grad-CAM attribution heatmap generation
7. Post-hoc confidence calibration
8. Natural language reasoning synthesis
"""

from typing import Dict, Any, Union, Optional
from pathlib import Path
import numpy as np
import cv2
import torch

from preprocessing.pipeline import PreprocessingPipeline
from models.docguard import DocGuardModel
from localization.postprocess import PostProcessor
from ocr.anomaly_detector import OCRAnomalyDetector
from explainability.gradcam import GradCAM
from explainability.calibration import TemperatureScalingCalibrator
from explainability.reasoning import ForensicReasoningEngine
from data.generator import FORGERY_TYPE_NAMES, ForgeryType


class DocGuardInferencePipeline:
    """Production-ready inference orchestrator for document tamper detection."""

    def __init__(
        self,
        model: Optional[DocGuardModel] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize model
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = DocGuardModel(pretrained_backbone=False).to(self.device)
            if checkpoint_path and Path(checkpoint_path).exists():
                ckpt = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.preprocessor = PreprocessingPipeline(target_size=(512, 512))
        self.postprocessor = PostProcessor(threshold=0.5)
        self.ocr_detector = OCRAnomalyDetector()
        self.gradcam = GradCAM(self.model)
        self.calibrator = TemperatureScalingCalibrator(initial_temperature=1.2)
        self.reasoning_engine = ForensicReasoningEngine()

    def analyze(self, file_path_or_image: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """Runs complete forensic audit on a document.

        Args:
            file_path_or_image: File path (Image/PDF) or RGB numpy array.

        Returns:
            dict containing all forensic findings, scores, overlays, and explanation.
        """
        # Step 1: Preprocessing
        if isinstance(file_path_or_image, (str, Path)):
            prep_out = self.preprocessor.process_file(file_path_or_image)
        else:
            prep_out = self.preprocessor.process_image(file_path_or_image)

        rgb_tensor = prep_out["rgb_tensor"].to(self.device)
        dct_tensor = prep_out["dct_tensor"].to(self.device)
        resized_512 = prep_out["resized_512"]

        # Step 2: Deep Model Forward Pass
        with torch.no_grad():
            outputs = self.model(rgb_tensor, dct_tensor)

        mask_prob_np = outputs["mask_prob"].squeeze().cpu().numpy()  # (512, 512)
        binary_logit = outputs["binary_logits"].item()
        type_probs = outputs["type_prob"].squeeze().cpu().numpy()     # (5,)
        pred_type_idx = int(np.argmax(type_probs))

        # Step 3: Localization Postprocessing
        clean_mask, regions = self.postprocessor.process(mask_prob_np)

        # Decision rule: document is forged if binary classifier triggers OR non-trivial mask exists
        has_tampered_mask = len(regions) > 0 and regions[0].area_px > 100
        raw_prob = 1.0 / (1.0 + np.exp(-binary_logit))
        is_forged = bool(raw_prob > 0.5 or has_tampered_mask)

        # Step 4: Confidence Calibration
        calibrated_prob, confidence_score = self.calibrator.calibrate(binary_logit)
        if has_tampered_mask and not is_forged:
            is_forged = True
            confidence_score = max(confidence_score, 0.85)

        # Adjust type if authentic
        if not is_forged:
            pred_type_name = "authentic"
            pred_type_idx = 0
        else:
            # If type predicted authentic but visual mask triggered, default to most likely tamper
            if pred_type_idx == 0:
                pred_type_idx = int(np.argmax(type_probs[1:])) + 1
            pred_type_name = FORGERY_TYPE_NAMES.get(ForgeryType(pred_type_idx), "text_tamper")

        # Step 5: OCR Layout Anomaly Cross-Check
        ocr_anomalies, ocr_mask = self.ocr_detector.detect_anomalies(resized_512)

        # Step 6: Grad-CAM Heatmap Generation
        try:
            cam_norm, cam_color_rgb = self.gradcam.generate_heatmap(rgb_tensor, dct_tensor)
        except Exception:
            cam_norm = np.zeros((512, 512), dtype=np.float32)
            cam_color_rgb = np.zeros((512, 512, 3), dtype=np.uint8)

        # Step 7: Create Visual Overlays
        # Mask overlay (red highlight on resized document)
        mask_overlay = resized_512.copy()
        mask_indices = clean_mask > 0
        mask_overlay[mask_indices] = (
            mask_overlay[mask_indices].astype(np.float32) * 0.4
            + np.array([255, 30, 30], dtype=np.float32) * 0.6
        ).astype(np.uint8)

        # Draw bounding boxes on mask overlay
        for reg in regions:
            rx, ry, rw, rh = reg.box
            cv2.rectangle(mask_overlay, (rx, ry), (rx + rw, ry + rh), (255, 220, 0), 2)
            cv2.putText(
                mask_overlay,
                f"{reg.location_label} ({reg.mean_confidence:.2f})",
                (rx, max(ry - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1,
            )

        # Grad-CAM overlay blended on document
        cam_overlay = (resized_512.astype(np.float32) * 0.5 + cam_color_rgb.astype(np.float32) * 0.5).astype(np.uint8)

        # Step 8: Forensic Reasoning Synthesis
        reasoning_data = self.reasoning_engine.synthesize_report(
            is_forged=is_forged,
            calibrated_confidence=confidence_score,
            forgery_type_name=pred_type_name,
            regions=regions,
            ocr_anomalies=ocr_anomalies,
            skew_angle=prep_out["transform_info"]["skew_angle_deg"],
        )

        return {
            "verdict": reasoning_data["verdict"],
            "is_forged": is_forged,
            "confidence_pct": reasoning_data["confidence_pct"],
            "forgery_type": reasoning_data["forgery_type"],
            "summary": reasoning_data["summary"],
            "rationale": reasoning_data["rationale"],
            "regions_summary": reasoning_data["regions_summary"],
            "ocr_summary": reasoning_data["ocr_summary"],
            "regions": [
                {"box": r.box, "area": r.area_px, "confidence": r.mean_confidence, "label": r.location_label}
                for r in regions
            ],
            "ocr_anomalies": [
                {"type": a.anomaly_type, "message": a.message, "severity": a.severity}
                for a in ocr_anomalies
            ],
            "transform_info": prep_out["transform_info"],
            "images": {
                "original": prep_out["original"],
                "rectified": prep_out["rectified"],
                "resized_512": resized_512,
                "clean_mask": clean_mask,
                "mask_overlay": mask_overlay,
                "gradcam_overlay": cam_overlay,
            },
        }
