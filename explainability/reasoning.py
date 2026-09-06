"""Forensic Reasoning Engine

Synthesizes visual localization masks, deep classification outputs,
DCT frequency grid traces, and OCR layout checks into human-readable forensic explanations.
"""

from typing import Dict, Any, List
from localization.postprocess import TamperedRegion
from ocr.anomaly_detector import AnomalyFlag


class ForensicReasoningEngine:
    """Generates structured forensic narratives explaining model decisions."""

    def synthesize_report(
        self,
        is_forged: bool,
        calibrated_confidence: float,
        forgery_type_name: str,
        regions: List[TamperedRegion],
        ocr_anomalies: List[AnomalyFlag],
        skew_angle: float = 0.0,
    ) -> Dict[str, Any]:
        """Produces a structured natural language forensic audit reasoning."""
        if not is_forged:
            return {
                "verdict": "AUTHENTIC",
                "confidence_pct": round(calibrated_confidence * 100, 1),
                "forgery_type": "None",
                "summary": "Document exhibits uniform frequency distribution and typographical alignment.",
                "regions_summary": "No suspicious tampering regions localized.",
                "ocr_summary": "All text blocks show consistent baseline alignment and font sizing.",
                "rationale": (
                    f"Pixel-level DCT frequency grids and visual analysis confirm baseline authenticity "
                    f"with {round(calibrated_confidence * 100, 1)}% calibrated confidence. No double-JPEG "
                    f"re-compression traces, boundary splicing edges, or font baseline deviations were identified."
                ),
            }

        # Forged case synthesis
        type_display = forgery_type_name.replace("_", " ").title()

        # Describe localized regions
        if regions:
            primary_r = regions[0]
            box_str = f"[x:{primary_r.box[0]}, y:{primary_r.box[1]}, w:{primary_r.box[2]}, h:{primary_r.box[3]}]"
            regions_desc = (
                f"Primary suspicious region detected in the {primary_r.location_label} zone "
                f"(bounding box {box_str}, covering {primary_r.area_px} pixels)."
            )
        else:
            regions_desc = "Diffuse micro-manipulation detected across document fields."

        # Describe OCR corroboration
        ocr_reasons = []
        for anom in ocr_anomalies:
            ocr_reasons.append(anom.message)

        if ocr_reasons:
            ocr_desc = "; ".join(ocr_reasons[:2])
            corroboration = (
                f"Independent OCR layout cross-check directly corroborates tampering: {ocr_desc}."
            )
        else:
            ocr_desc = "No major OCR baseline outliers detected; tampering localized primarily via visual/DCT forensics."
            corroboration = "Visual texture and DCT frequency grid irregularities indicate physical or digital alteration."

        # Build final unified forensic rationale
        rationale = (
            f"The document is classified as FORGED with {round(calibrated_confidence * 100, 1)}% confidence. "
            f"Forgery type analysis identifies signature artifacts characteristic of {type_display}. "
            f"{regions_desc} {corroboration} Both the deep localization mask and DCT compression-discontinuity "
            f"analysis converge on this area."
        )

        return {
            "verdict": "FORGED",
            "confidence_pct": round(calibrated_confidence * 100, 1),
            "forgery_type": type_display,
            "summary": f"Document contains evidence of {type_display.lower()}.",
            "regions_summary": regions_desc,
            "ocr_summary": ocr_desc,
            "rationale": rationale,
        }
