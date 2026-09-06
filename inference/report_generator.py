"""PDF Forensic Audit Report Generator

Compiles complete document inspection results, visual overlays, OCR corroboration,
and reasoning narratives into a formal downloadable PDF certificate.
"""

from typing import Dict, Any
from pathlib import Path
import tempfile
import cv2
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_pdf_report(analysis_result: Dict[str, Any], output_path: str) -> str:
    """Renders analysis dictionary into a high-quality PDF report.

    Args:
        analysis_result: Result dictionary returned by DocGuardInferencePipeline.analyze.
        output_path: Destination file path.

    Returns:
        str: Absolute path of generated PDF.
    """
    dest_path = Path(output_path).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(dest_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocGuardTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "DocGuardSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=14,
    )
    heading2_style = ParagraphStyle(
        "DocGuardH2",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2C5282"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "DocGuardBody",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#2D3748"),
    )

    elements = []

    # Title Banner
    elements.append(Paragraph("DocGuard: Forensic Document Analysis Certificate", title_style))
    elements.append(
        Paragraph("Automated Deep Multi-Task Forensic Inspection & Tamper Localization", subtitle_style)
    )

    # Key Verdict Card Table
    verdict = analysis_result["verdict"]
    is_forged = analysis_result["is_forged"]
    verdict_color = colors.HexColor("#E53E3E") if is_forged else colors.HexColor("#38A169")
    confidence_str = f"{analysis_result['confidence_pct']}%"

    summary_data = [
        [
            Paragraph("<b>FORENSIC VERDICT</b>", body_style),
            Paragraph(f"<font color='{verdict_color.hexval()}'><b>{verdict}</b></font>", body_style),
            Paragraph("<b>CALIBRATED CONFIDENCE</b>", body_style),
            Paragraph(f"<b>{confidence_str}</b>", body_style),
        ],
        [
            Paragraph("<b>DETECTED FORGERY TYPE</b>", body_style),
            Paragraph(f"<b>{analysis_result['forgery_type']}</b>", body_style),
            Paragraph("<b>DOCUMENT ORIENTATION</b>", body_style),
            Paragraph(f"Deskew: {analysis_result['transform_info'].get('skew_angle_deg', 0.0):.1f}°", body_style),
        ],
    ]
    summary_table = Table(summary_data, colWidths=[140, 130, 140, 130])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 14))

    # Save temp images for report embedding
    temp_dir = tempfile.TemporaryDirectory()
    t_path = Path(temp_dir.name)

    imgs = analysis_result["images"]
    rect_img_path = t_path / "rectified.jpg"
    mask_img_path = t_path / "mask_overlay.jpg"
    cam_img_path = t_path / "cam_overlay.jpg"
    clean_mask_path = t_path / "clean_mask.jpg"

    cv2.imwrite(str(rect_img_path), cv2.cvtColor(imgs["rectified"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_img_path), cv2.cvtColor(imgs["mask_overlay"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(cam_img_path), cv2.cvtColor(imgs["gradcam_overlay"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(clean_mask_path), imgs["clean_mask"])

    # Visual Forensic Evidence Grid (2x2)
    elements.append(Paragraph("Visual Forensic Evidence", heading2_style))

    img_w, img_h = 250, 175
    img_grid = [
        [
            Paragraph("<b>1. Rectified / Enhanced Document</b>", body_style),
            Paragraph("<b>2. Pixel Tampering Localization Mask</b>", body_style),
        ],
        [
            RLImage(str(rect_img_path), width=img_w, height=img_h),
            RLImage(str(mask_img_path), width=img_w, height=img_h),
        ],
        [
            Paragraph("<b>3. Binary Segmentation Output</b>", body_style),
            Paragraph("<b>4. Grad-CAM Deep Attribution Heatmap</b>", body_style),
        ],
        [
            RLImage(str(clean_mask_path), width=img_w, height=img_h),
            RLImage(str(cam_img_path), width=img_w, height=img_h),
        ],
    ]
    grid_table = Table(img_grid, colWidths=[270, 270])
    grid_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
    ]))
    elements.append(grid_table)
    elements.append(Spacer(1, 12))

    # Findings & Rationale Section
    elements.append(Paragraph("Forensic Audit Findings & Rationale", heading2_style))
    elements.append(Paragraph(f"<b>Executive Summary:</b> {analysis_result['summary']}", body_style))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(f"<b>Spatial Localization:</b> {analysis_result['regions_summary']}", body_style))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(f"<b>OCR Layout Corroboration:</b> {analysis_result['ocr_summary']}", body_style))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(f"<b>Technical Rationale:</b> {analysis_result['rationale']}", body_style))
    elements.append(Spacer(1, 14))

    # Footer note
    footer_text = (
        "<i>Notice: Generated by DocGuard Hybrid Deep Forensic Engine. Intended for evidentiary support "
        "and compliance screening under ISO/IEC forensics guidelines.</i>"
    )
    elements.append(Paragraph(footer_text, ParagraphStyle("Footer", parent=styles["Italic"], fontSize=8, textColor=colors.HexColor("#718096"))))

    doc.build(elements)
    temp_dir.cleanup()
    return str(dest_path)
