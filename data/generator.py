"""Synthetic Document Forgery Generator

Generates high-fidelity authentic document templates (invoices, receipts, certificates)
and simulates pixel-accurate forgeries:
- Copy-Move (rotation, scaling, boundary blending)
- Splicing (inter-document patch transfer with color histogram matching)
- Text Tampering (inpainting + mismatched font/kerning re-rendering)
- Erasure (Navier-Stokes/Telea inpainting)
- Double-JPEG compression artifact injection
"""

from enum import IntEnum
from typing import Tuple, Dict, Any, Optional
import io
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


class ForgeryType(IntEnum):
    AUTHENTIC = 0
    COPY_MOVE = 1
    SPLICING = 2
    TEXT_TAMPER = 3
    ERASURE = 4


FORGERY_TYPE_NAMES = {
    ForgeryType.AUTHENTIC: "authentic",
    ForgeryType.COPY_MOVE: "copy_move",
    ForgeryType.SPLICING: "splicing",
    ForgeryType.TEXT_TAMPER: "text_tamper",
    ForgeryType.ERASURE: "erasure",
}


class DocumentForgeryGenerator:
    """Simulates realistic document templates and applies controlled tamper operations."""

    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size
        self._font_cache = {}

    def _get_font(self, size: int = 14) -> ImageFont.ImageFont:
        """Returns standard TrueType or default bitmap font."""
        if size not in self._font_cache:
            try:
                # Common system fonts on Windows
                font = ImageFont.truetype("arial.ttf", size)
            except Exception:
                try:
                    font = ImageFont.truetype("calibri.ttf", size)
                except Exception:
                    font = ImageFont.load_default()
            self._font_cache[size] = font
        return self._font_cache[size]

    def create_authentic_document(self, doc_type: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Synthesizes a realistic authentic document (invoice, receipt, or certificate).

        Args:
            doc_type: One of ['invoice', 'receipt', 'certificate'] or None (random).

        Returns:
            tuple: (RGB image np.ndarray (512, 512, 3), metadata_dict)
        """
        if doc_type is None:
            doc_type = random.choice(["invoice", "receipt", "certificate"])

        w, h = self.target_size
        # Off-white / paper background with subtle paper grain texture
        base_color = random.randint(245, 255)
        img_arr = np.full((h, w, 3), base_color, dtype=np.uint8)

        # Add mild random paper texture noise
        noise = np.random.normal(0, 1.5, (h, w, 3))
        img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)

        pil_img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(pil_img)

        text_boxes = []

        if doc_type == "invoice":
            # Header
            draw.rectangle([20, 20, w - 20, 50], fill=(40, 70, 120))
            draw.text((30, 26), "COMMERCIAL INVOICE", fill=(255, 255, 255), font=self._get_font(18))

            # Invoice info
            inv_no = f"INV-2026-{random.randint(1000, 9999)}"
            date_str = f"Date: 2026-0{random.randint(1, 9)}-{random.randint(10, 28)}"
            draw.text((30, 65), f"Invoice #: {inv_no}", fill=(30, 30, 30), font=self._get_font(12))
            draw.text((w - 180, 65), date_str, fill=(30, 30, 30), font=self._get_font(12))
            text_boxes.append((w - 180, 65, w - 40, 80, date_str))

            # Bill To box
            draw.rectangle([30, 90, 230, 160], outline=(180, 180, 180), width=1)
            draw.text((35, 95), "BILL TO:", fill=(80, 80, 80), font=self._get_font(10))
            draw.text((35, 110), "Acme Global Corp.", fill=(20, 20, 20), font=self._get_font(12))
            draw.text((35, 128), "124 Innovation Way", fill=(50, 50, 50), font=self._get_font(11))
            draw.text((35, 144), "San Jose, CA 95110", fill=(50, 50, 50), font=self._get_font(11))

            # Table Header
            draw.rectangle([20, 180, w - 20, 205], fill=(220, 225, 235))
            draw.text((30, 185), "ITEM DESCRIPTION", fill=(30, 30, 30), font=self._get_font(11))
            draw.text((280, 185), "QTY", fill=(30, 30, 30), font=self._get_font(11))
            draw.text((350, 185), "UNIT", fill=(30, 30, 30), font=self._get_font(11))
            draw.text((420, 185), "TOTAL", fill=(30, 30, 30), font=self._get_font(11))

            # Line items
            items = [
                ("Cloud Compute Tier A", 2, 450.00),
                ("Database Replication Node", 1, 320.00),
                ("Dedicated SSL Certificate", 3, 45.00),
                ("Enterprise Forensic License", 1, 850.00),
            ]
            y_offset = 215
            subtotal = 0.0
            for desc, qty, rate in items:
                line_tot = qty * rate
                subtotal += line_tot
                draw.text((30, y_offset), desc, fill=(40, 40, 40), font=self._get_font(11))
                draw.text((285, y_offset), str(qty), fill=(40, 40, 40), font=self._get_font(11))
                draw.text((345, y_offset), f"${rate:.2f}", fill=(40, 40, 40), font=self._get_font(11))
                draw.text((415, y_offset), f"${line_tot:.2f}", fill=(40, 40, 40), font=self._get_font(11))
                draw.line([(20, y_offset + 20), (w - 20, y_offset + 20)], fill=(230, 230, 230), width=1)
                y_offset += 25

            # Summary Box
            draw.rectangle([300, y_offset + 10, w - 20, y_offset + 85], outline=(150, 150, 150), width=1)
            draw.text((310, y_offset + 15), "Subtotal:", fill=(50, 50, 50), font=self._get_font(11))
            draw.text((420, y_offset + 15), f"${subtotal:.2f}", fill=(50, 50, 50), font=self._get_font(11))
            tax = subtotal * 0.0825
            total = subtotal + tax
            draw.text((310, y_offset + 35), "Tax (8.25%):", fill=(50, 50, 50), font=self._get_font(11))
            draw.text((420, y_offset + 35), f"${tax:.2f}", fill=(50, 50, 50), font=self._get_font(11))

            tot_str = f"${total:.2f}"
            draw.text((310, y_offset + 60), "TOTAL DUE:", fill=(10, 10, 10), font=self._get_font(12))
            draw.text((415, y_offset + 60), tot_str, fill=(180, 20, 20), font=self._get_font(13))
            text_boxes.append((415, y_offset + 60, w - 25, y_offset + 80, tot_str))

            # Stamp
            self._draw_stamp(draw, (100, y_offset + 40), "APPROVED", (30, 140, 60))

        elif doc_type == "receipt":
            # Store Header
            draw.text((160, 25), "METRO GROCERS & PHARMACY", fill=(20, 20, 20), font=self._get_font(14))
            draw.text((180, 45), "Store #4029 - San Jose, CA", fill=(60, 60, 60), font=self._get_font(10))
            draw.line([(30, 65), (w - 30, 65)], fill=(100, 100, 100), width=1)

            # Receipt Date & Cashier
            date_str = "09/06/2026 14:32"
            draw.text((40, 75), f"DATE: {date_str}", fill=(40, 40, 40), font=self._get_font(10))
            draw.text((w - 180, 75), "CASHIER: Sarah M.", fill=(40, 40, 40), font=self._get_font(10))
            text_boxes.append((40, 75, 180, 90, date_str))

            # Line items
            r_items = [
                ("ORGANIC MILK 1 GAL", 4.99),
                ("ARTISAN SOURDOUGH", 3.49),
                ("COLOMBIA ROAST COFFEE", 12.99),
                ("PROTEIN ENERGY BARS", 8.49),
                ("ASPIRIN 100CT", 6.29),
            ]
            y = 105
            r_tot = 0.0
            for name, price in r_items:
                r_tot += price
                draw.text((40, y), name, fill=(30, 30, 30), font=self._get_font(11))
                draw.text((w - 90, y), f"${price:.2f}", fill=(30, 30, 30), font=self._get_font(11))
                y += 22

            draw.line([(30, y + 5), (w - 30, y + 5)], fill=(120, 120, 120), width=1)
            y += 15
            tot_str = f"${r_tot:.2f}"
            draw.text((40, y), "TOTAL AMOUNT:", fill=(10, 10, 10), font=self._get_font(13))
            draw.text((w - 100, y), tot_str, fill=(10, 10, 10), font=self._get_font(13))
            text_boxes.append((w - 100, y, w - 30, y + 20, tot_str))

            # Barcode lines simulation
            for bx in range(80, w - 80, 4):
                if random.random() > 0.3:
                    draw.line([(bx, y + 50), (bx, y + 90)], fill=(0, 0, 0), width=random.choice([1, 2, 3]))

        else:  # Certificate
            draw.rectangle([20, 20, w - 20, h - 20], outline=(160, 120, 40), width=3)
            draw.rectangle([26, 26, w - 26, h - 26], outline=(190, 160, 80), width=1)
            draw.text((120, 60), "CERTIFICATE OF ACHIEVEMENT", fill=(140, 100, 30), font=self._get_font(16))
            draw.text((180, 100), "This is proudly presented to:", fill=(70, 70, 70), font=self._get_font(11))
            name_str = "Dr. Alexander V. Wright"
            draw.text((140, 130), name_str, fill=(20, 20, 20), font=self._get_font(16))
            text_boxes.append((140, 130, 360, 160, name_str))
            draw.line([(120, 165), (w - 120, 165)], fill=(180, 150, 70), width=1)

            draw.text((110, 190), "For outstanding scientific contributions to", fill=(60, 60, 60), font=self._get_font(11))
            draw.text((130, 210), "Deep Document Image Forensics & Security", fill=(30, 30, 30), font=self._get_font(12))

            # Date & Signature
            date_str = "Dated: September 06, 2026"
            draw.text((50, 320), date_str, fill=(60, 60, 60), font=self._get_font(11))
            text_boxes.append((50, 320, 220, 340, date_str))

            # Stamp
            self._draw_stamp(draw, (380, 320), "OFFICIAL SEAL", (160, 40, 40))

        doc_array = np.array(pil_img, dtype=np.uint8)
        metadata = {
            "doc_type": doc_type,
            "text_boxes": text_boxes,
            "shape": doc_array.shape,
        }
        return doc_array, metadata

    def _draw_stamp(
        self, draw: ImageDraw.ImageDraw, center: Tuple[int, int], text: str, color: Tuple[int, int, int]
    ):
        """Draws a round forensic stamp with border."""
        cx, cy = center
        r = 32
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=2)
        draw.ellipse([cx - r + 4, cy - r + 4, cx + r - 4, cy + r - 4], outline=color, width=1)
        draw.text((cx - 24, cy - 6), text[:8], fill=color, font=self._get_font(8))

    def apply_copy_move(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, ForgeryType]:
        """Duplicates a document region and pastes it to a new location with slight rotation/scaling."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        result = image.copy()

        # Select random patch (e.g. 50x30 to 120x60)
        pw = random.randint(50, 120)
        ph = random.randint(25, 60)
        src_x = random.randint(20, w - pw - 20)
        src_y = random.randint(20, h - ph - 20)

        patch = image[src_y : src_y + ph, src_x : src_x + pw].copy()

        # Choose destination avoiding full overlap
        dst_x = random.randint(20, w - pw - 20)
        dst_y = random.randint(20, h - ph - 20)
        while abs(dst_x - src_x) < pw and abs(dst_y - src_y) < ph:
            dst_x = random.randint(20, w - pw - 20)
            dst_y = random.randint(20, h - ph - 20)

        # Mild rotation (-10 to +10 deg)
        angle = random.uniform(-10.0, 10.0)
        center = (pw / 2.0, ph / 2.0)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_patch = cv2.warpAffine(patch, rot_mat, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Soft edge feathering
        feather = cv2.GaussianBlur(np.ones((ph, pw), dtype=np.float32), (5, 5), 1.0)[:, :, np.newaxis]
        target_bg = result[dst_y : dst_y + ph, dst_x : dst_x + pw].astype(np.float32)
        blended = (rotated_patch.astype(np.float32) * feather + target_bg * (1.0 - feather)).astype(np.uint8)

        result[dst_y : dst_y + ph, dst_x : dst_x + pw] = blended
        # Ground truth mask marks destination tampered region
        mask[dst_y : dst_y + ph, dst_x : dst_x + pw] = 1.0

        return result, mask, ForgeryType.COPY_MOVE

    def apply_splicing(
        self, recipient: np.ndarray, donor: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, ForgeryType]:
        """Slices a region from a donor document, matches histogram, and pastes into recipient."""
        h, w = recipient.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        result = recipient.copy()

        if donor is None:
            donor, _ = self.create_authentic_document()

        pw = random.randint(60, 140)
        ph = random.randint(30, 70)
        src_x = random.randint(10, donor.shape[1] - pw - 10)
        src_y = random.randint(10, donor.shape[0] - ph - 10)
        donor_patch = donor[src_y : src_y + ph, src_x : src_x + pw].copy()

        dst_x = random.randint(20, w - pw - 20)
        dst_y = random.randint(20, h - ph - 20)
        recipient_bg = result[dst_y : dst_y + ph, dst_x : dst_x + pw]

        # Simple color illumination matching to avoid trivial color-only detection
        matched_patch = donor_patch.astype(np.float32)
        for c in range(3):
            bg_mean = np.mean(recipient_bg[:, :, c])
            patch_mean = np.mean(donor_patch[:, :, c]) + 1e-5
            matched_patch[:, :, c] = np.clip(matched_patch[:, :, c] * (bg_mean / patch_mean), 0, 255)
        matched_patch = matched_patch.astype(np.uint8)

        result[dst_y : dst_y + ph, dst_x : dst_x + pw] = matched_patch
        mask[dst_y : dst_y + ph, dst_x : dst_x + pw] = 1.0

        return result, mask, ForgeryType.SPLICING

    def apply_text_tamper(
        self, image: np.ndarray, metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, ForgeryType]:
        """Inpaints an existing text box and re-renders altered text with mismatched font/size."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        result = image.copy()

        text_boxes = metadata.get("text_boxes", [])
        if text_boxes:
            x1, y1, x2, y2, orig_text = random.choice(text_boxes)
        else:
            # Random text box coordinate if none provided
            x1, y1 = random.randint(100, 300), random.randint(100, 400)
            x2, y2 = x1 + 120, y1 + 25
            orig_text = "TOTAL $999.00"

        bx_w, bx_h = max(x2 - x1, 20), max(y2 - y1, 15)

        # Inpaint original text box to erase it
        inpaint_mask = np.zeros((h, w), dtype=np.uint8)
        inpaint_mask[y1:y2, x1:x2] = 255
        inpainted = cv2.inpaint(result, inpaint_mask, 3, cv2.INPAINT_TELEA)

        # Re-render forged replacement text with a mismatched font size/color
        pil_img = Image.fromarray(inpainted)
        draw = ImageDraw.Draw(pil_img)

        # Generate altered number or string
        fake_values = ["$8,940.50", "$14,200.00", "2029-12-31", "VOIDED", "Prof. Robert Sterling"]
        fake_text = random.choice(fake_values)

        # Use slightly mismatched size (e.g. 15 instead of 11) to trigger OCR/visual inconsistency
        forged_font = self._get_font(size=random.choice([14, 16, 17]))
        draw.text((x1 + 2, y1 + 1), fake_text, fill=(180, 10, 10), font=forged_font)

        result = np.array(pil_img, dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1.0

        return result, mask, ForgeryType.TEXT_TAMPER

    def apply_erasure(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, ForgeryType]:
        """Erases a content area using Navier-Stokes/Telea inpainting."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)

        rw = random.randint(60, 120)
        rh = random.randint(20, 50)
        rx = random.randint(30, w - rw - 30)
        ry = random.randint(30, h - rh - 30)

        inpaint_mask = np.zeros((h, w), dtype=np.uint8)
        inpaint_mask[ry : ry + rh, rx : rx + rw] = 255

        erased = cv2.inpaint(image, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
        mask[ry : ry + rh, rx : rx + rw] = 1.0

        return erased, mask, ForgeryType.ERASURE

    def apply_jpeg_compression(self, image: np.ndarray, quality: int = 75) -> np.ndarray:
        """Simulates JPEG compression re-encoding at a specified quality factor."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encoded = cv2.imencode(".jpg", image, encode_param)
        if not success:
            return image
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return decoded

    def generate_sample(
        self,
        forgery_type: Optional[ForgeryType] = None,
        apply_double_jpeg: bool = True,
        primary_q: int = 90,
        secondary_q: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generates a complete training/eval sample with pixel ground truth and metadata."""
        auth_doc, meta = self.create_authentic_document()

        # Apply primary compression to authentic baseline
        if apply_double_jpeg:
            auth_doc = self.apply_jpeg_compression(auth_doc, quality=primary_q)

        if forgery_type is None:
            # 30% authentic, 70% forged across 4 tampering classes
            if random.random() < 0.3:
                forgery_type = ForgeryType.AUTHENTIC
            else:
                forgery_type = random.choice([
                    ForgeryType.COPY_MOVE,
                    ForgeryType.SPLICING,
                    ForgeryType.TEXT_TAMPER,
                    ForgeryType.ERASURE,
                ])

        if forgery_type == ForgeryType.AUTHENTIC:
            final_img = auth_doc
            mask = np.zeros((self.target_size[0], self.target_size[1]), dtype=np.float32)
        elif forgery_type == ForgeryType.COPY_MOVE:
            final_img, mask, _ = self.apply_copy_move(auth_doc)
        elif forgery_type == ForgeryType.SPLICING:
            donor, _ = self.create_authentic_document()
            final_img, mask, _ = self.apply_splicing(auth_doc, donor)
        elif forgery_type == ForgeryType.TEXT_TAMPER:
            final_img, mask, _ = self.apply_text_tamper(auth_doc, meta)
        elif forgery_type == ForgeryType.ERASURE:
            final_img, mask, _ = self.apply_erasure(auth_doc)

        # Apply secondary JPEG compression to inject authentic DCT grid artifact
        if apply_double_jpeg and forgery_type != ForgeryType.AUTHENTIC:
            if secondary_q is None:
                secondary_q = random.choice([55, 65, 75, 85])
            final_img = self.apply_jpeg_compression(final_img, quality=secondary_q)

        return {
            "image": final_img,
            "mask": mask,
            "forgery_type": int(forgery_type),
            "forgery_type_name": FORGERY_TYPE_NAMES[forgery_type],
            "is_forged": float(forgery_type != ForgeryType.AUTHENTIC),
            "metadata": meta,
        }
