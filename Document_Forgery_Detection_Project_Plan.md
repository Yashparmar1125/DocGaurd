# Document Image Forgery Detection — Complete Technical Project Document

*Prepared as a foundation for implementation, research paper, presentation, and viva*

---

## 1. Project Title (choose one)

1. **DocGuard: A Hybrid CNN-Transformer Framework for Multi-Task Document Forgery Detection, Localization, and Explanation**
2. **Explainable Multi-Task Learning for Copy-Move, Splicing, and Text-Tampering Detection in Scanned Documents**
3. **DCT-Aware Deep Forensics: Robust Localization of Tampered Regions in Document Images under Compression**
4. **VeriDoc: An OCR-Guided Deep Learning Pipeline for Forgery Detection and Region-Level Explanation in Identity and Financial Documents**
5. **Cross-Domain Document Forgery Detection: Combining Visual Forensics, OCR Consistency, and Explainable Localization**

**Recommended primary title:** *"DocGuard: A Hybrid CNN-Transformer Framework for Multi-Task Document Forgery Detection, Localization, and Explanation."* It signals the three things examiners/reviewers look for immediately: the method family (hybrid CNN-Transformer), the tasks (detection + localization), and the differentiator (explanation).

---

## 2. Problem Statement

### 2.1 What document forgery is
Document forgery is the deliberate alteration of a document's visual or textual content — digitally or physically — to misrepresent facts for financial, legal, or identity fraud. It ranges from crude edits (retyped amounts, pasted photos) to sophisticated digital manipulation using image editors, generative inpainting, or diffusion-based text synthesis that leaves few visible artifacts.

### 2.2 Why it matters
Forged invoices, receipts, ID cards, certificates, and contracts drive insurance fraud, loan fraud, visa/immigration fraud, and academic-credential fraud. Recent surveys note that identity-document attacks are shifting from opportunistic physical tricks toward AI-assisted and fully generative synthesis, which existing rule-based and purely physical-artifact detectors struggle to catch — meaning detection systems must increasingly combine artifact-level visual forensics with semantic/textual consistency checks.

### 2.3 Types of forgery targeted
Copy-move, splicing, text tampering (insert/delete/replace characters or fields), signature/stamp/seal tampering, and erasure/inpainting on scanned or photographed documents. (Full analysis in Section 5.)

### 2.4 Limitations of existing approaches
- **Manual verification:** slow, subjective, doesn't scale, and is blind to pixel-level manipulation that isn't visually obvious.
- **Rule-based / metadata checks (EXIF, font databases):** trivially defeated by re-saving, screenshotting, or printing-and-rescanning a document, which strips metadata entirely.
- **Classical forensic techniques alone (ELA, noise/JPEG-artifact analysis):** effective on natural photographs but weaker on documents, which are flat, high-contrast, text-dominant images where compression and re-scanning already introduce noise that resembles tampering traces — producing many false positives.
- **Pure classification CNNs:** output only "forged/authentic," with no explanation of *where* or *why*, which is unacceptable for a forensic or compliance use case.

### 2.5 Why AI/ML is suitable
Deep networks can learn subtle, non-obvious statistical inconsistencies (noise-level discontinuities, double-JPEG-compression grids, font/edge irregularities) that are far below human visual perception thresholds, and can be trained end-to-end to jointly classify and localize forgeries — something handcrafted forensic rules cannot easily combine.

### 2.6 Final problem statement
> *"Given a scanned or photographed document image, design and implement a deep-learning-based system that (a) classifies the document as authentic or forged, (b) localizes the tampered region(s) at pixel/region granularity, (c) identifies the likely forgery type, and (d) produces a human-interpretable explanation (heatmap + textual reason) for its decision — while remaining robust to common real-world distortions such as JPEG re-compression, scanning noise, and resizing."*

---

## 3. Scope of the Project

### 3.1 Input (MVP)
- Scanned document images (JPEG/PNG/TIFF)
- Smartphone photographs of documents
- Single-page PDFs (converted to images internally)
- Target document classes: **receipts, invoices, forms, certificates, ID-style cards** — chosen because these are the classes with public, labeled tampering datasets (Section 6).

**Advanced/future input:** multi-page PDFs with cross-page consistency checks, video-based document capture (liveness), hyperspectral scans (ink analysis) — out of MVP scope; hyperspectral/ink-analysis requires specialized hardware most student projects don't have access to, so it is explicitly excluded rather than falsely claimed.

### 3.2 Output

| Output | MVP or Advanced |
|---|---|
| Authentic vs. Forged classification | **MVP** |
| Forgery confidence/probability score | **MVP** |
| Pixel-level forgery localization mask | **MVP** |
| Forgery-type classification (copy-move / splicing / text-tamper / erasure) | **MVP** |
| Grad-CAM heatmap overlay | **MVP** |
| OCR-based textual inconsistency explanation | **MVP** |
| Downloadable PDF forensic report | **MVP** |
| Signature/stamp-specific forgery detection | Advanced |
| Cross-document / cross-template consistency checking | Advanced |
| Source-scanner/printer identification | Advanced (future work) |
| Adversarial-robustness certification | Advanced (future work) |
| Metadata-manipulation detection | **Not claimed** — metadata is destroyed by scanning/screenshotting in most realistic pipelines, so this is explicitly out of scope (see Section 5.I and Section 28) |


---

## 4. End-to-End Workflow

```
Input Document (image / photo / PDF)
        ↓
Document Validation (format, size, corruption check)
        ↓
PDF → Image Conversion (if needed)
        ↓
Preprocessing (denoise, deskew, contrast-normalize)
        ↓
Document Alignment / Rectification (perspective correction, crop to boundary)
        ↓
Dual Feature Extraction:
   ├─ RGB stream (CNN/Transformer backbone)
   └─ Forensic stream (DCT / noise-residual features)
        ↓
Deep Learning Detection (multi-task backbone)
        ↓
Forgery Localization Head → pixel-level mask
        ↓
Forgery-Type Classification Head → {copy-move, splicing, text-tamper, erasure, authentic}
        ↓
OCR Cross-Check (text-region consistency, font/spacing anomaly)
        ↓
Explainability (Grad-CAM heatmap + rule-based textual reasoning)
        ↓
Confidence Calibration
        ↓
Final Forensic Report (JSON + downloadable PDF)
```

This departs from the naive "classify → localize as two separate models" pipeline suggested in the prompt template. Recent document-forgery-localization work (e.g., ADCD-Net, ICCV 2025; DocForgeNet) shows that classification and localization should share a backbone and be trained jointly (multi-task), because a model forced to *localize* the tampered pixels is regularized into learning genuinely forgery-relevant features rather than dataset-specific shortcuts — this also directly gives you the classification decision as a by-product (any non-empty predicted mask ⇒ forged), so a separate detection stage is redundant.

### Stage-by-stage detail

**1. Document Validation** — Rejects corrupt files, wrong formats, or resolutions too low for forensic analysis (<300px shortest side). *Library:* Pillow/OpenCV file-integrity checks. No ML needed.

**2. PDF/Image Conversion** — `pdf2image` (poppler-based) rasterizes PDF pages at 300 DPI. Alternative: `PyMuPDF` (faster, no external binary dependency) — **recommended** because it avoids a system-level poppler dependency, simplifying deployment.

**3. Preprocessing** — See Section 8 for the exact selected pipeline (deskew + CLAHE + mild denoise). Purpose: remove *nuisance* variation (skew, lighting) while explicitly preserving the noise/compression signal the forensic stream depends on — this is why global denoising is deliberately kept mild (see Section 8 caveat).

**4. Alignment/Rectification** — Document-boundary detection (largest 4-point contour via Canny + `cv2.findContours`) followed by a perspective transform (`cv2.getPerspectiveTransform`) crops and de-warps photographed documents to a flat, top-down view. Necessary because photographs (vs. flatbed scans) are rarely perfectly frontal, and geometric distortion hurts both OCR accuracy and localization-mask alignment.

**5. Forgery Feature Extraction** — Two parallel streams feed the backbone:
   - **RGB/spatial stream:** learns texture, edge, and semantic inconsistency.
   - **Forensic stream:** block-wise 8×8 DCT coefficients (as in JPEG) computed on the luminance channel, which expose double-compression grid misalignment — the single strongest, most literature-validated signal for detecting spliced/copy-moved regions in re-compressed document images (used by ADCD-Net, CAT-Net, DocForgeNet).

**6. Deep Learning Detection (Section 10–11)** — Hybrid CNN+DCT backbone (Section 11) produces shared features consumed by two heads.

**7. Forgery Localization** — A segmentation decoder (U-Net-style) upsamples backbone features to a full-resolution binary/soft mask marking tampered pixels.

**8. Forgery-Type Classification** — A lightweight classification head over the same pooled features predicts {authentic, copy-move, splicing, text-tamper, erasure}.

**9. OCR Cross-Check** — Tesseract/PaddleOCR extracts text boxes; font-height, baseline-alignment, and inter-character-spacing statistics are compared against a document-type template or against the rest of the document, flagging boxes that are statistical outliers (used as a rule-based corroborating signal, not a separate deep model, to keep the system tractable — see Section 13).

**10. Explainability** — Grad-CAM on the last convolutional block of the CNN stream + the predicted mask are combined into a single heatmap overlay; a short template-based textual explanation is generated from the flagged region's location and the predicted forgery type.

**11. Confidence Calibration** — Temperature scaling (Guo et al., 2017) applied post-hoc to the classification logits so the reported probability is statistically meaningful rather than an overconfident raw softmax output.

**12. Final Report** — All of the above assembled into a JSON payload (for the web app) and a printable PDF (for the "downloadable forensic report" demo requirement).


---

## 5. Types of Forgery We Should Detect

| Type | Description | Realistically detectable with our pipeline? |
|---|---|---|
| **A. Copy-Move** | Region duplicated within the same document | **Yes.** Core capability of DCT/noise-residual + localization head; well-supported by CASIA v1/v2 and DocTamper. |
| **B. Splicing** | Content pasted in from a different image/document | **Yes.** Same mechanism as copy-move; DCT double-compression signal is even stronger here since the pasted region often has different compression history. |
| **C. Text Tampering** | Characters/words inserted, deleted, replaced | **Yes, this is our primary target.** DocTamper, T-SROIE and RTM datasets exist specifically for this; font/OCR consistency checks add a second detection axis. |
| **D. Image Tampering** (photo/logo swapped) | Embedded photo or logo altered | **Partially.** Detectable as a special case of splicing/copy-move if the region is large enough; small logo edits are harder given typical image resolution. |
| **E. Signature Forgery** | Signature imitated or copy-pasted | **Only the copy-pasted case, reliably.** Freehand-imitated ("skilled forgery") signature detection is a separate biometric problem (needs dedicated signature-verification datasets like CEDAR/ICDAR SigComp) — treat as an **advanced/optional module** (Section 14), not core MVP. |
| **F. Stamp/Seal Forgery** | Stamp copied, altered, or fake | **Partially** — copy-pasted stamps detectable via splicing pipeline; generating a stamp from scratch that matches ink/pressure texture is out of scope. |
| **G. Erasure / Inpainting** | Original content removed and background reconstructed | **Yes, with caveats.** Detectable when the inpainting leaves texture/frequency inconsistency; modern diffusion-based inpainting is harder and is flagged as a known limitation (Section 29). |
| **H. Document Replacement** | An entire page swapped | **Out of MVP scope** — this is a page/document-level consistency problem (needs multi-page context), not a single-image pixel-forensics problem. Note as future work. |
| **I. Metadata Manipulation** | EXIF/PDF metadata edited | **Not claimed.** Any document that has been scanned, screenshotted, or re-saved loses or fabricates metadata regardless of tampering status, making this signal unreliable for our target input types (photos/scans). We explicitly do **not** build or claim a metadata module. |

**Bottom line for the report:** claim A, B, C, and G as core detected forgery types; claim D, E (copy-paste only), F (copy-paste only) as partially covered; explicitly state H and I are out of scope and explain why (don't silently omit them — examiners will ask).


---

## 6. Dataset Analysis

| Dataset | Source / Access | Document type | Size | Forgery types | Ground truth | Localization masks | Fit for our project |
|---|---|---|---|---|---|---|---|
| **DocTamper** | Public (Qu et al., ICDAR-track paper 2023; BaiduDrive/Kaggle mirrors) | Contracts, receipts, invoices, books (photographed) | 170,000 images, cropped to 512×512; ~120k train fake, 30k test fake, plus FCD (2k)/SCD (18k) cross-domain splits | Copy-move, splicing, print-based edits, text tamper | Yes (synthetic, but pixel-accurate) | **Yes, pixel-level** | **Primary training dataset** — largest, most-used benchmark specifically for document tampering localization; has standardized cross-domain test protocol ("Doc Protocol") used by SOTA papers (ADCD-Net, DocForgeNet), so our results are directly comparable to published numbers. |
| **T-SROIE** | Public (2022, extended from ICDAR-SROIE receipts) | Scanned receipts | ~12.7k real / 2.7k fake (train), ~8.5k real/1.6k fake (test) after standard 512×512 cropping | AI-generated (SRNet) text tampering | Yes | Yes, pixel-level | **Cross-domain evaluation set** — tests generalization from DocTamper training to a different tamper-generation method (SRNet-based), which is exactly the "robustness/generalization" evaluation the project needs (Section 22–23). |
| **RTM (RealTextManipulation, 2025)** | Public benchmark | Scanned forms, diverse document types | ~22.3k real / ~3.4k fake (test split) | Copy-move, splicing, print, erasure — mix of synthetic **and manually manipulated** | Yes | Yes | **Second cross-domain test set** — because it includes manually (not purely rule-generated) manipulated documents, it best approximates real-world forgeries and is the strongest test of whether the model overfits to DocTamper's synthetic generation pipeline. |
| **CASIA v1.0 / v2.0** | Public (Dong et al., 2013) | General natural images (not document-specific) | v1: 800 authentic/921 spliced; v2: ~7,491 authentic/5,123 tampered (copy-move+splicing) | Copy-move, splicing | Yes (image-level labels; region masks available via third-party annotations, e.g. Nam Thanh et al.) | Partial (community-provided masks) | **Supplementary/pretraining only** — not document-specific (mostly natural scene photos), but useful for pretraining the forensic (DCT/noise) stream before fine-tuning on DocTamper, since it is small enough to train quickly and is the most widely benchmarked splicing/copy-move dataset in the forensics literature. |
| **OSTF (2025)** | Public | Natural scene text | 62,278 real / 6,998 fake | AI-generated text-based + image-based forgery (8 different AIGC editors) | Yes | Yes | Optional — useful only if the project extends to "AI-generated text edit" robustness; not essential for MVP. |
| **ICDAR Receipt Forgery Dataset (2023)** | Public (Martínez Tornés et al., ICDAR 2023) | Receipts | Moderate size (paper-specific) | Realistic receipt-specific tampering | Yes | Varies | Optional supplementary set specific to receipts, useful for the "invoice/receipt" use case named in the input scope. |

### 6.1 Should we also create our own forged samples?
**Yes — recommended, but as an augmentation strategy, not the primary training source.** Public datasets like DocTamper already give scale and pixel masks; self-generated data should be used to (a) cover document templates relevant to a specific demo use case (e.g., a sample invoice template), and (b) create a small, manually-curated real-world test set for the qualitative demo, since every public dataset above is either purely synthetic or has a specific generation pipeline whose artifacts a model can learn to shortcut on. Section 7 gives the exact generation recipe.


---

## 7. Dataset Creation / Forgery Generation (Self-Made Data)

Goal for every generated sample: **(original, forged, binary forgery mask, forgery-type label)**.

### Text manipulation (primary focus, mirrors DocTamper's own generation logic)
- **Font/size replacement:** re-render a detected text-box region with a different font or size using PIL `ImageDraw`, matched to the background color sampled from the surrounding pixels.
- **Text insertion/deletion/replacement:** use OCR to detect a text box → inpaint the original box (see erasure below) → render new text in its place. The mask = the union of the erased + newly-rendered bounding boxes.
- Automatically generates a **pixel-accurate ground-truth mask** because the edited region is known exactly (the bounding box used for rendering).

### Image manipulation
- **Copy-paste (copy-move):** randomly select a rectangular/segmented region, paste it elsewhere in the *same* image at a random offset, optionally with rotation/scaling (5–15°, 0.9–1.1× scale) and Gaussian-blur edge blending to mimic realistic forgery. Mask = source **and** destination regions (or destination only, depending on the evaluation convention chosen — DocTamper uses destination-only; we should follow the same convention for comparability).
- **Splicing:** paste a region from a *different* image/document. Apply matching color/illumination adjustment (histogram matching) so the splice isn't trivially detectable by color alone (which would make the task artificially easy and not generalize).
- **Compression re-encoding:** save the forged image at a different JPEG quality factor than the original (simulates the double-compression artifact that DCT-based forensics detect) — this step is essential, not optional, because it's the exact signal the forensic stream is designed to exploit.

### Signature / stamp manipulation
- Copy-paste an existing signature/stamp crop to a new position, or scale/rotate it — same mechanism as copy-move, applied to a cropped signature/stamp region library.

### Erasure / inpainting
- Select a region (e.g., a text box or object), remove it using OpenCV's `cv2.inpaint` (Telea or Navier-Stokes algorithm) for simple background-fill erasure, or a pretrained inpainting model (e.g., LaMa) for more realistic results if compute allows. Mask = the erased region.

### Producing ground-truth masks automatically
Every manipulation function above should be written to **return the modified image and its binary mask together** (e.g., a Python decorator/wrapper pattern), never generated after the fact by diffing images (diffing is unreliable when compression is re-applied, since JPEG re-encoding changes untouched pixels too).

### Recommended augmentation intensity
Apply the manipulations at multiple severities per source image (small/medium/large region, mild/aggressive blending) — this creates a harder, more realistic difficulty curve than dataset templates that always use one severity, directly supporting the robustness evaluation in Section 22.


---

## 8. Image Preprocessing — Exact Pipeline We Will Use

**Selected pipeline (in order):**

1. **Format/orientation normalization** — EXIF-orientation correction, convert to RGB.
2. **Document boundary detection + perspective correction** (only for photographed, not flatbed-scanned, inputs) — Canny edge detection → `cv2.findContours` → largest 4-point polygon → `cv2.getPerspectiveTransform` + `cv2.warpPerspective`.
3. **Deskew** — Hough Transform on text-line edges to estimate rotation angle, then `cv2.warpAffine` to correct small residual rotation (< document-boundary correction, this handles skew *within* an already-cropped scan).
4. **Mild contrast normalization** — CLAHE (Contrast-Limited Adaptive Histogram Equalization) applied only to the luminance channel, with a conservative clip limit (2.0). *Why mild:* aggressive contrast/denoising would destroy the very noise and compression artifacts the forensic (DCT) stream depends on — this is the single most important preprocessing decision and must be explicitly justified in the report/viva.
5. **Resize** — resize the long side to a fixed size (e.g., 1024px) for the alignment/OCR stage, but feed the model fixed-size **crops** (512×512, matching the DocTamper protocol) rather than a globally resized/warped whole image, since resizing the whole image would blur the DCT block-grid signal.
6. **Normalization** — per-channel mean/std normalization (ImageNet statistics if using a pretrained backbone) applied only to the RGB stream, *not* to the DCT stream (which uses raw coefficient statistics).

**Explicitly NOT used globally:** Gaussian/median denoising, Otsu thresholding, background removal. Reasoning: these are standard OCR-preprocessing steps that actively erase the forensic evidence needed for tamper detection (they're appropriate for text-recognition-only pipelines, not forgery-forensics pipelines). We only apply light denoising locally, inside the OCR sub-branch, on a copy of the image used solely for text extraction — never on the copy fed to the forensic stream.


---

## 9. Traditional Image Forensics Features

| Technique | Principle | Detects | Advantages | Limitations | Include? |
|---|---|---|---|---|---|
| **Error Level Analysis (ELA)** | Re-compress at known JPEG quality, diff against original; tampered regions compress differently | Splicing, copy-paste in JPEG images | Cheap, interpretable, no training | Very noisy on already-low-quality scans; many false positives on text-heavy documents | **Include as a visualization/explanation aid only**, not a decision feature |
| **Noise inconsistency** | Local noise-level estimation (e.g., wavelet-based); pasted regions have different sensor/compression noise | Splicing, copy-move | Works even without strong edges | Documents are often noise-poor (flat scans), weakening the signal | Optional supporting feature |
| **JPEG double-compression / DCT artifacts** | Re-saving a JPEG twice creates detectable periodic block-grid inconsistencies | Splicing, copy-move, replacement | **Strongest, most literature-validated signal for document forgery** (ADCD-Net, CAT-Net, DocForgeNet all rely on it) | Fails if final output is never re-JPEG-compressed (e.g., stays lossless PNG) — must be paired with RGB stream | **Include — this is a core input to our forensic stream** |
| **Copy-move detection (block-matching/keypoint)** | Match duplicated regions via block correlation or keypoints | Copy-move only | Very precise when it works | Fails under heavy transformation, needs a separate algorithm from splicing detection | Superseded by the learned localization head; keep as a **classical fallback/sanity-check module** |
| **Edge inconsistency** | Discontinuous or double edges at splice boundaries | Splicing | Simple, fast | Weak on clean paste-and-blend edits | Implicitly learned by the CNN; not a separate module |
| **Texture inconsistency** (e.g., LBP) | Local Binary Patterns capture micro-texture; pasted regions differ statistically | Splicing | Lightweight, interpretable | Weak alone, needs an SVM/classifier on top | Used only as an ablation baseline (Section 24), not in the final model |
| **Color inconsistency** | Illumination/white-balance mismatch across regions | Splicing | Intuitive | Documents are mostly grayscale/near-white, so color signal is very weak | Not included |
| **Frequency-domain analysis (general)** | FFT/DCT global spectrum analysis | Resampling, GAN-generated artifacts | Detects some generative-model traces | Broad, low precision alone | Folded into the DCT stream, not a standalone module |
| **DCT features** | Block-wise cosine-transform coefficient statistics | Compression traces | Core signal (see above) | — | **Include (core)** |
| **DWT / Wavelet features** | Multi-resolution frequency decomposition | Noise/texture inconsistency | Complements DCT at other scales | Adds complexity without clear document-specific gain over DCT alone in recent literature | Not included (documented as considered-and-rejected) |
| **Local Binary Patterns (LBP)** | Texture micro-pattern histogram | Splicing/texture anomaly | Fast, classical | Weak on document images (mostly text/whitespace) | Ablation baseline only |
| **SIFT / ORB keypoint matching** | Keypoint correspondence to find duplicated regions | Copy-move | Robust to rotation/scale | Documents have few distinctive keypoints (lots of repeated text/lines → many false matches) | Not included as primary; noted as a known-poor-fit for text documents |

**Final recommendation:** the only traditional forensic feature carried into the final deep model is **block-wise DCT** (as a second input stream alongside RGB). ELA is retained purely as a human-readable visualization aid in the report/demo (Section 30), not as a model input, since it is too noisy to threshold reliably. All other classical features are used only for the ablation study (Section 24) to empirically justify this choice rather than asserting it.


---

## 10. Deep Learning Model Selection

| Architecture | Purpose | Classification | Localization | Advantages | Disadvantages | Compute | Fit for document forgery |
|---|---|---|---|---|---|---|---|
| **Plain CNN (from scratch)** | Baseline | Yes | No (without a decoder) | Simple, fast to train | Weak on subtle forensic cues, no localization | Low | Baseline only (Section 23) |
| **ResNet-50** | Backbone | Yes (with head) | With added decoder | Strong, well-understood, pretrained weights widely available, residual connections help gradient flow at depth | Not designed for localization out of the box | Medium | **Good backbone choice** |
| **EfficientNet** | Backbone | Yes | With decoder | Best accuracy/compute trade-off among CNN backbones | Slightly harder to fine-tune/tune scaling coefficients well | Low–Medium | Good alternative if compute-constrained |
| **DenseNet** | Backbone | Yes | With decoder | Feature reuse helps with limited data | Higher memory usage due to dense connections | Medium–High | Not selected — memory cost not justified by document-specific gains |
| **MobileNet** | Backbone | Yes | With decoder | Very lightweight, good for deployment/edge | Lower ceiling on accuracy | Very Low | Good for a deployable/edge demo variant, not the primary research model |
| **Vision Transformer (ViT)** | Backbone | Yes | With decoder | Strong global context modeling | Needs large data or strong pretraining to avoid overfitting on modest datasets | High | Used as the *second* stream in our hybrid (global context complements CNN's local sensitivity) |
| **Swin Transformer** | Backbone | Yes | Yes (hierarchical, segmentation-friendly) | Hierarchical windows suit dense prediction tasks; used successfully in recent document tamper-detection work | Heavier to train than plain CNN | Medium–High | Strong alternative transformer backbone; noted in literature (Swin-T segmentation for tamper detection) |
| **U-Net** | Localization decoder | No (needs added head) | **Yes — purpose-built** | Simple, proven, works well with modest data via skip connections | No native classification output | Low–Medium | **Selected as localization decoder** |
| **U-Net++** | Localization decoder | No | Yes (improved) | Denser skip connections improve fine-boundary localization | More parameters/compute than U-Net | Medium | Considered; marginal gain not worth the added complexity for MVP |
| **DeepLabV3+** | Localization decoder | No | Yes (atrous conv, strong for irregular shapes) | Good multi-scale context via atrous spatial pyramid pooling | Heavier, tuning atrous rates adds complexity | Medium–High | Strong alternative; keep as an ablation comparison |
| **Mask R-CNN** | Instance segmentation | Yes (per instance) | Yes | Good for discrete, separable tampered objects | Overkill / awkward for pixel-diffuse tampering like text edits; slower | High | Not selected — better suited to bounded objects (e.g., stamps) than diffuse text-region tampering |
| **YOLO-seg** | Detection+segmentation | Yes | Yes (box + mask) | Fast, good for real-time demo | Less precise boundary localization than U-Net-family decoders | Medium | Considered for the optional signature/stamp detector (Section 14), not the core forgery mask |

### Final selection and justification
**Backbone:** a **dual-stream hybrid** — a CNN (ResNet-50, ImageNet-pretrained) processing the RGB image, fused with a lightweight transformer/attention block processing block-wise DCT features — because this combination is what the current state-of-the-art document-tampering-localization literature (DocForgeNet's dual cross-stream CNN+transformer fusion; ADCD-Net's adaptive DCT + content disentanglement) converges on, and it directly matches our two-stream feature-extraction design in Section 4.

**Decoder:** **U-Net-style decoder** for the localization head — chosen over U-Net++/DeepLabV3+ for MVP because it trains reliably on a moderate GPU with a moderate dataset size, and its accuracy gap versus the heavier decoders is small relative to the added training time, which matters for a student project timeline; U-Net++ or DeepLabV3+ are reasonable **ablation/stretch-goal** upgrades once the base pipeline works (Section 24).

**Not the newest model, and why that's fine:** we are *not* selecting the largest available transformer or a diffusion-based forensics model, because our dataset size (hundreds of thousands of DocTamper crops, but a single student GPU) and timeline favor a well-validated, moderately sized hybrid over a state-of-the-art-but-fragile architecture that needs heavy compute and tuning to reproduce.


---

## 11. Final Model Architecture

```
Document Image (512×512 crop, RGB)
        ↓
   ┌────────────────────┬─────────────────────────┐
   │  RGB Stream         │  Forensic Stream          │
   │  ResNet-50           │  8×8 block DCT on Y-channel│
   │  (ImageNet-pretrained│  → shallow CNN/attention   │
   │  conv backbone)       │  encoder on DCT coeffs      │
   └────────┬────────────┴──────────┬────────────────┘
            │                        │
            └──────── Cross-Attention Fusion ─────────┘
                            ↓
                  Shared Fused Feature Map
                     ↓                 ↓
     ┌───────────────────────┐   ┌───────────────────────────┐
     │ Classification Head    │   │ Localization Head (U-Net    │
     │ (GAP → FC → softmax)   │   │ decoder w/ skip connections)│
     │ → Authentic/Forged      │   │ → per-pixel forgery mask     │
     │ → Forgery-type (5-way)  │   │ → confidence per pixel         │
     └───────────────────────┘   └───────────────────────────┘
                            ↓
                 Confidence Calibration (temperature scaling)
                            ↓
                 Explainability (Grad-CAM + mask overlay)
```

- **Backbone:** ResNet-50 (RGB) + lightweight CNN (DCT), fused via cross-attention (a small multi-head attention block, not the full ResNet self-attention, to keep parameter count manageable).
- **Classification head:** global average pooling → fully connected → softmax, two outputs (binary authentic/forged, and 5-way forgery-type, trained jointly).
- **Localization head:** U-Net-style decoder with skip connections back to the RGB stream's intermediate feature maps, producing a full-resolution sigmoid mask.
- **Loss functions:** see Section 20.
- **Activation functions:** ReLU (or GELU in the transformer-fusion block) throughout; sigmoid on the mask output; softmax on classification logits.
- **Optimizer:** AdamW (decoupled weight decay improves generalization over plain Adam for this scale of model).
- **Learning-rate strategy:** warmup for the first ~5% of steps, then cosine decay; backbone (pretrained) uses a 5–10× lower LR than the randomly-initialized decoder/fusion layers (differential learning rates), since the pretrained weights need gentler fine-tuning.
- **Regularization:** weight decay (AdamW default ~0.01), dropout (0.2–0.3) in the classification head, and label smoothing (0.1) on the classification loss to reduce overconfidence.
- **Data augmentation:** see Section 21 — must be forensic-aware (i.e., must not accidentally strip the DCT signal; see Section 8 caveat).


---

## 12. Multi-Modal / Hybrid Approach

| Modality | Signal captured |
|---|---|
| Image (RGB + DCT) | Pixel/frequency-level tampering traces |
| OCR/text | Font, spacing, alignment, semantic-field anomalies (e.g., a date that doesn't parse, a total that doesn't sum) |
| Metadata | **Excluded** (Section 5.I) — unreliable for scans/photos |
| Visual forensic features | Folded into the DCT stream (Section 9) |

**Should we combine image + OCR/text? Yes — but as late fusion, not early feature concatenation.** The image model and the OCR-consistency checker operate at different granularities (pixel mask vs. text-box statistics) and are trained/validated independently; combining them at the *decision* level (each flags suspicious regions, and the explanation module reports both) is simpler, more debuggable, and avoids forcing an OCR error to corrupt the vision model's gradients. This is a genuine, justified architectural choice, not a novelty claim — see Section 17 for why we don't over-claim this as new.

```
Document Image → CNN+DCT hybrid → forgery mask + type + confidence
Document Text (OCR) → font/spacing/field statistics → suspicious-text-box list
                    ↓ (late fusion: union / cross-reference by region)
        Combined Forensic Report (Section 16)
```

---

## 13. OCR Component

OCR supports forgery detection by flagging:
- Font inconsistency (mixed fonts within a field that should be uniform)
- Text-baseline/alignment inconsistency (a retyped line sits slightly off-baseline)
- Spacing anomalies (character kerning that doesn't match the rest of the document)
- Unexpected text regions (text present where the template has none)
- Modified dates/amounts/names (values that fail a field-specific sanity/format check, e.g. a date field that isn't a valid date)

| OCR engine | Strengths | Weaknesses | Recommended? |
|---|---|---|---|
| **Tesseract** | Free, mature, works offline, good on clean printed text, easy Python bindings (`pytesseract`) | Weaker on noisy/handwritten/skewed text | **Recommended for MVP** — sufficient for printed receipts/forms/certificates and simplest to integrate |
| **EasyOCR** | Good multilingual support, deep-learning-based, decent on natural scenes | Slower, heavier GPU dependency | Optional upgrade if multilingual receipts are needed |
| **PaddleOCR** | Strong accuracy on structured documents, provides layout/box confidence useful for our font/spacing checks | More complex setup, heavier dependency footprint | **Recommended as the production-quality option** if GPU time budget allows — its structured text-box output is the most convenient for the spacing/alignment statistics we need |

**Final choice:** start with **Tesseract** for MVP speed of integration; swap in **PaddleOCR** once the pipeline works end-to-end, since its layout analysis materially improves the font/spacing anomaly signal.

---

## 14. Signature and Stamp Detection (Optional/Advanced Module)

Dedicated detection of signatures/stamps/seals/logos is **not required for MVP** but is a reasonable stretch goal. Recommended approach if pursued: a small **YOLO-based object detector** (YOLOv8n or similar lightweight variant) fine-tuned to localize signature/stamp/logo bounding boxes first, then apply the copy-move/splicing pipeline (Section 4) specifically within those cropped regions. YOLO is preferred here (over Mask R-CNN) because signatures/stamps are compact, roughly rectangular regions where box detection is sufficient and speed matters more than fine mask boundaries. Integration point: this module's output boxes are simply added as extra "regions of interest" fed into the localization/OCR cross-check stage — it doesn't require retraining the core hybrid backbone.

---

## 15. Forgery Localization

| Decoder | Compared |
|---|---|
| U-Net | **Selected** (Section 10 justification) |
| U-Net++ | Ablation/stretch goal |
| DeepLabV3+ | Ablation/stretch goal |
| Mask R-CNN | Not selected (Section 10) |
| YOLO-seg | Used only for the optional signature/stamp module (Section 14) |
| Transformer segmentation (e.g., SegFormer-style) | Considered but adds training complexity without a clear document-specific benefit at our data scale; noted as future work |

**Mask generation:** the localization head outputs a per-pixel sigmoid score map at the input resolution (512×512); thresholding at 0.5 gives a binary mask, but the raw probability map is what's shown in the heatmap (Section 16) since it's more informative than a hard binary cut.

**Mask evaluation:** IoU and Dice coefficient against the ground-truth mask (Section 22), computed per-image and averaged, following the DocTamper/ADCD-Net evaluation protocol so results are directly comparable to published baselines.


---

## 16. Explainable AI

| Method | Fit for this project |
|---|---|
| **Grad-CAM** | **Selected** — works on any standard CNN without architecture changes, cheap to compute at inference time, and the coarse heatmap is intuitive for a non-technical report reader |
| Grad-CAM++ | Marginal improvement in localization sharpness; worth trying as an easy upgrade since it's a drop-in replacement, but not required |
| Integrated Gradients | More rigorous attribution, but noisier/harder-to-interpret visualizations for non-technical stakeholders; higher compute cost | 
| SHAP | Expensive for image models at this resolution; better suited to tabular/OCR-field-level explanation if pursued as a stretch goal |
| Attention maps | Already naturally available from the DCT cross-attention fusion block (Section 11) — can be visualized as a secondary "why the forensic stream flagged this" map, complementing Grad-CAM |

**Note:** since we already produce a genuine, trained **localization mask**, Grad-CAM here is used as a *secondary, corroborating* explanation for the classification decision — the primary spatial explanation is the segmentation mask itself, which is more precise than any post-hoc attribution method. This distinction matters for the viva: the mask is a first-class model output, Grad-CAM is a diagnostic overlay on top of the classifier.

**Final output format:**
```
Prediction: FORGED
Confidence: 94.2%
Forgery Type: Text Tampering
Suspicious Region: Date field (top-right, bounding box [x1,y1,x2,y2])
Localization Mask: [overlay image]
Grad-CAM Heatmap: [overlay image]
OCR Cross-Check: Font mismatch detected in flagged region (Arial vs. surrounding Times New Roman)
Reason: Pixel-level DCT double-compression signature and OCR font inconsistency both localize to the same region, consistent with text replacement.
```

---

## 17. Novelty / What Can We Do That Is New?

**Important framing for the report:** none of the individual pieces below (CNN+DCT fusion, OCR cross-check, Grad-CAM, multi-task learning) are individually novel — each has precedent in the cited literature. A realistic, honestly-scoped student novelty claim is about the **specific combination and evaluation rigor**, not invention of a new algorithm.

| # | Idea | What existing systems do | What we add | Difficulty | Benefit | Realistic? |
|---|---|---|---|---|---|---|
| 1 | Hybrid CNN+Transformer/DCT architecture | DocForgeNet, ADCD-Net already do dual-stream CNN+DCT/transformer fusion for document forgery localization | Reproduce this proven design at smaller scale, validated on our own cross-domain test protocol | Medium | High (strong base performance) | Yes — but **not claimed as novel**, cite these papers explicitly |
| 2 | Forensic features + deep learning | Standard in the field (CAT-Net, TruFor, ADCD-Net) | Same | Low | High | Yes — not novel, standard practice |
| 3 | OCR + visual forgery detection fusion | Rare in *public document-tampering-localization* papers (most focus purely on pixel forensics; OCR-based text-consistency checking is more common in identity-document survey literature but not tightly integrated with pixel-level localization models) | A working, evaluated late-fusion pipeline combining pixel-level localization with OCR font/spacing anomaly detection, with a joint report format | Medium | Medium-High — directly differentiates a student project from a pure-CV reproduction | **Yes — realistic, genuine differentiator for this project's scope** |
| 4 | Multi-task learning (classification + localization jointly) | Standard practice in recent forgery localization papers | Reproduce, with our own ablation proving joint training beats separate models on our specific dataset mix | Low-Medium | Medium (mainly a training-efficiency/regularization argument) | Yes — good ablation-study material |
| 5 | Explainable forgery detection (mask + Grad-CAM + textual reasoning) | Segmentation masks are common; combined mask+Grad-AM+auto-generated textual explanation reports are less commonly packaged together in student-scale document-forensics projects | An end-to-end explanation pipeline producing a human-readable forensic report, not just a mask | Low-Medium | High for demo/practical value | **Yes — realistic, strong demo differentiator** |
| 6 | Synthetic forgery generation pipeline (Section 7) | Datasets like DocTamper already do this at scale | Our own template-specific generator for the demo document types, producing pixel-accurate masks by construction | Low | Medium (mainly for demo realism, not research novelty) | Yes, low risk |
| 7 | Document-type-aware detection (separate calibration/thresholds per document class) | Rare — most benchmarks train one model across mixed document types without per-type calibration | Per-document-type confidence calibration (Section 11's temperature scaling, fit separately per class) | Medium | Medium — improves real-world usability | Feasible as a focused experiment |
| 8 | Confidence calibration | Not always addressed in forensics papers (raw softmax often reported as "confidence") | Explicit post-hoc calibration (Section 11) with a reliability-diagram evaluation | Low | Medium — directly improves trustworthiness of the reported "94.2%" style output | Yes, straightforward and worth including |
| 9 | Robustness to JPEG re-compression/screenshot/resize | Addressed by some papers (ADCD-Net specifically targets compression robustness) via architecture; less commonly evaluated end-to-end with a full benchmark suite in a student project | A dedicated robustness test suite (Section 22.3) evaluating the *full* pipeline (not just the backbone) under these distortions | Medium | High — this is what makes results credible outside the synthetic-benchmark bubble | **Yes — realistic and valuable** |
| 10 | Adversarial robustness | Active research area, generally out of scope for available time/expertise | Not pursued | High | Low ROI for a student timeline | **No — future work only** |
| 11 | Cross-domain/generalization testing | Standard in recent papers (DocTamper's own FCD/SCD splits, T-SROIE, RTM exist for this) | Use the existing standardized cross-domain protocol rather than inventing a new one | Low | High — directly required to make any performance claim credible | **Yes — do this, it's not novel but it is necessary** |

### Top 3 contributions to actually implement
1. **OCR + pixel-level localization fusion with a joint forensic report** (idea #3) — the clearest genuine differentiator given the project's realistic scope.
2. **Explainable, human-readable forensic report pipeline** (idea #5) — turns a research model into something a non-technical evaluator (or viva panel) can immediately understand and trust.
3. **Rigorous cross-domain + distortion-robustness evaluation** (ideas #9 + #11) — this is what separates "we trained a model that gets 95% on the test split it was trained on" from a credible, defensible result, and is exactly what recent SOTA papers use to establish claims.


---

## 18. Proposed Unique System — "DocGuard"

```
Document Input (scan/photo/PDF)
        ↓
Preprocessing (Section 8: deskew, rectify, CLAHE — forensic-signal-preserving)
        ↓
   ┌────────────────────────┬─────────────────────────┐
   │  Visual Forensic Stream  │   OCR Analysis Stream       │
   │  (RGB + DCT hybrid CNN)  │  (Tesseract/PaddleOCR →     │
   │                            │   font/spacing/field checks) │
   └────────────┬─────────────┴──────────────┬─────────────┘
                │                              │
        Multi-Task Deep Backbone         Rule-based Anomaly Flags
                │                              │
   ┌────────────┼───────────────┐             │
   │            │               │             │
Authentic/    Forgery-Type   Pixel-Level      │
 Forged       Classification  Localization    │
   │            │               │             │
   └────────────┴───────┬───────┴─────────────┘
                         ↓
              Late Fusion / Report Aggregation
                         ↓
              Explainable AI (Grad-CAM + mask overlay
                 + auto-generated textual reasoning)
                         ↓
              Confidence Calibration
                         ↓
              Final Forensic Report (JSON + PDF)
```

**Why this beats a simple CNN classifier:** a plain classifier gives one number and no explanation — unusable for any real forensic, legal, or compliance workflow, where a human always needs to verify *why* a document was flagged. DocGuard instead produces (a) a calibrated confidence score, (b) a pixel-accurate location, (c) a forgery-type label, and (d) two independent corroborating signals (pixel forensics + OCR anomaly) that either agree (high-trust flag) or disagree (flagged for manual review) — which is a materially more decision-useful output, and is defensible in a viva as more than "we added more layers."

---

## 19. Algorithms We Will Actually Use — Final Table

### Core algorithms

| Component | Algorithm/Model | Purpose | Why Selected | Alternative |
|---|---|---|---|---|
| RGB backbone | ResNet-50 (ImageNet-pretrained) | Spatial/semantic feature extraction | Proven, well-supported, good accuracy/compute balance | EfficientNet, Swin Transformer |
| Forensic backbone | Block-wise 8×8 DCT + shallow CNN | Compression-artifact/double-JPEG detection | Strongest literature-validated signal for document tampering | DWT features |
| Fusion | Cross-attention block | Combine RGB + DCT streams | Lets each stream attend to relevant regions of the other; used by DocForgeNet-style architectures | Simple feature concatenation |
| Localization decoder | U-Net | Pixel-level forgery mask | Reliable, proven, modest compute | U-Net++, DeepLabV3+ |
| Classification head | FC + softmax (2-way + 5-way) | Authentic/forged + forgery type | Simple, standard | — |
| OCR engine | Tesseract (MVP) / PaddleOCR (upgrade) | Text extraction + font/spacing anomaly detection | Free, mature, easy integration | EasyOCR |
| Explainability | Grad-CAM | Visual explanation overlay | Cheap, architecture-agnostic, intuitive | Grad-CAM++, Integrated Gradients |
| Calibration | Temperature scaling | Meaningful confidence scores | Simple, proven post-hoc method | Platt scaling |

### Supporting algorithms

| Component | Algorithm | Purpose |
|---|---|---|
| Document boundary detection | Canny edge detection + `cv2.findContours` | Locate document edges for perspective correction |
| Perspective correction | `cv2.getPerspectiveTransform` | De-warp photographed documents |
| Deskew | Hough Transform | Correct residual rotation |
| Contrast enhancement | CLAHE | Mild, forensic-signal-preserving contrast normalization |
| Inpainting (for dataset generation) | OpenCV Telea/Navier-Stokes inpainting | Generate erasure-forgery training samples |

### Optional advanced algorithms

| Component | Algorithm | Purpose |
|---|---|---|
| Signature/stamp detector | YOLOv8n (fine-tuned) | Localize signature/stamp regions for focused copy-move checks |
| Localization decoder upgrade | U-Net++ or DeepLabV3+ | Ablation comparison against base U-Net |
| Explainability upgrade | Grad-CAM++ | Sharper heatmaps |

---

## 20. Loss Functions

**Classification (authentic/forged + forgery-type):**
- Binary Cross-Entropy for the authentic/forged head.
- Categorical Cross-Entropy (with label smoothing 0.1) for the 5-way forgery-type head, applied only on forged samples.

**Segmentation (localization mask):**
- **Combined Dice + BCE loss**, since Dice handles the severe class imbalance of forgery masks (most pixels in any document are authentic; the tampered region is usually a small minority of pixels) while pixel-wise BCE stabilizes early training when the Dice gradient is unstable on near-empty masks.
- Optionally add **Focal Loss** in place of plain BCE if class imbalance remains a problem after Dice weighting (empirically verify during Phase 4, Section 27).

**Multi-task combination:**

L_total = λ_cls · L_cls + λ_type · L_type + λ_seg · (α · L_Dice + (1-α) · L_BCE)

Recommended starting weights: λ_cls = 1.0, λ_type = 0.5 (secondary task, lower weight since it's only defined for forged samples), λ_seg = 1.0, α = 0.5. These are starting points, not fixed values — tune via validation-set performance per task (Section 21).


---

## 21. Training Strategy

- **Split:** standard DocTamper train set for training/validation (e.g., 90/10 split within the official train set), official DocTamper-Test/FCD/SCD held out purely for evaluation; T-SROIE and RTM used **only** for cross-domain evaluation, never for training, to keep the generalization claim valid.
- **Cross-validation:** not necessary given DocTamper's scale (120k+ training fakes); a single held-out validation split is sufficient and standard practice for this dataset size.
- **Batch size:** start at 16–32 (GPU-memory dependent for 512×512 dual-stream inputs); increase if memory allows.
- **Epochs:** start with 20–30 epochs with early stopping on validation IoU (localization) + F1 (classification); document-tampering models in the literature typically converge within this range given a pretrained backbone.
- **Optimizer:** AdamW, initial LR 1e-4 for new layers, 1e-5 for the pretrained ResNet backbone (differential LR).
- **Scheduler:** cosine decay with linear warmup (first ~5% of total steps).
- **Early stopping:** patience of 5 epochs on validation combined metric (IoU + F1), to avoid overfitting to DocTamper's synthetic generation artifacts.
- **Class imbalance handling:** Dice loss (Section 20) for the segmentation imbalance; for the forgery-type classifier, use class-weighted cross-entropy if forgery types are unevenly represented in the training mix.
- **Data augmentation:** random crop (within the 512×512 protocol), horizontal flip, mild brightness/contrast jitter, and — critically — **re-JPEG-compression augmentation at varying quality factors**, since this both increases robustness (Section 22.3) and matches the real-world distortion the DCT stream needs to generalize across. Avoid aggressive blur/noise augmentation on the RGB stream that would wash out the same forensic signal Section 8 protects.
- **Transfer learning:** ResNet-50 backbone initialized from ImageNet weights; DCT-stream and fusion/decoder layers trained from scratch.
- **Freeze/unfreeze strategy:** freeze the ResNet backbone for the first 2–3 epochs (train only the new heads/decoder/DCT stream), then unfreeze and fine-tune the whole network end-to-end at the lower differential LR — this "warm start" avoids the pretrained features being destroyed by large early gradients from randomly-initialized new layers.

---

## 22. Evaluation

### Classification
Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC — PR-AUC is emphasized over ROC-AUC as the headline metric if the authentic/forged class balance in the deployed use case is skewed (report both, but discuss PR-AUC's robustness to imbalance explicitly).

### Localization / Segmentation
IoU, Dice coefficient, pixel accuracy, precision, recall — computed following the DocTamper/ADCD-Net "Doc Protocol" evaluation convention so results are directly comparable to published numbers rather than an ad hoc metric.

### Practical robustness
Test the **full trained pipeline** (not just the backbone in isolation) against:
- JPEG re-compression at multiple quality factors
- Gaussian blur
- Additive noise
- Resizing (up/down)
- Screenshot-style re-capture (simulated: render to screen resolution, re-save as PNG)
- Different (simulated) scanner profiles — vary brightness/contrast/color-cast
- Cross-domain document types (train on DocTamper, test on T-SROIE and RTM)

Each metric matters because: accuracy/F1 alone can hide a model that only ever predicts "authentic" on an imbalanced set (hence PR-AUC); IoU/Dice directly measure the *localization* quality that is the project's core deliverable, not just detection; and the robustness suite is what proves the system works outside the exact synthetic-generation pipeline it was trained on — a paper or viva claim of "94% accuracy" is meaningless without this.

---

## 23. Baseline vs. Proposed System

| Model | Description | Purpose |
|---|---|---|
| **Baseline 1** | Traditional forensics: DCT/ELA/LBP features → SVM classifier, no localization | Shows the ceiling of pre-deep-learning forensic methods |
| **Baseline 2** | Simple CNN classifier (ResNet-50 fine-tuned, classification-only, no localization/DCT stream) | Shows the gap a plain classifier leaves — no localization, weaker on subtle traces |
| **Baseline 3** | Transfer-learning classification + a bolted-on U-Net localization head trained *separately* (not jointly, not DCT-aware) | Isolates the value of joint multi-task training and the DCT stream specifically |
| **Proposed (DocGuard)** | Full hybrid RGB+DCT, multi-task, OCR-fused, explainable pipeline | The complete system |

**Experiments:** train all four on the identical DocTamper split; report classification F1/AUC and localization IoU/Dice for each; then run the full robustness suite (Section 22.3) on all four to show the proposed system doesn't just win on the clean test set but generalizes and survives distortion better — this comparison is what demonstrates genuine improvement, not just a marginally higher single accuracy number.


---

## 24. Ablation Study

| Model | Configuration |
|---|---|
| A | RGB CNN only (no DCT stream, no OCR fusion) |
| B | RGB CNN + DCT/forensic stream (no fusion attention — simple concatenation) |
| C | RGB CNN + DCT stream + cross-attention fusion (no OCR) |
| D | Full model (C) + OCR anomaly fusion, but classification/localization trained separately (not multi-task) |
| E | Full proposed DocGuard architecture (multi-task, fused, calibrated) |

**Scientific purpose:** this progression isolates the marginal contribution of each design decision — A→B shows whether the DCT stream helps at all; B→C shows whether attention-based fusion beats naive concatenation; C→D shows whether OCR fusion adds value; D→E shows whether joint multi-task training beats separate models. Reporting all five prevents an unsupported "our combination is better" claim and gives concrete evidence for the Section 17 novelty claims — this is standard practice and directly answerable in a viva ("why do you have a DCT stream?" → "see Model A vs. B").

---

## 25. System Architecture (Text Diagram)

```
Frontend (React/HTML upload UI)
        ↓  (HTTPS, multipart file upload)
Backend API (FastAPI)
        ↓
Preprocessing Service (OpenCV: rectify, deskew, CLAHE)
        ↓
ML Inference Pipeline (PyTorch model server)
   ├─ RGB + DCT hybrid backbone
   ├─ Classification + localization heads
   └─ OCR module (Tesseract/PaddleOCR)
        ↓
Explainability Module (Grad-CAM overlay + textual reasoning generator)
        ↓
Result Storage (PostgreSQL for metadata/history; local/object storage for uploaded images + generated reports)
        ↓
Result & Visualization (returned to frontend: JSON + downloadable PDF report)
```

**Technology stack:**
- **Python** — implementation language for the entire ML pipeline.
- **OpenCV** — preprocessing, rectification, classical forensic feature computation.
- **PyTorch** — model implementation and training (recommended over TensorFlow here mainly because most of the cited recent forgery-localization papers — ADCD-Net, DocForgeNet — ship PyTorch reference code, easing reproduction/comparison).
- **Tesseract/PaddleOCR** — OCR component.
- **FastAPI** — backend API (async-friendly, good fit for a Python ML inference service, lighter than Flask for this use case).
- **React** (or plain HTML/JS for a lighter MVP) — frontend upload/result UI.
- **PostgreSQL** — only if the project needs persistent history/multi-user results; **for a pure single-session demo, a lightweight SQLite or even no database at all is sufficient** — don't add PostgreSQL/MongoDB complexity unless the report specifically needs a "history of past scans" feature.

---

## 26. Implementation Folder Structure

```
project/
│
├── data/                # raw + generated datasets, DocTamper/T-SROIE/RTM subsets, self-generated samples
├── preprocessing/        # rectification, deskew, CLAHE, DCT extraction scripts
├── models/                # model definitions (backbone, fusion, heads)
├── training/              # training loop, loss functions, config files
├── inference/              # inference pipeline wrapping preprocessing + model + postprocessing
├── ocr/                     # OCR wrapper + font/spacing anomaly detection logic
├── localization/             # mask post-processing, thresholding, mask-to-bbox conversion
├── explainability/             # Grad-CAM implementation, textual-reasoning template generator
├── evaluation/                   # metric computation, ablation/baseline experiment scripts
├── backend/                        # FastAPI app, routes, request/response schemas
├── frontend/                        # upload UI, result visualization components
└── README.md                          # setup instructions, how to reproduce experiments
```


---

## 27. Complete Implementation Roadmap

| Phase | Tasks | Expected Output | Algorithms/Tools | Dependencies | Definition of Done |
|---|---|---|---|---|---|
| **1. Dataset** | Download DocTamper, T-SROIE, RTM, CASIA v2; build self-generation scripts (Section 7) | Ready-to-train dataset + generation pipeline | Section 6–7 tools | None | Loader produces (image, mask, type-label) triples correctly for all sources |
| **2. Preprocessing** | Implement rectification, deskew, CLAHE, DCT extraction | Preprocessing module | OpenCV | Phase 1 | Visual QA on sample images confirms forensic signal preserved (Section 8) |
| **3. Baseline model** | Train Baseline 1 (classical+SVM) and Baseline 2 (plain CNN classifier) | Baseline metrics | scikit-learn, PyTorch | Phase 1–2 | Reproducible baseline numbers logged |
| **4. Forgery localization** | Implement RGB+DCT hybrid + U-Net decoder, train Model E core (no OCR yet) | Trained localization model | PyTorch | Phase 1–3 | IoU/Dice on DocTamper-Test within reasonable range of published baselines |
| **5. OCR integration** | Implement OCR wrapper + font/spacing anomaly detector | OCR module | Tesseract/PaddleOCR | Phase 2 | Anomaly flags correlate with known forged text regions on validation samples |
| **6. Hybrid/multi-task model** | Joint training of classification + localization + OCR late fusion | Full DocGuard model | PyTorch | Phase 3–5 | End-to-end pipeline produces classification+mask+type+OCR flags on a held-out sample |
| **7. Explainability** | Grad-CAM implementation, textual-reasoning generator | Explanation module | Captum/custom Grad-CAM | Phase 6 | Generates the Section 16 output format correctly |
| **8. Evaluation** | Run full metric suite, ablation study, baseline comparison, robustness suite | Results tables/figures for report | Section 22–24 tools | Phase 3–7 | All tables in Sections 22–24 populated with real numbers |
| **9. Web application** | Build FastAPI backend + frontend upload/result UI | Working demo app | FastAPI, React | Phase 6–7 | End-to-end upload → report flow works in a browser |
| **10. Research paper/report** | Write up methodology, results, novelty discussion, citations | Final report/paper draft | — | Phase 8 | Draft covers all sections of this document with real experimental numbers |

---

## 28. What We Should NOT Implement

- **Metadata-manipulation detection** — unreliable given scan/photo/screenshot inputs (Section 5.I).
- **Freehand ("skilled") signature-forgery detection as a core claim** — this is a distinct biometric-verification research problem requiring dedicated signature datasets and different modeling (online/offline signature verification literature), not something the document-forensics pipeline here supports; only claim the copy-paste-signature case.
- **Source-scanner/printer identification** — a legitimate but separate research area (see the printer-source-identification literature in Section 1's background reading) requiring its own dataset and features; don't bolt it on as an afterthought.
- **Full adversarial-robustness certification** — genuinely valuable but requires specialized expertise/compute beyond a typical student timeline; note as future work rather than attempting a shallow, unconvincing version.
- **Hyperspectral ink analysis** — requires specialized scanning hardware most student projects don't have access to; don't claim it even though it appears in the literature (Section on hyperspectral document forensics), since it can't be validated without the hardware.
- **Document-replacement (whole-page swap) detection** — needs multi-page/cross-document context modeling that's a different problem from single-image pixel forensics; don't quietly fold it into the pixel-level localization claim.
- **Overly large/novel transformer architectures "because they're new"** — Section 10 explicitly justifies *not* chasing the newest published model; a fragile, hard-to-reproduce SOTA model that can't be trained to convergence on available compute is worse for a project report than a solid, well-evaluated moderate model.
- **A full relational database with user accounts/auth for the demo** — unnecessary complexity unless the report specifically requires a multi-user product angle; keep the demo backend as simple as the requirements allow (Section 25).


---

## 29. Risks and Limitations

| Risk | Mitigation |
|---|---|
| **Dataset bias** — DocTamper's synthetic generation pipeline has its own statistical signature that a model can shortcut-learn instead of learning genuine forgery traces | Mandatory cross-domain evaluation on T-SROIE and RTM (Section 22–23), which use different generation methods |
| **Synthetic forgery limitations** — self-generated and DocTamper-style forgeries may not capture the subtlety of a skilled human forger using professional editing software | Include the RTM dataset's manually-manipulated samples; be explicit in the report that MVP results are strongest on synthetic-style tampering |
| **Domain shift** — real-world scans/phone photos differ from benchmark image conditions | Robustness suite (Section 22.3); collect a small manually-curated real-world test set for qualitative demo validation |
| **False positives** — flagging authentic documents as forged, especially where legitimate re-compression/scanning noise resembles tampering traces | Confidence calibration (Section 11); report precision explicitly, not just recall, since false positives have real consequences in a compliance workflow |
| **False negatives** — sophisticated or low-contrast tampering missed | Ensemble corroboration (pixel forensics + OCR, Section 12) so a single missed signal doesn't silently fail |
| **OCR errors** — misread text producing spurious anomaly flags | Treat OCR-based flags as *corroborating* evidence only, never as the sole basis for a forged verdict (Section 18's late-fusion design already enforces this) |
| **Compression effects** — aggressive re-compression can degrade both forensic signal and OCR accuracy | JPEG-quality-factor augmentation during training (Section 21) explicitly targets this |
| **Document-quality variation** — poor scans, low light, heavy shadows | Document-boundary detection + CLAHE preprocessing (Section 8), and reporting a "low document quality" warning when the alignment/OCR confidence is low, rather than a false forgery verdict |
| **Model explainability limitations** — Grad-CAM heatmaps can be imprecise or misleading, and users may over-trust a wrong heatmap | Present the trained localization mask (not Grad-CAM) as the primary spatial evidence, with Grad-CAM explicitly labeled as a secondary/diagnostic overlay (Section 16) |

---

## 30. Expected Final Demo

```
User uploads a document image/PDF via the web UI
        ↓
System runs preprocessing (rectify, deskew) — shows the corrected image
        ↓
Prediction appears:
  "FORGED — 92.7% confidence"
        ↓
Forgery region highlighted on the image (mask overlay)
        ↓
Forgery type shown:
  "Text Tampering"
        ↓
OCR cross-check panel:
  "Font mismatch detected: date field uses a different font than the rest of the document"
        ↓
Grad-CAM heatmap shown as a secondary diagnostic overlay
        ↓
Final report: downloadable PDF containing the original image, the highlighted mask,
  the heatmap, the OCR findings, the confidence score, and a plain-language summary
```

This is realistic and implementable with the components already specified in Sections 4, 11, 13, 16, and 25 — no component in the demo requires anything beyond what's in the roadmap.


---

## 31. Final Recommendation

**A. Final problem statement:** Design a deep-learning system that classifies documents as authentic/forged, localizes tampered regions at pixel level, identifies the forgery type, and produces a human-interpretable explanation, robust to real-world distortions (Section 2.6).

**B. Final workflow:** Input → Validation → PDF/Image conversion → Preprocessing (forensic-signal-preserving) → Alignment/Rectification → Dual RGB+DCT feature extraction → Multi-task hybrid model (classification + localization) → OCR cross-check → Explainability (mask + Grad-CAM + text) → Confidence calibration → Final report (Section 4).

**C. Exact algorithms/models:** ResNet-50 (RGB stream) + block-DCT CNN (forensic stream) fused via cross-attention → U-Net localization decoder + FC classification heads; Tesseract/PaddleOCR for text anomaly checks; Grad-CAM for secondary explanation; temperature scaling for calibration (Section 19).

**D. Dataset(s):** DocTamper (primary training + in-domain test), T-SROIE and RTM (cross-domain evaluation), CASIA v2 (optional forensic-stream pretraining), plus a self-generated demo/template dataset (Section 6–7).

**E. Final architecture:** the dual-stream RGB+DCT multi-task hybrid with U-Net localization decoder and OCR late-fusion, as diagrammed in Sections 11 and 18.

**F. Top 3 novel contributions:** (1) OCR + pixel-level localization fusion with a joint forensic report; (2) full explainable-report pipeline (mask + heatmap + text); (3) rigorous cross-domain and distortion-robustness evaluation (Section 17).

**G. Evaluation metrics:** Accuracy/Precision/Recall/F1/ROC-AUC/PR-AUC for classification; IoU/Dice/pixel-accuracy for localization; a dedicated robustness suite across JPEG re-compression, blur, noise, resizing, screenshot-capture, and cross-domain datasets (Section 22).

**H. Technology stack:** Python, OpenCV, PyTorch, Tesseract/PaddleOCR, FastAPI, React (or lightweight HTML/JS), SQLite (or PostgreSQL only if multi-user history is required) (Section 25).

**I. Implement first:** Phases 1–4 (dataset, preprocessing, baselines, core localization model) — get a working RGB+DCT localization model evaluated on DocTamper-Test before adding OCR fusion or the web app.

**J. Leave as future work:** signature freehand-forgery verification, source-scanner identification, adversarial robustness, hyperspectral ink analysis, whole-document/page-replacement detection (Section 28).

---

## One-Page Cheat Sheet

**Workflow:** Input → Preprocess (forensic-preserving) → Rectify → RGB+DCT dual-stream extraction → Multi-task hybrid model → Localization mask + type + OCR check → Grad-CAM + text explanation → Calibrated confidence → Report.

**Core algorithms:** ResNet-50 (RGB) + block-DCT CNN (forensic) → cross-attention fusion → U-Net (localization) + FC heads (classification); Tesseract/PaddleOCR (text anomaly); Grad-CAM (explanation); temperature scaling (calibration).

**Datasets:** DocTamper (train + in-domain test) → T-SROIE + RTM (cross-domain test) → CASIA v2 (optional pretrain) → self-generated demo set.

**Architecture name:** DocGuard — hybrid CNN+DCT, multi-task (classify + localize), OCR-fused, explainable.

**Novelty (realistic):** OCR+pixel-localization fusion; full explainable-report pipeline; rigorous cross-domain/robustness evaluation. *(Not novel: the base CNN+DCT fusion itself — cite ADCD-Net/DocForgeNet.)*

**Evaluation:** Accuracy/F1/ROC-AUC/PR-AUC (classification) · IoU/Dice (localization) · robustness suite (JPEG/blur/noise/resize/screenshot/cross-domain).

**Forgery types covered:** copy-move, splicing, text tampering, erasure (core); copy-pasted signature/stamp (partial). **Not covered:** skilled signature forgery, metadata tampering, whole-page replacement.

**Loss:** BCE (classification) + CE w/ label smoothing (forgery type) + Dice+BCE (segmentation), weighted sum.

**Stack:** Python, OpenCV, PyTorch, Tesseract/PaddleOCR, FastAPI, React.

---

## Key References (for citation in the report)

- Qu, C. et al. "DocTamper: Towards Robust Tampered Text Detection in Document Image with New Dataset and New Solution." (dataset + benchmark used for training/evaluation)
- Wong, K. et al. "ADCD-Net: Robust Document Image Forgery Localization via Adaptive DCT Feature and Hierarchical Content Disentanglement." ICCV 2025. (architecture inspiration for the DCT-aware localization stream)
- "DocForgeNet: Dual Cross-Stream Fusion Network for Robust Forgery Detection in Scanned Documents." Springer. (dual RGB/DCT cross-stream fusion design inspiration)
- Dong, J., Wang, W., Tan, T. "CASIA Image Tampering Detection Evaluation Database." ChinaSIP 2013. (supplementary copy-move/splicing dataset)
- Martínez Tornés, B. et al. "Receipt Dataset for Document Forgery Detection." ICDAR 2023. (receipt-specific dataset)
- "From Forgeries to Foundation Models: A Systematic Survey of Identity Document Attack and Detection." arXiv 2607.01442. (background on identity-document forgery taxonomy and the shift toward generative attacks)
- ForensicHub benchmark paper (arXiv 2505.11003) — for the standardized "Doc Protocol" cross-domain evaluation splits (DocTamper/T-SROIE/RTM/OSTF).
- Guo, C. et al. "On Calibration of Modern Neural Networks." ICML 2017. (temperature scaling for confidence calibration)

*Note: verify exact author lists, venues, and page numbers against the original sources before final submission — some of the above were retrieved from preprint/aggregator listings and should be cross-checked against the publisher's canonical citation.*
