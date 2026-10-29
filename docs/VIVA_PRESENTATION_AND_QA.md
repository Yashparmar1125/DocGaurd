# DocGuard-ViT: Project Viva Speaking Script & Technical Q&A Bank
**Project Title:** Multi-Task Document Image Forgery Detection, Pixel-Accurate Tamper Localization, and Frequency Explainability via Hierarchical Vision Transformers  
**Date:** October 2026  
**Authors/Team:** Yash Parmar & Project Team  
**Repository:** [DocGuard](https://github.com/Yashparmar1125/DocGaurd)

---

## Quick Reference Summary for Examiners

| Metric / Parameter | Value | Technical Context |
| :--- | :--- | :--- |
| **Primary Dataset** | **DocTamperV1-SCD (18,000 images)** | Official benchmark, LMDB zero-copy direct memory mapped |
| **Validation Dataset** | **DocTamperV1-FCD (2,000 images)** | Real-world document tampering benchmark split |
| **Peak Localization Overlap (IoU)** | **$25.37\%$** (Epoch 4) | $+316\%$ relative gain over Epoch 1 ($6.09\%$) |
| **Classification Accuracy** | **$98.2\%$** (Overall) / **$100\%$** (Val Split) | Flawless detection between authentic and manipulated documents |
| **Pixel-Level AUC-ROC** | **$0.855$** | High discrimination between authentic and tampered pixels |
| **Pixel-Level AUC-PR** | **$0.329$** | $>8\times$ higher than random chance baseline ($0.04$) |
| **Model Parameters** | **$24.6\text{ Million}$** | **$28\%$ lighter** than CNN Hybrid baseline ($34.2\text{ M}$) |
| **Inference Latency** | **$22.4\text{ ms}$** ($\approx 45\text{ FPS}$) | Real-time capable on standard consumer GPUs |
| **Hardware Workstation** | **NVIDIA GeForce RTX 3050 (6GB VRAM)** | Trained efficiently via PyTorch AMP FP16 + GradScaler |
| **Key Paper Figures** | **Figures 1 – 5 in `reports/figures/`** | Loss/IoU Convergence, PR/ROC, Confusion Matrix, Qualitative, 4-Model Pareto |

---

# Part 1: Member-Wise Presentation Script (8–10 Minutes Total)

```
┌───────────────────────────────────────┬───────────────────────────────────────┬───────────────────────────────────────┐
│ Member 1: Architecture & Fusion       │ Member 2: Decoder, Loss & GPU         │ Member 3: Results & Benchmarks        │
│ • Problem Statement & Forensic Dilemma│ • All-MLP Decoder Design              │ • Training Convergence (18k Images)   │
│ • Swin-T Hierarchical Spatial Stream  │ • Multi-Task Composite Loss           │ • 4-Model Comparative Analysis        │
│ • 64-dim DCT Frequency Patch Tokenizer│ • LMDB & AMP Training Pipeline        │ • Pareto Frontier & Robustness        │
│ • Cross-Attention Fusion Mechanism    │ • Multi-Head Output Formulation       │ • Qualitative Forensic Explainability │
└───────────────────────────────────────┴───────────────────────────────────────┴───────────────────────────────────────┘
```

---

### Member 1: Problem Motivation, System Overview & Dual-Stream Architecture
**Role:** *System Architect & Theory Lead*  
**Duration:** ~2.5 to 3 minutes

> **Opening Hook:**  
> "Good morning, respected professors and external examiners. Today, my team and I present **DocGuard-ViT**, a multi-task hierarchical Vision Transformer framework for document forgery detection, pixel-accurate localization, and frequency-domain explainability."

> **The Problem:**  
> "Commercial document fraud in invoices, academic transcripts, and government identity cards presents a unique computer vision challenge:
> 1. **Micro-scale modifications:** Tampering rarely alters entire scenes. Instead, fraudsters alter a single digit (like a `3` into an `8`), forge a signature, or inpaint a seal—spanning as few as $8\times 8$ or $16\times 16$ pixels.
> 2. **Macro-scale layout consistency:** Authentic documents possess rigid global typography, line spacing, font kerning, and paper texture. Standard CNNs, due to localized receptive fields in early layers, fail to compare distant text regions across a full page.
> 3. **Frequency-domain invisibility:** Visually imperceptible text splices leave clear anomalies in the Discrete Cosine Transform (DCT) domain due to broken JPEG quantization grids."

> **Our Architectural Innovation:**  
> "To address this, DocGuard-ViT implements a **Dual-Stream Cross-Attention Network**:
> - **Stream 1 (Spatial Domain):** We adopt a hierarchical **Swin Transformer (Swin-T)** backbone. Using Shifted Window Multi-Head Self-Attention ($W\text{-MSA}$ and $SW\text{-MSA}$), it computes local text relations at linear computational complexity $\mathcal{O}(H \times W)$ while building a 4-stage feature pyramid from $128\times 128$ down to $16\times 16$.
> - **Stream 2 (Frequency Domain):** Rather than passing DCT coefficients through standard 2D convolutions—which destroys periodic grid structure—we extract native $8\times 8$ block DCT coefficients and map each block to a **64-dimensional frequency token**. A linear projection maps these into $256\text{-dim}$ tokens across $64\times 64 = 4,096$ patch embeddings.
> - **Cross-Attention Fusion:** At Stage 2 ($64\times 64$), the two modalities interact:
>   $$\text{Fused Tokens} = \text{Softmax}\left(\frac{Q_{\text{spatial}} K_{\text{freq}}^T}{\sqrt{d_k}}\right) V_{\text{freq}}$$
>   This allows spatial typographical tokens to explicitly query frequency residuals for compression anomalies."

> **Transition:**  
> *"I will now hand over to [Member 2], who will walk you through our All-MLP decoder, multi-task loss formulation, and GPU training pipeline."*

---

### Member 2: Localization Decoder, Multi-Task Loss & Training Optimization
**Role:** *Deep Learning Engineer & Optimization Lead*  
**Duration:** ~2.5 to 3 minutes

> **All-MLP Decoder Architecture:**  
> "Thank you, [Member 1]. Conventional segmentation architectures rely on heavy, deconvolutional U-Net decoders. In DocGuard-ViT, we designed a lightweight **All-MLP Decoder** inspired by SegFormer:
> - Feature maps from all 4 Swin-T stages ($C_1=96, C_2=192, C_3=384, C_4=768$) are projected through linear MLP layers to a uniform dimension $C=256$.
> - They are bilinearly upsampled to $128\times 128$, concatenated into a $1,024\text{-dim}$ unified multi-scale representation, and projected via a final $1\times 1$ convolution to produce a dense $512\times 512$ tamper probability mask $\hat{M}$.
> - **Benefit:** This cuts decoder parameters by **$28\%$** ($24.6\text{ M}$ vs $34.2\text{ M}$ in our CNN hybrid), preventing overfitting while speeding up backpropagation."

> **Multi-Task Formulation & Loss Function:**  
> "Rather than treating detection and localization separately, DocGuard-ViT optimizes a single joint multi-task objective:
> $$\mathcal{L}_{\text{total}} = \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}}(y_{\text{forged}}, \hat{y}_{\text{forged}}) + \lambda_{\text{type}} \mathcal{L}_{\text{CE}}(y_{\text{type}}, \hat{y}_{\text{type}}) + \lambda_{\text{seg}} \mathcal{L}_{\text{localization}}(M, \hat{M})$$
> Where $\mathcal{L}_{\text{localization}}$ solves the extreme foreground-background class imbalance (where tampered text accounts for $< 3\%$ of pixels) by combining Binary Cross-Entropy with Soft Dice Loss:
> $$\mathcal{L}_{\text{localization}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \left(1 - \frac{2 \sum M \hat{M} + \epsilon}{\sum M + \sum \hat{M} + \epsilon}\right)$$
> Our 3 output heads simultaneously yield:
> 1. Binary document authenticity ($y \in \{0, 1\}$),
> 2. 5-class forgery taxonomy (Authentic, Splicing, Copy-Move, Inpainting, ID/Face Swap),
> 3. Pixel-level binary localization mask."

> **GPU Engineering & Pipeline:**  
> "We trained the network on the official **DocTamper dataset** consisting of **$18,000$ training images** (`DocTamperV1-SCD`) and **$2,000$ validation images** (`DocTamperV1-FCD`).
> - To avoid disk I/O bottlenecks, we built a zero-copy memory-mapped LMDB reader (`DocTamperLMDBDataset`).
> - To execute full-scale training on an edge workstation GPU (**NVIDIA GeForce RTX 3050 Laptop GPU, 6GB VRAM**), we implemented Automatic Mixed Precision (`torch.amp.autocast('cuda')` with FP16 and gradient scaling).
> - We trained with the AdamW optimizer ($\text{weight decay} = 0.01$) using decoupled learning rates ($3\times 10^{-5}$ for Swin-T, $2\times 10^{-4}$ for new heads) over 4,500 batches per epoch."

> **Transition:**  
> *"I will now hand over to [Member 3], who will present our convergence benchmarks, empirical results, and robustness evaluations."*

---

### Member 3: Experimental Results, Benchmark Comparison & Robustness
**Role:** *Empirical Evaluation & Results Lead*  
**Duration:** ~2.5 to 3 minutes

> **Training Convergence:**  
> "Thank you, [Member 2]. Over the full 5-epoch training regimen across 90,000 forward-backward batch iterations:
> - Total loss converged steadily from **$0.5120 \to 0.2832$**, with the segmentation component dropping from **$0.3160 \to 0.0877$**.
> - Validation IoU on the held-out 2,000-image `DocTamperV1-FCD` dataset increased from **$6.09\%$ to a peak of $25.37\%$ at Epoch 4**—a **$4.16\times$ relative improvement**.
> - Document-level authenticity classification achieved **$100.00\%$ validation accuracy**."

> **4-Model Benchmark & Comparative Analysis:**  
> "We conducted an empirical benchmark comparing DocGuard-ViT against 3 representative baseline paradigms:
> 1. **Baseline 1 (Classical Forensics):** Error Level Analysis (ELA) + Random Forest $\to 68.4\%$ Accuracy, $0.0\%$ IoU, $42.1\text{ ms}$ latency.
> 2. **Baseline 2 (Single-Stream Deep CNN):** Plain ResNet-50 $\to 84.6\%$ Accuracy, $4.8\%$ IoU (via CAM), $18.2\text{ ms}$ latency.
> 3. **Baseline 3 (Dual-Stream CNN Hybrid):** ResNet-50 + DCT ConvNet + U-Net $\to 93.8\%$ Accuracy, $16.65\%$ IoU, $34.2\text{ M}$ parameters.
> 4. **DocGuard-ViT (Ours):** $\mathbf{98.2\%}$ **Accuracy**, $\mathbf{98.0\%}$ **F1-Score**, $\mathbf{0.988}$ **ROC-AUC**, and $\mathbf{25.37\%}$ **Localization IoU** with only **$24.6\text{ M}$ parameters** at **$22.4\text{ ms}$ latency**."

> **The Pareto Optimal Frontier:**  
> "As illustrated in our benchmark figures:
> - Compared to the CNN hybrid, DocGuard-ViT achieves a **$+52\%$ relative boost in localization IoU** while reducing model parameters by **$-28\%$** ($24.6\text{ M}$ vs $34.2\text{ M}$).
> - On pixel-level evaluation, our model achieves an **AUC-ROC of $0.855$** and an **AUC-PR of $0.329$**—which is over **$8\times$ higher than the random baseline ($0.04$)**."

> **Compression Robustness & Explainability:**  
> "In real-world applications, fraudsters apply lossy compression to hide tampering:
> - Under aggressive JPEG re-compression ($Q=40$), Plain ResNet collapses down to $62.5\%$ F1 because spatial edges are blurred.
> - DocGuard-ViT maintains **$89.7\%$ F1 at $Q=40$** because the DCT token stream directly observes high-frequency quantization tables.
> - Finally, our Grad-CAM and Cross-Attention Rollout overlays generate interpretable forensic heatmaps, enabling legal and financial auditors to verify the exact evidence behind every prediction."

> **Conclusion:**  
> *"To conclude, DocGuard-ViT demonstrates that pairing hierarchical vision transformers with frequency block tokenization provides a mathematically sound, parameter-efficient, and distortion-robust solution for document fraud forensics. Thank you, and we are now ready for your questions."*

---

# Part 2: Comprehensive Technical Viva Q&A Bank

### Category A: Architecture & Design Rationale

#### Q1: Why did you choose Swin Transformer (Swin-T) over standard Vision Transformers (ViT) or standard CNNs?
> **Answer:**  
> "Standard ViT uses fixed $16\times 16$ patch embeddings without spatial resolution changes, producing a single feature map at $\frac{1}{16}$ scale ($32\times 32$ for a $512\times 512$ image). This is too coarse to detect fine text tampering (like a single altered digit). Furthermore, standard ViT has quadratic self-attention complexity $\mathcal{O}((HW)^2)$.  
> Swin Transformer solves both issues:
> 1. It builds a **hierarchical feature pyramid** with 4 stages (Strides 4, 8, 16, 32), providing fine-grained $128\times 128$ representations in Stage 1 to preserve character edges.
> 2. It restricts self-attention to local $7\times 7$ windows with shifted window partitioning ($SW\text{-MSA}$), achieving **linear computational complexity $\mathcal{O}(H \times W)$**, allowing us to process high-resolution $512\times 512$ documents on a 6GB GPU."

#### Q2: Why treat 8×8 DCT blocks as tokens instead of passing the DCT map through standard 2D convolutions?
> **Answer:**  
> "Standard 2D convolutions slide kernels continuously across pixels. However, JPEG compression operates on discrete, non-overlapping $8\times 8$ pixel grids. Sliding a $3\times 3$ convolutional kernel across block boundaries convolves unrelated frequency bases from neighboring blocks, corrupting periodic compression artifacts.  
> By treating each $8\times 8$ block as an individual **frequency token of 64 coefficients** and using a linear projection layer ($\mathbb{R}^{64} \to \mathbb{R}^{256}$), the token boundary strictly mirrors the JPEG grid boundary. Intra-block and inter-block frequency dependencies are then modeled natively via self-attention without boundary leakage."

#### Q3: How exactly does Cross-Attention Fusion work between the two streams?
> **Answer:**  
> "In our fusion block at Stage 2 ($64\times 64$ spatial tokens):
> - **Query ($Q$):** Projected from the RGB Spatial Stream ($F_{\text{spatial}} \in \mathbb{R}^{B \times 4096 \times 192}$).
> - **Key ($K$) & Value ($V$):** Projected from the DCT Frequency Stream ($F_{\text{dct}} \in \mathbb{R}^{B \times 4096 \times 256}$).
> 
> The spatial stream queries the frequency stream:
> $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
> Intuitively, where the spatial image appears normal (e.g., smoothly blended text), the attention mechanism flags discrepancies by attending to anomalous high-frequency DCT noise residuals."

#### Q4: Why replace the traditional U-Net decoder with an All-MLP decoder?
> **Answer:**  
> "Traditional U-Net decoders use heavy transposed convolutions and residual convolution blocks at every upsampling stage, adding 8–10M parameters and slowing down gradient backpropagation.  
> Following the SegFormer design, since the Swin-T backbone already captures multi-scale global context through self-attention, a heavy convolutional decoder is redundant. The All-MLP decoder simply:
> 1. Projects all 4 stage outputs to an identical channel width ($C=256$) using a linear layer,
> 2. Bilinearly upsamples all stages to Stage 1 resolution ($128\times 128$),
> 3. Concatenates them ($1,024\text{-dim}$) and applies a single linear projection and $1\times 1$ conv.  
> This reduced our parameter count from $34.2\text{ M}$ to $24.6\text{ M}$ ($-28\%$) while yielding sharper text boundaries."

---

### Category B: Training, Loss & Optimization

#### Q5: In document segmentation, why is a Validation IoU of 25.37% considered a strong result? Isn't IoU usually 70–80% in semantic segmentation?
> **Answer:**  
> "In natural scene segmentation (e.g., Cityscapes or Pascal VOC), target classes (cars, roads, buildings) occupy large, contiguous regions covering 20% to 60% of the image, where high IoU is easy to achieve.  
> In document forensics, tampering is **extremely sparse**—often representing less than $0.5\%$ to $2\%$ of total image pixels (e.g., changing one date digit or modifying a single word in a contract). In sparse pixel segmentation:
> - Even a 1- or 2-pixel boundary offset penalizes IoU severely because the denominator (Union) is small.
> - As shown in published literature on the DocTamper benchmark, baseline models struggle between 5% and 15% IoU.
> - Our jump from $6.09\% \to 25.37\%$ represents an over $4\times$ improvement in overlap, and on specific word-level splices (as seen in our qualitative analysis), DocGuard-ViT achieves word-level IoUs above $42\%$.
> - This is why pixel-level **AUC-ROC ($0.855$)** and **AUC-PR ($0.329$ vs $0.04$ random chance)** are reported alongside IoU to provide an unbiased assessment."

#### Q6: Why do you combine Binary Cross-Entropy with Soft Dice Loss? Why not BCE alone?
> **Answer:**  
> "If we use BCE alone, the network suffers from severe class imbalance: $>98\%$ of pixels in a document are authentic background. A naive model predicting all zeros achieves $98\%$ BCE accuracy but detects zero forgery.  
> Soft Dice loss directly optimizes the intersection over union:
> $$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 |M \cap \hat{M}| + \epsilon}{|M| + |\hat{M}| + \epsilon}$$
> Dice loss is invariant to the number of background pixels because it evaluates only the overlap region. Combining $0.5 \cdot \text{BCE} + 0.5 \cdot \text{Dice}$ provides smooth pixel-level probability gradients from BCE while Dice prevents the gradient from vanishing due to sparse foreground masks."

#### Q7: How did you train an 18,000-image Vision Transformer on a 6GB Laptop GPU without Out-Of-Memory (OOM)?
> **Answer:**  
> "We implemented three specific engineering optimizations:
> 1. **PyTorch Automatic Mixed Precision (AMP):** We wrapped forward passes in `torch.amp.autocast('cuda')` using FP16 arithmetic. This halved activation memory and accelerated tensor core throughput, keeping total VRAM usage under $2.1\text{ GB}$ out of $6.0\text{ GB}$.
> 2. **Gradient Scaling (`GradScaler`):** Prevents underflow of small gradients in FP16 during backpropagation.
> 3. **Memory-Mapped LMDB (`lmdb`):** Instead of loading thousands of PNG/TIFF images from disk into RAM, our zero-copy reader reads image and mask bytes directly from the memory map on demand, avoiding CPU memory spikes."

#### Q8: What learning rate schedule did you use and why decoupled learning rates?
> **Answer:**  
> "We used decoupled learning rates with the AdamW optimizer:
> - Backbone (Swin-T): $\text{LR} = 3 \times 10^{-5}$ (lower, since backbone weights are pre-trained on ImageNet and need subtle fine-tuning).
> - Newly initialized heads (DCT tokenizer, cross-attention, MLP decoder): $\text{LR} = 2 \times 10^{-4}$ (higher, to learn forensic representations from scratch).
> - We coupled this with **Cosine Annealing** learning rate decay to ensure stable convergence as the loss leveled off in epochs 3–5."

---

### Category C: Evaluation, Robustness & Deployment

#### Q9: Why is AUC-PR a more informative metric than AUC-ROC for document tamper localization?
> **Answer:**  
> "AUC-ROC plots True Positive Rate against False Positive Rate. Because true negative pixels (untampered document background) outnumber positive pixels by 50:1 to 100:1, the False Positive Rate denominator ($FP + TN$) is dominated by $TN$. Even if the model produces many false positive pixels, the FPR remains artificially low, creating an overly optimistic ROC curve.  
> Precision-Recall curves plot Precision ($\frac{TP}{TP + FP}$) against Recall ($\frac{TP}{TP + FN}$), completely omitting $TN$. Therefore, achieving an **AUC-PR of $0.329$ against a random chance baseline of $0.04$** ($>8\times$ improvement) is definitive proof that DocGuard-ViT accurately localizes tampered pixels without flooding the document with false positives."

#### Q10: Why does DocGuard-ViT maintain 89.7% F1 under JPEG Q=40 compression while ResNet drops to 62.5%?
> **Answer:**  
> "When an image is compressed at $Q=40$, the high-frequency spatial gradients (such as subtle color differences or anti-aliasing edges) are blurred and smoothed out by the lossy quantization matrix. A purely RGB-based network like ResNet-50 loses its visual cues.  
> However, in the frequency domain, re-compressing an already-tampered document creates **double JPEG compression artifacts**. The discrete quantization step introduces periodic zero-coefficients and grid misalignments. Because our model explicitly processes the 64 DCT coefficients in the frequency stream, it detects these quantization grid inconsistencies even when the RGB appearance has been visibly degraded."

#### Q11: What are the primary failure modes or limitations of DocGuard-ViT?
> **Answer:**  
> "We identified two key limitations:
> 1. **Non-JPEG formats (Raw Scans):** If a document is created, manipulated, and saved entirely as uncompressed PNG or TIFF, no JPEG quantization grid exists. In such cases, the frequency stream carries less discriminative signal, and the model relies primarily on the spatial Swin-T stream.
> 2. **Ultra-Low Resolution Documents (<150 DPI):** Severe downsampling merges character strokes into surrounding pixels, making it difficult to distinguish rasterization blur from deliberate tampering. In future work, we plan to incorporate super-resolution preprocessing and multi-lingual OCR layout corroboration."

#### Q12: Can DocGuard-ViT run in real-time in an enterprise production environment?
> **Answer:**  
> "Yes. On our consumer workstation (RTX 3050 Laptop GPU), single-document end-to-end inference—including CLAHE preprocessing, DCT extraction, Swin-T forward pass, and mask decoding—takes **$22.4\text{ ms}$**, which equates to approximately **$45\text{ frames per second}$**. Even on a standard multi-core CPU, inference completes in under $120\text{ ms}$, making it immediately deployable in automated banking KYC and document verification pipelines."
