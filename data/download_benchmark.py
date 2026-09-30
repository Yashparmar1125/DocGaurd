"""Benchmark Dataset Downloader and Setup Utility

Prepares real-world or procedural benchmark document forgery data in:
    data/benchmark/
        images/
        masks/

Usage:
    # 1. Download sample from HuggingFace RealText-V2:
    python -m data.download_benchmark --source realtext --num-samples 25

    # 2. Or generate an offline high-resolution benchmark corpus:
    python -m data.download_benchmark --source synthetic --num-samples 100

    # 3. Verify existing dataset folder (e.g. DocTamper after manual download):
    python -m data.download_benchmark --verify
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from data.generator import DocumentForgeryGenerator, ForgeryType


def setup_synthetic_benchmark(dest_dir: Path, num_samples: int = 100):
    """Generates an offline benchmark dataset of authentic and forged documents with masks."""
    images_dir = dest_dir / "images"
    masks_dir = dest_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Generating {num_samples} benchmark document samples in {dest_dir}...")
    generator = DocumentForgeryGenerator(target_size=(512, 512))

    forgery_types = [
        ForgeryType.AUTHENTIC,
        ForgeryType.COPY_MOVE,
        ForgeryType.SPLICING,
        ForgeryType.TEXT_TAMPER,
        ForgeryType.ERASURE,
    ]

    for i in tqdm(range(num_samples), desc="Generating samples"):
        ft = forgery_types[i % len(forgery_types)]
        sample = generator.generate_sample(forgery_type=ft, apply_double_jpeg=True)

        img_filename = f"doc_{i:05d}_{sample['forgery_type_name']}.png"
        mask_filename = f"doc_{i:05d}_{sample['forgery_type_name']}_mask.png"

        bgr = cv2.cvtColor(sample["image"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(images_dir / img_filename), bgr)

        mask_uint8 = (sample["mask"] * 255).astype(np.uint8)
        cv2.imwrite(str(masks_dir / mask_filename), mask_uint8)

    print(f"[+] Done! Generated {num_samples} samples.")
    print(f"    Images: {images_dir}")
    print(f"    Masks:  {masks_dir}")


def setup_realtext_sample(dest_dir: Path, num_samples: int = 25):
    """Downloads sample document images from HuggingFace vankey/RealText-V2."""
    images_dir = dest_dir / "images"
    masks_dir = dest_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        files = api.list_repo_files(repo_id="vankey/RealText-V2", repo_type="dataset")
        img_files = [f for f in files if f.startswith("test/image/") and f.endswith(".jpg")]
        if not img_files:
            print("[-] No image files found in RealText-V2 test split. Falling back to synthetic.")
            setup_synthetic_benchmark(dest_dir, num_samples)
            return

        to_download = img_files[:num_samples]
        print(f"[*] Downloading {len(to_download)} RealText-V2 samples from Hugging Face...")
        for f in tqdm(to_download, desc="Downloading RealText"):
            local_p = hf_hub_download(repo_id="vankey/RealText-V2", filename=f, repo_type="dataset")
            target_img = images_dir / Path(f).name
            # Copy or save image
            img = cv2.imread(local_p)
            if img is not None:
                cv2.imwrite(str(target_img), img)
                # Corresponding mask path in repo
                mask_rel = f.replace("test/image/", "test/mask/").replace(".jpg", ".png")
                try:
                    local_mask_p = hf_hub_download(repo_id="vankey/RealText-V2", filename=mask_rel, repo_type="dataset")
                    mask = cv2.imread(local_mask_p, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        cv2.imwrite(str(masks_dir / target_img.with_suffix(".png").name), mask)
                except Exception:
                    # If mask is missing, create blank authentic mask
                    h, w = img.shape[:2]
                    cv2.imwrite(str(masks_dir / target_img.with_suffix(".png").name), np.zeros((h, w), dtype=np.uint8))

        print(f"[+] Download complete: {len(to_download)} samples saved to {dest_dir}.")
    except Exception as e:
        print(f"[-] HuggingFace download encountered: {e}. Generating procedural benchmark instead.")
        setup_synthetic_benchmark(dest_dir, num_samples)


def verify_benchmark(dest_dir: Path):
    """Verifies image and mask pairs in data_dir."""
    from data.benchmark_dataset import find_image_mask_pairs
    images, masks = find_image_mask_pairs(dest_dir)
    print(f"\n--- Benchmark Dataset Verification: {dest_dir} ---")
    print(f"Total document images discovered: {len(images)}")
    paired = sum(1 for m in masks if m is not None)
    print(f"Images with corresponding ground-truth mask: {paired}")
    print(f"Unmasked images (treated as authentic): {len(images) - paired}")
    if len(images) > 0:
        print(f"Sample pair:")
        print(f"  Image: {images[0]}")
        print(f"  Mask:  {masks[0]}")
    print("--------------------------------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="DocGuard Benchmark Dataset Downloader & Setup")
    parser.add_argument(
        "--source",
        choices=["realtext", "synthetic"],
        default="synthetic",
        help="Source of benchmark data ('realtext' from HuggingFace, or 'synthetic')",
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        default="data/benchmark",
        help="Destination directory for benchmark data",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to prepare",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Inspect and verify the destination directory",
    )

    args = parser.parse_args()
    dest = Path(args.dest_dir)

    if args.verify:
        verify_benchmark(dest)
        return

    if args.source == "realtext":
        setup_realtext_sample(dest, args.num_samples)
    else:
        setup_synthetic_benchmark(dest, args.num_samples)

    verify_benchmark(dest)


if __name__ == "__main__":
    main()
