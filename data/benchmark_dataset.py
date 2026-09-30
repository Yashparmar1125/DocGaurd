"""Benchmark Document Tampering Dataset Loader

Supports loading real-world document tampering benchmarks (e.g., DocTamper, RealText-V2,
or custom image/mask directories) into DocGuard's dual-stream RGB + DCT representation.
"""

from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing.pipeline import PreprocessingPipeline


class BenchmarkDocumentDataset(Dataset):
    """Dataset loader for benchmark tampering datasets with image and mask pairs.

    Expected directory structure:
        data_dir/
            images/   (*.png, *.jpg, *.jpeg)
            masks/    (*.png, *.jpg, *.bmp)  # Grayscale: 255 = tampered, 0 = authentic
    Or matching names between images/ and masks/. If a mask is missing, it is treated as authentic.
    """

    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: Optional[List[Optional[Path]]] = None,
        target_size: Tuple[int, int] = (512, 512),
        apply_clahe: bool = True,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths or [None] * len(image_paths)
        self.target_size = target_size
        self.preprocessor = PreprocessingPipeline(
            target_size=target_size,
            apply_rectify=False,
            apply_deskew=False,
            apply_clahe=apply_clahe,
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_p = self.image_paths[idx]
        mask_p = self.mask_paths[idx]

        # Load RGB image
        bgr = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        if bgr is None:
            # Fallback blank document if file read fails
            bgr = np.ones((self.target_size[0], self.target_size[1], 3), dtype=np.uint8) * 255
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Preprocess through dual-stream pipeline
        processed = self.preprocessor.process_image(rgb)
        rgb_tensor = processed["rgb_tensor"].squeeze(0)  # (3, 512, 512)
        dct_tensor = processed["dct_tensor"].squeeze(0)  # (21, 512, 512)

        # Load Ground-Truth Mask
        if mask_p is not None and Path(mask_p).exists():
            mask_raw = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            if mask_raw is not None:
                mask_resized = cv2.resize(mask_raw, self.target_size, interpolation=cv2.INTER_NEAREST)
                mask = (mask_resized > 127).astype(np.float32)
            else:
                mask = np.zeros(self.target_size, dtype=np.float32)
        else:
            mask = np.zeros(self.target_size, dtype=np.float32)

        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)  # (1, 512, 512)

        is_tampered = float(mask.sum() > 0)
        # Infer basic type or default to 1 (Text Tamper / Copy-Move) when forged, 0 when authentic
        forgery_type = 1 if is_tampered > 0 else 0

        return {
            "rgb": rgb_tensor,
            "dct": dct_tensor,
            "mask": mask_tensor,
            "is_forged": torch.tensor(is_tampered, dtype=torch.float32),
            "forgery_type": torch.tensor(forgery_type, dtype=torch.long),
            "type_name": "forged" if is_tampered > 0 else "authentic",
            "file_name": img_p.name,
        }


def find_image_mask_pairs(
    data_dir: Path,
) -> Tuple[List[Path], List[Optional[Path]]]:
    """Discovers paired images and masks in a standard benchmark directory."""
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"

    if not images_dir.exists():
        # Maybe data_dir itself contains images directly
        images_dir = data_dir

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    image_paths: List[Path] = []
    for ext in extensions:
        image_paths.extend(images_dir.glob(ext))
        image_paths.extend(images_dir.glob(ext.upper()))

    image_paths = sorted(list(set(image_paths)))
    mask_paths: List[Optional[Path]] = []

    for img_p in image_paths:
        stem = img_p.stem
        # Try matching mask by name in masks_dir
        candidate_masks = [
            masks_dir / f"{stem}.png",
            masks_dir / f"{stem}_mask.png",
            masks_dir / f"{stem}_gt.png",
            masks_dir / f"{stem}.jpg",
            masks_dir / f"{stem}_gt.jpg",
            masks_dir / img_p.name,
        ]
        matched_mask = None
        for cand in candidate_masks:
            if cand.exists():
                matched_mask = cand
                break
        mask_paths.append(matched_mask)

    return image_paths, mask_paths


def create_benchmark_dataloaders(
    data_dir: str or Path,
    split_ratio: float = 0.8,
    batch_size: int = 4,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Constructs train and validation DataLoaders from a benchmark dataset folder."""
    data_dir = Path(data_dir)
    images, masks = find_image_mask_pairs(data_dir)

    if len(images) == 0:
        raise ValueError(f"No images found in {data_dir} (checked {data_dir / 'images'})")

    np.random.seed(seed)
    indices = np.random.permutation(len(images))
    split_idx = int(len(images) * split_ratio)

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_images = [images[i] for i in train_indices]
    train_masks = [masks[i] for i in train_indices]

    val_images = [images[i] for i in val_indices]
    val_masks = [masks[i] for i in val_indices]

    train_ds = BenchmarkDocumentDataset(train_images, train_masks)
    val_ds = BenchmarkDocumentDataset(val_images, val_masks)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader
