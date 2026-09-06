"""Document Forgery PyTorch Dataset and DataLoader Factory

Provides batching, dual-stream feature extraction, and forensic augmentations.
"""

from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .generator import DocumentForgeryGenerator, ForgeryType
from preprocessing.pipeline import PreprocessingPipeline


class DocumentForgeryDataset(Dataset):
    """PyTorch Dataset producing dual-stream inputs and multi-task supervision targets."""

    def __init__(
        self,
        length: int = 500,
        target_size: Tuple[int, int] = (512, 512),
        double_jpeg: bool = True,
        seed: Optional[int] = None,
    ):
        self.length = length
        self.target_size = target_size
        self.double_jpeg = double_jpeg
        self.generator = DocumentForgeryGenerator(target_size=target_size)
        self.preprocessor = PreprocessingPipeline(
            target_size=target_size, apply_rectify=False, apply_deskew=False, apply_clahe=True
        )
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        # Generate sample on the fly with reproducible or random parameters
        sample = self.generator.generate_sample(apply_double_jpeg=self.double_jpeg)

        # Preprocess through dual-stream pipeline
        processed = self.preprocessor.process_image(sample["image"])

        # Tensors:
        # rgb_tensor: (3, 512, 512)
        rgb_tensor = processed["rgb_tensor"].squeeze(0)
        # dct_tensor: (21, 512, 512)
        dct_tensor = processed["dct_tensor"].squeeze(0)

        # Mask tensor: (1, 512, 512)
        mask = sample["mask"]
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        is_forged = torch.tensor(sample["is_forged"], dtype=torch.float32)
        forgery_type = torch.tensor(sample["forgery_type"], dtype=torch.long)

        return {
            "rgb": rgb_tensor,
            "dct": dct_tensor,
            "mask": mask_tensor,
            "is_forged": is_forged,
            "forgery_type": forgery_type,
            "type_name": sample["forgery_type_name"],
        }


def create_dataloaders(
    train_size: int = 400,
    val_size: int = 100,
    batch_size: int = 8,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Factory helper to construct train and validation DataLoaders."""
    train_ds = DocumentForgeryDataset(length=train_size, seed=42)
    val_ds = DocumentForgeryDataset(length=val_size, seed=999)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader
