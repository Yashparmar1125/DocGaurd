"""DocTamper LMDB Dataset Loader for DocGuard

Reads official DocTamper LMDB environments directly from disk into DocGuard's
dual-stream RGB + DCT pipeline with zero unpacking required.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import cv2
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing.pipeline import PreprocessingPipeline


class DocTamperLMDBDataset(Dataset):
    """PyTorch Dataset reading directly from DocTamper LMDB database."""

    def __init__(
        self,
        lmdb_path: str or Path,
        max_samples: Optional[int] = None,
        target_size: Tuple[int, int] = (512, 512),
        apply_clahe: bool = True,
    ):
        self.lmdb_path = str(lmdb_path)
        self.target_size = target_size
        self.preprocessor = PreprocessingPipeline(
            target_size=target_size,
            apply_rectify=False,
            apply_deskew=False,
            apply_clahe=apply_clahe,
        )

        # Open environment briefly to read total number of samples
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            num_samples_raw = txn.get(b"num-samples")
            if num_samples_raw is not None:
                self.num_samples = int(num_samples_raw.decode("utf-8"))
            else:
                # Fallback: estimate from total database entries divided by 2
                self.num_samples = env.stat()["entries"] // 2
        env.close()

        if max_samples is not None and max_samples < self.num_samples:
            self.num_samples = max_samples

        self.env = None
        self.txn = None

    def _init_db(self):
        """Worker-safe delayed LMDB environment initialization."""
        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=128,
        )
        self.txn = self.env.begin(write=False)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.env is None:
            self._init_db()

        img_key = f"image-{idx:09d}".encode("utf-8")
        label_key = f"label-{idx:09d}".encode("utf-8")

        img_bytes = self.txn.get(img_key)
        label_bytes = self.txn.get(label_key)

        if img_bytes is None:
            raise KeyError(f"Image key {img_key.decode()} not found in {self.lmdb_path}")

        # Decode image
        img_buf = np.frombuffer(img_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
        if bgr is None:
            bgr = np.ones((self.target_size[0], self.target_size[1], 3), dtype=np.uint8) * 255
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Preprocess through dual-stream pipeline
        processed = self.preprocessor.process_image(rgb)
        rgb_tensor = processed["rgb_tensor"].squeeze(0)  # (3, 512, 512)
        dct_tensor = processed["dct_tensor"].squeeze(0)  # (21, 512, 512)

        # Decode ground-truth mask
        if label_bytes is not None:
            mask_buf = np.frombuffer(label_bytes, dtype=np.uint8)
            mask_raw = cv2.imdecode(mask_buf, cv2.IMREAD_GRAYSCALE)
            if mask_raw is not None:
                if mask_raw.shape != self.target_size:
                    mask_raw = cv2.resize(mask_raw, self.target_size, interpolation=cv2.INTER_NEAREST)
                mask = (mask_raw > 127).astype(np.float32)
            else:
                mask = np.zeros(self.target_size, dtype=np.float32)
        else:
            mask = np.zeros(self.target_size, dtype=np.float32)

        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)  # (1, 512, 512)
        is_tampered = float(mask.sum() > 0)
        forgery_type = 1 if is_tampered > 0 else 0

        return {
            "rgb": rgb_tensor,
            "dct": dct_tensor,
            "mask": mask_tensor,
            "is_forged": torch.tensor(is_tampered, dtype=torch.float32),
            "forgery_type": torch.tensor(forgery_type, dtype=torch.long),
            "type_name": "forged" if is_tampered > 0 else "authentic",
            "file_name": f"doctamper_{idx:09d}",
        }


def create_doctamper_dataloaders(
    train_lmdb: str or Path = "data/dataset/DocTamperV1-SCD",
    val_lmdb: str or Path = "data/dataset/DocTamperV1-FCD",
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    batch_size: int = 4,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Constructs train and validation DataLoaders directly from DocTamper LMDB directories."""
    train_ds = DocTamperLMDBDataset(train_lmdb, max_samples=max_train_samples)
    val_ds = DocTamperLMDBDataset(val_lmdb, max_samples=max_val_samples)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader
