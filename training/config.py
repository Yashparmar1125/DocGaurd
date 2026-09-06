"""Training Configuration Dataclass"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrainingConfig:
    # Model & Input Dimensions
    target_size: Tuple[int, int] = (512, 512)
    num_forgery_types: int = 5
    embed_dim: int = 512

    # Training Parameters
    epochs: int = 15
    batch_size: int = 8
    num_workers: int = 0

    # Differential Learning Rates (Section 11 & 21)
    lr_backbone: float = 1e-5     # Pretrained ResNet-50 fine-tuning
    lr_new_layers: float = 1e-4   # DCT stream, Fusion, Decoder, Heads
    weight_decay: float = 0.01

    # Multi-Task Loss Weights (Section 20)
    lambda_cls: float = 1.0
    lambda_type: float = 0.5
    lambda_seg: float = 1.0
    alpha_dice: float = 0.5
    label_smoothing: float = 0.1

    # Regularization & Early Stopping
    early_stopping_patience: int = 5
    freeze_backbone_epochs: int = 2  # Warm start: train heads/decoder first before backbone
