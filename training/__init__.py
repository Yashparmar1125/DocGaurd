"""DocGuard Training Package

Contains:
- MultiTaskDocGuardLoss: Combined Dice + BCE + Label-smoothed Cross-Entropy loss
- DocGuardTrainer: Multi-task training loop with differential learning rate schedules
- TrainingConfig: Default hyperparameters and configuration dataclass
"""

from .losses import MultiTaskDocGuardLoss, DiceLoss
from .config import TrainingConfig
from .trainer import DocGuardTrainer

__all__ = [
    "MultiTaskDocGuardLoss",
    "DiceLoss",
    "TrainingConfig",
    "DocGuardTrainer",
]
