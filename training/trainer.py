"""DocGuard Multi-Task Model Trainer

Executes training with differential learning rates, early stopping on joint IoU+F1,
and warm-start backbone freezing.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .losses import MultiTaskDocGuardLoss
from .config import TrainingConfig
from models.docguard import DocGuardModel


class DocGuardTrainer:
    """Orchestrates multi-task training for DocGuard."""

    def __init__(
        self,
        model: DocGuardModel,
        config: TrainingConfig = None,
        device: Optional[str] = None,
        checkpoint_dir: str = "checkpoints",
    ):
        self.config = config or TrainingConfig()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.loss_fn = MultiTaskDocGuardLoss(
            lambda_cls=self.config.lambda_cls,
            lambda_type=self.config.lambda_type,
            lambda_seg=self.config.lambda_seg,
            alpha_dice=self.config.alpha_dice,
            label_smoothing=self.config.label_smoothing,
        )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = self._build_differential_optimizer()
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.config.epochs, eta_min=1e-6
        )

    def _build_differential_optimizer(self) -> AdamW:
        """Separates pretrained RGB backbone parameters from new layer parameters."""
        backbone_params = list(self.model.rgb_stream.parameters())
        backbone_ids = list(map(id, backbone_params))
        new_params = [p for p in self.model.parameters() if id(p) not in backbone_ids]

        param_groups = [
            {"params": backbone_params, "lr": self.config.lr_backbone},
            {"params": new_params, "lr": self.config.lr_new_layers},
        ]
        return AdamW(param_groups, weight_decay=self.config.weight_decay)

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Trains for one epoch."""
        self.model.train()

        # Warm start: freeze RGB backbone for the first few epochs if configured
        if epoch < self.config.freeze_backbone_epochs:
            for p in self.model.rgb_stream.parameters():
                p.requires_grad = False
        else:
            for p in self.model.rgb_stream.parameters():
                p.requires_grad = True

        running_losses = {"total": 0.0, "cls": 0.0, "type": 0.0, "seg": 0.0}
        total_batches = len(dataloader)

        for batch in dataloader:
            rgb = batch["rgb"].to(self.device)
            dct = batch["dct"].to(self.device)
            targets = {
                "mask": batch["mask"].to(self.device),
                "is_forged": batch["is_forged"].to(self.device),
                "forgery_type": batch["forgery_type"].to(self.device),
            }

            self.optimizer.zero_grad()
            outputs = self.model(rgb, dct)
            loss_dict = self.loss_fn(outputs, targets)

            loss_dict["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            running_losses["total"] += loss_dict["loss_total"].item()
            running_losses["cls"] += loss_dict["loss_cls"].item()
            running_losses["type"] += loss_dict["loss_type"].item()
            running_losses["seg"] += loss_dict["loss_seg"].item()

        self.scheduler.step()

        return {k: v / max(total_batches, 1) for k, v in running_losses.items()}

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluates model on validation set computing loss, IoU, and F1."""
        self.model.eval()
        running_loss = 0.0
        intersections = 0.0
        unions = 0.0
        correct_cls = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                rgb = batch["rgb"].to(self.device)
                dct = batch["dct"].to(self.device)
                targets = {
                    "mask": batch["mask"].to(self.device),
                    "is_forged": batch["is_forged"].to(self.device),
                    "forgery_type": batch["forgery_type"].to(self.device),
                }

                outputs = self.model(rgb, dct)
                loss_dict = self.loss_fn(outputs, targets)
                running_loss += loss_dict["loss_total"].item()

                # Segmentation IoU
                pred_binary_mask = (outputs["mask_prob"] > 0.5).float()
                gt_mask = targets["mask"]
                inter = (pred_binary_mask * gt_mask).sum().item()
                union = ((pred_binary_mask + gt_mask) > 0).float().sum().item()
                intersections += inter
                unions += union

                # Classification accuracy
                pred_cls = (outputs["forgery_prob"] > 0.5).float().squeeze()
                correct_cls += (pred_cls == targets["is_forged"]).sum().item()
                total_samples += targets["is_forged"].size(0)

        mean_iou = intersections / max(unions, 1e-5)
        cls_acc = correct_cls / max(total_samples, 1)

        return {
            "val_loss": running_loss / max(len(dataloader), 1),
            "val_iou": mean_iou,
            "val_acc": cls_acc,
        }

    def save_checkpoint(self, filename: str = "docguard_best.pt"):
        """Saves weights and model state."""
        save_path = self.checkpoint_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
        }, save_path)
        return save_path
