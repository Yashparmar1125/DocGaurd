"""DocGuard Multi-Task Model Training Script

Usage:
    python train.py --epochs 10 --batch-size 8 --train-samples 300 --val-samples 60
"""

import argparse
from pathlib import Path
import torch

from models.docguard import DocGuardModel
from data.dataset import create_dataloaders
from training.trainer import DocGuardTrainer
from training.config import TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Train DocGuard Multi-Task Forgery Detector")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--train-samples", type=int, default=200, help="Number of synthetic train samples")
    parser.add_argument("--val-samples", type=int, default=50, help="Number of synthetic val samples")
    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="Backbone differential learning rate")
    parser.add_argument("--lr-new", type=float, default=1e-4, help="New layers differential learning rate")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save model weights")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== DocGuard Multi-Task Training ===")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_backbone=args.lr_backbone,
        lr_new_layers=args.lr_new,
    )

    print("\n[1/3] Generating synthetic datasets and dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_size=args.train_samples,
        val_size=args.val_samples,
        batch_size=args.batch_size,
    )

    print("\n[2/3] Initializing DocGuard Dual-Stream Model...")
    model = DocGuardModel(pretrained_backbone=True)
    trainer = DocGuardTrainer(model=model, config=config, device=device, checkpoint_dir=args.output_dir)

    print("\n[3/3] Commencing Multi-Task Training Loop...")
    best_score = 0.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.validate(val_loader)

        score = val_metrics["val_iou"] + val_metrics["val_acc"]
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_metrics['total']:.4f} (Cls: {train_metrics['cls']:.3f}, Seg: {train_metrics['seg']:.3f}) | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val Acc: {val_metrics['val_acc'] * 100:.1f}% | "
            f"Val IoU: {val_metrics['val_iou']:.4f}"
        )

        if score > best_score:
            best_score = score
            save_path = trainer.save_checkpoint("docguard_best.pt")
            print(f"  -> Saved new best checkpoint to {save_path}")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
