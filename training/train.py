"""DocGuard Multi-Task Training Script

CLI Entrypoint for training both the CNN-Transformer Hybrid and Vision Transformer (DocGuard-ViT)
on real benchmark datasets (e.g., DocTamper, RealText-V2) or synthetic procedural data.

Usage:
    # 1. Train Vision Transformer on benchmark data:
    python -m training.train --architecture transformer --dataset benchmark --data-dir data/benchmark --epochs 10 --batch-size 4

    # 2. Train Hybrid model on synthetic data:
    python -m training.train --architecture hybrid --dataset synthetic --epochs 10 --batch-size 4
"""

import argparse
from pathlib import Path
import torch

from training.config import TrainingConfig
from training.trainer import DocGuardTrainer
from models.docguard import DocGuardModel
from models.docguard_transformer import DocGuardTransformerModel
from data.dataset import create_dataloaders
from data.benchmark_dataset import create_benchmark_dataloaders


def main():
    parser = argparse.ArgumentParser(description="DocGuard Multi-Task Model Training CLI")
    parser.add_argument(
        "--architecture",
        choices=["hybrid", "transformer"],
        default="transformer",
        help="Model architecture: 'hybrid' (ResNet-50 + U-Net) or 'transformer' (Swin-T + SegFormer)",
    )
    parser.add_argument(
        "--dataset",
        choices=["doctamper", "benchmark", "synthetic"],
        default="doctamper",
        help="Dataset type: 'doctamper' (LMDB from data/dataset), 'benchmark' (images/masks), or 'synthetic'",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/dataset",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of training samples for quick runs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per step (default 4 fits within 6GB VRAM)",
    )
    parser.add_argument(
        "--lr-backbone",
        type=float,
        default=1e-5,
        help="Learning rate for pretrained backbone (Swin-T or ResNet)",
    )
    parser.add_argument(
        "--lr-new",
        type=float,
        default=1e-4,
        help="Learning rate for newly initialized layers (decoder, fusion, heads)",
    )
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=2,
        help="Epochs to keep pretrained backbone frozen before fine-tuning",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Initialize backbone with ImageNet pretrained weights (default True)",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="Output checkpoint filename (default: docguard_<arch>_best.pt)",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f"  DocGuard Forensic Training")
    print(f"  Architecture:  {args.architecture.upper()}")
    print(f"  Dataset:       {args.dataset} ({args.data_dir if args.dataset == 'benchmark' else 'procedural'})")
    print(f"  Device:        {device.upper()}")
    print(f"  Epochs:        {args.epochs} | Batch Size: {args.batch_size}")
    print(f"=======================================================\n")

    # 1. Initialize Model
    if args.architecture == "transformer":
        model = DocGuardTransformerModel(
            pretrained=args.pretrained,
            num_classes=5,
        )
    else:
        model = DocGuardModel(
            pretrained=args.pretrained,
            num_classes=5,
        )

    # 2. Prepare DataLoaders
    if args.dataset == "doctamper":
        from data.doctamper_dataset import create_doctamper_dataloaders
        train_lmdb = Path(args.data_dir) / "DocTamperV1-SCD"
        val_lmdb = Path(args.data_dir) / "DocTamperV1-FCD"
        if not train_lmdb.exists():
            train_lmdb = Path(args.data_dir) / "DocTamperV1-TrainingSet"

        print(f"[*] Loading DocTamper LMDB from: {args.data_dir}")
        train_loader, val_loader = create_doctamper_dataloaders(
            train_lmdb=train_lmdb,
            val_lmdb=val_lmdb,
            max_train_samples=args.max_samples,
            max_val_samples=min(args.max_samples // 4, 500) if args.max_samples else None,
            batch_size=args.batch_size,
        )
    elif args.dataset == "benchmark":
        data_path = Path(args.data_dir)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Benchmark folder '{args.data_dir}' not found. Run python -m data.download_benchmark first."
            )
        train_loader, val_loader = create_benchmark_dataloaders(
            data_dir=data_path,
            batch_size=args.batch_size,
        )
    else:
        train_loader, val_loader = create_dataloaders(
            train_size=100,
            val_size=20,
            batch_size=args.batch_size,
        )

    print(f"[*] Training batches: {len(train_loader)} | Validation batches: {len(val_loader)}")

    # 3. Setup Config and Trainer
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_backbone=args.lr_backbone,
        lr_new_layers=args.lr_new,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
    )

    trainer = DocGuardTrainer(
        model=model,
        config=config,
        device=device,
        checkpoint_dir="checkpoints",
    )

    ckpt_name = args.checkpoint_name or f"docguard_{args.architecture}_best.pt"
    best_score = -1.0

    # 4. Training Loop
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        train_metrics = trainer.train_epoch(train_loader, epoch=epoch)
        print(
            f"Train | Loss: {train_metrics['total']:.4f} "
            f"(Cls: {train_metrics['cls']:.4f}, Seg: {train_metrics['seg']:.4f}, Type: {train_metrics['type']:.4f})"
        )

        val_metrics = trainer.validate(val_loader)
        print(
            f"Val   | Loss: {val_metrics['val_loss']:.4f} | "
            f"IoU: {val_metrics['val_iou']:.4f} | Acc: {val_metrics['val_acc']:.2%}"
        )

        # Composite score: Validation IoU (60%) + Classification Accuracy (40%)
        score = 0.6 * val_metrics["val_iou"] + 0.4 * val_metrics["val_acc"]
        if score > best_score:
            best_score = score
            saved_path = trainer.save_checkpoint(filename=ckpt_name)
            print(f"  [+] Saved new best checkpoint -> {saved_path} (Score: {score:.4f})")

    print(f"\n[+] Training Complete! Best model saved to checkpoints/{ckpt_name}\n")


if __name__ == "__main__":
    main()
