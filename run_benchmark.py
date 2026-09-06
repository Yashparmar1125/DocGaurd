"""Benchmark Comparison Runner

Executes evaluation comparing:
- Baseline 1: ELA + Random Forest
- Baseline 2: Plain ResNet-50 Classifier
- DocGuard: Proposed Hybrid CNN-Transformer Framework

Outputs the comparison table matching Section 23 of the Project Document.
"""

import argparse
import torch
from evaluation.benchmark import BenchmarkRunner
from models.docguard import DocGuardModel


def format_table(results: dict):
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC", "Mean IoU", "Mean Dice"]
    rows = []

    for model_name, m in results.items():
        row = [
            model_name,
            f"{m.get('accuracy', 0.0) * 100:.1f}%",
            f"{m.get('precision', 0.0) * 100:.1f}%",
            f"{m.get('recall', 0.0) * 100:.1f}%",
            f"{m.get('f1_score', 0.0):.3f}",
            f"{m.get('roc_auc', 0.0):.3f}",
            f"{m.get('pr_auc', 0.0):.3f}",
            f"{m.get('mean_iou', 0.0):.3f}" if m.get('mean_iou', 0.0) > 0 else "N/A",
            f"{m.get('mean_dice', 0.0):.3f}" if m.get('mean_dice', 0.0) > 0 else "N/A",
        ]
        rows.append(row)

    # Print markdown table
    col_widths = [max(len(str(item)) for item in col) for col in zip(headers, *rows)]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep_line = "-|-".join("-" * w for w in col_widths)

    print("\n" + "=" * len(header_line))
    print("DOCGUARD BENCHMARK EVALUATION RESULTS (Section 23)")
    print("=" * len(header_line))
    print(header_line)
    print(sep_line)
    for r in rows:
        print(" | ".join(val.ljust(w) for val, w in zip(r, col_widths)))
    print("=" * len(header_line) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run DocGuard Benchmark Evaluation")
    parser.add_argument("--test-samples", type=int, default=30, help="Number of held-out test samples")
    parser.add_argument("--train-samples", type=int, default=80, help="Number of baseline training samples")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmark on device: {device}")

    # Load or initialize DocGuard model
    model = DocGuardModel(pretrained_backbone=False)
    runner = BenchmarkRunner(num_test_samples=args.test_samples, device=device)

    results = runner.run_benchmark(docguard_model=model, train_baseline_samples=args.train_samples)
    format_table(results)


if __name__ == "__main__":
    main()
