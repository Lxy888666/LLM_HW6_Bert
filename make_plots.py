from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from punctuation import LABELS


LABEL_NAMES = {
    "O": "O",
    "，": "comma",
    "。": "period",
    "？": "question",
    "！": "exclamation",
    "；": "semicolon",
    "：": "colon",
    "、": "enumeration",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create figures for the punctuation restoration report.")
    parser.add_argument("--training-results", type=Path, default=Path("outputs/bert-punctuation/training_results.json"))
    parser.add_argument("--evaluation-results", type=Path, default=Path("outputs/evaluation/evaluation_results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training = load(args.training_results)
    evaluation = load(args.evaluation_results)
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 180, "font.size": 10})

    history = training["history"]
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train loss")
    ax.plot(epochs, [row["validation_loss"] for row in history], marker="o", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted cross-entropy")
    ax.set_title("Training and validation loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "training_curve.png")
    plt.close(fig)

    per_class = evaluation["test_metrics"]["per_class"]
    labels = LABELS[1:]
    plot_labels = [LABEL_NAMES[label] for label in labels]
    f1_values = [per_class[label]["f1-score"] for label in labels]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bars = ax.bar(plot_labels, f1_values, color="#2f6f9f")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Punctuation label")
    ax.set_ylabel("F1")
    ax.set_title("Test F1 by punctuation class")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, f1_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "class_f1.png")
    plt.close(fig)

    throughput = evaluation["throughput_results"]
    batch_sizes = [row["batch_size"] for row in throughput]
    rates = [row["characters_per_second"] for row in throughput]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(batch_sizes, rates, marker="o", linewidth=2, color="#3a7d44")
    ax.set_xlabel("Inference batch size")
    ax.set_ylabel("Characters per second")
    ax.set_title("Batched sliding-window throughput")
    ax.set_xticks(batch_sizes)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "throughput.png")
    plt.close(fig)

    matrix = np.asarray(evaluation["test_metrics"]["confusion_matrix"], dtype=float)
    row_sums = np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    normalized = matrix / row_sums
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)), [LABEL_NAMES[label] for label in LABELS], rotation=35, ha="right")
    ax.set_yticks(range(len(LABELS)), [LABEL_NAMES[label] for label in LABELS])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Normalized test confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(args.output_dir / "confusion_matrix.png")
    plt.close(fig)
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
