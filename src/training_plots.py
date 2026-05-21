from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir


def plot_training_curves(input_csv: str | Path, output_png: str | Path) -> None:
    """Plot loss and accuracy curves from a saved training log CSV."""
    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"Training log is empty: {input_csv}")

    output_png = Path(output_png)
    ensure_dir(output_png.parent)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(df["epoch"], df["train_loss"], marker="o", label="train")
    axes[0].plot(df["epoch"], df["val_loss"], marker="o", label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and validation loss")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_acc"], marker="o", label="train")
    axes[1].plot(df["epoch"], df["val_acc"], marker="o", label="validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and validation accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves from a saved training CSV.")
    parser.add_argument("--input", required=True, help="Path to *_train_log.csv.")
    parser.add_argument("--output", required=True, help="Path to the PNG file to write.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_training_curves(args.input, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
