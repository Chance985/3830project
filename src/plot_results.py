from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir


def method_label(row: pd.Series) -> str:
    if row["method"] == "gated":
        return f"gated@{float(row['threshold']):.1f}"
    return row["method"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate project tables and plots from raw CSV results.")
    parser.add_argument("--input", default="results/tables/raw_results.csv")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument("--figures-dir", default="results/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables_dir = ensure_dir(args.tables_dir)
    figures_dir = ensure_dir(args.figures_dir)
    df = pd.read_csv(args.input)
    df["method_label"] = df.apply(method_label, axis=1)
    corrupted = df[df["corruption"] != "clean"].copy()

    main = (
        corrupted.groupby("method_label", as_index=False)
        .agg(
            avg_accuracy=("accuracy", "mean"),
            avg_error_rate=("error_rate", "mean"),
            avg_runtime_sec=("runtime_sec", "mean"),
            selected_fraction=("selected_fraction", "mean"),
        )
        .sort_values("avg_accuracy", ascending=False)
    )
    main.to_csv(tables_dir / "main_comparison.csv", index=False)

    by_corruption = (
        corrupted.groupby(["method_label", "corruption"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), error_rate=("error_rate", "mean"))
        .sort_values(["corruption", "method_label"])
    )
    by_corruption.to_csv(tables_dir / "by_corruption.csv", index=False)

    by_severity = (
        corrupted.groupby(["method_label", "severity"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), error_rate=("error_rate", "mean"))
        .sort_values(["method_label", "severity"])
    )
    by_severity.to_csv(tables_dir / "by_severity.csv", index=False)

    threshold = corrupted[corrupted["method"] == "gated"].copy()
    if not threshold.empty:
        threshold_summary = (
            threshold.groupby("threshold", as_index=False)
            .agg(
                avg_accuracy=("accuracy", "mean"),
                avg_error_rate=("error_rate", "mean"),
                selected_fraction=("selected_fraction", "mean"),
                avg_runtime_sec=("runtime_sec", "mean"),
            )
            .sort_values("threshold")
        )
        threshold_summary.to_csv(tables_dir / "threshold_ablation.csv", index=False)

    pivot = by_corruption.pivot(index="corruption", columns="method_label", values="accuracy")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Corruption type")
    ax.set_title("Accuracy by synthetic corruption type")
    ax.legend(title="Method", fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "accuracy_by_corruption.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, group in by_severity.groupby("method_label"):
        plt.plot(group["severity"], group["accuracy"], marker="o", label=name)
    plt.xlabel("Corruption severity")
    plt.ylabel("Average accuracy")
    plt.title("Accuracy versus corruption severity")
    plt.xticks(sorted(corrupted["severity"].unique()))
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "accuracy_vs_severity.png", dpi=200)
    plt.close()

    if not threshold.empty:
        threshold_summary = pd.read_csv(tables_dir / "threshold_ablation.csv")
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax1.plot(threshold_summary["threshold"], threshold_summary["avg_accuracy"], marker="o", color="#1f77b4")
        ax1.set_xlabel("Confidence threshold")
        ax1.set_ylabel("Average corrupted accuracy", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax2 = ax1.twinx()
        ax2.plot(
            threshold_summary["threshold"],
            threshold_summary["selected_fraction"],
            marker="s",
            color="#d62728",
        )
        ax2.set_ylabel("Selected sample fraction", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        plt.title("Confidence-gating ablation")
        fig.tight_layout()
        plt.savefig(figures_dir / "threshold_ablation.png", dpi=200)
        plt.close()

    if "clean" in set(df["corruption"]):
        clean = df[df["corruption"] == "clean"].copy()
        clean[["method_label", "accuracy", "error_rate", "nll", "ece"]].to_csv(
            tables_dir / "clean_accuracy.csv", index=False
        )

    print(f"wrote tables to {tables_dir}")
    print(f"wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
