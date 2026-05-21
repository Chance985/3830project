from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir


METHOD_ORDER = {
    "source": 0,
    "bn": 1,
    "entropy": 2,
    "gated@0.5": 3,
    "gated@0.7": 4,
    "gated@0.9": 5,
}


def method_label(row: pd.Series) -> str:
    if row["method"] == "gated":
        return f"gated@{float(row['threshold']):.1f}"
    return row["method"]


def method_sort_key(label: str) -> tuple[int, str]:
    return (METHOD_ORDER.get(label, 100), label)


def sort_by_method(
    df: pd.DataFrame,
    method_col: str = "method_label",
    leading_cols: list[str] | None = None,
    trailing_cols: list[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    out["_method_order"] = out[method_col].map(lambda value: method_sort_key(str(value))[0])
    sort_cols = [*(leading_cols or []), "_method_order", method_col, *(trailing_cols or [])]
    out = out.sort_values(sort_cols).drop(columns=["_method_order"])
    return out


def normalize_raw(df: pd.DataFrame, assume_batch_size: int | None) -> pd.DataFrame:
    out = df.copy()
    if "seed" not in out.columns:
        out["seed"] = 3830
    if "dataset" not in out.columns:
        out["dataset"] = "synthetic"
    if "batch_size" not in out.columns:
        out["batch_size"] = assume_batch_size if assume_batch_size is not None else pd.NA
    if "num_batches" not in out.columns:
        if assume_batch_size is not None and "num_samples" in out.columns:
            out["num_batches"] = out["num_samples"].apply(lambda n: max(1, math.ceil(float(n) / assume_batch_size)))
        else:
            out["num_batches"] = pd.NA
    if "runtime_per_batch_sec" not in out.columns:
        out["runtime_per_batch_sec"] = out.apply(
            lambda row: row["runtime_sec"] / row["num_batches"]
            if pd.notna(row["num_batches"]) and row["num_batches"] > 0
            else pd.NA,
            axis=1,
        )
    out["method_label"] = out.apply(method_label, axis=1)
    return out


def aggregate_seed_means(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    per_seed = (
        df.groupby(["seed", *group_cols], as_index=False)
        .agg({column: "mean" for column in value_cols})
    )
    aggregations = {}
    for column in value_cols:
        aggregations[f"{column}_mean"] = (column, "mean")
        aggregations[f"{column}_std"] = (column, "std")
    summary = per_seed.groupby(group_cols, as_index=False).agg(
        **aggregations,
        n_seeds=("seed", "nunique"),
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate project tables and plots from raw CSV results.")
    parser.add_argument("--input", default="results/tables/raw_results.csv")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument(
        "--assume-batch-size",
        type=int,
        default=None,
        help="Used only for older raw CSVs that do not contain num_batches/runtime_per_batch_sec.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables_dir = ensure_dir(args.tables_dir)
    figures_dir = ensure_dir(args.figures_dir)
    df = normalize_raw(pd.read_csv(args.input), args.assume_batch_size)
    corrupted = df[df["corruption"] != "clean"].copy()

    main = aggregate_seed_means(
        corrupted,
        ["method_label"],
        ["accuracy", "error_rate", "runtime_sec", "runtime_per_batch_sec", "selected_fraction"],
    )
    main = main.rename(
        columns={
            "accuracy_mean": "avg_accuracy",
            "accuracy_std": "std_accuracy",
            "error_rate_mean": "avg_error_rate",
            "error_rate_std": "std_error_rate",
            "runtime_sec_mean": "avg_runtime_sec",
            "runtime_sec_std": "std_runtime_sec",
            "runtime_per_batch_sec_mean": "avg_runtime_per_batch_sec",
            "runtime_per_batch_sec_std": "std_runtime_per_batch_sec",
            "selected_fraction_mean": "selected_fraction",
            "selected_fraction_std": "std_selected_fraction",
        }
    )
    accuracy_lookup = dict(zip(main["method_label"], main["avg_accuracy"]))
    source_acc = accuracy_lookup.get("source")
    entropy_acc = accuracy_lookup.get("entropy")
    main["improvement_over_source"] = (
        main["avg_accuracy"] - source_acc if source_acc is not None else pd.NA
    )
    main["improvement_over_entropy"] = (
        main["avg_accuracy"] - entropy_acc if entropy_acc is not None else pd.NA
    )
    main = sort_by_method(main)
    main.to_csv(tables_dir / "main_comparison.csv", index=False)

    cost = main[
        [
            "method_label",
            "avg_accuracy",
            "std_accuracy",
            "avg_runtime_sec",
            "avg_runtime_per_batch_sec",
            "selected_fraction",
            "n_seeds",
        ]
    ].copy()
    cost.to_csv(tables_dir / "cost_summary.csv", index=False)

    by_corruption = aggregate_seed_means(
        corrupted,
        ["method_label", "corruption"],
        ["accuracy", "error_rate"],
    )
    by_corruption = by_corruption.rename(
        columns={
            "accuracy_mean": "accuracy",
            "accuracy_std": "std_accuracy",
            "error_rate_mean": "error_rate",
            "error_rate_std": "std_error_rate",
        }
    )
    by_corruption = sort_by_method(by_corruption, leading_cols=["corruption"])
    by_corruption.to_csv(tables_dir / "by_corruption.csv", index=False)

    by_severity = aggregate_seed_means(
        corrupted,
        ["method_label", "severity"],
        ["accuracy", "error_rate"],
    )
    by_severity = by_severity.rename(
        columns={
            "accuracy_mean": "accuracy",
            "accuracy_std": "std_accuracy",
            "error_rate_mean": "error_rate",
            "error_rate_std": "std_error_rate",
        }
    )
    by_severity = sort_by_method(by_severity, trailing_cols=["severity"])
    by_severity.to_csv(tables_dir / "by_severity.csv", index=False)

    threshold = corrupted[corrupted["method"] == "gated"].copy()
    if not threshold.empty:
        threshold_summary = aggregate_seed_means(
            threshold,
            ["threshold"],
            ["accuracy", "error_rate", "runtime_sec", "runtime_per_batch_sec", "selected_fraction"],
        )
        threshold_summary = threshold_summary.rename(
            columns={
                "accuracy_mean": "avg_accuracy",
                "accuracy_std": "std_accuracy",
                "error_rate_mean": "avg_error_rate",
                "error_rate_std": "std_error_rate",
                "runtime_sec_mean": "avg_runtime_sec",
                "runtime_sec_std": "std_runtime_sec",
                "runtime_per_batch_sec_mean": "avg_runtime_per_batch_sec",
                "runtime_per_batch_sec_std": "std_runtime_per_batch_sec",
                "selected_fraction_mean": "selected_fraction",
                "selected_fraction_std": "std_selected_fraction",
            }
        )
        threshold_summary = threshold_summary.sort_values("threshold")
        threshold_summary.to_csv(tables_dir / "threshold_ablation.csv", index=False)

    by_corruption_for_plot = pd.read_csv(tables_dir / "by_corruption.csv")
    by_severity_for_plot = pd.read_csv(tables_dir / "by_severity.csv")

    pivot = by_corruption_for_plot.pivot(index="corruption", columns="method_label", values="accuracy")
    ordered_columns = sorted(pivot.columns, key=method_sort_key)
    pivot = pivot[ordered_columns]
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Corruption type")
    ax.set_title("Accuracy by corruption type")
    ax.legend(title="Method", fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "accuracy_by_corruption.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, group in by_severity_for_plot.groupby("method_label", sort=False):
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
        clean_summary = aggregate_seed_means(
            clean,
            ["method_label"],
            ["accuracy", "error_rate", "nll", "ece"],
        )
        clean_summary = clean_summary.rename(
            columns={
                "accuracy_mean": "accuracy",
                "accuracy_std": "std_accuracy",
                "error_rate_mean": "error_rate",
                "error_rate_std": "std_error_rate",
                "nll_mean": "nll",
                "nll_std": "std_nll",
                "ece_mean": "ece",
                "ece_std": "std_ece",
            }
        )
        clean_summary = sort_by_method(clean_summary)
        clean_summary.to_csv(tables_dir / "clean_accuracy.csv", index=False)

    print(f"wrote tables to {tables_dir}")
    print(f"wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
