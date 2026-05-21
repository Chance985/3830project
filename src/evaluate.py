from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from .corruptions import CORRUPTIONS
from .datasets import make_eval_loader, parse_list
from .models import build_model, load_checkpoint
from .tta_methods import clone_model, evaluate_stream
from .utils import ensure_dir, get_device, load_config, seed_everything, write_rows_csv


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate source and TTA methods on corrupted CIFAR-10.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", default="results/tables/raw_results.csv")
    parser.add_argument("--methods", default="source,bn,entropy,gated")
    parser.add_argument("--corruptions", default=None)
    parser.add_argument("--severities", default=None)
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--tta-lr", type=float, default=None)
    parser.add_argument("--tta-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    seed_everything(seed)
    device = get_device(args.device)

    data_dir = args.data_dir or cfg["data_dir"]
    eval_cfg = cfg["eval"]
    tta_cfg = cfg["tta"]
    batch_size = args.batch_size or eval_cfg["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else eval_cfg["num_workers"]
    subset_size = args.subset_size if args.subset_size is not None else eval_cfg["subset_size"]
    corruptions = parse_list(args.corruptions or eval_cfg["corruptions"])
    severities = _parse_int_list(args.severities) if args.severities else list(eval_cfg["severities"])
    thresholds = _parse_float_list(args.thresholds) if args.thresholds else list(tta_cfg["thresholds"])
    methods = parse_list(args.methods)
    tta_lr = args.tta_lr if args.tta_lr is not None else tta_cfg["lr"]
    tta_steps = args.tta_steps if args.tta_steps is not None else tta_cfg["steps"]

    unknown = sorted(set(corruptions).difference(CORRUPTIONS))
    if unknown:
        raise ValueError(f"Unknown corruptions requested: {unknown}")

    base_model = build_model(cfg["model"]["name"], cfg["model"]["num_classes"])
    load_checkpoint(args.checkpoint, base_model, map_location=device)
    base_model.to(device)

    eval_jobs = []
    if args.include_clean:
        eval_jobs.append((None, 0))
    for corruption in corruptions:
        for severity in severities:
            eval_jobs.append((corruption, severity))

    rows = []
    for corruption, severity in tqdm(eval_jobs, desc="evaluation jobs"):
        loader = make_eval_loader(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            subset_size=subset_size,
            corruption=corruption,
            severity=severity if corruption is not None else None,
        )
        corruption_name = corruption or "clean"
        for method in methods:
            if method == "gated":
                method_thresholds = thresholds
            else:
                method_thresholds = [None]

            for threshold in method_thresholds:
                model = clone_model(base_model, device)
                result = evaluate_stream(
                    model=model,
                    loader=loader,
                    device=device,
                    method=method,
                    lr=tta_lr,
                    steps=tta_steps,
                    threshold=threshold,
                )
                rows.append(
                    {
                        "method": method,
                        "threshold": "" if threshold is None else threshold,
                        "corruption": corruption_name,
                        "severity": severity,
                        "accuracy": result.accuracy,
                        "error_rate": result.error_rate,
                        "nll": result.nll,
                        "entropy": result.entropy,
                        "ece": result.ece,
                        "runtime_sec": result.runtime_sec,
                        "selected_fraction": result.selected_fraction,
                        "num_samples": result.num_samples,
                        "subset_size": subset_size or "",
                        "tta_lr": tta_lr if method in {"entropy", "gated"} else "",
                        "tta_steps": tta_steps if method in {"entropy", "gated"} else "",
                    }
                )
                print(rows[-1])

    output = Path(args.output)
    ensure_dir(output.parent)
    write_rows_csv(output, rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
