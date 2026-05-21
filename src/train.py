from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from .datasets import make_eval_loader, make_train_loader
from .models import build_model, save_checkpoint
from .training_plots import plot_training_curves
from .utils import accuracy_from_logits, ensure_dir, get_device, load_config, seed_everything, write_rows_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 source model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default="resnet18_cifar10")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--curves-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    seed_everything(seed)

    data_dir = args.data_dir or cfg["data_dir"]
    checkpoint_dir = ensure_dir(args.checkpoint_dir or cfg["checkpoint_dir"])
    log_dir = ensure_dir(args.log_dir or cfg.get("log_dir", "logs"))
    curves_dir = ensure_dir(args.curves_dir or cfg.get("figures_dir", "results/figures"))
    train_cfg = cfg["train"]
    epochs = args.epochs or train_cfg["epochs"]
    batch_size = args.batch_size or train_cfg["batch_size"]
    lr = args.lr or train_cfg["lr"]
    weight_decay = args.weight_decay if args.weight_decay is not None else train_cfg["weight_decay"]
    momentum = args.momentum if args.momentum is not None else train_cfg["momentum"]
    num_workers = args.num_workers if args.num_workers is not None else train_cfg["num_workers"]
    subset_size = args.subset_size if args.subset_size is not None else train_cfg["subset_size"]
    device = get_device(args.device)

    train_loader = make_train_loader(data_dir, batch_size, num_workers, seed, subset_size)
    val_loader = make_eval_loader(data_dir, batch_size, num_workers, seed, subset_size=None)

    model = build_model(cfg["model"]["name"], cfg["model"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_acc = 0.0
    rows = []
    run_start = time.perf_counter()
    best_path = checkpoint_dir / f"{args.run_name}_best.pt"
    last_path = checkpoint_dir / f"{args.run_name}_last.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_count = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for inputs, targets in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(inputs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size_actual = targets.size(0)
            train_loss += loss.item() * batch_size_actual
            train_correct += logits.argmax(dim=1).eq(targets).sum().item()
            train_count += batch_size_actual
            pbar.set_postfix(loss=f"{train_loss / train_count:.3f}", acc=f"{train_correct / train_count:.3f}")

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_acc_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(inputs)
                loss = criterion(logits, targets)
                n = targets.size(0)
                val_loss += loss.item() * n
                val_acc_sum += accuracy_from_logits(logits, targets) * n
                val_count += n

        val_acc = val_acc_sum / val_count
        row = {
            "epoch": epoch,
            "train_loss": train_loss / train_count,
            "train_acc": train_correct / train_count,
            "val_loss": val_loss / val_count,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
            "elapsed_sec": time.perf_counter() - run_start,
            "subset_size": subset_size or "",
        }
        rows.append(row)
        print(row)

        save_checkpoint(
            str(last_path),
            model,
            epoch=epoch,
            best_acc=max(best_acc, val_acc),
            extra={"config": cfg, "run_name": args.run_name},
        )
        if val_acc >= best_acc:
            best_acc = val_acc
            save_checkpoint(
                str(best_path),
                model,
                epoch=epoch,
                best_acc=best_acc,
                extra={"config": cfg, "run_name": args.run_name},
            )

    log_path = log_dir / f"{args.run_name}_train_log.csv"
    curve_path = curves_dir / f"{args.run_name}_training_curves.png"
    write_rows_csv(log_path, rows)
    plot_training_curves(log_path, curve_path)
    print(f"best checkpoint: {best_path}")
    print(f"best clean accuracy: {best_acc:.4f}")
    print(f"training log: {log_path}")
    print(f"training curves: {curve_path}")


if __name__ == "__main__":
    main()
