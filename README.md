# Evaluating and Improving Test-Time Adaptation for Image Classification under Distribution Shift

This repository contains a reproducible PyTorch course project on test-time adaptation (TTA) for CIFAR-10 image classification under synthetic image corruptions. The proposed method is Confidence-Gated Entropy Minimization: adapt only on test samples whose maximum softmax confidence exceeds a threshold.

## What Is Included

- CIFAR-10 source training with a CIFAR-style ResNet18.
- Synthetic corruptions: Gaussian noise, Gaussian blur, motion blur, brightness, contrast, pixelation, and JPEG compression.
- TTA methods:
  - Source-only inference.
  - BatchNorm adaptation.
  - Entropy minimization on BatchNorm affine parameters.
  - Confidence-gated entropy minimization with thresholds 0.5, 0.7, and 0.9.
- CSV result logs, summary tables, figures, a report draft, and a presentation outline.

## Setup

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The experiments were run with Python 3.9.7, PyTorch 2.8.0+cu128, and an NVIDIA GeForce RTX 4060 Laptop GPU. CPU execution is supported but slower.

## Reproduce the Completed Run

Train the source model:

```powershell
python -m src.train --epochs 5 --batch-size 256 --num-workers 0 --run-name resnet18_cifar10_5ep
```

Evaluate all methods on the seeded 2,000-image CIFAR-10 test subset:

```powershell
python -m src.evaluate --checkpoint checkpoints\resnet18_cifar10_5ep_best.pt --batch-size 256 --num-workers 0 --subset-size 2000 --include-clean --output results\tables\raw_results.csv
```

Generate tables and figures:

```powershell
python -m src.plot_results --input results\tables\raw_results.csv --tables-dir results\tables --figures-dir results\figures
```

For a fast smoke test:

```powershell
python -m src.evaluate --checkpoint checkpoints\resnet18_cifar10_5ep_best.pt --methods source --corruptions gaussian_noise --severities 1 --subset-size 64 --output results\tables\smoke.csv
```

## Current Results

The included run trained ResNet18 for 5 epochs on the full CIFAR-10 training split. The best full clean test accuracy during training was 70.54%. The final TTA evaluation used a fixed 2,000-image test subset for all methods and corruptions.

Key files:

- Raw results: `results/tables/raw_results.csv`
- Main comparison: `results/tables/main_comparison.csv`
- Corruption breakdown: `results/tables/by_corruption.csv`
- Severity breakdown: `results/tables/by_severity.csv`
- Threshold ablation: `results/tables/threshold_ablation.csv`
- Figures: `results/figures/`
- Report draft: `report/project_report.md`
- Presentation outline: `slides/presentation_outline.md`

## Notes on Academic Integrity

The implementation is original project code. External ideas, datasets, and architectures are cited in `report/references.bib` and in the report draft. The reported numbers are generated from saved CSV files in this repository; no experimental result is fabricated.
