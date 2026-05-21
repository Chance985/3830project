# Evaluating Test-Time Adaptation under Distribution Shift

This repository contains a PyTorch course project on CIFAR-10 test-time adaptation (TTA). The proposed variant is confidence-gated entropy minimization: update only on test samples whose maximum softmax confidence exceeds a threshold.

## Included Components

- CIFAR-style ResNet18 source training.
- Checkpoint saving for best and last model states.
- Training CSV logs and training-curve PNGs.
- Synthetic corruptions: Gaussian noise, Gaussian blur, motion blur, brightness, contrast, pixelation, and JPEG compression.
- Optional local CIFAR-10-C loading from `.npy` files.
- Source-only, BatchNorm adaptation, entropy minimization, and confidence-gated TTA.
- Seeded subset, full-test, and multi-seed evaluation modes.
- CSV-derived tables and plots for corruption, severity, threshold, and cost analysis.

## Setup

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The final run was produced with Python 3.9.7, PyTorch 2.8.0+cu128, and an NVIDIA GeForce RTX 4060 Laptop GPU. CPU execution works but is slower.

## Final Reproduction Commands

Train the 50-epoch source model:

```powershell
python -m src.train --epochs 50 --batch-size 256 --num-workers 0 --seed 3830 --run-name resnet18_cifar10_50ep
```

Run the final full-test synthetic corruption evaluation:

```powershell
python -m src.evaluate --checkpoint checkpoints\resnet18_cifar10_50ep_best.pt --dataset synthetic --methods source,bn,entropy,gated --thresholds 0.5,0.7,0.9 --batch-size 256 --num-workers 0 --seed 3830 --full-test --include-clean --output results\tables\raw_results.csv
```

Regenerate tables and figures from the saved raw CSV:

```powershell
python -m src.plot_results --input results\tables\raw_results.csv --tables-dir results\tables --figures-dir results\figures
```

The training command automatically writes the training curve. To regenerate it from the saved log:

```powershell
python -m src.training_plots --input logs\resnet18_cifar10_50ep_train_log.csv --output results\figures\resnet18_cifar10_50ep_training_curves.png
```

## Final Saved Results

The final numeric results use a 50-epoch CIFAR-style ResNet18, the full 10,000-image CIFAR-10 test set, seven synthetic corruptions, five severity levels, and seed 3830. CIFAR-10-C was not used, and multi-seed experiments were not run.

Clean source-only accuracy:

- Best clean accuracy during training: 93.78% at epoch 48.
- Final epoch clean accuracy: 93.66%.
- Clean source-only accuracy from final evaluation: 93.78%.

Average corrupted accuracy:

| Method | Accuracy | Gain vs source | Gain vs entropy |
|---|---:|---:|---:|
| Source-only | 59.05% | 0.00 pp | -20.02 pp |
| BatchNorm adaptation | 76.56% | 17.51 pp | -2.51 pp |
| Entropy minimization | 79.08% | 20.02 pp | 0.00 pp |
| Gated TTA, threshold 0.5 | 79.04% | 19.98 pp | -0.04 pp |
| Gated TTA, threshold 0.7 | 79.04% | 19.99 pp | -0.03 pp |
| Gated TTA, threshold 0.9 | 79.18% | 20.12 pp | 0.10 pp |

The evidence supports a conservative conclusion: TTA substantially improves accuracy under corruption, while confidence-gated TTA only marginally changes performance relative to standard entropy minimization.

## Output Locations

- Raw final results: `results/tables/raw_results.csv`
- Main comparison: `results/tables/main_comparison.csv`
- Cost summary: `results/tables/cost_summary.csv`
- Per-corruption table: `results/tables/by_corruption.csv`
- Per-severity table: `results/tables/by_severity.csv`
- Clean accuracy table: `results/tables/clean_accuracy.csv`
- Threshold ablation: `results/tables/threshold_ablation.csv`
- Corruption plot: `results/figures/accuracy_by_corruption.png`
- Severity plot: `results/figures/accuracy_vs_severity.png`
- Threshold plot: `results/figures/threshold_ablation.png`
- Training curve: `results/figures/resnet18_cifar10_50ep_training_curves.png`
- Report: `report/project_report.md`
- Presentation outline: `slides/presentation_outline.md`

The final 50-epoch checkpoints are saved locally as `checkpoints/resnet18_cifar10_50ep_best.pt` and `checkpoints/resnet18_cifar10_50ep_last.pt`. They are regenerable and are ignored for GitHub to avoid committing large binary files.

The earlier 5-epoch, 2,000-image subset artifacts are archived under `results/preliminary_5ep_subset/` and should be treated only as preliminary provenance.

## Optional Experiments

Full multi-seed retraining is implemented via `scripts/train_multiseed_50.sh`, and evaluation of those trained checkpoints is implemented via `scripts/run_multiseed_trained_eval.sh`. Evaluation-only multi-seed support is available through `scripts/run_multiseed_eval.sh`. These were not executed for the final reported numbers.

CIFAR-10-C is supported only from a local directory. The code does not download it automatically. Expected layout:

```text
path\to\CIFAR-10-C\
  labels.npy
  gaussian_noise.npy
  gaussian_blur.npy
  motion_blur.npy
  brightness.npy
  contrast.npy
  pixelate.npy
  jpeg_compression.npy
```

Example command:

```powershell
python -m src.evaluate --checkpoint checkpoints\resnet18_cifar10_50ep_best.pt --dataset cifar10c --cifar10c-dir path\to\CIFAR-10-C --batch-size 256 --num-workers 0 --subset-size 2000 --include-clean --output results\tables\raw_results_cifar10c.csv
```

## Verification

```powershell
python -m compileall src
python -m src.plot_results --input results\tables\raw_results.csv --tables-dir results\tables --figures-dir results\figures
```

All reported tables and plots are regenerated from saved CSV files. Do not report CIFAR-10-C or multi-seed results unless those commands have actually been run.

## Academic Integrity

The implementation is original project code. External ideas, datasets, and architectures are cited in `report/references.bib` and in the report. The reported numbers are generated from saved CSV files in this repository; no experimental result is fabricated.
