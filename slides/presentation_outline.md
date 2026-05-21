# Presentation Outline

## Slide 1: Title

Evaluating Test-Time Adaptation for CIFAR-10 under Distribution Shift

Visual: one clean CIFAR-10 image beside corrupted versions.

## Slide 2: Problem and Motivation

- Image classifiers are trained on clean data.
- Deployment images may contain blur, noise, compression, or lighting shifts.
- Target labels are unavailable at test time.
- Source-only accuracy can drop sharply under corruption.

Visual: compact pipeline from clean training data to corrupted test stream.

## Slide 3: Goal

Evaluate whether test-time adaptation improves corrupted-image accuracy, and whether confidence gating improves standard entropy minimization.

Comparison methods:

- Source-only inference
- BatchNorm adaptation
- Entropy minimization
- Confidence-gated entropy minimization

## Slide 4: Experimental Setup

- Dataset: CIFAR-10
- Model: CIFAR-style ResNet18
- Training: 50 epochs, seed 3830
- Evaluation: full 10,000-image CIFAR-10 test set
- Shifts: 7 synthetic corruptions x 5 severity levels
- CIFAR-10-C and multi-seed runs were not executed

Visual: small table of corruption types and severities.

## Slide 5: Method Overview

- Source-only: no adaptation
- BatchNorm adaptation: update running statistics only
- Entropy minimization: update BatchNorm affine parameters
- Confidence-gated TTA: entropy loss only for samples with confidence >= threshold

Visual: show entropy minimization with a confidence gate before the loss.

## Slide 6: Main Results

Use `results/tables/main_comparison.csv`.

| Method | Avg. acc. | Gain vs source | Gain vs entropy |
|---|---:|---:|---:|
| Source-only | 59.05% | 0.00 pp | -20.02 pp |
| BatchNorm adaptation | 76.56% | 17.51 pp | -2.51 pp |
| Entropy minimization | 79.08% | 20.02 pp | 0.00 pp |
| Gated, tau=0.5 | 79.04% | 19.98 pp | -0.04 pp |
| Gated, tau=0.7 | 79.04% | 19.99 pp | -0.03 pp |
| Gated, tau=0.9 | 79.18% | 20.12 pp | 0.10 pp |

Main takeaway: TTA helps substantially; confidence gating is marginal and threshold-sensitive.

## Slide 7: Accuracy by Corruption

Use `results/figures/accuracy_by_corruption.png`.

Talking point: adapted methods are usually above source-only, but the size of the gain depends on corruption type.

## Slide 8: Accuracy by Severity

Use `results/figures/accuracy_vs_severity.png`.

Talking point: all methods degrade as severity increases, but TTA keeps a clear gap over source-only.

## Slide 9: Threshold Ablation

Use `results/figures/threshold_ablation.png` and `results/tables/threshold_ablation.csv`.

- tau=0.5 selects 97.5% of samples and averages 79.04%.
- tau=0.7 selects 90.3% of samples and averages 79.04%.
- tau=0.9 selects 80.4% of samples and averages 79.18%.

Main takeaway: the best gated result is only 0.10 percentage points above entropy minimization, so it should not be overstated.

## Slide 10: Cost Analysis

Use `results/tables/cost_summary.csv`.

Suggested table columns:

- Method
- Average corrupted accuracy
- Runtime per stream
- Runtime per batch
- Selected sample fraction

Talking point: gating changes the loss mask but does not reduce runtime in this implementation because the full batch is still processed.

## Slide 11: Limitations

- Synthetic corruptions were used instead of CIFAR-10-C.
- The final result is single-seed.
- The scope is CIFAR-10 and ResNet18.
- Confidence thresholds are sensitive to the source model and corruption.

## Slide 12: Conclusion

- Distribution shift hurts source-only image classification.
- Test-time adaptation substantially improves corrupted accuracy.
- Confidence-gated TTA is simple and reproducible, but the observed gain over entropy minimization is marginal.
- Stronger evidence requires CIFAR-10-C, multi-seed experiments, stronger backbones, and adaptive threshold selection.
