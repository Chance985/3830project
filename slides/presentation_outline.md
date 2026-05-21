# Presentation Outline

## Slide 1: Title

Evaluating and Improving Test-Time Adaptation for Image Classification under Distribution Shift

## Slide 2: Problem

- Image classifiers trained on clean data can fail under corruptions.
- Deployment data may contain blur, noise, compression, or lighting changes.
- Test labels are usually unavailable.

## Slide 3: Research Question

Can confidence-gated test-time adaptation improve corrupted-image accuracy compared with:

- Source-only inference
- BatchNorm adaptation
- Standard entropy minimization

## Slide 4: Setup

- Dataset: CIFAR-10
- Model: CIFAR-style ResNet18
- Training: 5 epochs on clean CIFAR-10
- Evaluation: 2,000-image seeded test subset
- Shifts: 7 synthetic corruptions x 5 severity levels

## Slide 5: Methods

- Source-only: no test-time update
- BatchNorm adaptation: update running statistics only
- Entropy minimization: update BatchNorm affine parameters
- Confidence-gated TTA: entropy loss only for samples with confidence >= threshold

## Slide 6: Main Results

| Method | Avg. corrupted accuracy |
|---|---:|
| Source-only | 49.67% |
| BatchNorm adaptation | 61.50% |
| Entropy minimization | 62.28% |
| Gated, tau=0.5 | 62.30% |
| Gated, tau=0.7 | 62.25% |
| Gated, tau=0.9 | 62.01% |

Main takeaway: adaptation helps a lot; gating is only marginally different from entropy minimization in this run.

## Slide 7: Corruption Breakdown

Use `results/figures/accuracy_by_corruption.png`.

Talking point: the largest gains occur on severe blur, contrast, noise, and pixelation. JPEG is less damaging, so there is less room to improve.

## Slide 8: Severity Trend

Use `results/figures/accuracy_vs_severity.png`.

Talking point: all methods degrade with severity, but adapted methods keep a large gap over source-only.

## Slide 9: Threshold Ablation

Use `results/figures/threshold_ablation.png`.

- tau=0.5 selects 71.8% of samples and gives the best gated result.
- tau=0.9 selects 24.3% and loses accuracy.
- Stricter gating reduces updates but may remove useful adaptation signal.

## Slide 10: Limitations

- Synthetic corruptions, not CIFAR-10-C benchmark results.
- One trained model and one evaluation seed.
- Source model trained only 5 epochs.
- Runtime does not improve because the implementation still processes full batches.

## Slide 11: Conclusion

- Test-time adaptation substantially improves corrupted accuracy.
- Confidence gating is a simple and reproducible extension.
- In this experiment, gating matched entropy minimization but did not clearly outperform it.
- Next steps: CIFAR-10-C, stronger model, multiple seeds, better filtering rules.
