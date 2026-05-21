# Evaluating and Improving Test-Time Adaptation for Image Classification under Distribution Shift

## Abstract

Image classifiers trained on clean data often lose accuracy when test images contain common corruptions. This project evaluates whether test-time adaptation can recover some of that loss without using test labels. A ResNet18 was trained on CIFAR-10 and evaluated on seven synthetic corruption types at five severity levels. I compared source-only inference, BatchNorm adaptation, entropy minimization, and a simple Confidence-Gated Entropy Minimization method that only updates on samples whose maximum softmax confidence exceeds a threshold. On a seeded 2,000-image CIFAR-10 test subset, source-only inference averaged 49.67% accuracy across corruptions. BatchNorm adaptation improved this to 61.50%, entropy minimization to 62.28%, and the best confidence-gated setting to 62.30%. The gate reduced the fraction of samples used for adaptation, but its accuracy gain over ungated entropy minimization was very small in this setting.

## 1. Introduction and Problem Definition

Distribution shift is a practical problem for image classification: a model trained on clean images may see noisy, blurred, compressed, or lighting-shifted images at deployment time. Retraining with labels from the target distribution is often unavailable, so test-time adaptation tries to adjust the model during evaluation using only unlabeled test batches.

The core question in this project is: can confidence-gated test-time adaptation improve robustness under corruptions compared with source-only inference, BatchNorm adaptation, and standard entropy minimization? The motivation is that entropy minimization can update the model using uncertain or wrong predictions. A confidence gate is a small change: if the model is not confident on a test sample, that sample is excluded from the entropy loss.

## 2. Related Work and Background

CIFAR-10 is a standard 10-class image classification dataset [@krizhevsky2009cifar10; @uciCifar10]. Common corruption benchmarks such as CIFAR-10-C evaluate whether classifiers remain reliable under realistic image perturbations [@hendrycks2019robustness; @hendrycks2019cifar10czenodo]. This project uses synthetic corruptions generated from CIFAR-10 test images instead of downloading CIFAR-10-C, so the results should be read as a controlled course experiment rather than a benchmark claim.

The source model is a CIFAR-adapted ResNet18, based on residual networks [@he2016resnet]. BatchNorm adaptation relies on updating normalization statistics at test time [@ioffe2015batchnorm]. Entropy minimization encourages confident predictions on unlabeled data [@grandvalet2004entropy], and TENT applies this idea to fully test-time adaptation by updating BatchNorm affine parameters [@wang2021tent]. Confidence scores have also been used for detecting unreliable predictions and for selective prediction [@hendrycks2017baseline; @geifman2017selective]. The confidence gate here is a lightweight sample-selection rule inspired by that idea; it is not claimed as a new general TTA algorithm.

## 3. Methodology

The source-only baseline trains a ResNet18 on clean CIFAR-10 and evaluates it without any updates. BatchNorm adaptation places the model in training mode during evaluation so BatchNorm running statistics update from each test stream, while all weights remain frozen.

Entropy minimization adapts only BatchNorm affine parameters. For a test batch \(x\), the model predicts class probabilities \(p_\theta(y|x)\), and the update minimizes

\[
L_{\text{ent}}(x) = -\sum_c p_\theta(c|x)\log p_\theta(c|x).
\]

Confidence-Gated Entropy Minimization uses the same objective but filters samples before computing the loss. For each image \(x_i\), confidence is the maximum softmax probability:

\[
s_i = \max_c p_\theta(c|x_i).
\]

Only samples with \(s_i \ge \tau\) contribute to the entropy loss. I tested \(\tau \in \{0.5, 0.7, 0.9\}\). Labels are never used during adaptation.

## 4. Implementation Details

The code is written in Python and PyTorch. CIFAR-10 is downloaded through `torchvision`. The ResNet18 first convolution was changed to a 3x3 stride-1 convolution and max pooling was removed, which is a common CIFAR-style adjustment for 32x32 images. The model was trained for 5 epochs with SGD, momentum 0.9, weight decay 0.0005, cosine learning-rate decay, and seed 3830.

Because CIFAR-10-C download and processing was not used, I implemented synthetic corruptions directly on CIFAR-10 test images: Gaussian noise, Gaussian blur, horizontal motion blur, brightness shift, contrast shift, pixelation, and JPEG compression. Each corruption has five severities. Stochastic corruptions are seeded per image so all methods see the same corrupted inputs.

Evaluation used batch size 256, one adaptation step per batch, TTA learning rate 0.001, and a fixed 2,000-image test subset. The full source training log and raw evaluation outputs are saved under `logs/` and `results/tables/`.

## 5. Experiments

The main metric is accuracy. I also report error rate, negative log-likelihood, entropy, expected calibration error, runtime per corruption stream, and the fraction of samples selected for adaptation. The main comparison averages over 7 corruptions and 5 severities.

Clean subset accuracy was 71.20% for source-only inference. On clean data, adaptation slightly reduced accuracy: BatchNorm adaptation reached 70.30%, entropy minimization 70.70%, and gated variants about 70.60-70.70%. This suggests adaptation should be used when shift is expected, not blindly on already-clean data.

### Table 1. Main corrupted-distribution comparison

| Method | Avg. accuracy (%) | Error rate (%) | Runtime / stream (s) | Selected for entropy update (%) |
|---|---:|---:|---:|---:|
| Source-only | 49.67 | 50.33 | 0.84 | 0.0 |
| BatchNorm adaptation | 61.50 | 38.50 | 0.83 | 100.0 |
| Entropy minimization | 62.28 | 37.72 | 1.62 | 100.0 |
| Gated entropy, tau=0.5 | 62.30 | 37.70 | 1.63 | 71.8 |
| Gated entropy, tau=0.7 | 62.25 | 37.75 | 1.63 | 47.1 |
| Gated entropy, tau=0.9 | 62.01 | 37.99 | 1.63 | 24.3 |

### Table 2. Accuracy by corruption type

| Corruption | Source | BatchNorm | Entropy | Gated tau=0.5 |
|---|---:|---:|---:|---:|
| Brightness | 54.08 | 60.65 | 61.70 | 61.74 |
| Contrast | 46.05 | 68.20 | 68.70 | 68.63 |
| Gaussian blur | 48.49 | 61.53 | 61.98 | 62.01 |
| Gaussian noise | 50.52 | 61.81 | 62.90 | 62.97 |
| JPEG | 61.92 | 62.50 | 63.24 | 63.29 |
| Motion blur | 40.43 | 60.00 | 60.77 | 60.76 |
| Pixelate | 46.20 | 55.81 | 56.65 | 56.68 |

### Table 3. Accuracy by corruption severity

| Severity | Source | BatchNorm | Entropy | Gated tau=0.5 |
|---:|---:|---:|---:|---:|
| 1 | 65.09 | 68.38 | 68.81 | 68.78 |
| 2 | 58.27 | 66.42 | 67.09 | 67.10 |
| 3 | 48.96 | 62.79 | 63.60 | 63.61 |
| 4 | 41.66 | 58.14 | 59.16 | 59.21 |
| 5 | 34.36 | 51.77 | 52.73 | 52.79 |

### Threshold ablation

The threshold ablation shows the expected selection tradeoff: increasing the threshold reduces how many samples update the model. However, stricter thresholds did not improve average accuracy in this run. The best average accuracy was tau=0.5 at 62.30%, only 0.02 percentage points above ungated entropy minimization.

| Threshold | Avg. accuracy (%) | Selected samples (%) |
|---:|---:|---:|
| 0.5 | 62.30 | 71.8 |
| 0.7 | 62.25 | 47.1 |
| 0.9 | 62.01 | 24.3 |

Figures generated from the same CSV files are saved as `results/figures/accuracy_by_corruption.png`, `results/figures/accuracy_vs_severity.png`, and `results/figures/threshold_ablation.png`.

## 6. Results and Analysis

The main result is that test-time adaptation clearly helps under the synthetic corruptions. Source-only accuracy falls to 49.67%, while BatchNorm adaptation recovers almost 12 percentage points. Entropy minimization adds another 0.78 percentage points over BatchNorm adaptation.

Confidence gating did not provide a large accuracy gain over standard entropy minimization. The tau=0.5 gate was slightly best on average, but the difference from ungated entropy minimization is too small to treat as a strong improvement. The stricter tau=0.9 setting selected only 24.3% of samples and performed worse, which suggests that skipping too many samples can remove useful adaptation signal.

The gate is still informative as a diagnostic. It shows that many adaptation updates can be removed with little change in accuracy. In this implementation, runtime did not decrease because the full batch is still forwarded and backpropagated; the mask only changes which samples contribute to the loss. A more optimized implementation would need to actually skip computation for filtered samples before claiming a runtime benefit.

The corruption breakdown shows that adaptation helps most on blur, contrast, noise, and pixelation, where source-only accuracy drops sharply. JPEG corruption is less damaging, so there is less room for improvement. Accuracy decreases with severity for every method, but the gap between source-only and adapted methods grows at high severity.

## 7. Conclusion

This project implemented and evaluated source-only inference, BatchNorm adaptation, entropy minimization, and confidence-gated entropy minimization for CIFAR-10 image classification under synthetic distribution shift. The experiments support two conclusions. First, simple test-time adaptation can substantially improve robustness under common corruptions. Second, confidence gating is not automatically better than ungated entropy minimization: in this run it matched entropy minimization closely at low thresholds but did not produce a meaningful average improvement.

Future work should evaluate CIFAR-10-C directly, use stronger source models, run multiple random seeds, and test more principled filtering rules such as entropy filtering with diversity or anti-forgetting constraints.

## References

- [@krizhevsky2009cifar10] A. Krizhevsky. *Learning Multiple Layers of Features from Tiny Images*. Technical report, University of Toronto, 2009.
- [@uciCifar10] UC Irvine Machine Learning Repository. *CIFAR-10*. DOI: 10.24432/C5889J.
- [@hendrycks2019robustness] D. Hendrycks and T. Dietterich. *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*. ICLR, 2019.
- [@hendrycks2019cifar10czenodo] D. Hendrycks and T. Dietterich. *CIFAR-10-C and CIFAR-10-P*. Zenodo, 2019. DOI: 10.5281/zenodo.2535967.
- [@he2016resnet] K. He, X. Zhang, S. Ren, and J. Sun. *Deep Residual Learning for Image Recognition*. CVPR, 2016.
- [@ioffe2015batchnorm] S. Ioffe and C. Szegedy. *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*. ICML, 2015.
- [@grandvalet2004entropy] Y. Grandvalet and Y. Bengio. *Semi-supervised Learning by Entropy Minimization*. NeurIPS, 2004.
- [@wang2021tent] D. Wang, E. Shelhamer, S. Liu, B. Olshausen, and T. Darrell. *Tent: Fully Test-Time Adaptation by Entropy Minimization*. ICLR, 2021.
- [@hendrycks2017baseline] D. Hendrycks and K. Gimpel. *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks*. ICLR, 2017.
- [@geifman2017selective] Y. Geifman and R. El-Yaniv. *Selective Classification for Deep Neural Networks*. NeurIPS, 2017.

BibTeX entries are provided in `report/references.bib`.

## AI Usage Statement

AI tools were used to help organize the project structure, generate and debug Python/PyTorch code, design experiment scripts, summarize relevant literature, and improve the clarity of the report. All experimental results, analysis, and conclusions were reviewed against the saved logs, CSV files, and generated figures. No experimental result was invented.

## Contribution Statement

The submitted repository contains the complete implementation, experiment scripts, saved results, report draft, and presentation outline for this course project. Human review is still recommended before submission, especially for formatting, course-specific rubric requirements, and whether to run additional seeds or CIFAR-10-C experiments.
