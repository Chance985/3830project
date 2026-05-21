# Evaluating Test-Time Adaptation for Image Classification under Distribution Shift

## Abstract

Image classifiers trained on clean data can lose substantial accuracy when test images contain noise, blur, compression artifacts, or lighting changes. This project evaluates whether test-time adaptation (TTA) can recover accuracy under this kind of unlabeled distribution shift. A CIFAR-style ResNet18 was trained on CIFAR-10 for 50 epochs and evaluated on the full 10,000-image CIFAR-10 test set with seven synthetic corruption types at five severity levels. Source-only inference averaged 59.05% corrupted accuracy. BatchNorm adaptation improved this to 76.56%, entropy minimization to 79.08%, and the best confidence-gated threshold to 79.18%. The main conclusion is conservative: TTA helps substantially, while confidence gating only gives a marginal, threshold-sensitive change relative to standard entropy minimization.

## 1. Introduction

Distribution shift is a practical problem for image classification. A model trained on clean images may later encounter noisy, blurred, compressed, or overexposed images, and target labels are usually unavailable at deployment time. Test-time adaptation addresses this setting by updating the model during evaluation using only unlabeled test batches.

This project asks whether confidence-gated TTA improves corrupted-image accuracy compared with source-only inference, BatchNorm adaptation, and standard entropy minimization. The motivation is that entropy minimization can update on uncertain or incorrect predictions, so filtering low-confidence samples may reduce harmful updates.

## 2. Background

CIFAR-10 is a standard 10-class image classification dataset [@krizhevsky2009cifar10; @uciCifar10]. CIFAR-10-C is a common benchmark for robustness under corruptions [@hendrycks2019robustness; @hendrycks2019cifar10czenodo]. The executed final results in this report use synthetic corruptions generated from CIFAR-10 test images, not CIFAR-10-C.

The source model is a CIFAR-adapted ResNet18 [@he2016resnet]. BatchNorm adaptation updates normalization statistics at test time [@ioffe2015batchnorm]. Entropy minimization encourages confident predictions on unlabeled data [@grandvalet2004entropy], and TENT adapts BatchNorm affine parameters with this objective [@wang2021tent]. Confidence gating is a simple sample-selection rule motivated by confidence-based reliability estimates [@hendrycks2017baseline; @geifman2017selective].

## 3. Methods

The source-only baseline evaluates the trained ResNet18 without updates. BatchNorm adaptation places the model in training mode during evaluation so BatchNorm running statistics update from each test stream, while all weights remain frozen.

Entropy minimization adapts only BatchNorm affine parameters. For a test batch \(x\), the model predicts probabilities \(p_\theta(y|x)\), and the update minimizes:

\[
L_{\text{ent}}(x) = -\sum_c p_\theta(c|x)\log p_\theta(c|x).
\]

Confidence-Gated Entropy Minimization uses the same objective but filters samples before computing the loss. For each image \(x_i\), confidence is:

\[
s_i = \max_c p_\theta(c|x_i).
\]

Only samples with \(s_i >= \tau\) contribute to the entropy loss. I evaluated \(\tau \in \{0.5, 0.7, 0.9\}\). Labels are never used during adaptation.

## 4. Experimental Setup

The code is written in Python and PyTorch. CIFAR-10 is loaded through `torchvision`. The ResNet18 first convolution was changed to a 3x3 stride-1 convolution and max pooling was removed for 32x32 images.

The final source model was trained for 50 epochs with SGD, momentum 0.9, weight decay 0.0005, cosine learning-rate decay, batch size 256, and seed 3830. The best full clean test accuracy during training was 93.78% at epoch 48; final epoch clean accuracy was 93.66%. The final evaluation used the best checkpoint, batch size 256, seed 3830, one adaptation step per batch, and TTA learning rate 0.001.

The corruption evaluation used the full 10,000-image CIFAR-10 test set. The synthetic corruptions were Gaussian noise, Gaussian blur, motion blur, brightness, contrast, pixelation, and JPEG compression, each at severities 1 through 5. All methods used the same images, corruption types, severities, and random seed. Stochastic synthetic corruptions are seeded per image.

## 5. Results

The main metric is average corrupted accuracy over 7 corruption types and 5 severity levels. Results are from one training seed and one evaluation seed, so no standard deviation is reported.

### Table 1. Main corrupted-distribution comparison

| Method | Avg. acc. (%) | Gain vs source (pp) | Gain vs entropy (pp) | Selected (%) | Time / batch (s) |
|---|---:|---:|---:|---:|---:|
| Source-only | 59.05 | 0.00 | -20.02 | 0.0 | 0.134 |
| BatchNorm adaptation | 76.56 | 17.51 | -2.51 | 100.0 | 0.132 |
| Entropy minimization | 79.08 | 20.02 | 0.00 | 100.0 | 0.223 |
| Gated entropy, tau=0.5 | 79.04 | 19.98 | -0.04 | 97.5 | 0.227 |
| Gated entropy, tau=0.7 | 79.04 | 19.99 | -0.03 | 90.3 | 0.228 |
| Gated entropy, tau=0.9 | 79.18 | 20.12 | 0.10 | 80.4 | 0.226 |

Clean source-only accuracy was 93.78%. Source-only corrupted accuracy averaged 59.05%, a 34.72 percentage-point drop from clean accuracy. BatchNorm adaptation and entropy minimization recover much of this loss under synthetic corruptions. On clean images, adaptation slightly reduced accuracy, so these methods should be used when distribution shift is expected rather than blindly applied to clean streams.

Per-corruption and per-severity tables are saved as `results/tables/by_corruption.csv` and `results/tables/by_severity.csv`. The corresponding plots are `results/figures/accuracy_by_corruption.png` and `results/figures/accuracy_vs_severity.png`. The confidence-threshold ablation is saved as `results/tables/threshold_ablation.csv` and `results/figures/threshold_ablation.png`. All tables and figures were regenerated from `results/tables/raw_results.csv`.

## 6. Analysis

The strongest result is that TTA helps substantially under distribution shift. BatchNorm adaptation improves average corrupted accuracy by 17.51 percentage points over source-only inference. Entropy minimization improves by 20.02 percentage points over source-only and by 2.51 percentage points over BatchNorm adaptation.

Confidence gating does not clearly outperform standard entropy minimization. The best gated setting, tau=0.9, is only 0.10 percentage points above entropy minimization, while tau=0.5 and tau=0.7 are slightly below it. This is too small to claim a meaningful advantage. The threshold also matters: higher thresholds select fewer samples, and the best threshold differs across corruption conditions.

The gate is still useful diagnostically. It shows that entropy updates can be restricted to 80-98% of samples while maintaining almost the same average accuracy. However, this implementation does not reduce runtime because the full batch is still forwarded and backpropagated; the mask only changes which samples contribute to the loss.

## 7. Limitations and Future Work

The final results use synthetic corruptions instead of CIFAR-10-C. CIFAR-10-C support is implemented for local `.npy` files, but it was not executed for the reported numbers.

The evaluation is single-seed. Multi-seed evaluation scripts are implemented, but no multi-seed result is reported here. The conclusions are also limited to CIFAR-10, a ResNet18 source model, and batch-wise adaptation of BatchNorm parameters.

Confidence gating is threshold-sensitive and may depend on source-model quality. A weaker or poorly calibrated model may produce different confidence distributions, changing which samples pass the gate.

Future work should run CIFAR-10-C, repeat the experiment over multiple seeds, test stronger backbones, and replace fixed thresholds with adaptive threshold selection.

## 8. Conclusion

This project implemented and evaluated source-only inference, BatchNorm adaptation, entropy minimization, and confidence-gated entropy minimization for CIFAR-10 image classification under corruption shift. The evidence supports a conservative conclusion: TTA substantially improves corrupted accuracy, but confidence-gated TTA only matches standard entropy minimization closely and gives at most a marginal, threshold-sensitive gain in this final run.

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
