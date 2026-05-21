# Literature Notes: Test-Time Adaptation Under Corruption Shift

Search date: 2026-05-21. Scope: concise background for a 6-page ML course report on image classification under distribution shift. Citation keys refer to `references.bib`.

## Datasets and Distribution Shift

- **CIFAR-10** (`krizhevsky2009cifar10`, `uciCifar10`). CIFAR-10 is a standard small-image classification benchmark: 60,000 color images, 32 x 32 pixels, 10 classes. Krizhevsky's technical report describes CIFAR-10 as a labeled subset collected from the 80 Million Tiny Images source; the current UCI record lists the dataset DOI `10.24432/C5889J`. Use CIFAR-10 as the clean/source-domain benchmark.

- **CIFAR-10-C / common corruptions** (`hendrycks2019robustness`, `hendrycks2019cifar10czenodo`). Hendrycks and Dietterich introduced corruption robustness benchmarks to evaluate common, non-adversarial visual corruptions. The CIFAR-10-C Zenodo record states that each corruption `.npy` contains 50,000 images: the original CIFAR-10 test set corrupted at five severities, 10,000 images per severity. This supports reporting accuracy by corruption type and/or mean corruption error across severity levels.

## Model Components

- **ResNet** (`he2016resnet`). Residual networks learn residual mappings via shortcut connections, making very deep CNNs trainable and strong baselines for visual recognition. For this project, ResNet is the natural backbone because CIFAR-10/CIFAR-10-C experiments commonly use ResNet-family classifiers and because residual blocks separate representation learning from the normalization layers adapted by TTA methods.

- **Batch Normalization** (`ioffe2015batchnorm`). BatchNorm normalizes layer activations using batch statistics and learns affine scale/shift parameters. It was introduced to stabilize and accelerate deep network training. In test-time adaptation, BN is important because methods can recompute normalization statistics on target batches and/or update only BN affine parameters, yielding a small, stable adaptation parameter set.

## Entropy Minimization and TTA

- **Entropy minimization principle** (`grandvalet2004entropy`). Minimum entropy regularization was proposed for semi-supervised learning: unlabeled examples can shape decision boundaries by encouraging confident predictions, consistent with the cluster assumption. This is the conceptual ancestor of entropy-based unsupervised objectives at test time.

- **TENT** (`wang2021tent`). TENT performs fully test-time adaptation using only test inputs and the pretrained model. It minimizes prediction entropy on test batches, estimates normalization statistics, and updates channel-wise BN affine transformations online. The paper reports gains on corrupted CIFAR-10/100 and ImageNet-C without changing source training, making it directly relevant to CIFAR-10-C adaptation experiments.

## Confidence / Entropy Gating Rationale

- **Softmax confidence as an error/OOD signal** (`hendrycks2017baseline`). Hendrycks and Gimpel showed that correctly classified examples tend to have higher maximum softmax probabilities than misclassified or out-of-distribution examples. This motivates using confidence or entropy to decide which examples are reliable enough to influence adaptation.

- **Selective prediction framing** (`geifman2017selective`). Selective classification formalizes the tradeoff between coverage and risk: a model can abstain or filter low-confidence predictions to reduce error on retained examples. For TTA, this supports gating updates rather than adapting on every target sample.

- **Calibration caveat** (`guo2017calibration`). Modern neural networks can be poorly calibrated, and depth, width, weight decay, and BatchNorm affect calibration. Therefore, confidence/entropy gates should be treated as heuristics unless calibrated or validated; thresholds should be tuned on held-out data or analyzed in ablations.

- **Entropy filtering in TTA** (`niu2022eata`). EATA extends entropy-minimization TTA by selecting reliable, non-redundant test samples and excluding high-entropy samples, arguing that high-entropy inputs can produce noisy gradients that disrupt adaptation. This is the most direct citation for entropy/confidence-based gating of TTA updates.

## Suggested Report Usage

- Introduce CIFAR-10 as the clean classification benchmark and CIFAR-10-C as the controlled corruption-shift benchmark.
- Present ResNet + BatchNorm as the base model structure enabling lightweight TTA.
- Explain TENT as source-free, online entropy minimization over BN affine parameters.
- Justify gating as reducing harmful updates from uncertain samples, while noting calibration risk.

