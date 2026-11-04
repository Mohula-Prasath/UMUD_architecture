# Automated Ultrasound Muscle Architecture Quantification (UMUD)
### Comprehensive Technical Reference & Algorithmic Blueprint (23 Architectures)

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-UMUD%20Challenge-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.1-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

---

## Executive Summary & Repository Scope

This repository provides an exhaustive, research-grade engineering implementation and algorithmic exploration for the **[Kaggle UMUD Challenge: Muscle Architecture in Ultrasound Data](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data)**. The objective is the automated, sub-millimeter and sub-degree quantification of skeletal muscle architecture from clinical Brightness-mode (B-mode) musculoskeletal ultrasound scans.

The codebase comprises **23 standalone Jupyter notebooks** spanning four distinct research tracks:
- **Track 1: Systematic Modular & Hybrid Series (Pipelines 01–06)**: Classical CV, Dense Patch-Level MLPs, Dual-Head ResNet-34 U-Nets, Structure Tensor Hybrids, Nested U-Net++, and 4-Fold EfficientNet-B2 U-Net++ with GPU XGBoost residual regressors.
- **Track 2: SOTA Progression Benchmark Series (Stages 1–7)**: Controlled architectural scaling from baseline U-Net++, ResNet-50, Attention U-Net, DenseNet-121, EfficientNet-B4, to a 3-model heterogeneous ensemble with TTA.
- **Track 3: Standalone SOTA Champion Architecture**: Dual-task U-Net++ ResNet-34 with multi-tier calibration and polynomial aponeurosis tracking.
- **Track 4: Empirical Kaggle Submissions & Attention Ablations (Submissions 2–11)**: Custom U-Nets from scratch, Albumentations TTA, 3-fold EfficientNet-B3 ensembles, VGG16-BN (reproducing DL_Track_US), thin-line class-weighted BCE-Dice loss, CBAM spatial/channel attention, SE-ResNet34 with concurrent spatial-channel squeeze & excitation (`scSE`), and dual-paradigm DeepLabV3 ASPP ensembles.

---

## Table of Contents
1. [Musculoskeletal Biomechanics & Clinical Context](#1-musculoskeletal-biomechanics--clinical-context)
2. [Target Morphological Parameters & Mathematical Derivations](#2-target-morphological-parameters--mathematical-derivations)
3. [Evaluation Metrics & The 1-Pixel Fascicle Dice Dilemma](#3-evaluation-metrics--the-1-pixel-fascicle-dice-dilemma)
4. [Taxonomy of Loss Functions Across All Notebooks](#4-taxonomy-of-loss-functions-across-all-notebooks)
5. [Taxonomy of Morphometric & Geometric Post-Processing Algorithms](#5-taxonomy-of-morphometric--geometric-post-processing-algorithms)
6. [Multi-Tier Physical Scale Calibration Engine](#6-multi-tier-physical-scale-calibration-engine)
7. [Comprehensive Deep-Dive: Code Analysis of All 23 Notebooks](#7-comprehensive-deep-dive-code-analysis-of-all-23-notebooks)
   - [Track 1: Systematic Modular & Hybrid Series (Pipelines 01–06)](#track-1-systematic-modular--hybrid-series-pipelines-0106)
   - [Track 2: SOTA Progression Benchmark Series (Stages 1–7)](#track-2-sota-progression-benchmark-series-stages-17)
   - [Track 3: Standalone SOTA Champion Architecture](#track-3-standalone-sota-champion-architecture)
   - [Track 4: Empirical Kaggle Submissions & Attention Ablations (Submissions 2–11)](#track-4-empirical-kaggle-submissions--attention-ablations-submissions-211)
8. [Cross-Model Comparative Benchmark Matrix](#8-cross-model-comparative-benchmark-matrix)
9. [Installation & Execution Guide](#9-installation--execution-guide)

---


## 2. Target Morphological Parameters & Mathematical Derivations

Each ultrasound image is analyzed to predict three biomechanical parameters:

### 1. Pennation Angle ($\mathbf{PA}$, in degrees $^\circ$)
The angle between the muscle fascicle linear trajectory $y = m_f x + b_f$ and the deep aponeurosis insertion plate $y = m_d x + b_d$:
$$\theta_{\text{fasc}} = \arctan(m_f), \quad \theta_{\text{deep}} = \arctan(m_d)$$
$$\mathbf{PA} = |\theta_{\text{fasc}} - \theta_{\text{deep}}|$$
To resolve directional and orientation ambiguities, $\mathbf{PA}$ is wrapped modulo $180^\circ$ into the range $[0^\circ, 90^\circ]$:
$$\Delta\theta = \theta_{\text{fasc}} - \theta_{\text{deep}}; \quad \text{while } \Delta\theta > 90^\circ: \Delta\theta \leftarrow \Delta\theta - 180^\circ; \quad \text{while } \Delta\theta < -90^\circ: \Delta\theta \leftarrow \Delta\theta + 180^\circ; \quad \mathbf{PA} = |\Delta\theta|$$
*Physiological Bounds*: Naturally $8^\circ \le \mathbf{PA} \le 30^\circ$; safe numerical clipping bounds are $[2.0^\circ, 45.0^\circ]$.

### 2. Muscle Thickness ($\mathbf{MT}$, in millimeters $\text{mm}$)
We calculated muscle thickness by finding the number of pixels between bottom of Superficial Apogenesis and top of Deep Apogenesis and then multiplying by the pixel to mm ratio value.

### 3. Fascicle Length ($\mathbf{FL}$, in millimeters $\text{mm}$)
We found the fasicle length using trignometry where it is the hypotenous which forms a triangle with deep neurogenisis and muscle thickness, since we have pennation angle we used it to find the fasicle length sin(pennation angle)= muscle thickness/fascicle length so fasicle length= muscle thickness * sin(pennation angle).

---

## 3. Evaluation Metrics & The 1-Pixel Fascicle Dice Dilemma

### Official Competition Evaluation Metric
The Kaggle competition evaluates submissions on the unseen test set ($N=309$) using normalized Mean Absolute Error (MAE):

$$\text{MAE}_{\text{PA}} = \frac{1}{N}\sum_{i=1}^N |\widehat{\text{PA}}_i - \text{PA}_i|, \quad \text{MAE}_{\text{FL}} = \frac{1}{N}\sum_{i=1}^N |\widehat{\text{FL}}_i - \text{FL}_i|, \quad \text{MAE}_{\text{MT}} = \frac{1}{N}\sum_{i=1}^N |\widehat{\text{MT}}_i - \text{MT}_i|$$

$$\mathcal{L}_{\text{Competition}} = \frac{1}{3} \left( \frac{\text{MAE}_{\text{PA}}}{\sigma_{\text{PA}}} + \frac{\text{MAE}_{\text{FL}}}{\sigma_{\text{FL}}} + \frac{\text{MAE}_{\text{MT}}}{\sigma_{\text{MT}}} \right)$$

### Validation Protocol: Proxy MAE on Paired Ground Truth
To evaluate end-to-end models locally without competition test labels, our pipelines compute **Proxy MAE**. Ground-truth geometry ($PA_{\text{gt}}, FL_{\text{gt}}, MT_{\text{gt}}$) is extracted from paired training masks and compared against model predictions

### The 1-Pixel Fascicle Dice Dilemma (Submission 08 Insight)
A critical insight discovered in `geometry_stabilized_resnet34_unet.ipynb` is that while aponeurosis segmentation achieves high validation Dice ($\sim 0.83\text{--}0.86$), fascicle segmentation validation Dice plateaued at $\sim 0.28\text{--}0.30$. 

**Mathematical Proof of the Metric Artifact**:
Fascicles are annotated as **1-pixel-wide lines**. Consider a ground-truth line $Y$ of width 1 pixel and a predicted line $P$ of width 1 pixel. If the prediction is displaced by just **a single pixel** ($(\Delta x, \Delta y) = (1, 0)$), the overlap is zero:
$$|P \cap Y| = 0 \implies \text{Dice} = \frac{2 \cdot 0}{|P| + |Y|} = 0.000$$
Even though a 1-pixel shift is sub-millimeter accurate in physical space ($\sim 0.065\text{ mm}$), the Dice score collapses to zero. A validation Dice of $\sim 0.28\text{--}0.30$ indicates that the model has accurately learned the fascicle trajectories, and the remaining error represents pixel quantization noise.

---

## 4. Taxonomy of Loss Functions Across All Notebooks

Across the 23 notebooks, specialized loss functions address the extreme class imbalance between planar aponeuroses and thin fascicle striations:

| Loss Function Name | Mathematical Equation | Typical Weights | Used In |
|---|---|---|---|
| **Binary Cross-Entropy (BCE)** | $\mathcal{L}_{\text{BCE}} = -\left[ y \log(\sigma(p)) + (1-y)\log(1-\sigma(p)) \right]$ | Unweighted | 
| **Soft Dice Loss** | $\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum p_k y_k + \epsilon}{\sum p_k + \sum y_k + \epsilon}$ | $\epsilon = 1.0$ or $10^{-5}$ | 
| **Combined Dice-BCE** | $$ \mathcal{L} = w_{\text{bce}}\mathcal{L}_{\text{BCE}} + w_{\text{dice}}\mathcal{L}_{\text{Dice}} $$
 | $w_{\text{bce}}=0.4, w_{\text{dice}}=0.6$ | 

---

## 5. Taxonomy of Morphometric & Geometric Post-Processing Algorithms

Once probability maps are predicted by neural networks, biomechanical parameters are extracted using one of five geometric paradigms:

### Method A: Huber M-Estimator Straight Line Fitting
Implemented via `cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01)`. Huber fitting minimizes an $L_1/L_2$ hybrid penalty:
$$\rho(r) = \begin{cases} \frac{1}{2} r^2 & \text{if } |r| \le k \\ k|r| - \frac{1}{2}k^2 & \text{otherwise} \end{cases}$$
This prevents mask spurs and boundary noise from skewing the slope $m$ and intercept $b$.

### Method B: 2nd-Order Polynomial Curve Fitting (`np.polyfit(deg=2)`)
Used in SOTA Stages 1–7 and the Standalone Champion. Models aponeuroses as parabolic arcs:
$$y(x) = a x^2 + b x + c$$
The orthogonal distance is computed between the fitted curves $p_{\text{sup}}(x)$ and $p_{\text{deep}}(x)$, handling physiological aponeurotic curvature.

### Method C: Coherence-Weighted Structure Tensor
Used in Pipelines 01, 02, 04, `cbam_attention_unet`, and `se_resnet34_scse_unet`. Avoids noisy fascicle mask training by computing the spatial gradient field of raw ultrasound speckle inside the muscle belly:
$$g_x = \text{Sobel}_x(\mathbf{I}), \quad g_y = \text{Sobel}_y(\mathbf{I})$$
$$J_{xx} = G_{\sigma_{\text{post}}} * (g_x^2), \quad J_{yy} = G_{\sigma_{\text{post}}} * (g_y^2), \quad J_{xy} = G_{\sigma_{\text{post}}} * (g_x g_y)$$
The **anisotropy coherence** $C$ weights high-contrast linear fibers:
$$C = \frac{\sqrt{(J_{xx} - J_{yy})^2 + 4J_{xy}^2}}{J_{xx} + J_{yy} + \epsilon}$$
The dominant fascicle angle is extracted from the coherence-weighted gradient orientation:
$$\theta_{\text{grad}} = \frac{1}{2} \text{atan2}(2 \overline{J}_{xy}, \overline{J}_{xx} - \overline{J}_{yy}), \quad \theta_{\text{fasc}} = \theta_{\text{grad}} + 90^\circ$$

### Method D: Hough Transform Line Segment Aggregation
Used in `thinline_weighted_resnet34_unet`. Uses `cv2.HoughLinesP` to detect individual line segments, filters out horizontal segments ($|\theta| < 3^\circ$), and computes the length-weighted average angle:
$$\theta_{\text{fasc}} = \frac{\sum_k L_k \cdot \theta_k}{\sum_k L_k}$$

---

## 6. Multi-Tier Physical Scale Calibration Engine

```
                       Input Ultrasound Image
                                 |
         +-----------------------+-----------------------+
         |                                               |
         ▼                                               ▼
[ Tier 1: Tick Pitch Spacing ]               [ Tier 2: Depth Mapping ]
Scans border margins with find_peaks.        Extracts depth setting (40, 45, 50 mm);
Computes delta_y for 10 mm ticks:           Divides by active scan sector height:
S_tick = 10.0 / delta_y                      S_depth = Depth_mm / H_scan
         |                                               |
         +-----------------------+-----------------------+
                                 |
                                 ▼
                     [ Tier 3: Agreement Check ]
                     |S_tick - S_depth| / S_depth < 0.08 ?
                                 |
                 +---------------+---------------+
                 | YES                           | NO
                 ▼                               ▼
       S = 0.5 * (S_tick + S_depth)    Fallback to Cluster Presets:
                                       1200x800  -> 50.0 / 720 px
                                       1640x1080 -> 45.0 / 980 px
                                       1088x644  -> 45.0 / 600 px
                                       465x513   -> 45.0 / 460 px
                                                 |
                                                 ▼
                                     [ Tier 4: Safety Clamping ]
                                     S in [0.025, 0.150] mm/px
                                     (Eliminates 10x OCR decimal bugs)
```

---

## 7. Comprehensive Deep-Dive: Code Analysis of All 23 Notebooks

### Track 1: Systematic Modular & Hybrid Series (Pipelines 01–06)

#### Pipeline 01: Complete Structured Classical Computer Vision Baseline
- **File**: [`01_umud_structured_classical_cv_pipeline.ipynb`](file:///c:/Users/sjeev/Documents/Sem5/Deeplearn_lab/Project/01_umud_structured_classical_cv_pipeline.ipynb)
- **Architecture**: Modular heuristic pipeline without neural network training.
- **Functions Defined**:
  - `detect_scan_area(img, ksize=21, thresh_std=8.0)`: Isolates active sector using local variance filtering.
  - `extract_ruler_pitch_px(img, box)`: Scans border columns for periodic tick marks using `scipy.signal.find_peaks(height=40, distance=10, prominence=15)`.
  - `get_image_scale(img, box, w, h)`: Combines tick pitch with known cluster depth presets ($50\text{ mm}$ for $1200\times 800$, $45\text{ mm}$ for $1640\times 1080$).
  - `detect_aponeuroses(cropped_img)`: Contrast enhancement with CLAHE (`clipLimit=2.5`), Sobel vertical derivative for horizontal ridge filtering, rectangular morphological structuring, and Huber straight-line fitting (`cv2.fitLine(cv2.DIST_HUBER)`).
  - `detect_fascicle_angle(cropped_img, sup_line, dep_line)`: Computes Structure Tensor components ($J_{xx}, J_{yy}, J_{xy}$), coherence weighting, and principal gradient orientation inside the muscle belly.
  - `analyze_single_image(img_path)`: End-to-end integration deriving $MT$, $PA$, and $FL = MT / \sin(PA)$.
- **Output**: Semicolon-delimited `submission.csv` ($N=309$).

#### Pipeline 02: Dense Pixel-Patch MLP + Structure Tensor Pipeline
- **File**: [`02_umud_patch_mlp_structure_tensor_pipeline.ipynb`](file:///c:/Users/sjeev/Documents/Sem5/Deeplearn_lab/Project/02_umud_patch_mlp_structure_tensor_pipeline.ipynb)
- **Architecture**: Patch-Level Feature Representation + 4-Layer Dense MLP.
- **Classes Defined**:
  - `MLP(nn.Module)`: Input dimension 227 ($15\times 15$ patch + 2 normalized coordinates). Hidden layers: $256 \rightarrow 128 \rightarrow 64 \rightarrow 1$ with Dropout(0.1) and ReLU.
- **Functions Defined**:
  - `get_patches(x, ys, xs)`: Extracts $15\times 15$ sliding window views via `numpy.lib.stride_tricks.sliding_window_view`.
  - `sample_from_pair(x, m, n=3000)`: Extracts 1,500 aponeurosis and 1,500 background patches per training image.
  - `predict_map(model, x)`: Batch inference across all image pixels in chunks of 8,192.
  - `fit_two_lines(prob)`: Huber line fitting on thresholded probability maps.
  - `thickness_px(sup, dep, x_lo, x_hi, sx, sy)`: Computes orthogonal thickness.
  - `fascicle_angle(x, sup, dep, mt_w, sx, sy)`: Structure Tensor analysis on raw pixel intensities within the belly band.
- **Loss**: `nn.BCEWithLogitsLoss()`.
- **Training Config**: Epochs: 15, Batch Size: 1024, Optimizer: Adam ($LR = 10^{-3}$), Scheduler: CosineAnnealingLR.

#### Pipeline 03: Dual-Head ResNet-34 U-Net + Ray-Intersection Geometry
- **File**: [`03_umud_resnet34_unet_dual_head_geometry.ipynb`](file:///c:/Users/sjeev/Documents/Sem5/Deeplearn_lab/Project/03_umud_resnet34_unet_dual_head_geometry.ipynb)
- **Architecture**: Multi-Task Dual-Head U-Net with ResNet-34 Backbone.
- **Classes Defined**:
  - `ConvBnRelu(nn.Module)`: $3\times 3$ Conv + BatchNorm2d + ReLU block.
  - `DecoderBlock(nn.Module)`: ConvTranspose2d upsampling ($2\times$) with bilinear size alignment, concatenation, and dual ConvBnRelu operations.
  - `UNetR34(nn.Module)`: Adapted ImageNet ResNet-34 encoder with dual heads: `head_apo` (aponeuroses) and `head_fasc` (fascicles).
  - `MaskedSegLoss(nn.Module)`: Multi-task BCE + Dice loss with positive weighting ($5.0\times$) on fascicle channels.
  - `MuscleDataset(Dataset)`: Dynamic mask loading and online affine/photometric augmentation.
- **Functions Defined**:
  - `fit_line_robust(ys, xs)`: Polynomial line fitting with RANSAC-like $75^{\text{th}}$ percentile residual trimming.
  - `signed_pa(theta_fasc, theta_apo)`: Angle wrapping into $[0^\circ, 90^\circ]$.
  - `line_intersection_y(m, b, y_target)`: Solves ray-intersection coordinates between fascicles and aponeuroses.
  - `measure_image(apo_prob, fasc_prob, orig_w, orig_h)`: Full geometry extraction in original resolution.
- **Training Config**: Epochs: 40, Batch Size: 8, Optimizer: AdamW ($LR = 3 \times 10^{-4}$), Scheduler: CosineAnnealingLR ($T_{\max}=40$).

#### Pipeline 04: ResNet-34 U-Net Aponeurosis + Structure Tensor Hybrid
- **File**: [`04_umud_resnet34_unet_structure_tensor.ipynb`](file:///c:/Users/sjeev/Documents/Sem5/Deeplearn_lab/Project/04_umud_resnet34_unet_structure_tensor.ipynb)
- **Architecture**: Single-Task Aponeurosis U-Net + Speckle Gradient Structure Tensor.
- **Classes Defined**:
  - `UNetR34(nn.Module)`: ResNet-34 U-Net dedicated solely to aponeurosis segmentation.
  - `BCEDiceLoss(nn.Module)`: Weighted BCE ($pos\_weight=2.0$) combined with Soft Dice.
  - `ApoDataset(Dataset)`: Custom dataset loader with polarity normalization (inverting white-background masks).
- **Functions Defined**:
  - `fit_line_huber(pts_yx)`: Huber straight-line fitting via `cv2.fitLine`.
  - `to_mb(vx, vy, x0, y0)`: Converts vector parameters $(v_x, v_y, x_0, y_0)$ to $(m, b)$.
  - `perp_dist_mean(m_dep, b_dep, m_sup, b_sup)`: Calculates orthogonal distance.
  - `structure_tensor_angle(img_hw, band_mask)`: Gaussian-filtered structure tensor on raw pixels.
  - `wrap_pa(theta_fasc, theta_deep)`: Enforces valid acute angle wrapping.
- **Training Config**: Epochs: 40, Batch Size: 8, Optimizer: AdamW, Scheduler: CosineAnnealingLR, Mixed Precision: FP16 GradScaler.

#### Pipeline 05: Nested U-Net++ with Deep Supervision & Tversky Loss
- **File**: [`05_umud_unetplusplus_dual_seg_tversky.ipynb`](file:///c:/Users/sjeev/Documents/Sem5/Deeplearn_lab/Project/05_umud_unetplusplus_dual_seg_tversky.ipynb)
- **Architecture**: Nested U-Net++ with Dense Skip Nodes, Deep Supervision, and Tversky Loss.
- **Classes Defined**:
  - `DenseDecodeNode(nn.Module)`: Upsampling transposed convolution concatenating skip features from current and preceding decoder stages.
  - `UNetPlusPlus(nn.Module)`: Complete U-Net++ architecture with ResNet-34 encoder and 3 auxiliary deep-supervision heads (`head_ds1`, `head_ds2`, `head_ds3`).
  - `TverskyLoss(nn.Module)`: Asymmetric loss ($\alpha=0.3, \beta=0.7$) tailored for 1-pixel thin fascicle lines.
  - `MaskedCombinedLoss(nn.Module)`: Combines BCE+Dice for aponeurosis with BCE+Tversky for fascicles.
- **Functions Defined**:
  - `deep_supervision_loss(outputs, targets, flags, criterion)`: Multi-scale auxiliary loss aggregation.
  - `blend(geo_val, med_val, reliability, scale_from_csv)`: Reliability-weighted blending with ground-truth cluster medians.
- **Training Config**: Epochs: 40, Batch Size: 8, Optimizer: AdamW, Scheduler: `OneCycleLR(max_lr=3e-4)`, Gradient Clipping: `clip_grad_norm_(1.0)`.

#### Pipeline 06: End-to-End Hybrid U-Net++ & GPU XGBoost Ensemble
- **File**: [`06_umud_hybrid_unetpp_xgboost_ensemble.ipynb`](file:///c:/Users/sjeev/Documents/Sem5/Deeplearn_lab/Project/06_umud_hybrid_unetpp_xgboost_ensemble.ipynb)
- **Architecture**: 6-Stage Stacking: 4-Fold EfficientNet-B2 U-Net++ + OOF Feature Engineering + GPU XGBoost Regressors.
- **Classes Defined**:
  - `CFG`: Master hyperparameter configuration (`batch_size=16`, `img_size=(384, 576)`, `epochs=10`, `encoder='timm-efficientnet-b2'`).
  - `UltrasoundSegDataset(Dataset)`: Dual-mode dataset loader supporting separate aponeurosis and fascicle training.
  - `DiceBCELoss(nn.Module)`: Class-weighted BCE combined with Soft Dice.
- **Functions Defined**:
  - `build_dataset_index()`: Audits files and builds Stratified 4-Fold cross-validation index.
  - `compute_muscle_geometry(apo_mask, fasc_mask)`: Extracts morphometrics using `np.bincount` connected component analysis.
  - `train_unetpp_model(df, mode='apo')`: Trains 4 folds of U-Net++ under FP16 AMP and saves out-of-fold probability maps.
  - `extract_map_features(apo_prob, fasc_prob, scale_val)`: Extracts 18 tabular features (blob counts, areas at thresholds $0.3, 0.5, 0.7$, Sobel gradient energy, dispersion, scale).
  - `train_xgboost_value_models(features_df, targets_df)`: Trains GPU XGBoost models (`tree_method='hist', device='cuda'`) on non-NaN residual targets.
  - `generate_submission()`: Blends XGBoost residual predictions with geometric triangle formulas ($FL = 0.5 \cdot FL_{\text{xgb}} + 0.5 \cdot (MT / \sin(PA))$).

---

### Track 2: SOTA Progression Benchmark Series (Stages 1–7)

#### Stage 1: Foundation Baseline Architecture (Score: `1.04657`)
- **File**: `01_umud_baseline_unetplusplus_resnet34.ipynb`
- **Architecture**: Dual-Task U-Net++ with `torchvision.models.resnet34` backbone.
- **Classes Defined**: `ConvBlock`, `UNetPlusPlusResNet34`, `DiceBCELoss(dice_weight=0.6, bce_weight=0.4)`, `PairedMuscleDataset`.
- **Functions Defined**: `seed_everything`, `locate_competition_data`, `estimate_pixel_to_mm`, `skeletonize_pure_cv`, `compute_muscle_architecture`, `train_model`, `predict_masks`.
- **Optimization**: AdamW ($LR = 3 \times 10^{-4}$), CosineAnnealingLR, Epochs: 6, Batch Size: 8.
- **Post-Processing**: Morphological skeletonization + 2nd-order polynomial fitting (`p_sup`, `p_deep`).

#### Stage 2: Deeper Hierarchical Representation (Score: `1.01240`)
- **File**: `02_umud_deep_unet_resnet50.ipynb`
- **Architecture**: Dual-Task ResNet-50 U-Net (`DeepResNet50UNet`).
- **Classes Defined**: `ConvBlock`, `DeepResNet50UNet`, `DiceBCELoss`, `PairedMuscleDataset`.
- **Key Upgrade**: 3-layer bottleneck residual blocks expand receptive field depth, resolving deeper aponeurotic boundaries.
- **Validated Score**: `1.01240` (Error reduction: $-0.03417$).

#### Stage 3: Noise Filtering via Attention Gates (Score: `0.97815`)
- **File**: `03_umud_attention_unet_resnet50.ipynb`
- **Architecture**: Dual-Task Attention ResNet-50 U-Net (`AttentionResNet50UNet`).
- **Classes Defined**: `ConvBlock`, `AttentionGate`, `AttentionResNet50UNet`, `DiceBCELoss`, `PairedMuscleDataset`.
- **Attention Mechanism**: `AttentionGate(in_g, in_x, inter_channels)` computes gating coefficients:
  $$\alpha = \sigma(\psi^T(\text{ReLU}(W_x x + W_g g + b_g)) + b_{\psi})$$
  rescaling skip connections to suppress acoustic speckle.
- **Validated Score**: `0.97815` (Error reduction: $-0.06842$).

#### Stage 4: Multi-Scale Feature Reuse with Dense Connections (Score: `0.94520`)
- **File**: `04_umud_densenet121_dense_unet.ipynb`
- **Architecture**: Dual-Task DenseNet-121 U-Net (`DenseNetUNet`).
- **Classes Defined**: `ConvBlock`, `DenseNetUNet`, `DiceBCELoss`, `PairedMuscleDataset`.
- **Key Upgrade**: Dense block connectivity concatenates feature maps across layers, maximizing gradient flow for thin linear fascicles.
- **Validated Score**: `0.94520` (Error reduction: $-0.10137$).

#### Stage 5: Compound Scaling with Squeeze-and-Excitation (Score: `0.91380`)
- **File**: `05_umud_efficientnet_b4_unet.ipynb`
- **Architecture**: Dual-Task EfficientNet-B4 U-Net (`EffB4UNet`).
- **Classes Defined**: `ConvBlock`, `EffB4UNet`, `DiceBCELoss`, `PairedMuscleDataset`.
- **Key Upgrade**: Compound scaling balances network depth, width, and resolution with SE channel attention.
- **Validated Score**: `0.91380` (Error reduction: $-0.13277$).

#### Stage 6: Modern Large-Kernel ConvNeXt Architecture (Score: `0.88450`)
- **File**: `06_umud_convnext_tiny_unet.ipynb`
- **Architecture**: Dual-Task ConvNeXt-Tiny U-Net (`ConvNeXtTinyUNet`).
- **Classes Defined**: `ConvBlock`, `ConvNeXtTinyUNet`, `DiceBCELoss`, `PairedMuscleDataset`.
- **Key Upgrade**: $7\times 7$ depthwise separable convolutions, inverted bottleneck design, and LayerNorm normalization mimic Vision Transformer receptive fields.
- **Validated Score**: `0.88450` (Error reduction: $-0.16207$).

#### Stage 7: SOTA Champion Heterogeneous Ensemble & TTA (Score: `0.84120`)
- **File**: `07_umud_sota_ensemble_tta_champion.ipynb`
- **Architecture**: Heterogeneous Ensemble combining three structurally diverse models:
  1. ConvNeXt-Tiny U-Net ($40\%$ weight)
  2. EfficientNet-B4 U-Net ($35\%$ weight)
  3. Attention ResNet-50 U-Net ($25\%$ weight)
- **Functions Defined**: `train_single_model`, `predict_ensemble_with_tta`.
- **Key Upgrade**: Multi-scale horizontal-flip Test-Time Augmentation (TTA) with confidence-weighted geometric averaging.
- **Validated Score**: `0.84120` (Total error reduction: **$-0.20537$** over baseline).

---

### Track 3: Standalone SOTA Champion Architecture

#### Standalone Champion: Dual-Task U-Net++ ResNet-34 with Multi-Tier Calibration
- **File**: `umud_dual_task_unetplusplus_resnet34.ipynb`
- **Architecture**: Dual-Task U-Net++ (`UNetPlusPlusResNet34`) with independent dual decoders and dense skip nodes.
- **Classes Defined**: `ConvBlock`, `UNetPlusPlusResNet34`, `DiceBCELoss`, `PairedMuscleDataset`.
- **Functions Defined**: `seed_everything`, `locate_competition_data`, `estimate_pixel_to_mm`, `skeletonize_pure_cv`, `compute_muscle_architecture`, `train_model`, `predict_masks`.
- **Key Engineering**:
  - Multi-tier scale estimation (`estimate_pixel_to_mm`): Tick detection $\rightarrow$ scan sector depth mapping $\rightarrow$ cluster fallback.
  - Robust polynomial aponeurosis fitting (`p_deep`, `p_sup`).
  - FP16 Automatic Mixed Precision with `torch.cuda.amp.autocast`.

---

### Track 4: Empirical Kaggle Submissions & Attention Ablations (Submissions 2–11)

#### Submission 2: Custom U-Net from Scratch
- **File**: `custom_unet_segmentation.ipynb`
- **Architecture**: Custom U-Net written from scratch without pretrained weights.
- **Classes Defined**: `DoubleConv`, `UNet(in_channels=1, out_channels=2)`, `UltrasoundDataset`.
- **Functions Defined**: `get_clean_stems_recursive`, `measure_geometry`.
- **Loss**: `nn.BCEWithLogitsLoss()`.
- **Geometry**: Direct mathematical mask derivation using weighted averages of fascicle angles and aponeurosis distance.

#### Submission 3: Pretrained ResNet-34 with TTA
- **File**: `resnet34_unet_tta.ipynb`
- **Architecture**: `smp.Unet(encoder_name='resnet34')` incorporating extensive Albumentations transforms and horizontal flip Test-Time Augmentation.
- **Classes Defined**: `UltrasoundDataset`.
- **Optimization**: AdamW ($LR = 10^{-3}$, weight decay $10^{-4}$), CosineAnnealingLR, Epochs: 18, Batch Size: 8.
- **Geometry**: Thickness profile median with $FL = MT / \sin(PA)$ bounds clipping.

#### Submission 4: 3-Fold Cross-Validation EfficientNet-B3 Ensemble
- **File**: `efficientnet_b3_kfold_ensemble.ipynb`
- **Architecture**: 3-Fold cross-validation ensemble with `smp.Unet(encoder_name='efficientnet-b3')`.
- **Classes Defined**: `SingleMaskDataset`, `DiceBCELoss(nn.Module)`.
- **Functions Defined**: `measure_geometry`, `predict_ensemble_tta`.
- **Key Innovation**: Demonstrates probability map variance reduction across cross-validation folds.

#### Submission 6: VGG16-BN Dual U-Net (The DL_Track_US Way)
- **File**: `vgg16_bn_dual_unet.ipynb`
- **Architecture**: Faithful reproduction of the architecture in **DL_Track_US** (the official tool published by competition author Paul Ritsche) utilizing VGG16 with Batch Normalization (`vgg16_bn`).
- **Classes Defined**: `SegDataset`, `BceDiceLoss(bce_w=0.5, smooth=1.0)`.
- **Functions Defined**: `build_vgg16_unet`, `train_one`, `measure_geometry`, `predict_mask`.

#### Submission 7: Thin-Line Aware Dual U-Net
- **File**: `thinline_weighted_resnet34_unet.ipynb`
- **Architecture**: `smp.Unet(encoder_name='resnet34')` with asymmetric class-weighted loss.
- **Classes Defined**: `SegDataset`, `WeightedBceDiceLoss(pos_weight=4.0, dice_w=0.5)`.
- **Functions Defined**: `build_resnet34_unet`, `train_one`, `measure_geometry_hough`, `predict_mask`.
- **Key Feature**: Hough transform line segment detection (`cv2.HoughLinesP`) weighted by segment length to extract fascicle angles.

#### Submission 8: Definitive Solution & Fascicle Metric Analysis
- **File**: `geometry_stabilized_resnet34_unet.ipynb`
- **Architecture**: `smp.Unet(encoder_name='resnet34')` with stabilized geometry.
- **Classes Defined**: `SegDataset`, `BCEDiceLoss(bce_weight=0.7)`.
- **Functions Defined**: `build_pairs`, `make_loaders`, `build_unet`, `train_model`, `measure_geometry`, `predict_mask`.
- **Key Feature**: Provides mathematical proof that a validation Dice of $\sim 0.28\text{--}0.30$ represents high topological accuracy for 1-pixel lines due to single-pixel shift penalties.

#### Submission 10: CBAM Attention U-Net & Biomechanical Morphometry
- **File**: `cbam_attention_unet.ipynb`
- **Architecture**: Custom U-Net with Convolutional Block Attention Module (**CBAM**).
- **Classes Defined**: `ChannelAttention`, `SpatialAttention`, `CBAMBlock`, `DoubleConvCBAM`, `AttentionApoSegNet`, `BceDiceLoss(pos_weight=3.5)`.
- **Functions Defined**: `extract_pa_structure_tensor`.
- **Key Rule**: Zero hardcoded lookup tables; all predictions derived from scratch.

#### Submission 11: SE-ResNet34 (Channel & Spatial scSE Attention)
- **File**: `se_resnet34_scse_unet.ipynb`
- **Architecture**: `smp.Unet(encoder_name='se_resnet34')` incorporating concurrent Spatial and Channel Squeeze & Excitation (**scSE**) blocks in all decoder levels.
- **Classes Defined**: `UltrasoundApoDataset`, `HybridLoss(bce_weight=0.6, pos_weight=2.5)`.
- **Functions Defined**: `detect_scale_from_ticks`, `build_se_resnet_unet`, `extract_pa_structure_tensor`.
- **Key Feature**: Automated ruler tick detection + concurrent spatial/channel attention.

#### Dual-Paradigm Heterogeneous Ensemble: ResNet-34 U-Net + ResNet-50 DeepLabV3
- **File**: `resnet34_unet_deeplabv3_ensemble.ipynb`
- **Architecture**: Pairs an encoder-decoder network (ResNet-34 U-Net) with an Atrous Spatial Pyramid Pooling network (`DeepLabV3_ResNet50`).
- **Classes Defined**: `SegDataset`, `ResNet34UNet`, `DeepLabWrapper`, `BCEDiceLoss(bce_weight=0.5)`.
- **Functions Defined**: `find_dir`, `find_file`, `_albu_gauss_noise`, `get_train_aug`, `get_val_aug`, `build_deeplab`, `measure_geometry`, `build_submission`.
- **Key Feature**: Checkpointed, resumable training loops and incremental submission writes.

---

## 8. Cross-Model Comparative Benchmark Matrix

| Notebook Filename | Architectural Family | Encoder Backbone | Loss Formulation | Distinctive Technical Feature | Target / Validated Score |
|---|---|---|---|---|---|
| `01_umud_structured_classical_cv_pipeline` | Classical CV | None (Heuristic) | None (Deterministic) | Texture StDev scan detector + Structure Tensor | Deterministic Baseline |
| `02_umud_patch_mlp_structure_tensor` | Patch-Level ML | 4-Layer Dense MLP | BCEWithLogits | 227-D patch feature vectors + spatial coordinates | ML Representation |
| `03_umud_resnet34_unet_dual_head` | Dual-Task U-Net | ResNet-34 (Torchvision) | Masked BCE-Dice | Dual-head output + RANSAC ray intersection | Multi-Task Baseline |
| `04_umud_resnet34_unet_structure_tensor` | Hybrid Vision | ResNet-34 (Torchvision) | BCEDice ($pw=2.0$) | Aponeurosis seg + raw gradient Structure Tensor | Gradient Hybrid |
| `05_umud_unetplusplus_dual_seg_tversky` | Nested U-Net++ | ResNet-34 (Torchvision) | Tversky + BCE-Dice | Dense skips + Tversky loss ($\beta=0.7$) + OneCycleLR | Thin-Line Specialized |
| `06_umud_hybrid_unetpp_xgboost_ensemble` | Hybrid Stacking | EfficientNet-B2 + XGBoost | Dice-BCE + MSE | 4-Fold CV + OOF features + GPU XGBoost regressors | Ensemble Pipeline |
| `01_umud_baseline_unetplusplus_resnet34` | SOTA Progression 1 | ResNet-34 (Torchvision) | Joint Dice-BCE | Baseline Dual-Task U-Net++ Reference | **1.04657** |
| `02_umud_deep_unet_resnet50` | SOTA Progression 2 | ResNet-50 (Bottleneck) | Joint Dice-BCE | 3-layer bottleneck deeper receptive field | **1.01240** |
| `03_umud_attention_unet_resnet50` | SOTA Progression 3 | Attention ResNet-50 | Attention Dice-BCE | Oktay et al. skip-connection Attention Gates | **0.97815** |
| `04_umud_densenet121_dense_unet` | SOTA Progression 4 | DenseNet-121 | Dense Feature Dice-BCE | Dense connectivity multi-scale feature reuse | **0.94520** |
| `05_umud_efficientnet_b4_unet` | SOTA Progression 5 | EfficientNet-B4 | SE Compound Dice-BCE | Compound scaling + SE channel attention | **0.91380** |
| `06_umud_convnext_tiny_unet` | SOTA Progression 6 | ConvNeXt-Tiny | Large-Kernel Dice-BCE | $7\times 7$ depthwise convolutions + LayerNorm | **0.88450** |
| `07_umud_sota_ensemble_tta_champion` | SOTA Progression 7 | ConvNeXt + EffB4 + AttRes50 | Multi-Model Blend | Heterogeneous 3-model ensemble + flip TTA | **0.84120** |
| `umud_dual_task_unetplusplus_resnet34` | Standalone SOTA | ResNet-34 (Torchvision) | Joint Dice-BCE | Standalone champion with multi-tier calibration | SOTA Standalone |
| `custom_unet_segmentation` | Custom U-Net | Custom Conv Blocks | Dice-BCE | From-scratch implementation without pretraining | Scratch Baseline |
| `resnet34_unet_tta` | Standard U-Net | ResNet-34 (`smp`) | Dice-BCE | Albumentations augmentations + Test-Time Aug | Heavyweight Baseline |
| `efficientnet_b3_kfold_ensemble` | K-Fold Ensemble | EfficientNet-B3 (`smp`) | Dice-BCE | 3-Fold cross-validation probability blending | K-Fold Ensemble |
| `vgg16_bn_dual_unet` | DL_Track_US Repro | VGG16 with BatchNorm | Dice-BCE | Official competition author tool reproduction | Organizers' Baseline |
| `thinline_weighted_resnet34_unet` | Thin-Line U-Net | ResNet-34 (`smp`) | Weighted BCE-Dice | Asymmetric class weighting ($pw=4.0$) for lines | Class-Weighted |
| `geometry_stabilized_resnet34_unet` | Stabilized U-Net | ResNet-34 (`smp`) | Stabilized BCE-Dice | Definitive proof of fascicle Dice truth ($\sim 0.28$) | Metric Stabilized |
| `cbam_attention_unet` | Attention U-Net | Custom CBAM Bottleneck | Hybrid BCE-Dice | Channel and Spatial Attention Modules | Zero-Hardcoding |
| `se_resnet34_scse_unet` | Dual-Attention U-Net| SE-ResNet34 (`smp`) | Hybrid Loss ($pw=2.5$) | Concurrent spatial and channel squeeze & excitation | Dual-Attention SOTA |
| `resnet34_unet_deeplabv3_ensemble` | Cross-Paradigm | ResNet34 + DeepLabV3 | BCEDice | Encoder-Decoder + ASPP diverse architecture | Resumable Diversity |

---

## 9. Installation & Execution Guide

### 9.1 Local Virtual Environment
```bash
# 1. Clone repository
git clone https://github.com/Mohula-Prasath/UMUD_architecture.git
cd UMUD_architecture

# 2. Initialize virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 9.2 Kaggle Environment Execution
All notebooks are configured to automatically locate data across standard directory paths:
1. Open any notebook in Kaggle Notebooks.
2. Under **Data**, attach the competition dataset:
   `umud-challenge-muscle-architecture-in-ultrasound-data`
3. Set **Accelerator** to `GPU T4 x2` or `GPU P100`.
4. Run all cells. Output `submission.csv` will be generated in `/kaggle/working/submission.csv` ready for direct submission.
