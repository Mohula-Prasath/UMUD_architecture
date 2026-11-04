# UMUD Challenge: Dataset Specification & Data Audit Report

This document provides a comprehensive technical reference for the dataset used in the **Ultrasound Muscle Architecture in Ultrasound Data (UMUD) Challenge** hosted on Kaggle:
[https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data)

---

## 1. Dataset Overview

The UMUD dataset consists of musculoskeletal B-mode ultrasound scans collected across diverse clinical ultrasound transducers, acquisition depths, and patient cohorts. The objective is the automated morphological quantification of three primary architectural biomarkers:
1. **Pennation Angle ($\mathbf{PA}$ in degrees)**: The insertion angle of muscle fascicle fibers into the deep aponeurosis.
2. **Fascicle Length ($\mathbf{FL}$ in millimeters)**: The trajectory length of the fascicle between superficial and deep aponeurotic boundaries.
3. **Muscle Thickness ($\mathbf{MT}$ in millimeters)**: The perpendicular physical distance separating superficial and deep aponeuroses.

```
UMUD Dataset Structure
├── apo_imgs/          (1,048 TIF images — Aponeurosis training pool)
├── apo_masks/         (1,048 TIF masks  — Binary aponeurosis ground truth)
├── fasc_imgs/         (2,761 TIF images — Fascicle training pool)
├── fasc_masks/        (2,761 TIF masks  — Binary fascicle ground truth)
└── test_imgs/         (  309 TIF/PNG images — Unlabeled competition evaluation set)
```

---

## 2. Quantitative Census & Cohort Partitioning

### 2.1 File Counts & Overlap Matrix

A critical discovery of our Phase 1 Data Audit is that **100% of images in `apo_imgs` also exist within `fasc_imgs`**:

| Modality / Split | Raw Image Count | Mask Annotation Count | File Format | Bit Depth | Color Channels |
|---|---|---|---|---|---|
| **Aponeurosis Pool (`apo_imgs`)** | 1,048 | 1,048 | `.tif` | 8-bit uint8 | RGB (grayscale container) |
| **Fascicle Pool (`fasc_imgs`)** | 2,761 | 2,761 | `.tif` | 8-bit uint8 | Grayscale / RGB |
| **Shared Dual-Supervised Pool** | **1,048 (100% of apo)** | **1,048 paired** | `.tif` | 8-bit uint8 | Multi-task ground truth |
| **Fascicle-Only Supervised Pool** | 1,713 (62% of fasc) | 1,713 | `.tif` | 8-bit uint8 | Fascicle-only ground truth |
| **Test Evaluation Pool (`test_imgs`)** | 309 | 0 (Hidden) | `.tif` (251), `.png` (58) | 8-bit uint8 | RGB / Grayscale |

> **Implication for Deep Learning**:
> Because 1,048 images possess simultaneous ground-truth annotations for both aponeuroses and fascicles, models can be trained with **joint dual-task supervision** (active dual decoder heads). The remaining 1,713 images supervise the fascicle segmentation head alone using masked loss gating.

---

## 3. Transducer Resolution Census & Domain Shifts

The dataset exhibits wide resolution heterogeneity stemming from multi-center acquisitions using standard cart-based ultrasound systems and handheld point-of-care (POCUS) scanners:

### 3.1 Training Resolution Breakdown

| Image Resolution ($W \times H$) | Aponeurosis Images | Fascicle Images | Primary Clinical Source |
|---|---|---|---|
| **$1200 \times 800$** | **732 (69.8%)** | **1,714 (62.1%)** | Standard High-Resolution Cart System (50 mm depth) |
| **$1640 \times 1080$** | 0 (0.0%) | **645 (23.4%)** | Wide-Field Extended View Transducer (45 mm depth) |
| **$760 \times 500$** | 0 (0.0%) | **216 (7.8%)** | Mid-Field Diagnostic Scanner (50 mm depth) |
| **$512 \times 512$** | **188 (17.9%)** | **79 (2.9%)** | Square Crop / Research Preset (40 mm depth) |
| **$800 \times 652$** | 76 (7.3%) | 42 (1.5%) | Curvilinear / Convex Diagnostic Array |
| **$800 \times 600$** | 15 (1.4%) | 65 (2.4%) | Standard SVGA Legacy Output |
| **$660 \times 556$** | 23 (2.2%) | 0 (0.0%) | Specialized Probe Window |
| **$1234 \times 688$** | 13 (1.2%) | 0 (0.0%) | Wide Aspect Array |
| **$1074 \times 688$** | 1 (0.1%) | 0 (0.0%) | Panoramic Acquisition |

### 3.2 Test Set Resolution Breakdown & Out-of-Distribution Domain Shift

The 309 test images include significant resolution shifts not represented in the training set:

| Test Resolution ($W \times H$) | Image Count | % of Test Set | Domain Characteristics |
|---|---|---|---|
| **$1200 \times 800$** | **239** | **77.3%** | In-distribution high-resolution cart system |
| **$1088 \times 644$** | **50** | **16.2%** | Moderate domain shift (unseen training aspect ratio) |
| **$1069 \times 853$** | **11** | **3.6%** | High aspect ratio shift |
| **$\sim 465 \times 513$** (e.g. 461x513, 465x513) | **7** | **2.3%** | **Severe Domain Shift: Philips Lumify handheld scanner** |
| **$959 \times 853$** | **1** | **0.3%** | Out-of-distribution aspect ratio |
| **Total Test Images** | **309** | **100.0%** | Evaluated on hidden leaderboard |

---

## 4. Key Data Artifacts & Engineering Pitfalls

### 4.1 Resolution Mismatch Between Images and Ground-Truth Masks
- **Observed Behavior**: In many training instances, mask dimensions do not match the corresponding raw image dimensions (e.g., a $1200 \times 800$ raw image paired with an $1152 \times 864$ ground-truth mask).
- **Engineering Fix**: The data loader must dynamically resize masks to the active image shape (or standard model input shape) using **Nearest-Neighbor interpolation (`cv2.INTER_NEAREST`)**. Bilinear interpolation introduces non-binary intermediate values ($0 < x < 255$) that corrupt 1-pixel thin fascicle trajectories.

### 4.2 Inverted Masks (White Background Anomaly)
- **Observed Behavior**: Approximately $4\%$ of aponeurosis masks have inverted polarity, where background pixels are stored as $255$ and aponeurosis foreground pixels are stored as $0$ (indicated by $> 50\%$ mean mask intensity).
- **Engineering Fix**: Dynamic polarity normalization during loading:
  ```python
  if (mask > 128).mean() > 0.50:
      mask = 255 - mask  # Invert back to standard foreground=255, background=0
  ```

### 4.3 Aponeurosis Mask Topology
- **Dual-Band Structure**: Natural anatomy contains two distinct aponeurotic bands (Superficial at top, Deep at bottom).
- **Single vs. Two-Component Split**: Roughly $50\%$ of ground-truth aponeurosis masks merge superficial and deep bands into a single connected component or provide only one band. Robust pipelines employ midline spatial splitting ($y < \bar{y}$ vs. $y \ge \bar{y}$) when connected component labeling detects $N < 2$ components.

### 4.4 Fascicle Line Sparsity
- **Geometric Nature**: Fascicle annotations are continuous 1-pixel-wide lines spanning from superficial to deep aponeurosis.
- **Extreme Class Imbalance**: Positive pixel ratio is typically $< 0.8\%$ of total image area. Standard binary cross-entropy (BCE) causes models to predict all-zero masks. Mitigation requires high positive class weights ($pw \in [4.0, 8.0]$), Tversky loss ($\beta=0.7$), or structure-tensor gradient field alternatives.

---

## 5. Physical Scale Calibration ($\text{mm} / \text{pixel}$)

Because ground-truth targets ($FL$, $MT$) are evaluated in millimeters, converting pixel distances requires accurate scale factors:

### 5.1 Multi-Tier Calibration Engine
1. **Tier 1 (OCR & Tick Pitch Spacing)**: Border columns are scanned for periodic intensity peaks corresponding to $10\text{ mm}$ (or $5\text{ mm}$) physical ruler marks using `scipy.signal.find_peaks`. Spacing $\Delta y$ yields:
   $$\text{mm\_per\_px} = \frac{10.0}{\Delta y}$$
2. **Tier 2 (Transducer Depth Mapping)**: Transducer acquisition depth (40, 45, or 50 mm) divided by active scan sector height $H_{\text{scan}}$:
   $$\text{mm\_per\_px} = \frac{\text{Depth}_{\text{mm}}}{H_{\text{scan}}}$$
3. **Tier 3 (Resolution Fallback Profiles)**:
   - $1200 \times 800 \rightarrow 50.0 / 720 \approx 0.0694\text{ mm/px}$
   - $1640 \times 1080 \rightarrow 45.0 / 980 \approx 0.0459\text{ mm/px}$
   - $1088 \times 644 \rightarrow 45.0 / 600 \approx 0.0750\text{ mm/px}$
   - $465 \times 513 \rightarrow 45.0 / 460 \approx 0.0978\text{ mm/px}$
4. **Safety Clamping**: Due to OCR decimal errors (reading $0.065$ as $0.65$), all calibrated values are strictly bounded within physiological limits:
   $$\text{mm\_per\_px} \in [0.025, 0.150]$$

---

## 6. Target Variable Definitions & Bounds

| Parameter | Unit | Anatomical Meaning | Typical Adult Range | Safety Bounds |
|---|---|---|---|---|
| **Pennation Angle ($\mathbf{PA}$)** | Degrees ($^\circ$) | Angle between muscle fascicle and deep aponeurosis | $8^\circ - 30^\circ$ | $[2.0^\circ, 45.0^\circ]$ |
| **Muscle Thickness ($\mathbf{MT}$)** | Millimeters ($\text{mm}$) | Orthogonal distance between superficial and deep aponeuroses | $10\text{ mm} - 35\text{ mm}$ | $[5.0\text{ mm}, 60.0\text{ mm}]$ |
| **Fascicle Length ($\mathbf{FL}$)** | Millimeters ($\text{mm}$) | Fiber bundle length: $FL = MT / \sin(PA)$ | $35\text{ mm} - 120\text{ mm}$ | $[15.0\text{ mm}, 250.0\text{ mm}]$ |

---

## 7. Associated Metadata Files

- `dataset_metadata.csv`: Table of 3,070 records detailing `image_id`, `split`, mask presence flags, resolution, and cluster groups.
- `universal_scale_metadata_v2.csv`: Comprehensive OCR scale audit detailing tick counts, ruler spans, and extracted depths for all training images.
- `ground_truth_pseudo_labels.csv`: Reference geometric morphometry derived from ground-truth masks for supervised validation.

