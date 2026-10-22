# Agri_Sorter Multi-Fruit Dataset Pipeline & Conveyor Augmentations

This directory houses the dataset pipeline, schema definitions, and raw data integration workflows for the **Agri_Sorter** automated multi-fruit defect sorting system.

The pipeline prepares high-quality YOLO-formatted datasets targeting real-time edge inference on **Raspberry Pi 4** and **Hailo-8L AI HAT** accelerators.

---

## 1. Multi-Fruit Class Taxonomy (20 Compound Classes)

Agri_Sorter classifies four fruit types across five distinct conditions into **20 compound classes** as defined in [`app/contracts.py`](file:///home/taiki/Agri_Sorter/app/contracts.py):

$$\text{Class Name} = \langle\text{fruit}\rangle\_\langle\text{condition}\rangle$$

| Class ID | Compound Class Name | Fruit Type | Condition | Sorter Decision | Action / Mechanism |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **0** | `apple_fresh` | Apple | Fresh | **PASS** (FRESH) | Proceeds along main conveyor to packaging |
| **1** | `apple_rotten` | Apple | Rotten | **REJECT** | Diverter servo actuated to reject chute |
| **2** | `apple_bruised` | Apple | Bruised | **REJECT** | Diverter servo actuated to reject chute |
| **3** | `apple_scarred` | Apple | Scarred | **REJECT** | Diverter servo actuated to reject chute |
| **4** | `apple_bug_bite` | Apple | Bug Bite | **REJECT** | Diverter servo actuated to reject chute |
| **5** | `banana_fresh` | Banana | Fresh | **PASS** (FRESH) | Proceeds along main conveyor to packaging |
| **6** | `banana_rotten` | Banana | Rotten | **REJECT** | Diverter servo actuated to reject chute |
| **7** | `banana_bruised` | Banana | Bruised | **REJECT** | Diverter servo actuated to reject chute |
| **8** | `banana_scarred` | Banana | Scarred | **REJECT** | Diverter servo actuated to reject chute |
| **9** | `banana_bug_bite` | Banana | Bug Bite | **REJECT** | Diverter servo actuated to reject chute |
| **10** | `orange_fresh` | Orange | Fresh | **PASS** (FRESH) | Proceeds along main conveyor to packaging |
| **11** | `orange_rotten` | Orange | Rotten | **REJECT** | Diverter servo actuated to reject chute |
| **12** | `orange_bruised` | Orange | Bruised | **REJECT** | Diverter servo actuated to reject chute |
| **13** | `orange_scarred` | Orange | Scarred | **REJECT** | Diverter servo actuated to reject chute |
| **14** | `orange_bug_bite` | Orange | Bug Bite | **REJECT** | Diverter servo actuated to reject chute |
| **15** | `mango_fresh` | Mango | Fresh | **PASS** (FRESH) | Proceeds along main conveyor to packaging |
| **16** | `mango_rotten` | Mango | Rotten | **REJECT** | Diverter servo actuated to reject chute |
| **17** | `mango_bruised` | Mango | Bruised | **REJECT** | Diverter servo actuated to reject chute |
| **18** | `mango_scarred` | Mango | Scarred | **REJECT** | Diverter servo actuated to reject chute |
| **19** | `mango_bug_bite` | Mango | Bug Bite | **REJECT** | Diverter servo actuated to reject chute |

---

## 2. Public Dataset Sources & Ingestion

The pipeline (`scripts/dataset_pipeline.py`) can download and merge multiple public datasets from Roboflow Universe, Kaggle, and direct web mirrors:

### A. Roboflow Universe Datasets
- **Apple Defects (MIT License)**: [Roboflow Universe `jaiveer-singh-outlook-in/apple-defects`](https://universe.roboflow.com/jaiveer-singh-outlook-in/apple-defects)
  - Features: Rot, Bruising, Insect bites, Scars (russetting/cracks).
- **Apple Defect Detection (`apple-defects-yafjq`)**: [Roboflow Universe](https://universe.roboflow.com/)
  - Features: Fresh apples, rot, mechanical bruising.
- **Banana Quality & Defect Detection (CC-BY 4.0)**: Roboflow Universe banana ripeness & peel defect projects.
- **Citrus & Orange Disease Detection (CC-BY 4.0)**: Roboflow Universe citrus greening, canker, and mold projects.
- **Mango Quality & Anthracnose (CC-BY 4.0)**: Roboflow Universe mango surface blemish and stem rot projects.

### B. Kaggle Public Datasets
- **Fruits Fresh and Rotten (`sriramr/fruits-fresh-and-rotten-for-classification`)**:
  - Contains fresh and rotten apples, bananas, and oranges.
  - License: CC BY-SA 4.0.
- **Fresh and Stale Fruit & Vegetables (`raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables`)**:
  - Multi-class freshness dataset for commercial sorting.
  - License: CC0: Public Domain.
- **Fruit Defect Detection YOLO Dataset (`shashwatwork/fruit-defect-detection`)**:
  - Bounding box annotations for external fruit surface anomalies.
  - License: Open Data Commons Attribution License (ODC-By).

---

## 3. Dataset Licensing & Compliance

When training models with public data, comply with individual source licenses:
- **CC-BY 4.0 (Creative Commons Attribution 4.0 International)**:
  - Permits commercial use, sharing, and modification.
  - Requires explicit credit to dataset creators and notice of changes.
- **CC BY-SA 4.0 (ShareAlike)**:
  - Derivative works and modifications must be distributed under compatible terms.
- **MIT License**:
  - Permissive free software license granting unrestricted commercial and non-commercial rights.
- **CC0 1.0 (Public Domain Dedication)**:
  - Completely unrestricted worldwide use with no attribution obligations.

---

## 4. Label Mapping Logic

Incoming datasets frequently use divergent naming conventions (e.g., `russetting`, `canker`, `anthracnose`, `green_mold`). The pipeline includes a canonical synonym resolution engine (`LabelMapper`):

### Synonym & Ontology Mappings:

1. **Condition Mappings**:
   - **`fresh`**: `fresh`, `good`, `clean`, `healthy`, `sound`, `ripe`, `ripened`, `grade_a`, `normal`, `perfect`, `undamaged`.
   - **`rotten`**: `rot`, `rotten`, `decay`, `decayed`, `mold`, `moldy`, `bad`, `anthracnose`, `spoil`, `spoiled`, `breakdown`, `fungal`, `necrotic`, `black_rot`, `bitter_rot`, `stem_end_rot`, `cigar_end_rot`, `overripe`, `blight`, `mildew`.
   - **`bruised`**: `bruise`, `bruised`, `bruising`, `impact`, `pressure`, `indent`, `sapburn`, `soft`, `damage`, `squished`, `crush`, `mechanical_injury`, `mechanical_damage`.
   - **`scarred`**: `scar`, `scarred`, `scab`, `scratch`, `russet`, `russetting`, `crack`, `cracked`, `canker`, `blotch`, `blemish`, `scuff`, `rust`, `cork`, `corky`, `peel_blemish`, `wind_scar`, `skin_defect`, `melanose`, `greening`, `sigatoka`.
   - **`bug_bite`**: `bug`, `bug_bite`, `bugbite`, `bite`, `worm`, `wormhole`, `insect`, `insect_bite`, `pest`, `weevil`, `fly`, `fruit_fly`, `moth`, `codling_moth`, `puncture`, `borer`, `larva`, `thrip`, `thrips`, `mite`, `scale_insect`, `aphid`.

2. **Fruit Disambiguation & `--fruit-hint`**:
   - If incoming labels include the fruit name (e.g. `rotten_banana`, `Fresh Apple`), the compound class is deduced automatically.
   - For single-fruit datasets where labels only specify defects (e.g., `rot`, `bruise`), provide `--fruit-hint apple` (or `banana`, `orange`, `mango`).

3. **Custom Label Overrides**:
   You can provide a JSON or YAML label map override via `--label-map <path>`:
   ```yaml
   # custom_label_map.yaml
   my_custom_class_1: apple_scarred
   mold_spot: orange_rotten
   fruit_borer: mango_bug_bite
   ```

---

## 5. Conveyor Belt Physical Augmentation Suite

Field conveyor belts present optical distortions absent in clean lab datasets. Agri_Sorter applies conveyor-specific physical transformations on the training partition:

```
+-------------------------------------------------------------+
|                      CONVEYOR BELT ROI                      |
|                                                             |
|   ---> [ Motion Blur: Linear transit along belt axis ] --->  |
|                                                             |
|   * [ Lighting Shifts: Industrial overhead flicker / shadow ]|
|   o [ Tumbling Rotation: Rolling as fruit drops onto belt ]  |
|   ^ [ Caliber / Distance Scaling: Camera-to-surface depth ]  |
+-------------------------------------------------------------+
```

### Augmentation Parameters:

| Transformation | Parameter Range | Physical Simulation |
| :--- | :--- | :--- |
| **Motion Blur** | Kernel size $k \in \{3, 5, 7, 9, 11, 15\}$ (odd), angle $\theta \in [-8^\circ, +8^\circ]$ | Simulates linear transit velocity ($v \approx 0.12\text{ m/s}$) along conveyor direction. |
| **Lighting Shifts** | Contrast multiplier $\alpha \in [0.75, 1.25]$, Brightness offset $\beta \in [-30, +30]$ | Simulates overhead warehouse fluorescent/LED flicker and ambient shadow transitions. |
| **Fruit Tumbling & Rolling** | Rotation angle $\theta \in [-180^\circ, +180^\circ]$ around image center | Simulates fruit tumbling and rolling into varied orientations as it settles on the belt. |
| **Distance & Caliber Scaling** | Scale factor $s \in [0.85, 1.15]$ | Simulates fruit caliber variations and camera distance shifts. |

### Exact Affine Bounding Box Transformation Math:

Given image dimensions $(W, H)$, center $(c_x, c_y) = (W/2, H/2)$, rotation angle $\theta$, and scale $s$, the affine transformation matrix $M \in \mathbb{R}^{2 \times 3}$ is:

$$M = \begin{bmatrix} \alpha & \beta & (1-\alpha) c_x - \beta c_y \\ -\beta & \alpha & \beta c_x + (1-\alpha) c_y \end{bmatrix}$$
where $\alpha = s \cos\theta$ and $\beta = s \sin\theta$.

Each normalized bounding box $(c_x^{\text{norm}}, c_y^{\text{norm}}, w^{\text{norm}}, h^{\text{norm}})$ is unnormalized to four pixel corners:
$$C = \begin{bmatrix} x_1 & x_2 & x_2 & x_1 \\ y_1 & y_1 & y_2 & y_2 \\ 1 & 1 & 1 & 1 \end{bmatrix}$$

The rotated corners are computed via:
$$C' = M \cdot C$$

The new axis-aligned bounding box enclosing all four transformed corners is computed:
$$x_1' = \min(C'_{0, :}), \quad y_1' = \min(C'_{1, :}), \quad x_2' = \max(C'_{0, :}), \quad y_2' = \max(C'_{1, :})$$

Coordinates are strictly clamped to $[0, W]$ and $[0, H]$. If $w' < 4\text{ px}$ or $h' < 4\text{ px}$, the degenerate box is discarded; otherwise, it is converted back to normalized YOLO format.

---

## 6. Deduplication & Stratified 80/10/10 Split

1. **Dual Hashing Deduplication**:
   - **MD5 Checksum**: Removes exact identical file duplicates across merged repositories.
   - **64-bit Average Hash (aHash)**: Resizes grayscale images to $8 \times 8$, computes mean luminance, and calculates Hamming distance:
     $$\text{dist}(h_1, h_2) = \text{popcount}(h_1 \oplus h_2)$$
     Images with $\text{dist} \le 4$ bits are flagged and discarded as perceptual duplicates.

2. **Strict Stratified Split**:
   - The unique dataset is partitioned strictly into:
     - **80% Training Split** (`images/train/`, `labels/train/`)
     - **10% Validation Split** (`images/val/`, `labels/val/`)
     - **10% Test Split** (`images/test/`, `labels/test/`)
   - Stratification ensures class balance across splits with fixed random seeds (`--seed 42`).

---

## 7. Class Distribution & Minimum 300 Image Warning Threshold

For reliable fine-tuning of YOLO11n on the Hailo-8L accelerator (INT8 quantization):
- **Minimum Threshold**: Every compound class should contain at least **300 images**.
- The pipeline scans all annotations across splits and prints an ASCII distribution table.
- If any of the 20 classes has fewer than 300 images, an explicit `WARNING` banner is raised highlighting the underrepresented classes and recommending targeted synthetic augmentation or additional video capture.

---

## 8. CLI Usage Guide

### 1. Run Offline Synthetic Verification Demo (No API Key Required)
```bash
python3 scripts/dataset_pipeline.py --synthetic-demo --synthetic-count 100 --output-dir data/merged
```

### 2. Ingest Local Datasets (e.g. Existing Apple Defect Data)
```bash
python3 scripts/dataset_pipeline.py \
    --source-dirs data/external/apple_defects \
    --fruit-hint apple \
    --output-dir data/merged
```

### 3. Download from Roboflow Universe
```bash
export ROBOFLOW_API_KEY="your_api_key_here"
python3 scripts/dataset_pipeline.py \
    --roboflow-project apple-defects-yafjq \
    --fruit-hint apple \
    --output-dir data/merged
```

### 4. Download from Kaggle
```bash
# Requires ~/.kaggle/kaggle.json configured
python3 scripts/dataset_pipeline.py \
    --kaggle-datasets sriramr/fruits-fresh-and-rotten-for-classification \
    --output-dir data/merged
```

### 5. Inspect Help & Options
```bash
python3 scripts/dataset_pipeline.py --help
```
