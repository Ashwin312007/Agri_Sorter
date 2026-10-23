# Agri_Sorter Multi-Fruit Dataset Pipeline & Conveyor Augmentations

This directory houses the dataset pipeline, schema definitions, catalog registry, and data ingestion workflows for the **Agri_Sorter** automated multi-fruit defect sorting system.

The pipeline prepares high-quality YOLO-formatted datasets targeting real-time edge inference on **Raspberry Pi 4** and the **Hailo-8L AI HAT** (13 TOPS INT8) accelerator.

---

## 1. Multi-Fruit Class Taxonomy & Supported Fruits

Agri_Sorter classifies target fruits across five distinct conditions into standardized compound classes:

$$\text{Class Name} = \langle\text{fruit}\rangle\_\langle\text{condition}\rangle$$

### Supported Fruits:
1. **Apple** (`apple`)
2. **Guava** (`guava`)
3. **Orange** (`orange`)
4. **Mango** (`mango`)
5. **Pomegranate** (`pomegranate`)
6. **Banana** (`banana` - legacy support)

### Standard Conditions:
- `fresh` (PASS - Fruit proceeds along primary conveyor to packaging)
- `rotten` (REJECT - Active fungal decay, bacterial blight, stylar rot, mold, soft rot)
- `bruised` (REJECT - Impact depression, mechanical compression, sapburn)
- `scarred` (REJECT - Russeting, corky scab, red algal rust, canker, healed skin lesions)
- `bug_bite` (REJECT - Insect puncture, fruit fly oviposition, caterpillar borer hole, larvae)

### Canonical Compound Class Table (20 Primary Edge Classes):
| Class ID | Compound Class Name | Fruit Type | Condition | Sorter Decision | Action / Mechanism |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **0** | `apple_fresh` | Apple | Fresh | **PASS** (FRESH) | Proceeds along main conveyor to packaging |
| **1** | `apple_rotten` | Apple | Rotten | **REJECT** | Diverter servo actuated to reject chute |
| **2** | `apple_bruised` | Apple | Bruised | **REJECT** | Diverter servo actuated to reject chute |
| **3** | `apple_scarred` | Apple | Scarred | **REJECT** | Diverter servo actuated to reject chute |
| **4** | `apple_bug_bite` | Apple | Bug Bite | **REJECT** | Diverter servo actuated to reject chute |
| **5** | `guava_fresh` | Guava | Fresh | **PASS** (FRESH) | Proceeds along main conveyor to packaging |
| **6** | `guava_rotten` | Guava | Rotten | **REJECT** | Diverter servo actuated to reject chute |
| **7** | `guava_bruised` | Guava | Bruised | **REJECT** | Diverter servo actuated to reject chute |
| **8** | `guava_scarred` | Guava | Scarred | **REJECT** | Diverter servo actuated to reject chute |
| **9** | `guava_bug_bite` | Guava | Bug Bite | **REJECT** | Diverter servo actuated to reject chute |
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

*Extended Classes (IDs 20–29)*: Pomegranate (`pomegranate_fresh` .. `pomegranate_bug_bite`) and legacy Banana (`banana_fresh` .. `banana_bug_bite`) are registered in `ALL_COMPOUND_CLASSES` and dynamically indexed when running with `--dataset-preset all` or `--dataset-preset pomegranate`.

---

## 2. User-Provided Dataset Catalog (18 Registered Sources + Local)

The dataset catalog registered in [`scripts/dataset_pipeline.py`](file:///home/taiki/Agri_Sorter/scripts/dataset_pipeline.py) strictly integrates the user-specified datasets from Roboflow Universe and Kaggle:

### A. Roboflow Universe Datasets (10 Sources)
| Catalog ID | Workspace | Project | Target Fruit | Preset | License | Universe Web URL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `rf_guava_wedlv` | `guava-wedlv` | `guava-3hezg` | Guava | `guava` | CC BY 4.0 | [https://universe.roboflow.com/guava-wedlv/guava-3hezg](https://universe.roboflow.com/guava-wedlv/guava-3hezg) |
| `rf_guava_thao` | `thao-ulmra` | `guava-qnkee-1uz3i` | Guava | `guava` | CC BY 4.0 | [https://universe.roboflow.com/thao-ulmra/guava-qnkee-1uz3i](https://universe.roboflow.com/thao-ulmra/guava-qnkee-1uz3i) |
| `rf_apple_defect` | `appledefectsdetection` | `apple-defect-detection-pkekm` | Apple | `apple` | CC BY 4.0 | [https://universe.roboflow.com/appledefectsdetection/apple-defect-detection-pkekm](https://universe.roboflow.com/appledefectsdetection/apple-defect-detection-pkekm) |
| `rf_apple_surface` | `applesort-esgll` | `apple_surface_defects` | Apple | `apple` | CC BY 4.0 | [https://universe.roboflow.com/applesort-esgll/apple_surface_defects](https://universe.roboflow.com/applesort-esgll/apple_surface_defects) |
| `rf_mango_defect` | `kaycee-kaye-y-villanueva-mzmks` | `mango-defect-detection` | Mango | `mango` | CC BY 4.0 | [https://universe.roboflow.com/kaycee-kaye-y-villanueva-mzmks/mango-defect-detection](https://universe.roboflow.com/kaycee-kaye-y-villanueva-mzmks/mango-defect-detection) |
| `rf_mango_grading` | `mangoclass` | `mangograding` | Mango | `mango` | CC BY 4.0 | [https://universe.roboflow.com/mangoclass/mangograding](https://universe.roboflow.com/mangoclass/mangograding) |
| `rf_orange_defect` | `mydetect-o2wvw` | `orange-defect` | Orange | `orange` | CC BY 4.0 | [https://universe.roboflow.com/mydetect-o2wvw/orange-defect](https://universe.roboflow.com/mydetect-o2wvw/orange-defect) |
| `rf_orange_inspection` | `marcel-craig` | `orange-inspection` | Orange | `orange` | CC BY 4.0 | [https://universe.roboflow.com/marcel-craig/orange-inspection](https://universe.roboflow.com/marcel-craig/orange-inspection) |
| `rf_pomegranate_gq` | `orin-f84rr` | `pomegranate-gqopm` | Pomegranate | `pomegranate` | CC BY 4.0 | [https://universe.roboflow.com/orin-f84rr/pomegranate-gqopm](https://universe.roboflow.com/orin-f84rr/pomegranate-gqopm) |
| `rf_pomegranate_rj` | `projects-jn6uf` | `pomegranate-rjwdq` | Pomegranate | `pomegranate` | CC BY 4.0 | [https://universe.roboflow.com/projects-jn6uf/pomegranate-rjwdq](https://universe.roboflow.com/projects-jn6uf/pomegranate-rjwdq) |

### B. Kaggle Datasets (8 Sources)
| Catalog ID | Kaggle Slug | Target Fruit | Preset | License | Kaggle Web URL |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `kg_fruit_disease` | `ateebnoone/fruits-dataset-for-fruit-disease-classification` | Multi | `all` | CC0 | [Kaggle fruits-dataset-for-fruit-disease-classification](https://www.kaggle.com/datasets/ateebnoone/fruits-dataset-for-fruit-disease-classification) |
| `kg_fruits_diseases` | `amdatas/fruits-diseases-classification-dataset` | Multi | `all` | CC BY-SA 4.0 | [Kaggle fruits-diseases-classification-dataset](https://www.kaggle.com/datasets/amdatas/fruits-diseases-classification-dataset) |
| `kg_fresh_rotten` | `sriramr/fruits-fresh-and-rotten-for-classification` | Multi | `all` | CC BY-SA 4.0 | [Kaggle fruits-fresh-and-rotten-for-classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification) |
| `kg_fresh_stale` | `swoyam2609/fresh-and-stale-classification` | Multi | `all` | CC0 | [Kaggle fresh-and-stale-classification](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification) |
| `kg_pomegranate_disease` | `sujaykapadnis/pomegranate-fruit-diseases-dataset` | Pomegranate | `pomegranate` | CC BY 4.0 | [Kaggle pomegranate-fruit-diseases-dataset](https://www.kaggle.com/datasets/sujaykapadnis/pomegranate-fruit-diseases-dataset) |
| `kg_guava_dataset` | `noamaanabdulazeem/guava-dataset` | Guava | `guava` | CC0 | [Kaggle guava-dataset](https://www.kaggle.com/datasets/noamaanabdulazeem/guava-dataset) |
| `kg_orange_dataset` | `sgandhi2003/orange-fruit-dataset` | Orange | `orange` | CC BY 4.0 | [Kaggle orange-fruit-dataset](https://www.kaggle.com/datasets/sgandhi2003/orange-fruit-dataset) |
| `kg_fruits_360` | `moltean/fruits` | Multi | `all` | CC BY-SA 4.0 | [Kaggle fruits](https://www.kaggle.com/datasets/moltean/fruits) |

### C. Preserved Local Dataset
- **Local Apple Defects**: Located at `data/external/apple_defects` (Roboflow project `jaiveer-singh-outlook-in/apple-defects`, MIT License). Ingested automatically by preset `apple` or `all`.

---

## 3. Plant Pathology Mapping Logic

Incoming datasets label biological diseases and plant pathologies rather than industrial conveyor conditions. The pipeline includes an expanded `LabelMapper` ontology that automatically maps plant diseases into standard sorter conditions:

| Plant Pathology / Anomaly | Botanical / Agricultural Cause | Sorter Condition | Sorter Compound Class Example |
| :--- | :--- | :---: | :--- |
| **`styler_rot`** / `stylar_rot` | *Phomopsis psidii* / *Guignardia psidii* fungal end rot on guava | **`rotten`** | `guava_rotten` |
| **`bacterial_blight`** | *Xanthomonas axonopodis* necrotic brown spots on pomegranate | **`rotten`** | `pomegranate_rotten` |
| **`anthracnose`** | *Colletotrichum gloeosporioides* dark sunken decay on mango / guava | **`rotten`** | `mango_rotten` |
| **`red_rust`** | *Cephaleuros virescens* algal rust forming rough corky crusts on guava | **`scarred`** | `guava_scarred` |
| **`fruit_fly`** | *Bactrocera dorsalis* oviposition puncture holes and larval rot | **`bug_bite`** | `guava_bug_bite` |
| **`scab`** | *Venturia inaequalis* / *Pestalotiopsis* rough dark corky lesions | **`scarred`** | `apple_scarred` |
| **`caterpillar`** | *Deudorix isocrates* borer entry holes on pomegranate / mango | **`bug_bite`** | `pomegranate_bug_bite` |
| **`lesion`** / `lesions` | Mechanical abrasion, healed surface blemishes, skin cracks | **`scarred`** | `orange_scarred` |

---

## 4. Download & Ingestion Workflows

### A. List Catalog Sources
To display a formatted table of all 18 registered catalog datasets, direct URLs, and licenses:
```bash
python3 scripts/dataset_pipeline.py --list-sources
```

### B. Ingest by Preset Flag (`--dataset-preset`)
The `--dataset-preset` flag filters and downloads datasets by fruit group:
- `all`: Ingests all 18 registered datasets + local apple dataset.
- `apple`: Ingests `rf_apple_defect`, `rf_apple_surface`, and `data/external/apple_defects`.
- `guava`: Ingests `rf_guava_wedlv`, `rf_guava_thao`, and `kg_guava_dataset`.
- `orange`: Ingests `rf_orange_defect`, `rf_orange_inspection`, and `kg_orange_dataset`.
- `mango`: Ingests `rf_mango_defect` and `rf_mango_grading`.
- `pomegranate`: Ingests `rf_pomegranate_gq`, `rf_pomegranate_rj`, and `kg_pomegranate_disease`.

```bash
# Ingest all guava datasets with Roboflow API key
export ROBOFLOW_API_KEY="your_roboflow_key"
python3 scripts/dataset_pipeline.py --dataset-preset guava --output-dir data/merged
```

### C. Direct Roboflow Universe Ingestion
```bash
python3 scripts/dataset_pipeline.py \
    --roboflow-workspace guava-wedlv \
    --roboflow-project guava-3hezg \
    --roboflow-key $ROBOFLOW_API_KEY \
    --fruit-hint guava \
    --output-dir data/merged
```

### D. Direct Kaggle Dataset Ingestion
Ensure your Kaggle API key is installed at `~/.kaggle/kaggle.json`:
```bash
python3 scripts/dataset_pipeline.py \
    --kaggle-datasets sujaykapadnis/pomegranate-fruit-diseases-dataset \
    --fruit-hint pomegranate \
    --output-dir data/merged
```

### E. Offline Synthetic Verification Demo
Run verified end-to-end dataset generation and splitting offline without requiring any external tokens or network connection:
```bash
python3 scripts/dataset_pipeline.py --synthetic-demo --synthetic-count 100 --output-dir data/merged
```

---

## 5. Conveyor Belt Physical Augmentations

Field conveyor belts introduce optical and mechanical variations absent in clean lab photo sessions. Agri_Sorter applies conveyor-specific physical transformations to the training split:

```
+-------------------------------------------------------------+
|                      CONVEYOR BELT ROI                      |
|                                                             |
|   ---> [ Motion Blur: Linear transit along belt axis ] --->  |
|                                                             |
|   * [ Lighting Shifts: Industrial overhead flicker / shadow ]|
|   o [ Tumbling Rotation: Rolling as fruit settles on belt ]  |
|   ^ [ Caliber / Distance Scaling: Camera-to-surface depth ]  |
+-------------------------------------------------------------+
```

| Transformation | Parameter Range | Physical Simulation |
| :--- | :--- | :--- |
| **Motion Blur** | Kernel $k \in \{3, 5, 7, 9, 11, 15\}$, angle $\theta \in [-8^\circ, +8^\circ]$ | Simulates linear transit velocity ($v \approx 0.12\text{ m/s}$) along conveyor axis. |
| **Lighting Shifts** | Contrast $\alpha \in [0.75, 1.25]$, Brightness $\beta \in [-30, +30]$ | Simulates overhead warehouse fluorescent/LED flicker and ambient shadow transitions. |
| **Fruit Tumbling & Rolling** | Rotation angle $\theta \in [-180^\circ, +180^\circ]$ around image center | Simulates tumbling and rolling into varied orientations as fruit settles on the belt. |
| **Distance & Caliber Scaling** | Scale factor $s \in [0.85, 1.15]$ | Simulates fruit caliber variations and camera mounting depth shifts. |

---

## 6. Deduplication & Stratified 80/10/10 Split

1. **Dual Hashing Deduplication**:
   - **MD5 Checksum**: Removes exact binary duplicates across overlapping repositories.
   - **64-bit Average Hash (aHash)**: Resizes grayscale images to $8 \times 8$, calculates luminance mean, and evaluates Hamming distance $\text{dist} \le 4$ bits to detect and eliminate near-duplicate frames.

2. **Strict Stratified Split**:
   - Unique data is strictly partitioned into:
     - **80% Training Split** (`images/train/`, `labels/train/`)
     - **10% Validation Split** (`images/val/`, `labels/val/`)
     - **10% Test Split** (`images/test/`, `labels/test/`)
   - Guarantees zero data leakage and balanced class ratios.

---

## 7. Quality Assurance: Minimum 300 Images Threshold

To ensure robust INT8 quantization and high recall on the **Hailo-8L AI HAT**:
- **Minimum Threshold**: Every compound class requires at least **300 images**.
- The pipeline scans all splits, displays an ASCII distribution summary table, and issues explicit warnings for underrepresented classes (< 300 samples) prompting targeted synthetic augmentation or additional camera footage collection.
