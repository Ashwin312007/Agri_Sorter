"""Unit tests for Multi-Fruit Dataset Pipeline and Conveyor Augmentations."""

import os
from pathlib import Path
import shutil
import tempfile
from typing import List

import cv2
import numpy as np
import pytest
import yaml

from app.contracts import COMPOUND_CLASSES, CONDITIONS, FRUITS
from scripts.dataset_pipeline import (
    BBox,
    CLASS_TO_ID,
    ID_TO_CLASS,
    DATASET_CATALOG,
    ConveyorAugmentor,
    Deduplicator,
    DatasetPipeline,
    LabelMapper,
    SampleRecord,
    SyntheticDemoGenerator,
    get_catalog_entries_for_preset,
    print_catalog_sources,
)


# ==============================================================================
# TEST FIXTURES
# ==============================================================================

@pytest.fixture
def temp_dirs():
    """Create isolated temporary input and output directories."""
    tmp_in = tempfile.mkdtemp(prefix="test_data_in_")
    tmp_out = tempfile.mkdtemp(prefix="test_data_out_")
    yield Path(tmp_in), Path(tmp_out)
    shutil.rmtree(tmp_in, ignore_errors=True)
    shutil.rmtree(tmp_out, ignore_errors=True)


# ==============================================================================
# LABEL MAPPER TESTS
# ==============================================================================

def test_label_mapper_compound_classes():
    """Verify that all 20 canonical compound class strings map to their exact ID."""
    mapper = LabelMapper()
    for idx, class_name in enumerate(COMPOUND_CLASSES):
        assert mapper.map_label(class_name) == idx
        # Test case-insensitivity and hyphen replacement
        assert mapper.map_label(class_name.upper()) == idx
        assert mapper.map_label(class_name.replace("_", "-")) == idx


def test_label_mapper_synonyms():
    """Verify robust synonym mapping across various dataset nomenclature styles."""
    mapper = LabelMapper()

    synonym_cases = [
        ("fresh_apple", "apple_fresh"),
        ("Apple_Good", "apple_fresh"),
        ("rottenapple", "apple_rotten"),
        ("black_rot_apple", "apple_rotten"),
        ("apple_bruising", "apple_bruised"),
        ("apple-scab", "apple_scarred"),
        ("apple_russetting", "apple_scarred"),
        ("apple_insect_bite", "apple_bug_bite"),
        ("wormhole_apple", "apple_bug_bite"),
        ("fresh_banana", "banana_fresh"),
        ("banana_anthracnose", "banana_rotten"),
        ("bruised_banana", "banana_bruised"),
        ("banana_peel_scar", "banana_scarred"),
        ("banana_thrip", "banana_bug_bite"),
        ("fresh_orange", "orange_fresh"),
        ("citrus_green_mold", "orange_rotten"),
        ("orange_soft_bruise", "orange_bruised"),
        ("citrus_canker", "orange_scarred"),
        ("orange_fruit_fly", "orange_bug_bite"),
        ("fresh_mango", "mango_fresh"),
        ("mango_stem_end_rot", "mango_rotten"),
        ("mango_sapburn", "mango_bruised"),
        ("mango_rust_scar", "mango_scarred"),
        ("mango_weevil_bite", "mango_bug_bite"),
    ]

    for raw_label, expected_compound in synonym_cases:
        mapped_id = mapper.map_label(raw_label)
        assert mapped_id is not None, f"Failed to map: {raw_label}"
        assert ID_TO_CLASS[mapped_id] == expected_compound


def test_label_mapper_plant_pathologies():
    """Verify specific plant pathology mapping into standard conditions.

    Required pathologies: styler_rot, bacterial_blight, anthracnose, red_rust,
    fruit_fly, scab, caterpillar, lesion.
    Target fruits: guava, pomegranate, apple, orange, mango.
    """
    mapper = LabelMapper()

    # styler_rot -> rotten
    assert ID_TO_CLASS[mapper.map_label("styler_rot", fruit_hint="guava")] == "guava_rotten"
    assert ID_TO_CLASS[mapper.map_label("guava_stylar_rot")] == "guava_rotten"

    # bacterial_blight -> rotten
    assert ID_TO_CLASS[mapper.map_label("bacterial_blight", fruit_hint="pomegranate")] == "pomegranate_rotten"
    assert ID_TO_CLASS[mapper.map_label("pomegranate_bacterial_blight")] == "pomegranate_rotten"

    # anthracnose -> rotten
    assert ID_TO_CLASS[mapper.map_label("anthracnose", fruit_hint="mango")] == "mango_rotten"
    assert ID_TO_CLASS[mapper.map_label("mango_anthracnose")] == "mango_rotten"
    assert ID_TO_CLASS[mapper.map_label("guava_anthracnose")] == "guava_rotten"

    # red_rust -> scarred
    assert ID_TO_CLASS[mapper.map_label("red_rust", fruit_hint="guava")] == "guava_scarred"
    assert ID_TO_CLASS[mapper.map_label("guava_red_rust")] == "guava_scarred"

    # fruit_fly -> bug_bite
    assert ID_TO_CLASS[mapper.map_label("fruit_fly", fruit_hint="guava")] == "guava_bug_bite"
    assert ID_TO_CLASS[mapper.map_label("orange_fruit_fly")] == "orange_bug_bite"

    # scab -> scarred
    assert ID_TO_CLASS[mapper.map_label("scab", fruit_hint="apple")] == "apple_scarred"
    assert ID_TO_CLASS[mapper.map_label("guava_scab")] == "guava_scarred"

    # caterpillar -> bug_bite
    assert ID_TO_CLASS[mapper.map_label("caterpillar", fruit_hint="pomegranate")] == "pomegranate_bug_bite"
    assert ID_TO_CLASS[mapper.map_label("pomegranate_caterpillar")] == "pomegranate_bug_bite"

    # lesion -> scarred
    assert ID_TO_CLASS[mapper.map_label("lesion", fruit_hint="orange")] == "orange_scarred"
    assert ID_TO_CLASS[mapper.map_label("guava_lesions")] == "guava_scarred"

    # Target fruits: fresh checks
    assert ID_TO_CLASS[mapper.map_label("fresh_guava")] == "guava_fresh"
    assert ID_TO_CLASS[mapper.map_label("fresh_pomegranate")] == "pomegranate_fresh"
    assert ID_TO_CLASS[mapper.map_label("good", fruit_hint="guava")] == "guava_fresh"
    assert ID_TO_CLASS[mapper.map_label("healthy", fruit_hint="pomegranate")] == "pomegranate_fresh"


def test_label_mapper_fruit_hint():
    """Verify fruit_hint resolution for datasets that only label the defect name."""
    mapper = LabelMapper()

    # Labels from typical Roboflow datasets without fruit prefixes
    assert mapper.map_label("bruising", fruit_hint="apple") == CLASS_TO_ID["apple_bruised"]
    assert mapper.map_label("crack", fruit_hint="apple") == CLASS_TO_ID["apple_scarred"]
    assert mapper.map_label("insect_bite", fruit_hint="apple") == CLASS_TO_ID["apple_bug_bite"]
    assert mapper.map_label("rotten", fruit_hint="apple") == CLASS_TO_ID["apple_rotten"]
    assert mapper.map_label("russetting", fruit_hint="apple") == CLASS_TO_ID["apple_scarred"]

    # Banana fruit hint
    assert mapper.map_label("rot", fruit_hint="banana") == CLASS_TO_ID["banana_rotten"]
    assert mapper.map_label("bruise", fruit_hint="banana") == CLASS_TO_ID["banana_bruised"]

    # Mango fruit hint
    assert mapper.map_label("anthracnose", fruit_hint="mango") == CLASS_TO_ID["mango_rotten"]


def test_label_mapper_custom_override(temp_dirs):
    """Verify custom label mapping overrides loaded from YAML."""
    tmp_in, _ = temp_dirs
    custom_yaml = tmp_in / "custom_map.yaml"
    custom_yaml.write_text(
        "custom_apple_anomaly: apple_bruised\n"
        "exotic_pest_mark: mango_bug_bite\n"
    )

    mapper = LabelMapper.from_file(custom_yaml)
    assert mapper.map_label("custom_apple_anomaly") == CLASS_TO_ID["apple_bruised"]
    assert mapper.map_label("exotic_pest_mark") == CLASS_TO_ID["mango_bug_bite"]


# ==============================================================================
# DEDUPLICATOR TESTS
# ==============================================================================

def test_deduplicator(temp_dirs):
    """Verify MD5 exact duplicate and aHash perceptual duplicate detection."""
    tmp_in, _ = temp_dirs
    dedup = Deduplicator(ahash_threshold=4, hash_method="both")

    # 1. Base synthetic image
    img1 = np.full((120, 120, 3), 45, dtype=np.uint8)
    cv2.circle(img1, (60, 60), 30, (30, 40, 200), -1)
    img1_path = tmp_in / "img1.jpg"
    cv2.imwrite(str(img1_path), img1)

    md5_1 = dedup.compute_md5(img1_path)
    ahash_1 = dedup.compute_ahash(img1)

    is_dup, _ = dedup.is_duplicate(md5_1, ahash_1)
    assert not is_dup
    dedup.register(md5_1, ahash_1)

    # 2. Exact file duplicate (different file path, identical bytes)
    img2_path = tmp_in / "img2.jpg"
    shutil.copyfile(img1_path, img2_path)
    md5_2 = dedup.compute_md5(img2_path)
    ahash_2 = dedup.compute_ahash(img1)

    is_dup_exact, reason_exact = dedup.is_duplicate(md5_2, ahash_2)
    assert is_dup_exact
    assert "exact_md5" in reason_exact

    # 3. Near duplicate (recompressed or 1-pixel circle shift)
    img3 = img1.copy()
    cv2.circle(img3, (61, 60), 30, (30, 40, 200), -1)
    img3_path = tmp_in / "img3.jpg"
    cv2.imwrite(str(img3_path), img3)
    md5_3 = dedup.compute_md5(img3_path)
    ahash_3 = dedup.compute_ahash(img3)

    is_dup_near, reason_near = dedup.is_duplicate(md5_3, ahash_3)
    assert is_dup_near
    assert "ahash" in reason_near

    # 4. Completely different image (white rectangle on black background)
    img4 = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.rectangle(img4, (10, 10), (100, 100), (255, 255, 255), -1)
    img4_path = tmp_in / "img4.jpg"
    cv2.imwrite(str(img4_path), img4)
    md5_4 = dedup.compute_md5(img4_path)
    ahash_4 = dedup.compute_ahash(img4)

    is_dup_diff, _ = dedup.is_duplicate(md5_4, ahash_4)
    assert not is_dup_diff


# ==============================================================================
# SPLITTER TESTS
# ==============================================================================

def test_strictly_stratified_split(temp_dirs):
    """Verify that split strictly partitions 80% Train, 10% Val, 10% Test."""
    tmp_in, tmp_out = temp_dirs
    pipeline = DatasetPipeline(output_dir=tmp_out, raw_dir=tmp_in, split_ratio=(0.8, 0.1, 0.1), seed=42)

    # Test with varying sample counts
    for sample_count in [20, 50, 100, 175]:
        dummy_samples: List[SampleRecord] = []
        for i in range(sample_count):
            dummy_samples.append(SampleRecord(
                image_path=Path(f"/fake/img_{i}.jpg"),
                bboxes=[BBox(class_id=i % 20, cx=0.5, cy=0.5, w=0.2, h=0.2)],
                md5_hash=f"hash_{i}",
                ahash=i,
                primary_class=i % 20,
            ))

        train, val, test = pipeline.split_strictly_stratified(dummy_samples)

        expected_train = int(round(sample_count * 0.8))
        expected_val = int(round(sample_count * 0.1))
        expected_test = sample_count - expected_train - expected_val

        assert len(train) == expected_train
        assert len(val) == expected_val
        assert len(test) == expected_test
        assert len(train) + len(val) + len(test) == sample_count

        # Ensure no overlap
        train_hashes = {s.md5_hash for s in train}
        val_hashes = {s.md5_hash for s in val}
        test_hashes = {s.md5_hash for s in test}
        assert train_hashes.isdisjoint(val_hashes)
        assert train_hashes.isdisjoint(test_hashes)
        assert val_hashes.isdisjoint(test_hashes)


# ==============================================================================
# CONVEYOR AUGMENTATION TESTS
# ==============================================================================

def test_conveyor_augmentations():
    """Verify motion blur, lighting shifts, and affine rotation/scaling with bbox transformations."""
    augmentor = ConveyorAugmentor(
        blur_kernel_max=11,
        brightness_range=(-20, 20),
        contrast_range=(0.8, 1.2),
        rotation_range=(-45.0, 45.0),
        scale_range=(0.9, 1.1),
    )

    test_img = np.full((640, 640, 3), 42, dtype=np.uint8)
    cv2.circle(test_img, (320, 320), 80, (25, 140, 240), -1)

    orig_bboxes = [
        BBox(class_id=0, cx=0.5, cy=0.5, w=0.25, h=0.25),
        BBox(class_id=1, cx=0.3, cy=0.3, w=0.15, h=0.15),
    ]

    # 1. Motion blur test
    blurred = augmentor.apply_motion_blur(test_img, kernel_size=9, angle_deg=5.0)
    assert blurred.shape == test_img.shape
    assert blurred.dtype == np.uint8

    # 2. Lighting shift test
    lighted = augmentor.apply_lighting_shift(test_img, alpha=1.1, beta=15.0)
    assert lighted.shape == test_img.shape
    assert lighted.dtype == np.uint8

    # 3. Affine transform (rotation + scale) test
    aug_img, aug_bboxes = augmentor.apply_affine_transform(test_img, orig_bboxes, angle_deg=35.0, scale=1.05)
    assert aug_img.shape == test_img.shape
    assert len(aug_bboxes) == 2

    for b in aug_bboxes:
        # Check normalized range
        assert 0.0 <= b.cx <= 1.0
        assert 0.0 <= b.cy <= 1.0
        assert 0.0 < b.w <= 1.0
        assert 0.0 < b.h <= 1.0

    # 4. Full stochastic pipeline
    stoch_img, stoch_bboxes = augmentor.augment(test_img, orig_bboxes)
    assert stoch_img.shape == test_img.shape
    assert len(stoch_bboxes) > 0


# ==============================================================================
# SYNTHETIC DEMO GENERATOR TESTS
# ==============================================================================

def test_synthetic_generator():
    """Verify synthetic generation across all 4 fruits and 5 conditions."""
    generator = SyntheticDemoGenerator(img_size=640)
    import random
    rng = random.Random(1337)

    for fruit in FRUITS:
        for condition in CONDITIONS:
            img, bbox = generator.generate_sample(fruit, condition, rng)

            assert img.shape == (640, 640, 3)
            assert img.dtype == np.uint8

            # Verify class ID matches compound class
            expected_class = f"{fruit}_{condition}"
            assert bbox.class_id == CLASS_TO_ID[expected_class]

            # Verify bbox normalization and non-zero dimensions
            assert 0.0 < bbox.cx < 1.0
            assert 0.0 < bbox.cy < 1.0
            assert 0.0 < bbox.w < 1.0
            assert 0.0 < bbox.h < 1.0


# ==============================================================================
# DISTRIBUTION & WARNINGS TESTS
# ==============================================================================

def test_distribution_and_warning_threshold(temp_dirs):
    """Verify warning detection when classes have fewer than 300 samples."""
    tmp_in, tmp_out = temp_dirs
    pipeline = DatasetPipeline(
        output_dir=tmp_out,
        raw_dir=tmp_in,
        min_class_threshold=300,
    )

    # Create dummy distribution where some classes exceed 300 and some are below
    dist = {i: {"train": 0, "val": 0, "test": 0, "total": 0} for i in range(len(COMPOUND_CLASSES))}

    # Make class 0 have 350 samples (OK)
    dist[0] = {"train": 280, "val": 35, "test": 35, "total": 350}
    # Make class 1 have 50 samples (LOW)
    dist[1] = {"train": 40, "val": 5, "test": 5, "total": 50}

    warnings = pipeline.print_distribution_and_warnings(dist)

    # Class 0 should not be in warnings; class 1 and others (with total 0) should be
    warned_names = " ".join(warnings)
    assert "apple_fresh" not in warned_names
    assert "apple_rotten" in warned_names


# ==============================================================================
# END-TO-END PIPELINE INTEGRATION TEST
# ==============================================================================

def test_end_to_end_synthetic_pipeline(temp_dirs):
    """Run full synthetic demo end-to-end and inspect generated dataset artifacts."""
    tmp_in, tmp_out = temp_dirs
    pipeline = DatasetPipeline(
        output_dir=tmp_out,
        raw_dir=tmp_in,
        min_class_threshold=300,
        split_ratio=(0.8, 0.1, 0.1),
        augment_train=True,
        augment_factor=1,
        seed=42,
    )

    # Run synthetic demo with 40 images
    pipeline.run_synthetic_demo(count=40)

    # Verify directory structure
    assert (tmp_out / "data.yaml").is_file()
    for split in ["train", "val", "test"]:
        assert (tmp_out / "images" / split).is_dir()
        assert (tmp_out / "labels" / split).is_dir()

    # Verify data.yaml contents
    with open(tmp_out / "data.yaml", "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    assert data_cfg["nc"] == 20
    assert len(data_cfg["names"]) == 20
    assert data_cfg["names"][0] == "apple_fresh"
    assert data_cfg["names"][19] == "mango_bug_bite"

    # Verify that label files are non-empty and well-formed
    train_labels = list((tmp_out / "labels" / "train").glob("*.txt"))
    assert len(train_labels) > 0
    with open(train_labels[0], "r", encoding="utf-8") as lf:
        lines = lf.readlines()
        assert len(lines) >= 1
        parts = lines[0].strip().split()
        assert len(parts) == 5
        cls_id = int(parts[0])
        assert 0 <= cls_id < 20


# ==============================================================================
# DATASET CATALOG & PRESET TESTS
# ==============================================================================

def test_dataset_catalog_18_user_sources():
    """Verify that all 18 user-provided Roboflow and Kaggle datasets are registered."""
    expected_roboflow = [
        "rf_guava_wedlv",
        "rf_guava_thao",
        "rf_apple_defect",
        "rf_apple_surface",
        "rf_mango_defect",
        "rf_mango_grading",
        "rf_orange_defect",
        "rf_orange_inspection",
        "rf_pomegranate_gq",
        "rf_pomegranate_rj",
    ]
    expected_kaggle = [
        "kg_fruit_disease",
        "kg_fruits_diseases",
        "kg_fresh_rotten",
        "kg_fresh_stale",
        "kg_pomegranate_disease",
        "kg_guava_dataset",
        "kg_orange_dataset",
        "kg_fruits_360",
    ]

    for rf_id in expected_roboflow:
        assert rf_id in DATASET_CATALOG, f"Missing Roboflow dataset: {rf_id}"
        entry = DATASET_CATALOG[rf_id]
        assert entry.source_type == "roboflow"
        assert entry.workspace is not None
        assert entry.project is not None
        assert entry.url.startswith("https://universe.roboflow.com")
        assert entry.license == "CC BY 4.0"

    for kg_id in expected_kaggle:
        assert kg_id in DATASET_CATALOG, f"Missing Kaggle dataset: {kg_id}"
        entry = DATASET_CATALOG[kg_id]
        assert entry.source_type == "kaggle"
        assert entry.slug is not None
        assert entry.url.startswith("https://www.kaggle.com/datasets")

    # Local apple defects should also be preserved
    assert "local_apple_defects" in DATASET_CATALOG
    assert DATASET_CATALOG["local_apple_defects"].source_type == "local"


def test_dataset_presets():
    """Verify filtering catalog sources by preset."""
    # Guava preset: rf_guava_wedlv, rf_guava_thao, kg_guava_dataset
    guava_entries = get_catalog_entries_for_preset("guava")
    guava_ids = {e.id for e in guava_entries}
    assert "rf_guava_wedlv" in guava_ids
    assert "rf_guava_thao" in guava_ids
    assert "kg_guava_dataset" in guava_ids

    # Pomegranate preset
    pom_entries = get_catalog_entries_for_preset("pomegranate")
    pom_ids = {e.id for e in pom_entries}
    assert "rf_pomegranate_gq" in pom_ids
    assert "rf_pomegranate_rj" in pom_ids
    assert "kg_pomegranate_disease" in pom_ids

    # Apple preset
    apple_entries = get_catalog_entries_for_preset("apple")
    apple_ids = {e.id for e in apple_entries}
    assert "rf_apple_defect" in apple_ids
    assert "rf_apple_surface" in apple_ids
    assert "local_apple_defects" in apple_ids

    # All preset returns all 18 + local (19 entries)
    all_entries = get_catalog_entries_for_preset("all")
    assert len(all_entries) >= 18
