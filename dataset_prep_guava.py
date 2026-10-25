#!/usr/bin/env python3
"""Dataset preparation for Guava Damage Detection.

Merges and remaps:
- RF-04 (chisilia/detection-guava-disease)
- RF-05 (guava-h8gxt/guava-disease-mozns)

Unified class mapping (4 classes):
  0: guava_fresh   (healthy, fresh)
  1: guava_rotten  (Phytopthora, Styler and Root, Anthracnose, fruit fly, rot)
  2: guava_bruised (Brown Spot, surface abrasion, blemish, bruised)
  3: guava_damaged (Scab, scab, damaged)

Standardized YOLO splits: 70% train, 20% val, 10% test.
Output: dataset_guava/ and dataset_guava/data.yaml.
"""
import argparse
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

UNIFIED_CLASSES = [
    "guava_fresh",
    "guava_rotten",
    "guava_bruised",
    "guava_damaged",
]

# Raw class name (normalized lowercase, whitespace/hyphen to underscore) -> unified class id
CLASS_MAPPING = {
    # 0: guava_fresh
    "healthy": 0,
    "fresh": 0,
    "guava_fresh": 0,

    # 1: guava_rotten
    "phytopthora": 1,
    "styler_and_root": 1,
    "styler-and-root": 1,
    "styler and root": 1,
    "anthracnose": 1,
    "fruit_fly": 1,
    "fruit-fly": 1,
    "fruit fly": 1,
    "rotten": 1,
    "rot": 1,
    "guava_rotten": 1,

    # 2: guava_bruised
    "brown_spot": 2,
    "brown-spot": 2,
    "brown spot": 2,
    "brownspot": 2,
    "surface_abrasion": 2,
    "surface-abrasion": 2,
    "surface abrasion": 2,
    "blemish": 2,
    "blemishes": 2,
    "bruised": 2,
    "bruise": 2,
    "bruising": 2,
    "guava_bruised": 2,

    # 3: guava_damaged
    "scab": 3,
    "scabs": 3,
    "damaged": 3,
    "damage": 3,
    "guava_damaged": 3,
}

DROPPED_CLASSES = set()


def remap_class(raw_name: str) -> Optional[int]:
    """Returns new class id (0-3) or None if class should be dropped / is unknown."""
    norm = raw_name.strip().lower()
    if norm in DROPPED_CLASSES:
        return None
    if norm in CLASS_MAPPING:
        return CLASS_MAPPING[norm]
    # Try normalized underscores
    norm_u = norm.replace(" ", "_").replace("-", "_")
    if norm_u in CLASS_MAPPING:
        return CLASS_MAPPING[norm_u]
    return None


def load_source_yaml_names(data_yaml_path: Path) -> Dict[int, str]:
    """Loads class index to class name mapping from a YOLO data.yaml file."""
    if not data_yaml_path.exists():
        return {}
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = data.get("names", [])
    if isinstance(names, list):
        return {i: str(name) for i, name in enumerate(names)}
    elif isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {}


def find_image_label_pairs(dataset_dir: Path, source_names: Dict[int, str]) -> List[Tuple[Path, List[Tuple[int, float, float, float, float]]]]:
    """Traverses dataset_dir, discovers image files and matching YOLO label files,
    remaps classes, and filters out dropped/empty annotations.
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    pairs = []

    image_paths = [p for p in dataset_dir.rglob("*") if p.suffix.lower() in valid_exts]

    for img_path in image_paths:
        label_path = None
        parts = list(img_path.parts)
        if "images" in parts:
            idx = len(parts) - 1 - parts[::-1].index("images")
            parts[idx] = "labels"
            candidate = Path(*parts).with_suffix(".txt")
            if candidate.exists():
                label_path = candidate

        if label_path is None:
            candidate = img_path.with_suffix(".txt")
            if candidate.exists():
                label_path = candidate

        boxes = []
        if label_path is not None and label_path.exists():
            with open(label_path, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts_line = line.strip().split()
                    if len(parts_line) < 5:
                        continue
                    try:
                        raw_cls_idx = int(parts_line[0])
                        cx, cy, w, h = map(float, parts_line[1:5])
                    except ValueError:
                        continue

                    raw_cls_name = source_names.get(raw_cls_idx, str(raw_cls_idx))
                    new_cls_id = remap_class(raw_cls_name)
                    if new_cls_id is not None:
                        boxes.append((new_cls_id, cx, cy, w, h))

        if boxes:
            pairs.append((img_path, boxes))

    return pairs


def create_unified_dataset(
    source_dirs: List[Path],
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, int]:
    """Merges sources, splits into train/val/test, and writes standardized YOLO dataset."""
    random.seed(seed)
    all_samples = []

    for src_dir in source_dirs:
        if not src_dir.exists():
            print(f"[WARN] Source directory does not exist: {src_dir}")
            continue

        yaml_file = src_dir / "data.yaml"
        if not yaml_file.exists():
            yaml_candidates = list(src_dir.glob("*/data.yaml"))
            if yaml_candidates:
                yaml_file = yaml_candidates[0]

        source_names = load_source_yaml_names(yaml_file)
        print(f"Source: {src_dir} | Detected classes: {source_names}")
        samples = find_image_label_pairs(src_dir, source_names)
        print(f"Found {len(samples)} valid labeled images in {src_dir}")
        all_samples.extend(samples)

    if not all_samples:
        print("[ERROR] No samples found to merge.")
        return {"total": 0}

    random.shuffle(all_samples)
    n = len(all_samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": all_samples[:n_train],
        "val": all_samples[n_train:n_train + n_val],
        "test": all_samples[n_train + n_val:],
    }

    print(f"\nSplits created: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])} (Total: {n})")

    # Clean / prepare output directory
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)

    for split_name in ["train", "val", "test"]:
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    class_counts = {name: 0 for name in UNIFIED_CLASSES}
    split_class_counts = {s: {name: 0 for name in UNIFIED_CLASSES} for s in splits}

    for split_name, samples in splits.items():
        for idx, (img_path, boxes) in enumerate(samples):
            # Unique filename: source directory + original stem + index
            src_tag = img_path.parts[-4] if len(img_path.parts) >= 4 else "src"
            out_stem = f"{src_tag}_{img_path.stem}_{idx}"
            out_img = output_dir / "images" / split_name / f"{out_stem}{img_path.suffix.lower()}"
            out_lbl = output_dir / "labels" / split_name / f"{out_stem}.txt"

            shutil.copy2(img_path, out_img)
            with open(out_lbl, "w", encoding="utf-8") as lf:
                for cls_id, cx, cy, w, h in boxes:
                    lf.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    class_name = UNIFIED_CLASSES[cls_id]
                    class_counts[class_name] += 1
                    split_class_counts[split_name][class_name] += 1

    # Standard data.yaml
    data_yaml_content = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(UNIFIED_CLASSES),
        "names": UNIFIED_CLASSES,
    }

    yaml_out_path = output_dir / "data.yaml"
    with open(yaml_out_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_content, f, sort_keys=False)

    print(f"\nDataset written to: {output_dir}")
    print(f"data.yaml written to: {yaml_out_path}")
    print("\nClass annotation counts across all splits:")
    for cls_name, count in class_counts.items():
        print(f"  {cls_name:15s}: {count}")

    print("\nPer-split class distribution:")
    for s_name, counts in split_class_counts.items():
        print(f"  [{s_name:5s}] " + ", ".join(f"{k}: {v}" for k, v in counts.items()))

    stats = {
        "total_images": n,
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        **class_counts,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Guava defect dataset preparation & remapping")
    parser.add_argument("--rf04-dir", type=str, default="raw_downloads/guava/rf04", help="Directory for RF-04")
    parser.add_argument("--rf05-dir", type=str, default="raw_downloads/guava/rf05", help="Directory for RF-05")
    parser.add_argument("--output-dir", type=str, default="dataset_guava", help="Output directory for unified dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random split seed")
    args = parser.parse_args()

    source_dirs = [Path(args.rf04_dir), Path(args.rf05_dir)]
    create_unified_dataset(source_dirs, Path(args.output_dir), seed=args.seed)


if __name__ == "__main__":
    main()
