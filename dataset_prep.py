#!/usr/bin/env python3
"""Dataset preparation for Apple Damage Detection.

Merges and remaps:
- RF-01 (Apple Surface Defects)
- RF-02 (Apple Fruit Disease Detection)

Unified class mapping (4 classes):
  0: apple_fresh   (from HEALTHY)
  1: apple_rotten  (from ROT, rotten)
  2: apple_bruised (from BLOTCH, bruising)
  3: apple_damaged (from SCAB, crack, insect_bite)

Dropped classes:
  - russetting / rusetting
  - objects

Generates standardized YOLO splits (train 70%, val 20%, test 10%) and data.yaml.
Can load from local directories or download via Roboflow API.
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
    "apple_fresh",
    "apple_rotten",
    "apple_bruised",
    "apple_damaged",
]

# Raw class name (lowercased) -> unified class id
CLASS_MAPPING = {
    "healthy": 0,
    "rot": 1,
    "rotten": 1,
    "blotch": 2,
    "bruising": 2,
    "bruised": 2,
    "scab": 3,
    "crack": 3,
    "insect_bite": 3,
    "insect-bite": 3,
    "damaged": 3,
}

DROPPED_CLASSES = {
    "russetting",
    "rusetting",
    "objects",
    "object",
}


def remap_class(raw_name: str) -> Optional[int]:
    """Returns new class id (0-3) or None if class should be dropped / is unknown."""
    norm = raw_name.strip().lower().replace(" ", "_")
    if norm in DROPPED_CLASSES:
        return None
    return CLASS_MAPPING.get(norm, None)


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

    remaps classes, and filters out dropped annotations.
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    pairs = []

    # Find all images recursively
    image_paths = [p for p in dataset_dir.rglob("*") if p.suffix.lower() in valid_exts]

    for img_path in image_paths:
        # Standard YOLO layout puts labels in sibling 'labels' directory or same directory
        # Try both:
        label_path = None
        # Option 1: replace 'images' in path with 'labels'
        parts = list(img_path.parts)
        if "images" in parts:
            idx = len(parts) - 1 - parts[::-1].index("images")
            parts[idx] = "labels"
            candidate = Path(*parts).with_suffix(".txt")
            if candidate.exists():
                label_path = candidate

        # Option 2: adjacent file with .txt
        if label_path is None:
            candidate = img_path.with_suffix(".txt")
            if candidate.exists():
                label_path = candidate

        boxes = []
        if label_path is not None and label_path.exists():
            with open(label_path, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        raw_cls_idx = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                    except ValueError:
                        continue

                    raw_cls_name = source_names.get(raw_cls_idx, str(raw_cls_idx))
                    new_cls_id = remap_class(raw_cls_name)
                    if new_cls_id is not None:
                        boxes.append((new_cls_id, cx, cy, w, h))

        # We keep the sample if it has valid annotations
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
            # search one level down
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

    # Deduplicate by image file name stem if needed, or unique prefix
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
    for split_name in ["train", "val", "test"]:
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    class_counts = {name: 0 for name in UNIFIED_CLASSES}

    for split_name, samples in splits.items():
        for idx, (img_path, boxes) in enumerate(samples):
            # Unique filename to avoid collision across sources
            out_stem = f"{img_path.parent.name}_{img_path.stem}_{idx}"
            out_img = output_dir / "images" / split_name / f"{out_stem}{img_path.suffix.lower()}"
            out_lbl = output_dir / "labels" / split_name / f"{out_stem}.txt"

            shutil.copy2(img_path, out_img)
            with open(out_lbl, "w", encoding="utf-8") as lf:
                for cls_id, cx, cy, w, h in boxes:
                    lf.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    class_counts[UNIFIED_CLASSES[cls_id]] += 1

    # Write standardized data.yaml
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
    print("Class annotation counts across all splits:")
    for cls_name, count in class_counts.items():
        print(f"  {cls_name:15s}: {count}")

    stats = {
        "total_images": n,
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        **class_counts,
    }
    return stats


def download_roboflow_dataset(api_key: str, workspace: str, project_name: str, version_num: int, dest_dir: Path) -> Path:
    """Downloads dataset from Roboflow in YOLOv5 format."""
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("roboflow package not installed. Run: pip install roboflow")

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project_name)
    version = proj.version(version_num)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dataset = version.download("yolov5", location=str(dest_dir))
    return Path(dataset.location)


def main():
    parser = argparse.ArgumentParser(description="Apple defect dataset preparation & remapping")
    parser.add_argument("--rf01-dir", type=str, default=None, help="Local directory for RF-01 (Apple Surface Defects)")
    parser.add_argument("--rf02-dir", type=str, default=None, help="Local directory for RF-02 (Apple Fruit Disease Detection)")
    parser.add_argument("--output-dir", type=str, default="dataset_apple", help="Output directory for unified dataset")
    parser.add_argument("--rf-key", type=str, default=os.environ.get("ROBOFLOW_API_KEY"), help="Roboflow API key")
    parser.add_argument("--download-rf", action="store_true", help="Download datasets from Roboflow")
    parser.add_argument("--seed", type=int, default=42, help="Random split seed")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    source_dirs = []

    if args.download_rf:
        if not args.rf_key:
            sys.exit("Error: --download-rf specified but no Roboflow API key provided via --rf-key or ROBOFLOW_API_KEY.")
        raw_downloads = Path("raw_downloads")
        print("Downloading RF-01...")
        d1 = download_roboflow_dataset(args.rf_key, "applesort-esgll", "apple_surface_defects", 1, raw_downloads / "rf01")
        source_dirs.append(d1)
        print("Downloading RF-02...")
        d2 = download_roboflow_dataset(args.rf_key, "fruit-disease-detection", "apple-fruit-disease-detection", 1, raw_downloads / "rf02")
        source_dirs.append(d2)
    else:
        if args.rf01_dir:
            source_dirs.append(Path(args.rf01_dir))
        if args.rf02_dir:
            source_dirs.append(Path(args.rf02_dir))

    if not source_dirs:
        print("[INFO] No input directories specified. Creating sample scaffold or checking default paths.")
        default_raw = Path("raw_datasets")
        if default_raw.exists():
            source_dirs = [p for p in default_raw.iterdir() if p.is_dir()]

    if not source_dirs:
        print("Please provide source datasets with --rf01-dir and --rf02-dir, or use --download-rf with --rf-key.")
        print(f"Unified classes configured: {UNIFIED_CLASSES}")
        return

    create_unified_dataset(source_dirs, output_path, seed=args.seed)


if __name__ == "__main__":
    main()
