#!/usr/bin/env python3
"""Agri_Sorter YOLO Apple Defect Dataset Downloader & Validator.

Downloads or generates a YOLO-format defect dataset for training YOLO11n on:
- fresh_apple
- rot
- bruise

Supports:
1. Roboflow Universe download via API key (--roboflow-key, --project)
2. Direct zip URL download (--url)
3. Verified starter sample generator (--sample or default when no key/url given)
4. Comprehensive dataset validation (checking image-label pairs and normalized bboxes)

Usage:
    python scripts/download_yolo_dataset.py --sample
    python scripts/download_yolo_dataset.py --roboflow-key <KEY> --project apple-defects-yafjq
    python scripts/download_yolo_dataset.py --url https://example.com/apple_defects.zip
    python scripts/download_yolo_dataset.py --validate-only --output data/external/apple_defects
"""
import argparse
import io
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

DEFAULT_OUTPUT_DIR = Path("data/external/apple_defects")
DEFAULT_PROJECT = "apple-defects-yafjq"
DEFAULT_CLASSES = {
    0: "fresh_apple",
    1: "rot",
    2: "bruise",
}


def create_sample_dataset(output_dir: Path, classes: Dict[int, str]) -> Path:
    """Generate a verified starter dataset with synthetic apple defect images and YOLO labels."""
    print(f"\n[+] Generating verified starter apple defect dataset at: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": 18,
        "val": 6,
        "test": 3,
    }

    # Ensure clean split directories
    for split in splits:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    img_size = 640
    np.random.seed(42)

    total_images = 0
    total_annotations = 0

    for split, count in splits.items():
        for i in range(count):
            img_filename = f"apple_{split}_{i:03d}.jpg"
            lbl_filename = f"apple_{split}_{i:03d}.txt"

            img_path = output_dir / "images" / split / img_filename
            lbl_path = output_dir / "labels" / split / lbl_filename

            # 1. Generate conveyor belt background (dark slate gray with fine texture)
            frame = np.full((img_size, img_size, 3), (38, 38, 42), dtype=np.uint8)
            noise = np.random.randint(-10, 10, (img_size, img_size, 3), dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Draw conveyor belt guide lines
            cv2.line(frame, (0, 80), (img_size, 80), (55, 55, 60), 2)
            cv2.line(frame, (0, img_size - 80), (img_size, img_size - 80), (55, 55, 60), 2)

            # Assign class round-robin across fresh (0), rot (1), bruise (2)
            assigned_class = i % len(classes)

            # Apple dimensions
            radius = np.random.randint(90, 140)
            cx = np.random.randint(radius + 40, img_size - radius - 40)
            cy = np.random.randint(radius + 90, img_size - radius - 90)

            # Base apple color (red, golden-yellow, or green)
            color_variant = np.random.choice(["red", "yellow_green"])
            if color_variant == "red":
                base_color = (35, 45, 190)  # BGR
            else:
                base_color = (40, 170, 130)

            # Draw apple body (slightly flattened circle)
            cv2.ellipse(frame, (cx, cy), (radius, int(radius * 0.94)), 0, 0, 360, base_color, -1)
            # Add subtle radial lighting highlight
            cv2.ellipse(
                frame,
                (cx - int(radius * 0.25), cy - int(radius * 0.25)),
                (int(radius * 0.35), int(radius * 0.25)),
                -25, 0, 360,
                (int(base_color[0] * 1.3), int(base_color[1] * 1.3), min(255, int(base_color[2] * 1.3))),
                -1,
            )
            # Stem cavity indentation
            cv2.ellipse(frame, (cx, cy - int(radius * 0.88)), (int(radius * 0.2), int(radius * 0.1)), 0, 0, 360, (20, 20, 25), -1)

            # If defect, add visible visual artifacts
            if assigned_class == 1:  # rot
                rot_r = int(radius * np.random.uniform(0.25, 0.45))
                rot_cx = cx + int(np.random.uniform(-radius * 0.35, radius * 0.35))
                rot_cy = cy + int(np.random.uniform(-radius * 0.35, radius * 0.35))
                # Dark necrotic rot core
                cv2.circle(frame, (rot_cx, rot_cy), rot_r, (25, 35, 45), -1)
                # Fungal softer ring
                cv2.circle(frame, (rot_cx, rot_cy), int(rot_r * 1.25), (45, 60, 75), 3)
            elif assigned_class == 2:  # bruise
                bruise_r = int(radius * np.random.uniform(0.28, 0.50))
                bruise_cx = cx + int(np.random.uniform(-radius * 0.3, radius * 0.3))
                bruise_cy = cy + int(np.random.uniform(-radius * 0.3, radius * 0.3))
                # Indented bruise discoloration (mottled brown)
                overlay = frame.copy()
                cv2.ellipse(overlay, (bruise_cx, bruise_cy), (bruise_r, int(bruise_r * 0.8)), 30, 0, 360, (30, 70, 95), -1)
                cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

            # Save synthetic image
            cv2.imwrite(str(img_path), frame)
            total_images += 1

            # Compute YOLO normalized bounding box (cx, cy, w, h in [0, 1])
            x1 = max(0, cx - radius)
            y1 = max(0, cy - radius)
            x2 = min(img_size, cx + radius)
            y2 = min(img_size, cy + radius)

            bbox_cx = ((x1 + x2) / 2.0) / img_size
            bbox_cy = ((y1 + y2) / 2.0) / img_size
            bbox_w = (x2 - x1) / float(img_size)
            bbox_h = (y2 - y1) / float(img_size)

            label_line = f"{assigned_class} {bbox_cx:.6f} {bbox_cy:.6f} {bbox_w:.6f} {bbox_h:.6f}\n"
            lbl_path.write_text(label_line)
            total_annotations += 1

    # Write data.yaml
    data_yaml_path = output_dir / "data.yaml"
    data_yaml_content = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {int(k): str(v) for k, v in classes.items()},
    }
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(data_yaml_content, f, sort_keys=False)

    print(f"✅ Generated {total_images} sample images across train/val/test with {total_annotations} YOLO annotations.")
    print(f"✅ Created dataset configuration: {data_yaml_path}")
    return data_yaml_path


def download_from_roboflow(
    api_key: str,
    project_name: str,
    output_dir: Path,
    version: int = 1,
) -> Path:
    """Download YOLOv11/v8 dataset from Roboflow Universe using API key."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[!] roboflow package is not installed. Installing or use --sample.")
        raise RuntimeError("Please install roboflow via: pip install roboflow")

    print(f"\n[+] Connecting to Roboflow project '{project_name}' (v{version})...")
    rf = Roboflow(api_key=api_key)

    # Allow format 'workspace/project' or just 'project'
    if "/" in project_name:
        workspace_id, proj_id = project_name.split("/", 1)
        project = rf.workspace(workspace_id).project(proj_id)
    else:
        # Default workspace for user
        project = rf.workspace().project(project_name)

    dataset = project.version(version).download("yolov11", location=str(output_dir))
    data_yaml = output_dir / "data.yaml"
    if not data_yaml.exists():
        found = list(output_dir.glob("**/data.yaml"))
        if found:
            data_yaml = found[0]
    print(f"✅ Roboflow dataset downloaded to: {data_yaml}")
    return data_yaml


def download_from_url(url: str, output_dir: Path) -> Path:
    """Download and extract a YOLO dataset zip file from a direct URL."""
    print(f"\n[+] Downloading YOLO dataset from: {url} ...")
    output_dir.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Agri_Sorter/1.0 (Embedded Linux/Edge AI)"},
    )
    with urllib.request.urlopen(req) as response:
        zip_data = response.read()

    print(f"[+] Extracting {len(zip_data) / (1024*1024):.2f} MB archive to {output_dir}...")
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        zf.extractall(output_dir)

    data_yaml = output_dir / "data.yaml"
    if not data_yaml.exists():
        found = list(output_dir.glob("**/data.yaml"))
        if found:
            data_yaml = found[0]
        else:
            raise FileNotFoundError(f"data.yaml not found inside extracted archive at {output_dir}")

    print(f"✅ Extracted dataset to: {output_dir}")
    return data_yaml


def validate_dataset(dataset_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    """Validate YOLO dataset integrity:

    1. Checks data.yaml existence and structure.
    2. Verifies images and labels in train/val/test splits.
    3. Checks bounding box normalization [0, 1] and class indices.
    4. Confirms non-empty image files and decoding integrity.
    """
    print(f"\n=================================================================")
    print(f"            YOLO Dataset Integrity Validation                    ")
    print(f"  Target: {dataset_dir.resolve()}")
    print(f"=================================================================")

    data_yaml_path = dataset_dir / "data.yaml"
    if not data_yaml_path.exists():
        found = list(dataset_dir.glob("**/data.yaml"))
        if found:
            data_yaml_path = found[0]
        else:
            print(f"❌ ERROR: data.yaml not found in {dataset_dir}")
            return False, {"error": "data.yaml missing"}

    try:
        with open(data_yaml_path, "r") as f:
            meta = yaml.safe_load(f)
    except Exception as exc:
        print(f"❌ ERROR: Could not parse {data_yaml_path}: {exc}")
        return False, {"error": f"Invalid YAML: {exc}"}

    names = meta.get("names", {})
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    elif not isinstance(names, dict):
        names = {}

    print(f"[+] Classes defined in data.yaml: {names}")
    base_path = Path(meta.get("path", dataset_dir))
    if not base_path.is_absolute():
        base_path = (data_yaml_path.parent / base_path).resolve()

    stats: Dict[str, Any] = {
        "classes": names,
        "splits": {},
        "total_images": 0,
        "total_boxes": 0,
        "class_counts": {k: 0 for k in names.keys()},
        "errors": [],
    }

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for split_key in ["train", "val", "test"]:
        if split_key not in meta:
            continue

        split_rel = meta[split_key]
        split_img_dir = (base_path / split_rel).resolve()
        if not split_img_dir.exists():
            alt_dir = data_yaml_path.parent / split_rel
            if alt_dir.exists():
                split_img_dir = alt_dir.resolve()

        if not split_img_dir.exists():
            print(f"⚠️  Split '{split_key}' directory not found: {split_img_dir}")
            continue

        # Look for corresponding labels directory
        # Case A: images/<split> -> labels/<split>
        # Case B: <split>/images -> <split>/labels
        if "images" in split_img_dir.parts:
            parts = list(split_img_dir.parts)
            idx = len(parts) - 1 - parts[::-1].index("images")
            parts[idx] = "labels"
            split_lbl_dir = Path(*parts)
        else:
            split_lbl_dir = split_img_dir.parent / "labels" / split_img_dir.name

        images = [f for f in split_img_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
        split_boxes = 0
        split_errors = 0

        for img_p in images:
            lbl_p = split_lbl_dir / (img_p.stem + ".txt")
            if not lbl_p.exists():
                stats["errors"].append(f"Missing label file for: {img_p.name}")
                split_errors += 1
                continue

            # Check image readable
            img_arr = cv2.imread(str(img_p))
            if img_arr is None or img_arr.size == 0:
                stats["errors"].append(f"Corrupted or unreadable image: {img_p}")
                split_errors += 1
                continue

            # Parse label file
            try:
                lines = lbl_p.read_text().strip().splitlines()
                for line_idx, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        stats["errors"].append(f"Invalid label line ({lbl_p.name}:{line_idx}): '{line}'")
                        split_errors += 1
                        continue

                    cid = int(parts[0])
                    coords = [float(x) for x in parts[1:5]]

                    if cid not in names:
                        stats["errors"].append(f"Class ID {cid} not in data.yaml names: {lbl_p.name}")
                        split_errors += 1

                    for val in coords:
                        if not (0.0 <= val <= 1.05):  # allow tiny rounding margin
                            stats["errors"].append(f"BBox coordinate out of range [0, 1] ({val}): {lbl_p.name}")
                            split_errors += 1

                    stats["class_counts"][cid] = stats["class_counts"].get(cid, 0) + 1
                    split_boxes += 1
            except Exception as exc:
                stats["errors"].append(f"Failed parsing label {lbl_p}: {exc}")
                split_errors += 1

        stats["splits"][split_key] = {
            "image_count": len(images),
            "box_count": split_boxes,
            "errors": split_errors,
        }
        stats["total_images"] += len(images)
        stats["total_boxes"] += split_boxes

    print("-" * 65)
    print(f"{'Split':<10} | {'Images':<10} | {'Boxes':<10} | {'Errors':<10}")
    print("-" * 65)
    for sp, d in stats["splits"].items():
        print(f"{sp:<10} | {d['image_count']:<10} | {d['box_count']:<10} | {d['errors']:<10}")
    print("-" * 65)
    print(f"Total Images: {stats['total_images']} | Total BBoxes: {stats['total_boxes']}")
    print(f"Class Distribution:")
    for cid, name in names.items():
        print(f"  [{cid}] {name:<15}: {stats['class_counts'].get(cid, 0)} boxes")

    if stats["errors"]:
        print(f"\n❌ Validation found {len(stats['errors'])} error(s):")
        for err in stats["errors"][:10]:
            print(f"   - {err}")
        if len(stats["errors"]) > 10:
            print(f"   ... and {len(stats['errors']) - 10} more.")
        return False, stats

    print("\n✅ DATASET INTEGRITY VERIFIED: All image-label pairs and bounding boxes valid!")
    return True, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agri_Sorter YOLO Apple Defect Dataset Downloader & Validator."
    )
    parser.add_argument(
        "--roboflow-key",
        type=str,
        default=None,
        help="Roboflow API key for automatic Universe download",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=DEFAULT_PROJECT,
        help=f"Roboflow project name (default: {DEFAULT_PROJECT})",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Roboflow project version (default: 1)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Direct URL to a zipped YOLO defect dataset",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        default=False,
        help="Generate verified starter sample dataset with fresh/rot/bruise defects",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        default=False,
        help="Only run validation on existing dataset without downloading",
    )
    args = parser.parse_args()

    out_path = Path(args.output)

    if args.validate_only:
        valid, _ = validate_dataset(out_path)
        sys.exit(0 if valid else 1)

    if args.roboflow_key:
        data_yaml = download_from_roboflow(
            api_key=args.roboflow_key,
            project_name=args.project,
            output_dir=out_path,
            version=args.version,
        )
    elif args.url:
        data_yaml = download_from_url(url=args.url, output_dir=out_path)
    else:
        # Default fallback or explicit --sample: generate starter set
        data_yaml = create_sample_dataset(out_path, classes=DEFAULT_CLASSES)

    valid, _ = validate_dataset(out_path)
    if not valid:
        print("❌ Dataset downloaded/generated but failed integrity validation!")
        sys.exit(1)

    print(f"\n🎉 Dataset ready at: {out_path.resolve()}")
    print(f"   Config: {data_yaml}")
    print(f"   Next step: Train YOLO11n with:")
    print(f"     python scripts/train_yolo_defects.py --data {data_yaml}\n")


if __name__ == "__main__":
    main()
