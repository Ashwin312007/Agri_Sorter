#!/usr/bin/env python3
"""Agri_Sorter Multi-Fruit Dataset Pipeline & Conveyor Belt Augmentation Suite.

Downloads, merges, deduplicates, and splits multi-fruit defect datasets across 4 fruits:
  - apple, banana, orange, mango
and 5 conditions:
  - fresh, rotten, bruised, scarred, bug_bite
into 20 compound classes defined in app/contracts.py:
  0..4:   apple_fresh, apple_rotten, apple_bruised, apple_scarred, apple_bug_bite
  5..9:   banana_fresh, banana_rotten, banana_bruised, banana_scarred, banana_bug_bite
  10..14: orange_fresh, orange_rotten, orange_bruised, orange_scarred, orange_bug_bite
  15..19: mango_fresh, mango_rotten, mango_bruised, mango_scarred, mango_bug_bite

Pipeline Features:
1. Public Dataset Ingestion:
   - Roboflow Universe API & direct export URLs
   - Kaggle fruit freshness & defect datasets (CLI or archive extraction)
   - Sample direct zip URLs
   - Local directory ingestion with optional fruit hint
2. Label Remapping:
   - Comprehensive synonym dictionary mapping raw classes to the 20 compound classes
   - Support for custom JSON / YAML mapping files
3. Deduplication:
   - Cryptographic MD5 hash (exact duplicates)
   - Perceptual aHash (Hamming distance threshold for near duplicates)
4. Strict Split:
   - Stratified train / val / test split strictly partitioned as 80% / 10% / 10%
5. Conveyor Belt Physical Augmentations:
   - Horizontal motion blur (simulating conveyor belt linear transit)
   - Brightness and contrast shifts (simulating fluctuating overhead warehouse lighting)
   - Affine rotation (simulating fruit tumbling/rolling)
   - Distance scaling (simulating camera-to-belt distance variation)
   - Automatic affine bounding box coordinate transformation & border clipping
6. Quality Assurance & Threshold Warnings:
   - ASCII distribution summary table across splits
   - Explicit WARNING if any of the 20 compound classes has fewer than 300 images
7. Synthetic Demo Mode:
   - Fully offline generation of multi-fruit conveyor images & annotations (--synthetic-demo)

Usage Examples:
    # 1. Run offline synthetic verification demo
    python3 scripts/dataset_pipeline.py --synthetic-demo --output-dir data/merged

    # 2. Ingest local directories (e.g. apple defect dataset)
    python3 scripts/dataset_pipeline.py --source-dirs data/external/apple_defects --fruit-hint apple

    # 3. Download from Roboflow Universe
    python3 scripts/dataset_pipeline.py --roboflow-project apple-defects-yafjq --roboflow-key <KEY>

    # 4. Download from Kaggle
    python3 scripts/dataset_pipeline.py --kaggle-datasets sriramr/fruits-fresh-and-rotten-for-classification
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
import urllib.request
import zipfile

import cv2
import numpy as np
import yaml

# Resolve Repo Root and import contracts
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from app.contracts import COMPOUND_CLASSES, CONDITIONS, FRUITS
except ImportError:
    # Fallback to definition if imported standalone outside repo
    FRUITS = ["apple", "banana", "orange", "mango"]
    CONDITIONS = ["fresh", "rotten", "bruised", "scarred", "bug_bite"]
    COMPOUND_CLASSES = [f"{fruit}_{cond}" for fruit in FRUITS for cond in CONDITIONS]

CLASS_TO_ID = {name: i for i, name in enumerate(COMPOUND_CLASSES)}
ID_TO_CLASS = {i: name for i, name in enumerate(COMPOUND_CLASSES)}

DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "merged"
DEFAULT_MIN_CLASS_THRESHOLD = 300


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class BBox:
    """YOLO Normalized Bounding Box."""
    class_id: int
    cx: float  # [0, 1]
    cy: float  # [0, 1]
    w: float   # [0, 1]
    h: float   # [0, 1]

    def to_yolo_line(self) -> str:
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}\n"


@dataclass
class SampleRecord:
    """An image sample record with bounding box annotations."""
    image_path: Path
    bboxes: List[BBox]
    md5_hash: str
    ahash: int
    primary_class: int


# ==============================================================================
# LABEL MAPPER
# ==============================================================================

class LabelMapper:
    """Maps arbitrary incoming dataset labels and class IDs to the 20 compound classes."""

    CONDITION_SYNONYMS = {
        "fresh": [
            "fresh", "good", "clean", "healthy", "sound", "ripe", "ripened",
            "grade_a", "normal", "perfect", "undamaged"
        ],
        "rotten": [
            "rot", "rotten", "decay", "decayed", "mold", "moldy", "bad",
            "anthracnose", "spoil", "spoiled", "breakdown", "fungal", "fungus",
            "necrotic", "black_rot", "bitter_rot", "stem_end_rot", "cigar_end_rot",
            "overripe", "blight", "mildew"
        ],
        "bruised": [
            "bruise", "bruised", "bruising", "impact", "pressure", "indent",
            "sapburn", "soft", "damage", "squished", "crush", "mechanical_injury",
            "mechanical_damage"
        ],
        "scarred": [
            "scar", "scarred", "scab", "scratch", "russet", "russetting",
            "crack", "cracked", "canker", "blotch", "blemish", "scuff", "rust",
            "cork", "corky", "peel_blemish", "wind_scar", "skin_defect",
            "melanose", "greening", "sigatoka"
        ],
        "bug_bite": [
            "bug", "bug_bite", "bugbite", "bite", "worm", "wormhole", "insect",
            "insect_bite", "pest", "weevil", "fly", "fruit_fly", "moth",
            "codling_moth", "puncture", "borer", "larva", "thrip", "thrips",
            "mite", "scale_insect", "aphid"
        ],
    }

    FRUIT_SYNONYMS = {
        "apple": ["apple", "apples", "malus"],
        "banana": ["banana", "bananas", "musa"],
        "orange": ["orange", "oranges", "citrus", "mandarin", "tangerine"],
        "mango": ["mango", "mangoes", "mangos", "mangifera"],
    }

    def __init__(self, custom_map: Optional[Dict[str, str]] = None):
        self.custom_map: Dict[str, str] = custom_map or {}

    @classmethod
    def from_file(cls, path: Path) -> "LabelMapper":
        """Load custom mapping from JSON or YAML file."""
        if not path.is_file():
            raise FileNotFoundError(f"Label map file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return cls(custom_map={str(k).lower().strip(): str(v).lower().strip() for k, v in data.items()})

    def map_label(self, raw_label: str, fruit_hint: Optional[str] = None) -> Optional[int]:
        """Map raw text label to compound class ID (0..19), or None if unmappable."""
        cleaned = raw_label.lower().strip().replace("-", "_").replace(" ", "_")

        # 1. Check custom overrides first
        if cleaned in self.custom_map:
            target = self.custom_map[cleaned]
            if target in CLASS_TO_ID:
                return CLASS_TO_ID[target]

        # 2. Direct exact compound class match
        if cleaned in CLASS_TO_ID:
            return CLASS_TO_ID[cleaned]

        # 3. Identify Fruit
        detected_fruit: Optional[str] = None
        for fruit_name, syns in self.FRUIT_SYNONYMS.items():
            for syn in syns:
                if syn in cleaned:
                    detected_fruit = fruit_name
                    break
            if detected_fruit:
                break

        if not detected_fruit and fruit_hint:
            hint_clean = fruit_hint.lower().strip()
            if hint_clean in FRUITS:
                detected_fruit = hint_clean

        # 4. Identify Condition
        detected_condition: Optional[str] = None
        for cond_name, syns in self.CONDITION_SYNONYMS.items():
            for syn in syns:
                if re.search(r"\b" + re.escape(syn) + r"\b", cleaned) or syn in cleaned:
                    detected_condition = cond_name
                    break
            if detected_condition:
                break

        # 5. Combine Fruit and Condition
        if detected_fruit and detected_condition:
            compound = f"{detected_fruit}_{detected_condition}"
            return CLASS_TO_ID.get(compound)

        # Fallback: if only fruit identified, default to fresh
        if detected_fruit and not detected_condition:
            return CLASS_TO_ID.get(f"{detected_fruit}_fresh")

        return None

    def map_data_yaml(self, data_yaml_path: Path, fruit_hint: Optional[str] = None) -> Dict[int, int]:
        """Read YOLO data.yaml and map old class indices to target compound class IDs."""
        if not data_yaml_path.is_file():
            return {}

        with open(data_yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        names = cfg.get("names", {})
        mapping: Dict[int, int] = {}

        if isinstance(names, list):
            for idx, name in enumerate(names):
                mapped_id = self.map_label(str(name), fruit_hint=fruit_hint)
                if mapped_id is not None:
                    mapping[idx] = mapped_id
        elif isinstance(names, dict):
            for idx_key, name in names.items():
                try:
                    idx = int(idx_key)
                except ValueError:
                    continue
                mapped_id = self.map_label(str(name), fruit_hint=fruit_hint)
                if mapped_id is not None:
                    mapping[idx] = mapped_id

        return mapping


# ==============================================================================
# DEDUPLICATOR
# ==============================================================================

class Deduplicator:
    """Deduplicates images using cryptographic MD5 and perceptual aHash."""

    def __init__(self, ahash_threshold: int = 4, hash_method: str = "both"):
        self.ahash_threshold = ahash_threshold
        self.hash_method = hash_method
        self.seen_md5: Set[str] = set()
        self.seen_ahashes: List[int] = []

    @staticmethod
    def compute_md5(file_path: Path) -> str:
        """Compute MD5 checksum of raw image file."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def compute_ahash(img: np.ndarray) -> int:
        """Compute 64-bit Average Hash (aHash) from OpenCV image."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        avg = resized.mean()
        h = 0
        for b in (resized > avg).flatten():
            h = (h << 1) | int(b)
        return h

    def is_duplicate(self, md5_val: str, ahash_val: int) -> Tuple[bool, str]:
        """Check if image hash matches an existing sample. Returns (is_dup, reason)."""
        # Exact MD5 match
        if self.hash_method in ["both", "md5"] and md5_val in self.seen_md5:
            return True, f"exact_md5 ({md5_val[:8]})"

        # Perceptual aHash match
        if self.hash_method in ["both", "ahash"]:
            for seen_h in self.seen_ahashes:
                dist = (ahash_val ^ seen_h).bit_count()
                if dist <= self.ahash_threshold:
                    return True, f"ahash_dist_{dist} (<= {self.ahash_threshold})"

        return False, ""

    def register(self, md5_val: str, ahash_val: int) -> None:
        """Register unique image hash in seen index."""
        self.seen_md5.add(md5_val)
        self.seen_ahashes.append(ahash_val)


# ==============================================================================
# CONVEYOR BELT AUGMENTOR
# ==============================================================================

class ConveyorAugmentor:
    """Conveyor-specific physical augmentations with exact YOLO bbox affine transforms."""

    def __init__(
        self,
        blur_kernel_max: int = 15,
        brightness_range: Tuple[int, int] = (-30, 30),
        contrast_range: Tuple[float, float] = (0.75, 1.25),
        rotation_range: Tuple[float, float] = (-180.0, 180.0),
        scale_range: Tuple[float, float] = (0.85, 1.15),
        conveyor_bg_color: Tuple[int, int, int] = (40, 42, 45),
    ):
        self.blur_kernel_max = blur_kernel_max
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.conveyor_bg_color = conveyor_bg_color

    def apply_motion_blur(self, img: np.ndarray, kernel_size: int, angle_deg: float = 0.0) -> np.ndarray:
        """Simulate horizontal conveyor belt motion blur."""
        if kernel_size <= 1:
            return img

        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = (kernel_size - 1) // 2
        kernel[center, :] = 1.0 / kernel_size

        if abs(angle_deg) > 1e-2:
            M = cv2.getRotationMatrix2D((center, center), angle_deg, 1.0)
            kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
            ksum = np.sum(kernel)
            if ksum > 0:
                kernel /= ksum

        return cv2.filter2D(img, -1, kernel)

    def apply_lighting_shift(self, img: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """Simulate overhead factory lighting fluctuations (brightness and contrast)."""
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    def apply_affine_transform(
        self,
        img: np.ndarray,
        bboxes: List[BBox],
        angle_deg: float,
        scale: float,
    ) -> Tuple[np.ndarray, List[BBox]]:
        """Simulate fruit tumbling/rolling (rotation) and camera distance (scaling).

        Applies rotation and scale around image center and computes tight enclosing
        bounding boxes for all rotated objects, clipping to image bounds.
        """
        H, W = img.shape[:2]
        cx_img, cy_img = W / 2.0, H / 2.0
        M = cv2.getRotationMatrix2D((cx_img, cy_img), angle_deg, scale)

        transformed_img = cv2.warpAffine(img, M, (W, H), borderValue=self.conveyor_bg_color)

        new_bboxes: List[BBox] = []
        for bbox in bboxes:
            # Convert normalized bbox to pixel corners
            x1 = (bbox.cx - bbox.w / 2.0) * W
            y1 = (bbox.cy - bbox.h / 2.0) * H
            x2 = (bbox.cx + bbox.w / 2.0) * W
            y2 = (bbox.cy + bbox.h / 2.0) * H

            corners = np.array([
                [x1, y1, 1.0],
                [x2, y1, 1.0],
                [x2, y2, 1.0],
                [x1, y2, 1.0],
            ], dtype=np.float32)

            rot_corners = (M @ corners.T).T  # Shape (4, 2)

            rx1 = float(np.min(rot_corners[:, 0]))
            ry1 = float(np.min(rot_corners[:, 1]))
            rx2 = float(np.max(rot_corners[:, 0]))
            ry2 = float(np.max(rot_corners[:, 1]))

            # Clip box to image boundaries
            rx1 = max(0.0, min(float(W), rx1))
            ry1 = max(0.0, min(float(H), ry1))
            rx2 = max(0.0, min(float(W), rx2))
            ry2 = max(0.0, min(float(H), ry2))

            bw = rx2 - rx1
            bh = ry2 - ry1

            # Discard degenerate/clipped boxes
            if bw < 4.0 or bh < 4.0:
                continue

            new_cx = ((rx1 + rx2) / 2.0) / float(W)
            new_cy = ((ry1 + ry2) / 2.0) / float(H)
            new_w = bw / float(W)
            new_h = bh / float(H)

            # Strict bounds clamp
            new_cx = max(0.0, min(1.0, new_cx))
            new_cy = max(0.0, min(1.0, new_cy))
            new_w = max(0.0, min(1.0, new_w))
            new_h = max(0.0, min(1.0, new_h))

            new_bboxes.append(BBox(bbox.class_id, new_cx, new_cy, new_w, new_h))

        return transformed_img, new_bboxes

    def augment(
        self,
        img: np.ndarray,
        bboxes: List[BBox],
        rng: Optional[random.Random] = None,
    ) -> Tuple[np.ndarray, List[BBox]]:
        """Apply stochastic conveyor augmentations (blur, lighting, rotate, scale)."""
        r = rng or random.Random()

        # 1. Tumbling rotation & distance scaling
        angle = r.uniform(self.rotation_range[0], self.rotation_range[1])
        scale = r.uniform(self.scale_range[0], self.scale_range[1])
        aug_img, aug_bboxes = self.apply_affine_transform(img, bboxes, angle, scale)

        # 2. Conveyor belt motion blur (horizontal +/- 10 degrees)
        if r.random() < 0.7:
            k_size = r.choice([3, 5, 7, 9, 11, min(self.blur_kernel_max, 15)])
            blur_angle = r.uniform(-8.0, 8.0)
            aug_img = self.apply_motion_blur(aug_img, k_size, blur_angle)

        # 3. Overhead lighting shift
        if r.random() < 0.8:
            alpha = r.uniform(self.contrast_range[0], self.contrast_range[1])
            beta = r.uniform(self.brightness_range[0], self.brightness_range[1])
            aug_img = self.apply_lighting_shift(aug_img, alpha, beta)

        return aug_img, aug_bboxes


# ==============================================================================
# SYNTHETIC DEMO GENERATOR
# ==============================================================================

class SyntheticDemoGenerator:
    """Generates synthetic conveyor belt images for all 4 fruits and 5 defect conditions."""

    FRUIT_PALETTES = {
        "apple": [(35, 45, 195), (45, 175, 120), (35, 200, 220)],  # Red, Green, Golden
        "banana": [(40, 215, 245), (45, 200, 230)],                  # Yellow, Yellow-green
        "orange": [(25, 140, 240), (20, 155, 250)],                  # Bright orange
        "mango": [(35, 185, 235), (45, 160, 215)],                   # Golden yellow, blush
    }

    def __init__(self, img_size: int = 640):
        self.img_size = img_size

    def create_conveyor_background(self, rng: random.Random) -> np.ndarray:
        """Render slate-gray conveyor belt surface with motion grain and guide rails."""
        bg = np.full((self.img_size, self.img_size, 3), (40, 42, 45), dtype=np.uint8)

        # Add subtle conveyor belt surface texture noise
        noise = rng.randint(-8, 8)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Longitudinal belt seams / motion guide stripes
        cv2.line(bg, (0, 70), (self.img_size, 70), (58, 60, 64), 2)
        cv2.line(bg, (0, self.img_size - 70), (self.img_size, self.img_size - 70), (58, 60, 64), 2)

        # Top and bottom metallic guide rails
        cv2.rectangle(bg, (0, 0), (self.img_size, 45), (85, 90, 95), -1)
        cv2.line(bg, (0, 45), (self.img_size, 45), (30, 32, 35), 2)

        cv2.rectangle(bg, (0, self.img_size - 45), (self.img_size, self.img_size), (85, 90, 95), -1)
        cv2.line(bg, (0, self.img_size - 45), (self.img_size, self.img_size - 45), (30, 32, 35), 2)

        return bg

    def draw_fruit_shape(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        fruit: str,
        cx: int,
        cy: int,
        radius: int,
        rng: random.Random,
    ) -> Tuple[int, int, int, int]:
        """Draw fruit body onto image and mask; returns tight pixel bounding box (x, y, w, h)."""
        color_choices = self.FRUIT_PALETTES[fruit]
        base_color = color_choices[rng.randint(0, len(color_choices) - 1)]

        if fruit == "apple":
            # Slightly flattened sphere
            cv2.ellipse(img, (cx, cy), (radius, int(radius * 0.94)), 0, 0, 360, base_color, -1)
            cv2.ellipse(mask, (cx, cy), (radius, int(radius * 0.94)), 0, 0, 360, 255, -1)
            # Specular lighting highlight
            cv2.ellipse(
                img,
                (cx - int(radius * 0.28), cy - int(radius * 0.25)),
                (int(radius * 0.35), int(radius * 0.22)),
                -25, 0, 360,
                (min(255, int(base_color[0] * 1.3)), min(255, int(base_color[1] * 1.3)), min(255, int(base_color[2] * 1.3))),
                -1,
            )
            # Stem cavity
            cv2.ellipse(img, (cx, cy - int(radius * 0.85)), (int(radius * 0.18), int(radius * 0.08)), 0, 0, 360, (20, 22, 28), -1)

        elif fruit == "banana":
            # Curved crescent banana
            length = int(radius * 2.2)
            thick = int(radius * 0.65)
            tilt = rng.uniform(-25.0, 25.0)

            pts_top = []
            pts_bot = []
            for t in np.linspace(-1.0, 1.0, 24):
                x = t * (length * 0.5)
                y_mid = (t ** 2) * (length * 0.18)
                pts_top.append([x, y_mid - thick * 0.5])
                pts_bot.append([x, y_mid + thick * 0.5])

            poly_local = np.array(pts_top + pts_bot[::-1], dtype=np.float32)
            rad = math.radians(tilt)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
            poly_rot = (poly_local @ R.T) + np.array([cx, cy], dtype=np.float32)
            poly = poly_rot.astype(np.int32)

            cv2.fillPoly(img, [poly], base_color)
            cv2.fillPoly(mask, [poly], 255)

            # Banana pedicel (stem) and tip
            cv2.circle(img, (int(poly[0][0]), int(poly[0][1])), max(3, int(thick * 0.2)), (30, 80, 95), -1)
            cv2.circle(img, (int(poly[len(pts_top) - 1][0]), int(poly[len(pts_top) - 1][1])), max(3, int(thick * 0.18)), (20, 20, 25), -1)

        elif fruit == "orange":
            # Spherical orange with stippled rind pores
            cv2.circle(img, (cx, cy), radius, base_color, -1)
            cv2.circle(mask, (cx, cy), radius, 255, -1)
            # Rind dimples
            for _ in range(18):
                px = cx + rng.randint(-int(radius * 0.7), int(radius * 0.7))
                py = cy + rng.randint(-int(radius * 0.7), int(radius * 0.7))
                if (px - cx) ** 2 + (py - cy) ** 2 < (radius * 0.8) ** 2:
                    cv2.circle(img, (px, py), 1, (int(base_color[0] * 0.8), int(base_color[1] * 0.85), int(base_color[2] * 0.85)), -1)
            # Calyx / button
            cv2.circle(img, (cx, cy), max(2, int(radius * 0.08)), (40, 110, 50), -1)

        elif fruit == "mango":
            # Asymmetrical kidney-shaped mango
            rx, ry = radius, int(radius * 0.68)
            cv2.ellipse(img, (cx, cy), (rx, ry), 25, 0, 360, base_color, -1)
            cv2.ellipse(mask, (cx, cy), (rx, ry), 25, 0, 360, 255, -1)
            # Pinkish-red shoulder blush
            cv2.ellipse(
                img,
                (cx - int(rx * 0.3), cy - int(ry * 0.2)),
                (int(rx * 0.4), int(ry * 0.4)),
                10, 0, 360,
                (50, 75, 210),
                -1,
            )
            # Stem end
            cv2.circle(img, (cx - int(rx * 0.8), cy - int(ry * 0.3)), max(2, int(radius * 0.07)), (25, 35, 45), -1)

        bx, by, bw, bh = cv2.boundingRect(mask)
        return bx, by, bw, bh

    def draw_defect(
        self,
        img: np.ndarray,
        fruit: str,
        condition: str,
        cx: int,
        cy: int,
        radius: int,
        rng: random.Random,
    ) -> None:
        """Render realistic visual defect features corresponding to condition."""
        if condition == "fresh":
            return  # Clean unblemished skin

        dx = cx + rng.randint(-int(radius * 0.35), int(radius * 0.35))
        dy = cy + rng.randint(-int(radius * 0.35), int(radius * 0.35))

        if condition == "rotten":
            # Sunken necrotic spot with fungal ring
            rot_r = int(radius * rng.uniform(0.25, 0.42))
            if fruit == "orange":
                # Penicillium green-white mold
                cv2.circle(img, (dx, dy), int(rot_r * 1.3), (120, 175, 110), -1)
                cv2.circle(img, (dx, dy), rot_r, (40, 70, 45), -1)
            else:
                # Black/brown rot core
                cv2.circle(img, (dx, dy), int(rot_r * 1.25), (40, 55, 70), -1)
                cv2.circle(img, (dx, dy), rot_r, (25, 30, 40), -1)

        elif condition == "bruised":
            # Soft indented brownish discoloration
            bruise_r = int(radius * rng.uniform(0.30, 0.50))
            overlay = img.copy()
            cv2.ellipse(overlay, (dx, dy), (bruise_r, int(bruise_r * 0.75)), rng.randint(0, 180), 0, 360, (28, 65, 95), -1)
            cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

        elif condition == "scarred":
            # Rough corky scab / russet scar
            scar_len = int(radius * rng.uniform(0.35, 0.65))
            pt1 = (dx - scar_len // 2, dy - rng.randint(-10, 10))
            pt2 = (dx + scar_len // 2, dy + rng.randint(-10, 10))
            cv2.line(img, pt1, pt2, (65, 85, 105), max(3, int(radius * 0.08)))
            # Jagged russet spots
            for _ in range(6):
                px = dx + rng.randint(-scar_len // 2, scar_len // 2)
                py = dy + rng.randint(-8, 8)
                cv2.circle(img, (px, py), rng.randint(2, 4), (55, 75, 95), -1)

        elif condition == "bug_bite":
            # Tiny dark puncture hole with chlorotic halo
            halo_r = max(4, int(radius * 0.12))
            hole_r = max(2, int(radius * 0.05))
            cv2.circle(img, (dx, dy), halo_r, (45, 95, 130), -1)
            cv2.circle(img, (dx, dy), hole_r, (15, 15, 20), -1)

    def generate_sample(
        self,
        fruit: str,
        condition: str,
        rng: random.Random,
    ) -> Tuple[np.ndarray, BBox]:
        """Generate a complete single-fruit synthetic conveyor image with YOLO bbox."""
        img = self.create_conveyor_background(rng)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        radius = rng.randint(85, 125)
        cx = rng.randint(radius + 70, self.img_size - radius - 70)
        cy = rng.randint(radius + 70, self.img_size - radius - 70)

        bx, by, bw, bh = self.draw_fruit_shape(img, mask, fruit, cx, cy, radius, rng)
        self.draw_defect(img, fruit, condition, cx, cy, radius, rng)

        # Compute normalized YOLO coordinates
        norm_cx = (bx + bw / 2.0) / float(self.img_size)
        norm_cy = (by + bh / 2.0) / float(self.img_size)
        norm_w = bw / float(self.img_size)
        norm_h = bh / float(self.img_size)

        compound_name = f"{fruit}_{condition}"
        class_id = CLASS_TO_ID[compound_name]

        bbox = BBox(class_id, norm_cx, norm_cy, norm_w, norm_h)
        return img, bbox


# ==============================================================================
# DOWNLOADERS (ROBOFLOW, KAGGLE, URL)
# ==============================================================================

def download_zip_url(url: str, output_dir: Path) -> Path:
    """Download and extract a zip file from a direct URL."""
    print(f"\n[+] Downloading dataset archive from URL:\n    {url}")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "downloaded_dataset.zip"

    req = urllib.request.Request(url, headers={"User-Agent": "Agri_Sorter/1.0"})
    with urllib.request.urlopen(req) as resp, open(archive_path, "wb") as out_f:
        shutil.copyfileobj(resp, out_f)

    print(f"[+] Extracting archive ({archive_path.stat().st_size / (1024*1024):.1f} MB)...")
    extract_dir = output_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(extract_dir)

    print(f"[+] Extraction complete -> {extract_dir}")
    return extract_dir


def download_roboflow(
    project: str,
    api_key: str,
    workspace: Optional[str] = None,
    version: int = 1,
    output_dir: Path = DEFAULT_RAW_DIR / "roboflow",
) -> Path:
    """Download YOLOv8/YOLO11 formatted dataset from Roboflow Universe."""
    print(f"\n[+] Connecting to Roboflow Universe for project: {project}...")
    dest_dir = output_dir / project
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Try roboflow Python SDK if installed
    try:
        from roboflow import Roboflow  # type: ignore
        rf = Roboflow(api_key=api_key)
        ws_name = workspace or rf.workspace().name
        proj = rf.workspace(ws_name).project(project)
        dataset = proj.version(version).download("yolov8", location=str(dest_dir))
        print(f"[+] Downloaded via Roboflow SDK to: {dataset.location}")
        return Path(dataset.location)
    except Exception as e:
        print(f"[*] Roboflow SDK not available or threw ({e}). Attempting direct REST export...")

    # 2. Direct Roboflow Universe REST download URL
    url = f"https://api.roboflow.com/{workspace or 'roboflow'}/{project}/{version}/yolov8?api_key={api_key}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Agri_Sorter/1.0"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            zip_url = data.get("yolov8", {}).get("zip") or data.get("export", {}).get("link")
            if zip_url:
                return download_zip_url(zip_url, dest_dir)
    except Exception as rest_err:
        print(f"[-] REST API request failed: {rest_err}")

    raise RuntimeError(
        f"Failed to download Roboflow dataset '{project}'. Verify your API key and project name, "
        "or pass a direct zip URL via --urls."
    )


def download_kaggle(
    dataset_name: str,
    output_dir: Path = DEFAULT_RAW_DIR / "kaggle",
) -> Path:
    """Download and extract a Kaggle public dataset."""
    dest_dir = output_dir / dataset_name.replace("/", "_")
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[+] Fetching Kaggle dataset: {dataset_name}...")

    # Check if kaggle CLI is available
    if shutil.which("kaggle"):
        cmd = f"kaggle datasets download -d {dataset_name} -p {dest_dir} --unzip"
        print(f"[*] Running: {cmd}")
        ret = os.system(cmd)
        if ret == 0:
            print(f"[+] Successfully downloaded Kaggle dataset to: {dest_dir}")
            return dest_dir

    print("[-] Kaggle CLI not found or returned error.")
    print("    To download manually, visit: https://www.kaggle.com/datasets/" + dataset_name)
    print(f"    Extract images into: {dest_dir} and re-run with --source-dirs {dest_dir}")
    raise RuntimeError(f"Kaggle download failed for {dataset_name}.")


# ==============================================================================
# DATASET PIPELINE ENGINE
# ==============================================================================

class DatasetPipeline:
    """End-to-end dataset pipeline: ingests, remaps, dedups, splits strictly 80/10/10, and augments."""

    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        raw_dir: Path = DEFAULT_RAW_DIR,
        min_class_threshold: int = DEFAULT_MIN_CLASS_THRESHOLD,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        augment_train: bool = True,
        augment_factor: int = 1,
        seed: int = 42,
        label_mapper: Optional[LabelMapper] = None,
        deduplicator: Optional[Deduplicator] = None,
        augmentor: Optional[ConveyorAugmentor] = None,
    ):
        self.output_dir = Path(output_dir)
        self.raw_dir = Path(raw_dir)
        self.min_class_threshold = min_class_threshold
        self.split_ratio = split_ratio
        self.augment_train = augment_train
        self.augment_factor = augment_factor
        self.seed = seed
        self.rng = random.Random(seed)

        self.label_mapper = label_mapper or LabelMapper()
        self.deduplicator = deduplicator or Deduplicator()
        self.augmentor = augmentor or ConveyorAugmentor()

    def discover_and_parse_sources(
        self,
        source_dirs: List[Path],
        fruit_hint: Optional[str] = None,
    ) -> List[SampleRecord]:
        """Scan source directories, parse labels/annotations, and remap to compound classes."""
        parsed_samples: List[SampleRecord] = []
        dup_count = 0
        total_scanned = 0

        for s_dir in source_dirs:
            if not s_dir.exists():
                print(f"[!] Warning: Source directory does not exist: {s_dir}")
                continue

            print(f"\n[+] Ingesting dataset source: {s_dir}")

            # Check for data.yaml mapping
            yaml_path = s_dir / "data.yaml"
            class_id_map: Dict[int, int] = {}
            if yaml_path.is_file():
                class_id_map = self.label_mapper.map_data_yaml(yaml_path, fruit_hint=fruit_hint)
                print(f"    Loaded data.yaml with {len(class_id_map)} mapped class IDs: {class_id_map}")

            # Find all image files
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            all_images = [p for p in s_dir.rglob("*") if p.suffix.lower() in valid_exts]
            print(f"    Found {len(all_images)} image files to evaluate.")

            for img_path in all_images:
                total_scanned += 1
                # 1. Deduplication Check
                try:
                    md5_val = self.deduplicator.compute_md5(img_path)
                    img_bgr = cv2.imread(str(img_path))
                    if img_bgr is None:
                        continue
                    ahash_val = self.deduplicator.compute_ahash(img_bgr)
                except Exception:
                    continue

                is_dup, reason = self.deduplicator.is_duplicate(md5_val, ahash_val)
                if is_dup:
                    dup_count += 1
                    continue

                # 2. Parse Bounding Box Annotations
                bboxes = self._find_and_parse_bboxes(img_path, s_dir, class_id_map, fruit_hint, img_bgr)
                if not bboxes:
                    continue

                self.deduplicator.register(md5_val, ahash_val)
                primary_class = bboxes[0].class_id
                parsed_samples.append(SampleRecord(
                    image_path=img_path,
                    bboxes=bboxes,
                    md5_hash=md5_val,
                    ahash=ahash_val,
                    primary_class=primary_class,
                ))

        print(f"\n[+] Ingestion Summary: {total_scanned} scanned, {dup_count} duplicates removed, {len(parsed_samples)} valid unique samples.")
        return parsed_samples

    def _find_and_parse_bboxes(
        self,
        img_path: Path,
        source_root: Path,
        class_id_map: Dict[int, int],
        fruit_hint: Optional[str],
        img_bgr: np.ndarray,
    ) -> List[BBox]:
        """Find matching YOLO label txt file or auto-extract contour bbox for folder classification."""
        bboxes: List[BBox] = []

        # Case A: Matching .txt label file in same dir or corresponding 'labels' dir
        label_candidates = [
            img_path.with_suffix(".txt"),
            Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt"),
            source_root / "labels" / f"{img_path.stem}.txt",
        ]

        txt_file: Optional[Path] = None
        for cand in label_candidates:
            if cand.is_file():
                txt_file = cand
                break

        if txt_file:
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            raw_id = int(parts[0])
                            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        except ValueError:
                            continue

                        # Remap class id
                        target_id: Optional[int] = None
                        if raw_id in class_id_map:
                            target_id = class_id_map[raw_id]
                        elif 0 <= raw_id < len(COMPOUND_CLASSES):
                            target_id = raw_id

                        if target_id is not None:
                            bboxes.append(BBox(target_id, cx, cy, w, h))
            return bboxes

        # Case B: Folder-based classification dataset (e.g. Kaggle freshapples/ or rottenbanana/)
        parent_dir_name = img_path.parent.name
        mapped_cls = self.label_mapper.map_label(parent_dir_name, fruit_hint=fruit_hint)
        if mapped_cls is not None:
            # Auto-extract bounding box via fruit contour
            H, W = img_bgr.shape[:2]
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            # Threshold against light or dark background
            _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > (W * H * 0.05):
                    bx, by, bw, bh = cv2.boundingRect(c)
                    cx = (bx + bw / 2.0) / float(W)
                    cy = (by + bh / 2.0) / float(H)
                    w = bw / float(W)
                    h = bh / float(H)
                    bboxes.append(BBox(mapped_cls, cx, cy, w, h))
                    return bboxes

            # Fallback centered object box
            bboxes.append(BBox(mapped_cls, 0.5, 0.5, 0.82, 0.82))

        return bboxes

    def split_strictly_stratified(
        self,
        samples: List[SampleRecord],
    ) -> Tuple[List[SampleRecord], List[SampleRecord], List[SampleRecord]]:
        """Stratified train/val/test split strictly adhering to 80% / 10% / 10% total ratio."""
        N = len(samples)
        if N == 0:
            return [], [], []

        r_train, r_val, r_test = self.split_ratio
        target_train = int(round(N * r_train))
        target_val = int(round(N * r_val))
        target_test = N - target_train - target_val

        # Group by primary class
        class_groups = defaultdict(list)
        for s in samples:
            class_groups[s.primary_class].append(s)

        for items in class_groups.values():
            self.rng.shuffle(items)

        train, val, test = [], [], []
        unassigned = []

        for cls_id, items in sorted(class_groups.items()):
            n = len(items)
            n_tr = int(n * r_train)
            n_va = int(n * r_val)
            n_te = int(n * r_test)

            train.extend(items[:n_tr])
            val.extend(items[n_tr:n_tr + n_va])
            test.extend(items[n_tr + n_va:n_tr + n_va + n_te])
            unassigned.extend(items[n_tr + n_va + n_te:])

        # Distribute remaining unassigned items strictly to hit exact target quotas
        self.rng.shuffle(unassigned)
        for item in unassigned:
            if len(train) < target_train:
                train.append(item)
            elif len(val) < target_val:
                val.append(item)
            else:
                test.append(item)

        assert len(train) == target_train, f"Train split count mismatch: {len(train)} vs {target_train}"
        assert len(val) == target_val, f"Val split count mismatch: {len(val)} vs {target_val}"
        assert len(test) == target_test, f"Test split count mismatch: {len(test)} vs {target_test}"

        return train, val, test

    def build_yolo_dataset(
        self,
        train_samples: List[SampleRecord],
        val_samples: List[SampleRecord],
        test_samples: List[SampleRecord],
    ) -> Dict[str, Any]:
        """Write partitioned images and label files to destination directory with conveyor augmentations."""
        print(f"\n[+] Building YOLO dataset structure at: {self.output_dir}")
        splits = {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
        }

        # Clean/create destination split directories
        for split_name in splits:
            (self.output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

        class_distribution: Dict[int, Dict[str, int]] = {
            i: {"train": 0, "val": 0, "test": 0, "total": 0} for i in range(len(COMPOUND_CLASSES))
        }

        for split_name, sample_list in splits.items():
            print(f"    Processing '{split_name}' split ({len(sample_list)} base images)...")
            for idx, sample in enumerate(sample_list):
                base_name = f"agri_{split_name}_{idx:05d}"
                dst_img = self.output_dir / "images" / split_name / f"{base_name}.jpg"
                dst_lbl = self.output_dir / "labels" / split_name / f"{base_name}.txt"

                # Read and save base image
                img_bgr = cv2.imread(str(sample.image_path))
                if img_bgr is None:
                    continue
                cv2.imwrite(str(dst_img), img_bgr)

                # Write base labels
                with open(dst_lbl, "w", encoding="utf-8") as lf:
                    for bbox in sample.bboxes:
                        lf.write(bbox.to_yolo_line())
                        class_distribution[bbox.class_id][split_name] += 1
                        class_distribution[bbox.class_id]["total"] += 1

                # Conveyor Augmentations (applied to train split only)
                if split_name == "train" and self.augment_train:
                    for aug_i in range(self.augment_factor):
                        aug_img, aug_bboxes = self.augmentor.augment(img_bgr, sample.bboxes, self.rng)
                        if not aug_bboxes:
                            continue

                        aug_base_name = f"{base_name}_aug_{aug_i}"
                        aug_dst_img = self.output_dir / "images" / split_name / f"{aug_base_name}.jpg"
                        aug_dst_lbl = self.output_dir / "labels" / split_name / f"{aug_base_name}.txt"

                        cv2.imwrite(str(aug_dst_img), aug_img)
                        with open(aug_dst_lbl, "w", encoding="utf-8") as alf:
                            for bbox in aug_bboxes:
                                alf.write(bbox.to_yolo_line())
                                class_distribution[bbox.class_id]["train"] += 1
                                class_distribution[bbox.class_id]["total"] += 1

        # Write data.yaml
        data_yaml_path = self.output_dir / "data.yaml"
        yaml_content = {
            "path": str(self.output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(COMPOUND_CLASSES),
            "names": {i: name for i, name in enumerate(COMPOUND_CLASSES)},
        }
        with open(data_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, sort_keys=False)

        print(f"[+] YOLO data configuration written to: {data_yaml_path}")
        return class_distribution

    def print_distribution_and_warnings(self, dist: Dict[int, Dict[str, int]]) -> List[str]:
        """Print ASCII summary table of class distributions and explicit warnings for low classes."""
        print("\n" + "=" * 90)
        print("                        AGRI_SORTER MULTI-FRUIT DATASET SUMMARY")
        print("=" * 90)
        header = f"{'ID':<3} | {'Class Name':<20} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8} | {'Status':<12}"
        print(header)
        print("-" * 90)

        underrepresented: List[str] = []
        total_train = sum(dist[i]["train"] for i in dist)
        total_val = sum(dist[i]["val"] for i in dist)
        total_test = sum(dist[i]["test"] for i in dist)
        grand_total = sum(dist[i]["total"] for i in dist)

        for i, class_name in enumerate(COMPOUND_CLASSES):
            c_tr = dist[i]["train"]
            c_va = dist[i]["val"]
            c_te = dist[i]["test"]
            c_tot = dist[i]["total"]

            if c_tot < self.min_class_threshold:
                status = f"LOW (<{self.min_class_threshold})"
                underrepresented.append(f"{class_name} ({c_tot} images)")
            else:
                status = "OK"

            print(f"{i:<3} | {class_name:<20} | {c_tr:<8} | {c_va:<8} | {c_te:<8} | {c_tot:<8} | {status:<12}")

        print("-" * 90)
        print(f"TOTALS | {'20 Compound Classes':<20} | {total_train:<8} | {total_val:<8} | {total_test:<8} | {grand_total:<8} |")
        print("=" * 90)

        # Print explicit warnings
        if underrepresented:
            print("\n" + "!" * 90)
            print(f"⚠️  WARNING: {len(underrepresented)} of 20 class(es) have fewer than {self.min_class_threshold} images:")
            for item in underrepresented:
                print(f"    - {item} < {self.min_class_threshold} threshold")
            print("\nACTION REQUIRED: Acquire additional real-world conveyor footage or use targeted")
            print("synthetic augmentations before production deployment on Hailo-8L / Raspberry Pi 4.")
            print("!" * 90 + "\n")
        else:
            print(f"\n[+] SUCCESS: All 20 compound classes satisfy the minimum threshold of {self.min_class_threshold} images!\n")

        return underrepresented

    def run_synthetic_demo(self, count: int = 100) -> None:
        """Run verified end-to-end synthetic conveyor demo pipeline without requiring API keys."""
        print(f"\n{'='*70}")
        print(f"🚀 INITIATING AGRI_SORTER SYNTHETIC MULTI-FRUIT DEMO (Count: {count})")
        print(f"{'='*70}")

        demo_raw_dir = self.raw_dir / "synthetic_demo"
        demo_raw_dir.mkdir(parents=True, exist_ok=True)

        generator = SyntheticDemoGenerator(img_size=640)
        demo_samples: List[SampleRecord] = []

        print(f"[+] Generating {count} synthetic multi-fruit conveyor images...")
        for i in range(count):
            # Round-robin assign across all 20 compound classes
            cls_idx = i % len(COMPOUND_CLASSES)
            fruit = FRUITS[cls_idx // len(CONDITIONS)]
            condition = CONDITIONS[cls_idx % len(CONDITIONS)]

            img, bbox = generator.generate_sample(fruit, condition, self.rng)
            img_path = demo_raw_dir / f"syn_{i:04d}_{fruit}_{condition}.jpg"
            lbl_path = demo_raw_dir / f"syn_{i:04d}_{fruit}_{condition}.txt"

            cv2.imwrite(str(img_path), img)
            with open(lbl_path, "w", encoding="utf-8") as lf:
                lf.write(bbox.to_yolo_line())

            md5_val = self.deduplicator.compute_md5(img_path)
            ahash_val = self.deduplicator.compute_ahash(img)
            self.deduplicator.register(md5_val, ahash_val)

            demo_samples.append(SampleRecord(
                image_path=img_path,
                bboxes=[bbox],
                md5_hash=md5_val,
                ahash=ahash_val,
                primary_class=bbox.class_id,
            ))

        print(f"[+] Successfully generated {len(demo_samples)} synthetic multi-fruit samples.")

        # Split 80/10/10
        train, val, test = self.split_strictly_stratified(demo_samples)
        print(f"[+] Stratified Split: Train={len(train)} (80%), Val={len(val)} (10%), Test={len(test)} (10%)")

        # Build YOLO dataset with conveyor augmentations
        dist = self.build_yolo_dataset(train, val, test)

        # Print summary and threshold warnings
        self.print_distribution_and_warnings(dist)
        print(f"[+] Synthetic demo verification successfully completed. Dataset at: {self.output_dir}")


# ==============================================================================
# CLI PARSER
# ==============================================================================

def build_cli_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Agri_Sorter Multi-Fruit YOLO Dataset Pipeline & Conveyor Augmentations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group_mode = parser.add_argument_group("Execution Mode")
    group_mode.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Generate synthetic conveyor belt images and run full pipeline offline without API tokens",
    )
    group_mode.add_argument(
        "--synthetic-count",
        type=int,
        default=100,
        help="Number of synthetic images to generate in --synthetic-demo mode",
    )

    group_paths = parser.add_argument_group("Paths & Output")
    group_paths.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Target directory for processed YOLO dataset",
    )
    group_paths.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Cache directory for raw dataset downloads",
    )
    group_paths.add_argument(
        "--source-dirs",
        type=Path,
        nargs="+",
        default=[],
        help="Local directories containing datasets to merge",
    )
    group_paths.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help="Path to custom JSON or YAML label mapping configuration",
    )

    group_download = parser.add_argument_group("Public Dataset Download Sources")
    group_download.add_argument(
        "--roboflow-project",
        type=str,
        default=None,
        help="Roboflow Universe project slug (e.g. apple-defects-yafjq)",
    )
    group_download.add_argument(
        "--roboflow-workspace",
        type=str,
        default=None,
        help="Roboflow workspace name (optional)",
    )
    group_download.add_argument(
        "--roboflow-key",
        type=str,
        default=os.environ.get("ROBOFLOW_API_KEY"),
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
    )
    group_download.add_argument(
        "--urls",
        type=str,
        nargs="+",
        default=[],
        help="Direct zip URLs of fruit defect datasets to download and merge",
    )
    group_download.add_argument(
        "--kaggle-datasets",
        type=str,
        nargs="+",
        default=[],
        help="Kaggle dataset slugs (e.g. sriramr/fruits-fresh-and-rotten-for-classification)",
    )
    group_download.add_argument(
        "--fruit-hint",
        type=str,
        choices=FRUITS,
        default=None,
        help="Fruit type hint for single-fruit datasets with unbranded defect labels",
    )

    group_aug = parser.add_argument_group("Conveyor Belt Augmentation & Split Settings")
    group_aug.add_argument(
        "--min-class-threshold",
        type=int,
        default=DEFAULT_MIN_CLASS_THRESHOLD,
        help="Minimum required images per compound class before raising an explicit warning",
    )
    group_aug.add_argument(
        "--split-ratio",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        help="Strict Train, Val, Test split ratios (must sum to 1.0)",
    )
    group_aug.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable conveyor belt augmentations on training split",
    )
    group_aug.add_argument(
        "--augment-factor",
        type=int,
        default=1,
        help="Number of augmented conveyor variants generated per training image",
    )
    group_aug.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible deduplication, splits, and augmentations",
    )

    return parser


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

def main() -> None:
    """CLI execution entrypoint."""
    parser = build_cli_parser()
    args = parser.parse_args()

    # Validate split ratios sum to 1.0
    r_sum = sum(args.split_ratio)
    if not math.isclose(r_sum, 1.0, rel_tol=1e-3):
        parser.error(f"Split ratios must sum to 1.0, got: {args.split_ratio} (sum = {r_sum:.3f})")

    # Load custom label mapper if provided
    label_mapper = LabelMapper.from_file(args.label_map) if args.label_map else LabelMapper()

    # Initialize Pipeline
    pipeline = DatasetPipeline(
        output_dir=args.output_dir,
        raw_dir=args.raw_dir,
        min_class_threshold=args.min_class_threshold,
        split_ratio=tuple(args.split_ratio),
        augment_train=not args.no_augment,
        augment_factor=args.augment_factor,
        seed=args.seed,
        label_mapper=label_mapper,
    )

    # 1. Synthetic Demo Mode
    if args.synthetic_demo:
        pipeline.run_synthetic_demo(count=args.synthetic_count)
        return

    # 2. Collect and Download Sources
    active_sources: List[Path] = list(args.source_dirs)

    # Download from Roboflow Universe
    if args.roboflow_project:
        if not args.roboflow_key:
            parser.error("--roboflow-key (or ROBOFLOW_API_KEY environment variable) is required for Roboflow download.")
        rf_path = download_roboflow(
            project=args.roboflow_project,
            api_key=args.roboflow_key,
            workspace=args.roboflow_workspace,
            output_dir=args.raw_dir / "roboflow",
        )
        active_sources.append(rf_path)

    # Download from direct URLs
    for idx, url in enumerate(args.urls):
        url_dir = args.raw_dir / "urls" / f"source_{idx}"
        url_path = download_zip_url(url, url_dir)
        active_sources.append(url_path)

    # Download from Kaggle
    for kaggle_slug in args.kaggle_datasets:
        kg_path = download_kaggle(kaggle_slug, output_dir=args.raw_dir / "kaggle")
        active_sources.append(kg_path)

    if not active_sources:
        print("\n[-] No input dataset sources specified.")
        print("    Use --synthetic-demo to test the pipeline locally with synthetic multi-fruit data,")
        print("    or provide --source-dirs, --roboflow-project, --kaggle-datasets, or --urls.")
        print("\nRun with --help for all available options.\n")
        sys.exit(1)

    # Ingest, Parse, Deduplicate
    samples = pipeline.discover_and_parse_sources(active_sources, fruit_hint=args.fruit_hint)
    if not samples:
        print("\n[-] No valid images or annotations found in provided sources. Exiting.")
        sys.exit(1)

    # Split strictly 80/10/10
    train, val, test = pipeline.split_strictly_stratified(samples)

    # Build YOLO dataset with conveyor augmentations
    distribution = pipeline.build_yolo_dataset(train, val, test)

    # Print summary and check thresholds
    pipeline.print_distribution_and_warnings(distribution)


if __name__ == "__main__":
    main()
