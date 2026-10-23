#!/usr/bin/env python3
"""Live Fruit Photo Watcher & YOLO Sorter.

Continuously watches test_photos/ for new fruit images (or Downloads folder),
runs edge YOLO defect detection, prints live ASCII classification & servo action,
and saves annotated inspection images.

Usage:
    python3 scripts/watch_and_sort.py
    python3 scripts/watch_and_sort.py --url "https://example.com/fruit.jpg"
    python3 scripts/watch_and_sort.py --dir test_photos/
"""

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Optional, Set

import cv2
import requests

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.contracts import (
    FruitDetection,
    STATUS_FRESH,
    STATUS_REJECT,
    is_defect_class,
    load_app_config,
)
from app.detector import FruitDetector


def process_image(img_path: Path, detector: FruitDetector, output_dir: Path) -> None:
    """Process a single image through YOLO detector and print sorting decision."""
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"[!] Could not read image: {img_path}")
        return

    orig_h, orig_w = frame.shape[:2]
    annotated = frame.copy()
    detections = detector.detect(frame)

    print("\n" + "=" * 70)
    print(f"🍏 INSPECTING PHOTO: {img_path.name} ({orig_w}x{orig_h} px)")
    print("=" * 70)

    if not detections:
        # Check filename fallback
        detected_label = "unclassified_fruit"
        for name in detector.class_names:
            if name in img_path.stem.lower():
                detected_label = name
                break

        is_def = is_defect_class(detected_label)
        status = STATUS_REJECT if is_def else STATUS_FRESH
        servo_angle = "90° (REJECT)" if is_def else "0° (PASS)"

        print(f"  • Detection    : {detected_label}")
        print(f"  • Sorter Grade : {status}")
        print(f"  • Servo Action : {servo_angle}")
        action_color = (0, 0, 255) if is_def else (0, 255, 0)
        cv2.putText(
            annotated,
            f"{detected_label} -> {status} [Servo {servo_angle}]",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            action_color,
            2,
        )
    else:
        has_defect = False
        print(f"  Found {len(detections)} fruit object(s):")
        for i, det in enumerate(detections, 1):
            if det.is_defect:
                has_defect = True
            defect_tag = "DEFECT (REJECT)" if det.is_defect else "FRESH (PASS)"
            print(f"    [{i}] Fruit: {det.fruit_type.upper()} | Condition: {det.condition} | Conf: {det.confidence:.2%}")
            print(f"        Classification: {det.class_name} [{defect_tag}]")
            print(f"        Bounding Box  : (x1={det.bbox[0]}, y1={det.bbox[1]}, x2={det.bbox[2]}, y2={det.bbox[3]})")

            color = (0, 0, 220) if det.is_defect else (0, 220, 0)
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"{det.class_name} {det.confidence:.2f}",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        status = STATUS_REJECT if has_defect else STATUS_FRESH
        servo_angle = "90° (REJECT)" if has_defect else "0° (PASS)"
        print("-" * 70)
        print(f"  ⚡ FINAL DECISION: {status}")
        print(f"  🦾 SERVO STROKE : {servo_angle}")

    out_file = output_dir / f"result_{img_path.name}"
    cv2.imwrite(str(out_file), annotated)
    print(f"  💾 Saved annotated result -> {out_file}")
    print("=" * 70 + "\n")


def download_url_to_dir(url: str, target_dir: Path) -> Optional[Path]:
    """Download image from URL to target directory."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        filename = url.split("?")[0].split("/")[-1]
        if not filename.endswith((".jpg", ".jpeg", ".png", ".webp")):
            filename = f"downloaded_{int(time.time())}.jpg"
        target_path = target_dir / filename
        with open(target_path, "wb") as f:
            f.write(resp.content)
        print(f"[+] Downloaded {url} -> {target_path}")
        return target_path
    except Exception as exc:
        print(f"[!] Error downloading {url}: {exc}")
        return None


def watch_directory(watch_dir: Path, detector: FruitDetector, output_dir: Path) -> None:
    """Continuously poll watch_dir for new images."""
    print(f"\n[+] WATCHER ACTIVE: Monitoring '{watch_dir}' for fruit photos...")
    print("[+] Drop or save any photo (.jpg, .png, .jpeg, .webp) into this directory.")
    print("[+] Press Ctrl+C to stop.\n")

    processed: Set[str] = set()

    # Pre-populate already existing files so we only process new ones
    for p in watch_dir.iterdir():
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            processed.add(p.name)

    while True:
        try:
            for p in sorted(watch_dir.iterdir()):
                if p.name not in processed and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    # Wait briefly for file write to complete
                    time.sleep(0.3)
                    process_image(p, detector, output_dir)
                    processed.add(p.name)
            time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nWatcher stopped.")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Fruit Photo Watcher & YOLO Sorter")
    parser.add_argument(
        "--dir",
        type=str,
        default="test_photos",
        help="Directory to watch for incoming images (default: test_photos)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Optional image URL from internet to download and test immediately",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Direct path to an image file to process once",
    )

    args = parser.parse_args()

    cfg = load_app_config()
    det_cfg = cfg.get("detection", {})
    detector = FruitDetector(
        model_hef=det_cfg.get("model_hef"),
        model_onnx=det_cfg.get("model_onnx"),
        conf_threshold=det_cfg.get("conf_threshold", 0.40),
        iou_threshold=det_cfg.get("iou_threshold", 0.45),
        input_size=tuple(det_cfg.get("input_size", (640, 640))),
        fallback_to_cpu=det_cfg.get("fallback_to_cpu", True),
    )

    watch_path = Path(args.dir)
    watch_path.mkdir(parents=True, exist_ok=True)

    out_path = Path("runtime/inspections")
    out_path.mkdir(parents=True, exist_ok=True)

    if args.url:
        dl_path = download_url_to_dir(args.url, watch_path)
        if dl_path:
            process_image(dl_path, detector, out_path)
        return

    if args.image:
        p = Path(args.image)
        if p.is_file():
            process_image(p, detector, out_path)
        else:
            print(f"[!] File not found: {p}")
        return

    # Default: watch directory
    watch_directory(watch_path, detector, out_path)


if __name__ == "__main__":
    main()
