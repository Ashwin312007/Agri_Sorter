#!/usr/bin/env python3
"""Agri_Sorter Model Verification Utility.

Verifies vision models (detector and classifier):
- Prefers lightweight ONNX models via OpenCV DNN (zero PyTorch bloatware on Raspberry Pi 4).
- Supports PyTorch weights (.pt) if running on development workstation with Ultralytics installed.
"""
import sys
import time
from pathlib import Path
import numpy as np

# Ensure repo root is on sys.path
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from app.pi_inference import load_vision_engine, find_class_id, classify_apple, crop_box


def verify_with_opencv_dnn(det_onnx: Path, cls_onnx: Path) -> bool:
    """Verify models using lightweight OpenCV DNN engine."""
    print(f"\n[1/4] Loading ONNX detector via OpenCV DNN: {det_onnx}")
    detector = load_vision_engine(det_onnx, task="detect")

    apple_cid = find_class_id(detector, "apple")
    print(f"[1/4] SUCCESS: 'apple' class found: ID {apple_cid} ('{detector.names.get(apple_cid, 'apple')}')")

    print("\n[2/4] Generating synthetic test frame (640x640x3)...")
    synthetic_frame = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    det_results = detector.track(synthetic_frame, imgsz=416, conf=0.30)
    det_time_ms = (time.perf_counter() - t0) * 1000.0
    boxes = det_results[0].boxes
    print(f"[2/4] SUCCESS: Detector inference ran in {det_time_ms:.1f}ms. Tracked boxes: {len(boxes)}")

    print(f"\n[3/4] Loading ONNX classifier via OpenCV DNN: {cls_onnx}")
    classifier = load_vision_engine(cls_onnx, task="classify")
    print(f"[3/4] SUCCESS: Classifier classes: {classifier.names}")

    print("\n[4/4] Generating synthetic crop (224x224x3)...")
    crop = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    label, conf = classify_apple(classifier, crop)
    cls_time_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[4/4] SUCCESS: Classifier inference ran in {cls_time_ms:.1f}ms.")
    print(f"[4/4] Prediction: '{label}' with confidence: {conf:.4f}")

    print("\n=================================================================")
    print("VERIFICATION COMPLETE: Both ONNX models executed cleanly via OpenCV DNN!")
    print("=================================================================")
    return True


def verify_with_ultralytics(det_pt: Path, cls_pt: Path) -> bool:
    """Verify PyTorch models on workstation using Ultralytics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n" + "=" * 70)
        print("ℹ️  NOTE: PyTorch weight files (.pt) detected:")
        print(f"   Detector:   {det_pt}")
        print(f"   Classifier: {cls_pt}")
        print("\n   'ultralytics' is not installed in this environment.")
        print("   On Raspberry Pi 4, Agri_Sorter uses lightweight ONNX models via OpenCV DNN.")
        print("   To export these .pt models to ONNX for Pi 4 deployment, run on your workstation:")
        print("     python scripts/export_to_onnx.py")
        print("=" * 70 + "\n")
        return True

    print(f"\n[1/4] Loading PyTorch detector from: {det_pt}")
    detector = YOLO(str(det_pt))
    det_names = detector.names
    print(f"[1/4] Detector total classes: {len(det_names)}")
    apple_cid = find_class_id(detector, "apple")
    print(f"[1/4] SUCCESS: 'apple' found in detector classes: Class ID {apple_cid} ('{det_names[apple_cid]}')")

    print("\n[2/4] Generating synthetic test frame (640x640x3)...")
    synthetic_frame = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    det_results = detector(synthetic_frame, imgsz=416, verbose=False)
    boxes = det_results[0].boxes
    print(f"[2/4] SUCCESS: Detector inference ran cleanly. Detected boxes: {len(boxes)}")

    print(f"\n[3/4] Loading PyTorch classifier from: {cls_pt}")
    classifier = YOLO(str(cls_pt))
    print(f"[3/4] Classifier total classes: {len(classifier.names)}")

    print("\n[4/4] Generating synthetic crop (224x224x3)...")
    crop = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    cls_results = classifier(crop, imgsz=224, verbose=False)
    probs = cls_results[0].probs
    top1_id = probs.top1
    top1_conf = float(probs.top1conf)
    top1_name = classifier.names.get(top1_id, str(top1_id))
    print(f"[4/4] SUCCESS: Classifier inference ran cleanly. Top-1: '{top1_name}' ({top1_conf:.4f})")

    print("\n=================================================================")
    print("VERIFICATION COMPLETE: Both .pt models executed cleanly via Ultralytics!")
    print("=================================================================")
    return True


def main():
    print("========================================")
    print("      Agri_Sorter Model Verification    ")
    print("========================================")

    detector_onnx = base_dir / "models" / "detector" / "yolo11n.onnx"
    classifier_onnx = base_dir / "models" / "external" / "best.onnx"
    detector_pt = base_dir / "models" / "detector" / "yolo11n.pt"
    classifier_pt = base_dir / "models" / "external" / "best.pt"

    # Option A: Both ONNX files exist -> run OpenCV DNN verification
    if detector_onnx.is_file() and classifier_onnx.is_file():
        success = verify_with_opencv_dnn(detector_onnx, classifier_onnx)
        sys.exit(0 if success else 1)

    # Option B: PyTorch .pt files exist
    if detector_pt.is_file() and classifier_pt.is_file():
        success = verify_with_ultralytics(detector_pt, classifier_pt)
        sys.exit(0 if success else 1)

    print("ERROR: Neither ONNX nor PyTorch models found in models/ directory!")
    sys.exit(1)


if __name__ == "__main__":
    main()
