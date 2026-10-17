#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np

def main():
    base_dir = Path(__file__).resolve().parent.parent
    detector_path = base_dir / "models" / "detector" / "yolo11n.pt"
    classifier_path = base_dir / "models" / "external" / "best.pt"

    print("========================================")
    print("      Agri_Sorter Model Verification    ")
    print("========================================")

    # 1. Verify Detector
    print(f"[1/4] Loading detector from: {detector_path}")
    if not detector_path.exists():
        print(f"ERROR: Detector file not found at {detector_path}")
        sys.exit(1)

    from ultralytics import YOLO
    detector = YOLO(str(detector_path))

    # Check that 'apple' is present in detector classes and print class ID
    det_names = detector.names
    print(f"[1/4] Detector total classes: {len(det_names)}")
    apple_cid = None
    for cid, name in det_names.items():
        if name.lower() == "apple":
            apple_cid = cid
            break

    if apple_cid is not None:
        print(f"[1/4] SUCCESS: 'apple' found in detector classes: Class ID {apple_cid} ('{det_names[apple_cid]}')")
    else:
        print("[1/4] ERROR: 'apple' NOT found in detector classes!")
        sys.exit(1)

    # 2. Test Detector Inference
    print("\n[2/4] Generating synthetic test frame (640x640x3)...")
    synthetic_frame = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    print("[2/4] Running detector(synthetic_frame, imgsz=416, verbose=False)...")
    det_results = detector(synthetic_frame, imgsz=416, verbose=False)
    boxes = det_results[0].boxes
    print(f"[2/4] SUCCESS: Detector inference ran cleanly. Detected boxes: {len(boxes)}")

    # 3. Verify Classifier
    print(f"\n[3/4] Loading classifier from: {classifier_path}")
    if not classifier_path.exists():
        print(f"ERROR: Classifier file not found at {classifier_path}")
        sys.exit(1)

    classifier = YOLO(str(classifier_path))
    cls_names = classifier.names
    print(f"[3/4] Classifier total classes: {len(cls_names)}")
    print(f"[3/4] Classifier class mapping: {cls_names}")

    # 4. Test Classifier Inference
    print("\n[4/4] Generating synthetic crop (224x224x3)...")
    crop = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    print("[4/4] Running classifier(crop, imgsz=224, verbose=False)...")
    cls_results = classifier(crop, imgsz=224, verbose=False)
    
    probs = cls_results[0].probs
    top1_id = probs.top1
    top1_conf = float(probs.top1conf)
    top1_name = cls_names.get(top1_id, str(top1_id))

    print(f"[4/4] SUCCESS: Classifier inference ran cleanly.")
    print(f"[4/4] Top-1 prediction: Class ID {top1_id} ('{top1_name}') with confidence: {top1_conf:.4f}")
    print("[4/4] Full class probability distribution:")
    for cid, name in cls_names.items():
        score = float(probs.data[cid])
        print(f"       Class {cid} ({name}): {score:.4f}")

    print("\n=================================================================")
    print("VERIFICATION COMPLETE: Both detector & classifier executed cleanly!")
    print("=================================================================")

if __name__ == "__main__":
    main()
