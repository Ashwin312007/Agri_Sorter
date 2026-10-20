#!/usr/bin/env python3
"""Workstation utility to export PyTorch/Ultralytics models (.pt) to ONNX (.onnx).

Run this script on a development/training workstation (with PyTorch and Ultralytics)
to generate lightweight ONNX models optimized for Raspberry Pi 4 OpenCV DNN inference.

Usage:
    python scripts/export_to_onnx.py
    python scripts/export_to_onnx.py --detector models/detector/yolo11n.pt --classifier models/external/best.pt
"""
import argparse
import os
import sys
from pathlib import Path


def check_dependencies() -> bool:
    """Verify that export dependencies are installed on workstation."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import ultralytics
    except ImportError:
        missing.append("ultralytics")

    try:
        import onnx
    except ImportError:
        missing.append("onnx")

    if missing:
        print("\n" + "=" * 70)
        print("❌ MISSING WORKSTATION EXPORT DEPENDENCIES:")
        print(f"   The following packages are required for ONNX export: {', '.join(missing)}")
        print("\nTo install on your development PC / training workstation, run:")
        print("   pip install ultralytics onnx onnxruntime onnxslim")
        print("=" * 70 + "\n")
        return False
    return True


def export_model(
    model_path: Path,
    imgsz: int,
    task: str = "detect",
    dynamic: bool = False,
    opset: int = 12,
) -> Path:
    """Export a YOLO model to ONNX format."""
    from ultralytics import YOLO

    print(f"\n[+] Loading {task} model: {model_path}")
    model = YOLO(str(model_path))

    print(f"[+] Exporting to ONNX (imgsz={imgsz}, dynamic={dynamic}, opset={opset}, simplify=True)...")
    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=dynamic,
        opset=opset,
        simplify=True,
    )
    exported = Path(exported_path)
    size_mb = exported.stat().st_size / (1024 * 1024)
    print(f"✅ Successfully exported {task} model to: {exported} ({size_mb:.2f} MB)")
    return exported


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Export Agri_Sorter YOLO weights (.pt) to ONNX for Raspberry Pi 4 OpenCV DNN."
    )
    parser.add_argument(
        "--detector",
        type=str,
        default=str(repo_root / "models" / "detector" / "yolo11n.pt"),
        help="Path to detector .pt model",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default=str(repo_root / "models" / "external" / "best.pt"),
        help="Path to classifier .pt model",
    )
    parser.add_argument(
        "--detector-size",
        type=int,
        default=416,
        help="Image size for detector (default: 416)",
    )
    parser.add_argument(
        "--classifier-size",
        type=int,
        default=224,
        help="Image size for classifier (default: 224)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version (default: 12, recommended for OpenCV DNN)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        default=False,
        help="Enable dynamic input shape (default: False, static shape is faster on ARM)",
    )
    args = parser.parse_args()

    print("=================================================================")
    print("      Agri_Sorter Workstation ONNX Model Exporter                ")
    print("=================================================================")

    if not check_dependencies():
        sys.exit(1)

    det_path = Path(args.detector)
    cls_path = Path(args.classifier)

    exported_files = []

    if det_path.is_file():
        det_out = export_model(
            model_path=det_path,
            imgsz=args.detector_size,
            task="detect",
            dynamic=args.dynamic,
            opset=args.opset,
        )
        exported_files.append(det_out)
    else:
        print(f"⚠️  Detector file not found at: {det_path}; skipping.")

    if cls_path.is_file():
        cls_out = export_model(
            model_path=cls_path,
            imgsz=args.classifier_size,
            task="classify",
            dynamic=args.dynamic,
            opset=args.opset,
        )
        exported_files.append(cls_out)
    else:
        print(f"⚠️  Classifier file not found at: {cls_path}; skipping.")

    if exported_files:
        print("\n" + "=" * 70)
        print("🎉 EXPORT COMPLETED SUCCESSFULLY!")
        print("Exported ONNX Models for Raspberry Pi 4:")
        for f in exported_files:
            print(f"  - {f} ({f.stat().st_size / (1024*1024):.2f} MB)")
        print("\nNext steps:")
        print("  1. Copy the .onnx files to your Raspberry Pi 4 'models/' directory.")
        print("  2. Run 'python scripts/check_rpi4_hardware.py' on the Pi 4 to benchmark.")
        print("=" * 70 + "\n")
    else:
        print("\n❌ No models were exported. Please check file paths.")
        sys.exit(1)


if __name__ == "__main__":
    main()
