#!/usr/bin/env python3
"""Workstation utility to export PyTorch/Ultralytics models (.pt) to ONNX (.onnx).

Run this script on a development/training workstation (with PyTorch and Ultralytics)
to generate lightweight ONNX models optimized for Raspberry Pi 4 / Hailo edge inference.
Supports the unified 4-class defect hierarchy model naming ('yolo11n_fruit').

Usage:
    python scripts/export_to_onnx.py
    python scripts/export_to_onnx.py --detector models/detector/yolo11n_fruit.pt --detector-output models/detector/yolo11n_fruit.onnx
"""
import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.contracts import UNIFIED_DEFECT_CLASSES


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


def patch_onnx_for_opencv_dnn(onnx_path: Path) -> None:
    """Ensure ONNX detect models are compatible with OpenCV 4.6.0+ DNN engine.

    OpenCV 4.6.0 (default on Debian 12 / Raspberry Pi OS Bookworm) fails on YOLO11/v8
    anchor stride multiplication if the stride constant is shape (1, N) instead of
    matching the bbox dimension (1, 4, N).
    """
    try:
        import numpy as np
        import onnx
        from onnx import numpy_helper

        model = onnx.load(str(onnx_path))
        modified = False

        for node in model.graph.node:
            if node.op_type == "Mul":
                for in_name in node.input:
                    for init in list(model.graph.initializer):
                        if init.name == in_name:
                            arr = numpy_helper.to_array(init)
                            if arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] > 100:
                                arr_tiled = np.tile(arr, (1, 4, 1))
                                new_init = numpy_helper.from_array(arr_tiled.astype(np.float32), name=init.name)
                                model.graph.initializer.remove(init)
                                model.graph.initializer.append(new_init)
                                modified = True
                                print(f"[+] Patched initializer '{init.name}' ({arr.shape} -> {arr_tiled.shape}) for OpenCV 4.6.0 compatibility.")

        if modified:
            onnx.save(model, str(onnx_path))
    except Exception as exc:
        print(f"[!] Note: Could not post-process ONNX for OpenCV 4.6.0: {exc}")


def export_model(
    model_path: Path,
    imgsz: int,
    task: str = "detect",
    dynamic: bool = False,
    opset: int = 12,
    output_path: Optional[Path] = None,
) -> Path:
    """Export a YOLO model to ONNX format and save companion class metadata JSON."""
    from ultralytics import YOLO

    model_path = Path(model_path).resolve()
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

    # If specific output destination is requested (e.g. yolo11n_fruit.onnx), move/copy to it
    if output_path is not None:
        target_path = Path(output_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if exported != target_path:
            shutil.copy(exported, target_path)
            exported = target_path

    if task == "detect":
        patch_onnx_for_opencv_dnn(exported)

    # Save sibling metadata JSON with class names for lightweight OpenCV DNN on Pi 4
    try:
        import json
        names = model.names
        if isinstance(names, dict):
            names_dict = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, (list, tuple)):
            names_dict = {i: str(v) for i, v in enumerate(names)}
        else:
            names_dict = {i: name for i, name in enumerate(UNIFIED_DEFECT_CLASSES)}

        meta_dict = {
            "task": task,
            "model_name": exported.stem,
            "names": names_dict,
            "num_classes": len(names_dict),
            "imgsz": imgsz,
        }
        json_path = exported.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(meta_dict, jf, indent=2)
        print(f"[+] Saved sibling class metadata to: {json_path}")
    except Exception as exc:
        print(f"[!] Note: Could not save class metadata JSON: {exc}")

    size_mb = exported.stat().st_size / (1024 * 1024)
    print(f"✅ Successfully exported {task} model to: {exported} ({size_mb:.2f} MB)")
    return exported


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    # Auto-detect default detector: prefer yolo11n_fruit.pt if exists, else yolo11n.pt
    default_det = repo_root / "models" / "detector" / "yolo11n_fruit.pt"
    if not default_det.is_file():
        default_det = repo_root / "models" / "detector" / "yolo11n.pt"

    parser = argparse.ArgumentParser(
        description="Export Agri_Sorter YOLO weights (.pt) to ONNX for Raspberry Pi 4 / Hailo edge deployment."
    )
    parser.add_argument(
        "--detector",
        type=str,
        default=str(default_det),
        help="Path to detector .pt model (default: models/detector/yolo11n_fruit.pt or yolo11n.pt)",
    )
    parser.add_argument(
        "--detector-output",
        type=str,
        default=str(repo_root / "models" / "detector" / "yolo11n_fruit.onnx"),
        help="Output path for exported detector ONNX (default: models/detector/yolo11n_fruit.onnx)",
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
        default=640,
        help="Image size for detector (default: 640)",
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
            output_path=Path(args.detector_output) if args.detector_output else None,
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
        print("Exported ONNX Models for Raspberry Pi 4 / Hailo-8L:")
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
