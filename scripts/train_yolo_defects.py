#!/usr/bin/env python3
"""Agri_Sorter YOLO Apple Defect Trainer & Edge ONNX Exporter.

Trains YOLO11n on the external apple defect dataset (fresh_apple, rot, bruise)
and automatically exports the fine-tuned detector to static ONNX optimized for
Raspberry Pi 4 OpenCV DNN inference.

Usage:
    python scripts/train_yolo_defects.py
    python scripts/train_yolo_defects.py --data data/external/apple_defects/data.yaml --epochs 10 --batch 8
"""
import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_to_onnx import export_model, patch_onnx_for_opencv_dnn


def check_train_dependencies() -> bool:
    """Verify PyTorch and Ultralytics are available for training."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import ultralytics
    except ImportError:
        missing.append("ultralytics")

    if missing:
        print("\n" + "=" * 70)
        print("❌ MISSING TRAINING DEPENDENCIES:")
        print(f"   Packages required: {', '.join(missing)}")
        print("\nTo train on your workstation, install requirements:")
        print("   pip install -r requirements-train.txt")
        print("   or run with workstation python venv:")
        print("   /home/taiki/Documents/Agri_Sorter/venv/bin/python scripts/train_yolo_defects.py")
        print("=" * 70 + "\n")
        return False
    return True


def auto_select_device(requested_device: str) -> str:
    """Select appropriate compute device (CUDA GPU if available, else CPU)."""
    if requested_device != "auto":
        return requested_device

    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def train_and_export(
    data_yaml: Path,
    base_model: Path,
    epochs: int = 30,
    imgsz: int = 416,
    batch: int = 16,
    device: str = "auto",
    export_onnx: bool = True,
    output_pt: Path = REPO_ROOT / "models" / "detector" / "yolo11n.pt",
    output_onnx: Path = REPO_ROOT / "models" / "detector" / "yolo11n.onnx",
    project: str = "runs/detect",
    name: str = "apple_defects",
) -> Tuple[Path, Optional[Path]]:
    """Train YOLO11n on apple defects and auto-export to OpenCV DNN-compatible ONNX."""
    from ultralytics import YOLO

    device = auto_select_device(device)
    print("\n=================================================================")
    print("      Agri_Sorter YOLO11n Apple Defect Training Pipeline         ")
    print("=================================================================")
    print(f"  Dataset Config : {data_yaml}")
    print(f"  Base Model     : {base_model}")
    print(f"  Epochs         : {epochs}")
    print(f"  Image Size     : {imgsz}")
    print(f"  Batch Size     : {batch}")
    print(f"  Compute Device : {device}")
    print(f"  Export ONNX    : {export_onnx}")
    print(f"  Target Pt      : {output_pt}")
    print(f"  Target ONNX    : {output_onnx}")
    print("=================================================================\n")

    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found at {data_yaml}. "
            f"Run 'python scripts/download_yolo_dataset.py --sample' first."
        )

    output_pt.parent.mkdir(parents=True, exist_ok=True)
    output_onnx.parent.mkdir(parents=True, exist_ok=True)

    print(f"[+] Loading base model: {base_model}...")
    model = YOLO(str(base_model))

    print(f"[+] Initiating YOLO11n defect training...")
    train_results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
    )

    # Locate best trained weights
    best_weights: Optional[Path] = None
    if hasattr(model, "trainer") and getattr(model.trainer, "best", None):
        best_weights = Path(model.trainer.best)

    if best_weights is None or not best_weights.is_file():
        candidate = Path(project) / name / "weights" / "best.pt"
        if candidate.is_file():
            best_weights = candidate
        else:
            last_candidate = Path(project) / name / "weights" / "last.pt"
            if last_candidate.is_file():
                best_weights = last_candidate
            else:
                best_weights = base_model

    print(f"\n[+] Copying best weights to target: {output_pt}")
    shutil.copyfile(best_weights, output_pt)
    print(f"✅ Saved trained PyTorch weights to: {output_pt} ({output_pt.stat().st_size / (1024*1024):.2f} MB)")

    # Read trained classes from model
    trained_model = YOLO(str(output_pt))
    print(f"[+] Model classes: {trained_model.names}")

    # Export classes metadata JSON for Pi 4 OpenCV DNN
    try:
        import json
        names = trained_model.names
        meta_dict = {
            "task": "detect",
            "names": {int(k): str(v) for k, v in names.items()} if isinstance(names, dict) else list(names),
            "imgsz": imgsz,
        }
        json_path = output_pt.with_suffix(".json")
        with open(json_path, "w") as jf:
            json.dump(meta_dict, jf, indent=2)
        print(f"✅ Saved class metadata JSON to: {json_path}")
        if output_onnx.with_suffix(".json").resolve() != json_path.resolve():
            with open(output_onnx.with_suffix(".json"), "w") as jf:
                json.dump(meta_dict, jf, indent=2)
    except Exception as exc:
        print(f"[!] Warning: Could not save class metadata JSON: {exc}")

    exported_onnx_path: Optional[Path] = None
    if export_onnx:
        print(f"\n[+] Auto-exporting {output_pt} to Raspberry Pi 4 ONNX format...")
        raw_onnx = export_model(
            model_path=output_pt,
            imgsz=imgsz,
            task="detect",
            dynamic=False,
            opset=12,
        )
        if raw_onnx.resolve() != output_onnx.resolve():
            shutil.copyfile(raw_onnx, output_onnx)

        # Ensure OpenCV 4.6.0 ARM NEON compatibility
        patch_onnx_for_opencv_dnn(output_onnx)
        exported_onnx_path = output_onnx
        size_mb = output_onnx.stat().st_size / (1024 * 1024)
        print(f"✅ Exported edge ONNX model to: {output_onnx} ({size_mb:.2f} MB)")

    print("\n=================================================================")
    print("🎉 TRAINING AND EXPORT COMPLETED SUCCESSFULLY!")
    print(f"  PyTorch Weights : {output_pt}")
    if exported_onnx_path:
        print(f"  Edge ONNX Model : {exported_onnx_path}")
    print("=================================================================\n")
    return output_pt, exported_onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agri_Sorter YOLO11n Apple Defect Trainer & Edge ONNX Exporter"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(REPO_ROOT / "data" / "external" / "apple_defects" / "data.yaml"),
        help="Path to YOLO data.yaml configuration",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(REPO_ROOT / "models" / "detector" / "yolo11n.pt"),
        help="Base YOLO model weights (.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=416,
        help="Image size for training and export (default: 416)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: 'auto', 'cpu', '0', etc. (default: auto)",
    )
    parser.add_argument(
        "--export-onnx",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically export best model to ONNX (default: True)",
    )
    parser.add_argument(
        "--output-pt",
        type=str,
        default=str(REPO_ROOT / "models" / "detector" / "yolo11n.pt"),
        help="Destination path for best PyTorch weights",
    )
    parser.add_argument(
        "--output-onnx",
        type=str,
        default=str(REPO_ROOT / "models" / "detector" / "yolo11n.onnx"),
        help="Destination path for edge ONNX model",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Training run project directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="apple_defects",
        help="Training run name",
    )
    args = parser.parse_args()

    if not check_train_dependencies():
        sys.exit(1)

    train_and_export(
        data_yaml=Path(args.data),
        base_model=Path(args.model),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        export_onnx=args.export_onnx,
        output_pt=Path(args.output_pt),
        output_onnx=Path(args.output_onnx),
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
