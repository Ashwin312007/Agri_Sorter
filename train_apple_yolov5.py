#!/usr/bin/env python3
"""YOLOv5 training and ONNX export script for Apple Damage Detection.

Executes:
1. Verification of class names in data.yaml (must match: apple_fresh, apple_rotten, apple_bruised, apple_damaged)
2. Setup/cloning of ultralytics/yolov5 if needed
3. Training yolov5n (416x416, 100 epochs, patience 20)
4. Validation and per-class evaluation
5. ONNX export (opset 12, simplified) with class names embedded in model metadata
"""
import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path

import yaml

EXPECTED_CLASSES = ["apple_fresh", "apple_rotten", "apple_bruised", "apple_damaged"]


def verify_classes(data_yaml_path: Path):
    """Verifies that data.yaml defines exactly the 4 required apple classes."""
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}")

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    names = data.get("names", [])
    if isinstance(names, dict):
        class_list = [names[k] for k in sorted(names)]
    elif isinstance(names, list):
        class_list = names
    else:
        raise ValueError(f"Unrecognized names format in data.yaml: {names}")

    print(f"Loaded class names from {data_yaml_path}: {class_list}")
    assert sorted(class_list) == sorted(EXPECTED_CLASSES), (
        f"Class mismatch! Expected {EXPECTED_CLASSES}, but found {class_list}"
    )
    print("[PASS] Class verification passed successfully.")
    return class_list


def ensure_yolov5_repo(yolov5_dir: Path):
    """Clones ultralytics/yolov5 if not present."""
    if not (yolov5_dir / "train.py").exists():
        print(f"Cloning ultralytics/yolov5 repository into {yolov5_dir}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ultralytics/yolov5.git", str(yolov5_dir)],
            check=True,
        )
        print("Installing YOLOv5 requirements...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(yolov5_dir / "requirements.txt")],
            check=True,
        )


def train_model(
    yolov5_dir: Path,
    data_yaml: Path,
    imgsz: int = 416,
    batch_size: int = 16,
    epochs: int = 100,
    patience: int = 20,
    weights: str = "yolov5n.pt",
    project: str = "runs",
    name: str = "apple",
    device: str = "",
):
    """Executes YOLOv5 training."""
    cmd = [
        sys.executable,
        str(yolov5_dir / "train.py"),
        "--img", str(imgsz),
        "--batch", str(batch_size),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--data", str(data_yaml.resolve()),
        "--weights", weights,
        "--project", project,
        "--name", name,
        "--exist-ok",
    ]
    if device:
        cmd.extend(["--device", device])

    print("Running training command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def evaluate_model(yolov5_dir: Path, weights_path: Path, data_yaml: Path, imgsz: int = 416):
    """Evaluates validation metrics (mAP50, per-class metrics)."""
    cmd = [
        sys.executable,
        str(yolov5_dir / "val.py"),
        "--weights", str(weights_path),
        "--data", str(data_yaml.resolve()),
        "--img", str(imgsz),
        "--task", "val",
    ]
    print("\nRunning evaluation:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def export_onnx(
    yolov5_dir: Path,
    weights_path: Path,
    imgsz: int = 416,
    opset: int = 12,
    simplify: bool = True,
) -> Path:
    """Exports model to ONNX with opset 12 and onnxsim simplification."""
    cmd = [
        sys.executable,
        str(yolov5_dir / "export.py"),
        "--weights", str(weights_path),
        "--include", "onnx",
        "--img", str(imgsz),
        "--opset", str(opset),
    ]
    if simplify:
        cmd.append("--simplify")

    print("\nRunning export command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    onnx_path = weights_path.with_suffix(".onnx")
    if not onnx_path.exists():
        raise FileNotFoundError(f"Exported ONNX file not found at {onnx_path}")
    return onnx_path


def embed_and_verify_onnx_metadata(onnx_path: Path, class_names: list):
    """Ensures class names are properly embedded in ONNX metadata."""
    try:
        import onnx
    except ImportError:
        print("[WARN] 'onnx' package not found for metadata verification. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "onnx"], check=True)
        import onnx

    model = onnx.load(str(onnx_path))
    meta_dict = {prop.key: prop.value for prop in model.metadata_props}

    names_str = str(dict(enumerate(class_names)))
    if "names" not in meta_dict or meta_dict["names"] != names_str:
        print("Embedding class names into ONNX model metadata...")
        # Update or add metadata
        meta_prop = None
        for prop in model.metadata_props:
            if prop.key == "names":
                meta_prop = prop
                break
        if meta_prop is None:
            meta_prop = model.metadata_props.add()
            meta_prop.key = "names"
        meta_prop.value = names_str
        onnx.save(model, str(onnx_path))

    # Verify with onnxruntime if available
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        cust_meta = sess.get_modelmeta().custom_metadata_map
        if "names" in cust_meta:
            names_parsed = ast.literal_eval(cust_meta["names"])
            print(f"[PASS] ONNX Runtime verified metadata 'names': {names_parsed}")
    except Exception as e:
        print(f"[INFO] ONNX verification check: {e}")

    print(f"Verified ONNX model at: {onnx_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Apple Fruit Damage Detector - YOLOv5 Training & Export")
    parser.add_argument("--data", type=str, default="dataset_apple/data.yaml", help="Path to data.yaml")
    parser.add_argument("--weights", type=str, default="yolov5n.pt", help="Initial weights (default yolov5n.pt)")
    parser.add_argument("--imgsz", type=int, default=416, help="Input image size (416)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--yolov5-dir", type=str, default="yolov5", help="Path to yolov5 repository")
    parser.add_argument("--device", type=str, default="", help="Device: cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--export-only", type=str, default=None, help="Skip training and export this .pt weights file")
    args = parser.parse_args()

    data_yaml_path = Path(args.data)
    class_names = verify_classes(data_yaml_path)

    yolov5_dir = Path(args.yolov5_dir)
    ensure_yolov5_repo(yolov5_dir)

    if args.export_only:
        best_pt = Path(args.export_only)
    else:
        train_model(
            yolov5_dir=yolov5_dir,
            data_yaml=data_yaml_path,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            weights=args.weights,
            device=args.device,
        )
        best_pt = Path("runs/apple/weights/best.pt")

    evaluate_model(yolov5_dir, best_pt, data_yaml_path, imgsz=args.imgsz)
    onnx_path = export_onnx(yolov5_dir, best_pt, imgsz=args.imgsz, opset=12, simplify=True)
    embed_and_verify_onnx_metadata(onnx_path, class_names)
    print(f"\n[DONE] Pipeline complete. Final ONNX deliverable: {onnx_path}")


if __name__ == "__main__":
    main()
