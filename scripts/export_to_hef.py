#!/usr/bin/env python3
"""Agri_Sorter PyTorch to Hailo HEF End-to-End Edge Export Pipeline.

Exports PyTorch YOLO11 detector (.pt) -> Static ONNX (opset 12/17, 640x640) ->
Hailo HEF (.hef) targeted for Raspberry Pi AI HAT (Hailo-8L accelerator).

Workflow:
  1. PyTorch (.pt) -> Static ONNX (.onnx) at 640x640 resolution (opset 17 or 12).
  2. Hailo Model Script (.alls) generation with input normalization and Hailo-8L optimization.
  3. Calibration dataset loading (INT8 post-training quantization).
  4. Hailo Dataflow Compiler (DFC) compilation into .hef format.
  5. Fallback simulation when hailo_sdk is not installed, generating complete artifacts
     and shell scripts for running DFC on Hailo-enabled workstations.

Usage:
    python scripts/export_to_hef.py
    python scripts/export_to_hef.py --hw-arch hailo8l --imgsz 640 --calib-count 64
"""
import argparse
import json
import logging
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.contracts import UNIFIED_DEFECT_CLASSES, COMPOUND_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("export_to_hef")

SIMULATED_HEF_MAGIC = b"HAILO_SIM_HEF_01\x00\x00"


def check_hailo_sdk() -> bool:
    """Check if Hailo Dataflow Compiler (hailo_sdk) is installed."""
    try:
        from hailo_sdk_client import ClientRunner  # type: ignore # noqa: F401
        return True
    except ImportError:
        try:
            import hailo_sdk_client  # type: ignore # noqa: F401
            return True
        except ImportError:
            return False


def patch_onnx_for_opencv_dnn(onnx_path: Path) -> None:
    """Patch ONNX model constants to maintain compatibility with OpenCV 4.6.0+ DNN engine."""
    try:
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
                                logger.info("Patched initializer '%s' for OpenCV DNN compatibility", init.name)

        if modified:
            onnx.save(model, str(onnx_path))
    except Exception as exc:
        logger.debug("Note: ONNX patching skipped or not needed: %s", exc)


def export_pt_to_onnx(
    pt_path: Path,
    onnx_path: Path,
    imgsz: int = 640,
    opset: int = 17,
    simplify: bool = True,
) -> Path:
    """Export PyTorch weights (.pt) to static ONNX model (640x640) for edge inference.

    Args:
        pt_path: Path to PyTorch model file.
        onnx_path: Destination path for the .onnx file.
        imgsz: Image resolution (static 640x640 required for Hailo-8L).
        opset: ONNX opset version (12 or 17 recommended for Hailo DFC).
        simplify: Whether to run onnxsim simplification.

    Returns:
        Path to the exported ONNX model.
    """
    pt_path = Path(pt_path).resolve()
    onnx_path = Path(onnx_path).resolve()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO

        logger.info("Loading PyTorch model from: %s", pt_path)
        model = YOLO(str(pt_path))

        logger.info(
            "Exporting to ONNX (imgsz=%d, dynamic=False, opset=%d, simplify=%s)...",
            imgsz,
            opset,
            simplify,
        )
        exported_file = model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=False,  # Hailo requires static input shapes
            opset=opset,
            simplify=simplify,
        )
        if Path(exported_file).resolve() != onnx_path:
            shutil_path = Path(exported_file).resolve()
            if shutil_path.is_file():
                import shutil
                shutil.copy(shutil_path, onnx_path)

        patch_onnx_for_opencv_dnn(onnx_path)
        logger.info("Successfully exported ONNX model to: %s", onnx_path)
        return onnx_path

    except ImportError:
        logger.warning(
            "Ultralytics/PyTorch not installed in this environment. "
            "Checking if existing ONNX model exists at %s",
            onnx_path,
        )
        if onnx_path.is_file():
            logger.info("Using existing ONNX model at: %s", onnx_path)
            return onnx_path
        else:
            logger.warning(
                "Neither PyTorch nor target ONNX model found. "
                "Will continue with simulated ONNX stub if needed for pipeline validation."
            )
            return onnx_path


def generate_hailo_model_script(
    alls_path: Path,
    model_name: str = "yolo11n_fruit",
    hw_arch: str = "hailo8l",
    input_shape: Tuple[int, int, int] = (640, 640, 3),
) -> Path:
    """Generate Hailo Model Script (.alls) for Hailo Dataflow Compiler optimization."""
    alls_path = Path(alls_path).resolve()
    alls_path.parent.mkdir(parents=True, exist_ok=True)

    alls_content = f"""# ==============================================================================
# Hailo Model Script (.alls) for Agri_Sorter Edge Fruit Detector
# Target Accelerator: Raspberry Pi AI HAT ({hw_arch})
# Model: {model_name}
# Input Resolution: {input_shape[0]}x{input_shape[1]}x{input_shape[2]}
# ==============================================================================

# Normalization parameters (rescale [0, 255] RGB pixel range to [0.0, 1.0])
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])

# Hardware allocation configuration for Hailo-8L
set_power_mode(ULTRA_PERFORMANCE)

# Performance optimization allocation for 4 unified defect classes
performance_param(compiler_optimization_level=max)
"""
    with open(alls_path, "w", encoding="utf-8") as f:
        f.write(alls_content)

    logger.info("Generated Hailo Model Script (.alls) at: %s", alls_path)
    return alls_path


def generate_synthetic_calibration_images(
    count: int = 64,
    imgsz: Tuple[int, int] = (640, 640),
) -> np.ndarray:
    """Generate realistic synthetic fruit images for calibration when real images are scarce."""
    h, w = imgsz
    images = np.zeros((count, h, w, 3), dtype=np.uint8)

    # Fruit color palettes (BGR format): apple red, guava green, orange, mango greenish-red
    palettes = [
        (40, 40, 210),   # Apple Red
        (50, 200, 100),  # Guava Green
        (20, 140, 240),  # Orange
        (50, 180, 180),  # Mango
    ]

    np.random.seed(42)
    for i in range(count):
        # Conveyor belt dark gray background with subtle noise
        bg_val = np.random.randint(45, 65)
        img = np.full((h, w, 3), bg_val, dtype=np.uint8)
        noise = np.random.randint(-10, 10, (h, w, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Draw 1-3 fruit instances
        num_fruits = np.random.randint(1, 4)
        for _ in range(num_fruits):
            cx = np.random.randint(w // 4, 3 * w // 4)
            cy = np.random.randint(h // 4, 3 * h // 4)
            axes = (np.random.randint(40, 90), np.random.randint(40, 90))
            angle = np.random.randint(0, 180)
            color = palettes[np.random.randint(0, len(palettes))]
            cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, color, -1)

            # Maybe draw a defect spot (dark bruise or scar)
            if np.random.rand() > 0.4:
                dcx = cx + np.random.randint(-20, 20)
                dcy = cy + np.random.randint(-20, 20)
                daxes = (np.random.randint(8, 20), np.random.randint(8, 20))
                cv2.ellipse(img, (dcx, dcy), daxes, 0, 0, 360, (25, 30, 40), -1)

        # Convert to RGB for Hailo input
        images[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return images


def load_calibration_dataset(
    calib_source: Optional[Union[Path, str, np.ndarray]] = None,
    count: int = 64,
    imgsz: Tuple[int, int] = (640, 640),
) -> np.ndarray:
    """Load calibration images for INT8 post-training quantization.

    Args:
        calib_source: Directory with calibration images, single image, or numpy array.
        count: Desired number of calibration samples (default 64).
        imgsz: Image resolution (default 640x640).

    Returns:
        Numpy array of shape (N, H, W, 3) in RGB format.
    """
    if isinstance(calib_source, np.ndarray):
        if calib_source.ndim == 4 and calib_source.shape[1:3] == imgsz:
            logger.info("Using provided calibration numpy array with %d samples", len(calib_source))
            return calib_source[:count]

    image_paths: List[Path] = []
    if calib_source is not None:
        source_path = Path(calib_source).resolve()
        if source_path.is_dir():
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                image_paths.extend(source_path.rglob(ext))
        elif source_path.is_file():
            image_paths.append(source_path)

    # If no source provided or no images found, search fallback datasets in project
    if not image_paths:
        fallback_dirs = [
            REPO_ROOT / "data" / "external" / "apple_defects" / "images" / "train",
            REPO_ROOT / "data" / "external" / "apple_defects" / "train" / "images",
            REPO_ROOT / "data" / "compound_fruits" / "images" / "train",
        ]
        for fdir in fallback_dirs:
            if fdir.is_dir():
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    image_paths.extend(fdir.glob(ext))
                if image_paths:
                    logger.info("Found %d calibration images in %s", len(image_paths), fdir)
                    break

    loaded_images: List[np.ndarray] = []
    for p in image_paths[:count]:
        try:
            bgr = cv2.imread(str(p))
            if bgr is not None:
                resized = cv2.resize(bgr, imgsz, interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                loaded_images.append(rgb)
        except Exception as exc:
            logger.debug("Failed reading %s: %s", p, exc)

    if len(loaded_images) >= count:
        calib_arr = np.array(loaded_images[:count], dtype=np.uint8)
        logger.info("Loaded %d real calibration images for INT8 quantization", len(calib_arr))
        return calib_arr

    # If fewer images than count, supplement with synthetic samples
    needed = count - len(loaded_images)
    logger.info(
        "Found %d real calibration images. Supplementing with %d synthetic calibration samples...",
        len(loaded_images),
        needed,
    )
    synthetic_arr = generate_synthetic_calibration_images(count=needed, imgsz=imgsz)

    if loaded_images:
        real_arr = np.array(loaded_images, dtype=np.uint8)
        combined = np.concatenate([real_arr, synthetic_arr], axis=0)
        return combined[:count]
    return synthetic_arr[:count]


def create_simulated_hef(
    hef_path: Path,
    model_name: str = "yolo11n_fruit",
    hw_arch: str = "hailo8l",
    input_shape: Tuple[int, int, int] = (640, 640, 3),
    classes: Optional[List[str]] = None,
) -> Path:
    """Create a simulated HEF container file with metadata header for environments without hailo_sdk."""
    if classes is None:
        classes = list(UNIFIED_DEFECT_CLASSES)

    hef_path = Path(hef_path).resolve()
    hef_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "format": "SIMULATED_HAILO_HEF",
        "version": "1.0.0",
        "model_name": model_name,
        "hw_arch": hw_arch,
        "target": "Raspberry Pi AI HAT",
        "input_shape": [1, input_shape[2], input_shape[0], input_shape[1]],
        "classes": classes,
        "num_classes": len(classes),
        "quantization": "INT8",
        "timestamp": time.time(),
        "created_by": "Agri_Sorter Hailo Export Pipeline",
        "instructions": (
            "This is a simulated HEF artifact for CI and local evaluation. "
            "To generate native binary HEF for production Raspberry Pi AI HAT deployment, "
            "execute compile_hailo_hef.sh on an x86 workstation with Hailo Dataflow Compiler."
        ),
    }

    meta_bytes = json.dumps(metadata, indent=2).encode("utf-8")
    meta_len = len(meta_bytes)

    with open(hef_path, "wb") as f:
        f.write(SIMULATED_HEF_MAGIC)
        f.write(struct.pack("<I", meta_len))
        f.write(meta_bytes)
        # Pad with dummy binary payload
        dummy_weights = np.zeros((1024,), dtype=np.uint8).tobytes()
        f.write(dummy_weights)

    logger.info("Generated simulated HEF artifact at: %s (size: %d bytes)", hef_path, hef_path.stat().st_size)
    return hef_path


def generate_compilation_shell_script(
    script_path: Path,
    onnx_path: Path,
    alls_path: Path,
    hef_path: Path,
    hw_arch: str = "hailo8l",
) -> Path:
    """Generate a ready-to-run shell script for compiling HEF with Hailo DFC on an x86 workstation."""
    script_path = Path(script_path).resolve()
    script_path.parent.mkdir(parents=True, exist_ok=True)

    content = f"""#!/usr/bin/env bash
# ==============================================================================
# Agri_Sorter Hailo Dataflow Compiler (DFC) Compilation Script
# Target: Raspberry Pi AI HAT ({hw_arch})
# ==============================================================================
set -euo pipefail

echo "=================================================================="
echo " Starting Hailo Dataflow Compiler for Raspberry Pi AI HAT ({hw_arch})"
echo "=================================================================="

ONNX_MODEL="{onnx_path}"
ALLS_SCRIPT="{alls_path}"
OUTPUT_HEF="{hef_path}"
CALIB_COUNT=64

if ! command -v hailo &> /dev/null; then
    echo "[-] hailo CLI not found in PATH."
    echo "    Please activate your Hailo DFC virtualenv:"
    echo "    source /path/to/hailo_virtualenv/bin/activate"
    exit 1
fi

echo "[+] Step 1: Parsing ONNX model to Hailo Archive (HAR)..."
hailo parser onnx "$ONNX_MODEL" --hw-arch {hw_arch}

HAR_FILE="${{ONNX_MODEL%.*}}.har"

echo "[+] Step 2: Optimizing with INT8 Post-Training Quantization..."
hailo optimize "$HAR_FILE" --model-script "$ALLS_SCRIPT" --hw-arch {hw_arch}

echo "[+] Step 3: Compiling to Hailo Executable Format (HEF)..."
hailo compiler "$HAR_FILE" --hw-arch {hw_arch} --output "$OUTPUT_HEF"

echo "[+] Compilation finished successfully! Output: $OUTPUT_HEF"
"""
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)
    script_path.chmod(0o755)

    logger.info("Generated standalone Hailo compilation script at: %s", script_path)
    return script_path


def compile_to_hef(
    onnx_path: Path,
    alls_path: Path,
    calib_data: np.ndarray,
    hef_path: Path,
    hw_arch: str = "hailo8l",
    model_name: str = "yolo11n_fruit",
    classes: Optional[List[str]] = None,
) -> Path:
    """Compile ONNX model to Hailo HEF using Hailo Dataflow Compiler (or simulation fallback)."""
    if classes is None:
        classes = list(UNIFIED_DEFECT_CLASSES)

    onnx_path = Path(onnx_path).resolve()
    alls_path = Path(alls_path).resolve()
    hef_path = Path(hef_path).resolve()
    hef_path.parent.mkdir(parents=True, exist_ok=True)

    if check_hailo_sdk():
        logger.info("Hailo Dataflow Compiler detected. Starting native HEF compilation...")
        try:
            from hailo_sdk_client import ClientRunner

            logger.info("Initializing ClientRunner targeting hw_arch=%s...", hw_arch)
            runner = ClientRunner(hw_arch=hw_arch)

            logger.info("Translating ONNX model from %s...", onnx_path)
            runner.translate_onnx_model(str(onnx_path), model_name=model_name)

            if alls_path.is_file():
                logger.info("Loading model script %s...", alls_path)
                runner.load_model_script(str(alls_path))

            logger.info("Running INT8 quantization optimization (%d calibration frames)...", len(calib_data))
            runner.optimize(calib_data)

            logger.info("Compiling model into HEF...")
            hef_binary = runner.compile()

            with open(hef_path, "wb") as f:
                f.write(hef_binary)

            logger.info("Native HEF compilation succeeded! Saved to: %s", hef_path)
            return hef_path
        except Exception as exc:
            logger.error("Native Hailo DFC compilation failed: %s. Falling back to simulation.", exc)

    # Fallback when hailo_sdk is not installed
    print("\n" + "=" * 76)
    print(" ℹ️  HAILO DATAFLOW COMPILER (hailo_sdk) NOT DETECTED IN ENVIRONMENT")
    print("=" * 76)
    print(f" Target Accelerator : Raspberry Pi AI HAT ({hw_arch})")
    print(f" Source ONNX Model  : {onnx_path}")
    print(f" Model Script (.alls): {alls_path}")
    print(f" Target HEF Output  : {hef_path}")
    print(f" Calibration Samples: {len(calib_data)} frames prepared")
    print("-" * 76)
    print(" Generated Files:")
    print(f"   1. ONNX Model (static 640x640): {onnx_path}")
    print(f"   2. Hailo Model Script (.alls) : {alls_path}")
    print(f"   3. Standalone Script          : {hef_path.parent / 'compile_hailo_hef.sh'}")
    print(f"   4. Simulated HEF Binary       : {hef_path}")
    print("-" * 76)
    print(" Workstation Compilation Instructions:")
    print("   To compile native HEF for deployment on the Raspberry Pi AI HAT,")
    print("   install Hailo Dataflow Compiler v3.28+ on an x86 workstation:")
    print("     pip install hailo_dataflow_compiler-*.whl")
    print(f"     bash {hef_path.parent / 'compile_hailo_hef.sh'}")
    print("=" * 76 + "\n")

    create_simulated_hef(
        hef_path=hef_path,
        model_name=model_name,
        hw_arch=hw_arch,
        input_shape=(640, 640, 3),
        classes=classes,
    )
    generate_compilation_shell_script(
        script_path=hef_path.parent / "compile_hailo_hef.sh",
        onnx_path=onnx_path,
        alls_path=alls_path,
        hef_path=hef_path,
        hw_arch=hw_arch,
    )
    return hef_path


def run_export_pipeline(
    pt_model: Optional[Path] = None,
    onnx_model: Optional[Path] = None,
    hef_output: Path = REPO_ROOT / "models" / "detector" / "yolo11n_fruit.hef",
    alls_output: Path = REPO_ROOT / "models" / "detector" / "yolo11n_fruit.alls",
    calib_dir: Optional[Path] = None,
    calib_count: int = 64,
    hw_arch: str = "hailo8l",
    imgsz: int = 640,
    opset: int = 17,
    classes: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Execute end-to-end export pipeline: PyTorch -> ONNX -> Hailo HEF."""
    if classes is None:
        classes = list(UNIFIED_DEFECT_CLASSES)

    # Resolve default PyTorch model path (prefer yolo11n_fruit.pt, fall back to yolo11n.pt)
    if pt_model is None:
        cand_pt = REPO_ROOT / "models" / "detector" / "yolo11n_fruit.pt"
        pt_model = cand_pt if cand_pt.is_file() else REPO_ROOT / "models" / "detector" / "yolo11n.pt"
    pt_model = Path(pt_model).resolve()

    # Resolve default ONNX model path (prefer yolo11n_fruit.onnx, fall back to yolo11n.onnx)
    if onnx_model is None:
        cand_onnx = REPO_ROOT / "models" / "detector" / "yolo11n_fruit.onnx"
        onnx_model = cand_onnx if cand_onnx.is_file() else REPO_ROOT / "models" / "detector" / "yolo11n.onnx"
    onnx_model = Path(onnx_model).resolve()

    hef_output = Path(hef_output).resolve()
    alls_output = Path(alls_output).resolve()

    logger.info("Starting end-to-end export pipeline to Raspberry Pi AI HAT (%s)...", hw_arch)

    # 1. Export PyTorch to ONNX
    if pt_model.is_file():
        export_pt_to_onnx(pt_path=pt_model, onnx_path=onnx_model, imgsz=imgsz, opset=opset)
    elif onnx_model.is_file():
        logger.info("Using existing ONNX model: %s", onnx_model)
    else:
        logger.warning("No .pt or .onnx found at defaults. Creating target directories.")
        onnx_model.parent.mkdir(parents=True, exist_ok=True)

    # 2. Generate Hailo Model Script (.alls)
    generate_hailo_model_script(
        alls_path=alls_output,
        model_name="yolo11n_fruit",
        hw_arch=hw_arch,
        input_shape=(imgsz, imgsz, 3),
    )

    # 3. Load / Prepare Calibration Dataset
    calib_data = load_calibration_dataset(
        calib_source=calib_dir,
        count=calib_count,
        imgsz=(imgsz, imgsz),
    )

    # 4. Compile to HEF (Native or Simulated Fallback)
    compile_to_hef(
        onnx_path=onnx_model,
        alls_path=alls_output,
        calib_data=calib_data,
        hef_path=hef_output,
        hw_arch=hw_arch,
        model_name="yolo11n_fruit",
        classes=classes,
    )

    # 5. Write metadata companion JSON
    companion_json = hef_output.parent / f"{hef_output.stem}.json"
    companion_meta = {
        "task": "detect",
        "model_name": hef_output.stem,
        "hw_arch": hw_arch,
        "imgsz": imgsz,
        "names": {i: name for i, name in enumerate(classes)},
        "num_classes": len(classes),
        "hef_path": str(hef_output),
        "onnx_path": str(onnx_model),
    }
    with open(companion_json, "w", encoding="utf-8") as f:
        json.dump(companion_meta, f, indent=2)
    logger.info("Exported metadata companion JSON: %s", companion_json)

    return {
        "onnx": onnx_model,
        "alls": alls_output,
        "hef": hef_output,
        "json": companion_json,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Agri_Sorter PyTorch -> Static ONNX -> Hailo-8L HEF Export Pipeline."
    )
    parser.add_argument(
        "--pt-model",
        type=Path,
        default=None,
        help="Path to PyTorch (.pt) weights (default: models/detector/yolo11n_fruit.pt or yolo11n.pt).",
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=None,
        help="Path to target static ONNX model (default: models/detector/yolo11n_fruit.onnx or yolo11n.onnx).",
    )
    parser.add_argument(
        "--hef-output",
        type=Path,
        default=REPO_ROOT / "models" / "detector" / "yolo11n_fruit.hef",
        help="Path for compiled Hailo HEF file.",
    )
    parser.add_argument(
        "--alls-output",
        type=Path,
        default=REPO_ROOT / "models" / "detector" / "yolo11n_fruit.alls",
        help="Path for Hailo model script (.alls).",
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        default=None,
        help="Directory containing images for INT8 calibration.",
    )
    parser.add_argument("--calib-count", type=int, default=64, help="Number of calibration images (default: 64).")
    parser.add_argument("--hw-arch", default="hailo8l", help="Target Hailo architecture (default: hailo8l).")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolution (default: 640).")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset (default: 17).")

    args = parser.parse_args()

    run_export_pipeline(
        pt_model=args.pt_model,
        onnx_model=args.onnx_model,
        hef_output=args.hef_output,
        alls_output=args.alls_output,
        calib_dir=args.calib_dir,
        calib_count=args.calib_count,
        hw_arch=args.hw_arch,
        imgsz=args.imgsz,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
