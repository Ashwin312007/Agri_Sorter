#!/usr/bin/env python3
"""Build self-contained Raspberry Pi 4 deployment bundle.

Gathers:
- Models: best.onnx, best_mango.onnx, best_guava.onnx, best_pomegranate.onnx
- Edge code: detect.py, web_detect.py, check_env.py
- Setup scripts: setup_pi.sh, run_pi.sh
- Service files: systemd/fruit_sorter.service
- Deployment docs: README.md

Produces:
- Directory: deploy_rpi4/
- Archive: fruit_sorter_rpi4.zip
"""
import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEPLOY_DIR = ROOT / "deploy_rpi4"
ZIP_PATH = ROOT / "fruit_sorter_rpi4.zip"

ITEMS_TO_COPY = [
    # Models
    ("best.onnx", "best.onnx"),
    ("best_mango.onnx", "best_mango.onnx"),
    ("best_guava.onnx", "best_guava.onnx"),
    ("best_pomegranate.onnx", "best_pomegranate.onnx"),

    # Edge code
    ("detect.py", "detect.py"),
    ("web_detect.py", "web_detect.py"),
    ("check_env.py", "check_env.py"),
    ("find_cam.py", "find_cam.py"),

    # Setup scripts
    ("setup_pi.sh", "setup_pi.sh"),
    ("run_pi.sh", "run_pi.sh"),
    ("find_cam.sh", "find_cam.sh"),

    # Systemd
    ("systemd/fruit_sorter.service", "systemd/fruit_sorter.service"),

    # Documentation
    ("README.md", "README.md"),
]


def build_package():
    print(f"[1/4] Preparing deployment directory at {DEPLOY_DIR}...")
    if DEPLOY_DIR.exists():
        shutil.rmtree(DEPLOY_DIR)
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    (DEPLOY_DIR / "systemd").mkdir(parents=True, exist_ok=True)
    (DEPLOY_DIR / "saved").mkdir(parents=True, exist_ok=True)

    print("[2/4] Copying deployment artifacts...")
    for src_rel, dst_rel in ITEMS_TO_COPY:
        src = ROOT / src_rel
        dst = DEPLOY_DIR / dst_rel
        if not src.exists():
            print(f"  [ERROR] Required file missing: {src}")
            return False
        shutil.copy2(src, dst)
        size_mb = src.stat().st_size / (1024 * 1024)
        print(f"  Copied {src_rel:30s} -> {dst_rel} ({size_mb:.2f} MB)")

    # 3. Create Zip file
    print(f"\n[3/4] Packaging into zip archive: {ZIP_PATH}...")
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in DEPLOY_DIR.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(DEPLOY_DIR)
                zipf.write(file_path, arcname)

    zip_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"  Created {ZIP_PATH.name} ({zip_mb:.2f} MB)")

    print("\n[4/4] Package verification:")
    with zipfile.ZipFile(ZIP_PATH, "r") as zipf:
        namelist = zipf.namelist()
        print(f"  Zip contents ({len(namelist)} files):")
        for name in sorted(namelist):
            print(f"    - {name}")

    print("\n=== Raspberry Pi 4 Deployment Package Ready! ===")
    return True


if __name__ == "__main__":
    build_package()
