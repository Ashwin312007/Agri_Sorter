#!/usr/bin/env python3
"""Download Pomegranate datasets for damage detection.

Downloads:
- RF-P01: catkin-d68vv/fresh_pomegranate-0ppkl (v1)
- RF-P02: fruit-spoilage-detection/a-b-o-l-p-g-spoiled-detection (v2)
- RF-P03: im-ririo/pomegranate-disease-detection-29ocz-yz4hn (v2)
- RF-P04: bin-wnwyu/pomegranate-hhj6l (v1)
"""
import os
import sys
from pathlib import Path
from roboflow import Roboflow

API_KEY = os.environ.get("ROBOFLOW_API_KEY", "PjWN0R9ZZ8SfBRq6nxBC")
BASE_DIR = Path("raw_downloads/pomegranate")

DATASETS = [
    {
        "id": "rf_p01",
        "workspace": "catkin-d68vv",
        "project": "fresh_pomegranate-0ppkl",
        "version": 1,
    },
    {
        "id": "rf_p02",
        "workspace": "fruit-spoilage-detection",
        "project": "a-b-o-l-p-g-spoiled-detection",
        "version": 2,
    },
    {
        "id": "rf_p03",
        "workspace": "im-ririo",
        "project": "pomegranate-disease-detection-29ocz-yz4hn",
        "version": 2,
    },
    {
        "id": "rf_p04",
        "workspace": "bin-wnwyu",
        "project": "pomegranate-hhj6l",
        "version": 1,
    },
]


def download_all():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=API_KEY)

    for item in DATASETS:
        target_dir = BASE_DIR / item["id"]
        yaml_path = target_dir / "data.yaml"
        if yaml_path.exists():
            print(f"[SKIP] {item['id']} already downloaded at {target_dir}")
            continue

        print(f"\n[DOWNLOAD] {item['id']} ({item['workspace']}/{item['project']} v{item['version']})...")
        try:
            proj = rf.workspace(item["workspace"]).project(item["project"])
            ver = proj.version(item["version"])
            ver.download("yolov5", location=str(target_dir))
            print(f"[OK] {item['id']} downloaded successfully to {target_dir}")
        except Exception as e:
            print(f"[ERROR] Failed downloading {item['id']}: {e}")
            sys.exit(1)


if __name__ == "__main__":
    download_all()
