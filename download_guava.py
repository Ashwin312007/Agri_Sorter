import os
import shutil
from pathlib import Path
from roboflow import Roboflow

API_KEY = "PjWN0R9ZZ8SfBRq6nxBC"
rf = Roboflow(api_key=API_KEY)

datasets = [
    {
        "name": "rf04",
        "workspace": "chisilia",
        "project": "detection-guava-disease",
        "version": 1,
        "dest": Path("raw_downloads/guava/rf04")
    },
    {
        "name": "rf05",
        "workspace": "guava-h8gxt",
        "project": "guava-disease-mozns",
        "version": 1,
        "dest": Path("raw_downloads/guava/rf05")
    }
]

for ds in datasets:
    dest = ds["dest"]
    print(f"\n--- Downloading {ds['name']}: {ds['workspace']}/{ds['project']} v{ds['version']} ---")
    if dest.exists():
        print(f"Directory {dest} already exists. Cleaning up...")
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    
    project = rf.workspace(ds["workspace"]).project(ds["project"])
    version = project.version(ds["version"])
    downloaded = version.download("yolov5", location=str(dest), overwrite=True)
    print(f"Downloaded {ds['name']} to {downloaded.location}")

print("\nAll Guava downloads finished successfully!")
