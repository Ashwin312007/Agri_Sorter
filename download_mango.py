import os
import shutil
from pathlib import Path
from roboflow import Roboflow

API_KEY = "PjWN0R9ZZ8SfBRq6nxBC"
rf = Roboflow(api_key=API_KEY)

datasets = [
    {
        "name": "rf07",
        "workspace": "mintras-workspace",
        "project": "mango-defect-detection-pc4gs",
        "version": 1,
        "dest": Path("raw_downloads/mango/rf07")
    },
    {
        "name": "rf08",
        "workspace": "kaycee-kaye-y-villanueva-mzmks",
        "project": "mango-defect-detection",
        "version": 1,
        "dest": Path("raw_downloads/mango/rf08")
    },
    {
        "name": "rf09",
        "workspace": "mangoclass",
        "project": "mangograding",
        "version": 1,
        "dest": Path("raw_downloads/mango/rf09")
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

print("\nAll downloads finished successfully!")
