"""Train the external-defect segmentation model off-device."""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="YOLO data.yaml")
    p.add_argument("--model", default="yolo11n-seg.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    args = p.parse_args()
    out = Path("models/external")
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                device=args.device, project=str(out), name="apple_defects")
    best = out / "apple_defects" / "weights" / "best.pt"
    if best.exists():
        (out / "best.pt").write_bytes(best.read_bytes())
        print(f"Saved {out / 'best.pt'}")


if __name__ == "__main__":
    main()
