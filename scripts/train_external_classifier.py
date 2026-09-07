"""Train and test a lightweight apple fresh-versus-rotten classifier."""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/external/apple_freshness")
    parser.add_argument("--model", default="yolo11n-cls.pt")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data = Path(args.data)
    for split in ("train", "val", "test"):
        if not (data / split).is_dir():
            raise FileNotFoundError(f"Missing {data / split}; run the prepare script first.")
    output = Path("models/external")
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz,
                device=args.device, project=str(output), name="apple_freshness",
                patience=8, exist_ok=True)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError("Training ended without producing best.pt")
    (output / "best.pt").write_bytes(best.read_bytes())
    print(f"MODEL={output / 'best.pt'}")
    print("Run scripts/evaluate_external_classifier.py to score the held-out test set.")


if __name__ == "__main__":
    main()
