"""Evaluate the trained classifier on the held-out test folder without multiprocessing."""
import argparse
from pathlib import Path
from ultralytics import YOLO

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/external/best.pt")
    parser.add_argument("--test", default="data/external/apple_freshness/test")
    parser.add_argument("--imgsz", type=int, default=224)
    args = parser.parse_args()
    model = YOLO(args.model)
    total = correct = 0
    labels = [p.name.lower() for p in Path(args.test).iterdir() if p.is_dir()]
    confusion = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for actual in labels:
        for image in (Path(args.test) / actual).iterdir():
            if image.suffix.lower() not in EXTENSIONS:
                continue
            result = model.predict(str(image), imgsz=args.imgsz, verbose=False)[0]
            predicted = result.names[result.probs.top1].lower()
            confusion[actual][predicted] = confusion[actual].get(predicted, 0) + 1
            correct += predicted == actual
            total += 1
    print(f"TEST_ACCURACY={correct / total:.4f} ({correct}/{total})")
    print("CONFUSION_MATRIX")
    print("actual,predicted,count")
    for actual, row in confusion.items():
        for predicted, count in row.items():
            print(f"{actual},{predicted},{count}")


if __name__ == "__main__":
    main()
