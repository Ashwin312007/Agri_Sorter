"""Create deterministic train/validation/test splits for apple freshness images."""
import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def copy_split(files, destination, ratios):
    random.shuffle(files)
    total = len(files)
    train_end = round(total * ratios[0])
    val_end = train_end + round(total * ratios[1])
    splits = {"train": files[:train_end], "val": files[train_end:val_end], "test": files[val_end:]}
    for split, items in splits.items():
        target = destination / split / files[0].parent.name.lower()
        target.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(items):
            shutil.copy2(image, target / f"{index:04d}_{image.name}")
    return {name: len(items) for name, items in splits.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Apple directory containing Fresh/ and Rotten/")
    parser.add_argument("--output", default="data/external/apple_freshness")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    source, output = Path(args.source), Path(args.output)
    if output.exists():
        raise FileExistsError(f"{output} already exists. Remove it only if you want a new split.")
    counts = {}
    for class_name in ("Fresh", "Rotten"):
        files = [p for p in (source / class_name).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        if not files:
            raise FileNotFoundError(f"No images found in {source / class_name}")
        counts[class_name] = copy_split(files, output, (.70, .15, .15))
    print(counts)


if __name__ == "__main__":
    main()
