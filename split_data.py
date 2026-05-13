"""
split_data.py
Randomly splits images (and their per-image JSON annotations) from a source
folder into train / valid / test subfolders.

Useful for randomising a dataset before training. Not required if you are
using Label Studio export directly (which already provides split COCO JSONs).

Usage:
    # 80/10/10 split
    python3 split_data.py \
        --source data/all_images \
        --output data \
        --train 0.8 --valid 0.1 --test 0.1 \
        --seed 42

    # 90/10 split (no test set)
    python3 split_data.py \
        --source data/all_images \
        --output data \
        --train 0.9 --valid 0.1 \
        --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  required=True,
                        help="Folder containing images (and optional per-image JSON files)")
    parser.add_argument("--output",  required=True,
                        help="Root output folder — subfolders train/valid/test created here")
    parser.add_argument("--train",   type=float, default=0.8, help="Train fraction (default 0.8)")
    parser.add_argument("--valid",   type=float, default=0.1, help="Valid fraction (default 0.1)")
    parser.add_argument("--test",    type=float, default=0.1, help="Test fraction (default 0.1)")
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--copy",    action="store_true",
                        help="Copy files instead of moving them (default: move)")
    return parser.parse_args()


def main():
    args = parse_args()

    total = args.train + args.valid + args.test
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"train + valid + test must sum to 1.0, got {total:.3f}")

    source = Path(args.source)
    output = Path(args.output)

    image_files = sorted([
        f for f in source.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_files:
        print(f"❌ No images found in {source}")
        return

    random.seed(args.seed)
    random.shuffle(image_files)

    n = len(image_files)
    n_train = int(n * args.train)
    n_valid   = int(n * args.val)
    n_test  = n - n_train - n_valid  # remainder goes to test

    splits = {}
    if args.train > 0:
        splits["train"] = image_files[:n_train]
    if args.valid > 0:
        splits["valid"]   = image_files[n_train:n_train + n_valid]
    if args.test > 0 and n_test > 0:
        splits["test"]  = image_files[n_train + n_valid:]

    print(f"Total images : {n}")
    for split, files in splits.items():
        print(f"  {split:6s}: {len(files)}")

    transfer = shutil.copy2 if args.copy else shutil.move
    action   = "Copying" if args.copy else "Moving"

    for split, files in splits.items():
        dest = output / split
        dest.mkdir(parents=True, exist_ok=True)
        print(f"\n{action} {len(files)} files → {dest}")

        for img_path in files:
            transfer(str(img_path), str(dest / img_path.name))

            # Move matching JSON annotation if it exists
            json_path = source / f"{img_path.stem}.json"
            if json_path.exists():
                transfer(str(json_path), str(dest / json_path.name))

    print("\n✅ Split complete.")


if __name__ == "__main__":
    main()
