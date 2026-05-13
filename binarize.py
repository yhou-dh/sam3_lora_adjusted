"""
binarize.py
Applies Otsu binarization to all images in nested subfolders.
Saves results to a mirrored output directory structure.

Usage:
    python3 binarize.py --input_root data --output_root data_binary
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image


def binarize_image(img_path: Path, output_path: Path):
    img = Image.open(img_path).convert("L")  # grayscale
    arr = np.array(img)

    # Otsu threshold
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary).save(str(output_path))


def process_all(input_root: str, output_root: str):
    input_root  = Path(input_root)
    output_root = Path(output_root)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    all_images = [
        p for p in input_root.rglob("*")
        if p.suffix.lower() in extensions
    ]

    print(f"Found {len(all_images)} images under {input_root}")

    for i, img_path in enumerate(all_images):
        rel_path    = img_path.relative_to(input_root)
        output_path = output_root / rel_path.with_suffix(".png")
        binarize_image(img_path, output_path)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(all_images)}...")

    print(f"✅ Done. Saved to {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root",  required=True, help="Root folder with images")
    parser.add_argument("--output_root", required=True, help="Root folder for binarized output")
    args = parser.parse_args()
    process_all(args.input_root, args.output_root)
