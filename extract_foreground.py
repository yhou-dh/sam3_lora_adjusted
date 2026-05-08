"""
extract_foreground.py
Extracts foreground objects from images using RLE masks saved during inference.
Loops over all book folders automatically.
Outputs PNG files with transparent backgrounds, preserving strokes via dilation.

Usage:
    # All books:
    python3.10 extract_foreground.py \
        --predictions_root predictions/lora_masks \
        --image_root finerbook \
        --padding 10 \
        --min_score 0.8

    # Single book:
    python3.10 extract_foreground.py \
        --predictions_root predictions/lora_masks \
        --image_root finerbook \
        --book bdj_qm \
        --padding 10
"""

import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from pycocotools import mask as mask_utils


def decode_rle(rle: dict) -> np.ndarray:
    """Decode RLE dict to H x W binary numpy array."""
    return mask_utils.decode(rle).astype(np.uint8)


def dilate_mask(mask: np.ndarray, padding: int) -> np.ndarray:
    """Expand mask outward by `padding` pixels to preserve strokes/edges."""
    kernel = np.ones((padding, padding), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def extract_foreground(img_array: np.ndarray, mask: np.ndarray) -> Image.Image:
    """
    Apply mask as alpha channel and crop to bounding box.
    Returns RGBA PIL image with transparent background.
    """
    rgba = np.zeros((*img_array.shape[:2], 4), dtype=np.uint8)
    rgba[:, :, :3] = img_array[:, :, :3]
    rgba[:, :, 3]  = (mask * 255).astype(np.uint8)

    # Crop to tight bounding box of mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = rgba[rmin:rmax+1, cmin:cmax+1]

    return Image.fromarray(cropped, 'RGBA')


KEEP_CLASSES = {"0", "1", "2"}  # 0=human, 1=illustration, 2=polearm
CLASS_NAMES  = {"0": "human", "1": "illustration", "2": "polearm"}
CLASS_ABBR   = {"0": "hm",    "1": "il",           "2": "ar"}


def process_book(pred_path: Path, image_dir: Path, output_dir: Path,
                 padding: int, min_score: float, min_detections: int):

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(pred_path) as f:
        predictions = json.load(f)

    print(f"\n  Images     : {len(predictions)}")
    print(f"  Image dir  : {image_dir}")
    print(f"  Output dir : {output_dir}")

    total_saved   = 0
    total_skipped = 0

    for item in predictions:
        fname    = item["file_name"]
        img_path = image_dir / fname

        if not img_path.exists():
            print(f"    ⚠ Image not found: {fname}, skipping.")
            total_skipped += 1
            continue

        img_array = np.array(Image.open(img_path).convert("RGB"))

        for det in item["detections"]:
            prompt         = det["prompt"]
            scores         = det.get("scores", [])
            masks_rle      = det.get("masks_rle", [])
            num_detections = det.get("num_detections", 0)

            if num_detections < min_detections:
                continue

            # Skip classes we don't need
            if str(prompt) not in KEEP_CLASSES:
                continue

            if not masks_rle:
                continue

            for i, (rle, score) in enumerate(zip(masks_rle, scores)):
                if score < min_score:
                    continue

                # Decode and dilate mask
                mask         = decode_rle(rle)
                mask_dilated = dilate_mask(mask, padding)

                # Extract foreground
                result = extract_foreground(img_array, mask_dilated)
                if result is None:
                    continue

                # Save into class subfolder
                class_dir = output_dir / CLASS_NAMES.get(str(prompt), str(prompt))
                class_dir.mkdir(parents=True, exist_ok=True)
                stem       = Path(fname).stem
                clean_stem = stem.split('__')[-1] if '__' in stem else stem
                class_abbr = CLASS_ABBR.get(str(prompt), str(prompt))
                out_fname  = f"{clean_stem}_{class_abbr}_{i+1}.png"
                result.save(str(class_dir / out_fname))
                total_saved += 1

    print(f"  ✅ Saved {total_saved} crops, skipped {total_skipped} images")
    return total_saved, total_skipped


def process_all(predictions_root: str, image_root: str, padding: int,
                min_score: float, min_detections: int, book: str = None):

    predictions_root = Path(predictions_root)
    image_root       = Path(image_root)

    # Find all book folders that have a summaries/book_predictions.json
    if book:
        book_folders = [predictions_root / book]
    else:
        book_folders = sorted([
            d for d in predictions_root.iterdir()
            if d.is_dir() and (d / "summaries" / "book_predictions.json").exists()
        ])

    if not book_folders:
        print("❌ No book folders with book_predictions.json found.")
        return

    print(f"\n{'='*60}")
    print(f"Foreground Extraction")
    print(f"  Books found : {len(book_folders)}")
    print(f"  Padding     : {padding}px")
    print(f"  Min score   : {min_score}")
    print(f"{'='*60}")

    grand_total_saved   = 0
    grand_total_skipped = 0

    for book_dir in book_folders:
        bookname   = book_dir.name
        pred_path  = book_dir / "summaries" / "book_predictions.json"
        image_dir  = image_root / bookname
        output_dir = book_dir / "foreground"

        print(f"\n{'─'*50}")
        print(f"Book: {bookname}")

        if not image_dir.exists():
            print(f"  ⚠ Image folder not found: {image_dir}, skipping.")
            continue

        saved, skipped = process_book(
            pred_path      = pred_path,
            image_dir      = image_dir,
            output_dir     = output_dir,
            padding        = padding,
            min_score      = min_score,
            min_detections = min_detections,
        )
        grand_total_saved   += saved
        grand_total_skipped += skipped

    print(f"\n{'='*60}")
    print(f"✅ All books done!")
    print(f"   Total crops saved   : {grand_total_saved}")
    print(f"   Total images skipped: {grand_total_skipped}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_root", required=True,
                        help="Root folder containing book subfolders with summaries/book_predictions.json")
    parser.add_argument("--image_root",       required=True,
                        help="Root folder containing book subfolders with source images")
    parser.add_argument("--book",             default=None,
                        help="Process a single book only (optional)")
    parser.add_argument("--padding",          type=int,   default=10,
                        help="Pixels to dilate mask (preserves strokes, default 10)")
    parser.add_argument("--min_score",        type=float, default=0.8,
                        help="Minimum confidence score (default 0.8)")
    parser.add_argument("--min_detections",   type=int,   default=1,
                        help="Minimum detections per prompt (default 1)")
    args = parser.parse_args()

    process_all(
        predictions_root = args.predictions_root,
        image_root       = args.image_root,
        padding          = args.padding,
        min_score        = args.min_score,
        min_detections   = args.min_detections,
        book             = args.book,
    )
