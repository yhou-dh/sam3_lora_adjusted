"""
extract_armed.py
Extracts armed humans by detecting overlap between human and polearm masks
within illustration context.

Logic:
  For each image:
    For each human detection:
      Check if human mask overlaps OR bbox is near any polearm
      → armed: save to human_armed/
      → unarmed: save to human_unarmed/
    For each illustration detection:
      Save bbox crop to illustration_bbox/

Usage:
    python3.10 extract_armed.py \
        --predictions_root predictions/lora \
        --image_root finerbook \
        --output_root predictions/armed \
        --padding 10 \
        --min_score 0.8 \
        --overlap_dilation 40 \
        --box_margin 50

    # Single book test:
    python3.10 extract_armed.py \
        --predictions_root predictions/lora \
        --image_root finerbook \
        --output_root predictions/armed \
        --book bdj_qm
"""

import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from pycocotools import mask as mask_utils

# Prompt index mapping
IDX_TO_CLASS = {0: "human", 1: "illustration", 2: "polearm"}

POLEARM_MIN_SCORE = 0.85  # lower threshold for polearms


def decode_rle(rle: dict) -> np.ndarray:
    return mask_utils.decode(rle).astype(np.uint8)


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    kernel = np.ones((pixels, pixels), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def masks_overlap(mask1: np.ndarray, mask2: np.ndarray, dilation: int = 20) -> bool:
    """Check if two masks overlap or are within `dilation` pixels of each other."""
    dilated1 = dilate_mask(mask1, dilation)
    return bool(np.any(dilated1 & mask2))


def boxes_near(box1: list, box2: list, margin: int = 50) -> bool:
    """Check if two bounding boxes are close or overlapping within margin pixels."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1_1 -= margin
    y1_1 -= margin
    x2_1 += margin
    y2_1 += margin
    return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)


def crop_rgba(img_array: np.ndarray, mask: np.ndarray, padding: int) -> Image.Image:
    """Crop image to mask region with padding, transparent background."""
    mask_dilated = dilate_mask(mask, padding)
    rgba = np.zeros((*img_array.shape[:2], 4), dtype=np.uint8)
    rgba[:, :, :3] = img_array[:, :, :3]
    rgba[:, :, 3]  = (mask_dilated * 255).astype(np.uint8)

    rows = np.any(mask_dilated, axis=1)
    cols = np.any(mask_dilated, axis=0)
    if not rows.any() or not cols.any():
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return Image.fromarray(rgba[rmin:rmax+1, cmin:cmax+1], 'RGBA')


def bbox_crop(img_array: np.ndarray, box: list, padding: int = 10) -> Image.Image:
    """Crop image to bounding box [x1, y1, x2, y2] with padding."""
    h, w = img_array.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return Image.fromarray(img_array[y1:y2, x1:x2], 'RGB')


def process_book(pred_path: Path, image_dir: Path, output_root: Path,
                 padding: int, min_score: float, overlap_dilation: int,
                 box_margin: int):

    with open(pred_path) as f:
        predictions = json.load(f)

    bookname = pred_path.parent.parent.name

    # Output directories
    armed_dir   = output_root / bookname / "human_armed"
    unarmed_dir = output_root / bookname / "human_unarmed"
    illus_dir   = output_root / bookname / "illustration_bbox"
    armed_dir.mkdir(parents=True, exist_ok=True)
    unarmed_dir.mkdir(parents=True, exist_ok=True)
    illus_dir.mkdir(parents=True, exist_ok=True)

    stats = {"armed": 0, "unarmed": 0, "illustration": 0, "skipped": 0}

    for item in predictions:
        fname    = item["file_name"]
        img_path = image_dir / fname

        if not img_path.exists():
            stats["skipped"] += 1
            continue

        img_array = np.array(Image.open(img_path).convert("RGB"))
        stem      = Path(fname).stem

        # Collect detections by class
        humans        = []
        polearms      = []
        illustrations = []

        for det in item["detections"]:
            cls       = IDX_TO_CLASS.get(int(det["prompt"]), str(det["prompt"]))
            scores    = det.get("scores", [])
            boxes     = det.get("boxes", [])
            masks_rle = det.get("masks_rle", [])

            for i, (rle, score, box) in enumerate(zip(masks_rle, scores, boxes)):
                threshold = POLEARM_MIN_SCORE if cls == "polearm" else min_score
                if score < threshold:
                    continue
                mask  = decode_rle(rle)
                entry = (mask, score, box, i)
                if cls == "human":
                    humans.append(entry)
                elif cls == "polearm":
                    polearms.append(entry)
                elif cls == "illustration":
                    illustrations.append(entry)

        # Save illustration bbox crops
        for i, (mask, score, box, idx) in enumerate(illustrations):
            result = bbox_crop(img_array, box, padding)
            if result:
                result.save(str(illus_dir / f"{stem}_il_{i+1}.jpg"))
                stats["illustration"] += 1

        # Match humans to polearms via mask overlap OR bbox proximity
        for i, (h_mask, h_score, h_box, h_idx) in enumerate(humans):
            overlapping_polearms = [
                (p_mask, p_box) for (p_mask, _, p_box, _) in polearms
                if masks_overlap(h_mask, p_mask, dilation=overlap_dilation)
                or boxes_near(h_box, p_box, margin=box_margin)
            ]
            is_armed = len(overlapping_polearms) > 0

            if is_armed:
                # Combine human mask with all overlapping polearm masks
                combined_mask = h_mask.copy()
                for p_mask, _ in overlapping_polearms:
                    combined_mask = np.maximum(combined_mask, p_mask)

                result = crop_rgba(img_array, combined_mask, padding)
                if result is None:
                    continue
                result.save(str(armed_dir / f"{stem}_hm_armed_{i+1}.png"))
                stats["armed"] += 1
            else:
                result = crop_rgba(img_array, h_mask, padding)
                if result is None:
                    continue
                result.save(str(unarmed_dir / f"{stem}_hm_unarmed_{i+1}.png"))
                stats["unarmed"] += 1

    print(f"  armed={stats['armed']}  unarmed={stats['unarmed']}  "
          f"illustration={stats['illustration']}  skipped={stats['skipped']}")
    return stats


def process_all(predictions_root: str, image_root: str, output_root: str,
                padding: int, min_score: float, overlap_dilation: int,
                box_margin: int, book: str = None):

    predictions_root = Path(predictions_root)
    image_root       = Path(image_root)
    output_root      = Path(output_root)

    if book:
        book_folders = [predictions_root / book]
    else:
        book_folders = sorted([
            d for d in predictions_root.iterdir()
            if d.is_dir() and (d / "summaries" / "book_predictions.json").exists()
        ])

    if not book_folders:
        print("No book folders with book_predictions.json found.")
        return

    print(f"\n{'='*60}")
    print(f"Armed Human Extraction")
    print(f"  Books            : {len(book_folders)}")
    print(f"  Padding          : {padding}px")
    print(f"  Min score        : {min_score}")
    print(f"  Overlap dilation : {overlap_dilation}px")
    print(f"  Box margin       : {box_margin}px")
    print(f"{'='*60}")

    grand = {"armed": 0, "unarmed": 0, "illustration": 0, "skipped": 0}

    for book_dir in book_folders:
        bookname  = book_dir.name
        pred_path = book_dir / "summaries" / "book_predictions.json"
        image_dir = image_root / bookname

        print(f"\n{'─'*50}")
        print(f"Book: {bookname}")

        if not image_dir.exists():
            print(f"  Image folder not found, skipping.")
            continue

        stats = process_book(
            pred_path        = pred_path,
            image_dir        = image_dir,
            output_root      = output_root,
            padding          = padding,
            min_score        = min_score,
            overlap_dilation = overlap_dilation,
            box_margin       = box_margin,
        )
        for k in grand:
            grand[k] += stats[k]

    print(f"\n{'='*60}")
    print(f"All done!")
    print(f"  Armed humans   : {grand['armed']}")
    print(f"  Unarmed humans : {grand['unarmed']}")
    print(f"  Illustrations  : {grand['illustration']}")
    print(f"  Skipped images : {grand['skipped']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_root",  required=True)
    parser.add_argument("--image_root",        required=True)
    parser.add_argument("--output_root",       required=True)
    parser.add_argument("--book",              default=None)
    parser.add_argument("--padding",           type=int,   default=10)
    parser.add_argument("--min_score",         type=float, default=0.8)
    parser.add_argument("--overlap_dilation",  type=int,   default=40)
    parser.add_argument("--box_margin",        type=int,   default=50,
                        help="Pixels to expand human bbox when checking polearm proximity (default 50)")
    args = parser.parse_args()

    process_all(
        predictions_root = args.predictions_root,
        image_root       = args.image_root,
        output_root      = args.output_root,
        padding          = args.padding,
        min_score        = args.min_score,
        overlap_dilation = args.overlap_dilation,
        box_margin       = args.box_margin,
        book             = args.book,
    )
