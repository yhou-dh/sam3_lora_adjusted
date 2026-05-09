"""
extract_armed_bw.py
Extracts armed humans using illustration bbox as context boundary.

Logic:
  For each illustration detection:
    Find all human and polearm masks whose center falls inside the illustration bbox
    
    If 1 human + 1-2 polearms:
      → combine all masks → save as human_armed
    
    If 2+ humans + polearms:
      → assign each polearm to nearest human by center distance
      → save each human+assigned polearms as human_armed
    
    If human only (no polearm inside illustration):
      → save human mask as human_unarmed
    
    Always save illustration bbox crop to illustration_bbox/

Usage:
    python3.10 extract_armed_bw.py \
        --predictions_root predictions/lora \
        --image_root finerbook \
        --output_root predictions/armed_bw \
        --padding 10 \
        --min_score 0.9 \
        --polearm_min_score 0.85

    # Single book test:
    python3.10 extract_armed_bw.py \
        --predictions_root predictions/lora \
        --image_root finerbook \
        --output_root predictions/armed_bw \
        --book zggyjf
"""

import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from pycocotools import mask as mask_utils

IDX_TO_CLASS = {0: "human", 1: "illustration", 2: "polearm"}

MAX_ASSIGN_DISTANCE = 200  # pixels — max distance to assign polearm to human


def decode_rle(rle: dict) -> np.ndarray:
    return mask_utils.decode(rle).astype(np.uint8)


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    kernel = np.ones((pixels, pixels), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def box_center(box: list) -> tuple:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def center_in_box(center: tuple, box: list) -> bool:
    """Check if a center point falls inside a bounding box."""
    cx, cy = center
    x1, y1, x2, y2 = box
    return x1 <= cx <= x2 and y1 <= cy <= y2


def box_distance(box1: list, box2: list) -> float:
    c1 = box_center(box1)
    c2 = box_center(box2)
    return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) ** 0.5


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
                 padding: int, min_score: float, polearm_min_score: float):

    with open(pred_path) as f:
        predictions = json.load(f)

    bookname = pred_path.parent.parent.name

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

        # Collect all detections
        all_humans        = []  # (mask, score, box)
        all_polearms      = []
        all_illustrations = []

        for det in item["detections"]:
            cls       = IDX_TO_CLASS.get(int(det["prompt"]), str(det["prompt"]))
            scores    = det.get("scores", [])
            boxes     = det.get("boxes", [])
            masks_rle = det.get("masks_rle", [])

            threshold = polearm_min_score if cls == "polearm" else min_score

            for rle, score, box in zip(masks_rle, scores, boxes):
                if score < threshold:
                    continue
                mask = decode_rle(rle)
                entry = (mask, score, box)
                if cls == "human":
                    all_humans.append(entry)
                elif cls == "polearm":
                    all_polearms.append(entry)
                elif cls == "illustration":
                    all_illustrations.append(entry)

        # ── Process each illustration as context boundary ─────────────────
        for il_idx, (il_mask, il_score, il_box) in enumerate(all_illustrations):

            # Save illustration bbox crop
            il_result = bbox_crop(img_array, il_box, padding)
            if il_result:
                il_result.save(str(illus_dir / f"{stem}_il_{il_idx+1}.jpg"))
                stats["illustration"] += 1

            # Find humans whose center falls inside this illustration bbox
            humans_in = [
                (mask, score, box) for (mask, score, box) in all_humans
                if center_in_box(box_center(box), il_box)
            ]

            # Find polearms whose center falls inside this illustration bbox
            polearms_in = [
                (mask, score, box) for (mask, score, box) in all_polearms
                if center_in_box(box_center(box), il_box)
            ]

            if not humans_in:
                continue

            # ── Assign polearms to nearest human ─────────────────────────
            # For each polearm find nearest human
            human_polearm_map = {i: [] for i in range(len(humans_in))}

            for p_mask, p_score, p_box in polearms_in:
                nearest_hi   = None
                nearest_dist = float('inf')
                for hi, (h_mask, h_score, h_box) in enumerate(humans_in):
                    dist = box_distance(h_box, p_box)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_hi   = hi
                if nearest_hi is not None and nearest_dist <= MAX_ASSIGN_DISTANCE:
                    human_polearm_map[nearest_hi].append((p_mask, p_box))

            # ── Save each human with assigned polearms ────────────────────
            for hi, (h_mask, h_score, h_box) in enumerate(humans_in):
                assigned_polearms = human_polearm_map.get(hi, [])
                is_armed = len(assigned_polearms) > 0

                if is_armed:
                    combined_mask = h_mask.copy()
                    for p_mask, _ in assigned_polearms:
                        combined_mask = np.maximum(combined_mask, p_mask)

                    result = crop_rgba(img_array, combined_mask, padding)
                    if result is None:
                        continue
                    result.save(str(armed_dir / f"{stem}_il{il_idx+1}_hm{hi+1}_armed.png"))
                    stats["armed"] += 1
                else:
                    result = crop_rgba(img_array, h_mask, padding)
                    if result is None:
                        continue
                    result.save(str(unarmed_dir / f"{stem}_il{il_idx+1}_hm{hi+1}_unarmed.png"))
                    stats["unarmed"] += 1

    print(f"  armed={stats['armed']}  unarmed={stats['unarmed']}  "
          f"illustration={stats['illustration']}  skipped={stats['skipped']}")
    return stats


def process_all(predictions_root: str, image_root: str, output_root: str,
                padding: int, min_score: float, polearm_min_score: float,
                book: str = None):

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
    print(f"Armed Human Extraction (Illustration Context)")
    print(f"  Books              : {len(book_folders)}")
    print(f"  Padding            : {padding}px")
    print(f"  Min score (human)  : {min_score}")
    print(f"  Min score (polearm): {polearm_min_score}")
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
            pred_path         = pred_path,
            image_dir         = image_dir,
            output_root       = output_root,
            padding           = padding,
            min_score         = min_score,
            polearm_min_score = polearm_min_score,
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
    parser.add_argument("--predictions_root",   required=True)
    parser.add_argument("--image_root",         required=True)
    parser.add_argument("--output_root",        required=True)
    parser.add_argument("--book",               default=None)
    parser.add_argument("--padding",            type=int,   default=10)
    parser.add_argument("--min_score",          type=float, default=0.9)
    parser.add_argument("--polearm_min_score",  type=float, default=0.85)
    args = parser.parse_args()

    process_all(
        predictions_root  = args.predictions_root,
        image_root        = args.image_root,
        output_root       = args.output_root,
        padding           = args.padding,
        min_score         = args.min_score,
        polearm_min_score = args.polearm_min_score,
        book              = args.book,
    )
