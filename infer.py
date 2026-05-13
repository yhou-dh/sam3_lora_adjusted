"""
infer.py
Unified SAM3 LoRA inference script.

Replaces: infer_vis.py, infer_vmask.py

Modes:
  single  — input is a single book folder (all images inside processed directly)
  batch   — input is a parent folder; each immediate subfolder is a book (expects images/ inside each)
  nested  — input is a root; any leaf subfolder containing images is a book

Outputs per book:
  - Visualisation PNGs  → <predictions_root>/<book_name>/
  - book_predictions.json → <predictions_root>/<book_name>/summaries/
    (includes RLE masks if --masks is set)

Usage:
    # Single book folder
    python3 infer.py \
        --input data/test/images \
        --mode single \
        --predictions_root predictions/lora \
        --config configs/my_config-lite.yaml \
        --weights outputs/sam3_lora_lite/best_lora_weights.pt \
        --prompts human illustration polearm \
        --masks

    # All immediate subfolders of data/
    python3 infer.py \
        --input data \
        --mode batch \
        --predictions_root predictions/lora \
        --config configs/my_config-lite.yaml \
        --weights outputs/sam3_lora_lite/best_lora_weights.pt \
        --prompts human illustration polearm \
        --masks \
        --skip_done

    # Recursively find all leaf folders with images
    python3 infer.py \
        --input finerbook \
        --mode nested \
        --predictions_root predictions/lora \
        --config configs/my_config-lite.yaml \
        --weights outputs/sam3_lora_lite/best_lora_weights.pt \
        --prompts human illustration polearm \
        --masks \
        --skip_done

    # Base model
    python3 infer.py \
        --input data/test/images \
        --mode single \
        --predictions_root predictions/base \
        --config configs/base_config.yaml \
        --weights base \
        --prompts human illustration polearm
"""

import os
import shutil
import sys
import json
import argparse
import numpy as np
from pathlib import Path


IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp"]


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 LoRA Inference")
    parser.add_argument("--input",               required=True,
                        help="Input folder. For 'single': the book folder itself. "
                             "For 'batch': parent with one subfolder per book. "
                             "For 'nested': root with arbitrary depth; leaf folders containing images are books.")
    parser.add_argument("--mode",                choices=["single", "batch", "nested"],
                        default="batch",
                        help="Traversal mode (default: batch)")
    parser.add_argument("--predictions_root",    default="predictions/lora",
                        help="Root folder for output predictions (default: predictions/lora)")
    parser.add_argument("--base_dir",            default=str(Path.home() / "sam3_lora_adjusted"),
                        help="Repo root for resolving relative config/weights paths")
    parser.add_argument("--config",              default="configs/my_config-lite.yaml")
    parser.add_argument("--weights",             default="outputs/sam3_lora_lite/best_lora_weights.pt",
                        help="Weights path, or 'base' for base model")
    parser.add_argument("--prompts",             nargs="+",
                        default=["human", "illustration", "polearm"])
    parser.add_argument("--detection_threshold", type=float, default=0.8)
    parser.add_argument("--nms_iou_threshold",   type=float, default=0.15)
    parser.add_argument("--masks",               action="store_true",
                        help="Save RLE masks in book_predictions.json")
    parser.add_argument("--skip_done",           action="store_true",
                        help="Skip books that already have book_predictions.json")
    return parser.parse_args()


# ── Folder discovery ─────────────────────────────────────────────────────────

def collect_images(folder: Path) -> list:
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(folder.glob(ext))
    return sorted(paths)


def discover_books(input_path: Path, mode: str) -> list:
    """
    Returns list of (book_name, image_dir) tuples.
    book_name is used as the subfolder name under predictions_root.
    """
    input_path = input_path.resolve()

    if mode == "single":
        imgs = collect_images(input_path)
        if not imgs:
            print(f"⚠ No images found in {input_path}")
            return []
        return [(input_path.name, input_path)]

    elif mode == "batch":
        books = []
        for d in sorted(input_path.iterdir()):
            if d.is_dir():
                imgs = collect_images(d)
                if imgs:
                    books.append((d.name, d))
                else:
                    print(f"  ⚠ No images in {d.name}, skipping.")
        return books

    elif mode == "nested":
        # Leaf folders: directories that contain images (not just subdirs)
        books = []
        for d in sorted(input_path.rglob("*")):
            if d.is_dir():
                imgs = collect_images(d)
                if imgs:
                    # Use path relative to input as book name (joining with __)
                    rel = d.relative_to(input_path)
                    book_name = "__".join(rel.parts) if rel.parts else d.name
                    books.append((book_name, d))
        return books

    return []


# ── Mask encoding ─────────────────────────────────────────────────────────────

def encode_mask(mask_array):
    from pycocotools import mask as mask_utils
    mask_uint8 = np.asfortranarray(mask_array.astype(np.uint8))
    rle = mask_utils.encode(mask_uint8)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


# ── Per-book inference ────────────────────────────────────────────────────────

def process_book(inferencer, book_name: str, image_dir: Path,
                 predictions_root: Path, prompts: list,
                 save_masks: bool, skip_done: bool):

    summary_path = predictions_root / book_name / "summaries" / "book_predictions.json"

    if skip_done and summary_path.exists():
        print(f"  ✓ Skipping {book_name} (already done)")
        return

    output_dir = predictions_root / book_name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_image_paths = collect_images(image_dir)
    print(f"\n{'='*50}")
    print(f"Book : {book_name}")
    print(f"Images: {len(all_image_paths)}")
    print(f"Masks : {save_masks}")
    print(f"{'='*50}")

    book_predictions_data = []

    for img_path in all_image_paths:
        # Visualisation
        predictions = inferencer.predict(str(img_path), text_prompts=prompts)
        vis_path = output_dir / f"{img_path.stem}_multi.png"
        inferencer.visualize(predictions, str(vis_path))

        if isinstance(predictions, dict):
            for result in predictions.values():
                if isinstance(result, dict) and 'prompt' in result:
                    print(f"  {result['prompt']}: {result.get('num_detections', 0)} detections")

        # Build prediction record
        image_data = {"file_name": img_path.name, "detections": []}

        if isinstance(predictions, dict):
            for prompt, result_dict in predictions.items():
                if not (isinstance(result_dict, dict)
                        and 'boxes' in result_dict
                        and 'scores' in result_dict):
                    continue

                boxes_list  = result_dict['boxes'].tolist()  if result_dict['boxes']  is not None else []
                scores_list = result_dict['scores'].tolist() if result_dict['scores'] is not None else []

                det = {
                    "prompt":         prompt,
                    "boxes":          boxes_list,
                    "scores":         scores_list,
                    "num_detections": result_dict.get('num_detections', len(boxes_list)),
                }

                if save_masks:
                    masks_rle = []
                    raw_masks = result_dict.get('masks')
                    if raw_masks is not None:
                        for i in range(raw_masks.shape[0]):
                            masks_rle.append(encode_mask(raw_masks[i]))
                    det["masks_rle"] = masks_rle

                image_data["detections"].append(det)

        book_predictions_data.append(image_data)

    # Save summary JSON
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(book_predictions_data, f, indent=4)

    size_mb = summary_path.stat().st_size / 1024 / 1024
    print(f"\n✅ Saved {summary_path} ({size_mb:.2f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    base_dir   = Path(args.base_dir).expanduser()
    input_path = Path(args.input).expanduser()
    pred_root  = Path(args.predictions_root)

    # Make predictions_root absolute if relative
    if not pred_root.is_absolute():
        pred_root = base_dir / pred_root

    # Add repo to sys.path
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    from infer_sam import SAM3LoRAInference

    # Discover books
    books = discover_books(input_path, args.mode)
    if not books:
        print("❌ No books found. Check --input and --mode.")
        return

    print(f"\nFound {len(books)} book(s) to process.")
    print(f"Mode         : {args.mode}")
    print(f"Masks        : {args.masks}")
    print(f"Prompts      : {args.prompts}")
    print(f"Weights      : {args.weights}")
    print(f"Output root  : {pred_root}")

    # Initialize inferencer once
    original_cwd = os.getcwd()
    try:
        os.chdir(str(base_dir))
        inferencer = SAM3LoRAInference(
            config_path=args.config,
            weights_path=args.weights,
            detection_threshold=args.detection_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
        )
    finally:
        os.chdir(original_cwd)

    # Process each book
    for book_name, image_dir in books:
        process_book(
            inferencer     = inferencer,
            book_name      = book_name,
            image_dir      = image_dir,
            predictions_root = pred_root,
            prompts        = args.prompts,
            save_masks     = args.masks,
            skip_done      = args.skip_done,
        )

    print("\n✅ All done!")


if __name__ == "__main__":
    main()
