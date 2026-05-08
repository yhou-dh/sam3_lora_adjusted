import os
import shutil
import sys
from pathlib import Path
import json
import numpy as np
from pycocotools import mask as mask_utils

BASE_DIR = Path('/home/yumeng.hou/sam3_lora_adjusted')
BOOK_DIR = BASE_DIR / "finerbook"
bookname = "bdj_qm"  # test on one book first

if '/home/yumeng.hou/sam3_lora_adjusted' not in sys.path:
    sys.path.insert(0, '/home/yumeng.hou/sam3_lora_adjusted')

from infer_sam import SAM3LoRAInference


def encode_mask(mask_array):
    """Encode a boolean/float [H, W] mask to compact RLE format."""
    mask_uint8 = np.asfortranarray(mask_array.astype(np.uint8))
    rle = mask_utils.encode(mask_uint8)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def save_book_predictions_summary(inferencer_obj, bookname, all_image_paths, output_root_dir):
    summary_output_dir = output_root_dir / bookname / "summaries"
    summary_output_dir.mkdir(parents=True, exist_ok=True)

    book_predictions_data = []
    print(f"\nExtracting and saving predictions for book: {bookname}...")

    prompts = ["human", "illustration", "polearm", "sword", "crossbow", "formation"]

    for img_path in all_image_paths:
        image_data = {"file_name": img_path.name, "detections": []}

        predictions = inferencer_obj.predict(
            str(img_path),
            text_prompts=prompts
        )

        if isinstance(predictions, dict):
            for prompt, result_dict in predictions.items():
                if not (isinstance(result_dict, dict)
                        and 'boxes' in result_dict
                        and 'scores' in result_dict):
                    print(f"  Warning: Result for prompt '{prompt}' in "
                          f"{img_path.name} missing keys. Skipping.")
                    continue

                boxes_list  = result_dict['boxes'].tolist()  if result_dict['boxes']  is not None else []
                scores_list = result_dict['scores'].tolist() if result_dict['scores'] is not None else []

                # ── RLE mask encoding ────────────────────────────────────
                masks_rle = []
                raw_masks = result_dict.get('masks')
                if raw_masks is not None:
                    print(f"  Mask shape: {raw_masks.shape}, dtype: {raw_masks.dtype}")
                    for i in range(raw_masks.shape[0]):
                        masks_rle.append(encode_mask(raw_masks[i]))

                image_data["detections"].append({
                    "prompt":         prompt,
                    "boxes":          boxes_list,
                    "scores":         scores_list,
                    "num_detections": result_dict.get('num_detections', len(boxes_list)),
                    "masks_rle":      masks_rle,
                })
        else:
            print(f"  Warning: Predictions for {img_path.name} was not a dict.")

        book_predictions_data.append(image_data)

    output_summary_path = summary_output_dir / "book_predictions.json"
    with open(output_summary_path, 'w') as f:
        json.dump(book_predictions_data, f, indent=4)
    print(f"Predictions saved to {output_summary_path}")
    print(f"File size: {output_summary_path.stat().st_size / 1024 / 1024:.2f} MB")


# ── Initialize inferencer once ───────────────────────────────────────────────
original_cwd = os.getcwd()
try:
    os.chdir('/home/yumeng.hou/sam3_lora_adjusted')
    inferencer = SAM3LoRAInference(
        config_path="configs/my_config-lite.yaml",
        weights_path="outputs/sam3_lora_lite/best_lora_weights.pt",
        detection_threshold=0.8,
        nms_iou_threshold=0.15
    )
finally:
    os.chdir(original_cwd)

# ── Process single book ───────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Processing book: {bookname}")
print(f"{'='*50}")

image_dir  = BOOK_DIR / bookname
output_dir = BASE_DIR / "predictions" / "lora_masks" / bookname

if not image_dir.exists():
    print(f"  ⚠ Folder not found: {image_dir}")
    exit(1)

# Collect images
image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp"]
all_image_paths = []
for ext in image_extensions:
    all_image_paths.extend(image_dir.glob(ext))

if not all_image_paths:
    print(f"  ⚠ No images found in {image_dir}")
    exit(1)

print(f"  Found {len(all_image_paths)} images")

# Clean and create output dir
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Run inference + save visualisations
for img_path in all_image_paths:
    predictions = inferencer.predict(
        str(img_path),
        text_prompts=["human", "illustration", "polearm", "sword", "crossbow", "formation"]
    )

    output_path = output_dir / f"{img_path.stem}_multi.png"
    inferencer.visualize(predictions, str(output_path))

    if isinstance(predictions, dict):
        for result in predictions.values():
            if isinstance(result, dict) and 'prompt' in result and 'num_detections' in result:
                print(f"  {result['prompt']}: {result['num_detections']} detections")

# Save summary JSON with RLE masks
save_book_predictions_summary(
    inferencer, bookname, all_image_paths,
    BASE_DIR / "predictions" / "lora_masks"
)

print("\n✅ Done!")