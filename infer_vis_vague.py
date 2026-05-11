import os
import shutil 
import sys
import argparse
from pathlib import Path
import json
import numpy as np

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--max_images', type=int, default=None, help='Limit number of images to process')
args = parser.parse_args()

bookname = "test"

BASE_DIR = Path('/home/yumeng.hou/sam3_lora_adjusted')
image_dir = BASE_DIR / f"data/{bookname}/images"
output_dir = BASE_DIR / f"predictions/lora/{bookname}"

if '/home/yumeng.hou/sam3_lora_adjusted' not in sys.path:
    sys.path.insert(0, '/home/yumeng.hou/sam3_lora_adjusted')

from infer_sam import SAM3LoRAInference


def save_book_predictions_summary(inferencer_obj, bookname, all_image_paths, output_root_dir):
    summary_output_dir = output_root_dir / bookname / "summaries"
    summary_output_dir.mkdir(parents=True, exist_ok=True)

    book_predictions_data = []
    print(f"\nExtracting and saving predictions for book: {bookname}...")

    current_prompts_for_summary = ["human", "illustration", "polearm or sword", "vertical Chinese"]

    for img_path in all_image_paths:
        image_data = {"file_name": img_path.name, "detections": []}

        predictions_for_summary = inferencer_obj.predict(
            str(img_path),
            text_prompts=current_prompts_for_summary
        )

        if isinstance(predictions_for_summary, dict):
            for prompt, result_dict in predictions_for_summary.items():
                if isinstance(result_dict, dict) and 'boxes' in result_dict and 'scores' in result_dict:
                    boxes_list  = result_dict['boxes'].tolist() if result_dict['boxes'] is not None else []
                    scores_list = result_dict['scores'].tolist() if result_dict['scores'] is not None else []
                    image_data["detections"].append({
                        "prompt": prompt,
                        "boxes":  boxes_list,
                        "scores": scores_list,
                        "num_detections": result_dict.get('num_detections', len(boxes_list)) if result_dict['boxes'] is not None else 0
                    })
                else:
                    print(f"  Warning: Result for prompt '{prompt}' in {img_path.name} missing keys.")
        else:
            print(f"  Warning: Predictions for {img_path.name} was not a dictionary.")

        book_predictions_data.append(image_data)

    output_summary_path = summary_output_dir / "book_predictions.json"
    with open(output_summary_path, 'w') as f:
        json.dump(book_predictions_data, f, indent=4)
    print(f"All book predictions saved to {output_summary_path}")


# ── Initialize inferencer ─────────────────────────────────────────────────────
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

# ── Clean output dir ──────────────────────────────────────────────────────────
if output_dir.exists():
    print(f"Removing existing output directory: {output_dir}")
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# ── Collect images ────────────────────────────────────────────────────────────
image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp"]
all_image_paths = []
for ext in image_extensions:
    all_image_paths.extend(image_dir.glob(ext))

if args.max_images:
    all_image_paths = all_image_paths[:args.max_images]
    print(f"Limiting to first {args.max_images} images")

print(f"Processing {len(all_image_paths)} images...")

# ── Run inference ─────────────────────────────────────────────────────────────
for img_path in all_image_paths:
    predictions = inferencer.predict(
        str(img_path),
        text_prompts=["human", "illustration", "polearm or sword", "vertical Chinese"]
    )

    output_path = output_dir / f"{img_path.stem}_multi.png"
    inferencer.visualize(predictions, str(output_path))

    if isinstance(predictions, dict):
        for result in predictions.values():
            if isinstance(result, dict) and 'prompt' in result and 'num_detections' in result:
                print(f"  {result['prompt']}: {result['num_detections']} detections")
    else:
        print(f"  Warning: Predictions for {img_path.name} is not a dictionary.")

save_book_predictions_summary(inferencer, bookname, all_image_paths, BASE_DIR / "predictions/lora")
