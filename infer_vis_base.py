import os
import shutil 
import sys
from pathlib import Path
import json
import numpy as np

bookname = "test"

BASE_DIR = Path('/home/yumeng.hou/sam3_lora_adjusted')  # ADD HERE
image_dir = BASE_DIR / f"data/{bookname}/images"
base_all_output_dir = BASE_DIR / f"predictions/base/{bookname}"


if '/home/yumeng.hou/sam3_lora_adjusted' not in sys.path:
    sys.path.insert(0, '/home/yumeng.hou/sam3_lora_adjusted')

from infer_sam import SAM3LoRAInference

# Function to save book predictions summary
def save_book_predictions_summary(inferencer_obj, bookname, all_image_paths, output_root_dir):
    summary_output_dir = output_root_dir / bookname/ "summaries"
    summary_output_dir.mkdir(parents=True, exist_ok=True)

    book_predictions_data = []
    print(f"\nExtracting and saving predictions for book: {bookname}...")

    # Define prompts for the summary. Using the multi-prompt set from the cell.
    current_prompts_for_summary = ["human", "illustration", "polearm"]

    for img_path in all_image_paths:
        image_data = {
            "file_name": img_path.name,
            "detections": []
        }

        predictions_for_summary = inferencer_obj.predict(
            str(img_path),
            text_prompts=current_prompts_for_summary
        )

        if isinstance(predictions_for_summary, dict):
            for prompt, result_dict in predictions_for_summary.items():
                if isinstance(result_dict, dict) and 'boxes' in result_dict and 'scores' in result_dict:
                    # Convert numpy arrays to lists for JSON serialization
                    boxes_list = result_dict['boxes'].tolist() if result_dict['boxes'] is not None else []
                    scores_list = result_dict['scores'].tolist() if result_dict['scores'] is not None else []

                    image_data["detections"].append({
                        "prompt": prompt,
                        "boxes": boxes_list,
                        "scores": scores_list,
                        "num_detections": result_dict.get('num_detections', len(boxes_list)) if result_dict['boxes'] is not None else 0
                    })
                else:
                    print(f"  Warning: Result for prompt '{prompt}' in {img_path.name} is not a valid dictionary or missing keys. Skipping for summary.")
        else:
            print(f"  Warning: Predictions for {img_path.name} was not a dictionary. Skipping summary for this image.")

        book_predictions_data.append(image_data)

    # Save the compiled data to a JSON file
    output_summary_path = summary_output_dir / "book_predictions_base.json"
    with open(output_summary_path, 'w') as f:
        json.dump(book_predictions_data, f, indent=4)

    print(f"All book predictions (excluding masks) saved to {output_summary_path}")



# Save current working directory

original_cwd_base_full = os.getcwd()

try:
    os.chdir('/home/yumeng.hou/sam3_lora_adjusted')

    base_inferencer_full = SAM3LoRAInference(
        config_path="configs/base_config.yaml",
        weights_path='base',
        detection_threshold=0.8,
        nms_iou_threshold=0.15
    )
finally:
    os.chdir(original_cwd_base_full)

image_dir = Path(f"data/{bookname}/images")

# Clean up existing output directory if it exists
if base_all_output_dir.exists():
    print(f"Removing existing output directory: {base_all_output_dir}")
    shutil.rmtree(base_all_output_dir)

# Create all necessary directories
base_all_output_dir.mkdir(parents=True, exist_ok=True)

image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp"]
all_image_paths = []
for ext in image_extensions:
    all_image_paths.extend(image_dir.glob(ext))


# Define output directory for base model predictions for all images
base_all_output_dir = Path(f"predictions/base/{bookname}")

print(f"\nRunning base model inference on all {len(all_image_paths)} images...")

# Process all images with multiple prompts using the base inferencer
for img_path in all_image_paths:
    # Detect multiple object types at once
    predictions = base_inferencer_full.predict(
        str(img_path),
        text_prompts=["human", "illustration", "polearm"]
    )

    output_path = base_all_output_dir / f"{img_path.stem}_multi.png"
    base_inferencer_full.visualize(predictions, str(output_path))

    # Print summary
    if isinstance(predictions, dict):
        for result in predictions.values():
            if isinstance(result, dict) and 'prompt' in result and 'num_detections' in result:
                print(f"  {result['prompt']}: {result['num_detections']} detections")
    else:
        print(f"  Warning: Predictions for {img_path.name} was not a dictionary. Cannot print detailed summary.")

# Call the new function to save the book predictions summary
save_book_predictions_summary(base_inferencer_full, bookname, all_image_paths, BASE_DIR / "predictions/base")