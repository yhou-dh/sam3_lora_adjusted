# MILens — Local CLI Reference
Manuscript Illustration Lens

For HPC/SLURM usage see `HPC_CLI.md`.
This guide uses plain `python3` commands for local reproduction.

---

## 0. Every Session

> Always run commands from the project root folder:
> ```bash
> cd ~/Documents/GitHub/sam3_lora_adjusted
> ```
> All relative paths (e.g. `data/`, `predictions/`, `pairs/`) resolve from here.

```bash
cd ~/sam3_lora_adjusted   # or wherever you cloned the repo
export HF_TOKEN="your_token_here"
```

---

## 1. Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install "huggingface-hub>=1.5.0,<2.0"
pip install gradio opencv-python-headless Pillow pycocotools tqdm
pip install --no-deps -e .
```

---

## 2. Data Preparation *(supplementary)*

### Random Data Split
Only needed if you want to randomly shuffle and split raw images into train/val/test before annotation. If using Label Studio export directly, skip this step.

```bash
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
```

### Convert Roboflow results to COCO format
Only needed if your annotations were exported from Roboflow (individual JSON per image) and need to be merged into a single COCO JSON.

```bash
python3 convert_roboflow_to_coco.py \
    --data_root data \
    --splits train valid test
```

### Convert COCO/YOLO → SAM3 format
Only needed if your annotations are in COCO or YOLO format and need to be converted to the SAM3 per-image JSON format expected by the training pipeline.

```bash
# From COCO JSON
python3 prepare_data.py coco \
    --coco_json data/train/_annotations.coco.json \
    --images_dir data/train \
    --output_dir data/ \
    --split train

# Validate a split
python3 prepare_data.py validate \
    --data_dir data/ \
    --split train
```

### Binarization
Only needed if you want to preprocess images with Otsu binarization before training.

```bash
python3 binarize.py \
    --input_root data \
    --output_root data_binary
```

---

## 3. Training

```bash
python3 train_sam3_lora_native.py \
    --config configs/my_config-lite.yaml
```

---

## 4. Validation

### Per-class (illustration, human, polearm)
```bash
# LoRA on valid
python3 validate_sam3_perclass.py \
    --config configs/my_config-lite.yaml \
    --weights outputs/sam3_lora_lite/best_lora_weights.pt \
    --val_data_dir data/valid \
    --classes illustration human polearm

# LoRA on test
python3 validate_sam3_perclass.py \
    --config configs/my_config-lite.yaml \
    --weights outputs/sam3_lora_lite/best_lora_weights.pt \
    --val_data_dir data/test \
    --classes illustration human polearm

# Base model on valid
python3 validate_sam3_perclass.py \
    --val_data_dir data/valid \
    --use-base-model \
    --classes illustration human polearm

# Base model on test
python3 validate_sam3_perclass.py \
    --val_data_dir data/test \
    --use-base-model \
    --classes illustration human polearm
```

### Combined (all classes as one)
```bash
python3 validate_sam3_lora.py \
    --config configs/my_config-lite.yaml \
    --weights outputs/sam3_lora_lite/best_lora_weights.pt \
    --val_data_dir data/valid
```

---

## 5. Inference

### Single book folder (LoRA)
```bash
python3 infer.py \
    --input data/test \
    --mode single \
    --predictions_root predictions/lora \
    --config configs/my_config-lite.yaml \
    --weights outputs/sam3_lora_lite/best_lora_weights.pt \
    --prompts human illustration polearm \
    --masks
```

### Single book folder (Base model)
```bash
python3 infer.py \
    --input data/test \
    --mode single \
    --predictions_root predictions/base \
    --config configs/base_config.yaml \
    --weights base \
    --prompts human illustration polearm
```

### All immediate subfolders (batch)
```bash
python3 infer.py \
    --input data \
    --mode batch \
    --predictions_root predictions/lora \
    --config configs/my_config-lite.yaml \
    --weights outputs/sam3_lora_lite/best_lora_weights.pt \
    --prompts human illustration polearm \
    --masks \
    --skip_done
```

### All leaf folders recursively (nested)
```bash
python3 infer.py \
    --input finerbook \
    --mode nested \
    --predictions_root predictions/lora \
    --config configs/my_config-lite.yaml \
    --weights outputs/sam3_lora_lite/best_lora_weights.pt \
    --prompts human illustration polearm \
    --masks \
    --skip_done
```

Check progress:
```bash
find predictions/ -name "*.png" | wc -l
```

---

## 6. Post-Inference Evaluation

### Detection metrics (mAP, AR)
```bash
python3 evaluate_detections.py \
    --predictions predictions/lora/test/summaries/book_predictions.json \
    --annotations data/test/_annotations.coco.json \
    --model_name LoRA

python3 evaluate_detections.py \
    --predictions predictions/base/test/summaries/book_predictions_base.json \
    --annotations data/test/_annotations.coco.json \
    --model_name Base
```

### Threshold sweep (find optimal confidence threshold)
```bash
python3 threshold_sweep.py \
    --predictions predictions/lora/test/summaries/book_predictions.json \
    --annotations data/test/_annotations.coco.json \
    --model_name LoRA
```

---

## 7. Extraction

### By mask (transparent PNG, requires RLE masks from inference)
```bash
python3 extract_foreground.py \
    --predictions_root predictions/lora \
    --image_root data \
    --padding 10 \
    --min_score 0.8
```

### By bbox (JPG crops)
```bash
python3 extract_bbox.py \
    --predictions_root predictions/lora \
    --image_root data \
    --padding 10 \
    --min_score 0.8
```

### Armed human extraction (requires RLE masks)
```bash
python3 extract_armed.py \
    --predictions_root predictions/lora \
    --image_root data \
    --output_root predictions/armed \
    --padding 10 \
    --min_score 0.8 \
    --overlap_dilation 20
```

---


## 8. Post-Training Diagnostics

### Loss diagnostics
```bash
python3 analyze_loss.py
```

### LoRA vs Base comparison
```bash
python3 compare_lora_base_batch.py \
    --images data/test/img1.jpg data/test/img2.jpg \
    --data-dir data/test \
    --config configs/my_config-lite.yaml \
    --weights outputs/sam3_lora_lite/best_lora_weights.pt \
    --output outputs/comparison/comparison.png
```

---

## 9. Image-Text Data Pairing *(supplementary)*

Pair segmented illustration crops with text and metadata from CSV files to create image-text datasets.
Outputs image-text pairs and enriched pairs with all CSV fields.

```bash
# Mask crops, combined output
python3 pair_data.py \
    --csv_dir pairs/csv \
    --predictions_root predictions/lora \
    --extraction_type mask \
    --output_dir pairs/output \
    --mode combined

# Bbox crops, per-book output
python3 pair_data.py \
    --csv_dir pairs/csv \
    --predictions_root predictions/lora \
    --extraction_type bbox \
    --output_dir pairs/output \
    --mode per_book

# Both combined and per-book
python3 pair_data.py \
    --csv_dir pairs/csv \
    --predictions_root predictions/lora \
    --extraction_type mask \
    --output_dir pairs/output \
    --mode both
```

Outputs per run:
- `image_text_pairs.json/.csv` — `{image_path, text}`
- `image_text_enriched.json/.csv` — `{image_path, text, count_person, count_weapon, verified, ...}`

---


---

## 10. Pair Gallery Visualisation

Generate a self-contained HTML gallery from image-text pairs, switchable between simple (image + text) and enriched (image + text + all metadata) views.

```bash
python3 visualise_pairs.py \
    --pairs pairs/output/image_text_pairs.json \
    --enriched pairs/output/image_text_enriched.json \
    --image_base predictions/lora \
    --output pairs/output/gallery.html
```

Then open in browser:
```bash
open pairs/output/gallery.html   # Mac
```

---

## 11. Web App

```bash
pip install gradio
python3 app.py
# Open http://localhost:7860
```

---

## Key Rules

| Rule | Detail |
|---|---|
| Python version | `python3` (3.9+) |
| Masks for extraction | Always run inference with `--masks` if you plan to extract |
| Single book | `infer.py --book <name>` |
| All books | `infer.py --book_root <folder>` |
| Base model | Pass `--weights base` to any infer/validate script |
| Logs | All app logs saved to `logs/` with timestamps |
