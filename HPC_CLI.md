# MILens — HPC Command Reference
Manuscript Illustration Lens

---

## Monitor

```bash
squeue -u yumeng.hou              # check running jobs
squeue --start -u yumeng.hou      # check estimated start time
scancel <jobid>                   # cancel a job
tail -f <logfile>.out             # watch live log
cat <logfile>.out                 # read full log
du -sh ~/                         # check storage
git pull origin main              # update code
```

---

## 0. Every Login

```bash
cd ~/sam3_lora_adjusted
export PATH="/home/yumeng.hou/.local/bin:$PATH"
export HF_TOKEN="hf_token"
```

---

## 1. Clone Repo *(first time only)*

```bash
cd ~
git clone https://github.com/YHou-DH/sam3_lora_adjusted.git
cd sam3_lora_adjusted
```

---

## 1.5 Data Preparation *(as needed)*

### Random Data Split *(supplementary)*
Only needed if you want to randomly shuffle and split raw images before annotation.

```bash
python3 split_data.py \
    --source data/all_images \
    --output data \
    --train 0.8 --valid 0.1 --test 0.1 \
    --seed 42
```

### Convert Roboflow results to COCO format
```bash
python3 convert_roboflow_to_coco.py \
    --data_root data \
    --splits train valid test
```

### Convert COCO or YOLO annotations to SAM3 format
```bash
# From COCO JSON
python3 prepare_data.py coco \
    --coco_json data/train/_annotations.coco.json \
    --images_dir data/train \
    --output_dir data/ \
    --split train

# From YOLO
python3 prepare_data.py yolo \
    --yolo_dir data/ \
    --output_dir data/ \
    --classes human illustration polearm \
    --split train

# Validate a split
python3 prepare_data.py validate \
    --data_dir data/ \
    --split train
```

> Note: Train/valid/test splitting into COCO format is handled separately —
> use Label Studio export directly or the data split scripts in the project root.

---

## 2. Install Dependencies *(first time only)*

```bash
cat > fix_torch.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=fix_torch
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=fix_torch_%j.out

python3.10 -m pip uninstall -y torch torchvision torchaudio
python3.10 -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python3.10 -m pip install --user "huggingface-hub>=1.5.0,<2.0"
python3.10 -m pip install --user --no-deps -e ~/sam3_lora_adjusted/

python3.10 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
EOF

sbatch fix_torch.sh
```

---

## 3. Update Code

```bash
cd ~/sam3_lora_adjusted
git pull origin main
```

---

## 4. Training

```bash
cat > train.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=sam3_train
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=train_%j.out

export HF_TOKEN="hf_token"
cd ~/sam3_lora_adjusted
python3.10 train_sam3_lora_native.py --config configs/my_config-lite.yaml
EOF

sbatch train.sh
```

---

## 5. Validation

### 5.1 Validation per class


#5.1 Validation per class

```bash
# LoRA on valid
cat > validate_lora_valid.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=sam3_validate
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=validate_%j.out
export HF_TOKEN="yuor token here"
cd ~/sam3_lora_adjusted
/usr/bin/python3 validate_sam3_perclass.py \
  --config configs/my_config-lite.yaml \
  --weights outputs/sam3_lora_lite/best_lora_weights.pt \
  --val_data_dir data/valid \
  --classes illustration human polearm
EOF
# LoRA on test
cat > validate_lora_test.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=sam3_test
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=validate_test_%j.out
export HF_TOKEN="yuor token here"
cd ~/sam3_lora_adjusted
/usr/bin/python3 validate_sam3_perclass.py \
  --config configs/my_config-lite.yaml \
  --weights outputs/sam3_lora_lite/best_lora_weights.pt \
  --val_data_dir data/test \
  --classes illustration human polearm
EOF
# Base on valid
cat > validate_base_valid.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=base_valid
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=validate_base_valid_%j.out
export HF_TOKEN="yuor token here"
cd ~/sam3_lora_adjusted
/usr/bin/python3 validate_sam3_perclass.py \
  --val_data_dir data/valid \
  --use-base-model \
  --classes illustration human polearm
EOF
# Base on test
cat > validate_base_test.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=base_test
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=validate_base_test_%j.out
export HF_TOKEN="yuor token here"
cd ~/sam3_lora_adjusted
/usr/bin/python3 validate_sam3_perclass.py \
  --val_data_dir data/test \
  --use-base-model \
  --classes illustration human polearm
EOF
sbatch validate_lora_valid.sh
sbatch validate_lora_test.sh
sbatch validate_base_valid.sh
sbatch validate_base_test.sh

```


#5.2 Validation (by looking at selected classess without differenting them)

```bash
# LoRA on valid
cat > validate.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=sam3_validate
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=validate_%j.out

export HF_TOKEN="hf_token"
cd ~/sam3_lora_adjusted
python3.10 validate_sam3_lora.py \
  --config configs/my_config-lite.yaml \
  --weights outputs/sam3_lora_lite/best_lora_weights.pt \
  --val_data_dir data/valid
EOF

# LoRA on test
cat > validate_test.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=sam3_test
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=validate_test_%j.out

export HF_TOKEN="hf_token"
cd ~/sam3_lora_adjusted
python3.10 validate_sam3_lora.py \
  --config configs/my_config-lite.yaml \
  --weights outputs/sam3_lora_lite/best_lora_weights.pt \
  --val_data_dir data/test
EOF

# Base on valid
cat > validate_base_valid.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=base_valid
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=validate_base_valid_%j.out

export HF_TOKEN="hf_token"
cd ~/sam3_lora_adjusted
python3.10 validate_sam3_lora.py \
  --val_data_dir data/valid \
  --use-base-model
EOF

# Base on test
cat > validate_base_test.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=base_test
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=validate_base_test_%j.out

export HF_TOKEN="hf_token"
cd ~/sam3_lora_adjusted
python3.10 validate_sam3_lora.py \
  --val_data_dir data/test \
  --use-base-model
EOF

sbatch validate.sh
sbatch validate_test.sh
sbatch validate_base_valid.sh
sbatch validate_base_test.sh
```

---

## 6. Inference

> `infer.py` is the unified inference script replacing `infer_vis.py`, `infer_vis_base.py`, `infer_vis2.py`, and `infer_vmask.py`.
> Use `--mode` to control folder traversal and `--masks` to save RLE masks.

### 6.1 Single book (LoRA or Base)

```bash
# LoRA inference
cat > infer_lora.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=infer_lora
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=infer_lora_%j.out

export HF_TOKEN="hf_token"
cd ~/sam3_lora_adjusted
python3.10 infer.py \
    --input data/test \
    --mode single \
    --predictions_root predictions/lora \
    --config configs/my_config-lite.yaml \
    --weights outputs/sam3_lora_lite/best_lora_weights.pt \
    --prompts human illustration polearm \
    --masks
EOF

# Base inference
cat > infer_base.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=infer_base
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=infer_base_%j.out

export HF_TOKEN="hf_token"
cd ~/sam3_lora_adjusted
python3.10 infer.py \
    --input data/test \
    --mode single \
    --predictions_root predictions/base \
    --config configs/base_config.yaml \
    --weights base \
    --prompts human illustration polearm \
    --masks
EOF

sbatch infer_lora.sh
sbatch infer_base.sh
```

Check progress:
```bash
find ~/sam3_lora_adjusted/predictions/ -name "*.png" | wc -l
watch -n 10 "find ~/sam3_lora_adjusted/predictions/ -name '*.png' | wc -l"
```

### 6.2 All immediate subfolders (batch, boxes only)

```bash
cat > infer_lora2.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=infer_lora2
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=infer_lora2_%j.out

export HF_TOKEN="hf_token"
cd ~/sam3_lora_adjusted
python3.10 infer.py \
    --input data \
    --mode batch \
    --predictions_root predictions/lora \
    --config configs/my_config-lite.yaml \
    --weights outputs/sam3_lora_lite/best_lora_weights.pt \
    --prompts human illustration polearm \
    --skip_done
EOF

sbatch infer_lora2.sh
```

### 6.3 All books with RLE masks (needed for segmentation/pairing)

```bash
cat > infer_masks.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=infer_masks
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=infer_masks_%j.out

export HF_TOKEN="hf_token"
cd ~/sam3_lora_adjusted
python3.10 infer.py \
    --input finerbook \
    --mode nested \
    --predictions_root predictions/lora \
    --config configs/my_config-lite.yaml \
    --weights outputs/sam3_lora_lite/best_lora_weights.pt \
    --prompts human illustration polearm \
    --masks \
    --skip_done
EOF

sbatch infer_masks.sh
```

## 7 Segmentation

### 7.1 Segment by mask

```bash
cat > extract_fg.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=extract_fg
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=extract_fg_%j.out

cd ~/sam3_lora_adjusted

python3.10 extract_foreground.py \
    --predictions_root predictions/lora \
    --image_root data \
    --padding 10 \
    --min_score 0.9
EOF

sbatch extract_fg.sh
```

### 7.2 Segment by bbox

```bash
cat > extract_bbox.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=extract_bbox
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=extract_bbox_%j.out

cd ~/sam3_lora_adjusted

python3.10 extract_bbox.py \
    --predictions_root predictions/lora \
    --image_root data \
    --padding 10 \
    --min_score 0.9
EOF

sbatch extract_bbox.sh
```

### 7.3 Armed human extraction

Extracts armed vs unarmed humans by detecting overlap between human and polearm masks.

```bash
cat > extract_armed.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=extract_armed
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=extract_armed_%j.out

cd ~/sam3_lora_adjusted

python3.10 extract_armed.py \
    --predictions_root predictions/lora \
    --image_root data \
    --output_root predictions/armed \
    --padding 10 \
    --min_score 0.8 \
    --overlap_dilation 20
EOF

sbatch extract_armed.sh
```

Outputs go to `predictions/armed/<book>/human_armed/`, `human_unarmed/`, `illustration_bbox/`.

### 7.4 Binarization

Applies Otsu binarization to all images in nested subfolders, saving to a mirrored output directory.

```bash
cat > binarize.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=binarize
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=binarize_%j.out

cd ~/sam3_lora_adjusted
python3 binarize.py \
    --input_root data \
    --output_root data_binary
EOF

sbatch binarize.sh
```

## 8. Upload Results to HuggingFace

```bash
cd ~/sam3_lora_adjusted

# Verify JSONs exist first
find ~/sam3_lora_adjusted -name "book_predictions.json"
find ~/sam3_lora_adjusted -name "book_predictions_base.json"

# Zip and upload predictions
zip -r predictions.zip predictions/
zip -r validation_logs.zip validate_*.out
zip -r outputs.zip outputs/

python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='hf_token')
for f, name in [
    ('predictions.zip',     'predictions.zip'),
    ('validation_logs.zip', 'validation_logs.zip'),
    ('outputs.zip',         'outputs.zip'),
]:
    api.upload_file(
        path_or_fileobj='/home/yumeng.hou/sam3_lora_adjusted/' + f,
        path_in_repo=name,
        repo_id='yhlela/lora_ma',
        repo_type='dataset'
    )
    print(f'Uploaded {f}')
"

# Upload specific JSON files only
python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='hf_token')
api.upload_file(
    path_or_fileobj='/home/yumeng.hou/sam3_lora_adjusted/predictions/lora/test/summaries/book_predictions.json',
    path_in_repo='predictions/lora/book_predictions.json',
    repo_id='yhlela/lora_ma',
    repo_type='dataset'
)
api.upload_file(
    path_or_fileobj='/home/yumeng.hou/sam3_lora_adjusted/predictions/base/test/summaries/book_predictions_base.json',
    path_in_repo='predictions/base/book_predictions_base.json',
    repo_id='yhlela/lora_ma',
    repo_type='dataset'
)
print('Done!')
"
```

---

## 9. Post-Evaluation

```bash
cat > eval.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=eval_%j.out

cd ~/sam3_lora_adjusted

python3.10 evaluate_detections.py \
    --predictions predictions/lora/test/summaries/book_predictions.json \
    --annotations data/test/_annotations.coco.json \
    --model_name LoRA

python3.10 evaluate_detections.py \
    --predictions predictions/base/test/summaries/book_predictions_base.json \
    --annotations data/test/_annotations.coco.json \
    --model_name Base
EOF

sbatch eval.sh
```

---

## 10. Threshold Sweep

```bash
cat > sweep.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=threshold_sweep
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=sweep_%j.out

cd ~/sam3_lora_adjusted

python3.10 threshold_sweep.py \
    --predictions predictions/lora/test/summaries/book_predictions.json \
    --annotations data/test/_annotations.coco.json \
    --model_name LoRA

python3.10 threshold_sweep.py \
    --predictions predictions/base/test/summaries/book_predictions_base.json \
    --annotations data/test/_annotations.coco.json \
    --model_name Base
EOF

sbatch sweep.sh
```

---

## 11. Post-Training Diagnostics

### Loss diagnostics
```bash
# Standalone diagnostic script — prints loss weight analysis to stdout
python3 analyze_loss.py
```

### Compare LoRA vs Base (batch)
```bash
python3 compare_lora_base_batch.py \
    --lora_predictions predictions/lora/test/summaries/book_predictions.json \
    --base_predictions predictions/base/test/summaries/book_predictions_base.json \
    --annotations data/test/_annotations.coco.json \
    --output_dir outputs/comparison/
```

---

## 12. Image-Text Data Pairing *(supplementary)*

Pair segmented illustration crops with text and metadata from CSV files.

```bash
cat > pair_data.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=pair_data
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=pair_data_%j.out

cd ~/sam3_lora_adjusted
python3 pair_data.py \
    --csv_dir pairs/csv \
    --predictions_root predictions/lora \
    --extraction_type mask \
    --output_dir pairs/output \
    --mode both
EOF

sbatch pair_data.sh
```

---

## Key Rules

| Rule | Detail |
|---|---|
| Never run heavy commands on login node | Always use `sbatch` |
| Always request GPU | `--gres=gpu:1` |
| Python for torch | `python3.10` |
| Python for HuggingFace uploads | `python3` |
| Working directory | `~/sam3_lora_adjusted` |
| Git branch | `main` |
| Always set HF_TOKEN | In every SLURM job |
| Unified inference | `infer.py --mode single/batch/nested` |
| Masks for extraction/pairing | Always add `--masks` to `infer.py` |