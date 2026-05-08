# HPC Command Reference — SAM3 LoRA

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
#SBATCH --time=7:00:00
#SBATCH --output=train_%j.out

export HF_TOKEN="hf_token"
cd ~/sam3_lora_adjusted
python3.10 train_sam3_lora_native.py --config configs/my_config-lite.yaml
EOF

sbatch train.sh
```

---

## 5. Validation

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
#SBATCH --time=1:00:00
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
#SBATCH --time=1:00:00
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
#SBATCH --time=1:00:00
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
#SBATCH --time=1:00:00
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

```bash
# LoRA inference (test set)
cat > infer_lora.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=infer_lora
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=infer_lora_%j.out

export HF_TOKEN="hf_token"
cd ~/sam3_lora_adjusted
python3.10 infer_vis.py
EOF

# Base inference (test set)
cat > infer_base.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=infer_base
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=infer_base_%j.out

export HF_TOKEN="hf_token"
cd ~/sam3_lora_adjusted
python3.10 infer_vis_base.py
EOF

# LoRA inference (all books in finerbook/)
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
python3.10 infer_vis2.py
EOF

sbatch infer_lora.sh
sbatch infer_base.sh
sbatch infer_lora2.sh
```

Check progress:
```bash
find ~/sam3_lora_adjusted/predictions/ -name "*.png" | wc -l

# Auto-refresh every 10 seconds
watch -n 10 "find ~/sam3_lora_adjusted/predictions/ -name '*.png' | wc -l"
```

---

### 6.1 Batch inference:
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

export HF_TOKEN="your_token_here"
cd ~/sam3_lora_adjusted
python3.10 infer_vis2.py
EOF

sbatch infer_lora2.sh
```


### 6.2 Batch inference with mask:


cat > infer_vmask.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=infer_vmask
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=infer_vmask_%j.out

cd ~/sam3_lora_adjusted
python3.10 infer_vmask.py
EOF

sbatch infer_vmask.sh


### 6.3 Segment by mask

cat > extract.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=extract_fg
#SBATCH --partition=voltagepark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=extract_%j.out

cd ~/sam3_lora_adjusted

python3.10 extract_foreground.py \
    --predictions_root predictions/lora_masks \
    --image_root finerbook \
    --padding 10 \
    --min_score 0.9
EOF

sbatch extract.sh

---

## 7. Upload Results to HuggingFace

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

## 8. Evaluation

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

## 9. Threshold Sweep

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