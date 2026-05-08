"""
evaluate_detections.py
Evaluates SAM3 LoRA / Base model detection results against COCO ground truth.

Usage:
    python3 evaluate_detections.py \
        --predictions predictions/lora/test/summaries/book_predictions.json \
        --annotations data/test/_annotations.coco.json \
        --model_name LoRA

    python3 evaluate_detections.py \
        --predictions predictions/base/test/summaries/book_predictions_base.json \
        --annotations data/test/_annotations.coco.json \
        --model_name Base
"""

import json
import argparse
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ── Prompt → COCO category ID mapping ────────────────────────────────────────
PROMPT_TO_CATEGORY_ID = {
    "human":        5,
    "illustration": 6,
    "polearm":      7,
    # extend here if you add more prompts
}


def xywh_from_xyxy(box):
    """Convert [x1, y1, x2, y2] → [x, y, w, h] (COCO format)."""
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def load_predictions(pred_path: str, filename_to_id: dict) -> list:
    """
    Read book_predictions.json and convert to COCO results format.
    Returns a list of dicts: {image_id, category_id, bbox, score}
    """
    with open(pred_path) as f:
        preds = json.load(f)

    results = []
    skipped_files = 0

    for item in preds:
        fname = item["file_name"]
        image_id = filename_to_id.get(fname)

        if image_id is None:
            skipped_files += 1
            continue

        for det in item["detections"]:
            prompt = det["prompt"]

            # prompt may be stored as string name or integer index
            if isinstance(prompt, int):
                # map index → name using the order used during inference
                idx_to_name = {i: name for i, name in
                               enumerate(PROMPT_TO_CATEGORY_ID.keys())}
                prompt = idx_to_name.get(prompt)

            category_id = PROMPT_TO_CATEGORY_ID.get(prompt)
            if category_id is None:
                continue

            boxes  = det.get("boxes",  [])
            scores = det.get("scores", [])

            for box, score in zip(boxes, scores):
                results.append({
                    "image_id":   image_id,
                    "category_id": category_id,
                    "bbox":       xywh_from_xyxy(box),
                    "score":      float(score),
                })

    if skipped_files:
        print(f"  ⚠ Skipped {skipped_files} images (not found in annotations)")

    return results


def evaluate(pred_path: str, ann_path: str, model_name: str):
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"  Predictions : {pred_path}")
    print(f"  Annotations : {ann_path}")
    print(f"{'='*60}\n")

    # ── Load ground truth ────────────────────────────────────────────────────
    coco_gt = COCO(ann_path)

    # Build filename → image_id lookup
    filename_to_id = {
        img["file_name"].replace("images/", "").replace("images\\", ""): img["id"]
        for img in coco_gt.dataset["images"]
    }

    # ── Load & convert predictions ───────────────────────────────────────────
    results = load_predictions(pred_path, filename_to_id)
    print(f"  Total detections loaded: {len(results)}")

    if not results:
        print("  ❌ No valid detections found. Check prompt names and file names.")
        return

    # ── Run COCO evaluation ──────────────────────────────────────────────────
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")

    # Only evaluate on the categories we actually predicted
    eval_cat_ids = list(PROMPT_TO_CATEGORY_ID.values())
    coco_eval.params.catIds = eval_cat_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # ── Per-class AP ─────────────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("  Per-class AP@50:95")
    print(f"{'─'*40}")

    id_to_name = {v: k for k, v in PROMPT_TO_CATEGORY_ID.items()}

    per_class = {}
    for cat_id in eval_cat_ids:
        coco_eval_cat = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval_cat.params.catIds = [cat_id]
        coco_eval_cat.evaluate()
        coco_eval_cat.accumulate()
        ap = coco_eval_cat.stats[0]  # mAP@50:95
        ap50 = coco_eval_cat.stats[1]  # mAP@50
        name = id_to_name[cat_id]
        per_class[name] = {"AP@50:95": round(float(ap), 4),
                           "AP@50":    round(float(ap50), 4)}
        print(f"  {name:<20} AP@50:95={ap:.4f}  AP@50={ap50:.4f}")

    # ── Precision / Recall / F1 ──────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("  Summary Metrics")
    print(f"{'─'*40}")
    stats = coco_eval.stats
    metrics = {
        "mAP@50:95":  round(float(stats[0]), 4),
        "mAP@50":     round(float(stats[1]), 4),
        "mAP@75":     round(float(stats[2]), 4),
        "AR@1":       round(float(stats[6]), 4),
        "AR@10":      round(float(stats[7]), 4),
        "AR@100":     round(float(stats[8]), 4),
    }
    for k, v in metrics.items():
        print(f"  {k:<15} {v:.4f}")

    # ── Save results to JSON ─────────────────────────────────────────────────
    output = {
        "model":      model_name,
        "summary":    metrics,
        "per_class":  per_class,
    }
    out_path = Path(pred_path).parent / f"eval_results_{model_name.lower()}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"\n  ✅ Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True,
                        help="Path to book_predictions.json")
    parser.add_argument("--annotations", required=True,
                        help="Path to _annotations.coco.json")
    parser.add_argument("--model_name",  default="model",
                        help="Label for this model (e.g. LoRA or Base)")
    args = parser.parse_args()

    evaluate(args.predictions, args.annotations, args.model_name)
