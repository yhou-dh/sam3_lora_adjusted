"""
threshold_sweep.py
Sweeps confidence thresholds and computes Precision, Recall, F1 vs threshold.
Finds the optimal threshold per class and overall.

Usage:
    python3.10 threshold_sweep.py \
        --predictions predictions/lora/test/summaries/book_predictions.json \
        --annotations data/test/_annotations.coco.json \
        --model_name LoRA \
        --iou_threshold 0.5
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
}

IDX_TO_PROMPT = {i: name for i, name in enumerate(PROMPT_TO_CATEGORY_ID.keys())}


def xywh_from_xyxy(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def load_all_detections(pred_path: str, filename_to_id: dict) -> list:
    """Load all detections regardless of threshold."""
    with open(pred_path) as f:
        preds = json.load(f)

    results = []
    for item in preds:
        fname = item["file_name"]
        image_id = filename_to_id.get(fname)
        if image_id is None:
            continue

        for det in item["detections"]:
            prompt = det["prompt"]
            if isinstance(prompt, int):
                prompt = IDX_TO_PROMPT.get(prompt)
            category_id = PROMPT_TO_CATEGORY_ID.get(prompt)
            if category_id is None:
                continue

            for box, score in zip(det.get("boxes", []), det.get("scores", [])):
                results.append({
                    "image_id":    image_id,
                    "category_id": category_id,
                    "bbox":        xywh_from_xyxy(box),
                    "score":       float(score),
                    "prompt":      prompt,
                })
    return results


def compute_metrics_at_threshold(coco_gt, all_detections, threshold, cat_ids):
    """Filter detections by threshold and run COCO eval."""
    filtered = [d for d in all_detections if d["score"] >= threshold]
    if not filtered:
        return None

    try:
        coco_dt = coco_gt.loadRes(filtered)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
        return {
            "mAP@50:95": float(stats[0]),
            "mAP@50":    float(stats[1]),
            "mAP@75":    float(stats[2]),
            "AR@100":    float(stats[8]),
        }
    except Exception:
        return None


def compute_pr_f1(coco_gt, all_detections, threshold, cat_id):
    """Compute precision, recall, F1 for a single category at a threshold."""
    filtered = [d for d in all_detections
                if d["score"] >= threshold and d["category_id"] == cat_id]

    gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(catIds=[cat_id]))
    n_gt = len(gt_anns)

    if not filtered or n_gt == 0:
        return 0.0, 0.0, 0.0

    try:
        coco_dt = coco_gt.loadRes(filtered)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.catIds = [cat_id]
        coco_eval.params.iouThrs = np.array([0.5])
        coco_eval.evaluate()
        coco_eval.accumulate()

        # Extract precision and recall from accumulation
        precision = coco_eval.eval["precision"]  # shape: T x R x K x A x M
        recall_thrs = coco_eval.params.recThrs

        # precision at iou=0.5, all areas, max dets=100
        prec = precision[0, :, 0, 0, 2]
        prec = prec[prec > -1]

        if len(prec) == 0:
            return 0.0, 0.0, 0.0

        mean_prec = float(np.mean(prec))

        # recall = max recall threshold with non-zero precision
        rec_idx = np.where(prec > 0)[0]
        mean_rec = float(recall_thrs[rec_idx[-1]]) if len(rec_idx) > 0 else 0.0

        f1 = (2 * mean_prec * mean_rec / (mean_prec + mean_rec)
              if (mean_prec + mean_rec) > 0 else 0.0)

        return mean_prec, mean_rec, f1

    except Exception:
        return 0.0, 0.0, 0.0


def sweep(pred_path: str, ann_path: str, model_name: str):
    print(f"\n{'='*60}")
    print(f"  Threshold Sweep: {model_name}")
    print(f"{'='*60}\n")

    coco_gt = COCO(ann_path)

    filename_to_id = {
        img["file_name"].replace("images/", "").replace("images\\", ""): img["id"]
        for img in coco_gt.dataset["images"]
    }

    all_detections = load_all_detections(pred_path, filename_to_id)
    print(f"  Total detections: {len(all_detections)}\n")

    cat_ids = list(PROMPT_TO_CATEGORY_ID.values())
    id_to_name = {v: k for k, v in PROMPT_TO_CATEGORY_ID.items()}

    thresholds = np.arange(0.1, 0.96, 0.05).round(2)

    # ── Overall mAP sweep ────────────────────────────────────────────────────
    print("Overall mAP@50 vs Threshold:")
    print(f"  {'Threshold':<12} {'mAP@50':<10} {'mAP@50:95':<12} {'AR@100'}")
    print(f"  {'─'*50}")

    overall_results = []
    for thr in thresholds:
        metrics = compute_metrics_at_threshold(coco_gt, all_detections, thr, cat_ids)
        if metrics:
            overall_results.append((thr, metrics))
            print(f"  {thr:<12.2f} {metrics['mAP@50']:<10.4f} "
                  f"{metrics['mAP@50:95']:<12.4f} {metrics['AR@100']:.4f}")

    # Best overall threshold by mAP@50
    if overall_results:
        best_thr, best_metrics = max(overall_results, key=lambda x: x[1]["mAP@50"])
        print(f"\n  ✅ Best overall threshold (mAP@50): {best_thr:.2f} "
              f"→ mAP@50={best_metrics['mAP@50']:.4f}")

    # ── Per-class F1 sweep ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Per-class F1 vs Threshold:")

    per_class_best = {}
    for cat_id in cat_ids:
        name = id_to_name[cat_id]
        print(f"\n  [{name}]")
        print(f"  {'Threshold':<12} {'Precision':<12} {'Recall':<10} {'F1'}")
        print(f"  {'─'*45}")

        class_results = []
        for thr in thresholds:
            p, r, f1 = compute_pr_f1(coco_gt, all_detections, thr, cat_id)
            class_results.append((thr, p, r, f1))
            print(f"  {thr:<12.2f} {p:<12.4f} {r:<10.4f} {f1:.4f}")

        if class_results:
            best = max(class_results, key=lambda x: x[3])
            per_class_best[name] = {
                "best_threshold": best[0],
                "precision":      round(best[1], 4),
                "recall":         round(best[2], 4),
                "f1":             round(best[3], 4),
            }
            print(f"\n  ✅ Best threshold for {name}: {best[0]:.2f} "
                  f"→ F1={best[3]:.4f} (P={best[1]:.4f}, R={best[2]:.4f})")

    # ── Save results ─────────────────────────────────────────────────────────
    output = {
        "model":           model_name,
        "best_overall":    {"threshold": float(best_thr), **best_metrics} if overall_results else {},
        "per_class_best":  per_class_best,
        "overall_sweep":   [(float(t), m) for t, m in overall_results],
    }

    out_path = Path(pred_path).parent / f"threshold_sweep_{model_name.lower()}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"\n  ✅ Sweep results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--annotations",  required=True)
    parser.add_argument("--model_name",   default="model")
    args = parser.parse_args()

    sweep(args.predictions, args.annotations, args.model_name)
