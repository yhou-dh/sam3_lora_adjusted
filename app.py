"""
app.py
SAM3 LoRA Pipeline — Gradio Web Interface
Run locally: python3 app.py

Tabs:
  0. Settings      — base_dir, weights, prompts, masks toggle
  1. Data Prep     — convert/validate annotations (optional)
  2. Training      — run training job
  3. Validation    — per-class or combined validation
  4. Inference     — run inference with visualisation + JSON output
  5. Evaluation    — compare predictions vs ground truth
  6. Extraction    — foreground mask / bbox / armed extraction

Requirements:
    pip install gradio
"""

import gradio as gr
import subprocess
import sys
import os
import json
import threading
from pathlib import Path
from datetime import datetime


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_base_dir():
    return str(Path.home() / "sam3_lora_adjusted")

def resolve_dir(path: str) -> str:
    """Expand ~ and resolve to absolute path."""
    return str(Path(path).expanduser().resolve())


def run_command(cmd: list, log_fn, cwd=None):
    """Run a subprocess and stream output line-by-line via log_fn callback."""
    resolved_cwd = str(Path(cwd or get_base_dir()).expanduser().resolve())
    log_fn(f"\n{'='*60}")
    log_fn(f"$ {' '.join(cmd)}")
    log_fn(f"cwd: {resolved_cwd}")
    log_fn(f"{'='*60}\n")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=resolved_cwd,
    )

    for line in iter(process.stdout.readline, ""):
        log_fn(line.rstrip())

    process.stdout.close()
    process.wait()

    if process.returncode == 0:
        log_fn(f"\n✅ Done (exit 0)")
    else:
        log_fn(f"\n❌ Failed (exit {process.returncode})")

    return process.returncode


def save_log(log_text: str, label: str) -> str:
    """Save log to a timestamped file and return the path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(get_base_dir()) / "logs"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / f"{label}_{ts}.txt"
    path.write_text(log_text)
    return str(path)


# ── Tab builders ─────────────────────────────────────────────────────────────

def build_settings_tab():
    with gr.Tab("⚙️ Settings"):
        gr.Markdown("### Global Configuration\nThese values are inherited by all tabs.")
        with gr.Row():
            base_dir  = gr.Textbox(label="Base directory",  value=str(Path.cwd()), scale=3)
            weights   = gr.Textbox(label="Weights path (or 'base')",
                                   value="outputs/sam3_lora_lite/best_lora_weights.pt", scale=3)
        with gr.Row():
            config    = gr.Textbox(label="Config YAML",
                                   value="configs/my_config-lite.yaml", scale=3)
            prompts   = gr.Textbox(label="Prompts (space-separated)",
                                   value="human illustration polearm", scale=3)
        with gr.Row():
            use_masks = gr.Checkbox(label="Always save RLE masks during inference", value=True)
            det_thr   = gr.Slider(0.1, 1.0, value=0.8,  step=0.05, label="Detection threshold")
            nms_thr   = gr.Slider(0.1, 1.0, value=0.15, step=0.05, label="NMS IoU threshold")
        gr.Markdown("*Changes take effect immediately for all subsequent runs.*")
    return base_dir, weights, config, prompts, use_masks, det_thr, nms_thr


def build_dataprep_tab(base_dir):
    with gr.Tab("📦 Data Prep *(optional)*"):
        gr.Markdown("### Data Preparation\nConvert annotations or validate your dataset splits.")
        with gr.Row():
            dp_data_root = gr.Textbox(label="Data root", value="data", scale=2)
            dp_command   = gr.Dropdown(
                label="Command",
                choices=["validate", "coco", "yolo"],
                value="validate",
                scale=1,
            )
            dp_split     = gr.Dropdown(label="Split", choices=["train", "valid", "test", "all"],
                                       value="train", scale=1)
        with gr.Accordion("COCO / YOLO extra options", open=False):
            dp_coco_json  = gr.Textbox(label="COCO JSON path (coco only)")
            dp_images_dir = gr.Textbox(label="Images dir (coco only)")
            dp_classes    = gr.Textbox(label="Class names comma-separated (yolo only)",
                                       value="human,illustration,polearm")

        dp_run_btn = gr.Button("▶ Run", variant="primary")
        dp_log     = gr.Textbox(label="Output", lines=20, interactive=False)
        dp_save    = gr.Button("💾 Save log")
        dp_saved   = gr.Textbox(label="Saved to", interactive=False)

    def run_dataprep(base_dir, data_root, command, split, coco_json, images_dir, classes):
        log_lines = []
        def log(l): log_lines.append(l); return "\n".join(log_lines)

        splits = ["train", "valid", "test"] if split == "all" else [split]
        output = []

        for s in splits:
            cmd = [sys.executable, "prepare_data.py", command]
            if command == "validate":
                cmd += ["--data_dir", data_root, "--split", s]
            elif command == "coco":
                cmd += ["--coco_json", coco_json, "--images_dir", images_dir,
                        "--output_dir", data_root, "--split", s]
            elif command == "yolo":
                cmd += ["--yolo_dir", data_root, "--output_dir", data_root,
                        "--classes", classes, "--split", s]

            def log_fn(line):
                log_lines.append(line)

            run_command(cmd, log_fn, cwd=base_dir)
            output.append("\n".join(log_lines))

        return "\n".join(output)

    dp_run_btn.click(
        run_dataprep,
        inputs=[base_dir, dp_data_root, dp_command, dp_split,
                dp_coco_json, dp_images_dir, dp_classes],
        outputs=dp_log,
    )
    dp_save.click(
        lambda log: save_log(log, "dataprep"),
        inputs=dp_log, outputs=dp_saved,
    )


def build_training_tab(base_dir, config, weights):
    with gr.Tab("🏋️ Training"):
        gr.Markdown("### Training\nFine-tune SAM3 with LoRA. Skip and set weights path in Settings if using a pre-trained model.")
        with gr.Row():
            tr_config = gr.Textbox(label="Config YAML", value="configs/my_config-lite.yaml", scale=3)
        tr_run_btn = gr.Button("▶ Start Training", variant="primary")
        tr_stop    = gr.Button("⏹ Stop", variant="stop")
        tr_log     = gr.Textbox(label="Training log", lines=25, interactive=False)
        tr_save    = gr.Button("💾 Save log")
        tr_saved   = gr.Textbox(label="Saved to", interactive=False)

        _proc = [None]

        def run_training(base_dir, config):
            log_lines = []
            def log_fn(line):
                log_lines.append(line)
            cmd = [sys.executable, "train_sam3_lora_native.py", "--config", config]
            run_command(cmd, log_fn, cwd=base_dir)
            return "\n".join(log_lines)

        tr_run_btn.click(run_training, inputs=[base_dir, tr_config], outputs=tr_log)
        tr_save.click(lambda log: save_log(log, "training"), inputs=tr_log, outputs=tr_saved)


def build_validation_tab(base_dir, config, weights):
    with gr.Tab("📊 Validation"):
        gr.Markdown("### Validation\nEvaluate model performance per class.")
        with gr.Row():
            val_data_dir  = gr.Textbox(label="Val data dir", value="data/valid", scale=2)
            val_classes   = gr.Textbox(label="Classes (space-separated, leave blank for all)",
                                       value="illustration human polearm", scale=2)
        with gr.Row():
            val_use_base  = gr.Checkbox(label="Use base model (no LoRA)", value=False)
            val_perclass  = gr.Checkbox(label="Per-class metrics", value=True)
        val_run_btn = gr.Button("▶ Run Validation", variant="primary")
        val_log     = gr.Textbox(label="Output", lines=25, interactive=False)
        val_save    = gr.Button("💾 Save log")
        val_saved   = gr.Textbox(label="Saved to", interactive=False)

        def run_validation(base_dir, config, weights, val_data_dir,
                           val_classes, use_base, perclass):
            log_lines = []
            def log_fn(line): log_lines.append(line)

            script = "validate_sam3_perclass.py" if perclass else "validate_sam3_lora.py"
            cmd = [sys.executable, script,
                   "--val_data_dir", val_data_dir]

            if not use_base:
                cmd += ["--config", config, "--weights", weights]
            else:
                cmd += ["--use-base-model"]

            if perclass and val_classes.strip():
                cmd += ["--classes"] + val_classes.strip().split()

            run_command(cmd, log_fn, cwd=base_dir)
            return "\n".join(log_lines)

        val_run_btn.click(
            run_validation,
            inputs=[base_dir, config, weights, val_data_dir,
                    val_classes, val_use_base, val_perclass],
            outputs=val_log,
        )
        val_save.click(lambda log: save_log(log, "validation"), inputs=val_log, outputs=val_saved)


def build_inference_tab(base_dir, config, weights, prompts, use_masks, det_thr, nms_thr):
    with gr.Tab("🔍 Inference"):
        gr.Markdown("### Inference\nRun SAM3 on a book folder. Outputs visualisation PNGs + `book_predictions.json`.")
        with gr.Row():
            inf_book_root       = gr.Textbox(label="Book root (folder containing book subfolders)",
                                             value="data", scale=2)
            inf_predictions_root = gr.Textbox(label="Predictions output root",
                                              value="predictions/lora", scale=2)
        with gr.Row():
            inf_mode      = gr.Radio(label="Mode",
                                     choices=["single", "batch", "nested"],
                                     value="batch",
                                     info="single=one folder, batch=immediate subfolders, nested=all leaf folders")
            inf_skip_done = gr.Checkbox(label="Skip already-processed books", value=True)

        inf_run_btn = gr.Button("▶ Run Inference", variant="primary")
        inf_log     = gr.Textbox(label="Output", lines=25, interactive=False)
        inf_save    = gr.Button("💾 Save log")
        inf_saved   = gr.Textbox(label="Saved to", interactive=False)

        def run_inference(base_dir, config, weights, prompts, use_masks,
                          det_thr, nms_thr, book_root, pred_root,
                          inf_mode, skip_done):
            log_lines = []
            def log_fn(line): log_lines.append(line)

            prompt_list = prompts.strip().split()

            cmd = [sys.executable, "infer.py",
                   "--input",               book_root,
                   "--mode",                inf_mode,
                   "--base_dir",            base_dir,
                   "--predictions_root",    pred_root,
                   "--config",              config,
                   "--weights",             weights,
                   "--detection_threshold", str(det_thr),
                   "--nms_iou_threshold",   str(nms_thr),
                   "--prompts"] + prompt_list

            if use_masks:
                cmd.append("--masks")
            if skip_done:
                cmd.append("--skip_done")

            run_command(cmd, log_fn, cwd=base_dir)
            return "\n".join(log_lines)

        inf_run_btn.click(
            run_inference,
            inputs=[base_dir, config, weights, prompts, use_masks,
                    det_thr, nms_thr, inf_book_root, inf_predictions_root,
                    inf_mode, inf_skip_done],
            outputs=inf_log,
        )
        inf_save.click(lambda log: save_log(log, "inference"), inputs=inf_log, outputs=inf_saved)


def build_evaluation_tab(base_dir):
    with gr.Tab("📈 Evaluation"):
        gr.Markdown("### Post-Inference Evaluation\nCompare model predictions against ground truth annotations.")
        with gr.Row():
            ev_pred1   = gr.Textbox(label="Predictions JSON (model 1, e.g. LoRA)",
                                    value="predictions/lora/test/summaries/book_predictions.json",
                                    scale=3)
            ev_name1   = gr.Textbox(label="Model 1 name", value="LoRA", scale=1)
        with gr.Row():
            ev_pred2   = gr.Textbox(label="Predictions JSON (model 2, optional)",
                                    placeholder="predictions/base/test/summaries/book_predictions.json",
                                    scale=3)
            ev_name2   = gr.Textbox(label="Model 2 name", value="Base", scale=1)
        with gr.Row():
            ev_ann     = gr.Textbox(label="Ground truth annotations (COCO JSON)",
                                    value="data/test/_annotations.coco.json", scale=3)
            ev_mode    = gr.Dropdown(label="Mode", choices=["evaluate", "sweep"],
                                     value="evaluate", scale=1)

        ev_run_btn = gr.Button("▶ Run Evaluation", variant="primary")
        ev_log     = gr.Textbox(label="Output", lines=25, interactive=False)
        ev_save    = gr.Button("💾 Save log")
        ev_saved   = gr.Textbox(label="Saved to", interactive=False)

        def run_evaluation(base_dir, pred1, name1, pred2, name2, ann, mode):
            log_lines = []
            def log_fn(line): log_lines.append(line)

            script = "evaluate_detections.py" if mode == "evaluate" else "threshold_sweep.py"

            for pred, name in [(pred1, name1), (pred2, name2)]:
                if not pred.strip():
                    continue
                cmd = [sys.executable, script,
                       "--predictions", pred,
                       "--annotations",  ann,
                       "--model_name",   name]
                run_command(cmd, log_fn, cwd=base_dir)

            return "\n".join(log_lines)

        ev_run_btn.click(
            run_evaluation,
            inputs=[base_dir, ev_pred1, ev_name1, ev_pred2, ev_name2, ev_ann, ev_mode],
            outputs=ev_log,
        )
        ev_save.click(lambda log: save_log(log, "evaluation"), inputs=ev_log, outputs=ev_saved)


def build_extraction_tab(base_dir):
    with gr.Tab("✂️ Extraction"):
        gr.Markdown("### Extraction\nCrop objects from images using mask or bbox predictions.")
        with gr.Row():
            ex_mode           = gr.Radio(label="Mode",
                                         choices=["foreground (mask)", "bbox", "armed humans"],
                                         value="bbox")
            ex_pred_root      = gr.Textbox(label="Predictions root",
                                           value="predictions/lora", scale=2)
            ex_image_root     = gr.Textbox(label="Image root", value="data", scale=2)
        with gr.Row():
            ex_output_root    = gr.Textbox(label="Output root (armed only)",
                                           value="predictions/armed", scale=2)
            ex_book           = gr.Textbox(label="Single book (leave blank for all)",
                                           placeholder="test", scale=2)
        with gr.Row():
            ex_padding        = gr.Slider(0, 50, value=10, step=1, label="Padding (px)")
            ex_min_score      = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Min score")
            ex_overlap_dil    = gr.Slider(0, 100, value=20, step=5,
                                          label="Overlap dilation px (armed only)")

        ex_run_btn = gr.Button("▶ Run Extraction", variant="primary")
        ex_log     = gr.Textbox(label="Output", lines=20, interactive=False)
        ex_save    = gr.Button("💾 Save log")
        ex_saved   = gr.Textbox(label="Saved to", interactive=False)

        def run_extraction(base_dir, mode, pred_root, image_root,
                           output_root, book, padding, min_score, overlap_dil):
            log_lines = []
            def log_fn(line): log_lines.append(line)

            script_map = {
                "foreground (mask)": "extract_foreground.py",
                "bbox":              "extract_bbox.py",
                "armed humans":      "extract_armed.py",
            }
            script = script_map[mode]

            cmd = [sys.executable, script,
                   "--predictions_root", pred_root,
                   "--image_root",       image_root,
                   "--padding",          str(padding),
                   "--min_score",        str(min_score)]

            if mode == "armed humans":
                cmd += ["--output_root", output_root,
                        "--overlap_dilation", str(overlap_dil)]

            if book.strip():
                cmd += ["--book", book.strip()]

            run_command(cmd, log_fn, cwd=base_dir)
            return "\n".join(log_lines)

        ex_run_btn.click(
            run_extraction,
            inputs=[base_dir, ex_mode, ex_pred_root, ex_image_root,
                    ex_output_root, ex_book, ex_padding, ex_min_score, ex_overlap_dil],
            outputs=ex_log,
        )
        ex_save.click(lambda log: save_log(log, "extraction"), inputs=ex_log, outputs=ex_saved)


# ── Main app ─────────────────────────────────────────────────────────────────

def build_visualise_tab(base_dir):
    with gr.Tab("🖼️ Pair Gallery"):
        gr.Markdown("### Paired Results Gallery\nGenerate a browsable HTML gallery from image-text pairs. Toggle between simple and enriched views.")
        with gr.Row():
            vis_pairs    = gr.Textbox(label="Pairs JSON", value="pairs/output/image_text_pairs.json", scale=3)
            vis_enriched = gr.Textbox(label="Enriched JSON (optional)", value="pairs/output/image_text_enriched.json", scale=3)
        with gr.Row():
            vis_img_base = gr.Textbox(label="Image base folder", value="predictions", scale=3)
            vis_output   = gr.Textbox(label="Output HTML", value="pairs/output/gallery.html", scale=3)
        vis_run_btn  = gr.Button("▶ Generate Gallery", variant="primary")
        vis_log      = gr.Textbox(label="Output", lines=8, interactive=False)
        vis_open     = gr.Textbox(label="Open in browser", interactive=False)

        def run_visualise(base_dir, pairs, enriched, img_base, output):
            log_lines = []
            def log_fn(line): log_lines.append(line)
            cmd = [sys.executable, "visualise_pairs.py",
                   "--pairs",      pairs,
                   "--image_base", img_base,
                   "--output",     output]
            if enriched.strip():
                cmd += ["--enriched", enriched]
            run_command(cmd, log_fn, cwd=base_dir)
            from pathlib import Path
            out_path = Path(base_dir) / output
            browser_path = f"file://{out_path.resolve()}" if out_path.exists() else "Not found"
            return "\n".join(log_lines), browser_path

        vis_run_btn.click(
            run_visualise,
            inputs=[base_dir, vis_pairs, vis_enriched, vis_img_base, vis_output],
            outputs=[vis_log, vis_open],
        )

def build_pairing_tab(base_dir):
    with gr.Tab("🔗 Image-Text Pairing"):
        gr.Markdown("### Image-Text Pairing\nPair segmented illustration crops with text and metadata from CSV files.")
        with gr.Row():
            pr_csv_dir       = gr.Textbox(label="CSV folder", value="pairs/csv", scale=2)
            pr_pred_root     = gr.Textbox(label="Predictions root", value="predictions/lora", scale=2)
        with gr.Row():
            pr_extraction    = gr.Radio(label="Extraction type",
                                        choices=["mask", "bbox"], value="mask",
                                        info="mask=foreground/ crops, bbox=bbox/ crops")
            pr_mode          = gr.Radio(label="Output mode",
                                        choices=["combined", "per_book", "both"], value="combined")
        pr_output_dir    = gr.Textbox(label="Output folder", value="pairs/output")
        pr_run_btn       = gr.Button("▶ Run Pairing", variant="primary")
        pr_log           = gr.Textbox(label="Output", lines=20, interactive=False)
        pr_save          = gr.Button("💾 Save log")
        pr_saved         = gr.Textbox(label="Saved to", interactive=False)

        def run_pairing(base_dir, csv_dir, pred_root, extraction, mode, output_dir):
            log_lines = []
            def log_fn(line): log_lines.append(line)
            cmd = [sys.executable, "pair_data.py",
                   "--csv_dir",          csv_dir,
                   "--predictions_root", pred_root,
                   "--extraction_type",  extraction,
                   "--output_dir",       output_dir,
                   "--mode",             mode]
            run_command(cmd, log_fn, cwd=base_dir)
            return "\n".join(log_lines)

        pr_run_btn.click(
            run_pairing,
            inputs=[base_dir, pr_csv_dir, pr_pred_root, pr_extraction, pr_mode, pr_output_dir],
            outputs=pr_log,
        )
        pr_save.click(lambda log: save_log(log, "pairing"), inputs=pr_log, outputs=pr_saved)


with gr.Blocks(title="MILens") as demo:

    gr.Markdown("""
# MILens — Manuscript Illustration Lens
**Data Prep → Training → Validation → Inference → Evaluation → Extraction → Image-Text Pairing**

Configure global settings in the ⚙️ Settings tab first. All logs can be exported.
""")

    base_dir, weights, config, prompts, use_masks, det_thr, nms_thr = build_settings_tab()

    build_dataprep_tab(base_dir)
    build_training_tab(base_dir, config, weights)
    build_validation_tab(base_dir, config, weights)
    build_inference_tab(base_dir, config, weights, prompts, use_masks, det_thr, nms_thr)
    build_evaluation_tab(base_dir)
    build_extraction_tab(base_dir)
    build_pairing_tab(base_dir)
    build_visualise_tab(base_dir)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
        theme=gr.themes.Base(
            primary_hue="slate",
            secondary_hue="zinc",
            neutral_hue="zinc",
            font=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
        ),
    )
