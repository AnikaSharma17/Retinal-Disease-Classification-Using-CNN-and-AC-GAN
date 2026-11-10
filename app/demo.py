"""Gradio demo: upload a fundus photo, get per-class probabilities and a Grad-CAM overlay.

    python -m app.demo --run outputs/runs/<run>          # or set RETINA_RUN_DIR
    python -m app.demo                                   # newest run under outputs/runs

Research prototype only. The model is trained on a few dozen images and is not a medical
device; the UI says so on every prediction.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

DISCLAIMER = (
    "**Research prototype - not for clinical use.** Trained on a very small dataset; outputs are "
    "illustrative and must not inform any medical decision."
)


def latest_run(root: str = "outputs/runs") -> Path | None:
    runs = [p.parent for p in Path(root).glob("*/inference.json")]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def build_app(run_dir: str | Path):
    import gradio as gr

    from src.explain.gradcam import fundus_mask, gradcam, outside_mask_fraction, overlay
    from src.predict import Predictor

    predictor = Predictor(run_dir)
    info = predictor.info
    metrics_path = Path(run_dir) / "metrics.json"
    test_summary = ""
    if metrics_path.exists():
        from src.evaluation.metrics import format_headline

        test = json.loads(metrics_path.read_text()).get("test")
        if test:
            test_summary = f"Held-out test metrics of this run: `{format_headline(test)}`"

    def infer(image: np.ndarray | None):
        if image is None:
            return None, None, "Upload an image first."
        if image.ndim == 2:
            image = np.stack([image] * 3, -1)
        image = image[..., :3].astype(np.uint8)
        result = predictor.predict_array(image)
        model_in = predictor.preprocess(image)
        from src.data.dataset import prepare_image

        shown = prepare_image(image, info["img_size"], enhance=False)
        cam, _ = gradcam(predictor.model, model_in)
        outside = outside_mask_fraction(cam, fundus_mask(shown))
        pred = result["prediction"]
        lines = [f"**Prediction:** {', '.join(pred) if isinstance(pred, list) else pred}"]
        if result["low_confidence"]:
            lines.append("Low confidence: treat as inconclusive.")
        if outside > 0.3:
            lines.append(f"Warning: {outside:.0%} of the model's attention is outside the retina "
                         "(possible artifact / shortcut).")
        lines.append(DISCLAIMER)
        return result["probabilities"], overlay(shown, cam), "\n\n".join(lines)

    with gr.Blocks(title="Retinal fundus classifier (research demo)") as demo:
        gr.Markdown(f"# Retinal fundus classifier\nRun: `{Path(run_dir).name}` - task: {info['task']} - "
                    f"classes: {', '.join(info['classes'])}\n\n{test_summary}\n\n{DISCLAIMER}")
        with gr.Row():
            inp = gr.Image(type="numpy", label="Fundus photograph")
            with gr.Column():
                probs = gr.Label(label="Probabilities", num_top_classes=len(info["classes"]))
                cam = gr.Image(label="Grad-CAM (where the model looked)")
        notes = gr.Markdown()
        gr.Button("Analyse", variant="primary").click(infer, inp, [probs, cam, notes])
    return demo


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=os.environ.get("RETINA_RUN_DIR"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args(argv)
    run_dir = Path(args.run) if args.run else latest_run()
    if run_dir is None or not (run_dir / "inference.json").exists():
        print("No trained run found. Train one first:")
        print("  python -m src.train --config configs/multilabel_efficientnet.yaml")
        return 1
    build_app(run_dir).launch(server_name=args.host, server_port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
