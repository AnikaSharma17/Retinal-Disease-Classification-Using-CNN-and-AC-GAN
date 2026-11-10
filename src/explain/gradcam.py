"""Grad-CAM heatmaps and a shortcut-learning check.

    python -m src.explain.gradcam --run outputs/runs/<run> [--split test] [--max 16]

Writes <run>/gradcam/<split>/*.png (image | heatmap overlay) and gradcam_summary_<split>.json with,
per image, the share of Grad-CAM mass that falls OUTSIDE the circular fundus area. A model
that relies on borders, text or camera artifacts instead of the retina shows a high share;
that is the main thing to check before trusting any fundus classifier.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def fundus_mask(img_rgb: np.ndarray, threshold: int = 15) -> np.ndarray:
    """Boolean mask of the illuminated retinal disc (largest bright connected component)."""
    gray = cv2.cvtColor(np.asarray(img_rgb, np.uint8), cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw)
    if n <= 1:
        return np.ones(gray.shape, bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == largest


def outside_mask_fraction(heatmap: np.ndarray, mask: np.ndarray) -> float:
    total = float(heatmap.sum())
    return float(heatmap[~mask].sum() / total) if total > 0 else 0.0


def overlay(img_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    hm = cv2.applyColorMap(np.uint8(255 * np.clip(heatmap, 0, 1)), cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    return np.uint8(np.clip((1 - alpha) * img_rgb + alpha * hm, 0, 255))


def gradcam(model, image: np.ndarray, class_index: int | None = None,
            layer_name: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (heatmap HxW in [0,1], model probabilities) for one float/uint8 image in [0,255]."""
    import tensorflow as tf

    from src.models.factory import last_conv_layer_name

    layer = model.get_layer(layer_name or last_conv_layer_name(model))
    grad_model = tf.keras.Model(model.inputs, [layer.output, model.output])
    x = tf.convert_to_tensor(np.asarray(image, np.float32)[None])
    with tf.GradientTape() as tape:
        conv, preds = grad_model(x, training=False)
        idx = int(tf.argmax(preds[0])) if class_index is None else class_index
        score = preds[:, idx]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2))  # channel importance
    cam = tf.nn.relu(tf.reduce_sum(conv * weights[:, None, None, :], axis=-1))[0].numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cam.astype(np.float32), preds[0].numpy()


def explain_run(run_dir: str | Path, split: str = "test", max_images: int = 16) -> dict:
    import tensorflow as tf

    from src.data.dataset import prepare_image, read_rgb

    run_dir = Path(run_dir)
    info = json.loads((run_dir / "inference.json").read_text())
    model = tf.keras.models.load_model(run_dir / "model.keras")
    pred_file = run_dir / f"predictions_{split}.csv"
    if not pred_file.exists():
        raise FileNotFoundError(f"{pred_file} not found; run training first")
    rows = list(csv.DictReader(open(pred_file, encoding="utf-8")))[:max_images]
    out = run_dir / "gradcam" / split
    out.mkdir(parents=True, exist_ok=True)
    per_image = []
    for i, r in enumerate(rows):
        raw = prepare_image(read_rgb(r["path"]), info["img_size"], enhance=False)
        model_in = prepare_image(read_rgb(r["path"]), info["img_size"], info["enhance"])
        cam, probs = gradcam(model, model_in)
        frac = outside_mask_fraction(cam, fundus_mask(raw))
        panel = np.concatenate([raw, overlay(raw, cam)], axis=1)
        name = f"{i:03d}_true-{r['true'].replace(';', '+')}_pred-{r['pred'].replace(';', '+')}.png"
        cv2.imwrite(str(out / name), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        per_image.append({"path": r["path"], "true": r["true"], "pred": r["pred"],
                          "outside_fundus_fraction": round(frac, 4), "file": name})
    fr = [p["outside_fundus_fraction"] for p in per_image]
    summary = {"split": split, "n": len(per_image),
               "mean_outside_fundus_fraction": float(np.mean(fr)) if fr else None,
               "images": per_image}
    (run_dir / f"gradcam_summary_{split}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max", type=int, default=16)
    args = ap.parse_args(argv)
    s = explain_run(args.run, args.split, args.max)
    print(f"wrote {s['n']} Grad-CAM panels; mean attention outside the fundus: {s['mean_outside_fundus_fraction']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
