"""Predict one or more fundus images with a trained run.

    python -m src.predict --run outputs/runs/<run> image1.jpg [image2.jpg ...]

Prints JSON: per-class probabilities, the decision, and a low-confidence flag. Research use
only; this is not a diagnostic device.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


class Predictor:
    def __init__(self, run_dir: str | Path, low_confidence: float = 0.6):
        import tensorflow as tf

        self.run_dir = Path(run_dir)
        self.info = json.loads((self.run_dir / "inference.json").read_text())
        self.model = tf.keras.models.load_model(self.run_dir / "model.keras")
        self.low_confidence = low_confidence

    def preprocess(self, img_rgb: np.ndarray) -> np.ndarray:
        from src.data.dataset import prepare_image

        return prepare_image(img_rgb, self.info["img_size"], self.info["enhance"])

    def predict_array(self, img_rgb: np.ndarray) -> dict:
        x = self.preprocess(img_rgb).astype("float32")[None]
        probs = self.model.predict(x, verbose=0)[0]
        classes = self.info["classes"]
        result = {"probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)}}
        if self.info["task"] == "multiclass":
            top = int(np.argmax(probs))
            result["prediction"] = classes[top]
            result["low_confidence"] = bool(probs[top] < self.low_confidence)
        else:
            thr = self.info["threshold"]
            positives = [c for c, p in zip(classes, probs) if p >= thr]
            result["prediction"] = positives or [self.info["negative_label"]]
            # "uncertain" if any label sits close to the threshold
            result["low_confidence"] = bool(np.any(np.abs(probs - thr) < (self.low_confidence - 0.5)))
        return result

    def predict_path(self, path: str | Path) -> dict:
        from src.data.dataset import read_rgb

        return {"image": str(path), **self.predict_array(read_rgb(path))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("images", nargs="+")
    args = ap.parse_args(argv)
    p = Predictor(args.run)
    print(json.dumps([p.predict_path(i) for i in args.images], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
