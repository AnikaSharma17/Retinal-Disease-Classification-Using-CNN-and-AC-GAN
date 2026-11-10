"""Evaluate a saved classifier on a held-out <split>/<class>/ image folder.

Replaces the former Final/test.py (hard-coded paths, dropped the last partial batch) and
C2_Final/test2.py (called model.evaluate on an undefined `model`).

Usage (from the repo root):
    python -m legacy.classifiers.evaluate --model models/legacy/multiclass/c3_HypvsCHvsHD.h5 \
        --data-dir dataset/test --classes Hypertension Cholesterol "Heart Disease"
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from src.preprocessing import enhance_image

IMG_SIZE = (128, 128)


def evaluate(model_path: str, data_dir: str, classes: list[str], batch_size: int = 16) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    model = load_model(model_path)
    n_out = model.output_shape[-1]
    if n_out != len(classes):
        raise ValueError(f"model has {n_out} outputs but {len(classes)} classes were given: {classes}")

    gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        data_dir, target_size=IMG_SIZE, batch_size=batch_size, class_mode="categorical",
        classes=classes, shuffle=False,
    )
    if gen.samples == 0:
        raise ValueError(f"no images found under {data_dir} for classes {classes}")

    # Same enhancement as training, applied after rescale; iterate every batch, incl. the last partial one.
    probs, y_true = [], []
    for i in range(len(gen)):
        x, y = gen[i]
        x = np.stack([enhance_image(img) for img in x])
        probs.append(model.predict(x, verbose=0))
        y_true.append(y.argmax(axis=1))
    probs = np.concatenate(probs)
    y_true = np.concatenate(y_true)
    y_pred = probs.argmax(axis=1)

    counts = np.bincount(y_true, minlength=len(classes))
    result = {
        "n": int(len(y_true)),
        "class_counts": dict(zip(classes, counts.tolist())),
        "accuracy": float((y_pred == y_true).mean()),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(classes))).tolist(),
        "report": classification_report(
            y_true, y_pred, labels=range(len(classes)), target_names=classes, zero_division=0
        ),
    }
    present = counts > 0
    if present.all():
        if len(classes) == 2:
            result["auc"] = float(roc_auc_score(y_true, probs[:, 1]))
        else:
            result["auc_ovr_macro"] = float(roc_auc_score(y_true, probs, multi_class="ovr", labels=range(len(classes))))
    else:
        missing = [c for c, ok in zip(classes, present) if not ok]
        result["auc_note"] = f"AUC not computed: no test images for {missing}"
    if counts[present].min(initial=0) < 10:
        result["warning"] = "fewer than 10 test images for at least one class; metrics are not reliable"
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--classes", nargs="+", required=True, help="class folder names, in the model's output order")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args(argv)

    r = evaluate(args.model, args.data_dir, args.classes, args.batch_size)
    print(f"n={r['n']}  per-class={r['class_counts']}")
    print(f"accuracy={r['accuracy']:.4f}")
    for k in ("auc", "auc_ovr_macro", "auc_note", "warning"):
        if k in r:
            print(f"{k}: {r[k]}")
    print("confusion matrix (rows=true, cols=pred):")
    for row in r["confusion_matrix"]:
        print("  ", row)
    print(r["report"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
