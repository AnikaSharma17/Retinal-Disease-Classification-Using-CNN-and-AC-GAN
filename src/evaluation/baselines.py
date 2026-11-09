"""Non-deep baselines every CNN result must beat, evaluated on the same split.

    python -m src.evaluation.baselines --config configs/multilabel_efficientnet.yaml [--embeddings]

* prior: predicts the training-set class frequencies for every image (AUC = 0.5 by construction)
* colour_lr: logistic regression on per-channel colour histograms + simple texture statistics
* embedding_lr (--embeddings): logistic regression on frozen ImageNet EfficientNet-B0 features,
  the strongest cheap baseline for tiny datasets
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ExperimentConfig, load_config
from src.data.dataset import Records, load_images, read_manifest, select_records
from src.evaluation.metrics import compute_metrics, format_headline


def handcrafted_features(images: np.ndarray) -> np.ndarray:
    feats = []
    for img in images:
        hist = [cv2.calcHist([img], [c], None, [16], [0, 256]).ravel() for c in range(3)]
        hist = np.concatenate(hist) / img[..., 0].size
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        stats = [gray.mean(), gray.std(), np.abs(lap).mean(), lap.var(),
                 *img.reshape(-1, 3).mean(0), *img.reshape(-1, 3).std(0)]
        feats.append(np.concatenate([hist, np.array(stats) / 255.0]))
    return np.array(feats, np.float32)


def embedding_features(images: np.ndarray, batch: int = 16) -> np.ndarray:
    import tensorflow as tf

    base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", pooling="avg",
                                                input_shape=images.shape[1:])
    return base.predict(images.astype("float32"), batch_size=batch, verbose=0)


def _fit_predict(cfg: ExperimentConfig, Xtr, ytr, Xte) -> np.ndarray:
    k = len(cfg.data.classes)
    if cfg.data.task == "multiclass":
        present = np.unique(ytr)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced"))
        clf.fit(Xtr, ytr)
        probs = np.zeros((len(Xte), k))
        probs[:, present] = clf.predict_proba(Xte)
        return probs
    probs = np.zeros((len(Xte), k))
    for j in range(k):
        y = ytr[:, j]
        if y.min() == y.max():  # label constant in training: predict its frequency
            probs[:, j] = y.mean()
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced"))
        clf.fit(Xtr, y)
        probs[:, j] = clf.predict_proba(Xte)[:, 1]
    return probs


def prior_probs(cfg: ExperimentConfig, train: Records, n: int) -> np.ndarray:
    k = len(cfg.data.classes)
    if cfg.data.task == "multiclass":
        freq = np.bincount(train.targets, minlength=k) / max(len(train), 1)
    else:
        freq = train.targets.mean(0)
    return np.tile(freq, (n, 1))


def run_baselines(cfg: ExperimentConfig, embeddings: bool = False, split: str = "test") -> dict:
    rows = read_manifest(cfg.data.manifest)
    train = select_records(rows, cfg.data.classes, cfg.data.task, cfg.data.negative_label, ("train", "val"))
    test = select_records(rows, cfg.data.classes, cfg.data.task, cfg.data.negative_label, (split,))
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"need train and {split} images (got {len(train)}, {len(test)})")
    m = lambda probs: compute_metrics(cfg.data.task, test.targets, probs, cfg.data.classes,  # noqa: E731
                                      cfg.train.threshold, cfg.bootstrap, cfg.seed)
    out = {"prior": m(prior_probs(cfg, train, len(test)))}
    size = min(cfg.data.img_size, 224)
    xtr = load_images(train.paths, size, cfg.data.enhance)
    xte = load_images(test.paths, size, cfg.data.enhance)
    out["colour_lr"] = m(_fit_predict(cfg, handcrafted_features(xtr), train.targets, handcrafted_features(xte)))
    if embeddings:
        out["embedding_lr"] = m(_fit_predict(cfg, embedding_features(xtr), train.targets, embedding_features(xte)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--embeddings", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    res = run_baselines(cfg, args.embeddings)
    for name, met in res.items():
        print(f"{name:<14} {format_headline(met)}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
