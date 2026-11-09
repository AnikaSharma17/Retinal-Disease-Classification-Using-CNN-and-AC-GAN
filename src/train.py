"""Train and evaluate one experiment from a YAML config.

    python -m src.train --config configs/multilabel_efficientnet.yaml
    python -m src.train --config configs/multilabel_efficientnet.yaml --set cv_folds=5 train.epochs=15

Single run (cv_folds=0): fit on `train`, early-stop on `val`, report val + test metrics.
Cross-validation (cv_folds>=2): StratifiedGroupKFold over train+val with a FIXED epoch budget
(no early stopping, so out-of-fold predictions are untouched by model selection), then one
model is fit on all of train+val and evaluated once on the held-out `test` split.

Each run writes to outputs/runs/<name>-<timestamp>/:
    config.yaml, metadata.json, history.json, metrics.json, predictions_<split>.csv,
    model.keras, inference.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from src.config import ExperimentConfig, load_config
from src.data.dataset import Records, load_images, make_dataset, read_manifest, select_records, synthetic_records
from src.evaluation.metrics import compute_metrics, format_headline, headline
from src.utils import environment_info, file_sha256, git_state, set_seed


def _class_weight(cfg: ExperimentConfig, y: np.ndarray) -> dict | None:
    if cfg.data.task != "multiclass" or not cfg.train.class_weight or len(y) == 0:
        return None
    k = len(cfg.data.classes)
    counts = np.bincount(y, minlength=k).astype(float)
    present = counts > 0
    w = np.zeros(k)
    w[present] = len(y) / (present.sum() * counts[present])
    return {i: float(w[i]) for i in range(k)}


def _compile(model, cfg: ExperimentConfig, lr: float):
    import tensorflow as tf

    if cfg.data.task == "multiclass":
        loss, metrics = "sparse_categorical_crossentropy", ["accuracy"]
    else:
        loss = "binary_crossentropy"
        metrics = [tf.keras.metrics.AUC(multi_label=True, num_labels=len(cfg.data.classes), name="auc")]
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss, metrics=metrics)


def fit(cfg: ExperimentConfig, train: Records, val: Records | None, early_stopping: bool = True, verbose: int = 2):
    """Two-phase fit (head, then top of backbone). Returns (model, history dict)."""
    import tensorflow as tf

    from src.models.factory import build_model, unfreeze_top

    if len(train) == 0:
        raise ValueError("no training images selected for this config")
    size, enh = cfg.data.img_size, cfg.data.enhance
    x_tr = load_images(train.paths, size, enh)
    y_tr = train.targets.astype("float32") if cfg.data.task == "multilabel" else train.targets
    ds_tr = make_dataset(x_tr, y_tr, cfg.train.batch_size, True, cfg.augment, cfg.seed)
    ds_val = None
    if val is not None and len(val):
        y_val = val.targets.astype("float32") if cfg.data.task == "multilabel" else val.targets
        ds_val = make_dataset(load_images(val.paths, size, enh), y_val, cfg.train.batch_size, False)

    model, base = build_model(cfg.model.arch, size, len(cfg.data.classes), cfg.data.task,
                              cfg.model.pretrained, cfg.model.dropout)
    callbacks = []
    if early_stopping and ds_val is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.train.early_stopping_patience, restore_best_weights=True))
    cw = _class_weight(cfg, train.targets)

    _compile(model, cfg, cfg.train.lr)
    h1 = model.fit(ds_tr, validation_data=ds_val, epochs=cfg.train.epochs, class_weight=cw,
                   callbacks=callbacks, verbose=verbose)
    history = {k: [float(v) for v in vals] for k, vals in h1.history.items()}

    if base is not None and cfg.train.fine_tune_epochs > 0:
        unfreeze_top(base, cfg.model.fine_tune_layers)
        _compile(model, cfg, cfg.train.fine_tune_lr)
        start = len(history.get("loss", []))
        h2 = model.fit(ds_tr, validation_data=ds_val, initial_epoch=start,
                       epochs=start + cfg.train.fine_tune_epochs, class_weight=cw,
                       callbacks=callbacks, verbose=verbose)
        for k, vals in h2.history.items():
            history.setdefault(k, []).extend(float(v) for v in vals)
    history["phase2_start_epoch"] = len(h1.history.get("loss", []))
    return model, history


def predict(model, paths: list[str], cfg: ExperimentConfig) -> np.ndarray:
    if not paths:
        return np.zeros((0, len(cfg.data.classes)), np.float32)
    x = load_images(paths, cfg.data.img_size, cfg.data.enhance).astype("float32")
    return model.predict(x, batch_size=cfg.train.batch_size, verbose=0)


def _write_predictions(path: Path, rec: Records, probs: np.ndarray, cfg: ExperimentConfig) -> None:
    classes = cfg.data.classes
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "group_id", "true", "pred", *[f"p_{c}" for c in classes]])
        for i, p in enumerate(rec.paths):
            if cfg.data.task == "multiclass":
                true, pred = classes[rec.targets[i]], classes[int(probs[i].argmax())]
            else:
                true = ";".join(c for c, t in zip(classes, rec.targets[i]) if t) or cfg.data.negative_label
                thr = cfg.train.threshold
                pred = ";".join(c for c, q in zip(classes, probs[i]) if q >= thr) or cfg.data.negative_label
            w.writerow([p, rec.groups[i], true, pred, *[f"{q:.6f}" for q in probs[i]]])


def _metrics(cfg: ExperimentConfig, rec: Records, probs: np.ndarray) -> dict | None:
    if len(rec) == 0:
        return None
    return compute_metrics(cfg.data.task, rec.targets, probs, cfg.data.classes,
                           cfg.train.threshold, cfg.bootstrap, cfg.seed)


def load_split_records(cfg: ExperimentConfig) -> dict[str, Records]:
    rows = read_manifest(cfg.data.manifest)
    out = {s: select_records(rows, cfg.data.classes, cfg.data.task, cfg.data.negative_label, (s,))
           for s in ("train", "val", "test")}
    if cfg.synthetic.dir and cfg.synthetic.per_class > 0:
        syn = synthetic_records(cfg.synthetic.dir, cfg.data.classes, cfg.data.task,
                                cfg.synthetic.per_class, cfg.seed)
        out["train"] = out["train"].concat(syn)
        out["n_synthetic"] = len(syn)
    return out


def _count(rec: Records, cfg: ExperimentConfig) -> dict:
    if cfg.data.task == "multiclass":
        c = np.bincount(rec.targets, minlength=len(cfg.data.classes)) if len(rec) else [0] * len(cfg.data.classes)
        return dict(zip(cfg.data.classes, map(int, c)))
    pos = rec.targets.sum(0) if len(rec) else [0] * len(cfg.data.classes)
    d = dict(zip(cfg.data.classes, map(int, pos)))
    d[f"{cfg.data.negative_label} (all-negative)"] = int((rec.targets.sum(1) == 0).sum()) if len(rec) else 0
    return d


def make_run_dir(cfg: ExperimentConfig, run_dir: str | None = None) -> Path:
    path = Path(run_dir) if run_dir else Path(cfg.output_dir) / f"{cfg.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model_bundle(model, cfg: ExperimentConfig, run_dir: Path) -> None:
    model.save(run_dir / "model.keras")
    (run_dir / "inference.json").write_text(json.dumps({
        "classes": cfg.data.classes, "task": cfg.data.task, "negative_label": cfg.data.negative_label,
        "img_size": cfg.data.img_size, "enhance": cfg.data.enhance, "threshold": cfg.train.threshold,
        "arch": cfg.model.arch,
    }, indent=2))


def run(cfg: ExperimentConfig, run_dir: str | None = None, verbose: int = 2) -> dict:
    set_seed(cfg.seed, cfg.deterministic)
    out_dir = make_run_dir(cfg, run_dir)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    data = load_split_records(cfg)
    meta = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": git_state(), "env": environment_info(),
        "manifest_sha256": file_sha256(cfg.data.manifest),
        "counts": {s: _count(data[s], cfg) for s in ("train", "val", "test")},
        "n_synthetic_train": data.get("n_synthetic", 0),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"[{cfg.name}] run dir: {out_dir}\n  counts: {meta['counts']}")

    results: dict = {"name": cfg.name, "run_dir": out_dir.as_posix(), "counts": meta["counts"]}
    if cfg.cv_folds >= 2:
        results["cv"] = cross_validate(cfg, data["train"].concat(data["val"]), verbose)
        model, history = fit(cfg, data["train"].concat(data["val"]), None, early_stopping=False, verbose=verbose)
        evaluated = {"test": data["test"]}
    else:
        model, history = fit(cfg, data["train"], data["val"], verbose=verbose)
        evaluated = {"val": data["val"], "test": data["test"]}
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    for split, rec in evaluated.items():
        probs = predict(model, rec.paths, cfg)
        if len(rec):
            _write_predictions(out_dir / f"predictions_{split}.csv", rec, probs, cfg)
        m = _metrics(cfg, rec, probs)
        results[split] = m
        if verbose and m:
            print(f"  {split}: {format_headline(m)}")
    save_model_bundle(model, cfg, out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


def cross_validate(cfg: ExperimentConfig, pool: Records, verbose: int = 2) -> dict:
    from sklearn.model_selection import StratifiedGroupKFold

    strata = pool.strata()
    real = np.array([not g.startswith("syn:") for g in pool.groups])
    real_idx = np.flatnonzero(real)
    syn = pool.subset(~real)  # synthetic images only ever join training folds
    sgkf = StratifiedGroupKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    k = len(cfg.data.classes)
    oof = np.full((len(real_idx), k), np.nan, np.float32)
    folds = []
    real_pool = pool.subset(real)
    for f, (tr, va) in enumerate(sgkf.split(np.zeros(len(real_idx)), strata[real_idx],
                                            [pool.groups[i] for i in real_idx])):
        set_seed(cfg.seed + f, cfg.deterministic)
        tr_rec, va_rec = real_pool.subset(np.isin(np.arange(len(real_pool)), tr)), real_pool.subset(
            np.isin(np.arange(len(real_pool)), va))
        model, _ = fit(cfg, tr_rec.concat(syn) if len(syn) else tr_rec, None, early_stopping=False, verbose=0)
        oof[va] = predict(model, va_rec.paths, cfg)
        fm = _metrics(cfg, va_rec, oof[va])
        folds.append(headline(fm))
        if verbose:
            print(f"  fold {f + 1}/{cfg.cv_folds}: {format_headline(fm)}")
    done = ~np.isnan(oof).any(1)
    oof_metrics = _metrics(cfg, real_pool.subset(done), oof[done])
    summary = {}
    for key in folds[0]:
        if key == "n":
            continue
        vals = [fd[key]["value"] for fd in folds if fd[key]["value"] is not None]
        summary[key] = {"mean": float(np.mean(vals)) if vals else None,
                        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None, "n_folds": len(vals)}
    if verbose:
        print(f"  out-of-fold: {format_headline(oof_metrics)}")
    return {"folds": folds, "fold_summary": summary, "out_of_fold": oof_metrics}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    ap.add_argument("--run-dir", help="write outputs here instead of outputs/runs/<name>-<time>")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    run(cfg, args.run_dir, verbose=0 if args.quiet else 2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
