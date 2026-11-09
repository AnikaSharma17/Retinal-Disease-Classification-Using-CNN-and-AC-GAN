"""Classification metrics with bootstrap confidence intervals (TensorFlow-free).

Small medical datasets make point estimates misleading, so every headline metric is
reported with a percentile bootstrap CI, and metrics that are undefined for the given
labels (e.g. AUC when a class has no positives) are reported as None, never guessed.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def _safe(fn: Callable[..., float], *args, **kwargs) -> float | None:
    """Call a sklearn metric; return None when it is undefined for these labels."""
    try:
        with warnings.catch_warnings():  # degenerate resamples are expected; result is None/NaN-checked
            warnings.simplefilter("ignore")
            v = float(fn(*args, **kwargs))
    except ValueError:
        return None
    return None if np.isnan(v) else v


MIN_N_FOR_CI = 10


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """ECE: |accuracy - confidence| averaged over equal-width confidence bins, weighted by bin size."""
    confidence = np.asarray(confidence, float)
    correct = np.asarray(correct, float)
    if confidence.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(confidence, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - confidence[m].mean())
    return float(ece)


def bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float | None],
    y_true: np.ndarray,
    probs: np.ndarray,
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Percentile bootstrap CI; resamples where the metric is undefined are skipped.

    Returns None below MIN_N_FOR_CI samples: with a handful of points the resampling
    distribution is degenerate (e.g. AUC [1.00, 1.00] from three images) and would look
    more certain than the point estimate deserves.
    """
    if n <= 0 or len(y_true) < MIN_N_FOR_CI:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        ix = rng.integers(0, len(y_true), len(y_true))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = metric(y_true[ix], probs[ix])
        if v is not None and not np.isnan(v):
            vals.append(v)
    if len(vals) < max(20, n // 10):
        return None
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ---------------------------------------------------------------- multiclass
def _mc_auc(y, p):
    k = p.shape[1]
    if len(np.unique(y)) < k:
        return None
    if k == 2:
        return _safe(roc_auc_score, y, p[:, 1])
    return _safe(roc_auc_score, y, p, multi_class="ovr", average="macro", labels=list(range(k)))


def _mc_bal_acc(y, p):
    return _safe(balanced_accuracy_score, y, p.argmax(1))


def _mc_macro_f1(y, p):
    return _safe(f1_score, y, p.argmax(1), average="macro", labels=list(range(p.shape[1])), zero_division=0)


def _mc_acc(y, p):
    return float((p.argmax(1) == y).mean())


def multiclass_metrics(y_true, probs, classes: list[str], n_boot: int = 1000, seed: int = 0) -> dict:
    y_true = np.asarray(y_true, int)
    probs = np.asarray(probs, float)
    k = len(classes)
    if probs.shape != (len(y_true), k):
        raise ValueError(f"probs shape {probs.shape} != ({len(y_true)}, {k})")
    y_pred = probs.argmax(1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(k)))
    per_class = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        positives = int(cm[i].sum())
        per_class[c] = {
            "support": positives,
            "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "precision": float(tp / (tp + fp)) if tp + fp else None,
            "auc_ovr": _safe(roc_auc_score, (y_true == i).astype(int), probs[:, i])
            if 0 < positives < len(y_true) else None,
        }
    out = {
        "task": "multiclass",
        "n": int(len(y_true)),
        "classes": list(classes),
        "accuracy": _mc_acc(y_true, probs),
        "balanced_accuracy": _mc_bal_acc(y_true, probs),
        "macro_f1": _mc_macro_f1(y_true, probs),
        "auc_macro_ovr": _mc_auc(y_true, probs),
        "ece": expected_calibration_error(probs.max(1), y_pred == y_true),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "ci95": {},
    }
    for name, fn in (("accuracy", _mc_acc), ("balanced_accuracy", _mc_bal_acc),
                     ("macro_f1", _mc_macro_f1), ("auc_macro_ovr", _mc_auc)):
        if out[name] is not None:
            out["ci95"][name] = bootstrap_ci(fn, y_true, probs, n_boot, seed=seed)
    return out


# ---------------------------------------------------------------- multilabel
def _ml_macro_auc(Y, P):
    aucs = [roc_auc_score(Y[:, j], P[:, j]) for j in range(Y.shape[1]) if 0 < Y[:, j].sum() < len(Y)]
    return float(np.mean(aucs)) if aucs else None


def multilabel_metrics(Y_true, probs, classes: list[str], threshold: float = 0.5,
                       n_boot: int = 1000, seed: int = 0) -> dict:
    Y = np.asarray(Y_true, int)
    P = np.asarray(probs, float)
    if Y.shape != P.shape or Y.shape[1] != len(classes):
        raise ValueError(f"shape mismatch: Y {Y.shape}, probs {P.shape}, {len(classes)} classes")
    pred = (P >= threshold).astype(int)
    per_label = {}
    for j, c in enumerate(classes):
        y, p, yhat = Y[:, j], P[:, j], pred[:, j]
        tp = int(((yhat == 1) & (y == 1)).sum())
        fn = int(((yhat == 0) & (y == 1)).sum())
        fp = int(((yhat == 1) & (y == 0)).sum())
        tn = int(((yhat == 0) & (y == 0)).sum())
        defined = 0 < y.sum() < len(y)
        per_label[c] = {
            "positives": int(y.sum()),
            "auc": _safe(roc_auc_score, y, p) if defined else None,
            "average_precision": _safe(average_precision_score, y, p) if y.sum() else None,
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "f1": _safe(f1_score, y, yhat, zero_division=0),
        }
    out = {
        "task": "multilabel",
        "n": int(len(Y)),
        "classes": list(classes),
        "threshold": threshold,
        "auc_macro": _ml_macro_auc(Y, P),
        "macro_f1": _safe(f1_score, Y, pred, average="macro", zero_division=0),
        "subset_accuracy": float((pred == Y).all(1).mean()) if len(Y) else None,
        "hamming_loss": float((pred != Y).mean()) if len(Y) else None,
        "ece": expected_calibration_error(P.ravel(), (pred == Y).ravel()),
        "per_label": per_label,
        "ci95": {},
    }
    if out["auc_macro"] is not None:
        out["ci95"]["auc_macro"] = bootstrap_ci(_ml_macro_auc, Y, P, n_boot, seed=seed)
    return out


def compute_metrics(task: str, y_true, probs, classes, threshold: float = 0.5, n_boot: int = 1000,
                    seed: int = 0) -> dict:
    if task == "multiclass":
        return multiclass_metrics(y_true, probs, classes, n_boot, seed)
    return multilabel_metrics(y_true, probs, classes, threshold, n_boot, seed)


def headline(m: dict) -> dict:
    """The few numbers worth putting in a results table, with CIs."""
    keys = ("accuracy", "balanced_accuracy", "macro_f1", "auc_macro_ovr") if m["task"] == "multiclass" \
        else ("auc_macro", "macro_f1", "subset_accuracy")
    return {k: {"value": m.get(k), "ci95": m["ci95"].get(k)} for k in keys} | {"n": m["n"]}


def format_headline(m: dict) -> str:
    parts = [f"n={m['n']}"]
    for k, v in headline(m).items():
        if k == "n":
            continue
        if v["value"] is None:
            parts.append(f"{k}=n/a")
        elif v["ci95"]:
            parts.append(f"{k}={v['value']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
        else:
            parts.append(f"{k}={v['value']:.3f} (no CI)")
    return "  ".join(parts)
