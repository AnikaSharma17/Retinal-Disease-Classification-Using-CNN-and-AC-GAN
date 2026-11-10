from pathlib import Path

import numpy as np
import pytest

from src.config import config_from_dict, load_config
from src.evaluation.metrics import (
    bootstrap_ci,
    expected_calibration_error,
    format_headline,
    headline,
    multiclass_metrics,
    multilabel_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = {"data": {"classes": ["A", "B"], "task": "multiclass"}}


# ------------------------------------------------------------------ config
@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_experiment_configs_load(path):
    import yaml

    raw = yaml.safe_load(path.read_text())
    if "variants" in raw:  # ablation spec: every variant must produce a valid config
        for overrides in raw["variants"].values():
            load_config(ROOT / raw["base"], overrides)
    elif "data" in raw:
        assert load_config(path).data.classes
    # acgan.yaml is checked in test_tf_pipeline (needs TF-free loader only)


def test_overrides_are_typed():
    cfg = config_from_dict(BASE, ["train.epochs=3", "augment.enabled=false", "data.classes=[X, Y, Z]"])
    assert cfg.train.epochs == 3 and cfg.augment.enabled is False and cfg.data.classes == ["X", "Y", "Z"]


@pytest.mark.parametrize("bad", [
    {"data": {"classes": ["A", "B"], "task": "regression"}},
    {"data": {"classes": ["A"], "task": "multiclass"}},
    {"data": {"classes": ["A", "A"]}},
    {"data": {"classes": ["A", "Control"], "task": "multilabel"}},
    {"data": {"classes": ["A", "B"]}, "model": {"arch": "vgg99"}},
    {"data": {"classes": ["A", "B"]}, "synthetic": {"per_class": 5}},
    {"data": {"classes": ["A", "B"]}, "cv_folds": 1},
])
def test_invalid_configs_rejected(bad):
    with pytest.raises(ValueError):
        config_from_dict(bad)


def test_unknown_keys_rejected():
    with pytest.raises(KeyError):
        config_from_dict({**BASE, "trian": {"epochs": 1}})
    with pytest.raises(KeyError):
        config_from_dict(BASE, ["train.epoch=1"])


# ------------------------------------------------------------------ metrics
def test_multiclass_perfect_and_confusion():
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.eye(3)[y] * 0.9 + 0.1 / 3
    m = multiclass_metrics(y, p, ["a", "b", "c"], n_boot=200)
    assert m["accuracy"] == 1.0 and m["auc_macro_ovr"] == 1.0
    assert m["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert m["per_class"]["a"]["sensitivity"] == 1.0 and m["per_class"]["a"]["specificity"] == 1.0


def test_multiclass_auc_undefined_when_class_absent():
    y = np.array([0, 0, 1, 1])
    p = np.full((4, 3), 1 / 3)
    m = multiclass_metrics(y, p, ["a", "b", "c"], n_boot=50)
    assert m["auc_macro_ovr"] is None
    assert m["per_class"]["c"]["sensitivity"] is None and m["per_class"]["c"]["auc_ovr"] is None


def test_multiclass_balanced_accuracy_exposes_majority_guessing():
    y = np.array([0] * 9 + [1])
    p = np.tile([0.9, 0.1], (10, 1))
    m = multiclass_metrics(y, p, ["maj", "min"], n_boot=0)
    assert m["accuracy"] == 0.9 and m["balanced_accuracy"] == 0.5 and m["auc_macro_ovr"] == 0.5


def test_multilabel_metrics():
    Y = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    P = np.array([[0.9, 0.2], [0.8, 0.7], [0.1, 0.6], [0.2, 0.1]])
    m = multilabel_metrics(Y, P, ["x", "y"], n_boot=200)
    assert m["auc_macro"] == 1.0 and m["subset_accuracy"] == 1.0 and m["hamming_loss"] == 0.0
    assert m["per_label"]["x"]["positives"] == 2


def test_multilabel_label_without_positives_is_undefined():
    Y = np.array([[1, 0], [0, 0], [1, 0]])
    P = np.array([[0.9, 0.2], [0.1, 0.1], [0.7, 0.3]])
    m = multilabel_metrics(Y, P, ["x", "y"], n_boot=0)
    assert m["per_label"]["y"]["auc"] is None
    assert m["auc_macro"] == 1.0  # averaged over labels where AUC is defined


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        multiclass_metrics([0, 1], np.zeros((2, 3)), ["a", "b"])
    with pytest.raises(ValueError):
        multilabel_metrics(np.zeros((2, 2)), np.zeros((2, 3)), ["a", "b"])


def test_ece():
    assert expected_calibration_error(np.array([1.0, 1.0]), np.array([1, 1])) == 0.0
    assert expected_calibration_error(np.array([0.9] * 10), np.array([1] * 5 + [0] * 5)) == pytest.approx(0.4)


def test_bootstrap_ci_contains_estimate_and_narrows_with_n():
    rng = np.random.default_rng(0)

    def acc(y, p):
        return float((p.argmax(1) == y).mean())

    def ci_width(n):
        y = rng.integers(0, 2, n)
        p = np.eye(2)[np.where(rng.random(n) < 0.8, y, 1 - y)]
        lo, hi = bootstrap_ci(acc, y, p, n=500)
        assert lo <= acc(y, p) <= hi
        return hi - lo

    assert ci_width(400) < ci_width(20)


def test_headline_formatting_handles_missing():
    y = np.array([0, 0])
    m = multiclass_metrics(y, np.array([[0.6, 0.4], [0.7, 0.3]]), ["a", "b"], n_boot=0)
    h = headline(m)
    assert h["auc_macro_ovr"]["value"] is None
    assert "auc_macro_ovr=n/a" in format_headline(m)


def test_no_ci_below_minimum_n():
    # three images gave AUC 1.00 with a "CI" of [1.00, 1.00]; such CIs are now withheld
    y = np.array([0, 0, 1])
    p = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]])
    m = multiclass_metrics(y, p, ["a", "b"], n_boot=500)
    assert m["auc_macro_ovr"] == 1.0 and m["ci95"]["auc_macro_ovr"] is None
    assert "auc_macro_ovr=1.000 (no CI)" in format_headline(m)


def test_report_builds_from_run_artifacts(tmp_path):
    import json

    from src.experiments.report import build_report

    y = np.array([0, 1] * 6)
    p = np.eye(2)[y] * 0.8 + 0.1
    m = multiclass_metrics(y, p, ["a", "b"], n_boot=100)
    run = tmp_path / "runs" / "toy"
    run.mkdir(parents=True)
    counts = {"train": {"a": 1}, "val": {"a": 1}, "test": {"a": 6, "b": 6}}
    (run / "metrics.json").write_text(json.dumps({"test": m, "counts": counts}))
    (run / "gradcam_summary_test.json").write_text(
        json.dumps({"split": "test", "n": 2, "mean_outside_fundus_fraction": 0.1}))
    text = build_report(tmp_path)
    assert "| toy | 12 | 1.00 [" in text and "mean 0.10" in text
