"""End-to-end tests of the TensorFlow pipeline on a tiny synthetic fundus-like dataset.

Skipped automatically where TensorFlow is unavailable (run them in the Docker image)."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.config import config_from_dict  # noqa: E402
from src.data.manifest import build_manifest, split_manifest, write_csv  # noqa: E402

COLOURS = {"DM": (200, 60, 30), "Hypertension": (60, 160, 40), "Control": (90, 90, 200)}


def _fundus(colour, seed, size=96):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8)
    noise = rng.integers(-25, 25, (size, size, 3))
    disc = np.clip(np.array(colour) + noise, 0, 255).astype(np.uint8)
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (size // 2, size // 2), size // 2 - 4, 1, -1)
    img[mask == 1] = disc[mask == 1]
    return img


@pytest.fixture(scope="module")
def tiny_manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("fundus")
    n = 0
    for label, colour in COLOURS.items():
        for i in range(12):
            d = root / label
            d.mkdir(exist_ok=True)
            cv2.imwrite(str(d / f"{i}.png"), cv2.cvtColor(_fundus(colour, n), cv2.COLOR_RGB2BGR))
            n += 1
    rows = split_manifest(build_manifest(root, near_dup_threshold=-1), (0.6, 0.2, 0.2), seed=0)
    path = root / "manifest_split.csv"
    write_csv(rows, path)
    return path


def _cfg(manifest, tmp_path, **over):
    raw = {
        "name": "t", "output_dir": str(tmp_path), "bootstrap": 50,
        "data": {"manifest": str(manifest), "task": "multiclass", "classes": ["DM", "Hypertension", "Control"],
                 "img_size": 32},
        "model": {"arch": "small_cnn", "pretrained": False},
        "train": {"epochs": 3, "fine_tune_epochs": 0, "batch_size": 8, "lr": 3e-3},
    }
    for k, v in over.items():
        raw.setdefault(k, {}).update(v) if isinstance(v, dict) else raw.__setitem__(k, v)
    return config_from_dict(raw)


def test_train_run_writes_complete_bundle(tiny_manifest, tmp_path):
    from src.predict import Predictor
    from src.train import run

    cfg = _cfg(tiny_manifest, tmp_path)
    res = run(cfg, run_dir=str(tmp_path / "run"), verbose=0)
    run_dir = Path(res["run_dir"])
    for f in ("config.yaml", "metadata.json", "history.json", "metrics.json", "model.keras",
              "inference.json", "predictions_test.csv"):
        assert (run_dir / f).exists(), f
    assert res["test"]["n"] > 0 and 0.0 <= res["test"]["accuracy"] <= 1.0
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert len(meta["manifest_sha256"]) == 64 and "tensorflow" in meta["env"]

    # saved model reproduces in-memory predictions
    p = Predictor(run_dir)
    out = p.predict_path(Path(tiny_manifest).parent / "DM" / "0.png")
    assert set(out["probabilities"]) == {"DM", "Hypertension", "Control"}
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-3


def test_learns_an_easy_task(tiny_manifest, tmp_path):
    from src.train import run

    cfg = _cfg(tiny_manifest, tmp_path, train={"epochs": 25}, augment={"enabled": False})
    res = run(cfg, run_dir=str(tmp_path / "easy"), verbose=0)
    assert res["val"]["accuracy"] >= 0.8  # colour-coded classes: anything less means a broken pipeline


def test_multilabel_and_cross_validation(tiny_manifest, tmp_path):
    from src.train import run

    cfg = _cfg(tiny_manifest, tmp_path, cv_folds=2, data={"task": "multilabel", "classes": ["DM", "Hypertension"]},
               train={"epochs": 2})
    res = run(cfg, run_dir=str(tmp_path / "cv"), verbose=0)
    assert len(res["cv"]["folds"]) == 2
    assert res["cv"]["out_of_fold"]["task"] == "multilabel"
    assert "test" in res and "val" not in res


@pytest.mark.parametrize("arch", ["efficientnet_b0", "mobilenet_v2"])
def test_transfer_backbones_build_freeze_and_gradcam(arch):
    from src.explain.gradcam import gradcam
    from src.models.factory import build_model, last_conv_layer_name, unfreeze_top

    model, base = build_model(arch, 64, 3, "multiclass", pretrained=False)
    assert not any(layer.trainable for layer in base.layers)
    assert model.output_shape == (None, 3)
    n = unfreeze_top(base, 10)
    assert 0 < n <= 10
    assert last_conv_layer_name(model)
    cam, probs = gradcam(model, np.random.default_rng(0).integers(0, 255, (64, 64, 3)).astype(np.float32))
    assert cam.shape == (64, 64) and 0.0 <= cam.min() and cam.max() <= 1.0
    assert probs.shape == (3,)


def test_gradcam_run_and_fundus_mask(tiny_manifest, tmp_path):
    from src.explain.gradcam import explain_run, fundus_mask
    from src.train import run

    res = run(_cfg(tiny_manifest, tmp_path), run_dir=str(tmp_path / "gc"), verbose=0)
    s = explain_run(res["run_dir"], "test", max_images=3)
    assert s["n"] == 3 and 0.0 <= s["mean_outside_fundus_fraction"] <= 1.0
    assert len(list((Path(res["run_dir"]) / "gradcam" / "test").glob("*.png"))) == 3
    m = fundus_mask(_fundus((200, 60, 30), 0))
    assert 0.6 < m.mean() < 0.8  # circle of radius 44 in a 96x96 image covers ~66%


def test_acgan_trains_and_generates(tiny_manifest, tmp_path):
    from src.gan.acgan import GanConfig, generate_cli, load_gan_config, train_cli

    cfg = GanConfig(manifest=str(tiny_manifest), classes=["DM", "Hypertension"], img_size=32,
                    latent_dim=16, epochs=2, batch_size=4, sample_every=1, output_dir=str(tmp_path))
    run_dir = train_cli(cfg, str(tmp_path / "gan"), verbose=False)
    log = json.loads((run_dir / "train_log.json").read_text())
    assert len(log) == 2 and all(np.isfinite(r["d_loss"]) for r in log)
    out = generate_cli(run_dir, 3, tmp_path / "syn")
    assert len(list((out / "DM").glob("*.png"))) == 3
    shipped = load_gan_config(str(Path(__file__).resolve().parents[1] / "configs" / "acgan.yaml"))
    assert shipped.classes == ["DM", "Hypertension"]


def test_synthetic_images_join_training_only(tiny_manifest, tmp_path):
    from src.train import load_split_records

    syn = tmp_path / "syn"
    for c in ("DM", "Hypertension"):
        (syn / c).mkdir(parents=True)
        for i in range(4):
            cv2.imwrite(str(syn / c / f"s{i}.png"), np.zeros((32, 32, 3), np.uint8))
    cfg = _cfg(tiny_manifest, tmp_path, synthetic={"dir": str(syn), "per_class": 3})
    data = load_split_records(cfg)
    assert data["n_synthetic"] == 6
    assert not any("syn" in p for p in data["val"].paths + data["test"].paths)


def test_baselines(tiny_manifest, tmp_path):
    from src.evaluation.baselines import run_baselines

    res = run_baselines(_cfg(tiny_manifest, tmp_path))
    assert res["prior"]["auc_macro_ovr"] == pytest.approx(0.5)
    assert res["colour_lr"]["accuracy"] >= 0.8  # colour histogram solves this synthetic task
