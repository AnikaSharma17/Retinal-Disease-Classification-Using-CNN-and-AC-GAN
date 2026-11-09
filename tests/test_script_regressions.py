"""Static regression tests for the legacy training/AC-GAN scripts (legacy/).

The scripts execute training at import time and need TensorFlow, so these checks parse them
with `ast` instead of importing them. They pin the Stage-1 fixes (ML-2, BUG-1..4, ENG-1).
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPTS = sorted((ROOT / "legacy" / "classifiers").rglob("train_*.py"))
ACGAN = ROOT / "legacy" / "acgan"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _assignments(tree: ast.Module, name: str) -> list[ast.expr]:
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]


def test_found_all_training_scripts():
    assert len(TRAIN_SCRIPTS) == 7


@pytest.mark.parametrize("script", TRAIN_SCRIPTS, ids=lambda p: p.name)
def test_num_classes_derived_from_target_classes(script):
    # ML-2: C3 scripts had num_classes = 5 with three target classes.
    (value,) = _assignments(_tree(script), "num_classes")
    assert ast.unparse(value) == "len(target_classes)"


@pytest.mark.parametrize("script", TRAIN_SCRIPTS, ids=lambda p: p.name)
def test_uses_shared_enhance_image(script):
    # ML-1: no local (buggy) copy; the fixed implementation is imported.
    tree = _tree(script)
    assert not any(isinstance(n, ast.FunctionDef) and n.name == "enhance_image" for n in ast.walk(tree))
    assert any(isinstance(n, ast.ImportFrom) and n.module == "src.preprocessing"
               and any(a.name == "enhance_image" for a in n.names) for n in ast.walk(tree))


ALL_PY = sorted([*(ROOT / "src").rglob("*.py"), *(ROOT / "legacy").rglob("*.py")])


@pytest.mark.parametrize("script", ALL_PY, ids=lambda p: p.name)
def test_no_hardcoded_user_paths(script):
    # ENG-1
    assert "C:\\Users\\" not in script.read_text(encoding="utf-8")


def test_acgan_adam_uses_learning_rate():
    # BUG-3: Adam(lr=...) is rejected by Keras 3.
    for call in ast.walk(_tree(ACGAN / "train_acgan.py")):
        if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "Adam":
            assert "lr" not in {k.arg for k in call.keywords}


def test_acgan_fake_samples_pass_n_classes():
    # BUG-4: fake labels must be drawn from the same number of classes as the models.
    funcs = {n.name: n for n in ast.walk(_tree(ACGAN / "train_acgan.py")) if isinstance(n, ast.FunctionDef)}
    fake = funcs["generate_fake_samples"]
    assert "n_classes" in [a.arg for a in fake.args.args]
    inner = [c for c in ast.walk(fake) if isinstance(c, ast.Call)
             and getattr(c.func, "id", None) == "generate_latent_points"]
    assert inner and len(inner[0].args) == 3
    for name in ("define_discriminator", "define_generator", "generate_latent_points"):
        defaults = funcs[name].args.defaults
        assert ast.unparse(defaults[-1]) == "N_CLASSES", name


def test_acgan_eval_does_not_add_channel_axis():
    # BUG-2: expand_dims on RGB arrays produced 5-D input.
    tree = _tree(ACGAN / "evaluate_acgan.py")
    calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "expand_dims"]
    assert calls == []


def test_acgan_eval_image_size_matches_discriminator():
    # BUG-2: images were loaded at 128x128 for a 64x64 discriminator.
    train_src = (ACGAN / "train_acgan.py").read_text(encoding="utf-8")
    eval_tree = _tree(ACGAN / "evaluate_acgan.py")
    assert "def define_discriminator(in_shape=(64, 64, 3)" in train_src
    sizes = {ast.unparse(k.value) for c in ast.walk(eval_tree) if isinstance(c, ast.Call)
             for k in c.keywords if k.arg == "target_size"}
    assert sizes == {"(64, 64)"}


def test_evaluate_cli_loads_its_own_model():
    # BUG-1: the old binary eval script called model.evaluate on an undefined `model`.
    tree = _tree(ROOT / "legacy" / "classifiers" / "evaluate.py")
    assert any(isinstance(c, ast.Call) and getattr(c.func, "id", None) == "load_model" for c in ast.walk(tree))
    assert not (ROOT / "legacy" / "classifiers" / "binary" / "evaluate_binary.py").exists()
