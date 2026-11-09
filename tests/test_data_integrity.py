import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.integrity import check_records, is_augmented_name, scan_tree
from src.data.integrity import main as integrity_main
from src.data.manifest import build_manifest, materialize, split_manifest


def _img(path: Path, seed: int, quality: int = 95) -> Path:
    rng = np.random.default_rng(seed)
    # smooth random field so a perceptual hash is stable under re-encoding
    small = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(small).resize((128, 128), Image.BICUBIC).save(path, quality=quality)
    return path


@pytest.mark.parametrize(
    "name, expected",
    [("image_0_1062.jpeg", True), ("image_3_17.png", True), ("_0_5521.png", True),
     ("15.jpg", False), ("image_final.jpeg", False), ("patient_12.jpg", False)],
)
def test_is_augmented_name(name, expected):
    assert is_augmented_name(name) is expected


def test_detects_cross_split_and_cross_label_duplicates(tmp_path):
    a = _img(tmp_path / "train" / "DM" / "1.jpg", seed=1)
    for dup in (tmp_path / "test" / "DM" / "1.jpg", tmp_path / "train" / "Hypertension" / "x.jpg"):
        dup.parent.mkdir(parents=True)
        shutil.copy(a, dup)
    _img(tmp_path / "val" / "Control" / "2.jpg", seed=2)
    _img(tmp_path / "train" / "Control" / "image_0_99.jpeg", seed=3)

    report = check_records(scan_tree(tmp_path))
    assert report.n_files == 5 and report.n_unique == 3
    assert len(report.cross_split) == 1 and len(report.cross_split[0]["paths"]) == 3
    assert len(report.cross_label) == 1
    assert report.counts["train"]["Control"] == {"total": 1, "augmented_named": 1}
    assert report.has_blocking_issues
    assert integrity_main([str(tmp_path)]) == 1


def test_clean_tree_passes(tmp_path):
    for i, split in enumerate(["train", "val", "test"]):
        _img(tmp_path / split / "DM" / f"{i}.jpg", seed=10 + i)
    assert integrity_main([str(tmp_path)]) == 0


def test_manifest_merges_labels_and_excludes_augmented(tmp_path):
    a = _img(tmp_path / "DM" / "1.jpg", seed=1)
    (tmp_path / "Heart Disease").mkdir()
    shutil.copy(a, tmp_path / "Heart Disease" / "7.jpg")
    _img(tmp_path / "DM" / "image_0_5.jpeg", seed=9)
    _img(tmp_path / "Control" / "3.jpg", seed=3)

    rows = build_manifest(tmp_path)
    assert len(rows) == 2
    labels = sorted(r["labels"] for r in rows)
    assert labels == ["Control", "DM;Heart Disease"]
    assert len(build_manifest(tmp_path, include_augmented=True)) == 3


def test_reencoded_copy_is_grouped_as_near_duplicate(tmp_path):
    _img(tmp_path / "DM" / "1.jpg", seed=1, quality=95)
    _img(tmp_path / "DM" / "1_reencoded.jpg", seed=1, quality=60)  # different bytes, same picture
    _img(tmp_path / "DM" / "2.jpg", seed=2)
    rows = build_manifest(tmp_path)
    assert len({r["sha256"] for r in rows}) == 3
    by_name = {Path(r["path"]).name: r["group_id"] for r in rows}
    assert by_name["1.jpg"] == by_name["1_reencoded.jpg"] != by_name["2.jpg"]


def test_patient_ids_group_images(tmp_path):
    _img(tmp_path / "DM" / "left.jpg", seed=1)
    _img(tmp_path / "DM" / "right.jpg", seed=2)
    rows = build_manifest(tmp_path, patients={"left.jpg": "P1", "right.jpg": "P1"})
    assert len({r["group_id"] for r in rows}) == 1


def _synthetic_rows(n_per_label=20):
    rows, k = [], 0
    for label in ["Control", "DM", "Hypertension", "DM;Hypertension"]:
        for _ in range(n_per_label):
            gid = f"g{k // 2:05d}"  # pairs share a group (e.g. two eyes of one patient)
            rows.append({"group_id": gid, "sha256": f"{k:064x}", "path": f"{k}.jpg",
                         "labels": label, "n_labels": label.count(";") + 1})
            k += 1
    return rows


def test_split_never_puts_a_group_in_two_splits():
    out = split_manifest(_synthetic_rows(), (0.7, 0.15, 0.15), seed=1)
    splits_per_group = defaultdict(set)
    for r in out:
        splits_per_group[r["group_id"]].add(r["split"])
    assert all(len(s) == 1 for s in splits_per_group.values())


def test_split_is_stratified_and_deterministic():
    a = split_manifest(_synthetic_rows(), seed=7)
    b = split_manifest(_synthetic_rows(), seed=7)
    assert [r["split"] for r in a] == [r["split"] for r in b]
    for label in ["Control", "DM", "Hypertension", "DM;Hypertension"]:
        assert {r["split"] for r in a if r["labels"] == label} == {"train", "val", "test"}


def test_split_rejects_bad_ratios():
    with pytest.raises(ValueError):
        split_manifest(_synthetic_rows(), (0.8, 0.15, 0.15))


def test_materialize_skips_multilabel(tmp_path):
    src = tmp_path / "src"
    a = _img(src / "DM" / "1.jpg", seed=1)
    (src / "Hypertension").mkdir()
    shutil.copy(a, src / "Hypertension" / "1.jpg")
    _img(src / "Control" / "2.jpg", seed=2)
    rows = split_manifest(build_manifest(src), seed=0)
    copied, skipped = materialize(rows, tmp_path / "out")
    assert (copied, skipped) == (1, 1)
    assert len(list((tmp_path / "out").rglob("*.jpg"))) == 1
