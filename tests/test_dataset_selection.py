import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from src.data.dataset import Records, select_records, synthetic_records

ROOT = Path(__file__).resolve().parents[1]

ROWS = [
    {"path": "a.jpg", "labels": "DM", "group_id": "g1", "split": "train"},
    {"path": "b.jpg", "labels": "Hypertension", "group_id": "g2", "split": "train"},
    {"path": "c.jpg", "labels": "DM;Hypertension", "group_id": "g3", "split": "val"},
    {"path": "d.jpg", "labels": "Control", "group_id": "g4", "split": "test"},
    {"path": "e.jpg", "labels": "Glaucoma", "group_id": "g5", "split": "train"},
    {"path": "f.jpg", "labels": "Control;DM", "group_id": "g6", "split": "train"},
]


def test_multiclass_excludes_multilabel_and_out_of_scope():
    r = select_records(ROWS, ["DM", "Hypertension"], "multiclass")
    assert r.paths == ["a.jpg", "b.jpg"] and r.targets.tolist() == [0, 1]


def test_multilabel_multi_hot_with_negative_controls():
    r = select_records(ROWS, ["DM", "Hypertension"], "multilabel")
    assert r.paths == ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    assert r.targets.tolist() == [[1, 0], [0, 1], [1, 1], [0, 0]]


def test_split_filter_and_empty_shapes():
    assert select_records(ROWS, ["DM", "Hypertension"], "multiclass", splits=("test",)).targets.shape == (0,)
    assert select_records(ROWS, ["DM", "Hypertension"], "multilabel", splits=("nope",)).targets.shape == (0, 2)


def test_records_helpers():
    r = select_records(ROWS, ["DM", "Hypertension"], "multilabel")
    sub = r.subset([True, False, True, False])
    assert sub.paths == ["a.jpg", "c.jpg"]
    both = sub.concat(sub)
    assert len(both) == 4 and both.strata().tolist() == ["10", "11", "10", "11"]


def test_synthetic_records_capped_and_train_only(tmp_path):
    for c, n in (("DM", 5), ("Hypertension", 2)):
        (tmp_path / c).mkdir()
        for i in range(n):
            (tmp_path / c / f"syn_{i}.png").write_bytes(b"")
    r = synthetic_records(tmp_path, ["DM", "Hypertension"], "multiclass", per_class=3, seed=0)
    assert np.bincount(r.targets).tolist() == [3, 2]
    assert set(r.splits) == {"train"} and all(g.startswith("syn:") for g in r.groups)


def test_committed_manifest_has_no_group_across_splits():
    """CI data gate on the committed split (DATA-1 regression)."""
    path = ROOT / "data" / "manifest_split.csv"
    if not path.exists():
        pytest.skip("no committed manifest")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    splits = defaultdict(set)
    for r in rows:
        splits[r["group_id"]].add(r["split"])
    assert all(len(s) == 1 for s in splits.values())
    assert len({r["sha256"] for r in rows}) == len(rows)


def test_records_dataclass_len():
    assert len(Records([], np.zeros((0,), int), [], [])) == 0
