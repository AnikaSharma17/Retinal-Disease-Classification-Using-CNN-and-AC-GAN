"""Turn the split manifest into arrays/tf.data pipelines for a given experiment.

Selection rules (see DataConfig):
* multiclass: keep images with exactly one label and that label in `classes`. Images with
  several labels (comorbidity or mislabelling) or out-of-scope labels are excluded, never
  silently assigned to one of their labels.
* multilabel: target is multi-hot over `classes`; images labelled only `negative_label`
  (healthy controls) are all-zero; images with any other out-of-scope label are excluded.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from src.preprocessing import enhance_image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Records:
    paths: list[str]
    targets: np.ndarray  # (N,) int for multiclass, (N, K) {0,1} for multilabel
    groups: list[str]
    splits: list[str]

    def __len__(self) -> int:
        return len(self.paths)

    def subset(self, mask) -> Records:
        idx = np.flatnonzero(np.asarray(mask))
        return Records([self.paths[i] for i in idx], self.targets[idx],
                       [self.groups[i] for i in idx], [self.splits[i] for i in idx])

    def concat(self, other: Records) -> Records:
        return Records(self.paths + other.paths, np.concatenate([self.targets, other.targets]),
                       self.groups + other.groups, self.splits + other.splits)

    def strata(self) -> np.ndarray:
        """One label per record, for stratification (label-set string for multilabel)."""
        if self.targets.ndim == 1:
            return self.targets
        return np.array(["".join(map(str, row)) for row in self.targets])


def read_manifest(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows and "split" not in rows[0]:
        raise ValueError(f"{path} has no 'split' column; run `python -m src.data.manifest split` first")
    return rows


def select_records(rows: list[dict], classes: list[str], task: str, negative_label: str = "Control",
                   splits: tuple[str, ...] | None = None) -> Records:
    paths, targets, groups, row_splits = [], [], [], []
    index = {c: i for i, c in enumerate(classes)}
    for r in rows:
        if splits is not None and r["split"] not in splits:
            continue
        labels = [lab for lab in r["labels"].split(";") if lab]
        if task == "multiclass":
            if len(labels) != 1 or labels[0] not in index:
                continue
            target = index[labels[0]]
        elif task == "multilabel":
            if any(lab not in index and lab != negative_label for lab in labels):
                continue
            if negative_label in labels and len(labels) > 1:
                continue  # "healthy" plus a disease label is contradictory
            target = np.zeros(len(classes), np.int64)
            for lab in labels:
                if lab in index:
                    target[index[lab]] = 1
        else:
            raise ValueError(f"unknown task {task!r}")
        paths.append(r["path"])
        targets.append(target)
        groups.append(r["group_id"])
        row_splits.append(r["split"])
    shape = (0,) if task == "multiclass" else (0, len(classes))
    arr = np.array(targets, np.int64) if targets else np.zeros(shape, np.int64)
    return Records(paths, arr, groups, row_splits)


def synthetic_records(syn_dir: str | Path, classes: list[str], task: str, per_class: int, seed: int) -> Records:
    """Up to `per_class` generated images per class from <syn_dir>/<class>/, always split='train'."""
    rng = np.random.default_rng(seed)
    paths, targets = [], []
    for i, c in enumerate(classes):
        files = sorted(p for p in (Path(syn_dir) / c).glob("*") if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            continue
        chosen = rng.choice(len(files), size=min(per_class, len(files)), replace=False)
        for j in sorted(chosen):
            paths.append(files[j].as_posix())
            if task == "multiclass":
                targets.append(i)
            else:
                t = np.zeros(len(classes), np.int64)
                t[i] = 1
                targets.append(t)
    shape = (0,) if task == "multiclass" else (0, len(classes))
    arr = np.array(targets, np.int64) if targets else np.zeros(shape, np.int64)
    return Records(paths, arr, [f"syn:{p}" for p in paths], ["train"] * len(paths))


def read_rgb(path: str | Path) -> np.ndarray:
    """Read an image as uint8 RGB; works with non-ASCII Windows paths (cv2.imread does not)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def prepare_image(img: np.ndarray, img_size: int, enhance: bool) -> np.ndarray:
    """Resize (area interpolation) and optionally equalize; returns uint8 RGB."""
    interp = cv2.INTER_AREA if min(img.shape[:2]) > img_size else cv2.INTER_CUBIC
    out = cv2.resize(img, (img_size, img_size), interpolation=interp)
    return enhance_image(out) if enhance else out


@lru_cache(maxsize=4096)
def _load_cached(path: str, img_size: int, enhance: bool) -> np.ndarray:
    arr = prepare_image(read_rgb(path), img_size, enhance)
    arr.setflags(write=False)
    return arr


def load_images(paths: list[str], img_size: int, enhance: bool) -> np.ndarray:
    if not paths:
        return np.zeros((0, img_size, img_size, 3), np.uint8)
    return np.stack([_load_cached(p, img_size, enhance) for p in paths])


def augmentation_layers(cfg, seed: int):
    import tensorflow as tf
    from tensorflow.keras import layers

    ops = []
    if cfg.flip:
        ops.append(layers.RandomFlip("horizontal", seed=seed))
    if cfg.rotation:
        ops.append(layers.RandomRotation(cfg.rotation, fill_mode="constant", seed=seed + 1))
    if cfg.zoom:
        ops.append(layers.RandomZoom(cfg.zoom, fill_mode="constant", seed=seed + 2))
    if cfg.translate:
        ops.append(layers.RandomTranslation(cfg.translate, cfg.translate, fill_mode="constant", seed=seed + 3))
    if cfg.contrast:
        ops.append(layers.RandomContrast(cfg.contrast, seed=seed + 4))
    return tf.keras.Sequential(ops, name="augment")


def make_dataset(x: np.ndarray, y: np.ndarray, batch_size: int, training: bool, augment_cfg=None,
                 seed: int = 0, sample_weight: np.ndarray | None = None):
    """tf.data pipeline; images stay float32 in [0, 255] (models rescale internally)."""
    import tensorflow as tf

    tensors = (x, y) if sample_weight is None else (x, y, sample_weight)
    ds = tf.data.Dataset.from_tensor_slices(tensors)
    if training:
        ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda img, *rest: (tf.cast(img, tf.float32), *rest), num_parallel_calls=tf.data.AUTOTUNE)
    if training and augment_cfg is not None and augment_cfg.enabled:
        aug = augmentation_layers(augment_cfg, seed)
        ds = ds.map(lambda img, *rest: (tf.clip_by_value(aug(img, training=True), 0.0, 255.0), *rest),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
