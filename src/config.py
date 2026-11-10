"""Experiment configuration: YAML file + dotted command-line overrides.

    cfg = load_config("configs/multilabel_efficientnet.yaml", ["train.epochs=5", "seed=1"])
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TASKS = ("multiclass", "multilabel")
ARCHS = ("small_cnn", "efficientnet_b0", "mobilenet_v2")


@dataclass
class DataConfig:
    manifest: str = "data/manifest_split.csv"
    classes: list[str] = field(default_factory=list)
    # multiclass: one label per image (images with other/extra labels are excluded).
    # multilabel: multi-hot over `classes`; images labelled only `negative_label` are all-zero.
    task: str = "multiclass"
    negative_label: str = "Control"
    img_size: int = 224
    enhance: bool = True  # luminance histogram equalization (src.preprocessing)


@dataclass
class ModelConfig:
    arch: str = "efficientnet_b0"
    pretrained: bool = True  # ImageNet weights for transfer-learning backbones
    dropout: float = 0.3
    fine_tune_layers: int = 20  # backbone layers unfrozen in phase 2 (BatchNorm stays frozen)


@dataclass
class TrainConfig:
    epochs: int = 30  # phase 1: head only (backbone frozen) / whole net for small_cnn
    fine_tune_epochs: int = 10  # phase 2: top `fine_tune_layers` unfrozen
    batch_size: int = 16
    lr: float = 1e-3
    fine_tune_lr: float = 1e-5
    early_stopping_patience: int = 8
    class_weight: str | None = "balanced"  # multiclass only; None disables
    threshold: float = 0.5  # multilabel decision threshold


@dataclass
class AugmentConfig:
    enabled: bool = True
    flip: bool = True
    rotation: float = 0.05  # fraction of a full turn (0.05 = +/-18 degrees)
    zoom: float = 0.1
    translate: float = 0.05
    contrast: float = 0.1


@dataclass
class SyntheticConfig:
    dir: str | None = None  # <dir>/<class>/*.png written by `python -m src.gan.acgan generate`
    per_class: int = 0  # synthetic images added to the TRAIN split only, per class


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    seed: int = 42
    deterministic: bool = False  # tf op determinism (slower)
    cv_folds: int = 0  # >1: StratifiedGroupKFold over train+val; test stays held out
    bootstrap: int = 1000  # resamples for confidence intervals
    output_dir: str = "outputs/runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)

    def validate(self) -> ExperimentConfig:
        if self.data.task not in TASKS:
            raise ValueError(f"data.task must be one of {TASKS}, got {self.data.task!r}")
        if self.model.arch not in ARCHS:
            raise ValueError(f"model.arch must be one of {ARCHS}, got {self.model.arch!r}")
        if not self.data.classes:
            raise ValueError("data.classes must list at least one class")
        if len(set(self.data.classes)) != len(self.data.classes):
            raise ValueError(f"duplicate class in data.classes: {self.data.classes}")
        if self.data.task == "multiclass" and len(self.data.classes) < 2:
            raise ValueError("multiclass needs at least two classes")
        if self.data.task == "multilabel" and self.data.negative_label in self.data.classes:
            raise ValueError("negative_label must not be one of the multilabel classes")
        if self.synthetic.per_class and not self.synthetic.dir:
            raise ValueError("synthetic.per_class > 0 requires synthetic.dir")
        if self.cv_folds == 1 or self.cv_folds < 0:
            raise ValueError("cv_folds must be 0 (off) or >= 2")
        return self

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _build(cls, values: dict, where: str):
    if not isinstance(values, dict):
        raise TypeError(f"{where or 'config'} must be a mapping, got {type(values).__name__}")
    kwargs = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for key, value in values.items():
        if key not in fields:
            raise KeyError(f"unknown config key {where + key!r}; valid: {sorted(fields)}")
        sub = _NESTED.get((cls, key))
        kwargs[key] = _build(sub, value, f"{where}{key}.") if sub else value
    return cls(**kwargs)


_NESTED = {
    (ExperimentConfig, "data"): DataConfig,
    (ExperimentConfig, "model"): ModelConfig,
    (ExperimentConfig, "train"): TrainConfig,
    (ExperimentConfig, "augment"): AugmentConfig,
    (ExperimentConfig, "synthetic"): SyntheticConfig,
}


def apply_overrides(raw: dict, overrides: list[str] | None) -> dict:
    """Apply ``a.b.c=value`` overrides; values are parsed as YAML (numbers, bools, lists)."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        key, value = item.split("=", 1)
        node = raw
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return raw


def config_from_dict(raw: dict, overrides: list[str] | None = None) -> ExperimentConfig:
    return _build(ExperimentConfig, apply_overrides(dict(raw), overrides), "").validate()


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw.setdefault("name", Path(path).stem)
    return config_from_dict(raw, overrides)
