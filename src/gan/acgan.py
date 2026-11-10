"""Class-conditional AC-GAN trained on THIS project's fundus images (train split only).

    python -m src.gan.acgan train --config configs/acgan.yaml [--set gan.epochs=200]
    python -m src.gan.acgan generate --run outputs/gan/<run> --per-class 50 --out outputs/synthetic/<run>

Differences from legacy/acgan/train_acgan.py (which trained on an unrelated glaucoma set):
* data comes from the split manifest, so no validation/test image is ever seen by the GAN
  and synthetic images can be added to the classifier's training split only;
* explicit GradientTape loop (no `trainable` toggling across compiled models);
* one-sided label smoothing and DiffAugment-style translation/colour augmentation applied
  to real AND fake images before the discriminator, the standard fix for tiny datasets;
* periodic checkpoints and sample grids instead of saving two models every epoch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass
class GanConfig:
    name: str = "acgan"
    manifest: str = "data/manifest_split.csv"
    classes: list[str] = field(default_factory=list)
    img_size: int = 64
    enhance: bool = True
    latent_dim: int = 128
    epochs: int = 300
    batch_size: int = 16
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    aux_weight: float = 1.0
    label_smoothing: float = 0.1
    diffaugment: bool = True
    sample_every: int = 25
    seed: int = 42
    output_dir: str = "outputs/gan"


def load_gan_config(path: str, overrides: list[str] | None = None) -> GanConfig:
    from src.config import apply_overrides

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw = apply_overrides(raw, overrides)
    raw = raw.get("gan", raw) | {k: v for k, v in raw.items() if k != "gan"}
    known = set(GanConfig.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise KeyError(f"unknown GAN config keys: {sorted(unknown)}")
    cfg = GanConfig(**raw)
    if cfg.img_size % 16:
        raise ValueError("img_size must be a multiple of 16")
    if len(cfg.classes) < 2:
        raise ValueError("an AC-GAN needs at least two classes")
    return cfg


# ----------------------------------------------------------------------------- models
def build_generator(latent_dim: int, n_classes: int, img_size: int):
    import tensorflow as tf
    from tensorflow.keras import layers

    start = img_size // 16
    z = layers.Input((latent_dim,), name="z")
    label = layers.Input((1,), dtype="int32", name="label")
    emb = layers.Flatten()(layers.Embedding(n_classes, latent_dim)(label))
    x = layers.Multiply()([z, emb])
    x = layers.Dense(start * start * 512, use_bias=False)(x)
    x = layers.Reshape((start, start, 512))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    for filters in (256, 128, 64, 32):  # 4 x upsample-by-2 = x16
        x = layers.Conv2DTranspose(filters, 4, strides=2, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    out = layers.Conv2D(3, 3, padding="same", activation="tanh", name="image")(x)
    return tf.keras.Model([z, label], out, name="generator")


def build_discriminator(n_classes: int, img_size: int):
    import tensorflow as tf
    from tensorflow.keras import layers

    img = layers.Input((img_size, img_size, 3), name="image")
    x = img
    for i, filters in enumerate((64, 128, 256, 512)):
        x = layers.Conv2D(filters, 4, strides=2, padding="same")(x)
        if i:
            x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    real_fake = layers.Dense(1, name="real_fake")(x)  # logits
    aux = layers.Dense(n_classes, name="aux_class")(x)  # logits
    return tf.keras.Model(img, [real_fake, aux], name="discriminator")


def diff_augment(x, seed=None):
    """DiffAugment (Zhao et al., 2020): random brightness/saturation/contrast + translation."""
    import tensorflow as tf

    b = tf.shape(x)[0]
    x = x + (tf.random.uniform([b, 1, 1, 1]) - 0.5) * 0.4
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    x = (x - mean) * tf.random.uniform([b, 1, 1, 1], 0.5, 1.5) + mean
    mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
    x = (x - mean) * tf.random.uniform([b, 1, 1, 1], 0.5, 1.5) + mean
    size = tf.shape(x)[1]
    pad = size // 8
    x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], constant_values=-1.0)
    offs = tf.random.uniform([2], 0, 2 * pad + 1, dtype=tf.int32)
    return tf.slice(x, [0, offs[0], offs[1], 0], [-1, size, size, -1])


# ----------------------------------------------------------------------------- training
def load_training_images(cfg: GanConfig) -> tuple[np.ndarray, np.ndarray]:
    from src.data.dataset import load_images, read_manifest, select_records

    rec = select_records(read_manifest(cfg.manifest), cfg.classes, "multiclass", splits=("train",))
    if len(rec) == 0:
        raise ValueError("no single-label training images for these classes")
    x = load_images(rec.paths, cfg.img_size, cfg.enhance).astype(np.float32) / 127.5 - 1.0
    return x, rec.targets.astype(np.int32)


class ACGAN:
    def __init__(self, cfg: GanConfig):
        import tensorflow as tf

        self.cfg = cfg
        k = len(cfg.classes)
        self.g = build_generator(cfg.latent_dim, k, cfg.img_size)
        self.d = build_discriminator(k, cfg.img_size)
        self.g_opt = tf.keras.optimizers.Adam(cfg.lr_g, beta_1=0.5)
        self.d_opt = tf.keras.optimizers.Adam(cfg.lr_d, beta_1=0.5)
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self._step = tf.function(self._train_step)

    def _aug(self, x):
        return diff_augment(x) if self.cfg.diffaugment else x

    def _train_step(self, real, labels):
        import tensorflow as tf

        b = tf.shape(real)[0]
        k = len(self.cfg.classes)
        z = tf.random.normal([b, self.cfg.latent_dim])
        fake_labels = tf.random.uniform([b, 1], 0, k, dtype=tf.int32)
        with tf.GradientTape() as dt, tf.GradientTape() as gt:
            fake = self.g([z, fake_labels], training=True)
            rf_real, cls_real = self.d(self._aug(real), training=True)
            rf_fake, cls_fake = self.d(self._aug(fake), training=True)
            smooth = 1.0 - self.cfg.label_smoothing
            d_adv = self.bce(tf.fill(tf.shape(rf_real), smooth), rf_real) + self.bce(tf.zeros_like(rf_fake), rf_fake)
            d_aux = self.ce(labels, cls_real) + self.ce(fake_labels, cls_fake)
            d_loss = d_adv + self.cfg.aux_weight * d_aux
            g_adv = self.bce(tf.ones_like(rf_fake), rf_fake)
            g_aux = self.ce(fake_labels, cls_fake)
            g_loss = g_adv + self.cfg.aux_weight * g_aux
        self.d_opt.apply_gradients(zip(dt.gradient(d_loss, self.d.trainable_variables), self.d.trainable_variables))
        self.g_opt.apply_gradients(zip(gt.gradient(g_loss, self.g.trainable_variables), self.g.trainable_variables))
        real_acc = tf.reduce_mean(tf.cast(tf.argmax(cls_real, 1, output_type=tf.int32) == labels, tf.float32))
        return {"d_loss": d_loss, "g_loss": g_loss, "d_aux": d_aux, "aux_acc_real": real_acc}

    def generate(self, per_class: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        k = len(self.cfg.classes)
        labels = np.repeat(np.arange(k), per_class).astype(np.int32)[:, None]
        z = rng.standard_normal((len(labels), self.cfg.latent_dim)).astype(np.float32)
        imgs = self.g.predict([z, labels], batch_size=64, verbose=0)
        return to_uint8(imgs), labels[:, 0]

    def train(self, x: np.ndarray, y: np.ndarray, out_dir: Path, verbose: bool = True) -> list[dict]:
        import tensorflow as tf

        ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(x), seed=self.cfg.seed) \
            .batch(self.cfg.batch_size, drop_remainder=len(x) >= self.cfg.batch_size).repeat(1)
        log = []
        for epoch in range(1, self.cfg.epochs + 1):
            agg: dict[str, list] = {}
            for real, labels in ds:
                for k_, v in self._step(real, labels).items():
                    agg.setdefault(k_, []).append(float(v))
            row = {"epoch": epoch, **{k_: float(np.mean(v)) for k_, v in agg.items()}}
            log.append(row)
            if epoch % self.cfg.sample_every == 0 or epoch == self.cfg.epochs:
                save_grid(self.generate(4, seed=epoch)[0], len(self.cfg.classes), out_dir / f"samples_{epoch:04d}.png")
                if verbose:
                    print(" ".join(f"{k_}={v:.3f}" if isinstance(v, float) else f"{k_}={v}" for k_, v in row.items()))
        return log


def to_uint8(imgs: np.ndarray) -> np.ndarray:
    return np.uint8(np.clip((imgs + 1.0) * 127.5, 0, 255))


def save_grid(imgs: np.ndarray, rows: int, path: Path) -> None:
    cols = len(imgs) // rows
    grid = np.concatenate([np.concatenate(list(imgs[r * cols:(r + 1) * cols]), axis=1) for r in range(rows)], axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def train_cli(cfg: GanConfig, run_dir: str | None = None, verbose: bool = True) -> Path:
    from src.utils import set_seed

    set_seed(cfg.seed)
    out = Path(run_dir) if run_dir else Path(cfg.output_dir) / f"{cfg.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    x, y = load_training_images(cfg)
    if verbose:
        print(f"training AC-GAN on {len(x)} train-split images: "
              f"{dict(zip(cfg.classes, np.bincount(y, minlength=len(cfg.classes)).tolist()))}")
    gan = ACGAN(cfg)
    log = gan.train(x, y, out, verbose)
    gan.g.save(out / "generator.keras")
    (out / "train_log.json").write_text(json.dumps(log, indent=2))
    return out


def generate_cli(run_dir: str | Path, per_class: int, out_dir: str | Path, seed: int = 0) -> Path:
    import tensorflow as tf

    run_dir, out_dir = Path(run_dir), Path(out_dir)
    cfg = GanConfig(**yaml.safe_load((run_dir / "config.yaml").read_text()))
    gan = ACGAN.__new__(ACGAN)
    gan.cfg = cfg
    gan.g = tf.keras.models.load_model(run_dir / "generator.keras")
    imgs, labels = gan.generate(per_class, seed)
    for i, (img, lab) in enumerate(zip(imgs, labels)):
        d = out_dir / cfg.classes[lab]
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"syn_{i:05d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return out_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    t.add_argument("--set", nargs="*", default=[])
    t.add_argument("--run-dir")
    g = sub.add_parser("generate")
    g.add_argument("--run", required=True)
    g.add_argument("--per-class", type=int, default=50)
    g.add_argument("--out", required=True)
    g.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.cmd == "train":
        print(f"saved to {train_cli(load_gan_config(args.config, args.set), args.run_dir)}")
    else:
        print(f"wrote images to {generate_cli(args.run, args.per_class, args.out, args.seed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
