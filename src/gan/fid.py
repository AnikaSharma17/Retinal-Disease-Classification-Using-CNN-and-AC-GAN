"""FID and KID between real and generated images (InceptionV3 pool features).

    python -m src.gan.fid --manifest data/manifest_split.csv --classes DM Hypertension \
        --fake outputs/synthetic/<run> [--split train]

KID (Binkowski et al., 2018) is unbiased and usable with tens of images; FID is biased
upward at small n and is reported for comparability only. Both are computed per class
(class-conditional quality) and pooled.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import linalg


def fid_from_features(a: np.ndarray, b: np.ndarray) -> float:
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def kid_from_features(a: np.ndarray, b: np.ndarray, n_subsets: int = 50, subset_size: int = 100,
                      seed: int = 0) -> tuple[float, float]:
    """Mean and std of the unbiased MMD^2 with a cubic polynomial kernel over random subsets."""
    rng = np.random.default_rng(seed)
    d = a.shape[1]
    m = min(subset_size, len(a), len(b))
    if m < 2:
        raise ValueError("KID needs at least 2 images on each side")
    vals = []
    for _ in range(n_subsets):
        x = a[rng.choice(len(a), m, replace=False)]
        y = b[rng.choice(len(b), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        mmd = ((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
               - 2 * kxy.mean())
        vals.append(mmd)
    return float(np.mean(vals)), float(np.std(vals))


def inception_features(images_uint8: np.ndarray, batch: int = 32) -> np.ndarray:
    import tensorflow as tf

    model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", pooling="avg",
                                              input_shape=(299, 299, 3))
    feats = []
    for i in range(0, len(images_uint8), batch):
        x = tf.image.resize(images_uint8[i:i + batch].astype("float32"), (299, 299))
        feats.append(model(tf.keras.applications.inception_v3.preprocess_input(x), training=False).numpy())
    return np.concatenate(feats)


def evaluate(manifest: str, classes: list[str], fake_dir: str, split: str = "train", img_size: int = 64,
             enhance: bool = True) -> dict:
    from src.data.dataset import IMAGE_EXTS, load_images, read_manifest, select_records

    rec = select_records(read_manifest(manifest), classes, "multiclass", splits=(split,))
    real_all = load_images(rec.paths, img_size, enhance)
    fr = inception_features(real_all)
    results: dict = {"per_class": {}}
    fake_feats = []
    for i, c in enumerate(classes):
        files = sorted(str(p) for p in (Path(fake_dir) / c).glob("*") if p.suffix.lower() in IMAGE_EXTS)
        real_c = fr[rec.targets == i]
        if not files or len(real_c) < 2:
            results["per_class"][c] = {"n_real": int(len(real_c)), "n_fake": len(files), "kid": None, "fid": None}
            continue
        ff = inception_features(load_images(files, img_size, enhance=False))
        fake_feats.append(ff)
        kid = kid_from_features(real_c, ff)
        results["per_class"][c] = {"n_real": int(len(real_c)), "n_fake": len(files),
                                   "kid_mean": kid[0], "kid_std": kid[1], "fid": fid_from_features(real_c, ff)}
    if fake_feats:
        ff = np.concatenate(fake_feats)
        kid = kid_from_features(fr, ff)
        results["pooled"] = {"n_real": int(len(fr)), "n_fake": int(len(ff)), "kid_mean": kid[0],
                             "kid_std": kid[1], "fid": fid_from_features(fr, ff)}
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/manifest_split.csv")
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--fake", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    res = evaluate(args.manifest, args.classes, args.fake, args.split, args.img_size)
    print(json.dumps(res, indent=2))
    if args.out:
        args.out.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
