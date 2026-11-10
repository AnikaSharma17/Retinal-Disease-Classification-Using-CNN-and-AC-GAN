"""Run a set of config variants over several seeds and tabulate test metrics.

    python -m src.experiments.ablation --spec configs/ablation_augmentation.yaml

Spec format:
    base: configs/multilabel_efficientnet.yaml
    seeds: [1, 2, 3]
    common: ["train.epochs=20"]          # optional overrides for every variant
    variants:
      real_only:    ["augment.enabled=false"]
      classic_aug:  []
      classic_plus_gan: ["synthetic.dir=outputs/synthetic/acgan", "synthetic.per_class=20"]

Writes outputs/ablations/<spec>-<time>/results.json and results.md (mean +/- std over seeds of
the test headline metrics). Differences smaller than the seed-to-seed std are not evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from src.config import load_config
from src.evaluation.metrics import headline


def run_spec(spec_path: str | Path, out_root: str = "outputs/ablations", verbose: bool = True) -> Path:
    from src.train import run

    spec = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    out = Path(out_root) / f"{Path(spec_path).stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    for variant, overrides in spec["variants"].items():
        results[variant] = []
        for seed in spec.get("seeds", [42]):
            cfg = load_config(spec["base"], [*spec.get("common", []), *overrides, f"seed={seed}",
                                             f"name={variant}-s{seed}"])
            r = run(cfg, run_dir=str(out / "runs" / f"{variant}-s{seed}"), verbose=0)
            results[variant].append({"seed": seed, "test": headline(r["test"]) if r.get("test") else None})
            if verbose:
                print(f"{variant} seed={seed}: {results[variant][-1]['test']}")
    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "results.md").write_text(to_markdown(results))
    return out


def to_markdown(results: dict) -> str:
    metrics = None
    lines = []
    for variant, runs in results.items():
        tests = [r["test"] for r in runs if r["test"]]
        if not tests:
            continue
        metrics = metrics or [k for k in tests[0] if k != "n"]
        cells = []
        for m in metrics:
            vals = [t[m]["value"] for t in tests if t[m]["value"] is not None]
            if not vals:
                cells.append("n/a")
            elif len(vals) == 1:
                cells.append(f"{vals[0]:.3f}")
            else:
                cells.append(f"{np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}")
        lines.append(f"| {variant} | {len(tests)} | {tests[0]['n']} | " + " | ".join(cells) + " |")
    header = "| variant | seeds | n test | " + " | ".join(metrics or []) + " |"
    sep = "|" + "---|" * (3 + len(metrics or []))
    return "\n".join([header, sep, *lines]) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out-root", default="outputs/ablations")
    args = ap.parse_args(argv)
    out = run_spec(args.spec, args.out_root)
    print((out / "results.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
