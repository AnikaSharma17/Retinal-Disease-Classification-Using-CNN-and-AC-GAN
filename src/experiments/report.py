"""Collect saved run artifacts into one markdown results report.

    python -m src.experiments.report [--out outputs/report.md]

Reads only files written by the pipeline (metrics.json, baselines, Grad-CAM summary, KID/FID,
ablation tables), so every number in the report is traceable to a run directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _fmt(v: dict | None) -> str:
    if not v or v.get("value") is None:
        return "n/a"
    ci = v.get("ci95")
    return f"{v['value']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else f"{v['value']:.2f}†"


def _headline(m: dict | None) -> dict:
    from src.evaluation.metrics import headline

    return headline(m) if m else {}


def _row(label: str, m: dict | None, keys: list[str]) -> str:
    h = _headline(m)
    return f"| {label} | {h.get('n', 0)} | " + " | ".join(_fmt(h.get(k)) for k in keys) + " |"


def build_report(root: str | Path = "outputs") -> str:
    root = Path(root)
    out = ["# Results report", "",
           "All numbers are on the held-out **test** split unless noted; brackets are percentile-bootstrap "
           "95% CIs. `n/a` means the metric is undefined for the available labels; † means n < 10, too few "
           "samples for a meaningful CI (the point estimate is essentially noise).", ""]

    groups = {"multilabel": ["auc_macro", "macro_f1", "subset_accuracy"],
              "multiclass": ["accuracy", "balanced_accuracy", "macro_f1", "auc_macro_ovr"]}
    runs = {p.parent.name: json.loads(p.read_text()) for p in sorted((root / "runs").glob("*/metrics.json"))}
    baselines = {p.stem: json.loads(p.read_text()) for p in sorted((root / "baselines").glob("*.json"))}
    for task, keys in groups.items():
        rows = []
        bl_name = "multilabel" if task == "multilabel" else "dm_vs_hyp"
        for name, res in baselines.get(bl_name, {}).items():
            rows.append(_row(f"baseline: {name}", res, keys))
        for name, res in runs.items():
            t = res.get("test") or (res.get("cv") or {}).get("out_of_fold")
            if t and t["task"] == task:
                rows.append(_row(name, res.get("test"), keys))
                if res.get("cv"):
                    rows.append(_row(f"{name} (5-fold out-of-fold, train+val)", res["cv"]["out_of_fold"], keys))
        if rows:
            out += [f"## {task}", "", "| model | n | " + " | ".join(keys) + " |",
                    "|" + "---|" * (2 + len(keys)), *rows, ""]

    for name, res in runs.items():
        out.append(f"- `{name}` split counts: train {res['counts']['train']}, val {res['counts']['val']}, "
                   f"test {res['counts']['test']}")
    out.append("")

    gc = sorted((root / "runs").glob("*/gradcam_summary_*.json"))
    if gc:
        out += ["## Grad-CAM shortcut check", "",
                "Share of Grad-CAM mass outside the retinal disc (0 = all attention on the retina).", ""]
        for p in gc:
            s = json.loads(p.read_text())
            out.append(f"- `{p.parent.name}` ({s['split']}, n={s['n']}): mean {s['mean_outside_fundus_fraction']:.2f}")
        out.append("")

    kid = root / "gan" / "acgan" / "kid_fid.json"
    if kid.exists():
        k = json.loads(kid.read_text())
        out += ["## AC-GAN sample quality (train split, 64x64)", "",
                "| class | n real | n fake | KID (mean ± std) | FID |", "|---|---|---|---|---|"]
        for c, v in {**k.get("per_class", {}), "pooled": k.get("pooled")}.items():
            if v and v.get("kid_mean") is not None:
                out.append(f"| {c} | {v['n_real']} | {v['n_fake']} | {v['kid_mean']:.3f} ± {v['kid_std']:.3f} | "
                           f"{v['fid']:.0f} |")
        out += ["", "KID/FID are on InceptionV3 features; lower is better. FID is strongly biased at this n.", ""]

    for res in sorted((root / "ablations").glob("*/results.md")):
        out += [f"## Ablation: {res.parent.name.rsplit('-', 2)[0]}", "", "Test metrics, mean ± std over seeds.", "",
                res.read_text().strip(), ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--out", type=Path, default=Path("outputs/report.md"))
    args = ap.parse_args(argv)
    text = build_report(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
