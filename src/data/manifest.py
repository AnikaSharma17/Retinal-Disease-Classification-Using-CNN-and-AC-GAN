"""Build a de-duplicated image manifest and a leakage-free, group-aware split.

The manifest has one row per unique image (sha256). An image filed under several label
folders keeps all its labels (``labels`` is ';'-joined), so comorbidity / mislabelling is
visible instead of becoming contradictory training targets. Near-duplicates (re-encoded or
resized copies, found with a perceptual hash) share a ``group_id`` and are always assigned to
the same split. When patient IDs become available, pass them with ``--patients`` so that
all images of one patient share a group as well.

Usage (from the repo root):
    # 1) build: originals only (offline-augmented copies are excluded by default)
    python -m src.data.manifest build <source_dir> --out data/manifest.csv

    # 2) split by group, stratified by label set; optionally copy single-label images into
    #    <dir>/<split>/<label>/ so the existing flow_from_directory scripts can use it
    python -m src.data.manifest split data/manifest.csv --out data/manifest_split.csv \
        --ratios 0.7 0.15 0.15 --seed 42 --materialize data/clean
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from src.data.integrity import dhash, file_sha256, hamming, scan_tree

FIELDS = ["group_id", "sha256", "path", "all_paths", "labels", "n_labels", "source_splits"]
SPLITS = ("train", "val", "test")


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def build_manifest(
    source: Path,
    include_augmented: bool = False,
    near_dup_threshold: int = 10,
    patients: dict[str, str] | None = None,
) -> list[dict]:
    """Return manifest rows for every unique image under ``source``.

    ``patients`` optionally maps a file name (as it appears on disk) to a patient ID.
    """
    records = [r for r in scan_tree(Path(source)) if include_augmented or not r.augmented_name]
    by_hash: dict[str, list] = defaultdict(list)
    for r in records:
        r.sha256 = file_sha256(r.path)
        by_hash[r.sha256].append(r)

    rows = []
    for sha, group in sorted(by_hash.items()):
        group.sort(key=lambda r: r.path.as_posix())
        labels = sorted({r.label for r in group})
        rows.append(
            {
                "sha256": sha,
                "path": group[0].path.as_posix(),
                "all_paths": ";".join(r.path.as_posix() for r in group),
                "labels": ";".join(labels),
                "n_labels": len(labels),
                "source_splits": ";".join(sorted({r.split or "-" for r in group})),
            }
        )

    # Group near-duplicates (and same-patient images) so they never straddle splits.
    uf = _UnionFind(len(rows))
    if near_dup_threshold >= 0:
        hashes = [dhash(Path(row["path"])) for row in rows]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if hamming(hashes[i], hashes[j]) <= near_dup_threshold:
                    uf.union(i, j)
    if patients:
        first_by_patient: dict[str, int] = {}
        for i, row in enumerate(rows):
            for p in row["all_paths"].split(";"):
                pid = patients.get(Path(p).name)
                if pid is None:
                    continue
                if pid in first_by_patient:
                    uf.union(first_by_patient[pid], i)
                else:
                    first_by_patient[pid] = i
    roots = sorted({uf.find(i) for i in range(len(rows))})
    gid = {root: f"g{n:05d}" for n, root in enumerate(roots)}
    for i, row in enumerate(rows):
        row["group_id"] = gid[uf.find(i)]
    return rows


def split_manifest(
    rows: list[dict], ratios: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 42
) -> list[dict]:
    """Assign whole groups to train/val/test, stratified by each group's label set."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)

    strata: dict[str, list[str]] = defaultdict(list)
    for g, members in groups.items():
        key = ";".join(sorted({lab for m in members for lab in m["labels"].split(";")}))
        strata[key].append(g)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for key in sorted(strata):
        gids = sorted(strata[key])
        rng.shuffle(gids)
        n = len(gids)
        n_test = round(n * ratios[2])
        n_val = round(n * ratios[1])
        if n >= 3:  # every stratum that can afford it is represented in val and test
            n_test, n_val = max(n_test, 1), max(n_val, 1)
        n_train = n - n_val - n_test
        if n_train < 1 and n > 0:  # keep at least one training example per stratum
            if n_val > n_test:
                n_val -= 1
            else:
                n_test = max(n_test - 1, 0)
        for i, g in enumerate(gids):
            assignment[g] = "test" if i < n_test else "val" if i < n_test + n_val else "train"

    return [{**row, "split": assignment[row["group_id"]]} for row in rows]


def materialize(rows: list[dict], out_dir: Path) -> tuple[int, int]:
    """Copy single-label images to <out_dir>/<split>/<label>/. Returns (copied, skipped_multilabel)."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    copied = skipped = 0
    for row in rows:
        if int(row["n_labels"]) != 1:
            skipped += 1
            continue
        src = Path(row["path"])
        dst = out_dir / row["split"] / row["labels"] / f"{row['sha256'][:12]}{src.suffix.lower()}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied, skipped


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = FIELDS + [k for k in rows[0] if k not in FIELDS] if rows else FIELDS
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarize(rows: list[dict]) -> str:
    counts: dict[tuple, int] = defaultdict(int)
    for r in rows:
        counts[(r.get("split", "-"), r["labels"])] += 1
    lines = [f"unique images: {len(rows)}   groups: {len({r['group_id'] for r in rows})}   "
             f"multi-label: {sum(int(r['n_labels']) > 1 for r in rows)}"]
    lines += [f"  {s:<6} {lab:<40} {n}" for (s, lab), n in sorted(counts.items())]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="scan a labelled image tree into a de-duplicated manifest")
    b.add_argument("source", type=Path)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--include-augmented", action="store_true",
                   help="keep files named like Keras offline augmentation output (not recommended)")
    b.add_argument("--near-dup-threshold", type=int, default=10,
                   help="max Hamming distance between 256-bit dHashes to group as near-duplicates; -1 disables")
    b.add_argument("--patients", type=Path,
                   help="CSV with columns filename,patient_id to group images of one patient")

    s = sub.add_parser("split", help="group-aware stratified train/val/test split")
    s.add_argument("manifest", type=Path)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--materialize", type=Path,
                   help="copy single-label images to DIR/<split>/<label>/ (DIR is wiped first)")

    args = ap.parse_args(argv)
    if args.cmd == "build":
        patients = None
        if args.patients:
            patients = {r["filename"]: r["patient_id"] for r in read_csv(args.patients)}
        rows = build_manifest(args.source, args.include_augmented, args.near_dup_threshold, patients)
        write_csv(rows, args.out)
    else:
        rows = split_manifest(read_csv(args.manifest), tuple(args.ratios), args.seed)
        write_csv(rows, args.out)
        if args.materialize:
            copied, skipped = materialize(rows, args.materialize)
            print(f"materialized {copied} images to {args.materialize} "
                  f"(skipped {skipped} multi-label images)")
    print(summarize(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
