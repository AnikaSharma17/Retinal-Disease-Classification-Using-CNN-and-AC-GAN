"""Dataset integrity checks: duplicate images across splits/labels and offline-augmented files.

Usage (from the repo root):
    python -m src.data.integrity dataset            # tree laid out as <split>/<label>/<file>
    python -m src.data.integrity dataset --json report.json

Exits with status 1 when a blocking problem is found (an image shared between splits,
or one image filed under more than one label), so it can gate training in CI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_NAMES = {"train", "val", "valid", "validation", "test"}

# Keras ImageDataGenerator(save_to_dir=..., save_prefix="image") writes "<prefix>_<i>_<rand>.<fmt>".
_AUGMENTED_NAME = re.compile(r"^[A-Za-z]*_\d+_\d+\.(jpe?g|png)$", re.IGNORECASE)


def is_augmented_name(filename: str) -> bool:
    """True if the file name matches the Keras offline-augmentation naming pattern."""
    return bool(_AUGMENTED_NAME.match(filename))


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def dhash(path: Path, hash_size: int = 16) -> int:
    """Difference hash; robust to re-encoding and resizing, not to rotation/flip."""
    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
        px = list(img.getdata())
    bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left = px[row * (hash_size + 1) + col]
            right = px[row * (hash_size + 1) + col + 1]
            bits = (bits << 1) | (left > right)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@dataclass
class ImageRecord:
    path: Path
    label: str
    split: str | None
    sha256: str = ""
    augmented_name: bool = False


def scan_tree(root: Path) -> list[ImageRecord]:
    """Scan <root>/<split>/<label>/<file> or <root>/<label>/<file>."""
    root = Path(root)
    records: list[ImageRecord] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = path.relative_to(root).parts
        if len(rel) >= 3 and rel[0].lower() in SPLIT_NAMES:
            split, label = rel[0], rel[1]
        elif len(rel) >= 2:
            split, label = None, rel[-2]
        else:
            continue
        records.append(
            ImageRecord(path=path, label=label, split=split, augmented_name=is_augmented_name(path.name))
        )
    return records


@dataclass
class IntegrityReport:
    n_files: int = 0
    n_unique: int = 0
    counts: dict = field(default_factory=dict)  # split -> label -> {"total", "augmented_named"}
    cross_split: list = field(default_factory=list)  # [{"sha256", "paths"}]
    cross_label: list = field(default_factory=list)
    within_split_dupes: list = field(default_factory=list)

    @property
    def augmented_fraction(self) -> float:
        aug = sum(c["augmented_named"] for s in self.counts.values() for c in s.values())
        return aug / self.n_files if self.n_files else 0.0

    @property
    def has_blocking_issues(self) -> bool:
        return bool(self.cross_split or self.cross_label)

    def to_dict(self) -> dict:
        return {
            "n_files": self.n_files,
            "n_unique": self.n_unique,
            "augmented_fraction": round(self.augmented_fraction, 4),
            "counts": self.counts,
            "cross_split": self.cross_split,
            "cross_label": self.cross_label,
            "within_split_dupes": self.within_split_dupes,
        }


def check_records(records: list[ImageRecord]) -> IntegrityReport:
    report = IntegrityReport(n_files=len(records))
    by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in records:
        if not r.sha256:
            r.sha256 = file_sha256(r.path)
        by_hash[r.sha256].append(r)
        split_counts = report.counts.setdefault(r.split or "-", {})
        c = split_counts.setdefault(r.label, {"total": 0, "augmented_named": 0})
        c["total"] += 1
        c["augmented_named"] += int(r.augmented_name)
    report.n_unique = len(by_hash)

    for sha, group in sorted(by_hash.items()):
        if len(group) < 2:
            continue
        entry = {"sha256": sha, "paths": [r.path.as_posix() for r in group]}
        splits = {r.split for r in group}
        labels = {r.label for r in group}
        if len(splits) > 1:
            report.cross_split.append(entry)
        if len(labels) > 1:
            report.cross_label.append(entry)
        if len(splits) == 1 and len(labels) == 1:
            report.within_split_dupes.append(entry)
    return report


def format_report(report: IntegrityReport) -> str:
    lines = [
        f"files: {report.n_files}   unique (sha256): {report.n_unique}   "
        f"offline-augmented by name: {report.augmented_fraction:.1%}",
        "",
        f"{'split':<8}{'label':<16}{'total':>7}{'aug-named':>11}{'originals':>11}",
    ]
    for split, labels in sorted(report.counts.items()):
        for label, c in sorted(labels.items()):
            lines.append(
                f"{split:<8}{label:<16}{c['total']:>7}{c['augmented_named']:>11}"
                f"{c['total'] - c['augmented_named']:>11}"
            )
    for title, items in (
        ("IDENTICAL IMAGE IN MORE THAN ONE SPLIT (leakage)", report.cross_split),
        ("IDENTICAL IMAGE UNDER MORE THAN ONE LABEL", report.cross_label),
        ("duplicate within one split/label", report.within_split_dupes),
    ):
        lines += ["", f"{title}: {len(items)}"]
        for e in items:
            lines.append(f"  {e['sha256'][:12]}  " + " | ".join(e["paths"]))
    if report.augmented_fraction > 0:
        lines += [
            "",
            "WARNING: offline-augmented files present. Augmented copies of one source image cannot be",
            "traced back to it, so splitting a tree that contains them leaks near-duplicates across",
            "splits. Split the originals (src.data.manifest) and augment the training split on the fly.",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--json", type=Path, help="also write the report as JSON")
    args = ap.parse_args(argv)

    report = check_records(scan_tree(args.root))
    print(format_report(report))
    if args.json:
        args.json.write_text(json.dumps(report.to_dict(), indent=2))
    return 1 if report.has_blocking_issues else 0


if __name__ == "__main__":
    sys.exit(main())
