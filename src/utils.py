"""Reproducibility helpers: seeding, run metadata."""
from __future__ import annotations

import hashlib
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
    except ImportError:
        return
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state(repo: str | Path = ".") -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def environment_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__}
    try:
        import tensorflow as tf
        info["tensorflow"] = tf.__version__
        info["gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    except ImportError:
        pass
    return info
