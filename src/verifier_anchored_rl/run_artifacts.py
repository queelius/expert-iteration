"""Run-directory artifacts for long experiment runs: a unique provenance
directory, metadata, append-only JSONL, and an atomic checkpoint save/load.

Pure filesystem and JSON plus ``torch.save``; no model and no GPU, so it is
unit-testable on any machine. The harness (``experiments/run_outer_loop.py``)
composes these with the outer-loop callbacks to make a run observable and
resumable. Motivated by the 14.7 hour first GPU run that left only three
summary lines (see experiments/results/expert_iteration_rounds.md)."""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def seed_everything(seed: int) -> None:
    """Seed python random, numpy global RNG, and torch (CPU + CUDA if available).

    Torch is imported lazily so this module remains importable on machines
    without torch installed. Call this BEFORE constructing any nn.Module to
    ensure reproducible weight initialisation."""
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_json_atomic(path: "str | Path", obj: "dict[str, Any]") -> None:
    """Write *obj* as indented JSON to *path* atomically (write to a ``.tmp``
    sibling then ``os.replace``), matching the safety of ``save_checkpoint_atomic``.
    A crash mid-write never leaves torn JSON at *path*."""
    path = Path(path)
    tmp = str(path) + ".tmp"
    tmp_path = Path(tmp)
    tmp_path.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, str(path))


def utc_timestamp() -> str:
    """Compact UTC timestamp like ``20260619T134501Z`` for run-dir names."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def create_run_dir(
    *,
    root: str,
    tag: str | None,
    metadata: dict[str, Any],
    timestamp: str | None = None,
) -> Path:
    """Create ``<root>/<timestamp>[-<tag>]/`` and write ``metadata.json`` with
    status ``running`` and a ``start_time``. Returns the run directory path."""
    ts = timestamp or utc_timestamp()
    name = f"{ts}-{tag}" if tag else ts
    run_dir = Path(root) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata)
    meta.setdefault("start_time", datetime.now(timezone.utc).isoformat())
    meta["status"] = "running"
    write_json_atomic(run_dir / "metadata.json", meta)
    return run_dir


def update_status(run_dir: Path, status: str) -> None:
    """Rewrite ``metadata.json`` with a new status and an ``end_time``."""
    path = Path(run_dir) / "metadata.json"
    meta = json.loads(path.read_text())
    meta["status"] = status
    meta["end_time"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(path, meta)


def append_jsonl(path: str, obj: dict[str, Any]) -> None:
    """Append one JSON object as a line. Flushed so a crash keeps prior lines."""
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")
        f.flush()


def save_checkpoint_atomic(path: str, state: dict[str, Any]) -> None:
    """Write a checkpoint to ``path.tmp`` then ``os.replace`` it into place, so
    a crash mid-write never leaves a torn checkpoint at ``path``."""
    import torch

    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> dict[str, Any]:
    """Load a checkpoint dict (CPU map location). Not ``weights_only``: the
    checkpoint carries optimizer and buffer state, not only tensors."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


__all__ = [
    "append_jsonl",
    "create_run_dir",
    "load_checkpoint",
    "save_checkpoint_atomic",
    "seed_everything",
    "update_status",
    "utc_timestamp",
    "write_json_atomic",
]
