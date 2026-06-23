# src/verifier_anchored_rl/experiment.py
"""Experimental-framework library: provenance capture, the experiment
manifest/index, and the raw-results schema. See
docs/superpowers/specs/2026-06-23-experimental-framework-design.md.

Importable without torch/transformers/vllm: those are read lazily, only for
version strings, and their absence is tolerated.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .run_artifacts import utc_timestamp


def _git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _pip_freeze() -> str:
    try:
        return subprocess.check_output(["python", "-m", "pip", "freeze"],
                                       text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""


def _version(mod: str) -> str | None:
    try:
        import importlib
        return getattr(importlib.import_module(mod), "__version__", None)
    except Exception:
        return None


def capture_provenance(argv: list[str], *, seeds: dict,
                       extra: dict | None = None) -> dict:
    """All facts needed to reproduce a run or experiment. Tolerates missing
    git and missing optional libraries (records None / "nogit")."""
    sha = _git(["rev-parse", "--short=8", "HEAD"])
    freeze = _pip_freeze()
    prov: dict[str, Any] = {
        "git": {
            "sha": sha or "nogit",
            "dirty": bool(_git(["status", "--porcelain"])) if sha else False,
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]) or "nogit",
        },
        "command": {"argv": list(argv), "shell": shlex.join(argv)},
        "seeds": dict(seeds),
        "env": {
            "python": platform.python_version(),
            "numpy": _version("numpy"),
            "torch": _version("torch"),
            "transformers": _version("transformers"),
            "vllm": _version("vllm"),
            "platform": platform.platform(),
        },
        "env_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
        "env_freeze_text": freeze,
        "host": {"hostname": platform.node()},
        "timestamps": {"created": utc_timestamp()},
    }
    if extra:
        prov.update(extra)
    return prov


def provenance_from_metadata(meta: dict) -> dict:
    """Shim: map an existing run's metadata.json into the unified provenance
    schema, filling derivable fields. Lets experiments ingest today's runs
    without refactoring the run harness."""
    args = dict(meta.get("args", {}))
    embedder = args.get("embedder", "hidden_state")
    return {
        "git": {"sha": meta.get("git_sha", "nogit"),
                "dirty": bool(meta.get("git_dirty", False)),
                "branch": meta.get("git_branch", "nogit")},
        "command": {"argv": [], "shell": ""},
        "seeds": {"global": args.get("seed", meta.get("seed"))},
        "env": {"python": meta.get("python"), "torch": meta.get("torch"),
                "transformers": meta.get("transformers"), "vllm": meta.get("vllm"),
                "platform": None},
        "env_freeze_sha256": "",
        "host": {"hostname": meta.get("hostname"), "gpu": meta.get("gpu"),
                 "cuda": meta.get("cuda")},
        "model": args.get("model", meta.get("model")),
        "embedder": {"kind": embedder, "id": args.get("embedder_model")},
        "timestamps": {"created": meta.get("start_time")},
    }


__all__ = ["capture_provenance", "provenance_from_metadata"]
