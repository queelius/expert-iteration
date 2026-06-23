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


# one JSONL row per evaluated (dataset, problem, seed, budget) cell.
RAW_SCHEMA: dict[str, Any] = {
    "dataset": str, "dataset_version": str, "split": str, "problem_id": str,
    "seed": int, "budget_B": int, "correct": bool, "pred": str, "gold": str,
    "verifier_verdict": str, "n_turns": int, "gen_calls": int,
    "gen_tokens": int, "prefill_calls": int, "prefill_tokens": int,
    "flops_est": (int, float, type(None)), "wall_s": (int, float),
    "run_id": str,
}


def validate_raw_row(row: dict) -> None:
    """Raise ValueError if ``row`` is missing a required key or has a wrong
    type. Bools are excluded from the int check (bool is a subclass of int)."""
    for key, typ in RAW_SCHEMA.items():
        if key not in row:
            raise ValueError(f"raw row missing required key {key!r}")
        val = row[key]
        if typ is int and isinstance(val, bool):
            raise ValueError(f"raw row {key!r} must be int, got bool")
        if not isinstance(val, typ):
            raise ValueError(
                f"raw row {key!r} must be {typ}, got {type(val).__name__}")


from .run_artifacts import append_jsonl, write_json_atomic

_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "experiments" / "templates"


class Experiment:
    """A container tying a set of runs to one hypothesis. Creates
    exp/<date>-<slug>/ with a machine-readable manifest, registers the
    experiment in the program index, and offers a validated raw-results sink.
    Runs are referenced by id and never moved."""

    def __init__(self, slug: str, hypothesis: str, *, root: str = "experiments/results",
                 provenance: dict | None = None, date: str | None = None):
        self.root = Path(root)
        self.slug = slug
        self.date = date or utc_timestamp()[:8]
        self.exp_dir = self.root / "exp" / f"{self.date}-{slug}"
        (self.exp_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.exp_dir / "analysis" / "figures").mkdir(parents=True, exist_ok=True)
        (self.exp_dir / "analysis" / "tables").mkdir(parents=True, exist_ok=True)
        manifest_path = self.exp_dir / "manifest.json"
        if manifest_path.exists():
            self._manifest = json.loads(manifest_path.read_text())
        else:
            self._manifest = {
                "slug": slug, "date": self.date, "hypothesis": hypothesis,
                "status": "running", "headline": "",
                "provenance": provenance or {}, "runs": [], "datasets": [],
                "seeds": [], "analysis": {"script": "analysis/analyze.py",
                                          "last_run": None, "code_sha": None},
            }
            self._write_manifest()
        self._upsert_index()
        md = self.exp_dir / "EXPERIMENT.md"
        if not md.exists():
            tmpl = (_TEMPLATE_DIR / "EXPERIMENT.md").read_text()
            md.write_text(tmpl.format(slug=slug, date=self.date,
                                      status=self._manifest["status"],
                                      hypothesis=hypothesis))

    def _write_manifest(self) -> None:
        write_json_atomic(self.exp_dir / "manifest.json", self._manifest)

    def _upsert_index(self) -> None:
        index = self.root / "index.jsonl"
        rows = []
        if index.exists():
            rows = [json.loads(l) for l in index.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r.get("slug") != self.slug]
        rows.append({"slug": self.slug, "date": self.date,
                     "hypothesis": self._manifest["hypothesis"],
                     "status": self._manifest["status"],
                     "headline": self._manifest["headline"],
                     "exp_dir": str(self.exp_dir)})
        self.root.mkdir(parents=True, exist_ok=True)
        index.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def add_run(self, run_id_or_dir: str) -> None:
        run_id = Path(run_id_or_dir).name
        if run_id not in self._manifest["runs"]:
            self._manifest["runs"].append(run_id)
            self._write_manifest()

    def append_raw(self, row: dict) -> None:
        validate_raw_row(row)
        append_jsonl(str(self.exp_dir / "raw" / "results.jsonl"), row)

    def set_headline(self, text: str, status: str) -> None:
        self._manifest["headline"] = text
        self._manifest["status"] = status
        self._write_manifest()
        self._upsert_index()


def rebuild_index_md(root: str = "experiments/results") -> Path:
    """Render the human-readable INDEX.md from index.jsonl."""
    root = Path(root)
    index = root / "index.jsonl"
    rows = []
    if index.exists():
        rows = [json.loads(l) for l in index.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r.get("date", ""), reverse=True)
    lines = ["# Experiment index", "",
             "| date | slug | status | headline |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r.get('date','')} | {r.get('slug','')} | "
                     f"{r.get('status','')} | {r.get('headline','')} |")
    out = root / "INDEX.md"
    out.write_text("\n".join(lines) + "\n")
    return out


__all__ = ["capture_provenance", "provenance_from_metadata",
           "RAW_SCHEMA", "validate_raw_row", "Experiment", "rebuild_index_md"]
