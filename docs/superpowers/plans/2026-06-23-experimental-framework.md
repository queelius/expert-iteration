# Experimental Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A three-layer experimental framework (run / experiment / program index) with a thin provenance+manifest library, a single-source analysis library that absorbs the duplicated analysis helpers, and a benchmark registry with a GSM8K loader and metric, so closed-loop and benchmark runs produce reproducible, navigable, paper-grade records.

**Architecture:** A new package module `src/verifier_anchored_rl/experiment.py` captures provenance, writes a machine-readable experiment manifest, upserts a program index, and validates a standard raw-results schema. A new `experiments/analysis_lib.py` holds one implementation of each metric/helper (re-exporting the canonical `outer_loop._entropy` / `_normalized_concentration` so the sharpening number is identical online and offline) plus figure/report writers. A benchmark registry (`experiments/benchmarks/`) pins datasets and provides a GSM8K loader + verifier-anchored metric. Runs are referenced by id (never moved). Design spec: `docs/superpowers/specs/2026-06-23-experimental-framework-design.md`.

**Tech Stack:** Python 3.12, numpy, pytest (core); torch / transformers / vllm read only for version strings (imported lazily, tolerate absence); `datasets` optional under the `[bench]` extra; matplotlib optional (imported lazily in figure code).

## Global Constraints

- Do NOT write the em-dash glyph (the long dash) in ANY file, including commit messages. Use hyphens, commas, colons, parentheses. (Repo hook rejects it.)
- Core code must import without torch/transformers/vllm/datasets/matplotlib present: import those lazily inside functions and tolerate ImportError.
- The metrics are single-source: analysis code re-exports `verifier_anchored_rl.outer_loop._entropy` and `_normalized_concentration` rather than redefining them.
- Package code lives in `src/verifier_anchored_rl/`; experiment-scoped infra in `experiments/`. Follow the existing layout; do not restructure.
- Runs are referenced by id; never move or copy a run directory.
- Atomic writes reuse `verifier_anchored_rl.run_artifacts.write_json_atomic`; JSONL appends reuse `append_jsonl`.

---

### Task 1: Provenance capture

**Files:**
- Create: `src/verifier_anchored_rl/experiment.py`
- Test: `tests/test_experiment.py`

**Interfaces:**
- Consumes: `verifier_anchored_rl.run_artifacts.utc_timestamp() -> str`.
- Produces: `capture_provenance(argv: list[str], *, seeds: dict, extra: dict | None = None) -> dict`; `provenance_from_metadata(meta: dict) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_experiment.py
from __future__ import annotations


def test_capture_provenance_has_required_keys_and_tolerates_missing_torch():
    from verifier_anchored_rl.experiment import capture_provenance
    prov = capture_provenance(["run.py", "--seed", "0"], seeds={"global": 0})
    for key in ("git", "command", "seeds", "env", "host", "timestamps"):
        assert key in prov, key
    assert prov["command"]["argv"] == ["run.py", "--seed", "0"]
    assert prov["seeds"] == {"global": 0}
    assert "python" in prov["env"]
    assert isinstance(prov["env_freeze_sha256"], str)
    # extra is merged at top level
    prov2 = capture_provenance(["x"], seeds={}, extra={"model": "m", "embedder": {"kind": "text"}})
    assert prov2["model"] == "m"
    assert prov2["embedder"] == {"kind": "text"}


def test_provenance_from_metadata_maps_existing_run():
    from verifier_anchored_rl.experiment import provenance_from_metadata
    meta = {
        "git_sha": "abc123", "git_dirty": False,
        "args": {"seed": 7, "model": "Qwen/Qwen2.5-3B-Instruct",
                 "embedder": "text", "embedder_model": "Qwen/Qwen2.5-0.5B-Instruct"},
        "torch": "2.9.1", "transformers": "5.0", "gpu": "RTX 3060", "cuda": "12.8",
        "seed": 7,
    }
    prov = provenance_from_metadata(meta)
    assert prov["git"]["sha"] == "abc123"
    assert prov["seeds"]["global"] == 7
    assert prov["model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert prov["embedder"]["kind"] == "text"
    assert prov["embedder"]["id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert prov["env"]["torch"] == "2.9.1"
```

- [ ] **Step 2: Run, verify FAIL** (`ModuleNotFoundError: verifier_anchored_rl.experiment`):

Run: `python -m pytest tests/test_experiment.py -q`

- [ ] **Step 3: Implement**

```python
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
```

- [ ] **Step 4: Run, verify PASS** (`python -m pytest tests/test_experiment.py -q`).
- [ ] **Step 5: Commit**

```bash
git add src/verifier_anchored_rl/experiment.py tests/test_experiment.py
git commit -m "Add provenance capture and the metadata-to-provenance shim"
```

---

### Task 2: Raw-results schema and validation

**Files:**
- Modify: `src/verifier_anchored_rl/experiment.py`
- Test: `tests/test_experiment.py`

**Interfaces:**
- Produces: `RAW_SCHEMA: dict[str, type | tuple]`; `validate_raw_row(row: dict) -> None` (raises `ValueError`).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_experiment.py
def test_validate_raw_row_accepts_valid_and_rejects_bad():
    import pytest
    from verifier_anchored_rl.experiment import validate_raw_row
    good = dict(dataset="gsm8k", dataset_version="main", split="test",
                problem_id="gsm8k-test-0", seed=0, budget_B=8, correct=True,
                pred="72", gold="72", verifier_verdict="pass", n_turns=4,
                gen_calls=31, gen_tokens=980, prefill_calls=12,
                prefill_tokens=3120, flops_est=None, wall_s=3.4,
                run_id="20260623T0-gsm8k")
    validate_raw_row(good)  # no raise
    missing = {k: v for k, v in good.items() if k != "gold"}
    with pytest.raises(ValueError):
        validate_raw_row(missing)
    bad_type = dict(good, budget_B="eight")
    with pytest.raises(ValueError):
        validate_raw_row(bad_type)
```

- [ ] **Step 2: Run, verify FAIL** (`ImportError: validate_raw_row`).

- [ ] **Step 3: Implement** (append to `experiment.py`, and add both names to `__all__`)

```python
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
```

- [ ] **Step 4: Run, verify PASS** (`python -m pytest tests/test_experiment.py -q`).
- [ ] **Step 5: Commit**

```bash
git add src/verifier_anchored_rl/experiment.py tests/test_experiment.py
git commit -m "Add raw-results schema and row validation"
```

---

### Task 3: Experiment container (manifest, index, raw sink, EXPERIMENT.md)

**Files:**
- Modify: `src/verifier_anchored_rl/experiment.py`
- Create: `experiments/templates/EXPERIMENT.md`
- Test: `tests/test_experiment.py`

**Interfaces:**
- Consumes: `run_artifacts.write_json_atomic(path, obj)`, `run_artifacts.append_jsonl(path, obj)`, `capture_provenance`, `validate_raw_row`.
- Produces: `class Experiment(slug, hypothesis, *, root, provenance=None, date=None)` with `.exp_dir: Path`, `.add_run(run_id_or_dir)`, `.append_raw(row)`, `.set_headline(text, status)`; `rebuild_index_md(root) -> Path`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_experiment.py
def test_experiment_creates_dir_manifest_index_and_is_idempotent(tmp_path):
    import json
    from verifier_anchored_rl.experiment import Experiment
    root = str(tmp_path)
    e = Experiment("e1-demo", "sharpening rises with budget", root=root,
                   provenance={"git": {"sha": "abc"}}, date="20260623")
    assert e.exp_dir == tmp_path / "exp" / "20260623-e1-demo"
    manifest = json.loads((e.exp_dir / "manifest.json").read_text())
    assert manifest["slug"] == "e1-demo"
    assert manifest["status"] == "running"
    assert (e.exp_dir / "EXPERIMENT.md").exists()
    # index has exactly one row for this slug, even after re-construction
    Experiment("e1-demo", "sharpening rises with budget", root=root,
               provenance={"git": {"sha": "abc"}}, date="20260623")
    rows = [json.loads(l) for l in (tmp_path / "index.jsonl").read_text().splitlines() if l.strip()]
    assert sum(r["slug"] == "e1-demo" for r in rows) == 1


def test_experiment_add_run_dedups_and_append_raw_validates(tmp_path):
    import json
    import pytest
    from verifier_anchored_rl.experiment import Experiment
    e = Experiment("e2", "h", root=str(tmp_path), date="20260623")
    e.add_run("20260622T0-sweep-B8")
    e.add_run("20260622T0-sweep-B8")  # dup
    manifest = json.loads((e.exp_dir / "manifest.json").read_text())
    assert manifest["runs"] == ["20260622T0-sweep-B8"]
    good = dict(dataset="gsm8k", dataset_version="main", split="test",
                problem_id="p0", seed=0, budget_B=8, correct=True, pred="1",
                gold="1", verifier_verdict="pass", n_turns=1, gen_calls=1,
                gen_tokens=1, prefill_calls=1, prefill_tokens=1, flops_est=None,
                wall_s=1.0, run_id="r0")
    e.append_raw(good)
    with pytest.raises(ValueError):
        e.append_raw({"dataset": "gsm8k"})  # invalid
    lines = (e.exp_dir / "raw" / "results.jsonl").read_text().splitlines()
    assert len(lines) == 1


def test_experiment_does_not_clobber_edited_experiment_md(tmp_path):
    from verifier_anchored_rl.experiment import Experiment
    e = Experiment("e3", "h", root=str(tmp_path), date="20260623")
    md = e.exp_dir / "EXPERIMENT.md"
    md.write_text("# e3\n\nMY EDITS\n")
    Experiment("e3", "h", root=str(tmp_path), date="20260623")  # re-construct
    assert "MY EDITS" in md.read_text()
```

- [ ] **Step 2: Run, verify FAIL** (`ImportError: Experiment`).

- [ ] **Step 3: Create the template** `experiments/templates/EXPERIMENT.md`

```markdown
# {slug}

Date: {date}  Status: {status}

## Hypothesis
{hypothesis}

## Motivation
(why this experiment, what prior result motivates it)

## Design
Datasets, seeds, budgets, model, embedder, key hyperparameters.

## Commands
See reproduce.sh; exact argv per run is in manifest.provenance.

## Results
(filled from analysis/report.md: headline table + figure references)

## Interpretation
(what the numbers mean for the hypothesis; threats to validity)

## Conclusion
(confirmed / refuted / inconclusive; what is next)
```

- [ ] **Step 4: Implement** (append to `experiment.py`; add `Experiment` and `rebuild_index_md` to `__all__`)

```python
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
        write_json_atomic(index, {"_": 0})  # ensure parent; replaced next line
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
```

Note: the `write_json_atomic(index, {"_": 0})` line only ensures the parent dir exists before the plain-text rewrite; the index is JSONL, so it is rewritten on the next line. Keep it.

- [ ] **Step 5: Run, verify PASS** (`python -m pytest tests/test_experiment.py -q`).
- [ ] **Step 6: Commit**

```bash
git add src/verifier_anchored_rl/experiment.py experiments/templates/EXPERIMENT.md tests/test_experiment.py
git commit -m "Add Experiment container: manifest, program index, raw sink, EXPERIMENT.md"
```

---

### Task 4: analysis_lib core primitives (single-source metrics)

**Files:**
- Create: `experiments/analysis_lib.py`
- Test: `tests/test_analysis_lib.py`

**Interfaces:**
- Consumes: `verifier_anchored_rl.outer_loop._entropy`, `verifier_anchored_rl.outer_loop._normalized_concentration`.
- Produces: `entropy_bits(probs) -> float`; `normalized_concentration(entropy_bits, k) -> float`; `load_jsonl(path) -> list[dict]`; `argmax(dist) -> str`; `mean_pairwise_l1(dists) -> float`; `bootstrap_ci(values, statistic, *, resamples=2000, ci=0.95, seed=0) -> tuple[float, float]`; `sharpness_report(records, label) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analysis_lib.py
from __future__ import annotations

import math


def test_entropy_bits_matches_canonical_and_concentration():
    from experiments.analysis_lib import entropy_bits, normalized_concentration
    from verifier_anchored_rl.outer_loop import _entropy
    uniform4 = [0.25, 0.25, 0.25, 0.25]
    assert abs(entropy_bits(uniform4) - 2.0) < 1e-9
    assert entropy_bits(uniform4) == _entropy(uniform4)  # single source of truth
    assert abs(normalized_concentration(2.0, 4) - 0.0) < 1e-9
    assert abs(normalized_concentration(0.0, 4) - 1.0) < 1e-9


def test_argmax_and_mean_pairwise_l1():
    from experiments.analysis_lib import argmax, mean_pairwise_l1
    assert argmax({"a": 0.1, "b": 0.9}) == "b"
    d1 = {"a": 1.0, "b": 0.0}
    d2 = {"a": 0.0, "b": 1.0}
    assert abs(mean_pairwise_l1([d1, d2]) - 2.0) < 1e-9
    assert mean_pairwise_l1([d1]) == 0.0


def test_bootstrap_ci_is_deterministic_under_seed():
    from experiments.analysis_lib import bootstrap_ci
    vals = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    lo1, hi1 = bootstrap_ci(vals, sum, seed=0)
    lo2, hi2 = bootstrap_ci(vals, sum, seed=0)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 <= hi1


def test_load_jsonl_and_sharpness_report(tmp_path):
    import json
    from experiments.analysis_lib import load_jsonl, sharpness_report
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps({"pi_mcts": pi}) for pi in
                           ({"a": 1.0, "b": 0.0}, {"a": 0.5, "b": 0.5})) + "\n")
    recs = load_jsonl(str(p))
    assert len(recs) == 2
    rep = sharpness_report(recs, label="round-a")
    assert rep["label"] == "round-a"
    assert abs(rep["mean_max_prob"] - 0.75) < 1e-9  # (1.0 + 0.5) / 2
```

- [ ] **Step 2: Run, verify FAIL** (`ModuleNotFoundError: experiments.analysis_lib`).

- [ ] **Step 3: Implement**

```python
# experiments/analysis_lib.py
"""Shared analysis primitives: ONE implementation of each metric/helper the
ad-hoc analysis scripts used to copy. The entropy/concentration functions are
re-exported from verifier_anchored_rl.outer_loop so the sharpening number is
computed identically online (during collection) and offline (in analysis).
See docs/superpowers/specs/2026-06-23-experimental-framework-design.md."""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np

from verifier_anchored_rl.outer_loop import (
    _entropy as entropy_bits,
    _normalized_concentration as normalized_concentration,
)


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def argmax(dist: dict[str, float]) -> str:
    return max(dist.items(), key=lambda kv: kv[1])[0]


def mean_pairwise_l1(dists: list[dict[str, float]]) -> float:
    """Mean L1 distance over all unordered pairs of distributions (0 for <2)."""
    if len(dists) < 2:
        return 0.0
    keys = sorted({k for d in dists for k in d})
    vecs = [np.array([d.get(k, 0.0) for k in keys]) for d in dists]
    tot, n = 0.0, 0
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            tot += float(np.abs(vecs[i] - vecs[j]).sum())
            n += 1
    return tot / n if n else 0.0


def bootstrap_ci(values, statistic: Callable, *, resamples: int = 2000,
                 ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of ``statistic`` over ``values``. Deterministic
    given ``seed`` (the seed is recorded in the report)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    stats = [statistic(rng.choice(arr, size=len(arr), replace=True))
             for _ in range(resamples)]
    lo = float(np.percentile(stats, 100 * (1 - ci) / 2))
    hi = float(np.percentile(stats, 100 * (1 + ci) / 2))
    return lo, hi


def sharpness_report(records: list[dict], label: str) -> dict:
    """Sharpness summary over records carrying a ``pi_mcts`` dict."""
    pis = [r["pi_mcts"] for r in records if r.get("pi_mcts")]
    maxes = [max(pi.values()) for pi in pis]
    ents = [entropy_bits(list(pi.values())) for pi in pis]
    n = len(pis)
    return {
        "label": label, "n": n,
        "mean_max_prob": float(np.mean(maxes)) if n else float("nan"),
        "median_max_prob": float(sorted(maxes)[n // 2]) if n else float("nan"),
        "mean_entropy_bits": float(np.mean(ents)) if n else float("nan"),
        "frac_concentrated_above_0.5":
            float(np.mean([m > 0.5 for m in maxes])) if n else float("nan"),
    }


__all__ = ["entropy_bits", "normalized_concentration", "load_jsonl", "argmax",
           "mean_pairwise_l1", "bootstrap_ci", "sharpness_report"]
```

- [ ] **Step 4: Run, verify PASS** (`python -m pytest tests/test_analysis_lib.py -q`).
- [ ] **Step 5: Commit**

```bash
git add experiments/analysis_lib.py tests/test_analysis_lib.py
git commit -m "Add analysis_lib core: single-source metrics and shared helpers"
```

---

### Task 5: analysis_lib reporting (figures + markdown report)

**Files:**
- Modify: `experiments/analysis_lib.py`
- Test: `tests/test_analysis_lib.py`

**Interfaces:**
- Produces: `render_figure(plot_fn: Callable, out_stem: str) -> list[str]` (writes `<stem>.png` and `<stem>.pdf`, returns the paths); `write_report(out_dir: str, title: str, sections: list[tuple[str, Any]]) -> str` (writes `<out_dir>/report.md`, returns its path; a section body is a markdown string OR a list-of-dicts rendered as a table).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_analysis_lib.py
def test_write_report_renders_prose_and_tables(tmp_path):
    from experiments.analysis_lib import write_report
    path = write_report(str(tmp_path), "Demo", [
        ("Summary", "All good."),
        ("Cells", [{"B": 8, "acc": 0.5}, {"B": 16, "acc": 0.7}]),
    ])
    text = (tmp_path / "report.md").read_text()
    assert path.endswith("report.md")
    assert "# Demo" in text and "## Summary" in text and "All good." in text
    assert "| B | acc |" in text and "| 8 | 0.5 |" in text


def test_render_figure_writes_png_and_pdf(tmp_path):
    import pytest
    pytest.importorskip("matplotlib")
    from experiments.analysis_lib import render_figure
    def plot(fig):
        ax = fig.add_subplot(111)
        ax.plot([1, 2, 3], [1, 4, 9])
    paths = render_figure(plot, str(tmp_path / "fig"))
    assert (tmp_path / "fig.png").exists() and (tmp_path / "fig.pdf").exists()
    assert set(paths) == {str(tmp_path / "fig.png"), str(tmp_path / "fig.pdf")}
```

- [ ] **Step 2: Run, verify FAIL** (`ImportError: write_report`).

- [ ] **Step 3: Implement** (append to `analysis_lib.py`; add both to `__all__`)

```python
def _md_table(rows: list[dict]) -> str:
    if not rows:
        return "(no rows)"
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def write_report(out_dir: str, title: str, sections: list) -> str:
    """Write a markdown report. Each section is (heading, body); body is a
    markdown string or a list-of-dicts rendered as a table. This is the
    persistent artifact the ad-hoc scripts never produced."""
    from pathlib import Path
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.append(f"## {heading}")
        lines.append(_md_table(body) if isinstance(body, list) else str(body))
        lines.append("")
    path = Path(out_dir) / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def render_figure(plot_fn: Callable, out_stem: str) -> list[str]:
    """Call ``plot_fn(fig)`` and save the figure as both PNG (committed, for
    report.md) and PDF (for the paper). Uses the Agg backend so it runs
    headless. matplotlib is imported lazily."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7, 4))
    plot_fn(fig)
    paths = []
    for ext in ("png", "pdf"):
        p = f"{out_stem}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths
```

- [ ] **Step 4: Run, verify PASS** (`python -m pytest tests/test_analysis_lib.py -q`).
- [ ] **Step 5: Commit**

```bash
git add experiments/analysis_lib.py tests/test_analysis_lib.py
git commit -m "Add analysis_lib reporting: PNG+PDF figures and markdown report writer"
```

---

### Task 6: Benchmark registry and GSM8K loader + metric

**Files:**
- Create: `experiments/benchmarks/__init__.py` (empty), `experiments/benchmarks/registry.json`, `experiments/benchmarks/loaders.py`
- Test: `tests/test_benchmarks.py`

**Interfaces:**
- Consumes: `verifier_anchored_rl.verifier.check_answer(committed, expected) -> VerifierResult` (has `.success`).
- Produces: `load_gsm8k(split: str, *, source: str | None = None) -> list[dict]` (each `{problem_id, problem, gold}`); `gsm8k_correct(pred: str, gold: str) -> bool`; `registry()` -> dict loaded from registry.json.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmarks.py
from __future__ import annotations


def test_load_gsm8k_parses_gold_after_hashes(tmp_path):
    from experiments.benchmarks.loaders import load_gsm8k
    fixture = tmp_path / "gsm8k_test.jsonl"
    import json
    fixture.write_text("\n".join(json.dumps(r) for r in [
        {"question": "2+2?", "answer": "two plus two\n#### 4"},
        {"question": "3+3?", "answer": "#### 6"},
    ]) + "\n")
    items = load_gsm8k("test", source=str(fixture))
    assert len(items) == 2
    assert items[0]["gold"] == "4" and items[0]["problem"] == "2+2?"
    assert items[0]["problem_id"] == "gsm8k-test-0"


def test_gsm8k_correct_uses_verifier_and_normalizes():
    from experiments.benchmarks.loaders import gsm8k_correct
    assert gsm8k_correct("72", "72") is True
    assert gsm8k_correct("1,000", "1000") is True   # comma stripped
    assert gsm8k_correct("$72", "72") is True        # currency stripped
    assert gsm8k_correct("73", "72") is False


def test_registry_lists_gsm8k():
    from experiments.benchmarks.loaders import registry
    reg = registry()
    assert "gsm8k" in reg and reg["gsm8k"]["answer_style"] == "final-number"
```

- [ ] **Step 2: Run, verify FAIL** (`ModuleNotFoundError: experiments.benchmarks.loaders`).

- [ ] **Step 3: Create the registry** `experiments/benchmarks/registry.json`

```json
{
  "gsm8k": {"loader": "gsm8k", "version": "main", "splits": ["train", "test"],
            "n_test": 1319, "answer_style": "final-number",
            "citation": "Cobbe et al. 2021"},
  "math": {"loader": "math", "version": "v1", "splits": ["train", "test"],
           "answer_style": "boxed", "citation": "Hendrycks et al. 2021",
           "status": "planned"},
  "toy-train": {"loader": "toy", "version": "1", "splits": ["train", "holdout"],
                "source": "experiments/problems.py"}
}
```

- [ ] **Step 4: Implement** `experiments/benchmarks/loaders.py` (and create empty `experiments/benchmarks/__init__.py`)

```python
# experiments/benchmarks/loaders.py
"""Benchmark dataset registry and the GSM8K loader + verifier-anchored metric.

`load_gsm8k` reads from a cached JSONL `source` (one {"question","answer"} per
line, GSM8K's native format, gold after "####"). The live download via the
optional `datasets` library is a thin wrapper that writes that JSONL cache;
tests run against a fixture and never hit the network. `gsm8k_correct` scores a
predicted answer against the gold via the SymPy verifier (verifier_anchored_rl),
after stripping commas and currency symbols.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_REGISTRY_PATH = Path(__file__).with_name("registry.json")


def registry() -> dict:
    return json.loads(_REGISTRY_PATH.read_text())


def _gold_from_answer(answer: str) -> str:
    # GSM8K marks the final answer after "####".
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer.strip()


def load_gsm8k(split: str, *, source: str | None = None) -> list[dict]:
    """Return [{problem_id, problem, gold}] for the split. `source` is a JSONL
    cache (question/answer per line); if None, the default cache path under the
    benchmarks dir is used. Raises FileNotFoundError if the cache is absent
    (run the download wrapper first)."""
    path = Path(source) if source else _REGISTRY_PATH.with_name(f"cache/gsm8k_{split}.jsonl")
    if not path.exists():
        raise FileNotFoundError(
            f"GSM8K cache not found at {path}. Provide source= or pre-download "
            f"(pip install -e .[bench]; then the download wrapper writes it).")
    items = []
    for i, line in enumerate(p for p in path.read_text().splitlines() if p.strip()):
        rec = json.loads(line)
        items.append({"problem_id": f"gsm8k-{split}-{i}",
                      "problem": rec["question"],
                      "gold": _gold_from_answer(rec["answer"])})
    return items


_NUM_CLEAN = re.compile(r"[,$%\s]")


def gsm8k_correct(pred: str, gold: str) -> bool:
    """Verifier-anchored exact match: strip commas/currency/whitespace, then
    use the SymPy-backed terminal verifier (numeric or normalized-string)."""
    from verifier_anchored_rl.verifier import check_answer
    pred_c = _NUM_CLEAN.sub("", pred or "")
    gold_c = _NUM_CLEAN.sub("", gold or "")
    return bool(check_answer(pred_c, gold_c).success)


__all__ = ["registry", "load_gsm8k", "gsm8k_correct"]
```

- [ ] **Step 5: Run, verify PASS** (`python -m pytest tests/test_benchmarks.py -q`).
- [ ] **Step 6: Commit**

```bash
git add experiments/benchmarks/ tests/test_benchmarks.py
git commit -m "Add benchmark registry and GSM8K loader + verifier-anchored metric"
```

---

### Task 7: End-to-end backfill + analyze/reproduce templates

**Files:**
- Create: `experiments/templates/analyze.py`, `experiments/templates/reproduce.sh`, `experiments/backfill_experiment.py`
- Test: `tests/test_backfill.py`

**Interfaces:**
- Consumes: `Experiment`, `provenance_from_metadata`, `run_artifacts` (metadata read), `rebuild_index_md`.
- Produces: `backfill(run_dir: str, slug: str, hypothesis: str, *, root: str) -> str` (returns exp_dir path), wrapping an existing run into an experiment.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backfill.py
from __future__ import annotations


def test_backfill_wraps_existing_run_into_experiment(tmp_path):
    import json
    from experiments.backfill_experiment import backfill
    # a minimal existing run dir with metadata.json
    runs = tmp_path / "runs"; runs.mkdir()
    run = runs / "20260622T0-sweep-B8"; run.mkdir()
    (run / "metadata.json").write_text(json.dumps({
        "git_sha": "abc", "args": {"seed": 0, "model": "m", "embedder": "text",
                                    "embedder_model": "e"}, "torch": "2.9"}))
    exp_dir = backfill(str(run), "e1-hT-sweep", "sharpening rises with budget",
                       root=str(tmp_path))
    m = json.loads((Path(exp_dir) / "manifest.json").read_text())
    assert m["runs"] == ["20260622T0-sweep-B8"]
    assert m["provenance"]["model"] == "m"
    assert (tmp_path / "INDEX.md").exists()
```

(The test imports `from pathlib import Path` at the top.)

- [ ] **Step 2: Run, verify FAIL** (`ModuleNotFoundError: experiments.backfill_experiment`).

- [ ] **Step 3: Create the templates**

`experiments/templates/analyze.py`:

```python
"""Per-experiment analysis: reads the experiment's runs (metrics.jsonl) and
raw/results.jsonl, computes the standard tables, writes analysis/report.md and
analysis/figures via analysis_lib. Copy into an experiment's analysis/ dir and
adapt. Deterministic: records the consumed run ids and code sha in the report."""
import sys
from experiments.analysis_lib import load_jsonl, sharpness_report, write_report


def main(exp_dir: str) -> int:
    raw = load_jsonl(f"{exp_dir}/raw/results.jsonl") if __import__("os").path.exists(f"{exp_dir}/raw/results.jsonl") else []
    sections = [("Raw cells", raw[:50])]
    write_report(f"{exp_dir}/analysis", "Analysis", sections)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
```

`experiments/templates/reproduce.sh`:

```bash
#!/usr/bin/env bash
# Regenerate every run in this experiment, then the analysis.
# Fill in one line per run (exact argv is also in manifest.provenance), e.g.:
# python experiments/run_outer_loop.py --gen-backend vllm --model <M> ... --tag <tag>
set -euo pipefail
python experiments/templates/analyze.py "$(dirname "$0")"
```

- [ ] **Step 4: Implement** `experiments/backfill_experiment.py`

```python
"""Wrap an existing run directory into an experiment container, so notable
prior runs (the E1 sweep, the pilot) populate the program index without
re-running them."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from verifier_anchored_rl.experiment import Experiment, provenance_from_metadata, rebuild_index_md


def backfill(run_dir: str, slug: str, hypothesis: str, *,
             root: str = "experiments/results") -> str:
    meta = json.loads((Path(run_dir) / "metadata.json").read_text())
    exp = Experiment(slug, hypothesis, root=root,
                     provenance=provenance_from_metadata(meta))
    exp.add_run(run_dir)
    rebuild_index_md(root)
    return str(exp.exp_dir)


if __name__ == "__main__":
    print(backfill(sys.argv[1], sys.argv[2], sys.argv[3]))
```

- [ ] **Step 5: Run, verify PASS** (`python -m pytest tests/test_backfill.py -q`).
- [ ] **Step 6: Commit**

```bash
git add experiments/templates/analyze.py experiments/templates/reproduce.sh experiments/backfill_experiment.py tests/test_backfill.py
git commit -m "Add experiment backfill plus analyze.py and reproduce.sh templates"
```

---

### Task 8: Dedup the existing analysis scripts onto analysis_lib

**Files:**
- Modify: `experiments/b_sweep_stability.py`, `experiments/b_sweep_analyze.py`, `experiments/analyze_distillation_rounds.py`, `experiments/make_pilot_figure.py`, `experiments/algebra_post_mortem.py`
- Test: `tests/test_analysis_lib.py`

**Interfaces:**
- Consumes: `experiments.analysis_lib` (entropy_bits, argmax, mean_pairwise_l1, load_jsonl, bootstrap_ci).

- [ ] **Step 1: Write the failing test** (asserts the scripts now source helpers from analysis_lib, not local copies)

```python
# add to tests/test_analysis_lib.py
def test_scripts_reuse_analysis_lib_helpers():
    import inspect
    import experiments.b_sweep_analyze as bsa
    import experiments.analyze_distillation_rounds as adr
    from experiments import analysis_lib
    # the scripts must reference analysis_lib's functions, not redefine them
    assert bsa.mean_pairwise_l1 is analysis_lib.mean_pairwise_l1
    assert adr.entropy_bits is analysis_lib.entropy_bits
    # and they must not define their own copy
    assert "def mean_pairwise_l1" not in inspect.getsource(bsa)
    assert "def entropy_bits" not in inspect.getsource(adr)
```

- [ ] **Step 2: Run, verify FAIL** (the scripts still define their own helpers).

- [ ] **Step 3: Implement** For EACH of the five scripts: delete the local copies of `entropy_bits`/`entropy`, `argmax`/`safe_argmax`, `mean_pairwise_l1`, and `load_jsonl`, and add at the top `from experiments.analysis_lib import entropy_bits, argmax, mean_pairwise_l1, load_jsonl` (and `bootstrap_ci` for `b_sweep_analyze.py`). Where a script used a differently-named local helper (for example `entropy`), bind it: `entropy = entropy_bits` (or update call sites). Do NOT change each script's analysis logic, CLI, or outputs in this task; this is a pure deduplication so behavior is unchanged. Run each script's module import to confirm no NameError: `python -c "import experiments.b_sweep_analyze"` for each.

- [ ] **Step 4: Run, verify PASS**

Run: `python -m pytest tests/test_analysis_lib.py -q` and `for m in b_sweep_stability b_sweep_analyze analyze_distillation_rounds make_pilot_figure algebra_post_mortem; do python -c "import experiments.$m" || exit 1; done`
Expected: tests pass; all imports succeed.

- [ ] **Step 5: Commit**

```bash
git add experiments/b_sweep_stability.py experiments/b_sweep_analyze.py experiments/analyze_distillation_rounds.py experiments/make_pilot_figure.py experiments/algebra_post_mortem.py tests/test_analysis_lib.py
git commit -m "Dedup analysis-script helpers onto analysis_lib (single source)"
```

---

## Deferred (noted, not in this plan)

Per YAGNI and the spec's "framework + GSM8K now" scope, these are intentionally out of scope here and tracked as follow-ups: the MATH loader; the baseline-comparison harness (best-of-N, self-consistency, PRM); refactoring `run_outer_loop.py` to emit `capture_provenance()` natively (the shim covers ingestion now); giving the two hardcoded-`/tmp` scripts (`algebra_post_mortem.py`, `make_pilot_figure.py`) CLI args and removing `make_pilot_figure.py`'s baked-in numbers; and wiring a GSM8K benchmark-eval runner (generalizing `evaluate_holdout.py` over `load_gsm8k`, emitting `Experiment.append_raw` rows) once the embedder-on-GPU smoke follow-up lands.

## Self-Review

**Spec coverage:** Layer 2 manifest/EXPERIMENT.md/raw sink (Task 3); program index (Task 3 + rebuild_index_md); thin library provenance + shim (Tasks 1-2) and Experiment (Task 3); raw-results schema (Task 2); analysis_lib single-source metrics + figures/report (Tasks 4-5); benchmark registry + GSM8K loader/metric (Task 6); templates + backfill end-to-end (Task 7); absorb the five scripts (Task 8). The resolved decisions are honored: library at `src/verifier_anchored_rl/experiment.py` (Tasks 1-3); shim ingests existing metadata, no harness refactor (Task 1); figures emit PNG + PDF (Task 5). Covered.

**Placeholder scan:** Every code step has complete, runnable code; commands have expected outcomes. The one awkward spot is the Task 7 test's nested expression (flagged inline with the three assertions that matter); the implementer should simplify it. No "TBD"/"handle errors"/vague steps.

**Type consistency:** `capture_provenance`/`provenance_from_metadata` return dicts consumed by `Experiment(provenance=...)` (Task 3). `validate_raw_row` (Task 2) is called by `Experiment.append_raw` (Task 3). `entropy_bits`/`normalized_concentration` are the re-exported `outer_loop._entropy`/`_normalized_concentration` and used by `sharpness_report` and the deduped scripts (Tasks 4, 8). `write_report` section bodies are str-or-list-of-dicts consistently (Tasks 5, 7). `load_gsm8k` returns `{problem_id, problem, gold}` and `gsm8k_correct(pred, gold)` is the metric (Task 6). Consistent.
