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


def bootstrap_ci(values: Any, statistic: Callable, *, resamples: int = 2000,
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


__all__ = ["entropy_bits", "normalized_concentration", "load_jsonl", "argmax",
           "mean_pairwise_l1", "bootstrap_ci", "sharpness_report",
           "write_report", "render_figure"]
