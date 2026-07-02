"""Merge and analyze B-sweep stability runs (publication-weight statistics).

``experiments/archive/b_sweep_stability.py`` writes one JSON per invocation, each containing
per-cell ``pi_mcts_per_run`` lists. Cells are keyed by (problem, B); runs of
the same cell collected in different invocations (different ``--base-seed``
ranges) are concatenated here, so a K=6 pilot cell plus a K=14 extension
merge into one K=20 cell.

Per merged cell this reports:

* K (number of independent searches from the same fixed initial state)
* sharpness: mean and std of max_a pi_MCTS(a|s); mean entropy (bits)
* modal-argmax agreement: fraction of runs whose recommended verb equals
  the modal recommended verb, with a bootstrap percentile CI
* pairwise-argmax agreement: probability two independent searches agree on
  the recommended verb (all K-choose-2 pairs), with a bootstrap CI --
  the primary stability statistic (smoother than modal agreement, and the
  quantity that matters operationally: "if I collect once, how likely is a
  second collection to produce the same training argmax?")
* mean pairwise L1 (kept for the sharpness-confound narrative)
* distinct argmax verbs across runs

Merging refuses to combine files whose sampling configuration differs
(model, temperature, max_tokens, c_puct, discount, depth_limit, max_turns).

Optionally (--figure PATH) renders the budget--stability figure: pairwise
agreement vs B with CI whiskers, and mean entropy vs B, per problem.

Usage::

    python experiments/b_sweep_analyze.py \
        /tmp/b_sweep.json /tmp/bsw_alg_a.json /tmp/bsw_alg_b.json ... \
        --output-json /tmp/b_sweep_merged.json \
        --figure paper/figures/b_sweep.pdf
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from typing import Any

import numpy as np

CONFIG_KEYS = (
    "model",
    "temperature",
    "max_tokens",
    "c_puct",
    "discount",
    "depth_limit",
    "max_turns",
)


def entropy_bits(dist: dict[str, float]) -> float:
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


def argmax_verb(dist: dict[str, float]) -> str:
    return max(dist.items(), key=lambda kv: kv[1])[0]


def mean_pairwise_l1(dists: list[dict[str, float]]) -> float:
    verbs = set().union(*dists)
    pairs = list(itertools.combinations(range(len(dists)), 2))
    if not pairs:
        return 0.0
    total = 0.0
    for i, j in pairs:
        total += sum(abs(dists[i].get(v, 0.0) - dists[j].get(v, 0.0)) for v in verbs)
    return total / len(pairs)


def pairwise_agreement(argmaxes: list[str]) -> float:
    """P(two independent searches recommend the same verb), over all pairs.

    Closed form from per-verb counts: sum_c n_c(n_c-1) / (K(K-1)).
    """
    k = len(argmaxes)
    if k < 2:
        return 1.0
    counts = Counter(argmaxes)
    return sum(n * (n - 1) for n in counts.values()) / (k * (k - 1))


def modal_agreement(argmaxes: list[str]) -> tuple[str, float]:
    modal, count = Counter(argmaxes).most_common(1)[0]
    return modal, count / len(argmaxes)


def bootstrap_ci(
    values: list[str],
    statistic,
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a statistic of the per-run argmax list."""
    rng = np.random.default_rng(seed)
    k = len(values)
    arr = np.array(values)
    stats = np.empty(resamples)
    for r in range(resamples):
        sample = arr[rng.integers(0, k, size=k)]
        stats[r] = statistic(list(sample))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def merge_files(paths: list[str]) -> tuple[dict, dict[tuple[str, int], list[dict]]]:
    """Return (reference_config, cell_runs) where cell_runs maps
    (problem, B) -> concatenated pi_mcts_per_run lists."""
    ref_config: dict | None = None
    cells: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        config = {k: data.get("args", {}).get(k) for k in CONFIG_KEYS}
        if ref_config is None:
            ref_config = config
        elif config != ref_config:
            diffs = {
                k: (ref_config[k], config[k])
                for k in CONFIG_KEYS
                if ref_config[k] != config[k]
            }
            raise SystemExit(
                f"{path}: sampling config differs from {paths[0]}: {diffs}; "
                f"refusing to merge incomparable sweeps"
            )
        for cell in data["cells"]:
            key = (cell["problem_key"], int(cell["B"]))
            cells[key].extend(cell["pi_mcts_per_run"])
    assert ref_config is not None
    return ref_config, dict(cells)


def analyze_cell(problem: str, b: int, dists: list[dict[str, float]]) -> dict[str, Any]:
    argmaxes = [argmax_verb(d) for d in dists]
    modal, modal_frac = modal_agreement(argmaxes)
    pair = pairwise_agreement(argmaxes)
    modal_ci = bootstrap_ci(argmaxes, lambda v: modal_agreement(v)[1])
    pair_ci = bootstrap_ci(argmaxes, pairwise_agreement)
    max_probs = [max(d.values()) for d in dists]
    return {
        "problem_key": problem,
        "B": b,
        "K": len(dists),
        "mean_max_prob": round(float(np.mean(max_probs)), 4),
        "std_max_prob": round(float(np.std(max_probs)), 4),
        "mean_entropy_bits": round(
            float(np.mean([entropy_bits(d) for d in dists])), 4
        ),
        "modal_argmax": modal,
        "modal_agreement": round(modal_frac, 4),
        "modal_agreement_ci95": [round(modal_ci[0], 4), round(modal_ci[1], 4)],
        "pairwise_agreement": round(pair, 4),
        "pairwise_agreement_ci95": [round(pair_ci[0], 4), round(pair_ci[1], 4)],
        "mean_pairwise_l1": round(mean_pairwise_l1(dists), 4),
        "distinct_argmax_verbs": len(set(argmaxes)),
        "argmax_counts": dict(Counter(argmaxes)),
    }


def render_figure(cells: list[dict[str, Any]], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    problems = sorted({c["problem_key"] for c in cells})
    colors = {"multiply_small": "#3F88C5", "algebra_toy_1": "#D2691E"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9))

    for prob in problems:
        rows = sorted((c for c in cells if c["problem_key"] == prob), key=lambda c: c["B"])
        bs = [c["B"] for c in rows]
        agree = [c["pairwise_agreement"] for c in rows]
        lo = [c["pairwise_agreement"] - c["pairwise_agreement_ci95"][0] for c in rows]
        hi = [c["pairwise_agreement_ci95"][1] - c["pairwise_agreement"] for c in rows]
        color = colors.get(prob, None)
        ax1.errorbar(
            bs, agree, yerr=[lo, hi], marker="o", capsize=3,
            label=prob.replace("_", r"\_"), color=color,
        )
        ent = [c["mean_entropy_bits"] for c in rows]
        ax2.plot(bs, ent, marker="o", label=prob.replace("_", r"\_"), color=color)

    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xlabel(r"simulation budget $B$")
        ax.legend(fontsize=8, frameon=False)
    ax1.set_ylabel("pairwise argmax agreement")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("(A) Target stability vs budget", fontsize=10)
    ax2.set_ylabel(r"mean entropy of $\pi_{\mathrm{MCTS}}$ (bits)")
    ax2.set_title("(B) Target sharpness vs budget", fontsize=10)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote figure {path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", help="b_sweep_stability JSON outputs to merge")
    p.add_argument("--output-json", default=None)
    p.add_argument("--figure", default=None, help="optional output PDF path")
    args = p.parse_args(argv)

    config, cell_runs = merge_files(args.inputs)
    print(f"Merged {len(args.inputs)} files; config: {config}")

    analyzed = [
        analyze_cell(prob, b, dists)
        for (prob, b), dists in sorted(cell_runs.items())
    ]

    print()
    print("=" * 100)
    print("B-SWEEP MERGED ANALYSIS")
    print("=" * 100)
    print(
        f"{'problem':<16} {'B':>4} {'K':>3} {'maxP':>6} {'H':>6} "
        f"{'modal':>22} {'pairAgree [CI95]':>22} {'L1':>6} {'#verbs':>6}"
    )
    print("-" * 100)
    for c in analyzed:
        modal_str = f"{c['modal_argmax'][:12]}({c['modal_agreement']:.2f})"
        pair_str = (
            f"{c['pairwise_agreement']:.2f} "
            f"[{c['pairwise_agreement_ci95'][0]:.2f},"
            f"{c['pairwise_agreement_ci95'][1]:.2f}]"
        )
        print(
            f"{c['problem_key']:<16} {c['B']:>4} {c['K']:>3} "
            f"{c['mean_max_prob']:>6.2f} {c['mean_entropy_bits']:>6.2f} "
            f"{modal_str:>22} {pair_str:>22} "
            f"{c['mean_pairwise_l1']:>6.3f} {c['distinct_argmax_verbs']:>6}"
        )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"config": config, "cells": analyzed}, f, indent=2)
        print(f"\nwrote {args.output_json}")

    if args.figure:
        render_figure(analyzed, args.figure)

    return 0


if __name__ == "__main__":
    sys.exit(main())
