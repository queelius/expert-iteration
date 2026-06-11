"""Generate the headline figure for the paper's pilot section.

Two-panel layout:

  Panel A: Sharpness across pilot rounds (mean max prob bars + entropy text).
           Shows v2, v3-train, v3-holdout, v4-train, v4-holdout.

  Panel B: Per-problem held-out argmax agreement (grouped bars).
           Four problems on x-axis; three head families (v2, v3, v4) per group.

Output: paper/figures/pilot_results.pdf (vector PDF, no embedded fonts beyond
the default matplotlib output suitable for arXiv).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Data sources
# -----------------------------------------------------------------------------

ROUNDS = [
    ("v2", "/tmp/distillation_v2.jsonl",          "v2\n(uniform)"),
    ("v3-train", "/tmp/distillation_v3_train.jsonl", "v3-train\n(v2 prior)"),
    ("v3-holdout", "/tmp/distillation_v3_holdout.jsonl", "v3-holdout\n(v2 prior)"),
    ("v4-train", "/tmp/distillation_v4_train.jsonl", "v4-train\n(v3 prior)"),
    ("v4-holdout", "/tmp/distillation_v4_holdout.jsonl", "v4-holdout\n(v3 prior)"),
    ("v4b-train", "/tmp/distillation_v4b_train.jsonl", "v4b-train\n(v3 prior)"),
    ("v4b-holdout", "/tmp/distillation_v4b_holdout.jsonl", "v4b-holdout\n(v3 prior)"),
]

GENERALIZATION = {
    # heads -> dict(problem -> (agree%, mean_L1)); all evaluated on v3-holdout
    # (the canonical apples-to-apples evaluation set across rounds).
    "v2 heads": {
        "multiply_small_ho":  (93.3, 0.556),
        "factor_ho":          (100.0, 0.306),
        "gsm8k_toy_ho":       (20.0, 0.476),
        "algebra_toy_ho":     (13.3, 0.823),
    },
    "v3 heads": {
        "multiply_small_ho":  (100.0, 0.226),
        "factor_ho":          (100.0, 0.167),
        "gsm8k_toy_ho":       (26.7, 0.509),
        "algebra_toy_ho":     (60.0, 0.356),
    },
    "v4 heads": {
        "multiply_small_ho":  (100.0, 0.190),
        "factor_ho":          (100.0, 0.159),
        "gsm8k_toy_ho":       (33.3, 0.411),
        "algebra_toy_ho":     (26.7, 0.868),
    },
    "v4b heads": {
        "multiply_small_ho":  (100.0, 0.124),
        "factor_ho":          (100.0, 0.135),
        "gsm8k_toy_ho":       (33.3, 0.369),
        "algebra_toy_ho":     (13.3, 0.758),
    },
}

PROBLEM_LABELS = {
    "multiply_small_ho": "multiply\n(arithmetic)",
    "factor_ho":         "factor\n(factoring)",
    "gsm8k_toy_ho":      "gsm8k\n(word)",
    "algebra_toy_ho":    "algebra\n(solve-for-x)",
}
PROBLEM_ORDER = ["multiply_small_ho", "factor_ho", "gsm8k_toy_ho", "algebra_toy_ho"]


def load_sharpness(path: str):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    max_probs = [max(r["pi_mcts"].values()) for r in recs if r.get("pi_mcts")]
    if not max_probs:
        raise SystemExit(
            f"{path}: no records with pi_mcts (empty or partial collection "
            f"file); refusing to bake NaN/IndexError into the paper figure"
        )
    entropies = []
    for r in recs:
        if not r.get("pi_mcts"):
            continue
        ent = -sum(p * math.log2(p) for p in r["pi_mcts"].values() if p > 0)
        entropies.append(ent)
    # Use the same median convention as analyze_distillation_rounds.py:
    # sorted(values)[N // 2] (a single observed record's value), not the
    # numpy-average-of-two-middle convention. This keeps the figure's
    # annotation byte-equivalent with the paper's Table 1.
    sorted_max = sorted(max_probs)
    return {
        "n": len(recs),
        "mean_max": float(np.mean(max_probs)),
        "median_max": float(sorted_max[len(sorted_max) // 2]),
        "mean_entropy": float(np.mean(entropies)),
        "frac_concentrated": sum(p >= 0.5 for p in max_probs) / max(1, len(max_probs)),
    }


def main():
    # ------- Data ------
    sharpness = [(label, load_sharpness(path)) for _, path, label in ROUNDS]

    # ------- Figure --------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # ---- Panel A: sharpness ----
    labels = [s[0] for s in sharpness]
    mean_maxes = [s[1]["mean_max"] for s in sharpness]
    median_maxes = [s[1]["median_max"] for s in sharpness]
    entropies = [s[1]["mean_entropy"] for s in sharpness]
    x = np.arange(len(labels))

    # color: v2 (uniform) gets a muted color; v3/v4/v4b trained-prior get
    # distinct family shades (blue=v3, orange=v4, teal=v4b; train darker,
    # holdout lighter)
    colors = ["#888888", "#3F88C5", "#88BCD9", "#D2691E", "#E6B380", "#2A9D8F", "#7FC8BE"]
    if len(colors) < len(labels):
        raise SystemExit(
            f"Panel A has {len(labels)} rounds but only {len(colors)} colors; "
            f"extend the colors list (matplotlib would silently cycle)"
        )
    bars = axL.bar(x, mean_maxes, color=colors[: len(labels)], edgecolor="black", linewidth=0.5, alpha=0.92)
    axL.set_ylabel(r"mean $\max_a\,\pi_{\mathrm{MCTS}}(a\mid s)$")
    axL.set_xticks(x)
    axL.set_xticklabels(labels, fontsize=8)
    axL.set_ylim(0, 0.72)
    axL.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.55)
    axL.text(len(labels) - 0.4, 0.51, "concentrated", color="gray", fontsize=8, ha="right")
    axL.set_title("(A) Distillation-target sharpness across rounds", fontsize=11)
    # annotate each bar with median (small) and entropy (smaller, below)
    for i, (mm, med, ent) in enumerate(zip(mean_maxes, median_maxes, entropies)):
        axL.text(i, mm + 0.012, f"med {med:.2f}", ha="center", fontsize=7.5)
        axL.text(i, mm + 0.038, f"H {ent:.2f}b", ha="center", fontsize=7.5, color="#444")

    # ---- Panel B: per-problem held-out argmax agreement ----
    heads = list(GENERALIZATION.keys())
    n_heads = len(heads)
    n_problems = len(PROBLEM_ORDER)
    bar_w = 0.20
    xp = np.arange(n_problems)
    head_colors = {
        "v2 heads": "#888888",
        "v3 heads": "#3F88C5",
        "v4 heads": "#D2691E",
        "v4b heads": "#2A9D8F",
    }
    missing = [h for h in heads if h not in head_colors]
    if missing:
        raise SystemExit(
            f"Panel B heads {missing} have no entry in head_colors; "
            f"add colors before regenerating the figure"
        )
    for h_idx, head in enumerate(heads):
        agree = [GENERALIZATION[head][p][0] for p in PROBLEM_ORDER]
        offset = (h_idx - (n_heads - 1) / 2) * bar_w
        axR.bar(xp + offset, agree, width=bar_w, label=head, color=head_colors[head],
                edgecolor="black", linewidth=0.4, alpha=0.92)
        for i, val in enumerate(agree):
            axR.text(xp[i] + offset, val + 1.5, f"{val:.0f}", ha="center", fontsize=6.8)

    axR.set_ylabel("held-out argmax agreement (%)")
    axR.set_xticks(xp)
    axR.set_xticklabels([PROBLEM_LABELS[p] for p in PROBLEM_ORDER], fontsize=9)
    axR.set_ylim(0, 118)
    axR.axhline(100, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    axR.set_title(r"(B) Held-out $\arg\max_a\,p_\theta$ vs $\arg\max_a\,\pi_{\mathrm{MCTS}}$",
                  fontsize=11)
    axR.legend(loc="upper right", fontsize=8, frameon=False, ncol=4, bbox_to_anchor=(1.0, 1.15))

    plt.tight_layout(pad=1.2)

    out_dir = Path(__file__).resolve().parents[1] / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pilot_results.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")

    # Sanity-check sharpness numbers printed for cross-check with the paper table
    print("\nSharpness check:")
    for label, s in sharpness:
        print(f"  {label!r:<14}: N={s['n']}, mean_max={s['mean_max']:.3f}, "
              f"med_max={s['median_max']:.3f}, H={s['mean_entropy']:.3f}")


if __name__ == "__main__":
    sys.exit(main())
