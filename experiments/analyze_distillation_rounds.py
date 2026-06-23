"""Compare distillation datasets from successive expert-iteration rounds.

Given two JSONL datasets (e.g. v2 collected with uniform prior, v3 collected
with trained pi_theta as PUCT prior), report:

1. **Sharpness shift** of the pi_MCTS distributions: mean max-probability and
   mean entropy. If the trained prior is producing useful guidance, the v3
   distributions should be MORE concentrated (higher max-prob, lower
   entropy) than v2 at the same B.

2. **Prior-vs-MCTS agreement**: when the trained prior was used during
   search, where did the prior's argmax agree with the eventual MCTS
   argmax? Pure agreement means MCTS rubber-stamped the prior. Pure
   disagreement means MCTS overrode it. Healthy expert iteration shows
   *partial* agreement: the prior biased search but MCTS still added
   information via Q-value backup.

3. **Generalization quality**: optionally load a TrainableActor checkpoint
   and predict pi_theta on the held-out dataset's serialized states.
   Report argmax agreement with the held-out MCTS targets. This is the
   zero-shot generalization metric.

Usage::

    # v2 -> v3 round comparison
    python experiments/analyze_distillation_rounds.py \\
        --round-a /tmp/distillation_v2.jsonl \\
        --round-b /tmp/distillation_v3_train.jsonl \\
        --round-a-label v2 \\
        --round-b-label v3-train

    # generalization check: v2 heads on v3-holdout
    python experiments/analyze_distillation_rounds.py \\
        --round-a /tmp/distillation_v3_holdout.jsonl \\
        --round-a-label v3-holdout \\
        --generalization-checkpoint /tmp/heads_v2.pt
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # generalization check needs torch; sharpness/agreement do not

from experiments.analysis_lib import entropy_bits, load_jsonl


def entropy_of(dist: dict[str, float]) -> float:
    """Shannon entropy in bits over a probability dict (zeros ignored)."""
    return entropy_bits(list(dist.values()))


def safe_argmax(dist: dict[str, float]) -> str | None:
    if not dist:
        return None
    return max(dist.items(), key=lambda kv: kv[1])[0]


# -----------------------------------------------------------------------------
# Sharpness comparison
# -----------------------------------------------------------------------------


def sharpness_report(records: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Mean max-probability and mean entropy of pi_MCTS over a dataset."""
    max_probs = [max(r["pi_mcts"].values()) for r in records if r.get("pi_mcts")]
    entropies = [entropy_of(r["pi_mcts"]) for r in records if r.get("pi_mcts")]
    if not max_probs:
        # Either the dataset is empty or no record carries a pi_mcts
        # (e.g. an interrupted/partial collection file): report it as
        # empty rather than dividing by zero below.
        return {"label": label, "n_records": 0}
    concentrated = sum(1 for p in max_probs if p >= 0.5)
    return {
        "label": label,
        "n_records": len(records),
        "mean_max_prob": sum(max_probs) / len(max_probs),
        "median_max_prob": sorted(max_probs)[len(max_probs) // 2],
        "mean_entropy_bits": sum(entropies) / len(entropies),
        "n_concentrated_above_0.5": concentrated,
        "frac_concentrated_above_0.5": concentrated / len(max_probs),
    }


def print_sharpness_table(reports: list[dict[str, Any]]) -> None:
    print()
    print("=" * 90)
    print("SHARPNESS COMPARISON (pi_MCTS)")
    print("=" * 90)
    print(
        f"{'label':<14} {'N':>5} {'mean max':>10} {'med max':>9} "
        f"{'H (bits)':>10} {'n_max>=0.5':>11} {'frac':>7}"
    )
    print("-" * 90)
    for r in reports:
        if r["n_records"] == 0:
            print(f"{r['label']:<14} {'(empty)':>5}")
            continue
        print(
            f"{r['label']:<14} {r['n_records']:>5} {r['mean_max_prob']:>10.3f} "
            f"{r['median_max_prob']:>9.3f} {r['mean_entropy_bits']:>10.3f} "
            f"{r['n_concentrated_above_0.5']:>11d} {r['frac_concentrated_above_0.5']:>7.2f}"
        )


# -----------------------------------------------------------------------------
# Prior-vs-pi_MCTS agreement (round B has priors set during search)
# -----------------------------------------------------------------------------


def prior_agreement_report(records: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """How often did the (search-time) prior's argmax match the resulting pi_MCTS argmax?

    For v2-style data this is uniform-vs-mcts, which is uninformative
    (uniform has no argmax). We still report it for symmetry but flag
    that the priors look uniform.
    """
    if not records:
        return {"label": label, "n_records": 0}

    n_with_prior = 0
    n_priors_uniform = 0
    n_argmax_agree = 0
    n_prior_in_top2 = 0
    l1_distances = []

    for r in records:
        # Try to extract the priors that MCTS used at the root.
        # Old records may not carry this. Heuristic: pi_mcts vs uniform 1/k.
        pi_mcts = r.get("pi_mcts", {})
        if not pi_mcts:
            continue

        # Priors are stored under "priors" by run_llm_mcts_with_prior; the
        # collect script doesn't store them (only stores derived pi_mcts).
        # So for v3 we infer that priors were the trained ones (we know
        # the script ran with --checkpoint) but we don't have the actual
        # vector. We instead approximate by looking at the prior surrogate
        # we DO have: visit_counts vs uniform.
        priors = r.get("priors")
        if priors is None:
            # No priors stored -> skip detailed agreement check
            continue
        n_with_prior += 1

        verbs = sorted(set(pi_mcts) | set(priors))
        u = 1.0 / max(1, len(verbs))
        is_uniform = all(abs(priors.get(v, 0) - u) < 0.02 for v in verbs)
        if is_uniform:
            n_priors_uniform += 1

        prior_argmax = safe_argmax(priors)
        mcts_argmax = safe_argmax(pi_mcts)
        if prior_argmax == mcts_argmax:
            n_argmax_agree += 1

        # top-2 prior verbs
        top2_prior = sorted(priors.items(), key=lambda kv: -kv[1])[:2]
        top2_keys = {kv[0] for kv in top2_prior}
        if mcts_argmax in top2_keys:
            n_prior_in_top2 += 1

        # L1 between prior and pi_mcts
        l1 = sum(abs(priors.get(v, 0) - pi_mcts.get(v, 0)) for v in verbs)
        l1_distances.append(l1)

    return {
        "label": label,
        "n_records": len(records),
        "n_with_prior_stored": n_with_prior,
        "n_priors_uniform": n_priors_uniform,
        "argmax_agreement": n_argmax_agree,
        "argmax_agreement_pct": round(100 * n_argmax_agree / max(1, n_with_prior), 1) if n_with_prior else None,
        "prior_top2_contains_mcts_argmax": n_prior_in_top2,
        "mean_l1_prior_vs_pi_mcts": sum(l1_distances) / len(l1_distances) if l1_distances else None,
    }


def print_prior_agreement(reports: list[dict[str, Any]]) -> None:
    print()
    print("=" * 90)
    print("PRIOR-vs-pi_MCTS AGREEMENT (when prior was stored on the record)")
    print("=" * 90)
    for r in reports:
        if r.get("n_with_prior_stored", 0) == 0:
            print(f"  {r['label']}: no priors stored on records (skip)")
            continue
        print(
            f"  {r['label']}: {r['n_with_prior_stored']} records with priors; "
            f"argmax-agreement {r['argmax_agreement']}/{r['n_with_prior_stored']} "
            f"({r['argmax_agreement_pct']}%); "
            f"prior top-2 contains mcts-argmax {r['prior_top2_contains_mcts_argmax']}/{r['n_with_prior_stored']}; "
            f"mean L1 = {r['mean_l1_prior_vs_pi_mcts']:.3f}"
        )


# -----------------------------------------------------------------------------
# Per-problem breakdown of recommended action
# -----------------------------------------------------------------------------


def recommended_breakdown(records: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """For each (problem_key, turn), count distribution of recommended_action across seeds."""
    by_key: dict[tuple[str, int], list[str]] = defaultdict(list)
    for r in records:
        key = (r["problem_key"], r["turn"])
        rec = r.get("recommended_action")
        if rec:
            by_key[key].append(rec)

    groups: dict[str, dict[str, int]] = {}
    for k, v in by_key.items():
        cnt: dict[str, int] = defaultdict(int)
        for x in v:
            cnt[x] += 1
        # Sort by descending count for readable output.
        ordered = dict(sorted(cnt.items(), key=lambda kv: -kv[1]))
        groups[f"{k[0]}@t{k[1]}"] = ordered

    return {"label": label, "groups": groups}


def print_recommended_breakdown(reports: list[dict[str, Any]]) -> None:
    print()
    print("=" * 90)
    print("RECOMMENDED-ACTION CONSISTENCY ACROSS SEEDS (per problem-turn)")
    print("=" * 90)
    for r in reports:
        if not r["groups"]:
            print(f"  {r['label']}: empty")
            continue
        print(f"\n  {r['label']}:")
        for grp, counts in sorted(r["groups"].items()):
            counts_str = ", ".join(f"{k}:{v}" for k, v in counts.items())
            print(f"    {grp:<28} {counts_str}")


# -----------------------------------------------------------------------------
# Generalization: predict pi_theta on a dataset and compare to its pi_mcts
# -----------------------------------------------------------------------------


def generalization_report(
    records: list[dict[str, Any]],
    checkpoint_path: str,
    label: str,
) -> dict[str, Any]:
    """Run a TrainableActor on the dataset's serialized states and compare
    argmax + L1 to the MCTS targets. Tests zero-shot quality of the model
    on (potentially out-of-distribution) data.
    """
    if torch is None:
        raise ImportError("torch required for generalization check")
    from verifier_anchored_rl.heads import (
        NUM_VERBS,
        VERB_NAME_TO_INDEX,
        VERB_ORDER,
        encode_batch,
    )
    from verifier_anchored_rl.trained_prior import load_actor_from_checkpoint

    model, _ckpt = load_actor_from_checkpoint(checkpoint_path)

    # Filter to records with an embedding.
    usable = [r for r in records if r.get("embedding") and r.get("pi_mcts")]
    if not usable:
        return {"label": label, "n_records": 0, "checkpoint": checkpoint_path}

    batch = encode_batch(usable)
    with torch.no_grad():
        pi_theta = model.pi(batch["embedding"], batch["available_mask"])
        pi_theta_np = pi_theta.cpu().numpy()

    n_agree = 0
    l1_total = 0.0
    per_problem: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "agree": 0, "l1": 0.0})
    rows: list[dict[str, Any]] = []

    for i, rec in enumerate(usable):
        pi_t_dict = {VERB_ORDER[j].value: float(pi_theta_np[i, j]) for j in range(NUM_VERBS)}
        # Restrict to AVAILABLE verbs (the same support train_heads.evaluate
        # argmaxes over), NOT to pi_mcts's keys: at budgets B < |available|
        # MCTS leaves some available verbs unexpanded, and restricting to
        # expanded verbs would drop the model's true argmax and silently
        # inflate the agreement metric. model.pi already zeroes unavailable
        # positions, so this filter only removes exact zeros at B >= |A|.
        available = set(rec.get("available_verbs") or rec["pi_mcts"].keys())
        pi_t_avail = {k: v for k, v in pi_t_dict.items() if k in available}
        if pi_t_avail:
            s = sum(pi_t_avail.values())
            if s > 0:
                pi_t_avail = {k: v / s for k, v in pi_t_avail.items()}
        theta_argmax = safe_argmax(pi_t_avail)
        mcts_argmax = safe_argmax(rec["pi_mcts"])
        agree = theta_argmax == mcts_argmax
        if agree:
            n_agree += 1

        verbs = sorted(set(pi_t_avail) | set(rec["pi_mcts"]))
        l1 = sum(abs(pi_t_avail.get(v, 0) - rec["pi_mcts"].get(v, 0)) for v in verbs)
        l1_total += l1

        pk = rec["problem_key"]
        per_problem[pk]["n"] += 1
        per_problem[pk]["agree"] += 1 if agree else 0
        per_problem[pk]["l1"] += l1

        rows.append({
            "problem_key": pk,
            "turn": rec["turn"],
            "seed": rec.get("seed"),
            "theta_argmax": theta_argmax,
            "mcts_argmax": mcts_argmax,
            "agree": agree,
            "l1": round(l1, 3),
        })

    summary_per_problem = {
        pk: {
            "n": v["n"],
            "argmax_agreement_pct": round(100 * v["agree"] / v["n"], 1),
            "mean_l1": round(v["l1"] / v["n"], 3),
        }
        for pk, v in per_problem.items()
    }

    return {
        "label": label,
        "n_records": len(usable),
        "checkpoint": checkpoint_path,
        "argmax_agreement_total": n_agree,
        "argmax_agreement_pct": round(100 * n_agree / len(usable), 1),
        "mean_l1_pi_theta_vs_pi_mcts": round(l1_total / len(usable), 3),
        "per_problem": summary_per_problem,
        "rows": rows,
    }


def print_generalization(report: dict[str, Any]) -> None:
    print()
    print("=" * 90)
    print("GENERALIZATION: pi_theta(s) vs pi_MCTS(s) on held-out states")
    print("=" * 90)
    print(f"  checkpoint:           {report['checkpoint']}")
    print(f"  dataset:              {report['label']} ({report['n_records']} records)")
    print(
        f"  argmax agreement:     {report['argmax_agreement_total']}/{report['n_records']} "
        f"({report['argmax_agreement_pct']}%)"
    )
    print(f"  mean L1(pi_theta, pi_MCTS): {report['mean_l1_pi_theta_vs_pi_mcts']}")
    print()
    print("  Per-problem breakdown:")
    for pk, v in report["per_problem"].items():
        print(f"    {pk:<22} n={v['n']:<3} agree={v['argmax_agreement_pct']:>5}%  meanL1={v['mean_l1']}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--round-a", required=True, help="path to round A JSONL")
    p.add_argument("--round-a-label", default="A")
    p.add_argument("--round-b", default=None, help="optional second round to compare")
    p.add_argument("--round-b-label", default="B")
    p.add_argument(
        "--generalization-checkpoint",
        default=None,
        help="optional TrainableActor checkpoint to evaluate on the round-A dataset",
    )
    p.add_argument(
        "--output-json",
        default=None,
        help="optional JSON dump of all metrics",
    )
    args = p.parse_args(argv)

    a = load_jsonl(args.round_a)
    print(f"Round A ({args.round_a_label}): {len(a)} records from {args.round_a}")
    sharpness = [sharpness_report(a, args.round_a_label)]
    prior_agree = [prior_agreement_report(a, args.round_a_label)]
    recommended = [recommended_breakdown(a, args.round_a_label)]

    if args.round_b:
        b = load_jsonl(args.round_b)
        print(f"Round B ({args.round_b_label}): {len(b)} records from {args.round_b}")
        sharpness.append(sharpness_report(b, args.round_b_label))
        prior_agree.append(prior_agreement_report(b, args.round_b_label))
        recommended.append(recommended_breakdown(b, args.round_b_label))

    print_sharpness_table(sharpness)
    print_prior_agreement(prior_agree)
    print_recommended_breakdown(recommended)

    gen_report = None
    if args.generalization_checkpoint:
        gen_report = generalization_report(
            a,
            args.generalization_checkpoint,
            args.round_a_label,
        )
        print_generalization(gen_report)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(
                {
                    "sharpness": sharpness,
                    "prior_agreement": prior_agree,
                    "recommended_breakdown": recommended,
                    "generalization": gen_report,
                },
                f,
                indent=2,
            )
        print(f"\nWrote metrics to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
