"""Does the MCTS distillation target stabilize as the simulation budget B grows?

This is the linchpin measurement behind the pre-registered study's choice of
a large search budget. The pilot collected at B=20 and found the visit-count
targets (pi_MCTS) noisy across independent collections, especially on the
high-ambiguity algebra family. The hypothesis is that the noise is sampling
variance of the visit-count estimator, so a larger B should shrink it.

We measure that directly, isolating the budget effect from the trajectory
cascade (see the reproducibility probe in expert_iteration_rounds.md): for a
*fixed* initial state, we run K independent MCTS searches (different rng
seeds -> different stochastic LLM transitions) at each B, then report how
much the resulting pi_MCTS distributions disagree across runs. Because every
search starts from the same state, any disagreement is the estimator's own
variance, not a divergent trajectory.

Metrics per (problem, B):
  * mean pairwise L1 across the K runs' pi_MCTS (comparable to the 0.2-0.55
    numbers in the reproducibility probe and the v4/v4b post-mortem)
  * mean per-verb standard deviation of the probability
  * modal-argmax agreement: fraction of runs whose argmax equals the modal
    argmax across runs (1.0 = all runs agree on the recommended verb)
  * mean entropy of pi_MCTS (sharpness; expected to fall as B grows)

No embeddings are fetched (uniform prior; the head is not involved), so this
runs against a chat-only server.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from typing import Any

import numpy as np

from verifier_anchored_rl import (
    LLMContextMDP,
    MCTSConfig,
    OpenAICompatibleLLMClient,
    mcts_search,
    visit_count_policy,
)
from experiments.analysis_lib import entropy_bits, mean_pairwise_l1

try:
    from problems import PROBLEMS
except ImportError:
    from experiments.problems import PROBLEMS


def pi_mcts_of(root) -> dict[str, float]:
    pi = visit_count_policy(root, temperature=1.0)
    return {action.verb.value: float(p) for action, p in pi.items()}


def mean_per_verb_std(dists: list[dict[str, float]]) -> float:
    verbs = set().union(*dists) if dists else set()
    if not verbs:
        return 0.0
    stds = []
    for v in verbs:
        vals = [d.get(v, 0.0) for d in dists]
        stds.append(float(np.std(vals)))
    return float(np.mean(stds))


def modal_argmax_agreement(dists: list[dict[str, float]]) -> tuple[str, float]:
    argmaxes = [max(d.items(), key=lambda kv: kv[1])[0] for d in dists if d]
    if not argmaxes:
        return "-", 0.0
    modal, count = Counter(argmaxes).most_common(1)[0]
    return modal, count / len(argmaxes)


def run_cell(
    problem_key: str,
    problem_def: dict[str, str],
    *,
    b: int,
    num_runs: int,
    base_seed: int,
    llm_client,
    depth_limit: int,
    max_turns: int,
    temperature: float,
    max_tokens: int,
    discount: float,
    c_puct: float,
    verbose: bool = True,
) -> dict[str, Any]:
    mdp = LLMContextMDP(
        problem_statement=problem_def["problem"],
        expected_answer=problem_def["expected"],
        llm_client=llm_client,
        depth_limit=depth_limit,
        max_turns=max_turns,
        temperature=temperature,
        max_tokens=max_tokens,
        prompt_strategy="full",
    )
    state = mdp.initial_state()
    config = MCTSConfig(
        num_simulations=b,
        c_puct=c_puct,
        discount=discount,
        rollout_depth_limit=0,
        prior_fn=None,
        rollout_policy=None,
    )

    dists: list[dict[str, float]] = []
    for run in range(num_runs):
        rng = np.random.default_rng(base_seed + run)
        t0 = time.time()
        root = mcts_search(mdp, state, config, rng)
        dur = time.time() - t0
        pi = pi_mcts_of(root)
        dists.append(pi)
        if verbose:
            am = max(pi.items(), key=lambda kv: kv[1])[0] if pi else "-"
            print(
                f"    {problem_key} B={b} run {run}: {dur:5.1f}s "
                f"argmax={am:<16} pi={ {k: round(v,2) for k,v in pi.items()} }",
                flush=True,
            )

    l1 = mean_pairwise_l1(dists)
    pv_std = mean_per_verb_std(dists)
    modal, modal_frac = modal_argmax_agreement(dists)
    mean_ent = float(np.mean([entropy_bits(list(d.values())) for d in dists]))
    return {
        "problem_key": problem_key,
        "B": b,
        "num_runs": num_runs,
        "mean_pairwise_l1": round(l1, 4),
        "mean_per_verb_std": round(pv_std, 4),
        "modal_argmax": modal,
        "modal_argmax_agreement": round(modal_frac, 4),
        "mean_entropy_bits": round(mean_ent, 4),
        "pi_mcts_per_run": dists,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://192.168.0.204:11434/v1")
    p.add_argument("--model", default="qwen2.5:3b")
    p.add_argument(
        "--problems", nargs="+", default=["multiply_small", "algebra_toy_1"]
    )
    p.add_argument("--b-values", type=int, nargs="+", default=[10, 40, 160])
    p.add_argument("--num-runs", type=int, default=6)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--depth-limit", type=int, default=3)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--c-puct", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--output-json", default=None)
    args = p.parse_args(argv)

    print("=== B-sweep target-stability probe ===", flush=True)
    print(f"model: {args.model}", flush=True)
    print(f"problems: {args.problems}", flush=True)
    print(f"B values: {args.b_values}", flush=True)
    print(f"runs per cell: {args.num_runs} (independent rng seeds)", flush=True)
    print(f"temp={args.temperature}, c_puct={args.c_puct}, discount={args.discount}", flush=True)
    print(
        f"NOTE: every search starts from the same fixed initial state, so "
        f"cross-run disagreement is estimator variance, not trajectory drift.",
        flush=True,
    )

    llm_client = OpenAICompatibleLLMClient(
        base_url=args.base_url, model=args.model, timeout_s=args.timeout, max_retries=1
    )

    started = time.time()
    cells: list[dict[str, Any]] = []
    for key in args.problems:
        if key not in PROBLEMS:
            print(f"  ! unknown problem {key!r}, skipping", flush=True)
            continue
        for b in args.b_values:
            print(f"\n-- {key}, B={b} --", flush=True)
            cell = run_cell(
                key,
                PROBLEMS[key],
                b=b,
                num_runs=args.num_runs,
                base_seed=args.base_seed,
                llm_client=llm_client,
                depth_limit=args.depth_limit,
                max_turns=args.max_turns,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                discount=args.discount,
                c_puct=args.c_puct,
            )
            cells.append(cell)
            print(
                f"  => B={b:<4} meanL1={cell['mean_pairwise_l1']:.3f}  "
                f"perVerbStd={cell['mean_per_verb_std']:.3f}  "
                f"modalArgmax={cell['modal_argmax']}({cell['modal_argmax_agreement']:.2f})  "
                f"H={cell['mean_entropy_bits']:.2f}b",
                flush=True,
            )

    total = time.time() - started
    print("\n" + "=" * 88, flush=True)
    print("B-SWEEP STABILITY SUMMARY", flush=True)
    print("=" * 88, flush=True)
    print(
        f"{'problem':<16} {'B':>5} {'meanL1':>8} {'perVerbStd':>11} "
        f"{'modalAgree':>11} {'entropy':>8}",
        flush=True,
    )
    print("-" * 88, flush=True)
    for c in cells:
        print(
            f"{c['problem_key']:<16} {c['B']:>5} {c['mean_pairwise_l1']:>8.3f} "
            f"{c['mean_per_verb_std']:>11.3f} "
            f"{c['modal_argmax']}({c['modal_argmax_agreement']:.2f})".rjust(11)
            + f" {c['mean_entropy_bits']:>8.2f}",
            flush=True,
        )
    print(f"\nTotal wall-clock: {total:.1f}s", flush=True)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"cells": cells, "args": vars(args)}, f, indent=2)
        print(f"Wrote {args.output_json}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
