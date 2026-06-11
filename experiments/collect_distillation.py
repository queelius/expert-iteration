"""Collect the first distillation dataset from MCTS rollouts on toy problems.

For each toy problem in the smoke-test set, this script:

1. Walks a scripted three-step policy
   (``FORM_HYPOTHESIS`` -> ``INVOKE_VERIFIER`` -> ``COMMIT_ANSWER``)
   through the LLM-driven context MDP.
2. At every non-terminal state along the trajectory, runs an MCTS search
   from that state with a uniform PUCT prior and records:

   * the serialized state,
   * the available verb set,
   * the empirical visit-count distribution pi_MCTS,
   * the chance-node mean values Q_MCTS,
   * the recommended verb (argmax visit),
   * the corresponding LLM embedding of the serialized state.

3. Writes one JSONL record per (problem, turn).

The records are the seed dataset for the first round of expert-iteration
distillation: a parametric pi_theta is trained on the visit-count
distributions by cross-entropy, and Q_psi is trained on the chance-node
Q estimates by MSE. The embedding is the input feature vector for both
heads. We deliberately collect over a scripted policy (rather than over
the MCTS-recommended action) so that the dataset surfaces what MCTS
would have said *at the states the scripted policy actually visits*.
The dataset can be extended in later sessions by running the same
script with a different policy or larger MCTS budget.

A summary comparison table over the initial states of each problem is
printed to stdout at the end (this gives us the "A" deliverable in one
shot with the data we already collect for "B").

Usage::

    python experiments/collect_distillation.py \\
        --base-url http://192.168.0.204:11434/v1 \\
        --embed-url http://192.168.0.204:11434/api/embed \\
        --model qwen2.5:3b \\
        --num-simulations 10 \\
        --output /tmp/distillation_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import numpy as np

from verifier_anchored_rl import (
    LLMContextMDP,
    MCTSConfig,
    MathAction,
    OpenAICompatibleLLMClient,
    Verb,
    available_actions,
    mcts_search,
    root_q_estimates,
    serialize_state,
    visit_count_policy,
)
from verifier_anchored_rl.embedding_client import make_embedding_client
from verifier_anchored_rl.mcts import DecisionNode
from verifier_anchored_rl.trained_prior import (
    EmbeddingCache,
    load_actor_from_checkpoint,
    make_trained_prior,
)


# -----------------------------------------------------------------------------
# Toy problems: shared registry in experiments/problems.py
# -----------------------------------------------------------------------------

try:
    # Run as a script: python experiments/collect_distillation.py
    from problems import (
        PROBLEMS,
        PROBLEM_GROUPS,
        resolve_problem_selection,
        scripted_verb,
    )
except ImportError:
    # Imported as a module from the repo root.
    from experiments.problems import (
        PROBLEMS,
        PROBLEM_GROUPS,
        resolve_problem_selection,
        scripted_verb,
    )


# -----------------------------------------------------------------------------
# Read MCTS root into a JSONL-friendly record
# -----------------------------------------------------------------------------


def root_record(root: DecisionNode) -> dict[str, Any]:
    """Extract (visit counts, Q estimates, priors, pi_MCTS, recommended action) at the root."""
    visit_counts = {}
    q_estimates = {}
    chance_samples = {}
    priors = {}
    for action, chance in root.children.items():
        verb_name = action.verb.value
        visit_counts[verb_name] = chance.visits
        q_estimates[verb_name] = chance.mean_value
        chance_samples[verb_name] = len(chance.children)
        priors[verb_name] = round(float(chance.prior), 6)

    pi_dict = visit_count_policy(root, temperature=1.0)
    pi_mcts = {action.verb.value: float(p) for action, p in pi_dict.items()}

    q_root_dict = root_q_estimates(root)
    q_mcts = {action.verb.value: float(q) for action, q in q_root_dict.items()}

    if root.children:
        recommended = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
        rec_verb = recommended.verb.value
    else:
        rec_verb = None

    return {
        "root_visits": root.visits,
        "visit_counts": visit_counts,
        "mean_value_per_action": {k: round(v, 6) for k, v in q_estimates.items()},
        "chance_node_children_per_action": chance_samples,
        # The PUCT prior MCTS actually used at each root edge. This is what
        # lets analyze_distillation_rounds' prior-vs-pi_MCTS agreement
        # metric run on collected datasets (it was previously a dead path:
        # no producer wrote a 'priors' field).
        "priors": priors,
        "pi_mcts": {k: round(v, 6) for k, v in pi_mcts.items()},
        "q_mcts": {k: round(v, 6) for k, v in q_mcts.items()},
        "recommended_action": rec_verb,
    }


# -----------------------------------------------------------------------------
# Per-problem rollout
# -----------------------------------------------------------------------------


def collect_for_problem(
    problem_key: str,
    problem_def: dict[str, str],
    *,
    seed: int,
    llm_client: OpenAICompatibleLLMClient,
    embedding_cache: EmbeddingCache,
    config: MCTSConfig,
    rng: np.random.Generator,
    depth_limit: int,
    max_turns: int,
    temperature: float,
    max_tokens: int,
    seed_llm: bool = True,
    output_handle=None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run a scripted trajectory, running MCTS + fetching embedding at every non-terminal state.

    Returns one record per non-terminal state visited. If ``output_handle`` is
    provided, also writes each record as JSONL immediately (crash-resilient).
    """
    if verbose:
        print(f"\n--- {problem_key} (seed={seed}): {problem_def['problem']!r} (expected={problem_def['expected']!r}) ---", flush=True)

    mdp = LLMContextMDP(
        problem_statement=problem_def["problem"],
        expected_answer=problem_def["expected"],
        llm_client=llm_client,
        depth_limit=depth_limit,
        max_turns=max_turns,
        temperature=temperature,
        max_tokens=max_tokens,
        prompt_strategy="full",
        seed_llm=seed_llm,
    )

    state = mdp.initial_state()
    records: list[dict[str, Any]] = []

    while not mdp.is_terminal(state) and state.turn < max_turns:
        avail = available_actions(state, depth_limit=mdp.depth_limit)
        if not avail:
            if verbose:
                print(f"  turn={state.turn}: no actions available, stopping", flush=True)
            break

        if verbose:
            print(f"  turn={state.turn}: running MCTS (B={config.num_simulations}, rollout_depth={config.rollout_depth_limit})...", flush=True)

        serialized = serialize_state(state)
        t0 = time.time()
        root = mcts_search(mdp, state, config, rng)
        mcts_duration = time.time() - t0

        if verbose:
            visit_counts = {a.verb.value: c.visits for a, c in root.children.items()}
            print(f"    MCTS done in {mcts_duration:.1f}s; visits={visit_counts}", flush=True)

        t0 = time.time()
        embedding = embedding_cache.get(serialized)
        embed_duration = time.time() - t0
        if verbose:
            if embedding is not None:
                cached_note = "(cached)" if embed_duration < 0.05 else ""
                print(
                    f"    embedding ({len(embedding)}-d) in {embed_duration:.2f}s {cached_note}".rstrip(),
                    flush=True,
                )
            else:
                print(f"    embedding FAILED in {embed_duration:.2f}s", flush=True)

        rec = {
            "problem_key": problem_key,
            "seed": seed,
            "problem_statement": problem_def["problem"],
            "expected_answer": problem_def["expected"],
            "turn": state.turn,
            "serialized_state": serialized,
            "available_verbs": [v.value for v in avail],
            **root_record(root),
            "mcts_wallclock_s": round(mcts_duration, 3),
            "embedding_wallclock_s": round(embed_duration, 3) if embedding is not None else None,
            "embedding_dim": len(embedding) if embedding is not None else None,
            "embedding": embedding,
        }
        records.append(rec)
        if output_handle is not None:
            output_handle.write(json.dumps(rec) + "\n")
            output_handle.flush()

        # Advance via scripted policy.
        chosen_verb = scripted_verb(state.turn, avail)
        if verbose:
            print(f"    scripted policy -> {chosen_verb.value}", flush=True)
        action = MathAction(verb=chosen_verb)
        state, reward = mdp.sample_transition(state, action)
        if verbose:
            print(f"    reward={reward:+.3f}, new turn={state.turn}", flush=True)

    if verbose:
        commit = state.committed_answer
        print(f"  final: committed={commit!r}, terminal={mdp.is_terminal(state)}", flush=True)

    return records


# -----------------------------------------------------------------------------
# Comparison table over initial states
# -----------------------------------------------------------------------------


def print_initial_comparison(records: list[dict[str, Any]]) -> None:
    """Initial-state comparison across problems."""
    initial = [r for r in records if r["turn"] == 0]
    if not initial:
        print("\n(no initial records to compare)", flush=True)
        return

    print("\n" + "=" * 90, flush=True)
    print("INITIAL-STATE MCTS COMPARISON", flush=True)
    print("=" * 90, flush=True)

    verbs_order = [v.value for v in Verb]
    print(f"\n{'problem':<18} {'recommended':<18} {'Q̂(rec)':<10} {'visit_counts':<40}", flush=True)
    print("-" * 90, flush=True)
    for r in initial:
        rec_verb = r["recommended_action"] or "-"
        q_rec = r["q_mcts"].get(rec_verb, 0.0) if rec_verb != "-" else 0.0
        vc = r.get("visit_counts", {})
        vc_str = ", ".join(f"{v}:{vc[v]}" for v in verbs_order if v in vc)
        print(f"{r['problem_key']:<18} {rec_verb:<18} {q_rec:+.4f}   {vc_str:<40}", flush=True)

    print("\nFull pi_MCTS per problem:", flush=True)
    for r in initial:
        print(f"  {r['problem_key']}: " + ", ".join(
            f"{v}={r['pi_mcts'].get(v, 0):.2f}" for v in verbs_order if v in r["pi_mcts"]
        ), flush=True)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://192.168.0.204:11434/v1")
    p.add_argument("--embed-url", default="http://192.168.0.204:11434/api/embed")
    p.add_argument("--model", default="qwen2.5:3b")
    p.add_argument("--embed-model", default=None, help="defaults to --model if not set")
    p.add_argument("--num-simulations", type=int, default=10)
    p.add_argument("--c-puct", type=float, default=1.0)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--rollout-depth", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--depth-limit", type=int, default=3)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="MCTS rng seeds to loop over (one trajectory per seed per problem)",
    )
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument(
        "--append",
        action="store_true",
        help="append to --output instead of overwriting (resumes interrupted runs)",
    )
    p.add_argument(
        "--problems",
        nargs="+",
        choices=list(PROBLEMS.keys()) + list(PROBLEM_GROUPS.keys()) + ["all"],
        default=["all"],
        help=(
            "which toy problems to include. Pass individual keys, or one of "
            "the group aliases: 'train', 'holdout', 'all'."
        ),
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "optional path to a trained TrainableActor checkpoint. When set, "
            "MCTS uses pi_theta as the PUCT prior instead of uniform; this is "
            "how an expert-iteration round drives search with the previous "
            "round's learned policy."
        ),
    )
    p.add_argument(
        "--prior-smoothing",
        type=float,
        default=0.02,
        help="label-smoothing epsilon for the trained prior",
    )
    p.add_argument(
        "--no-llm-seed",
        action="store_true",
        help=(
            "do NOT forward a per-call sampling seed to the LLM (fully "
            "stochastic transitions). Default is seeded transitions, which "
            "make collections reproducible at matched --seeds up to the "
            "server's compute-level determinism."
        ),
    )
    p.add_argument(
        "--output",
        default="/tmp/distillation_v1.jsonl",
        help="output JSONL file path",
    )
    args = p.parse_args(argv)

    embed_model = args.embed_model or args.model

    # Resolve problem-set selection: group aliases expand to their members,
    # 'all' to every problem, individual keys pass through (deduplicated,
    # order-preserving).
    problem_keys = resolve_problem_selection(args.problems)

    prior_mode = "trained" if args.checkpoint else "uniform"
    print(f"=== Distillation data collection ===", flush=True)
    print(f"model: {args.model} | embed_model: {embed_model}", flush=True)
    print(f"problems: {problem_keys}", flush=True)
    print(f"seeds: {args.seeds}", flush=True)
    print(f"MCTS B={args.num_simulations}, rollout_depth={args.rollout_depth}, c_puct={args.c_puct}, discount={args.discount}", flush=True)
    print(f"prior: {prior_mode}" + (f" (checkpoint={args.checkpoint}, smoothing={args.prior_smoothing})" if args.checkpoint else ""), flush=True)
    print(f"LLM temp={args.temperature}, max_tokens={args.max_tokens}", flush=True)
    print(f"output: {args.output} (mode: {'append' if args.append else 'overwrite'})", flush=True)

    llm_client = OpenAICompatibleLLMClient(
        base_url=args.base_url,
        model=args.model,
        timeout_s=args.timeout,
        max_retries=1,
    )

    # Single EmbeddingCache shared across all problems, seeds, and turns:
    # serves both the trained prior (during MCTS) and the per-record
    # embedding written to JSONL. The endpoint-shape chooser supports both
    # Ollama (/api/embed) and OpenAI-compatible (/v1/embeddings) backends.
    embedding_cache = EmbeddingCache(
        make_embedding_client(args.embed_url, embed_model, timeout_s=args.timeout)
    )

    # Build the trained prior BEFORE opening the output file: a failed
    # checkpoint load must not truncate a previously collected dataset
    # (open(..., "w") wipes the file on context entry).
    prior_fn = None
    if args.checkpoint:
        model, ckpt = load_actor_from_checkpoint(args.checkpoint)
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"\nLoaded TrainableActor: {n_params:,} params, "
            f"embed_dim={ckpt['embed_dim']}",
            flush=True,
        )
        prior_fn = make_trained_prior(
            model,
            embedding_cache,
            smoothing=args.prior_smoothing,
        )

    config = MCTSConfig(
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        discount=args.discount,
        rollout_depth_limit=args.rollout_depth,
        prior_fn=prior_fn,
        rollout_policy=None,
    )

    all_records: list[dict[str, Any]] = []
    started = time.time()

    output_mode = "a" if args.append else "w"
    with open(args.output, output_mode) as output_handle:
        for seed in args.seeds:
            for key in problem_keys:
                rng = np.random.default_rng(seed)
                records = collect_for_problem(
                    key,
                    PROBLEMS[key],
                    seed=seed,
                    llm_client=llm_client,
                    embedding_cache=embedding_cache,
                    config=config,
                    rng=rng,
                    depth_limit=args.depth_limit,
                    max_turns=args.max_turns,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    seed_llm=not args.no_llm_seed,
                    output_handle=output_handle,
                    verbose=True,
                )
                all_records.extend(records)
                if records:
                    print(
                        f"  ({len(all_records)} records written so far in this run)",
                        flush=True,
                    )

    total_duration = time.time() - started

    print(f"\n=== Wrote {len(all_records)} records to {args.output} ===", flush=True)
    if args.append:
        print(
            "(--append run: the summaries below cover only THIS run's "
            "records, not the full on-disk file)",
            flush=True,
        )
    print(f"Total wall-clock: {total_duration:.1f}s", flush=True)
    print(f"Embedding cache: {embedding_cache.stats()}", flush=True)

    # Initial-state comparison.
    print_initial_comparison(all_records)

    # Per-problem records summary.
    print("\n" + "=" * 90, flush=True)
    print("DATASET BREAKDOWN", flush=True)
    print("=" * 90, flush=True)
    per_problem = {}
    for r in all_records:
        per_problem.setdefault(r["problem_key"], []).append(r)
    for key, recs in per_problem.items():
        turns = [r["turn"] for r in recs]
        mcts_total = sum(r["mcts_wallclock_s"] for r in recs)
        embed_total = sum(r.get("embedding_wallclock_s") or 0 for r in recs)
        embed_failures = sum(1 for r in recs if r.get("embedding") is None)
        embed_dim = next((r["embedding_dim"] for r in recs if r.get("embedding_dim")), None)
        print(
            f"  {key}: {len(recs)} records (turns {turns}), "
            f"MCTS total {mcts_total:.1f}s, embed total {embed_total:.1f}s, "
            f"embed dim {embed_dim}, {embed_failures} embed failures",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
