"""Run MCTS with a trained pi_theta as the PUCT prior, compare against uniform.

Loads a TrainableActor checkpoint produced by ``experiments/train_heads.py``,
turns its policy head into a ``PriorFn`` that ``mcts_search`` accepts, and
runs MCTS at the initial state of one or more toy problems under both
priors. The point is to test whether a learned prior shifts the visit-count
distribution in a measurable way.

The two runs use the same MCTS RNG seed and the same LLM temperature,
and every LLM transition is sampled with a server-side seed derived from
that rng, so each individual run is reproducible. The uniform and trained
searches still diverge once the prior steers them down different branches
(the seed sequence is shared, the call sequences are not), so observed
differences reflect the prior plus the divergent trajectories it induces;
single-seed comparisons should be read accordingly. For each problem the
script reports::

    problem      uniform_pi_MCTS                     trained_pi_MCTS
    -------      ------------------                  ------------------
    multiply_small                                   (which verb shifted?)
    ...

and writes a JSON dump of every chance edge for offline analysis.

The script auto-caches state embeddings: MCTS revisits the same root
across all simulations, so the embedding is fetched once per MCTS search
(not once per simulation). For descendant states encountered during
selection, embeddings are cached by serialized state string.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    print("torch is required: pip install torch", file=sys.stderr)
    sys.exit(1)

from verifier_anchored_rl import (
    LLMContextMDP,
    MCTSConfig,
    OpenAICompatibleLLMClient,
    mcts_search,
    visit_count_policy,
)
from verifier_anchored_rl.embedding_client import make_embedding_client
from verifier_anchored_rl.mcts import DecisionNode
from verifier_anchored_rl.trained_prior import (
    EmbeddingCache,
    load_actor_from_checkpoint,
    make_trained_prior,
)

try:
    # Run as a script: python experiments/run_llm_mcts_with_prior.py
    from problems import PROBLEMS, resolve_problem_selection
except ImportError:
    # Imported as a module from the repo root.
    from experiments.problems import PROBLEMS, resolve_problem_selection





# -----------------------------------------------------------------------------
# Per-problem comparison
# -----------------------------------------------------------------------------


def run_one_comparison(
    problem_key: str,
    problem_def: dict[str, str],
    *,
    llm_client: OpenAICompatibleLLMClient,
    config_uniform: MCTSConfig,
    config_trained: MCTSConfig,
    depth_limit: int,
    max_turns: int,
    temperature: float,
    max_tokens: int,
    rng_seed: int,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run two MCTS searches at the same initial state: uniform prior vs trained prior."""
    if verbose:
        print(f"\n--- {problem_key}: {problem_def['problem']!r} ---", flush=True)

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

    # --- uniform prior ---
    if verbose:
        print(f"  MCTS with UNIFORM prior...", flush=True)
    t0 = time.time()
    rng = np.random.default_rng(rng_seed)
    root_uniform = mcts_search(mdp, state, config_uniform, rng)
    uniform_duration = time.time() - t0
    uniform_summary = summarize_root(root_uniform)
    uniform_summary["wallclock_s"] = round(uniform_duration, 2)
    if verbose:
        print(f"    done in {uniform_duration:.1f}s; visits={uniform_summary['visit_counts']}", flush=True)

    # --- trained prior ---
    if verbose:
        print(f"  MCTS with TRAINED prior...", flush=True)
    t0 = time.time()
    rng = np.random.default_rng(rng_seed)
    root_trained = mcts_search(mdp, state, config_trained, rng)
    trained_duration = time.time() - t0
    trained_summary = summarize_root(root_trained)
    trained_summary["wallclock_s"] = round(trained_duration, 2)
    if verbose:
        print(f"    done in {trained_duration:.1f}s; visits={trained_summary['visit_counts']}", flush=True)

    return {
        "problem_key": problem_key,
        "problem_statement": problem_def["problem"],
        "expected_answer": problem_def["expected"],
        "uniform": uniform_summary,
        "trained": trained_summary,
    }


def summarize_root(root: DecisionNode) -> dict[str, Any]:
    visit_counts = {}
    q_means = {}
    chance_samples = {}
    priors = {}
    for action, chance in root.children.items():
        verb_name = action.verb.value
        visit_counts[verb_name] = chance.visits
        q_means[verb_name] = round(chance.mean_value, 4)
        chance_samples[verb_name] = len(chance.children)
        priors[verb_name] = round(chance.prior, 4)

    pi_dict = visit_count_policy(root, temperature=1.0)
    pi_mcts = {action.verb.value: round(float(p), 4) for action, p in pi_dict.items()}

    if root.children:
        recommended = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
        rec_verb = recommended.verb.value
    else:
        rec_verb = None

    return {
        "root_visits": root.visits,
        "untried_actions": [a.verb.value for a in root.untried_actions],
        "visit_counts": visit_counts,
        "priors": priors,
        "mean_value_per_action": q_means,
        "pi_mcts": pi_mcts,
        "chance_node_children_per_action": chance_samples,
        "recommended_action": rec_verb,
    }


# -----------------------------------------------------------------------------
# Comparison print
# -----------------------------------------------------------------------------


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    print()
    print("=" * 100)
    print("MCTS COMPARISON: UNIFORM vs TRAINED PRIOR")
    print("=" * 100)

    for r in results:
        print(f"\n{r['problem_key']}:  {r['problem_statement']!r}")
        rec_u = r['uniform']['recommended_action'] or '-'
        rec_t = r['trained']['recommended_action'] or '-'
        print(f"  Uniform  recommended={rec_u:<18} visits={r['uniform']['visit_counts']}")
        print(f"  Trained  recommended={rec_t:<18} visits={r['trained']['visit_counts']}")
        print(f"           priors={r['trained']['priors']}")
        # Compute KL divergence-ish from uniform on visit distributions.
        u_pi = r["uniform"]["pi_mcts"]
        t_pi = r["trained"]["pi_mcts"]
        keys = sorted(set(u_pi) | set(t_pi))
        l1 = sum(abs(u_pi.get(k, 0) - t_pi.get(k, 0)) for k in keys)
        print(f"           L1(uniform_pi_MCTS, trained_pi_MCTS) = {l1:.3f}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://192.168.0.204:11434/v1")
    p.add_argument("--embed-url", default="http://192.168.0.204:11434/api/embed")
    p.add_argument("--model", default="qwen2.5:3b")
    p.add_argument(
        "--embed-model",
        default="nomic-embed-text",
        help="must match the embedder the checkpoint was trained on (embed_dim)",
    )
    p.add_argument("--checkpoint", required=True, help="trained heads .pt file")
    p.add_argument("--num-simulations", type=int, default=20)
    p.add_argument("--c-puct", type=float, default=1.0)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--rollout-depth", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--depth-limit", type=int, default=3)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--prior-smoothing", type=float, default=0.02)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument(
        "--problems",
        nargs="+",
        choices=list(PROBLEMS.keys()) + ["all", "train", "train_v4", "train_v4b", "holdout"],
        default=["multiply_small"],
    )
    p.add_argument("--summary-json", default=None)
    args = p.parse_args(argv)

    embed_model = args.embed_model
    problem_keys = resolve_problem_selection(args.problems)

    print(f"=== MCTS comparison: uniform vs trained prior ===", flush=True)
    print(f"model: {args.model} | embed_model: {embed_model}", flush=True)
    print(f"checkpoint: {args.checkpoint}", flush=True)
    print(f"problems: {problem_keys}", flush=True)
    print(f"MCTS B={args.num_simulations}, rollout_depth={args.rollout_depth}, c_puct={args.c_puct}", flush=True)
    print(f"rng_seed={args.rng_seed}, prior_smoothing={args.prior_smoothing}", flush=True)

    model, ckpt = load_actor_from_checkpoint(args.checkpoint)
    print(
        f"\nLoaded TrainableActor: "
        f"{sum(p.numel() for p in model.parameters()):,} params, "
        f"embed_dim={ckpt['embed_dim']}",
        flush=True,
    )

    llm_client = OpenAICompatibleLLMClient(
        base_url=args.base_url,
        model=args.model,
        timeout_s=args.timeout,
        max_retries=1,
    )

    results: list[dict[str, Any]] = []
    started = time.time()

    # The trained search fetches embeddings through this cache; the uniform
    # search (prior_fn=None) never touches embeddings at all.
    cache_trained = EmbeddingCache(
        make_embedding_client(args.embed_url, embed_model, timeout_s=args.timeout)
    )

    config_uniform = MCTSConfig(
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        discount=args.discount,
        rollout_depth_limit=args.rollout_depth,
        prior_fn=None,
        rollout_policy=None,
    )
    trained_prior = make_trained_prior(model, cache_trained, smoothing=args.prior_smoothing)
    config_trained = MCTSConfig(
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        discount=args.discount,
        rollout_depth_limit=args.rollout_depth,
        prior_fn=trained_prior,
        rollout_policy=None,
    )

    for key in problem_keys:
        result = run_one_comparison(
            key,
            PROBLEMS[key],
            llm_client=llm_client,
            config_uniform=config_uniform,
            config_trained=config_trained,
            depth_limit=args.depth_limit,
            max_turns=args.max_turns,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            rng_seed=args.rng_seed,
        )
        results.append(result)

    total_duration = time.time() - started

    print_comparison_table(results)

    print()
    print("=" * 100)
    print(f"Total wall-clock: {total_duration:.1f}s", flush=True)
    print(f"Embedding cache (trained run): {cache_trained.stats()}", flush=True)

    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump(
                {
                    "results": results,
                    "total_wallclock_s": round(total_duration, 2),
                    "embedding_fetches_trained": cache_trained.fetch_count,
                    "embedding_hits_trained": cache_trained.hit_count,
                    "embedding_failures_trained": cache_trained.failure_count,
                    "embedding_fetch_wallclock_s_trained": round(cache_trained.fetch_wallclock_s, 2),
                    "args": vars(args),
                },
                f,
                indent=2,
            )
        print(f"Wrote {args.summary_json}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
