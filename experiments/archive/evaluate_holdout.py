"""Held-out task-success evaluation of a trained prior (experiment apparatus).

Loads a closed-loop run's checkpoint and measures whether the trained policy
SOLVES held-out problems (the verifier confirms the committed answer), at B=0
(greedy, the fast inference mode: argmax of the learned prior, no search) and at
B>0 (search, the test-time-compute mode: MCTS with the trained prior and the
target-Q leaf, then argmax visit count). This is the in-loop analog of the
pilot's held-out generalization metric, and it exercises the inference-time
compute knob the framework describes.

Idempotent and resumable: results are written to holdout_eval.json in the run
dir and accumulated per (budget, problem, seed), so re-invoking skips cells
already evaluated (unless --force). Verb generation uses the constrained chat
path (the only path that produces valid actions on the small model).

Stack reconstruction:
  _build_eval_stack reads the run's metadata.json to reconstruct the SAME
  embedder and generation backend used during training.  Missing keys default
  to the legacy hidden_state/transformers behavior so old E1 checkpoints still
  evaluate without any change to their metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def rollout(mdp, state_actor, *, budget, c_puct, discount, max_turns, rng):
    """Run the policy to a terminal state (or max_turns) and report whether the
    committed answer verifies. budget=0 is greedy (argmax of the learned prior);
    budget>0 runs MCTS and takes the argmax visit count."""
    from verifier_anchored_rl.mcts import MCTSConfig, best_action, mcts_search

    state = mdp.initial_state()
    total = 0.0
    turns = 0
    while not mdp.is_terminal(state) and state.turn < max_turns:
        actions = list(mdp.actions(state))
        if not actions:
            break
        if budget and budget > 0:
            cfg = MCTSConfig(
                num_simulations=budget, c_puct=c_puct, discount=discount,
                prior_fn=state_actor.prior, value_fn=state_actor.leaf_value,
            )
            root = mcts_search(mdp, state, cfg, rng)
            if not root.children:
                break
            chosen = best_action(root)
        else:
            probs = state_actor.prior(state, actions)
            chosen = actions[int(np.argmax(probs))]
        state, reward = mdp.sample_transition(state, chosen, rng)
        total += reward
        turns += 1

    committed = state.committed_answer
    success = False
    if committed is not None and mdp.expected_answer is not None:
        from verifier_anchored_rl.verifier import check_answer

        success = bool(check_answer(committed, mdp.expected_answer).success)
    return {"committed": committed, "success": success, "turns": turns,
            "total_reward": total}


def _build_eval_stack(run_dir, depth_limit: int, cli_model=None):
    """Reconstruct the embedder/generation stack used during training.

    Reads run_dir/metadata.json and mirrors the selection from
    run_outer_loop._build_stack.  Missing keys in metadata default to the
    legacy hidden_state/transformers behavior so old E1 checkpoints evaluate
    correctly without metadata changes.

    Parameters
    ----------
    run_dir : path-like
        Path to the run directory holding metadata.json and checkpoint.pt.
    depth_limit : int
        Depth limit passed through to StateActor.
    cli_model : str or None
        If provided, must match the model recorded in metadata (prevents silent
        wrong-model evaluation).  Pass None to use the metadata model silently.

    Returns
    -------
    (llm, actor, state_actor)
    """
    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.planner_actor import PrefillCache, StateActor

    run_dir = Path(run_dir)
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        raise SystemExit(
            f"evaluate_holdout: metadata.json not found in {run_dir}.  "
            "Cannot reconstruct the training stack without run metadata."
        )

    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise SystemExit(
            f"evaluate_holdout: failed to read metadata.json in {run_dir}: {exc}"
        ) from exc

    run_args = meta.get("args", {})

    # Model: prefer the metadata record; accept a matching CLI override.
    meta_model = run_args.get("model") or meta.get("model")
    if meta_model and cli_model and cli_model != meta_model:
        raise SystemExit(
            f"evaluate_holdout: --model {cli_model!r} contradicts the model "
            f"recorded in metadata ({meta_model!r}).  "
            "Pass --model without an argument or use the metadata model to avoid "
            "a silent wrong-model evaluation."
        )
    model = meta_model or cli_model
    if model is None:
        raise SystemExit(
            "evaluate_holdout: no model found in metadata.json and --model not provided."
        )

    # Defaults: legacy hidden_state + transformers for old checkpoints that
    # pre-date the gen_backend/embedder keys.
    gen_backend = run_args.get("gen_backend", "transformers")
    embedder = run_args.get("embedder", "hidden_state")

    # Generation backend.
    if gen_backend == "vllm":
        from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient
        vllm_url = run_args.get("vllm_url", "http://localhost:8000/v1")
        llm = OpenAICompatibleLLMClient(
            base_url=vllm_url, model=model, guided_json=True)
    else:
        from verifier_anchored_rl.transformers_client import TransformersLLMClient
        llm = TransformersLLMClient(model_name=model, constrain_json=True)

    # State embedder.
    if embedder == "text":
        from verifier_anchored_rl.embedder import TextEmbedder
        from verifier_anchored_rl.transformers_client import TransformersEmbeddingClient
        embedder_model = run_args.get("embedder_model", model)
        emb_client = TransformersEmbeddingClient(model_name=embedder_model)
        embed_dim = int(emb_client.model.config.hidden_size)
        encoder = TextEmbedder(emb_client, embed_dim=embed_dim)
    else:
        # hidden_state: legacy h_T path; the gen model must be in-process.
        if gen_backend != "transformers":
            raise SystemExit(
                "evaluate_holdout: embedder=hidden_state requires gen-backend=transformers "
                "(the h_T encoder needs the in-process generation model; "
                "vLLM does not expose hidden states). "
                "Use --embedder text with vLLM, or check the run metadata."
            )
        from verifier_anchored_rl.shared_forward import SharedForwardEncoder
        encoder = SharedForwardEncoder.from_llm_client(llm)

    actor = HiddenStateActor(embed_dim=encoder.embed_dim)
    state_actor = StateActor(encoder, actor, PrefillCache(capacity=4096),
                             depth_limit=depth_limit)
    return llm, actor, state_actor


def _load_results(out_path, force, run_dir, model):
    """Load accumulated holdout_eval results, falling back to a fresh dict on
    torn or missing files.  A crash mid-write must not abort the next resume.

    Parameters
    ----------
    out_path : Path
        Path to holdout_eval.json.
    force : bool
        If True, ignore any existing file and return a fresh dict.
    run_dir : str
        Recorded in the fresh dict for provenance.
    model : str
        Recorded in the fresh dict for provenance.

    Returns
    -------
    dict with keys "checkpoint", "model", "per_cell".
    """
    fresh = {
        "checkpoint": str(Path(run_dir) / "checkpoint.pt"),
        "model": model,
        "per_cell": {},
    }
    if force or not out_path.exists():
        return fresh
    try:
        return json.loads(out_path.read_text())
    except (json.JSONDecodeError, OSError):
        # Torn file from a previous crashed write: start fresh rather than
        # aborting with a parse error, losing all evaluated cells.
        return fresh


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="run dir holding checkpoint.pt")
    p.add_argument("--model", default=None,
                   help="model id; if provided, must match metadata.json (default: read from metadata)")
    p.add_argument("--budgets", type=int, nargs="+", default=[0, 8])
    p.add_argument("--problems", nargs="+", default=["holdout"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--discount", type=float, default=1.0)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--depth-limit", type=int, default=3)
    p.add_argument("--force", action="store_true",
                   help="re-evaluate cells already in holdout_eval.json")
    args = p.parse_args(argv)

    try:
        from problems import PROBLEMS, resolve_problem_selection
    except ImportError:
        from experiments.problems import PROBLEMS, resolve_problem_selection
    from verifier_anchored_rl.llm_context_mdp import LLMContextMDP
    from verifier_anchored_rl.run_artifacts import load_checkpoint, write_json_atomic

    run_dir = Path(args.run_dir)
    out_path = run_dir / "holdout_eval.json"

    llm, actor, state_actor = _build_eval_stack(
        run_dir, args.depth_limit, cli_model=args.model)

    results = _load_results(out_path, args.force, run_dir=run_dir,
                            model=args.model or "")

    ckpt = load_checkpoint(str(run_dir / "checkpoint.pt"))
    actor.load_state_dict(ckpt["actor_state"])

    keys = resolve_problem_selection(args.problems)
    print(f"=== holdout eval: {len(keys)} problems, budgets={args.budgets}, "
          f"seeds={args.seeds}, run_dir={run_dir} ===", flush=True)

    for B in args.budgets:
        for key in keys:
            for seed in args.seeds:
                cell = f"B{B}/{key}/seed{seed}"
                if cell in results["per_cell"] and not args.force:
                    continue
                mdp = LLMContextMDP(
                    problem_statement=PROBLEMS[key]["problem"],
                    expected_answer=PROBLEMS[key]["expected"],
                    llm_client=llm, depth_limit=args.depth_limit,
                    max_turns=args.max_turns)
                res = rollout(mdp, state_actor, budget=B, c_puct=args.c_puct,
                              discount=args.discount, max_turns=args.max_turns,
                              rng=np.random.default_rng(seed))
                res.update(budget=B, problem=key, seed=seed)
                results["per_cell"][cell] = res
                write_json_atomic(out_path, results)  # checkpoint progress
                print(f"{cell}: success={res['success']} turns={res['turns']} "
                      f"committed={res['committed']!r}", flush=True)

    summary = {}
    for B in args.budgets:
        rs = [v for v in results["per_cell"].values() if v["budget"] == B]
        summary[f"B{B}"] = {
            "success_rate": (sum(r["success"] for r in rs) / len(rs)) if rs else None,
            "n": len(rs),
        }
    results["summary"] = summary
    write_json_atomic(out_path, results)
    print("summary:", json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
