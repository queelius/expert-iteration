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


def _build_eval_stack(model: str, depth_limit: int):
    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.planner_actor import PrefillCache, StateActor
    from verifier_anchored_rl.shared_forward import SharedForwardEncoder
    from verifier_anchored_rl.transformers_client import TransformersLLMClient

    llm = TransformersLLMClient(model_name=model, constrain_json=True)
    encoder = SharedForwardEncoder.from_llm_client(llm)
    actor = HiddenStateActor(embed_dim=encoder.embed_dim)
    state_actor = StateActor(encoder, actor, PrefillCache(capacity=4096),
                             depth_limit=depth_limit)
    return llm, actor, state_actor


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="run dir holding checkpoint.pt")
    p.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
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
    from verifier_anchored_rl.run_artifacts import load_checkpoint

    run_dir = Path(args.run_dir)
    out_path = run_dir / "holdout_eval.json"
    results = (
        json.loads(out_path.read_text())
        if out_path.exists() and not args.force
        else {"checkpoint": str(run_dir / "checkpoint.pt"), "model": args.model,
              "per_cell": {}}
    )

    ckpt = load_checkpoint(str(run_dir / "checkpoint.pt"))
    llm, actor, state_actor = _build_eval_stack(args.model, args.depth_limit)
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
                out_path.write_text(json.dumps(results, indent=2))  # checkpoint progress
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
    out_path.write_text(json.dumps(results, indent=2))
    print("summary:", json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
