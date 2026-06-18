"""Run the outer collection-and-train loop on toy problems (method.tex 3.7).

This is the GPU-bound integration of the three CPU-verified modules
(architecture/train/replay) with the planner (mcts) and the in-process
encoder (shared_forward). It runs a few outer iterations on the toy `train`
problem group, sharing one frozen model between generation and the heads'
embeddings, and prints the per-iteration loss curve and FLOP counters.

Usage::

    python experiments/run_outer_loop.py \\
        --model Qwen/Qwen2.5-Math-1.5B-Instruct \\
        --iterations 3 --num-simulations 16 --train-steps 200 \\
        --batch-size 16 --max-turns 8
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--num-simulations", type=int, default=16)
    p.add_argument("--train-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--min-buffer", type=int, default=16)
    p.add_argument("--capacity", type=int, default=2048)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--discount", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--sync-interval", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--depth-limit", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--problems", nargs="+", default=["train"])
    args = p.parse_args(argv)

    import torch

    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.llm_context_mdp import LLMContextMDP
    from verifier_anchored_rl.outer_loop import run_outer_loop
    from verifier_anchored_rl.planner_actor import PrefillCache, StateActor
    from verifier_anchored_rl.replay import ReplayBuffer
    from verifier_anchored_rl.shared_forward import SharedForwardEncoder
    from verifier_anchored_rl.train import Trainer
    from verifier_anchored_rl.transformers_client import TransformersLLMClient

    try:
        from problems import PROBLEMS, resolve_problem_selection
    except ImportError:
        from experiments.problems import PROBLEMS, resolve_problem_selection

    keys = resolve_problem_selection(args.problems)
    problems = [(PROBLEMS[k]["problem"], PROBLEMS[k]["expected"]) for k in keys]
    print(f"=== Outer loop: {len(problems)} problems, model={args.model} ===",
          flush=True)

    llm_client = TransformersLLMClient(model_name=args.model)
    encoder = SharedForwardEncoder.from_llm_client(llm_client)
    cache = PrefillCache(capacity=4096)
    actor = HiddenStateActor(embed_dim=encoder.embed_dim)
    state_actor = StateActor(encoder, actor, cache, depth_limit=args.depth_limit)
    buffer = ReplayBuffer(capacity=args.capacity)
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    trainer = Trainer(actor, opt, buffer, sync_interval=args.sync_interval)

    def mdp_factory(problem):
        return LLMContextMDP(
            problem_statement=problem[0], expected_answer=problem[1],
            llm_client=llm_client, encoder=encoder, prefill_cache=cache,
            depth_limit=args.depth_limit, max_turns=args.max_turns)

    stats = run_outer_loop(
        problems, mdp_factory, state_actor, trainer, buffer,
        n_iterations=args.iterations, train_steps_per_iter=args.train_steps,
        batch_size=args.batch_size, min_buffer_size=args.min_buffer,
        num_simulations=args.num_simulations, c_puct=args.c_puct,
        discount=args.discount, tau=args.tau, max_turns=args.max_turns,
        rng=np.random.default_rng(args.seed), encoder=encoder)

    print("\n=== Per-iteration summary ===", flush=True)
    for s in stats:
        first = s.losses[0]["total"] if s.losses else float("nan")
        last = s.losses[-1]["total"] if s.losses else float("nan")
        print(
            f"iter {s.iteration}: decisions={s.n_decisions} "
            f"sims={s.n_simulations} buffer={s.buffer_size} "
            f"prefill={s.prefill_calls} ({s.prefill_tokens} tok) "
            f"gen={s.gen_calls} ({s.gen_tokens} tok) "
            f"L_total {first:.4f} -> {last:.4f}",
            flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
