"""Rung-1 runner: LLM-as-controller over the Deliberation MDP.

Runs the controller-teacher loop on a list of problems and writes one
JSONL trajectory log (the Phase-B behavior-cloning dataset) plus a
summary line per problem. With --backend echo it needs no model at all
(a smoke run of the full interface); with --backend http it talks to any
OpenAI-compatible server for BOTH the controller and the generator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from verifier_anchored_rl.deliberation_mdp import DeliberationMDP
from verifier_anchored_rl.llm_client import (
    EchoLLMClient, OpenAICompatibleLLMClient,
)
from verifier_anchored_rl.llm_controller import LLMController, run_episode

DEMO_PROBLEMS = [
    {"problem": "What is 6*7?", "answer": "42"},
    {"problem": "What is 12 + 30?", "answer": "42"},
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["echo", "http"], default="echo")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--controller-model", default=None,
                    help="teacher model; defaults to --model")
    ap.add_argument("--problems", type=Path, default=None,
                    help="JSONL file with {problem, answer} records")
    ap.add_argument("--out", type=Path, default=Path("trajectories.jsonl"))
    ap.add_argument("--depth-limit", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.backend == "echo":
        gen_client = EchoLLMClient()
        ctrl_client = EchoLLMClient()
    else:
        gen_client = OpenAICompatibleLLMClient(
            base_url=args.base_url, model=args.model)
        ctrl_client = OpenAICompatibleLLMClient(
            base_url=args.base_url,
            model=args.controller_model or args.model)

    if args.problems is not None:
        problems = [json.loads(l) for l in args.problems.read_text().splitlines() if l.strip()]
    else:
        problems = DEMO_PROBLEMS

    controller = LLMController(ctrl_client)
    rng = np.random.default_rng(args.seed)
    with args.out.open("w") as fh:
        for i, p in enumerate(problems):
            mdp = DeliberationMDP(problem=p["problem"],
                                  expected_answer=p["answer"],
                                  depth_limit=args.depth_limit,
                                  llm_client=gen_client)
            summary = run_episode(
                mdp, controller, rng,
                log_fn=lambda rec: fh.write(json.dumps(rec) + "\n"),
                episode_id=i,
                meta={"backend": args.backend, "model": args.model,
                      "controller_model": args.controller_model or args.model,
                      "seed": args.seed, "problem_index": i})
            print(f"[{i}] reward={summary.reward:+.2f} "
                  f"turns={summary.turns} llm_calls={summary.llm_calls} "
                  f"problem={p['problem']!r}")


if __name__ == "__main__":
    main()
