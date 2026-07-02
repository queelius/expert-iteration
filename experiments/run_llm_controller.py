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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _client_tokens(*clients) -> int:
    """Total billable tokens (prompt + completion) across HTTP clients."""
    total = 0
    for c in clients:
        total += getattr(c, "prompt_tokens", 0) + getattr(c, "generated_tokens", 0)
    return total


def _run_one(i, p, args, gen_client, ctrl_client, write_lock, fh):
    """One episode on its own controller instance (its fallback flag is
    per-episode mutable state) and its own seeded rng; records are buffered
    and written contiguously under the lock."""
    controller = LLMController(ctrl_client)
    rng = np.random.default_rng(args.seed + i)
    buffer: list[dict] = []
    summary = run_episode(
        mdp := DeliberationMDP(problem=p["problem"],
                               expected_answer=p["answer"],
                               depth_limit=args.depth_limit,
                               llm_client=gen_client),
        controller, rng,
        log_fn=buffer.append,
        episode_id=i,
        meta={"backend": args.backend, "model": args.model,
              "controller_model": args.controller_model or args.model,
              "seed": args.seed, "problem_index": i})
    with write_lock:
        for rec in buffer:
            fh.write(json.dumps(rec) + "\n")
        fh.flush()
    return i, p, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["echo", "http", "openai"],
                    default="echo",
                    help="openai = https://api.openai.com/v1 with the key "
                         "from --api-key-env")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--controller-model", default=None,
                    help="teacher model; defaults to --model")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY",
                    help="environment variable holding the bearer token "
                         "(used by http and openai backends when set)")
    ap.add_argument("--problems", type=Path, default=None,
                    help="JSONL file with {problem, answer} records")
    ap.add_argument("--out", type=Path, default=Path("trajectories.jsonl"))
    ap.add_argument("--depth-limit", type=int, default=16)
    ap.add_argument("--max-problems", type=int, default=None,
                    help="budget guard: run at most this many problems")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent episodes (each on its own controller)")
    ap.add_argument("--token-budget", type=int, default=None,
                    help="hard stop: no new episodes once total measured "
                         "tokens (prompt + completion) exceed this")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.backend == "echo":
        gen_client = EchoLLMClient()
        ctrl_client = EchoLLMClient()
    else:
        import os
        base_url = ("https://api.openai.com/v1" if args.backend == "openai"
                    else args.base_url)
        api_key = os.environ.get(args.api_key_env) or None
        if args.backend == "openai" and not api_key:
            raise SystemExit(
                f"--backend openai needs a key in ${args.api_key_env}")
        gen_client = OpenAICompatibleLLMClient(
            base_url=base_url, model=args.model, api_key=api_key)
        ctrl_client = OpenAICompatibleLLMClient(
            base_url=base_url,
            model=args.controller_model or args.model,
            api_key=api_key)

    if args.problems is not None:
        problems = [json.loads(l) for l in args.problems.read_text().splitlines() if l.strip()]
    else:
        problems = DEMO_PROBLEMS
    if args.max_problems is not None:
        problems = problems[: args.max_problems]

    write_lock = threading.Lock()
    wins = 0
    done = 0
    budget_hit = False
    with args.out.open("w") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = set()
        it = iter(enumerate(problems))
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < args.workers:
                if (args.token_budget is not None
                        and _client_tokens(gen_client, ctrl_client)
                        > args.token_budget):
                    budget_hit = True
                    exhausted = True
                    break
                try:
                    i, p = next(it)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(pool.submit(_run_one, i, p, args,
                                        gen_client, ctrl_client,
                                        write_lock, fh))
            if not pending:
                break
            for fut in as_completed(pending):
                pending.discard(fut)
                i, p, summary = fut.result()
                done += 1
                wins += summary.reward > 0
                print(f"[{i}] reward={summary.reward:+.2f} "
                      f"turns={summary.turns} llm_calls={summary.llm_calls} "
                      f"problem={p['problem'][:60]!r}")
                break   # refill the pool after each completion

    total_prompt = sum(getattr(c, "prompt_tokens", 0)
                       for c in {id(gen_client): gen_client,
                                 id(ctrl_client): ctrl_client}.values())
    total_gen = sum(getattr(c, "generated_tokens", 0)
                    for c in {id(gen_client): gen_client,
                              id(ctrl_client): ctrl_client}.values())
    print(f"\nepisodes={done} wins={wins} "
          f"prompt_tokens={total_prompt} completion_tokens={total_gen}"
          + (" [TOKEN BUDGET HIT: run stopped early]" if budget_hit else ""))


if __name__ == "__main__":
    main()
