"""Rung-1 runner: LLM-as-controller over the Deliberation MDP.

Runs the controller-teacher loop on a list of problems and writes one
JSONL trajectory log (the Phase-B behavior-cloning dataset) plus a
summary line per problem. With --backend echo it needs no model at all
(a smoke run of the full interface); with --backend http it talks to any
OpenAI-compatible server; with --backend openai it talks to
api.openai.com using the key from --api-key-env.

The controller (teacher) and the generator can use DIFFERENT backends,
which is the design's intended collection regime (design.md section 7.3):
a strong API teacher picks (verb, target) while the small deployed
generator writes all content, so trajectories reflect the deployed
model's state distribution. Example:

    python experiments/run_llm_controller.py \
        --backend http --base-url http://gpu-box:11434/v1 \
        --model qwen2.5:1.5b-instruct \
        --controller-backend openai --controller-model gpt-4.1-mini \
        --problems experiments/problems/gsm8k-train-500.jsonl \
        --workers 3 --token-budget 15000000 --out trajectories.jsonl
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
from verifier_anchored_rl.llm_controller import (
    CONTROLLER_SYSTEM, CONTROLLER_SYSTEM_STRUCTURED, LLMController, run_episode,
)

_CONTROLLER_PROMPTS = {"default": CONTROLLER_SYSTEM,
                       "structured": CONTROLLER_SYSTEM_STRUCTURED}

DEMO_PROBLEMS = [
    {"problem": "What is 6*7?", "answer": "42"},
    {"problem": "What is 12 + 30?", "answer": "42"},
]


def _make_client(backend: str, base_url: str, model, api_key_env: str):
    """One generation client for a role. echo ignores the other params."""
    if backend == "echo":
        return EchoLLMClient()
    import os
    resolved = ("https://api.openai.com/v1" if backend == "openai"
                else base_url)
    api_key = os.environ.get(api_key_env) or None
    if backend == "openai" and not api_key:
        raise SystemExit(f"backend openai needs a key in ${api_key_env}")
    return OpenAICompatibleLLMClient(base_url=resolved, model=model,
                                     api_key=api_key)


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
    system = _CONTROLLER_PROMPTS[args.controller_prompt]
    if args.solve_claims == "on":
        # The on-arm teacher must not contradict the on-arm lens: the
        # default text says unknowns never verify, which is exactly what
        # solve-claims relaxes for case-goal facts.
        system += (
            "\n\nSolve-claims are enabled: a fact under an equation case "
            'may state the solution directly, like "x = 5"; the checker '
            "substitutes it into that case's equation.")
    controller = LLMController(ctrl_client,
                               system=system,
                               max_tokens=args.controller_max_tokens)
    rng = np.random.default_rng(args.seed + i)
    buffer: list[dict] = []
    summary = run_episode(
        mdp := DeliberationMDP(problem=p["problem"],
                               expected_answer=p["answer"],
                               depth_limit=args.depth_limit,
                               llm_client=gen_client,
                               solve_claims=args.solve_claims == "on"),
        controller, rng,
        log_fn=buffer.append,
        episode_id=i,
        meta={"backend": args.backend, "model": args.model,
              "controller_backend": args.controller_backend or args.backend,
              "controller_model": args.controller_model or args.model,
              "controller_prompt": args.controller_prompt,
              "solve_claims": args.solve_claims,
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
    ap.add_argument("--controller-backend",
                    choices=["echo", "http", "openai"], default=None,
                    help="teacher backend; defaults to --backend. Lets a "
                         "strong API teacher drive a local generator.")
    ap.add_argument("--controller-base-url", default=None,
                    help="teacher base URL; defaults to --base-url")
    ap.add_argument("--controller-prompt",
                    choices=list(_CONTROLLER_PROMPTS), default="default",
                    help="'structured' foregrounds split/abandon for "
                         "case-analysis problems (casework dataset)")
    ap.add_argument("--controller-max-tokens", type=int, default=64,
                    help="completion cap for the teacher's action choice; "
                         "large models narrate before the JSON and get "
                         "truncated at the 64 default (experiment E11)")
    ap.add_argument("--solve-claims", choices=["on", "off"], default="off",
                    help="E13 arm: COMPUTE on an equation goal may claim "
                         "'x = 5', verified by substitution (spec "
                         "2026-07-03-solve-claim-compute)")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY",
                    help="environment variable holding the bearer token "
                         "(used by http and openai backends when set)")
    ap.add_argument("--problems", type=Path, default=None,
                    help="JSONL file with {problem, answer} records")
    ap.add_argument("--out", type=Path, default=Path("trajectories.jsonl"))
    ap.add_argument("--depth-limit", type=int, default=16)
    ap.add_argument("--max-problems", type=int, default=None,
                    help="budget guard: run at most this many problems")
    ap.add_argument("--start-index", type=int, default=0,
                    help="resume support: skip the first N problems while "
                         "keeping each surviving problem's TRUE index as its "
                         "episode id, so a resumed shard's ids never collide "
                         "with an earlier shard's on merge")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent episodes (each on its own controller)")
    ap.add_argument("--token-budget", type=int, default=None,
                    help="hard stop: no new episodes once total measured "
                         "tokens (prompt + completion) exceed this")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gen_client = _make_client(args.backend, args.base_url, args.model,
                              args.api_key_env)
    ctrl_client = _make_client(args.controller_backend or args.backend,
                               args.controller_base_url or args.base_url,
                               args.controller_model or args.model,
                               args.api_key_env)

    if args.problems is not None:
        problems = [json.loads(l) for l in args.problems.read_text().splitlines() if l.strip()]
    else:
        problems = DEMO_PROBLEMS
    # Enumerate BEFORE slicing so each surviving problem keeps its true index
    # (its episode id / meta.problem_index) across a resumed run.
    indexed = list(enumerate(problems))[args.start_index:]
    if args.max_problems is not None:
        indexed = indexed[: args.max_problems]

    write_lock = threading.Lock()
    wins = 0
    done = 0
    budget_hit = False
    with args.out.open("w") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = set()
        it = iter(indexed)
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
