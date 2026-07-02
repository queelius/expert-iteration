"""End-to-end LLM smoke test against a remote Ollama / vLLM server.

The script drives ``LLMContextMDP`` against a real model through the
OpenAI-compatible HTTP API. There is no trained policy or Q-head; verbs
are selected by a hand-coded scripted policy or a uniform-random one,
because the smoke test only asks whether the framework's loop closes
end to end against a real model and a real verifier.

What this script verifies, in increasing strictness:

1. Network reachability of the inference server.
2. The model produces JSON that conforms to per-verb schemas.
3. The structured-record parser accepts the LLM output.
4. The symbolic verifier accepts at least one correct verifier call.
5. ``COMMIT_ANSWER`` yields the terminal verifier outcome we expect.

If any of (1) through (5) fails, the failure mode is informative. If
(2) fails repeatedly, the model is too small or the prompt is too
demanding. If (3) fails, the serializer-parser round trip has slipped
on something the LLM produced. If (4) fails, the verifier dispatch
needs adjustment. If (5) fails, ``check_answer`` is too strict or the
LLM committed the wrong answer.

Usage
-----
::

    python experiments/run_llm_smoke.py \\
        --base-url http://192.168.0.204:11434/v1 \\
        --model qwen2.5-math:1.5b

The defaults assume an Ollama server at the IP and port above. Pass
``--problem`` to substitute a different toy problem, ``--policy`` to
choose between ``scripted`` and ``random``, and ``--seed`` for
reproducibility of the random policy.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from typing import Sequence

from verifier_anchored_rl import (
    Fact,
    LLMContextMDP,
    MathAction,
    OpenAICompatibleLLMClient,
    Verb,
    available_actions,
    serialize_state,
)


# -----------------------------------------------------------------------------
# Toy problems
# -----------------------------------------------------------------------------


PROBLEMS = {
    "multiply_small": {
        "problem": "Compute 7 * 6.",
        "expected": "42",
        "given_facts": [],
    },
    "gsm8k_toy_1": {
        "problem": (
            "Alice has 3 boxes of apples. Each box contains 12 apples. "
            "She gives away 7 apples. How many apples does she have left?"
        ),
        "expected": "29",
        "given_facts": [],
    },
    "algebra_toy_1": {
        "problem": "Solve for x: 2x + 5 = 13.",
        "expected": "4",
        "given_facts": [],
    },
    "factor": {
        "problem": "Factor the expression x^2 - 9.",
        "expected": "(x - 3)*(x + 3)",
        "given_facts": [],
    },
}


SCRIPTED_VERB_SEQUENCE: Sequence[Verb] = (
    Verb.FORM_HYPOTHESIS,
    Verb.INVOKE_VERIFIER,
    Verb.COMMIT_ANSWER,
)


# -----------------------------------------------------------------------------
# Policies
# -----------------------------------------------------------------------------


def scripted_policy(turn: int, available: Sequence[Verb]) -> Verb:
    """Hand-coded three-step sequence: form hypothesis, verify, commit.

    Falls back to the first available verb if the scripted choice is
    unavailable at the current turn.
    """
    if turn < len(SCRIPTED_VERB_SEQUENCE):
        candidate = SCRIPTED_VERB_SEQUENCE[turn]
        if candidate in available:
            return candidate
    return available[0]


def random_policy(rng: random.Random, available: Sequence[Verb]) -> Verb:
    """Uniform-random verb selection over the available set."""
    return rng.choice(list(available))


# -----------------------------------------------------------------------------
# Smoke test loop
# -----------------------------------------------------------------------------


def run_one(
    problem_key: str,
    mdp: LLMContextMDP,
    policy_name: str,
    rng: random.Random,
    *,
    verbose: bool = True,
) -> dict:
    """Run one trajectory and return a summary dict."""
    state = mdp.initial_state(
        given_facts=[Fact(**fd) for fd in PROBLEMS[problem_key]["given_facts"]],
    )
    verbs_used: list[str] = []
    rewards: list[float] = []
    failures: list[str] = []

    started = time.time()
    while not mdp.is_terminal(state):
        available = available_actions(state, depth_limit=mdp.depth_limit)
        if not available:
            failures.append(f"turn {state.turn}: no actions available")
            break

        if policy_name == "scripted":
            verb = scripted_policy(state.turn, available)
        else:
            verb = random_policy(rng, available)

        if verbose:
            print(f"\n--- turn {state.turn}: {verb.value} ---")
            print(f"available: {[v.value for v in available]}")

        action = MathAction(verb=verb)
        before = state
        state, reward = mdp.sample_transition(state, action)
        verbs_used.append(verb.value)
        rewards.append(reward)

        if verbose:
            print(f"reward: {reward:+.3f}")
            print(f"new turn count: {state.turn}")
            if state.facts != before.facts:
                added = [f for f in state.facts if f not in before.facts]
                for f in added:
                    print(f"  + fact {f.id} ({f.source}): {f.statement}")
            if state.hypotheses != before.hypotheses:
                added = [h for h in state.hypotheses if h not in before.hypotheses]
                for h in added:
                    parent = h.parent_hypothesis or h.parent_subgoal or "(none)"
                    print(f"  + hypothesis {h.id} (parent={parent}): {h.statement}")
            if state.verifier_log != before.verifier_log:
                added = [v for v in state.verifier_log if v not in before.verifier_log]
                for v in added:
                    print(
                        f"  + verifier_call {v.id} ({v.verb}, success={v.success}):"
                        f" {v.input} -> {v.output}"
                    )
            if state.committed_answer is not None and before.committed_answer is None:
                print(f"  ! committed answer: {state.committed_answer!r}")

    duration = time.time() - started

    if verbose:
        print("\n=== final state ===")
        print(serialize_state(state))

    return {
        "problem": problem_key,
        "expected": PROBLEMS[problem_key]["expected"],
        "committed": state.committed_answer,
        "policy": policy_name,
        "verbs_used": verbs_used,
        "rewards": rewards,
        "total_reward": sum(rewards),
        "turns": state.turn,
        "terminal": mdp.is_terminal(state),
        "duration_seconds": round(duration, 2),
        "failures": failures,
    }


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://192.168.0.204:11434/v1",
        help="OpenAI-compatible inference server URL.",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-math:1.5b",
        help="Model name as known by the inference server (Ollama tag).",
    )
    parser.add_argument(
        "--problem",
        choices=list(PROBLEMS.keys()) + ["all"],
        default="multiply_small",
        help="Which toy problem to run, or 'all' to run every problem in turn.",
    )
    parser.add_argument(
        "--policy",
        choices=["scripted", "random"],
        default="scripted",
        help="Verb-selection policy.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth-limit", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to write a JSON summary of the run.",
    )
    args = parser.parse_args(argv)

    client = OpenAICompatibleLLMClient(
        base_url=args.base_url,
        model=args.model,
        timeout_s=args.timeout,
        max_retries=args.max_retries,
    )

    problem_keys = (
        list(PROBLEMS.keys()) if args.problem == "all" else [args.problem]
    )

    summaries = []
    rng = random.Random(args.seed)

    for key in problem_keys:
        problem_def = PROBLEMS[key]
        if not args.quiet:
            print(f"\n========== {key} ==========")
            print(f"problem: {problem_def['problem']}")
            print(f"expected answer: {problem_def['expected']}")
            print(f"model: {args.model}")
            print(f"base url: {args.base_url}")
            print(f"policy: {args.policy}")

        mdp = LLMContextMDP(
            problem_statement=problem_def["problem"],
            expected_answer=problem_def["expected"],
            llm_client=client,
            depth_limit=args.depth_limit,
            max_turns=args.max_turns,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            prompt_strategy="full",
        )

        try:
            summary = run_one(key, mdp, args.policy, rng, verbose=not args.quiet)
        except Exception as e:
            summary = {
                "problem": key,
                "expected": problem_def["expected"],
                "error": f"{type(e).__name__}: {e}",
            }

        if not args.quiet:
            print(f"\n--- summary ({key}) ---")
            print(json.dumps(summary, indent=2, default=str))

        summaries.append(summary)

    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump(summaries, f, indent=2, default=str)
        if not args.quiet:
            print(f"\nWrote summary JSON to {args.summary_json}")

    failures = sum(1 for s in summaries if "error" in s or s.get("total_reward", 0) < 0)
    successes = len(summaries) - failures
    if not args.quiet:
        print(f"\n==== Smoke test complete: {successes}/{len(summaries)} problems with non-negative reward ====")

    return 0 if successes > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
