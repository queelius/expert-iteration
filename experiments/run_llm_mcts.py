"""First MCTS run against the LLM-driven context MDP.

Goals:

1. Verify ``mcts_search`` runs end-to-end on ``LLMContextMDP`` (satisfies
   ``GenerativeMDP[MathState, MathAction]``).
2. With no learned policy yet, use the uniform PUCT prior. This is the
   floor for what trained ``pi_theta`` has to beat at $B = 0$ and the
   worst-case prior version of deployment-time search mode.
3. Time every LLM call individually so we can attribute wall-clock
   between MCTS bookkeeping and the LLM (and detect cold reloads).

The script runs ONE MCTS search at the initial state of a toy problem
and reports:

* Visit counts at the root.
* Q estimates per verb at the root.
* Recommended action (argmax visit).
* Per-LLM-call timings.
* Total wall-clock and the LLM share of it.

No trajectory execution after the search. The point of this script is
"does MCTS work on LLMContextMDP and what does the chance-node estimator
say after $B$ simulations?", not "can MCTS solve a problem."
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from verifier_anchored_rl import (
    LLMContextMDP,
    MCTSConfig,
    OpenAICompatibleLLMClient,
    Verb,
    available_actions,
    mcts_search,
    root_q_estimates,
    visit_count_policy,
    serialize_state,
)
from verifier_anchored_rl.mcts import DecisionNode


# -----------------------------------------------------------------------------
# Toy problems (same canonical four from the smoke test)
# -----------------------------------------------------------------------------


try:
    # Run as a script: python experiments/run_llm_mcts.py
    from problems import PROBLEMS
except ImportError:
    # Imported as a module from the repo root.
    from experiments.problems import PROBLEMS



# -----------------------------------------------------------------------------
# Timing wrapper around the LLM client
# -----------------------------------------------------------------------------


@dataclass
class TimingLLMClient:
    """Wrap an LLMClient and record one timestamped entry per ``complete_json`` call.

    Emits a one-line progress message per call so the operator can see the
    search running in real time and detect cold reloads or stuck calls.
    """

    inner: Any
    calls: list[dict[str, Any]] = field(default_factory=list)
    verbose: bool = True

    def complete_json(
        self,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        started = time.time()
        verb = _extract_verb(user)
        idx = len(self.calls) + 1
        if self.verbose:
            print(f"  [{idx:>3d}] -> {verb or '?':<16s}", end="", flush=True)
        try:
            result = self.inner.complete_json(
                system=system,
                user=user,
                schema=schema,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            ok = True
        except Exception as e:
            duration = time.time() - started
            self.calls.append(
                {
                    "ok": False,
                    "duration_s": round(duration, 3),
                    "error": f"{type(e).__name__}: {e}",
                    "verb_hint": verb,
                }
            )
            if self.verbose:
                print(f"  FAILED in {duration:.1f}s: {type(e).__name__}", flush=True)
            raise
        duration = time.time() - started
        self.calls.append(
            {
                "ok": ok,
                "duration_s": round(duration, 3),
                "verb_hint": verb,
            }
        )
        if self.verbose:
            print(f"  ok in {duration:5.1f}s", flush=True)
        return result


import re as _re
_VERB_RE = _re.compile(r"Verb:\s*([A-Z_]+)")


def _extract_verb(user_message: str) -> str | None:
    m = _VERB_RE.search(user_message)
    return m.group(1) if m else None


# -----------------------------------------------------------------------------
# Tree visualization
# -----------------------------------------------------------------------------


def _state_summary(state) -> str:
    """One-line summary of a MathState for tree display."""
    parts = [f"turn={state.turn}"]
    if state.facts:
        parts.append(f"facts={len(state.facts)}")
    if state.hypotheses:
        parts.append(f"hyp={len(state.hypotheses)}")
    if state.verifier_log:
        last = state.verifier_log[-1]
        parts.append(
            f"vcalls={len(state.verifier_log)} (last: {last.verb} success={last.success})"
        )
    if state.committed_answer is not None:
        parts.append(f"commit={state.committed_answer!r}")
    if state.abandoned:
        parts.append(f"abandoned={list(state.abandoned)}")
    return " ".join(parts)


def _terminal_mark(node) -> str:
    if node.terminal:
        return " [TERMINAL]"
    return ""


def print_tree(
    root: DecisionNode,
    max_depth: int = 8,
    max_chance_children: int = 4,
) -> None:
    """Pretty-print the MCTS tree from the root.

    The tree alternates decision nodes (states) and chance nodes
    (state-action edges). At each decision node we print state summary
    and visit count; at each chance node we print the verb, visits,
    mean value (Q-hat), and the immediate sampled-successor count.

    ``max_depth`` is in alternating-node units (1 root level + N
    descendant rounds). ``max_chance_children`` truncates the
    sampled-successor list per chance node.
    """
    print()
    print("=" * 78)
    print(f"MCTS TREE (max_depth={max_depth})")
    print("=" * 78)
    _walk_decision(root, depth=0, max_depth=max_depth, max_chance_children=max_chance_children, prefix="")
    print("=" * 78)


def _walk_decision(node, *, depth, max_depth, max_chance_children, prefix):
    header = f"[D] {_state_summary(node.state)}  N={node.visits}{_terminal_mark(node)}"
    print(f"{prefix}{header}")
    if depth >= max_depth or node.terminal:
        if node.untried_actions and not node.terminal:
            untried = [a.verb.value for a in node.untried_actions]
            print(f"{prefix}    (untried: {', '.join(untried)})")
        return
    if node.untried_actions:
        untried = [a.verb.value for a in node.untried_actions]
        print(f"{prefix}    (untried: {', '.join(untried)})")

    children_items = sorted(node.children.items(), key=lambda kv: -kv[1].visits)
    for i, (action, chance) in enumerate(children_items):
        is_last_chance = i == len(children_items) - 1
        chance_connector = "└─" if is_last_chance else "├─"
        chance_indent = "   " if is_last_chance else "│  "

        q = chance.mean_value
        line = (
            f"{action.verb.value}  N={chance.visits}  Q̂={q:+.4f}  "
            f"(prior={chance.prior:.3f}, {len(chance.children)} sampled successor"
            f"{'s' if len(chance.children) != 1 else ''})"
        )
        print(f"{prefix}{chance_connector} [C] {line}")

        successors = chance.children[:max_chance_children]
        truncated = len(chance.children) - len(successors)
        for j, child in enumerate(successors):
            is_last_d = (j == len(successors) - 1) and truncated == 0
            d_connector = "└─" if is_last_d else "├─"
            child_prefix = prefix + chance_indent + ("   " if is_last_d else "│  ")
            print(f"{prefix}{chance_indent}{d_connector}", end="")
            _walk_decision(
                child,
                depth=depth + 1,
                max_depth=max_depth,
                max_chance_children=max_chance_children,
                prefix=child_prefix,
            )
        if truncated:
            print(f"{prefix}{chance_indent}   (+ {truncated} more)")


# -----------------------------------------------------------------------------
# Summarize the root node after the search
# -----------------------------------------------------------------------------


def summarize_root(root: DecisionNode) -> dict[str, Any]:
    """Return a human-friendly summary of root visit counts / Q estimates."""
    visit_counts = {}
    q_estimates = {}
    chance_samples = {}
    for action, chance in root.children.items():
        verb = action.verb.value
        visit_counts[verb] = chance.visits
        q_estimates[verb] = (
            round(chance.total_value / chance.visits, 4) if chance.visits else None
        )
        chance_samples[verb] = len(chance.children)

    pi = visit_count_policy(root, temperature=1.0)
    pi_by_verb = {action.verb.value: round(p, 4) for action, p in pi.items()}

    q_root = root_q_estimates(root)
    q_root_by_verb = {action.verb.value: round(q, 4) for action, q in q_root.items()}

    if root.children:
        recommended = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
        rec_verb = recommended.verb.value
    else:
        rec_verb = None

    return {
        "root_visits": root.visits,
        "untried_actions": [a.verb.value for a in root.untried_actions],
        "expanded_actions": list(visit_counts.keys()),
        "visit_counts": visit_counts,
        "chance_node_children_per_action": chance_samples,
        "mean_value_per_action": q_estimates,
        "pi_mcts": pi_by_verb,
        "q_estimate_root_q_estimates": q_root_by_verb,
        "recommended_action": rec_verb,
    }


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://192.168.0.204:11434/v1")
    p.add_argument("--model", default="qwen2.5:3b")
    p.add_argument(
        "--problem",
        choices=list(PROBLEMS.keys()),
        default="multiply_small",
    )
    p.add_argument("--num-simulations", type=int, default=10)
    p.add_argument("--c-puct", type=float, default=1.0)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--rollout-depth", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--depth-limit", type=int, default=3)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--summary-json", default=None)
    args = p.parse_args(argv)

    print(f"\n========== MCTS root search on '{args.problem}' ==========", flush=True)
    print(f"model: {args.model}", flush=True)
    print(f"B = {args.num_simulations} simulations", flush=True)
    print(f"c_puct = {args.c_puct}, discount = {args.discount}", flush=True)
    print(f"rollout_depth_limit = {args.rollout_depth}", flush=True)
    print(f"LLM temperature = {args.temperature}, max_tokens = {args.max_tokens}", flush=True)
    print(f"prior: uniform (no learned pi_theta)", flush=True)
    print()

    raw_client = OpenAICompatibleLLMClient(
        base_url=args.base_url,
        model=args.model,
        timeout_s=args.timeout,
        max_retries=1,
    )
    client = TimingLLMClient(inner=raw_client)

    problem_def = PROBLEMS[args.problem]
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
    state = mdp.initial_state()

    print("initial state available actions:", flush=True)
    for v in available_actions(state, depth_limit=mdp.depth_limit):
        print(f"  - {v.value}", flush=True)
    print()

    config = MCTSConfig[type(state), Any](
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        discount=args.discount,
        rollout_depth_limit=args.rollout_depth,
        prior_fn=None,
        rollout_policy=None,
    )

    rng = np.random.default_rng(args.seed)

    started = time.time()
    root = mcts_search(mdp, state, config, rng)
    duration = time.time() - started

    print_tree(root, max_depth=8, max_chance_children=4)

    summary = summarize_root(root)
    summary["wall_clock_total_s"] = round(duration, 2)
    summary["llm_call_count"] = len(client.calls)
    durations = [c["duration_s"] for c in client.calls if c["ok"]]
    summary["llm_call_durations_s"] = durations
    if durations:
        summary["llm_call_mean_s"] = round(statistics.mean(durations), 3)
        summary["llm_call_median_s"] = round(statistics.median(durations), 3)
        summary["llm_call_min_s"] = round(min(durations), 3)
        summary["llm_call_max_s"] = round(max(durations), 3)
        summary["llm_call_stdev_s"] = (
            round(statistics.stdev(durations), 3) if len(durations) > 1 else 0.0
        )
        summary["llm_total_s"] = round(sum(durations), 2)
        summary["llm_share_of_wallclock"] = (
            round(sum(durations) / duration, 3) if duration > 0 else None
        )
    summary["failures"] = sum(1 for c in client.calls if not c["ok"])
    summary["calls_by_verb"] = {
        v.value: sum(1 for c in client.calls if c.get("verb_hint") == v.value)
        for v in Verb
    }

    print("\n--- summary ---", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {args.summary_json}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
