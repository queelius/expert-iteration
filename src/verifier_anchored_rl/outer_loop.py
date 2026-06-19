"""The outer collection-and-train loop (method.tex 3.7, Algorithm 1).

Approximate Q-learning with MCTS-supplied targets, alternated with policy
distillation. MCTS is the policy-improvement operator: each iteration it
turns the current heads (online policy as prior, target-Q as bootstrap leaf
value) into stronger targets pi_MCTS and Q_MCTS, which the Trainer distills
back into the heads, which then drive better search next iteration.

This module is the orchestration only. The loss/optimizer step, priority
update, and target sync live in train.Trainer; the prior/leaf reads live in
planner_actor.StateActor; the leaf hook lives in mcts. The loop is CPU-
testable end to end with a FakeEncoder; the real run uses SharedForwardEncoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .mcts import MCTSConfig, mcts_search, root_q_estimates, visit_count_policy
from .replay import sample_from_record


def loss_summary(losses: list[dict[str, float]]) -> dict[str, float]:
    """First, last, and mean of each loss component over a list of per-step
    loss dicts. All NaN if the list is empty (training did not run)."""
    nan = float("nan")
    if not losses:
        return {
            "q_first": nan, "q_last": nan, "q_mean": nan,
            "pi_first": nan, "pi_last": nan, "pi_mean": nan,
            "total_first": nan, "total_last": nan, "total_mean": nan,
        }

    def col(key):
        return [d[key] for d in losses]

    out = {}
    for key in ("q", "pi", "total"):
        vals = col(key)
        out[f"{key}_first"] = vals[0]
        out[f"{key}_last"] = vals[-1]
        out[f"{key}_mean"] = sum(vals) / len(vals)
    return out


@dataclass
class IterationStats:
    """Per-iteration record: the loss curve, FLOP counters, and counts."""

    iteration: int
    losses: list[dict[str, float]] = field(default_factory=list)
    prefill_calls: int = 0
    prefill_tokens: int = 0
    gen_calls: int = 0
    gen_tokens: int = 0
    n_trajectories: int = 0
    n_decisions: int = 0
    n_simulations: int = 0
    buffer_size: int = 0
    wall_seconds: float = 0.0
    q_first: float = float("nan")
    q_last: float = float("nan")
    pi_first: float = float("nan")
    pi_last: float = float("nan")
    total_first: float = float("nan")
    total_last: float = float("nan")


def _flop_counters(encoder: Any) -> dict[str, int]:
    """Read the encoder's token/call counters (zeros if absent)."""
    return {
        "prefill_calls": int(getattr(encoder, "prefill_calls", 0)),
        "prefill_tokens": int(getattr(encoder, "prefill_tokens", 0)),
        "gen_calls": int(getattr(encoder, "gen_calls", 0)),
        "gen_tokens": int(getattr(encoder, "gen_tokens", 0)),
    }


def collect_iteration(
    problems: list[Any],
    mdp_factory: Callable[[Any], Any],
    state_actor: Any,
    config: MCTSConfig,
    buffer: Any,
    rng: np.random.Generator,
    *,
    tau: float,
    max_turns: int,
    record_sink: Callable[[dict[str, Any]], None] | None = None,
    on_decision: Callable[[dict[str, Any]], None] | None = None,
    on_trajectory_end: Callable[[int], None] | None = None,
    start_problem: int = 0,
    iteration_index: int = 0,
) -> dict[str, int]:
    """Drive one trajectory per problem, adding one ReplaySample per node.

    See module docstring. ``start_problem`` skips the first k problems (for
    resume); ``on_decision`` fires per decision with a labeled info dict;
    ``on_trajectory_end(done)`` fires after each completed trajectory with the
    1-based count of trajectories finished this iteration.
    """
    from .llm_context_mdp import available_actions

    n_traj = 0
    n_dec = 0
    enc = getattr(state_actor, "encoder", None)
    for p_idx, problem in enumerate(problems):
        if p_idx < start_problem:
            continue
        mdp = mdp_factory(problem)
        state = mdp.initial_state()
        n_traj += 1
        while not mdp.is_terminal(state) and state.turn < max_turns:
            avail = list(available_actions(state, depth_limit=mdp.depth_limit))
            if not avail:
                break
            root = mcts_search(mdp, state, config, rng)
            if not root.children:
                break
            pi = visit_count_policy(root, temperature=tau)
            q = root_q_estimates(root)
            record = {
                "embedding": state_actor.embed(state).detach().cpu().tolist(),
                "available_verbs": [a.value for a in avail],
                "pi_mcts": {a.verb.value: float(p) for a, p in pi.items()},
                "q_mcts": {a.verb.value: float(v) for a, v in q.items()},
            }
            buffer.add(sample_from_record(record))
            if record_sink is not None:
                record_sink(record)
            n_dec += 1

            actions = list(pi.keys())
            probs = np.array([pi[a] for a in actions], dtype=float)
            probs = probs / probs.sum()
            chosen = actions[int(rng.choice(len(actions), p=probs))]

            if on_decision is not None:
                top = max(pi.items(), key=lambda kv: kv[1])
                on_decision({
                    "iteration": iteration_index,
                    "iteration_problem_index": p_idx,
                    "turn": int(state.turn),
                    "chosen_verb": chosen.verb.value,
                    "pi_argmax_verb": top[0].verb.value,
                    "pi_argmax_prob": float(top[1]),
                    "q_chosen": float(q.get(chosen, float("nan"))),
                    "prefill_calls": int(getattr(enc, "prefill_calls", 0)),
                    "prefill_tokens": int(getattr(enc, "prefill_tokens", 0)),
                    "gen_calls": int(getattr(enc, "gen_calls", 0)),
                    "gen_tokens": int(getattr(enc, "gen_tokens", 0)),
                })

            state, _reward = mdp.sample_transition(state, chosen, rng)

        if on_trajectory_end is not None:
            on_trajectory_end(p_idx + 1)

    return {
        "n_trajectories": n_traj,
        "n_decisions": n_dec,
        "n_simulations": n_dec * config.num_simulations,
    }


def run_outer_loop(
    problems: list[Any],
    mdp_factory: Callable[[Any], Any],
    state_actor: Any,
    trainer: Any,
    buffer: Any,
    *,
    n_iterations: int,
    train_steps_per_iter: int,
    batch_size: int,
    min_buffer_size: int,
    num_simulations: int,
    c_puct: float,
    discount: float,
    tau: float,
    max_turns: int,
    rng: np.random.Generator,
    encoder: Any = None,
    record_sink: Callable[[dict[str, Any]], None] | None = None,
) -> list[IterationStats]:
    """Run Algorithm 1 for ``n_iterations``: collect, then train M steps.

    The MCTS prior is the online policy head and the leaf value is the
    target-Q head, both via ``state_actor``. Training is gated behind a
    ``min_buffer_size`` warmup. Target sync fires inside ``trainer`` on its
    own N-step schedule.
    """
    stats: list[IterationStats] = []
    for it in range(1, n_iterations + 1):
        config = MCTSConfig(
            num_simulations=num_simulations,
            c_puct=c_puct,
            discount=discount,
            prior_fn=state_actor.prior,
            value_fn=state_actor.leaf_value,
        )
        before = _flop_counters(encoder)
        # Clear the prefill cache so each iteration counts its own encoder
        # calls from scratch. The cache is an intra-iteration optimization
        # (one forward per decision); cross-iteration reuse would make the
        # FLOP counter deltas meaningless.
        state_actor.cache.clear()
        counts = collect_iteration(
            problems, mdp_factory, state_actor, config, buffer, rng,
            tau=tau, max_turns=max_turns, record_sink=record_sink,
        )
        after = _flop_counters(encoder)

        losses: list[dict[str, float]] = []
        if len(buffer) >= min_buffer_size:
            losses = trainer.run(train_steps_per_iter, batch_size, rng)

        stats.append(IterationStats(
            iteration=it,
            losses=losses,
            prefill_calls=after["prefill_calls"] - before["prefill_calls"],
            prefill_tokens=after["prefill_tokens"] - before["prefill_tokens"],
            gen_calls=after["gen_calls"] - before["gen_calls"],
            gen_tokens=after["gen_tokens"] - before["gen_tokens"],
            n_trajectories=counts["n_trajectories"],
            n_decisions=counts["n_decisions"],
            n_simulations=counts["n_simulations"],
            buffer_size=len(buffer),
        ))
    return stats


__all__ = ["IterationStats", "collect_iteration", "loss_summary", "run_outer_loop"]
