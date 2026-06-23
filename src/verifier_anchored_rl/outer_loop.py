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

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .mcts import MCTSConfig, mcts_search, root_q_estimates, visit_count_policy
from .replay import sample_from_record


def _entropy(probs: Any) -> float:
    """Shannon entropy in bits of a probability vector. Zero-probability
    entries are skipped (0 log 0 = 0). A delta is 0 bits; a uniform over k is
    log2(k) bits. Used to quantify how sharp the MCTS visit distribution and
    the learned prior are (lower entropy = more concentrated = sharper)."""
    return -sum(p * math.log2(p) for p in probs if p > 0.0)


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
    visit_entropy: float = float("nan")
    prior_entropy: float = float("nan")
    visit_concentration: float = float("nan")
    prior_concentration: float = float("nan")


def _flop_counters(encoder: Any, gen_source: Any = None) -> dict[str, int]:
    """Read prefill counters from the encoder and generation counters from
    ``gen_source`` (zeros if absent). When generation runs on a separate
    backend from the embedding encoder (the constrained chat path generates via
    the llm_client while the encoder only embeds), pass that backend as
    ``gen_source``; it exposes ``generate_calls``/``generated_tokens`` (the
    llm_client) or ``gen_calls``/``gen_tokens`` (the encoder). Defaults to the
    encoder, so the one-forward path is unchanged."""
    gen = gen_source if gen_source is not None else encoder
    return {
        "prefill_calls": int(getattr(encoder, "prefill_calls", 0)),
        "prefill_tokens": int(getattr(encoder, "prefill_tokens", 0)),
        "gen_calls": int(getattr(gen, "generate_calls", getattr(gen, "gen_calls", 0))),
        "gen_tokens": int(getattr(gen, "generated_tokens", getattr(gen, "gen_tokens", 0))),
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
    gen_source: Any = None,
) -> dict[str, float]:
    """Drive one trajectory per problem, adding one ReplaySample per node.

    See module docstring. ``start_problem`` skips the first k problems (for
    resume); ``on_decision`` fires per decision with a labeled info dict;
    ``on_trajectory_end(done)`` fires after each completed trajectory with the
    1-based count of trajectories finished this iteration.
    """
    from .llm_context_mdp import available_actions

    n_traj = 0
    n_dec = 0
    visit_entropy_sum = 0.0
    prior_entropy_sum = 0.0
    visit_conc_sum = 0.0
    prior_conc_sum = 0.0
    conc_count = 0
    enc = getattr(state_actor, "encoder", None)
    # Generation runs on gen_source (the llm_client) on the two-forward path;
    # fall back to the encoder for the one-forward path. Read prefill counts
    # from the encoder, gen counts from gen_source (matching _flop_counters).
    gen_src = gen_source if gen_source is not None else enc
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

            # Sharpening diagnostics (method.tex 3.7 success criterion: the
            # learned prior sharpens so search visits concentrate). The prior
            # read reuses the cached state prefill, so it adds only a tiny heads
            # pass and does not touch the prefill/gen FLOP counters.
            prior_probs = state_actor.prior(state, actions)
            visit_entropy = _entropy(pi.values())
            prior_entropy = _entropy(prior_probs)
            visit_entropy_sum += visit_entropy
            prior_entropy_sum += prior_entropy
            # Normalized concentration 1 - H/log2(k) (confound-free across the
            # variable available-verb count k); higher = sharper. Undefined for
            # k=1, so those decisions are excluded from the mean.
            k = len(actions)
            if k > 1:
                lk = math.log2(k)
                visit_conc_sum += 1.0 - visit_entropy / lk
                prior_conc_sum += 1.0 - prior_entropy / lk
                conc_count += 1

            if on_decision is not None:
                top = max(pi.items(), key=lambda kv: kv[1])
                on_decision({
                    "iteration": iteration_index,
                    "iteration_problem_index": p_idx,
                    "turn": int(state.turn),
                    "chosen_verb": chosen.verb.value,
                    "pi_argmax_verb": top[0].verb.value,
                    "pi_argmax_prob": float(top[1]),
                    "pi_mcts_entropy": float(visit_entropy),
                    "prior_entropy": float(prior_entropy),
                    "q_chosen": float(q.get(chosen, float("nan"))),
                    "prefill_calls": int(getattr(enc, "prefill_calls", 0)),
                    "prefill_tokens": int(getattr(enc, "prefill_tokens", 0)),
                    "gen_calls": int(getattr(gen_src, "generate_calls",
                                             getattr(gen_src, "gen_calls", 0))),
                    "gen_tokens": int(getattr(gen_src, "generated_tokens",
                                              getattr(gen_src, "gen_tokens", 0))),
                })

            state, _reward = mdp.sample_transition(state, chosen, rng)

        if on_trajectory_end is not None:
            on_trajectory_end(p_idx + 1)

    nan = float("nan")
    return {
        "n_trajectories": n_traj,
        "n_decisions": n_dec,
        "n_simulations": n_dec * config.num_simulations,
        "mean_visit_entropy": visit_entropy_sum / n_dec if n_dec else nan,
        "mean_prior_entropy": prior_entropy_sum / n_dec if n_dec else nan,
        "mean_visit_concentration": visit_conc_sum / conc_count if conc_count else nan,
        "mean_prior_concentration": prior_conc_sum / conc_count if conc_count else nan,
    }


def collect_iteration_concurrent(
    problems, mdp_factory, state_actor, config, buffer, *,
    base_seed, tau, max_turns, max_workers=8,
    record_sink=None, on_decision=None, gen_source=None, iteration_index=0,
):
    """Run one trajectory per problem CONCURRENTLY (thread pool). Each worker
    drives a normal sequential MCTS trajectory with its OWN rng (sub-seeded from
    base_seed plus the problem index) and returns its ReplaySamples; the shared
    llm_client (vLLM) batches the concurrent generation requests. The buffer is
    populated single-threaded after the workers finish, so it is thread-safe and
    deterministic regardless of max_workers."""
    import concurrent.futures as cf
    import threading

    import numpy as np

    from .llm_context_mdp import available_actions
    from .mcts import mcts_search, root_q_estimates, visit_count_policy

    log_lock = threading.Lock()

    def run_one(p_idx, problem):
        rng = np.random.default_rng(base_seed + 1 + p_idx)
        mdp = mdp_factory(problem)
        state = mdp.initial_state()
        records = []
        while not mdp.is_terminal(state) and state.turn < max_turns:
            avail = list(available_actions(state, depth_limit=mdp.depth_limit))
            if not avail:
                break
            root = mcts_search(mdp, state, config, rng)
            if not root.children:
                break
            pi = visit_count_policy(root, temperature=tau)
            q = root_q_estimates(root)
            rec = {
                "embedding": state_actor.embed(state).detach().cpu().tolist(),
                "available_verbs": [a.value for a in avail],
                "pi_mcts": {a.verb.value: float(p) for a, p in pi.items()},
                "q_mcts": {a.verb.value: float(v) for a, v in q.items()},
            }
            records.append(rec)
            actions = list(pi.keys())
            probs = np.array([pi[a] for a in actions], dtype=float)
            probs = probs / probs.sum()
            chosen = actions[int(rng.choice(len(actions), p=probs))]
            if on_decision is not None:
                top = max(pi.items(), key=lambda kv: kv[1])
                with log_lock:
                    on_decision({
                        "iteration": iteration_index,
                        "iteration_problem_index": p_idx,
                        "turn": int(state.turn),
                        "chosen_verb": chosen.verb.value,
                        "pi_argmax_verb": top[0].verb.value,
                        "pi_argmax_prob": float(top[1]),
                        "gen_calls": int(getattr(gen_source, "generate_calls",
                                                 getattr(gen_source, "gen_calls", 0))),
                        "gen_tokens": int(getattr(gen_source, "generated_tokens",
                                                  getattr(gen_source, "gen_tokens", 0))),
                    })
            state, _r = mdp.sample_transition(state, chosen, rng)
        return records

    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(lambda ip: run_one(*ip), list(enumerate(problems))))

    n_dec = 0
    for recs in results:
        for rec in recs:
            buffer.add(sample_from_record(rec))
            if record_sink is not None:
                record_sink(rec)
            n_dec += 1
    return {
        "n_trajectories": len(problems),
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
    gen_source: Any = None,
    concurrent: bool = False,
    collect_workers: int = 8,
    base_seed: int = 0,
    record_sink: Callable[[dict[str, Any]], None] | None = None,
    on_decision: Callable[[dict[str, Any]], None] | None = None,
    on_train_step: Callable[[dict[str, Any]], None] | None = None,
    on_iteration_end: Callable[["IterationStats"], None] | None = None,
    save_checkpoint: Callable[[dict[str, Any]], None] | None = None,
    log_every: int = 25,
    start_iteration: int = 1,
    start_problem: int = 0,
) -> list[IterationStats]:
    """Run Algorithm 1 for ``n_iterations``: collect, then train M steps.

    Observability and resume hooks are all optional and default-off, so the
    original behavior is unchanged when none are passed. ``start_iteration`` /
    ``start_problem`` resume a partially completed run (1-based iteration index;
    ``start_problem`` only applies to ``start_iteration``).

    When ``concurrent=True``, collection uses ``collect_iteration_concurrent``
    (a thread pool sized by ``collect_workers``, so a shared vLLM client batches
    the trajectories' generation requests); resume granularity is per-iteration
    (no mid-collection trajectory checkpoint) and the entropy/concentration
    diagnostics are deferred to offline analysis of records.jsonl, so they come
    back NaN on the iteration stats.
    """
    stats: list[IterationStats] = []
    for it in range(start_iteration, n_iterations + 1):
        t0 = time.monotonic()
        config = MCTSConfig(
            num_simulations=num_simulations,
            c_puct=c_puct,
            discount=discount,
            prior_fn=state_actor.prior,
            value_fn=state_actor.leaf_value,
        )
        before = _flop_counters(encoder, gen_source)
        state_actor.cache.clear()

        def _on_trajectory_end(done: int) -> None:
            if save_checkpoint is not None:
                save_checkpoint({
                    "phase": "collecting",
                    "iteration": it,
                    "problems_done_this_iter": done,
                })

        if concurrent:
            # vLLM path: many trajectories run at once so the shared client
            # batches their generation. Resume granularity is per-iteration
            # (no mid-collection trajectory checkpoint); the per-iteration
            # "training" checkpoint below still fires. Sharpening diagnostics
            # are deferred to offline analysis of records.jsonl (pi_mcts per
            # decision), so they come back NaN here.
            counts = collect_iteration_concurrent(
                problems, mdp_factory, state_actor, config, buffer,
                base_seed=base_seed, tau=tau, max_turns=max_turns,
                max_workers=collect_workers, record_sink=record_sink,
                on_decision=on_decision, gen_source=gen_source,
                iteration_index=it,
            )
        else:
            counts = collect_iteration(
                problems, mdp_factory, state_actor, config, buffer, rng,
                tau=tau, max_turns=max_turns, record_sink=record_sink,
                on_decision=on_decision, on_trajectory_end=_on_trajectory_end,
                start_problem=(start_problem if it == start_iteration else 0),
                iteration_index=it, gen_source=gen_source,
            )
        after = _flop_counters(encoder, gen_source)

        losses: list[dict[str, float]] = []
        if len(buffer) >= min_buffer_size:
            def _on_step(global_step: int, step_losses: dict[str, float]) -> None:
                if on_train_step is not None and (
                    global_step % log_every == 0 or global_step == 1
                ):
                    on_train_step({"global_step": global_step, **step_losses})
            losses = trainer.run(
                train_steps_per_iter, batch_size, rng, on_step=_on_step
            )

        if save_checkpoint is not None:
            save_checkpoint({
                "phase": "training",
                "iteration": it,
                "problems_done_this_iter": len(problems),
            })

        summary = loss_summary(losses)
        s = IterationStats(
            iteration=it,
            losses=losses,
            prefill_calls=after["prefill_calls"] - before["prefill_calls"],
            prefill_tokens=after["prefill_tokens"] - before["prefill_tokens"],
            gen_calls=after["gen_calls"] - before["gen_calls"],
            gen_tokens=after["gen_tokens"] - before["gen_tokens"],
            n_trajectories=int(counts["n_trajectories"]),
            n_decisions=int(counts["n_decisions"]),
            n_simulations=int(counts["n_simulations"]),
            buffer_size=len(buffer),
            wall_seconds=time.monotonic() - t0,
            q_first=summary["q_first"], q_last=summary["q_last"],
            pi_first=summary["pi_first"], pi_last=summary["pi_last"],
            total_first=summary["total_first"], total_last=summary["total_last"],
            visit_entropy=counts.get("mean_visit_entropy", float("nan")),
            prior_entropy=counts.get("mean_prior_entropy", float("nan")),
            visit_concentration=counts.get("mean_visit_concentration", float("nan")),
            prior_concentration=counts.get("mean_prior_concentration", float("nan")),
        )
        stats.append(s)
        if on_iteration_end is not None:
            on_iteration_end(s)
    return stats


__all__ = [
    "IterationStats",
    "collect_iteration",
    "collect_iteration_concurrent",
    "loss_summary",
    "run_outer_loop",
]
