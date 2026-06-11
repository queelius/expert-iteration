"""Smoke-test script: run MCTS from the example gridworld start state.

Builds the layout from design.md (section 5.1), runs MCTS with 2000
simulations from the start state, then prints:

* the chance-node Q estimates Qhat_MCTS(start, a)
* the visit-count policy pi_MCTS(a | start)
* the exact V*(start) and pi*(start) from value iteration
* a side-by-side comparison

Run from the repository root::

    python experiments/run_mcts.py
"""

from __future__ import annotations

import numpy as np

from verifier_anchored_rl.gridworld import GridAction, Gridworld
from verifier_anchored_rl.mcts import (
    MCTSConfig,
    best_action,
    mcts_search,
    root_q_estimates,
    visit_count_policy,
)
from verifier_anchored_rl.value_iteration import value_iteration


_ARROW = {
    GridAction.UP: "^",
    GridAction.DOWN: "v",
    GridAction.LEFT: "<",
    GridAction.RIGHT: ">",
}


def _format_action_table(
    title: str, table: dict[GridAction, float], fmt: str = "{:+.4f}"
) -> str:
    lines = [title]
    for a in GridAction:
        val = table.get(a)
        if val is None:
            cell = "  (not expanded)"
        else:
            cell = fmt.format(val)
        lines.append(f"  {a.name:<5} ({_ARROW[a]})  {cell}")
    return "\n".join(lines)


def main() -> None:
    layout = [
        "S..G",
        ".#.P",
        "....",
    ]
    env = Gridworld(layout, step_reward=-0.01, goal_reward=1.0, pit_reward=-1.0)
    gamma = 0.99

    print("Layout:")
    print(env.render_layout())
    print()

    # ---- value-iteration baseline -------------------------------------
    V_vi, policy_vi = value_iteration(env, gamma=gamma, theta=1e-6, max_iters=10_000)
    start = env.start
    vi_best = policy_vi[start]
    print(f"V*(start={start}) = {V_vi[start]:+.4f}")
    print(f"pi*(start) = {vi_best.name} ({_ARROW[vi_best]})")
    print()

    # ---- MCTS ---------------------------------------------------------
    config = MCTSConfig(
        num_simulations=2_000,
        c_puct=1.0,
        discount=gamma,
        rollout_depth_limit=50,
    )
    rng = np.random.default_rng(2026)
    root = mcts_search(env, start, config, rng)

    q = root_q_estimates(root)
    pi = visit_count_policy(root, temperature=1.0)
    mcts_best = best_action(root)

    print(f"MCTS run with {config.num_simulations} simulations, "
          f"c_puct = {config.c_puct}, gamma = {config.discount}")
    print()
    print(_format_action_table("Qhat_MCTS(start, a):", q, fmt="{:+.4f}"))
    print()
    print(_format_action_table("pi_MCTS(a | start) (visit-count distribution):", pi, fmt="{:.4f}"))
    print()

    # ---- side-by-side comparison --------------------------------------
    agreement = "agree" if mcts_best == vi_best else "disagree"
    print(
        f"MCTS recommends {mcts_best.name} ({_ARROW[mcts_best]}); "
        f"VI says {vi_best.name} ({_ARROW[vi_best]}). They {agreement}."
    )
    if mcts_best in q:
        gap = V_vi[start] - q[mcts_best]
        print(
            f"Value gap: V*(start) - Qhat_MCTS(start, {mcts_best.name}) "
            f"= {gap:+.4f} (sampled estimator; small gaps are expected)."
        )


if __name__ == "__main__":
    main()
