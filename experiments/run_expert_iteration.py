"""Smoke-test script: train a tabular policy on the example gridworld.

Reproduces the simplest expert-iteration loop from design.md (section 9)
on the canonical 3 by 4 gridworld layout from design.md (section 5.1):

1. Build the gridworld and a fresh ``TabularPolicy`` (uniform prior).
2. Run 10 iterations of expert iteration: each iteration collects 20
   episodes of MCTS-guided trajectories with 50 MCTS sims per step and
   then takes one cross-entropy gradient step per recorded example.
3. After each iteration, print the iteration number, mean episode
   return, and mean episode length.
4. After training, evaluate the policy in fast mode (greedy argmax, no
   search at inference) and in search mode (MCTS with B = 50 sims at
   action selection). Print both.
5. Compare with the exact V*(start) computed by value iteration.

Run from the repository root::

    python experiments/run_expert_iteration.py
"""

from __future__ import annotations

import argparse

import numpy as np

from verifier_anchored_rl.eval import evaluate_policy, evaluate_with_search
from verifier_anchored_rl.expert_iteration import (
    ExpertIterationConfig,
    IterationResult,
    run_expert_iteration,
)
from verifier_anchored_rl.gridworld import Gridworld
from verifier_anchored_rl.policies import TabularPolicy
from verifier_anchored_rl.value_iteration import value_iteration


def main(argv: list[str] | None = None) -> dict[str, float | int]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--evaluation-max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    if any(value <= 0 for name, value in vars(args).items() if name != "seed"):
        parser.error("all training, search, and evaluation budgets must be positive")
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

    # ---- Value-iteration baseline ------------------------------------
    V_vi, _ = value_iteration(env, gamma=gamma, theta=1e-6, max_iters=10_000)
    print(f"V*(start) (from value iteration) = {V_vi[env.start]:+.4f}")
    print()

    # ---- Expert iteration training -----------------------------------
    policy: TabularPolicy = TabularPolicy()
    config = ExpertIterationConfig(
        num_iterations=args.iterations,
        episodes_per_iteration=args.episodes,
        max_episode_steps=args.max_steps,
        mcts_simulations_per_step=args.simulations,
        mcts_c_puct=1.0,
        mcts_temperature=1.0,
        discount=gamma,
        learning_rate=0.2,
        train_steps_per_iteration=1,
    )

    print(
        f"Running {config.num_iterations} iterations of "
        f"{config.episodes_per_iteration} episodes each, "
        f"{config.mcts_simulations_per_step} MCTS sims per step..."
    )
    print(
        f"{'iter':>4} {'mean_return':>12} {'mean_length':>12} {'num_examples':>14}"
    )

    def report(r: IterationResult) -> None:
        print(
            f"{r.iteration:>4d} {r.mean_episode_return:>12.4f} "
            f"{r.mean_episode_length:>12.2f} {r.num_examples:>14d}"
        )

    rng = np.random.default_rng(args.seed)
    rounds = run_expert_iteration(
        env=env,
        initial_state_fn=lambda r: env.start,
        policy=policy,
        config=config,
        rng=rng,
        on_iteration=report,
    )
    print()

    # ---- Post-training evaluation ------------------------------------
    eval_episodes = args.evaluation_episodes
    eval_max_steps = args.evaluation_max_steps

    fast_rng = np.random.default_rng(11)
    fast = evaluate_policy(
        env=env,
        initial_state_fn=lambda r: env.start,
        policy=policy,
        discount=gamma,
        num_episodes=eval_episodes,
        max_steps=eval_max_steps,
        rng=fast_rng,
    )
    print(
        f"Fast mode (B = 0): mean_return = {fast['mean_return']:+.4f} "
        f"(std = {fast['std_return']:.4f}), "
        f"mean_episode_length = {fast['mean_episode_length']:.2f}"
    )

    search_rng = np.random.default_rng(11)
    search = evaluate_with_search(
        env=env,
        initial_state_fn=lambda r: env.start,
        policy=policy,
        mcts_simulations=args.simulations,
        discount=gamma,
        num_episodes=eval_episodes,
        max_steps=eval_max_steps,
        rng=search_rng,
        c_puct=config.mcts_c_puct,
    )
    print(
        f"Search mode (B = {args.simulations}): mean_return = {search['mean_return']:+.4f} "
        f"(std = {search['std_return']:.4f}), "
        f"mean_episode_length = {search['mean_episode_length']:.2f}"
    )
    print()

    gap_fast = V_vi[env.start] - fast["mean_return"]
    gap_search = V_vi[env.start] - search["mean_return"]
    print(
        f"Gap to V*(start):  fast = {gap_fast:+.4f},  search = {gap_search:+.4f}"
    )
    print(
        "(Smaller gaps are better; fast mode reflects the parametric policy "
        "alone, search mode reflects policy + MCTS together.)"
    )
    return {
        "iterations": len(rounds),
        "training_examples": sum(r.num_examples for r in rounds),
        "optimal_value": V_vi[env.start],
        "fast_return": fast["mean_return"],
        "search_return": search["mean_return"],
    }


if __name__ == "__main__":
    main()
