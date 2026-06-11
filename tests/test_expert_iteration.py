"""Tests for the expert-iteration training loop and the eval utilities.

These tests are deliberately small to keep the suite fast (under ten
seconds total). The SAGB convergence test uses the hand-computable
MDP from ``tests.test_value_iteration`` so the success criterion is
unambiguous: after training, the policy should put more probability mass
on "risky" than on "safe" at the start state, because Q*(S, risky) = 6
exceeds Q*(S, safe) = 4 under the design.md parameters.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from verifier_anchored_rl.eval import evaluate_policy, evaluate_with_search
from verifier_anchored_rl.expert_iteration import (
    ExpertIterationConfig,
    IterationResult,
    PolicyTarget,
    run_expert_iteration,
)
from verifier_anchored_rl.gridworld import Gridworld
from verifier_anchored_rl.policies import TabularPolicy

from tests.test_mcts import SAGBGenerative
from tests.test_value_iteration import _S


def test_expert_iteration_runs_on_gridworld() -> None:
    """Smoke test: a tiny gridworld run produces well-formed results.

    Three iterations, five episodes each, twenty MCTS sims per step.
    The loop should complete, the result list should have one entry per
    iteration, and the mean episode return should be a finite float on
    every iteration.
    """
    layout = [
        "S.G",
        "...",
        "..P",
    ]
    env = Gridworld(layout, step_reward=-0.01, goal_reward=1.0, pit_reward=-1.0)
    policy: TabularPolicy = TabularPolicy()
    config = ExpertIterationConfig(
        num_iterations=3,
        episodes_per_iteration=5,
        max_episode_steps=20,
        mcts_simulations_per_step=20,
        discount=0.99,
        learning_rate=0.2,
    )
    rng = np.random.default_rng(0)
    results = run_expert_iteration(
        env=env,
        initial_state_fn=lambda r: env.start,
        policy=policy,
        config=config,
        rng=rng,
    )
    assert len(results) == 3
    for i, result in enumerate(results):
        assert isinstance(result, IterationResult)
        assert result.iteration == i
        assert result.num_examples > 0
        assert math.isfinite(result.mean_episode_return)
        assert math.isfinite(result.mean_episode_length)


def test_expert_iteration_improves_policy_on_sagb() -> None:
    """After enough expert-iteration rounds, the policy prefers "risky" at S.

    For the SAGB MDP with gamma = 1 the optimal action at S is "risky"
    (Q*(S, risky) = 6, Q*(S, safe) = 4). MCTS recovers the right Q
    ordering at the root, so the visit-count target distribution puts
    most of its mass on "risky" already in the very first iteration.
    Expert iteration distills that signal into the policy logits, so
    after a few rounds the policy itself should prefer "risky" by a
    clear margin (probability strictly greater than 0.5 at S).

    Convergence criterion: P_theta(risky | S) > 0.5 after training.
    """
    env = SAGBGenerative()
    policy: TabularPolicy = TabularPolicy()
    # c_puct = 8.0 is comparatively high; this is the right scale for the
    # SAGB MDP where Q values are on the order of 4 to 10 and the per-
    # iteration MCTS budget is small. With a smaller exploration constant
    # the early visit-count targets are very noisy (because Q(risky) has
    # high rollout variance) and a few unlucky targets can lock the
    # policy into preferring "safe" before the search has resolved the
    # right ordering. The value of c_puct does not need to be tuned for
    # the gridworld: there the per-step rewards live on a different
    # scale and a smaller c_puct of 1.0 to 2.0 is appropriate.
    config = ExpertIterationConfig(
        num_iterations=8,
        episodes_per_iteration=8,
        max_episode_steps=10,
        mcts_simulations_per_step=128,
        mcts_c_puct=8.0,
        mcts_temperature=1.0,
        discount=1.0,
        learning_rate=0.2,
    )
    rng = np.random.default_rng(42)
    results = run_expert_iteration(
        env=env,
        initial_state_fn=lambda r: _S,
        policy=policy,
        config=config,
        rng=rng,
    )
    probs = policy.probs(_S, ["safe", "risky"])
    p_safe, p_risky = probs[0], probs[1]
    assert p_risky > p_safe, (
        f"Policy should prefer 'risky' at S after training; "
        f"got P(safe)={p_safe:.4f}, P(risky)={p_risky:.4f}"
    )
    assert p_risky > 0.5, (
        f"Policy at S should put majority mass on 'risky'; "
        f"got P(risky)={p_risky:.4f}"
    )
    # Sanity: results list length matches the configured iteration count.
    assert len(results) == config.num_iterations


def test_evaluate_policy_returns_dict_with_mean_return() -> None:
    """The eval dict has the documented keys."""
    layout = ["S.G", "...", "..P"]
    env = Gridworld(layout)
    policy: TabularPolicy = TabularPolicy()
    rng = np.random.default_rng(1)
    summary = evaluate_policy(
        env=env,
        initial_state_fn=lambda r: env.start,
        policy=policy,
        discount=0.99,
        num_episodes=3,
        max_steps=20,
        rng=rng,
    )
    assert set(summary.keys()) >= {
        "mean_return",
        "std_return",
        "mean_episode_length",
    }
    assert math.isfinite(summary["mean_return"])
    assert summary["std_return"] >= 0.0
    assert summary["mean_episode_length"] >= 0.0


def test_evaluate_with_search_at_zero_sims_matches_fast_mode() -> None:
    """With zero MCTS simulations, search mode delegates to fast mode.

    Discussion: a 1-sim MCTS is technically well-defined (the loop runs
    once, expanding the root, doing one rollout, and backing up), but it
    would still produce a tree-based action that need not agree with the
    policy's greedy choice. The principled behavior at simulations = 0
    is to fall back to greedy policy rollout. We special-case this to
    avoid asking ``best_action`` to choose among zero expanded children.
    The test asserts that the two paths agree exactly on the same seed.
    """
    layout = ["S.G", "...", "..P"]
    env = Gridworld(layout)
    policy: TabularPolicy = TabularPolicy()
    # Give the policy a non-uniform structure so greedy is non-trivial.
    from verifier_anchored_rl.gridworld import GridAction, GridState

    policy.set_logit(GridState(0, 0), GridAction.RIGHT, 1.0)

    rng_fast = np.random.default_rng(7)
    rng_search = np.random.default_rng(7)
    fast = evaluate_policy(
        env=env,
        initial_state_fn=lambda r: env.start,
        policy=policy,
        discount=0.99,
        num_episodes=5,
        max_steps=25,
        rng=rng_fast,
    )
    search = evaluate_with_search(
        env=env,
        initial_state_fn=lambda r: env.start,
        policy=policy,
        mcts_simulations=0,
        discount=0.99,
        num_episodes=5,
        max_steps=25,
        rng=rng_search,
    )
    assert math.isclose(fast["mean_return"], search["mean_return"], abs_tol=1e-12)
    assert math.isclose(
        fast["mean_episode_length"], search["mean_episode_length"], abs_tol=1e-12
    )
    assert fast["num_episodes"] == search["num_episodes"]
