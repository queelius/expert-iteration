"""Tests for the depth-limited expectimax planner."""

from __future__ import annotations

import math

from verifier_anchored_rl.expectimax import expectimax_policy, expectimax_value
from verifier_anchored_rl.gridworld import Gridworld
from verifier_anchored_rl.value_iteration import value_iteration

from tests.test_value_iteration import SAGBMDP, _A, _S


def test_expectimax_matches_value_iteration_on_sagb() -> None:
    mdp = SAGBMDP()
    V_vi, _ = value_iteration(mdp, gamma=1.0, theta=1e-9, max_iters=1_000)
    # The S/A/G/B MDP terminates within 2 ply from S; depth 10 is more
    # than enough for the expectimax value to equal V*.
    for state in [_S, _A]:
        v = expectimax_value(mdp, state, depth=10, gamma=1.0)
        assert math.isclose(v, V_vi[state], abs_tol=1e-9), (
            f"expectimax({state}) = {v}, value_iteration = {V_vi[state]}"
        )

    # Best action at S should agree with value iteration.
    assert expectimax_policy(mdp, _S, depth=10, gamma=1.0) == "risky"


def test_expectimax_returns_zero_at_depth_zero() -> None:
    mdp = SAGBMDP()
    assert expectimax_value(mdp, _S, depth=0, gamma=1.0) == 0.0


def test_expectimax_returns_zero_at_terminal() -> None:
    mdp = SAGBMDP()
    from tests.test_value_iteration import _B, _G

    assert expectimax_value(mdp, _G, depth=5, gamma=1.0) == 0.0
    assert expectimax_value(mdp, _B, depth=5, gamma=1.0) == 0.0


def test_expectimax_matches_value_iteration_on_small_gridworld() -> None:
    # Use a small layout so that we can afford a depth large enough for
    # expectimax to approximate the infinite-horizon value iteration
    # result. With gamma = 0.9, depth = 25 has a remaining factor of
    # 0.9 ** 25 = 0.072, which is small enough for a loose tolerance.
    layout = [
        "S.G",
        "..P",
    ]
    env = Gridworld(layout)
    gamma = 0.9
    V_vi, _ = value_iteration(env, gamma=gamma, theta=1e-9)

    cache: dict = {}
    for state in env.states():
        if env.is_terminal(state):
            continue
        v_em = expectimax_value(env, state, depth=25, gamma=gamma, cache=cache)
        # The remaining-horizon error is bounded by gamma**depth times
        # the max reward magnitude. With max |reward| = 1.0 the bound is
        # 0.9 ** 25 ~= 0.072, so allow 0.1 slack.
        assert math.isclose(v_em, V_vi[state], abs_tol=0.1), (
            f"expectimax({state}) = {v_em}, value_iteration = {V_vi[state]}"
        )


def test_expectimax_cache_reuse_matches_uncached() -> None:
    mdp = SAGBMDP()
    cache: dict = {}
    v_cached = expectimax_value(mdp, _S, depth=5, gamma=1.0, cache=cache)
    v_uncached = expectimax_value(mdp, _S, depth=5, gamma=1.0)
    assert math.isclose(v_cached, v_uncached, abs_tol=1e-12)
    # The cache should be non-empty after the call.
    assert len(cache) > 0
