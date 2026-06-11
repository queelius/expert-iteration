"""Tests for the value iteration solver."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import pytest

from verifier_anchored_rl.gridworld import GridState, Gridworld
from verifier_anchored_rl.value_iteration import value_iteration


# ----------------------------------------------------------------------
# Hand-computable S/A/G/B MDP from design.md section 10.2.
#
#     S --safe----> G              reward 4 (deterministic)
#     S --risky--> A  with p=0.8   reward  0
#     S --risky--> B  with p=0.2   reward -10
#     A --finish--> G              reward 10 (deterministic)
#     G terminal
#     B terminal
#
# With gamma = 1 and terminal values zero:
#     V*(G) = 0
#     V*(B) = 0
#     V*(A) = 10
#     Q*(S, safe)  = 4
#     Q*(S, risky) = 0.8 * 10 + 0.2 * (-10) = 6
#     V*(S) = 6, pi*(S) = risky
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _Node:
    """Simple state identifier for the toy MDP."""

    name: str


_S = _Node("S")
_A = _Node("A")
_G = _Node("G")
_B = _Node("B")


class SAGBMDP:
    """The S/A/G/B MDP described above, implementing FiniteMDP."""

    def states(self) -> Sequence[_Node]:
        return [_S, _A, _G, _B]

    def actions(self, state: _Node) -> Sequence[str]:
        if state == _S:
            return ["safe", "risky"]
        if state == _A:
            return ["finish"]
        return []

    def transitions(
        self, state: _Node, action: str
    ) -> Sequence[tuple[_Node, float, float]]:
        if state == _S and action == "safe":
            return [(_G, 1.0, 4.0)]
        if state == _S and action == "risky":
            return [(_A, 0.8, 0.0), (_B, 0.2, -10.0)]
        if state == _A and action == "finish":
            return [(_G, 1.0, 10.0)]
        return []

    def is_terminal(self, state: _Node) -> bool:
        return state in (_G, _B)


def test_value_iteration_on_hand_computable_mdp() -> None:
    mdp = SAGBMDP()
    V, policy = value_iteration(mdp, gamma=1.0, theta=1e-9, max_iters=1_000)
    assert math.isclose(V[_G], 0.0, abs_tol=1e-9)
    assert math.isclose(V[_B], 0.0, abs_tol=1e-9)
    assert math.isclose(V[_A], 10.0, abs_tol=1e-9)
    assert math.isclose(V[_S], 6.0, abs_tol=1e-9)
    assert policy[_S] == "risky"
    assert policy[_A] == "finish"


def test_value_iteration_terminal_values_are_zero() -> None:
    layout = ["S.G", "..P"]
    env = Gridworld(layout)
    V, _ = value_iteration(env, gamma=0.99, theta=1e-9)
    for state in env.states():
        if env.is_terminal(state):
            assert V[state] == 0.0


def test_value_near_goal_higher_than_near_pit() -> None:
    # Layout: S in top-left, goal at top-right, pit at bottom-right.
    # The cell just left of the goal should have higher value than the
    # cell just left of the pit.
    layout = [
        "S..G",
        "...P",
    ]
    env = Gridworld(layout)
    V, _ = value_iteration(env, gamma=0.99, theta=1e-9)
    near_goal = GridState(0, 2)
    near_pit = GridState(1, 2)
    assert V[near_goal] > V[near_pit]
    # Sanity: V[near_goal] should be positive (close to goal_reward
    # discounted by one step, minus a small step penalty).
    assert V[near_goal] > 0.0


def test_optimal_policy_avoids_walking_into_pit() -> None:
    # The cell immediately to the left of the pit should prefer not to
    # move RIGHT (toward the pit) under the optimal policy.
    from verifier_anchored_rl.gridworld import GridAction

    layout = [
        "S..G",
        "...P",
    ]
    env = Gridworld(layout)
    _V, policy = value_iteration(env, gamma=0.99, theta=1e-9)
    near_pit = GridState(1, 2)
    assert policy[near_pit] != GridAction.RIGHT


def test_value_iteration_converges_within_max_iters() -> None:
    layout = ["S.G", "..P"]
    env = Gridworld(layout)
    # If max_iters is too low we still return cleanly; this just exercises
    # the path.
    V, policy = value_iteration(env, gamma=0.9, theta=1e-9, max_iters=5_000)
    # Every non-terminal state should have an assigned greedy action.
    for state in env.states():
        if not env.is_terminal(state):
            assert state in policy
