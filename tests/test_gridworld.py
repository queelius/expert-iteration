"""Tests for the stochastic Gridworld environment."""

from __future__ import annotations

import math

import numpy as np
import pytest

from verifier_anchored_rl.gridworld import GridAction, GridState, Gridworld


def _example_layout() -> list[str]:
    """The canonical example from design.md (section 5.1)."""
    return [
        "S..G",
        ".#.P",
        "....",
    ]


def test_transition_probabilities_sum_to_one() -> None:
    env = Gridworld(_example_layout())
    for state in env.states():
        if env.is_terminal(state):
            continue
        for action in env.actions(state):
            outcomes = env.transitions(state, action)
            total = sum(prob for (_, prob, _) in outcomes)
            assert math.isclose(total, 1.0, abs_tol=1e-12), (
                f"transitions for ({state}, {action}) sum to {total}, not 1.0"
            )


def test_action_into_wall_keeps_agent_in_place() -> None:
    # In the example layout, cell (0,0) is the start. Action LEFT walks
    # off the grid; intended outcome should leave the agent at (0,0).
    env = Gridworld(_example_layout())
    state = GridState(0, 0)
    outcomes = dict()
    for ns, prob, _reward in env.transitions(state, GridAction.LEFT):
        outcomes[ns] = outcomes.get(ns, 0.0) + prob
    # The intended-direction outcome (LEFT) is blocked by the left edge,
    # so the agent stays at (0,0) with at least 0.8 probability.
    assert outcomes.get(GridState(0, 0), 0.0) >= 0.8 - 1e-12

    # Cell (0,1) borders the wall at (1,1). Action DOWN's intended outcome
    # is blocked, so the agent stays at (0,1) with at least 0.8 probability.
    state = GridState(0, 1)
    outcomes = dict()
    for ns, prob, _reward in env.transitions(state, GridAction.DOWN):
        outcomes[ns] = outcomes.get(ns, 0.0) + prob
    assert outcomes.get(GridState(0, 1), 0.0) >= 0.8 - 1e-12


def test_action_off_grid_keeps_agent_in_place() -> None:
    env = Gridworld(_example_layout())
    # Top-right corner G is terminal, so use the corner just below the
    # top row to test the right-edge boundary. (0,3) is goal terminal,
    # try (1,3) which is the pit; instead use (2,3) which is ordinary.
    state = GridState(2, 3)
    outcomes = dict()
    for ns, prob, _reward in env.transitions(state, GridAction.RIGHT):
        outcomes[ns] = outcomes.get(ns, 0.0) + prob
    # RIGHT walks off the grid; the agent stays at (2,3).
    assert outcomes.get(GridState(2, 3), 0.0) >= 0.8 - 1e-12


def test_terminal_states_have_no_actions() -> None:
    env = Gridworld(_example_layout())
    goal = GridState(0, 3)
    pit = GridState(1, 3)
    assert env.is_terminal(goal)
    assert env.is_terminal(pit)
    assert list(env.actions(goal)) == []
    assert list(env.actions(pit)) == []


def test_goal_and_pit_rewards_are_correct() -> None:
    env = Gridworld(_example_layout())
    # State (0,2) is adjacent to goal at (0,3); action RIGHT's intended
    # outcome lands on the goal.
    state = GridState(0, 2)
    goal = GridState(0, 3)
    outcomes = {
        ns: (prob, reward) for (ns, prob, reward) in env.transitions(state, GridAction.RIGHT)
    }
    assert goal in outcomes
    _prob, reward = outcomes[goal]
    assert reward == env.goal_reward

    # State (1,2) is adjacent to pit at (1,3); action RIGHT's intended
    # outcome lands on the pit.
    state = GridState(1, 2)
    pit = GridState(1, 3)
    outcomes = {
        ns: (prob, reward) for (ns, prob, reward) in env.transitions(state, GridAction.RIGHT)
    }
    assert pit in outcomes
    _prob, reward = outcomes[pit]
    assert reward == env.pit_reward


def test_step_reward_applied_for_ordinary_landings() -> None:
    env = Gridworld(_example_layout(), step_reward=-0.5, goal_reward=2.0, pit_reward=-3.0)
    state = GridState(0, 0)
    # All RIGHT outcomes land on ordinary cells: (0,1), (0,0) if blocked,
    # (1,0). Reward for each non-terminal landing should be step_reward.
    for ns, _prob, reward in env.transitions(state, GridAction.RIGHT):
        if env.is_terminal(ns):
            continue
        assert reward == -0.5


def test_states_excludes_walls() -> None:
    env = Gridworld(_example_layout())
    wall = GridState(1, 1)
    assert wall not in env.states()
    # And no transition from any non-terminal state lands at the wall.
    for state in env.states():
        if env.is_terminal(state):
            continue
        for action in env.actions(state):
            for ns, _prob, _reward in env.transitions(state, action):
                assert ns != wall


def test_sample_transition_returns_valid_outcome() -> None:
    env = Gridworld(_example_layout())
    rng = np.random.default_rng(42)
    state = GridState(0, 0)
    for _ in range(100):
        ns, reward = env.sample_transition(state, GridAction.RIGHT, rng)
        # The reward should be one of step/goal/pit (no goal/pit reachable
        # from (0,0) in one step, so it must be step_reward).
        assert reward == env.step_reward
        # The next state should appear in the exact-model transition list.
        valid = {o[0] for o in env.transitions(state, GridAction.RIGHT)}
        assert ns in valid


def test_sample_transition_from_terminal_raises() -> None:
    env = Gridworld(_example_layout())
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        env.sample_transition(GridState(0, 3), GridAction.UP, rng)
