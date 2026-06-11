"""Stochastic Gridworld environment.

Implements the small rectangular grid described in design.md (section 5).
Actions are noisy: the intended direction is taken with probability 0.8,
each perpendicular direction with probability 0.1. If the intended next
cell is a wall or off-grid, the agent stays in place. Goal and pit cells
are terminal.

The ``Gridworld`` class implements both the FiniteMDP and GenerativeMDP
protocols from :mod:`verifier_anchored_rl.mdp`.

Terminal convention: ``actions(terminal) == []`` and ``is_terminal`` is
True. Transitions out of a terminal state are not defined and callers
should guard with ``is_terminal`` before requesting actions or
transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

CellKind = str  # one of "S", ".", "#", "G", "P"


@dataclass(frozen=True)
class GridState:
    """A grid cell coordinate (zero-indexed, row-major)."""

    row: int
    col: int


class GridAction(Enum):
    """The four cardinal-direction actions."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


# Movement deltas in (row, col) for each action.
_DELTAS: dict[GridAction, tuple[int, int]] = {
    GridAction.UP: (-1, 0),
    GridAction.DOWN: (1, 0),
    GridAction.LEFT: (0, -1),
    GridAction.RIGHT: (0, 1),
}

# Perpendicular pair for each intended action (the two "noise" outcomes).
_PERPENDICULAR: dict[GridAction, tuple[GridAction, GridAction]] = {
    GridAction.UP: (GridAction.LEFT, GridAction.RIGHT),
    GridAction.DOWN: (GridAction.LEFT, GridAction.RIGHT),
    GridAction.LEFT: (GridAction.UP, GridAction.DOWN),
    GridAction.RIGHT: (GridAction.UP, GridAction.DOWN),
}

# Probability mass for each of the three outcomes (intended, perp1, perp2).
INTENDED_PROB: float = 0.8
PERP_PROB: float = 0.1


class Gridworld:
    """Stochastic Gridworld MDP.

    Parameters
    ----------
    layout:
        List of equal-length strings. Each character is one of:
        ``"S"`` (start), ``"."`` (ordinary cell), ``"#"`` (wall),
        ``"G"`` (goal terminal), ``"P"`` (pit terminal).
    step_reward:
        Reward for transitions that land on an ordinary cell.
    goal_reward:
        Reward for transitions that land on the goal cell.
    pit_reward:
        Reward for transitions that land on the pit cell.
    """

    def __init__(
        self,
        layout: Sequence[str],
        step_reward: float = -0.01,
        goal_reward: float = 1.0,
        pit_reward: float = -1.0,
    ) -> None:
        if not layout:
            raise ValueError("layout must be non-empty")
        width = len(layout[0])
        for row in layout:
            if len(row) != width:
                raise ValueError("layout rows must all have the same length")

        self._grid: list[list[CellKind]] = [list(row) for row in layout]
        self.n_rows: int = len(self._grid)
        self.n_cols: int = width
        self.step_reward: float = step_reward
        self.goal_reward: float = goal_reward
        self.pit_reward: float = pit_reward

        # Locate the start state. There must be exactly one.
        starts = [
            GridState(r, c)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            if self._grid[r][c] == "S"
        ]
        if len(starts) != 1:
            raise ValueError(f"layout must have exactly one 'S' cell, found {len(starts)}")
        self.start: GridState = starts[0]

        # Cache the full state list (every grid cell, including walls).
        # We include walls because callers may iterate; ``actions`` will
        # return an empty list for them. ``states()`` returns only the
        # non-wall cells, which is the usable state space.
        self._states: list[GridState] = [
            GridState(r, c)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            if self._grid[r][c] != "#"
        ]

    # ----- helpers -----

    def cell_kind(self, state: GridState) -> CellKind:
        """Return the character at ``state``."""
        return self._grid[state.row][state.col]

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.n_rows and 0 <= col < self.n_cols

    def _is_wall(self, row: int, col: int) -> bool:
        return self._grid[row][col] == "#"

    def _resolve_move(self, state: GridState, direction: GridAction) -> GridState:
        """Apply a single movement, with wall/boundary stay-in-place."""
        d_row, d_col = _DELTAS[direction]
        new_row, new_col = state.row + d_row, state.col + d_col
        if not self._in_bounds(new_row, new_col) or self._is_wall(new_row, new_col):
            return state
        return GridState(new_row, new_col)

    def _reward_for_entering(self, next_state: GridState) -> float:
        kind = self.cell_kind(next_state)
        if kind == "G":
            return self.goal_reward
        if kind == "P":
            return self.pit_reward
        return self.step_reward

    # ----- FiniteMDP interface -----

    def states(self) -> Sequence[GridState]:
        return list(self._states)

    def actions(self, state: GridState) -> Sequence[GridAction]:
        if self.is_terminal(state):
            return []
        return list(GridAction)

    def transitions(
        self, state: GridState, action: GridAction
    ) -> Sequence[tuple[GridState, float, float]]:
        if self.is_terminal(state):
            return []

        perp_a, perp_b = _PERPENDICULAR[action]
        outcomes: list[tuple[GridState, float, float]] = [
            (self._resolve_move(state, action), INTENDED_PROB, 0.0),
            (self._resolve_move(state, perp_a), PERP_PROB, 0.0),
            (self._resolve_move(state, perp_b), PERP_PROB, 0.0),
        ]

        # Fill in rewards (depend only on next_state under the
        # design.md reward convention).
        outcomes = [
            (next_state, prob, self._reward_for_entering(next_state))
            for (next_state, prob, _) in outcomes
        ]

        # Merge entries that share (next_state, reward) by summing
        # probabilities. Two of the three movement outcomes may collide
        # when a wall sends the agent back to the same cell.
        merged: dict[tuple[GridState, float], float] = {}
        for next_state, prob, reward in outcomes:
            key = (next_state, reward)
            merged[key] = merged.get(key, 0.0) + prob

        return [(ns, p, r) for (ns, r), p in merged.items()]

    def is_terminal(self, state: GridState) -> bool:
        kind = self.cell_kind(state)
        return kind in ("G", "P")

    # ----- GenerativeMDP interface -----

    def sample_transition(
        self, state: GridState, action: GridAction, rng: np.random.Generator
    ) -> tuple[GridState, float]:
        """Sample one (next_state, reward) pair from the dynamics."""
        if self.is_terminal(state):
            raise ValueError("cannot sample a transition from a terminal state")
        outcomes = self.transitions(state, action)
        next_states = [o[0] for o in outcomes]
        probs = np.array([o[1] for o in outcomes], dtype=float)
        rewards = [o[2] for o in outcomes]
        # Renormalize to absorb any rounding drift.
        probs = probs / probs.sum()
        idx = int(rng.choice(len(next_states), p=probs))
        return next_states[idx], rewards[idx]

    # ----- pretty printing -----

    def render_layout(self) -> str:
        """Return the original layout as a single multi-line string."""
        return "\n".join("".join(row) for row in self._grid)
