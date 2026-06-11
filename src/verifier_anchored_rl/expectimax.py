"""Depth-limited expectimax planner.

Implements the recursive value function from design.md (section 7).
At a terminal state or when depth reaches zero, the returned value is
0. Otherwise the value is the maximum over actions of the expected
one-step return plus the discounted recursive value at the next state.

An optional memoization cache, keyed by ``(state, depth)``, exposes the
overlap between tree expansion and Bellman-style reuse.
"""

from __future__ import annotations

from typing import Hashable, TypeVar

from .mdp import FiniteMDP

S = TypeVar("S", bound=Hashable)
A = TypeVar("A")


def expectimax_value(
    mdp: FiniteMDP[S, A],
    state: S,
    depth: int,
    gamma: float,
    cache: dict | None = None,
) -> float:
    """Return the depth-limited expectimax value of ``state``.

    Parameters
    ----------
    mdp:
        Exact-model MDP.
    state:
        State to evaluate.
    depth:
        Remaining ply depth. ``depth == 0`` returns 0.0.
    gamma:
        Discount factor.
    cache:
        Optional dict for memoization. If provided, results are cached
        under ``(state, depth)``.
    """
    if mdp.is_terminal(state) or depth == 0:
        return 0.0

    if cache is not None:
        key = (state, depth)
        if key in cache:
            return cache[key]

    actions = list(mdp.actions(state))
    if not actions:
        result = 0.0
    else:
        best = float("-inf")
        for action in actions:
            q = 0.0
            for next_state, prob, reward in mdp.transitions(state, action):
                q += prob * (
                    reward
                    + gamma * expectimax_value(mdp, next_state, depth - 1, gamma, cache)
                )
            if q > best:
                best = q
        result = best

    if cache is not None:
        cache[key] = result

    return result


def expectimax_policy(
    mdp: FiniteMDP[S, A],
    state: S,
    depth: int,
    gamma: float,
    cache: dict | None = None,
) -> A:
    """Return the best action at ``state`` under depth-limited expectimax.

    Tie-breaking is deterministic (the first action achieving the max
    wins, in the order returned by ``mdp.actions(state)``).
    """
    actions = list(mdp.actions(state))
    if not actions:
        raise ValueError("no actions available at the given state")

    best_action = actions[0]
    best_value = float("-inf")
    for action in actions:
        q = 0.0
        for next_state, prob, reward in mdp.transitions(state, action):
            q += prob * (
                reward
                + gamma * expectimax_value(mdp, next_state, depth - 1, gamma, cache)
            )
        if q > best_value:
            best_value = q
            best_action = action
    return best_action
