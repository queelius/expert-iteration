"""Exact value iteration for finite MDPs.

Implements the synchronous Bellman backup from design.md (section 6).
Terminal states are forced to V = 0; for every non-terminal state the
new value is the maximum over actions of the expected one-step return
plus discounted next-state value. After convergence, a greedy policy is
extracted from V using the same backup.
"""

from __future__ import annotations

from typing import Hashable, TypeVar

from .mdp import FiniteMDP

S = TypeVar("S", bound=Hashable)
A = TypeVar("A")


def _q_value(
    mdp: FiniteMDP[S, A], state: S, action: A, V: dict[S, float], gamma: float
) -> float:
    """Compute the Bellman backup Q(s, a) given the current value table."""
    q = 0.0
    for next_state, prob, reward in mdp.transitions(state, action):
        q += prob * (reward + gamma * V[next_state])
    return q


def extract_greedy_policy(
    mdp: FiniteMDP[S, A], V: dict[S, float], gamma: float
) -> dict[S, A]:
    """Return a greedy policy from ``V`` (one action per non-terminal state)."""
    policy: dict[S, A] = {}
    for state in mdp.states():
        if mdp.is_terminal(state):
            continue
        actions = list(mdp.actions(state))
        if not actions:
            continue
        best_action = max(actions, key=lambda a: _q_value(mdp, state, a, V, gamma))
        policy[state] = best_action
    return policy


def value_iteration(
    mdp: FiniteMDP[S, A],
    gamma: float,
    theta: float = 1e-6,
    max_iters: int = 10_000,
) -> tuple[dict[S, float], dict[S, A]]:
    """Run synchronous value iteration to convergence.

    Parameters
    ----------
    mdp:
        Any object satisfying :class:`FiniteMDP`.
    gamma:
        Discount factor in [0, 1].
    theta:
        Convergence threshold on the max value change between iterations.
    max_iters:
        Maximum number of sweeps before giving up.

    Returns
    -------
    (V, policy):
        ``V`` maps every state to its optimal value (terminals are 0.0).
        ``policy`` maps every non-terminal state to a greedy best action.
    """
    states = list(mdp.states())
    V: dict[S, float] = {s: 0.0 for s in states}

    for _ in range(max_iters):
        delta = 0.0
        new_V: dict[S, float] = dict(V)

        for state in states:
            if mdp.is_terminal(state):
                new_V[state] = 0.0
                continue

            actions = list(mdp.actions(state))
            if not actions:
                # No legal actions and not terminal: treat as absorbing
                # zero. This case is not expected under the design.md
                # convention but we handle it gracefully.
                new_V[state] = 0.0
                continue

            new_V[state] = max(_q_value(mdp, state, a, V, gamma) for a in actions)
            delta = max(delta, abs(new_V[state] - V[state]))

        V = new_V

        if delta < theta:
            break

    policy = extract_greedy_policy(mdp, V, gamma)
    return V, policy
