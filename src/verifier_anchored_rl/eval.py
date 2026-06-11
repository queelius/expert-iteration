"""Policy evaluation utilities.

Two complementary modes are provided, mirroring the inference modes
discussed in the framework paper (section 4.8) scaled down to the
gridworld setting:

* ``evaluate_policy`` (fast mode, B = 0): roll out the policy by taking
  its greedy argmax action at every step. No search at inference time.
  This measures the quality of the distilled parametric policy itself.
* ``evaluate_with_search`` (search mode, B > 0): roll out the policy
  with MCTS guidance at every action selection. The policy supplies the
  prior; MCTS supplies the visit-count action. This measures the
  combined system's quality, and the gap between the two modes
  quantifies how much value the search procedure still adds on top of
  the policy.

Both functions return a dict with the same keys so that experiments can
report them side by side.
"""

from __future__ import annotations

from typing import Callable, Hashable, TypeVar

import numpy as np

from .mcts import MCTSConfig, best_action, mcts_search
from .mdp import GenerativeMDP
from .policies import TabularPolicy, policy_as_prior

S = TypeVar("S", bound=Hashable)
A = TypeVar("A", bound=Hashable)


def _greedy_action(policy: TabularPolicy[S, A], state: S, actions: list[A]) -> A:
    """Return the argmax-logit action; ties broken by ``actions`` order."""
    best = actions[0]
    best_logit = policy.logit(state, best)
    for a in actions[1:]:
        l = policy.logit(state, a)
        if l > best_logit:
            best_logit = l
            best = a
    return best


def _summarize(returns: list[float], lengths: list[int]) -> dict:
    """Pack the per-episode returns and lengths into a summary dict."""
    if not returns:
        return {
            "mean_return": 0.0,
            "std_return": 0.0,
            "mean_episode_length": 0.0,
            "num_episodes": 0,
        }
    arr = np.array(returns, dtype=float)
    return {
        "mean_return": float(arr.mean()),
        "std_return": float(arr.std(ddof=0)),
        "mean_episode_length": float(np.mean(lengths)),
        "num_episodes": len(returns),
    }


def evaluate_policy(
    env: GenerativeMDP[S, A],
    initial_state_fn: Callable[[np.random.Generator], S],
    policy: TabularPolicy[S, A],
    discount: float,
    num_episodes: int,
    max_steps: int,
    rng: np.random.Generator,
) -> dict:
    """Roll out the greedy policy and report return statistics.

    Picks the argmax-logit action at every visited state (fast mode,
    B = 0 in the framework paper's terminology). The returned dict has
    keys ``mean_return``, ``std_return``, ``mean_episode_length``, and
    ``num_episodes``.
    """
    returns: list[float] = []
    lengths: list[int] = []

    for _ in range(num_episodes):
        state = initial_state_fn(rng)
        episode_return = 0.0
        discount_power = 1.0
        steps = 0

        while steps < max_steps and not env.is_terminal(state):
            actions = list(env.actions(state))
            if not actions:
                break
            action = _greedy_action(policy, state, actions)
            next_state, reward = env.sample_transition(state, action, rng)
            episode_return += discount_power * reward
            discount_power *= discount
            state = next_state
            steps += 1

        returns.append(episode_return)
        lengths.append(steps)

    return _summarize(returns, lengths)


def evaluate_with_search(
    env: GenerativeMDP[S, A],
    initial_state_fn: Callable[[np.random.Generator], S],
    policy: TabularPolicy[S, A],
    mcts_simulations: int,
    discount: float,
    num_episodes: int,
    max_steps: int,
    rng: np.random.Generator,
    c_puct: float = 1.0,
) -> dict:
    """Roll out the policy with MCTS guidance at every action selection.

    For each step, MCTS is run with the policy as its prior; the action
    is the visit-count argmax (deterministic, low temperature limit).
    This is the framework paper's search mode (B > 0).

    If ``mcts_simulations <= 0`` the call delegates to
    ``evaluate_policy``: with zero simulations the search tree has no
    expansions and ``best_action`` is undefined, so the only sensible
    behavior is to fall back to the policy's greedy action. This keeps
    the API uniform without a special-cased "B = 0" branch leaking into
    the rest of the codebase.
    """
    if mcts_simulations <= 0:
        return evaluate_policy(
            env=env,
            initial_state_fn=initial_state_fn,
            policy=policy,
            discount=discount,
            num_episodes=num_episodes,
            max_steps=max_steps,
            rng=rng,
        )

    mcts_config = MCTSConfig(
        num_simulations=mcts_simulations,
        c_puct=c_puct,
        discount=discount,
        prior_fn=policy_as_prior(policy),
    )

    returns: list[float] = []
    lengths: list[int] = []

    for _ in range(num_episodes):
        state = initial_state_fn(rng)
        episode_return = 0.0
        discount_power = 1.0
        steps = 0

        while steps < max_steps and not env.is_terminal(state):
            root = mcts_search(env, state, mcts_config, rng)
            if not root.children:
                # Search expanded nothing (terminal or pathological);
                # fall back to the greedy policy action so the rollout
                # can still terminate cleanly.
                actions = list(env.actions(state))
                if not actions:
                    break
                action = _greedy_action(policy, state, actions)
            else:
                action = best_action(root)
            next_state, reward = env.sample_transition(state, action, rng)
            episode_return += discount_power * reward
            discount_power *= discount
            state = next_state
            steps += 1

        returns.append(episode_return)
        lengths.append(steps)

    return _summarize(returns, lengths)


__all__ = [
    "evaluate_policy",
    "evaluate_with_search",
]
