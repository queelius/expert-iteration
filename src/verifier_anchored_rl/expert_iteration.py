"""Expert iteration training loop.

Implements the simple expert-iteration loop from design.md (section 9):
each outer iteration runs a batch of episodes in which MCTS, guided by
the current policy as a prior, produces improved action distributions at
every visited state. Those improved distributions are stored as
distillation targets and consumed by a cross-entropy gradient step on
the policy logits. The next outer iteration begins with a stronger
prior, which in turn makes MCTS sample-efficient enough to keep pushing
the targets forward.

The framing follows design.md (section 1.9): MCTS is a sample-efficient
policy improvement operator, expert iteration is the joint distillation
into a parametric policy. The agent learns from sampled reward-bearing
transitions; MCTS just makes each sample do more work. Nothing here
requires a neural network: a tabular policy is sufficient to demonstrate
the loop on the gridworld and on the hand-computable SAGB MDP.

Default behavior choices (documented per design.md):

* The MCTS root value used as a side-channel in each ``PolicyTarget`` is
  the visit-weighted mean of the root chance-node Q estimates. This is
  the V-from-root estimator implied by V(s) = E_{a ~ pi_MCTS}[Q(s, a)]
  using pi_MCTS = N(s, a) / N(s) and Qhat = total / visits per edge.
  Concretely: sum over expanded children of (N_child / total_visits) *
  Qhat_child. The most-visited-child mean would also be defensible.
* Action selection during episode rollouts samples from the visit-count
  policy at temperature ``config.mcts_temperature``. Argmax (temperature
  0) is the alternative the design document mentions; sampling preserves
  some exploration during data collection and is the default.
* Training processes the collected examples in collection order, one
  cross-entropy gradient step per (state, target) pair, repeated
  ``train_steps_per_iteration`` times. Shuffling and minibatching would
  be a sensible refinement; the simple loop is what design.md (section
  9.1) recommends to start.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, Hashable, Sequence, TypeVar

import numpy as np

from .mcts import (
    DecisionNode,
    MCTSConfig,
    mcts_search,
    visit_count_policy,
)
from .mdp import GenerativeMDP
from .policies import TabularPolicy, policy_as_prior

S = TypeVar("S", bound=Hashable)
A = TypeVar("A", bound=Hashable)


# ---------------------------------------------------------------------------
# Data classes.
# ---------------------------------------------------------------------------


@dataclass
class PolicyTarget(Generic[S, A]):
    """One distilled training target produced by MCTS at a single state.

    Per design.md (section 9.2):

    * ``state``: the root state at which MCTS was run.
    * ``target_probs``: the visit-count distribution pi_MCTS(a | s)
      derived from the root's chance-node visit counts. This is the
      distillation target for the policy cross-entropy update.
    * ``root_value``: the MCTS value estimate at the root. Stored as a
      side channel so downstream code (value head, advantage estimator,
      diagnostics) can use it without re-running search.
    """

    state: S
    target_probs: dict[A, float]
    root_value: float


@dataclass
class ExpertIterationConfig:
    """Hyperparameters for ``run_expert_iteration``.

    All defaults are picked to keep tests fast while remaining
    representative of the loop's behavior on the gridworld and the
    hand-computable SAGB MDP. Production runs will typically increase
    the per-iteration episode count and the MCTS simulation budget.
    """

    num_iterations: int = 20
    episodes_per_iteration: int = 50
    max_episode_steps: int = 100
    mcts_simulations_per_step: int = 100
    mcts_c_puct: float = 1.0
    mcts_temperature: float = 1.0
    discount: float = 0.99
    learning_rate: float = 0.1
    train_steps_per_iteration: int = 1
    eval_every: int = 1


@dataclass
class IterationResult:
    """Per-iteration diagnostics returned from ``run_expert_iteration``.

    The fields here are deliberately a minimal core; additional fields
    can be appended as new diagnostics are wired in without breaking
    callers that only read ``iteration``, ``num_examples``,
    ``mean_episode_return``, and ``mean_episode_length``.
    """

    iteration: int
    num_examples: int
    mean_episode_return: float
    mean_episode_length: float
    episode_returns: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _root_value(root: DecisionNode[S, A]) -> float:
    """Estimate V(root.state) from the search tree.

    Uses the visit-weighted mean of the root chance-node Q estimates:

        V(s) = sum_{a in expanded} (N(s, a) / N(s)) * Qhat(s, a).

    This is consistent with V(s) = E_{a ~ pi_MCTS(. | s)}[Q(s, a)] using
    the visit-count policy pi_MCTS, and it is robust to ties (no need to
    pick "the" best child). Returns 0 for terminal or unexpanded roots
    by convention.
    """
    if not root.children:
        return 0.0
    total_visits = sum(c.visits for c in root.children.values())
    if total_visits <= 0:
        return 0.0
    return sum(
        (c.visits / total_visits) * c.mean_value for c in root.children.values()
    )


def _sample_action(
    target_probs: dict[A, float],
    rng: np.random.Generator,
) -> A:
    """Sample an action from ``target_probs``.

    Falls back to deterministic argmax if the distribution sums to zero
    (which should not happen with a well-formed visit-count policy but
    is handled defensively).
    """
    actions = list(target_probs.keys())
    probs = np.array(list(target_probs.values()), dtype=float)
    total = probs.sum()
    if total <= 0.0:
        return actions[0]
    probs = probs / total
    idx = int(rng.choice(len(actions), p=probs))
    return actions[idx]


# ---------------------------------------------------------------------------
# Main loop.
# ---------------------------------------------------------------------------


def run_expert_iteration(
    env: GenerativeMDP[S, A],
    initial_state_fn: Callable[[np.random.Generator], S],
    policy: TabularPolicy[S, A],
    config: ExpertIterationConfig,
    rng: np.random.Generator,
    on_iteration: Callable[[IterationResult], None] | None = None,
) -> list[IterationResult]:
    """Run expert iteration and return the per-iteration diagnostics.

    For each outer iteration the loop:

    1. Collects ``episodes_per_iteration`` episodes. Each episode starts
       at ``initial_state_fn(rng)`` and proceeds for at most
       ``max_episode_steps`` decision steps or until the environment
       reports a terminal state. At every visited non-terminal state,
       MCTS is run with the current policy as its prior, the visit-count
       distribution is recorded as a ``PolicyTarget``, and an action is
       sampled from that distribution for the actual environment
       transition. The undiscounted-but-environment-rewarded return is
       accumulated as ``gamma`` powers of the observed rewards.
    2. Performs ``train_steps_per_iteration`` passes over the collected
       examples, in collection order, calling ``policy.train_step`` for
       each ``PolicyTarget``. This is the policy distillation step.
    3. Records an ``IterationResult`` with the iteration index, number
       of training examples, and per-episode summary statistics.

    Parameters
    ----------
    env:
        Any environment satisfying :class:`GenerativeMDP`.
    initial_state_fn:
        Callable that, given the rng, returns the starting state for one
        episode. Typically ``lambda rng: env.start`` for the gridworld.
    policy:
        Tabular policy whose logits are updated in place.
    config:
        ``ExpertIterationConfig`` of hyperparameters.
    rng:
        ``numpy.random.Generator`` used for MCTS, action sampling, and
        initial-state sampling.
    on_iteration:
        Optional callback invoked at the end of each outer iteration
        with the ``IterationResult``. Useful for progress reporting or
        for collecting evaluation metrics without modifying the loop.
    """
    mcts_config = MCTSConfig(
        num_simulations=config.mcts_simulations_per_step,
        c_puct=config.mcts_c_puct,
        discount=config.discount,
        prior_fn=policy_as_prior(policy),
    )

    results: list[IterationResult] = []

    for iteration in range(config.num_iterations):
        examples: list[PolicyTarget[S, A]] = []
        episode_returns: list[float] = []
        episode_lengths: list[int] = []

        for _ in range(config.episodes_per_iteration):
            state = initial_state_fn(rng)
            episode_return = 0.0
            discount_power = 1.0
            steps = 0

            while steps < config.max_episode_steps and not env.is_terminal(state):
                root = mcts_search(env, state, mcts_config, rng)
                if not root.children:
                    # No expansions happened (terminal or all simulations
                    # were absorbed without any expansion); nothing to
                    # distill or to act on. Break the inner loop.
                    break

                target = visit_count_policy(root, temperature=config.mcts_temperature)
                examples.append(
                    PolicyTarget(
                        state=state,
                        target_probs=target,
                        root_value=_root_value(root),
                    )
                )

                action = _sample_action(target, rng)
                next_state, reward = env.sample_transition(state, action, rng)
                episode_return += discount_power * reward
                discount_power *= config.discount
                state = next_state
                steps += 1

            episode_returns.append(episode_return)
            episode_lengths.append(steps)

        # Policy distillation: cross-entropy gradient step on each
        # (state, target) pair, repeated train_steps_per_iteration times.
        for _ in range(config.train_steps_per_iteration):
            for example in examples:
                policy.train_step(
                    example.state,
                    example.target_probs,
                    config.learning_rate,
                )

        mean_return = (
            float(np.mean(episode_returns)) if episode_returns else 0.0
        )
        mean_length = (
            float(np.mean(episode_lengths)) if episode_lengths else 0.0
        )
        result = IterationResult(
            iteration=iteration,
            num_examples=len(examples),
            mean_episode_return=mean_return,
            mean_episode_length=mean_length,
            episode_returns=episode_returns,
        )
        results.append(result)

        if on_iteration is not None:
            on_iteration(result)

    return results


__all__ = [
    "PolicyTarget",
    "ExpertIterationConfig",
    "IterationResult",
    "run_expert_iteration",
]
