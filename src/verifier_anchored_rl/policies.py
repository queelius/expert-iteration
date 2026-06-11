"""Parametric policies for expert iteration.

This module provides the parametric policy that closes the expert
iteration loop: a tabular softmax over per-(state, action) logits.

The pedagogical framing here follows design.md (sections 1 and 9): the
central object is an agent interacting with an environment and learning
from sampled reward-bearing transitions. MCTS is one (optional) way to
generate improved policy targets; expert iteration is the joint
distillation of those MCTS-improved estimates into a persistent
parametric policy. The policy in this module is the simplest such
parameterization, deliberately tabular per design.md section 9.1:

    "Do not begin with a neural network. Use a tabular softmax policy
    prior."

The softmax form is

    p_theta(a | s) = exp(l_theta(s, a)) / sum_b exp(l_theta(s, b))

over the action set available at state s. Cross-entropy with respect to
a target distribution q over the same action set has gradient (per
logit)

    d/d l(s, a) CE(q || p_theta) = p_theta(a | s) - q(a | s)

so one stochastic gradient step at learning rate eta is

    l(s, a) <- l(s, a) - eta * (p_theta(a | s) - q(a | s))

which is what ``TabularPolicy.train_step`` implements.

A small module-level adaptor ``policy_as_prior`` wraps a
``TabularPolicy`` as a callable matching the MCTS ``prior_fn`` signature,
so the same policy object can serve simultaneously as the distillation
target and as the search prior. This is the loop closure: each round of
MCTS uses the current policy as a prior, produces improved visit-count
targets, and the targets train the policy for the next round.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Generic, Hashable, Sequence, TypeVar

S = TypeVar("S", bound=Hashable)
A = TypeVar("A", bound=Hashable)


@dataclass
class TabularPolicy(Generic[S, A]):
    """Tabular softmax policy with per-(state, action) logits.

    Missing entries in ``logits`` default to ``default_logit``, which
    means an unseen state induces a uniform softmax over its action set
    (any constant logit gives a uniform softmax). Only the relative
    differences between logits at a given state affect the probabilities,
    so the absolute scale of ``default_logit`` is irrelevant; setting
    it to zero is the natural default.

    Parameters
    ----------
    logits:
        Dict mapping (state, action) pairs to scalar logits. Created
        lazily as ``train_step`` or ``set_logit`` introduces entries.
    default_logit:
        Logit value returned for any (state, action) not yet stored.
        Defaults to 0.0, which gives a uniform prior over actions until
        the policy receives gradient updates.
    """

    logits: dict[tuple[S, A], float] = field(default_factory=dict)
    default_logit: float = 0.0

    def logit(self, state: S, action: A) -> float:
        """Return the stored logit, falling back to ``default_logit``."""
        return self.logits.get((state, action), self.default_logit)

    def set_logit(self, state: S, action: A, value: float) -> None:
        """Set a logit explicitly.

        Useful for tests, for warm-starting the policy from some
        precomputed reference, or for unit-level inspection. The training
        loop normally calls ``train_step`` instead.
        """
        self.logits[(state, action)] = float(value)

    def probs(self, state: S, actions: Sequence[A]) -> list[float]:
        """Return softmax probabilities over ``actions`` at ``state``.

        Uses the standard max-subtraction trick for numerical stability.
        Returns an empty list if ``actions`` is empty (e.g. terminal
        states under the package-wide convention).
        """
        if not actions:
            return []
        raw = [self.logit(state, a) for a in actions]
        m = max(raw)
        exps = [math.exp(x - m) for x in raw]
        z = sum(exps)
        if z == 0.0:
            # Pathological case: should not occur for finite logits since
            # exp(x - max) >= exp(0) for the argmax entry. Return uniform.
            n = len(actions)
            return [1.0 / n] * n
        return [e / z for e in exps]

    def train_step(
        self,
        state: S,
        target_probs: dict[A, float],
        learning_rate: float,
    ) -> None:
        """Take one cross-entropy gradient step at ``state``.

        The target distribution ``target_probs`` must be over actions
        actually available at ``state``. The gradient of the cross
        entropy CE(target || p_theta) with respect to the logits is

            d/d l(s, a) = p_theta(a | s) - target(a | s)

        and the update is

            l(s, a) <- l(s, a) - learning_rate * (p_theta - target).

        Any action that appears in ``target_probs`` participates in the
        update; actions absent from ``target_probs`` are skipped (their
        target is implicitly zero only if they belong to the same support,
        but we follow the convention that the caller passes the full
        target distribution over the active action set).
        """
        if not target_probs:
            return
        actions = list(target_probs.keys())
        current = self.probs(state, actions)
        for a, p_theta, p_target in zip(actions, current, target_probs.values()):
            grad = p_theta - p_target
            new_logit = self.logit(state, a) - learning_rate * grad
            self.logits[(state, a)] = new_logit


def policy_as_prior(
    policy: TabularPolicy[S, A],
) -> Callable[[S, Sequence[A]], list[float]]:
    """Adapt a ``TabularPolicy`` to the MCTS ``prior_fn`` signature.

    The returned callable has signature ``(state, actions) -> probs``
    and returns the softmax probability of each action at the given
    state, in the same order as ``actions``. It is the bridge that lets
    expert iteration use the current policy as the MCTS prior on every
    outer iteration.
    """

    def prior(state: S, actions: Sequence[A]) -> list[float]:
        return policy.probs(state, actions)

    return prior


__all__ = [
    "TabularPolicy",
    "policy_as_prior",
]
