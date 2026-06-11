"""MDP interfaces.

Two structural-typing protocols (PEP 544) describe the environment surface
used by the planning and learning algorithms in this package:

* ``FiniteMDP`` exposes an exact model. It enumerates states, lists
  available actions per state, and returns the full discrete transition
  distribution for every (state, action) pair. Value iteration and
  expectimax consume this interface.

* ``GenerativeMDP`` exposes a sampling interface. Given a state, an action,
  and a random generator, it draws one (next_state, reward) pair. MCTS and
  any sample-based learner consume this interface.

The same environment can implement both; Gridworld does so.

The interfaces follow the RL framing in design.md (section 1): the central
object is an agent interacting with an environment and learning from
sampled reward-bearing transitions. The two protocols expose the two
classical access patterns to the underlying MDP.
"""

from __future__ import annotations

from typing import Protocol, Sequence, TypeVar, runtime_checkable

S = TypeVar("S")
A = TypeVar("A")


@runtime_checkable
class FiniteMDP(Protocol[S, A]):
    """Exact-model interface for finite, discrete MDPs.

    Implementations must enumerate the state space and, for each
    (state, action) pair, return the discrete distribution over
    (next_state, probability, reward) outcomes.
    """

    def states(self) -> Sequence[S]:
        """Return the (finite) state space."""
        ...

    def actions(self, state: S) -> Sequence[A]:
        """Return the actions available in ``state``.

        Terminal states are expected to return an empty sequence under
        the convention adopted by the package.
        """
        ...

    def transitions(self, state: S, action: A) -> Sequence[tuple[S, float, float]]:
        """Return the discrete (next_state, probability, reward) outcomes.

        The probabilities of the returned outcomes (for a fixed state,
        action pair) must sum to 1 within numerical tolerance. Outcomes
        with the same next_state but different rewards are kept as
        separate entries; outcomes with the same next_state and reward
        should be merged by summing their probabilities.
        """
        ...

    def is_terminal(self, state: S) -> bool:
        """Return True if ``state`` is terminal (absorbing, zero value)."""
        ...


@runtime_checkable
class GenerativeMDP(Protocol[S, A]):
    """Sampling interface for MDPs.

    Used by MCTS and any sample-based algorithm that does not need to
    enumerate transitions.
    """

    def actions(self, state: S) -> Sequence[A]:
        """Return the actions available in ``state``."""
        ...

    def sample_transition(self, state: S, action: A, rng) -> tuple[S, float]:
        """Sample one (next_state, reward) pair from the MDP dynamics.

        ``rng`` is expected to be a ``numpy.random.Generator`` (or any
        object that exposes the same sampling methods).
        """
        ...

    def is_terminal(self, state: S) -> bool:
        """Return True if ``state`` is terminal."""
        ...
