"""Deterministic comparison policies for workspace design experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .model import (
    Action,
    CandidateNode,
    EventOutcome,
    State,
    Verb,
    child_goals,
    goal_candidates,
)
from .runtime import Transition, WorkspaceEnvironment


INITIALIZATION_EMPTY_ROOT = "empty-root-v1"
INITIALIZATION_DIRECT_CANDIDATE_PREFIX = "direct-candidate-prefix-v1"
INITIALIZATION_MODES = frozenset({
    INITIALIZATION_EMPTY_ROOT,
    INITIALIZATION_DIRECT_CANDIDATE_PREFIX,
})


@runtime_checkable
class WorkspacePolicy(Protocol):
    def choose(
        self,
        state: State,
        legal_actions: tuple[Action, ...],
    ) -> Action:
        ...


@dataclass(frozen=True, slots=True)
class DirectPolicy:
    """One direct worker answer followed by symbolic submission."""

    def choose(
        self,
        state: State,
        legal_actions: tuple[Action, ...],
    ) -> Action:
        root_candidates = goal_candidates(state, 0)
        action = (
            Action(Verb.ANSWER, 0)
            if not root_candidates
            else Action(Verb.SUBMIT, root_candidates[-1].id)
        )
        if action not in legal_actions:
            raise ValueError("direct policy action is not legal")
        return action


@dataclass(frozen=True, slots=True)
class FixedDecompositionPolicy:
    """Decompose once, answer every child, synthesize, then submit."""

    def choose(
        self,
        state: State,
        legal_actions: tuple[Action, ...],
    ) -> Action:
        root_children = child_goals(state, 0)
        attempted_decompose = any(
            event.action == Action(Verb.DECOMPOSE, 0)
            for event in state.history
        )
        if not root_children and not attempted_decompose:
            action = Action(Verb.DECOMPOSE, 0)
        elif not root_children:
            root_candidates = goal_candidates(state, 0)
            action = (
                Action(Verb.ANSWER, 0)
                if not root_candidates
                else Action(Verb.SUBMIT, root_candidates[-1].id)
            )
        else:
            for child in root_children:
                if not goal_candidates(state, child.id):
                    action = Action(Verb.ANSWER, child.id)
                    break
            else:
                root_candidates = goal_candidates(state, 0)
                synthesized = [
                    candidate
                    for candidate in root_candidates
                    if _created_by_action(
                        state, candidate, Verb.SYNTHESIZE
                    )
                ]
                attempted_synthesis = any(
                    event.action == Action(Verb.SYNTHESIZE, 0)
                    for event in state.history
                )
                if synthesized:
                    action = Action(Verb.SUBMIT, synthesized[-1].id)
                elif not attempted_synthesis:
                    action = Action(Verb.SYNTHESIZE, 0)
                else:
                    root_answers = [
                        candidate
                        for candidate in root_candidates
                        if _created_by_action(state, candidate, Verb.ANSWER)
                    ]
                    action = (
                        Action(Verb.ANSWER, 0)
                        if not root_answers
                        else Action(Verb.SUBMIT, root_answers[-1].id)
                    )
        if action not in legal_actions:
            raise ValueError("fixed decomposition policy action is not legal")
        return action


def _created_by_action(
    state: State,
    candidate: CandidateNode,
    verb: Verb,
) -> bool:
    return any(
        event.action == Action(verb, candidate.goal_id)
        and candidate.id in event.created_ids
        and event.outcome is EventOutcome.APPLIED
        for event in state.history
    )


@dataclass(frozen=True, slots=True)
class Episode:
    initial_state: State
    transitions: tuple[Transition, ...]

    @property
    def final_state(self) -> State:
        return (
            self.initial_state
            if not self.transitions
            else self.transitions[-1].next_state
        )

    @property
    def reward(self) -> int:
        return self.final_state.reward


@dataclass(frozen=True, slots=True)
class BaselineAnchoredEpisode:
    """One deterministic direct prefix followed by policy decisions."""

    prefix_transition: Transition
    policy_transitions: tuple[Transition, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.prefix_transition, Transition):
            raise TypeError("baseline prefix must be a Transition")
        if self.prefix_transition.action != Action(Verb.ANSWER, 0):
            raise ValueError("baseline prefix must be root ANSWER")
        if type(self.policy_transitions) is not tuple or any(
            not isinstance(transition, Transition)
            for transition in self.policy_transitions
        ):
            raise TypeError("policy transitions must be a Transition tuple")
        previous = self.prefix_transition
        for transition in self.policy_transitions:
            if transition.state != previous.next_state:
                raise ValueError("baseline-anchored transition chain is broken")
            previous = transition

    @property
    def initial_state(self) -> State:
        return self.prefix_transition.state

    @property
    def controller_initial_state(self) -> State:
        return self.prefix_transition.next_state

    @property
    def transitions(self) -> tuple[Transition, ...]:
        return (self.prefix_transition,) + self.policy_transitions

    @property
    def final_state(self) -> State:
        return (
            self.prefix_transition.next_state
            if not self.policy_transitions
            else self.policy_transitions[-1].next_state
        )

    @property
    def reward(self) -> int:
        return self.final_state.reward


def run_policy_episode(
    environment: WorkspaceEnvironment,
    policy: WorkspacePolicy,
) -> Episode:
    if not isinstance(environment, WorkspaceEnvironment):
        raise TypeError("episode requires a WorkspaceEnvironment")
    if not isinstance(policy, WorkspacePolicy):
        raise TypeError("episode policy has an invalid type")
    initial = environment.initial_state()
    state = initial
    transitions: list[Transition] = []
    while not state.terminal:
        legal = environment.legal_actions(state)
        if not legal:
            raise RuntimeError("nonterminal workspace has no legal actions")
        action = policy.choose(state, legal)
        if action not in legal:
            raise ValueError("workspace policy returned an illegal action")
        transition = environment.step(state, action)
        transitions.append(transition)
        state = transition.next_state
    return Episode(initial, tuple(transitions))


def run_baseline_anchored_episode(
    environment: WorkspaceEnvironment,
    policy: WorkspacePolicy,
) -> BaselineAnchoredEpisode:
    """Run the exact direct call before the first policy-selected action."""
    if not isinstance(environment, WorkspaceEnvironment):
        raise TypeError("episode requires a WorkspaceEnvironment")
    if not isinstance(policy, WorkspacePolicy):
        raise TypeError("episode policy has an invalid type")
    initial = environment.initial_state()
    prefix = environment.step(initial, Action(Verb.ANSWER, 0))
    state = prefix.next_state
    transitions: list[Transition] = []
    while not state.terminal:
        legal = environment.legal_actions(state)
        if not legal:
            raise RuntimeError("nonterminal workspace has no legal actions")
        action = policy.choose(state, legal)
        if action not in legal:
            raise ValueError("workspace policy returned an illegal action")
        transition = environment.step(state, action)
        transitions.append(transition)
        state = transition.next_state
    return BaselineAnchoredEpisode(prefix, tuple(transitions))


__all__ = [
    "INITIALIZATION_DIRECT_CANDIDATE_PREFIX",
    "INITIALIZATION_EMPTY_ROOT",
    "INITIALIZATION_MODES",
    "BaselineAnchoredEpisode",
    "DirectPolicy",
    "Episode",
    "FixedDecompositionPolicy",
    "WorkspacePolicy",
    "run_policy_episode",
    "run_baseline_anchored_episode",
]
