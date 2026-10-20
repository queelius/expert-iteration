"""Immutable state and action algebra for the permissive workspace."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias


STATE_SCHEMA_VERSION = "permissive-reasoning-workspace-v1"
ACTION_SCHEMA_VERSION = "workspace-action-v1"
TRANSITION_SCHEMA_VERSION = "workspace-transition-v1"
REWARD_CONTRACT_VERSION = "symbolic-terminal-only-v1"
ACTION_AVAILABILITY_ALL = "all-affordances-v1"
ACTION_AVAILABILITY_ACTIVE_FRONTIER = "active-frontier-v1"
ACTION_AVAILABILITY_MODES = frozenset({
    ACTION_AVAILABILITY_ALL,
    ACTION_AVAILABILITY_ACTIVE_FRONTIER,
})


class Verb(str, Enum):
    DECOMPOSE = "decompose"
    ANSWER = "answer"
    SYNTHESIZE = "synthesize"
    REVISE = "revise"
    SUBMIT = "submit"


VERB_ORDER = (
    Verb.DECOMPOSE,
    Verb.ANSWER,
    Verb.SYNTHESIZE,
    Verb.REVISE,
    Verb.SUBMIT,
)
_VERB_RANK = {verb: index for index, verb in enumerate(VERB_ORDER)}


@dataclass(frozen=True, slots=True)
class Action:
    verb: Verb
    target: int

    def __post_init__(self) -> None:
        if not isinstance(self.verb, Verb):
            raise ValueError("action verb has an invalid type")
        if type(self.target) is not int or self.target < 0:
            raise ValueError("action target must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class GoalNode:
    id: int
    parent_goal: int | None
    task: str
    created_turn: int

    def __post_init__(self) -> None:
        if type(self.id) is not int or self.id < 0:
            raise ValueError("goal id must be a nonnegative integer")
        if self.parent_goal is not None and (
            type(self.parent_goal) is not int or self.parent_goal < 0
        ):
            raise ValueError("goal parent must be null or a node id")
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("goal task must be nonempty text")
        if type(self.created_turn) is not int or self.created_turn < 0:
            raise ValueError("goal created_turn must be nonnegative")


@dataclass(frozen=True, slots=True)
class CandidateNode:
    id: int
    goal_id: int
    reasoning: str
    answer: str
    created_turn: int
    revises: int | None = None

    def __post_init__(self) -> None:
        for name, value in (("id", self.id), ("goal_id", self.goal_id)):
            if type(value) is not int or value < 0:
                raise ValueError(f"candidate {name} must be a node id")
        if not isinstance(self.reasoning, str):
            raise ValueError("candidate reasoning must be text")
        if not isinstance(self.answer, str):
            raise ValueError("candidate answer must be text")
        if type(self.created_turn) is not int or self.created_turn < 0:
            raise ValueError("candidate created_turn must be nonnegative")
        if self.revises is not None and (
            type(self.revises) is not int or self.revises < 0
        ):
            raise ValueError("candidate revises must be null or a node id")


Node: TypeAlias = GoalNode | CandidateNode


class EventOutcome(str, Enum):
    APPLIED = "applied"
    GENERATION_ERROR = "generation_error"
    SUBMIT_ACCEPTED = "submit_accepted"
    SUBMIT_REJECTED = "submit_rejected"


@dataclass(frozen=True, slots=True)
class ActionEvent:
    turn: int
    action: Action
    outcome: EventOutcome
    created_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.turn) is not int or self.turn < 0:
            raise ValueError("event turn must be nonnegative")
        if not isinstance(self.action, Action):
            raise ValueError("event action has an invalid type")
        if not isinstance(self.outcome, EventOutcome):
            raise ValueError("event outcome has an invalid type")
        if type(self.created_ids) is not tuple or any(
            type(node_id) is not int or node_id < 0
            for node_id in self.created_ids
        ):
            raise ValueError("event created_ids must be node ids")
        if len(set(self.created_ids)) != len(self.created_ids):
            raise ValueError("event created_ids contain duplicates")
        if self.outcome is EventOutcome.APPLIED:
            if self.action.verb is Verb.SUBMIT or not self.created_ids:
                raise ValueError("applied generation must create nodes")
        elif self.created_ids:
            raise ValueError("non-applied events cannot create nodes")
        if self.action.verb is Verb.SUBMIT and self.outcome not in {
            EventOutcome.SUBMIT_ACCEPTED,
            EventOutcome.SUBMIT_REJECTED,
        }:
            raise ValueError("SUBMIT requires a symbolic submit outcome")
        if self.action.verb is not Verb.SUBMIT and self.outcome in {
            EventOutcome.SUBMIT_ACCEPTED,
            EventOutcome.SUBMIT_REJECTED,
        }:
            raise ValueError("only SUBMIT can carry a symbolic outcome")


@dataclass(frozen=True, slots=True)
class State:
    nodes: tuple[Node, ...]
    history: tuple[ActionEvent, ...]
    turn: int
    turn_limit: int
    terminal: bool = False
    reward: int = 0

    def __post_init__(self) -> None:
        if type(self.nodes) is not tuple or not self.nodes:
            raise ValueError("state nodes must be a nonempty tuple")
        if type(self.history) is not tuple:
            raise ValueError("state history must be a tuple")
        if type(self.turn) is not int or self.turn < 0:
            raise ValueError("state turn must be nonnegative")
        if type(self.turn_limit) is not int or self.turn_limit <= 0:
            raise ValueError("state turn_limit must be positive")
        if self.turn > self.turn_limit:
            raise ValueError("state turn exceeds its limit")
        if len(self.history) != self.turn:
            raise ValueError("state history must contain every learned turn")
        if tuple(event.turn for event in self.history) != tuple(range(self.turn)):
            raise ValueError("state event turns must be contiguous")
        if type(self.terminal) is not bool:
            raise ValueError("state terminal must be boolean")
        if type(self.reward) is not int or self.reward not in {0, 1}:
            raise ValueError("state reward must be symbolic binary reward")

        if tuple(node.id for node in self.nodes) != tuple(range(len(self.nodes))):
            raise ValueError("state node ids must be dense and ordered")
        root = self.nodes[0]
        if not isinstance(root, GoalNode) or root.parent_goal is not None:
            raise ValueError("state node zero must be the root Goal")
        if root.created_turn != 0:
            raise ValueError("root Goal must exist before the first turn")

        goals: dict[int, GoalNode] = {}
        candidates: dict[int, CandidateNode] = {}
        for node in self.nodes:
            if node.created_turn > self.turn:
                raise ValueError("node was created after the current turn")
            if isinstance(node, GoalNode):
                if node.parent_goal is not None:
                    parent = goals.get(node.parent_goal)
                    if parent is None or parent.id >= node.id:
                        raise ValueError("goal parent must be an earlier Goal")
                goals[node.id] = node
            elif isinstance(node, CandidateNode):
                owner = goals.get(node.goal_id)
                if owner is None or owner.id >= node.id:
                    raise ValueError("candidate owner must be an earlier Goal")
                if node.revises is not None:
                    prior = candidates.get(node.revises)
                    if prior is None or prior.goal_id != node.goal_id:
                        raise ValueError("candidate revision must share its Goal")
                candidates[node.id] = node
            else:
                raise ValueError("state contains an unsupported node type")

        created_by_event = tuple(
            node_id
            for event in self.history
            for node_id in event.created_ids
        )
        if created_by_event != tuple(range(1, len(self.nodes))):
            raise ValueError("state history must account for every generated node")
        for event in self.history:
            for node_id in event.created_ids:
                if self.nodes[node_id].created_turn != event.turn:
                    raise ValueError("node created_turn differs from its event")

        submitted = bool(
            self.history and self.history[-1].action.verb is Verb.SUBMIT
        )
        horizon = self.turn == self.turn_limit
        if self.terminal != (submitted or horizon):
            raise ValueError("state terminal flag differs from submit or horizon")
        accepted = bool(
            submitted
            and self.history[-1].outcome is EventOutcome.SUBMIT_ACCEPTED
        )
        if self.reward != int(accepted):
            raise ValueError("state reward differs from symbolic submit outcome")


def initial_state(problem: str, *, turn_limit: int) -> State:
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("problem must be nonempty text")
    return State(
        nodes=(GoalNode(0, None, problem, 0),),
        history=(),
        turn=0,
        turn_limit=turn_limit,
    )


def node_at(state: State, node_id: int) -> Node:
    if not isinstance(state, State):
        raise TypeError("node lookup requires a State")
    if type(node_id) is not int or not 0 <= node_id < len(state.nodes):
        raise KeyError(f"no node with id {node_id!r}")
    return state.nodes[node_id]


def child_goals(state: State, goal_id: int) -> tuple[GoalNode, ...]:
    target = node_at(state, goal_id)
    if not isinstance(target, GoalNode):
        raise TypeError("child lookup target must be a Goal")
    return tuple(
        node
        for node in state.nodes
        if isinstance(node, GoalNode) and node.parent_goal == goal_id
    )


def goal_candidates(state: State, goal_id: int) -> tuple[CandidateNode, ...]:
    target = node_at(state, goal_id)
    if not isinstance(target, GoalNode):
        raise TypeError("candidate lookup target must be a Goal")
    return tuple(
        node
        for node in state.nodes
        if isinstance(node, CandidateNode) and node.goal_id == goal_id
    )


def goal_depth(state: State, goal_id: int) -> int:
    target = node_at(state, goal_id)
    if not isinstance(target, GoalNode):
        raise TypeError("depth target must be a Goal")
    depth = 0
    cursor = target
    while cursor.parent_goal is not None:
        parent = node_at(state, cursor.parent_goal)
        if not isinstance(parent, GoalNode):
            raise AssertionError("validated goal ancestry changed type")
        cursor = parent
        depth += 1
    return depth


def available_actions(state: State, *, max_depth: int) -> tuple[Action, ...]:
    if not isinstance(state, State):
        raise TypeError("available_actions requires a State")
    if type(max_depth) is not int or max_depth < 1:
        raise ValueError("max_depth must be a positive integer")
    if state.terminal:
        return ()

    actions: list[Action] = []
    for node in state.nodes:
        if isinstance(node, GoalNode):
            children = child_goals(state, node.id)
            if not children and goal_depth(state, node.id) < max_depth:
                actions.append(Action(Verb.DECOMPOSE, node.id))
            actions.append(Action(Verb.ANSWER, node.id))
            if children and all(goal_candidates(state, child.id) for child in children):
                actions.append(Action(Verb.SYNTHESIZE, node.id))
        elif isinstance(node, CandidateNode):
            actions.append(Action(Verb.REVISE, node.id))
            if node.goal_id == 0:
                actions.append(Action(Verb.SUBMIT, node.id))
    return tuple(sorted(actions, key=lambda action: (_VERB_RANK[action.verb], action.target)))


def validate_action_availability_mode(mode: str) -> str:
    if mode not in ACTION_AVAILABILITY_MODES:
        raise ValueError("unknown action availability mode")
    return mode


def _creation_action(state: State, node_id: int) -> Action:
    for event in state.history:
        if node_id in event.created_ids:
            return event.action
    raise AssertionError("validated state has no creation event for node")


def protected_direct_candidate_id(state: State) -> int | None:
    """Return the Candidate created by an applied turn-zero root ANSWER."""
    if not state.history:
        return None
    event = state.history[0]
    if (
        event.turn != 0
        or event.action != Action(Verb.ANSWER, 0)
        or event.outcome is not EventOutcome.APPLIED
    ):
        return None
    candidates = [
        node.id
        for node in state.nodes
        if (
            isinstance(node, CandidateNode)
            and node.goal_id == 0
            and node.id in event.created_ids
        )
    ]
    if len(candidates) != 1:
        raise AssertionError("root ANSWER did not create exactly one Candidate")
    return candidates[0]


def _has_fresh_synthesis(
    state: State,
    goal: GoalNode,
    children: tuple[GoalNode, ...],
) -> bool:
    latest_child_turn = max(
        candidate.created_turn
        for child in children
        for candidate in goal_candidates(state, child.id)
    )
    return any(
        candidate.created_turn >= latest_child_turn
        and _creation_action(state, candidate.id).verb is Verb.SYNTHESIZE
        for candidate in goal_candidates(state, goal.id)
    )


def active_frontier_actions(
    state: State,
    *,
    max_depth: int,
) -> tuple[Action, ...]:
    """Return the bounded current frontier while preserving direct recovery."""
    if not isinstance(state, State):
        raise TypeError("active_frontier_actions requires a State")
    if type(max_depth) is not int or max_depth < 1:
        raise ValueError("max_depth must be a positive integer")
    if state.terminal:
        return ()

    goals = tuple(
        node for node in state.nodes if isinstance(node, GoalNode)
    )
    candidates_by_goal = {
        goal.id: goal_candidates(state, goal.id) for goal in goals
    }
    actions: list[Action] = []
    for goal in goals:
        children = child_goals(state, goal.id)
        candidates = candidates_by_goal[goal.id]
        if (
            not children
            and goal_depth(state, goal.id) < max_depth
            and (goal.id == 0 or not candidates)
        ):
            actions.append(Action(Verb.DECOMPOSE, goal.id))
        if not candidates:
            actions.append(Action(Verb.ANSWER, goal.id))
        if (
            children
            and all(candidates_by_goal[child.id] for child in children)
            and not _has_fresh_synthesis(state, goal, children)
        ):
            actions.append(Action(Verb.SYNTHESIZE, goal.id))

    for goal in goals:
        candidates = candidates_by_goal[goal.id]
        if not candidates:
            continue
        latest = candidates[-1]
        if not any(candidate.revises is not None for candidate in candidates):
            actions.append(Action(Verb.REVISE, latest.id))

    root_candidates = candidates_by_goal[0]
    if root_candidates:
        submit_ids = {root_candidates[-1].id}
        protected = protected_direct_candidate_id(state)
        if protected is not None:
            submit_ids.add(protected)
        actions.extend(Action(Verb.SUBMIT, node_id) for node_id in submit_ids)

    return tuple(sorted(
        actions,
        key=lambda action: (_VERB_RANK[action.verb], action.target),
    ))


def actions_for_mode(
    state: State,
    *,
    max_depth: int,
    mode: str = ACTION_AVAILABILITY_ALL,
) -> tuple[Action, ...]:
    validate_action_availability_mode(mode)
    if mode == ACTION_AVAILABILITY_ALL:
        return available_actions(state, max_depth=max_depth)
    return active_frontier_actions(state, max_depth=max_depth)


__all__ = [
    "ACTION_AVAILABILITY_ACTIVE_FRONTIER",
    "ACTION_AVAILABILITY_ALL",
    "ACTION_AVAILABILITY_MODES",
    "ACTION_SCHEMA_VERSION",
    "REWARD_CONTRACT_VERSION",
    "STATE_SCHEMA_VERSION",
    "TRANSITION_SCHEMA_VERSION",
    "Action",
    "ActionEvent",
    "CandidateNode",
    "EventOutcome",
    "GoalNode",
    "Node",
    "State",
    "VERB_ORDER",
    "Verb",
    "actions_for_mode",
    "active_frontier_actions",
    "available_actions",
    "child_goals",
    "goal_candidates",
    "goal_depth",
    "initial_state",
    "node_at",
    "protected_direct_candidate_id",
    "validate_action_availability_mode",
]
