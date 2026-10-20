"""Canonical legal action menu for the v2 reasoning contract."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from .model import (
    Action,
    Claim,
    ClaimStatus,
    Connective,
    Evidence,
    Goal,
    GoalObjective,
    GoalStatus,
    Graph,
    State,
    Verb,
    direct_child_goals,
    get,
    is_active,
    is_committable,
    validate_graph,
)


ACTION_SEMANTICS_VERSION = "claim-closure-v2"

VERB_ORDER: tuple[Verb, ...] = (
    Verb.DECOMPOSE,
    Verb.PROPOSE_RESULT,
    Verb.PROPOSE_SOLUTION,
    Verb.DERIVE,
    Verb.AGGREGATE,
    Verb.PROPOSE_REFUTATION,
    Verb.VERIFY,
    Verb.COMMIT,
    Verb.PRUNE,
)

POLICY_VERB_ORDER: tuple[Verb, ...] = (
    Verb.DECOMPOSE,
    Verb.PROPOSE_RESULT,
    Verb.PROPOSE_SOLUTION,
    Verb.PROPOSE_REFUTATION,
)

ACTION_VERB_ORDER: tuple[str, ...] = tuple(
    verb.value for verb in VERB_ORDER
)

ACTION_TARGET_KINDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        Verb.DECOMPOSE.value: ("goal",),
        Verb.PROPOSE_RESULT.value: ("goal",),
        Verb.PROPOSE_SOLUTION.value: ("goal",),
        Verb.DERIVE.value: ("goal",),
        Verb.AGGREGATE.value: ("goal",),
        Verb.PROPOSE_REFUTATION.value: ("goal",),
        Verb.VERIFY.value: ("claim",),
        Verb.COMMIT.value: ("claim",),
        Verb.PRUNE.value: ("goal", "claim"),
    }
)

_AGGREGATE_OBJECTIVES = frozenset(
    {
        GoalObjective.SUM,
        GoalObjective.MAX,
        GoalObjective.MIN,
        GoalObjective.IDENTITY,
    }
)


def aggregate_ready(graph: Graph, goal: Goal) -> bool:
    """Return whether closure can construct this Goal's Aggregate Claim."""
    if not isinstance(graph, Graph) or not isinstance(goal, Goal):
        raise TypeError("aggregate_ready needs a Graph and Goal")
    if goal.status is not GoalStatus.OPEN or not is_active(graph, goal.id):
        return False
    if goal.obligation.objective not in _AGGREGATE_OBJECTIVES:
        return False
    if goal.connective is not Connective.EXHAUSTIVE_CASE:
        return False
    children = direct_child_goals(graph, goal.id)
    return bool(children) and all(
        child.committed_claim is not None for child in children
    )


def _goal_allows(state: State, goal: Goal, verb: Verb) -> bool:
    if goal.status is not GoalStatus.OPEN or not is_active(
        state.graph, goal.id
    ):
        return False
    if verb is Verb.DECOMPOSE:
        return (
            goal.obligation.objective in _AGGREGATE_OBJECTIVES
            and not direct_child_goals(state.graph, goal.id)
        )
    if verb is Verb.PROPOSE_RESULT:
        return goal.obligation.objective in {
            GoalObjective.SUM,
            GoalObjective.MAX,
            GoalObjective.MIN,
        }
    if verb is Verb.PROPOSE_SOLUTION:
        return goal.obligation.objective in {
            GoalObjective.BRANCH,
            GoalObjective.IDENTITY,
        }
    if verb is Verb.PROPOSE_REFUTATION:
        return goal.obligation.objective is GoalObjective.BRANCH
    return False


def _claim_allows(state: State, claim: Claim, verb: Verb) -> bool:
    if verb is Verb.VERIFY:
        return (
            claim.status is ClaimStatus.UNCHECKED
            and is_active(state.graph, claim.id)
        )
    if verb is Verb.COMMIT:
        return is_committable(state.graph, claim.id)
    if verb is Verb.PRUNE:
        return not claim.pruned and is_active(state.graph, claim.id)
    return False


def requires_internal_closure(state: State) -> bool:
    """Return whether deterministic verifier closure must run before policy."""
    for node in state.graph.nodes:
        if isinstance(node, Claim) and is_active(state.graph, node.id):
            if node.status is ClaimStatus.UNCHECKED or is_committable(
                state.graph, node.id
            ):
                return True
        if isinstance(node, Goal) and aggregate_ready(state.graph, node):
            return True
    return False


def available_actions(state: State, turn_limit: int) -> tuple[Action, ...]:
    """Return all legal pairs in fixed verb order, then ascending target id."""
    if not isinstance(state, State):
        raise TypeError("available_actions needs a State")
    if type(turn_limit) is not int or turn_limit < 0:
        raise ValueError("turn_limit must be a nonnegative integer")
    validate_graph(state.graph)
    root = get(state.graph, state.graph.root_id)
    if not isinstance(root, Goal):
        raise ValueError("state root is not a Goal")
    if (
        state.turn >= turn_limit
        or root.status is GoalStatus.SOLVED
        or requires_internal_closure(state)
    ):
        return ()

    actions: list[Action] = []
    for verb in POLICY_VERB_ORDER:
        for node in state.graph.nodes:
            if isinstance(node, Goal) and _goal_allows(state, node, verb):
                actions.append(Action(verb, node.id))
            elif isinstance(node, Evidence):
                continue
    return tuple(actions)


__all__ = [
    "ACTION_SEMANTICS_VERSION",
    "ACTION_TARGET_KINDS",
    "ACTION_VERB_ORDER",
    "POLICY_VERB_ORDER",
    "VERB_ORDER",
    "aggregate_ready",
    "available_actions",
    "requires_internal_closure",
]
