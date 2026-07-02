"""The Deliberation MDP: (verb, target) actions over the reductio forest.

State is a reductio.Graph plus a turn counter. Three verbs call the frozen
LLM through a lens (SPLIT, FORM_HYPOTHESIS, COMPUTE); three never do
(VERIFY, COMMIT, ABANDON), which is what closes the self-verification hole:
the policy can neither author what gets checked nor what it is checked
against. All legality is computed in available_actions, once.

See docs/superpowers/specs/2026-07-02-deliberation-mdp-integration-design.md
and design.md sections 3-8.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .reductio import (
    Fact, Goal, Graph, Hypothesis, NodeId,
    get, holds, is_abandoned, is_verified, new_graph, solved,
)


class Verb(Enum):
    SPLIT = "split"
    FORM_HYPOTHESIS = "form_hypothesis"
    COMPUTE = "compute"
    VERIFY = "verify"
    COMMIT = "commit"
    ABANDON = "abandon"


@dataclass(frozen=True)
class DeliberationState:
    graph: Graph
    turn: int


@dataclass(frozen=True)
class DeliberationAction:
    verb: Verb
    target: NodeId


def _is_open_goal(graph: Graph, node_id: NodeId) -> bool:
    n = get(graph, node_id)
    return (isinstance(n, Goal)
            and not solved(graph, node_id)
            and not is_abandoned(graph, node_id))


def _witnesses(graph: Graph, goal: Goal) -> tuple[NodeId, ...]:
    """Discharging witnesses of a goal, ascending by id (most recent last)."""
    out = [wid for wid in goal.candidates if is_verified(graph, wid)]
    out += [wid for wid in goal.facts if holds(graph, wid)]
    return tuple(sorted(out))


def available_actions(state: DeliberationState,
                      depth_limit: int | None = None
                      ) -> tuple[DeliberationAction, ...]:
    """Every legal (verb, target) pair, computed in one place.

    With a depth_limit, an exhausted state offers nothing (the episode is
    terminal). Legality per verb follows spec section 3.
    """
    if depth_limit is not None and state.turn >= depth_limit:
        return ()
    g = state.graph
    acts: list[DeliberationAction] = []
    for node in g.nodes:
        nid = node.id
        if isinstance(node, Goal):
            if not _is_open_goal(g, nid):
                continue
            acts.append(DeliberationAction(Verb.SPLIT, nid))
            acts.append(DeliberationAction(Verb.FORM_HYPOTHESIS, nid))
            acts.append(DeliberationAction(Verb.COMPUTE, nid))
            if _witnesses(g, node):
                acts.append(DeliberationAction(Verb.COMMIT, nid))
            acts.append(DeliberationAction(Verb.ABANDON, nid))
        elif isinstance(node, Hypothesis):
            if is_abandoned(g, nid):
                continue
            if not is_verified(g, nid):
                acts.append(DeliberationAction(Verb.VERIFY, nid))
                acts.append(DeliberationAction(Verb.ABANDON, nid))
        elif isinstance(node, Fact):
            if node.source == "derived" and not is_verified(g, nid):
                acts.append(DeliberationAction(Verb.VERIFY, nid))
    return tuple(acts)
