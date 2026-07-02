"""The verb algebra: total pure functions Graph -> Result.

Slimmed per design.md: split attaches child subgoals directly to the
parent Goal and records the connective as non-gating metadata; commit
witnesses are Fact or Hypothesis only (the verifier-vouched-witness rule).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .graph import (
    Graph, NodeId, Goal, Hypothesis, Fact, VerifierCall,
    get, _append, _replace, holds, is_verified,
)
from .verify import Verifier

_CONNECTIVES = ("conjunctive", "exhaustive_case")


@dataclass(frozen=True)
class Ok:
    graph: Graph


@dataclass(frozen=True)
class Err:
    message: str


Result = Ok | Err


def _exists(graph: Graph, node_id: NodeId) -> bool:
    return 0 <= node_id < len(graph.nodes)


def split(graph: Graph, goal_id: NodeId, connective: str,
          child_statements: tuple[str, ...]) -> Result:
    if not _exists(graph, goal_id):
        return Err(f"no node with id {goal_id}")
    goal = get(graph, goal_id)
    if not isinstance(goal, Goal):
        return Err(f"node {goal_id} is not a Goal")
    if connective not in _CONNECTIVES:
        return Err(f"unknown connective {connective!r}")
    if not child_statements:
        return Err("split needs at least one child statement")
    if goal.children and goal.connective != connective:
        return Err(
            f"goal {goal_id} was already split with connective "
            f"{goal.connective!r}; refusing conflicting {connective!r}")

    g = graph
    child_ids: list[NodeId] = []
    for stmt in child_statements:
        g, cid = _append(g, lambda nid, s=stmt: Goal(
            id=nid, statement=s, assumptions=goal.assumptions))
        child_ids.append(cid)

    updated = replace(get(g, goal_id),
                      children=goal.children + tuple(child_ids),
                      connective=connective)
    return Ok(_replace(g, updated))


def form_hypothesis(graph: Graph, goal_id: NodeId, statement: str) -> Result:
    if not _exists(graph, goal_id):
        return Err(f"no node with id {goal_id}")
    goal = get(graph, goal_id)
    if not isinstance(goal, Goal):
        return Err(f"node {goal_id} is not a Goal")
    g, hyp_id = _append(graph, lambda nid: Hypothesis(id=nid, statement=statement))
    updated = replace(get(g, goal_id), candidates=goal.candidates + (hyp_id,))
    return Ok(_replace(g, updated))


def compute(graph: Graph, goal_id: NodeId, statement: str,
            premise_ids: tuple[NodeId, ...]) -> Result:
    if not _exists(graph, goal_id):
        return Err(f"no node with id {goal_id}")
    goal = get(graph, goal_id)
    if not isinstance(goal, Goal):
        return Err(f"node {goal_id} is not a Goal")
    for pid in premise_ids:
        if not _exists(graph, pid):
            return Err(f"no premise with id {pid}")
        if not isinstance(get(graph, pid), (Fact, Hypothesis)):
            return Err(f"premise {pid} is not a Fact or Hypothesis")
    g, fact_id = _append(graph, lambda nid: Fact(
        id=nid, statement=statement, source="derived", premises=tuple(premise_ids)))
    updated = replace(get(g, goal_id), facts=goal.facts + (fact_id,))
    return Ok(_replace(g, updated))


_STATEMENT_KINDS = (Goal, Hypothesis, Fact)


def verify(graph: Graph, node_id: NodeId, verb: str, expected: str | None,
           verifier: Verifier) -> Result:
    if not _exists(graph, node_id):
        return Err(f"no node with id {node_id}")
    node = get(graph, node_id)
    if not isinstance(node, _STATEMENT_KINDS):
        return Err(f"node {node_id} carries no statement to verify")
    result = verifier.check(node.statement, verb, expected)
    for n in graph.nodes:
        if (isinstance(n, VerifierCall) and n.target == node_id and n.verb == verb
                and n.expected == expected and n.success == result.success
                and n.observed == result.observed):
            return Ok(graph)  # idempotent
    g, _ = _append(graph, lambda nid: VerifierCall(
        id=nid, target=node_id, verb=verb, expected=expected,
        success=result.success, observed=result.observed))
    return Ok(g)


def commit(graph: Graph, goal_id: NodeId, witness_id: NodeId) -> Result:
    if not _exists(graph, goal_id) or not _exists(graph, witness_id):
        return Err("commit references a missing node")
    goal = get(graph, goal_id)
    if not isinstance(goal, Goal):
        return Err(f"node {goal_id} is not a Goal")
    w = get(graph, witness_id)
    ok = False
    if isinstance(w, Fact):
        ok = holds(graph, witness_id)   # v1: verifier-vouched truth, not statement relevance
    elif isinstance(w, Hypothesis):
        ok = witness_id in goal.candidates and is_verified(graph, witness_id)
    if not ok:
        return Err(f"witness {witness_id} does not discharge goal {goal_id}")
    pairs = tuple(p for p in graph.solved_by if p[0] != goal_id)
    pairs = tuple(sorted(pairs + ((goal_id, witness_id),)))
    return Ok(replace(graph, solved_by=pairs))


def abandon(graph: Graph, node_id: NodeId) -> Result:
    if not _exists(graph, node_id):
        return Err(f"no node with id {node_id}")
    return Ok(replace(graph, abandoned=graph.abandoned | {node_id}))
