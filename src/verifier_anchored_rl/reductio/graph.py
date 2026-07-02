"""The immutable reasoning forest.

Slimmed model (new-design.md section 3): no Method node layer. SPLIT
attaches child subgoals directly to the parent Goal; the connective is
optional, non-gating metadata. solved/verified/holds are derived
predicates, never control flow; the only hard gate in the system is the
verifier-vouched witness check in commit (actions.py).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

NodeId = int


@dataclass(frozen=True)
class Goal:
    id: NodeId
    statement: str
    assumptions: tuple[str, ...] = ()
    children: tuple[NodeId, ...] = ()     # subgoals attached by SPLIT
    candidates: tuple[NodeId, ...] = ()   # candidate hypotheses
    connective: str | None = None         # "conjunctive" | "exhaustive_case"; metadata only


@dataclass(frozen=True)
class Hypothesis:
    id: NodeId
    statement: str


@dataclass(frozen=True)
class Fact:
    id: NodeId
    statement: str
    source: str                            # "given" | "derived"
    premises: tuple[NodeId, ...] = ()


@dataclass(frozen=True)
class VerifierCall:
    id: NodeId
    target: NodeId
    verb: str
    expected: str | None
    success: bool
    observed: str


Node = Goal | Hypothesis | Fact | VerifierCall


@dataclass(frozen=True)
class Graph:
    nodes: tuple[Node, ...]
    root_id: NodeId
    abandoned: frozenset[NodeId] = frozenset()
    solved_by: tuple[tuple[NodeId, NodeId], ...] = ()


def get(graph: Graph, node_id: NodeId) -> Node:
    return graph.nodes[node_id]


def _append(graph: Graph, make_node: Callable[[NodeId], Node]) -> tuple[Graph, NodeId]:
    new_id = len(graph.nodes)
    node = make_node(new_id)
    assert node.id == new_id
    return replace(graph, nodes=graph.nodes + (node,)), new_id


def _replace(graph: Graph, node: Node) -> Graph:
    nodes = graph.nodes[: node.id] + (node,) + graph.nodes[node.id + 1 :]
    return replace(graph, nodes=nodes)


def new_graph(problem: str, givens: tuple[str, ...] = ()) -> Graph:
    root = Goal(id=0, statement=problem)
    nodes: tuple[Node, ...] = (root,)
    for i, stmt in enumerate(givens, start=1):
        nodes = nodes + (Fact(id=i, statement=stmt, source="given"),)
    return Graph(nodes=nodes, root_id=0)


def is_abandoned(graph: Graph, node_id: NodeId) -> bool:
    return node_id in graph.abandoned


def is_verified(graph: Graph, node_id: NodeId) -> bool:
    return any(
        isinstance(n, VerifierCall) and n.target == node_id and n.success
        for n in graph.nodes
    )


def holds(graph: Graph, node_id: NodeId) -> bool:
    n = get(graph, node_id)
    if isinstance(n, Fact):
        return n.source == "given" or is_verified(graph, node_id)
    if isinstance(n, Hypothesis):
        return is_verified(graph, node_id)
    return False


def _committed_witness(graph: Graph, goal_id: NodeId) -> NodeId | None:
    for g_id, w_id in graph.solved_by:
        if g_id == goal_id:
            return w_id
    return None


def solved(graph: Graph, node_id: NodeId) -> bool:
    if is_abandoned(graph, node_id):
        return False
    n = get(graph, node_id)
    if isinstance(n, (Fact, Hypothesis)):
        return holds(graph, node_id)
    if isinstance(n, Goal):
        w = _committed_witness(graph, node_id)
        if w is None:
            return False
        wn = get(graph, w)
        if isinstance(wn, Fact):
            return holds(graph, w)
        if isinstance(wn, Hypothesis):
            return w in n.candidates and is_verified(graph, w)
    return False
