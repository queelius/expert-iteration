"""Pure structural scoping. No prompts, no action menus."""

from __future__ import annotations

from dataclasses import dataclass

from .graph import Graph, NodeId, Goal, get


@dataclass(frozen=True)
class Subgraph:
    focus: NodeId
    node_ids: tuple[NodeId, ...]
    graph: Graph


def _children(graph: Graph, node_id: NodeId) -> tuple[NodeId, ...]:
    n = get(graph, node_id)
    if isinstance(n, Goal):
        return n.children + n.candidates + n.facts
    return ()


def _descendants(graph: Graph, node_id: NodeId, acc: set[NodeId]) -> None:
    acc.add(node_id)
    for c in _children(graph, node_id):
        if c not in acc:
            _descendants(graph, c, acc)


def _ancestors(graph: Graph, node_id: NodeId) -> set[NodeId]:
    # parent = the unique node whose children include node_id; walk to the root.
    found: set[NodeId] = set()
    cur = node_id
    while cur != graph.root_id:
        parent = next(
            (n.id for n in graph.nodes if cur in _children(graph, n.id)), None)
        if parent is None:
            break
        found.add(parent)
        cur = parent
    return found


def scope(graph: Graph, node_id: NodeId) -> Subgraph:
    acc: set[NodeId] = set()
    _descendants(graph, node_id, acc)
    acc |= _ancestors(graph, node_id)
    return Subgraph(focus=node_id, node_ids=tuple(sorted(acc)), graph=graph)
