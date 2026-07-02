"""The lens layer: scoped views, verb suffixes, schemas, sanitization.

Each LLM-calling verb renders a grounded-local view of the state (the
problem, the ancestor path, the target's subtree; siblings hidden; closed
subtrees compacted to statement plus verdict), appends a verb-specific
suffix (a natural-language field list plus a one-shot example, which small
models follow far more reliably than raw JSON Schema), and constrains the
generation with a JSON schema. Statements written back into the graph are
sanitized so LLM output cannot spoof serialized node lines.
"""

from __future__ import annotations

import re
from typing import Any

from .deliberation_mdp import DeliberationState, Verb
from .reductio import (
    Fact, Goal, Hypothesis, get, is_abandoned, scope, solved,
)
from .reductio.graph import Graph, NodeId, VerifierCall

_MARKER = re.compile(r"\[([A-Z_]+) id=(\d+)\]")

_LLM_VERBS = frozenset({Verb.SPLIT, Verb.FORM_HYPOTHESIS, Verb.COMPUTE})


def sanitize(text: str) -> str:
    """Neutralize serialized-marker injection in LLM-authored text.

    Newlines and runs of whitespace collapse to single spaces (the
    serializer's parse regex only matches at line starts), and any
    remaining marker-shaped substring has its brackets swapped for
    parentheses so it cannot match at all. Content is kept, not censored.
    """
    flat = " ".join(text.split())
    return _MARKER.sub(lambda m: "(" + m.group(0)[1:-1] + ")", flat)


def _is_closed_goal(graph: Graph, node_id: NodeId) -> bool:
    n = get(graph, node_id)
    return isinstance(n, Goal) and (
        solved(graph, node_id) or is_abandoned(graph, node_id))


def _detail(graph: Graph, node) -> str:
    if isinstance(node, Goal):
        if node.connective is not None:
            return f"{node.statement} connective={node.connective}"
        return node.statement
    if isinstance(node, (Hypothesis, Fact)):
        return node.statement
    if isinstance(node, VerifierCall):
        return f"target={node.target} success={node.success}"
    return ""


def _subtree_ids(graph: Graph, node_id: NodeId) -> set[NodeId]:
    acc: set[NodeId] = set()
    frontier = [node_id]
    while frontier:
        nid = frontier.pop()
        if nid in acc:
            continue
        acc.add(nid)
        n = get(graph, nid)
        if isinstance(n, Goal):
            frontier.extend(n.children + n.candidates + n.facts)
    return acc


def render_view(graph: Graph, target: NodeId) -> str:
    """The grounded-local view: problem, ancestor path, target subtree.

    Closed goals other than the target itself render as a single line,
    statement plus verdict, with their subtrees omitted (the compaction
    that replaces a SUMMARIZE verb; design.md section 4.3).
    """
    sub = scope(graph, target)
    closed = {nid for nid in sub.node_ids
              if nid != target and _is_closed_goal(graph, nid)}
    hidden: set[NodeId] = set()
    for cid in closed:
        hidden |= _subtree_ids(graph, cid) - {cid}
    hidden -= {target}
    lines = [f"Problem: {get(graph, graph.root_id).statement}",
             f"focus={target}"]
    for nid in sub.node_ids:
        if nid in hidden:
            continue
        node = get(graph, nid)
        kind = type(node).__name__.upper()
        line = f"[{kind} id={nid}] {_detail(graph, node)}"
        if nid in closed:
            verdict = "solved" if solved(graph, nid) else "abandoned"
            line += f" ({verdict})"
        lines.append(line)
    return "\n".join(lines)


def global_prefix(state: DeliberationState) -> str:
    """The whole-forest serialization used for the global state embedding."""
    return (f"State:\n{render_view(state.graph, state.graph.root_id)}"
            f"\n[TURN {state.turn}]")


def _live_premise_ids(graph: Graph) -> list[NodeId]:
    return [n.id for n in graph.nodes if isinstance(n, (Fact, Hypothesis))]


def suffix(verb: Verb, target: NodeId, state: DeliberationState) -> str:
    """The per-verb prompt tail: verb name, target line, field list, example."""
    if verb not in _LLM_VERBS:
        raise ValueError(f"{verb} takes no LLM generation")
    tgt = get(state.graph, target)
    fields = _fields_description(verb, state)
    return (
        f"\n\nVerb: {verb.name}\n"
        f"Target: [id={target}] {tgt.statement}\n\n"
        f"{fields}\n\n"
        f"Return only the JSON object."
    )


def _fields_description(verb: Verb, state: DeliberationState) -> str:
    if verb is Verb.SPLIT:
        return (
            "Produce a JSON object with these fields:\n"
            '  - "connective": "conjunctive" (solve all parts) or '
            '"exhaustive_case" (cases covering the target)\n'
            '  - "children": a list of 2 to 6 short statements, each a '
            "smaller sub-goal or case\n"
            "Example:\n"
            '  {"connective": "exhaustive_case", '
            '"children": ["solutions with x < 0", "solutions with x >= 0"]}'
        )
    if verb is Verb.FORM_HYPOTHESIS:
        return (
            "Produce a JSON object with this field:\n"
            '  - "statement": a short tentative claim addressing the target '
            "goal, as checkable math where possible\n"
            "Example:\n"
            '  {"statement": "x = 4"}'
        )
    if verb is Verb.COMPUTE:
        ids = _live_premise_ids(state.graph)
        opts = ", ".join(str(i) for i in ids) if ids else "(none yet)"
        return (
            "Produce a JSON object with these fields:\n"
            '  - "statement": one checkable computation step as an equation, '
            'e.g. "7*6 = 42" (pure math, no prose)\n'
            f'  - "premises": (optional) a list of ids among {{{opts}}} this '
            "step uses\n"
            "Example:\n"
            '  {"statement": "7*6 = 42", "premises": []}'
        )
    raise ValueError(f"{verb} takes no LLM generation")


def schema_for(verb: Verb, target: NodeId, state: DeliberationState) -> dict[str, Any]:
    """The constrained-generation schema for an LLM verb."""
    if verb is Verb.SPLIT:
        return {
            "type": "object",
            "properties": {
                "connective": {"type": "string",
                               "enum": ["conjunctive", "exhaustive_case"]},
                "children": {"type": "array", "minItems": 1, "maxItems": 6,
                             "items": {"type": "string"}},
            },
            "required": ["connective", "children"],
        }
    if verb is Verb.FORM_HYPOTHESIS:
        return {
            "type": "object",
            "properties": {"statement": {"type": "string"}},
            "required": ["statement"],
        }
    if verb is Verb.COMPUTE:
        ids = _live_premise_ids(state.graph)
        return {
            "type": "object",
            "properties": {
                "statement": {"type": "string"},
                "premises": {"type": "array",
                             "items": {"enum": ids} if ids else {"type": "integer"}},
            },
            "required": ["statement"],
        }
    raise ValueError(f"{verb} takes no LLM generation")
