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

from .deliberation_mdp import DeliberationState, Verb, _witnesses
from .reductio import (
    Fact, Goal, Hypothesis, get, is_abandoned, is_verified, scope, solved,
)
from .reductio.graph import Graph, NodeId, VerifierCall

# Matches both marker shapes the render emits ([KIND id=N] and
# [KIND id=N parent=M]) so sanitized LLM text can spoof neither.
_MARKER = re.compile(r"\[([A-Z_]+) id=(\d+)(?: parent=(\d+))?\]")

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
        out = node.statement
        if node.connective is not None:
            out += f" connective={node.connective}"
        # Commit-readiness: an open goal with a discharging witness shows
        # WHICH node a commit would use (the most recent verified evidence,
        # the same one _do_commit picks). Without this a controller must
        # guess whether commit is dischargeable and by what.
        if not solved(graph, node.id) and not is_abandoned(graph, node.id):
            ws = _witnesses(graph, node)
            if ws:
                out += f" (evidence: node {ws[-1]} verified)"
        return out
    if isinstance(node, (Hypothesis, Fact)):
        return f"{node.statement}{_verify_marker(graph, node.id)}"
    if isinstance(node, VerifierCall):
        return f"target={node.target} success={node.success}"
    return ""


def _verify_marker(graph: Graph, node_id: NodeId) -> str:
    """The node's verification status, rendered inline.

    Verified-versus-tentative is the design's one load-bearing distinction
    (design.md section 3.2), and a failed check must change the observation,
    or a deterministic controller sees an identical state after a failed
    VERIFY and repeats the same action forever. VerifierCall records are not
    reachable by the scope traversal, so their outcome is surfaced here, on
    the target's own line.
    """
    if is_verified(graph, node_id):
        return " (verified)"
    failures = sum(1 for n in graph.nodes
                   if isinstance(n, VerifierCall) and n.target == node_id
                   and not n.success)
    if failures:
        return f" (verify failed x{failures})"
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

    Rendered AS A TREE: children are indented under their parent and every
    non-root node carries a parent= tag, so the controller can see which
    goal owns which fact (a flat id-ordered list erases exactly the
    structure a commit decision needs; found via the casework probes).

    Closed goals other than the target itself render as a single line,
    statement plus verdict, with their subtrees omitted (the compaction
    that replaces a SUMMARIZE verb; design.md section 4.3). Recursion
    stops at a compacted goal, which is what hides its subtree.
    """
    sub = scope(graph, target)
    in_scope = set(sub.node_ids)
    compacted = {nid for nid in sub.node_ids
                 if nid != target
                 and _is_closed_goal(graph, nid)
                 and target not in _subtree_ids(graph, nid)}
    lines = [f"Problem: {get(graph, graph.root_id).statement}",
             f"focus={target}"]

    def emit(nid: NodeId, parent: NodeId | None, depth: int) -> None:
        if nid not in in_scope:
            return                       # hidden sibling of an ancestor
        node = get(graph, nid)
        kind = type(node).__name__.upper()
        tag = (f"[{kind} id={nid}]" if parent is None
               else f"[{kind} id={nid} parent={parent}]")
        line = "  " * depth + f"{tag} {_detail(graph, node)}"
        if _is_closed_goal(graph, nid) and nid != target:
            verdict = "solved" if solved(graph, nid) else "abandoned"
            line += f" ({verdict})"
        lines.append(line)
        if isinstance(node, Goal) and nid not in compacted:
            for child in node.children + node.candidates + node.facts:
                emit(child, nid, depth + 1)

    emit(graph.root_id, None, 0)
    return "\n".join(lines)


def global_prefix(state: DeliberationState,
                  depth_limit: int | None = None) -> str:
    """The whole-forest serialization used for the global state embedding.

    With a depth_limit the turn marker shows the budget ([TURN 3/16]);
    "finish before the limit" is unactionable when the horizon is hidden.
    """
    turn = (f"[TURN {state.turn}/{depth_limit}]" if depth_limit is not None
            else f"[TURN {state.turn}]")
    return (f"State:\n{render_view(state.graph, state.graph.root_id)}"
            f"\n{turn}")


def _live_premise_ids(graph: Graph) -> list[NodeId]:
    return [n.id for n in graph.nodes
            if isinstance(n, (Fact, Hypothesis)) and not is_abandoned(graph, n.id)]


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
            "goal, as checkable GROUND math where possible. The symbolic "
            "checker can only verify statements with no unknowns: a "
            'hypothesis like "x = 4" can never be verified and so can '
            "never serve as commit evidence.\n"
            "Example:\n"
            '  {"statement": "6*7 = 42"}'
        )
    if verb is Verb.COMPUTE:
        ids = _live_premise_ids(state.graph)
        opts = ", ".join(str(i) for i in ids) if ids else "(none yet)"
        return (
            "Produce a JSON object with these fields:\n"
            '  - "statement": ONE checkable step as a GROUND equation '
            "(numbers and operations only, NO variables or unknowns), with "
            "the value it establishes on the right-hand side. If the target "
            "goal is an equation to solve for a variable, SOLVE it and put "
            'the solution on the right: goal "4x - 16 = 4" becomes '
            '"(4 + 16)/4 = 5"; goal "2x + 5 = 13" becomes "(13 - 5)/2 = 4". '
            "To combine the results of separate cases at a parent goal, "
            'compute with their values: for a sum of solutions, "5 + 3 = 8". '
            'Good: "7*6 = 42", "(4 + 16)/4 = 5", "5 + 3 = 8". Bad (has an '
            'unknown, the check rejects it): "4x - 16 = 4", "2*x = 8".\n'
            f'  - "premises": (optional) a list of ids among {{{opts}}} this '
            "step uses\n"
            "Example:\n"
            '  {"statement": "(4 + 16)/4 = 5", "premises": []}'
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
                "children": {"type": "array", "minItems": 2, "maxItems": 6,
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
