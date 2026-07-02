"""Neutral fixed-format serialization. No prompts, no action menus."""

from __future__ import annotations

import re

from .graph import Goal, Hypothesis, Fact, VerifierCall, get
from .lens import Subgraph

_LINE = re.compile(r"\[([A-Z_]+) id=(\d+)\] (.*)")


def _detail(node) -> str:
    if isinstance(node, Goal):
        if node.connective is not None:
            return f"{node.statement} connective={node.connective}"
        return node.statement
    if isinstance(node, (Hypothesis, Fact)):
        return node.statement
    if isinstance(node, VerifierCall):
        return f"target={node.target} success={node.success}"
    return ""


def render(sub: Subgraph) -> str:
    lines = [f"focus={sub.focus}"]
    for nid in sub.node_ids:
        node = get(sub.graph, nid)
        kind = type(node).__name__.upper()
        lines.append(f"[{kind} id={nid}] {_detail(node)}")
    return "\n".join(lines)


def parse(text: str) -> list[tuple[str, int, str]]:
    rows: list[tuple[str, int, str]] = []
    for line in text.splitlines():
        m = _LINE.match(line)
        if m:
            rows.append((m.group(1), int(m.group(2)), m.group(3)))
    return rows
