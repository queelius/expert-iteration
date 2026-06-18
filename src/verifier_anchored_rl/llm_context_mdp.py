"""LLM context MDP: schema, action set, serializer, and protocol stub.

This module implements the **interfaces and serializer** for the
verifier-anchored context MDP described in design.md section 21 and in the
two methodology papers (paper/ and paper-recursive/). It deliberately does
**not** invoke any language model. Per design.md section 2.3 ("Phase 3:
LLM Context MDP Design Stub") and CLAUDE.md section 16 (non-goals), this
file is a stub: it fixes the data types and the serialization that a real
implementation would consume, so that downstream components (a frozen
encoder, a verb-instantiator, a verifier dispatcher) can be slotted in
later without re-architecting the surrounding machinery.

The module contains four artifacts:

1. ``Fact``, ``Subgoal``, ``Hypothesis``, ``VerifierCall``: frozen
   dataclasses for the four record types. Identifiers are stable strings;
   parent pointers reference identifiers, not nested objects, so the
   state stays hashable and the tree is implicit.

2. ``MathState``: the tuple
   ``(P, F, G_open, G_closed, H, V, c, A, d)`` from Definition 1 in
   paper/sections/method.tex. Frozen and hashable so that MCTS chance
   nodes can key on it.

3. ``MathAction``: the seven-verb action set
   (``DECOMPOSE``, ``COMPUTE_SUBEXPR``, ``FORM_HYPOTHESIS``,
   ``CHECK_CASE``, ``INVOKE_VERIFIER``, ``BACKTRACK``, ``COMMIT_ANSWER``)
   with an optional ``target_id`` field. The id-target field is what
   makes the action space cardinality-stable across arbitrary recursion
   depth: only the verb is categorical; the target is generated as a
   text continuation by the verb-instantiator (companion paper, section
   4.4).

4. ``serialize_state(MathState) -> str``: the flat token serialization
   with depth markers used in both papers
   (``[HYPOTHESIS id=hyp_2 parent=hyp_1 depth=2] {statement}
   [/HYPOTHESIS]``). This is the canonical bridge between the structured
   state and the frozen LLM's input. ``parse_state(str) -> MathState``
   inverts it for round-trip testing.

A ``LLMContextMDP`` skeleton implementing ``GenerativeMDP`` is included
at the bottom of the module. ``sample_transition`` raises
``NotImplementedError`` with a docstring describing the call sequence a
real implementation would perform: verb-instantiator -> LLM continuation
-> structured-record parse -> verifier dispatch (if the verb is
``INVOKE_VERIFIER``) -> next-state assembly -> reward attribution. The
skeleton is what lets a future implementer plug in an LLM client (and a
SymPy or rule-based verifier) without changing the rest of the package.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal, Sequence


# -----------------------------------------------------------------------------
# Record types (section 4.2 of paper/sections/method.tex)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Fact:
    """A verified or given fact.

    ``source`` distinguishes the three reasons a fact is in the state:
    ``"given"`` for problem-statement premises, ``"derived"`` for
    derivations the agent has produced (the framework paper's fact-vs-
    hypothesis distinction makes these distinct from unverified
    hypotheses), and ``"verified"`` for derivations promoted by a
    successful ``VerifierCall``. Promoting a hypothesis to a fact via
    verifier call is the operational meaning of ``INVOKE_VERIFIER``.
    """

    id: str
    statement: str
    source: Literal["given", "derived", "verified"]
    derivation: str | None = None


@dataclass(frozen=True)
class Subgoal:
    """An open or closed subgoal.

    ``parent_subgoal`` enables nested subgoals. ``children`` lists the
    identifiers of child subgoals or hypotheses spawned from this one;
    it is a tuple (immutable) of strings, not of objects, so the tree
    stays implicit and the state stays hashable.
    """

    id: str
    statement: str
    parent_subgoal: str | None = None
    children: tuple[str, ...] = ()


@dataclass(frozen=True)
class Hypothesis:
    """A tentative claim. May refine a subgoal or another hypothesis.

    The two parent fields encode the two ways a hypothesis can sit in
    the implicit tree: ``parent_subgoal`` for hypotheses that address an
    open subgoal directly; ``parent_hypothesis`` for sub-hypotheses
    introduced by ``DECOMPOSE`` or ``CHECK_CASE`` on a parent
    hypothesis. Companion paper Definition 2 computes depth as the
    length of the longest parent chain following either pointer.
    """

    id: str
    statement: str
    parent_subgoal: str | None = None
    parent_hypothesis: str | None = None
    children: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerifierCall:
    """A record of one ``INVOKE_VERIFIER`` call.

    ``verb`` is the verifier sub-verb (the symbolic checker's operation,
    e.g. ``"check_arith"``, ``"factor"``). ``success`` is the only field
    the verifier-anchored reward (paper section 4.6) consumes. The
    ``follow_up_to`` field links chained verifier calls (e.g. simplify
    then check).
    """

    id: str
    verb: Literal["check_arith", "solve", "simplify", "factor", "evaluate"]
    input: str
    output: str
    success: bool
    target_id: str | None = None
    follow_up_to: str | None = None


# -----------------------------------------------------------------------------
# State (Definition 1 in paper/sections/method.tex)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class MathState:
    """The tuple (P, F, G_open, G_closed, H, V, c, A, d).

    Fields:

    * ``problem`` (P): the immutable problem statement.
    * ``facts`` (F): tuple of ``Fact`` records.
    * ``open_subgoals`` (G_open) / ``closed_subgoals`` (G_closed): tuples
      of ``Subgoal`` records. The closed list is for tree-reuse signal,
      not for the agent's primary decision; closed subgoals are not
      actionable.
    * ``hypotheses`` (H): tuple of ``Hypothesis`` records (both
      productive and unproductive subtrees live here; the depth limit
      is enforced by ``available_actions`` and by MCTS root masks).
    * ``verifier_log`` (V): tuple of ``VerifierCall`` records.
    * ``committed_answer`` (c): the final answer text once
      ``COMMIT_ANSWER`` has fired, otherwise ``None``.
    * ``abandoned`` (A): tuple of identifiers explicitly abandoned by
      ``BACKTRACK``.
    * ``turn`` (d): the integer turn count.

    The Markov bet (paper section 4.2 final paragraph): all information
    relevant to future decisions lives in these slots. The text encoder
    for the frozen LLM (a downstream component) reads the result of
    ``serialize_state(self)``.
    """

    problem: str
    facts: tuple[Fact, ...] = ()
    open_subgoals: tuple[Subgoal, ...] = ()
    closed_subgoals: tuple[Subgoal, ...] = ()
    hypotheses: tuple[Hypothesis, ...] = ()
    verifier_log: tuple[VerifierCall, ...] = ()
    committed_answer: str | None = None
    abandoned: tuple[str, ...] = ()
    turn: int = 0

    def is_terminal(self) -> bool:
        """Terminal iff a committed answer is present.

        The terminal verifier fires on ``committed_answer`` to produce
        the verifier-anchored terminal reward (paper section 4.6,
        Equation rewardterminal).
        """
        return self.committed_answer is not None

    # -- identifier lookups -------------------------------------------------

    def fact_by_id(self, id_: str) -> Fact | None:
        return next((f for f in self.facts if f.id == id_), None)

    def subgoal_by_id(self, id_: str) -> Subgoal | None:
        for collection in (self.open_subgoals, self.closed_subgoals):
            hit = next((g for g in collection if g.id == id_), None)
            if hit is not None:
                return hit
        return None

    def hypothesis_by_id(self, id_: str) -> Hypothesis | None:
        return next((h for h in self.hypotheses if h.id == id_), None)

    def all_ids(self) -> set[str]:
        """Every identifier currently in the state.

        Used by ``MathAction.validate_target`` to check that a verb
        with a ``target_id`` is naming an existing entity.
        """
        ids: set[str] = set()
        ids.update(f.id for f in self.facts)
        ids.update(g.id for g in self.open_subgoals)
        ids.update(g.id for g in self.closed_subgoals)
        ids.update(h.id for h in self.hypotheses)
        ids.update(v.id for v in self.verifier_log)
        return ids

    # -- depth (Definition 2 of companion paper) ---------------------------

    def hypothesis_depth(self, hyp_id: str) -> int:
        """Length of the longest parent-pointer chain ending at ``hyp_id``.

        Implements companion paper Definition 2 restricted to the
        hypothesis-typed slice of the implicit tree. A hypothesis
        attached directly to a subgoal (no ``parent_hypothesis``) has
        depth 1 by the companion paper's convention; chains follow
        ``parent_hypothesis`` upward until the root. Cycles are not
        possible in a well-formed state but are guarded with a visited
        set defensively.
        """
        depth = 0
        visited: set[str] = set()
        current = self.hypothesis_by_id(hyp_id)
        while current is not None and current.id not in visited:
            visited.add(current.id)
            depth += 1
            if current.parent_hypothesis is None:
                return depth
            current = self.hypothesis_by_id(current.parent_hypothesis)
        return depth


# -----------------------------------------------------------------------------
# Action set (Definition 2 in paper/sections/method.tex)
# -----------------------------------------------------------------------------


class Verb(Enum):
    """The seven categorical verbs.

    Order is fixed: this is the categorical output set of the
    ``Q``-head and the policy head, and the MCTS chance-node index. Do
    not reorder.
    """

    DECOMPOSE = "DECOMPOSE"
    COMPUTE_SUBEXPR = "COMPUTE_SUBEXPR"
    FORM_HYPOTHESIS = "FORM_HYPOTHESIS"
    CHECK_CASE = "CHECK_CASE"
    INVOKE_VERIFIER = "INVOKE_VERIFIER"
    BACKTRACK = "BACKTRACK"
    COMMIT_ANSWER = "COMMIT_ANSWER"


@dataclass(frozen=True)
class MathAction:
    """A verb plus optional target identifier.

    The verb is the categorical dimension. The ``target_id`` is the
    identifier-targeted refinement (e.g. ``DECOMPOSE(hyp_3)`` says
    "decompose hypothesis hyp_3"). For verbs that do not refer to an
    existing entity (``COMPUTE_SUBEXPR``, ``COMMIT_ANSWER``,
    ``FORM_HYPOTHESIS`` when introducing a top-level claim), the target
    is ``None``.

    Companion paper section 4.4: this design is what keeps the
    categorical action space at cardinality seven regardless of how
    deep the implicit tree grows. The id is supplied by the
    verb-instantiator as part of its LLM-generated continuation.

    ``content`` is the free-form continuation the verb-instantiator
    produced (e.g. the text of a new hypothesis). It is not part of
    the categorical action; it is the content that the verb adds to
    the state on transition. Including it on the action object keeps
    the next-state assembly local to ``LLMContextMDP.sample_transition``.
    """

    verb: Verb
    target_id: str | None = None
    content: str | None = None

    def validate_target(self, state: MathState) -> bool:
        """Check the ``target_id`` (if any) refers to an extant entity.

        Returns ``True`` for actions with ``target_id is None`` and for
        actions whose ``target_id`` exists in ``state.all_ids()``.
        Returns ``False`` otherwise. The verb-instantiator failure mode
        described in companion paper section 4.4 (invalid id from the
        LLM continuation) corresponds to a ``False`` here.
        """
        if self.target_id is None:
            return True
        return self.target_id in state.all_ids()


def available_actions(
    state: MathState,
    *,
    depth_limit: int | None = 3,
) -> Sequence[Verb]:
    """The availability predicate from method.tex Table 1.

    Returns the subset of verbs whose precondition is satisfied in
    ``state``. The depth-limit mask (companion paper Definition 2 and
    section 4.3) is applied to ``FORM_HYPOTHESIS``, ``DECOMPOSE``, and
    ``CHECK_CASE`` when ``depth_limit`` is not ``None``: the verbs
    become unavailable on any state whose deepest hypothesis has
    reached ``depth_limit``. This is the hard mask the companion paper
    argues for (versus a soft penalty).

    The framework paper section 4.3 keeps ``FORM_HYPOTHESIS``
    unconditionally available; ``depth_limit=None`` recovers that
    behavior.
    """
    available: list[Verb] = []

    has_open_subgoal = len(state.open_subgoals) > 0
    has_hypothesis = len(state.hypotheses) > 0
    has_answer_bearing_claim = (
        any(h.parent_hypothesis is None for h in state.hypotheses)
        or any(f.source in {"derived", "verified"} for f in state.facts)
    )

    max_hyp_depth = (
        max((state.hypothesis_depth(h.id) for h in state.hypotheses), default=0)
    )
    depth_ok = depth_limit is None or max_hyp_depth < depth_limit

    if (has_open_subgoal or has_hypothesis) and depth_ok:
        available.append(Verb.DECOMPOSE)
    available.append(Verb.COMPUTE_SUBEXPR)
    if depth_ok:
        available.append(Verb.FORM_HYPOTHESIS)
    if has_hypothesis and depth_ok:
        available.append(Verb.CHECK_CASE)
    if has_hypothesis or any(
        f.source in {"derived", "verified"} for f in state.facts
    ):
        available.append(Verb.INVOKE_VERIFIER)
    if has_hypothesis or has_open_subgoal:
        available.append(Verb.BACKTRACK)
    if has_answer_bearing_claim:
        available.append(Verb.COMMIT_ANSWER)

    return tuple(available)


# -----------------------------------------------------------------------------
# Serialization (paper-recursive/sections/method.tex section 4.5 template)
# -----------------------------------------------------------------------------


_INT_RE = re.compile(r"\d+")


def _subgoal_depth(state: MathState, sg: Subgoal) -> int:
    depth = 0
    visited: set[str] = set()
    current: Subgoal | None = sg
    while current is not None and current.id not in visited:
        visited.add(current.id)
        if current.parent_subgoal is None:
            return depth
        depth += 1
        current = state.subgoal_by_id(current.parent_subgoal)
    return depth


def serialize_state(state: MathState) -> str:
    """Render the state in the canonical flat-token template.

    The template comes from paper-recursive/sections/method.tex
    section 4.5: ``[PROBLEM]``, ``[FACT]``, ``[OPEN_SUBGOAL]``,
    ``[HYPOTHESIS]``, ``[VERIFIER_CALL]``, ``[COMMITTED_ANSWER]``, and
    ``[DEPTH]`` markers around content, with ``id=``, ``parent=``, and
    ``depth=`` fields on entities that participate in the implicit
    tree. The ``parent=none`` literal indicates a root.

    The serialization is the canonical bridge between ``MathState`` and
    the frozen LLM's input. Two trajectories that reach the same
    structured state produce the same serialization, modulo trajectory
    order of introduction (the canonicalization deferred in paper
    section 4.2). The serializer is deterministic in the dataclass
    insertion order; ``parse_state(serialize_state(s)) == s`` is the
    round-trip invariant the tests check.
    """
    lines: list[str] = []
    lines.append(f"[PROBLEM] {state.problem} [/PROBLEM]")

    for f in state.facts:
        deriv = "" if f.derivation is None else f" derivation={_quote(f.derivation)}"
        lines.append(
            f"[FACT id={f.id} source={f.source}{deriv}] {f.statement} [/FACT]"
        )

    for label, collection in (
        ("OPEN_SUBGOAL", state.open_subgoals),
        ("CLOSED_SUBGOAL", state.closed_subgoals),
    ):
        for sg in collection:
            parent = sg.parent_subgoal if sg.parent_subgoal is not None else "none"
            depth = _subgoal_depth(state, sg)
            lines.append(
                f"[{label} id={sg.id} parent={parent} depth={depth}]"
                f" {sg.statement} [/{label}]"
            )

    for h in state.hypotheses:
        parent = (
            h.parent_hypothesis
            if h.parent_hypothesis is not None
            else (h.parent_subgoal if h.parent_subgoal is not None else "none")
        )
        depth = state.hypothesis_depth(h.id)
        lines.append(
            f"[HYPOTHESIS id={h.id} parent={parent} depth={depth}]"
            f" {h.statement} [/HYPOTHESIS]"
        )

    for v in state.verifier_log:
        target = "none" if v.target_id is None else v.target_id
        follow = (
            ""
            if v.follow_up_to is None
            else f" follow_up_to={v.follow_up_to}"
        )
        lines.append(
            f"[VERIFIER_CALL id={v.id} target={target} verb={v.verb}"
            f" success={'true' if v.success else 'false'}{follow}]"
        )
        lines.append(f"   in:  {v.input}")
        lines.append(f"   out: {v.output}")
        lines.append("[/VERIFIER_CALL]")

    answer = state.committed_answer if state.committed_answer is not None else "<none>"
    lines.append(f"[COMMITTED_ANSWER] {answer} [/COMMITTED_ANSWER]")
    lines.append(f"[ABANDONED] {' '.join(state.abandoned) if state.abandoned else '<none>'} [/ABANDONED]")
    lines.append(f"[DEPTH] {state.turn} [/DEPTH]")
    lines.append("[STATE_END]")
    return "\n".join(lines)


def _quote(s: str) -> str:
    return '"' + s.replace('"', r"\"") + '"'


def _unquote(s: str) -> str:
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1].replace(r"\"", '"')
    return s


# Token-block patterns: every block opens with ``[TAG ...]`` and closes
# with ``[/TAG]``. Verifier-call blocks span multiple lines.

_TAG_OPEN = re.compile(r"^\[(?P<tag>[A-Z_]+)(?P<attrs>(?: [^\]]*)?)\]\s*(?P<body>.*?)\s*$")
_TAG_CLOSE = re.compile(r"\s*\[/(?P<tag>[A-Z_]+)\]\s*$")
_ATTR = re.compile(r"(\w+)=(\"(?:[^\"\\]|\\.)*\"|\S+)")


def parse_state(text: str) -> MathState:
    """Inverse of ``serialize_state`` for the single-line blocks.

    Verifier-call blocks span three rendered lines (open, in/out
    indented, close); other blocks are single-line. The parser walks
    the text line by line and dispatches by tag. The round-trip
    invariant tested in ``tests/test_llm_context_mdp.py`` is
    ``parse_state(serialize_state(s)) == s``.

    Parser errors raise ``ValueError`` with a line number. A real
    implementation hooked to an LLM would catch these and recover via
    the retry-with-correction loop described in companion paper
    section 4.4.
    """
    problem: str = ""
    facts: list[Fact] = []
    open_sub: list[Subgoal] = []
    closed_sub: list[Subgoal] = []
    hyps: list[Hypothesis] = []
    vcalls: list[VerifierCall] = []
    committed: str | None = None
    abandoned: tuple[str, ...] = ()
    turn = 0

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        if line.strip() == "[STATE_END]":
            break

        m_open = _TAG_OPEN.match(line)
        if m_open is None:
            raise ValueError(f"line {i}: cannot parse: {line!r}")

        tag = m_open.group("tag")
        attrs = _parse_attrs(m_open.group("attrs"))
        body_with_close = m_open.group("body")

        if tag == "VERIFIER_CALL":
            in_line = lines[i + 1].strip()
            out_line = lines[i + 2].strip()
            close_line = lines[i + 3].strip()
            if close_line != "[/VERIFIER_CALL]":
                raise ValueError(
                    f"line {i + 3}: expected [/VERIFIER_CALL], got {close_line!r}"
                )
            vcalls.append(
                VerifierCall(
                    id=attrs["id"],
                    verb=attrs["verb"],  # type: ignore[arg-type]
                    input=in_line.removeprefix("in:").strip(),
                    output=out_line.removeprefix("out:").strip(),
                    success=(attrs["success"] == "true"),
                    target_id=None if attrs["target"] == "none" else attrs["target"],
                    follow_up_to=attrs.get("follow_up_to"),
                )
            )
            i += 4
            continue

        m_close = _TAG_CLOSE.search(body_with_close)
        if m_close is None:
            raise ValueError(f"line {i}: no closing tag in {line!r}")
        body = body_with_close[: m_close.start()].rstrip()

        if tag == "PROBLEM":
            problem = body
        elif tag == "FACT":
            facts.append(
                Fact(
                    id=attrs["id"],
                    statement=body,
                    source=attrs["source"],  # type: ignore[arg-type]
                    derivation=_unquote(attrs["derivation"]) if "derivation" in attrs else None,
                )
            )
        elif tag in {"OPEN_SUBGOAL", "CLOSED_SUBGOAL"}:
            sg = Subgoal(
                id=attrs["id"],
                statement=body,
                parent_subgoal=None if attrs["parent"] == "none" else attrs["parent"],
            )
            (open_sub if tag == "OPEN_SUBGOAL" else closed_sub).append(sg)
        elif tag == "HYPOTHESIS":
            parent_field = attrs["parent"]
            parent_sg: str | None = None
            parent_hyp: str | None = None
            if parent_field.startswith("hyp_"):
                parent_hyp = parent_field
            elif parent_field == "none":
                parent_sg = None
            else:
                parent_sg = parent_field
            hyps.append(
                Hypothesis(
                    id=attrs["id"],
                    statement=body,
                    parent_subgoal=parent_sg,
                    parent_hypothesis=parent_hyp,
                )
            )
        elif tag == "COMMITTED_ANSWER":
            committed = None if body.strip() == "<none>" else body
        elif tag == "ABANDONED":
            abandoned = (
                () if body.strip() == "<none>" else tuple(body.split())
            )
        elif tag == "DEPTH":
            m = _INT_RE.search(body)
            if m is None:
                raise ValueError(f"line {i}: depth body not an int: {body!r}")
            turn = int(m.group())
        else:
            raise ValueError(f"line {i}: unknown tag {tag!r}")

        i += 1

    return MathState(
        problem=problem,
        facts=tuple(facts),
        open_subgoals=tuple(open_sub),
        closed_subgoals=tuple(closed_sub),
        hypotheses=tuple(hyps),
        verifier_log=tuple(vcalls),
        committed_answer=committed,
        abandoned=abandoned,
        turn=turn,
    )


def _parse_attrs(attr_str: str) -> dict[str, str]:
    if not attr_str:
        return {}
    out: dict[str, str] = {}
    for k, v in _ATTR.findall(attr_str):
        out[k] = v
    return out


# -----------------------------------------------------------------------------
# Per-verb prompt templates and JSON schemas
# -----------------------------------------------------------------------------

# The verb-instantiator is asked to produce a small JSON object whose
# shape depends on the chosen verb. Schemas are deliberately narrow:
# the only freedom is the target-identifier (when applicable) and the
# free-form content. Enumerating valid target ids in the prompt keeps
# the LLM's job concrete and the parse cheap.


_SYSTEM_PROMPT = (
    "You are the verb-instantiator for a verifier-anchored deliberation MDP. "
    "Given a serialized state and a chosen action verb, produce a single JSON "
    "object that specifies (a) the target identifier the verb operates on (when "
    "applicable), and (b) the free-form content the verb introduces. The state "
    "uses bracketed tags ([FACT], [OPEN_SUBGOAL], [HYPOTHESIS], [VERIFIER_CALL], "
    "etc.) and each entity carries a stable id (fact_*, sg_*, hyp_*, vcall_*). "
    "Target ids you produce must match an existing id of the appropriate type. "
    "Return JSON only; no surrounding prose or explanation."
)


def _ids_of_type(state: MathState, prefix: str) -> list[str]:
    return sorted(i for i in state.all_ids() if i.startswith(prefix))


def _schema_for(verb: Verb, state: MathState) -> dict[str, Any]:
    """JSON schema for the verb-instantiator response, parameterized by the state."""
    sg_ids = _ids_of_type(state, "sg_")
    hyp_ids = _ids_of_type(state, "hyp_")
    fact_ids = _ids_of_type(state, "fact_")
    decomposable = sg_ids + hyp_ids

    if verb is Verb.DECOMPOSE:
        return {
            "type": "object",
            "properties": {
                "target_id": {"type": "string", "enum": decomposable or [""]},
                "children": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 6,
                },
            },
            "required": ["target_id", "children"],
        }
    if verb is Verb.COMPUTE_SUBEXPR:
        return {
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
                "result": {"type": "string"},
                "derivation": {"type": "string"},
            },
            "required": ["expression", "result"],
        }
    if verb is Verb.FORM_HYPOTHESIS:
        return {
            "type": "object",
            "properties": {
                "parent_subgoal_id": {
                    "type": ["string", "null"],
                    "enum": [*sg_ids, None] if sg_ids else [None],
                },
                "statement": {"type": "string"},
            },
            "required": ["statement"],
        }
    if verb is Verb.CHECK_CASE:
        return {
            "type": "object",
            "properties": {
                "target_id": {"type": "string", "enum": hyp_ids or [""]},
                "cases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 6,
                },
            },
            "required": ["target_id", "cases"],
        }
    if verb is Verb.INVOKE_VERIFIER:
        return {
            "type": "object",
            "properties": {
                "target_id": {
                    "type": "string",
                    "enum": (hyp_ids + fact_ids) or [""],
                },
                "verifier_verb": {
                    "type": "string",
                    "enum": [
                        "check_arith",
                        "simplify",
                        "factor",
                        "solve",
                        "evaluate",
                    ],
                },
                "input_expression": {"type": "string"},
                "expected": {"type": ["string", "null"]},
            },
            "required": ["target_id", "verifier_verb", "input_expression"],
        }
    if verb is Verb.BACKTRACK:
        return {
            "type": "object",
            "properties": {
                "target_id": {
                    "type": "string",
                    "enum": (hyp_ids + sg_ids) or [""],
                },
                "reason": {"type": "string"},
            },
            "required": ["target_id"],
        }
    if verb is Verb.COMMIT_ANSWER:
        return {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
            },
            "required": ["answer"],
        }
    raise ValueError(f"unknown verb: {verb!r}")


def _state_prefix(state: MathState) -> str:
    """The verb-independent prefix of the verb-instantiator prompt: the
    serialized state, ending exactly at the ``[STATE_END]`` sentinel.

    This is the shared span that the benchmark architecture's single
    forward pass prefills once (method.tex Section 3.4): ``embed(s) =
    h_T`` is the hidden state at this span's final token (the sentinel),
    and the same prefill's KV cache seeds the verb continuation. Holding
    the id-block and verb fields in the per-verb suffix keeps ``h_T``
    literally at ``[STATE_END]`` and keeps the cached prefix the
    expensive, verb-independent part.
    """
    return f"State:\n{serialize_state(state)}"


def _verb_suffix(verb: Verb, state: MathState) -> str:
    """The per-verb suffix appended after the shared state prefix:
    addressable identifiers, the chosen verb, and its field description.

    ``_state_prefix(state) + _verb_suffix(verb, state)`` is byte-identical
    to the historical ``_prompt_for`` output, so the HTTP and mock paths
    are unchanged.
    """
    sg_ids = _ids_of_type(state, "sg_")
    hyp_ids = _ids_of_type(state, "hyp_")
    fact_ids = _ids_of_type(state, "fact_")

    id_lines = []
    if fact_ids:
        id_lines.append(f"Facts: {', '.join(fact_ids)}")
    if sg_ids:
        id_lines.append(f"Open subgoals: {', '.join(sg_ids)}")
    if hyp_ids:
        id_lines.append(f"Hypotheses: {', '.join(hyp_ids)}")
    id_block = "\n".join(id_lines) if id_lines else "(no addressable entities yet)"

    fields = _verb_fields_description(verb, state)

    return (
        f"\n\n"
        f"Existing identifiers:\n{id_block}\n\n"
        f"Verb: {verb.value}\n\n"
        f"{fields}\n\n"
        f"Return only the JSON object."
    )


def _prompt_for(verb: Verb, state: MathState) -> str:
    """User-message body for the verb-instantiator call.

    The prompt encodes the expected JSON shape as a natural-language
    field list with a one-shot example, rather than handing the model a
    JSON Schema dict. Small instruct-tuned models tend to reproduce
    JSON Schema structures verbatim (returning the schema's
    ``properties``/``required``/``type`` keys instead of values
    matching them); the field-list-plus-example form sidesteps this
    failure mode.

    Composed from ``_state_prefix`` (the shared, cacheable state span)
    and ``_verb_suffix`` (the per-verb tail) so the benchmark
    architecture's shared-forward encoder can prefill the prefix once;
    the concatenation here is byte-identical to the historical single
    f-string.
    """
    return _state_prefix(state) + _verb_suffix(verb, state)


def _verb_fields_description(verb: Verb, state: MathState) -> str:
    """Field list + one-shot example for a verb. Natural language, not JSON Schema."""
    sg_ids = _ids_of_type(state, "sg_")
    hyp_ids = _ids_of_type(state, "hyp_")
    fact_ids = _ids_of_type(state, "fact_")
    decomposable = sg_ids + hyp_ids
    invoke_targets = hyp_ids + fact_ids
    backtrack_targets = hyp_ids + sg_ids

    def _enum(opts: list[str]) -> str:
        return ", ".join(f'"{o}"' for o in opts) if opts else "(none currently available)"

    if verb is Verb.DECOMPOSE:
        return (
            "Produce a JSON object with these fields:\n"
            f'  - "target_id": one of {{{_enum(decomposable)}}}\n'
            '  - "children": a list of 2 to 6 short strings, each a smaller sub-claim or sub-goal\n'
            "Example:\n"
            '  {"target_id": "sg_root", "children": ["First sub-claim.", "Second sub-claim."]}'
        )
    if verb is Verb.COMPUTE_SUBEXPR:
        return (
            "Produce a JSON object with these fields:\n"
            '  - "expression": a math expression to evaluate (e.g. "7 * 6")\n'
            '  - "result": the value the expression evaluates to (e.g. "42")\n'
            '  - "derivation": (optional) one short sentence explaining how you computed it\n'
            "Example:\n"
            '  {"expression": "7 * 6", "result": "42", "derivation": "Direct multiplication."}'
        )
    if verb is Verb.FORM_HYPOTHESIS:
        opts = _enum(sg_ids) if sg_ids else '(use null)'
        return (
            "Produce a JSON object with these fields:\n"
            f'  - "parent_subgoal_id": one of {{{opts}}} or null\n'
            '  - "statement": a short tentative claim addressing the subgoal (or the problem)\n'
            "Example:\n"
            '  {"parent_subgoal_id": "sg_root", "statement": "The product equals 42."}'
        )
    if verb is Verb.CHECK_CASE:
        return (
            "Produce a JSON object with these fields:\n"
            f'  - "target_id": one of {{{_enum(hyp_ids)}}}\n'
            '  - "cases": a list of 2 to 6 short case statements that partition the target hypothesis\n'
            "Example:\n"
            '  {"target_id": "hyp_1", "cases": ["x > 0", "x = 0", "x < 0"]}'
        )
    if verb is Verb.INVOKE_VERIFIER:
        return (
            "Produce a JSON object with these fields:\n"
            f'  - "target_id": one of {{{_enum(invoke_targets)}}}\n'
            '  - "verifier_verb": which symbolic check to run. Pick based on the form of the claim:\n'
            '      * "check_arith"  for "A equals B" claims (numeric or algebraic)\n'
            '      * "evaluate"     for numeric expressions (sets expected=value, or null)\n'
            '      * "solve"        for equations of the form "lhs = rhs" in one variable\n'
            '      * "simplify"     for "A simplifies to B" claims\n'
            '      * "factor"       for "A factors as B" claims\n'
            '  - "input_expression": pure SymPy-parseable math. Examples: "7*6", "(x+1)**2",\n'
            '      "x**2 - 9", "2*x + 5 = 13". NEVER include English prose like\n'
            '      "Compute ...", "Show that ...", or the problem statement itself.\n'
            '  - "expected": the value or form you claim. SymPy-parseable string, or null\n'
            '      only when verifier_verb is "evaluate".\n'
            "Examples:\n"
            '  {"target_id": "hyp_1", "verifier_verb": "check_arith", "input_expression": "7*6", "expected": "42"}\n'
            '  {"target_id": "hyp_2", "verifier_verb": "solve", "input_expression": "2*x + 5 = 13", "expected": "4"}\n'
            '  {"target_id": "hyp_3", "verifier_verb": "factor", "input_expression": "x**2 - 9", "expected": "(x-3)*(x+3)"}\n'
            '  {"target_id": "hyp_4", "verifier_verb": "evaluate", "input_expression": "sqrt(16) + 1", "expected": "5"}'
        )
    if verb is Verb.BACKTRACK:
        return (
            "Produce a JSON object with these fields:\n"
            f'  - "target_id": one of {{{_enum(backtrack_targets)}}}\n'
            '  - "reason": one short sentence explaining why this branch is abandoned\n'
            "Example:\n"
            '  {"target_id": "hyp_1", "reason": "Leads to contradictory subclaims."}'
        )
    if verb is Verb.COMMIT_ANSWER:
        return (
            "Produce a JSON object with this field:\n"
            '  - "answer": the final answer as a short string (number, expression, or word)\n'
            "Example:\n"
            '  {"answer": "42"}'
        )
    raise ValueError(f"unknown verb: {verb!r}")


# -----------------------------------------------------------------------------
# Reward and termination conventions
# -----------------------------------------------------------------------------

# Per framework paper section 4.6 (verifier-anchored-reward-only):
# * Non-terminal, non-verifier verbs contribute zero reward.
# * INVOKE_VERIFIER yields a small shaped signal (+/-) so the policy
#   learns to spend verifier calls; the magnitude is small relative
#   to the terminal reward so it cannot dominate.
# * COMMIT_ANSWER yields the terminal verifier outcome (+/-1).
# * A turn-budget cap can convert a non-terminal episode into a
#   terminal "no commit" outcome with negative reward.

INTERMEDIATE_VERIFIER_REWARD = 0.05
TERMINAL_SUCCESS_REWARD = 1.0
TERMINAL_FAILURE_REWARD = -1.0
MASKED_ACTION_REWARD = -0.01

# Backward-compatible private aliases (existing call sites).
_INTERMEDIATE_VERIFIER_REWARD = INTERMEDIATE_VERIFIER_REWARD
_TERMINAL_SUCCESS_REWARD = TERMINAL_SUCCESS_REWARD
_TERMINAL_FAILURE_REWARD = TERMINAL_FAILURE_REWARD


# -----------------------------------------------------------------------------
# LLM context MDP (real implementation, GenerativeMDP[MathState, MathAction])
# -----------------------------------------------------------------------------


@dataclass
class LLMContextMDP:
    """Concrete ``GenerativeMDP[MathState, MathAction]`` over an LLM-driven context.

    ``sample_transition`` runs one full step of the verifier-anchored
    deliberation: it prompts the frozen LLM for a verb-instantiated
    continuation, parses that continuation into structured-record
    updates, dispatches the symbolic verifier when the action is
    ``INVOKE_VERIFIER``, assembles the next state, and attributes the
    reward (zero for non-verifier intermediate steps, small shaped
    signal for verifier calls, terminal ``+/-1`` for ``COMMIT_ANSWER``).

    Constructor parameters:

    * ``problem_statement``: the immutable text of the problem.
    * ``expected_answer``: the ground-truth label used by
      ``COMMIT_ANSWER`` to set the terminal reward. For test-time
      evaluation, the same MDP class is constructed per problem with
      the labeled answer; for inference at deployment, set this to
      ``None`` and ``COMMIT_ANSWER`` will return a non-terminal-but-
      committed state with zero reward (the agent committed without
      ground-truth feedback).
    * ``llm_client``: an ``LLMClient`` (the mock for tests, the
      OpenAI-compatible HTTP client for real runs).
    * ``depth_limit``: forwarded to ``available_actions``; defaults
      to 3 per companion paper section 4.3.
    * ``max_turns``: hard turn-count cap. Reaching the cap converts
      the episode to terminal with ``_TERMINAL_FAILURE_REWARD`` (no
      commit was produced in budget). Defaults to 40, chosen so
      that a problem typically commits in 5 to 20 turns under the
      framework's intended deliberation patterns and the cap is a
      safety net rather than a tight budget.
    """

    problem_statement: str
    expected_answer: str | None = None
    llm_client: Any = field(default=None, repr=False)
    depth_limit: int | None = 3
    max_turns: int = 40
    temperature: float = 0.8
    max_tokens: int = 384
    # Prompt assembly strategy for the verb-instantiator. "full" sends
    # the entire serialized state on every call (the smoke-test
    # baseline, and the simplest path). "focused" sends a verb-specific
    # focused view with a separate target-picker pass; planned as an
    # ablation arm. Set via constructor; not changed mid-trajectory.
    prompt_strategy: Literal["full", "focused"] = "full"
    # When True (default), ``sample_transition`` derives a per-call
    # sampling seed from the planner's rng and forwards it to the LLM
    # client, so collections at matched seeds are reproducible up to the
    # server's compute-level determinism. Set False for fully stochastic
    # collection (production deployment, or to measure the unseeded
    # baseline against the seeded one on the same server).
    seed_llm: bool = True

    def initial_state(self, given_facts: Sequence[Fact] = ()) -> MathState:
        """Return the initial state for ``self.problem_statement``.

        ``given_facts`` packages problem-statement premises as ``Fact``
        records with ``source="given"``. An initial open subgoal named
        ``sg_root`` packages "produce the final answer" so the agent
        has something to decompose against on the first turn.
        """
        return MathState(
            problem=self.problem_statement,
            facts=tuple(given_facts),
            open_subgoals=(
                Subgoal(id="sg_root", statement="Produce the final answer."),
            ),
            turn=0,
        )

    def actions(self, state: MathState) -> Sequence[MathAction]:
        return tuple(
            MathAction(verb=v)
            for v in available_actions(state, depth_limit=self.depth_limit)
        )

    def is_terminal(self, state: MathState) -> bool:
        return state.is_terminal() or state.turn >= self.max_turns

    def sample_transition(
        self, state: MathState, action: MathAction, rng: Any = None
    ) -> tuple[MathState, float]:
        """One verb-instantiated step.

        Sequence of operations:

        1. Build the prompt and schema for ``action.verb``.
        2. Call ``self.llm_client.complete_json`` for a structured
           continuation.
        3. Validate the continuation's target identifier (if any)
           against the current state's id set.
        4. Assemble the next state via the verb's specific handler.
        5. Attribute the reward.

        Failure modes (LLM produced unparseable JSON, target id does
        not exist, verifier raised an internal error) return the
        unchanged state with a small negative shaped reward
        (``-0.01``) so the policy learns to avoid them. This is the
        "masked action" recovery described in companion paper
        section 4.4.
        """
        from .llm_client import LLMResponseError

        if self.llm_client is None:
            raise RuntimeError(
                "LLMContextMDP.sample_transition: no llm_client configured. "
                "Pass a MockLLMClient, EchoLLMClient, or "
                "OpenAICompatibleLLMClient at construction."
            )

        verb = action.verb
        if self.prompt_strategy == "focused":
            raise NotImplementedError(
                "prompt_strategy='focused' is planned as an ablation arm but "
                "not implemented yet. Use prompt_strategy='full' for the "
                "smoke test; focused-view templates will be wired in once "
                "the smoke test runs end to end against a real LLM."
            )
        schema = _schema_for(verb, state)
        user = _prompt_for(verb, state)

        # Derive a per-call sampling seed from the caller's rng so that LLM
        # transitions are reproducible at matched seeds. The MCTS planner
        # threads its seeded np.random.Generator into this method; without
        # forwarding a seed, the server samples freely at temperature > 0
        # and two collections at identical --seeds diverge (observed as
        # L1 = 0.553 between matched-condition runs in the pilot).
        sampling_seed = (
            int(rng.integers(0, 2**31 - 1))
            if (rng is not None and self.seed_llm)
            else None
        )

        try:
            response = self.llm_client.complete_json(
                system=_SYSTEM_PROMPT,
                user=user,
                schema=schema,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=sampling_seed,
            )
        except LLMResponseError:
            return self._masked_action(state)

        handler = _VERB_HANDLERS[verb]
        try:
            return handler(self, state, response)
        except (KeyError, ValueError, TypeError):
            return self._masked_action(state)

    def _masked_action(self, state: MathState) -> tuple[MathState, float]:
        """Bump the turn counter and return a small negative reward."""
        next_state = _with(state, turn=state.turn + 1)
        return next_state, MASKED_ACTION_REWARD

    # -- per-verb handlers -------------------------------------------------
    #
    # Handlers assume the response has passed the client-side schema
    # validation (types, enums, array bounds). Because MockLLMClient and
    # other non-validating clients hand responses through verbatim, each
    # handler still guards the fields whose misuse silently corrupts
    # state (a string in an array slot iterates per character; a None
    # statement serializes as the literal "None"). Guards raise
    # ValueError, which sample_transition converts into the masked-action
    # recovery: the same outcome as any other malformed continuation.

    def _handle_decompose(
        self, state: MathState, resp: dict[str, Any]
    ) -> tuple[MathState, float]:
        target_id = resp["target_id"]
        children = resp["children"]
        if isinstance(children, str) or not isinstance(children, (list, tuple)):
            raise ValueError("DECOMPOSE children must be a list of statements")
        if target_id not in state.all_ids():
            return self._masked_action(state)
        new_state = _decompose_target(state, target_id, children)
        return _with(new_state, turn=new_state.turn + 1), 0.0

    def _handle_compute_subexpr(
        self, state: MathState, resp: dict[str, Any]
    ) -> tuple[MathState, float]:
        expression = _scalar_to_str(resp["expression"])
        result = _scalar_to_str(resp["result"])
        derivation = resp.get("derivation")
        if derivation is not None:
            derivation = _scalar_to_str(derivation)
            if not isinstance(derivation, str):
                raise ValueError("COMPUTE_SUBEXPR derivation must be a string")
        if not isinstance(expression, str) or not isinstance(result, str):
            raise ValueError("COMPUTE_SUBEXPR expression/result must be strings")
        fact_id = f"fact_{_next_index(state, 'fact_')}"
        new_fact = Fact(
            id=fact_id,
            statement=f"{expression} = {result}",
            source="derived",
            derivation=derivation,
        )
        new_state = _with(
            state,
            facts=state.facts + (new_fact,),
            turn=state.turn + 1,
        )
        return new_state, 0.0

    def _handle_form_hypothesis(
        self, state: MathState, resp: dict[str, Any]
    ) -> tuple[MathState, float]:
        statement = _scalar_to_str(resp["statement"])
        if not isinstance(statement, str) or not statement.strip():
            raise ValueError("FORM_HYPOTHESIS statement must be a non-empty string")
        parent_sg = resp.get("parent_subgoal_id")
        if parent_sg is not None and parent_sg not in state.all_ids():
            parent_sg = None
        hyp_id = f"hyp_{_next_index(state, 'hyp_')}"
        new_hyp = Hypothesis(
            id=hyp_id,
            statement=statement,
            parent_subgoal=parent_sg,
        )
        return (
            _with(
                state,
                hypotheses=state.hypotheses + (new_hyp,),
                turn=state.turn + 1,
            ),
            0.0,
        )

    def _handle_check_case(
        self, state: MathState, resp: dict[str, Any]
    ) -> tuple[MathState, float]:
        target_id = resp["target_id"]
        cases = resp["cases"]
        if isinstance(cases, str) or not isinstance(cases, (list, tuple)):
            raise ValueError("CHECK_CASE cases must be a list of statements")
        if state.hypothesis_by_id(target_id) is None:
            return self._masked_action(state)
        new_state = _spawn_case_children(state, target_id, cases)
        return _with(new_state, turn=new_state.turn + 1), 0.0

    def _handle_invoke_verifier(
        self, state: MathState, resp: dict[str, Any]
    ) -> tuple[MathState, float]:
        from .verifier import run_verifier

        target_id = resp["target_id"]
        verifier_verb = resp["verifier_verb"]
        input_expression = resp["input_expression"]
        expected = resp.get("expected")

        if target_id not in state.all_ids():
            return self._masked_action(state)

        result = run_verifier(verifier_verb, input_expression, expected)
        vcall_id = f"vcall_{_next_index(state, 'vcall_')}"
        vcall = VerifierCall(
            id=vcall_id,
            verb=verifier_verb,
            input=input_expression,
            output=result.observed,
            success=result.success,
            target_id=target_id,
        )

        new_state = _with(
            state,
            verifier_log=state.verifier_log + (vcall,),
            turn=state.turn + 1,
        )

        if result.success:
            promoted = _promote_to_fact(new_state, target_id, input_expression, result.observed)
            return promoted, _INTERMEDIATE_VERIFIER_REWARD
        return new_state, -_INTERMEDIATE_VERIFIER_REWARD

    def _handle_backtrack(
        self, state: MathState, resp: dict[str, Any]
    ) -> tuple[MathState, float]:
        target_id = resp["target_id"]
        if target_id not in state.all_ids():
            return self._masked_action(state)
        new_state = _abandon_target(state, target_id)
        return _with(new_state, turn=new_state.turn + 1), 0.0

    def _handle_commit_answer(
        self, state: MathState, resp: dict[str, Any]
    ) -> tuple[MathState, float]:
        from .verifier import check_answer

        # The client-side validator coerces JSON numbers to strings for
        # "type": "string" fields, so a numeric answer like 42 arrives
        # here as "42" on the real-client path; the coercion is repeated
        # for non-validating clients (MockLLMClient). A response with no
        # usable answer (None, empty, wrong type) raises ValueError, which
        # sample_transition converts into the masked-action recovery: the
        # commit does not happen, the episode continues, and the policy
        # pays the same small shaped penalty every other verb pays for a
        # malformed continuation. Terminating with the failure reward here
        # would punish transient JSON noise as if the verifier had
        # rejected a real answer.
        answer = _scalar_to_str(resp.get("answer"))
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("COMMIT_ANSWER response carried no usable answer")

        committed_state = _with(
            state,
            committed_answer=answer,
            turn=state.turn + 1,
        )
        if self.expected_answer is None:
            return committed_state, 0.0
        result = check_answer(answer, self.expected_answer)
        reward = (
            _TERMINAL_SUCCESS_REWARD if result.success else _TERMINAL_FAILURE_REWARD
        )
        return committed_state, reward


# Bind handlers after class definition so __dict__ lookup works.
_VERB_HANDLERS: dict[Verb, Callable[["LLMContextMDP", MathState, dict[str, Any]], tuple[MathState, float]]] = {
    Verb.DECOMPOSE: LLMContextMDP._handle_decompose,
    Verb.COMPUTE_SUBEXPR: LLMContextMDP._handle_compute_subexpr,
    Verb.FORM_HYPOTHESIS: LLMContextMDP._handle_form_hypothesis,
    Verb.CHECK_CASE: LLMContextMDP._handle_check_case,
    Verb.INVOKE_VERIFIER: LLMContextMDP._handle_invoke_verifier,
    Verb.BACKTRACK: LLMContextMDP._handle_backtrack,
    Verb.COMMIT_ANSWER: LLMContextMDP._handle_commit_answer,
}


# -----------------------------------------------------------------------------
# State assembly helpers
# -----------------------------------------------------------------------------


def _scalar_to_str(value: Any) -> Any:
    """Coerce a JSON number to its canonical string form; pass everything
    else through unchanged.

    Mirrors the client-side validator's number-to-string coercion so the
    handlers behave identically under non-validating clients (Mock, Echo).
    Integral floats drop the trailing ``.0`` (``42.0`` -> ``"42"``).
    Booleans are NOT coerced: ``True`` as an answer is malformed, not "True".
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _with(state: MathState, **changes: Any) -> MathState:
    """Functional replace: produce a new MathState with the given fields swapped."""
    data = {
        "problem": state.problem,
        "facts": state.facts,
        "open_subgoals": state.open_subgoals,
        "closed_subgoals": state.closed_subgoals,
        "hypotheses": state.hypotheses,
        "verifier_log": state.verifier_log,
        "committed_answer": state.committed_answer,
        "abandoned": state.abandoned,
        "turn": state.turn,
    }
    data.update(changes)
    return MathState(**data)


def _next_index(state: MathState, prefix: str) -> int:
    """Return the smallest integer not yet used as an id with the given prefix."""
    existing: set[int] = set()
    for i in state.all_ids():
        if i.startswith(prefix):
            tail = i[len(prefix):]
            if tail.isdigit():
                existing.add(int(tail))
    n = 1
    while n in existing:
        n += 1
    return n


def _decompose_target(
    state: MathState, target_id: str, children_statements: Sequence[str]
) -> MathState:
    """Spawn child subgoals or hypotheses under ``target_id``.

    The child entity type matches the parent type. Subgoals beget
    subgoals; hypotheses beget hypotheses (depth+1).
    """
    if target_id.startswith("sg_"):
        sg = state.subgoal_by_id(target_id)
        if sg is None:
            return state
        existing_children = list(sg.children)
        new_subgoals: list[Subgoal] = []
        next_idx = _next_index(state, "sg_")
        for stmt in children_statements:
            child_id = f"sg_{next_idx}"
            next_idx += 1
            new_subgoals.append(
                Subgoal(id=child_id, statement=stmt, parent_subgoal=target_id)
            )
            existing_children.append(child_id)
        updated_sg = Subgoal(
            id=sg.id,
            statement=sg.statement,
            parent_subgoal=sg.parent_subgoal,
            children=tuple(existing_children),
        )
        open_sg = tuple(updated_sg if g.id == target_id else g for g in state.open_subgoals)
        return _with(
            state,
            open_subgoals=open_sg + tuple(new_subgoals),
        )

    if target_id.startswith("hyp_"):
        parent = state.hypothesis_by_id(target_id)
        if parent is None:
            return state
        existing_children = list(parent.children)
        new_hyps: list[Hypothesis] = []
        next_idx = _next_index(state, "hyp_")
        for stmt in children_statements:
            child_id = f"hyp_{next_idx}"
            next_idx += 1
            new_hyps.append(
                Hypothesis(
                    id=child_id,
                    statement=stmt,
                    parent_hypothesis=target_id,
                )
            )
            existing_children.append(child_id)
        updated_parent = Hypothesis(
            id=parent.id,
            statement=parent.statement,
            parent_subgoal=parent.parent_subgoal,
            parent_hypothesis=parent.parent_hypothesis,
            children=tuple(existing_children),
        )
        hyps = tuple(updated_parent if h.id == target_id else h for h in state.hypotheses)
        return _with(state, hypotheses=hyps + tuple(new_hyps))

    return state


def _spawn_case_children(
    state: MathState, target_id: str, cases: Sequence[str]
) -> MathState:
    """Like _decompose_target for the hypothesis branch only, with case-style children."""
    return _decompose_target(state, target_id, cases)


def _promote_to_fact(
    state: MathState, target_id: str, expression: str, observed: str
) -> MathState:
    """After a successful verifier call, add a derived Fact mirroring the verified claim.

    The original Hypothesis (or Fact) record is left in place; the
    new ``source="verified"`` Fact captures the promoted claim. This
    matches the framework paper section 4.2 distinction: a
    Hypothesis becomes a Fact via verifier success rather than being
    silently mutated.
    """
    fact_id = f"fact_{_next_index(state, 'fact_')}"
    new_fact = Fact(
        id=fact_id,
        statement=f"{expression} = {observed}" if observed else expression,
        source="verified",
        derivation=f"verified from {target_id}",
    )
    return _with(state, facts=state.facts + (new_fact,))


def _abandon_target(state: MathState, target_id: str) -> MathState:
    """Mark ``target_id`` abandoned. Open subgoals also move to closed."""
    abandoned = state.abandoned + (target_id,)
    if target_id.startswith("sg_"):
        sg = state.subgoal_by_id(target_id)
        if sg is not None and sg in state.open_subgoals:
            open_sg = tuple(g for g in state.open_subgoals if g.id != target_id)
            closed_sg = state.closed_subgoals + (sg,)
            return _with(
                state,
                open_subgoals=open_sg,
                closed_subgoals=closed_sg,
                abandoned=abandoned,
            )
    return _with(state, abandoned=abandoned)


__all__ = [
    "Fact",
    "Subgoal",
    "Hypothesis",
    "VerifierCall",
    "MathState",
    "Verb",
    "MathAction",
    "available_actions",
    "serialize_state",
    "parse_state",
    "LLMContextMDP",
    "INTERMEDIATE_VERIFIER_REWARD",
    "TERMINAL_SUCCESS_REWARD",
    "TERMINAL_FAILURE_REWARD",
    "MASKED_ACTION_REWARD",
]
