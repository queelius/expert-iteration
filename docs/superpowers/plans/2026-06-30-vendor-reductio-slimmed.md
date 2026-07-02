# Vendor reductio (slimmed) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Roll the standalone `reductio` library (github.com/queelius/reductio @ `7c306c5`) into this repo as `src/verifier_anchored_rl/reductio/`, slimming out the `Method` node layer so the code matches `new-design.md` (SPLIT attaches child subgoals directly to the goal; the connective is optional, non-gating metadata), then decommission the standalone repo.

**Architecture:** The vendored subpackage is the domain-agnostic state core of `new-design.md` Section 3: an immutable reasoning forest (`Goal`, `Hypothesis`, `Fact`, `VerifierCall` frozen dataclasses in a `Graph`), a verb algebra of pure functions `Graph -> Ok(Graph) | Err` (`split`, `form_hypothesis`, `compute`, `verify`, `commit`, `abandon`), a `Verifier` protocol with the SymPy implementation quarantined in an adapter, structural lens scoping, and marker serialization. The slim removes the `Method` node kind entirely: `Goal` gains `children` and `connective` fields, `commit` accepts only Fact/Hypothesis witnesses, and no exhaustiveness goal is reified.

**Tech Stack:** Python 3.11+, stdlib-only core, pytest; `sympy` only inside `adapters/` (already provided by this repo's `[llm]` extra).

## Global Constraints

- Core modules (`graph.py`, `actions.py`, `verify.py`, `lens.py`, `serialize.py`, `__init__.py`) import ONLY the standard library. `sympy` may appear only in `adapters/sympy_math.py`. Enforced by the boundary test in Task 4.
- All node dataclasses are `@dataclass(frozen=True)`. No node carries a `reward` or `value` field (reward lives in the MDP layer, not the state core). Enforced by the boundary test.
- Every verb is a total pure function `Graph -> Result` returning `Ok(new_graph)` or `Err(message)`; verbs never raise on bad input and never mutate.
- `solved` / `is_verified` / `holds` are derived predicates. They must never gate anything except `commit`'s witness check (the verifier-vouched-witness rule of `new-design.md` Section 3.2).
- The working tree contains unrelated uncommitted changes (slide decks, spec edits, `new-design.md`). ALWAYS `git add` explicit file paths. NEVER `git add -A`, `git add -u`, or `git add .`.
- Run tests with `python3 -m pytest` (the ambient python3 is /home/spinoza/venv/bin/python3 and has pytest; the repo's pyproject sets `pythonpath = ["src", "."]`, so no install step is needed).
- Do not write the long-dash glyph in any file (a hook rejects it). Use hyphens, commas, or colons.
- After each task, also run the full existing suite (`python3 -m pytest -q`) to prove no regression of the classical anchors (gridworld, expectimax, MCTS).
- Reference the vendor source in the first commit body: `Vendored from github.com/queelius/reductio @ 7c306c5, with the Method node layer slimmed out per new-design.md.`
- Name coexistence is expected and temporary: the legacy `llm_context_mdp.py` exports its own `Fact`, `Hypothesis`, `VerifierCall`, and the vendored subpackage defines classes with the same names. They are distinct classes in distinct modules and never meet in this plan. Do NOT rename either side; the legacy ones are deleted in the later integration work.

## The slim, stated precisely (read before any task)

The standalone reductio models OR-alternative proof attempts with an explicit `Method` node: `Goal.methods -> Method(connective, required) -> child Goals`, an `exhaustive_case` split reifies an extra exhaustiveness goal, and `commit` accepts a Method witness when all its required children are solved. Per `new-design.md` Section 3.3 (subgoals are optional scaffolding; AND/OR is optional non-gating metadata; the only hard gate is the verifier on the committed root answer), the vendored version:

1. Deletes the `Method` class. `Node = Goal | Hypothesis | Fact | VerifierCall`.
2. Gives `Goal` two new fields: `children: tuple[NodeId, ...] = ()` (subgoals attached by SPLIT) and `connective: str | None = None` (set by SPLIT; metadata only, nothing reads it for control flow).
3. `split` appends child Goals directly and records the connective on the parent. A second split with the SAME connective appends more children; a second split with a DIFFERENT connective is an `Err` (lossy metadata is worse than no metadata).
4. `split` no longer reifies an exhaustiveness goal (that obligation machinery is deferred per `new-design.md` Section 3.3).
5. `commit` witnesses are Fact (must hold) or Hypothesis (must be a candidate of the goal and verified). The Method witness route is gone. A parent goal with all children solved is still NOT solved until the parent itself is committed to its own witness; a parent CAN be committed while children remain open or abandoned. Both directions are tested.
6. `lens` child traversal becomes `Goal -> children + candidates`.
7. `serialize` drops the METHOD line kind and renders the connective on the goal line when set.

---

### Task 1: Subpackage skeleton, slimmed graph.py, verify.py

**Files:**
- Create: `src/verifier_anchored_rl/reductio/__init__.py` (placeholder; final exports in Task 4)
- Create: `src/verifier_anchored_rl/reductio/graph.py`
- Create: `src/verifier_anchored_rl/reductio/verify.py`
- Create: `tests/reductio/__init__.py` (empty)
- Test: `tests/reductio/test_graph.py`
- Test: `tests/reductio/test_predicates.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `Goal(id, statement, assumptions=(), children=(), candidates=(), connective=None)`, `Hypothesis(id, statement)`, `Fact(id, statement, source, premises=())`, `VerifierCall(id, target, verb, expected, success, observed)`, `Graph(nodes, root_id, abandoned=frozenset(), solved_by=())`, `NodeId = int`, `new_graph(problem, givens=()) -> Graph`, `get(graph, node_id) -> Node`, `_append(graph, make_node) -> (Graph, NodeId)`, `_replace(graph, node) -> Graph`, `is_abandoned/is_verified/holds/solved(graph, node_id) -> bool`, `_committed_witness(graph, goal_id) -> NodeId | None`; `VerifierResult(success, observed)`, `Verifier` Protocol with `check(statement, verb, expected) -> VerifierResult`, `MockVerifier(table=())`.

- [ ] **Step 1: Create the package init placeholders**

`src/verifier_anchored_rl/reductio/__init__.py`:

```python
"""reductio: the immutable reasoning forest and verb algebra (state core).

Vendored from github.com/queelius/reductio and slimmed to the minimal
state model of new-design.md: no Method node layer; SPLIT attaches child
subgoals directly and records the connective as non-gating metadata.
Public API is assembled in Task 4.
"""
```

`tests/reductio/__init__.py`: empty file.

- [ ] **Step 2: Write the failing tests**

`tests/reductio/test_graph.py`:

```python
from verifier_anchored_rl.reductio.graph import Goal, Fact, new_graph, get


def test_new_graph_seeds_root_goal_and_givens():
    g = new_graph("prove P", givens=("a=1", "b=2"))
    root = get(g, g.root_id)
    assert isinstance(root, Goal)
    assert root.statement == "prove P"
    facts = [n for n in g.nodes if isinstance(n, Fact)]
    assert [f.statement for f in facts] == ["a=1", "b=2"]
    assert all(f.source == "given" for f in facts)


def test_graph_is_a_value_equal_and_hashable():
    g1 = new_graph("p", givens=("a=1",))
    g2 = new_graph("p", givens=("a=1",))
    assert g1 == g2
    assert hash(g1) == hash(g2)
    assert len({g1, g2}) == 1


def test_node_id_equals_index():
    g = new_graph("p", givens=("a=1", "b=2"))
    for i, n in enumerate(g.nodes):
        assert n.id == i


def test_goal_defaults_have_no_children_and_no_connective():
    g = new_graph("p")
    root = get(g, 0)
    assert root.children == ()
    assert root.candidates == ()
    assert root.connective is None
```

`tests/reductio/test_predicates.py`:

```python
from verifier_anchored_rl.reductio.graph import (
    Graph, Goal, Fact, Hypothesis, VerifierCall,
    is_abandoned, is_verified, holds, solved,
)


def _graph(nodes, root_id=0, abandoned=frozenset(), solved_by=()):
    return Graph(nodes=tuple(nodes), root_id=root_id,
                 abandoned=frozenset(abandoned), solved_by=tuple(solved_by))


def test_given_fact_holds_and_is_solved():
    g = _graph([Goal(0, "p"), Fact(1, "a=1", "given")])
    assert holds(g, 1)
    assert solved(g, 1)


def test_derived_fact_holds_only_when_verified():
    g = _graph([Goal(0, "p"), Fact(1, "x=2", "derived")])
    assert not holds(g, 1)
    g2 = _graph([
        Goal(0, "p"),
        Fact(1, "x=2", "derived"),
        VerifierCall(2, target=1, verb="check", expected="x=2", success=True, observed="x=2"),
    ])
    assert is_verified(g2, 1)
    assert holds(g2, 1)


def test_goal_solved_only_when_committed_to_a_discharging_witness():
    nodes = [
        Goal(0, "x=2"),
        Fact(1, "x=2", "derived"),
        VerifierCall(2, target=1, verb="check", expected="x=2", success=True, observed="x=2"),
    ]
    g = _graph(nodes)
    assert not solved(g, 0)
    g_committed = _graph(nodes, solved_by=((0, 1),))
    assert solved(g_committed, 0)


def test_hypothesis_witness_must_be_candidate_and_verified():
    nodes = [
        Goal(0, "g", candidates=(1,)),
        Hypothesis(1, "answer"),
        VerifierCall(2, target=1, verb="check", expected=None, success=True, observed="answer"),
    ]
    g = _graph(nodes, solved_by=((0, 1),))
    assert solved(g, 0)
    # same witness, but not a candidate of the goal: not solved
    nodes_nc = [
        Goal(0, "g"),
        Hypothesis(1, "answer"),
        VerifierCall(2, target=1, verb="check", expected=None, success=True, observed="answer"),
    ]
    g_nc = _graph(nodes_nc, solved_by=((0, 1),))
    assert not solved(g_nc, 0)


def test_children_do_not_gate_or_grant_parent_solved():
    # Parent with both children solved is NOT solved without its own commit
    # (structure is scaffolding, never control flow: new-design.md section 3.3).
    nodes = [
        Goal(0, "g", children=(1, 2), connective="conjunctive"),
        Goal(1, "g1"), Goal(2, "g2"),
        Fact(3, "g1", "given"), Fact(4, "g2", "given"),
    ]
    g = _graph(nodes, solved_by=((1, 3), (2, 4)))
    assert solved(g, 1) and solved(g, 2)
    assert not solved(g, 0)
    # Conversely: parent committed to its own verified witness IS solved
    # even though a child is open and another abandoned.
    nodes2 = [
        Goal(0, "g", children=(1, 2), connective="conjunctive", candidates=(5,)),
        Goal(1, "g1"), Goal(2, "g2"),
        Fact(3, "g1", "given"), Fact(4, "g2", "given"),
        Hypothesis(5, "the answer"),
        VerifierCall(6, target=5, verb="check", expected=None, success=True, observed="the answer"),
    ]
    g2 = _graph(nodes2, abandoned={2}, solved_by=((0, 5),))
    assert solved(g2, 0)


def test_abandoned_blocks_solved():
    nodes = [Goal(0, "x=2"), Fact(1, "x=2", "given")]
    g = _graph(nodes, abandoned={0}, solved_by=((0, 1),))
    assert is_abandoned(g, 0)
    assert not solved(g, 0)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/reductio/ -q`
Expected: collection errors or failures with `ModuleNotFoundError: No module named 'verifier_anchored_rl.reductio.graph'`.

- [ ] **Step 4: Write graph.py (slimmed) and verify.py (verbatim)**

`src/verifier_anchored_rl/reductio/graph.py`:

```python
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
```

`src/verifier_anchored_rl/reductio/verify.py` (verbatim from the standalone repo):

```python
"""The Verifier protocol and a mock implementation for tests.

No domain library is imported here. Real verifiers live in reductio.adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class VerifierResult:
    success: bool
    observed: str


@runtime_checkable
class Verifier(Protocol):
    def check(self, statement: str, verb: str, expected: str | None) -> VerifierResult:
        ...


@dataclass(frozen=True)
class MockVerifier:
    table: tuple[tuple[str, bool], ...] = ()

    def check(self, statement: str, verb: str, expected: str | None) -> VerifierResult:
        success = any(s == statement and ok for s, ok in self.table)
        return VerifierResult(success=success, observed=statement)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/reductio/ -q`
Expected: 10 passed.

- [ ] **Step 6: Full-suite regression check**

Run: `python3 -m pytest -q`
Expected: all tests pass (existing suite + 10 new).

- [ ] **Step 7: Commit**

```bash
git add src/verifier_anchored_rl/reductio/__init__.py \
        src/verifier_anchored_rl/reductio/graph.py \
        src/verifier_anchored_rl/reductio/verify.py \
        tests/reductio/__init__.py \
        tests/reductio/test_graph.py \
        tests/reductio/test_predicates.py
git commit -m "feat(reductio): vendor state core, slimmed graph model

Vendored from github.com/queelius/reductio @ 7c306c5, with the Method
node layer slimmed out per new-design.md: Goal carries children and a
non-gating connective; solved routes are Fact/Hypothesis witness only."
```

---

### Task 2: Slimmed verb algebra (actions.py)

**Files:**
- Create: `src/verifier_anchored_rl/reductio/actions.py`
- Test: `tests/reductio/test_split.py`
- Test: `tests/reductio/test_introduce.py`
- Test: `tests/reductio/test_verify.py`
- Test: `tests/reductio/test_commit_abandon.py`

**Interfaces:**
- Consumes: everything Task 1 produces (`Graph`, node classes, `get`, `_append`, `_replace`, predicates; `Verifier`, `VerifierResult`, `MockVerifier`).
- Produces: `Ok(graph)`, `Err(message)`, `Result = Ok | Err`; `split(graph, goal_id, connective, child_statements) -> Result`, `form_hypothesis(graph, goal_id, statement) -> Result`, `compute(graph, statement, premise_ids) -> Result`, `verify(graph, node_id, verb, expected, verifier) -> Result`, `commit(graph, goal_id, witness_id) -> Result`, `abandon(graph, node_id) -> Result`.

- [ ] **Step 1: Write the failing tests**

`tests/reductio/test_split.py`:

```python
from verifier_anchored_rl.reductio.graph import new_graph, get
from verifier_anchored_rl.reductio.actions import split, Ok, Err


def test_conjunctive_split_attaches_children_directly():
    g = new_graph("solve g")
    r = split(g, 0, "conjunctive", ("g1", "g2"))
    assert isinstance(r, Ok)
    g2 = r.graph
    root = get(g2, 0)
    assert root.connective == "conjunctive"
    assert [get(g2, c).statement for c in root.children] == ["g1", "g2"]


def test_exhaustive_case_split_does_not_reify_exhaustiveness_goal():
    g = new_graph("solve g")
    r = split(g, 0, "exhaustive_case", ("case A", "case B"))
    root = get(r.graph, 0)
    assert root.connective == "exhaustive_case"
    statements = [get(r.graph, c).statement for c in root.children]
    assert statements == ["case A", "case B"]


def test_second_split_same_connective_appends_children():
    g = new_graph("g")
    g = split(g, 0, "conjunctive", ("a",)).graph
    g = split(g, 0, "conjunctive", ("b",)).graph
    root = get(g, 0)
    assert [get(g, c).statement for c in root.children] == ["a", "b"]


def test_second_split_conflicting_connective_is_err():
    g = new_graph("g")
    g = split(g, 0, "conjunctive", ("a",)).graph
    r = split(g, 0, "exhaustive_case", ("b",))
    assert isinstance(r, Err)
    assert "conjunctive" in r.message


def test_split_inherits_assumptions():
    import dataclasses
    g = new_graph("g")
    root = dataclasses.replace(get(g, 0), assumptions=("h1", "h2"))
    g = dataclasses.replace(g, nodes=(root,) + g.nodes[1:])
    g = split(g, 0, "conjunctive", ("g1",)).graph
    child = get(g, get(g, 0).children[0])
    assert child.assumptions == ("h1", "h2")


def test_split_rejects_unknown_goal():
    g = new_graph("g")
    r = split(g, 99, "conjunctive", ("g1",))
    assert isinstance(r, Err)
    assert "99" in r.message


def test_split_rejects_non_goal_target():
    g = new_graph("g", givens=("a=1",))
    r = split(g, 1, "conjunctive", ("g1",))  # node 1 is a Fact
    assert isinstance(r, Err)


def test_split_rejects_bad_connective_or_empty_children():
    g = new_graph("g")
    assert isinstance(split(g, 0, "nonsense", ("g1",)), Err)
    assert isinstance(split(g, 0, "conjunctive", ()), Err)
```

`tests/reductio/test_introduce.py`:

```python
from verifier_anchored_rl.reductio.graph import new_graph, get, Hypothesis, Fact
from verifier_anchored_rl.reductio.actions import form_hypothesis, compute, Ok, Err


def test_form_hypothesis_adds_candidate():
    g = new_graph("g")
    r = form_hypothesis(g, 0, "answer is 42")
    assert isinstance(r, Ok)
    root = get(r.graph, 0)
    assert len(root.candidates) == 1
    h = get(r.graph, root.candidates[0])
    assert isinstance(h, Hypothesis)
    assert h.statement == "answer is 42"


def test_form_hypothesis_rejects_non_goal():
    g = new_graph("g", givens=("a=1",))
    assert isinstance(form_hypothesis(g, 1, "x"), Err)


def test_compute_adds_derived_fact_with_premises():
    g = new_graph("g", givens=("a=2", "b=3"))
    r = compute(g, "a*b=6", premise_ids=(1, 2))
    assert isinstance(r, Ok)
    facts = [n for n in r.graph.nodes if isinstance(n, Fact) and n.source == "derived"]
    assert len(facts) == 1
    assert facts[0].statement == "a*b=6"
    assert facts[0].premises == (1, 2)


def test_compute_rejects_bad_premise():
    g = new_graph("g")            # only the root Goal exists
    assert isinstance(compute(g, "x=1", premise_ids=(0,)), Err)   # node 0 is a Goal
    assert isinstance(compute(g, "x=1", premise_ids=(99,)), Err)  # missing
```

`tests/reductio/test_verify.py`:

```python
from verifier_anchored_rl.reductio.graph import new_graph, get, VerifierCall, is_verified
from verifier_anchored_rl.reductio.actions import split, compute, verify, Ok, Err
from verifier_anchored_rl.reductio.verify import MockVerifier


def test_verify_appends_call_and_marks_verified_on_success():
    g = new_graph("g", givens=("a=2", "b=3"))
    g = compute(g, "a*b=6", premise_ids=(1, 2)).graph   # node 3, derived fact
    v = MockVerifier(table=(("a*b=6", True),))
    r = verify(g, 3, "check", "a*b=6", v)
    assert isinstance(r, Ok)
    calls = [n for n in r.graph.nodes if isinstance(n, VerifierCall)]
    assert len(calls) == 1 and calls[0].success
    assert is_verified(r.graph, 3)


def test_verify_failure_records_call_but_not_verified():
    g = new_graph("g", givens=("a=2",))
    g = compute(g, "a=3", premise_ids=(1,)).graph       # node 2
    v = MockVerifier()                                  # nothing succeeds
    r = verify(g, 2, "check", "a=3", v)
    assert isinstance(r, Ok)
    calls = [n for n in r.graph.nodes if isinstance(n, VerifierCall) and n.target == 2]
    assert len(calls) == 1 and calls[0].success is False
    assert not is_verified(r.graph, 2)


def test_verify_is_idempotent():
    g = new_graph("g", givens=("a=2", "b=3"))
    g = compute(g, "a*b=6", premise_ids=(1, 2)).graph
    v = MockVerifier(table=(("a*b=6", True),))
    g1 = verify(g, 3, "check", "a*b=6", v).graph
    g2 = verify(g1, 3, "check", "a*b=6", v).graph
    assert g1 == g2


def test_verify_rejects_node_without_statement():
    g = new_graph("g")
    g = compute(g, "x=1", premise_ids=()).graph         # node 1, a Fact: fine
    g = split(g, 0, "conjunctive", ("g1",)).graph       # node 2, a child Goal
    # VerifierCall nodes carry no statement; create one via verify, then target it.
    v = MockVerifier(table=(("x=1", True),))
    g = verify(g, 1, "check", "x=1", v).graph           # node 3, a VerifierCall
    r = verify(g, 3, "check", None, v)                  # verifying a VerifierCall
    assert isinstance(r, Err)
```

`tests/reductio/test_commit_abandon.py`:

```python
from verifier_anchored_rl.reductio.graph import new_graph, get, solved, is_abandoned
from verifier_anchored_rl.reductio.actions import (
    split, form_hypothesis, compute, verify, commit, abandon, Ok, Err,
)
from verifier_anchored_rl.reductio.verify import MockVerifier


def _solved_leaf_goal():
    # root goal "x=2"; a verified derived fact "x=2".
    g = new_graph("x=2")
    g = compute(g, "x=2", premise_ids=()).graph              # node 1
    g = verify(g, 1, "check", "x=2", MockVerifier((("x=2", True),))).graph  # node 2
    return g


def test_commit_goal_to_matching_verified_fact():
    g = _solved_leaf_goal()
    assert not solved(g, 0)
    r = commit(g, 0, 1)
    assert isinstance(r, Ok)
    assert solved(r.graph, 0)


def test_commit_rejects_unverified_fact():
    g = new_graph("x=2")
    g = compute(g, "x=2", premise_ids=()).graph              # node 1, not verified
    r = commit(g, 0, 1)
    assert isinstance(r, Err)


def test_commit_accepts_verified_fact_v1_enforces_truth_not_relevance():
    # v1 deliberately does not enforce statement relevance: a verified/holding fact
    # may discharge a goal. Relevance (does it actually answer the goal) is the
    # agent's claim, with a structural entailment check deferred.
    g = new_graph("x=2")
    g = compute(g, "y=9", premise_ids=()).graph              # unrelated derived fact, node 1
    g = verify(g, 1, "check", "y=9", MockVerifier((("y=9", True),))).graph
    r = commit(g, 0, 1)
    assert isinstance(r, Ok)
    assert solved(r.graph, 0)


def test_commit_rejects_goal_witness():
    # With the Method layer gone, a Goal is never a commit witness.
    g = new_graph("g")
    g = split(g, 0, "conjunctive", ("a",)).graph             # node 1, child Goal
    r = commit(g, 0, 1)
    assert isinstance(r, Err)


def test_commit_parent_while_children_open_or_abandoned():
    # Subgoals are optional scaffolding (new-design.md section 3.3): the root
    # can be committed from its own verified hypothesis regardless of children.
    g = new_graph("g")
    g = split(g, 0, "conjunctive", ("a", "b")).graph         # children 1, 2
    g = abandon(g, 2).graph
    g = form_hypothesis(g, 0, "the answer").graph            # node 3
    hyp = get(g, 0).candidates[0]
    g = verify(g, hyp, "check", "the answer",
               MockVerifier((("the answer", True),))).graph
    r = commit(g, 0, hyp)
    assert isinstance(r, Ok)
    assert solved(r.graph, 0)


def test_abandon_marks_and_blocks_solved():
    g = _solved_leaf_goal()
    g = commit(g, 0, 1).graph
    assert solved(g, 0)
    g2 = abandon(g, 0).graph
    assert is_abandoned(g2, 0)
    assert not solved(g2, 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/reductio/ -q`
Expected: the four new files fail with `ModuleNotFoundError: No module named 'verifier_anchored_rl.reductio.actions'`; Task 1 tests still pass.

- [ ] **Step 3: Write actions.py (slimmed)**

`src/verifier_anchored_rl/reductio/actions.py`:

```python
"""The verb algebra: total pure functions Graph -> Result.

Slimmed per new-design.md: split attaches child subgoals directly to the
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


def compute(graph: Graph, statement: str, premise_ids: tuple[NodeId, ...]) -> Result:
    for pid in premise_ids:
        if not _exists(graph, pid):
            return Err(f"no premise with id {pid}")
        if not isinstance(get(graph, pid), (Fact, Hypothesis)):
            return Err(f"premise {pid} is not a Fact or Hypothesis")
    g, _ = _append(graph, lambda nid: Fact(
        id=nid, statement=statement, source="derived", premises=tuple(premise_ids)))
    return Ok(g)


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/reductio/ -q`
Expected: 32 passed (10 from Task 1 + 22 new).

- [ ] **Step 5: Full-suite regression check**

Run: `python3 -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/verifier_anchored_rl/reductio/actions.py \
        tests/reductio/test_split.py \
        tests/reductio/test_introduce.py \
        tests/reductio/test_verify.py \
        tests/reductio/test_commit_abandon.py
git commit -m "feat(reductio): slimmed verb algebra

split attaches children directly (connective as non-gating metadata,
conflicting re-split is Err); commit witnesses are Fact/Hypothesis only."
```

---

### Task 3: Lens scoping and serialization

**Files:**
- Create: `src/verifier_anchored_rl/reductio/lens.py`
- Create: `src/verifier_anchored_rl/reductio/serialize.py`
- Test: `tests/reductio/test_lens.py`
- Test: `tests/reductio/test_serialize.py`

**Interfaces:**
- Consumes: `Graph`, `NodeId`, `Goal`, `Hypothesis`, `Fact`, `VerifierCall`, `get` from Task 1; `split` from Task 2 (tests only).
- Produces: `Subgraph(focus, node_ids, graph)`, `scope(graph, node_id) -> Subgraph`; `render(sub: Subgraph) -> str`, `parse(text) -> list[tuple[str, int, str]]`.

- [ ] **Step 1: Write the failing tests**

`tests/reductio/test_lens.py`:

```python
from verifier_anchored_rl.reductio.graph import new_graph
from verifier_anchored_rl.reductio.actions import split, form_hypothesis
from verifier_anchored_rl.reductio.lens import scope


def test_scope_includes_focus_descendants_and_ancestors():
    g = new_graph("root")
    g = split(g, 0, "conjunctive", ("a", "b")).graph     # children: 1 (a), 2 (b)
    g = split(g, 1, "conjunctive", ("a1",)).graph        # under goal 1: child 3 (a1)
    sub = scope(g, 1)                                     # focus the "a" subgoal
    ids = set(sub.node_ids)
    assert sub.focus == 1
    assert 1 in ids                       # focus
    assert 3 in ids                       # descendant subgoal
    assert 0 in ids                       # ancestor path to root
    assert 2 not in ids                   # sibling subgoal "b" is out of scope


def test_scope_includes_candidate_hypotheses_of_focus():
    g = new_graph("root")
    g = split(g, 0, "conjunctive", ("a", "b")).graph     # children 1, 2
    g = form_hypothesis(g, 1, "claim for a").graph       # node 3, candidate of 1
    ids = set(scope(g, 1).node_ids)
    assert 3 in ids


def test_scope_of_root_is_whole_forest_backbone():
    g = new_graph("root")
    g = split(g, 0, "conjunctive", ("a",)).graph
    sub = scope(g, 0)
    assert set(sub.node_ids) == {0, 1}
```

`tests/reductio/test_serialize.py`:

```python
from verifier_anchored_rl.reductio.graph import new_graph
from verifier_anchored_rl.reductio.actions import split
from verifier_anchored_rl.reductio.lens import scope
from verifier_anchored_rl.reductio.serialize import render, parse


def test_render_is_deterministic_and_neutral():
    g = new_graph("root")
    g = split(g, 0, "conjunctive", ("a",)).graph
    text = render(scope(g, 0))
    assert text.splitlines()[0] == "focus=0"
    assert "[GOAL id=0] root connective=conjunctive" in text
    assert "[GOAL id=1] a" in text
    assert "METHOD" not in text
    # neutral: no prompt or menu words
    assert "prompt" not in text.lower() and "choose" not in text.lower()


def test_parse_round_trips_node_lines():
    g = new_graph("root")
    g = split(g, 0, "conjunctive", ("a",)).graph
    text = render(scope(g, 0))
    rows = parse(text)
    assert ("GOAL", 0, "root connective=conjunctive") in rows
    assert ("GOAL", 1, "a") in rows
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/reductio/ -q`
Expected: new files fail with `ModuleNotFoundError` for `lens` / `serialize`; earlier tests still pass.

- [ ] **Step 3: Write lens.py and serialize.py**

`src/verifier_anchored_rl/reductio/lens.py`:

```python
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
        return n.children + n.candidates
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
```

`src/verifier_anchored_rl/reductio/serialize.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/reductio/ -q`
Expected: 37 passed (32 + 5 new).

- [ ] **Step 5: Full-suite regression check**

Run: `python3 -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/verifier_anchored_rl/reductio/lens.py \
        src/verifier_anchored_rl/reductio/serialize.py \
        tests/reductio/test_lens.py \
        tests/reductio/test_serialize.py
git commit -m "feat(reductio): lens scoping and marker serialization

Goal traversal is children + candidates; serializer renders the
connective on the goal line and has no METHOD line kind."
```

---

### Task 4: SymPy adapter, public API, boundary test

**Files:**
- Create: `src/verifier_anchored_rl/reductio/adapters/__init__.py` (empty)
- Create: `src/verifier_anchored_rl/reductio/adapters/sympy_math.py`
- Modify: `src/verifier_anchored_rl/reductio/__init__.py` (replace placeholder body with exports)
- Test: `tests/reductio/test_sympy_adapter.py`
- Test: `tests/reductio/test_boundary.py`

**Interfaces:**
- Consumes: `VerifierResult` from Task 1; all public names from Tasks 1-3.
- Produces: `SymPyVerifier` (implements the `Verifier` protocol); package-level re-exports: `Graph, NodeId, Goal, Hypothesis, Fact, VerifierCall, new_graph, get, solved, is_verified, is_abandoned, holds, Result, Ok, Err, split, form_hypothesis, compute, verify, commit, abandon, Verifier, VerifierResult, MockVerifier, Subgraph, scope, render, parse`.

- [ ] **Step 1: Write the failing tests**

`tests/reductio/test_sympy_adapter.py`:

```python
import pytest


def test_adapter_satisfies_protocol():
    pytest.importorskip("sympy")
    from verifier_anchored_rl.reductio.adapters.sympy_math import SymPyVerifier
    from verifier_anchored_rl.reductio.verify import Verifier
    assert isinstance(SymPyVerifier(), Verifier)


def test_equal_expressions_verify():
    pytest.importorskip("sympy")
    from verifier_anchored_rl.reductio.adapters.sympy_math import SymPyVerifier
    v = SymPyVerifier()
    assert v.check("2*(x+1)", "simplify", "2*x+2").success


def test_unequal_expressions_do_not_verify():
    pytest.importorskip("sympy")
    from verifier_anchored_rl.reductio.adapters.sympy_math import SymPyVerifier
    v = SymPyVerifier()
    assert not v.check("x+1", "simplify", "x+2").success


def test_equation_equality():
    pytest.importorskip("sympy")
    from verifier_anchored_rl.reductio.adapters.sympy_math import SymPyVerifier
    v = SymPyVerifier()
    assert v.check("Eq(7*6, 42)", "check", "True").success or \
           v.check("7*6", "evaluate", "42").success


def test_parse_error_is_not_an_exception():
    pytest.importorskip("sympy")
    from verifier_anchored_rl.reductio.adapters.sympy_math import SymPyVerifier
    v = SymPyVerifier()
    r = v.check("@@bad@@", "check", "x")
    assert r.success is False
```

`tests/reductio/test_boundary.py`:

```python
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_public_api_is_importable():
    import verifier_anchored_rl.reductio as reductio
    for name in ["new_graph", "split", "form_hypothesis", "compute",
                 "verify", "commit", "abandon", "scope", "render",
                 "Verifier", "Ok", "Err"]:
        assert hasattr(reductio, name), name


def test_reductio_modules_add_no_heavy_imports():
    # The parent package (verifier_anchored_rl/__init__.py) eagerly imports
    # numpy and the LLM client stack, and Python must execute it before any
    # subpackage import. So the boundary claim for the vendored core is a
    # DELTA claim: importing reductio's modules adds nothing banned beyond
    # what the host package already loaded. Run in a fresh interpreter so
    # the surrounding test session cannot mask a leak.
    code = (
        "import importlib, sys\n"
        "importlib.import_module('verifier_anchored_rl')\n"
        "before = set(sys.modules)\n"
        "mods = ['verifier_anchored_rl.reductio',\n"
        "        'verifier_anchored_rl.reductio.graph',\n"
        "        'verifier_anchored_rl.reductio.actions',\n"
        "        'verifier_anchored_rl.reductio.verify',\n"
        "        'verifier_anchored_rl.reductio.lens',\n"
        "        'verifier_anchored_rl.reductio.serialize']\n"
        "for m in mods: importlib.import_module(m)\n"
        "added = set(sys.modules) - before\n"
        "banned = {'sympy', 'torch', 'httpx', 'requests', 'numpy'}\n"
        "leaked = banned & added\n"
        "assert not leaked, f'reductio import pulled in: {leaked}'\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    subprocess.run([sys.executable, "-c", code], check=True,
                   cwd=str(REPO_ROOT), env=env)


def test_no_reward_or_value_field_on_nodes():
    from verifier_anchored_rl.reductio import graph
    for cls in (graph.Goal, graph.Hypothesis, graph.Fact, graph.VerifierCall):
        fields = cls.__dataclass_fields__.keys()
        assert "reward" not in fields and "value" not in fields


def test_method_layer_is_gone():
    from verifier_anchored_rl.reductio import graph
    assert not hasattr(graph, "Method")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/reductio/test_sympy_adapter.py tests/reductio/test_boundary.py -q`
Expected: adapter tests fail with `ModuleNotFoundError` (or skip if sympy is absent); boundary fails on missing public API names.

- [ ] **Step 3: Write the adapter and the public API**

`src/verifier_anchored_rl/reductio/adapters/__init__.py`: empty file.

`src/verifier_anchored_rl/reductio/adapters/sympy_math.py` (verbatim from the standalone repo):

```python
"""SymPy implementation of the Verifier protocol. The only module importing sympy."""

from __future__ import annotations

from dataclasses import dataclass

import sympy

from ..verify import VerifierResult


def _to_expr(text: str):
    expr = sympy.sympify(text)
    if isinstance(expr, sympy.Equality):
        return expr.lhs - expr.rhs
    # sympy.sympify('True') may return Python bool True rather than sympy.true;
    # handle both forms so that the Eq-vs-True path works correctly.
    if expr is sympy.true or expr is True:
        return sympy.Integer(0)
    if expr is sympy.false or expr is False:
        return sympy.Integer(1)
    return expr


@dataclass(frozen=True)
class SymPyVerifier:
    def check(self, statement: str, verb: str, expected: str | None) -> VerifierResult:
        try:
            lhs = _to_expr(statement)
            if expected is None:
                return VerifierResult(success=True, observed=str(lhs))
            rhs = _to_expr(expected)
            success = sympy.simplify(lhs - rhs) == 0
            return VerifierResult(success=bool(success), observed=str(sympy.sympify(statement)))
        except (sympy.SympifyError, TypeError, SyntaxError, AttributeError):
            return VerifierResult(success=False, observed="<parse-error>")
```

Replace the body of `src/verifier_anchored_rl/reductio/__init__.py` with:

```python
"""reductio: the immutable reasoning forest and verb algebra (state core).

Vendored from github.com/queelius/reductio and slimmed to the minimal
state model of new-design.md: no Method node layer; SPLIT attaches child
subgoals directly and records the connective as non-gating metadata.
"""

from .graph import (
    Graph, NodeId, Goal, Hypothesis, Fact, VerifierCall,
    new_graph, get, solved, is_verified, is_abandoned, holds,
)
from .actions import (
    Result, Ok, Err, split, form_hypothesis, compute, verify, commit, abandon,
)
from .verify import Verifier, VerifierResult, MockVerifier
from .lens import Subgraph, scope
from .serialize import render, parse

__all__ = [
    "Graph", "NodeId", "Goal", "Hypothesis", "Fact", "VerifierCall",
    "new_graph", "get", "solved", "is_verified", "is_abandoned", "holds",
    "Result", "Ok", "Err", "split", "form_hypothesis", "compute", "verify",
    "commit", "abandon", "Verifier", "VerifierResult", "MockVerifier",
    "Subgraph", "scope", "render", "parse",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/reductio/ -q`
Expected: 46 passed (or 41 passed, 5 skipped if sympy is not installed locally).

- [ ] **Step 5: Full-suite regression check**

Run: `python3 -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/verifier_anchored_rl/reductio/adapters/__init__.py \
        src/verifier_anchored_rl/reductio/adapters/sympy_math.py \
        src/verifier_anchored_rl/reductio/__init__.py \
        tests/reductio/test_sympy_adapter.py \
        tests/reductio/test_boundary.py
git commit -m "feat(reductio): SymPy adapter, public API, core boundary test

Core imports stdlib only (fresh-interpreter boundary test); sympy is
quarantined in adapters; Method is asserted gone."
```

---

### Task 5: Worked example (the new-design section 3.1 forest)

**Files:**
- Test: `tests/reductio/test_worked_example.py`

**Interfaces:**
- Consumes: the full public API from Task 4.
- Produces: nothing new; this is the integration test that pins the slimmed semantics to the worked forest in `new-design.md` Section 3.1.

- [ ] **Step 1: Write the test (it should pass immediately if Tasks 1-4 are correct; treat a failure as a bug in those tasks, not in the test)**

`tests/reductio/test_worked_example.py`:

```python
"""The abs-value case analysis of new-design.md section 3.1, end to end.

Solve |x-3| + |x+2| = 7 by exhaustive case split. Exercises: SPLIT with
direct children, per-case hypothesis discharge (including a verified
no-solution result), ABANDON of a dead-end case, subgoal COMMIT as
bookkeeping, and the root COMMIT from a verified root hypothesis.
"""

from verifier_anchored_rl.reductio import (
    new_graph, split, form_hypothesis, compute, verify, commit, abandon,
    get, solved, is_abandoned, MockVerifier,
)


def _discharge(g, goal_id, claim):
    """form -> verify -> commit a goal via a verified hypothesis."""
    g = form_hypothesis(g, goal_id, claim).graph
    hyp = get(g, goal_id).candidates[-1]
    g = verify(g, hyp, "check", claim, MockVerifier(((claim, True),))).graph
    return commit(g, goal_id, hyp).graph


def test_absolute_value_case_analysis_solves_root():
    g = new_graph("solutions of |x-3|+|x+2|=7")
    g = split(g, 0, "exhaustive_case", (
        "solutions with x<-2",
        "solutions with -2<=x<3",
        "solutions with x>=3",
    )).graph
    case_a, case_b, case_c = get(g, 0).children

    # case x < -2: 1-2x = 7 leads to x = -3, verified.
    g = _discharge(g, case_a, "x=-3")
    # case -2 <= x < 3: 5 = 7 is a contradiction; the case is a dead end.
    g = compute(g, "5=7", premise_ids=()).graph
    g = abandon(g, case_b).graph
    # case x >= 3: 2x-1 = 7 leads to x = 4, verified.
    g = _discharge(g, case_c, "x=4")

    assert solved(g, case_a) and solved(g, case_c)
    assert is_abandoned(g, case_b) and not solved(g, case_b)

    # Root closes from its own verified hypothesis; children were scaffolding.
    g = _discharge(g, 0, "x=-3 or x=4")
    assert solved(g, 0)


def test_root_commit_does_not_require_all_children_closed():
    # Same structure, but the policy skips two cases entirely and still
    # closes the root from a verified hypothesis: structure never gates.
    g = new_graph("solutions of |x-3|+|x+2|=7")
    g = split(g, 0, "exhaustive_case", ("case A", "case B", "case C")).graph
    g = _discharge(g, 0, "x=-3 or x=4")
    assert solved(g, 0)
```

- [ ] **Step 2: Run the test**

Run: `python3 -m pytest tests/reductio/test_worked_example.py -v`
Expected: 2 passed. If either fails, debug Tasks 1-4 (the test encodes the spec).

- [ ] **Step 3: Full-suite run**

Run: `python3 -m pytest -q`
Expected: all pass (48 reductio tests plus the existing suite).

- [ ] **Step 4: Commit**

```bash
git add tests/reductio/test_worked_example.py
git commit -m "test(reductio): worked example pins new-design section 3.1 semantics"
```

---

### Task 6: Decommission the standalone repo; update catalog and memory

**Files:**
- Modify: `/home/spinoza/github/reductio/README.md` (prepend absorption notice)
- Modify: `/home/spinoza/github/CLAUDE.md` (catalog rows mentioning reductio)
- Modify: `/home/spinoza/.claude/projects/-home-spinoza-github-expert-iteration/memory/reductio-repo.md`

**Interfaces:**
- Consumes: Tasks 1-5 merged (the vendored code exists and passes).
- Produces: a single source of truth; no separate library to maintain.

- [ ] **Step 1: Prepend the absorption notice to the standalone repo's README**

At the very top of `/home/spinoza/github/reductio/README.md`, above the existing title, insert:

```markdown
> **Absorbed (2026-06-30).** This library now lives inside
> [`expert-iteration`](https://github.com/queelius/expert-iteration) as
> `src/verifier_anchored_rl/reductio/` (vendored at `7c306c5` and slimmed:
> the Method node layer was removed; SPLIT attaches child subgoals directly
> and records the connective as non-gating metadata). This repository is
> archived and will not receive further changes.

```

- [ ] **Step 2: Commit and push the notice, then archive on GitHub**

```bash
cd /home/spinoza/github/reductio
git add README.md
git commit -m "docs: absorbed into expert-iteration; archiving"
git push origin main
gh repo archive queelius/reductio --yes
```

Expected: push succeeds; `gh repo archive` reports the repo is archived.

- [ ] **Step 3: Update the ~/github/ catalog**

In `/home/spinoza/github/CLAUDE.md` (NOT a git repo; plain file edits):

1. Directory Organization table: change the `reductio/` row description to: `Archived standalone repo; the reasoning-forest core now lives in expert-iteration (src/verifier_anchored_rl/reductio/)`.
2. Top-Level Standalone Repos table: update the `reductio/` row description the same way.
3. The `expert-iteration/` rows (both tables): append `; includes the vendored reductio state core` to the description.

- [ ] **Step 4: Update the memory file**

Rewrite `/home/spinoza/.claude/projects/-home-spinoza-github-expert-iteration/memory/reductio-repo.md` body (keep frontmatter, update description) to record: reductio was vendored into `expert-iteration` at `src/verifier_anchored_rl/reductio/` on 2026-06-30 (source SHA `7c306c5`), slimmed during roll-in (Method node layer removed; SPLIT attaches children directly; connective is non-gating metadata; commit witnesses are Fact/Hypothesis only), tests live in `tests/reductio/`, and the standalone GitHub repo is archived. Keep the design-facts list that still applies (pure verb algebra, pluggable Verifier, sympy quarantined in adapters, derived predicates).

- [ ] **Step 5: Verify the vendored suite one more time and finish**

```bash
cd /home/spinoza/github/expert-iteration
python3 -m pytest tests/reductio/ -q
```

Expected: all pass. No commit in this repo for this step (catalog and memory files live outside it; the reductio repo was committed in Step 2).

---

## Self-review notes

- Spec coverage: the slim (points 1-7 of "The slim, stated precisely") maps to Tasks 1 (points 1, 2), 2 (points 3, 4, 5), 3 (points 6, 7); the decommission requirement maps to Task 6; new-design section 3.1 semantics are pinned by Task 5.
- Deliberately dropped from the standalone repo: `test_smoke.py` (`__version__` is meaningless in a vendored subpackage; importability is covered by the boundary test) and the Method-dependent tests, which are replaced by their slimmed equivalents.
- Type consistency: `split(graph, goal_id, connective, child_statements)`, `commit(graph, goal_id, witness_id)`, `Goal.children`/`Goal.connective` are used with identical signatures in Tasks 1-5.
- The boundary test runs the import check in a fresh interpreter because the vendored core sits inside `verifier_anchored_rl`, whose sibling modules (numpy, httpx users) may already be imported by the surrounding test session; the standalone repo's in-process check would false-positive here.
