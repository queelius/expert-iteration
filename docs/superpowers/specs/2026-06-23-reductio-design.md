# reductio: an immutable AND-OR proof forest for verifier-anchored reasoning

**Status:** design, slimmed revision. Supersedes an earlier content-addressed-DAG draft of this spec (kept in git history).
**Form:** a new standalone repository, `reductio`.
**Boundary:** a pristine core. No LLM, no MCTS, no reward, no learning. `expert-iteration` imports `reductio` and maps its structural proof-status predicates onto rewards and values on its own side.
**Design stance:** deliberately minimal. v1 is an immutable AND-OR tree (forest) with plain ids and structural equality. Content-addressing, multi-parent DAG sharing, statement normalization, semantic-equality edges, and MCTS transposition were designed in full, judged to be more machinery than the Phase-1 workload needs, and demoted to deferred upgrades (section 11). The upgrade path is additive and sits behind the same verb interface.

## 1. Purpose

`reductio` provides one thing: an immutable **AND-OR proof forest** (a "reasoning graph") and a pure **verb algebra** of total functions `Graph -> Result[Graph]` over it. The graph is a persistent value: every verb returns a new graph that shares the unchanged parts, so two graphs compare and hash by structural equality.

This is the representation the `expert-iteration` action-space-lens design (`2026-06-23-action-space-lens-design.md`) needs underneath it. That design defined the verbs (`SPLIT`, `FORM_HYPOTHESIS`, `COMPUTE`, `VERIFY`, `COMMIT`, `ABANDON`) and the lens as actions over a "goal tree." `reductio` makes the data structure precise and gives each verb an exact, invariant-preserving semantics. The RL machinery (the `GenerativeMDP` wrapper, reward assignment, MCTS, the heads, expert iteration, the LLM client, prompt rendering) stays in `expert-iteration` and consumes `reductio`.

## 2. The state: an immutable AND-OR proof forest

### 2.1 Nodes

Every node is a frozen dataclass with a stable plain id (assigned at creation, for example a monotonic counter), propositional fields, and a small amount of stored status. Node kinds:

- **Goal** (an OR node): a problem or subgoal. Fields: `statement`, `assumptions` (the in-scope hypotheses/givens). Solved iff any attached `Method` or discharged candidate `Hypothesis` is solved (derived, section 4). The root is a `Goal` from the constructor.
- **Method** (an AND node): the decomposition a `SPLIT` produces. Fields: `connective in {conjunctive, exhaustive_case}`, the required child goals. Solved iff all required goals are solved (derived). For `exhaustive_case` the exhaustiveness obligation is a reified ordinary `Goal` child (2.3), so the uniform AND rule discharges it with no special case.
- **Hypothesis**: a tentative candidate solution to a goal. Field: `statement`.
- **Fact**: an established statement. Fields: `statement`, `source in {given, derived}`, `premises` (ids). Verified-ness is derived (a `Fact` is verified iff it has a successful incoming `VerifierCall`); it is not a stored `source` value.
- **VerifierCall**: a record of a checker invocation. Fields: `target_id, verb, expected, success, observed`.

Status: `abandoned` and the commit witness (`solved_by` on a goal) are **stored decisions** (markers/fields). `solved` and `verified` are **derived predicates** (section 4). Containment is a tree (each node has exactly one structural parent); derivation and witness relationships are **by-id references** into the forest.

### 2.2 Edges and references

- **Containment (the tree backbone)**: `goal -has_method-> method`, `method -requires-> goal`, `goal -candidate-> hypothesis`. Each node has exactly one structural parent (the root goal has none), which makes the backbone a tree/forest.
- **References (by id, not ownership)**: `fact -derived_from-> premises`, `verifiercall -checks-> node`, `goal -solved_by-> witness` (set by `COMMIT`). References always point to nodes that already exist, so the reference graph is acyclic by construction. No ancestor-check is needed: because v1 does not collapse distinct statements into one node, a freshly created subgoal is never identical to an ancestor, so a circular proof cannot form structurally.

### 2.3 AND-OR semantics

The forest alternates classically (Nilsson problem reduction / AND-OR graphs): goals are OR nodes (alternative approaches), methods are AND nodes (all subgoals required), with facts and verified hypotheses as the leaves that discharge goals. A **case-split is disjunctive in the domain but conjunctive in the proof**: to discharge `G` by cases you must solve every case AND show exhaustiveness. So a `SPLIT(exhaustive_case)` produces an AND `Method` whose required children are the case goals plus a distinguished **exhaustiveness `Goal`**. The genuine OR is a goal carrying more than one candidate method or hypothesis.

## 3. Identity, equality, persistence

- **Ids** are plain stable identifiers assigned at creation; they are not content hashes. Two syntactically different but semantically equal statements are simply distinct nodes in v1 (no automatic collapse; that is the deferred content-addressing upgrade, section 11). Two textually identical goals under different assumptions are also distinct, because `assumptions` is part of the `Goal` and of structural equality.
- **Equality and hashing are structural.** A `Graph` is an immutable value (frozen dataclasses throughout), so structural equality and hashing identify two states iff their forests are identical. That is all MCTS needs in Phase 1. Transposition (detecting semantically-equal states reached by different paths) is deferred.
- **Persistence.** Every verb returns a new `Graph`; an update path-copies the affected node up to the root and shares the rest. Ids are stable across path-copying, so by-id references survive an update.

## 4. Derived predicates

`solved`, `abandoned`, and `verified` are queries, not eagerly-pushed flags, computed as a memoized fold over the forest (the AO* solved-label backup, applied as a pure query rather than a search):

- `solved(Goal) = OR over its methods and discharged candidate hypotheses of solved(.)`
- `solved(Method) = AND over its required goals of solved(.)`
- A `Fact` is `verified` iff it has a successful incoming `VerifierCall`; `solved(verified Fact) = true`.
- `abandoned` is a stored marker. An AND-`Method` with an abandoned required child has `solved() = False` while the subtree is retained; an OR-`Goal` can still be solved by a non-abandoned alternative.

Computed bottom-up, memoized per immutable `Graph`. `reductio` exposes these as structural predicates only, never a numeric reward or value.

## 5. The action algebra

The root `Goal` and given `Fact`s enter via a **constructor** in `graph.py`, outside the verb algebra. `COMPUTE` (derive), `VERIFY` (check), and `COMMIT` (close) are kept orthogonal: no verb does two. Every verb is a total function `Graph -> Result[Graph]` (`Result` is success-with-new-graph or rejection-with-error-and-unchanged-graph), written as a uniform pure function so the deferred kernel/vocabulary and meta-learning directions (section 11) stay a registration concern rather than a rewrite.

- **`new_graph(problem, givens)`**: build the root `Goal` plus the given `Fact`s.
- **`SPLIT(graph, goal_id, connective, child_statements)`**: add a `Method` plus one child `Goal` per statement (each inheriting/extending the parent's assumptions); `exhaustive_case` also adds the reified exhaustiveness `Goal`. Conjunctive child order is canonicalized so the same split in any order yields equal graphs.
- **`FORM_HYPOTHESIS(graph, goal_id, statement)`**: add a `Hypothesis` candidate on the goal.
- **`COMPUTE(graph, statement, premise_ids)`**: add a derived `Fact` referencing existing premises. Does not bless or close anything.
- **`VERIFY(graph, node_id, verb, expected)`**: invoke the supplied `Verifier` (the math adapter is SymPy; the core depends on no domain library); add a `VerifierCall`. On success the checked node's fact becomes verified (derived), usable as a typed `COMMIT` witness. Re-verifying the same node with the same `(verb, expected)` produces an equal graph (idempotent); the invocation count is not recorded here (anti-farming counting lives in `expert-iteration`).
- **`COMMIT(graph, goal_id, witness_id)`**: set `goal.solved_by = witness`. The witness is type-constrained: a verified `Fact` whose statement matches the goal, an all-required-solved `Method`, or a discharged `Hypothesis`. An untyped or unverified witness is rejected. Solved-ness everywhere is then queried as a derived fixpoint, so multi-witness and grandparent propagation is order-independent.
- **`ABANDON(graph, node_id)`**: mark the node abandoned (node and edges retained).

Existence and witness-typing checks live in `actions.py`.

## 6. Lens and serialization

The LLM/generation boundary is kept clean by splitting concerns:

- **`lens.scope(graph, node_id) -> Subgraph`** (in `lens.py`): pure structural scope. Given a focus node, return a scoped immutable sub-forest: the node's method/subgoal neighborhood, its candidate hypotheses, its derived/verified facts, the ancestor path to the root, and the in-scope assumptions. It knows nothing about prompts, action menus, or how generation is produced. Pure `graph -> graph`.
- **`serialize.render(subgraph) -> str`** (in `serialize.py`): a fixed, deterministic, neutral textual format (structured markers for node kind, id, statement, status, references), plus `parse` back where round-tripping is useful. It knows only this fixed format, never prompts or per-verb action menus (those live in `expert-iteration`).

"Generation writes back locally via the verbs" means `expert-iteration` reads ids from the lens and calls `reductio`'s pure verbs. `reductio` never imports or models generation.

## 7. The Verifier protocol

`verify.py` defines a structural-subtyping `Verifier` Protocol; the core depends on neither SymPy nor any other domain library. A domain supplies a `Verifier`; it is the only source of `verified` status. The reference **math `Verifier`** is a SymPy adapter (`adapters/sympy_math.py`), mirroring the existing `verifier_anchored_rl/verifier.py` logic without importing it or any RL concept. Keeping verification behind a protocol is what stops SymPy from specializing the core to mathematics: `reductio` is a general verifier-anchored reasoning substrate, and math is one adapter. (No `Normalizer` protocol in v1: statement normalization existed only to feed content-addressing, which is deferred.)

## 8. Module layout

```
reductio/
  pyproject.toml        # core deps: standard library only. Dev: pytest.
                        # Optional extra [math]: sympy, for the math Verifier adapter only.
  README.md
  src/reductio/
    __init__.py         # public API: new_graph, the verbs, the Verifier protocol, lens.scope, serialize. Re-exports only.
    graph.py            # frozen AND-OR forest nodes (Goal/Method/Hypothesis/Fact/VerifierCall), stable ids,
                        # the persistent (path-copying) Graph value, new_graph constructor, derived solved/abandoned/verified.
    actions.py          # the verb algebra (total Graph -> Result[Graph]); existence + witness-typing checks.
    verify.py           # the Verifier protocol (structural subtyping). No implementation in the core.
    lens.py             # pure scope(graph, node_id) -> Subgraph only.
    serialize.py        # render(subgraph) -> str fixed-format + parse-back; neutral text only.
    adapters/
      __init__.py
      sympy_math.py     # the math Verifier adapter (SymPy). The only place SymPy lives; imported with the [math] extra.
  tests/                # core suites use a mock Verifier; the math adapter has its own suite (incl. the worked example).
  docs/
```

## 9. Testing strategy

- **Structural equality / persistence**: a `Graph` is a value; identical forests compare and hash equal; a verb returns a new graph sharing untouched subtrees (object identity of the unchanged parts), and by-id references survive the path-copy.
- **Derived-status fixpoint**: the AND/OR rules; multi-witness propagation is order-independent; an AND-`Method` with an abandoned required child is `solved() = False` with the subtree retained; an `exhaustive_case` `Method` is solved iff all cases AND the reified exhaustiveness goal are solved.
- **Verb lifecycle typing**: `COMMIT` rejects an unverified or untyped witness; the `FORM_HYPOTHESIS -> VERIFY -> COMMIT` path closes a goal; `COMPUTE` alone never closes a goal; `VERIFY` is idempotent and records no count.
- **Acyclicity by construction**: a reference only ever targets a pre-existing node; no verb can produce a cycle.
- **Boundary**: `reductio` imports with no torch/LLM/MCTS/networking and no SymPy in the core (SymPy only via the `[math]` adapter); no node has a reward/value field; `lens.scope` output contains no prompt/action-menu strings.
- **Serializer round-trip**: `render(parse(text)) == text` on the fixed format.
- **Worked example**: the `|x-3| + |x+2| = 7` problem builds the expected AND-OR forest (root goal, an `exhaustive_case` Method with three case goals plus an exhaustiveness goal, two verified facts, one abandoned case, a root commit), matching the trace from the action-space-lens design.

## 10. Boundary (what stays in expert-iteration)

`reductio` reports only proof status (`solved`/`unsolved`/`abandoned`/`verified`) as structural predicates. `expert-iteration` maps status to reward and value on its side (terminal reward when the root goal becomes solved; a gated, capped intermediate signal when a new verified `Fact` appears). The `GenerativeMDP` wrapper, MCTS, the heads, expert iteration, the LLM client, prompt/action-menu rendering, and the anti-farming verifier-call counting all stay in `expert-iteration`. `reductio` has no numeric reward or value field anywhere, and the verifier-call count is never in the graph.

## 11. Deferred directions (recorded, not built now)

These were considered and deliberately deferred. v1 must not foreclose them, and it does not (the verbs are a stable interface; the structures below are additive behind it).

- **Content-addressing, normalization, semantic equality, and multi-parent DAG sharing.** Make node identity a hash of a normalized statement (hash-consing) so identical statements collapse to one shared multi-parent node, with a status overlay separating identity from standing, an enforced acyclicity precondition (because content-collapse can otherwise create circular proofs), and semantic equality reified as `EquivFact` edges. This buys lemma reuse across branches and MCTS transposition. **Add when** a workload shows real reconvergence (the same lemma needed in multiple branches, closer to theorem proving) or MCTS measurably wastes time re-deriving transposed states. Prior art for the upgrade: hash-consing (Goto 1974), content-addressed code and Merkle DAGs (Unison, IPFS/IPLD), hashing modulo alpha-equivalence (Maziarz et al., PLDI 2021), congruence closure / EUF (Nelson-Oppen; Downey-Sethi-Tarjan), and e-graphs / equality saturation (egg, egglog). A lighter intermediate step is a state-level canonical serialization for transposition only, without per-node content-addressing.
- **Kernel / vocabulary split and pluggable actions.** Lift the node taxonomy and verbs off a domain-blind graph kernel, with the AND-OR reasoning calculus as one swappable vocabulary and actions as registerable functions. Forward-compat constraint already honored: every verb is a uniform pure function `Graph -> Result[Graph]`.
- **Actions as an evolvable population (meta-learning).** Because actions are uniform pure functions and `reductio` is verifier-anchored, an LLM can generate candidate action types and evolution or hill-climbing can search the population, scored by productivity (verified-fact yield and solve rate per unit cost).

## 12. Non-goals

- No LLM, no MCTS, no reward, no learning, no networking inside `reductio`.
- No content-addressing, hash-consing, statement normalization, semantic-equality edges, or multi-parent DAG sharing in v1 (all deferred, section 11).
- No numeric reward or value field on any node.
- No prompt or action-menu rendering in `reductio` (that is `expert-iteration`'s concern).
- No search (which goal to expand next) in `reductio`; that is MCTS in `expert-iteration`.
- No domain-specific dependency in the core: the `Verifier` is a pluggable protocol, and the SymPy adapter is an optional extra. The core depends only on the standard library.

## 13. Prior-art grounding (v1)

- **AND-OR graphs and AO\*** (Nilsson problem reduction; AO* search; proof-number search): the OR-goal / AND-method structure and the solved-label backup that `reductio` adopts as a derived fixpoint query (section 4), not a search procedure. Search (choosing which goal to expand) lives in `expert-iteration`'s MCTS.
- **Persistent / immutable data structures** (path-copying): every verb returns a new `Graph` sharing the untouched subtrees; states are values, which is what MCTS needs to snapshot and compare.

(The richer content-addressing prior art is recorded in section 11, since it justifies the deferred upgrade, not v1.)
