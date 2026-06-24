# reductio: an immutable content-addressed AND-OR proof DAG for verifier-anchored reasoning

**Status:** design, pending spec review (2026-06-23)
**Form:** a new standalone repository, `reductio`.
**Boundary:** a pristine core abstraction. No LLM, no MCTS, no reward, no learning. `expert-iteration` imports `reductio` and maps its structural proof-status predicates onto rewards and values on its own side.
**Provenance:** refined from a brainstorming session plus a prior-art-grounded adversarial design review (hash-consing, Unison/Merkle-DAG, hashing-modulo-alpha-equivalence, congruence closure, AND-OR graphs / AO*).

## 1. Purpose

`reductio` provides one thing: an immutable, content-addressed **AND-OR proof DAG** (a "reasoning graph") and a pure **verb algebra** of total functions `State -> Result[State]` over it. The graph is a persistent value: every verb returns a new graph that structurally shares all unchanged nodes, so two graphs compare and hash by the identity of their root ids.

This is the representation the `expert-iteration` action-space-lens design (see `2026-06-23-action-space-lens-design.md`) needs underneath it. That design defined the verbs (`SPLIT`, `FORM_HYPOTHESIS`, `COMPUTE`, `VERIFY`, `COMMIT`, `ABANDON`) and the lens as actions over a "goal tree." `reductio` makes the data structure precise (an AND-OR DAG, not a tree) and gives each verb an exact, invariant-preserving graph semantics. The RL machinery (the `GenerativeMDP` wrapper, reward assignment, MCTS, the heads, expert iteration, the LLM client) stays in `expert-iteration` and consumes `reductio`.

## 2. The state: an immutable AND-OR proof DAG

### 2.1 Two-layer model (identity vs status)

A `reductio` graph is a pair:

1. **The immutable node/edge store** (Layer 1): content-addressed, hash-consed, purely propositional. Nodes here never change; actions only intern new nodes.
2. **The status overlay** (Layer 2): a map keyed by node id, carrying everything mutable (the `abandoned` marker and the `goal_solved_by` commit edges). Actions that change status return a new graph that shares the entire store and updates only the overlay.

```
Graph = (store: ContentAddressedStore, overlay: StatusOverlay)
```

This split is load-bearing. It is what lets `VERIFY`, `COMMIT`, and `ABANDON` change a goal's standing without re-hashing any node (which would fork the node's identity and dangle every inbound edge). The hash covers only the immutable proposition; standing lives in the overlay.

### 2.2 Node taxonomy

Each node kind is a frozen dataclass whose fields are exactly the hashed payload (propositional only, no status):

- **Goal**: a problem or subgoal. Implicitly an **OR** node: solved iff any attached `Method` or discharged candidate `Hypothesis` is solved. Payload: `(normalized_statement, assumption_context_id)`. The root is a `Goal` seeded by the graph constructor. The `assumption_context_id` makes two textually identical goals under different assumptions distinct nodes (see 3.3).
- **Method**: the reified decomposition a `SPLIT` produces. An **AND** node: solved iff all required child goals are solved. Payload: `(connective_tag, canonical_child_goal_ids)`, `connective_tag in {conjunctive, exhaustive_case}`. The exhaustiveness obligation of a case-split is not a tag flag: it is reified as an ordinary `Goal` child (see 2.4).
- **Hypothesis**: a tentative candidate solution to a goal. Payload: `(normalized_statement, parent_goal_context_id)`.
- **Fact**: an established statement. Payload: `(normalized_statement, source in {given, derived}, canonical_premise_ids)`. Note: `verified` is not a `source` value. Verified-ness is a derived query over an incoming successful `VerifierCall`, kept structural, so blessing a fact never re-hashes it.
- **VerifierCall**: a record of a checker invocation result. Payload: `(target_id, verb, normalized_expected, success, normalized_observed)`. Content-addressed as a timeless result ("this check, with this expected, passed/failed"), so it idempotently dedupes. The invocation count (the anti-farming signal) is deliberately not modeled here; it lives in `expert-iteration`'s episode log.
- **EquivFact**: a reified semantic-equality claim linking two distinct nodes, produced only by the verifier. Payload: `(node_id_a, node_id_b, witness_verifiercall_id)`. This is how `x + 0` and `x` come to share a solution without colliding their hashes (see 3.4).

### 2.3 Edge taxonomy

Edges are typed and directed toward dependencies. An edge is identified by `(kind, source_id, target_id)`. All edges live in the immutable store (and are part of the source node's hashed child-id list) except the commit edge, which is an overlay entry.

- `goal_has_method`: Goal -> Method (an OR alternative)
- `method_requires`: Method -> Goal (an AND child; includes the reified exhaustiveness goal)
- `goal_candidate`: Goal -> Hypothesis
- `fact_derived_from`: Fact -> {Fact | Hypothesis} (multi-premise; one edge per premise)
- `verifiercall_checks`: VerifierCall -> node
- `equiv_between`: EquivFact -> {node, node}
- `goal_solved_by`: Goal -> {Fact | Method | Hypothesis} (OVERLAY edge): the typed witness that closes a goal. The witness is type-constrained (see 5, `COMMIT`), never an arbitrary node.

Because store edges are part of each node's hashed child-id list (Merkle-DAG dependencies-by-hash), the whole graph is addressed by its root id(s). The overlay edge (`goal_solved_by`) and the `abandoned` marker are the only mutable state, keyed by id, never hashed.

### 2.4 AND-OR semantics

The graph alternates classically (Nilsson problem-reduction / AND-OR graphs): goals are OR nodes (alternative approaches), methods are AND nodes (all subgoals required), with facts and verified hypotheses as the leaves that discharge goals.

A **case-split is disjunctive in the domain but conjunctive in the proof**: to discharge `G` by cases you must solve every case AND show the cases are exhaustive. So a `SPLIT(exhaustive_case)` produces an AND `Method` whose required children are the case goals plus a distinguished **exhaustiveness `Goal`** (reified as an ordinary `method_requires` child). The uniform AND rule then discharges it with no special case. The genuine OR is a goal carrying more than one candidate method or hypothesis.

## 3. Identity and normalization

### 3.1 Content-addressing (hash-consing, syntactic)

Node identity is **content-addressed, syntactic only**, in the Unison / Merkle-DAG tradition. A node id is a strong wide hash (>= 256 bits; SHA-256 baseline, BLAKE2b acceptable) over a deterministic tuple:

```
id(node) = H( kind_tag, normalizer_version, normalized_payload, assumption_context_id, tuple_of_canonical_child_ids )
```

Nodes are interned bottom-up through a hashcons table, so a child's id is already its canonical content, recursively. Consequences obtained for free:

1. Structural equality and hashing are O(1) id comparisons.
2. The whole graph is addressed by its root id, giving a stable transposition key. (This is justified as intrinsic to immutable value semantics: a value should be comparable regardless of any consumer. Downstream MCTS transposition is then an incidental benefit, not a design driver.)
3. Identical subgoals collapse to one shared multi-parent node, giving free lemma reuse.

Hard rules:
- (a) The hashed payload is purely propositional and never includes verification, solved, abandoned, or bless state. Status changes cannot fork identity (the overlay handles status).
- (b) The hash addresses syntactic form only, documented explicitly. Semantic equality is handled by reified `EquivFact` edges, never by collisions.
- (c) A strong wide hash plus the stored canonical form, so an id-equality can be confirmed by content comparison. A weak/truncated hash equality is never treated as identity.
- (d) Normalization is deterministic and versioned (a `normalizer_version` byte is folded into the hash), so the same statement never hashes two ways across SymPy versions or dict/set iteration order. This preserves persisted transposition stability.
- (e) Acyclicity is a precondition (4), required because bottom-up interning is only well-defined on a DAG.

### 3.2 Normalization pipeline

"Same statement" is defined **syntactically modulo normalization** for v1, stated plainly so no one expects free semantic lemma reuse. Normalization is a **pluggable `Normalizer` protocol** (`canonical_key(statement) -> bytes`, required total, deterministic, and versioned). The core ships no domain normalizer and depends on no domain library; one must be supplied (a trivial string-identity normalizer suffices for tests and for non-math domains). The reference **math normalizer** is a SymPy adapter living outside the core (section 9), so the AND-OR DAG, content-addressing, verb algebra, derived status, and lens stay domain-agnostic. The math adapter concentrates all the "what is identical" difficulty into one deterministic, total function applied before hashing, operating on structure, never on rendered text (no regex on statements):

1. **Parse** to a SymPy AST (reuse the `sympify` path; never string-munge).
2. **Alpha-normalization**: rename bound variables (quantifiers, integrals, summation indices, lambdas) to De Bruijn / positional indices, so `forall x. P(x)` and `forall y. P(y)` hash identically. (Unison does exactly this; Maziarz et al., "Hashing Modulo Alpha-Equivalence", PLDI 2021, is the efficient general algorithm and the documented fallback if hashing open subterms ever bottlenecks. For statement-sized closed terms, plain De Bruijn then hash suffices.)
3. **AC-normalization**: sort operands of commutative/associative operators into a canonical order, and fold rational/numeric literals, so `a + b` and `b + a` collapse. The hash-consing literature is explicit that this is what keeps hit rate high and prevents DAG fragmentation.
4. **Emit** a deterministic SymPy canonical structural key (for example `srepr` of the AC-and-alpha-normalized form) as the normalized payload.

Normalization is **non-heuristic and total**: it never calls `simplify`, `factor`, or `solve`, because those are undecidable, version-dependent, and non-transitive. (The existing `verifier_anchored_rl/verifier.py` decides equality with exactly that heuristic, which is precisely why it must stay out of the hash.)

### 3.3 Assumption-context in goal identity

A bare statement is not enough to identify a goal. `x > 0` under hypothesis `H1` is not the same obligation as `x > 0` under `H2`; sharing them would let a solution discharged under one parent's assumptions silently discharge the other. So a `Goal`'s (and `Hypothesis`'s) hashed payload includes an `assumption_context_id`: the hash of the normalized set of in-scope hypotheses and given facts.

**Rule:** two goals are the same node iff `(normalized_statement, normalized_in_scope_assumption_set)` are equal. Context-free lemmas still collapse (free reuse); cross-context discharge is prevented.

### 3.4 Semantic equality as `EquivFact`, not collision

Semantic equality (`x + x = 2x`, `sin^2 + cos^2 = 1`) has no computable canonical representative, so it cannot live in the hash. It is promoted to a first-class, verifier-produced `EquivFact` edge between two distinct, separately-hashed nodes. Lemma-reuse-by-meaning is then an auditable verifier claim, not a silent hash collision.

If semantic sharing must eventually auto-induce structural sharing, escalate in order, kept strictly orthogonal to the proof DAG: (i) richer deterministic normalization, (ii) a persistent congruence-closure / EUF layer over verified equalities (Nelson-Oppen; Downey-Sethi-Tarjan), (iii) a full e-graph / egglog layer. An e-class ("statements equal") and an OR-goal ("methods might solve") are different relations and are never conflated.

**Pinned future operation (out of v1).** A verb that produces `EquivFact`s on demand (a `RELATE` / `CANONICALIZE` operation, driven by the pluggable `Verifier`, or later by an LLM) is a natural extension. Because it would go through the `Verifier` / `Normalizer` protocols rather than calling a domain library directly, it adds semantic reuse without specializing the core to any one domain. Pinned for now.

## 4. Invariants

1. **Acyclicity (enforced precondition).** Before any dependency edge is added (`method_requires`, `fact_derived_from`, `goal_solved_by`), the target must not be an ancestor of the source in the current DAG; otherwise the verb is rejected and returns the unchanged graph plus an error. This guard is what makes content-address collapse safe: a subgoal that normalizes to an ancestor is refused, not silently turned into a circular proof ("A by A"). One ancestor-check guards `SPLIT`, `COMPUTE`, and `COMMIT` uniformly.
2. **Immutable propositional identity.** A node's content-hash covers only `(kind, normalized statement, assumption_context, canonical child ids)`. Status is never hashed. Enforced by frozen dataclasses with only propositional fields; no verb mutates a node payload.
3. **Bottom-up internment / maximal sharing.** Every node is interned after its children; identical content yields one shared node; the graph is addressed by its root id(s). Normalization is total and versioned, so the same statement always hashes identically across runs and SymPy versions.
4. **Context-scoped goal identity.** Two goals share a node iff their `(statement, in-scope assumptions)` are equal (3.3).
5. **Derived-status fixpoint (no pushed flags).** `solved(Goal) = OR over its methods/discharged candidates of solved(.)`; `solved(Method) = AND over required goals of solved(.)`; `solved(verified Fact) = true`. Computed bottom-up, memoized, order-independent. `abandoned` is an overlay annotation; an AND-Method with an abandoned required child has `solved() = False` while the subtree is retained.
6. **Typed COMMIT witness.** `goal_solved_by` targets are constrained to a verified `Fact`, an all-required-solved `Method`, or a discharged `Hypothesis`; never an arbitrary node.
7. **Exhaustiveness reified.** An `exhaustive_case` `Method` is solved iff all required goals including the reified exhaustiveness `Goal` are solved (uniform AND rule, no special tag).
8. **VerifierCall result-idempotence.** `VerifierCall` nodes dedupe by content; the graph never records invocation counts (anti-farming call-counting belongs to the RL layer).
9. **Structural status reporting only.** `reductio` exposes `solved`/`unsolved`/`abandoned`/`verified` as structural predicates, never a numeric reward or value.

## 5. The action algebra

Each verb is a total function `Graph -> Result[Graph]` (`Result` is success-with-new-graph or rejection-with-error-and-unchanged-graph). The root `Goal` and given `Fact`s enter via a **graph constructor** in `graph.py`, outside the verb algebra.

`COMPUTE` (derive), `VERIFY` (check / bless), and `COMMIT` (close) are kept orthogonal: no verb does two of these.

- **`SPLIT(graph, goal_id, connective, child_statements)`**
  - Effect: interns a new `Method` (connective in `{conjunctive, exhaustive_case}`) plus one new `Goal` per child statement (each inheriting/extending the parent's assumption-context), wires `goal_has_method` and `method_requires`. For `exhaustive_case`, additionally interns and wires a distinguished exhaustiveness `Goal` as one more required child.
  - Pre: `goal_id` is an existing `Goal`; `connective` valid; child statements non-empty. Ancestor-check: no interned child goal may be an ancestor of `goal_id`.
  - Invariants: acyclicity by the ancestor-check; bottom-up internment (children before the method); conjunctive child order is canonicalized (sorted by child id), so the same split in any order yields the same `Method` id; `exhaustive_case` preserves case order if semantically meaningful. No status mutation.

- **`FORM_HYPOTHESIS(graph, goal_id, statement)`**
  - Effect: interns a `Hypothesis` (statement + goal's context), wires `goal_candidate`.
  - Pre: `goal_id` is an existing `Goal`. No ancestor-check (a leaf candidate adds no dependency edge).
  - Invariants: content-addressed, so forming the same hypothesis twice is a graph no-op. No status mutation.

- **`COMPUTE(graph, statement, premise_ids)`**
  - Effect: interns a derived `Fact` (`source = derived`, payload includes statement + canonical premise ids), wires `fact_derived_from`. Does not bless or close anything.
  - Pre: every premise exists and is a `Fact` or `Hypothesis`. Ancestor-check: the new fact must not be an ancestor of any premise.
  - Invariants: acyclicity by ancestor-check; fact interned after premises; never sets solved/verified status (keeps derive -> check -> close clean).

- **`VERIFY(graph, node_id, verb, expected)`**
  - Effect: invokes the supplied `Verifier` (the math adapter is SymPy; the core depends on no domain library) on the node's statement with `(verb, expected)`. Interns a `VerifierCall` recording `(target, verb, expected, success, observed)`, wires `verifiercall_checks`. On success it also interns a verified `Fact` carrying the node's statement (premise = the checked node) for use as a typed `COMMIT` witness, and may intern an `EquivFact` when the verifier establishes an equality. Verified-ness is thereafter a derived query over the incoming successful `VerifierCall`, not a stored flag.
  - Pre: `node_id` exists and carries a statement; `verb` in the verifier's enum; `expected` well-formed.
  - Invariants: identity stability (never mutates the checked node, so its id is unchanged and no inbound edge dangles); idempotent (re-verifying with the same `(verb, expected)` yields the same `VerifierCall` id, graph unchanged; no count recorded here); acyclicity (call/fact interned after their targets).

- **`COMMIT(graph, goal_id, witness_id)`**
  - Effect: adds an overlay `goal_solved_by(goal_id -> witness_id)` edge. Does not recompute or store any aggregate; solved-ness everywhere is queried as a derived fixpoint.
  - Pre: `goal_id` is a `Goal`. `witness_id` is type-constrained: either (a) a `Fact` that is verified (incoming successful `VerifierCall`) whose statement matches the goal's (possibly via an `EquivFact` bridge), or (b) a `Method` whose required goals are all derived-solved, or (c) a discharged `Hypothesis`. An untyped or unverified witness is rejected. Ancestor-check: the witness must not be an ancestor of `goal_id` (refuses "A proved by A").
  - Invariants: acyclicity by ancestor-check; no identity fork (writes only the overlay); order-independence (solved-ness is derived, so multi-parent and grandparent propagation is automatic and order-free).

- **`ABANDON(graph, node_id)`**
  - Effect: sets `overlay.status[node_id] = abandoned`. The node and all its edges are retained (store untouched).
  - Pre: `node_id` exists.
  - Invariants: identity stability (only the overlay changes); derived semantics (an AND-Method with an abandoned required child has `solved() = False`, subtree retained; an OR-Goal can still be solved by a non-abandoned alternative). `ABANDON`, `VERIFY`-status, and `COMMIT`-status are uniform overlay updates.

## 6. Derived predicates

`solved`, `abandoned`, and `verified` are queries, not stored flags, computed as a memoized fixpoint over the AND-OR DAG (5 above). This is the AO* / solved-label backup pattern from AND-OR graph search, applied as a pure query rather than a search: a node's status is read off its children. Memoization is keyed by node id (sound because nodes are immutable); the overlay is the only thing that can change a derived value, so the memo is invalidated per overlay version.

## 7. Lens and serialization

The LLM/generation boundary is kept clean by splitting concerns:

- **`lens.scope(graph, node_id) -> Subgraph`** (in `lens.py`): pure structural scope. Given a focus node, return a scoped immutable sub-DAG: the node's method/subgoal neighborhood (one or more levels), its candidate hypotheses, its derived/verified facts, the ancestor path to the root, and the in-scope assumption set. `scope` knows nothing about prompts, action menus, token budgets, or how generation is produced. It is pure `graph -> graph`.
- **`serialize.render(subgraph) -> str`** (in `serialize.py`): a fixed, deterministic, neutral textual format (structured markers for node kind, id, statement, status, edges), plus a `parse` back where round-tripping is useful. It knows only this fixed format, never prompts or per-verb action menus. (The existing repo's identifier/target-referencing-verb menu is prompt-shaping and lives in `expert-iteration`, not here.)

"Generation writes back locally via the verbs" means `expert-iteration` reads ids from the lens and calls `reductio`'s pure verbs. `reductio` never imports or models generation.

## 8. The Verifier and Normalizer protocols

The core defines two structural-subtyping protocols and depends on neither SymPy nor any other domain library:

- `Normalizer`: `canonical_key(statement) -> bytes` (section 3.2), total, deterministic, versioned.
- `Verifier`: checks a statement and produces a `VerifierResult`, and when an equality is established, the witness for an `EquivFact`. The verifier is the only source of `verified` status and of `EquivFact` edges.

A domain supplies both. The reference **math adapter** implements them with SymPy, mirroring the existing `verifier_anchored_rl/verifier.py` logic without importing it (and without importing any RL concept). Keeping normalization and verification behind protocols is exactly what stops SymPy from specializing the core to mathematics: `reductio` is a general verifier-anchored reasoning-graph substrate, and math is one adapter. A non-math domain (for example code-with-unit-tests, or a logic with a proof checker) supplies its own `Normalizer` and `Verifier` and reuses the entire graph, action, status, lens, and serialization core unchanged.

## 9. Module layout

```
reductio/
  pyproject.toml        # core deps: standard library only (hashlib for hashing). Dev: pytest.
                        # Optional extra [math]: sympy, for the math adapter only. No torch, LLM, MCTS, networking anywhere.
  README.md
  src/reductio/
    __init__.py         # public API: Graph constructor, verbs, Normalizer + Verifier protocols, lens.scope, serialize. Re-exports only.
    graph.py            # frozen node dataclasses (propositional only), typed edges, hashcons table + bottom-up interning,
                        # the (store, overlay) Graph value, root+givens constructor, ancestor-check, derived solved/abandoned queries, id = H(...).
    normalize.py        # the Normalizer protocol + a trivial string-identity normalizer. No domain library, no heuristic simplify/factor/solve.
    actions.py          # the verb algebra (total Graph -> Result[Graph]); preconditions (ancestor-check, witness typing) live here.
    verify.py           # the Verifier protocol (structural subtyping). No implementation in the core.
    lens.py             # pure scope(graph, node_id) -> Subgraph only.
    serialize.py        # render(subgraph) -> str fixed-format + parse-back; neutral text only.
    adapters/
      __init__.py
      sympy_math.py     # the math adapter: a SymPy Normalizer (parse -> De Bruijn -> AC-sort + literal fold -> srepr key)
                        # and a SymPy Verifier. The ONLY place SymPy lives; imported only with the [math] extra.
  tests/                # core suites use the trivial normalizer + a mock verifier; the math adapter has its own suite (incl. the worked example).
  docs/
```

## 10. Testing strategy

- **Normalization** (`normalize.py`): idempotence (`normalize(normalize(s)) == normalize(s)`); stability across runs and SymPy versions (golden-hash fixtures pinned to `normalizer_version`); alpha-invariance (`forall x. P(x)` and `forall y. P(y)` hash equal); AC-invariance (`a+b` and `b+a` equal; `a*b*c` permutations collapse); non-heuristic (`x+x` and `2*x` hash differently in v1, documenting syntactic identity).
- **Content-addressing** (`graph.py`): identical statements intern to one node; shared subgoals are one multi-parent node; the whole-graph root id is a stable key; context-distinct goals (`x>0` under `H1` vs `H2`) intern to distinct nodes; the wide-hash collision path confirms by stored content.
- **Acyclicity enforcement** (`actions.py`): every dependency-edge verb rejects an edge whose target is an ancestor of the source, returning the unchanged graph plus an error; a circular-proof attempt is refused.
- **Identity stability under status change**: `VERIFY`, `COMMIT`, `ABANDON` never change any node id; re-`VERIFY` with the same `(verb, expected)` leaves the graph unchanged; no invocation count appears in the graph.
- **Derived-status fixpoint**: the AND/OR rules; multi-parent propagation is order-independent (commit children in different orders yields identical derived status); an AND-Method with an abandoned required child is `solved() = False` with subtree retained; `exhaustive_case` solved iff all cases AND the reified exhaustiveness goal solved.
- **Verb lifecycle typing**: `COMMIT` rejects an unverified or untyped witness; the `FORM_HYPOTHESIS -> VERIFY -> COMMIT` path closes a goal; `COMPUTE` alone never closes a goal.
- **Semantic-equality-as-edge**: a verifier-established equality (`x+0 = x`) produces an `EquivFact` between two distinct nodes (not a collision); `COMMIT` can bridge a goal to a verified fact via that `EquivFact`.
- **Persistence / structural sharing**: a verb returns a new graph sharing all untouched nodes (object identity of unchanged subtrees); graphs are hashable and comparable as values.
- **Boundary**: the `reductio` core imports with no torch/LLM/MCTS/networking modules and no SymPy (SymPy is reachable only via the `[math]` adapter); no node has a reward/value field; `lens.scope` output contains no prompt/action-menu strings.
- **Serializer round-trip**: `render(parse(text)) == text` on the fixed format for representative subgraphs.
- **Worked example**: the `|x-3| + |x+2| = 7` problem builds the expected AND-OR DAG (root goal, an `exhaustive_case` Method with three case goals plus an exhaustiveness goal, two verified facts, one abandoned case, a root commit), matching the trace from the action-space-lens design.

## 11. Prior-art grounding

- **Hash-consing** (Goto 1974; Zhu et al. 2025): construction-time interning into a maximally-shared immutable DAG; child ids are the canonical content of the parent; O(1) structural equality; pre-hash AC-normalization to raise hit rate. This is `reductio`'s store.
- **Content-addressed code / Merkle DAGs** (Unison; IPFS/IPLD): hash the normalized AST not surface syntax; dependencies-by-hash; immutability with ancestor propagation; free multi-parent sharing. Unison confirms syntactic (not semantic) identity and the alpha-rename-binders-to-positional normalization.
- **Hashing Modulo Alpha-Equivalence** (Maziarz et al., PLDI 2021): the efficient general algorithm for alpha-correct hashing of terms with binders; the documented fallback if open-subterm hashing bottlenecks. For closed statement-sized terms, plain De Bruijn then hash suffices.
- **Congruence closure / EUF** (Nelson-Oppen; Downey-Sethi-Tarjan) and **e-graphs / equality saturation** (egg, egglog): the principled route from syntactic identity to semantic identity, deferred and kept orthogonal (3.4).
- **AND-OR graphs and AO\*** (Nilsson, problem reduction; AO* search; proof-number search): the OR-goal / AND-method structure and the solved-label backup that `reductio` adopts as a derived fixpoint query (6) rather than a search procedure. Search itself (choosing which goal to expand) lives in `expert-iteration`'s MCTS, not in `reductio`.

## 12. Boundary (what stays in expert-iteration)

`reductio` reports only proof status (`solved`/`unsolved`/`abandoned`/`verified`) as structural predicates. `expert-iteration` maps status to reward and value on its side (for example: terminal reward when the root goal becomes solved; a gated, capped intermediate signal when a new verified `Fact` appears). The `GenerativeMDP` wrapper, MCTS, the heads, expert iteration, the LLM client, the prompt/action-menu rendering, and the anti-farming verifier-call counting all stay in `expert-iteration`. `reductio` has no numeric reward or value field anywhere, and the verifier-call count is never in the graph.

## 13. Open questions (to resolve during implementation, not blockers)

1. **Assumption-context granularity**: is a goal's in-scope assumption set the full transitive ancestor set, or only locally-introduced assumptions? Full-transitive is sound but reduces sharing; needs a precise inherit-vs-introduce rule for sub-`SPLIT`.
2. **Discharged-Hypothesis promotion**: the exact typed transition by which a verified `Hypothesis` becomes a `COMMIT`-eligible witness (does `VERIFY` mint a `Fact` from the hypothesis, and does `COMMIT` take the `Hypothesis` or the minted `Fact`?).
3. **`EquivFact` and `COMMIT` bridging**: the precise rule for closing a goal by a `Fact` that is only `EquivFact`-equal (not normalize-equal) to the goal, including whether bridging is transitive and the non-transitivity risk of chained heuristic equalities.
4. **SymPy version drift**: `srepr` is stable but not guaranteed across major SymPy releases; decide whether to pin SymPy, snapshot a canonical-key spec, or add a re-hash migration for persisted graphs.
5. **De Bruijn sufficiency**: whether plain De Bruijn then hash covers all statement shapes, or whether the Maziarz et al. algorithm is needed for open subterms in practice.
6. **Hashcons GC**: weak-value table (a la JuliaSymbolics) to bound table growth once persisted MCTS trees exist, versus a strong table for stable transposition within an episode.
7. **Serializer format**: exact marker syntax, id rendering, status projection, and whether parse-back must be lossless for the overlay or only for the store.

## 14. Non-goals

- No LLM, no MCTS, no reward, no learning, no networking inside `reductio`.
- No semantic equality inside the hash (it is a reified `EquivFact` edge).
- No heuristic `simplify`/`factor`/`solve` inside normalization.
- No numeric reward or value field on any node.
- No prompt or action-menu rendering in `reductio` (that is `expert-iteration`'s concern).
- No search (which goal to expand next) in `reductio`; that is MCTS in `expert-iteration`.
- No domain-specific dependency (SymPy or otherwise) in the core: `Normalizer` and `Verifier` are pluggable protocols, and the math (SymPy) adapter is an optional extra. The core depends only on the standard library.
