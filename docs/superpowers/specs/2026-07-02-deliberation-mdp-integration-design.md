# Deliberation MDP: expert-iteration consumes the reductio core (Phase A)

**Status:** design approved in brainstorming (2026-07-02), pending spec review
**Scope:** replace the legacy `llm_context_mdp.py` monolith with an MDP built
on the vendored `verifier_anchored_rl.reductio` core, ending at a working
Rung-1 LLM-as-controller runner. Controller training (pointer head, behavior
cloning, RL fine-tune) is Phase B, a separate cycle.
**Framing (do not regress):** RL first, MCTS second. The verifier is the sole
reward source. `design.md` is the normative reference throughout.

## 1. Goal and shape

Phase A delivers the environment half of `design.md`:

1. A `DeliberationState` wrapping `reductio.Graph` and a turn counter.
2. A `(verb, target)` action space over the six-verb set, with per-verb
   availability masks computed in one place.
3. A lens layer: per-verb scoped views, schemas, and prompt suffixes, with
   statement sanitization and closed-subtree compaction.
4. A verifier adapter porting the legacy five SymPy operations to the
   `reductio.verify.Verifier` protocol, with the vacuous-success hole closed.
5. `DeliberationMDP` satisfying the de-facto environment contract
   (`actions`, `sample_transition`, `is_terminal`, `initial_state()`,
   `depth_limit`).
6. A Rung-1 runner: an LLM controller-teacher picks `(verb, target)` from the
   rendered state; the small frozen generator fills content under the lens;
   the verifier checks; trajectories are logged for Phase B behavior cloning.
7. Deletion of `llm_context_mdp.py`, with its load-bearing behavioral
   invariants re-pinned against the new model and the package namespace
   cleaned of the legacy `Fact`/`Hypothesis`/`VerifierCall` re-exports.

Decisions locked at brainstorm time: the goal-to-fact edge is added to the
vendored core (2), hypothesis-nesting from the legacy model is dropped (a
case split on a hypothesis is re-expressed as an exhaustive-case `SPLIT` of a
goal), `CRITIQUE` is deferred to an ablation, and Phase A ends at the Rung-1
runner with no changes to the heads' architecture.

## 2. Core change: the goal-to-fact edge

The lens must render "the target's facts", but reductio facts currently
attach to nothing. Change to the vendored core (TDD, in
`src/verifier_anchored_rl/reductio/`):

- `Goal` gains `facts: tuple[NodeId, ...] = ()`.
- `compute` gains a `goal_id` parameter:
  `compute(graph, goal_id, statement, premise_ids) -> Result`. It validates
  `goal_id` names a `Goal`, appends the derived `Fact`, and appends its id to
  that goal's `facts` tuple. Given facts created by `new_graph` attach to the
  root goal.
- `lens._children` for a `Goal` becomes `children + candidates + facts`, so
  `scope` reaches a goal's facts and their verifier calls stay findable via
  target ids.
- `serialize.render` is unchanged in shape; facts now appear inside scoped
  views because scope reaches them.
- Affected tests updated; the worked example gains an assertion that a
  computed fact is visible in the scope of its goal.

This is the only Phase-A change to the vendored core. Everything else in
this spec lives outside `reductio/`.

## 3. State and action types

New module `src/verifier_anchored_rl/deliberation_mdp.py` (plus siblings
below), replacing the monolith:

```python
@dataclass(frozen=True)
class DeliberationState:
    graph: reductio.Graph
    turn: int

class Verb(Enum):
    SPLIT = "split"
    FORM_HYPOTHESIS = "form_hypothesis"
    COMPUTE = "compute"
    VERIFY = "verify"
    COMMIT = "commit"
    ABANDON = "abandon"

@dataclass(frozen=True)
class DeliberationAction:
    verb: Verb
    target: NodeId
```

Both are frozen and hashable (MCTS keys decision nodes by state equality;
the outer loop keys visit counts by action equality). The action carries no
free text: all content comes from the LLM inside `sample_transition`, and
`COMMIT`'s witness is selected by the environment (Section 5), keeping the
action space exactly `verbs x live-targets`.

`available_actions(state) -> tuple[DeliberationAction, ...]` computes all
legality in one place (no per-handler re-validation):

- `SPLIT`: target is an open goal (not solved, not abandoned).
- `FORM_HYPOTHESIS`: target is an open goal.
- `COMPUTE`: target is an open goal (the fact attaches to it).
- `VERIFY`: target is a hypothesis or a derived fact, not yet verified.
- `COMMIT`: target is an open goal with at least one discharging witness
  (a candidate hypothesis that is verified, or an attached fact that holds).
  Goal openness lives here, closing the re-commit gap found in review.
- `ABANDON`: target is an open goal or an unverified hypothesis.

The root goal is a valid target like any other; a root `COMMIT` is terminal.

## 4. The lens layer

New module `src/verifier_anchored_rl/lenses.py`:

- `view(state, verb, target) -> str`: `reductio.lens.scope` on the target,
  rendered by `reductio.serialize.render`, prefixed with the problem
  statement. Closed subtrees (solved or abandoned goals other than the
  target's ancestors) render as one line: statement plus verdict.
- `suffix(verb, target, state) -> str`: the verb-specific instruction. Keeps
  the legacy field-list-plus-example prompting trick (empirically better for
  small models than raw JSON Schema).
- `schema(verb, target, state) -> dict`: the JSON schema for constrained
  generation. Content fields only (child statements for `SPLIT`, a statement
  for `FORM_HYPOTHESIS`/`COMPUTE` plus premise ids constrained to live ids);
  `VERIFY`, `COMMIT`, and `ABANDON` need no LLM call at all in this design
  (see Section 5), so they have no schema.
- **Sanitization:** statements written into the graph are cleaned at apply
  time: newlines collapsed to spaces and any substring matching the marker
  pattern `[KIND id=N]` neutralized, so LLM output cannot spoof node lines
  in later renders (the marker-injection hole from review).
- The global embedding string for the encoder path remains one flat string:
  `render(scope(graph, root))` plus a turn marker. `PrefillCache` keys on it
  unchanged.

## 5. Verbs: who calls the LLM, and the two closure verbs

- `SPLIT`, `FORM_HYPOTHESIS`, `COMPUTE`: one constrained LLM generation from
  `view ++ suffix`, parsed against the schema, sanitized, then applied via
  the corresponding pure reductio verb. Parse or generation failure falls
  back to the masked-action step (below).
- `VERIFY`: no LLM call. The environment invokes the verifier adapter on the
  target's own statement (Section 6) via `reductio.actions.verify` with
  `expected=None`. The policy never supplies `expected`; this closes the
  self-verification hole structurally.
- `COMMIT`: no LLM call. The environment selects the witness
  deterministically: the most recently added discharging witness among the
  goal's verified candidates and holding attached facts. If none exists the
  action was unavailable. Root commit computes the terminal reward by
  comparing the witness statement against the environment's ground-truth
  label using the ported `check_answer` logic; subgoal commit is reward-free
  bookkeeping.
- `ABANDON`: no LLM call; `reductio.actions.abandon`.

**Masked-action fallback (kept from legacy):** on LLM failure or an illegal
parsed result, the state advances by `turn + 1` only, with reward
`MASKED_ACTION_REWARD = -0.01`.

## 6. Verifier adapter

New module `src/verifier_anchored_rl/verifier_adapter.py`:

- `MathVerifier` implements `reductio.verify.Verifier`. `check(statement,
  verb, expected)` dispatches on `verb` to the five ported SymPy operations
  from legacy `verifier.py` (`check_arith`, `solve`, `simplify`, `factor`,
  `evaluate`).
- **The vacuous-success hole is closed:** with `expected=None` the adapter
  DECIDES the statement (an equation or relation is checked by SymPy
  simplification; success means the statement is true). Undecidable or
  unparsable statements return `success=False`. It never returns a
  parse-only `True`.
- `check_answer(committed, expected) -> VerifierResult` is kept (SymPy
  equivalence, then normalized-string fallback) and used only by the root
  `COMMIT` handler with the environment's label.
- Phase-A `VERIFY` actions use a single default operation (`check_arith`
  semantics: decide the statement). The per-operation choice returns as an
  action refinement when the pointer head lands in Phase B.
- Legacy `verifier.py` is retired with the monolith; the adapter supersedes
  both it and the vendored `SymPyVerifier` for this MDP (the vendored
  adapter remains in the reductio subpackage for standalone use).

## 7. Reward and termination

Per `design.md` Section 6, terminal-only:

- Root `COMMIT`: `+1.0` if `check_answer` accepts the witness against the
  environment label, else `-1.0`. Terminal.
- `turn >= depth_limit` with no root commit: reward `0.0`. Terminal.
  (Behavioral change from legacy, which forced `-1.0`; `design.md` governs.)
- `VERIFY`: reward `0.0` in v1. The anti-farming gates (per-episode cap,
  dedup per target, tautology rejection) are ported and active as
  *predicates*, with the reward coefficient `INTERMEDIATE_VERIFIER_REWARD =
  0.0` by default so turning the sanctioned shaping channel on is a config
  change, not a code change.
- Root `ABANDON`: terminal with reward `0.0`. Giving up is not a wrong
  answer; scoring it like the timeout keeps a calibrated agent free to stop
  rather than commit a guess it believes is wrong.
- Everything else, including the masked fallback's `-0.01`: as stated above.

`is_terminal(state)`: root goal solved, or root abandoned, or
`turn >= depth_limit`.

## 8. The MDP and the environment contract

`DeliberationMDP` in `deliberation_mdp.py`:

- Constructor: `problem`, `expected_answer`, `givens`, `depth_limit`,
  one generation backend (`llm_client` or `encoder`, as legacy), `verifier`
  (defaults to `MathVerifier()`), `rng` handling as legacy.
- Satisfies `GenerativeMDP` plus the undocumented but load-bearing surface:
  `initial_state() -> DeliberationState` and a `depth_limit` attribute.
- `sample_transition(state, action, rng)`: availability recheck (a stale
  action from a search tree maps to the masked fallback), then the Section-5
  flow, then the Section-7 reward, returning
  `(DeliberationState(graph', turn + 1), reward)`.

Re-pointing (kept minimal, Phase A must leave the full suite green):

- `heads.py` imports `Verb` from the new module; `NUM_VERBS` becomes 6
  automatically. No architectural change.
- `planner_actor.py` and `outer_loop.py` re-point `available_actions` and
  the state-prefix serializer to the new modules; action objects still
  expose `.verb`, so their access pattern is unchanged.
- `trained_prior.py` re-points its serializer import. Experiment scripts
  split by one criterion: classical scripts (gridworld, value iteration,
  expectimax, MCTS, expert iteration) are untouched; scripts that exercise
  the monolith directly move to `experiments/archive/` alongside the pilot
  artifacts already there, since the Rung-1 runner (Section 9) supersedes
  them.

## 9. Rung-1 runner (the Phase-A deliverable)

New module `src/verifier_anchored_rl/llm_controller.py` plus
`experiments/run_llm_controller.py`:

- `LLMController`: prompts a controller-teacher LLM (any `LLMClient`; may be
  a larger remote model) with the full rendered state plus the legal action
  menu (verb, target id, target statement), constrained to emit one
  `(verb, target)` choice by the same constrained-JSON machinery.
- The runner loops: teacher picks the action, `DeliberationMDP` executes it
  (small generator fills content where the verb calls for it), until
  terminal. Each step is logged as JSONL: rendered state, legal actions,
  chosen action, reward, terminal flag. This log is the Phase-B behavior
  cloning dataset, so its format is part of this spec's deliverable.
- The runner reports per-problem outcome, turns used, and LLM calls made
  (the LC row of the paper's pre-registered grid).

## 10. Deletion and invariant re-pinning

`llm_context_mdp.py`, legacy `verifier.py`, and `tests/test_llm_context_mdp.py`
are deleted in the final task, after these behavioral invariants are
re-pinned against the new stack in new tests:

- masked-action fallback on LLM failure (reward `-0.01`, turn advances);
- budget exhaustion terminates with reward `0.0` (note: changed from legacy);
- anti-farming gate predicates fire correctly (cap, dedup, tautology);
- `GenerativeMDP` conformance and state/action hashability (MCTS-safe);
- serializer sanitization defeats marker injection;
- adapter: `check(stmt, verb, None)` decides truth and never returns
  vacuous success; `check_answer` equivalence and fallback behavior.

Package `__init__.py` drops the legacy `Fact`/`Hypothesis`/`VerifierCall`/
`MathState`/`MathAction` re-exports; reductio types are imported from the
subpackage explicitly (no top-level aliasing), ending the name collision.

## 11. Non-goals (Phase A)

- No pointer head, no per-node encoder vectors, no head retraining, no
  behavior cloning, no RL runs (all Phase B).
- No `CRITIQUE` verb (ablation, later).
- No AND/OR aggregation semantics (connective stays non-gating metadata).
- No RAG verb, no GNN encoder work.
- No regression of the classical anchors or the reductio suite.

## 12. Open questions deliberately deferred to Phase B

1. Witness selection at `COMMIT` when several witnesses discharge (Phase A:
   most recent; Phase B may expose it to the pointer head as a second
   target).
2. Per-operation `VERIFY` (solve/factor/simplify) as action refinements.
3. Whether the teacher's action menu biases Rung-1 trajectories in ways
   that matter for behavior cloning (DAgger-style corrections if so).
