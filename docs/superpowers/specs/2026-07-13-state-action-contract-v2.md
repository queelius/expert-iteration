# State and action contract v2

Date: 2026-07-13

Status: implemented through runnable collection, strict replay, replay-sealed
dataset conversion, grouped splitting, the exact legal-menu comparator,
behavior-cloning plus Monte Carlo warm-start, manifest-bound checkpoints, and
checkpoint-backed learned control. This includes the immutable typed core,
casework symbolic adapter, canonical observation, deterministic closure,
one-turn macro transition, typed generator lenses, bound teacher controller,
exact environment, collection, and transport contracts, canonical run writer,
role-bound raw-call diagnostics, replay from a trusted problem source,
value-sensitive dataset and checkpoint contracts, deterministic hash and
zero-text encoders, and a four-verb factored policy/Q network with joint
legal-pair decoding. The E23 hosted integration gate and decomposition
diagnostics through E29 are complete, but the representation gate remains
uncleared and none of their artifacts is a usable training collection. No v2
checkpoint exists. Nonmutating strict schema validation is implemented and
bound by current role-local generator and teacher wire contracts. Canonical
live and diagnostic replay paths derive the strict mode from those ids; prior
wire ids select legacy coercion only for exact historical replay. The next
representation gate, fresh training data, execution of the implemented
warm-start, and the first off-policy Q-learning stage remain pending. The
narrow v1.1 repairs remain a reproduction and diagnostic path.

## Framing and invariant

This remains an RL-first system. MCTS is a strictly additive accelerator,
never the spine. A frozen LLM generates content and a small external
controller learns the policy and action values.

The symbolic verifier is the only reward source. VERIFY, COMMIT, deterministic
ready-AGGREGATE, and PRUNE never call the LLM. No reward model, learned
critic-as-judge, or LLM judge is introduced by this contract.

A selected legal action whose generation, payload conversion, action algebra,
or symbolic check rejects receives reward 0. Its cost is the consumed turn
under the finite horizon. An out-of-menu pair is an integration fault and does
not become a canonical transition. The archived $-0.01$ mask penalty is
available only through the named `legacy-mask-penalty-v1` reproduction contract
and is forbidden for new training.

## Why v1 is not a durable MDP

The Phase B audit found five contract failures:

1. A rejected immutable claim remained VERIFY-legal under the same
   deterministic verifier context. The resulting action changed only the turn.
2. Descendants of solved or abandoned goals remained legal while root-state
   compaction hid their content.
3. A compacted solved goal retained only the word `solved`, not the committed
   result needed by its parent.
4. `COMMIT(goal)` hid which verified witness the environment would select.
   Adding a newer verified claim could make an older correct claim
   unselectable.
5. Teacher, replay, and live learned inference did not consume one canonical
   observation. Replay reconstructed status from rendered text, including
   model-authored substrings.

Two controller defects compounded these failures. Per-node features omitted
owner and provenance, making some distinct targets exactly symmetric. Greedy
deployment selected the largest verb marginal and then a target, which need
not maximize the factored joint pair probability.

## Required invariants

The next contract must satisfy all of these mechanically:

- Every legal target has one complete structured record in the controller
  observation.
- A legal target is active. Every goal ancestor is open.
- Deterministic VERIFY has terminal claim status:
  `unchecked -> verified | rejected`.
- A solved goal's compact record includes the exact committed claim id,
  kind, payload, and statement.
- Every learned choice is explicit in the action. Deterministic COMMIT closure
  records the exact Claim it commits; no witness is selected silently.
- A stable policy state contains no unchecked, committable, or ready-aggregate
  work. Closure drains it before the next controller decision.
- Status, provenance, and ownership come from graph fields, never from parsing
  authored text.
- Two targets with different transition consequences have distinguishable
  structured features.
- Greedy B=0 deployment maximizes one score over all legal pairs.
- Generation transport and schema retries are internal to generation.
  Semantic rejections become bounded, observable state.
- Schema-validation behavior must be explicitly versioned. Exact-type
  validation is nonmutating. Numeric-string coercion exists only under its
  named legacy contract for historical replay and is forbidden for new
  collection. `typed-policy-generation-v2.1` and
  `controller-choice-wire-v2.1` bind exact validation for the current model
  roles. Their prior `v2` ids select the named legacy behavior only for exact
  replay. Runtime and diagnostics derive the mode from these role-local ids and
  reject unknown bindings before model execution or replay.
- Goal obligations and their digests contain no answer or solution-set label.
- Claim payload is authoritative; every statement is a deterministic render of
  typed fields.
- Every replay transition contains validated current and next observations.
- Observation schema and policy-input contract are distinct, independently
  versioned fields.

## Parallel v2 core boundary

V2 is a parallel core, not an in-place mutation of `reductio` or the v1
`DeliberationMDP`. It has its own immutable Graph, State, Action, verifier,
observation builder, transition logger, replay converter, and checkpoint
loader. V1 and `controller-observation-v1.1` remain frozen for E18-E22
reproduction. A v1 graph, trajectory, or checkpoint is never relabeled or
silently upgraded to v2.

The two cores may share transport and narrowly scoped symbolic parsing
utilities. They do not share action enums, legality functions, node classes,
serialization constants, or checkpoint defaults. V2 becomes eligible for
collection only after its graph, verifier, action, observation, transition,
diagnostic, feature, and checkpoint invariants pass together. These boundaries
are implemented in the parallel core. Hosted collection first resolves source,
environment, and collection digests without network access. Execution requires
a preregistration id and those exact expected digests. Incomplete runs remain
under explicit `.partial` paths; final paths appear only after strict replay
links every role-bound raw call and decoded payload to its canonical row and
the complete ordered source is present.

## Trusted goal obligation

Every Goal owns an immutable `GoalObligation` with exactly four fields:

```text
GoalObligation:
  variable: str
  equation: str
  domain: str
  objective: branch | sum | max | min | identity
```

The strings are canonical symbolic expressions accepted by the pinned parser,
not natural-language text. The root obligation is created by a trusted
casework adapter, never by the LLM. For casework-v1 the adapter reads the
symbolic equation and the domain information needed to represent explicit
problem constraints, validates the single variable, and maps the public
question type as follows:

| casework question type | root objective |
| --- | --- |
| `sum_of_solutions` | `sum` |
| `largest_solution` | `max` |
| `smallest_solution` | `min` |
| `find_x` | `identity` |

The adapter must not read, copy, hash, or retain `answer` or `solution_set`.
Those labels remain outside the MDP for post-hoc dataset auditing only. The
adapter never places them in verifier context, environment manifests, prompts,
observations, or transition rows. Root reward is produced only by the symbolic
owner-discharge result. An external label comparison may audit the harness but
cannot change reward. The adapter also must not pass the full
`difficulty.params` object through as trusted state, because several families
store answer-derived fields there. A problem whose
equation, variable, domain, or objective cannot be constructed without a label
is rejected at the adapter boundary.

DECOMPOSE proposes child obligations with the same variable and equation,
symbolically restricted child domains, and objective `branch`. Apply-time
symbolic validation requires the child domains to be pairwise disjoint and to
cover the parent domain exactly. Invalid or incomplete partitions are rejected
before any child Goal is added. The LLM may propose domains; it cannot certify
their coverage.

## Canonical state and payloads

```text
Goal:
  id
  parent_goal
  statement
  obligation: GoalObligation
  assumptions
  connective
  status: open | solved | pruned
  committed_claim

Evidence:
  id
  scope_goal
  statement
  premises
  source: trusted_given
  committable: false

Claim:
  id
  owner_goal
  kind: result | solution | derivation | aggregate | refutation
  payload
  premises
  statement
  status: unchecked | verified | rejected | pruned
  verifier_result
  discharge_result
  discharge_status: unchecked | accepted | rejected

Attempt:
  action
  outcome_code
  rejection_reason
  verifier_context
```

The payload is a closed, kind-indexed union. Each payload is a flat JSON object
with exactly these fields:

```text
result:     {objective: sum | max | min | identity, value: str}
solution:   {variable: str, value: str}
derivation: {relation: eq | lt | le | gt | ge, lhs: str, rhs: str}
aggregate:  {reducer: sum | max | min | identity, result: str}
refutation: {mode: unique_candidate_outside_domain, variable: str, candidate: str}
```

Premise ids live only on Claim. They are not duplicated inside payloads.
Expression strings are parsed and re-serialized canonically before storage.
Unknown keys, missing keys, duplicate premises, non-canonical expressions, and
a payload whose type does not match Claim.kind are apply errors. There is no
free-form certificate field.

Payload is authoritative and Claim.statement is derived. The LLM never writes
statement independently. The renderer is deterministic:

```text
result:     result(<objective>) = <value>
solution:   <variable> = <value>
derivation: <lhs> <op> <rhs>
aggregate:  <reducer>(committed child outcomes) = <result>
refutation: <variable> = <candidate> is outside the owner domain
```

The fixed relation map is `eq -> =`, `lt -> <`, `le -> <=`,
`gt -> >`, and `ge -> >=`. For example, a solution payload
`{variable: x, value: 5}` renders as `x = 5`; an aggregate payload
`{reducer: sum, result: 8}` renders as
`sum(committed child outcomes) = 8`. Deserialization recomputes statement and
rejects a mismatch. This prevents a truthful payload from carrying a spoofed
or contradictory display string.

`verified` means the kind-specific symbolic validity contract passed.
`discharge_status` is a separate owner-specific symbolic result. A verified
claim can remain ineligible for COMMIT. Trusted givens are Evidence, not
candidate outcomes, and are never COMMIT targets.

## Verification and discharge

VERIFY invokes one fixed route selected by Claim.kind. Its verifier context is
a digest of the verifier contract, purpose, kind, owner id and obligation,
canonical payload, and ordered premises. The same immutable context can be
checked only once. A failed validity check sets Claim.status to `rejected` and
leaves discharge `unchecked`; a passed validity check sets status to
`verified` and records a separate accepted or rejected discharge result.

The initial casework contract is:

| Claim kind | Symbolic validity | Owner discharge |
| --- | --- | --- |
| result | `objective` equals the non-branch owner objective; `value` parses; the verifier independently computes the complete finite in-domain solution set and its exact objective reduction | accepted exactly when `value` equals that reduction; no dataset answer or solution-set label is read |
| solution | `variable` equals the owner variable; `value` parses, lies in the owner domain, and satisfies the owner equation | accepted only when the verifier establishes that the complete in-domain solution set is exactly `{value}` and the owner objective is `branch` or `identity` |
| derivation | the typed relation over `lhs` and `rhs` is symbolically entailed by active Evidence, verified Claim premises, and the owner obligation | always rejected; a derivation is useful evidence but never closes a Goal |
| aggregate | `result` is recomputed by applying `reducer` to numeric outcomes of the committed child Claim premises; refutation outcomes contribute no numeric value | accepted only when premises name the committed Claim of every direct child, every child outcome is verified and discharging, the stored decomposition is an exact domain cover, reducer equals the owner objective, and result is exact |
| refutation | mode is `unique_candidate_outside_domain`, variable equals the owner variable, candidate satisfies the owner equation, and candidate is outside the owner domain | accepted only when the verifier additionally establishes that the owner equation has no solutions in the owner domain |

The aggregate premise set includes committed refutation claims so every direct
child remains auditable, even though those claims contribute no numeric value.
A merely pruned Goal is not evidence of an empty branch and cannot satisfy
aggregate coverage. PRUNE is search cleanup, not a symbolic refutation.

This separation blocks three shortcuts: a true but goal-irrelevant identity
cannot discharge its owner, one out-of-domain number cannot refute a branch
that has another live solution, and a numerically correct reduction cannot
hide an omitted child.

## Action algebra and learned boundary

The full typed environment algebra is:

```text
DECOMPOSE(goal)
PROPOSE_RESULT(goal)
PROPOSE_SOLUTION(goal)
DERIVE(goal)
AGGREGATE(goal)
PROPOSE_REFUTATION(goal)
VERIFY(claim)
COMMIT(claim)
PRUNE(goal_or_claim)
```

Claim kind is action-visible because it selects a different schema and
symbolic verification route. Hiding it in generator output would recreate the
credit blur that promoting the target was meant to remove.

- DECOMPOSE proposes owner-scoped child GoalObligations. Apply-time symbolic
  validation, not the LLM, certifies their exact domain partition.
- PROPOSE_RESULT creates an owner-objective Result Claim. It intentionally
  allows the generator to answer sum, max, or min directly without first
  decomposing. The verifier still recomputes the complete solution set.
- PROPOSE_SOLUTION creates a Solution Claim from exactly `variable` and
  `value`.
- DERIVE creates a typed relation Claim with explicit Claim.premises.
- AGGREGATE creates a reducer/result Claim over the exact committed child
  outcomes named by Claim.premises. Its owner must have an exhaustive-case
  decomposition.
- PROPOSE_REFUTATION creates a `unique_candidate_outside_domain` Claim for one
  candidate. It does not assert emptiness merely by being well formed.
- VERIFY runs the validity route and, only after validity succeeds, the
  separate owner-discharge route. It is unavailable after the immutable
  context is accepted or rejected.
- COMMIT targets one verified, owner-discharging Claim and closes exactly that
  Claim's owner Goal. It is terminal exactly when the owner is root.
- PRUNE is irreversible, cascades inactivity through descendants, and is
  unavailable at the root. It provides no symbolic evidence and cannot stand
  in for a refutation.

The serialized action semantics id is `claim-closure-v2`. The full lowercase
algebra order is fixed as:

```text
decompose, propose_result, propose_solution, derive, aggregate,
propose_refutation, verify, commit, prune
```

The initial casework controller learns only this fixed policy order:

```text
decompose, propose_result, propose_solution, propose_refutation
```

Its objective-specific menu is derived only from the public trusted
obligation, never hidden feasibility or answer labels:

| Goal objective | Learned choices |
| --- | --- |
| `sum`, `max`, `min` | PROPOSE_RESULT; DECOMPOSE while undecomposed |
| `identity` | PROPOSE_SOLUTION; DECOMPOSE while undecomposed |
| `branch` | PROPOSE_SOLUTION; PROPOSE_REFUTATION |

DERIVE remains in the generic typed core, but it is absent from this casework
policy because ground casework derivations cannot discharge a Goal. PRUNE also
remains a root-safe core operation but is absent from the learned menu: an
exhaustive child must be solved or symbolically refuted for exact aggregation,
not silently discarded. Identity Result payloads remain valid in the generic
core, while the casework policy uses PROPOSE_SOLUTION for identity.

VERIFY, COMMIT, and ready AGGREGATE are deterministic closure operations, not
controller decisions. After each learned action, the environment repeatedly:

1. verifies the triggering unchecked Claim, then any remaining active
   unchecked Claim in ascending id order;
2. commits each verified, discharge-accepted Claim in ascending id order;
3. for the lowest-id open Goal whose exhaustive children are all committed,
   constructs the exact reducer payload and ordered child premises, creates an
   Aggregate Claim, and repeats from step 1.

The triggering Claim takes priority over older pending work. Closure ends at
the next stable policy state or a terminal root state. It makes no LLM call,
does not consume another controller turn, and produces no BC example of its
own. Each closure event records the exact affected or created Claim id. A
learned action on the final available turn receives its full deterministic
closure before horizon failure is decided.

Evidence is never a target. Intermediate states requiring closure have an
empty learned menu and must not be sent to the controller. The four learned
verbs call the frozen generator; deterministic closure does not.

The decomposition GenerationContract receives the proposed connective as well
as the child obligations. The initial casework contract accepts only
`exhaustive_case`; a conjunctive rendering of an exact case partition is an
algebra rejection, not a permanently non-aggregating tree.

## Generator lens versioning and hosted gates

The environment contract selects the exact generator request builder through
its lens contract. `typed-claim-lens-v2.1` is current. The
`typed-claim-lens-v2` builder remains byte-exact for legacy replay and
reconstructs all 66 sealed E23 generator requests. An unknown lens contract is
an error; it never falls back to the current prompt.

Generator schema behavior is independently selected by the role-local wire
contract. `typed-policy-generation-v2.1` binds nonmutating strict validation
for current generation, while `typed-policy-generation-v2` binds legacy
numeric-string coercion only for exact replay. The teacher role follows the
same rule through current `controller-choice-wire-v2.1` and historical
`controller-choice-wire-v2`. Diagnostic replay derives and applies the same
mode from each recorded role contract. The outer environment and collection
record schemas remain unchanged.

E23 passed the five-problem hosted integration gate for collection,
diagnostics, strict replay, and artifact publication. Its 66-decision artifact
is integration evidence, not training data. E24 held the 66 pre-states,
actions, generator distribution, transport, decoding, action algebra, and
verifier fixed while changing only the lens to v2.1. The probe made 66
generator calls and zero controller calls. Typed-unusable output was 0/66 and
scalar usability was 33/33, so the first two preregistered gates passed.
Applied DECOMPOSE drafts were 0/33, so the decomposition gate failed.

E25 and E26 isolated symbolic rescoring and bounded semantic feedback without
clearing the application gate. E27 aligned the feedback input key with the
required `domains` output key, reaching 32/33 typed full-domain drafts but
0/33 applied. E28 then introduced model-authored separator selection, but
hidden number-to-string schema coercion violated its frozen string-only
contract. Its coerced semantic rows, primary-gate counts, and empirical
conclusions are quarantined; its raw attempts and strict no-coercion audit
remain valid protocol diagnostics and are explicitly bound into E29. E29
tested the separator-only representation with exact string typing and the
frozen one-retry policy. Only 8/33 logical cases ended with typed payloads, and
all 25 schema-triggered retries exhausted. The eight accepted drafts applied
but covered only one of five problems, so all three gates failed. E29 created
no controller decision, reward, trajectory, closure, or training row and does
not authorize boundary integration or collection.

`typed-claim-lens-v2.1` remains the current integrated lens. No separator-only
lens is integrated, and E23-E29 are not designated for the first v2 training
collection. The next representation candidate must be separately versioned
and preregistered. It accepts model-authored JSON number tokens, preserves the
exact decimal lexeme, and converts that lexeme to an exact rational in trusted
code without binary-float conversion. It must keep controller behavior,
reward, the learned action menu, retry policy, topology, symbolic scoring, and
the training path fixed. A passing representation gate still requires a fresh
same-five end-to-end confirmation before the 20-problem data-yield probe.

## Canonical controller observation

The only v2 controller state serialization is
`controller-observation-v2`. Its top-level keys are exact:

```json
{
  "schema_version": "controller-observation-v2",
  "environment_contract_sha256": "<64 lowercase hex characters>",
  "root_goal_id": 0,
  "turn": 3,
  "turn_limit": 16,
  "previous_outcome": null,
  "goals": [],
  "evidence": [],
  "claims": [],
  "legal_pairs": []
}
```

Each Goal record has exactly:

```json
{
  "id": 0,
  "parent_goal": null,
  "statement": "...",
  "obligation": {
    "variable": "x",
    "equation": "...",
    "domain": "...",
    "objective": "sum"
  },
  "assumptions": [],
  "connective": null,
  "status": "open",
  "committed_claim": null
}
```

For a compact solved Goal, `committed_claim` is exactly
`{id, kind, payload, statement, verifier_result, discharge_result}`. Both
receipts are accepted and use the same pinned verifier contract. The closed
descendant Claim may be omitted, so this inline object is the authoritative
compact outcome. When a solved or pruned Goal hides its decomposition
descendants, its serialized `connective` is null; the prior transition retains
the auditable decomposition history.

Each Evidence record has exactly
`{id, scope_goal, statement, premises, source, committable}`, where source is
`trusted_given` and committable is false. Each Claim record has exactly:

```json
{
  "id": 4,
  "owner_goal": 1,
  "kind": "solution",
  "payload": {"variable": "x", "value": "5"},
  "premises": [],
  "statement": "x = 5",
  "status": "verified",
  "verifier_result": {
    "contract_id": "...",
    "context_digest": "...",
    "accepted": true,
    "reason_code": null
  },
  "discharge_result": {
    "contract_id": "...",
    "context_digest": "...",
    "accepted": true,
    "reason_code": null
  },
  "discharge_status": "accepted"
}
```

For an active Claim, `verifier_result` is null exactly while status is
`unchecked`; `verified` and `rejected` require the four shown fields. A pruned
Claim retains its prior result, if any, so pruning cannot erase audit history.
An accepted result has null reason_code; a rejected result has a bounded
reason_code from the pinned verifier contract. `discharge_result` is null for
unchecked or validity-rejected Claims and is required after accepted validity.
Its accepted bit must agree with `discharge_status`. Validity and discharge
receipts must use the same verifier contract. Status, result, and discharge
combinations are validated mechanically.

`previous_outcome`, when non-null, has exactly:

```json
{
  "action": {"verb": "verify", "target": 4},
  "code": "applied",
  "reason": null,
  "verifier_context": {
    "contract_id": "...",
    "purpose": "validity",
    "claim_kind": "solution",
    "owner_goal": 1,
    "context_digest": "..."
  }
}
```

Outcome code is `applied` or `rejected`. A rejected outcome requires one of
`stale_action`, `generation_error`, `payload_error`, or `algebra_rejected`; an
applied outcome has null reason. A VERIFY whose symbolic result is false is an
applied transition because it immutably changes the Claim to `rejected`.
`stale_action` remains a bounded diagnostic value, but the synchronous v2
macro API rejects an out-of-menu selection before generation and writes no
canonical transition row.

At a stable controller decision, `previous_outcome` is the outcome of the
previous learned action, not the final internal closure event. Its verifier
context is therefore null. Applied VERIFY receipts use the same bounded
outcome schema inside `closure_events`; they require the exact validity
context shown above. Intermediate closure states may be serialized for tests
and diagnostics, but they have an empty learned menu and are not replay rows.

A compact terminal next observation may omit the Goal targeted by the previous
learned action. Transition validation binds that target to the current
observation before stepping. The next observation therefore validates the
learned verb and nonnegative target id without requiring a hidden target record.
Applied internal VERIFY diagnostics still require the exact full Claim record.

Every legal pair has exactly `{verb, target}` and uses one of the four learned
verbs. Goal, Evidence, and Claim ids share one global namespace. Records are
sorted by id; legal pairs are sorted by the fixed full algebra order and then
target id. Premise order is semantic and is preserved. JSON object keys are
serialized with the pinned canonical JSON serializer; NaN, infinities,
unknown keys, and missing keys are errors.

Environment-bound validation additionally round-trips every obligation,
payload, and visible decomposition through the pinned GenerationContract. It
binds every visible receipt to the pinned verifier contract id and its exact
validity and discharge rejection-code sets. Canonical observation bytes and
digests require both contracts; structural validation alone cannot produce a
replay digest. Macro-transition validation must still reproduce deterministic
closure before trusting an accepted receipt or reward.

The observation omits duplicated or presentation-only fields: no top-level
problem copy, `state_text`, `text`, rendered flag, owner-path cache, depth,
status booleans, legal-verbs cache, or verifier-call history. Root Goal
statement carries the problem. Owner paths, depth, active status, and node
feature context are derived from graph records. A node serializer therefore
takes `(observation, node_id)`, not an isolated record with duplicated owner
text.

The observation includes the full active graph plus compact records for closed
Goals. Every legal target has exactly one active typed record, every ancestor
Goal is open, and target type agrees with action semantics. Rendered teacher
text is a deterministic presentation adapter over this object. It is never
parsed back into state.

## Canonical transition record

Every v2 run starts with one record having exactly:

```text
schema_version: controller-run-header-v2
environment_contract
collection_contract
environment_contract_sha256
collection_contract_sha256
```

The environment contract pins transition, verifier, reward, lens, generation
schema, max tokens, seed policy, implementation identities, action orders,
target kinds, and canonical serializer. The collection contract pins both
model identities and implementations, exact secret-free transports,
controller and generator sampling settings, behavior source, deterministic
fallback policy, policy-input and output contracts, and the trusted problem
source digest and adapter. A teacher presentation contract and a learned
structured-feature contract are different values; both may consume the same
observation schema. Neither may overwrite or masquerade as
`controller-observation-v2`. The header is written before the first episode so
a crashed run remains interpretable.

Every decision then writes one `controller-transition-v2` record with exactly
these keys:

```text
schema_version
environment_contract_sha256
collection_contract_sha256
episode_key
problem_id
turn
observation
action
learned_effect
outcome
closure_events
reward
next_observation
terminal
behavior
diagnostic_ref
```

`action` is exactly `{verb, target}`. `behavior` is exactly
`{source: teacher | learned_policy, bc_eligible: bool, fallback: bool}`.
`bc_eligible` is derived and is true exactly for a non-fallback teacher
decision. `learned_effect` is the exact persisted typed draft. A decomposition
records its connective plus every child statement and obligation. A proposal
records kind, payload, empty premises, and the exact allocated Claim id when
applied. Algebra rejection retains its rejected draft with a null Claim id.
Generation or payload failure before a typed draft exists records null.

`diagnostic_ref` resolves to a decision receipt binding collection digest,
episode, turn, observation digest, source, proposed and selected actions,
fallback, and both role calls when hosted. Each call resolves to append-only
HTTP attempt records containing the secret-free request body, raw response,
raw content, usage, status, error, retry indices, and effective seed. Strict
validation rebuilds the controller presentation and generator lens request,
replays the retry transcript, decodes both terminal payloads, and binds them to
the selected action and `learned_effect`. Diagnostics are never replay state or
reward inputs.

One transition spans one learned decision through deterministic closure to the
next stable decision boundary. `outcome` is the learned action outcome.
`closure_events` is an ordered array whose entries have exactly:

```text
sequence
claim_id
outcome
```

`sequence` is contiguous from zero. `outcome.action.verb` is `aggregate`,
`verify`, or `commit`. For AGGREGATE, `claim_id` is the newly allocated exact
Aggregate Claim id; for VERIFY and COMMIT, it equals the targeted Claim id.
Every applied VERIFY event carries the exact validity verifier context, while
other internal outcomes carry no context. A generator failure has no closure
events. Internal events are never separate controller rows and are never BC
eligible.

`next_observation` is required even when terminal. It makes the next legal
mask explicit for off-policy Q-learning. Validators require: action is in the
current learned menu; outcome.action equals action; closure events follow the
deterministic priority and exact claim linkage above; next turn is current turn
plus one; next previous_outcome equals the learned action outcome; reward
agrees with the named symbolic-only reward contract; and terminal agrees with
the fully closed next state. The environment digest must agree across the
header, transition, and both observations. The collection digest must agree
across the header, transition, trusted problem resolver, and diagnostic
sidecar. There are no duplicated `legal`, `state_render`, or free-form feedback
fields.

Reward is zero at every transition except verifier-backed root success,
including rejected generation, symbolic rejection, and horizon expiry. A
terminal root with a committed, discharge-accepted Claim receives +1. After
closure on the final available turn, an open root receives 0 and terminates,
as required by `symbolic-only-v1`. This terminal predicate is computed only
from the trusted graph and symbolic verifier receipts; it never reads a dataset
answer or calls an LLM. Successful closure on the final turn takes precedence
over horizon expiry.

Strict replay is header-first and sequence-complete. It accepts canonical
JSONL only, resolves each initial state through the problem source and adapter
pinned by the collection contract, then re-executes every episode from turn
zero without gaps, duplicates, or post-terminal rows. Interleaved episodes are
allowed, but each episode must form one complete chain. Structural record
validation alone is not a training trust boundary.

For hosted collection, every transition requires a one-use diagnostic receipt,
including fallback and non-BC rows. Replay reconstructs the exact role-specific
request and retry transcript from raw provider envelopes, checks the selected
controller action, and checks the generator's decoded draft or bounded failure
against the applied learned effect. A row cannot become BC-eligible from a
schema-valid receipt whose controller payload names a different action.

## Dataset and checkpoint boundary

`phase-b-v2` consumes only explicit `controller-transition-v2` records. It
does not select a branch based on field presence, parse a render, manufacture
placeholder nodes, or combine v1.1 and v2 rows. Conversion validates the
entire observation and transition before deriving returns. Dataset rows retain
both observations and both legal masks.

Conversion accepts only a run sealed by strict replay. Each immutable example
also retains exact current and next structured features, selected verb and
target indices, behavior provenance, diagnostic reference, symbolic reward,
terminal flag, discount, and Monte Carlo return. The dataset digest is
value-sensitive over the complete canonical record. The grouped split assigns
whole problem ids, binds exact train and validation membership, and contributes
its own digest to the checkpoint. The rung-2 trainer applies masked verb and
conditional pointer cross-entropy to every provenance-eligible teacher row,
without success filtering, and selected-pair Monte Carlo Q regression to every
training row, including fallback and zero-return rows. The exact legal-menu
comparator follows the same provenance-only label rule.

A v2 checkpoint has exactly four top-level fields:

```text
checkpoint_schema: phase-b-controller-checkpoint-v2
state_dict
contract_manifest
contract_sha256
```

The manifest is the single metadata source. The first v2 BC checkpoint has
exactly this nested shape; angle-bracketed values are run values, not optional
keys:

```yaml
manifest_schema: phase-b-contract-manifest-v2
observation_contract:
  schema: controller-observation-v2
  canonical_serializer: canonical-json-v2
  outcome_schema: transition-outcome-v2
  receipt_schema: symbolic-check-receipt-v2
transition_contract:
  run_header_schema: controller-run-header-v2
  record_schema: controller-transition-v2
  semantics: reasoning-macro-transition-v2
  closure_schema: deterministic-closure-events-v2
  closure_priority: [verify, commit, aggregate]
  closure_turn_cost: 0
  closure_bc_eligible: false
  previous_outcome_source: learned_action
action_contract:
  semantics: claim-closure-v2
  algebra_verb_order:
    [decompose, propose_result, propose_solution, derive, aggregate,
     propose_refutation, verify, commit, prune]
  policy_verb_order:
    [decompose, propose_result, propose_solution, propose_refutation]
  target_kinds:
    decompose: [goal]
    propose_result: [goal]
    propose_solution: [goal]
    derive: [goal]
    aggregate: [goal]
    propose_refutation: [goal]
    verify: [claim]
    commit: [claim]
    prune: [goal, claim]
environment_contract: <exact EnvironmentContract.to_record() object>
environment_contract_sha256: <64 lowercase hex characters>
training_collection_contract: <exact CollectionContract.to_record() object>
training_collection_contract_sha256: <64 lowercase hex characters>
feature_contract:
  version: phase-b-features-v2
  global_serializer: canonical-json-v2
  node_serializer: typed-node-context-v2
  structural_scalars:
    [turn_fraction, remaining_turn_fraction, log_node_count,
     verified_claim_fraction, solved_goal_fraction, pruned_node_fraction]
encoder_contract:
  type: <hash | frozen_embedding | zero_text>
  implementation: <fully qualified encoder implementation>
  algorithm_contract: <versioned algorithm identifier>
  model: <string or null>
  model_digest: <string or null>
  dimension: <int>
policy_contract:
  architecture: factored-controller-v2
  model_config:
    hidden_dim: <int>
    repr_dim: <int>
    node_repr_dim: <int>
    verb_embed_dim: <int>
  feature_dim: <int>
  state_dict_sha256: <64 lowercase hex characters>
  decision_rule: joint
  acting_score: behavior-cloning-probability-v2
training_target_contract:
  target_type: monte-carlo-return-v1
  gamma: <float>
  reward_contract: symbolic-only-v1
dataset_contract:
  schema: phase-b-v2
  dataset_sha256: <64 lowercase hex characters>
  split_sha256: <64 lowercase hex characters>
```

It pins `controller-observation-v2`, `controller-transition-v2`, bounded
outcome schema, macro closure semantics, `claim-closure-v2`, separate full
algebra and learned verb orders, exact target kinds, the complete environment
and teacher collection contracts and their digests, symbolic verifier and
reward contracts, canonical JSON and node serializers, ordered scalar
features, exact encoder implementation, algorithm, identity, digest, and
dimension, model configuration and feature dimension, value-sensitive model
state digest, global-joint decoding and acting score, target/discount
definition, dataset schema and SHA-256, and split SHA-256. The first v2
checkpoint accepts exactly one training environment and collection contract;
multi-contract training requires a later explicit contract revision.

`contract_sha256` is the SHA-256 of canonical manifest JSON. A loader first
checks the digest, then the exact manifest shape, runtime constants, encoder,
state-dict shapes and values, and live MDP contract. The saved teacher
collection contract is not compared byte for byte with deployment. Instead,
deployment must preserve the environment and frozen generator distribution
while replacing the teacher with a local, checkpoint-identified learned
controller using `phase-b-features-v2`, joint decoding, and forbidden fallback.
The checkpoint manifest digest is the learned controller revision. Missing
fields, unknown fields, dimension-preserving serializer changes, same-shape
weight swaps, and user-supplied schema overrides are load errors. Metadata is
not duplicated beside the manifest and missing v2 metadata never defaults to
runtime values.

Legacy checkpoints keep their existing loader and require an explicit legacy
flag. They cannot request v2 observation or action semantics. V2 checkpoints
never use legacy defaults or the legacy override.

## Required contract tests

Before collection, the parallel v2 core must pass all of these boundaries:

1. GoalObligation construction uses only approved non-label fields and fails
   if `answer`, `solution_set`, or answer-derived difficulty fields enter the
   object or its digest.
2. Every payload kind accepts only its exact flat keys, canonicalizes symbolic
   strings, and reproduces the fixed statement byte for byte.
3. Verification and discharge tests cover every row of the kind table,
   including direct objective Results, true but irrelevant derivations,
   incomplete aggregate premises, non-exhaustive decompositions, and an
   outside candidate beside another live solution.
4. Node ids are globally unique; parent, owner, scope, premise, and commit
   references exist with the right type; Goal ancestry is acyclic.
5. Every learned legal target is active and represented exactly once. Target
   kinds match the four policy verbs; Evidence is never legal; closure-pending
   states have no learned menu; closed ancestors have no legal descendants.
6. Deterministic closure verifies each immutable context once, commits exactly
   the discharge-accepted Claim, constructs ready aggregates from the exact
   committed children, records exact Claim ids, and consumes no controller
   turn or BC row.
7. A solved compact Goal retains the exact committed id, kind, payload,
   derived statement, validity receipt, and discharge receipt. Mixed verifier
   contracts, unbounded reasons, and authored status or provenance are
   rejected.
8. Canonical observation bytes and digests are stable under irrelevant
   internal iteration order. Environment-bound validation rejects
   non-canonical symbols and unpinned receipts; structural validation rejects
   every unknown or missing field and every invalid enum or reference.
9. Macro-transition validation checks learned action/menu, outcome/action,
   closure order and Claim linkage, next learned outcome, turn, terminal,
   symbolic reward, and contract-digest linkage. Terminal transitions still
   carry next state and its empty legal mask.
10. Teacher presentation, logging, replay, and live feature adapters consume
    the same stable-decision observation while recording distinct policy-input
    contracts. No closure-intermediate state becomes a BC example.
11. V2 replay rejects legacy and mixed schemas. The v1.1 reproduction path
    remains byte stable and cannot emit a v2 row.
12. V2 loading rejects an absent or tampered manifest and every action,
    feature, encoder, dataset, policy, or live-MDP mismatch. Explicit legacy
    loading still reproduces the E19 contract.

The current 505-test v2 suite covers all twelve boundaries, including exact
applied-effect reconstruction, complete trusted-source replay, one-use hosted
controller and generator call linkage, structured feature alignment,
value-sensitive dataset and checkpoint loading, grouped split integrity,
provenance-gated warm-start losses, no-clobber collection finalization, and
checkpoint-backed joint action selection. Hosted collection additionally
requires an experiment-specific pre-registration and a provider endpoint whose
asserted model revision is adequate for that experiment's reproducibility
claim. The full repository suite has 891 passing tests and 1 skipped CUDA-only
test.

## Policy and value

The architecture may retain the useful factorization
`pi(verb | s) * pi(target | s, verb)` for training. Sampling may follow that
factorization. Greedy deployment must compute the product for every legal pair
and take its global argmax.

The Q head remains a joint pair scorer. Behavior cloning is only the warm
start. Off-policy Q-learning on controller-generated trajectories remains the
RL spine, followed by policy improvement or distillation. E20 and E21 tested
the cloned acting heads only, not that RL stage.

## V1.1 diagnostic boundary

The current implementation can isolate the following repairs without changing
the archived verb ids:

- exclude trusted givens from COMMIT witnesses;
- mask rejected VERIFY;
- mask targets below solved or abandoned goals;
- render committed outcomes in compacted goals;
- use graph-derived structured observations;
- use joint-pair greedy decoding;
- set generation temperature explicitly;
- count and retain malformed provider replies;
- validate dataset schema, encoder identity and digest, feature dimension,
  verb order, decoding rule, and action semantics at checkpoint load;
- bind observation, outcome, action targets, transition flags, verifier,
  reward, ordered feature transforms, training targets, and policy decoding in
  one canonical manifest; store its digest in every new checkpoint.

These switches are useful for one-manipulation diagnostics. They do not solve
hidden COMMIT witness selection or goal relevance, so they are not the final
contract. The repaired legacy serialization is named
`controller-observation-v1.1`; the v2 name is reserved for the typed
Evidence/Claim schema, direct Result path, learned four-verb menu, and
deterministic exact-claim closure.

## E22 interpretation

The preregistered zero-text E22 ablation reached 111/225 exact held-out pairs
(49.33%), below E19's 144/225 and the exact legal-menu lookup's 146/225. Verb
accuracy stayed near E19 while conditional pointer accuracy fell sharply.
Text therefore carried predictive target information, but E19 still did not
beat the content-free lookup. Because the historical text includes ids,
render annotations, and protocol state, E22 does not establish semantic
generalization. Fresh training data must use the structured observation.

The implemented v2 rung-2 path warm-starts Q by Monte Carlo regression. No
usable v2 training collection or v2 checkpoint exists yet. E20 and E21 acted
from legacy behavior-cloning heads, and no target-network Bellman update,
controller exploration loop, or gated Q-to-policy distillation exists yet.
Implementing
that stage before the structural claim and action contract stabilized would
have trained the RL spine against the wrong MDP. The structural contract and
warm-start boundary are now fixed, while the generator representation must
clear the next registered gate before training data collection.

## Evaluation sequence

1. Completed: audit E18 and E19 against an exact legal-menu lookup baseline.
2. Completed: run the zero-text E22 ablation with all E19 settings fixed.
3. Completed: implement the parallel typed core, trusted obligation adapter,
   direct Result path, deterministic payload renderers, kind-specific symbolic
   discharge routes, learned menu, root-safe PRUNE, and canonical observation.
4. Completed: implement deterministic closure execution and one-turn macro
   transitions with exact symbolic replay.
5. Completed: implement typed generator lenses, digest-gated partial-safe
   collection, run logging, strict replay, diagnostics, replay-sealed dataset
   conversion, grouped splits, exact legal-menu comparator, deterministic
   BC/MC warm-start, manifest-bound checkpoint loading, and checkpoint-backed
   learned controller.
6. Completed: run the E23 five-problem hosted integration gate. Preserve it as
   integration evidence, not training data.
7. Completed: run the E24 fixed-decision generator probe. The wire and scalar
   gates passed; the DECOMPOSE gate failed.
8. Completed: run the E25-E29 decomposition diagnostics. E28 is quarantined;
   the exact E29 rerun failed all three gates and rejects the string-valued
   separator wire for this generator.
9. Completed: bind schema-validation behavior through current role-local
   generator and teacher wire ids, thread it through both live model roles and
   diagnostic replay, and preserve prior ids for exact legacy replay.
10. Next: design, prove, test, and pre-register the separately versioned exact
   JSON-number boundary representation. If it passes its gate, run a fresh
   same-five end-to-end confirmation and then the 20-problem data-yield probe.
11. Then collect fresh structured trajectories and execute the implemented
   warm-start with exact menu lookup, grouped seeds, zero-text comparison, and
   matched decoding in every BC comparison.
12. Then implement controller exploration, replay next-state masks,
   target-network Bellman updates, and gated Q-to-policy distillation.

No hosted collection should mix the diagnostic effects in steps 3 through 10
inside one experimental arm. Contract migration and invariant testing happen
before empirical comparison.
