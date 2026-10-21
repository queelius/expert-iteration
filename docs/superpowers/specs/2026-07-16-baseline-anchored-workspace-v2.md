# Baseline-Anchored Permissive Workspace v2

Status: implemented. E36 confirmed the root result factor, E37 exercised the
exact shared-direct-Candidate branch, E38 confirmed the REVISE
answer-container factor on sealed E37 states, E39 localized a coordinator
selection failure, E40 showed latest-revision dominance, E41 retained an
independent-solution worker prompt, and E42 retained an explicit coordinator
routing prompt on replayed outcomes. This document does not authorize another
hosted model call or a broader performance claim.

## 1. Purpose

E35 showed that direct recoverability as one legal action is too weak. The
GPT-4.1-mini controller chose `DECOMPOSE(root)` before seeing any output from
the frozen worker on 48 of 60 problems. It solved 19 of those problems, while
the exact direct worker solved 47. The redesigned decision process first
obtains the exact direct Candidate, then asks the controller whether to submit
it or spend additional inference trying to improve it.

The redesign has five independently versioned factors. Implementing them
together is convenient, but an experiment changes only one factor unless a
later ledger entry explicitly registers a new bundled system version.

## 2. Invariants

- The symbolic terminal verifier remains the only reward source.
- Generated Goals and Candidates are unverified workspace content.
- No intermediate LLM output is accepted or rejected for mathematical truth.
- The hidden expected answer never enters state, a lens, a controller request,
  or a worker request.
- The frozen worker and the controller remain separate roles.
- Legacy E31-E35 contracts and request bytes remain replayable.
- RL remains the training spine. MCTS is a strictly additive accelerator.

## 3. Versioned factors

### 3.1 Initialization

`empty-root-v1` is the legacy initial state. The first controller decision sees
only the root Goal.

`direct-candidate-prefix-v1` executes `ANSWER(root)` deterministically from the
empty state using the exact `direct-math-json-v1` request. The resulting
transition is fully logged and consumes turn zero, but it is not selected by a
controller and is never a behavior-cloning label. A successfully decoded
Candidate is unverified and receives reward zero. The first controller
decision sees the resulting turn-one state.

If the direct prefix exhausts its parse retry, the generation-error transition
is retained. The controller then sees the root Goal, the failed prefix event,
and no Candidate. No hidden fallback or unlogged retry is permitted.

The direct control for a shared-prefix experiment submits the exact prefix
Candidate without another worker call. A teacher branch starts from the exact
same immutable prefix state. Prefix generation and branch continuation are
separate provenance roles.

### 3.2 Root result contract

`workspace-answer-free-v1` is the legacy worker instruction.

`verifier-ready-root-answer-v1` changes only worker instructions for a
Candidate owned by the root Goal. `ANSWER(root)`, `SYNTHESIZE(root)`, and
`REVISE(root_candidate)` must place only the requested final scalar or symbolic
expression in the `answer` field. Variable assignments, equations, units,
prose, code fences, and alternative lists belong in `reasoning`, unless the
root problem explicitly requests that answer form. Candidate admission remains
permissive. Trusted code does not repair, extract, or judge the generated
answer. The symbolic verifier still decides terminal reward only at SUBMIT.

Child Goal answer instructions are unchanged.

### 3.3 Action availability

`all-affordances-v1` is the legacy legal menu.

`active-frontier-v1` derives a bounded current frontier:

- `ANSWER(goal)` is legal only when the Goal has no Candidate.
- `DECOMPOSE(goal)` is legal only when the Goal has no children and its depth
  is below the bound. A non-root Goal with a Candidate cannot be decomposed.
  The root may be decomposed while preserving the direct prefix Candidate.
- `SYNTHESIZE(goal)` is legal only when every child has a Candidate and no
  Candidate created by a prior SYNTHESIZE is fresh relative to the latest
  child Candidate.
- `REVISE(candidate)` is legal only for the latest Candidate of its Goal and
  only when that Goal has not already produced a revision Candidate.
- For non-root Goals, stale Candidates have no actions.
- `SUBMIT(candidate)` is legal for the protected direct prefix Candidate and
  for the latest root Candidate. If these are the same node, it appears once.

The protected prefix Candidate is the node created by the applied turn-zero
`ANSWER(root)` event under `direct-candidate-prefix-v1`. Protection is derived
from immutable history, not an answer-bearing field.

The active-frontier observation exposes the derived protected Candidate id so
the controller does not need to reconstruct this safety-critical role by
joining node and event records.

The mode changes legality only. It does not delete nodes or rewrite history.

### 3.4 Teacher choice wire

`verb-target-choice-v1` is the legacy wire. The teacher returns a verb and a
target independently, so it can form a pair absent from the legal menu.

`legal-menu-index-v1` presents the exact ordered legal menu with stable integer
indices and asks the teacher to return one `action_index`. The response schema
enumerates only valid indices. Trusted code maps that index to the internal
`(verb, target)` pair. The controller action remains the pair; only the LLM
teacher transport changes. Parse or bounds failure uses the same explicit
fallback rule and is logged. The fallback selects `ANSWER(root)` when that
exact direct action is legal, otherwise `SUBMIT(protected_direct_candidate)`
when it is legal, otherwise the first ordered legal action.

### 3.5 REVISE result contract

`workspace-revise-open-object-v1` is the exact E37 legacy contract. It asks for
improved reasoning and an answer field but admits any top-level JSON object.
The legacy decoder can preserve an object without `answer` as Candidate text
for replay compatibility.

`workspace-revise-answer-container-v1` changes only the REVISE instruction and
its local response schema. The prompt displays one explicit Candidate object
with `reasoning` and `answer`. The schema requires a top-level `answer` whose
value is a nonempty string or finite number, permits optional unverified
`reasoning`, and rejects other top-level fields. The decoder cannot fall back
to a copied lens object under this mode. These are protocol constraints only:
trusted code does not accept or reject the mathematical content, and only a
later root SUBMIT can produce symbolic reward.

Prompt plus schema form one linked versioned contract. Guided JSON is disabled
for the current worker route, so the schema is not visible in the first hosted
request, while the prompt alone cannot enforce local admission. Legacy mode
remains the default for exact E31-E37 request replay.

## 4. State and transition semantics

The immutable Goal, Candidate, and Action records remain unchanged. The direct
prefix is an ordinary generative transition, so exact replay uses the existing
action algebra and worker payload decoder. A redesigned episode has:

1. empty root state at turn zero;
2. deterministic direct-prefix transition;
3. zero or more controller-selected transitions; and
4. terminal SUBMIT or horizon expiry.

The prefix counts against the turn limit. This keeps worker calls and state
growth visible rather than granting hidden computation. A redesigned
collection record distinguishes `deterministic_prefix`, `teacher`, `fixed`,
and `direct_control` decision sources. Only rows with an actual controller
decision and complete teacher provenance can enter behavior cloning.

## 5. Observation and action presentation

The full canonical observation continues to contain nodes, history, remaining
turns, terminal reward, and the exact legal menu. The observation schema id
binds the action-availability mode. Under the indexed teacher wire, the request
also contains an explicit list of `{index, verb, target}` entries. No natural
language alias is authoritative over the indexed list.

## 6. Collection contract

A new run contract must bind:

- initialization mode;
- root result contract;
- action-availability mode;
- teacher-choice wire;
- REVISE result contract;
- prefix and branch seed policies;
- worker and teacher identities and transport settings;
- problem and hidden-label digests;
- every inherited workspace contract id; and
- role-bound raw attempts plus partial-safe publication.

Existing `workspace-hosted-run-contract-v1.1` collectors must reject nonlegacy
modes. A new collector may reuse their source and artifact utilities but must
not silently change a v1.1 header.

The implemented redesign collector is a bound single-arm collector. It does
not yet provide an exact shared-prefix paired direct-versus-teacher branch.
That comparison requires a separate paired contract and collector before it
can support a causal reading.

## 7. Required tests

- The prefix request is byte-identical to the direct baseline request.
- The prefix creates an unverified Candidate with zero reward and no teacher
  decision.
- Direct control submits the exact prefix Candidate without another worker
  call.
- A failed prefix is visible and does not leak the hidden answer.
- Root result instructions change only for root-owned Candidate generation.
- Every active-frontier legality rule above has a positive and negative test.
- The protected prefix Candidate remains submittable after structured work.
- Indexed choice cannot construct an illegal pair.
- Legacy requests, observations, run-contract hashes, and tests remain
  unchanged.
- The answer-container REVISE mode rejects a lens echo, admits an explicit
  answer-bearing Candidate, and never judges its algebra.
- Publication is partial-safe and excludes credentials and hidden labels.

## 8. Experiment order

E36 completed the first step on 59 exact sealed E35 pre-SYNTHESIZE states.
Verifier-ready root prompting solved 49/59 versus 27/59 for exact legacy
prompting, with 22 verifier-ready-only and zero legacy-only cases. Integration
passed and the registered positive gate cleared. This retains
`verifier-ready-root-answer-v1` as a causal prompt mechanism on the reused
source; it is not evidence that a teacher or decomposition beats direct
prompting.

E37 then generated each Qwen2.5-3B direct Candidate once and forked exact local
submission from GPT-4.1-mini continuation. Both solved 43/60, with 43 both,
zero discordances, and 17 neither. Every teacher path ultimately submitted the
protected direct Candidate. This establishes descriptive baseline preservation
but no improvement. The teacher requested extra work disproportionately on
direct failures, but none of 14 REVISE calls produced an answer-bearing
revision because the open object contract admitted lens echoes or exhausted on
them.

E38 replayed those 14 exact pre-REVISE states under the legacy and
answer-container contracts. The new contract produced 14/14 usable Candidates
versus 0/14 for legacy, with all pairs answer-container-only and one-sided exact
probability `1 / 2^14`. All treatment calls completed once; legacy admitted
five lens echoes and exhausted nine calls after a retry. Across 10 unique root
episodes, source and revised Candidates were both symbolically correct on
three, revision-only on one, source-only on zero, and neither on six. No
controller chose or submitted these Candidates, so E38 establishes generation
repair rather than end-to-end improvement.

E39 retained the new REVISE contract in a complete paired shared-prefix run.
Direct submission and teacher continuation each solved 43/60, with 43 both
and 17 neither. All 12 selected revisions were valid. In those cases the
baseline and revision were both correct on four, revision-only on three,
baseline-only on zero, and neither on five. Both Candidate 1 and the revision
were legal SUBMIT targets after every REVISE, but the generic GPT-4.1-mini
planner submitted Candidate 1 all 12 times. The exact run therefore exposed a
46/60 available-Candidate oracle while realizing 43/60. The oracle is only a
mechanism diagnostic; symbolic SUBMIT remains the sole reward source.

E40 then applied the retained REVISE contract once to all 60 exact E39 shared
prefix States. Every call produced a usable Candidate. Symbolic correctness
cells were 43 both, six revision-only, zero baseline-only, and 11 neither, so
latest revision and the available-Candidate oracle were 49/60 versus 43/60 for
protected baseline. E40 made no coordinator or SUBMIT call. It therefore shows
promising self-revision content on a reused development source, not terminal
teacher reward.

E41 changed only the REVISE instruction and paired every exact State and seed.
Retained revisions were correct on 49/60 and independent revisions on 55/60,
with 49 both, six independent-only, zero retained-only, and five neither. The
independent prompt was also correct on all 43 protected-baseline successes and
12 additional cases. Retain it for the next routing screen. E41 made no
coordinator or SUBMIT call, so it is not terminal teacher reward.

E42 paired generic and explicit GPT-4.1-mini routing prompts on exact E41
replays. Generic realized 46/60 verifier-scored terminal successes and explicit
routing 51/60: 46 both, five routing-only, zero generic-only, and nine neither.
Explicit routing preserved all 43 baseline successes and realized 8/12
revision-only recoveries, four short of the 55/60 fixed replay oracle. Retain
the routing prompt. Because E42 replayed worker output, it is a coordinator
prompt mechanism result rather than a live end-to-end teacher advantage.

The remaining sequence is:

1. run a live shared-prefix system comparison combining independent REVISE and
   explicit routing against exact direct submission, preferably on a fresh
   harder verifier-covered source;
2. compare coordinator model capacity only after that live system result;
3. retain active-frontier masking and indexed choice as fixed system settings,
   with separate ablations if an individual effect is claimed; and
4. only then evaluate end-to-end teacher advantage on a fresh harder source.

Each ledger entry pre-registers one manipulation, measure, gate, and reading
before any hosted request.
