# Permissive Reasoning Workspace v1

Status: implementation target for the post-E29 redesign.

## 1. Purpose

The controller orchestrates a frozen worker LLM over an external reasoning
workspace. The controller sees the complete workspace and chooses a typed
`(verb, target)` action. The worker sees an action-specific hybrid lens with
the root problem, target, ancestry, relevant local work, and remaining budget.

Intermediate LLM output is untrusted workspace content. Admitting it to the
workspace does not assert that it is true and supplies no reward. The symbolic
terminal answer checker is the only reward source.

## 2. State

State is immutable and contains:

- one root Goal holding the original problem;
- generated child Goals holding natural-language tasks;
- Candidate nodes holding unverified reasoning and an answer string;
- an ordered, complete action history;
- the learned-turn budget and terminal reward.

Node identifiers are dense, stable integers. Generated content never changes
node identity, legality, reward, or verifier truth by itself.

## 3. Learned actions

- `DECOMPOSE(goal)` asks the worker for two to five child tasks and appends
  them as Goals. Only JSON shape and resource bounds are checked.
- `ANSWER(goal)` asks the worker to answer one Goal and appends a Candidate.
- `SYNTHESIZE(goal)` asks the worker to combine completed direct-child work
  and appends a Candidate for the parent Goal.
- `REVISE(candidate)` asks the worker to improve a Candidate and appends a new
  Candidate for the same Goal.
- `SUBMIT(candidate)` is legal only for a root Candidate. It never calls an
  LLM. It invokes the symbolic answer checker and terminates the episode.

There is no intermediate algebra rejection. A malformed provider response or
missing required container is a generation error. A mathematically poor but
structurally usable generation is an applied transition whose downstream value
is learned from terminal symbolic reward.

Non-direct Candidate generation admits any top-level JSON object. Candidate
extraction prefers an `answer` at the top level, under `candidate`, or in the
last answer-bearing item under `candidates`. When none is present, the complete
object is retained deterministically as unverified Candidate content. This
keeps workspace-shaped model replies instead of treating their wrapper as a
mathematical or semantic failure. Resource bounds still apply.

## 4. Lenses

The initial `ANSWER(root)` request is byte-for-byte the direct baseline request.
This is the direct recoverability invariant.

Other worker actions use a hybrid local lens containing:

- the root problem;
- the selected action;
- the target Goal or Candidate;
- the Goal ancestry;
- direct child Goals and their Candidate work when relevant;
- previous attempts on the exact action and target;
- the remaining turn budget.

Child `ANSWER` has an explicit context mode. `target-only-v1` exposes only the
existing target-local lens. `preceding-siblings-v1` additionally exposes the
ordered Goals and Candidates of completed siblings that precede the selected
child under the same parent. This work is nested under the immediate parent's
entry in the root-to-target ancestry path, so the fixed worker instruction and
all non-lens prompt prose remain unchanged. Incomplete and later siblings stay
hidden. The mode is part of the hosted run contract, and it never changes the
initial `ANSWER(root)` direct request.

Unrelated branch transcripts are omitted. The teacher controller receives the
full canonical workspace observation and explicit legal action list.

## 5. Reward and termination

All non-`SUBMIT` rewards are zero. `SUBMIT` returns reward 1 exactly when the
symbolic answer checker accepts the submitted root Candidate, otherwise zero.
Any `SUBMIT` terminates. Horizon expiry terminates with zero reward. No LLM
judge, learned critic, reward model, or intermediate heuristic supplies reward.

## 6. Initial experiment gates

1. Direct parity: `ANSWER(root) -> SUBMIT` must preserve the raw direct request
   and symbolic score.
2. Fixed decomposition: compare direct against `DECOMPOSE -> ANSWER(children)
   -> SYNTHESIZE -> SUBMIT` before introducing a learned or LLM controller.
   A failed SYNTHESIZE is attempted once, then the comparison policy falls
   back to a root ANSWER so an exact failed pair cannot consume the horizon.
3. Lens ablation: compare hybrid, target-only, and full-workspace lenses on
   identical fixed action traces.
4. Controller value: compare a strong teacher with direct-only, fixed
   decomposition, and random legal policies under matched worker settings.
5. Teacher bandwidth and scale are later single manipulations. Free-form
   teacher directives are an oracle upper bound, not part of the v1
   distillable `(verb, target)` contract.
