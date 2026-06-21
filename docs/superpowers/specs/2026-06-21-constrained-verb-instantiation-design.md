# Constrained, chat-framed verb instantiation (design)

Date: 2026-06-21. Status: approved, pending spec review.

## Motivation

The first two closed-loop GPU runs produced no prior sharpening. Investigation
of the run's own records (`experiments/results/runs/20260619T193634Z-headline`)
showed the cause is not the search budget: every decision had byte-identical Q
across all four actions and a frozen available-verb set, the signature of
`LLMContextMDP.sample_transition` falling into `_masked_action` (increment turn,
reward -0.01, state unchanged) on essentially every verb. A direct probe
confirmed it: the encoder's raw-completion path masked 4 of 4 verbs (output was
not JSON at all: instruction echo, an invented bracket DSL, repeated special
tokens), and the plain chat-template path masked 3 of 4. The closed loop was
running on no-op transitions, so Q and pi were uniform by construction and the
prior could never sharpen. Because verbs never succeeded, no hypothesis ever
formed, so COMMIT_ANSWER never became available and the verifier reward never
fired: fixing verb instantiation is upstream of the entire learning signal.

A spike (chat template + JSON-schema-constrained decoding via
lm-format-enforcer, on the same Qwen2.5-Math-1.5B-Instruct model) recovered 3 of
4 verbs as valid and non-masked, including semantically correct arithmetic
(`COMPUTE_SUBEXPR` produced `7 * 6 = 42`). The 4th failure was not the model: it
was lm-format-enforcer rejecting a mixed-type enum (`["sg_root", null]`) in the
`FORM_HYPOTHESIS` schema, which we can represent differently. The fix is
therefore: generate verb content under the chat template with schema-constrained
decoding, and make the verb schemas constraint-parseable.

## Goal

Verb instantiation reliably yields schema-valid, handler-accepted actions
(target: all available verbs non-masked on the toy families), so the closed loop
runs on real transitions and the sharpening and L_Q signals become measurable.
Keep the one-forward-per-decision architecture (method.tex 3.4) as the target.

## Non-goals

- Not trying to make the small model semantically expert. Constrained decoding
  guarantees valid structure, not correct math; correctness is judged at
  COMMIT_ANSWER by the verifier reward, as designed.
- No change to the learning algorithm, the heads, the replay buffer, or MCTS.
- Not switching base models in this work (the substrate question is separate).

## Components

### 1. `constrained_json.py` (new, small, CPU-testable)

A single responsibility: given a JSON schema and a tokenizer, return a
`prefix_allowed_tokens_fn(batch_id, input_ids) -> list[int]` suitable for
`model.generate(..., prefix_allowed_tokens_fn=...)`, built from
lm-format-enforcer's core API (`JsonSchemaParser`, `TokenEnforcer`,
`TokenEnforcerTokenizerData`).

It must NOT import `lmformatenforcer.integrations.transformers`: that module
fails on transformers 5.x (it imports `PreTrainedTokenizerBase` from a path that
moved). We build the `TokenEnforcerTokenizerData` directly from the tokenizer's
vocabulary and eos id, replicating the few lines the integration helper would
have run. Building the tokenizer data is expensive (it walks the vocab), so it is
built once per tokenizer and cached; the per-schema parser and enforcer are
cheap and built per call. A module-level fallback shim (assign the moved symbol
back onto `transformers.tokenization_utils`) is documented in case a future
lm-format-enforcer release re-couples to the integration module.

### 2. Constraint-parseable verb schemas (`llm_context_mdp.py`)

Audit every verb schema in `_schema_for` for lm-format-enforcer parseability and
remove unsupported constructs. The known offender is `FORM_HYPOTHESIS`'s
`parent_subgoal_id` mixed enum `[*sg_ids, None]`. Replace the mixed string-or-null
enum with a constraint-parseable representation: a string enum that includes a
`"none"` sentinel (`[*sg_ids, "none"]`), and map `"none"` back to `None` in the
verb handler so downstream semantics are unchanged. Apply the same treatment to
any other schema the audit flags. A CPU test (below) asserts every verb schema,
across representative states, is accepted by `JsonSchemaParser` without raising,
so this class of regression is caught without a GPU.

### 3. Chat template plus constraint in the shared-forward encoder (one-forward)

`SharedForwardEncoder` keeps the prefill-once, reuse-for-generation contract,
now with chat framing and an optional constraint. To make the verb-independent
prefix a clean chat boundary, the state goes in the system message and the
verb-specific instruction goes in the user message:

```
<|im_start|>system
{instruction preamble}\n{serialized state}<|im_end|>
<|im_start|>user
{verb-specific instruction}<|im_end|>
<|im_start|>assistant
```

- The cacheable, verb-independent prefix is the rendered text through the
  user-turn-open marker (preamble + state + the fixed chat scaffolding up to and
  including `<|im_start|>user\n`). `embed(s) = h_T` is taken at the final token
  of this prefix; it is state-dependent (it has attended to the full system
  turn) and verb-independent, so one prefill serves the heads and all verbs.
- The verb-specific suffix is `{verb instruction}<|im_end|>\n<|im_start|>assistant\n`.
- Generation extends the cached prefill with the suffix and runs with the
  schema's `prefix_allowed_tokens_fn`.

The prefix/suffix split is derived by rendering the full chat prompt for a verb
and locating the boundary; a test asserts the prefix is identical across verbs
and that prefix+suffix tokenizes to the full chat-rendered prompt.

### 4. MDP wiring (`llm_context_mdp.py`)

`sample_transition`'s encoder branch passes the verb's schema-derived constraint
into the encoder's generation call and uses the chat-framed prefix/suffix. The
existing masked-action recovery stays as a backstop for the rare residual parse
or handler failure. The non-encoder (HTTP/mock) paths are unchanged.

### 5. Dependency

Add `lm-format-enforcer` to the `[hf]` extra in `pyproject.toml`. It is light
(pulls only `interegular`). `jsonschema` is already a dependency.

## Sequencing (correctness before optimization)

1. `constrained_json.py` + schema fixes + a constrained chat-generation path,
   validated to recover all available verbs (the spike, hardened). This gives a
   working loop even if generation is a separate forward from `embed`.
2. Fold generation into the one-forward shared prefill with the chat split.
   If the one-forward chat split degrades verb validity versus the validated
   state-in-user spike, fall back to state-in-user and accept a second forward
   for generation, recorded honestly in the FLOP accounting. Either way the loop
   is correct; one-forward is the optimization, not the correctness gate.

## Testing

- CPU, no model: `constrained_json` builds a `prefix_allowed_tokens_fn` from a
  schema using a tiny fake tokenizer; every verb schema across representative
  states is accepted by `JsonSchemaParser` (guards the mixed-enum class of bug);
  the `"none"` sentinel round-trips through the `FORM_HYPOTHESIS` handler to
  `None`.
- Tokenizer-only, no model (cheap, no GPU): the chat prefix/suffix split is
  verb-independent in the prefix and reconstructs the full chat prompt.
- GPU validation: reuse the spike. After the schema fix, assert all available
  verbs at the initial state (and a couple of deeper states) generate valid,
  non-masked transitions. This gates relaunching the headline run.

## Risks

- lm-format-enforcer transformers-5.x coupling: mitigated by building from the
  core API and not importing the broken integration module (verified workable in
  the spike via a one-line shim; the real code avoids the module entirely).
- One-forward chat split fiddliness: mitigated by the correctness-first
  sequencing and the documented state-in-user fallback.
- Constrained decoding fixes format, not semantics: expected and acceptable. The
  verifier reward at COMMIT_ANSWER remains the correctness signal; the immediate
  win is non-masked transitions and a live learning signal.

## After the fix

Relaunch the headline closed-loop run (the same 5-iteration config, or a
budget chosen with the now-working transitions in mind) and read the L_Q trend
and the prior/visit sharpening that this run was meant to measure.
