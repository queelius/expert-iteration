# Separating Controller State from Worker Language

Cutoff: 2026-07-18

## Novelty Verdict

The broad architecture is occupied. Prior work already separates external state, a rendered model view, an action interface, parsing, and writeback. Prior work also shows that rigid structured input or output requirements can reduce reasoning accuracy, especially near a small model's capability boundary.

The defensible residual is a controlled two-consumer study:

> Given one authoritative state with identical semantic information, what representation should a compact controller observe, and what rendering and output contract should a frozen small worker receive?

No primary source located in this search cleanly crosses machine-native controller representation with language-native worker rendering under information-matched semantics and exact terminal reward. This is a causal interface-separation question, not a new agent architecture.

## Current Project Evidence

E47P now supplies a development result from 12 sealed States and 432 frozen
Qwen2.5-3B calls. A target-local worker view beat the same-target full-State
view by 9/72 = 12.5 points on held evaluation repetitions, with a 90%
whole-State bootstrap interval of [2.78, 23.61] points. The corresponding
discovery effect was 8/72 = 11.11 points. The registered locality screen
passed, but the discovery-selected fixed child tied direct problem-only
prompting at 10/24, and State-specific child choices did not repeat across
splits. Therefore E47P supplies no selector or semantic-observability result.

The post-outcome mechanism audit strongly implicates bad-draft anchoring. Full
child calls repeated the source root answer in 65/72 evaluation cells versus
38/72 local-child cells. Every one of the 11 reward-discordant evaluation
cells had full copy the root answer while local did not. Operational failures
had zero net contribution to the paired effect. This exact-copy audit was not
preregistered and remains exploratory.

The immediate design implication is separation, not more target-policy
training: the controller may need the complete authoritative State, while the
worker should receive a purpose-built view that excludes unneeded untrusted
draft content. E47P does not isolate whether the useful removal is the root
draft, sibling nodes, serialization length, or their interaction. A future
study must cross those context components directly and retain problem-only as
the decisive baseline. The canonical E47P run and diagnostic hashes are
`75944f77775728bea2a65939457f28f9d6e730e36cba10c6123c47f9d3c155b2`
and
`ab0fa0e523e0e35b839b09464f7270caf2ef3965519da8d3a43c9f2396893e99`;
full results and claim limits are in the experiment ledger.

## Closest System Precedents

- [Graph of Thoughts](https://arxiv.org/abs/2308.09687) has graph reasoning state, a controller selecting transformations and targets, a Prompter rendering selected graph context, and a Parser writing results back.
- [HarnessBridge](https://arxiv.org/abs/2606.12882) retains authoritative raw history while a learned harness projects a generator-visible history and mediates proposed actions.
- [VISTA](https://arxiv.org/abs/2606.30005) uses typed addressable context blocks and a rendered workspace.
- [Beyond Compaction (CWL)](https://arxiv.org/abs/2606.11213) uses a typed dependency-linked episode DAG for deterministic context management.
- [HiLight](https://arxiv.org/abs/2604.22565) learns non-destructive evidence highlighting for a frozen solver.
- [StateFlow](https://arxiv.org/abs/2403.11322) uses fixed states with state-specific prompts and transitions.
- [SWE-agent](https://arxiv.org/abs/2405.15793) separates environment state, observation rendering, action interface, history processing, and parsing.
- [TSCG](https://arxiv.org/abs/2605.04107) compiles machine-readable tool schemas into model-visible text.
- [Natural-Language Agent Harnesses](https://arxiv.org/abs/2603.25723) argues for studying harness representations and ablates harness modules.

These sources make internal-state versus worker-view separation an established systems principle.

## Structured-Format Pressure

- [Let Me Speak Freely?](https://arxiv.org/abs/2408.02442) reports that constraining reasoning into JSON, XML, or YAML can reduce accuracy and that reasoning naturally before formatting can recover performance.
- [The Format Tax](https://arxiv.org/abs/2604.03616) studies open and API models and reports that prompt-side formatting requests explain much of the loss, with two-turn reasoning then reformatting recovering many cases.
- [The Constraint Tax](https://arxiv.org/abs/2605.26128) focuses on 0.5B to 3B models and reports a tradeoff between schema validity and answer correctness, with delayed constraints helping.
- [Draft-Constrained Constrained Decoding](https://arxiv.org/abs/2603.03305) uses an unconstrained draft before a constrained second pass.
- [Thinking Before Constraining](https://arxiv.org/abs/2601.07525) separates free reasoning from constrained decoding within one generation procedure.

Therefore, free reasoning followed by formatting is not novel. The remaining question is where structure should live in a two-consumer controller-worker system.

## Stage 1: Worker Interface Factorial

Hold one canonical authoritative state fixed. Cross two input renderings with two output contracts:

| Worker input | Worker output |
|---|---|
| typed or JSON lens | strict schema during reasoning |
| typed or JSON lens | free-form reasoning with deterministic domain parser |
| deterministic prose lens | strict schema during reasoning |
| deterministic prose lens | free-form reasoning with deterministic domain parser |

Requirements:

- 120 to 200 held-out exact-verifiable tasks;
- at least two small workers, initially around 3B and 7B;
- at least two domains;
- the same facts, order, target, and action semantics in both input renderings;
- near-matched input token counts;
- matched worker calls and output budgets;
- randomized arm order; and
- frozen deterministic rendering and parsing before evaluation.

Primary outcome:

- exact terminal symbolic reward.

Secondary diagnostics:

- parse usability;
- wrong-but-schema-valid rate;
- correct-but-unparseable rate;
- tokens, latency, and cost; and
- target-local state change and final-answer propagation.

Kill the output-format claim if free-form output improves exact correctness by less than 5 points or loses more than 2 points of usable candidates after deterministic parsing.

Kill the input-rendering claim if prose versus typed input differs by less than 5 points with an interval spanning the preregistered practical range.

## Stage 2: Action Headroom

Only after choosing the worker-facing interface, collect a cached all-actions table from shared prefix states:

- approximately 200 states;
- every legal `(verb,target)` action once or twice;
- exact terminal outcome under a fixed continuation;
- best fixed policy;
- statewise and cross-fitted oracle;
- bidirectional action discordance; and
- uplift relative to a fixed baseline action.

Stop adaptive control if the cross-fitted oracle is less than 5 to 8 points above best fixed, or fewer than 10 percent of states show crossed action preferences.

## Stage 3: Controller Observation

Using the same cached outcome table, compare controller observations without new worker calls:

- legal menu and phase only;
- lean scalar and topology features;
- information-matched flat event features;
- raw transcript embedding;
- typed tree features;
- typed tree plus semantic text; and
- shuffled semantic text as a shortcut control.

Compare a linear probe, a tiny MLP, direct per-action value scoring, and the planned factored controller at matched capacity. Split by problem family or structural template. Evaluate held-out verifier reward and action regret, not teacher-action imitation.

Kill the typed-tree claim if it adds less than 3 points of held-out policy value over information-matched flat features, or if a linear or fixed router matches it.

## Fresh End-to-End Confirmation

Confirm only the following policies on untouched tasks:

- direct prompting;
- compute-matched independent sampling;
- best fixed harness;
- strongest simple router; and
- best learned policy, if the previous gates pass.

The learned system must beat direct prompting and the best fixed harness under matched calls, tokens, and verifier logic. Beating only a weak or malformed structured baseline is not evidence for orchestration.

## Paper Framing

Possible title:

> Who Needs Structure? Separating Controller State from Worker Language in Frozen-LLM Harnesses

The contribution would be the controlled placement of structure and its interaction with exact action value. It should not claim novelty for tree state, lenses, external control, natural-language rendering, or delayed formatting separately.
