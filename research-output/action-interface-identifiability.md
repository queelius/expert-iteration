# Action-Interface Identifiability for Frozen-LLM Control

Cutoff: 2026-07-18

## Decision

The fastest prerequisite for the original controller program is:

> Do named reasoning actions such as `DECOMPOSE`, `REVISE`, `ANSWER`, and `SYNTHESIZE` induce stable, causally distinct transitions in frozen language models?

The project currently treats an action label as if it denotes a stable transition kernel. In practice, the transition is implemented by a natural-language prompt. Its meaning may change with prompt wording, worker family, model scale, current trace, target lens, and decoding randomness. A policy cannot learn a coherent action value if variation inside a nominal action is as large as variation between actions.

This is narrower than proposing another controller and can produce a useful negative result without training one. However, generic prompt sensitivity is already a mature literature. After the final collision check, this is best treated as a required validity screen and a supporting contribution unless it demonstrates large downstream control regret, model-by-action reversals, or structural collapse that ordinary prompt-sensitivity studies do not measure.

## Formal Question

For a replayable state `s`, nominal verb `v`, target `t`, prompt realization `p`, worker `m`, and decoding draw `z`, define the worker transition and exact terminal reward as:

```text
s' = T(s, v, t, p, m, z)
y  = R(fixed_continuation(s'))
```

The action interface is identifiable only if:

1. different nominal verbs cause meaningfully different transition or reward distributions;
2. independently written realizations of the same verb induce similar distributions;
3. the mapping is sufficiently stable across the intended worker models; and
4. at least two actions have crossed preferences on the same population of states.

The relevant quantities are:

- between-verb exact-reward effect;
- within-verb prompt-realization effect;
- verb by realization interaction;
- verb by worker interaction;
- statewise best-action agreement across realizations and workers;
- regret from transferring an action rule across prompt realizations or workers;
- transition fidelity, measured through deterministic parsing and state deltas; and
- best-fixed-to-statewise-oracle headroom.

Structural transition diagnostics are analysis only. The symbolic verifier remains the sole reward source.

## Closest Prior Art

### Common-prefix action value is occupied

[Calibration Is Not Control](https://arxiv.org/abs/2606.21399) defines intervention advantage and executes alternative actions from identical prefixes. A paper whose contribution is only common-prefix action uplift would not be novel.

### Prompted action libraries are occupied

- [RLoT](https://arxiv.org/abs/2505.14140) uses Reason-one-step, Decompose, Debate, Refine, and Terminate.
- [Agentic Chain-of-Thought Steering](https://arxiv.org/abs/2606.03965) uses strategy actions plus free-form steering messages.
- [Neural Chain-of-Thought Search](https://arxiv.org/abs/2601.11340) uses discourse operators around a frozen worker.
- [TemplateRL](https://arxiv.org/abs/2505.15692) defines template actions as prompts intended to elicit reasoning types, groups paraphrases under a shared action label, and ablates action-library size.

These papers generally assume that the label defines the intervention. TemplateRL's paraphrase construction is the closest pressure, but it does not report a same-state causal comparison of independently written realizations.

### Action-set diagnostics are conditional on the interface

[A Regime Theory of Controller Class Selection](https://arxiv.org/abs/2605.06339) makes its conclusions conditional on the chosen action set, loss, features, and judge. [ReasonOps](https://arxiv.org/abs/2605.29192) reports observational relationships between operators, difficulty, and model identity. Neither establishes that prompted verbs have stable causal semantics.

### Cross-worker transfer evidence conflicts

- RLoT reports broadly consistent cross-worker behavior.
- [AdaReasoner](https://arxiv.org/abs/2505.17312) reports degradation when transferring configurations across workers and describes model-specific reward landscapes.
- ReasonOps reports limited cross-model operator transfer.

The residual contribution is not generic transfer. It is a matched counterfactual test of the same nominal action on the same serialized state.

### Generic prompt sensitivity and action boundaries are occupied

- [POSIX](https://aclanthology.org/2024.findings-emnlp.852/) defines an intent-preserving prompt-sensitivity index and compares prompt perturbations across open models.
- [ProSA](https://aclanthology.org/2024.findings-emnlp.108/) studies prompt sensitivity, including reasoning-oriented tasks.
- [Flaw or Artifact?](https://aclanthology.org/2025.emnlp-main.1006/) finds that rigid or heuristic evaluation can exaggerate prompt sensitivity. The proposed exact symbolic reward avoids that particular confound.
- [Measuring LLMs' Sensitivity to Paraphrased Opinion Prompts](https://aclanthology.org/2026.wassa-1.5/) evaluates human-validated paraphrases across multiple models.
- [Action Boundary Blindness](https://aclanthology.org/2026.acl-long.1711/) studies action granularity, scope, and completeness across six agent benchmarks and reports gains from explicit boundary prompting.
- [Beyond Prompt Engineering](https://aclanthology.org/2026.findings-acl.2084/) studies lexical prompt sensitivity at large scale and identifies explicit action directives as a stability mechanism.

These papers lower standalone novelty. The required residual is not "prompts matter." It is that prompt realization changes the estimated transition kernel, best action, or downstream controller regret from an identical state under exact evaluation.

## Minimum Screen

### Population

- 60 fresh, exact-verifiable prefix states.
- Direct baseline accuracy between roughly 40 and 75 percent to avoid floor and ceiling effects.
- Two frozen workers from different model families, initially one 7B and one 14B or 32B model.
- Four nominal verbs with legal and meaningful effects at every selected state.
- Two independently authored prompt realizations per verb.

This produces 960 branch calls before any additional fixed continuation calls:

```text
60 states * 2 workers * 4 verbs * 2 realizations = 960
```

Use identical fixed continuation and verifier logic after every branch. If a continuation requires another worker call, count and match it in every arm.

### Prompt realizations

The two realizations should be written independently from the action contract, not produced by superficial synonym replacement. They should preserve intended inputs, target, output freedom, and budget while differing in instruction framing and phrasing.

Freeze them before outcomes are observed. Randomize arm execution order. Avoid choosing the better prompt after seeing test reward.

### Outcomes

Primary:

- exact terminal symbolic reward.

Secondary diagnostics:

- parser success;
- intended target changed or preserved;
- number and type of children created;
- target-local versus unrelated state changes;
- final-answer change;
- generated tokens, latency, and cost; and
- correct-to-wrong damage and wrong-to-correct rescue.

No LLM judge or learned process reward is used.

### Analysis

Fit a mixed-effects or hierarchical model with state as a repeated unit and terms for verb, realization, worker, and their interactions. Also report paired nonparametric estimates because binary terminal reward and 60 states may make asymptotic model estimates fragile.

Report:

1. best fixed action for each worker and realization;
2. statewise oracle and cross-fitted oracle;
3. pairwise action discordance;
4. best-action agreement across prompt realizations;
5. cross-worker action-rank correlation;
6. transfer regret across realizations and workers; and
7. the ratio of within-action realization variance to between-action variance.

## Go and Kill Rules

Proceed to a full 120-state, three-worker study if at least one of these holds:

- a prompt-realization effect changes exact reward by at least 5 points;
- a worker by action interaction reverses action rankings with at least 5 points of transfer regret;
- two or more actions each uniquely win on at least 10 screen states; or
- nominal actions collapse structurally despite different labels, with tight uncertainty.

Stop this as a standalone paper if all of the following hold:

- action rankings are stable across realizations and workers;
- transfer regret is below 2 points in every cell;
- realization effects are smaller than the preregistered practical threshold; and
- the stable action effects merely reproduce existing action-library ablations.

Stop adaptive policy training on this source if:

- the cross-fitted statewise oracle is less than 5 points above the best fixed action; or
- there is no bidirectional action complementarity.

## Full Study

The full version should use:

- 120 or more fresh states;
- three workers from at least two families, covering approximately 7B to 32B;
- two exact-verifiable domains;
- three independently written realizations for the most important verbs;
- grouped splits by problem family;
- fixed continuation and token-matched arms; and
- a held-out replication after all analysis choices are frozen.

The state corpus can then support two secondary analyses at low marginal cost:

1. cross-worker counterfactual action-ranking transfer; and
2. verb-versus-target-versus-interaction headroom on states with a crossed legal submatrix.

## Paper Framing

Possible title if the screen reveals large control consequences:

> Are Reasoning Actions Real? Identifiability of Prompted Interventions for Frozen Language Models

Defensible contributions:

- a causal definition and test of prompted-action identifiability;
- same-state, same-worker comparisons across independently authored realizations;
- exact-verifier evaluation of action stability and damage;
- cross-worker action-ranking transfer under controlled state; and
- a preflight criterion for deciding whether controller learning is meaningful.

If the screen finds only ordinary accuracy variation without action-rank reversals or policy regret, include it as a validity section in the semantic-state observability paper rather than presenting it alone.

Claims not to make:

- first prompted reasoning-action library;
- first common-prefix intervention comparison;
- first external controller for a frozen LLM;
- first cross-model controller-transfer study; or
- first use of decomposition, revision, or synthesis actions.

## Relationship to the Original Project

This pivot does not abandon the RL controller question. It tests whether its action space denotes a learnable MDP. If actions are stable and crossed preferences exist, the resulting outcome table becomes legitimate data for behavior cloning and off-policy Q-learning. If they are unstable, the immediate engineering response is to canonicalize, parameterize, or learn the worker-facing action realization before learning a policy over nominal verbs.
