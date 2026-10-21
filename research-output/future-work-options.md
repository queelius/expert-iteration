# Future-Work Options and Novelty Pressure

Cutoff: 2026-07-18

Status: completed comparison at the cutoff. Directions are ranked after checking both the closest paper's stated limitations and follow-on prior art.

## Governing Distinction

The controller must learn action benefit, not merely current-answer correctness.

For a frozen common-prefix state `s`, define an intervention's verifier-measured uplift relative to a baseline action:

```text
uplift(s, a) = R(rollout from s under a) - R(rollout from s under baseline)
```

For multiple actions, the useful estimand is an action-conditioned counterfactual outcome or conditional action treatment effect. This is deliberately different from generic confidence, correctness probability, process score, and Step Potential Advantage Estimation ([SPAE, arXiv:2601.03823](https://arxiv.org/abs/2601.03823)), which changes model weights and addresses RLVR credit assignment.

## Author-Stated Directions and Their Value

| Source | Author-stated limitation or direction | Prior-art pressure and project reading |
|---|---|---|
| [RLoT](https://arxiv.org/abs/2505.14140) | Noisy LLM self-evaluation leads to wrong actions; refine can overcorrect. The conclusion calls for more adaptive and efficient reasoning. | Motivates exact action-outcome labels and state comparison. It already occupies the tiny RL navigator. |
| [Adaptive Graph of Thoughts](https://arxiv.org/abs/2502.05078) | Optimize graph bounds, specialize component agents, apply fine-tuning or RL to edge selection and complexity checks, and reduce hallucination feedback loops. | A learned controller for semantic graph expansion is explicitly anticipated. The residual must show a causal advantage from content-aware live-node targeting, exact verification, or a particular state representation. |
| [Framework of Thoughts](https://arxiv.org/abs/2602.16512) | Build fully automatic dynamic prompting schemes, jointly optimize prompts and hyperparameters, improve graph interfaces, and relax safe graph-rewrite rules. | Generic learned dynamic-graph control is explicitly invited. A fast project should instead test a falsifiable assumption, such as whether semantic state predicts intervention value beyond topology and legal actions. |
| [GRADE](https://arxiv.org/abs/2606.22741) | Instrument dependencies, build data with observed dependencies and step faults, validate pruning, repair fragile structure, check stale reliance, and study anomaly detection. | Failure prediction and graph-based localization are occupied. Action-conditioned verifier uplift remains distinct, but GRADE's lean provenance-aware features and transfer protocol are mandatory baselines. |
| [Agentic CoT Steering](https://arxiv.org/abs/2606.03965) | Larger reasoners and adaptive budget selection. | Both are straightforward scale extensions, not strong novelty. Budget routing is heavily crowded. |
| [Offline Harness Control](https://arxiv.org/abs/2607.05458) | Online improvement, targeted support collection, native protocols, calibrated process metrics. | Support-aware collection is promising but explicitly claimed. Process metrics may be diagnostics, never reward here. |
| [AdaReasoner](https://arxiv.org/abs/2505.17312) | Continuous actions, gradient prompt generation, reduced per-task training, better transfer. | Continuous prompts and transfer are crowded. Conflicting transfer reports across papers support a boundary study. |
| [CoRefine](https://arxiv.org/abs/2602.08948) | Better confidence calibration, cheaper labels, non-math domains. | Generic calibration and cheaper supervision are crowded. Exact counterfactual action outcomes remain a cleaner label. |
| [HarnessBridge](https://arxiv.org/abs/2606.12882) | More domains and repeated evaluation. | Domain expansion alone is weak. Its observation/action projection makes state and lens comparisons timely. |
| [Multi-Head Latent Control](https://arxiv.org/abs/2607.14277) | Robustness, calibration, and more intervention heads. | Generic new heads are claimed. Compare latent state with symbolic and textual state under identical action labels instead. |
| [Tyen et al.](https://arxiv.org/abs/2311.08516) | More data and more sophisticated or iterative error classifiers. | A stronger localizer is incremental. The important empirical threshold is whether localization accuracy can exceed the net-benefit threshold. |
| [GLoRE](https://arxiv.org/abs/2402.10963) | Better critiques and verifier reasoning or tools. | Learned critics conflict with project reward invariants. Exact-verifier localization is already partly occupied by VERGE and CausalFlow. |
| [Calibration Is Not Control](https://arxiv.org/abs/2606.21399) | Richer controllers, more structured interventions, online action-conditioned policies. | This is very close to the proposed pivot and makes a generic multi-action extension high risk. |
| [CausalFlow](https://arxiv.org/abs/2605.25338) | Lower proposal and re-execution cost, better structured traces. | A learned cheap target policy is an obvious extension, so it needs a more specific claim and strong matched-compute evidence. |
| [MemoHarness](https://arxiv.org/abs/2607.14159) | Learned controllers, online accumulation, larger held-out splits, component attribution. | Nearly every generic harness-memory extension is already named. Fresh negative evaluation evidence lowers priority. |
| [TTHE](https://arxiv.org/abs/2607.08124) | Prequential tests, shuffled batches, information ablations, multiple seeds, compute matching, safety guardrails. | These are evaluation requirements, not an open method claim. |
| [DialogXpert](https://arxiv.org/abs/2505.17795) | Dynamic LLM action prior and multimodal input. | Dynamic action priors are claimed. Its subjective rewards and critic are outside scope. |
| [ReQueR](https://arxiv.org/abs/2604.25444) | Agentic and code workflows, recursive improvement. | It acknowledges cascading errors and inability to create capability absent from the worker. Screen worker/action headroom first. |

## Ranked Candidate Questions

The numbering below follows execution dependency as well as publication value. Candidate 1 is the fastest validity screen, but generic prompt-sensitivity and action-boundary work lowers its standalone novelty. The strongest paper package combines candidate 3 with candidate 1 as a preflight and candidate 2 as its action-headroom gate.

### 1. Action-Interface Identifiability

Question:

Do nominal reasoning actions induce stable, causally distinct transitions across independently authored prompt realizations and frozen workers?

Closest prior art:

- [Calibration Is Not Control](https://arxiv.org/abs/2606.21399) for same-prefix action advantage.
- [TemplateRL](https://arxiv.org/abs/2505.15692) for prompt-defined template actions, paraphrase grouping, and action-library ablations.
- [RLoT](https://arxiv.org/abs/2505.14140) and [Agentic CoT Steering](https://arxiv.org/abs/2606.03965) for named reasoning-action libraries.
- [Regime Theory](https://arxiv.org/abs/2605.06339), whose diagnostics are conditional on the chosen action interface.
- [POSIX](https://aclanthology.org/2024.findings-emnlp.852/), [ProSA](https://aclanthology.org/2024.findings-emnlp.108/), and [Action Boundary Blindness](https://aclanthology.org/2026.acl-long.1711/) for generic prompt and action-interface sensitivity.

Why it may still matter:

- Existing controller papers generally assume that a prompt-defined action label denotes one intervention.
- No located primary source re-executes identical states with independently authored realizations and compares within-action variation with between-action variation under exact reward.
- A negative result would identify a basic MDP misspecification that affects both LLM teachers and learned policies.

Novelty risk: medium to high.

Feasibility: very high.

First experiment:

- 60 states, two workers, four verbs, and two prompt realizations, for 960 branch calls before fixed continuations.
- Report verb, realization, worker, and interaction effects; structural transition deltas; best-action agreement; transfer regret; and best-fixed-to-oracle headroom.

Kill criterion:

- Stop if actions are stable across prompts and workers, transfer regret remains below 2 points, and the result adds nothing beyond existing action-library ablations.

Full design: [`action-interface-identifiability.md`](action-interface-identifiability.md).

### 2. Sequential Action-Viability Census

Question:

Can a complete or randomized common-prefix action-outcome matrix determine, before RL training, whether the correct policy class is fixed, partitioned, instance-adaptive, or sequential?

Closest prior art:

- [Regime Theory](https://arxiv.org/abs/2605.06339)
- [Statistical Scouting](https://arxiv.org/abs/2605.09618)
- [Select-then-Solve](https://arxiv.org/abs/2604.06753)
- [Calibration Is Not Control](https://arxiv.org/abs/2606.21399)

Why it may still matter:

- Existing work largely studies one-shot protocol or intervention selection.
- A typed `(verb, target)` action space creates combinatorial support and sequential identifiability problems.
- Exact verification makes complete counterfactual labels possible on suitable tasks.

Novelty risk: medium.

Feasibility: high.

First experiment:

1. Choose two workers and at least two exact-verifier domains.
2. At the same replayable prefix, execute every admissible action under common random seeds where supported.
3. Report best fixed, per-instance oracle, target-only oracle, verb-only oracle, random, and compute-matched repeated sampling.
4. Measure bidirectional discordance, oracle gap, support by action, and cross-fitted predictability.

Kill criterion:

- Stop adaptive policy work if the best fixed action is within a preregistered small margin of the instance oracle across workers and domains.

### 3. Which Observable State Predicts Action Uplift?

Question:

Under identical frozen-worker counterfactuals, which state representation best predicts exact action uplift: raw trace text, self-evaluation, confidence features, symbolic forest plus verifier receipts, frozen text embeddings, hidden trajectories, or combinations?

Closest prior art:

- RLoT for self-evaluation state.
- CoRefine for confidence trajectories.
- Multi-Head Latent Control for hidden trajectories.
- HarnessBridge for compressed active state.
- CausalFlow and VERGE for structured and verifier-derived state.

Why it may still matter:

- No located paper performs this controlled comparison with the same actions, frozen workers, and exact counterfactual labels.
- A negative result is interpretable: the action may have headroom but the state lacks sufficient predictive information.

Novelty risk: medium to high.

Feasibility: high for text, symbolic, and embedding states; medium for hidden states.

First experiment:

- Use the census from candidate 1.
- Freeze the train/test split before feature extraction.
- Compare calibrated linear probes, small MLPs, and the planned factored controller at matched capacity.
- Score action regret, policy value, calibration, and cross-worker transfer, not just classification accuracy.

Kill criterion:

- If no representation beats best fixed out of sample despite stable oracle headroom, report an observability limit and stop scaling policy complexity.

### 4. One-Call Semantic Node Intervention

Question:

Under exactly one additional worker call, can a policy select a semantic reasoning node and retry lens that beats whole-root independent revision?

Closest prior art:

- Tyen et al., GLoRE, StepCo, Structure Enables Effective Self-Localization, CausalFlow, Calibration Is Not Control, VERGE, and Mechanic.

Why it may still matter:

- The exact one-call constraint and factorization across semantic targets and intervention verbs are narrower than most existing methods.
- It tests the original lens hypothesis directly.

Novelty risk: high.

Feasibility: contingent. E43R1 currently fails the substrate test.

Prerequisites:

1. The structured direct baseline must approach the raw worker baseline.
2. At least two targets must have bidirectional wins.
3. The per-instance target oracle must beat whole-root revision materially.
4. A held-out target predictor must recover a meaningful part of that gap.

Kill criterion:

- Stop after a preregistered pilot if branch lenses remain worse than root retry or if only one action dominates.

### 5. Support-Aware Counterfactual Collection

Question:

Can data collection request only missing or uncertain `(state, verb, target)` outcomes and close the offline policy gap more efficiently than uniform exploration?

Closest prior art:

- Offline harness control explicitly proposes targeted support collection.
- Standard DAgger, active learning, and offline-to-online RL cover the generic algorithms.
- [LLM4Cov](https://arxiv.org/abs/2602.16953) is adjacent on coverage-guided collection.

Novelty risk: medium to high.

Feasibility: medium.

Gate:

- Do not run until candidates 1 and 2 establish both complementarity and predictable action uplift.

### 6. Cross-Worker Boundaries of Controller Transfer

Question:

When do exact-verifier symbolic controller features transfer across frozen workers, and when are action values worker-specific?

Motivation:

- RLoT reports transfer.
- AdaReasoner reports meaningful transfer loss and model-specific reward landscapes.
- ReQueR and HarnessBridge make broader generalization claims.

Novelty risk: medium to high for a boundary study, very high for a generic universal-controller claim.

Feasibility: medium.

### 7. Prequential Harness Adaptation

Question:

Can exact-verifier harness changes selected on previous tasks improve the next held-out batch?

Prior-art pressure:

- TTHE explicitly proposes the evaluation.
- MemoHarness searches related control surfaces.
- [Rethinking the Evaluation of Harness Evolution](https://arxiv.org/abs/2607.12227) reports no consistent advantage over simple matched-budget scaling and weak held-out generalization.

Novelty risk: very high.

Feasibility: medium.

Priority: low.

### 8. RGoT Simulator Sufficiency Audit

Question:

Does the fitted success model used by Reinforced Graph of Thoughts preserve the policy that would be optimal against actual frozen-LLM transitions?

Closest prior art:

- [Reinforced Graph of Thoughts](https://arxiv.org/abs/2605.22195) models operation success primarily from complexity and trains the graph controller cheaply in that simulator.
- Generic simulator misspecification and sim-to-real control are mature areas, so the result must change the selected LLM policy rather than report calibration alone.

First experiment:

- Reuse the released RGoT implementation on two workers and three deterministic tasks.
- Compare its complexity-only success model with item- and history-aware models on held-out Brier score and log loss.
- Train the same cheap controller in each simulator and measure regret against real worker executions.

Novelty risk: medium.

Feasibility: medium to high.

Kill criterion:

- Stop if richer conditioning improves Brier score by less than 0.02 and changes real-worker policy value by less than 2 points across tasks and workers.

## Provisional Recommendation

Run candidate 1 before learning another controller. If prompted actions are identifiable and exhibit crossed preferences, combine candidates 2 and 3 into the primary controller study:

> Can a frozen-LLM controller beat its best fixed harness? A study of sequential action uplift under exact verification.

Use four gates:

1. Identifiability: require nominal actions to be distinguishable and sufficiently stable across prompt realizations.
2. Complementarity: require bidirectional discordance and a meaningful best-fixed-to-oracle gap.
3. Predictability: require held-out, cross-fitted prediction of per-action verifier uplift.
4. Policy value: require the learned policy to beat raw generation, the best fixed action and lens, simple routers, and compute-matched sampling.

If identifiability fails, fix or study the action interface before policy learning. If complementarity fails, use the best fixed verified harness. If complementarity passes but predictability fails, the defensible result is a limit-of-observability study. Only if the first three gates pass should behavior cloning and off-policy Q-learning proceed.

## Minimal Targeted-Retry Design

If the one-call semantic-node question is retained, start with the smallest action space that can falsify it.

State:

```text
S = {
  problem,
  decomposition_tree,
  dependencies,
  node_goals,
  node_outputs,
  root_draft,
  attempt_history,
  remaining_budget
}
```

Actions:

```text
KEEP
RETRY_ROOT
RETRY(node_i, dependency_path_lens)
```

Do not initially combine verb discovery with target selection. Use one retry verb, one fixed local lens, and one common synthesis transition. Only after target selection shows value should a separate experiment add `CRITIQUE_AND_REVISE`.

The controller must score legal actions from terminal symbolic outcomes. It must not learn an error probability and treat it as action value. A wrong node may not be worth retrying, and retry can damage a correct answer.

## Required Baselines and Measures

Baselines:

- raw direct generation;
- compute-matched independent whole-answer sampling;
- `KEEP`;
- `RETRY_ROOT`;
- random valid node;
- first, last, and deepest node;
- lowest-confidence node, when confidence is available without extra worker calls;
- first-error localizer;
- worker self-selection;
- GPT-4.1-mini or a stronger teacher selector;
- learned action-value selector;
- cross-fitted oracle, estimated on retry seeds disjoint from evaluation seeds.

Primary measures:

- exact symbolic accuracy;
- wrong-to-correct rescue rate;
- correct-to-wrong damage rate;
- net lift;
- best-fixed-to-oracle gap;
- oracle-gap closure;
- selector regret;
- selected-node output change;
- final-answer change;
- propagation or synthesis failure;
- worker calls, generated tokens, latency, and cost per rescue.

Statistical requirements:

- Use paired bootstrap confidence intervals and McNemar tests for paired correctness.
- Estimate an oracle with repeated development outcomes and evaluate it on held-out retry randomness. A maximum over one noisy branch per action is optimistically biased.
- Split related problem templates or sources as groups.
- Test at least two worker families and more than one scale before making a general claim.

## Final Stop Rules

1. Stop the controller direction if a cross-fitted node oracle does not materially beat the best fixed target and whole-answer retry.
2. Drop the lens claim if token-matched full-context retry performs as well as the local lens.
3. Drop the learned-controller claim if it does not beat the best heuristic, worker self-selection, strong-LLM routing, and compute-matched sampling.
4. Stop target-policy training if structured generation remains materially worse than raw generation.
5. Treat stable oracle headroom with failed deployable selectors as a limit-of-observability result, not as evidence that a larger controller will necessarily solve the problem.
