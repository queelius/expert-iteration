# Fast, Tractable Research Pivot Portfolio

Cutoff: 2026-07-18

Status: completed novelty review at the cutoff. The ranking favors questions that one person can answer quickly, that retain value under a negative result, and that are less likely to be preempted than a generic controller method.

## 1. Decision Summary

The leading paper is a semantic-state observability audit rather than another orchestration algorithm:

> Does semantic state predict exact action value beyond legal menu, phase, topology, node identifiers, and serialization order?

It requires an action-identifiability preflight that combines:

1. identical replayable prefix states;
2. every candidate action from each prefix;
3. independently authored prompt realizations for each nominal verb;
4. two frozen workers from different families; and
5. exact downstream verifier outcomes under a fixed continuation.

[Calibration Is Not Control](https://arxiv.org/abs/2606.21399) makes common-prefix action advantage occupied, and [TemplateRL](https://arxiv.org/abs/2505.15692) makes prompted template actions and action-library ablations occupied. [POSIX](https://aclanthology.org/2024.findings-emnlp.852/), [ProSA](https://aclanthology.org/2024.findings-emnlp.108/), and [Action Boundary Blindness](https://aclanthology.org/2026.acl-long.1711/) also make generic prompt and action-interface sensitivity crowded. Action stability is therefore a validity gate and supporting result unless it changes downstream policy rankings or regret substantially.

If the actions are stable and have crossed preferences, the second study asks whether semantic controller state predicts their exact utility beyond legal menu, phase, topology, identifiers, and order. If the actions are not stable, action canonicalization or realization becomes the research problem and policy training should stop.

## 2. Ranking

Scores use a 1 to 5 scale. Higher feasibility and negative-result value are better. Higher preemption risk is worse.

| Rank | Question | Novelty | Feasibility | Negative-result value | Preemption risk |
|---|---|---:|---:|---:|---:|
| 1 | Does semantic state add action value beyond menu and phase? | 4 | 4.5 | 5 | 2 to 3 |
| 2 | Are prompted reasoning actions stable interventions? | 2.5 | 5 | 4 | 4 |
| 3 | Are node-target policies equivariant to node IDs and ordering? | 4 | 5 | 2 | 2 |
| 4 | Which lens is causally sufficient for a focused retry? | 3.5 | 4 | 5 | 3 |
| 5 | Does value come from the verb, target, or their interaction? | 3 | 4 | 5 | 4 |
| 6 | Should rich structure be private controller state rather than worker-facing format? | 3 | 4.5 | 5 | 3 to 4 |
| 7 | Is RGoT's fitted LLM simulator control-sufficient? | 3.5 | 3.5 | 4 | 2 to 3 |
| 8 | Is failure dominated by targeting, local execution, or propagation? | 2.5 | 4 | 5 | 4 |

## 3. Leading Screen: Are the Actions Identifiable?

Question:

> Does a nominal action have a more stable causal effect than an independently written realization of its prompt, and does that action meaning transfer across workers?

Minimum screen:

1. Select 60 fresh exact-verifiable prefix states with direct accuracy away from floor and ceiling.
2. Use two workers from different families, four meaningful verbs, and two independently authored prompt realizations per verb.
3. Execute all 960 state, worker, verb, and realization branches under a fixed continuation.
4. Measure exact reward, transition deltas, best-action agreement, prompt-transfer regret, worker-transfer regret, best fixed value, and a cross-fitted statewise oracle.
5. Fit verb, realization, worker, and interaction effects while treating state as a repeated unit.

Continue if action rankings reverse across prompts or workers, prompt wording changes exact reward by at least 5 points, or at least two actions each uniquely win on at least 10 states. Stop adaptive policy work on the source if the cross-fitted oracle is less than 5 points above best fixed or preferences are not bidirectional.

The full design, overlap analysis, and paper framing are in [`action-interface-identifiability.md`](action-interface-identifiability.md).

## 4. Does the Controller Read the State?

### 4.1 Semantic observability

Question:

> With topology, phase, and legal actions held fixed, does semantic node content improve terminal action selection?

Why this is distinct:

- [Calibration Is Not Control](https://arxiv.org/abs/2606.21399) compares scalar risk with full-prefix action-conditioned control, but does not hold a structured legal menu and topology fixed while ablating node semantics.
- [The Effect of State Representation on LLM Agent Behavior](https://arxiv.org/abs/2506.15624) varies natural-language history in routing games, not controller prediction of intervention utility.
- [From Context to Action](https://arxiv.org/abs/2410.23555) studies context in web agents, not a semantic-node action interface around a frozen worker.
- [HarnessBridge](https://arxiv.org/abs/2606.12882) learns a projected generator-visible state, but does not test whether a separate factored controller uses semantic node content beyond the legal menu.

Minimum decisive experiment:

1. Collect 80 to 120 common-prefix states with 3 to 4 legal actions and at least three outcome replicates per action.
2. Require at least 20 percent of states to have nonconstant action values and require both helpful and harmful interventions.
3. Group train and test splits by source problem or template.
4. Compare menu-only, topology plus menu, full semantic state, shuffled semantic text, and per-action value models.
5. Construct contrast sets whose menus and topology match but whose best empirical actions differ.
6. Evaluate held-out terminal symbolic reward, action regret, and best-fixed-to-policy lift.

Kill criteria:

- Stop before controller training if a cross-fitted oracle is less than 5 points above the best fixed action.
- Stop if fewer than 20 percent of states show action heterogeneity.
- Drop the semantic-state claim if full state fails to beat menu-only by at least 5 points and the paired interval includes zero.

Negative result:

A replicated null across two workers or task families would support a useful conclusion: the elaborate semantic state is not operationally observable to the tested controller, even when the action space has measurable headroom.

### 4.2 Node identity and order equivariance

Question:

> Does the target selector follow semantic nodes, or positions and identifiers?

Experiment:

1. Apply 20 random node-ID renamings and sibling-order permutations to each held-out state.
2. Remap legal actions and expected targets under the same permutation.
3. Measure equivariance, selected-target consistency, and replayed terminal utility for the teacher LLM, flat classifier, factored NN, and a permutation-equivariant or permutation-augmented model.

This requires no new worker calls when action outcomes map bijectively.

Kill criteria:

- Do not treat this as a standalone result if policies retain more than 95 percent equivariance and lose less than 2 points of replay utility.
- A meaningful shortcut finding requires more than 10 points of degradation and recovery of at least half the loss through architecture or augmentation.

### 4.3 Unseen-menu and tree-size generalization

Question:

> Does the factorization into a verb head and node pointer provide compositional generalization, or only memorize legal-menu signatures?

Experiment:

- Split the action-value corpus by complete legal-menu signature and node count.
- Compare exact-menu lookup, flat pair classification, separate verb and pointer heads, and direct per-action value scoring.
- Use held-out verifier utility as the primary outcome and imitation accuracy only as a diagnostic.

Kill criterion:

The factored model must beat lookup and flat classification by at least 5 points on unseen menus while staying within 2 points in distribution. Otherwise the factorization has no demonstrated benefit.

## 5. Independent Fallback: Causal Lens Sufficiency

Question:

> With target and token count fixed, which local information is necessary for a retry to improve the final verified answer?

Closest pressure:

- [Decomposed Prompting](https://arxiv.org/abs/2210.02406) establishes modular subproblem prompting.
- [Question Decomposition Improves Faithfulness](https://arxiv.org/abs/2307.11768) studies decomposition and context.
- [THREAD](https://aclanthology.org/2025.naacl-long.427/) uses isolated child contexts that return selected tokens to a parent.
- [Mechanic](https://arxiv.org/abs/2603.24465) isolates formal subgoals in clean local contexts.
- [Decomposed Prompting Does Not Fix Knowledge Gaps](https://arxiv.org/abs/2602.04853) supplies recent negative evidence.

The residual is a controlled dependency-relevance study, not the broad claim that local context helps.

Minimum experiment:

1. Select 100 fresh common-prefix targets.
2. Compare target-only, ancestor path, dependency path, preceding sibling, local subtree, token-matched full context, and token-matched distractor lenses.
3. Hold the worker, prompt contract, target, retry seed, and token count fixed where possible.
4. Manipulate one interference variable at a time, such as irrelevant sibling count or a conflicting sibling result.
5. Measure exact terminal success, local output change, final-answer propagation, rescue, damage, and tokens.

Kill criteria:

- Stop the pilot unless the dependency lens gains at least 5 net points over both target-only and token-matched full context.
- Require a paired interval excluding zero in a preregistered high-interference regime.
- If the gain disappears under token matching, attribute it to context length rather than dependency structure.

## 6. Private Structure, Natural Worker View

Hypothesis:

> Rich typed state may help the controller while simultaneously harming a capacity-limited worker when exposed as a schema or required output format.

Why it fits the local evidence:

- E43R1 forced structured generation and obtained 1/12 structured submits and 2/12 branch retries, versus 6/12 for direct or root retry.
- [Let Me Speak Freely?](https://aclanthology.org/2024.emnlp-industry.91/) reports that stricter format restrictions can reduce reasoning accuracy.
- [The Format Tax](https://arxiv.org/abs/2604.03616) reports that prompt-side format instructions cause most of the loss for open-weight models and that separating reasoning from formatting recovers much of it.
- [The Constraint Tax](https://arxiv.org/abs/2605.26128) reports a validity-correctness tradeoff for sub-3B models and recommends free reasoning followed by late packaging.
- [Capacity, Not Format](https://arxiv.org/abs/2606.09410) reports information-matched and schema-gradient evidence that the penalty is largest near the model's capability boundary.

Possible crossed experiment:

| Controller observation | Worker lens and output |
|---|---|
| flat text | free-form natural language |
| typed tree | free-form natural language |
| flat text | simultaneous structured output |
| typed tree | simultaneous structured output |
| typed tree | free reasoning, then external or second-stage parsing |

This could separate state value from worker format tax. However, the generic separation principle is occupied. The residual would have to be the two-consumer interaction: what should be visible to the controller versus the frozen worker, under identical underlying information and exact outcomes.

The staged information-matched experiment is specified in [`controller-worker-interface.md`](controller-worker-interface.md).

Novelty kill:

Drop this as a paper claim if a primary source already crosses controller representation with generator-visible rendering, or if a simple free-form worker condition removes all apparent benefit of controller-side structure.

## 7. Lower-Priority Diagnostics

### Stage bottlenecks

Measure three oracle stages separately:

1. target oracle with a fixed repair prompt;
2. repair oracle given the correct target; and
3. propagation oracle given a successful local repair.

This is useful engineering evidence, but [Tyen et al.](https://arxiv.org/abs/2311.08516), [GLoRE](https://arxiv.org/abs/2402.10963), and [Thought-ICS](https://arxiv.org/abs/2602.02416) already separate localization and correction. Do not present the decomposition itself as novel.

### Model-scale threshold

The question of when decomposition helps across scale is crowded by compositionality-gap and decomposition studies. Pursue only if a small pilot shows a replicated sign reversal across model sizes with component verification and matched prompts.

### Teacher versus outcome labels

Compare GPT-4.1-mini action choices with actual return-derived actions for internal data-quality decisions. This is not a standalone paper because behavior-cloning versus return-weighted training and action-conditioned value learning are already occupied.

## 8. Immediate Work Order

1. Run the 960-branch action-identifiability screen on a fresh source.
2. Compute cross-worker action-ranking transfer from the same branches.
3. Run the node permutation audit on all reusable controller states.
4. Determine whether existing logs contain matched-menu contrast sets and action-outcome heterogeneity.
5. Run semantic observability probes only after action stability and headroom pass.
6. Run a 20 to 30 item free-form versus structured-output pilot before investing in new decomposition code.
7. Proceed to controller training only if action identifiability, complementarity, and semantic predictability pass.

## 9. Claims to Avoid

- Do not claim that external control of a frozen LLM is novel.
- Do not claim that common-prefix intervention advantage is novel.
- Do not claim that a semantic reasoning tree, graph-derived lens, or child writeback is novel.
- Do not claim that free reasoning followed by delayed structure is novel.
- Do not claim controller value from imitation accuracy.
- Do not infer learnability from an in-sample action oracle.
