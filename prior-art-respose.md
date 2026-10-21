# Prior Art and Research Direction Assessment

Cutoff: 2026-07-18

This memo assesses the original external-controller project, the explicit tree-state and `(verb,target)` design, the targeted-retry pivot, and faster research alternatives. It distinguishes reported paper results from this review's inferences. Recent 2026 arXiv manuscripts are treated as priority evidence, not settled consensus.

Detailed evidence is preserved in:

- [`research-output/source-ledger.md`](research-output/source-ledger.md): paper-by-paper evidence and overlap analysis;
- [`research-output/tree-state-prior-art.md`](research-output/tree-state-prior-art.md): explicit state, node action, lens, and writeback lineage;
- [`research-output/future-work-options.md`](research-output/future-work-options.md): author-stated future work and novelty pressure;
- [`research-output/fast-pivot-portfolio.md`](research-output/fast-pivot-portfolio.md): ranked one-person pivots and kill criteria;
- [`research-output/action-interface-identifiability.md`](research-output/action-interface-identifiability.md): the required action-validity preflight; and
- [`research-output/controller-worker-interface.md`](research-output/controller-worker-interface.md): the controller-state versus worker-language experiment.

## Executive Decision

### 1. The broad original claim is not novel

The claim that a lightweight external neural policy can orchestrate a frozen small LLM through prompted reasoning or harness actions is directly occupied. The most important precedents are:

- [RLoT](https://arxiv.org/abs/2505.14140): a sub-3,000-parameter DQN selects prompted reasoning operations, including decompose and refine, for frozen 7B to 14B and API workers.
- [Agentic Chain-of-Thought Steering](https://arxiv.org/abs/2606.03965): a separate controller receives a supervised teacher warm start and then RL training while frozen 1.5B to 8B reasoners execute its steering actions.
- [Learning to Control LLM Agent Harnesses with Offline RL](https://arxiv.org/abs/2607.05458): a 64-hidden-unit external MLP learns typed harness actions around a frozen LLM from an offline buffer.
- [Neural Chain-of-Thought Search](https://arxiv.org/abs/2601.11340): tiny external heads distill teacher preferences over reasoning operators for frozen 1.5B and 7B workers.
- [Reinforced Graph of Thoughts](https://arxiv.org/abs/2605.22195): small RL networks learn to append, backtrack, or stop in a graph of operations executed by a fixed LLM.

The July 5 harness-control paper is unusually close, but it is not the only or earliest decisive collision. RLoT is closer to the original tiny reasoning controller. Agentic CoT Steering is closer to the teacher-warm-start plus RL recipe.

### 2. The explicit tree, action pointer, lens, and writeback are also occupied as components

[Graph of Thoughts](https://arxiv.org/abs/2308.09687) already maintains a graph reasoning state, selects transformations and target thoughts, renders selected graph context into prompts, parses results, and writes them back. [DeAR](https://proceedings.neurips.cc/paper_files/paper/2024/hash/01025a4e79355bb37a10ba39605944b5-Abstract-Conference.html) maintains a semantic subquestion tree and applies Decompose, Analyze, and Rethink with child-to-parent updates. [TacticZero](https://arxiv.org/abs/2102.09756) is an older formal precedent for a learned action explicitly factored as target fringe, target goal, tactic, and arguments.

Recent work narrows the claim further:

- [Atomic Task Graph](https://arxiv.org/abs/2607.01942) provides recursively refined semantic nodes, node-relevant history, node-local state, localized repair, and preservation of validated regions.
- [TopoAgent](https://arxiv.org/abs/2607.14658) provides dependency-derived node lenses, node memory, writeback, and failure-triggered node fission.
- [Framework of Thoughts](https://arxiv.org/abs/2602.16512) separates a mutable execution graph from a derived reasoning graph and restricts each operation to a scoped ancestor and descendant view.
- [ReCAP](https://proceedings.neurips.cc/paper_files/paper/2025/hash/88af3540325dd0b70617a9ab605f294d-Abstract-Conference.html), [THREAD](https://aclanthology.org/2025.naacl-long.427/), and [STEP](https://arxiv.org/abs/2506.21030) implement recursive child contexts and return or propagation interfaces.

The exact conjunction of a tiny learned pointer, arbitrary live natural-language nodes, deterministic versioned lenses, a frozen small worker, and terminal symbolic reward was not found. That is a narrow integration gap, not a persuasive architecture-level novelty claim.

### 3. The current targeted-retry pivot is high risk

Learning where to retry is crowded by error localization, prefix-preserving regeneration, local refinement, thought-level backtracking, exact repair, and action-conditioned intervention work. The closest pressures include [Tyen et al.](https://arxiv.org/abs/2311.08516), [GLoRE](https://arxiv.org/abs/2402.10963), [Thought-ICS](https://arxiv.org/abs/2602.02416), [CausalFlow](https://arxiv.org/abs/2605.25338), [Calibration Is Not Control](https://arxiv.org/abs/2606.21399), and [Mechanic](https://arxiv.org/abs/2603.24465).

The one-call semantic-node retry question remains testable, but it currently has both high novelty risk and negative local evidence. It should not be the first new investment.

### 4. The immediate preflight is an action-interface audit

The first validity question is:

> Do named reasoning actions define stable, causally distinct transition kernels for frozen LLMs?

Controller papers treat labels such as `DECOMPOSE`, `REVISE`, or `SYNTHESIZE` as actions, but the transition is implemented by natural-language instructions. The nominal action may change meaning across prompt realizations, worker families, model scales, state renderings, and targets. If within-action prompt variation is as large as between-action variation, the controller is learning an ill-specified MDP.

[Calibration Is Not Control](https://arxiv.org/abs/2606.21399) already covers same-prefix action advantage. [TemplateRL](https://arxiv.org/abs/2505.15692) already covers prompt-defined template actions, paraphrase grouping, and action-library ablations. General prompt sensitivity is also crowded by [POSIX](https://aclanthology.org/2024.findings-emnlp.852/), [ProSA](https://aclanthology.org/2024.findings-emnlp.108/), and [Flaw or Artifact?](https://aclanthology.org/2025.emnlp-main.1006/). [Action Boundary Blindness](https://aclanthology.org/2026.acl-long.1711/) directly studies action granularity, scope, and completeness in LLM agents.

The residual is a same-state causal comparison of independently authored action realizations that measures exact downstream policy ranking and control regret. This is a fast and necessary diagnostic, but after the final collision check its standalone novelty is only medium. The stronger paper is the semantic-state observability audit, with action stability as a validity section.

### 5. Do not train another controller until four gates pass

1. **Action identifiability:** nominal actions must be distinguishable and sufficiently stable across prompt realizations.
2. **Complementarity:** at least two actions must have bidirectional wins and a meaningful cross-fitted oracle gap over the best fixed action.
3. **Observability:** held-out state must predict exact action uplift, not merely current-answer correctness.
4. **Policy value:** a learned policy must beat direct prompting, compute-matched sampling, and the best fixed harness on untouched data.

The current evidence does not pass these gates.

## System Being Assessed

The original proposal combines:

1. a small external neural controller, eventually around 5 million parameters or less;
2. a frozen 3B to 32B worker that performs language generation;
3. a persistent nested workspace of goals, subgoals, facts, attempts, and candidates;
4. factored actions represented as `(verb,target_live_node)`;
5. an action- and target-specific lens shown to the worker;
6. parsed worker output written back near the target;
7. behavior-cloning initialization from a strong LLM teacher;
8. off-policy Q-learning from actual transitions;
9. terminal symbolic verification as the only reward; and
10. optional search as an additive accelerator rather than the method's spine.

The intended scientific claim is not simply that the system can run. It must show that the learned policy, structured state, or lens provides value over simpler fixed and direct alternatives under matched compute.

## What the Local Experiments Establish

| Experiment | Result | Valid interpretation |
|---|---:|---|
| E39 | direct 43/60; teacher-controlled retained revision 43/60 | GPT-4.1-mini made revision decisions, but the end-to-end teacher did not improve accuracy. Candidate oracle was 46/60. |
| E40 | fixed retained revision 49/60; direct 43/60 | A fixed revision prompt improved six cases with no direct-only loss on this source. |
| E41 | fixed independent revision 55/60; retained revision 49/60; direct 43/60 | The strongest observed result is a fixed prompt, not adaptive control. Independent revision had 12 unique wins and no direct-only losses. |
| E42 | generic coordinator 46/60; explicit coordinator 51/60; fixed oracle 55/60 | Coordinator prompting matters, but the GPT-4.1-mini coordinator did not beat the fixed independent-revision policy. This was replay analysis. |
| E43 | invalid | Schema failure prevents a target-policy conclusion. |
| E43R1 | structured submit 1/12; branch retry 2/12; root retry/direct 6/12 | Calls were accepted, but the current decomposition, lenses, and structured generation harmed the Qwen2.5-3B worker. |

Honest conclusion:

- There is real prompt-engineering progress: E41 found a much stronger fixed revision procedure.
- There is no demonstrated LLM-controller gain over the best fixed policy.
- E41 does not contain routing headroom because independent revision weakly dominates direct generation on the observed pairs.
- The current structured workspace and branch lenses have not earned their complexity.
- These are development results on a small source, not publication-ready general claims.

## Direct Controller Prior Art

| Work | What it already establishes | Material difference from this project |
|---|---|---|
| [RLoT](https://arxiv.org/abs/2505.14140) | Tiny DQN, frozen small workers, prompted reasoning verbs, sequential MDP | Learned PRM, LLM self-evaluation state, no semantic node pointer, repeated calls |
| [Agentic CoT Steering](https://arxiv.org/abs/2606.03965) | Separate controller, teacher trajectories, supervised warm start, RL, frozen reasoners | 4B controller, free-form steering, no exact symbolic reward or persistent forest |
| [Offline Harness Control](https://arxiv.org/abs/2607.05458) | Small MLP, typed harness actions, frozen executor, offline RL | Compact flat harness state, task-rubric reward, no semantic target or lens |
| [CoRefine](https://arxiv.org/abs/2602.08948) | Roughly 211K controller selects halt, rethink, or alternative for frozen 8B and 32B models | Confidence trajectory state, no target pointer, oracle-derived labels |
| [Neural CoT Search](https://arxiv.org/abs/2601.11340) | Tiny heads, frozen worker, teacher-distilled reasoning operators, optional search | Hidden-state and KV access, discourse positions rather than semantic nodes |
| [Dialogue Action Tokens](https://arxiv.org/abs/2406.11978) | Behavior initialization plus off-policy actor-critic control of frozen 7B and 8B LLMs | Continuous action tokens and subjective dialogue reward |
| [MaAS](https://arxiv.org/abs/2502.04180) | Neural controller builds query-specific typed operator DAGs for a fixed LLM | Plans before intermediate content exists; no live semantic-node pointer |
| [HarnessBridge](https://arxiv.org/abs/2606.12882) | Learned observation projection, active-state indexing, and action mediation around a frozen agent | 0.8B controller, coding focus, supervised LLM-judged labels |
| [AdaReasoner](https://arxiv.org/abs/2505.17312) | Factorized external heads select prompt, temperature, and reasoning steps | Contextual bandit over query embeddings, external evaluators, no semantic state |
| [Prompting Policies](https://arxiv.org/abs/2605.14443) | Lightweight RL prompter intervenes on a frozen black-box worker | Uses scalar rewards and textual critiques; no exact structured target interface |
| [PPDPP](https://arxiv.org/abs/2311.00262) | Plug-in policy planner with supervised initialization followed by RL around frozen LLMs | Dialogue strategies and large controller rather than exact-verifier reasoning |

Inference from this table:

> Controller size, frozen-worker control, typed actions, teacher initialization, off-policy learning, factorized heads, and adaptive lenses are not individually novel. A contribution must show a specific empirical phenomenon that these systems do not test.

## Explicit State, Action, Lens, and Writeback Prior Art

### Semantic task-state graphs

| Work | Overlap with the proposed workspace | Remaining difference |
|---|---|---|
| [Graph of Thoughts](https://arxiv.org/abs/2308.09687) | Persistent typed thought graph, operation plus thought selection, graph-derived prompt, parser writeback | Static human-authored operation graph, not learned arbitrary pointer |
| [DeAR](https://proceedings.neurips.cc/paper_files/paper/2024/hash/01025a4e79355bb37a10ba39605944b5-Abstract-Conference.html) | Semantic subquestion tree, Decompose/Analyze/Rethink, child-local work, parent update | Fixed queue and LLM cycle, no tiny policy or isolated lens ablation |
| [Adaptive Graph of Thoughts](https://arxiv.org/abs/2502.05078) | Hierarchically addressed DAG, semantic subproblems, selective recursive expansion | LLM complexity heuristic rather than compact learned pointer |
| [Knowledge Graph of Thoughts](https://arxiv.org/abs/2504.02670) | Persistent graph store, graph queries, tool choice, result integration | Knowledge graph and LLM controller, not exact-reward neural pointer |
| [Atomic Task Graph](https://arxiv.org/abs/2607.01942) | Interface-preserving node refinement, relevant-history lens, node state, local repair, validated-region freezing | Fixed compilation and scheduling rather than learned arbitrary action |
| [TopoAgent](https://arxiv.org/abs/2607.14658) | DAG subproblems, prerequisite lens, node memory, writeback, child fission | Fixed topological schedule and timeout rule |
| [Framework of Thoughts](https://arxiv.org/abs/2602.16512) | Dynamic execution and reasoning graphs, scoped visibility, local graph rewrites | Framework scheduler rather than learned semantic-node control |
| [GRADE](https://arxiv.org/abs/2606.22741) | Execution and dependency graph with edge provenance; failure prediction and localization | Passive run diagnosis rather than writable task state or action value |

The recent ATG and TopoAgent papers are especially important because they appeared in July 2026 and substantially occupy the state, dependency lens, node-memory, local-repair, and writeback story. Their effect sizes still require independent scrutiny because they are recent preprints.

### Recursive lens systems

- [ReCAP](https://proceedings.neurips.cc/paper_files/paper/2025/hash/88af3540325dd0b70617a9ab605f294d-Abstract-Conference.html) uses a recursive context tree and returns child results to refine a parent plan.
- [THREAD](https://aclanthology.org/2025.naacl-long.427/) defines child-context and return functions for isolated recursive threads.
- [STEP](https://arxiv.org/abs/2506.21030) conditions a subgoal on its parent, preceding same-level node, and observations.
- [Mechanic](https://arxiv.org/abs/2603.24465) isolates exact formal subgoals in clean local contexts and preserves verified work.

Local child prompting, preceding-sibling context, and child return are therefore not novel. A lens paper must identify which information is causally necessary under information- and token-matched controls.

### Formal and pre-LLM structural ancestors

- [TacticZero](https://arxiv.org/abs/2102.09756) uses a learned action factored into proof fringe, semantic goal, tactic, and arguments, with exact theorem-prover transitions.
- [Blackboard Architecture for Control](https://doi.org/10.1016/0004-3702(85)90063-3) provides shared persistent hypotheses, goals, an action agenda, a focus object, and writeback.
- [Additive AND/OR Graphs](https://hdl.handle.net/2318/103092) and [HTN Planning](https://www.cs.umd.edu/~nau/papers/erol1994htn.pdf) provide semantic problem reduction, selectable task nodes, and upward propagation.
- [DeepHOL](https://proceedings.mlr.press/v97/bansal19a.html) and [HyperTree Proof Search](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a8901c5e85fb8e1823bbf0f755053672-Abstract-Conference.html) provide learned control over persistent exact proof state.
- [Neural Program Repair by Jointly Learning to Localize and Repair](https://arxiv.org/abs/1904.01720) and [Hoppity](https://openreview.net/forum?id=SJeqs6EFvB) provide localization plus repair or graph-node plus edit factorization.

The `(verb,target)` abstraction is reasonable, but it is established machinery.

### Important distinction: operation graph versus semantic state graph

Tree of Thoughts and most search systems organize alternative continuations. RGoT learns an operation-plan graph. The project's proposed workspace instead stores semantic goals, facts, attempts, and dependencies that survive across actions.

That distinction remains useful for experimental design, but it does not restore broad novelty. GoT, DeAR, AGoT, KGoT, ReCAP, ATG, TopoAgent, and formal proof systems already maintain content-bearing structured state.

## Targeted Repair Prior Art

| Work | Occupied claim | Residual distinction |
|---|---|---|
| [Tyen et al.](https://arxiv.org/abs/2311.08516) | External error localization, preserved prefix, regeneration from target, frozen worker | No semantic forest or multi-verb exact one-call policy |
| [GLoRE](https://arxiv.org/abs/2402.10963) | Separate when, where, and how to refine | Trained verifier and refiners rather than black-box worker control |
| [Thought-ICS](https://arxiv.org/abs/2602.02416) | Semantic thought nodes, first-error localization, backtracking and resampling | Mainly linear trace; self or oracle localizer |
| [CausalFlow](https://arxiv.org/abs/2605.25338) | Structured dependencies, selected-step intervention, downstream reexecution, exact task outcome | Expensive post-hoc causal repair rather than amortized one-call pointer |
| [Calibration Is Not Control](https://arxiv.org/abs/2606.21399) | Same-prefix action-conditioned intervention advantage | Binary or smaller action interfaces; structured multi-target actions left open |
| [VERGE](https://arxiv.org/abs/2601.20055) | Formal claim decomposition, localized exact feedback, iterative repair | Large-model formalization pipeline |
| [Mechanic](https://arxiv.org/abs/2603.24465) | Verifier-derived isolated subgoal lens and preserved verified prefix | Formal theorem proving rather than induced natural-language state |

Inference:

> A paper cannot claim novelty for learning where to retry, preserving a correct prefix, using exact feedback to localize repair, or estimating same-prefix action benefit. The narrow residual is a controlled interaction among semantic targets, prompt-defined verbs, a fixed one-call budget, and black-box small workers.

## Claim-by-Claim Novelty Map

| Proposed claim | Verdict | Strongest pressure |
|---|---|---|
| External NN controls a frozen LLM | Occupied | RLoT; Prompting Policies; offline harness control |
| Tiny controller is sufficient | Occupied | RLoT; Neural CoT Search; offline harness control; RGoT |
| Teacher warm start followed by RL | Occupied | Agentic CoT Steering; PPDPP; Dialogue Action Tokens |
| Typed reasoning actions | Occupied | RLoT; ACTS; Neural CoT Search; TemplateRL |
| Factorized controller heads | Occupied | AdaReasoner and general parameterized-action RL |
| Persistent semantic tree or graph | Occupied | GoT; DeAR; AGoT; KGoT; ATG |
| Operation and target pointer | Occupied conceptually | GoT; TacticZero; graph program repair |
| Graph-derived local lens | Occupied | GoT Prompter; THREAD; STEP; ATG; TopoAgent; FoT |
| Child writeback and propagation | Occupied | DeAR; ReCAP; THREAD; ATG; TopoAgent |
| Learned tiny policy over an operation graph | Occupied | RGoT |
| Learned arbitrary pointer over natural-language workspace nodes | Narrow residual | No exact match found, but obvious from adjacent work |
| Exact symbolic verifier as sole reward | Meaningful protocol distinction | Most close work uses PRMs, judges, shaping, or formal environments |
| One additional call and matched compute | Meaningful evaluation distinction | Not an architecture novelty claim |
| Information-matched tree versus flat state for action uplift | Plausible residual | GRADE and state-representation studies raise the baseline |
| Same-state stability of prompted action realizations | Narrow residual and validity gate | Prompt-sensitivity work, TemplateRL, and Action Boundary Blindness lower standalone novelty |

## Ranked Fast Research Options

| Rank | Question | Novelty outlook | One-person feasibility | Useful negative result? |
|---|---|---|---|---|
| 1 | Does semantic state predict action value beyond menu, phase, topology, IDs, and order? | Medium to high | High after an outcome census | Yes, establishes an observability limit |
| 2 | Are prompted reasoning actions stable interventions? | Medium to low alone; stronger as preflight | Very high, 1 to 3 weeks | Useful if it changes action rankings or control regret |
| 3 | Which lens information is causally sufficient under matched tokens? | Medium | High | Yes, identifies when local context fails |
| 4 | Where should structure live: controller state or worker language? | Medium | High | Yes, separates state value from format tax |
| 5 | Is RGoT's fitted operation simulator control-sufficient? | Medium | Medium to high with released code | Yes, if misspecification changes policy value |
| 6 | Does value come from verb, target, or their interaction? | Medium to low | High after crossed action data | Yes, can invalidate the pointer head |
| 7 | One-call semantic-node retry | High novelty risk | Currently blocked by substrate quality | Only with replicated causal controls |
| 8 | Another generic controller architecture | Low | Technically feasible, scientifically weak | No |

## Required Preflight: Action Identifiability

Use 60 fresh exact-verifiable prefix states, two worker families, four meaningful verbs, and two independently written prompt realizations per verb:

```text
60 states * 2 workers * 4 verbs * 2 realizations = 960 branch calls
```

Any fixed continuation calls must be identical and counted in every arm.

Primary measure:

- exact terminal symbolic reward.

Secondary deterministic diagnostics:

- parser success;
- target-local state change;
- children created or removed;
- unrelated state mutation;
- final-answer propagation;
- rescue and damage; and
- tokens, latency, and cost.

Report verb, prompt-realization, worker, and interaction effects; best-action agreement; prompt-transfer and worker-transfer regret; best fixed value; bidirectional discordance; and a cross-fitted statewise oracle.

Proceed to a full 120-state, three-worker, two-domain study if at least one condition holds:

- prompt realizations differ by at least 5 exact-reward points;
- a worker by action interaction produces at least 5 points of transfer regret;
- at least two actions each uniquely win on at least 10 screen states; or
- distinct nominal actions collapse to the same structural transition with tight uncertainty.

Stop it as a standalone paper if actions are stable, transfer regret remains below 2 points in every cell, and the result merely reproduces known action-library effects.

## Primary Experiment: Does the Controller Read the State?

Run this only if the action-identifiability preflight establishes a meaningful action space and headroom.

Collect 80 to 120 common-prefix states with three to four legal actions and repeated outcomes. Compare:

- menu and phase only;
- topology plus menu;
- lean scalar features;
- full semantic state;
- shuffled semantic text;
- raw transcript or frozen text embedding;
- information-matched flat event features; and
- direct per-action value models.

Use grouped splits by problem family and construct matched-menu contrast sets whose best actions differ. Evaluate held-out exact policy value, action regret, and oracle-gap closure, not teacher imitation accuracy.

Drop the semantic-state claim if it fails to beat menu-only and information-matched flat state by at least a preregistered 3 to 5 points. [GRADE](https://arxiv.org/abs/2606.22741) makes lean topology, size, and edge-provenance baselines mandatory.

Add an almost-free shortcut audit:

- randomly rename node IDs;
- permute sibling order;
- remap legal actions consistently; and
- measure target equivariance and replayed utility.

A meaningful shortcut finding requires more than 10 points of degradation and recovery of at least half the loss through equivariant architecture or augmentation.

## Independent Experiment: Causal Lens Sufficiency

Hold target, worker, action, continuation, and token budget fixed. Compare:

- target only;
- ancestor path;
- dependency closure;
- preceding sibling;
- local subtree;
- token-matched full context; and
- token-matched distractor context.

Measure exact terminal success, local output change, final-answer propagation, rescue, damage, and cost. Manipulate irrelevant sibling count or conflicting sibling information one factor at a time.

Do not claim a lens effect unless dependency context gains at least 5 net points over both target-only and token-matched full context with a paired interval excluding zero in a preregistered high-interference regime. If the effect disappears under token matching, attribute it to context length rather than graph structure.

## Independent Experiment: Controller State Versus Worker Language

Prior work already shows that rigid schemas can tax reasoning, including [Let Me Speak Freely?](https://arxiv.org/abs/2408.02442), [The Format Tax](https://arxiv.org/abs/2604.03616), and [The Constraint Tax](https://arxiv.org/abs/2605.26128). Delayed formatting is not novel.

The residual is a two-consumer factorial experiment. Keep one authoritative state and identical facts, then cross:

| Controller observation | Worker input and output |
|---|---|
| flat or typed state | deterministic prose or typed input |
| flat or typed state | strict schema or free-form output plus deterministic parser |

First isolate worker input and output taxes without a learned controller. Then collect the action-outcome table under the winning worker interface. Finally compare flat and typed controller observations offline on identical outcomes.

Stop the typed-tree claim if it adds less than 3 points of held-out policy value over information-matched flat features. Stop the entire orchestration claim unless fresh matched-budget evaluation beats direct prompting and the best fixed harness.

## Future-Work Harvest and Pivot Value

Author-stated future directions converge in several crowded areas:

- AGoT proposes fine-tuning or RL for edge selection and complexity checks.
- FoT invites fully automatic dynamic graph schemes and joint prompt and hyperparameter optimization.
- RGoT proposes combining learned operation planning with KGoT-style graph state.
- ReCAP proposes executable graph memory and memory-aware routing.
- THREAD leaves better child-context and return functions open.
- Offline harness control proposes targeted support collection and online improvement.
- Calibration Is Not Control proposes richer structured interventions and online action-conditioned policies.
- CausalFlow proposes cheaper localization, lighter traces, and reduced reexecution.
- GRADE proposes observed dependency instrumentation, graph-guided repair, and pre-failure checks.

These directions support the plausibility of learned structured control, but they sharply reduce novelty. Implementing a paper's named future work is not automatically novel, especially when several adjacent papers name the same direction.

The most useful less-occupied openings are:

1. action-interface identifiability;
2. causal separation of action headroom from state observability;
3. information-matched placement of structure between controller and worker;
4. policy consequences of RGoT simulator misspecification; and
5. cross-worker counterfactual action rankings computed on identical states.

## Required Baselines

Every adaptive experiment should include:

- raw direct generation;
- compute-matched independent sampling;
- `KEEP` or submit-current;
- best fixed action and lens;
- random legal action;
- simple phase, topology, and scalar heuristics;
- worker self-selection;
- GPT-4.1-mini or stronger teacher selection;
- flat per-action value scoring;
- factored verb and target heads, only when crossed support exists; and
- a cross-fitted oracle estimated from development randomness disjoint from evaluation randomness.

Report exact accuracy, rescue, damage, best-fixed-to-oracle gap, oracle-gap closure, action regret, invalid actions, worker calls, generated tokens, latency, and cost per rescue.

## Hard Stop Rules

1. Stop controller training if a cross-fitted oracle is less than 5 points above the best fixed policy.
2. Stop adaptive routing if preferences are one-way rather than bidirectional.
3. Fix the action interface if prompt-realization effects dominate nominal-verb effects.
4. Drop the tree claim if information-matched flat state performs the same.
5. Drop the lens claim if token-matched full context performs the same.
6. Drop the target head if the target oracle adds less than 3 points beyond the verb oracle, or the pointer nearly always selects the current frontier.
7. Drop learned control if a fixed schedule, heuristic, self-router, or teacher router matches it.
8. Stop structured generation if it remains materially below raw generation.
9. Do not infer learnability from an in-sample maximum over noisy one-shot branches.
10. Do not proceed to long-horizon RL until the one-step action problem is identifiable, complementary, and predictable.

## Claims Not to Make

- First external neural policy to orchestrate a frozen LLM.
- First tiny controller for reasoning actions.
- First supervised-warm-start plus RL controller.
- First semantic reasoning tree or graph.
- First `(verb,target)` or operation-pointer action.
- First local graph-derived lens.
- First child writeback or parent propagation.
- First node-targeted retry or error-localized regeneration.
- First same-prefix action-advantage study.
- First use of exact verification for localized repair.
- First demonstration that structured output can hurt reasoning.
- First dynamic graph controller, unless the claim is narrowed far beyond the existing literature.

## Recommended Paper Strategy

The strongest fast paper package is:

> **Does a Frozen-LLM Controller Read the State? Semantic Observability Under Exact Action Outcomes**

It should include the 960-branch action-identifiability screen as a preregistered validity gate, then compare menu-only, topology, flat semantic, and explicit tree state on a cached action-outcome table. This gives the paper a concrete controller-state result even when prompt realizations are stable.

If the action screen instead shows large prompt-realization effects, action-rank reversals, or control regret across workers, it can support a separate or expanded paper:

> **Are Reasoning Actions Real? Identifiability of Prompted Interventions for Frozen Language Models**

If worker formatting is the dominant failure, the alternative is:

> **Who Needs Structure? Separating Controller State from Worker Language in Frozen-LLM Harnesses**

The original tiny-controller program can remain as the long-term method, but it should become conditional on these diagnostics rather than the next paper by default.

## Search Method and Limits

The review used primary papers from arXiv, ACL Anthology, PMLR, NeurIPS proceedings, OpenReview, and original classic sources. Search families included:

- frozen LLM plus external controller, policy, router, prompter, or harness;
- RL, offline RL, Q-learning, behavior cloning, and teacher steering;
- typed reasoning actions, action tokens, prompt policies, and operator selection;
- semantic task tree, thought graph, workspace, blackboard, node memory, and execution graph;
- target lens, local context, recursive thread, dependency closure, and context curation;
- error localization, backtracking, targeted retry, repair, and causal intervention;
- structured output, constrained decoding, format tax, and delayed packaging; and
- author-stated limitations and future work from the closest papers.

The search also followed backward references and recent July 2026 papers. Absence of an exact match is not proof of novelty. The July cluster is moving quickly, so the narrow residual claims should be rechecked immediately before submission.

## Final Assessment

The original research idea is technically coherent and connected to an active problem, but its broad novelty has been overtaken. The present experiments do not yet show that a teacher controller, semantic tree, or local lens beats a strong fixed prompt. Continuing directly to a larger teacher or NN policy would add cost without resolving the foundational uncertainty.

The project remains research-worthy if it pivots from proposing another controller to testing the assumptions that make controllers meaningful. Action identifiability is the fastest validity screen. Semantic observability is the strongest fast paper candidate, and causal lens sufficiency is the independent fallback. This sequence gives a one-person project clear stop rules, reuses exact verification, and preserves the long-term RL objective only where the empirical substrate justifies it.
