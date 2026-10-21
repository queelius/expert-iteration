# Source Ledger: External Control of Frozen LLMs

Cutoff: 2026-07-18

Status: completed evidence ledger at the cutoff. The final assessment is in
[`../prior-art-respose.md`](../prior-art-respose.md).

## Method Under Review

The original proposal is not merely "use an LLM to help another LLM." Its intended conjunction is:

1. a small external neural controller, eventually under roughly 5 million parameters;
2. a frozen, relatively small LLM that performs all language generation;
3. a structured, nested reasoning workspace rather than an unstructured transcript;
4. factored actions represented as `(verb, target)`, including a pointer over live nodes;
5. an action-specific lens that scopes the context shown to the frozen worker;
6. behavior-cloning warm start from a strong teacher followed by off-policy Q-learning;
7. exact terminal feedback from a symbolic verifier, with no learned reward model or LLM judge;
8. optional search as an additive accelerator; and
9. a matched-compute empirical requirement that the controller beat direct prompting and the best fixed policy.

The current candidate pivot is narrower: select one semantic node and one retry or repair action under an exact one-extra-call budget, then regenerate only through the corresponding lens.

## Headline Finding

The broad original claim is no longer defensible as novel. Several papers independently cover the major components, and at least two cover nearly the whole architecture:

- [RLoT: RL of Thoughts](https://arxiv.org/abs/2505.14140) uses a tiny learned RL navigator to select prompted reasoning operations for frozen small LLMs.
- [Agentic Chain-of-Thought Steering](https://arxiv.org/abs/2606.03965) trains a separate controller by supervised warm start and RL to steer a frozen reasoner step by step.
- [Learning to Control LLM Agent Harnesses with Offline Reinforcement Learning](https://arxiv.org/abs/2607.05458) learns a small external action policy over a frozen LLM harness from offline trajectories.

The July 5 harness paper is therefore important but not uniquely decisive. RLoT is the closest predecessor to the original tiny-controller reasoning architecture. Agentic Chain-of-Thought Steering is the closest predecessor to the proposed teacher-warm-start plus RL training pattern.

The current targeted-retry pivot is also crowded. Error localization, prefix-preserving retry, local refinement, thought-level backtracking, counterfactual action benefit, and exact-verifier repair all have direct predecessors. A publishable question, if any, must be defined more narrowly than "learn where to retry."

## A. Direct Predecessors to the Original Architecture

### A1. RLoT: RL of Thoughts

Primary source: [arXiv:2505.14140](https://arxiv.org/abs/2505.14140)

Reported design:

- Treats long reasoning as an MDP.
- Uses a Double Dueling DQN navigator with fewer than 3,000 parameters.
- Keeps the worker LLM frozen.
- Tests Qwen2.5 7B and 14B, Llama-family 8B, DeepSeek-family 7B, and GPT-4o-mini workers.
- State includes an LLM-generated self-evaluation across seven reasoning dimensions.
- Discrete actions invoke prompted reasoning blocks: reason one step, decompose, debate, refine, and terminate.
- Uses a learned process reward model in training.

Reported result and limitation:

- The paper reports gains as large as 13.4 percentage points and some cross-model transfer.
- Its manual audit found the self-evaluation accurate in 82 of 100 cases.
- The authors identify noisy self-evaluation as a source of wrong actions and report that refine can damage an already-correct step.
- Its experiments use a learned PRM, repeated inference, and hard-example-focused training, so the reported gain does not isolate policy selection under this project's one-call, exact-verifier protocol.
- Its fixed-sequence, repeated-block, supervised-navigator, and LLM-navigator ablations are mandatory analogues for this project.

Overlap assessment:

- This already establishes the central proposition that a very small RL policy can orchestrate a frozen small LLM through prompted reasoning verbs.
- It also includes `DECOMPOSE` and `REFINE`, so those verbs do not supply novelty.
- Remaining distinctions are a semantic-node pointer, an explicit nested forest, action-specific lenses, exact symbolic reward, and the proposed teacher/off-policy training recipe.
- Those distinctions form a narrower conjunction. They do not rescue a broad novelty claim.

Threat level: critical.

### A2. Agentic Chain-of-Thought Steering

Primary source: [arXiv:2606.03965](https://arxiv.org/abs/2606.03965)

Reported design:

- A separate controller steers a frozen reasoner during generation.
- State contains the reasoning trace, current step, history, and remaining token budget.
- An action contains a discrete strategy and a free-form steering message.
- Strategies include understand, plan, execute, explore, check, summarize, and conclude.
- A Qwen3 4B controller is first trained on synthetic expert steering trajectories, then optimized with GRPO.
- Frozen reasoners include 1.5B, 7B, and 8B models.

Author-stated limits:

- Evaluation is limited to reasoners up to 8B and two model families.
- The token budget is supplied by the user rather than adapted by the controller.
- The authors propose larger reasoners and adaptive budget selection.

Overlap assessment:

- This is a direct precedent for using a capable teacher to create controller demonstrations, then improving the controller by RL while leaving the worker frozen.
- Its controller is much larger than this project's planned tiny policy.
- It has no explicit semantic-node pointer, exact symbolic reward, or replayable reasoning forest.

Threat level: critical.

### A3. Learning to Control LLM Agent Harnesses with Offline RL

Primary source: [arXiv:2607.05458](https://arxiv.org/abs/2607.05458)

Reported design:

- Models a frozen-LLM harness as a finite-horizon MDP.
- Uses a one-hidden-layer MLP with 64 hidden units.
- Selects among seven actions: observe, retrieve, call tool, draft, check, revise, and submit.
- Trains from a fixed trajectory buffer by offline advantage-weighted regression.
- Uses terminal task-rubric reward and studies process behavior such as checking before submission.

Author-stated directions:

- Online improvement and targeted data collection beyond support in the fixed buffer.
- Evaluation in native benchmark protocols.
- Better-calibrated process measures.
- More graded revision and claim-level evidence checks.

Overlap assessment:

- Establishes compact external policy learning over typed harness actions for a frozen LLM.
- Its state is a compact trajectory summary, not a semantic reasoning forest.
- Its reward and process detectors do not satisfy this project's exact symbolic-reward invariant.
- Targeted collection is explicitly proposed future work, so that idea alone is not a novel fallback.

Threat level: critical for the broad harness-control claim.

### A4. CoRefine

Primary source: [arXiv:2602.08948](https://arxiv.org/abs/2602.08948)

Reported design:

- A roughly 211,000-parameter Conv1D controller reads confidence features from a frozen LLM's full reasoning trace.
- It chooses halt, rethink, or alternative.
- Training labels are derived from oracle comparisons among candidate trajectories.
- The method is tested with frozen 8B and 32B models.
- The paper discloses and corrects a development label-leak bug.

Author-stated directions:

- Better confidence calibration.
- Cheaper supervision.
- Evaluation beyond mathematics.
- Reducing suboptimal controller decisions.
- Evaluation is math-centered, some contest evaluations are small, and the method requires backbone confidence or log-probability access.

Overlap assessment:

- Directly precedes a tiny learned controller selecting retry modes for a frozen reasoner.
- It uses trace confidence rather than a structured forest and has no semantic target pointer.
- It makes generic halt/rethink/alternative control unavailable as a novelty claim.

Threat level: high.

### A5. HarnessBridge

Primary source: [arXiv:2606.12882](https://arxiv.org/abs/2606.12882)

Reported design:

- A learned bidirectional controller mediates observations and actions around a frozen coding agent.
- Observation projection selects pass, compress, or drop, creating a compact decision-relevant lens and active-state index.
- Action projection selects pass or reject and can provide grounded feedback.
- The controller is a roughly 0.8B Qwen3.5 model trained by supervised learning.

Author-stated limits:

- Evaluation is limited mainly to coding.
- Results are based on single runs.
- The authors propose extension to other tool domains.

Overlap assessment:

- Strongly crowds claims about learned lenses, active-state indexing, and action mediation around a frozen generator.
- It is not a tiny NN policy, does not use exact symbolic reward, and does not use an RL-first training scheme.

Threat level: high for state/lens novelty.

### A6. AdaReasoner

Primary source: [arXiv:2505.17312](https://arxiv.org/abs/2505.17312)

Reported design:

- Treats configuration selection for a frozen LLM as a contextual bandit.
- State is a query embedding.
- Factorized heads select an instruction or prompt, temperature, and reasoning steps.
- Uses external reward-model or GPT evaluation signals.

Author-stated directions and limits:

- Continuous action spaces and gradient-based prompt generation.
- Per-task few-shot training and RL overhead.
- Cross-worker transfer can deteriorate because reward landscapes are model-specific.

Overlap assessment:

- Establishes factorized external control heads and adaptive prompt configuration.
- It does not use sequential semantic targets or exact symbolic transition reward.
- Continuous or generative prompt actions are already an explicit future-work claim and are a weak fallback.

Threat level: high for factored-head and adaptive-prompt claims.

### A7. Prompting Policies for Multi-step Reasoning and Tool-Use in Black-box LLMs with Iterative Distillation of Experience

Primary source: [arXiv:2605.14443](https://arxiv.org/abs/2605.14443)

Reported design:

- Trains a lightweight prompter by RL to intervene on a larger frozen black-box worker.
- Uses an experience buffer containing scalar rewards and textual critiques.
- Reports results on BBEH and tau-bench.

Overlap assessment:

- Crowds the generic claim that an external neural prompting policy can improve a frozen black-box LLM.
- Its dense critique and reward machinery differs from exact symbolic verification.
- The remaining distinction must be sequential typed `(verb, target)` control and exact-verifier transitions, not frozen-worker prompt policy learning itself.

Threat level: high.

### A8. Multi-Head Latent Control

Primary source: [arXiv:2607.14277](https://arxiv.org/abs/2607.14277)

Reported design:

- Adds lightweight decision heads over hidden trajectories of frozen language or vision-language models.
- A capability head can route to a stronger model.
- A resolution head chooses clarification, tool use, abstention, or direct answer.

Author-stated directions:

- More robust and calibrated heads.
- More intervention types, including escalation, clarification, tools, abstention, safety, and routing.

Overlap assessment:

- Crowds generic multi-head intervention selection.
- Requires hidden-state access and uses evaluator-derived labels, unlike a black-box exact-verifier setting.
- It creates a useful comparator for a controlled state-representation study.

Threat level: medium to high.

### A9. DialogXpert

Primary source: [arXiv:2505.17795](https://arxiv.org/abs/2505.17795)

Reported design:

- A frozen LLM proposes dialogue actions.
- A compact external Q-network selects among candidates using frozen BERT embeddings and temporal-difference learning.

Author-stated limits and directions:

- Dynamic adjustment of the LLM action prior.
- Multimodal input.
- Subjective scalar reward design, critic inconsistency, state-design weaknesses, and rollout cost.

Overlap assessment:

- Direct precedent for a compact Q-network controlling actions proposed around a frozen LLM.
- The domain, state, and reward differ substantially, but external Q-learning is not novel.

Threat level: medium to high.

### A10. Earlier Policy and Prompt Controllers

These establish older components of the approach:

- [PPDPP](https://arxiv.org/abs/2311.00262): a tunable plug-in policy planner for proactive dialogue with frozen LLMs, trained with supervised learning and RL.
- [ART: Ask, Refine, Trust](https://arxiv.org/abs/2311.07961): smaller learned models decide when a larger pretrained or API LLM should refine and whether to retain the original or refined answer.
- [RLPrompt](https://arxiv.org/abs/2205.12548): a frozen LLM is controlled by a learned discrete prompt policy.
- [TEMPERA](https://arxiv.org/abs/2211.11890): test-time prompt editing is formulated as an RL problem.
- [Active Context Curation](https://arxiv.org/abs/2604.11462): an RL-trained curator sequentially selects context for a frozen executor.
- [HiLight](https://arxiv.org/abs/2604.22565): a lightweight actor highlights task-relevant spans for a frozen solver using task reward.
- [ReQueR](https://arxiv.org/abs/2604.25444): an RL-trained refiner rewrites a query into a decomposition for a frozen LLM environment.

Together these make broad claims about plug-in policies, learned prompt selection, context lenses, and decomposition unavailable.

### A11. Neural Chain-of-Thought Search

Primary source: [arXiv:2601.11340](https://arxiv.org/abs/2601.11340)

Reported design:

- Adds two linear heads, about 26,000 parameters for the 1.5B worker, over a frozen reasoner's hidden states.
- At double-newline decision points, the heads evaluate discourse operators such as `Therefore`, `Wait`, `Alternatively`, and `First`.
- A potential head distills the action distribution of a 7B or 32B teacher.
- A progress head predicts relative position in the target trace.
- One-step parallel KV-cache lookahead scores every candidate operator.
- Frozen 1.5B and 7B reasoning workers reportedly gain about 3.5 to 4 accuracy points while generating fewer tokens.

Author-stated directions:

- RL beyond teacher supervision.
- Dynamic entropy-triggered interventions.
- Global MCTS.
- Multilingual and creative operator sets.

Overlap assessment:

- This is a direct precedent for extremely small external heads, a frozen small worker, teacher distillation, discrete reasoning operators, and optional search.
- It requires internal hidden-state and KV-cache access, uses local discourse-token actions, and has no semantic target graph or symbolic-verifier reward.
- Teacher distillation, reasoning-operator actions, and adding MCTS are not available as broad novelty claims.

Threat level: critical.

### A12. Prompt Optimization for LLM Code Generation via RL

Primary source: [arXiv:2605.19102](https://arxiv.org/abs/2605.19102)

Reported design:

- A PPO policy reads a MiniLM prompt embedding.
- It chooses direct generation, lexical mutation, or semantic rewriting for as many as ten iterations.
- Code generators remain frozen.
- Unit-test execution supplies shaped feedback.

Reported caveats and directions:

- The controller parameter count is not disclosed.
- Semantic rewriting delegates to another pretrained model.
- State omits generated code, failure details, reward history, and prior actions.
- Its strict Pass@1 label counts success if any of as many as ten attempts passes, so it is not conventional single-generation Pass@1.
- Proposed work includes traceback-aware state, transfer, curriculum, larger benchmarks, and IDE integration.

Overlap assessment:

- Establishes a neural RL policy over typed prompt operations around a frozen generator with executable verification.
- It lacks semantic targets, a persistent workspace, and symbolic-terminal-only reward.

Threat level: high on the verifier-grounded prompt-control axis.

### A13. Dialogue Action Tokens

Primary source: [arXiv:2406.11978](https://arxiv.org/abs/2406.11978)

Reported design:

- A small MLP reads the final hidden state of a frozen Llama2-7B or Llama3-8B worker.
- It emits a continuous action vector that maps to two prefix tokens steering the next dialogue turn.
- Training starts from self-cloned behavior and then uses offline TD3+BC actor-critic learning with replay.

Overlap assessment:

- Strongly precedes the proposed behavior initialization plus off-policy value-learning recipe for controlling a frozen 7B or 8B model.
- It uses continuous hidden-state action tokens, subjective GPT-evaluated dialogue rewards, and no typed verb, target pointer, workspace, lens, or exact verifier.

Threat level: high for the learning recipe.

### A14. Matryoshka

Primary source: [arXiv:2410.20749](https://arxiv.org/abs/2410.20749)

Reported design:

- Treats a large black-box LLM as an environment.
- A lightweight white-box LLM controller provides intermediate prompt guidance.
- The controller decomposes complex tasks into intermediate outputs and is trained to steer the black-box generator toward preferences over multiple turns.
- Evaluates reasoning, planning, and personalization tasks.

Overlap assessment:

- An early and explicit controller-generator framework for driving a frozen or inaccessible black-box LLM through intermediate prompts.
- Its controller is itself a language model rather than a tiny factored NN, and it does not supply the exact structured-state and symbolic-verifier conjunction.

Threat level: high for the broad controller-generator framing.

### A15. Embedding-Aligned Language Models

Primary source: [arXiv:2406.00024](https://arxiv.org/abs/2406.00024)

Reported design:

- Treats a pretrained LLM as an environment.
- An RL agent iteratively steers generation toward a target region of an embedding space.
- Studies state-dependent action-set design.

Overlap assessment:

- Establishes the general formulation of a pretrained LLM as an environment controlled by an external RL policy.
- The objective is latent embedding alignment rather than exact task reasoning, and the state and actions differ substantially.

Threat level: medium as a conceptual ancestor.

### A16. Multi-agent Architecture Search via Agentic Supernet

Primary source: [arXiv:2502.04180](https://arxiv.org/abs/2502.04180)

Reported design:

- A feed-forward controller reads a query embedding and previously selected operator profiles.
- It builds a query-specific DAG from typed operators including chain of thought, debate, self-consistency, self-refine, ensemble, testing, ReAct, and early exit.
- A fixed LLM executes the selected graph.
- Training optimizes outcome utility minus token cost by Monte Carlo gradient estimation.

Overlap assessment:

- Directly precedes a learned neural controller constructing a typed reasoning graph around a fixed LLM.
- It plans before seeing intermediate results, points to operator types or layers rather than live problem nodes, and does not use verifier-only reward, behavior cloning, off-policy Q-learning, or a node lens.
- Small task-specific training splits and separately evolved prompts make attribution difficult.

Threat level: critical for typed-operation graph claims.

### A17. Planning Like Human

Primary source: [ACL 2024](https://aclanthology.org/2024.acl-long.262/)

Reported design:

- A RoBERTa policy model has policy and Q heads over dialogue history.
- Offline RL weights the policy loss using cumulative Q values.
- MCTS-guided self-play and actor-critic temporal-difference learning follow.
- A confidence margin chooses a fast policy or search.
- GPT-3.5 remains the language generator.

Overlap assessment:

- Strong precedent for policy and Q heads, offline plus online value learning, and optional search around a fixed LLM generator.
- It uses a large controller, dialogue strategies, LLM critics and simulators, and no symbolic verifier, node pointer, or workspace lens.

Threat level: high for the proposed learning and optional-search recipe.

### A18. PYTHALAB-MERA

Primary source: [arXiv:2605.08468](https://arxiv.org/abs/2605.08468)

Reported design:

- A 16-feature LinUCB controller around frozen local Qwen3-4B selects among eight retrieval actions.
- State includes the task, current file, history, previous validation report, episodic memory, syntax structure, skills, failures, progress, and duration.
- A fail-fast validation stack checks syntax, interface, imports, runtime, and behavior.

Overlap assessment:

- Direct precedent for an external controller over rich structured state, decomposed context actions, a frozen small worker, lens-like retrieval, and validation-based acceptance.
- It is a linear bandit with shaped progress and cost reward, not a neural factored Q policy with semantic problem-node actions.
- Its main evaluation is very small and lacks several component ablations, so it is a conceptual collision rather than decisive evidence of effectiveness.

Threat level: high for structured-state and validation-controlled retrieval claims.

### A19. OLIVIA

Primary source: [arXiv:2605.11169](https://arxiv.org/abs/2605.11169)

Reported design:

- A LinUCB layer reads the final hidden state of frozen 4B or 7B ReAct agents.
- It selects candidate tools, starts from action-name embeddings, and updates online from immediate tool-membership feedback.

Overlap assessment:

- Establishes an explicit online action layer over a frozen small LLM.
- It is a linear bandit requiring internal hidden states and privileged tool labels, and it evaluates tool-selection F1 rather than final task success.

Threat level: medium.

### A20. SEAM

Primary source: [arXiv:2602.02556](https://arxiv.org/abs/2602.02556)

Reported design:

- A 0.6B model produces structured problem analysis, relevant experience, and a plan for a frozen 4B or 8B executor.
- GRPO uses executor rollouts, binary task success, and structural-completeness signals.

Overlap assessment:

- A close one-shot structured-lens and small-controller precedent.
- The controller is still a language model, is not an iterative factored policy, and uses reward components outside the symbolic-terminal-only invariant.

Threat level: medium to high.

### A21. Memento / AgentFly

Primary source: [arXiv:2508.16153](https://arxiv.org/abs/2508.16153)

Reported design:

- A memory-augmented MDP surrounds a frozen planner or executor.
- A neural case selector uses Q values and online soft Q-learning with replay.
- Retrieved memories are scoped to subtasks and tools.

Overlap assessment:

- Strong precedent for structured external memory, scoped context, and learned Q control around a frozen LLM.
- It controls retrieval rather than typed reasoning-node operations, and its deployed parametric selector is less clearly a full sequential TD controller.

Threat level: medium to high.

### A22. Learning Evidence Highlighting for Frozen LLMs

Primary source: [arXiv:2604.22565](https://arxiv.org/abs/2604.22565)

Reported design:

- A 0.6B to 4B actor scores input tokens and samples a factorized highlight mask under a budget.
- Non-destructive tags emphasize selected evidence for frozen 14B or black-box solvers.
- Grouped policy-gradient training uses task metrics without evidence labels.

Author-stated directions:

- Cache-aware multi-turn serving.
- Rewrite-versus-highlight comparisons.
- Study when emphasis is better than deletion.

Overlap assessment:

- Close precedent for learned target selection and a non-destructive lens for a frozen solver.
- It is a one-shot token mask with a relatively large actor, not a sequential semantic-node controller.

Threat level: high for generic lens-learning claims.

### A23. Plug-and-Play Policy Planner

Primary source: [arXiv:2311.00262](https://arxiv.org/abs/2311.00262)

Reported design:

- A RoBERTa-large strategy planner controls frozen GPT-3.5, Vicuna, or Llama2 dialogue generation.
- It starts with supervised learning on annotated strategy actions and then uses online REINFORCE from simulated goal rewards.

Overlap assessment:

- Historical direct precedent for an external controller plus frozen LLM plus supervised warm start followed by RL.
- It uses a 355M controller, dialogue actions, and subjective critic reward rather than a tiny pointer policy and exact verifier.

Threat level: high for the training pattern.

### A24. Reinforced Graph of Thoughts

Primary source: [arXiv:2605.22195](https://arxiv.org/abs/2605.22195)

Reported design:

- Tiny PPO actor and critic networks, each a two-layer MLP with 64 units per layer, construct and traverse a graph of operations for a fixed LLM.
- A task supplies typed prompt or executable operations in a constrained layered DAG.
- At each step, the controller chooses stop, backtrack, or append an operation.
- Observation includes graph depth and breadth, global and local task complexity, layerwise operation history, previous actions, and previous score.
- Training uses a Bernoulli simulator fit from repeated LLM operation trials rather than direct text-state transitions.
- Reward includes intermediate operation scores and terminal ground-truth checks.

Author-stated directions:

- Automated task-to-agent training pipelines.
- Libraries of domain-specific agents.
- RAG pre-processing and post-processing.
- Combination with Knowledge Graph of Thoughts.

Overlap assessment:

- This is one of the closest architecture-level predecessors: tiny external NN, explicit structured reasoning graph, typed operations, stop and backtrack, RL, fixed LLM, and exact ground-truth task evaluation.
- It selects the next layer operation rather than a factored verb and pointer to an arbitrary live content node.
- Its graph is an operation-plan DAG rather than an evolving semantic workspace with node status and writeback.
- It trains on a complexity-conditioned Bernoulli abstraction and shaped scores, not real worker transitions and verifier-only off-policy learning.

Threat level: critical.

Implication:

- The project cannot claim that a tiny RL network learning to construct a graph of LLM operations is new.
- The remaining empirical question is whether content-aware live-node targeting, scoped lenses, real worker transitions, and exact verifier-only learning outperform RGoT-style complexity-conditioned graph construction.

### A25. Adaptive Graph of Thoughts

Primary source: [arXiv:2502.05078](https://arxiv.org/abs/2502.05078)

Reported design:

- Recursively decomposes a query into a directed acyclic graph of interdependent semantic subproblems.
- Every node has a unique hierarchical address called its heritage and stores a thought, layer strategy, answer, and optional nested graph.
- An LLM complexity check decides whether a node is answered directly or expanded into a nested graph.
- Later nodes consume completed graph state, including the final answers produced by nested child graphs.
- The graph is bounded by depth, layer, and per-layer node limits and can terminate early when it recognizes a sufficient answer.

Author-stated directions:

- Optimize depth, layer, and node limits.
- Delegate thought, strategy, and edge generation to specialized agents or models.
- Use fine-tuning or reinforcement learning to bias edge selection and complexity checks.
- Reduce hallucination and positive-feedback loops caused by repeatedly consuming model-generated graph content.

Overlap assessment:

- This directly occupies dynamic semantic decomposition, hierarchical node identity, selective expansion, graph-conditioned evaluation, and child-to-downstream information flow.
- Its explicit proposal to apply RL to edge selection and complexity checks anticipates a learned structural controller.
- It uses LLM agents for all structural decisions, not a tiny external policy, and does not expose the general live-node `(verb, target)` action interface or exact-verifier-only reward.

Threat level: critical for the semantic-tree and adaptive-focus claims.

### A26. Tree-of-Traversals

Primary source: [ACL 2024 paper](https://aclanthology.org/2024.acl-long.665/)

Reported design:

- Maintains a local knowledge subgraph and repeatedly expands it until enough evidence exists to answer.
- Uses an action-state machine with `THINK`, `ANSWER`, `EXPAND_KG`, `SELECT_ENTITIES`, and `SELECT_RELATION` actions.
- State-specific prompt templates expose relevant choices from the current graph in a compact YAML serialization.
- Entity and relation selections update the graph through deterministic knowledge-graph interfaces.
- Best-first tree search selects among action and thought states, assigns LLM-generated node values, and backtracks.

Overlap assessment:

- This is a direct precedent for typed action plus graph target, graph-derived local context, deterministic writeback, and backtracking around a frozen black-box LLM.
- The graph being edited is external knowledge, while the search tree stores alternative action histories. It is not a learned pointer over semantic reasoning subproblems.
- Its value function is an LLM evaluation, not exact symbolic reward.

Threat level: high for generic action-state-machine, lens, and writeback claims.

### A27. TacticZero

Primary source: [arXiv:2102.09756](https://arxiv.org/abs/2102.09756), published at ICLR 2021.

Reported design:

- Formulates interactive theorem proving as an MDP whose state retains multiple proof fringes, each containing a set of open semantic goals sufficient to complete one partial proof.
- A transition applies a tactic to one selected goal, replaces that goal with zero or more formal subgoals, and appends the resulting fringe while retaining alternative derivations for backtracking.
- The action is explicitly factorized as `(fringe, goal, tactic, arguments)`.
- Separate learned modules select a fringe, a goal inside it, a tactic, and tactic arguments.
- The HOL4 proof assistant supplies machine-checked transitions, failures, completion, and exact proof validity.

Overlap assessment:

- This is a strong pre-LLM ancestor of the project's structured forest, live target pointer, typed operation, exact transition, and learned backtracking policy.
- The semantic nodes and operations come from a formal proof language, not induced natural-language decomposition.
- It controls a theorem prover rather than prompting a frozen LLM and uses step-level environment feedback in addition to terminal completion.

Threat level: critical for claims that the factored forest-action architecture itself is novel.

### A28. VISTA: Typed Addressable Working Memory

Primary source: [arXiv:2606.30005](https://arxiv.org/abs/2606.30005)

Reported design:

- Represents an agent's working context as typed, addressable blocks with stable handles.
- A dashboard exposes each block's token cost, recency, access history, storage status, and remaining context budget.
- The agent selects context-management actions over block targets, including archive, recover, and delete, alongside ordinary environment actions.
- Archived payloads remain exactly recoverable outside the active prompt.
- The same training-free interface is evaluated across several frontier backbones and context-pressure regimes.

Overlap assessment:

- This directly occupies typed addressable state, target-specific actions, target-derived context, and non-destructive external storage.
- It treats model self-management as an elicitation problem and uses the worker LLM itself as the policy, not a tiny external controller.
- Its blocks are context artifacts rather than a goal and subgoal reasoning tree.

Threat level: high for generic addressable-state and context-action claims.

### A29. Harness-1

Primary source: [arXiv:2606.02373](https://arxiv.org/abs/2606.02373)

Reported design:

- Externalizes search state into a candidate pool, curated set, full-text store, evidence graph, verification records, search history, and context-budget renderer.
- The policy acts on a rendered working memory rather than reconstructing state from a transcript.
- Structured actions search, read, curate, review, verify, and submit; several actions edit or target persistent artifacts.
- The paper reports component ablations for importance tags, compression, auto-seeding, evidence-graph visibility, verification, review, and deduplication.

Overlap assessment:

- Strong recent evidence that structured external state, graph summaries, and artifact-editing actions can be a learned policy substrate.
- The policy is a 20B model trained with SFT and on-policy RL, not a small external controller around a separately frozen worker.
- The evidence graph is mechanically derived and specialized to retrieval.

Threat level: high for broad structured-state-as-controller-substrate claims.

### A30. Progress-Aware Belief Update

Primary source: [arXiv:2602.09138](https://arxiv.org/abs/2602.09138)

Reported design:

- Replaces full interaction history with a compact belief representation that predicts relative task progress and selectively retains actions and observations.
- Trains with progress-derived filtering and action augmentation on offline trajectories.
- Reports ablations showing that useful history components vary by environment and action-space structure.
- The authors explicitly propose hierarchical representations and learned latent-state models as future work.

Overlap assessment:

- Reinforces that controller state design and selective observation are active research questions, not implementation details.
- It modifies an LLM agent policy and uses synthesized progress supervision rather than an external tiny controller with terminal symbolic reward.

Threat level: medium for a generic state-representation paper, lower for the proposed semantic observability audit.

### A31. Atomic Task Graph

Primary source: [arXiv:2607.01942](https://arxiv.org/abs/2607.01942), submitted 2026-07-02.

Reported design:

- Recursively replaces a coarse semantic task node with an interface-preserving subgraph until every node is an atomic tool-use unit.
- Allows each refinement call to access only history directly relevant to the current node, making context narrower with depth.
- Records the full sequence of graph refinements and each executed node's input, output, status, and error.
- Executes ready nodes by dependency order and can run independent branches in parallel.
- Localizes a failure to an atomic node or lowest common historical ancestor, constructs the smallest affected repair subgraph, freezes validated regions, and reintegrates the repair while preserving the external interface.
- Reports large gains over ReAct, Reflexion, ToT, Plan-over-Graph, and CAMEL on ALFWorld, WebShop, and ScienceWorld with 7B to 8B backbones.

Author-stated limits:

- Dependence on the worker's decomposition ability.
- Difficult localization under noisy observations and long-range dependencies.
- Text-only evaluation and extra overhead on simple tasks.

Overlap assessment:

- This is a direct collision with the project's semantic graph, target-derived lens, node-local state, writeback, failure localization, local repair, and preservation of validated work.
- It is a fixed training-free compilation and scheduling framework, not a learned policy over arbitrary `(verb,target)` pairs.
- Its reward is benchmark-specific and not restricted to this project's terminal symbolic verifier.
- It is a very recent unreviewed preprint, so its reported effect sizes require replication before being treated as settled evidence.

Threat level: critical for the broad state, lens, and localized-repair story.

### A32. TopoAgent

Primary source: [arXiv:2607.14658](https://arxiv.org/abs/2607.14658), submitted 2026-07-16.

Reported design:

- Decomposes multimodal scientific questions into a DAG of atomic subproblems.
- Stores completed-node states in dynamic memory and defines the context for node `A_i` as the union of its completed prerequisite-node states.
- Writes tool results into the selected node's memory and verifies whether the node has reached a stop state.
- If a node cannot finish within a step limit, an LLM fission function replaces it with finer subnodes, executes them locally, and exposes their aggregate under the parent's namespace.
- Reports experiments across six multimodal backbones on math, physics, and chemistry benchmarks.

Overlap assessment:

- This independently implements a dependency-defined lens, node-local memory and writeback, targeted failure detection, and dynamic node decomposition.
- Node order is topological and node fission is a fixed failure rule rather than a learned arbitrary pointer policy.
- The system uses model and tool self-verification rather than terminal symbolic reward only.
- It is a two-day-old unreviewed preprint at the search cutoff.

Threat level: critical for dependency-lens, writeback, and dynamic-decomposition novelty.

### A33. Framework of Thoughts

Primary source: [arXiv:2602.16512](https://arxiv.org/abs/2602.16512), version 2 dated 2026-07-04.

Reported design:

- Distinguishes an execution graph of operations and information-flow edges from a reasoning graph of generated thoughts and their dependencies.
- Retains the full execution history while allowing operations to add or remove future operations and connections during execution.
- Lets an operation inspect only the subgraph induced by itself, its ancestors, and its descendants.
- Protects completed ancestors and shared descendants from unsafe rewrites while allowing an operation to rewrite its exclusive-descendant region.
- Adds scheduling, parallel execution, process and persistent caches, and prompt or hyperparameter optimization.
- Reimplements Tree of Thoughts, Graph of Thoughts, and ProbTree, reporting runtime and cost improvements. It does not evaluate a newly learned automatic graph-editing policy.

Author-stated future work:

- Fully automatic prompting schemes that demonstrate when dynamic graphs help across more problem types.
- Parallel optimization of multiple prompts, joint prompt and hyperparameter optimization, friendlier graph abstractions, and less restrictive parallel graph-modification rules.

Overlap assessment:

- This occupies a general dynamic, addressable execution-state substrate with scoped visibility and graph-local mutation.
- The project's node lens resembles FoT's induced ancestor and descendant view, although FoT exposes an operation neighborhood rather than a learned target-specific minimal context.
- FoT supplies framework machinery and a fixed scheduler, not a compact external policy trained from terminal symbolic outcomes to choose `(verb,target)`.
- Its future-work invitation makes a generic "learn dynamic graph control" claim especially vulnerable to being viewed as an anticipated application.

Threat level: critical for broad dynamic-state and scoped-graph-view novelty; medium for a causal state-observability study.

### A34. GRADE: Graph Representation of LLM Agent Dependency and Execution

Primary source: [arXiv:2606.22741](https://arxiv.org/abs/2606.22741), submitted 2026-06-22.

Reported design and results:

- Represents an agent run as one typed graph with execution edges for run order and dependency edges for what each step read, wrote, or relied on.
- Grades dependency edges by whether they are observed, declared, or inferred instead of treating all edges as equally reliable.
- Tests six agent-run corpora from tool use, coding, databases, and web navigation.
- Reports that size-normalized dependency features remain above chance under leave-one-corpus-out failure prediction, while run size falls below chance on two held-out corpora.
- Reports execution-graph features that localize faulting steps on one multi-agent failure corpus.
- Finds that generic graph networks can overfit run size or ignore edge provenance, motivating source-aware structural features.

Author-stated directions and limits:

- Instrument tools to make read and write dependencies observed rather than inferred.
- Build data containing both observed dependencies and step-level fault labels.
- Validate graph-derived pruning offline, repair or re-plan around fragile structure, check stale reliance before failure, and study fleet-scale structural anomalies.
- Current transfer coverage is dominated by coding and database tasks, and the dependency-localization experiment is missing for lack of jointly annotated data.

Overlap assessment:

- GRADE makes dependency-aware graph state and graph-based failure prediction or localization occupied claims.
- It studies passive diagnosis of completed agent runs, not action-conditioned value or control of a frozen worker.
- Its warning that richer graph models can overfit is directly relevant: a planned GNN or tree encoder must beat lean topology, size, menu, and provenance-aware feature baselines out of distribution.
- The residual observability question must predict which intervention helps, including possible damage, rather than merely whether the current run is wrong.

Threat level: high for generic graph-state observability; medium for exact action-uplift prediction.

### A35. Recursive Context Trees and Return Interfaces

Primary sources:

- [ReCAP, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/88af3540325dd0b70617a9ab605f294d-Abstract-Conference.html)
- [THREAD, NAACL 2025](https://aclanthology.org/2025.naacl-long.427/)
- [STEP, arXiv:2506.21030](https://arxiv.org/abs/2506.21030)

Reported designs:

- ReCAP maintains recursive plan and context state, executes a current subtask in a scoped view, and returns results to refine the parent plan.
- THREAD formalizes a child-context function and a return function for isolated recursive threads.
- STEP traverses a subgoal tree and conditions a node on its parent, the preceding same-level node, and current observations.

Overlap assessment:

- These papers occupy recursive child isolation, target-dependent context, preceding-sibling context, and child-to-parent return.
- Their controllers are fixed procedures or LLM decisions, not tiny learned policies trained only from terminal exact outcomes.
- A lens contribution must therefore be a controlled dependency-sufficiency result, not the existence of scoped recursive prompting.

Threat level: high for lens and writeback novelty.

### A36. Older Structured-Control Ancestors

Primary sources:

- [Blackboard Architecture for Control](https://doi.org/10.1016/0004-3702(85)90063-3)
- [Additive AND/OR Graphs](https://hdl.handle.net/2318/103092)
- [HTN Planning](https://www.cs.umd.edu/~nau/papers/erol1994htn.pdf)
- [DeepHOL](https://proceedings.mlr.press/v97/bansal19a.html)
- [HyperTree Proof Search](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a8901c5e85fb8e1823bbf0f755053672-Abstract-Conference.html)
- [Neural Program Repair by Jointly Learning to Localize and Repair](https://arxiv.org/abs/1904.01720)
- [Hoppity](https://openreview.net/forum?id=SJeqs6EFvB)

Overlap assessment:

- Blackboard systems provide the conceptual pattern of shared persistent hypotheses and goals, an action agenda, a selected focus object, and writeback.
- AND/OR and HTN methods provide semantic problem reduction, selectable task nodes, and upward completion or value propagation.
- Proof search supplies exact formal state transitions, target goals, learned tactics, backtracking, and exact validation.
- Program repair supplies explicit localization plus edit factorization over graph nodes.

Threat level: critical for claims that an addressable workspace, `(operation,target)`, or local structured rewrite is a new architecture pattern.

## B. Direct Predecessors to Targeted Retry and Repair

### B1. LLMs Cannot Find Reasoning Errors, but Can Correct Them Given the Error Location

Primary source: [arXiv:2311.08516](https://arxiv.org/abs/2311.08516)

Reported design and result:

- Separates error localization from correction.
- Given an error target, preserves the prefix and regenerates the erroneous step and suffix with the same frozen generator.
- Compares oracle and random targets.
- Trains a smaller external classifier on out-of-domain data to predict the first error, explicitly supporting frozen or API LLM use.
- Simulations suggest target accuracy around 60 to 70 percent is needed for net benefit; the learned classifier did not reach that threshold.

Author-stated directions and limits:

- More training data and more sophisticated or iterative classifiers.
- Artificial error data, class imbalance, and limited evaluation constrain the result.

Overlap assessment:

- This is an exact threat to "learn a node to retry while keeping the worker frozen."
- Semantic tree nodes, multiple action verbs, exact one-call control, and symbolic verification remain distinctions.

Threat level: critical.

### B2. GLoRE

Primary source: [arXiv:2402.10963](https://arxiv.org/abs/2402.10963)

Reported design:

- Explicitly separates when, where, and how to refine.
- A synthetic stepwise outcome reward model scores prefixes and locates a likely first error.
- Global and local refiners are trained, with the local refiner conditioned on error location.
- Candidate outputs are combined and reranked.

Author-stated directions:

- Better error critiques.
- Verifier reasoning chains and tool assistance.

Overlap assessment:

- Occupies error-targeted local repair and learned refinement selection.
- It fine-tunes verifier and refiner models, reranks multiple candidates, and does not retain the exact frozen-worker, one-call, symbolic-reward constraint.

Threat level: critical.

### B3. StepCo

Primary source: [arXiv:2410.12934](https://arxiv.org/abs/2410.12934)

Reported design:

- A process-supervised verifier finds the first low-quality reasoning step.
- The method preserves the preceding prefix and prompts the LLM to revise the suffix.
- Verification and revision repeat iteratively.

Author-stated limits:

- English-only evaluation.
- Focus on numeric or entity-style outputs.
- Future work includes other languages and open-ended outputs.

Overlap assessment:

- Directly occupies verifier-guided prefix-preserving retry.
- It uses a learned verifier rather than an exact terminal verifier and is not a one-call learned target policy.

Threat level: high.

### B4. Structure Enables Effective Self-Localization of Errors in LLMs

Primary source: [arXiv:2602.02416](https://arxiv.org/abs/2602.02416)

Reported design:

- Represents reasoning as semantic thoughts in a tree or thought-level MDP.
- Localizes the first bad thought, backtracks to the last correct prefix, and resamples.
- Evaluates both oracle and autonomous verification across models from roughly 3B to 120B.
- Reports large oracle-localization gains.

Overlap assessment:

- Strongly crowds structured-thought state, semantic target selection, and backtracking.
- A learned tiny external controller, multiple typed interventions, an exact one-call budget, and symbolic terminal labels remain distinctions.

Threat level: critical.

### B5. CausalFlow

Primary source: [arXiv:2605.25338](https://arxiv.org/abs/2605.25338)

Reported design:

- Uses structured execution traces and step-level counterfactual interventions.
- Re-executes downstream steps after an intervention.
- Uses exact or task-specific verification to measure a causal responsibility score.
- Searches for minimal validated repairs.
- Reports converting 42.7 percent of failed executions across four domains.

Reported limits:

- Repair proposal quality and re-execution cost are important bottlenecks.
- Structured logging can reduce initial task accuracy.
- Dependency annotations may be noisy.
- Structured logging reportedly reduced initial GSM8K accuracy from 88.1 percent to 75.0 percent in the evaluated setting.
- The paper proposes approximating its Causal Responsibility Score without repeated re-execution.

Overlap assessment:

- Occupies counterfactual localization, structured traces, exact-verifier repair, and minimal intervention.
- It primarily uses frontier agents and expensive re-execution, not a tiny learned controller over a small frozen worker.

Threat level: critical for causal-localization claims.

### B6. Calibration Is Not Control

Primary source: [arXiv:2606.21399](https://arxiv.org/abs/2606.21399)

Reported design:

- Defines the benefit of an intervention relative to continuing from the same prefix.
- Creates common-prefix counterfactual branches and executes alternative actions from the identical state.
- Trains lightweight action-conditioned controllers for frozen 7B or 8B agents.
- The reported lightweight controller is a random forest trained from terminal action outcomes.
- Shows that scalar correctness calibration does not necessarily minimize action regret.

Author-stated limits and directions:

- Only a small action set: continue, one intervention, or quit.
- A simple prefix-only controller and offline, single-rollout setting.
- Future work proposes richer controllers, multiple structured interventions, and online action-conditioned policies.

Overlap assessment:

- This directly occupies verifier-measured intervention benefit, common-prefix action comparisons, and lightweight control.
- The project must not claim action uplift, counterfactual branching, or action-conditioned value as new.
- A semantic target pointer and richer structured state are a possible extension, but one explicitly invited by the paper.

Threat level: critical.

### B7. VERGE

Primary source: [arXiv:2601.20055](https://arxiv.org/abs/2601.20055)

Reported design:

- Decomposes answers into atomic claims and formalizes suitable claims for SMT checking.
- Uses minimal correction subsets to locate claims implicated in verification failure.
- Feeds localized feedback into iterative refinement.

Reported limits:

- High latency.
- Formalization is difficult for smaller models.
- Decidable logic cannot express every claim.

Overlap assessment:

- Occupies exact or formal localized repair.
- It uses a very large model and a claim-level formalization pipeline, not a small frozen worker with a learned node policy.

Threat level: high.

### B8. Mechanic

Primary source: [arXiv:2603.24465](https://arxiv.org/abs/2603.24465)

Reported design:

- Uses Lean placeholders to isolate unresolved proof subgoals.
- Presents a clean, self-contained subproblem context.
- Solves subgoals independently while preserving the verified proof prefix.

Overlap assessment:

- Occupies the local-lens intuition in formal theorem proving: isolate a target, preserve verified context, and retry locally.
- It does not learn a generic external policy.

Threat level: medium to high for the lens rationale.

### B9. Other Structured Repair and Decomposition Work

- [DeAR](https://proceedings.neurips.cc/paper_files/paper/2024/file/01025a4e79355bb37a10ba39605944b5-Paper-Conference.pdf) builds a semantic question-decomposition tree, solves and self-checks node rationales, then updates parents from children. It uses fixed thresholds and multiple calls rather than a learned budgeted node selector. Its stated directions include lower latency, annotation-free generalization, more STEM and programming tasks, and RL for adaptive self-thinking.
- [LM2](https://arxiv.org/abs/2404.02255) uses a frozen GPT-3.5 solver with trained decomposer and verifier models. It generates subquestions sequentially, accepts or rejects subanswers, and constructs solver context from approved results. It directly precedes learned orchestration of a frozen solver through decomposition, but uses dense mistake labels and multiple calls.
- [Q*](https://arxiv.org/abs/2406.14283) trains a plug-in Q model from terminal rewards, rollouts, offline fitted Q iteration, or stronger-model completions, then guides a frozen LLM through A* search. It chooses prospective next-line actions and uses multiple candidate trajectories rather than one post-hoc semantic-node intervention.
- [Rewarding Progress](https://arxiv.org/abs/2410.08146) defines a process advantage verifier as the change in future success probability induced by a reasoning step. It learns from terminal-correctness rollouts and guides beam search or online RL. This establishes terminal-outcome-trained step advantage, although not node-specific retry value.
- [Small Language Models Need Strong Verifiers](https://arxiv.org/abs/2404.17140) shows that small models' self-correction depends strongly on verifier quality.
- [Least-to-Most Prompting](https://arxiv.org/abs/2205.10625), [Decomposed Prompting](https://arxiv.org/abs/2210.02406), [ADaPT](https://arxiv.org/abs/2311.05772), [Tree of Thoughts](https://arxiv.org/abs/2305.10601), [Self-Refine](https://arxiv.org/abs/2303.17651), and [SCREWS](https://arxiv.org/abs/2309.13075) establish decomposition, branch selection, and iterative revision patterns.

## C. Strategy Routing, Harness Search, and Negative Evidence

### C1. Select-then-Solve

Primary source: [arXiv:2604.06753](https://arxiv.org/abs/2604.06753)

Reported design and result:

- Compares direct answering, chain of thought, ReAct, plan-and-execute, reflection, and code-oriented reasoning.
- Evaluates four model families across ten benchmarks.
- Reports substantial per-task oracle complementarity.
- A lightweight embedding router reportedly beats the best fixed paradigm by 2.8 percentage points.

Implication:

- Generic reasoning-strategy routing is already occupied.
- The best fixed strategy and a learned one-shot strategy router are mandatory baselines.

### C2. Statistical Scouting Finds Debate-Safe but Not Debate-Useful Cases

Primary source: [arXiv:2605.09618](https://arxiv.org/abs/2605.09618)

Reported design and result:

- Compares direct, voting, and debate protocols under a matched maximum generated-token ceiling.
- Finds meaningful oracle headroom for selecting cases that benefit from debate.
- Logistic-regression and gradient-boosted routers fail to beat the best fixed policy reliably.
- Uncertainty predicts cases where debate is safe more readily than cases where it is useful.

Implication:

- Detecting current-answer uncertainty is not the same as predicting action benefit.
- Oracle headroom is necessary but not sufficient. The signal must be predictable out of sample.

### C3. Test-Time Harness Evolution

Primary sources:

- [TTHE](https://arxiv.org/abs/2607.08124)
- [MemoHarness](https://arxiv.org/abs/2607.14159)
- [Rethinking the Evaluation of Harness Evolution for Agents](https://arxiv.org/abs/2607.12227)

Reported findings:

- TTHE evolves a harness at test time and explicitly identifies selection regret versus proposal coverage.
- MemoHarness searches several harness control surfaces and stores experience, but reports small held-out sets, incomplete ablations, cache-dependent cost, and heuristic control.
- The evaluation reassessment reports that automatic harness evolution does not consistently beat matched-feedback or matched-budget test-time scaling and generalizes poorly to held-out tasks.

Implication:

- Generic harness evolution is crowded and has fresh negative evidence.
- Fixed-policy, matched-feedback, matched-budget, and held-out evaluations are mandatory.

### C4. Metacognitive Harnesses

Primary source: [arXiv:2605.14186](https://arxiv.org/abs/2605.14186)

Reported design:

- Elicits pre-solve feeling-of-knowing and post-solve judgment-of-learning signals from a frozen model.
- A fixed harness uses those signals to trust, retry, or aggregate.

Implication:

- Generic self-evaluation state is crowded.
- The remaining question is whether observations predict exact counterfactual action benefit, not whether they predict correctness.

### C5. Action-Interface Stability

Primary sources:

- [Calibration Is Not Control](https://arxiv.org/abs/2606.21399)
- [TemplateRL](https://arxiv.org/abs/2505.15692)
- [ReasonOps](https://arxiv.org/abs/2605.29192)
- [A Regime Theory of Controller Class Selection](https://arxiv.org/abs/2605.06339)
- [POSIX](https://aclanthology.org/2024.findings-emnlp.852/)
- [ProSA](https://aclanthology.org/2024.findings-emnlp.108/)
- [Flaw or Artifact?](https://aclanthology.org/2025.emnlp-main.1006/)
- [Action Boundary Blindness](https://aclanthology.org/2026.acl-long.1711/)

Reported designs and limits:

- Calibration Is Not Control defines intervention advantage and executes candidate actions from identical prefixes. Generic same-prefix action uplift is therefore occupied.
- TemplateRL defines a template action as a prompt intended to elicit a reasoning type, groups surface paraphrases under one action label, and ablates action-library size. It does not report a same-state interventional comparison across independently written action realizations.
- ReasonOps measures associations between reasoning operators, problem difficulty, and model identity, including limited cross-model transfer. Its analysis is primarily observational rather than a matched action intervention.
- Regime Theory states that controller-class diagnostics are conditional on the specified action set, loss, features, and judge. It does not validate whether the prompted action interface itself is stable.
- POSIX and ProSA make general intent-preserving prompt sensitivity a mature evaluation topic.
- Flaw or Artifact? shows that apparent prompt sensitivity can be inflated by rigid or heuristic evaluation. Exact symbolic reward avoids that specific confound.
- Action Boundary Blindness studies failures of action granularity, scope, and completeness across six agent benchmarks and reports improvement from explicit boundary prompting. It does not compare independently authored realizations of the same controller action, but it directly occupies broad claims about unreliable action semantics.

Residual question:

> Does within-action variation across prompt realizations or workers exceed between-action variation, and does this instability change exact downstream action value?

Threat level: medium to high. The same-state control-regret test appears residual, but generic prompt sensitivity, action-boundary reliability, action ablation, common-prefix advantage, and cross-model operator analysis are occupied. Treat action identifiability as a necessary validity screen unless the control consequences are large and systematic enough to support a standalone paper.

### C6. Controller State Versus Worker Rendering

System precedents:

- [Graph of Thoughts](https://arxiv.org/abs/2308.09687) separates graph state, controller choice, prompt rendering, parsing, and writeback.
- [HarnessBridge](https://arxiv.org/abs/2606.12882) retains authoritative history while a learned harness projects a generator-visible history and mediates actions.
- [VISTA](https://arxiv.org/abs/2606.30005) and [Beyond Compaction (CWL)](https://arxiv.org/abs/2606.11213) use typed addressable state and deterministic context management.
- [StateFlow](https://arxiv.org/abs/2403.11322), [SWE-agent](https://arxiv.org/abs/2405.15793), and [TSCG](https://arxiv.org/abs/2605.04107) separate system state or schemas from state-specific model-visible interfaces.
- [HiLight](https://arxiv.org/abs/2604.22565) learns a non-destructive task-relevant lens for a frozen solver.

Structured-format evidence:

- [Let Me Speak Freely?](https://arxiv.org/abs/2408.02442) reports reasoning losses under structured output restrictions and recovery from reasoning before formatting.
- [The Format Tax](https://arxiv.org/abs/2604.03616) reports that prompt-side format requests explain much of the loss across open and API models, with delayed reformatting recovering many cases.
- [The Constraint Tax](https://arxiv.org/abs/2605.26128) reports validity-versus-correctness tradeoffs for 0.5B to 3B workers and improvement from delayed constraint application.
- [Draft-Constrained Constrained Decoding](https://arxiv.org/abs/2603.03305) and [Thinking Before Constraining](https://arxiv.org/abs/2601.07525) separate unconstrained reasoning from constrained packaging.

Overlap assessment:

- External machine-native state, worker-visible projection, learned lenses, and delayed formatting are all occupied individually.
- No located primary work cleanly crosses information-matched controller representation with worker input rendering and output contract under exact terminal reward.
- The residual is a causal two-consumer interface experiment, documented in [`controller-worker-interface.md`](controller-worker-interface.md), not a new architecture claim.

Threat level: medium to high.

## D. Current Local Evidence

Primary project source: [`../docs/superpowers/research/EXPERIMENTS.md`](../docs/superpowers/research/EXPERIMENTS.md)

| Experiment | Comparison | Result | Valid reading |
|---|---|---:|---|
| E39 | direct vs teacher-controlled retained revision | 43/60 vs 43/60 | The coordinator made 12 revision decisions but always retained the old candidate. Candidate oracle was 46/60. No end-to-end gain. |
| E40 | fixed retained revision vs protected direct baseline | 49/60 vs 43/60 | A fixed revision policy improved six cases with no baseline-only loss. |
| E41 | fixed independent revision vs E40 retained revision vs direct | 55/60 vs 49/60 vs 43/60 | The fixed independent revision prompt was the dominant observed policy, with 12 revision-only wins and no direct-only wins. |
| E42 | generic vs explicit GPT-4.1-mini routing prompt on E41 replay | 46/60 vs 51/60; oracle 55/60 | Prompting the coordinator matters, but neither coordinator beat the fixed action. This was replay analysis, not live end-to-end control. |
| E43 | target-value census | invalid | Schema failure prevents any conclusion about target policy value. |
| E43R1 | repaired two-key structured prompt, 12 cases | structured submit 1/12, branch retries 2/12, root/direct 6/12 | The integration accepted all calls, but decomposition and branch lenses substantially harmed the 3B worker. |

Current empirical interpretation:

1. The strongest result is a better fixed revision prompt, not a controller result.
2. E42 has one-way action dominance, so it does not establish instance-level routing headroom.
3. E43R1 says the present decomposition and lens are not yet a viable substrate for policy learning.
4. Controller training should remain gated on structured-baseline viability, bidirectional action complementarity, oracle headroom over best fixed, and out-of-sample predictability.

## E. Novelty Verdict at the Cutoff

### Claims that should not be made

- A small external NN can orchestrate a frozen LLM.
- RL can learn which reasoning strategy to prompt.
- A controller can choose whether or how to revise a frozen model's answer.
- A controller can select or curate the context shown to a frozen executor.
- A semantic reasoning tree makes local backtracking possible.
- Error localization can guide prefix-preserving regeneration.
- Common-prefix counterfactual outcomes can define intervention value.
- Terminal-outcome rollouts can train a Q or advantage model over reasoning steps.
- Exact or formal verification can localize and guide repairs.
- Factorized policy heads or typed actions are new in isolation.

### Narrow question that may remain

Under exactly one additional frozen-worker call, can an external policy select among semantic reasoning-node targets and typed retry lenses to maximize exact symbolic-verifier uplift, outperforming the raw worker, the best fixed action and lens, random targeting, self-localization, an error-localization classifier, a large-LLM selector, and compute-matched sampling on held-out problems?

Novelty risk: high.

This question survives only as a narrow conjunction. It cannot claim novelty for frozen-worker control, targeting, retry, action advantage, counterfactual branching, semantic thought structure, or symbolic repair separately.

### Exact intersection not found as of the cutoff

No located paper combines all of the following:

- a frozen 3B to 14B worker;
- a pre-existing semantic decomposition tree;
- `KEEP`, whole-answer retry, and multiple node-specific local-lens retries;
- a shared lightweight controller trained only from terminal symbolic outcomes;
- at most one additional worker call; and
- strict comparison with fixed, heuristic, worker-selected, strong-LLM-selected, and learned policies under call and token matching.

This is evidence of a narrow untested intersection, not proof of novelty. Both Calibration Is Not Control and CausalFlow make it an obvious extension.

## F. Residual Search Uncertainty

- Several July 2026 manuscripts are too recent for meaningful forward-citation coverage.
- No located paper combines semantic node pointers, exact one-call counterfactual labels, and a tiny black-box controller, but absence of a search hit is not proof of novelty.
- Recent arXiv venue and version metadata may change after this cutoff.
- A submission should rerun the narrow action-interface, semantic-state, and July-cluster searches immediately before claiming novelty.
