# Prior Art for Explicit Tree State, Node Actions, Lenses, and Writeback

Cutoff: 2026-07-18

Status: completed primary-source review at the cutoff. This file is intentionally separate because the representation question is more specific than generic external LLM control.

## 1. What Representation Is Being Assessed?

The project's motivating state is an evolving semantic workspace, not merely a search log:

```text
workspace = forest(
  nodes = goals, subgoals, hypotheses, facts, candidates, attempts,
  edges = parent/child, dependency, evidence, ownership,
  node_fields = content, type, status, score or receipt, provenance,
  global_fields = remaining budget, history, legal action menu
)

action = (verb, target_live_node)
lens = render(target, ancestors, dependencies, selected siblings, local attempts)
transition = call frozen worker on lens, parse result, write back near target,
             then update dependent or ancestral state
```

Six claims must be assessed separately:

1. **Explicit structure:** reasoning content is stored as a persistent tree or graph rather than only as a flat transcript.
2. **Semantic nodes:** nodes are goals, subproblems, claims, or artifacts, not merely alternative token continuations.
3. **Addressability:** the controller can choose a currently live node.
4. **Factored action:** the controller chooses an operation type and a target, explicitly or implicitly.
5. **Lens:** the target determines the bounded context supplied to the worker.
6. **Writeback:** the worker's output updates that node or its neighborhood and can propagate upward or downstream.

Search-tree papers often satisfy only item 1. The closest prior art satisfies four or five of the six.

## 2. Headline Verdict

The representation rationale is well grounded, but the broad design is not novel.

- [Graph of Thoughts](https://arxiv.org/abs/2308.09687) already defines a persistent Graph Reasoning State whose vertices are typed thoughts, whose edges encode dependencies, and whose controller selects both a transformation and the thoughts to which it applies. Its Prompter encodes the selected graph context and its Parser writes the result back into the graph.
- [DeAR](https://proceedings.neurips.cc/paper_files/paper/2024/file/01025a4e79355bb37a10ba39605944b5-Paper-Conference.pdf) builds a semantic subquestion tree. Each node stores a question, rationale, and score. Its fixed cycle applies Decompose, Analyze, and Rethink to selected nodes, and child results update selected ancestors.
- [Knowledge Graph of Thoughts](https://arxiv.org/abs/2504.02670) maintains a dynamically evolving task-state graph in a graph store; a controller queries it, selects tools, and integrates results back into the graph.
- [Adaptive Graph of Thoughts](https://arxiv.org/abs/2502.05078) recursively builds a DAG of interdependent semantic subproblems, gives each node a hierarchical address, and selectively expands complex nodes into nested graphs.
- [Reinforced Graph of Thoughts](https://arxiv.org/abs/2605.22195) adds a tiny RL controller over a structured graph of operations for a fixed LLM.
- [Atomic Task Graph](https://arxiv.org/abs/2607.01942) recursively compiles semantic nodes into interface-preserving subgraphs, exposes node-relevant history, records node-local execution state, and repairs only the affected subgraph while freezing validated regions.
- [TopoAgent](https://arxiv.org/abs/2607.14658) defines a dependency-derived context lens over node memory, writes results back by node, and dynamically splits a failed node into finer subnodes.
- [Framework of Thoughts](https://arxiv.org/abs/2602.16512) separates a mutable execution graph from a derived reasoning graph and gives operations scoped ancestor and descendant views.
- [ReCAP](https://proceedings.neurips.cc/paper_files/paper/2025/hash/88af3540325dd0b70617a9ab605f294d-Abstract-Conference.html), [THREAD](https://aclanthology.org/2025.naacl-long.427/), and [STEP](https://arxiv.org/abs/2506.21030) provide additional direct precedents for recursive child contexts and return or propagation interfaces.

Taken together, these papers contain the core intuition behind `(verb, target)`, local graph-derived prompting, and writeback. The defensible residual is narrower: a tiny learned pointer over arbitrary live semantic nodes, with a rigorously specified lens and exact symbolic outcomes, evaluated causally against the same information in flat form and against fixed graph workflows.

An older formal-methods predecessor further narrows the architectural claim: [TacticZero](https://arxiv.org/abs/2102.09756) learns an explicitly factored `(fringe, goal, tactic, arguments)` action over a persistent set of proof fringes and open semantic goals, with proof-assistant transitions and backtracking.

Novelty risk for the representation alone: high.

Plausibility as a useful inductive bias: moderate, but not yet established by the project's experiments.

## 3. Closest Direct Predecessors

### 3.1 Graph of Thoughts

Primary source: [Besta et al., arXiv:2308.09687](https://arxiv.org/abs/2308.09687)

Formal representation:

- Reasoning is a directed graph `G = (V, E)`.
- A vertex contains an initial, intermediate, or final solution artifact.
- An edge means the destination thought was constructed using the source thought as direct input.
- Nodes may have heterogeneous classes, such as plan versus paragraph.
- The dynamic Graph Reasoning State stores every thought, its status, validity, score, and executed operation.

Operations and targeting:

- Transformations include generation from a thought, aggregation of several thoughts, refinement of a thought through a self-loop, scoring, validation, ranking, keeping the best thoughts, repetition, and removal.
- The Controller selects thoughts from the Graph Reasoning State, selects which transformation to apply to which thoughts, and decides when to finish.
- The Prompter has full access to the graph and encodes the selected inputs into a prompt.
- The Parser extracts the worker output into a thought state and updates the Graph Reasoning State.

Exact overlap:

- persistent addressable graph state;
- typed semantic nodes and dependency edges;
- operation plus target-thought selection;
- target-dependent prompt construction;
- parser-mediated writeback;
- local refine and multi-node aggregate operations.

Important difference:

- In the published system, a static, human-authored Graph of Operations dictates the execution plan. The controller is software, not a learned tiny policy over arbitrary live-node actions.
- Scoring can use humans, LLMs, or task-specific functions, and is not restricted to terminal symbolic reward.

Assessment:

> The proposed `(verb, target)` interface is best described as a learned, factored generalization of the Graph of Thoughts Controller API, not a new representation primitive.

Threat level: critical.

### 3.2 DeAR: Decompose, Analyze and Rethink

Primary source: [Xue et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/01025a4e79355bb37a10ba39605944b5-Paper-Conference.pdf)

Formal representation:

- A reasoning tree `T = (N, E)` is progressively constructed and updated.
- Each node is a tuple `(question, rationale, coherence_score)`.
- Parent-child edges mean the parent question was decomposed into the child subquestion.
- The root rationale is the final answer-bearing solution.

Operations and targeting:

- `Decompose(node)` creates child subquestions.
- `Analyze(child)` solves and self-checks a child.
- `Rethink(ancestor, child_result)` selects related ancestors and updates their rationales from new child information.
- A queue chooses the next edge node, and the process repeats until no further decomposition is requested or a depth limit is reached.

Exact overlap:

- semantic subproblem tree rather than alternative full-solution branches;
- persistent node-local content and status-like scores;
- verb-target semantics;
- child-local worker calls;
- writeback and upward propagation.

Important difference:

- Node selection and verb order are fixed heuristics plus LLM decisions, not a small learned policy.
- Context lenses are encoded in stage-specific prompts but are not isolated as an independently ablated mechanism.
- Self-check and coherence scores are LLM-derived.

Assessment:

> DeAR is the closest prior art to the semantic decomposition tree and local-to-parent writeback story. A learned pointer cannot be presented without treating DeAR's fixed node queue and cycle as a primary baseline.

Threat level: critical.

### 3.3 Knowledge Graph of Thoughts

Primary source: [Besta et al., arXiv:2504.02670](https://arxiv.org/abs/2504.02670)

Reported representation and workflow:

- A property graph, RDF graph, or NetworkX adjacency representation stores an evolving task state.
- Nodes and triples store tool outputs and extracted LLM knowledge explicitly.
- A Graph Executor queries the state, identifies missing information, and emits retrieval or insertion operations.
- A Tool Executor selects tools based on graph state and previous attempts.
- Tool results are validated and integrated into the graph; graph entries can also be removed.

Overlap:

- persistent dynamic graph state;
- controller observation through graph queries;
- targeted retrieval as a lens rather than always showing the full history;
- typed insert, retrieve, enhance, solve, and tool actions;
- explicit integration of outputs back into state.

Difference:

- The controllers are LLMs, not a tiny neural pointer policy.
- The graph is a task-relevant knowledge graph rather than a goal and subgoal tree.
- Robustness uses self-consistency and LLM-as-judge mechanisms that do not satisfy project invariants.

Threat level: high.

### 3.4 Adaptive Graph of Thoughts

Primary source: [Pandey et al., arXiv:2502.05078](https://arxiv.org/abs/2502.05078)

Reported representation and workflow:

- AGoT recursively decomposes a query into a DAG of interdependent subproblems.
- Every node has a hierarchical heritage address and stores a thought, layer strategy, answer, and optionally a nested graph.
- A complexity-check action decides whether to solve a node directly or recursively expand it.
- A completed child graph's final answer informs later nodes and the parent computation.
- Depth, layers, and nodes per layer are bounded, while the graph can self-terminate early.

Difference:

- The structural actions are performed by prompted LLM agents, not a tiny external pointer policy.
- The schedule is layer-wise and the key choice is complexity-triggered expansion, not an arbitrary verb applied to any live node.
- Evaluation is not restricted to exact symbolic reward.

Novelty implication:

- The paper explicitly proposes fine-tuning or RL to improve edge selection and complexity checks. A learned controller for semantic graph expansion is therefore both an obvious extension and named future work.
- A defensible contribution must compare against AGoT's complexity heuristic and show value from the more general node-addressed policy under matched compute.

Threat level: critical.

### 3.5 Reinforced Graph of Thoughts

Primary source: [Riesen and von Niederhausern, arXiv:2605.22195](https://arxiv.org/abs/2605.22195)

Reported representation and policy:

- A tiny PPO actor and critic construct and traverse a structured graph of operations.
- State features include depth, breadth, task complexity, local complexity, operation history, previous action, and previous score.
- Actions are stop, backtrack, or append a typed operation.

Overlap:

- tiny external RL controller;
- graph-structured reasoning plan;
- typed operations and backtracking;
- fixed LLM environment;
- exact task checks among the evaluation signals.

Difference:

- The graph is a constrained layered operation DAG, not an arbitrary content graph.
- There is no pointer to an already-live semantic thought node.
- Training occurs in a Bernoulli operation simulator conditioned primarily on task complexity.
- Reward is shaped by intermediate scores.

Threat level: critical for the combined tree-plus-tiny-RL architecture.

### 3.6 Thought-ICS

Primary source: [Structure Enables Effective Self-Localization of Errors in LLMs, arXiv:2602.02416](https://arxiv.org/abs/2602.02416)

- Represents a trace as semantically complete thought nodes in a tree or thought-level MDP.
- Selects a likely first erroneous thought, backtracks to the preceding good prefix, and resamples.
- Directly supports the idea that semantic boundaries make error targets more useful than raw token positions.

Difference:

- Branches are alternate continuations of a mainly linear reasoning path, not sibling subproblems with dependencies.
- Targeting is self-localization or oracle localization, not a tiny multi-action node policy.

Threat level: high for semantic node targeting and retry.

### 3.7 CausalFlow

Primary source: [arXiv:2605.25338](https://arxiv.org/abs/2605.25338)

- Stores a structured dependency trace.
- Intervenes on a selected step, re-executes downstream computation, and measures whether exact task success changes.
- Finds minimal validated repairs and exposes propagation dependencies.

Difference:

- Trace steps are agent execution events rather than an explicit subgoal tree.
- It uses expensive counterfactual re-execution rather than an amortized tiny pointer at deployment.

Threat level: high for target, writeback, propagation, and causal-value claims.

### 3.8 Mechanic and Formal Proof Search

Primary sources:

- [Mechanic, arXiv:2603.24465](https://arxiv.org/abs/2603.24465)
- [HyperTree Proof Search, arXiv:2205.11491](https://arxiv.org/abs/2205.11491)
- [TreeThink, arXiv:2607.11258](https://arxiv.org/abs/2607.11258)

Mechanic uses formal placeholders to isolate unresolved subgoals, constructs clean self-contained contexts, solves subgoals separately, and reassembles them into a preserved proof. This is a direct precedent for a verifier-derived local lens and deterministic writeback.

HyperTree Proof Search represents proof search as a hypertree of goals and tactics, learns from proof search, and relies on the proof assistant for exact state transitions and validation. TreeThink exposes proof states from Lean, Rocq, and Isabelle and supports modular tree search and node evaluation.

[TacticZero](https://arxiv.org/abs/2102.09756) is the strongest architectural ancestor in this family:

- state is a collection of proof fringes, each containing open semantic goals;
- an action is `(fringe, goal, tactic, arguments)`;
- applying the action replaces the chosen goal with formal subgoals and retains other fringes for backtracking; and
- separate learned policy modules select the fringe, goal, tactic, and arguments.

Difference:

- The formal system supplies exact goal boundaries, dependencies, transitions, and verification.
- The research problem is tactic generation or search, not inducing reliable semantic nodes from natural language.

Threat level: high as a structural ancestor, especially for claims that exact verification plus a goal tree naturally yields local actions.

### 3.9 Tree-of-Traversals

Primary source: [ACL 2024](https://aclanthology.org/2024.acl-long.665/)

- Maintains a local knowledge subgraph and uses an action-state machine with think, answer, expand graph, select entities, and select relation actions.
- Prompt templates are conditioned on the current action state and serialize the relevant local graph and legal targets.
- Selecting an entity and relation deterministically retrieves edges and writes them into the local graph.
- Best-first search selects and backtracks over thought and action nodes using LLM-generated values.

Difference:

- Targets are knowledge-graph entities and relations, not semantic reasoning subproblems.
- The evolving knowledge graph and the alternative-trajectory search tree are distinct structures.
- Control is tree search plus LLM scoring, not a learned compact policy or exact verifier.

Threat level: high for generic `(verb, target)`, graph lens, and writeback language.

### 3.10 Branch-and-Bound and Learned Node Selection

Primary sources:

- [RL for Node Selection in Branch-and-Bound, arXiv:2310.00112](https://arxiv.org/abs/2310.00112)
- [Learning to Branch with Graph Pointer Networks, arXiv:2307.01434](https://arxiv.org/abs/2307.01434)

These are outside LLM reasoning but directly establish the architecture pattern:

- the full evolving search tree or solver graph is the state;
- a GNN encodes variable-sized structured state;
- a pointer-like head selects a live node or variable;
- the selected target triggers a fixed typed solver operation; and
- exact solver progress supplies the learning objective.

Assessment:

> A graph encoder plus pointer over a live frontier is established machinery. The LLM-specific contribution must lie in semantic state induction, lens effects, or a demonstrated control regime, not the pointer architecture itself.

Threat level: high for the structural ML pattern, low for the exact application.

### 3.11 Atomic Task Graph

Primary source: [Zhang et al., arXiv:2607.01942](https://arxiv.org/abs/2607.01942)

- A coarse semantic task node is recursively replaced by an interface-preserving subgraph until nodes are atomic tool calls.
- At every refinement, the LLM may see only history directly relevant to that node.
- The system records graph evolution and every node's input, output, status, and error.
- Failure localization traces failed atoms to their lowest common historical ancestor.
- Repair covers only the failed region, relevant upstream context, and affected downstream nodes; validated regions remain frozen.

Difference:

- Refinement, topological scheduling, and local repair are fixed framework operations rather than actions selected by a learned compact policy.
- The domains are interactive tool tasks rather than exact-verifier reasoning problems.

Assessment:

> ATG is the closest current collision with the full semantic-state, lens, localized-writeback, and repair motivation. The remaining claim is policy learning and controlled causal value, not the graph substrate.

Threat level: critical. Evidence maturity is low because this is a July 2026 preprint.

### 3.12 TopoAgent

Primary source: [arXiv:2607.14658](https://arxiv.org/abs/2607.14658)

- Maintains a DAG of atomic semantic subproblems and node-indexed dynamic memory.
- Defines the local context for node `A_i` as the union of states at its completed prerequisites.
- Updates the selected node's memory with tool results.
- When a node exceeds its execution limit, a fixed rule invokes LLM fission into child nodes and later aggregates their results under the parent identity.

Difference:

- Topological scheduling and timeout-triggered fission replace learned node and verb selection.
- Evaluation is multimodal scientific tool use and uses model or tool verification.

Threat level: critical for dependency-lens, writeback, and dynamic-decomposition claims. Evidence maturity is especially low because the preprint was submitted two days before the search cutoff.

### 3.13 Framework of Thoughts

Primary source: [Fricke et al., arXiv:2602.16512](https://arxiv.org/abs/2602.16512), version 2 dated 2026-07-04.

- Separates a mutable execution graph, whose nodes are operations, from a derived reasoning graph, whose nodes are thoughts and whose edges record dependencies.
- Operations can generate thoughts and rewrite the future execution graph while retaining the complete execution history.
- An operation sees only itself and the subgraph induced by its ancestors and descendants.
- Completed ancestors and shared descendants are protected, while an operation may rewrite its exclusive descendants.
- A controller and scheduler execute ready operations, including parallel execution where graph dependencies permit it.

Difference:

- `Controller` in FoT is an execution engine, not a learned compact neural policy that chooses arbitrary semantic-node actions.
- The reported evaluation reimplements existing schemes and measures framework efficiency and prompt optimization. It does not demonstrate that an automatically learned graph policy improves terminal reasoning accuracy.

Assessment:

> FoT occupies the general dynamic graph substrate and a scoped graph-derived view. The defensible residual is an empirical control question: whether semantic state makes exact intervention value predictable, and whether a learned policy earns its cost over fixed scheduling.

Threat level: critical for broad execution-state and graph-lens claims.

### 3.14 GRADE

Primary source: [Zhao, arXiv:2606.22741](https://arxiv.org/abs/2606.22741), submitted 2026-06-22.

- Uses typed step nodes with separate execution and dependency edge layers.
- Marks dependency edges as observed, declared, or inferred, making edge provenance part of the representation.
- Demonstrates failure prediction from dependency features and fault localization from execution features across agent-run corpora.
- Reports that generic graph neural networks can misread inferred structure and overfit run size, while lean provenance-aware features transfer better.

Difference:

- The graph describes an execution trace after the fact rather than a writable semantic problem workspace.
- The prediction target is failure or fault location, not terminal value for alternative interventions.

Assessment:

> A tree encoder cannot be justified merely because the state is a graph. It must outperform size, topology, legal-menu, edge-provenance, and lean feature baselines on held-out action utility.

Threat level: high for broad claims that graph state improves diagnosis, and a mandatory baseline for the observability pivot.

### 3.15 Recursive Context and Return Interfaces

Primary sources:

- [ReCAP, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/88af3540325dd0b70617a9ab605f294d-Abstract-Conference.html)
- [THREAD, NAACL 2025](https://aclanthology.org/2025.naacl-long.427/)
- [STEP, arXiv:2506.21030](https://arxiv.org/abs/2506.21030)

These systems make the local-lens idea concrete in different ways:

- ReCAP maintains a recursive context tree, gives the current child a root or path-local view, and returns child results to refine the parent plan.
- THREAD creates isolated recursive threads using an explicit child-context function and sends selected child output back through a return function.
- STEP traverses a subgoal tree and supplies a node with its parent, the previous same-level node, and current observations before propagating completion upward.

Difference:

- Scheduling is fixed or chosen by the same LLM rather than a separately trained compact policy.
- These papers do not test a factorial comparison of information-matched lenses under terminal symbolic reward.

Assessment:

> Recursive child isolation and return are occupied. The remaining lens claim must identify which dependencies are causally necessary, under matched information and tokens.

Threat level: high for local-context and preceding-sibling novelty.

### 3.16 Pre-LLM Structural Ancestors

The operation-on-target pattern is much older than current LLM systems:

- [Blackboard Architecture for Control](https://doi.org/10.1016/0004-3702(85)90063-3) maintains shared hypotheses, goals, and partial results; a control component selects a knowledge-source action and focus object, then writes results back.
- [Additive AND/OR Graphs](https://hdl.handle.net/2318/103092) formalizes problem reduction, tip selection, and upward propagation of solved status and value.
- [HTN Planning](https://www.cs.umd.edu/~nau/papers/erol1994htn.pdf) selects a compound task and replaces it with a constrained network of subtasks.
- [DeepHOL](https://proceedings.mlr.press/v97/bansal19a.html) and [HyperTree Proof Search](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a8901c5e85fb8e1823bbf0f755053672-Abstract-Conference.html) use learned guidance over persistent formal goal or hypergraph state with exact prover transitions.
- [Neural Program Repair by Jointly Learning to Localize and Repair](https://arxiv.org/abs/1904.01720) separates localization and repair, while [Hoppity](https://openreview.net/forum?id=SJeqs6EFvB) predicts a program-graph node and edit operation.

Assessment:

> Addressable workspaces, problem reduction, operation-target factorization, local rewrite, and upward closure are established abstractions. LLM-specific novelty must come from induced semantic state, a controlled worker interface, or a demonstrated action-value regime.

Threat level: critical for any claim that `(verb,target)` is itself new.

## 4. Search Trees That Are Related but Not Equivalent

| Work | What the tree represents | Why it is not the same state |
|---|---|---|
| [Tree of Thoughts](https://arxiv.org/abs/2305.10601) | Alternative partial solution continuations | Nodes are search candidates, not persistent sibling subproblems with ownership and writeback. |
| [RAP](https://arxiv.org/abs/2305.14992) | MCTS states and proposed reasoning actions | The LLM acts as world model and evaluator; no external learned live-node pointer. |
| [LATS](https://arxiv.org/abs/2310.04406) | Agent trajectories explored by MCTS | A path is a history of actions and observations, not a semantic workspace decomposition. |
| [TS-LLM](https://arxiv.org/abs/2309.17179) | Reasoning candidates under PUCT | Learned policy and value guide search, but training modifies or distills into the LLM and uses a learned outcome reward model. |
| [Guided Stream of Search](https://arxiv.org/abs/2410.02992) | A stream encoding search operations and backtracking | It establishes operation-level actions but internalizes them in the generator rather than exposing a persistent external node graph. |
| [MaAS](https://arxiv.org/abs/2502.04180) | Query-specific DAG of reasoning operators | It plans an operator graph before intermediate content exists; targets are layers or operator profiles, not live semantic nodes. |
| [Neural CoT Search](https://arxiv.org/abs/2601.11340) | A locally searched path of discourse operators | It chooses the next discourse token at a fixed delimiter, with no explicit semantic node store. |
| [Tree-of-Traversals](https://aclanthology.org/2024.acl-long.665/) | Alternative thought and graph-action histories over an evolving local KG | It has typed graph targets and graph-derived prompts, but the targets are entities and relations rather than semantic subproblems. |

These works are still required baselines and conceptual ancestors. They do not, however, fully preempt a content-aware semantic workspace study.

## 5. Component Novelty Map

| Component | Strongest precedent | Status after search |
|---|---|---|
| Persistent graph of typed reasoning content | Graph of Thoughts; KGoT | Occupied |
| Semantic subquestion tree | DeAR; LM2; decomposition prompting | Occupied |
| Hierarchically addressed semantic DAG with selective recursive expansion | Adaptive Graph of Thoughts | Occupied |
| Controller chooses operation and thought target | Graph of Thoughts Controller | Occupied conceptually; learned tiny factorization remains narrower |
| Child result updates parent | DeAR Rethink | Occupied |
| Target-dependent graph prompt | Graph of Thoughts Prompter; KGoT graph queries | Occupied conceptually |
| Parse worker output into graph state | Graph of Thoughts Parser; KGoT integration | Occupied |
| Tiny RL controller over graph operations | RGoT | Occupied |
| Semantic-thought error target and backtrack | Thought-ICS | Occupied |
| Exact counterfactual target value and downstream replay | CausalFlow | Occupied |
| Verifier-derived isolated subgoal lens | Mechanic and formal theorem proving | Occupied in formal domains |
| GNN plus pointer over live structured candidates | branch-and-bound and neural proof search | Occupied outside natural-language reasoning |
| Factored target plus operation over a persistent semantic forest | TacticZero | Occupied in formal theorem proving |
| Interface-preserving semantic-node refinement and local subgraph repair | Atomic Task Graph | Occupied |
| Dependency-only node lens, node memory, and failure-triggered child fission | TopoAgent | Occupied |
| Mutable execution graph separated from derived reasoning graph, with scoped operation views | Framework of Thoughts | Occupied |
| Dependency-aware run graph for failure prediction and fault localization | GRADE | Occupied for diagnosis, not action-conditioned control |
| Recursive child context and explicit return to parent | ReCAP; THREAD; STEP | Occupied |
| Shared addressable workspace plus operation on focus object | Blackboard control systems | Occupied conceptually |
| Problem-node reduction and upward status propagation | AND/OR graphs; HTN planning | Occupied |
| Node localization plus graph edit | Program repair systems | Occupied |
| Tiny learned pointer over arbitrary semantic workspace nodes with exact one-call black-box transitions | No exact match located yet | Narrow residual, high risk |
| Controlled proof that structure or lens beats an information-matched flat state | No exact match located yet | Best candidate empirical contribution |

## 6. What Might Still Be Publishable?

The strongest contribution is not another tree schema. It is a causal representation study:

> Does an explicit semantic workspace make the value of node-targeted interventions more predictable, or improve a frozen small worker, beyond the same information serialized as a flat trace under matched calls and tokens?

This requires separating four effects that are currently bundled:

1. **Decomposition effect:** Does creating subproblem nodes help before any lens or controller is used?
2. **Lens effect:** Given the same target, does local graph-derived context beat full or flat context?
3. **Targeting effect:** Is there meaningful per-instance variation in which node should be acted on?
4. **Representation effect:** Given identical legal actions and information, does a tree or graph encoder predict action uplift better than a flat encoder?

The cleanest design crosses representation and lens while holding content fixed:

| Factor | Arms |
|---|---|
| Information | identical node texts, statuses, receipts, and history |
| Serialization | flat chronological trace; compact table; tree traversal; typed graph |
| Lens | full context; ancestor path; target plus dependencies; local subtree; non-destructive highlight |
| Policy | best fixed; heuristic; teacher LLM; calibrated linear probe; tiny MLP or GNN pointer |

Primary outcomes:

- exact terminal success;
- best-fixed-to-oracle target gap;
- held-out action regret;
- wrong-to-correct and correct-to-wrong transitions;
- generated tokens and worker calls;
- target-node change and final-answer propagation rates.

Go/no-go rules:

- Drop the tree claim if information-matched flat state performs the same.
- Drop the lens claim if full-context or simple highlighting performs the same.
- Drop learned targeting if best fixed or a simple heuristic matches the cross-fitted oracle.
- Drop long-horizon RL until the one-step node-selection problem is learnable.

## 7. Current Evidence from This Project

The existing evidence does not yet validate the representation:

- E41 shows a fixed whole-root independent revision reaching 55/60 on the development source.
- E42's controller does not beat that fixed action.
- E43R1 shows the current two-branch decomposition and local lenses reaching only 1/12 for structured submission and 2/12 for branch retries, versus 6/12 for direct or root retry.

ATG and TopoAgent make this negative result more important, not less: the project must compare against fixed dependency scheduling, fixed local repair, and failure-triggered fission rather than assuming a learned pointer is necessary.

This is not proof that semantic tree state is useless. It is direct evidence that the present decomposition, lens, and synthesis transition do not yet earn their complexity for Qwen2.5-3B. A representation paper would need to treat this negative result as the starting point, not assume that structure helps.

## 8. Residual Search Uncertainty

- No exact paper was located with a tiny external neural pointer over arbitrary live natural-language subproblem nodes and terminal symbolic reward only.
- GoT, KGoT, AGoT, RGoT, FoT, ATG, and TopoAgent make that exact conjunction an obvious integration, so absence of a hit is weak novelty evidence.
- TacticZero and program-repair systems already provide explicit target plus operation factorization outside natural-language reasoning.
- The July 2026 cluster is too recent for reliable forward-citation coverage and should be rechecked immediately before submission.
