# Action Space as Lenses over a Verifiable Context MDP

**Status:** design approved in brainstorming (2026-06-23), pending spec review
**Scope:** the action space of the LLM context MDP in `verifier_anchored_rl`; updates to `paper/`, `design.md`, and `docs/slides/` follow from this spec
**Framing (do not regress):** RL first, MCTS second. The central object is an agent (policy + Q heads) learning over an induced context MDP whose only reward source is a symbolic verifier. MCTS is a sample-efficient policy-improvement operator that supplies stronger training targets and a test-time compute knob, not the central object.

## 1. Problem

The current action space is **verb-only**. The Q head is categorical over seven verbs (`DECOMPOSE`, `COMPUTE_SUBEXPR`, `FORM_HYPOTHESIS`, `CHECK_CASE`, `INVOKE_VERIFIER`, `BACKTRACK`, `COMMIT_ANSWER`); the action's **target identifier and free-form content are both sampled by the frozen LLM** as part of the stochastic transition. Three problems followed, all confirmed by an audit of the code, `design.md`, `method.tex`, and the decks:

1. **Credit blur.** `Q(s, verb)` averages over every target the LLM might pick, so the agent never learns *which node to act on*. `design.md`, `method.tex`, and the slides all flag this as an unresolved tension.
2. **`BACKTRACK` is misnamed.** In code it is already a forward prune (mark `abandoned`, move subgoal open to closed, retain all history). It is not a rollback. The word "backtrack" invited a rollback reading that the implementation never had.
3. **Asymmetric closure.** `COMMIT_ANSWER` is root-only and target-free, so a subgoal can only be *abandoned*, never *solved*. There is no positive closure for a subgoal.

This spec redesigns the action space to fix all three, minimizes the verb set against the prior-art literature, and preserves the verifier-anchored reward invariant.

## 2. Core principle: an action is a lens

An action is `a = (verb, target)`, where `target` is an identifier into the structured state (a goal tree of facts, hypotheses, and subgoals). Each verb is a **lens**: a focusing operation on the state with a scoped read and a localized write.

A verb is fully specified by five fields:

| Field | Role |
| --- | --- |
| `avail_v(s, t)` | state-conditioned legality mask |
| `schema_v(s, t)` | JSON schema constraining the LLM output |
| `view_v(s, t)` | scoped read (the lens *get*): the projection of `s` rendered into the LLM context |
| `apply_v(s, t, o)` | localized write (the lens *put*): the update applied to `t` and its subtree given LLM output `o` |
| reward | mostly zero; verifier-anchored (Section 4) |

The transition is `o ~ frozenLLM(view_v(s, t) ++ suffix_v)`, then `s' = apply_v(s, t, o)`. Formally `P(s' | s, a)` integrates over LLM continuations, so the MDP is generative (sample, do not enumerate), exactly as the existing `GenerativeMDP` protocol requires.

### 2.1 Action type and the factored head

The target is promoted into the chosen action rather than left to LLM sampling. The policy factors as `pi(verb | s) * pi(target | s, verb)`: a categorical **verb head** plus a **pointer head** that scores the identifiers present in the state. MCTS branches over `(verb, target)` pairs. This is what fixes the credit blur: value and search now range over what to operate on, not just the cognitive move. The effective action set is `verbs x valid-targets`, which grows with the tree, which is precisely why the head is a pointer over live identifiers and not a fixed K-vector.

### 2.2 Global value, local generation

The frozen LLM generates content from the **local** view `view_v(s, t)`. The learned heads value on the **global** state: `Q_psi(emb(s), verb, target)` and `pi(emb(s))` embed the whole tree, because whether an action helps depends on the whole problem. The full state `s` remains the Markov state; the view is a projection for generation, never the state itself. The embedding `emb(s)` comes from a **decoupled embedder** (a text-embedding endpoint), not the generator's hidden state, so generation can be served by vLLM and the encoder is pluggable: the legacy LLM `[STATE_END]` hidden state is a selectable baseline, and a graph neural network over the reasoning tree (Section 11) is the natural structural encoder. The split is a clean fit for global value and local generation: the embedder sees the whole tree, the generator sees the local view.

### 2.3 Recursion is derived, not asserted

`view_v(s, t)` scoped to `t`'s subtree is the state of the sub-MDP rooted at `t`. So acting on a subgoal is stepping the sub-MDP induced by that subgoal, with identical machinery. Locality, modularity, and recursion are the same property. This is the recursion story the work needs, now derived from the action interface rather than deferred to a separate paper (the former `paper-recursive/` companion was deleted; the recursion lives here and in the main paper).

### 2.4 Default view scope

Each verb defines its own `view_v` (fully customizable). The **default** view, when the target is a subgoal, is **grounded-local**: the problem statement, the ancestor path root to target, and the target's own subtree (its facts, hypotheses, open children). Sibling subgoals and unrelated branches are hidden. Verbs that need more (for example a computation pulling specific facts) override the default. Scoped views are load-bearing for the deployable-small-model claim: small models reason poorly across long contexts, so the lens keeps each generation short and on topic.

## 3. The verb set: 6 core, 1 optional

From the original nine candidates (the seven above plus the proposed `CRITIQUE` and `SUMMARIZE`), the set reduces to six core verbs plus one optional, with `SUMMARIZE` dropped and `CHECK_CASE` folded.

| Verb | Family | Target | Realized by | Reward | Node effect |
| --- | --- | --- | --- | --- | --- |
| `SPLIT(goal, connective, exhaustive)` | structure | a goal | LLM | none | spawn children; connective in {AND, OR} recorded on the edge |
| `FORM_HYPOTHESIS(goal)` | propose | a goal | LLM | none | add a Hypothesis (tentative, unverified) |
| `COMPUTE_SUBEXPR(goal)` | propose | a goal / operands | LLM | none | add a Fact, source = derived |
| `VERIFY(node)` | assess (symbolic) | a hypothesis or fact | SymPy | gated, capped (sole reward) | on success, promote to source = verified |
| `COMMIT(goal, hypothesis)` | resolve (+) | a goal and the hypothesis to promote | promote | terminal at root only | close goal as solved; record solution |
| `ABANDON(node)` | resolve (-) | a goal or hypothesis | bookkeeping | none | negative closure, forward prune, history retained |
| `CRITIQUE(node, goal)` *(optional)* | assess (LLM) | a hypothesis vs its goal | LLM | none | reward-free annotation; may spawn a concern child |

Dropped: `SUMMARIZE`. Folded: `CHECK_CASE` into `SPLIT`. Renamed: `BACKTRACK` to `ABANDON`. Generalized: `COMMIT_ANSWER` to a recursive `COMMIT`.

### 3.1 SPLIT (folds DECOMPOSE and CHECK_CASE)

`DECOMPOSE` (conjunctive subgoals: solve all children) and `CHECK_CASE` (disjunctive or exhaustive cases: cover the parent) are the same move (expand a goal into children) differing only in the AND/OR semantics of the children. They merge into `SPLIT(goal, connective in {AND, OR}, exhaustive: bool)`. The connective is a structured property recorded on the edge, not inferred, because `COMMIT`'s aggregation rule depends on it: a conjunctive goal closes only when all children close; a disjunctive or exhaustive goal closes per its coverage rule. Prior art supports this directly: only ADaPT distinguishes conjunctive from disjunctive composition, and it does so as an And/Or label on a single decomposition act, never as two verbs.

### 3.2 FORM_HYPOTHESIS and COMPUTE_SUBEXPR (kept distinct)

These are the propose family. They are kept as two sibling verbs rather than merged into one `PROPOSE(kind)` verb, because the epistemic status of the node they write back differs (a Hypothesis is tentative and unverified; a Fact with source = derived is a computation), the write-back differs, and the rendered view differs. The distinction is load-bearing: `VERIFY`'s reward gate fires on the promotion of a node to source = verified, and a derived fact must never be treated as verified without a verifier route. Whether to instead expose a single parameterized `PROPOSE(kind in {hypothesis, fact})` is logged as an ablation (Section 7), since the only difference is whether the policy head learns cleaner distributions from explicit verbs.

### 3.3 VERIFY (the defining verb, sole reward)

`VERIFY` (the renamed `INVOKE_VERIFIER`) invokes the symbolic oracle (SymPy). On a gated, capped success it promotes the target node to source = verified and emits the reward. It is the only channel that touches reward. It is kept strictly distinct from `CRITIQUE` (Section 3.7). This is the project's principal novelty relative to the entire surveyed literature: hard symbolic verification as the sole reward source.

### 3.4 COMMIT (recursive, verifier-gated at the root)

`COMMIT(goal, hypothesis)` promotes a hypothesis to a goal's solution and closes the goal as solved. It is recursive: any goal node can be committed.

- Root commit is terminal and gated by the verifier verdict (+1 on a correct answer, -1 on incorrect, matching the current `COMMIT_ANSWER` terminal). The structural fact `target == root` is what makes a commit terminal, so there is no separate `GENERATE_FINAL_ANSWER` verb.
- Subgoal commit is reward-free bookkeeping: it records the subgoal's solution for its parent. A subgoal's correctness, when checkable, flows through `VERIFY`, not through `COMMIT`. Credit for good subgoal solving propagates back through the discounted return and the MCTS backup, not through a subgoal reward.
- Aggregation reads the `SPLIT` connective: a conjunctive parent requires all children committed; a disjunctive or exhaustive parent requires the coverage rule.

This is "propose then commit": the final answer is `FORM_HYPOTHESIS(root)` followed by `COMMIT(root, that hypothesis)`.

### 3.5 ABANDON (negative closure, kept on B=0 grounds)

`ABANDON(node)` is the renamed `BACKTRACK`: negative closure that forward-prunes a branch while retaining its full history (the revert-to-prefix-with-history representation, matching Self-Backtracking). It is symmetric to `COMMIT`: every goal now closes either positively (solved) or negatively (abandoned), root or subgoal.

Prior art overwhelmingly treats backtracking and pruning as a search-control primitive (MCTS select, expand, backprop), not a learned verb. `ABANDON` is kept as a verb anyway, for one framework-specific reason: at inference budget `B = 0` the deployed small model runs no MCTS, so if pruning lived only in the planner the policy could not express "this branch is dead, move on." Self-Backtracking is the precedent for internalizing the prune as a learned, emittable action precisely so the model can do it without a search controller. `ABANDON` is framed strictly as goal-state closure recorded in the verifiable state, explicitly not as the planner's backup or re-descent. Whether the recorded `ABANDON` marker measurably beats MCTS-only pruning is logged as an ablation (Section 7).

### 3.6 SUMMARIZE (dropped, subsumed by the lens)

`SUMMARIZE` is removed as a verb. No surveyed system makes summarization a policy action; compression is always a serialization or scoped-view concern (LATS state-as-history, GoT Aggregate, fixed serialization). In this design the lens already controls what slice of the tree is rendered, so compressing a closed subtree to "statement plus outcome" is a deterministic rendering property of `view_v`, not a state-mutating action. Closed (committed or abandoned) subtrees render as their statement and verdict.

### 3.7 CRITIQUE (optional, reward-free)

`CRITIQUE(node, goal)` is an LLM self-assessment of a hypothesis against its goal. It writes a reward-free critique record and may spawn a child concern node (a sub-hypothesis or subgoal capturing what must be resolved); the policy decides whether to address, abandon, or commit anyway. It is kept distinct from `VERIFY` (the sharpest coherence boundary): `VERIFY` is a sound symbolic oracle and the sole reward source, `CRITIQUE` is an unreliable LLM self-judge that must never gate reward. Conflating them is the documented failure mode (CRITIC). `CRITIQUE` is optional and not part of the completeness core; many strong systems omit verbal critique, so whether it earns its tokens in a verifier-anchored setting is an ablation (Section 7).

## 4. The verifier-anchor invariant (reward)

The symbolic verifier is the **only** reward source. This is preserved and reinforced, not relaxed. The literature actively supports it: DeepSeek-R1 explicitly rejects learned process-reward models (reward hacking and ill-defined step boundaries); CRITIC shows that self-critique without external grounding is unreliable; the most reliable systems (AlphaProof, AlphaGeometry, rStar-Math, R1, Tulu-3) are all hard-verifier-anchored.

- Most verbs return reward 0.
- `VERIFY` returns a gated, capped positive reward only on a genuinely new promotion to a verified fact. The anti-farming gates (hard cumulative cap, dedup per target, reject tautologies, value-tie to target) are retained unchanged.
- Root `COMMIT` returns the terminal verdict (+1 / -1).
- `CRITIQUE`, subgoal `COMMIT`, and the (now lens-based) compression are all reward-free.
- "More signal" means more verifier-derived shaping, never a reward model or an LLM-judge reward. The moment reward comes from a model's opinion rather than the verifier's check, the project has changed.

## 5. Second-order decisions (resolved)

1. **Subgoal commit gate.** Resolved at design time as reward-free bookkeeping; verification, when applicable, routes through `VERIFY`. (The stricter verify-then-commit gate is recorded as an option but not adopted by default.)
2. **Critique output.** Record plus optional concern child, reward-free.
3. **SPLIT vs separate verbs.** `DECOMPOSE` and `CHECK_CASE` merge into `SPLIT`; the AND/OR distinction is preserved as a recorded edge property.
4. **Action granularity.** An action may write a localized multi-node chain (for example a step-by-step expansion under a subgoal), still atomic at the MDP level; the verb decides how much it writes. Intrinsically multi-node verbs (`SPLIT`) already do this. No macro solve-it-all verb is added, because that would bypass the per-step search and verification that are the point.
5. **Phase boundary (honesty).** The paper and decks present this `(verb, target)` plus lens system as the method, with explicit margin notes marking what the Phase-1 code realizes today (verb-only, global view, seven verbs) versus the designed target (pointer head, scoped lenses, the reduced verb set, `CRITIQUE`). We do not claim as built what is still spec.

## 6. Prior-art grounding (summary)

Across roughly thirty systems (ToT, GoT, RAP, LATS, ReST-MCTS*, rStar and rStar-Math, MCTSr, TS-LLM, Self-Refine, Reflexion, CRITIC, ReAct, Least-to-Most, Decomposed Prompting, Plan-and-Solve, ADaPT, PoT and PAL, AlphaProof, AlphaGeometry, DeepSeek-R1, Tulu-3, STaR and V-STaR, Stream of Search and GSoS, Searchformer, Self-Backtracking, SCoRe, ASTRO) the operation vocabulary collapses to five families: propose/expand, verify/evaluate, critique/refine, commit/close, prune/backtrack.

Key data points for this design:
- **Minimal verifier-anchored systems are tiny.** AlphaGeometry has exactly two verbs (construct, deduce); AlphaProof is a single tactic space. This validates aggressive minimization.
- **Reward source is the load-bearing axis.** DeepSeek-R1 rejects learned PRMs; CRITIC shows external grounding is what makes critique reliable. Strong support for verifier-only reward.
- **Backtracking is search-control by default.** Every MCTS system puts it in selection; it becomes a learned action only in the internalized-search cluster (Self-Backtracking, ReVISE, SCoRe, ASTRO, GSoS), which is exactly the B=0 deployment case that justifies keeping `ABANDON`.
- **Closest structural analogue:** TS-LLM (PUCT MCTS plus learned value plus policy distillation, that is, generalized policy iteration). It differs from this project principally in reward source: it trains an outcome reward model, whereas this project uses a symbolic verifier as the sole, untrained reward. That difference is the main novelty.
- **GSoS operation-level MDP** (each search operation is one discrete action) is the closest analogue to the `(verb, target)` action MDP and validates operation-level over token-level credit assignment.

## 7. Open questions (ablations, not decided here)

1. `PROPOSE(kind)` as one parameterized verb versus two sibling verbs (`FORM_HYPOTHESIS`, `COMPUTE_SUBEXPR`). The epistemic distinction survives either way; the question is policy-head learnability.
2. Does a policy-emitted, recorded `ABANDON` measurably beat MCTS-only pruning?
3. Does reward-free `CRITIQUE` improve trajectories in a verifier-anchored setting, or just add tokens?
4. Is the exhaustiveness obligation of a disjunctive `SPLIT` itself a `VERIFY` target (a logical check), or an unchecked structural flag (a soundness hole if unchecked)?

## 8. Relationship to current code

The current `verifier_anchored_rl` implements the verb-only baseline: seven verbs, a categorical Q head, a global `serialize_state`, and an LLM verb-instantiator that samples target and content. This spec diverges from that baseline in four places, in rough order of effort:

1. Rename `BACKTRACK` to `ABANDON`; generalize `COMMIT_ANSWER` to a recursive `COMMIT(goal, hypothesis)`; fold `CHECK_CASE` into `SPLIT(connective)`; remove `SUMMARIZE` from any plan to add it; add the optional `CRITIQUE`.
2. Promote the target into the action: add the pointer head and make MCTS branch over `(verb, target)`.
3. Add per-verb scoped `view_v` (grounded-local default) and localized `apply_v` write-backs; keep the heads on the global state embedding.
4. Persist the `SPLIT` connective and the committed/abandoned closure status structurally so aggregation and rendering can read them.

Phase-1 validation anchors (gridworld, expectimax, MCTS triangulation) are unaffected.

## 9. Downstream artifacts (follow from this spec)

Priority reflects an upcoming meeting: slides first, then the documents that capture the work so it is not lost.

1. **Slides (highest priority, needed for the meeting): `docs/slides/2026-06-23-verifier-anchored-rl-talk.tex` and the 10-minute cut.** Update the verbs frame (the seven-verb set becomes the reduced set) and add a highlighted **lens** beat. Make the lens the headline of the redesign: an action renders a scoped, grounded-local view of the goal tree and writes back locally, and that is what lets a small language model reason far more effectively (short, on-topic contexts instead of one long flat trace), tying the lens directly to the deployable-small-model thesis. Also surface the `(verb, target)` factored action and the verifier-anchor.
2. **`paper/sections/method.tex`:** rewrite the action-space section to the lens model and the reduced verb set, with notes marking Phase-1 code versus the designed target.
3. **De-lean the paper on outdated experiments.** All current experimental results predate this redesign and no longer reflect the system. Move the outdated experiment artifacts to an archive or deprecated location (for example `experiments/archive/`) rather than deleting them, and rewrite the paper's experiment, methodology, and results sections (including `paper/sections/appendix_pilot.tex`) so they SCAFFOLD the planned evaluation without claiming the stale numbers. Remove prose that leans on the old verb set or old runs (for example the pilot's modal-verb-flip observation).
4. **`design.md`:** update the action-space section and the recursion story to this spec.
5. **Scrub the deleted `paper-recursive/`:** remove references in `README.md` (lines 9-10, 33), `CLAUDE.md` (line 7), and the companion-paper mention in `paper/sections/method.tex`.

All paper-touching subagent briefs must restate the RL-first directive verbatim (per repo `CLAUDE.md`).

## 10. Non-goals

- No reward model and no learned process-reward; the verifier is the sole reward source.
- No LLM-judge reward; `CRITIQUE` is reward-free information.
- No macro solve-it-all verb that bypasses search and verification.
- No regression of the Phase-1 gridworld, expectimax, and MCTS correctness anchors.
- No MCTS-first reframing in any artifact.

## 11. Pinned extensions (recorded, not in scope now)

- **`RETRIEVE` (RAG exemplar retrieval).** A candidate introduce-from-corpus verb: fetch a solved example similar to a goal or subgoal and write it into that node's subtree as grounding. It fits the lens framing cleanly (an introduce verb whose content comes from a corpus rather than the LLM) and is reward-free. Pinned, not adopted, for three reasons: it adds an external corpus and retriever dependency beyond the verifier-anchored core; a retrieved example can leak the answer, which is exactly the compute and data-leak confound the evaluation plan already guards against; and it does not change the thesis. Revisit after the core is validated.
- **`GraphEmbedder` (GNN state encoder).** The heads currently embed the serialized state through a decoupled text-embedding endpoint. Because the state is a goal tree (a graph), a graph neural network over that structure is the natural encoder: it would respect the true structure (parent/child links, fact-vs-hypothesis edges, the `SPLIT` connectives) rather than a linearized serialization. The `Encoder` protocol already admits it as a drop-in (the code anticipates a `GraphEmbedder`). Pinned as a future encoder upgrade; orthogonal to the action-space design.
