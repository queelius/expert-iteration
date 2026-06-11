---
title: "Verifier-Anchored Reinforcement Learning for Small Language Model Deliberation"
subtitle: ""
author: "Alexander Towell"
affiliation: "Department of Computer Science, Southern Illinois University Edwardsville"
email: "atowell@siue.edu"
orcid: "0000-0001-6443-9897"
format: "latex-article"
manuscript: "paper/main.tex"
bibliography: "paper/references.bib"
sections_dir: "paper/sections/"
figures_dir: "paper/figures/"
stage: "drafting"
paper_type: "methodology + experimental plan"
target_venue: "arXiv preprint"
created: "2026-05-22"
last_updated: "2026-05-23"
---

# Project Status

**Stage:** revising. All sections substantively drafted end-to-end: abstract, introduction, background, method, experimental plan, related work, discussion, conclusion. Architecture figure (TikZ) is in place. PDF compiles to 35 pages. The repository also contains an authoritative design document (`design.md`, 1899 lines) that the paper expounds and motivates; design.md section 1 (RL-first directive) is the load-bearing framing constraint and supersedes any older MCTS-first framing in the design document.

**Paper type:** methodology and experimental-plan paper. No empirical results yet; results section deferred to a follow-up after the implementation runs. The pre-registered baseline grid is the load-bearing methodological contribution.

**Target venue:** arXiv preprint (generic article-class format). Specific venue to be selected after first complete draft; candidates include NeurIPS, ICML, ICLR, or a focused workshop on reasoning and test-time compute.

# Central Claim (Thesis)

Small non-reasoning base language models, paired with MCTS-distilled policy and Q-heads over a small verb set, can close the compute-versus-accuracy gap to larger reasoning models on verifiable reasoning tasks (math, logic, formal proofs, programming).

The framework rests on four design commitments:

1. Verifier-anchored signals enter as Bellman targets; learned-aesthetic signals (reasoning-quality classifiers, deception detectors) enter only as MCTS priors, never as reward.
2. A shared frozen LLM serves as both encoder (last-token hidden state, via structural-marker serialization of `MathState`) and stochastic transition model (continuation generation from a `[STATE_END]` sentinel).
3. A multi-head Q architecture over a small fixed verb set (seven actions) rather than embedding-based `(s, a)` scoring, justified by the available action cardinality and gradient-routing properties.
4. A pre-registered baseline grid that defines the compute axis and threats to validity before training begins, making test-time-compute claims falsifiable.

# Outline

Sections are stubbed in `paper/sections/`:

1. `abstract.tex`: first-draft prose complete (single paragraph, ~250 words).
2. `introduction.tex`: motivation, position, contribution list, and paper-structure preview stubbed.
3. `background.tex`: subsection structure for MDPs, Bellman optimality, value iteration and expectimax, MCTS with chance-node bifurcation, expert iteration, and the LLM-as-context-MDP framing. Key equations seeded.
4. `method.tex`: core section. Subsections for overview, state schema (MathState with `Fact`, `Hypothesis`, `VerifierCall`), action space (seven-verb set with state-conditioned availability), architecture (shared LLM and trainable heads), MCTS with chance-node bifurcation, reward structure, training loop (algorithm block included), and inference modes.
5. `experimental_plan.tex`: benchmarks (GSM8K, MATH, AIME, Lean as future option), base models (Qwen2.5-1.5B-Base, Llama-3.2-1B-Base, SmolLM2-1.7B), pre-registered baseline grid B0 through B5, compute accounting with three reported metrics, headline frontier hypothesis, sanity-check ablations, threats to validity.
6. `related_work.tex`: five subsections covering games and AlphaZero lineage, verifier-anchored LLM reasoning, test-time compute scaling, MCTS for language model reasoning, and self-improvement.
7. `discussion.tex`: limitations, extensions, a position subsection on verifier-anchored versus aesthetic signals, threats to the empirical story, and the connection to LLM deliberation beyond math.
8. `conclusion.tex`: brief stub (~0.5 page target).

# Files and Layout

| Artifact | Path |
|----------|------|
| Manuscript root | `paper/main.tex` |
| Bibliography | `paper/references.bib` |
| Section sources | `paper/sections/*.tex` |
| Figures | `paper/figures/` (empty) |
| Design document | `design.md` (authoritative for design decisions) |
| Repository guidance | `CLAUDE.md` |
| LaTeX gitignore | `paper/.gitignore` |

The LaTeX scaffold compiles cleanly with `pdflatex main.tex` (verified, produces 5 pages of stubs).

# References

Starter citations in `paper/references.bib` cover:

- **Search and RL foundations:** Sutton & Barto, UCT (Kocsis-Szepesvari 2006), original MCTS (Coulom 2006), Browne et al. survey.
- **AlphaGo lineage:** AlphaGo, AlphaGo Zero, AlphaZero, MuZero.
- **Expert iteration:** Anthony et al. 2017.
- **Verifier-anchored LLM reasoning:** Lightman et al. "Let's Verify Step by Step", AlphaCode, AlphaProof (DeepMind 2024 blog reference; formal paper forthcoming), GSM8K (Cobbe et al.), MATH (Hendrycks et al.).
- **Recent LLM reasoning:** DeepSeek-R1, OpenAI o1 (system card), STaR, V-STaR.
- **Test-time compute scaling:** Snell et al. 2024, chain-of-thought (Wei et al.).
- **Base models:** Qwen2.5, Llama 3, SmolLM2.
- **MCTS for LLMs:** Feng et al. 2023, Xie et al. 2024.
- **Constrained decoding:** Outlines (Willard and Louf 2023).

Several entries marked `TODO` (with comment notes) for citation verification before submission. The bib file contains the verification instructions at top.

# Review History

## 2026-05-23: multi-agent editorial review

**Recommendation:** minor revision for arXiv preprint; major revision before venue submission.

**Outcome.** Paper is substantively sound. Build is clean (35 pages, no compile errors, all 28 bib entries cited, all references resolve). RL-first reframe applied uniformly across all sections; no residual MCTS-first language. Notation discipline is high; the math is clean.

**Top three blocking issues (for venue submission, not arXiv).**

1. **rStar-Math (Microsoft, Jan 2025) is the most consequential missing citation.** Closest prior published system to the framework: small base models, MCTS-based self-training, process reward model from MCTS rollouts. Add as lead citation in §6.4 with 2 to 3 sentences contrasting against the framework's structured-state and pre-registered-methodology choices.

2. **The pre-registered "B4 strictly above B1 at matched FLOPs" hypothesis under-specifies operational parameters.** What is $N_{\mathrm{large}}$? What FLOP range counts? What statistical threshold defines "strictly above"? Without these, the hypothesis can be re-interpreted post-hoc. Tighten in §5.5 plus add a brief §5.8 "Statistical analysis plan" subsection.

3. **Two formal claims overstate their support.** Proposition 1 in §4.5 (chance-node convergence) understates its conditions; should be a Remark citing kocsis2006bandit as the operative template. The Goodhart claim "the signal IS the objective" (§4.6) and "the signal and the objective are the same object, so optimizing against the signal cannot introduce a Goodhart gap" (§6.3) overstates; should distinguish intrinsic Goodhart (proxy-vs-objective drift, ruled out by verifier-anchored signals) from extrinsic Goodhart (specification-vs-intent gap, still present, addressed by strengthening the verifier).

**Top five non-blocking issues.** DeepSeek-R1's no-MCTS success at scale is not explicitly contrasted (one paragraph in §6.2); contribution C2 (shared-encoder architecture) should be reframed or the KV-cache-reuse implementation novelty should be promoted into the contribution list; sanity-check ablations A1/A2/A3 need pre-registered numerical pass/fail thresholds; §4.5's opening paragraph (350 words) should be split into three for the chance-node bifurcation argument to land properly; eight missing precedent citations (Math-Shepherd, OmegaPRM, TS-LLM, ToT, ReST/ReSTEM, Brown et al. 2024, Gao et al. 2023, Skalse et al. 2022).

**Strengths to preserve.** RL-first framing throughout. Pre-registered baseline grid with three-metric compute accounting and explicit pre-commitment to falsifiable hypothesis (the paper's strongest methodological contribution). Verifier-vs-aesthetic Goodhart asymmetry (the paper's most original conceptual contribution). Structured-context schema with explicit Fact/Hypothesis/Subgoal/VerifierCall records (more careful than any surveyed prior LLM-MCTS system). InvokeVerifier as a first-class agent action (operationally novel). Notation discipline across all sections.

**Finding counts.** Critical: 0 | Major: 8 | Minor: 22 | Suggestions: many.

**Cross-verifications performed.** 5: M1 logic-checker against novelty-assessor; M2 logic-checker against novelty-assessor; M1 methodology-auditor against §5.5 manuscript text; rStar-Math omission against literature-context; Goodhart quote text verified against §4.6 (line 812 of method.tex) and §6.3 (line 132 of discussion.tex) of manuscript.

**Reports.**
- Unified report: `.papermill/reviews/2026-05-23/review.md` (2138 words; under the 3000-word cap)
- Specialist reports: `.papermill/reviews/2026-05-23/specialists/{logic-checker,novelty-assessor,methodology-auditor,prose-auditor,citation-verifier,format-validator,literature-context}.md`

# Decision Log

Key design decisions captured here for traceability. The full conversation-level reasoning lives in `design.md` §21.

- **Domain:** verifiable math/logic/algorithmic problems with non-LLM checkers. (Rationale: terminal reward is binary, no learned judge, no preference data.)
- **Base model framing:** small non-reasoning base LLMs (no Instruct, no RLHF, no o1/R1-style post-training). Keeps deliberation-loop contribution attributable.
- **Action space:** option B from the design dialogue (coarse verbs plus LLM-generated content). Seven verbs: DECOMPOSE, COMPUTE_SUBEXPR, FORM_HYPOTHESIS, CHECK_CASE, INVOKE_VERIFIER, BACKTRACK, COMMIT_ANSWER.
- **State representation:** structured shell with LLM-generated text in slots (option C from the state-representation discussion). MathState dataclass with `facts` vs `hypotheses` as the load-bearing distinction. Stable `id` fields on all entities; forward-compatible parent-pointer and children fields on `Hypothesis`, `Subgoal`, and `VerifierCall` for phase-2 recursive structural decomposition (productive recursion only; meta-criticism suppressed by reward signal plus depth limit).
- **Encoder choice:** shared LLM as both encoder and generator, with small adapter MLP between LLM hidden state and Q/policy heads. External sentence-transformers reserved for side jobs (memory retrieval, dedup hashing).
- **Q architecture:** single network with K scalar output heads (one per action), not K separate networks. Target Q network for Bellman target stability.
- **Reward structure:** terminal-only verifier-anchored reward in phase 1; intermediate verifier-anchored rewards as fallback if sparsity blocks progress; learned-aesthetic signals never as Bellman targets.
- **Evaluation:** pre-registered baseline grid B0 through B5 with three compute metrics (FLOPs, forward passes, wall-clock).

# Open Questions Carried into Implementation

1. Canonicalization of `MathState` for MCTS tree reuse (deferred to phase 2).
2. Activation of the recursive parent-pointer fields on `Hypothesis`, `Subgoal`, and `VerifierCall` (phase 2). Schema is forward-compatible; phase 1 leaves these at defaults.
3. Migration to option C (progressive widening with LLM-proposed actions) once fixed-verb training is stable.
4. LoRA on last LLM transformer layers if frozen-encoder Q-learning underfits.
5. Per-channel Q heads if reward decomposition becomes relevant (likely not for verifiable-math phase).
6. Persistent search trees across user interactions for anytime deliberation.
7. Migration from terminal-only to verifier-anchored intermediate rewards if sample efficiency demands it.

# Next Action

Draft `paper/sections/method.tex`. The core section anchors everything else; in particular, the architecture subsection needs:

- A figure (`paper/figures/architecture.pdf` or TikZ inline) showing the data flow from `MathState` through serialization, frozen LLM forward, last-token pool, adapter MLP, and the three heads (Q, policy, verb-instantiator).
- Concrete prose for the MathState definition block and the seven-verb action-space subsection.
- The reward principle subsection (verifier-anchored vs aesthetic) is short but load-bearing; it should be polished early because the discussion-section position subsection extends it.

After `method.tex` is roughed in, draft `introduction.tex` so contribution claims align with what the method actually does.

# History

- **2026-05-22:** Repository scaffolded. `design.md` extended with §21 (Concrete Specialization, ~267 lines) capturing decisions from the design conversation. Paper scaffold created at `paper/` with `main.tex`, `references.bib`, eight stub section files, `.gitignore`, and `figures/` (empty). LaTeX scaffold verified to compile. Papermill state file initialized.
- **2026-05-22:** Schema upgrade. Added stable `id` fields to `Fact`, `Subgoal`, `Hypothesis`, and `VerifierCall`. Added forward-compatible parent-pointer and children fields to `Hypothesis`, `Subgoal`, and `VerifierCall` to enable phase-2 structural-decomposition recursion without requiring a future schema migration. Distinguished productive recursion (structural decomposition; pays off in verifier-anchored reward) from unproductive recursion (meta-criticism; rarely productive on verifiable math). Updated `design.md` §21.3, §21.4 (id-based action targeting), and §21.9 (open questions).
- **2026-05-23:** Method drafting batch 1. Subagents drafted §4.1 Overview and §4.5-§4.8 (MCTS, reward, training, inference) in `method.tex`. Architecture figure built in TikZ at `figures/architecture.tex` (three-column layout, dashed arrow for KV-cache reuse, dimension annotations). PDF reached 18 pages. Added `\usepackage{tikz}` to `main.tex`. Two underfull hbox warnings in abstract remained from prior draft, unrelated to figure work.
- **2026-05-23:** RL-first reframe. Title changed from "MCTS Expert Iteration over Verifiable Context MDPs" to "Verifier-Anchored Reinforcement Learning for Small Language Model Deliberation" to align with `design.md` §1's RL-first directive (the MCTS-first regression was caught by the user, fixed by five parallel subagents in one batch). Abstract and introduction rewritten to lead with approximate $Q$-learning over a context MDP, with MCTS reframed as a sample-efficient policy improvement operator and a test-time-compute knob. Background restructured per design.md §1.6 ordering: MDP, Bellman, $Q$-learning (new subsection), approximate $Q$-learning (new subsection), exploration (new subsection), MCTS as sample-efficient policy improvement (reframed), expert iteration as joint $Q$/policy distillation (reframed), LLM-as-context-MDP. Method section reframed throughout: §4.5 retitled from "MCTS with Chance-Node Bifurcation" to "MCTS as a Policy Improvement Operator". PDF now 24 pages.
- **2026-05-23:** End-to-end drafting completed. Three parallel subagents drafted experimental plan (~5 pages, baseline grid table preserved, pre-registered hypothesis stated as "on GSM8K and MATH levels 1 through 3, the B4 curve lies strictly above the B1 curve at matched total inference FLOPs"), related work (~2 pages, five subsections positioning the framework against AlphaZero lineage, verifier-anchored LLM reasoning, test-time-compute scaling, MCTS-for-LLM work, and self-improvement systems), and discussion + conclusion (~3 + 0.5 pages, with sec:disc-verifier as the load-bearing position subsection on verifier-vs-aesthetic signal asymmetry). PDF now 35 pages. Stage transitioned from drafting to revising.
- **2026-05-23:** Drafted the remaining method subsections (4.1 overview, 4.5 MCTS with chance-node bifurcation, 4.6 reward structure, 4.7 training loop, 4.8 inference modes). MCTS subsection formalizes per-edge backup and includes a chance-node convergence proposition. Reward subsection states the verifier-anchored-only principle with the Goodhart argument. Training subsection adds loss equations and replay-buffer details. Inference subsection covers fast and search modes with persistent-tree forward reference. Uncommented the architecture figure block in 4.4 and added Figure~\ref{fig:architecture} reference in the opening paragraph. Full compile sequence (pdflatex/bibtex/pdflatex/pdflatex) clean at 18 pages.
- **2026-05-23:** Multi-agent editorial review completed (seven specialists: logic-checker, methodology-auditor, novelty-assessor, prose-auditor, citation-verifier, format-validator, plus merged literature-scout output). Recommendation: minor revision for arXiv preprint; major revision before venue submission. Three blocking issues for venue submission: (1) rStar-Math omission, (2) pre-registered hypothesis under-specification, (3) Proposition 1 overstatement and Goodhart claim overstatement. Five non-blocking issues including DeepSeek-R1 contrast, C2 contribution reframing, ablation thresholds, §4.5 paragraph split, and missing precedent citations. Strengths to preserve: RL-first framing, pre-registered grid, verifier-vs-aesthetic asymmetry, structured-context schema, InvokeVerifier as first-class action. Unified report at `.papermill/reviews/2026-05-23/review.md`; specialist reports under `.papermill/reviews/2026-05-23/specialists/`.
