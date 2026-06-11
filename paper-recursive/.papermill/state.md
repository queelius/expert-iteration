---
title: "Bounded Structural Recursion in Verifier-Anchored Reinforcement Learning"
subtitle: "When and How Deliberation Should Decompose Itself"
author: "Alexander Towell"
affiliation: "Department of Computer Science, Southern Illinois University Edwardsville"
email: "atowell@siue.edu"
orcid: "0000-0001-6443-9897"
format: "latex-article"
manuscript: "paper-recursive/main.tex"
bibliography: "paper-recursive/references.bib"
sections_dir: "paper-recursive/sections/"
figures_dir: "paper-recursive/figures/"
stage: "addressing review"
paper_type: "methodology + position paper (companion)"
target_venue: "arXiv preprint (companion submission, same or adjacent venue as framework paper)"
created: "2026-05-22"
last_updated: "2026-05-27"
companion_of: "paper/"
---

# Project Status

**Stage:** revising. All sections substantively drafted end-to-end: abstract, introduction (anchored in the Pythagorean-theorem decomposition example), background (with hierarchical-RL and tree-encoder subsections), method (productive-vs-unproductive distinction, depth-limit mechanism, action targeting, serialization, MCTS interaction, canonicalization note), experimental plan (pre-registered depth-limit ablation), related work (six subsections: hierarchical RL, tree encoders, MCTS-for-LLM, formal-proof systems, self-improvement, position summary), discussion (limitations, extensions, when-recursion-pays-off, threats, implicit-vs-explicit hierarchy position), and conclusion. PDF compiles to 23 pages.

**Paper type:** methodology and position companion to the framework paper at `paper/`. Develops the recursive structural decomposition design choice in depth. Pre-registers a depth-limit ablation as the central empirical contribution; no results yet.

**Target venue:** arXiv preprint, with the framework paper as a companion submission. Specific venue selected after both papers reach complete first drafts; candidates align with the framework paper's (NeurIPS, ICML, ICLR, or a reasoning/test-time-compute workshop).

# Central Claim (Thesis)

Structural recursion in MCTS expert iteration over verifiable context MDPs has two qualitatively different modes (productive structural decomposition versus unproductive meta-criticism), and a bounded per-hypothesis tree depth limit ($\ell_{\mathrm{tree}} \in \{3, 4\}$) is the structural prior that selects the productive mode without foreclosing the unproductive one's structural possibility. The seven-verb action set of the framework paper is sufficient for arbitrary recursive depth via identifier-based targeting; no new categorical actions are required. Flat token serialization with depth markers lets a frozen LLM resolve the implicit tree through native causal self-attention, with no graph neural network or tree-LSTM.

The paper rests on four substantive arguments:

1. Productivity is defined relative to a verifier and a depth budget. A recursive subtree is productive when terminal verifier reward can enter at the leaves and propagate up via MCTS; unproductive otherwise.
2. The depth limit is enforced as a hard mask on the three recursive verbs (`DECOMPOSE`, `CHECK_CASE`, `FORM_HYPOTHESIS` when attached to a parent), not as a soft penalty. Soft penalties violate the framework paper's verifier-anchored-reward-only principle.
3. Action targeting through stable identifiers means `DECOMPOSE(hyp_id)` works regardless of where `hyp_id` sits in the implicit tree. The branching factor at decision nodes stays at seven; only the state space grows tree-structured.
4. Frozen-LLM serialization with depth markers (`[HYPOTHESIS id=hyp_2 parent=hyp_1 depth=2]`) gives causal self-attention the positional anchors and identifier resolution it needs to handle the implicit tree without explicit tree encoders.

# Outline

Sections are stubbed in `paper-recursive/sections/`:

1. `abstract.tex`: near-complete prose draft (one paragraph, ~250 words).
2. `introduction.tex`: motivation, position, five-contribution list, relation to the framework paper, paper structure. Mix of drafted prose and TODO stubs.
3. `background.tex`: notation recap from the framework paper, structured context MDP and recursive schema recap, hierarchical RL background (options, MAXQ, feudal networks), tree-structured encoder background and why we avoid them. Drafted prose.
4. `method.tex`: substantively drafted. Subsections for overview, productive-vs-unproductive recursion (with formal Definition 1), depth limit as structural prior (with Definition 2 and amended availability table), action targeting at arbitrary depth, encoding recursive state without recursive encoders, MCTS interaction, canonicalization and idempotence note. About 3 to 4 pages of prose.
5. `experimental_plan.tex`: setup inherited from framework paper (benchmarks, base models, architecture); the depth-limit ablation grid (R0, R3, R-infty); three pre-registered hypotheses (H1 expressivity, H2 rumination, H3 depth distribution); secondary measurements (trajectory composition, verifier-call frequency by depth, MCTS budget allocation); threats to validity; reporting discipline.
6. `related_work.tex`: hierarchical RL positioning (options, MAXQ, feudal), recursive neural networks and tree encoders, MCTS for LLM reasoning (including Tree of Thoughts and AlphaProof), self-improvement and compositional generalization, position summary. Mostly TODO stubs with prose hooks.
7. `discussion.tex`: limitations, extensions, when recursion pays off, threats specific to the recursion claim, implicit versus explicit hierarchy as a methodological note. TODO stubs.
8. `conclusion.tex`: brief stub (~0.5 page target).

# Files and Layout

| Artifact | Path |
|----------|------|
| Manuscript root | `paper-recursive/main.tex` |
| Bibliography | `paper-recursive/references.bib` |
| Section sources | `paper-recursive/sections/*.tex` |
| Figures | `paper-recursive/figures/` (empty) |
| LaTeX gitignore | `paper-recursive/.gitignore` |
| Companion paper | `paper/` (sibling repo, the framework paper) |
| Authoritative design | `design.md` (parent repo, shared with framework paper) |
| Project guidance | `CLAUDE.md` (parent repo) |

# References

The bibliography in `paper-recursive/references.bib` inherits relevant entries from the framework paper (AlphaZero, expert iteration, MCTS foundations, base models, verifier-anchored LLM reasoning, LoRA, target networks) and adds:

- **Hierarchical RL:** Sutton, Precup, Singh 1999 (options framework); Dietterich 2000 (MAXQ); Vezhnevets et al. 2017 (feudal networks); Bacon, Harb, Precup 2017 (option-critic); Nachum et al. 2018 (data-efficient hierarchical RL).
- **Recursive neural networks and tree encoders:** Tai, Socher, Manning 2015 (tree-LSTM); Socher et al. 2013 (recursive sentiment models); Velickovic et al. 2018 (graph attention networks); Lee et al. 2019 (set transformer).
- **LLM reasoning and tree search:** Yao et al. 2023 (tree of thoughts); Yang et al. 2023 (LeanDojo) for the formal-proof connection.

Entries marked `TODO` in the bib file flag citations that need verification before submission.

# Review History


- **2026-05-27:** Multi-agent editorial review pass completed. Report at `paper-recursive/.papermill/reviews/2026-05-27/review.md`. Recommendation: **minor-revision**. Three venue-blocking findings: (B1) missing rStar-Math citation and contrast in §5.3 (framework paper already cites; cross-paper coherence gap); (B2) missing HyperTree Proof Search (Lample et al., 2022) in §5.4; (B3) statistical analysis plan absent in §experimental-plan (no Holm-Bonferroni / paired-bootstrap / effect-size thresholds). Plus seven major findings (unwritten threats-to-validity subsection, Definition 1 quantifier ambiguity, H2 mechanism falsifiability, single-value depth=3 grid, six uncited bib entries, hard-mask conflation, TODO §4.7 canonicalization subsection). Strengths: identifier-based action targeting is a clean novel contribution; Pythagorean motivating example anchors well; implicit-vs-explicit hierarchy position is the paper's strongest argumentative passage; build is clean; 23-page length is appropriate for a companion. Stage transitioned from "revising" to "addressing review".
- **2026-05-27 (resolved same day):** All three blockers and three cheapest majors closed. B1 (rStar-Math) and B2 (HyperTree Proof Search): bib entries added; contrast paragraphs added in §5.3 and §5.4. B3 (statistical analysis plan): new `\subsection{Statistical Analysis Plan}` mirrors the framework paper's Holm-Bonferroni-corrected paired-bootstrap protocol (R=10000 resamples, pre-registered budgets B in {100, 300, 1000}, family-wise alpha=0.05); H1/H2 hypotheses tightened with one-percentage-point effect-size thresholds; new sub-hypothesis H2a separates mechanism (depth-tail signature) from outcome (pass@1 inequality). M1: four threats-to-validity paragraphs drafted (depth-limit-does-nothing, training-instability vs rumination, single-depth-3-risk, verifier-strength). M2: Definition 1 disambiguated into syntactic vs operational productivity with consistent use through method, discussion, and experimental plan. M5: five previously-uncited entries cited (browne in background, wei/snell/openai in §5.3, deepseekr1 in §5.5); unused hu2022lora removed. Remaining open: M3 (canonicalization stub at method.tex §4.7 still TODO), M4 partially addressed via H2a + depth sweep, M6 (hard-mask conflation of simulation vs policy depth), M7 (TODO §4.7 subsection), plus 9 minor issues. Build clean: 27 pages, all labels and citations resolve, banned-phrase check clear, full pdflatex -> bibtex -> pdflatex -> pdflatex cycle verified.



# Decision Log

Captured here for traceability. The substantive design reasoning sits in `paper-recursive/sections/method.tex` and (for the parts inherited from the framework paper) in `design.md` Section 21.

- **Scope:** companion to the framework paper at `paper/`. Single design dimension developed in depth: structural recursion via implicit-tree state.
- **Title:** "Bounded Structural Recursion in MCTS Expert Iteration: When and How Deliberation Should Decompose Itself". Foregrounds the depth-limit as the load-bearing mechanism; subtitle emphasizes the operational question.
- **Definition of productivity:** Definition 1 in the method section makes productivity a property of the trajectory under a verifier, not of the syntactic recursion structure. This is deliberate: productivity is what reward signal density determines, not what the schema permits.
- **Depth-limit enforcement:** hard mask on the three recursive verbs (`DECOMPOSE`, `CHECK_CASE`, and `FORM_HYPOTHESIS` when attached to a parent). Soft penalties were considered and rejected on framework-paper-consistency grounds (the framework paper restricts Bellman targets to verifier-anchored signals).
- **Ablation grid:** three settings (R0, R3, R-infty). R0 reproduces framework paper phase 1; R3 is the predicted productive regime; R-infty is the predicted unproductive regime. Pre-registered hypotheses commit to specific outcomes.
- **Encoding:** flat serialization with depth markers, no tree-LSTM or GNN. Argument is qualitative; the present paper does not test against an explicit-tree-encoder baseline. That comparison is a follow-up.
- **Cross-paper convention:** notation is shared via custom commands defined identically in `paper/main.tex` and `paper-recursive/main.tex`. The recursive paper adds `\dmax` and `\depthlim` as new shorthands.
- **Cross-references to framework paper:** by description rather than by fixed title or arXiv id (neither has been published). Once both papers have stable identifiers, replace descriptive references with formal citations.

# Open Questions Carried into Drafting

1. Whether to include an explicit tree-LSTM-as-encoder baseline in the ablation. Currently out of scope; the depth-limit comparison is the central question.
2. Whether the depth-limit value should be allowed to vary by problem difficulty (adaptive depth limit). Discussed as an extension; not exercised in the headline ablation.
3. Whether a verifier composition argument (parent claim's verification derivable from children's) belongs in this paper or in a separate verifier-design companion. Currently flagged as a follow-up direction in the discussion section.
4. Whether the canonicalization question (made harder by recursion) deserves its own short section. Currently a TODO subsection placeholder in the method section.
5. How to handle the citation of the framework paper itself once it has an arXiv identifier. Currently referenced descriptively.

# Next Action

Draft the introduction (`paper-recursive/sections/introduction.tex`) into full prose. The contribution list is in place but the motivation, position, and relation-to-framework paragraphs are stubs. The method section's prose can guide the introduction's voice and emphasis.

After the introduction, draft the related-work section's positioning paragraphs (currently TODO stubs with prose hooks); the hierarchical-RL background is set up in `background.tex`, so the related-work job is positioning rather than survey.

The discussion section's limitations and extensions can be drafted after a draft of the experimental-plan section settles, since several discussion items reference the ablation grid.

# History

- **2026-05-22:** Repository scaffolded as a companion to the framework paper at `paper/`. Directory `paper-recursive/` created with `main.tex` (shared notation commands from framework paper plus `\dmax` and `\depthlim`), `references.bib` (framework-paper subset plus hierarchical-RL and tree-encoder additions), eight section files (`abstract.tex`, `introduction.tex`, `background.tex`, `method.tex`, `experimental_plan.tex`, `related_work.tex`, `discussion.tex`, `conclusion.tex`), `.gitignore`, and `figures/` (empty). Method section drafted substantively (productive-vs-unproductive distinction; depth-limit mechanism; action targeting; serialization treatment; MCTS interaction). Abstract is a complete one-paragraph draft. Background and experimental plan have drafted prose with some TODO stubs. Related work and discussion are TODO outlines with prose hooks. LaTeX scaffold verified to compile.
- **2026-05-23:** Introduction polished from paragraph stubs to full prose (~1170 prose words, ~1.5 pages of LaTeX). Motivation anchored in a Pythagorean-theorem decomposition example: $a^2 + b^2 = c^2$ splits into a right-angle geometric sub-claim and an arithmetic sub-claim, each verifier-checkable independently. The five-contribution list refined and mapped explicitly to the method section's six subsections. Full pdflatex/bibtex/pdflatex/pdflatex build cycle verified clean; total document is 15 pages.
- **2026-05-23:** RL-first reframe to align with framework paper. Title changed from "Bounded Structural Recursion in MCTS Expert Iteration" to "Bounded Structural Recursion in Verifier-Anchored Reinforcement Learning" (subtitle preserved). Abstract and introduction rewritten to lead with the framework paper's approximate $Q$-learning system rather than with MCTS expert iteration; the Pythagorean example was preserved. Method and background reframed throughout: surrounding language now treats the framework as approximate $Q$-learning with MCTS as policy improvement operator; the recursion content (Definition 1 productive/unproductive trajectories, Definition 2 depth limit, availability table, serialization listing, seven-verb scheme, MCTS-interaction note) preserved exactly. Method net length change ~9% within the under-10% target. PDF now 17 pages.
- **2026-05-23:** End-to-end drafting completed. Two parallel subagents drafted related work (~3 pages, six subsections: hierarchical RL with implicit-via-parent-pointer-references contrast, recursive neural networks and tree encoders, MCTS for LLM reasoning, recursive decomposition in formal proof systems, self-improvement and compositional generalization, position summary) and discussion + conclusion (~3.5 + 0.4 pages, with the implicit-vs-explicit hierarchy methodological note as the position subsection). PDF now 23 pages. Stage transitioned from drafting to revising.
- **2026-05-27:** Multi-agent papermill review pass completed and unified report written to `.papermill/reviews/2026-05-27/review.md` (3 blockers, 7 majors, 9 minors; recommendation minor-revision). Three blockers and three cheapest majors immediately addressed: (B1) added `guan2025rstarmath` to bibliography and a contrast paragraph in §5.3; (B2) added `lample2022hypertree` and a paragraph in §5.4 contrasting flat-state parent pointers against explicit hypertree DAG; (B3) added a new `\subsection{Statistical Analysis Plan}` mirroring the framework paper's Holm-Bonferroni-corrected paired-bootstrap protocol with R=10000 resamples and pre-registered inference-compute budgets B in {100, 300, 1000}; (M1) drafted four threats-to-validity paragraphs (depth-limit-does-nothing, training-instability vs rumination, single-depth-3-risk, verifier-strength); (M2) disambiguated Definition 1 into syntactic-vs-operational productivity with consistent use through method, discussion, and experimental plan; (M5) cited five of the six previously-uncited entries (`browne2012survey` in background, `wei2022cot` and `snell2024scaling` and `openai2024o1` in §5.3, `guo2025deepseekr1` in §5.5) and removed the unused `hu2022lora` entry. Added one new sub-hypothesis H2a that makes H2's mechanism attribution falsifiable independent of its outcome. Full pdflatex -> bibtex -> pdflatex -> pdflatex cycle clean, PDF now 27 pages, all labels and citations resolve, banned-phrase check clear. Stage transitioned from revising to addressing-review (remaining: 4 majors, 9 minors).
