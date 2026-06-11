# Multi-Agent Review Report

**Date**: 2026-05-27
**Paper**: "Bounded Structural Recursion in Verifier-Anchored Reinforcement Learning: When and How Deliberation Should Decompose Itself" (Towell, 23 pages)
**Recommendation**: **minor-revision**

## Executive Summary (top 5 findings)

1. **[BLOCKER] rStar-Math citation absent (cross-paper inconsistency).** The framework paper cites and contrasts against rStar-Math (Guan et al., 2025) after its 2026-05-23 review pass; the companion does not. The companion's distinguishing claim (structural recursion at the hypothesis level) is precisely what rStar-Math (token-level MCTS) does not do. Remediation: add the citation to §5.3 and contrast in 2-3 sentences.

2. **[BLOCKER] HyperTree Proof Search not cited.** Lample et al. (2022) is the closest prior work for bounded-depth recursive subgoal trees. Without engagement, the central depth-limit mechanism looks under-grounded. Remediation: add citation and one-paragraph contrast in §5.4 (Formal Proof Systems).

3. **[BLOCKER] Statistical analysis plan missing.** H1, H2, H3 commit to qualitative directional claims without effect-size thresholds, paired-bootstrap procedure, or Holm-Bonferroni correction. The framework paper has this; the companion does not. Remediation: add a statistical-analysis-plan subsection to §experimental-plan mirroring the framework paper's.

4. **[MAJOR] Threats-to-validity subsection is unwritten.** §experimental-plan §4 (Threats to Validity) is entirely TODO comments. The content is named (compute leaks, verifier coverage, contamination, seed variance, plus depth-specific threats), but the prose is missing. Remediation: draft the section using the TODO bullets as scaffolding.

5. **[MAJOR] Definition 1 quantifier ambiguity.** Definition 1 uses "there exists a finite trajectory" without specifying whether productivity is syntactic (some trajectory reaches a leaf) or operational (the trained policy reaches a leaf with positive probability). The discussion implicitly uses both meanings. Remediation: introduce *syntactically productive* vs *operationally productive* distinction or strengthen Definition 1 with a policy-conditioned quantifier.

**Finding counts**: Blocker: 3 | Major: 7 | Minor: 9 | Suggestions: 4

## Summary

**Overall assessment**: This is a tightly-scoped methodology and position companion to a previously-reviewed framework paper. The writing is strong, the conceptual framing (productive vs unproductive recursion as verifier-reach) is original, and the most distinctive technical contribution (identifier-based action targeting at fixed verb cardinality) is well-developed. The principal weaknesses concentrate in two domains: (i) the literature engagement has not been updated to match the framework paper's 2026-05-23 revisions (rStar-Math is the headline gap), and (ii) the experimental plan claims pre-registration but lacks the statistical-analysis specificity that pre-registration requires.

**Strengths**:
1. Identifier-based action targeting at fixed verb cardinality is a clean architectural insight; the seven-verb action set extends to arbitrary recursion depth without progressive widening (novelty-assessor, logic-checker).
2. Pythagorean motivating example anchors the abstract claim well (prose-auditor).
3. Implicit-vs-explicit hierarchy methodological position in §discussion is the paper's strongest argumentative passage (prose-auditor).
4. Build is clean; no unresolved labels or citations (format-validator).
5. Length (23 pages) is appropriate for a companion to a 37-page framework paper; the companion does not redo framework content (format-validator, prose-auditor).
6. Hard-mask vs soft-penalty argument is principled on verifier-anchored-reward grounds (logic-checker).

**Weaknesses**:
1. rStar-Math (the most direct contemporary contender) is not cited (novelty-assessor, citation-verifier, literature-context).
2. HyperTree Proof Search (the closest analog for bounded-depth subgoal trees) is not cited (novelty-assessor, citation-verifier, literature-context).
3. Statistical analysis plan absent; pre-registration claims are weakened (methodology-auditor).
4. Threats-to-validity subsection is unwritten (methodology-auditor).
5. Definition 1's quantifier is ambiguous; "productive" used in multiple senses (logic-checker).
6. Six uncited bibliography entries inflate the bibliography (citation-verifier).
7. Seven TODO-marked bib entries (most can be cleared after metadata verification) (citation-verifier).

## Critical Issues (Blockers)

### B1: rStar-Math citation absent (cross-paper inconsistency)

- **Sources**: literature-context, novelty-assessor (N1), citation-verifier (C1)
- **Location**: §5.3 (related_work.tex MCTS for LLM Reasoning), bibliography
- **Quoted text** (related_work.tex lines 128-141): The §5.3 subsection lists Feng et al. (2023), Xie et al. (2024), and Tree of Thoughts (Yao et al. 2023) as the MCTS-for-LLM landscape. rStar-Math is absent.
- **Cross-paper check verified**: framework paper's `references.bib` line 244 has `@article{guan2025rstarmath, ...}` and `related_work.tex` lines 124 and 157 cite and contrast against it. The companion has neither the citation nor the contrast.
- **Problem**: rStar-Math (Guan et al., 2025) is the closest contemporary contender: small-LLM math reasoning, MCTS-driven, verifier-anchored. The companion's distinguishing claim is structural recursion at the hypothesis level, which is precisely what rStar-Math (token-level MCTS) does not exercise. A reviewer following recent literature will flag this immediately.
- **Suggestion**: Add to `references.bib`:
  ```bibtex
  @article{guan2025rstarmath,
    title={{rStar-Math}: Small {LLMs} Can Master Math Reasoning with Self-Evolved Deep Thinking},
    author={Guan, Xinyu and Zhang, Li Lyna and Liu, Yifei and Shang, Ning and Sun, Youran and Zhu, Yi and Yang, Fan and Yang, Mao},
    journal={arXiv preprint arXiv:2501.04519},
    year={2025}
  }
  ```
  Add 2-3 sentences to §5.3 (MCTS for LLM Reasoning) contrasting hypothesis-level structural recursion against rStar-Math's token-level MCTS.
- **Cross-verified**: Yes, against framework paper's bib and related_work.tex (confirmed citation exists there).

### B2: HyperTree Proof Search not cited

- **Sources**: literature-context, novelty-assessor (N2), citation-verifier (C2)
- **Location**: §5.4 (related_work.tex Recursive Decomposition in Formal Proof Systems), bibliography
- **Problem**: HyperTree Proof Search (Lample et al., 2022, NeurIPS) is the closest prior work for the bounded-depth recursive subgoal tree mechanism. The companion's depth-limit prior has direct precedent in HyperTree's bounded-depth expansion. The companion's distinction (implicit hierarchy via flat-state parent pointers; HyperTree uses explicit hypertree DAG) is a real architectural difference worth surfacing.
- **Suggestion**: Add to `references.bib`:
  ```bibtex
  @inproceedings{lample2022hypertree,
    title={{HyperTree} Proof Search for Neural Theorem Proving},
    author={Lample, Guillaume and Lacroix, Timoth{\'e}e and Lachaux, Marie-Anne and Rodriguez, Aurelien and Hayat, Amaury and Lavril, Thibaut and Ebner, Gabriel and Martinet, Xavier},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2022}
  }
  ```
  Cite in §5.4 alongside AlphaProof and LeanDojo. Add 2-3 sentences contrasting bounded-depth hypertree expansion (Lample) against bounded-depth flat-state parent pointers (this paper).
- **Cross-verified**: Yes; HyperTree is widely cited in 2022-2025 formal-proof work that the §5.4 subsection should engage with.

### B3: Statistical analysis plan missing

- **Source**: methodology-auditor (M1)
- **Location**: §experimental-plan §2 (Depth-Limit Ablation) and §5 (Reporting Discipline)
- **Quoted text** (experimental_plan.tex lines 78-95): H1, H2, H3 commit to qualitative claims ("R3 achieves higher pass@1 than R0", etc.) without specifying effect-size thresholds, statistical tests, or multiple-comparison correction.
- **Cross-paper check**: The framework paper's review resolved this with a Holm-Bonferroni-corrected paired-bootstrap plan. The companion lacks the analog.
- **Problem**: A pre-registered paper without a statistical analysis plan is pre-registered in name only. The hypotheses are "weakly falsifiable" (any positive difference passes H1, any negative falsifies it), but the strength of the falsification is undefined.
- **Suggestion**: Add a "Statistical analysis plan" paragraph to §exp-reporting:
  - Pre-register the effect-size threshold (e.g., R3 outperforms R0 on MATH pass@1 by >= 1 percentage point, averaged across >= 3 seeds).
  - Pre-register the test: paired bootstrap on per-problem outcomes (10,000 resamples).
  - Pre-register Holm-Bonferroni correction across H1, H2, H3.
  - Pre-register the inference-compute budgets (B = 100, 300, 1000 simulations).
- **Cross-verified**: Yes, against the framework paper's experimental plan (which now has this content).

## Major Issues

### M1: Threats-to-validity subsection is TODO comments

- **Source**: methodology-auditor (M3)
- **Location**: §experimental-plan §4 (sec:exp-threats), experimental_plan.tex lines 127-144
- **Problem**: The entire subsection is unwritten. The TODO comment names four threats (inherited threats; H3 catches no-decompose-anyway; training instability from unbounded states; single-value depth=3 risk), but no prose has been drafted. The discussion section has partial overlap in §6.4, but the experimental plan needs its own dedicated threats subsection per pre-registration convention.
- **Suggestion**: Draft 4-6 paragraphs covering the four named threats. The TODO content is good scaffolding.

### M2: Definition 1 quantifier ambiguity

- **Source**: logic-checker (L1)
- **Location**: §4.2 method.tex lines 101-108
- **Quoted text**: "A recursive subtree rooted at hypothesis $h$ is *productive* if there exists a finite trajectory from $h$ to a leaf that terminates in a successful \textsc{InvokeVerifier} call on a descendant hypothesis or in a \textsc{CommitAnswer} that the terminal verifier accepts."
- **Problem**: The existential quantifier "there exists a finite trajectory" is satisfied by any subtree with a verifier-reachable descendant, including subtrees the trained policy never visits. The paper uses "productive" elsewhere in an operational sense (trajectories the policy actually realizes).
- **Suggestion**: Either (a) strengthen Definition 1 to specify the policy: "is productive under policy $\pi$ if, with positive probability under the trajectory distribution induced by $\pi$, ..." or (b) introduce two notions ("syntactically productive" as written; "operationally productive" under the policy) and use them consistently.
- **Cross-verified**: Read methodology-auditor's report; the operational sense is exactly what H2 (rumination) is asking the policy to do. The distinction matters.

### M3: H2's mechanism claim not falsifiable

- **Source**: methodology-auditor (M4)
- **Location**: §experimental-plan §2 (sec:exp-ablation), experimental_plan.tex lines 85-88
- **Problem**: H2 claims R-infty underperforms R3 "due to simulation-budget dilution along unproductive recursive paths." The causal attribution is what the secondary measurements should support, but H2 as written conflates the outcome (lower pass@1) with the mechanism (budget dilution). If R-infty underperforms for another reason (training instability, gradient pathologies), H2 is "confirmed" but the mechanism claim is unsupported.
- **Suggestion**: Add a sub-hypothesis H2a making the mechanism claim falsifiable: "Conditional on R-infty underperforming R3 by at least the threshold, MCTS budget allocation in R-infty shows >= X% of visits at depth >= 4 and verifier-call frequency at depth >= 4 below Y per trajectory."

### M4: Ablation grid has one middle value (depth=3)

- **Source**: methodology-auditor (M2)
- **Location**: §experimental-plan §2 (sec:exp-ablation), table tab:depth-ablation
- **Problem**: The headline grid is three settings (0, 3, infinity). If the productive regime is at depth 2 or depth 4, the headline grid could confirm the central claim by accident or miss it entirely. The TODO at experimental_plan.tex line 143 mentions a follow-up sweep across {1, 2, 3, 4, 5}, but that's currently a comment, not part of the pre-registration.
- **Suggestion**: Promote the depth sweep from TODO to pre-registered secondary experiment. Specify the held-out subset size and selection method.

### M5: Six uncited bibliography entries

- **Source**: citation-verifier (C3)
- **Location**: `references.bib`
- **Problem**: `browne2012survey`, `guo2025deepseekr1`, `hu2022lora`, `openai2024o1`, `snell2024scaling`, `wei2022cot` are present in the bib but not cited in any section. Inflates the bibliography without supporting argument.
- **Suggestion**: Either cite (CoT in introduction motivation; o1 and snell as test-time-compute exemplars; browne in §3 background MCTS recap; deepseekr1 in §5.5 self-improvement) or remove. LoRA can be removed since the companion does not exercise adapter mechanics.

### M6: Hard-mask vs soft-penalty conflation of simulation depth and policy depth

- **Source**: logic-checker (L3)
- **Location**: §4.3 method.tex lines 198-203
- **Quoted text**: "a soft penalty makes the policy learnable but the state space exploration unbounded: an MCTS simulation could still descend arbitrarily deep, paying the penalty at each step but accruing no other information."
- **Problem**: A soft penalty *with discounting* would eventually make further descent negative-expected-return, bounding the optimal policy's depth. The "MCTS simulation could still descend" is true of unbounded *simulation*, not unbounded *policy depth*. The two are conflated.
- **Suggestion**: Reframe to distinguish behavioral disincentive (policy under modified reward) from exploration bound (MCTS rollout). The hard mask's distinguishing property is that it bounds simulation depth at the MDP level, before MCTS has a chance to descend.

### M7: TODO subsection in method.tex line 444 (canonicalization)

- **Source**: methodology-auditor, prose-auditor (cross-derived)
- **Location**: §4.7 method.tex lines 444-457
- **Problem**: The entire §4.7 (sec:method-canon) subsection is a TODO comment block. The introduction (line 150) and method-overview (line 50) both reference §4.7 as if it were drafted. Either draft the subsection (2-3 paragraphs) or remove the references.
- **Suggestion**: Either draft the 2-3 paragraph subsection (the TODO content is good scaffolding: recursion makes canonicalization harder; framework paper's deferral is justified; semantically equivalent states may produce distinct Bellman targets weakening generalization), or remove the subsection and adjust forward references.

## Minor Issues

### m1: Serialization example depth inconsistency

- **Source**: logic-checker (L2)
- **Location**: method.tex lines 304-309
- **Problem**: The example shows `hyp_1` at depth=1 with parent=sg_2 (depth=1), but Definition 2 (length of longest parent-pointer chain) gives depth 2 if chains cross entity types. Either Definition 2 should clarify per-entity-type chains, or the example should be corrected.
- **Suggestion**: Clarify in Definition 2 that depth is computed per-entity-type (hypothesis depth follows parent_hypothesis only), or change the example so depth values are consistent with cross-type chains.

### m2: "Productive" used in three senses

- **Source**: logic-checker (L4)
- **Location**: Multiple sections (Definition 1, introduction lines 13-21, H1 in experimental_plan)
- **Suggestion**: One-sentence clarification after Definition 1 distinguishing definitional/syntactic, operational, and regime senses.

### m3: MCTS budget allocation measurement under-specified

- **Source**: methodology-auditor (M5)
- **Location**: §experimental-plan §3 (sec:exp-secondary), line 120-125
- **Suggestion**: Specify whether "depth distribution of visited nodes" means search-tree depth (MCTS depth from root) or state-space depth (implicit-tree depth). Both are interesting; the measurement plan should distinguish them.

### m4: H3 "harder problem tail" undefined

- **Source**: methodology-auditor (M6)
- **Location**: experimental_plan.tex line 91-92
- **Suggestion**: Quantify: e.g., "problems in the top quartile by MATH difficulty rating."

### m5: Three-seed reporting may not match framework paper

- **Source**: methodology-auditor (M7)
- **Suggestion**: Verify framework paper's reporting discipline post-revision and match.

### m6: Implicit-vs-explicit hierarchy position buried in discussion

- **Source**: prose-auditor (P1)
- **Suggestion**: Surface the core argument (one paragraph) in the introduction's position section or in §5.6 (Position Summary), in addition to its current home in §6.5.

### m7: "Leaf" used in Definition 1 without prior definition

- **Source**: prose-auditor (P2)
- **Suggestion**: One-line clarification: "by 'leaf' we mean an entity with no children entries in the implicit tree."

### m8: Page count (23 pages) exceeds standard ML venue limits

- **Source**: format-validator (F1)
- **Suggestion**: For arXiv submission no action needed. For venue submission, plan for substantial cutting (target 8-10 pages).

### m9: Seven TODO-marked bib entries need metadata verification

- **Source**: citation-verifier (C4)
- **Suggestion**: Verify and remove TODO markers for bacon2017option, velickovic2018gat, yang2023leandojo, yang2024qwen25, dubey2024llama3, allal2024smollm2 (all metadata appears correct per spot-check). Update or accept blog-post citation for deepmind2024alphaproof and openai2024o1.

## Suggestions

1. **Add a tree-diagram figure** illustrating the implicit hypothesis tree (parent pointers) corresponding to the serialization template. The serialization template lists the tree linearly; a small visual would aid quick comprehension. (format-validator F4)

2. **Strengthen the serialization argument** with citations to Shiv and Quirk (2019) on tree positional encodings, GraphCodeBERT (Guo et al., 2020) or Hellendoorn (2020) on code transformers handling nested structure, and RoPE (Su 2024) or ALiBi (Press 2022) for the long-context positional encoding argument. (citation-verifier C5, novelty-assessor N3)

3. **Add a second motivating example** (e.g., proof by induction decomposed into base and step cases) to broaden appeal beyond geometry/arithmetic. (prose-auditor P3)

4. **Reframe Contribution 5** (pre-registered ablation) as "an empirical commitment to testing Contributions 1-4" rather than as a parallel contribution. (prose-auditor)

## Detailed Notes by Domain

### Logic and Proofs

Definitions are well-formed but Definition 1 has an existential-quantifier ambiguity (L1) that propagates into how the paper uses "productive" in three distinct senses (L4). The hard-mask vs soft-penalty argument has a subtle conflation between simulation depth and policy depth (L3). The serialization example has a small depth-counting inconsistency relative to Definition 2 (L2). MCTS backup logic (method.tex lines 428-442) is consistent: backups follow the search tree's decision-chance-node structure, not the implicit hypothesis tree. No critical logic errors.

### Novelty and Contribution

The most distinctive contribution is identifier-based action targeting at fixed verb cardinality (Contribution 3); this is high novelty. The productive-vs-unproductive framing (Contribution 1) is original in naming, less so in mechanism. The hard-mask depth-limit (Contribution 2) has direct precedent in HyperTree Proof Search and rStar-Math, neither of which is cited. The serialization argument (Contribution 4) is plausible but under-supported by recent literature. The pre-registered ablation (Contribution 5) is a methodological commitment, not technical novelty.

The two largest novelty gaps are missing citations to rStar-Math (N1) and HyperTree Proof Search (N2); both are also citation findings (C1, C2). These are venue-blocking.

### Methodology

The headline ablation grid (R0, R3, R-infty) with three pre-registered hypotheses (H1, H2, H3) is sound in structure. Three independent gaps: (i) statistical analysis plan absent, with no effect-size thresholds, test specification, or multiple-comparison correction (M1, headline blocker); (ii) threats-to-validity subsection is unwritten TODO comments (M3); (iii) H2's mechanism claim (budget dilution) is not falsifiable independently of the pass@1 outcome (M4). The single-value-depth-3 grid (M2) is a robustness concern that the existing TODO mitigates with a planned follow-up sweep; promote the sweep into the pre-registration.

### Writing and Presentation

Writing is among the strongest in this review pass. Pythagorean motivating example anchors the abstract claim well. Implicit-vs-explicit hierarchy methodological position in §6.5 is the paper's strongest argumentative passage. Notation is consistent. No banned phrases. The principal prose suggestion is to surface the implicit-vs-explicit hierarchy argument earlier (it currently lives in the discussion).

### Citations and References

36 bibliography entries; 30 cited, 6 uncited (browne, guo, hu, openai, snell, wei). Seven entries marked TODO; spot-checking shows most can be cleared. Two critical missing citations: rStar-Math (Guan et al., 2025) and HyperTree Proof Search (Lample et al., 2022). Additional minor missing citations would strengthen specific arguments: Shiv and Quirk (2019), RAP (Hao et al., 2023), and one long-context positional encoding paper.

### Formatting and Production

Build is clean (pdflatex + bibtex + pdflatex + pdflatex cycle). All labels resolve; all citations resolve. Two cosmetic hbox warnings. 23 pages is appropriate for a companion to a 37-page framework paper. Bibliography is rendered correctly. No figures (figures/ directory empty); the serialization template uses a listings block. Page count exceeds standard venue limits but the declared target (arXiv) has no limit.

## Cross-Paper Coherence

The companion paper has fallen behind the framework paper's revisions. Specific coherence gaps:

| Gap | Companion location | Framework location (post-revision) |
|-----|---------------------|-------------------------------------|
| rStar-Math citation | Absent (related_work.tex §5.3) | references.bib + related_work.tex lines 124, 157 |
| Statistical analysis plan (Holm-Bonferroni, paired bootstrap) | Absent (experimental_plan.tex §5) | Present in framework's §5.5 |
| rStar-Math contrast | Absent | Present (framework's §6.4) |

The framework paper went through a multi-agent review pass on 2026-05-23 that produced three review-blocker fixes. The companion was drafted in parallel and has not absorbed the same revisions. Before submission, the companion should be brought into coherence with the post-revision framework paper.

No conflicts in terminology or claims detected; the companion's claims are consistent with the framework paper's (which is expected since the same author drafted both).

## Literature Context Summary

The four threads the paper engages with (hierarchical RL, tree encoders, MCTS for LLM reasoning, formal proof systems) are appropriately scoped. The hierarchical RL discussion is well-positioned. The tree-encoder discussion is dated (cites 2013-2019 work) and would benefit from 2019-2024 work on tree positional encodings and code transformers. The MCTS-for-LLM discussion has a major gap (rStar-Math missing). The formal-proof discussion has a major gap (HyperTree Proof Search missing).

The paper's most distinctive technical contribution (identifier-based action targeting at fixed verb cardinality) is sufficiently novel that the literature gaps are addressable through citation additions rather than rethinking. The two venue-blocking citation gaps (rStar-Math, HyperTree) are 2-3 sentences of contrast each plus the bib entries.

## Recommendation Justification

**minor-revision**: The paper is methodologically sound, well-written, and develops a coherent contribution. Three blocker issues (B1, B2, B3) and seven major issues are all addressable with focused editing (1-2 days). No fundamental redesign is required. The five-contribution list is solid; the recursive design choice is clearly developed; the pre-registered ablation makes the central claim falsifiable.

The decision boundary between minor-revision and major-revision is whether the blockers require substantial new content or just inserting and contrasting against missing prior work. B1 and B2 are citation insertions plus short contrasts; B3 is a statistical-analysis-plan paragraph that the framework paper can be cribbed from. None require new technical content or restructuring. **Minor revision** is the right call.

Conditional: if the target venue is anything other than arXiv (NeurIPS/ICML/ICLR standard track), the page-count cut would be substantial enough to push toward **major revision** for that specific target. For the declared arXiv target, minor revision is appropriate.

## Review Metadata

- **Agents used**: literature-scout (broad+targeted, merged), logic-checker, novelty-assessor, methodology-auditor, prose-auditor, citation-verifier, format-validator.
- **Cross-verifications performed**: 3 (rStar-Math gap cross-checked between novelty-assessor, citation-verifier, and framework paper sources; Definition 1 ambiguity cross-checked between logic-checker and methodology-auditor on operational vs syntactic use; threats-to-validity cross-checked between methodology-auditor and prose-auditor for cross-section coverage).
- **Disagreements noted**: None.
- **Manuscript build verified**: Yes (23 pages, clean build, all labels and citations resolve).
- **Cross-paper coherence checked**: Yes (against framework paper at /home/spinoza/github/expert-iteration/paper/ and against framework review at /home/spinoza/github/expert-iteration/.papermill/reviews/2026-05-23/review.md).
