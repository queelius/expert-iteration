# Revision Report (WAVE 1: content fixes)

**Date**: 2026-06-19
**Source review**: `.papermill/reviews/2026-06-19/review.md`
**Scope**: Critical + Major (Critical, M1-M11, P-12) + mechanical Minor 1-18; pre-registration TODO markers for M6, M8, M9, and the deferred half of M7. Suggestions NOT applied (out of scope). Structure-and-Length restructure (Section C) deferred to WAVE 2 (not done here).
**Build**: clean, 54 pages, 0 undefined refs, 0 undefined citations, 0 overfull hboxes > 15pt, no em-dash glyph (and no en/figure/horizontal Unicode dash either).

## Build
- `cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` -> exit 0 (clean rebuild from `latexmk -C`).
- Pages: 54 (was 51; growth is from added related-work paragraphs and TODO markers; the page reduction is the separate WAVE 2 structure pass).
- `grep -rlP '\x{2014}' paper/sections paper/references.bib paper/main.tex` -> no matches (exit 1).

---

## Fixed (Verified)

### Chance-node bifurcation is Stochastic MuZero's mechanism (Critical, source: novelty-assessor / citation-verifier / both scouts)
- **Location**: method.tex sec:method-overview and sec:method-mcts; related_work.tex sec:rw-mcts-llm.
- **Fix applied**: Added `\citep{antonoglou2022stochastic}` (Stochastic MuZero, ICLR 2022) at every chance-node/afterstate claim. Replaced the "deterministic-game tutorials" strawman with a one-paragraph contrast: chance-node/afterstate bifurcation is borrowed from Stochastic MuZero and adopted directly; the genuine novelty is narrowed to its realization over the non-enumerable LLM-induced transition (s' set by verb-instantiator tokens plus a lossy parser, so the sampled chance node is the only available estimator). Demoted the mechanism from "structural contribution" to "borrowed-and-applied" (new paragraph "One mechanism we deliberately do not claim as novel..."), and reinforced the verifier-anchored Q-learning core (C1) and the shared-encoder "one forward, not two" architecture (C2) as the two load-bearing novelties (numbered novel list in method overview; "two load-bearing distinctions" in related work). Genuine novelty preserved, not retracted. RL-first framing intact.
- **Verified by**: novelty-assessor -- RESOLVED (Stochastic MuZero cited at the claim, strawman gone, demotion explicit, novelty narrowed correctly, C1/C2 reinforced, no MCTS-first drift).

### M1. VerifierQ uncited: closest prior on the Q-learning core (Major, source: citation-verifier / novelty-assessor / both scouts)
- **Location**: related_work.tex sec:rw-verifiers.
- **Fix applied**: Added `\citep{qi2024verifierq}` and a contrast paragraph: VerifierQ is verifier-as-Q and offline (IQL/CQL, modified Bellman); this paper is verifier-as-action (InvokeVerifier), online expert-iteration-distilled multi-head Q over a structured Markov state. Three clean differentiating axes (action-vs-Q, online-vs-offline, categorical-multi-head-vs-free-form-scorer).
- **Verified by**: novelty-assessor -- RESOLVED (accurate, correctly placed, no overclaim, strengthens RL-first positioning).

### M2. Missing related-work cluster (Major, source: citation-verifier / novelty-assessor / both scouts)
- **Location**: related_work.tex sec:rw-mcts-llm and sec:rw-selfimprove; experimental_plan.tex B1 definition.
- **Fix applied**: One-sentence contrasts added with correct keys: Tree of Thoughts (`yao2023tot`) and RAP (`hao2023rap`, named the conceptual ancestor of the context-MDP framing) in sec:rw-mcts-llm; ReST-MCTS* (`zhang2024restmcts`) and Math-Shepherd (`wang2024mathshepherd`) as the per-step-distillation comparators in sec:rw-selfimprove; Brown et al. 2024 / Large Language Monkeys (`brown2024monkeys`) where the B1 best-of-N baseline is defined.
- **Verified by**: citation-verifier -- RESOLVED (each cited with a one-sentence contrast in the right place; all bib entries verified accurate against real sources: ToT NeurIPS 2023, RAP EMNLP 2023, ReST-MCTS* NeurIPS 2024, Math-Shepherd ACL 2024, Brown 2024 arXiv).

### M3. Goodhart argument uncited + lightman claim-citation mismatch (Major, source: novelty-assessor / citation-verifier)
- **Location**: method.tex sec:method-reward; discussion.tex sec:disc-verifier; introduction.tex.
- **Fix applied**: Grounded the intrinsic-Goodhart argument in `\citep{skalse2022reward, gao2023scaling}` at both reward-hacking assertions (method + discussion), with minimal attribution insertions that leave the already-correct intrinsic/extrinsic distinction intact. Corrected the lightman2024verify mismatch in the introduction: lightman is now cited only for the process-vs-outcome verifier-supervision point it actually establishes; the reward-hacking point is attributed to Skalse/Gao as the paper's own framing.
- **Verified by**: citation-verifier -- RESOLVED (Skalse/Gao cited for the claims they support; lightman no longer cited for reward-hacking; entries verified: Skalse NeurIPS 2022, Gao ICML 2023, Lightman ICLR 2024).

### M4. alphaproof blog -> Nature 2025 paper (Major, source: citation-verifier / broad scout)
- **Location**: references.bib (`deepmind2024alphaproof`).
- **Fix applied**: Replaced the `@misc` blog entry with the Nature 2025 `@article` (Hubert, Mehta, Sartran et al., "Olympiad-Level Formal Mathematical Reasoning with Reinforcement Learning", DOI 10.1038/s41586-025-09833-y). Dropped the dated "silver-medal" framing.
- **Verified by**: citation-verifier -- RESOLVED (DOI and authors confirmed against Nature; no silver-medal language; all in-text \citep resolve).

### M5. Venue updates for five published-since-arXiv entries + bib top-comment (Major, source: citation-verifier)
- **Location**: references.bib.
- **Fix applied**: guan2025rstarmath -> ICML 2025; feng2023alphazerollm -> ICML 2024; snell2024scaling -> ICLR 2025; hendrycks2021math -> NeurIPS 2021 Datasets and Benchmarks Track; hosseini2024vstar -> COLM 2024; wei2022cot -> @inproceedings NeurIPS 2022. All changed @article -> @inproceedings with booktitle. Bib top-comment rewritten from the overstated "All entries verified..." to an honest "arXiv-tracked entries are updated to their published venue where one exists; remaining arXiv-only entries should be re-checked..."
- **Verified by**: citation-verifier -- RESOLVED (every venue/year confirmed against official proceedings; top-comment no longer overstated).

### M10. Conclusion overclaim vs the pre-registered hypothesis (Major, source: logic-checker / methodology-auditor)
- **Location**: conclusion.tex final paragraph.
- **Fix applied**: Replaced "matches a substantially larger reasoning model at matched inference compute" with the pre-registered hypothesis (heavy-search B4 strictly above best-of-N B1 at matched compute on the benchmarks where both produce signal). B5 restated as "a reference point, not a target"; the B4-to-B5 distance reframed as calibration of how much of the gap is closed. Also softened the quantitative 15pp "demonstrated" to "shows" in the conclusion body (Minor 17).
- **Verified by**: logic-checker -- RESOLVED (headline now B4-above-B1, B5 a reference, consistent with sec:exp-frontier and the intro's "meaningful fraction of the gap"; no residual overclaim).

### M11. Notation collisions (tau, N, d) + P-12 PUCT self-contradiction (Major / borderline-major, source: prose-auditor)
- **Location**: background.tex, method.tex (incl. Algorithm 1), eq:puct.
- **Fix applied**: Renamed the secondary meanings, kept the primaries. softmax-policy / Boltzmann temperature -> `\tau_{\pi}` (background); target-sync interval -> `N_{\mathrm{sync}}` (method text, Algorithm 1, hyperparameters); hidden dimension -> `d_h` (eq:embed, eq:adapter). Kept visit count N(s,a), the distillation temperature `\tau` in eq:pimcts/eq:visit-policy, and the turn count d. P-12: eq:puct exploitation term changed to `\widehat{Q}(s,a)` with the sentence corrected to "the chance node's empirical mean W(s,a)/N(s,a) once visited and the head estimate Q_psi otherwise," matching background.tex's eq:puct-bg.
- **Verified by**: prose-auditor -- RESOLVED (every renamed symbol consistent, no remaining bare-tau softmax temperature, no bare-N sync interval, no bare-d hidden dimension; kept meanings untouched; PUCT contradiction gone; no new collision). Note: the pilot sharpness-table record-count column header remains a table-bound `$N$`; the auditor confirmed this is disambiguated by table context and not a required-scope collision, so it was left as-is to keep the diff minimal.

### Minor mechanical fixes 1-18 (verified by the owning domain verifiers above where applicable)
- **1. Stray "gridworld"** (method.tex rem:mcts-consistency) -> "verifiable-math tasks". Done.
- **2. CE units unlabeled** (pilot_results.tex) -> labeled CE in nats; added the explicit nats/bits reconciliation (1.855 bits ~ 1.286 nats for v3-train) so the entropy lower-bound is consistent; table column labeled "(nats)". Done.
- **3. Compute-axis mischaracterization** (discussion.tex) -> reworded to keep FLOPs primary while noting generation (not prefill) dominates and the FLOP accounting must charge every generated token inside the search; removed "denominated in generated tokens". Done.
- **4. "class imbalance" -> "class dilution"** (abstract.tex). Done.
- **5. Per-edge backup subscript convention** -> unified background eq:per-edge-backup to the method's `r_{t+1}` convention. Done.
- **6. rem:mcts-consistency lead softened** -> now estimates `E[r + gamma V-hat(s')]` from the leaf evaluator, coinciding with the V*-based Bellman expectation only in the exact-leaf limit. Done.
- **7. eq:rewardterminal cited for "intermediate rewards"** (related_work.tex) -> reworded so the cite points at the terminal reward and the intermediate-reward fallback of sec:method-reward is named separately. Done.
- **8. v4b is rebalance-plus-expansion** (pilot_results.tex sec:pilot-v4b) -> added a sentence noting it both raises algebra's share and adds three fresh instances, and that this strengthens the refutation. Done.
- **9. Figure caption "restoring its training share"** -> "raising its training share above the baseline". Done.
- **10. background LoRA-vs-phase-1** -> "small action-value head over a frozen LLM backbone, with LoRA adaptation ... reserved as a phase-2 option". Done.
- **15. Three overfull hboxes** (Reproducibility) -> converted the long `\texttt{}` paths to `\path{}` (url package). Build now reports 0 overfull hboxes > 15pt. Done.
- **16 (= P-12).** Done (above).
- **17. Hedging verbs** -> abstract "demonstrates...by 15 pp" -> "shows"; "establishes the compute accounting" -> "provides"; intro contribution-list 15pp "demonstrates" -> "shows"; conclusion 15pp "demonstrated...improves" -> "shows a ... improvement". Kept "demonstrates the loop closing/closes" (pilot, intro position) per the brief. Done.
- **18. Citation/format hygiene** -> wei2022cot @article -> @inproceedings; DOIs added to Nature/Science entries (alphago, alphago-zero, alphazero, muzero, alphacode, mnih2015); `\usepackage[hyphens]{url}` added for bibliography URL hyphenation. The bold-small-caps font fallback (`T1/lmr/bx/sc`) is the one remaining cosmetic warning; the review classifies these as cosmetic font fallbacks, and a font-package change risks altering rendering, so it is left as-is and flagged here. Done (except the cosmetic font fallback, intentionally left).

---

## Deferred as visible TODO markers (author's scientific commitments; no numbers invented)

All markers use `\textcolor{red}{\textbf{[PRE-REG TODO: ...]}}` and are greppable via `PRE-REG TODO`.

### M6. Ablation pass/fail criteria qualitative (experimental_plan.tex sec:exp-ablations)
- TODO inserted: commit numerical A1/A2/A3 thresholds; suggested reuse of the bootstrap CI-excludes-zero rule plus the |Delta|=0.02 band from sec:exp-stats.

### M7 (deferred half). B1 FLOP-match definition (experimental_plan.tex)
- Unified all three statements to ONE definition: per-problem FLOP-match to B4, with B1 reported as a swept curve (table row, roles paragraph, headline-frontier paragraph). TODO inserted at the roles paragraph asking the author to confirm per-problem-match-to-B4 (vs the old B2-B4 average).

### M8. Conjunctive 10/10 falsification criterion: no power or n (experimental_plan.tex sec:exp-frontier and sec:exp-stats)
- TODO inserted (sec:exp-frontier): add a power analysis, or restrict the conjunction to points where a gap is expected, or pre-register a "supported with saturation" verdict for top-FLOP-point failures, and state n.
- TODO inserted (sec:exp-stats multiple-comparisons): state per-benchmark test-set size n and a power target / the saturation fallback.

### M9. "Pre-committed" thresholds left blank
- Target-stability gate: TODO inserted in pilot_results.tex sec:pilot-lessons and cross-referenced in discussion.tex sec:disc-pilot; suggested pairwise recommended-verb agreement >= 0.70 at the collection budget, stratified by family.
- Verifier-coverage audit: TODO inserted in experimental_plan.tex sec:exp-threats; suggested a verifier-audit acceptance threshold mirroring the 5% contamination convention.

---

## Stale Findings
None. Every quoted passage in the review was located in the manuscript and matched (allowing for the one Minor-7 anchor that I had already touched within this same pass).

## Retry Failures (Reverted)
None. All dispatched fixes passed verification on the first attempt.

## Not Touched (per scope)
- The three pending-run spots left exactly as-is: pilot_results.tex sec:pilot-closedloop last paragraph ("The first end-to-end run ... is executing as we write"); abstract clause "currently executing, with those learning-signal results pending"; sec:pilot-caveats "pending a run still in progress."
- experiments/, experiments/results/, src/ (owned by a GPU run in another session).
- Section C (Structure-and-Length) restructure: deferred to WAVE 2. No material moved to an appendix; Background/Discussion not condensed.
- Suggestions section: not applied (out of scope).

## Open items for the author (cosmetic / optional)
- `deepmind2024alphaproof` bib key still embeds "2024" though the entry is the Nature 2025 paper; all in-text \citep resolve correctly, so this is cosmetic. Rename to `deepmind2025alphaproof` (and update 3 call sites) if desired.
- Bold-small-caps font fallback (`T1/lmr/bx/sc`) remains as the single cosmetic build warning; the review classifies this as cosmetic. A font-package change to silence it is left to the author to avoid altering rendering.
- Optional zero-ambiguity tidy (not required by M11 scope): rename the pilot sharpness-table record-count column header from `$N$` to e.g. `$N_{\text{rec}}$`.

## Summary
- Fixed and verified: Critical (1) + Major M1-M5, M10, M11/P-12 (8) + mechanical Minor 1-10, 15, 16, 17, 18 (15) = 24 findings fixed and verified (4 specialist verifiers, all RESOLVED on first pass).
- Deferred as visible TODO markers: M6, M7 (confirm-half), M8, M9 = 4 commitments, 7 red markers placed.
- Failed: 0.
