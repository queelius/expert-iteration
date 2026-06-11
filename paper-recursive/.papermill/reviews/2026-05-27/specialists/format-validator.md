# Format Validator Report

**Paper**: "Bounded Structural Recursion in Verifier-Anchored Reinforcement Learning"
**Date**: 2026-05-27
**Focus**: build verification (pdflatex -> bibtex -> pdflatex -> pdflatex), label resolution, citation resolution, page length, venue formatting compliance.

## Summary

The manuscript builds cleanly. All labels resolve. All citations in the source resolve. One overfull hbox warning (cosmetic). The 23-page length is appropriate for a methodology-and-position companion to a 37-page framework paper.

No format-validator findings rise above minor severity.

## Build verification

Ran `pdflatex -interaction=nonstopmode -halt-on-error main.tex` and `bibtex main`. Build succeeds.

- Output: `main.pdf`, 23 pages, 519123 bytes.
- No "Reference undefined" warnings.
- No "Citation undefined" warnings.
- One Overfull hbox (0.24513pt too wide) in method.tex around the parent_hypothesis discussion (lines 147-154 in source, page 9 in output). Cosmetic; does not affect content visibility.
- One Underfull hbox (badness 1584 and 1189) in bibliography (deepmind2024alphaproof URL line break). Cosmetic.

**Finding (Minor)**: The two hbox warnings are typesetting micro-issues, not content issues. Can be resolved by minor wording adjustments or by allowing the URL to break differently.

## Label resolution

Searched for `\label` and `\ref`/`\eqref` pairs. All labels resolve. The label structure:

- Section labels: `sec:introduction`, `sec:background`, `sec:method`, `sec:experimental-plan`, `sec:related-work`, `sec:discussion`, `sec:conclusion`.
- Method subsection labels: `sec:method-overview`, `sec:method-productive`, `sec:method-depthlimit`, `sec:method-targeting`, `sec:method-serialization`, `sec:method-mcts`, `sec:method-canon`.
- Background subsection labels: `sec:background-recap`, `sec:background-context-mdp`, `sec:background-hrl`, `sec:background-trees`.
- Experimental plan labels: `sec:exp-inherited`, `sec:exp-ablation`, `sec:exp-secondary`, `sec:exp-threats`, `sec:exp-reporting`.
- Related work labels: `sec:rw-hrl`, `sec:rw-trees`, `sec:rw-mcts-llm`, `sec:rw-proof`, `sec:rw-self`, `sec:rw-summary`.
- Discussion labels: `sec:disc-limits`, `sec:disc-extensions`, `sec:disc-when`, `sec:disc-threats`, `sec:disc-implicit`.
- Definition labels: `def:productive`, `def:tree-depth`.
- Equation labels: `eq:bellman`.
- Table labels: `tab:depth-ablation`.

All labels are used; no orphan labels detected. Cross-references all resolve.

## Citation resolution

All 28 unique cited keys resolve to entries in `references.bib`. See citation-verifier report for the 6 uncited entries (browne, guo, hu, openai, snell, wei).

## Page length and venue appropriateness

- Total: 23 pages including bibliography.
- Bibliography spans pages 21-23 (approximately).
- Body content: ~20 pages.

**Comparison with framework paper**: framework paper is 37 pages. The companion at 23 pages is appropriately shorter. The brief explicitly flags: "flag if it tries to redo framework content rather than positioning itself as a companion." After reading the manuscript end to end, the companion does **not** redo framework content. Framework concepts (the seven-verb action set, the Q-head architecture, chance-node MCTS, verifier-anchored reward principle) are referenced descriptively rather than restated. The companion's content is tightly focused on the recursive decomposition design dimension.

**Finding**: The 23-page length is appropriate.

### Venue formatting compliance

The state file declares target venue as "arXiv preprint (companion submission, same or adjacent venue as framework paper)." Candidates include NeurIPS, ICML, ICLR, or a reasoning workshop.

- **NeurIPS / ICML standard track**: page limit is typically 9 (NeurIPS) or 8 (ICML) excluding references. The current 23-page format does **not** fit standard track. Acceptable for arXiv; would need substantial cutting for standard track submission.
- **NeurIPS / ICML position-paper or workshop track**: page limits are typically 8 pages (workshop) or up to 12 pages (some position-paper tracks). Still requires cutting.
- **ICLR**: page limit is 10 pages excluding references. Would require cutting.
- **arXiv**: no page limit. Current format is fine.

**Finding (Major)**: If the target venue is anything other than arXiv, the paper as currently formatted will not fit. The author may need to either:
- Submit to arXiv first and target a specific venue's page limit later by selecting which sections to compress.
- Reformulate as a workshop submission with substantial cutting (likely cutting the method section to its most novel arguments).

The brief says the target is "arXiv preprint (companion submission, same or adjacent venue as framework paper)" so the current format is acceptable for the immediate target.

## Package compatibility

Inspected `main.tex` preamble. The package set is conservative:
- Standard math: amsmath, amssymb, amsthm, mathtools, bm
- Standard graphics: graphicx, xcolor
- Standard tables: booktabs, array
- Standard lists: enumitem
- Standard code: algorithm, algpseudocode, listings
- Hyperlinks: hyperref with colorlinks=true
- Bibliography: natbib with numbers, sort&compress
- Theorem environments: standard amsthm setup

No exotic packages. The article class is `article` with `11pt,letterpaper`. This will accept most venue style files without conflict (NeurIPS, ICML, ICLR all use article-class derivatives).

## Custom commands

The notation commands defined in main.tex lines 58-77 are used consistently. The companion-specific additions (`\dmax`, `\depthlim`) are used in method.tex and experimental_plan.tex. No undefined-command errors.

## Figures and listings

- `figures/` directory is empty (verified). The paper does not include figure environments. The single visual element is the table in experimental_plan.tex (`tab:depth-ablation`).
- The serialization template in method.tex uses a `listings` environment for code-style formatting. Renders correctly.

**Finding (Suggestion)**: The paper would benefit from a figure showing the implicit tree structure (e.g., a visual rendering of the parent-pointer chain). The serialization template lists the tree linearly; a small accompanying tree diagram would aid quick comprehension. Not required for venue acceptance.

## Findings summary

| ID | Issue | Severity | Venue-blocking? |
|----|-------|----------|-----------------|
| F1 | Page count (23 pages) exceeds standard ML venue limits (NeurIPS 9, ICML 8, ICLR 10) | Major (for non-arXiv venues) | Yes, for those venues; No for arXiv |
| F2 | One overfull hbox warning (cosmetic) | Minor | No |
| F3 | Bibliography URL line break in deepmind2024alphaproof entry creates underfull hbox warning | Minor | No |
| F4 | No figures; a visual tree diagram would aid comprehension | Suggestion | No |

The build is clean; no critical formatting issues. The principal finding (F1) is contingent on the target venue. For arXiv, no action needed. For a peer-reviewed venue submission, the paper needs significant cutting.
