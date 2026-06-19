# Format Validation Report

## Build status: CLEAN-WITH-WARNINGS. Page count: 51.

Verified by clean rebuild in a temp copy (latexmk -pdf, 4 pdflatex + 3 bibtex passes, exit 0, 836 KB PDF).
Independently confirmed by orchestrator: 0 undefined references, 0 undefined citations, 0 multiply-defined
labels, all 29 bib entries cited, all 29 \cite keys resolve, no "??" in PDF, no em-dash glyph in any source.

## Critical
None.

## Major
- **F-1. Three overfull hboxes** in the Reproducibility subsection (pilot_results.tex:754-782), from long
  \texttt{} path strings in inline prose: heads.py (3.10pt), analyze_distillation_rounds.py (2.16pt),
  trained_prior.py (21.97pt, visible on pp.38-39). Fix: wrap paths in \path{} (url, already loaded via
  hyperref), or move the file list into an itemize, or add \allowbreak after each "/".

## Minor
- **F-2.** Font shape T1/lmr/bx/sc undefined (bold small-caps): \textsc inside \paragraph headings at
  method.tex:267,286 (InvokeVerifier, CommitAnswer) silently falls back to bold upright. Cosmetic.
- **F-3.** Font shape T1/lmtt/bx/n substituted by T1/lmtt/b/n (bold monospace) at method.tex:85,336,379;
  visually indistinguishable. No action for arXiv.
- **F-4.** Three underfull hboxes in the bibliography (deepmind2024alphaproof, openai2024o1) from long URLs
  breaking poorly (worst badness 5607). Fix: \usepackage[hyphens]{url} or \sloppy around the bibliography,
  or shorten URLs.

## Reference / citation resolution
No undefined references; every \ref/\eqref resolves. 41 of 98 labels are defined-but-never-\ref'd
(informational: many subsections referred to by prose, not number). All 29 \citep/\citet keys map bijectively
to the 29 bib entries; no missing keys.

## Assets
3 figures referenced, 3 found: architecture.tex (TikZ, compiles, 6 libraries), pilot_results.pdf (p.31),
b_sweep.pdf (p.36). No missing or unreferenced assets. All 5 tables and the algorithm block compile without
column-spec or overfull errors.

## Page / section breakdown (from log page stamps)
Total 51 pages. Abstract+Intro ~2 (pp.1-2); Background 6 (pp.3-8); Method 14 (pp.9-22, figure p.13);
Experimental Plan 6 (pp.23-28); Pilot Results 11 (pp.29-39); Related Work 2 (pp.40-41); Discussion 6 (pp.42-47);
Conclusion ~0.5 (p.47); Bibliography 4 (pp.48-51). Method (14) and Pilot Results (11) dominate: 25 of 47 content
pages.

## Venue compliance
arXiv preprint target: article class, 11pt, 1in margins, plainnat, named author with ORCID, \today date.
Appropriate for arXiv; no flags. NeurIPS/ICML/ICLR submission would require swapping to the venue template.
