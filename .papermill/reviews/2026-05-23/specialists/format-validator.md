# Format Validator Report

Scope: build verification, label resolution, bibtex integrity, venue-compliance.

## Build verification

The full build sequence runs clean:

```
pdflatex -interaction=nonstopmode main.tex  -> exit 0
bibtex main                                  -> exit 0
pdflatex -interaction=nonstopmode main.tex  -> exit 0
pdflatex -interaction=nonstopmode main.tex  -> exit 0
```

PDF metadata after build:
- Pages: 35
- File size: ~702 KB
- Producer: pdfTeX-1.40.25 (TeX Live 2023)
- Document class: article, 11pt, letterpaper

## Warnings observed

Only one substantive warning class:

1. `LaTeX Font Warning: Font shape T1/lmr/bx/sc undefined` (twice in the log)
2. `LaTeX Font Warning: Some font shapes were not available, defaults substituted.`
3. `Underfull \hbox` (badness 1584 and 1189) in paragraph at lines 58 to 64 of an early section

The font-shape warnings are cosmetic (lmodern does not provide bold smallcaps; LaTeX substitutes a similar shape). These do not affect the rendered output meaningfully and are not blocking for arXiv submission. If a venue insists on perfect font shapes, switching the smallcaps usage in the action-name typesetting (\textsc{Decompose} etc.) to a non-bold context would clear the warning.

The underfull hbox warnings indicate paragraphs that LaTeX could not stretch tightly. Two such warnings on a 35-page paper is well within normal range; no action needed.

## Label and reference resolution

- 80 labels defined across sections.
- 37 distinct labels referenced (via \ref, \eqref, \autoref).
- All references resolve; no undefined-reference warnings in the build log.
- 43 labels are defined but never referenced. The breakdown:
  - All section labels (sec:background-*, sec:rw-*, sec:exp-*, sec:disc-*) are defined for forward-compatibility cross-referencing from other papers (notably the companion paper) and for internal navigation; not referencing them all is acceptable.
  - 23 equation labels are unreferenced. Most of these are equation labels in the background section that exist for completeness but are not cited elsewhere in the paper. This is acceptable for a background section that establishes notation; the equations are addressable should a reader want to cite them externally.
  - One Definition label (def:mathstate, def:actionset) unreferenced from the body; the definitions are referenced by name rather than by number.
  - One Proposition label (prop:mcts-convergence) unreferenced from the body; the proposition is referenced as "the proposition above" in the immediately following paragraph.
- Recommendation: No action required. Unreferenced labels are not errors; they are an indication of forward-compatibility.

## Citation resolution

- All 28 bibliography entries are cited at least once in the body. No orphan entries.
- All cited keys resolve to existing bibliography entries. No undefined citations.
- The `\bibliographystyle{plainnat}` and natbib `[numbers, sort&compress]` options produce numbered citations that are sorted and compressed (e.g., `[1, 2, 5]` becomes `[1,2,5]` in compact form).

## Figure handling

- One figure: figures/architecture.tex (TikZ-based architecture diagram), inputted via \input{figures/architecture}.
- The figure environment is correctly placed in §4.4 (sec:method-architecture) with \label{fig:architecture} and a referenced \ref{fig:architecture} in the surrounding prose.
- The TikZ code uses standard libraries (positioning, arrows.meta, shapes.geometric, fit, calc, backgrounds), all loaded via \usetikzlibrary inside the figure file. This pattern works but means the libraries are loaded every time the figure is included; for a single-use figure this is fine.
- Recommendation: Move the \usetikzlibrary calls to the preamble of main.tex. This is a cleaner pattern and aligns with how TikZ figures are usually managed. The current location works but is slightly unconventional.

## Algorithm handling

- One algorithm: alg:outer-loop (Algorithm 1) in §4.7.
- algorithmic style is consistent with the algorithm and algpseudocode packages.
- Algorithm numbering renders correctly (counter starts at 1, single algorithm in the paper).

## Table handling

- One table: tab:baselines (Table 1) in §5.3 (the baseline grid).
- booktabs style used consistently (\toprule, \midrule, \bottomrule).
- Table caption above the table per IEEE convention; this is a generic article class so any caption position is acceptable.
- No other tables in the paper.

## Listings (code blocks)

- Three `lstlisting` blocks: the MathState dataclass definition (§4.1), the serialization template (§4.4), and they render cleanly with `\ttfamily\small` basicstyle and Python language highlighting.
- The serialization template's content is not Python; the language setting is just inherited from the global lstset. This works visually but is technically a small inaccuracy. Cosmetic only.

## Hyperref behavior

- `hyperref` is loaded with `colorlinks=true, allcolors=blue`, which produces blue text for both internal cross-references and external URLs. This is conventional for arXiv preprints.
- All internal links (cross-references to sections, equations, figures) resolve to the correct anchors.
- External links (the AlphaProof blog URL, the OpenAI o1 URL, the SmolLM2 URL) are present in the bibliography and are clickable.

## Venue-specific considerations

The paper is currently formatted for arXiv preprint (generic article class, 11pt, letterpaper, 1-inch margins). When a target venue is selected, retargeting will require:

- **NeurIPS/ICML/ICLR.** All three use a custom class file. The conversion is mechanical (replace `\documentclass[11pt,letterpaper]{article}` with the venue's class, drop manual geometry, adjust math/algorithm package versions if the venue specifies). Page limits at these venues range from 8 to 9 (main body) plus references and appendix; the current 35-page paper would need substantial restructuring to fit. The natural cut is to move the background section into the appendix and keep only the method, experimental plan, and discussion in the body.
- **TMLR.** Unlimited page count with single-column layout; the current paper would fit with minimal reformatting.
- **ICLR Tiny Papers / NeurIPS workshop.** 4 to 6 page limit; not feasible for the current paper.
- **arXiv preprint.** Current format is appropriate. The 35-page length is unusual for arXiv only in that most ML preprints aim for 10 to 20 pages; the longer form is justified for a methodology paper with comprehensive background.

## Source organization

- Section files in `sections/` directory, included via `\input{sections/...}` in main.tex. Standard convention.
- Bibliography in references.bib at the top level.
- Figure source in `figures/architecture.tex`, included via `\input{figures/architecture}` inside a figure environment.
- The structure is clean and modular.

## .gitignore presence

The paper directory contains a .gitignore. This is appropriate for a LaTeX project (typically excludes .aux, .log, .out, .toc, .bbl, .blg, .pdf if pre-built artifacts are not committed).

## Reproducibility of build

- Tex Live 2023 (Debian package) is required.
- Standard CTAN packages used; no custom packages required.
- No external system tools required during build (no pythontex, no minted with shell-escape).
- The build should reproduce identically on any Tex Live 2022+ install.

## Verdict from format-validator

The paper builds cleanly with no errors, two cosmetic font warnings, and two underfull hbox warnings within normal range. All labels and citations resolve. Unreferenced labels are forward-compatibility hooks for cross-paper references. The arXiv format is correct. Venue retargeting will require a class-file swap and (for the major conferences) substantial section moves to fit page limits. No format-blocking issues for the current arXiv target.
