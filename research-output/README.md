# Research Direction Review

Archive notice (2026-10-20): this directory preserves earlier investigations
and proposals, not verified current publication claims. For the public archive,
start with [the overview](../docs/research-overview.md) and
[release readiness](../docs/release-readiness.md).

Cutoff: 2026-07-18

This directory holds the durable work products for the prior-art and research-direction review.

- [`../prior-art-respose.md`](../prior-art-respose.md): final decision memo requested by the project owner. This remains the main entry point.
- [`source-ledger.md`](source-ledger.md): paper-by-paper evidence, overlap analysis, and search notes. Findings are recorded here before synthesis so they are not lost.
- [`tree-state-prior-art.md`](tree-state-prior-art.md): dedicated analysis of persistent tree or graph state, node-addressed actions, local lenses, and writeback.
- [`fast-pivot-portfolio.md`](fast-pivot-portfolio.md): ranked, one-person research pivots with minimum experiments and novelty kill criteria.
- [`action-interface-identifiability.md`](action-interface-identifiability.md): required validity screen for whether prompted reasoning verbs are stable causal interventions before policy learning.
- [`controller-worker-interface.md`](controller-worker-interface.md): controlled separation of machine-native controller state from language-native worker rendering and output.
- [`future-work-options.md`](future-work-options.md): author-stated limitations, proposed fallback questions, prior-art pressure, and go/no-go experiments.

## Working Rules

1. Use primary sources wherever possible.
2. Separate a paper's reported result from this review's inference.
3. Treat 2026 arXiv papers as recent evidence, not settled consensus.
4. Do not infer novelty from the absence of a search hit.
5. Evaluate every adaptive policy against the best fixed policy and a compute-matched sampling baseline.
6. Preserve the project invariant: the symbolic verifier is the only reward source.
7. Keep the research framing RL first. Search is an optional accelerator, not the method's spine.
