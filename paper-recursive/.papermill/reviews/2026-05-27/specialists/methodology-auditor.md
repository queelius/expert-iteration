# Methodology Auditor Report

**Paper**: "Bounded Structural Recursion in Verifier-Anchored Reinforcement Learning"
**Date**: 2026-05-27
**Focus**: pre-registered ablation grid (R0, R3, R-infty), three hypotheses (H1, H2, H3), secondary measurements, statistical analysis plan.

## Summary

The methodological skeleton is sound: three pre-registered hypotheses, an ablation grid, secondary measurements that diagnose mechanism, and a reporting-discipline commitment. The principal gap is that the **statistical analysis plan is absent**. The framework paper has a Holm-Bonferroni-corrected paired-bootstrap plan; the companion does not. This is venue-blocking for a pre-registered design.

A second methodological concern is that R3 is fixed at depth 3 and the headline result rests on one chosen value. The TODO note about a follow-up sweep mitigates this but doesn't fully resolve it.

## H1 (Expressivity): falsifiable but under-specified

**Quoted text** (experimental_plan.tex, lines 81-84):
> "H1 (expressivity): on MATH (the benchmark with the most decomposable problem structure), R3 achieves higher pass@1 than R0 at matched inference compute. Falsified if R0 matches or exceeds R3 at every compute budget tested."

**Finding (Major)**: H1 commits to "higher pass@1" without specifying:
- the effect-size threshold (is a 0.5% pass@1 improvement enough?)
- the significance test
- the inference compute budgets tested
- whether the comparison is paired across problems

The framework paper's reviewer report flagged the same issue and the response was a Holm-Bonferroni-corrected paired-bootstrap plan. The companion's H1 has no analog. As written, the hypothesis is "weakly" falsifiable: any positive difference passes, any negative difference falsifies, but statistical significance is undefined.

**Suggested remediation**: Add a statistical analysis plan subsection. At minimum:
- Pre-register the effect-size threshold (e.g., R3 must outperform R0 by at least 1 percentage point on MATH pass@1, averaged across at least three seeds).
- Pre-register the test (e.g., paired bootstrap on per-problem outcomes with 10,000 resamples, Holm-Bonferroni corrected for the three hypotheses).
- Pre-register the compute budgets (e.g., B = 100, 300, 1000 simulations per decision).

**Severity**: Major; potentially venue-blocking for a venue that takes pre-registration seriously.

## H2 (Rumination): falsifiable but symmetric structure issue

**Quoted text** (experimental_plan.tex, lines 85-88):
> "H2 (rumination): R$\infty$ achieves lower pass@1 than R3 at matched inference compute, due to simulation-budget dilution along unproductive recursive paths. Falsified if R$\infty$ matches or exceeds R3."

**Finding (Major)**: H2 has the same statistical specification gap as H1.

A second issue: H2 attributes the predicted lower performance to "simulation-budget dilution along unproductive recursive paths." This causal attribution is what the secondary measurements (verifier-call frequency by depth, MCTS budget allocation) are supposed to support. But H2 as stated doesn't make the causal attribution falsifiable on its own. If R-infty does perform worse, the data confirms H2; if R-infty performs better, H2 is falsified, but the mechanism claim is undertested.

**Suggested remediation**: Add a sub-hypothesis H2a: "Conditional on R-infty underperforming R3 by at least the threshold, the MCTS budget allocation in R-infty shows at least X% of visits at depth >= 4, and the verifier-call frequency at depth >= 4 is below Y per trajectory." This makes the mechanism claim falsifiable independently of the pass@1 result.

**Severity**: Major.

## H3 (Depth distribution): well-formed

**Quoted text** (experimental_plan.tex, lines 89-94):
> "H3 (depth distribution): in successful R3 trajectories, the modal observed tree depth at CommitAnswer time is 1 or 2, with depth 3 used only on the harder problem tail. Falsified if depth 0 or depth 3 dominate, indicating that the depth-limit mechanism is not exercising the intended regime."

**Finding (Minor)**: H3 is the cleanest of the three because it commits to a specific empirical pattern (mode at 1 or 2, depth 3 used only on harder problems). The "harder problem tail" is fuzzy; a quantitative cutoff (e.g., problems in the top quartile by difficulty rating) would make this fully falsifiable.

**Severity**: Minor.

## Ablation grid: undertest concern

**Finding (Major)**: The ablation grid has three settings (depth 0, 3, infinity). The middle setting is one value. If the productive regime is at depth 2 (rather than 3), the paper's central claim could be confirmed by accident; if it's at depth 4, the headline ablation misses the regime.

The threats subsection (experimental_plan.tex, lines 140-144) notes: "We will additionally sweep depthlim in {1, 2, 3, 4, 5} on a held-out subset of MATH after the headline three-setting result is in, to map the productive regime in finer resolution." This is good, but it's currently a TODO comment, not part of the pre-registration.

**Suggested remediation**: Promote the depth sweep from TODO comment to pre-registered secondary experiment. Specify the held-out subset size and selection method.

**Severity**: Major.

## Secondary measurements: methodologically sound

The three secondary measurements (trajectory composition by depth, verifier-call frequency by depth, MCTS budget allocation by depth) are well-chosen and directly diagnose the mechanism claims. They are pre-registered in §experimental_plan §3 (lines 105-125), which is good practice.

**Finding (Minor)**: The MCTS budget allocation measurement is not defined precisely. "Depth distribution of visited nodes in the search tree" is ambiguous: does this count search-tree depth (MCTS depth from root) or state-space depth (recursion depth in the implicit hypothesis tree)? Both matter, and they are different.

**Suggested remediation**: Specify both in the measurement plan. Search-tree depth tells you how MCTS allocates simulation budget; implicit-tree depth tells you what kinds of states the search prefers to expand.

**Severity**: Minor.

## Threats to validity: incomplete

**Finding (Major)**: The threats subsection (experimental_plan.tex, lines 127-144) is entirely TODO comments, not drafted prose. The lines starting `% TODO: This subsection mirrors the framework paper's threats but adds the depth-limit-specific ones.` represent a substantial drafting gap, not a polishing gap.

In particular, the bullet-point list inside the TODO comment names four threats:
- Inherited threats (compute leaks, verifier coverage, benchmark contamination, seed variance)
- Depth-specific: policy never decomposes anyway (H3 catches this)
- Depth-specific: training instability from unbounded state spaces
- Depth-specific: single depthlim=3 may not be optimal

These need to be drafted into prose. The TODO content is good; the drafting is missing.

**Severity**: Major. The discussion section duplicates some of this content in lines 149-182 of discussion.tex, but the experimental plan should have its own dedicated threats subsection.

## Reporting discipline: appropriately committed

**Quoted text** (experimental_plan.tex, lines 146-162):
> "All three settings are trained to a fixed total compute budget... All three settings use identical random seeds... Mean and standard deviation across at least three training seeds are reported... Compute accounting reports total inference FLOPs, total LLM forward passes, and wall-clock seconds..."

**Finding**: The reporting discipline section is appropriately committed. Three seeds is on the lower end for an RL paper; the framework paper's review likely commented similarly. If the framework paper has been revised to use five seeds, the companion should match.

**Severity**: Minor.

## Cross-paper consistency check

The framework paper's review (which the brief references) flagged the following methodology issues that should be checked in the companion:
1. Holm-Bonferroni-corrected paired-bootstrap (the framework paper resolved this; companion has not).
2. Pre-registered effect-size thresholds (companion has not).
3. Compute-matched comparison criteria (companion partially addresses this with "matched inference compute" in H1/H2 but does not define the metric).

The companion paper does *not* match the framework paper's statistical rigor.

## Findings summary

| ID | Issue | Severity | Venue-blocking? |
|----|-------|----------|-----------------|
| M1 | Missing statistical analysis plan (paired bootstrap, Holm-Bonferroni, effect-size thresholds) for H1, H2, H3 | Major | Yes, for pre-registration venues |
| M2 | Ablation grid has only one middle value (depth=3); follow-up sweep is TODO comment | Major | No, but improvable |
| M3 | Threats to validity subsection is TODO comments, not drafted prose | Major | Yes (drafting gap) |
| M4 | H2's mechanism claim (budget dilution) not falsifiable as written | Major | No |
| M5 | MCTS budget allocation measurement under-specified (search-tree vs implicit-tree depth) | Minor | No |
| M6 | H3's "harder problem tail" not quantitatively cut off | Minor | No |
| M7 | Three-seed reporting may not match framework paper's revised standard | Minor | No |

The single biggest gap is M1. A pre-registered paper without a statistical analysis plan is a pre-registered paper in name only.
