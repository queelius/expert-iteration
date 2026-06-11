# Methodology Auditor Report

Scope: experimental design, statistical rigor, reproducibility. The paper is a pre-registered experimental-plan paper, so the methodology IS the contribution. The audit is correspondingly load-bearing.

## Summary

The pre-registered baseline grid (B0 through B5), three-metric compute accounting (FLOPs, forward passes, wall-clock), explicit falsifiable hypothesis ("B4 strictly above B1 at matched FLOPs on GSM8K and MATH 1-3"), three sanity-check ablations (A1, A2, A3), and seven-item threats-to-validity list are unusually disciplined for a 2026 LLM-reasoning methodology paper. The pre-registration is meaningful rather than ceremonial in most respects. There are nonetheless several specific places where the pre-commitment is looser than the rhetorical claim, and a small number of methodological gaps worth closing before the implementation begins.

## Findings

### MAJOR

**M1. The B4 versus B1 falsifiable hypothesis has a load-bearing parameter (the FLOP range over which "strictly above" must hold) that is not pre-registered.**
- Location: §5.5, exp-frontier subsection
- Quoted text: "the B4 curve lies strictly above the B1 curve at matched total inference FLOPs across the FLOP range they share."
- Problem: "the FLOP range they share" is operationally underspecified. Three concrete questions:
  - (a) What is the minimum FLOP budget at which both methods are evaluated? B1 has a natural minimum (best-of-1, that is, single sample) but B4 has a per-decision-tree minimum that depends on the chosen $N_{\mathrm{large}}$ value, which is itself not pre-registered.
  - (b) What is the maximum FLOP budget? B1 can scale arbitrarily (just increase $N$); B4's maximum is bounded by per-problem latency. Where is the cap?
  - (c) "Strictly above" how strictly? Is one accuracy point sufficient? Five points? Statistically significant separation at the seed-variance level?
- The paper's seed-variance threats-to-validity ("at least three training seeds per configuration ... with standard deviation reported on every figure") sets up the apparatus for a significance test but does not commit to one for the falsification criterion.
- Suggestion: Tighten the pre-registration. Specifically:
  - Pre-register $N_{\mathrm{large}}$ as a function of base model size (e.g., $N_{\mathrm{large}} = 256$ for the 1.5B base; scale accordingly).
  - Pre-register the FLOP range as (e.g.) the matched range of B1 with $N \in [4, 256]$, with the B1 curve interpolated at the B4 FLOP positions.
  - Pre-register "strictly above" as "the B4 mean accuracy is at least two standard deviations above the B1 mean accuracy at every comparison FLOP value." Or some similar statistical criterion.
- Without these clarifications, "the hypothesis was falsified" can be re-interpreted post-hoc by selectively choosing which FLOP range to consider as the comparison band.

**M2. B5 (capability ceiling) is partially pre-registered but the model selection is left as "a public distillation of an o1-style or R1-style system, for instance" without an exact commit.**
- Location: §5.3, §5.2
- Quoted text (§5.3): "B5 is the capability ceiling. It runs a larger reasoning model (a public distillation of an o1-style or R1-style system, for instance) under greedy decoding"
- Quoted text (§5.2): "[B5] serves only as a capability-ceiling reference point for the compute-versus-accuracy frontier, and its identity is allowed to differ across benchmarks if a stronger published reasoning model exists for one but not another."
- Problem: B5 is a baseline in the pre-registered grid. If its identity is "allowed to differ across benchmarks" and is not committed to specific models, then post-hoc selection of B5 to make the gap look more or less favorable to the framework is possible. Even though the paper frames B5 as a reference point rather than a target, the optics of B5 selection still matter for the headline figure.
- Suggestion: Pre-commit to specific B5 model identities per benchmark, with a documented exception process (e.g., "the B5 model for benchmark X is the public release that achieves the highest reported score on the public leaderboard as of 2026-MM-DD; substitution requires a written justification before any training begins"). The current language is too permissive for a pre-registered plan.

**M3. The sanity-check ablations (A1, A2, A3) test the right things but do not pre-register thresholds.**
- Location: §5.6
- The ablations correctly check (A1) bare MCTS without trained heads, (A2) trained value head with uniform policy prior, (A3) trained heads without search. Each ablation has a stated expectation ("A1 lies visibly below B4", "A2 below B4 means the policy distillation is doing real work", "B4-minus-A3 gap is the marginal value of test-time MCTS").
- Problem: "visibly below" and "indistinguishable from" are not operational. What absolute or relative gap counts as "the ablation passed" vs "the ablation failed"? Without thresholds, the ablation reports descriptive numbers rather than passing/failing criteria.
- Suggestion: Pre-register at least one numerical threshold per ablation. For example: "A1 passes if its mean accuracy on GSM8K is within five percentage points of B0; otherwise the bare MCTS scaffold is doing nontrivial work and the framework's attribution to trained heads is called into question." Similar thresholds for A2 and A3.

### MAJOR (gap; not a defect but a missing piece)

**M4. There is no pre-registered statistical analysis plan.**
- Location: should be added to §5 as a standalone subsection
- Problem: Three training seeds and eight inference replicates per test problem are pre-registered, but the analysis pipeline that turns these numbers into a "B4 above B1" verdict is not specified. Specifically:
  - How are the eight inference replicates aggregated per problem (mean? majority vote? mean and report variance)?
  - How are the three seeds aggregated (mean over seed of per-problem mean? mean of per-seed accuracy? bootstrap CI over seeds?).
  - What significance test applies to "B4 strictly above B1 across the FLOP range"? A pointwise Welch t-test at each FLOP value? A nonparametric test of curve dominance?
- The infrastructure for statistical claims is there (multi-seed, multi-replicate). What is missing is the pre-registered pipeline that the eventual headline figure will be run through.
- Suggestion: Add a brief subsection §5.8 "Statistical analysis plan" with a one-paragraph spec: aggregation rules, the test statistic for the falsifiable hypothesis, the alpha level, and what corrections (if any) are made for the multiple-comparison structure of "B4 above B1 at every FLOP value."

### MINOR

**m1. The "compute accounting" section commits to "one H100" for wall-clock measurement.**
- Location: §5.4
- Problem: An H100 may not be available to readers reproducing the experiment in 2027 or beyond, and many academic labs use A100s, L40s, RTX 4090s, or rented cloud instances of various shapes. The commitment is good for the headline numbers but the operational definition of "wall-clock seconds at fixed hardware" should accommodate the realistic compute landscape.
- Suggestion: Soften to "wall-clock seconds at fixed hardware (one H100 or, when an H100 is not available, the closest commodity equivalent with the substitution documented in the release notes)."

**m2. The "best-of-$N$ selector" pre-registration uses the terminal verifier as selector.**
- Location: §5.7 ("Best-of-$N$ selector choice")
- Quoted text: "We pre-register the selector as the same terminal verifier used to score all methods in the grid: if the verifier accepts at least one of the $N$ samples, B1 scores it as correct; otherwise, B1 scores it as incorrect."
- Problem: This is actually pass@$N$, not best-of-$N$. The paper calls it best-of-$N$ but describes pass@$N$ semantics. Pass@$N$ is the strongest possible baseline because it gives B1 perfect oracle access, which the paper explicitly acknowledges ("This is the strongest honest baseline for B1, and it is the standard against which the B4 curve has to win.").
- This is a deliberate choice to maximize the strength of the baseline, which is methodologically sound. But the terminology is loose. "Best-of-$N$" suggests a selector that picks one output, not an oracle that accepts if any one is correct.
- Suggestion: Either (a) consistently use "pass@$N$" throughout, or (b) keep "best-of-$N$" but add the operational definition where the term is first introduced (in the baseline table) so the reader understands the selector convention.

**m3. The benchmark contamination check is described but the contamination-detection method is generic.**
- Location: §5.7 ("Benchmark contamination")
- Quoted text: "substring matching the canonicalized problem statement against the base model's tokenizer and against publicly documented training sets"
- Problem: "Against the base model's tokenizer" is not a contamination-detection method. The tokenizer encodes text; it does not store training data. Presumably the intended check is substring matching against the base model's documented pretraining corpus, where one is available. Qwen2.5 and SmolLM2 have partially documented corpora; Llama3 does not document its training data publicly. The plan should specify what to do when the documentation is incomplete (most cases for frontier base models).
- Suggestion: Replace "against the base model's tokenizer" with "against the publicly documented pretraining corpus for each base model where available; for base models whose pretraining corpus is not public, the contamination check is run only against the publicly documented benchmark training/test split metadata."

**m4. The "five percent contamination threshold" is pre-registered but its meaning is not.**
- Location: §5.7
- What does "contamination rate exceeds 5%" trigger? Reanalysis on contamination-removed subsets? Exclusion of the benchmark from the falsifiable hypothesis? A footnote? The threshold value is fixed; the consequence of crossing it is not.
- Suggestion: Pre-commit to one consequence. E.g.: "If contamination on a benchmark exceeds 5%, the headline frontier figure is reported both on the full benchmark and on a contamination-removed subset, with both numbers reported in the abstract."

**m5. The discount factor $\gamma$ is described as "set in [0.99, 1.0]" with $\gamma = 1$ a defensible starting choice.**
- Location: §4.7 ("Hyperparameters worth flagging")
- Problem: $\gamma = 1$ on a finite-horizon MDP is acceptable. But $\gamma = 1$ on an MDP with reward $-1$ for incorrect commits (as in eq:rewardterminal) and depth limit $d_{\max} \approx 20$ creates a subtle interaction with the per-edge backup: any incorrect committed answer at any depth contributes $-1$ to every ancestor chance node's $W$, undiscounted. The training-loop bias this creates for the heads is worth documenting.
- Suggestion: Add a sentence to §4.7 acknowledging that $\gamma = 1$ with negative terminal rewards makes early-vs-late commit decisions value-equivalent, and that this is monitored via the per-step value targets during early training; if commit-aversion emerges, $\gamma < 1$ is the first lever.

**m6. The "headline frontier" describes one curve per method, but the curve parameterization for B0 and B5 is unclear.**
- Location: §5.5 ("Headline Frontier")
- Quoted text: "the base-greedy and ceiling baselines contribute single points."
- Problem: This is technically correct (B0 has no compute knob; B5 is a fixed model at greedy decoding). But on a compute-versus-accuracy plot, single points need an x-coordinate. What is the FLOP cost of B0 and B5? B0 is the LLM forward count for a single greedy completion at its committed length; B5 is the larger model's analogous count. These should be defined explicitly so they appear on the plot correctly.
- Suggestion: Specify: "B0 is plotted at the FLOP cost of one greedy decoding pass to the model's natural stop condition; B5 is plotted at the FLOP cost of one greedy decoding pass through the ceiling model to its natural stop condition. Both single-point baselines are plotted on the same x-axis as the multi-point curves of B1 through B4."

### SUGGESTIONS

- §5.7 ("Seed variance") notes "at least three training seeds per configuration." Three is the minimum for any seed-variance claim and is what most ML reviewers will accept; five would be standard for a strong empirical paper, but the cost may dominate. The pre-registration of three is appropriate; flagging this as "at least three (target: five if compute permits)" would be slightly stronger.
- The threats-to-validity list does not include "MCTS implementation bugs," which historically have been a source of inflated results in tree-search papers. A sentence on planned regression tests (e.g., reproducing a known MCTS-on-deterministic-game benchmark result before the headline runs begin) would strengthen the implementation discipline.
- The plan defers Lean Mathlib to phase 2 explicitly. The deferral is well-justified (tactic serialization, error ingestion, Mathlib dependency management are substantial engineering investments). A reader will accept this. A one-sentence statement of what would re-promote Lean to phase 1 (e.g., "if a Lean-tactic baseline becomes available off-the-shelf via LeanDojo or a similar library") would future-proof the deferral.
- The companion paper on bounded structural recursion is mentioned in §4.1 with an explicit "pre-registers an ablation on the depth-limit parameter" claim. The framework paper's experimental plan does not pre-register a corresponding ablation. This is internally consistent (depth-limit is phase-2 territory), but the relationship between the two pre-registration documents could be made explicit, perhaps in §5.8 if one is added.

## Reproducibility

- All datasets cited and identifiable (GSM8K, MATH, AIME 2024 and 2025, Lean Mathlib).
- All base models cited and identifiable (Qwen2.5-1.5B-Base, Llama-3.2-1B-Base, SmolLM2-1.7B).
- All hyperparameters relevant to reproducibility named ($\gamma$, $c_{\PUCT}$, $\tau$, $N$ target-sync interval, $\lambda$ loss balance, adapter dimension $m$, head capacities, $d_{\max}$).
- Released-alongside-code commitments include the audit prompts and verdicts for the verifier audit and the seed-specific aggregation in the release notes.
- The framework is reproducible from the paper alone modulo the verb-specific prompt prefixes for the verb-instantiator, which are described as "fixed templates" but not exhibited. A short appendix with the seven prefixes would close the reproducibility gap entirely.

## Verdict from methodology-auditor

The pre-registration is substantive and meaningful, not ceremonial. Four MAJOR items would strengthen the pre-registration's bite: (M1) tighten the FLOP range and "strictly above" criterion; (M2) commit to specific B5 model identities; (M3) pre-register numerical thresholds for ablation pass/fail; (M4) add a statistical analysis plan. Six MINOR items are operational specificity. The framework's experimental discipline is the paper's strongest methodological contribution and should be preserved; the suggested tightenings only sharpen it.
