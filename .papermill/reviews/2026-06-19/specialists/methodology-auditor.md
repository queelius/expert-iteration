# Methodology Audit Report

## Summary
A well-constructed framework + pre-registered-plan + feasibility-pilot paper. The pre-registered
plan is unusually rigorous for the genre: fixed baseline grid with non-overlapping roles, FLOP-
accounted compute axis with named leak modes, fully specified statistical analysis plan, pre-committed
falsification criterion. The pilot is honestly scoped to feasibility and reports an informatively-negative
result (v4b) that strengthens credibility. Remaining issues are pre-registration precision gaps (numerical
thresholds left qualitative) and internal inconsistencies that will bite when numbers land, not framework
defects. RL-first framing consistent; no regression.

## Critical
None (judged as framework + plan + feasibility).

## Major

### M-1. Ablation pass/fail criteria are qualitative, not numerical
- sec:exp-ablations (experimental_plan.tex:282-316). A1 "visibly below B4", A2 "indistinguishable from
  B4", A3 "gap is zero or negative." Not pre-registered decision rules. The headline frontier and the
  stats plan are exemplary (paired bootstrap, R=10000, Holm-Bonferroni, |Delta|=0.02 band); the ablations,
  which do the attribution work, get none of it. An ablation without a threshold is a post-hoc degree of
  freedom, undercutting "fixed before training begins."
- Fix: point the ablations at the existing bootstrap machinery (A1 passes iff B4-minus-A1 CI excludes zero
  and is positive at every point, and A1-minus-B0 within the 0.02 band; A2/A3 analogous). Costs nothing.

### M-2. B1's FLOP-matching target is stated three different ways
- Table caption "FLOP-matched to B2-B4" (121-122); prose "average ... spent by B2 through B4" (142-144);
  frontier "average FLOPs spent by B4 on that same problem" (229-232). Not equivalent (B2 is near-zero search).
  Load-bearing because the falsification test compares B4 vs B1 at K=5 shared FLOP points.
- Fix: pick one definition (per-problem FLOP-match to B4 is the strongest honest control, aligns with the
  stated intent), use it in all three places; clarify B1 is a swept curve, not a point.

### M-3. "Strictly above at ALL K=5 points after Holm-Bonferroni on 10 tests" risks Type II inflation
- sec:exp-frontier:234-245, sec:exp-stats:344-349. Conjunctive 10/10-reject criterion is anti-cherry-picking
  (good) but severe on the false-negative side. Diminishing returns are expected, so the highest-FLOP point
  is where the gap is smallest and most likely to fail; one saturated-point failure sinks the whole claim.
  No pre-registered power analysis; test-set sizes n never stated.
- Fix: add a power analysis (min detectable effect at chosen n), or restrict the conjunction to points where
  a gap is expected (pre-registered), or pre-register that 9/10 or 8/10 with failures confined to the top point
  is "supported with saturation" distinct from "falsified." State n.

### M-4. The target-stability monitoring gate has no committed threshold
- sec:pilot-lessons:711-720, disc-pilot:248-256 repeatedly invoke "a pre-committed threshold" for pairwise
  recommended-verb agreement but never give a number. The B-sweep supplies data to set one (algebra 0.73 at
  B=160; 0.31-0.32 at B=20). A gate referenced as pre-committed but numerically blank is not pre-committed,
  and it is the pilot's main transferable safeguard.
- Fix: commit a number grounded in the sweep (e.g. pairwise agreement >= 0.70 at the collection budget,
  stratified by family).

## Minor
- **m-1.** Per-problem accuracy = mean across seeds collapses seed variance before bootstrapping; defensible
  and stated, but the headline CI is problem-level-only (lower bound on total uncertainty). Add a seed-clustered
  robustness bootstrap.
- **m-2.** The |Delta|<0.02 effect-size escape hatch is silent on the mixed case (some points reject, some not);
  0.02 asserted without justification. Pre-register the mixed-outcome verdict and justify 0.02.
- **m-3.** background.tex:197-199 says Q_psi is "trained via LoRA adapters," contradicting the phase-1 frozen-LLM
  design (LoRA is phase-2). Muddies the clean-attribution argument. Fix: "small head over a frozen backbone
  (LoRA a phase-2 option)."
- **m-4.** B5 ceiling "identity allowed to differ across benchmarks" makes the ceiling curve a moving target;
  no falsification weight, so low stakes, but the "closes the gap to frontier" narrative needs a guardrail
  (log B5 identity/FLOPs/source per benchmark; name the specific model in any cross-benchmark statement).
- **m-5.** Single training seed in the pilot vs >=3 mandated; disclosed everywhere (good). Add one sentence
  tying the single seed to the measured B=20 target instability: read the 15pp as existence-of-effect, not
  effect-size.
- **m-6.** Verifier-coverage audit (200 verdicts/benchmark) has no pre-registered pass/fail threshold, unlike
  the contamination threat (5%). Mirror the contamination convention.

## Strengths
Falsification criterion genuinely pre-committed and operationalized (more rigor than most published empirical
work). Baseline grid roles non-overlapping; only defensible "missing baseline" is majority-vote / self-consistency
at matched FLOPs (best-of-N-with-verifier-selector partly subsumes it; recommend adding or justifying exclusion).
Compute accounting honest with named leak modes and an empirical per-decision anchor (79:1 generation:prefill).
v4b is a textbook controlled experiment reporting an informatively-negative result that complicates the authors'
own first hypothesis. B-sweep is sound and self-correcting (catches its own K=6 artifact; diagnoses its own
preferred metric as confounded and switches it). The three "run in progress" disclosures handled correctly.

## Reproducibility checklist
Algorithm fully specified: yes. Data: yes (pilot), partial (plan: n never given). Hyperparameters: yes.
Code: yes (named scripts/datasets/checkpoints/seeds). Stats: appropriate design; completeness gaps per M-1/M-3/M-4/m-6.
Baselines: adequate; resolve M-2 and consider majority-vote.
