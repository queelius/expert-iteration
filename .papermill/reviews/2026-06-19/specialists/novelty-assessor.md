# Novelty Assessment Report

## Summary
Genuine novelty lies in the SYNTHESIS (verifier-anchored structured-state context MDP + InvokeVerifier
action + chance-node MCTS over a non-enumerable LLM transition + multi-head Q distillation + pre-registration),
not in any single ingredient. Framework is correctly and consistently RL-first (no MCTS-first drift). The
related-work section has two load-bearing omissions (VerifierQ, Stochastic MuZero) that directly undercut two
specific novelty claims, plus a cluster of MCTS-for-LLM omissions (ToT, RAP, ReST-MCTS*, Math-Shepherd,
AlphaLLM) that make positioning look incomplete. C1-C4 survive as novel-in-combination; C5/C6 are genuine
methodology/feasibility contributions honestly framed as such.

## Contribution analysis
- **C1 (Q-learning context MDP + structured state + InvokeVerifier action)**: novel-in-combination. Closest
  prior: VerifierQ (core Q-learning axis, NOT cited), RAP (structured-state ancestor, NOT cited), ARTIST/
  ReCall/ReSearch (verifier-call-as-action, NOT cited). Framing is over-confident given omissions. Residual
  novelty: verifier-call as a typed action inside chance-node MCTS over a Fact/Hypothesis state with promotion.
  Severity: Major (over-claimed; cite-and-contrast).
- **C2 (shared frozen-LLM encoder+transition, multi-head Q over fixed verbs)**: most defensible technical
  contribution, appropriately scoped. "One forward, not two" KV-cache sharing is concrete and validated. Cite
  AlphaLLM on the coarse-action bet. Severity: Minor.
- **C3 (verifier-anchored vs learned-aesthetic Goodhart asymmetry)**: principle is partly pre-existing (DeepSeek-R1
  cited makes the verifier-over-critic case; Skalse 2022 / Gao 2023 are the foundational basis, NOT cited).
  Novel element: prior-channel quarantine as a structural commitment + intrinsic/extrinsic taxonomy. The
  overoptimization argument (discussion.tex:113-123) is asserted as if novel with zero citation. Severity: Major
  (foundational citations missing; principle narrower than wording suggests).
- **C4 (MCTS as policy-improvement operator + TTC knob)**: operator role borrowed and acknowledged. The
  structural element presented as a contribution, decision/chance-node bifurcation, is NOT novel: it is Stochastic
  MuZero's afterstate/chance-node mechanism. Genuine novelty: applying it to the non-enumerable LLM transition.
  The "deterministic-game tutorials collapse these" framing contrasts a known mechanism against a strawman.
  Severity: Critical pending citation; downgradeable to Major once cited-and-reframed.
- **C5 (pre-registered plan)**: genuinely new AS APPLIED HERE, correctly labeled methodology not algorithm.
  No over-claim. Situate against ML pre-registration literature. Severity: none.
- **C6 (toy pilot)**: genuinely new feasibility work, correctly and honestly caveated. Severity: none.

## Positioning verification
1. "closest prior published system is rStar-Math": accurate on the MCTS axis; incomplete on the Q-learning
   axis (VerifierQ) and structured-state axis (RAP). Scope to "MCTS-for-LLM systems."
2. Four-axis rStar-Math differentiation: axis 1 (chance-node) over-claimed (Stochastic MuZero); axis 2
   (multi-head Q over verbs) accurate; axis 3 (InvokeVerifier) partly over-claimed (ARTIST/ReCall); axis 4
   (pre-registration) accurate.
3. STaR/V-STaR-only self-improvement contrast: incomplete; ReST-MCTS*, Math-Shepherd, OmegaPRM are the actual
   nearest neighbors for per-step distillation. Residual ("per-step Q in a Bellman update" vs PRM step labels)
   is defensible but must be drawn against the right comparators.
4. Chance-node novelty framing ("deterministic-game tutorials" / "tutorial presentations"): over-claimed;
   attribution problem (math is correct, pedagogy good).

## Missing comparisons (ranked)
- Critical: Stochastic MuZero (chance-node mechanism, C4); VerifierQ (Q-learning core, C1).
- Major: Tree of Thoughts; RAP; ReST-MCTS*/Math-Shepherd; Skalse 2022 + Gao 2023 (Goodhart, C3); Brown 2024 (B1).
- Minor: ARTIST/ReCall/ReSearch (C1); AlphaLLM (C2); OmegaPRM; ReST-EM.

## Synthesis verdict
No single element is fully pre-empted; no prior work does structured-Fact/Hypothesis-state + verifier-as-action +
chance-node-MCTS over an LLM transition + multi-head-Q-distillation + pre-registration together. The novelty is in
the synthesis, which is intact. Principal exposure is citation gaps, not duplication. Recommendation: reinforce
C1/C2 (the Q-learning core and the shared-encoder architecture) as the load-bearing novelties and demote chance-node
bifurcation to "borrowed-and-applied," which both fixes the over-claim and strengthens the RL-first center.

## RL-first check
Confirmed RL-first throughout (contributions ordered Q-core-first; C4 explicitly "not the central learning
algorithm"; Position paragraph names Q_psi as the center; background ordering correct; related work explicitly
"inverts the narrative emphasis"). No flag. One subtle note: the most reviewer-exposed structural novelty currently
sits on the MCTS axis (chance-node bifurcation, which is pre-existing) rather than the Q-learning core; rebalancing
toward C1/C2 would help.
