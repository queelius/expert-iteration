# Logic / Correctness / Math Report

## Summary
Mathematically and logically sound at the core: every equation checked is internally
consistent, RL-first framing holds (no MCTS-first drift), the Goodhart intrinsic/extrinsic
argument is valid and honestly hedged, the v4b causal inference is correctly scoped, and all
~15 arithmetic claims recompute correctly (independently re-verified by the orchestrator).
Substantive issues are cross-section consistency defects and one invisible unit mismatch.

## Critical
None. No proof or derivation is wrong.

## Major

### L-1. Stray "gridworld" reference with a broken substantive cross-reference
- method.tex:711 (rem:mcts-consistency): "the operative test of the estimator's practical
  adequacy on our gridworld, rather than as a proof of the asymptotic statement."
- "gridworld" appears exactly once in the paper; the subject is the LLM context MDP and the
  toy-problem pilot, and the cross-referenced sec:exp-ablations is on GSM8K/MATH. Copy-paste
  artifact from design.md. Points the reader to a test that does not exist.
- Fix: "on our verifiable-math tasks" or "in the pilot."

### L-2. Conclusion's "expected headline" overclaims relative to the pre-registered hypothesis
- conclusion.tex:46-50: "...a small base model ... matches a substantially larger reasoning
  model at matched inference compute."
- The pre-registered falsifiable hypothesis is strictly "B4 strictly exceeds B1 (best-of-N)";
  the plan explicitly disclaims B5 as a target (experimental_plan.tex:164, "B5 is a reference,
  not a target"). Stating an aspirational match-B5 headline that exceeds the committed criterion
  is exactly the conflation pre-registration is meant to prevent.
- Fix: restate in terms of B4-strictly-above-B1; describe the B4-to-B5 distance as calibration
  of how much of the gap is closed (matches intro's "meaningful fraction of the gap").

### L-3. Inconsistent leaf evaluator: online Q_psi (method) vs target Q-bar-psi (closed-loop)
- method.tex:619-620 and eq:puct:643 use Q_psi as leaf evaluator / exploitation term;
  pilot_results.tex:569 says the closed-loop run used "the target-Q head as the bootstrap leaf
  value." Different operators; consequential because Q_MCTS (the training target) embeds the leaf
  evaluator. The paper never reconciles eq:bellmantarget (single-sample y with Q-bar-psi, explicitly
  NOT the training target) with the Q_MCTS target the loop regresses on.
- Fix: make method.tex and eq:puct consistent with the implementation (state the leaf value is the
  target network), or note the closed-loop run deviated from the Phase-1 spec; clarify eq:bellmantarget
  vs eq:qmcts.

## Minor

- **L-4 (units).** pilot_results.tex:171-174: the CE lower-bound argument (H(p,q) >= H(p)) holds only
  in matched units. Entropy column is "in bits" (v2 2.070), CE column is unlabeled (v2 1.4512). If CE
  were bits it would sit below the target entropy, violating the bound; numbers are consistent only if CE
  is in nats (1.4512 nats = 2.094 bits >= 2.070). Story is actually clean (heads sit just above the floor)
  but the reader cannot tell. Fix: label CE units (nats), or convert to a shared base.
- **L-5.** discussion.tex:282-285 says the compute axis "is denominated in generated tokens per decision,"
  but sec:exp-compute pre-commits to total inference FLOPs as primary. Reword to preserve FLOPs-primary.
- **L-6.** abstract.tex:32 "class imbalance" vs "class dilution" everywhere else (the mechanism is
  dilution; v4b over-corrects to 30%, above baseline). Fix: "class dilution."
- **L-7.** Per-edge backup subscript convention differs between background eq:per-edge-backup (r_t) and
  method eq:backup (r_{t+1}) for the same reward. Both internally consistent, numerically identical, but
  not textually alignable. Fix: unify the convention.
- **L-8.** rem:mcts-consistency headline sentence asserts consistency for the V*-based expectation; with a
  fixed frozen leaf evaluator the chance-node mean converges to E[r + gamma V-hat(s')], not the V* one,
  unless V-hat = V*. Caveat 1 partly addresses this; soften the lead sentence. Also a small UCT-vs-PUCT
  tension noted in caveat 2 (honest).
- **L-9.** related_work.tex:80-82 cites eq:rewardterminal (terminal-only) for "intermediate rewards"; the
  intermediate rewards live in the next unlabeled paragraph. Imprecise; reword or label.
- **L-10.** v4b is not a pure rebalance: it also enlarges the set (7->10) and adds fresh instances under the
  same noisy B=20 regime. The inference treats v4b as a clean intervention on the dilution variable. The
  confound actually reinforces the paper's own conclusion (added instances inject target noise). Add one
  sentence acknowledging rebalance-plus-expansion.
- **L-11.** fig:pilot-headline caption says v4b "restor[es] its training share"; v4b over-restores to 30%
  (above baseline 25%). Body text (line 354-355) is correct. Fix caption to "raising above the baseline."

## Verified sound
All headline arithmetic (56.7->71.7 = 15.0pp; L1 0.540->0.314 = 41.9%; Q-MSE 39.1%; CE 10.9%;
gsm8k +6.7pp; algebra -33.3pp; dilution 42.8%; 9000/114 = 78.9); per-class-to-overall consistency in
tab:pilot-holdout (34/60, 43/60, 39/60, 37/60); composition arithmetic; all Bellman / Q-learning / MCTS /
loss / architecture equations; the Goodhart intrinsic/extrinsic logic; the v4b modus-tollens; the mean-L1-vs-
agreement confound argument; the statistical analysis plan internal consistency; RL-first framing throughout.
The three pending-result disclosures are mutually consistent (intentional, not flagged); they must stay
consistent when numbers land.
