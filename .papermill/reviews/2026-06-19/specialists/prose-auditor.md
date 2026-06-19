# Prose Audit Report

## Summary
Well-written, internally disciplined methodology paper: RL-first through-line held consistently,
math notation centralized in macros and mostly used carefully, pilot negative findings reported with
unusual honesty. Dominant issues are prose-level bloat (near-duplicate arguments, a few >250-word walls)
and a small set of genuine notation collisions (one, tau, load-bearing across three roles). No em-dash
glyph present anywhere. No MCTS-first drift.

## Critical
None.

## Major

- **P-1 (notation: tau overloaded).** tau denotes (a) softmax-policy temperature (eq:softmax-policy),
  (b) visit-count distillation temperature (eq:visit-policy / eq:pimcts), and (c) LLM sampling temperature
  0.7 (pilot). Three distinct scalars; reuse never flagged; the two MCTS-side uses appear in two sections with
  no cross-link. Fix: rename the softmax-policy temperature to tau_pi (or beta); reserve plain tau for
  visit-count distillation; add a reminder at first reuse.
- **P-2 (notation: N overloaded).** N = visit count N(s,a)/N(s), target-sync interval N (method.tex:508,884),
  best-of-N, and pilot record-count column. The dangerous clash is sync-interval N vs visit-count N in the same
  Method section (Algorithm 1 "Every N steps" sits below code incrementing N(s,a)). Fix: rename sync interval to
  N_sync (or kappa); keep N(s,a).
- **P-3 (notation: d collides).** d = turn count (def:mathstate, [DEPTH] {d}, d<d_max) AND hidden dimension
  (eq:embed h_T in R^d, eq:adapter). Same Method section. Fix: rename hidden dim to d_h (or D).
- **P-4 (duplication).** Frozen-LLM attribution argument near-duplicated in method.tex:522-539 and 941-951
  (param count, GPU sufficiency, clean attribution in nearly identical words; the attribution claim also recurs
  in method.tex:359-360 and experimental_plan.tex:72-83). Fix: cut the second to its one new fact (gradient-flow
  scope) and back-reference the first.
- **P-5 (duplication, partly justified).** Goodhart argument at near-full length in method.tex:815-831 and
  discussion.tex:113-160 (the decision-boundary sentence and intrinsic/extrinsic split appear in both). Forward
  reference is intentional but the overlap exceeds it (method already does the full taxonomy). Fix: trim method
  to the principle only; let discussion own the taxonomy.
- **P-6 (wall).** method.tex:547-583 is still a ~350-word single paragraph bundling four ideas, with a ~180-word
  final sentence. Fix: split at "The MCTS variant we use sits between two extremes" (564) and "Sampled chance
  nodes are therefore the natural estimator" (578); break the final sentence.

## Minor (selected)
- **P-7.** Abstract is overloaded (~330 words: framework + architecture + reward + pre-registration + two-round
  pilot + rebalancing + closed-loop-pending + compute accounting). Trim closed-loop to one clause; drop the 42%
  and "single training seed" parenthetical from the abstract (keep 57%->72%). Target ~250 words.
- **P-8.** "class dilution" vs "class imbalance" (abstract.tex:32) vs "class-balance"; pick "class dilution."
- **P-9.** Per-edge backup warning ("tutorials get this wrong / collapsing biases the estimate") stated in three
  places (background:286-289, method:686-688, related_work:140-142). Keep concept in background + implementation in
  method (justified layering); cut the editorial warning to one location.
- **P-10.** State symbol s vs C_t: reconciled but the seam is rough; add one clause at sec:background-llm-mdp start
  ("we write C_t here, s elsewhere; same object").
- **P-11.** State-tuple symbols (G_open, c, A, V) vs dataclass names vs serialization markers vs lowercase
  collection names: four namings for the same containers. Add a one-line symbol-to-code mapping after def:mathstate.
  Note V (verifier log) collides with V* and leaf value V-hat; prefer the long name.
- **P-12 (borderline major).** PUCT exploitation symbol: background uses a separate symbol Q-hat for the
  visited-or-prior value; method writes Q_psi for both, then says "Q_psi is the chance node's empirical mean once
  visited and the head estimate Q_psi otherwise" (method.tex:648-649), a self-contradiction in one sentence. Fix:
  use Q-hat in eq:puct and rewrite to "Q-hat is the chance node's empirical mean W/N once visited, and the head
  estimate Q_psi otherwise."
- **P-13.** Hedging: "demonstrates" / "establishes" used for single-seed feasibility evidence (abstract:27,37;
  intro:103; conclusion:23). Soften "demonstrates" to "shows/indicates" for the quantitative 15pp claim; "establishes
  the compute accounting" -> "provides." Keep "demonstrates the loop closes" (legitimately demonstrable).
- **P-14.** "verb" vs "action" vs "cognitive move" vs "cognitive verb" vs "control action": standardize on
  "action" (MDP object) + "verb" (categorical label); drop "cognitive move" / "cognitive verb."
- **P-15.** pilot_results.tex:569-572 closed-loop loop description is a verbless sentence fragment; recast with
  "The full method is an online loop:".
- **P-16.** V-hat leaf value rendered as \hat V (background) vs \widehat V (method); leaf state s_T vs s_L.
  Standardize on \widehat V and s_L.

## Strengths
RL-first directive honored throughout; related-work explicitly states/justifies the inversion. Intellectual
honesty in the pilot is exemplary (negative second-round result, refuted-own-hypothesis v4b test, replaced a
confounded stability metric). The three "pending run" disclosures stated plainly, not buried. Centralized notation
macros. Pedagogical background (expanded sums before expectation; every symbol defined before abbreviation).
No em-dash glyph anywhere (verified).
