# Casework probe findings: five probes, four machinery fixes, one open decision

Date: 2026-07-02. Status: probing halted by the pre-committed stop rule
(solve rate below 30% after the prompt-syntax regression was excluded).
Runs: `experiments/runs/2026-07-02-rung1-casework-probe20*.jsonl` (20
episodes each, first 4 problems per family of `datasets/casework-v1/train`,
gpt-4.1-mini as both teacher-controller and generator, structured prompt,
depth limit 16, about $0.15 per probe).

## Why these probes existed

The GSM8K Rung-1 sweep (500 episodes, 70.8% solved) confirmed structural-verb
collapse: `split` + `abandon` were 0.5% of 3,906 decisions, because GSM8K is
linearly solvable. `casework-v1` was built so that case analysis is required
for correctness. The probes asked: with problems that demand structure and a
prompt that teaches it, does a teacher LLM actually exercise `split`/`abandon`
and still solve the problems?

First answer: yes to structure, no to solving. Second answer: every observed
failure is controller-side, and most are failures the trained Phase-B
controller removes by construction.

## Probe ledger

| Probe | Fix applied before it | Solved | Structural share | Fallback | Failure exposed |
|---|---|---|---|---|---|
| 1 | none | 0/20 | 31.9% (degenerate) | 0.6% | root re-split loop: SPLIT legal on already-split goals; controller re-split the root up to 7x and never descended |
| 2 | SPLIT only on childless goals (commit 44e5412... action-space fix) | 0/20 | 8.9% | 1.3% | generator computes in the wrong direction (evaluates f, restates the equation) instead of solving for x; 15x duplicate-fact loops |
| 3 | COMPUTE lens teaches solve-for-x ("4x - 16 = 4" -> "(4 + 16)/4 = 5"); duplicate-COMPUTE masked | 2/20 | 12.3% | 1.3% | case arcs complete (split -> compute -> verify -> commit works end to end) but the controller re-computes verified facts instead of committing the case and aggregating at the root |
| 4 | per-goal dup guard; commit-each-case playbook with "commit:N" placeholder ids | 1/20 | 16.0% | 18.6% | SELF-INFLICTED: the "commit:N" pseudo-keys taught the model to emit non-menu action strings; all 59 fallbacks resolved to menu[0] |
| 5 | placeholder ids removed; "copy the action verbatim from the menu" | 1/20 | 22.6% | 14.6% | the real fallback driver: at 42/46 fallbacks NO commit was legal; the teacher attempts commit before verify. Plus nested splitting: root split exactly once per episode (guard works) but case-children split 30+ times |

## What the probes bought (permanent machinery fixes, all committed)

1. **SPLIT legal only on childless goals** (`available_actions`). An
   exhaustive split names its full case set in one call; re-splitting only
   duplicates. Found because casework made `split` fire at all; latent on
   GSM8K.
2. **Per-goal duplicate-COMPUTE guard** (masked action). Kills the
   same-fact-every-turn loop while still allowing a case's value to be
   restated at the parent for aggregation.
3. **Solve-for-x COMPUTE lens.** The ground-equation constraint is fine for
   arithmetic but under-determined for "solve this equation": the lens now
   teaches re-expressing the solution as a ground identity whose right-hand
   side is the value the terminal verdict scores.
4. **Aggregation forms verified.** `max(5, 3) = 5` and `5 + 3 = 8` pass both
   the SymPy checker and the terminal RHS verdict, so sum / largest /
   survivor questions all have clean root-level closing moves.

## The headline finding (for the deck and paper)

**Structure fires when the problems demand it**: structural share went from
0.5% on GSM8K to 12-23% on casework (split in 20/20 episodes, abandon
appearing on the dead-case families). The action space is exercisable; the
dataset does its job.

**Close-out is the bottleneck, and it is a controller failure, not a state or
verifier failure.** The winning line needs roughly 10 correct sequential
decisions (split; per case: compute, verify, commit; root: aggregate, verify,
commit). gpt-4.1-mini's per-decision protocol fidelity is too low, and the
errors compound: premature commit attempts (illegal -> fallback -> wasted
turn), nested splits of trivial case-goals, re-computing instead of closing.

**The Phase-B argument writes itself.** Two of the three dominant failure
modes are structurally impossible for the trained controller:
- Illegal-action fallbacks cannot happen: the pointer head is a softmax over
  the legal menu; it selects FROM legality by construction, whereas an
  LLM-as-controller emits free text that must happen to match a menu key.
- Action-repetition loops are what value-based training extinguishes first: a
  masked action earns -0.01 and changes nothing; its Q-value collapses.
The third (premature commit) is a sequencing error that BC + Q-learning can
learn from exactly the negative examples these probes produced.

## The open decision (user's call, in order of my recommendation)

A. **Stronger teacher probe.** The design intends a STRONG teacher driving a
   small generator; gpt-4.1-mini is deployment-scale, not teacher-scale. One
   20-episode probe with a stronger model (for example gpt-4.1, or a
   reasoning-tier model) costs well under $1 and tests whether the protocol
   is followable at higher capability. If yes, the 240-episode casework
   collection uses the strong teacher.
B. **Design change: COMPUTE accepts solve-claims** ("x = 5"; the verifier
   substitutes into the goal equation and checks). Shortens the winning line
   (no solve-in-head re-expression), more natural for algebra. Touches the
   no-unknowns COMPUTE invariant and the verifier; needs a spec first.
C. **Ship as-is.** Record the structural-share finding and the machinery
   fixes; leave casework solve rate to Phase B (where the controller failures
   are addressed by construction) and to the design-true local-generator
   regime.

These are not mutually exclusive: A is cheap and informative regardless; B
can be spec'd independently; C is the floor already banked.

## Honesty notes

- Probe 4's regression was self-inflicted (prompt pseudo-syntax) and probe 4
  therefore measured nothing about the design; it is included because the
  correction (verbatim-menu instruction) is part of the record.
- 20-episode probes have wide confidence intervals; the 0-10% solve rates are
  distinguishable from GSM8K's 70.8% but not finely from each other. The
  qualitative failure taxonomy (from reading trajectories) is the evidence
  that matters, not the point estimates.
- Nothing here validates the casework dataset as a benchmark for general
  models; it validates it as a structure-exerciser for this action space.
