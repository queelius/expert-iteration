# Experiment ledger

One row per controlled experiment. Discipline: a single manipulation per
experiment, an outcome measure named in advance, and the reading
pre-registered BEFORE the run (what result would mean what). Retro-logged
entries (E1-E9) predate the ledger; their "pre-registered" column records
what was actually written down in advance in the session record, which for
the early ones was less than it should have been. Costs are OpenAI spend,
approximate.

Conventions: "probe" = 20 episodes on the first 4 problems per family of
`datasets/casework-v1/train.jsonl`, depth limit 16, structured prompt
unless noted. "mini" = gpt-4.1-mini. Run files in `experiments/runs/`.

## Completed

| id | date | manipulation (single) | measure | result | cost |
|----|------|----------------------|---------|--------|------|
| E1 | 07-02 | none (baseline system run): mini teacher+generator on GSM8K x500 | solve rate; verb mix | 70.8%; split+abandon 0.5% (structural collapse) | ~$1.50 |
| E2 | 07-02 | remove the harness: mini direct chain-of-thought, same 500 problems, same scoring | solve rate | 94.4%; overlap one-sided (harness-only wins: 3) | $0.13 |
| E3 | 07-02 | problem distribution: GSM8K -> casework-v1 (structure required) | structural verb share | 0.5% -> 12-32% across probes; split 20/20 episodes | $0.15/probe |
| E4 | 07-02 | action space: SPLIT restricted to childless goals (after probe-1 re-split loop) | solve rate; split count | splits 97 -> 25; solve still 0/20 (failure moved downstream) | $0.15 |
| E5 | 07-02 | lens: COMPUTE taught solve-for-x re-expression + dup-COMPUTE masked | solve rate | 0/20 -> 2/20; first complete case arcs | $0.15 |
| E6 | 07-02 | prompt: commit-each-case playbook (contained a self-inflicted placeholder-id bug) | solve rate; fallback | 1/20; fallback 18.6% (measured the bug, not the design) | $0.15 |
| E7 | 07-02 | prompt: placeholder ids removed, verbatim-menu instruction | fallback rate | still 14.6%: fallbacks were NOT prompt syntax; 42/46 were illegal-commit intents | $0.15 |
| E8 | 07-02 | teacher capability x40: gpt-4.1 replaces mini, flat render held fixed | solve rate | 1/20, identical to mini. Capability is not the bottleneck | $0.85 |
| E9 | 07-02/03 | observation: tree render + parent tags + commit-readiness + rejection feedback, capability held fixed (mini) | solve rate; fallback source | 1/20 -> 5/20 (v6), 4/20 (v7 replicate); abandon 4/4 on dead-case families. INFORMATION BEAT CAPABILITY | $0.30 |
| E9b | 07-03 | fill the 2x2: gpt-4.1 teacher on the FIXED interface | solve rate | 4/20, same as mini. Render effect replicates across models; capability adds nothing in either column | $0.85 |
| E10 | 07-03 | remove the harness on casework: mini direct, all 240 train problems | solve rate (per family, per n_dead) | **100.0% (240/240)**, every family, every cut. Casework-v1 has no headroom for ANY harness; harness tax there is ~75-80 pts vs 24 on GSM8K | $0.02 |

## Standing conclusions the table supports

1. Structural verbs fire iff the distribution demands them (E3) - dataset
   design works as a behavior elicitor.
2. The binding constraint inside the harness was observability, not model
   capability (E8 vs E9: 40x capability bought nothing; the render bought
   4-5x). Fixed-capability information manipulations are the cheapest wins.
3. The residual harness failures at 20-25% are protocol overhead on
   problems the generator finds trivial (E10): the ground-COMPUTE
   re-expression dance and the ~10-step winning line, not math ability.
4. Casework-v1 is a behavior elicitor, not a challenge testbed: a testbed
   needs direct-baseline headroom (target band 40-70%).

## Pre-registered queue

### E11: teacher max_tokens truncation test
- Manipulation: controller max_tokens 64 -> 192, everything else fixed
  (gpt-4.1 teacher, tree view, same 20 problems).
- Hypothesis: gpt-4.1's 22 unparseable-reply fallbacks (a signature mini
  does not produce) are narration truncated at 64 tokens.
- Measure: fallback rate; unparseable-reply count; solve rate.
- Reading: unparseable count collapses (<5) => truncation confirmed; if
  solve also rises materially (>=8/20), 4.1 becomes the collection teacher
  candidate again. Unparseable count unchanged => hypothesis dead, mini
  stays the teacher on cost.
- Cost: ~$0.85.

### E12: casework-v2 difficulty calibration sweep
- Manipulation: generator difficulty knobs (two-absolute-value sums,
  3-branch piecewise, magnitude scaling), one config per batch.
- Measure: mini DIRECT baseline accuracy per ~40-problem batch ($0.02 each).
- Reading: keep configs landing in the 40-70% direct band; below 40% is too
  hard for a mini generator to solve per-case, above 70% has no headroom.
- Note: this calibrates the TESTBED; it does not touch the harness.

### E13: solve-claim COMPUTE ablation (after the spec + build)
- Manipulation: solve-claim COMPUTE on vs off, same teacher (mini), same
  interface, same 20 problems.
- Hypothesis: the ground-re-expression dance is the largest single tax
  (E10 conclusion 3); letting COMPUTE claim "x = 5" with substitution
  verification shortens the winning line ~1/3 and removes the failure mode
  probes 3-5 exposed.
- Measure: solve rate; median turns; masked-action count.
- Reading: on-arm solve >= 2x off-arm => adopt; no difference => the tax
  is elsewhere (protocol length per se), revisit granularity more broadly.
- Cost: ~$0.15.

## Method notes

- 20-episode probes distinguish 5% from 70%, not 20% from 30%: treat
  single-probe deltas under ~15 points as noise; replicate before
  concluding (E9 v6 vs v7 is the worked example: 5/20 vs 4/20, same
  conclusion).
- Never change two things between probes. E6 is the cautionary entry: a
  prompt rewrite bundled with new pseudo-syntax measured its own bug.
- Log everything the run produces (raw replies included); E7's diagnosis
  was possible only because fallback intents were finally logged verbatim.
- Baselines first: E10 cost two cents and re-scoped an entire workstream.
