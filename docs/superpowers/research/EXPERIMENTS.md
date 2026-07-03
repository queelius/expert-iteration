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
| E1 | 07-02 | none (baseline system run): mini teacher+generator on GSM8K x500 | solve rate; verb mix | 70.8%; split+abandon 0.5% (structural collapse). **QUARANTINED 07-03**: collected under the flat pre-E9 render, so the accuracy numbers characterize a superseded interface and the trajectories are excluded from behavior-cloning (observations in the superseded format, not re-renderable post hoc). Retained as the lab record; GSM8K re-enters as a pre-registered eval target once the casework loop shows progress | ~$1.50 |
| E2 | 07-02 | remove the harness: mini direct chain-of-thought, same 500 problems, same scoring | solve rate | 94.4%; overlap one-sided (harness-only wins: 3) | $0.13 |
| E3 | 07-02 | problem distribution: GSM8K -> casework-v1 (structure required) | structural verb share | 0.5% -> 12-23% across non-degenerate probes (probe 1 hit 31.9% by pathological re-splitting; excluded per the findings memo); split 20/20 episodes | $0.15/probe |
| E4 | 07-02 | action space: SPLIT restricted to childless goals (after probe-1 re-split loop) | solve rate; split count | splits 97 -> 25; solve still 0/20 (failure moved downstream) | $0.15 |
| E5 | 07-02 | lens: COMPUTE taught solve-for-x re-expression + dup-COMPUTE masked | solve rate | 0/20 -> 2/20; first complete case arcs | $0.15 |
| E6 | 07-02 | prompt: commit-each-case playbook (contained a self-inflicted placeholder-id bug) | solve rate; fallback | 1/20; fallback 18.6% (measured the bug, not the design) | $0.15 |
| E7 | 07-02 | prompt: placeholder ids removed, verbatim-menu instruction | fallback rate | still 14.6%: fallbacks were NOT prompt syntax; 42/46 were illegal-commit intents | $0.15 |
| E8 | 07-02 | teacher capability x40: gpt-4.1 replaces mini, flat render held fixed | solve rate | 1/20, identical to mini. Capability is not the bottleneck | $0.85 |
| E9 | 07-02/03 | observation: tree render + parent tags + commit-readiness + rejection feedback, capability held fixed (mini) | solve rate; fallback source | 1/20 -> 5/20 (v6), 4/20 (v7 replicate); abandon 4/4 on dead-case families. INFORMATION BEAT CAPABILITY | $0.30 |
| E9b | 07-03 | fill the 2x2: gpt-4.1 teacher on the FIXED interface | solve rate | 4/20, same as mini. Render effect replicates across models; capability adds nothing in either column | $0.85 |
| E10 | 07-03 | remove the harness on casework: mini direct, all 240 train problems | solve rate (per family, per n_dead) | **100.0% (240/240)**, every family, every cut. Casework-v1 has no headroom for ANY harness; harness tax there is ~75-80 pts vs 24 on GSM8K | $0.02 |
| E11 | 07-03 | controller max_tokens 64 -> 192 (gpt-4.1 teacher, tree view, all else fixed) | unparseable-reply count (pre-registered: <5 confirms truncation) | 25 vs 22: UNCHANGED. Hypothesis dead; solve 4/20 unchanged; mini stays the collection teacher on cost. Bonus: f3's first-ever harness win; illegal-intent signature (compute on a closed case) is cross-model stable | $0.75 |

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

### E12 (continuing): casework-v2 difficulty calibration sweep
- Manipulation: generator difficulty knobs, one config per batch.
- Measure: mini DIRECT baseline accuracy per ~40-problem batch ($0.02 each).
- Reading: keep configs landing in the 40-70% direct band; below 40% is too
  hard for a mini generator to solve per-case, above 70% has no headroom.
- Note: this calibrates the TESTBED; it does not touch the harness.
- **Round 1 result (07-03): NULL.** f4 (three-region), f6 (extraneous
  trap), v1-five at scale=10, f4+f6 at scale=10: 40/40 = 100% on all four
  (160/160 total, ~$0.08). Parameter magnitude and these family
  structures cannot challenge mini anywhere in single-variable
  linear/quadratic algebra.
- **Round 2 result (07-03): NULL again.** f7 (tiered-pricing word skins):
  39/40, the single miss being exactly the designed tier-boundary slip
  (13x$4 + 1x$2 = $54 -> 14 GB, model said 15); f8 (integer counting
  under band + divisibility): 40/40 at scale 1 AND scale 3. Semantic
  extraction and enumeration discipline are also saturated for mini;
  the f7 trap mechanism is real but fires at 2.5%.
- **E12 closing synthesis (rounds 1+2, 279/280 across seven configs):**
  no reachable knob of this generator challenges gpt-4.1-mini. The
  correct conclusion is NOT an arms race toward competition problems: it
  is that the headroom question was being asked of the wrong model. The
  design's deployment target is a LOCAL SMALL generator (Qwen
  1.5B/3B-class); the decisive comparison is small-model-direct vs
  small-model-plus-controller, so the 40-70% band must be measured on
  the deployment-class model.
- **Round 3 result (07-03): RESOLVED on the deployment model.**
  Qwen2.5-1.5B-instruct direct, local CPU via Ollama, zero API cost,
  same scoring as always:

  | config | Qwen-1.5B | mini |
  |--------|-----------|------|
  | casework-v1 train (240) | **39.2%** | 100% |
  | f3 / f1a / f1b / f2a / f2b | 75 / 40 / 35 / 31 / 15% | 100% |
  | f7 tiered word problems | 5.0% | 98% |
  | f8 integer counting | 5.0% | 100% |

  Casework-v1 as shipped sits at the 40% band edge for the deployment
  model, with a per-family spread from 15% to 75%: v2 is a MIXING
  exercise over existing families, not new generator work. Dead-case
  problems are harder for Qwen (33% vs 43%), the design's predicted
  failure axis. ~14% of Qwen replies are unusable even with the
  type-loose schema. The decisive comparison now has a concrete target:
  Qwen-1.5B + trained controller must beat 39% on casework-v1.
  Incidental method note: the small model returns schema-spirited but
  type-loose JSON (reasoning as a list, answer as a bare number); the
  baseline schema was loosened to measure math rather than JSON
  pedantry, and the same looseness will show up as masked actions in
  the harness's generator role.

### E13: solve-claim COMPUTE ablation - RESOLVED 07-03: DO NOT ADOPT
- Manipulation: solve-claim COMPUTE on vs off, same teacher (mini), same
  interface, same 20 problems. Control arms: v6 (5/20), v7 (4/20).
- Pre-registered reading: on-arm >= 2x off-arm (>= 9/20) => adopt; no
  difference => the tax is protocol length per se.
- **Result: 4/20, identical to control.** The mechanism works (solve-claim
  facts landed and verified; f1a went 0-1 -> 2 wins) but the plateau is
  unmoved; fallbacks rose to 25%. Verdict per pre-registration: not
  adopted; the flag stays off (code remains, spec'd and tested, inert by
  default). Conclusion: the tax is the ~10-step protocol executed at
  ~90 percent per-step teacher fidelity, not any single step's content.
  With E6-E8, E11, and E13 all falsified, every teacher-side patch axis
  (prompts, capability, completion length, step granularity) is now
  eliminated; the one intervention that worked was observational (E9).
  The remaining lever is the one the design names: a trained controller
  with perfect menu fidelity by construction (Phase B). Cost: $0.20.

### E14: design-true regime probe - RESOLVED 07-03: fix the generator interface first
- **Result: 0/20 wins**, 303 decisions, masked 34%, teacher fallbacks 30%.
  The failure is precisely localized to the generator-facing schemas:
  SPLIT masked 62/99 (63%) and COMPUTE 40/73 (55%), form_hypothesis 0/12.
  Qwen cannot reliably satisfy the strict payload schemas (enum + typed
  array), the same type-loose-JSON axis E12 round 3 found; and a masked
  SPLIT leaves the goal childless so SPLIT stays legal and the teacher
  re-orders it (99 split attempts / 20 episodes). Per the pre-registered
  reading: fix the generator-facing lens/schema interface (lenient
  coercion at the payload boundary, mirroring the baseline-schema fix)
  BEFORE the full design-true collection; Phase B's premise is untouched.
- Original registration follows.

### E14 (original registration): design-true regime probe (teacher controller + local Qwen generator)
- Configuration: gpt-4.1-mini (OpenAI) picks (verb, target); Qwen2.5-1.5B
  (local Ollama CPU) writes all content; casework-v1 first-20; tree view;
  structured prompt; solve-claims off.
- Purpose: the long-pending prerequisite for the design-true collection:
  measure the small generator's masked-action rate and the win rate on
  the TRUE state distribution (design.md section 7.3).
- Pre-registered expectation: solve rate BELOW the teacher+mini-generator
  20-25 percent and below Qwen-direct 39.2 percent; this is not failure
  (Rung 1's product is decision data; the decisive comparison is the
  TRAINED controller + Qwen vs 39.2 percent).
- Reading: any wins + masked-action rate under ~40 percent of decisions
  => proceed to the full 240-episode design-true collection (~$1.50-2
  teacher-side). Zero wins or masked-dominated episodes => fix the
  generator-facing lens/schema interface first (the type-loose JSON
  axis found in E12 round 3), not Phase B's premise.
- Cost: ~$0.10 teacher-side; generator is free local CPU (~40 min).

### E15: design-true probe with coercion - RESOLVED 07-03: content, not shape
- **Result: 0/20; SPLIT mask 63% -> 8% (coercion worked), COMPUTE mask 65%
  (duplicates/unverifiable content), teacher fallbacks 32%.** With the
  shape barrier gone, the small generator's CONTENT lands and is poor:
  junk split children ("solve_statement"), restated equations as facts;
  the teacher then flounders on polluted trees. Per the pre-registered
  second branch: this is the deployment model's real limitation, exactly
  what Phase B must work within. Design-true collection proceeds at
  REDUCED scale (decision data is the product), and the open question of
  a stronger local generator (Qwen-3B / the GPU box) is a user decision.
- Original registration follows.

### E15 (original registration): design-true probe with generator-boundary coercion
- Manipulation (single, vs E14): lenient wire schemas + apply-time
  coercion for the generator's payloads (SPLIT children objects -> their
  longest string value; connective normalized; premises coerced to live
  int ids). Teacher, generator, problems, prompts unchanged from E14.
- Measure: typed-verb mask rate (E14: SPLIT 63%, COMPUTE 55%); wins.
- Reading: mask rate under ~20% and any wins => the boundary was the
  blocker; proceed to the full design-true collection. Mask rate down
  but zero wins => the residual is generator CONTENT (restating the
  case equation instead of solving it), which is the small model's
  actual limitation and exactly what Phase B must work within; collect
  anyway (decision data is the product) but at reduced scale.
- Cost: ~$0.10 teacher-side; local CPU generator (~40 min).

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
