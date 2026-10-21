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

### E16 - RESOLVED 07-03: 3B clears the content cliff per-case; root aggregation is the last choke
- (a) 3B direct on casework-240: **68.8%** (vs 1.5B 39.2%): top of the
  useful band; if 3B is adopted the decisive target becomes 68.8%.
- (b) Probe: **0/20**, but the failure narrowed dramatically: split
  content is now GOOD (mask 3/46, clean two-case trees), teacher
  fallbacks fell 32% -> 8%, and CASES GET SOLVED end to end. The single
  remaining choke is the root-level aggregation COMPUTE: 163/200 (82%)
  masked, dominated by the duplicate guard: the generator, re-prompted
  for the same target, re-emits the same rejected statement; rejection
  feedback reaches the TEACHER but the GENERATOR is memoryless, so
  episodes burn turns 7-15 in a mask loop at the root.
- Next single manipulation (E17, cheap): make the COMPUTE lens
  duplicate-aware: list the target's existing fact statements in the
  suffix with "produce a DIFFERENT statement", and/or bump generation
  temperature on a masked retry. Reading: any root aggregations landing
  + first design-true wins => collect with 3B against the 68.8% target.

### E16 (original registration): generator capability (Qwen-1.5B -> 3B), interface held fixed
- Manipulation (single, vs E15): the local generator only. Two runs:
  (a) 3B direct baseline on casework-240 (re-anchors the decisive-
  comparison target); (b) the standard design-true probe.
- Measure: (a) direct %; (b) wins, COMPUTE mask rate (E15: 65%), and
  split-content quality (junk-children rate, by inspection).
- Reading: probe wins > 0 with COMPUTE mask under ~35% => 3B clears the
  content cliff; design-true collection uses 3B and the target becomes
  the 3B direct number. Still 0 wins with junk content => 1.5B/3B class
  is below the content floor for typed generation; escalate to the GPU
  box (7B) or start Phase B on mini-as-both data only.
- Cost: ~$0.10 teacher-side; local CPU (baseline ~2h + probe ~1h).

### E17 - RESOLVED 07-07: NULL; design-true probing halted
- Manipulation (single, vs E16): COMPUTE suffix lists the target's
  existing facts with "must be different from all of these".
- Result: **0/20**; COMPUTE mask 83% vs E16's 82% (UNCHANGED, the
  pre-registered failure); overall mask 52% -> 39% but teacher fallbacks
  rebounded 8% -> 30%; root aggregation barely attempted (5 tries);
  commit:0 legal in 1/20 episodes; episodes now die in varied ways
  (nested case-splitting, abandoned live cases).
- Verdict: the duplicate list did not break the loop, and the failure is
  no longer one mechanism: with a 1.5B-3B generator the ~10-step
  protocol always finds a weakest link. Four design-true probes
  (E14-E17) at 0/20 is the stop signal, same rule as the E6-E7 prompt
  saturation. Design-true collection is BLOCKED on generator quality
  (7B-class on the GPU box when free), not on any single interface fix.
  Phase B proceeds on the mini-as-both casework data (which has wins).

#### Post-hoc audit correction (2026-07-13): E16/E17
- The E16 attribution to a root-aggregation choke is not supported by the
  transition log. Of 200 COMPUTEs, only 3 targeted the root and 2 of those
  were masked. The other 197 targeted non-root nodes and 161 of those were
  masked. Root COMMIT was legal in 0/20 episodes, and both primary children
  were solved in only 2/20 episodes. Thus the reported 163/200 COMPUTE masks
  diagnose a broadly distributed generation/protocol failure, not a root
  retry loop.
- E17 likewise attempted only 5 root COMPUTEs, 3 of which were masked. Its
  null result does not isolate root duplicate handling. The defensible joint
  reading is the existing stop decision after four zero-win probes: this
  1.5B-3B generator/controller protocol failed at several stages. References
  to root aggregation as the single or last choke are superseded by this
  audit; the raw results are unchanged.

### E18 - RESOLVED 07-09: factored Phase B path carries held-out signal
- Manipulation (single): replace the fixed menu-frequency action rule with a
  trained factored verb+pointer controller. This is the first Phase B training
  run, not a generator or live-policy experiment. It uses deterministic hashed
  lexical features as a cheap encoder diagnostic; frozen-LLM hidden states are
  a later, separate manipulation.
- Data: the four approved mini-as-both tree-render runs (v6, v7, E11, E13),
  converted with `phase-b-v1`. Keep all four replicas of a problem together:
  problem indices 0-14 train, 15-19 holdout (one holdout from each family).
  Policy cross-entropy uses only non-fallback decisions from positive-return
  training episodes. Q regression uses every training decision and its full
  Monte Carlo return, including masked-action constants and losing episodes.
- Fixed configuration: seed 0; 80 epochs; batch 64; AdamW at 3e-3 with 1e-4
  weight decay; Q-loss weight 0.25; hash dimension 512; hidden/repr/node-repr
  dimensions 128/64/64. No holdout metric is used for tuning.
- Primary measure: exact held-out `(verb,target)` match on all non-fallback
  teacher decisions. Baseline: the most frequent successful-training verb that
  is legal in the state, then the first legal target for that verb. Secondary
  diagnostics: verb match, pointer match conditional on the demonstrated verb,
  exact match restricted to rendered targets, and pre-training exact match.
- Pre-registered reading: learned exact match at least 10 percentage points
  above the menu-frequency baseline AND above the same network before training
  means the converter, masks, and factored learning path carry held-out signal;
  proceed to a separately registered frozen-LLM encoder run before live B=0.
  Exact match at or below the menu baseline despite falling training loss means
  the 171 available successful teacher labels and/or lexical encoder do not
  generalize; collect more winning data or upgrade the encoder before live
  episodes. Failure to reduce the component losses is an implementation/data
  failure, not evidence about the controller thesis. This five-problem grouped
  holdout is a diagnostic, so a margin near 10 points must be replicated before
  any substantive claim.
- **Result:** 1,067 decisions converted from 80 episodes; 871 non-fallback
  teacher labels, 171 of them in 17 positive-return episodes. The grouped
  holdout contains 225 eligible decisions. Exact `(verb,target)` match rose
  from **12.0% before training to 61.8% after training**, versus **3.6%** for
  the pre-registered menu-frequency rule (uniform-legal-pair expectation:
  12.7%). Verb match was 71.1%; pointer match conditional on the demonstrated
  verb was 78.7%; exact match on rendered targets was 61.8%; exact match on the
  holdout's 42 positive-episode decisions was 88.1%. BC loss fell from 1.36 at
  epoch 1 to 0.047 at epoch 80; Q MSE fell from 0.419 to 0.126. The 58-point
  margin over the registered baseline clears the 10-point proceed threshold.
- **Data audit:** 121 rows expose a legal target omitted by closed-subtree
  compaction in the archived state render; 70 demonstrations choose such a
  target (66 are VERIFY). The converter preserves these as explicit
  `rendered=false` placeholders rather than changing the action menu. Their
  exclusion barely changes exact match (61.8% overall vs 61.8% rendered), so
  they do not explain the result. Future collection must log structured node
  records directly and must not offer actions on hidden nodes.
- **Verdict:** the converter, pair masks, factored verb/pointer heads, and MC-Q
  warm-start work end to end. Per pre-registration, proceed to a separately
  registered frozen-model encoder run before live B=0. This is an offline
  action-match diagnostic, not evidence that the live controller beats the
  39.2% Qwen-1.5B direct baseline. Raw result:
  `experiments/runs/2026-07-09-E18-phase-b-bc-hash-seed0.json`.

### E19 - RESOLVED 07-09: frozen text embeddings preserve the signal
- Manipulation (single, vs E18): replace the 512-dimensional signed lexical
  hash with the installed frozen `nomic-embed-text:latest` endpoint
  (Ollama digest recorded in the cache). State and per-node strings are
  content-addressed and embedded once. The converter, grouped 0-14/15-19
  split, successful-episode policy labels, all-trajectory MC-Q labels, head
  architecture, seed, optimizer, 80 epochs, batch size, and every other E18
  hyperparameter remain fixed. The adapter input dimension changes
  mechanically from 518 to embedding-dimension-plus-6.
- Primary measure: exact held-out `(verb,target)` match on the same 225
  non-fallback decisions. Secondary measures and the fixed menu-frequency
  baseline are identical to E18. E18's reference is 61.8% exact match.
- Pre-registered reading: exact match at least 56.8% (within 5 points of E18)
  and at least 10 points above the menu baseline means the design's frozen
  text-embedding rung preserves the Phase B signal; adopt it for the first
  separately registered live B=0 controller run. At least 66.8% means the
  semantic encoder improves materially over lexical hashing. Below 56.8% means
  this encoder/data scale loses useful protocol features; do not run live,
  instead collect more successful labels or move to generator hidden states.
  A result within 5 points of either boundary requires a seed replicate before
  the next-stage decision.
- **Result:** the cache contains 1,639 unique texts at 768 dimensions, pinned to
  Ollama model digest
  `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`.
  Held-out exact pair match was **64.0%**, 2.2 points above E18 and above the
  56.8% preservation threshold. Verb match was 72.4%; pointer match conditional
  on the demonstrated verb was 80.4%; rendered-target exact match was 63.2%.
  BC loss fell from 1.33 to 0.146 and Q MSE from 0.474 to 0.209. The semantic
  encoder did not clear the 66.8% material-improvement threshold, but it
  preserved the offline signal with the design-compatible encoder rung.
- **Verdict:** adopt frozen Nomic state/node embeddings for the first separately
  registered live B=0 run. This remains an offline action-match result, not a
  solve-rate result. Raw result:
  `experiments/runs/2026-07-09-E19-phase-b-bc-nomic-seed0.json`.

#### Post-hoc audit correction (2026-07-13): E18/E19
- The registered menu-frequency comparator was too weak to establish that
  either text encoder carried held-out content. A stronger content-free
  lookup maps each exact legal-menu signature to its most frequent successful
  training `(verb,target)` pair. Ties use global positive-training pair
  frequency, then lexical order; unseen menus use the same global counts.
  It trains on 129 positive rows and scores **146/225 (64.8889%)** on the full
  holdout and **34/42 (80.95%)** on positive-episode holdout decisions. It
  sees 167/225 holdout signatures and is correct on **143/167 (85.63%)** of
  those. There are 35 positive-training signatures and 57 evaluation
  signatures.
- Against that comparator, E18 scores 61.8% overall and 88.1% on positive
  decisions; E19 scores 64.0% overall and 76.19% on positive decisions.
  Neither learned model beats the content-free lookup overall. The supported
  conclusion is therefore narrower: the heads learned a highly repeated
  structural/protocol phase signal, much of which is already present in the
  legal menu. These runs do not establish semantic-text generalization.
- E19 was also 2.8 points from its registered 66.8% material-improvement
  boundary. Its registration required a seed replicate for a result within 5
  points of either boundary, but E20 proceeded without that replicate. This
  missed decision rule and the stronger baseline are prospective constraints
  on further Phase B claims; they do not alter the archived measurements.
- E18-E22 regress Q against archived returns that include the historical
  non-verifier -0.01 masked-action penalty. That is the named
  `legacy-mask-penalty-v1` reproduction contract, not the reward contract
  for new training. The live and v2 `symbolic-only-v1` contract assigns
  zero reward to rejection and lets the consumed turn carry its cost. This
  correction preserves the archived metrics while restoring the repository's
  symbolic-verifier-only Bellman-target invariant prospectively.

### E20 - RESOLVED 07-09: offline cloning does not transfer to Qwen states
- Manipulation (single): Qwen2.5-1.5B direct prompting versus the same frozen
  Qwen2.5-1.5B generator orchestrated by the E19 greedy factored controller at
  search budget B=0. The learned controller uses frozen Nomic embeddings and
  checkpoint `2026-07-09-E19-phase-b-bc-nomic-seed0.pt`; neither frozen model
  trains during rollout.
- Data/configuration: first 20 records of `datasets/casework-v1/test.jsonl`
  (four per family), one worker, seed 0, controller depth limit 16. The direct
  arm uses the existing zero-temperature direct-baseline prompt and identical
  symbolic answer checker. This split has no E18/E19 teacher decisions in
  training or model selection. Raw generator payloads, exact mask reasons, and
  policy probabilities are logged for every controller transition.
- Primary measure: paired solve counts. Secondary measures: controller masked
  action rate and reason distribution, root-COMMIT reach rate, and verb/target
  policy confidence. The existing 39.2% train-set direct result remains the
  project-level target, but this probe's decision uses its own matched direct
  arm because 20 test problems are noisy.
- Pre-registered reading: controller wins at least three more problems than
  direct (a 15-point margin) with masked actions below 40% means the learned
  controller shows live promise; replicate on all 60 test problems. Any
  controller wins but a margin under three means the policy operates live but
  has not beaten direct prompting; inspect the logged failure distribution
  before choosing data collection versus RL fine-tuning. Zero controller wins
  or a masked-action rate at least 40% reactivates the E17 generator-quality
  stop: do not patch the controller from one probe, move to the 7B-class
  generator or collect design-true successful trajectories first.
- **Result:** direct Qwen solved **8/20 (40%)**; the E19 controller plus the
  same Qwen generator solved **0/20**. All controller episodes exhausted the
  16-turn budget and none made root COMMIT. Masking was only 17/320 decisions
  (5.3%: 8 generation errors, 8 reductio rejections, 1 duplicate COMPUTE), so
  the failure is not wire-format fidelity. The policy chose VERIFY 186 times;
  among 169 attempts with a following observation, only **1** verified and 162
  visibly failed. It repeated a previously attempted episode-target VERIFY 157
  times. It made zero root COMPUTEs, `commit:0` was legal in 0/20 episodes, and
  it chose 42 non-root SPLITs. Mean chosen verb probability was 0.931, so the
  loops were confident rather than sampling noise.
- **Verdict:** the pre-registered zero-win stop fires. E18/E19 action match was
  on mini-generated teacher states and does not transfer to Qwen-generated
  states: Qwen almost never emits verifier-acceptable facts, and the cloned
  policy has no successful labels for recovery after a failed VERIFY. Do not
  patch the controller from this probe and do not run the 60-problem arm. The
  next live prerequisite is successful design-true data from a 7B-class frozen
  generator (with the now-complete payload/mask logging), followed by BC and
  the already-designed off-policy Q fine-tune on that state distribution. Raw
  arms: `experiments/runs/2026-07-09-E20-direct-qwen15b-test20.jsonl` and
  `experiments/runs/2026-07-09-E20-controller-qwen15b-test20.jsonl`.

### E21 - RESOLVED 07-12: hosted Qwen2.5 capacity calibration
- Purpose: cheaply test whether the E20 verifier-content cliff changes between
  7B and 32B before paying for a larger design-true collection. This is a
  five-problem diagnostic, not an accuracy estimate.
- Models (one ordered capability sweep, same non-reasoning instruct family):
  `Qwen/Qwen2.5-7B-Instruct:together`,
  `Qwen/Qwen2.5-14B-Instruct:featherless-ai`, and
  `Qwen/Qwen2.5-32B-Instruct:featherless-ai`, all through the Hugging Face
  Inference Providers OpenAI-compatible router. Provider suffixes are pinned,
  not auto-routed. Current routed prices per million input/output tokens are
  $0.30/$0.30, $0.1078/$0.28, and $0.68/$1.20 respectively.
- Configuration: first five records of `datasets/casework-v1/test.jsonl` (one
  per family), seed 0, one worker, depth 16. For each model, compare its existing
  zero-temperature direct prompt with the unchanged E19 greedy controller at
  B=0. Nomic encoder, checkpoint, lenses, verifier, prompts, and action budget
  stay fixed. Log raw generator payloads, exact mask reasons, and policy
  probabilities. Expected total spend is below $0.10 based on E20 token volume.
- Primary diagnostic: verifier-successful facts and controller wins at each
  size. Secondary: matched direct wins, root-COMMIT reach, mask rate/reasons,
  repeated failed VERIFY count, and token cost.
- Pre-registered reading: the smallest model with at least one controller win
  and at least two verifier-successful facts becomes the generator for a
  separately registered 20-problem confirmation. No controller wins but a
  monotone rise in verifier-successful facts means capacity helps content but
  the mini-state clone still lacks recovery data; collect design-true teacher
  trajectories with the smallest model showing at least two verified facts,
  then retrain. Zero controller wins and fewer than two verified facts even at
  32B means model size alone does not clear the protocol/content interface; do
  not scale this harness further without redesigning the generator contract.
- **Result:** direct wins at 7B/14B/32B were **5/5, 5/5, and 4/5**. The
  corresponding E19-controller wins were **1/5, 0/5, and 0/5** over 77, 72,
  and 80 decisions. Verifier-successful facts rose from **7 to 8 to 10**, but
  failed VERIFY attempts were 18, 20, and 12; 16, 18, and 11 of those failures
  repeated an already-failed episode-target pair. A VERIFY was counted as
  successful when that exact action disappeared from the next state's legal
  menu; final-turn repeats inherit the earlier deterministic result for the
  same episode-target pair.
- **Control diagnostics:** root COMMIT became legal in 1/5, 2/5, and 0/5
  controller episodes and was selected once, twice, and zero times. Only the
  7B root COMMIT was correct. Masking was 33/77 (42.9%), 22/72 (30.6%), and
  35/80 (43.8%). Reasons were respectively 33 generation errors; 18 generation
  errors plus 4 duplicate COMPUTEs; and 27 generation errors plus 8 reductio
  rejections. Thus the hosted protocol remained brittle even as verifier-valid
  content increased.
- **Cost audit:** runner-counted input/output tokens from schema-valid calls,
  direct plus controller, were 6,105/1,025 at 7B, 8,668/1,310 at 14B, and
  10,962/1,345 at 32B. At the pre-registered routed prices the logged-call arm
  estimates were $0.002139, $0.001301, and $0.009068, for **$0.012508 total**.
  This is a lower bound on invoice cost if the provider billed malformed 200
  responses: the client updates usage counters only after schema validation.
  Token totals printed by the runners and the structured audit are preserved
  in `experiments/runs/2026-07-12-E21-summary.json`.
- **Verdict:** the first pre-registered rule fires at 7B: it is the smallest
  model with a controller win and at least two verified facts, so 7B is the
  candidate for a separately registered 20-problem confirmation. This is not
  evidence that the controller beats direct prompting: on the paired five-case
  diagnostic it lost 1/5 to 5/5. The monotone fact count suggests additional
  model capacity can improve generated content, but it did not improve
  closed-loop completion. Provider changes are confounded with size because no
  single routed provider served all three pinned models; treat this as a
  practical hosted-configuration sweep, not a causal parameter-count ablation.
  Raw arms are the six `2026-07-12-E21-{direct,controller}-qwen25-*-test5.jsonl`
  files under `experiments/runs/`.

#### Post-hoc audit correction (2026-07-13): E20/E21
- Both live controller arms selected actions from the behavior-cloning heads
  only. The fitted Q head did not control action selection, and no off-policy
  Q-learning update had been run. E20/E21 therefore test BC transfer, not the
  planned Phase B off-policy Q policy.
- E21's direct arm used generation temperature 0.0, while controller-side
  generation inherited the client default of 1.0. The direct/controller
  comparison is confounded by that mismatch in addition to the already noted
  provider/model confound.
- The reported verifier-successful fact counts of 7, 8, and 10 include true
  but goal-irrelevant statements such as `11=11` and `3<4`. Symbolic validity
  is not evidence that a claim advances its owner goal, so the monotone count
  does not support the stated inference that capacity improved useful
  generated content.
- Malformed provider responses were retained only as exception summaries, not
  as raw replies, and usage was recorded only after schema validation. The
  cost remains a lower bound, and the claim that raw payloads were logged for
  every transition is incorrect. These logging and temperature defects must
  be fixed before another hosted comparison.

### E22 - RESOLVED 07-13: text helps target selection but does not beat menu lookup
- Manipulation (single, vs E19): replace every 768-dimensional frozen Nomic
  state and node text embedding with an identically shaped all-zero vector.
  Retain the six scalar features, legal masks, factored heads, E19 data,
  grouped 0-14/15-19 split, seed 0, optimizer, losses, batch size, 80 epochs,
  and all other training settings exactly. Evaluation uses the historical
  `tree-render-v1` observation and legacy sequential verb-then-pointer
  decision rule so zeroing text is the only manipulation. The historical
  E19-matched converter setting is gamma 1.0, not the converter's newer 0.99
  default.
- Primary measure: exact held-out `(verb,target)` match on the same 225
  non-fallback decisions. Fixed comparisons are E19 at **64.0%** and the
  post-hoc exact legal-menu lookup at **146/225 (64.8889%)**.
- Pre-registered reading: exact match at least **59.0%**, within 5 percentage
  points of E19, means the text embeddings were not materially contributing
  to E19 and E18/E19 should be treated as structural/menu imitation
  diagnostics. A result below 59.0% is evidence that the text vectors carried
  some predictive signal, but it is not evidence of semantic generalization.
  Such a claim requires a seed replicate and performance above the fixed
  64.8889% content-free baseline. This is a one-seed diagnostic and no
  holdout result will be used for tuning.
- **Result:** exact pair match was **111/225 (49.33%)**, below the 59.0%
  boundary, E19's 144/225, and the content-free lookup's 146/225. Positive-
  episode exact match was **29/42 (69.05%)**, versus E19's 32/42 and the
  lookup's 34/42. Verb accuracy remained **73.33%**, close to E19's 72.44%,
  while pointer accuracy conditional on the demonstrated verb fell to
  **54.67%** from E19's 80.44%. The zero-text BC loss ended at 0.788 versus
  E19's 0.146.
- **Verdict:** the pre-registered below-59.0% branch fires. Frozen text
  vectors carried predictive target information, but E19 still did not beat
  the content-free menu lookup. Because the historical text also encodes ids,
  render annotations, and protocol state, this is not evidence of semantic
  generalization. The next controller dataset must use the structured v2
  observation and must report the exact menu lookup before any semantic
  ablation. Raw result:
  `experiments/runs/2026-07-13-E22-phase-b-bc-zero-seed0.json`.
  The exact temporary dataset was retained long enough to record SHA-256
  `c4553695690f528f6b5516b3e9388eed513b816a095f2bdeb7d93cc6e7e96e25`;
  the result artifact now also records its four inputs and exact conversion
  and training commands.

### E23 - RESOLVED 07-13: hosted v2 end-to-end acceptance gate
- Purpose: establish that the production v2 collector can cross both hosted
  model transports, preserve every raw attempt, complete the exact typed MDP,
  and publish only a strictly replayed run. This is a five-problem integration
  gate, not an accuracy estimate or a direct-versus-controller comparison.
- Manipulation (single): replace the tested offline clients with one pinned
  hosted role pair while holding the implemented v2 source, environment,
  collection, action, observation, closure, diagnostic, and replay contracts
  fixed.
- Source: records 0-4 of `datasets/casework-v1/train.jsonl`, one from each v1
  family, with problem ids `cw1-ccdf9e1567dc`, `cw1-2640597e2425`,
  `cw1-52e3b6b3ec25`, `cw1-ddad3be834f4`, and `cw1-4ba0fc91c9e8`.
  Problem-source SHA-256:
  `f3be145c1e536a98d4207c54ef85b5e0e516bce777c6ccd3d2a1f846c8881426`.
- Controller: OpenAI `gpt-4.1-mini-2025-04-14` at
  `https://api.openai.com/v1`, with the snapshot id also recorded as its
  revision. Generator: Hugging Face Inference Providers routed to the pinned
  Together provider with model
  `Qwen/Qwen2.5-7B-Instruct:together`; asserted Hub revision
  `a09a35458c702b33eeacc393d103063234e8bc28`. The router currently reports
  Together as the only live provider at $0.30 per million input and output
  tokens. The generator revision is asserted configuration metadata, not a
  provider-attested served-weight identity.
- Fixed settings: five sequential episodes, turn limit 16, temperature 0.0
  for both roles, 64 completion tokens for both roles, base seed 0 with the
  turn-and-retry offset policy, 120-second timeout, one parse retry, six
  429/5xx backoff retries, and provider guided JSON disabled for both roles.
  Standard OpenAI-compatible JSON-object mode remains enabled by the client.
- Prepared contract digests: environment
  `9316c93bc2cdf9b2ce507a989882af69669f8bf886e1f2ad6ce802c77dfdcfa7`;
  collection
  `627a013c0aa4cc7c73e99f944209c1760a8b69b9f616030f8c8e8cf18cc321de`.
  Hosted authorization must match these and the source digest exactly.
- Primary measure: the collector publishes both final artifacts, strict replay
  returns exactly the five ordered source episodes, every hosted attempt is
  retained and role-linked, stdout counters reconcile with the sidecar under
  the documented client semantics, and the sidecar contains at least one
  accepted HTTP attempt for each model role. Conversion of the sealed replay
  at gamma 1.0 must succeed, and every episode must contribute at least one
  provenance-eligible teacher label. Final run:
  `experiments/runs/2026-07-13-E23-v2-hosted-smoke5.jsonl`. Final diagnostics:
  `experiments/runs/2026-07-13-E23-v2-hosted-smoke5.diagnostics.jsonl`.
- Pre-registered reading: all primary conditions mean the hosted v2
  collection boundary is operational and a separately registered larger
  collection may proceed. Any replay, linkage, counter, conversion, or
  per-episode label failure is an implementation or transport failure with no
  behavioral reading; do not scale. Failure to publish final artifacts means
  collection or replay rejected the run; retain only `.partial` evidence and
  diagnose before any rerun. Solved count, fallback count, applied-generator
  rate, positive symbolic-reward rows, retry reasons, token usage, and cost are
  secondary diagnostics with no pass threshold and support no performance
  claim from five cases. At least one symbolic success confirms useful critic
  signal. Zero wins still passes integration, but makes the next 20 episodes a
  data-yield probe rather than bulk collection. Expected spend is below $0.10
  based on E21, but every billable attempt must be read from the diagnostic
  sidecar rather than that estimate.
- **Result:** the primary acceptance branch passed. Both final artifacts were
  published after strict replay of all five ordered problems and 66 decisions.
  Sealed conversion at gamma 1.0 produced 66 examples with dataset SHA-256
  `f0705564d5770376a6685d53a71defd9c2e3a0792c4ec416feee366521979e3f`.
  All 66 decisions were provenance-eligible teacher labels; per-problem label
  counts in source order were 16, 16, 16, 16, and 2. No controller fallback
  occurred. Final run SHA-256:
  `5efea19bd29619aba71d6d147135aababd7ded474230918b7d77b72d8b7368f6`.
  Final diagnostic SHA-256:
  `0418043eb3abb50d86ad03c01946cc83602e76c8bd2c91f41defb87d3af6e369`.
- **Transport audit:** the sidecar contains 66 controller calls, 66 generator
  calls, 160 HTTP attempts, and 66 composite decision receipts. All 66
  controller attempts were accepted. Generator attempts ended as 38 accepted,
  42 parse errors, and 14 schema errors; 28 of 66 generator calls exhausted
  the single retry. Sidecar token sums exactly matched stdout: controller
  33,316 input and 693 output tokens over 66 successful HTTP attempts;
  generator 52,432 input and 3,359 output tokens over 94 successful HTTP
  attempts. There were no HTTP-status or network failures.
- **Behavioral diagnostics:** only 5/66 learned effects applied: three direct
  Results and two Solutions. The other outcomes were 33 algebra rejections and
  28 payload errors. All 33 DECOMPOSE decisions rejected. Deterministic closure
  ran VERIFY five times and COMMIT once. The quadratic problem closed in two
  learned turns, yielding the run's one positive reward and one solved episode.
  This confirms critic signal but is not an accuracy estimate.
- **Cost:** at the pinned rates, controller usage was approximately $0.014435
  and generator usage $0.016737, for $0.031173 total. This covers every
  successful HTTP attempt reported by the sidecar, including malformed model
  replies that consumed a parse retry.
- **Post-hoc engineering diagnosis:** two generator prompt-contract mismatches
  dominate. DECOMPOSE repeatedly emitted interval notation with infinity
  glyphs, while the trusted adapter accepts only the documented ASCII
  relational grammar. Scalar instructions used the literal placeholder
  `symbolic value`, which the generator copied into seven accepted but
  algebra-invalid Results. On 21 other calls it copied and truncated the typed
  state instead of returning the requested two-field or one-field object.
  These observations do not change the registered acceptance reading.
- **Verdict:** the hosted collection boundary is operational, but a larger
  collection would mostly clone a teacher oscillating over generator failures.
  Do not bulk-collect yet. Align the generator lens instructions with the
  accepted domain and scalar grammar, test that single prompt-contract change,
  and then register the 20-problem data-yield probe.
- **Post-hoc controller-state audit, 07-14:** exact replay of the E23 artifact
  shows that every decision had exactly two legal pairs, both targeting root
  Goal 0. Each of the four unsolved episodes followed an exact 16-turn
  alternation between DECOMPOSE and its objective-specific scalar proposal.
  The solved episode chose DECOMPOSE and then PROPOSE_RESULT. Thus the complete
  run contains 33 DECOMPOSE, 17 PROPOSE_RESULT, and 16 PROPOSE_SOLUTION
  decisions, with no fallback or malformed controller reply. The observation
  exposes only the immediately previous outcome, so an A-B-A recurrence does
  not expose A's earlier failure when A is reconsidered two turns later. This
  is an offline artifact diagnosis, not a new experiment or a causal result.
  It motivates a bounded per-action failure-history manipulation if the
  model-only E30 screen below does not improve symbolic solve count. No hosted
  request was made and no credit was spent for this audit.

### E24 - RESOLVED 07-13: v2.1 generator-lens counterfactual over 66 sealed decisions
- Purpose: test whether the versioned generator lens correction clears the
  exact wire and decomposition failures observed in E23 before another
  end-to-end collection. This is a generator-interface acceptance gate, not an
  accuracy estimate, a controller experiment, a trajectory, or training data.
- Manipulation (single): change only the generator lens contract from
  `typed-claim-lens-v2` to `typed-claim-lens-v2.1`. The new contract replaces
  copyable placeholders with the trusted adapter's complete ASCII relational
  domain grammar and finite scalar grammar. Model, provider, transport, wire
  schemas, payload decoder, action semantics, verifier, temperature, token
  limit, retry budgets, all 66 replayed pre-states and actions, and every
  original first-attempt seed remain fixed. Re-observing each fixed state under
  the new environment changes its embedded environment digest only because the
  lens contract changed; the ordered manifest binds both old and new
  observation and request digests.
- Historical control: the sealed E23 run and sidecar, SHA-256
  `5efea19bd29619aba71d6d147135aababd7ded474230918b7d77b72d8b7368f6`
  and
  `0418043eb3abb50d86ad03c01946cc83602e76c8bd2c91f41defb87d3af6e369`.
  Strict replay supplies exactly 66 decisions in canonical order: 33
  DECOMPOSE, 17 PROPOSE_RESULT, and 16 PROPOSE_SOLUTION. E23 produced 38/66
  typed-usable generator calls, 12/33 typed-usable scalar calls, and 0/33
  applied decompositions.
- Generator and fixed settings: Hugging Face Inference Providers routed to
  Together, model `Qwen/Qwen2.5-7B-Instruct:together`, asserted upstream Hub
  revision `a09a35458c702b33eeacc393d103063234e8bc28`, temperature 0.0,
  64 completion tokens, one parse retry, six 429/5xx backoff retries,
  120-second timeout, standard JSON-object mode, provider guided JSON off, and
  the original turn-plus-retry seed schedule. The served-weight revision
  remains asserted rather than provider-attested.
- Frozen preparation: source problem SHA-256
  `f3be145c1e536a98d4207c54ef85b5e0e516bce777c6ccd3d2a1f846c8881426`;
  source environment
  `9316c93bc2cdf9b2ce507a989882af69669f8bf886e1f2ad6ce802c77dfdcfa7`;
  probe environment
  `664deedf1caa44efff16daac1888c2e3966aa37e9a2631a62036643a9e681c71`;
  collection
  `627a013c0aa4cc7c73e99f944209c1760a8b69b9f616030f8c8e8cf18cc321de`;
  ordered case manifest
  `629205cc6f0047a19b81f31750c1fa0c74df1e30f2a142b026d1bdfcf100049d`;
  probe contract
  `3ef15fcccc0c3ce4fc109ae07ef3ad20ef29cb7677146b9fdce8f329352b019c`.
  Preregistration id: `E24-v2-lens-v21-paired66`. Planned artifact:
  `experiments/runs/2026-07-13-E24-v2-lens-v21-paired66.jsonl`.
- Execution and logging: issue exactly one current-v2.1 generator call for
  each sealed row, with no controller call and no arm-produced state fed to
  another row. Retain every HTTP attempt, retry request, raw response, usage
  payload, decoded typed draft, algebra application, and symbolic closure
  diagnostic. Write exclusively to a sibling `.partial` path and expose the
  final path only after the complete artifact validates against the authorized
  contract and manifest.
- Primary measures: (1) typed-unusable calls, where no typed draft survives
  transport, retry, and payload decoding, at most **6/66**; (2) typed-usable
  scalar calls at least **30/33** across PROPOSE_RESULT and PROPOSE_SOLUTION;
  and (3) symbolically applied DECOMPOSE drafts at least **17/33**. The first
  two thresholds require at least 90% terminal interface usability. The
  decomposition threshold is the precomputed engineering floor: ASCII
  normalization of E23's already typed domain arrays leaves 17 structurally
  plausible exact-cover partitions (one absolute-value case and all 16
  piecewise cases).
- Pre-registered reading: meeting all three thresholds means v2.1 is a viable
  generator prompt contract; next run a separately registered same-five
  end-to-end confirmation, then the 20-problem data-yield probe. More than six
  typed-unusable calls, or fewer than 30 typed-usable scalar calls, means prompt
  control remains inadequate; test provider-enforced standard `json_schema`
  as the next single manipulation before changing the MDP. If both usability
  thresholds pass but fewer than 17 decompositions apply, the wire shape is
  repaired but decomposition semantics or state feedback remains inadequate;
  test bounded structured rejection feedback next. No result changes the
  controller, reward, action menu, or training path from this probe alone.
- Secondary diagnostics, with no decision threshold: raw attempt outcomes,
  provider-reported token use and cost, scalar claim validity and discharge,
  closure events, and counterfactual symbolic reward on each independent
  recorded pre-state. These are mechanism diagnostics, not solve rate. Expected
  HF spend is below $0.05 based on E23, and the artifact sidecar records every
  reported billable attempt.
- Result: the authorized artifact is
  `experiments/runs/2026-07-13-E24-v2-lens-v21-paired66.jsonl`, SHA-256
  `507c3b818ebee6c160bec021b84c60290f4af1a6fdcae725d2fa4efae74a5efc`,
  406,410 bytes and 200 canonical JSONL records. The canonical reader validates
  it, the temporary `.partial` path is absent, and its cardinality is exact:
  one header, 66 HTTP attempts, 66 call records, 66 case records, and one
  summary. It contains 66 unique source decisions and probe calls over the five
  sealed problems. No controller call or new trajectory row was produced.
- Interface measures: all 66 first HTTP attempts were accepted, with no retry,
  parse, schema, provider-response, HTTP-status, or transport error. All 66
  payloads were wire-usable and decoded. Typed-unusable is therefore **0/66**,
  passing the at-most-6 gate, and typed-usable scalar output is **33/33**,
  passing the at-least-30 gate. Reported usage is 45,264 prompt tokens and 868
  completion tokens. At the frozen $0.30 per million input and output rates,
  generator cost is $0.0138396.
- Decomposition measure: all 33 DECOMPOSE payloads decoded, but **0/33** applied,
  failing the at-least-17 gate. The bounded algebra diagnoses are 15
  `unsupported_domain`, eight `malformed_scalar`, eight
  `decomposition_domain_overlap`, and two
  `decomposition_domain_not_exhaustive`. Thirty outputs used `Reals` somewhere
  in a child-domain expression, and 15 placed exact `Reals` beside other
  children. The v2.1 instruction exposed `Reals` as a valid complete-domain
  form while demanding two or more partition children; the model often treated
  it as a sentinel or comparison bound instead of constructing an exact
  partition.
- Secondary scalar diagnostics: 19/33 scalar drafts applied, while 14 repeated
  a Claim already present in the fixed historical pre-state and were rejected
  as duplicates. Of the applied Claims, nine verified and discharged for a
  counterfactual +1, and ten failed the symbolic verifier. The nine positive
  rows are concentrated in only two problems: eight repeated Result drafts on
  one problem and one Result draft on another. They are row-level diagnostics
  over repeated historical states, not nine independent solves or an accuracy
  estimate.
- Verdict: the first two preregistered thresholds pass and the third fails.
  `typed-claim-lens-v2.1` repairs the wire and typed-payload contract, but its
  full-domain DECOMPOSE content contract is not viable for this 7B generator.
  The preregistered decomposition-semantics/state-feedback branch fires. Do not
  run the same-five end-to-end confirmation, the 20-problem yield probe, or any
  v2 training collection yet. Register one bounded structured semantic-
  rejection-feedback manipulation before considering a separately versioned
  decomposition representation.

### E25 - RESOLVED 07-13: corrected domain-parser rescore over 33 sealed E24 drafts
- Purpose: isolate a post-hoc implementation defect before testing the E24
  feedback branch. The bounded-domain regular expression tries `<` before
  `<=`. At an upper inclusive comparator, the shorter alternative can consume
  `<` and leave a scalar beginning with `=`, causing `malformed_scalar`. The
  declared grammar includes upper `<=`, so this is an implementation error,
  but changing executable acceptance semantics still requires new contract
  identities and a separate result.
- Manipulation (single): in a new domain-grammar profile, order the bounded
  comparator alternatives longest first, `<=` before `<`. Preserve both the
  executable old parser and its original adapter implementation identity behind
  the legacy `casework-goal-obligation-v1`,
  `casework-claim-verifier-v1`, and `casework-exact-aggregate-v1` identities.
  Bind the corrected profile to `casework-goal-obligation-v1.1`,
  `casework-claim-verifier-v1.1`, and
  `casework-exact-aggregate-v1.1`, and reject mixed profile triples. Thread the
  selected profile through canonicalization, solving, coverage, coverage
  digests, verification, receipt freshness, and aggregation. Payload semantics,
  rejection-code sets, graph/action/transition schemas, symbolic-only reward,
  model outputs, and every other parser rule remain fixed.
- Fixed input: the 33 decoded DECOMPOSE drafts in the canonical E24 artifact,
  SHA-256
  `507c3b818ebee6c160bec021b84c60290f4af1a6fdcae725d2fa4efae74a5efc`,
  matched to their exact sealed E23 pre-states and actions. The rescore must
  first validate the complete E24 artifact, strict E23 reconstruction under the
  exact legacy profile, and a legacy rescore reproducing every stored E24
  application status and algebra detail. Each ordered row binds its E24 index,
  source decision id, action, typed-draft digest, source and probe observation
  digests, and legacy result. It makes no hosted call, controller call,
  generator call, or trajectory row.
- Planned artifact:
  `experiments/runs/2026-07-13-E25-v2-domain-parser-rescore33.jsonl`.
- Frozen source bindings: E23 run
  `5efea19bd29619aba71d6d147135aababd7ded474230918b7d77b72d8b7368f6`;
  E23 diagnostic
  `0418043eb3abb50d86ad03c01946cc83602e76c8bd2c91f41defb87d3af6e369`;
  source environment
  `9316c93bc2cdf9b2ce507a989882af69669f8bf886e1f2ad6ce802c77dfdcfa7`;
  source collection
  `627a013c0aa4cc7c73e99f944209c1760a8b69b9f616030f8c8e8cf18cc321de`;
  source problem projection
  `f3be145c1e536a98d4207c54ef85b5e0e516bce777c6ccd3d2a1f846c8881426`;
  E24 artifact
  `507c3b818ebee6c160bec021b84c60290f4af1a6fdcae725d2fa4efae74a5efc`;
  E24 probe contract
  `3ef15fcccc0c3ce4fc109ae07ef3ad20ef29cb7677146b9fdce8f329352b019c`;
  E24 environment
  `664deedf1caa44efff16daac1888c2e3966aa37e9a2631a62036643a9e681c71`;
  and E24 case manifest
  `629205cc6f0047a19b81f31750c1fa0c74df1e30f2a142b026d1bdfcf100049d`.
- Frozen E25 bindings: corrected environment
  `3396ce7464c8bf4a8cd60c5aa26172c02dc4ed4589ca7eb9e2bcfd906119ee54`;
  adapter implementation
  `verifier_anchored_rl.reasoning_v2.math_adapter.CaseworkMathVerifier`;
  legacy profile
  `103fb711a9daed75b29ef5f2bc6837b1f25ed38c47988da7cc6fbf54c74f7431`;
  corrected profile
  `42ee4bf1f481dd92bebac990b8fc0c95d280a4a5c1054b0b9dc4d5070f6581c8`;
  legacy expectation
  `a0c4fda099d343937384c2027914e3f278646cbd2ed3f894d61c9e99b46e0f4a`;
  corrected expectation
  `4e6beb6b72ab7fa05aa7986ea69b80250ad23e71698780de1671a47ae8e71d56`;
  ordered 33-case rescore manifest
  `8955f15991ac60d9d99ec85399dcb573a178ab44cf4ddbdd07921e20d7f9fcb8`;
  and rescore contract
  `ec307795e8dc39ae70515ca5036f5442f5784b72d40ed269fa3bb654ccbd842a`.
  Counts are exactly 66 reconstructed source decisions and 33 selected
  DECOMPOSE drafts. Preregistration id:
  `E25-v2-domain-parser-rescore33`.
- Execution authorization: the CLI must receive every frozen digest and count
  above, plus the exact planned output path, before it opens the exclusive
  `.partial` path. It must reproduce the registered legacy and corrected
  distributions before linking the final path. Preparation is zero-network and
  created neither a final nor partial artifact.
- Primary acceptance is an exact deterministic regression, not an empirical
  threshold. The legacy rescore must reproduce 15 `unsupported_domain`, eight
  `malformed_scalar`, eight `decomposition_domain_overlap`, two
  `decomposition_domain_not_exhaustive`, and zero applied. Static inspection of
  the only eight affected drafts fixes the corrected expectation at 15
  `unsupported_domain`, zero `malformed_scalar`, 15
  `decomposition_domain_overlap`, three
  `decomposition_domain_not_exhaustive`, and zero applied. Any different count
  or per-row legacy mismatch is an implementation or analysis failure and
  blocks interpretation.
- Pre-registered reading: exact agreement validates the corrected profile and
  confirms that the parser defect only changes rejection classification on
  these sealed drafts. E24's 0/33 behavioral verdict remains unchanged, and its
  bounded structured-feedback branch remains next under the corrected profile.
  No E25 result can authorize an end-to-end or data-yield run.
- Secondary diagnostics, with no decision threshold: exact old and new
  rejection-code counts per case, canonicalized child obligations when
  accepted, and profile-specific coverage digests. The diagnostic-only artifact
  uses its own canonical header, 33 ordered case records, and summary; a strict
  reader validates it before a no-clobber `.partial` publication. It must not
  use a canonical controller-run or training-data schema. These are
  deterministic adapter diagnostics, not model accuracy, reward evidence, or
  training data.
- Result: the authorized artifact is
  `experiments/runs/2026-07-13-E25-v2-domain-parser-rescore33.jsonl`, SHA-256
  `f2be14e62cd04d680b454545cd253bfdb05c8ed5e404212b5b90bfcb7c8b1307`,
  76,323 bytes and 35 canonical records: one diagnostic header, 33 paired
  cases, and one summary. Its strict reader validates, its result-manifest
  digest is
  `fb2919c2a043a4a1e241f149716cd176a23e16a2d898a55f7b9e5c71faeddabd`,
  and no `.partial` remains. It contains no model call, controller decision,
  trajectory, reward record, or training row, and incurred zero hosted cost.
- Exact regression: the legacy arm reproduced every stored E24 application
  status and reason. Its aggregate counts are 15 `unsupported_domain`, eight
  `malformed_scalar`, eight `decomposition_domain_overlap`, two
  `decomposition_domain_not_exhaustive`, and 0/33 applied. The corrected arm
  produced exactly the registered 15 `unsupported_domain`, zero
  `malformed_scalar`, 15 `decomposition_domain_overlap`, three
  `decomposition_domain_not_exhaustive`, and 0/33 applied. All 33 paired status
  transitions are `algebra_rejected -> algebra_rejected`.
- Verdict: E25 validates the v1.1 domain profile and confirms that the upper
  inclusive comparator defect changed rejection classification only on these
  drafts. It does not revise E24's behavioral result: full-domain authoring
  remains 0/33 applied. Keep the legacy profile for exact E23/E24 replay, use
  the corrected obligation, verifier, aggregate, and environment identities
  for new work, and proceed to the registered bounded structured-feedback
  branch. Do not run an end-to-end confirmation, yield collection, or training
  collection from E25.

### E26 - RESOLVED 07-13: one-shot bounded semantic repair over 33 sealed decompositions
- Purpose: test the E24 preregistered feedback branch after E25 removed the
  parser-classification confound. This asks whether one bounded, symbolic
  diagnosis lets the same 7B generator repair its own typed DECOMPOSE draft.
  It is a generator-interface diagnostic, not controller evaluation, solve
  rate, a trajectory, reward evidence, or training data.
- Manipulation (single): retain each byte-exact E24
  `typed-claim-lens-v2.1` base request, including its legacy environment and
  receipt identity, and append one canonical semantic-feedback object. Then
  apply the returned draft post hoc under the corrected E25 v1.1 adapter. A
  preparation check found that 28/33 immutable historical pre-states carry v1
  verification receipts and therefore cannot be truthfully re-observed under
  the v1.1 verifier. This probe must not strip or migrate a receipt, substitute
  an environment digest, or otherwise relabel those states. Keeping the base
  request byte-exact makes the bounded feedback the only model-facing change;
  the corrected application arm is an explicitly cross-version diagnostic,
  not a production transition. The feedback object has exactly
  `schema_version`, `failed_domains`, `reason_code`, and `repair_constraint`.
  `failed_domains` is the exact ordered E24 domains array, `reason_code` is the
  corrected E25 bounded code, and `schema_version` is
  `decomposition-semantic-feedback-v1`.
- Frozen feedback map:
  - `unsupported_domain`: `Every child must use the target variable in one
    allowed relational form. Because a decomposition has two or more children,
    never use Reals as a child or as a scalar or comparison operand.`
  - `decomposition_domain_overlap`: `Make the child domains pairwise disjoint.
    Never include Reals beside another child. At every shared boundary, use the
    same finite scalar on both sides and include it in exactly one child: pair a
    strict upper side with an inclusive lower side, or an inclusive upper side
    with a strict lower side.`
  - `decomposition_domain_not_exhaustive`: `Make the union of child domains
    equal the target domain with no gap. Adjacent children must meet at the same
    boundary, included by exactly one child.`
  After the canonical object, append exactly: `Revise the failed domains once.
  Return only the same one-field JSON object required above.` Never expose a raw
  exception or unbounded adapter message. The complete map, selected object,
  and final request bytes must be digest-bound.
- Fixed source: E24 artifact
  `507c3b818ebee6c160bec021b84c60290f4af1a6fdcae725d2fa4efae74a5efc`
  and E25 artifact
  `f2be14e62cd04d680b454545cd253bfdb05c8ed5e404212b5b90bfcb7c8b1307`,
  including the exact 33 ordered drafts, source decisions, pre-states, actions,
  original first-attempt seeds, and corrected reason codes. Preparation must
  validate both canonical readers, strict E23 reconstruction, the E24 probe
  contract, the E25 rescore contract
  `ec307795e8dc39ae70515ca5036f5442f5784b72d40ed269fa3bb654ccbd842a`,
  and every paired row before selecting cases.
- Generator and fixed settings: Hugging Face Inference Providers routed to
  Together, model `Qwen/Qwen2.5-7B-Instruct:together`, asserted upstream Hub
  revision `a09a35458c702b33eeacc393d103063234e8bc28`, temperature 0.0,
  64 completion tokens, one parse retry, six 429/5xx backoff retries,
  120-second timeout, standard JSON-object mode, provider guided JSON off, and
  each row's original E24 seed. Served-weight revision remains asserted rather
  than provider-attested.
- Execution: issue exactly one independent repair generator call for each of
  the 33 sealed cases. A transport parse retry remains internal to that call;
  no semantic second retry is allowed. Make no controller call, do not feed a
  repaired result into another row, and create no trajectory or canonical
  training record. Retain every HTTP attempt, request, raw response, usage,
  decoded draft, corrected-profile algebra result, and per-problem result. Bind
  both the byte-exact E24 base-request digest and the corrected E25 application
  environment/profile; they must never be presented as one environment state.
- Primary gates: (1) at least **30/33** repairs decode to a typed
  DecompositionDraft; (2) at least **17/33** apply under the corrected v1.1
  adapter; and (3) at least one repair applies on **4/5 distinct problem ids**.
  The trusted adapter still requires exact parent coverage, pairwise
  disjointness, and at most one solution per child. Repeated rows from one task
  cannot satisfy the distinct-problem gate.
- Pre-registered reading: all three gates passing makes one bounded semantic
  repair viable. Then integrate it as a separately versioned, generator-
  internal DECOMPOSE semantic retry and run a separately registered same-five
  end-to-end confirmation from fresh v1.1 initial states. Never migrate these
  historical v1 receipts into that confirmation. If typed decoding passes but
  either semantic gate fails, retire full-domain authoring and test the
  separately versioned boundary-only payload next. If typed decoding fails,
  diagnose the repair wire before changing action semantics. No E26 result
  directly authorizes the 20-problem yield probe or training collection.
- Secondary diagnostics, with no decision threshold: corrected rejection-code
  migration, raw attempt outcomes, tokens, and cost. Do not compute or retain
  counterfactual closure, reward, learned-menu, refutation-route, or child-value
  fields: those would either require pretending that legacy receipt-bearing
  states are current v1.1 states or expand the registered manipulation with a
  new diagnostic solver surface.
- Frozen authorization: preregistration id `E26-v2-decompose-repair33`; 66
  source decisions, 33 DECOMPOSE cases, and five distinct problems. Source run
  `5efea19bd29619aba71d6d147135aababd7ded474230918b7d77b72d8b7368f6`,
  source diagnostic
  `0418043eb3abb50d86ad03c01946cc83602e76c8bd2c91f41defb87d3af6e369`,
  source environment
  `9316c93bc2cdf9b2ce507a989882af69669f8bf886e1f2ad6ce802c77dfdcfa7`,
  request environment
  `664deedf1caa44efff16daac1888c2e3966aa37e9a2631a62036643a9e681c71`,
  source collection
  `627a013c0aa4cc7c73e99f944209c1760a8b69b9f616030f8c8e8cf18cc321de`,
  and problem source
  `f3be145c1e536a98d4207c54ef85b5e0e516bce777c6ccd3d2a1f846c8881426`.
  E24 probe contract
  `3ef15fcccc0c3ce4fc109ae07ef3ad20ef29cb7677146b9fdce8f329352b019c`
  and case manifest
  `629205cc6f0047a19b81f31750c1fa0c74df1e30f2a142b026d1bdfcf100049d`.
  E25 rescore contract
  `ec307795e8dc39ae70515ca5036f5442f5784b72d40ed269fa3bb654ccbd842a`,
  case manifest
  `8955f15991ac60d9d99ec85399dcb573a178ab44cf4ddbdd07921e20d7f9fcb8`,
  and result manifest
  `fb2919c2a043a4a1e241f149716cd176a23e16a2d898a55f7b9e5c71faeddabd`.
  Corrected scoring environment
  `3396ce7464c8bf4a8cd60c5aa26172c02dc4ed4589ca7eb9e2bcfd906119ee54`
  and profile
  `42ee4bf1f481dd92bebac990b8fc0c95d280a4a5c1054b0b9dc4d5070f6581c8`.
  Feedback map
  `56b646c26c48f6c1f5ba675aa322f49bfb4919df6c0ceace9ca08d9f36c2ded7`,
  ordered case manifest
  `4377ceda24ff9731c5d28f3b4ff3678b513f7f016c1b7471d158356e0df982c1`,
  byte-exact base-request manifest
  `ddc854aef8431f78550315f45a99633e67bb32e8d77bef06d6e21b612167286b`,
  and repair-request manifest
  `0cc914d1c3ec1c685b419a76ce3af603296e10b2f07da742d08c3f3096795b30`.
  Generator identity
  `5fc23203d49417008fb63f5356f2a8e86cc249e77cbc3fac5b6a262858af8e58`,
  transport
  `b0e2806c348fcde14bf2bbc3c48656eb7547f3b7142afc93f99ba52c40896f2a`,
  generation settings
  `9059e1a6bbfb1422344ea9a6c9b9c5b85879f9fc0fe56de6ed2a4e1f846cd463`,
  and complete repair contract
  `69643e2c01b103b6c1bda3ec296a1519806bcdce92286981bf3cd812deb72110`.
  Planned artifact:
  `experiments/runs/2026-07-13-E26-v2-decompose-repair33.jsonl`. The contract
  binds the observed 2026-07-13 Together rate of $0.30 per million input and
  output tokens as an estimate from provider-reported usage, not a billing
  receipt. Expected HF spend is below $0.03. Zero-network contract replay and
  the strict 427-test reasoning-v2 suite passed before this authorization.
- Result: the strict reader published
  `experiments/runs/2026-07-13-E26-v2-decompose-repair33.jsonl`, SHA-256
  `7290e10733c1195edcea42684881afa4681f4c03f08e7aa7cf317a386623b1bd`,
  385,838 bytes and 129 canonical JSONL records, with no partial artifact.
  Trusted preparation, exact authorization, all 94 unique record ids, and the
  final summary independently revalidated. The run made 33 logical generator
  calls and 61 HTTP attempts: five attempts were accepted and 56 were schema
  errors; every other raw attempt outcome was zero. All 56 schema errors were
  HTTP-200 responses containing syntactically valid JSON with exactly the sole
  wrong field `failed_domains` instead of `domains`. Each of 28 failed calls
  used its registered parse retry; 27 retry pairs repeated byte-identical
  wrong content and one changed a boundary while retaining the wrong field.
  The call-level `transport_error` terminal label is client taxonomy for these
  schema failures, not evidence of a network or provider transport failure.
- Gate reading: typed decoding was **5/33**, below 30/33; corrected v1.1
  application was **0/33**, below 17/33; and applications covered **0/5**
  problems, below 4/5. All three preregistered gates failed. Each problem had
  one decoded repair. Those five were algebra-rejected: two
  `malformed_scalar`, one `decomposition_domain_overlap`, one
  `decomposition_domain_not_exhaustive`, and one `unsupported_domain`. This
  result does not establish whether bounded feedback can repair full-domain
  authoring because the feedback object collided with the output key contract
  on 28/33 calls. Follow the registered typed-failure branch: diagnose and
  version only the repair wire before changing DECOMPOSE action semantics.
- Usage and cost: all 61 attempts reported 53,209 prompt and 1,362 completion
  tokens, 54,571 total. The bound rate estimate is $0.0159627 input plus
  $0.0004086 output, **$0.0163713 total**. This is not a billing receipt. E26
  created no controller decision, trajectory, reward, closure, or training
  record and authorizes neither a data-yield probe nor training collection.

### E27 - RESOLVED 07-13: key-aligned feedback over 33 sealed repairs
- Purpose: isolate the E26 feedback/output key collision before changing
  DECOMPOSE semantics. E26 returned HTTP-200 valid JSON with the sole key
  `failed_domains` on all 56 rejected attempts. E27 asks whether aligning that
  one input key with the already required output key restores the typed repair
  wire and, secondarily, whether any resulting drafts apply. It remains a
  generator-interface diagnostic, not controller evaluation, solve rate,
  reward evidence, a trajectory, or training data.
- Manipulation (single): reuse every byte-exact E24 base request recovered and
  bound by E26, plus every E26 case, reason code, repair constraint, suffix,
  model setting, original seed, decoder, and corrected v1.1 scorer. Change
  only the canonical feedback object from
  `{"failed_domains": D, "reason_code": R, "repair_constraint": C,
  "schema_version": "decomposition-semantic-feedback-v1"}` to
  `{"domains": D, "reason_code": R, "repair_constraint": C,
  "schema_version": "decomposition-semantic-feedback-v2"}`. Here `D` is
  still the exact ordered rejected E24 domain array; this rename does not make
  it an accepted partition. Keep the exact E26 suffix: `Revise the failed
  domains once. Return only the same one-field JSON object required above.`
  The client must still accept only an exact one-field `domains` output and
  must not introduce a `failed_domains` alias, provider-guided JSON, a role
  split, an output example, or any action-semantic change in this arm.
  For every initial call body, preparation must prove that canonical E27 bytes
  equal the paired E26 initial repair body after exactly the feedback key
  substitution `failed_domains` to `domains` and feedback schema substitution
  v1 to v2. Transport parse retries necessarily append the preceding raw model
  response under the unchanged retry contract; they are retained diagnostics,
  not additional byte-paired prompt manipulations or independent cases.
- Fixed source: E26 artifact
  `7290e10733c1195edcea42684881afa4681f4c03f08e7aa7cf317a386623b1bd`
  and repair contract
  `69643e2c01b103b6c1bda3ec296a1519806bcdce92286981bf3cd812deb72110`.
  Preparation must reconstruct the exact E23-E26 trusted chain, validate the
  explicit E26 authorization and strict reader, and bind all 33 ordered E26
  cases and their first-attempt seeds before deriving E27 requests. The E26
  legacy request environment and corrected v1.1 scoring environment remain
  distinct and must stay separately labeled.
- Generator and execution: retain `Qwen/Qwen2.5-7B-Instruct:together`, asserted
  upstream revision `a09a35458c702b33eeacc393d103063234e8bc28`, Together via
  Hugging Face, temperature 0.0, 64 completion tokens, one internal parse
  retry, six 429/5xx backoff retries, 120-second timeout, standard JSON-object
  mode, guided JSON off, and each original E24 seed. Issue exactly one
  independent logical generator call for each of the 33 sealed cases, with no
  controller call, cross-row chaining, trajectory, or canonical training row.
  Retain all raw attempts and usage, and publish only after strict replay.
- Primary gates, over 33 logical cases rather than 61 correlated attempts:
  (1) at least **30/33** terminal outputs have the exact one-field key set
  `domains` and decode to a typed DecompositionDraft; (2) at least **17/33**
  apply under the corrected v1.1 adapter; and (3) applications cover at least
  **4/5 distinct problem ids**. Repeated rows cannot satisfy the problem gate.
- Pre-registered reading: if the typed gate fails, key alignment did not repair
  the feedback wire; diagnose the feedback channel again without changing
  DECOMPOSE semantics. If the typed gate passes but either semantic gate
  fails, retire full-domain authoring and test a separately versioned
  boundary-only payload. If all three pass, separately integrate an internal
  semantic retry and run a fresh-v1.1 same-five end-to-end confirmation. No E27
  result directly authorizes a data-yield probe or training collection.
- Secondary diagnostics, with no threshold: exact copies of the input domain
  array, raw wrong-key shapes, retry outcomes, corrected reason migration,
  tokens, and estimated cost. Never treat retries as independent cases or
  accept `failed_domains` as a production alias. Planned artifact:
  `experiments/runs/2026-07-13-E27-v2-decompose-repair-key33.jsonl`. Expected HF
  spend is below $0.03.
- Frozen authorization: preregistration id `E27-v2-decompose-repair-key33`; 66
  source decisions, 33 ordered repair cases, and five distinct problems.
  Source environment
  `9316c93bc2cdf9b2ce507a989882af69669f8bf886e1f2ad6ce802c77dfdcfa7`,
  E24/E26 request environment
  `664deedf1caa44efff16daac1888c2e3966aa37e9a2631a62036643a9e681c71`,
  corrected scoring environment
  `3396ce7464c8bf4a8cd60c5aa26172c02dc4ed4589ca7eb9e2bcfd906119ee54`,
  and corrected profile
  `42ee4bf1f481dd92bebac990b8fc0c95d280a4a5c1054b0b9dc4d5070f6581c8`.
  E26 artifact
  `7290e10733c1195edcea42684881afa4681f4c03f08e7aa7cf317a386623b1bd`,
  repair contract
  `69643e2c01b103b6c1bda3ec296a1519806bcdce92286981bf3cd812deb72110`,
  case manifest
  `4377ceda24ff9731c5d28f3b4ff3678b513f7f016c1b7471d158356e0df982c1`,
  result manifest
  `a82e87becbfd0e00c125784702d2332f90dcee4a743ded4c361e6bad54849536`,
  and validated summary
  `b0c8dbe1f6d3ed9d6ab43f2b66f292ce6fb0aacf0d01e21a4d3f40c88b7e4d6d`.
  E27 feedback map
  `1ccb32d1dab579315dc2d280e4601fb622e9e61355998ab2cec9fe7d04206023`,
  ordered case manifest
  `5a6b78190e0a54a39ad32e23e6570cdb93c8adc14c52efc3f8696574e2999977`,
  byte-exact base-request manifest
  `ddc854aef8431f78550315f45a99633e67bb32e8d77bef06d6e21b612167286b`,
  exact E26 repair-request manifest
  `0cc914d1c3ec1c685b419a76ce3af603296e10b2f07da742d08c3f3096795b30`,
  and key-aligned E27 request manifest
  `cde25cee43a8a6250b974ad21317ea31606eabbb1ec4a0d64434c61ce33e8f6f`.
  Generator identity
  `5fc23203d49417008fb63f5356f2a8e86cc249e77cbc3fac5b6a262858af8e58`,
  transport
  `b0e2806c348fcde14bf2bbc3c48656eb7547f3b7142afc93f99ba52c40896f2a`,
  generation settings
  `9059e1a6bbfb1422344ea9a6c9b9c5b85879f9fc0fe56de6ed2a4e1f846cd463`,
  and complete E27 contract
  `e3c4fc625c1cb79e2217a2fb86d8603a3b9955e9bac1fd2e59cb47fada078696`.
  The contract binds the same observed 2026-07-13 Together estimate of $0.30
  per million input and output tokens, not a billing receipt. Zero-network
  contract replay, 12 focused tests, and the full 439-test reasoning-v2 suite
  passed before this authorization. The final and partial output paths were
  absent at authorization time.
- Result: strict publication produced
  `experiments/runs/2026-07-13-E27-v2-decompose-repair-key33.jsonl`, SHA-256
  `56f0e55ce70fd7ad327126a8f2e43f7a4ae011ea482e4d473f2f3032f5bbf55c`,
  265,455 bytes and 103 canonical JSONL records, with no partial artifact.
  Both frozen authorizations, the complete E23-E26 reconstruction, 68 unique
  record ids, and the E27 strict reader independently revalidated.
- Gate reading: the wire was usable on **33/33** cases and exact one-field
  typed decoding reached **32/33**, passing the 30/33 gate. Corrected v1.1
  application remained **0/33**, below 17/33, and covered **0/5** problems,
  below 4/5. The two semantic gates failed. All 32 decoded drafts were
  algebra-rejected: 15 `decomposition_domain_overlap`, 10
  `unsupported_domain`, four `malformed_scalar`, and three
  `decomposition_domain_not_exhaustive`. The remaining response used
  `domains` plus `reason_code` and `schema_version`; the exact one-field stage
  correctly classified it as a payload error. No response used the old
  `failed_domains` key. Eight responses copied the complete rejected input
  domain array exactly and all eight remained rejected.
- Attempts and cost: 33 logical calls made 35 HTTP-200 attempts. There were 33
  accepted attempts and two truncated-JSON parse errors, each followed by the
  authorized seed-offset retry and an accepted result. Every logical call
  ended transport-accepted. All attempts reported 29,069 prompt and 846
  completion tokens, 29,915 total. The bound estimate is $0.0087207 input plus
  $0.0002538 output, **$0.0089745 total**, not a billing receipt.
- Verdict: the registered key alignment repaired the feedback wire but did not
  repair full-domain authoring. Follow the registered semantic-failure branch:
  retire LLM-authored domain strings and next test a separately versioned
  boundary-only payload whose exact disjoint and exhaustive domains are
  constructed by trusted environment code. E27 created no controller decision,
  closure, reward, trajectory, or training record and authorizes neither a
  data-yield probe nor training collection.

### E28 - INVALIDATED 07-13: trusted topology from model-authored boundaries
- Purpose: test the boundary-only branch selected by E27 without resuming
  LLM-authored domain strings. E28 asks the same frozen 7B generator to choose
  only scalar separators. Trusted environment code owns interval topology and
  endpoint inclusion, while the unchanged corrected v1.1 symbolic adapter
  remains the only authority on child atomicity. This is a representation and
  GenerationContract diagnostic, not controller evaluation, reward evidence,
  a trajectory, solve rate, or training data.
- Manipulation (single): return to the 33 sealed E24 DECOMPOSE pre-states,
  actions, first-attempt seeds, byte-exact lens views, model distribution, and
  transport. Bind E27 only as the result that authorized this branch. Replace
  the full-domain DECOMPOSE output instruction and schema with the separately
  versioned `decomposition-boundaries-v1` payload. Do not append E26/E27 repair
  feedback. Do not change the controller, action pair, learned menu, reward,
  corrected v1.1 scorer, or any scalar returned by the model.
- Exact output schema: one object with the sole required field `boundaries`, an
  array of one to five strings, `additionalProperties: false`. The strict
  decoder additionally requires ASCII strings and the existing finite rational
  scalar grammar: digits, spaces, `+`, `-`, `*`, `/`, parentheses, and decimal
  points, with neither `**` nor `//`.
- Exact boundary instruction: `Return a JSON object with exactly one field
  named boundaries. Set boundaries to an array of 1 to 5 distinct ASCII finite
  rational scalar strings, ordered from least to greatest. A scalar may contain
  only digits, spaces, +, -, *, /, parentheses, and decimal points, with no **
  or //. Do not use x, Reals, comparisons, interval notation, roots, infinity,
  or any other field. A boundary is a separator, not a solution. Every boundary
  must lie strictly inside the target domain. Include a boundary strictly
  between every adjacent pair of distinct in-domain solutions, using the fewest
  boundaries that satisfy this condition, so each induced interval contains at
  most one solution. If there are fewer than two in-domain solutions, still
  return one interior boundary. Return only one JSON object.` There is no output
  example, repair suffix, provider-guided JSON, or scalar coercion.
- Trusted construction: parse the parent domain under corrected v1.1 and each
  boundary with the exact finite rational parser. Reject a nonfinite,
  duplicate, non-increasing, or non-interior value. Never sort, deduplicate,
  clip, move, add, or drop a model value. For ordered `b1, ..., bk`, construct
  the parent intersection with `(-inf, b1)`, `[b1, b2)`, ..., `[bk, inf)`,
  preserving any parent endpoint inclusion. Render the resulting sets in the
  existing corrected grammar using each exact accepted model-authored boundary
  lexeme. Parsed semantic values validate finiteness, uniqueness, order, and
  interior membership only; they may be retained as diagnostics but must never
  replace or normalize a lexeme in a constructed child. Create `k+1` child
  obligations with the parent variable and equation, `branch` objective,
  exhaustive-case connective, and deterministic `case 1, ...` statements. The
  constructor must not solve the equation or choose a boundary. The unchanged
  corrected v1.1 coverage and atomicity check then accepts or rejects the draft.
- Constructor gate before hosted execution: on the five exact sealed parent
  tasks, non-prompt oracle boundaries `4`, `6`, `2`, `4`, and `-3`, in source
  order, must each construct two disjoint exhaustive children and pass corrected
  v1.1 atomicity. This validates expressivity and implementation only; it is not
  a model baseline or empirical result. Any post-construction
  `unsupported_domain`, `malformed_scalar`, overlap, or nonexhaustive code is an
  implementation fault and must prevent publication rather than count as a
  generator failure.
- Fixed source and generator: strict-reconstruct E23-E25 and the 33 ordered E24
  DECOMPOSE cases; strict-read E27 artifact
  `56f0e55ce70fd7ad327126a8f2e43f7a4ae011ea482e4d473f2f3032f5bbf55c`
  under contract
  `e3c4fc625c1cb79e2217a2fb86d8603a3b9955e9bac1fd2e59cb47fada078696`
  solely to bind the branch reading. Retain
  `Qwen/Qwen2.5-7B-Instruct:together`, asserted upstream revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, Together through Hugging Face,
  temperature 0.0, 64 completion tokens, one parse retry, six 429/5xx backoff
  retries, 120-second timeout, standard JSON-object mode, guided JSON off, and
  each original E24 seed. Issue exactly 33 independent logical calls, retain
  every raw attempt and usage record, and publish only after strict replay.
  The E24 request states and environment remain byte-exact legacy
  receipt-bearing sources, while application is explicitly post hoc under the
  corrected E25 v1.1 scoring environment. Keep those environments separately
  bound and labeled; never re-observe, strip, migrate, or relabel a receipt.
- Primary gates: (1) at least **30/33** responses have the exact one-field
  boundary payload and pass finite-rational, ordering, uniqueness, and interior
  validation; (2) at least **17/33** trusted constructed drafts apply under the
  corrected v1.1 adapter; and (3) applications cover at least **4/5 distinct
  problem ids**. Count logical cases, not transport retries. Report one-solution
  and multi-solution parent tasks separately without adding another threshold.
- Pre-registered reading: if the boundary payload gate fails, diagnose the new
  wire/scalar contract without changing topology. If it passes but either
  semantic gate fails, the model cannot reliably select separators on these
  cases; do not resume full-domain prompting or collect training data. If all
  gates pass, integrate the versioned boundary DECOMPOSE path and run a
  separately registered same-five end-to-end confirmation from fresh v1.1
  initial states before any 20-problem yield probe. No E28 outcome directly
  authorizes training.
- Secondary diagnostics, with no threshold: boundary arrays, validation
  failures, constructed domains, corrected atomicity reasons, per-problem and
  one-solution versus multi-solution counts, retries, tokens, and estimated
  cost. Planned artifact:
  `experiments/runs/2026-07-13-E28-v2-boundary-decompose33.jsonl`. Expected HF
  spend is below $0.03. Before execution, freeze every source, environment,
  model, transport, instruction/schema, oracle-constructor manifest, ordered
  case/request manifest, complete E28 contract, and output-path binding here.
- Frozen authorization: preregistration id
  `E28-v2-boundary-decompose33`; 66 source decisions, 33 ordered DECOMPOSE
  cases, and five distinct problems. Source environment
  `9316c93bc2cdf9b2ce507a989882af69669f8bf886e1f2ad6ce802c77dfdcfa7`,
  legacy E24 request environment
  `664deedf1caa44efff16daac1888c2e3966aa37e9a2631a62036643a9e681c71`,
  corrected scoring environment
  `3396ce7464c8bf4a8cd60c5aa26172c02dc4ed4589ca7eb9e2bcfd906119ee54`,
  and corrected profile
  `42ee4bf1f481dd92bebac990b8fc0c95d280a4a5c1054b0b9dc4d5070f6581c8`.
  E24 artifact
  `507c3b818ebee6c160bec021b84c60290f4af1a6fdcae725d2fa4efae74a5efc`,
  probe contract
  `3ef15fcccc0c3ce4fc109ae07ef3ad20ef29cb7677146b9fdce8f329352b019c`,
  and case manifest
  `629205cc6f0047a19b81f31750c1fa0c74df1e30f2a142b026d1bdfcf100049d`.
  Branch authorization is bound to E27 artifact
  `56f0e55ce70fd7ad327126a8f2e43f7a4ae011ea482e4d473f2f3032f5bbf55c`,
  contract
  `e3c4fc625c1cb79e2217a2fb86d8603a3b9955e9bac1fd2e59cb47fada078696`,
  case manifest
  `5a6b78190e0a54a39ad32e23e6570cdb93c8adc14c52efc3f8696574e2999977`,
  and validated summary
  `e936a50d74cbfa67559d248e1c12ef3db6021408bbf333d8ae9ebc350429726b`.
  The exact boundary instruction digest is
  `6004c2f9324b15321228201d96bdbdc421052811aa70064a33d27baf629b47ec`,
  payload-schema digest
  `58743116168367a400b9a15af8d336caef850f9e60be62e2fa01782d4e9c31e9`,
  trusted-constructor digest
  `6c9873799b9b1dd54eb11c523823e2cac0fb57cdc46a78c2366915f60aa20dfb`,
  and five-case oracle manifest
  `b4af1720ed561f3c65fa44f3092e8aada05130121151aa91fe24ecbf1e5fe3d3`.
  The ordered E28 case manifest is
  `c8b89f7f25a07d3519404e4e30e7d477bf7be6a8f108e4386e22a9ce26ea5052`,
  byte-exact E24 base-request manifest
  `ddc854aef8431f78550315f45a99633e67bb32e8d77bef06d6e21b612167286b`,
  and boundary-request manifest
  `d78bb4976c3f0b8098a87260abc1d96bed4c599cb89a4b1b4f668558aa436c3d`.
  Generator identity
  `5fc23203d49417008fb63f5356f2a8e86cc249e77cbc3fac5b6a262858af8e58`,
  transport
  `b0e2806c348fcde14bf2bbc3c48656eb7547f3b7142afc93f99ba52c40896f2a`,
  inherited generation settings
  `9059e1a6bbfb1422344ea9a6c9b9c5b85879f9fc0fe56de6ed2a4e1f846cd463`,
  and complete E28 contract
  `6dbfd0e04b0c702ad1cc5342868d932a97cb56c904bb98bc7fb1dc45e993ca06`.
  The contract separately binds boundary lens
  `typed-claim-lens-v2.2-boundaries` while retaining the inherited v2.1
  generation settings and byte-exact legacy view. The planned final and
  partial output paths were absent at authorization time. Two independent
  zero-network contract replays agreed. The 53-test focused E28 suite, the
  23-test E26/E27 regression, and the full repository suite with 869 passed
  and one skipped all passed before authorization. The contract binds the
  observed 2026-07-13 Together estimate of $0.30 per million input and output
  tokens, not a billing receipt.
- Invalidated artifact: the authorized runner published
  `experiments/runs/2026-07-13-E28-v2-boundary-decompose33.jsonl`, SHA-256
  `f42e36bc426169ca2edb25237c26443e40eea7952b10242d25590e45bb96f404`,
  212,471 bytes and 101 canonical newline-terminated records, with no partial
  artifact. It recorded 33 calls and 33 HTTP-200 attempts, all as accepted,
  plus a post-coercion summary of 33 typed payloads, 32 applications, and five
  applied problems. Those three counts are invalid and must not be cited as an
  E28 gate result.
- Invalidation cause: the shared client schema validator carried a legacy
  numeric-to-string coercion for every JSON-schema string field, including
  array items. This mutated parsed replies before the boundary decoder and
  before strict replay. E28 explicitly prohibited scalar coercion and required
  exact model-authored string lexemes. The reader reproduced the same hidden
  coercion, so internal replay consistency did not detect the contract breach.
  For example, raw `{"boundaries":[-5.0,1.0]}` became stored
  `{"boundaries":["-5","1"]}`, changing both JSON type and lexeme.
- Raw no-coercion audit: all 33 first attempts otherwise used the sole
  `boundaries` field, valid cardinality, increasing finite rational values, and
  interior boundaries. Seventeen arrays were all-number, eight were mixed,
  and eight were all-string. In total, 42/71 items were JSON numbers. Under the
  registered exact schema, only **8/33** attempts were accepted and **25/33**
  were schema errors requiring the authorized parse retry. All eight accepted
  attempts belonged to `cw1-52e3b6b3ec25` and applied, giving first-attempt
  diagnostics of 8 typed, 8 applied, and one problem. These are not terminal
  E28 measures: the 25 required retry responses were never requested, so the
  registered logical-case gates, final usage, and final cost cannot be
  reconstructed from this artifact.
- Invalid-run transport and spend: the 33 invalid attempts reported 21,570
  prompt and 384 completion tokens, 21,954 total. The bound estimate is
  $0.006471 input plus $0.0001152 output, **$0.0065862 total**, not a billing
  receipt. The artifact is retained as a quarantined protocol diagnostic and
  creates no controller decision, reward, trajectory, closure, or training
  row.
- Verdict: E28 has no valid primary-gate outcome. Do not integrate the
  boundary path from its post-coercion summary and do not collect training
  data. First version and bind an exact no-coercion schema-validation contract,
  add raw-number and mixed-array regression tests to both live and replay
  paths, and register a new protocol-corrected replication. That replication
  must retain the same 33 initial request bodies, seeds, model, transport,
  retry budget, constructor, scorer, thresholds, and reading while allowing
  the already specified retry mechanism to run on numeric first replies. Bind
  this invalid artifact and its raw audit for disclosure, never as a source of
  accepted boundary payloads.

### E29 - RESOLVED 07-13: exact no-coercion boundary replication
- Purpose: run the protocol-corrected replication required by the E28
  invalidation. E29 tests the same boundary-only representation on the same 33
  sealed DECOMPOSE cases, with the intended exact string schema enforced before
  any payload reaches trusted construction. This is a GenerationContract
  diagnostic, not controller evaluation, reward evidence, a trajectory, solve
  rate, or training data. E29 is not an independent fresh sample because its
  requests repeat E28 and its design follows inspection of E28 raw responses.
- Manipulation (single): replace E28's unintended legacy numeric-to-string
  schema-validation behavior with the separately versioned, nonmutating
  `json-schema-strict-types-v1` mode. Retain every initial request byte, seed,
  model identity, transport setting, response format, lenient JSON object
  parser, boundary instruction and schema, trusted constructor, corrected v1.1
  symbolic scorer, retry budget, gate, and reading. Numeric or mixed boundary
  arrays are schema errors. They are never converted, normalized, decoded,
  constructed, or scored.
- Exact validation contract: require an exact built-in JSON object with the
  sole required `boundaries` field, an exact built-in array of one to five
  exact strings, and no additional fields. Enforce required fields,
  `additionalProperties: false`, array cardinality, and item types recursively,
  including rejection of Python boolean/integer aliases. Validation never
  mutates its input. ASCII rational grammar, order, uniqueness, and strict
  parent-domain interior remain post-transport typed checks; they do not cause
  a semantic retry.
- Exact retry consequence: retain the already frozen one-parse-retry policy.
  On `parse_error`, `provider_response_error`, `schema_error`, or
  `transport_error`, append the current run's nonempty raw reply verbatim as an
  assistant message, append the literal generic correction `Your previous
  response could not be parsed. Return a single valid JSON object matching the
  schema. Return JSON only.`, and use seed `initial_seed +
  parse_retry_index`. The budget is one and there is no semantic retry. A
  numeric second reply exhausts the logical call. Exercising this unchanged
  policy is a deterministic consequence of the validator manipulation, not a
  second manipulation.
- Quarantined E28 control: strict-read the E28 header, request bodies, seeds,
  provider response envelopes, raw attempt records, and call/case bindings
  under the exact E28 artifact hash. Reclassify raw first replies without
  coercion. Bind only per-case hashes, JSON type labels, exact outcomes, and an
  aggregate audit into E29 runnable state. Never use E28's stored
  `boundary_payload`, decoded payload, construction, application, legacy
  summary counts, oracle values, or raw reply strings as an E29 model input,
  retry, expected payload, threshold, or scoring result. For the eight raw
  all-string controls, independently decode, construct, and symbolically score
  the raw strict payload to validate the audit rather than trusting E28 derived
  fields.
- Fixed source and generator: reconstruct E23-E27, reconstruct the exact E28
  v1 contract at its original output path, and verify invalidated E28 artifact
  `f42e36bc426169ca2edb25237c26443e40eea7952b10242d25590e45bb96f404`.
  Retain the same 33 E28 initial request bodies and first-attempt seeds,
  `Qwen/Qwen2.5-7B-Instruct:together`, asserted revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, Together through Hugging Face,
  temperature 0.0, 64 completion tokens, one parse retry, six 429/5xx backoff
  retries, 120-second timeout, standard JSON-object mode, and guided JSON off.
  Issue 33 logical calls, retain every physical attempt and usage record, and
  publish only after strict raw-transcript replay.
- Primary gates, unchanged from E28: (1) at least **30/33** logical cases end
  with the exact one-field boundary payload and pass rational, order,
  uniqueness, and interior validation; (2) at least **17/33** trusted drafts
  apply under the corrected v1.1 symbolic adapter; and (3) applications cover
  at least **4/5 distinct problem ids**. Count logical cases, not retries.
  Report one-solution and multi-solution parents separately without another
  threshold.
- Pre-registered reading: if the typed gate fails, the exact one-retry wire
  contract is inadequate; do not coerce values, change topology, or collect
  training data. If typed decoding passes but either semantic gate fails, the
  model still cannot select separators reliably; retire this branch without
  resuming full-domain authoring. If all three gates pass, integrate only the
  separately versioned boundary path and register a fresh corrected-v1.1
  same-five end-to-end confirmation before any 20-problem yield probe. No E29
  outcome directly authorizes training.
- Secondary diagnostics, with no threshold: first-attempt accepted versus
  schema-error counts; all-number, mixed, all-string, and unclassified first
  arrays; numeric item count; schema retries repaired versus exhausted; E28
  versus E29 first-raw content-hash matches; terminal payloads and boundary
  lexemes; construction and corrected atomicity reasons; per-problem and
  one-solution versus multi-solution counts; attempts, usage, and estimated
  cost. A retry echoes only the fresh E29 failed reply, but that echo may make
  quoting the same separators easier. Therefore success measures the corrected
  whole protocol, not an independent second semantic choice. The 33 rows are
  paired repeated states over five problems, not 33 independent problems.
  Provider-served weights remain asserted rather than attested.
- Planned artifact:
  `experiments/runs/2026-07-13-E29-v2-boundary-exact-schema33.jsonl`. Expected
  HF spend is below $0.03 at the same observed $0.30 per million input and
  output token estimate, not a billing receipt. The final and partial paths
  must both be absent at authorization.
- Frozen bindings, independently reconstructed three times: source environment
  `9316c93bc2cdf9b2ce507a989882af69669f8bf886e1f2ad6ce802c77dfdcfa7`,
  request environment
  `664deedf1caa44efff16daac1888c2e3966aa37e9a2631a62036643a9e681c71`,
  corrected scoring environment
  `3396ce7464c8bf4a8cd60c5aa26172c02dc4ed4589ca7eb9e2bcfd906119ee54`,
  and corrected profile
  `42ee4bf1f481dd92bebac990b8fc0c95d280a4a5c1054b0b9dc4d5070f6581c8`.
  E28 contract
  `6dbfd0e04b0c702ad1cc5342868d932a97cb56c904bb98bc7fb1dc45e993ca06`,
  header
  `e9f0e4429055d24c0798a043f26d3eff534b31383107ab18e034fba0902f5464`,
  case manifest
  `c8b89f7f25a07d3519404e4e30e7d477bf7be6a8f108e4386e22a9ce26ea5052`,
  boundary-request manifest
  `d78bb4976c3f0b8098a87260abc1d96bed4c599cb89a4b1b4f668558aa436c3d`,
  and invalid legacy summary
  `df2f2172196415ac911c32f72f1bf62ba43a849192d9b734e18a40fbd14fc3d5`.
  Raw audit
  `3eb3ddff5433a49e2c80f042407335388ad30a556f1dd19fce2e1fbd7bb0f9d7`
  records 33 attempts, 17 all-number arrays, eight mixed, eight all-string,
  42/71 numeric items, eight exact accepts, 25 schema errors with missing
  retries, and eight independently applied controls over one problem. Raw
  control manifest
  `da475c66b2365958826e503315603ac053b76e29d4a2ca220db5d327f7623704`.
  Exact validation contract
  `e1c539bb0a91b340379b2d7c46bdc12fa86040f88de381f8f138ef73704d05fe`,
  retry-policy contract
  `07e11c7b0e40bd0e9ce4bc6188a03b6289556dc31bffeb5bb1cd1f6588c325fd`,
  E29 case manifest
  `9921b96955a3953c379403bf5bfb3986f0f30a18b53a0d93191e58086c4ec863`,
  unchanged base-request manifest
  `ddc854aef8431f78550315f45a99633e67bb32e8d77bef06d6e21b612167286b`,
  unchanged boundary-request manifest
  `d78bb4976c3f0b8098a87260abc1d96bed4c599cb89a4b1b4f668558aa436c3d`,
  generator
  `5fc23203d49417008fb63f5356f2a8e86cc249e77cbc3fac5b6a262858af8e58`,
  transport
  `b0e2806c348fcde14bf2bbc3c48656eb7547f3b7142afc93f99ba52c40896f2a`,
  generation settings
  `9059e1a6bbfb1422344ea9a6c9b9c5b85879f9fc0fe56de6ed2a4e1f846cd463`,
  and complete E29 contract
  `c6cbe4410ea63e6b7b0329ac7c681d721faac16a9472064e1f8bc05cc4e01700`.
  The preregistered E29 header for the authorization below is
  `0b3aa69dcd916805c32542ed09d34f64d004cffc36e1b716859e03454526c58e`.
- Frozen authorization: preregistration id
  `E29-v2-boundary-exact-schema33`, 66 source decisions, 33 ordered
  DECOMPOSE cases, and five distinct problems. Three independent zero-network
  reconstructions agreed on every source, audit, request, validator, retry,
  case, generator, transport, and complete-contract digest. The focused
  58-attempt corrected-replication simulation exercised all 25 numeric/mixed
  schema retries, exact assistant echoes, seed offsets, terminal payload
  identity, strict raw replay, retry exhaustion, and artifact tamper rejection.
  The adjacent E28/E29 and client matrix passed 97 tests. The full repository
  suite passed 885 tests with one CUDA-only skip and one existing PyTorch
  warning. Compile/import, diff whitespace, and forbidden em-dash checks passed.
  The planned final and partial paths were absent after all tests and at
  authorization. No HF key was read and no hosted request occurred during
  preparation or testing. E28's artifact, runner, and under-specified v1
  contract remain permanently quarantined. This authorization permits exactly
  one hosted E29 execution under the frozen CLI and bindings above; it does not
  authorize any prompt, parser, topology, threshold, scorer, or retry change.
- Result: strict publication produced
  `experiments/runs/2026-07-13-E29-v2-boundary-exact-schema33.jsonl`, SHA-256
  `abb5c47c5366fea62fd286e20c0b4c6d5d12993781924c31b1834165577f02b2`,
  319,380 bytes and 126 canonical newline-terminated ASCII JSONL records, with
  no partial artifact. The record surface is exactly one frozen header, 58
  generator attempts, 33 generator calls, 33 cases, and one summary. The
  published header hash is
  `0b3aa69dcd916805c32542ed09d34f64d004cffc36e1b716859e03454526c58e`.
  Independent source reconstruction, provider-envelope projection, strict
  transcript replay, construction, symbolic scoring, and summary derivation
  all passed byte-exactly.
- Exact retry result: the 33 first replies comprised 17 all-number, eight
  mixed, and eight all-string arrays, with 72 boundary items and 42 JSON
  numbers. Exact validation accepted eight and classified 25 as schema errors.
  Every failed reply was echoed verbatim with the frozen correction and seed
  plus one. The 25 correction replies comprised 18 all-number and seven mixed
  arrays, 61 items and 41 numbers; all 25 were schema errors again. Thus 25
  retries were exhausted and zero were repaired. All 58 attempts were HTTP 200
  with reported usage; no HTTP, parse, provider-response, or network-transport
  error occurred. The 25 unsuccessful logical-call records use the shared
  generic `transport_error` label, but each final physical outcome is
  `schema_error`.
- Paired control diagnostic: E29 first raw content matched E28 in 30/33 cases.
  Index 6 changed `[-3,"1","5"]` to `[-3,"1","5","7"]`, index 19 changed
  `["1.0"]` to `["1"]`, and index 25 changed `[-4.5,"-1.0"]` to
  `[-4.5,"-1.5"]`. These have no threshold and do not change the reading. The
  extra index-6 value explains E29's 72 first items versus E28's 71. Numeric
  and mixed replies never entered payload, construction, or scoring fields;
  accepted string lexemes remained exact throughout.
- Gate reading: wire usability and exact typed decoding reached **8/33**, below
  30/33. All eight trusted drafts applied under the symbolic verifier, so
  symbolic application reached **8/33**, below 17/33, and covered **1/5**
  problems, below 4/5. All three primary gates failed. There were no algebra
  rejections: 25 cases did not run because their one strict retry exhausted.
  Every accept was one of indices 16-23 for `cw1-52e3b6b3ec25`; its eight
  cases were typed and applied with accepted coverage. One-solution parents
  reached 8/16 over one of two problems. Multi-solution parents reached 0/17
  over three problems. The other four problem ids each reached zero.
- Attempts and cost: all 58 attempts reported 38,774 prompt and 695 completion
  tokens, 39,469 total. The bound estimate is $0.0116322 input plus $0.0002085
  output, **$0.0118407 total**, not a billing receipt. Cumulative hosted spend
  estimated through E29 is approximately **$0.0743493**.
- Verdict: the preregistered exact one-retry string wire is inadequate for this
  7B generator. Do not coerce numeric values, integrate this boundary path,
  change trusted topology, resume full-domain authoring, or collect controller
  or training data from E29. E29 created no controller decision, reward,
  trajectory, closure, or training row. Any further scalar-representation or
  transport design is a new GenerationContract manipulation requiring its own
  version, constructor proof, tests, and preregistration; E29 authorizes no
  follow-on hosted experiment.
- Post-resolution integration audit, offline only: canonical generator and
  teacher wrappers plus diagnostic replay had still inherited the low-level
  client's legacy schema default. The engineering fix versions the role-local
  wire ids: current `typed-policy-generation-v2.1` and
  `controller-choice-wire-v2.1` select nonmutating strict validation, while
  their prior `v2` ids explicitly select legacy coercion only for exact replay.
  Runtime and diagnostic replay derive the mode from those ids and reject
  unknown bindings. Outer environment and collection record schemas were not
  changed. Strict parsing also rejects non-JSON numeric constants and injected
  nonfinite floats; the mock backend reproduces selected legacy coercion on an
  isolated payload. A clean 109-test historical artifact and hash matrix
  passed after the fix, preserving the frozen request and artifact bindings.
  The final repository suite passed 891 tests with one CUDA-only skip and the
  existing PyTorch warning. This audit is not a new experiment, made no hosted
  request, spent no HF credit, and does not alter E29's measures, result, or
  verdict.

### E30 - SUPERSEDED BEFORE EXECUTION 07-14: strong HF teacher paired current-v2 screen
- Resolution: the user redirected the work to a separately versioned,
  permissive workspace experiment before either E30 arm was executed. No E30
  final or partial artifact exists, no hosted request was made, and no credit
  was spent under this authorization. E30 remains here as the immutable plan
  that was registered, but it is closed and cannot authorize a later run.
- Purpose: directly test whether substantially greater teacher capability
  improves end-to-end symbolic success under the current v2 state and action
  contracts. This is a five-problem paired screen, not an accuracy estimate,
  training collection, direct-prompt comparison, or RL result. It replaces
  generator-interface diagnostics as the immediate experimental priority.
- Manipulation (single): replace only the controller model stack. The control
  arm uses OpenAI `gpt-4.1-mini-2025-04-14`. The treatment arm uses
  `Qwen/Qwen3-235B-A22B-Instruct-2507:deepinfra` through Hugging Face Inference
  Providers, with asserted Hub revision
  `ac9c66cc9b46af7306746a9250f23d47083d689e`. The treatment has 235B total
  parameters and 22B activated parameters. Provider-served weights are
  asserted configuration metadata, not attested identities. Controller model,
  provider, and endpoint form one inseparable treatment stack; all request,
  environment, generator, transition, reward, and evaluation settings are
  fixed between arms.
- Fixed source: records 0-4 of `datasets/casework-v1/train.jsonl`, the same five
  ordered problem ids as E23. Problem-source SHA-256 is
  `f3be145c1e536a98d4207c54ef85b5e0e516bce777c6ccd3d2a1f846c8881426`.
  These repeated problems make E30 a paired engineering screen and prohibit a
  population-performance claim.
- Fixed current environment: `typed-claim-lens-v2.1`, strict
  `typed-policy-generation-v2.1`, corrected v1.1 casework symbolic contracts,
  four learned verbs, deterministic closure, `symbolic-only-v1` reward, turn
  limit 16, and environment SHA-256
  `3e9a1e24a702e0a618f9e40305f47cd96c28f8637bdd8d1ad727955ff8903c42`.
  Both controllers use `typed-controller-presentation-v2` and strict
  `controller-choice-wire-v2.1`; the prompt and full canonical observation are
  byte-identical whenever the trajectory state is identical.
- Fixed generator: Hugging Face Inference Providers routed to Together,
  `Qwen/Qwen2.5-7B-Instruct:together`, asserted Hub revision
  `a09a35458c702b33eeacc393d103063234e8bc28`. Both roles use temperature 0.0,
  64 completion tokens, base seed 0 with turn-and-retry offsets, one parse
  retry, six 429/5xx backoff retries, 120-second timeout, standard JSON-object
  mode, provider guided JSON disabled, and nonmutating strict schema
  validation. The same label-free source adapter, legal menu, fallback policy,
  action algebra, symbolic verifier, replay, and publication boundary apply to
  both arms.
- Frozen collection bindings: mini control collection SHA-256
  `197522dea0c6985b7ab3505b1492c7c7b615cdc0ffc608ec1e9174ef5b58be29`;
  Qwen3 treatment collection SHA-256
  `690d4e5e41e3f39ba561d431907724f09510b8bbda11302d4246ae32bd1db8a7`.
  Zero-network preparation produced the same source and environment digests in
  both arms without reading either API key.
- Planned artifacts:
  `experiments/runs/2026-07-14-E30-teacher-mini-current5.jsonl` and
  `.diagnostics.jsonl`, plus
  `experiments/runs/2026-07-14-E30-teacher-qwen3-235b-current5.jsonl` and
  `.diagnostics.jsonl`. All final and sibling `.partial` paths must be absent
  before authorization. Each arm publishes only after role-bound raw-call
  validation, exact symbolic replay, and complete ordered source coverage.
- Primary measure and gate: compare symbolic root solves over the five paired
  problems. The treatment screen passes only if Qwen3 solves at least **2/5**
  and at least **one more problem** than the freshly run mini control. Count a
  solve only from terminal +1 under `symbolic-only-v1`; no label, LLM judge,
  learned critic, or intermediate heuristic can supply reward.
- Pre-registered reading: if the treatment passes, do not claim accuracy;
  register a 20-problem paired confirmation with the same controller stacks
  and current contracts before collection or training. If Qwen3 ties or loses,
  or solves fewer than two, larger-teacher capability alone has not cleared
  the screen. Next version the controller state with bounded exact per-action
  attempt counts and last failure reasons, then test only that state
  manipulation with the better wire-reliable teacher. If both arms fail mostly
  before typed application, retain the paired result but continue to attribute
  the common failure to the fixed generator boundary, not teacher quality.
- Secondary diagnostics, with no threshold: per-problem reward and turns;
  exact action sequence and verb counts; count of decisions that reuse an
  exact pair after an earlier zero-reward attempt in the same episode;
  DECOMPOSE choices after the first rejected DECOMPOSE; controller fallback and
  strict-wire errors; generator typed, algebra, and payload outcomes;
  deterministic closure events; all raw role attempts, retries, provider usage,
  and estimated cost. These diagnose mechanism but cannot override the primary
  symbolic solve gate.
- Cost and authorization boundary: the HF model listing observed on 07-14
  reports DeepInfra Qwen3-235B-A22B at $0.09 per million input tokens and $0.10
  per million output tokens; HF routing adds no markup. Mini and the fixed
  generator retain their provider rates. Expected paired spend is below $0.20,
  not a billing receipt. E30 authorizes exactly these two five-problem arms
  after absence, digest, test, and strict replay checks; it authorizes no model
  sweep, state change, prompt change, action change, 20-problem run, or
  training collection.

### E31 - RESOLVED PARTIAL/INVALID 07-14: direct versus fixed decomposition workspace screen
- Resolution, partial and invalid for the primary comparison: the direct arm
  completed all 20 ordered problems and produced 17/20 symbolic solves with
  zero generation errors. It made 20 accepted calls with no retries, reporting
  3,040 prompt and 2,137 completion tokens. The fixed arm was stopped after one
  complete episode when the same failure repeated in the next episode. Its
  partial trace contains 12 transitions: one accepted DECOMPOSE, four accepted
  child ANSWER actions, then seven failed SYNTHESIZE actions to the horizon.
  Across the retained partial diagnostic, DECOMPOSE succeeded twice and child
  ANSWER succeeded seven times, but all 14 SYNTHESIZE calls returned useful
  answers wrapped under `candidates`; both attempts for each call failed the
  top-level candidate schema, for 28 schema errors. The partial contains 23
  calls, 37 billed attempts, 41,897 prompt tokens, and 7,238 completion tokens.
  No fixed-arm final artifact was published. The primary paired gate is not
  evaluated, and the partial zero reward is not a mathematical result. The
  diagnostic identifies a candidate-admission and repeated-failure policy bug.
  E31 authorizes no rerun; the separately registered E32 contract below is the
  only follow-on execution.
- Purpose: test the minimum mechanism behind the permissive-workspace
  hypothesis before adding an LLM teacher. Does giving the same 7B worker a
  fixed `DECOMPOSE -> ANSWER(children) -> SYNTHESIZE -> SUBMIT` trace solve
  more GSM8K problems than the exact direct `ANSWER(root) -> SUBMIT` path?
  This is a paired engineering screen, not an accuracy estimate, controller
  result, RL result, or compute-efficiency claim.
- Manipulation (single): change only the action policy from `direct` to
  `fixed_decomposition`. The model, provider, source, hidden answer binding,
  prompts attached to each action, sampling, verifier, and publication
  boundary are fixed. The decomposition arm intentionally uses more worker
  calls. A positive result would establish only that this scaffold can buy
  performance with extra inference. It would not establish that decomposition
  is compute efficient or that a controller can choose it well.
- Frozen source: ordered rows 100-119 of
  `experiments/problems/gsm8k-train-500.jsonl`. Missing dataset ids are bound
  as `dataset-row-000100` through `dataset-row-000119`. The label-free ordered
  problem projection SHA-256 is
  `563135d13240dec66e7845747eb1e71c493b7fd7c2f38c0adeec007d0ee16398`.
  The separate hidden id-answer binding SHA-256 is
  `614efa0fe244b9211488aca88f4c05a350e30e0cc2615be7bbd5c93347b66cab`.
  Neither answer text nor an answer-bearing field enters policy state or an
  LLM request. These previously used training-split problems make E31 an
  engineering mechanism screen and prohibit a held-out or population claim.
- Frozen worker: Hugging Face Inference Providers routed to Together, model
  `Qwen/Qwen2.5-7B-Instruct:together`, asserted upstream Hub revision
  `a09a35458c702b33eeacc393d103063234e8bc28`. Provider-served weights remain
  asserted configuration metadata, not an attested identity. Both arms use
  temperature 0.0, 700 completion tokens per generative action, base seed 0
  with episode-turn offsets, one parse retry, six 429/5xx backoff retries,
  120-second timeout, standard JSON-object mode, provider guided JSON
  disabled, and strict nonmutating schema validation.
- Frozen workspace: `permissive-reasoning-workspace-v1`, five learned actions
  `DECOMPOSE`, `ANSWER`, `SYNTHESIZE`, `REVISE`, and `SUBMIT`, maximum depth 3,
  turn limit 12, and the hybrid local lens in
  `docs/superpowers/specs/2026-07-14-permissive-workspace-v1.md`. Generated
  child tasks and candidates are unverified workspace content. Only bounded
  JSON container shape is checked during generation. `SUBMIT` makes no LLM
  call, invokes the existing symbolic answer checker, terminates the episode,
  and supplies the only binary reward. The initial `ANSWER(root)` request is
  shared exactly with `experiments/run_direct_baseline.py`.
- Frozen run bindings: direct run-contract SHA-256
  `239e86342835b35aa6e8cfcdee2a6faedbdab9e17f8089f349964037bd6aec8f`;
  fixed-decomposition run-contract SHA-256
  `9f77b6ae4562e880a340eb355676fd08c0db0a55f73a2792ec78d5d766f37f95`.
  Zero-network preparation resolved both hashes without reading the HF token.
- Planned artifacts:
  `experiments/runs/2026-07-14-E31-direct-qwen25-7b-gsm20.jsonl` and
  `.diagnostics.jsonl`, plus
  `experiments/runs/2026-07-14-E31-fixed-decomposition-qwen25-7b-gsm20.jsonl`
  and `.diagnostics.jsonl`. Final and sibling `.partial` paths must be absent
  before execution. Every raw HTTP attempt, retry, request body, response,
  usage record, decoded payload, workspace transition, symbolic receipt, and
  call-id binding is retained. Complete paired artifacts publish without
  clobber only after all 20 ordered episodes finish.
- Primary measure and gate: paired terminal symbolic solve count. The mechanism
  screen passes only if fixed decomposition solves at least three more of the
  20 problems than direct prompting. A difference within plus or minus two is
  inconclusive at this sample size. A direct advantage of at least three is a
  failed mechanism screen for this fixed scaffold. Count success only from a
  terminal accepted `SUBMIT`; no LLM judge, intermediate heuristic, learned
  critic, or reward model can affect the measure.
- Pre-registered reading: on a pass, register a teacher-policy comparison
  before making teacher calls, retaining direct and fixed decomposition as
  controls. On an inconclusive or failed screen, do not scale the teacher yet.
  Use the sealed paired traces to select one next manipulation among the lens,
  decomposition prompt, or synthesis prompt, then register that manipulation.
  Structural or provider failures affecting either arm invalidate only the
  affected execution and retain its `.partial` diagnostics; they do not count
  as mathematical failures or authorize an unregistered rerun.
- Secondary diagnostics, with no threshold: per-problem reward and action
  trace; child count and child text; worker generation errors and retries;
  submitted answer; prompt, completion, and total tokens; provider-reported
  usage; estimated cost; and paired wins, losses, and ties. Diagnostics explain
  mechanism but cannot override the primary gate.
- Cost and authorization boundary: the previously observed Together listing
  for this model was $0.30 per million input tokens and $0.30 per million
  output tokens. Expected total spend is below $0.10, not a billing receipt.
  E31 authorizes exactly these two 20-problem arms after the frozen tests,
  digest checks, credential check, and artifact-absence check. It authorizes no
  teacher call, model sweep, prompt change, lens change, source change, rerun,
  training collection, or follow-on experiment.

### E32 - RESOLVED 07-14: permissive candidate admission recovery arm
- Resolution: both gates resolved. The candidate-admission integration target
  passed for SYNTHESIZE: all 20 ordered episodes published, nested candidate
  wrappers were admitted, every episode reached SUBMIT, and no failed
  SYNTHESIZE pair repeated. Two DECOMPOSE calls exhausted their single parse
  retry and used the bounded root-ANSWER fallback. The performance gate failed:
  fixed decomposition solved 9/20 versus the sealed direct control's 17/20, an
  eight-problem deficit. Paired outcomes were eight solved by both, nine by
  direct only, one by fixed only, and two by neither. Removing both episodes
  with DECOMPOSE generation errors still leaves direct ahead 15/18 to 9/18.
  E32 made 100 unique worker calls and retained 102 attempts: 98 accepted
  attempts and four schema-error attempts from the two failed calls. Provider
  usage was 60,252 prompt and 11,408 completion tokens, 71,660 total, or about
  $0.021498 at the preregistered planning rate. The robust primary reading is
  negative: this fixed scaffold buys substantially more inference and lowers
  symbolic success.
- Post-hoc mechanism audit: the dominant state/lens mismatch is that the
  decomposer emitted ordered, dependent steps, while ANSWER on a child hid
  earlier sibling tasks and candidates. Later children therefore recomputed
  missing intermediate values and often contradicted one another. Synthesis
  commonly propagated those errors and sometimes introduced fresh arithmetic
  or answer-format errors. This diagnosis cannot alter E32's failed gate. The
  next experiment should manipulate only the child lens by adding completed
  preceding-sibling work; it must retain the sealed direct and E32 controls and
  must not yet introduce a teacher.
- Purpose: recover the intended fixed-decomposition mechanism screen after the
  E31 integration failure, without rerunning or changing the valid 17/20 E31
  direct control. This is one fixed-policy arm on the same 20 ordered problems.
- Manipulation (single): replace the non-direct candidate admission and failure
  contract `workspace-worker-prompts-v1` with
  `workspace-worker-prompts-v1.1`. Prompt text and hybrid lens bytes are
  unchanged. The response schema now admits any top-level JSON object. The
  deterministic decoder prefers a top-level `answer`, then an `answer` under
  `candidate`, then the last answer-bearing item under `candidates`; an object
  with no recognized answer is retained whole as unverified candidate content.
  No mathematical semantics are checked. The associated failure rule permits
  only one failed SYNTHESIZE before a root ANSWER fallback, preventing the exact
  horizon loop observed in E31. These are one versioned candidate-admission and
  failure-boundary correction, not two measured treatments; E32 cannot
  attribute any outcome to one component.
- Frozen control and source: reuse the sealed E31 direct artifact at 17/20 and
  make no new direct call. E32 uses the same GSM8K rows 100-119, ordered ids,
  label-free projection SHA-256
  `563135d13240dec66e7845747eb1e71c493b7fd7c2f38c0adeec007d0ee16398`,
  and hidden label binding SHA-256
  `614efa0fe244b9211488aca88f4c05a350e30e0cc2615be7bbd5c93347b66cab`.
- Frozen worker and workspace: retain the exact E31 HF Together
  `Qwen/Qwen2.5-7B-Instruct:together` binding and asserted revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, temperature 0.0, 700 tokens,
  episode-turn seed offsets from zero, one parse retry, six backoff retries,
  120-second timeout, JSON-object mode, guided JSON disabled, maximum depth 3,
  turn limit 12, action policy, hybrid lens, symbolic checker, reward, and
  publication boundary. The E32 run-contract SHA-256 is
  `59600aef4239cc5b08658ccabf6ebc29f55d447a7e610d1f202b0516cbb1ebc6`.
- Planned artifacts:
  `experiments/runs/2026-07-14-E32-fixed-decomposition-v11-qwen25-7b-gsm20.jsonl`
  and `.diagnostics.jsonl`. Final and sibling `.partial` paths must be absent.
  The same complete, secret-free attempt and transition logging applies.
- Primary gates, read in order: first, integration passes only if all 20
  episodes publish, no structurally usable nested candidate is rejected, and
  no failed SYNTHESIZE pair repeats. Only after that gate passes, compare its
  terminal symbolic solve count with the sealed E31 direct 17/20. The scaffold
  performance gate passes only at 20/20, a gain of three. Results from 15/20
  through 19/20 are inconclusive at this sample size. A result of 14/20 or
  lower fails because direct leads by at least three. Extra inference remains
  unmatched, so even 20/20 would show a performance purchase, not efficiency.
- Pre-registered reading: if both gates pass, register the strong-teacher
  policy against the sealed direct and E32 fixed controls. If integration
  passes but performance does not, inspect only sealed paired traces and
  register one lens, decomposition, or synthesis manipulation. If integration
  fails, retain `.partial` diagnostics and do not interpret symbolic scores.
- Cost and authorization boundary: E32 authorizes exactly this one 20-problem
  fixed arm after its frozen tests, digest checks, cached-token presence, and
  path-absence check. Expected incremental spend is below $0.10 at the
  previously observed $0.30 per million input and output tokens. It authorizes
  no direct rerun, teacher call, model change, source change, unregistered
  repair, or second execution.

### E33 - RESOLVED GATE-PASS/CAUSAL-INCONCLUSIVE 07-16: completed preceding-sibling child context
- Resolution: the integration gate passed and the preregistered lens-effect
  threshold passed numerically. All 20 episodes and both final artifacts
  published under the frozen contract. E33 solved 12/20 versus sealed E32
  fixed decomposition at 9/20. Paired outcomes were eight solved by both, four
  by E33 only, one by E32 only, and seven by neither, for the registered net
  gain of three. The direct-performance gate failed: sealed E31 direct remains
  ahead 17/20 to 12/20, with 11 both, six direct only, one E33 only, and two
  neither.
- Causal validity audit: the numerical threshold pass is not a clean causal
  estimate of the lens manipulation. All 20 E32 and E33 DECOMPOSE request
  objects were byte-identical, but the hosted provider returned different
  decompositions on rows 101, 108, 111, and 115. On the 14 episodes with an
  identical applied decomposition, first-child request objects were also
  byte-identical, but row 113 returned a different first-child result. Those
  five upstream response differences account for the entire aggregate net
  gain. Among the 13 episodes with identical decomposition and first-child
  results, E33 gained row 100 and lost row 102, for no net change. This post-hoc
  audit does not erase the preregistered threshold pass, but it prevents a
  claim that preceding-sibling context caused the observed `+3`.
- Mechanism examples: row 100 had identical upstream work; predecessor context
  changed the wrong stationery result and submission `21` into the correct
  `15`, matching the intended dependency-flow mechanism. Row 102 also had
  identical upstream work; E33 preserved correct arithmetic but propagated
  units and submitted `25ml` instead of E32's accepted `25`. The context can
  repair dependency flow while increasing final-answer formatting risk.
- Integration and usage audit: the causal-prefix field appeared in all 41
  eligible later-child requests and no other request. Seventy-six structured
  future-task absence checks passed; no hidden-answer field, credential, or
  teacher call appeared. All episodes reached SUBMIT. Two DECOMPOSE calls, on
  rows 103 and 117, each exhausted one parse retry; no SYNTHESIZE call failed
  or repeated. The run made 99 logical worker calls and retained 101 HTTP
  attempts, 97 accepted and four schema errors. Provider usage was 69,681
  prompt and 13,322 completion tokens, 83,003 total, or approximately
  $0.0249009 at the preregistered planning rate.
- Published artifact SHA-256 values: run
  `f36adf09c3b8c4791d323edb48c44b26afce4b48492df849f9575a1da039dd7c`;
  diagnostics
  `05236ad9877b1617b1072723bb870f8760ec5f200e7faf4b06c3d26adb03c175`.
  No partial path remains. E33 made no teacher call and authorizes no rerun.
- Robust reading: the lens mechanism remains plausible, but fixed
  decomposition is still five solves below direct and the apparent improvement
  is causally inconclusive. A teacher experiment is eligible for separate
  registration under the literal preregistered rule, but must not treat E33's
  raw `+3` as established lens value. A causal lens confirmation should share
  or replay the upstream decomposition and first-child prefix across arms, or
  use contemporaneous randomized arms, before attributing a gain to context.
- Purpose: test the state-projection diagnosis from E32 before introducing or
  scaling an LLM teacher. Does exposing causally available earlier sibling work
  to later child `ANSWER` calls improve the same fixed decomposition scaffold?
  This is a paired engineering mechanism screen on previously used training
  problems, not an accuracy estimate, teacher result, RL result, or
  compute-efficiency claim.
- Manipulation (single): replace the effective E32 child context mode
  `target-only-v1` with `preceding-siblings-v1`. When `ANSWER` targets a
  non-root Goal, the hybrid lens additionally includes the ordered Goals and
  all Candidates of siblings that precede the target under its immediate
  parent and already have at least one Candidate. The work is nested under the
  immediate parent's entry in the root-to-target ancestry path. Incomplete and
  later siblings remain absent, and the field is absent when the causal prefix
  is empty. Initial `ANSWER(root)`, first-child `ANSWER`, all non-`ANSWER`
  actions, worker instruction text, and all other prompt prose remain
  byte-identical to the target-only mode. The mode is bound by
  `workspace-child-answer-context-v1` and the hosted run contract rather than
  selected implicitly.
- Frozen controls and source: reuse without new calls the sealed E31 direct
  artifact at 17/20 and the sealed E32 fixed-decomposition artifact at 9/20.
  Run exactly one new fixed-policy arm on ordered rows 100-119 of
  `experiments/problems/gsm8k-train-500.jsonl`, ids `dataset-row-000100`
  through `dataset-row-000119`. The label-free ordered problem projection
  SHA-256 is
  `563135d13240dec66e7845747eb1e71c493b7fd7c2f38c0adeec007d0ee16398`.
  The separate hidden id-answer binding SHA-256 is
  `614efa0fe244b9211488aca88f4c05a350e30e0cc2615be7bbd5c93347b66cab`.
  Expected answers remain outside policy state and every LLM request.
- Frozen worker, workspace, and policy: retain Hugging Face Inference Providers
  routed to Together, model `Qwen/Qwen2.5-7B-Instruct:together`, asserted
  upstream Hub revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, temperature 0.0, 700 completion
  tokens per generative action, episode-turn seed offsets from base seed 0, one
  parse retry, six 429/5xx backoff retries, 120-second timeout, standard JSON
  object mode, and guided JSON disabled. Retain maximum depth 3, turn limit 12,
  `fixed_decomposition`, the v1.1 permissive candidate wire, failure bounds,
  action legality, symbolic terminal checker, reward, and publication boundary.
  No teacher or controller model is called. `SUBMIT` remains the only operation
  that invokes the symbolic checker or can produce reward.
- Frozen preparation and artifacts: zero-network preparation produced run
  contract SHA-256
  `95ec2d6bfcbc088312598e155eeca0bcc5805675bf0a860ecac652bb51168c7d`
  under `workspace-hosted-run-contract-v1.1`. The only authorized outputs are
  `experiments/runs/2026-07-16-E33-fixed-preceding-siblings-qwen25-7b-gsm20.jsonl`
  and
  `experiments/runs/2026-07-16-E33-fixed-preceding-siblings-qwen25-7b-gsm20.diagnostics.jsonl`.
  Final and sibling `.partial` paths must be absent before execution. Retain
  every raw attempt, request, response, usage record, transition, symbolic
  receipt, and call-id binding. Publish without clobber only after all 20
  ordered episodes finish.
- Pre-execution gates: the focused workspace suite must pass the direct-byte,
  first-child parity, causal-prefix ordering, completion, later-sibling
  exclusion, hidden-label isolation, run-contract binding, and HTTP transport
  tests. The repository-wide `pytest -q` gate must remain green. Execution must
  recheck both source digests, the run-contract digest, cached HF-token
  presence, and absence of all four final and partial output paths.
- Integration gate, read first: all 20 episodes and both final artifacts must
  publish under the frozen header. Diagnostics must show the preceding prefix
  in eligible later-child request bodies, no later-sibling content in an
  earlier-child request, no hidden answer leakage, and no exact failed
  `SYNTHESIZE` pair repeated to the horizon. If this gate fails, retain partial
  diagnostics and do not interpret symbolic scores.
- Primary lens-effect measure and gate: paired terminal symbolic solve count
  against sealed E32 fixed decomposition at 9/20. A result of at least 12/20
  passes, because the causal lens gains at least three problems. Results from
  7/20 through 11/20 are inconclusive at this sample size because the change is
  within plus or minus two. A result at or below 6/20 is a negative lens result.
- Secondary scaffold measure and gate: compare with sealed E31 direct at
  17/20. Only 20/20 passes the direct-performance gate with the preregistered
  gain of three. Results from 15/20 through 19/20 are inconclusive. A result at
  or below 14/20 fails because direct leads by at least three. Extra inference
  is unmatched, so even 20/20 would show a performance purchase rather than an
  efficiency gain.
- Pre-registered reading: if integration and the lens-effect gate pass, the
  causal state projection is useful enough to register a separate strong
  teacher experiment against sealed direct and fixed controls, even if this
  fixed policy has not yet cleared the direct-performance gate. If the lens
  result is inconclusive or negative, inspect only the sealed paired traces and
  register one decomposition or synthesis manipulation next. No outcome from
  E33 itself authorizes an unregistered teacher call, rerun, prompt repair,
  model sweep, source change, or training collection.
- Secondary diagnostics, with no threshold: per-problem paired reward; child
  count and text; predecessor Candidate values; submitted answer; action trace;
  generation errors and retries; prompt and completion tokens; provider usage;
  estimated cost; paired wins, losses, and ties against E31 and E32. These can
  explain the mechanism but cannot override either primary gate.
- Cost and authorization boundary: at the previously observed Together rate of
  $0.30 per million input and output tokens, expected incremental spend is
  below $0.10, not a billing receipt. E33 authorizes exactly this one 20-problem
  arm after every frozen gate above. It authorizes no control rerun, teacher
  call, second execution, or follow-on experiment.

### E34 - RESOLVED POSITIVE 07-16: shared-prefix paired child-context confirmation
- Resolution: integration and the preregistered positive lens gate passed.
  Every one of the 18 exact shared prefixes produced two terminal branches.
  Preceding-sibling context solved 12/18 and target-only context solved 9/18.
  Paired outcomes were eight both, four preceding-only, one target-only, and
  five neither, so `P - T = 3`. The same five discordant problem ids from E33
  recur under the controlled prefix: preceding wins rows 100, 111, 113, and
  115, while target-only wins row 102. This confirms a causal lens benefit on
  this development screen and clears the registered rule for retaining
  `preceding-siblings-v1` in the first permissive-workspace teacher experiment.
- Integration audit: all 18 prefixes, 36 terminal branches, 120 logical calls,
  120 accepted attempts, and 156 transitions published. Both modes made 42
  ANSWER, 18 SYNTHESIZE, and 18 SUBMIT transitions. There were no generation
  errors, retries, action-trace differences, hidden-answer fields, credentials,
  or teacher calls. All 42 preceding ANSWER requests contained the causal
  prefix; all 42 target-only ANSWER requests omitted it; no other action
  contained it. Each paired hosted round shared its model, temperature,
  maximum tokens, and effective seed. Each mode occupied first provider
  position exactly 30 times.
- Secondary order diagnostic: all five discordances occurred among the nine
  cases whose first treated round called target-only first: that stratum had
  four preceding-only, one target-only, two both, and two neither. The nine
  cases beginning preceding-first had six both and three neither. This
  interaction has no preregistered threshold and cannot override the balanced
  primary result. The recurrence of the exact prior discordant ids supports a
  problem-specific context mechanism, but a larger fresh sample is required.
  The exact two-sided binomial/McNemar diagnostic for four versus one discordant
  wins is 0.375; E34 makes no population-effect claim.
- Usage and cost: preceding context used 48,527 prompt and 9,486 completion
  tokens across 60 calls. Target-only used 38,160 prompt and 8,292 completion
  tokens across 60 calls. Aggregate usage was 86,687 prompt and 17,778
  completion tokens, 104,465 total, or approximately $0.0313395 at the
  preregistered planning rate. The positive result purchases more prompt and
  completion tokens and is not an efficiency claim.
- Published artifact SHA-256 values: run
  `0e19189bec86512a3960ceeed2f5c860fdd60fec2fa11ac799d828c4f8d6e7d1`;
  diagnostics
  `6d48e40d026829fa5f0ca266649b0f193045a6da2fc32d95da08dc8a52e086cf`.
  No partial path remains. E34 made no teacher call and authorizes no rerun.
- Robust reading: completed preceding-sibling context is retained as the
  current child lens. The next experiment may register the first permissive-
  workspace teacher, but it must use a larger fresh harder slice with
  contemporaneous direct and fixed controls. The saturated E31-E34 development
  rows cannot establish teacher advantage. Root numeric finalization and exact
  direct recovery after failed decomposition remain separate future
  manipulations and must not be bundled into that teacher comparison.
- Purpose: resolve E33's causal ambiguity without regenerating or selecting a
  new upstream decomposition. Does completed preceding-sibling context improve
  terminal symbolic success when target-only and preceding-sibling branches
  begin from the exact same decomposition and first-child Candidate? This is a
  paired engineering confirmation on reused training problems, not an accuracy,
  teacher, RL, or compute-efficiency result.
- Manipulation (single): fork each immutable source prefix into
  `target-only-v1` and `preceding-siblings-v1`. The branch mode is the only
  treatment. Both branches retain the same state, fixed-decomposition policy,
  worker, request templates, action legality, horizon, verifier, reward,
  sampling parameters, and source-episode turn seed. At each paired round both
  actions are chosen before either branch executes. Hosted order follows
  `checkerboard-case-round-order-v1`: the first branch alternates by paired case
  plus round. Corresponding branch calls therefore use the same model,
  temperature, maximum tokens, and effective seed without assigning one mode a
  consistent temporal position.
- Frozen source: strict-read sealed E32 run
  `experiments/runs/2026-07-14-E32-fixed-decomposition-v11-qwen25-7b-gsm20.jsonl`,
  file SHA-256
  `18c51c244c51ff8c6e63768950672d95ce9272d9ae4b67fa24946d8c6ece826b`,
  under source run contract
  `59600aef4239cc5b08658ccabf6ebc29f55d447a7e610d1f202b0516cbb1ebc6`.
  Reconstruct every canonical observation and transition chain before
  selecting prefixes. Selection is structural and pre-treatment: retain an
  episode exactly when DECOMPOSE and its first child ANSWER both applied, then
  branch from the first-child next-state. Rows 103 and 117 are excluded because
  DECOMPOSE produced no child workspace. No reward or E33 outcome enters this
  selection. The 18-case shared-prefix SHA-256 is
  `06c89f18da456d4edc8169bf244de9fe5e971568537bc07f703a7468572f9e56`.
- Frozen cases and labels: ordered ids are rows 100-102, 104-116, and 118-119
  under their full `dataset-row-000NNN` ids. Their label-free problem projection
  SHA-256 is
  `8d28fe4195f1267f8a64b9a3ad25fb16e57051cc64aeb7ea950307afd57314a0`.
  Their separate hidden id-answer binding SHA-256 is
  `97c5063e5b791b6d770733c750e3d82fdd64bafc4b1ed33c5cd187af62a1679d`.
  Expected answers remain outside every prefix state, policy input, and LLM
  request.
- Frozen worker and execution: retain Hugging Face Inference Providers routed
  to Together, model `Qwen/Qwen2.5-7B-Instruct:together`, asserted upstream Hub
  revision `a09a35458c702b33eeacc393d103063234e8bc28`, temperature 0.0,
  700 completion tokens, source-episode turn offsets from seed 0, one parse
  retry, six 429/5xx backoff retries, 120-second timeout, standard JSON-object
  mode, guided JSON disabled, maximum depth 3, turn limit 12, the v1.1
  permissive candidate wire, and the unchanged fixed policy. No DECOMPOSE,
  first-child, direct-control, teacher, or controller call is made. Each branch
  continues with the remaining child ANSWER actions, one SYNTHESIZE, and
  symbolic SUBMIT.
- Frozen contract and artifacts: zero-network preparation produced paired run
  contract SHA-256
  `4b740e87cd0648de7ff9151c9871c2aa4b5933c441f2a05dbfaee82dded5ac52`
  under `workspace-paired-lens-contract-v1`. The only authorized outputs are
  `experiments/runs/2026-07-16-E34-paired-shared-prefix-lens-gsm18.jsonl`
  and
  `experiments/runs/2026-07-16-E34-paired-shared-prefix-lens-gsm18.diagnostics.jsonl`.
  Final and sibling `.partial` paths must be absent before execution. Both
  branches, every prefix, transition, request, response, retry, usage record,
  symbolic receipt, role, branch position, and call-id binding must be retained.
- Pre-execution gates: the sealed source file and contract digests, exact prefix
  reconstruction, selected problem and hidden-label digests, run-contract
  digest, cached-token presence, and all four output paths must be rechecked.
  The focused suite must cover tamper rejection, identical branch starts,
  same-seed paired calls, checkerboard order, lens isolation, secret exclusion,
  and partial-safe publication. Repository-wide `pytest -q` must remain green.
- Integration gate, read first: all 18 paired cases, 36 terminal branches, and
  both final artifacts must publish. With no downstream generation error, the
  fixed source entails exactly 120 logical worker calls and 156 transitions.
  Both branches must have identical action traces within every case; every
  paired hosted round must share its effective seed and sampling settings; the
  causal field must appear only in eligible preceding-sibling ANSWER requests;
  no hidden answer or credential may appear; and no teacher call may occur.
  Any unrepaired downstream generation error, broken pair, action divergence,
  or incomplete publication fails integration and blocks the symbolic reading.
  A repaired internal parse retry remains an attempt-level diagnostic, not an
  integration failure.
- Primary measure and gate: paired terminal symbolic reward. Let `P` be the
  number solved only by preceding-sibling context and `T` the number solved only
  by target-only context. The lens is positive only if `P - T >= 3`, negative
  only if `T - P >= 3`, and inconclusive when the difference is within plus or
  minus two. Report both marginal solve counts and all four paired cells. This
  development probe does not support a population effect claim even if its gate
  passes.
- Pre-registered reading: if integration passes and the lens is positive,
  retain `preceding-siblings-v1` for the separately registered teacher
  experiment. If negative, retain `target-only-v1`. If inconclusive, make no
  lens-value claim and default to target-only as the smaller prompt before
  testing the separately identified root-finalization contract. In every case,
  the first teacher evaluation must use a larger fresh harder slice with
  contemporaneous direct and fixed controls; E34 itself authorizes no teacher
  call or behavior-cloning collection.
- Secondary diagnostics, with no threshold: per-case submitted answers and
  action traces; branch-order strata; later-child and synthesis payload changes;
  generation errors and retries; prompt and completion tokens; paired request
  token deltas; provider usage; exact McNemar/binomial discordance diagnostic;
  and estimated cost. These explain the mechanism but cannot override the
  integration or primary gate.
- Cost and authorization boundary: 120 logical calls are expected. At the
  previously observed Together rate of $0.30 per million input and output
  tokens, expected incremental spend is below $0.10, not a billing receipt.
  E34 authorizes exactly this one complete paired run after every frozen gate.
  It authorizes no rerun, direct call, upstream regeneration, teacher call,
  model sweep, prompt repair, source change, or training collection.

### E35 - RESOLVED NEGATIVE/CEILING-LIMITED 07-16: first permissive teacher on held-out casework60
- Resolution: integration passed and all six final artifacts published. Exact
  direct prompting solved 59/60, fixed preceding-sibling decomposition solved
  27/60, and the GPT-4.1-mini teacher policy solved 31/60. The direct arm was
  unexpectedly at ceiling, so the positive teacher-over-direct gate was
  mathematically unreachable after that control completed. The negative gate
  resolved decisively: teacher/direct paired cells were 31 both, zero
  teacher-only, 28 direct-only, and one neither. The direct-favoring exact
  one-sided discordance probability is `1 / 2^28`, approximately
  `3.73e-9`; the exact two-sided value is approximately `7.45e-9`.
- Teacher versus fixed was inconclusive under the registered gate: 24 both,
  seven teacher-only, three fixed-only, and 26 neither, for a net teacher gain
  of four and one-sided probability 0.171875. Fixed versus direct was 27 both,
  zero fixed-only, 32 direct-only, and one neither. The three-arm oracle union
  was still only 59/60: no structured arm solved the sole direct failure.
- By balanced family, direct/fixed/teacher solves out of 12 were `12/8/10`
  for f1a, `12/4/3` for f1b, `12/4/8` for f2a, `11/5/5` for f2b, and
  `12/6/5` for f3. These narrow synthetic-family counts are diagnostics, not
  separate hypothesis tests.
- The teacher's initial choice explains most of the regression. It selected
  root ANSWER on 12 problems and solved all 12. It selected root DECOMPOSE on
  48 and solved 19; direct solved 47 of those same 48 and fixed decomposition
  solved 20. GPT-4.1-mini therefore did not adapt downstream better than the
  fixed scaffold on the subset it routed into decomposition. A larger teacher
  alone is not the next supported manipulation.
- The terminal result contract is also materially broken for structured
  paths. Among rejected submissions, 11 teacher answers and 13 fixed answers
  were the exact correct scalar wrapped only as `x = n`. Nine additional
  teacher failures contained the expected scalar elsewhere in a longer answer,
  including several explicit correct largest-root statements, but that looser
  string diagnostic is not a symbolic rescore. These rows retain reward zero.
  The direct lens requests a bare final scalar; root SYNTHESIZE, later root
  ANSWER, and root REVISE do not preserve the same verifier-ready answer-field
  contract.
- The action presentation and availability mask add separate failure modes.
  The teacher's legal menu grew from two choices to a maximum of 27; failed
  episodes averaged 9.35 listed actions per decision versus 6.68 for solved
  episodes. Answered Goals, stale Candidates, nested decomposition, and old
  revisions remain actionable. All 324 teacher HTTP replies parsed, but six
  proposed illegal cross-pairings, such as `SUBMIT` targeting root Goal 0
  instead of a listed Candidate id, and invoked fallback. Returning a single
  menu index would preserve the internal `(verb, target)` action while making
  pair legality structural at the teacher wire.
- Integration details: direct used 120 transitions and 60 accepted worker
  attempts with no generation error. Fixed used 337 transitions, 277 logical
  worker requests, 278 attempts, and one exhausted generation error. Teacher
  used 324 transitions, 324 accepted teacher calls, 265 logical worker
  requests, 268 worker attempts, three exhausted worker generation errors, six
  illegal-action fallbacks, 59 SUBMIT endings, and one horizon ending. Only one
  exact action pair repeated within a teacher episode, so generic pair
  oscillation was not the main failure. The hosted windows overlapped: the
  direct provider timestamps fall inside both structured-arm ranges. No hidden
  answer field, credential, or authorization header appears in an artifact.
- Usage and estimated cost: worker roles reported 349,382 prompt and 82,965
  completion tokens, approximately $0.1297041 at the observed Together
  $0.30-per-million rates. The teacher reported 257,408 prompt and 3,042
  completion tokens, approximately $0.1078304 at the current official
  GPT-4.1-mini rates of $0.40 per million input and $1.60 per million output
  tokens. E35 total estimated spend is $0.2375345, not a billing receipt. The
  cumulative estimated HF worker spend across E23-E35 is $0.2980854.
- Artifact SHA-256 values: direct run
  `0f52251658660bbd0312b75171ef6a1f967f7b954aaa180f7f5397feeeebff88`
  and diagnostic
  `24e1db21ca3fa324ab8dc8fd14bf5a74865b63f293cdf48aa46bef44b3132edc`;
  fixed run
  `b8c4dc7f8296b553b3cd4ff2c22348cbfac895508fa2cb48a79a56020619685b`
  and diagnostic
  `a98d934a8351686acaade7b2e2f6220c017d3efb090c7c1eca613c40704697aa`;
  teacher run
  `3897d2dc289d66de56a49f0f2b3e3c50c785ac76a3d1e7ef23f6693dd97817ef`
  and diagnostic
  `eb4225c4f07ea5f1d68729b01f35e01f0ab56a78a98a41f93b4aa113df5d68fa`.
  No E35 partial path remains.
- Robust reading: do not collect behavior-cloning labels from E35 and do not
  rerun it. The teacher needs transition knowledge about this worker, not just
  semantic problem-solving capability. The strongest state redesign is to
  generate the exact direct Candidate as a shared deterministic prefix, then
  let the controller inspect it and choose SUBMIT, REVISE, or structured work.
  This makes the workspace an improvement layer over the recoverable baseline
  instead of asking an uncalibrated teacher whether to obtain the baseline at
  all. Before another performance comparison, separately repair the root
  verifier-ready answer-field contract. Active-candidate action masking and an
  indexed teacher choice are distinct later manipulations. The held-out
  casework60 set cannot measure improvement for this 7B worker because direct
  is 59/60; a future teacher comparison needs either a harder verifier-covered
  source or a smaller worker with measured headroom.
- Purpose: test whether a strong LLM controller can use the permissive
  workspace and validated child lens to make a frozen 7B worker solve more
  held-out structure-requiring problems than either exact direct prompting or
  a deterministic decomposition policy. This is the first permissive teacher
  evaluation. It is not an RL result, behavior-cloning collection,
  compute-efficiency claim, or general math benchmark claim.
- Manipulation: action policy is the single three-level factor. The direct arm
  executes `ANSWER(root) -> SUBMIT`. The fixed arm executes one root
  DECOMPOSE, answers every generated child in order, SYNTHESIZEs the root, and
  SUBMITs, with its bounded direct recovery paths unchanged. The teacher arm
  uses OpenAI `gpt-4.1-mini-2025-04-14` to choose only one listed legal
  `(verb, target)` pair from the full canonical workspace at each turn. The
  same frozen worker executes every selected generative action through the
  same action-specific lens. All worker, workspace, lens, source, verifier,
  reward, sampling, and publication settings are fixed across arms.
- Frozen source: all 60 ordered records of
  `datasets/casework-v1/test.jsonl`, balanced at 12 examples in each of the
  five casework families. This project-held-out split has not appeared by id
  in an earlier run artifact. Its label-free ordered problem projection
  SHA-256 is
  `a98aef83ee4eee5945d632d56e25862e884811b0b73b6bc56a153b8edcfd72a3`.
  Its separate hidden id-answer binding SHA-256 is
  `683cee1cabb4192f688f09b3be87e8f6a7bdb7dd19a06aa2027f83c2ffc7b5b3`.
  Only `id` and `problem` enter policy state; answer, symbolic form, solution
  set, family, difficulty, and provenance remain outside every LLM request.
- Frozen worker and workspace: Hugging Face Inference Providers routed to
  Together, `Qwen/Qwen2.5-7B-Instruct:together`, asserted Hub revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, temperature 0.0, 700 maximum
  completion tokens, base seed 0 with episode-turn offsets, one parse retry,
  six 429/5xx backoff retries, 120-second timeout, standard JSON-object mode,
  guided JSON disabled, strict nonmutating candidate admission, maximum depth
  3, turn limit 12, and `preceding-siblings-v1` child context. Initial
  `ANSWER(root)` remains byte-identical to the direct request. Only root
  SUBMIT calls the symbolic checker and supplies binary reward.
- Frozen teacher: OpenAI `gpt-4.1-mini-2025-04-14`, with the dated model id
  also serving as asserted revision, through `https://api.openai.com/v1`.
  The teacher uses `workspace-global-planner-prompt-v1`, the full
  `workspace-observation-v1` plus exact ordered legal menu, temperature 0.0,
  64 maximum completion tokens, base seed 0 with episode-turn offsets, one
  parse retry, six 429/5xx backoff retries, 120-second timeout, JSON-object
  mode, guided JSON disabled, and strict `(verb, target)` validation. An
  invalid or illegal response falls back to direct root ANSWER when legal,
  otherwise the first legal action, and is logged. The teacher never sees the
  hidden answer and never verifies, scores, or supplies reward.
- Frozen contracts: zero-network preparation produced direct run-contract
  SHA-256
  `1f7d003292cadd172816b59d7e97c01afb66e817654f8d037a289b760157f2e4`,
  fixed preceding-sibling run-contract SHA-256
  `709e52804c08a7a976007394c83155da5899081971668afad86b93bbf46e1ceb`,
  and teacher run-contract SHA-256
  `6a483fdaed8a7d8c8f2243b4c8b87ee7cd2bb877a9da82b5cfb5e7cdb61a571d`
  under `workspace-hosted-run-contract-v1.1`.
- Planned artifacts:
  `experiments/runs/2026-07-16-E35-direct-qwen25-7b-casework-test60.jsonl`
  and `.diagnostics.jsonl`;
  `experiments/runs/2026-07-16-E35-fixed-preceding-qwen25-7b-casework-test60.jsonl`
  and `.diagnostics.jsonl`; and
  `experiments/runs/2026-07-16-E35-teacher-gpt41mini-qwen25-7b-casework-test60.jsonl`
  and `.diagnostics.jsonl`. All final and sibling `.partial` paths must be
  absent before execution. Each arm runs in source order in its own canonical
  collector. The three collectors must be launched within the same execution
  window so their hosted calls overlap in time; provider timestamps and raw
  attempts retain the audit trail. This is a contemporaneous control, not a
  shared-output design, so residual provider nondeterminism must be reported.
- Pre-execution gates: recheck all source, hidden-label, and run-contract
  digests; cached HF token and OpenAI credential presence; six final and six
  partial output paths absent; repository-wide `pytest -q` at 918 passed and 1
  skipped; and clean `git diff --check` plus prohibited-glyph scan on the
  experiment changes. No hosted request is authorized before these checks.
- Integration gate, read first: all three 60-episode final artifact pairs must
  publish under their exact contracts and ordered source. Every transition
  must use a listed legal action and preserve state invariants; every terminal
  reward must come only from the symbolic SUBMIT receipt; all worker and
  teacher calls and attempts must have role-bound ids; controls must make zero
  teacher calls; teacher decisions must cover every teacher-arm transition;
  and no hidden label or credential may enter an artifact. A missing arm,
  unrepaired infrastructure failure, or incomplete publication blocks all
  performance readings. Repaired HTTP backoff or parse retries remain logged
  diagnostics rather than mathematical failures.
- Co-primary measures: paired terminal symbolic reward for teacher versus
  direct, and teacher versus fixed. For each comparison report both marginal
  solve counts and the four paired cells. Let `W` be teacher-only solves and
  `L` control-only solves. A comparison is a positive screen only when
  `W - L >= 5` and the exact one-sided binomial probability of at least `W`
  wins among `W + L` discordant pairs under probability 0.5 is at most 0.10.
  It is a negative screen only when the corresponding control satisfies the
  same rule against teacher. All other results are inconclusive. E35 fully
  passes only if teacher is positive against both controls; it partially
  passes if positive against direct but not fixed. Report the exact two-sided
  discordance probability as a secondary uncertainty diagnostic.
- Pre-registered reading: a full pass justifies a separately registered fresh
  teacher-trajectory collection for behavior cloning. A partial pass shows
  that extra workspace inference can beat the small model alone but does not
  establish adaptive action-policy value; diagnose the teacher/fixed traces
  before choosing one state, action, or prompt manipulation. If teacher is
  inconclusive or negative against direct, do not collect training labels and
  select exactly one failure-linked manipulation from the sealed traces. If
  all arms are near ceiling, move to a separately versioned harder source
  without interpreting the missing delta as policy equivalence. No result on
  this narrow synthetic test split supports a general-math population claim.
- Secondary diagnostics, with no threshold: per-problem and per-family reward;
  all three pairwise tables; turns, action traces, verb and target counts;
  teacher initial-action distribution, exact-pair repetitions, fallbacks, and
  horizon endings; child counts and texts; submitted candidates; generation
  errors and retries; prompt and completion tokens by role; provider timing
  overlap; estimated cost; and an oracle union across arms. Diagnostics select
  the next preregistered manipulation but cannot override the integration or
  co-primary gates.
- Cost and authorization boundary: expected worker spend is below $0.50 at the
  previously observed Together rate of $0.30 per million input and output
  tokens. Expected mini teacher spend is below $1.00 under the bounded horizon;
  these are conservative estimates, not billing receipts. E35 authorizes
  exactly these three complete 60-problem arms after every gate above. It
  authorizes no rerun, model sweep, prompt or state change, teacher replacement,
  source change, behavior-cloning collection, or training run.

### Post-E35 implementation note - baseline-anchored workspace v2 (no experiment)
- Local implementation follows
  `docs/superpowers/specs/2026-07-16-baseline-anchored-workspace-v2.md` and made
  no hosted model call. This note records software state only. It is not E36,
  contains no performance result, and authorizes no inference spend.
- Four factors are independently bound: `direct-candidate-prefix-v1`,
  `verifier-ready-root-answer-v1`, `active-frontier-v1`, and
  `legal-menu-index-v1`. Legacy defaults and E31-E35 request contracts remain
  available for replay.
- The direct prefix is an ordinary logged `ANSWER(root)` transition using the
  exact direct request, but its decision source is `deterministic_prefix` and
  it is never behavior-cloning eligible. A failed prefix remains visible. The
  active observation identifies the protected direct Candidate, and that
  Candidate stays submittable after structured work.
- The teacher wire now maps one index to the exact legal menu. On malformed or
  out-of-bounds output, the deterministic fallback chooses direct root ANSWER
  when legal, protected direct SUBMIT when legal, and only then the first
  ordered legal action. This prevents a teacher transport failure after the
  prefix from defaulting to DECOMPOSE.
- `src/verifier_anchored_rl/workspace_v1/redesign.py` supplies the bound run
  contract, provenance-safe records, partial-safe collector, and BC eligibility
  field. `experiments/run_workspace_redesign.py` prepares contracts without
  network access and requires exact source, hidden-label, and run-contract
  digests for hosted execution.
- Limitation: this collector executes one arm at a time. It does not establish
  byte-identical shared direct prefixes across a direct arm and a teacher arm.
  Any causal prefix-preservation comparison needs a separately implemented and
  preregistered paired branch collector.
- Verification: the workspace-focused suite has 35 passing tests. The full
  repository suite passed 926 tests with one CUDA-only skip and the preexisting
  `tests/test_heads.py` tensor-conversion warning. Static compilation, scoped
  diff checks, and the prohibited-glyph scan also passed.

### E36 - RESOLVED POSITIVE 07-17: paired verifier-ready root result contract on sealed E35 states
- Resolution: integration passed and both final artifacts published. All 59
  exact source prefixes produced both branches, 118 hosted generations, 118
  local symbolic submissions, 236 transitions, 59 paired summaries, and one
  run summary. Every hosted generation was accepted on its first attempt, so
  there were zero parse retries, backoff retries, or exhausted generation
  errors. Paired observations, actions, model settings, and effective seeds
  matched; every ready request differed from its free request only by the one
  registered instruction suffix; checkerboard order, call-role binding, and
  symbolic-only reward held; no teacher call, hidden answer field, credential,
  or partial artifact remained.
- The registered result is positive. Verifier-ready root results solved 49/59
  versus 27/59 for the exact legacy prompt. Paired cells were 27 both, 22
  verifier-ready-only, zero legacy-only, and 10 neither, for `R - F = 22`.
  The exact one-sided discordant-pair probability is `1 / 2^22`, approximately
  `2.38419e-7`; the exact two-sided value is approximately `4.76837e-7`.
  This clears both the registered net-five and probability gates.
- The answer-shape audit supports the intended mechanism but shows that the
  prompt sometimes changed result selection too. The legacy arm produced 27
  symbolically accepted answers, 12 rejected answers whose assignment RHS was
  exactly correct, and 20 other rejected answers. The ready arm produced 49
  accepted answers, two rejected correct assignment wrappers, and eight other
  rejected answers. Among the 22 ready-only cases, 10 legacy answers were
  exact correct assignment wrappers and nine more contained the exact expected
  scalar as a standalone token inside prose or alternatives. The remaining
  three legacy answers did not contain the expected scalar and changed from
  `-8, -10` to `-18`, `-13` to `-9`, and `-1, -7` to `-8`. These mechanical
  categories are diagnostics, not alternate rewards or symbolic rescoring.
- The direction was not confined to provider position: free-first cases had
  12 both, 13 ready-only, and five neither; ready-first cases had 15 both, nine
  ready-only, and five neither. The exact legacy branch reproduced the sealed
  E35 fixed reward on all 59 selected problems and reproduced the submitted
  answer text on 58/59. This substantially limits provider drift as an
  explanation for the paired result. All 118 provider responses reported
  `Qwen/Qwen2.5-7B-Instruct-Turbo` between 2026-07-17 22:45:55 and 22:55:06
  UTC while the requested and contract-bound routed model id remained
  `Qwen/Qwen2.5-7B-Instruct:together`.
- Usage was 110,894 prompt tokens and 21,382 completion tokens across 118
  logical calls. By branch, free used 53,323 prompt plus 13,694 completion
  tokens; ready used 57,571 prompt plus 7,688 completion tokens. The treatment
  added 4,248 prompt tokens but reduced completion by 6,006, for 1,758 fewer
  total tokens. Estimated cost is $0.0396828 at $0.30 per million input and
  output tokens, not a billing receipt. Cumulative estimated HF worker spend
  across E23-E36 is therefore approximately $0.3377682.
- Artifact SHA-256 values: run
  `8cf12f3c370fc7e7cb93ae797825e4b7edced385a8a3a6490fb183ad2f1b9a6c`
  and diagnostics
  `071e73b3c53691af4c3f051a9f3d30ddfd2d3aad17f93df20a02e4428009d016`.
- Robust reading: retain `verifier-ready-root-answer-v1` for the next
  separately registered direct-prefix teacher experiment. E36 establishes a
  large causal prompt-contract mechanism on reused sealed E35 states; it does
  not show that decomposition, a teacher policy, or a learned controller beats
  exact direct prompting. The next boundary must fork direct submission and
  teacher continuation from the same exact generated direct Candidate on a
  verifier-covered source with measured direct-worker headroom. E36 supplies
  no behavior-cloning rows and authorizes no rerun or next hosted call.
- Purpose: test whether an explicit verifier-ready root answer-field
  instruction increases terminal symbolic success when the same frozen 7B
  worker finalizes the exact same completed child workspace. This is a paired
  prompt-contract mechanism test on reused E35 states. It is not a teacher,
  controller, behavior-cloning, RL, general-math, or fresh-population result.
- Single manipulation: root result mode has two levels.
  `workspace-answer-free-v1` exactly reproduces the sealed E35
  `SYNTHESIZE(root)` request. `verifier-ready-root-answer-v1` inserts only the
  versioned instruction that the root Candidate `answer` field contain the
  directly checkable requested scalar or expression, with assignments,
  equations, units, prose, code fences, and alternatives placed in
  `reasoning` unless the problem requests that form. System prompt, complete
  hybrid lens, state, action, schema, strict validation mode, child context,
  model, temperature, token limit, effective seed, and symbolic checker are
  identical within a pair. Trusted code does not extract, repair, canonicalize,
  or judge generated intermediate content.
- Frozen source: the canonical E35 fixed preceding-sibling run
  `experiments/runs/2026-07-16-E35-fixed-preceding-qwen25-7b-casework-test60.jsonl`,
  SHA-256
  `b8c4dc7f8296b553b3cd4ff2c22348cbfac895508fa2cb48a79a56020619685b`,
  under source run-contract SHA-256
  `709e52804c08a7a976007394c83155da5899081971668afad86b93bbf46e1ceb`.
  Strict preparation reconstructs every source transition and exact-replays
  every selected legacy request. It selects the 59 pre-SYNTHESIZE observations
  that contain at least one root child, a Candidate for every root child, no
  root Candidate, a legal root SYNTHESIZE action, and budget for SYNTHESIZE
  plus SUBMIT. Episode `cw1-1dcd933e33fd` is excluded because its source
  decomposition failed and it never reached a pre-SYNTHESIZE state. Selection
  depends only on the pre-treatment structural state, not the source synthesis
  output or reward.
- Frozen selected bindings: the ordered 59-case prefix projection SHA-256 is
  `4c06db248b9659e253412666e57170b6a5b8bf82ba6f1ba3b4f8f28595dbf8db`.
  The selected label-free problem projection SHA-256 is
  `34f1c35ac7d0c76559ae3b1d754f5547a5c48bb3c9b771a329706b07b7ea4907`;
  the separate hidden id-answer binding SHA-256 is
  `3890aef6ce3bfdbf21e24859a7349fac72e7ae6ee1b66fce120401a565c0dc79`.
  Only problem text and the reconstructed workspace enter worker requests.
  Hidden answers enter only local symbolic SUBMIT.
- Frozen worker: Hugging Face Inference Providers routed to Together,
  `Qwen/Qwen2.5-7B-Instruct:together`, asserted Hub revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, through
  `https://router.huggingface.co/v1`; temperature 0.0, 700 maximum completion
  tokens, base seed 0 with source-episode and source-turn offsets, one parse
  retry, six 429/5xx backoff retries, 120-second timeout, guided JSON disabled,
  and strict nonmutating JSON validation. Maximum workspace depth is 3, source
  turn limit is 12, child context is `preceding-siblings-v1`, action
  availability is legacy `all-affordances-v1`, and no teacher is instantiated.
- Frozen paired execution: both branches start from the same immutable source
  State and execute only `SYNTHESIZE(root)`. Their hosted generations are
  adjacent, before either local submission, and use the same effective seed.
  Case order is source order. Checkerboard branch order is free-first for the
  30 even case indexes and ready-first for the 29 odd indexes. A successfully
  generated root Candidate is immediately submitted through the unchanged
  symbolic verifier. SUBMIT makes no model call. The exact prepared run
  contract under `workspace-paired-root-result-contract-v1` has SHA-256
  `4bfdd7716343913a05c290a495a3bf587be9b6d0da2a6baf1b52a22a7fa9a3c3`.
- Planned artifacts:
  `experiments/runs/2026-07-17-E36-paired-root-result-qwen25-7b-casework59.jsonl`
  and
  `experiments/runs/2026-07-17-E36-paired-root-result-qwen25-7b-casework59.diagnostics.jsonl`.
  Both final paths and both sibling `.partial` paths must be absent before
  execution. Raw attempts are role-bound as `free_root_result_worker` or
  `verifier_ready_root_result_worker`; all prefixes, requests, payloads,
  receipts, usage, branch positions, and call ids are retained.
- Pre-execution gates: re-run zero-network preparation and match all five
  source, prefix, selected-problem, hidden-binding, and run-contract digests;
  confirm the cached HF credential without printing it; confirm all four
  output paths are absent; retain the 40-test workspace result; and retain the
  repository-wide result of 931 passed, one CUDA-only skip, and the preexisting
  `tests/test_heads.py` warning. Compilation, scoped diff checks, and the
  prohibited-glyph scan must remain clean. No request is authorized before
  these gates and this ledger entry.
- Integration gate, read first: all 59 prefix records, 118 paired generation
  calls, 118 local SUBMIT transitions, 236 total transitions, 59 case
  summaries, one run summary, and both final artifacts must publish. Every pair
  must start from byte-identical observations; both generated actions must be
  `SYNTHESIZE(root)`; the ready request must equal the free request after
  removing exactly one versioned treatment suffix; paired model, temperature,
  token limit, and seed must match; branch position must follow the checkerboard
  schedule; both branches must reach symbolic SUBMIT; no teacher call may
  occur; and no hidden answer or credential may appear. Any unrepaired
  generation error, missing branch, request drift, broken pair, incomplete
  publication, or role/call-id mismatch fails integration and blocks the
  primary reading. A repaired parse or HTTP retry remains an attempt-level
  diagnostic, not an integration failure.
- Primary measure and gate: paired terminal symbolic reward. Let `R` be the
  number solved only by `verifier-ready-root-answer-v1` and `F` the number
  solved only by `workspace-answer-free-v1`. Report both marginal solve counts
  and the four paired cells. The contract is positive only if `R - F >= 5` and
  the exact one-sided binomial probability of at least `R` ready wins among
  `R + F` discordant pairs under probability 0.5 is at most 0.10. It is
  negative only if `F - R >= 5` and the reverse one-sided probability is at
  most 0.10. Otherwise it is inconclusive. Report the exact two-sided
  discordance probability as a secondary uncertainty diagnostic.
- Pre-registered reading: if integration passes and the result is positive,
  retain `verifier-ready-root-answer-v1` for the separately registered
  direct-prefix teacher experiment. If negative, retain the legacy free
  contract and inspect only whether the instruction changed mathematical
  content, without weakening the symbolic checker. If inconclusive, make no
  root-contract benefit claim and do not bundle this mode with active-frontier
  masking or indexed teacher choice in a performance comparison. Because these
  states reuse E35, even a positive result establishes only a causal mechanism
  on this sealed development source.
- Secondary diagnostics, with no threshold: exact generated and submitted
  answers; mechanically identified exact scalar, assignment-wrapped, units,
  prose, list, or other answer shapes; source E35 outcome; branch-order strata;
  retries and generation errors; prompt and completion tokens; request-length
  delta; provider timestamps; and estimated cost. These diagnostics explain
  the prompt mechanism but cannot override integration or the primary gate.
- Cost and authorization boundary: 118 logical hosted calls are expected. At
  the previously observed Together rate of $0.30 per million input and output
  tokens, expected incremental spend is below $0.15, not a billing receipt.
  E36 authorizes exactly one complete paired 59-case run after every frozen
  gate. It authorizes no rerun, upstream regeneration, teacher call, model or
  source change, prompt adjustment, action-mask change, behavior-cloning
  collection, or training run.

### E37 - RESOLVED INCONCLUSIVE 07-17: shared Qwen-3B direct Candidate versus teacher continuation
- Resolution: integration passed and both final artifacts published. Exact
  shared direct submission solved 43/60 and teacher continuation also solved
  43/60. Paired cells were 43 both, zero teacher-only, zero direct-only, and
  17 neither. With no discordant pairs, `T - D = 0`; the registered positive
  and negative gates both fail, and the exact paired probability is 1.0 by the
  no-discordance convention. Direct was below the 56/60 ceiling rule.
- Baseline preservation succeeded descriptively but the registered qualifier
  does not apply without a positive screen. Every one of the 60 teacher paths
  ultimately submitted Candidate 1, the protected shared direct Candidate.
  The teacher therefore broke zero direct-correct answers and recovered zero
  direct-wrong answers. The oracle union remained 43/60.
- The state carried useful post-hoc routing signal even though the continuation
  produced no gain. The teacher submitted immediately on 49 cases: 39 of 43
  direct-correct cases and 10 of 17 direct-wrong cases. It requested more work
  on 11 cases: four direct-correct and seven direct-wrong. Thus 7/11 selected
  work cases were direct failures versus 10/49 immediate-submit cases. The
  post-hoc odds ratio is 6.825 and the one-sided exact hypergeometric
  probability is approximately 0.00795. This was not a registered efficacy
  measure and supports only the diagnostic that the direct Candidate made
  worker-specific uncertainty more visible to the teacher.
- Continuation generation is the clearest next boundary. The teacher made 24
  generative choices: two DECOMPOSE, six ANSWER, 14 REVISE, and two
  SYNTHESIZE. Mechanically, eight returned the requested container shape, eight
  were admitted despite echoing a lens-shaped object without an answer
  container, and eight exhausted parsing after long echoes. All 14 REVISE
  calls failed to produce an answer-bearing revision: six lens echoes were
  admitted by the open `{type: object}` Candidate schema and eight exhausted
  after one retry. Both root SYNTHESIZE outputs were answer-shaped but
  symbolically wrong. These are generation-contract diagnostics, not alternate
  rewards or rescored branches.
- Integration details: all 60 shared prefixes completed on their first attempt
  and each exact direct request created one protected Candidate. The run has
  204 transitions: 60 shared prefixes, 60 direct SUBMIT transitions, and 84
  teacher transitions. It made 84 logical worker calls (60 prefix plus 24
  continuation) across 92 attempts, and 84 teacher calls across 84 attempts.
  There were eight exhausted continuation generation errors, zero teacher
  fallbacks, zero horizons, and no missing call binding. All 266 run records
  and 177 diagnostic records are canonical; fork observations match exactly;
  credentials, hidden answer fields, and partial artifacts are absent.
- Usage: worker attempts reported 41,390 prompt and 24,676 completion tokens;
  teacher calls reported 65,837 prompt and 504 completion tokens. The official
  GPT-4.1-mini rates imply approximately $0.0271412 of teacher spend. The HF
  router and model metadata expose no Featherless price for this model, so no
  defensible HF dollar estimate is added. Provider timestamps span
  2026-07-18 00:05:55 through 00:14:02 UTC. Worker responses report
  `Qwen/Qwen2.5-3B-Instruct`; teacher responses report the exact dated mini id.
- Artifact SHA-256 values: run
  `67b8d353122f86201dd20ad8b39545318e3af5ac634e3e837646c17f18fb0c48`
  and diagnostics
  `dda68061f83dba7d49d3f8a8c4c7d2998350230bedb809eb6d8d17dfd6e8ac20`.
- Robust reading: the shared baseline anchor fixed E35's catastrophic
  direct-answer loss but did not improve accuracy. Do not use E37 for BC and
  do not rerun it. The next supported single manipulation is a versioned
  answer-container contract for REVISE, tested on exact sealed E37 pre-REVISE
  states before another end-to-end teacher run. At minimum it must require an
  explicit answer-bearing Candidate and prevent a copied lens object from
  becoming the Candidate answer. Whether the teacher should later compare or
  submit new work is a separate action-policy question; E37 cannot test it
  because no valid root revision was produced.
- Purpose: test whether the redesigned GPT-4.1-mini teacher continuation can
  improve on one exact Qwen2.5-3B direct Candidate while retaining that
  Candidate as a protected fallback. This is the first exact shared-prefix
  teacher comparison and a reused-development system screen. It is not a
  behavior-cloning collection, RL result, compute-efficiency claim, component
  attribution, or general-math population claim.
- Single manipulation: after one shared deterministic `ANSWER(root)` worker
  transition, the branch policy has two levels. `direct_submit` submits the
  exact protected Candidate locally without another model call.
  `teacher_continuation` gives GPT-4.1-mini the exact same immutable
  post-prefix state and lets it select actions until SUBMIT or horizon. The
  direct Candidate remains submittable throughout. No direct output is
  regenerated, copied between independent hosted calls, or repaired by trusted
  code.
- Fixed redesigned interface: both outcomes use
  `direct-candidate-prefix-v1`, `preceding-siblings-v1`, and the retained
  `verifier-ready-root-answer-v1` contract. The teacher uses
  `active-frontier-v1` and `legal-menu-index-v1`. The indexed prompt is
  `workspace-global-planner-index-prompt-v1.1`, which restores the five action
  definitions present in the legacy planner prompt; its earlier v1 text made
  no hosted call. E37 evaluates this complete continuation system but cannot
  attribute any result separately to the active mask or indexed wire. Those
  remain separately versioned later ablations.
- Frozen source: all 60 ordered records of
  `datasets/casework-v1/test.jsonl`. This source was first held out for E35 and
  is now a reused development screen. Its label-free ordered problem
  projection SHA-256 is
  `a98aef83ee4eee5945d632d56e25862e884811b0b73b6bc56a153b8edcfd72a3`;
  its separate hidden id-answer binding SHA-256 is
  `683cee1cabb4192f688f09b3be87e8f6a7bdb7dd19a06aa2027f83c2ffc7b5b3`.
  Only `id` and `problem` enter policy state. The hidden answer enters only the
  local symbolic SUBMIT checker.
- Worker selection and binding: the earlier local Qwen2.5-3B direct run solved
  68.8 percent of casework-v1 train, so 3B supplies a capability band with
  more likely headroom than E35's 59/60 7B control. That prior number is only a
  source-selection heuristic because E37 changes provider, precision, and
  split. E37's exact shared prefixes are its contemporaneous direct control.
  Hugging Face model metadata checked without inference on 07-17 reports
  `Qwen/Qwen2.5-3B-Instruct` live only through Featherless AI at asserted Hub
  revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`. The routed model id is
  `Qwen/Qwen2.5-3B-Instruct:featherless-ai` through
  `https://router.huggingface.co/v1`, temperature 0.0, 700 maximum completion
  tokens, base seed 0 with episode-turn offsets, one parse retry, six 429/5xx
  backoff retries, 120-second timeout, JSON-object mode, guided JSON disabled,
  and strict nonmutating Candidate admission.
- Frozen teacher and environment: OpenAI
  `gpt-4.1-mini-2025-04-14`, with the dated model id also serving as asserted
  revision, through `https://api.openai.com/v1`; temperature 0.0, 64 maximum
  completion tokens, base seed 0 with episode-turn offsets, one parse retry,
  six 429/5xx backoff retries, 120-second timeout, JSON-object mode, and guided
  JSON disabled. Workspace maximum depth is 3 and turn limit is 12 including
  the shared prefix. SUBMIT makes no model call and the symbolic verifier is
  the only reward source.
- Frozen contract: zero-network preparation selected all 60 cases and produced
  exact run-contract SHA-256
  `d71464f51c27f2080f385cfcffc5bc47524596324237b8dded0fd971c5ef2f28`
  under `workspace-paired-direct-teacher-contract-v1`. Every transition is
  evaluation-only and has `behavior_cloning_eligible=false`, regardless of
  outcome or teacher provenance.
- Planned artifacts:
  `experiments/runs/2026-07-17-E37-paired-direct-teacher-gpt41mini-qwen25-3b-casework-test60.jsonl`
  and
  `experiments/runs/2026-07-17-E37-paired-direct-teacher-gpt41mini-qwen25-3b-casework-test60.diagnostics.jsonl`.
  Both final paths and both sibling `.partial` paths must be absent before
  execution. Raw attempts are role-bound as `shared_prefix_worker`,
  `teacher_continuation_worker`, or `teacher_controller`.
- Pre-execution gates: re-run zero-network preparation and match the two source
  digests plus run-contract digest; confirm the cached HF and OpenAI
  credentials without printing them; confirm all four output paths are absent;
  retain the 17-test focused result for paired, redesign, and root-result
  boundaries; require repository-wide `pytest -q` to pass; and require static
  compilation, scoped `git diff --check`, and the prohibited-glyph scan to be
  clean. No hosted request is authorized before this entry and every gate.
- Integration gate, read first: all 60 shared prefixes, 60 direct SUBMIT
  transitions, complete teacher continuations, 60 paired summaries, one run
  summary, and both final artifacts must publish. Each prefix must make exactly
  one worker call, use the exact direct request, create exactly one protected
  root Candidate, and have no exhausted generation error. The direct branch
  must submit that Candidate with no worker or teacher call. The first teacher
  observation must equal the direct SUBMIT observation byte for byte. Every
  teacher transition must bind exactly one controller call and every
  generative teacher action exactly one continuation-worker call. All actions
  must be legal under the bound active menu; reward must come only from local
  symbolic SUBMIT; no hidden answer or credential may appear. Any unavailable
  prefix pair, broken state fork, missing call binding, incomplete publication,
  or contract drift fails integration and blocks performance reading.
  Repaired HTTP or parse retries remain attempt diagnostics.
- Primary measure and gate: paired terminal symbolic reward. Let `T` be cases
  solved only by teacher continuation and `D` cases solved only by direct
  submission. Report both marginal solve counts and all four paired cells. The
  system screen is positive only if `T - D >= 5` and the exact one-sided
  binomial probability of at least `T` teacher wins among `T + D` discordant
  pairs under probability 0.5 is at most 0.10. It is negative only if direct
  satisfies the same rule in reverse. Otherwise it is inconclusive. Report the
  exact two-sided probability as a secondary uncertainty diagnostic.
- Preservation qualifier and ceiling rule: a positive screen is called
  baseline-preserving only if `D <= 3`; otherwise it shows a net improvement
  that still discards too many correct direct Candidates. If direct submission
  solves at least 56/60, the registered net-five positive gate is unreachable,
  so record the run as ceiling-limited rather than interpreting a missing gain
  as teacher equivalence.
- Pre-registered reading: a baseline-preserving positive result justifies a
  separately registered confirmation on a fresh harder verifier-covered
  source, followed only then by a fresh teacher-trajectory collection. A
  positive result that fails preservation requires trace diagnosis and one
  separately registered state, action, or prompt manipulation before any
  collection. An inconclusive or negative result authorizes no training data;
  use the direct-correct/teacher-broken and direct-wrong/teacher-action strata
  to select exactly one next ablation. E37 rows themselves never enter BC.
- Secondary diagnostics, with no threshold: direct and teacher submitted
  answers; per-family outcomes; first teacher action conditioned on direct
  correctness; preservation rate among direct-correct cases; recovery rate
  among direct-wrong cases; teacher action, target, menu-size, fallback,
  generation-error, and horizon counts; exact repeated pairs; oracle union;
  prompt and completion tokens by role; provider response model and timing;
  retries; and estimated cost. These diagnostics cannot override integration
  or the primary gate.
- Cost and authorization boundary: the horizon permits at most 60 shared worker
  calls, 660 teacher calls, and 660 continuation-worker calls. Actual use is
  expected to be much lower. Featherless metadata exposes no router price, so
  a precise HF estimate cannot be preregistered; the account has a $5 recharge
  and the bounded run must fail rather than change provider if credit is
  insufficient. At current official GPT-4.1-mini rates, bounded teacher spend
  is expected below $1.00. E37 authorizes exactly one complete 60-case run
  after every gate. It authorizes no smoke call, rerun, model or provider
  substitution, source or prompt change, fixed-policy arm, behavior-cloning
  collection, or training run.

### E38 - RESOLVED POSITIVE 07-17: answer-container REVISE on sealed E37 states
- Resolution: integration passed and both final artifacts published. The
  legacy open-object branch produced zero usable answer containers, while the
  answer-container branch produced 14/14. All paired cells were treatment-only:
  zero both, 14 answer-container-only, zero open-object-only, and zero neither.
  Thus `A - O = 14`; the registered positive gate passes. The exact one-sided
  binomial probability is `1/16384 = 0.00006103515625`, and the exact
  two-sided discordance probability is `0.0001220703125`.
- The mechanism is unambiguous. The treatment completed all 14 calls on the
  first attempt and every result created one explicit revised Candidate. The
  contemporary legacy control admitted five workspace-shaped objects without
  top-level answers and exhausted parsing on nine calls after one retry each.
  It therefore produced zero answer-bearing Candidates. The treatment removed
  both the copied-lens admission failure and the retry/truncation failure on
  these sealed states.
- Symbolic correctness is secondary and more limited than container success.
  Thirteen action states targeted root Candidates and include repeated states
  from two episodes. The selected source Candidate was correct on 6/13 action
  states and the treatment revision on 7/13: six both, one treatment-only, zero
  source-only, and six neither. Collapsing repeated actions leaves 10 unique
  root episodes: three both, one treatment-only, zero source-only, and six
  neither. Thus the new contract preserved all three source-correct unique root
  episodes and repaired one of seven source-wrong episodes in this selected
  subset. This is not an end-to-end teacher gain because no controller chose or
  submitted the generated Candidates. The one child-target state has no root
  symbolic reading.
- Integration audit: the run contains 58 canonical records: one header, 14
  exact source records, 28 paired transitions, 14 paired summaries, and one run
  summary. Diagnostics contain one header and 37 canonical HTTP attempts bound
  to 28 logical calls. The 14 treatment calls all returned once; the legacy
  arm made 23 attempts, comprising five returns, nine parse retries, and nine
  terminal raises. Every pair shared the exact observation, action, worker,
  temperature, token limit, and effective source seed; only the registered
  instruction and schema differed. Checkerboard order, source bindings, call
  roles, and behavior-cloning exclusions matched. No hidden answer, credential,
  partial artifact, teacher call, or unbound call id was found.
- Usage: the treatment used 11,183 prompt and 1,856 completion tokens. The
  legacy control used 24,291 prompt and 15,496 completion tokens including its
  retries and long lens echoes. Total reported worker usage was 35,474 prompt
  plus 17,352 completion tokens across 37 HTTP attempts. Featherless metadata
  still exposes no price, so no dollar estimate is invented. Provider response
  model was `Qwen/Qwen2.5-3B-Instruct` on every attempt; provider timestamps
  span 2026-07-18 04:56:58 through 05:00:39 UTC.
- Artifact SHA-256 values: run
  `1f41eedad90115f5f872ae6a4a7bf0843b691dacf9043f972a0aca582a2b3f48`
  and diagnostics
  `bc2925e2bd59680ef858792dfaf61b83a9d9fbad369d9ad61e2f7e79dc909562`.
  The registered reading now retains
  `workspace-revise-answer-container-v1`. The next supported experiment is a
  separately preregistered E37-style shared-direct-Candidate comparison in
  which this REVISE contract is the only change. Keep the current coordinator
  prompt and model fixed for that causal test. If valid Candidates still do not
  improve terminal reward, isolate coordinator candidate-comparison prompting
  on the resulting exact post-revision decision states, then test coordinator
  model scale as a separate factor. E38 remains evaluation-only and supplies no
  behavior-cloning rows.
- Purpose: isolate whether the frozen Qwen2.5-3B worker can turn the exact
  states where the E37 teacher selected `REVISE` into usable answer-bearing
  Candidates. This is a reused-development generation-contract experiment,
  not an end-to-end teacher result, controller-prompt comparison, reward
  improvement claim, behavior-cloning collection, RL result, or population
  estimate.
- Frozen source: strict zero-network replay selects all 14 E37
  `teacher_continuation` transitions whose selected action was `REVISE`.
  Their ordered `(source_episode_index, source_turn)` keys are `(7,1)`,
  `(9,1)`, `(12,4)`, `(17,1)`, `(17,2)`, `(17,3)`, `(22,1)`, `(27,1)`,
  `(33,1)`, `(38,1)`, `(47,1)`, `(50,1)`, `(52,1)`, and `(52,2)`. Thirteen
  target root Candidates and one targets a child Candidate. The sealed E37
  run SHA-256 is
  `67b8d353122f86201dd20ad8b39545318e3af5ac634e3e837646c17f18fb0c48`,
  its diagnostics SHA-256 is
  `dda68061f83dba7d49d3f8a8c4c7d2998350230bedb809eb6d8d17dfd6e8ac20`,
  and its run-contract SHA-256 is
  `d71464f51c27f2080f385cfcffc5bc47524596324237b8dded0fd971c5ef2f28`.
  Exact reconstructed observations, actions, legacy requests, source call
  bindings, and source outcomes produce selection SHA-256
  `a82fae5ea8bc907cdd318ae4f5ddd048a8ea91208d1b07e169f817c95ea91121`.
  The all-60 label-free problem projection remains
  `a98aef83ee4eee5945d632d56e25862e884811b0b73b6bc56a153b8edcfd72a3`;
  the hidden id-answer binding remains
  `683cee1cabb4192f688f09b3be87e8f6a7bdb7dd19a06aa2027f83c2ffc7b5b3`.
- Single manipulation: from each byte-identical pre-action State, execute the
  same recorded `REVISE(target)` under two levels of the versioned
  `workspace-revise-result-contract-v1`. The control
  `workspace-revise-open-object-v1` exactly replays the E37 instruction and
  open `{type: object}` schema. The treatment
  `workspace-revise-answer-container-v1` replaces only that instruction with
  an explicit top-level Candidate template and replaces only that schema with
  a strict object requiring `answer`, allowing optional `reasoning`, and
  rejecting other top-level fields. `answer` must be a nonempty string or
  finite number. The mathematical content remains unrestricted and receives
  no learned or LLM judgment. Prompt plus schema are one linked response
  contract: guided JSON is disabled, so the schema is not provider-visible on
  the first call, while the prompt alone cannot enforce local admission.
- Frozen worker and execution: both levels use the exact E37 worker identity,
  `Qwen/Qwen2.5-3B-Instruct:featherless-ai` through the Hugging Face router and
  Featherless AI, at asserted Hub revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`. Temperature is 0.0, maximum
  completion length is 700, base seed is 0 with the original episode-turn
  offset, there is one parse retry and six 429/5xx backoff retries, timeout is
  120 seconds, response mode is JSON object, and guided JSON is disabled.
  `preceding-siblings-v1`, `verifier-ready-root-answer-v1`, depth 3, turn limit
  12, and `active-frontier-v1` are fixed. Case-level branch order alternates.
  No teacher or controller call occurs. The exact E38 run-contract SHA-256 is
  `73739a6d941d0c6153780e4ad8ab88276e147d01264102787147fdeac233f146`.
- Planned artifacts:
  `experiments/runs/2026-07-17-E38-paired-revise-answer-container-qwen25-3b-casework14.jsonl`
  and
  `experiments/runs/2026-07-17-E38-paired-revise-answer-container-qwen25-3b-casework14.diagnostics.jsonl`.
  Both final paths and both sibling `.partial` paths must be absent before
  execution. Attempts are role-bound as `open_object_revise_worker` or
  `answer_container_revise_worker`; every published transition has
  `behavior_cloning_eligible=false`.
- Pre-execution gates: re-run zero-network preparation and match all six
  source, selection, problem, hidden-binding, and run-contract digests;
  confirm the cached HF credential without printing it; confirm all four
  output paths are absent; retain the focused workspace tests; require
  repository-wide `pytest -q` to pass; and require static compilation, scoped
  diff checks, trailing-whitespace checks, and the prohibited-glyph scan to be
  clean. No hosted request is authorized before this entry and every gate.
- Integration gate, read first: publish 14 source records, 28 generation
  transitions, 14 paired summaries, one run summary, and both final artifacts.
  Both branches for every pair must start from the exact sealed observation,
  execute the exact sealed action, use the same worker identity, temperature,
  token limit, and effective seed, and differ only in the registered REVISE
  instruction and schema. Each branch must bind exactly one logical worker
  call; exhausted parse or transport errors are measured outcomes rather than
  integration failures. Branch order must follow the checkerboard schedule.
  Source requests and diagnostics must remain exactly bound, no teacher call
  may occur, and no hidden answer or credential may appear. Missing pairs,
  request drift, call-id mismatch, source drift, incomplete publication, or
  contract drift fails integration and blocks the primary reading.
- Primary measure and gate: paired production of a usable answer container,
  mechanically defined before any symbolic correctness check as an admitted
  response with a top-level `answer` whose normalized value is a nonempty
  string or finite number and which creates one revised Candidate. Let `A` be
  answer-container-only successes and `O` open-object-only successes. Report
  both marginal counts and all four paired cells. The contract screen is
  positive only if `A - O >= 5` and the exact one-sided binomial probability
  of at least `A` treatment wins among `A + O` discordant pairs under
  probability 0.5 is at most 0.10. It is negative only if the control meets
  the same rule in reverse. Otherwise it is inconclusive. Report the exact
  two-sided discordance probability as a secondary uncertainty diagnostic.
- Pre-registered reading: a positive result retains
  `workspace-revise-answer-container-v1` and authorizes one separately
  registered end-to-end shared-direct-Candidate teacher comparison in which
  this REVISE contract is the only change from E37. That experiment must occur
  before attributing reward improvement to this generation repair. A negative
  result rejects this template and selects one new worker-prompt manipulation.
  An inconclusive result retains neither benefit claim nor training data and
  uses raw response shapes to select one manipulation. Coordinator model and
  coordinator prompting remain important later factors, but are held absent
  here so they can be tested separately after the worker can create candidates.
- Secondary diagnostics, with no threshold: exhausted generation errors;
  exact raw payloads and response shapes; top-level keys; answer changes from
  the selected Candidate; symbolic correctness for the 13 root-target revised
  Candidates only; correctness of the selected source Candidate; branch-order
  strata; retries; prompt and completion tokens; provider response model and
  timestamps; and cost if the provider exposes a defensible rate. Symbolic
  checks remain the only correctness measure and cannot override the primary
  container gate.
- Cost and authorization boundary: exactly 28 logical hosted worker calls are
  expected, with at most one parse retry per call. Featherless metadata exposes
  no router price, so no dollar estimate is invented; the account has a $5
  recharge and the bounded run must fail rather than change provider. E38
  authorizes exactly one complete 14-pair run after every frozen gate. It
  authorizes no smoke call, rerun, source regeneration, teacher call, model or
  provider substitution, other prompt or schema change, end-to-end teacher
  run, behavior-cloning collection, or training run.

### E39 - RESOLVED INCONCLUSIVE 07-17: retained REVISE contract in teacher continuation
- Purpose: test whether the retained E38 answer-container generation repair
  lets the unchanged GPT-4.1-mini controller improve terminal symbolic reward
  over one exact shared Qwen2.5-3B direct Candidate. This is a reused-
  development end-to-end system screen and evaluation-only. It is not a causal
  estimate of the E38 component by itself, a coordinator-prompt or model-scale
  comparison, a behavior-cloning collection, an RL result, or a population
  claim.
- Single manipulation relative to the sealed E37 system contract: change
  `revise_result_mode` from `workspace-revise-open-object-v1` to the retained
  `workspace-revise-answer-container-v1`. The latter changes only the REVISE
  instruction and its local strict answer-bearing schema, exactly as tested in
  E38. State, action meanings and targets, direct-prefix initialization,
  preceding-sibling lens, root-result instruction, active-frontier menu,
  indexed coordinator wire, coordinator prompt, worker and coordinator models,
  sampling settings, horizon, source, reward, and symbolic verifier remain
  fixed. The new paired run contract is
  `workspace-paired-direct-teacher-contract-v1.1`. The exact E37 legacy mode
  remains byte-replayable at its original
  `d71464f51c27f2080f385cfcffc5bc47524596324237b8dded0fd971c5ef2f28`
  digest.
- Frozen source: all 60 ordered records of
  `datasets/casework-v1/test.jsonl`, now a reused development screen. The
  label-free ordered problem projection SHA-256 is
  `a98aef83ee4eee5945d632d56e25862e884811b0b73b6bc56a153b8edcfd72a3`;
  the hidden id-answer binding SHA-256 is
  `683cee1cabb4192f688f09b3be87e8f6a7bdb7dd19a06aa2027f83c2ffc7b5b3`.
  Only `id` and `problem` enter state. The hidden answer enters only local
  symbolic SUBMIT.
- Frozen worker: `Qwen/Qwen2.5-3B-Instruct:featherless-ai` through the Hugging
  Face router and Featherless AI at asserted Hub revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`, temperature 0.0, 700 maximum
  completion tokens, seed 0 with episode-turn offsets, one parse retry, six
  429/5xx backoff retries, 120-second timeout, JSON-object mode, and guided JSON
  disabled. The exact direct request creates each shared prefix Candidate once.
  Direct control submits that exact Candidate locally without another model
  call; teacher continuation starts from the same immutable post-prefix State.
- Frozen coordinator and environment: OpenAI
  `gpt-4.1-mini-2025-04-14` through `https://api.openai.com/v1`, temperature
  0.0, 64 maximum completion tokens, seed 0 with episode-turn offsets, one
  parse retry, six 429/5xx backoff retries, 120-second timeout, JSON-object mode,
  and guided JSON disabled. The coordinator uses
  `workspace-global-planner-index-prompt-v1.1` and
  `legal-menu-index-v1`. The worker uses `preceding-siblings-v1`,
  `verifier-ready-root-answer-v1`, and the retained REVISE contract. Maximum
  depth is 3 and turn limit is 12 including the shared prefix. SUBMIT makes no
  model call, and the symbolic verifier is the only reward source.
- Frozen contract: zero-network preparation selected all 60 cases and produced
  exact run-contract SHA-256
  `845d816ba9062529e72090008c70bb2e15798c21a9c86e23f330be1bbb65398a`.
  Every transition has `behavior_cloning_eligible=false`, including complete
  teacher decisions. E39 supplies no training rows regardless of outcome.
- Planned artifacts:
  `experiments/runs/2026-07-17-E39-paired-direct-teacher-revise-answer-gpt41mini-qwen25-3b-casework-test60.jsonl`
  and
  `experiments/runs/2026-07-17-E39-paired-direct-teacher-revise-answer-gpt41mini-qwen25-3b-casework-test60.diagnostics.jsonl`.
  Both final paths and both sibling `.partial` paths must be absent before
  execution. Raw attempts are role-bound as `shared_prefix_worker`,
  `teacher_continuation_worker`, or `teacher_controller`.
- Pre-execution gates: re-run zero-network preparation and match both source
  digests plus the run-contract digest; re-run the E37 preparation and match
  its legacy digest; confirm cached HF and OpenAI credentials without printing
  them; confirm all four output paths are absent; retain the focused E38/E39
  tests; require repository-wide `pytest -q` to pass; and require static
  compilation, scoped diff and whitespace checks, and the prohibited-glyph
  scan to be clean. No hosted request is authorized before this entry and all
  gates.
- Integration gate, read first: publish all 60 shared prefixes, 60 direct
  SUBMIT transitions, complete teacher continuations, 60 paired summaries,
  one run summary, and both final artifacts. Each prefix must make exactly one
  worker call, use the exact direct request, create one protected root
  Candidate, and have no exhausted generation error. Direct control must submit
  that Candidate without a model call. The first teacher observation must
  equal the direct SUBMIT observation byte for byte. Every teacher transition
  must bind one controller call and every generative action one continuation-
  worker call. Every REVISE request must use the retained instruction and
  schema; an applied REVISE must create one explicit answer-bearing Candidate.
  All actions must be legal under the bound active menu, and only local symbolic
  SUBMIT may produce reward. Missing pairs, broken forks, request drift, call-
  role mismatch, hidden fields, credentials, partial publication, or contract
  drift fails integration and blocks the performance reading. Exhausted
  continuation generation remains a measured outcome if the call is fully
  logged and bound.
- Primary measure and gate: paired terminal symbolic reward. Let `T` be cases
  solved only by teacher continuation and `D` cases solved only by direct
  submission. Report both marginal solve counts and all four paired cells. The
  screen is positive only if `T - D >= 5` and the exact one-sided binomial
  probability of at least `T` teacher wins among `T + D` discordant pairs under
  probability 0.5 is at most 0.10. It is negative only if direct satisfies the
  same rule in reverse. Otherwise it is inconclusive. Report the exact two-
  sided discordance probability as a secondary uncertainty diagnostic.
- Preservation qualifier and ceiling rule: a positive screen is baseline-
  preserving only if `D <= 3`. If direct submission solves at least 56/60, the
  net-five gate is unreachable; record the run as ceiling-limited rather than
  interpreting a missing gain as controller equivalence.
- Pre-registered reading: a baseline-preserving positive screen justifies a
  separately registered confirmation on a fresh harder verifier-covered
  source before any teacher-trajectory collection. A negative result supplies
  no training data and requires trace diagnosis. For an inconclusive result,
  separate generation from selection mechanically: count valid revised root
  Candidates, their symbolic correctness, whether the next coordinator menu
  contained them, and whether the coordinator submitted a correct revision or
  the protected baseline. If correct revisions exist but are not selected,
  the next experiment compares the current coordinator prompt against one
  versioned candidate-comparison prompt on exact sealed post-revision states.
  If revisions are mostly incorrect, iterate one worker REVISE prompt factor
  instead. Coordinator model scale follows prompt testing as a separate factor.
- Secondary diagnostics, with no threshold: direct and teacher submitted
  answers; paired results by family; first action conditioned on direct
  correctness; preservation and recovery rates; action and target counts;
  valid revision counts and top-level answers; symbolic correctness of root
  revisions; next action after each revision; baseline versus revision submit
  choice; teacher fallbacks; generation errors; horizons; oracle union; prompt
  and completion tokens by role; retries; provider response models and timing;
  and defensible cost. These cannot override integration or the primary gate.
- Cost and authorization boundary: the horizon permits at most 60 shared
  worker calls, 660 coordinator calls, and 660 continuation-worker calls, with
  actual use expected near E37. Featherless metadata exposes no router price,
  so no HF estimate is invented; the account has a $5 recharge and the run
  must fail rather than change provider. At current official GPT-4.1-mini
  rates, bounded coordinator spend is expected below $1.00. E39 authorizes
  exactly one complete 60-case run after every gate. It authorizes no smoke
  call, rerun, model or provider substitution, source, state, action, lens,
  coordinator prompt, horizon, or other worker-prompt change, fixed-policy
  arm, behavior-cloning collection, or training run.
- Execution and integration: the single authorized run completed and published
  254 canonical records and 145 diagnostic records. These comprised 192
  transitions, 60 paired summaries, one run summary, and 144 unique hosted
  calls. All 144 calls returned accepted HTTP 200 responses on their first
  attempt, with no parse or backoff retry. The exact run artifact SHA-256 is
  `c2aab70b5e71df573a1aefcd1d2310435d001dc906f178473423a509d9fc86f0`;
  the exact diagnostics SHA-256 is
  `5dec0e494a86e9d26042e9c311e672e0f91638eb3bd5fcef3bf02f7d11a3fb81`.
  Both final artifacts exist, no sibling partial remains, all call bindings and
  branch forks passed audit, and no hidden answer or credential entered model
  state or requests. Provider timestamps span 2026-07-18 05:30:05 through
  05:35:20 UTC.
- Primary result: exact shared direct submission and teacher continuation each
  solved 43/60 cases. Paired cells were 43 both, zero teacher-only, zero
  direct-only, and 17 neither. With no discordant pair, both the one-sided and
  two-sided exact probabilities are 1.0. The registered screen is therefore
  inconclusive, not positive or negative. Direct submission was below the
  56/60 ceiling rule. The anchor preserved every direct success, but the
  teacher recovered no direct failure.
- Selection audit: the teacher made 72 decisions, selecting immediate SUBMIT
  on 48 cases and REVISE then SUBMIT on 12. All 12 REVISE calls created a
  contract-valid answer-bearing root Candidate. The symbolic verifier marked
  the protected baseline correct on four of these cases and the revision
  correct on seven: four both, three revision-only, zero baseline-only, and
  five neither. Both Candidate 1 and the new Candidate were legal SUBMIT
  targets at every post-REVISE decision, yet GPT-4.1-mini selected Candidate 1
  in all 12. An oracle selector over only the available Candidates would have
  converted the same fixed run from 43/60 to 46/60. This oracle is a
  diagnostic upper bound, not a reward source or policy result.
- Cross-run qualifier: E39 and E37 issued byte-identical shared-prefix worker
  requests on all 60 cases, but only 45/60 returned byte-identical payloads
  and post-prefix observations. Temperature zero, provider seed, and identical
  request bytes did not make the hosted worker deterministic across runs.
  E39's within-run direct and teacher branches still share each Candidate
  exactly, so the registered paired efficacy reading is valid. E39 must not be
  interpreted as a causal cross-run estimate of the E38 contract factor.
- Usage and cost: 60 shared-prefix worker calls used 7,742 prompt and 8,017
  completion tokens; 12 continuation-worker calls used 9,029 prompt and 1,506
  completion tokens; 72 coordinator calls used 43,204 prompt and 432
  completion tokens. At the OpenAI rates frozen in the preregistration, the
  coordinator estimate is $0.0179728. Featherless metadata exposed no worker
  price, so no HF dollar amount is invented.
- Resolution: retain `workspace-revise-answer-container-v1` as the worker
  generation contract, but make no end-to-end improvement claim and collect no
  behavior-cloning rows. The registered trace diagnosis found usable correct
  revisions that the generic controller prompt never selected. The next
  authorized design step is a separately preregistered prompt-only comparison
  on the exact sealed post-REVISE states. It must reuse the stored Candidates,
  change only the coordinator prompt, and use local symbolic SUBMIT as the only
  correctness measure. Coordinator model scale remains a later separate
  factor.

### E40 - RESOLVED INSUFFICIENT 07-17: fixed-revision selection-corpus adequacy
- Purpose: create a broad, mechanically auditable Candidate-selection corpus
  before comparing coordinator prompts. E39's 12 teacher-selected revisions
  contain three revision-only cases and zero baseline-only cases, so always
  choosing the latest revision already attains their 7/12 available-Candidate
  oracle. That selected subset cannot distinguish mathematical comparison from
  a reversed provenance bias. E40 applies one fixed REVISE to every exact E39
  post-prefix state and asks only whether the resulting corpus contains enough
  bidirectional correctness discordance to support the next prompt experiment.
  It is a reused-development corpus screen, not a policy, teacher-efficacy,
  population, behavior-cloning, or RL experiment.
- Frozen source: all 60 exact shared-prefix `next_observation` States from E39,
  in source order. E39 run SHA-256 is
  `c2aab70b5e71df573a1aefcd1d2310435d001dc906f178473423a509d9fc86f0`,
  diagnostics SHA-256 is
  `5dec0e494a86e9d26042e9c311e672e0f91638eb3bd5fcef3bf02f7d11a3fb81`,
  and source run-contract SHA-256 is
  `845d816ba9062529e72090008c70bb2e15798c21a9c86e23f330be1bbb65398a`.
  Strict zero-network reconstruction binds all source observations, direct
  requests and payloads, source call ids, problem order, and fixed actions in
  selection SHA-256
  `8a55a2b22407603f0b1dc10bd2a17fff50fe0f208a7f2bc6bd00710c622fc6cf`.
  The label-free problem projection remains
  `a98aef83ee4eee5945d632d56e25862e884811b0b73b6bc56a153b8edcfd72a3`;
  the hidden id-answer binding remains
  `683cee1cabb4192f688f09b3be87e8f6a7bdb7dd19a06aa2027f83c2ffc7b5b3`.
  Only id, problem, exact workspace content, and legal actions enter worker
  requests. Hidden answers enter only local symbolic checks.
- Fixed action and single generation level: from each exact source State,
  execute `REVISE(protected_direct_candidate_id)` once. Every source protected
  Candidate is root-owned and the action is legal under `active-frontier-v1`.
  There is no policy or coordinator selection. The worker uses retained
  `workspace-revise-answer-container-v1`, `preceding-siblings-v1`, and
  `verifier-ready-root-answer-v1`; depth 3 and turn limit 12 are fixed. The
  answer-container schema admits a nonempty string or finite numeric answer
  with optional unverified reasoning. It does not judge truth or algebra.
- Frozen worker and execution: use E39's exact
  `Qwen/Qwen2.5-3B-Instruct:featherless-ai` identity through the Hugging Face
  router and Featherless AI at asserted Hub revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`. Temperature is 0.0, maximum
  completion length is 700, base seed is 0 with effective seed
  `12 * source_episode_index + source_turn`, there is one parse retry and six
  429/5xx backoff retries, timeout is 120 seconds, JSON-object mode is on, and
  guided JSON is off. No coordinator model is present. Exact E40 run-contract
  SHA-256 is
  `56a3fa807f92218f14c6e02a15da4448fab97daf70c6f42aa8ac5978b5a895e0`.
- Planned artifacts:
  `experiments/runs/2026-07-17-E40-fixed-revision-qwen25-3b-casework-test60.jsonl`
  and
  `experiments/runs/2026-07-17-E40-fixed-revision-qwen25-3b-casework-test60.diagnostics.jsonl`.
  Both final paths and both sibling partial paths must be absent before
  execution. Every attempt is role-bound as `fixed_revision_worker`, and every
  transition has `behavior_cloning_eligible=false`.
- Pre-execution gates: re-run zero-network preparation and match all six
  source-run, source-diagnostics, source-contract, selection, problem, and
  hidden-binding digests plus the E40 contract digest; confirm the cached HF
  credential without printing it; confirm all four output paths are absent;
  retain the 55 focused workspace tests; require repository-wide `pytest -q`
  to pass; and require static compilation, scoped diff and whitespace checks,
  and the prohibited-glyph scan to be clean. No hosted request is authorized
  before this entry and every gate.
- Integration gate, read first: publish one header, 60 source records, 60
  fixed-REVISE transitions, 60 case summaries, one run summary, and both final
  artifacts. Every transition must start from its exact sealed observation,
  target its protected root Candidate, use the retained prompt and strict
  schema, bind exactly one logical worker call, and use the registered
  effective seed. Exhausted parse or transport failure is a measured outcome
  rather than an integration failure if all attempts and the generation-error
  transition are logged. Missing cases, source drift, illegal actions, request
  drift, call mismatch, hidden fields, credentials, incomplete publication, or
  contract drift fails integration and blocks the adequacy reading.
- Primary adequacy measure and gate: count usable answer-bearing revisions and
  the four symbolic correctness cells between protected baseline and revision:
  both, baseline-only, revision-only, and neither. The corpus is adequate for
  a prompt-selection comparison only if at least 55/60 revisions are usable,
  at least five cases are baseline-only, and at least five are revision-only.
  This deterministic development gate requires both preservation and recovery
  decisions; it is not a significance test. Otherwise the corpus is
  insufficient for the proposed prompt comparison.
- Pre-registered reading: an adequate corpus authorizes implementation and
  separate preregistration of one paired coordinator experiment on these exact
  sealed post-REVISE States, with current generic and one explicit candidate-
  comparison prompt as the sole arm difference. An insufficient corpus
  authorizes no prompt result claim and no coordinator call; use the frozen
  correctness cells and raw revisions to choose one separately registered
  corpus-generation factor. In either case, E40 supplies no training rows and
  does not change the retained worker contract.
- Secondary diagnostics, with no threshold: baseline, revision, always-
  protected, always-latest, and available-Candidate oracle solve counts;
  correctness cells by family and E39 first action; answer changes; response
  shapes; generation errors; retries; prompt and completion tokens; provider
  response model and timestamps; and defensible cost. The local symbolic
  verifier is the only correctness measure.
- Cost and authorization boundary: exactly 60 logical hosted worker calls are
  expected, each with at most one parse retry. Featherless metadata exposes no
  router price, so no dollar estimate is invented; the account has a $5
  recharge and execution must fail rather than change provider. E40 authorizes
  exactly one complete 60-case run after every frozen gate. It authorizes no
  smoke call, rerun, source regeneration, coordinator call, model or provider
  substitution, prompt, schema, state, action, lens, seed, horizon, or source
  change, behavior-cloning collection, or training run.
- Execution and integration: the single authorized run completed and published
  exactly 182 canonical records and 61 diagnostic records: one header, 60
  source records, 60 fixed-REVISE transitions, 60 case summaries, one run
  summary, one diagnostic header, and 60 raw attempts. All 60 logical calls
  returned accepted HTTP 200 responses on their first attempt, with no parse
  or backoff retry. The exact run SHA-256 is
  `47e49a509fa779d1c7228477cca1b2009fb5d7232ef4a710b23b24baaf0fb355`;
  diagnostics SHA-256 is
  `9f6adfb5bb2a3c92cf5e8e0b73ed763d0bb8dc9b78b808e26df514b4fb0ed2cc`.
  Both final artifacts exist, no sibling partial remains, source observations,
  fixed actions, requests, seeds, and call ids passed exact audit, and no
  hidden answer or credential entered model state or requests. Provider
  timestamps span 2026-07-18 06:21:10 through 06:24:07 UTC.
- Primary adequacy result: all 60 calls created usable answer-bearing revised
  Candidates. Symbolic correctness cells were 43 both, zero baseline-only, six
  revision-only, and 11 neither. The protected baseline was correct on 43/60,
  the latest revision on 49/60, and the available-Candidate oracle also on
  49/60. The corpus clears usability and recovery but fails the registered
  minimum of five baseline-only preservation decisions. It is therefore
  insufficient for the proposed candidate-comparison prompt experiment.
- Mechanism diagnostics: revision changed the normalized answer on 11/60
  cases, producing six wrong-to-correct changes, zero correct-to-wrong changes,
  and five wrong-to-wrong changes. Revision-only cases were four `f1a`, one
  `f1b`, and one `f2a`. The 12 E39-selected revision calls are an exact subset
  of E40: all 12 worker requests and all 12 payloads match byte for byte under
  the same effective seeds. This contrasts with the independently generated
  direct prefixes, which were not fully reproducible across E37 and E39.
- Usage: the 60 Featherless calls used 47,002 prompt and 7,848 completion
  tokens. Provider metadata exposes no price, so no HF dollar estimate is
  invented. No coordinator or OpenAI call occurred.
- Resolution: make no candidate-comparison prompt claim and issue no
  coordinator call on this corpus. E40 shows a useful generation direction:
  under one fixed revision, the latest Candidate weakly dominates the
  protected baseline on every reused case and is symbolically correct on six
  additional cases. Because E40 has no SUBMIT transition, this is not a
  terminal-reward or teacher-policy result. The next experiment must isolate
  one corpus-generation prompt factor that encourages an independent
  alternative solution, seeking both recovery and preservation cases before
  comparing coordinator selection prompts. E40 supplies no training rows.

### E41 - RESOLVED POSITIVE, CORPUS INADEQUATE 07-17: retained versus independent REVISE prompt
- Purpose: test one worker-prompt factor and determine whether an independently
  derived alternative Candidate supplies the bidirectional correctness cases
  absent from E40. The retained arm asks Qwen2.5-3B to revise using visible
  local work. The treatment tells the same worker to solve independently from
  the problem conditions, treat the selected Candidate as untrusted, and
  derive its result before comparison. E41 is a reused-development worker-
  prompt and selection-corpus screen. It is not a controller, teacher-efficacy,
  terminal-reward, population, behavior-cloning, or RL experiment.
- Frozen source: the same 60 exact E39 shared-prefix `next_observation` States
  used by E40, in source order. E39 run SHA-256 is
  `c2aab70b5e71df573a1aefcd1d2310435d001dc906f178473423a509d9fc86f0`,
  diagnostics SHA-256 is
  `5dec0e494a86e9d26042e9c311e672e0f91638eb3bd5fcef3bf02f7d11a3fb81`,
  source run-contract SHA-256 is
  `845d816ba9062529e72090008c70bb2e15798c21a9c86e23f330be1bbb65398a`,
  and strict label-free source-selection SHA-256 is
  `8a55a2b22407603f0b1dc10bd2a17fff50fe0f208a7f2bc6bd00710c622fc6cf`.
  The label-free problem projection remains
  `a98aef83ee4eee5945d632d56e25862e884811b0b73b6bc56a153b8edcfd72a3`;
  the hidden id-answer binding remains
  `683cee1cabb4192f688f09b3be87e8f6a7bdb7dd19a06aa2027f83c2ffc7b5b3`.
  Hidden answers enter only local symbolic checks after generation.
- Single manipulation: from each exact State, execute the same legal
  `REVISE(protected_direct_candidate_id)` under two instruction modes. Control
  is retained `workspace-revise-answer-container-v1`. Treatment is
  `workspace-revise-independent-answer-container-v1`. Both use the exact same
  `workspace-revise-answer-container-v1` strict answer schema, worker system
  message, full hybrid lens, root-result suffix, transport, and local decoder.
  The implementation asserts that replacing the retained instruction with the
  independent instruction once makes request text byte-identical. Neither
  prompt judges truth; mathematical correctness is measured only afterward by
  the local symbolic verifier.
- Frozen worker and execution: both arms use
  `Qwen/Qwen2.5-3B-Instruct:featherless-ai` through the Hugging Face router and
  Featherless AI at asserted Hub revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`. Temperature is 0.0, maximum
  completion length is 700, effective seed is
  `12 * source_episode_index + source_turn`, there is one parse retry and six
  429/5xx backoff retries, timeout is 120 seconds, JSON-object mode is on, and
  guided JSON is off. `preceding-siblings-v1`,
  `verifier-ready-root-answer-v1`, active-frontier legality, depth 3, and turn
  limit 12 are fixed. Pair order alternates by case. No coordinator is present.
  Exact E41 run-contract SHA-256 is
  `611aaf66dba10fb3be5ba5ca5e08f490f2c747387656ac7e471bef1e8d04ae11`.
- Planned artifacts:
  `experiments/runs/2026-07-17-E41-paired-revision-prompt-qwen25-3b-casework-test60.jsonl`
  and
  `experiments/runs/2026-07-17-E41-paired-revision-prompt-qwen25-3b-casework-test60.diagnostics.jsonl`.
  Both final paths and both sibling partial paths must be absent. Attempts are
  role-bound as `retained_revision_worker` or
  `independent_revision_worker`; every transition has
  `behavior_cloning_eligible=false`.
- Pre-execution gates: re-run zero-network preparation and match all E39
  source, selection, problem, hidden-binding, and E41 contract digests; confirm
  the cached HF credential without printing it; confirm all four output paths
  are absent; retain the 59 focused workspace tests; require repository-wide
  `pytest -q` to pass; and require static compilation, scoped diff and
  whitespace checks, and the prohibited-glyph scan to be clean. No hosted
  request is authorized before this entry and every gate.
- Integration gate, read first: publish one header, 60 source records, 120
  transitions, 60 paired summaries, one run summary, and both final artifacts.
  Both arms for every pair must start from the exact sealed State, take the
  exact fixed action, use the same schema, model, settings, and effective seed,
  bind one logical worker call, and differ only in the registered instruction.
  Branch order must follow the checkerboard schedule. Exhausted calls remain
  measured outcomes if fully logged. Missing cases, source or request drift,
  non-isolated prompts, illegal actions, call mismatch, hidden fields,
  credentials, incomplete publication, or contract drift fails integration
  and blocks every reading.
- Primary prompt measure and gate: paired symbolic correctness of the two
  generated revisions. Let `I` be independent-only correct revisions and `R`
  retained-only correct revisions. Report both marginal solve counts and all
  four paired cells. The independent prompt is positive only if `I - R >= 5`
  and the exact one-sided binomial probability of at least `I` treatment wins
  among `I + R` discordant pairs under probability 0.5 is at most 0.10. It is
  negative only if retained meets the same rule in reverse. Otherwise it is
  inconclusive. Report the exact two-sided probability as a secondary
  uncertainty diagnostic.
- Selection-corpus qualifier: independently of the prompt-efficacy reading,
  count treatment usability and treatment correctness cells against the
  protected baseline. The treatment corpus is adequate for a later candidate-
  comparison prompt only if at least 55/60 revisions are usable, at least five
  cases are baseline-only, and at least five are independent-revision-only.
  This deterministic gate requires both preservation and recovery decisions;
  it is not a population or significance claim.
- Pre-registered reading: a positive and adequate result retains the
  independent prompt and authorizes a separately registered coordinator-prompt
  comparison on its exact post-revision States. An inconclusive but adequate
  result may also authorize that selection-mechanism experiment, but does not
  support a worker-prompt improvement claim. An inadequate result authorizes no
  coordinator call: if the treatment again weakly dominates baseline, pivot to
  a separately registered initial routing prompt; if it is mostly worse,
  reject it and select one new generation factor. A negative result rejects
  the treatment regardless of corpus adequacy. E41 supplies no training rows.
- Secondary diagnostics, with no threshold: answer containers; generation
  errors; baseline-versus-revision cells for both arms; protected, retained,
  independent, and available-Candidate oracle solve counts; answer changes;
  paired outcomes by family; branch order; exact E40 retained-arm replay;
  retries; prompt and completion tokens by role; response models and provider
  timestamps; and defensible cost. These cannot override integration or the
  primary/qualifier readings.
- Cost and authorization boundary: exactly 120 logical hosted worker calls are
  expected, each with at most one parse retry. Featherless metadata exposes no
  router price, so no dollar estimate is invented; the account has a $5
  recharge and the run must fail rather than change provider. E41 authorizes
  exactly one complete 60-pair run after every frozen gate. It authorizes no
  smoke call, rerun, source regeneration, coordinator call, model or provider
  substitution, schema, state, action, lens, seed, horizon, or source change,
  behavior-cloning collection, or training run.
- Execution and integration: the single authorized run completed and published
  exactly 242 canonical records and 121 diagnostics: one header, 60 source
  records, 120 paired transitions, 60 paired summaries, one run summary, one
  diagnostic header, and 120 raw attempts. Every call returned an accepted
  HTTP 200 response on its first attempt, with no parse or backoff retry and
  120 unique logical call ids. The exact run SHA-256 is
  `58e25eee19fa9f16506d6253e1e7c6ae73b326126f1b13feba00a3c6cd7645f6`;
  diagnostics SHA-256 is
  `3fc3a683eff07e85d755fd614fb5bc4c8afd042f8c7f416d26d9247ac07d90ce`.
  Both final artifacts exist, no partial remains, all state, action, prompt-
  isolation, seed, branch-order, and call bindings passed audit, and no hidden
  answer or credential entered model state or requests. Provider timestamps
  span 2026-07-18 06:50:41 through 06:57:42 UTC.
- Primary prompt result: retained revision was symbolically correct on 49/60
  and independent revision on 55/60. Paired cells were 49 both, six
  independent-only, zero retained-only, and five neither. Thus `I - R = 6`;
  the exact one-sided probability is `1 / 2^6 = 0.015625` and the exact
  two-sided probability is 0.03125. The registered prompt screen is positive.
  Retain `workspace-revise-independent-answer-container-v1` for the next
  routing experiment on this reused development source.
- Selection-corpus qualifier: all 60 independent calls created usable
  Candidates. Against protected baseline, correctness cells were 43 both,
  zero baseline-only, 12 independent-only, and five neither. Independent
  revision therefore weakly dominated baseline and raised Candidate
  correctness from 43/60 to 55/60, but supplied no preservation decisions.
  The registered bidirectional-corpus qualifier fails, so E41 does not
  authorize a candidate-comparison coordinator call.
- Reproducibility diagnostic: all 60 retained-arm requests exactly matched E40
  under the same states and effective seeds, but only 52/60 payloads matched
  byte for byte. Marginal retained correctness nevertheless remained 49/60
  with the same baseline correctness cells. The paired E41 reading remains
  valid because each case shares its exact source State and uses checkerboard
  branch order, but temperature zero and provider seed do not guarantee
  cross-run payload identity.
- Usage: 60 retained calls used 47,002 prompt and 7,710 completion tokens; 60
  independent calls used 49,162 prompt and 8,916 completion tokens. Total
  Featherless usage was 96,164 prompt and 16,626 completion tokens. Provider
  metadata exposed no router price, so no HF dollar estimate is invented. No
  coordinator or OpenAI call occurred.
- Resolution: keep the independent instruction as a positive worker-prompt
  factor, but make no terminal teacher-reward, population, or selection claim.
  E41's fixed independent revision is correct on 12 additional reused cases
  and no fewer cases than protected baseline. Following the preregistered
  inadequate-but-dominant reading, the next experiment is a separately
  registered initial coordinator routing prompt on exact sealed prefix States.
  It will compare generic planning with explicit REVISE-versus-SUBMIT routing,
  replay E41's independent revisions without new worker calls, and use local
  symbolic SUBMIT as the only reward source. E41 supplies no training rows.

### E42 - RESOLVED POSITIVE 07-17: generic versus explicit initial routing prompt
- Purpose: test whether coordinator prompting can realize the value of the
  retained E41 independent revision at the first post-prefix decision. E39's
  generic GPT-4.1-mini prompt selected REVISE on only 12/60 cases and later
  ignored every revision. E41 now supplies one exact independent revision for
  every source State, correct on 55/60 versus 43/60 protected baselines. E42
  compares the current generic planner prompt with one explicit
  REVISE-versus-SUBMIT routing prompt while replaying those fixed action
  outcomes. It is a reused-development prompt mechanism screen, not a live
  worker, population, behavior-cloning, or RL experiment.
- Frozen revision source: all 60 exact E41 independent-revision transitions,
  in source order. E41 run SHA-256 is
  `58e25eee19fa9f16506d6253e1e7c6ae73b326126f1b13feba00a3c6cd7645f6`,
  diagnostics SHA-256 is
  `3fc3a683eff07e85d755fd614fb5bc4c8afd042f8c7f416d26d9247ac07d90ce`,
  and run-contract SHA-256 is
  `611aaf66dba10fb3be5ba5ca5e08f490f2c747387656ac7e471bef1e8d04ae11`.
  Every replay binds the exact source observation, independent request,
  answer-bearing next State, source worker call id, protected Candidate id,
  revised Candidate id, and source-transition digest. No worker call occurs in
  E42.
- Frozen coordinator source: use E39's exact GPT-4.1-mini identity and sampling
  contract. E39 run SHA-256 is
  `c2aab70b5e71df573a1aefcd1d2310435d001dc906f178473423a509d9fc86f0`;
  run-contract SHA-256 is
  `845d816ba9062529e72090008c70bb2e15798c21a9c86e23f330be1bbb65398a`.
  Combined strict zero-network reconstruction produces label-free routing-
  selection SHA-256
  `08f9cdd31a69d1c7b2aba729b42adad538563567f402ef46cef1745a2ada9a5c`.
  The problem projection remains
  `a98aef83ee4eee5945d632d56e25862e884811b0b73b6bc56a153b8edcfd72a3`;
  hidden id-answer binding remains
  `683cee1cabb4192f688f09b3be87e8f6a7bdb7dd19a06aa2027f83c2ffc7b5b3`.
  Hidden answers enter only local terminal symbolic SUBMIT.
- Single manipulation: both arms receive the same exact active-frontier
  observation, indexed legal action menu, user message, strict index schema,
  model, settings, and effective seed. The implementation asserts their user
  text and schema are identical and their system instructions differ. Control
  is `workspace-global-planner-index-prompt-v1.1`. Treatment is
  `workspace-revision-routing-index-prompt-v1`, which explicitly explains the
  bounded choice: inspect the current Candidate; REVISE invokes one independent
  solution whose newest Candidate will be submitted next; SUBMIT sends the
  selected Candidate immediately; Candidate protection and numeric id are not
  correctness evidence. The full three-action menu remains unchanged.
- Frozen coordinator: OpenAI `gpt-4.1-mini-2025-04-14` through
  `https://api.openai.com/v1`, temperature 0.0, maximum completion length 64,
  effective seed `12 * source_episode_index + source_turn`, one parse retry,
  six 429/5xx backoff retries, timeout 120 seconds, JSON-object mode on, guided
  JSON off, active-frontier legality, indexed choice, depth 3, and turn limit
  12. Pair order alternates by case. Exact E42 run-contract SHA-256 is
  `0b86b77ad5167178d13c0c304f346813f8688da52d244acffd8e03a142ed731c`.
- Fixed continuation and reward: selected `SUBMIT(1)` executes local symbolic
  submission of the protected Candidate. Selected `REVISE(1)` replays the
  exact E41 independent next State and deterministically executes local
  `SUBMIT(2)`. A different legal action receives the environment's zero
  intermediate reward and is recorded as a bounded nonterminal outcome; no
  fabricated verifier result or worker generation is supplied. The symbolic
  SUBMIT verifier is the only positive reward source. The fixed protected
  policy is 43/60, fixed independent-revision policy is 55/60, and five cases
  are unsolved by either; these are frozen action outcomes, not E42 results.
- Planned artifacts:
  `experiments/runs/2026-07-17-E42-paired-routing-prompt-gpt41mini-e41replay-casework-test60.jsonl`
  and
  `experiments/runs/2026-07-17-E42-paired-routing-prompt-gpt41mini-e41replay-casework-test60.diagnostics.jsonl`.
  Both final and sibling partial paths must be absent. Attempts are role-bound
  as `generic_routing_controller` or `explicit_routing_controller`; every
  decision has `behavior_cloning_eligible=false`.
- Pre-execution gates: replay every E41 revision, E39 coordinator, selection,
  problem, hidden-binding, and E42 contract digest; confirm the cached OpenAI
  credential without printing it; confirm all four artifact paths are absent;
  retain the 64 focused workspace tests; require repository-wide `pytest -q`
  to pass; and require static compilation, scoped diff and whitespace checks,
  and the prohibited-glyph scan to be clean. No hosted request is authorized
  before this entry and every gate.
- Integration gate, read first: publish one header, 60 source records, 120
  decisions, 60 paired summaries, one run summary, and both final artifacts.
  Every pair must share its exact State, menu, user request, schema, model,
  settings, and effective seed and differ only in the registered system
  instruction. Every decision binds exactly one coordinator call and a legal
  action. Fixed continuation must use only protected baseline or the exact
  bound E41 revision; terminal reward must come only from local symbolic
  SUBMIT. Branch order follows the checkerboard schedule. Missing cases,
  source, request, menu, prompt-isolation, seed, replay, action, call, hidden-
  field, credential, publication, or contract drift fails integration and
  blocks the performance reading.
- Primary measure and gate: paired bounded-policy symbolic reward. Let `Q` be
  cases solved only by the explicit routing prompt and `G` cases solved only by
  the generic prompt. Report both marginal solve counts and all four paired
  cells. The routing prompt is positive only if `Q - G >= 5` and the exact
  one-sided binomial probability of at least `Q` routing wins among `Q + G`
  discordant cases under probability 0.5 is at most 0.10. It is negative only
  if generic meets the same rule in reverse. Otherwise it is inconclusive.
  Report the exact two-sided probability as a secondary diagnostic.
- Preservation and ceiling: a positive result is baseline-preserving only if
  `G <= 3`. The frozen action-outcome oracle is 55/60. If generic solves at
  least 51/60, the net-five gate is unreachable; report the comparison as
  ceiling-limited rather than controller-prompt equivalence.
- Pre-registered reading: a baseline-preserving positive result retains the
  explicit routing prompt and authorizes a separately registered live shared-
  prefix system evaluation combining the already isolated independent worker
  prompt and routing prompt, preferably on a fresh harder verifier-covered
  source. An inconclusive result with unused oracle headroom triggers trace
  diagnosis by action, direct correctness, and revision-only status before any
  model-scale run. A negative result rejects the routing prompt. Coordinator
  model capacity remains a separate later factor. E42 supplies no training
  rows regardless of outcome.
- Secondary diagnostics, with no threshold: actions and routes by arm; reward
  by E41 baseline/revision cell and family; oracle realization; nonterminal
  choices; fallbacks; exact generic action replay versus E39; branch order;
  prompt and completion tokens by role; retries; response models and provider
  timestamps; and defensible cost. These cannot override integration or the
  primary gate.
- Cost and authorization boundary: exactly 120 logical OpenAI calls are
  expected, each with at most one parse retry. At the rates frozen in E39,
  estimated spend is below $0.10. E42 authorizes exactly one complete 60-pair
  run after every gate. It authorizes no smoke call, rerun, worker call, source
  regeneration, model or provider substitution, user presentation, schema,
  state, action menu, replay, seed, horizon, or source change, behavior-cloning
  collection, or training run.
- Execution and integration: the single authorized run completed and published
  exactly 242 canonical records and 121 diagnostics: one header, 60 sources,
  120 decisions, 60 paired summaries, one run summary, one diagnostic header,
  and 120 raw attempts. Every call returned an accepted HTTP 200 response on
  its first attempt, with no parse or backoff retry and 120 unique call ids.
  The exact run SHA-256 is
  `8f886f109e9ba8f4f7d25aed6db2cbc1ae0140ffd92bcd51ac16cf6dcd0bd64d`;
  diagnostics SHA-256 is
  `8867f2b4dd2c2f2c6311b3c85d23c3fac036dfc1e0838087ba9af15e496f8eb6`.
  Both final artifacts exist, no partial remains, every source, replay,
  prompt-isolation, menu, seed, action, continuation, reward, branch-order, and
  call binding passed audit, and no hidden answer or credential entered model
  state or requests. No worker call occurred. Provider timestamps span
  2026-07-18 07:30:47 through 07:32:38 UTC.
- Primary result: generic routing solved 46/60 and explicit routing solved
  51/60. Paired cells were 46 both, five routing-only, zero generic-only, and
  nine neither. Thus `Q - G = 5`; the exact one-sided probability is
  `1 / 2^5 = 0.03125` and the exact two-sided probability is 0.0625. The
  registered prompt screen is positive and baseline-preserving with zero
  generic-only loss. It is not ceiling-limited because generic was below
  51/60. Retain `workspace-revision-routing-index-prompt-v1` for the next live
  system evaluation.
- Routing mechanism: generic selected 50 SUBMIT, eight REVISE, and two
  DECOMPOSE actions. Explicit routing selected 41 SUBMIT and 19 REVISE, with no
  DECOMPOSE. In the 43 cases where baseline and independent revision were both
  correct, generic realized 42 after one nonterminal DECOMPOSE while routing
  realized all 43. In the 12 revision-only cases, generic selected REVISE and
  solved four; routing selected REVISE and solved eight. Both prompts failed
  all five neither cases. Explicit routing therefore preserved every baseline
  success and realized 8/12 available recoveries, leaving four correct sealed
  revisions unused. Its 51/60 result remains four points below the 55/60 fixed
  action-outcome oracle.
- Reproducibility diagnostic: E42 generic and the E39 first coordinator action
  used exact same prompts, source States, model settings, and effective seeds,
  yet actions matched on 53/60 cases. This is consistent with prior hosted
  nondeterminism and prevents interpreting cross-run action changes as a
  contract effect. The E42 paired prompt reading is internally controlled and
  uses checkerboard branch order.
- Usage and cost: 60 generic calls used 34,474 prompt and 360 completion
  tokens; 60 routing calls used 36,574 prompt and 360 completion tokens. Total
  coordinator usage was 71,048 prompt and 720 completion tokens. At the rates
  frozen in E39, estimated OpenAI spend is $0.0295712. No HF usage occurred.
- Resolution: retain the explicit routing prompt as a positive coordinator-
  prompt factor on this reused replay screen. This is the first controlled
  coordinator prompt comparison in the redesigned workspace to improve local
  verifier-scored terminal outcomes, but it is not yet a live end-to-end
  teacher advantage because worker generations were replayed from E41. The
  next experiment is a separately registered live shared-prefix system
  evaluation that combines the already isolated independent worker prompt and
  routing prompt, preferably on a fresh harder verifier-covered source. No E42
  decision is behavior-cloning eligible and no training row is produced.

### E43 - COMPLETE 07-18: terminal-value target census, all gates failed
- Purpose: test the narrow post-E42 research hypothesis before training another
  controller. The hypothesis is not that a model can localize an erroneous
  step. It is that, under one additional worker call, the target maximizing
  expected terminal symbolic reward can differ from both the first erroneous
  node and every global fixed action. E43 is an action-outcome census and
  headroom gate, not a learned-policy, teacher, behavior-cloning, or RL run.
- Fresh source: generate 60 previously unused casework problems with the
  existing construction-verified generator, seed 44, scale 1, and exactly the
  balanced families `f1a,f1b,f2a,f2b`. The planned source path is
  `experiments/problems/2026-07-18-E43-casework-target60.jsonl`. The model sees
  only each natural-language root problem and the selected branch task. Hidden
  root answers, family ids, parameters, branch answers, and live/dead labels
  remain outside every model request and published policy state.
  Pre-execution amendment: seed 43 produced one content-id collision with the
  sealed E35 source (`cw1-183be22c063c`). Seed 44 was checked against all prior
  run problem ids and had zero collisions. This amendment and collision check
  occurred before any hosted call or contract digest was created.
- Fixed graph: project every problem into one root and two ordered branch
  goals from its construction metadata. No LLM performs decomposition. Each
  branch asks for its integer solution if it satisfies the stated branch
  domain, or JSON null otherwise. The initial structured state is produced by
  one worker call per branch. A deterministic reducer consumes only the two
  returned answer fields and applies the root question operation (sum,
  largest, smallest, or unique surviving value). It does not inspect
  reasoning, compare against hidden labels, or call a verifier.
- Action set: `SUBMIT`, `RETRY_ROOT`, `RETRY_BRANCH_0`, and
  `RETRY_BRANCH_1`. `SUBMIT` sends the current deterministic root Candidate.
  `RETRY_ROOT` makes one independent whole-problem call and sends that answer.
  A branch retry independently replaces only the selected branch Candidate,
  reruns the same deterministic reducer, and sends its answer. Every action is
  followed by exactly one local terminal check. The existing symbolic answer
  verifier is the only reward source. No intermediate candidate is rejected
  for algebra, truth, relevance, or agreement with hidden metadata.
- Worker: Hugging Face routed inference through Featherless,
  `Qwen/Qwen2.5-3B-Instruct:featherless-ai`, immutable revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`, temperature 0.2, maximum
  completion length 256, JSON-object requests without guided JSON, one parse
  retry, six 429/5xx backoff retries, and timeout 120 seconds. Effective seeds
  are role-, case-, target-, and replicate-bound. Initial branch and direct
  call order and retry-target order use deterministic checkerboard schedules.
- Repetitions and cross-fitting: collect one initial call for each branch and
  one independent direct root call, then four independent retry calls for
  each of the three retry targets. Replicates 0-1 are the discovery half and
  replicates 2-3 are the evaluation half. The cross-fitted value policy chooses
  the action with greatest discovery mean separately for each problem, then
  reports only its evaluation-half mean. `SUBMIT` has its deterministic
  current reward in both halves. Ties prefer `SUBMIT`, then `RETRY_ROOT`, then
  the two branch targets, so a focused intervention must show positive
  evidence to displace a simpler action.
- Fixed and localization comparisons: report initial direct, structured
  `SUBMIT`, each fixed retry target, the best global fixed action selected on
  discovery and scored on evaluation, a three-call whole-problem majority
  baseline, the cross-fitted value policy, and a descriptive realized oracle.
  After collection only, derive exact branch correctness from hidden generator
  metadata. The first-error oracle retries the first incorrect branch, or
  submits when both branches are correct, and is scored on the same evaluation
  replicates. Hidden correctness is a diagnostic policy only and never becomes
  reward, training input, worker context, or an LLM judgment.
- Primary value-allocation gate: E43 supports instance-level target learning
  only if the cross-fitted value policy exceeds the discovery-selected best
  global fixed action by at least 0.08 evaluation reward per problem and the
  problem-bootstrap 90 percent confidence interval for that difference has a
  lower endpoint above zero. Otherwise the best fixed action explains the
  available action value and pointer training is blocked.
- Primary novelty gate: the value policy must also exceed the hidden
  first-error oracle by at least 0.05 evaluation reward per problem, and at
  least five problems must have a value-selected action different from the
  first-error action with a positive evaluation advantage for the value
  action. Otherwise E43 supplies no evidence that intervention selection is a
  different objective from error localization, and that paper framing is
  rejected.
- Regime and support gates: at least ten problems must show different
  evaluation rewards across the three retry targets, and structured `SUBMIT`
  accuracy must lie from 0.20 through 0.85 inclusive. A failure of either gate
  means this source/model pairing cannot identify a target policy. It
  authorizes a separately registered difficulty or worker change, not pointer
  training or reinterpretation of E43.
- Secondary diagnostics, with no threshold: direct and structured solve
  counts; paired cells against direct; per-family results; malformed answer
  fields; initial semantic-error patterns; target discovery/evaluation
  stability; action counts; root-versus-focused retry outcomes; matched
  three-call direct majority; prompt and completion tokens; retries; response
  models and provider timestamps; and defensible provider-reported cost.
  Secondary measures cannot override a failed integration, value-allocation,
  novelty, regime, or support gate.
- Planned artifacts:
  `experiments/runs/2026-07-18-E43-target-value-qwen25-3b-casework60.jsonl`
  and its sibling `.diagnostics.jsonl`. Canonical records must include one
  header, 60 label-free source states, every initial call, all 720 retry
  transitions, 60 case summaries, and one run summary. Raw attempts are
  role-bound. The final and partial paths must be absent before execution, and
  hidden answers or credentials must not appear in model requests or
  label-free state records.
- Pre-execution gates: generate and bind the exact fresh source; prepare and
  record its source, hidden-evaluation, and run-contract SHA-256 digests;
  confirm the cached HF credential without printing it; require focused tests
  and repository-wide `pytest -q` to pass; require static compilation, scoped
  diff and whitespace checks, and the prohibited-glyph scan to be clean; and
  audit the prepared prompt/state records for hidden-field absence. No hosted
  request is authorized before this entry and all gates.
- Prepared contract, recorded before hosted execution:
  dataset `130c29c1476cb017a992d5c5889ce2f6e3be12b8f9bbaccfec00a1e6a6e47486`;
  label-free projection
  `3ea03c4f2a9534382ea85c97b3530392e38778c1e74f9cf1bf3dd429cc5f6e3e`;
  hidden evaluation binding
  `8bcd760642960394859a6415298fb7aa3f0a73e6a338ce12347dfda3536a085f`;
  run contract
  `2ef4ffe2a78089cf7c8a1fc324c90a6b3cbcff6612e437cbd47984bef699bca0`.
  Pre-execution audit superseded projection
  `b61c93b6f12887a5839bb66bd71eab61dcad7e761333fb0c005b6a8bc13a4880`
  and run contract
  `1ecceb0007d5babca70346eb7b203551d8029eac82a9740b5bce5fd8e01ecbcf`
  because the source projection still carried the hidden family id. The
  family key was removed before hosted execution; prompts were already free
  of that field. Dataset bytes and hidden evaluation binding did not change.
- Cost and authorization boundary: 60 cases authorize exactly 180 logical
  initial calls and 720 logical retry calls, each with at most one parse retry.
  Provider metadata currently exposes no route price, so no dollar estimate is
  invented. Stop before any teacher selector or training call. E44 is
  authorized only if every E43 integration and primary gate passes and is
  separately pre-registered.
- Execution and integrity: the sealed run completed all 60 cases, 900 logical
  calls, and 720 retry transitions under run contract
  `2ef4ffe2a78089cf7c8a1fc324c90a6b3cbcff6612e437cbd47984bef699bca0`.
  The canonical artifact has exactly 1,742 records: one header, 60 source
  states, 900 logical calls, 720 transitions, 60 case summaries, and one run
  summary. Its SHA-256 is
  `8b91b79e30ba352d75b4a900ec7b2786714bb7efc50bb09c62ac428b3c389b8f`.
  The diagnostic artifact has one header and 1,277 role-bound HTTP attempts;
  its SHA-256 is
  `8852a47459acba4d4470855ae0a98a27e5b6b5a8dd291dc5813afb52d3830f5b`.
  All 900 logical call ids have a unique matching diagnostic call id. Every
  HTTP attempt returned status 200. Credential and hidden-field scans were
  empty. Final repository tests passed: 965 passed and one CUDA-only test was
  skipped.
- Integration result: 524 logical calls returned an accepted Candidate and
  376 ended in a schema-level generation error after the registered retry.
  There were 70,927 completion tokens and 207,060 prompt tokens over 1,277
  provider attempts. The phase breakdown exposes a prompt-contract failure:
  initial branch 0 was invalid on 60/60 calls, initial branch 1 on 60/60, and
  initial root on 34/60. Retry errors were 33/240 for branch 0, 34/240 for
  branch 1, and 155/240 for root. Invalid payloads usually replaced the
  required `answer` key with `x`, `solution`, `value`, or a question-specific
  key. Some retry payloads copied `usable` or `error_type` from the displayed
  current Candidate. Thus both initial structured Candidates were unusable on
  every case. `SUBMIT` necessarily returned zero, and replacing only one
  branch necessarily left the other unusable, so both branch-retry arms also
  returned zero. This is an interface-invalid experiment, not a clean test of
  target selection.
- Registered measures: direct initial accuracy was 1/60 (0.0167), matched
  direct majority-of-three was 2/60 (0.0333), and structured `SUBMIT` was
  0/60. Evaluation accuracy was 0 for both fixed branch retries and 0.025 for
  fixed root retry, making `RETRY_ROOT` the discovery-selected global fixed
  action. The cross-fit value policy scored 0.0167, below the best fixed score
  by 0.0083 with bootstrap 90% interval [-0.025, 0]. It selected `SUBMIT` on
  58 cases and `RETRY_ROOT` on two. The hidden first-error policy scored zero;
  value exceeded it by only 0.0167, with one positive target disagreement.
  The realized oracle was 0.025. Only two problems had heterogeneous
  evaluation rewards, and structured-submit accuracy was zero. Allocation,
  novelty, regime, and support gates all failed. All six root-retry successes
  across the four repetitions occurred in `f2a`; every other family/action
  total was zero.
- Reading: E43 produces no evidence for learnable target-value allocation and
  no behavior-cloning row. E44, a selector comparison, and pointer training
  are not authorized. The next defensible experiment is separately
  pre-registering one prompt-only manipulation that gives initial and retry
  calls the same exact two-key JSON demonstration and removes environment-only
  Candidate fields from the displayed retry attempt. It must first restore a
  usable structured regime before target-policy experiments resume.

### E43R1 - COMPLETE 07-18: JSON integration fixed, structured regime failed
- Purpose: determine whether E43 failed because Qwen2.5-3B was not shown the
  required JSON shape. This is an integration calibration, not a target-value,
  selector, teacher, behavior-cloning, or RL experiment. It changes exactly
  one factor: append the same canonical output-contract suffix to every root
  and branch user prompt, after all task and retry context. The suffix requires
  exactly `{"reasoning":"brief derivation","answer":0}`, with no other keys,
  and permits JSON null for an invalid branch. System messages, schemas,
  strict validation, current-Candidate presentation, reducer, verifier,
  sampling, model, and call order remain unchanged.
- Paired source and baseline: use the first 12 ordered E43 problems, which are
  exactly three from each of `f1a,f1b,f2a,f2b`. Bind the sealed E43 artifact
  SHA-256
  `8b91b79e30ba352d75b4a900ec7b2786714bb7efc50bb09c62ac428b3c389b8f`
  and extract matched E43 acceptance outcomes for the same problem, phase,
  target, and repetition-zero call. Hidden metadata remains outside every
  prompt and visible state.
- Calls and seeds: make six logical calls per problem: initial root, initial
  branch 0, initial branch 1, retry root repetition zero, retry branch 0
  repetition zero, and retry branch 1 repetition zero. The 72 logical calls
  reuse E43's exact seeds. The frozen worker, provider route, revision,
  temperature 0.2, maximum 256 tokens, strict candidate schema, one parse
  retry, six backoff retries, and 120-second timeout remain unchanged.
- State and reward: initial structured submission reduces the two new initial
  branch Candidates. Each branch retry replaces only its selected new initial
  Candidate, while root retry is independent. The symbolic terminal verifier
  is the only reward. Schema-invalid payloads become unusable Candidates; no
  algebra, relevance, or truth filter is added.
- Primary integration gate: at least 11/12 initial branch-0 calls, 11/12
  initial branch-1 calls, 11/12 retry branch-0 calls, and 11/12 retry branch-1
  calls must be schema accepted. At least 20/24 combined root calls must be
  accepted. The matched initial-branch acceptance gain over E43 must be at
  least 0.80. Failure stops prompt repair and authorizes no full census.
- Regime gate: new structured-submit accuracy must be from 3/12 through 10/12
  inclusive. Below 3/12 means the worker or task prompt still cannot support
  the graph after interface repair. Above 10/12 leaves too little initial
  error headroom for target allocation. Both integration and regime gates
  must pass before a separately pre-registered repaired full census.
- Secondary diagnostics: paired schema outcomes by phase and target; invalid
  key sets; initial branch semantic correctness; direct, structured-submit,
  and each one-retry terminal reward; family counts; token usage; attempt
  outcomes; and provider metadata. These do not override either primary gate.
- Planned artifacts:
  `experiments/runs/2026-07-18-E43R1-two-key-prompt-qwen25-3b-casework12.jsonl`
  and sibling `.diagnostics.jsonl`. Canonical records must contain one header,
  12 label-free source states, 72 logical calls, 36 retry transitions, 12 case
  summaries, and one run summary. Raw attempts are role-bound and credentials
  are absent. No-clobber publication and partial-file rules match E43.
- Pre-execution gates: bind source projection, hidden evaluation, baseline run,
  prompt, and run-contract digests; require focused and repository-wide tests,
  compilation, diff and whitespace checks, prohibited-glyph scan, cached HF
  credential, absent output paths, and a forbidden-field prompt/state audit.
  No hosted call is authorized before this entry and all gates.
- Prepared contract, recorded before hosted execution: dataset
  `130c29c1476cb017a992d5c5889ce2f6e3be12b8f9bbaccfec00a1e6a6e47486`;
  12-problem label-free projection
  `34388c43fc0aeca7dc1ed5e0f0d5afe0428dd78490138f0a10dca1710c057f8a`;
  hidden evaluation binding
  `dd721635e413da4a255f89cf14201d5d4d5f8ac514bd22750a0941dc14934f50`;
  matched baseline selection
  `f37ce7ad3172c1a59139b3ce4e5c7a619bb4d7cb19596bdfabe30a07b098339e`;
  run contract
  `2f21611bc4b0292906d4cfef94a1c6342ce049198b7f82ebeffb8ac1bb34e87d`.
- Cost and stopping boundary: authorize exactly 72 logical worker calls with
  at most one parse retry. Stop after E43R1 analysis. No selector, teacher, or
  training call is authorized by this entry.
- Execution and integrity: the run completed exactly 12 cases, 72 logical
  calls, and 36 retry transitions under contract
  `2f21611bc4b0292906d4cfef94a1c6342ce049198b7f82ebeffb8ac1bb34e87d`.
  The canonical artifact has the registered 134 records and SHA-256
  `9292cb78dcc0ad3727c64851d4423151d272c3bd3fc8b9b82c54aac3a01ea2b5`.
  The diagnostic artifact has one header and 72 role-bound attempts, with
  SHA-256
  `143f35436e51cd9bbaf6c312e7ba53485d77ea496f696c9b678ce2db118a81b1`.
  Every attempt returned status 200 and an accepted payload. Usage was 13,946
  prompt tokens and 4,620 completion tokens. No generation, parse, schema,
  transport, credential, or hidden-field error occurred.
- Integration gate: all six call strata were accepted on 12/12 calls. The
  matched E43 counts were 0/12 for each initial branch, 6/12 initial root,
  10/12 retry branch 0, 9/12 retry branch 1, and 5/12 retry root. E43R1's
  initial-branch acceptance gain was therefore exactly 1.00, and combined root
  acceptance was 24/24. Every registered integration threshold passed. The
  causal reading is narrow and clear: placing the exact two-key contract last
  in the prompt completely repaired Qwen2.5-3B's JSON interface on this slice.
- Regime gate: direct initial and independent root retry each solved 6/12.
  Structured initial submission solved only 1/12, below the registered 3/12
  floor. Branch-0 and branch-1 retry each solved 2/12. Initial branch semantic
  correctness was 5/12 for each branch, and no case had both branches marked
  semantically correct. The sole structured success came from two wrong branch
  values whose deterministic reduction happened to equal the correct root
  scalar. Thus parseability did not restore useful decomposed state.
- Reading: the regime gate and combined gate failed. No repaired full census,
  selector, teacher, behavior cloning, or pointer training is authorized.
  E43R1 shows that output-contract engineering is necessary but not sufficient:
  for this 3B worker and deterministic branch lens, focused linear subproblems
  were materially less reliable than direct whole-problem prompting. The next
  research decision must change one substantive factor, such as worker model,
  lens/task wording, or tolerant output canonicalization, and calibrate it on a
  small paired slice before revisiting target selection.

### E43R2 - PRE-REGISTERED 07-18: tolerant first-attempt payload replay
- Purpose: test whether E43's strict Candidate key contract discarded useful
  initial structured work. This is a sealed offline counterfactual replay, not
  a new worker, target-value, selector, teacher, behavior-cloning, or RL run.
  It changes one factor: replace strict two-key schema acceptance with a
  deterministic, label-blind payload canonicalizer. No hosted call is made.
- Sealed inputs: bind the E43 canonical run SHA-256
  `8b91b79e30ba352d75b4a900ec7b2786714bb7efc50bb09c62ac428b3c389b8f`,
  diagnostic SHA-256
  `8852a47459acba4d4470855ae0a98a27e5b6b5a8dd291dc5813afb52d3830f5b`,
  run contract
  `2ef4ffe2a78089cf7c8a1fc324c90a6b3cbcff6612e437cbd47984bef699bca0`,
  and the same 60-problem E43 dataset, label-free projection, and hidden
  evaluation binding. Only E43 initial root, branch-0, and branch-1 calls are
  eligible, exactly 180 logical calls.
- First-attempt rule: for each eligible logical call, use only the HTTP attempt
  with the smallest `attempt_index`, which must be zero. This is the response a
  tolerant interface would have consumed before E43's corrective parse retry.
  Later attempts are excluded. Root retries remain independent but are not
  needed for this question. Branch retries and all target-value outcomes are
  excluded because their prompts were conditioned on E43's unusable initial
  Candidates and are not valid counterfactual transitions.
- Canonicalizer, fixed before replay: parse a top-level JSON object, first by
  exact JSON and then by the client's existing outer-object extraction rule.
  A top-level `answer` scalar takes priority. Otherwise remove only the exact
  metadata keys `reasoning`, `reason`, `explanation`, `work`, `steps`,
  `analysis`, `usable`, and `error_type`. Accept the remaining value only when
  exactly one scalar field remains, or when every remaining scalar normalizes
  to the same value. Scalars are JSON string, finite number, or null; booleans,
  arrays, objects, empty strings, conflicting values, and unparsable content
  are unusable. Numeric strings use the existing finite-integer reducer. No
  key-name whitelist, algebra check, hidden label, verifier call, or semantic
  repair participates in canonicalization.
- Reconstruction: canonicalize initial root and both initial branches. Apply
  E43's unchanged deterministic two-branch reducer and terminal symbolic
  verifier. Compute direct-initial and structured-submit reward. Hidden branch
  semantic correctness is a post-replay diagnostic only.
- Integration gate: at least 54/60 calls in each of initial root, initial
  branch 0, and initial branch 1 must canonicalize to a usable Candidate. A
  failure means tolerant field naming alone cannot provide a stable worker
  interface.
- Regime gate: structured-submit accuracy must be from 12/60 through 51/60
  inclusive, and each branch must be semantically correct on at least 18/60
  cases. Below those floors means the branch lens remains too weak even after
  accepting its payload; above the submit ceiling leaves insufficient initial
  target headroom. Integration and regime gates must both pass before a live
  tolerant-retry calibration is authorized.
- Secondary diagnostics: canonicalization disposition and source field names;
  paired strict-versus-tolerant acceptance; direct and structured solve counts;
  paired direct/structured cells; family counts; branch correctness patterns;
  compensating-error roots; and source-attempt identity checks. Secondary
  diagnostics cannot override a failed gate.
- Planned artifact:
  `experiments/runs/2026-07-18-E43R2-tolerant-first-attempt-replay-casework60.jsonl`.
  It must contain one header, 180 canonicalization records, 60 case summaries,
  and one run summary. The sealed diagnostic remains the raw-attempt source;
  no new diagnostic artifact is created. Publication is no-clobber and binds
  all source hashes and the exact selected-attempt digest.
- Pre-execution gates: implement and test the canonicalizer and strict source
  loader; print and record all prepared digests; require focused and full
  repository tests, compilation, diff and whitespace checks, prohibited-glyph
  scan, absent output path, and hidden-field audit. No reward computation is
  authorized before this entry and all gates.
- Prepared contract, recorded before replay: dataset
  `130c29c1476cb017a992d5c5889ce2f6e3be12b8f9bbaccfec00a1e6a6e47486`;
  label-free projection
  `3ea03c4f2a9534382ea85c97b3530392e38778c1e74f9cf1bf3dd429cc5f6e3e`;
  hidden evaluation binding
  `8bcd760642960394859a6415298fb7aa3f0a73e6a338ce12347dfda3536a085f`;
  selected first attempts
  `47a97f48f113ea0e72e2a627619c65730d36947d230d8ae31d900e0301818940`;
  run contract
  `06e045e371c75cfa5bd890eeb8fe61630e63ee60a2a48d51911790ea5ab6f3d0`.
- Stopping boundary: stop after the registered replay and audit. No live call,
  selector, teacher, or training action is authorized by this entry.
- Execution and integrity: the sealed replay completed all 60 cases without a
  hosted request. The published artifact contains exactly 242 records: one
  header, 180 canonicalizations, 60 case summaries, and one run summary. Its
  SHA-256 is
  `28a1b27575087979ecfa72f3f2753ba0c913daf9f2d4e52fb9c2d179320a02e3`.
  All 180 `(problem_id, target)` keys are unique, all selected source attempts
  have attempt index zero, and every selected `(call_id, attempt_index)` exists
  in the sealed E43 diagnostic. The run contract and every prepared digest
  exactly matched the pre-registration. Independent `jq` recomputation from
  the 60 case rows reproduced the reported solve, usability, semantic-branch,
  and paired-outcome counts. JSON framing, no-clobber publication, partial-file
  absence, ASCII, credential, and hidden-field scans were clean. Focused tests
  passed 22/22 and the repository-wide gate passed with 981 tests, one
  CUDA-only skip, and one pre-existing PyTorch warning.
- Integration result: tolerant canonicalization yielded usable Candidates for
  58/60 initial root calls and 60/60 calls in each initial branch. The two root
  failures were nonscalar payloads; the other 178 payloads used either the
  exact `answer` key or one unambiguous scalar field. All three registered
  54/60 usability floors passed. Strict E43 acceptance on the same first
  attempts had been 26/60 for root and 0/60 for both branches, so strict field
  naming had discarded substantial syntactically unambiguous work.
- Regime result: direct initial submission solved 13/60 and tolerant structured
  submission solved 18/60. Branch 0 was semantically correct on 30/60 cases and
  branch 1 on 27/60, above both registered 18/60 floors. The paired terminal
  cells were seven both, six direct-only, 11 structured-only, and 36 neither.
  Four structured successes were compensating-error roots. The structured
  18/60 score lies inside the registered 12/60 through 51/60 interval, so both
  the integration and regime gates passed.
- Reading and resolution: E43R2 is RESOLVED POSITIVE. A deterministic,
  label-blind tolerant boundary recovers a usable decomposed state and
  complementary terminal successes from the sealed E43 first attempts. This
  overturns only E43's conclusion that the structured state was unusable. It
  does not estimate retry-target values: E43's live retry prompts were
  conditioned on strict unusable Candidates and remain invalid as tolerant
  counterfactual transitions. A separately pre-registered live tolerant-retry
  calibration is authorized, but no selector, teacher, behavior cloning, or RL
  is authorized. The complete-answer root-versus-target observability pivot is
  also a separate estimand: it removes branch reduction and writeback, and must
  receive its own registration rather than being described as an E43 repair.

### E44I - PRE-REGISTERED 07-18: complete-answer action-census integration
- Purpose and scope: exercise the provider boundary and fail-closed artifact
  pipeline for a new complete-answer revision-target census. This is an
  integration smoke, not an efficacy, semantic observability, selector,
  controller, behavior-cloning, RL, or MCTS result. It changes the E43 design
  by removing child answer reduction, writeback, and intermediate correctness
  decisions. No accuracy or action-complementarity threshold is permitted at
  this stage.
- Frozen state: generate one Qwen base draft, then ask the state builder to
  segment the displayed problem and untrusted draft into exactly three ordered
  semantic nodes. Schema-valid node text is admitted without judging algebra,
  truth, relevance, or intermediate correctness. The visible state contains
  the complete problem, base draft, all three nodes, topology, phase, remaining
  per-counterfactual budget, four hosted `REVISE` actions, and the zero-call
  `SUBMIT(root)` control. Failed builders create bounded fallback nodes but
  remain in the all-case analysis and fail the whole-stage builder gate.
- Frozen hosted actions: `(REVISE, root)`, `(REVISE, node_1)`,
  `(REVISE, node_2)`, and `(REVISE, node_3)`. Every action sees the identical
  full state and complete-answer instruction; the selected target ID is the
  only prompt difference. Each call returns a complete root answer. The
  deterministic `SUBMIT(root)` value is the verifier reward of the frozen base
  draft and makes no hosted call.
- Repetition and order: four action repetitions per state. Repetitions zero
  and one are discovery; two and three are evaluation. A problem-ID-ranked
  four-row Williams schedule makes every target occupy every call position
  once and makes every other target its predecessor once. All targets within
  one state and repetition share the registered seed. Discovery-selected
  results are explicitly a within-state reward-probed repeatability oracle,
  not cross-problem cross-fitting or a deployable state-to-action policy.
- Reward isolation: candidate collection and logging use no hidden label.
  All 16 hosted revision calls for a case finish before the symbolic verifier
  is invoked on any action candidate. The symbolic terminal verifier is the
  only reward source. Unusable outputs receive reward zero. No LLM judge,
  correctness critic, reward model, algebra filter, or semantic repair is
  present.
- Source: first two rows of
  `experiments/problems/2026-07-18-E44-action-census-calibration14.jsonl`,
  problem IDs `cw1-3ed474a04b63` and `cw1-bc3cca7bcfbf`. The 14-row source
  regenerates byte-for-byte with `generate_casework.py`, seed 45, families
  `f4,f6,f7,f8`, scale 1, and split `calibration`. Its SHA-256 is
  `4d3ae6dc8a2cbcbbb0a8d4292a9eb0553064edf60b58fe78d6dbfcd4577c2b5e`.
  All 14 IDs are unique and have zero collisions with stored JSONL artifacts.
  Family, answer, symbolic form, solution set, difficulty, parameters, and
  provenance are absent from model-visible projections.
- Models: frozen worker request model
  `Qwen/Qwen2.5-3B-Instruct:featherless-ai` through the Hugging Face router and
  Featherless, with Hub revision metadata
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`; temperature 0.2, maximum 384
  tokens, seed 440000, one parse retry, and six backoff retries. Frozen state
  builder `gpt-4.1-mini-2025-04-14` through OpenAI; temperature 0.0, maximum
  384 tokens, seed 441000, one parse retry, and six backoff retries. The Hub
  mapping and OpenAI model lookup were live before registration. Provider and
  revision values are registered request-side metadata; neither provider
  attests an immutable served weight revision, and the audit must report that
  limitation.
- Exact size: two cases, 36 logical calls total (two base, two builder, and 32
  action calls), 32 transitions, and 42 canonical run records. The diagnostic
  has one header and at least 36 attempt records; retries may increase the row
  count. It must bind exactly 36 unique call IDs and stay within each model's
  registered retry budgets. `SUBMIT` adds no call, transition, or record.
- Integration gate: require 2/2 usable base drafts, 2/2 schema-usable builders,
  2/2 complete-node states, 2/2 exact-text-distinct-node states, and 32/32
  usable revision outputs. Require the complete target-by-repetition grid,
  Williams order, common seeds, target-only prompt difference, raw-attempt
  retry replay, provider-response projection, usage replay, exact record field
  sets, symbolic reward replay, hidden-field and credential scans, ASCII JSONL,
  no-clobber publication, and no remaining partial files. First-attempt
  acceptance and retry counts are reported interface diagnostics, not gates.
  Any failure stops interpretation; no case may be filtered or replaced.
- Planned artifacts:
  `experiments/runs/2026-07-18-E44I-action-census-smoke2.jsonl` and
  `experiments/runs/2026-07-18-E44I-action-census-smoke2.diagnostics.jsonl`.
  The partial artifacts are audited before atomic no-clobber publication, and
  the sealed final artifacts are audited again by the runner.
- Prepared bindings: label-free problem projection
  `67ae2df4d822030c55c4d58e810ec6801ea8054cabc99cdf9cfad93e6825f8b4`;
  hidden evaluation
  `e5cead8dd7855cd146846074b3f42bd8bfe4b6a9857b3022a16366df2dddcf77`;
  base prompt
  `b30d5e28a67a029b672037e3765f65ea6e720b93287baa0cd0ec5a5118b622c1`;
  builder prompt
  `110895e0c84bafc8a64f6eba00f77f6203f490c017ce3ad995068369b4516d0a`;
  action prompt
  `62061e8f77ffa9af1b06357f019c5fd8f763871578042a6539d14e2bdbb9e94d`;
  exact Python package tree
  `857a63347a05d93042fdedbee9707406c1ff4769b963e074e1abe8ad802d1ee7`;
  Python 3.11.15, httpx 0.28.1, and SymPy 1.14.0; run contract
  `e4d5d692601508973a5ed3350926b796207a3bf50a0b69d8d3b28ec4e82fa09f`.
- Pre-execution gates: focused and repository-wide tests, compilation,
  whitespace check, prohibited-glyph scan, source regeneration and collision
  audit, valid `queelius` Hugging Face authentication, valid OpenAI model
  access, absent final and partial output paths, and exact recomputation of all
  prepared bindings. No hosted call is authorized until this entry and every
  gate pass.
- Cost and stopping boundary: only the 36-call integration smoke is authorized
  by this entry. Log provider-reported token usage; Featherless pricing is not
  exposed in the route metadata, so do not invent a dollar cost. Stop after
  audit and resolution. A passing smoke authorizes only a separately
  pre-registered 12-case pilot; it does not authorize the pilot, full census,
  selector, training, or model replication by itself.
- Execution and integrity: the registered 36 logical calls and 32 transitions
  completed under contract
  `e4d5d692601508973a5ed3350926b796207a3bf50a0b69d8d3b28ec4e82fa09f`.
  The canonical artifact has exactly 42 records and SHA-256
  `5cf6592d3cc96888c47fda87e346fc2ba81195326489d535454ac70ea6555d03`.
  The diagnostic has one header and 44 attempts, 45 records total, with
  SHA-256
  `74943acd5c3370363d38add145470f5e461c3172e38af51699b34d7e068d5129`.
  Prepublication and sealed-artifact replay passed call-ID, Williams-order,
  prompt-isolation, common-seed, request-body, retry-transcript,
  provider-projection, state-hash, usage, reward, summary, exact-field,
  forbidden-field, credential, no-clobber, and partial-file checks. The final
  full-suite pre-execution gate passed 988 tests with one CUDA-only skip and
  one pre-existing PyTorch warning.
- Interface result: both state builders were usable, complete, and
  exact-text-distinct. Only 1/2 base drafts and 27/32 action outputs were
  usable, with per-target totals of 7/8 root, 7/8 node 1, 7/8 node 2, and 6/8
  node 3. Call failures were one base and five action calls. All four targets
  happened to be usable on all four evaluation calls, but no reward or
  complementarity reading is permitted after the failed integration gate.
  Worker usage was 27,619 prompt and 11,542 completion tokens over 42 HTTP
  generations; builder usage was 538 prompt and 306 completion tokens over
  two generations.
- Failure diagnosis and resolution: E44I is RESOLVED NEGATIVE at the
  integration gate. Every unusable action came from the first, longer `f4`
  case. Five action calls exhausted the 384-token output cap while emitting
  case analysis and ended as truncated JSON after their single retry. The
  failed base call first truncated, then returned the correct answer in an
  object missing the required `reasoning` field. The `f4` state builder was
  shape-valid but necessarily segmented an unusable base placeholder. This is
  an output-capacity failure, not algebra rejection, intermediate
  verification, semantic-node rejection, or symbolic reward behavior. The
  pilot remains unauthorized. A new integration smoke may change only the
  worker output-token cap.

### E44I-R1 - PRE-REGISTERED 07-18: 768-token worker integration repair
- Single manipulation: repeat E44I on the identical two cases with every
  state, prompt, schema, model, provider, revision metadata, seed, retry
  budget, order, verifier, analysis, and audit rule frozen, changing only the
  worker maximum output from 384 to 768 tokens. The builder remains capped at
  384 tokens. This tests the diagnosed truncation mechanism; it is not a
  tolerant parser, prompt revision, state change, or efficacy experiment.
- Size and gates: exactly 36 logical calls, 32 transitions, and 42 canonical
  records. Diagnostic size remains retry-dependent. Reuse E44I's integration
  gate exactly: 2/2 usable bases and builders, 2/2 complete and
  exact-text-distinct node states, 32/32 usable actions, and every registered
  replay, isolation, secret, field-set, publication, and no-partial check. No
  accuracy, reward, or complementarity threshold is permitted. All cases and
  unusable outputs remain included.
- Prepared bindings: dataset
  `4d3ae6dc8a2cbcbbb0a8d4292a9eb0553064edf60b58fe78d6dbfcd4577c2b5e`;
  label-free projection
  `67ae2df4d822030c55c4d58e810ec6801ea8054cabc99cdf9cfad93e6825f8b4`;
  hidden evaluation
  `e5cead8dd7855cd146846074b3f42bd8bfe4b6a9857b3022a16366df2dddcf77`;
  prompt bindings
  `b30d5e28a67a029b672037e3765f65ea6e720b93287baa0cd0ec5a5118b622c1`,
  `110895e0c84bafc8a64f6eba00f77f6203f490c017ce3ad995068369b4516d0a`,
  and `62061e8f77ffa9af1b06357f019c5fd8f763871578042a6539d14e2bdbb9e94d`;
  exact Python package tree
  `857a63347a05d93042fdedbee9707406c1ff4769b963e074e1abe8ad802d1ee7`;
  run contract
  `b98708236a5bb3403f0c0944b47af6926c841c335cce3c347ff86a9887d4a108`.
- Planned artifacts:
  `experiments/runs/2026-07-18-E44I-R1-action-census-smoke2.jsonl` and
  `experiments/runs/2026-07-18-E44I-R1-action-census-smoke2.diagnostics.jsonl`.
  Require the same pre-execution gates and absent final and partial paths.
- Stopping boundary: authorize only this 36-call repair and stop for audit.
  Passing authorizes only a separately registered 12-case pilot with the
  768-token worker cap. Failure leaves the pilot unauthorized and requires a
  new one-factor interface decision.
- Execution and integrity: the registered 36 calls and 32 transitions
  completed with 42 canonical records under contract
  `b98708236a5bb3403f0c0944b47af6926c841c335cce3c347ff86a9887d4a108`.
  The canonical SHA-256 is
  `1b50ddf41464ca7ad034072e28fa5550efa9408383de4f49f34e55e59b137951`;
  the 38-record diagnostic SHA-256 is
  `f61d56037beb7de7430db80cee11fac251c31cf341b2e1a0c7d7a7aee0cf7d24`.
  Every registered artifact replay and publication check passed, and no
  partial remained.
- Result and resolution: the single manipulation repaired the worker boundary
  exactly: 2/2 base drafts and 32/32 action outputs were usable, all on their
  first attempts. Only 1/2 builders, complete-node states, and distinct-node
  states were usable, so E44I-R1 is RESOLVED NEGATIVE at the integration gate
  and the pilot remains unauthorized. The failed `f4` builder emitted 384
  completion tokens on both attempts and truncated in the third node just
  before completing the JSON; the other builder completed in 208 tokens.
  Worker usage was 25,983 prompt and 9,360 completion tokens over 34
  generations; builder usage was 1,872 prompt and 976 completion tokens over
  three generations. This isolates a builder output-capacity failure. A new
  integration run may change only the builder output-token cap.

### E44I-R2 - PRE-REGISTERED 07-18: 768-token builder integration repair
- Single manipulation: repeat E44I-R1 with the worker cap fixed at 768 and
  change only the state-builder maximum output from 384 to 768 tokens. Freeze
  every case, prompt, schema, model, provider, revision metadata, temperature,
  seed, retry budget, Williams order, verifier, analysis, audit, and artifact
  rule. This tests the diagnosed builder truncation and changes no state
  semantics or acceptance rule.
- Size and gates: exactly 36 logical calls, 32 transitions, and 42 canonical
  records, with retry-dependent diagnostic size. Reuse the original E44I
  integration gate without modification: 2/2 usable bases and builders, 2/2
  complete and exact-text-distinct node states, 32/32 usable actions, and all
  registered replay, isolation, field, secret, publication, and partial-file
  checks. No reward threshold or filtered case is permitted.
- Frozen bindings: dataset
  `4d3ae6dc8a2cbcbbb0a8d4292a9eb0553064edf60b58fe78d6dbfcd4577c2b5e`;
  label-free projection
  `67ae2df4d822030c55c4d58e810ec6801ea8054cabc99cdf9cfad93e6825f8b4`;
  hidden evaluation
  `e5cead8dd7855cd146846074b3f42bd8bfe4b6a9857b3022a16366df2dddcf77`;
  unchanged base, builder, and action prompt digests
  `b30d5e28a67a029b672037e3765f65ea6e720b93287baa0cd0ec5a5118b622c1`,
  `110895e0c84bafc8a64f6eba00f77f6203f490c017ce3ad995068369b4516d0a`,
  and `62061e8f77ffa9af1b06357f019c5fd8f763871578042a6539d14e2bdbb9e94d`;
  exact Python package tree
  `857a63347a05d93042fdedbee9707406c1ff4769b963e074e1abe8ad802d1ee7`;
  run contract
  `b87cf3775483a71598346602baf836690c607c96987dacc89c3d39ed8e23bd20`.
- Planned artifacts:
  `experiments/runs/2026-07-18-E44I-R2-action-census-smoke2.jsonl` and
  `experiments/runs/2026-07-18-E44I-R2-action-census-smoke2.diagnostics.jsonl`.
  Require the same static gates, exact contract recomputation, and absent final
  and partial paths.
- Stopping boundary: authorize only this 36-call builder-cap repair and stop
  for audit. Passing authorizes only a separately pre-registered 12-case pilot
  with both caps at 768. Failure requires a new one-factor interface decision.
- Execution and integrity: all 36 registered logical calls and 32 transitions
  completed under contract
  `b87cf3775483a71598346602baf836690c607c96987dacc89c3d39ed8e23bd20`.
  The canonical artifact contains exactly 42 records and has SHA-256
  `2b7cd85e4be3d36c3a7eb683d7e9c2067ee97bdf7019fd95cb0483a75bc1d074`.
  The diagnostic contains one header and 36 attempts, 37 records total, and
  has SHA-256
  `fd4e2523e2a46e7a6d6c77a91a1663a3c91f24d435701b8220a0de21c23f0ac9`.
  Every registered prepublication and sealed-artifact replay, isolation,
  exact-field, hidden-field, credential, publication, and partial-file check
  passed. The observed response model strings matched the requested Qwen
  route and exact GPT model. Neither provider attested an immutable served
  weight revision, as registered.
- Result and resolution: E44I-R2 PASSED the integration gate. Both base drafts,
  both builders, both complete and exact-text-distinct node states, and all
  32 revision outputs were usable on their first attempts. There were no call
  failures, parse retries, or unusable outputs. Builder usage was 896 prompt
  and 583 completion tokens over two generations. Worker usage was 31,823
  prompt and 9,302 completion tokens over 34 generations. The symbolic
  verifier happened to accept every base and action answer on these two
  integration cases, but this stage precluded reward and headroom inference.
  The result authorizes only the separately registered pilot below.

### E44P - PRE-REGISTERED 07-18: complete-answer action-census pilot
- Purpose: estimate whether complete-answer target-conditioned revisions show
  enough base headroom, stable recovery, stable regression, and repeatable
  action heterogeneity to justify a larger confirmatory census. This is a
  signal-screening pilot, not an efficacy result, selector evaluation,
  deployable controller, behavior-cloning result, RL result, or MCTS result.
  No positive performance claim may be made from 12 cases.
- Single progression: retain the E44I-R2 implementation, state, prompts,
  schemas, acceptance rules, models, providers, both 768-token caps,
  temperatures, seeds, retries, action order, reward isolation, verifier,
  artifact protocol, and analysis unchanged. Change only from integration to
  pilot sampling: use the 12 disjoint source rows at indices 2 through 13.
  All cases remain in every analysis; no failure may be filtered or replaced.
- Source: rows 2 through 13 of
  `experiments/problems/2026-07-18-E44-action-census-calibration14.jsonl`.
  The exact ordered IDs are `cw1-88b5649691fe`, `cw1-ad1ed518063e`,
  `cw1-1311efa77cf0`, `cw1-0487eaf517ae`, `cw1-ffab2fab2a9f`,
  `cw1-75085a42ae0f`, `cw1-b2073d244244`, `cw1-9d6ef4998023`,
  `cw1-9e9f9ff20fab`, `cw1-187505f66244`, `cw1-d391713f6617`, and
  `cw1-6859921b4cde`. This gives exactly three cases from each generated family
  `f4`, `f6`, `f7`, and `f8`, although family is hidden from both models.
- Exact size: 216 logical calls total (12 base, 12 builder, and 192 action
  calls), 192 transitions, and 242 canonical records. The diagnostic has one
  header and at least 216 attempts; retries may increase its size. Require all
  216 unique call IDs and enforce every registered retry budget. `SUBMIT` is a
  deterministic zero-call protected control.
- Frozen measurements: report base, four fixed-revision, best fixed-revision,
  protected fixed-policy, revision-only discovery-to-evaluation repeatability,
  and protected discovery-to-evaluation repeatability rewards. Report stable
  heterogeneous revision vectors, stable targeted-node-over-root states,
  stable recovery states, and stable regression states. A stable result means
  the relevant Boolean relation holds in both discovery repetitions and both
  evaluation repetitions. Repeatability uses verifier-probed discovery
  rewards to select within each state, so it is an outcome oracle and never a
  deployable policy. Report paired replicate cells, per-problem sign counts,
  and problem-cluster bootstrap intervals, but do not treat action calls as
  independent observations.
- Interface gates: require 12/12 usable builders, complete-node states, and
  exact-text-distinct-node states; at least 11/12 usable bases; at least 46/48
  usable outputs for every target and 184/192 overall; and at least 23/24
  usable evaluation outputs for every target. Report first-attempt acceptance
  and retries without gating on them. Require the complete Williams grid,
  common seeds, target-only action-prompt difference, verifier timing,
  request, response, retry, usage, state, reward, summary, field-set,
  forbidden-field, credential, ASCII, no-clobber, and no-partial audits.
- Signal gates: require base success on 3 through 9 of 12 cases; best fixed
  revision evaluation reward and best protected fixed-policy evaluation reward
  each between 0.20 and 0.85 inclusive; at least one stable recovery state;
  at least one stable regression state; at least one stable heterogeneous
  revision state; and at least one stable targeted-node-over-root state. These
  gates screen for measurable headroom and bidirectional action effects. They
  do not establish semantic-state value, controller performance, or efficacy,
  and the realized repeatability oracle is not itself a gate.
- Prepared bindings: dataset
  `4d3ae6dc8a2cbcbbb0a8d4292a9eb0553064edf60b58fe78d6dbfcd4577c2b5e`;
  label-free problem projection
  `1b48b182df6c0e7dd30ce80a75a9f0757ab896fff7fdf2eb66dbcec8e95ceb7d`;
  hidden evaluation
  `2376afa61758eab49d2237dce317e14affec76f3a56403dda25e2c4a6d03964e`;
  base, builder, and action prompt digests
  `b30d5e28a67a029b672037e3765f65ea6e720b93287baa0cd0ec5a5118b622c1`,
  `110895e0c84bafc8a64f6eba00f77f6203f490c017ce3ad995068369b4516d0a`,
  and `62061e8f77ffa9af1b06357f019c5fd8f763871578042a6539d14e2bdbb9e94d`;
  exact Python package tree
  `857a63347a05d93042fdedbee9707406c1ff4769b963e074e1abe8ad802d1ee7`;
  Python 3.11.15, httpx 0.28.1, and SymPy 1.14.0; run contract
  `1a392924947d689d8be8ff3f6bd70b239234251d4c32169d1dde872c1c243e88`.
- Planned artifacts:
  `experiments/runs/2026-07-18-E44P-action-census-pilot12.jsonl` and
  `experiments/runs/2026-07-18-E44P-action-census-pilot12.diagnostics.jsonl`.
  Require absent final and partial paths, exact contract recomputation, and
  the unchanged static gates before execution.
- Stopping rule: authorize only this 12-case pilot and then stop for sealed
  audit and registered-gate resolution. A failed gate stops this design branch
  without a larger run. A passing pilot authorizes only preparation and
  separate pre-registration of a disjoint confirmatory census; it does not
  authorize that census, training, selector claims, model replication, RL, or
  MCTS by itself. Log provider token usage. Do not infer an unreported dollar
  cost from token counts.
- Execution and integrity: all 216 registered logical calls and 192
  transitions completed under contract
  `1a392924947d689d8be8ff3f6bd70b239234251d4c32169d1dde872c1c243e88`.
  The canonical artifact has exactly 242 records and SHA-256
  `716597aa00a90df9eda5934e298ccd303af088779d2a0f348f89686054832447`.
  The diagnostic has one header and 217 attempts, 218 records total, and
  SHA-256
  `bd3af1786324e8012144c379a6d252f6d4345e621c44cc1317c9acf46213893a`.
  All registered prepublication and sealed-artifact audits passed. The 216
  unique call IDs, 12 states, full Williams grids, common target seeds,
  target-only prompt difference, verifier timing, retry transcript, requests,
  provider projections, usage, state hashes, symbolic rewards, summary,
  exact fields, hidden-field and credential scans, ASCII encoding, no-clobber
  publication, and removal of partial files all replayed exactly. Served
  revisions remain provider-unattested as registered.
- Interface result: all 12 bases, builders, complete-node states, and
  exact-text-distinct-node states were usable. Total usable revision outputs
  were 47/48 root, 47/48 node 1, 44/48 node 2, and 48/48 node 3, or 186/192
  overall. Evaluation usability was 23/24 root, 24/24 node 1, 22/24 node 2,
  and 24/24 node 3. Six unusable candidates were confined to one problem:
  one root, one node 1, and all four node 2 calls. The node 2 total and
  evaluation gates therefore failed. All 12 base and builder calls were
  first-attempt accepted. Revision calls had 191/192 first-attempt accepted
  responses and one successful parse retry.
- Signal result: base reward was 4/12. Every fixed revision target had the
  identical discovery and evaluation reward of 5/12; the best protected fixed
  policy was therefore also 5/12. Across the 24 paired evaluation replicates
  for every target versus root, ten were both correct, fourteen were both
  wrong, and zero favored either side. There was one stable recovery state and
  no stable regression state, no heterogeneous revision state, and no root or
  targeted-node advantage state. Revision and protected repeatability both
  selected root-equivalent outcomes and scored 5/12, giving zero delta and a
  problem-bootstrap 90% interval `[0.0, 0.0]` against their best fixed
  baselines. These are descriptive outcome-oracle readings, not selector or
  policy estimates.
- Usage: the 193 revision attempts used 164,754 prompt and 40,505 completion
  tokens. The 12 base attempts used 1,605 prompt and 2,670 completion tokens.
  The 12 state-builder attempts used 4,867 prompt and 2,565 completion tokens.
  The route exposed no defensible Featherless dollar price, so none is
  inferred.
- Resolution: E44P is RESOLVED NEGATIVE and the larger same-design census is
  not authorized. It passed the registered base headroom, fixed-policy range,
  protected-policy range, and stable-recovery gates, but failed node 2
  usability, stable regression, stable revision heterogeneity, and stable
  targeted-node-over-root gates. The reward null is not an artifact of the
  six unusable outputs: the other eleven fully usable states also had
  identical root and node reward vectors. Offline text diagnostics found
  target-dependent reasoning in 32/48 state-repetition groups but differing
  final answers in only 5/48; four of those five groups came from the one
  unusable-output problem, and every differing answer was verifier-wrong.
  The builder also collapsed to the same document layout in all 12 states:
  node 1 restated the exact problem, node 2 held reasoning or calculation,
  and node 3 held an answer or conclusion. These are sections, not distinct
  subproblems or revision objectives. Moreover, on the sole stable-recovery
  problem `cw1-75085a42ae0f`, the base draft answered 9 but the builder's
  node 3 supplied the corrected enumeration and answer 5 while explicitly
  calling the draft wrong. This violated the registered non-repair builder
  role. All four targets then answered 5 because every action saw every node.
  The only recovery is therefore compatible with builder correction leakage
  plus a generic revision call, not useful target selection. Thus the selected
  ID weakly perturbed wording while the full displayed state dominated the
  final answer. Any continuation must be a separately registered
  state-construction or action-view experiment, not a scaled repetition of
  E44P.

### E45I - PRE-REGISTERED 07-18: actionable-check state integration
- Purpose and status: test the provider and artifact boundary for one
  state-construction revision motivated by E44P. This is an adaptive
  development-set mechanism smoke, not an accuracy, efficacy, causal,
  selector, behavior-cloning, RL, or MCTS result. No reward or improvement
  threshold is permitted. Passing authorizes only a separately registered
  12-case adaptive pilot.
- Single semantic manipulation: replace E44's draft-section builder with
  `actionable_checks_v1`. GPT-4.1-mini must return exactly three strict
  `operation`/`object` pairs. The prompt requests problem-specific,
  pairwise-nonoverlapping revision tasks, assigns no fixed role to node
  position, and forbids solving, correctness verdicts, computed results,
  corrected expressions, proposed replacements, and answer-bearing hints.
  The state exposes `operation` as title and `object` as content. The Qwen
  action prompt, full-state view, target-ID-only action difference, answer
  schema, temperatures, token caps, seeds, retries, Williams order,
  repetition split, verifier, analysis, and artifact protocol stay frozen.
- Sealed base replay: make no new base call. Replay the exact two usable base
  Candidates from E44I-R2 run
  `2b7cd85e4be3d36c3a7eb683d7e9c2067ee97bdf7019fd95cb0483a75bc1d074`
  and diagnostic
  `fd4e2523e2a46e7a6d6c77a91a1663a3c91f24d435701b8220a0de21c23f0ac9`,
  whose source contract is
  `b87cf3775483a71598346602baf836690c607c96987dacc89c3d39ed8e23bd20`.
  The ordered replay projection, including every source call ID, source record
  hash, request hash, seed, canonical Candidate and Candidate hash, is
  `31c220e360989fffa14aa2683d0c96757f6ecbd833ae0b4f39d120e7b7fe84db`.
  Replay records are provenance, not logical calls, and must have zero
  `base_worker` attempts and usage.
- Source and size: use source rows zero and one, problem IDs
  `cw1-3ed474a04b63` and `cw1-bc3cca7bcfbf`. Dataset SHA-256 is
  `4d3ae6dc8a2cbcbbb0a8d4292a9eb0553064edf60b58fe78d6dbfcd4577c2b5e`;
  label-free projection is
  `67ae2df4d822030c55c4d58e810ec6801ea8054cabc99cdf9cfad93e6825f8b4`;
  hidden evaluation binding is
  `e5cead8dd7855cd146846074b3f42bd8bfe4b6a9857b3022a16366df2dddcf77`.
  Execute exactly 34 new logical calls (two builders and 32 actions), 32
  transitions, and 42 canonical records. The diagnostic has one header and at
  least 34 attempts. It must bind exactly 34 new unique call IDs; the two
  replayed source call IDs are provenance only.
- Models: worker request model
  `Qwen/Qwen2.5-3B-Instruct:featherless-ai`, registered Hub revision metadata
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`, temperature 0.2, maximum 768
  tokens, seed 440000, one parse retry, and six backoff retries. Builder
  `gpt-4.1-mini-2025-04-14`, temperature 0.0, maximum 768 tokens, seed 441000,
  one parse retry, and six backoff retries. Provider revisions remain
  unattested served-weight metadata and must be reported as such.
- Integration gates: require exact sealed-base replay, zero base HTTP attempts,
  2/2 usable builders, two complete and normalized-pairwise-distinct states,
  six nonempty `operation`/`object` checks, and 32/32 usable action outputs.
  Inspect a state-only projection for section-like nodes, overlap, explicit
  resolved work, verdicts, corrections, or answer leakage without consulting
  action rewards. This is an interface-validity audit, not an LLM correctness
  judge or reward source. Any clear violation fails the stage, remains in the
  artifact, and forbids replacement or filtering. Require every registered
  call, request, response, retry, usage, state, Williams, common-seed,
  target-isolation, verifier-timing, reward, summary, exact-field,
  hidden-field, credential, ASCII, no-clobber, and no-partial replay check.
  First-attempt acceptance is reported but is not a gate.
- Prepared bindings: unchanged base prompt
  `b30d5e28a67a029b672037e3765f65ea6e720b93287baa0cd0ec5a5118b622c1`;
  new builder prompt and strict schema
  `102d69a8667d2990389e0b5de7065650bd7583cf1ad6569503ad13b34ed37ac4`;
  unchanged action prompt
  `62061e8f77ffa9af1b06357f019c5fd8f763871578042a6539d14e2bdbb9e94d`;
  exact Python package tree
  `cb0af99c5a7ace6d3d7019063daade3f828299e4c971d3b813a88d384c6803b7`;
  Python 3.11.15, httpx 0.28.1, and SymPy 1.14.0; run contract
  `224a41a9a8bfca93516ce4c9501c59bb27d1be0246972e9107972506ab3d6c0b`.
  The contract uses header v4, diagnostic v2, run-summary v3, and analysis v3.
- Static gates: the exact package passed 991 tests with one CUDA-only skip and
  one pre-existing PyTorch warning. Focused action-census tests passed 10/10,
  including forged replay Candidate, zero-base-call, retry, reward, seed,
  extra-field, and failure-driven recovery regressions. Compilation, Black,
  whitespace, and prohibited-glyph checks must remain clean, output and partial
  paths must be absent, and the prepared contract must recompute exactly.
- Planned artifacts:
  `experiments/runs/2026-07-18-E45I-actionable-state-smoke2.jsonl` and
  `experiments/runs/2026-07-18-E45I-actionable-state-smoke2.diagnostics.jsonl`.
  Only this smoke is authorized. Stop after sealed audit and state inspection;
  do not inspect reward as evidence or proceed without a separate pilot entry.
- Execution and integrity: all 34 registered new logical calls and 32
  transitions completed under contract
  `224a41a9a8bfca93516ce4c9501c59bb27d1be0246972e9107972506ab3d6c0b`.
  The canonical artifact has 42 records and SHA-256
  `d0a16e0662f91442f7237233b82270e49b2a5dca504bc567d92b0068215abcec`.
  The diagnostic has one header and 34 attempts, 35 records total, and
  SHA-256
  `83441d309b7cb277e4d32cc81ffa12b186be3c45490a3c7ce05b9c01869241e2`.
  The sealed replay, zero-base-attempt, call-ID, request, retry, usage, state,
  Williams-order, common-seed, target-isolation, verifier-timing, reward,
  summary, exact-field, hidden-field, credential, ASCII, no-clobber, and
  no-partial audits passed. All two builder and 32 action calls were accepted
  on their first attempts. Served revisions remain provider-unattested as
  registered.
- Interface result: both builders were structurally usable, both states were
  complete and normalized-pairwise-distinct, all six operation/object pairs
  were nonempty, and all 32 action outputs were usable. The two builder calls
  used 1,182 prompt and 261 generated tokens. The 32 worker calls used 25,728
  prompt and 8,184 generated tokens. No defensible route price is inferred
  from token counts.
- State-only inspection: the revision-task framing is materially better than
  E44's fixed problem/reasoning/answer sections, but the semantic gate failed.
  Three of six builder objects copied explicit solved values from the draft:
  the first state's node 3 included `{x = -9, x = 14}` and asked whether the
  smallest was correct, while the second state's nodes 2 and 3 included
  `x = -18` and `x = -6`. The second state's node 1 asked to check the whole
  case split, so it also subsumed the two per-case checks and violated the
  requested non-overlap. These are prompt-role violations even though they
  copy the frozen draft rather than introduce a corrected answer. This audit
  used only problem, draft, and state text, not action reward.
- Resolution: E45I is RESOLVED NEGATIVE. Provider integration and artifact
  integrity passed, but explicit resolved work and hierarchical overlap failed
  the registered state-validity gate. E45P is not authorized, no reward or
  accuracy reading from this smoke supports progression, and none of its
  records are behavior-cloning eligible. The next state-construction attempt
  must eliminate free-form answer-bearing builder text structurally, rather
  than relying on another prompt admonition. A separately registered design
  may let the builder choose controlled audit operations and disjoint IDs of
  spans from the frozen draft, with node text reconstructed deterministically.

### E46I - PRE-REGISTERED 07-18: anchored reasoning-state integration
- Purpose and status: test one structurally non-authoring replacement for
  E45's free-form state builder. This is an adaptive development-set
  representation and transport smoke, not an accuracy, efficacy, causal,
  controller, observability, behavior-cloning, RL, or MCTS result. No reward
  or improvement threshold is permitted. Passing authorizes only a separately
  registered 12-case adaptive pilot.
- Single semantic manipulation: replace `actionable_checks_v1` with
  `anchored_checks_v1`. Trusted code deterministically splits only the frozen
  Candidate reasoning into exact, offset-bearing sentence spans. The builder
  returns exactly three controlled operation/span groups and no prose. Each
  group contains one span or two consecutive spans. The closed operations are
  `recheck_model`, `recompute_local`, `test_admissibility`, `check_coverage`,
  and `check_finalization`; operations may repeat. Trusted code rejects
  unknown, repeated, nonconsecutive, overlapping, or extra-valued selections,
  sorts accepted groups by source offset, assigns node IDs, generates titles
  from fixed templates, and copies node content from the exact frozen
  reasoning slice. Problem and answer fields are not selectable spans.
- Frozen factors and interpretation: keep the E45 worker, complete-state
  action view, target-ID-only action difference, whole-problem revision
  instruction, answer schema, verifier, temperatures, token caps, seeds,
  retries, Williams order, four repetitions, analysis, and artifact protocol.
  This isolates state construction. A node is a teacher-proposed audit macro,
  so this does not yet isolate factored operation and target effects. The
  builder can signal a suspicion through its discrete selection. Every action
  still sees the full state, so a null cannot distinguish an unhelpful state
  from an insufficiently local lens.
- Sealed base replay: make no base call. Replay the exact two Candidates from
  E44I-R2 run
  `2b7cd85e4be3d36c3a7eb683d7e9c2067ee97bdf7019fd95cb0483a75bc1d074`
  and diagnostic
  `fd4e2523e2a46e7a6d6c77a91a1663a3c91f24d435701b8220a0de21c23f0ac9`,
  whose source contract is
  `b87cf3775483a71598346602baf836690c607c96987dacc89c3d39ed8e23bd20`.
  The ordered replay projection is
  `31c220e360989fffa14aa2683d0c96757f6ecbd833ae0b4f39d120e7b7fe84db`.
  Replay records are provenance, not logical calls, and require zero
  `base_worker` attempts and usage.
- Span binding: the two frozen Candidates produce exactly 14 and 7 eligible
  reasoning spans. The ordered binding includes every Candidate hash,
  reasoning hash, span ID, character offset, text hash, and exact text under
  `action-census-reasoning-catalog-binding-v1`; its SHA-256 is
  `6a6804028e1623de622b34596d25b6cc1083ed47d6ac12a083aa57411516ed19`.
  Contract preparation must fail before calls if this projection changes or a
  Candidate has fewer than three eligible spans.
- Source and size: use source rows zero and one, problem IDs
  `cw1-3ed474a04b63` and `cw1-bc3cca7bcfbf`. Dataset SHA-256 is
  `4d3ae6dc8a2cbcbbb0a8d4292a9eb0553064edf60b58fe78d6dbfcd4577c2b5e`;
  label-free projection is
  `67ae2df4d822030c55c4d58e810ec6801ea8054cabc99cdf9cfad93e6825f8b4`;
  hidden evaluation binding is
  `e5cead8dd7855cd146846074b3f42bd8bfe4b6a9857b3022a16366df2dddcf77`.
  Execute exactly 34 new logical calls, two builders and 32 actions, 32
  transitions, and 42 canonical records. The diagnostic has one header and at
  least 34 attempts. The two replayed source call IDs remain provenance only.
- Models: worker request model
  `Qwen/Qwen2.5-3B-Instruct:featherless-ai`, registered Hub revision metadata
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`, temperature 0.2, maximum 768
  tokens, seed 440000, one parse retry, and six backoff retries. Builder
  `gpt-4.1-mini-2025-04-14`, temperature 0.0, maximum 768 tokens, seed 441000,
  one parse retry, and six backoff retries. Provider revisions remain
  unattested served-weight metadata.
- Reward-blind state and interface gates: before consulting action reward,
  require exact catalog replay, zero base HTTP attempts, 2/2 usable builders,
  and six accepted groups. Every group must use an allowed operation, contain
  one or two consecutive valid span IDs, and be disjoint from every other
  group in its state. Reconstructed titles must match fixed templates,
  contents must be exact source slices, and node order must replay from source
  offsets. The builder response must contribute zero free-form node text and
  must expose no answer field. Inspect operation/span applicability and task
  complementarity from problem, draft, and state only; any clear nonsensical
  operation or semantic subsumption fails the stage and remains in the
  artifact. Require 32/32 usable action outputs and report first-attempt
  acceptance without gating on it.
- Artifact gates: require every registered call, exact request and response,
  retry transcript, usage, state, catalog, Williams order, common seed,
  target-only prompt difference, verifier timing, reward replay, summary,
  field set, hidden-field scan, credential scan, ASCII encoding, no-clobber
  publication, and no-partial cleanup. The symbolic verifier remains the only
  reward source. Rewards are logged for integrity but are not evidence for
  smoke progression.
- Prepared bindings: unchanged base and action prompt digests
  `b30d5e28a67a029b672037e3765f65ea6e720b93287baa0cd0ec5a5118b622c1`
  and `62061e8f77ffa9af1b06357f019c5fd8f763871578042a6539d14e2bdbb9e94d`;
  anchored builder prompt, schema template, and span projection digest
  `a6472b16257950dfe552f242fa3f4627b97283d0b5902c79247b3f0770889d74`;
  exact Python package tree
  `3cc0123fb52ef456da708472605662ef0c6aac2073a5a95c6b641be3cac38480`;
  Python 3.11.15, httpx 0.28.1, and SymPy 1.14.0; run contract
  `9c8fc64086d62a92468571711893fc37cd0544be6fff371927b730fb5d189389`.
- Static gates: the exact package passed 993 tests with one CUDA-only skip and
  one pre-existing PyTorch warning. The workspace-v1 slice passed 102/102 and
  focused action-census tests passed 12/12. Compilation, Black for Python
  3.11, whitespace, and prohibited-glyph checks are clean. Planned output and
  partial paths must be absent and the prepared contract must recompute
  exactly before execution.
- Planned artifacts:
  `experiments/runs/2026-07-18-E46I-anchored-state-smoke2.jsonl` and
  `experiments/runs/2026-07-18-E46I-anchored-state-smoke2.diagnostics.jsonl`.
  Only this smoke is authorized. Stop after sealed audit and reward-blind
  state inspection. Do not run or pre-register E46P unless every smoke gate
  passes.
- Execution and integrity: all 34 registered calls and 32 transitions
  completed under contract
  `9c8fc64086d62a92468571711893fc37cd0544be6fff371927b730fb5d189389`.
  The canonical artifact has 42 records and SHA-256
  `a4ea04d0c8cc29fd515b0d917f3be30a32764b0dbdf17d1c88184a19ccf78948`.
  The diagnostic has one header and 34 attempts, 35 records total, and
  SHA-256
  `92d4e4d76d8c129c28039d42e12e1f4b05f7a1651e22f5d310668ad9cbe93f18`.
  Exact replay, zero-base-attempt, catalog, call-ID, request, retry, usage,
  state, order, common-seed, target-isolation, verifier-timing, reward,
  summary, field, hidden-field, credential, ASCII, no-clobber, and no-partial
  audits passed. All two builder and 32 action HTTP responses were accepted
  on their first attempts. Served revisions remain provider-unattested.
- Reward-blind interface result: only one of two accepted builder responses
  passed the registered structural decoder. The first chose `S001` for
  `recheck_model`, `S004,S005` for `recompute_local`, and `S007,S008` for
  `test_admissibility`. Trusted reconstruction produced three exact,
  source-disjoint, complementary checks covering setup, the first interval,
  and the middle interval. The second chose `S001`, `S002,S003`, and
  `S004,S006`. Its last group paired the second-case setup with its later
  admissibility check while skipping the intervening solve sentence. This is
  semantically coherent but violated the preregistered consecutive-ID rule,
  so the state correctly remained a failed fallback under this contract.
- Worker interface result: 31/32 action Candidates were usable. The only
  unusable output was node 3, repetition 3 on the valid first state: the
  worker returned reasoning that the selected middle interval had no solution
  and an empty top-level answer, despite the frozen complete-root instruction.
  This is a genuine locality hazard, not a transport or generation failure,
  and remains in the artifact. The two builders used 2,396 prompt and 102
  generated tokens; the 32 workers used 26,384 prompt and 7,264 generated
  tokens. No defensible route price is inferred.
- Resolution: E46I is RESOLVED NEGATIVE and E46P is not authorized. It failed
  the 2/2 builder and 32/32 worker usability gates. No reward or accuracy
  reading from the smoke is used for this decision, and none of its rows are
  behavior-cloning eligible. The builder failure exposes an unnecessary
  representation restriction rather than free-form leakage: two selected IDs
  can instead denote ordered endpoints of one inclusive frozen-source range.
  A separately registered integration repair may change only that range
  interpretation and its matching prompt wording. Its worker-usability gate
  may use the pilot-consistent tolerance of at least 31/32 total, at least 7/8
  per target, and at least 3/4 evaluation outputs per target, explicitly as an
  adaptive response to this reward-blind interface observation.

### E46I-R1 - PRE-REGISTERED 07-18: inclusive-range integration repair
- Purpose and status: repair only the unnecessary consecutive-ID rejection
  exposed by E46I. This remains an adaptive, reward-blind representation and
  transport smoke. It is not a pilot, efficacy result, causal comparison,
  controller result, observability result, or training source. No reward or
  improvement threshold is permitted.
- Single manipulation: replace `anchored_checks_v1` with
  `anchored_ranges_v1`. One selected ID denotes one frozen reasoning span;
  two increasing IDs denote ordered endpoints of the inclusive source range
  from the first through the second. Trusted code expands each range, rejects
  unknown or reversed endpoints and overlap between expanded ranges, sorts
  accepted ranges by first source offset, and reconstructs one exact frozen
  source slice. The builder still emits only a closed operation and one or two
  IDs, with no prose. All operation keys, reasoning segmentation, catalogs,
  state and action schemas, worker view, target mechanism, model settings,
  seeds, order, repetitions, verifier, analysis, and artifacts remain frozen.
- Frozen sources and size: reuse the E46I two-case problem, hidden-label,
  sealed-base, and 14/7-span catalog bindings exactly. Their SHA-256 values are
  `67ae2df4d822030c55c4d58e810ec6801ea8054cabc99cdf9cfad93e6825f8b4`,
  `e5cead8dd7855cd146846074b3f42bd8bfe4b6a9857b3022a16366df2dddcf77`,
  `31c220e360989fffa14aa2683d0c96757f6ecbd833ae0b4f39d120e7b7fe84db`,
  and `6a6804028e1623de622b34596d25b6cc1083ed47d6ac12a083aa57411516ed19`.
  Make zero base calls and exactly 34 new calls, two builders plus 32 actions,
  32 transitions, 42 canonical records, and at least 35 diagnostic records.
- Reward-blind gates: require exact replay and catalog reconstruction, zero
  base attempts, 2/2 usable builders, and six valid controlled ranges. Every
  endpoint must resolve, be increasing when paired, and expand to a source
  interval disjoint from the other two intervals. Node titles, IDs, ordering,
  and contents must reconstruct exactly, and builder output must contribute
  zero free-form node text or answer field. Inspect operation/range
  applicability, locality, and complementarity without action reward; any
  clear nonsensical operation, whole-draft range, or semantic subsumption
  fails and remains logged. Require at least 31/32 usable action outputs, at
  least 7/8 for each target, and at least 3/4 evaluation outputs for each
  target. These adaptive thresholds match the E44 pilot tolerance and treat
  the E46 empty-answer locality event as an interface outcome, not a reason to
  alter the worker prompt.
- Artifact gates: retain every E46I replay, call, request, response, retry,
  usage, state, order, common-seed, target-isolation, verifier-timing, reward,
  summary, exact-field, hidden-field, credential, ASCII, no-clobber, and
  no-partial requirement. Rewards are logged only for symbolic replay
  integrity and are not inspected as progression evidence. First-attempt
  acceptance is reported but is not a gate.
- Prepared bindings: unchanged base and action prompt digests
  `b30d5e28a67a029b672037e3765f65ea6e720b93287baa0cd0ec5a5118b622c1`
  and `62061e8f77ffa9af1b06357f019c5fd8f763871578042a6539d14e2bdbb9e94d`;
  range-builder prompt, schema template, and span projection digest
  `5fec29618738b9a1c66937b4b9ad712e8de841798657eb3485705d5c621ba486`;
  exact Python package tree
  `b6c86cf7b69007eba8b07de8d57a48febb01ca5c5158972ce81e9f5bf05dfe9c`;
  Python 3.11.15, httpx 0.28.1, and SymPy 1.14.0; run contract
  `255b5392de1b765104a1f98c884bdd1ceb9f0c656570d915fa7321fc2fb0dd7d`.
- Static gates: the exact package passed 994 tests with one CUDA-only skip and
  one pre-existing warning. Workspace-v1 passed 103/103 and focused census
  tests passed 13/13. Compilation, Black for Python 3.11, whitespace, and
  prohibited-glyph checks are clean. Planned output and partial paths must be
  absent and the exact contract must recompute before calls.
- Planned artifacts:
  `experiments/runs/2026-07-18-E46I-R1-anchored-range-smoke2.jsonl` and
  `experiments/runs/2026-07-18-E46I-R1-anchored-range-smoke2.diagnostics.jsonl`.
  Only this integration repair is authorized. Stop for sealed audit and
  reward-blind state inspection. Passing authorizes only separate E46P
  pre-registration; failure forbids the pilot and another same-branch prompt
  repair.
- Execution and integrity: all 34 registered calls and 32 transitions
  completed under contract
  `255b5392de1b765104a1f98c884bdd1ceb9f0c656570d915fa7321fc2fb0dd7d`.
  The canonical artifact has 42 records and SHA-256
  `9699f497e3450d8d0421dbfdfa0113039d531a12fe5b84970c8eed5932c9d811`.
  The diagnostic has one header and 34 attempts, 35 records total, and
  SHA-256
  `1c15bd4949ec2742ba7b7037c592bd422e1fdde20c85e6103e6703958c1ecd64`.
  Every registered sealed replay, zero-base, catalog, call, request, retry,
  usage, state, order, common-seed, target-isolation, verifier-timing, reward,
  summary, field, hidden-field, credential, ASCII, no-clobber, and no-partial
  audit passed. All two builder and 32 worker responses were accepted on their
  first attempts, and all 34 logical outputs were usable.
- Reward-blind state result: both builders and all six ranges passed. For the
  first problem, the state selected setup `S001`, first-branch recomputation
  `S004-S006`, and finalization `S013-S014`. For the second, it selected case
  setup `S001`, second-branch recomputation `S004-S005`, and admissibility
  `S006`. Every interval is source-local, all intervals are pairwise disjoint,
  each operation fits its frozen text, and no selected task subsumes another.
  Titles and contents replay exactly from trusted templates and source
  offsets. Solved values visible in child content are exact excerpts already
  present in the frozen root reasoning, not builder-authored corrections; the
  builder emitted no prose and could not select the answer field.
- Usage and resolution: the builders used 2,462 prompt and 98 generated
  tokens; workers used 27,184 prompt and 7,620 generated tokens. No route
  price is inferred. E46I-R1 PASSES every registered reward-blind interface
  and integrity gate. Its reward summary is not used as evidence and its rows
  remain ineligible for behavior cloning. A separately preregistered E46P
  adaptive pilot is authorized only to test whether these valid targets
  produce repeatable symbolic action-value structure.

### E46P - PRE-REGISTERED 07-18: anchored-range action-value pilot
- Purpose and status: run the first outcome-bearing screen of structurally
  valid, teacher-proposed audit ranges. This is an adaptive pilot on twelve
  previously observed development cases, not a confirmatory, causal,
  controller, observability, generalization, or training result. The fixed
  full-state worker lens remains a material limitation.
- Frozen mechanism: use `anchored_ranges_v1` exactly as E46I-R1. Freeze the
  five operation keys, inclusive-range interpretation, sentence projection,
  catalog construction, canonical node order, full-state action prompt,
  target-ID-only request difference, answer schema, worker and builder models,
  settings, retry policy, four repetitions, Williams schedule, common seeds,
  symbolic verifier, analysis, and artifact protocol. Make no base calls.
- Source and replay: use source rows 2 through 13, problem projection
  `1b48b182df6c0e7dd30ce80a75a9f0757ab896fff7fdf2eb66dbcec8e95ceb7d`
  and hidden evaluation binding
  `2376afa61758eab49d2237dce317e14affec76f3a56403dda25e2c4a6d03964e`.
  Replay exact base Candidates from E44P run
  `716597aa00a90df9eda5934e298ccd303af088779d2a0f348f89686054832447`,
  diagnostic
  `bd3af1786324e8012144c379a6d252f6d4345e621c44cc1317c9acf46213893a`,
  and source contract
  `1a392924947d689d8be8ff3f6bd70b239234251d4c32169d1dde872c1c243e88`.
  The ordered base projection is
  `a8eb8ea3f9f4ba0eef9f9fe8073e098a5cc4898380f7c648fd479ad48f90f424`.
- Span binding: the twelve frozen reasonings produce 4, 6, 7, 10, 6, 6, 9,
  15, 8, 5, 14, and 14 source spans. The exact ordered Candidate, reasoning,
  offset, span-text, and hash projection is
  `e06840c7978ff7bb158f919e26918cc6b54c79066d4f02581b6f0e1754f14481`.
  Any replay or catalog change fails before calls.
- Exact size: execute 204 new logical calls, twelve builders plus 192 actions,
  192 transitions, and 242 canonical records. The diagnostic has one header
  and at least 204 attempts, 205 records total. Require 204 unique new call
  IDs. The twelve replayed base call IDs remain provenance only and base
  attempts and usage must be zero.
- Reward-blind state gates: require 12/12 usable builders, 36/36 valid
  controlled ranges, and twelve complete, address-distinct states. Every
  endpoint must resolve and be increasing when paired; expanded ranges must
  be pairwise disjoint. Titles, contents, node IDs, source ordering, and hashes
  must replay exactly, and the builder must contribute no free-form node text
  or answer field. Inspect all 36 operation/range pairs before outcome
  analysis. Any clearly nonsensical operation, whole-draft range, or task that
  semantically subsumes another selected task fails the state gate, remains in
  the artifact, and stops signal interpretation without filtering or case
  replacement. Report operation counts, range widths, normalized source
  positions, and node-position distributions.
- Worker interface gates: require at least 46/48 usable outputs for every
  target, at least 184/192 overall, and at least 23/24 evaluation outputs for
  every target. Report first-attempt acceptance and all failure modes without
  gating on first-attempt rate. Require the complete target-repetition grid,
  target-only prompt isolation, and common target seeds.
- Symbolic signal gates: inspect these only after every state and artifact
  gate passes. Require at least two states with the same unique best revision
  target in both the two-repetition discovery and evaluation splits. Their
  stable winners must cover at least two different target IDs and include at
  least one node target. Require at least one clean targeted recovery: replayed
  base wrong, all sixteen action outputs usable, all four root revisions wrong,
  and one stable winning node correct in all four repetitions. Require the
  discovery-selected revision evaluation reward to exceed the best fixed
  revision target by at least 1/12. Require base and best-fixed evaluation
  performance each between 3/12 and 10/12 inclusive to retain measurable
  headroom. Report stable recovery, regression, root-over-node,
  node-over-root, heterogeneity, strict-winner counts, every paired cell,
  per-problem signs, and problem-cluster bootstrap intervals. Action calls are
  never treated as independent samples.
- Interpretation and stopping: passing all signal gates supports only that a
  frozen full-state worker exhibited repeatable, state-dependent symbolic
  outcome differences among LLM-proposed audit ranges on an adaptive screen.
  It does not establish a deployable selector, state observability,
  improvement over direct prompting, cost effectiveness, or controller
  performance. Pass authorizes only separately preregistered fresh-source
  contemporaneous confirmation and observability ablations. State or
  interface failure stops this representation without outcome claims. A valid
  state with failed signal gates forbids a larger same-design census,
  behavior cloning, Q-learning, or pointer training; it may motivate only a
  separately registered target-local-lens isolation or a clean factored
  operation-by-span action experiment.
- Artifact and model bindings: retain every E46 replay, request, response,
  retry, usage, state, order, seed, verifier-timing, reward, summary,
  exact-field, hidden-field, credential, ASCII, no-clobber, and no-partial
  audit. Worker is
  `Qwen/Qwen2.5-3B-Instruct:featherless-ai` with registered revision metadata
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`; builder is
  `gpt-4.1-mini-2025-04-14`. Settings are the frozen E46I-R1 values and served
  revisions remain unattested.
- Prepared bindings: dataset
  `4d3ae6dc8a2cbcbbb0a8d4292a9eb0553064edf60b58fe78d6dbfcd4577c2b5e`;
  base, builder, and action prompt digests
  `b30d5e28a67a029b672037e3765f65ea6e720b93287baa0cd0ec5a5118b622c1`,
  `5fec29618738b9a1c66937b4b9ad712e8de841798657eb3485705d5c621ba486`,
  and `62061e8f77ffa9af1b06357f019c5fd8f763871578042a6539d14e2bdbb9e94d`;
  package tree
  `b6c86cf7b69007eba8b07de8d57a48febb01ca5c5158972ce81e9f5bf05dfe9c`;
  Python 3.11.15, httpx 0.28.1, SymPy 1.14.0; run contract
  `b5ab0b01079883ba7f93d79f30480195aeabf2333abcdf05f50f64717dba0ae2`.
  The exact package passed 994 tests with one CUDA-only skip and one
  pre-existing warning; workspace-v1 passed 103/103 and focused census tests
  13/13.
- Planned artifacts:
  `experiments/runs/2026-07-18-E46P-anchored-range-pilot12.jsonl` and
  `experiments/runs/2026-07-18-E46P-anchored-range-pilot12.diagnostics.jsonl`.
  Require absent final and partial paths, clean static checks, and exact
  contract recomputation. Run only this pilot, then stop for sealed audit and
  registered-gate resolution. No pilot row is behavior-cloning eligible.
- Result, sealed 07-18: the run completed all 204 registered logical calls and
  192 transitions. The canonical run has 242 records and SHA-256
  `d69d22d4fc6e51ceb41e0ab7af9801f7f006c0c11467384c358d58f8408dc058`.
  The diagnostic has 206 records, one header plus 205 HTTP attempts, and
  SHA-256
  `dfbe7bbee1243f123f39049c8f14133ad474a456f03e1343154f2d5155b79f22`.
  The run contract is the prepared
  `b5ab0b01079883ba7f93d79f30480195aeabf2333abcdf05f50f64717dba0ae2`.
  Exact replay, catalog binding, request isolation, seeds, record counts,
  role-bound attempts, source projection, hidden-field separation, verifier
  replay, no-clobber, and artifact audits all pass. There were zero hosted
  base calls. Builders used 14,481 prompt and 629 generated tokens across
  thirteen attempts. Workers used 149,760 prompt and 43,313 generated tokens
  across 192 attempts. Provider metadata still exposes no worker route price,
  so no price is inferred.
- State-gate result: FAIL. Only 11/12 builders were usable, producing 33/36
  controlled ranges and eleven complete, address-distinct states. The final
  problem, `cw1-6859921b4cde`, produced the same discrete response on its first
  attempt and parse retry: `recheck_model` on `S001`, `test_admissibility` on
  `S005,S006,S007`, and `check_finalization` on `S012,S013,S014`. Both latter
  arrays exceeded the registered two-endpoint schema even though each is an
  ordered, coherent local range. The rejected response contains no authored
  mathematical prose. This is an interface-cardinality failure, not a
  symbolic or semantic rejection of its algebra.
- Two reward-blind inspections independently found no structural error among
  the eleven accepted states: all endpoints resolved and increased, expanded
  ranges were disjoint, exact offsets and hashes replayed, trusted node
  reconstruction was exact, no range covered the full draft, and no selected
  task unequivocally subsumed another. Six states had clearly apt and
  complementary operation-range pairs. Five were marginal but not registered
  failures: they omitted salient later work, clustered checks on one branch,
  or weakly paired `recheck_model` with a restatement or terminal conclusion.
  These omissions are retained as descriptive limitations, not filtered.
- Accepted-state tabulation: `recheck_model` appeared 11 times,
  `recompute_local` 9, `test_admissibility` 7, `check_finalization` 6, and
  `check_coverage` 0. Expanded widths were 25 one-span, 7 two-span, and 1
  three-span range, with mean 1.273, median 1, and maximum 3. Normalized range
  midpoints had mean 0.449 and median 0.400; early, middle, and late terciles
  contained 13, 10, and 10 ranges. Node-slot midpoint means were 0.075, 0.434,
  and 0.838. Node 1 was early in 9/11 states and used `recheck_model` in 10/11;
  node 3 was late in 8/11 and used `check_finalization` in 6/11. Thus node IDs
  encode strong positional and operation templates, an explicit concern for
  any later observability claim.
- Worker-interface result: FAIL independently of the state failure. Total
  usable outputs were root 45/48, node 1 44/48, node 2 46/48, and node 3
  46/48, versus a registered minimum of 46/48 for every target. Evaluation
  counts were root 22/24, node 1 22/24, node 2 24/24, and node 3 23/24,
  versus 23/24 for every target. Overall usability was 181/192, below 184/192.
  All eleven unusable revisions were transport-accepted, schema-valid JSON
  objects whose `answer` field was empty, so Candidate validation marked them
  unusable. One occurred on problem index 2 and ten on the taxi problem at
  index 4. There were no action-worker HTTP or schema-generation failures and
  no evidence that the 768-token cap caused this interface failure.
- Resolution: E46P is RESOLVED NEGATIVE under its registered state and
  interface gates. Per the stopping rule, its symbolic reward summary is not
  interpreted, no outcome cell is reported as evidence, and every row remains
  ineligible for behavior cloning or value learning. The only authorized
  continuation is a separately registered schema-tolerance integration check
  for the observed ordered-ID cardinality. It cannot retroactively repair or
  authorize reward interpretation for E46P.

### E46I-R2 - PRE-REGISTERED 07-18: ordered range-list tolerance smoke
- Purpose: test one final, narrow state-interface repair on the exact E46P
  builder failure. The safety invariant is that the builder selects only
  controlled operations and frozen source IDs. Requiring it to encode every
  multi-span range as exactly two endpoints is not part of that invariant and
  rejected an otherwise coherent discrete selection.
- Sole manipulation: replace `anchored_ranges_v1` with a new versioned
  `anchored_range_lists_v1` state-builder contract. A node may contain one or
  more unique, strictly increasing source IDs. Trusted code treats the first
  and last IDs as inclusive endpoints; intermediate IDs are redundant but
  valid range witnesses. The schema maximum is the sealed catalog size.
  Pairwise expanded-range disjointness, exact source reconstruction, the five
  operation keys, three-node count, no-free-form-output invariant, answer
  exclusion, and all other state semantics remain unchanged. Preserve
  `anchored_ranges_v1` exactly for historical replay.
- Frozen conditions: use only source row 13, problem
  `cw1-6859921b4cde`, and its exact E44P base Candidate. Keep the E46P worker,
  builder, model revisions, temperatures, token limits, seeds, parse retry,
  action prompt, four repetitions, Williams order, symbolic verifier,
  artifact protocol, and replay source unchanged. Make no hosted base call.
- Exact size: one builder logical call plus sixteen action logical calls,
  sixteen transitions, and 22 canonical run records. The diagnostic has one
  header and at least seventeen attempts. A builder parse retry increases only
  the diagnostic attempt count and hosted generation count. Require seventeen
  unique logical call IDs and exact replay provenance.
- Reward-blind gates: require 1/1 usable builder, 3/3 valid controlled ranges,
  one complete address-distinct state, and 16/16 usable worker revisions.
  Every selected ID must resolve and increase strictly, expanded ranges must
  be pairwise disjoint, and trusted replay must reproduce exact titles,
  contents, order, hashes, and state hash. Builder output must contain only
  operation keys and source IDs. Fail on a whole-draft range, clearly
  nonsensical operation-range pairing, or one task that semantically subsumes
  another. Inspect the three pairs before any outcome view. Require every
  artifact, source, catalog, request-isolation, seed, role, retry, usage,
  hidden-field, verifier-replay, ASCII, and no-clobber audit.
- Interpretation and stopping: this smoke has no reward estimand. Do not use
  or report its rewards as evidence. Failure ends the cardinality-repair
  branch. Pass establishes only that the observed representation can be
  transported without arbitrary ID-count rejection and authorizes a separate
  preregistration decision about a complete twelve-case rerun. It does not
  authorize combining new rows with E46P or changing the worker token limit.
- Planned artifacts:
  `experiments/runs/2026-07-18-E46I-R2-range-list-smoke1.jsonl` and
  `experiments/runs/2026-07-18-E46I-R2-range-list-smoke1.diagnostics.jsonl`.
  Before any hosted call, record exact prepared source, evaluation, replay,
  catalog, prompt, package, and run-contract bindings below and verify absent
  final and partial paths. No row is behavior-cloning eligible.
- Pre-call feasibility amendment, 07-18: local `--print-contracts` preparation
  made no network call and failed closed because the sealed E44P replay
  protocol requires the complete ordered source projection. It cannot select
  one base Candidate while retaining the source contract, original problem
  indices, and effective seed schedule. Do not weaken that replay invariant
  or add an untested offset mechanism for this smoke. Before any hosted call,
  supersede only the source size, exact call count, aggregate worker gate, and
  artifact names above as follows. Use all twelve E44P cases in their original
  order, rows 2 through 13, and replay the complete sealed base projection.
  Execute twelve builder calls plus 192 action calls, 192 transitions, and 242
  canonical run records. Require 12/12 usable builders, 36/36 controlled
  ranges, and twelve complete address-distinct states. Require at least 46/48
  usable outputs for each worker target, 184/192 overall, and 23/24 evaluation
  outputs for each target. All per-state semantic and structural gates remain
  unchanged. Use planned artifacts
  `experiments/runs/2026-07-18-E46I-R2-range-list-smoke12.jsonl` and
  `experiments/runs/2026-07-18-E46I-R2-range-list-smoke12.diagnostics.jsonl`.
  This is still a reward-blind interface smoke. Passing authorizes only a
  separately preregistered outcome run; no E46I-R2 reward may be inspected or
  combined with E46P.
- Prepared bindings, before hosted execution: dataset
  `4d3ae6dc8a2cbcbbb0a8d4292a9eb0553064edf60b58fe78d6dbfcd4577c2b5e`;
  problem projection
  `1b48b182df6c0e7dd30ce80a75a9f0757ab896fff7fdf2eb66dbcec8e95ceb7d`;
  hidden evaluation binding
  `2376afa61758eab49d2237dce317e14affec76f3a56403dda25e2c4a6d03964e`;
  base projection
  `a8eb8ea3f9f4ba0eef9f9fe8073e098a5cc4898380f7c648fd479ad48f90f424`;
  anchored catalog
  `e06840c7978ff7bb158f919e26918cc6b54c79066d4f02581b6f0e1754f14481`.
  Base, builder, and action prompt digests are
  `b30d5e28a67a029b672037e3765f65ea6e720b93287baa0cd0ec5a5118b622c1`,
  `14b8462ed2fe56b1cba8c85253f0e4b473e70f359bbae1ab1a2a6cca9ec0029f`,
  and `62061e8f77ffa9af1b06357f019c5fd8f763871578042a6539d14e2bdbb9e94d`.
  Package tree is
  `0d4f1ed64bd109e1de9b6f978c82c3c6780d86bba79142f9c09dcca5648f596d`
  under Python 3.11.15, httpx 0.28.1, and SymPy 1.14.0. Run contract is
  `d671586b0d4d13b71e3fbfadf707946a4337b51f97a6e47abe6b300b0aa41428`.
  The exact package passed 995 tests with one CUDA-only skip and one
  pre-existing PyTorch warning; workspace-v1 passed 104/104 and focused
  census tests 14/14. Static formatting, compilation, diff, ASCII, path
  absence, and local contract preparation checks pass.
- Result, sealed 07-18: the run completed all 204 logical calls and 192
  transitions, publishing 242 canonical run records. Run SHA-256 is
  `88fe1bd9f4f4ddcf10e92332553fc0b0c31d5cbe89329d02e67cd5475c670b6b`.
  The diagnostic contains one header and 206 attempts, with SHA-256
  `775d85e4e0ba74f6f07279d80dfc5c0db7880e500f049d077f94253693278617`.
  There were twelve state-builder attempts, 194 action-worker attempts, and
  zero base-worker attempts. Every source, replay, catalog, contract, request,
  target isolation, common-seed, call-id, retry, usage, state-hash,
  verifier-replay, hidden-field, role, record-count, no-clobber, and artifact
  audit passes. Providers do not attest immutable served weight revisions.
- State gate: PASS. All 12 builders were accepted on their first attempts,
  all twelve states were usable, complete, and address-distinct, and all 36
  controlled selections were valid. Every ID resolved, was unique and
  strictly increasing, expanded ranges were pairwise disjoint, exact source
  offsets, hashes, contents, titles, order, and state hashes replayed, and
  builder JSON contained only operation keys and source IDs. No individual
  range covered a whole draft, no builder-authored prose or new answer value
  entered a node, and no selected task plainly subsumed another. The three
  three-ID lists reconstructed correctly: problem indices 8 and 10 selected
  three-span recomputation ranges, and the formerly rejected final case
  selected `S012,S013,S014` for finalization.
- Reward-blind semantic inspection: no selection met the registered clear
  failure rule. Case 0's three ranges collectively cover its short four-span
  draft. Case 4 rechecks a low-information tariff statement. Case 8 rechecks
  only variable naming and misses the central tier-applicability issue. Case 9
  rechecks a problem restatement. The final case's `S012-S014` range mixes an
  admissibility check with finalization. These are material localization
  weaknesses but are not a whole-draft range, nonsensical operation, or task
  subsumption, so they are retained without filtering.
- State tabulation: `recheck_model` appears 12 times, `recompute_local` 10,
  `test_admissibility` 4, `check_finalization` 10, and `check_coverage` 0.
  Expanded widths are 23 one-span, 10 two-span, and 3 three-span ranges, with
  mean 1.444. Normalized midpoint mean is 0.452 and median 0.412; early,
  middle, and late terciles contain 15, 10, and 11 ranges. Node-slot midpoint
  means are 0.017, 0.450, and 0.891. Every node 1 uses `recheck_model`, 11/12
  start at `S001`, and 8/12 states use the exact operation sequence
  `recheck_model,recompute_local,check_finalization` displayed in the prompt
  example. Node 3 uses finalization in 10/12. This strong positional and
  example-template signature indicates limited semantic tailoring and makes
  node ID, phase, and operation strong nonsemantic baselines for any future
  observability test.
- Worker interface gate: FAIL. Usable totals were root 47/48, node 1 45/48,
  node 2 45/48, and node 3 44/48, versus at least 46/48 for every target.
  Evaluation usability was root 24/24 and only 22/24 for each node target,
  versus at least 23/24. Overall usability was 181/192, below 184/192. Ten
  unusable outputs on the taxi problem were transport-accepted schema-valid
  objects with an empty `answer`. Their reasoning repeated the draft's false
  assumption that a trip shorter than 15 miles must still pay for 15 miles,
  despite the complete problem and independent-solve instruction. One node 2
  call on problem index 2 returned malformed `answer:` JSON twice and failed
  after its parse retry. A separate first-attempt schema error recovered on
  retry. This used 154,463 worker prompt and 43,479 generated tokens. Builders
  used 13,296 prompt and 580 generated tokens. No unexposed route price is
  inferred.
- Resolution: E46I-R2 is RESOLVED NEGATIVE overall. It establishes that the
  ordered range-list representation removes the arbitrary endpoint-count
  rejection, but the frozen full-state worker interface still fails its
  registered usability gate. Do not inspect or report the run's reward cells,
  do not combine it with E46P, and do not run an E46P-R1 outcome pilot,
  behavior cloning, Q-learning, or pointer training on these rows. The
  cardinality-repair branch is closed. The failure directly motivates only a
  separately registered target-local-lens isolation or a clean factored
  operation-by-span experiment: the current target marker leaves the worker
  exposed to the entire erroneous draft and is not a localized intervention.

### E47I - PRE-REGISTERED 07-18: target-local lens factorial integration
- Purpose: establish a transport-valid experiment for the next isolated
  research question: does a real target-local projection improve a frozen
  small worker relative to the current full-state target marker, and can
  either structured view compete with direct problem-only prompting? E47I is
  integration-only. It has no reward estimand and cannot support a lens,
  target, controller, or accuracy claim.
- Source: read only the run header and `census_state` records from sealed
  E46I-R2 run
  `88fe1bd9f4f4ddcf10e92332553fc0b0c31d5cbe89329d02e67cd5475c670b6b`,
  diagnostic
  `775d85e4e0ba74f6f07279d80dfc5c0db7880e500f049d077f94253693278617`,
  and source contract
  `d671586b0d4d13b71e3fbfadf707946a4337b51f97a6e47abe6b300b0aa41428`.
  Select source problem indices 0 and 11 before outcomes: the shortest state,
  whose three children collectively cover four spans, and the formerly
  rejected state whose final node uses a three-ID range. Their ids are
  `cw1-88b5649691fe` and `cw1-6859921b4cde`. Selection uses no transition,
  case-summary, run-summary, reward, analysis, or hidden answer field. Bind
  exact source observations and state hashes into a new source-selection
  digest before calls.
- Pre-call source-boundary clarification, recorded before any E47 hosted
  request: the complete source files must be read to verify their registered
  SHA-256 digests, but preparation parses and retains only the run header,
  `census_state` rows, and diagnostic header. It does not parse or retain a
  source transition, case summary, run summary, reward, analysis, or attempt.
- Single causal factor and direct reference: for each exact source State and
  each target in `root,node_1,node_2,node_3`, contemporaneously call two view
  modes. `full_state_v1` is the byte-exact E46 action request: the complete
  state plus a selected-target marker. `target_local_v1` retains the same
  system message, answer schema, final instruction, target marker, model, and
  settings, but replaces the full state with a trusted projection containing
  only problem id, complete problem text, the selected REVISE action, and the
  selected node. Root-local exposes only the root Candidate; child-local
  exposes only that child's fixed title and exact anchored source content. It
  must exclude the root Candidate and every nonselected child. A ninth
  `problem_only_v1` arm uses the retained direct base request and is the
  simpler-system benchmark, not part of the full-versus-local causal contrast.
- Frozen execution: use only
  `Qwen/Qwen2.5-3B-Instruct:featherless-ai` through the Hugging Face router,
  with registered revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`, temperature 0.2, 768 maximum
  completion tokens, one parse retry, six 429/5xx retries, 120-second timeout,
  JSON-object mode, guided JSON disabled, and base seed 451000. There is no
  builder, coordinator, controller, verifier, COMMIT, or learned model call.
  The local symbolic verifier is invoked only after generation by the trusted
  experiment boundary.
- Repetitions and order: run four repetitions per arm and case, retaining the
  discovery/evaluation labels 0-1 and 2-3 only for compatibility with a later
  outcome protocol. All nine calls within a case-repetition block share the
  same initial seed; a registered parse retry increments only that call's
  effective seed. Use a fixed 18-row Williams schedule for nine arms,
  consisting of nine balanced rows and their reversals, indexed by source
  problem and repetition. Record arm position. This stage does not interpret
  order strata.
- Exact size and artifacts: make 72 logical worker calls, two cases times nine
  arms times four repetitions, and publish 72 transitions. With no retry, the
  diagnostic has one header plus 72 attempts. The canonical run has one
  header, two source-state records, 72 transitions, two case summaries, and
  one run summary, 78 records total. Planned artifacts are
  `experiments/runs/2026-07-18-E47I-target-local-factorial-smoke2.jsonl` and
  `experiments/runs/2026-07-18-E47I-target-local-factorial-smoke2.diagnostics.jsonl`.
- Failure-to-reward invariant: every scheduled arm produces exactly one
  transition. An exhausted transport or schema call, malformed payload,
  absent answer, or empty answer becomes a Candidate failure with symbolic
  terminal reward zero. It is never filtered, imputed, retried outside the
  registered client policy, or used to fail integration by itself. This is
  the live `symbolic-only-v1` environment behavior: failed generative actions
  consume their call and receive no positive verifier reward. Usability and
  failure mode remain required diagnostics.
- Reward-blind integration gates: require all 72 scheduled transitions, nine
  arms per case-repetition block, unique logical call IDs, exact common seeds,
  Williams positions, model/settings parity, exact full-state request replay,
  and exact target-local projection. Exclusion is structural: child-local has
  no root Candidate field or record and no nonselected node field or record;
  root-local has no child field or record. It is not a substring ban because
  the complete problem or selected source may legitimately repeat text or a
  scalar that also occurs elsewhere. Direct requests must contain no state or
  target. Bind
  every request, raw attempt, retry, response, usage field, source state,
  state hash, Candidate, and arm position. Require exact source and hidden
  evaluation separation, no credential publication, canonical ASCII JSONL,
  final/partial no-clobber behavior, verifier replay, and complete artifact
  replay. Report failures and tokens without inspecting reward counts.
- Interpretation and stopping: passing authorizes only a separately
  preregistered twelve-case development outcome run with the exact same
  2-by-4 view-target factorial and direct reference. One implementation-only
  repair may be separately registered if a request, publication, or replay
  invariant fails. No reward or target choice may guide that repair. A passing
  smoke supplies no behavior-cloning row. The later outcome run must count all
  failures as zero and must distinguish three decisions: whether local beats
  full view, whether local targets have repeatable action-value heterogeneity,
  and whether a discovery-selected local target beats direct prompting. Only
  the conjunction could justify fresh confirmation or controller training.
- Pre-call boundary: implement and test the collector, then append exact
  source-selection, problem, hidden-binding, prompt, package, model, and run
  contract hashes below. Require the focused and full test suites, static
  formatting, compilation, diff, prohibited-glyph, credential, and absent-path
  checks to pass before any hosted request. Expected spend is small relative
  to the authorized $20 budget, but no unexposed provider price is invented.
- Pre-call verification: `.venv/bin/python` is Python 3.11.15. The focused
  E47 suite passed 4 tests, including sealed-source, failure-as-zero, exact
  replay, and tamper checks. The full suite passed 999 tests with one expected
  CUDA skip and one pre-existing PyTorch warning in 951.04 seconds. Black,
  compilation, the prohibited-glyph scan, and both sealed source projections
  also passed. No E47 hosted request had been made when these results and the
  structural-exclusion clarification were recorded.
- Frozen pre-call bindings: preregistration id
  `E47I-target-local-factorial-smoke2`; dataset
  `4d3ae6dc8a2cbcbbb0a8d4292a9eb0553064edf60b58fe78d6dbfcd4577c2b5e`;
  source selection
  `b4e1a06e1344d66af64497dc7b44e6f18c12b3b5a54995ae27983f7e526e382f`;
  selected public problem projection
  `3fcda59bd21f5bcb885e3fce0c79e3a742471471a72e0d62101a3bbdf2e25002`;
  hidden evaluation binding
  `69103d6dba063530fa7ea58ac5755f36a3ff0ffffe2503851024e15024cccceb`;
  nine-arm order schedule
  `b14f1d7c4b3e8bd3e7bc45a7f07953ec98fae47d2b92d90530ef0e81881b4a5b`;
  problem-only prompt
  `b30d5e28a67a029b672037e3765f65ea6e720b93287baa0cd0ec5a5118b622c1`;
  full-state prompt
  `62061e8f77ffa9af1b06357f019c5fd8f763871578042a6539d14e2bdbb9e94d`;
  target-local prompt
  `8e67fd14d93a68f696adfc089fa75793d8946b0e4aef0d28c94976e96442e5c1`;
  Python package tree
  `c6defce92a5c44bbf27b6edf0ffe6788a77c693bfbd54dd2e6c99ecfbaf5c8db`;
  Python 3.11.15, httpx 0.28.1, and SymPy 1.14.0; and run contract
  `867f93473620373478cacc096ca3ca2a1ee9dc7230c6bcfe7c02320c04a60481`.
  The worker and its registered revision and settings are exactly those in
  the frozen-execution paragraph above. These bindings were appended before
  any hosted call.
- Sealed integration artifacts: canonical run
  `dc014be483024dd3218e2c7c9e0bb60df8b70df743a12493784e093ad4833c7e`
  and diagnostic
  `4ea1038be0747a4ac5bf4888ae40880c6202d4f9ebf3c5657186bc7710c7a74a`.
  The run has exactly 78 records and the diagnostic has 77: one header, 72
  logical-call attempts plus four retry attempts. Partial paths are absent.
- Reward-blind integration result: PASS. All 72 scheduled calls produced 72
  transitions over the exact two sealed States. Exact call-id, request-hash,
  source-state, prompt-isolation, common-seed, Williams-order, summary,
  retry-transcript, raw-provider-response, usage, forbidden-field, Candidate,
  and local symbolic-verifier replay checks passed. All response-model fields
  reported `Qwen/Qwen2.5-3B-Instruct`; the route does not attest the registered
  immutable weight revision.
- Operational diagnostics: 71/72 logical calls returned usable Candidates.
  The only failure was `local_node_3`: four of its eight calls needed the
  registered parse retry, three recovered, and one exhausted that retry. That
  failed generative action remained as its scheduled transition with an
  unusable Candidate and symbolic terminal reward zero, as required. Every
  other arm returned 8/8 usable Candidates. Worker usage was 76 provider
  generations, 39,760 prompt tokens, and 18,271 generated tokens. No reward
  count, answer, or embedded analysis was inspected for this integration run.
- Resolution: E47I is RESOLVED PASS. It establishes only that the exact
  full-versus-local factorial and direct reference can be executed and
  replayed with live failure-as-zero semantics. It makes no accuracy, lens,
  action-value, controller, or state-quality claim. It authorizes the
  separately preregistered twelve-case development outcome run and supplies
  no behavior-cloning, Q-learning, or pointer-training row.

### E47P - PRE-REGISTERED 07-18: target-local lens factorial development pilot
- Purpose and claim limit: estimate three separate mechanisms on the twelve
  sealed development States: whether a child-local view improves the same
  frozen worker call relative to full-state distraction, whether a simple
  fixed child-local policy beats direct prompting and root self-revision, and
  whether child-target values repeat across independent generation splits.
  E47P is an adaptive development screen with twelve independent State
  clusters. It is not a confirmation, semantic-observability result,
  controller result, end-to-end efficiency result, or training dataset.
- Source: use all ordered source problem indices 0 through 11 from the exact
  sealed E46I-R2 run, diagnostic, source contract, and dataset registered for
  E47I. Preparation hash-reads the complete files but parses and retains only
  the run header, twelve `census_state` rows, and diagnostic header. It binds
  the ordered public State projection and hidden symbolic-evaluation binding
  before calls. No E46 or E47I transition, reward, answer, analysis, or target
  choice is read or reused.
- Single worker manipulation and references: retain the exact nine E47I arms,
  prompts, schemas, target-local projections, and structural-isolation rules.
  The causal comparison is `target_local_v1` versus `full_state_v1` at the
  same child target and seed. `problem_only_v1` is the direct one-call
  reference. `local_root` is the simpler iterative self-revision reference;
  it is never treated as a child lens or allowed to select a child policy.
- Frozen execution: use only the E47I worker route, model, registered revision,
  temperature 0.2, 768-token maximum completion, one parse retry, six 429/5xx
  retries, 120-second timeout, and guided JSON disabled. Use new base seed
  452000, never an E46 or E47I worker seed. There is no builder, coordinator,
  controller, learned model, LLM judge, or verifier call during generation.
  The local symbolic verifier is the only reward source and runs only at the
  trusted experiment boundary after each scheduled generation.
- Size and order: run four repetitions for every arm and State, with discovery
  repetitions 0-1 and evaluation repetitions 2-3. All nine arms in a
  State-repetition block share the same initial seed. Retain the fixed 18-row
  Williams and reversal schedule and exact arm positions. Run all 432 logical
  calls with no outcome-dependent stop. The canonical run must contain one
  header, 12 source records, 432 transitions, 12 case summaries, and one run
  summary, 458 records total. The diagnostic has one header plus every raw
  attempt. Planned artifacts are
  `experiments/runs/2026-07-18-E47P-target-local-factorial12.jsonl` and
  `experiments/runs/2026-07-18-E47P-target-local-factorial12.diagnostics.jsonl`.
- Failure invariant: every transport, provider, parse, schema, absent-answer,
  and empty-answer failure remains in its scheduled cell as an unusable
  Candidate with symbolic terminal reward zero. No failed row is filtered,
  replaced, imputed, or rerun outside the registered client retry policy.
- Independent unit and shared inference: the independent generalization unit
  is the State, not an individual call. For every comparison report the point
  effect, paired cells, all twelve State-level effects, positive/negative/tie
  State counts, 10,000 whole-State bootstrap 90% and 95% intervals, and an
  exhaustive `2^12` whole-State one-sided sign-flip reference probability.
  Bootstrap seeds for Questions 1 through 3 are 457001, 457002, and 457003.
  The sign-flip value is an exchangeability reference, not a design-exact
  test, because condition-to-call-slot assignment was not randomized. Also
  report effects within each of the four three-State families and every
  leave-one-family-out estimate. The screens below use 90% interval lower
  endpoints only as development decisions; they do not establish a
  confirmatory alpha level.
- Question 1, broad child-locality effect: for each State `i` and split `S`,
  define `L_i(S)` as the mean over child targets `node_1,node_2,node_3` and
  repetitions in `S` of `reward(local)-reward(full)` at the same target. The
  primary estimate is the mean of `L_i(evaluation)`. The development screen
  passes only if its evaluation effect is at least `1/12`, equivalently six
  net local wins among 72 paired cells; its whole-State bootstrap 90% lower
  endpoint is above zero; and its discovery effect is positive. Report every
  target-specific effect but do not use a single win on a target as a gate.
- Question 2, fixed child policy versus simpler systems: select one global
  fixed child target using only aggregate discovery accuracy of the three
  local-child arms, with tie order `node_1,node_2,node_3`. Freeze it before
  evaluation. Compare that same selected local-child arm on the 24 evaluation
  cells against direct problem-only, `local_root`, and full-state revision at
  the same child target. The direct screen requires an effect of at least
  `1/8`, equivalently three net wins among 24, and a whole-State bootstrap 90%
  lower endpoint above zero. The tree-specific fixed-policy screen also
  requires the selected child to beat `local_root` with a positive effect and
  a 90% lower endpoint above zero. This is a conditional one-worker-call
  comparison. It excludes the cost and accuracy effects of producing the
  initial draft and State, so it cannot support an end-to-end efficiency
  claim.
- Question 3, reward-probed child-target repeatability: within each State,
  select the unique discovery-best local child target. On a discovery tie,
  use the already frozen global fixed child target, not a target-ID tie break.
  On evaluation compare this State-specific outcome-probed selection with the
  global fixed child target. Its adaptive-value screen requires an effect of
  at least `1/12`, equivalently two net wins among 24; a whole-State bootstrap
  90% lower endpoint above zero; and a State-specific target different from
  the fixed target on at least three States.
- Question 3 interaction diagnostic: form discovery and evaluation matrices
  of local-child target means, double-center each by State and target to
  remove State difficulty and global target-ID effects, and take their
  cross-split inner product. Compare it with 100,000 fixed-seed permutations
  of complete evaluation State rows, using seed 457004 and the plus-one
  correction. Preserve target columns and within-State dependence. Require a
  positive interaction statistic, but report rather than gate on its
  permutation probability because the pilot is underpowered. Report
  discovery tie counts, State selections, States differing from the fixed
  target, and strict discovery/evaluation winner recurrences. Question 3
  passes only when both adaptive value and positive interaction pass. It can
  authorize a later semantic-observability audit, never establish semantic
  observability itself.
- Power reading fixed before outcomes: with only twelve State clusters, the
  pilot is powered only for large effects. Under an illustrative paired-cell
  discordance of 0.30 and within-State difference correlation of 0.25, an
  approximate 80% detectable effect is about 25.5 percentage points for
  Question 1 and 33.0 points for Question 2. Rough confirmation sizes are 31
  States for a 15-point Question 1 effect, 70 for 10 points, 29 for a 20-point
  Question 2 effect, and 52 for 15 points. A null or interval-crossing result
  may therefore be inconclusive rather than proof of no effect.
- Separate stopping decisions: never collapse the three screens into one
  scientific claim. If child locality passes but the child policy fails
  direct or `local_root`, record locality as a mechanism only and prefer the
  simpler system. If the fixed child policy passes but repeatability fails,
  retain the fixed policy and do not train a selector. Only if all three pass
  may the next experiment test semantic-state observability on fresh States
  using a frozen state-only predictor against fixed target, target-ID/order,
  topology/menu, shuffled-semantic, and information-matched flat baselines.
  E47P never directly authorizes behavior cloning, Q-learning, pointer
  training, or a publishable efficacy claim.
- Integrity and pre-call boundary: require the exact 12-by-4-by-9 grid,
  unique call ids, common initial seeds, Williams positions, closed record
  schemas, exact request and State replay, structural lens isolation, complete
  raw response and retry replay, symbolic reward replay, forbidden-field and
  credential scans, canonical ASCII JSONL, no-clobber publication, absent
  partials, prompt hashes unchanged from E47I, focused and full green tests,
  compilation, Black, diff, and prohibited-glyph checks. Append the exact
  all-State selection, public problem, hidden evaluation, order, pilot
  analysis, package, model, and run-contract bindings below before the first
  hosted E47P request. Expected usage is about six E47I runs, safely within
  the authorized $20 budget at the observed route scale, but do not invent a
  provider price that the route does not expose.
- Pre-call verification and bindings: focused E47 tests passed 7/7, workspace
  tests passed 111/111, and the full suite passed 1002 tests with one expected
  CUDA skip and one pre-existing PyTorch warning in 1072.91 seconds. Black,
  compilation, diff, prohibited-glyph, credential, sealed-source replay, and
  absent-output checks passed. Frozen preregistration id is
  `E47P-target-local-factorial12`; dataset
  `4d3ae6dc8a2cbcbbb0a8d4292a9eb0553064edf60b58fe78d6dbfcd4577c2b5e`;
  all-State source selection
  `db0d91e16c104ec6aa035173e2e048fa202d72b9bf61d502af3fe961a220a203`;
  public problem projection
  `1b48b182df6c0e7dd30ce80a75a9f0757ab896fff7fdf2eb66dbcec8e95ceb7d`;
  hidden evaluation binding
  `2376afa61758eab49d2237dce317e14affec76f3a56403dda25e2c4a6d03964e`;
  Williams schedule
  `7e5615d98fd9878197f13e52ac742575447ff135f095c2e71f311bb05b2b7bc7`;
  problem-only, full-state, and target-local prompt bindings respectively
  `b30d5e28a67a029b672037e3765f65ea6e720b93287baa0cd0ec5a5118b622c1`,
  `62061e8f77ffa9af1b06357f019c5fd8f763871578042a6539d14e2bdbb9e94d`,
  and `8e67fd14d93a68f696adfc089fa75793d8946b0e4aef0d28c94976e96442e5c1`;
  pilot analysis schema `action-lens-pilot-analysis-v2`; Python package tree
  `14a0de0d6fbc035696271dd4d16530dea3a2c72537984ac18da26e954555fb20`;
  Python 3.11.15, httpx 0.28.1, and SymPy 1.14.0; and run contract
  `294ae913786fa7e202e2f1560a8ec716b04c23f5276a2fd4d64aeac1d3407a4c`.
  The worker identity, revision, and settings are exactly those registered
  above. No E47P hosted request had been made when these bindings were
  appended.
- Execution and sealed artifacts: the complete 432-call run exited zero after
  its collector audit and atomic publication. The canonical run is
  `experiments/runs/2026-07-18-E47P-target-local-factorial12.jsonl`, SHA-256
  `75944f77775728bea2a65939457f28f9d6e730e36cba10c6123c47f9d3c155b2`,
  with the registered 458 records. Its diagnostic is
  `experiments/runs/2026-07-18-E47P-target-local-factorial12.diagnostics.jsonl`,
  SHA-256
  `ab0fa0e523e0e35b839b09464f7270caf2ef3965519da8d3a43c9f2396893e99`,
  with 439 records. Both partials are absent. The run contains 12 unique
  source States and all 432 scheduled transitions under run contract
  `294ae913786fa7e202e2f1560a8ec716b04c23f5276a2fd4d64aeac1d3407a4c`.
- Integrity result: source-State, request-hash, call-id, common-seed, arm-order,
  retry-transcript, symbolic-reward, summary, and usage replay all passed.
  Prompt isolation, exact response-model identity, forbidden-field scanning,
  canonical ASCII, and record-layout checks passed. The provider returned the
  requested model name but did not attest an immutable served weight revision,
  as registered. The worker made 438 generation attempts for 432 logical
  calls, consuming 234,488 prompt and 105,158 generated tokens. Six calls used
  the registered parse retry. Three exhausted it, one each in `local_node_2`,
  `local_node_3`, and `local_root`, and remained in place as symbolic zeros.
  Across accepted and exhausted calls there were ten unusable outputs: six in
  the three full-child arms, two in the three local-child arms, two in
  `local_root`, and none in `problem_only` or `full_root`.
- Descriptive accuracy over all four repetitions, not an inferential claim:
  `problem_only` solved 18/48, `local_root` 16/48, and `full_root` 17/48.
  Local child arms `node_1,node_2,node_3` solved 23/48, 24/48, and 19/48;
  their full-state counterparts solved 17/48, 16/48, and 16/48.
- Question 1 result, broad child locality: RESOLVED PASS for the registered
  development screen. On the held evaluation repetitions, child-local solved
  33/72 cells and same-target full-state solved 24/72, an effect of 9/72 =
  12.5 points. There were 10 local-only, one full-only, 23 both-correct, and
  38 both-wrong cells. Five States had positive effects, one negative, and six
  tied; State effects were
  `[0,1/6,0,0,0,1/3,-1/6,1/3,0,1/6,0,2/3]`. The whole-State
  bootstrap intervals were [2.78, 23.61] points at 90% and [1.39, 26.39]
  points at 95%; the exhaustive sign-flip reference probability was 0.0625.
  Discovery independently had an 8/72 = 11.11-point effect, satisfying the
  registered direction check, but its 90% interval touched zero. Evaluation
  target-specific effects were 2/24 for `node_1`, 5/24 for `node_2`, and 2/24
  for `node_3`; these were registered diagnostics, not target-selection gates.
  Family effects were -5.56 points in `f4`, +33.33 in `f6`, zero in `f7`, and
  +22.22 in `f8`. All leave-one-family-out effects remained positive, from
  5.56 to 18.52 points. This is a twelve-State development screen, not a
  confirmation; in particular, its exchangeability reference did not cross a
  conventional 0.05 threshold and family heterogeneity is material.
- Question 2 result, fixed child policy: RESOLVED FAIL. Discovery selected
  `node_1` from local-child accuracies 13/24, 11/24, and 9/24 for
  `node_1,node_2,node_3`. On evaluation the frozen `node_1` policy and direct
  problem-only prompting both solved 10/24, effect zero, 90% State-bootstrap
  interval [-8.33, 8.33] points, with one win and one loss. It solved 10/24
  against `local_root` at 8/24 and same-target full-state at 8/24, but both
  +8.33-point intervals had lower endpoint zero. The direct and tree-specific
  screens therefore both failed. The post-selection observation that
  `node_2` solved 13/24 evaluation cells is not reusable evidence for a fixed
  policy; discovery had ranked it second.
- Question 3 result, reward-probed target repeatability: RESOLVED FAIL.
  Discovery tied on nine of twelve States, so the registered fallback chose
  global `node_1`; the final State-specific choices were eleven `node_1` and
  one `node_2`, only one State different from the fixed policy versus the
  required three. The adaptive policy gained one of 24 evaluation cells over
  fixed, +4.17 points with 90% interval [0, 12.5], below the registered
  two-cell minimum. Only one State had the same strict unique winner across
  splits. The double-centered cross-split interaction was positive at 1/6
  with centered correlation 0.165, but its 100,000-row-permutation one-sided
  probability was 0.2696. The adaptive-value and combined repeatability
  screens failed. This supplies no semantic-observability evidence.
- Root self-revision diagnostic: `local_root` solved 8/24 evaluation cells
  against direct at 10/24, effect -8.33 points with 90% interval [-16.67, 0].
  It also trailed `full_root` by one cell. Exposing the draft at the root did
  not provide a simpler iterative advantage.
- Post-outcome implementation audit, explicitly exploratory: an independent
  table reconstruction reproduced every accuracy, paired cell, State effect,
  and screen above. Five evaluation child pairs contained at least one
  unusable output, but their net local-minus-full reward contribution was
  zero, so operational failures did not create the Question 1 effect. Exact
  string comparison then found that full-child outputs repeated the source
  root answer in 65/72 evaluation cells versus 38/72 local-child cells. In all
  eleven evaluation reward discordances, full copied the root answer and local
  did not. Discovery showed the same pattern: 63/72 versus 32/72, and all
  twelve discordances had full copy while local did not. These copy statistics
  were not preregistered and are mechanism diagnostics only.
- Further post-outcome stratification, also exploratory: the sealed E46I-R2
  source summaries marked eight root drafts wrong and four correct. Among
  wrong-draft States, evaluation full-child solved 0/48, local-child 10/48,
  and problem-only 2/16. Among correct-draft States they solved 24/24, 23/24,
  and 8/8 respectively. This conditioning mixes source correctness with
  problem difficulty and must not be read causally. Together with the exact
  copy audit, however, it identifies the most plausible measured contrast:
  the target-local projection often removes anchoring on a bad root draft,
  while the full view preserves it. It does not show that semantic target
  identity carries stable action value.
- Resolution: E47P is RESOLVED. It found a development-level context-locality
  mechanism relative to a distractor-rich full State, but no advantage for
  the discovery-frozen child policy over direct prompting and no repeatable
  State-specific target signal. Follow the preregistered stopping rule: prefer
  the simpler problem-only system, do not train a selector, and do not launch
  the semantic-observability audit from this pilot. E47P contributes no
  behavior-cloning, Q-learning, pointer, controller-performance, end-to-end
  efficiency, or publishable efficacy row. A future fresh-State experiment
  may separately isolate root-draft exposure from sibling-context exposure,
  but that would be a new context-ablation question rather than continuation
  of a successful target-selection branch.

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
