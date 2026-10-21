# Handoff: state of the work as of 2026-07-17

> Archive preparation update (2026-10-20): the research below is historical.
> Start with `README.md` and `docs/release-readiness.md` for the current work.
> Do not resume the old experimental critical path merely from this handoff.

Public archive update (2026-10-21): version 1.0.0 is published at
https://doi.org/10.5281/zenodo.23526852. Exact source and download verification
are in `docs/publication.json`. The central contribution ID is
`expert-iteration-lessons`, indexed at
https://github.com/queelius/research-records/tree/main/records/expert-iteration-lessons.
Read-only archival follows the publication-receipt commit. The historical
research and current archive verification are distinct records.

(Replaces the stale 2026-06-18 handoff, which described the pre-redesign
outer loop.) Written for resuming this project in another agent/tool.
Read this file, then `CLAUDE.md` (rules), `design.md` (system spec,
source of truth), and `docs/superpowers/research/EXPERIMENTS.md` (the
experiment ledger, E1-E42, the single most important record). Branch:
`action-space-redesign`. As of 2026-07-17, the test suite has 955 passed,
1 skipped (CUDA-only), and one preexisting warning in `tests/test_heads.py`.
The E18-E42, contract-v2, and permissive-workspace work below is not yet
committed.

## 1. What this project is

A verifier-anchored RL system: a tiny EXTERNAL controller network (verb
head + pointer head + Q head; the first 198K-parameter prototype is trained)
learns to orchestrate a FROZEN small LLM by issuing typed (verb, target)
actions over an immutable reasoning forest. A symbolic verifier (SymPy)
is the ONLY reward source. The general design permits terminal +1/-1 at root
closure and zero elsewhere; the live v2 `symbolic-only-v1` contract assigns
+1 to verifier-backed root success and 0 otherwise. No learned reward model,
PRM, ORM, critic, or LLM-judge, ever. Explicitly NOT o1/R1: RL never touches
the LLM's weights.

Archived v1 trajectories used a non-verifier -0.01 masked-action penalty.
That value is preserved only under the explicit
`legacy-mask-penalty-v1` reward contract for exact reproduction. Live and
v2 work use `symbolic-only-v1`; a rejected action consumes a turn but adds
zero reward.

Framing directive (must be restated verbatim in any doc/paper subagent
brief): RL first, MCTS second. The training spine is off-policy
value-based Q-learning with n-step or MC returns; MCTS is a strictly
additive accelerator and the test-time compute knob, never the spine.

The current implementation has six v1 verbs: SPLIT, FORM_HYPOTHESIS,
COMPUTE (LLM-calling, lens-scoped); VERIFY, COMMIT, ABANDON (no LLM
call). The 07-13 audit found that this action algebra hides claim kind,
goal relevance, and the witness chosen by `COMMIT(goal)`. The parallel v2
core now implements typed, owner-scoped Evidence/Claim records, direct Result
claims, separate symbolic validity and owner-discharge receipts, a strict
casework adapter, a four-verb learned menu, and canonical observations.
VERIFY, exact-claim COMMIT, and ready AGGREGATE are deterministic closure,
not learned decisions. The one-turn macro transition and exact symbolic replay
validator are implemented. The label-free casework problem source,
no-network contract preparation, digest-authorized partial-safe hosted
collector, typed generation, bound teacher controller, exact environment and
transport contracts, canonical run records, role-bound raw-call diagnostics,
and trusted-source replay are implemented. Replay-sealed dataset conversion,
whole-problem grouped splitting, the exact legal-menu comparator,
deterministic CPU behavior-cloning plus Monte Carlo warm-start,
value-sensitive checkpoint loading, and the checkpoint-backed learned
controller are also implemented. The current generator prompt contract is
`typed-claim-lens-v2.1`; the exact `typed-claim-lens-v2` request builder is
retained for legacy replay. Current generator and teacher wire contracts are
`typed-policy-generation-v2.1` and `controller-choice-wire-v2.1`; both bind
nonmutating strict schema validation. Their prior `v2` IDs map to legacy
coercion only for exact historical replay. E23 exercised the production
collection boundary.
E24 and E26-E29 are generator diagnostics; E25 is an offline symbolic rescore.
They are integration and interface diagnostics, not a usable training
collection. E28's coerced semantic rows, primary-gate counts, and conclusions
are permanently quarantined, while its raw attempts remain valid audit inputs.
The strict E29 rerun failed every preregistered gate. Execution of the v2
warm-start still requires fresh training data. See
`docs/superpowers/specs/2026-07-13-state-action-contract-v2.md`.

Three-rung training ladder: (1) LLM-as-controller (a teacher LLM picks
actions; trajectories logged as behavior-cloning data), (2) BC
warm-start of the heads, (3) off-policy Q-learning fine-tune. Rung 1 and
the exploratory v1 BC experiments are done. The v2 generator, run-record,
replay, feature, dataset, split, comparator, warm-start, checkpoint, and
learned-controller boundaries are complete. E34 established the causal value
of `preceding-siblings-v1` on its reused development screen. E35 then ran the
first permissive teacher on 60 held-out casework problems: direct solved 59,
fixed decomposition 27, and GPT-4.1-mini teacher control 31. The current
baseline-anchored workspace v2 implements a direct-prefix state, a root
answer-field contract, active-frontier legality, and indexed teacher choices
as independently bound modes. E36 isolated the root answer-field factor on 59
exact sealed E35 states: verifier-ready solved 49/59 versus legacy 27/59, with
22 verifier-ready-only and zero legacy-only. Retain that root contract. The
E37 exact shared-direct-Candidate comparison then used Qwen2.5-3B, which had
real headroom at 43/60 direct. Teacher continuation was also 43/60: 43 both,
zero discordances, and 17 neither. The baseline anchor prevented every
regression, but the teacher always submitted the protected direct Candidate.
All 14 selected REVISE calls failed to create an answer-bearing revision. E38
then isolated those exact states: the new answer-container contract produced
14/14 usable revised Candidates versus 0/14 for the contemporary legacy
control, with 14 treatment-only pairs. Retain that contract. E39 then repeated
the exact shared-direct-Candidate comparison with that retained contract.
Direct and teacher again tied at 43/60, but all 12 revisions were usable and
seven were correct. Three corrected baseline failures, and none damaged the
four correct baselines in the revised subset. Both submit targets were legal
after every revision, yet the controller submitted Candidate 1 all 12 times.
The empirical critical path is now a coordinator-prompt comparison on those
sealed post-revision states, but E40 showed that broader selection coverage is
needed first. Fixed revision on all 60 E39 prefix states produced 60/60 valid
Candidates and 49/60 correct revisions versus 43/60 correct baselines: 43
both, six revision-only, zero baseline-only, and 11 neither. This latest-
revision dominance cannot test preservation-aware selection. The next isolated
factor is an independent-solution REVISE prompt intended to create a useful
bidirectional Candidate corpus. E41 found that factor positive: retained
revision solved 49/60 and independent revision 55/60, with six independent-
only and zero retained-only cases. Against baseline, independent revision had
43 both, 12 revision-only, zero baseline-only, and five neither. It still
cannot test candidate preservation, but its dominance authorizes the next
initial coordinator routing-prompt comparison: choose REVISE versus SUBMIT,
then replay the sealed independent revision. E42 found that factor positive:
generic routing solved 46/60 and explicit routing 51/60, with five routing-only
and zero generic-only cases. Explicit routing preserved all 43 baseline
successes and realized 8/12 available recoveries, four below the 55/60 replay
oracle. The next critical experiment is a live shared-prefix evaluation of the
combined independent worker and explicit routing prompts on a fresh harder
verifier-covered source. Coordinator scale follows as a separate factor. No
usable v2 or permissive-workspace training collection or checkpoint exists.

## 2. Hard rules (violating these has bitten us)

- NEVER write the em-dash glyph in any file: a global hook rejects the
  write. Use commas/colons/parens/hyphens.
- `git add` explicit file paths only; never `-A`/`-u`/`.` (the working
  tree carries unrelated modified slide decks and an old spec edit that
  must not be swept into commits).
- The reward invariant above is load-bearing. Aesthetic signals may
  enter only as search priors, never as Bellman targets.
- Controlled-experiment discipline: one manipulation per experiment,
  outcome measure and its reading pre-registered in the ledger BEFORE
  the run, all trajectories and raw replies logged. 20-episode probes
  cannot distinguish deltas under ~15 points; replicate before
  concluding.
- Lab-notebook vs paper: debugging arcs and bug-fix war stories stay in
  the ledger; outward artifacts (paper, deck, emails) carry only
  forward-facing claims. The user feels strongly about this.
- Paper STUB discipline: `\stub{...}` marks unfinished content; never
  present stubbed content as complete; never resurrect archived pilot
  numbers. The GSM8K pilot (E1) is QUARANTINED (see 4.2).
- Style rules for the paper: define every symbol at first use; expanded
  sums before expectation notation; hand-computable example before
  general algorithm.

## 3. Infrastructure facts

- OpenAI: gpt-4.1-mini was the workhorse teacher. Experiment-specific model,
  transport, token, and price settings are retained in the ledger and runs.
  Account balances, credential locations, and current access are not part of
  the public reproduction contract.
- Hugging Face: historical inference cost estimates follow.
  E23-E29 used approximately $0.0743493 of reported HF generator inference in
  total, including the invalid E28 run. E31-E32 added approximately $0.0377916,
  including the stopped E31 fixed-arm partial, for approximately $0.1121409
  total across E23-E32. E33 added approximately $0.0249009, for approximately
  $0.1370418 total across E23-E33. E34 added approximately $0.0313395, for
  approximately $0.1683813 total across E23-E34. E35 added approximately
  $0.1297041 in HF worker spend, for approximately $0.2980854 across E23-E35,
  plus approximately $0.1078304 in OpenAI teacher spend. E36 added
  approximately $0.0396828 in HF worker spend, for approximately $0.3377682
  across E23-E36. E37 added approximately $0.0271412 in OpenAI teacher spend.
  Its Featherless worker usage was 41,390 prompt plus 24,676 completion tokens,
  and E38 added 35,474 prompt plus 17,352 completion tokens on the same route.
  E39 added 16,771 prompt plus 9,523 completion tokens on Featherless and
  approximately $0.0179728 in OpenAI coordinator spend. E40 added 47,002
  prompt plus 7,848 completion tokens on Featherless; the route still exposed
  no price.
  E41 added 96,164 prompt plus 16,626 completion tokens on Featherless, again
  without exposed router pricing.
  E42 added approximately $0.0295712 in OpenAI coordinator spend and no HF
  usage.
  Current HF metadata exposes no price for that route, so do not add an
  invented HF estimate for either run. The priced totals are provider-rate
  estimates, not billing receipts. New hosted experiments require a separate
  registered comparison and spending authorization.
- Local Ollama (this machine, CPU): `qwen2.5:1.5b-instruct` and
  `qwen2.5:3b-instruct` pulled; OpenAI-compatible endpoint at
  `http://localhost:11434/v1` (pass any non-empty dummy key env var).
  ~40 min per 20-episode probe; ~2 h per 240-problem baseline.
- Historical local GPU serving was optional. No local machine address or
  running model service is required for the archive's supported reproduction.
- Long CPU jobs: launch with `setsid nohup ... & disown` (foreground
  shells get killed at ~10 min in the previous harness).

## 4. Results (ledger entries E1-E42; run files in experiments/runs/)

### 4.1 Direct-prompt baselines (experiments/run_direct_baseline.py)

| model | GSM8K-500 | casework-v1 (240) | f7 | f8 |
|---|---|---|---|---|
| gpt-4.1-mini | 94.4% | 100% | 98% | 100% |
| Qwen2.5-1.5B (local) | - | 39.2% | 5% | 5% |
| Qwen2.5-3B (local) | - | 68.8% | - | - |

Qwen-1.5B per family: f3 75, f1a 40, f1b 35, f2a 31, f2b 15 percent;
dead-case problems harder (33% vs 43%); ~14% unusable replies.

**The decisive comparison**: trained controller + frozen Qwen generator
must beat that generator's own direct number (39.2% for 1.5B, 68.8% for
3B) on casework-v1. That is the success criterion of the whole project.

### 4.2 Harness (LLM-as-controller, Rung 1) results

- mini as BOTH teacher and generator on casework-v1 probes: plateau
  **20-25%** (best 5/20). Teacher-side remedies exhausted, all null
  under pre-registration: prompt engineering (E6-E7), a 5x-price teacher
  (E8, E9b), completion budget 64->192 (E11), substitution-verified
  solve-claims (E13; built and tested but OFF by default,
  `--solve-claims`).
- Structural verbs fire iff the distribution demands them: 0.5% of
  decisions on GSM8K vs 12-23% on casework, SPLIT in 20/20 episodes
  (E3). This is why the casework dataset exists.
- GSM8K 500-episode run (E1, 70.8%): **QUARANTINED**. Collected under a
  superseded flat state rendering; the numbers characterize the old
  interface and the trajectories are excluded from BC (not
  re-renderable). GSM8K/MATH re-enter later as eval targets only.
- The historical v1 observation interface used by these runs: tree render
  with parent= tags, commit-readiness
  annotation "(evidence: node N verified)", verify-failure markers,
  rejection feedback to the controller, TURN k/N budget.

### 4.3 Design-true regime (API teacher + local Qwen generator): HALTED

Four probes, all 0/20, each changing one diagnosed cause:
- E14: strict wire schemas masked 55-63% of typed-verb output ->
  loosened schemas + apply-time coercion (committed, tested).
- E15: coercion worked (SPLIT mask 63%->8%) but 1.5B CONTENT is junk.
- E16: 3B improved per-case content and reduced teacher fallbacks to 8%,
  but the later transition audit disproved the claimed root-aggregation
  choke. Only 3/200 COMPUTEs targeted root (2 masked); 197 targeted
  non-root nodes (161 masked). Root COMMIT was legal in 0/20 episodes,
  and both primary children were solved in only 2/20.
- E17: duplicate-aware COMPUTE suffix (lists existing facts, "must be
  different") did not improve the 83% COMPUTE mask. Only 5 COMPUTEs
  targeted root (3 masked), so this also did not isolate root retry
  handling.

The defensible stop-rule verdict is that the v1 1.5B-3B
generator/controller protocol failed at several stages, not at one last
aggregation choke. E21 later showed that scaling the hosted generator to
7B-32B did not make the unchanged E19 controller competitive with direct
prompting. Do not resume this v1 path with a larger collection before the
typed claim contract is implemented.

### 4.4 Datasets

- `datasets/casework-v1/`: 240 train / 60 test, five families, balanced,
  content-hash ids, per-record provenance (generator version, git SHA,
  sympy version, seed), dataset card (HF-style + datasheet), sha256
  checksums, CC-BY-4.0. Labels verified by construction AND independent
  re-derivation (solveset or exhaustive enumeration).
- `experiments/generate_casework.py`: 9 families total (f4, f6, f7, f8
  calibrated but held out of v1), `--scale` knob, `--exclude`
  cross-split guard. casework-v2 = reweighting existing families once
  the target generator is settled; no new generator work needed.

### 4.5 Phase B results (E18-E22)

- E18 implemented the versioned converter and true factored verb+pointer+Q
  heads. Hashed features reached **139/225 (61.8%)** exact held-out action
  match (12.0% before training; 3.6% weak menu-frequency baseline).
- E19 changed only the encoder to frozen Nomic embeddings and reached
  **144/225 (64.0%)**.
  Checkpoint: `experiments/runs/2026-07-09-E19-phase-b-bc-nomic-seed0.pt`.
- The 07-13 audit added the correct content-free comparator: exact legal-menu
  signature lookup trained only on positive rows. It reached **146/225
  (64.89%)** overall and 34/42 on positive holdout decisions. It beats E18
  and E19 overall; 167/225 holdout menus were seen and lookup was correct on
  143/167. E18/E19 therefore demonstrate repeated structural/protocol phase
  imitation, not semantic-text generalization. E19 also skipped a required
  seed replicate before E20.
- E22 held all E19 settings fixed and replaced every 768-dimensional state
  and node text embedding with zeros. It reached **111/225 (49.33%)**. Verb
  accuracy stayed 73.33%, while pointer accuracy conditional on the teacher
  verb fell from E19's 80.44% to 54.67%. Text carries predictive target
  signal, but E19 still loses to the content-free lookup, and the historical
  render mixes semantics with ids and protocol annotations. E22 is not
  evidence of semantic generalization.
- E20 tested live B=0 on 20 held-out casework problems. Qwen-1.5B direct
  solved 8/20; the E19 controller plus the same Qwen solved 0/20. Masking was
  only 5.3%, but the controller chose VERIFY 186 times; only one of 169
  attempts with a following observation verified, and 157 repeated an
  episode-target VERIFY. Root COMMIT was never legal. Mini-generated BC data
  did not transfer to Qwen-generated states. The pre-registered stop fired.
- E21 tested hosted Qwen2.5 7B, 14B, and 32B on the same five held-out cases.
  Direct/controller wins were 5/1, 5/0, and 4/0; masking was 42.9%, 30.6%,
  and 43.8%. The reported 7, 8, and 10 verifier-successful facts include
  goal-irrelevant truths, so they do not show that capacity improved useful
  content. The direct arm used temperature 0.0 while controller generation
  inherited 1.0. Malformed provider replies were logged only as exception
  summaries, not raw responses, and their usage may be missing from the
  $0.012508 valid-call estimate. Do not run the proposed 7B confirmation on
  E19: the comparison and cost accounting are confounded.
- E20 and E21 acted from the BC verb/pointer heads only. The fitted Q head did
  not select actions, and no off-policy Q-learning update had run. These are
  BC-transfer tests, not tests of the planned RL policy.

### 4.6 V2 integration and generator gates (E23-E29)

- E23 passed the five-problem hosted integration gate: the production
  collector crossed both model transports, published its artifacts only after
  strict replay, and retained all 66 decisions. It is an integration artifact,
  not training data.
- E24 changed only the generator lens contract to
  `typed-claim-lens-v2.1` over those 66 sealed pre-states and actions. It made
  66 generator calls and zero controller calls. Typed-unusable output fell to
  0/66 and scalar usability reached 33/33, so the first two preregistered gates
  passed. Applied DECOMPOSE drafts remained 0/33, so the decomposition gate
  failed.
- E25 corrected the symbolic rescore and confirmed 0/33 applied DECOMPOSE
  drafts. E26 added bounded semantic feedback and reached 5/33 typed drafts
  but 0/33 applied. E27 aligned the feedback input key with the required
  `domains` output key and reached 32/33 typed full-domain drafts but 0/33
  applied, so full-domain authoring was retired.
- E28 appeared to clear a separator-only boundary probe, but its frozen
  string-only contract was violated by hidden number-to-string schema
  coercion. Its coerced semantic rows, primary-gate counts, and empirical
  conclusions are quarantined. Its raw attempts and strict no-coercion audit
  remain valid protocol diagnostics and are explicitly bound into E29.
- E29 reran the separator-only representation with exact string typing and
  the frozen one-retry policy. Only 8/33 logical cases ended with typed
  payloads, and all 25 schema-triggered retries exhausted. The eight accepted
  drafts applied but covered only one of five problems, so all three gates
  failed. E29 created no controller decision, reward, trajectory, closure, or
  training row and does not authorize boundary integration or collection.
- The current string-valued boundary branch is halted. The next candidate is
  a separately versioned JSON-number boundary representation with exact
  decimal-lexeme parsing in trusted code. It remains a design proposal until
  its contract, proof obligations, tests, and preregistration are frozen. No
  usable v2 training collection or checkpoint has been produced.

### 4.7 Permissive workspace mechanism screen (E30-E38)

- E30 was superseded before execution when the user redirected the work from
  scaling the strict v2 teacher to testing a clean permissive workspace. It
  made no hosted call and has no artifact.
- The new workspace stores natural-language Goals and unverified Candidates.
  Its learned actions are DECOMPOSE, ANSWER, SYNTHESIZE, REVISE, and SUBMIT.
  The controller sees the full canonical state; the worker sees an action-
  specific hybrid lens. Only root SUBMIT calls the symbolic checker and can
  produce reward. Initial ANSWER(root) is the exact raw direct request.
- E31's direct Qwen2.5-7B arm completed at 17/20 on GSM8K rows 100-119. The
  first fixed arm was stopped after one episode because useful SYNTHESIZE
  replies were wrapped under `candidates`, rejected by the candidate wire, and
  repeated to the horizon. Its `.partial` run and diagnostics are retained and
  are not a performance result.
- E32 widened only non-direct candidate admission and bounded failed synthesis.
  The corrected fixed arm completed all 20 problems at 9/20 versus the sealed
  direct 17/20. Paired counts were eight both, nine direct-only, one fixed-only,
  and two neither. Removing the two DECOMPOSE generation-error episodes leaves
  direct ahead 15/18 to 9/18. The fixed scaffold therefore failed its
  preregistered performance gate despite substantially more inference.
- The post-hoc trace audit found the next state/lens issue: DECOMPOSE emitted
  ordered dependent tasks, while child ANSWER omitted completed earlier sibling
  work. Later children recomputed missing values, conflicted, and gave
  SYNTHESIZE bad inputs.
- E33 added only completed preceding-sibling Goals and Candidates to later
  child ANSWER lenses. It completed at 12/20, versus E32 fixed at 9/20 and E31
  direct at 17/20. The registered lens threshold passed by exactly three, but
  the direct-performance gate failed by five. Paired E33/E32 counts were eight
  both, four E33-only, one E32-only, and seven neither. Paired E33/direct counts
  were 11 both, one E33-only, six direct-only, and two neither.
- E33's integration boundary passed: all 20 episodes submitted; all 41 eligible
  later-child requests contained the causal prefix; no future sibling content,
  hidden label, or teacher call appeared; and no failed SYNTHESIZE repeated.
  Two DECOMPOSE calls exhausted their parse retry. The run retained 99 logical
  calls, 101 attempts, 69,681 prompt tokens, and 13,322 completion tokens.
- The causal audit is limiting: all DECOMPOSE requests were byte-identical to
  E32, but four returned different decompositions. One more identical
  first-child request returned a different result. Those five upstream changes
  account for the raw net gain. On the 13 clean common-prefix episodes, E33 won
  one and lost one. Treat the lens as plausible, not established. Row 100 shows
  the intended dependency benefit; row 102 shows a new final-format failure,
  `25ml` instead of accepted `25`.
- E34 removed that ambiguity by replaying each successful E32 DECOMPOSE plus
  first-child prefix once, then branching the two lens modes from the exact same
  state. The 18 paired cases produced 12/18 for preceding-siblings and 9/18 for
  target-only: eight both, four preceding-only, one target-only, and five
  neither. These were the same five discordant ids observed in E33.
- E34's integration boundary passed with 18 shared prefixes, 36 terminal
  branches, 120 accepted logical calls, identical within-case action traces,
  balanced provider position, and no generation retries. Every corresponding
  round bound the same worker model, temperature, token limit, and effective
  seed. All 42 eligible preceding-sibling ANSWER requests contained the causal
  context and all 42 target-only requests omitted it.
- The registered `P - T >= 3` gate passed exactly. All five discordances fell
  in one initial-order stratum and the exact two-sided discordant-pair test is
  0.375, so E34 supports a causal mechanism on this reused development screen,
  not a population-effect claim. Retain `preceding-siblings-v1`; evaluate the
  teacher on a larger fresh harder slice with contemporaneous controls.
- E35 ran contemporaneous direct, fixed preceding-sibling, and GPT-4.1-mini
  teacher policies on all 60 project-held-out casework-v1 test problems. Direct
  solved 59/60, fixed solved 27/60, and teacher solved 31/60. Teacher versus
  direct was 31 both, zero teacher-only, 28 direct-only, and one neither. This
  is a decisive negative teacher result, not policy equivalence under a ceiling.
  Teacher versus fixed was 24 both, seven teacher-only, three fixed-only, and
  26 neither, which missed the registered net-five and probability gates.
- The teacher chose initial root ANSWER 12 times and solved all 12. It chose
  initial root DECOMPOSE 48 times and solved 19, while direct solved 47 of those
  same 48. The controller lacked worker-specific transition calibration before
  choosing whether to obtain the recoverable direct Candidate.
- E35 also exposed three separable interface defects. Eleven rejected teacher
  submissions contained the exact correct scalar only as `x = n`; the legal
  menu grew to 27 because stale and already-answered work remained actionable;
  and six parsed teacher replies formed illegal verb-target cross-pairings.
  Root result formatting, direct-prefix state initialization, active-action
  masking, and indexed teacher choices must not be bundled into one experiment.
- The baseline-anchored workspace v2 now implements those four factors as
  explicit modes. A deterministic exact direct transition precedes policy
  decisions and is excluded from behavior-cloning eligibility. Root-only
  verifier-ready instructions remain permissive content admission. The active
  menu keeps the protected direct Candidate submittable, and the observation
  identifies it directly. Indexed teacher output selects a legal menu entry;
  malformed output falls back to direct ANSWER or protected direct SUBMIT.
  `redesign.py` records decision source and role-bound call provenance, and
  `experiments/run_workspace_redesign.py` requires prepared digests before
  hosted execution. No hosted inference or performance result exists for this
  implementation. Its current collector is single-arm, so a causal shared-
  prefix comparison still needs a paired branch boundary.
- E36 isolated only the root result instruction from 59 exact E35
  pre-SYNTHESIZE states. Verifier-ready root prompting solved 49/59 versus
  27/59 for the byte-replayed legacy prompt: 27 both, 22 ready-only, zero
  legacy-only, and 10 neither. The one-sided exact probability is `1 / 2^22`.
  Integration passed with 118 first-attempt hosted generations, 118 local
  symbolic submissions, zero generation errors, matched within-pair seeds,
  and the registered instruction as the only request delta.
- Of the 22 E36 recoveries, 10 legacy answers were exact correct assignment
  wrappers and nine more contained the exact expected scalar inside prose or
  alternatives; three changed result selection. The legacy branch reproduced
  E35 reward on all 59 selected states. Retain
  `verifier-ready-root-answer-v1`, but do not interpret E36 as evidence that a
  teacher or decomposition beats direct prompting. E36 contains no BC rows.
- E37 added the exact shared-prefix paired boundary and used hosted
  Qwen2.5-3B on all 60 reused casework test problems. Shared direct submission
  and GPT-4.1-mini teacher continuation both solved 43/60. Paired cells were 43
  both, zero teacher-only, zero direct-only, and 17 neither. This is
  inconclusive under the registered gate, not a positive result and not a
  ceiling artifact.
- The baseline anchor did its safety job: every teacher path ultimately
  submitted protected Candidate 1, so it broke zero correct direct answers.
  It also recovered zero failures. The teacher requested work on 11 cases,
  including seven of the 17 direct failures but only four of the 43 direct
  successes. That post-hoc routing association suggests the direct Candidate
  exposes useful uncertainty, but it is not an efficacy claim.
- The next failure is sharply localized. E37 made 14 REVISE calls: six copied
  a lens-shaped object that the open Candidate schema admitted as answer text,
  and eight exhausted parsing after long echoes. Zero produced an
  answer-bearing revision. Across all 24 continuation generations, only eight
  returned the requested container, eight malformed echoes were admitted, and
  eight exhausted. The teacher then submitted the baseline in every case.
  E37 is evaluation-only and supplies no BC rows.
- E38 replayed those exact 14 teacher-selected pre-REVISE states with the same
  Qwen2.5-3B worker. The registered answer-container contract produced 14/14
  usable Candidates versus 0/14 for the legacy open-object control: all 14
  pairs were treatment-only, with one-sided exact probability `1 / 2^14`.
  Treatment calls all completed once. The legacy arm admitted five lens-shaped
  objects without answers and exhausted nine calls after a retry.
- Symbolic correctness remains a separate, narrower result. Across 10 unique
  root episodes, selected source Candidates were correct on three and new
  revisions on four: three both, one revision-only, zero source-only, and six
  neither. Thus the contract creates selectable work and repaired one selected
  failure without regressing the three selected correct roots, but E38 had no
  controller or SUBMIT action and is not an end-to-end accuracy result. It is
  evaluation-only and supplies no BC rows.
- E39 retained the answer-container contract in a complete shared-prefix run.
  Direct submission and teacher continuation again solved 43/60, with 43 both
  and 17 neither, so the preregistered efficacy screen was inconclusive. All
  12 revisions were valid. In that subset the baseline and revision were both
  correct on four cases, revision-only on three, baseline-only on zero, and
  neither on five. Both submit targets were legal after every revision, but
  GPT-4.1-mini submitted protected Candidate 1 in all 12 cases. Selecting the
  correct available Candidate could have raised this exact run to 46/60. This
  localizes the immediate bottleneck to coordinator selection rather than
  answer-container generation. E39 is evaluation-only and supplies no BC
  rows.
- E40 applied one fixed answer-container REVISE to every exact E39 post-prefix
  State. All 60 revisions were usable. Baseline and revision were both correct
  on 43 cases, revision-only on six, baseline-only on zero, and neither on 11.
  Thus latest revision and the available-Candidate oracle were 49/60 versus
  43/60 for protected baseline. The answer changed on 11 cases, with six
  wrong-to-correct and zero correct-to-wrong changes. This failed the
  preregistered bidirectional-corpus gate because it supplied no preservation
  decision. E40 has no coordinator or SUBMIT and supplies no BC rows.
- E41 paired the retained revision prompt with an independent-solution prompt
  on those same 60 exact States. Retained was correct on 49/60 and independent
  on 55/60: 49 both, six independent-only, zero retained-only, and five
  neither. The one-sided exact probability is `1 / 2^6`. Against protected
  baseline, independent revision had 43 both, 12 revision-only, zero baseline-
  only, and five neither. Retain the independent prompt for a routing screen,
  but do not claim terminal reward: E41 has no coordinator or SUBMIT and no BC
  rows.
- E42 paired the generic GPT-4.1-mini planner prompt with an explicit initial
  routing prompt while replaying exact E41 revisions. Generic solved 46/60 and
  explicit routing 51/60: 46 both, five routing-only, zero generic-only, and
  nine neither. The one-sided exact probability is `1 / 2^5`. Routing selected
  REVISE 19 times versus eight for generic, preserved all 43 baseline
  successes, and realized 8/12 revision-only recoveries. It remains four below
  the 55/60 replay oracle. E42 has verifier-scored terminal SUBMIT but replays
  worker output, so it is a coordinator prompt mechanism result rather than a
  live end-to-end teacher advantage. It supplies no BC rows.

## 5. Code map (src/verifier_anchored_rl/)

- `reductio/`: vendored immutable forest core + pure verb algebra.
  Invariants: frozen nodes, no reward/value fields, stdlib-only core,
  structure never gates (connective is metadata), commit requires a
  verifier-vouched witness.
- `reasoning_v2/model.py`, `actions.py`, and `verifier.py`: parallel immutable
  typed graph and total action algebra. Result, Solution, Derivation,
  Aggregate, and Refutation payloads are authoritative; validity and discharge
  receipts must use one verifier contract. Generated structures pass through
  a required GenerationContract before storage.
- `reasoning_v2/math_adapter.py`: fail-closed casework obligation builder,
  symbolic canonicalizer, exact decomposition coverage, complete finite
  solution computation, objective Result verification, and bounded validity
  and discharge reason-code sets. It never reads answer labels.
- `reasoning_v2/casework_source.py`, `collection.py`, and
  `experiments/collect_reasoning_v2.py`: immutable label-free problem-source
  projection, ordered source digest, zero-network contract preparation,
  preregistration and expected-digest authorization, exclusive `.partial`
  output, strict replay before publication, and no-clobber finalization. E23
  exercised this boundary as a five-problem integration gate; its artifact is
  excluded from training.
- `reasoning_v2/menu.py` and `observation.py`: `claim-closure-v2` exposes only
  DECOMPOSE, PROPOSE_RESULT, PROPOSE_SOLUTION, and PROPOSE_REFUTATION to the
  learned policy. Closure-pending states have no learned menu. Canonical active
  and compact records retain full validity/discharge receipts, and
  environment-bound bytes require both pinned contracts.
- `reasoning_v2/closure.py` and `transition.py`: deterministic VERIFY, COMMIT,
  and exact ready-AGGREGATE closure; ordered Claim-linked events; one learned
  turn per macro step; final-turn closure before horizon expiry; +1/0
  `symbolic-only-v1` reward; and contract-aware replay of every applied effect
  before its graph, receipts, or reward are trusted.
- `reasoning_v2/lenses.py`, `controller.py`, `runtime.py`, `contracts.py`, and
  `records.py`: exact typed generator and teacher requests, immutable semantic
  and model-distribution bindings, runtime-owned legal fallback, canonical run
  records, and decision provenance bound to episode and observation digest.
  `typed-claim-lens-v2.1` is current; `typed-claim-lens-v2` remains byte-exact
  for legacy replay and reconstructs all 66 sealed E23 generator requests.
  The current generator and teacher wire IDs bind strict validation; their
  historical IDs bind legacy validation. Runtime and diagnostic replay derive
  the mode from those role-local IDs and reject binding drift before HTTP.
- `reasoning_v2/semantic_repair.py`, `key_aligned_repair.py`,
  `boundary_probe.py`, and `boundary_exact_probe.py`: generator-only
  diagnostics for E26-E29. The boundary modules are experiment probes, not
  active runtime components, and no separator-only lens is integrated.
- `reasoning_v2/diagnostics.py` and `replay.py`: append-only raw HTTP attempt
  capture for both model roles, one-use composite decision receipts, exact
  offline request/retry/action/effect reconstruction, trusted problem-source
  resolution, and complete interleaved episode replay. Hosted rows require a
  linked receipt even when they are fallback or non-BC.
- `reasoning_v2/features.py`, `checkpoint.py`, and `learned_controller.py`:
  owner-sensitive typed feature contexts, exact encoder and model manifests,
  value-sensitive Torch state hashing, separate teacher-training and
  checkpoint-backed live collection validation, four verb/pointer/Q heads,
  and global joint legal-pair decoding.
- `reasoning_v2/dataset.py`, `baselines.py`, and `training.py`: conversion only
  from sealed strict replay, immutable current and next structured features,
  whole-problem grouped splits, a provenance-only exact legal-menu comparator,
  and deterministic CPU behavior-cloning plus Monte Carlo Q warm-start. BC
  consumes every provenance-eligible teacher row without success filtering;
  Q regression consumes every training transition. The focused v2 suite has
  505 tests.
- `workspace_v1/`: the post-E29 permissive reasoning workspace. `model.py`
  defines immutable Goals, Candidates, action history, and the five-action
  menu. `lenses.py` owns exact direct recoverability plus worker and teacher
  projections. `runtime.py` admits bounded unverified model content and keeps
  expected answers outside state; SUBMIT alone calls the symbolic checker.
  `policies.py` supplies direct and fixed controls, `controller.py` supplies the
  full-state LLM teacher, and `experiment.py` plus `records.py` provide bound
  label-free sources, hidden-label digests, raw attempt logging, partial-safe
  publication, and no-clobber artifacts. `experiments/run_workspace_v1.py` is
  the legacy zero-network preparation and hosted execution command.
  `redesign.py` adds the baseline-anchored run contract, prefix-versus-policy
  provenance, and behavior-cloning eligibility. `model.py`, `lenses.py`,
  `runtime.py`, and `controller.py` also contain independently switchable
  active-frontier, verifier-ready root, and indexed-choice modes while their
  defaults preserve the E31-E35 interface.
  `experiments/run_workspace_redesign.py` is the digest-gated redesigned arm
  runner. `paired_root_result.py` reconstructs sealed pre-finalization states
  and executes the exact shared-state root prompt comparison;
  `experiments/run_workspace_paired_root_result.py` is its digest-gated runner.
  `paired_direct_teacher.py` generates one exact direct Candidate, submits it
  locally for the control, and continues the immutable shared state under the
  teacher; `experiments/run_workspace_paired_direct_teacher.py` is its
  digest-gated runner. `paired_revise.py` reconstructs the exact E37
  teacher-selected REVISE states and pairs the legacy and answer-container
  contracts; `experiments/run_workspace_paired_revise.py` is its digest-gated
  runner. The workspace-focused suite has 50 tests.
- `deliberation_mdp.py`: the v1 context MDP plus independently switchable
  v1.1 guards. Rejected deterministic VERIFY actions and descendants of
  solved/abandoned goals are masked by default; trusted givens are
  noncommittable and generation temperature is explicit. Legacy switches
  reproduce archived behavior. The older SPLIT,
  duplicate-COMPUTE, coercion, and solve-claim controls remain.
- `lenses.py`: generator and teacher renders. Compacted solved goals now
  include their exact committed outcome by default, with a legacy switch.
- `controller_observation.py`: canonical graph-derived v1.1 repair,
  `controller-observation-v1.1`. Every legal target has a complete structured
  record with owner path, provenance, status, exact legal verbs, premises,
  verifier attempts, and committed outcome. Model-authored text is never
  parsed for controller status.
- `DeliberationState.previous_outcome`: immutable typed action outcome,
  populated by every transition. Rejection state is therefore visible to
  teacher, replay, learned policy, and search without runner-local feedback.
- `llm_controller.py`: teacher decisions and trajectory logs consume the
  canonical observation, including a validated prior action outcome, while
  preserving the existing prompt and fallback controls.
- `llm_client.py`: OpenAI-compatible; 429/5xx backoff; parse-retry echoes the
  failed reply. Successful HTTP generation attempts are counted before schema
  parsing, and the final malformed raw reply is retained on failure. Schema
  behavior has separately versioned legacy-coercion and nonmutating
  strict-type modes. E29 explicitly binds strict mode. Canonical v2 now binds
  strict mode independently through `typed-policy-generation-v2.1` and
  `controller-choice-wire-v2.1`; the prior IDs select legacy behavior only for
  exact replay. The low-level client default remains legacy for compatibility,
  but canonical live and diagnostic paths always pass the bound mode.
- `verifier_adapter.py`: MathVerifier decide + solve_check
  (substitution); check_answer (terminal: RHS fallback, tautology
  rejection).
- General Torch components (all tested): `heads.py`,
  `embedder.py`, `planner_actor.py`, `shared_forward.py`, `train.py`,
  `replay.py`.
- `phase_b.py`: versioned replay conversion. Structured v1.1 records are
  accepted only when their observation version, exact legal menu, bounded
  action outcome, and source MDP contract match; historical tree-render-v1
  conversion remains reproducible under its named legacy reward contract.
- `phase_b_features.py`: hashed, Nomic, and zero-text feature paths. Structured
  node features include ownership/provenance; the zero path supports E22.
- `phase_b_heads.py` and `phase_b_policy.py`: factored heads with normalized
  joint legal-pair probabilities. New greedy deployment globally maximizes
  the pair probability; legacy checkpoints can select sequential decoding.
  New checkpoints carry a hashed manifest of observation, outcome, action,
  transition, reward, feature, encoder, target, and decoding contracts. Live
  loading rejects semantic drift; old unbound checkpoints and generator
  distribution shifts require separate explicit declarations.
- `phase_b_baselines.py`: exact legal-menu signature lookup used by the E22
  audit and emitted by the BC runner.
- Legacy v1/v1.1 experiment entrypoints: `convert_phase_b.py`,
  `embed_phase_b.py`, `run_phase_b_bc.py`, and `run_phase_b_controller.py`.
  The BC and live runners record observation schema and decision rule; the
  live runner also exposes every v1.1 switch and generation temperature.
- V2 collection entrypoint: `collect_reasoning_v2.py`. Preparation prints the
  exact source, environment, and collection digests without secrets or HTTP;
  hosted execution requires a preregistration id and all three expected
  digests.

The v1.1 repairs still do not solve hidden COMMIT witness selection or
distinguish symbolic validity from owner-goal discharge. Use them only for
reproduction. The v2 implementation boundary is complete through the learned
controller, exact baseline, and deterministic warm-start. E23 validates the
hosted integration boundary. E24-E27 and E29 reject the tested decomposition
representations; E28's coerced semantic result is quarantined. None is a
usable training collection. Execution of training and evaluation remain
empirical work, no v2 checkpoint exists, and off-policy Bellman learning is not
implemented yet.

Runner: `experiments/run_llm_controller.py`. Flags: `--backend
{echo,http,openai}`; mixed `--controller-backend/-model/-base-url`;
`--controller-prompt {default,structured}`; `--controller-max-tokens`;
`--solve-claims {on,off}`; `--start-index` (resume, keeps true episode
ids); `--max-problems`; `--workers`; `--token-budget`; `--seed`.
Run with PYTHONPATH=src (or `pip install -e .`).

## 6. Legacy behavior-cloning data

USABLE for reproducing v1 E18-E22 only (tree-render interface,
mini-as-both, casework), in experiments/runs/:
`2026-07-02-rung1-casework-probe20-v6-treeview.jsonl`,
`2026-07-03-rung1-casework-probe20-v7.jsonl`,
`2026-07-03-rung1-casework-E11-gpt41-maxtok192.jsonl`,
`2026-07-03-rung1-casework-E13-solveclaims-on.jsonl`: 80 episodes,
1,067 decisions, 17 positive-return episodes. The design-true runs (E14-E17)
add ~1,250 decisions with zero wins (useful as negatives and
masked-action labels). QUARANTINED: everything GSM8K (flat render) and
casework probes 1-5 (pre-tree-render).

Do not parse these renders into the v2 schema or grow mini-as-both data merely
to improve live Qwen behavior. E20 shows that this does not repair the
generator-state mismatch. The schema break requires fresh structured data.

## 7. Publication artifacts

- Paper `paper/main.tex` (19 pp, builds with latexmk): full scaffold +
  written Preliminary Findings section; TikZ figures (simple
  architecture, episode timeline, training ladder in the body; detailed
  architecture in Appendix A). A multi-agent review was applied (2
  Criticals + ~16 Majors fixed); reports in
  `.papermill/reviews/2026-07-03/`. Casework is described as "prepared
  for public release" (the repo is private; do not claim published).
- Deck `docs/slides/2026-07-02-design-overview.pdf` (11 pp): reviewed,
  every number verified against run logs, GSM8K removed, no war
  stories; sent (or ready to send) to coauthors.
- Experiment ledger: `docs/superpowers/research/EXPERIMENTS.md`.
- Specs under `docs/superpowers/specs/`: casework generator, solve-claim
  COMPUTE, deliberation-MDP integration, and the 07-13 state/action contract
  v2 implemented contract.

## 8. Next steps, in priority order

E21 and E30 follow-ups are superseded. E31-E42 are sealed. Do not rerun them or
train from their engineering traces. E36 retains the verifier-ready root
contract, and E38 retains `workspace-revise-answer-container-v1`. E37 proves
that exact baseline anchoring can prevent regression with headroom, but it
found no accuracy gain and produces no behavior-cloning data.

1. Run a separately preregistered live shared-prefix comparison of exact direct
   submission against the retained system: independent-solution REVISE prompt,
   explicit GPT-4.1-mini routing prompt, and fixed latest-Candidate SUBMIT after
   REVISE. Prefer a fresh harder verifier-covered source and keep state, action,
   lens, schema, worker, verifier, seeds, and horizon fixed.
2. Compare coordinator model capacity separately after that live prompt-system
   result, without changing worker, prompt, state, action, lens, or source.
   Do not bundle a larger teacher with a prompt, state, action, lens, worker,
   or horizon change.
3. Only after a repaired teacher beats exact direct submission on a fresh
   harder verifier-covered source, collect fresh
   behavior-cloning trajectories, run the implemented grouped split and
   content-free comparator, and warm-start the controller heads.
4. Then run off-policy Q-learning with symbolic terminal reward as the only
   Bellman signal and make learned Q control live action selection. MCTS remains
   a strictly additive accelerator after the RL spine works.

## 9. Common commands

```bash
uv run pytest -q                            # current full suite
uv run python experiments/run_workspace_v1.py --help
uv run python experiments/run_workspace_redesign.py --help
uv run python experiments/run_workspace_paired_root_result.py --help
uv run python experiments/run_workspace_paired_revise.py --help
uv run python experiments/collect_reasoning_v2.py --help
PYTHONPATH=src python3 experiments/run_llm_controller.py --backend echo \
  --problems datasets/casework-v1/train.jsonl --max-problems 3 --out /tmp/smoke.jsonl
PYTHONPATH=src python3 experiments/run_direct_baseline.py \
  --model qwen2.5:3b-instruct --base-url http://localhost:11434/v1 \
  --api-key-env DUMMY --problems datasets/casework-v1/train.jsonl \
  --out /tmp/base.jsonl                     # DUMMY = any set env var
python3 experiments/generate_casework.py --n 40 --seed 0 --families f7 --out /tmp/f7.jsonl
cd paper && latexmk -pdf main.tex           # 19 pp
cd docs/slides && latexmk -pdf 2026-07-02-design-overview.tex
```
