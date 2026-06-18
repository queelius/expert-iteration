# Expert iteration: v2 -> v3 -> v4 on the LLM context MDP

Headline numbers from the first true expert-iteration round on the LLM
context MDP, qwen2.5:3b on a local Ollama GPU box (192.168.0.204).

## Setup

* Same 4 training problems for v2 and v3: `multiply_small`, `gsm8k_toy_1`,
  `algebra_toy_1`, `factor`.
* Same 4 held-out problems for evaluation: `multiply_small_ho`,
  `gsm8k_toy_ho`, `algebra_toy_ho`, `factor_ho`. Structurally parallel
  to the training set but with different surface content.
* Same MCTS budget B=20, c_puct=1.0, gamma=0.99, rollout_depth=0.
* Same LLM (qwen2.5:3b) and embedder (qwen2.5:3b @ 2048-d) for both
  rounds.
* 5 random seeds per problem -> 60 records per dataset.
* TrainableActor head architecture: 2048 -> 512 -> 256 (LayerNorm + GELU
  + Dropout 0) shared encoder, separate 7-way policy and 7-way Q heads.
  1,188,110 parameters.
* Adam, lr=1e-3, full-batch, 200 epochs, q_loss_weight=1.0 for both
  rounds.

The only deliberate change between v2 and v3 is the MCTS prior:

* v2: uniform PUCT prior (1/k over available verbs)
* v3: pi_theta from /tmp/heads_v2.pt as the PUCT prior, with label
  smoothing epsilon=0.02

## 1. Distillation targets sharpened

Empirical pi_MCTS distributions across all 60 records of each dataset.

| dataset                    |   N | mean max | median max | entropy (bits) | frac max>=0.5 |
|----------------------------|----:|---------:|-----------:|---------------:|--------------:|
| v2 (training, uniform)     |  60 |    0.452 |       0.40 |          2.070 |   35% (21/60) |
| v3-train (training, trained) | 60 |  0.549 |       0.70 |          1.855 |   62% (37/60) |
| v3-holdout (held-out, trained) | 60 | 0.558 |     0.70 |          1.833 |   63% (38/60) |

Median max-probability jumped from 0.40 to 0.70 across the round. Held-out
problems under the trained prior produced slightly *sharper* targets than
training problems under the uniform prior, evidence that the prior is doing
real work rather than leaking into the target.

## 2. Cross-seed consistency improved at the right places

Recommended-action distribution across 5 seeds, per (problem, turn).
Selected states:

| state                  | v2 (uniform)                   | v3-train (trained)      |
|------------------------|--------------------------------|-------------------------|
| `multiply_small @ t0`  | CS:3, FH:2                     | **CS:5** (unanimous)    |
| `factor @ t1`          | IV:4, BT:1                     | **IV:5** (unanimous)    |
| `gsm8k_toy_1 @ t0`     | D:3, BT:2                      | D:4, BT:1               |
| `algebra_toy_1 @ t0`   | CS:3, BT:2                     | CS:3, BT:2 (tie kept)   |
| `multiply_small @ t2`  | CA:2, BT:2, IV:1               | **CA:4, FH:1**          |

The algebra tie persists across rounds, which is correct: the model
preserves uncertainty when the MCTS data itself is genuinely split.

## 3. v3 head training converged to a lower floor

After 200 full-batch epochs on each round's 60 records:

| | v2 (trained on v2) | v3 (trained on v3-train) |
|---|---:|---:|
| Final policy cross-entropy | 1.4512 | **1.2934** (-10.9%) |
| Final Q MSE | 0.006829 | **0.004156** (-39.1%) |
| Training-set argmax agreement | 85.0% | **91.7%** |

The drop in the policy-loss floor is exactly what we expected: the
distillation targets in v3-train are sharper (lower target entropy ->
lower cross-entropy floor). The drop in Q MSE indicates lower inter-seed
variance in the chance-node Q estimates, also expected because the
trained prior concentrates visits onto fewer chance nodes per (state,
action) pair.

## 4. Held-out generalization (the headline)

Zero-shot evaluation: run each head on the same 60-record `v3-holdout`
dataset and compare pi_theta(s) to the empirical pi_MCTS(s).

| | v2 heads | v3 heads |
|---|---:|---:|
| Argmax agreement | 56.7% (34/60) | **71.7% (43/60)** |
| Mean L1 to pi_MCTS | 0.540 | **0.314** (-42%) |

Per-problem:

| problem | v2 agree | v3 agree | v2 L1 | v3 L1 |
|---|---:|---:|---:|---:|
| `multiply_small_ho` (9*8)             |  93.3% | **100%** | 0.556 | **0.226** |
| `factor_ho` (x^2+5x+6)                | 100%   |   100%   | 0.306 | **0.167** |
| `gsm8k_toy_ho` (Bob/marbles word prob.) |  20.0% |    26.7% | 0.476 |     0.509 |
| `algebra_toy_ho` (3y-7=8)             |  13.3% | **60.0%** | 0.823 | **0.356** |

The largest absolute transfer gain is on algebra (+47 pp). The word
problem family (`gsm8k`) is the hardest to transfer: argmax improved
modestly but L1 ticked up, meaning v3 became more confident but slightly
mis-distributed there. Suggested follow-up: add more gsm8k variants to
the training set in v4.

## 5. Cost story

v2 wall-clock: ~13 min for 60 records (running alone on the GPU).
v3 wall-clock: ~82 min for 60 records, *but* sharing the GPU with
v3-holdout running in parallel. The per-process slowdown is from shared
GPU contention, not from the trained prior.

Per-MCTS-search times in v3 (trained prior, single problem):

* turn 0: 55-110s (still searches broadly because no action is yet
  forced)
* turn 1: 25-30s (often committed to one path)
* turn 2: 25-30s (frequently terminal-bound)

vs. v2 (uniform prior):

* turn 0: 50-75s
* turn 1: 60-80s
* turn 2: 70-90s

Net per-record cost in v3 alone would be ~40% lower than v2 because the
trained prior routes search away from LLM-call-heavy branches.

## Conclusion

A single round of expert iteration on the LLM context MDP closed the
loop:

* Sharper distillation targets (entropy down 0.215 bits, median max
  +0.30).
* Cleaner cross-seed agreement on training problems.
* Lower training-loss floor (CE -11%, MSE -39%) confirming the targets
  themselves carry less noise.
* +15 percentage points in zero-shot held-out argmax agreement (57% ->
  72%).
* 42% reduction in held-out L1 distance.
* Cheaper per-record MCTS (when not GPU-sharing) because the trained
  prior terminates more searches via COMMIT_ANSWER.

All metrics moved in the same direction with one iteration, which is
the framework's central empirical claim.

## Artifacts

All produced 2026-06-09 on this run:

* `/tmp/distillation_v2.jsonl` -- 60 records, uniform prior
* `/tmp/distillation_v3_train.jsonl` -- 60 records, trained prior (v2 heads)
* `/tmp/distillation_v3_holdout.jsonl` -- 60 records, trained prior (v2 heads), held-out problems
* `/tmp/heads_v2.pt`, `/tmp/heads_v3.pt` -- TrainableActor checkpoints (1.19M params)
* `/tmp/heads_v2_eval.json`, `/tmp/heads_v3_eval.json` -- training histories + post-eval
* `/tmp/v2_vs_v3_train.json`, `/tmp/v2heads_on_holdout.json`, `/tmp/v3heads_on_holdout.json` -- metric dumps
* `/tmp/mcts_compare.json` -- single-shot uniform-vs-trained PUCT comparison (4 problems, B=20)


# Round 3: v4 (v3 prior, expanded gsm8k training set)

After v3, the weakest generalization point was `gsm8k_toy_ho` at 27%
argmax agreement. The intuitive intervention: add more gsm8k variants
to the training set. v4 used 7 training problems: the original 4 plus
3 new gsm8k variants (`gsm8k_toy_2,3,4`). MCTS prior = v3 heads.
B = 20, 5 seeds, target 105 records.

Mid-run incident: during gsm8k_toy_3 seed=1 turn=1, the LLM produced a
COMMIT_ANSWER response with `answer = None`, which crashed
`verifier._normalize_answer`. Patched `LLMContextMDP._handle_commit_answer`
to treat missing/non-string answers as a failed commit (terminal failure
reward), added two regression tests, and resumed collection from
checkpoint with `--append`.

## v2 -> v3 -> v4 sharpness

| dataset | N | mean max | median | entropy | frac max>=0.5 |
|---|---:|---:|---:|---:|---:|
| v2 (uniform, 4 problems) | 60 | 0.452 | 0.400 | 2.070 | 35% |
| v3-train (v2 prior, 4 problems) | 60 | 0.549 | 0.700 | 1.855 | 62% |
| v3-holdout | 60 | 0.558 | 0.700 | 1.833 | 63% |
| **v4-train (v3 prior, 7 problems)** | **105** | **0.506** | **0.450** | **1.945** | **49%** |
| v4-holdout (v3 prior) | 60 | 0.585 | 0.700 | 1.762 | 68% |

v4-train is *less sharp* than v3-train. The 3 new gsm8k variants are
problems v3 prior had never seen, so the prior acted nearly uniform on
those states. The flat targets pull the average down.

v4-holdout, where v3 prior had seen structural cousins of the held-out
problems during training, is *sharper* than v3-holdout. The prior
generalizes when there is a structural anchor.

## v4 head training

200 full-batch Adam epochs on 105 records, q_loss_weight=1.0, same
architecture as v2 and v3 (1.19M params, embed_dim=2048, hidden=512,
repr=256).

| | v2 | v3 | v4 |
|---|---:|---:|---:|
| Final policy CE | 1.4512 | **1.2934** | 1.3574 |
| Final Q MSE | 0.006829 | **0.004156** | 0.005477 |
| Training-set argmax % | 85.0% | **91.7%** | 88.6% |

v4 sits between v2 and v3 on every training metric -- consistent with
the lower target sharpness.

## v4 generalization regressed

On the canonical `v3-holdout` dataset (used to score v2, v3, and v4):

| heads | argmax % | mean L1 |
|---|---:|---:|
| v2 | 56.7% | 0.540 |
| **v3** | **71.7%** | **0.314** |
| v4 | 65.0% | 0.407 |

Per-problem breakdown on v3-holdout:

| problem | v3 agree | v4 agree | delta |
|---|---:|---:|---|
| `multiply_small_ho` | 100% | 100% | tie |
| `factor_ho` | 100% | 100% | tie |
| `gsm8k_toy_ho` | 26.7% | 33.3% | **+6.7pp** (targeted gain) |
| `algebra_toy_ho` | 60.0% | **26.7%** | **-33.3pp** (collateral loss) |

The intended intervention worked: gsm8k generalization improved by
adding gsm8k training variants. But **algebra catastrophically regressed**
by 33 percentage points.

## Post-mortem: why did algebra regress?

Two independent causes, surfaced by `experiments/algebra_post_mortem.py`:

### Cause 1: class dilution (expected)

The training composition shifted from a balanced 25/25/25/25 in v2 and
v3 to 14/57/14/14 in v4 (arithmetic / gsm8k / algebra / factor). The
algebra signal was diluted from 25% to 14% of training data. With a
brittle problem type, that under-fits the algebra pattern.

### Cause 2: target instability across rounds (unexpected)

Comparing `algebra_toy_1` MCTS targets between v3-train and v4-train,
same problem and same (seed, turn) pairs:

* **mean L1(v3-train algebra, v4-train algebra) = 0.633**
* **argmax agreement = 7/15 (47%)**

CORRECTION (2026-06-09, post-v4b): this comparison confounds prior
change with sampling noise. v3-train was collected with the v2-heads
prior and v4-train with the v3-heads prior, NOT the same prior as
originally written here. The clean same-prior test became available
after v4b: **v4-train vs v4b-train (both v3-heads prior, same
problems, same MCTS seeds) gives mean L1 = 0.553 with 7/15 argmax
agreement** on algebra_toy_1. The instability is real and nearly as
large under fully matched conditions; the prior change adds little on
top. Root cause: the MCTS rng seed controls action selection inside
the search, but the LLM transitions are sampled at temperature 0.7
with NO sampling seed passed to the server (verified: no seed param
in llm_client.py), so every collection run draws different
continuations. v3-train's "lucky" algebra targets were not
reproducible.

### Cause 3 (mechanism): v3 was riding a brittle shortcut

Step-by-step inspection of v3 head predictions on algebra_toy_ho records
shows v3 predicts COMMIT_ANSWER for nearly every turn >= 1 state. This
shortcut happens to match algebra's noisy MCTS targets (which also
recommend commit by turn 1-2 in many seeds). v3 was getting 60% on
algebra largely by hard-coding "by turn 1, commit." v4 broke this
shortcut because it saw gsm8k_toy_{2,3,4} states where the right verb at
turn 1 is COMPUTE_SUBEXPR, not COMMIT_ANSWER. v4 learned the gsm8k
pattern correctly but lost the algebra shortcut.

## What this means for the framework paper

The methodology paper now has a multi-round iteration story with a
constructive negative result:

* **Round 1 -> 2 (v2 -> v3):** monotonic improvement on every metric.
  Trained prior produces sharper distillation targets, lower
  training-loss floor, +15 percentage points held-out argmax. Closes
  the loop.
* **Round 2 -> 3 (v3 -> v4):** illustrates structural limits.
  - Naive data expansion can improve the targeted problem type and
    cause collateral regression in under-represented types.
  - High-variance problem families (algebra, where verb choice is
    genuinely ambiguous) need either a larger MCTS budget B or more
    seeds to produce stable distillation targets. At B=20 the targets
    themselves change across rounds.
  - A model that appears to generalize may be riding a brittle shortcut
    that future training breaks. The cure for that is more diverse data,
    not less.

## Suggested follow-ups

1. **v4b**: balanced expansion -- add 2 algebra variants and 1 factor
   variant so composition is 10/40/30/20 (arithmetic/gsm8k/algebra/factor).
   Tests whether the algebra regression is recoverable by rebalancing
   class weight alone. **EXECUTED 2026-06-09; see "Round 2b" below.
   Result: refuted. Rebalancing did not recover algebra.**
2. **Higher B on algebra**: re-collect algebra rows at B=40 or B=100
   and measure whether the across-round L1 drops below 0.3. This is
   the deeper fix for high-variance problem types. Elevated to the
   leading follow-up by the v4b result.
3. **Per-class loss weighting** during head training to compensate for
   class imbalance without re-collecting data. De-prioritized by the
   v4b result: composition correction at the data level did not help,
   so loss-level correction is unlikely to either.

## v4 artifacts

* `/tmp/distillation_v4_train.jsonl` -- 105 records, v3 prior, 7 problems
* `/tmp/distillation_v4_holdout.jsonl` -- 60 records, v3 prior, held-out problems
* `/tmp/heads_v4.pt`, `/tmp/heads_v4_eval.json`
* `/tmp/v3_vs_v4_train.json`, `/tmp/v3heads_on_v4holdout.json`,
  `/tmp/v4heads_on_v4holdout.json`, `/tmp/v4heads_on_v3holdout.json`
* `/tmp/algebra_post_mortem.log` -- side-by-side v3 vs v4 algebra trace


# Round 2b: v4b (controlled rebalance test of the class-dilution hypothesis)

v4b re-runs the second-round head fit on a 10-problem training set:
the 7 problems of v4 plus `algebra_toy_2` ("Solve 4x - 3 = 9"),
`algebra_toy_3` ("Solve 7x + 2 = 16"), and `factor_2`
("Factor x^2 - 16"). Composition shifts from v4's 14/57/14/14 to
10/40/30/20 (arithmetic/gsm8k/algebra/factor); algebra's share rises
from 14% to 30%, above even the balanced 25% of v2/v3. Everything
else held fixed: v3 heads as prior, B=20, seeds 0-4, same
architecture and training hyperparameters. 150 train records
(~2h45m wall-clock, partially GPU-shared) + 60 holdout records.

If class dilution were the dominant mechanism behind the v3->v4
algebra regression, restoring algebra's share should restore algebra
generalization.

## v4b sharpness

| dataset | N | mean max | median | entropy | frac max>=0.5 |
|---|---:|---:|---:|---:|---:|
| v4b-train (v3 prior, 10 problems) | 150 | 0.516 | 0.550 | 1.927 | 53% |
| v4b-holdout (v3 prior) | 60 | 0.588 | 0.700 | 1.756 | 72% |

v4b-train sits between v4-train (0.506) and v3-train (0.549): the
larger set mixes seen and unseen problem variants in a different
ratio. v4b-holdout is the sharpest holdout collection so far.

## v4b head training

| | v2 | v3 | v4 | v4b |
|---|---:|---:|---:|---:|
| Final policy CE | 1.4512 | **1.2934** | 1.3574 | 1.3501 |
| Final Q MSE | 0.006829 | **0.004156** | 0.005477 | 0.005348 |
| Training argmax % | 85.0% | **91.7%** | 88.6% | 88.7% |

## v4b held-out generalization (on v3-holdout, apples-to-apples)

| | v2 | v3 | v4 | v4b |
|---|---:|---:|---:|---:|
| Overall argmax % | 56.7% | **71.7%** | 65.0% | 61.7% |
| Mean L1 | 0.540 | **0.314** | 0.407 | 0.347 |

Per-problem argmax agreement:

| problem | v2 | v3 | v4 | v4b |
|---|---:|---:|---:|---:|
| multiply_small_ho | 93.3% | 100% | 100% | 100% |
| factor_ho | 100% | 100% | 100% | 100% |
| gsm8k_toy_ho | 20.0% | 26.7% | 33.3% | 33.3% |
| algebra_toy_ho | 13.3% | **60.0%** | 26.7% | **13.3%** |

On its own holdout collection (v4b-holdout): 66.7% overall, algebra
40.0%. The 13.3% vs 40.0% gap for the SAME v4b heads across the two
holdout collections comes from the holdout targets themselves
differing (v3-holdout was collected with the v2-heads prior,
v4b-holdout with the v3-heads prior).

Verified reproducibility nuance (state-dependence): at fully matched
conditions (v4-holdout vs v4b-holdout: same v3-heads prior, same
problems, same MCTS seeds, different LLM-sampling realizations), the
held-out algebra targets reproduce almost EXACTLY (mean L1 = 0.087,
14/15 argmax agreement) because those states are dominated by a
terminal commit with deterministic verifier reward. The same matched
comparison on the training algebra states (v4-train vs v4b-train)
does NOT reproduce (L1 = 0.553, 7/15). Instability concentrates
exactly where verb choice is genuinely ambiguous, which is where the
target-stability monitoring metric should look.

## Verdict: class-dilution hypothesis refuted as dominant mechanism

Algebra fell to 13.3% (below v4's 26.7%) despite algebra's training
share DOUBLING from 14% to 30%. Mean L1 on algebra improved
(0.868 -> 0.758), so v4b's distributions are closer in distance while
the argmax matches less, consistent with noisy targets whose argmax
is near-arbitrary among 2-3 plausible verbs.

Shortcut check: on v3-holdout algebra records at turn >= 1, the
"commit at turn >= 1" rate is v3 = 9/10 (90%), v4 = 2/10 (20%),
v4b = 2/10 (20%). The shortcut is broken by ANY second-round fit
regardless of composition, confirming the shortcut-breakage mechanism
is composition-independent.

Surviving explanation: target instability (across-seed L1 = 0.633 on
matched algebra states at B=20) plus shortcut dependence. Both are
now in the paper (Section 5.4) and the monitoring requirements in the
pre-registered plan were updated accordingly.

## v4b artifacts

* `/tmp/distillation_v4b_train.jsonl` -- 150 records, v3 prior, 10 problems
* `/tmp/distillation_v4b_holdout.jsonl` -- 60 records, v3 prior
* `/tmp/heads_v4b.pt`, `/tmp/heads_v4b_eval.json`
* `/tmp/v3_vs_v4b_train.json`, `/tmp/v4bheads_on_v3holdout.json`,
  `/tmp/v4bheads_on_v4bholdout.json`


# Reproducibility probe (2026-06-10): does per-call LLM seeding fix target instability?

After the code review added `seed_llm` plumbing (the MDP derives a per-call
sampling seed from the planner's rng and forwards it to the LLM client),
this probe tests whether seeding delivers the collection-level
reproducibility the v4/v4b post-mortem found missing.

**Setup.** Two sequential collections (no self-contention), same `--seeds 0`,
problems `multiply_small` + `algebra_toy_1`, B=10, on the *current* GPU box
(reconfigured since the pilot: new server, embeddings disabled). L1 measured
between matched (problem, seed, turn) `pi_MCTS` records. **N is small (6 matched
records per condition); treat numbers as directional.**

| condition (current box) | mean L1 | exact | algebra family |
|---|---:|---:|---|
| unseeded temp=0.7 | 0.200 | 3/6 | diverges |
| seeded temp=0.7   | 0.255 | 5/11 | 0/3 diverges |
| seeded temp=0 (greedy) | 0.100 | 5/6 | **3/3 exact** |

(old-box unseeded algebra baseline from the v4-vs-v4b post-mortem: L1 0.553)

**Findings.**

1. **Per-call seeding does NOT fix collection reproducibility at temp>0.**
   Seeded (0.255) is statistically indistinguishable from unseeded (0.200)
   at this N. Isolated single calls DO reproduce at matched seed (verified:
   6/6 identical on a real COMMIT_ANSWER prompt in isolation), so the seed
   is honored at the call level. The failure is structural: an MCTS search's
   *sequence* of LLM calls depends on the responses (each response is the
   next state, which determines the next prompt and consumes the rng), so a
   single non-deterministic call early in the search desynchronizes the
   rng-to-state mapping and every downstream seed lands on a different prompt.
   You can seed the calls but not the tree they induce.

2. **Greedy decoding (temp=0) is the effective lever.** It cut L1 to 0.100
   and made the high-ambiguity `algebra_toy_1` family fully reproducible
   (0/3 -> 3/3 exact). Greedy reduces per-call non-determinism enough that
   the call sequences stay aligned across runs. Caveat: greedy collapses the
   stochastic MDP to deterministic, removing the transition stochasticity
   MCTS is meant to sample, so it changes what is being learned -- it is a
   reproducibility/realism tradeoff, not a free fix.

3. **A compute-level floor remains (~0.1) even at greedy.** One record
   (multiply_small turn 2) still diverged at temp=0 with no sampling RNG in
   play, so the residual is GPU-compute non-determinism (reduction-order /
   atomics flipping a near-tie argmax). This is the irreducible floor of the
   stack as configured.

**Implication for the pre-registered study.** The lever for stable
distillation targets is NOT per-call seeding; it is (a) larger B, so the
visit-count distribution averages over occasional divergent trajectories,
and (b) optionally greedy collection for a reproducible (if deterministic)
target. The pilot's B=20 was too small for the visit distribution to be
robust to the ~0.1 floor; the pre-registered B=500 should be far more
stable by averaging. **The key untested follow-up is the B-sweep**: measure
L1 vs B (e.g. B in {10, 40, 160}) to confirm targets stabilize with budget.

**Code added.** `LLMContextMDP(seed_llm: bool = True)` and
`collect_distillation --no-llm-seed`. The seed feature is retained: it makes
*individual* transitions reproducible (useful for debugging and for the
greedy path) even though it does not make the tree search reproducible.

**Infra note.** The GPU box currently returns HTTP 501 on both `/api/embed`
and `/v1/embeddings` ("does not support embeddings. Start it with
`--embeddings`"). Embeddings are needed for head training but not for
uniform-prior collection; this probe ran fine with embeddings disabled
(the Batch-2 negative-cache degraded gracefully). Re-enable embeddings on
the server before the next head-training round.


# B-sweep: does the distillation target stabilize with the search budget? (2026-06-10)

Follow-up to the reproducibility probe. For a *fixed* initial state we ran
K=6 independent MCTS searches (different rng seeds -> independent stochastic
LLM transitions) at B in {10, 40, 160}, on `multiply_small` (low-ambiguity
control) and `algebra_toy_1` (the high-ambiguity family whose instability
broke the v4/v4b analysis). Same start state every run, so cross-run spread
is the visit-count estimator's own variance, not trajectory drift.
Script: `experiments/b_sweep_stability.py`; data: `/tmp/b_sweep.json`.
qwen2.5:3b, temp=0.7, uniform prior, ~94 min wall-clock.

| problem | B | mean max-prob | entropy (bits) | argmax agree | mean pairwise L1 |
|---|---:|---:|---:|---:|---:|
| multiply_small | 10  | 0.32 | 1.96 | 4/6 | 0.133 |
| multiply_small | 40  | 0.59 | 1.57 | 3/6 | 0.827 |
| multiply_small | 160 | 0.84 | 0.84 | **6/6** | 0.176 |
| algebra_toy_1  | 10  | 0.30 | 1.97 | 4/6 | 0.213 |
| algebra_toy_1  | 40  | 0.47 | 1.73 | 3/6 | 0.653 |
| algebra_toy_1  | 160 | 0.74 | 1.17 | **5/6** | 0.678 |

**Findings (N=6 per cell; directional, not a scaling law).**

1. **Sharpness rises monotonically with B.** Mean max-probability climbs
   (0.30-0.32 -> 0.84/0.74) and entropy falls monotonically (1.97 -> 0.84
   for multiply, 1.17 for algebra). Bigger budget = sharper targets, as
   the estimator concentrates visits on the high-value verb(s).

2. **The mean-pairwise-L1 metric is NON-monotonic and sharpness-confounded.**
   It is *low* at B=10 not because runs agree but because every run is
   near-uniform (diffuse distributions are trivially L1-close); it *spikes*
   at B=40 when runs have sharpened onto DIFFERENT verbs; and for the easy
   problem it falls again at B=160 once runs concentrate on the SAME verb.
   This means the pilot's own L1 instability numbers (0.2-0.55, measured at
   B=20) were a mix of real disagreement and the flat-distribution artifact.
   Argmax agreement is the cleaner stability signal.

3. **The recommended verb converges at high B.** All 6 multiply runs agree
   at B=160 (up from a shallow 4/6 of a near-flat distribution at B=10);
   algebra reaches 5/6. The dip to 3/6 at B=40 is the "committed but
   disagreeing" transition zone: runs are sharp enough to commit but not
   sharp enough to commit to the same verb.

4. **The pilot's B=20 sits in the worst zone.** B=20 is just past the
   diffuse-and-trivially-close regime (B=10) and squarely inside the
   committed-but-disagreeing transition (B≈40). That is the budget at which
   independent collections disagree MOST on the argmax, which directly
   explains the pilot's observed target instability (v4 vs v4b matched-seed
   L1 = 0.553 on algebra). The instability was real, but B=20 was nearly
   the worst possible budget for it.

5. **Cost.** B=160 searches took ~240-260s each on this box. The
   pre-registered B=500 would be ~13-17 min per search -- expensive, but the
   convergence trend (6/6 and 5/6 argmax agreement already at B=160) is the
   evidence that the budget buys reproducible targets where B=20 could not.

**Implication for the pre-registered study.** The lever for stable
distillation targets is the search budget, not per-call seeding. B=500 (the
pre-registered heavy-search value) sits well past the convergence crossover
observed here, so the targets it produces should be substantially more
reproducible than the pilot's B=20 targets. The high-ambiguity family
(algebra) converges more slowly (5/6 vs 6/6 at B=160), so the
target-stability diagnostic should be stratified by problem family, with the
budget set by the slowest-converging family rather than the average.


# B-sweep at publication weight: uniform K=20, five budgets (2026-06-10)

Extension of the K=6 sweep to uniform K=20 per cell across
B in {10, 20, 40, 80, 160}, both problems, same fixed-initial-state
protocol (cross-run spread = estimator variance, not trajectory drift).
Existing K=6 runs (seeds 0-5) merged with +14 runs (seeds 6-19) at
{10,40,160}; fresh K=20 (seeds 0-19) at {20,80}. ~9,900 new simulations,
~9.5h wall-clock total across four sequential jobs (GPU shared at times).
Primary stability statistic: PAIRWISE argmax agreement (probability two
independent collections produce the same training argmax), bootstrap
percentile CIs (R=10,000). Merge + stats: `experiments/b_sweep_analyze.py`;
data: /tmp/b_sweep_merged.json; figure: paper/figures/b_sweep.pdf.

| problem | B | K | mean maxP | H (bits) | modal (frac) | pairwise agree [CI95] | mean L1 |
|---|---:|---:|---:|---:|---|---:|---:|
| algebra_toy_1 | 10 | 20 | 0.34 | 1.94 | FORM_HYP (0.40) | 0.25 [0.22,0.42] | 0.292 |
| algebra_toy_1 | 20 | 20 | 0.38 | 1.90 | COMPUTE (0.45) | 0.32 [0.25,0.49] | 0.376 |
| algebra_toy_1 | 40 | 20 | 0.43 | 1.80 | BACKTRACK (0.45) | 0.38 [0.29,0.53] | 0.558 |
| algebra_toy_1 | 80 | 20 | 0.49 | 1.73 | COMPUTE (0.60) | 0.46 [0.36,0.66] | 0.580 |
| algebra_toy_1 | 160 | 20 | 0.69 | 1.28 | BACKTRACK (0.85) | 0.73 [0.56,1.00] | 0.634 |
| multiply_small | 10 | 20 | 0.35 | 1.92 | FORM_HYP (0.55) | 0.41 [0.33,0.61] | 0.259 |
| multiply_small | 20 | 20 | 0.45 | 1.82 | COMPUTE (0.40) | 0.31 [0.30,0.46] | 0.507 |
| multiply_small | 40 | 20 | 0.65 | 1.42 | COMPUTE (0.70) | 0.52 [0.36,0.81] | 0.643 |
| multiply_small | 80 | 20 | 0.80 | 0.98 | COMPUTE (0.85) | 0.73 [0.56,1.00] | 0.589 |
| multiply_small | 160 | 20 | 0.88 | 0.68 | COMPUTE (0.90) | 0.81 [0.61,1.00] | 0.456 |

**Findings (K=20; these CORRECT the K=6 narrative).**

1. **Sharpness rises monotonically with B on both families** (maxP and
   entropy, every cell). Unchanged from K=6.

2. **The K=6 "dip at B=40" was small-sample noise.** At K=20 the
   algebra agreement curve is strictly monotone (0.25 -> 0.73). The
   "committed-but-disagreeing transition zone at B=40" story is DEAD.
   A weak dip remains on multiply at B=20 (0.41 -> 0.31, CIs overlap).
   The defensible claim: **B=20 is the low-agreement floor of the grid**
   (0.32 algebra, 0.31 multiply: the two lowest cells outside algebra
   B=10), not "the worst point of a non-monotonic curve."

3. **The L1 confound is stronger than first reported.** On algebra,
   mean pairwise L1 RISES monotonically (0.29 -> 0.63) while true
   stability also rises: L1 is ANTI-correlated with stability there,
   because one sharp minority run on a different verb costs ~1.3 L1
   while diffuse-vs-diffuse costs ~0.2. On multiply L1 is an inverted U
   peaking at B=40. Distributional distance is not a stability metric
   at any budget; pairwise argmax agreement is.

4. **Convergence is family-dependent**: multiply reaches 0.73 by B=80
   and 0.81 at B=160; algebra needs B=160 for 0.73. Budgets must be set
   by the slowest family (already a design commitment).

5. **Budget-indexed targets are a hard-family phenomenon**: algebra's
   modal verb flips COMPUTE(B=80) -> BACKTRACK(B=160) even as
   within-budget agreement rises; multiply locks onto COMPUTE from
   B=20 up. "Stable target" = within-budget stability; comparisons
   must fix B (already a design commitment, now with evidence).

**Paper actions taken**: Table 5 replaced with the K=20 grid + CIs;
"worst point"/"transition zone" claims rewritten to the floor framing in
sections 5.3/5.5/5.6; figure paper/figures/b_sweep.pdf added; the K=6
-> K=20 dip dissolution kept as an explicit small-sample cautionary
note (a diagnostic run at K=6 can manufacture a transition zone).


## Infra post-mortem: why the pilot's embedder died (resolved 2026-06-12)

The box's Ollama was updated to 0.30.7 around 2026-06-10. Root cause of
the /api/embed 501 for qwen2.5:3b, established from the trackers:

* The error text ("This server does not support embeddings. Start it
  with `--embeddings`") is llama.cpp's llama-server message, surfacing
  through Ollama's new llama.cpp integration (the 0.30 line layered
  llama.cpp in as a runner "for improved compatibility and performance").
* Ollama has been migrating architectures from the legacy llama.cpp
  runner to its own engine since 2025, and embedding support repeatedly
  regressed along the way (ollama#12368: Qwen3-Embedding broke at v0.12.0
  when "qwen models are now using the ollama backend"; ollama#10824:
  Qwen2-based embedding models still unsupported on the new engine;
  llama.cpp#15406: /v1/embeddings 501s after build 5630 even with
  embeddings enabled).
* Net effect on 0.30.7: dedicated embedding models load with the
  embedding path enabled and work (verified: nomic-embed-text 768-d
  deterministic, qwen3-embedding:0.6b 1024-d deterministic once warm);
  generative models load completion-only, so embed requests reach a
  runner started without the embeddings flag -> 501. The pilot's
  embed-with-the-generation-model usage was a permissive default of the
  old version, not a documented capability, and it went away with the
  engine migration rather than by an announced deprecation.

Consequence (already absorbed in commit 1112a4f): pilot artifacts are a
sealed 2048-d archive; new rounds default to nomic-embed-text (768-d);
embedder/checkpoint dim mismatches fail loudly. If old-space continuity
is ever required: pin an older Ollama on a second port, or serve the
qwen2.5:3b GGUF via a bare llama-server with embeddings enabled (pin a
known-good build; upstream had a 501-even-with-flag regression).


## In-process Transformers backend: frodo smoke PASS (2026-06-18)

Stage 2 of the migration off the Ollama HTTP serving stack (post-mortem
above). The frozen model now runs in-process on the GPU box (frodo,
RTX 3060 12 GB) via `TransformersLLMClient` / `TransformersEmbeddingClient`,
both implementing the existing protocols so the MDP/MCTS/heads are
untouched.

Environment (run provenance): torch 2.12.0+cu130, transformers 5.12.0,
Python 3.12, CUDA 13.0, driver 580.159.03, uv-managed venv at
`frodo:~/expert-iteration/.venv`. Substrate Qwen/Qwen2.5-1.5B-Instruct
(hidden_size 1536, bf16, 2.88 GiB VRAM, 4.6s load).

Smoke (`experiments/run_transformers_smoke.py`) verified all four
properties the HTTP stack could not provide:

* seeded-generation reproducibility: seed 42 / temp 0.7 -> identical
  output across 3 calls (the property server-side seeding could not
  guarantee under load);
* hidden-state embedding: masked-mean pool of the last hidden layer,
  dim 1536, L2 = 1.0, deterministic (the adapter architecture's real
  input, not the embed-endpoint proxy);
* countable forwards: 3 generate calls / 72 tokens reported by the
  client (the compute-axis truth an API hides);
* one frozen model serving both roles via share_model_with (one copy
  in VRAM).

Latencies: ~0.3s warm generation, 0.16s embedding, B=3 MCTS in 2.9s
(~25 tok/s incl. search overhead). The 1.5B substrate is comfortable on
12 GB.

API note for future readers: transformers 5.x `apply_chat_template(...,
return_tensors="pt")` returns a BatchEncoding (input_ids +
attention_mask), not a bare tensor; the client uses return_dict=True and
passes both to generate(). 13 client tests use injected fakes honoring
this contract; no GPU needed for the test suite.


## Shared single-forward encoder: frodo verification (2026-06-18)

Module 2 of the benchmark architecture (method.tex 3.4 "one forward, not
two"): one state prefill yields both h_T for the heads and the KV cache
for verb generation, replacing the pilot's two independent forwards (one
embed-endpoint call, one full re-encode for generation).
`SharedForwardEncoder`, verified on frodo against Qwen2.5-1.5B.

Refactor: `_prompt_for` split into `_state_prefix(state)` (ends at the
[STATE_END] sentinel) + `_verb_suffix(verb, state)`, byte-identical
concatenation, so the HTTP/mock paths are unchanged and the shared
prefix is exactly the verb-independent, cacheable span.

Verified properties:

* **h_T equivalence: max|h_T_shared - h_T_standalone| = 0.00e+00** (both
  bf16 and fp32). The hidden state from the shared prefill is bit-
  identical to a standalone forward over the same prefix: the heads see
  the same embedding whether or not the cache is reused.
* **greedy generation equivalence: token-identical under fp32** (cached-
  prefill vs a fresh full prefix+suffix prefill produce the same tokens).
  Under bf16 they agree except for an occasional final token (observed:
  "Divide by 2." vs "Divide by 2:"): the documented GPU-compute /
  reduction-order floor (cf. the B-sweep section), amplified by bf16,
  not a plumbing error. The fp32 run isolates this cleanly.
* **prefill saving: 53%** for a 4-verb node (1 state prefill of 228 tok +
  4 short suffixes = 616 tok, vs 4 full re-prefills = 1300 tok). The
  saving grows with state length and branching factor.

Two findings worth carrying forward:

1. A cache-plumbing bug was caught only by the real-model smoke comparing
   against ground truth: passing the suffix alone to generate (instead of
   the full prefix+suffix with the cache) makes generate infer a wrong
   attention mask, and the continuation degenerates
   (target_id "none", empty children). The fix passes the full sequence
   plus the prefix cache; generate then recomputes only the suffix
   onward. The fake-based unit tests could not have caught this (it is
   real-attention behavior) -- the frodo smoke is load-bearing.
2. Under raw completion (no chat template, the faithful 3.4 form), the
   1.5B model's output is structurally valid JSON but sometimes
   semantically off on a fresh state (it produced valid content here once
   the cache was correct). Whether the benchmark study uses raw
   completion or chat-template framing for the verb-instantiator is an
   open question to settle empirically; the encoder supports either via
   the prefix/suffix text it is handed.


## Section 3.7 training loop: replay + combined objective (2026-06-18)

Module 3 of the benchmark architecture, and the headline algorithm:
approximate Q-learning with MCTS-supplied targets, alternated with policy
distillation. Three pieces the pilot lacked, now built and CPU-verified:

* `train.py`: the combined objective L_total = L_Q + lambda*L_pi
  (Eqs. lossq/losspi/losstotal). L_Q is masked MSE of Q_psi against the
  stored MCTS chance-node estimate Q_MCTS (the paper's stated novelty:
  the target is Q_MCTS, not a single-sample bootstrapped Bellman target);
  L_pi is masked cross-entropy of pi_theta against pi_MCTS. Losses use the
  actor's explicit q()/policy_logits() accessors, not the forward tuple.
* `replay.py`: ReplayBuffer (prioritized by mean TD error, FIFO eviction
  under a capacity cap, stores Q_MCTS+pi_MCTS per sample so stale entries
  stay correct). `sample_from_record` reuses heads.encode_batch so the
  buffer ingests the exact collect_distillation JSONL schema.
* `train.Trainer`: runs minibatch steps, updates priorities from TD
  errors, syncs the target-Q network every N steps (Alg.1 line 14).

The actor gained a `policy_logits(h)` accessor so the losses are
unambiguous about which head is which.

End-to-end validation (`experiments/run_train_demo.py`, CPU, the pilot's
v2 records, stored 2048-d embeddings, 400 steps):

* q_loss 0.111 -> 0.012 (~10x), pi_loss 1.756 -> 1.46;
* the pi_loss floor (~1.46) MATCHES the pilot's v2 final policy
  cross-entropy of 1.4512 on the same data -- the new architecture +
  training loop reproduce the pilot's training behavior (the cross-entropy
  floor is the target entropy), while adding the replay buffer and
  target-Q sync the pilot did manually-across-sessions or not at all.

CPU-tested: 12 train tests (buffer FIFO/priority/sampling, losses finite,
the combined objective drives BOTH heads down >50%/>10% on a fixed
buffer, target sync on interval, priority update). 177 tests green.

Remaining for the full study: the OUTER collection-and-train loop that
wires this core to the shared-forward encoder + the planner (MCTS
collection with the current heads as priors and the target-Q as leaf
value -> replay add -> M train steps -> repeat), and the FLOP-accounting
of per-decision forwards for the compute axis. Both are GPU/orchestration
layers on top of the three CPU-verified modules now in place.


## Outer loop: collect -> train -> re-prior (2026-06-18)

The outer collection-and-train loop (method.tex 3.7, Algorithm 1) wiring
the three CPU-verified modules (architecture/train/replay) to the planner
(mcts) and the in-process encoder (shared_forward). What it adds over the
pilot's one-shot offline distillation:

* a `value_fn` leaf-evaluation hook in MCTS (additive; value_fn=None keeps
  the rollout path, so the section-10 triangulation tests are untouched).
  The leaf value is max_a' Q_target(s', a'), the bootstrap of approximate
  Q-learning, so search is now AlphaZero-style rather than pure rollout.
* a StateActor bridge (planner_actor.py) that supplies the MCTS prior
  (online policy head) and leaf value (target-Q head) plus embed(s) over
  MathState, all sharing one prefill per node via a state-prefix-keyed
  PrefillCache (the "one forward per decision" property).
* an additive encoder generation path in LLMContextMDP (one forward per
  node via encoder.instantiate_json; the llm_client path is unchanged when
  no encoder is passed).
* the loop itself (outer_loop.py): collect_iteration drives one trajectory
  per problem, samples the next verb from pi_MCTS at temperature tau, and
  adds one ReplaySample per node; run_outer_loop alternates collection with
  Trainer.run(M) under a warmup gate and records per-iteration FLOP/loss
  counts in IterationStats.

CPU-tested end to end with FakeEncoder + EchoLLMClient: the load-bearing
test asserts the loop CLOSES (after training the learned prior diverges
from uniform and shifts the MCTS visit distribution), not merely that it
runs. Full suite green.

Frodo run: `python experiments/run_outer_loop.py` (record numbers here
after the first GPU run).
