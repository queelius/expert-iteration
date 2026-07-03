# Casework problem generator: a structure-exercising synthetic dataset

Status: spec (nothing built). Companion to the Deliberation MDP integration
spec (2026-07-02) and motivated by the GSM8K Rung-1 diagnosis below.

## 1. Why

The completed 500-problem GSM8K Rung-1 sweep (70.8% solved, 3,906 teacher
decisions) confirms a stable dataset property: structural verbs collapse.
`split` was chosen 10 times and `abandon` 11 times out of 3,906 decisions
(about 0.5% combined), because GSM8K problems are linearly solvable and the
optimal policy really is `compute -> verify -> commit`. Behavior cloning on
this data alone learns a linear-chain policy; the AND-OR forest never earns
its keep and Phase B's pointer head sees no interesting targets.

The fix is a problem distribution where case analysis is *required for
correctness*: the answer cannot be computed without exploring every case,
and some cases are dead ends that must be pruned. Because our symbolic
verifier (SymPy) can decide these problems, we can generate them in reverse:
sample the solution structure first, then render the problem statement
around it. Ground truth is correct by construction, generation is
deterministic and free (no LLM anywhere in the pipeline), and difficulty is
a knob we control (a curriculum for the later RL rung).

## 2. Constraints (binding)

1. **No changes to the MDP, lenses, verifier, or reward.** The generator
   produces `{problem, answer, meta}` records consumed by the existing
   runner; `DeliberationMDP`, `MathVerifier`, and the terminal-verdict logic
   are untouched. If a family cannot be expressed under the existing
   machinery, the family is redesigned, not the machinery.
2. **Single-scalar answers.** The terminal verdict compares the committed
   evidence's right-hand side against one expected value. Families with
   multiple solutions must ask a scalar question *about* the solution set
   ("the sum of all solutions", "the largest solution"). This is the load-
   bearing trick: the scalar cannot be known without doing the full
   casework, but the final commit stays a single checkable number.
3. **Integer-clean by construction.** Sample the roots first, derive the
   coefficients, so every intermediate computation is a ground equation over
   integers (or simple rationals) that the COMPUTE lens's no-unknowns
   constraint can express.
4. **The problem statement must not enumerate the cases.** "Solve
   |2x - 3| = 7; report the sum of all solutions" leaves `split` a genuine
   decision. A statement that lists "case 1: ..., case 2: ..." dictates the
   structure and teaches nothing.
5. **Deterministic and seeded.** Same seed, same dataset, byte for byte.
   Generation is pure Python + SymPy.
6. **No LLM-judge, no learned reward, anywhere** (repo-wide invariant,
   restated here because dataset generation is a tempting place to slip one
   in "just for filtering").

## 3. Problem families (v1: three)

### F1: absolute value, linear inside

Symbolic form: `|a*(x - m)| = a*k` rendered as `|a*x + b| = c` with
`b = -a*m`, `c = a*k`; roots are `m + k` and `m - k`.

- Variant F1a (both cases live): question is "the sum of all solutions"
  (answer `2m`) or "the largest solution" (answer `m + k`). Sum requires
  both roots; largest requires knowing both to compare.
- Variant F1b (one dead case): add a side condition, "given x > d", with
  `d` sampled strictly between the roots. One case's candidate violates the
  condition and must be abandoned; question is "find x" (answer: the
  surviving root).

Sampling: integers `a in 1..5`, `m in -9..9`, `k in 1..9`; for F1b,
`d in (m - k, m + k)`.

### F2: piecewise-linear equation with a domain-dead branch

Symbolic form: `f(x) = a1*x + b1` for `x < t`, `f(x) = a2*x + b2` for
`x >= t`; solve `f(x) = v`.

Reverse construction: pick threshold `t`; pick the live branch and a live
solution `xL` inside that branch's domain; set that branch's coefficients
so it passes through `(xL, v)`. For the dead branch, pick coefficients
whose candidate `xD = (v - b)/a` lands *outside* its own domain (candidate
exists arithmetically but violates the branch condition, forcing an
`abandon`). Variant F2a: exactly one live branch, question "find x".
Variant F2b: both branches live, question "the sum of all solutions".

### F3: quadratic with integer roots

Symbolic form: sample integer roots `r1 < r2`, expand
`(x - r1)*(x - r2) = x^2 - (r1+r2)*x + r1*r2`; render as
"solve x^2 + p*x + q = 0". Question: "the largest solution" (answer `r2`)
or "the product of the solutions" is forbidden (readable off `q` without
casework); "sum" is likewise forbidden (Vieta reads it off `p`). Largest
or smallest root only.

Deferred to v2: nested case splits (casework inside a case), unsatisfiable
problems (a "no solution" answer needs an answer-format convention the
terminal verdict does not have), inequality problems with interval answers,
and word-problem skins over the symbolic cores.

## 4. Feasibility trace (existing machinery, no changes)

Problem: "Solve |2*x - 3| = 7. Report the sum of all solutions."
Expected answer: "3" (roots 5 and -2).

1. `split:0` with connective `exhaustive_case`: children "case 2x-3 = 7
   (x >= 3/2)" and "case 3-2x = 7 (x < 3/2)" (statements authored by the
   frozen generator through the SPLIT lens).
2. In case 1: `compute` the ground equation "(7 + 3)/2 = 5"; `verify`;
   `commit` the case on the verified fact.
3. In case 2: `compute` "(3 - 7)/2 = -2"; `verify`; `commit`.
4. At the root: `compute` "5 + (-2) = 3" (premises: the two case facts);
   `verify`; `commit:0`.
5. Terminal verdict: right-hand side of the most recent verified evidence
   is "3"; matches the expected answer; reward +1.

The dead-case variant is identical except one case's candidate violates the
side condition; the correct action there is `abandon` on that case, and the
root aggregation uses only the surviving fact.

## 5. Deliverables

1. **`experiments/generate_casework.py`**: seeded CLI generator.
   - `--n`, `--seed`, `--families f1a,f1b,f2a,f2b,f3` (default: all,
     round-robin balanced), `--out`, `--split {train,test}`.
   - Output: JSONL records shaped for publication, not just internal use:
     ```json
     {"id": "cw1-<12-hex content hash>",
      "problem": "...natural-language statement...",
      "answer": "41",
      "symbolic": "Eq(Abs(2*x - 3), 7)",
      "solution_set": ["-2", "5"],
      "question_type": "sum_of_solutions",
      "family": "f1a", "n_cases": 2, "n_dead": 0,
      "difficulty": {"tier": 1, "params": {"a": 2, "m": 1, "k": 3}},
      "provenance": {"generator": "generate_casework.py",
                     "generator_version": "1.0.0",
                     "git_sha": "<commit>", "sympy_version": "<ver>",
                     "seed": 0, "index_in_seed": 17}}
     ```
     The `id` is a content hash of `(symbolic, question_type)` so it is
     stable across regenerations and detects accidental duplicates. The
     runner consumes only `problem`/`answer` and ignores the rest.
   - Internal invariant check: every record's `answer` is re-derived with
     an independent `sympy.solveset` call on the symbolic form before it is
     written; a mismatch is a generator bug and raises.
2. **Tests** (`tests/test_generate_casework.py`):
   - determinism: same seed twice, identical bytes;
   - answer correctness: for each family, SymPy re-solve agrees with the
     recorded answer across a seed sweep;
   - structure guarantees: F1b/F2a records have `n_dead >= 1`; F1a/F2b/F3
     have `n_dead = 0`; statements never contain the word "case";
   - integer-cleanliness: all recorded answers parse as integers;
   - id stability and uniqueness: content hash unchanged on regeneration,
     no collisions within a split, and train/test ids are disjoint.
3. **Dataset as a publishable artifact**: `datasets/casework-v1/` with
   - `train.jsonl` (240 problems, 48 per family, seed 0) and `test.jsonl`
     (60 problems, 12 per family, seed 1); disjoint seeds AND a cross-split
     `id`-collision check so no test problem duplicates a train problem;
   - `README.md`: a dataset card (Hugging Face conventions: YAML front
     matter with license/task tags, then Datasheets-for-Datasets style
     sections: motivation, composition, generation process, intended
     uses, limitations, verification method, per-family statistics table);
   - `checksums.sha256` over the data files;
   - license: CC-BY-4.0, stated in the card and the YAML front matter;
   - regeneration command recorded verbatim in the card, with the git SHA
     the files were generated at (byte-reproducibility is the claim, and
     the determinism test backs it).
4. **Structure-aware teacher prompt**: `CONTROLLER_SYSTEM_STRUCTURED` in
   `llm_controller.py`, selected by a new runner flag
   `--controller-prompt {default,structured}` (default unchanged). It adds:
   when a goal requires exhaustive case analysis (absolute values,
   piecewise definitions), `split` with `exhaustive_case` before computing;
   `abandon` a case whose candidate violates a stated condition; aggregate
   verified case results with a final root-level `compute` before
   `commit:0`. It must NOT mention specific problem families' algebra (the
   teacher decides structure; the generator LLM does the math).
5. **Contrast batch** (run, not code): about 240 episodes with the hosted
   teacher (gpt-4.1-mini) on the casework dataset, echo-backend smoke run
   first. Estimated cost: episodes run longer than GSM8K's median 7 turns
   (expect 10-14), so roughly 3M prompt tokens, about $1-2 at current
   rates with prompt caching.

## 6. Success criteria (measurable)

- Structural share: `split` + `abandon` at or above 10% of decisions on the
  casework batch (vs 0.5% on GSM8K).
- `split` chosen in at least 60% of episodes on families where casework is
  required for the scalar (F1a, F2b, F3).
- `abandon` chosen in at least 50% of episodes on dead-case families
  (F1b, F2a).
- Solve rate at or above 50% overall (this is a harder distribution; the
  number matters less than the verb mix, but a floor guards against the
  batch being noise).
- Per-family breakdown reported (solve rate, verb mix, median turns) using
  `meta.family` from the problem file joined to episode logs.

If the structural-share criterion fails with the structured prompt, the
diagnosis order is: (a) inspect whether the teacher solves casework
linearly anyway (i.e. the scalar trick leaks a shortcut; fix the family),
before (b) touching prompt wording, before (c) any thought of machinery
changes.

## 7. Honesty notes

- This dataset is synthetic and narrow by design; it demonstrates that the
  action space *can* be exercised, not that the system generalizes. Report
  it as a structure-contrast batch, never as a benchmark result.
- The scalar-question trick (constraint 2) is a deliberate accommodation of
  the current terminal verdict. If v2 ever adds set-valued answers, that is
  a spec'd change to the verdict convention, not a quiet relaxation.
