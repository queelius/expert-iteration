---
license: cc-by-4.0
task_categories:
  - text2text-generation
  - question-answering
language:
  - en
tags:
  - mathematical-reasoning
  - synthetic
  - case-analysis
  - symbolic-verification
  - reinforcement-learning
pretty_name: Casework v1 (structure-exercising math word problems)
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.jsonl
      - split: test
        path: test.jsonl
---

# Casework v1

A small, fully synthetic dataset of elementary algebra problems whose defining
property is that **case analysis is required for correctness**. Each problem
has a single integer answer that cannot be determined without reasoning over
two cases, and in half the families one case is a dead end that must be pruned.

The dataset was built to exercise the structural operations (`split`,
`abandon`) of a deliberation system whose action space includes decomposing a
goal into cases and abandoning dead ends. On linearly-solvable benchmarks
(for example GSM8K) those operations are never optimal and therefore never
appear in behavior data; Casework makes them optimal by construction. The data
is equally usable as a standalone check of whether a model or agent handles
extraneous-solution and domain-restriction reasoning, which language models
are known to get wrong.

Every problem is generated in reverse (sample the solution structure, then
render the statement) and independently re-solved with SymPy before it is
written, so the labels are correct by construction, not by a solver we trust.

## Quick facts

| | |
|---|---|
| Problems | 300 (240 train, 60 test) |
| Families | 5, balanced (48 train / 12 test each) |
| Answer type | single integer (range -18 to 18 in this release) |
| Answers verified by | independent `sympy.solveset` re-derivation |
| Splits | disjoint by construction (different seeds + content-id collision guard) |
| License | CC-BY-4.0 |
| Generator | `experiments/generate_casework.py` (version 1.0.0) |

## Record schema

One JSON object per line:

```json
{
  "id": "cw1-2640597e2425",
  "problem": "Solve for x given that x < 12: |5x - 30| = 35. Report the value of x.",
  "answer": "-1",
  "symbolic": "Eq(Abs(5*x - 30), 35)",
  "solution_set": ["-1"],
  "question_type": "find_x",
  "family": "f1b",
  "n_cases": 2,
  "n_dead": 1,
  "difficulty": {"tier": 2, "params": {"a": 5, "m": 6, "k": 7, "b": -30, "c": 35, "d": 12, "direction": "lt"}},
  "provenance": {
    "generator": "generate_casework.py",
    "generator_version": "1.0.0",
    "git_sha": "199fc681b406...",
    "sympy_version": "1.14.0",
    "seed": 0,
    "index_in_seed": 1,
    "split": "train"
  }
}
```

- `id` is a content hash of `(symbolic, question_type)`. It is stable across
  regenerations and identifies duplicate problems; it is the join key.
- `answer` is a string holding one integer. It is the field a grader compares
  against (SymPy-normalized equality tolerates `-1`, `-1.0`, etc.).
- `solution_set` is every VALID solution after domain restriction (so a
  dead-case family lists only the survivors). `n_cases - len(solution_set)`
  equals `n_dead`, and this identity is checked at generation time.
- `symbolic` is the canonical equation; `difficulty.params` are the raw
  sampled parameters, sufficient to re-derive everything.
- Consumers that only need to solve the problem read `problem` and `answer`
  and ignore the rest.

## Families

| id | form | structure | question |
|----|------|-----------|----------|
| `f1a` | `|a x + b| = c` | both roots valid | sum of solutions, or largest |
| `f1b` | `|a x + b| = c`, with `x > d` or `x < d` | one root killed by the condition | the surviving x |
| `f2a` | piecewise linear `f(x) = v` | one branch's candidate falls outside its domain | the surviving x |
| `f2b` | piecewise linear `f(x) = v` | both branches yield an in-domain solution | sum of solutions |
| `f3` | `x^2 + p x + q = 0`, integer roots | two distinct roots | largest or smallest root |

Worked examples (one per family, from the train split):

- **f1a** `Solve for all real values of x: |x - 3| = 7. Report the sum of all solutions for x.`  ->  `6`
- **f1b** `Solve for x given that x > 4: |5x| = 30. Report the value of x.`  ->  `6`
- **f2a** `A function is defined by f(x) = -4x when x < 0, and f(x) = -4x - 4 when x >= 0. Find the value of x for which f(x) = -4.`  ->  `0`
- **f2b** `A function is defined by f(x) = -x + 1 when x < 3, and f(x) = x - 2 when x >= 3. Report the sum of all values of x for which f(x) = 1.`  ->  `3`
- **f3** `Solve the equation x^2 - x = 0. Report the largest value of x.`  ->  `1`

## Design note: the single-scalar trick

Multi-solution problems ask a scalar question ABOUT the solution set (the sum,
the largest, the surviving root) rather than for the set itself. The scalar
cannot be known without doing the full casework, but the answer stays one
checkable integer. This keeps the dataset compatible with any grader that
compares a single value, and it is why every answer is one number even though
most problems have two cases.

The statements never enumerate the cases (they never contain the word "case"),
so decomposition is left to the solver.

---

# Datasheet

Following the "Datasheets for Datasets" structure (Gebru et al., 2021).

## Motivation

Created to supply behavior data in which case-decomposition and dead-end
pruning are the correct actions, for training and evaluating a deliberation
controller whose structural operations do not appear on linearly-solvable
math benchmarks. Secondarily, it is a compact test of extraneous-solution and
domain-restriction handling. It was produced as part of a verifier-anchored
reinforcement-learning project (Alexander Towell).

## Composition

300 instances, each a self-contained elementary-algebra problem with a single
integer answer, across five families (absolute-value equations with and
without a side condition, piecewise-linear equations with and without a dead
branch, and integer-root quadratics), balanced across families in both splits.
Each instance carries its symbolic form, full valid solution set, structural
counts, sampled parameters, and generation provenance. There is no personal,
sensitive, or scraped content; every instance is machine-generated.

## Collection process

No collection: instances are sampled deterministically from a seeded random
number generator. The generation procedure samples a solution structure (roots,
threshold, which branch is dead) and derives integer coefficients so that all
intermediate arithmetic is integer-clean, then renders a natural-language
statement. `train` uses seed 0, `test` uses seed 1, and `test` additionally
excludes any content id present in `train`, so the splits are disjoint.

## Preprocessing / labeling

Labels are produced by the generator and then independently re-derived by a
separate `sympy.solveset` pass over each equation and its domain; the answer
field comes from that independent pass. A mismatch, a non-integer solution, a
structural-count inconsistency, or a statement that leaks the word "case"
raises at generation time rather than shipping.

## Uses

Intended for: training and evaluating step-wise reasoning agents and
controllers on case analysis; behavior-cloning data for structural operations;
a diagnostic for extraneous-solution and domain-restriction errors. Not
intended as a general math benchmark: it is narrow and synthetic by design,
covers only linear and quadratic single-variable algebra, and has integer
answers in a bounded range. Results on it demonstrate that case-analysis
behavior can be exercised and measured, not that a model generalizes.

## Distribution and maintenance

Released under CC-BY-4.0. Versioned as `casework-v1`; future families or
answer conventions (for example set-valued answers) will ship as a new
version rather than mutating this one. Byte-level reproducibility of the data
fields is guaranteed by the seed; provenance records the exact generator
version, git SHA, and SymPy version used for this release.

## Limitations and known biases

- Single-variable linear and quadratic algebra only; no geometry, calculus,
  or multivariate reasoning.
- Integer answers in a small range; a model could in principle exploit the
  bounded output space. Graders should not reveal the range.
- The natural-language templates are formulaic; surface diversity is low. A
  future version should add word-problem skins over the symbolic cores.
- `n_cases` is always 2 in v1; nested casework is deferred to v2.

## Regeneration

From the repository root, at the git SHA recorded in `provenance.git_sha`:

```bash
python experiments/generate_casework.py --n 240 --seed 0 --split train \
    --out datasets/casework-v1/train.jsonl
python experiments/generate_casework.py --n 60 --seed 1 --split test \
    --exclude datasets/casework-v1/train.jsonl \
    --out datasets/casework-v1/test.jsonl
```

Verify integrity with `sha256sum -c checksums.sha256`.

## Citation

```bibtex
@misc{towell2026casework,
  title  = {Casework v1: Structure-Exercising Math Word Problems},
  author = {Towell, Alexander Richard},
  year   = {2026},
  note   = {Synthetic, SymPy-verified. CC-BY-4.0.}
}
```
