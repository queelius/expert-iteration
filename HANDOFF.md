# Handoff: resume the outer-loop work on the GPU box

**Date:** 2026-06-18. **Author machine:** bilbo (CPU). **Target:** frodo (GPU).

## TL;DR

The outer collection-and-train loop (`method.tex` 3.7, Algorithm 1) is
implemented, merged to `main`, and CPU-verified end to end (191 tests pass).
It is the headline algorithm: approximate Q-learning with MCTS-supplied
targets, alternated with policy distillation. The loop is built
encoder-agnostic and tested with a CPU `FakeEncoder`, so bilbo proves the
orchestration is correct without a GPU. The one thing that needs the GPU box
is the real model forwards. **That is the next step: run it on frodo and
record the numbers.**

## Resume on frodo (exact steps)

```bash
git clone https://github.com/queelius/expert-iteration.git
cd expert-iteration

# hf  = torch + transformers (the in-process frozen model)
# llm = httpx + sympy (sympy is non-optional: the verifier reward uses it)
# dev = pytest
pip install -e ".[hf,llm,dev]"

python -m pytest -q                            # sanity: expect 191 passed, 1 warning
python experiments/run_outer_loop.py --help    # confirm the harness wires up
python experiments/run_outer_loop.py           # the real run (downloads Qwen2.5-Math-1.5B-Instruct)
```

Useful flags (defaults in parentheses): `--model` (Qwen/Qwen2.5-Math-1.5B-Instruct),
`--iterations` (3), `--num-simulations` (16), `--train-steps` (200),
`--batch-size` (16), `--c-puct` (1.5), `--discount` (1.0), `--tau` (1.0),
`--sync-interval` (200), `--max-turns` (8), `--problems` (train).

## What the run produces and what to look for

`run_outer_loop` prints one summary line per outer iteration:
decisions, simulations, buffer size, prefill/gen FLOP counters, and the
combined-loss curve (`L_total` first -> last). Across iterations you want to
see: the buffer fill, `L_Q` trend down (the robustly monotone signal of
approximate Q-learning; the pilot saw roughly 10x), and the policy prior
sharpen so search visits concentrate. The CPU test already proved the loop
CLOSES (training moves the prior, which steers MCTS differently); on frodo
you are measuring the real magnitudes and the per-decision FLOP cost.

**Record results** by appending a dated entry to
`experiments/results/expert_iteration_rounds.md` (the running journal). The
existing 2026-06-18 entry has a placeholder line "record numbers here after
the first GPU run"; fill it in. Do not use the em-dash glyph in any file
(a soul plugin hook rejects writes that contain it; use commas, colons,
periods, or parentheses).

## Map of the work

- **Design spec:** `docs/superpowers/specs/2026-06-18-outer-loop-design.md`
- **Implementation plan:** `docs/superpowers/plans/2026-06-18-outer-loop.md`
- **Running journal (append results here):** `experiments/results/expert_iteration_rounds.md`
- **Modules added/changed this branch:**
  - `src/verifier_anchored_rl/mcts.py`: `value_fn` leaf hook (additive)
  - `src/verifier_anchored_rl/planner_actor.py` (new): `StateActor`, `Encoder` protocol, `PrefillCache`
  - `src/verifier_anchored_rl/llm_context_mdp.py`: additive `encoder`/`prefill_cache` generation path
  - `src/verifier_anchored_rl/outer_loop.py` (new): `collect_iteration`, `run_outer_loop`, `IterationStats`
  - `experiments/run_outer_loop.py` (new): the frodo harness
  - tests: `test_mcts.py`, `test_planner_actor.py`, `test_outer_loop.py`, `test_llm_context_mdp.py`, `fakes.py`

## Key design decisions (so you do not have to re-derive them)

- **MCTS leaf value = target-Q (Algorithm 1).** `MCTSConfig.value_fn` is a
  new optional hook; `value_fn=None` is byte-identical to the old rollout
  path, so the section-10 triangulation tests are untouched. `StateActor.leaf_value`
  supplies `max_a' Q_target(s', a')`.
- **MCTS prior = online policy head.** `StateActor.prior` (masked softmax),
  wired as `MCTSConfig.prior_fn`.
- **One forward per node.** A state-prefix-keyed `PrefillCache` (shared by
  the heads path and the MDP generation path) makes encode-once emergent
  across MCTS's interleaved prior/value/generation calls. The cache is
  cleared in place at the start of each outer iteration so the per-iteration
  FLOP counters are honest.
- **Trajectory control:** each trajectory advances by SAMPLING the next verb
  from `pi_MCTS` at temperature `tau` (not greedy argmax), for buffer
  diversity. `pi_MCTS` at the same `tau` is the distillation target.
- **Encoder coupling:** the MDP's encoder path is additive. With no encoder
  it uses `llm_client.complete_json` unchanged (Mock/Echo/HTTP). The frodo
  harness shares ONE `SharedForwardEncoder` (built from the in-process
  `TransformersLLMClient`) and ONE `PrefillCache` between the MDP and the
  `StateActor`, so generation and the heads reuse the same prefill.
- **The loop-closes test** (`tests/test_outer_loop.py`) is the load-bearing
  one: it proves training MOVES the prior (validated across torch seeds
  {0,1,2,3,42}) and that the learned prior steers search differently from
  uniform. `L_Q`-decrease is covered separately in `test_train.py`.

## Open items / next moves

1. **First GPU run + journal numbers** (the immediate task above).
2. **Hyperparameter tuning** on the validation split of the smallest
   benchmark before headline runs: `gamma` in [0.99, 1.0], `c_puct` in
   [1.0, 1.5] (higher early), `tau` annealed near 1 then below, target-sync
   interval `N` ~1000 steps (`method.tex` 3.7).
3. **FLOP accounting refinement:** the per-iteration counters are a thin
   layer over the encoder's prefill/gen token counts; the compute-vs-accuracy
   frontier in the experimental plan builds on these.
4. **Deferred (not in scope yet):** `prompt_strategy="focused"` ablation
   (currently `NotImplementedError`); AlphaZero-style sample-early /
   argmax-late move selection (v1 uses one `tau`).

## Syncing results back to bilbo

After a run on frodo, commit and push so bilbo can pull:

```bash
git add experiments/results/expert_iteration_rounds.md
git commit -m "frodo: first outer-loop run results"   # end the message with the repo's standard trailers
git push
```
