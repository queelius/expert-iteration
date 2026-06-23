# Robust experimental framework (design)

Date: 2026-06-23. Status: draft for review.

## Motivation

The per-run layer is solid: every `experiments/results/runs/<ts>-<tag>/` carries
`metadata.json` (argv, git sha, env, GPU, seed), `metrics.jsonl`,
`records.jsonl` (per-decision raw), `progress.log`, `summary.md`, an atomic
`checkpoint.pt`, and an idempotent `holdout_eval.json`. What is missing is
everything ABOVE a single run:

1. No experiment container. A sweep emits N runs but nothing ties them together
   except the hand-written journal (`expert_iteration_rounds.md`). There is no
   machine-readable manifest recording the hypothesis, the exact commands, the
   runs involved, and the headline result.
2. Analysis is ad-hoc. The five existing analysis scripts
   (`b_sweep_stability.py`, `b_sweep_analyze.py`, `analyze_distillation_rounds.py`,
   `algebra_post_mortem.py`, `make_pilot_figure.py`) each reimplement
   `entropy_bits` / `argmax` / `mean_pairwise_l1` / `load_jsonl`, two hardcode
   `/tmp/` input paths, one bakes results inline, and NONE persist a markdown
   report. Outputs land in inconsistent places (`/tmp`, `paper/figures`, stdout).
3. No program-level index to navigate the research arc.
4. No benchmark registry or standardized raw-results schema for GSM8K/MATH, which
   the AAAI campaign requires.

Bugs and lost provenance here are costly: they force experiment re-runs. The
framework makes provenance, indexing, and the raw-results schema enforced by a
thin library, and the writeup/analysis a templated, re-runnable step.

## Goal

A three-layer experimental framework:
- Run (unchanged): one training/eval invocation and its provenance dir.
- Experiment (new): a container tying a set of runs to one hypothesis, with a
  machine-readable manifest, a human `EXPERIMENT.md`, consolidated raw results, a
  reproducible `analyze.py` emitting `analysis/{report.md, figures/, tables/}`,
  and a `reproduce.sh`.
- Program index (new): `INDEX.md` + `index.jsonl` listing every experiment with
  status and headline result.

Plus a thin library that captures provenance, writes/validates the manifest and
index, defines the raw-results schema, and provides the shared analysis
primitives the five scripts currently duplicate; a benchmark registry with a
GSM8K loader and metric; and markdown templates.

Decisions locked with the user (2026-06-23): hybrid formalization (thin library +
templates, not a full enforced framework, not convention-only); build the
framework + benchmark registry + GSM8K now (MATH next; baseline-comparison
harness later).

## Non-goals

- MATH loader and the baseline-comparison harness (best-of-N, self-consistency,
  PRM, etc.): next increments, designed to slot into the same registry/raw schema.
- A web dashboard or DB. Markdown + JSONL on disk is the store.
- Changing the learning algorithm, MDP, reward, or the vLLM pipeline.
- Re-deriving the metric math. The framework adopts the audited definitions of
  entropy and concentration (see the concurrent correctness audit) as the single
  source of truth in `analysis_lib`, so any audit fix lands in exactly one place.

## Architecture

### Layer 1: Run (unchanged)

`experiments/results/runs/<ts>-<tag>/` as today. The framework only READS run
dirs (by id); it never moves them (checkpoints are large and resume paths must
stay stable).

### Layer 2: Experiment

```
experiments/results/exp/<date>-<slug>/
  EXPERIMENT.md     # hypothesis, motivation, design, method, commands, results, interpretation, conclusion
  manifest.json     # machine-readable provenance + run links + dataset ids + seeds + status
  reproduce.sh      # exact commands to regenerate every run (and the analysis)
  raw/results.jsonl # consolidated benchmark outcomes across runs (one row per problem/seed/budget)
  analysis/
    analyze.py      # deterministic: reads runs/ + raw/ -> writes report.md, figures/, tables/
    report.md       # generated in-depth writeup (committed)
    figures/        # generated (PNG for repo, PDF mirror for paper)
    tables/         # generated CSV/JSON
```

### Layer 3: Program index

```
experiments/results/
  INDEX.md          # human: one row per experiment {slug, hypothesis, status, headline, date}
  index.jsonl       # machine: append/update one object per experiment
  benchmarks/
    registry.json   # dataset registry (below)
    cache/          # downloaded/derived splits (gitignored if large; checksum in registry)
```

## The thin library

New module `src/verifier_anchored_rl/experiment.py` (importable; unit-testable
without GPU), plus `experiments/analysis_lib.py` for analysis primitives.

### Provenance capture

```python
def capture_provenance(argv: list[str], *, seeds: dict, extra: dict | None = None) -> dict:
    """All facts needed to reproduce: returns a dict with
      git: {sha, dirty, branch}
      command: the full argv (and a shell-quoted string)
      seeds: {global, base_seed, ...}
      env: {python, numpy, torch, transformers, vllm, platform}
      env_freeze_sha256: hash of `pip freeze` output (the full text saved alongside)
      host: {hostname, gpu_name, driver, cuda}
      model: {id, dtype, device}
      embedder: {kind, id, dim}
      reward_constants: {terminal_success, terminal_failure, intermediate_verifier, masked_action}
      timestamps: {created}
    """
```

The run harness already records most of this in `metadata.json`. Rather than
refactor the harness now, a shim (`provenance_from_metadata(meta) -> dict`) maps
an existing run's `metadata.json` into the unified schema and fills the missing
fields (pip-freeze hash, embedder block, reward constants, branch) where
derivable, so experiments can ingest today's runs immediately. A later increment
makes the harness emit `capture_provenance(...)` natively; both produce the same
schema.

### Experiment object

```python
class Experiment:
    def __init__(self, slug: str, hypothesis: str, *, root="experiments/results"): ...
    # creates exp/<date>-<slug>/, writes manifest.json (status="running"),
    # upserts a row into index.jsonl.

    def add_run(self, run_dir: str | Path) -> None: ...
        # records the run id in manifest.runs[]; does not move the run.

    def append_raw(self, row: dict) -> None: ...
        # validates row against RAW_SCHEMA, appends to raw/results.jsonl.

    def set_headline(self, text: str, status: str) -> None: ...
        # status in {running, complete, abandoned}; updates manifest + index.

    def render_experiment_md(self, **fields) -> None: ...
        # fills EXPERIMENT.md from the template if absent (never clobbers edits).
```

`manifest.json` schema:

```json
{
  "slug": "e1-budget-sweep-fixed",
  "date": "2026-06-23",
  "hypothesis": "closed-loop sharpening strengthens with search budget B",
  "status": "complete",
  "headline": "concentration rises 0.04->0.11->0.12 across B=8,16,32",
  "provenance": { "...capture_provenance()..." },
  "runs": ["20260622T194410Z-sweep-fixed-B8", "..."],
  "datasets": [{"id": "toy-train", "version": "1", "split": "train"}],
  "seeds": [0],
  "analysis": {"script": "analysis/analyze.py", "last_run": null, "code_sha": null}
}
```

`index.jsonl` row: `{slug, date, hypothesis, status, headline, exp_dir}` (upserted
by slug).

### Raw-results schema (benchmark outcomes)

One JSONL row per evaluated cell, the single source for accuracy, pass@k, and the
compute-vs-accuracy frontier:

```json
{
  "dataset": "gsm8k", "dataset_version": "main", "split": "test",
  "problem_id": "gsm8k-test-0042", "seed": 0, "budget_B": 8,
  "correct": true, "pred": "72", "gold": "72", "verifier_verdict": "pass",
  "n_turns": 4, "gen_calls": 31, "gen_tokens": 980,
  "prefill_calls": 12, "prefill_tokens": 3120, "flops_est": null,
  "wall_s": 3.4, "run_id": "20260623T...-gsm8k-3b-vllm"
}
```

`RAW_SCHEMA` is a dict checked by `append_raw` (required keys + types); a missing
or mistyped field raises rather than silently writing a bad row.

### Shared analysis primitives

`experiments/analysis_lib.py` holds ONE implementation of each duplicated
helper, imported by all analysis code:

```python
def load_jsonl(path) -> list[dict]
def entropy_bits(dist: dict | list) -> float           # the audited definition
def normalized_concentration(dist, k: int) -> float    # 1 - H/log2(k), k>1 only
def argmax(dist: dict) -> str
def mean_pairwise_l1(dists: list[dict]) -> float
def bootstrap_ci(values, statistic, *, resamples=10000, ci=0.95) -> tuple
def sharpness_report(records, label) -> dict           # from analyze_distillation_rounds
def render_figure(plot_fn, out_path) -> None           # Agg backend, PNG+PDF, bbox_inches tight
def write_report(out_dir, sections: list[tuple[str, str|dict]]) -> Path  # tables+prose -> report.md
```

`entropy_bits` / `normalized_concentration` here are the same definitions used in
`outer_loop.py`; the goal is one canonical metric module both import, so the
sharpening number is computed identically online and in analysis.

## Benchmark registry + GSM8K

`experiments/benchmarks/registry.json`:

```json
{
  "gsm8k":  {"loader": "gsm8k", "version": "main", "splits": ["train","test"], "n_test": 1319, "answer_style": "final-number", "citation": "Cobbe et al. 2021"},
  "math":   {"loader": "math",  "version": "v1",   "splits": ["train","test"], "answer_style": "boxed", "citation": "Hendrycks et al. 2021", "status": "planned"},
  "toy-train": {"loader": "toy", "version": "1", "splits": ["train","holdout"], "source": "experiments/problems.py"}
}
```

`experiments/benchmarks/loaders.py`:
- `load_gsm8k(split)` -> list of `{problem_id, problem, gold}` (gold = the number
  after `####`). Sourced from the HF `datasets` cache or a checked-in/cached
  JSONL; the registry stores the version and a checksum so a run pins the exact
  data. `datasets` is an optional dep (`pip install -e .[bench]`).
- `gsm8k_correct(pred, gold) -> bool`: numeric exact-match via the existing SymPy
  verifier (normalize commas/units, compare numerically). Reuses `verifier.py`.

The benchmark eval generalizes `evaluate_holdout.py`'s rollout: for each
`(problem, seed, budget_B)` cell, roll out the trained policy to a terminal
state, extract the committed answer, score with the dataset metric, and
`experiment.append_raw(row)`. Idempotent per cell (skip rows already in
`raw/results.jsonl`), matching `holdout_eval.json`'s current behavior.

## Templates

`experiments/templates/EXPERIMENT.md`:

```markdown
# <slug>

Date: <date>  Status: <status>

## Hypothesis
<one falsifiable sentence>

## Motivation
<why this experiment, what prior result motivates it>

## Design
Datasets: <ids+versions+splits>. Seeds: <list>. Budgets B: <list>.
Substrate/model: <id>. Embedder: <kind/id/dim>. Key hyperparameters: <...>.

## Commands
See reproduce.sh. (Exact argv per run, also in manifest.provenance.)

## Results
<filled from analysis/report.md: the headline table + figure references>

## Interpretation
<what the numbers mean for the claim; threats to validity>

## Conclusion
<confirmed / refuted / inconclusive; what is next>
```

`experiments/templates/analyze.py` skeleton: imports `analysis_lib`, reads the
experiment's `runs[]` (metrics.jsonl) and `raw/results.jsonl`, computes the
standard tables, calls `render_figure` and `write_report`, stamps the code sha
and consumed run ids into `manifest.analysis`.

`experiments/templates/reproduce.sh`: the exact per-run commands plus the final
`python exp/<slug>/analysis/analyze.py`.

## Migration

- Existing run dirs are untouched and remain valid (the schema is additive).
- The five analysis scripts are reimplemented as thin callers of `analysis_lib`
  (or kept but importing it), fixing: hardcoded `/tmp` paths become args;
  `make_pilot_figure.py` reads a generalization table instead of baked-in numbers;
  every one writes a `report.md` (no more stdout-only). They become the canonical
  `analyze.py` content for their respective experiments.
- A one-time backfill script wraps the existing notable runs (E1 sweep, the
  pilot) into `exp/` dirs with manifests pointing at their run ids, so the index
  is populated. The journal `expert_iteration_rounds.md` stays as the narrative
  log and links to experiment dirs.

## Reproducibility and determinism

- `reproduce.sh` + pinned dataset version + recorded seeds + git sha + pip-freeze
  hash = a re-runnable experiment. The pip-freeze TEXT is saved in the exp dir
  (not just the hash) so the env is reconstructable.
- Analysis is deterministic (no `Date.now`/random in figures; bootstrap CIs use a
  fixed seed recorded in the report).
- `analyze.py` records which run ids and which code sha produced the current
  `report.md`, so a stale analysis is detectable.

## Testing

CPU-only, no GPU:
- `capture_provenance` returns all required keys; tolerates missing git/torch.
- `Experiment` creates the dir, writes a schema-valid manifest, upserts exactly
  one index row on repeat construction, never clobbers an edited `EXPERIMENT.md`.
- `append_raw` rejects rows missing/mistyping required fields; accepts valid rows;
  is idempotent against an existing `raw/results.jsonl` (skip-seen).
- `analysis_lib` primitives: `entropy_bits` matches `outer_loop._entropy` on
  shared inputs (the single-source-of-truth check); `normalized_concentration`
  matches the harness; `bootstrap_ci` is deterministic under a fixed seed.
- `load_gsm8k` parses the `####` gold; `gsm8k_correct` handles commas/units and
  rejects wrong answers; both run on a tiny fixture (no network in tests).

## Resolved decisions (2026-06-23)

1. Library location: `src/verifier_anchored_rl/experiment.py` (importable
   package), so the run harness and analysis share one provenance schema.
2. Provenance migration: a shim maps the existing `metadata.json` into the
   unified schema NOW (lower risk); refactoring the run harness to emit the
   schema natively is deferred to a later increment.
3. Figures: emit BOTH a committed PNG (for the repo / report.md) and a PDF mirror
   (for the paper), via `render_figure`.

## Risks

- Scope creep into the full benchmark/baseline suite. Mitigation: GSM8K only now;
  MATH/baselines slot into the same registry and raw schema later.
- Duplicated metric definitions drifting. Mitigation: one canonical metric module
  imported by both `outer_loop.py` and `analysis_lib`.
- `datasets` dependency footprint. Mitigation: optional extra; cache to JSONL with
  a checksum so runs do not depend on live download.
