# Reproducing the Archive

Run commands from the repository root. Python 3.11 or newer is required.

## 1. Recalculate the Selected Numbers

No installation or credentials are needed:

```bash
python analysis/reproduce.py --output-dir /tmp/expert-iteration-tables
```

Expected: `Reconstructed 8 comparisons`, plus `tables.json` and `tables.md`.
Compare these with the checked-in `analysis/tables.json` and `analysis/tables.md`.
The manifest pins five JSONL input files by SHA-256.

This checks source-file identity, supported schemas, paired coverage, duplicate
observations, and selected record-level outcomes. E20 episode rewards are
summed from turns and compared with saved summaries. E41/E42 correctness comes
from transition/decision records. E47 selections are derived from discovery
repetitions and evaluated on the separate evaluation repetitions.

This is **metric reconstruction**, not complete scientific replay. It does
not independently recheck all mathematical answers, audit every historical
transport receipt, or prove the recorded labels correct. It introduces no new
confidence intervals. The complete research checkout retains deeper replay
code and diagnostics for further audit.

Missing files, modified bytes, unsupported schemas, and incomplete paired
records cause a nonzero exit, not a partial success table.

## 2. Run the Local Teaching Example

```bash
python -m pip install -e ".[dev]"
python experiments/run_expert_iteration.py \
  --iterations 2 --episodes 4 --simulations 8 \
  --evaluation-episodes 4 --max-steps 20 \
  --evaluation-max-steps 20 --seed 2026
python -m pytest -q tests/test_archive_reproduction.py tests/test_archive_tutorial.py
```

See the [tutorial](tutorial.md) for what the numbers mean.
Installation may require the network; the example and analysis do not.
The tests explicitly deny socket connections for the archive workloads.

## 3. Run the Wider Historical Tests

```bash
uv sync --extra dev --extra llm
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch
uv run pytest -q
```

The symbolic and HTTP mock tests need the `llm` extra even when no real model
is called. Some neural tests skip without Torch, while some older tests fail
to handle its absence. The CPU-only install avoids downloading CUDA packages;
CUDA-only checks still need appropriate hardware. Some constraints tests also
need `lm-format-enforcer`. Record skips rather than calling them passes.
The [readiness record](release-readiness.md) reports the environment actually
tested for this archive, not an old test count copied from a handoff. The wider
suite is not currently certified green: three older symbolic-replay tests hit
a two-minute per-test limit during archive preparation. The compact archive
tests pass independently of that subsystem.

For focused coverage:

```bash
uv run --with pytest-cov pytest -q \
  tests/test_archive_reproduction.py tests/test_archive_tutorial.py \
  --cov=analysis.reproduce --cov=experiments.run_expert_iteration \
  --cov-branch --cov-report=term-missing
```

## 4. Optional Hosted Replication

Hosted reruns are not necessary to check the tables and are not launched by
any archive command. Historical runner settings, prompts, model names,
provider routes, and schema versions are recorded in the original run files.

A hosted model name or requested revision is not proof that a provider served
identical weights. Endpoints, prices, sampling behavior, and model
availability can change. A new run is a replication under its recorded
conditions, not guaranteed byte-for-byte reproduction of the old run.

Do not copy a historical balance estimate or spending permission into a new
run. New research needs its own question, comparison, budget, and record.

## Release Candidate

The local packet is deliberately narrower than the entire working repository.
It includes the selected outcomes, analysis, tutorial, source package, and
focused tests. It does not claim to include every historical hosted diagnostic.
The full research history remains in the source repository.

After committing the reviewed source, build a deterministic local packet:

```bash
python analysis/build_packet.py --revision COMMIT_SHA \
  --output dist/expert-iteration-candidate.zip
```

Replace `COMMIT_SHA` with the exact reviewed commit. The builder reads files
from that commit, not from dirty working-tree contents. It refuses to overwrite
an existing output. `ARCHIVE-MANIFEST.json` inside the ZIP records the source
commit and SHA-256 of every included source/evidence file.

Building a packet neither publishes it nor completes its rights/privacy review.
