#!/usr/bin/env bash
# Regenerate every run in this experiment, then the analysis.
# Fill in one line per run (exact argv is also in manifest.provenance), e.g.:
# python experiments/run_outer_loop.py --gen-backend vllm --model <M> ... --tag <tag>
set -euo pipefail
python experiments/templates/analyze.py "$(dirname "$0")"
