"""Wrap an existing run directory into an experiment container, so notable
prior runs (the E1 sweep, the pilot) populate the program index without
re-running them."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from verifier_anchored_rl.experiment import Experiment, provenance_from_metadata, rebuild_index_md


def backfill(run_dir: str, slug: str, hypothesis: str, *,
             root: str = "experiments/results") -> str:
    meta = json.loads((Path(run_dir) / "metadata.json").read_text())
    exp = Experiment(slug, hypothesis, root=root,
                     provenance=provenance_from_metadata(meta))
    exp.add_run(run_dir)
    rebuild_index_md(root)
    return str(exp.exp_dir)


if __name__ == "__main__":
    print(backfill(sys.argv[1], sys.argv[2], sys.argv[3]))
