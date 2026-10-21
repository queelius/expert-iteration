"""Convert approved Rung-1 trajectory logs into Phase B decision rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verifier_anchored_rl.phase_b import convert_runs, dataset_summary, write_jsonl


DEFAULT_RUNS = (
    Path("experiments/runs/2026-07-02-rung1-casework-probe20-v6-treeview.jsonl"),
    Path("experiments/runs/2026-07-03-rung1-casework-probe20-v7.jsonl"),
    Path("experiments/runs/2026-07-03-rung1-casework-E11-gpt41-maxtok192.jsonl"),
    Path("experiments/runs/2026-07-03-rung1-casework-E13-solveclaims-on.jsonl"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=list(DEFAULT_RUNS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()

    rows = convert_runs(args.inputs, gamma=args.gamma)
    count = write_jsonl(rows, args.out)
    summary = dataset_summary(rows)
    if count != summary["decisions"]:
        raise RuntimeError("written row count does not match summary")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
