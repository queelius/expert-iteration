"""Resumable budget sweep for the closed-loop expert-iteration run (experiment E1).

Motivation. At B=8 the closed loop sharpens for one round, then the signal is
noisy and non-monotone (target instability at low budget; see the journal entry
of 2026-06-21). The pilot's offline B-sweep showed that the search budget is the
lever for stable distillation targets, with convergence only near B=160. This
sweep runs the *closed online loop* across a range of budgets to test whether
sustained, monotone sharpening strengthens with B.

Resumable at two levels. Each per-budget run is itself a resumable outer-loop
run (run_outer_loop.py writes a per-trajectory checkpoint and accepts --resume).
This orchestrator skips budgets whose run already reached status "completed" and
resumes a budget whose run was interrupted (status "failed"/"running"), so
re-invoking it continues the sweep from where it stopped. Each per-budget run
records its motivation via --notes in metadata.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HARNESS = Path(__file__).with_name("run_outer_loop.py")


def _find_run_dir(run_root: Path, tag: str) -> Path | None:
    """The newest run directory for ``tag`` (run dirs are ``<timestamp>-<tag>``)."""
    cands = sorted(run_root.glob(f"*-{tag}"))
    return cands[-1] if cands else None


def _status(run_dir: Path) -> str | None:
    try:
        return json.loads((run_dir / "metadata.json").read_text()).get("status")
    except Exception:
        return None


def _commands_for(args, run_root: Path) -> list[list[str]]:
    """Build the per-budget run_outer_loop.py command list, applying the
    skip/resume/fresh decision. Pure (no subprocess), so it is unit-testable."""
    cmds: list[list[str]] = []
    for B in args.budgets:
        tag = f"{args.tag_prefix}-B{B}"
        rd = _find_run_dir(run_root, tag)
        st = _status(rd) if rd else None
        common = [
            "--iterations", str(args.iterations),
            "--num-simulations", str(B),
            "--train-steps", str(args.train_steps),
            "--problems", *args.problems,
            "--seed", str(args.seed),
            "--run-root", str(run_root),
        ]
        if st == "completed":
            cmds.append(["SKIP", f"B={B}", str(rd)])
        elif rd is not None:
            cmds.append([sys.executable, "-u", str(HARNESS), "--resume", str(rd)] + common)
        else:
            notes = (
                f"E1 budget sweep: does closed-loop sharpening strengthen with "
                f"search budget? B={B}. At B=8 one round sharpens then the signal "
                f"is noisy (target instability); pilot B-sweep showed convergence "
                f"near B=160."
            )
            cmds.append(
                [sys.executable, "-u", str(HARNESS), "--tag", tag, "--notes", notes]
                + common
            )
    return cmds


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--train-steps", type=int, default=100)
    p.add_argument("--problems", nargs="+", default=["train"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-root", default="experiments/results/runs")
    p.add_argument("--tag-prefix", default="sweep-budget")
    args = p.parse_args(argv)

    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    for cmd in _commands_for(args, run_root):
        if cmd[0] == "SKIP":
            print(f"[sweep] {cmd[1]}: already completed at {cmd[2]}, skipping",
                  flush=True)
            continue
        kind = "resuming" if "--resume" in cmd else "starting"
        print(f"[sweep] {kind}: {' '.join(cmd[3:])}", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"[sweep] run exited {rc}; stopping sweep (re-invoke to resume).",
                  flush=True)
            return rc
    print("[sweep] all budgets complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
