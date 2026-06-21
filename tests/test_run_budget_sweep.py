"""Resumability logic for the budget-sweep orchestrator (CPU, no model).

The sweep must skip budgets whose run already completed, resume a budget whose
run was interrupted, and start a fresh run otherwise. _commands_for is pure
(no subprocess), so the decision is testable directly.
"""

from __future__ import annotations

import argparse
import json


def _args(run_root, budgets=(8, 16, 32)):
    return argparse.Namespace(
        budgets=list(budgets), iterations=3, train_steps=100,
        problems=["train"], seed=0, run_root=str(run_root),
        tag_prefix="sweep-budget",
    )


def _make_run(run_root, tag, status):
    d = run_root / f"20260621T000000Z-{tag}"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"status": status}))
    return d


def test_sweep_skips_completed_resumes_interrupted_starts_fresh(tmp_path):
    from experiments.run_budget_sweep import _commands_for

    _make_run(tmp_path, "sweep-budget-B8", "completed")     # done -> skip
    rd16 = _make_run(tmp_path, "sweep-budget-B16", "failed")  # interrupted -> resume
    # B32 has no run dir -> fresh

    cmds = _commands_for(_args(tmp_path), tmp_path)
    assert len(cmds) == 3

    b8, b16, b32 = cmds
    assert b8[0] == "SKIP" and b8[1] == "B=8"
    assert "--resume" in b16 and str(rd16) in b16
    assert "--resume" not in b32 and "--tag" in b32
    # fresh run records a motivation via --notes
    assert "--notes" in b32 and "budget sweep" in b32[b32.index("--notes") + 1]
    # the budget is wired through as num-simulations
    assert b32[b32.index("--num-simulations") + 1] == "32"
