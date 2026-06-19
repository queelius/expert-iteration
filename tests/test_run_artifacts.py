"""Tests for run-directory artifacts: metadata, JSONL append, atomic
checkpoint save/load. Pure filesystem, no GPU."""

from __future__ import annotations

import json

import pytest


def test_create_run_dir_writes_metadata(tmp_path):
    from verifier_anchored_rl.run_artifacts import create_run_dir

    run_dir = create_run_dir(
        root=str(tmp_path), tag="smoke",
        metadata={"model": "fake", "seed": 0},
        timestamp="20260619T000000Z",
    )
    meta = json.loads((tmp_path / run_dir.name / "metadata.json").read_text())
    assert meta["model"] == "fake"
    assert meta["status"] == "running"
    assert "20260619T000000Z" in run_dir.name
    assert "smoke" in run_dir.name


def test_append_jsonl_appends_one_line_per_call(tmp_path):
    from verifier_anchored_rl.run_artifacts import append_jsonl

    path = tmp_path / "metrics.jsonl"
    append_jsonl(str(path), {"iteration": 1})
    append_jsonl(str(path), {"iteration": 2})
    lines = path.read_text().strip().splitlines()
    assert [json.loads(x)["iteration"] for x in lines] == [1, 2]


def test_atomic_save_load_checkpoint_round_trips(tmp_path):
    pytest.importorskip("torch")
    import torch
    from verifier_anchored_rl.run_artifacts import (
        save_checkpoint_atomic, load_checkpoint,
    )

    path = tmp_path / "checkpoint.pt"
    save_checkpoint_atomic(str(path), {"x": torch.tensor([1.0, 2.0]), "step": 7})
    assert not (tmp_path / "checkpoint.pt.tmp").exists()   # tmp cleaned up
    loaded = load_checkpoint(str(path))
    assert loaded["step"] == 7
    assert torch.equal(loaded["x"], torch.tensor([1.0, 2.0]))


def test_update_status_rewrites_metadata(tmp_path):
    from verifier_anchored_rl.run_artifacts import create_run_dir, update_status

    run_dir = create_run_dir(
        root=str(tmp_path), tag=None, metadata={"model": "fake"},
        timestamp="20260619T000000Z",
    )
    update_status(run_dir, "completed")
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["status"] == "completed"
    assert "end_time" in meta
