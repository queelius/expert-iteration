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


def test_seed_everything_makes_actor_init_reproducible():
    torch = pytest.importorskip("torch")
    from verifier_anchored_rl.run_artifacts import seed_everything
    from verifier_anchored_rl.architecture import HiddenStateActor

    def _first_linear_weight(actor):
        # Skip LayerNorm weight (constant init, all ones) and bias parameters;
        # grab the first nn.Linear weight tensor, which draws from the RNG.
        for name, p in actor.named_parameters():
            if "weight" in name and p.dim() >= 2:
                return p.clone()
        raise AssertionError("No 2D weight parameter found in HiddenStateActor")

    seed_everything(0)
    actor_a = HiddenStateActor(embed_dim=16)
    param_a = _first_linear_weight(actor_a)

    seed_everything(0)
    actor_b = HiddenStateActor(embed_dim=16)
    param_b = _first_linear_weight(actor_b)

    assert torch.equal(param_a, param_b), "Same seed must yield identical init weights"

    seed_everything(99)
    actor_c = HiddenStateActor(embed_dim=16)
    param_c = _first_linear_weight(actor_c)
    assert not torch.equal(param_a, param_c), "Different seed must yield different weights"


def test_write_json_atomic_roundtrip_and_guarded(tmp_path):
    from verifier_anchored_rl.run_artifacts import write_json_atomic

    path = tmp_path / "meta.json"
    obj = {"status": "running", "seed": 42, "nested": {"a": [1, 2, 3]}}

    write_json_atomic(path, obj)
    assert not (tmp_path / "meta.json.tmp").exists(), ".tmp file must not remain"
    loaded = json.loads(path.read_text())
    assert loaded == obj, "Round-trip must preserve all fields"

    # Writing twice to the same path replaces cleanly.
    obj2 = {"status": "completed", "seed": 42}
    write_json_atomic(path, obj2)
    assert not (tmp_path / "meta.json.tmp").exists()
    assert json.loads(path.read_text()) == obj2
