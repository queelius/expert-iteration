from __future__ import annotations


def test_capture_provenance_has_required_keys_and_tolerates_missing_torch():
    from verifier_anchored_rl.experiment import capture_provenance
    prov = capture_provenance(["run.py", "--seed", "0"], seeds={"global": 0})
    for key in ("git", "command", "seeds", "env", "host", "timestamps"):
        assert key in prov, key
    assert prov["command"]["argv"] == ["run.py", "--seed", "0"]
    assert prov["seeds"] == {"global": 0}
    assert "python" in prov["env"]
    assert isinstance(prov["env_freeze_sha256"], str)
    # extra is merged at top level
    prov2 = capture_provenance(["x"], seeds={}, extra={"model": "m", "embedder": {"kind": "text"}})
    assert prov2["model"] == "m"
    assert prov2["embedder"] == {"kind": "text"}


def test_provenance_from_metadata_maps_existing_run():
    from verifier_anchored_rl.experiment import provenance_from_metadata
    meta = {
        "git_sha": "abc123", "git_dirty": False,
        "args": {"seed": 7, "model": "Qwen/Qwen2.5-3B-Instruct",
                 "embedder": "text", "embedder_model": "Qwen/Qwen2.5-0.5B-Instruct"},
        "torch": "2.9.1", "transformers": "5.0", "gpu": "RTX 3060", "cuda": "12.8",
        "seed": 7,
    }
    prov = provenance_from_metadata(meta)
    assert prov["git"]["sha"] == "abc123"
    assert prov["seeds"]["global"] == 7
    assert prov["model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert prov["embedder"]["kind"] == "text"
    assert prov["embedder"]["id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert prov["env"]["torch"] == "2.9.1"


def test_validate_raw_row_accepts_valid_and_rejects_bad():
    import pytest
    from verifier_anchored_rl.experiment import validate_raw_row
    good = dict(dataset="gsm8k", dataset_version="main", split="test",
                problem_id="gsm8k-test-0", seed=0, budget_B=8, correct=True,
                pred="72", gold="72", verifier_verdict="pass", n_turns=4,
                gen_calls=31, gen_tokens=980, prefill_calls=12,
                prefill_tokens=3120, flops_est=None, wall_s=3.4,
                run_id="20260623T0-gsm8k")
    validate_raw_row(good)  # no raise
    missing = {k: v for k, v in good.items() if k != "gold"}
    with pytest.raises(ValueError):
        validate_raw_row(missing)
    bad_type = dict(good, budget_B="eight")
    with pytest.raises(ValueError):
        validate_raw_row(bad_type)


def test_experiment_creates_dir_manifest_index_and_is_idempotent(tmp_path):
    import json
    from verifier_anchored_rl.experiment import Experiment
    root = str(tmp_path)
    e = Experiment("e1-demo", "sharpening rises with budget", root=root,
                   provenance={"git": {"sha": "abc"}}, date="20260623")
    assert e.exp_dir == tmp_path / "exp" / "20260623-e1-demo"
    manifest = json.loads((e.exp_dir / "manifest.json").read_text())
    assert manifest["slug"] == "e1-demo"
    assert manifest["status"] == "running"
    assert (e.exp_dir / "EXPERIMENT.md").exists()
    # index has exactly one row for this slug, even after re-construction
    Experiment("e1-demo", "sharpening rises with budget", root=root,
               provenance={"git": {"sha": "abc"}}, date="20260623")
    rows = [json.loads(l) for l in (tmp_path / "index.jsonl").read_text().splitlines() if l.strip()]
    assert sum(r["slug"] == "e1-demo" for r in rows) == 1


def test_experiment_add_run_dedups_and_append_raw_validates(tmp_path):
    import json
    import pytest
    from verifier_anchored_rl.experiment import Experiment
    e = Experiment("e2", "h", root=str(tmp_path), date="20260623")
    e.add_run("20260622T0-sweep-B8")
    e.add_run("20260622T0-sweep-B8")  # dup
    manifest = json.loads((e.exp_dir / "manifest.json").read_text())
    assert manifest["runs"] == ["20260622T0-sweep-B8"]
    good = dict(dataset="gsm8k", dataset_version="main", split="test",
                problem_id="p0", seed=0, budget_B=8, correct=True, pred="1",
                gold="1", verifier_verdict="pass", n_turns=1, gen_calls=1,
                gen_tokens=1, prefill_calls=1, prefill_tokens=1, flops_est=None,
                wall_s=1.0, run_id="r0")
    e.append_raw(good)
    with pytest.raises(ValueError):
        e.append_raw({"dataset": "gsm8k"})  # invalid
    lines = (e.exp_dir / "raw" / "results.jsonl").read_text().splitlines()
    assert len(lines) == 1


def test_experiment_does_not_clobber_edited_experiment_md(tmp_path):
    from verifier_anchored_rl.experiment import Experiment
    e = Experiment("e3", "h", root=str(tmp_path), date="20260623")
    md = e.exp_dir / "EXPERIMENT.md"
    md.write_text("# e3\n\nMY EDITS\n")
    Experiment("e3", "h", root=str(tmp_path), date="20260623")  # re-construct
    assert "MY EDITS" in md.read_text()
