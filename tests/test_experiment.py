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
