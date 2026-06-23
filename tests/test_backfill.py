from __future__ import annotations

from pathlib import Path


def test_backfill_wraps_existing_run_into_experiment(tmp_path):
    import json
    from experiments.backfill_experiment import backfill
    # a minimal existing run dir with metadata.json
    runs = tmp_path / "runs"; runs.mkdir()
    run = runs / "20260622T0-sweep-B8"; run.mkdir()
    (run / "metadata.json").write_text(json.dumps({
        "git_sha": "abc", "args": {"seed": 0, "model": "m", "embedder": "text",
                                    "embedder_model": "e"}, "torch": "2.9"}))
    exp_dir = backfill(str(run), "e1-hT-sweep", "sharpening rises with budget",
                       root=str(tmp_path))
    m = json.loads((Path(exp_dir) / "manifest.json").read_text())
    assert m["runs"] == ["20260622T0-sweep-B8"]
    assert m["provenance"]["model"] == "m"
    assert (tmp_path / "INDEX.md").exists()
