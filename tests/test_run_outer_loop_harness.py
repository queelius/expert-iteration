"""Light harness test: a real run dir with metadata, progress log, and a
metrics line is produced. The outer loop itself is monkeypatched to a stub so
this stays CPU-only and fast (the real loop is covered in test_outer_loop)."""

from __future__ import annotations

import json


def test_harness_builds_run_dir_and_writes_metrics(tmp_path, monkeypatch):
    import experiments.run_outer_loop as harness
    from verifier_anchored_rl.outer_loop import IterationStats

    # Stub the heavy stack construction (returns the 8 values main unpacks).
    monkeypatch.setattr(harness, "_build_stack",
                        lambda args: ("ENC", "SA", "TR", "BUF", "ACTOR", "OPT", "LLM", "CACHE"))
    # Stub the checkpoint-saver factory to a no-op closure.
    monkeypatch.setattr(harness, "_save_checkpoint_for", lambda *a, **k: (lambda state: None))

    def fake_loop(*a, **kw):
        kw["on_iteration_end"](IterationStats(iteration=1, n_decisions=3,
                                              n_simulations=18, buffer_size=3,
                                              total_first=2.0, total_last=1.5))
        kw["save_checkpoint"]({"phase": "training", "iteration": 1,
                               "problems_done_this_iter": 1})
        return [IterationStats(iteration=1, n_decisions=3, n_simulations=18,
                               buffer_size=3, total_first=2.0, total_last=1.5)]

    monkeypatch.setattr(harness, "run_outer_loop", fake_loop)

    run_root = tmp_path / "runs"
    rc = harness.main([
        "--iterations", "1", "--problems", "multiply_small",
        "--run-root", str(run_root), "--tag", "unit",
    ])
    assert rc == 0
    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    rd = run_dirs[0]
    assert (rd / "metadata.json").exists()
    assert (rd / "metrics.jsonl").exists()
    meta = json.loads((rd / "metadata.json").read_text())
    assert meta["status"] == "completed"
    metrics = (rd / "metrics.jsonl").read_text().strip().splitlines()
    assert json.loads(metrics[0])["iteration"] == 1


def test_harness_records_backend_and_embedder_selectors(tmp_path, monkeypatch):
    import json
    import experiments.run_outer_loop as harness
    from verifier_anchored_rl.outer_loop import IterationStats

    monkeypatch.setattr(harness, "_build_stack",
        lambda args: ("ENC", "SA", "TR", "BUF", "ACTOR", "OPT", "LLM", "CACHE"))
    monkeypatch.setattr(harness, "_save_checkpoint_for", lambda *a, **k: (lambda state: None))
    monkeypatch.setattr(harness, "run_outer_loop",
        lambda *a, **kw: (kw["on_iteration_end"](IterationStats(
            iteration=1, n_decisions=1, n_simulations=4, buffer_size=1,
            total_first=1.0, total_last=1.0)) or
            [IterationStats(iteration=1, n_decisions=1, n_simulations=4,
                            buffer_size=1, total_first=1.0, total_last=1.0)]))

    run_root = tmp_path / "runs"
    rc = harness.main([
        "--iterations", "1", "--problems", "multiply_small",
        "--run-root", str(run_root), "--tag", "selectors",
        "--gen-backend", "transformers", "--embedder", "hidden_state",
        "--collect-workers", "4", "--vllm-url", "http://x:9/v1",
    ])
    assert rc == 0
    rd = list(run_root.iterdir())[0]
    meta = json.loads((rd / "metadata.json").read_text())
    assert meta["args"]["gen_backend"] == "transformers"
    assert meta["args"]["embedder"] == "hidden_state"
    assert meta["args"]["collect_workers"] == 4
    assert meta["args"]["vllm_url"] == "http://x:9/v1"
