# tests/test_analysis_lib.py
from __future__ import annotations

import math


def test_entropy_bits_matches_canonical_and_concentration():
    from experiments.analysis_lib import entropy_bits, normalized_concentration
    from verifier_anchored_rl.outer_loop import _entropy
    uniform4 = [0.25, 0.25, 0.25, 0.25]
    assert abs(entropy_bits(uniform4) - 2.0) < 1e-9
    assert entropy_bits(uniform4) == _entropy(uniform4)  # single source of truth
    assert abs(normalized_concentration(2.0, 4) - 0.0) < 1e-9
    assert abs(normalized_concentration(0.0, 4) - 1.0) < 1e-9


def test_argmax_and_mean_pairwise_l1():
    from experiments.analysis_lib import argmax, mean_pairwise_l1
    assert argmax({"a": 0.1, "b": 0.9}) == "b"
    d1 = {"a": 1.0, "b": 0.0}
    d2 = {"a": 0.0, "b": 1.0}
    assert abs(mean_pairwise_l1([d1, d2]) - 2.0) < 1e-9
    assert mean_pairwise_l1([d1]) == 0.0


def test_bootstrap_ci_is_deterministic_under_seed():
    from experiments.analysis_lib import bootstrap_ci
    vals = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    lo1, hi1 = bootstrap_ci(vals, sum, seed=0)
    lo2, hi2 = bootstrap_ci(vals, sum, seed=0)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 <= hi1


def test_load_jsonl_and_sharpness_report(tmp_path):
    import json
    from experiments.analysis_lib import load_jsonl, sharpness_report
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps({"pi_mcts": pi}) for pi in
                           ({"a": 1.0, "b": 0.0}, {"a": 0.5, "b": 0.5})) + "\n")
    recs = load_jsonl(str(p))
    assert len(recs) == 2
    rep = sharpness_report(recs, label="round-a")
    assert rep["label"] == "round-a"
    assert abs(rep["mean_max_prob"] - 0.75) < 1e-9  # (1.0 + 0.5) / 2
