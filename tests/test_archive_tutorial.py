"""The documented small example runs without a model or network."""

import math
import socket

from experiments.run_expert_iteration import main


def test_documented_tutorial_command(monkeypatch, capsys):
    def denied(*args, **kwargs):
        raise AssertionError("tutorial attempted network access")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    result = main([
        "--iterations", "2", "--episodes", "4", "--simulations", "8",
        "--evaluation-episodes", "4", "--max-steps", "20",
        "--evaluation-max-steps", "20", "--seed", "2026",
    ])
    assert result["iterations"] == 2
    assert result["training_examples"] > 0
    assert all(math.isfinite(result[k]) for k in ("optimal_value", "fast_return", "search_return"))
    output = capsys.readouterr().out
    assert "Fast mode (B = 0)" in output
    assert "Search mode (B = 8)" in output


def test_tutorial_rejects_empty_training():
    import pytest
    with pytest.raises(SystemExit):
        main(["--episodes", "0"])
