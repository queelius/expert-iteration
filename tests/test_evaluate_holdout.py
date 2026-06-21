"""CPU test of the held-out rollout logic (Echo MDP + FakeEncoder, no GPU)."""

from __future__ import annotations

import pytest


def _stack():
    from tests.fakes import FakeEncoder
    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.planner_actor import PrefillCache, StateActor

    enc = FakeEncoder(embed_dim=16)
    actor = HiddenStateActor(embed_dim=16)
    return StateActor(enc, actor, PrefillCache(), depth_limit=3)


def _mdp():
    from verifier_anchored_rl.llm_client import EchoLLMClient
    from verifier_anchored_rl.llm_context_mdp import LLMContextMDP

    return LLMContextMDP(problem_statement="Compute 7*6", expected_answer="42",
                         llm_client=EchoLLMClient(), depth_limit=3, max_turns=4)


@pytest.mark.parametrize("budget", [0, 2])
def test_rollout_terminates_and_reports_success_bool(budget):
    pytest.importorskip("torch")
    import numpy as np
    from experiments.evaluate_holdout import rollout

    res = rollout(_mdp(), _stack(), budget=budget, c_puct=1.5, discount=1.0,
                  max_turns=4, rng=np.random.default_rng(0))
    assert set(res) == {"committed", "success", "turns", "total_reward"}
    assert isinstance(res["success"], bool)
    assert 0 <= res["turns"] <= 4
