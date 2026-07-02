import io
import json

import numpy as np
import pytest

pytest.importorskip("sympy")

from verifier_anchored_rl.deliberation_mdp import (
    DeliberationAction, DeliberationMDP, Verb,
)
from verifier_anchored_rl.llm_client import EchoLLMClient, MockLLMClient
from verifier_anchored_rl.llm_controller import LLMController, run_episode


def test_controller_choice_is_constrained_to_legal_menu():
    client = MockLLMClient(script={
        ("CONTROLLER_CHOICE",): {"action": "form_hypothesis:0"}})
    mdp = DeliberationMDP(problem="p?", expected_answer="42", depth_limit=4,
                          llm_client=EchoLLMClient())
    controller = LLMController(client)
    action = controller.choose(mdp, mdp.initial_state())
    assert action == DeliberationAction(Verb.FORM_HYPOTHESIS, 0)


def test_controller_falls_back_to_first_legal_on_bad_choice():
    client = MockLLMClient(script={
        ("CONTROLLER_CHOICE",): {"action": "commit:999"}})   # not on the menu
    mdp = DeliberationMDP(problem="p?", expected_answer="42", depth_limit=4,
                          llm_client=EchoLLMClient())
    controller = LLMController(client)
    state = mdp.initial_state()
    action = controller.choose(mdp, state)
    assert action == mdp.actions(state)[0]


def test_run_episode_terminates_and_logs_jsonl():
    mdp = DeliberationMDP(problem="p?", expected_answer="42", depth_limit=3,
                          llm_client=EchoLLMClient())
    controller = LLMController(EchoLLMClient())   # picks the first menu entry
    buf = io.StringIO()
    summary = run_episode(mdp, controller, np.random.default_rng(0),
                          log_fn=lambda rec: buf.write(json.dumps(rec) + "\n"))
    assert summary.turns <= 3
    lines = [json.loads(l) for l in buf.getvalue().splitlines()]
    assert len(lines) == summary.turns
    for rec in lines:
        assert {"problem", "turn", "state_render", "legal", "chosen",
                "reward", "terminal"} <= set(rec)
    assert lines[-1]["terminal"] is True
