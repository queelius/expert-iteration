import io
import json

import numpy as np
import pytest

pytest.importorskip("sympy")

from verifier_anchored_rl.deliberation_mdp import (
    DeliberationAction, DeliberationMDP, Verb,
)
from verifier_anchored_rl.llm_client import EchoLLMClient, MockLLMClient
from verifier_anchored_rl.llm_controller import (
    CONTROLLER_SYSTEM, CONTROLLER_SYSTEM_STRUCTURED, LLMController, run_episode,
)


def test_controller_choice_is_constrained_to_legal_menu():
    client = MockLLMClient(script={
        ("CONTROLLER_CHOICE",): {"action": "form_hypothesis:0"}})
    mdp = DeliberationMDP(problem="p?", expected_answer="42", depth_limit=4,
                          llm_client=EchoLLMClient())
    controller = LLMController(client)
    action = controller.choose(mdp, mdp.initial_state())
    assert action == DeliberationAction(Verb.FORM_HYPOTHESIS, 0)
    assert controller.last_choice_fallback is False


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
                          log_fn=lambda rec: buf.write(json.dumps(rec) + "\n"),
                          episode_id=7, meta={"backend": "echo"})
    assert summary.turns <= 3
    lines = [json.loads(l) for l in buf.getvalue().splitlines()]
    decisions = [rec for rec in lines if not rec.get("summary")]
    assert len(decisions) == summary.turns
    for rec in decisions:
        assert {"episode", "problem", "turn", "state_render", "legal",
                "chosen", "reward", "terminal", "fallback"} <= set(rec)
        assert rec["episode"] == 7
    assert decisions[-1]["terminal"] is True
    summary_rec = lines[-1]
    assert summary_rec["summary"] is True
    assert summary_rec["episode"] == 7
    assert {"total_reward", "turns", "llm_calls", "meta"} <= set(summary_rec)
    assert summary_rec["turns"] == summary.turns
    assert summary_rec["llm_calls"] == summary.llm_calls
    assert summary_rec["total_reward"] == summary.reward
    assert summary_rec["meta"] == {"backend": "echo"}


def test_controller_marks_fallback_on_llm_error():
    class _Boom:
        def complete_json(self, *a, **k):
            from verifier_anchored_rl.llm_client import LLMResponseError
            raise LLMResponseError("nope")
    mdp = DeliberationMDP(problem="p?", expected_answer="42", depth_limit=4,
                          llm_client=EchoLLMClient())
    controller = LLMController(_Boom())
    state = mdp.initial_state()
    action = controller.choose(mdp, state)
    assert action == mdp.actions(state)[0]
    assert controller.last_choice_fallback is True


def test_structured_prompt_is_distinct_and_foregrounds_structural_verbs():
    assert CONTROLLER_SYSTEM_STRUCTURED != CONTROLLER_SYSTEM
    low = CONTROLLER_SYSTEM_STRUCTURED.lower()
    # It must teach when to decompose and prune...
    assert "split" in low and "abandon" in low and "case" in low
    # ...and must NOT carry the default prompt's discouragement of split.
    assert "rarely needed" not in low


def test_runner_maps_controller_prompt_choices():
    import experiments.run_llm_controller as runner
    assert runner._CONTROLLER_PROMPTS["default"] is CONTROLLER_SYSTEM
    assert runner._CONTROLLER_PROMPTS["structured"] is CONTROLLER_SYSTEM_STRUCTURED
