import io
import json

import numpy as np
import pytest

pytest.importorskip("sympy")

from verifier_anchored_rl.deliberation_mdp import (
    ActionOutcome,
    DeliberationAction,
    DeliberationMDP,
    DeliberationState,
    OutcomeCode,
    RejectionReason,
    Verb,
)
from verifier_anchored_rl.controller_observation import (
    CONTROLLER_OBSERVATION_VERSION,
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
    assert summary_rec["meta"]["backend"] == "echo"
    assert (
        summary_rec["meta"]["observation_schema"]
        == CONTROLLER_OBSERVATION_VERSION
    )
    assert summary_rec["meta"]["mdp_contract"] == mdp.contract_record()


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


class _Scripted:
    """Returns a fixed action string; records every user message."""
    def __init__(self, action):
        self.action = action
        self.users = []

    def complete_json(self, system, user, schema, **kw):
        self.users.append(user)
        return {"action": self.action}


def test_choose_normalizes_whitespace_and_case():
    mdp = DeliberationMDP(problem="p?", expected_answer="42", depth_limit=4,
                          llm_client=EchoLLMClient())
    controller = LLMController(_Scripted("  SPLIT:0 "))
    state = mdp.initial_state()
    action = controller.choose(mdp, state)
    assert action == DeliberationAction(Verb.SPLIT, 0)
    assert controller.last_choice_fallback is False


def test_fallback_prefers_a_legal_verify():
    from verifier_anchored_rl.reductio import compute
    mdp = DeliberationMDP(problem="p?", expected_answer="42", depth_limit=8,
                          llm_client=EchoLLMClient())
    s0 = mdp.initial_state()
    g = compute(s0.graph, 0, "6*7 = 42", premise_ids=()).graph  # fact 1
    state = type(s0)(g, 1)
    controller = LLMController(_Scripted("not-a-menu-key"))
    action = controller.choose(mdp, state)
    assert controller.last_choice_fallback is True
    assert action.verb is Verb.VERIFY                 # not menu[0]
    assert controller.last_raw_response == {"action": "not-a-menu-key"}


def test_choose_threads_feedback_and_turn_budget_into_the_message():
    mdp = DeliberationMDP(problem="p?", expected_answer="42", depth_limit=7,
                          llm_client=EchoLLMClient())
    client = _Scripted("split:0")
    controller = LLMController(client)
    initial = mdp.initial_state()
    state = DeliberationState(
        initial.graph,
        initial.turn,
        ActionOutcome(
            DeliberationAction(Verb.COMPUTE, 0),
            OutcomeCode.REJECTED,
            RejectionReason.DUPLICATE_COMPUTE,
        ),
    )
    controller.choose(mdp, state)
    assert "REJECTED (duplicate_compute)" in client.users[0]
    assert "[TURN 0/7]" in client.users[0]


def test_run_episode_feeds_rejection_back_and_logs_raw_response():
    # Turn 1: a legal choice whose generation fails (no scripted SPLIT
    # payload in the mock) -> masked; turn 2's message must carry the
    # rejection notice.
    mdp = DeliberationMDP(problem="p?", expected_answer="42", depth_limit=2,
                          llm_client=MockLLMClient(script={}))
    client = _Scripted("split:0")
    controller = LLMController(client)
    records = []
    run_episode(mdp, controller, np.random.default_rng(0),
                log_fn=records.append)
    assert records[0]["masked"] is True
    assert records[0]["transition_debug"]["mask_reason"] == "generation_error"
    assert records[0]["outcome"] == {
        "action": "split:0",
        "code": "rejected",
        "reason": "generation_error",
    }
    assert 'was REJECTED' in client.users[1]
    assert "split:0" in client.users[1]
    assert records[1]["controller_observation"]["previous_outcome"] == {
        "action": "split:0",
        "code": "rejected",
        "reason": "generation_error",
    }


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


def test_random_fallback_never_ends_the_episode():
    # At a state whose menu is only root actions (including abandon:0), an
    # unusable reply must never resolve to an irreversible root action: a
    # probe episode was lost to a random-fallback abandon:0 after six turns
    # of perfect play.
    mdp = DeliberationMDP(problem="p?", expected_answer="42", depth_limit=8,
                          llm_client=EchoLLMClient())
    state = mdp.initial_state()      # menu: split/form/compute/abandon on root
    controller = LLMController(_Scripted("garbage"))
    rng = np.random.default_rng(0)
    for _ in range(50):
        action = controller.choose(mdp, state, rng=rng)
        assert controller.last_choice_fallback is True
        assert not (action.target == 0 and action.verb in
                    (Verb.ABANDON, Verb.COMMIT))


def test_controller_schema_has_no_wire_enum():
    # The enum lives in choose()'s own lookup, not the wire schema: client-
    # side schema validation would turn every illegal-action intent into
    # opaque parse retries (observed: 42 laundered fallbacks in one probe).
    captured = {}

    class _SchemaCapture:
        def complete_json(self, system, user, schema, **kw):
            captured["schema"] = schema
            return {"action": "split:0"}

    mdp = DeliberationMDP(problem="p?", expected_answer="42", depth_limit=4,
                          llm_client=EchoLLMClient())
    LLMController(_SchemaCapture()).choose(mdp, mdp.initial_state())
    assert "enum" not in str(captured["schema"])


def test_malformed_teacher_reply_is_logged_verbatim():
    class Malformed:
        def complete_json(self, *args, **kwargs):
            from verifier_anchored_rl.llm_client import LLMResponseError

            raise LLMResponseError(
                "schema failure", raw_response="{bad teacher json")

    mdp = DeliberationMDP(
        problem="p",
        expected_answer="1",
        depth_limit=1,
        llm_client=MockLLMClient(script={}),
    )
    controller = LLMController(Malformed())
    records = []

    run_episode(
        mdp,
        controller,
        np.random.default_rng(0),
        log_fn=records.append,
    )

    assert records[0]["raw_response"] == "{bad teacher json"
