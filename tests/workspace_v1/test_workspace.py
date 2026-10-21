from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import pytest

import experiments.run_direct_baseline as direct_baseline
from verifier_anchored_rl.llm_client import LLMResponseError
from verifier_anchored_rl.verifier_adapter import RichResult
from verifier_anchored_rl.workspace_v1 import (
    CHILD_CONTEXT_PRECEDING_SIBLINGS,
    CHILD_CONTEXT_TARGET_ONLY,
    DIRECT_SCHEMA,
    DIRECT_SYSTEM,
    Action,
    CandidateNode,
    DirectPolicy,
    EventOutcome,
    FixedDecompositionPolicy,
    GoalNode,
    LLMTeacherPolicy,
    Verb,
    WorkspaceEnvironment,
    build_hybrid_lens,
    build_observation,
    build_worker_request,
    decode_children,
    direct_request,
    decode_teacher_action,
    goal_candidates,
    run_policy_episode,
)


class QueueClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete_json(
        self,
        system: str,
        user: str,
        schema: Mapping[str, object],
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        seed: int | None = None,
        schema_validation_mode: str = "json-schema-numeric-string-coercion-v1",
    ):
        self.calls.append({
            "system": system,
            "user": user,
            "schema": deepcopy(dict(schema)),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "schema_validation_mode": schema_validation_mode,
        })
        if not self.responses:
            raise AssertionError("worker received an unexpected call")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return deepcopy(response)


class CountingVerifier:
    def __init__(self, expected: str):
        self.expected = expected
        self.calls = []

    def __call__(self, committed: str, expected: str) -> RichResult:
        self.calls.append((committed, expected))
        success = committed == self.expected and expected == self.expected
        return RichResult(
            success=success,
            observed=committed,
            reason=None if success else "private diagnostic with hidden label",
        )


def environment(
    responses,
    *,
    expected="5",
    turn_limit=12,
    child_context_mode=CHILD_CONTEXT_TARGET_ONLY,
):
    worker = QueueClient(responses)
    verifier = CountingVerifier(expected)
    env = WorkspaceEnvironment(
        problem="Solve |4x - 16| = 4 and report the largest solution.",
        expected_answer=expected,
        worker=worker,
        turn_limit=turn_limit,
        max_depth=3,
        child_context_mode=child_context_mode,
        temperature=0.0,
        max_tokens=700,
        answer_verifier=verifier,
    )
    return env, worker, verifier


def test_initial_root_answer_is_exact_direct_baseline_request():
    env, _, _ = environment([])
    state = env.initial_state()
    request = build_worker_request(state, Action(Verb.ANSWER, 0))
    direct = direct_request(env.problem)

    assert request.system == direct.system == direct_baseline.SYSTEM == DIRECT_SYSTEM
    assert request.user == direct.user
    assert request.schema == direct.schema == direct_baseline.SCHEMA == DIRECT_SCHEMA

    predecessor_mode = build_worker_request(
        state,
        Action(Verb.ANSWER, 0),
        child_context_mode=CHILD_CONTEXT_PRECEDING_SIBLINGS,
    )
    assert predecessor_mode == request


def test_direct_policy_recovers_one_worker_call_and_symbolic_submit():
    env, worker, verifier = environment([
        {"reasoning": ["4x-16 is 4 or -4"], "answer": 5},
    ])

    episode = run_policy_episode(env, DirectPolicy())

    assert [transition.action for transition in episode.transitions] == [
        Action(Verb.ANSWER, 0),
        Action(Verb.SUBMIT, 1),
    ]
    assert len(worker.calls) == 1
    assert worker.calls[0]["system"] == direct_baseline.SYSTEM
    assert worker.calls[0]["user"] == direct_request(env.problem).user
    assert worker.calls[0]["temperature"] == 0.0
    assert worker.calls[0]["max_tokens"] == 700
    assert worker.calls[0]["seed"] is None
    assert verifier.calls == [("5", "5")]
    assert episode.reward == 1
    assert episode.final_state.terminal


def test_failed_submit_is_terminal_zero_and_hides_expected_from_state():
    env, worker, verifier = environment([
        {"reasoning": "guess", "answer": "4"},
    ])

    episode = run_policy_episode(env, DirectPolicy())
    observation = build_observation(episode.final_state, max_depth=3)

    assert len(worker.calls) == 1
    assert verifier.calls == [("4", "5")]
    assert episode.reward == 0
    assert episode.final_state.history[-1].outcome is EventOutcome.SUBMIT_REJECTED
    assert "private diagnostic" not in repr(observation)
    assert "expected_answer" not in repr(observation)


def test_decompose_accepts_natural_language_children_without_algebra():
    decomposition = {
        "children": [
            "Solve the branch 4x - 16 = 4.",
            "Solve the branch 4x - 16 = -4.",
            {"task": "Compare the branch solutions and select the maximum."},
        ],
        "unrequested_note": "This is unverified planning text.",
    }
    env, worker, verifier = environment([decomposition])
    state = env.initial_state()

    transition = env.step(state, Action(Verb.DECOMPOSE, 0))

    assert transition.next_state.history[-1].outcome is EventOutcome.APPLIED
    assert [node.task for node in transition.next_state.nodes[1:]] == [
        "Solve the branch 4x - 16 = 4.",
        "Solve the branch 4x - 16 = -4.",
        "Compare the branch solutions and select the maximum.",
    ]
    assert all(isinstance(node, GoalNode) for node in transition.next_state.nodes)
    assert len(worker.calls) == 1
    assert verifier.calls == []
    assert transition.reward == 0


def test_standard_interval_notation_is_workspace_content_not_a_rejection():
    env, _, verifier = environment([
        {"children": ["(-infinity, 3]", "(3, infinity)"]},
    ])

    transition = env.step(
        env.initial_state(), Action(Verb.DECOMPOSE, 0)
    )

    assert transition.next_state.history[-1].outcome is EventOutcome.APPLIED
    assert [node.task for node in transition.next_state.nodes[1:]] == [
        "(-infinity, 3]",
        "(3, infinity)",
    ]
    assert verifier.calls == []


def test_decomposition_decoder_preserves_loose_json_items():
    assert decode_children({
        "children": [
            {"task": "first", "purpose": "ignored metadata"},
            7,
            None,
            {"arbitrary": ["structured", "content"]},
        ]
    }) == (
        "first",
        "7",
        "null",
        '{"arbitrary":["structured","content"]}',
    )


def test_generation_error_is_structural_and_retains_attempt_history():
    env, _, verifier = environment([
        LLMResponseError("unparseable response", raw_response="not json"),
        {"children": ["child one", "child two"]},
    ])
    initial = env.initial_state()

    failed = env.step(initial, Action(Verb.DECOMPOSE, 0))
    retried = env.step(failed.next_state, Action(Verb.DECOMPOSE, 0))

    assert failed.generation_error
    assert failed.next_state.nodes == initial.nodes
    assert failed.next_state.history[-1].outcome is EventOutcome.GENERATION_ERROR
    assert retried.next_state.history[-1].outcome is EventOutcome.APPLIED
    assert len(retried.next_state.history) == 2
    assert verifier.calls == []


def test_fixed_decomposition_policy_focuses_children_then_synthesizes():
    env, worker, verifier = environment([
        {"children": ["Solve the positive branch", "Solve the negative branch"]},
        {"reasoning": "positive branch", "answer": "5"},
        {"reasoning": "negative branch", "answer": "3"},
        {"reasoning": "take the larger child result", "answer": "5"},
    ])

    episode = run_policy_episode(env, FixedDecompositionPolicy())

    assert [transition.action for transition in episode.transitions] == [
        Action(Verb.DECOMPOSE, 0),
        Action(Verb.ANSWER, 1),
        Action(Verb.ANSWER, 2),
        Action(Verb.SYNTHESIZE, 0),
        Action(Verb.SUBMIT, 5),
    ]
    assert len(worker.calls) == 4
    assert verifier.calls == [("5", "5")]
    assert episode.reward == 1

    child_request = worker.calls[1]["user"]
    assert "Solve the positive branch" in child_request
    assert env.problem in child_request
    assert "Solve the negative branch" not in child_request

    synthesis_request = worker.calls[3]["user"]
    assert "Solve the positive branch" in synthesis_request
    assert "positive branch" in synthesis_request
    assert "Solve the negative branch" in synthesis_request
    assert "negative branch" in synthesis_request


def test_fixed_decomposition_falls_back_to_direct_after_generation_error():
    env, worker, verifier = environment([
        LLMResponseError("bad decomposition"),
        {"reasoning": "solve directly", "answer": "5"},
    ])

    episode = run_policy_episode(env, FixedDecompositionPolicy())

    assert [transition.action for transition in episode.transitions] == [
        Action(Verb.DECOMPOSE, 0),
        Action(Verb.ANSWER, 0),
        Action(Verb.SUBMIT, 1),
    ]
    assert len(worker.calls) == 2
    assert verifier.calls == [("5", "5")]
    assert episode.reward == 1


def test_synthesis_accepts_nested_workspace_candidate_shape():
    env, _, verifier = environment([
        {"children": ["positive branch", "negative branch"]},
        {"reasoning": "positive", "answer": "5"},
        {"reasoning": "negative", "answer": "3"},
        {
            "candidates": [{
                "answer": "5",
                "reasoning": "select the larger child answer",
                "goal_id": 2,
                "kind": "candidate",
            }],
            "goal": {"id": 0, "kind": "goal"},
        },
    ])

    episode = run_policy_episode(env, FixedDecompositionPolicy())

    assert episode.reward == 1
    assert episode.transitions[-2].action == Action(Verb.SYNTHESIZE, 0)
    assert episode.transitions[-2].generation_error is False
    assert verifier.calls == [("5", "5")]


def test_fixed_policy_does_not_repeat_failed_synthesis_to_horizon():
    env, _, verifier = environment([
        {"children": ["positive branch", "negative branch"]},
        {"reasoning": "positive", "answer": "5"},
        {"reasoning": "negative", "answer": "3"},
        LLMResponseError("bad synthesis"),
        {"reasoning": "fallback root answer", "answer": "5"},
    ])

    episode = run_policy_episode(env, FixedDecompositionPolicy())

    assert [transition.action.verb for transition in episode.transitions] == [
        Verb.DECOMPOSE,
        Verb.ANSWER,
        Verb.ANSWER,
        Verb.SYNTHESIZE,
        Verb.ANSWER,
        Verb.SUBMIT,
    ]
    assert sum(
        transition.action.verb is Verb.SYNTHESIZE
        for transition in episode.transitions
    ) == 1
    assert verifier.calls == [("5", "5")]
    assert episode.reward == 1


def test_hybrid_lens_contains_ancestry_and_no_hidden_label():
    env, _, _ = environment([
        {"children": ["first child", "second child"]},
    ])
    first = env.step(
        env.initial_state(), Action(Verb.DECOMPOSE, 0)
    )
    lens = build_hybrid_lens(first.next_state, Action(Verb.ANSWER, 1))

    assert lens["root_problem"] == env.problem
    assert lens["target"]["id"] == 1
    assert lens["ancestry"][0]["id"] == 0
    assert lens["ancestry"][-1]["id"] == 1
    assert "expected_answer" not in repr(lens)


def test_preceding_sibling_context_is_ordered_completed_and_causal():
    env, worker, _ = environment(
        [
            {"children": [
                "FIRST_TASK_SENTINEL",
                "SECOND_TASK_SENTINEL",
                "LATER_TASK_SENTINEL",
            ]},
            {"reasoning": "FIRST_REASONING_SENTINEL", "answer": "11"},
            {"reasoning": "SECOND_REASONING_SENTINEL", "answer": "22"},
        ],
        expected="PRIVATE_LABEL_SENTINEL",
        child_context_mode=CHILD_CONTEXT_PRECEDING_SIBLINGS,
    )
    decomposed = env.step(
        env.initial_state(), Action(Verb.DECOMPOSE, 0)
    )

    first_request = build_worker_request(
        decomposed.next_state,
        Action(Verb.ANSWER, 1),
        child_context_mode=CHILD_CONTEXT_PRECEDING_SIBLINGS,
    )
    target_only_first = build_worker_request(
        decomposed.next_state,
        Action(Verb.ANSWER, 1),
        child_context_mode=CHILD_CONTEXT_TARGET_ONLY,
    )
    assert first_request == target_only_first

    first_answered = env.step(
        decomposed.next_state, Action(Verb.ANSWER, 1)
    )
    second_lens = build_hybrid_lens(
        first_answered.next_state,
        Action(Verb.ANSWER, 2),
        child_context_mode=CHILD_CONTEXT_PRECEDING_SIBLINGS,
    )
    second_parent = next(
        item for item in second_lens["ancestry"] if item["id"] == 0
    )
    preceding = second_parent["completed_preceding_child_work"]
    assert [item["goal"]["id"] for item in preceding] == [1]
    assert preceding[0]["goal"]["task"] == "FIRST_TASK_SENTINEL"
    assert preceding[0]["candidates"][0]["answer"] == "11"
    assert "FIRST_REASONING_SENTINEL" in repr(preceding)
    assert "LATER_TASK_SENTINEL" not in repr(second_lens)
    assert "PRIVATE_LABEL_SENTINEL" not in repr(second_lens)

    target_only_second = build_hybrid_lens(
        first_answered.next_state,
        Action(Verb.ANSWER, 2),
        child_context_mode=CHILD_CONTEXT_TARGET_ONLY,
    )
    assert "completed_preceding_child_work" not in repr(target_only_second)
    assert "FIRST_TASK_SENTINEL" not in repr(target_only_second)

    second_answered = env.step(
        first_answered.next_state, Action(Verb.ANSWER, 2)
    )
    third_lens = build_hybrid_lens(
        second_answered.next_state,
        Action(Verb.ANSWER, 3),
        child_context_mode=CHILD_CONTEXT_PRECEDING_SIBLINGS,
    )
    third_parent = next(
        item for item in third_lens["ancestry"] if item["id"] == 0
    )
    third_preceding = third_parent["completed_preceding_child_work"]
    assert [item["goal"]["id"] for item in third_preceding] == [1, 2]
    assert [
        item["candidates"][0]["answer"] for item in third_preceding
    ] == ["11", "22"]
    assert "LATER_TASK_SENTINEL" in repr(third_lens["target"])
    assert "PRIVATE_LABEL_SENTINEL" not in repr(worker.calls)


def test_submit_never_builds_or_calls_a_worker_request():
    env, worker, _ = environment([
        {"reasoning": "done", "answer": "5"},
    ])
    answered = env.step(env.initial_state(), Action(Verb.ANSWER, 0))
    candidate = goal_candidates(answered.next_state, 0)[0]
    assert isinstance(candidate, CandidateNode)

    with pytest.raises(ValueError, match="SUBMIT never"):
        build_worker_request(
            answered.next_state,
            Action(Verb.SUBMIT, candidate.id),
        )
    submitted = env.step(
        answered.next_state,
        Action(Verb.SUBMIT, candidate.id),
    )

    assert len(worker.calls) == 1
    assert submitted.worker_request is None
    assert submitted.submit_receipt is not None
    assert submitted.submit_receipt.accepted


def test_teacher_sees_full_state_and_selects_one_exact_legal_action():
    env, _, _ = environment([])
    client = QueueClient([{"verb": "decompose", "target": 0}])
    teacher = LLMTeacherPolicy(client, max_depth=3, base_seed=11)
    state = env.initial_state()

    selected = teacher.choose(state, env.legal_actions(state))

    assert selected == Action(Verb.DECOMPOSE, 0)
    assert len(teacher.decisions) == 1
    assert not teacher.decisions[0].fallback
    assert '"legal_actions"' in client.calls[0]["user"]
    assert env.problem in client.calls[0]["user"]
    assert client.calls[0]["seed"] == 11


def test_teacher_illegal_choice_falls_back_to_direct_root_answer():
    env, _, _ = environment([])
    client = QueueClient([{"verb": "submit", "target": 999}])
    teacher = LLMTeacherPolicy(client, max_depth=3)
    state = env.initial_state()

    selected = teacher.choose(state, env.legal_actions(state))

    assert decode_teacher_action({"verb": "submit", "target": 999}) == Action(
        Verb.SUBMIT, 999
    )
    assert selected == Action(Verb.ANSWER, 0)
    assert teacher.decisions[0].fallback
    assert teacher.decisions[0].proposed_action == Action(Verb.SUBMIT, 999)
