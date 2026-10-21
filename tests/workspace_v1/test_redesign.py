from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

import experiments.run_workspace_redesign as redesign_runner
from verifier_anchored_rl.llm_client import (
    LLMResponseError,
    OpenAICompatibleLLMClient,
)
from verifier_anchored_rl.verifier_adapter import RichResult
from verifier_anchored_rl.workspace_v1 import (
    ACTION_AVAILABILITY_ACTIVE_FRONTIER,
    CHILD_CONTEXT_TARGET_ONLY,
    INITIALIZATION_DIRECT_CANDIDATE_PREFIX,
    ROOT_RESULT_FREE,
    ROOT_RESULT_VERIFIER_READY,
    TEACHER_CHOICE_MENU_INDEX,
    Action,
    CandidateNode,
    DirectPolicy,
    LLMTeacherPolicy,
    ModelIdentity,
    ProblemExample,
    RedesignRunContract,
    Verb,
    WorkspaceEnvironment,
    build_observation,
    build_teacher_request,
    build_worker_request,
    collect_redesign_run,
    decode_teacher_index,
    direct_request,
    evaluation_binding_sha256,
    goal_candidates,
    problem_source_sha256,
    run_baseline_anchored_episode,
)


class QueueClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

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
            raise AssertionError("LLM received an unexpected call")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return deepcopy(response)


class ExactVerifier:
    def __call__(self, committed: str, expected: str) -> RichResult:
        return RichResult(
            success=committed == expected,
            observed=committed,
            reason=None if committed == expected else "private diagnostic",
        )


def environment(
    responses,
    *,
    expected: str = "5",
    turn_limit: int = 12,
    root_result_mode: str = ROOT_RESULT_VERIFIER_READY,
) -> tuple[WorkspaceEnvironment, QueueClient]:
    worker = QueueClient(responses)
    return (
        WorkspaceEnvironment(
            problem="Solve |4x - 16| = 4 and report the largest solution.",
            expected_answer=expected,
            worker=worker,
            turn_limit=turn_limit,
            max_depth=3,
            child_context_mode=CHILD_CONTEXT_TARGET_ONLY,
            temperature=0.0,
            max_tokens=700,
            answer_verifier=ExactVerifier(),
            root_result_mode=root_result_mode,
            action_availability_mode=ACTION_AVAILABILITY_ACTIVE_FRONTIER,
        ),
        worker,
    )


def test_direct_prefix_is_exact_unverified_baseline_and_direct_control():
    env, worker = environment([
        {"reasoning": "solve directly", "answer": "5"},
    ])

    episode = run_baseline_anchored_episode(env, DirectPolicy())

    assert episode.prefix_transition.worker_request == direct_request(env.problem)
    assert episode.prefix_transition.reward == 0
    assert episode.prefix_transition.action == Action(Verb.ANSWER, 0)
    assert [item.action for item in episode.policy_transitions] == [
        Action(Verb.SUBMIT, 1),
    ]
    assert len(worker.calls) == 1
    assert episode.reward == 1


def test_failed_prefix_is_visible_and_keeps_the_private_label_out_of_state():
    env, _ = environment(
        [LLMResponseError("invalid response", raw_response="bad-json")],
        expected="PRIVATE_LABEL_SENTINEL",
    )

    failed = env.step(env.initial_state(), Action(Verb.ANSWER, 0))
    observation = build_observation(
        failed.next_state,
        max_depth=3,
        action_availability_mode=ACTION_AVAILABILITY_ACTIVE_FRONTIER,
    )

    assert failed.generation_error
    assert len(failed.next_state.nodes) == 1
    assert failed.next_state.history[0].outcome.value == "generation_error"
    assert Action(Verb.ANSWER, 0) in env.legal_actions(failed.next_state)
    assert observation["protected_direct_candidate_id"] is None
    assert "PRIVATE_LABEL_SENTINEL" not in repr(observation)


def test_verifier_ready_contract_changes_root_requests_only():
    env, _ = environment([
        {"reasoning": "direct", "answer": "5"},
        {"children": ["first branch", "second branch"]},
    ])
    initial = env.initial_state()
    assert build_worker_request(
        initial,
        Action(Verb.ANSWER, 0),
        root_result_mode=ROOT_RESULT_VERIFIER_READY,
    ) == direct_request(env.problem)

    prefixed = env.step(initial, Action(Verb.ANSWER, 0)).next_state
    decomposed = env.step(prefixed, Action(Verb.DECOMPOSE, 0)).next_state
    root_actions = (
        (prefixed, Action(Verb.ANSWER, 0)),
        (decomposed, Action(Verb.SYNTHESIZE, 0)),
        (prefixed, Action(Verb.REVISE, 1)),
    )
    for state, action in root_actions:
        free = build_worker_request(
            state,
            action,
            root_result_mode=ROOT_RESULT_FREE,
        )
        ready = build_worker_request(
            state,
            action,
            root_result_mode=ROOT_RESULT_VERIFIER_READY,
        )
        assert "directly checkable by its symbolic verifier" not in free.user
        assert "directly checkable by its symbolic verifier" in ready.user
        assert free.system == ready.system
        assert free.schema == ready.schema

    child_action = Action(Verb.ANSWER, 2)
    assert build_worker_request(
        decomposed,
        child_action,
        root_result_mode=ROOT_RESULT_FREE,
    ) == build_worker_request(
        decomposed,
        child_action,
        root_result_mode=ROOT_RESULT_VERIFIER_READY,
    )


def test_active_frontier_tracks_fresh_work_and_protects_direct_candidate():
    env, _ = environment([
        {"reasoning": "direct", "answer": "5"},
        {"children": ["first branch", "second branch"]},
        {"reasoning": "first", "answer": "2"},
        {"reasoning": "second", "answer": "3"},
        {"reasoning": "initial synthesis", "answer": "5"},
        {"reasoning": "revised first", "answer": "4"},
        {"reasoning": "fresh synthesis", "answer": "7"},
        {"reasoning": "revised root", "answer": "5"},
    ])
    state = env.step(
        env.initial_state(), Action(Verb.ANSWER, 0)
    ).next_state
    assert env.legal_actions(state) == (
        Action(Verb.DECOMPOSE, 0),
        Action(Verb.REVISE, 1),
        Action(Verb.SUBMIT, 1),
    )
    observation = build_observation(
        state,
        max_depth=3,
        action_availability_mode=ACTION_AVAILABILITY_ACTIVE_FRONTIER,
    )
    assert observation["protected_direct_candidate_id"] == 1

    state = env.step(state, Action(Verb.DECOMPOSE, 0)).next_state
    legal = env.legal_actions(state)
    assert Action(Verb.DECOMPOSE, 0) not in legal
    assert Action(Verb.ANSWER, 2) in legal
    assert Action(Verb.DECOMPOSE, 2) in legal
    assert Action(Verb.SYNTHESIZE, 0) not in legal

    state = env.step(state, Action(Verb.ANSWER, 2)).next_state
    legal = env.legal_actions(state)
    assert Action(Verb.ANSWER, 2) not in legal
    assert Action(Verb.DECOMPOSE, 2) not in legal
    assert Action(Verb.REVISE, 4) in legal
    assert Action(Verb.SYNTHESIZE, 0) not in legal

    state = env.step(state, Action(Verb.ANSWER, 3)).next_state
    assert Action(Verb.SYNTHESIZE, 0) in env.legal_actions(state)
    state = env.step(state, Action(Verb.SYNTHESIZE, 0)).next_state
    legal = env.legal_actions(state)
    assert Action(Verb.SYNTHESIZE, 0) not in legal
    assert Action(Verb.REVISE, 1) not in legal
    assert Action(Verb.REVISE, 6) in legal
    assert Action(Verb.SUBMIT, 1) in legal
    assert Action(Verb.SUBMIT, 6) in legal

    state = env.step(state, Action(Verb.REVISE, 4)).next_state
    legal = env.legal_actions(state)
    assert Action(Verb.REVISE, 4) not in legal
    assert Action(Verb.REVISE, 7) not in legal
    assert Action(Verb.SYNTHESIZE, 0) in legal

    state = env.step(state, Action(Verb.SYNTHESIZE, 0)).next_state
    legal = env.legal_actions(state)
    assert Action(Verb.SYNTHESIZE, 0) not in legal
    assert Action(Verb.REVISE, 6) not in legal
    assert Action(Verb.REVISE, 8) in legal
    assert Action(Verb.SUBMIT, 1) in legal
    assert Action(Verb.SUBMIT, 6) not in legal
    assert Action(Verb.SUBMIT, 8) in legal

    state = env.step(state, Action(Verb.REVISE, 8)).next_state
    legal = env.legal_actions(state)
    assert not any(
        action.verb is Verb.REVISE
        and isinstance(state.nodes[action.target], CandidateNode)
        and state.nodes[action.target].goal_id == 0
        for action in legal
    )
    assert Action(Verb.SUBMIT, 1) in legal
    assert Action(Verb.SUBMIT, 9) in legal


def test_indexed_teacher_maps_only_menu_entries_and_preserves_baseline_on_error():
    env, _ = environment([
        {"reasoning": "direct", "answer": "5"},
    ])
    state = env.step(
        env.initial_state(), Action(Verb.ANSWER, 0)
    ).next_state
    legal = env.legal_actions(state)
    request = build_teacher_request(
        state,
        max_depth=3,
        action_availability_mode=ACTION_AVAILABILITY_ACTIVE_FRONTIER,
        choice_mode=TEACHER_CHOICE_MENU_INDEX,
    )

    assert request.schema["properties"]["action_index"]["enum"] == [0, 1, 2]
    assert '"protected_direct_candidate_id":1' in request.user
    assert decode_teacher_index({"action_index": 1}, legal) == legal[1]
    with pytest.raises(ValueError, match="outside the legal menu"):
        decode_teacher_index({"action_index": 99}, legal)

    teacher = LLMTeacherPolicy(
        QueueClient([{"action_index": 99}]),
        max_depth=3,
        action_availability_mode=ACTION_AVAILABILITY_ACTIVE_FRONTIER,
        choice_mode=TEACHER_CHOICE_MENU_INDEX,
    )
    selected = teacher.choose(state, legal)

    assert selected == Action(Verb.SUBMIT, 1)
    assert teacher.decisions[0].fallback
    assert teacher.decisions[0].proposed_action is None
    assert teacher.decisions[0].payload_json == '{"action_index":99}'


class ResponseRouter:
    def __init__(self, responses: dict[str, list[dict[str, object]]]):
        self.responses = {
            model: list(values) for model, values in responses.items()
        }
        self.requests: list[dict[str, object]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        httpx = pytest.importorskip("httpx")
        router = self

        class Response:
            status_code = 200

            def __init__(self, payload: dict[str, object]):
                self._payload = payload
                self.text = json.dumps(payload)

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return self._payload

        class Client:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, url, *, headers=None, json=None):
                assert isinstance(json, dict)
                model = json["model"]
                content = router.responses[model].pop(0)
                router.requests.append({
                    "url": url,
                    "headers": dict(headers or {}),
                    "body": json,
                })
                return Response({
                    "choices": [{
                        "message": {"content": json_module.dumps(content)}
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 4,
                        "total_tokens": 14,
                    },
                })

        json_module = json
        monkeypatch.setattr(httpx, "Client", Client)


def identity(model: str) -> ModelIdentity:
    return ModelIdentity(
        provider="test-provider",
        model=model,
        revision=f"{model}-revision",
        base_url="https://example.test/v1",
        guided_json=False,
        timeout_s=30.0,
        parse_retries=0,
        backoff_retries=0,
    )


def openai_client(model: str, api_key: str) -> OpenAICompatibleLLMClient:
    return OpenAICompatibleLLMClient(
        base_url="https://example.test/v1",
        model=model,
        api_key=api_key,
        timeout_s=30.0,
        max_retries=0,
        max_backoff_retries=0,
        guided_json=False,
    )


def redesign_contract(
    examples: tuple[ProblemExample, ...],
    *,
    policy_name: str = "teacher",
) -> RedesignRunContract:
    teacher = identity("teacher-model") if policy_name == "teacher" else None
    return RedesignRunContract(
        preregistration_id="E-REDESIGN-TEST",
        arm=policy_name,
        policy_name=policy_name,
        problem_source_sha256=problem_source_sha256(examples),
        evaluation_binding_sha256=evaluation_binding_sha256(examples),
        problem_ids=tuple(example.problem_id for example in examples),
        turn_limit=8,
        max_depth=3,
        initialization_mode=INITIALIZATION_DIRECT_CANDIDATE_PREFIX,
        child_context_mode=CHILD_CONTEXT_TARGET_ONLY,
        root_result_mode=ROOT_RESULT_VERIFIER_READY,
        action_availability_mode=ACTION_AVAILABILITY_ACTIVE_FRONTIER,
        teacher_choice_mode=TEACHER_CHOICE_MENU_INDEX,
        worker=identity("worker-model"),
        worker_temperature=0.0,
        worker_max_tokens=700,
        worker_seed=0,
        teacher=teacher,
        teacher_temperature=0.0 if teacher is not None else None,
        teacher_max_tokens=64 if teacher is not None else None,
        teacher_seed=100 if teacher is not None else None,
    )


def read_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_redesign_contract_binds_each_independent_factor():
    examples = (ProblemExample("p1", "Compute a value.", "5"),)
    base = redesign_contract(examples)
    header = base.header()

    assert header["settings"]["initialization_mode"] == (
        INITIALIZATION_DIRECT_CANDIDATE_PREFIX
    )
    assert header["settings"]["root_result_mode"] == (
        ROOT_RESULT_VERIFIER_READY
    )
    assert header["settings"]["action_availability_mode"] == (
        ACTION_AVAILABILITY_ACTIVE_FRONTIER
    )
    assert header["settings"]["teacher_choice_mode"] == (
        TEACHER_CHOICE_MENU_INDEX
    )
    alternatives = (
        replace(base, initialization_mode="empty-root-v1"),
        replace(base, root_result_mode=ROOT_RESULT_FREE),
        replace(base, action_availability_mode="all-affordances-v1"),
        replace(base, teacher_choice_mode="verb-target-choice-v1"),
    )
    assert all(item.sha256 != base.sha256 for item in alternatives)
    assert len({item.sha256 for item in alternatives}) == len(alternatives)


def test_redesign_runner_prepares_contract_without_credentials_or_network(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    problems = tmp_path / "problems.jsonl"
    problems.write_text(
        '{"id":"p1","problem":"Compute a value.","answer":"5"}\n',
        encoding="ascii",
    )

    result = redesign_runner.main([
        "--print-contracts",
        "--preregistration-id",
        "E-PREPARE-TEST",
        "--arm",
        "direct",
        "--policy",
        "direct",
        "--problems",
        str(problems),
        "--start-index",
        "0",
        "--episode-count",
        "1",
        "--turn-limit",
        "8",
        "--max-depth",
        "3",
        "--initialization-mode",
        INITIALIZATION_DIRECT_CANDIDATE_PREFIX,
        "--child-context-mode",
        CHILD_CONTEXT_TARGET_ONLY,
        "--root-result-mode",
        ROOT_RESULT_VERIFIER_READY,
        "--action-availability-mode",
        ACTION_AVAILABILITY_ACTIVE_FRONTIER,
        "--teacher-choice-mode",
        TEACHER_CHOICE_MENU_INDEX,
        "--worker-provider",
        "test-provider",
        "--worker-model",
        "worker-model",
        "--worker-revision",
        "worker-revision",
        "--worker-base-url",
        "https://example.test/v1",
        "--no-worker-guided-json",
        "--worker-timeout-s",
        "30",
        "--worker-parse-retries",
        "0",
        "--worker-backoff-retries",
        "0",
        "--worker-temperature",
        "0.0",
        "--worker-max-tokens",
        "700",
        "--worker-seed",
        "0",
    ])

    assert result == 0
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["schema_version"] == "workspace-redesign-preparation-v1"
    assert prepared["settings"]["initialization_mode"] == (
        INITIALIZATION_DIRECT_CANDIDATE_PREFIX
    )
    assert len(prepared["run_contract_sha256"]) == 64


def test_redesign_collection_separates_prefix_and_teacher_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    router = ResponseRouter({
        "worker-model": [{"reasoning": "direct", "answer": "5"}],
        "teacher-model": [{"action_index": 2}],
    })
    router.install(monkeypatch)
    examples = (
        ProblemExample("p1", "Compute a value.", "PRIVATE_LABEL_SENTINEL"),
    )
    run_path = tmp_path / "redesign.jsonl"
    diagnostic_path = tmp_path / "redesign.diagnostics.jsonl"

    result = collect_redesign_run(
        run_path=run_path,
        diagnostic_path=diagnostic_path,
        contract=redesign_contract(examples),
        examples=examples,
        worker_client=openai_client("worker-model", "WORKER_SECRET"),
        teacher_client=openai_client("teacher-model", "TEACHER_SECRET"),
    )

    assert result.transition_count == 2
    assert result.controller_decision_count == 1
    records = read_records(run_path)
    transitions = [
        record
        for record in records
        if record.get("schema_version")
        == "workspace-redesign-transition-record-v1"
    ]
    assert [record["decision_source"] for record in transitions] == [
        "deterministic_prefix",
        "teacher",
    ]
    assert transitions[0]["teacher_call_id"] is None
    assert transitions[0]["teacher_decision"] is None
    assert transitions[0]["behavior_cloning_eligible"] is False
    assert transitions[1]["teacher_call_id"] is not None
    assert transitions[1]["teacher_decision"] is not None
    assert transitions[1]["behavior_cloning_eligible"] is True
    assert transitions[1]["action"] == {"verb": "submit", "target": 1}
    assert transitions[1]["worker_call_id"] is None
    summary = next(
        record
        for record in records
        if record.get("record_type") == "episode_summary"
    )
    assert summary["prefix_worker_call_id"] == transitions[0]["worker_call_id"]
    assert summary["teacher_call_ids"] == [transitions[1]["teacher_call_id"]]
    combined = run_path.read_text() + diagnostic_path.read_text()
    assert "PRIVATE_LABEL_SENTINEL" not in combined
    assert "WORKER_SECRET" not in combined
    assert "TEACHER_SECRET" not in combined
    assert len(router.requests) == 2
