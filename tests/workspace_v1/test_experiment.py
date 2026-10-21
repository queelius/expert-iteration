from __future__ import annotations

import json
from pathlib import Path

import pytest

from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient
from verifier_anchored_rl.workspace_v1 import (
    CHILD_CONTEXT_PRECEDING_SIBLINGS,
    CHILD_CONTEXT_TARGET_ONLY,
    ModelIdentity,
    ProblemExample,
    RunContract,
    collect_workspace_run,
    evaluation_binding_sha256,
    problem_source_record,
    problem_source_sha256,
    load_problem_examples,
)


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
                assert isinstance(model, str)
                content = router.responses[model].pop(0)
                router.requests.append({
                    "url": url,
                    "headers": dict(headers or {}),
                    "body": json,
                })
                payload = {
                    "choices": [{
                        "message": {
                            "content": json_module.dumps(content),
                        }
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 4,
                        "total_tokens": 14,
                    },
                }
                return Response(payload)

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


def client(model: str, *, api_key: str) -> OpenAICompatibleLLMClient:
    return OpenAICompatibleLLMClient(
        base_url="https://example.test/v1",
        model=model,
        api_key=api_key,
        timeout_s=30.0,
        max_retries=0,
        max_backoff_retries=0,
        guided_json=False,
    )


def contract(
    examples: tuple[ProblemExample, ...],
    *,
    policy: str,
    teacher: ModelIdentity | None = None,
    child_context_mode: str = CHILD_CONTEXT_TARGET_ONLY,
) -> RunContract:
    return RunContract(
        preregistration_id="E-TEST",
        arm=policy,
        policy_name=policy,
        problem_source_sha256=problem_source_sha256(examples),
        evaluation_binding_sha256=evaluation_binding_sha256(examples),
        problem_ids=tuple(example.problem_id for example in examples),
        turn_limit=8,
        max_depth=3,
        child_context_mode=child_context_mode,
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


def test_source_projection_is_label_free_but_evaluation_digest_binds_label():
    left = (ProblemExample("p1", "Compute a value.", "private-one"),)
    right = (ProblemExample("p1", "Compute a value.", "private-two"),)

    assert problem_source_record(left) == problem_source_record(right)
    assert problem_source_sha256(left) == problem_source_sha256(right)
    assert evaluation_binding_sha256(left) != evaluation_binding_sha256(right)
    assert "private-one" not in repr(problem_source_record(left))


def test_run_contract_binds_child_context_mode():
    examples = (ProblemExample("p1", "Compute a value.", "5"),)
    target_only = contract(examples, policy="fixed_decomposition")
    preceding = contract(
        examples,
        policy="fixed_decomposition",
        child_context_mode=CHILD_CONTEXT_PRECEDING_SIBLINGS,
    )

    assert target_only.header()["settings"]["child_context_mode"] == (
        CHILD_CONTEXT_TARGET_ONLY
    )
    assert preceding.header()["settings"]["child_context_mode"] == (
        CHILD_CONTEXT_PRECEDING_SIBLINGS
    )
    assert target_only.sha256 != preceding.sha256


def test_problem_loader_assigns_stable_ids_when_dataset_has_none(tmp_path: Path):
    source = tmp_path / "problems.jsonl"
    source.write_text(
        '{"problem":"first","answer":"1"}\n'
        '{"problem":"second","answer":"2"}\n',
        encoding="utf-8",
    )

    selected = load_problem_examples(
        source, start_index=1, episode_count=1
    )

    assert selected[0].problem_id == "dataset-row-000001"
    assert selected[0].problem == "second"


def test_direct_collection_publishes_secret_free_complete_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    router = ResponseRouter({
        "worker-model": [{"reasoning": "an attempt", "answer": "wrong"}],
    })
    router.install(monkeypatch)
    examples = (
        ProblemExample("p1", "Compute the requested value.", "PRIVATE_LABEL"),
    )
    run_path = tmp_path / "run.jsonl"
    diagnostic_path = tmp_path / "diagnostic.jsonl"
    worker = client("worker-model", api_key="TOP_SECRET_TOKEN")

    result = collect_workspace_run(
        run_path=run_path,
        diagnostic_path=diagnostic_path,
        contract=contract(examples, policy="direct"),
        examples=examples,
        worker_client=worker,
    )

    assert result.solved_count == 0
    assert result.transition_count == 2
    assert run_path.exists() and diagnostic_path.exists()
    assert not (tmp_path / "run.jsonl.partial").exists()
    assert not (tmp_path / "diagnostic.jsonl.partial").exists()
    combined = run_path.read_text() + diagnostic_path.read_text()
    assert "PRIVATE_LABEL" not in combined
    assert "TOP_SECRET_TOKEN" not in combined
    assert "expected_answer" not in combined
    assert router.requests[0]["headers"]["Authorization"] == (
        "Bearer TOP_SECRET_TOKEN"
    )

    records = read_records(run_path)
    transitions = [
        record
        for record in records
        if record.get("schema_version")
        == "workspace-transition-record-v1"
    ]
    assert [record["action"]["verb"] for record in transitions] == [
        "answer",
        "submit",
    ]
    assert transitions[0]["worker_call_id"] is not None
    assert transitions[1]["worker_call_id"] is None
    assert transitions[1]["submit_receipt"]["reason_code"] == (
        "incorrect_answer"
    )
    summary = records[-1]
    assert summary["record_type"] == "run_summary"
    assert summary["usage"]["worker"] == {
        "generate_calls": 1,
        "prompt_tokens": 10,
        "generated_tokens": 4,
    }

    diagnostics = read_records(diagnostic_path)
    assert [record["record_type"] for record in diagnostics] == [
        "diagnostic_header",
        "http_attempt",
    ]
    assert diagnostics[1]["role"] == "worker"
    with pytest.raises(FileExistsError):
        collect_workspace_run(
            run_path=run_path,
            diagnostic_path=diagnostic_path,
            contract=contract(examples, policy="direct"),
            examples=examples,
            worker_client=worker,
        )


def test_teacher_collection_aligns_each_decision_and_worker_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    router = ResponseRouter({
        "teacher-model": [
            {"verb": "answer", "target": 0},
            {"verb": "submit", "target": 1},
        ],
        "worker-model": [{"reasoning": "2 + 3 = 5", "answer": "5"}],
    })
    router.install(monkeypatch)
    examples = (ProblemExample("p1", "Compute 2 + 3.", "5"),)
    teacher_identity = identity("teacher-model")
    run_path = tmp_path / "teacher.jsonl"
    diagnostic_path = tmp_path / "teacher.diagnostics.jsonl"

    result = collect_workspace_run(
        run_path=run_path,
        diagnostic_path=diagnostic_path,
        contract=contract(
            examples,
            policy="teacher",
            teacher=teacher_identity,
        ),
        examples=examples,
        worker_client=client("worker-model", api_key="worker-secret"),
        teacher_client=client("teacher-model", api_key="teacher-secret"),
    )

    assert result.solved_count == 1
    transitions = [
        record
        for record in read_records(run_path)
        if record.get("schema_version")
        == "workspace-transition-record-v1"
    ]
    assert [record["action"]["verb"] for record in transitions] == [
        "answer",
        "submit",
    ]
    assert all(record["teacher_call_id"] for record in transitions)
    assert transitions[0]["worker_call_id"] is not None
    assert transitions[1]["worker_call_id"] is None
    assert all(record["teacher_decision"] for record in transitions)
    roles = [
        record["role"]
        for record in read_records(diagnostic_path)
        if record["record_type"] == "http_attempt"
    ]
    assert roles == ["teacher", "worker", "teacher"]
    assert not router.responses["teacher-model"]
    assert not router.responses["worker-model"]


def test_fixed_collection_admits_nested_synthesis_through_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    router = ResponseRouter({
        "worker-model": [
            {"children": ["compute one part", "compute another part"]},
            {"reasoning": "first part", "answer": "2"},
            {"reasoning": "second part", "answer": "3"},
            {
                "candidates": [{
                    "reasoning": "combine the two parts",
                    "answer": "5",
                }],
                "goal": {"id": 0, "kind": "goal"},
            },
        ],
    })
    router.install(monkeypatch)
    examples = (ProblemExample("p1", "Compute 2 + 3.", "5"),)
    run_path = tmp_path / "fixed.jsonl"
    diagnostic_path = tmp_path / "fixed.diagnostics.jsonl"

    result = collect_workspace_run(
        run_path=run_path,
        diagnostic_path=diagnostic_path,
        contract=contract(examples, policy="fixed_decomposition"),
        examples=examples,
        worker_client=client("worker-model", api_key="worker-secret"),
    )

    assert result.solved_count == 1
    attempts = [
        record
        for record in read_records(diagnostic_path)
        if record["record_type"] == "http_attempt"
    ]
    assert len(attempts) == 4
    assert all(record["attempt"]["outcome"] == "accepted" for record in attempts)
    transitions = [
        record
        for record in read_records(run_path)
        if record.get("schema_version")
        == "workspace-transition-record-v1"
    ]
    assert [record["action"]["verb"] for record in transitions] == [
        "decompose",
        "answer",
        "answer",
        "synthesize",
        "submit",
    ]
    assert transitions[-2]["worker_payload"]["candidates"][0]["answer"] == "5"
    assert transitions[-1]["reward"] == 1


def test_fixed_collection_transports_only_completed_preceding_siblings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    router = ResponseRouter({
        "worker-model": [
            {"children": [
                "FIRST_TASK_SENTINEL",
                "SECOND_TASK_SENTINEL",
                "LATER_TASK_SENTINEL",
            ]},
            {"reasoning": "FIRST_RESULT_SENTINEL", "answer": "2"},
            {"reasoning": "SECOND_RESULT_SENTINEL", "answer": "3"},
            {"reasoning": "THIRD_RESULT_SENTINEL", "answer": "4"},
            {"reasoning": "combine", "answer": "5"},
        ],
    })
    router.install(monkeypatch)
    examples = (
        ProblemExample("p1", "Compute a composed value.", "PRIVATE_LABEL"),
    )

    result = collect_workspace_run(
        run_path=tmp_path / "fixed-preceding.jsonl",
        diagnostic_path=tmp_path / "fixed-preceding.diagnostics.jsonl",
        contract=contract(
            examples,
            policy="fixed_decomposition",
            child_context_mode=CHILD_CONTEXT_PRECEDING_SIBLINGS,
        ),
        examples=examples,
        worker_client=client("worker-model", api_key="worker-secret"),
    )

    assert result.solved_count == 0
    prompts = [
        request["body"]["messages"][1]["content"]
        for request in router.requests
    ]
    assert "SECOND_TASK_SENTINEL" not in prompts[1]
    assert "LATER_TASK_SENTINEL" not in prompts[1]
    assert "FIRST_TASK_SENTINEL" in prompts[2]
    assert "FIRST_RESULT_SENTINEL" in prompts[2]
    assert "LATER_TASK_SENTINEL" not in prompts[2]
    assert prompts[3].index("FIRST_RESULT_SENTINEL") < prompts[3].index(
        "SECOND_RESULT_SENTINEL"
    )
    assert "PRIVATE_LABEL" not in repr(router.requests)
