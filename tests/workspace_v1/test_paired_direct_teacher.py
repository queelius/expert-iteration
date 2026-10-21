from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

import experiments.run_workspace_paired_direct_teacher as paired_runner
from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient
from verifier_anchored_rl.workspace_v1 import (
    ACTION_AVAILABILITY_ACTIVE_FRONTIER,
    BRANCH_DIRECT,
    BRANCH_TEACHER,
    CHILD_CONTEXT_PRECEDING_SIBLINGS,
    ROOT_RESULT_VERIFIER_READY,
    REVISE_ANSWER_CONTAINER_INSTRUCTION,
    REVISE_RESULT_ANSWER_CONTAINER,
    REVISE_RESULT_OPEN_OBJECT,
    TEACHER_CHOICE_MENU_INDEX,
    TEACHER_INDEX_PROMPT_VERSION,
    ModelIdentity,
    ProblemExample,
    build_paired_direct_teacher_contract,
    build_teacher_request,
    collect_paired_direct_teacher_run,
)


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


def client(model: str, secret: str) -> OpenAICompatibleLLMClient:
    return OpenAICompatibleLLMClient(
        base_url="https://example.test/v1",
        model=model,
        api_key=secret,
        timeout_s=30.0,
        max_retries=0,
        max_backoff_retries=0,
        guided_json=False,
    )


def examples() -> tuple[ProblemExample, ...]:
    return (
        ProblemExample("p1", "Return the scalar five.", "5"),
        ProblemExample("p2", "Return the scalar seven.", "7"),
    )


def contract(
    items: tuple[ProblemExample, ...],
    *,
    revise_result_mode: str = REVISE_RESULT_OPEN_OBJECT,
):
    return build_paired_direct_teacher_contract(
        preregistration_id="E-PAIR-DIRECT-TEACHER-TEST",
        examples=items,
        turn_limit=8,
        max_depth=3,
        child_context_mode=CHILD_CONTEXT_PRECEDING_SIBLINGS,
        root_result_mode=ROOT_RESULT_VERIFIER_READY,
        action_availability_mode=ACTION_AVAILABILITY_ACTIVE_FRONTIER,
        teacher_choice_mode=TEACHER_CHOICE_MENU_INDEX,
        worker=identity("worker-model"),
        worker_temperature=0.0,
        worker_max_tokens=700,
        worker_seed=0,
        teacher=identity("teacher-model"),
        teacher_temperature=0.0,
        teacher_max_tokens=64,
        teacher_seed=100,
        revise_result_mode=revise_result_mode,
    )


class ResponseRouter:
    def __init__(self):
        self.responses = {
            "worker-model": [
                {"reasoning": "direct correct", "answer": "5"},
                {"reasoning": "direct wrong", "answer": "0"},
                {"reasoning": "revision correct", "answer": "7"},
            ],
            "teacher-model": [
                {"action_index": 2},
                {"action_index": 1},
                {"action_index": 2},
            ],
        }
        self.requests: list[dict[str, object]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        httpx = pytest.importorskip("httpx")
        router = self

        class Response:
            status_code = 200

            def __init__(self, payload: Mapping[str, object]):
                self._payload = dict(payload)
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
                    "body": deepcopy(json),
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


def read_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_indexed_teacher_prompt_retains_action_semantics():
    items = examples()
    request_contract = contract(items)
    assert TEACHER_INDEX_PROMPT_VERSION.endswith("v1.1")

    from verifier_anchored_rl.workspace_v1 import WorkspaceEnvironment

    class NoCallWorker:
        def complete_json(self, *args, **kwargs):
            raise AssertionError("worker should not be called")

    environment = WorkspaceEnvironment(
        problem=items[0].problem,
        expected_answer=items[0].expected_answer,
        worker=NoCallWorker(),
        turn_limit=request_contract.turn_limit,
        max_depth=request_contract.max_depth,
        action_availability_mode=ACTION_AVAILABILITY_ACTIVE_FRONTIER,
    )
    state = environment.initial_state()
    request = build_teacher_request(
        state,
        max_depth=3,
        action_availability_mode=ACTION_AVAILABILITY_ACTIVE_FRONTIER,
        choice_mode=TEACHER_CHOICE_MENU_INDEX,
    )

    assert "DECOMPOSE creates unverified child tasks" in request.system
    assert "SUBMIT invokes the symbolic checker" in request.system


def test_contract_binds_source_models_and_every_redesign_mode():
    base = contract(examples())
    header = base.header()

    assert header["settings"]["branches"] == [
        BRANCH_DIRECT,
        BRANCH_TEACHER,
    ]
    assert header["settings"]["action_availability_mode"] == (
        ACTION_AVAILABILITY_ACTIVE_FRONTIER
    )
    assert header["settings"]["teacher_choice_mode"] == (
        TEACHER_CHOICE_MENU_INDEX
    )
    assert header["contracts"]["collection_purpose"] == "evaluation-only-v1"
    alternatives = (
        replace(base, worker_seed=1),
        replace(base, teacher_seed=101),
        replace(base, action_availability_mode="all-affordances-v1"),
        replace(base, teacher_choice_mode="verb-target-choice-v1"),
    )
    assert all(item.sha256 != base.sha256 for item in alternatives)

    revised = replace(
        base,
        revise_result_mode=REVISE_RESULT_ANSWER_CONTAINER,
    )
    revised_header = revised.header()
    assert revised_header["schema_version"].endswith("v1.1")
    assert revised_header["run_contract_version"].endswith("v1.1")
    assert revised_header["contracts"]["revise_result"] == (
        "workspace-revise-result-contract-v1"
    )
    assert revised_header["settings"]["revise_result_mode"] == (
        REVISE_RESULT_ANSWER_CONTAINER
    )


def test_e37_legacy_contract_digest_remains_exact():
    root = Path(__file__).resolve().parents[2]
    items = tuple(
        ProblemExample.from_record(json.loads(line))
        for line in (root / "datasets/casework-v1/test.jsonl")
        .read_text(encoding="ascii")
        .splitlines()
    )
    worker = ModelIdentity(
        provider="huggingface-inference-providers:featherless-ai",
        model="Qwen/Qwen2.5-3B-Instruct:featherless-ai",
        revision="aa8e72537993ba99e69dfaafa59ed015b17504d1",
        base_url="https://router.huggingface.co/v1",
        guided_json=False,
        timeout_s=120.0,
        parse_retries=1,
        backoff_retries=6,
    )
    teacher = ModelIdentity(
        provider="openai",
        model="gpt-4.1-mini-2025-04-14",
        revision="gpt-4.1-mini-2025-04-14",
        base_url="https://api.openai.com/v1",
        guided_json=False,
        timeout_s=120.0,
        parse_retries=1,
        backoff_retries=6,
    )
    legacy = build_paired_direct_teacher_contract(
        preregistration_id="E37",
        examples=items,
        turn_limit=12,
        max_depth=3,
        child_context_mode=CHILD_CONTEXT_PRECEDING_SIBLINGS,
        root_result_mode=ROOT_RESULT_VERIFIER_READY,
        action_availability_mode=ACTION_AVAILABILITY_ACTIVE_FRONTIER,
        teacher_choice_mode=TEACHER_CHOICE_MENU_INDEX,
        worker=worker,
        worker_temperature=0.0,
        worker_max_tokens=700,
        worker_seed=0,
        teacher=teacher,
        teacher_temperature=0.0,
        teacher_max_tokens=64,
        teacher_seed=0,
    )

    assert legacy.sha256 == (
        "d71464f51c27f2080f385cfcffc5bc47524596324237b8dded0fd971c5ef2f28"
    )


def test_runner_prepares_contract_without_credentials_or_network(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    problems = tmp_path / "problems.jsonl"
    problems.write_text(
        '{"id":"p1","problem":"Return five.","answer":"5"}\n',
        encoding="ascii",
    )
    result = paired_runner.main([
        "--print-contracts",
        "--preregistration-id",
        "E-PAIR-PREPARE-TEST",
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
        "--child-context-mode",
        CHILD_CONTEXT_PRECEDING_SIBLINGS,
        "--root-result-mode",
        ROOT_RESULT_VERIFIER_READY,
        "--action-availability-mode",
        ACTION_AVAILABILITY_ACTIVE_FRONTIER,
        "--teacher-choice-mode",
        TEACHER_CHOICE_MENU_INDEX,
        "--revise-result-mode",
        REVISE_RESULT_ANSWER_CONTAINER,
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
        "--teacher-provider",
        "test-provider",
        "--teacher-model",
        "teacher-model",
        "--teacher-revision",
        "teacher-revision",
        "--teacher-base-url",
        "https://example.test/v1",
        "--no-teacher-guided-json",
        "--teacher-timeout-s",
        "30",
        "--teacher-parse-retries",
        "0",
        "--teacher-backoff-retries",
        "0",
        "--teacher-temperature",
        "0.0",
        "--teacher-max-tokens",
        "64",
        "--teacher-seed",
        "100",
    ])

    assert result == 0
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["case_count"] == 1
    assert prepared["schema_version"].endswith("v1.1")
    assert prepared["settings"]["revise_result_mode"] == (
        REVISE_RESULT_ANSWER_CONTAINER
    )
    assert len(prepared["run_contract_sha256"]) == 64


def test_collection_shares_prefix_and_teacher_can_recover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    router = ResponseRouter()
    router.install(monkeypatch)
    items = examples()
    run_path = tmp_path / "paired.jsonl"
    diagnostic_path = tmp_path / "paired.diagnostics.jsonl"

    result = collect_paired_direct_teacher_run(
        run_path=run_path,
        diagnostic_path=diagnostic_path,
        contract=contract(
            items,
            revise_result_mode=REVISE_RESULT_ANSWER_CONTAINER,
        ),
        examples=items,
        worker_client=client("worker-model", "WORKER_SECRET"),
        teacher_client=client("teacher-model", "TEACHER_SECRET"),
    )

    assert result.case_count == 2
    assert result.transition_count == 7
    assert result.solved_by_branch == {
        BRANCH_DIRECT: 1,
        BRANCH_TEACHER: 2,
    }
    assert len(router.requests) == 6
    assert sum(
        request["body"]["model"] == "worker-model"
        for request in router.requests
    ) == 3
    revision_request = next(
        request
        for request in router.requests
        if REVISE_ANSWER_CONTAINER_INSTRUCTION
        in request["body"]["messages"][1]["content"]
    )
    assert revision_request["body"]["model"] == "worker-model"

    records = read_records(run_path)
    transitions = [
        record
        for record in records
        if record.get("schema_version")
        == "workspace-paired-direct-teacher-transition-v1"
    ]
    assert all(not record["behavior_cloning_eligible"] for record in transitions)
    for case_index in range(2):
        case = [
            record
            for record in transitions
            if record["paired_case_index"] == case_index
        ]
        prefix = next(
            record for record in case if record["paired_phase"] == "shared_prefix"
        )
        direct = next(
            record
            for record in case
            if record["paired_phase"] == "direct_submission"
        )
        teacher_first = next(
            record
            for record in case
            if record["paired_phase"] == "teacher_continuation"
        )
        assert direct["observation"] == prefix["next_observation"]
        assert teacher_first["observation"] == prefix["next_observation"]
        assert direct["worker_call_id"] is None
        assert direct["teacher_call_id"] is None
    summaries = [
        record
        for record in records
        if record.get("record_type") == "paired_case_summary"
    ]
    assert [record["paired_outcome"] for record in summaries] == [
        "both",
        "teacher_only",
    ]

    diagnostics = read_records(diagnostic_path)
    roles = [
        record["role"]
        for record in diagnostics
        if record.get("record_type") == "http_attempt"
    ]
    assert roles == [
        "shared_prefix_worker",
        "teacher_controller",
        "shared_prefix_worker",
        "teacher_controller",
        "teacher_continuation_worker",
        "teacher_controller",
    ]
    combined = run_path.read_text() + diagnostic_path.read_text()
    assert "WORKER_SECRET" not in combined
    assert "TEACHER_SECRET" not in combined
    assert "expected_answer" not in combined
    assert not (tmp_path / "paired.jsonl.partial").exists()
    assert not (tmp_path / "paired.diagnostics.jsonl.partial").exists()
