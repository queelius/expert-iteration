from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

import experiments.run_workspace_paired_revise as paired_runner
from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient
from verifier_anchored_rl.workspace_v1 import (
    ACTION_AVAILABILITY_ACTIVE_FRONTIER,
    CHILD_CONTEXT_PRECEDING_SIBLINGS,
    REVISE_ANSWER_CONTAINER_INSTRUCTION,
    REVISE_ANSWER_CONTAINER_SCHEMA,
    REVISE_OPEN_OBJECT_INSTRUCTION,
    REVISE_RESULT_ANSWER_CONTAINER,
    REVISE_RESULT_OPEN_OBJECT,
    ROOT_RESULT_VERIFIER_READY,
    WORKSPACE_CANDIDATE_SCHEMA,
    Action,
    CandidateNode,
    Verb,
    WorkspaceEnvironment,
    build_paired_revise_contract,
    build_worker_request,
    collect_paired_revise_run,
    load_problem_examples,
    prepare_revise_replays,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUN = ROOT / (
    "experiments/runs/"
    "2026-07-17-E37-paired-direct-teacher-gpt41mini-"
    "qwen25-3b-casework-test60.jsonl"
)
SOURCE_DIAGNOSTICS = ROOT / (
    "experiments/runs/"
    "2026-07-17-E37-paired-direct-teacher-gpt41mini-"
    "qwen25-3b-casework-test60.diagnostics.jsonl"
)
PROBLEMS = ROOT / "datasets/casework-v1/test.jsonl"
SOURCE_RUN_SHA256 = (
    "67b8d353122f86201dd20ad8b39545318e3af5ac634e3e837646c17f18fb0c48"
)
SOURCE_DIAGNOSTIC_SHA256 = (
    "dda68061f83dba7d49d3f8a8c4c7d2998350230bedb809eb6d8d17dfd6e8ac20"
)
SOURCE_CONTRACT_SHA256 = (
    "d71464f51c27f2080f385cfcffc5bc47524596324237b8dded0fd971c5ef2f28"
)
SELECTION_SHA256 = (
    "a82fae5ea8bc907cdd318ae4f5ddd048a8ea91208d1b07e169f817c95ea91121"
)


def prepared():
    examples = load_problem_examples(PROBLEMS, start_index=0, episode_count=60)
    return prepare_revise_replays(
        source_run_path=SOURCE_RUN,
        source_diagnostic_path=SOURCE_DIAGNOSTICS,
        examples=examples,
        expected_source_run_contract_sha256=SOURCE_CONTRACT_SHA256,
        max_depth=3,
    )


def client(source) -> OpenAICompatibleLLMClient:
    identity = source.worker
    return OpenAICompatibleLLMClient(
        base_url=identity.base_url,
        model=identity.model,
        api_key="PRIVATE_TEST_TOKEN",
        timeout_s=identity.timeout_s,
        max_retries=identity.parse_retries,
        max_backoff_retries=identity.backoff_retries,
        guided_json=identity.guided_json,
    )


def read_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_sealed_e37_preparation_selects_all_exact_revise_states():
    source = prepared()

    assert source.source_run_sha256 == SOURCE_RUN_SHA256
    assert source.source_diagnostic_sha256 == SOURCE_DIAGNOSTIC_SHA256
    assert source.selection_sha256 == SELECTION_SHA256
    assert len(source.cases) == 14
    assert [
        (case.source_episode_index, case.source_turn)
        for case in source.cases
    ] == [
        (7, 1),
        (9, 1),
        (12, 4),
        (17, 1),
        (17, 2),
        (17, 3),
        (22, 1),
        (27, 1),
        (33, 1),
        (38, 1),
        (47, 1),
        (50, 1),
        (52, 1),
        (52, 2),
    ]
    assert sum(case.source_generation_error for case in source.cases) == 8
    assert not any(case.source_answer_container for case in source.cases)
    assert sum(case.target_is_root for case in source.cases) == 13


def test_revise_answer_container_changes_only_instruction_and_schema():
    source = prepared()
    case = source.cases[0]
    legacy = build_worker_request(
        case.state,
        case.action,
        child_context_mode=CHILD_CONTEXT_PRECEDING_SIBLINGS,
        root_result_mode=ROOT_RESULT_VERIFIER_READY,
        revise_result_mode=REVISE_RESULT_OPEN_OBJECT,
    )
    strict = build_worker_request(
        case.state,
        case.action,
        child_context_mode=CHILD_CONTEXT_PRECEDING_SIBLINGS,
        root_result_mode=ROOT_RESULT_VERIFIER_READY,
        revise_result_mode=REVISE_RESULT_ANSWER_CONTAINER,
    )

    assert legacy.schema == WORKSPACE_CANDIDATE_SCHEMA
    assert strict.schema == REVISE_ANSWER_CONTAINER_SCHEMA
    assert legacy.system == strict.system
    assert REVISE_OPEN_OBJECT_INSTRUCTION in legacy.user
    assert REVISE_ANSWER_CONTAINER_INSTRUCTION in strict.user
    assert legacy.user.replace(
        REVISE_OPEN_OBJECT_INSTRUCTION,
        REVISE_ANSWER_CONTAINER_INSTRUCTION,
        1,
    ) == strict.user


class StaticWorker:
    def __init__(self, payload):
        self.payload = payload

    def complete_json(self, *args, **kwargs):
        return deepcopy(self.payload)


def test_strict_revise_runtime_rejects_lens_echo_and_accepts_answer():
    source = prepared()
    case = source.cases[0]
    example = next(
        example
        for example in source.examples
        if example.problem_id == case.problem_id
    )
    common = {
        "problem": example.problem,
        "expected_answer": example.expected_answer,
        "turn_limit": source.turn_limit,
        "max_depth": source.max_depth,
        "child_context_mode": CHILD_CONTEXT_PRECEDING_SIBLINGS,
        "root_result_mode": ROOT_RESULT_VERIFIER_READY,
        "revise_result_mode": REVISE_RESULT_ANSWER_CONTAINER,
        "action_availability_mode": ACTION_AVAILABILITY_ACTIVE_FRONTIER,
    }
    echoed = WorkspaceEnvironment(
        worker=StaticWorker({"target": {"answer": "5"}}),
        **common,
    ).step(case.state, case.action)
    accepted = WorkspaceEnvironment(
        worker=StaticWorker({"reasoning": "revised", "answer": "5"}),
        **common,
    ).step(case.state, case.action)

    assert echoed.generation_error
    assert not accepted.generation_error
    created_id = accepted.next_state.history[-1].created_ids[0]
    candidate = accepted.next_state.nodes[created_id]
    assert isinstance(candidate, CandidateNode)
    assert candidate.answer == "5"
    assert candidate.revises == case.action.target


def test_contract_binds_source_selection_and_modes():
    source = prepared()
    contract = build_paired_revise_contract(
        preregistration_id="E-REVISE-PAIR-TEST",
        prepared=source,
    )
    header = contract.header()

    assert header["source"]["selection_sha256"] == SELECTION_SHA256
    assert header["settings"]["modes"] == [
        REVISE_RESULT_OPEN_OBJECT,
        REVISE_RESULT_ANSWER_CONTAINER,
    ]
    assert contract.sha256 != replace(contract, worker_seed=1).sha256
    with pytest.raises(ValueError, match="revision modes differ"):
        replace(contract, modes=tuple(reversed(contract.modes)))


def test_runner_prepares_without_credentials_or_network(
    capsys: pytest.CaptureFixture[str],
):
    result = paired_runner.main([
        "--print-contracts",
        "--preregistration-id",
        "E-REVISE-PREPARE-TEST",
        "--source-run",
        str(SOURCE_RUN),
        "--source-diagnostics",
        str(SOURCE_DIAGNOSTICS),
        "--source-run-contract-sha256",
        SOURCE_CONTRACT_SHA256,
        "--problems",
        str(PROBLEMS),
        "--start-index",
        "0",
        "--episode-count",
        "60",
        "--max-depth",
        "3",
    ])

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["case_count"] == 14
    assert output["selection_sha256"] == SELECTION_SHA256
    assert output["source_run_sha256"] == SOURCE_RUN_SHA256


class ReviseRouter:
    def __init__(self, examples):
        self.examples = tuple(examples)
        self.requests: list[dict[str, object]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        httpx = pytest.importorskip("httpx")
        router = self

        class Response:
            status_code = 200

            def __init__(self, payload):
                self._payload = payload
                self.text = json.dumps(payload)

            def raise_for_status(self):
                return None

            def json(self):
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
                user = json["messages"][1]["content"]
                example = next(
                    example
                    for example in router.examples
                    if example.problem in user
                )
                if REVISE_ANSWER_CONTAINER_INSTRUCTION in user:
                    content = {
                        "reasoning": "explicit revised answer",
                        "answer": example.expected_answer,
                    }
                else:
                    content = {
                        "target": {"answer": example.expected_answer},
                        "target_candidates": [],
                    }
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


def test_collection_pairs_exact_states_and_records_container_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = prepared()
    router = ReviseRouter(source.examples)
    router.install(monkeypatch)
    contract = build_paired_revise_contract(
        preregistration_id="E-REVISE-PAIR-TEST",
        prepared=source,
    )
    run_path = tmp_path / "paired-revise.jsonl"
    diagnostic_path = tmp_path / "paired-revise.diagnostics.jsonl"

    result = collect_paired_revise_run(
        run_path=run_path,
        diagnostic_path=diagnostic_path,
        contract=contract,
        prepared=source,
        worker_client=client(source),
    )

    assert result.case_count == 14
    assert result.transition_count == 28
    assert result.answer_containers_by_mode == {
        REVISE_RESULT_OPEN_OBJECT: 0,
        REVISE_RESULT_ANSWER_CONTAINER: 14,
    }
    assert result.root_symbolic_success_by_mode == {
        REVISE_RESULT_OPEN_OBJECT: 0,
        REVISE_RESULT_ANSWER_CONTAINER: 13,
    }
    assert len(router.requests) == 28
    assert all(
        request["headers"]["Authorization"] == "Bearer PRIVATE_TEST_TOKEN"
        for request in router.requests
    )

    records = read_records(run_path)
    sources = [
        record
        for record in records
        if record.get("record_type") == "paired_case_source"
    ]
    transitions = [
        record
        for record in records
        if record.get("schema_version") == "workspace-paired-revise-transition-v1"
    ]
    summaries = [
        record
        for record in records
        if record.get("record_type") == "paired_case_summary"
    ]
    assert len(sources) == len(summaries) == 14
    assert len(transitions) == 28
    assert all(
        summary["paired_outcome"] == "answer_container_only"
        for summary in summaries
    )
    assert all(not transition["behavior_cloning_eligible"] for transition in transitions)
    for case_index, source_record in enumerate(sources):
        branches = [
            record
            for record in transitions
            if record["paired_case_index"] == case_index
        ]
        assert len(branches) == 2
        assert all(
            record["observation"] == source_record["source_observation"]
            for record in branches
        )
        first_mode = (
            REVISE_RESULT_OPEN_OBJECT
            if case_index % 2 == 0
            else REVISE_RESULT_ANSWER_CONTAINER
        )
        assert next(
            record for record in branches if record["branch_position"] == 0
        )["arm"] == first_mode
        open_branch = next(
            record
            for record in branches
            if record["arm"] == REVISE_RESULT_OPEN_OBJECT
        )
        answer_branch = next(
            record
            for record in branches
            if record["arm"] == REVISE_RESULT_ANSWER_CONTAINER
        )
        assert open_branch["worker_request"]["user"].replace(
            REVISE_OPEN_OBJECT_INSTRUCTION,
            REVISE_ANSWER_CONTAINER_INSTRUCTION,
            1,
        ) == answer_branch["worker_request"]["user"]

    diagnostics = read_records(diagnostic_path)
    attempts = [
        record
        for record in diagnostics
        if record.get("record_type") == "http_attempt"
    ]
    assert len(attempts) == 28
    assert attempts[0]["role"] == "open_object_revise_worker"
    assert attempts[2]["role"] == "answer_container_revise_worker"
    combined = run_path.read_text() + diagnostic_path.read_text()
    assert "PRIVATE_TEST_TOKEN" not in combined
    assert "expected_answer" not in combined
    assert not (tmp_path / "paired-revise.jsonl.partial").exists()
    assert not (tmp_path / "paired-revise.diagnostics.jsonl.partial").exists()
