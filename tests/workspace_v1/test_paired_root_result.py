from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

import experiments.run_workspace_paired_root_result as paired_runner
from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient
from verifier_anchored_rl.workspace_v1 import (
    ROOT_RESULT_FREE,
    ROOT_RESULT_VERIFIER_READY,
    ROOT_RESULT_VERIFIER_READY_INSTRUCTION,
    ModelIdentity,
    build_paired_root_result_contract,
    collect_paired_root_result_run,
    load_problem_examples,
    prepare_root_finalizations,
    state_from_observation,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUN = ROOT / (
    "experiments/runs/"
    "2026-07-16-E35-fixed-preceding-qwen25-7b-casework-test60.jsonl"
)
PROBLEMS = ROOT / "datasets/casework-v1/test.jsonl"
SOURCE_RUN_SHA256 = (
    "b8c4dc7f8296b553b3cd4ff2c22348cbfac895508fa2cb48a79a56020619685b"
)
SOURCE_CONTRACT_SHA256 = (
    "709e52804c08a7a976007394c83155da5899081971668afad86b93bbf46e1ceb"
)
PREFIX_SHA256 = (
    "4c06db248b9659e253412666e57170b6a5b8bf82ba6f1ba3b4f8f28595dbf8db"
)
EXCLUDED_ID = "cw1-1dcd933e33fd"


def identity() -> ModelIdentity:
    return ModelIdentity(
        provider="test-provider",
        model="worker-model",
        revision="worker-revision",
        base_url="https://example.test/v1",
        guided_json=False,
        timeout_s=30.0,
        parse_retries=0,
        backoff_retries=0,
    )


def client() -> OpenAICompatibleLLMClient:
    return OpenAICompatibleLLMClient(
        base_url="https://example.test/v1",
        model="worker-model",
        api_key="PRIVATE_TEST_TOKEN",
        timeout_s=30.0,
        max_retries=0,
        max_backoff_retries=0,
        guided_json=False,
    )


def prepared():
    examples = load_problem_examples(PROBLEMS, start_index=0, episode_count=60)
    return prepare_root_finalizations(
        source_run_path=SOURCE_RUN,
        examples=examples,
        expected_source_run_contract_sha256=SOURCE_CONTRACT_SHA256,
        max_depth=3,
    )


class RootResultRouter:
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
                ready = ROOT_RESULT_VERIFIER_READY_INSTRUCTION in user
                answer = (
                    example.expected_answer
                    if ready
                    else f"x = {example.expected_answer}"
                )
                content = {
                    "reasoning": "paired root result test",
                    "answer": answer,
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


def read_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_sealed_e35_preparation_selects_exact_root_finalization_states():
    result = prepared()

    assert result.source_run_sha256 == SOURCE_RUN_SHA256
    assert result.source_run_contract_sha256 == SOURCE_CONTRACT_SHA256
    assert result.prefix_sha256 == PREFIX_SHA256
    assert len(result.cases) == len(result.examples) == 59
    assert result.excluded_problem_ids == (EXCLUDED_ID,)
    assert result.cases[0].problem_id == "cw1-b05baf11892d"
    assert result.cases[-1].problem_id == "cw1-4bbd45ef40eb"
    assert all(case.prefix_state.turn + 2 <= 12 for case in result.cases)


def test_root_preparation_rejects_wrong_contract_and_observation_tamper():
    examples = load_problem_examples(PROBLEMS, start_index=0, episode_count=60)
    with pytest.raises(ValueError, match="source run contract differs"):
        prepare_root_finalizations(
            source_run_path=SOURCE_RUN,
            examples=examples,
            expected_source_run_contract_sha256="0" * 64,
            max_depth=3,
        )

    source = prepared()
    observation = source.cases[0].source_record(max_depth=3)[
        "prefix_observation"
    ]
    tampered = deepcopy(observation)
    tampered["remaining_turns"] += 1
    with pytest.raises(ValueError, match="does not reconstruct exactly"):
        state_from_observation(tampered, max_depth=3)


def test_root_contract_binds_source_treatment_worker_and_schedule():
    source = prepared()
    contract = build_paired_root_result_contract(
        preregistration_id="E-ROOT-PAIR-TEST",
        prepared=source,
        worker=identity(),
        worker_temperature=0.0,
        worker_max_tokens=700,
        worker_seed=0,
    )

    assert contract.header()["source"]["prefix_sha256"] == PREFIX_SHA256
    assert contract.header()["settings"]["branch_modes"] == [
        ROOT_RESULT_FREE,
        ROOT_RESULT_VERIFIER_READY,
    ]
    assert contract.sha256 != replace(contract, worker_seed=1).sha256
    with pytest.raises(ValueError, match="branch modes differ"):
        replace(
            contract,
            branch_modes=tuple(reversed(contract.branch_modes)),
        )


def test_root_result_runner_prepares_without_credentials_or_network(
    capsys: pytest.CaptureFixture[str],
):
    result = paired_runner.main([
        "--print-contracts",
        "--preregistration-id",
        "E-ROOT-PREPARE-TEST",
        "--source-run",
        str(SOURCE_RUN),
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
    output = json.loads(capsys.readouterr().out)
    assert output["case_count"] == 59
    assert output["prefix_sha256"] == PREFIX_SHA256
    assert output["excluded_problem_ids"] == [EXCLUDED_ID]
    assert len(output["run_contract_sha256"]) == 64


def test_paired_root_collection_isolates_prompt_and_submits_both_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = prepared()
    router = RootResultRouter(source.examples)
    router.install(monkeypatch)
    contract = build_paired_root_result_contract(
        preregistration_id="E-ROOT-PAIR-TEST",
        prepared=source,
        worker=identity(),
        worker_temperature=0.0,
        worker_max_tokens=700,
        worker_seed=0,
    )
    run_path = tmp_path / "paired-root.jsonl"
    diagnostic_path = tmp_path / "paired-root.diagnostics.jsonl"

    result = collect_paired_root_result_run(
        run_path=run_path,
        diagnostic_path=diagnostic_path,
        contract=contract,
        prepared=source,
        worker_client=client(),
    )

    assert result.case_count == 59
    assert result.transition_count == 236
    assert result.solved_by_mode == {
        ROOT_RESULT_FREE: 0,
        ROOT_RESULT_VERIFIER_READY: 59,
    }
    assert len(router.requests) == 118
    assert all(
        request["headers"]["Authorization"] == "Bearer PRIVATE_TEST_TOKEN"
        for request in router.requests
    )
    for index in range(0, len(router.requests), 2):
        first, second = router.requests[index:index + 2]
        first_body = first["body"]
        second_body = second["body"]
        assert first_body["seed"] == second_body["seed"]
        assert first_body["temperature"] == second_body["temperature"]
        assert first_body["model"] == second_body["model"]
        first_user = first_body["messages"][1]["content"]
        second_user = second_body["messages"][1]["content"]
        if ROOT_RESULT_VERIFIER_READY_INSTRUCTION in first_user:
            ready_user, free_user = first_user, second_user
        else:
            free_user, ready_user = first_user, second_user
        assert ready_user.replace(
            ROOT_RESULT_VERIFIER_READY_INSTRUCTION,
            "",
            1,
        ) == free_user

    records = read_records(run_path)
    prefixes = [
        record
        for record in records
        if record.get("record_type") == "paired_case_prefix"
    ]
    summaries = [
        record
        for record in records
        if record.get("record_type") == "paired_case_summary"
    ]
    transitions = [
        record
        for record in records
        if record.get("schema_version") == "workspace-transition-record-v1"
    ]
    assert len(prefixes) == len(summaries) == 59
    assert all(
        summary["paired_outcome"] == "verifier_ready_only"
        for summary in summaries
    )
    assert all(all(summary["generation_complete"].values()) for summary in summaries)
    for prefix in prefixes:
        generations = [
            record
            for record in transitions
            if record["paired_case_index"] == prefix["paired_case_index"]
            and record["paired_phase"] == "generation"
        ]
        assert len(generations) == 2
        assert all(
            record["observation"] == prefix["prefix_observation"]
            for record in generations
        )
        expected_first = (
            ROOT_RESULT_FREE
            if prefix["paired_case_index"] % 2 == 0
            else ROOT_RESULT_VERIFIER_READY
        )
        first = next(
            record for record in generations if record["branch_position"] == 0
        )
        assert first["arm"] == expected_first
    submissions = [
        record for record in transitions if record["paired_phase"] == "submission"
    ]
    assert len(submissions) == 118
    assert all(record["worker_call_id"] is None for record in submissions)
    assert all(record["worker_request"] is None for record in submissions)

    diagnostics = read_records(diagnostic_path)
    roles = [
        record["role"]
        for record in diagnostics
        if record.get("record_type") == "http_attempt"
    ]
    assert roles[:4] == [
        "free_root_result_worker",
        "verifier_ready_root_result_worker",
        "verifier_ready_root_result_worker",
        "free_root_result_worker",
    ]
    combined = run_path.read_text() + diagnostic_path.read_text()
    assert "expected_answer" not in combined
    assert "PRIVATE_TEST_TOKEN" not in combined
    assert not (tmp_path / "paired-root.jsonl.partial").exists()
    assert not (tmp_path / "paired-root.diagnostics.jsonl.partial").exists()
    with pytest.raises(FileExistsError):
        collect_paired_root_result_run(
            run_path=run_path,
            diagnostic_path=diagnostic_path,
            contract=contract,
            prepared=source,
            worker_client=client(),
        )
