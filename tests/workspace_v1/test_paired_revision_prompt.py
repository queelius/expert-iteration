from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

import experiments.run_workspace_paired_revision_prompt as paired_runner
from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient
from verifier_anchored_rl.verifier_adapter import check_answer
from verifier_anchored_rl.workspace_v1 import (
    CHILD_CONTEXT_PRECEDING_SIBLINGS,
    PAIRED_REVISION_PROMPT_TRANSITION_VERSION,
    REVISE_ANSWER_CONTAINER_INSTRUCTION,
    REVISE_INDEPENDENT_ANSWER_CONTAINER_INSTRUCTION,
    REVISE_RESULT_ANSWER_CONTAINER,
    REVISE_RESULT_INDEPENDENT_ANSWER_CONTAINER,
    ROOT_RESULT_VERIFIER_READY,
    build_paired_revision_prompt_contract,
    build_worker_request,
    collect_paired_revision_prompt_run,
    load_problem_examples,
    prepare_fixed_revisions,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUN = ROOT / (
    "experiments/runs/"
    "2026-07-17-E39-paired-direct-teacher-revise-answer-"
    "gpt41mini-qwen25-3b-casework-test60.jsonl"
)
SOURCE_DIAGNOSTICS = ROOT / (
    "experiments/runs/"
    "2026-07-17-E39-paired-direct-teacher-revise-answer-"
    "gpt41mini-qwen25-3b-casework-test60.diagnostics.jsonl"
)
PROBLEMS = ROOT / "datasets/casework-v1/test.jsonl"
SOURCE_CONTRACT_SHA256 = (
    "845d816ba9062529e72090008c70bb2e15798c21a9c86e23f330be1bbb65398a"
)
E41_CONTRACT_SHA256 = (
    "611aaf66dba10fb3be5ba5ca5e08f490f2c747387656ac7e471bef1e8d04ae11"
)


def prepared():
    examples = load_problem_examples(PROBLEMS, start_index=0, episode_count=60)
    return prepare_fixed_revisions(
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


def test_independent_request_changes_only_revise_instruction():
    source = prepared()
    case = source.cases[0]
    retained = build_worker_request(
        case.state,
        case.action,
        child_context_mode=CHILD_CONTEXT_PRECEDING_SIBLINGS,
        root_result_mode=ROOT_RESULT_VERIFIER_READY,
        revise_result_mode=REVISE_RESULT_ANSWER_CONTAINER,
    )
    independent = build_worker_request(
        case.state,
        case.action,
        child_context_mode=CHILD_CONTEXT_PRECEDING_SIBLINGS,
        root_result_mode=ROOT_RESULT_VERIFIER_READY,
        revise_result_mode=REVISE_RESULT_INDEPENDENT_ANSWER_CONTAINER,
    )

    assert retained.system == independent.system
    assert retained.schema == independent.schema
    assert retained.schema_validation_mode == independent.schema_validation_mode
    assert REVISE_ANSWER_CONTAINER_INSTRUCTION in retained.user
    assert REVISE_INDEPENDENT_ANSWER_CONTAINER_INSTRUCTION in independent.user
    assert retained.user.replace(
        REVISE_ANSWER_CONTAINER_INSTRUCTION,
        REVISE_INDEPENDENT_ANSWER_CONTAINER_INSTRUCTION,
        1,
    ) == independent.user


def test_contract_binds_prompt_modes_and_source():
    source = prepared()
    contract = build_paired_revision_prompt_contract(
        preregistration_id="E41",
        prepared=source,
    )

    assert contract.sha256 == E41_CONTRACT_SHA256
    assert contract.header()["settings"]["modes"] == [
        REVISE_RESULT_ANSWER_CONTAINER,
        REVISE_RESULT_INDEPENDENT_ANSWER_CONTAINER,
    ]
    assert contract.sha256 != replace(contract, worker_seed=1).sha256


def test_runner_prepares_without_credentials_or_network(
    capsys: pytest.CaptureFixture[str],
):
    result = paired_runner.main([
        "--print-contracts",
        "--preregistration-id",
        "E41",
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
    assert output["case_count"] == 60
    assert output["run_contract_sha256"] == E41_CONTRACT_SHA256


class PromptRouter:
    def __init__(self, source):
        self.examples = tuple(source.examples)
        baseline_correct = []
        for case, example in zip(source.cases, source.examples, strict=True):
            candidate = case.state.nodes[case.action.target]
            if check_answer(candidate.answer, example.expected_answer).success:
                baseline_correct.append(example.problem_id)
        self.independent_regressions = set(baseline_correct[:10])
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
                    item for item in router.examples if item.problem in user
                )
                independent = (
                    REVISE_INDEPENDENT_ANSWER_CONTAINER_INSTRUCTION in user
                )
                answer = example.expected_answer
                if independent and example.problem_id in router.independent_regressions:
                    answer = "999999999"
                content = {"reasoning": "paired revision", "answer": answer}
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


def test_collection_pairs_only_the_prompt_factor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = prepared()
    router = PromptRouter(source)
    router.install(monkeypatch)
    contract = build_paired_revision_prompt_contract(
        preregistration_id="E41",
        prepared=source,
    )
    run_path = tmp_path / "paired-prompt.jsonl"
    diagnostic_path = tmp_path / "paired-prompt.diagnostics.jsonl"

    result = collect_paired_revision_prompt_run(
        run_path=run_path,
        diagnostic_path=diagnostic_path,
        contract=contract,
        prepared=source,
        worker_client=client(source),
    )

    assert result.case_count == 60
    assert result.transition_count == 120
    assert result.answer_containers_by_mode == {
        REVISE_RESULT_ANSWER_CONTAINER: 60,
        REVISE_RESULT_INDEPENDENT_ANSWER_CONTAINER: 60,
    }
    assert result.correctness_cells_by_mode[
        REVISE_RESULT_ANSWER_CONTAINER
    ] == {"both": 43, "revision_only": 17}
    assert result.correctness_cells_by_mode[
        REVISE_RESULT_INDEPENDENT_ANSWER_CONTAINER
    ] == {"baseline_only": 10, "both": 33, "revision_only": 17}
    assert result.independent_corpus_adequate
    assert len(router.requests) == 120

    records = read_records(run_path)
    transitions = [
        record
        for record in records
        if record.get("schema_version")
        == PAIRED_REVISION_PROMPT_TRANSITION_VERSION
    ]
    assert len(transitions) == 120
    for case_index in range(60):
        branches = [
            record
            for record in transitions
            if record["case_index"] == case_index
        ]
        assert len(branches) == 2
        assert branches[0]["observation"] == branches[1]["observation"]
        assert branches[0]["worker_request"]["schema"] == (
            branches[1]["worker_request"]["schema"]
        )
        first_mode = (
            REVISE_RESULT_ANSWER_CONTAINER
            if case_index % 2 == 0
            else REVISE_RESULT_INDEPENDENT_ANSWER_CONTAINER
        )
        assert next(
            record for record in branches if record["branch_position"] == 0
        )["arm"] == first_mode
    diagnostics = read_records(diagnostic_path)
    attempts = [
        record
        for record in diagnostics
        if record.get("record_type") == "http_attempt"
    ]
    assert len(attempts) == 120
    combined = run_path.read_text() + diagnostic_path.read_text()
    assert "PRIVATE_TEST_TOKEN" not in combined
    assert "expected_answer" not in combined
    assert not (tmp_path / "paired-prompt.jsonl.partial").exists()
    assert not (tmp_path / "paired-prompt.diagnostics.jsonl.partial").exists()
