from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

import experiments.run_workspace_fixed_revision as fixed_runner
from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient
from verifier_anchored_rl.workspace_v1 import (
    FIXED_REVISION_TRANSITION_VERSION,
    REVISE_ANSWER_CONTAINER_INSTRUCTION,
    build_fixed_revision_contract,
    collect_fixed_revision_run,
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
SOURCE_RUN_SHA256 = (
    "c2aab70b5e71df573a1aefcd1d2310435d001dc906f178473423a509d9fc86f0"
)
SOURCE_DIAGNOSTIC_SHA256 = (
    "5dec0e494a86e9d26042e9c311e672e0f91638eb3bd5fcef3bf02f7d11a3fb81"
)
SOURCE_CONTRACT_SHA256 = (
    "845d816ba9062529e72090008c70bb2e15798c21a9c86e23f330be1bbb65398a"
)
SELECTION_SHA256 = (
    "8a55a2b22407603f0b1dc10bd2a17fff50fe0f208a7f2bc6bd00710c622fc6cf"
)
E40_CONTRACT_SHA256 = (
    "56a3fa807f92218f14c6e02a15da4448fab97daf70c6f42aa8ac5978b5a895e0"
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


def test_preparation_seals_all_e39_shared_prefix_states():
    source = prepared()

    assert source.source_run_sha256 == SOURCE_RUN_SHA256
    assert source.source_diagnostic_sha256 == SOURCE_DIAGNOSTIC_SHA256
    assert source.selection_sha256 == SELECTION_SHA256
    assert len(source.cases) == 60
    assert [case.source_episode_index for case in source.cases] == list(range(60))
    assert {case.source_turn for case in source.cases} == {1}
    assert all(case.action.target == 1 for case in source.cases)


def test_contract_binds_exact_source_and_fixed_worker_settings():
    source = prepared()
    contract = build_fixed_revision_contract(
        preregistration_id="E40",
        prepared=source,
    )

    assert contract.sha256 == E40_CONTRACT_SHA256
    assert contract.header()["source"]["selection_sha256"] == SELECTION_SHA256
    assert contract.header()["settings"]["revise_result_mode"] == (
        "workspace-revise-answer-container-v1"
    )
    assert contract.sha256 != replace(contract, worker_seed=1).sha256


def test_runner_prepares_without_credentials_or_network(
    capsys: pytest.CaptureFixture[str],
):
    result = fixed_runner.main([
        "--print-contracts",
        "--preregistration-id",
        "E40",
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
    assert output["selection_sha256"] == SELECTION_SHA256
    assert output["run_contract_sha256"] == E40_CONTRACT_SHA256


class RevisionRouter:
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
                assert REVISE_ANSWER_CONTAINER_INSTRUCTION in user
                example = next(
                    item for item in router.examples if item.problem in user
                )
                content = {
                    "reasoning": "fixed revised answer",
                    "answer": example.expected_answer,
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


def test_collection_creates_auditable_selection_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = prepared()
    router = RevisionRouter(source.examples)
    router.install(monkeypatch)
    contract = build_fixed_revision_contract(
        preregistration_id="E40",
        prepared=source,
    )
    run_path = tmp_path / "fixed-revision.jsonl"
    diagnostic_path = tmp_path / "fixed-revision.diagnostics.jsonl"

    result = collect_fixed_revision_run(
        run_path=run_path,
        diagnostic_path=diagnostic_path,
        contract=contract,
        prepared=source,
        worker_client=client(source),
    )

    assert result.case_count == result.transition_count == 60
    assert result.answer_container_count == 60
    assert result.correctness_cells == {"both": 43, "revision_only": 17}
    assert not result.selection_corpus_adequate
    assert len(router.requests) == 60
    records = read_records(run_path)
    transitions = [
        record
        for record in records
        if record.get("schema_version") == FIXED_REVISION_TRANSITION_VERSION
    ]
    assert len(transitions) == 60
    assert all(not record["behavior_cloning_eligible"] for record in transitions)
    assert all(record["answer_container"] for record in transitions)
    diagnostics = read_records(diagnostic_path)
    attempts = [
        record
        for record in diagnostics
        if record.get("record_type") == "http_attempt"
    ]
    assert len(attempts) == 60
    assert all(record["role"] == "fixed_revision_worker" for record in attempts)
    combined = run_path.read_text() + diagnostic_path.read_text()
    assert "PRIVATE_TEST_TOKEN" not in combined
    assert "expected_answer" not in combined
    assert not (tmp_path / "fixed-revision.jsonl.partial").exists()
    assert not (tmp_path / "fixed-revision.diagnostics.jsonl.partial").exists()
