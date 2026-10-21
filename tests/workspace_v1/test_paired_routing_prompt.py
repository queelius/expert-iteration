from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

import experiments.run_workspace_paired_routing_prompt as routing_runner
from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient
from verifier_anchored_rl.workspace_v1 import (
    ACTION_AVAILABILITY_ACTIVE_FRONTIER,
    PAIRED_ROUTING_PROMPT_DECISION_VERSION,
    TEACHER_CHOICE_MENU_INDEX,
    TEACHER_PROMPT_GENERIC,
    TEACHER_PROMPT_REVISION_ROUTING,
    build_paired_routing_prompt_contract,
    build_teacher_request,
    collect_paired_routing_prompt_run,
    load_problem_examples,
    prepare_routing_replays,
)


ROOT = Path(__file__).resolve().parents[2]
REVISION_RUN = ROOT / (
    "experiments/runs/"
    "2026-07-17-E41-paired-revision-prompt-qwen25-3b-"
    "casework-test60.jsonl"
)
REVISION_DIAGNOSTICS = ROOT / (
    "experiments/runs/"
    "2026-07-17-E41-paired-revision-prompt-qwen25-3b-"
    "casework-test60.diagnostics.jsonl"
)
TEACHER_SOURCE_RUN = ROOT / (
    "experiments/runs/"
    "2026-07-17-E39-paired-direct-teacher-revise-answer-"
    "gpt41mini-qwen25-3b-casework-test60.jsonl"
)
PROBLEMS = ROOT / "datasets/casework-v1/test.jsonl"
REVISION_RUN_SHA256 = (
    "58e25eee19fa9f16506d6253e1e7c6ae73b326126f1b13feba00a3c6cd7645f6"
)
REVISION_DIAGNOSTIC_SHA256 = (
    "3fc3a683eff07e85d755fd614fb5bc4c8afd042f8c7f416d26d9247ac07d90ce"
)
REVISION_CONTRACT_SHA256 = (
    "611aaf66dba10fb3be5ba5ca5e08f490f2c747387656ac7e471bef1e8d04ae11"
)
TEACHER_SOURCE_RUN_SHA256 = (
    "c2aab70b5e71df573a1aefcd1d2310435d001dc906f178473423a509d9fc86f0"
)
TEACHER_SOURCE_CONTRACT_SHA256 = (
    "845d816ba9062529e72090008c70bb2e15798c21a9c86e23f330be1bbb65398a"
)
SELECTION_SHA256 = (
    "08f9cdd31a69d1c7b2aba729b42adad538563567f402ef46cef1745a2ada9a5c"
)
E42_CONTRACT_SHA256 = (
    "0b86b77ad5167178d13c0c304f346813f8688da52d244acffd8e03a142ed731c"
)


def prepared():
    examples = load_problem_examples(PROBLEMS, start_index=0, episode_count=60)
    return prepare_routing_replays(
        revision_run_path=REVISION_RUN,
        revision_diagnostic_path=REVISION_DIAGNOSTICS,
        expected_revision_run_contract_sha256=REVISION_CONTRACT_SHA256,
        teacher_source_run_path=TEACHER_SOURCE_RUN,
        expected_teacher_source_run_sha256=TEACHER_SOURCE_RUN_SHA256,
        expected_teacher_source_run_contract_sha256=(
            TEACHER_SOURCE_CONTRACT_SHA256
        ),
        examples=examples,
        max_depth=3,
    )


def client(source) -> OpenAICompatibleLLMClient:
    identity = source.teacher
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


def test_preparation_seals_all_independent_revisions_and_teacher_settings():
    source = prepared()

    assert source.revision_run_sha256 == REVISION_RUN_SHA256
    assert source.revision_diagnostic_sha256 == REVISION_DIAGNOSTIC_SHA256
    assert source.selection_sha256 == SELECTION_SHA256
    assert len(source.cases) == 60
    assert {case.source_turn for case in source.cases} == {1}
    assert all(case.protected_candidate_id == 1 for case in source.cases)
    assert all(case.revised_candidate_id == 2 for case in source.cases)
    assert source.teacher.model == "gpt-4.1-mini-2025-04-14"


def test_routing_prompt_changes_only_the_indexed_system_instruction():
    source = prepared()
    case = source.cases[0]
    generic = build_teacher_request(
        case.state,
        max_depth=3,
        action_availability_mode=ACTION_AVAILABILITY_ACTIVE_FRONTIER,
        choice_mode=TEACHER_CHOICE_MENU_INDEX,
        prompt_mode=TEACHER_PROMPT_GENERIC,
    )
    routing = build_teacher_request(
        case.state,
        max_depth=3,
        action_availability_mode=ACTION_AVAILABILITY_ACTIVE_FRONTIER,
        choice_mode=TEACHER_CHOICE_MENU_INDEX,
        prompt_mode=TEACHER_PROMPT_REVISION_ROUTING,
    )

    assert generic.system != routing.system
    assert generic.user == routing.user
    assert generic.schema == routing.schema
    assert "global planner" in generic.system
    assert "routing controller" in routing.system


def test_contract_binds_replays_teacher_and_prompt_modes():
    source = prepared()
    contract = build_paired_routing_prompt_contract(
        preregistration_id="E42",
        prepared=source,
    )

    assert contract.sha256 == E42_CONTRACT_SHA256
    assert contract.header()["source"]["selection_sha256"] == SELECTION_SHA256
    assert contract.header()["settings"]["prompt_modes"] == [
        TEACHER_PROMPT_GENERIC,
        TEACHER_PROMPT_REVISION_ROUTING,
    ]
    assert contract.sha256 != replace(contract, teacher_seed=1).sha256


def test_runner_prepares_without_credentials_or_network(
    capsys: pytest.CaptureFixture[str],
):
    result = routing_runner.main([
        "--print-contracts",
        "--preregistration-id",
        "E42",
        "--revision-run",
        str(REVISION_RUN),
        "--revision-diagnostics",
        str(REVISION_DIAGNOSTICS),
        "--revision-run-contract-sha256",
        REVISION_CONTRACT_SHA256,
        "--teacher-source-run",
        str(TEACHER_SOURCE_RUN),
        "--teacher-source-run-sha256",
        TEACHER_SOURCE_RUN_SHA256,
        "--teacher-source-run-contract-sha256",
        TEACHER_SOURCE_CONTRACT_SHA256,
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
    assert output["run_contract_sha256"] == E42_CONTRACT_SHA256


class RoutingRouter:
    def __init__(self):
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
                system = json["messages"][0]["content"]
                action_index = 1 if "routing controller" in system else 2
                content = {"action_index": action_index}
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


def test_collection_pairs_prompt_only_and_replays_fixed_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = prepared()
    router = RoutingRouter()
    router.install(monkeypatch)
    contract = build_paired_routing_prompt_contract(
        preregistration_id="E42",
        prepared=source,
    )
    run_path = tmp_path / "routing.jsonl"
    diagnostic_path = tmp_path / "routing.diagnostics.jsonl"

    result = collect_paired_routing_prompt_run(
        run_path=run_path,
        diagnostic_path=diagnostic_path,
        contract=contract,
        prepared=source,
        teacher_client=client(source),
    )

    assert result.case_count == 60
    assert result.decision_count == 120
    assert result.solved_by_mode == {
        TEACHER_PROMPT_GENERIC: 43,
        TEACHER_PROMPT_REVISION_ROUTING: 55,
    }
    assert result.paired_outcomes == {
        "both": 43,
        "routing_only": 12,
        "neither": 5,
    }
    assert len(router.requests) == 120
    records = read_records(run_path)
    decisions = [
        record
        for record in records
        if record.get("schema_version") == PAIRED_ROUTING_PROMPT_DECISION_VERSION
    ]
    assert len(decisions) == 120
    generic = [
        record for record in decisions if record["arm"] == TEACHER_PROMPT_GENERIC
    ]
    routing = [
        record
        for record in decisions
        if record["arm"] == TEACHER_PROMPT_REVISION_ROUTING
    ]
    assert all(record["routing_action"]["verb"] == "submit" for record in generic)
    assert all(record["routing_action"]["verb"] == "revise" for record in routing)
    assert all(record["terminal_submit_transition"] is not None for record in decisions)
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
    assert not (tmp_path / "routing.jsonl.partial").exists()
    assert not (tmp_path / "routing.diagnostics.jsonl.partial").exists()
