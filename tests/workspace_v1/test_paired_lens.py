from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient
from verifier_anchored_rl.workspace_v1.experiment import (
    ModelIdentity,
    load_problem_examples,
)
from verifier_anchored_rl.workspace_v1.lenses import (
    CHILD_CONTEXT_PRECEDING_SIBLINGS,
    CHILD_CONTEXT_TARGET_ONLY,
)
from verifier_anchored_rl.workspace_v1.paired_lens import (
    build_paired_contract,
    collect_paired_lens_run,
    prepare_paired_prefixes,
    state_from_observation,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUN = ROOT / (
    "experiments/runs/"
    "2026-07-14-E32-fixed-decomposition-v11-qwen25-7b-gsm20.jsonl"
)
PROBLEMS = ROOT / "experiments/problems/gsm8k-train-500.jsonl"
SOURCE_RUN_SHA256 = (
    "18c51c244c51ff8c6e63768950672d95ce9272d9ae4b67fa24946d8c6ece826b"
)
SOURCE_CONTRACT_SHA256 = (
    "59600aef4239cc5b08658ccabf6ebc29f55d447a7e610d1f202b0516cbb1ebc6"
)
PREFIX_SHA256 = (
    "06c89f18da456d4edc8169bf244de9fe5e971568537bc07f703a7468572f9e56"
)


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
    examples = load_problem_examples(
        PROBLEMS,
        start_index=100,
        episode_count=20,
    )
    return prepare_paired_prefixes(
        source_run_path=SOURCE_RUN,
        examples=examples,
        expected_source_run_contract_sha256=SOURCE_CONTRACT_SHA256,
        max_depth=3,
    )


class DynamicRouter:
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
                user = json["messages"][1]["content"]
                if user.startswith("Operation: ANSWER"):
                    preceding = "completed_preceding_child_work" in user
                elif user.startswith("Operation: SYNTHESIZE"):
                    preceding = "PRECEDING_RESULT_SENTINEL" in user
                else:
                    raise AssertionError("paired branch made an unexpected call")
                content = {
                    "reasoning": (
                        "PRECEDING_RESULT_SENTINEL"
                        if preceding
                        else "TARGET_RESULT_SENTINEL"
                    ),
                    "answer": "15" if preceding else "21",
                }
                router.requests.append({
                    "url": url,
                    "headers": dict(headers or {}),
                    "body": deepcopy(json),
                })
                payload = {
                    "choices": [{
                        "message": {"content": json_module.dumps(content)}
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


def read_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_sealed_e32_preparation_selects_exact_shared_prefixes():
    result = prepared()

    assert result.source_run_sha256 == SOURCE_RUN_SHA256
    assert result.source_run_contract_sha256 == SOURCE_CONTRACT_SHA256
    assert result.prefix_sha256 == PREFIX_SHA256
    assert len(result.cases) == len(result.examples) == 18
    assert result.excluded_problem_ids == (
        "dataset-row-000103",
        "dataset-row-000117",
    )
    assert result.cases[0].problem_id == "dataset-row-000100"
    assert result.cases[-1].problem_id == "dataset-row-000119"

    first = result.cases[0].prefix_state
    root_children = [
        node for node in first.nodes if getattr(node, "parent_goal", None) == 0
    ]
    candidates = [
        node for node in first.nodes if hasattr(node, "goal_id")
    ]
    assert len(root_children) == 4
    assert len(candidates) == 1
    assert candidates[0].goal_id == root_children[0].id


def test_source_preparation_rejects_wrong_contract_and_observation_tamper():
    examples = load_problem_examples(
        PROBLEMS,
        start_index=100,
        episode_count=20,
    )
    with pytest.raises(ValueError, match="source run contract differs"):
        prepare_paired_prefixes(
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


def test_paired_contract_binds_prefix_worker_and_schedule():
    source = prepared()
    contract = build_paired_contract(
        preregistration_id="E-PAIR-TEST",
        prepared=source,
        worker=identity(),
        worker_temperature=0.0,
        worker_max_tokens=700,
        worker_seed=0,
    )

    assert contract.header()["source"]["prefix_sha256"] == PREFIX_SHA256
    assert contract.header()["settings"]["branch_modes"] == [
        CHILD_CONTEXT_TARGET_ONLY,
        CHILD_CONTEXT_PRECEDING_SIBLINGS,
    ]
    assert contract.sha256 != replace(contract, worker_seed=1).sha256


def test_paired_collection_forks_exact_prefix_and_alternates_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    router = DynamicRouter()
    router.install(monkeypatch)
    source = prepared()
    contract = build_paired_contract(
        preregistration_id="E-PAIR-TEST",
        prepared=source,
        worker=identity(),
        worker_temperature=0.0,
        worker_max_tokens=700,
        worker_seed=0,
    )
    run_path = tmp_path / "paired.jsonl"
    diagnostic_path = tmp_path / "paired.diagnostics.jsonl"

    result = collect_paired_lens_run(
        run_path=run_path,
        diagnostic_path=diagnostic_path,
        contract=contract,
        prepared=source,
        worker_client=client(),
    )

    assert result.case_count == 18
    assert result.transition_count == 156
    assert len(router.requests) == 120
    assert all(
        request["headers"]["Authorization"] == "Bearer PRIVATE_TEST_TOKEN"
        for request in router.requests
    )
    records = read_records(run_path)
    case_prefixes = [
        record
        for record in records
        if record.get("record_type") == "paired_case_prefix"
    ]
    case_summaries = [
        record
        for record in records
        if record.get("record_type") == "paired_case_summary"
    ]
    assert len(case_prefixes) == len(case_summaries) == 18
    assert all(
        set(summary["rewards"])
        == {CHILD_CONTEXT_TARGET_ONLY, CHILD_CONTEXT_PRECEDING_SIBLINGS}
        for summary in case_summaries
    )
    transitions = [
        record
        for record in records
        if record.get("schema_version") == "workspace-transition-record-v1"
    ]
    for prefix in case_prefixes:
        initial = [
            record
            for record in transitions
            if record["paired_case_index"] == prefix["paired_case_index"]
            and record["paired_round_index"] == 0
        ]
        assert len(initial) == 2
        assert all(
            record["observation"] == prefix["prefix_observation"]
            for record in initial
        )
    for record in transitions:
        expected_first = (
            CHILD_CONTEXT_TARGET_ONLY
            if (record["paired_case_index"] + record["paired_round_index"]) % 2
            == 0
            else CHILD_CONTEXT_PRECEDING_SIBLINGS
        )
        if record["branch_position"] == 0:
            assert record["arm"] == expected_first
    answer_requests = [
        record["worker_request"]["user"]
        for record in transitions
        if record["action"]["verb"] == "answer"
    ]
    assert any(
        "completed_preceding_child_work" in request
        for request in answer_requests
    )
    assert any(
        "completed_preceding_child_work" not in request
        for request in answer_requests
    )
    for index in range(0, len(router.requests), 2):
        pair = router.requests[index:index + 2]
        assert len(pair) == 2
        assert pair[0]["body"]["seed"] == pair[1]["body"]["seed"]
        assert pair[0]["body"]["temperature"] == pair[1]["body"][
            "temperature"
        ]
        assert pair[0]["body"]["model"] == pair[1]["body"]["model"]
    diagnostics = read_records(diagnostic_path)
    roles = [
        record["role"]
        for record in diagnostics
        if record.get("record_type") == "http_attempt"
    ]
    assert roles[:4] == [
        "target_only_worker",
        "preceding_siblings_worker",
        "preceding_siblings_worker",
        "target_only_worker",
    ]
    combined = run_path.read_text() + diagnostic_path.read_text()
    assert "expected_answer" not in combined
    assert "PRIVATE_TEST_TOKEN" not in combined
    assert not (tmp_path / "paired.jsonl.partial").exists()
    assert not (tmp_path / "paired.diagnostics.jsonl.partial").exists()
    with pytest.raises(FileExistsError):
        collect_paired_lens_run(
            run_path=run_path,
            diagnostic_path=diagnostic_path,
            contract=contract,
            prepared=source,
            worker_client=client(),
        )
