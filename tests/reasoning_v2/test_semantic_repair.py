import json
import threading
from collections import Counter
from copy import copy, deepcopy
from pathlib import Path

import pytest

import experiments.probe_reasoning_v2_semantic_repair as probe_cli
from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient
from verifier_anchored_rl.reasoning_v2.casework_source import (
    CaseworkProblemSource,
)
from verifier_anchored_rl.reasoning_v2.contracts import (
    canonical_json_bytes,
    canonical_json_sha256,
)
from verifier_anchored_rl.reasoning_v2.diagnostics import (
    DIAGNOSTIC_ATTEMPT_VERSION,
    DIAGNOSTIC_CALL_VERSION,
)
from verifier_anchored_rl.reasoning_v2.semantic_repair import (
    SEMANTIC_REPAIR_CASE_VERSION,
    SEMANTIC_REPAIR_FEEDBACK_CONSTRAINTS,
    SEMANTIC_REPAIR_FEEDBACK_MAP_VERSION,
    SEMANTIC_REPAIR_FEEDBACK_SCHEMA_VERSION,
    SEMANTIC_REPAIR_FEEDBACK_SUFFIX,
    SEMANTIC_REPAIR_HEADER_VERSION,
    SEMANTIC_REPAIR_SUMMARY_VERSION,
    SemanticRepairProbeAuthorization,
    SemanticRepairProbeValidationError,
    build_semantic_repair_feedback,
    build_semantic_repair_request,
    prepare_semantic_repair_probe,
    read_semantic_repair_probe_artifact,
    run_semantic_repair_probe,
    semantic_repair_feedback_map_record,
    semantic_repair_feedback_map_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "datasets/casework-v1/train.jsonl"
RUN_PATH = ROOT / "experiments/runs/2026-07-13-E23-v2-hosted-smoke5.jsonl"
DIAGNOSTIC_PATH = ROOT / (
    "experiments/runs/2026-07-13-E23-v2-hosted-smoke5.diagnostics.jsonl"
)
E24_PATH = ROOT / (
    "experiments/runs/2026-07-13-E24-v2-lens-v21-paired66.jsonl"
)
E25_PATH = ROOT / (
    "experiments/runs/2026-07-13-E25-v2-domain-parser-rescore33.jsonl"
)
E26_OUTPUT = (
    "experiments/runs/2026-07-13-E26-v2-decompose-repair33.jsonl"
)

RUN_SHA256 = "5efea19bd29619aba71d6d147135aababd7ded474230918b7d77b72d8b7368f6"
DIAGNOSTIC_SHA256 = (
    "0418043eb3abb50d86ad03c01946cc83602e76c8bd2c91f41defb87d3af6e369"
)
SOURCE_SHA256 = (
    "f3be145c1e536a98d4207c54ef85b5e0e516bce777c6ccd3d2a1f846c8881426"
)
SOURCE_ENVIRONMENT_SHA256 = (
    "9316c93bc2cdf9b2ce507a989882af69669f8bf886e1f2ad6ce802c77dfdcfa7"
)
E24_SHA256 = "507c3b818ebee6c160bec021b84c60290f4af1a6fdcae725d2fa4efae74a5efc"
E25_SHA256 = "f2be14e62cd04d680b454545cd253bfdb05c8ed5e404212b5b90bfcb7c8b1307"
E24_CONTRACT_SHA256 = (
    "3ef15fcccc0c3ce4fc109ae07ef3ad20ef29cb7677146b9fdce8f329352b019c"
)
E25_CONTRACT_SHA256 = (
    "ec307795e8dc39ae70515ca5036f5442f5784b72d40ed269fa3bb654ccbd842a"
)
E24_ENVIRONMENT_SHA256 = (
    "664deedf1caa44efff16daac1888c2e3966aa37e9a2631a62036643a9e681c71"
)
CORRECTED_ENVIRONMENT_SHA256 = (
    "3396ce7464c8bf4a8cd60c5aa26172c02dc4ed4589ca7eb9e2bcfd906119ee54"
)
CORRECTED_PROFILE_SHA256 = (
    "42ee4bf1f481dd92bebac990b8fc0c95d280a4a5c1054b0b9dc4d5070f6581c8"
)

EXPECTED_FEEDBACK = {
    "unsupported_domain": (
        "Every child must use the target variable in one allowed relational "
        "form. Because a decomposition has two or more children, never use "
        "Reals as a child or as a scalar or comparison operand."
    ),
    "decomposition_domain_overlap": (
        "Make the child domains pairwise disjoint. Never include Reals beside "
        "another child. At every shared boundary, use the same finite scalar "
        "on both sides and include it in exactly one child: pair a strict "
        "upper side with an inclusive lower side, or an inclusive upper side "
        "with a strict lower side."
    ),
    "decomposition_domain_not_exhaustive": (
        "Make the union of child domains equal the target domain with no gap. "
        "Adjacent children must meet at the same boundary, included by exactly "
        "one child."
    ),
}
EXPECTED_FEEDBACK_SUFFIX = (
    "Revise the failed domains once. Return only the same one-field JSON "
    "object required above."
)

VALID_DOMAINS = {
    "cw1-ccdf9e1567dc": ["x < 4", "x >= 4"],
    "cw1-2640597e2425": ["x < 0", "0 <= x < 12"],
    "cw1-52e3b6b3ec25": ["x < 2", "x >= 2"],
    "cw1-ddad3be834f4": ["x < 4", "x >= 4"],
    "cw1-4ba0fc91c9e8": ["x < 0", "x >= 0"],
}


class _Response:
    status_code = 200
    headers = {}

    def __init__(self, content):
        self._payload = {
            "choices": [{"message": {"content": content}}],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        }
        self.text = json.dumps(self._payload, separators=(",", ":"))

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _install_responses(monkeypatch, responses):
    import httpx

    queue = list(responses)
    calls = []
    lock = threading.Lock()

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            with lock:
                calls.append({
                    "url": url,
                    "headers": dict(headers or {}),
                    "body": deepcopy(json),
                })
                response = queue.pop(0)
                if isinstance(response, BaseException):
                    raise response
                return response

    monkeypatch.setattr(httpx, "Client", _Client)
    monkeypatch.setattr(
        "verifier_anchored_rl.llm_client.time.sleep", lambda delay: None
    )
    monkeypatch.setattr(
        "verifier_anchored_rl.llm_client.random.uniform",
        lambda low, high: 0.0,
    )
    return calls


def _source():
    return CaseworkProblemSource.from_jsonl(SOURCE_PATH).select_slice(0, 5)


@pytest.fixture(scope="module")
def prepared():
    return prepare_semantic_repair_probe(
        source_run_path=RUN_PATH,
        source_diagnostic_path=DIAGNOSTIC_PATH,
        e24_artifact_path=E24_PATH,
        e25_artifact_path=E25_PATH,
        problem_source=_source(),
        planned_output_path=E26_OUTPUT,
    )


def _generator_client(prepared, *, api_key="GENERATOR_SECRET"):
    transport = prepared.contract.generator_transport
    return OpenAICompatibleLLMClient(
        base_url=transport.base_url,
        model=transport.model,
        api_key=api_key,
        timeout_s=transport.timeout_s,
        max_retries=transport.parse_retry_budget,
        max_backoff_retries=transport.backoff_retry_budget,
        guided_json=transport.guided_json,
    )


def _artifact_records(path):
    return [json.loads(line) for line in path.read_text("ascii").splitlines()]


def _write_records(path, records):
    path.write_bytes(
        b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    )


def _path_snapshot(path):
    return None if not path.exists() else path.read_bytes()


def _valid_content(case):
    return json.dumps(
        {"domains": VALID_DOMAINS[case.problem_id]},
        separators=(",", ":"),
    )


def _authorization(prepared, **overrides):
    contract = prepared.contract
    values = {
        "preregistration_id": "E26-test",
        "expected_source_run_sha256": contract.source_run_sha256,
        "expected_source_diagnostic_sha256": (
            contract.source_diagnostic_sha256
        ),
        "expected_source_environment_contract_sha256": (
            contract.source_environment_contract_sha256
        ),
        "expected_request_environment_contract_sha256": (
            contract.request_environment_contract_sha256
        ),
        "expected_source_collection_contract_sha256": (
            contract.source_collection_contract_sha256
        ),
        "expected_source_problem_source_sha256": (
            contract.source_problem_source_sha256
        ),
        "expected_e24_artifact_sha256": contract.e24_artifact_sha256,
        "expected_e24_probe_contract_sha256": (
            contract.e24_probe_contract_sha256
        ),
        "expected_e24_case_manifest_sha256": (
            contract.e24_case_manifest_sha256
        ),
        "expected_e25_artifact_sha256": contract.e25_artifact_sha256,
        "expected_e25_rescore_contract_sha256": (
            contract.e25_rescore_contract_sha256
        ),
        "expected_e25_rescore_case_manifest_sha256": (
            contract.e25_rescore_case_manifest_sha256
        ),
        "expected_e25_result_manifest_sha256": (
            contract.e25_result_manifest_sha256
        ),
        "expected_scoring_environment_contract_sha256": (
            contract.scoring_environment_contract_sha256
        ),
        "expected_corrected_profile_sha256": (
            contract.corrected_profile_sha256
        ),
        "expected_feedback_map_sha256": contract.feedback_map_sha256,
        "expected_case_manifest_sha256": contract.case_manifest_sha256,
        "expected_base_request_manifest_sha256": (
            contract.base_request_manifest_sha256
        ),
        "expected_repair_request_manifest_sha256": (
            contract.repair_request_manifest_sha256
        ),
        "expected_generator_sha256": contract.generator_sha256,
        "expected_generator_transport_sha256": (
            contract.generator_transport_sha256
        ),
        "expected_generation_settings_sha256": (
            contract.generation_settings_sha256
        ),
        "expected_contract_sha256": contract.sha256,
        "expected_source_decision_count": contract.source_decision_count,
        "expected_case_count": contract.case_count,
        "expected_problem_count": contract.problem_count,
        "expected_output_path": contract.planned_output_path,
    }
    values.update(overrides)
    return SemanticRepairProbeAuthorization(**values)


def _cli_source_args(prepared):
    return [
        "--source-run",
        str(RUN_PATH),
        "--source-diagnostic",
        str(DIAGNOSTIC_PATH),
        "--e24-artifact",
        str(E24_PATH),
        "--e25-artifact",
        str(E25_PATH),
        "--problems",
        str(SOURCE_PATH),
        "--start-index",
        "0",
        "--episode-count",
        "5",
        "--expected-source-decision-count",
        "66",
        "--expected-decompose-count",
        "33",
        "--expected-problem-count",
        "5",
        "--output",
        prepared.contract.planned_output_path,
    ]


def _cli_authorization_args(prepared, *, contract_sha256=None):
    contract = prepared.contract
    return [
        "--preregistration-id",
        "E26-test",
        "--expected-source-run-sha256",
        contract.source_run_sha256,
        "--expected-source-diagnostic-sha256",
        contract.source_diagnostic_sha256,
        "--expected-source-environment-sha256",
        contract.source_environment_contract_sha256,
        "--expected-request-environment-sha256",
        contract.request_environment_contract_sha256,
        "--expected-source-collection-sha256",
        contract.source_collection_contract_sha256,
        "--expected-source-problem-sha256",
        contract.source_problem_source_sha256,
        "--expected-e24-artifact-sha256",
        contract.e24_artifact_sha256,
        "--expected-e24-probe-contract-sha256",
        contract.e24_probe_contract_sha256,
        "--expected-e24-case-manifest-sha256",
        contract.e24_case_manifest_sha256,
        "--expected-e25-artifact-sha256",
        contract.e25_artifact_sha256,
        "--expected-e25-rescore-contract-sha256",
        contract.e25_rescore_contract_sha256,
        "--expected-e25-rescore-manifest-sha256",
        contract.e25_rescore_case_manifest_sha256,
        "--expected-e25-result-manifest-sha256",
        contract.e25_result_manifest_sha256,
        "--expected-scoring-environment-sha256",
        contract.scoring_environment_contract_sha256,
        "--expected-corrected-profile-sha256",
        contract.corrected_profile_sha256,
        "--expected-feedback-map-sha256",
        contract.feedback_map_sha256,
        "--expected-case-manifest-sha256",
        contract.case_manifest_sha256,
        "--expected-base-request-manifest-sha256",
        contract.base_request_manifest_sha256,
        "--expected-repair-request-manifest-sha256",
        contract.repair_request_manifest_sha256,
        "--expected-generator-sha256",
        contract.generator_sha256,
        "--expected-generator-transport-sha256",
        contract.generator_transport_sha256,
        "--expected-generation-settings-sha256",
        contract.generation_settings_sha256,
        "--expected-contract-sha256",
        contract.sha256 if contract_sha256 is None else contract_sha256,
        "--generator-api-key-env",
        "MISSING_HF_TOKEN",
    ]


def test_feedback_map_and_terminal_instruction_are_frozen():
    assert SEMANTIC_REPAIR_FEEDBACK_SCHEMA_VERSION == (
        "decomposition-semantic-feedback-v1"
    )
    assert SEMANTIC_REPAIR_FEEDBACK_CONSTRAINTS == EXPECTED_FEEDBACK
    assert semantic_repair_feedback_map_record() == {
        "schema_version": SEMANTIC_REPAIR_FEEDBACK_MAP_VERSION,
        "constraints": EXPECTED_FEEDBACK,
    }
    assert semantic_repair_feedback_map_sha256() == canonical_json_sha256(
        semantic_repair_feedback_map_record()
    )
    assert SEMANTIC_REPAIR_FEEDBACK_SUFFIX == EXPECTED_FEEDBACK_SUFFIX


def test_prepare_selects_exact_e24_e25_rows_and_cross_version_bindings(
    prepared,
):
    prepared.validate_trusted()
    contract = prepared.contract.to_record()
    assert contract["source_run_sha256"] == RUN_SHA256
    assert contract["source_diagnostic_sha256"] == DIAGNOSTIC_SHA256
    assert contract["source_problem_source_sha256"] == SOURCE_SHA256
    assert contract["source_environment_contract_sha256"] == (
        SOURCE_ENVIRONMENT_SHA256
    )
    assert contract["e24_artifact_sha256"] == E24_SHA256
    assert contract["e25_artifact_sha256"] == E25_SHA256
    assert contract["e24_probe_contract_sha256"] == E24_CONTRACT_SHA256
    assert contract["e25_rescore_contract_sha256"] == E25_CONTRACT_SHA256
    assert contract["request_environment_contract_sha256"] == (
        E24_ENVIRONMENT_SHA256
    )
    assert contract["scoring_environment_contract_sha256"] == (
        CORRECTED_ENVIRONMENT_SHA256
    )
    assert contract["corrected_profile_sha256"] == CORRECTED_PROFILE_SHA256
    assert contract["cost_basis"] == {
        "currency": "USD",
        "input_usd_per_million_tokens": "0.30",
        "output_usd_per_million_tokens": "0.30",
        "rate_basis": "hf-together-published-rate-observed-2026-07-13",
        "interpretation_contract": (
            "estimate-from-provider-reported-usage-not-billing-receipt-v1"
        ),
    }
    assert contract["request_environment_contract_sha256"] != (
        contract["scoring_environment_contract_sha256"]
    )
    assert len(prepared.cases) == 33
    assert [case.index for case in prepared.cases] == list(range(33))
    manifests = [case.manifest_record() for case in prepared.cases]
    assert [record["e24_case_index"] for record in manifests] == list(
        range(0, 66, 2)
    )
    assert {case.action.verb.value for case in prepared.cases} == {
        "decompose"
    }
    assert {case.action.target for case in prepared.cases} == {0}
    assert [case.effective_seed for case in prepared.cases] == [
        index % 16 for index in range(0, 66, 2)
    ]
    assert Counter(case.problem_id for case in prepared.cases) == {
        "cw1-ccdf9e1567dc": 8,
        "cw1-2640597e2425": 8,
        "cw1-52e3b6b3ec25": 8,
        "cw1-ddad3be834f4": 8,
        "cw1-4ba0fc91c9e8": 1,
    }
    assert Counter(case.corrected_reason_code for case in prepared.cases) == {
        "unsupported_domain": 15,
        "decomposition_domain_overlap": 15,
        "decomposition_domain_not_exhaustive": 3,
    }
    assert all(
        case.failed_domains
        == tuple(child.domain for child in case.source_draft.child_obligations)
        for case in prepared.cases
    )
    assert contract["case_manifest_sha256"] == canonical_json_sha256(
        manifests
    )


def test_requests_reuse_e24_bytes_and_append_only_exact_feedback(prepared):
    for case in prepared.cases:
        feedback = build_semantic_repair_feedback(
            case.failed_domains, case.corrected_reason_code
        )
        assert feedback == {
            "schema_version": SEMANTIC_REPAIR_FEEDBACK_SCHEMA_VERSION,
            "failed_domains": list(case.failed_domains),
            "reason_code": case.corrected_reason_code,
            "repair_constraint": EXPECTED_FEEDBACK[
                case.corrected_reason_code
            ],
        }
        assert feedback == case.feedback
        assert canonical_json_sha256(feedback) == case.feedback_sha256

        request, base_body, repair_body = build_semantic_repair_request(
            prepared, case
        )
        assert base_body == case.base_request_body
        assert repair_body == case.repair_request_body
        assert canonical_json_sha256(base_body) == case.base_request_sha256
        assert canonical_json_sha256(repair_body) == (
            case.repair_request_sha256
        )
        assert case.base_request_sha256 == (
            case.source_case.source_case.probe_request_sha256
        )
        assert base_body["seed"] == repair_body["seed"] == (
            case.effective_seed
        )

        assert set(base_body) == set(repair_body)
        for key in set(base_body) - {"messages"}:
            assert base_body[key] == repair_body[key]
        assert len(base_body["messages"]) == len(repair_body["messages"]) == 2
        assert base_body["messages"][0] == repair_body["messages"][0]
        assert base_body["messages"][1]["role"] == "user"
        assert repair_body["messages"][1]["role"] == "user"
        base_user = base_body["messages"][1]["content"]
        feedback_json = canonical_json_bytes(feedback).decode("ascii")
        exact_append = (
            "\n\n" + feedback_json + "\n" + EXPECTED_FEEDBACK_SUFFIX
        )
        assert repair_body["messages"][1]["content"] == (
            base_user + exact_append
        )
        assert request.user == base_user + exact_append
        assert E24_ENVIRONMENT_SHA256 in base_user
        assert E24_ENVIRONMENT_SHA256 in request.user
        assert CORRECTED_ENVIRONMENT_SHA256 not in request.user

        substituted = json.loads(canonical_json_bytes(repair_body))
        substituted["messages"][1]["content"] = substituted[
            "messages"
        ][1]["content"].replace(
            E24_ENVIRONMENT_SHA256, CORRECTED_ENVIRONMENT_SHA256
        )
        assert canonical_json_sha256(substituted) != (
            case.repair_request_sha256
        )


def test_prepare_reads_no_key_and_makes_no_http(monkeypatch):
    calls = _install_responses(monkeypatch, [])
    monkeypatch.delenv("HF_TOKEN", raising=False)

    result = prepare_semantic_repair_probe(
        source_run_path=RUN_PATH,
        source_diagnostic_path=DIAGNOSTIC_PATH,
        e24_artifact_path=E24_PATH,
        e25_artifact_path=E25_PATH,
        problem_source=_source(),
        planned_output_path=E26_OUTPUT,
    )

    result.validate_trusted()
    assert calls == []


def test_every_authorization_binding_is_checked_before_output_and_http(
    monkeypatch, prepared
):
    final_path = ROOT / E26_OUTPUT
    partial_path = ROOT / f"{E26_OUTPUT}.partial"
    final_before = _path_snapshot(final_path)
    partial_before = _path_snapshot(partial_path)
    calls = _install_responses(monkeypatch, [])
    authorization = _authorization(prepared)
    checked = []
    for field_name in authorization.__dataclass_fields__:
        if not field_name.startswith("expected_"):
            continue
        bad = copy(authorization)
        value = getattr(bad, field_name)
        if isinstance(value, int):
            replacement = value + 1
        elif field_name == "expected_output_path":
            replacement = "experiments/runs/not-authorized.jsonl"
        else:
            replacement = ("0" if value[0] != "0" else "1") + value[1:]
        object.__setattr__(bad, field_name, replacement)
        with pytest.raises(ValueError, match=r"authorized .* differs"):
            run_semantic_repair_probe(
                output_path=E26_OUTPUT,
                prepared=prepared,
                generator_client=_generator_client(prepared),
                authorization=bad,
            )
        checked.append(field_name)

    assert len(checked) == 27
    assert calls == []
    assert _path_snapshot(final_path) == final_before
    assert _path_snapshot(partial_path) == partial_before


@pytest.fixture(scope="module")
def executed_probe(tmp_path_factory):
    directory = tmp_path_factory.mktemp("semantic-repair")
    output_path = directory / "repair.jsonl"
    local_prepared = prepare_semantic_repair_probe(
        source_run_path=RUN_PATH,
        source_diagnostic_path=DIAGNOSTIC_PATH,
        e24_artifact_path=E24_PATH,
        e25_artifact_path=E25_PATH,
        problem_source=_source(),
        planned_output_path=output_path,
    )
    responses = []
    for case in local_prepared.cases:
        if case.index == 0:
            responses.append(_Response("not-json"))
        if case.index == 1:
            content = '{"domains":[" ","x >= 0"]}'
        elif case.index == 2:
            content = '{"domains":["x < 6","x >= 6"]}'
        else:
            content = _valid_content(case)
        responses.append(_Response(content))
    monkeypatch = pytest.MonkeyPatch()
    calls = _install_responses(monkeypatch, responses)
    client = _generator_client(local_prepared)
    try:
        result = run_semantic_repair_probe(
            output_path=output_path,
            prepared=local_prepared,
            generator_client=client,
            authorization=_authorization(local_prepared),
        )
    finally:
        monkeypatch.undo()
    return local_prepared, result, output_path, calls, client


def test_stubbed_probe_is_33_independent_calls_with_raw_retry_and_gates(
    executed_probe,
):
    prepared, result, output_path, calls, client = executed_probe
    assert result.output_path == output_path
    assert result.case_count == 33
    assert result.wire_usable_count == 33
    assert result.decoded_count == 32
    assert result.symbolic_applied_count == 31
    assert result.applied_problem_count == 5
    assert client.generate_calls == 34
    assert len(calls) == 34
    assert {call["url"] for call in calls} == {
        "https://router.huggingface.co/v1/chat/completions"
    }
    assert all(
        call["headers"]["Authorization"] == "Bearer GENERATOR_SECRET"
        for call in calls
    )
    assert "GENERATOR_SECRET" not in output_path.read_text("ascii")
    assert not output_path.with_name("repair.jsonl.partial").exists()

    call_offset = 0
    for case in prepared.cases:
        assert calls[call_offset]["body"] == case.repair_request_body
        assert calls[call_offset]["body"]["seed"] == case.effective_seed
        call_offset += 1
        if case.index == 0:
            assert calls[call_offset]["body"]["seed"] == (
                case.effective_seed + 1
            )
            call_offset += 1
    assert call_offset == 34

    records = _artifact_records(output_path)
    schemas = Counter(record["schema_version"] for record in records)
    assert schemas == {
        SEMANTIC_REPAIR_HEADER_VERSION: 1,
        DIAGNOSTIC_ATTEMPT_VERSION: 34,
        DIAGNOSTIC_CALL_VERSION: 33,
        SEMANTIC_REPAIR_CASE_VERSION: 33,
        SEMANTIC_REPAIR_SUMMARY_VERSION: 1,
    }
    attempts = [
        record
        for record in records
        if record["schema_version"] == DIAGNOSTIC_ATTEMPT_VERSION
    ]
    cases = [
        record
        for record in records
        if record["schema_version"] == SEMANTIC_REPAIR_CASE_VERSION
    ]
    summary = records[-1]
    assert read_semantic_repair_probe_artifact(
        output_path,
        prepared=prepared,
        authorization=_authorization(prepared),
    ) == summary
    for case, record in zip(prepared.cases, cases, strict=True):
        assert {
            key: record[key] for key in case.manifest_record()
        } == case.manifest_record()
        assert record["feedback"] == case.feedback
        assert record["wire"] == {
            "usable": True,
            "terminal_outcome": "transport_accepted",
            "final_attempt_outcome": "accepted",
            "error_type": None,
        }
        if case.index == 1:
            assert record["decode"] == {
                "status": "payload_error",
                "error_type": "PolicyPayloadError",
                "draft": None,
            }
            assert record["corrected_application"] is None
        else:
            assert record["decode"]["status"] == "decoded"
            assert record["decode"]["error_type"] is None
            expected_status = (
                "algebra_rejected" if case.index == 2 else "applied"
            )
            assert record["corrected_application"][
                "application_status"
            ] == expected_status
            expected_reason = (
                "child_not_atomic"
                if case.index == 2
                else None
            )
            assert record["corrected_application"]["reason_code"] == (
                expected_reason
            )

    first_call_id = cases[0]["repair_call_record_id"].split(":")[1]
    first_attempts = [
        attempt for attempt in attempts if attempt["call_id"] == first_call_id
    ]
    assert [attempt["outcome"] for attempt in first_attempts] == [
        "parse_error",
        "accepted",
    ]
    assert [attempt["disposition"] for attempt in first_attempts] == [
        "parse_retry",
        "return",
    ]
    assert [attempt["effective_seed"] for attempt in first_attempts] == [0, 1]

    assert summary["case_count"] == 33
    assert summary["problem_count"] == 5
    assert summary["raw_attempt_count"] == 34
    assert summary["generator_call_count"] == 33
    assert summary["wire_usable_count"] == 33
    assert summary["decode_counts"] == {
        "decoded": 32,
        "not_run": 0,
        "payload_error": 1,
    }
    assert summary["symbolic_application_counts"] == {
        "algebra_rejected": 1,
        "applied": 31,
        "not_run": 1,
    }
    assert summary["applied_problem_count"] == 5
    assert set(summary["applied_problem_ids"]) == set(VALID_DOMAINS)
    assert summary["source_reason_counts"] == {
        "decomposition_domain_not_exhaustive": 3,
        "decomposition_domain_overlap": 15,
        "unsupported_domain": 15,
    }
    assert summary["corrected_result_reason_counts"] == {
        "applied": 31,
        "child_not_atomic": 1,
        "not_run": 1,
    }
    assert summary["raw_attempt_outcome_counts"] == {
        "accepted": 33,
        "http_status_error": 0,
        "parse_error": 1,
        "provider_response_error": 0,
        "schema_error": 0,
        "transport_error": 0,
    }
    assert summary["reported_usage"] == {
        "attempt_count": 34,
        "completion_tokens": 68,
        "prompt_tokens": 102,
        "total_tokens": 170,
    }
    assert summary["reported_cost"] == {
        "currency": "USD",
        "input_usd_per_million_tokens": "0.30",
        "output_usd_per_million_tokens": "0.30",
        "rate_basis": "hf-together-published-rate-observed-2026-07-13",
        "interpretation_contract": (
            "estimate-from-provider-reported-usage-not-billing-receipt-v1"
        ),
        "prompt_cost_usd": "0.0000306",
        "completion_cost_usd": "0.0000204",
        "total_cost_usd": "0.0000510",
    }
    assert summary["primary_thresholds"] == {
        "typed_decoded_min": 30,
        "applied_min": 17,
        "applied_problem_min": 4,
        "problem_total": 5,
    }
    assert summary["typed_decoded_count"] >= 30
    assert summary["symbolic_applied_count"] >= 17
    assert summary["applied_problem_count"] >= 4

    artifact_text = output_path.read_text("ascii")
    for forbidden in (
        "controller-run-header-v2",
        "controller-transition-v2",
        "reasoning-v2-dataset",
        "counterfactual_reward",
        "counterfactual_terminal",
        "training_record",
    ):
        assert forbidden not in artifact_text


def test_strict_reader_rejects_request_result_summary_and_schema_tampering(
    tmp_path, executed_probe
):
    prepared, _, output_path, _, _ = executed_probe
    authorization = _authorization(prepared)
    with pytest.raises(TypeError, match="prepared"):
        read_semantic_repair_probe_artifact(
            output_path, authorization=authorization
        )
    with pytest.raises(TypeError, match="authorization"):
        read_semantic_repair_probe_artifact(
            output_path, prepared=prepared
        )
    original = _artifact_records(output_path)

    def preregistration_mutation(records):
        records[0]["preregistration_id"] = "E26-not-authorized"

    def request_mutation(records):
        attempt = next(
            record
            for record in records
            if record["schema_version"] == DIAGNOSTIC_ATTEMPT_VERSION
        )
        attempt["request_body"]["messages"][1]["content"] += " "

    def raw_projection_mutation(records):
        attempt = next(
            record
            for record in records
            if record["schema_version"] == DIAGNOSTIC_ATTEMPT_VERSION
            and record["outcome"] == "accepted"
        )
        attempt["raw_content"] = '{"domains":["x < 1","x >= 1"]}'

    def manifest_mutation(records):
        case = next(
            record
            for record in records
            if record["schema_version"] == SEMANTIC_REPAIR_CASE_VERSION
        )
        case["repair_request_sha256"] = "f" * 64

    def score_mutation(records):
        case = next(
            record
            for record in records
            if record["schema_version"] == SEMANTIC_REPAIR_CASE_VERSION
            and record["corrected_application"] is not None
        )
        case["corrected_application"]["coverage"]["context_digest"] = "f" * 64

    def summary_mutation(records):
        records[-1]["applied_problem_count"] = 4

    def unsupported_schema_mutation(records):
        records.insert(-1, {"schema_version": "controller-transition-v2"})

    for name, mutate in (
        ("preregistration", preregistration_mutation),
        ("request", request_mutation),
        ("raw", raw_projection_mutation),
        ("manifest", manifest_mutation),
        ("score", score_mutation),
        ("summary", summary_mutation),
        ("schema", unsupported_schema_mutation),
    ):
        records = deepcopy(original)
        mutate(records)
        path = tmp_path / f"{name}.jsonl"
        _write_records(path, records)
        with pytest.raises(SemanticRepairProbeValidationError):
            read_semantic_repair_probe_artifact(
                path,
                prepared=prepared,
                authorization=authorization,
            )


@pytest.fixture(scope="module")
def guarded_probe(tmp_path_factory):
    directory = tmp_path_factory.mktemp("semantic-repair-guard")
    output_path = directory / "guard.jsonl"
    local_prepared = prepare_semantic_repair_probe(
        source_run_path=RUN_PATH,
        source_diagnostic_path=DIAGNOSTIC_PATH,
        e24_artifact_path=E24_PATH,
        e25_artifact_path=E25_PATH,
        problem_source=_source(),
        planned_output_path=output_path,
    )
    return local_prepared, output_path


def test_existing_outputs_never_clobber_and_interruption_keeps_partial(
    monkeypatch, guarded_probe
):
    prepared, output_path = guarded_probe
    partial_path = output_path.with_name(f"{output_path.name}.partial")
    calls = _install_responses(monkeypatch, [])
    output_path.write_text("owned\n", encoding="ascii")
    with pytest.raises(FileExistsError, match="already exists"):
        run_semantic_repair_probe(
            output_path=output_path,
            prepared=prepared,
            generator_client=_generator_client(prepared),
            authorization=_authorization(prepared),
        )
    assert calls == []
    assert output_path.read_text("ascii") == "owned\n"

    output_path.unlink()
    partial_path.write_text("partial-owned\n", encoding="ascii")
    with pytest.raises(FileExistsError, match="already exists"):
        run_semantic_repair_probe(
            output_path=output_path,
            prepared=prepared,
            generator_client=_generator_client(prepared),
            authorization=_authorization(prepared),
        )
    assert calls == []
    assert partial_path.read_text("ascii") == "partial-owned\n"

    partial_path.unlink()
    interrupted_calls = _install_responses(
        monkeypatch, [KeyboardInterrupt("test interruption")]
    )
    with pytest.raises(KeyboardInterrupt, match="test interruption"):
        run_semantic_repair_probe(
            output_path=output_path,
            prepared=prepared,
            generator_client=_generator_client(prepared),
            authorization=_authorization(prepared),
        )
    assert len(interrupted_calls) == 1
    assert not output_path.exists()
    assert partial_path.exists()
    with pytest.raises(SemanticRepairProbeValidationError):
        read_semantic_repair_probe_artifact(
            partial_path,
            prepared=prepared,
            authorization=_authorization(prepared),
        )


def test_cli_help_exits_before_preparation(monkeypatch):
    monkeypatch.setattr(
        probe_cli,
        "prepare_semantic_repair_probe",
        lambda **kwargs: pytest.fail("help attempted preparation"),
    )

    with pytest.raises(SystemExit) as error:
        probe_cli.main(["--help"])

    assert error.value.code == 0


def test_cli_print_contract_reads_no_key_and_makes_no_http(
    monkeypatch, capsys, prepared
):
    final_path = ROOT / E26_OUTPUT
    partial_path = ROOT / f"{E26_OUTPUT}.partial"
    final_before = _path_snapshot(final_path)
    partial_before = _path_snapshot(partial_path)
    calls = _install_responses(monkeypatch, [])
    monkeypatch.setattr(
        probe_cli, "prepare_semantic_repair_probe", lambda **kwargs: prepared
    )
    monkeypatch.setattr(
        probe_cli,
        "_secret",
        lambda *args: pytest.fail("print mode attempted secret lookup"),
    )
    monkeypatch.setattr(
        probe_cli,
        "run_semantic_repair_probe",
        lambda **kwargs: pytest.fail("print mode attempted execution"),
    )

    exit_code = probe_cli.main([
        "--print-contract",
        *_cli_source_args(prepared),
    ])

    assert exit_code == 0
    assert calls == []
    assert _path_snapshot(final_path) == final_before
    assert _path_snapshot(partial_path) == partial_before
    summary = json.loads(capsys.readouterr().out)
    assert summary["schema_version"] == (
        "decomposition-semantic-repair-preparation-v1"
    )
    assert summary["repair_contract"] == prepared.contract.to_record()
    assert summary["repair_contract_sha256"] == prepared.contract.sha256
    assert summary["feedback_map"] == semantic_repair_feedback_map_record()
    assert summary["decompose_case_count"] == 33
    assert summary["problem_count"] == 5


def test_cli_authorization_mismatch_precedes_key_output_and_http(
    monkeypatch, prepared
):
    final_path = ROOT / E26_OUTPUT
    partial_path = ROOT / f"{E26_OUTPUT}.partial"
    final_before = _path_snapshot(final_path)
    partial_before = _path_snapshot(partial_path)
    calls = _install_responses(monkeypatch, [])
    monkeypatch.setattr(
        probe_cli, "prepare_semantic_repair_probe", lambda **kwargs: prepared
    )
    monkeypatch.setattr(
        probe_cli,
        "_secret",
        lambda *args: pytest.fail("mismatch attempted secret lookup"),
    )
    monkeypatch.setattr(
        probe_cli,
        "run_semantic_repair_probe",
        lambda **kwargs: pytest.fail("mismatch attempted execution"),
    )

    with pytest.raises(SystemExit) as error:
        probe_cli.main([
            "--execute-hosted",
            *_cli_source_args(prepared),
            *_cli_authorization_args(prepared, contract_sha256="f" * 64),
        ])

    assert error.value.code == 2
    assert calls == []
    assert _path_snapshot(final_path) == final_before
    assert _path_snapshot(partial_path) == partial_before
