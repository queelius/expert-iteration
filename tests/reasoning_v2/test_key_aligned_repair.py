import json
import threading
from collections import Counter
from copy import copy, deepcopy
from pathlib import Path

import pytest

import experiments.probe_reasoning_v2_key_aligned_repair as probe_cli
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
from verifier_anchored_rl.reasoning_v2.key_aligned_repair import (
    KEY_ALIGNED_REPAIR_CASE_VERSION,
    KEY_ALIGNED_REPAIR_FEEDBACK_CONSTRAINTS,
    KEY_ALIGNED_REPAIR_FEEDBACK_MAP_VERSION,
    KEY_ALIGNED_REPAIR_FEEDBACK_SCHEMA_VERSION,
    KEY_ALIGNED_REPAIR_FEEDBACK_SUFFIX,
    KEY_ALIGNED_REPAIR_HEADER_VERSION,
    KEY_ALIGNED_REPAIR_SUMMARY_VERSION,
    KeyAlignedRepairProbeAuthorization,
    KeyAlignedRepairProbeValidationError,
    build_key_aligned_repair_feedback,
    build_key_aligned_repair_request,
    key_aligned_repair_feedback_map_record,
    key_aligned_repair_feedback_map_sha256,
    prepare_key_aligned_repair_probe,
    read_key_aligned_repair_probe_artifact,
    run_key_aligned_repair_probe,
)
from verifier_anchored_rl.reasoning_v2.microprobe import file_sha256
from verifier_anchored_rl.reasoning_v2.semantic_repair import (
    SemanticRepairProbeAuthorization,
    prepare_semantic_repair_probe,
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
E26_PATH = ROOT / (
    "experiments/runs/2026-07-13-E26-v2-decompose-repair33.jsonl"
)
E26_OUTPUT = "experiments/runs/2026-07-13-E26-v2-decompose-repair33.jsonl"
E27_OUTPUT = (
    "experiments/runs/2026-07-13-E27-v2-decompose-repair-key33.jsonl"
)

SOURCE_ENVIRONMENT_SHA256 = (
    "9316c93bc2cdf9b2ce507a989882af69669f8bf886e1f2ad6ce802c77dfdcfa7"
)
REQUEST_ENVIRONMENT_SHA256 = (
    "664deedf1caa44efff16daac1888c2e3966aa37e9a2631a62036643a9e681c71"
)
SCORING_ENVIRONMENT_SHA256 = (
    "3396ce7464c8bf4a8cd60c5aa26172c02dc4ed4589ca7eb9e2bcfd906119ee54"
)
CORRECTED_PROFILE_SHA256 = (
    "42ee4bf1f481dd92bebac990b8fc0c95d280a4a5c1054b0b9dc4d5070f6581c8"
)
E26_SHA256 = "7290e10733c1195edcea42684881afa4681f4c03f08e7aa7cf317a386623b1bd"
E26_CONTRACT_SHA256 = (
    "69643e2c01b103b6c1bda3ec296a1519806bcdce92286981bf3cd812deb72110"
)
E26_CASE_MANIFEST_SHA256 = (
    "4377ceda24ff9731c5d28f3b4ff3678b513f7f016c1b7471d158356e0df982c1"
)
E26_RESULT_MANIFEST_SHA256 = (
    "a82e87becbfd0e00c125784702d2332f90dcee4a743ded4c361e6bad54849536"
)
E26_SUMMARY_SHA256 = (
    "b0c8dbe1f6d3ed9d6ab43f2b66f292ce6fb0aacf0d01e21a4d3f40c88b7e4d6d"
)
E26_REQUEST_MANIFEST_SHA256 = (
    "0cc914d1c3ec1c685b419a76ce3af603296e10b2f07da742d08c3f3096795b30"
)
E26_GENERATOR_SHA256 = (
    "5fc23203d49417008fb63f5356f2a8e86cc249e77cbc3fac5b6a262858af8e58"
)
E26_GENERATOR_TRANSPORT_SHA256 = (
    "b0e2806c348fcde14bf2bbc3c48656eb7547f3b7142afc93f99ba52c40896f2a"
)
E26_GENERATION_SETTINGS_SHA256 = (
    "9059e1a6bbfb1422344ea9a6c9b9c5b85879f9fc0fe56de6ed2a4e1f846cd463"
)
E27_BOUND_E26_REQUEST_MANIFEST_SHA256 = (
    "0cc914d1c3ec1c685b419a76ce3af603296e10b2f07da742d08c3f3096795b30"
)

EXPECTED_CONSTRAINTS = {
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
EXPECTED_SUFFIX = (
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


def _sealed_e26_authorization():
    source_probe = prepare_semantic_repair_probe(
        source_run_path=RUN_PATH,
        source_diagnostic_path=DIAGNOSTIC_PATH,
        e24_artifact_path=E24_PATH,
        e25_artifact_path=E25_PATH,
        problem_source=_source(),
        planned_output_path=E26_OUTPUT,
    )
    contract = source_probe.contract
    values = {"preregistration_id": "E26-v2-decompose-repair33"}
    for name in SemanticRepairProbeAuthorization.__dataclass_fields__:
        if not name.startswith("expected_"):
            continue
        contract_name = name.removeprefix("expected_")
        if contract_name == "contract_sha256":
            value = contract.sha256
        elif contract_name == "output_path":
            value = contract.planned_output_path
        else:
            value = getattr(contract, contract_name)
        values[name] = value
    return SemanticRepairProbeAuthorization(**values)


@pytest.fixture(scope="module")
def e26_authorization():
    return _sealed_e26_authorization()


@pytest.fixture(scope="module")
def prepared(e26_authorization):
    return prepare_key_aligned_repair_probe(
        source_run_path=RUN_PATH,
        source_diagnostic_path=DIAGNOSTIC_PATH,
        e24_artifact_path=E24_PATH,
        e25_artifact_path=E25_PATH,
        e26_artifact_path=E26_PATH,
        e26_authorization=e26_authorization,
        expected_e26_artifact_sha256=E26_SHA256,
        problem_source=_source(),
        planned_output_path=E27_OUTPUT,
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


def _valid_content(case):
    return json.dumps(
        {"domains": VALID_DOMAINS[case.problem_id]},
        separators=(",", ":"),
    )


def _artifact_records(path):
    return [json.loads(line) for line in path.read_text("ascii").splitlines()]


def _write_records(path, records):
    path.write_bytes(
        b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    )


def _path_snapshot(path):
    return None if not path.exists() else path.read_bytes()


def _authorization(prepared, **overrides):
    contract = prepared.contract
    values = {
        "preregistration_id": "E27-test",
        "expected_source_environment_contract_sha256": (
            contract.source_environment_contract_sha256
        ),
        "expected_request_environment_contract_sha256": (
            contract.request_environment_contract_sha256
        ),
        "expected_scoring_environment_contract_sha256": (
            contract.scoring_environment_contract_sha256
        ),
        "expected_corrected_profile_sha256": (
            contract.corrected_profile_sha256
        ),
        "expected_e26_artifact_sha256": contract.e26_artifact_sha256,
        "expected_e26_repair_contract_sha256": (
            contract.e26_repair_contract_sha256
        ),
        "expected_e26_case_manifest_sha256": (
            contract.e26_case_manifest_sha256
        ),
        "expected_e26_result_manifest_sha256": (
            contract.e26_result_manifest_sha256
        ),
        "expected_e26_summary_sha256": contract.e26_summary_sha256,
        "expected_feedback_map_sha256": contract.feedback_map_sha256,
        "expected_case_manifest_sha256": contract.case_manifest_sha256,
        "expected_base_request_manifest_sha256": (
            contract.base_request_manifest_sha256
        ),
        "expected_e26_repair_request_manifest_sha256": (
            contract.e26_repair_request_manifest_sha256
        ),
        "expected_key_aligned_request_manifest_sha256": (
            contract.key_aligned_request_manifest_sha256
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
    return KeyAlignedRepairProbeAuthorization(**values)


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
        "--e26-artifact",
        str(E26_PATH),
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
    authorization = _authorization(prepared)
    args = ["--preregistration-id", authorization.preregistration_id]
    for field_name in authorization.__dataclass_fields__:
        if not (
            field_name.startswith("expected_")
            and field_name.endswith("_sha256")
        ):
            continue
        value = getattr(authorization, field_name)
        if field_name == "expected_contract_sha256" and contract_sha256:
            value = contract_sha256
        flag = "--" + field_name.removeprefix("expected_").replace("_", "-")
        args.extend([flag, value])
    args.extend(["--generator-api-key-env", "MISSING_HF_TOKEN"])
    return args


def test_feedback_v2_is_exactly_key_aligned_with_e26():
    assert KEY_ALIGNED_REPAIR_FEEDBACK_SCHEMA_VERSION == (
        "decomposition-semantic-feedback-v2"
    )
    assert KEY_ALIGNED_REPAIR_FEEDBACK_CONSTRAINTS == EXPECTED_CONSTRAINTS
    assert KEY_ALIGNED_REPAIR_FEEDBACK_SUFFIX == EXPECTED_SUFFIX
    assert key_aligned_repair_feedback_map_record() == {
        "schema_version": KEY_ALIGNED_REPAIR_FEEDBACK_MAP_VERSION,
        "constraints": EXPECTED_CONSTRAINTS,
    }
    assert key_aligned_repair_feedback_map_sha256() == canonical_json_sha256(
        key_aligned_repair_feedback_map_record()
    )


def test_prepare_binds_exact_sealed_e26_source_and_33_cases(prepared):
    prepared.validate_trusted()
    contract = prepared.contract.to_record()
    assert contract["source_environment_contract_sha256"] == (
        SOURCE_ENVIRONMENT_SHA256
    )
    assert contract["request_environment_contract_sha256"] == (
        REQUEST_ENVIRONMENT_SHA256
    )
    assert contract["scoring_environment_contract_sha256"] == (
        SCORING_ENVIRONMENT_SHA256
    )
    assert contract["corrected_profile_sha256"] == CORRECTED_PROFILE_SHA256
    assert contract["e26_artifact_sha256"] == E26_SHA256
    assert contract["e26_repair_contract_sha256"] == E26_CONTRACT_SHA256
    assert contract["e26_case_manifest_sha256"] == E26_CASE_MANIFEST_SHA256
    assert contract["e26_result_manifest_sha256"] == (
        E26_RESULT_MANIFEST_SHA256
    )
    assert contract["e26_summary_sha256"] == E26_SUMMARY_SHA256
    assert contract["e26_repair_request_manifest_sha256"] == (
        E27_BOUND_E26_REQUEST_MANIFEST_SHA256
    )
    assert prepared.source_probe.contract.repair_request_manifest_sha256 == (
        E26_REQUEST_MANIFEST_SHA256
    )
    assert contract["e26_repair_request_manifest_sha256"] == (
        prepared.source_probe.contract.repair_request_manifest_sha256
    )
    assert contract["generator_sha256"] == E26_GENERATOR_SHA256
    assert contract["generator_transport_sha256"] == (
        E26_GENERATOR_TRANSPORT_SHA256
    )
    assert contract["generation_settings_sha256"] == (
        E26_GENERATION_SETTINGS_SHA256
    )
    assert contract["generator_sha256"] == (
        prepared.source_probe.contract.generator_sha256
    )
    assert contract["generator_transport_sha256"] == (
        prepared.source_probe.contract.generator_transport_sha256
    )
    assert contract["generation_settings_sha256"] == (
        prepared.source_probe.contract.generation_settings_sha256
    )
    assert contract["source_decision_count"] == 66
    assert contract["case_count"] == 33
    assert contract["problem_count"] == 5
    assert len(prepared.cases) == 33
    assert [case.index for case in prepared.cases] == list(range(33))
    assert [
        case.manifest_record()["e26_case_index"] for case in prepared.cases
    ] == list(range(33))
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
    assert Counter(
        case.source_case.corrected_reason_code for case in prepared.cases
    ) == {
        "unsupported_domain": 15,
        "decomposition_domain_overlap": 15,
        "decomposition_domain_not_exhaustive": 3,
    }
    assert contract["case_manifest_sha256"] == canonical_json_sha256([
        case.manifest_record() for case in prepared.cases
    ])


def test_first_attempt_body_delta_is_only_key_and_feedback_schema(prepared):
    for case in prepared.cases:
        request, base_body, aligned_body = build_key_aligned_repair_request(
            prepared, case
        )
        source = case.source_case
        e26_body = source.repair_request_body
        old_feedback = source.feedback
        new_feedback = build_key_aligned_repair_feedback(
            source.failed_domains, source.corrected_reason_code
        )
        assert new_feedback == case.feedback
        assert set(new_feedback) == {
            "schema_version",
            "domains",
            "reason_code",
            "repair_constraint",
        }
        assert "failed_domains" not in new_feedback
        assert new_feedback["domains"] == old_feedback["failed_domains"]
        old_rest = dict(old_feedback)
        new_rest = dict(new_feedback)
        old_domains = old_rest.pop("failed_domains")
        new_domains = new_rest.pop("domains")
        assert old_domains == new_domains
        assert old_rest.pop("schema_version") == (
            "decomposition-semantic-feedback-v1"
        )
        assert new_rest.pop("schema_version") == (
            KEY_ALIGNED_REPAIR_FEEDBACK_SCHEMA_VERSION
        )
        assert old_rest == new_rest

        assert base_body == source.base_request_body
        assert canonical_json_sha256(base_body) == (
            case.base_request_sha256
        )
        assert canonical_json_sha256(e26_body) == (
            case.e26_repair_request_sha256
        )
        assert canonical_json_sha256(aligned_body) == (
            case.key_aligned_request_sha256
        )
        assert set(e26_body) == set(aligned_body)
        for key in set(e26_body) - {"messages"}:
            assert e26_body[key] == aligned_body[key]
        assert e26_body["messages"][0] == aligned_body["messages"][0]
        old_json = canonical_json_bytes(old_feedback).decode("ascii")
        new_json = canonical_json_bytes(new_feedback).decode("ascii")
        old_user = e26_body["messages"][1]["content"]
        expected_user = old_user.replace(old_json, new_json)
        assert old_user.count(old_json) == 1
        assert expected_user.count(new_json) == 1
        assert aligned_body["messages"][1] == {
            "role": "user",
            "content": expected_user,
        }
        assert request.user == expected_user
        assert expected_user.endswith(new_json + "\n" + EXPECTED_SUFFIX)


def test_prepare_reads_no_key_and_makes_no_http(
    monkeypatch, e26_authorization
):
    calls = _install_responses(monkeypatch, [])
    monkeypatch.delenv("HF_TOKEN", raising=False)

    result = prepare_key_aligned_repair_probe(
        source_run_path=RUN_PATH,
        source_diagnostic_path=DIAGNOSTIC_PATH,
        e24_artifact_path=E24_PATH,
        e25_artifact_path=E25_PATH,
        e26_artifact_path=E26_PATH,
        e26_authorization=e26_authorization,
        expected_e26_artifact_sha256=E26_SHA256,
        problem_source=_source(),
        planned_output_path=E27_OUTPUT,
    )

    result.validate_trusted()
    assert calls == []


def test_prepare_rejects_unsealed_e26_digest_authorization_and_header(
    tmp_path, e26_authorization
):
    common = {
        "source_run_path": RUN_PATH,
        "source_diagnostic_path": DIAGNOSTIC_PATH,
        "e24_artifact_path": E24_PATH,
        "e25_artifact_path": E25_PATH,
        "problem_source": _source(),
        "planned_output_path": tmp_path / "result.jsonl",
    }
    with pytest.raises(
        KeyAlignedRepairProbeValidationError,
        match="E26 artifact digest differs",
    ):
        prepare_key_aligned_repair_probe(
            **common,
            e26_artifact_path=E26_PATH,
            e26_authorization=e26_authorization,
            expected_e26_artifact_sha256="0" * 64,
        )

    bad_authorization = copy(e26_authorization)
    object.__setattr__(
        bad_authorization, "expected_contract_sha256", "f" * 64
    )
    with pytest.raises(
        KeyAlignedRepairProbeValidationError,
        match="frozen authorization",
    ):
        prepare_key_aligned_repair_probe(
            **common,
            e26_artifact_path=E26_PATH,
            e26_authorization=bad_authorization,
            expected_e26_artifact_sha256=E26_SHA256,
        )

    records = _artifact_records(E26_PATH)
    records[0]["preregistration_id"] = "E26-unsealed"
    tampered_path = tmp_path / "tampered-e26.jsonl"
    _write_records(tampered_path, records)
    with pytest.raises(
        KeyAlignedRepairProbeValidationError,
        match="preregistration id differs",
    ):
        prepare_key_aligned_repair_probe(
            **common,
            e26_artifact_path=tampered_path,
            e26_authorization=e26_authorization,
            expected_e26_artifact_sha256=file_sha256(tampered_path),
        )


def test_every_authorization_binding_precedes_output_and_http(
    monkeypatch, prepared
):
    final_path = ROOT / E27_OUTPUT
    partial_path = ROOT / f"{E27_OUTPUT}.partial"
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
            run_key_aligned_repair_probe(
                output_path=E27_OUTPUT,
                prepared=prepared,
                generator_client=_generator_client(prepared),
                authorization=bad,
            )
        checked.append(field_name)

    assert len(checked) == 22
    assert calls == []
    assert _path_snapshot(final_path) == final_before
    assert _path_snapshot(partial_path) == partial_before


@pytest.fixture(scope="module")
def executed_probe(tmp_path_factory, e26_authorization):
    directory = tmp_path_factory.mktemp("key-aligned-repair")
    output_path = directory / "repair.jsonl"
    local_prepared = prepare_key_aligned_repair_probe(
        source_run_path=RUN_PATH,
        source_diagnostic_path=DIAGNOSTIC_PATH,
        e24_artifact_path=E24_PATH,
        e25_artifact_path=E25_PATH,
        e26_artifact_path=E26_PATH,
        e26_authorization=e26_authorization,
        expected_e26_artifact_sha256=E26_SHA256,
        problem_source=_source(),
        planned_output_path=output_path,
    )
    responses = []
    for case in local_prepared.cases:
        if case.index == 0:
            responses.append(_Response(json.dumps(
                {"failed_domains": list(case.source_case.failed_domains)},
                separators=(",", ":"),
            )))
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
        result = run_key_aligned_repair_probe(
            output_path=output_path,
            prepared=local_prepared,
            generator_client=client,
            authorization=_authorization(local_prepared),
        )
    finally:
        monkeypatch.undo()
    return local_prepared, result, output_path, calls, client


def test_stubbed_probe_has_33_calls_strict_wire_stages_and_all_gates(
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
        initial = calls[call_offset]["body"]
        assert initial == case.key_aligned_request_body
        assert initial["seed"] == case.effective_seed
        call_offset += 1
        if case.index == 0:
            retry = calls[call_offset]["body"]
            assert retry["seed"] == case.effective_seed + 1
            assert len(retry["messages"]) == len(initial["messages"]) + 2
            assert retry != case.source_case.repair_request_body
            call_offset += 1
    assert call_offset == 34

    records = _artifact_records(output_path)
    schemas = Counter(record["schema_version"] for record in records)
    assert schemas == {
        KEY_ALIGNED_REPAIR_HEADER_VERSION: 1,
        DIAGNOSTIC_ATTEMPT_VERSION: 34,
        DIAGNOSTIC_CALL_VERSION: 33,
        KEY_ALIGNED_REPAIR_CASE_VERSION: 33,
        KEY_ALIGNED_REPAIR_SUMMARY_VERSION: 1,
    }
    attempts = [
        record
        for record in records
        if record["schema_version"] == DIAGNOSTIC_ATTEMPT_VERSION
    ]
    cases = [
        record
        for record in records
        if record["schema_version"] == KEY_ALIGNED_REPAIR_CASE_VERSION
    ]
    summary = records[-1]
    authorization = _authorization(prepared)
    assert read_key_aligned_repair_probe_artifact(
        output_path,
        prepared=prepared,
        authorization=authorization,
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
        assert record["exact_input_copy"] is False
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
            expected_reason = "child_not_atomic" if case.index == 2 else None
            assert record["corrected_application"]["reason_code"] == (
                expected_reason
            )

    first_call_id = cases[0]["repair_call_record_id"].split(":")[1]
    first_attempts = [
        attempt for attempt in attempts if attempt["call_id"] == first_call_id
    ]
    assert [attempt["outcome"] for attempt in first_attempts] == [
        "schema_error",
        "accepted",
    ]
    assert [attempt["disposition"] for attempt in first_attempts] == [
        "parse_retry",
        "return",
    ]
    assert [attempt["effective_seed"] for attempt in first_attempts] == [0, 1]
    assert set(json.loads(first_attempts[0]["raw_content"])) == {
        "failed_domains"
    }

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
    assert summary["exact_input_copy_count"] == 0
    assert summary["symbolic_application_counts"] == {
        "algebra_rejected": 1,
        "applied": 31,
        "not_run": 1,
    }
    assert summary["applied_problem_count"] == 5
    assert set(summary["applied_problem_ids"]) == set(VALID_DOMAINS)
    assert summary["corrected_result_reason_counts"] == {
        "applied": 31,
        "child_not_atomic": 1,
        "not_run": 1,
    }
    assert summary["raw_attempt_outcome_counts"] == {
        "accepted": 33,
        "http_status_error": 0,
        "parse_error": 0,
        "provider_response_error": 0,
        "schema_error": 1,
        "transport_error": 0,
    }
    assert summary["reported_usage"] == {
        "attempt_count": 34,
        "completion_tokens": 68,
        "prompt_tokens": 102,
        "total_tokens": 170,
    }
    assert summary["reported_cost"]["total_cost_usd"] == "0.0000510"
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


def test_strict_reader_rejects_wire_feedback_result_and_summary_tampering(
    tmp_path, executed_probe
):
    prepared, _, output_path, _, _ = executed_probe
    authorization = _authorization(prepared)
    with pytest.raises(TypeError, match="prepared"):
        read_key_aligned_repair_probe_artifact(
            output_path, authorization=authorization
        )
    with pytest.raises(TypeError, match="authorization"):
        read_key_aligned_repair_probe_artifact(
            output_path, prepared=prepared
        )
    original = _artifact_records(output_path)

    def preregistration_mutation(records):
        records[0]["preregistration_id"] = "E27-not-authorized"

    def initial_request_mutation(records):
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

    def feedback_key_mutation(records):
        case = next(
            record
            for record in records
            if record["schema_version"] == KEY_ALIGNED_REPAIR_CASE_VERSION
        )
        case["feedback"]["failed_domains"] = case["feedback"].pop("domains")

    def manifest_mutation(records):
        case = next(
            record
            for record in records
            if record["schema_version"] == KEY_ALIGNED_REPAIR_CASE_VERSION
        )
        case["key_aligned_request_sha256"] = "f" * 64

    def score_mutation(records):
        case = next(
            record
            for record in records
            if record["schema_version"] == KEY_ALIGNED_REPAIR_CASE_VERSION
            and record["corrected_application"] is not None
        )
        case["corrected_application"]["coverage"]["context_digest"] = (
            "f" * 64
        )

    def exact_copy_mutation(records):
        case = next(
            record
            for record in records
            if record["schema_version"] == KEY_ALIGNED_REPAIR_CASE_VERSION
        )
        case["exact_input_copy"] = True

    def summary_mutation(records):
        records[-1]["typed_decoded_count"] -= 1

    def unsupported_schema_mutation(records):
        records.insert(-1, {"schema_version": "controller-transition-v2"})

    for name, mutate in (
        ("preregistration", preregistration_mutation),
        ("initial-request", initial_request_mutation),
        ("raw-projection", raw_projection_mutation),
        ("feedback-key", feedback_key_mutation),
        ("manifest", manifest_mutation),
        ("score", score_mutation),
        ("exact-copy", exact_copy_mutation),
        ("summary", summary_mutation),
        ("unsupported-schema", unsupported_schema_mutation),
    ):
        records = deepcopy(original)
        mutate(records)
        path = tmp_path / f"{name}.jsonl"
        _write_records(path, records)
        with pytest.raises(KeyAlignedRepairProbeValidationError):
            read_key_aligned_repair_probe_artifact(
                path,
                prepared=prepared,
                authorization=authorization,
            )


@pytest.fixture(scope="module")
def guarded_probe(tmp_path_factory, e26_authorization):
    directory = tmp_path_factory.mktemp("key-aligned-repair-guard")
    output_path = directory / "guard.jsonl"
    local_prepared = prepare_key_aligned_repair_probe(
        source_run_path=RUN_PATH,
        source_diagnostic_path=DIAGNOSTIC_PATH,
        e24_artifact_path=E24_PATH,
        e25_artifact_path=E25_PATH,
        e26_artifact_path=E26_PATH,
        e26_authorization=e26_authorization,
        expected_e26_artifact_sha256=E26_SHA256,
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
        run_key_aligned_repair_probe(
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
        run_key_aligned_repair_probe(
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
        run_key_aligned_repair_probe(
            output_path=output_path,
            prepared=prepared,
            generator_client=_generator_client(prepared),
            authorization=_authorization(prepared),
        )
    assert len(interrupted_calls) == 1
    assert not output_path.exists()
    assert partial_path.exists()
    with pytest.raises(KeyAlignedRepairProbeValidationError):
        read_key_aligned_repair_probe_artifact(
            partial_path,
            prepared=prepared,
            authorization=_authorization(prepared),
        )


def test_cli_help_exits_before_preparation(monkeypatch):
    monkeypatch.setattr(
        probe_cli,
        "prepare_key_aligned_repair_probe",
        lambda **kwargs: pytest.fail("help attempted preparation"),
    )

    with pytest.raises(SystemExit) as error:
        probe_cli.main(["--help"])

    assert error.value.code == 0


def test_cli_print_contract_reads_no_key_and_makes_no_http_or_write(
    monkeypatch, capsys, prepared, e26_authorization
):
    final_path = ROOT / E27_OUTPUT
    partial_path = ROOT / f"{E27_OUTPUT}.partial"
    final_before = _path_snapshot(final_path)
    partial_before = _path_snapshot(partial_path)
    calls = _install_responses(monkeypatch, [])
    preparation_kwargs = {}

    def fake_prepare(**kwargs):
        preparation_kwargs.update(kwargs)
        return prepared

    monkeypatch.setattr(
        probe_cli, "prepare_key_aligned_repair_probe", fake_prepare
    )
    monkeypatch.setattr(
        probe_cli,
        "_secret",
        lambda *args: pytest.fail("print mode attempted secret lookup"),
    )
    monkeypatch.setattr(
        probe_cli,
        "run_key_aligned_repair_probe",
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
    assert preparation_kwargs["e26_authorization"] == e26_authorization
    assert preparation_kwargs["expected_e26_artifact_sha256"] == E26_SHA256
    record = json.loads(capsys.readouterr().out)
    assert record["schema_version"] == (
        "decomposition-key-aligned-repair-preparation-v1"
    )
    assert record["repair_contract"] == prepared.contract.to_record()
    assert record["repair_contract_sha256"] == prepared.contract.sha256
    assert record["source_e26_repair_contract"] == (
        prepared.source_probe.contract.to_record()
    )
    assert record["source_e26_summary"] == prepared.source_summary
    assert record["feedback_map"] == key_aligned_repair_feedback_map_record()
    assert record["decompose_case_count"] == 33
    assert record["problem_count"] == 5


def test_cli_authorization_mismatch_precedes_key_output_and_http(
    monkeypatch, prepared
):
    final_path = ROOT / E27_OUTPUT
    partial_path = ROOT / f"{E27_OUTPUT}.partial"
    final_before = _path_snapshot(final_path)
    partial_before = _path_snapshot(partial_path)
    calls = _install_responses(monkeypatch, [])
    monkeypatch.setattr(
        probe_cli,
        "prepare_key_aligned_repair_probe",
        lambda **kwargs: prepared,
    )
    monkeypatch.setattr(
        probe_cli,
        "_secret",
        lambda *args: pytest.fail("mismatch attempted secret lookup"),
    )
    monkeypatch.setattr(
        probe_cli,
        "run_key_aligned_repair_probe",
        lambda **kwargs: pytest.fail("mismatch attempted execution"),
    )

    with pytest.raises(SystemExit) as error:
        probe_cli.main([
            "--execute-hosted",
            *_cli_source_args(prepared),
            *_cli_authorization_args(
                prepared, contract_sha256="f" * 64
            ),
        ])

    assert error.value.code == 2
    assert calls == []
    assert _path_snapshot(final_path) == final_before
    assert _path_snapshot(partial_path) == partial_before
