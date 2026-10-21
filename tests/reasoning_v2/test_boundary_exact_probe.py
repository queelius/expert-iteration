import json
from copy import deepcopy

import pytest

import experiments.probe_reasoning_v2_boundaries_exact as exact_cli
import tests.reasoning_v2.test_boundary_probe as e28_test
import verifier_anchored_rl.reasoning_v2.boundary_exact_probe as exact_module
from verifier_anchored_rl.llm_client import (
    OPENAI_JSON_RETRY_INSTRUCTION,
    SCHEMA_VALIDATION_STRICT_V1,
)
from verifier_anchored_rl.reasoning_v2.boundary_exact_probe import (
    EXACT_BOUNDARY_PROBE_CASE_VERSION,
    EXACT_BOUNDARY_PROBE_CONTRACT_VERSION,
    EXACT_BOUNDARY_PROBE_SUMMARY_VERSION,
    ExactBoundaryProbeAuthorization,
    ExactBoundaryProbeValidationError,
    exact_boundary_retry_policy_record,
    exact_boundary_retry_policy_sha256,
    exact_boundary_schema_validation_record,
    exact_boundary_schema_validation_sha256,
    prepare_exact_boundary_probe,
    read_exact_boundary_probe_artifact,
    run_exact_boundary_probe,
)
from verifier_anchored_rl.reasoning_v2.boundary_probe import (
    BoundaryDecompositionProbeAuthorization,
)
from verifier_anchored_rl.reasoning_v2.contracts import canonical_json_sha256
from verifier_anchored_rl.reasoning_v2.diagnostics import (
    DIAGNOSTIC_ATTEMPT_VERSION,
)


E29_OUTPUT = (
    "experiments/runs/2026-07-13-E29-v2-boundary-exact-schema33.jsonl"
)
RAW_AUDIT_SHA256 = (
    "3eb3ddff5433a49e2c80f042407335388ad30a556f1dd19fce2e1fbd7bb0f9d7"
)
RAW_CONTROL_MANIFEST_SHA256 = (
    "da475c66b2365958826e503315603ac053b76e29d4a2ca220db5d327f7623704"
)
E29_CONTRACT_SHA256 = (
    "c6cbe4410ea63e6b7b0329ac7c681d721faac16a9472064e1f8bc05cc4e01700"
)
E29_CASE_MANIFEST_SHA256 = (
    "9921b96955a3953c379403bf5bfb3986f0f30a18b53a0d93191e58086c4ec863"
)
SCHEMA_VALIDATION_SHA256 = (
    "e1c539bb0a91b340379b2d7c46bdc12fa86040f88de381f8f138ef73704d05fe"
)
RETRY_POLICY_SHA256 = (
    "07e11c7b0e40bd0e9ce4bc6188a03b6289556dc31bffeb5bb1cd1f6588c325fd"
)


@pytest.fixture(scope="module")
def source_chain():
    return e28_test.source_chain.__wrapped__()


@pytest.fixture(scope="module")
def prepared_e28(source_chain):
    return e28_test.prepared.__wrapped__(source_chain)


@pytest.fixture(scope="module")
def e28_authorization(prepared_e28):
    return e28_test._authorization_from_contract(
        BoundaryDecompositionProbeAuthorization,
        "E28-v2-boundary-decompose33",
        prepared_e28.contract,
    )


def _prepare(source_chain, e28_authorization, output_path):
    return prepare_exact_boundary_probe(
        source_run_path=e28_test.RUN_PATH,
        source_diagnostic_path=e28_test.DIAGNOSTIC_PATH,
        e24_artifact_path=e28_test.E24_PATH,
        e25_artifact_path=e28_test.E25_PATH,
        e26_artifact_path=e28_test.E26_PATH,
        e27_artifact_path=e28_test.E27_PATH,
        e28_artifact_path=e28_test.E28_INVALID_PATH,
        e26_authorization=source_chain[0],
        e27_authorization=source_chain[1],
        e28_authorization=e28_authorization,
        expected_e26_artifact_sha256=e28_test.E26_SHA256,
        expected_e27_artifact_sha256=e28_test.E27_SHA256,
        expected_e28_artifact_sha256=e28_test.E28_INVALID_SHA256,
        problem_source=e28_test._source(),
        planned_output_path=output_path,
    )


@pytest.fixture(scope="module")
def prepared(source_chain, e28_authorization):
    return _prepare(source_chain, e28_authorization, E29_OUTPUT)


def _authorization(prepared):
    return e28_test._authorization_from_contract(
        ExactBoundaryProbeAuthorization,
        "E29-v2-boundary-exact-schema33",
        prepared.contract,
    )


def _e28_attempts():
    return [
        record
        for record in e28_test._artifact_records(e28_test.E28_INVALID_PATH)
        if record.get("schema_version") == DIAGNOSTIC_ATTEMPT_VERSION
    ]


def _valid_content(case):
    return json.dumps(
        {"boundaries": e28_test.ORACLE_BOUNDARIES[case.problem_id]},
        separators=(",", ":"),
    )


def _case_records(records):
    return [
        record
        for record in records
        if record.get("schema_version") == EXACT_BOUNDARY_PROBE_CASE_VERSION
    ]


def test_prepare_binds_strict_mode_raw_control_and_unchanged_e28_requests(
    prepared,
):
    prepared.validate_trusted()
    contract = prepared.contract
    record = contract.to_record()

    assert record["schema_version"] == EXACT_BOUNDARY_PROBE_CONTRACT_VERSION
    assert contract.invalidated_e28_artifact_sha256 == (
        e28_test.E28_INVALID_SHA256
    )
    assert contract.invalidated_e28_probe_contract_sha256 == (
        "6dbfd0e04b0c702ad1cc5342868d932a97cb56c904bb98bc7fb1dc45e993ca06"
    )
    assert contract.invalidated_e28_raw_audit_sha256 == RAW_AUDIT_SHA256
    assert contract.invalidated_e28_raw_control_manifest_sha256 == (
        RAW_CONTROL_MANIFEST_SHA256
    )
    assert contract.sha256 == E29_CONTRACT_SHA256
    assert contract.case_manifest_sha256 == E29_CASE_MANIFEST_SHA256
    assert contract.boundary_request_manifest_sha256 == (
        contract.invalidated_e28_boundary_request_manifest_sha256
    )
    assert contract.boundary_request_manifest_sha256 == (
        prepared.source_e28_probe.contract.boundary_request_manifest_sha256
    )
    assert contract.schema_validation_mode == SCHEMA_VALIDATION_STRICT_V1
    assert contract.schema_validation_contract_sha256 == (
        SCHEMA_VALIDATION_SHA256
    )
    assert contract.retry_policy_contract_sha256 == (
        RETRY_POLICY_SHA256
    )
    assert exact_boundary_schema_validation_sha256() == SCHEMA_VALIDATION_SHA256
    assert exact_boundary_retry_policy_sha256() == RETRY_POLICY_SHA256
    assert record["schema_validation"] == (
        exact_boundary_schema_validation_record()
    )
    assert record["retry_policy"] == exact_boundary_retry_policy_record()
    assert contract.planned_output_path == E29_OUTPUT
    assert [case.effective_seed for case in prepared.cases] == [
        case.source_case.effective_seed for case in prepared.cases
    ]
    assert all(
        case.invalidated_e28_raw_control.initial_request_sha256
        == case.source_case.boundary_request_sha256
        for case in prepared.cases
    )
    assert prepared.invalidated_e28_raw_audit.to_record() == {
        "schema_version": "e28-no-coercion-raw-audit-v1",
        "artifact_sha256": e28_test.E28_INVALID_SHA256,
        "contract_sha256": (
            "6dbfd0e04b0c702ad1cc5342868d932a97cb56c904bb98bc7fb1dc45e993ca06"
        ),
        "raw_attempt_count": 33,
        "all_number_array_count": 17,
        "mixed_array_count": 8,
        "all_string_array_count": 8,
        "boundary_item_count": 71,
        "numeric_item_count": 42,
        "exact_schema_accepted_count": 8,
        "numeric_schema_error_count": 25,
        "accepted_problem_ids": ["cw1-52e3b6b3ec25"],
        "first_attempt_typed_count": 8,
        "first_attempt_applied_count": 8,
        "first_attempt_applied_problem_count": 1,
        "required_retry_not_executed_count": 25,
        "legacy_summary_status": "invalidated-not-an-empirical-result-v1",
    }


def test_retry_policy_hashes_literal_instruction_echo_outcomes_budget_and_seed():
    policy = exact_boundary_retry_policy_record()

    assert policy == {
        "schema_version": "boundary-exact-schema-retry-policy-v1",
        "correction_instruction": OPENAI_JSON_RETRY_INSTRUCTION,
        "assistant_echo_rule": (
            "append-current-run-raw-content-when-truthy-before-correction-v1"
        ),
        "retryable_outcomes": [
            "parse_error",
            "provider_response_error",
            "schema_error",
            "transport_error",
        ],
        "parse_retry_budget": 1,
        "seed_formula": "initial_seed + parse_retry_index",
        "semantic_retry": False,
    }
    assert exact_boundary_retry_policy_sha256() == canonical_json_sha256(policy)


def test_prepare_rejects_wrong_e28_digest_before_source_replay(
    monkeypatch, source_chain, e28_authorization, tmp_path
):
    def fail_if_replayed(**kwargs):
        raise AssertionError("source replay must not run before E28 digest check")

    monkeypatch.setattr(
        exact_module, "prepare_boundary_decomposition_probe", fail_if_replayed
    )
    with pytest.raises(
        ExactBoundaryProbeValidationError,
        match="invalidated E28 artifact digest differs",
    ):
        prepare_exact_boundary_probe(
            source_run_path=e28_test.RUN_PATH,
            source_diagnostic_path=e28_test.DIAGNOSTIC_PATH,
            e24_artifact_path=e28_test.E24_PATH,
            e25_artifact_path=e28_test.E25_PATH,
            e26_artifact_path=e28_test.E26_PATH,
            e27_artifact_path=e28_test.E27_PATH,
            e28_artifact_path=e28_test.E28_INVALID_PATH,
            e26_authorization=source_chain[0],
            e27_authorization=source_chain[1],
            e28_authorization=e28_authorization,
            expected_e26_artifact_sha256=e28_test.E26_SHA256,
            expected_e27_artifact_sha256=e28_test.E27_SHA256,
            expected_e28_artifact_sha256="0" * 64,
            problem_source=e28_test._source(),
            planned_output_path=tmp_path / "never-created.jsonl",
        )
    assert not (tmp_path / "never-created.jsonl").exists()


def test_strict_corrected_replication_retries_numeric_raw_and_replays_artifact(
    monkeypatch, source_chain, e28_authorization, tmp_path
):
    output = tmp_path / "e29-exact.jsonl"
    prepared = _prepare(source_chain, e28_authorization, output)
    authorization = _authorization(prepared)
    e28_attempts = _e28_attempts()
    responses = []
    for case, control, attempt in zip(
        prepared.cases,
        prepared.invalidated_e28_raw_controls,
        e28_attempts,
        strict=True,
    ):
        responses.append(e28_test._Response(attempt["raw_content"]))
        if control.exact_wire_outcome == "schema_error":
            responses.append(e28_test._Response(_valid_content(case)))
    calls = e28_test._install_responses(monkeypatch, responses)

    result = run_exact_boundary_probe(
        output_path=output,
        prepared=prepared,
        generator_client=e28_test._generator_client(prepared),
        authorization=authorization,
    )

    assert result.case_count == 33
    assert result.wire_usable_count == 33
    assert len(calls) == 58
    cursor = 0
    for case, control, source_attempt in zip(
        prepared.cases,
        prepared.invalidated_e28_raw_controls,
        e28_attempts,
        strict=True,
    ):
        initial = calls[cursor]["body"]
        cursor += 1
        assert initial == case.boundary_request_body
        assert initial["seed"] == case.effective_seed
        if control.exact_wire_outcome != "schema_error":
            continue
        retry = calls[cursor]["body"]
        cursor += 1
        assert retry["messages"][:-2] == initial["messages"]
        assert retry["messages"][-2] == {
            "role": "assistant",
            "content": source_attempt["raw_content"],
        }
        assert retry["messages"][-1] == {
            "role": "user",
            "content": OPENAI_JSON_RETRY_INSTRUCTION,
        }
        assert retry["seed"] == case.effective_seed + 1

    summary = read_exact_boundary_probe_artifact(
        output, prepared=prepared, authorization=authorization
    )
    assert summary["schema_version"] == EXACT_BOUNDARY_PROBE_SUMMARY_VERSION
    assert summary["raw_attempt_count"] == 58
    assert summary["generator_call_count"] == 33
    assert summary["wire_usable_count"] == 33
    assert summary["typed_decoded_count"] == 33
    assert summary["raw_attempt_outcome_counts"] == {
        "accepted": 33,
        "http_status_error": 0,
        "parse_error": 0,
        "provider_response_error": 0,
        "schema_error": 25,
        "transport_error": 0,
    }
    assert summary["retry_diagnostics"] == {
        "schema_version": "boundary-exact-schema-retry-diagnostics-v1",
        "case_count": 33,
        "first_attempt_accepted_count": 8,
        "first_attempt_schema_error_count": 25,
        "first_attempt_other_outcome_count": 0,
        "schema_retry_case_count": 25,
        "schema_retry_repaired_count": 25,
        "schema_retry_exhausted_count": 0,
        "first_array_all_number_count": 17,
        "first_array_mixed_count": 8,
        "first_array_all_string_count": 8,
        "first_array_unclassified_count": 0,
        "first_boundary_item_count": 71,
        "first_numeric_item_count": 42,
        "e28_first_raw_content_hash_match_count": 33,
    }
    records = e28_test._artifact_records(output)
    for record in _case_records(records):
        assert all(
            type(value) is str for value in record["boundary_payload"]["boundaries"]
        )

    tampered = deepcopy(records)
    retry_index = next(
        index
        for index, record in enumerate(tampered)
        if record.get("schema_version") == DIAGNOSTIC_ATTEMPT_VERSION
        and record.get("parse_retry_index") == 1
    )
    tampered[retry_index]["request_body"]["messages"][-2]["content"] = "changed"
    bad_retry = tmp_path / "bad-retry.jsonl"
    e28_test._write_records(bad_retry, tampered)
    with pytest.raises(ExactBoundaryProbeValidationError):
        read_exact_boundary_probe_artifact(
            bad_retry, prepared=prepared, authorization=authorization
        )

    tampered = deepcopy(records)
    first_case = next(
        record
        for record in tampered
        if record.get("schema_version") == EXACT_BOUNDARY_PROBE_CASE_VERSION
    )
    first_case["boundary_payload"]["boundaries"][0] = 1
    bad_payload = tmp_path / "bad-payload.jsonl"
    e28_test._write_records(bad_payload, tampered)
    with pytest.raises(ExactBoundaryProbeValidationError):
        read_exact_boundary_probe_artifact(
            bad_payload, prepared=prepared, authorization=authorization
        )

    tampered = deepcopy(records)
    tampered[-1]["retry_diagnostics"][
        "e28_first_raw_content_hash_match_count"
    ] = 32
    bad_summary = tmp_path / "bad-summary.jsonl"
    e28_test._write_records(bad_summary, tampered)
    with pytest.raises(ExactBoundaryProbeValidationError):
        read_exact_boundary_probe_artifact(
            bad_summary, prepared=prepared, authorization=authorization
        )


def test_second_numeric_reply_is_terminal_unusable_and_diagnosed(
    monkeypatch, source_chain, e28_authorization, tmp_path
):
    output = tmp_path / "e29-exhausted.jsonl"
    prepared = _prepare(source_chain, e28_authorization, output)
    authorization = _authorization(prepared)
    numeric = json.dumps({"boundaries": [1]}, separators=(",", ":"))
    responses = [e28_test._Response(numeric), e28_test._Response(numeric)]
    responses.extend(
        e28_test._Response(_valid_content(case))
        for case in prepared.cases[1:]
    )
    e28_test._install_responses(monkeypatch, responses)

    result = run_exact_boundary_probe(
        output_path=output,
        prepared=prepared,
        generator_client=e28_test._generator_client(prepared),
        authorization=authorization,
    )
    summary = read_exact_boundary_probe_artifact(
        output, prepared=prepared, authorization=authorization
    )

    assert result.wire_usable_count == 32
    assert summary["wire_unusable_count"] == 1
    assert summary["raw_attempt_outcome_counts"]["schema_error"] == 2
    assert summary["retry_diagnostics"]["schema_retry_case_count"] == 1
    assert summary["retry_diagnostics"]["schema_retry_repaired_count"] == 0
    assert summary["retry_diagnostics"]["schema_retry_exhausted_count"] == 1
    records = e28_test._artifact_records(output)
    first_case = _case_records(records)[0]
    assert first_case["wire"] == {
        "usable": False,
        "terminal_outcome": "transport_error",
        "final_attempt_outcome": "schema_error",
        "error_type": "LLMResponseError",
    }
    assert first_case["boundary_payload"] is None
    assert first_case["decode"]["status"] == "not_run"


def test_authorization_mismatch_prevents_http_and_output(
    monkeypatch, source_chain, e28_authorization, tmp_path
):
    output = tmp_path / "unauthorized.jsonl"
    prepared = _prepare(source_chain, e28_authorization, output)
    authorization = _authorization(prepared)
    object.__setattr__(authorization, "expected_contract_sha256", "0" * 64)
    calls = e28_test._install_responses(monkeypatch, [])

    with pytest.raises(
        ValueError, match="authorized exact boundary contract sha256 differs"
    ):
        run_exact_boundary_probe(
            output_path=output,
            prepared=prepared,
            generator_client=e28_test._generator_client(prepared),
            authorization=authorization,
        )
    assert calls == []
    assert not output.exists()
    assert not output.with_name(f"{output.name}.partial").exists()


def _cli_source_args(prepared):
    return [
        "--source-run", str(e28_test.RUN_PATH),
        "--source-diagnostic", str(e28_test.DIAGNOSTIC_PATH),
        "--e24-artifact", str(e28_test.E24_PATH),
        "--e25-artifact", str(e28_test.E25_PATH),
        "--e26-artifact", str(e28_test.E26_PATH),
        "--e27-artifact", str(e28_test.E27_PATH),
        "--e28-artifact", str(e28_test.E28_INVALID_PATH),
        "--problems", str(e28_test.SOURCE_PATH),
        "--start-index", "0",
        "--episode-count", "5",
        "--expected-source-decision-count", "66",
        "--expected-decompose-count", "33",
        "--expected-problem-count", "5",
        "--output", prepared.contract.planned_output_path,
    ]


def _cli_authorization_args(prepared, *, contract_sha256=None):
    contract = prepared.contract
    args = ["--preregistration-id", "E29-v2-boundary-exact-schema33"]
    for field_name in exact_cli._E29_DIGEST_FIELDS:
        contract_name = field_name.removeprefix("expected_")
        if contract_name == "contract_sha256":
            value = contract.sha256
            if contract_sha256 is not None:
                value = contract_sha256
        else:
            value = getattr(contract, contract_name)
        args.extend([
            "--" + contract_name.replace("_", "-"),
            value,
        ])
    return args


def test_cli_print_contract_reads_no_key_writes_no_output_or_http(
    monkeypatch, prepared, capsys
):
    output = e28_test.ROOT / prepared.contract.planned_output_path
    partial = output.with_name(f"{output.name}.partial")
    output_before = e28_test._path_snapshot(output)
    partial_before = e28_test._path_snapshot(partial)
    preparation_kwargs = {}
    calls = e28_test._install_responses(monkeypatch, [])

    def fake_prepare(**kwargs):
        preparation_kwargs.update(kwargs)
        return prepared

    monkeypatch.setattr(exact_cli, "prepare_exact_boundary_probe", fake_prepare)
    monkeypatch.setattr(
        exact_cli,
        "_secret",
        lambda *args: pytest.fail("print mode attempted secret lookup"),
    )
    monkeypatch.setattr(
        exact_cli,
        "run_exact_boundary_probe",
        lambda **kwargs: pytest.fail("print mode attempted execution"),
    )

    assert exact_cli.main(["--print-contract", *_cli_source_args(prepared)]) == 0

    assert calls == []
    assert e28_test._path_snapshot(output) == output_before
    assert e28_test._path_snapshot(partial) == partial_before
    assert preparation_kwargs["e28_authorization"] == (
        exact_cli._sealed_e28_authorization()
    )
    assert preparation_kwargs["expected_e28_artifact_sha256"] == (
        e28_test.E28_INVALID_SHA256
    )
    record = json.loads(capsys.readouterr().out)
    assert record["schema_version"] == (
        "decomposition-boundary-exact-preparation-v1"
    )
    assert record["exact_boundary_contract"] == prepared.contract.to_record()
    assert record["invalidated_e28_raw_audit"] == (
        prepared.invalidated_e28_raw_audit.to_record()
    )
    assert record["schema_validation"] == (
        exact_boundary_schema_validation_record()
    )
    assert record["retry_policy"] == exact_boundary_retry_policy_record()


def test_cli_authorization_mismatch_precedes_key_output_and_http(
    monkeypatch, prepared
):
    output = e28_test.ROOT / prepared.contract.planned_output_path
    partial = output.with_name(f"{output.name}.partial")
    output_before = e28_test._path_snapshot(output)
    partial_before = e28_test._path_snapshot(partial)
    calls = e28_test._install_responses(monkeypatch, [])
    monkeypatch.setattr(
        exact_cli,
        "prepare_exact_boundary_probe",
        lambda **kwargs: prepared,
    )
    monkeypatch.setattr(
        exact_cli,
        "_secret",
        lambda *args: pytest.fail("mismatch attempted secret lookup"),
    )
    monkeypatch.setattr(
        exact_cli,
        "run_exact_boundary_probe",
        lambda **kwargs: pytest.fail("mismatch attempted execution"),
    )

    with pytest.raises(SystemExit) as error:
        exact_cli.main([
            "--execute-hosted",
            *_cli_source_args(prepared),
            *_cli_authorization_args(prepared, contract_sha256="0" * 64),
        ])

    assert error.value.code == 2
    assert calls == []
    assert e28_test._path_snapshot(output) == output_before
    assert e28_test._path_snapshot(partial) == partial_before
