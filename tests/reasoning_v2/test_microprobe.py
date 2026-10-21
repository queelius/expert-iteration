import json
import threading
from collections import Counter
from pathlib import Path

import pytest

import experiments.probe_reasoning_v2_generator as probe_cli
from verifier_anchored_rl.llm_client import (
    OpenAICompatibleLLMClient,
    build_openai_compatible_json_body,
)
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
from verifier_anchored_rl.reasoning_v2.lenses import (
    LEGACY_LENS_CONTRACT_VERSION,
    LENS_CONTRACT_VERSION,
    build_policy_lens_request,
)
from verifier_anchored_rl.reasoning_v2.microprobe import (
    GENERATOR_MICROPROBE_CASE_VERSION,
    GENERATOR_MICROPROBE_SUMMARY_VERSION,
    GeneratorMicroprobeAuthorization,
    GeneratorMicroprobeValidationError,
    file_sha256,
    prepare_generator_microprobe,
    read_generator_microprobe_artifact,
    run_generator_microprobe,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "datasets/casework-v1/train.jsonl"
RUN_PATH = (
    ROOT / "experiments/runs/2026-07-13-E23-v2-hosted-smoke5.jsonl"
)
DIAGNOSTIC_PATH = ROOT / (
    "experiments/runs/"
    "2026-07-13-E23-v2-hosted-smoke5.diagnostics.jsonl"
)
RUN_SHA256 = "5efea19bd29619aba71d6d147135aababd7ded474230918b7d77b72d8b7368f6"
DIAGNOSTIC_SHA256 = (
    "0418043eb3abb50d86ad03c01946cc83602e76c8bd2c91f41defb87d3af6e369"
)
SOURCE_SHA256 = (
    "f3be145c1e536a98d4207c54ef85b5e0e516bce777c6ccd3d2a1f846c8881426"
)


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
                    "body": json,
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
    return prepare_generator_microprobe(
        run_path=RUN_PATH,
        diagnostic_path=DIAGNOSTIC_PATH,
        problem_source=_source(),
        probe_lens_contract=LENS_CONTRACT_VERSION,
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


def _authorization(prepared, *, probe_sha256=None):
    return GeneratorMicroprobeAuthorization(
        "E24-test",
        prepared.contract.source_run_sha256,
        prepared.contract.source_diagnostic_sha256,
        prepared.contract.sha256 if probe_sha256 is None else probe_sha256,
    )


def _request_body(prepared, case, *, probe):
    environment = (
        prepared.probe_environment if probe else prepared.source_environment
    )
    lens_contract = (
        prepared.contract.probe_lens_contract
        if probe
        else prepared.contract.source_lens_contract
    )
    observation = case.probe_observation if probe else case.source_observation
    request = build_policy_lens_request(
        observation,
        case.action,
        lens_contract=lens_contract,
    )
    return request, build_openai_compatible_json_body(
        model=prepared.contract.generator.model,
        system=request.system,
        user=request.user,
        schema=request.schema,
        temperature=environment.contract.generation_temperature,
        max_tokens=environment.contract.generation_max_tokens,
        seed=case.effective_seed,
        guided_json=prepared.contract.generator_transport.guided_json,
    )


def _cli_source_args():
    return [
        "--source-run",
        str(RUN_PATH),
        "--source-diagnostic",
        str(DIAGNOSTIC_PATH),
        "--problems",
        str(SOURCE_PATH),
        "--start-index",
        "0",
        "--episode-count",
        "5",
        "--probe-lens-contract",
        LENS_CONTRACT_VERSION,
        "--expected-decision-count",
        "66",
    ]


def _default_content(case):
    if case.action.verb.value == "decompose":
        return '{"domains":["x < 0","x >= 0"]}'
    if case.action.verb.value == "propose_refutation":
        return '{"candidate":"999"}'
    return '{"value":"999"}'


def _probe_responses(prepared, overrides=None):
    overrides = {} if overrides is None else overrides
    responses = []
    for case in prepared.cases:
        contents = overrides.get(case.index, (_default_content(case),))
        responses.extend(_Response(content) for content in contents)
    return responses


def _artifact_records(path):
    return [
        json.loads(line)
        for line in path.read_text("ascii").splitlines()
    ]


@pytest.fixture(scope="module")
def executed_probe(prepared, tmp_path_factory):
    overrides = {
        0: (
            "not-json",
            '{"domains":["x <= 4","x > 4"]}',
        ),
        1: ('{"value":""}',),
        2: ('{"domains":["x < 0","x >= 0"]}',),
        3: ('{"value":"5"}',),
        4: ('{"domains":"bad"}', '{"domains":"bad"}'),
        5: ("not-json", "still-not-json"),
    }
    monkeypatch = pytest.MonkeyPatch()
    calls = _install_responses(
        monkeypatch, _probe_responses(prepared, overrides)
    )
    output_path = tmp_path_factory.mktemp("microprobe") / "probe.jsonl"
    try:
        result = run_generator_microprobe(
            output_path=output_path,
            prepared=prepared,
            generator_client=_generator_client(prepared),
            authorization=_authorization(prepared),
        )
    finally:
        monkeypatch.undo()
    return result, output_path, calls


def test_prepare_is_exact_fixed_generator_only_projection(prepared):
    prepared.validate_trusted()
    assert file_sha256(RUN_PATH) == RUN_SHA256
    assert file_sha256(DIAGNOSTIC_PATH) == DIAGNOSTIC_SHA256
    assert prepared.contract.source_run_sha256 == RUN_SHA256
    assert prepared.contract.source_diagnostic_sha256 == DIAGNOSTIC_SHA256
    assert prepared.contract.source_problem_source_sha256 == SOURCE_SHA256
    assert prepared.contract.source_decision_count == 66
    assert prepared.contract.case_manifest_sha256 == canonical_json_sha256([
        case.manifest_record() for case in prepared.cases
    ])
    assert prepared.contract.to_record()["case_manifest_sha256"] == (
        prepared.contract.case_manifest_sha256
    )
    assert prepared.contract.source_lens_contract == (
        LEGACY_LENS_CONTRACT_VERSION
    )
    assert prepared.contract.probe_lens_contract == LENS_CONTRACT_VERSION
    assert len(prepared.cases) == 66
    assert [case.index for case in prepared.cases] == list(range(66))
    assert Counter(case.action.verb.value for case in prepared.cases) == {
        "decompose": 33,
        "propose_result": 17,
        "propose_solution": 16,
    }
    assert {case.action.target for case in prepared.cases} == {0}
    assert prepared.cases[0].effective_seed == 0
    assert all(
        case.effective_seed == case.state.turn for case in prepared.cases
    )
    assert [len(episode.decisions) for episode in prepared.replayed_run.episodes] == [
        16,
        16,
        16,
        16,
        2,
    ]


def test_prepare_reconstructs_every_legacy_request_and_only_changes_lens(
    prepared,
):
    for case in prepared.cases:
        source_request, source_body = _request_body(
            prepared, case, probe=False
        )
        probe_request, probe_body = _request_body(
            prepared, case, probe=True
        )
        assert canonical_json_sha256(source_body) == (
            case.source_request_sha256
        )
        assert canonical_json_sha256(probe_body) == case.probe_request_sha256
        assert source_body["model"] == probe_body["model"]
        assert source_body["temperature"] == probe_body["temperature"]
        assert source_body["max_tokens"] == probe_body["max_tokens"]
        assert source_body["seed"] == probe_body["seed"]
        assert source_request.schema == probe_request.schema
        assert '"schema_version":"typed-claim-lens-v2"' in (
            source_request.user
        )
        assert '"schema_version":"typed-claim-lens-v2.1"' in (
            probe_request.user
        )
        if case.action.verb.value != "decompose":
            assert "symbolic value" in source_request.user
        assert "symbolic value" not in probe_request.user
        probe_request.user.encode("ascii")


def test_prepare_reads_no_secret_and_makes_no_http(monkeypatch):
    calls = _install_responses(monkeypatch, [])
    monkeypatch.delenv("HF_TOKEN", raising=False)

    result = prepare_generator_microprobe(
        run_path=RUN_PATH,
        diagnostic_path=DIAGNOSTIC_PATH,
        problem_source=_source(),
        probe_lens_contract=LENS_CONTRACT_VERSION,
    )

    result.validate_trusted()
    assert calls == []


def test_authorization_mismatch_precedes_output_and_http(
    monkeypatch, tmp_path, prepared
):
    calls = _install_responses(monkeypatch, [])
    output_path = tmp_path / "mismatch.jsonl"

    with pytest.raises(ValueError, match="contract digest differs"):
        run_generator_microprobe(
            output_path=output_path,
            prepared=prepared,
            generator_client=_generator_client(prepared),
            authorization=_authorization(prepared, probe_sha256="f" * 64),
        )

    assert calls == []
    assert not output_path.exists()
    assert not (tmp_path / "mismatch.jsonl.partial").exists()


def test_probe_persists_raw_attempts_and_separate_stage_classifications(
    prepared, executed_probe
):
    result, output_path, calls = executed_probe

    assert result.output_path == output_path
    assert result.case_count == 66
    assert len(calls) == 69
    assert {call["url"] for call in calls} == {
        "https://router.huggingface.co/v1/chat/completions"
    }
    assert all(
        call["headers"]["Authorization"] == "Bearer GENERATOR_SECRET"
        for call in calls
    )
    assert "GENERATOR_SECRET" not in output_path.read_text("ascii")
    assert not output_path.with_name("probe.jsonl.partial").exists()

    records = _artifact_records(output_path)
    attempts = [
        record for record in records
        if record.get("schema_version") == DIAGNOSTIC_ATTEMPT_VERSION
    ]
    call_records = [
        record for record in records
        if record.get("schema_version") == DIAGNOSTIC_CALL_VERSION
    ]
    cases = [
        record for record in records
        if record.get("schema_version") == GENERATOR_MICROPROBE_CASE_VERSION
    ]
    summary = records[-1]
    assert len(attempts) == 69
    assert len(call_records) == 66
    assert len(cases) == 66
    assert summary["schema_version"] == GENERATOR_MICROPROBE_SUMMARY_VERSION
    assert read_generator_microprobe_artifact(output_path) == summary

    for prepared_case, record in zip(prepared.cases, cases, strict=True):
        assert {
            key: record[key] for key in prepared_case.manifest_record()
        } == prepared_case.manifest_record()
        assert record["probe_call_record_id"].startswith("generator:")

    first_attempts = [
        attempt for attempt in attempts
        if attempt["call_id"] == cases[0]["probe_call_record_id"].split(":")[1]
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
    assert cases[0]["wire"]["usable"] is True
    assert cases[0]["decode"]["status"] == "decoded"
    assert cases[0]["counterfactual_symbolic"]["application_status"] == (
        "applied"
    )

    assert cases[1]["wire"]["usable"] is True
    assert cases[1]["decode"]["status"] == "payload_error"
    assert cases[1]["counterfactual_symbolic"] is None

    assert cases[2]["decode"]["status"] == "decoded"
    assert cases[2]["counterfactual_symbolic"]["application_status"] == (
        "algebra_rejected"
    )
    assert cases[2]["counterfactual_symbolic"]["macro_outcome"][
        "reason"
    ] == "algebra_rejected"

    symbolic = cases[3]["counterfactual_symbolic"]
    assert symbolic["application_status"] == "applied"
    assert [
        event["outcome"]["action"]["verb"]
        for event in symbolic["closure_events"]
    ] == ["verify", "commit"]
    assert symbolic["counterfactual_reward"] == 1
    assert symbolic["counterfactual_terminal"] is True

    assert cases[4]["wire"]["usable"] is False
    assert cases[4]["wire"]["final_attempt_outcome"] == "schema_error"
    assert cases[4]["decode"]["status"] == "not_run"
    assert cases[4]["counterfactual_symbolic"] is None
    assert cases[5]["wire"]["usable"] is False
    assert cases[5]["wire"]["final_attempt_outcome"] == "parse_error"
    assert cases[5]["decode"]["status"] == "not_run"

    assert summary["raw_attempt_count"] == 69
    assert summary["generator_call_count"] == 66
    assert summary["wire_usable_count"] == 64
    assert summary["wire_unusable_count"] == 2
    assert summary["decode_counts"] == {
        "decoded": 63,
        "not_run": 2,
        "payload_error": 1,
    }
    assert summary["raw_attempt_outcome_counts"] == {
        "accepted": 64,
        "http_status_error": 0,
        "parse_error": 3,
        "provider_response_error": 0,
        "schema_error": 2,
        "transport_error": 0,
    }
    assert summary["reported_usage"] == {
        "attempt_count": 69,
        "completion_tokens": 138,
        "prompt_tokens": 207,
        "total_tokens": 345,
    }


def test_reader_rejects_a_tampered_case_bound_by_the_manifest(
    tmp_path, prepared, executed_probe
):
    _, output_path, _ = executed_probe
    records = _artifact_records(output_path)
    header = records[0]
    summary = records[-1]
    assert header["probe_contract"]["case_manifest_sha256"] == (
        prepared.contract.case_manifest_sha256
    )
    assert header["case_manifest_sha256"] == (
        prepared.contract.case_manifest_sha256
    )
    assert summary["case_manifest_sha256"] == (
        prepared.contract.case_manifest_sha256
    )
    case_record = next(
        record for record in records
        if record.get("schema_version") == GENERATOR_MICROPROBE_CASE_VERSION
    )
    case_record["probe_request_sha256"] = "f" * 64
    tampered_path = tmp_path / "tampered.jsonl"
    tampered_path.write_bytes(b"".join(
        canonical_json_bytes(record) + b"\n" for record in records
    ))

    with pytest.raises(GeneratorMicroprobeValidationError):
        read_generator_microprobe_artifact(tampered_path)


def test_interruption_keeps_only_an_exclusive_partial_artifact(
    monkeypatch, tmp_path, prepared
):
    calls = _install_responses(
        monkeypatch, [KeyboardInterrupt("test interruption")]
    )
    output_path = tmp_path / "interrupted.jsonl"

    with pytest.raises(KeyboardInterrupt, match="test interruption"):
        run_generator_microprobe(
            output_path=output_path,
            prepared=prepared,
            generator_client=_generator_client(prepared),
            authorization=_authorization(prepared),
        )

    assert len(calls) == 1
    assert not output_path.exists()
    assert (tmp_path / "interrupted.jsonl.partial").exists()


def test_existing_final_or_partial_is_never_clobbered(
    monkeypatch, tmp_path, prepared
):
    calls = _install_responses(monkeypatch, [])
    output_path = tmp_path / "owned.jsonl"
    output_path.write_text("owned\n", encoding="ascii")

    with pytest.raises(FileExistsError, match="already exists"):
        run_generator_microprobe(
            output_path=output_path,
            prepared=prepared,
            generator_client=_generator_client(prepared),
            authorization=_authorization(prepared),
        )

    assert calls == []
    assert output_path.read_text("ascii") == "owned\n"

    output_path.unlink()
    partial_path = tmp_path / "owned.jsonl.partial"
    partial_path.write_text("partial-owned\n", encoding="ascii")
    with pytest.raises(FileExistsError, match="already exists"):
        run_generator_microprobe(
            output_path=output_path,
            prepared=prepared,
            generator_client=_generator_client(prepared),
            authorization=_authorization(prepared),
        )
    assert calls == []
    assert partial_path.read_text("ascii") == "partial-owned\n"


def test_cli_help_exits_before_source_or_network(monkeypatch):
    monkeypatch.setattr(
        probe_cli,
        "prepare_generator_microprobe",
        lambda **kwargs: pytest.fail("help attempted probe preparation"),
    )

    with pytest.raises(SystemExit) as error:
        probe_cli.main(["--help"])

    assert error.value.code == 0


def test_cli_print_contract_reads_no_key_and_makes_no_http(
    monkeypatch, capsys, prepared
):
    calls = _install_responses(monkeypatch, [])
    monkeypatch.setattr(
        probe_cli,
        "_secret",
        lambda *args: pytest.fail("prepare mode attempted secret lookup"),
    )
    monkeypatch.delenv("UNUSED_HF_TOKEN", raising=False)

    exit_code = probe_cli.main([
        "--print-contract",
        *_cli_source_args(),
    ])

    assert exit_code == 0
    assert calls == []
    summary = json.loads(capsys.readouterr().out)
    assert summary["schema_version"] == "generator-microprobe-preparation-v1"
    assert summary["probe_contract_sha256"] == prepared.contract.sha256
    assert summary["probe_contract"] == prepared.contract.to_record()
    assert summary["decision_count"] == 66
    assert summary["source_environment_contract_sha256"] == (
        "9316c93bc2cdf9b2ce507a989882af69669f8bf886e1f2ad6ce802c77dfdcfa7"
    )
    assert summary["probe_environment_contract_sha256"] == (
        "664deedf1caa44efff16daac1888c2e3966aa37e9a2631a62036643a9e681c71"
    )


def test_cli_authorization_mismatch_precedes_key_output_and_http(
    monkeypatch, tmp_path, prepared
):
    calls = _install_responses(monkeypatch, [])
    monkeypatch.setattr(
        probe_cli,
        "_secret",
        lambda *args: pytest.fail("mismatch attempted secret lookup"),
    )
    output_path = tmp_path / "unauthorized.jsonl"

    with pytest.raises(SystemExit) as error:
        probe_cli.main([
            "--execute-hosted",
            *_cli_source_args(),
            "--preregistration-id",
            "E24-test",
            "--expected-source-run-sha256",
            prepared.contract.source_run_sha256,
            "--expected-source-diagnostic-sha256",
            prepared.contract.source_diagnostic_sha256,
            "--expected-probe-contract-sha256",
            "f" * 64,
            "--generator-api-key-env",
            "MISSING_HF_TOKEN",
            "--output",
            str(output_path),
        ])

    assert error.value.code == 2
    assert calls == []
    assert not output_path.exists()
    assert not (tmp_path / "unauthorized.jsonl.partial").exists()
