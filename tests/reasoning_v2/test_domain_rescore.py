import json
from pathlib import Path

import pytest

import experiments.rescore_reasoning_v2_domains as rescore_cli
import verifier_anchored_rl.reasoning_v2.domain_rescore as rescore_module
from verifier_anchored_rl.reasoning_v2.actions import DecompositionDraft
from verifier_anchored_rl.reasoning_v2.casework_source import (
    CaseworkProblemSource,
)
from verifier_anchored_rl.reasoning_v2.contracts import (
    canonical_json_bytes,
    canonical_json_sha256,
)
from verifier_anchored_rl.reasoning_v2.domain_rescore import (
    DOMAIN_RESCORE_CASE_VERSION,
    DOMAIN_RESCORE_EXPECTED_CORRECTED_REASONS,
    DOMAIN_RESCORE_EXPECTED_LEGACY_REASONS,
    DOMAIN_RESCORE_SUMMARY_VERSION,
    DomainRescoreAuthorization,
    DomainRescoreValidationError,
    prepare_domain_rescore,
    read_domain_rescore_artifact,
    run_domain_rescore,
    score_decomposition_draft,
)
from verifier_anchored_rl.reasoning_v2.math_adapter import (
    CORRECTED_DOMAIN_GRAMMAR_PROFILE,
    LEGACY_DOMAIN_GRAMMAR_PROFILE,
    CaseworkMathVerifier,
    validate_decomposition_coverage,
    validate_decomposition_coverage_v11,
)
from verifier_anchored_rl.reasoning_v2.model import (
    Action,
    Connective,
    GoalObjective,
    GoalObligation,
    State,
    Verb,
    new_graph,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "datasets/casework-v1/train.jsonl"
RUN_PATH = ROOT / "experiments/runs/2026-07-13-E23-v2-hosted-smoke5.jsonl"
DIAGNOSTIC_PATH = ROOT / (
    "experiments/runs/"
    "2026-07-13-E23-v2-hosted-smoke5.diagnostics.jsonl"
)
E24_PATH = ROOT / (
    "experiments/runs/2026-07-13-E24-v2-lens-v21-paired66.jsonl"
)
PLANNED_OUTPUT = (
    "experiments/runs/2026-07-13-E25-v2-domain-parser-rescore33.jsonl"
)

RUN_SHA256 = "5efea19bd29619aba71d6d147135aababd7ded474230918b7d77b72d8b7368f6"
DIAGNOSTIC_SHA256 = (
    "0418043eb3abb50d86ad03c01946cc83602e76c8bd2c91f41defb87d3af6e369"
)
SOURCE_SHA256 = (
    "f3be145c1e536a98d4207c54ef85b5e0e516bce777c6ccd3d2a1f846c8881426"
)
E24_SHA256 = "507c3b818ebee6c160bec021b84c60290f4af1a6fdcae725d2fa4efae74a5efc"
E24_CONTRACT_SHA256 = (
    "3ef15fcccc0c3ce4fc109ae07ef3ad20ef29cb7677146b9fdce8f329352b019c"
)
E24_MANIFEST_SHA256 = (
    "629205cc6f0047a19b81f31750c1fa0c74df1e30f2a142b026d1bdfcf100049d"
)


def _source():
    return CaseworkProblemSource.from_jsonl(SOURCE_PATH).select_slice(0, 5)


def _prepare(output):
    return prepare_domain_rescore(
        source_run_path=RUN_PATH,
        source_diagnostic_path=DIAGNOSTIC_PATH,
        e24_artifact_path=E24_PATH,
        problem_source=_source(),
        planned_output_path=output,
    )


@pytest.fixture(scope="module")
def prepared():
    return _prepare(PLANNED_OUTPUT)


def _authorization(prepared, **overrides):
    contract = prepared.contract
    values = {
        "preregistration_id": "E25-test",
        "expected_source_run_sha256": contract.source_run_sha256,
        "expected_source_diagnostic_sha256": (
            contract.source_diagnostic_sha256
        ),
        "expected_source_environment_contract_sha256": (
            contract.source_environment_contract_sha256
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
        "expected_e24_probe_environment_contract_sha256": (
            contract.e24_probe_environment_contract_sha256
        ),
        "expected_corrected_environment_contract_sha256": (
            contract.corrected_environment_contract_sha256
        ),
        "expected_e24_case_manifest_sha256": (
            contract.e24_case_manifest_sha256
        ),
        "expected_rescore_case_manifest_sha256": (
            contract.rescore_case_manifest_sha256
        ),
        "expected_legacy_profile_sha256": contract.legacy_profile_sha256,
        "expected_corrected_profile_sha256": (
            contract.corrected_profile_sha256
        ),
        "expected_legacy_expectation_sha256": (
            contract.legacy_expectation_sha256
        ),
        "expected_corrected_expectation_sha256": (
            contract.corrected_expectation_sha256
        ),
        "expected_rescore_contract_sha256": contract.sha256,
        "expected_source_decision_count": contract.source_decision_count,
        "expected_case_count": contract.case_count,
        "expected_output_path": contract.planned_output_path,
    }
    values.update(overrides)
    return DomainRescoreAuthorization(**values)


def _artifact_records(path):
    return [json.loads(line) for line in path.read_text("ascii").splitlines()]


def _fake_profile_result(profile, reason, index):
    return {
        "profile_id": profile.domain_grammar_profile_id,
        "application_status": "algebra_rejected",
        "reason_code": reason,
        "canonical_children": None,
        "coverage": {
            "accepted": False,
            "context_digest": canonical_json_sha256({
                "index": index,
                "profile": profile.domain_grammar_profile_id,
                "reason": reason,
            }),
            "reason_code": reason,
        },
    }


def _install_registered_fake_scorer(monkeypatch, prepared, *, mismatch=False):
    corrected_reasons = [
        reason
        for reason, count in prepared.contract.expected_corrected_reason_counts
        for _ in range(count)
    ]
    if mismatch:
        corrected_reasons[0] = "malformed_scalar"
    call_count = 0

    def fake_score(**kwargs):
        nonlocal call_count
        case_index = call_count // 2
        legacy_arm = call_count % 2 == 0
        call_count += 1
        if legacy_arm:
            reason = prepared.cases[case_index].e24_legacy_reason_code
        else:
            reason = corrected_reasons[case_index]
        assert reason is not None
        return _fake_profile_result(kwargs["profile"], reason, case_index)

    monkeypatch.setattr(
        rescore_module, "score_decomposition_draft", fake_score
    )
    return lambda: call_count


@pytest.fixture(scope="module")
def executed_rescore(tmp_path_factory):
    output = tmp_path_factory.mktemp("domain-rescore") / "rescore.jsonl"
    prepared = _prepare(output)
    patcher = pytest.MonkeyPatch()
    calls = _install_registered_fake_scorer(patcher, prepared)
    try:
        result = run_domain_rescore(
            output_path=output,
            prepared=prepared,
            authorization=_authorization(prepared),
        )
    finally:
        patcher.undo()
    return prepared, result, output, calls()


def test_preparation_is_deterministic_exact_and_strictly_reconstructed(prepared):
    prepared.validate_trusted()
    contract = prepared.contract
    assert contract.source_run_sha256 == RUN_SHA256
    assert contract.source_diagnostic_sha256 == DIAGNOSTIC_SHA256
    assert contract.source_problem_source_sha256 == SOURCE_SHA256
    assert contract.e24_artifact_sha256 == E24_SHA256
    assert contract.e24_probe_contract_sha256 == E24_CONTRACT_SHA256
    assert contract.e24_case_manifest_sha256 == E24_MANIFEST_SHA256
    assert contract.source_decision_count == 66
    assert contract.case_count == 33
    assert contract.planned_output_path == PLANNED_OUTPUT
    assert len(prepared.source_microprobe.cases) == 66
    assert len(prepared.cases) == 33
    assert [case.index for case in prepared.cases] == list(range(33))
    assert [case.source_case.index for case in prepared.cases] == list(
        range(0, 66, 2)
    )
    assert all(case.action == Action(Verb.DECOMPOSE, 0) for case in prepared.cases)
    assert canonical_json_sha256([
        case.manifest_record() for case in prepared.cases
    ]) == contract.rescore_case_manifest_sha256
    assert all(
        canonical_json_sha256({
            "draft_type": "decomposition",
            "connective": case.draft.connective.value,
            "child_statements": list(case.draft.child_statements),
            "child_obligations": [
                {
                    "variable": child.variable,
                    "equation": child.equation,
                    "domain": child.domain,
                    "objective": child.objective.value,
                }
                for child in case.draft.child_obligations
            ],
        }) == case.e24_draft_sha256
        for case in prepared.cases
    )


def test_contract_binds_shared_implementation_profiles_and_expectations(prepared):
    contract = prepared.contract
    assert contract.legacy_profile.domain_grammar_profile_id == (
        LEGACY_DOMAIN_GRAMMAR_PROFILE.profile_id
    )
    assert contract.corrected_profile.domain_grammar_profile_id == (
        CORRECTED_DOMAIN_GRAMMAR_PROFILE.profile_id
    )
    assert contract.legacy_profile.adapter_implementation == (
        contract.corrected_profile.adapter_implementation
    )
    assert contract.legacy_profile.implementation_id == (
        contract.corrected_profile.implementation_id
    )
    assert contract.legacy_profile_sha256 == contract.legacy_profile.sha256
    assert contract.corrected_profile_sha256 == contract.corrected_profile.sha256
    assert contract.expected_legacy_applied_count == 0
    assert contract.expected_corrected_applied_count == 0
    assert contract.expected_legacy_reason_counts == (
        DOMAIN_RESCORE_EXPECTED_LEGACY_REASONS
    )
    assert contract.expected_corrected_reason_counts == (
        DOMAIN_RESCORE_EXPECTED_CORRECTED_REASONS
    )
    assert contract.corrected_environment_contract_sha256 != (
        contract.e24_probe_environment_contract_sha256
    )


def test_contract_preparation_repeats_exactly(prepared):
    repeated = _prepare(PLANNED_OUTPUT)
    assert repeated.contract == prepared.contract
    assert repeated.contract.sha256 == prepared.contract.sha256
    assert repeated.cases == prepared.cases


def test_preparation_makes_no_http(monkeypatch):
    import httpx

    class _NoNetwork:
        def __init__(self, *args, **kwargs):
            pytest.fail("domain rescore preparation attempted network access")

    monkeypatch.setattr(httpx, "Client", _NoNetwork)
    result = _prepare(PLANNED_OUTPUT)
    result.validate_trusted()


def test_authorization_mismatch_precedes_output(monkeypatch, tmp_path, prepared):
    output = tmp_path / "unauthorized.jsonl"
    monkeypatch.setattr(
        rescore_module,
        "score_decomposition_draft",
        lambda **kwargs: pytest.fail("unauthorized rescore reached scoring"),
    )
    with pytest.raises(ValueError, match="authorized rescore contract differs"):
        run_domain_rescore(
            output_path=output,
            prepared=prepared,
            authorization=_authorization(
                prepared, expected_rescore_contract_sha256="f" * 64
            ),
        )
    assert not output.exists()
    assert not (tmp_path / "unauthorized.jsonl.partial").exists()


def test_single_draft_exposes_only_the_upper_inclusive_parser_change(prepared):
    legacy = CaseworkMathVerifier()
    corrected = CaseworkMathVerifier.corrected_v11()
    obligation = GoalObligation(
        variable="x",
        equation="Eq(x, 5)",
        domain="Reals",
        objective=GoalObjective.IDENTITY,
    )
    graph = new_graph("solve", obligation, generation_contract=legacy)
    state = State(graph, turn=0)
    action = Action(Verb.DECOMPOSE, 0)
    draft = DecompositionDraft(
        connective=Connective.EXHAUSTIVE_CASE,
        child_statements=("low", "middle", "high"),
        child_obligations=(
            GoalObligation("x", "Eq(x, 5)", "x < 0", GoalObjective.BRANCH),
            GoalObligation(
                "x", "Eq(x, 5)", "0 <= x <= 10", GoalObjective.BRANCH
            ),
            GoalObligation("x", "Eq(x, 5)", "x > 10", GoalObjective.BRANCH),
        ),
    )
    old = score_decomposition_draft(
        state=state,
        action=action,
        draft=draft,
        adapter=legacy,
        coverage_validator=validate_decomposition_coverage,
        profile=prepared.contract.legacy_profile,
    )
    new = score_decomposition_draft(
        state=state,
        action=action,
        draft=draft,
        adapter=corrected,
        coverage_validator=validate_decomposition_coverage_v11,
        profile=prepared.contract.corrected_profile,
    )
    assert old["application_status"] == "algebra_rejected"
    assert old["reason_code"] == "malformed_scalar"
    assert new["application_status"] == "applied"
    assert new["reason_code"] is None
    assert [child["domain"] for child in new["canonical_children"]] == [
        "x < 0",
        "0 <= x <= 10",
        "x > 10",
    ]
    assert old["coverage"]["context_digest"] != (
        new["coverage"]["context_digest"]
    )


def test_stubbed_publication_is_canonical_and_no_clobber(executed_rescore):
    prepared, result, output, call_count = executed_rescore
    assert call_count == 66
    assert result.output_path == output
    assert result.case_count == 33
    assert result.legacy_applied_count == 0
    assert result.corrected_applied_count == 0
    assert not output.with_name(f"{output.name}.partial").exists()
    summary = read_domain_rescore_artifact(output)
    assert summary["schema_version"] == DOMAIN_RESCORE_SUMMARY_VERSION
    assert summary["legacy_reason_counts"] == dict(
        DOMAIN_RESCORE_EXPECTED_LEGACY_REASONS
    )
    assert summary["corrected_reason_counts"] == dict(
        DOMAIN_RESCORE_EXPECTED_CORRECTED_REASONS
    )
    records = _artifact_records(output)
    cases = [
        record
        for record in records
        if record.get("schema_version") == DOMAIN_RESCORE_CASE_VERSION
    ]
    assert len(cases) == 33
    assert records[-1] == summary

    with pytest.raises(FileExistsError, match="already exists"):
        run_domain_rescore(
            output_path=output,
            prepared=prepared,
            authorization=_authorization(prepared),
        )


def test_reader_rejects_tampered_paired_result(
    tmp_path, executed_rescore
):
    _, _, output, _ = executed_rescore
    records = _artifact_records(output)
    case = next(
        record
        for record in records
        if record.get("schema_version") == DOMAIN_RESCORE_CASE_VERSION
    )
    case["corrected"]["coverage"]["context_digest"] = "f" * 64
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_bytes(b"".join(
        canonical_json_bytes(record) + b"\n" for record in records
    ))
    with pytest.raises(DomainRescoreValidationError):
        read_domain_rescore_artifact(tampered)


def test_unexpected_distribution_keeps_only_partial(
    monkeypatch, tmp_path
):
    output = tmp_path / "unexpected.jsonl"
    prepared = _prepare(output)
    calls = _install_registered_fake_scorer(
        monkeypatch, prepared, mismatch=True
    )
    with pytest.raises(
        DomainRescoreValidationError,
        match="differs from the registered expectation",
    ):
        run_domain_rescore(
            output_path=output,
            prepared=prepared,
            authorization=_authorization(prepared),
        )
    assert calls() == 66
    assert not output.exists()
    assert output.with_name(f"{output.name}.partial").exists()


def test_preparation_rejects_a_tampered_e24_artifact(tmp_path):
    records = _artifact_records(E24_PATH)
    case = next(
        record
        for record in records
        if record.get("schema_version") == "generator-paired-microprobe-case-v1"
    )
    case["probe_request_sha256"] = "f" * 64
    tampered = tmp_path / "e24-tampered.jsonl"
    tampered.write_bytes(b"".join(
        canonical_json_bytes(record) + b"\n" for record in records
    ))
    with pytest.raises(ValueError):
        prepare_domain_rescore(
            source_run_path=RUN_PATH,
            source_diagnostic_path=DIAGNOSTIC_PATH,
            e24_artifact_path=tampered,
            problem_source=_source(),
            planned_output_path=tmp_path / "unused.jsonl",
        )


def _cli_source_args(output):
    return [
        "--source-run", str(RUN_PATH),
        "--source-diagnostic", str(DIAGNOSTIC_PATH),
        "--e24-artifact", str(E24_PATH),
        "--problems", str(SOURCE_PATH),
        "--start-index", "0",
        "--episode-count", "5",
        "--expected-source-decision-count", "66",
        "--expected-decompose-count", "33",
        "--output", str(output),
    ]


def test_cli_help_exits_before_preparation(monkeypatch):
    monkeypatch.setattr(
        rescore_cli,
        "prepare_domain_rescore",
        lambda **kwargs: pytest.fail("help attempted rescore preparation"),
    )
    with pytest.raises(SystemExit) as error:
        rescore_cli.main(["--help"])
    assert error.value.code == 0


def test_cli_print_contract_is_zero_write_and_zero_execution(
    monkeypatch, capsys, prepared, tmp_path
):
    planned_output = tmp_path / "print-only.jsonl"
    monkeypatch.setattr(
        rescore_cli, "prepare_domain_rescore", lambda **kwargs: prepared
    )
    monkeypatch.setattr(
        rescore_cli,
        "run_domain_rescore",
        lambda **kwargs: pytest.fail("print-contract attempted execution"),
    )
    exit_code = rescore_cli.main([
        "--print-contract",
        *_cli_source_args(planned_output),
    ])
    assert exit_code == 0
    record = json.loads(capsys.readouterr().out)
    assert record["schema_version"] == "domain-parser-rescore-preparation-v1"
    assert record["rescore_contract_sha256"] == prepared.contract.sha256
    assert record["source_decision_count"] == 66
    assert record["decompose_case_count"] == 33
    assert not planned_output.exists()
    assert not planned_output.with_name("print-only.jsonl.partial").exists()


def test_cli_rejects_digest_mismatch_before_execution(
    monkeypatch, prepared
):
    monkeypatch.setattr(
        rescore_cli, "prepare_domain_rescore", lambda **kwargs: prepared
    )
    monkeypatch.setattr(
        rescore_cli,
        "run_domain_rescore",
        lambda **kwargs: pytest.fail("mismatch attempted execution"),
    )
    contract = prepared.contract
    args = [
        "--execute",
        *_cli_source_args(PLANNED_OUTPUT),
        "--preregistration-id", "E25-test",
        "--expected-source-run-sha256", contract.source_run_sha256,
        "--expected-source-diagnostic-sha256", contract.source_diagnostic_sha256,
        "--expected-source-environment-sha256", contract.source_environment_contract_sha256,
        "--expected-source-collection-sha256", contract.source_collection_contract_sha256,
        "--expected-source-problem-sha256", contract.source_problem_source_sha256,
        "--expected-e24-artifact-sha256", contract.e24_artifact_sha256,
        "--expected-e24-probe-contract-sha256", contract.e24_probe_contract_sha256,
        "--expected-e24-probe-environment-sha256", contract.e24_probe_environment_contract_sha256,
        "--expected-corrected-environment-sha256", contract.corrected_environment_contract_sha256,
        "--expected-e24-case-manifest-sha256", contract.e24_case_manifest_sha256,
        "--expected-rescore-case-manifest-sha256", contract.rescore_case_manifest_sha256,
        "--expected-legacy-profile-sha256", contract.legacy_profile_sha256,
        "--expected-corrected-profile-sha256", contract.corrected_profile_sha256,
        "--expected-legacy-expectation-sha256", contract.legacy_expectation_sha256,
        "--expected-corrected-expectation-sha256", contract.corrected_expectation_sha256,
        "--expected-rescore-contract-sha256", "f" * 64,
    ]
    with pytest.raises(SystemExit) as error:
        rescore_cli.main(args)
    assert error.value.code == 2
