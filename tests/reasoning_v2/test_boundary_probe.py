import json
import threading
from collections import Counter
from copy import copy, deepcopy
from pathlib import Path

import pytest

import experiments.probe_reasoning_v2_boundaries as probe_cli
import verifier_anchored_rl.reasoning_v2.boundary_probe as boundary_probe_module
from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient
from verifier_anchored_rl.reasoning_v2.boundary_probe import (
    BoundaryDecompositionProbeAuthorization,
    BoundaryDecompositionProbeValidationError,
    BoundaryPayload,
    boundary_payload_schema_record,
    boundary_payload_schema_sha256,
    build_boundary_request,
    construct_boundary_decomposition,
    decode_boundary_payload,
    prepare_boundary_decomposition_probe,
    read_boundary_decomposition_probe_artifact,
    run_boundary_decomposition_probe,
    validate_boundary_oracle_cases,
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
from verifier_anchored_rl.reasoning_v2.key_aligned_repair import (
    KeyAlignedRepairProbeAuthorization,
    prepare_key_aligned_repair_probe,
    read_key_aligned_repair_probe_artifact,
)
from verifier_anchored_rl.reasoning_v2.lenses import (
    LENS_CONTRACT_VERSION,
    build_policy_lens_request,
)
from verifier_anchored_rl.reasoning_v2.microprobe import file_sha256
from verifier_anchored_rl.reasoning_v2.model import (
    Action,
    Goal,
    GoalObjective,
    GoalObligation,
    Graph,
    State,
    Verb,
)
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
E27_PATH = ROOT / (
    "experiments/runs/2026-07-13-E27-v2-decompose-repair-key33.jsonl"
)
E28_INVALID_PATH = ROOT / (
    "experiments/runs/2026-07-13-E28-v2-boundary-decompose33.jsonl"
)
E26_OUTPUT = "experiments/runs/2026-07-13-E26-v2-decompose-repair33.jsonl"
E27_OUTPUT = (
    "experiments/runs/2026-07-13-E27-v2-decompose-repair-key33.jsonl"
)
E28_OUTPUT = "experiments/runs/2026-07-13-E28-v2-boundary-decompose33.jsonl"

E24_SHA256 = "507c3b818ebee6c160bec021b84c60290f4af1a6fdcae725d2fa4efae74a5efc"
E26_SHA256 = "7290e10733c1195edcea42684881afa4681f4c03f08e7aa7cf317a386623b1bd"
E27_SHA256 = "56f0e55ce70fd7ad327126a8f2e43f7a4ae011ea482e4d473f2f3032f5bbf55c"
E28_INVALID_SHA256 = (
    "f42e36bc426169ca2edb25237c26443e40eea7952b10242d25590e45bb96f404"
)
E27_CONTRACT_SHA256 = (
    "e3c4fc625c1cb79e2217a2fb86d8603a3b9955e9bac1fd2e59cb47fada078696"
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
GENERATOR_SHA256 = (
    "5fc23203d49417008fb63f5356f2a8e86cc249e77cbc3fac5b6a262858af8e58"
)
GENERATOR_TRANSPORT_SHA256 = (
    "b0e2806c348fcde14bf2bbc3c48656eb7547f3b7142afc93f99ba52c40896f2a"
)
GENERATION_SETTINGS_SHA256 = (
    "9059e1a6bbfb1422344ea9a6c9b9c5b85879f9fc0fe56de6ed2a4e1f846cd463"
)

BOUNDARY_PAYLOAD_VERSION = "decomposition-boundaries-v1"
BOUNDARY_CONTRACT_VERSION = "decomposition-boundary-probe-contract-v1"
BOUNDARY_HEADER_VERSION = "decomposition-boundary-probe-header-v1"
BOUNDARY_CASE_VERSION = "decomposition-boundary-probe-case-v1"
BOUNDARY_SUMMARY_VERSION = "decomposition-boundary-probe-summary-v1"
BOUNDARY_LENS_VERSION = "typed-claim-lens-v2.2-boundaries"
EXPECTED_INSTRUCTION = (
    "Return a JSON object with exactly one field named boundaries. Set "
    "boundaries to an array of 1 to 5 distinct ASCII finite rational scalar "
    "strings, ordered from least to greatest. A scalar may contain only "
    "digits, spaces, +, -, *, /, parentheses, and decimal points, with no ** "
    "or //. Do not use x, Reals, comparisons, interval notation, roots, "
    "infinity, or any other field. A boundary is a separator, not a solution. "
    "Every boundary must lie strictly inside the target domain. Include a "
    "boundary strictly between every adjacent pair of distinct in-domain "
    "solutions, using the fewest boundaries that satisfy this condition, so "
    "each induced interval contains at most one solution. If there are fewer "
    "than two in-domain solutions, still return one interior boundary. Return "
    "only one JSON object."
)
ORACLE_BOUNDARIES = {
    "cw1-ccdf9e1567dc": ["4"],
    "cw1-2640597e2425": ["6"],
    "cw1-52e3b6b3ec25": ["2"],
    "cw1-ddad3be834f4": ["4"],
    "cw1-4ba0fc91c9e8": ["-3"],
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


def _authorization_from_contract(cls, preregistration_id, contract):
    values = {"preregistration_id": preregistration_id}
    for name in cls.__dataclass_fields__:
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
    return cls(**values)


@pytest.fixture(scope="module")
def source_chain():
    semantic = prepare_semantic_repair_probe(
        source_run_path=RUN_PATH,
        source_diagnostic_path=DIAGNOSTIC_PATH,
        e24_artifact_path=E24_PATH,
        e25_artifact_path=E25_PATH,
        problem_source=_source(),
        planned_output_path=E26_OUTPUT,
    )
    e26_authorization = _authorization_from_contract(
        SemanticRepairProbeAuthorization,
        "E26-v2-decompose-repair33",
        semantic.contract,
    )
    key_aligned = prepare_key_aligned_repair_probe(
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
    e27_authorization = _authorization_from_contract(
        KeyAlignedRepairProbeAuthorization,
        "E27-v2-decompose-repair-key33",
        key_aligned.contract,
    )
    e27_summary = read_key_aligned_repair_probe_artifact(
        E27_PATH,
        prepared=key_aligned,
        authorization=e27_authorization,
    )
    assert e27_summary["wire_usable_count"] == 33
    assert e27_summary["typed_decoded_count"] == 32
    assert e27_summary["symbolic_applied_count"] == 0
    assert e27_summary["applied_problem_count"] == 0
    return e26_authorization, e27_authorization, key_aligned, e27_summary


@pytest.fixture(scope="module")
def prepared(source_chain):
    e26_authorization, e27_authorization, _, _ = source_chain
    return prepare_boundary_decomposition_probe(
        source_run_path=RUN_PATH,
        source_diagnostic_path=DIAGNOSTIC_PATH,
        e24_artifact_path=E24_PATH,
        e25_artifact_path=E25_PATH,
        e26_artifact_path=E26_PATH,
        e27_artifact_path=E27_PATH,
        e26_authorization=e26_authorization,
        e27_authorization=e27_authorization,
        expected_e26_artifact_sha256=E26_SHA256,
        expected_e27_artifact_sha256=E27_SHA256,
        problem_source=_source(),
        planned_output_path=E28_OUTPUT,
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


def _authorization(prepared, **overrides):
    authorization = _authorization_from_contract(
        BoundaryDecompositionProbeAuthorization,
        "E28-test",
        prepared.contract,
    )
    for name, value in overrides.items():
        object.__setattr__(authorization, name, value)
    return authorization


def _valid_content(case):
    return json.dumps(
        {"boundaries": ORACLE_BOUNDARIES[case.problem_id]},
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


def _prepare_for_output(source_chain, output_path):
    e26_authorization, e27_authorization, _, _ = source_chain
    return prepare_boundary_decomposition_probe(
        source_run_path=RUN_PATH,
        source_diagnostic_path=DIAGNOSTIC_PATH,
        e24_artifact_path=E24_PATH,
        e25_artifact_path=E25_PATH,
        e26_artifact_path=E26_PATH,
        e27_artifact_path=E27_PATH,
        e26_authorization=e26_authorization,
        e27_authorization=e27_authorization,
        expected_e26_artifact_sha256=E26_SHA256,
        expected_e27_artifact_sha256=E27_SHA256,
        problem_source=_source(),
        planned_output_path=output_path,
    )


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
        "--e27-artifact",
        str(E27_PATH),
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


def test_boundary_payload_schema_and_instruction_are_exact():
    assert boundary_payload_schema_record() == {
        "type": "object",
        "properties": {
            "boundaries": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 5,
            }
        },
        "required": ["boundaries"],
        "additionalProperties": False,
    }
    assert boundary_payload_schema_sha256() == canonical_json_sha256(
        boundary_payload_schema_record()
    )


def test_boundary_decoder_preserves_accepted_lexemes_without_coercion():
    source = {
        "boundaries": ["-3", "4/2", "2.5", "(7 + 1) / 2"]
    }
    payload = decode_boundary_payload(source)

    assert isinstance(payload, BoundaryPayload)
    assert payload.boundaries == tuple(source["boundaries"])
    assert source == {
        "boundaries": ["-3", "4/2", "2.5", "(7 + 1) / 2"]
    }


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"boundaries": [],},
        {"boundaries": ["0", "1", "2", "3", "4", "5"]},
        {"boundaries": [1]},
        {"boundaries": [True]},
        {"boundaries": [""]},
        {"boundaries": ["x"]},
        {"boundaries": ["sqrt(2)"]},
        {"boundaries": ["oo"]},
        {"boundaries": ["2**3"]},
        {"boundaries": ["4//2"]},
        {"boundaries": ["1/0"]},
        {"boundaries": ["2", "4/2"]},
        {"boundaries": ["2", "1"]},
        {"boundaries": ["1"], "domains": ["x < 1", "x >= 1"]},
        {"boundaries": ["\u221e"]},
    ],
)
def test_boundary_decoder_rejects_bad_shape_scalar_duplicates_and_order(value):
    before = deepcopy(value)
    with pytest.raises(ValueError):
        decode_boundary_payload(value)
    assert value == before


def _state_with_parent_domain(domain, *, equation="DO_NOT_PARSE_OR_SOLVE"):
    obligation = GoalObligation(
        variable="x",
        equation=equation,
        domain=domain,
        objective=GoalObjective.SUM,
    )
    graph = Graph(
        nodes=(
            Goal(
                id=0,
                parent_goal=None,
                statement="opaque",
                obligation=obligation,
            ),
        ),
        root_id=0,
    )
    return State(graph=graph, turn=0), Action(Verb.DECOMPOSE, 0)


@pytest.mark.parametrize(
    ("parent_domain", "boundaries", "expected_domains"),
    [
        ("Reals", ["4/2"], ["x < 4/2", "x >= 4/2"]),
        ("x < 10", ["2"], ["x < 2", "2 <= x < 10"]),
        ("x <= 10", ["2"], ["x < 2", "2 <= x <= 10"]),
        ("x > -10", ["2"], ["-10 < x < 2", "x >= 2"]),
        ("x >= -10", ["2"], ["-10 <= x < 2", "x >= 2"]),
        (
            "-10 < x < 10",
            ["2"],
            ["-10 < x < 2", "2 <= x < 10"],
        ),
        (
            "-10 <= x <= 10",
            ["2"],
            ["-10 <= x < 2", "2 <= x <= 10"],
        ),
        (
            "Reals",
            ["-3", "4 / 2"],
            ["x < -3", "-3 <= x < 4 / 2", "x >= 4 / 2"],
        ),
    ],
)
def test_constructor_preserves_parent_endpoints_half_open_topology_and_lexemes(
    parent_domain, boundaries, expected_domains
):
    state, action = _state_with_parent_domain(parent_domain)
    before = state
    payload = decode_boundary_payload({"boundaries": boundaries})

    draft = construct_boundary_decomposition(state, action, payload)

    assert [child.domain for child in draft.child_obligations] == expected_domains
    assert [child.equation for child in draft.child_obligations] == [
        "DO_NOT_PARSE_OR_SOLVE"
    ] * len(expected_domains)
    assert [child.variable for child in draft.child_obligations] == [
        "x"
    ] * len(expected_domains)
    assert [child.objective for child in draft.child_obligations] == [
        GoalObjective.BRANCH
    ] * len(expected_domains)
    assert state == before


@pytest.mark.parametrize(
    ("parent_domain", "boundary"),
    [
        ("x < 12", "12"),
        ("x < 12", "13"),
        ("x <= 12", "12"),
        ("x > -2", "-2"),
        ("x >= -2", "-3"),
        ("-2 < x < 12", "-2"),
        ("-2 <= x <= 12", "12"),
    ],
)
def test_constructor_rejects_parent_endpoint_and_exterior_without_clipping(
    parent_domain, boundary
):
    state, action = _state_with_parent_domain(parent_domain)
    payload = decode_boundary_payload({"boundaries": [boundary]})
    with pytest.raises(ValueError):
        construct_boundary_decomposition(state, action, payload)


def test_decoder_rejects_lexically_distinct_semantic_duplicate():
    value = {"boundaries": ["1", "2/2"]}
    before = deepcopy(value)
    with pytest.raises(ValueError):
        decode_boundary_payload(value)
    assert value == before


def test_prepare_binds_e24_cases_e27_branch_and_distinct_environments(
    prepared, source_chain
):
    _, _, key_aligned, e27_summary = source_chain
    prepared.validate_trusted()
    contract = prepared.contract.to_record()
    assert contract["schema_version"] == BOUNDARY_CONTRACT_VERSION
    assert contract["source_environment_contract_sha256"] == (
        SOURCE_ENVIRONMENT_SHA256
    )
    assert contract["request_environment_contract_sha256"] == (
        REQUEST_ENVIRONMENT_SHA256
    )
    assert contract["scoring_environment_contract_sha256"] == (
        SCORING_ENVIRONMENT_SHA256
    )
    assert contract["request_environment_contract_sha256"] != (
        contract["scoring_environment_contract_sha256"]
    )
    assert contract["corrected_profile_sha256"] == CORRECTED_PROFILE_SHA256
    assert contract["e24_artifact_sha256"] == E24_SHA256
    assert contract["e27_artifact_sha256"] == E27_SHA256
    assert contract["e27_repair_contract_sha256"] == E27_CONTRACT_SHA256
    assert contract["e27_case_manifest_sha256"] == (
        key_aligned.contract.case_manifest_sha256
    )
    assert contract["e27_summary_sha256"] == canonical_json_sha256(
        e27_summary
    )
    assert contract["generator_sha256"] == GENERATOR_SHA256
    assert contract["generator_transport_sha256"] == (
        GENERATOR_TRANSPORT_SHA256
    )
    assert contract["generation_settings_sha256"] == (
        GENERATION_SETTINGS_SHA256
    )
    assert contract["boundary_payload_schema_sha256"] == (
        boundary_payload_schema_sha256()
    )
    assert contract["source_decision_count"] == 66
    assert contract["case_count"] == 33
    assert contract["problem_count"] == 5
    assert len(prepared.cases) == 33
    assert [case.index for case in prepared.cases] == list(range(33))
    assert Counter(case.problem_id for case in prepared.cases) == {
        "cw1-ccdf9e1567dc": 8,
        "cw1-2640597e2425": 8,
        "cw1-52e3b6b3ec25": 8,
        "cw1-ddad3be834f4": 8,
        "cw1-4ba0fc91c9e8": 1,
    }
    assert [case.effective_seed for case in prepared.cases] == [
        index % 16 for index in range(0, 66, 2)
    ]
    assert contract["case_manifest_sha256"] == canonical_json_sha256([
        case.manifest_record() for case in prepared.cases
    ])


def test_boundary_request_changes_only_v21_instruction_and_schema(prepared):
    old_marker = (
        "Return a JSON object with exactly one field named domains."
    )
    terminal = "\nReturn only one JSON object."
    for case in prepared.cases:
        old_request = build_policy_lens_request(
            case.probe_observation,
            case.action,
            lens_contract=LENS_CONTRACT_VERSION,
        )
        request, base_body, boundary_body = build_boundary_request(
            prepared, case
        )
        old_start = old_request.user.index(old_marker)
        old_end = old_request.user.index(terminal, old_start)
        expected_user = (
            old_request.user[:old_start]
            + EXPECTED_INSTRUCTION
            + old_request.user[old_end + len(terminal):]
        )

        assert request.system == old_request.system
        assert request.user == expected_user
        assert request.user.count("Return only one JSON object.") == 1
        assert request.schema == boundary_payload_schema_record()
        assert old_request.schema != request.schema
        assert base_body == case.base_request_body
        assert boundary_body == case.boundary_request_body
        assert canonical_json_sha256(base_body) == case.base_request_sha256
        assert canonical_json_sha256(boundary_body) == (
            case.boundary_request_sha256
        )
        assert set(base_body) == set(boundary_body)
        for key in set(base_body) - {"messages"}:
            assert base_body[key] == boundary_body[key]
        assert base_body["messages"][0] == boundary_body["messages"][0]
        assert base_body["messages"][1]["content"] == old_request.user
        assert boundary_body["messages"][1]["content"] == expected_user
        assert base_body["seed"] == boundary_body["seed"] == (
            case.effective_seed
        )
        assert '"schema_version":"typed-claim-lens-v2.1"' in (
            expected_user[:old_start]
        )
        assert BOUNDARY_LENS_VERSION not in expected_user[:old_start]
        for forbidden in (
            "decomposition-semantic-feedback-v1",
            "decomposition-semantic-feedback-v2",
            "repair_constraint",
            "Revise the failed domains once",
            '"failed_domains"',
        ):
            assert forbidden not in expected_user


def test_prepare_reads_no_key_and_makes_no_http(monkeypatch, source_chain):
    e26_authorization, e27_authorization, _, _ = source_chain
    calls = _install_responses(monkeypatch, [])
    monkeypatch.delenv("HF_TOKEN", raising=False)

    result = prepare_boundary_decomposition_probe(
        source_run_path=RUN_PATH,
        source_diagnostic_path=DIAGNOSTIC_PATH,
        e24_artifact_path=E24_PATH,
        e25_artifact_path=E25_PATH,
        e26_artifact_path=E26_PATH,
        e27_artifact_path=E27_PATH,
        e26_authorization=e26_authorization,
        e27_authorization=e27_authorization,
        expected_e26_artifact_sha256=E26_SHA256,
        expected_e27_artifact_sha256=E27_SHA256,
        problem_source=_source(),
        planned_output_path=E28_OUTPUT,
    )

    result.validate_trusted()
    assert calls == []


def test_strict_reader_rejects_numeric_string_coercion_in_e28_artifact(prepared):
    assert file_sha256(E28_INVALID_PATH) == E28_INVALID_SHA256
    authorization = _authorization_from_contract(
        BoundaryDecompositionProbeAuthorization,
        "E28-v2-boundary-decompose33",
        prepared.contract,
    )

    with pytest.raises(
        BoundaryDecompositionProbeValidationError,
        match="failed shared transcript validation",
    ):
        read_boundary_decomposition_probe_artifact(
            E28_INVALID_PATH,
            prepared=prepared,
            authorization=authorization,
        )


def test_every_authorization_binding_precedes_output_key_and_http(
    monkeypatch, prepared
):
    final_path = ROOT / E28_OUTPUT
    partial_path = ROOT / f"{E28_OUTPUT}.partial"
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
            run_boundary_decomposition_probe(
                output_path=E28_OUTPUT,
                prepared=prepared,
                generator_client=_generator_client(prepared),
                authorization=bad,
            )
        checked.append(field_name)

    assert checked
    assert calls == []
    assert _path_snapshot(final_path) == final_before
    assert _path_snapshot(partial_path) == partial_before


@pytest.fixture(scope="module")
def executed_probe(tmp_path_factory, source_chain):
    e26_authorization, e27_authorization, _, _ = source_chain
    directory = tmp_path_factory.mktemp("boundary-probe")
    output_path = directory / "probe.jsonl"
    local_prepared = prepare_boundary_decomposition_probe(
        source_run_path=RUN_PATH,
        source_diagnostic_path=DIAGNOSTIC_PATH,
        e24_artifact_path=E24_PATH,
        e25_artifact_path=E25_PATH,
        e26_artifact_path=E26_PATH,
        e27_artifact_path=E27_PATH,
        e26_authorization=e26_authorization,
        e27_authorization=e27_authorization,
        expected_e26_artifact_sha256=E26_SHA256,
        expected_e27_artifact_sha256=E27_SHA256,
        problem_source=_source(),
        planned_output_path=output_path,
    )
    responses = []
    for case in local_prepared.cases:
        if case.index == 0:
            responses.append(_Response('{"domains":["x < 4","x >= 4"]}'))
        if case.index == 1:
            content = '{"boundaries":["4","3"]}'
        elif case.index == 2:
            content = '{"boundaries":["10"]}'
        else:
            content = _valid_content(case)
        responses.append(_Response(content))
    monkeypatch = pytest.MonkeyPatch()
    calls = _install_responses(monkeypatch, responses)
    client = _generator_client(local_prepared)
    try:
        result = run_boundary_decomposition_probe(
            output_path=output_path,
            prepared=local_prepared,
            generator_client=client,
            authorization=_authorization(local_prepared),
        )
    finally:
        monkeypatch.undo()
    return local_prepared, result, output_path, calls, client


def test_stubbed_probe_has_33_calls_strict_stages_and_all_gates(executed_probe):
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
    assert not output_path.with_name("probe.jsonl.partial").exists()

    call_offset = 0
    for case in prepared.cases:
        initial = calls[call_offset]["body"]
        assert initial == case.boundary_request_body
        assert initial["seed"] == case.effective_seed
        call_offset += 1
        if case.index == 0:
            retry = calls[call_offset]["body"]
            assert retry["seed"] == case.effective_seed + 1
            assert len(retry["messages"]) == len(initial["messages"]) + 2
            call_offset += 1
    assert call_offset == 34

    records = _artifact_records(output_path)
    schemas = Counter(record["schema_version"] for record in records)
    assert schemas == {
        BOUNDARY_HEADER_VERSION: 1,
        DIAGNOSTIC_ATTEMPT_VERSION: 34,
        DIAGNOSTIC_CALL_VERSION: 33,
        BOUNDARY_CASE_VERSION: 33,
        BOUNDARY_SUMMARY_VERSION: 1,
    }
    attempts = [
        record
        for record in records
        if record["schema_version"] == DIAGNOSTIC_ATTEMPT_VERSION
    ]
    cases = [
        record
        for record in records
        if record["schema_version"] == BOUNDARY_CASE_VERSION
    ]
    summary = records[-1]
    authorization = _authorization(prepared)
    assert read_boundary_decomposition_probe_artifact(
        output_path,
        prepared=prepared,
        authorization=authorization,
    ) == summary

    for case, record in zip(prepared.cases, cases, strict=True):
        assert {
            key: record[key] for key in case.manifest_record()
        } == case.manifest_record()
        assert record["wire"] == {
            "usable": True,
            "terminal_outcome": "transport_accepted",
            "final_attempt_outcome": "accepted",
            "error_type": None,
        }
        if case.index == 1:
            assert record["decode"] == {
                "status": "payload_error",
                "error_type": "BoundaryPayloadError",
                "payload": None,
            }
            assert record["construction"] is None
            assert record["corrected_application"] is None
        else:
            assert record["decode"]["status"] == "decoded"
            assert record["decode"]["error_type"] is None
            assert record["construction"] is not None
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

    first_call_id = cases[0]["boundary_call_record_id"].split(":")[1]
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
    assert summary["typed_decoded_count"] == 32
    assert summary["symbolic_applied_count"] == 31
    assert summary["applied_problem_count"] == 5
    assert set(summary["applied_problem_ids"]) == set(ORACLE_BOUNDARIES)
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


def test_typed_gate_includes_strict_parent_interior_validation(
    monkeypatch, tmp_path, source_chain
):
    output_path = tmp_path / "parent-interior.jsonl"
    prepared = _prepare_for_output(source_chain, output_path)
    targets = [
        case
        for case in prepared.cases
        if case.problem_id == "cw1-2640597e2425"
    ][:2]
    invalid_boundaries = {
        targets[0].index: "12",
        targets[1].index: "13",
    }
    for boundary in invalid_boundaries.values():
        assert decode_boundary_payload(
            {"boundaries": [boundary]}
        ).boundaries == (boundary,)
    responses = [
        _Response(
            json.dumps(
                {
                    "boundaries": (
                        [invalid_boundaries[case.index]]
                        if case.index in invalid_boundaries
                        else ORACLE_BOUNDARIES[case.problem_id]
                    )
                },
                separators=(",", ":"),
            )
        )
        for case in prepared.cases
    ]
    calls = _install_responses(monkeypatch, responses)

    result = run_boundary_decomposition_probe(
        output_path=output_path,
        prepared=prepared,
        generator_client=_generator_client(prepared),
        authorization=_authorization(prepared),
    )

    records = _artifact_records(output_path)
    cases = [
        record
        for record in records
        if record["schema_version"] == BOUNDARY_CASE_VERSION
    ]
    target_records = [
        record for record in cases if record["index"] in invalid_boundaries
    ]
    summary = records[-1]
    assert len(calls) == 33
    assert result.wire_usable_count == 33
    assert result.decoded_count == 31
    assert result.symbolic_applied_count == 31
    assert all(record["wire"]["usable"] is True for record in target_records)
    assert all(record["construction"] is None for record in target_records)
    assert all(
        record["corrected_application"] is None for record in target_records
    )
    assert summary["typed_decoded_count"] == 31
    assert summary["typed_decoded_count"] == sum(
        record["construction"] is not None for record in cases
    )
    assert summary["symbolic_applied_count"] == 31


@pytest.mark.parametrize(
    "reason_code",
    [
        "unsupported_domain",
        "malformed_scalar",
        "decomposition_domain_overlap",
        "decomposition_domain_not_exhaustive",
    ],
)
def test_trusted_topology_fault_aborts_publication(
    monkeypatch, tmp_path, source_chain, reason_code
):
    output_path = tmp_path / f"implementation-fault-{reason_code}.jsonl"
    prepared = _prepare_for_output(source_chain, output_path)
    score = boundary_probe_module._score_boundary_draft

    def inject_fault(prepared_probe, case, draft):
        application = deepcopy(score(prepared_probe, case, draft))
        application["application_status"] = "algebra_rejected"
        application["reason_code"] = reason_code
        return application

    monkeypatch.setattr(
        boundary_probe_module, "_score_boundary_draft", inject_fault
    )
    calls = _install_responses(
        monkeypatch, [_Response(_valid_content(prepared.cases[0]))]
    )

    with pytest.raises(
        BoundaryDecompositionProbeValidationError,
        match="implementation fault",
    ):
        run_boundary_decomposition_probe(
            output_path=output_path,
            prepared=prepared,
            generator_client=_generator_client(prepared),
            authorization=_authorization(prepared),
        )

    partial_path = output_path.with_name(f"{output_path.name}.partial")
    assert len(calls) == 1
    assert not output_path.exists()
    assert partial_path.exists()
    with pytest.raises(BoundaryDecompositionProbeValidationError):
        read_boundary_decomposition_probe_artifact(
            partial_path,
            prepared=prepared,
            authorization=_authorization(prepared),
        )


def test_child_not_atomic_is_a_publishable_generator_rejection(executed_probe):
    prepared, _, output_path, _, _ = executed_probe
    records = _artifact_records(output_path)
    case = next(
        record
        for record in records
        if record["schema_version"] == BOUNDARY_CASE_VERSION
        and record["index"] == prepared.cases[2].index
    )

    assert output_path.exists()
    assert case["construction"] is not None
    assert case["corrected_application"]["application_status"] == (
        "algebra_rejected"
    )
    assert case["corrected_application"]["reason_code"] == "child_not_atomic"
    assert records[-1]["symbolic_application_counts"]["algebra_rejected"] == 1


def test_five_oracle_boundaries_construct_apply_and_preserve_source_receipts(
    prepared,
):
    expected_domains = {
        "cw1-ccdf9e1567dc": ["x < 4", "x >= 4"],
        "cw1-2640597e2425": ["x < 6", "6 <= x < 12"],
        "cw1-52e3b6b3ec25": ["x < 2", "x >= 2"],
        "cw1-ddad3be834f4": ["x < 4", "x >= 4"],
        "cw1-4ba0fc91c9e8": ["x < -3", "x >= -3"],
    }
    oracle_records = validate_boundary_oracle_cases(prepared)
    assert [record["problem_id"] for record in oracle_records] == list(
        ORACLE_BOUNDARIES
    )
    assert [record["boundaries"] for record in oracle_records] == list(
        ORACLE_BOUNDARIES.values()
    )
    assert all(
        record["corrected_application"]["application_status"] == "applied"
        for record in oracle_records
    )

    representatives = {}
    for case in prepared.cases:
        representatives.setdefault(case.problem_id, case)
    for problem_id, case in representatives.items():
        state_before = case.state
        graph_before = case.state.graph
        payload = decode_boundary_payload({
            "boundaries": ORACLE_BOUNDARIES[problem_id]
        })
        draft = construct_boundary_decomposition(case.state, case.action, payload)
        assert [child.domain for child in draft.child_obligations] == (
            expected_domains[problem_id]
        )
        assert case.state == state_before
        assert case.state.graph == graph_before


def test_strict_reader_rejects_request_payload_construction_and_summary_tampering(
    tmp_path, executed_probe
):
    prepared, _, output_path, _, _ = executed_probe
    authorization = _authorization(prepared)
    with pytest.raises(TypeError, match="prepared"):
        read_boundary_decomposition_probe_artifact(
            output_path, authorization=authorization
        )
    with pytest.raises(TypeError, match="authorization"):
        read_boundary_decomposition_probe_artifact(
            output_path, prepared=prepared
        )
    original = _artifact_records(output_path)

    def preregistration_mutation(records):
        records[0]["preregistration_id"] = "E28-not-authorized"

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
        attempt["raw_content"] = '{"boundaries":["0"]}'

    def manifest_mutation(records):
        case = next(
            record
            for record in records
            if record["schema_version"] == BOUNDARY_CASE_VERSION
        )
        case["boundary_request_sha256"] = "f" * 64

    def index_bool_mutation(records):
        case = next(
            record
            for record in records
            if record["schema_version"] == BOUNDARY_CASE_VERSION
        )
        assert case["index"] == 0
        case["index"] = False

    def effective_seed_bool_mutation(records):
        case = next(
            record
            for record in records
            if record["schema_version"] == BOUNDARY_CASE_VERSION
        )
        assert case["effective_seed"] == 0
        case["effective_seed"] = False

    def wire_usable_int_mutation(records):
        case = next(
            record
            for record in records
            if record["schema_version"] == BOUNDARY_CASE_VERSION
        )
        assert case["wire"]["usable"] is True
        case["wire"]["usable"] = 1

    def wire_error_type_mutation(records):
        case = next(
            record
            for record in records
            if record["schema_version"] == BOUNDARY_CASE_VERSION
        )
        assert case["wire"]["error_type"] is None
        case["wire"]["error_type"] = "LLMResponseError"

    def payload_mutation(records):
        case = next(
            record
            for record in records
            if record["schema_version"] == BOUNDARY_CASE_VERSION
            and record["decode"]["status"] == "decoded"
        )
        case["decode"]["payload"]["boundaries"] = ["0"]

    def construction_mutation(records):
        case = next(
            record
            for record in records
            if record["schema_version"] == BOUNDARY_CASE_VERSION
            and record["construction"] is not None
        )
        case["construction"]["constructed_domains"][0] += " "

    def score_mutation(records):
        case = next(
            record
            for record in records
            if record["schema_version"] == BOUNDARY_CASE_VERSION
            and record["corrected_application"] is not None
        )
        case["corrected_application"]["coverage"]["context_digest"] = (
            "f" * 64
        )

    def summary_mutation(records):
        records[-1]["typed_decoded_count"] -= 1

    def unsupported_schema_mutation(records):
        records.insert(-1, {"schema_version": "controller-transition-v2"})

    for name, mutate in (
        ("preregistration", preregistration_mutation),
        ("request", request_mutation),
        ("raw", raw_projection_mutation),
        ("manifest", manifest_mutation),
        ("index-bool", index_bool_mutation),
        ("effective-seed-bool", effective_seed_bool_mutation),
        ("wire-usable-int", wire_usable_int_mutation),
        ("wire-error-type", wire_error_type_mutation),
        ("payload", payload_mutation),
        ("construction", construction_mutation),
        ("score", score_mutation),
        ("summary", summary_mutation),
        ("schema", unsupported_schema_mutation),
    ):
        records = deepcopy(original)
        mutate(records)
        path = tmp_path / f"{name}.jsonl"
        _write_records(path, records)
        with pytest.raises(BoundaryDecompositionProbeValidationError):
            read_boundary_decomposition_probe_artifact(
                path,
                prepared=prepared,
                authorization=authorization,
            )


@pytest.fixture(scope="module")
def guarded_probe(tmp_path_factory, source_chain):
    e26_authorization, e27_authorization, _, _ = source_chain
    directory = tmp_path_factory.mktemp("boundary-probe-guard")
    output_path = directory / "guard.jsonl"
    local_prepared = prepare_boundary_decomposition_probe(
        source_run_path=RUN_PATH,
        source_diagnostic_path=DIAGNOSTIC_PATH,
        e24_artifact_path=E24_PATH,
        e25_artifact_path=E25_PATH,
        e26_artifact_path=E26_PATH,
        e27_artifact_path=E27_PATH,
        e26_authorization=e26_authorization,
        e27_authorization=e27_authorization,
        expected_e26_artifact_sha256=E26_SHA256,
        expected_e27_artifact_sha256=E27_SHA256,
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
        run_boundary_decomposition_probe(
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
        run_boundary_decomposition_probe(
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
        run_boundary_decomposition_probe(
            output_path=output_path,
            prepared=prepared,
            generator_client=_generator_client(prepared),
            authorization=_authorization(prepared),
        )
    assert len(interrupted_calls) == 1
    assert not output_path.exists()
    assert partial_path.exists()
    with pytest.raises(BoundaryDecompositionProbeValidationError):
        read_boundary_decomposition_probe_artifact(
            partial_path,
            prepared=prepared,
            authorization=_authorization(prepared),
        )


def test_cli_help_exits_before_preparation(monkeypatch):
    monkeypatch.setattr(
        probe_cli,
        "prepare_boundary_decomposition_probe",
        lambda **kwargs: pytest.fail("help attempted preparation"),
    )

    with pytest.raises(SystemExit) as error:
        probe_cli.main(["--help"])

    assert error.value.code == 0


def test_cli_print_contract_binds_sealed_chain_without_key_http_or_write(
    monkeypatch, capsys, prepared, source_chain
):
    final_path = ROOT / E28_OUTPUT
    partial_path = ROOT / f"{E28_OUTPUT}.partial"
    final_before = _path_snapshot(final_path)
    partial_before = _path_snapshot(partial_path)
    calls = _install_responses(monkeypatch, [])
    preparation_kwargs = {}

    def fake_prepare(**kwargs):
        preparation_kwargs.update(kwargs)
        return prepared

    monkeypatch.setattr(
        probe_cli, "prepare_boundary_decomposition_probe", fake_prepare
    )
    monkeypatch.setattr(
        probe_cli,
        "_secret",
        lambda *args: pytest.fail("print mode attempted secret lookup"),
    )
    monkeypatch.setattr(
        probe_cli,
        "run_boundary_decomposition_probe",
        lambda **kwargs: pytest.fail("print mode attempted execution"),
    )

    exit_code = probe_cli.main([
        "--print-contract",
        *_cli_source_args(prepared),
    ])

    e26_authorization, e27_authorization, _, _ = source_chain
    assert exit_code == 0
    assert calls == []
    assert _path_snapshot(final_path) == final_before
    assert _path_snapshot(partial_path) == partial_before
    assert preparation_kwargs["e26_authorization"] == e26_authorization
    assert preparation_kwargs["e27_authorization"] == e27_authorization
    assert preparation_kwargs["expected_e26_artifact_sha256"] == E26_SHA256
    assert preparation_kwargs["expected_e27_artifact_sha256"] == E27_SHA256
    record = json.loads(capsys.readouterr().out)
    assert record["schema_version"] == (
        "decomposition-boundary-probe-preparation-v1"
    )
    assert record["boundary_contract"] == prepared.contract.to_record()
    assert record["boundary_contract_sha256"] == prepared.contract.sha256
    assert record["source_e27_contract"] == (
        prepared.source_e27_probe.contract.to_record()
    )
    assert record["source_e27_summary"] == prepared.source_e27_summary
    assert record["boundary_payload_schema"] == (
        boundary_payload_schema_record()
    )
    assert record["boundary_oracle_manifest"] == (
        validate_boundary_oracle_cases(prepared)
    )
    assert record["decompose_case_count"] == 33
    assert record["problem_count"] == 5


def test_cli_authorization_mismatch_precedes_key_output_and_http(
    monkeypatch, prepared
):
    final_path = ROOT / E28_OUTPUT
    partial_path = ROOT / f"{E28_OUTPUT}.partial"
    final_before = _path_snapshot(final_path)
    partial_before = _path_snapshot(partial_path)
    calls = _install_responses(monkeypatch, [])
    monkeypatch.setattr(
        probe_cli,
        "prepare_boundary_decomposition_probe",
        lambda **kwargs: prepared,
    )
    monkeypatch.setattr(
        probe_cli,
        "_secret",
        lambda *args: pytest.fail("mismatch attempted secret lookup"),
    )
    monkeypatch.setattr(
        probe_cli,
        "run_boundary_decomposition_probe",
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
