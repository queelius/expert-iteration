"""Exact-schema corrected replication of the sealed E28 boundary probe."""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..llm_client import (
    LLMResponseError,
    OPENAI_JSON_RETRY_INSTRUCTION,
    OpenAICompatibleLLMClient,
    SCHEMA_VALIDATION_LEGACY_COERCE_V1,
    SCHEMA_VALIDATION_STRICT_V1,
    classify_openai_compatible_json_content,
    project_openai_compatible_response,
)
from .boundary_probe import (
    BOUNDARY_EXPECTED_CASES,
    BOUNDARY_EXPECTED_PROBLEMS,
    BOUNDARY_EXPECTED_SOURCE_DECISIONS,
    BOUNDARY_INSTRUCTION,
    BOUNDARY_LENS_CONTRACT_VERSION,
    BOUNDARY_PROBE_CASE_VERSION,
    BOUNDARY_PROBE_SUMMARY_VERSION,
    BOUNDARY_REQUEST_BRIDGE_CONTRACT,
    BOUNDARY_SELECTION_CONTRACT,
    BoundaryDecompositionProbeAuthorization,
    BoundaryDecompositionProbeCase,
    BoundaryDecompositionProbeContract,
    BoundaryDecompositionProbeValidationError,
    PreparedBoundaryDecompositionProbe,
    _BOUNDARY_STRUCTURAL_FAULT_REASONS,
    _boundary_constructor_record,
    _boundary_constructor_sha256,
    _boundary_decode_and_score,
    _boundary_header,
    _boundary_instruction_sha256,
    _boundary_is_sha256,
    _boundary_path_record,
    _boundary_primary_thresholds,
    _boundary_summary,
    _build_boundary_oracle_manifest,
    build_boundary_request,
    boundary_payload_schema_record,
    boundary_payload_schema_sha256,
    prepare_boundary_decomposition_probe,
    validate_boundary_oracle_cases,
)
from .casework_source import CaseworkProblemSource
from .contracts import (
    CANONICAL_JSON_VERSION,
    ModelIdentity,
    ModelTransportContract,
    canonical_json_bytes,
    canonical_json_sha256,
)
from .diagnostics import (
    DIAGNOSTIC_ATTEMPT_VERSION,
    DIAGNOSTIC_CALL_VERSION,
    validate_attempt_record,
    validate_call_record,
)
from .key_aligned_repair import KeyAlignedRepairProbeAuthorization
from .lenses import LENS_CONTRACT_VERSION, llm_transport_contract
from .microprobe import file_sha256
from .semantic_repair import (
    SemanticRepairProbeAuthorization,
    SemanticRepairProbeValidationError,
    _SemanticRepairArtifactWriter,
    _bounded_error_type,
    _cost_basis_record,
    _read_canonical_records,
    _reported_cost_record,
    _validate_application_record,
    _validate_call_transcript,
)


EXACT_BOUNDARY_PROBE_CONTRACT_VERSION = (
    "decomposition-boundary-exact-schema-probe-contract-v1"
)
EXACT_BOUNDARY_PROBE_HEADER_VERSION = (
    "decomposition-boundary-exact-schema-probe-header-v1"
)
EXACT_BOUNDARY_PROBE_CASE_VERSION = (
    "decomposition-boundary-exact-schema-probe-case-v1"
)
EXACT_BOUNDARY_PROBE_SUMMARY_VERSION = (
    "decomposition-boundary-exact-schema-probe-summary-v1"
)
EXACT_BOUNDARY_SCHEMA_VALIDATION_VERSION = (
    "boundary-exact-schema-validation-v1"
)
EXACT_BOUNDARY_RETRY_POLICY_VERSION = "boundary-exact-schema-retry-policy-v1"
EXACT_BOUNDARY_RETRY_DIAGNOSTICS_VERSION = (
    "boundary-exact-schema-retry-diagnostics-v1"
)
E28_RAW_AUDIT_VERSION = "e28-no-coercion-raw-audit-v1"
E28_RAW_CONTROL_CASE_VERSION = "invalidated-e28-raw-control-case-v1"
E28_LEGACY_SUMMARY_STATUS = "invalidated-not-an-empirical-result-v1"
EXACT_BOUNDARY_INTERPRETATION_CONTRACT = (
    "protocol-corrected-generator-diagnostic-not-trajectory-reward-or-training-v1"
)
EXACT_BOUNDARY_REPLICATION_CONTRACT = (
    "same-e28-initial-requests-seeds-model-transport-constructor-scorer-v1"
)
EXACT_BOUNDARY_SOURCE_CONTROL_CONTRACT = (
    "invalidated-e28-raw-responses-audit-only-never-model-input-v1"
)

_EXPECTED_E28_RAW_ATTEMPTS = 33
_EXPECTED_E28_EXACT_ACCEPTED = 8
_EXPECTED_E28_EXACT_SCHEMA_ERRORS = 25
_EXPECTED_E28_NUMERIC_ITEMS = 42
_EXPECTED_E28_BOUNDARY_ITEMS = 71
_EXPECTED_E28_ALL_NUMBER_ARRAYS = 17
_EXPECTED_E28_MIXED_ARRAYS = 8
_EXPECTED_E28_ALL_STRING_ARRAYS = 8
_EXPECTED_E28_ACCEPTED_PROBLEMS = ("cw1-52e3b6b3ec25",)
_EXACT_BOUNDARY_PREPARED_TRUST_SEAL = object()


class ExactBoundaryProbeValidationError(ValueError):
    """Raised when the E29 contract, preparation, or artifact is invalid."""


def exact_boundary_schema_validation_record() -> dict[str, object]:
    """Return the strict no-coercion validation contract used by E29."""
    return {
        "schema_version": EXACT_BOUNDARY_SCHEMA_VALIDATION_VERSION,
        "client_schema_validation_mode": SCHEMA_VALIDATION_STRICT_V1,
        "wire_schema_sha256": boundary_payload_schema_sha256(),
        "type_contract": "recursive-exact-json-types-without-coercion-v1",
        "object_contract": "required-and-no-additional-properties-v1",
        "array_contract": "item-types-and-cardinality-enforced-v1",
    }


def exact_boundary_schema_validation_sha256() -> str:
    return canonical_json_sha256(exact_boundary_schema_validation_record())


def exact_boundary_retry_policy_record() -> dict[str, object]:
    """Return the exact correction-turn protocol newly exercised by E29."""
    return {
        "schema_version": EXACT_BOUNDARY_RETRY_POLICY_VERSION,
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


def exact_boundary_retry_policy_sha256() -> str:
    return canonical_json_sha256(exact_boundary_retry_policy_record())


@dataclass(frozen=True, slots=True)
class InvalidatedE28RawControlCase:
    index: int
    problem_id: str
    source_attempt_record_sha256: str
    initial_request_sha256: str
    effective_seed: int
    raw_content_sha256: str
    boundary_item_json_types: tuple[str, ...]
    exact_wire_outcome: str
    schema_version: str = E28_RAW_CONTROL_CASE_VERSION

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("E28 raw control index is invalid")
        if not isinstance(self.problem_id, str) or not self.problem_id:
            raise ValueError("E28 raw control problem id is invalid")
        for name in (
            "source_attempt_record_sha256",
            "initial_request_sha256",
            "raw_content_sha256",
        ):
            if not _boundary_is_sha256(getattr(self, name)):
                raise ValueError(f"E28 raw control {name} is invalid")
        if type(self.effective_seed) is not int:
            raise ValueError("E28 raw control seed is invalid")
        if not self.boundary_item_json_types or not set(
            self.boundary_item_json_types
        ) <= {"number", "string"}:
            raise ValueError("E28 raw control item types are invalid")
        if self.exact_wire_outcome not in {"accepted", "schema_error"}:
            raise ValueError("E28 raw control exact outcome is invalid")
        if self.schema_version != E28_RAW_CONTROL_CASE_VERSION:
            raise ValueError("E28 raw control schema is invalid")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "index": self.index,
            "problem_id": self.problem_id,
            "source_attempt_record_sha256": self.source_attempt_record_sha256,
            "initial_request_sha256": self.initial_request_sha256,
            "effective_seed": self.effective_seed,
            "raw_content_sha256": self.raw_content_sha256,
            "boundary_item_json_types": list(self.boundary_item_json_types),
            "exact_wire_outcome": self.exact_wire_outcome,
        }


@dataclass(frozen=True, slots=True)
class InvalidatedE28RawAudit:
    artifact_sha256: str
    contract_sha256: str
    raw_attempt_count: int
    all_number_array_count: int
    mixed_array_count: int
    all_string_array_count: int
    boundary_item_count: int
    numeric_item_count: int
    exact_schema_accepted_count: int
    numeric_schema_error_count: int
    accepted_problem_ids: tuple[str, ...]
    first_attempt_typed_count: int
    first_attempt_applied_count: int
    first_attempt_applied_problem_count: int
    required_retry_not_executed_count: int
    legacy_summary_status: str = E28_LEGACY_SUMMARY_STATUS
    schema_version: str = E28_RAW_AUDIT_VERSION

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_sha256": self.artifact_sha256,
            "contract_sha256": self.contract_sha256,
            "raw_attempt_count": self.raw_attempt_count,
            "all_number_array_count": self.all_number_array_count,
            "mixed_array_count": self.mixed_array_count,
            "all_string_array_count": self.all_string_array_count,
            "boundary_item_count": self.boundary_item_count,
            "numeric_item_count": self.numeric_item_count,
            "exact_schema_accepted_count": self.exact_schema_accepted_count,
            "numeric_schema_error_count": self.numeric_schema_error_count,
            "accepted_problem_ids": list(self.accepted_problem_ids),
            "first_attempt_typed_count": self.first_attempt_typed_count,
            "first_attempt_applied_count": self.first_attempt_applied_count,
            "first_attempt_applied_problem_count": (
                self.first_attempt_applied_problem_count
            ),
            "required_retry_not_executed_count": (
                self.required_retry_not_executed_count
            ),
            "legacy_summary_status": self.legacy_summary_status,
        }


@dataclass(frozen=True, slots=True)
class _InvalidatedE28Source:
    audit: InvalidatedE28RawAudit
    controls: tuple[InvalidatedE28RawControlCase, ...]
    case_record_sha256s: tuple[str, ...]
    header_sha256: str
    legacy_summary_sha256: str


def _number_or_string_type(value: object) -> str:
    if type(value) in {int, float}:
        return "number"
    if type(value) is str:
        return "string"
    raise ExactBoundaryProbeValidationError(
        "invalidated E28 raw boundary has an unexpected JSON type"
    )


def _validate_e28_response_projection(attempt: Mapping[str, object]) -> None:
    raw_body = attempt.get("raw_response_body")
    if not isinstance(raw_body, str):
        raise ExactBoundaryProbeValidationError(
            "invalidated E28 attempt lacks a provider response"
        )
    try:
        response = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise ExactBoundaryProbeValidationError(
            "invalidated E28 provider response is malformed"
        ) from exc
    projection = project_openai_compatible_response(response)
    if projection.provider_error is not None:
        raise ExactBoundaryProbeValidationError(
            "invalidated E28 provider response projection failed"
        )
    usage = attempt.get("usage")
    if type(usage) is not dict or type(usage.get("raw")) is not dict:
        raise ExactBoundaryProbeValidationError(
            "invalidated E28 provider usage is missing"
        )
    projected_usage = (
        None
        if projection.raw_usage_json is None
        else json.loads(projection.raw_usage_json)
    )
    if (
        projection.raw_content != attempt.get("raw_content")
        or canonical_json_bytes(projected_usage)
        != canonical_json_bytes(usage["raw"])
        or projection.prompt_tokens != usage.get("prompt_tokens")
        or projection.completion_tokens != usage.get("completion_tokens")
        or projection.total_tokens != usage.get("total_tokens")
    ):
        raise ExactBoundaryProbeValidationError(
            "invalidated E28 provider response projection differs"
        )


def _audit_invalidated_e28_artifact(
    path: Path,
    *,
    expected_artifact_sha256: str,
    prepared: PreparedBoundaryDecompositionProbe,
    authorization: BoundaryDecompositionProbeAuthorization,
) -> _InvalidatedE28Source:
    if file_sha256(path) != expected_artifact_sha256:
        raise ExactBoundaryProbeValidationError(
            "invalidated E28 artifact digest differs"
        )
    prepared.validate_trusted()
    authorization.validate(prepared.contract)
    records = _read_canonical_records(path)
    expected_count = 2 + 3 * BOUNDARY_EXPECTED_CASES
    if len(records) != expected_count:
        raise ExactBoundaryProbeValidationError(
            "invalidated E28 artifact record count differs"
        )
    expected_header = _boundary_header(prepared, authorization)
    if canonical_json_bytes(records[0]) != canonical_json_bytes(expected_header):
        raise ExactBoundaryProbeValidationError(
            "invalidated E28 artifact header differs"
        )
    legacy_summary = records[-1]
    if legacy_summary.get("schema_version") != BOUNDARY_PROBE_SUMMARY_VERSION:
        raise ExactBoundaryProbeValidationError(
            "invalidated E28 legacy summary is missing"
        )

    controls: list[InvalidatedE28RawControlCase] = []
    case_record_sha256s: list[str] = []
    exact_accepted_problem_ids: set[str] = set()
    array_shapes: Counter[str] = Counter()
    boundary_item_count = 0
    numeric_item_count = 0
    first_attempt_typed = 0
    first_attempt_applied = 0
    first_attempt_applied_problem_ids: set[str] = set()
    for index, source_case in enumerate(prepared.cases):
        attempt = records[1 + index * 3]
        call = records[2 + index * 3]
        case_record = records[3 + index * 3]
        if attempt.get("schema_version") != DIAGNOSTIC_ATTEMPT_VERSION:
            raise ExactBoundaryProbeValidationError(
                f"invalidated E28 case {index} attempt is missing"
            )
        if call.get("schema_version") != DIAGNOSTIC_CALL_VERSION:
            raise ExactBoundaryProbeValidationError(
                f"invalidated E28 case {index} call is missing"
            )
        if case_record.get("schema_version") != BOUNDARY_PROBE_CASE_VERSION:
            raise ExactBoundaryProbeValidationError(
                f"invalidated E28 case {index} record is missing"
            )
        validate_attempt_record(attempt)
        validate_call_record(call, attempts=[attempt])
        if (
            attempt.get("role") != "generator"
            or attempt.get("attempt_index") != 0
            or attempt.get("parse_retry_index") != 0
            or attempt.get("backoff_retry_index") != 0
            or attempt.get("outcome") != "accepted"
            or attempt.get("disposition") != "return"
            or call.get("final_outcome") != "transport_accepted"
            or call.get("final_attempt_outcome") != "accepted"
        ):
            raise ExactBoundaryProbeValidationError(
                f"invalidated E28 case {index} is not one historical attempt"
            )
        if canonical_json_bytes(attempt.get("request_body")) != (
            source_case.boundary_request_body_bytes
        ) or attempt.get("effective_seed") != source_case.effective_seed:
            raise ExactBoundaryProbeValidationError(
                f"invalidated E28 case {index} initial request differs"
            )
        if case_record.get("boundary_call_record_id") != call.get("record_id"):
            raise ExactBoundaryProbeValidationError(
                f"invalidated E28 case {index} call binding differs"
            )
        source_manifest = source_case.manifest_record()
        if canonical_json_bytes({
            key: case_record.get(key) for key in source_manifest
        }) != canonical_json_bytes(source_manifest):
            raise ExactBoundaryProbeValidationError(
                f"invalidated E28 case {index} source manifest differs"
            )
        _validate_e28_response_projection(attempt)
        raw_content = attempt.get("raw_content")
        if not isinstance(raw_content, str):
            raise ExactBoundaryProbeValidationError(
                f"invalidated E28 case {index} raw content is missing"
            )
        legacy_outcome, _ = classify_openai_compatible_json_content(
            raw_content,
            boundary_payload_schema_record(),
            schema_validation_mode=SCHEMA_VALIDATION_LEGACY_COERCE_V1,
        )
        exact_outcome, exact_payload = classify_openai_compatible_json_content(
            raw_content,
            boundary_payload_schema_record(),
            schema_validation_mode=SCHEMA_VALIDATION_STRICT_V1,
        )
        if legacy_outcome != "accepted" or exact_outcome not in {
            "accepted", "schema_error"
        }:
            raise ExactBoundaryProbeValidationError(
                f"invalidated E28 case {index} raw classification differs"
            )
        try:
            raw_payload = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise ExactBoundaryProbeValidationError(
                f"invalidated E28 case {index} raw JSON is not exact"
            ) from exc
        if type(raw_payload) is not dict or set(raw_payload) != {"boundaries"}:
            raise ExactBoundaryProbeValidationError(
                f"invalidated E28 case {index} raw object shape differs"
            )
        raw_boundaries = raw_payload["boundaries"]
        if type(raw_boundaries) is not list or not (1 <= len(raw_boundaries) <= 5):
            raise ExactBoundaryProbeValidationError(
                f"invalidated E28 case {index} raw array shape differs"
            )
        item_types = tuple(_number_or_string_type(item) for item in raw_boundaries)
        number_count = item_types.count("number")
        boundary_item_count += len(item_types)
        numeric_item_count += number_count
        if number_count == len(item_types):
            array_shapes["all_number"] += 1
        elif number_count == 0:
            array_shapes["all_string"] += 1
        else:
            array_shapes["mixed"] += 1
        if exact_outcome == "accepted":
            exact_accepted_problem_ids.add(source_case.problem_id)
            if exact_payload is None:
                raise ExactBoundaryProbeValidationError(
                    f"invalidated E28 case {index} accepted without payload"
                )
            if canonical_json_bytes(exact_payload) != canonical_json_bytes(
                raw_payload
            ):
                raise ExactBoundaryProbeValidationError(
                    f"invalidated E28 case {index} strict payload mutated"
                )
            decode, _, application = _boundary_decode_and_score(
                prepared, source_case, exact_payload
            )
            if decode["status"] == "decoded":
                first_attempt_typed += 1
            if application is not None and (
                application["application_status"] == "applied"
            ):
                first_attempt_applied += 1
                first_attempt_applied_problem_ids.add(source_case.problem_id)
        controls.append(InvalidatedE28RawControlCase(
            index=index,
            problem_id=source_case.problem_id,
            source_attempt_record_sha256=canonical_json_sha256(attempt),
            initial_request_sha256=canonical_json_sha256(
                attempt["request_body"]
            ),
            effective_seed=attempt["effective_seed"],
            raw_content_sha256=canonical_json_sha256({
                "raw_content": raw_content
            }),
            boundary_item_json_types=item_types,
            exact_wire_outcome=exact_outcome,
        ))
        case_record_sha256s.append(canonical_json_sha256(case_record))

    exact_accepted = sum(
        control.exact_wire_outcome == "accepted" for control in controls
    )
    exact_schema_errors = len(controls) - exact_accepted
    audit = InvalidatedE28RawAudit(
        artifact_sha256=expected_artifact_sha256,
        contract_sha256=prepared.contract.sha256,
        raw_attempt_count=len(controls),
        all_number_array_count=array_shapes["all_number"],
        mixed_array_count=array_shapes["mixed"],
        all_string_array_count=array_shapes["all_string"],
        boundary_item_count=boundary_item_count,
        numeric_item_count=numeric_item_count,
        exact_schema_accepted_count=exact_accepted,
        numeric_schema_error_count=exact_schema_errors,
        accepted_problem_ids=tuple(sorted(exact_accepted_problem_ids)),
        first_attempt_typed_count=first_attempt_typed,
        first_attempt_applied_count=first_attempt_applied,
        first_attempt_applied_problem_count=len(first_attempt_applied_problem_ids),
        required_retry_not_executed_count=exact_schema_errors,
    )
    expected_facts = (
        _EXPECTED_E28_RAW_ATTEMPTS,
        _EXPECTED_E28_ALL_NUMBER_ARRAYS,
        _EXPECTED_E28_MIXED_ARRAYS,
        _EXPECTED_E28_ALL_STRING_ARRAYS,
        _EXPECTED_E28_BOUNDARY_ITEMS,
        _EXPECTED_E28_NUMERIC_ITEMS,
        _EXPECTED_E28_EXACT_ACCEPTED,
        _EXPECTED_E28_EXACT_SCHEMA_ERRORS,
        _EXPECTED_E28_ACCEPTED_PROBLEMS,
        _EXPECTED_E28_EXACT_ACCEPTED,
        _EXPECTED_E28_EXACT_ACCEPTED,
        1,
        _EXPECTED_E28_EXACT_SCHEMA_ERRORS,
    )
    observed_facts = (
        audit.raw_attempt_count,
        audit.all_number_array_count,
        audit.mixed_array_count,
        audit.all_string_array_count,
        audit.boundary_item_count,
        audit.numeric_item_count,
        audit.exact_schema_accepted_count,
        audit.numeric_schema_error_count,
        audit.accepted_problem_ids,
        audit.first_attempt_typed_count,
        audit.first_attempt_applied_count,
        audit.first_attempt_applied_problem_count,
        audit.required_retry_not_executed_count,
    )
    if observed_facts != expected_facts:
        raise ExactBoundaryProbeValidationError(
            "invalidated E28 exact-schema audit facts differ"
        )
    return _InvalidatedE28Source(
        audit=audit,
        controls=tuple(controls),
        case_record_sha256s=tuple(case_record_sha256s),
        header_sha256=canonical_json_sha256(records[0]),
        legacy_summary_sha256=canonical_json_sha256(legacy_summary),
    )


@dataclass(frozen=True, slots=True)
class ExactBoundaryProbeCase:
    index: int
    source_case: BoundaryDecompositionProbeCase
    invalidated_e28_case_record_sha256: str
    invalidated_e28_raw_control: InvalidatedE28RawControlCase

    def __post_init__(self) -> None:
        if self.index != self.source_case.index:
            raise ValueError("exact boundary case index differs")
        if not _boundary_is_sha256(self.invalidated_e28_case_record_sha256):
            raise ValueError("exact boundary E28 case digest is invalid")
        if self.invalidated_e28_raw_control.index != self.index:
            raise ValueError("exact boundary raw control index differs")
        if self.invalidated_e28_raw_control.problem_id != self.problem_id:
            raise ValueError("exact boundary raw control problem differs")
        if self.invalidated_e28_raw_control.effective_seed != self.effective_seed:
            raise ValueError("exact boundary raw control seed differs")
        if self.invalidated_e28_raw_control.initial_request_sha256 != (
            self.source_case.boundary_request_sha256
        ):
            raise ValueError("exact boundary raw control request differs")

    @property
    def problem_id(self) -> str:
        return self.source_case.problem_id

    @property
    def effective_seed(self) -> int:
        return self.source_case.effective_seed

    @property
    def boundary_request_body(self) -> dict[str, object]:
        return self.source_case.boundary_request_body

    def manifest_record(self) -> dict[str, object]:
        return {
            **self.source_case.manifest_record(),
            "invalidated_e28_case_record_sha256": (
                self.invalidated_e28_case_record_sha256
            ),
            "invalidated_e28_raw_control_sha256": canonical_json_sha256(
                self.invalidated_e28_raw_control.to_record()
            ),
        }


_EXACT_CONTRACT_DIGEST_FIELDS = (
    "source_environment_contract_sha256",
    "request_environment_contract_sha256",
    "scoring_environment_contract_sha256",
    "corrected_profile_sha256",
    "e24_artifact_sha256",
    "e24_probe_contract_sha256",
    "e24_case_manifest_sha256",
    "e27_artifact_sha256",
    "e27_repair_contract_sha256",
    "e27_case_manifest_sha256",
    "e27_summary_sha256",
    "invalidated_e28_artifact_sha256",
    "invalidated_e28_probe_contract_sha256",
    "invalidated_e28_header_sha256",
    "invalidated_e28_case_manifest_sha256",
    "invalidated_e28_boundary_request_manifest_sha256",
    "invalidated_e28_legacy_summary_sha256",
    "invalidated_e28_raw_audit_sha256",
    "invalidated_e28_raw_control_manifest_sha256",
    "boundary_instruction_sha256",
    "boundary_payload_schema_sha256",
    "boundary_constructor_sha256",
    "boundary_oracle_manifest_sha256",
    "schema_validation_contract_sha256",
    "retry_policy_contract_sha256",
    "case_manifest_sha256",
    "base_request_manifest_sha256",
    "boundary_request_manifest_sha256",
)


@dataclass(frozen=True, slots=True)
class ExactBoundaryProbeContract:
    source_environment_contract_sha256: str
    request_environment_contract_sha256: str
    scoring_environment_contract_sha256: str
    corrected_profile_sha256: str
    e24_artifact_sha256: str
    e24_probe_contract_sha256: str
    e24_case_manifest_sha256: str
    e27_artifact_sha256: str
    e27_repair_contract_sha256: str
    e27_case_manifest_sha256: str
    e27_summary_sha256: str
    invalidated_e28_artifact_sha256: str
    invalidated_e28_probe_contract_sha256: str
    invalidated_e28_header_sha256: str
    invalidated_e28_case_manifest_sha256: str
    invalidated_e28_boundary_request_manifest_sha256: str
    invalidated_e28_legacy_summary_sha256: str
    invalidated_e28_raw_audit_sha256: str
    invalidated_e28_raw_control_manifest_sha256: str
    boundary_instruction_sha256: str
    boundary_payload_schema_sha256: str
    boundary_constructor_sha256: str
    boundary_oracle_manifest_sha256: str
    schema_validation_contract_sha256: str
    retry_policy_contract_sha256: str
    case_manifest_sha256: str
    base_request_manifest_sha256: str
    boundary_request_manifest_sha256: str
    generator: ModelIdentity
    generator_transport: ModelTransportContract
    generation_schema: str
    lens_contract: str
    generation_temperature: float
    generation_max_tokens: int
    generation_seed_policy: str
    generation_seed: int
    source_decision_count: int
    case_count: int
    problem_count: int
    planned_output_path: str
    schema_version: str = EXACT_BOUNDARY_PROBE_CONTRACT_VERSION
    boundary_lens_contract: str = BOUNDARY_LENS_CONTRACT_VERSION
    selection_contract: str = BOUNDARY_SELECTION_CONTRACT
    request_bridge_contract: str = BOUNDARY_REQUEST_BRIDGE_CONTRACT
    schema_validation_mode: str = SCHEMA_VALIDATION_STRICT_V1
    canonical_serializer: str = CANONICAL_JSON_VERSION
    replication_contract: str = EXACT_BOUNDARY_REPLICATION_CONTRACT
    source_control_contract: str = EXACT_BOUNDARY_SOURCE_CONTROL_CONTRACT
    interpretation_contract: str = EXACT_BOUNDARY_INTERPRETATION_CONTRACT

    def __post_init__(self) -> None:
        _validate_exact_boundary_contract(self)

    @property
    def generator_sha256(self) -> str:
        return canonical_json_sha256(self.generator.to_record())

    @property
    def generator_transport_sha256(self) -> str:
        return self.generator_transport.sha256

    def generation_settings_record(self) -> dict[str, object]:
        return {
            "generation_schema": self.generation_schema,
            "lens_contract": self.lens_contract,
            "generation_temperature": self.generation_temperature,
            "generation_max_tokens": self.generation_max_tokens,
            "generation_seed_policy": self.generation_seed_policy,
            "generation_seed": self.generation_seed,
        }

    @property
    def generation_settings_sha256(self) -> str:
        return canonical_json_sha256(self.generation_settings_record())

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.to_record())

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {"schema_version": self.schema_version}
        for name in _EXACT_CONTRACT_DIGEST_FIELDS:
            record[name] = getattr(self, name)
        record.update({
            "generator": self.generator.to_record(),
            "generator_sha256": self.generator_sha256,
            "generator_transport": self.generator_transport.to_record(),
            "generator_transport_sha256": self.generator_transport_sha256,
            "generation_settings": self.generation_settings_record(),
            "generation_settings_sha256": self.generation_settings_sha256,
            "boundary_instruction": BOUNDARY_INSTRUCTION,
            "boundary_constructor": _boundary_constructor_record(),
            "schema_validation": exact_boundary_schema_validation_record(),
            "retry_policy": exact_boundary_retry_policy_record(),
            "schema_validation_mode": self.schema_validation_mode,
            "source_decision_count": self.source_decision_count,
            "case_count": self.case_count,
            "problem_count": self.problem_count,
            "planned_output_path": self.planned_output_path,
            "primary_thresholds": _boundary_primary_thresholds(),
            "cost_basis": _cost_basis_record(),
            "boundary_lens_contract": self.boundary_lens_contract,
            "selection_contract": self.selection_contract,
            "request_bridge_contract": self.request_bridge_contract,
            "canonical_serializer": self.canonical_serializer,
            "replication_contract": self.replication_contract,
            "source_control_contract": self.source_control_contract,
            "interpretation_contract": self.interpretation_contract,
        })
        return record


def _validate_exact_boundary_contract(contract: ExactBoundaryProbeContract) -> None:
    if contract.schema_version != EXACT_BOUNDARY_PROBE_CONTRACT_VERSION:
        raise ValueError("exact boundary contract schema is invalid")
    for name in _EXACT_CONTRACT_DIGEST_FIELDS:
        if not _boundary_is_sha256(getattr(contract, name)):
            raise ValueError(f"exact boundary contract {name} is invalid")
    if contract.boundary_instruction_sha256 != _boundary_instruction_sha256():
        raise ValueError("exact boundary instruction digest is invalid")
    if contract.boundary_payload_schema_sha256 != boundary_payload_schema_sha256():
        raise ValueError("exact boundary schema digest is invalid")
    if contract.boundary_constructor_sha256 != _boundary_constructor_sha256():
        raise ValueError("exact boundary constructor digest is invalid")
    if contract.schema_validation_contract_sha256 != (
        exact_boundary_schema_validation_sha256()
    ) or contract.schema_validation_mode != SCHEMA_VALIDATION_STRICT_V1:
        raise ValueError("exact boundary validation contract is invalid")
    if not isinstance(contract.generator, ModelIdentity):
        raise ValueError("exact boundary generator is invalid")
    if not isinstance(contract.generator_transport, ModelTransportContract):
        raise ValueError("exact boundary generator transport is invalid")
    if (
        contract.retry_policy_contract_sha256
        != exact_boundary_retry_policy_sha256()
        or contract.generator_transport.parse_retry_budget != 1
        or contract.generator_transport.parse_retry_contract
        != "assistant-echo-seed-offset-v1"
    ):
        raise ValueError("exact boundary retry policy contract is invalid")
    if (
        contract.generator_transport.base_url is None
        or contract.generator_transport.guided_json
        or contract.generator_transport.model != contract.generator.model
    ):
        raise ValueError("exact boundary hosted transport is invalid")
    if contract.lens_contract != LENS_CONTRACT_VERSION:
        raise ValueError("exact boundary source lens is invalid")
    if contract.boundary_lens_contract != BOUNDARY_LENS_CONTRACT_VERSION:
        raise ValueError("exact boundary lens contract is invalid")
    if contract.source_decision_count != BOUNDARY_EXPECTED_SOURCE_DECISIONS:
        raise ValueError("exact boundary source decision count is invalid")
    if contract.case_count != BOUNDARY_EXPECTED_CASES:
        raise ValueError("exact boundary case count is invalid")
    if contract.problem_count != BOUNDARY_EXPECTED_PROBLEMS:
        raise ValueError("exact boundary problem count is invalid")
    if contract.planned_output_path != _boundary_path_record(
        contract.planned_output_path
    ):
        raise ValueError("exact boundary output path is invalid")
    if (
        contract.selection_contract != BOUNDARY_SELECTION_CONTRACT
        or contract.request_bridge_contract != BOUNDARY_REQUEST_BRIDGE_CONTRACT
        or contract.canonical_serializer != CANONICAL_JSON_VERSION
        or contract.replication_contract != EXACT_BOUNDARY_REPLICATION_CONTRACT
        or contract.source_control_contract != EXACT_BOUNDARY_SOURCE_CONTROL_CONTRACT
        or contract.interpretation_contract
        != EXACT_BOUNDARY_INTERPRETATION_CONTRACT
    ):
        raise ValueError("exact boundary protocol contract is invalid")


@dataclass(frozen=True, slots=True)
class ExactBoundaryProbeAuthorization:
    preregistration_id: str
    expected_source_environment_contract_sha256: str
    expected_request_environment_contract_sha256: str
    expected_scoring_environment_contract_sha256: str
    expected_corrected_profile_sha256: str
    expected_e24_artifact_sha256: str
    expected_e24_probe_contract_sha256: str
    expected_e24_case_manifest_sha256: str
    expected_e27_artifact_sha256: str
    expected_e27_repair_contract_sha256: str
    expected_e27_case_manifest_sha256: str
    expected_e27_summary_sha256: str
    expected_invalidated_e28_artifact_sha256: str
    expected_invalidated_e28_probe_contract_sha256: str
    expected_invalidated_e28_header_sha256: str
    expected_invalidated_e28_case_manifest_sha256: str
    expected_invalidated_e28_boundary_request_manifest_sha256: str
    expected_invalidated_e28_legacy_summary_sha256: str
    expected_invalidated_e28_raw_audit_sha256: str
    expected_invalidated_e28_raw_control_manifest_sha256: str
    expected_boundary_instruction_sha256: str
    expected_boundary_payload_schema_sha256: str
    expected_boundary_constructor_sha256: str
    expected_boundary_oracle_manifest_sha256: str
    expected_schema_validation_contract_sha256: str
    expected_retry_policy_contract_sha256: str
    expected_case_manifest_sha256: str
    expected_base_request_manifest_sha256: str
    expected_boundary_request_manifest_sha256: str
    expected_generator_sha256: str
    expected_generator_transport_sha256: str
    expected_generation_settings_sha256: str
    expected_contract_sha256: str
    expected_source_decision_count: int
    expected_case_count: int
    expected_problem_count: int
    expected_output_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.preregistration_id, str) or not (
            self.preregistration_id.strip()
        ):
            raise ValueError("exact boundary preregistration id is empty")
        for name in self.__dataclass_fields__:
            if name.startswith("expected_") and name.endswith("sha256"):
                if not _boundary_is_sha256(getattr(self, name)):
                    raise ValueError(
                        f"exact boundary authorization {name} is invalid"
                    )
        if self.expected_source_decision_count != BOUNDARY_EXPECTED_SOURCE_DECISIONS:
            raise ValueError("authorized exact source decision count is invalid")
        if self.expected_case_count != BOUNDARY_EXPECTED_CASES:
            raise ValueError("authorized exact case count is invalid")
        if self.expected_problem_count != BOUNDARY_EXPECTED_PROBLEMS:
            raise ValueError("authorized exact problem count is invalid")
        if self.expected_output_path != _boundary_path_record(
            self.expected_output_path
        ):
            raise ValueError("authorized exact boundary output path is invalid")

    def validate(self, contract: ExactBoundaryProbeContract) -> None:
        for name in self.__dataclass_fields__:
            if not name.startswith("expected_"):
                continue
            contract_name = name.removeprefix("expected_")
            if contract_name == "contract_sha256":
                observed = contract.sha256
            elif contract_name == "output_path":
                observed = contract.planned_output_path
            else:
                observed = getattr(contract, contract_name)
            if getattr(self, name) != observed:
                label = contract_name.replace("_", " ")
                raise ValueError(f"authorized exact boundary {label} differs")


@dataclass(frozen=True, slots=True)
class PreparedExactBoundaryProbe:
    contract: ExactBoundaryProbeContract
    source_e28_probe: PreparedBoundaryDecompositionProbe
    invalidated_e28_raw_audit: InvalidatedE28RawAudit
    invalidated_e28_raw_controls: tuple[InvalidatedE28RawControlCase, ...]
    oracle_manifest: tuple[Mapping[str, object], ...]
    cases: tuple[ExactBoundaryProbeCase, ...]
    _trust_seal: object = field(init=False, repr=False, compare=False, default=None)
    _trusted_binding: object = field(
        init=False, repr=False, compare=False, default=None
    )

    def validate_trusted(self) -> None:
        _validate_prepared_exact_boundary_probe(self)


@dataclass(frozen=True, slots=True)
class ExactBoundaryProbeRun:
    output_path: Path
    contract_sha256: str
    case_count: int
    wire_usable_count: int
    decoded_count: int
    symbolic_applied_count: int
    applied_problem_count: int


def _raw_control_manifest_sha256(
    controls: tuple[InvalidatedE28RawControlCase, ...],
) -> str:
    return canonical_json_sha256([control.to_record() for control in controls])


def _exact_case_manifest_sha256(
    cases: tuple[ExactBoundaryProbeCase, ...],
) -> str:
    return canonical_json_sha256([case.manifest_record() for case in cases])


def _prepare_exact_boundary_probe(
    *,
    source_run_path: str | Path,
    source_diagnostic_path: str | Path,
    e24_artifact_path: str | Path,
    e25_artifact_path: str | Path,
    e26_artifact_path: str | Path,
    e27_artifact_path: str | Path,
    e28_artifact_path: str | Path,
    e26_authorization: SemanticRepairProbeAuthorization,
    e27_authorization: KeyAlignedRepairProbeAuthorization,
    e28_authorization: BoundaryDecompositionProbeAuthorization,
    expected_e26_artifact_sha256: str,
    expected_e27_artifact_sha256: str,
    expected_e28_artifact_sha256: str,
    problem_source: CaseworkProblemSource,
    planned_output_path: str | Path,
) -> PreparedExactBoundaryProbe:
    if not isinstance(e28_authorization, BoundaryDecompositionProbeAuthorization):
        raise TypeError("exact boundary E28 authorization has an invalid type")
    if not _boundary_is_sha256(expected_e28_artifact_sha256):
        raise ValueError("expected invalidated E28 artifact digest is invalid")
    e28_path = Path(e28_artifact_path)
    if file_sha256(e28_path) != expected_e28_artifact_sha256:
        raise ExactBoundaryProbeValidationError(
            "invalidated E28 artifact digest differs"
        )
    source_probe = prepare_boundary_decomposition_probe(
        source_run_path=source_run_path,
        source_diagnostic_path=source_diagnostic_path,
        e24_artifact_path=e24_artifact_path,
        e25_artifact_path=e25_artifact_path,
        e26_artifact_path=e26_artifact_path,
        e27_artifact_path=e27_artifact_path,
        e26_authorization=e26_authorization,
        e27_authorization=e27_authorization,
        expected_e26_artifact_sha256=expected_e26_artifact_sha256,
        expected_e27_artifact_sha256=expected_e27_artifact_sha256,
        problem_source=problem_source,
        planned_output_path=e28_authorization.expected_output_path,
    )
    e28_authorization.validate(source_probe.contract)
    source = _audit_invalidated_e28_artifact(
        e28_path,
        expected_artifact_sha256=expected_e28_artifact_sha256,
        prepared=source_probe,
        authorization=e28_authorization,
    )
    cases = tuple(
        ExactBoundaryProbeCase(
            index=source_case.index,
            source_case=source_case,
            invalidated_e28_case_record_sha256=case_sha256,
            invalidated_e28_raw_control=control,
        )
        for source_case, case_sha256, control in zip(
            source_probe.cases,
            source.case_record_sha256s,
            source.controls,
            strict=True,
        )
    )
    source_contract = source_probe.contract
    contract = ExactBoundaryProbeContract(
        source_environment_contract_sha256=(
            source_contract.source_environment_contract_sha256
        ),
        request_environment_contract_sha256=(
            source_contract.request_environment_contract_sha256
        ),
        scoring_environment_contract_sha256=(
            source_contract.scoring_environment_contract_sha256
        ),
        corrected_profile_sha256=source_contract.corrected_profile_sha256,
        e24_artifact_sha256=source_contract.e24_artifact_sha256,
        e24_probe_contract_sha256=source_contract.e24_probe_contract_sha256,
        e24_case_manifest_sha256=source_contract.e24_case_manifest_sha256,
        e27_artifact_sha256=source_contract.e27_artifact_sha256,
        e27_repair_contract_sha256=source_contract.e27_repair_contract_sha256,
        e27_case_manifest_sha256=source_contract.e27_case_manifest_sha256,
        e27_summary_sha256=source_contract.e27_summary_sha256,
        invalidated_e28_artifact_sha256=expected_e28_artifact_sha256,
        invalidated_e28_probe_contract_sha256=source_contract.sha256,
        invalidated_e28_header_sha256=source.header_sha256,
        invalidated_e28_case_manifest_sha256=(
            source_contract.case_manifest_sha256
        ),
        invalidated_e28_boundary_request_manifest_sha256=(
            source_contract.boundary_request_manifest_sha256
        ),
        invalidated_e28_legacy_summary_sha256=source.legacy_summary_sha256,
        invalidated_e28_raw_audit_sha256=canonical_json_sha256(
            source.audit.to_record()
        ),
        invalidated_e28_raw_control_manifest_sha256=(
            _raw_control_manifest_sha256(source.controls)
        ),
        boundary_instruction_sha256=source_contract.boundary_instruction_sha256,
        boundary_payload_schema_sha256=(
            source_contract.boundary_payload_schema_sha256
        ),
        boundary_constructor_sha256=source_contract.boundary_constructor_sha256,
        boundary_oracle_manifest_sha256=(
            source_contract.boundary_oracle_manifest_sha256
        ),
        schema_validation_contract_sha256=(
            exact_boundary_schema_validation_sha256()
        ),
        retry_policy_contract_sha256=exact_boundary_retry_policy_sha256(),
        case_manifest_sha256=_exact_case_manifest_sha256(cases),
        base_request_manifest_sha256=source_contract.base_request_manifest_sha256,
        boundary_request_manifest_sha256=(
            source_contract.boundary_request_manifest_sha256
        ),
        generator=source_contract.generator,
        generator_transport=source_contract.generator_transport,
        generation_schema=source_contract.generation_schema,
        lens_contract=source_contract.lens_contract,
        generation_temperature=source_contract.generation_temperature,
        generation_max_tokens=source_contract.generation_max_tokens,
        generation_seed_policy=source_contract.generation_seed_policy,
        generation_seed=source_contract.generation_seed,
        source_decision_count=source_contract.source_decision_count,
        case_count=source_contract.case_count,
        problem_count=source_contract.problem_count,
        planned_output_path=_boundary_path_record(planned_output_path),
    )
    prepared = PreparedExactBoundaryProbe(
        contract=contract,
        source_e28_probe=source_probe,
        invalidated_e28_raw_audit=source.audit,
        invalidated_e28_raw_controls=source.controls,
        oracle_manifest=source_probe.oracle_manifest,
        cases=cases,
    )
    binding = (
        prepared.contract,
        prepared.source_e28_probe,
        prepared.invalidated_e28_raw_audit,
        prepared.invalidated_e28_raw_controls,
        prepared.oracle_manifest,
        prepared.cases,
    )
    object.__setattr__(
        prepared, "_trust_seal", _EXACT_BOUNDARY_PREPARED_TRUST_SEAL
    )
    object.__setattr__(prepared, "_trusted_binding", binding)
    prepared.validate_trusted()
    return prepared


def prepare_exact_boundary_probe(**kwargs: object) -> PreparedExactBoundaryProbe:
    try:
        return _prepare_exact_boundary_probe(**kwargs)
    except (
        BoundaryDecompositionProbeValidationError,
        SemanticRepairProbeValidationError,
    ) as exc:
        raise ExactBoundaryProbeValidationError(
            "exact boundary preparation failed strict source validation"
        ) from exc


def _validate_prepared_exact_boundary_probe(
    prepared: PreparedExactBoundaryProbe,
) -> None:
    binding = (
        prepared.contract,
        prepared.source_e28_probe,
        prepared.invalidated_e28_raw_audit,
        prepared.invalidated_e28_raw_controls,
        prepared.oracle_manifest,
        prepared.cases,
    )
    if (
        prepared._trust_seal is not _EXACT_BOUNDARY_PREPARED_TRUST_SEAL
        or prepared._trusted_binding != binding
    ):
        raise ExactBoundaryProbeValidationError(
            "exact boundary preparation did not come from strict replay"
        )
    prepared.source_e28_probe.validate_trusted()
    contract = prepared.contract
    source_contract = prepared.source_e28_probe.contract
    if source_contract.sha256 != contract.invalidated_e28_probe_contract_sha256:
        raise ExactBoundaryProbeValidationError(
            "prepared invalidated E28 contract differs"
        )
    if canonical_json_sha256(
        prepared.invalidated_e28_raw_audit.to_record()
    ) != contract.invalidated_e28_raw_audit_sha256:
        raise ExactBoundaryProbeValidationError(
            "prepared invalidated E28 raw audit differs"
        )
    if _raw_control_manifest_sha256(
        prepared.invalidated_e28_raw_controls
    ) != contract.invalidated_e28_raw_control_manifest_sha256:
        raise ExactBoundaryProbeValidationError(
            "prepared invalidated E28 raw controls differ"
        )
    if tuple(case.index for case in prepared.cases) != tuple(
        range(contract.case_count)
    ):
        raise ExactBoundaryProbeValidationError(
            "prepared exact boundary cases are not ordered"
        )
    if _exact_case_manifest_sha256(prepared.cases) != contract.case_manifest_sha256:
        raise ExactBoundaryProbeValidationError(
            "prepared exact boundary case manifest differs"
        )
    if tuple(
        case.invalidated_e28_raw_control for case in prepared.cases
    ) != prepared.invalidated_e28_raw_controls:
        raise ExactBoundaryProbeValidationError(
            "prepared exact boundary case controls differ"
        )
    observed_oracle = _build_boundary_oracle_manifest(
        prepared.source_e28_probe.cases,
        prepared.source_e28_probe.source_e27_probe,
    )
    if (
        observed_oracle != prepared.oracle_manifest
        or canonical_json_sha256(list(observed_oracle))
        != contract.boundary_oracle_manifest_sha256
    ):
        raise ExactBoundaryProbeValidationError(
            "prepared exact boundary oracle differs"
        )
    for case in prepared.cases:
        _, _, boundary_body = build_boundary_request(
            prepared.source_e28_probe, case.source_case
        )
        if canonical_json_bytes(boundary_body) != (
            case.source_case.boundary_request_body_bytes
        ):
            raise ExactBoundaryProbeValidationError(
                "prepared exact boundary request changed"
            )
        if canonical_json_sha256(boundary_body) != (
            case.invalidated_e28_raw_control.initial_request_sha256
        ):
            raise ExactBoundaryProbeValidationError(
                "prepared exact boundary request differs from E28 control"
            )


def validate_exact_boundary_oracle_cases(
    prepared: PreparedExactBoundaryProbe,
) -> list[dict[str, object]]:
    if not isinstance(prepared, PreparedExactBoundaryProbe):
        raise TypeError("prepared exact boundary oracle input has invalid type")
    prepared.validate_trusted()
    observed = validate_boundary_oracle_cases(prepared.source_e28_probe)
    if canonical_json_bytes(observed) != canonical_json_bytes(
        list(prepared.oracle_manifest)
    ):
        raise ExactBoundaryProbeValidationError(
            "exact boundary oracle changed after preparation"
        )
    return observed


def _exact_boundary_header(
    prepared: PreparedExactBoundaryProbe,
    authorization: ExactBoundaryProbeAuthorization,
) -> dict[str, object]:
    contract = prepared.contract
    return {
        "schema_version": EXACT_BOUNDARY_PROBE_HEADER_VERSION,
        "canonical_serializer": CANONICAL_JSON_VERSION,
        "preregistration_id": authorization.preregistration_id,
        "probe_contract": contract.to_record(),
        "probe_contract_sha256": contract.sha256,
        "invalidated_e28_artifact_sha256": (
            contract.invalidated_e28_artifact_sha256
        ),
        "invalidated_e28_probe_contract_sha256": (
            contract.invalidated_e28_probe_contract_sha256
        ),
        "invalidated_e28_header_sha256": (
            contract.invalidated_e28_header_sha256
        ),
        "invalidated_e28_legacy_summary_sha256": (
            contract.invalidated_e28_legacy_summary_sha256
        ),
        "invalidated_e28_raw_audit": (
            prepared.invalidated_e28_raw_audit.to_record()
        ),
        "invalidated_e28_raw_audit_sha256": (
            contract.invalidated_e28_raw_audit_sha256
        ),
        "invalidated_e28_raw_control_manifest_sha256": (
            contract.invalidated_e28_raw_control_manifest_sha256
        ),
        "schema_validation": exact_boundary_schema_validation_record(),
        "schema_validation_contract_sha256": (
            contract.schema_validation_contract_sha256
        ),
        "retry_policy": exact_boundary_retry_policy_record(),
        "retry_policy_contract_sha256": (
            contract.retry_policy_contract_sha256
        ),
        "case_manifest_sha256": contract.case_manifest_sha256,
        "boundary_request_manifest_sha256": (
            contract.boundary_request_manifest_sha256
        ),
        "boundary_constructor_sha256": contract.boundary_constructor_sha256,
        "boundary_oracle_manifest_sha256": (
            contract.boundary_oracle_manifest_sha256
        ),
    }


def _current_first_array_shape(
    raw_content: object,
) -> tuple[str, int, int]:
    if not isinstance(raw_content, str):
        return "unclassified", 0, 0
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError:
        return "unclassified", 0, 0
    if type(payload) is not dict or set(payload) != {"boundaries"}:
        return "unclassified", 0, 0
    boundaries = payload["boundaries"]
    if type(boundaries) is not list or not boundaries:
        return "unclassified", 0, 0
    try:
        item_types = tuple(_number_or_string_type(item) for item in boundaries)
    except ExactBoundaryProbeValidationError:
        return "unclassified", 0, 0
    numeric = item_types.count("number")
    if numeric == len(item_types):
        shape = "all_number"
    elif numeric == 0:
        shape = "all_string"
    else:
        shape = "mixed"
    return shape, len(item_types), numeric


def _exact_boundary_retry_diagnostics(
    prepared: PreparedExactBoundaryProbe,
    attempt_groups: list[list[dict[str, object]]],
    calls: list[dict[str, object]],
) -> dict[str, object]:
    if not (
        len(attempt_groups) == len(calls) == len(prepared.cases)
    ):
        raise ExactBoundaryProbeValidationError(
            "exact boundary retry diagnostic cardinality differs"
        )
    first_outcomes: Counter[str] = Counter()
    first_shapes: Counter[str] = Counter()
    schema_retry_repaired = 0
    schema_retry_exhausted = 0
    boundary_item_count = 0
    numeric_item_count = 0
    raw_hash_matches = 0
    for case, attempts, call in zip(
        prepared.cases, attempt_groups, calls, strict=True
    ):
        if not attempts:
            raise ExactBoundaryProbeValidationError(
                "exact boundary retry diagnostic call has no attempt"
            )
        first = attempts[0]
        first_outcomes[first["outcome"]] += 1
        shape, items, numeric = _current_first_array_shape(
            first.get("raw_content")
        )
        first_shapes[shape] += 1
        boundary_item_count += items
        numeric_item_count += numeric
        if canonical_json_sha256({
            "raw_content": first.get("raw_content")
        }) == case.invalidated_e28_raw_control.raw_content_sha256:
            raw_hash_matches += 1
        if first["outcome"] == "schema_error":
            if call["final_outcome"] == "transport_accepted":
                schema_retry_repaired += 1
            else:
                schema_retry_exhausted += 1
    return {
        "schema_version": EXACT_BOUNDARY_RETRY_DIAGNOSTICS_VERSION,
        "case_count": len(prepared.cases),
        "first_attempt_accepted_count": first_outcomes["accepted"],
        "first_attempt_schema_error_count": first_outcomes["schema_error"],
        "first_attempt_other_outcome_count": (
            len(prepared.cases)
            - first_outcomes["accepted"]
            - first_outcomes["schema_error"]
        ),
        "schema_retry_case_count": first_outcomes["schema_error"],
        "schema_retry_repaired_count": schema_retry_repaired,
        "schema_retry_exhausted_count": schema_retry_exhausted,
        "first_array_all_number_count": first_shapes["all_number"],
        "first_array_mixed_count": first_shapes["mixed"],
        "first_array_all_string_count": first_shapes["all_string"],
        "first_array_unclassified_count": first_shapes["unclassified"],
        "first_boundary_item_count": boundary_item_count,
        "first_numeric_item_count": numeric_item_count,
        "e28_first_raw_content_hash_match_count": raw_hash_matches,
    }


def _exact_boundary_summary(
    contract: ExactBoundaryProbeContract,
    case_records: list[dict[str, object]],
    *,
    attempt_count: int,
    call_count: int,
    attempt_outcome_counts: Mapping[str, int],
    reported_usage: Mapping[str, int],
    retry_diagnostics: Mapping[str, object],
) -> dict[str, object]:
    summary = _boundary_summary(
        contract,
        case_records,
        attempt_count=attempt_count,
        call_count=call_count,
        attempt_outcome_counts=attempt_outcome_counts,
        reported_usage=reported_usage,
    )
    summary.update({
        "schema_version": EXACT_BOUNDARY_PROBE_SUMMARY_VERSION,
        "invalidated_e28_artifact_sha256": (
            contract.invalidated_e28_artifact_sha256
        ),
        "invalidated_e28_probe_contract_sha256": (
            contract.invalidated_e28_probe_contract_sha256
        ),
        "invalidated_e28_legacy_summary_sha256": (
            contract.invalidated_e28_legacy_summary_sha256
        ),
        "invalidated_e28_raw_audit_sha256": (
            contract.invalidated_e28_raw_audit_sha256
        ),
        "invalidated_e28_raw_control_manifest_sha256": (
            contract.invalidated_e28_raw_control_manifest_sha256
        ),
        "schema_validation_contract_sha256": (
            contract.schema_validation_contract_sha256
        ),
        "retry_policy_contract_sha256": (
            contract.retry_policy_contract_sha256
        ),
        "schema_validation_mode": contract.schema_validation_mode,
        "replication_contract": contract.replication_contract,
        "source_control_contract": contract.source_control_contract,
        "interpretation_contract": contract.interpretation_contract,
        "retry_diagnostics": dict(retry_diagnostics),
    })
    return summary


def _run_exact_boundary_probe(
    *,
    output_path: str | Path,
    prepared: PreparedExactBoundaryProbe,
    generator_client: OpenAICompatibleLLMClient,
    authorization: ExactBoundaryProbeAuthorization,
) -> ExactBoundaryProbeRun:
    if not isinstance(prepared, PreparedExactBoundaryProbe):
        raise TypeError("prepared exact boundary run input has invalid type")
    if not isinstance(authorization, ExactBoundaryProbeAuthorization):
        raise TypeError("exact boundary run authorization has invalid type")
    prepared.validate_trusted()
    authorization.validate(prepared.contract)
    validate_exact_boundary_oracle_cases(prepared)
    if type(generator_client) is not OpenAICompatibleLLMClient:
        raise TypeError("exact boundary generator client has invalid type")
    if generator_client.attempt_observer is not None:
        raise ValueError("exact boundary generator already has an observer")
    if llm_transport_contract(generator_client) != prepared.contract.generator_transport:
        raise ValueError("exact boundary generator transport differs")
    final_path = Path(output_path)
    if final_path.as_posix() != prepared.contract.planned_output_path:
        raise ValueError("exact boundary output differs from contract")
    partial_path = final_path.with_name(f"{final_path.name}.partial")
    if final_path.exists() or partial_path.exists():
        raise FileExistsError(
            "exact boundary output or partial output already exists"
        )
    if not final_path.parent.exists() or not final_path.parent.is_dir():
        raise FileNotFoundError("exact boundary output directory does not exist")

    attempt_counts = {
        "accepted": 0,
        "parse_error": 0,
        "schema_error": 0,
        "provider_response_error": 0,
        "http_status_error": 0,
        "transport_error": 0,
    }
    usage = {
        "attempt_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    case_records: list[dict[str, object]] = []
    attempt_groups: list[list[dict[str, object]]] = []
    call_records: list[dict[str, object]] = []
    writer: _SemanticRepairArtifactWriter | None = None
    callback = None
    try:
        writer = _SemanticRepairArtifactWriter(
            partial_path, _exact_boundary_header(prepared, authorization)
        )
        callback = writer.observe_attempt
        generator_client.attempt_observer = callback
        for case in prepared.cases:
            request, _, expected_body = build_boundary_request(
                prepared.source_e28_probe, case.source_case
            )
            writer.begin_call(case.index)
            payload: dict[str, Any] | None = None
            generation_error: LLMResponseError | None = None
            try:
                payload = generator_client.complete_json(
                    request.system,
                    request.user,
                    request.schema,
                    temperature=prepared.contract.generation_temperature,
                    max_tokens=prepared.contract.generation_max_tokens,
                    seed=case.effective_seed,
                    schema_validation_mode=prepared.contract.schema_validation_mode,
                )
            except LLMResponseError as exc:
                generation_error = exc
            call_record, attempts = writer.finish_call()
            attempt_groups.append(attempts)
            call_records.append(call_record)
            for attempt in attempts:
                attempt_counts[attempt["outcome"]] += 1
                observed_usage = attempt["usage"]
                if observed_usage["reported"]:
                    usage["attempt_count"] += 1
                    usage["prompt_tokens"] += (
                        observed_usage["prompt_tokens"] or 0
                    )
                    usage["completion_tokens"] += (
                        observed_usage["completion_tokens"] or 0
                    )
                    usage["total_tokens"] += (
                        observed_usage["total_tokens"] or 0
                    )
            if canonical_json_bytes(attempts[0]["request_body"]) != (
                canonical_json_bytes(expected_body)
            ):
                raise ExactBoundaryProbeValidationError(
                    "hosted exact boundary request differs from preparation"
                )
            if attempts[0]["effective_seed"] != case.effective_seed:
                raise ExactBoundaryProbeValidationError(
                    "hosted exact boundary request changed the E28 seed"
                )
            wire_usable = call_record["final_outcome"] == "transport_accepted"
            if wire_usable is (generation_error is not None):
                raise ExactBoundaryProbeValidationError(
                    "exact boundary return differs from raw diagnostics"
                )
            wire = {
                "usable": wire_usable,
                "terminal_outcome": call_record["final_outcome"],
                "final_attempt_outcome": call_record["final_attempt_outcome"],
                "error_type": _bounded_error_type(generation_error),
            }
            if not wire_usable:
                decode = {"status": "not_run", "error_type": None, "payload": None}
                construction = None
                application = None
            else:
                if payload is None:
                    raise ExactBoundaryProbeValidationError(
                        "accepted exact boundary call returned no payload"
                    )
                decode, construction, application = _boundary_decode_and_score(
                    prepared.source_e28_probe, case.source_case, payload
                )
            record = {
                "schema_version": EXACT_BOUNDARY_PROBE_CASE_VERSION,
                **case.manifest_record(),
                "invalidated_e28_raw_control": (
                    case.invalidated_e28_raw_control.to_record()
                ),
                "boundary_call_record_id": call_record["record_id"],
                "boundary_payload": payload if wire_usable else None,
                "wire": wire,
                "decode": decode,
                "construction": construction,
                "corrected_application": application,
            }
            case_records.append(record)
            writer.write_case(record)
        summary = _exact_boundary_summary(
            prepared.contract,
            case_records,
            attempt_count=writer.attempt_count,
            call_count=writer.call_count,
            attempt_outcome_counts=attempt_counts,
            reported_usage=usage,
            retry_diagnostics=_exact_boundary_retry_diagnostics(
                prepared, attempt_groups, call_records
            ),
        )
        writer.finish(summary)
        if generator_client.attempt_observer is callback:
            generator_client.attempt_observer = None
        validated = read_exact_boundary_probe_artifact(
            partial_path, prepared=prepared, authorization=authorization
        )
        if canonical_json_bytes(validated) != canonical_json_bytes(summary):
            raise ExactBoundaryProbeValidationError(
                "exact boundary summary changed during validation"
            )
        os.link(partial_path, final_path)
        partial_path.unlink()
        return ExactBoundaryProbeRun(
            output_path=final_path,
            contract_sha256=prepared.contract.sha256,
            case_count=len(case_records),
            wire_usable_count=summary["wire_usable_count"],
            decoded_count=summary["typed_decoded_count"],
            symbolic_applied_count=summary["symbolic_applied_count"],
            applied_problem_count=summary["applied_problem_count"],
        )
    finally:
        if generator_client.attempt_observer is callback:
            generator_client.attempt_observer = None
        if writer is not None:
            writer.abort()


def run_exact_boundary_probe(**kwargs: object) -> ExactBoundaryProbeRun:
    try:
        return _run_exact_boundary_probe(**kwargs)
    except SemanticRepairProbeValidationError as exc:
        raise ExactBoundaryProbeValidationError(
            "exact boundary run failed shared transcript validation"
        ) from exc


_EXACT_MANIFEST_KEYS = frozenset({
    "index",
    "e24_case_index",
    "e27_case_index",
    "episode_key",
    "problem_id",
    "action",
    "probe_observation_sha256",
    "effective_seed",
    "e27_case_record_sha256",
    "base_request_sha256",
    "boundary_request_sha256",
    "invalidated_e28_case_record_sha256",
    "invalidated_e28_raw_control_sha256",
})
_EXACT_CASE_KEYS = _EXACT_MANIFEST_KEYS | frozenset({
    "schema_version",
    "invalidated_e28_raw_control",
    "boundary_call_record_id",
    "boundary_payload",
    "wire",
    "decode",
    "construction",
    "corrected_application",
})


def _read_exact_boundary_probe_artifact(
    path: str | Path,
    *,
    prepared: PreparedExactBoundaryProbe,
    authorization: ExactBoundaryProbeAuthorization,
) -> dict[str, object]:
    if not isinstance(prepared, PreparedExactBoundaryProbe):
        raise TypeError("prepared exact boundary reader input has invalid type")
    if not isinstance(authorization, ExactBoundaryProbeAuthorization):
        raise TypeError("exact boundary reader authorization has invalid type")
    prepared.validate_trusted()
    authorization.validate(prepared.contract)
    records = _read_canonical_records(Path(path))
    if len(records) < 2:
        raise ExactBoundaryProbeValidationError(
            "exact boundary artifact is incomplete"
        )
    if canonical_json_bytes(records[0]) != canonical_json_bytes(
        _exact_boundary_header(prepared, authorization)
    ):
        raise ExactBoundaryProbeValidationError(
            "exact boundary artifact header differs"
        )
    summary = records[-1]
    if summary.get("schema_version") != EXACT_BOUNDARY_PROBE_SUMMARY_VERSION:
        raise ExactBoundaryProbeValidationError(
            "exact boundary artifact summary is missing"
        )

    pending_attempts: list[dict[str, object]] = []
    pending_call: dict[str, object] | None = None
    case_records: list[dict[str, object]] = []
    attempt_counts: Counter[str] = Counter()
    usage = {
        "attempt_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    call_count = 0
    raw_attempt_count = 0
    seen_call_ids: set[str] = set()
    attempt_groups: list[list[dict[str, object]]] = []
    call_records: list[dict[str, object]] = []
    for record in records[1:-1]:
        schema = record.get("schema_version")
        if schema == DIAGNOSTIC_ATTEMPT_VERSION:
            if pending_call is not None:
                raise ExactBoundaryProbeValidationError(
                    "exact boundary attempt follows a completed call"
                )
            validate_attempt_record(record)
            if record.get("role") != "generator":
                raise ExactBoundaryProbeValidationError(
                    "exact boundary attempt role is invalid"
                )
            pending_attempts.append(record)
            raw_attempt_count += 1
            attempt_counts[record["outcome"]] += 1
            observed_usage = record["usage"]
            if observed_usage["reported"]:
                usage["attempt_count"] += 1
                usage["prompt_tokens"] += observed_usage["prompt_tokens"] or 0
                usage["completion_tokens"] += (
                    observed_usage["completion_tokens"] or 0
                )
                usage["total_tokens"] += observed_usage["total_tokens"] or 0
            continue
        if schema == DIAGNOSTIC_CALL_VERSION:
            if pending_call is not None or not pending_attempts:
                raise ExactBoundaryProbeValidationError(
                    "exact boundary call has invalid placement"
                )
            validate_call_record(record, attempts=pending_attempts)
            call_id = record.get("call_id")
            if (
                record.get("role") != "generator"
                or not isinstance(call_id, str)
                or call_id in seen_call_ids
            ):
                raise ExactBoundaryProbeValidationError(
                    "exact boundary call identity or role is invalid"
                )
            seen_call_ids.add(call_id)
            pending_call = record
            call_count += 1
            continue
        if schema != EXACT_BOUNDARY_PROBE_CASE_VERSION:
            raise ExactBoundaryProbeValidationError(
                "exact boundary artifact has an unexpected record"
            )
        case_index = len(case_records)
        if case_index >= len(prepared.cases):
            raise ExactBoundaryProbeValidationError(
                "exact boundary artifact has too many cases"
            )
        if pending_call is None or not pending_attempts:
            raise ExactBoundaryProbeValidationError(
                "exact boundary case lacks its call transcript"
            )
        if set(record) != _EXACT_CASE_KEYS:
            raise ExactBoundaryProbeValidationError(
                f"exact boundary case {case_index} has invalid fields"
            )
        case = prepared.cases[case_index]
        if canonical_json_bytes(record["invalidated_e28_raw_control"]) != (
            canonical_json_bytes(case.invalidated_e28_raw_control.to_record())
        ):
            raise ExactBoundaryProbeValidationError(
                f"exact boundary case {case_index} raw control differs"
            )
        final_payload = _validate_call_transcript(
            pending_attempts,
            pending_call,
            initial_body=case.boundary_request_body,
            initial_seed=case.effective_seed,
            transport=prepared.contract.generator_transport,
            wire_schema=boundary_payload_schema_record(),
            schema_validation_mode=prepared.contract.schema_validation_mode,
        )
        if record["boundary_call_record_id"] != pending_call["record_id"]:
            raise ExactBoundaryProbeValidationError(
                "exact boundary case call id differs"
            )
        expected_usable = pending_call["final_outcome"] == "transport_accepted"
        expected_wire = {
            "usable": expected_usable,
            "terminal_outcome": pending_call["final_outcome"],
            "final_attempt_outcome": pending_call["final_attempt_outcome"],
            "error_type": None if expected_usable else "LLMResponseError",
        }
        if type(record["wire"]) is not dict or set(record["wire"]) != {
            "usable",
            "terminal_outcome",
            "final_attempt_outcome",
            "error_type",
        }:
            raise ExactBoundaryProbeValidationError(
                "exact boundary wire fields are invalid"
            )
        if (
            type(record["wire"].get("usable")) is not bool
            or canonical_json_bytes(record["wire"])
            != canonical_json_bytes(expected_wire)
        ):
            raise ExactBoundaryProbeValidationError(
                "exact boundary wire staging differs"
            )
        if expected_usable:
            if canonical_json_bytes(final_payload) != canonical_json_bytes(
                record["boundary_payload"]
            ):
                raise ExactBoundaryProbeValidationError(
                    "exact boundary payload differs from strict raw transcript"
                )
            decode, construction, application = _boundary_decode_and_score(
                prepared.source_e28_probe, case.source_case, final_payload
            )
        else:
            if record["boundary_payload"] is not None:
                raise ExactBoundaryProbeValidationError(
                    "unusable exact boundary call retained a payload"
                )
            decode = {"status": "not_run", "error_type": None, "payload": None}
            construction = None
            application = None
        if any(
            canonical_json_bytes(observed) != canonical_json_bytes(expected)
            for observed, expected in (
                (record["decode"], decode),
                (record["construction"], construction),
                (record["corrected_application"], application),
            )
        ):
            raise ExactBoundaryProbeValidationError(
                "exact boundary case staging differs from strict replay"
            )
        if application is not None:
            _validate_application_record(
                application,
                profile=(
                    prepared.source_e28_probe.source_e27_probe.source_probe.
                    source_rescore.contract.corrected_profile
                ),
                path=f"case {case_index}.application",
            )
        manifest = {key: record[key] for key in _EXACT_MANIFEST_KEYS}
        if canonical_json_bytes(manifest) != canonical_json_bytes(
            case.manifest_record()
        ):
            raise ExactBoundaryProbeValidationError(
                "exact boundary case manifest differs"
            )
        case_records.append(record)
        attempt_groups.append(pending_attempts)
        call_records.append(pending_call)
        pending_attempts = []
        pending_call = None
    if pending_attempts or pending_call is not None:
        raise ExactBoundaryProbeValidationError(
            "exact boundary artifact has an unresolved call"
        )
    if len(case_records) != prepared.contract.case_count:
        raise ExactBoundaryProbeValidationError(
            "exact boundary artifact case count differs"
        )
    artifact_manifest = [
        {key: row[key] for key in _EXACT_MANIFEST_KEYS}
        for row in case_records
    ]
    if canonical_json_sha256(artifact_manifest) != (
        prepared.contract.case_manifest_sha256
    ):
        raise ExactBoundaryProbeValidationError(
            "exact boundary artifact case manifest digest differs"
        )
    for outcome in (
        "accepted",
        "parse_error",
        "schema_error",
        "provider_response_error",
        "http_status_error",
        "transport_error",
    ):
        attempt_counts.setdefault(outcome, 0)
    expected_summary = _exact_boundary_summary(
        prepared.contract,
        case_records,
        attempt_count=raw_attempt_count,
        call_count=call_count,
        attempt_outcome_counts=attempt_counts,
        reported_usage=usage,
        retry_diagnostics=_exact_boundary_retry_diagnostics(
            prepared, attempt_groups, call_records
        ),
    )
    if canonical_json_bytes(summary) != canonical_json_bytes(expected_summary):
        raise ExactBoundaryProbeValidationError(
            "exact boundary artifact summary differs"
        )
    return dict(summary)


def read_exact_boundary_probe_artifact(
    path: str | Path,
    *,
    prepared: PreparedExactBoundaryProbe,
    authorization: ExactBoundaryProbeAuthorization,
) -> dict[str, object]:
    try:
        return _read_exact_boundary_probe_artifact(
            path, prepared=prepared, authorization=authorization
        )
    except SemanticRepairProbeValidationError as exc:
        raise ExactBoundaryProbeValidationError(
            "exact boundary artifact failed shared transcript validation"
        ) from exc


__all__ = [
    "E28_LEGACY_SUMMARY_STATUS",
    "E28_RAW_AUDIT_VERSION",
    "E28_RAW_CONTROL_CASE_VERSION",
    "EXACT_BOUNDARY_PROBE_CASE_VERSION",
    "EXACT_BOUNDARY_PROBE_CONTRACT_VERSION",
    "EXACT_BOUNDARY_PROBE_HEADER_VERSION",
    "EXACT_BOUNDARY_PROBE_SUMMARY_VERSION",
    "EXACT_BOUNDARY_SCHEMA_VALIDATION_VERSION",
    "EXACT_BOUNDARY_RETRY_POLICY_VERSION",
    "EXACT_BOUNDARY_RETRY_DIAGNOSTICS_VERSION",
    "ExactBoundaryProbeAuthorization",
    "ExactBoundaryProbeCase",
    "ExactBoundaryProbeContract",
    "ExactBoundaryProbeRun",
    "ExactBoundaryProbeValidationError",
    "InvalidatedE28RawAudit",
    "InvalidatedE28RawControlCase",
    "PreparedExactBoundaryProbe",
    "exact_boundary_schema_validation_record",
    "exact_boundary_schema_validation_sha256",
    "exact_boundary_retry_policy_record",
    "exact_boundary_retry_policy_sha256",
    "prepare_exact_boundary_probe",
    "read_exact_boundary_probe_artifact",
    "run_exact_boundary_probe",
    "validate_exact_boundary_oracle_cases",
]
