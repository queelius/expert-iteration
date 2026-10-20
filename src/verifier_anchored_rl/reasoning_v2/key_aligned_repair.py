"""Key-aligned semantic repair over the sealed E26 probe cases."""

from __future__ import annotations

import copy
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ..llm_client import (
    LLMResponseError,
    OpenAICompatibleLLMClient,
)
from .actions import DecompositionDraft
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
from .lenses import (
    LEGACY_GENERATION_SCHEMA_VERSION,
    LENS_CONTRACT_VERSION,
    LensRequest,
    decode_policy_payload,
    llm_transport_contract,
)
from .microprobe import file_sha256
from .model import Action, State, Verb
from .records import deserialize_action
from .semantic_repair import (
    SEMANTIC_REPAIR_EXPECTED_CASES,
    SEMANTIC_REPAIR_EXPECTED_PROBLEMS,
    SEMANTIC_REPAIR_EXPECTED_SOURCE_DECISIONS,
    SEMANTIC_REPAIR_FEEDBACK_CONSTRAINTS,
    SEMANTIC_REPAIR_FEEDBACK_SUFFIX,
    SEMANTIC_REPAIR_PRIMARY_APPLIED_MIN,
    SEMANTIC_REPAIR_PRIMARY_APPLIED_PROBLEMS_MIN,
    SEMANTIC_REPAIR_PRIMARY_TYPED_DECODED_MIN,
    SEMANTIC_REPAIR_CASE_VERSION,
    SemanticRepairProbeAuthorization,
    SemanticRepairProbeCase,
    SemanticRepairProbeContract,
    SemanticRepairProbeValidationError,
    PreparedSemanticRepairProbe,
    _base_body_from_repair,
    _bounded_error_type,
    _cost_basis_record,
    _SemanticRepairArtifactWriter,
    _read_canonical_records,
    _reported_cost_record,
    _request_material,
    _score_repair_draft,
    _serialize_decomposition_draft,
    _validate_application_record,
    _validate_attempt_chain,
    _validate_call_transcript,
    _validate_draft_record,
    prepare_semantic_repair_probe,
    read_semantic_repair_probe_artifact,
)
from .transition import PolicyPayloadError

KEY_ALIGNED_REPAIR_CONTRACT_VERSION = "decomposition-key-aligned-repair-contract-v1"
KEY_ALIGNED_REPAIR_HEADER_VERSION = "decomposition-key-aligned-repair-header-v1"
KEY_ALIGNED_REPAIR_CASE_VERSION = "decomposition-key-aligned-repair-case-v1"
KEY_ALIGNED_REPAIR_SUMMARY_VERSION = "decomposition-key-aligned-repair-summary-v1"
KEY_ALIGNED_REPAIR_FEEDBACK_SCHEMA_VERSION = "decomposition-semantic-feedback-v2"
KEY_ALIGNED_REPAIR_FEEDBACK_MAP_VERSION = "decomposition-semantic-feedback-map-v2"
KEY_ALIGNED_REPAIR_SELECTION_CONTRACT = "all-33-sealed-e26-cases-in-canonical-order-v1"
KEY_ALIGNED_REPAIR_REQUEST_BRIDGE_CONTRACT = (
    "e26-request-with-only-key-aligned-feedback-v2-substitution-v1"
)
KEY_ALIGNED_REPAIR_INTERPRETATION_CONTRACT = (
    "cross-version-generator-wire-diagnostic-not-trajectory-reward-or-training-v1"
)
KEY_ALIGNED_REPAIR_EXPECTED_SOURCE_DECISIONS = SEMANTIC_REPAIR_EXPECTED_SOURCE_DECISIONS
KEY_ALIGNED_REPAIR_EXPECTED_CASES = SEMANTIC_REPAIR_EXPECTED_CASES
KEY_ALIGNED_REPAIR_EXPECTED_PROBLEMS = SEMANTIC_REPAIR_EXPECTED_PROBLEMS
KEY_ALIGNED_REPAIR_PRIMARY_TYPED_DECODED_MIN = SEMANTIC_REPAIR_PRIMARY_TYPED_DECODED_MIN
KEY_ALIGNED_REPAIR_PRIMARY_APPLIED_MIN = SEMANTIC_REPAIR_PRIMARY_APPLIED_MIN
KEY_ALIGNED_REPAIR_PRIMARY_APPLIED_PROBLEMS_MIN = (
    SEMANTIC_REPAIR_PRIMARY_APPLIED_PROBLEMS_MIN
)
KEY_ALIGNED_REPAIR_INPUT_USD_PER_MILLION = "0.30"
KEY_ALIGNED_REPAIR_OUTPUT_USD_PER_MILLION = "0.30"
KEY_ALIGNED_REPAIR_FEEDBACK_SUFFIX = SEMANTIC_REPAIR_FEEDBACK_SUFFIX
KEY_ALIGNED_REPAIR_FEEDBACK_CONSTRAINTS: Mapping[str, str] = MappingProxyType(
    dict(SEMANTIC_REPAIR_FEEDBACK_CONSTRAINTS)
)

_SHA256_CHARS = frozenset("0123456789abcdef")
_PREPARED_TRUST_SEAL = object()


class KeyAlignedRepairProbeValidationError(ValueError):
    """Raised when a key-aligned repair contract or artifact is invalid."""


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_CHARS


def _path_record(path: str | Path) -> str:
    value = Path(path).as_posix()
    if not value or value.endswith(".partial") or not value.endswith(".jsonl"):
        raise ValueError("key-aligned output path must name a final JSONL artifact")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("key-aligned output path must be ASCII") from exc
    return value


def _exact_record(value: object, keys: frozenset[str], path: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise KeyAlignedRepairProbeValidationError(f"{path} has invalid fields")
    return value


def key_aligned_repair_feedback_map_record() -> dict[str, object]:
    return {
        "schema_version": KEY_ALIGNED_REPAIR_FEEDBACK_MAP_VERSION,
        "constraints": dict(KEY_ALIGNED_REPAIR_FEEDBACK_CONSTRAINTS),
    }


def key_aligned_repair_feedback_map_sha256() -> str:
    return canonical_json_sha256(key_aligned_repair_feedback_map_record())


def build_key_aligned_repair_feedback(
    domains: tuple[str, ...] | list[str], reason_code: str
) -> dict[str, object]:
    values = list(domains)
    if not 2 <= len(values) <= 6 or any(
        not isinstance(item, str) or not item.strip() for item in values
    ):
        raise ValueError("key-aligned repair domains are invalid")
    if reason_code not in KEY_ALIGNED_REPAIR_FEEDBACK_CONSTRAINTS:
        raise ValueError("key-aligned repair reason code is not bounded")
    return {
        "schema_version": KEY_ALIGNED_REPAIR_FEEDBACK_SCHEMA_VERSION,
        "domains": values,
        "reason_code": reason_code,
        "repair_constraint": KEY_ALIGNED_REPAIR_FEEDBACK_CONSTRAINTS[reason_code],
    }


def _feedback_append_text(feedback: Mapping[str, object]) -> str:
    return (
        "\n\n"
        + canonical_json_bytes(feedback).decode("ascii")
        + "\n"
        + KEY_ALIGNED_REPAIR_FEEDBACK_SUFFIX
    )


def _validate_key_aligned_delta(
    base_body: Mapping[str, object],
    e26_body: Mapping[str, object],
    aligned_body: Mapping[str, object],
    e26_feedback: Mapping[str, object],
    aligned_feedback: Mapping[str, object],
) -> None:
    expected_e26 = copy.deepcopy(base_body)
    expected_aligned = copy.deepcopy(base_body)
    for body, feedback in (
        (expected_e26, e26_feedback),
        (expected_aligned, aligned_feedback),
    ):
        messages = body.get("messages")
        if (
            type(messages) is not list
            or len(messages) != 2
            or type(messages[1]) is not dict
            or not isinstance(messages[1].get("content"), str)
        ):
            raise KeyAlignedRepairProbeValidationError(
                "E26 base request has an invalid message structure"
            )
        messages[1]["content"] += _feedback_append_text(feedback)
    if canonical_json_bytes(expected_e26) != canonical_json_bytes(e26_body):
        raise KeyAlignedRepairProbeValidationError(
            "sealed E26 request differs from its feedback append"
        )
    if canonical_json_bytes(expected_aligned) != canonical_json_bytes(aligned_body):
        raise KeyAlignedRepairProbeValidationError(
            "key-aligned request changed beyond its feedback substitution"
        )
    old = dict(e26_feedback)
    new = dict(aligned_feedback)
    if old.pop("failed_domains", None) != new.pop("domains", None):
        raise KeyAlignedRepairProbeValidationError(
            "key-aligned feedback changed the domain values"
        )
    if old.pop("schema_version", None) != "decomposition-semantic-feedback-v1":
        raise KeyAlignedRepairProbeValidationError(
            "source E26 feedback schema is invalid"
        )
    if new.pop("schema_version", None) != KEY_ALIGNED_REPAIR_FEEDBACK_SCHEMA_VERSION:
        raise KeyAlignedRepairProbeValidationError(
            "key-aligned feedback schema is invalid"
        )
    if old != new:
        raise KeyAlignedRepairProbeValidationError(
            "key-aligned feedback changed beyond the key and schema version"
        )


@dataclass(frozen=True, slots=True)
class KeyAlignedRepairProbeCase:
    index: int
    source_case: SemanticRepairProbeCase
    e26_case_record_sha256: str
    e26_result_sha256: str
    feedback_sha256: str
    base_request_sha256: str
    e26_repair_request_sha256: str
    key_aligned_request_sha256: str
    key_aligned_request_body_bytes: bytes = field(repr=False, compare=True)

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("key-aligned repair case index is invalid")
        if not isinstance(self.source_case, SemanticRepairProbeCase):
            raise ValueError("key-aligned source case is invalid")
        for name in (
            "e26_case_record_sha256",
            "e26_result_sha256",
            "feedback_sha256",
            "base_request_sha256",
            "e26_repair_request_sha256",
            "key_aligned_request_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"key-aligned case {name} is invalid")
        if self.base_request_sha256 != self.source_case.base_request_sha256:
            raise ValueError("key-aligned base request differs from E26")
        if self.e26_repair_request_sha256 != self.source_case.repair_request_sha256:
            raise ValueError("key-aligned source request differs from E26")
        if canonical_json_sha256(self.feedback) != self.feedback_sha256:
            raise ValueError("key-aligned feedback digest is invalid")
        try:
            body = json.loads(self.key_aligned_request_body_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("key-aligned request bytes are invalid") from exc
        if (
            type(body) is not dict
            or canonical_json_bytes(body) != self.key_aligned_request_body_bytes
            or canonical_json_sha256(body) != self.key_aligned_request_sha256
        ):
            raise ValueError("key-aligned request bytes are not canonical")

    @property
    def state(self) -> State:
        return self.source_case.state

    @property
    def action(self) -> Action:
        return self.source_case.action

    @property
    def effective_seed(self) -> int:
        return self.source_case.effective_seed

    @property
    def problem_id(self) -> str:
        return self.source_case.problem_id

    @property
    def probe_observation(self) -> Mapping[str, object]:
        return self.source_case.source_case.source_case.probe_observation

    @property
    def feedback(self) -> dict[str, object]:
        return build_key_aligned_repair_feedback(
            self.source_case.failed_domains,
            self.source_case.corrected_reason_code,
        )

    @property
    def key_aligned_request_body(self) -> dict[str, object]:
        return json.loads(self.key_aligned_request_body_bytes)

    def manifest_record(self) -> dict[str, object]:
        source = self.source_case.manifest_record()
        return {
            "index": self.index,
            "e26_case_index": self.source_case.index,
            "episode_key": source["episode_key"],
            "problem_id": source["problem_id"],
            "action": source["action"],
            "probe_observation_sha256": source["probe_observation_sha256"],
            "effective_seed": source["effective_seed"],
            "corrected_reason_code": source["corrected_reason_code"],
            "corrected_coverage_digest": source["corrected_coverage_digest"],
            "domains": list(self.source_case.failed_domains),
            "e26_case_record_sha256": self.e26_case_record_sha256,
            "e26_result_sha256": self.e26_result_sha256,
            "feedback_sha256": self.feedback_sha256,
            "base_request_sha256": self.base_request_sha256,
            "e26_repair_request_sha256": self.e26_repair_request_sha256,
            "key_aligned_request_sha256": self.key_aligned_request_sha256,
        }


_CONTRACT_DIGEST_FIELDS = (
    "source_environment_contract_sha256",
    "request_environment_contract_sha256",
    "scoring_environment_contract_sha256",
    "corrected_profile_sha256",
    "e26_artifact_sha256",
    "e26_repair_contract_sha256",
    "e26_case_manifest_sha256",
    "e26_result_manifest_sha256",
    "e26_summary_sha256",
    "feedback_map_sha256",
    "case_manifest_sha256",
    "base_request_manifest_sha256",
    "e26_repair_request_manifest_sha256",
    "key_aligned_request_manifest_sha256",
)


@dataclass(frozen=True, slots=True)
class KeyAlignedRepairProbeContract:
    source_environment_contract_sha256: str
    request_environment_contract_sha256: str
    scoring_environment_contract_sha256: str
    corrected_profile_sha256: str
    e26_artifact_sha256: str
    e26_repair_contract_sha256: str
    e26_case_manifest_sha256: str
    e26_result_manifest_sha256: str
    e26_summary_sha256: str
    feedback_map_sha256: str
    case_manifest_sha256: str
    base_request_manifest_sha256: str
    e26_repair_request_manifest_sha256: str
    key_aligned_request_manifest_sha256: str
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
    schema_version: str = KEY_ALIGNED_REPAIR_CONTRACT_VERSION
    selection_contract: str = KEY_ALIGNED_REPAIR_SELECTION_CONTRACT
    request_bridge_contract: str = KEY_ALIGNED_REPAIR_REQUEST_BRIDGE_CONTRACT
    canonical_serializer: str = CANONICAL_JSON_VERSION
    interpretation_contract: str = KEY_ALIGNED_REPAIR_INTERPRETATION_CONTRACT

    def __post_init__(self) -> None:
        _validate_contract(self)

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

    def to_record(self) -> dict[str, object]:
        return _contract_record(self)

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.to_record())

    @classmethod
    def from_record(cls, value: object) -> "KeyAlignedRepairProbeContract":
        return _contract_from_record(value)


@dataclass(frozen=True, slots=True)
class KeyAlignedRepairProbeAuthorization:
    preregistration_id: str
    expected_source_environment_contract_sha256: str
    expected_request_environment_contract_sha256: str
    expected_scoring_environment_contract_sha256: str
    expected_corrected_profile_sha256: str
    expected_e26_artifact_sha256: str
    expected_e26_repair_contract_sha256: str
    expected_e26_case_manifest_sha256: str
    expected_e26_result_manifest_sha256: str
    expected_e26_summary_sha256: str
    expected_feedback_map_sha256: str
    expected_case_manifest_sha256: str
    expected_base_request_manifest_sha256: str
    expected_e26_repair_request_manifest_sha256: str
    expected_key_aligned_request_manifest_sha256: str
    expected_generator_sha256: str
    expected_generator_transport_sha256: str
    expected_generation_settings_sha256: str
    expected_contract_sha256: str
    expected_source_decision_count: int
    expected_case_count: int
    expected_problem_count: int
    expected_output_path: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.preregistration_id, str)
            or not self.preregistration_id.strip()
        ):
            raise ValueError("key-aligned preregistration id is empty")
        for name in self.__dataclass_fields__:
            if name.startswith("expected_") and name.endswith("sha256"):
                if not _is_sha256(getattr(self, name)):
                    raise ValueError(f"key-aligned authorization {name} is invalid")
        if (
            self.expected_source_decision_count
            != KEY_ALIGNED_REPAIR_EXPECTED_SOURCE_DECISIONS
        ):
            raise ValueError("authorized source decision count is invalid")
        if self.expected_case_count != KEY_ALIGNED_REPAIR_EXPECTED_CASES:
            raise ValueError("authorized case count is invalid")
        if self.expected_problem_count != KEY_ALIGNED_REPAIR_EXPECTED_PROBLEMS:
            raise ValueError("authorized problem count is invalid")
        if self.expected_output_path != _path_record(self.expected_output_path):
            raise ValueError("authorized output path is invalid")

    def validate(self, contract: KeyAlignedRepairProbeContract) -> None:
        _validate_authorization(self, contract)


@dataclass(frozen=True, slots=True)
class PreparedKeyAlignedRepairProbe:
    contract: KeyAlignedRepairProbeContract
    source_probe: PreparedSemanticRepairProbe
    source_summary: Mapping[str, object]
    cases: tuple[KeyAlignedRepairProbeCase, ...]
    _trust_seal: object = field(init=False, repr=False, compare=False, default=None)
    _trusted_binding: object = field(
        init=False, repr=False, compare=False, default=None
    )

    def validate_trusted(self) -> None:
        _validate_prepared(self)


@dataclass(frozen=True, slots=True)
class KeyAlignedRepairProbeRun:
    output_path: Path
    contract_sha256: str
    case_count: int
    wire_usable_count: int
    decoded_count: int
    symbolic_applied_count: int
    applied_problem_count: int


def _primary_thresholds() -> dict[str, int]:
    return {
        "typed_decoded_min": KEY_ALIGNED_REPAIR_PRIMARY_TYPED_DECODED_MIN,
        "applied_min": KEY_ALIGNED_REPAIR_PRIMARY_APPLIED_MIN,
        "applied_problem_min": KEY_ALIGNED_REPAIR_PRIMARY_APPLIED_PROBLEMS_MIN,
        "problem_total": KEY_ALIGNED_REPAIR_EXPECTED_PROBLEMS,
    }


def _validate_contract(contract: KeyAlignedRepairProbeContract) -> None:
    if contract.schema_version != KEY_ALIGNED_REPAIR_CONTRACT_VERSION:
        raise ValueError("key-aligned repair contract schema is invalid")
    for name in _CONTRACT_DIGEST_FIELDS:
        if not _is_sha256(getattr(contract, name)):
            raise ValueError(f"key-aligned contract {name} is invalid")
    if contract.feedback_map_sha256 != key_aligned_repair_feedback_map_sha256():
        raise ValueError("key-aligned feedback map digest is invalid")
    if contract.request_environment_contract_sha256 == (
        contract.scoring_environment_contract_sha256
    ):
        raise ValueError(
            "key-aligned request and scoring environments must remain distinct"
        )
    if not isinstance(contract.generator, ModelIdentity):
        raise ValueError("key-aligned generator identity is invalid")
    if not isinstance(contract.generator_transport, ModelTransportContract):
        raise ValueError("key-aligned generator transport is invalid")
    if contract.generator_transport.base_url is None:
        raise ValueError("key-aligned repair requires a hosted generator")
    if contract.generator_transport.model != contract.generator.model:
        raise ValueError("key-aligned generator differs from transport")
    if contract.generation_schema != LEGACY_GENERATION_SCHEMA_VERSION:
        raise ValueError("key-aligned generation schema is invalid")
    if contract.lens_contract != LENS_CONTRACT_VERSION:
        raise ValueError("key-aligned repair requires the E24 v2.1 lens")
    if type(contract.generation_temperature) is not float:
        raise ValueError("key-aligned temperature is invalid")
    if (
        type(contract.generation_max_tokens) is not int
        or contract.generation_max_tokens <= 0
    ):
        raise ValueError("key-aligned token budget is invalid")
    if contract.generation_seed_policy != "turn-and-retry-offset-v1":
        raise ValueError("key-aligned seed policy is invalid")
    if type(contract.generation_seed) is not int or contract.generation_seed < 0:
        raise ValueError("key-aligned base seed is invalid")
    if contract.source_decision_count != KEY_ALIGNED_REPAIR_EXPECTED_SOURCE_DECISIONS:
        raise ValueError("key-aligned source decision count is invalid")
    if contract.case_count != KEY_ALIGNED_REPAIR_EXPECTED_CASES:
        raise ValueError("key-aligned case count is invalid")
    if contract.problem_count != KEY_ALIGNED_REPAIR_EXPECTED_PROBLEMS:
        raise ValueError("key-aligned problem count is invalid")
    if contract.planned_output_path != _path_record(contract.planned_output_path):
        raise ValueError("key-aligned planned output path is invalid")
    if contract.selection_contract != KEY_ALIGNED_REPAIR_SELECTION_CONTRACT:
        raise ValueError("key-aligned selection contract is invalid")
    if contract.request_bridge_contract != KEY_ALIGNED_REPAIR_REQUEST_BRIDGE_CONTRACT:
        raise ValueError("key-aligned request bridge is invalid")
    if contract.canonical_serializer != CANONICAL_JSON_VERSION:
        raise ValueError("key-aligned serializer is invalid")
    if contract.interpretation_contract != KEY_ALIGNED_REPAIR_INTERPRETATION_CONTRACT:
        raise ValueError("key-aligned interpretation contract is invalid")


def _contract_record(contract: KeyAlignedRepairProbeContract) -> dict[str, object]:
    record: dict[str, object] = {"schema_version": contract.schema_version}
    for name in _CONTRACT_DIGEST_FIELDS:
        record[name] = getattr(contract, name)
    record.update(
        {
            "generator": contract.generator.to_record(),
            "generator_sha256": contract.generator_sha256,
            "generator_transport": contract.generator_transport.to_record(),
            "generator_transport_sha256": contract.generator_transport_sha256,
            "generation_settings": contract.generation_settings_record(),
            "generation_settings_sha256": contract.generation_settings_sha256,
            "source_decision_count": contract.source_decision_count,
            "case_count": contract.case_count,
            "problem_count": contract.problem_count,
            "planned_output_path": contract.planned_output_path,
            "feedback_schema_version": KEY_ALIGNED_REPAIR_FEEDBACK_SCHEMA_VERSION,
            "feedback_suffix": KEY_ALIGNED_REPAIR_FEEDBACK_SUFFIX,
            "cost_basis": _cost_basis_record(),
            "primary_thresholds": _primary_thresholds(),
            "selection_contract": contract.selection_contract,
            "request_bridge_contract": contract.request_bridge_contract,
            "canonical_serializer": contract.canonical_serializer,
            "interpretation_contract": contract.interpretation_contract,
        }
    )
    return record


_SETTINGS_KEYS = frozenset(
    {
        "generation_schema",
        "lens_contract",
        "generation_temperature",
        "generation_max_tokens",
        "generation_seed_policy",
        "generation_seed",
    }
)
_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        *_CONTRACT_DIGEST_FIELDS,
        "generator",
        "generator_sha256",
        "generator_transport",
        "generator_transport_sha256",
        "generation_settings",
        "generation_settings_sha256",
        "source_decision_count",
        "case_count",
        "problem_count",
        "planned_output_path",
        "feedback_schema_version",
        "feedback_suffix",
        "cost_basis",
        "primary_thresholds",
        "selection_contract",
        "request_bridge_contract",
        "canonical_serializer",
        "interpretation_contract",
    }
)


def _contract_from_record(value: object) -> KeyAlignedRepairProbeContract:
    record = _exact_record(value, _CONTRACT_KEYS, "key-aligned contract")
    settings = _exact_record(
        record["generation_settings"], _SETTINGS_KEYS, "generation settings"
    )
    if record["feedback_schema_version"] != KEY_ALIGNED_REPAIR_FEEDBACK_SCHEMA_VERSION:
        raise KeyAlignedRepairProbeValidationError("feedback schema is invalid")
    if record["feedback_suffix"] != KEY_ALIGNED_REPAIR_FEEDBACK_SUFFIX:
        raise KeyAlignedRepairProbeValidationError("feedback suffix is invalid")
    if record["cost_basis"] != _cost_basis_record():
        raise KeyAlignedRepairProbeValidationError("cost basis is invalid")
    if record["primary_thresholds"] != _primary_thresholds():
        raise KeyAlignedRepairProbeValidationError("primary thresholds are invalid")
    try:
        contract = KeyAlignedRepairProbeContract(
            **{name: record[name] for name in _CONTRACT_DIGEST_FIELDS},
            generator=ModelIdentity.from_record(record["generator"]),
            generator_transport=ModelTransportContract.from_record(
                record["generator_transport"]
            ),
            generation_schema=settings["generation_schema"],
            lens_contract=settings["lens_contract"],
            generation_temperature=settings["generation_temperature"],
            generation_max_tokens=settings["generation_max_tokens"],
            generation_seed_policy=settings["generation_seed_policy"],
            generation_seed=settings["generation_seed"],
            source_decision_count=record["source_decision_count"],
            case_count=record["case_count"],
            problem_count=record["problem_count"],
            planned_output_path=record["planned_output_path"],
            schema_version=record["schema_version"],
            selection_contract=record["selection_contract"],
            request_bridge_contract=record["request_bridge_contract"],
            canonical_serializer=record["canonical_serializer"],
            interpretation_contract=record["interpretation_contract"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise KeyAlignedRepairProbeValidationError(
            "key-aligned contract record is invalid"
        ) from exc
    for name, expected in {
        "generator_sha256": contract.generator_sha256,
        "generator_transport_sha256": contract.generator_transport_sha256,
        "generation_settings_sha256": contract.generation_settings_sha256,
    }.items():
        if record[name] != expected:
            raise KeyAlignedRepairProbeValidationError(f"{name} differs")
    return contract


def _validate_authorization(
    authorization: KeyAlignedRepairProbeAuthorization,
    contract: KeyAlignedRepairProbeContract,
) -> None:
    checks = {
        "source environment": (
            authorization.expected_source_environment_contract_sha256,
            contract.source_environment_contract_sha256,
        ),
        "request environment": (
            authorization.expected_request_environment_contract_sha256,
            contract.request_environment_contract_sha256,
        ),
        "scoring environment": (
            authorization.expected_scoring_environment_contract_sha256,
            contract.scoring_environment_contract_sha256,
        ),
        "corrected profile": (
            authorization.expected_corrected_profile_sha256,
            contract.corrected_profile_sha256,
        ),
        "E26 artifact": (
            authorization.expected_e26_artifact_sha256,
            contract.e26_artifact_sha256,
        ),
        "E26 repair contract": (
            authorization.expected_e26_repair_contract_sha256,
            contract.e26_repair_contract_sha256,
        ),
        "E26 case manifest": (
            authorization.expected_e26_case_manifest_sha256,
            contract.e26_case_manifest_sha256,
        ),
        "E26 result manifest": (
            authorization.expected_e26_result_manifest_sha256,
            contract.e26_result_manifest_sha256,
        ),
        "E26 summary": (
            authorization.expected_e26_summary_sha256,
            contract.e26_summary_sha256,
        ),
        "feedback map": (
            authorization.expected_feedback_map_sha256,
            contract.feedback_map_sha256,
        ),
        "case manifest": (
            authorization.expected_case_manifest_sha256,
            contract.case_manifest_sha256,
        ),
        "base request manifest": (
            authorization.expected_base_request_manifest_sha256,
            contract.base_request_manifest_sha256,
        ),
        "E26 request manifest": (
            authorization.expected_e26_repair_request_manifest_sha256,
            contract.e26_repair_request_manifest_sha256,
        ),
        "key-aligned request manifest": (
            authorization.expected_key_aligned_request_manifest_sha256,
            contract.key_aligned_request_manifest_sha256,
        ),
        "generator": (
            authorization.expected_generator_sha256,
            contract.generator_sha256,
        ),
        "generator transport": (
            authorization.expected_generator_transport_sha256,
            contract.generator_transport_sha256,
        ),
        "generation settings": (
            authorization.expected_generation_settings_sha256,
            contract.generation_settings_sha256,
        ),
        "contract": (authorization.expected_contract_sha256, contract.sha256),
        "source decision count": (
            authorization.expected_source_decision_count,
            contract.source_decision_count,
        ),
        "case count": (authorization.expected_case_count, contract.case_count),
        "problem count": (authorization.expected_problem_count, contract.problem_count),
        "output path": (
            authorization.expected_output_path,
            contract.planned_output_path,
        ),
    }
    for label, (expected, observed) in checks.items():
        if expected != observed:
            raise ValueError(f"authorized {label} differs")


def _manifest_sha256(cases: tuple[KeyAlignedRepairProbeCase, ...]) -> str:
    return canonical_json_sha256([case.manifest_record() for case in cases])


def _request_manifest_sha256(
    cases: tuple[KeyAlignedRepairProbeCase, ...], field_name: str
) -> str:
    return canonical_json_sha256(
        [{"index": case.index, field_name: getattr(case, field_name)} for case in cases]
    )


def _e26_request_manifest_sha256(
    cases: tuple[KeyAlignedRepairProbeCase, ...],
) -> str:
    return canonical_json_sha256(
        [
            {
                "index": case.index,
                "repair_request_sha256": case.e26_repair_request_sha256,
            }
            for case in cases
        ]
    )


def _e26_result_sha256(record: Mapping[str, object]) -> str:
    return canonical_json_sha256(
        {
            "wire": record["wire"],
            "decode": record["decode"],
            "corrected_application": record["corrected_application"],
        }
    )


def _prepare_key_aligned_repair_probe(
    *,
    source_run_path: str | Path,
    source_diagnostic_path: str | Path,
    e24_artifact_path: str | Path,
    e25_artifact_path: str | Path,
    e26_artifact_path: str | Path,
    e26_authorization: SemanticRepairProbeAuthorization,
    expected_e26_artifact_sha256: str,
    problem_source: CaseworkProblemSource,
    planned_output_path: str | Path,
) -> PreparedKeyAlignedRepairProbe:
    if not isinstance(problem_source, CaseworkProblemSource):
        raise TypeError("key-aligned problem source has an invalid type")
    if not isinstance(e26_authorization, SemanticRepairProbeAuthorization):
        raise TypeError("E26 authorization has an invalid type")
    if not _is_sha256(expected_e26_artifact_sha256):
        raise ValueError("expected E26 artifact digest is invalid")
    e26_path = Path(e26_artifact_path)
    observed_e26_sha256 = file_sha256(e26_path)
    if observed_e26_sha256 != expected_e26_artifact_sha256:
        raise KeyAlignedRepairProbeValidationError("E26 artifact digest differs")
    records = _read_canonical_records(e26_path)
    if len(records) < 2:
        raise KeyAlignedRepairProbeValidationError("E26 artifact is incomplete")
    header = records[0]
    try:
        source_contract = SemanticRepairProbeContract.from_record(
            header["repair_contract"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise KeyAlignedRepairProbeValidationError("E26 contract is invalid") from exc
    if header.get("repair_contract_sha256") != source_contract.sha256:
        raise KeyAlignedRepairProbeValidationError("E26 contract digest differs")
    try:
        e26_authorization.validate(source_contract)
    except ValueError as exc:
        raise KeyAlignedRepairProbeValidationError(
            "E26 contract differs from its frozen authorization"
        ) from exc
    if header.get("preregistration_id") != e26_authorization.preregistration_id:
        raise KeyAlignedRepairProbeValidationError(
            "E26 preregistration id differs from authorization"
        )
    source_probe = prepare_semantic_repair_probe(
        source_run_path=source_run_path,
        source_diagnostic_path=source_diagnostic_path,
        e24_artifact_path=e24_artifact_path,
        e25_artifact_path=e25_artifact_path,
        problem_source=problem_source,
        planned_output_path=e26_authorization.expected_output_path,
    )
    if source_probe.contract.to_record() != source_contract.to_record():
        raise KeyAlignedRepairProbeValidationError(
            "E26 contract differs from strict E23-E25 reconstruction"
        )
    source_summary = read_semantic_repair_probe_artifact(
        e26_path,
        prepared=source_probe,
        authorization=e26_authorization,
    )
    e26_records = [
        record
        for record in records
        if record.get("schema_version") == SEMANTIC_REPAIR_CASE_VERSION
    ]
    if len(e26_records) != len(source_probe.cases):
        raise KeyAlignedRepairProbeValidationError("E26 case count differs")

    selected: list[KeyAlignedRepairProbeCase] = []
    result_manifest: list[dict[str, object]] = []
    for source_case, e26_record in zip(source_probe.cases, e26_records, strict=True):
        if {
            key: e26_record.get(key) for key in source_case.manifest_record()
        } != source_case.manifest_record():
            raise KeyAlignedRepairProbeValidationError(
                f"E26 case {source_case.index} differs from reconstruction"
            )
        feedback = build_key_aligned_repair_feedback(
            source_case.failed_domains, source_case.corrected_reason_code
        )
        _, base_body, aligned_body = _request_material(
            source_probe.source_rescore, source_case.source_case, feedback
        )
        _validate_key_aligned_delta(
            base_body,
            source_case.repair_request_body,
            aligned_body,
            source_case.feedback,
            feedback,
        )
        case_record_sha256 = canonical_json_sha256(e26_record)
        result_sha256 = _e26_result_sha256(e26_record)
        result_manifest.append(
            {
                "index": source_case.index,
                "e26_case_record_sha256": case_record_sha256,
                "e26_result_sha256": result_sha256,
            }
        )
        selected.append(
            KeyAlignedRepairProbeCase(
                index=source_case.index,
                source_case=source_case,
                e26_case_record_sha256=case_record_sha256,
                e26_result_sha256=result_sha256,
                feedback_sha256=canonical_json_sha256(feedback),
                base_request_sha256=canonical_json_sha256(base_body),
                e26_repair_request_sha256=source_case.repair_request_sha256,
                key_aligned_request_sha256=canonical_json_sha256(aligned_body),
                key_aligned_request_body_bytes=canonical_json_bytes(aligned_body),
            )
        )

    cases = tuple(selected)
    contract = KeyAlignedRepairProbeContract(
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
        e26_artifact_sha256=observed_e26_sha256,
        e26_repair_contract_sha256=source_contract.sha256,
        e26_case_manifest_sha256=source_contract.case_manifest_sha256,
        e26_result_manifest_sha256=canonical_json_sha256(result_manifest),
        e26_summary_sha256=canonical_json_sha256(source_summary),
        feedback_map_sha256=key_aligned_repair_feedback_map_sha256(),
        case_manifest_sha256=_manifest_sha256(cases),
        base_request_manifest_sha256=_request_manifest_sha256(
            cases, "base_request_sha256"
        ),
        e26_repair_request_manifest_sha256=_e26_request_manifest_sha256(cases),
        key_aligned_request_manifest_sha256=_request_manifest_sha256(
            cases, "key_aligned_request_sha256"
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
        case_count=len(cases),
        problem_count=len({case.problem_id for case in cases}),
        planned_output_path=_path_record(planned_output_path),
    )
    prepared = PreparedKeyAlignedRepairProbe(
        contract=contract,
        source_probe=source_probe,
        source_summary=dict(source_summary),
        cases=cases,
    )
    binding = (
        prepared.contract,
        prepared.source_probe,
        prepared.source_summary,
        prepared.cases,
    )
    object.__setattr__(prepared, "_trust_seal", _PREPARED_TRUST_SEAL)
    object.__setattr__(prepared, "_trusted_binding", binding)
    prepared.validate_trusted()
    return prepared


def prepare_key_aligned_repair_probe(
    *,
    source_run_path: str | Path,
    source_diagnostic_path: str | Path,
    e24_artifact_path: str | Path,
    e25_artifact_path: str | Path,
    e26_artifact_path: str | Path,
    e26_authorization: SemanticRepairProbeAuthorization,
    expected_e26_artifact_sha256: str,
    problem_source: CaseworkProblemSource,
    planned_output_path: str | Path,
) -> PreparedKeyAlignedRepairProbe:
    try:
        return _prepare_key_aligned_repair_probe(
            source_run_path=source_run_path,
            source_diagnostic_path=source_diagnostic_path,
            e24_artifact_path=e24_artifact_path,
            e25_artifact_path=e25_artifact_path,
            e26_artifact_path=e26_artifact_path,
            e26_authorization=e26_authorization,
            expected_e26_artifact_sha256=expected_e26_artifact_sha256,
            problem_source=problem_source,
            planned_output_path=planned_output_path,
        )
    except SemanticRepairProbeValidationError as exc:
        raise KeyAlignedRepairProbeValidationError(
            "key-aligned preparation failed shared E26 validation"
        ) from exc


def _validate_prepared(prepared: PreparedKeyAlignedRepairProbe) -> None:
    binding = (
        prepared.contract,
        prepared.source_probe,
        prepared.source_summary,
        prepared.cases,
    )
    if (
        prepared._trust_seal is not _PREPARED_TRUST_SEAL
        or prepared._trusted_binding != binding
    ):
        raise KeyAlignedRepairProbeValidationError(
            "key-aligned preparation did not come from strict replay"
        )
    prepared.source_probe.validate_trusted()
    if prepared.source_probe.contract.sha256 != (
        prepared.contract.e26_repair_contract_sha256
    ):
        raise KeyAlignedRepairProbeValidationError("prepared E26 contract differs")
    if canonical_json_sha256(prepared.source_summary) != (
        prepared.contract.e26_summary_sha256
    ):
        raise KeyAlignedRepairProbeValidationError("prepared E26 summary differs")
    if len(prepared.cases) != prepared.contract.case_count:
        raise KeyAlignedRepairProbeValidationError("prepared case count changed")
    if tuple(case.index for case in prepared.cases) != tuple(
        range(len(prepared.cases))
    ):
        raise KeyAlignedRepairProbeValidationError("prepared cases are not ordered")
    if _manifest_sha256(prepared.cases) != prepared.contract.case_manifest_sha256:
        raise KeyAlignedRepairProbeValidationError("prepared case manifest changed")
    request_fields = (
        ("base_request_sha256", prepared.contract.base_request_manifest_sha256),
        (
            "key_aligned_request_sha256",
            prepared.contract.key_aligned_request_manifest_sha256,
        ),
    )
    for field_name, expected in request_fields:
        if _request_manifest_sha256(prepared.cases, field_name) != expected:
            raise KeyAlignedRepairProbeValidationError(
                f"prepared {field_name} manifest changed"
            )
    if _e26_request_manifest_sha256(prepared.cases) != (
        prepared.contract.e26_repair_request_manifest_sha256
    ):
        raise KeyAlignedRepairProbeValidationError(
            "prepared E26 repair request manifest changed"
        )
    if prepared.contract.e26_repair_request_manifest_sha256 != (
        prepared.source_probe.contract.repair_request_manifest_sha256
    ):
        raise KeyAlignedRepairProbeValidationError(
            "prepared E26 request manifest differs from its source contract"
        )
    if (
        len({case.problem_id for case in prepared.cases})
        != prepared.contract.problem_count
    ):
        raise KeyAlignedRepairProbeValidationError("prepared problem count changed")
    result_manifest = []
    for case in prepared.cases:
        result_manifest.append(
            {
                "index": case.index,
                "e26_case_record_sha256": case.e26_case_record_sha256,
                "e26_result_sha256": case.e26_result_sha256,
            }
        )
        _validate_key_aligned_delta(
            case.source_case.base_request_body,
            case.source_case.repair_request_body,
            case.key_aligned_request_body,
            case.source_case.feedback,
            case.feedback,
        )
    if canonical_json_sha256(result_manifest) != (
        prepared.contract.e26_result_manifest_sha256
    ):
        raise KeyAlignedRepairProbeValidationError(
            "prepared E26 result manifest changed"
        )


def _build_key_aligned_repair_request(
    prepared: PreparedKeyAlignedRepairProbe,
    case: KeyAlignedRepairProbeCase,
) -> tuple[LensRequest, dict[str, object], dict[str, object]]:
    if not isinstance(prepared, PreparedKeyAlignedRepairProbe):
        raise TypeError("prepared key-aligned probe has an invalid type")
    if not isinstance(case, KeyAlignedRepairProbeCase):
        raise TypeError("key-aligned case has an invalid type")
    prepared.validate_trusted()
    if case not in prepared.cases:
        raise ValueError("key-aligned case is not in the prepared probe")
    request, base_body, aligned_body = _request_material(
        prepared.source_probe.source_rescore,
        case.source_case.source_case,
        case.feedback,
    )
    _validate_key_aligned_delta(
        base_body,
        case.source_case.repair_request_body,
        aligned_body,
        case.source_case.feedback,
        case.feedback,
    )
    if canonical_json_bytes(base_body) != case.source_case.base_request_body_bytes:
        raise KeyAlignedRepairProbeValidationError(
            "E24 base request changed after preparation"
        )
    if canonical_json_bytes(aligned_body) != case.key_aligned_request_body_bytes:
        raise KeyAlignedRepairProbeValidationError(
            "key-aligned request changed after preparation"
        )
    return request, base_body, aligned_body


def build_key_aligned_repair_request(
    prepared: PreparedKeyAlignedRepairProbe,
    case: KeyAlignedRepairProbeCase,
) -> tuple[LensRequest, dict[str, object], dict[str, object]]:
    try:
        return _build_key_aligned_repair_request(prepared, case)
    except SemanticRepairProbeValidationError as exc:
        raise KeyAlignedRepairProbeValidationError(
            "key-aligned request failed shared E26 validation"
        ) from exc


def _header(
    prepared: PreparedKeyAlignedRepairProbe,
    authorization: KeyAlignedRepairProbeAuthorization,
) -> dict[str, object]:
    contract = prepared.contract
    return {
        "schema_version": KEY_ALIGNED_REPAIR_HEADER_VERSION,
        "canonical_serializer": CANONICAL_JSON_VERSION,
        "preregistration_id": authorization.preregistration_id,
        "repair_contract": contract.to_record(),
        "repair_contract_sha256": contract.sha256,
        "e26_artifact_sha256": contract.e26_artifact_sha256,
        "e26_repair_contract_sha256": contract.e26_repair_contract_sha256,
        "e26_result_manifest_sha256": contract.e26_result_manifest_sha256,
        "request_environment_contract_sha256": (
            contract.request_environment_contract_sha256
        ),
        "scoring_environment_contract_sha256": (
            contract.scoring_environment_contract_sha256
        ),
        "corrected_profile_sha256": contract.corrected_profile_sha256,
        "feedback_map_sha256": contract.feedback_map_sha256,
        "case_manifest_sha256": contract.case_manifest_sha256,
        "base_request_manifest_sha256": contract.base_request_manifest_sha256,
        "e26_repair_request_manifest_sha256": (
            contract.e26_repair_request_manifest_sha256
        ),
        "key_aligned_request_manifest_sha256": (
            contract.key_aligned_request_manifest_sha256
        ),
    }


def _summary(
    contract: KeyAlignedRepairProbeContract,
    case_records: list[dict[str, object]],
    *,
    attempt_count: int,
    call_count: int,
    attempt_outcome_counts: Mapping[str, int],
    reported_usage: Mapping[str, int],
) -> dict[str, object]:
    problem_ids = sorted({row["problem_id"] for row in case_records})
    applied_problem_ids = sorted(
        {
            row["problem_id"]
            for row in case_records
            if row["corrected_application"] is not None
            and row["corrected_application"]["application_status"] == "applied"
        }
    )
    decode_counts = Counter(row["decode"]["status"] for row in case_records)
    for key in ("decoded", "payload_error", "not_run"):
        decode_counts.setdefault(key, 0)
    application_counts = Counter(
        (
            "not_run"
            if row["corrected_application"] is None
            else row["corrected_application"]["application_status"]
        )
        for row in case_records
    )
    for key in ("applied", "algebra_rejected", "not_run"):
        application_counts.setdefault(key, 0)
    source_reasons = Counter(row["corrected_reason_code"] for row in case_records)
    result_reasons: Counter[str] = Counter()
    migrations: Counter[str] = Counter()
    for row in case_records:
        application = row["corrected_application"]
        if application is None:
            result = "not_run"
        elif application["application_status"] == "applied":
            result = "applied"
        else:
            result = application["reason_code"]
        result_reasons[result] += 1
        migrations[f'{row["corrected_reason_code"]}->{result}'] += 1
    by_problem: dict[str, dict[str, int]] = {}
    for problem_id in problem_ids:
        rows = [row for row in case_records if row["problem_id"] == problem_id]
        by_problem[problem_id] = {
            "case_count": len(rows),
            "wire_usable_count": sum(row["wire"]["usable"] for row in rows),
            "typed_decoded_count": sum(
                row["decode"]["status"] == "decoded" for row in rows
            ),
            "symbolic_applied_count": sum(
                row["corrected_application"] is not None
                and row["corrected_application"]["application_status"] == "applied"
                for row in rows
            ),
        }
    typed_decoded = decode_counts["decoded"]
    wire_usable = sum(row["wire"]["usable"] for row in case_records)
    exact_copy_count = sum(row["exact_input_copy"] is True for row in case_records)
    return {
        "schema_version": KEY_ALIGNED_REPAIR_SUMMARY_VERSION,
        "repair_contract_sha256": contract.sha256,
        "e26_result_manifest_sha256": contract.e26_result_manifest_sha256,
        "case_manifest_sha256": contract.case_manifest_sha256,
        "base_request_manifest_sha256": contract.base_request_manifest_sha256,
        "e26_repair_request_manifest_sha256": (
            contract.e26_repair_request_manifest_sha256
        ),
        "key_aligned_request_manifest_sha256": (
            contract.key_aligned_request_manifest_sha256
        ),
        "case_count": len(case_records),
        "problem_count": len(problem_ids),
        "problem_ids": problem_ids,
        "raw_attempt_count": attempt_count,
        "generator_call_count": call_count,
        "wire_usable_count": wire_usable,
        "wire_unusable_count": len(case_records) - wire_usable,
        "typed_decoded_count": typed_decoded,
        "typed_unusable_count": len(case_records) - typed_decoded,
        "decode_counts": dict(sorted(decode_counts.items())),
        "exact_input_copy_count": exact_copy_count,
        "symbolic_applied_count": application_counts["applied"],
        "symbolic_application_counts": dict(sorted(application_counts.items())),
        "applied_problem_count": len(applied_problem_ids),
        "applied_problem_ids": applied_problem_ids,
        "by_problem": by_problem,
        "source_reason_counts": dict(sorted(source_reasons.items())),
        "corrected_result_reason_counts": dict(sorted(result_reasons.items())),
        "reason_migration_counts": dict(sorted(migrations.items())),
        "raw_attempt_outcome_counts": dict(attempt_outcome_counts),
        "reported_usage": dict(reported_usage),
        "reported_cost": _reported_cost_record(reported_usage),
        "primary_thresholds": _primary_thresholds(),
        "interpretation_contract": KEY_ALIGNED_REPAIR_INTERPRETATION_CONTRACT,
    }


def _run_key_aligned_repair_probe(
    *,
    output_path: str | Path,
    prepared: PreparedKeyAlignedRepairProbe,
    generator_client: OpenAICompatibleLLMClient,
    authorization: KeyAlignedRepairProbeAuthorization,
) -> KeyAlignedRepairProbeRun:
    if not isinstance(prepared, PreparedKeyAlignedRepairProbe):
        raise TypeError("prepared key-aligned probe has an invalid type")
    if not isinstance(authorization, KeyAlignedRepairProbeAuthorization):
        raise TypeError("key-aligned authorization has an invalid type")
    prepared.validate_trusted()
    authorization.validate(prepared.contract)
    if type(generator_client) is not OpenAICompatibleLLMClient:
        raise TypeError("key-aligned generator client has an invalid type")
    if generator_client.attempt_observer is not None:
        raise ValueError("key-aligned generator already has an observer")
    if (
        llm_transport_contract(generator_client)
        != prepared.contract.generator_transport
    ):
        raise ValueError("key-aligned generator transport differs from contract")
    final_path = Path(output_path)
    if final_path.as_posix() != prepared.contract.planned_output_path:
        raise ValueError("key-aligned output differs from contract")
    partial_path = final_path.with_name(f"{final_path.name}.partial")
    if final_path.exists() or partial_path.exists():
        raise FileExistsError("key-aligned output or partial output already exists")
    if not final_path.parent.exists() or not final_path.parent.is_dir():
        raise FileNotFoundError("key-aligned output directory does not exist")

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
    writer: _SemanticRepairArtifactWriter | None = None
    callback = None
    try:
        writer = _SemanticRepairArtifactWriter(
            partial_path, _header(prepared, authorization)
        )
        callback = writer.observe_attempt
        generator_client.attempt_observer = callback
        for case in prepared.cases:
            request, _, expected_body = build_key_aligned_repair_request(prepared, case)
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
                )
            except LLMResponseError as exc:
                generation_error = exc
            call_record, attempts = writer.finish_call()
            for attempt in attempts:
                attempt_counts[attempt["outcome"]] += 1
                observed_usage = attempt["usage"]
                if observed_usage["reported"]:
                    usage["attempt_count"] += 1
                    usage["prompt_tokens"] += observed_usage["prompt_tokens"] or 0
                    usage["completion_tokens"] += (
                        observed_usage["completion_tokens"] or 0
                    )
                    usage["total_tokens"] += observed_usage["total_tokens"] or 0
            if attempts[0]["request_body"] != expected_body:
                raise KeyAlignedRepairProbeValidationError(
                    "hosted key-aligned request differs from preparation"
                )
            if attempts[0]["effective_seed"] != case.effective_seed:
                raise KeyAlignedRepairProbeValidationError(
                    "hosted key-aligned request changed the E26 seed"
                )
            wire_usable = call_record["final_outcome"] == "transport_accepted"
            if wire_usable is (generation_error is not None):
                raise KeyAlignedRepairProbeValidationError(
                    "generator return differs from raw diagnostics"
                )
            wire = {
                "usable": wire_usable,
                "terminal_outcome": call_record["final_outcome"],
                "final_attempt_outcome": call_record["final_attempt_outcome"],
                "error_type": _bounded_error_type(generation_error),
            }
            exact_input_copy: bool | None = None
            if not wire_usable:
                decode = {"status": "not_run", "error_type": None, "draft": None}
                corrected_application = None
            else:
                if payload is None:
                    raise KeyAlignedRepairProbeValidationError(
                        "accepted key-aligned call returned no payload"
                    )
                exact_fields = set(payload) == {"domains"}
                if exact_fields and type(payload["domains"]) is list:
                    exact_input_copy = payload["domains"] == list(
                        case.source_case.failed_domains
                    )
                try:
                    if not exact_fields:
                        raise PolicyPayloadError(
                            "key-aligned payload must contain only domains"
                        )
                    draft = decode_policy_payload(
                        case.probe_observation, case.action, payload
                    )
                    if not isinstance(draft, DecompositionDraft):
                        raise PolicyPayloadError(
                            "key-aligned payload is not a decomposition"
                        )
                except PolicyPayloadError as exc:
                    decode = {
                        "status": "payload_error",
                        "error_type": type(exc).__name__,
                        "draft": None,
                    }
                    corrected_application = None
                else:
                    decode = {
                        "status": "decoded",
                        "error_type": None,
                        "draft": _serialize_decomposition_draft(draft),
                    }
                    corrected_application = _score_repair_draft(
                        prepared.source_probe, case.source_case, draft
                    )
            record = {
                "schema_version": KEY_ALIGNED_REPAIR_CASE_VERSION,
                **case.manifest_record(),
                "feedback": case.feedback,
                "repair_call_record_id": call_record["record_id"],
                "wire": wire,
                "decode": decode,
                "exact_input_copy": exact_input_copy,
                "corrected_application": corrected_application,
            }
            case_records.append(record)
            writer.write_case(record)
        summary = _summary(
            prepared.contract,
            case_records,
            attempt_count=writer.attempt_count,
            call_count=writer.call_count,
            attempt_outcome_counts=attempt_counts,
            reported_usage=usage,
        )
        writer.finish(summary)
        if generator_client.attempt_observer is callback:
            generator_client.attempt_observer = None
        validated = read_key_aligned_repair_probe_artifact(
            partial_path, prepared=prepared, authorization=authorization
        )
        if validated != summary:
            raise KeyAlignedRepairProbeValidationError(
                "key-aligned summary changed during validation"
            )
        os.link(partial_path, final_path)
        partial_path.unlink()
        return KeyAlignedRepairProbeRun(
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


def run_key_aligned_repair_probe(
    *,
    output_path: str | Path,
    prepared: PreparedKeyAlignedRepairProbe,
    generator_client: OpenAICompatibleLLMClient,
    authorization: KeyAlignedRepairProbeAuthorization,
) -> KeyAlignedRepairProbeRun:
    try:
        return _run_key_aligned_repair_probe(
            output_path=output_path,
            prepared=prepared,
            generator_client=generator_client,
            authorization=authorization,
        )
    except SemanticRepairProbeValidationError as exc:
        raise KeyAlignedRepairProbeValidationError(
            "key-aligned run failed shared E26 validation"
        ) from exc


_MANIFEST_KEYS = frozenset(
    {
        "index",
        "e26_case_index",
        "episode_key",
        "problem_id",
        "action",
        "probe_observation_sha256",
        "effective_seed",
        "corrected_reason_code",
        "corrected_coverage_digest",
        "domains",
        "e26_case_record_sha256",
        "e26_result_sha256",
        "feedback_sha256",
        "base_request_sha256",
        "e26_repair_request_sha256",
        "key_aligned_request_sha256",
    }
)
_HEADER_KEYS = frozenset(
    {
        "schema_version",
        "canonical_serializer",
        "preregistration_id",
        "repair_contract",
        "repair_contract_sha256",
        "e26_artifact_sha256",
        "e26_repair_contract_sha256",
        "e26_result_manifest_sha256",
        "request_environment_contract_sha256",
        "scoring_environment_contract_sha256",
        "corrected_profile_sha256",
        "feedback_map_sha256",
        "case_manifest_sha256",
        "base_request_manifest_sha256",
        "e26_repair_request_manifest_sha256",
        "key_aligned_request_manifest_sha256",
    }
)
_CASE_KEYS = frozenset(
    {
        *_MANIFEST_KEYS,
        "feedback",
        "repair_call_record_id",
        "wire",
        "decode",
        "exact_input_copy",
        "corrected_application",
        "schema_version",
    }
)
_FEEDBACK_KEYS = frozenset(
    {
        "schema_version",
        "domains",
        "reason_code",
        "repair_constraint",
    }
)
_WIRE_KEYS = frozenset(
    {
        "usable",
        "terminal_outcome",
        "final_attempt_outcome",
        "error_type",
    }
)
_DECODE_KEYS = frozenset({"status", "error_type", "draft"})


def _validate_feedback(
    value: object,
    *,
    domains: object,
    reason_code: object,
    digest: object,
    path: str,
) -> dict[str, object]:
    record = _exact_record(value, _FEEDBACK_KEYS, path)
    if type(domains) is not list or not isinstance(reason_code, str):
        raise KeyAlignedRepairProbeValidationError(f"{path} source is invalid")
    try:
        expected = build_key_aligned_repair_feedback(domains, reason_code)
    except ValueError as exc:
        raise KeyAlignedRepairProbeValidationError(f"{path} is invalid") from exc
    if record != expected or canonical_json_sha256(record) != digest:
        raise KeyAlignedRepairProbeValidationError(f"{path} differs")
    return record


def _read_key_aligned_repair_probe_artifact(
    path: str | Path,
    *,
    prepared: PreparedKeyAlignedRepairProbe,
    authorization: KeyAlignedRepairProbeAuthorization,
) -> dict[str, object]:
    if not isinstance(prepared, PreparedKeyAlignedRepairProbe):
        raise TypeError("prepared artifact validator has an invalid type")
    if not isinstance(authorization, KeyAlignedRepairProbeAuthorization):
        raise TypeError("artifact authorization has an invalid type")
    prepared.validate_trusted()
    authorization.validate(prepared.contract)
    records = _read_canonical_records(Path(path))
    if len(records) < 2:
        raise KeyAlignedRepairProbeValidationError("key-aligned artifact is incomplete")
    header = _exact_record(records[0], _HEADER_KEYS, "key-aligned header")
    if header["schema_version"] != KEY_ALIGNED_REPAIR_HEADER_VERSION:
        raise KeyAlignedRepairProbeValidationError("header schema is invalid")
    if header["canonical_serializer"] != CANONICAL_JSON_VERSION:
        raise KeyAlignedRepairProbeValidationError("header serializer is invalid")
    if header["preregistration_id"] != authorization.preregistration_id:
        raise KeyAlignedRepairProbeValidationError(
            "artifact preregistration differs from authorization"
        )
    contract = KeyAlignedRepairProbeContract.from_record(header["repair_contract"])
    if header["repair_contract_sha256"] != contract.sha256:
        raise KeyAlignedRepairProbeValidationError("header contract digest differs")
    if contract.to_record() != prepared.contract.to_record():
        raise KeyAlignedRepairProbeValidationError(
            "artifact contract differs from strict preparation"
        )
    header_checks = {
        "e26_artifact_sha256": contract.e26_artifact_sha256,
        "e26_repair_contract_sha256": contract.e26_repair_contract_sha256,
        "e26_result_manifest_sha256": contract.e26_result_manifest_sha256,
        "request_environment_contract_sha256": (
            contract.request_environment_contract_sha256
        ),
        "scoring_environment_contract_sha256": (
            contract.scoring_environment_contract_sha256
        ),
        "corrected_profile_sha256": contract.corrected_profile_sha256,
        "feedback_map_sha256": contract.feedback_map_sha256,
        "case_manifest_sha256": contract.case_manifest_sha256,
        "base_request_manifest_sha256": contract.base_request_manifest_sha256,
        "e26_repair_request_manifest_sha256": (
            contract.e26_repair_request_manifest_sha256
        ),
        "key_aligned_request_manifest_sha256": (
            contract.key_aligned_request_manifest_sha256
        ),
    }
    for name, expected in header_checks.items():
        if header[name] != expected:
            raise KeyAlignedRepairProbeValidationError(f"header {name} differs")

    summary = records[-1]
    if (
        type(summary) is not dict
        or summary.get("schema_version") != KEY_ALIGNED_REPAIR_SUMMARY_VERSION
    ):
        raise KeyAlignedRepairProbeValidationError("summary schema is invalid")
    pending_attempts: list[dict[str, object]] = []
    pending_call: dict[str, object] | None = None
    seen_call_ids: set[str] = set()
    case_records: list[dict[str, object]] = []
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
    raw_attempt_count = 0
    call_count = 0
    for record_index, value in enumerate(records[1:-1], start=1):
        schema = value.get("schema_version")
        if schema == DIAGNOSTIC_ATTEMPT_VERSION:
            if pending_call is not None:
                raise KeyAlignedRepairProbeValidationError(
                    "attempt appears before the preceding case"
                )
            validate_attempt_record(value)
            if value["role"] != "generator":
                raise KeyAlignedRepairProbeValidationError(
                    "artifact contains a non-generator attempt"
                )
            if pending_attempts and value["call_id"] != pending_attempts[0]["call_id"]:
                raise KeyAlignedRepairProbeValidationError(
                    "attempt changed call before a call record"
                )
            pending_attempts.append(value)
            _validate_attempt_chain(pending_attempts)
            raw_attempt_count += 1
            attempt_counts[value["outcome"]] += 1
            observed_usage = value["usage"]
            if observed_usage["reported"]:
                usage["attempt_count"] += 1
                usage["prompt_tokens"] += observed_usage["prompt_tokens"] or 0
                usage["completion_tokens"] += observed_usage["completion_tokens"] or 0
                usage["total_tokens"] += observed_usage["total_tokens"] or 0
            continue
        if schema == DIAGNOSTIC_CALL_VERSION:
            if pending_call is not None or not pending_attempts:
                raise KeyAlignedRepairProbeValidationError(
                    "call record is out of sequence"
                )
            validate_call_record(value, attempts=pending_attempts)
            if value["role"] != "generator" or value["call_id"] in seen_call_ids:
                raise KeyAlignedRepairProbeValidationError("call identity is invalid")
            seen_call_ids.add(value["call_id"])
            pending_call = value
            call_count += 1
            continue
        if schema != KEY_ALIGNED_REPAIR_CASE_VERSION:
            raise KeyAlignedRepairProbeValidationError(
                f"artifact record {record_index} is unsupported"
            )
        if pending_call is None or not pending_attempts:
            raise KeyAlignedRepairProbeValidationError(
                "case lacks its raw generator call"
            )
        case_index = len(case_records)
        record = _exact_record(value, _CASE_KEYS, f"key-aligned case {case_index}")
        if record["index"] != case_index or case_index >= len(prepared.cases):
            raise KeyAlignedRepairProbeValidationError("cases are not ordered")
        expected_case = prepared.cases[case_index]
        action = deserialize_action(record["action"], f"case {case_index}.action")
        if action.verb is not Verb.DECOMPOSE:
            raise KeyAlignedRepairProbeValidationError("action is not DECOMPOSE")
        for name in (
            "probe_observation_sha256",
            "corrected_coverage_digest",
            "e26_case_record_sha256",
            "e26_result_sha256",
            "feedback_sha256",
            "base_request_sha256",
            "e26_repair_request_sha256",
            "key_aligned_request_sha256",
        ):
            if not _is_sha256(record[name]):
                raise KeyAlignedRepairProbeValidationError(
                    f"case {case_index}.{name} is invalid"
                )
        feedback = _validate_feedback(
            record["feedback"],
            domains=record["domains"],
            reason_code=record["corrected_reason_code"],
            digest=record["feedback_sha256"],
            path=f"case {case_index}.feedback",
        )
        if record["repair_call_record_id"] != pending_call["record_id"]:
            raise KeyAlignedRepairProbeValidationError(
                "case does not resolve its call record"
            )
        aligned_body = pending_attempts[0]["request_body"]
        if canonical_json_sha256(aligned_body) != record["key_aligned_request_sha256"]:
            raise KeyAlignedRepairProbeValidationError("initial request digest differs")
        base_body = _base_body_from_repair(aligned_body, feedback)
        if canonical_json_sha256(base_body) != record["base_request_sha256"]:
            raise KeyAlignedRepairProbeValidationError("base request digest differs")
        _validate_key_aligned_delta(
            base_body,
            expected_case.source_case.repair_request_body,
            aligned_body,
            expected_case.source_case.feedback,
            feedback,
        )
        if pending_attempts[0]["effective_seed"] != record["effective_seed"]:
            raise KeyAlignedRepairProbeValidationError("initial seed differs")
        payload = _validate_call_transcript(
            pending_attempts,
            pending_call,
            initial_body=aligned_body,
            initial_seed=record["effective_seed"],
            transport=contract.generator_transport,
        )
        wire = _exact_record(record["wire"], _WIRE_KEYS, f"case {case_index}.wire")
        expected_usable = pending_call["final_outcome"] == "transport_accepted"
        if (
            type(wire["usable"]) is not bool
            or wire["usable"] is not expected_usable
            or wire["terminal_outcome"] != pending_call["final_outcome"]
            or wire["final_attempt_outcome"] != pending_call["final_attempt_outcome"]
        ):
            raise KeyAlignedRepairProbeValidationError("wire differs from call")
        if expected_usable:
            if wire["error_type"] is not None:
                raise KeyAlignedRepairProbeValidationError("usable wire has an error")
        elif wire["error_type"] != "LLMResponseError":
            raise KeyAlignedRepairProbeValidationError("wire error type is invalid")
        decode = _exact_record(
            record["decode"], _DECODE_KEYS, f"case {case_index}.decode"
        )
        expected_copy: bool | None = None
        decoded_draft: DecompositionDraft | None = None
        if payload is None:
            if decode != {"status": "not_run", "error_type": None, "draft": None}:
                raise KeyAlignedRepairProbeValidationError("not-run decode differs")
            if record["corrected_application"] is not None:
                raise KeyAlignedRepairProbeValidationError(
                    "not-run case has an application"
                )
        else:
            exact_fields = set(payload) == {"domains"}
            if exact_fields and type(payload["domains"]) is list:
                expected_copy = payload["domains"] == list(
                    expected_case.source_case.failed_domains
                )
            try:
                if not exact_fields:
                    raise PolicyPayloadError(
                        "key-aligned payload must contain only domains"
                    )
                candidate = decode_policy_payload(
                    expected_case.probe_observation,
                    expected_case.action,
                    payload,
                )
                if not isinstance(candidate, DecompositionDraft):
                    raise PolicyPayloadError("payload is not a decomposition")
                decoded_draft = candidate
            except PolicyPayloadError:
                if (
                    decode
                    != {
                        "status": "payload_error",
                        "error_type": "PolicyPayloadError",
                        "draft": None,
                    }
                    or record["corrected_application"] is not None
                ):
                    raise KeyAlignedRepairProbeValidationError(
                        "payload-error staging differs"
                    )
            else:
                if (
                    decode["status"] != "decoded"
                    or decode["error_type"] is not None
                    or decode["draft"] is None
                    or record["corrected_application"] is None
                ):
                    raise KeyAlignedRepairProbeValidationError(
                        "decoded staging differs"
                    )
                _validate_draft_record(
                    decode["draft"], payload, f"case {case_index}.draft"
                )
                _validate_application_record(
                    record["corrected_application"],
                    profile=prepared.source_probe.source_rescore.contract.corrected_profile,
                    path=f"case {case_index}.application",
                )
                if decode["draft"] != _serialize_decomposition_draft(decoded_draft):
                    raise KeyAlignedRepairProbeValidationError("decoded draft differs")
                expected_application = _score_repair_draft(
                    prepared.source_probe, expected_case.source_case, decoded_draft
                )
                if record["corrected_application"] != expected_application:
                    raise KeyAlignedRepairProbeValidationError(
                        "corrected application differs"
                    )
        if record["exact_input_copy"] is not expected_copy:
            raise KeyAlignedRepairProbeValidationError("exact-copy diagnostic differs")
        manifest = {key: record[key] for key in _MANIFEST_KEYS}
        if manifest != expected_case.manifest_record():
            raise KeyAlignedRepairProbeValidationError(
                "case differs from strict preparation"
            )
        if (
            canonical_json_bytes(base_body)
            != expected_case.source_case.base_request_body_bytes
        ):
            raise KeyAlignedRepairProbeValidationError("base request bytes differ")
        if (
            canonical_json_bytes(aligned_body)
            != expected_case.key_aligned_request_body_bytes
        ):
            raise KeyAlignedRepairProbeValidationError("aligned request bytes differ")
        case_records.append(record)
        pending_attempts = []
        pending_call = None
    if pending_attempts or pending_call is not None:
        raise KeyAlignedRepairProbeValidationError("artifact has an unresolved call")
    if len(case_records) != contract.case_count:
        raise KeyAlignedRepairProbeValidationError("artifact case count differs")
    if (
        canonical_json_sha256(
            [{key: row[key] for key in _MANIFEST_KEYS} for row in case_records]
        )
        != contract.case_manifest_sha256
    ):
        raise KeyAlignedRepairProbeValidationError("artifact manifest differs")
    expected_summary = _summary(
        contract,
        case_records,
        attempt_count=raw_attempt_count,
        call_count=call_count,
        attempt_outcome_counts=attempt_counts,
        reported_usage=usage,
    )
    if summary != expected_summary:
        raise KeyAlignedRepairProbeValidationError("artifact summary differs")
    return dict(summary)


def read_key_aligned_repair_probe_artifact(
    path: str | Path,
    *,
    prepared: PreparedKeyAlignedRepairProbe,
    authorization: KeyAlignedRepairProbeAuthorization,
) -> dict[str, object]:
    try:
        return _read_key_aligned_repair_probe_artifact(
            path, prepared=prepared, authorization=authorization
        )
    except SemanticRepairProbeValidationError as exc:
        raise KeyAlignedRepairProbeValidationError(
            "key-aligned artifact failed shared transcript validation"
        ) from exc


__all__ = [
    "KEY_ALIGNED_REPAIR_CASE_VERSION",
    "KEY_ALIGNED_REPAIR_CONTRACT_VERSION",
    "KEY_ALIGNED_REPAIR_EXPECTED_CASES",
    "KEY_ALIGNED_REPAIR_EXPECTED_PROBLEMS",
    "KEY_ALIGNED_REPAIR_EXPECTED_SOURCE_DECISIONS",
    "KEY_ALIGNED_REPAIR_FEEDBACK_CONSTRAINTS",
    "KEY_ALIGNED_REPAIR_FEEDBACK_MAP_VERSION",
    "KEY_ALIGNED_REPAIR_FEEDBACK_SCHEMA_VERSION",
    "KEY_ALIGNED_REPAIR_FEEDBACK_SUFFIX",
    "KEY_ALIGNED_REPAIR_HEADER_VERSION",
    "KEY_ALIGNED_REPAIR_SUMMARY_VERSION",
    "KeyAlignedRepairProbeAuthorization",
    "KeyAlignedRepairProbeCase",
    "KeyAlignedRepairProbeContract",
    "KeyAlignedRepairProbeRun",
    "KeyAlignedRepairProbeValidationError",
    "PreparedKeyAlignedRepairProbe",
    "build_key_aligned_repair_feedback",
    "build_key_aligned_repair_request",
    "key_aligned_repair_feedback_map_record",
    "key_aligned_repair_feedback_map_sha256",
    "prepare_key_aligned_repair_probe",
    "read_key_aligned_repair_probe_artifact",
    "run_key_aligned_repair_probe",
]
