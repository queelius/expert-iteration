"""Bounded one-shot semantic repair over sealed E24 decompositions."""

from __future__ import annotations

import copy
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Mapping

from ..llm_client import (
    LLMResponseError,
    OpenAICompatibleAttempt,
    OpenAICompatibleLLMClient,
    SCHEMA_VALIDATION_LEGACY_COERCE_V1,
    append_openai_compatible_parse_retry,
    build_openai_compatible_json_body,
    classify_openai_compatible_json_content,
    project_openai_compatible_response,
)
from .actions import DecompositionDraft
from .casework_source import CaseworkProblemSource
from .contracts import (
    CANONICAL_JSON_VERSION,
    EnvironmentContract,
    ModelIdentity,
    ModelTransportContract,
    build_environment_contract,
    canonical_json_bytes,
    canonical_json_sha256,
)
from .diagnostics import (
    DIAGNOSTIC_ATTEMPT_VERSION,
    DIAGNOSTIC_CALL_VERSION,
    build_attempt_record,
    build_call_record,
    validate_attempt_record,
    validate_call_record,
)
from .domain_rescore import (
    DOMAIN_RESCORE_CASE_VERSION,
    DOMAIN_RESCORE_EXPECTED_CASES,
    DOMAIN_RESCORE_EXPECTED_SOURCE_DECISIONS,
    DOMAIN_RESCORE_HEADER_VERSION,
    DomainAdapterProfileBinding,
    DomainRescoreCase,
    DomainRescoreContract,
    PreparedDomainRescore,
    prepare_domain_rescore,
    read_domain_rescore_artifact,
    score_decomposition_draft,
)
from .lenses import (
    LEGACY_GENERATION_SCHEMA_VERSION,
    LENS_CONTRACT_VERSION,
    LensRequest,
    build_policy_lens_request,
    decode_policy_payload,
    llm_transport_contract,
)
from .math_adapter import (
    CaseworkMathVerifier,
    validate_decomposition_coverage_v11,
)
from .microprobe import GENERATOR_MICROPROBE_CASE_VERSION, file_sha256
from .model import Action, GoalObjective, State, Verb
from .observation import serialize_obligation
from .records import (
    deserialize_action,
    parse_canonical_json_line,
)
from .transition import PolicyPayloadError


SEMANTIC_REPAIR_CONTRACT_VERSION = "decomposition-semantic-repair-contract-v1"
SEMANTIC_REPAIR_HEADER_VERSION = "decomposition-semantic-repair-header-v1"
SEMANTIC_REPAIR_CASE_VERSION = "decomposition-semantic-repair-case-v1"
SEMANTIC_REPAIR_SUMMARY_VERSION = "decomposition-semantic-repair-summary-v1"
SEMANTIC_REPAIR_FEEDBACK_SCHEMA_VERSION = "decomposition-semantic-feedback-v1"
SEMANTIC_REPAIR_FEEDBACK_MAP_VERSION = "decomposition-semantic-feedback-map-v1"
SEMANTIC_REPAIR_SELECTION_CONTRACT = (
    "all-33-e25-corrected-rejected-decompositions-in-canonical-order-v1"
)
SEMANTIC_REPAIR_REQUEST_BRIDGE_CONTRACT = (
    "byte-exact-e24-request-plus-feedback-corrected-v11-posthoc-score-v1"
)
SEMANTIC_REPAIR_INTERPRETATION_CONTRACT = (
    "cross-version-generator-diagnostic-not-trajectory-reward-or-training-v1"
)
SEMANTIC_REPAIR_EXPECTED_SOURCE_DECISIONS = DOMAIN_RESCORE_EXPECTED_SOURCE_DECISIONS
SEMANTIC_REPAIR_EXPECTED_CASES = DOMAIN_RESCORE_EXPECTED_CASES
SEMANTIC_REPAIR_EXPECTED_PROBLEMS = 5
SEMANTIC_REPAIR_PRIMARY_TYPED_DECODED_MIN = 30
SEMANTIC_REPAIR_PRIMARY_APPLIED_MIN = 17
SEMANTIC_REPAIR_PRIMARY_APPLIED_PROBLEMS_MIN = 4
SEMANTIC_REPAIR_INPUT_USD_PER_MILLION = "0.30"
SEMANTIC_REPAIR_OUTPUT_USD_PER_MILLION = "0.30"
SEMANTIC_REPAIR_FEEDBACK_SUFFIX = (
    "Revise the failed domains once. Return only the same one-field JSON "
    "object required above."
)

_FEEDBACK_CONSTRAINTS = {
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
SEMANTIC_REPAIR_FEEDBACK_CONSTRAINTS: Mapping[str, str] = MappingProxyType(
    _FEEDBACK_CONSTRAINTS
)

_SHA256_CHARS = frozenset("0123456789abcdef")
_PREPARED_TRUST_SEAL = object()


class SemanticRepairProbeValidationError(ValueError):
    """Raised when a semantic-repair contract or artifact is invalid."""


def semantic_repair_feedback_map_record() -> dict[str, object]:
    return {
        "schema_version": SEMANTIC_REPAIR_FEEDBACK_MAP_VERSION,
        "constraints": dict(SEMANTIC_REPAIR_FEEDBACK_CONSTRAINTS),
    }


def semantic_repair_feedback_map_sha256() -> str:
    return canonical_json_sha256(semantic_repair_feedback_map_record())


def _cost_basis_record() -> dict[str, object]:
    return {
        "currency": "USD",
        "input_usd_per_million_tokens": (
            SEMANTIC_REPAIR_INPUT_USD_PER_MILLION
        ),
        "output_usd_per_million_tokens": (
            SEMANTIC_REPAIR_OUTPUT_USD_PER_MILLION
        ),
        "rate_basis": "hf-together-published-rate-observed-2026-07-13",
        "interpretation_contract": (
            "estimate-from-provider-reported-usage-not-billing-receipt-v1"
        ),
    }


def _reported_cost_record(usage: Mapping[str, int]) -> dict[str, object]:
    million = Decimal(1_000_000)
    prompt_cost = (
        Decimal(usage["prompt_tokens"])
        * Decimal(SEMANTIC_REPAIR_INPUT_USD_PER_MILLION)
        / million
    )
    completion_cost = (
        Decimal(usage["completion_tokens"])
        * Decimal(SEMANTIC_REPAIR_OUTPUT_USD_PER_MILLION)
        / million
    )
    return {
        **_cost_basis_record(),
        "prompt_cost_usd": format(prompt_cost, "f"),
        "completion_cost_usd": format(completion_cost, "f"),
        "total_cost_usd": format(prompt_cost + completion_cost, "f"),
    }


def build_semantic_repair_feedback(
    failed_domains: tuple[str, ...] | list[str], reason_code: str
) -> dict[str, object]:
    domains = list(failed_domains)
    if not 2 <= len(domains) <= 6 or any(
        not isinstance(item, str) or not item.strip() for item in domains
    ):
        raise ValueError("semantic-repair failed domains are invalid")
    if reason_code not in SEMANTIC_REPAIR_FEEDBACK_CONSTRAINTS:
        raise ValueError("semantic-repair reason code is not bounded")
    return {
        "schema_version": SEMANTIC_REPAIR_FEEDBACK_SCHEMA_VERSION,
        "failed_domains": domains,
        "reason_code": reason_code,
        "repair_constraint": SEMANTIC_REPAIR_FEEDBACK_CONSTRAINTS[reason_code],
    }


@dataclass(frozen=True, slots=True)
class SemanticRepairProbeCase:
    index: int
    source_case: DomainRescoreCase
    corrected_reason_code: str
    corrected_coverage_digest: str
    e25_case_record_sha256: str
    e25_corrected_result_sha256: str
    failed_domains: tuple[str, ...]
    feedback_sha256: str
    base_request_sha256: str
    repair_request_sha256: str
    base_request_body_bytes: bytes = field(repr=False, compare=True)
    repair_request_body_bytes: bytes = field(repr=False, compare=True)

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("semantic-repair case index is invalid")
        if not isinstance(self.source_case, DomainRescoreCase):
            raise ValueError("semantic-repair source case is invalid")
        if self.corrected_reason_code not in SEMANTIC_REPAIR_FEEDBACK_CONSTRAINTS:
            raise ValueError("semantic-repair corrected reason is unbounded")
        for field_name in (
            "corrected_coverage_digest",
            "e25_case_record_sha256",
            "e25_corrected_result_sha256",
            "feedback_sha256",
            "base_request_sha256",
            "repair_request_sha256",
        ):
            if not _is_sha256(getattr(self, field_name)):
                raise ValueError(f"semantic-repair case {field_name} is invalid")
        if self.failed_domains != tuple(
            item.domain for item in self.source_case.draft.child_obligations
        ):
            raise ValueError("semantic-repair failed domains differ from E24")
        feedback = build_semantic_repair_feedback(
            self.failed_domains, self.corrected_reason_code
        )
        if canonical_json_sha256(feedback) != self.feedback_sha256:
            raise ValueError("semantic-repair feedback digest is invalid")
        for field_name, digest_name in (
            ("base_request_body_bytes", "base_request_sha256"),
            ("repair_request_body_bytes", "repair_request_sha256"),
        ):
            body_bytes = getattr(self, field_name)
            if not isinstance(body_bytes, bytes):
                raise ValueError(f"semantic-repair case {field_name} is invalid")
            try:
                body = json.loads(body_bytes)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError(
                    f"semantic-repair case {field_name} is invalid JSON"
                ) from exc
            if type(body) is not dict or canonical_json_bytes(body) != body_bytes:
                raise ValueError(
                    f"semantic-repair case {field_name} is not canonical"
                )
            if canonical_json_sha256(body) != getattr(self, digest_name):
                raise ValueError(
                    f"semantic-repair case {field_name} digest differs"
                )

    @property
    def state(self) -> State:
        return self.source_case.state

    @property
    def action(self) -> Action:
        return self.source_case.action

    @property
    def source_draft(self) -> DecompositionDraft:
        return self.source_case.draft

    @property
    def effective_seed(self) -> int:
        return self.source_case.source_case.effective_seed

    @property
    def problem_id(self) -> str:
        return self.source_case.source_case.problem_id

    @property
    def feedback(self) -> dict[str, object]:
        return build_semantic_repair_feedback(
            self.failed_domains, self.corrected_reason_code
        )

    @property
    def base_request_body(self) -> dict[str, object]:
        return json.loads(self.base_request_body_bytes)

    @property
    def repair_request_body(self) -> dict[str, object]:
        return json.loads(self.repair_request_body_bytes)

    def manifest_record(self) -> dict[str, object]:
        source = self.source_case.manifest_record()
        return {
            "index": self.index,
            "e24_case_index": source["e24_case_index"],
            "e25_case_index": self.source_case.index,
            "episode_key": source["episode_key"],
            "problem_id": source["problem_id"],
            "action": source["action"],
            "probe_observation_sha256": source["probe_observation_sha256"],
            "e24_draft_sha256": source["e24_draft_sha256"],
            "effective_seed": source["effective_seed"],
            "source_decision_record_id": source["source_decision_record_id"],
            "source_generator_call_id": source["source_generator_call_id"],
            "corrected_reason_code": self.corrected_reason_code,
            "corrected_coverage_digest": self.corrected_coverage_digest,
            "e25_case_record_sha256": self.e25_case_record_sha256,
            "e25_corrected_result_sha256": self.e25_corrected_result_sha256,
            "failed_domains": list(self.failed_domains),
            "feedback_sha256": self.feedback_sha256,
            "base_request_sha256": self.base_request_sha256,
            "repair_request_sha256": self.repair_request_sha256,
        }


@dataclass(frozen=True, slots=True)
class SemanticRepairProbeContract:
    source_run_sha256: str
    source_diagnostic_sha256: str
    source_environment_contract_sha256: str
    request_environment_contract_sha256: str
    source_collection_contract_sha256: str
    source_problem_source_sha256: str
    e24_artifact_sha256: str
    e24_probe_contract_sha256: str
    e24_case_manifest_sha256: str
    e25_artifact_sha256: str
    e25_rescore_contract_sha256: str
    e25_rescore_case_manifest_sha256: str
    e25_result_manifest_sha256: str
    scoring_environment_contract_sha256: str
    corrected_profile_sha256: str
    feedback_map_sha256: str
    case_manifest_sha256: str
    base_request_manifest_sha256: str
    repair_request_manifest_sha256: str
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
    schema_version: str = SEMANTIC_REPAIR_CONTRACT_VERSION
    selection_contract: str = SEMANTIC_REPAIR_SELECTION_CONTRACT
    request_bridge_contract: str = SEMANTIC_REPAIR_REQUEST_BRIDGE_CONTRACT
    canonical_serializer: str = CANONICAL_JSON_VERSION
    interpretation_contract: str = SEMANTIC_REPAIR_INTERPRETATION_CONTRACT

    def __post_init__(self) -> None:
        _validate_semantic_repair_contract(self)

    @property
    def generator_sha256(self) -> str:
        return canonical_json_sha256(self.generator.to_record())

    @property
    def generator_transport_sha256(self) -> str:
        return self.generator_transport.sha256

    @property
    def generation_settings_sha256(self) -> str:
        return canonical_json_sha256(self.generation_settings_record())

    def generation_settings_record(self) -> dict[str, object]:
        return {
            "generation_schema": self.generation_schema,
            "lens_contract": self.lens_contract,
            "generation_temperature": self.generation_temperature,
            "generation_max_tokens": self.generation_max_tokens,
            "generation_seed_policy": self.generation_seed_policy,
            "generation_seed": self.generation_seed,
        }

    def to_record(self) -> dict[str, object]:
        return _semantic_repair_contract_record(self)

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.to_record())

    @classmethod
    def from_record(cls, value: object) -> SemanticRepairProbeContract:
        return _semantic_repair_contract_from_record(value)


@dataclass(frozen=True, slots=True)
class SemanticRepairProbeAuthorization:
    preregistration_id: str
    expected_source_run_sha256: str
    expected_source_diagnostic_sha256: str
    expected_source_environment_contract_sha256: str
    expected_request_environment_contract_sha256: str
    expected_source_collection_contract_sha256: str
    expected_source_problem_source_sha256: str
    expected_e24_artifact_sha256: str
    expected_e24_probe_contract_sha256: str
    expected_e24_case_manifest_sha256: str
    expected_e25_artifact_sha256: str
    expected_e25_rescore_contract_sha256: str
    expected_e25_rescore_case_manifest_sha256: str
    expected_e25_result_manifest_sha256: str
    expected_scoring_environment_contract_sha256: str
    expected_corrected_profile_sha256: str
    expected_feedback_map_sha256: str
    expected_case_manifest_sha256: str
    expected_base_request_manifest_sha256: str
    expected_repair_request_manifest_sha256: str
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
            raise ValueError("semantic-repair preregistration id is empty")
        for field_name in (
            "expected_source_run_sha256",
            "expected_source_diagnostic_sha256",
            "expected_source_environment_contract_sha256",
            "expected_request_environment_contract_sha256",
            "expected_source_collection_contract_sha256",
            "expected_source_problem_source_sha256",
            "expected_e24_artifact_sha256",
            "expected_e24_probe_contract_sha256",
            "expected_e24_case_manifest_sha256",
            "expected_e25_artifact_sha256",
            "expected_e25_rescore_contract_sha256",
            "expected_e25_rescore_case_manifest_sha256",
            "expected_e25_result_manifest_sha256",
            "expected_scoring_environment_contract_sha256",
            "expected_corrected_profile_sha256",
            "expected_feedback_map_sha256",
            "expected_case_manifest_sha256",
            "expected_base_request_manifest_sha256",
            "expected_repair_request_manifest_sha256",
            "expected_generator_sha256",
            "expected_generator_transport_sha256",
            "expected_generation_settings_sha256",
            "expected_contract_sha256",
        ):
            if not _is_sha256(getattr(self, field_name)):
                raise ValueError(
                    f"semantic-repair authorization {field_name} is invalid"
                )
        if self.expected_source_decision_count != (
            SEMANTIC_REPAIR_EXPECTED_SOURCE_DECISIONS
        ):
            raise ValueError("authorized source decision count is invalid")
        if self.expected_case_count != SEMANTIC_REPAIR_EXPECTED_CASES:
            raise ValueError("authorized case count is invalid")
        if self.expected_problem_count != SEMANTIC_REPAIR_EXPECTED_PROBLEMS:
            raise ValueError("authorized problem count is invalid")
        if self.expected_output_path != _path_record(self.expected_output_path):
            raise ValueError("authorized output path is invalid")

    def validate(self, contract: SemanticRepairProbeContract) -> None:
        _validate_semantic_repair_authorization(self, contract)


@dataclass(frozen=True, slots=True)
class PreparedSemanticRepairProbe:
    contract: SemanticRepairProbeContract
    source_rescore: PreparedDomainRescore
    scoring_environment_contract: EnvironmentContract
    cases: tuple[SemanticRepairProbeCase, ...]
    _trust_seal: object = field(init=False, repr=False, compare=False, default=None)
    _trusted_binding: object = field(init=False, repr=False, compare=False, default=None)

    def validate_trusted(self) -> None:
        _validate_prepared_semantic_repair(self)


@dataclass(frozen=True, slots=True)
class SemanticRepairProbeRun:
    output_path: Path
    contract_sha256: str
    case_count: int
    wire_usable_count: int
    decoded_count: int
    symbolic_applied_count: int
    applied_problem_count: int


def build_semantic_repair_request(
    prepared: PreparedSemanticRepairProbe,
    case: SemanticRepairProbeCase,
) -> tuple[LensRequest, dict[str, object], dict[str, object]]:
    """Return the repair lens request, exact E24 body, and repaired body."""
    return _build_semantic_repair_request(prepared, case)


def prepare_semantic_repair_probe(
    *,
    source_run_path: str | Path,
    source_diagnostic_path: str | Path,
    e24_artifact_path: str | Path,
    e25_artifact_path: str | Path,
    problem_source: CaseworkProblemSource,
    planned_output_path: str | Path,
) -> PreparedSemanticRepairProbe:
    return _prepare_semantic_repair_probe(
        source_run_path=source_run_path,
        source_diagnostic_path=source_diagnostic_path,
        e24_artifact_path=e24_artifact_path,
        e25_artifact_path=e25_artifact_path,
        problem_source=problem_source,
        planned_output_path=planned_output_path,
    )


def run_semantic_repair_probe(
    *,
    output_path: str | Path,
    prepared: PreparedSemanticRepairProbe,
    generator_client: OpenAICompatibleLLMClient,
    authorization: SemanticRepairProbeAuthorization,
) -> SemanticRepairProbeRun:
    return _run_semantic_repair_probe(
        output_path=output_path,
        prepared=prepared,
        generator_client=generator_client,
        authorization=authorization,
    )


def read_semantic_repair_probe_artifact(
    path: str | Path,
    *,
    prepared: PreparedSemanticRepairProbe,
    authorization: SemanticRepairProbeAuthorization,
) -> dict[str, object]:
    return _read_semantic_repair_probe_artifact(
        Path(path), prepared=prepared, authorization=authorization
    )


# The implementation helpers follow the public surface so test collectors can
# import the contract while the module remains a single auditable unit.


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA256_CHARS
    )


def _path_record(path: str | Path) -> str:
    value = Path(path).as_posix()
    if not value or value.endswith(".partial") or not value.endswith(".jsonl"):
        raise ValueError(
            "semantic-repair output path must name a final JSONL artifact"
        )
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("semantic-repair output path must be ASCII") from exc
    return value


def _exact_record(
    value: object, keys: frozenset[str], path: str
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise SemanticRepairProbeValidationError(f"{path} keys are invalid")
    return value


def _nonnegative_integer(value: object, path: str) -> int:
    if type(value) is not int or value < 0:
        raise SemanticRepairProbeValidationError(
            f"{path} must be a nonnegative integer"
        )
    return value


def _nonempty_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SemanticRepairProbeValidationError(
            f"{path} must be nonempty text"
        )
    return value


_CONTRACT_DIGEST_FIELDS = (
    "source_run_sha256",
    "source_diagnostic_sha256",
    "source_environment_contract_sha256",
    "request_environment_contract_sha256",
    "source_collection_contract_sha256",
    "source_problem_source_sha256",
    "e24_artifact_sha256",
    "e24_probe_contract_sha256",
    "e24_case_manifest_sha256",
    "e25_artifact_sha256",
    "e25_rescore_contract_sha256",
    "e25_rescore_case_manifest_sha256",
    "e25_result_manifest_sha256",
    "scoring_environment_contract_sha256",
    "corrected_profile_sha256",
    "feedback_map_sha256",
    "case_manifest_sha256",
    "base_request_manifest_sha256",
    "repair_request_manifest_sha256",
)


def _validate_semantic_repair_contract(
    contract: SemanticRepairProbeContract,
) -> None:
    if contract.schema_version != SEMANTIC_REPAIR_CONTRACT_VERSION:
        raise ValueError("semantic-repair contract schema is invalid")
    for field_name in _CONTRACT_DIGEST_FIELDS:
        if not _is_sha256(getattr(contract, field_name)):
            raise ValueError(f"semantic-repair {field_name} is invalid")
    if contract.request_environment_contract_sha256 == (
        contract.scoring_environment_contract_sha256
    ):
        raise ValueError(
            "semantic-repair request and scoring environments must differ"
        )
    if contract.feedback_map_sha256 != semantic_repair_feedback_map_sha256():
        raise ValueError("semantic-repair feedback map digest is invalid")
    if not isinstance(contract.generator, ModelIdentity):
        raise ValueError("semantic-repair generator identity is invalid")
    if not isinstance(contract.generator_transport, ModelTransportContract):
        raise ValueError("semantic-repair generator transport is invalid")
    if contract.generator_transport.base_url is None:
        raise ValueError("semantic-repair requires a hosted generator")
    if contract.generator_transport.model != contract.generator.model:
        raise ValueError("semantic-repair generator differs from transport")
    if contract.generation_schema != LEGACY_GENERATION_SCHEMA_VERSION:
        raise ValueError("semantic-repair generation schema is invalid")
    if contract.lens_contract != LENS_CONTRACT_VERSION:
        raise ValueError("semantic-repair requires the E24 v2.1 lens")
    if type(contract.generation_temperature) is not float:
        raise ValueError("semantic-repair temperature is invalid")
    if (
        type(contract.generation_max_tokens) is not int
        or contract.generation_max_tokens <= 0
    ):
        raise ValueError("semantic-repair token budget is invalid")
    if contract.generation_seed_policy != "turn-and-retry-offset-v1":
        raise ValueError("semantic-repair seed policy is invalid")
    if type(contract.generation_seed) is not int or contract.generation_seed < 0:
        raise ValueError("semantic-repair base seed is invalid")
    if contract.source_decision_count != (
        SEMANTIC_REPAIR_EXPECTED_SOURCE_DECISIONS
    ):
        raise ValueError("semantic-repair source decision count is invalid")
    if contract.case_count != SEMANTIC_REPAIR_EXPECTED_CASES:
        raise ValueError("semantic-repair case count is invalid")
    if contract.problem_count != SEMANTIC_REPAIR_EXPECTED_PROBLEMS:
        raise ValueError("semantic-repair problem count is invalid")
    if contract.planned_output_path != _path_record(
        contract.planned_output_path
    ):
        raise ValueError("semantic-repair planned output path is invalid")
    if contract.selection_contract != SEMANTIC_REPAIR_SELECTION_CONTRACT:
        raise ValueError("semantic-repair selection contract is invalid")
    if contract.request_bridge_contract != (
        SEMANTIC_REPAIR_REQUEST_BRIDGE_CONTRACT
    ):
        raise ValueError("semantic-repair request bridge is invalid")
    if contract.canonical_serializer != CANONICAL_JSON_VERSION:
        raise ValueError("semantic-repair serializer is invalid")
    if contract.interpretation_contract != (
        SEMANTIC_REPAIR_INTERPRETATION_CONTRACT
    ):
        raise ValueError("semantic-repair interpretation contract is invalid")


def _semantic_repair_contract_record(
    contract: SemanticRepairProbeContract,
) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": contract.schema_version,
    }
    for field_name in _CONTRACT_DIGEST_FIELDS:
        record[field_name] = getattr(contract, field_name)
    record.update({
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
        "feedback_schema_version": SEMANTIC_REPAIR_FEEDBACK_SCHEMA_VERSION,
        "feedback_suffix": SEMANTIC_REPAIR_FEEDBACK_SUFFIX,
        "cost_basis": _cost_basis_record(),
        "primary_thresholds": {
            "typed_decoded_min": SEMANTIC_REPAIR_PRIMARY_TYPED_DECODED_MIN,
            "applied_min": SEMANTIC_REPAIR_PRIMARY_APPLIED_MIN,
            "applied_problem_min": (
                SEMANTIC_REPAIR_PRIMARY_APPLIED_PROBLEMS_MIN
            ),
            "problem_total": SEMANTIC_REPAIR_EXPECTED_PROBLEMS,
        },
        "selection_contract": contract.selection_contract,
        "request_bridge_contract": contract.request_bridge_contract,
        "canonical_serializer": contract.canonical_serializer,
        "interpretation_contract": contract.interpretation_contract,
    })
    return record


_GENERATION_SETTINGS_KEYS = frozenset({
    "generation_schema",
    "lens_contract",
    "generation_temperature",
    "generation_max_tokens",
    "generation_seed_policy",
    "generation_seed",
})
_PRIMARY_THRESHOLD_KEYS = frozenset({
    "typed_decoded_min",
    "applied_min",
    "applied_problem_min",
    "problem_total",
})
_CONTRACT_KEYS = frozenset({
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
})


def _semantic_repair_contract_from_record(
    value: object,
) -> SemanticRepairProbeContract:
    record = _exact_record(value, _CONTRACT_KEYS, "semantic-repair contract")
    settings = _exact_record(
        record["generation_settings"],
        _GENERATION_SETTINGS_KEYS,
        "semantic-repair generation settings",
    )
    thresholds = _exact_record(
        record["primary_thresholds"],
        _PRIMARY_THRESHOLD_KEYS,
        "semantic-repair primary thresholds",
    )
    expected_thresholds = {
        "typed_decoded_min": SEMANTIC_REPAIR_PRIMARY_TYPED_DECODED_MIN,
        "applied_min": SEMANTIC_REPAIR_PRIMARY_APPLIED_MIN,
        "applied_problem_min": SEMANTIC_REPAIR_PRIMARY_APPLIED_PROBLEMS_MIN,
        "problem_total": SEMANTIC_REPAIR_EXPECTED_PROBLEMS,
    }
    if thresholds != expected_thresholds:
        raise SemanticRepairProbeValidationError(
            "semantic-repair primary thresholds are invalid"
        )
    if record["feedback_schema_version"] != (
        SEMANTIC_REPAIR_FEEDBACK_SCHEMA_VERSION
    ) or record["feedback_suffix"] != SEMANTIC_REPAIR_FEEDBACK_SUFFIX:
        raise SemanticRepairProbeValidationError(
            "semantic-repair feedback contract is invalid"
        )
    if record["cost_basis"] != _cost_basis_record():
        raise SemanticRepairProbeValidationError(
            "semantic-repair cost basis is invalid"
        )
    try:
        kwargs = {name: record[name] for name in _CONTRACT_DIGEST_FIELDS}
        contract = SemanticRepairProbeContract(
            **kwargs,
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
        raise SemanticRepairProbeValidationError(
            "semantic-repair contract record is invalid"
        ) from exc
    digest_checks = {
        "generator_sha256": contract.generator_sha256,
        "generator_transport_sha256": contract.generator_transport_sha256,
        "generation_settings_sha256": contract.generation_settings_sha256,
    }
    for field_name, expected in digest_checks.items():
        if record[field_name] != expected:
            raise SemanticRepairProbeValidationError(
                f"semantic-repair {field_name} differs"
            )
    return contract


def _validate_semantic_repair_authorization(
    authorization: SemanticRepairProbeAuthorization,
    contract: SemanticRepairProbeContract,
) -> None:
    checks = {
        "source run": (
            authorization.expected_source_run_sha256,
            contract.source_run_sha256,
        ),
        "source diagnostic": (
            authorization.expected_source_diagnostic_sha256,
            contract.source_diagnostic_sha256,
        ),
        "request environment": (
            authorization.expected_request_environment_contract_sha256,
            contract.request_environment_contract_sha256,
        ),
        "source environment": (
            authorization.expected_source_environment_contract_sha256,
            contract.source_environment_contract_sha256,
        ),
        "source collection": (
            authorization.expected_source_collection_contract_sha256,
            contract.source_collection_contract_sha256,
        ),
        "source problem": (
            authorization.expected_source_problem_source_sha256,
            contract.source_problem_source_sha256,
        ),
        "E24 artifact": (
            authorization.expected_e24_artifact_sha256,
            contract.e24_artifact_sha256,
        ),
        "E24 probe contract": (
            authorization.expected_e24_probe_contract_sha256,
            contract.e24_probe_contract_sha256,
        ),
        "E24 case manifest": (
            authorization.expected_e24_case_manifest_sha256,
            contract.e24_case_manifest_sha256,
        ),
        "E25 artifact": (
            authorization.expected_e25_artifact_sha256,
            contract.e25_artifact_sha256,
        ),
        "E25 rescore contract": (
            authorization.expected_e25_rescore_contract_sha256,
            contract.e25_rescore_contract_sha256,
        ),
        "E25 rescore manifest": (
            authorization.expected_e25_rescore_case_manifest_sha256,
            contract.e25_rescore_case_manifest_sha256,
        ),
        "E25 result manifest": (
            authorization.expected_e25_result_manifest_sha256,
            contract.e25_result_manifest_sha256,
        ),
        "scoring environment": (
            authorization.expected_scoring_environment_contract_sha256,
            contract.scoring_environment_contract_sha256,
        ),
        "corrected profile": (
            authorization.expected_corrected_profile_sha256,
            contract.corrected_profile_sha256,
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
        "repair request manifest": (
            authorization.expected_repair_request_manifest_sha256,
            contract.repair_request_manifest_sha256,
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
        "contract": (
            authorization.expected_contract_sha256,
            contract.sha256,
        ),
        "source decision count": (
            authorization.expected_source_decision_count,
            contract.source_decision_count,
        ),
        "case count": (
            authorization.expected_case_count,
            contract.case_count,
        ),
        "problem count": (
            authorization.expected_problem_count,
            contract.problem_count,
        ),
        "output path": (
            authorization.expected_output_path,
            contract.planned_output_path,
        ),
    }
    for label, (expected, observed) in checks.items():
        if expected != observed:
            raise ValueError(f"authorized {label} differs")


def _read_canonical_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = parse_canonical_json_line(line)
            except (TypeError, ValueError) as exc:
                raise SemanticRepairProbeValidationError(
                    f"artifact line {line_number} is invalid"
                ) from exc
            if type(record) is not dict:
                raise SemanticRepairProbeValidationError(
                    f"artifact line {line_number} is not an object"
                )
            records.append(record)
    return records


def _serialize_decomposition_draft(
    draft: DecompositionDraft,
) -> dict[str, object]:
    return {
        "draft_type": "decomposition",
        "connective": draft.connective.value,
        "child_statements": list(draft.child_statements),
        "child_obligations": [
            serialize_obligation(item) for item in draft.child_obligations
        ],
    }


def _scoring_environment_contract(
    source_rescore: PreparedDomainRescore,
) -> EnvironmentContract:
    request_environment = source_rescore.source_microprobe.probe_environment.contract
    adapter = CaseworkMathVerifier.corrected_v11()
    scoring = build_environment_contract(
        generation_contract=adapter,
        verifier=adapter,
        aggregate_builder=adapter,
        lens_contract=request_environment.lens_contract,
        generation_schema=request_environment.generation_schema,
        generation_temperature=request_environment.generation_temperature,
        generation_max_tokens=request_environment.generation_max_tokens,
        generation_seed_policy=request_environment.generation_seed_policy,
        turn_limit=request_environment.turn_limit,
    )
    if scoring.sha256 != (
        source_rescore.contract.corrected_environment_contract_sha256
    ):
        raise SemanticRepairProbeValidationError(
            "semantic-repair scoring environment differs from E25"
        )
    return scoring


def _feedback_append_text(feedback: Mapping[str, object]) -> str:
    return (
        "\n\n"
        + canonical_json_bytes(feedback).decode("ascii")
        + "\n"
        + SEMANTIC_REPAIR_FEEDBACK_SUFFIX
    )


def _validate_request_delta(
    base_body: Mapping[str, object],
    repair_body: Mapping[str, object],
    feedback: Mapping[str, object],
) -> None:
    expected = copy.deepcopy(base_body)
    messages = expected.get("messages")
    if (
        type(messages) is not list
        or len(messages) != 2
        or type(messages[1]) is not dict
        or not isinstance(messages[1].get("content"), str)
    ):
        raise SemanticRepairProbeValidationError(
            "E24 base request has an invalid message structure"
        )
    messages[1]["content"] += _feedback_append_text(feedback)
    if canonical_json_bytes(expected) != canonical_json_bytes(repair_body):
        raise SemanticRepairProbeValidationError(
            "repair request changed beyond the bounded feedback append"
        )


def _request_material(
    source_rescore: PreparedDomainRescore,
    source_case: DomainRescoreCase,
    feedback: Mapping[str, object],
) -> tuple[LensRequest, dict[str, object], dict[str, object]]:
    microprobe = source_rescore.source_microprobe
    contract = microprobe.contract
    e24_case = source_case.source_case
    base_request = build_policy_lens_request(
        e24_case.probe_observation,
        e24_case.action,
        lens_contract=contract.probe_lens_contract,
    )
    base_body = build_openai_compatible_json_body(
        model=contract.generator.model,
        system=base_request.system,
        user=base_request.user,
        schema=base_request.schema,
        temperature=contract.generation_temperature,
        max_tokens=contract.generation_max_tokens,
        seed=e24_case.effective_seed,
        guided_json=contract.generator_transport.guided_json,
    )
    if canonical_json_sha256(base_body) != e24_case.probe_request_sha256:
        raise SemanticRepairProbeValidationError(
            "semantic-repair base request is not byte-exact E24"
        )
    repair_request = LensRequest(
        system=base_request.system,
        user=base_request.user + _feedback_append_text(feedback),
        schema=base_request.schema,
    )
    repair_body = build_openai_compatible_json_body(
        model=contract.generator.model,
        system=repair_request.system,
        user=repair_request.user,
        schema=repair_request.schema,
        temperature=contract.generation_temperature,
        max_tokens=contract.generation_max_tokens,
        seed=e24_case.effective_seed,
        guided_json=contract.generator_transport.guided_json,
    )
    _validate_request_delta(base_body, repair_body, feedback)
    return repair_request, base_body, repair_body


def _build_semantic_repair_request(
    prepared: PreparedSemanticRepairProbe,
    case: SemanticRepairProbeCase,
) -> tuple[LensRequest, dict[str, object], dict[str, object]]:
    if not isinstance(prepared, PreparedSemanticRepairProbe):
        raise TypeError("prepared semantic-repair probe has an invalid type")
    if not isinstance(case, SemanticRepairProbeCase):
        raise TypeError("semantic-repair case has an invalid type")
    prepared.validate_trusted()
    if case not in prepared.cases:
        raise ValueError("semantic-repair case is not in the prepared probe")
    request, base_body, repair_body = _request_material(
        prepared.source_rescore, case.source_case, case.feedback
    )
    if canonical_json_bytes(base_body) != case.base_request_body_bytes:
        raise SemanticRepairProbeValidationError(
            "semantic-repair base request changed after preparation"
        )
    if canonical_json_bytes(repair_body) != case.repair_request_body_bytes:
        raise SemanticRepairProbeValidationError(
            "semantic-repair request changed after preparation"
        )
    return request, base_body, repair_body


def _validate_e24_raw_decompositions(
    e24_path: Path,
    source_rescore: PreparedDomainRescore,
) -> None:
    selected = {
        case.source_case.index: case for case in source_rescore.cases
    }
    seen: set[int] = set()
    pending_attempts: list[dict[str, object]] = []
    pending_call: dict[str, object] | None = None
    for record in _read_canonical_records(e24_path)[1:-1]:
        schema = record.get("schema_version")
        if schema == DIAGNOSTIC_ATTEMPT_VERSION:
            pending_attempts.append(record)
            continue
        if schema == DIAGNOSTIC_CALL_VERSION:
            pending_call = record
            continue
        if schema != GENERATOR_MICROPROBE_CASE_VERSION:
            continue
        case_index = record.get("index")
        source_case = selected.get(case_index)
        if source_case is not None:
            if pending_call is None or not pending_attempts:
                raise SemanticRepairProbeValidationError(
                    "E24 decomposition lacks its raw generator call"
                )
            request, base_body, _ = _request_material(
                source_rescore,
                source_case,
                build_semantic_repair_feedback(
                    tuple(
                        item.domain
                        for item in source_case.draft.child_obligations
                    ),
                    "unsupported_domain",
                ),
            )
            if pending_attempts[0].get("request_body") != base_body:
                raise SemanticRepairProbeValidationError(
                    "E24 raw first request differs from reconstruction"
                )
            final = pending_attempts[-1]
            raw_body = final.get("raw_response_body")
            if not isinstance(raw_body, str):
                raise SemanticRepairProbeValidationError(
                    "E24 accepted decomposition lacks a raw response"
                )
            try:
                envelope = json.loads(raw_body)
                projection = project_openai_compatible_response(envelope)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise SemanticRepairProbeValidationError(
                    "E24 decomposition raw response is invalid"
                ) from exc
            if projection.provider_error is not None or (
                projection.raw_content is None
            ):
                raise SemanticRepairProbeValidationError(
                    "E24 decomposition raw response has no content"
                )
            outcome, payload = classify_openai_compatible_json_content(
                projection.raw_content, request.schema
            )
            if outcome != "accepted" or payload is None:
                raise SemanticRepairProbeValidationError(
                    "E24 decomposition raw content is not typed-usable"
                )
            exact_domains = [
                item.domain for item in source_case.draft.child_obligations
            ]
            if payload.get("domains") != exact_domains:
                raise SemanticRepairProbeValidationError(
                    "E24 raw domains differ from the exact ordered feedback domains"
                )
            try:
                draft = decode_policy_payload(
                    source_case.source_case.probe_observation,
                    source_case.action,
                    payload,
                )
            except PolicyPayloadError as exc:
                raise SemanticRepairProbeValidationError(
                    "E24 raw decomposition payload does not decode"
                ) from exc
            if draft != source_case.draft:
                raise SemanticRepairProbeValidationError(
                    "E24 raw decomposition differs from the sealed draft"
                )
            seen.add(case_index)
        pending_attempts = []
        pending_call = None
    if seen != set(selected):
        raise SemanticRepairProbeValidationError(
            "E24 raw validation did not cover all 33 decompositions"
        )


def _case_manifest_sha256(
    cases: tuple[SemanticRepairProbeCase, ...] | list[SemanticRepairProbeCase],
) -> str:
    return canonical_json_sha256([case.manifest_record() for case in cases])


def _request_manifest_sha256(
    cases: tuple[SemanticRepairProbeCase, ...] | list[SemanticRepairProbeCase],
    field_name: str,
) -> str:
    return canonical_json_sha256([
        {"index": case.index, field_name: getattr(case, field_name)}
        for case in cases
    ])


def _prepare_semantic_repair_probe(
    *,
    source_run_path: str | Path,
    source_diagnostic_path: str | Path,
    e24_artifact_path: str | Path,
    e25_artifact_path: str | Path,
    problem_source: CaseworkProblemSource,
    planned_output_path: str | Path,
) -> PreparedSemanticRepairProbe:
    if not isinstance(problem_source, CaseworkProblemSource):
        raise TypeError("semantic-repair problem source has an invalid type")
    e24_path = Path(e24_artifact_path)
    e25_path = Path(e25_artifact_path)
    e25_summary = read_domain_rescore_artifact(e25_path)
    e25_records = _read_canonical_records(e25_path)
    if len(e25_records) < 2:
        raise SemanticRepairProbeValidationError("E25 artifact is incomplete")
    header = e25_records[0]
    if header.get("schema_version") != DOMAIN_RESCORE_HEADER_VERSION:
        raise SemanticRepairProbeValidationError("E25 header schema is invalid")
    try:
        e25_contract = DomainRescoreContract.from_record(
            header["rescore_contract"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SemanticRepairProbeValidationError(
            "E25 rescore contract is invalid"
        ) from exc
    if header.get("rescore_contract_sha256") != e25_contract.sha256:
        raise SemanticRepairProbeValidationError(
            "E25 rescore contract digest is invalid"
        )

    source_rescore = prepare_domain_rescore(
        source_run_path=source_run_path,
        source_diagnostic_path=source_diagnostic_path,
        e24_artifact_path=e24_path,
        problem_source=problem_source,
        planned_output_path=e25_contract.planned_output_path,
    )
    source_rescore.validate_trusted()
    if source_rescore.contract.to_record() != e25_contract.to_record():
        raise SemanticRepairProbeValidationError(
            "E25 contract differs from strict E23/E24 reconstruction"
        )
    _validate_e24_raw_decompositions(e24_path, source_rescore)

    e25_cases = [
        record
        for record in e25_records
        if record.get("schema_version") == DOMAIN_RESCORE_CASE_VERSION
    ]
    if len(e25_cases) != len(source_rescore.cases):
        raise SemanticRepairProbeValidationError(
            "E25 case count differs from strict reconstruction"
        )
    scoring_environment = _scoring_environment_contract(source_rescore)
    adapter = CaseworkMathVerifier.corrected_v11()
    selected: list[SemanticRepairProbeCase] = []
    for source_case, e25_record in zip(
        source_rescore.cases, e25_cases, strict=True
    ):
        source_manifest = source_case.manifest_record()
        if {
            key: e25_record.get(key) for key in source_manifest
        } != source_manifest:
            raise SemanticRepairProbeValidationError(
                f"E25 case {source_case.index} differs from reconstruction"
            )
        if e25_record.get("decoded_draft") != (
            _serialize_decomposition_draft(source_case.draft)
        ):
            raise SemanticRepairProbeValidationError(
                f"E25 case {source_case.index} draft differs"
            )
        corrected = score_decomposition_draft(
            state=source_case.state,
            action=source_case.action,
            draft=source_case.draft,
            adapter=adapter,
            coverage_validator=validate_decomposition_coverage_v11,
            profile=source_rescore.contract.corrected_profile,
        )
        if e25_record.get("corrected") != corrected:
            raise SemanticRepairProbeValidationError(
                f"E25 case {source_case.index} corrected score differs"
            )
        reason = corrected["reason_code"]
        if (
            corrected["application_status"] != "algebra_rejected"
            or reason not in SEMANTIC_REPAIR_FEEDBACK_CONSTRAINTS
        ):
            raise SemanticRepairProbeValidationError(
                "E25 source is not one of the bounded corrected failures"
            )
        failed_domains = tuple(
            item.domain for item in source_case.draft.child_obligations
        )
        feedback = build_semantic_repair_feedback(failed_domains, reason)
        _, base_body, repair_body = _request_material(
            source_rescore, source_case, feedback
        )
        selected.append(SemanticRepairProbeCase(
            index=len(selected),
            source_case=source_case,
            corrected_reason_code=reason,
            corrected_coverage_digest=corrected["coverage"][
                "context_digest"
            ],
            e25_case_record_sha256=canonical_json_sha256(e25_record),
            e25_corrected_result_sha256=canonical_json_sha256(corrected),
            failed_domains=failed_domains,
            feedback_sha256=canonical_json_sha256(feedback),
            base_request_sha256=canonical_json_sha256(base_body),
            repair_request_sha256=canonical_json_sha256(repair_body),
            base_request_body_bytes=canonical_json_bytes(base_body),
            repair_request_body_bytes=canonical_json_bytes(repair_body),
        ))

    frozen_cases = tuple(selected)
    problem_count = len({case.problem_id for case in frozen_cases})
    microprobe_contract = source_rescore.source_microprobe.contract
    contract = SemanticRepairProbeContract(
        source_run_sha256=microprobe_contract.source_run_sha256,
        source_diagnostic_sha256=microprobe_contract.source_diagnostic_sha256,
        source_environment_contract_sha256=(
            microprobe_contract.source_environment_contract_sha256
        ),
        request_environment_contract_sha256=(
            microprobe_contract.probe_environment_contract_sha256
        ),
        source_collection_contract_sha256=(
            microprobe_contract.source_collection_contract_sha256
        ),
        source_problem_source_sha256=(
            microprobe_contract.source_problem_source_sha256
        ),
        e24_artifact_sha256=file_sha256(e24_path),
        e24_probe_contract_sha256=microprobe_contract.sha256,
        e24_case_manifest_sha256=microprobe_contract.case_manifest_sha256,
        e25_artifact_sha256=file_sha256(e25_path),
        e25_rescore_contract_sha256=source_rescore.contract.sha256,
        e25_rescore_case_manifest_sha256=(
            source_rescore.contract.rescore_case_manifest_sha256
        ),
        e25_result_manifest_sha256=e25_summary["result_manifest_sha256"],
        scoring_environment_contract_sha256=scoring_environment.sha256,
        corrected_profile_sha256=(
            source_rescore.contract.corrected_profile_sha256
        ),
        feedback_map_sha256=semantic_repair_feedback_map_sha256(),
        case_manifest_sha256=_case_manifest_sha256(frozen_cases),
        base_request_manifest_sha256=_request_manifest_sha256(
            frozen_cases, "base_request_sha256"
        ),
        repair_request_manifest_sha256=_request_manifest_sha256(
            frozen_cases, "repair_request_sha256"
        ),
        generator=microprobe_contract.generator,
        generator_transport=microprobe_contract.generator_transport,
        generation_schema=microprobe_contract.generation_schema,
        lens_contract=microprobe_contract.probe_lens_contract,
        generation_temperature=microprobe_contract.generation_temperature,
        generation_max_tokens=microprobe_contract.generation_max_tokens,
        generation_seed_policy=microprobe_contract.generation_seed_policy,
        generation_seed=microprobe_contract.generation_seed,
        source_decision_count=len(source_rescore.source_microprobe.cases),
        case_count=len(frozen_cases),
        problem_count=problem_count,
        planned_output_path=_path_record(planned_output_path),
    )
    prepared = PreparedSemanticRepairProbe(
        contract=contract,
        source_rescore=source_rescore,
        scoring_environment_contract=scoring_environment,
        cases=frozen_cases,
    )
    binding = (
        prepared.contract,
        prepared.source_rescore,
        prepared.scoring_environment_contract,
        prepared.cases,
    )
    object.__setattr__(prepared, "_trust_seal", _PREPARED_TRUST_SEAL)
    object.__setattr__(prepared, "_trusted_binding", binding)
    prepared.validate_trusted()
    return prepared


def _validate_prepared_semantic_repair(
    prepared: PreparedSemanticRepairProbe,
) -> None:
    binding = (
        prepared.contract,
        prepared.source_rescore,
        prepared.scoring_environment_contract,
        prepared.cases,
    )
    if (
        prepared._trust_seal is not _PREPARED_TRUST_SEAL
        or prepared._trusted_binding != binding
    ):
        raise SemanticRepairProbeValidationError(
            "semantic-repair preparation did not come from strict replay"
        )
    prepared.source_rescore.validate_trusted()
    if not isinstance(prepared.scoring_environment_contract, EnvironmentContract):
        raise SemanticRepairProbeValidationError(
            "semantic-repair scoring environment is invalid"
        )
    if prepared.scoring_environment_contract.sha256 != (
        prepared.contract.scoring_environment_contract_sha256
    ):
        raise SemanticRepairProbeValidationError(
            "prepared scoring environment differs from the contract"
        )
    if len(prepared.cases) != prepared.contract.case_count:
        raise SemanticRepairProbeValidationError(
            "prepared semantic-repair case count changed"
        )
    if tuple(case.index for case in prepared.cases) != tuple(
        range(len(prepared.cases))
    ):
        raise SemanticRepairProbeValidationError(
            "prepared semantic-repair cases are not ordered"
        )
    if _case_manifest_sha256(prepared.cases) != (
        prepared.contract.case_manifest_sha256
    ):
        raise SemanticRepairProbeValidationError(
            "prepared semantic-repair case manifest changed"
        )
    if _request_manifest_sha256(
        prepared.cases, "base_request_sha256"
    ) != prepared.contract.base_request_manifest_sha256:
        raise SemanticRepairProbeValidationError(
            "prepared base request manifest changed"
        )
    if _request_manifest_sha256(
        prepared.cases, "repair_request_sha256"
    ) != prepared.contract.repair_request_manifest_sha256:
        raise SemanticRepairProbeValidationError(
            "prepared repair request manifest changed"
        )
    if len({case.problem_id for case in prepared.cases}) != (
        prepared.contract.problem_count
    ):
        raise SemanticRepairProbeValidationError(
            "prepared semantic-repair problem count changed"
        )
    if prepared.contract.request_environment_contract_sha256 != (
        prepared.source_rescore.source_microprobe.contract.
        probe_environment_contract_sha256
    ):
        raise SemanticRepairProbeValidationError(
            "semantic-repair request environment is not E24"
        )
    if prepared.contract.source_environment_contract_sha256 != (
        prepared.source_rescore.source_microprobe.contract.
        source_environment_contract_sha256
    ):
        raise SemanticRepairProbeValidationError(
            "semantic-repair source environment is not E23"
        )
    for case in prepared.cases:
        if case.base_request_sha256 != (
            case.source_case.source_case.probe_request_sha256
        ):
            raise SemanticRepairProbeValidationError(
                "prepared base request digest differs from E24"
            )
        _validate_request_delta(
            case.base_request_body,
            case.repair_request_body,
            case.feedback,
        )


def _validate_attempt_chain(attempts: list[dict[str, object]]) -> None:
    previous: Mapping[str, object] | None = None
    for attempt in attempts:
        validate_attempt_record(attempt)
        if previous is None:
            if (
                attempt["attempt_index"] != 0
                or attempt["parse_retry_index"] != 0
                or attempt["backoff_retry_index"] != 0
            ):
                raise SemanticRepairProbeValidationError(
                    "semantic-repair call does not begin at attempt zero"
                )
        else:
            if attempt["call_id"] != previous["call_id"]:
                raise SemanticRepairProbeValidationError(
                    "semantic-repair attempt changed call identity"
                )
            if attempt["attempt_index"] != previous["attempt_index"] + 1:
                raise SemanticRepairProbeValidationError(
                    "semantic-repair attempts are not contiguous"
                )
            disposition = previous["disposition"]
            if disposition == "parse_retry":
                expected_parse = previous["parse_retry_index"] + 1
                expected_backoff = previous["backoff_retry_index"]
            elif disposition == "backoff_retry":
                expected_parse = previous["parse_retry_index"]
                expected_backoff = previous["backoff_retry_index"] + 1
            else:
                raise SemanticRepairProbeValidationError(
                    "terminal semantic-repair attempt has a successor"
                )
            if (
                attempt["parse_retry_index"] != expected_parse
                or attempt["backoff_retry_index"] != expected_backoff
            ):
                raise SemanticRepairProbeValidationError(
                    "semantic-repair retry indices are invalid"
                )
            if disposition == "parse_retry":
                previous_seed = previous["effective_seed"]
                expected_seed = (
                    None if previous_seed is None else previous_seed + 1
                )
                if attempt["effective_seed"] != expected_seed:
                    raise SemanticRepairProbeValidationError(
                        "semantic-repair parse retry seed is invalid"
                    )
            elif attempt["effective_seed"] != previous["effective_seed"]:
                raise SemanticRepairProbeValidationError(
                    "semantic-repair backoff changed its seed"
                )
        previous = attempt


class _SemanticRepairArtifactWriter:
    def __init__(self, path: Path, header: Mapping[str, object]):
        self.path = path
        self._handle: BinaryIO = path.open("xb")
        self._active_index: int | None = None
        self._active_call_id: str | None = None
        self._attempts: list[dict[str, object]] = []
        self.attempt_count = 0
        self.call_count = 0
        self._closed = False
        self._write(header)

    def _write(self, record: Mapping[str, object]) -> None:
        if self._closed:
            raise SemanticRepairProbeValidationError(
                "semantic-repair artifact writer is closed"
            )
        self._handle.write(canonical_json_bytes(record) + b"\n")
        self._handle.flush()

    def begin_call(self, case_index: int) -> None:
        if self._active_index is not None:
            raise SemanticRepairProbeValidationError(
                "semantic-repair call is already active"
            )
        self._active_index = case_index
        self._active_call_id = None
        self._attempts = []

    def observe_attempt(self, event: OpenAICompatibleAttempt) -> None:
        if self._active_index is None:
            raise SemanticRepairProbeValidationError(
                "semantic-repair observed an attempt outside a case"
            )
        record = build_attempt_record(event, role="generator")
        if self._active_call_id is None:
            self._active_call_id = event.call_id
        elif event.call_id != self._active_call_id:
            raise SemanticRepairProbeValidationError(
                "semantic-repair case emitted more than one call"
            )
        _validate_attempt_chain([*self._attempts, record])
        self._attempts.append(record)
        self.attempt_count += 1
        self._write(record)

    def finish_call(
        self,
    ) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
        if self._active_index is None or self._active_call_id is None:
            raise SemanticRepairProbeValidationError(
                "semantic-repair call produced no raw attempt"
            )
        _validate_attempt_chain(self._attempts)
        call = build_call_record(
            "generator", self._active_call_id, self._attempts
        )
        attempts = tuple(self._attempts)
        self._write(call)
        self._active_index = None
        self._active_call_id = None
        self._attempts = []
        self.call_count += 1
        return call, attempts

    def write_case(self, record: Mapping[str, object]) -> None:
        if self._active_index is not None:
            raise SemanticRepairProbeValidationError(
                "semantic-repair case written before call completion"
            )
        self._write(record)

    def finish(self, summary: Mapping[str, object]) -> None:
        if self._active_index is not None:
            raise SemanticRepairProbeValidationError(
                "semantic-repair artifact ended during a call"
            )
        self._write(summary)
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._closed = True

    def abort(self) -> None:
        if not self._closed:
            self._handle.flush()
            self._handle.close()
            self._closed = True


def _semantic_repair_header(
    prepared: PreparedSemanticRepairProbe,
    authorization: SemanticRepairProbeAuthorization,
) -> dict[str, object]:
    contract = prepared.contract
    return {
        "schema_version": SEMANTIC_REPAIR_HEADER_VERSION,
        "canonical_serializer": CANONICAL_JSON_VERSION,
        "preregistration_id": authorization.preregistration_id,
        "repair_contract": contract.to_record(),
        "repair_contract_sha256": contract.sha256,
        "request_environment_contract_sha256": (
            contract.request_environment_contract_sha256
        ),
        "scoring_environment_contract_sha256": (
            contract.scoring_environment_contract_sha256
        ),
        "corrected_profile_sha256": contract.corrected_profile_sha256,
        "feedback_map_sha256": contract.feedback_map_sha256,
        "case_manifest_sha256": contract.case_manifest_sha256,
        "base_request_manifest_sha256": (
            contract.base_request_manifest_sha256
        ),
        "repair_request_manifest_sha256": (
            contract.repair_request_manifest_sha256
        ),
    }


def _bounded_error_type(error: Exception | None) -> str | None:
    return None if error is None else type(error).__name__


def _score_repair_draft(
    prepared: PreparedSemanticRepairProbe,
    case: SemanticRepairProbeCase,
    draft: DecompositionDraft,
) -> dict[str, object]:
    return score_decomposition_draft(
        state=case.state,
        action=case.action,
        draft=draft,
        adapter=CaseworkMathVerifier.corrected_v11(),
        coverage_validator=validate_decomposition_coverage_v11,
        profile=prepared.source_rescore.contract.corrected_profile,
    )


def _new_summary(
    contract: SemanticRepairProbeContract,
    case_records: list[dict[str, object]],
    *,
    attempt_count: int,
    call_count: int,
    attempt_outcome_counts: Mapping[str, int],
    reported_usage: Mapping[str, int],
) -> dict[str, object]:
    problem_ids = sorted({record["problem_id"] for record in case_records})
    applied_problem_ids = sorted({
        record["problem_id"]
        for record in case_records
        if record["corrected_application"] is not None
        and record["corrected_application"]["application_status"] == "applied"
    })
    wire_usable = sum(record["wire"]["usable"] for record in case_records)
    decoded = sum(
        record["decode"]["status"] == "decoded" for record in case_records
    )
    application_counts = Counter(
        "not_run"
        if record["corrected_application"] is None
        else record["corrected_application"]["application_status"]
        for record in case_records
    )
    for key in ("applied", "algebra_rejected", "not_run"):
        application_counts.setdefault(key, 0)
    decode_counts = Counter(record["decode"]["status"] for record in case_records)
    for key in ("decoded", "payload_error", "not_run"):
        decode_counts.setdefault(key, 0)
    by_problem: dict[str, dict[str, int]] = {}
    for problem_id in problem_ids:
        rows = [row for row in case_records if row["problem_id"] == problem_id]
        by_problem[problem_id] = {
            "case_count": len(rows),
            "wire_usable_count": sum(row["wire"]["usable"] for row in rows),
            "decoded_count": sum(
                row["decode"]["status"] == "decoded" for row in rows
            ),
            "symbolic_applied_count": sum(
                row["corrected_application"] is not None
                and row["corrected_application"]["application_status"]
                == "applied"
                for row in rows
            ),
        }
    source_reason_counts = Counter(
        row["corrected_reason_code"] for row in case_records
    )
    result_reason_counts: Counter[str] = Counter()
    migration_counts: Counter[str] = Counter()
    for row in case_records:
        application = row["corrected_application"]
        if application is None:
            result = "not_run"
        elif application["application_status"] == "applied":
            result = "applied"
        else:
            result = application["reason_code"]
        result_reason_counts[result] += 1
        migration_counts[f'{row["corrected_reason_code"]}->{result}'] += 1
    return {
        "schema_version": SEMANTIC_REPAIR_SUMMARY_VERSION,
        "repair_contract_sha256": contract.sha256,
        "case_manifest_sha256": contract.case_manifest_sha256,
        "base_request_manifest_sha256": contract.base_request_manifest_sha256,
        "repair_request_manifest_sha256": (
            contract.repair_request_manifest_sha256
        ),
        "case_count": len(case_records),
        "problem_count": len(problem_ids),
        "problem_ids": problem_ids,
        "raw_attempt_count": attempt_count,
        "generator_call_count": call_count,
        "wire_usable_count": wire_usable,
        "wire_unusable_count": len(case_records) - wire_usable,
        "decoded_count": decoded,
        "typed_decoded_count": decoded,
        "typed_unusable_count": len(case_records) - decoded,
        "decode_counts": dict(sorted(decode_counts.items())),
        "symbolic_applied_count": application_counts["applied"],
        "symbolic_application_counts": dict(sorted(application_counts.items())),
        "applied_problem_count": len(applied_problem_ids),
        "applied_problem_ids": applied_problem_ids,
        "by_problem": by_problem,
        "source_reason_counts": dict(sorted(source_reason_counts.items())),
        "corrected_result_reason_counts": dict(
            sorted(result_reason_counts.items())
        ),
        "reason_migration_counts": dict(sorted(migration_counts.items())),
        "raw_attempt_outcome_counts": dict(attempt_outcome_counts),
        "reported_usage": dict(reported_usage),
        "reported_cost": _reported_cost_record(reported_usage),
        "primary_thresholds": {
            "typed_decoded_min": SEMANTIC_REPAIR_PRIMARY_TYPED_DECODED_MIN,
            "applied_min": SEMANTIC_REPAIR_PRIMARY_APPLIED_MIN,
            "applied_problem_min": (
                SEMANTIC_REPAIR_PRIMARY_APPLIED_PROBLEMS_MIN
            ),
            "problem_total": SEMANTIC_REPAIR_EXPECTED_PROBLEMS,
        },
        "interpretation_contract": SEMANTIC_REPAIR_INTERPRETATION_CONTRACT,
    }


def _run_semantic_repair_probe(
    *,
    output_path: str | Path,
    prepared: PreparedSemanticRepairProbe,
    generator_client: OpenAICompatibleLLMClient,
    authorization: SemanticRepairProbeAuthorization,
) -> SemanticRepairProbeRun:
    if not isinstance(prepared, PreparedSemanticRepairProbe):
        raise TypeError("prepared semantic-repair probe has an invalid type")
    if not isinstance(authorization, SemanticRepairProbeAuthorization):
        raise TypeError("semantic-repair authorization has an invalid type")
    prepared.validate_trusted()
    authorization.validate(prepared.contract)
    if type(generator_client) is not OpenAICompatibleLLMClient:
        raise TypeError("semantic-repair generator client has an invalid type")
    if generator_client.attempt_observer is not None:
        raise ValueError("semantic-repair generator already has an observer")
    if llm_transport_contract(generator_client) != (
        prepared.contract.generator_transport
    ):
        raise ValueError(
            "semantic-repair generator transport differs from the contract"
        )

    final_path = Path(output_path)
    if final_path.as_posix() != prepared.contract.planned_output_path:
        raise ValueError("semantic-repair output differs from the contract")
    partial_path = final_path.with_name(f"{final_path.name}.partial")
    if final_path.exists() or partial_path.exists():
        raise FileExistsError(
            "semantic-repair output or partial output already exists"
        )
    if not final_path.parent.exists() or not final_path.parent.is_dir():
        raise FileNotFoundError(
            "semantic-repair output directory does not exist"
        )

    contract = prepared.contract
    attempt_outcome_counts = {
        "accepted": 0,
        "parse_error": 0,
        "schema_error": 0,
        "provider_response_error": 0,
        "http_status_error": 0,
        "transport_error": 0,
    }
    reported_usage = {
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
            partial_path,
            _semantic_repair_header(prepared, authorization),
        )
        callback = writer.observe_attempt
        generator_client.attempt_observer = callback
        for case in prepared.cases:
            request, _, expected_repair_body = build_semantic_repair_request(
                prepared, case
            )
            writer.begin_call(case.index)
            payload: dict[str, Any] | None = None
            generation_error: LLMResponseError | None = None
            try:
                payload = generator_client.complete_json(
                    request.system,
                    request.user,
                    request.schema,
                    temperature=contract.generation_temperature,
                    max_tokens=contract.generation_max_tokens,
                    seed=case.effective_seed,
                )
            except LLMResponseError as exc:
                generation_error = exc
            call_record, attempts = writer.finish_call()
            for attempt in attempts:
                attempt_outcome_counts[attempt["outcome"]] += 1
                usage = attempt["usage"]
                if usage["reported"]:
                    reported_usage["attempt_count"] += 1
                    reported_usage["prompt_tokens"] += (
                        usage["prompt_tokens"] or 0
                    )
                    reported_usage["completion_tokens"] += (
                        usage["completion_tokens"] or 0
                    )
                    reported_usage["total_tokens"] += (
                        usage["total_tokens"] or 0
                    )
            if attempts[0]["request_body"] != expected_repair_body:
                raise SemanticRepairProbeValidationError(
                    "hosted repair request differs from preparation"
                )
            if attempts[0]["effective_seed"] != case.effective_seed:
                raise SemanticRepairProbeValidationError(
                    "hosted repair request changed the original seed"
                )
            wire_usable = call_record["final_outcome"] == "transport_accepted"
            if wire_usable is (generation_error is not None):
                raise SemanticRepairProbeValidationError(
                    "generator return differs from raw call diagnostics"
                )
            wire = {
                "usable": wire_usable,
                "terminal_outcome": call_record["final_outcome"],
                "final_attempt_outcome": call_record["final_attempt_outcome"],
                "error_type": _bounded_error_type(generation_error),
            }
            draft: DecompositionDraft | None = None
            if not wire_usable:
                decode = {
                    "status": "not_run",
                    "error_type": None,
                    "draft": None,
                }
                corrected_application = None
            else:
                if payload is None:
                    raise SemanticRepairProbeValidationError(
                        "accepted semantic-repair call returned no payload"
                    )
                try:
                    decoded = decode_policy_payload(
                        case.source_case.source_case.probe_observation,
                        case.action,
                        payload,
                    )
                    if not isinstance(decoded, DecompositionDraft):
                        raise PolicyPayloadError(
                            "semantic-repair payload is not a decomposition"
                        )
                    draft = decoded
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
                        prepared, case, draft
                    )
            record = {
                "schema_version": SEMANTIC_REPAIR_CASE_VERSION,
                **case.manifest_record(),
                "feedback": case.feedback,
                "repair_call_record_id": call_record["record_id"],
                "wire": wire,
                "decode": decode,
                "corrected_application": corrected_application,
            }
            case_records.append(record)
            writer.write_case(record)

        summary = _new_summary(
            contract,
            case_records,
            attempt_count=writer.attempt_count,
            call_count=writer.call_count,
            attempt_outcome_counts=attempt_outcome_counts,
            reported_usage=reported_usage,
        )
        writer.finish(summary)
        if generator_client.attempt_observer is callback:
            generator_client.attempt_observer = None
        validated = read_semantic_repair_probe_artifact(
            partial_path,
            prepared=prepared,
            authorization=authorization,
        )
        if validated != summary:
            raise SemanticRepairProbeValidationError(
                "semantic-repair summary changed during validation"
            )
        os.link(partial_path, final_path)
        partial_path.unlink()
        return SemanticRepairProbeRun(
            output_path=final_path,
            contract_sha256=contract.sha256,
            case_count=len(case_records),
            wire_usable_count=summary["wire_usable_count"],
            decoded_count=summary["decoded_count"],
            symbolic_applied_count=summary["symbolic_applied_count"],
            applied_problem_count=summary["applied_problem_count"],
        )
    finally:
        if generator_client.attempt_observer is callback:
            generator_client.attempt_observer = None
        if writer is not None:
            writer.abort()


_MANIFEST_KEYS = frozenset({
    "index",
    "e24_case_index",
    "e25_case_index",
    "episode_key",
    "problem_id",
    "action",
    "probe_observation_sha256",
    "e24_draft_sha256",
    "effective_seed",
    "source_decision_record_id",
    "source_generator_call_id",
    "corrected_reason_code",
    "corrected_coverage_digest",
    "e25_case_record_sha256",
    "e25_corrected_result_sha256",
    "failed_domains",
    "feedback_sha256",
    "base_request_sha256",
    "repair_request_sha256",
})
_HEADER_KEYS = frozenset({
    "schema_version",
    "canonical_serializer",
    "preregistration_id",
    "repair_contract",
    "repair_contract_sha256",
    "request_environment_contract_sha256",
    "scoring_environment_contract_sha256",
    "corrected_profile_sha256",
    "feedback_map_sha256",
    "case_manifest_sha256",
    "base_request_manifest_sha256",
    "repair_request_manifest_sha256",
})
_CASE_KEYS = _MANIFEST_KEYS | frozenset({
    "schema_version",
    "feedback",
    "repair_call_record_id",
    "wire",
    "decode",
    "corrected_application",
})
_SUMMARY_KEYS = frozenset({
    "schema_version",
    "repair_contract_sha256",
    "case_manifest_sha256",
    "base_request_manifest_sha256",
    "repair_request_manifest_sha256",
    "case_count",
    "problem_count",
    "problem_ids",
    "raw_attempt_count",
    "generator_call_count",
    "wire_usable_count",
    "wire_unusable_count",
    "decoded_count",
    "typed_decoded_count",
    "typed_unusable_count",
    "decode_counts",
    "symbolic_applied_count",
    "symbolic_application_counts",
    "applied_problem_count",
    "applied_problem_ids",
    "by_problem",
    "source_reason_counts",
    "corrected_result_reason_counts",
    "reason_migration_counts",
    "raw_attempt_outcome_counts",
    "reported_usage",
    "reported_cost",
    "primary_thresholds",
    "interpretation_contract",
})
_FEEDBACK_KEYS = frozenset({
    "schema_version", "failed_domains", "reason_code", "repair_constraint"
})
_WIRE_KEYS = frozenset({
    "usable", "terminal_outcome", "final_attempt_outcome", "error_type"
})
_DECODE_KEYS = frozenset({"status", "error_type", "draft"})
_DRAFT_KEYS = frozenset({
    "draft_type", "connective", "child_statements", "child_obligations"
})
_OBLIGATION_KEYS = frozenset({
    "variable", "equation", "domain", "objective"
})
_APPLICATION_KEYS = frozenset({
    "profile_id",
    "application_status",
    "reason_code",
    "canonical_children",
    "coverage",
})
_COVERAGE_KEYS = frozenset({"accepted", "context_digest", "reason_code"})
_DECOMPOSE_WIRE_SCHEMA: Mapping[str, object] = MappingProxyType({
    "type": "object",
    "properties": {
        "domains": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 6,
        }
    },
    "required": ["domains"],
})


def _validate_feedback_record(
    value: object,
    *,
    failed_domains: object,
    reason_code: object,
    feedback_sha256: object,
    path: str,
) -> dict[str, object]:
    record = _exact_record(value, _FEEDBACK_KEYS, path)
    if type(failed_domains) is not list:
        raise SemanticRepairProbeValidationError(
            f"{path} manifest domains are invalid"
        )
    try:
        expected = build_semantic_repair_feedback(
            failed_domains, reason_code
        )
    except ValueError as exc:
        raise SemanticRepairProbeValidationError(
            f"{path} is invalid"
        ) from exc
    if record != expected:
        raise SemanticRepairProbeValidationError(
            f"{path} differs from its bounded reason mapping"
        )
    if canonical_json_sha256(record) != feedback_sha256:
        raise SemanticRepairProbeValidationError(
            f"{path} digest differs"
        )
    return record


def _base_body_from_repair(
    repair_body: Mapping[str, object], feedback: Mapping[str, object]
) -> dict[str, object]:
    base = copy.deepcopy(repair_body)
    messages = base.get("messages")
    if (
        type(messages) is not list
        or len(messages) != 2
        or type(messages[1]) is not dict
        or not isinstance(messages[1].get("content"), str)
    ):
        raise SemanticRepairProbeValidationError(
            "semantic-repair request message structure is invalid"
        )
    append_text = _feedback_append_text(feedback)
    content = messages[1]["content"]
    if not content.endswith(append_text):
        raise SemanticRepairProbeValidationError(
            "semantic-repair request lacks the exact feedback append"
        )
    messages[1]["content"] = content[:-len(append_text)]
    _validate_request_delta(base, repair_body, feedback)
    return base


def _validate_attempt_projection(
    attempt: Mapping[str, object],
    wire_schema: Mapping[str, object] = _DECOMPOSE_WIRE_SCHEMA,
    schema_validation_mode: str = SCHEMA_VALIDATION_LEGACY_COERCE_V1,
) -> dict[str, Any] | None:
    outcome = attempt["outcome"]
    usage = attempt["usage"]
    if outcome in {"http_status_error", "transport_error"}:
        if (
            attempt["raw_content"] is not None
            or usage["reported"] is not False
            or usage["raw"] is not None
            or any(
                usage[name] is not None
                for name in (
                    "prompt_tokens", "completion_tokens", "total_tokens"
                )
            )
        ):
            raise SemanticRepairProbeValidationError(
                "HTTP failure has impossible projected content or usage"
            )
        return None
    status = attempt["http_status"]
    if type(status) is not int or not 200 <= status < 300:
        raise SemanticRepairProbeValidationError(
            "content outcome lacks a successful HTTP status"
        )
    raw_body = attempt["raw_response_body"]
    if not isinstance(raw_body, str):
        raise SemanticRepairProbeValidationError(
            "content outcome lacks its raw response body"
        )
    try:
        payload = json.loads(raw_body)
    except (json.JSONDecodeError, TypeError):
        payload = None
    projection = None
    if type(payload) is dict:
        projection = project_openai_compatible_response(payload)
    expected_content = None if projection is None else projection.raw_content
    expected_raw_usage = (
        None
        if projection is None or projection.raw_usage_json is None
        else json.loads(projection.raw_usage_json)
    )
    if attempt["raw_content"] != expected_content:
        raise SemanticRepairProbeValidationError(
            "raw content differs from the response envelope"
        )
    if canonical_json_bytes(usage["raw"]) != canonical_json_bytes(
        expected_raw_usage
    ):
        raise SemanticRepairProbeValidationError(
            "raw usage differs from the response envelope"
        )
    expected_counts = {
        "prompt_tokens": None if projection is None else projection.prompt_tokens,
        "completion_tokens": (
            None if projection is None else projection.completion_tokens
        ),
        "total_tokens": None if projection is None else projection.total_tokens,
    }
    if usage["reported"] is not (expected_raw_usage is not None):
        raise SemanticRepairProbeValidationError(
            "usage presence differs from the response envelope"
        )
    for field_name, expected in expected_counts.items():
        if usage[field_name] != expected:
            raise SemanticRepairProbeValidationError(
                f"{field_name} differs from the response envelope"
            )
    if projection is None or projection.provider_error is not None:
        derived_outcome = "provider_response_error"
        decoded = None
    else:
        if projection.raw_content is None:
            raise AssertionError("provider projection lost its content error")
        derived_outcome, decoded = classify_openai_compatible_json_content(
            projection.raw_content,
            wire_schema,
            schema_validation_mode=schema_validation_mode,
        )
    if attempt["outcome"] != derived_outcome:
        raise SemanticRepairProbeValidationError(
            "attempt outcome differs from its raw response"
        )
    return decoded


def _expected_disposition(
    attempt: Mapping[str, object], transport: ModelTransportContract
) -> str:
    outcome = attempt["outcome"]
    if outcome == "accepted":
        return "return"
    if outcome == "http_status_error":
        status = attempt["http_status"]
        retryable = status == 429 or status >= 500
        if (
            retryable
            and attempt["backoff_retry_index"]
            < transport.backoff_retry_budget
        ):
            return "backoff_retry"
        return "raise"
    if attempt["parse_retry_index"] < transport.parse_retry_budget:
        return "parse_retry"
    return "raise"


def _validate_call_transcript(
    attempts: list[dict[str, object]],
    call: Mapping[str, object],
    *,
    initial_body: Mapping[str, object],
    initial_seed: int,
    transport: ModelTransportContract,
    wire_schema: Mapping[str, object] = _DECOMPOSE_WIRE_SCHEMA,
    schema_validation_mode: str = SCHEMA_VALIDATION_LEGACY_COERCE_V1,
) -> dict[str, Any] | None:
    _validate_attempt_chain(attempts)
    validate_call_record(call, attempts=attempts)
    expected_body = copy.deepcopy(initial_body)
    final_payload: dict[str, Any] | None = None
    for attempt in attempts:
        if attempt["parse_retry_index"] > transport.parse_retry_budget:
            raise SemanticRepairProbeValidationError(
                "semantic-repair call exceeds parse retry budget"
            )
        if attempt["backoff_retry_index"] > transport.backoff_retry_budget:
            raise SemanticRepairProbeValidationError(
                "semantic-repair call exceeds backoff retry budget"
            )
        if canonical_json_bytes(attempt["request_body"]) != (
            canonical_json_bytes(expected_body)
        ):
            raise SemanticRepairProbeValidationError(
                "semantic-repair retry request body differs"
            )
        decoded = _validate_attempt_projection(
            attempt,
            wire_schema,
            schema_validation_mode,
        )
        disposition = _expected_disposition(attempt, transport)
        if attempt["disposition"] != disposition:
            raise SemanticRepairProbeValidationError(
                "semantic-repair retry disposition differs"
            )
        if attempt is attempts[-1]:
            final_payload = decoded
        if disposition == "parse_retry":
            append_openai_compatible_parse_retry(
                expected_body, attempt["raw_content"]
            )
            expected_body["seed"] = (
                initial_seed + attempt["parse_retry_index"] + 1
            )
    if call["final_outcome"] == "transport_accepted":
        if final_payload is None:
            raise SemanticRepairProbeValidationError(
                "accepted semantic-repair call lacks a decoded payload"
            )
        return final_payload
    return None


def _validate_obligation_record(value: object, path: str) -> None:
    record = _exact_record(value, _OBLIGATION_KEYS, path)
    for field_name in ("variable", "equation", "domain"):
        _nonempty_text(record[field_name], f"{path}.{field_name}")
    if record["objective"] not in {item.value for item in GoalObjective}:
        raise SemanticRepairProbeValidationError(
            f"{path}.objective is invalid"
        )


def _validate_draft_record(
    value: object, payload: Mapping[str, object], path: str
) -> None:
    record = _exact_record(value, _DRAFT_KEYS, path)
    if (
        record["draft_type"] != "decomposition"
        or record["connective"] != "exhaustive_case"
    ):
        raise SemanticRepairProbeValidationError(
            f"{path} type or connective is invalid"
        )
    domains = payload.get("domains")
    statements = record["child_statements"]
    obligations = record["child_obligations"]
    if (
        type(domains) is not list
        or type(statements) is not list
        or type(obligations) is not list
        or len(statements) != len(domains)
        or len(obligations) != len(domains)
    ):
        raise SemanticRepairProbeValidationError(
            f"{path} child cardinalities are invalid"
        )
    if statements != [f"case {index}" for index in range(1, len(domains) + 1)]:
        raise SemanticRepairProbeValidationError(
            f"{path} child statements are invalid"
        )
    for index, obligation in enumerate(obligations):
        _validate_obligation_record(obligation, f"{path}.children[{index}]")
        if (
            obligation["domain"] != domains[index].strip()
            or obligation["objective"] != GoalObjective.BRANCH.value
        ):
            raise SemanticRepairProbeValidationError(
                f"{path} domains differ from the raw payload"
            )
    variables = {item["variable"] for item in obligations}
    equations = {item["equation"] for item in obligations}
    if len(variables) != 1 or len(equations) != 1:
        raise SemanticRepairProbeValidationError(
            f"{path} children do not share one target obligation"
        )


def _validate_application_record(
    value: object,
    *,
    profile: DomainAdapterProfileBinding,
    path: str,
) -> str:
    record = _exact_record(value, _APPLICATION_KEYS, path)
    if record["profile_id"] != profile.domain_grammar_profile_id:
        raise SemanticRepairProbeValidationError(
            f"{path} does not use the corrected profile"
        )
    status = record["application_status"]
    if status not in {"applied", "algebra_rejected"}:
        raise SemanticRepairProbeValidationError(f"{path} status is invalid")
    reason = record["reason_code"]
    children = record["canonical_children"]
    if status == "applied":
        if reason is not None or type(children) is not list or not children:
            raise SemanticRepairProbeValidationError(
                f"{path} applied result is invalid"
            )
        for index, child in enumerate(children):
            _validate_obligation_record(child, f"{path}.children[{index}]")
    else:
        if reason not in profile.rejection_codes or children is not None:
            raise SemanticRepairProbeValidationError(
                f"{path} rejection is not bounded"
            )
    coverage = _exact_record(
        record["coverage"], _COVERAGE_KEYS, f"{path}.coverage"
    )
    if type(coverage["accepted"]) is not bool or not _is_sha256(
        coverage["context_digest"]
    ):
        raise SemanticRepairProbeValidationError(
            f"{path} coverage record is invalid"
        )
    if coverage["accepted"] is not (status == "applied") or (
        coverage["reason_code"] != reason
    ):
        raise SemanticRepairProbeValidationError(
            f"{path} coverage differs from application"
        )
    return status


def _read_semantic_repair_probe_artifact(
    path: Path,
    *,
    prepared: PreparedSemanticRepairProbe,
    authorization: SemanticRepairProbeAuthorization,
) -> dict[str, object]:
    if not isinstance(prepared, PreparedSemanticRepairProbe):
        raise TypeError("prepared artifact validator has an invalid type")
    if not isinstance(authorization, SemanticRepairProbeAuthorization):
        raise TypeError("artifact authorization has an invalid type")
    prepared.validate_trusted()
    authorization.validate(prepared.contract)
    records = _read_canonical_records(path)
    if len(records) < 2:
        raise SemanticRepairProbeValidationError(
            "semantic-repair artifact is incomplete"
        )
    header = _exact_record(records[0], _HEADER_KEYS, "semantic-repair header")
    if header["schema_version"] != SEMANTIC_REPAIR_HEADER_VERSION:
        raise SemanticRepairProbeValidationError(
            "semantic-repair header schema is invalid"
        )
    if header["canonical_serializer"] != CANONICAL_JSON_VERSION:
        raise SemanticRepairProbeValidationError(
            "semantic-repair header serializer is invalid"
        )
    if header["preregistration_id"] != authorization.preregistration_id:
        raise SemanticRepairProbeValidationError(
            "artifact preregistration id differs from authorization"
        )
    contract = SemanticRepairProbeContract.from_record(
        header["repair_contract"]
    )
    if header["repair_contract_sha256"] != contract.sha256:
        raise SemanticRepairProbeValidationError(
            "semantic-repair header contract digest differs"
        )
    header_checks = {
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
        "repair_request_manifest_sha256": (
            contract.repair_request_manifest_sha256
        ),
    }
    for field_name, expected in header_checks.items():
        if header[field_name] != expected:
            raise SemanticRepairProbeValidationError(
                f"semantic-repair header {field_name} differs"
            )
    if contract.to_record() != prepared.contract.to_record():
        raise SemanticRepairProbeValidationError(
            "artifact contract differs from strict preparation"
        )

    summary = _exact_record(
        records[-1], _SUMMARY_KEYS, "semantic-repair summary"
    )
    if summary["schema_version"] != SEMANTIC_REPAIR_SUMMARY_VERSION:
        raise SemanticRepairProbeValidationError(
            "semantic-repair summary schema is invalid"
        )
    pending_attempts: list[dict[str, object]] = []
    pending_call: dict[str, object] | None = None
    seen_call_ids: set[str] = set()
    case_records: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    raw_attempt_count = 0
    call_count = 0
    attempt_outcome_counts = {
        "accepted": 0,
        "parse_error": 0,
        "schema_error": 0,
        "provider_response_error": 0,
        "http_status_error": 0,
        "transport_error": 0,
    }
    reported_usage = {
        "attempt_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for record_index, value in enumerate(records[1:-1], start=1):
        schema = value.get("schema_version")
        if schema == DIAGNOSTIC_ATTEMPT_VERSION:
            if pending_call is not None:
                raise SemanticRepairProbeValidationError(
                    "attempt appears before the preceding case"
                )
            validate_attempt_record(value)
            if value["role"] != "generator":
                raise SemanticRepairProbeValidationError(
                    "semantic-repair artifact contains a non-generator attempt"
                )
            if pending_attempts and value["call_id"] != pending_attempts[0]["call_id"]:
                raise SemanticRepairProbeValidationError(
                    "attempt changed call before a call record"
                )
            pending_attempts.append(value)
            _validate_attempt_chain(pending_attempts)
            raw_attempt_count += 1
            attempt_outcome_counts[value["outcome"]] += 1
            usage = value["usage"]
            if usage["reported"]:
                reported_usage["attempt_count"] += 1
                reported_usage["prompt_tokens"] += usage["prompt_tokens"] or 0
                reported_usage["completion_tokens"] += (
                    usage["completion_tokens"] or 0
                )
                reported_usage["total_tokens"] += usage["total_tokens"] or 0
            continue
        if schema == DIAGNOSTIC_CALL_VERSION:
            if pending_call is not None or not pending_attempts:
                raise SemanticRepairProbeValidationError(
                    "semantic-repair call record is out of sequence"
                )
            validate_call_record(value, attempts=pending_attempts)
            if value["role"] != "generator":
                raise SemanticRepairProbeValidationError(
                    "semantic-repair artifact contains a non-generator call"
                )
            if value["call_id"] in seen_call_ids:
                raise SemanticRepairProbeValidationError(
                    "semantic-repair call id is repeated"
                )
            seen_call_ids.add(value["call_id"])
            pending_call = value
            call_count += 1
            continue
        if schema != SEMANTIC_REPAIR_CASE_VERSION:
            raise SemanticRepairProbeValidationError(
                f"semantic-repair record {record_index} is unsupported"
            )
        if pending_call is None or not pending_attempts:
            raise SemanticRepairProbeValidationError(
                "semantic-repair case lacks its raw generator call"
            )
        case_index = len(case_records)
        record = _exact_record(
            value, _CASE_KEYS, f"semantic-repair case {case_index}"
        )
        if record["index"] != case_index:
            raise SemanticRepairProbeValidationError(
                "semantic-repair cases are not in canonical order"
            )
        for field_name in ("e24_case_index", "e25_case_index", "effective_seed"):
            _nonnegative_integer(
                record[field_name],
                f"semantic-repair case {case_index}.{field_name}",
            )
        for field_name in ("episode_key", "problem_id", "source_decision_record_id"):
            _nonempty_text(
                record[field_name],
                f"semantic-repair case {case_index}.{field_name}",
            )
        action = deserialize_action(
            record["action"], f"semantic-repair case {case_index}.action"
        )
        if action.verb is not Verb.DECOMPOSE:
            raise SemanticRepairProbeValidationError(
                "semantic-repair action is not DECOMPOSE"
            )
        for field_name in (
            "probe_observation_sha256",
            "e24_draft_sha256",
            "corrected_coverage_digest",
            "e25_case_record_sha256",
            "e25_corrected_result_sha256",
            "feedback_sha256",
            "base_request_sha256",
            "repair_request_sha256",
        ):
            if not _is_sha256(record[field_name]):
                raise SemanticRepairProbeValidationError(
                    f"semantic-repair case {case_index}.{field_name} is invalid"
                )
        source_call_id = record["source_generator_call_id"]
        if (
            not isinstance(source_call_id, str)
            or len(source_call_id) != 32
            or set(source_call_id) - _SHA256_CHARS
        ):
            raise SemanticRepairProbeValidationError(
                "semantic-repair source generator call id is invalid"
            )
        feedback = _validate_feedback_record(
            record["feedback"],
            failed_domains=record["failed_domains"],
            reason_code=record["corrected_reason_code"],
            feedback_sha256=record["feedback_sha256"],
            path=f"semantic-repair case {case_index}.feedback",
        )
        if record["repair_call_record_id"] != pending_call["record_id"]:
            raise SemanticRepairProbeValidationError(
                "semantic-repair case does not resolve its call record"
            )
        repair_body = pending_attempts[0]["request_body"]
        if canonical_json_sha256(repair_body) != record["repair_request_sha256"]:
            raise SemanticRepairProbeValidationError(
                "semantic-repair request digest differs from first attempt"
            )
        base_body = _base_body_from_repair(repair_body, feedback)
        if canonical_json_sha256(base_body) != record["base_request_sha256"]:
            raise SemanticRepairProbeValidationError(
                "semantic-repair base request digest differs"
            )
        if pending_attempts[0]["effective_seed"] != record["effective_seed"]:
            raise SemanticRepairProbeValidationError(
                "semantic-repair case seed differs from first attempt"
            )
        payload = _validate_call_transcript(
            pending_attempts,
            pending_call,
            initial_body=repair_body,
            initial_seed=record["effective_seed"],
            transport=contract.generator_transport,
        )
        wire = _exact_record(
            record["wire"], _WIRE_KEYS, f"semantic-repair case {case_index}.wire"
        )
        expected_usable = pending_call["final_outcome"] == "transport_accepted"
        if (
            type(wire["usable"]) is not bool
            or wire["usable"] is not expected_usable
            or wire["terminal_outcome"] != pending_call["final_outcome"]
            or wire["final_attempt_outcome"]
            != pending_call["final_attempt_outcome"]
        ):
            raise SemanticRepairProbeValidationError(
                "semantic-repair wire classification differs from call"
            )
        if expected_usable:
            if wire["error_type"] is not None:
                raise SemanticRepairProbeValidationError(
                    "usable semantic-repair wire has an error"
                )
        else:
            if wire["error_type"] != "LLMResponseError":
                raise SemanticRepairProbeValidationError(
                    "unusable semantic-repair wire error type is invalid"
                )
        decode = _exact_record(
            record["decode"],
            _DECODE_KEYS,
            f"semantic-repair case {case_index}.decode",
        )
        status = decode["status"]
        if status not in {"decoded", "payload_error", "not_run"}:
            raise SemanticRepairProbeValidationError(
                "semantic-repair decode status is invalid"
            )
        if case_index >= len(prepared.cases):
            raise SemanticRepairProbeValidationError(
                "artifact has more cases than strict preparation"
            )
        expected_case = prepared.cases[case_index]
        decoded_draft: DecompositionDraft | None = None
        if payload is None:
            if (
                status != "not_run"
                or decode["error_type"] is not None
                or decode["draft"] is not None
                or record["corrected_application"] is not None
            ):
                raise SemanticRepairProbeValidationError(
                    "semantic-repair not-run decode has invalid staging"
                )
        else:
            try:
                candidate = decode_policy_payload(
                    expected_case.source_case.source_case.probe_observation,
                    expected_case.action,
                    payload,
                )
                if not isinstance(candidate, DecompositionDraft):
                    raise PolicyPayloadError(
                        "decoded payload is not a decomposition"
                    )
                decoded_draft = candidate
            except PolicyPayloadError:
                if (
                    status != "payload_error"
                    or decode["error_type"] != "PolicyPayloadError"
                    or decode["draft"] is not None
                    or record["corrected_application"] is not None
                ):
                    raise SemanticRepairProbeValidationError(
                        "semantic-repair payload-error staging differs"
                    )
            else:
                if (
                    status != "decoded"
                    or decode["error_type"] is not None
                    or decode["draft"] is None
                    or record["corrected_application"] is None
                ):
                    raise SemanticRepairProbeValidationError(
                        "semantic-repair decoded staging differs"
                    )
                _validate_draft_record(
                    decode["draft"],
                    payload,
                    f"semantic-repair case {case_index}.decode.draft",
                )
                _validate_application_record(
                    record["corrected_application"],
                    profile=prepared.source_rescore.contract.corrected_profile,
                    path=(
                        f"semantic-repair case {case_index}."
                        "corrected_application"
                    ),
                )
                if decode["draft"] != _serialize_decomposition_draft(
                    decoded_draft
                ):
                    raise SemanticRepairProbeValidationError(
                        "artifact draft differs from raw payload decoding"
                    )
                expected_application = _score_repair_draft(
                    prepared, expected_case, decoded_draft
                )
                if record["corrected_application"] != expected_application:
                    raise SemanticRepairProbeValidationError(
                        "artifact corrected score differs from recomputation"
                    )
        manifest_record = {key: record[key] for key in _MANIFEST_KEYS}
        if manifest_record != expected_case.manifest_record():
            raise SemanticRepairProbeValidationError(
                "artifact case differs from strict preparation"
            )
        if canonical_json_bytes(base_body) != expected_case.base_request_body_bytes:
            raise SemanticRepairProbeValidationError(
                "artifact base request differs from strict preparation"
            )
        if canonical_json_bytes(repair_body) != (
            expected_case.repair_request_body_bytes
        ):
            raise SemanticRepairProbeValidationError(
                "artifact repair request differs from strict preparation"
            )
        manifest.append(manifest_record)
        case_records.append(record)
        pending_attempts = []
        pending_call = None

    if pending_attempts or pending_call is not None:
        raise SemanticRepairProbeValidationError(
            "semantic-repair artifact ends with an unresolved call"
        )
    if len(case_records) != contract.case_count:
        raise SemanticRepairProbeValidationError(
            "semantic-repair artifact case count differs from contract"
        )
    if canonical_json_sha256(manifest) != contract.case_manifest_sha256:
        raise SemanticRepairProbeValidationError(
            "semantic-repair artifact manifest differs from contract"
        )
    base_manifest = [
        {"index": row["index"], "base_request_sha256": row["base_request_sha256"]}
        for row in case_records
    ]
    repair_manifest = [
        {
            "index": row["index"],
            "repair_request_sha256": row["repair_request_sha256"],
        }
        for row in case_records
    ]
    if canonical_json_sha256(base_manifest) != (
        contract.base_request_manifest_sha256
    ) or canonical_json_sha256(repair_manifest) != (
        contract.repair_request_manifest_sha256
    ):
        raise SemanticRepairProbeValidationError(
            "semantic-repair request manifests differ from contract"
        )
    expected_summary = _new_summary(
        contract,
        case_records,
        attempt_count=raw_attempt_count,
        call_count=call_count,
        attempt_outcome_counts=attempt_outcome_counts,
        reported_usage=reported_usage,
    )
    if summary != expected_summary:
        raise SemanticRepairProbeValidationError(
            "semantic-repair summary differs from validated records"
        )
    return dict(summary)


__all__ = [
    "SEMANTIC_REPAIR_CASE_VERSION",
    "SEMANTIC_REPAIR_CONTRACT_VERSION",
    "SEMANTIC_REPAIR_EXPECTED_CASES",
    "SEMANTIC_REPAIR_EXPECTED_PROBLEMS",
    "SEMANTIC_REPAIR_EXPECTED_SOURCE_DECISIONS",
    "SEMANTIC_REPAIR_FEEDBACK_CONSTRAINTS",
    "SEMANTIC_REPAIR_FEEDBACK_MAP_VERSION",
    "SEMANTIC_REPAIR_FEEDBACK_SCHEMA_VERSION",
    "SEMANTIC_REPAIR_FEEDBACK_SUFFIX",
    "SEMANTIC_REPAIR_HEADER_VERSION",
    "SEMANTIC_REPAIR_INTERPRETATION_CONTRACT",
    "SEMANTIC_REPAIR_INPUT_USD_PER_MILLION",
    "SEMANTIC_REPAIR_OUTPUT_USD_PER_MILLION",
    "SEMANTIC_REPAIR_PRIMARY_APPLIED_MIN",
    "SEMANTIC_REPAIR_PRIMARY_APPLIED_PROBLEMS_MIN",
    "SEMANTIC_REPAIR_PRIMARY_TYPED_DECODED_MIN",
    "SEMANTIC_REPAIR_REQUEST_BRIDGE_CONTRACT",
    "SEMANTIC_REPAIR_SELECTION_CONTRACT",
    "SEMANTIC_REPAIR_SUMMARY_VERSION",
    "PreparedSemanticRepairProbe",
    "SemanticRepairProbeAuthorization",
    "SemanticRepairProbeCase",
    "SemanticRepairProbeContract",
    "SemanticRepairProbeRun",
    "SemanticRepairProbeValidationError",
    "build_semantic_repair_feedback",
    "build_semantic_repair_request",
    "prepare_semantic_repair_probe",
    "read_semantic_repair_probe_artifact",
    "run_semantic_repair_probe",
    "semantic_repair_feedback_map_record",
    "semantic_repair_feedback_map_sha256",
]
