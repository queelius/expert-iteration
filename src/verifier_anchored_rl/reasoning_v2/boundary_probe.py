"""Boundary-only DECOMPOSE generation over sealed E24 decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .actions import DecompositionDraft
from .contracts import canonical_json_bytes, canonical_json_sha256
from .lenses import LensRequest, build_policy_lens_request
from .math_adapter import (
    CORRECTED_DOMAIN_GRAMMAR_PROFILE,
    _AdapterError,
    _parse_domain,
    _parse_scalar,
    _sympy,
)
from .model import (
    Action,
    Connective,
    Goal,
    GoalObjective,
    GoalObligation,
    State,
    Verb,
    get,
)


BOUNDARY_PAYLOAD_CONTRACT = "decomposition-boundaries-v1"
BOUNDARY_PAYLOAD_SCHEMA_VERSION = "decomposition-boundary-payload-v1"
BOUNDARY_MIN_ITEMS = 1
BOUNDARY_MAX_ITEMS = 5
BOUNDARY_INSTRUCTION = (
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
BOUNDARY_SCHEMA: Mapping[str, object] = {
    "type": "object",
    "properties": {
        "boundaries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": BOUNDARY_MIN_ITEMS,
            "maxItems": BOUNDARY_MAX_ITEMS,
        }
    },
    "required": ["boundaries"],
    "additionalProperties": False,
}


class BoundaryPayloadError(ValueError):
    """Raised when a boundary payload cannot be trusted."""


@dataclass(frozen=True, slots=True)
class BoundaryPayload:
    boundaries: tuple[str, ...]
    canonical_values: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": BOUNDARY_PAYLOAD_SCHEMA_VERSION,
            "boundaries": list(self.boundaries),
            "canonical_values": list(self.canonical_values),
        }


@dataclass(frozen=True, slots=True)
class BoundaryConstruction:
    payload: BoundaryPayload
    parent_obligation: GoalObligation
    draft: DecompositionDraft

    @property
    def constructed_domains(self) -> tuple[str, ...]:
        return tuple(child.domain for child in self.draft.child_obligations)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": "decomposition-boundary-construction-v1",
            "payload": self.payload.to_record(),
            "parent_obligation": {
                "variable": self.parent_obligation.variable,
                "equation": self.parent_obligation.equation,
                "domain": self.parent_obligation.domain,
                "objective": self.parent_obligation.objective.value,
            },
            "constructed_domains": list(self.constructed_domains),
        }


def boundary_schema_record() -> dict[str, object]:
    return json.loads(canonical_json_bytes(BOUNDARY_SCHEMA))


def boundary_schema_sha256() -> str:
    return canonical_json_sha256(boundary_schema_record())


def boundary_payload_schema_record() -> dict[str, object]:
    """Return the exact public wire schema for a boundary payload."""
    return boundary_schema_record()


def boundary_payload_schema_sha256() -> str:
    """Return the digest of the exact public boundary wire schema."""
    return boundary_schema_sha256()


def _parent_goal(state: State, action: Action) -> Goal:
    if not isinstance(state, State) or not isinstance(action, Action):
        raise TypeError("boundary construction inputs have invalid types")
    if action.verb is not Verb.DECOMPOSE:
        raise BoundaryPayloadError("boundary payload requires DECOMPOSE")
    try:
        node = get(state.graph, action.target)
    except KeyError as exc:
        raise BoundaryPayloadError("boundary target is missing") from exc
    if not isinstance(node, Goal):
        raise BoundaryPayloadError("boundary target is not a Goal")
    return node


def decode_boundary_payload(
    payload: object,
) -> BoundaryPayload:
    if type(payload) is not dict or set(payload) != {"boundaries"}:
        raise BoundaryPayloadError(
            "boundary payload must contain exactly the boundaries field"
        )
    raw = payload["boundaries"]
    if type(raw) is not list or not (
        BOUNDARY_MIN_ITEMS <= len(raw) <= BOUNDARY_MAX_ITEMS
    ):
        raise BoundaryPayloadError("boundary array has invalid cardinality")
    sp = _sympy()
    lexemes: list[str] = []
    values: list[Any] = []
    for value in raw:
        if not isinstance(value, str):
            raise BoundaryPayloadError("boundary values must be strings")
        try:
            value.encode("ascii")
            parsed = _parse_scalar(value)
        except (UnicodeEncodeError, _AdapterError) as exc:
            raise BoundaryPayloadError("malformed_scalar") from exc
        if values:
            relation = values[-1] < parsed
            if relation is not sp.S.true:
                if parsed == values[-1]:
                    raise BoundaryPayloadError("duplicate_boundary")
                raise BoundaryPayloadError("boundaries_not_increasing")
        lexemes.append(value)
        values.append(parsed)
    return BoundaryPayload(
        boundaries=tuple(lexemes),
        canonical_values=tuple(sp.sstr(value) for value in values),
    )


def _render_cells(
    parent_obligation: GoalObligation,
    payload: BoundaryPayload,
) -> tuple[str, ...]:
    sp = _sympy()
    parent = _parse_domain(
        parent_obligation.variable,
        parent_obligation.domain,
        CORRECTED_DOMAIN_GRAMMAR_PROFILE,
    )
    if not isinstance(parent, sp.Interval):
        raise BoundaryPayloadError("unsupported_parent_domain")
    values = tuple(_parse_scalar(value) for value in payload.boundaries)
    for value in values:
        if parent.contains(value) is not sp.S.true:
            raise BoundaryPayloadError("boundary_outside_parent_domain")
        if value == parent.start or value == parent.end:
            raise BoundaryPayloadError("boundary_not_strictly_interior")
    variable = parent_obligation.variable
    boundary_lexemes = payload.boundaries
    parent_start = None if parent.start is sp.S.NegativeInfinity else sp.sstr(
        parent.start
    )
    parent_end = None if parent.end is sp.S.Infinity else sp.sstr(parent.end)
    domains: list[str] = []
    first = boundary_lexemes[0]
    if parent_start is None:
        domains.append(f"{variable} < {first}")
    else:
        left = "<" if parent.left_open else "<="
        domains.append(f"{parent_start} {left} {variable} < {first}")
    for lower, upper in zip(boundary_lexemes, boundary_lexemes[1:]):
        domains.append(f"{lower} <= {variable} < {upper}")
    last = boundary_lexemes[-1]
    if parent_end is None:
        domains.append(f"{variable} >= {last}")
    else:
        right = "<" if parent.right_open else "<="
        domains.append(f"{last} <= {variable} {right} {parent_end}")
    return tuple(domains)


def construct_boundary_decomposition(
    state: State,
    action: Action,
    payload: BoundaryPayload,
) -> DecompositionDraft:
    if not isinstance(payload, BoundaryPayload):
        raise TypeError("decoded boundary payload has an invalid type")
    validated = decode_boundary_payload(
        {"boundaries": list(payload.boundaries)}
    )
    if validated != payload:
        raise BoundaryPayloadError(
            "decoded boundary payload metadata is inconsistent"
        )
    parent = _parent_goal(state, action).obligation
    domains = _render_cells(parent, payload)
    obligations = tuple(
        GoalObligation(
            variable=parent.variable,
            equation=parent.equation,
            domain=domain,
            objective=GoalObjective.BRANCH,
        )
        for domain in domains
    )
    draft = DecompositionDraft(
        connective=Connective.EXHAUSTIVE_CASE,
        child_statements=tuple(
            f"case {index}" for index in range(1, len(obligations) + 1)
        ),
        child_obligations=obligations,
    )
    return draft


def construct_boundary_details(
    state: State,
    action: Action,
    payload: BoundaryPayload,
) -> BoundaryConstruction:
    parent = _parent_goal(state, action).obligation
    draft = construct_boundary_decomposition(state, action, payload)
    return BoundaryConstruction(payload, parent, draft)


def build_boundary_lens_request(
    observation: Mapping[str, object],
    action: Action,
) -> tuple[LensRequest, LensRequest]:
    if action.verb is not Verb.DECOMPOSE:
        raise ValueError("boundary lens requires DECOMPOSE")
    source = build_policy_lens_request(observation, action)
    try:
        prefix, source_instruction = source.user.split("\n\n", 1)
    except ValueError as exc:
        raise ValueError("source DECOMPOSE request is malformed") from exc
    if not source_instruction.endswith("\nReturn only one JSON object."):
        raise ValueError("source DECOMPOSE suffix is malformed")
    boundary_user = prefix + "\n\n" + BOUNDARY_INSTRUCTION
    return source, LensRequest(
        system=source.system,
        user=boundary_user,
        schema=boundary_schema_record(),
    )


__all__ = [
    "BOUNDARY_INSTRUCTION",
    "BOUNDARY_MAX_ITEMS",
    "BOUNDARY_MIN_ITEMS",
    "BOUNDARY_PAYLOAD_CONTRACT",
    "BOUNDARY_PAYLOAD_SCHEMA_VERSION",
    "BOUNDARY_SCHEMA",
    "BoundaryConstruction",
    "BoundaryPayload",
    "BoundaryPayloadError",
    "boundary_payload_schema_record",
    "boundary_payload_schema_sha256",
    "boundary_schema_record",
    "boundary_schema_sha256",
    "build_boundary_lens_request",
    "construct_boundary_decomposition",
    "construct_boundary_details",
    "decode_boundary_payload",
]


# Hosted E28 probe harness. The boundary decoder and constructor above remain
# usable without importing any hosted transport or artifact machinery.
import os
from collections import Counter
from dataclasses import field
from types import MappingProxyType

from ..llm_client import (
    LLMResponseError,
    OpenAICompatibleLLMClient,
    SCHEMA_VALIDATION_STRICT_V1,
    build_openai_compatible_json_body,
)
from .casework_source import CaseworkProblemSource
from .contracts import (
    CANONICAL_JSON_VERSION,
    ModelIdentity,
    ModelTransportContract,
)
from .domain_rescore import score_decomposition_draft
from .key_aligned_repair import (
    KEY_ALIGNED_REPAIR_CASE_VERSION,
    KeyAlignedRepairProbeAuthorization,
    KeyAlignedRepairProbeCase,
    KeyAlignedRepairProbeValidationError,
    PreparedKeyAlignedRepairProbe,
    prepare_key_aligned_repair_probe,
    read_key_aligned_repair_probe_artifact,
)
from .lenses import LENS_CONTRACT_VERSION, llm_transport_contract
from .math_adapter import (
    CaseworkMathVerifier,
    validate_decomposition_coverage_v11,
)
from .microprobe import file_sha256
from .semantic_repair import (
    SemanticRepairProbeAuthorization,
    SemanticRepairProbeValidationError,
    _SemanticRepairArtifactWriter,
    _bounded_error_type,
    _cost_basis_record,
    _read_canonical_records,
    _reported_cost_record,
    _serialize_decomposition_draft,
)


BOUNDARY_PROBE_CONTRACT_VERSION = "decomposition-boundary-probe-contract-v1"
BOUNDARY_PROBE_HEADER_VERSION = "decomposition-boundary-probe-header-v1"
BOUNDARY_PROBE_CASE_VERSION = "decomposition-boundary-probe-case-v1"
BOUNDARY_PROBE_SUMMARY_VERSION = "decomposition-boundary-probe-summary-v1"
BOUNDARY_LENS_CONTRACT_VERSION = "typed-claim-lens-v2.2-boundaries"
BOUNDARY_CONSTRUCTOR_VERSION = "decomposition-boundary-constructor-v1"
BOUNDARY_ORACLE_VERSION = "decomposition-boundary-oracle-manifest-v1"
BOUNDARY_SELECTION_CONTRACT = "all-33-sealed-e24-decompose-cases-v1"
BOUNDARY_REQUEST_BRIDGE_CONTRACT = (
    "byte-exact-e24-request-with-only-decompose-instruction-and-schema-replaced-v1"
)
BOUNDARY_INTERPRETATION_CONTRACT = (
    "cross-version-generator-diagnostic-not-trajectory-reward-or-training-v1"
)
BOUNDARY_EXPECTED_SOURCE_DECISIONS = 66
BOUNDARY_EXPECTED_CASES = 33
BOUNDARY_EXPECTED_PROBLEMS = 5
BOUNDARY_PRIMARY_TYPED_DECODED_MIN = 30
BOUNDARY_PRIMARY_APPLIED_MIN = 17
BOUNDARY_PRIMARY_APPLIED_PROBLEMS_MIN = 4

_BOUNDARY_SHA256_CHARS = frozenset("0123456789abcdef")
_BOUNDARY_PREPARED_TRUST_SEAL = object()


class BoundaryDecompositionProbeValidationError(ValueError):
    """Raised when an E28 contract, preparation, or artifact is invalid."""


def _boundary_is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _BOUNDARY_SHA256_CHARS
    )


def _boundary_path_record(path: str | Path) -> str:
    value = Path(path).as_posix()
    if not value or value.endswith(".partial") or not value.endswith(".jsonl"):
        raise ValueError("boundary output path must name a final JSONL artifact")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("boundary output path must be ASCII") from exc
    return value


def _boundary_instruction_sha256() -> str:
    return canonical_json_sha256({"instruction": BOUNDARY_INSTRUCTION})


def _boundary_constructor_record() -> dict[str, object]:
    return {
        "schema_version": BOUNDARY_CONSTRUCTOR_VERSION,
        "parent_profile": {
            "profile_id": CORRECTED_DOMAIN_GRAMMAR_PROFILE.profile_id,
            "bounded_comparator_pattern": (
                CORRECTED_DOMAIN_GRAMMAR_PROFILE.bounded_comparator_pattern
            ),
            "implementation_id": CORRECTED_DOMAIN_GRAMMAR_PROFILE.implementation_id,
        },
        "scalar_parser": "casework-exact-finite-rational-v1",
        "topology": "parent-intersection-left-open-then-half-open-v1",
        "model_lexemes": "preserved-exactly-without-normalization-v1",
        "child_objective": GoalObjective.BRANCH.value,
        "connective": Connective.EXHAUSTIVE_CASE.value,
        "equation_access": "opaque-copy-only-v1",
    }


def _boundary_constructor_sha256() -> str:
    return canonical_json_sha256(_boundary_constructor_record())


def _boundary_primary_thresholds() -> dict[str, int]:
    return {
        "typed_decoded_min": BOUNDARY_PRIMARY_TYPED_DECODED_MIN,
        "applied_min": BOUNDARY_PRIMARY_APPLIED_MIN,
        "applied_problem_min": BOUNDARY_PRIMARY_APPLIED_PROBLEMS_MIN,
        "problem_total": BOUNDARY_EXPECTED_PROBLEMS,
    }


@dataclass(frozen=True, slots=True)
class BoundaryDecompositionProbeCase:
    index: int
    source_case: KeyAlignedRepairProbeCase
    e27_case_record_sha256: str
    base_request_sha256: str
    boundary_request_sha256: str
    base_request_body_bytes: bytes = field(repr=False, compare=True)
    boundary_request_body_bytes: bytes = field(repr=False, compare=True)

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("boundary case index is invalid")
        if not isinstance(self.source_case, KeyAlignedRepairProbeCase):
            raise ValueError("boundary source case is invalid")
        for name in (
            "e27_case_record_sha256",
            "base_request_sha256",
            "boundary_request_sha256",
        ):
            if not _boundary_is_sha256(getattr(self, name)):
                raise ValueError(f"boundary case {name} is invalid")
        for bytes_name, digest_name in (
            ("base_request_body_bytes", "base_request_sha256"),
            ("boundary_request_body_bytes", "boundary_request_sha256"),
        ):
            body_bytes = getattr(self, bytes_name)
            if not isinstance(body_bytes, bytes):
                raise ValueError(f"boundary case {bytes_name} is invalid")
            try:
                body = json.loads(body_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"boundary case {bytes_name} is invalid") from exc
            if (
                canonical_json_bytes(body) != body_bytes
                or canonical_json_sha256(body) != getattr(self, digest_name)
            ):
                raise ValueError(f"boundary case {bytes_name} is not canonical")

    @property
    def state(self) -> State:
        return self.source_case.state

    @property
    def action(self) -> Action:
        return self.source_case.action

    @property
    def problem_id(self) -> str:
        return self.source_case.problem_id

    @property
    def effective_seed(self) -> int:
        return self.source_case.effective_seed

    @property
    def probe_observation(self) -> Mapping[str, object]:
        return self.source_case.probe_observation

    @property
    def e24_case_index(self) -> int:
        return self.source_case.source_case.source_case.source_case.index

    @property
    def base_request_body(self) -> dict[str, object]:
        return json.loads(self.base_request_body_bytes)

    @property
    def boundary_request_body(self) -> dict[str, object]:
        return json.loads(self.boundary_request_body_bytes)

    def manifest_record(self) -> dict[str, object]:
        source = self.source_case.manifest_record()
        return {
            "index": self.index,
            "e24_case_index": self.e24_case_index,
            "e27_case_index": self.source_case.index,
            "episode_key": source["episode_key"],
            "problem_id": self.problem_id,
            "action": source["action"],
            "probe_observation_sha256": source["probe_observation_sha256"],
            "effective_seed": self.effective_seed,
            "e27_case_record_sha256": self.e27_case_record_sha256,
            "base_request_sha256": self.base_request_sha256,
            "boundary_request_sha256": self.boundary_request_sha256,
        }


_BOUNDARY_CONTRACT_DIGEST_FIELDS = (
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
    "boundary_instruction_sha256",
    "boundary_payload_schema_sha256",
    "boundary_constructor_sha256",
    "boundary_oracle_manifest_sha256",
    "case_manifest_sha256",
    "base_request_manifest_sha256",
    "boundary_request_manifest_sha256",
)


@dataclass(frozen=True, slots=True)
class BoundaryDecompositionProbeContract:
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
    boundary_instruction_sha256: str
    boundary_payload_schema_sha256: str
    boundary_constructor_sha256: str
    boundary_oracle_manifest_sha256: str
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
    schema_version: str = BOUNDARY_PROBE_CONTRACT_VERSION
    boundary_lens_contract: str = BOUNDARY_LENS_CONTRACT_VERSION
    selection_contract: str = BOUNDARY_SELECTION_CONTRACT
    request_bridge_contract: str = BOUNDARY_REQUEST_BRIDGE_CONTRACT
    canonical_serializer: str = CANONICAL_JSON_VERSION
    interpretation_contract: str = BOUNDARY_INTERPRETATION_CONTRACT

    def __post_init__(self) -> None:
        _validate_boundary_contract(self)

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
        for name in _BOUNDARY_CONTRACT_DIGEST_FIELDS:
            record[name] = getattr(self, name)
        record.update({
            "generator": self.generator.to_record(),
            "generator_sha256": self.generator_sha256,
            "generator_transport": self.generator_transport.to_record(),
            "generator_transport_sha256": self.generator_transport_sha256,
            "generation_settings": self.generation_settings_record(),
            "generation_settings_sha256": self.generation_settings_sha256,
            "boundary_payload_contract": BOUNDARY_PAYLOAD_CONTRACT,
            "boundary_lens_contract": self.boundary_lens_contract,
            "boundary_instruction": BOUNDARY_INSTRUCTION,
            "boundary_constructor": _boundary_constructor_record(),
            "source_decision_count": self.source_decision_count,
            "case_count": self.case_count,
            "problem_count": self.problem_count,
            "planned_output_path": self.planned_output_path,
            "primary_thresholds": _boundary_primary_thresholds(),
            "cost_basis": _cost_basis_record(),
            "selection_contract": self.selection_contract,
            "request_bridge_contract": self.request_bridge_contract,
            "canonical_serializer": self.canonical_serializer,
            "interpretation_contract": self.interpretation_contract,
        })
        return record


def _validate_boundary_contract(contract: BoundaryDecompositionProbeContract) -> None:
    if contract.schema_version != BOUNDARY_PROBE_CONTRACT_VERSION:
        raise ValueError("boundary contract schema is invalid")
    for name in _BOUNDARY_CONTRACT_DIGEST_FIELDS:
        if not _boundary_is_sha256(getattr(contract, name)):
            raise ValueError(f"boundary contract {name} is invalid")
    if contract.boundary_instruction_sha256 != _boundary_instruction_sha256():
        raise ValueError("boundary instruction digest is invalid")
    if contract.boundary_payload_schema_sha256 != boundary_payload_schema_sha256():
        raise ValueError("boundary payload schema digest is invalid")
    if contract.boundary_constructor_sha256 != _boundary_constructor_sha256():
        raise ValueError("boundary constructor digest is invalid")
    if not isinstance(contract.generator, ModelIdentity):
        raise ValueError("boundary generator identity is invalid")
    if not isinstance(contract.generator_transport, ModelTransportContract):
        raise ValueError("boundary generator transport is invalid")
    if contract.generator_transport.base_url is None:
        raise ValueError("boundary probe requires a hosted generator")
    if contract.generator_transport.guided_json:
        raise ValueError("boundary probe forbids provider-guided JSON")
    if contract.generator_transport.model != contract.generator.model:
        raise ValueError("boundary generator differs from transport")
    if contract.lens_contract != LENS_CONTRACT_VERSION:
        raise ValueError("boundary source lens is not E24 v2.1")
    if contract.boundary_lens_contract != BOUNDARY_LENS_CONTRACT_VERSION:
        raise ValueError("boundary lens contract is invalid")
    if contract.request_environment_contract_sha256 == contract.scoring_environment_contract_sha256:
        raise ValueError("boundary request and scoring environments must differ")
    if contract.source_decision_count != BOUNDARY_EXPECTED_SOURCE_DECISIONS:
        raise ValueError("boundary source decision count is invalid")
    if contract.case_count != BOUNDARY_EXPECTED_CASES:
        raise ValueError("boundary case count is invalid")
    if contract.problem_count != BOUNDARY_EXPECTED_PROBLEMS:
        raise ValueError("boundary problem count is invalid")
    if contract.planned_output_path != _boundary_path_record(contract.planned_output_path):
        raise ValueError("boundary output path is invalid")
    if contract.selection_contract != BOUNDARY_SELECTION_CONTRACT:
        raise ValueError("boundary selection contract is invalid")
    if contract.request_bridge_contract != BOUNDARY_REQUEST_BRIDGE_CONTRACT:
        raise ValueError("boundary request bridge is invalid")
    if contract.canonical_serializer != CANONICAL_JSON_VERSION:
        raise ValueError("boundary canonical serializer is invalid")
    if contract.interpretation_contract != BOUNDARY_INTERPRETATION_CONTRACT:
        raise ValueError("boundary interpretation contract is invalid")


@dataclass(frozen=True, slots=True)
class BoundaryDecompositionProbeAuthorization:
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
    expected_boundary_instruction_sha256: str
    expected_boundary_payload_schema_sha256: str
    expected_boundary_constructor_sha256: str
    expected_boundary_oracle_manifest_sha256: str
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
        if not isinstance(self.preregistration_id, str) or not self.preregistration_id.strip():
            raise ValueError("boundary preregistration id is empty")
        for name in self.__dataclass_fields__:
            if name.startswith("expected_") and name.endswith("sha256"):
                if not _boundary_is_sha256(getattr(self, name)):
                    raise ValueError(f"boundary authorization {name} is invalid")
        if self.expected_source_decision_count != BOUNDARY_EXPECTED_SOURCE_DECISIONS:
            raise ValueError("authorized source decision count is invalid")
        if self.expected_case_count != BOUNDARY_EXPECTED_CASES:
            raise ValueError("authorized case count is invalid")
        if self.expected_problem_count != BOUNDARY_EXPECTED_PROBLEMS:
            raise ValueError("authorized problem count is invalid")
        if self.expected_output_path != _boundary_path_record(self.expected_output_path):
            raise ValueError("authorized boundary output path is invalid")

    def validate(self, contract: BoundaryDecompositionProbeContract) -> None:
        checks = {
            "source environment": (self.expected_source_environment_contract_sha256, contract.source_environment_contract_sha256),
            "request environment": (self.expected_request_environment_contract_sha256, contract.request_environment_contract_sha256),
            "scoring environment": (self.expected_scoring_environment_contract_sha256, contract.scoring_environment_contract_sha256),
            "corrected profile": (self.expected_corrected_profile_sha256, contract.corrected_profile_sha256),
            "E24 artifact": (self.expected_e24_artifact_sha256, contract.e24_artifact_sha256),
            "E24 probe contract": (self.expected_e24_probe_contract_sha256, contract.e24_probe_contract_sha256),
            "E24 case manifest": (self.expected_e24_case_manifest_sha256, contract.e24_case_manifest_sha256),
            "E27 artifact": (self.expected_e27_artifact_sha256, contract.e27_artifact_sha256),
            "E27 repair contract": (self.expected_e27_repair_contract_sha256, contract.e27_repair_contract_sha256),
            "E27 case manifest": (self.expected_e27_case_manifest_sha256, contract.e27_case_manifest_sha256),
            "E27 summary": (self.expected_e27_summary_sha256, contract.e27_summary_sha256),
            "boundary instruction": (self.expected_boundary_instruction_sha256, contract.boundary_instruction_sha256),
            "boundary payload schema": (self.expected_boundary_payload_schema_sha256, contract.boundary_payload_schema_sha256),
            "boundary constructor": (self.expected_boundary_constructor_sha256, contract.boundary_constructor_sha256),
            "boundary oracle": (self.expected_boundary_oracle_manifest_sha256, contract.boundary_oracle_manifest_sha256),
            "case manifest": (self.expected_case_manifest_sha256, contract.case_manifest_sha256),
            "base request manifest": (self.expected_base_request_manifest_sha256, contract.base_request_manifest_sha256),
            "boundary request manifest": (self.expected_boundary_request_manifest_sha256, contract.boundary_request_manifest_sha256),
            "generator": (self.expected_generator_sha256, contract.generator_sha256),
            "generator transport": (self.expected_generator_transport_sha256, contract.generator_transport_sha256),
            "generation settings": (self.expected_generation_settings_sha256, contract.generation_settings_sha256),
            "contract": (self.expected_contract_sha256, contract.sha256),
            "source decision count": (self.expected_source_decision_count, contract.source_decision_count),
            "case count": (self.expected_case_count, contract.case_count),
            "problem count": (self.expected_problem_count, contract.problem_count),
            "output path": (self.expected_output_path, contract.planned_output_path),
        }
        for label, (expected, observed) in checks.items():
            if expected != observed:
                raise ValueError(f"authorized {label} differs")


@dataclass(frozen=True, slots=True)
class PreparedBoundaryDecompositionProbe:
    contract: BoundaryDecompositionProbeContract
    source_e27_probe: PreparedKeyAlignedRepairProbe
    source_e27_summary: Mapping[str, object]
    oracle_manifest: tuple[Mapping[str, object], ...]
    cases: tuple[BoundaryDecompositionProbeCase, ...]
    _trust_seal: object = field(init=False, repr=False, compare=False, default=None)
    _trusted_binding: object = field(init=False, repr=False, compare=False, default=None)

    def validate_trusted(self) -> None:
        _validate_prepared_boundary_probe(self)


@dataclass(frozen=True, slots=True)
class BoundaryDecompositionProbeRun:
    output_path: Path
    contract_sha256: str
    case_count: int
    wire_usable_count: int
    decoded_count: int
    symbolic_applied_count: int
    applied_problem_count: int


__all__ += [
    "BOUNDARY_LENS_CONTRACT_VERSION",
    "BOUNDARY_PROBE_CASE_VERSION",
    "BOUNDARY_PROBE_CONTRACT_VERSION",
    "BOUNDARY_PROBE_HEADER_VERSION",
    "BOUNDARY_PROBE_SUMMARY_VERSION",
    "BoundaryDecompositionProbeAuthorization",
    "BoundaryDecompositionProbeCase",
    "BoundaryDecompositionProbeContract",
    "BoundaryDecompositionProbeRun",
    "BoundaryDecompositionProbeValidationError",
    "PreparedBoundaryDecompositionProbe",
]


_BOUNDARY_ORACLE_BOUNDARIES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "cw1-ccdf9e1567dc": ("4",),
    "cw1-2640597e2425": ("6",),
    "cw1-52e3b6b3ec25": ("2",),
    "cw1-ddad3be834f4": ("4",),
    "cw1-4ba0fc91c9e8": ("-3",),
})


def _boundary_request_material(
    source_probe: PreparedKeyAlignedRepairProbe,
    source_case: KeyAlignedRepairProbeCase,
) -> tuple[LensRequest, dict[str, object], dict[str, object]]:
    source_request, boundary_request = build_boundary_lens_request(
        source_case.probe_observation,
        source_case.action,
    )
    contract = source_probe.contract
    settings = {
        "model": contract.generator.model,
        "temperature": contract.generation_temperature,
        "max_tokens": contract.generation_max_tokens,
        "seed": source_case.effective_seed,
        "guided_json": contract.generator_transport.guided_json,
    }
    base_body = build_openai_compatible_json_body(
        system=source_request.system,
        user=source_request.user,
        schema=source_request.schema,
        **settings,
    )
    boundary_body = build_openai_compatible_json_body(
        system=boundary_request.system,
        user=boundary_request.user,
        schema=boundary_request.schema,
        **settings,
    )
    if canonical_json_sha256(base_body) != source_case.base_request_sha256:
        raise BoundaryDecompositionProbeValidationError(
            "boundary source request is not byte-exact E24"
        )
    expected_base = source_case.source_case.base_request_body_bytes
    if canonical_json_bytes(base_body) != expected_base:
        raise BoundaryDecompositionProbeValidationError(
            "boundary source request bytes differ from E24"
        )
    if set(base_body) != set(boundary_body):
        raise BoundaryDecompositionProbeValidationError(
            "boundary request changed top-level request fields"
        )
    for key in set(base_body) - {"messages"}:
        if base_body[key] != boundary_body[key]:
            raise BoundaryDecompositionProbeValidationError(
                f"boundary request changed {key}"
            )
    old_messages = base_body.get("messages")
    new_messages = boundary_body.get("messages")
    if (
        type(old_messages) is not list
        or type(new_messages) is not list
        or len(old_messages) != 2
        or len(new_messages) != 2
        or old_messages[0] != new_messages[0]
        or old_messages[1].get("content") != source_request.user
        or new_messages[1].get("content") != boundary_request.user
    ):
        raise BoundaryDecompositionProbeValidationError(
            "boundary request changed beyond its user instruction"
        )
    return boundary_request, base_body, boundary_body


def _boundary_manifest_sha256(
    cases: tuple[BoundaryDecompositionProbeCase, ...],
) -> str:
    return canonical_json_sha256([case.manifest_record() for case in cases])


def _boundary_request_manifest_sha256(
    cases: tuple[BoundaryDecompositionProbeCase, ...], field_name: str
) -> str:
    return canonical_json_sha256([
        {"index": case.index, field_name: getattr(case, field_name)}
        for case in cases
    ])


def _score_boundary_draft(
    prepared: PreparedBoundaryDecompositionProbe,
    case: BoundaryDecompositionProbeCase,
    draft: DecompositionDraft,
) -> dict[str, object]:
    return score_decomposition_draft(
        state=case.state,
        action=case.action,
        draft=draft,
        adapter=CaseworkMathVerifier.corrected_v11(),
        coverage_validator=validate_decomposition_coverage_v11,
        profile=(
            prepared.source_e27_probe.source_probe.source_rescore.contract.
            corrected_profile
        ),
    )


def _build_boundary_oracle_manifest(
    cases: tuple[BoundaryDecompositionProbeCase, ...],
    source_probe: PreparedKeyAlignedRepairProbe,
) -> tuple[Mapping[str, object], ...]:
    by_problem: dict[str, BoundaryDecompositionProbeCase] = {}
    for case in cases:
        by_problem.setdefault(case.problem_id, case)
    if set(by_problem) != set(_BOUNDARY_ORACLE_BOUNDARIES):
        raise BoundaryDecompositionProbeValidationError(
            "boundary oracle problem ids differ from the sealed five"
        )
    profile = source_probe.source_probe.source_rescore.contract.corrected_profile
    records: list[Mapping[str, object]] = []
    for problem_id, boundaries in _BOUNDARY_ORACLE_BOUNDARIES.items():
        case = by_problem[problem_id]
        payload = decode_boundary_payload({"boundaries": list(boundaries)})
        draft = construct_boundary_decomposition(case.state, case.action, payload)
        application = score_decomposition_draft(
            state=case.state,
            action=case.action,
            draft=draft,
            adapter=CaseworkMathVerifier.corrected_v11(),
            coverage_validator=validate_decomposition_coverage_v11,
            profile=profile,
        )
        domains = [child.domain for child in draft.child_obligations]
        if len(domains) != 2 or application["application_status"] != "applied":
            raise BoundaryDecompositionProbeValidationError(
                f"boundary oracle failed for {problem_id}"
            )
        if problem_id == "cw1-2640597e2425" and domains != [
            "x < 6", "6 <= x < 12"
        ]:
            raise BoundaryDecompositionProbeValidationError(
                "boundary oracle failed parent-domain clipping proof"
            )
        records.append({
            "schema_version": BOUNDARY_ORACLE_VERSION,
            "problem_id": problem_id,
            "boundaries": list(boundaries),
            "constructed_domains": domains,
            "corrected_application": application,
        })
    return tuple(records)


def validate_boundary_oracle_cases(
    prepared: PreparedBoundaryDecompositionProbe,
) -> list[dict[str, object]]:
    if not isinstance(prepared, PreparedBoundaryDecompositionProbe):
        raise TypeError("prepared boundary oracle input has an invalid type")
    prepared.validate_trusted()
    observed = _build_boundary_oracle_manifest(
        prepared.cases, prepared.source_e27_probe
    )
    if observed != prepared.oracle_manifest:
        raise BoundaryDecompositionProbeValidationError(
            "boundary oracle manifest changed after preparation"
        )
    return [dict(record) for record in observed]


def _prepare_boundary_decomposition_probe(
    *,
    source_run_path: str | Path,
    source_diagnostic_path: str | Path,
    e24_artifact_path: str | Path,
    e25_artifact_path: str | Path,
    e26_artifact_path: str | Path,
    e27_artifact_path: str | Path,
    e26_authorization: SemanticRepairProbeAuthorization,
    e27_authorization: KeyAlignedRepairProbeAuthorization,
    expected_e26_artifact_sha256: str,
    expected_e27_artifact_sha256: str,
    problem_source: CaseworkProblemSource,
    planned_output_path: str | Path,
) -> PreparedBoundaryDecompositionProbe:
    if not isinstance(problem_source, CaseworkProblemSource):
        raise TypeError("boundary problem source has an invalid type")
    if not isinstance(e26_authorization, SemanticRepairProbeAuthorization):
        raise TypeError("boundary E26 authorization has an invalid type")
    if not isinstance(e27_authorization, KeyAlignedRepairProbeAuthorization):
        raise TypeError("boundary E27 authorization has an invalid type")
    if not _boundary_is_sha256(expected_e26_artifact_sha256):
        raise ValueError("expected E26 artifact digest is invalid")
    if not _boundary_is_sha256(expected_e27_artifact_sha256):
        raise ValueError("expected E27 artifact digest is invalid")
    e27_path = Path(e27_artifact_path)
    if file_sha256(e27_path) != expected_e27_artifact_sha256:
        raise BoundaryDecompositionProbeValidationError(
            "E27 artifact digest differs"
        )
    source_probe = prepare_key_aligned_repair_probe(
        source_run_path=source_run_path,
        source_diagnostic_path=source_diagnostic_path,
        e24_artifact_path=e24_artifact_path,
        e25_artifact_path=e25_artifact_path,
        e26_artifact_path=e26_artifact_path,
        e26_authorization=e26_authorization,
        expected_e26_artifact_sha256=expected_e26_artifact_sha256,
        problem_source=problem_source,
        planned_output_path=e27_authorization.expected_output_path,
    )
    e27_authorization.validate(source_probe.contract)
    source_summary = read_key_aligned_repair_probe_artifact(
        e27_path,
        prepared=source_probe,
        authorization=e27_authorization,
    )
    if (
        source_summary.get("case_count") != BOUNDARY_EXPECTED_CASES
        or source_summary.get("wire_usable_count") != BOUNDARY_EXPECTED_CASES
        or source_summary.get("typed_decoded_count") != 32
        or source_summary.get("symbolic_applied_count") != 0
        or source_summary.get("applied_problem_count") != 0
    ):
        raise BoundaryDecompositionProbeValidationError(
            "E27 summary does not authorize the boundary-only branch"
        )
    records = _read_canonical_records(e27_path)
    e27_cases = [
        record for record in records
        if record.get("schema_version") == KEY_ALIGNED_REPAIR_CASE_VERSION
    ]
    if len(e27_cases) != len(source_probe.cases):
        raise BoundaryDecompositionProbeValidationError(
            "E27 case count differs from strict preparation"
        )
    selected: list[BoundaryDecompositionProbeCase] = []
    for source_case, e27_record in zip(
        source_probe.cases, e27_cases, strict=True
    ):
        source_manifest = source_case.manifest_record()
        if {key: e27_record.get(key) for key in source_manifest} != source_manifest:
            raise BoundaryDecompositionProbeValidationError(
                f"E27 case {source_case.index} differs from reconstruction"
            )
        _, base_body, boundary_body = _boundary_request_material(
            source_probe, source_case
        )
        selected.append(BoundaryDecompositionProbeCase(
            index=source_case.index,
            source_case=source_case,
            e27_case_record_sha256=canonical_json_sha256(e27_record),
            base_request_sha256=canonical_json_sha256(base_body),
            boundary_request_sha256=canonical_json_sha256(boundary_body),
            base_request_body_bytes=canonical_json_bytes(base_body),
            boundary_request_body_bytes=canonical_json_bytes(boundary_body),
        ))
    cases = tuple(selected)
    oracle_manifest = _build_boundary_oracle_manifest(cases, source_probe)
    source_contract = source_probe.contract
    e26_contract = source_probe.source_probe.contract
    contract = BoundaryDecompositionProbeContract(
        source_environment_contract_sha256=source_contract.source_environment_contract_sha256,
        request_environment_contract_sha256=source_contract.request_environment_contract_sha256,
        scoring_environment_contract_sha256=source_contract.scoring_environment_contract_sha256,
        corrected_profile_sha256=source_contract.corrected_profile_sha256,
        e24_artifact_sha256=e26_contract.e24_artifact_sha256,
        e24_probe_contract_sha256=e26_contract.e24_probe_contract_sha256,
        e24_case_manifest_sha256=e26_contract.e24_case_manifest_sha256,
        e27_artifact_sha256=expected_e27_artifact_sha256,
        e27_repair_contract_sha256=source_contract.sha256,
        e27_case_manifest_sha256=source_contract.case_manifest_sha256,
        e27_summary_sha256=canonical_json_sha256(source_summary),
        boundary_instruction_sha256=_boundary_instruction_sha256(),
        boundary_payload_schema_sha256=boundary_payload_schema_sha256(),
        boundary_constructor_sha256=_boundary_constructor_sha256(),
        boundary_oracle_manifest_sha256=canonical_json_sha256(
            list(oracle_manifest)
        ),
        case_manifest_sha256=_boundary_manifest_sha256(cases),
        base_request_manifest_sha256=_boundary_request_manifest_sha256(
            cases, "base_request_sha256"
        ),
        boundary_request_manifest_sha256=_boundary_request_manifest_sha256(
            cases, "boundary_request_sha256"
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
        planned_output_path=_boundary_path_record(planned_output_path),
    )
    prepared = PreparedBoundaryDecompositionProbe(
        contract=contract,
        source_e27_probe=source_probe,
        source_e27_summary=dict(source_summary),
        oracle_manifest=oracle_manifest,
        cases=cases,
    )
    binding = (
        prepared.contract,
        prepared.source_e27_probe,
        prepared.source_e27_summary,
        prepared.oracle_manifest,
        prepared.cases,
    )
    object.__setattr__(prepared, "_trust_seal", _BOUNDARY_PREPARED_TRUST_SEAL)
    object.__setattr__(prepared, "_trusted_binding", binding)
    prepared.validate_trusted()
    return prepared


def prepare_boundary_decomposition_probe(**kwargs) -> PreparedBoundaryDecompositionProbe:
    try:
        return _prepare_boundary_decomposition_probe(**kwargs)
    except (
        KeyAlignedRepairProbeValidationError,
        SemanticRepairProbeValidationError,
    ) as exc:
        raise BoundaryDecompositionProbeValidationError(
            "boundary preparation failed strict source validation"
        ) from exc


def _validate_prepared_boundary_probe(
    prepared: PreparedBoundaryDecompositionProbe,
) -> None:
    binding = (
        prepared.contract,
        prepared.source_e27_probe,
        prepared.source_e27_summary,
        prepared.oracle_manifest,
        prepared.cases,
    )
    if (
        prepared._trust_seal is not _BOUNDARY_PREPARED_TRUST_SEAL
        or prepared._trusted_binding != binding
    ):
        raise BoundaryDecompositionProbeValidationError(
            "boundary preparation did not come from strict replay"
        )
    prepared.source_e27_probe.validate_trusted()
    contract = prepared.contract
    if prepared.source_e27_probe.contract.sha256 != contract.e27_repair_contract_sha256:
        raise BoundaryDecompositionProbeValidationError(
            "prepared E27 contract differs"
        )
    if canonical_json_sha256(prepared.source_e27_summary) != contract.e27_summary_sha256:
        raise BoundaryDecompositionProbeValidationError(
            "prepared E27 summary differs"
        )
    if tuple(case.index for case in prepared.cases) != tuple(range(contract.case_count)):
        raise BoundaryDecompositionProbeValidationError(
            "prepared boundary cases are not ordered"
        )
    if _boundary_manifest_sha256(prepared.cases) != contract.case_manifest_sha256:
        raise BoundaryDecompositionProbeValidationError(
            "prepared boundary case manifest changed"
        )
    for field_name, expected in (
        ("base_request_sha256", contract.base_request_manifest_sha256),
        ("boundary_request_sha256", contract.boundary_request_manifest_sha256),
    ):
        if _boundary_request_manifest_sha256(prepared.cases, field_name) != expected:
            raise BoundaryDecompositionProbeValidationError(
                f"prepared {field_name} manifest changed"
            )
    observed_oracle = _build_boundary_oracle_manifest(
        prepared.cases, prepared.source_e27_probe
    )
    if (
        observed_oracle != prepared.oracle_manifest
        or canonical_json_sha256(list(observed_oracle))
        != contract.boundary_oracle_manifest_sha256
    ):
        raise BoundaryDecompositionProbeValidationError(
            "prepared boundary oracle manifest changed"
        )
    for case in prepared.cases:
        _, base_body, boundary_body = _boundary_request_material(
            prepared.source_e27_probe, case.source_case
        )
        if canonical_json_bytes(base_body) != case.base_request_body_bytes:
            raise BoundaryDecompositionProbeValidationError(
                "prepared E24 base request changed"
            )
        if canonical_json_bytes(boundary_body) != case.boundary_request_body_bytes:
            raise BoundaryDecompositionProbeValidationError(
                "prepared boundary request changed"
            )


def build_boundary_request(
    prepared: PreparedBoundaryDecompositionProbe,
    case: BoundaryDecompositionProbeCase,
) -> tuple[LensRequest, dict[str, object], dict[str, object]]:
    if not isinstance(prepared, PreparedBoundaryDecompositionProbe):
        raise TypeError("prepared boundary request input has an invalid type")
    if not isinstance(case, BoundaryDecompositionProbeCase):
        raise TypeError("boundary request case has an invalid type")
    prepared.validate_trusted()
    if case not in prepared.cases:
        raise ValueError("boundary case is not in the prepared probe")
    request, base_body, boundary_body = _boundary_request_material(
        prepared.source_e27_probe, case.source_case
    )
    if canonical_json_bytes(base_body) != case.base_request_body_bytes:
        raise BoundaryDecompositionProbeValidationError(
            "E24 base request changed after preparation"
        )
    if canonical_json_bytes(boundary_body) != case.boundary_request_body_bytes:
        raise BoundaryDecompositionProbeValidationError(
            "boundary request changed after preparation"
        )
    return request, base_body, boundary_body


__all__ += [
    "build_boundary_request",
    "prepare_boundary_decomposition_probe",
    "validate_boundary_oracle_cases",
]


from .diagnostics import (
    DIAGNOSTIC_ATTEMPT_VERSION,
    DIAGNOSTIC_CALL_VERSION,
    validate_attempt_record,
    validate_call_record,
)
from .semantic_repair import (
    _validate_application_record,
    _validate_call_transcript,
)


_BOUNDARY_STRUCTURAL_FAULT_REASONS = frozenset({
    "unsupported_domain",
    "malformed_scalar",
    "decomposition_domain_overlap",
    "decomposition_domain_not_exhaustive",
})
_BOUNDARY_MULTI_SOLUTION_PROBLEMS = frozenset({
    "cw1-ccdf9e1567dc",
    "cw1-ddad3be834f4",
    "cw1-4ba0fc91c9e8",
})



def _boundary_header(
    prepared: PreparedBoundaryDecompositionProbe,
    authorization: BoundaryDecompositionProbeAuthorization,
) -> dict[str, object]:
    contract = prepared.contract
    return {
        "schema_version": BOUNDARY_PROBE_HEADER_VERSION,
        "canonical_serializer": CANONICAL_JSON_VERSION,
        "preregistration_id": authorization.preregistration_id,
        "probe_contract": contract.to_record(),
        "probe_contract_sha256": contract.sha256,
        "e24_artifact_sha256": contract.e24_artifact_sha256,
        "e27_artifact_sha256": contract.e27_artifact_sha256,
        "e27_repair_contract_sha256": contract.e27_repair_contract_sha256,
        "e27_summary_sha256": contract.e27_summary_sha256,
        "request_environment_contract_sha256": (
            contract.request_environment_contract_sha256
        ),
        "scoring_environment_contract_sha256": (
            contract.scoring_environment_contract_sha256
        ),
        "corrected_profile_sha256": contract.corrected_profile_sha256,
        "boundary_instruction_sha256": contract.boundary_instruction_sha256,
        "boundary_payload_schema_sha256": (
            contract.boundary_payload_schema_sha256
        ),
        "boundary_constructor_sha256": contract.boundary_constructor_sha256,
        "boundary_oracle_manifest_sha256": (
            contract.boundary_oracle_manifest_sha256
        ),
        "case_manifest_sha256": contract.case_manifest_sha256,
        "base_request_manifest_sha256": contract.base_request_manifest_sha256,
        "boundary_request_manifest_sha256": (
            contract.boundary_request_manifest_sha256
        ),
    }


def _boundary_summary(
    contract: BoundaryDecompositionProbeContract,
    case_records: list[dict[str, object]],
    *,
    attempt_count: int,
    call_count: int,
    attempt_outcome_counts: Mapping[str, int],
    reported_usage: Mapping[str, int],
) -> dict[str, object]:
    problem_ids = sorted({row["problem_id"] for row in case_records})
    applied_problem_ids = sorted({
        row["problem_id"]
        for row in case_records
        if row["corrected_application"] is not None
        and row["corrected_application"]["application_status"] == "applied"
    })
    decode_counts = Counter(row["decode"]["status"] for row in case_records)
    for key in ("decoded", "payload_error", "not_run"):
        decode_counts.setdefault(key, 0)
    application_counts = Counter(
        "not_run"
        if row["corrected_application"] is None
        else row["corrected_application"]["application_status"]
        for row in case_records
    )
    for key in ("applied", "algebra_rejected", "not_run"):
        application_counts.setdefault(key, 0)
    reason_counts: Counter[str] = Counter()
    for row in case_records:
        application = row["corrected_application"]
        if application is None:
            reason_counts["not_run"] += 1
        elif application["application_status"] == "applied":
            reason_counts["applied"] += 1
        else:
            reason_counts[application["reason_code"]] += 1
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
    by_solution_class: dict[str, dict[str, int]] = {}
    for label, selected_ids in (
        ("one_solution", set(problem_ids) - set(_BOUNDARY_MULTI_SOLUTION_PROBLEMS)),
        ("multi_solution", set(problem_ids) & set(_BOUNDARY_MULTI_SOLUTION_PROBLEMS)),
    ):
        rows = [row for row in case_records if row["problem_id"] in selected_ids]
        by_solution_class[label] = {
            "problem_count": len(selected_ids),
            "case_count": len(rows),
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
    return {
        "schema_version": BOUNDARY_PROBE_SUMMARY_VERSION,
        "probe_contract_sha256": contract.sha256,
        "e27_summary_sha256": contract.e27_summary_sha256,
        "boundary_oracle_manifest_sha256": (
            contract.boundary_oracle_manifest_sha256
        ),
        "case_manifest_sha256": contract.case_manifest_sha256,
        "base_request_manifest_sha256": contract.base_request_manifest_sha256,
        "boundary_request_manifest_sha256": (
            contract.boundary_request_manifest_sha256
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
        "symbolic_applied_count": application_counts["applied"],
        "symbolic_application_counts": dict(sorted(application_counts.items())),
        "corrected_result_reason_counts": dict(sorted(reason_counts.items())),
        "applied_problem_count": len(applied_problem_ids),
        "applied_problem_ids": applied_problem_ids,
        "by_problem": by_problem,
        "by_solution_class": by_solution_class,
        "raw_attempt_outcome_counts": dict(attempt_outcome_counts),
        "reported_usage": dict(reported_usage),
        "reported_cost": _reported_cost_record(reported_usage),
        "primary_thresholds": _boundary_primary_thresholds(),
        "interpretation_contract": BOUNDARY_INTERPRETATION_CONTRACT,
    }


def _boundary_decode_and_score(
    prepared: PreparedBoundaryDecompositionProbe,
    case: BoundaryDecompositionProbeCase,
    payload: object,
) -> tuple[dict[str, object], dict[str, object] | None, dict[str, object] | None]:
    try:
        decoded = decode_boundary_payload(payload)
        details = construct_boundary_details(case.state, case.action, decoded)
    except (BoundaryPayloadError, _AdapterError, ValueError) as exc:
        return (
            {
                "status": "payload_error",
                "error_type": type(exc).__name__,
                "payload": None,
            },
            None,
            None,
        )
    decode = {
        "status": "decoded",
        "error_type": None,
        "payload": decoded.to_record(),
    }
    construction = details.to_record()
    application = _score_boundary_draft(prepared, case, details.draft)
    reason = application["reason_code"]
    if reason in _BOUNDARY_STRUCTURAL_FAULT_REASONS:
        raise BoundaryDecompositionProbeValidationError(
            f"trusted boundary construction produced implementation fault {reason}"
        )
    return decode, construction, application


def _run_boundary_decomposition_probe(
    *,
    output_path: str | Path,
    prepared: PreparedBoundaryDecompositionProbe,
    generator_client: OpenAICompatibleLLMClient,
    authorization: BoundaryDecompositionProbeAuthorization,
) -> BoundaryDecompositionProbeRun:
    if not isinstance(prepared, PreparedBoundaryDecompositionProbe):
        raise TypeError("prepared boundary run input has an invalid type")
    if not isinstance(authorization, BoundaryDecompositionProbeAuthorization):
        raise TypeError("boundary run authorization has an invalid type")
    prepared.validate_trusted()
    authorization.validate(prepared.contract)
    validate_boundary_oracle_cases(prepared)
    if type(generator_client) is not OpenAICompatibleLLMClient:
        raise TypeError("boundary generator client has an invalid type")
    if generator_client.attempt_observer is not None:
        raise ValueError("boundary generator already has an observer")
    if llm_transport_contract(generator_client) != prepared.contract.generator_transport:
        raise ValueError("boundary generator transport differs from contract")
    final_path = Path(output_path)
    if final_path.as_posix() != prepared.contract.planned_output_path:
        raise ValueError("boundary output differs from contract")
    partial_path = final_path.with_name(f"{final_path.name}.partial")
    if final_path.exists() or partial_path.exists():
        raise FileExistsError("boundary output or partial output already exists")
    if not final_path.parent.exists() or not final_path.parent.is_dir():
        raise FileNotFoundError("boundary output directory does not exist")

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
            partial_path, _boundary_header(prepared, authorization)
        )
        callback = writer.observe_attempt
        generator_client.attempt_observer = callback
        for case in prepared.cases:
            request, _, expected_body = build_boundary_request(prepared, case)
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
                    schema_validation_mode=SCHEMA_VALIDATION_STRICT_V1,
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
                    usage["completion_tokens"] += observed_usage["completion_tokens"] or 0
                    usage["total_tokens"] += observed_usage["total_tokens"] or 0
            if attempts[0]["request_body"] != expected_body:
                raise BoundaryDecompositionProbeValidationError(
                    "hosted boundary request differs from preparation"
                )
            if attempts[0]["effective_seed"] != case.effective_seed:
                raise BoundaryDecompositionProbeValidationError(
                    "hosted boundary request changed the E24 seed"
                )
            wire_usable = call_record["final_outcome"] == "transport_accepted"
            if wire_usable is (generation_error is not None):
                raise BoundaryDecompositionProbeValidationError(
                    "boundary generator return differs from raw diagnostics"
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
                    raise BoundaryDecompositionProbeValidationError(
                        "accepted boundary call returned no payload"
                    )
                decode, construction, application = _boundary_decode_and_score(
                    prepared, case, payload
                )
            record = {
                "schema_version": BOUNDARY_PROBE_CASE_VERSION,
                **case.manifest_record(),
                "boundary_call_record_id": call_record["record_id"],
                "boundary_payload": payload if wire_usable else None,
                "wire": wire,
                "decode": decode,
                "construction": construction,
                "corrected_application": application,
            }
            case_records.append(record)
            writer.write_case(record)
        summary = _boundary_summary(
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
        validated = read_boundary_decomposition_probe_artifact(
            partial_path, prepared=prepared, authorization=authorization
        )
        if validated != summary:
            raise BoundaryDecompositionProbeValidationError(
                "boundary summary changed during validation"
            )
        os.link(partial_path, final_path)
        partial_path.unlink()
        return BoundaryDecompositionProbeRun(
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


def run_boundary_decomposition_probe(**kwargs) -> BoundaryDecompositionProbeRun:
    try:
        return _run_boundary_decomposition_probe(**kwargs)
    except SemanticRepairProbeValidationError as exc:
        raise BoundaryDecompositionProbeValidationError(
            "boundary run failed shared transcript validation"
        ) from exc


_BOUNDARY_MANIFEST_KEYS = frozenset({
    "index", "e24_case_index", "e27_case_index", "episode_key", "problem_id",
    "action", "probe_observation_sha256", "effective_seed",
    "e27_case_record_sha256", "base_request_sha256",
    "boundary_request_sha256",
})
_BOUNDARY_CASE_KEYS = _BOUNDARY_MANIFEST_KEYS | frozenset({
    "schema_version", "boundary_call_record_id", "boundary_payload", "wire",
    "decode", "construction", "corrected_application",
})


def _read_boundary_decomposition_probe_artifact(
    path: str | Path,
    *,
    prepared: PreparedBoundaryDecompositionProbe,
    authorization: BoundaryDecompositionProbeAuthorization,
) -> dict[str, object]:
    if not isinstance(prepared, PreparedBoundaryDecompositionProbe):
        raise TypeError("prepared boundary reader input has an invalid type")
    if not isinstance(authorization, BoundaryDecompositionProbeAuthorization):
        raise TypeError("boundary reader authorization has an invalid type")
    prepared.validate_trusted()
    authorization.validate(prepared.contract)
    records = _read_canonical_records(Path(path))
    if len(records) < 2:
        raise BoundaryDecompositionProbeValidationError(
            "boundary artifact is incomplete"
        )
    header = records[0]
    if canonical_json_bytes(header) != canonical_json_bytes(
        _boundary_header(prepared, authorization)
    ):
        raise BoundaryDecompositionProbeValidationError(
            "boundary artifact header differs"
        )
    summary = records[-1]
    if summary.get("schema_version") != BOUNDARY_PROBE_SUMMARY_VERSION:
        raise BoundaryDecompositionProbeValidationError(
            "boundary artifact summary is missing"
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
    for record in records[1:-1]:
        schema = record.get("schema_version")
        if schema == DIAGNOSTIC_ATTEMPT_VERSION:
            if pending_call is not None:
                raise BoundaryDecompositionProbeValidationError(
                    "boundary attempt follows a completed call"
                )
            validate_attempt_record(record)
            if record.get("role") != "generator":
                raise BoundaryDecompositionProbeValidationError(
                    "boundary attempt role is invalid"
                )
            pending_attempts.append(record)
            raw_attempt_count += 1
            attempt_counts[record["outcome"]] += 1
            observed_usage = record["usage"]
            if observed_usage["reported"]:
                usage["attempt_count"] += 1
                usage["prompt_tokens"] += observed_usage["prompt_tokens"] or 0
                usage["completion_tokens"] += observed_usage["completion_tokens"] or 0
                usage["total_tokens"] += observed_usage["total_tokens"] or 0
            continue
        if schema == DIAGNOSTIC_CALL_VERSION:
            if pending_call is not None or not pending_attempts:
                raise BoundaryDecompositionProbeValidationError(
                    "boundary call has invalid placement"
                )
            validate_call_record(record, attempts=pending_attempts)
            call_id = record.get("call_id")
            if (
                record.get("role") != "generator"
                or not isinstance(call_id, str)
                or call_id in seen_call_ids
            ):
                raise BoundaryDecompositionProbeValidationError(
                    "boundary call identity or role is invalid"
                )
            seen_call_ids.add(call_id)
            pending_call = record
            call_count += 1
            continue
        if schema != BOUNDARY_PROBE_CASE_VERSION:
            raise BoundaryDecompositionProbeValidationError(
                "boundary artifact has an unexpected record"
            )
        case_index = len(case_records)
        if case_index >= len(prepared.cases):
            raise BoundaryDecompositionProbeValidationError(
                "boundary artifact has too many cases"
            )
        if pending_call is None or not pending_attempts:
            raise BoundaryDecompositionProbeValidationError(
                "boundary case lacks its call transcript"
            )
        if set(record) != _BOUNDARY_CASE_KEYS:
            raise BoundaryDecompositionProbeValidationError(
                f"boundary case {case_index} has invalid fields"
            )
        case = prepared.cases[case_index]
        final_payload = _validate_call_transcript(
            pending_attempts,
            pending_call,
            initial_body=case.boundary_request_body,
            initial_seed=case.effective_seed,
            transport=prepared.contract.generator_transport,
            wire_schema=boundary_payload_schema_record(),
            schema_validation_mode=SCHEMA_VALIDATION_STRICT_V1,
        )
        if record["boundary_call_record_id"] != pending_call["record_id"]:
            raise BoundaryDecompositionProbeValidationError(
                "boundary case call id differs"
            )
        expected_usable = pending_call["final_outcome"] == "transport_accepted"
        expected_wire = {
            "usable": expected_usable,
            "terminal_outcome": pending_call["final_outcome"],
            "final_attempt_outcome": pending_call["final_attempt_outcome"],
            "error_type": None if expected_usable else "LLMResponseError",
        }
        if type(record["wire"]) is not dict or set(record["wire"]) != {
            "usable", "terminal_outcome", "final_attempt_outcome", "error_type"
        }:
            raise BoundaryDecompositionProbeValidationError(
                "boundary wire fields are invalid"
            )
        if (
            type(record["wire"].get("usable")) is not bool
            or canonical_json_bytes(record["wire"])
            != canonical_json_bytes(expected_wire)
        ):
            raise BoundaryDecompositionProbeValidationError(
                "boundary wire staging differs"
            )
        if expected_wire["usable"]:
            if canonical_json_bytes(final_payload) != canonical_json_bytes(
                record["boundary_payload"]
            ):
                raise BoundaryDecompositionProbeValidationError(
                    "boundary payload differs from raw transcript"
                )
            decode, construction, application = _boundary_decode_and_score(
                prepared, case, final_payload
            )
        else:
            if record["boundary_payload"] is not None:
                raise BoundaryDecompositionProbeValidationError(
                    "unusable boundary call retained a payload"
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
            raise BoundaryDecompositionProbeValidationError(
                "boundary case staging differs from strict replay"
            )
        if application is not None:
            _validate_application_record(
                application,
                profile=(
                    prepared.source_e27_probe.source_probe.source_rescore.contract.
                    corrected_profile
                ),
                path=f"case {case_index}.application",
            )
        manifest = {key: record[key] for key in _BOUNDARY_MANIFEST_KEYS}
        if canonical_json_bytes(manifest) != canonical_json_bytes(
            case.manifest_record()
        ):
            raise BoundaryDecompositionProbeValidationError(
                "boundary case manifest differs"
            )
        case_records.append(record)
        pending_attempts = []
        pending_call = None
    if pending_attempts or pending_call is not None:
        raise BoundaryDecompositionProbeValidationError(
            "boundary artifact has an unresolved call"
        )
    if len(case_records) != prepared.contract.case_count:
        raise BoundaryDecompositionProbeValidationError(
            "boundary artifact case count differs"
        )
    artifact_manifest = [
        {key: row[key] for key in _BOUNDARY_MANIFEST_KEYS}
        for row in case_records
    ]
    if canonical_json_sha256(artifact_manifest) != prepared.contract.case_manifest_sha256:
        raise BoundaryDecompositionProbeValidationError(
            "boundary artifact case manifest digest differs"
        )
    for outcome in (
        "accepted", "parse_error", "schema_error", "provider_response_error",
        "http_status_error", "transport_error",
    ):
        attempt_counts.setdefault(outcome, 0)
    expected_summary = _boundary_summary(
        prepared.contract,
        case_records,
        attempt_count=raw_attempt_count,
        call_count=call_count,
        attempt_outcome_counts=attempt_counts,
        reported_usage=usage,
    )
    if canonical_json_bytes(summary) != canonical_json_bytes(expected_summary):
        raise BoundaryDecompositionProbeValidationError(
            "boundary artifact summary differs"
        )
    return dict(summary)


def read_boundary_decomposition_probe_artifact(
    path: str | Path,
    *,
    prepared: PreparedBoundaryDecompositionProbe,
    authorization: BoundaryDecompositionProbeAuthorization,
) -> dict[str, object]:
    try:
        return _read_boundary_decomposition_probe_artifact(
            path, prepared=prepared, authorization=authorization
        )
    except SemanticRepairProbeValidationError as exc:
        raise BoundaryDecompositionProbeValidationError(
            "boundary artifact failed shared transcript validation"
        ) from exc


__all__ += [
    "read_boundary_decomposition_probe_artifact",
    "run_boundary_decomposition_probe",
]
