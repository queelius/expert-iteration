"""Deterministic paired domain-parser rescoring over a sealed microprobe."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping

from .actions import DecompositionDraft, Err, Ok, decompose
from .casework_source import CaseworkProblemSource
from .contracts import (
    CANONICAL_JSON_VERSION,
    EnvironmentContract,
    build_environment_contract,
    canonical_json_bytes,
    canonical_json_sha256,
    implementation_name,
)
from .math_adapter import (
    CORRECTED_DOMAIN_GRAMMAR_PROFILE,
    LEGACY_DOMAIN_GRAMMAR_PROFILE,
    CaseworkMathVerifier,
    CoverageDecision,
    DomainGrammarProfile,
    validate_decomposition_coverage,
    validate_decomposition_coverage_v11,
)
from .microprobe import (
    GENERATOR_MICROPROBE_CASE_VERSION,
    GENERATOR_MICROPROBE_HEADER_VERSION,
    GeneratorMicroprobeCase,
    PreparedGeneratorMicroprobe,
    file_sha256,
    prepare_generator_microprobe,
    read_generator_microprobe_artifact,
)
from .model import (
    Action,
    Connective,
    Goal,
    GoalObjective,
    GoalObligation,
    State,
    Verb,
    direct_child_goals,
    get,
)
from .records import (
    deserialize_action,
    parse_canonical_json_line,
    serialize_obligation,
)


DOMAIN_RESCORE_CONTRACT_VERSION = "domain-parser-rescore-contract-v1"
DOMAIN_RESCORE_HEADER_VERSION = "domain-parser-rescore-header-v1"
DOMAIN_RESCORE_CASE_VERSION = "domain-parser-rescore-case-v1"
DOMAIN_RESCORE_SUMMARY_VERSION = "domain-parser-rescore-summary-v1"
DOMAIN_RESCORE_SELECTION_CONTRACT = (
    "decoded-decompose-drafts-from-sealed-e24-in-canonical-order-v1"
)
DOMAIN_RESCORE_INTERPRETATION_CONTRACT = (
    "deterministic-adapter-rescore-not-model-reward-or-training-evidence-v1"
)
DOMAIN_RESCORE_EXPECTED_SOURCE_DECISIONS = 66
DOMAIN_RESCORE_EXPECTED_CASES = 33
DOMAIN_RESCORE_EXPECTED_LEGACY_APPLIED = 0
DOMAIN_RESCORE_EXPECTED_CORRECTED_APPLIED = 0
DOMAIN_RESCORE_EXPECTED_LEGACY_REASONS = (
    ("decomposition_domain_not_exhaustive", 2),
    ("decomposition_domain_overlap", 8),
    ("malformed_scalar", 8),
    ("unsupported_domain", 15),
)
DOMAIN_RESCORE_EXPECTED_CORRECTED_REASONS = (
    ("decomposition_domain_not_exhaustive", 3),
    ("decomposition_domain_overlap", 15),
    ("unsupported_domain", 15),
)

_SHA256_CHARS = frozenset("0123456789abcdef")
_PREPARED_TRUST_SEAL = object()


class DomainRescoreValidationError(ValueError):
    """Raised when a domain-parser rescore contract or artifact is invalid."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA256_CHARS
    )


def _nonempty_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DomainRescoreValidationError(f"{path} must be nonempty text")
    return value


def _exact_record(
    value: object, keys: frozenset[str], path: str
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise DomainRescoreValidationError(f"{path} keys are invalid")
    return value


def _sha256(value: object, path: str) -> str:
    if not _is_sha256(value):
        raise DomainRescoreValidationError(f"{path} is not a sha256 digest")
    return value


def _nonnegative_integer(value: object, path: str) -> int:
    if type(value) is not int or value < 0:
        raise DomainRescoreValidationError(
            f"{path} must be a nonnegative integer"
        )
    return value


def _path_record(path: str | Path) -> str:
    value = Path(path).as_posix()
    if not value or value.endswith(".partial") or not value.endswith(".jsonl"):
        raise ValueError("rescore output path must name a final JSONL artifact")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("rescore output path must be ASCII") from exc
    return value


@dataclass(frozen=True, slots=True)
class DomainAdapterProfileBinding:
    """Exact parser and adapter identities used by one rescore arm."""

    domain_grammar_profile_id: str
    bounded_comparator_pattern: str
    implementation_id: str
    adapter_implementation: str
    generation_contract_id: str
    obligation_contract_id: str
    verifier_contract_id: str
    aggregate_contract_id: str
    payload_semantics_id: str
    validity_rejection_codes: tuple[str, ...]
    discharge_rejection_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "domain_grammar_profile_id",
            "bounded_comparator_pattern",
            "implementation_id",
            "adapter_implementation",
            "generation_contract_id",
            "obligation_contract_id",
            "verifier_contract_id",
            "aggregate_contract_id",
            "payload_semantics_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"adapter profile {field_name} is invalid")
        if self.bounded_comparator_pattern not in {"<|<=", "<=|<"}:
            raise ValueError("adapter profile comparator pattern is invalid")
        for field_name in (
            "validity_rejection_codes",
            "discharge_rejection_codes",
        ):
            codes = getattr(self, field_name)
            if (
                type(codes) is not tuple
                or not codes
                or codes != tuple(sorted(set(codes)))
                or any(not isinstance(code, str) or not code for code in codes)
            ):
                raise ValueError(f"adapter profile {field_name} is invalid")

    @property
    def rejection_codes(self) -> frozenset[str]:
        return frozenset(
            self.validity_rejection_codes + self.discharge_rejection_codes
        )

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "domain_grammar_profile_id": self.domain_grammar_profile_id,
            "bounded_comparator_pattern": self.bounded_comparator_pattern,
            "implementation_id": self.implementation_id,
            "adapter_implementation": self.adapter_implementation,
            "generation_contract_id": self.generation_contract_id,
            "obligation_contract_id": self.obligation_contract_id,
            "verifier_contract_id": self.verifier_contract_id,
            "aggregate_contract_id": self.aggregate_contract_id,
            "payload_semantics_id": self.payload_semantics_id,
            "validity_rejection_codes": list(self.validity_rejection_codes),
            "discharge_rejection_codes": list(self.discharge_rejection_codes),
        }

    @classmethod
    def from_record(cls, value: object) -> DomainAdapterProfileBinding:
        record = _exact_record(value, _PROFILE_KEYS, "adapter profile")
        try:
            validity = record["validity_rejection_codes"]
            discharge = record["discharge_rejection_codes"]
            if type(validity) is not list or type(discharge) is not list:
                raise ValueError("adapter reason codes must be arrays")
            return cls(
                domain_grammar_profile_id=record[
                    "domain_grammar_profile_id"
                ],
                bounded_comparator_pattern=record[
                    "bounded_comparator_pattern"
                ],
                implementation_id=record["implementation_id"],
                adapter_implementation=record["adapter_implementation"],
                generation_contract_id=record["generation_contract_id"],
                obligation_contract_id=record["obligation_contract_id"],
                verifier_contract_id=record["verifier_contract_id"],
                aggregate_contract_id=record["aggregate_contract_id"],
                payload_semantics_id=record["payload_semantics_id"],
                validity_rejection_codes=tuple(validity),
                discharge_rejection_codes=tuple(discharge),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DomainRescoreValidationError(
                "adapter profile record is invalid"
            ) from exc


def _profile_binding(
    adapter: CaseworkMathVerifier,
    profile: DomainGrammarProfile,
) -> DomainAdapterProfileBinding:
    return DomainAdapterProfileBinding(
        domain_grammar_profile_id=profile.profile_id,
        bounded_comparator_pattern=profile.bounded_comparator_pattern,
        implementation_id=profile.implementation_id,
        adapter_implementation=implementation_name(adapter),
        generation_contract_id=adapter.generation_contract_id,
        obligation_contract_id=adapter.obligation_contract_id,
        verifier_contract_id=adapter.contract_id,
        aggregate_contract_id=adapter.aggregate_contract_id,
        payload_semantics_id=adapter.payload_semantics_id,
        validity_rejection_codes=tuple(sorted(adapter.validity_rejection_codes)),
        discharge_rejection_codes=tuple(
            sorted(adapter.discharge_rejection_codes)
        ),
    )


def _validate_reason_counts(
    value: tuple[tuple[str, int], ...], *, case_count: int, applied: int
) -> None:
    if (
        type(value) is not tuple
        or value != tuple(sorted(value))
        or len({reason for reason, _ in value}) != len(value)
    ):
        raise ValueError("rescore expected reason counts are invalid")
    total = 0
    for reason, count in value:
        if not isinstance(reason, str) or not reason:
            raise ValueError("rescore expected reason is invalid")
        if type(count) is not int or count <= 0:
            raise ValueError("rescore expected reason count is invalid")
        total += count
    if total + applied != case_count:
        raise ValueError("rescore expected outcomes do not cover every case")


def _expectation_record(
    applied_count: int, reason_counts: tuple[tuple[str, int], ...]
) -> dict[str, object]:
    return {
        "applied_count": applied_count,
        "reason_counts": dict(reason_counts),
    }


def _parse_expectation(
    value: object, path: str
) -> tuple[int, tuple[tuple[str, int], ...]]:
    record = _exact_record(value, _EXPECTATION_KEYS, path)
    applied = record["applied_count"]
    reasons = record["reason_counts"]
    if type(applied) is not int or applied < 0 or type(reasons) is not dict:
        raise DomainRescoreValidationError(f"{path} is invalid")
    if any(
        not isinstance(reason, str)
        or not reason
        or type(count) is not int
        or count <= 0
        for reason, count in reasons.items()
    ):
        raise DomainRescoreValidationError(f"{path} reason counts are invalid")
    return applied, tuple(sorted(reasons.items()))


@dataclass(frozen=True, slots=True)
class DomainRescoreContract:
    """Secret-free identity for one deterministic paired parser rescore."""

    source_run_sha256: str
    source_diagnostic_sha256: str
    source_environment_contract_sha256: str
    source_collection_contract_sha256: str
    source_problem_source_sha256: str
    e24_artifact_sha256: str
    e24_probe_contract_sha256: str
    e24_probe_environment_contract_sha256: str
    corrected_environment_contract_sha256: str
    e24_case_manifest_sha256: str
    rescore_case_manifest_sha256: str
    source_decision_count: int
    case_count: int
    planned_output_path: str
    legacy_profile: DomainAdapterProfileBinding
    corrected_profile: DomainAdapterProfileBinding
    legacy_profile_sha256: str
    corrected_profile_sha256: str
    expected_legacy_applied_count: int
    expected_corrected_applied_count: int
    expected_legacy_reason_counts: tuple[tuple[str, int], ...]
    expected_corrected_reason_counts: tuple[tuple[str, int], ...]
    schema_version: str = DOMAIN_RESCORE_CONTRACT_VERSION
    selection_contract: str = DOMAIN_RESCORE_SELECTION_CONTRACT
    canonical_serializer: str = CANONICAL_JSON_VERSION
    interpretation_contract: str = DOMAIN_RESCORE_INTERPRETATION_CONTRACT

    def __post_init__(self) -> None:
        if self.schema_version != DOMAIN_RESCORE_CONTRACT_VERSION:
            raise ValueError("rescore contract schema is invalid")
        for field_name in (
            "source_run_sha256",
            "source_diagnostic_sha256",
            "source_environment_contract_sha256",
            "source_collection_contract_sha256",
            "source_problem_source_sha256",
            "e24_artifact_sha256",
            "e24_probe_contract_sha256",
            "e24_probe_environment_contract_sha256",
            "corrected_environment_contract_sha256",
            "e24_case_manifest_sha256",
            "rescore_case_manifest_sha256",
            "legacy_profile_sha256",
            "corrected_profile_sha256",
        ):
            if not _is_sha256(getattr(self, field_name)):
                raise ValueError(f"rescore contract {field_name} is invalid")
        if self.source_decision_count != DOMAIN_RESCORE_EXPECTED_SOURCE_DECISIONS:
            raise ValueError("rescore source decision count is invalid")
        if self.case_count != DOMAIN_RESCORE_EXPECTED_CASES:
            raise ValueError("rescore case count is invalid")
        if self.planned_output_path != _path_record(self.planned_output_path):
            raise ValueError("rescore planned output path is invalid")
        if not isinstance(self.legacy_profile, DomainAdapterProfileBinding):
            raise ValueError("rescore legacy profile is invalid")
        if not isinstance(self.corrected_profile, DomainAdapterProfileBinding):
            raise ValueError("rescore corrected profile is invalid")
        if self.legacy_profile == self.corrected_profile:
            raise ValueError("rescore parser profiles must differ")
        if (
            self.legacy_profile.implementation_id
            != self.corrected_profile.implementation_id
        ):
            raise ValueError("rescore parser implementation identity changed")
        if (
            self.legacy_profile.adapter_implementation
            != self.corrected_profile.adapter_implementation
        ):
            raise ValueError("rescore adapter implementation identity changed")
        if self.legacy_profile_sha256 != self.legacy_profile.sha256:
            raise ValueError("rescore legacy profile digest is invalid")
        if self.corrected_profile_sha256 != self.corrected_profile.sha256:
            raise ValueError("rescore corrected profile digest is invalid")
        if self.expected_legacy_applied_count != (
            DOMAIN_RESCORE_EXPECTED_LEGACY_APPLIED
        ):
            raise ValueError("rescore legacy applied expectation is invalid")
        if self.expected_corrected_applied_count != (
            DOMAIN_RESCORE_EXPECTED_CORRECTED_APPLIED
        ):
            raise ValueError("rescore corrected applied expectation is invalid")
        if self.expected_legacy_reason_counts != (
            DOMAIN_RESCORE_EXPECTED_LEGACY_REASONS
        ):
            raise ValueError("rescore legacy reason expectation is invalid")
        if self.expected_corrected_reason_counts != (
            DOMAIN_RESCORE_EXPECTED_CORRECTED_REASONS
        ):
            raise ValueError("rescore corrected reason expectation is invalid")
        _validate_reason_counts(
            self.expected_legacy_reason_counts,
            case_count=self.case_count,
            applied=self.expected_legacy_applied_count,
        )
        _validate_reason_counts(
            self.expected_corrected_reason_counts,
            case_count=self.case_count,
            applied=self.expected_corrected_applied_count,
        )
        if not {
            reason for reason, _ in self.expected_legacy_reason_counts
        } <= self.legacy_profile.rejection_codes:
            raise ValueError("rescore legacy expected reason is unbounded")
        if not {
            reason for reason, _ in self.expected_corrected_reason_counts
        } <= self.corrected_profile.rejection_codes:
            raise ValueError("rescore corrected expected reason is unbounded")
        if self.selection_contract != DOMAIN_RESCORE_SELECTION_CONTRACT:
            raise ValueError("rescore selection contract is invalid")
        if self.canonical_serializer != CANONICAL_JSON_VERSION:
            raise ValueError("rescore canonical serializer is invalid")
        if self.interpretation_contract != (
            DOMAIN_RESCORE_INTERPRETATION_CONTRACT
        ):
            raise ValueError("rescore interpretation contract is invalid")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_run_sha256": self.source_run_sha256,
            "source_diagnostic_sha256": self.source_diagnostic_sha256,
            "source_environment_contract_sha256": (
                self.source_environment_contract_sha256
            ),
            "source_collection_contract_sha256": (
                self.source_collection_contract_sha256
            ),
            "source_problem_source_sha256": self.source_problem_source_sha256,
            "e24_artifact_sha256": self.e24_artifact_sha256,
            "e24_probe_contract_sha256": self.e24_probe_contract_sha256,
            "e24_probe_environment_contract_sha256": (
                self.e24_probe_environment_contract_sha256
            ),
            "corrected_environment_contract_sha256": (
                self.corrected_environment_contract_sha256
            ),
            "e24_case_manifest_sha256": self.e24_case_manifest_sha256,
            "rescore_case_manifest_sha256": (
                self.rescore_case_manifest_sha256
            ),
            "source_decision_count": self.source_decision_count,
            "case_count": self.case_count,
            "planned_output_path": self.planned_output_path,
            "legacy_profile": self.legacy_profile.to_record(),
            "corrected_profile": self.corrected_profile.to_record(),
            "legacy_profile_sha256": self.legacy_profile_sha256,
            "corrected_profile_sha256": self.corrected_profile_sha256,
            "expected_legacy": _expectation_record(
                self.expected_legacy_applied_count,
                self.expected_legacy_reason_counts,
            ),
            "expected_corrected": _expectation_record(
                self.expected_corrected_applied_count,
                self.expected_corrected_reason_counts,
            ),
            "selection_contract": self.selection_contract,
            "canonical_serializer": self.canonical_serializer,
            "interpretation_contract": self.interpretation_contract,
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.to_record())

    @property
    def legacy_expectation_sha256(self) -> str:
        return canonical_json_sha256(_expectation_record(
            self.expected_legacy_applied_count,
            self.expected_legacy_reason_counts,
        ))

    @property
    def corrected_expectation_sha256(self) -> str:
        return canonical_json_sha256(_expectation_record(
            self.expected_corrected_applied_count,
            self.expected_corrected_reason_counts,
        ))

    @classmethod
    def from_record(cls, value: object) -> DomainRescoreContract:
        record = _exact_record(value, _CONTRACT_KEYS, "rescore contract")
        try:
            legacy_applied, legacy_reasons = _parse_expectation(
                record["expected_legacy"], "rescore expected legacy"
            )
            corrected_applied, corrected_reasons = _parse_expectation(
                record["expected_corrected"],
                "rescore expected corrected",
            )
            return cls(
                source_run_sha256=record["source_run_sha256"],
                source_diagnostic_sha256=record[
                    "source_diagnostic_sha256"
                ],
                source_environment_contract_sha256=record[
                    "source_environment_contract_sha256"
                ],
                source_collection_contract_sha256=record[
                    "source_collection_contract_sha256"
                ],
                source_problem_source_sha256=record[
                    "source_problem_source_sha256"
                ],
                e24_artifact_sha256=record["e24_artifact_sha256"],
                e24_probe_contract_sha256=record[
                    "e24_probe_contract_sha256"
                ],
                e24_probe_environment_contract_sha256=record[
                    "e24_probe_environment_contract_sha256"
                ],
                corrected_environment_contract_sha256=record[
                    "corrected_environment_contract_sha256"
                ],
                e24_case_manifest_sha256=record[
                    "e24_case_manifest_sha256"
                ],
                rescore_case_manifest_sha256=record[
                    "rescore_case_manifest_sha256"
                ],
                source_decision_count=record["source_decision_count"],
                case_count=record["case_count"],
                planned_output_path=record["planned_output_path"],
                legacy_profile=DomainAdapterProfileBinding.from_record(
                    record["legacy_profile"]
                ),
                corrected_profile=DomainAdapterProfileBinding.from_record(
                    record["corrected_profile"]
                ),
                legacy_profile_sha256=record["legacy_profile_sha256"],
                corrected_profile_sha256=record[
                    "corrected_profile_sha256"
                ],
                expected_legacy_applied_count=legacy_applied,
                expected_corrected_applied_count=corrected_applied,
                expected_legacy_reason_counts=legacy_reasons,
                expected_corrected_reason_counts=corrected_reasons,
                schema_version=record["schema_version"],
                selection_contract=record["selection_contract"],
                canonical_serializer=record["canonical_serializer"],
                interpretation_contract=record["interpretation_contract"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DomainRescoreValidationError(
                "rescore contract record is invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class DomainRescoreAuthorization:
    """Explicit authorization for one frozen deterministic artifact."""

    preregistration_id: str
    expected_source_run_sha256: str
    expected_source_diagnostic_sha256: str
    expected_source_environment_contract_sha256: str
    expected_source_collection_contract_sha256: str
    expected_source_problem_source_sha256: str
    expected_e24_artifact_sha256: str
    expected_e24_probe_contract_sha256: str
    expected_e24_probe_environment_contract_sha256: str
    expected_corrected_environment_contract_sha256: str
    expected_e24_case_manifest_sha256: str
    expected_rescore_case_manifest_sha256: str
    expected_legacy_profile_sha256: str
    expected_corrected_profile_sha256: str
    expected_legacy_expectation_sha256: str
    expected_corrected_expectation_sha256: str
    expected_rescore_contract_sha256: str
    expected_source_decision_count: int
    expected_case_count: int
    expected_output_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.preregistration_id, str) or not (
            self.preregistration_id.strip()
        ):
            raise ValueError("rescore preregistration id must be nonempty")
        for field_name in (
            "expected_source_run_sha256",
            "expected_source_diagnostic_sha256",
            "expected_source_environment_contract_sha256",
            "expected_source_collection_contract_sha256",
            "expected_source_problem_source_sha256",
            "expected_e24_artifact_sha256",
            "expected_e24_probe_contract_sha256",
            "expected_e24_probe_environment_contract_sha256",
            "expected_corrected_environment_contract_sha256",
            "expected_e24_case_manifest_sha256",
            "expected_rescore_case_manifest_sha256",
            "expected_legacy_profile_sha256",
            "expected_corrected_profile_sha256",
            "expected_legacy_expectation_sha256",
            "expected_corrected_expectation_sha256",
            "expected_rescore_contract_sha256",
        ):
            if not _is_sha256(getattr(self, field_name)):
                raise ValueError(f"rescore authorization {field_name} is invalid")
        if self.expected_source_decision_count != (
            DOMAIN_RESCORE_EXPECTED_SOURCE_DECISIONS
        ):
            raise ValueError("authorized source decision count is invalid")
        if self.expected_case_count != DOMAIN_RESCORE_EXPECTED_CASES:
            raise ValueError("authorized rescore case count is invalid")
        if self.expected_output_path != _path_record(
            self.expected_output_path
        ):
            raise ValueError("rescore authorization output path is invalid")

    def validate(self, contract: DomainRescoreContract) -> None:
        checks = {
            "source run": (
                self.expected_source_run_sha256,
                contract.source_run_sha256,
            ),
            "source diagnostic": (
                self.expected_source_diagnostic_sha256,
                contract.source_diagnostic_sha256,
            ),
            "source environment": (
                self.expected_source_environment_contract_sha256,
                contract.source_environment_contract_sha256,
            ),
            "source collection": (
                self.expected_source_collection_contract_sha256,
                contract.source_collection_contract_sha256,
            ),
            "source problem": (
                self.expected_source_problem_source_sha256,
                contract.source_problem_source_sha256,
            ),
            "E24 artifact": (
                self.expected_e24_artifact_sha256,
                contract.e24_artifact_sha256,
            ),
            "E24 probe contract": (
                self.expected_e24_probe_contract_sha256,
                contract.e24_probe_contract_sha256,
            ),
            "E24 probe environment": (
                self.expected_e24_probe_environment_contract_sha256,
                contract.e24_probe_environment_contract_sha256,
            ),
            "corrected environment": (
                self.expected_corrected_environment_contract_sha256,
                contract.corrected_environment_contract_sha256,
            ),
            "E24 case manifest": (
                self.expected_e24_case_manifest_sha256,
                contract.e24_case_manifest_sha256,
            ),
            "rescore case manifest": (
                self.expected_rescore_case_manifest_sha256,
                contract.rescore_case_manifest_sha256,
            ),
            "legacy profile": (
                self.expected_legacy_profile_sha256,
                contract.legacy_profile_sha256,
            ),
            "corrected profile": (
                self.expected_corrected_profile_sha256,
                contract.corrected_profile_sha256,
            ),
            "legacy expectation": (
                self.expected_legacy_expectation_sha256,
                contract.legacy_expectation_sha256,
            ),
            "corrected expectation": (
                self.expected_corrected_expectation_sha256,
                contract.corrected_expectation_sha256,
            ),
            "rescore contract": (
                self.expected_rescore_contract_sha256,
                contract.sha256,
            ),
            "output path": (
                self.expected_output_path,
                contract.planned_output_path,
            ),
            "source decision count": (
                self.expected_source_decision_count,
                contract.source_decision_count,
            ),
            "rescore case count": (
                self.expected_case_count,
                contract.case_count,
            ),
        }
        for label, (expected, observed) in checks.items():
            if expected != observed:
                raise ValueError(f"authorized {label} differs")


@dataclass(frozen=True, slots=True)
class DomainRescoreCase:
    """One exact E24 decomposition draft and its reconstructed pre-state."""

    index: int
    source_case: GeneratorMicroprobeCase
    draft: DecompositionDraft
    e24_draft_sha256: str
    e24_legacy_application_status: str
    e24_legacy_reason_code: str | None
    e24_legacy_algebra_detail: str | None

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("rescore case index is invalid")
        if not isinstance(self.source_case, GeneratorMicroprobeCase):
            raise ValueError("rescore source case is invalid")
        if self.source_case.action.verb is not Verb.DECOMPOSE:
            raise ValueError("rescore source action is not DECOMPOSE")
        if not isinstance(self.draft, DecompositionDraft):
            raise ValueError("rescore draft is invalid")
        if not _is_sha256(self.e24_draft_sha256):
            raise ValueError("rescore draft digest is invalid")
        if self.e24_legacy_application_status not in {
            "applied",
            "algebra_rejected",
        }:
            raise ValueError("rescore E24 application status is invalid")
        rejected = self.e24_legacy_application_status == "algebra_rejected"
        if rejected != (self.e24_legacy_reason_code is not None):
            raise ValueError("rescore E24 reason differs from its status")
        if rejected != (self.e24_legacy_algebra_detail is not None):
            raise ValueError("rescore E24 detail differs from its status")

    @property
    def state(self) -> State:
        return self.source_case.state

    @property
    def action(self) -> Action:
        return self.source_case.action

    def manifest_record(self) -> dict[str, object]:
        source = self.source_case.manifest_record()
        return {
            "index": self.index,
            "e24_case_index": self.source_case.index,
            "episode_key": source["episode_key"],
            "problem_id": source["problem_id"],
            "action": source["action"],
            "source_observation_sha256": source[
                "source_observation_sha256"
            ],
            "probe_observation_sha256": source[
                "probe_observation_sha256"
            ],
            "source_request_sha256": source["source_request_sha256"],
            "probe_request_sha256": source["probe_request_sha256"],
            "effective_seed": source["effective_seed"],
            "source_decision_record_id": source[
                "source_decision_record_id"
            ],
            "source_generator_call_id": source[
                "source_generator_call_id"
            ],
            "e24_draft_sha256": self.e24_draft_sha256,
        }

    def legacy_reference_record(self) -> dict[str, object]:
        return {
            "application_status": self.e24_legacy_application_status,
            "reason_code": self.e24_legacy_reason_code,
            "algebra_detail": self.e24_legacy_algebra_detail,
        }


def _rescore_manifest_sha256(
    cases: tuple[DomainRescoreCase, ...] | list[DomainRescoreCase],
) -> str:
    return canonical_json_sha256([case.manifest_record() for case in cases])


@dataclass(frozen=True, slots=True)
class PreparedDomainRescore:
    """Strict-replay-sealed E25 inputs ready for deterministic execution."""

    contract: DomainRescoreContract
    source_microprobe: PreparedGeneratorMicroprobe
    cases: tuple[DomainRescoreCase, ...]
    _trust_seal: object = field(
        init=False, repr=False, compare=False, default=None
    )
    _trusted_binding: object = field(
        init=False, repr=False, compare=False, default=None
    )

    def validate_trusted(self) -> None:
        binding = (self.contract, self.source_microprobe, self.cases)
        if (
            self._trust_seal is not _PREPARED_TRUST_SEAL
            or self._trusted_binding != binding
        ):
            raise DomainRescoreValidationError(
                "rescore preparation did not come from strict replay"
            )
        self.source_microprobe.validate_trusted()
        if len(self.source_microprobe.cases) != (
            self.contract.source_decision_count
        ):
            raise DomainRescoreValidationError(
                "rescore source decision count changed"
            )
        if len(self.cases) != self.contract.case_count:
            raise DomainRescoreValidationError("rescore case count changed")
        if tuple(case.index for case in self.cases) != tuple(
            range(len(self.cases))
        ):
            raise DomainRescoreValidationError(
                "rescore cases are not in canonical order"
            )
        if _rescore_manifest_sha256(self.cases) != (
            self.contract.rescore_case_manifest_sha256
        ):
            raise DomainRescoreValidationError(
                "rescore cases differ from the authorized manifest"
            )
        if _profile_binding(
            CaseworkMathVerifier(), LEGACY_DOMAIN_GRAMMAR_PROFILE
        ) != self.contract.legacy_profile:
            raise DomainRescoreValidationError("legacy parser profile changed")
        if _profile_binding(
            CaseworkMathVerifier.corrected_v11(),
            CORRECTED_DOMAIN_GRAMMAR_PROFILE,
        ) != self.contract.corrected_profile:
            raise DomainRescoreValidationError(
                "corrected parser profile changed"
            )
        if _corrected_environment_contract(
            self.source_microprobe
        ).sha256 != self.contract.corrected_environment_contract_sha256:
            raise DomainRescoreValidationError(
                "corrected environment contract changed"
            )


@dataclass(frozen=True, slots=True)
class DomainRescoreRun:
    output_path: Path
    contract_sha256: str
    case_count: int
    legacy_applied_count: int
    corrected_applied_count: int


def _read_canonical_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = parse_canonical_json_line(line)
            except (TypeError, ValueError) as exc:
                raise DomainRescoreValidationError(
                    f"artifact line {line_number} is invalid"
                ) from exc
            if type(record) is not dict:
                raise DomainRescoreValidationError(
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


def _deserialize_decomposition_draft(value: object) -> DecompositionDraft:
    record = _exact_record(value, _DRAFT_KEYS, "E24 decomposition draft")
    if record["draft_type"] != "decomposition":
        raise DomainRescoreValidationError("E24 draft type is invalid")
    try:
        connective = Connective(record["connective"])
    except (TypeError, ValueError) as exc:
        raise DomainRescoreValidationError(
            "E24 decomposition connective is invalid"
        ) from exc
    statements = record["child_statements"]
    obligations = record["child_obligations"]
    if (
        type(statements) is not list
        or not statements
        or any(not isinstance(item, str) or not item.strip() for item in statements)
    ):
        raise DomainRescoreValidationError(
            "E24 decomposition statements are invalid"
        )
    if type(obligations) is not list or len(obligations) != len(statements):
        raise DomainRescoreValidationError(
            "E24 decomposition obligations are invalid"
        )
    decoded: list[GoalObligation] = []
    for index, value_item in enumerate(obligations):
        item = _exact_record(
            value_item,
            _OBLIGATION_KEYS,
            f"E24 decomposition obligation {index}",
        )
        try:
            decoded.append(GoalObligation(
                variable=item["variable"],
                equation=item["equation"],
                domain=item["domain"],
                objective=GoalObjective(item["objective"]),
            ))
        except (TypeError, ValueError) as exc:
            raise DomainRescoreValidationError(
                f"E24 decomposition obligation {index} is invalid"
            ) from exc
    draft = DecompositionDraft(
        connective=connective,
        child_statements=tuple(statements),
        child_obligations=tuple(decoded),
    )
    if _serialize_decomposition_draft(draft) != record:
        raise DomainRescoreValidationError(
            "E24 decomposition draft did not round trip exactly"
        )
    return draft


def _legacy_reference(value: object) -> tuple[str, str | None, str | None]:
    record = _exact_record(value, _E24_SYMBOLIC_KEYS, "E24 symbolic record")
    status = record["application_status"]
    detail = record["algebra_detail"]
    if status == "applied":
        if detail is not None:
            raise DomainRescoreValidationError(
                "applied E24 decomposition has an algebra detail"
            )
        return status, None, None
    if status != "algebra_rejected" or not isinstance(detail, str):
        raise DomainRescoreValidationError(
            "E24 decomposition application status is invalid"
        )
    prefix = "decomposition rejected: "
    if not detail.startswith(prefix) or not detail[len(prefix):]:
        raise DomainRescoreValidationError(
            "E24 decomposition detail is not a bounded adapter reason"
        )
    return status, detail[len(prefix):], detail


def _corrected_environment_contract(
    source_microprobe: PreparedGeneratorMicroprobe,
) -> EnvironmentContract:
    legacy = source_microprobe.probe_environment.contract
    corrected_adapter = CaseworkMathVerifier.corrected_v11()
    corrected = build_environment_contract(
        generation_contract=corrected_adapter,
        verifier=corrected_adapter,
        aggregate_builder=corrected_adapter,
        lens_contract=legacy.lens_contract,
        generation_schema=legacy.generation_schema,
        generation_temperature=legacy.generation_temperature,
        generation_max_tokens=legacy.generation_max_tokens,
        generation_seed_policy=legacy.generation_seed_policy,
        turn_limit=legacy.turn_limit,
    )
    legacy_record = legacy.to_record()
    corrected_record = corrected.to_record()
    changed_fields = {
        "obligation_contract",
        "generation_contract",
        "verifier_contract",
        "verifier_reason_codes_sha256",
        "aggregate_contract",
    }
    for field_name in changed_fields:
        legacy_record.pop(field_name)
        corrected_record.pop(field_name)
    if legacy_record != corrected_record:
        raise DomainRescoreValidationError(
            "corrected environment changed beyond the parser profile"
        )
    return corrected


def prepare_domain_rescore(
    *,
    source_run_path: str | Path,
    source_diagnostic_path: str | Path,
    e24_artifact_path: str | Path,
    problem_source: CaseworkProblemSource,
    planned_output_path: str | Path,
) -> PreparedDomainRescore:
    """Validate E24 and reconstruct its 33 DECOMPOSE source pre-states."""
    if not isinstance(problem_source, CaseworkProblemSource):
        raise TypeError("rescore problem source has an invalid type")
    e24_path = Path(e24_artifact_path)
    e24_summary = read_generator_microprobe_artifact(e24_path)
    records = _read_canonical_records(e24_path)
    if not records:
        raise DomainRescoreValidationError("E24 artifact is empty")
    e24_header = _exact_record(
        records[0], _E24_HEADER_KEYS, "E24 microprobe header"
    )
    if e24_header["schema_version"] != GENERATOR_MICROPROBE_HEADER_VERSION:
        raise DomainRescoreValidationError("E24 header schema is invalid")
    probe_contract = e24_header["probe_contract"]
    if type(probe_contract) is not dict:
        raise DomainRescoreValidationError("E24 probe contract is invalid")
    probe_lens_contract = probe_contract.get("probe_lens_contract")
    if not isinstance(probe_lens_contract, str):
        raise DomainRescoreValidationError(
            "E24 probe lens contract is invalid"
        )

    source_microprobe = prepare_generator_microprobe(
        run_path=source_run_path,
        diagnostic_path=source_diagnostic_path,
        problem_source=problem_source,
        probe_lens_contract=probe_lens_contract,
    )
    source_microprobe.validate_trusted()
    if source_microprobe.contract.to_record() != probe_contract:
        raise DomainRescoreValidationError(
            "E24 probe contract differs from strict E23 reconstruction"
        )
    if e24_header["probe_contract_sha256"] != source_microprobe.contract.sha256:
        raise DomainRescoreValidationError(
            "E24 probe contract digest differs from reconstruction"
        )
    if e24_summary["case_manifest_sha256"] != (
        source_microprobe.contract.case_manifest_sha256
    ):
        raise DomainRescoreValidationError(
            "E24 case manifest differs from reconstruction"
        )

    e24_cases = [
        record
        for record in records
        if record.get("schema_version") == GENERATOR_MICROPROBE_CASE_VERSION
    ]
    if len(e24_cases) != len(source_microprobe.cases):
        raise DomainRescoreValidationError(
            "E24 case count differs from strict reconstruction"
        )

    selected: list[DomainRescoreCase] = []
    for source_case, record in zip(
        source_microprobe.cases, e24_cases, strict=True
    ):
        source_manifest = source_case.manifest_record()
        if {
            key: record.get(key) for key in source_manifest
        } != source_manifest:
            raise DomainRescoreValidationError(
                f"E24 case {source_case.index} differs from reconstruction"
            )
        if source_case.action.verb is not Verb.DECOMPOSE:
            continue
        decode = _exact_record(
            record.get("decode"), _E24_DECODE_KEYS, "E24 decode record"
        )
        if decode["status"] != "decoded" or decode["error"] is not None:
            raise DomainRescoreValidationError(
                f"E24 DECOMPOSE case {source_case.index} is not decoded"
            )
        draft_record = decode["draft"]
        draft = _deserialize_decomposition_draft(draft_record)
        status, reason, detail = _legacy_reference(
            record.get("counterfactual_symbolic")
        )
        selected.append(DomainRescoreCase(
            index=len(selected),
            source_case=source_case,
            draft=draft,
            e24_draft_sha256=canonical_json_sha256(draft_record),
            e24_legacy_application_status=status,
            e24_legacy_reason_code=reason,
            e24_legacy_algebra_detail=detail,
        ))

    frozen_cases = tuple(selected)
    if len(source_microprobe.cases) != DOMAIN_RESCORE_EXPECTED_SOURCE_DECISIONS:
        raise DomainRescoreValidationError(
            "E24 source does not contain the expected 66 decisions"
        )
    if len(frozen_cases) != DOMAIN_RESCORE_EXPECTED_CASES:
        raise DomainRescoreValidationError(
            "E24 source does not contain exactly 33 DECOMPOSE drafts"
        )
    legacy_profile = _profile_binding(
        CaseworkMathVerifier(), LEGACY_DOMAIN_GRAMMAR_PROFILE
    )
    corrected_profile = _profile_binding(
        CaseworkMathVerifier.corrected_v11(),
        CORRECTED_DOMAIN_GRAMMAR_PROFILE,
    )
    corrected_environment = _corrected_environment_contract(
        source_microprobe
    )
    contract = DomainRescoreContract(
        source_run_sha256=source_microprobe.contract.source_run_sha256,
        source_diagnostic_sha256=(
            source_microprobe.contract.source_diagnostic_sha256
        ),
        source_environment_contract_sha256=(
            source_microprobe.contract.source_environment_contract_sha256
        ),
        source_collection_contract_sha256=(
            source_microprobe.contract.source_collection_contract_sha256
        ),
        source_problem_source_sha256=(
            source_microprobe.contract.source_problem_source_sha256
        ),
        e24_artifact_sha256=file_sha256(e24_path),
        e24_probe_contract_sha256=source_microprobe.contract.sha256,
        e24_probe_environment_contract_sha256=(
            source_microprobe.contract.probe_environment_contract_sha256
        ),
        corrected_environment_contract_sha256=corrected_environment.sha256,
        e24_case_manifest_sha256=(
            source_microprobe.contract.case_manifest_sha256
        ),
        rescore_case_manifest_sha256=_rescore_manifest_sha256(frozen_cases),
        source_decision_count=len(source_microprobe.cases),
        case_count=len(frozen_cases),
        planned_output_path=_path_record(planned_output_path),
        legacy_profile=legacy_profile,
        corrected_profile=corrected_profile,
        legacy_profile_sha256=legacy_profile.sha256,
        corrected_profile_sha256=corrected_profile.sha256,
        expected_legacy_applied_count=(
            DOMAIN_RESCORE_EXPECTED_LEGACY_APPLIED
        ),
        expected_corrected_applied_count=(
            DOMAIN_RESCORE_EXPECTED_CORRECTED_APPLIED
        ),
        expected_legacy_reason_counts=(
            DOMAIN_RESCORE_EXPECTED_LEGACY_REASONS
        ),
        expected_corrected_reason_counts=(
            DOMAIN_RESCORE_EXPECTED_CORRECTED_REASONS
        ),
    )
    prepared = PreparedDomainRescore(
        contract=contract,
        source_microprobe=source_microprobe,
        cases=frozen_cases,
    )
    binding = (prepared.contract, prepared.source_microprobe, prepared.cases)
    object.__setattr__(prepared, "_trust_seal", _PREPARED_TRUST_SEAL)
    object.__setattr__(prepared, "_trusted_binding", binding)
    prepared.validate_trusted()
    return prepared


CoverageValidator = Callable[
    [GoalObligation, tuple[GoalObligation, ...]], CoverageDecision
]


def score_decomposition_draft(
    *,
    state: State,
    action: Action,
    draft: DecompositionDraft,
    adapter: CaseworkMathVerifier,
    coverage_validator: CoverageValidator,
    profile: DomainAdapterProfileBinding,
) -> dict[str, object]:
    """Apply one exact draft and retain bounded parser diagnostics."""
    if not isinstance(state, State):
        raise TypeError("rescore state has an invalid type")
    if not isinstance(action, Action) or action.verb is not Verb.DECOMPOSE:
        raise TypeError("rescore action must be DECOMPOSE")
    if not isinstance(draft, DecompositionDraft):
        raise TypeError("rescore draft has an invalid type")
    if not isinstance(profile, DomainAdapterProfileBinding):
        raise TypeError("rescore profile has an invalid type")
    owner = get(state.graph, action.target)
    if not isinstance(owner, Goal):
        raise DomainRescoreValidationError("rescore target is not a Goal")
    coverage = coverage_validator(owner.obligation, draft.child_obligations)
    if not isinstance(coverage, CoverageDecision):
        raise DomainRescoreValidationError(
            "coverage validator returned an invalid decision"
        )
    direct = decompose(state.graph, action.target, draft, adapter)
    if isinstance(direct, Err):
        prefix = "decomposition rejected: "
        if not direct.message.startswith(prefix):
            raise DomainRescoreValidationError(
                f"rescore reached a non-parser rejection: {direct.message}"
            )
        reason = direct.message[len(prefix):]
        if reason not in profile.rejection_codes:
            raise DomainRescoreValidationError(
                f"rescore returned unbounded reason {reason!r}"
            )
        if coverage.accepted or coverage.reason_code != reason:
            raise DomainRescoreValidationError(
                "coverage and action rejection diagnostics differ"
            )
        status = "algebra_rejected"
        canonical_children = None
    elif isinstance(direct, Ok):
        if not coverage.accepted or coverage.reason_code is not None:
            raise DomainRescoreValidationError(
                "coverage and action acceptance diagnostics differ"
            )
        reason = None
        status = "applied"
        canonical_children = [
            serialize_obligation(child.obligation)
            for child in direct_child_goals(direct.graph, action.target)
        ]
        if len(canonical_children) != len(draft.child_obligations):
            raise DomainRescoreValidationError(
                "applied rescore returned the wrong child count"
            )
    else:
        raise DomainRescoreValidationError(
            "decomposition returned an invalid action result"
        )
    return {
        "profile_id": profile.domain_grammar_profile_id,
        "application_status": status,
        "reason_code": reason,
        "canonical_children": canonical_children,
        "coverage": {
            "accepted": coverage.accepted,
            "context_digest": coverage.context_digest,
            "reason_code": coverage.reason_code,
        },
    }


def _header(
    prepared: PreparedDomainRescore,
    authorization: DomainRescoreAuthorization,
) -> dict[str, object]:
    contract = prepared.contract
    return {
        "schema_version": DOMAIN_RESCORE_HEADER_VERSION,
        "canonical_serializer": CANONICAL_JSON_VERSION,
        "preregistration_id": authorization.preregistration_id,
        "rescore_contract": contract.to_record(),
        "rescore_contract_sha256": contract.sha256,
        "source_run_sha256": contract.source_run_sha256,
        "source_diagnostic_sha256": contract.source_diagnostic_sha256,
        "source_environment_contract_sha256": (
            contract.source_environment_contract_sha256
        ),
        "source_collection_contract_sha256": (
            contract.source_collection_contract_sha256
        ),
        "source_problem_source_sha256": contract.source_problem_source_sha256,
        "e24_artifact_sha256": contract.e24_artifact_sha256,
        "e24_probe_contract_sha256": contract.e24_probe_contract_sha256,
        "e24_probe_environment_contract_sha256": (
            contract.e24_probe_environment_contract_sha256
        ),
        "corrected_environment_contract_sha256": (
            contract.corrected_environment_contract_sha256
        ),
        "e24_case_manifest_sha256": contract.e24_case_manifest_sha256,
        "rescore_case_manifest_sha256": (
            contract.rescore_case_manifest_sha256
        ),
        "legacy_profile_sha256": contract.legacy_profile_sha256,
        "corrected_profile_sha256": contract.corrected_profile_sha256,
        "legacy_expectation_sha256": contract.legacy_expectation_sha256,
        "corrected_expectation_sha256": (
            contract.corrected_expectation_sha256
        ),
    }


class _ArtifactWriter:
    def __init__(self, path: Path, header: Mapping[str, object]) -> None:
        self.path = path
        self._handle: BinaryIO = path.open("xb")
        self._closed = False
        self.write(header)

    def write(self, record: Mapping[str, object]) -> None:
        if self._closed:
            raise RuntimeError("rescore artifact writer is closed")
        self._handle.write(canonical_json_bytes(dict(record)) + b"\n")
        self._handle.flush()

    def finish(self, summary: Mapping[str, object]) -> None:
        self.write(summary)
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._closed = True

    def abort(self) -> None:
        if not self._closed:
            self._handle.flush()
            self._handle.close()
            self._closed = True


def _summary(
    contract: DomainRescoreContract,
    case_records: list[dict[str, object]],
) -> dict[str, object]:
    legacy_status = Counter(
        record["legacy"]["application_status"] for record in case_records
    )
    corrected_status = Counter(
        record["corrected"]["application_status"] for record in case_records
    )
    legacy_reasons = Counter(
        record["legacy"]["reason_code"]
        for record in case_records
        if record["legacy"]["reason_code"] is not None
    )
    corrected_reasons = Counter(
        record["corrected"]["reason_code"]
        for record in case_records
        if record["corrected"]["reason_code"] is not None
    )
    paired = Counter(
        (
            f'{record["legacy"]["application_status"]}->'
            f'{record["corrected"]["application_status"]}'
        )
        for record in case_records
    )
    return {
        "schema_version": DOMAIN_RESCORE_SUMMARY_VERSION,
        "rescore_contract_sha256": contract.sha256,
        "rescore_case_manifest_sha256": (
            contract.rescore_case_manifest_sha256
        ),
        "corrected_environment_contract_sha256": (
            contract.corrected_environment_contract_sha256
        ),
        "legacy_profile_sha256": contract.legacy_profile_sha256,
        "corrected_profile_sha256": contract.corrected_profile_sha256,
        "legacy_expectation_sha256": contract.legacy_expectation_sha256,
        "corrected_expectation_sha256": (
            contract.corrected_expectation_sha256
        ),
        "result_manifest_sha256": canonical_json_sha256(case_records),
        "case_count": len(case_records),
        "legacy_applied_count": legacy_status["applied"],
        "corrected_applied_count": corrected_status["applied"],
        "legacy_application_counts": dict(sorted(legacy_status.items())),
        "corrected_application_counts": dict(
            sorted(corrected_status.items())
        ),
        "legacy_reason_counts": dict(sorted(legacy_reasons.items())),
        "corrected_reason_counts": dict(sorted(corrected_reasons.items())),
        "paired_status_counts": dict(sorted(paired.items())),
        "registered_legacy_applied_count": (
            contract.expected_legacy_applied_count
        ),
        "registered_corrected_applied_count": (
            contract.expected_corrected_applied_count
        ),
        "registered_legacy_reason_counts": dict(
            contract.expected_legacy_reason_counts
        ),
        "registered_corrected_reason_counts": dict(
            contract.expected_corrected_reason_counts
        ),
        "interpretation_contract": DOMAIN_RESCORE_INTERPRETATION_CONTRACT,
    }


def _validate_registered_outcomes(
    contract: DomainRescoreContract, summary: Mapping[str, object]
) -> None:
    checks = {
        "legacy applied count": (
            summary["legacy_applied_count"],
            contract.expected_legacy_applied_count,
        ),
        "corrected applied count": (
            summary["corrected_applied_count"],
            contract.expected_corrected_applied_count,
        ),
        "legacy reason counts": (
            summary["legacy_reason_counts"],
            dict(contract.expected_legacy_reason_counts),
        ),
        "corrected reason counts": (
            summary["corrected_reason_counts"],
            dict(contract.expected_corrected_reason_counts),
        ),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise DomainRescoreValidationError(
                f"rescore {label} differs from the registered expectation"
            )


def run_domain_rescore(
    *,
    output_path: str | Path,
    prepared: PreparedDomainRescore,
    authorization: DomainRescoreAuthorization,
) -> DomainRescoreRun:
    """Produce one authorized no-network paired legacy/v1.1 rescore."""
    if not isinstance(prepared, PreparedDomainRescore):
        raise TypeError("prepared rescore has an invalid type")
    if not isinstance(authorization, DomainRescoreAuthorization):
        raise TypeError("rescore authorization has an invalid type")
    prepared.validate_trusted()
    authorization.validate(prepared.contract)
    final_path = Path(output_path)
    if _path_record(final_path) != prepared.contract.planned_output_path:
        raise ValueError("rescore output differs from the frozen output path")
    partial_path = final_path.with_name(f"{final_path.name}.partial")
    if final_path.exists() or partial_path.exists():
        raise FileExistsError("rescore output or partial output already exists")
    if not final_path.parent.exists() or not final_path.parent.is_dir():
        raise FileNotFoundError("rescore output directory does not exist")

    writer: _ArtifactWriter | None = None
    try:
        writer = _ArtifactWriter(
            partial_path, _header(prepared, authorization)
        )
        case_records: list[dict[str, object]] = []
        legacy_adapter = CaseworkMathVerifier()
        corrected_adapter = CaseworkMathVerifier.corrected_v11()
        for case in prepared.cases:
            legacy = score_decomposition_draft(
                state=case.state,
                action=case.action,
                draft=case.draft,
                adapter=legacy_adapter,
                coverage_validator=validate_decomposition_coverage,
                profile=prepared.contract.legacy_profile,
            )
            corrected = score_decomposition_draft(
                state=case.state,
                action=case.action,
                draft=case.draft,
                adapter=corrected_adapter,
                coverage_validator=validate_decomposition_coverage_v11,
                profile=prepared.contract.corrected_profile,
            )
            reference = case.legacy_reference_record()
            expected_legacy = {
                "application_status": legacy["application_status"],
                "reason_code": legacy["reason_code"],
                "algebra_detail": (
                    None
                    if legacy["reason_code"] is None
                    else f'decomposition rejected: {legacy["reason_code"]}'
                ),
            }
            if reference != expected_legacy:
                raise DomainRescoreValidationError(
                    f"legacy rescore differs from E24 case {case.source_case.index}"
                )
            record = {
                "schema_version": DOMAIN_RESCORE_CASE_VERSION,
                **case.manifest_record(),
                "decoded_draft": _serialize_decomposition_draft(case.draft),
                "e24_legacy_reference": reference,
                "legacy": legacy,
                "corrected": corrected,
            }
            case_records.append(record)
            writer.write(record)

        summary = _summary(prepared.contract, case_records)
        _validate_registered_outcomes(prepared.contract, summary)
        writer.finish(summary)
        validated = read_domain_rescore_artifact(partial_path)
        if validated != summary:
            raise DomainRescoreValidationError(
                "rescore artifact summary changed during validation"
            )
        os.link(partial_path, final_path)
        partial_path.unlink()
        return DomainRescoreRun(
            output_path=final_path,
            contract_sha256=prepared.contract.sha256,
            case_count=len(case_records),
            legacy_applied_count=summary["legacy_applied_count"],
            corrected_applied_count=summary["corrected_applied_count"],
        )
    finally:
        if writer is not None:
            writer.abort()


def _validate_obligation_record(value: object, path: str) -> None:
    record = _exact_record(value, _OBLIGATION_KEYS, path)
    for field_name in ("variable", "equation", "domain"):
        _nonempty_text(record[field_name], f"{path}.{field_name}")
    try:
        GoalObjective(record["objective"])
    except (TypeError, ValueError) as exc:
        raise DomainRescoreValidationError(
            f"{path}.objective is invalid"
        ) from exc


def _validate_profile_result(
    value: object,
    profile: DomainAdapterProfileBinding,
    path: str,
) -> str:
    record = _exact_record(value, _PROFILE_RESULT_KEYS, path)
    if record["profile_id"] != profile.domain_grammar_profile_id:
        raise DomainRescoreValidationError(f"{path} profile id differs")
    status = record["application_status"]
    if status not in {"applied", "algebra_rejected"}:
        raise DomainRescoreValidationError(f"{path} status is invalid")
    reason = record["reason_code"]
    rejected = status == "algebra_rejected"
    if rejected:
        if reason not in profile.rejection_codes:
            raise DomainRescoreValidationError(f"{path} reason is unbounded")
    elif reason is not None:
        raise DomainRescoreValidationError(
            f"{path} applied result has a reason"
        )
    children = record["canonical_children"]
    if status == "applied":
        if type(children) is not list or not children:
            raise DomainRescoreValidationError(
                f"{path} applied result lacks canonical children"
            )
        for index, child in enumerate(children):
            _validate_obligation_record(child, f"{path}.children[{index}]")
    elif children is not None:
        raise DomainRescoreValidationError(
            f"{path} rejected result retains canonical children"
        )
    coverage = _exact_record(
        record["coverage"], _COVERAGE_KEYS, f"{path}.coverage"
    )
    if type(coverage["accepted"]) is not bool:
        raise DomainRescoreValidationError(
            f"{path}.coverage.accepted must be boolean"
        )
    _sha256(coverage["context_digest"], f"{path}.coverage.context_digest")
    if coverage["accepted"] != (status == "applied"):
        raise DomainRescoreValidationError(
            f"{path} coverage differs from application status"
        )
    if coverage["reason_code"] != reason:
        raise DomainRescoreValidationError(
            f"{path} coverage reason differs from application reason"
        )
    return status


def read_domain_rescore_artifact(
    path: str | Path,
) -> dict[str, object]:
    """Validate a completed deterministic rescore and return its summary."""
    records = _read_canonical_records(Path(path))
    if len(records) < 2:
        raise DomainRescoreValidationError("rescore artifact is incomplete")
    header = _exact_record(records[0], _HEADER_KEYS, "rescore header")
    if header["schema_version"] != DOMAIN_RESCORE_HEADER_VERSION:
        raise DomainRescoreValidationError("rescore header schema is invalid")
    if header["canonical_serializer"] != CANONICAL_JSON_VERSION:
        raise DomainRescoreValidationError(
            "rescore header serializer is invalid"
        )
    _nonempty_text(
        header["preregistration_id"], "rescore preregistration id"
    )
    contract = DomainRescoreContract.from_record(
        header["rescore_contract"]
    )
    if header["rescore_contract_sha256"] != contract.sha256:
        raise DomainRescoreValidationError(
            "rescore header contract digest is invalid"
        )
    header_checks = {
        "source_run_sha256": contract.source_run_sha256,
        "source_diagnostic_sha256": contract.source_diagnostic_sha256,
        "source_environment_contract_sha256": (
            contract.source_environment_contract_sha256
        ),
        "source_collection_contract_sha256": (
            contract.source_collection_contract_sha256
        ),
        "source_problem_source_sha256": contract.source_problem_source_sha256,
        "e24_artifact_sha256": contract.e24_artifact_sha256,
        "e24_probe_contract_sha256": contract.e24_probe_contract_sha256,
        "e24_probe_environment_contract_sha256": (
            contract.e24_probe_environment_contract_sha256
        ),
        "corrected_environment_contract_sha256": (
            contract.corrected_environment_contract_sha256
        ),
        "e24_case_manifest_sha256": contract.e24_case_manifest_sha256,
        "rescore_case_manifest_sha256": (
            contract.rescore_case_manifest_sha256
        ),
        "legacy_profile_sha256": contract.legacy_profile_sha256,
        "corrected_profile_sha256": contract.corrected_profile_sha256,
        "legacy_expectation_sha256": contract.legacy_expectation_sha256,
        "corrected_expectation_sha256": (
            contract.corrected_expectation_sha256
        ),
    }
    for field_name, expected in header_checks.items():
        if header[field_name] != expected:
            raise DomainRescoreValidationError(
                f"rescore header {field_name} differs from its contract"
            )

    summary = _exact_record(records[-1], _SUMMARY_KEYS, "rescore summary")
    if summary["schema_version"] != DOMAIN_RESCORE_SUMMARY_VERSION:
        raise DomainRescoreValidationError("rescore summary schema is invalid")
    case_records: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    for expected_index, value in enumerate(records[1:-1]):
        record = _exact_record(
            value, _CASE_KEYS, f"rescore case {expected_index}"
        )
        if record["schema_version"] != DOMAIN_RESCORE_CASE_VERSION:
            raise DomainRescoreValidationError(
                f"rescore case {expected_index} schema is invalid"
            )
        if record["index"] != expected_index:
            raise DomainRescoreValidationError(
                "rescore cases are not in canonical order"
            )
        _nonnegative_integer(
            record["e24_case_index"],
            f"rescore case {expected_index}.e24_case_index",
        )
        _nonempty_text(
            record["episode_key"], f"rescore case {expected_index}.episode_key"
        )
        _nonempty_text(
            record["problem_id"], f"rescore case {expected_index}.problem_id"
        )
        action = deserialize_action(
            record["action"], f"rescore case {expected_index}.action"
        )
        if action.verb is not Verb.DECOMPOSE:
            raise DomainRescoreValidationError(
                "rescore case action is not DECOMPOSE"
            )
        for field_name in (
            "source_observation_sha256",
            "probe_observation_sha256",
            "source_request_sha256",
            "probe_request_sha256",
            "e24_draft_sha256",
        ):
            _sha256(
                record[field_name],
                f"rescore case {expected_index}.{field_name}",
            )
        _nonnegative_integer(
            record["effective_seed"],
            f"rescore case {expected_index}.effective_seed",
        )
        _nonempty_text(
            record["source_decision_record_id"],
            f"rescore case {expected_index}.source_decision_record_id",
        )
        call_id = record["source_generator_call_id"]
        if (
            not isinstance(call_id, str)
            or len(call_id) != 32
            or set(call_id) - _SHA256_CHARS
        ):
            raise DomainRescoreValidationError(
                f"rescore case {expected_index} source call id is invalid"
            )
        draft = _deserialize_decomposition_draft(record["decoded_draft"])
        if canonical_json_sha256(_serialize_decomposition_draft(draft)) != (
            record["e24_draft_sha256"]
        ):
            raise DomainRescoreValidationError(
                f"rescore case {expected_index} draft digest differs"
            )
        reference = _exact_record(
            record["e24_legacy_reference"],
            _LEGACY_REFERENCE_KEYS,
            f"rescore case {expected_index}.e24_legacy_reference",
        )
        reference_status = reference["application_status"]
        if reference_status not in {"applied", "algebra_rejected"}:
            raise DomainRescoreValidationError(
                f"rescore case {expected_index} E24 status is invalid"
            )
        if reference_status == "applied":
            if reference["reason_code"] is not None or (
                reference["algebra_detail"] is not None
            ):
                raise DomainRescoreValidationError(
                    f"rescore case {expected_index} E24 acceptance is invalid"
                )
        else:
            reason = reference["reason_code"]
            if reason not in contract.legacy_profile.rejection_codes:
                raise DomainRescoreValidationError(
                    f"rescore case {expected_index} E24 reason is unbounded"
                )
            if reference["algebra_detail"] != (
                f"decomposition rejected: {reason}"
            ):
                raise DomainRescoreValidationError(
                    f"rescore case {expected_index} E24 detail differs"
                )
        legacy_status = _validate_profile_result(
            record["legacy"],
            contract.legacy_profile,
            f"rescore case {expected_index}.legacy",
        )
        _validate_profile_result(
            record["corrected"],
            contract.corrected_profile,
            f"rescore case {expected_index}.corrected",
        )
        if legacy_status != reference_status or (
            record["legacy"]["reason_code"] != reference["reason_code"]
        ):
            raise DomainRescoreValidationError(
                f"rescore case {expected_index} legacy result differs from E24"
            )
        manifest.append({key: record[key] for key in _MANIFEST_KEYS})
        case_records.append(record)

    if len(case_records) != contract.case_count:
        raise DomainRescoreValidationError(
            "rescore artifact case count differs from its contract"
        )
    if canonical_json_sha256(manifest) != (
        contract.rescore_case_manifest_sha256
    ):
        raise DomainRescoreValidationError(
            "rescore artifact manifest differs from its contract"
        )
    expected_summary = _summary(contract, case_records)
    _validate_registered_outcomes(contract, expected_summary)
    if summary != expected_summary:
        raise DomainRescoreValidationError(
            "rescore summary differs from its validated cases"
        )
    return dict(summary)


_PROFILE_KEYS = frozenset({
    "domain_grammar_profile_id",
    "bounded_comparator_pattern",
    "implementation_id",
    "adapter_implementation",
    "generation_contract_id",
    "obligation_contract_id",
    "verifier_contract_id",
    "aggregate_contract_id",
    "payload_semantics_id",
    "validity_rejection_codes",
    "discharge_rejection_codes",
})
_EXPECTATION_KEYS = frozenset({"applied_count", "reason_counts"})
_CONTRACT_KEYS = frozenset({
    "schema_version",
    "source_run_sha256",
    "source_diagnostic_sha256",
    "source_environment_contract_sha256",
    "source_collection_contract_sha256",
    "source_problem_source_sha256",
    "e24_artifact_sha256",
    "e24_probe_contract_sha256",
    "e24_probe_environment_contract_sha256",
    "corrected_environment_contract_sha256",
    "e24_case_manifest_sha256",
    "rescore_case_manifest_sha256",
    "source_decision_count",
    "case_count",
    "planned_output_path",
    "legacy_profile",
    "corrected_profile",
    "legacy_profile_sha256",
    "corrected_profile_sha256",
    "expected_legacy",
    "expected_corrected",
    "selection_contract",
    "canonical_serializer",
    "interpretation_contract",
})
_E24_HEADER_KEYS = frozenset({
    "schema_version",
    "canonical_serializer",
    "preregistration_id",
    "probe_contract",
    "probe_contract_sha256",
    "source_environment_contract_sha256",
    "probe_environment_contract_sha256",
    "case_manifest_sha256",
})
_E24_DECODE_KEYS = frozenset({"status", "error", "draft"})
_E24_SYMBOLIC_KEYS = frozenset({
    "interpretation_contract",
    "application_status",
    "algebra_detail",
    "macro_outcome",
    "closure_events",
    "created_claim",
    "counterfactual_reward",
    "counterfactual_terminal",
})
_DRAFT_KEYS = frozenset({
    "draft_type",
    "connective",
    "child_statements",
    "child_obligations",
})
_OBLIGATION_KEYS = frozenset({
    "variable", "equation", "domain", "objective"
})
_MANIFEST_KEYS = frozenset({
    "index",
    "e24_case_index",
    "episode_key",
    "problem_id",
    "action",
    "source_observation_sha256",
    "probe_observation_sha256",
    "source_request_sha256",
    "probe_request_sha256",
    "effective_seed",
    "source_decision_record_id",
    "source_generator_call_id",
    "e24_draft_sha256",
})
_LEGACY_REFERENCE_KEYS = frozenset({
    "application_status", "reason_code", "algebra_detail"
})
_COVERAGE_KEYS = frozenset({
    "accepted", "context_digest", "reason_code"
})
_PROFILE_RESULT_KEYS = frozenset({
    "profile_id",
    "application_status",
    "reason_code",
    "canonical_children",
    "coverage",
})
_HEADER_KEYS = frozenset({
    "schema_version",
    "canonical_serializer",
    "preregistration_id",
    "rescore_contract",
    "rescore_contract_sha256",
    "source_run_sha256",
    "source_diagnostic_sha256",
    "source_environment_contract_sha256",
    "source_collection_contract_sha256",
    "source_problem_source_sha256",
    "e24_artifact_sha256",
    "e24_probe_contract_sha256",
    "e24_probe_environment_contract_sha256",
    "corrected_environment_contract_sha256",
    "e24_case_manifest_sha256",
    "rescore_case_manifest_sha256",
    "legacy_profile_sha256",
    "corrected_profile_sha256",
    "legacy_expectation_sha256",
    "corrected_expectation_sha256",
})
_CASE_KEYS = _MANIFEST_KEYS | frozenset({
    "schema_version",
    "decoded_draft",
    "e24_legacy_reference",
    "legacy",
    "corrected",
})
_SUMMARY_KEYS = frozenset({
    "schema_version",
    "rescore_contract_sha256",
    "rescore_case_manifest_sha256",
    "corrected_environment_contract_sha256",
    "legacy_profile_sha256",
    "corrected_profile_sha256",
    "legacy_expectation_sha256",
    "corrected_expectation_sha256",
    "result_manifest_sha256",
    "case_count",
    "legacy_applied_count",
    "corrected_applied_count",
    "legacy_application_counts",
    "corrected_application_counts",
    "legacy_reason_counts",
    "corrected_reason_counts",
    "paired_status_counts",
    "registered_legacy_applied_count",
    "registered_corrected_applied_count",
    "registered_legacy_reason_counts",
    "registered_corrected_reason_counts",
    "interpretation_contract",
})


__all__ = [
    "DOMAIN_RESCORE_CASE_VERSION",
    "DOMAIN_RESCORE_CONTRACT_VERSION",
    "DOMAIN_RESCORE_EXPECTED_CASES",
    "DOMAIN_RESCORE_EXPECTED_SOURCE_DECISIONS",
    "DOMAIN_RESCORE_HEADER_VERSION",
    "DOMAIN_RESCORE_INTERPRETATION_CONTRACT",
    "DOMAIN_RESCORE_SELECTION_CONTRACT",
    "DOMAIN_RESCORE_SUMMARY_VERSION",
    "DomainAdapterProfileBinding",
    "DomainRescoreAuthorization",
    "DomainRescoreCase",
    "DomainRescoreContract",
    "DomainRescoreRun",
    "DomainRescoreValidationError",
    "PreparedDomainRescore",
    "prepare_domain_rescore",
    "read_domain_rescore_artifact",
    "run_domain_rescore",
    "score_decomposition_draft",
]
