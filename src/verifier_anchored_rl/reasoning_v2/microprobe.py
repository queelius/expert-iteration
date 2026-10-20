"""Paired generator-only probes over sealed reasoning v2 decisions."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, BinaryIO, Mapping

from ..llm_client import (
    LLMResponseError,
    OpenAICompatibleAttempt,
    OpenAICompatibleLLMClient,
    build_openai_compatible_json_body,
)
from .actions import (
    DecompositionDraft,
    Err,
    RefutationDraft,
    ResultDraft,
    SolutionDraft,
    decompose,
    propose_refutation,
    propose_result,
    propose_solution,
)
from .casework_source import CaseworkProblemSource
from .contracts import (
    CANONICAL_JSON_VERSION,
    CollectionContract,
    EnvironmentContract,
    ModelIdentity,
    ModelTransportContract,
    canonical_json_bytes,
    canonical_json_sha256,
)
from .diagnostics import (
    DIAGNOSTIC_ATTEMPT_VERSION,
    DIAGNOSTIC_CALL_VERSION,
    build_attempt_record,
    build_call_record,
    read_diagnostic_sidecar,
    validate_attempt_record,
    validate_call_record,
)
from .lenses import (
    LEGACY_GENERATION_SCHEMA_VERSION,
    SUPPORTED_LENS_CONTRACT_VERSIONS,
    build_policy_lens_request,
    decode_policy_payload,
    llm_transport_contract,
)
from .math_adapter import CaseworkMathVerifier
from .model import Action, Claim, OutcomeCode, State, Verb, get
from .observation import observation_sha256, serialize_verifier_context
from .records import (
    deserialize_action,
    deserialize_outcome,
    parse_canonical_json_line,
    serialize_action,
    serialize_obligation,
    serialize_outcome,
)
from .replay import ReplayedRun, replay_canonical_run
from .runtime import BoundEnvironment
from .transition import PolicyDraft, PolicyPayloadError


GENERATOR_MICROPROBE_CONTRACT_VERSION = (
    "generator-paired-microprobe-contract-v1"
)
GENERATOR_MICROPROBE_HEADER_VERSION = (
    "generator-paired-microprobe-header-v1"
)
GENERATOR_MICROPROBE_CASE_VERSION = "generator-paired-microprobe-case-v1"
GENERATOR_MICROPROBE_SUMMARY_VERSION = (
    "generator-paired-microprobe-summary-v1"
)
GENERATOR_MICROPROBE_SELECTION_CONTRACT = (
    "all-sealed-decisions-in-canonical-order-v1"
)
GENERATOR_MICROPROBE_INTERPRETATION_CONTRACT = (
    "counterfactual-symbolic-diagnostics-not-training-or-reward-evidence-v1"
)

_SHA256_CHARS = frozenset("0123456789abcdef")
_PREPARED_TRUST_SEAL = object()


class GeneratorMicroprobeValidationError(ValueError):
    """Raised when a generator microprobe contract or artifact is invalid."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA256_CHARS
    )


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one exact file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class GeneratorMicroprobeContract:
    """Secret-free binding for one all-row generator-only probe."""

    source_run_sha256: str
    source_diagnostic_sha256: str
    source_environment_contract_sha256: str
    probe_environment_contract_sha256: str
    source_collection_contract_sha256: str
    source_problem_source_sha256: str
    case_manifest_sha256: str
    source_decision_count: int
    source_lens_contract: str
    probe_lens_contract: str
    generator: ModelIdentity
    generator_transport: ModelTransportContract
    generation_schema: str
    generation_temperature: float
    generation_max_tokens: int
    generation_seed_policy: str
    generation_seed: int | None
    schema_version: str = GENERATOR_MICROPROBE_CONTRACT_VERSION
    selection_contract: str = GENERATOR_MICROPROBE_SELECTION_CONTRACT
    canonical_serializer: str = CANONICAL_JSON_VERSION
    interpretation_contract: str = (
        GENERATOR_MICROPROBE_INTERPRETATION_CONTRACT
    )

    def __post_init__(self) -> None:
        if self.schema_version != GENERATOR_MICROPROBE_CONTRACT_VERSION:
            raise ValueError("microprobe contract schema is invalid")
        for field_name in (
            "source_run_sha256",
            "source_diagnostic_sha256",
            "source_environment_contract_sha256",
            "probe_environment_contract_sha256",
            "source_collection_contract_sha256",
            "source_problem_source_sha256",
            "case_manifest_sha256",
        ):
            if not _is_sha256(getattr(self, field_name)):
                raise ValueError(f"microprobe {field_name} is invalid")
        if type(self.source_decision_count) is not int or self.source_decision_count <= 0:
            raise ValueError("microprobe source decision count is invalid")
        if self.source_lens_contract not in SUPPORTED_LENS_CONTRACT_VERSIONS:
            raise ValueError("microprobe source lens contract is unsupported")
        if self.probe_lens_contract not in SUPPORTED_LENS_CONTRACT_VERSIONS:
            raise ValueError("microprobe probe lens contract is unsupported")
        if self.probe_lens_contract == self.source_lens_contract:
            raise ValueError("microprobe must vary the lens contract")
        if not isinstance(self.generator, ModelIdentity):
            raise ValueError("microprobe generator identity is invalid")
        if not isinstance(self.generator_transport, ModelTransportContract):
            raise ValueError("microprobe generator transport is invalid")
        if self.generator_transport.base_url is None:
            raise ValueError("microprobe requires a hosted generator transport")
        if self.generator_transport.model != self.generator.model:
            raise ValueError("microprobe generator model differs from transport")
        if self.generation_schema != LEGACY_GENERATION_SCHEMA_VERSION:
            raise ValueError("microprobe generation schema is unsupported")
        if type(self.generation_temperature) is not float:
            raise ValueError("microprobe generation temperature is invalid")
        if type(self.generation_max_tokens) is not int or self.generation_max_tokens <= 0:
            raise ValueError("microprobe generation token budget is invalid")
        if self.generation_seed_policy != "turn-and-retry-offset-v1":
            raise ValueError("microprobe requires the offset seed policy")
        if type(self.generation_seed) is not int or self.generation_seed < 0:
            raise ValueError("microprobe generation seed is invalid")
        if self.selection_contract != GENERATOR_MICROPROBE_SELECTION_CONTRACT:
            raise ValueError("microprobe selection contract is invalid")
        if self.canonical_serializer != CANONICAL_JSON_VERSION:
            raise ValueError("microprobe canonical serializer is invalid")
        if self.interpretation_contract != GENERATOR_MICROPROBE_INTERPRETATION_CONTRACT:
            raise ValueError("microprobe interpretation contract is invalid")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_run_sha256": self.source_run_sha256,
            "source_diagnostic_sha256": self.source_diagnostic_sha256,
            "source_environment_contract_sha256": (
                self.source_environment_contract_sha256
            ),
            "probe_environment_contract_sha256": (
                self.probe_environment_contract_sha256
            ),
            "source_collection_contract_sha256": (
                self.source_collection_contract_sha256
            ),
            "source_problem_source_sha256": self.source_problem_source_sha256,
            "case_manifest_sha256": self.case_manifest_sha256,
            "source_decision_count": self.source_decision_count,
            "source_lens_contract": self.source_lens_contract,
            "probe_lens_contract": self.probe_lens_contract,
            "generator": self.generator.to_record(),
            "generator_transport": self.generator_transport.to_record(),
            "generation_schema": self.generation_schema,
            "generation_temperature": self.generation_temperature,
            "generation_max_tokens": self.generation_max_tokens,
            "generation_seed_policy": self.generation_seed_policy,
            "generation_seed": self.generation_seed,
            "selection_contract": self.selection_contract,
            "canonical_serializer": self.canonical_serializer,
            "interpretation_contract": self.interpretation_contract,
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.to_record())

    @classmethod
    def from_record(cls, value: object) -> GeneratorMicroprobeContract:
        expected = frozenset({
            "schema_version",
            "source_run_sha256",
            "source_diagnostic_sha256",
            "source_environment_contract_sha256",
            "probe_environment_contract_sha256",
            "source_collection_contract_sha256",
            "source_problem_source_sha256",
            "case_manifest_sha256",
            "source_decision_count",
            "source_lens_contract",
            "probe_lens_contract",
            "generator",
            "generator_transport",
            "generation_schema",
            "generation_temperature",
            "generation_max_tokens",
            "generation_seed_policy",
            "generation_seed",
            "selection_contract",
            "canonical_serializer",
            "interpretation_contract",
        })
        if type(value) is not dict or set(value) != expected:
            raise GeneratorMicroprobeValidationError(
                "microprobe contract record has invalid keys"
            )
        try:
            return cls(
                source_run_sha256=value["source_run_sha256"],
                source_diagnostic_sha256=value[
                    "source_diagnostic_sha256"
                ],
                source_environment_contract_sha256=value[
                    "source_environment_contract_sha256"
                ],
                probe_environment_contract_sha256=value[
                    "probe_environment_contract_sha256"
                ],
                source_collection_contract_sha256=value[
                    "source_collection_contract_sha256"
                ],
                source_problem_source_sha256=value[
                    "source_problem_source_sha256"
                ],
                case_manifest_sha256=value["case_manifest_sha256"],
                source_decision_count=value["source_decision_count"],
                source_lens_contract=value["source_lens_contract"],
                probe_lens_contract=value["probe_lens_contract"],
                generator=ModelIdentity.from_record(value["generator"]),
                generator_transport=ModelTransportContract.from_record(
                    value["generator_transport"]
                ),
                generation_schema=value["generation_schema"],
                generation_temperature=value["generation_temperature"],
                generation_max_tokens=value["generation_max_tokens"],
                generation_seed_policy=value["generation_seed_policy"],
                generation_seed=value["generation_seed"],
                schema_version=value["schema_version"],
                selection_contract=value["selection_contract"],
                canonical_serializer=value["canonical_serializer"],
                interpretation_contract=value["interpretation_contract"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GeneratorMicroprobeValidationError(
                "microprobe contract record is invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class GeneratorMicroprobeAuthorization:
    """Explicit hosted authorization bound to source files and probe contract."""

    preregistration_id: str
    expected_source_run_sha256: str
    expected_source_diagnostic_sha256: str
    expected_probe_contract_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.preregistration_id, str) or not self.preregistration_id.strip():
            raise ValueError("microprobe preregistration id must be nonempty")
        for field_name in (
            "expected_source_run_sha256",
            "expected_source_diagnostic_sha256",
            "expected_probe_contract_sha256",
        ):
            if not _is_sha256(getattr(self, field_name)):
                raise ValueError(f"microprobe authorization {field_name} is invalid")

    def validate(self, contract: GeneratorMicroprobeContract) -> None:
        if self.expected_source_run_sha256 != contract.source_run_sha256:
            raise ValueError("authorized source run digest differs")
        if self.expected_source_diagnostic_sha256 != contract.source_diagnostic_sha256:
            raise ValueError("authorized source diagnostic digest differs")
        if self.expected_probe_contract_sha256 != contract.sha256:
            raise ValueError("authorized microprobe contract digest differs")


@dataclass(frozen=True, slots=True)
class GeneratorMicroprobeCase:
    """One sealed historical decision projected to the changed lens only."""

    index: int
    episode_key: str
    problem_id: str
    state: State
    action: Action
    source_observation_bytes: bytes
    probe_observation_bytes: bytes
    source_observation_sha256: str
    probe_observation_sha256: str
    source_request_sha256: str
    probe_request_sha256: str
    effective_seed: int
    source_decision_record_id: str
    source_generator_call_id: str

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("microprobe case index is invalid")
        for field_name in ("episode_key", "problem_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"microprobe case {field_name} is invalid")
        if not isinstance(self.state, State):
            raise ValueError("microprobe case state is invalid")
        if not isinstance(self.action, Action):
            raise ValueError("microprobe case action is invalid")
        for field_name in (
            "source_observation_bytes",
            "probe_observation_bytes",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, bytes):
                raise ValueError(f"microprobe case {field_name} is invalid")
            try:
                decoded = json.loads(value)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError(
                    f"microprobe case {field_name} is invalid JSON"
                ) from exc
            if type(decoded) is not dict or canonical_json_bytes(decoded) != value:
                raise ValueError(
                    f"microprobe case {field_name} is not a canonical object"
                )
        for field_name in (
            "source_observation_sha256",
            "probe_observation_sha256",
            "source_request_sha256",
            "probe_request_sha256",
        ):
            if not _is_sha256(getattr(self, field_name)):
                raise ValueError(f"microprobe case {field_name} is invalid")
        if type(self.effective_seed) is not int or self.effective_seed < 0:
            raise ValueError("microprobe case effective seed is invalid")
        if (
            not isinstance(self.source_decision_record_id, str)
            or not self.source_decision_record_id.strip()
        ):
            raise ValueError(
                "microprobe case source decision record id is invalid"
            )
        if (
            not isinstance(self.source_generator_call_id, str)
            or len(self.source_generator_call_id) != 32
            or set(self.source_generator_call_id) - _SHA256_CHARS
        ):
            raise ValueError(
                "microprobe case source generator call id is invalid"
            )

    @property
    def source_observation(self) -> dict[str, object]:
        return json.loads(self.source_observation_bytes)

    @property
    def probe_observation(self) -> dict[str, object]:
        return json.loads(self.probe_observation_bytes)

    def manifest_record(self) -> dict[str, object]:
        """Return the ordered, secret-free identity bound by authorization."""
        return {
            "index": self.index,
            "episode_key": self.episode_key,
            "problem_id": self.problem_id,
            "action": serialize_action(self.action),
            "source_observation_sha256": self.source_observation_sha256,
            "probe_observation_sha256": self.probe_observation_sha256,
            "source_request_sha256": self.source_request_sha256,
            "probe_request_sha256": self.probe_request_sha256,
            "effective_seed": self.effective_seed,
            "source_decision_record_id": self.source_decision_record_id,
            "source_generator_call_id": self.source_generator_call_id,
        }


def _case_manifest_sha256(
    cases: tuple[GeneratorMicroprobeCase, ...] | list[GeneratorMicroprobeCase],
) -> str:
    return canonical_json_sha256([case.manifest_record() for case in cases])


@dataclass(frozen=True, slots=True)
class PreparedGeneratorMicroprobe:
    """Strict-replay-sealed inputs ready for zero-controller execution."""

    contract: GeneratorMicroprobeContract
    source_environment: BoundEnvironment
    probe_environment: BoundEnvironment
    source_collection: CollectionContract
    replayed_run: ReplayedRun
    cases: tuple[GeneratorMicroprobeCase, ...]
    _trust_seal: object = field(init=False, repr=False, compare=False, default=None)
    _trusted_binding: object = field(init=False, repr=False, compare=False, default=None)

    def validate_trusted(self) -> None:
        binding = (
            self.contract,
            self.source_environment,
            self.probe_environment,
            self.source_collection,
            self.replayed_run,
            self.cases,
        )
        if self._trust_seal is not _PREPARED_TRUST_SEAL or self._trusted_binding != binding:
            raise GeneratorMicroprobeValidationError(
                "microprobe preparation did not come from strict replay"
            )
        self.replayed_run.validate_trusted()
        if self.contract.source_environment_contract_sha256 != (
            self.source_environment.sha256
        ):
            raise GeneratorMicroprobeValidationError(
                "prepared source environment differs from the contract"
            )
        if self.contract.probe_environment_contract_sha256 != (
            self.probe_environment.sha256
        ):
            raise GeneratorMicroprobeValidationError(
                "prepared probe environment differs from the contract"
            )
        if self.contract.source_collection_contract_sha256 != (
            self.source_collection.sha256
        ):
            raise GeneratorMicroprobeValidationError(
                "prepared collection differs from the contract"
            )
        if len(self.cases) != self.contract.source_decision_count:
            raise GeneratorMicroprobeValidationError(
                "prepared case count differs from the contract"
            )
        if tuple(case.index for case in self.cases) != tuple(
            range(len(self.cases))
        ):
            raise GeneratorMicroprobeValidationError(
                "prepared cases are not in canonical order"
            )
        if _case_manifest_sha256(self.cases) != (
            self.contract.case_manifest_sha256
        ):
            raise GeneratorMicroprobeValidationError(
                "prepared cases differ from the authorized manifest"
            )


@dataclass(frozen=True, slots=True)
class GeneratorMicroprobeRun:
    output_path: Path
    contract_sha256: str
    case_count: int
    wire_usable_count: int
    decoded_count: int
    symbolic_applied_count: int


@dataclass(frozen=True, slots=True)
class _StaticDraftGenerator:
    draft: PolicyDraft

    def generate(self, state: State, action: Action) -> PolicyDraft:
        return self.draft


def _error_record(error: Exception) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _serialize_policy_draft(draft: PolicyDraft) -> dict[str, object]:
    if isinstance(draft, DecompositionDraft):
        return {
            "draft_type": "decomposition",
            "connective": draft.connective.value,
            "child_statements": list(draft.child_statements),
            "child_obligations": [
                serialize_obligation(item)
                for item in draft.child_obligations
            ],
        }
    if isinstance(draft, ResultDraft):
        return {
            "draft_type": "result",
            "objective": draft.objective.value,
            "value": draft.value,
        }
    if isinstance(draft, SolutionDraft):
        return {
            "draft_type": "solution",
            "variable": draft.variable,
            "value": draft.value,
        }
    if isinstance(draft, RefutationDraft):
        return {
            "draft_type": "refutation",
            "mode": draft.mode.value,
            "variable": draft.variable,
            "candidate": draft.candidate,
        }
    raise TypeError("microprobe policy draft has an invalid type")


def _serialize_symbolic_check(check: object) -> dict[str, object] | None:
    if check is None:
        return None
    return {
        "context": serialize_verifier_context(check.context),
        "accepted": check.accepted,
        "observed": check.observed,
        "reason_code": check.reason_code,
    }


def _apply_draft_directly(
    prepared: PreparedGeneratorMicroprobe,
    case: GeneratorMicroprobeCase,
    draft: PolicyDraft,
):
    graph = case.state.graph
    contract = prepared.probe_environment.generation_contract
    if case.action.verb is Verb.DECOMPOSE:
        return decompose(graph, case.action.target, draft, contract)
    if case.action.verb is Verb.PROPOSE_RESULT:
        return propose_result(graph, case.action.target, draft, contract)
    if case.action.verb is Verb.PROPOSE_SOLUTION:
        return propose_solution(graph, case.action.target, draft, contract)
    if case.action.verb is Verb.PROPOSE_REFUTATION:
        return propose_refutation(graph, case.action.target, draft, contract)
    raise GeneratorMicroprobeValidationError(
        "microprobe case action is not a generator verb"
    )


def _counterfactual_symbolic_record(
    prepared: PreparedGeneratorMicroprobe,
    case: GeneratorMicroprobeCase,
    draft: PolicyDraft,
) -> dict[str, object]:
    direct = _apply_draft_directly(prepared, case, draft)
    transition = prepared.probe_environment.step(
        case.state,
        case.action,
        _StaticDraftGenerator(draft),
    )
    direct_rejected = isinstance(direct, Err)
    macro_rejected = transition.outcome.code is OutcomeCode.REJECTED
    if direct_rejected != macro_rejected:
        raise GeneratorMicroprobeValidationError(
            "direct algebra and macro replay classifications differ"
        )
    if macro_rejected and (
        transition.outcome.reason is None
        or transition.outcome.reason.value != "algebra_rejected"
    ):
        raise GeneratorMicroprobeValidationError(
            "decoded draft received a non-algebra macro rejection"
        )

    created_claim: dict[str, object] | None = None
    if not direct_rejected and case.action.verb is not Verb.DECOMPOSE:
        claim_id = len(case.state.graph.nodes)
        claim = get(transition.next_state.graph, claim_id)
        if not isinstance(claim, Claim):
            raise GeneratorMicroprobeValidationError(
                "applied proposal did not retain its created claim"
            )
        created_claim = {
            "claim_id": claim.id,
            "kind": claim.kind.value,
            "status": claim.status.value,
            "discharge_status": claim.discharge_status.value,
            "verification": _serialize_symbolic_check(claim.verification),
            "discharge": _serialize_symbolic_check(claim.discharge),
        }

    return {
        "interpretation_contract": (
            GENERATOR_MICROPROBE_INTERPRETATION_CONTRACT
        ),
        "application_status": (
            "algebra_rejected" if direct_rejected else "applied"
        ),
        "algebra_detail": direct.message if isinstance(direct, Err) else None,
        "macro_outcome": serialize_outcome(transition.outcome),
        "closure_events": [
            {
                "sequence": event.sequence,
                "claim_id": event.claim_id,
                "outcome": serialize_outcome(event.outcome),
            }
            for event in transition.closure_events
        ],
        "created_claim": created_claim,
        "counterfactual_reward": transition.reward,
        "counterfactual_terminal": transition.terminal,
    }


def _validate_attempt_chain(
    attempts: list[dict[str, object]],
) -> None:
    previous: Mapping[str, object] | None = None
    for attempt in attempts:
        validate_attempt_record(attempt)
        if previous is None:
            if (
                attempt["attempt_index"] != 0
                or attempt["parse_retry_index"] != 0
                or attempt["backoff_retry_index"] != 0
            ):
                raise GeneratorMicroprobeValidationError(
                    "microprobe diagnostic call must begin at attempt zero"
                )
        else:
            if (
                attempt["role"] != previous["role"]
                or attempt["call_id"] != previous["call_id"]
            ):
                raise GeneratorMicroprobeValidationError(
                    "microprobe attempt sequence changed call identity"
                )
            if attempt["attempt_index"] != previous["attempt_index"] + 1:
                raise GeneratorMicroprobeValidationError(
                    "microprobe attempt indices are not contiguous"
                )
            disposition = previous["disposition"]
            if disposition == "parse_retry":
                expected_parse = previous["parse_retry_index"] + 1
                expected_backoff = previous["backoff_retry_index"]
            elif disposition == "backoff_retry":
                expected_parse = previous["parse_retry_index"]
                expected_backoff = previous["backoff_retry_index"] + 1
            else:
                raise GeneratorMicroprobeValidationError(
                    "terminal microprobe attempt has a successor"
                )
            if (
                attempt["parse_retry_index"] != expected_parse
                or attempt["backoff_retry_index"] != expected_backoff
            ):
                raise GeneratorMicroprobeValidationError(
                    "microprobe retry indices do not follow disposition"
                )
            previous_seed = previous["effective_seed"]
            current_seed = attempt["effective_seed"]
            if disposition == "parse_retry" and previous_seed is not None:
                if current_seed != previous_seed + 1:
                    raise GeneratorMicroprobeValidationError(
                        "microprobe parse retry did not increment its seed"
                    )
            elif current_seed != previous_seed:
                raise GeneratorMicroprobeValidationError(
                    "microprobe backoff retry changed its seed"
                )
        previous = attempt


class _MicroprobeArtifactWriter:
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
            raise GeneratorMicroprobeValidationError(
                "microprobe artifact writer is closed"
            )
        self._handle.write(canonical_json_bytes(record) + b"\n")
        self._handle.flush()

    def begin_call(self, case_index: int) -> None:
        if self._active_index is not None:
            raise GeneratorMicroprobeValidationError(
                "microprobe diagnostic call is already active"
            )
        self._active_index = case_index
        self._active_call_id = None
        self._attempts = []

    def observe_attempt(self, event: OpenAICompatibleAttempt) -> None:
        if self._active_index is None:
            raise GeneratorMicroprobeValidationError(
                "microprobe observed an attempt outside a case"
            )
        record = build_attempt_record(event, role="generator")
        if self._active_call_id is None:
            self._active_call_id = event.call_id
        elif event.call_id != self._active_call_id:
            raise GeneratorMicroprobeValidationError(
                "microprobe case emitted more than one generator call"
            )
        _validate_attempt_chain([*self._attempts, record])
        self._attempts.append(record)
        self.attempt_count += 1
        self._write(record)

    def finish_call(
        self,
    ) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
        if self._active_index is None or self._active_call_id is None:
            raise GeneratorMicroprobeValidationError(
                "microprobe generator call produced no diagnostic attempt"
            )
        _validate_attempt_chain(self._attempts)
        record = build_call_record(
            "generator", self._active_call_id, self._attempts
        )
        attempts = tuple(self._attempts)
        self._write(record)
        self._active_index = None
        self._active_call_id = None
        self._attempts = []
        self.call_count += 1
        return record, attempts

    def write_case(self, record: Mapping[str, object]) -> None:
        if self._active_index is not None:
            raise GeneratorMicroprobeValidationError(
                "microprobe case written before its call completed"
            )
        self._write(record)

    def finish(self, summary: Mapping[str, object]) -> None:
        if self._active_index is not None:
            raise GeneratorMicroprobeValidationError(
                "microprobe artifact ended during a generator call"
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


def _microprobe_header(
    prepared: PreparedGeneratorMicroprobe,
    authorization: GeneratorMicroprobeAuthorization,
) -> dict[str, object]:
    contract = prepared.contract
    return {
        "schema_version": GENERATOR_MICROPROBE_HEADER_VERSION,
        "canonical_serializer": CANONICAL_JSON_VERSION,
        "preregistration_id": authorization.preregistration_id,
        "probe_contract": contract.to_record(),
        "probe_contract_sha256": contract.sha256,
        "source_environment_contract_sha256": (
            contract.source_environment_contract_sha256
        ),
        "probe_environment_contract_sha256": (
            contract.probe_environment_contract_sha256
        ),
        "case_manifest_sha256": contract.case_manifest_sha256,
    }


def _probe_request(
    prepared: PreparedGeneratorMicroprobe,
    case: GeneratorMicroprobeCase,
):
    contract = prepared.contract
    request = build_policy_lens_request(
        case.probe_observation,
        case.action,
        lens_contract=contract.probe_lens_contract,
    )
    body = build_openai_compatible_json_body(
        model=contract.generator.model,
        system=request.system,
        user=request.user,
        schema=request.schema,
        temperature=contract.generation_temperature,
        max_tokens=contract.generation_max_tokens,
        seed=case.effective_seed,
        guided_json=contract.generator_transport.guided_json,
    )
    if canonical_json_sha256(body) != case.probe_request_sha256:
        raise GeneratorMicroprobeValidationError(
            "probe request differs from its prepared digest"
        )
    return request, body


def prepare_generator_microprobe(
    *,
    run_path: str | Path,
    diagnostic_path: str | Path,
    problem_source: CaseworkProblemSource,
    probe_lens_contract: str,
) -> PreparedGeneratorMicroprobe:
    """Strict-replay a reference run and bind every decision to a new lens."""
    if not isinstance(problem_source, CaseworkProblemSource):
        raise TypeError("microprobe problem source has an invalid type")
    if probe_lens_contract not in SUPPORTED_LENS_CONTRACT_VERSIONS:
        raise ValueError("microprobe probe lens contract is unsupported")
    source_run_path = Path(run_path)
    source_diagnostic_path = Path(diagnostic_path)
    run_digest = file_sha256(source_run_path)
    diagnostic_digest = file_sha256(source_diagnostic_path)

    with source_run_path.open("rb") as handle:
        first_line = handle.readline()
    header = parse_canonical_json_line(first_line)
    try:
        source_contract = EnvironmentContract.from_record(
            header["environment_contract"]
        )
        collection = CollectionContract.from_record(
            header["collection_contract"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GeneratorMicroprobeValidationError(
            "microprobe source run header is invalid"
        ) from exc
    adapter = CaseworkMathVerifier()
    source_environment = BoundEnvironment(
        source_contract, adapter, adapter, adapter
    )
    if probe_lens_contract == source_contract.lens_contract:
        raise ValueError("microprobe probe lens equals the source lens")
    probe_contract = replace(
        source_contract, lens_contract=probe_lens_contract
    )
    source_material = source_contract.to_record()
    probe_material = probe_contract.to_record()
    source_material.pop("lens_contract")
    probe_material.pop("lens_contract")
    if source_material != probe_material:
        raise AssertionError("microprobe environment changed beyond the lens")
    probe_environment = BoundEnvironment(
        probe_contract, adapter, adapter, adapter
    )

    diagnostic_index = read_diagnostic_sidecar(
        source_diagnostic_path,
        environment_contract_sha256=source_environment.sha256,
        collection_contract_sha256=collection.sha256,
        generator_transport_sha256=collection.generator_transport_sha256,
        controller_transport_sha256=collection.controller_transport_sha256,
    )
    replayed = replay_canonical_run(
        source_run_path,
        environment=source_environment,
        initial_state_resolver=problem_source,
        collection_contract=collection,
        diagnostic_index=diagnostic_index,
    )
    replayed.validate_trusted()

    cases: list[GeneratorMicroprobeCase] = []
    for episode in replayed.episodes:
        for decision in episode.decisions:
            transition = decision.transition
            action = transition.action
            source_observation = source_environment.observe(transition.state)
            probe_observation = probe_environment.observe(transition.state)
            source_observation_digest = observation_sha256(
                source_observation,
                source_environment.generation_contract,
                source_environment.verifier,
                environment_contract_sha256=source_environment.sha256,
            )
            probe_observation_digest = observation_sha256(
                probe_observation,
                probe_environment.generation_contract,
                probe_environment.verifier,
                environment_contract_sha256=probe_environment.sha256,
            )
            if decision.diagnostic_ref is None:
                raise GeneratorMicroprobeValidationError(
                    "microprobe source decision lacks hosted diagnostics"
                )
            resolved = diagnostic_index.resolve_decision_receipt(
                decision.diagnostic_ref
            )
            if resolved.generator_diagnostic is None:
                raise GeneratorMicroprobeValidationError(
                    "microprobe source decision lacks generator diagnostics"
                )
            attempts = resolved.generator_diagnostic.attempt_records
            if not attempts:
                raise GeneratorMicroprobeValidationError(
                    "microprobe source generator call has no attempt"
                )
            effective_seed = attempts[0]["effective_seed"]
            if type(effective_seed) is not int:
                raise GeneratorMicroprobeValidationError(
                    "microprobe source first attempt lacks an integer seed"
                )
            expected_seed = collection.generation_seed + transition.state.turn
            if effective_seed != expected_seed:
                raise GeneratorMicroprobeValidationError(
                    "microprobe source first seed differs from base plus turn"
                )

            source_request = build_policy_lens_request(
                source_observation,
                action,
                lens_contract=source_contract.lens_contract,
            )
            source_body = build_openai_compatible_json_body(
                model=collection.generator_transport.model,
                system=source_request.system,
                user=source_request.user,
                schema=source_request.schema,
                temperature=source_contract.generation_temperature,
                max_tokens=source_contract.generation_max_tokens,
                seed=effective_seed,
                guided_json=collection.generator_transport.guided_json,
            )
            if attempts[0]["request_body"] != source_body:
                raise GeneratorMicroprobeValidationError(
                    "microprobe cannot reproduce the source first request"
                )
            probe_request = build_policy_lens_request(
                probe_observation,
                action,
                lens_contract=probe_lens_contract,
            )
            probe_body = build_openai_compatible_json_body(
                model=collection.generator_transport.model,
                system=probe_request.system,
                user=probe_request.user,
                schema=probe_request.schema,
                temperature=probe_contract.generation_temperature,
                max_tokens=probe_contract.generation_max_tokens,
                seed=effective_seed,
                guided_json=collection.generator_transport.guided_json,
            )
            source_bytes = canonical_json_bytes(source_observation)
            probe_bytes = canonical_json_bytes(probe_observation)
            cases.append(GeneratorMicroprobeCase(
                index=len(cases),
                episode_key=episode.episode_key,
                problem_id=episode.problem_id,
                state=transition.state,
                action=action,
                source_observation_bytes=source_bytes,
                probe_observation_bytes=probe_bytes,
                source_observation_sha256=source_observation_digest,
                probe_observation_sha256=probe_observation_digest,
                source_request_sha256=canonical_json_sha256(source_body),
                probe_request_sha256=canonical_json_sha256(probe_body),
                effective_seed=effective_seed,
                source_decision_record_id=decision.diagnostic_ref.record_id,
                source_generator_call_id=(
                    resolved.generator_diagnostic.call_record["call_id"]
                ),
            ))

    frozen_cases = tuple(cases)
    contract = GeneratorMicroprobeContract(
        source_run_sha256=run_digest,
        source_diagnostic_sha256=diagnostic_digest,
        source_environment_contract_sha256=source_environment.sha256,
        probe_environment_contract_sha256=probe_environment.sha256,
        source_collection_contract_sha256=collection.sha256,
        source_problem_source_sha256=collection.problem_source_sha256,
        case_manifest_sha256=_case_manifest_sha256(frozen_cases),
        source_decision_count=len(frozen_cases),
        source_lens_contract=source_contract.lens_contract,
        probe_lens_contract=probe_lens_contract,
        generator=collection.generator,
        generator_transport=collection.generator_transport,
        generation_schema=source_contract.generation_schema,
        generation_temperature=source_contract.generation_temperature,
        generation_max_tokens=source_contract.generation_max_tokens,
        generation_seed_policy=source_contract.generation_seed_policy,
        generation_seed=collection.generation_seed,
    )
    prepared = PreparedGeneratorMicroprobe(
        contract=contract,
        source_environment=source_environment,
        probe_environment=probe_environment,
        source_collection=collection,
        replayed_run=replayed,
        cases=frozen_cases,
    )
    binding = (
        prepared.contract,
        prepared.source_environment,
        prepared.probe_environment,
        prepared.source_collection,
        prepared.replayed_run,
        prepared.cases,
    )
    object.__setattr__(prepared, "_trust_seal", _PREPARED_TRUST_SEAL)
    object.__setattr__(prepared, "_trusted_binding", binding)
    prepared.validate_trusted()
    return prepared


def run_generator_microprobe(
    *,
    output_path: str | Path,
    prepared: PreparedGeneratorMicroprobe,
    generator_client: OpenAICompatibleLLMClient,
    authorization: GeneratorMicroprobeAuthorization,
) -> GeneratorMicroprobeRun:
    """Execute every prepared row once without a controller or rollout."""
    if not isinstance(prepared, PreparedGeneratorMicroprobe):
        raise TypeError("prepared microprobe has an invalid type")
    if not isinstance(authorization, GeneratorMicroprobeAuthorization):
        raise TypeError("microprobe authorization has an invalid type")
    prepared.validate_trusted()
    authorization.validate(prepared.contract)
    if type(generator_client) is not OpenAICompatibleLLMClient:
        raise TypeError("microprobe generator client has an invalid type")
    if generator_client.attempt_observer is not None:
        raise ValueError("microprobe generator client already has an observer")
    if llm_transport_contract(generator_client) != (
        prepared.contract.generator_transport
    ):
        raise ValueError(
            "microprobe generator client differs from the probe contract"
        )

    final_path = Path(output_path)
    partial_path = final_path.with_name(f"{final_path.name}.partial")
    if final_path.exists() or partial_path.exists():
        raise FileExistsError(
            "microprobe output or partial output already exists"
        )
    if not final_path.parent.exists() or not final_path.parent.is_dir():
        raise FileNotFoundError(
            "microprobe output directory does not exist"
        )

    contract = prepared.contract
    wire_usable_count = 0
    decoded_count = 0
    symbolic_applied_count = 0
    attempt_outcome_counts = {
        "accepted": 0,
        "parse_error": 0,
        "schema_error": 0,
        "provider_response_error": 0,
        "http_status_error": 0,
        "transport_error": 0,
    }
    reported_usage_attempt_count = 0
    reported_prompt_tokens = 0
    reported_completion_tokens = 0
    reported_total_tokens = 0
    decode_counts = {"decoded": 0, "payload_error": 0, "not_run": 0}
    application_counts = {
        "applied": 0,
        "algebra_rejected": 0,
        "not_run": 0,
    }
    by_verb: dict[str, dict[str, int]] = {}
    writer: _MicroprobeArtifactWriter | None = None
    callback = None
    try:
        writer = _MicroprobeArtifactWriter(
            partial_path,
            _microprobe_header(prepared, authorization),
        )
        callback = writer.observe_attempt
        generator_client.attempt_observer = callback

        for case in prepared.cases:
            request, expected_body = _probe_request(prepared, case)
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

            call_record, attempt_records = writer.finish_call()
            for attempt_record in attempt_records:
                attempt_outcome_counts[attempt_record["outcome"]] += 1
                usage = attempt_record["usage"]
                if usage["reported"]:
                    reported_usage_attempt_count += 1
                    reported_prompt_tokens += usage["prompt_tokens"] or 0
                    reported_completion_tokens += (
                        usage["completion_tokens"] or 0
                    )
                    reported_total_tokens += usage["total_tokens"] or 0
            first_attempt = attempt_records[0]
            if canonical_json_sha256(first_attempt["request_body"]) != (
                case.probe_request_sha256
            ):
                raise GeneratorMicroprobeValidationError(
                    "hosted first request differs from the prepared digest"
                )
            if first_attempt["request_body"] != expected_body:
                raise GeneratorMicroprobeValidationError(
                    "hosted first request differs from local reconstruction"
                )
            if first_attempt["effective_seed"] != case.effective_seed:
                raise GeneratorMicroprobeValidationError(
                    "hosted first request differs from the original seed"
                )

            wire_usable = call_record["final_outcome"] == "transport_accepted"
            if wire_usable is (generation_error is not None):
                raise GeneratorMicroprobeValidationError(
                    "generator return and diagnostic outcome disagree"
                )
            wire = {
                "usable": wire_usable,
                "terminal_outcome": call_record["final_outcome"],
                "final_attempt_outcome": call_record[
                    "final_attempt_outcome"
                ],
                "error": (
                    None
                    if generation_error is None
                    else _error_record(generation_error)
                ),
            }

            draft: PolicyDraft | None = None
            if not wire_usable:
                decode = {
                    "status": "not_run",
                    "error": None,
                    "draft": None,
                }
                symbolic = None
                application_status = "not_run"
            else:
                if payload is None:
                    raise GeneratorMicroprobeValidationError(
                        "accepted generator call returned no payload"
                    )
                try:
                    draft = decode_policy_payload(
                        case.probe_observation,
                        case.action,
                        payload,
                    )
                except PolicyPayloadError as exc:
                    decode = {
                        "status": "payload_error",
                        "error": _error_record(exc),
                        "draft": None,
                    }
                    symbolic = None
                    application_status = "not_run"
                else:
                    decode = {
                        "status": "decoded",
                        "error": None,
                        "draft": _serialize_policy_draft(draft),
                    }
                    symbolic = _counterfactual_symbolic_record(
                        prepared, case, draft
                    )
                    application_status = symbolic["application_status"]

            if wire_usable:
                wire_usable_count += 1
            decode_counts[decode["status"]] += 1
            if decode["status"] == "decoded":
                decoded_count += 1
            application_counts[application_status] += 1
            if application_status == "applied":
                symbolic_applied_count += 1

            verb = case.action.verb.value
            verb_counts = by_verb.setdefault(verb, {
                "case_count": 0,
                "wire_usable_count": 0,
                "decoded_count": 0,
                "symbolic_applied_count": 0,
            })
            verb_counts["case_count"] += 1
            verb_counts["wire_usable_count"] += int(wire_usable)
            verb_counts["decoded_count"] += int(
                decode["status"] == "decoded"
            )
            verb_counts["symbolic_applied_count"] += int(
                application_status == "applied"
            )

            writer.write_case({
                "schema_version": GENERATOR_MICROPROBE_CASE_VERSION,
                **case.manifest_record(),
                "probe_call_record_id": call_record["record_id"],
                "wire": wire,
                "decode": decode,
                "counterfactual_symbolic": symbolic,
            })

        summary = {
            "schema_version": GENERATOR_MICROPROBE_SUMMARY_VERSION,
            "probe_contract_sha256": contract.sha256,
            "case_manifest_sha256": contract.case_manifest_sha256,
            "case_count": len(prepared.cases),
            "raw_attempt_count": writer.attempt_count,
            "generator_call_count": writer.call_count,
            "wire_usable_count": wire_usable_count,
            "wire_unusable_count": len(prepared.cases) - wire_usable_count,
            "decoded_count": decoded_count,
            "typed_usable_count": decoded_count,
            "typed_unusable_count": len(prepared.cases) - decoded_count,
            "decode_counts": decode_counts,
            "symbolic_applied_count": symbolic_applied_count,
            "symbolic_application_counts": application_counts,
            "by_verb": by_verb,
            "raw_attempt_outcome_counts": attempt_outcome_counts,
            "reported_usage": {
                "attempt_count": reported_usage_attempt_count,
                "prompt_tokens": reported_prompt_tokens,
                "completion_tokens": reported_completion_tokens,
                "total_tokens": reported_total_tokens,
            },
            "interpretation_contract": (
                GENERATOR_MICROPROBE_INTERPRETATION_CONTRACT
            ),
        }
        writer.finish(summary)
        if generator_client.attempt_observer is callback:
            generator_client.attempt_observer = None
        validated = read_generator_microprobe_artifact(partial_path)
        if validated != summary:
            raise GeneratorMicroprobeValidationError(
                "microprobe artifact summary changed during validation"
            )
        os.link(partial_path, final_path)
        partial_path.unlink()
        return GeneratorMicroprobeRun(
            output_path=final_path,
            contract_sha256=contract.sha256,
            case_count=len(prepared.cases),
            wire_usable_count=wire_usable_count,
            decoded_count=decoded_count,
            symbolic_applied_count=symbolic_applied_count,
        )
    finally:
        if generator_client.attempt_observer is callback:
            generator_client.attempt_observer = None
        if writer is not None:
            writer.abort()


_MANIFEST_KEYS = frozenset({
    "index",
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
})
_CASE_KEYS = _MANIFEST_KEYS | frozenset({
    "schema_version",
    "probe_call_record_id",
    "wire",
    "decode",
    "counterfactual_symbolic",
})
_HEADER_KEYS = frozenset({
    "schema_version",
    "canonical_serializer",
    "preregistration_id",
    "probe_contract",
    "probe_contract_sha256",
    "source_environment_contract_sha256",
    "probe_environment_contract_sha256",
    "case_manifest_sha256",
})
_SUMMARY_KEYS = frozenset({
    "schema_version",
    "probe_contract_sha256",
    "case_manifest_sha256",
    "case_count",
    "raw_attempt_count",
    "generator_call_count",
    "wire_usable_count",
    "wire_unusable_count",
    "decoded_count",
    "typed_usable_count",
    "typed_unusable_count",
    "decode_counts",
    "symbolic_applied_count",
    "symbolic_application_counts",
    "by_verb",
    "raw_attempt_outcome_counts",
    "reported_usage",
    "interpretation_contract",
})


def _exact_record(
    value: object, keys: frozenset[str], path: str
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise GeneratorMicroprobeValidationError(
            f"{path} keys are invalid"
        )
    return value


def _nonnegative_integer(value: object, path: str) -> int:
    if type(value) is not int or value < 0:
        raise GeneratorMicroprobeValidationError(
            f"{path} must be a nonnegative integer"
        )
    return value


def _nonempty_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GeneratorMicroprobeValidationError(
            f"{path} must be nonempty text"
        )
    return value


def _validate_error_record(value: object, path: str) -> None:
    record = _exact_record(
        value, frozenset({"type", "message"}), path
    )
    _nonempty_text(record["type"], f"{path}.type")
    _nonempty_text(record["message"], f"{path}.message")


def _validate_serialized_check(value: object, path: str) -> None:
    if value is None:
        return
    record = _exact_record(
        value,
        frozenset({"context", "accepted", "observed", "reason_code"}),
        path,
    )
    if type(record["context"]) is not dict:
        raise GeneratorMicroprobeValidationError(
            f"{path}.context must be an object"
        )
    if type(record["accepted"]) is not bool:
        raise GeneratorMicroprobeValidationError(
            f"{path}.accepted must be boolean"
        )
    if not isinstance(record["observed"], str):
        raise GeneratorMicroprobeValidationError(
            f"{path}.observed must be text"
        )
    reason = record["reason_code"]
    if reason is not None and (
        not isinstance(reason, str) or not reason.strip()
    ):
        raise GeneratorMicroprobeValidationError(
            f"{path}.reason_code must be null or nonempty text"
        )


def _validate_draft_record(
    value: object, action: Action, path: str
) -> None:
    if type(value) is not dict:
        raise GeneratorMicroprobeValidationError(
            f"{path} must be an object"
        )
    expected_type = {
        Verb.DECOMPOSE: "decomposition",
        Verb.PROPOSE_RESULT: "result",
        Verb.PROPOSE_SOLUTION: "solution",
        Verb.PROPOSE_REFUTATION: "refutation",
    }[action.verb]
    if value.get("draft_type") != expected_type:
        raise GeneratorMicroprobeValidationError(
            f"{path} differs from its action verb"
        )
    expected_keys = {
        "decomposition": frozenset({
            "draft_type",
            "connective",
            "child_statements",
            "child_obligations",
        }),
        "result": frozenset({"draft_type", "objective", "value"}),
        "solution": frozenset({"draft_type", "variable", "value"}),
        "refutation": frozenset({
            "draft_type", "mode", "variable", "candidate"
        }),
    }[expected_type]
    _exact_record(value, expected_keys, path)
    canonical_json_bytes(value)


def _validate_counterfactual_symbolic(
    value: object, action: Action, path: str
) -> str:
    record = _exact_record(
        value,
        frozenset({
            "interpretation_contract",
            "application_status",
            "algebra_detail",
            "macro_outcome",
            "closure_events",
            "created_claim",
            "counterfactual_reward",
            "counterfactual_terminal",
        }),
        path,
    )
    if record["interpretation_contract"] != (
        GENERATOR_MICROPROBE_INTERPRETATION_CONTRACT
    ):
        raise GeneratorMicroprobeValidationError(
            f"{path} interpretation contract is invalid"
        )
    status = record["application_status"]
    if status not in {"applied", "algebra_rejected"}:
        raise GeneratorMicroprobeValidationError(
            f"{path} application status is invalid"
        )
    detail = record["algebra_detail"]
    if status == "algebra_rejected" and (
        not isinstance(detail, str) or not detail.strip()
    ):
        raise GeneratorMicroprobeValidationError(
            f"{path} algebra detail differs from its status"
        )
    if status == "applied" and detail is not None:
        raise GeneratorMicroprobeValidationError(
            f"{path} applied result unexpectedly has algebra detail"
        )
    outcome = deserialize_outcome(
        record["macro_outcome"], f"{path}.macro_outcome"
    )
    if outcome.action != action:
        raise GeneratorMicroprobeValidationError(
            f"{path} macro outcome action differs"
        )
    if (outcome.code is OutcomeCode.APPLIED) is not (status == "applied"):
        raise GeneratorMicroprobeValidationError(
            f"{path} macro outcome differs from application status"
        )
    if status == "algebra_rejected" and (
        outcome.reason is None
        or outcome.reason.value != "algebra_rejected"
    ):
        raise GeneratorMicroprobeValidationError(
            f"{path} macro rejection reason is invalid"
        )

    events = record["closure_events"]
    if type(events) is not list:
        raise GeneratorMicroprobeValidationError(
            f"{path}.closure_events must be an array"
        )
    for index, event_value in enumerate(events):
        event = _exact_record(
            event_value,
            frozenset({"sequence", "claim_id", "outcome"}),
            f"{path}.closure_events[{index}]",
        )
        if event["sequence"] != index:
            raise GeneratorMicroprobeValidationError(
                f"{path} closure sequence is invalid"
            )
        _nonnegative_integer(
            event["claim_id"], f"{path}.closure_events[{index}].claim_id"
        )
        deserialize_outcome(
            event["outcome"], f"{path}.closure_events[{index}].outcome"
        )
    if status == "algebra_rejected" and events:
        raise GeneratorMicroprobeValidationError(
            f"{path} algebra rejection cannot have closure events"
        )

    created = record["created_claim"]
    expects_claim = status == "applied" and action.verb is not Verb.DECOMPOSE
    if expects_claim:
        claim = _exact_record(
            created,
            frozenset({
                "claim_id",
                "kind",
                "status",
                "discharge_status",
                "verification",
                "discharge",
            }),
            f"{path}.created_claim",
        )
        _nonnegative_integer(
            claim["claim_id"], f"{path}.created_claim.claim_id"
        )
        for field_name in ("kind", "status", "discharge_status"):
            _nonempty_text(
                claim[field_name], f"{path}.created_claim.{field_name}"
            )
        _validate_serialized_check(
            claim["verification"], f"{path}.created_claim.verification"
        )
        _validate_serialized_check(
            claim["discharge"], f"{path}.created_claim.discharge"
        )
    elif created is not None:
        raise GeneratorMicroprobeValidationError(
            f"{path} unexpectedly retains a created claim"
        )
    if record["counterfactual_reward"] not in {0, 1}:
        raise GeneratorMicroprobeValidationError(
            f"{path}.counterfactual_reward is invalid"
        )
    if type(record["counterfactual_terminal"]) is not bool:
        raise GeneratorMicroprobeValidationError(
            f"{path}.counterfactual_terminal must be boolean"
        )
    return status


def read_generator_microprobe_artifact(
    path: str | Path,
) -> dict[str, object]:
    """Validate a completed microprobe artifact and return its summary."""
    records: list[dict[str, Any]] = []
    with Path(path).open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = parse_canonical_json_line(line)
            except (TypeError, ValueError) as exc:
                raise GeneratorMicroprobeValidationError(
                    f"microprobe artifact line {line_number} is invalid"
                ) from exc
            if type(record) is not dict:
                raise GeneratorMicroprobeValidationError(
                    f"microprobe artifact line {line_number} is not an object"
                )
            records.append(record)
    if len(records) < 2:
        raise GeneratorMicroprobeValidationError(
            "microprobe artifact is incomplete"
        )

    header = _exact_record(records[0], _HEADER_KEYS, "microprobe header")
    if header["schema_version"] != GENERATOR_MICROPROBE_HEADER_VERSION:
        raise GeneratorMicroprobeValidationError(
            "microprobe header schema is invalid"
        )
    if header["canonical_serializer"] != CANONICAL_JSON_VERSION:
        raise GeneratorMicroprobeValidationError(
            "microprobe header serializer is invalid"
        )
    _nonempty_text(
        header["preregistration_id"], "microprobe header preregistration id"
    )
    contract = GeneratorMicroprobeContract.from_record(
        header["probe_contract"]
    )
    if header["probe_contract_sha256"] != contract.sha256:
        raise GeneratorMicroprobeValidationError(
            "microprobe header contract digest is invalid"
        )
    header_bindings = {
        "source_environment_contract_sha256": (
            contract.source_environment_contract_sha256
        ),
        "probe_environment_contract_sha256": (
            contract.probe_environment_contract_sha256
        ),
        "case_manifest_sha256": contract.case_manifest_sha256,
    }
    for field_name, expected in header_bindings.items():
        if header[field_name] != expected:
            raise GeneratorMicroprobeValidationError(
                f"microprobe header {field_name} differs from the contract"
            )

    summary = _exact_record(
        records[-1], _SUMMARY_KEYS, "microprobe summary"
    )
    if summary["schema_version"] != GENERATOR_MICROPROBE_SUMMARY_VERSION:
        raise GeneratorMicroprobeValidationError(
            "microprobe summary schema is invalid"
        )

    pending_attempts: list[dict[str, object]] = []
    pending_call: dict[str, object] | None = None
    pending_call_attempts: tuple[dict[str, object], ...] = ()
    seen_call_ids: set[str] = set()
    manifest: list[dict[str, object]] = []
    raw_attempt_count = 0
    generator_call_count = 0
    wire_usable_count = 0
    decoded_count = 0
    symbolic_applied_count = 0
    decode_counts = {"decoded": 0, "payload_error": 0, "not_run": 0}
    application_counts = {
        "applied": 0,
        "algebra_rejected": 0,
        "not_run": 0,
    }
    attempt_outcome_counts = {
        "accepted": 0,
        "parse_error": 0,
        "schema_error": 0,
        "provider_response_error": 0,
        "http_status_error": 0,
        "transport_error": 0,
    }
    reported_usage_attempt_count = 0
    reported_prompt_tokens = 0
    reported_completion_tokens = 0
    reported_total_tokens = 0
    by_verb: dict[str, dict[str, int]] = {}

    for record_index, record in enumerate(records[1:-1], start=1):
        schema = record.get("schema_version")
        if schema == DIAGNOSTIC_ATTEMPT_VERSION:
            if pending_call is not None:
                raise GeneratorMicroprobeValidationError(
                    "microprobe attempt appears before the preceding case"
                )
            validate_attempt_record(record)
            if pending_attempts and record["call_id"] != (
                pending_attempts[0]["call_id"]
            ):
                raise GeneratorMicroprobeValidationError(
                    "microprobe attempt changed call before a call record"
                )
            pending_attempts.append(record)
            _validate_attempt_chain(pending_attempts)
            raw_attempt_count += 1
            attempt_outcome_counts[record["outcome"]] += 1
            usage = record["usage"]
            if usage["reported"]:
                reported_usage_attempt_count += 1
                reported_prompt_tokens += usage["prompt_tokens"] or 0
                reported_completion_tokens += (
                    usage["completion_tokens"] or 0
                )
                reported_total_tokens += usage["total_tokens"] or 0
            continue

        if schema == DIAGNOSTIC_CALL_VERSION:
            if pending_call is not None or not pending_attempts:
                raise GeneratorMicroprobeValidationError(
                    "microprobe call record is out of sequence"
                )
            _validate_attempt_chain(pending_attempts)
            validate_call_record(record, attempts=pending_attempts)
            call_id = record["call_id"]
            if call_id in seen_call_ids:
                raise GeneratorMicroprobeValidationError(
                    "microprobe artifact repeats a generator call id"
                )
            seen_call_ids.add(call_id)
            pending_call = record
            pending_call_attempts = tuple(pending_attempts)
            pending_attempts = []
            generator_call_count += 1
            continue

        if schema != GENERATOR_MICROPROBE_CASE_VERSION:
            raise GeneratorMicroprobeValidationError(
                f"microprobe artifact record {record_index} is unsupported"
            )
        if pending_call is None or not pending_call_attempts:
            raise GeneratorMicroprobeValidationError(
                "microprobe case has no immediately preceding generator call"
            )
        case_record = _exact_record(
            record, _CASE_KEYS, f"microprobe case {len(manifest)}"
        )
        case_index = len(manifest)
        if case_record["index"] != case_index:
            raise GeneratorMicroprobeValidationError(
                "microprobe cases are not in canonical order"
            )
        _nonempty_text(
            case_record["episode_key"], f"microprobe case {case_index}.episode_key"
        )
        _nonempty_text(
            case_record["problem_id"], f"microprobe case {case_index}.problem_id"
        )
        action = deserialize_action(
            case_record["action"], f"microprobe case {case_index}.action"
        )
        if action.verb not in {
            Verb.DECOMPOSE,
            Verb.PROPOSE_RESULT,
            Verb.PROPOSE_SOLUTION,
            Verb.PROPOSE_REFUTATION,
        }:
            raise GeneratorMicroprobeValidationError(
                "microprobe case action is not a generator verb"
            )
        for field_name in (
            "source_observation_sha256",
            "probe_observation_sha256",
            "source_request_sha256",
            "probe_request_sha256",
        ):
            if not _is_sha256(case_record[field_name]):
                raise GeneratorMicroprobeValidationError(
                    f"microprobe case {case_index}.{field_name} is invalid"
                )
        _nonnegative_integer(
            case_record["effective_seed"],
            f"microprobe case {case_index}.effective_seed",
        )
        _nonempty_text(
            case_record["source_decision_record_id"],
            f"microprobe case {case_index}.source_decision_record_id",
        )
        source_call_id = case_record["source_generator_call_id"]
        if (
            not isinstance(source_call_id, str)
            or len(source_call_id) != 32
            or set(source_call_id) - _SHA256_CHARS
        ):
            raise GeneratorMicroprobeValidationError(
                f"microprobe case {case_index} source call id is invalid"
            )
        if case_record["probe_call_record_id"] != pending_call["record_id"]:
            raise GeneratorMicroprobeValidationError(
                "microprobe case does not resolve to the preceding call"
            )
        first_attempt = pending_call_attempts[0]
        if canonical_json_sha256(first_attempt["request_body"]) != (
            case_record["probe_request_sha256"]
        ):
            raise GeneratorMicroprobeValidationError(
                "microprobe case request digest differs from its first attempt"
            )
        if first_attempt["effective_seed"] != case_record["effective_seed"]:
            raise GeneratorMicroprobeValidationError(
                "microprobe case seed differs from its first attempt"
            )

        wire = _exact_record(
            case_record["wire"],
            frozenset({
                "usable",
                "terminal_outcome",
                "final_attempt_outcome",
                "error",
            }),
            f"microprobe case {case_index}.wire",
        )
        if type(wire["usable"]) is not bool:
            raise GeneratorMicroprobeValidationError(
                "microprobe wire usable flag must be boolean"
            )
        if (
            wire["terminal_outcome"] != pending_call["final_outcome"]
            or wire["final_attempt_outcome"]
            != pending_call["final_attempt_outcome"]
        ):
            raise GeneratorMicroprobeValidationError(
                "microprobe wire classification differs from its call"
            )
        expected_wire_usable = (
            pending_call["final_outcome"] == "transport_accepted"
        )
        if wire["usable"] is not expected_wire_usable:
            raise GeneratorMicroprobeValidationError(
                "microprobe wire usable flag differs from its call"
            )
        if expected_wire_usable:
            if wire["error"] is not None:
                raise GeneratorMicroprobeValidationError(
                    "usable microprobe wire unexpectedly has an error"
                )
            wire_usable_count += 1
        else:
            _validate_error_record(
                wire["error"], f"microprobe case {case_index}.wire.error"
            )

        decode = _exact_record(
            case_record["decode"],
            frozenset({"status", "error", "draft"}),
            f"microprobe case {case_index}.decode",
        )
        decode_status = decode["status"]
        if decode_status not in decode_counts:
            raise GeneratorMicroprobeValidationError(
                "microprobe decode status is invalid"
            )
        decode_counts[decode_status] += 1
        if decode_status == "decoded":
            if not expected_wire_usable or decode["error"] is not None:
                raise GeneratorMicroprobeValidationError(
                    "decoded microprobe case has invalid staging"
                )
            _validate_draft_record(
                decode["draft"], action, f"microprobe case {case_index}.decode.draft"
            )
            if case_record["counterfactual_symbolic"] is None:
                raise GeneratorMicroprobeValidationError(
                    "decoded microprobe case lacks symbolic diagnostics"
                )
            application_status = _validate_counterfactual_symbolic(
                case_record["counterfactual_symbolic"],
                action,
                f"microprobe case {case_index}.counterfactual_symbolic",
            )
            decoded_count += 1
        elif decode_status == "payload_error":
            if not expected_wire_usable or decode["draft"] is not None:
                raise GeneratorMicroprobeValidationError(
                    "microprobe payload error has invalid staging"
                )
            _validate_error_record(
                decode["error"], f"microprobe case {case_index}.decode.error"
            )
            if case_record["counterfactual_symbolic"] is not None:
                raise GeneratorMicroprobeValidationError(
                    "payload error cannot have symbolic diagnostics"
                )
            application_status = "not_run"
        else:
            if (
                expected_wire_usable
                or decode["error"] is not None
                or decode["draft"] is not None
                or case_record["counterfactual_symbolic"] is not None
            ):
                raise GeneratorMicroprobeValidationError(
                    "microprobe not-run decode has invalid staging"
                )
            application_status = "not_run"
        application_counts[application_status] += 1
        if application_status == "applied":
            symbolic_applied_count += 1

        verb = action.verb.value
        verb_counts = by_verb.setdefault(verb, {
            "case_count": 0,
            "wire_usable_count": 0,
            "decoded_count": 0,
            "symbolic_applied_count": 0,
        })
        verb_counts["case_count"] += 1
        verb_counts["wire_usable_count"] += int(expected_wire_usable)
        verb_counts["decoded_count"] += int(decode_status == "decoded")
        verb_counts["symbolic_applied_count"] += int(
            application_status == "applied"
        )
        manifest.append({key: case_record[key] for key in _MANIFEST_KEYS})
        pending_call = None
        pending_call_attempts = ()

    if pending_attempts or pending_call is not None:
        raise GeneratorMicroprobeValidationError(
            "microprobe artifact ends with an unresolved generator call"
        )
    if len(manifest) != contract.source_decision_count:
        raise GeneratorMicroprobeValidationError(
            "microprobe artifact case count differs from the contract"
        )
    manifest_sha256 = canonical_json_sha256(manifest)
    if manifest_sha256 != contract.case_manifest_sha256:
        raise GeneratorMicroprobeValidationError(
            "microprobe artifact cases differ from the contract manifest"
        )

    expected_summary = {
        "schema_version": GENERATOR_MICROPROBE_SUMMARY_VERSION,
        "probe_contract_sha256": contract.sha256,
        "case_manifest_sha256": contract.case_manifest_sha256,
        "case_count": len(manifest),
        "raw_attempt_count": raw_attempt_count,
        "generator_call_count": generator_call_count,
        "wire_usable_count": wire_usable_count,
        "wire_unusable_count": len(manifest) - wire_usable_count,
        "decoded_count": decoded_count,
        "typed_usable_count": decoded_count,
        "typed_unusable_count": len(manifest) - decoded_count,
        "decode_counts": decode_counts,
        "symbolic_applied_count": symbolic_applied_count,
        "symbolic_application_counts": application_counts,
        "by_verb": by_verb,
        "raw_attempt_outcome_counts": attempt_outcome_counts,
        "reported_usage": {
            "attempt_count": reported_usage_attempt_count,
            "prompt_tokens": reported_prompt_tokens,
            "completion_tokens": reported_completion_tokens,
            "total_tokens": reported_total_tokens,
        },
        "interpretation_contract": (
            GENERATOR_MICROPROBE_INTERPRETATION_CONTRACT
        ),
    }
    if summary != expected_summary:
        raise GeneratorMicroprobeValidationError(
            "microprobe summary differs from its validated records"
        )
    return dict(summary)


__all__ = [
    "GENERATOR_MICROPROBE_CASE_VERSION",
    "GENERATOR_MICROPROBE_CONTRACT_VERSION",
    "GENERATOR_MICROPROBE_HEADER_VERSION",
    "GENERATOR_MICROPROBE_INTERPRETATION_CONTRACT",
    "GENERATOR_MICROPROBE_SELECTION_CONTRACT",
    "GENERATOR_MICROPROBE_SUMMARY_VERSION",
    "GeneratorMicroprobeAuthorization",
    "GeneratorMicroprobeCase",
    "GeneratorMicroprobeContract",
    "GeneratorMicroprobeRun",
    "GeneratorMicroprobeValidationError",
    "PreparedGeneratorMicroprobe",
    "file_sha256",
    "prepare_generator_microprobe",
    "read_generator_microprobe_artifact",
    "run_generator_microprobe",
]
