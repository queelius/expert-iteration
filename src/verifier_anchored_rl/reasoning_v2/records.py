"""Canonical run headers and transition records for reasoning v2."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from .actions import (
    DecompositionDraft,
    RefutationDraft,
    ResultDraft,
    SolutionDraft,
)
from .contracts import (
    CollectionContract,
    EnvironmentContract,
    canonical_json_bytes,
    canonical_json_sha256,
    validate_collection_binding,
)
from .model import (
    Action,
    CheckPurpose,
    ClaimKind,
    Connective,
    GoalObjective,
    GoalObligation,
    OutcomeCode,
    RefutationMode,
    RejectionReason,
    TransitionOutcome,
    Verb,
    VerifierContext,
)
from .observation import (
    serialize_action,
    serialize_obligation,
    serialize_outcome,
    validate_observation_with_contract,
)
from .runtime import BoundEnvironment
from .provenance import BehaviorSource
from .transition import MacroTransition, PolicyDraft


RUN_HEADER_VERSION = "controller-run-header-v2"
TRANSITION_RECORD_VERSION = "controller-transition-v2"
DIAGNOSTIC_REF_VERSION = "controller-diagnostic-ref-v2"

_HEADER_KEYS = frozenset({
    "schema_version",
    "environment_contract",
    "collection_contract",
    "environment_contract_sha256",
    "collection_contract_sha256",
})
_TRANSITION_KEYS = frozenset({
    "schema_version",
    "environment_contract_sha256",
    "collection_contract_sha256",
    "episode_key",
    "problem_id",
    "turn",
    "observation",
    "action",
    "learned_effect",
    "outcome",
    "closure_events",
    "reward",
    "next_observation",
    "terminal",
    "behavior",
    "diagnostic_ref",
})
_ACTION_KEYS = frozenset({"verb", "target"})
_OUTCOME_KEYS = frozenset({
    "action", "code", "reason", "verifier_context"
})
_CONTEXT_KEYS = frozenset({
    "contract_id", "purpose", "claim_kind", "owner_goal", "context_digest"
})
_CLOSURE_EVENT_KEYS = frozenset({"sequence", "claim_id", "outcome"})
_BEHAVIOR_KEYS = frozenset({"source", "bc_eligible", "fallback"})
_DIAGNOSTIC_REF_KEYS = frozenset({"schema_version", "record_id"})
_DECOMPOSITION_EFFECT_KEYS = frozenset({
    "kind", "claim_id", "connective", "children"
})
_CHILD_EFFECT_KEYS = frozenset({"statement", "obligation"})
_PROPOSAL_EFFECT_KEYS = frozenset({
    "kind", "claim_id", "payload", "premises"
})


class RecordValidationError(ValueError):
    """Raised when a persisted v2 record violates its exact schema."""


@dataclass(frozen=True, slots=True)
class BehaviorMetadata:
    source: BehaviorSource
    fallback: bool

    def __post_init__(self) -> None:
        if not isinstance(self.source, BehaviorSource):
            raise ValueError("behavior source has an invalid type")
        if type(self.fallback) is not bool:
            raise ValueError("behavior fallback must be boolean")
        if (
            self.source is BehaviorSource.LEARNED_POLICY
            and self.fallback
        ):
            raise ValueError("learned-policy behavior cannot be a fallback")

    @property
    def bc_eligible(self) -> bool:
        return self.source is BehaviorSource.TEACHER and not self.fallback

    def to_record(self) -> dict[str, object]:
        return {
            "source": self.source.value,
            "bc_eligible": self.bc_eligible,
            "fallback": self.fallback,
        }

    @classmethod
    def from_record(cls, value: object) -> BehaviorMetadata:
        record = _exact_object(value, _BEHAVIOR_KEYS, "behavior")
        try:
            source = BehaviorSource(record["source"])
        except (TypeError, ValueError) as exc:
            raise RecordValidationError("behavior.source is invalid") from exc
        result = cls(
            source=source,
            fallback=_boolean(record["fallback"], "behavior.fallback"),
        )
        stored_bc = _boolean(
            record["bc_eligible"], "behavior.bc_eligible"
        )
        if stored_bc is not result.bc_eligible:
            raise RecordValidationError(
                "behavior.bc_eligible differs from source and fallback"
            )
        return result


@dataclass(frozen=True, slots=True)
class DiagnosticRef:
    record_id: str
    schema_version: str = DIAGNOSTIC_REF_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DIAGNOSTIC_REF_VERSION:
            raise ValueError("diagnostic reference schema is invalid")
        _nonempty_text(self.record_id, "diagnostic record_id")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
        }

    @classmethod
    def from_record(cls, value: object) -> DiagnosticRef:
        record = _exact_object(
            value, _DIAGNOSTIC_REF_KEYS, "diagnostic_ref"
        )
        return cls(
            schema_version=record["schema_version"],
            record_id=record["record_id"],
        )


def _exact_object(
    value: object, keys: frozenset[str], path: str
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise RecordValidationError(
            f"{path} keys must be exactly {sorted(keys)!r}"
        )
    return value


def _nonempty_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecordValidationError(f"{path} must be nonempty text")
    return value


def _integer(value: object, path: str) -> int:
    if type(value) is not int or value < 0:
        raise RecordValidationError(f"{path} must be a nonnegative integer")
    return value


def _boolean(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise RecordValidationError(f"{path} must be boolean")
    return value


def deserialize_action(value: object, path: str = "action") -> Action:
    record = _exact_object(value, _ACTION_KEYS, path)
    try:
        return Action(
            Verb(record["verb"]),
            _integer(record["target"], f"{path}.target"),
        )
    except (TypeError, ValueError) as exc:
        raise RecordValidationError(f"{path}.verb is invalid") from exc


def _deserialize_context(value: object, path: str) -> VerifierContext:
    record = _exact_object(value, _CONTEXT_KEYS, path)
    try:
        return VerifierContext(
            contract_id=_nonempty_text(
                record["contract_id"], f"{path}.contract_id"
            ),
            purpose=CheckPurpose(record["purpose"]),
            claim_kind=ClaimKind(record["claim_kind"]),
            owner_goal=_integer(record["owner_goal"], f"{path}.owner_goal"),
            context_digest=_nonempty_text(
                record["context_digest"], f"{path}.context_digest"
            ),
        )
    except (TypeError, ValueError) as exc:
        raise RecordValidationError(f"{path} is invalid") from exc


def deserialize_outcome(
    value: object, path: str = "outcome"
) -> TransitionOutcome:
    record = _exact_object(value, _OUTCOME_KEYS, path)
    action = deserialize_action(record["action"], f"{path}.action")
    try:
        code = OutcomeCode(record["code"])
        reason = (
            None
            if record["reason"] is None
            else RejectionReason(record["reason"])
        )
        context = (
            None
            if record["verifier_context"] is None
            else _deserialize_context(
                record["verifier_context"],
                f"{path}.verifier_context",
            )
        )
        return TransitionOutcome(action, code, reason, context)
    except (TypeError, ValueError) as exc:
        raise RecordValidationError(f"{path} is invalid") from exc


def _draft_kind(action: Action) -> str:
    return {
        Verb.DECOMPOSE: "decomposition",
        Verb.PROPOSE_RESULT: ClaimKind.RESULT.value,
        Verb.PROPOSE_SOLUTION: ClaimKind.SOLUTION.value,
        Verb.PROPOSE_REFUTATION: ClaimKind.REFUTATION.value,
    }[action.verb]


def serialize_learned_effect(
    transition: MacroTransition,
) -> dict[str, object] | None:
    draft = transition.policy_draft
    if draft is None:
        return None
    claim_id = (
        len(transition.state.graph.nodes)
        if transition.outcome.code is OutcomeCode.APPLIED
        and transition.action.verb is not Verb.DECOMPOSE
        else None
    )
    if isinstance(draft, DecompositionDraft):
        return {
            "kind": "decomposition",
            "claim_id": None,
            "connective": draft.connective.value,
            "children": [
                {
                    "statement": statement,
                    "obligation": serialize_obligation(obligation),
                }
                for statement, obligation in zip(
                    draft.child_statements,
                    draft.child_obligations,
                    strict=True,
                )
            ],
        }
    if isinstance(draft, ResultDraft):
        payload = {
            "objective": draft.objective.value,
            "value": draft.value,
        }
    elif isinstance(draft, SolutionDraft):
        payload = {"variable": draft.variable, "value": draft.value}
    elif isinstance(draft, RefutationDraft):
        payload = {
            "mode": draft.mode.value,
            "variable": draft.variable,
            "candidate": draft.candidate,
        }
    else:
        raise TypeError("transition policy draft has an unsupported type")
    return {
        "kind": _draft_kind(transition.action),
        "claim_id": claim_id,
        "payload": payload,
        "premises": [],
    }


def deserialize_learned_effect(
    value: object,
    action: Action,
    outcome: TransitionOutcome,
) -> PolicyDraft | None:
    if value is None:
        if outcome.code is OutcomeCode.APPLIED or (
            outcome.reason is RejectionReason.ALGEBRA_REJECTED
        ):
            raise RecordValidationError(
                "applied and algebra-rejected actions need a learned effect"
            )
        return None
    if action.verb is Verb.DECOMPOSE:
        record = _exact_object(
            value, _DECOMPOSITION_EFFECT_KEYS, "learned_effect"
        )
        if record["kind"] != "decomposition" or record["claim_id"] is not None:
            raise RecordValidationError("decomposition effect identity is invalid")
        try:
            connective = Connective(record["connective"])
        except (TypeError, ValueError) as exc:
            raise RecordValidationError(
                "learned_effect.connective is invalid"
            ) from exc
        children = record["children"]
        if type(children) is not list or not children:
            raise RecordValidationError(
                "learned_effect.children must be a nonempty array"
            )
        statements: list[str] = []
        obligations: list[GoalObligation] = []
        for index, child_value in enumerate(children):
            child = _exact_object(
                child_value,
                _CHILD_EFFECT_KEYS,
                f"learned_effect.children[{index}]",
            )
            statements.append(
                _nonempty_text(
                    child["statement"],
                    f"learned_effect.children[{index}].statement",
                )
            )
            obligation = _exact_object(
                child["obligation"],
                frozenset({"variable", "equation", "domain", "objective"}),
                f"learned_effect.children[{index}].obligation",
            )
            try:
                obligations.append(GoalObligation(
                    variable=obligation["variable"],
                    equation=obligation["equation"],
                    domain=obligation["domain"],
                    objective=GoalObjective(obligation["objective"]),
                ))
            except (TypeError, ValueError) as exc:
                raise RecordValidationError(
                    f"learned_effect.children[{index}].obligation is invalid"
                ) from exc
        return DecompositionDraft(
            connective,
            tuple(statements),
            tuple(obligations),
        )

    record = _exact_object(
        value, _PROPOSAL_EFFECT_KEYS, "learned_effect"
    )
    if record["kind"] != _draft_kind(action):
        raise RecordValidationError("learned effect kind differs from action")
    expected_claim_id = (
        None
        if outcome.code is OutcomeCode.REJECTED
        else _integer(record["claim_id"], "learned_effect.claim_id")
    )
    if outcome.code is OutcomeCode.REJECTED and record["claim_id"] is not None:
        raise RecordValidationError(
            "rejected learned effect cannot allocate a Claim"
        )
    if outcome.code is OutcomeCode.APPLIED and record["claim_id"] != expected_claim_id:
        raise RecordValidationError("learned effect Claim id is invalid")
    if record["premises"] != []:
        raise RecordValidationError("policy proposal premises must be empty")
    payload = record["payload"]
    if type(payload) is not dict:
        raise RecordValidationError("learned effect payload must be an object")
    try:
        if action.verb is Verb.PROPOSE_RESULT:
            if set(payload) != {"objective", "value"}:
                raise RecordValidationError("result effect payload keys are invalid")
            return ResultDraft(
                GoalObjective(payload["objective"]), payload["value"]
            )
        if action.verb is Verb.PROPOSE_SOLUTION:
            if set(payload) != {"variable", "value"}:
                raise RecordValidationError(
                    "solution effect payload keys are invalid"
                )
            return SolutionDraft(payload["variable"], payload["value"])
        if set(payload) != {"mode", "variable", "candidate"}:
            raise RecordValidationError(
                "refutation effect payload keys are invalid"
            )
        return RefutationDraft(
            RefutationMode(payload["mode"]),
            payload["variable"],
            payload["candidate"],
        )
    except (TypeError, ValueError) as exc:
        raise RecordValidationError("learned effect payload is invalid") from exc


def build_run_header(
    environment: BoundEnvironment,
    collection_contract: CollectionContract,
) -> dict[str, object]:
    if not isinstance(environment, BoundEnvironment):
        raise TypeError("run header environment has an invalid type")
    if not isinstance(collection_contract, CollectionContract):
        raise TypeError("run header collection contract has an invalid type")
    validate_collection_binding(
        environment.contract,
        collection_contract,
    )
    header = {
        "schema_version": RUN_HEADER_VERSION,
        "environment_contract": environment.contract.to_record(),
        "collection_contract": collection_contract.to_record(),
        "environment_contract_sha256": environment.sha256,
        "collection_contract_sha256": collection_contract.sha256,
    }
    validate_run_header(
        header,
        environment=environment,
        collection_contract=collection_contract,
    )
    return header


def validate_run_header(
    value: object,
    *,
    environment: BoundEnvironment | None = None,
    collection_contract: CollectionContract | None = None,
) -> tuple[EnvironmentContract, CollectionContract]:
    record = _exact_object(value, _HEADER_KEYS, "run_header")
    if record["schema_version"] != RUN_HEADER_VERSION:
        raise RecordValidationError("run header schema is invalid")
    try:
        env_contract = EnvironmentContract.from_record(
            record["environment_contract"]
        )
        collection = CollectionContract.from_record(
            record["collection_contract"]
        )
    except ValueError as exc:
        raise RecordValidationError(str(exc)) from exc
    if record["environment_contract_sha256"] != env_contract.sha256:
        raise RecordValidationError("run header environment digest is invalid")
    if record["collection_contract_sha256"] != collection.sha256:
        raise RecordValidationError("run header collection digest is invalid")
    try:
        validate_collection_binding(env_contract, collection)
    except (TypeError, ValueError) as exc:
        raise RecordValidationError(
            f"run header contracts are incompatible: {exc}"
        ) from exc
    if environment is not None and (
        environment.contract != env_contract
        or environment.sha256 != record["environment_contract_sha256"]
    ):
        raise RecordValidationError(
            "run header differs from the bound environment"
        )
    if collection_contract is not None and collection != collection_contract:
        raise RecordValidationError(
            "run header differs from the collection contract"
        )
    return env_contract, collection


def _serialize_closure_events(
    transition: MacroTransition,
) -> list[dict[str, object]]:
    return [
        {
            "sequence": event.sequence,
            "claim_id": event.claim_id,
            "outcome": serialize_outcome(event.outcome),
        }
        for event in transition.closure_events
    ]


def build_transition_record(
    transition: MacroTransition,
    *,
    environment: BoundEnvironment,
    collection_contract: CollectionContract,
    episode_key: str,
    problem_id: str,
) -> dict[str, object]:
    environment.validate_transition(transition)
    if not isinstance(collection_contract, CollectionContract):
        raise TypeError("collection_contract has an invalid type")
    if transition.collection_contract_sha256 != collection_contract.sha256:
        raise RecordValidationError(
            "transition does not come from the bound collection runtime"
        )
    _nonempty_text(episode_key, "episode_key")
    _nonempty_text(problem_id, "problem_id")
    provenance = transition.decision_provenance
    if provenance is None:
        raise RecordValidationError(
            "canonical records require bound controller provenance"
        )
    observation = environment.observe(transition.state)
    if provenance.episode_key is not None and (
        provenance.episode_key != episode_key
    ):
        raise RecordValidationError(
            "transition receipt episode_key differs from the record"
        )
    if provenance.observation_sha256 is not None and (
        provenance.observation_sha256
        != canonical_json_sha256(observation)
    ):
        raise RecordValidationError(
            "transition receipt observation differs from the record"
        )
    behavior = BehaviorMetadata(
        provenance.source,
        provenance.fallback,
    )
    diagnostic_ref = (
        None
        if provenance.diagnostic_record_id is None
        else DiagnosticRef(provenance.diagnostic_record_id)
    )
    record = {
        "schema_version": TRANSITION_RECORD_VERSION,
        "environment_contract_sha256": environment.sha256,
        "collection_contract_sha256": collection_contract.sha256,
        "episode_key": episode_key,
        "problem_id": problem_id,
        "turn": transition.state.turn,
        "observation": observation,
        "action": serialize_action(transition.action),
        "learned_effect": serialize_learned_effect(transition),
        "outcome": serialize_outcome(transition.outcome),
        "closure_events": _serialize_closure_events(transition),
        "reward": transition.reward,
        "next_observation": environment.observe(transition.next_state),
        "terminal": transition.terminal,
        "behavior": behavior.to_record(),
        "diagnostic_ref": (
            None if diagnostic_ref is None else diagnostic_ref.to_record()
        ),
    }
    validate_transition_record(
        record,
        environment=environment,
        collection_contract=collection_contract,
        expected_transition=transition,
    )
    return record


def validate_transition_record(
    value: object,
    *,
    environment: BoundEnvironment,
    collection_contract: CollectionContract | None = None,
    expected_transition: MacroTransition | None = None,
) -> None:
    record = _exact_object(value, _TRANSITION_KEYS, "transition")
    if record["schema_version"] != TRANSITION_RECORD_VERSION:
        raise RecordValidationError("transition record schema is invalid")
    if record["environment_contract_sha256"] != environment.sha256:
        raise RecordValidationError(
            "transition record environment digest is invalid"
        )
    collection_digest = record["collection_contract_sha256"]
    if not isinstance(collection_digest, str) or len(
        collection_digest
    ) != 64 or any(
        char not in "0123456789abcdef" for char in collection_digest
    ):
        raise RecordValidationError(
            "transition record collection digest is invalid"
        )
    if (
        collection_contract is not None
        and collection_digest != collection_contract.sha256
    ):
        raise RecordValidationError(
            "transition record collection digest differs from the header"
        )
    _nonempty_text(record["episode_key"], "transition.episode_key")
    _nonempty_text(record["problem_id"], "transition.problem_id")
    turn = _integer(record["turn"], "transition.turn")

    for field in ("observation", "next_observation"):
        try:
            validate_observation_with_contract(
                record[field],
                environment.generation_contract,
                environment.verifier,
                environment_contract_sha256=environment.sha256,
                turn_limit=environment.contract.turn_limit,
            )
        except (TypeError, ValueError) as exc:
            raise RecordValidationError(
                f"transition.{field} is invalid: {exc}"
            ) from exc
    observation = record["observation"]
    next_observation = record["next_observation"]
    if observation["turn"] != turn:
        raise RecordValidationError("transition turn differs from observation")

    action = deserialize_action(record["action"], "transition.action")
    if record["action"] not in observation["legal_pairs"]:
        raise RecordValidationError(
            "transition action is outside the current learned menu"
        )
    outcome = deserialize_outcome(record["outcome"], "transition.outcome")
    if outcome.action != action:
        raise RecordValidationError("transition outcome action differs")
    if outcome.reason is RejectionReason.STALE_ACTION:
        raise RecordValidationError(
            "stale actions cannot form canonical transition records"
        )
    draft = deserialize_learned_effect(
        record["learned_effect"], action, outcome
    )
    if outcome.code is OutcomeCode.REJECTED and record["closure_events"]:
        raise RecordValidationError("rejected transition cannot have closure")

    events = record["closure_events"]
    if type(events) is not list:
        raise RecordValidationError("closure_events must be an array")
    for index, value in enumerate(events):
        event = _exact_object(
            value,
            _CLOSURE_EVENT_KEYS,
            f"transition.closure_events[{index}]",
        )
        if event["sequence"] != index:
            raise RecordValidationError(
                "closure event sequence must be contiguous"
            )
        claim_id = _integer(
            event["claim_id"],
            f"transition.closure_events[{index}].claim_id",
        )
        internal = deserialize_outcome(
            event["outcome"],
            f"transition.closure_events[{index}].outcome",
        )
        if internal.code is not OutcomeCode.APPLIED or internal.action.verb not in {
            Verb.VERIFY, Verb.COMMIT, Verb.AGGREGATE
        }:
            raise RecordValidationError("closure event outcome is not internal")
        if internal.action.verb is Verb.VERIFY:
            context = internal.verifier_context
            if (
                context is None
                or context.contract_id != environment.verifier.contract_id
            ):
                raise RecordValidationError(
                    "closure VERIFY context differs from the verifier"
                )
        elif internal.verifier_context is not None:
            raise RecordValidationError(
                "non-VERIFY closure event has verifier context"
            )
        if internal.action.verb is Verb.AGGREGATE:
            if claim_id <= internal.action.target:
                raise RecordValidationError(
                    "aggregate closure event Claim id is invalid"
                )
        elif claim_id != internal.action.target:
            raise RecordValidationError(
                "closure event Claim id differs from its target"
            )

    if next_observation["turn"] != turn + 1:
        raise RecordValidationError(
            "next observation must consume one learned turn"
        )
    if next_observation["previous_outcome"] != record["outcome"]:
        raise RecordValidationError(
            "next observation does not retain the learned outcome"
        )
    if type(record["reward"]) is not int or record["reward"] not in {0, 1}:
        raise RecordValidationError("transition reward is invalid")
    terminal = _boolean(record["terminal"], "transition.terminal")
    root_id = next_observation["root_goal_id"]
    root = next(
        goal for goal in next_observation["goals"] if goal["id"] == root_id
    )
    root_solved = root["status"] == "solved"
    expected_terminal = (
        root_solved
        or next_observation["turn"] >= next_observation["turn_limit"]
    )
    if terminal is not expected_terminal:
        raise RecordValidationError("transition terminal predicate differs")
    if record["reward"] != (1 if root_solved else 0):
        raise RecordValidationError("transition reward predicate differs")
    if terminal == bool(next_observation["legal_pairs"]):
        raise RecordValidationError(
            "terminal flag and next learned menu disagree"
        )

    BehaviorMetadata.from_record(record["behavior"])
    if record["diagnostic_ref"] is not None:
        DiagnosticRef.from_record(record["diagnostic_ref"])

    if expected_transition is not None:
        if collection_contract is None:
            raise TypeError(
                "exact transition validation needs a collection contract"
            )
        if (
            expected_transition.collection_contract_sha256
            != collection_contract.sha256
        ):
            raise RecordValidationError(
                "transition collection provenance differs from the header"
            )
        environment.validate_transition(expected_transition)
        expected = {
            "turn": expected_transition.state.turn,
            "collection_contract_sha256": (
                expected_transition.collection_contract_sha256
            ),
            "action": serialize_action(expected_transition.action),
            "learned_effect": serialize_learned_effect(expected_transition),
            "outcome": serialize_outcome(expected_transition.outcome),
            "closure_events": _serialize_closure_events(expected_transition),
            "reward": expected_transition.reward,
            "terminal": expected_transition.terminal,
            "observation": environment.observe(expected_transition.state),
            "next_observation": environment.observe(
                expected_transition.next_state
            ),
            "behavior": BehaviorMetadata(
                expected_transition.decision_provenance.source,
                expected_transition.decision_provenance.fallback,
            ).to_record(),
            "diagnostic_ref": (
                None
                if expected_transition.decision_provenance.diagnostic_record_id
                is None
                else DiagnosticRef(
                    expected_transition.decision_provenance.diagnostic_record_id
                ).to_record()
            ),
        }
        for field, expected_value in expected.items():
            if record[field] != expected_value:
                raise RecordValidationError(
                    f"transition {field} differs from exact replay"
                )
        if draft != expected_transition.policy_draft:
            raise RecordValidationError(
                "transition learned effect differs from its typed draft"
            )


def canonical_record_bytes(
    value: object,
    *,
    environment: BoundEnvironment | None = None,
    collection_contract: CollectionContract | None = None,
) -> bytes:
    if type(value) is not dict:
        raise RecordValidationError("record must be an object")
    schema = value.get("schema_version")
    if schema == RUN_HEADER_VERSION:
        validate_run_header(value, environment=environment)
    elif schema == TRANSITION_RECORD_VERSION:
        if environment is None or collection_contract is None:
            raise TypeError(
                "transition canonicalization needs bound run contracts"
            )
        validate_transition_record(
            value,
            environment=environment,
            collection_contract=collection_contract,
        )
    else:
        raise RecordValidationError("record schema is unknown")
    return canonical_json_bytes(value)


def _reject_constant(value: str) -> None:
    raise RecordValidationError(f"non-finite JSON constant {value!r}")


def _pairs_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RecordValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_canonical_json_line(line: bytes) -> dict[str, object]:
    if not isinstance(line, bytes) or not line.endswith(b"\n"):
        raise RecordValidationError(
            "canonical JSONL record needs one trailing newline"
        )
    if line.endswith(b"\r\n") or line.count(b"\n") != 1:
        raise RecordValidationError("canonical JSONL record has invalid newlines")
    try:
        text = line[:-1].decode("ascii")
    except UnicodeDecodeError as exc:
        raise RecordValidationError("canonical JSONL must be ASCII") from exc
    if not text:
        raise RecordValidationError("canonical JSONL record cannot be blank")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RecordValidationError("canonical JSONL is invalid JSON") from exc
    if type(value) is not dict:
        raise RecordValidationError("canonical JSONL record must be an object")
    if canonical_json_bytes(value) + b"\n" != line:
        raise RecordValidationError("JSONL record is not canonically encoded")
    return value


class CanonicalRunWriter:
    """Header-first, append-only canonical JSONL writer."""

    def __init__(
        self,
        path: str | Path,
        header: dict[str, object],
        *,
        environment: BoundEnvironment,
        collection_contract: CollectionContract,
    ) -> None:
        validate_run_header(
            header,
            environment=environment,
            collection_contract=collection_contract,
        )
        self._environment = environment
        self._collection_contract = collection_contract
        self._header = header
        self._lock = threading.Lock()
        self._last_by_episode: dict[str, dict[str, object]] = {}
        self._closed = False
        self._poisoned = False
        self._file: BinaryIO = Path(path).open("xb")
        try:
            self._file.write(canonical_json_bytes(header) + b"\n")
            self._file.flush()
        except Exception:
            self._file.close()
            self._closed = True
            self._poisoned = True
            raise

    def append(
        self,
        record: dict[str, object],
        *,
        transition: MacroTransition,
    ) -> None:
        encoded = canonical_json_bytes(record)
        snapshot = json.loads(encoded)
        validate_transition_record(
            snapshot,
            environment=self._environment,
            collection_contract=self._collection_contract,
            expected_transition=transition,
        )
        episode_key = snapshot["episode_key"]
        with self._lock:
            if self._poisoned:
                raise RuntimeError("run writer is poisoned by an I/O failure")
            if self._closed:
                raise RuntimeError("run writer is closed")
            previous = self._last_by_episode.get(episode_key)
            if previous is None:
                if snapshot["turn"] != 0:
                    raise RecordValidationError(
                        "an episode must begin with turn zero"
                    )
                if snapshot["observation"]["previous_outcome"] is not None:
                    raise RecordValidationError(
                        "initial episode observation has a previous outcome"
                    )
            else:
                if previous["terminal"]:
                    raise RecordValidationError(
                        "cannot append after an episode terminal"
                    )
                if previous["problem_id"] != snapshot["problem_id"]:
                    raise RecordValidationError(
                        "episode problem_id changed between turns"
                    )
                if previous["next_observation"] != snapshot["observation"]:
                    raise RecordValidationError(
                        "episode observations do not chain exactly"
                    )
            try:
                self._file.write(encoded + b"\n")
                self._file.flush()
            except Exception:
                try:
                    self._file.close()
                finally:
                    self._closed = True
                    self._poisoned = True
                raise
            self._last_by_episode[episode_key] = snapshot

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._file.close()
                self._closed = True

    def __enter__(self) -> CanonicalRunWriter:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = [
    "BehaviorMetadata",
    "BehaviorSource",
    "CanonicalRunWriter",
    "DIAGNOSTIC_REF_VERSION",
    "DiagnosticRef",
    "RUN_HEADER_VERSION",
    "RecordValidationError",
    "TRANSITION_RECORD_VERSION",
    "build_run_header",
    "build_transition_record",
    "canonical_record_bytes",
    "deserialize_action",
    "deserialize_learned_effect",
    "deserialize_outcome",
    "parse_canonical_json_line",
    "serialize_learned_effect",
    "validate_run_header",
    "validate_transition_record",
]
