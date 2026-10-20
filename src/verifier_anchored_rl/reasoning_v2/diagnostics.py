"""Append-only generator diagnostics for reasoning v2 collection."""

from __future__ import annotations

import copy
import json
import re
import threading
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Mapping

from ..llm_client import (
    OpenAICompatibleAttempt,
    OpenAICompatibleLLMClient,
    append_openai_compatible_parse_retry,
    build_openai_compatible_json_body,
    classify_openai_compatible_json_content,
    project_openai_compatible_response,
)
from .actions import (
    DecompositionDraft,
    RefutationDraft,
    ResultDraft,
    SolutionDraft,
)
from .controller import (
    POLICY_INPUT_CONTRACT_VERSION,
    SUPPORTED_CONTROLLER_OUTPUT_CONTRACT_VERSIONS,
    ControllerPayloadError,
    TypedLLMController,
    build_controller_request,
    decode_controller_payload,
    schema_validation_mode_for_controller_output,
)
from .contracts import (
    CANONICAL_JSON_VERSION,
    CollectionContract,
    ModelTransportContract,
    canonical_json_bytes,
    validate_collection_binding,
)
from .lenses import (
    SUPPORTED_GENERATION_SCHEMA_VERSIONS,
    TypedPolicyDraftGenerator,
    build_policy_lens_request,
    decode_policy_payload,
    llm_transport_sha256,
    schema_validation_mode_for_generation_schema,
)
from .menu import POLICY_VERB_ORDER
from .model import (
    Action,
    ClaimKind,
    GoalObjective,
    GoalObligation,
    RefutationPayload,
    RejectionReason,
    ResultPayload,
    SolutionPayload,
    Verb,
)
from .observation import (
    ObservationValidationError,
    observation_sha256 as canonical_observation_sha256,
)
from .records import (
    DiagnosticRef,
    RecordValidationError,
    deserialize_action,
    parse_canonical_json_line,
    serialize_action,
)
from .provenance import BehaviorSource
from .runtime import BoundEnvironment
from .transition import PolicyDraft


DIAGNOSTIC_HEADER_VERSION = "generator-diagnostic-header-v2"
DIAGNOSTIC_ATTEMPT_VERSION = "generator-http-attempt-v2"
DIAGNOSTIC_CALL_VERSION = "generator-call-diagnostic-v2"
DECISION_RECEIPT_VERSION = "controller-decision-receipt-v2"
GENERATOR_DIAGNOSTIC_COVERAGE = "openai-compatible-http-attempts-v2"
CONTROLLER_RECEIPT_COVERAGE = "controller-decision-receipts-v2"
CONTROLLER_ATTEMPT_COVERAGE = "openai-compatible-http-attempts-v2"
DIAGNOSTIC_COVERAGE_UNAVAILABLE = "not-collected-v2"
_TYPED_GENERATOR_IMPLEMENTATION = (
    f"{TypedPolicyDraftGenerator.__module__}."
    f"{TypedPolicyDraftGenerator.__qualname__}"
)
_TYPED_CONTROLLER_IMPLEMENTATION = (
    f"{TypedLLMController.__module__}.{TypedLLMController.__qualname__}"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_UUID_RE = re.compile(r"[0-9a-f]{32}")
_HEADER_KEYS = frozenset({
    "schema_version",
    "canonical_serializer",
    "environment_contract_sha256",
    "collection_contract_sha256",
    "generator_transport_sha256",
    "controller_transport_sha256",
    "coverage",
})
_COVERAGE_KEYS = frozenset({
    "generator_attempts",
    "controller_decision_receipts",
    "controller_attempts",
})
_ATTEMPT_KEYS = frozenset({
    "schema_version",
    "record_id",
    "role",
    "call_id",
    "attempt_index",
    "parse_retry_index",
    "backoff_retry_index",
    "effective_seed",
    "request_body",
    "raw_response_body",
    "raw_content",
    "usage",
    "outcome",
    "disposition",
    "http_status",
    "error",
})
_USAGE_KEYS = frozenset({
    "reported",
    "raw",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
})
_ERROR_KEYS = frozenset({"type", "message"})
_CALL_KEYS = frozenset({
    "schema_version",
    "record_id",
    "role",
    "call_id",
    "attempt_record_ids",
    "final_outcome",
    "final_attempt_outcome",
})
_DECISION_RECEIPT_KEYS = frozenset({
    "schema_version",
    "record_id",
    "collection_contract_sha256",
    "episode_key",
    "turn",
    "observation_sha256",
    "source",
    "proposed_action",
    "selected_action",
    "fallback",
    "controller_call_ref",
    "generator_call_ref",
})
_REQUEST_REQUIRED_KEYS = frozenset({
    "model",
    "messages",
    "temperature",
    "max_tokens",
    "response_format",
})
_REQUEST_OPTIONAL_KEYS = frozenset({"guided_json", "seed"})
_SECRET_KEYS = frozenset({
    "api-key",
    "api_key",
    "authorization",
    "access-token",
    "access_token",
    "hf_token",
    "password",
    "secret",
})
_ATTEMPT_OUTCOMES = frozenset({
    "accepted",
    "parse_error",
    "schema_error",
    "provider_response_error",
    "http_status_error",
    "transport_error",
})
_DISPOSITIONS = frozenset({
    "return",
    "parse_retry",
    "backoff_retry",
    "raise",
})
_MODEL_ROLES = frozenset({"generator", "controller"})
_INDEX_READER_TOKEN = object()


class DiagnosticValidationError(ValueError):
    """Raised when a generator diagnostic record violates its contract."""


def _exact_object(
    value: object, keys: frozenset[str], path: str
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise DiagnosticValidationError(
            f"{path} keys must be exactly {sorted(keys)!r}"
        )
    return value


def _sha256(value: object, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DiagnosticValidationError(f"{path} must be a sha256 digest")
    return value


def _uuid(value: object, path: str) -> str:
    if not isinstance(value, str) or _UUID_RE.fullmatch(value) is None:
        raise DiagnosticValidationError(f"{path} must be a lowercase UUID")
    return value


def _nonempty_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DiagnosticValidationError(f"{path} must be nonempty text")
    return value


def _optional_text(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _nonempty_text(value, path)


def _integer(value: object, path: str) -> int:
    if type(value) is not int or value < 0:
        raise DiagnosticValidationError(
            f"{path} must be a nonnegative integer"
        )
    return value


def _optional_integer(value: object, path: str) -> int | None:
    if value is None:
        return None
    return _integer(value, path)


def _boolean(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise DiagnosticValidationError(f"{path} must be boolean")
    return value


def _reject_secret_keys(value: object, path: str = "request_body") -> None:
    if type(value) is dict:
        for key, child in value.items():
            if not isinstance(key, str):
                raise DiagnosticValidationError(
                    f"{path} contains a non-text key"
                )
            if key.lower() in _SECRET_KEYS:
                raise DiagnosticValidationError(
                    f"{path} contains a forbidden secret field"
                )
            _reject_secret_keys(child, f"{path}.{key}")
    elif type(value) is list:
        for index, child in enumerate(value):
            _reject_secret_keys(child, f"{path}[{index}]")


def _snapshot_value(snapshot: str, path: str) -> object:
    if not isinstance(snapshot, str):
        raise DiagnosticValidationError(f"{path} snapshot must be text")
    try:
        value = json.loads(snapshot)
    except (json.JSONDecodeError, TypeError) as exc:
        raise DiagnosticValidationError(
            f"{path} snapshot is invalid JSON"
        ) from exc
    if canonical_json_bytes(value).decode("ascii") != snapshot:
        raise DiagnosticValidationError(
            f"{path} snapshot is not canonical JSON"
        )
    return value


def build_diagnostic_header(
    *,
    environment_contract_sha256: str,
    collection_contract_sha256: str,
    generator_transport_sha256: str,
    controller_transport_sha256: str,
    controller_decision_receipts: bool = False,
    controller_attempts: bool = False,
) -> dict[str, object]:
    for field_name, value in (
        ("controller_decision_receipts", controller_decision_receipts),
        ("controller_attempts", controller_attempts),
    ):
        if type(value) is not bool:
            raise TypeError(f"{field_name} must be boolean")
    record = {
        "schema_version": DIAGNOSTIC_HEADER_VERSION,
        "canonical_serializer": CANONICAL_JSON_VERSION,
        "environment_contract_sha256": environment_contract_sha256,
        "collection_contract_sha256": collection_contract_sha256,
        "generator_transport_sha256": generator_transport_sha256,
        "controller_transport_sha256": controller_transport_sha256,
        "coverage": {
            "generator_attempts": GENERATOR_DIAGNOSTIC_COVERAGE,
            "controller_decision_receipts": (
                CONTROLLER_RECEIPT_COVERAGE
                if controller_decision_receipts
                else DIAGNOSTIC_COVERAGE_UNAVAILABLE
            ),
            "controller_attempts": (
                CONTROLLER_ATTEMPT_COVERAGE
                if controller_attempts
                else DIAGNOSTIC_COVERAGE_UNAVAILABLE
            ),
        },
    }
    validate_diagnostic_header(record)
    return record


def validate_diagnostic_header(
    value: object,
    *,
    environment_contract_sha256: str | None = None,
    collection_contract_sha256: str | None = None,
    generator_transport_sha256: str | None = None,
    controller_transport_sha256: str | None = None,
) -> None:
    record = _exact_object(value, _HEADER_KEYS, "diagnostic header")
    if record["schema_version"] != DIAGNOSTIC_HEADER_VERSION:
        raise DiagnosticValidationError("diagnostic header schema is invalid")
    if record["canonical_serializer"] != CANONICAL_JSON_VERSION:
        raise DiagnosticValidationError(
            "diagnostic canonical serializer is invalid"
        )
    environment_digest = _sha256(
        record["environment_contract_sha256"],
        "diagnostic environment_contract_sha256",
    )
    collection_digest = _sha256(
        record["collection_contract_sha256"],
        "diagnostic collection_contract_sha256",
    )
    transport_digest = _sha256(
        record["generator_transport_sha256"],
        "diagnostic generator_transport_sha256",
    )
    controller_digest = _sha256(
        record["controller_transport_sha256"],
        "diagnostic controller_transport_sha256",
    )
    coverage = _exact_object(
        record["coverage"], _COVERAGE_KEYS, "diagnostic coverage"
    )
    if coverage["generator_attempts"] != GENERATOR_DIAGNOSTIC_COVERAGE:
        raise DiagnosticValidationError("diagnostic coverage is invalid")
    if coverage["controller_decision_receipts"] not in {
        CONTROLLER_RECEIPT_COVERAGE,
        DIAGNOSTIC_COVERAGE_UNAVAILABLE,
    }:
        raise DiagnosticValidationError(
            "controller decision-receipt coverage is invalid"
        )
    if coverage["controller_attempts"] not in {
        CONTROLLER_ATTEMPT_COVERAGE,
        DIAGNOSTIC_COVERAGE_UNAVAILABLE,
    }:
        raise DiagnosticValidationError(
            "controller attempt coverage is invalid"
        )
    expected = (
        (environment_digest, environment_contract_sha256, "environment"),
        (collection_digest, collection_contract_sha256, "collection"),
        (transport_digest, generator_transport_sha256, "transport"),
        (
            controller_digest,
            controller_transport_sha256,
            "controller transport",
        ),
    )
    for actual, pinned, name in expected:
        if pinned is not None and actual != pinned:
            raise DiagnosticValidationError(
                f"diagnostic {name} digest differs from the bound run"
            )


def require_hosted_bc_diagnostic_coverage(value: object) -> None:
    """Fail closed until decision receipts and raw controller calls exist."""
    validate_diagnostic_header(value)
    coverage = value["coverage"]
    if (
        coverage["controller_decision_receipts"]
        != CONTROLLER_RECEIPT_COVERAGE
        or coverage["controller_attempts"]
        != CONTROLLER_ATTEMPT_COVERAGE
    ):
        raise DiagnosticValidationError(
            "hosted BC is gated on controller receipts and raw attempts"
        )


def _validate_request_body(
    value: object, effective_seed: int | None
) -> dict[str, Any]:
    if type(value) is not dict:
        raise DiagnosticValidationError("request_body must be an object")
    keys = set(value)
    if not _REQUEST_REQUIRED_KEYS <= keys or not keys <= (
        _REQUEST_REQUIRED_KEYS | _REQUEST_OPTIONAL_KEYS
    ):
        raise DiagnosticValidationError("request_body keys are invalid")
    _reject_secret_keys(value)
    _nonempty_text(value["model"], "request_body.model")
    if type(value["messages"]) is not list or not value["messages"]:
        raise DiagnosticValidationError(
            "request_body.messages must be a nonempty array"
        )
    if type(value["max_tokens"]) is not int or value["max_tokens"] <= 0:
        raise DiagnosticValidationError(
            "request_body.max_tokens must be a positive integer"
        )
    temperature = value["temperature"]
    if isinstance(temperature, bool) or type(temperature) not in {int, float}:
        raise DiagnosticValidationError(
            "request_body.temperature must be numeric"
        )
    if effective_seed is None:
        if "seed" in value:
            raise DiagnosticValidationError(
                "unseeded attempt request unexpectedly contains a seed"
            )
    elif value.get("seed") != effective_seed:
        raise DiagnosticValidationError(
            "effective seed differs from the request body"
        )
    canonical_json_bytes(value)
    return value


def build_attempt_record(
    event: OpenAICompatibleAttempt,
    *,
    role: str,
) -> dict[str, object]:
    if not isinstance(event, OpenAICompatibleAttempt):
        raise TypeError("diagnostic attempt event has an invalid type")
    if role not in _MODEL_ROLES:
        raise ValueError("diagnostic model role is invalid")
    request_body = _snapshot_value(
        event.request_body_json, "request_body"
    )
    raw_usage = (
        None
        if event.raw_usage_json is None
        else _snapshot_value(event.raw_usage_json, "usage.raw")
    )
    error = (
        None
        if event.error_type is None and event.error_message is None
        else {
            "type": event.error_type,
            "message": event.error_message,
        }
    )
    record = {
        "schema_version": DIAGNOSTIC_ATTEMPT_VERSION,
        "record_id": (
            f"{role}:{event.call_id}:attempt:{event.attempt_index}"
        ),
        "role": role,
        "call_id": event.call_id,
        "attempt_index": event.attempt_index,
        "parse_retry_index": event.parse_retry_index,
        "backoff_retry_index": event.backoff_retry_index,
        "effective_seed": event.effective_seed,
        "request_body": request_body,
        "raw_response_body": event.raw_response_body,
        "raw_content": event.raw_content,
        "usage": {
            "reported": event.raw_usage_json is not None,
            "raw": raw_usage,
            "prompt_tokens": event.prompt_tokens,
            "completion_tokens": event.completion_tokens,
            "total_tokens": event.total_tokens,
        },
        "outcome": event.outcome,
        "disposition": event.disposition,
        "http_status": event.http_status,
        "error": error,
    }
    validate_attempt_record(record)
    return record


def validate_attempt_record(value: object) -> None:
    record = _exact_object(value, _ATTEMPT_KEYS, "diagnostic attempt")
    if record["schema_version"] != DIAGNOSTIC_ATTEMPT_VERSION:
        raise DiagnosticValidationError("attempt schema is invalid")
    role = record["role"]
    if role not in _MODEL_ROLES:
        raise DiagnosticValidationError("attempt role is invalid")
    call_id = _uuid(record["call_id"], "attempt.call_id")
    attempt_index = _integer(record["attempt_index"], "attempt.index")
    if record["record_id"] != (
        f"{role}:{call_id}:attempt:{attempt_index}"
    ):
        raise DiagnosticValidationError("attempt record_id is invalid")
    _integer(record["parse_retry_index"], "attempt.parse_retry_index")
    _integer(record["backoff_retry_index"], "attempt.backoff_retry_index")
    effective_seed = _optional_integer(
        record["effective_seed"], "attempt.effective_seed"
    )
    _validate_request_body(record["request_body"], effective_seed)
    for field_name in ("raw_response_body", "raw_content"):
        value = record[field_name]
        if value is not None and not isinstance(value, str):
            raise DiagnosticValidationError(
                f"attempt.{field_name} must be text or null"
            )
    usage = _exact_object(record["usage"], _USAGE_KEYS, "attempt.usage")
    reported = _boolean(usage["reported"], "attempt.usage.reported")
    if reported is not (usage["raw"] is not None):
        raise DiagnosticValidationError(
            "attempt usage reported flag differs from raw usage"
        )
    canonical_json_bytes(usage["raw"])
    for field_name in (
        "prompt_tokens", "completion_tokens", "total_tokens"
    ):
        _optional_integer(
            usage[field_name], f"attempt.usage.{field_name}"
        )
    outcome = record["outcome"]
    disposition = record["disposition"]
    if outcome not in _ATTEMPT_OUTCOMES:
        raise DiagnosticValidationError("attempt outcome is invalid")
    if disposition not in _DISPOSITIONS:
        raise DiagnosticValidationError("attempt disposition is invalid")
    status = _optional_integer(record["http_status"], "attempt.http_status")
    error = record["error"]
    if outcome == "accepted":
        if disposition != "return" or error is not None:
            raise DiagnosticValidationError(
                "accepted attempt must return without an error"
            )
        if not isinstance(record["raw_content"], str):
            raise DiagnosticValidationError(
                "accepted attempt must retain raw content"
            )
    else:
        if disposition not in {"parse_retry", "backoff_retry", "raise"}:
            raise DiagnosticValidationError(
                "failed attempt disposition is invalid"
            )
        error_record = _exact_object(error, _ERROR_KEYS, "attempt.error")
        _nonempty_text(error_record["type"], "attempt.error.type")
        _nonempty_text(error_record["message"], "attempt.error.message")
    if outcome == "http_status_error":
        if status is None or status < 400:
            raise DiagnosticValidationError(
                "HTTP status error must retain its failing status"
            )
        if disposition not in {"backoff_retry", "raise"}:
            raise DiagnosticValidationError(
                "HTTP status error has an invalid retry class"
            )
    elif outcome == "transport_error":
        if status is not None or disposition == "backoff_retry":
            raise DiagnosticValidationError(
                "transport error has invalid HTTP metadata"
            )
    elif disposition == "backoff_retry":
        raise DiagnosticValidationError(
            "only an HTTP status error may use backoff retry"
        )


def _validate_attempt_sequence(
    previous: Mapping[str, object] | None,
    current: Mapping[str, object],
) -> None:
    if previous is None:
        if (
            current["attempt_index"] != 0
            or current["parse_retry_index"] != 0
            or current["backoff_retry_index"] != 0
        ):
            raise DiagnosticValidationError(
                "a diagnostic call must begin at attempt zero"
            )
        return
    if (
        current["role"] != previous["role"]
        or current["call_id"] != previous["call_id"]
    ):
        raise DiagnosticValidationError(
            "attempt sequence changed model role or call id"
        )
    if current["attempt_index"] != previous["attempt_index"] + 1:
        raise DiagnosticValidationError("attempt indices are not contiguous")
    disposition = previous["disposition"]
    if disposition == "parse_retry":
        expected_parse = previous["parse_retry_index"] + 1
        expected_backoff = previous["backoff_retry_index"]
    elif disposition == "backoff_retry":
        expected_parse = previous["parse_retry_index"]
        expected_backoff = previous["backoff_retry_index"] + 1
    else:
        raise DiagnosticValidationError(
            "a terminal attempt cannot have a successor"
        )
    if (
        current["parse_retry_index"] != expected_parse
        or current["backoff_retry_index"] != expected_backoff
    ):
        raise DiagnosticValidationError(
            "attempt retry indices do not follow their disposition"
        )
    previous_seed = previous["effective_seed"]
    current_seed = current["effective_seed"]
    if disposition == "parse_retry" and previous_seed is not None:
        if current_seed != previous_seed + 1:
            raise DiagnosticValidationError(
                "parse retry did not increment the effective seed"
            )
    elif current_seed != previous_seed:
        raise DiagnosticValidationError(
            "backoff retry changed the effective seed"
        )


def build_call_record(
    role: str,
    call_id: str,
    attempts: list[dict[str, object]],
) -> dict[str, object]:
    if role not in _MODEL_ROLES:
        raise DiagnosticValidationError("call role is invalid")
    _uuid(call_id, "call.call_id")
    if not attempts:
        raise DiagnosticValidationError(
            "a completed diagnostic call needs at least one HTTP attempt"
        )
    final = attempts[-1]
    if final["disposition"] not in {"return", "raise"}:
        raise DiagnosticValidationError(
            "diagnostic call cannot end on a retry disposition"
        )
    record = {
        "schema_version": DIAGNOSTIC_CALL_VERSION,
        "record_id": f"{role}:{call_id}",
        "role": role,
        "call_id": call_id,
        "attempt_record_ids": [attempt["record_id"] for attempt in attempts],
        "final_outcome": (
            "transport_accepted"
            if final["disposition"] == "return"
            else "transport_error"
        ),
        "final_attempt_outcome": final["outcome"],
    }
    validate_call_record(record, attempts=attempts)
    return record


def validate_call_record(
    value: object,
    *,
    attempts: list[dict[str, object]] | None = None,
) -> None:
    record = _exact_object(value, _CALL_KEYS, "diagnostic call")
    if record["schema_version"] != DIAGNOSTIC_CALL_VERSION:
        raise DiagnosticValidationError("call schema is invalid")
    role = record["role"]
    if role not in _MODEL_ROLES:
        raise DiagnosticValidationError("call role is invalid")
    call_id = _uuid(record["call_id"], "call.call_id")
    if record["record_id"] != f"{role}:{call_id}":
        raise DiagnosticValidationError("call record_id is invalid")
    attempt_ids = record["attempt_record_ids"]
    if type(attempt_ids) is not list or not attempt_ids:
        raise DiagnosticValidationError(
            "call attempt_record_ids must be a nonempty array"
        )
    expected_ids = [
        f"{role}:{call_id}:attempt:{index}"
        for index in range(len(attempt_ids))
    ]
    if attempt_ids != expected_ids:
        raise DiagnosticValidationError("call attempt references are invalid")
    if record["final_outcome"] not in {
        "transport_accepted", "transport_error"
    }:
        raise DiagnosticValidationError("call final_outcome is invalid")
    if record["final_attempt_outcome"] not in _ATTEMPT_OUTCOMES:
        raise DiagnosticValidationError(
            "call final_attempt_outcome is invalid"
        )
    if (
        record["final_outcome"] == "transport_accepted"
    ) is not (
        record["final_attempt_outcome"] == "accepted"
    ):
        raise DiagnosticValidationError(
            "call and final-attempt outcomes disagree"
        )
    if attempts is not None:
        if any(
            attempt["role"] != role or attempt["call_id"] != call_id
            for attempt in attempts
        ):
            raise DiagnosticValidationError(
                "call attempts differ from its role or call id"
            )
        if [attempt["record_id"] for attempt in attempts] != attempt_ids:
            raise DiagnosticValidationError(
                "call attempt references do not resolve exactly"
            )
        final = attempts[-1]
        if final["disposition"] not in {"return", "raise"}:
            raise DiagnosticValidationError(
                "call record cannot terminate a pending retry"
            )
        expected_outcome = (
            "transport_accepted"
            if final["disposition"] == "return"
            else "transport_error"
        )
        if (
            record["final_outcome"] != expected_outcome
            or record["final_attempt_outcome"] != final["outcome"]
        ):
            raise DiagnosticValidationError(
                "call terminal outcome differs from its final attempt"
            )


def build_decision_receipt(
    *,
    record_id: str,
    collection_contract_sha256: str,
    episode_key: str,
    turn: int,
    observation_sha256: str,
    source: str,
    proposed_action: Action | None,
    selected_action: Action,
    fallback: bool,
    controller_call_ref: DiagnosticRef | None,
    generator_call_ref: DiagnosticRef,
) -> dict[str, object]:
    if proposed_action is not None and not isinstance(proposed_action, Action):
        raise TypeError("proposed_action has an invalid type")
    if not isinstance(selected_action, Action):
        raise TypeError("selected_action has an invalid type")
    if controller_call_ref is not None and not isinstance(
        controller_call_ref, DiagnosticRef
    ):
        raise TypeError("controller_call_ref has an invalid type")
    if not isinstance(generator_call_ref, DiagnosticRef):
        raise TypeError("generator_call_ref has an invalid type")
    record = {
        "schema_version": DECISION_RECEIPT_VERSION,
        "record_id": record_id,
        "collection_contract_sha256": collection_contract_sha256,
        "episode_key": episode_key,
        "turn": turn,
        "observation_sha256": observation_sha256,
        "source": source,
        "proposed_action": (
            None
            if proposed_action is None
            else serialize_action(proposed_action)
        ),
        "selected_action": serialize_action(selected_action),
        "fallback": fallback,
        "controller_call_ref": (
            None
            if controller_call_ref is None
            else controller_call_ref.to_record()
        ),
        "generator_call_ref": generator_call_ref.to_record(),
    }
    validate_decision_receipt(record)
    return record


def validate_decision_receipt(value: object) -> None:
    record = _exact_object(
        value, _DECISION_RECEIPT_KEYS, "controller decision receipt"
    )
    if record["schema_version"] != DECISION_RECEIPT_VERSION:
        raise DiagnosticValidationError(
            "controller decision receipt schema is invalid"
        )
    _uuid(record["record_id"], "decision_receipt.record_id")
    _sha256(
        record["collection_contract_sha256"],
        "decision_receipt.collection_contract_sha256",
    )
    _nonempty_text(record["episode_key"], "decision_receipt.episode_key")
    _integer(record["turn"], "decision_receipt.turn")
    _sha256(
        record["observation_sha256"],
        "decision_receipt.observation_sha256",
    )
    source = record["source"]
    if source not in {"teacher", "learned_policy"}:
        raise DiagnosticValidationError(
            "decision receipt source is invalid"
        )
    try:
        proposed = (
            None
            if record["proposed_action"] is None
            else deserialize_action(
                record["proposed_action"],
                "decision_receipt.proposed_action",
            )
        )
        selected = deserialize_action(
            record["selected_action"],
            "decision_receipt.selected_action",
        )
        controller_ref = (
            None
            if record["controller_call_ref"] is None
            else DiagnosticRef.from_record(record["controller_call_ref"])
        )
        DiagnosticRef.from_record(record["generator_call_ref"])
    except (RecordValidationError, TypeError, ValueError) as exc:
        raise DiagnosticValidationError(
            f"decision receipt reference or action is invalid: {exc}"
        ) from exc
    if selected.verb not in POLICY_VERB_ORDER:
        raise DiagnosticValidationError(
            "decision receipt selected action is not learned"
        )
    fallback = _boolean(record["fallback"], "decision_receipt.fallback")
    if fallback:
        if source != "teacher":
            raise DiagnosticValidationError(
                "only teacher behavior may use a fallback"
            )
    elif proposed != selected:
        raise DiagnosticValidationError(
            "non-fallback proposed and selected actions must match"
        )
    if source == "learned_policy" and controller_ref is not None:
        raise DiagnosticValidationError(
            "learned-policy receipt cannot cite an LLM controller call"
        )


def require_bc_ready_decision_receipt(value: object) -> None:
    """Reject record-only BC promotion; readiness needs a resolved index."""
    validate_decision_receipt(value)
    if (
        value["source"] == "teacher"
        and value["fallback"] is False
        and value["controller_call_ref"] is None
    ):
        raise DiagnosticValidationError(
            "BC-eligible teacher receipt lacks a raw controller call"
        )
    raise DiagnosticValidationError(
        "a decision receipt alone cannot establish BC readiness; resolve "
        "and validate it through its diagnostic index"
    )


@dataclass(slots=True)
class _ActiveCall:
    attempts: list[dict[str, object]] = field(default_factory=list)


class DiagnosticSidecarWriter:
    """Capture role-bound model calls and compose decision receipts."""

    def __init__(
        self,
        path: str | Path,
        *,
        environment: BoundEnvironment,
        collection_contract: CollectionContract,
        generator_client: OpenAICompatibleLLMClient,
        controller_client: OpenAICompatibleLLMClient | None = None,
    ) -> None:
        if not isinstance(environment, BoundEnvironment):
            raise TypeError("diagnostic environment has an invalid type")
        if not isinstance(collection_contract, CollectionContract):
            raise TypeError("diagnostic collection contract has an invalid type")
        if not isinstance(generator_client, OpenAICompatibleLLMClient):
            raise TypeError(
                "generator diagnostics require an OpenAI-compatible client"
            )
        if controller_client is not None and not isinstance(
            controller_client, OpenAICompatibleLLMClient
        ):
            raise TypeError(
                "controller diagnostics require an OpenAI-compatible client"
            )
        if controller_client is generator_client:
            raise ValueError(
                "generator and controller diagnostics need distinct clients"
            )
        clients = {"generator": generator_client}
        if controller_client is not None:
            clients["controller"] = controller_client
        if any(client.attempt_observer is not None for client in clients.values()):
            raise ValueError("a diagnostic client already has an observer")

        generator_digest = llm_transport_sha256(generator_client)
        if collection_contract.generator_transport_sha256 != generator_digest:
            raise ValueError(
                "generator client differs from the collection contract"
            )
        controller_digest = collection_contract.controller_transport_sha256
        if controller_client is not None:
            if collection_contract.behavior_source != "teacher":
                raise ValueError(
                    "learned-policy collection cannot bind a controller client"
                )
            if llm_transport_sha256(controller_client) != controller_digest:
                raise ValueError(
                    "controller client differs from the collection contract"
                )

        self._environment_digest = environment.sha256
        self._collection_digest = collection_contract.sha256
        self._behavior_source = collection_contract.behavior_source
        self._generation_temperature = (
            environment.contract.generation_temperature
        )
        self._generation_max_tokens = (
            environment.contract.generation_max_tokens
        )
        self._generation_seed_policy = (
            environment.contract.generation_seed_policy
        )
        self._generation_seed = collection_contract.generation_seed
        self._controller_temperature = (
            collection_contract.controller_temperature
        )
        self._controller_max_tokens = (
            collection_contract.controller_max_tokens
        )
        self._controller_seed_policy = (
            collection_contract.controller_seed_policy
        )
        self._controller_seed = collection_contract.controller_seed
        self._clients = clients
        self._header = build_diagnostic_header(
            environment_contract_sha256=self._environment_digest,
            collection_contract_sha256=self._collection_digest,
            generator_transport_sha256=generator_digest,
            controller_transport_sha256=controller_digest,
            controller_decision_receipts=True,
            controller_attempts=controller_client is not None,
        )
        self._lock = threading.Lock()
        self._active: dict[tuple[str, str], _ActiveCall] = {}
        self._completed_calls: dict[str, dict[str, object]] = {}
        self._completed_attempts: dict[
            str, tuple[dict[str, object], ...]
        ] = {}
        self._decision_receipt_ids: set[str] = set()
        self._pending_generator_refs: ContextVar[
            tuple[DiagnosticRef, ...]
        ] = ContextVar(f"generator_diagnostic_refs_{id(self)}", default=())
        self._pending_controller_refs: ContextVar[
            tuple[DiagnosticRef, ...]
        ] = ContextVar(f"controller_diagnostic_refs_{id(self)}", default=())
        self._closed = False
        self._poisoned = False

        def observe_generator(event: OpenAICompatibleAttempt) -> None:
            self.observe_attempt("generator", event)

        def observe_controller(event: OpenAICompatibleAttempt) -> None:
            self.observe_attempt("controller", event)

        self._callbacks = {"generator": observe_generator}
        if controller_client is not None:
            self._callbacks["controller"] = observe_controller

        self._file: BinaryIO = Path(path).open("xb")
        try:
            self._file.write(canonical_json_bytes(self._header) + b"\n")
            self._file.flush()
        except Exception:
            self._file.close()
            self._closed = True
            self._poisoned = True
            raise
        for role, client in self._clients.items():
            client.attempt_observer = self._callbacks[role]

    @property
    def environment_contract_sha256(self) -> str:
        return self._environment_digest

    @property
    def collection_contract_sha256(self) -> str:
        return self._collection_digest

    @property
    def header(self) -> dict[str, object]:
        return json.loads(canonical_json_bytes(self._header))

    def _detach_observers_locked(self) -> None:
        for role, client in self._clients.items():
            if client.attempt_observer is self._callbacks[role]:
                client.attempt_observer = None

    def _write_locked(self, record: dict[str, object]) -> None:
        try:
            self._file.write(canonical_json_bytes(record) + b"\n")
            self._file.flush()
        except Exception:
            try:
                self._detach_observers_locked()
                self._file.close()
            finally:
                self._closed = True
                self._poisoned = True
            raise

    def _pending_for(
        self, role: str
    ) -> ContextVar[tuple[DiagnosticRef, ...]]:
        if role == "generator":
            return self._pending_generator_refs
        if role == "controller":
            return self._pending_controller_refs
        raise ValueError("diagnostic model role is invalid")

    def observe_attempt(
        self, role: str, event: OpenAICompatibleAttempt
    ) -> None:
        record = build_attempt_record(event, role=role)
        key = (role, event.call_id)
        with self._lock:
            if self._poisoned:
                raise RuntimeError(
                    "diagnostic writer is poisoned by an I/O failure"
                )
            if self._closed:
                raise RuntimeError("diagnostic writer is closed")
            active = self._active.setdefault(key, _ActiveCall())
            previous = active.attempts[-1] if active.attempts else None
            _validate_attempt_sequence(previous, record)
            self._write_locked(record)
            active.attempts.append(record)
            if event.disposition in {"return", "raise"}:
                call_record = build_call_record(
                    role, event.call_id, active.attempts
                )
                self._write_locked(call_record)
                del self._active[key]
                call_ref = DiagnosticRef(call_record["record_id"])
                self._completed_calls[call_ref.record_id] = call_record
                self._completed_attempts[call_ref.record_id] = tuple(
                    active.attempts
                )
                pending_var = self._pending_for(role)
                pending_var.set(pending_var.get() + (call_ref,))

    def take_call_ref(self, role: str) -> DiagnosticRef:
        pending_var = self._pending_for(role)
        pending = pending_var.get()
        if not pending:
            raise RuntimeError(f"no completed {role} diagnostic is pending")
        pending_var.set(pending[1:])
        return pending[0]

    def take_diagnostic_ref(self) -> DiagnosticRef:
        """Compatibility alias for generator-only transport tests."""
        return self.take_call_ref("generator")

    def _validate_role_call_locked(
        self,
        reference: DiagnosticRef,
        *,
        role: str,
        turn: int,
    ) -> dict[str, object]:
        call = self._completed_calls.get(reference.record_id)
        attempts = self._completed_attempts.get(reference.record_id)
        if call is None or attempts is None or call["role"] != role:
            raise DiagnosticValidationError(
                f"decision receipt {role} call does not resolve"
            )
        first = attempts[0]
        request = first["request_body"]
        if role == "generator":
            expected_temperature = self._generation_temperature
            expected_max_tokens = self._generation_max_tokens
            seed_policy = self._generation_seed_policy
            base_seed = self._generation_seed
        else:
            expected_temperature = self._controller_temperature
            expected_max_tokens = self._controller_max_tokens
            seed_policy = self._controller_seed_policy
            base_seed = self._controller_seed
        if request["temperature"] != expected_temperature:
            raise DiagnosticValidationError(
                f"{role} call temperature differs from its contract"
            )
        if request["max_tokens"] != expected_max_tokens:
            raise DiagnosticValidationError(
                f"{role} call token budget differs from its contract"
            )
        expected_seed = (
            None
            if seed_policy == "none-v1"
            else base_seed + turn
        )
        if first["effective_seed"] != expected_seed:
            raise DiagnosticValidationError(
                f"{role} call seed differs from its contract and turn"
            )
        return call

    def record_decision(
        self,
        *,
        episode_key: str,
        turn: int,
        observation_sha256: str,
        source: BehaviorSource,
        proposed_action: Action | None,
        selected_action: Action,
        fallback: bool,
    ) -> str:
        if not isinstance(source, BehaviorSource):
            raise TypeError("decision source has an invalid type")
        source_value = source.value
        if source_value != self._behavior_source:
            raise DiagnosticValidationError(
                "decision source differs from the collection contract"
            )
        generator_pending = self._pending_generator_refs.get()
        controller_pending = self._pending_controller_refs.get()
        if len(generator_pending) != 1:
            raise DiagnosticValidationError(
                "a decision needs exactly one pending generator call"
            )
        hosted_teacher = (
            source is BehaviorSource.TEACHER
            and "controller" in self._clients
        )
        expected_controller_calls = 1 if hosted_teacher else 0
        if len(controller_pending) != expected_controller_calls:
            raise DiagnosticValidationError(
                "decision has the wrong number of pending controller calls"
            )
        generator_ref = generator_pending[0]
        controller_ref = controller_pending[0] if controller_pending else None
        record_id = uuid.uuid4().hex
        record = build_decision_receipt(
            record_id=record_id,
            collection_contract_sha256=self._collection_digest,
            episode_key=episode_key,
            turn=turn,
            observation_sha256=observation_sha256,
            source=source_value,
            proposed_action=proposed_action,
            selected_action=selected_action,
            fallback=fallback,
            controller_call_ref=controller_ref,
            generator_call_ref=generator_ref,
        )
        with self._lock:
            if self._poisoned:
                raise RuntimeError(
                    "diagnostic writer is poisoned by an I/O failure"
                )
            if self._closed:
                raise RuntimeError("diagnostic writer is closed")
            self._validate_role_call_locked(
                generator_ref, role="generator", turn=turn
            )
            if controller_ref is not None:
                controller_call = self._validate_role_call_locked(
                    controller_ref, role="controller", turn=turn
                )
                if (
                    not fallback
                    and controller_call["final_outcome"]
                    != "transport_accepted"
                ):
                    raise DiagnosticValidationError(
                        "non-fallback teacher call was not accepted"
                    )
            if record_id in self._decision_receipt_ids:
                raise DiagnosticValidationError(
                    "decision receipt record_id is duplicated"
                )
            self._write_locked(record)
            self._decision_receipt_ids.add(record_id)
            self._pending_generator_refs.set(generator_pending[1:])
            self._pending_controller_refs.set(controller_pending[1:])
        return record_id

    def close(self) -> None:
        with self._lock:
            if self._active:
                raise RuntimeError(
                    "cannot close diagnostic writer with active calls"
                )
            if not self._closed:
                self._detach_observers_locked()
                self._file.close()
                self._closed = True

    def __enter__(self) -> DiagnosticSidecarWriter:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class ResolvedGeneratorDiagnostic:
    call_record: Mapping[str, object]
    attempt_records: tuple[Mapping[str, object], ...]


@dataclass(frozen=True, slots=True)
class ResolvedDecisionDiagnostic:
    receipt_record: Mapping[str, object]
    generator_diagnostic: ResolvedGeneratorDiagnostic
    controller_diagnostic: ResolvedGeneratorDiagnostic | None


def _turn_seed(
    policy: str, base_seed: int | None, turn: int, *, role: str
) -> int | None:
    if policy == "none-v1":
        if base_seed is not None:
            raise DiagnosticValidationError(
                f"{role} none seed policy carries a seed"
            )
        return None
    if type(base_seed) is not int:
        raise DiagnosticValidationError(
            f"{role} offset seed policy lacks a seed"
        )
    return base_seed + turn


def _validate_openai_transport_contract(
    transport: ModelTransportContract, *, role: str
) -> None:
    if transport.base_url is None or transport.model is None:
        raise DiagnosticValidationError(
            f"{role} raw attempts require a hosted transport"
        )
    expected = {
        "response_format_contract": "openai-json-object-v1",
        "parse_retry_contract": "assistant-echo-seed-offset-v1",
        "backoff_contract": "retry-after-exponential-jitter-v1",
    }
    for field_name, expected_value in expected.items():
        if getattr(transport, field_name) != expected_value:
            raise DiagnosticValidationError(
                f"{role} transport {field_name} cannot be replayed exactly"
            )


def _validate_attempt_response_projection(
    attempt: Mapping[str, object],
    *,
    role: str,
    schema: Mapping[str, object],
    schema_validation_mode: str,
) -> dict[str, Any] | None:
    """Derive recorded projections and outcome from the raw HTTP body."""
    outcome = attempt["outcome"]
    usage = attempt["usage"]
    if outcome in {"http_status_error", "transport_error"}:
        if (
            attempt["raw_content"] is not None
            or usage["reported"] is not False
            or usage["raw"] is not None
            or any(
                usage[field_name] is not None
                for field_name in (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                )
            )
        ):
            raise DiagnosticValidationError(
                f"{role} HTTP failure carries impossible content or usage"
            )
        return None

    status = attempt["http_status"]
    if type(status) is not int or not 200 <= status < 300:
        raise DiagnosticValidationError(
            f"{role} content outcome lacks a successful HTTP status"
        )
    raw_body = attempt["raw_response_body"]
    if not isinstance(raw_body, str):
        raise DiagnosticValidationError(
            f"{role} content outcome lacks its raw response body"
        )
    try:
        payload = json.loads(raw_body)
    except (json.JSONDecodeError, TypeError):
        payload = None

    projection = None
    if type(payload) is dict:
        projection = project_openai_compatible_response(payload)
    expected_content = (
        None if projection is None else projection.raw_content
    )
    expected_raw_usage = (
        None
        if projection is None or projection.raw_usage_json is None
        else json.loads(projection.raw_usage_json)
    )
    if attempt["raw_content"] != expected_content:
        raise DiagnosticValidationError(
            f"{role} raw content differs from its response envelope"
        )
    if canonical_json_bytes(usage["raw"]) != canonical_json_bytes(
        expected_raw_usage
    ):
        raise DiagnosticValidationError(
            f"{role} raw usage differs from its response envelope"
        )
    expected_counts = {
        "prompt_tokens": (
            None if projection is None else projection.prompt_tokens
        ),
        "completion_tokens": (
            None if projection is None else projection.completion_tokens
        ),
        "total_tokens": (
            None if projection is None else projection.total_tokens
        ),
    }
    if usage["reported"] is not (expected_raw_usage is not None):
        raise DiagnosticValidationError(
            f"{role} usage presence differs from its response envelope"
        )
    for field_name, expected_value in expected_counts.items():
        if usage[field_name] != expected_value:
            raise DiagnosticValidationError(
                f"{role} {field_name} differs from its response envelope"
            )

    decoded = None
    if projection is None or projection.provider_error is not None:
        derived_outcome = "provider_response_error"
    else:
        if projection.raw_content is None:
            raise AssertionError("provider projection lost its content error")
        derived_outcome, decoded = classify_openai_compatible_json_content(
            projection.raw_content,
            schema,
            schema_validation_mode=schema_validation_mode,
        )
    if outcome != derived_outcome:
        raise DiagnosticValidationError(
            f"{role} attempt outcome differs from its raw response"
        )
    return decoded


def _expected_attempt_disposition(
    attempt: Mapping[str, object],
    transport: ModelTransportContract,
) -> str:
    """Derive the exact disposition used by the live client."""
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


def _validate_bound_call_transcript(
    diagnostic: ResolvedGeneratorDiagnostic,
    *,
    role: str,
    transport: ModelTransportContract,
    system: str,
    user: str,
    schema: Mapping[str, object],
    temperature: float,
    max_tokens: int,
    seed_policy: str,
    base_seed: int | None,
    turn: int,
    schema_validation_mode: str,
) -> dict[str, Any] | None:
    """Rebuild every request body and decode an accepted terminal reply."""
    _validate_openai_transport_contract(transport, role=role)
    attempts = diagnostic.attempt_records
    if not attempts:
        raise DiagnosticValidationError(
            f"{role} call has no raw attempts"
        )
    initial_seed = _turn_seed(
        seed_policy, base_seed, turn, role=role
    )
    expected_body = build_openai_compatible_json_body(
        model=transport.model,
        system=system,
        user=user,
        schema=schema,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=initial_seed,
        guided_json=transport.guided_json,
    )
    final_decoded: dict[str, Any] | None = None
    for attempt in attempts:
        if attempt["parse_retry_index"] > transport.parse_retry_budget:
            raise DiagnosticValidationError(
                f"{role} call exceeds its parse retry budget"
            )
        if attempt["backoff_retry_index"] > transport.backoff_retry_budget:
            raise DiagnosticValidationError(
                f"{role} call exceeds its backoff retry budget"
            )
        if canonical_json_bytes(attempt["request_body"]) != (
            canonical_json_bytes(expected_body)
        ):
            raise DiagnosticValidationError(
                f"{role} request body differs from its typed contract"
            )
        decoded = _validate_attempt_response_projection(
            attempt,
            role=role,
            schema=schema,
            schema_validation_mode=schema_validation_mode,
        )
        expected_disposition = _expected_attempt_disposition(
            attempt, transport
        )
        if attempt["disposition"] != expected_disposition:
            raise DiagnosticValidationError(
                f"{role} attempt disposition differs from live retry policy"
            )
        if attempt is attempts[-1]:
            final_decoded = decoded
        disposition = attempt["disposition"]
        if disposition == "parse_retry":
            append_openai_compatible_parse_retry(
                expected_body, attempt["raw_content"]
            )
            if initial_seed is not None:
                expected_body["seed"] = (
                    initial_seed + attempt["parse_retry_index"] + 1
                )

    final = attempts[-1]
    if diagnostic.call_record["final_outcome"] != "transport_accepted":
        return None
    if final_decoded is None:
        raise DiagnosticValidationError(
            f"accepted {role} content lacks a decoded payload"
        )
    return final_decoded


def _target_obligation(
    observation: Mapping[str, object], action: Action
) -> GoalObligation:
    goals = observation["goals"]
    target = next(
        goal for goal in goals if goal["id"] == action.target
    )
    obligation = target["obligation"]
    return GoalObligation(
        variable=obligation["variable"],
        equation=obligation["equation"],
        domain=obligation["domain"],
        objective=GoalObjective(obligation["objective"]),
    )


def _canonicalize_policy_draft(
    observation: Mapping[str, object],
    action: Action,
    draft: PolicyDraft,
    environment: BoundEnvironment,
) -> PolicyDraft:
    """Reconstruct the draft retained by an applied learned action."""
    owner = _target_obligation(observation, action)
    contract = environment.generation_contract
    if action.verb is Verb.DECOMPOSE:
        if type(draft) is not DecompositionDraft:
            raise ValueError("decomposition draft has an invalid type")
        children = contract.validate_decomposition(
            owner, draft.connective, draft.child_obligations
        )
        if (
            type(children) is not tuple
            or len(children) != len(draft.child_statements)
            or any(
                not isinstance(child, GoalObligation)
                or child.objective is not GoalObjective.BRANCH
                for child in children
            )
        ):
            raise ValueError(
                "decomposition canonicalizer returned invalid children"
            )
        return DecompositionDraft(
            draft.connective, draft.child_statements, children
        )

    if action.verb is Verb.PROPOSE_RESULT:
        if type(draft) is not ResultDraft:
            raise ValueError("result draft has an invalid type")
        payload = contract.canonicalize_payload(
            owner,
            ClaimKind.RESULT,
            ResultPayload(draft.objective, draft.value),
        )
        if not isinstance(payload, ResultPayload):
            raise ValueError("result canonicalizer returned an invalid type")
        return ResultDraft(payload.objective, payload.value)

    if action.verb is Verb.PROPOSE_SOLUTION:
        if type(draft) is not SolutionDraft:
            raise ValueError("solution draft has an invalid type")
        payload = contract.canonicalize_payload(
            owner,
            ClaimKind.SOLUTION,
            SolutionPayload(draft.variable, draft.value),
        )
        if not isinstance(payload, SolutionPayload):
            raise ValueError("solution canonicalizer returned an invalid type")
        return SolutionDraft(payload.variable, payload.value)

    if action.verb is Verb.PROPOSE_REFUTATION:
        if type(draft) is not RefutationDraft:
            raise ValueError("refutation draft has an invalid type")
        payload = contract.canonicalize_payload(
            owner,
            ClaimKind.REFUTATION,
            RefutationPayload(
                draft.mode, draft.variable, draft.candidate
            ),
        )
        if not isinstance(payload, RefutationPayload):
            raise ValueError(
                "refutation canonicalizer returned an invalid type"
            )
        return RefutationDraft(
            payload.mode, payload.variable, payload.candidate
        )
    raise ValueError("generator action is not a learned policy verb")


@dataclass(frozen=True, slots=True)
class DecisionReceiptAssessment:
    resolved: ResolvedDecisionDiagnostic
    receipt_coverage: bool
    controller_attempt_coverage: bool
    controller_call_resolved: bool
    generator_request_bound: bool
    generator_effect_bound: bool
    controller_request_bound: bool
    controller_payload_bound: bool
    sufficient_for_bc: bool


@dataclass(frozen=True, slots=True, init=False)
class DiagnosticSidecarIndex:
    """Validated, detached snapshot of one canonical diagnostic sidecar."""

    _header: dict[str, object]
    _calls: dict[str, dict[str, object]]
    _attempts: dict[str, dict[str, object]]
    _decision_receipts: dict[str, dict[str, object]]

    @classmethod
    def _from_validated_records(
        cls,
        *,
        header: dict[str, object],
        calls: dict[str, dict[str, object]],
        attempts: dict[str, dict[str, object]],
        decision_receipts: dict[str, dict[str, object]],
        reader_token: object,
    ) -> DiagnosticSidecarIndex:
        if reader_token is not _INDEX_READER_TOKEN:
            raise TypeError(
                "diagnostic indexes must come from read_diagnostic_sidecar"
            )
        instance = object.__new__(cls)
        object.__setattr__(instance, "_header", copy.deepcopy(header))
        object.__setattr__(instance, "_calls", copy.deepcopy(calls))
        object.__setattr__(instance, "_attempts", copy.deepcopy(attempts))
        object.__setattr__(
            instance,
            "_decision_receipts",
            copy.deepcopy(decision_receipts),
        )
        return instance

    @property
    def header(self) -> Mapping[str, object]:
        return copy.deepcopy(self._header)

    @property
    def calls(self) -> Mapping[str, Mapping[str, object]]:
        return copy.deepcopy(self._calls)

    @property
    def attempts(self) -> Mapping[str, Mapping[str, object]]:
        return copy.deepcopy(self._attempts)

    @property
    def decision_receipts(self) -> Mapping[str, Mapping[str, object]]:
        return copy.deepcopy(self._decision_receipts)

    def resolve(
        self,
        reference: DiagnosticRef,
        *,
        role: str | None = None,
    ) -> ResolvedGeneratorDiagnostic:
        if not isinstance(reference, DiagnosticRef):
            raise TypeError("diagnostic reference has an invalid type")
        if role is not None and role not in _MODEL_ROLES:
            raise ValueError("diagnostic model role is invalid")
        try:
            call = self._calls[reference.record_id]
        except KeyError as exc:
            raise DiagnosticValidationError(
                "diagnostic reference does not resolve"
            ) from exc
        if role is not None and call["role"] != role:
            raise DiagnosticValidationError(
                f"diagnostic reference does not resolve to a {role} call"
            )
        attempt_records = tuple(
            copy.deepcopy(self._attempts[record_id])
            for record_id in call["attempt_record_ids"]
        )
        return ResolvedGeneratorDiagnostic(
            copy.deepcopy(call), attempt_records
        )

    def resolve_decision_receipt(
        self, reference: DiagnosticRef
    ) -> ResolvedDecisionDiagnostic:
        if not isinstance(reference, DiagnosticRef):
            raise TypeError("diagnostic reference has an invalid type")
        try:
            receipt = self._decision_receipts[reference.record_id]
        except KeyError as exc:
            raise DiagnosticValidationError(
                "diagnostic reference does not resolve to a decision receipt"
            ) from exc
        generator_ref = DiagnosticRef.from_record(receipt["generator_call_ref"])
        controller_ref = (
            None
            if receipt["controller_call_ref"] is None
            else DiagnosticRef.from_record(receipt["controller_call_ref"])
        )
        return ResolvedDecisionDiagnostic(
            copy.deepcopy(receipt),
            self.resolve(generator_ref, role="generator"),
            (
                None
                if controller_ref is None
                else self.resolve(controller_ref, role="controller")
            ),
        )

    def validate_decision_reference(
        self,
        reference: DiagnosticRef,
        *,
        environment: BoundEnvironment,
        collection_contract: CollectionContract,
        observation: Mapping[str, object],
        collection_contract_sha256: str,
        episode_key: str,
        turn: int,
        observation_sha256: str,
        selected_action: Action,
        source: str,
        fallback: bool,
        bc_eligible: bool,
        expected_policy_draft: PolicyDraft | None,
        expected_rejection_reason: RejectionReason | None,
        require_bc_trust: bool = False,
    ) -> DecisionReceiptAssessment:
        """Resolve and exact-check one transition-level decision receipt."""
        if not isinstance(environment, BoundEnvironment):
            raise TypeError("expected environment has an invalid type")
        if not isinstance(collection_contract, CollectionContract):
            raise TypeError(
                "expected collection_contract has an invalid type"
            )
        if not isinstance(observation, Mapping):
            raise TypeError("expected observation has an invalid type")
        _sha256(
            collection_contract_sha256,
            "expected collection_contract_sha256",
        )
        _nonempty_text(episode_key, "expected episode_key")
        _integer(turn, "expected turn")
        _sha256(observation_sha256, "expected observation_sha256")
        if not isinstance(selected_action, Action):
            raise TypeError("expected selected_action has an invalid type")
        if source not in {"teacher", "learned_policy"}:
            raise DiagnosticValidationError("expected source is invalid")
        _boolean(fallback, "expected fallback")
        _boolean(bc_eligible, "expected bc_eligible")
        if expected_policy_draft is not None and not isinstance(
            expected_policy_draft, PolicyDraft
        ):
            raise TypeError("expected_policy_draft has an invalid type")
        if expected_rejection_reason is not None and not isinstance(
            expected_rejection_reason, RejectionReason
        ):
            raise TypeError("expected_rejection_reason has an invalid type")
        expected_bc = source == "teacher" and not fallback
        if bc_eligible is not expected_bc:
            raise DiagnosticValidationError(
                "bc_eligible differs from source and fallback"
            )
        if collection_contract.sha256 != collection_contract_sha256:
            raise DiagnosticValidationError(
                "expected collection digest differs from its contract"
            )
        if collection_contract.behavior_source != source:
            raise DiagnosticValidationError(
                "decision source differs from its collection contract"
            )
        try:
            validate_collection_binding(
                environment.contract, collection_contract
            )
            computed_observation_sha256 = canonical_observation_sha256(
                observation,
                environment.generation_contract,
                environment.verifier,
                environment_contract_sha256=environment.sha256,
            )
        except (ObservationValidationError, TypeError, ValueError) as exc:
            raise DiagnosticValidationError(
                "decision observation or contract binding is invalid"
            ) from exc
        if computed_observation_sha256 != observation_sha256:
            raise DiagnosticValidationError(
                "expected observation digest differs from its observation"
            )
        if observation["turn"] != turn:
            raise DiagnosticValidationError(
                "decision turn differs from its observation"
            )
        validate_diagnostic_header(
            self._header,
            environment_contract_sha256=environment.sha256,
            collection_contract_sha256=collection_contract.sha256,
            generator_transport_sha256=(
                collection_contract.generator_transport_sha256
            ),
            controller_transport_sha256=(
                collection_contract.controller_transport_sha256
            ),
        )
        resolved = self.resolve_decision_receipt(reference)
        receipt = resolved.receipt_record
        expected = {
            "collection_contract_sha256": collection_contract_sha256,
            "episode_key": episode_key,
            "turn": turn,
            "observation_sha256": observation_sha256,
            "selected_action": serialize_action(selected_action),
            "source": source,
            "fallback": fallback,
        }
        for field_name, expected_value in expected.items():
            if receipt[field_name] != expected_value:
                raise DiagnosticValidationError(
                    f"decision receipt {field_name} differs from transition"
                )
        if (
            fallback
            and receipt["proposed_action"] is not None
            and receipt["proposed_action"] in observation["legal_pairs"]
        ):
            raise DiagnosticValidationError(
                "fallback receipt relabels a legal proposed action"
            )
        if (
            self._header["collection_contract_sha256"]
            != collection_contract_sha256
        ):
            raise DiagnosticValidationError(
                "diagnostic header collection differs from transition"
            )
        coverage = self._header["coverage"]
        receipt_coverage = (
            coverage["controller_decision_receipts"]
            == CONTROLLER_RECEIPT_COVERAGE
        )
        controller_attempt_coverage = (
            coverage["controller_attempts"]
            == CONTROLLER_ATTEMPT_COVERAGE
        )
        controller_call_resolved = (
            resolved.controller_diagnostic is not None
        )
        generator_request_bound = False
        generator_effect_bound = False
        known_generator_contract = (
            environment.contract.generation_schema
            in SUPPORTED_GENERATION_SCHEMA_VERSIONS
            and collection_contract.generator_implementation
            == _TYPED_GENERATOR_IMPLEMENTATION
        )
        if known_generator_contract:
            try:
                generator_request = build_policy_lens_request(
                    observation,
                    selected_action,
                    lens_contract=environment.contract.lens_contract,
                )
                generator_payload = _validate_bound_call_transcript(
                    resolved.generator_diagnostic,
                    role="generator",
                    transport=collection_contract.generator_transport,
                    system=generator_request.system,
                    user=generator_request.user,
                    schema=generator_request.schema,
                    temperature=(
                        environment.contract.generation_temperature
                    ),
                    max_tokens=(
                        environment.contract.generation_max_tokens
                    ),
                    seed_policy=(
                        environment.contract.generation_seed_policy
                    ),
                    base_seed=collection_contract.generation_seed,
                    turn=turn,
                    schema_validation_mode=(
                        schema_validation_mode_for_generation_schema(
                            environment.contract.generation_schema
                        )
                    ),
                )
            except (TypeError, ValueError) as exc:
                if isinstance(exc, DiagnosticValidationError):
                    raise
                raise DiagnosticValidationError(
                    "generator typed request cannot be reconstructed"
                ) from exc
            generator_request_bound = True
            if generator_payload is None:
                has_content = any(
                    attempt["raw_content"] is not None
                    for attempt in (
                        resolved.generator_diagnostic.attempt_records
                    )
                )
                derived_reason = (
                    RejectionReason.PAYLOAD_ERROR
                    if has_content
                    else RejectionReason.GENERATION_ERROR
                )
                if (
                    expected_policy_draft is not None
                    or expected_rejection_reason is not derived_reason
                ):
                    raise DiagnosticValidationError(
                        "generator transport failure differs from learned effect"
                    )
            else:
                try:
                    decoded_draft = decode_policy_payload(
                        observation, selected_action, generator_payload
                    )
                except ValueError as exc:
                    if (
                        expected_policy_draft is not None
                        or expected_rejection_reason
                        is not RejectionReason.PAYLOAD_ERROR
                    ):
                        raise DiagnosticValidationError(
                            "generator payload failure differs from learned effect"
                        ) from exc
                else:
                    try:
                        canonical_draft = _canonicalize_policy_draft(
                            observation,
                            selected_action,
                            decoded_draft,
                            environment,
                        )
                    except Exception:
                        canonical_draft = None
                    if expected_rejection_reason is None:
                        derived_draft = canonical_draft
                    elif (
                        expected_rejection_reason
                        is RejectionReason.ALGEBRA_REJECTED
                    ):
                        # Algebra rejection retains the pre-canonical draft.
                        derived_draft = decoded_draft
                    else:
                        derived_draft = None
                    if (
                        derived_draft is None
                        or derived_draft != expected_policy_draft
                    ):
                        raise DiagnosticValidationError(
                            "generator raw payload differs from learned effect"
                        )
            generator_effect_bound = True

        controller_request_bound = False
        controller_payload_bound = False
        known_controller_contract = (
            collection_contract.policy_input_contract
            == POLICY_INPUT_CONTRACT_VERSION
            and collection_contract.controller_output_contract
            in SUPPORTED_CONTROLLER_OUTPUT_CONTRACT_VERSIONS
            and collection_contract.controller_implementation
            == _TYPED_CONTROLLER_IMPLEMENTATION
        )
        if controller_call_resolved and known_controller_contract:
            try:
                controller_request = build_controller_request(
                    observation, environment
                )
                controller_payload = _validate_bound_call_transcript(
                    resolved.controller_diagnostic,
                    role="controller",
                    transport=collection_contract.controller_transport,
                    system=controller_request.system,
                    user=controller_request.user,
                    schema=controller_request.schema,
                    temperature=collection_contract.controller_temperature,
                    max_tokens=collection_contract.controller_max_tokens,
                    seed_policy=collection_contract.controller_seed_policy,
                    base_seed=collection_contract.controller_seed,
                    turn=turn,
                    schema_validation_mode=(
                        schema_validation_mode_for_controller_output(
                            collection_contract.controller_output_contract
                        )
                    ),
                )
                controller_request_bound = True
                proposed = (
                    None
                    if receipt["proposed_action"] is None
                    else deserialize_action(
                        receipt["proposed_action"],
                        "decision_receipt.proposed_action",
                    )
                )
                if controller_payload is None:
                    if proposed is not None:
                        raise DiagnosticValidationError(
                            "failed controller call proposed an action"
                        )
                else:
                    decoded = decode_controller_payload(controller_payload)
                    if decoded != proposed:
                        raise DiagnosticValidationError(
                            "controller raw payload differs from proposed action"
                        )
                    controller_payload_bound = True
            except (
                ControllerPayloadError,
                RecordValidationError,
                TypeError,
                ValueError,
            ) as exc:
                if isinstance(exc, DiagnosticValidationError):
                    raise
                raise DiagnosticValidationError(
                    "controller typed request or payload cannot be linked"
                ) from exc
        controller_proven = (
            controller_attempt_coverage
            and controller_call_resolved
            and controller_request_bound
            and controller_payload_bound
        )
        sufficient_for_bc = (
            bc_eligible
            and receipt_coverage
            and generator_request_bound
            and generator_effect_bound
            and controller_proven
        )
        assessment = DecisionReceiptAssessment(
            resolved=resolved,
            receipt_coverage=receipt_coverage,
            controller_attempt_coverage=controller_attempt_coverage,
            controller_call_resolved=controller_call_resolved,
            generator_request_bound=generator_request_bound,
            generator_effect_bound=generator_effect_bound,
            controller_request_bound=controller_request_bound,
            controller_payload_bound=controller_payload_bound,
            sufficient_for_bc=sufficient_for_bc,
        )
        if require_bc_trust and bc_eligible and not sufficient_for_bc:
            raise DiagnosticValidationError(
                "BC-eligible decision lacks fully linked model diagnostics"
            )
        return assessment


def read_diagnostic_sidecar(
    path: str | Path,
    *,
    environment_contract_sha256: str | None = None,
    collection_contract_sha256: str | None = None,
    generator_transport_sha256: str | None = None,
    controller_transport_sha256: str | None = None,
) -> DiagnosticSidecarIndex:
    with Path(path).open("rb") as handle:
        lines = list(handle)
    if not lines:
        raise DiagnosticValidationError("diagnostic sidecar is empty")
    try:
        header = parse_canonical_json_line(lines[0])
    except RecordValidationError as exc:
        raise DiagnosticValidationError(
            f"diagnostic header encoding is invalid: {exc}"
        ) from exc
    validate_diagnostic_header(
        header,
        environment_contract_sha256=environment_contract_sha256,
        collection_contract_sha256=collection_contract_sha256,
        generator_transport_sha256=generator_transport_sha256,
        controller_transport_sha256=controller_transport_sha256,
    )
    active: dict[tuple[str, str], list[dict[str, object]]] = {}
    calls: dict[str, dict[str, object]] = {}
    attempts: dict[str, dict[str, object]] = {}
    decision_receipts: dict[str, dict[str, object]] = {}
    consumed_generator_calls: set[str] = set()
    consumed_controller_calls: set[str] = set()
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines[1:], start=2):
        try:
            record = parse_canonical_json_line(line)
        except RecordValidationError as exc:
            raise DiagnosticValidationError(
                f"diagnostic line {line_number} encoding is invalid: {exc}"
            ) from exc
        schema = record.get("schema_version")
        if schema == DIAGNOSTIC_ATTEMPT_VERSION:
            validate_attempt_record(record)
            role = record["role"]
            if (
                role == "controller"
                and header["coverage"]["controller_attempts"]
                != CONTROLLER_ATTEMPT_COVERAGE
            ):
                raise DiagnosticValidationError(
                    "sidecar contains unannounced controller attempts"
                )
            call_id = record["call_id"]
            call_record_id = f"{role}:{call_id}"
            if call_record_id in calls:
                raise DiagnosticValidationError(
                    "attempt appears after its terminal call record"
                )
            previous_group = active.setdefault((role, call_id), [])
            previous = previous_group[-1] if previous_group else None
            _validate_attempt_sequence(previous, record)
            record_id = record["record_id"]
            if record_id in seen_ids:
                raise DiagnosticValidationError(
                    "diagnostic record_id is duplicated"
                )
            seen_ids.add(record_id)
            attempts[record_id] = record
            previous_group.append(record)
        elif schema == DIAGNOSTIC_CALL_VERSION:
            role = record.get("role")
            if (
                role == "controller"
                and header["coverage"]["controller_attempts"]
                != CONTROLLER_ATTEMPT_COVERAGE
            ):
                raise DiagnosticValidationError(
                    "sidecar contains an unannounced controller call"
                )
            call_id = record.get("call_id")
            group = active.get((role, call_id))
            if group is None:
                raise DiagnosticValidationError(
                    "call record has no preceding attempts"
                )
            validate_call_record(record, attempts=group)
            record_id = record["record_id"]
            if record_id in seen_ids:
                raise DiagnosticValidationError(
                    "diagnostic record_id is duplicated"
                )
            seen_ids.add(record_id)
            calls[record_id] = record
            del active[(role, call_id)]
        elif schema == DECISION_RECEIPT_VERSION:
            validate_decision_receipt(record)
            coverage = header["coverage"]
            if (
                coverage["controller_decision_receipts"]
                != CONTROLLER_RECEIPT_COVERAGE
            ):
                raise DiagnosticValidationError(
                    "sidecar contains unannounced decision receipts"
                )
            if (
                record["collection_contract_sha256"]
                != header["collection_contract_sha256"]
            ):
                raise DiagnosticValidationError(
                    "decision receipt collection digest differs from header"
                )
            generator_ref = DiagnosticRef.from_record(
                record["generator_call_ref"]
            )
            if generator_ref.record_id not in calls:
                raise DiagnosticValidationError(
                    "decision receipt generator call does not resolve"
                )
            if calls[generator_ref.record_id]["role"] != "generator":
                raise DiagnosticValidationError(
                    "decision receipt generator ref has the wrong role"
                )
            if generator_ref.record_id in consumed_generator_calls:
                raise DiagnosticValidationError(
                    "generator call is reused by multiple decision receipts"
                )
            controller_ref = (
                None
                if record["controller_call_ref"] is None
                else DiagnosticRef.from_record(record["controller_call_ref"])
            )
            controller_coverage = header["coverage"]["controller_attempts"]
            if controller_ref is None:
                if controller_coverage == CONTROLLER_ATTEMPT_COVERAGE:
                    raise DiagnosticValidationError(
                        "hosted controller receipt lacks its call reference"
                    )
            elif controller_coverage != CONTROLLER_ATTEMPT_COVERAGE:
                raise DiagnosticValidationError(
                    "controller call reference differs from sidecar coverage"
                )
            elif (
                controller_ref.record_id not in calls
                or calls[controller_ref.record_id]["role"] != "controller"
            ):
                raise DiagnosticValidationError(
                    "decision receipt controller call does not resolve"
                )
            elif (
                record["fallback"] is False
                and calls[controller_ref.record_id]["final_outcome"]
                != "transport_accepted"
            ):
                raise DiagnosticValidationError(
                    "non-fallback receipt cites a rejected controller call"
                )
            if (
                controller_ref is not None
                and controller_ref.record_id in consumed_controller_calls
            ):
                raise DiagnosticValidationError(
                    "controller call is reused by multiple decision receipts"
                )
            record_id = record["record_id"]
            if record_id in seen_ids:
                raise DiagnosticValidationError(
                    "diagnostic record_id is duplicated"
                )
            seen_ids.add(record_id)
            decision_receipts[record_id] = record
            consumed_generator_calls.add(generator_ref.record_id)
            if controller_ref is not None:
                consumed_controller_calls.add(controller_ref.record_id)
        else:
            raise DiagnosticValidationError(
                f"diagnostic line {line_number} has an unknown schema"
            )
    if active:
        raise DiagnosticValidationError(
            "diagnostic sidecar ends with incomplete calls"
        )
    return DiagnosticSidecarIndex._from_validated_records(
        header=header,
        calls=calls,
        attempts=attempts,
        decision_receipts=decision_receipts,
        reader_token=_INDEX_READER_TOKEN,
    )


__all__ = [
    "CONTROLLER_ATTEMPT_COVERAGE",
    "CONTROLLER_RECEIPT_COVERAGE",
    "DECISION_RECEIPT_VERSION",
    "DIAGNOSTIC_ATTEMPT_VERSION",
    "DIAGNOSTIC_CALL_VERSION",
    "DIAGNOSTIC_COVERAGE_UNAVAILABLE",
    "DIAGNOSTIC_HEADER_VERSION",
    "DecisionReceiptAssessment",
    "DiagnosticSidecarIndex",
    "DiagnosticSidecarWriter",
    "DiagnosticValidationError",
    "GENERATOR_DIAGNOSTIC_COVERAGE",
    "ResolvedDecisionDiagnostic",
    "ResolvedGeneratorDiagnostic",
    "build_attempt_record",
    "build_call_record",
    "build_decision_receipt",
    "build_diagnostic_header",
    "read_diagnostic_sidecar",
    "require_bc_ready_decision_receipt",
    "require_hosted_bc_diagnostic_coverage",
    "validate_attempt_record",
    "validate_call_record",
    "validate_decision_receipt",
    "validate_diagnostic_header",
]
