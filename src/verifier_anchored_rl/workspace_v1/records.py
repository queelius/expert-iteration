"""Canonical secret-free records for workspace v1 experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from .controller import TeacherDecision
from .lenses import build_observation
from .model import ACTION_AVAILABILITY_ALL, Action
from .runtime import Transition


RUN_HEADER_VERSION = "workspace-run-header-v1"
TRANSITION_RECORD_VERSION = "workspace-transition-record-v1"
EPISODE_SUMMARY_VERSION = "workspace-episode-summary-v1"
RUN_SUMMARY_VERSION = "workspace-run-summary-v1"
DIAGNOSTIC_HEADER_VERSION = "workspace-diagnostic-header-v1"
DIAGNOSTIC_ATTEMPT_VERSION = "workspace-http-attempt-v1"


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def action_record(action: Action | None) -> dict[str, object] | None:
    if action is None:
        return None
    if not isinstance(action, Action):
        raise TypeError("action record requires Action or None")
    return {"verb": action.verb.value, "target": action.target}


def teacher_decision_record(
    decision: TeacherDecision | None,
) -> dict[str, object] | None:
    if decision is None:
        return None
    if not isinstance(decision, TeacherDecision):
        raise TypeError("teacher decision record has an invalid type")
    return {
        "turn": decision.turn,
        "proposed_action": action_record(decision.proposed_action),
        "selected_action": action_record(decision.selected_action),
        "fallback": decision.fallback,
        "observation": json.loads(decision.observation_json),
        "request": {
            "system": decision.request.system,
            "user": decision.request.user,
            "schema": dict(decision.request.schema),
        },
        "payload": (
            None
            if decision.payload_json is None
            else json.loads(decision.payload_json)
        ),
    }


def transition_record(
    transition: Transition,
    *,
    experiment_id: str,
    arm: str,
    episode_index: int,
    problem_id: str,
    max_depth: int,
    worker_call_id: str | None,
    teacher_call_id: str | None,
    teacher_decision: TeacherDecision | None,
    action_availability_mode: str = ACTION_AVAILABILITY_ALL,
) -> dict[str, object]:
    if not isinstance(transition, Transition):
        raise TypeError("transition record requires a Transition")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment id must be nonempty")
    if not isinstance(arm, str) or not arm.strip():
        raise ValueError("experiment arm must be nonempty")
    if type(episode_index) is not int or episode_index < 0:
        raise ValueError("episode index must be nonnegative")
    if not isinstance(problem_id, str) or not problem_id.strip():
        raise ValueError("problem id must be nonempty")
    request = transition.worker_request
    receipt = transition.submit_receipt
    record = {
        "schema_version": TRANSITION_RECORD_VERSION,
        "experiment_id": experiment_id,
        "arm": arm,
        "episode_index": episode_index,
        "problem_id": problem_id,
        "turn": transition.state.turn,
        "observation": build_observation(
            transition.state,
            max_depth=max_depth,
            action_availability_mode=action_availability_mode,
        ),
        "action": action_record(transition.action),
        "teacher_decision": teacher_decision_record(teacher_decision),
        "teacher_call_id": teacher_call_id,
        "worker_request": (
            None
            if request is None
            else {
                "system": request.system,
                "user": request.user,
                "schema": dict(request.schema),
                "schema_validation_mode": request.schema_validation_mode,
            }
        ),
        "worker_call_id": worker_call_id,
        "worker_payload": (
            None
            if transition.worker_payload_json is None
            else json.loads(transition.worker_payload_json)
        ),
        "generation_error": transition.generation_error,
        "submit_receipt": None if receipt is None else asdict(receipt),
        "reward": transition.reward,
        "next_observation": build_observation(
            transition.next_state,
            max_depth=max_depth,
            action_availability_mode=action_availability_mode,
        ),
        "terminal": transition.terminal,
    }
    encoded = canonical_json_bytes(record)
    if b"expected_answer" in encoded:
        raise ValueError("transition record leaked the hidden expected answer")
    return record


__all__ = [
    "DIAGNOSTIC_ATTEMPT_VERSION",
    "DIAGNOSTIC_HEADER_VERSION",
    "EPISODE_SUMMARY_VERSION",
    "RUN_HEADER_VERSION",
    "RUN_SUMMARY_VERSION",
    "TRANSITION_RECORD_VERSION",
    "action_record",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "teacher_decision_record",
    "transition_record",
]
