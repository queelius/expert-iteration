"""Canonical observations and worker prompts for workspace v1."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from ..llm_client import SCHEMA_VALIDATION_STRICT_V1
from .model import (
    ACTION_AVAILABILITY_ALL,
    STATE_SCHEMA_VERSION,
    Action,
    ActionEvent,
    CandidateNode,
    GoalNode,
    State,
    Verb,
    actions_for_mode,
    child_goals,
    goal_candidates,
    node_at,
    protected_direct_candidate_id,
    validate_action_availability_mode,
)


OBSERVATION_SCHEMA_VERSION = "workspace-observation-v1"
ACTIVE_FRONTIER_OBSERVATION_SCHEMA_VERSION = (
    "workspace-observation-active-frontier-v1"
)
HYBRID_LENS_VERSION = "workspace-hybrid-local-lens-v1"
CHILD_CONTEXT_CONTRACT_VERSION = "workspace-child-answer-context-v1"
CHILD_CONTEXT_TARGET_ONLY = "target-only-v1"
CHILD_CONTEXT_PRECEDING_SIBLINGS = "preceding-siblings-v1"
CHILD_CONTEXT_MODES = frozenset({
    CHILD_CONTEXT_TARGET_ONLY,
    CHILD_CONTEXT_PRECEDING_SIBLINGS,
})
TEACHER_PRESENTATION_VERSION = "workspace-teacher-presentation-v1"
DIRECT_PROMPT_VERSION = "direct-math-json-v1"
WORKER_PROMPT_VERSION = "workspace-worker-prompts-v1.1"
ROOT_RESULT_CONTRACT_VERSION = "workspace-root-result-contract-v1"
ROOT_RESULT_FREE = "workspace-answer-free-v1"
ROOT_RESULT_VERIFIER_READY = "verifier-ready-root-answer-v1"
ROOT_RESULT_MODES = frozenset({
    ROOT_RESULT_FREE,
    ROOT_RESULT_VERIFIER_READY,
})
WORKER_PROMPT_ROOT_RESULT_VERSION = "workspace-worker-prompts-v1.2"
REVISE_RESULT_CONTRACT_VERSION = "workspace-revise-result-contract-v1"
REVISE_RESULT_OPEN_OBJECT = "workspace-revise-open-object-v1"
REVISE_RESULT_ANSWER_CONTAINER = "workspace-revise-answer-container-v1"
REVISE_RESULT_INDEPENDENT_ANSWER_CONTAINER = (
    "workspace-revise-independent-answer-container-v1"
)
REVISE_RESULT_MODES = frozenset({
    REVISE_RESULT_OPEN_OBJECT,
    REVISE_RESULT_ANSWER_CONTAINER,
    REVISE_RESULT_INDEPENDENT_ANSWER_CONTAINER,
})
WORKER_PROMPT_REVISE_ANSWER_VERSION = (
    "workspace-worker-revise-answer-container-v1"
)
WORKER_PROMPT_REVISE_INDEPENDENT_VERSION = (
    "workspace-worker-revise-independent-answer-container-v1"
)
ROOT_RESULT_VERIFIER_READY_INSTRUCTION = (
    " The answer field is the final answer to the root problem and must be "
    "directly checkable by its symbolic verifier. Put only the requested "
    "scalar or symbolic expression in answer. Put variable assignments, "
    "equations, units, prose, code fences, and alternative lists in "
    "reasoning unless the root problem explicitly requests that form. Do "
    "not change the mathematical answer merely to satisfy this format."
)

DIRECT_SYSTEM = (
    "You are a careful mathematical problem solver. Work through the "
    "problem step by step, then state the final numeric answer."
)
DIRECT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "reasoning": {},
        "answer": {},
    },
    "required": ["answer"],
}

DECOMPOSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "children": {
            "type": "array",
            "items": {},
            "minItems": 2,
            "maxItems": 5,
        }
    },
    "required": ["children"],
}

WORKSPACE_CANDIDATE_SCHEMA: dict[str, object] = {
    "type": "object",
}

REVISE_ANSWER_CONTAINER_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "reasoning": {},
        "answer": {"type": ["string", "number"]},
    },
    "required": ["answer"],
    "additionalProperties": False,
}

REVISE_OPEN_OBJECT_INSTRUCTION = (
    "Revise the selected Candidate for its Goal using the visible local work. "
    "Return improved reasoning plus an answer field."
)
REVISE_ANSWER_CONTAINER_INSTRUCTION = (
    "Revise the selected Candidate for its Goal using the visible local work. "
    "Return exactly one answer-bearing Candidate JSON object with this shape: "
    '{"reasoning":"your revised solution","answer":"your revised answer"}. '
    "The top-level answer must be a nonempty string or number. Do not return "
    "or copy the workspace lens, action, target, target_candidates, or any "
    "wrapper object."
)
REVISE_INDEPENDENT_ANSWER_CONTAINER_INSTRUCTION = (
    "Independently solve the selected Candidate's Goal from the problem "
    "conditions. Treat the selected Candidate's reasoning and answer as "
    "untrusted context: derive the result yourself before comparing it with "
    "the visible work, and do not copy an answer merely because it is already "
    "present. Return exactly one answer-bearing Candidate JSON object with "
    "this shape: "
    '{"reasoning":"your independent solution","answer":"your answer"}. '
    "The top-level answer must be a nonempty string or number. Do not return "
    "or copy the workspace lens, action, target, target_candidates, or any "
    "wrapper object."
)

_WORKER_SYSTEM = (
    "You are the frozen worker in an external reasoning workspace. Execute "
    "only the requested operation on the selected target. Your response is "
    "unverified workspace content. Do not choose the next action, invoke a "
    "verifier, assign reward, or claim that the workspace is correct."
)


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    system: str
    user: str
    schema: Mapping[str, object]
    schema_validation_mode: str = SCHEMA_VALIDATION_STRICT_V1

    def __post_init__(self) -> None:
        if not isinstance(self.system, str) or not self.system.strip():
            raise ValueError("worker system prompt must be nonempty")
        if not isinstance(self.user, str) or not self.user.strip():
            raise ValueError("worker user prompt must be nonempty")
        if not isinstance(self.schema, Mapping):
            raise ValueError("worker schema has an invalid type")


def direct_user_prompt(problem: str) -> str:
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("direct problem must be nonempty text")
    return (
        f"Problem: {problem}\n\n"
        "Produce a JSON object with these fields:\n"
        '  - "reasoning": your step-by-step solution\n'
        '  - "answer": the final numeric answer only (just the number)\n'
        "Return only the JSON object."
    )


def direct_request(problem: str) -> WorkerRequest:
    return WorkerRequest(
        system=DIRECT_SYSTEM,
        user=direct_user_prompt(problem),
        schema=json.loads(_canonical_json(DIRECT_SCHEMA)),
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _action_record(action: Action) -> dict[str, object]:
    return {"verb": action.verb.value, "target": action.target}


def _event_record(event: ActionEvent) -> dict[str, object]:
    return {
        "turn": event.turn,
        "action": _action_record(event.action),
        "outcome": event.outcome.value,
        "created_ids": list(event.created_ids),
    }


def _goal_record(state: State, goal: GoalNode) -> dict[str, object]:
    return {
        "id": goal.id,
        "kind": "goal",
        "parent_goal": goal.parent_goal,
        "task": goal.task,
        "created_turn": goal.created_turn,
        "child_goals": [child.id for child in child_goals(state, goal.id)],
        "candidates": [
            candidate.id for candidate in goal_candidates(state, goal.id)
        ],
    }


def _candidate_record(candidate: CandidateNode) -> dict[str, object]:
    return {
        "id": candidate.id,
        "kind": "candidate",
        "goal_id": candidate.goal_id,
        "reasoning": candidate.reasoning,
        "answer": candidate.answer,
        "created_turn": candidate.created_turn,
        "revises": candidate.revises,
    }


def build_observation(
    state: State,
    *,
    max_depth: int,
    action_availability_mode: str = ACTION_AVAILABILITY_ALL,
) -> dict[str, object]:
    """Build the full controller-visible workspace observation."""
    if not isinstance(state, State):
        raise TypeError("observation requires a State")
    validate_action_availability_mode(action_availability_mode)
    nodes: list[dict[str, object]] = []
    for node in state.nodes:
        if isinstance(node, GoalNode):
            nodes.append(_goal_record(state, node))
        else:
            nodes.append(_candidate_record(node))
    observation = {
        "schema_version": (
            OBSERVATION_SCHEMA_VERSION
            if action_availability_mode == ACTION_AVAILABILITY_ALL
            else ACTIVE_FRONTIER_OBSERVATION_SCHEMA_VERSION
        ),
        "state_schema": STATE_SCHEMA_VERSION,
        "root_goal_id": 0,
        "turn": state.turn,
        "turn_limit": state.turn_limit,
        "remaining_turns": state.turn_limit - state.turn,
        "terminal": state.terminal,
        "reward": state.reward,
        "nodes": nodes,
        "history": [_event_record(event) for event in state.history],
        "legal_actions": [
            _action_record(action)
            for action in actions_for_mode(
                state,
                max_depth=max_depth,
                mode=action_availability_mode,
            )
        ],
    }
    if action_availability_mode != ACTION_AVAILABILITY_ALL:
        observation["protected_direct_candidate_id"] = (
            protected_direct_candidate_id(state)
        )
    return observation


def _ancestry(state: State, goal_id: int) -> list[dict[str, object]]:
    goals: list[GoalNode] = []
    cursor = node_at(state, goal_id)
    if not isinstance(cursor, GoalNode):
        raise TypeError("lens ancestry target must be a Goal")
    while True:
        goals.append(cursor)
        if cursor.parent_goal is None:
            break
        parent = node_at(state, cursor.parent_goal)
        if not isinstance(parent, GoalNode):
            raise AssertionError("validated ancestry changed type")
        cursor = parent
    goals.reverse()
    return [_goal_record(state, goal) for goal in goals]


def _attempts(state: State, action: Action) -> list[dict[str, object]]:
    return [
        _event_record(event)
        for event in state.history
        if event.action == action
    ]


def validate_child_context_mode(mode: str) -> str:
    if mode not in CHILD_CONTEXT_MODES:
        raise ValueError("unknown child context mode")
    return mode


def validate_root_result_mode(mode: str) -> str:
    if mode not in ROOT_RESULT_MODES:
        raise ValueError("unknown root result mode")
    return mode


def validate_revise_result_mode(mode: str) -> str:
    if mode not in REVISE_RESULT_MODES:
        raise ValueError("unknown revise result mode")
    return mode


def _completed_preceding_sibling_work(
    state: State,
    goal: GoalNode,
) -> list[dict[str, object]]:
    if goal.parent_goal is None:
        return []
    result: list[dict[str, object]] = []
    for sibling in child_goals(state, goal.parent_goal):
        if sibling.id == goal.id:
            break
        candidates = goal_candidates(state, sibling.id)
        if candidates:
            result.append({
                "goal": _goal_record(state, sibling),
                "candidates": [
                    _candidate_record(candidate) for candidate in candidates
                ],
            })
    return result


def build_hybrid_lens(
    state: State,
    action: Action,
    *,
    child_context_mode: str = CHILD_CONTEXT_TARGET_ONLY,
) -> dict[str, object]:
    """Project state to the selected target without semantic filtering."""
    if not isinstance(state, State) or not isinstance(action, Action):
        raise TypeError("hybrid lens requires State and Action")
    validate_child_context_mode(child_context_mode)
    root = node_at(state, 0)
    if not isinstance(root, GoalNode):
        raise AssertionError("validated root changed type")

    target = node_at(state, action.target)
    if action.verb is Verb.REVISE:
        if not isinstance(target, CandidateNode):
            raise ValueError("REVISE lens target must be a Candidate")
        goal = node_at(state, target.goal_id)
    else:
        if not isinstance(target, GoalNode):
            raise ValueError("worker Goal action has a non-Goal target")
        goal = target
    if not isinstance(goal, GoalNode):
        raise AssertionError("validated candidate owner changed type")

    children = child_goals(state, goal.id)
    child_work = [
        {
            "goal": _goal_record(state, child),
            "candidates": [
                _candidate_record(candidate)
                for candidate in goal_candidates(state, child.id)
            ],
        }
        for child in children
    ]
    ancestry = _ancestry(state, goal.id)
    if (
        child_context_mode == CHILD_CONTEXT_PRECEDING_SIBLINGS
        and action.verb is Verb.ANSWER
        and goal.parent_goal is not None
    ):
        preceding_work = _completed_preceding_sibling_work(state, goal)
        if preceding_work:
            parent_record = next(
                record
                for record in ancestry
                if record["id"] == goal.parent_goal
            )
            parent_record["completed_preceding_child_work"] = preceding_work
    lens = {
        "schema_version": HYBRID_LENS_VERSION,
        "action": _action_record(action),
        "root_problem": root.task,
        "target": (
            _goal_record(state, target)
            if isinstance(target, GoalNode)
            else _candidate_record(target)
        ),
        "target_goal": _goal_record(state, goal),
        "ancestry": ancestry,
        "target_candidates": [
            _candidate_record(candidate)
            for candidate in goal_candidates(state, goal.id)
        ],
        "child_work": child_work,
        "previous_exact_attempts": _attempts(state, action),
        "turn": state.turn,
        "turn_limit": state.turn_limit,
        "remaining_turns": state.turn_limit - state.turn,
    }
    return lens


def _workspace_request(
    state: State,
    action: Action,
    instruction: str,
    schema: Mapping[str, object],
    *,
    child_context_mode: str,
) -> WorkerRequest:
    lens = _canonical_json(build_hybrid_lens(
        state,
        action,
        child_context_mode=child_context_mode,
    ))
    return WorkerRequest(
        system=_WORKER_SYSTEM,
        user=(
            f"Operation: {action.verb.name}\n"
            f"Target: {action.target}\n"
            f"Hybrid workspace lens:\n{lens}\n\n"
            f"{instruction}\nReturn only one JSON object."
        ),
        schema=json.loads(_canonical_json(schema)),
    )


def build_worker_request(
    state: State,
    action: Action,
    *,
    child_context_mode: str = CHILD_CONTEXT_TARGET_ONLY,
    root_result_mode: str = ROOT_RESULT_FREE,
    revise_result_mode: str = REVISE_RESULT_OPEN_OBJECT,
) -> WorkerRequest:
    """Build the exact worker call for one legal generative action."""
    if not isinstance(state, State) or not isinstance(action, Action):
        raise TypeError("worker request requires State and Action")
    validate_child_context_mode(child_context_mode)
    validate_root_result_mode(root_result_mode)
    validate_revise_result_mode(revise_result_mode)
    if action.verb is Verb.SUBMIT:
        raise ValueError("SUBMIT never builds an LLM request")
    if (
        action == Action(Verb.ANSWER, 0)
        and state.turn == 0
        and len(state.nodes) == 1
    ):
        root = node_at(state, 0)
        if not isinstance(root, GoalNode):
            raise AssertionError("validated root changed type")
        return direct_request(root.task)
    if action.verb is Verb.DECOMPOSE:
        return _workspace_request(
            state,
            action,
            (
                "Break the target task into 2 to 5 useful child tasks. Each "
                "child should be understandable as a focused task when shown "
                "with the root problem and its ancestry. Return a children "
                "array. Child items may be strings or objects containing a "
                "task field. Do not solve or verify the children."
            ),
            DECOMPOSE_SCHEMA,
            child_context_mode=child_context_mode,
        )
    if action.verb is Verb.ANSWER:
        instruction = (
            "Answer the target Goal. Use the root and ancestry only as "
            "context. Return reasoning plus an answer field containing the "
            "target task's answer."
        )
    elif action.verb is Verb.SYNTHESIZE:
        instruction = (
            "Synthesize a candidate answer for the target Goal from its "
            "child tasks and candidate work. Resolve conflicts yourself. "
            "Return reasoning plus an answer field."
        )
    elif action.verb is Verb.REVISE:
        instruction = {
            REVISE_RESULT_OPEN_OBJECT: REVISE_OPEN_OBJECT_INSTRUCTION,
            REVISE_RESULT_ANSWER_CONTAINER: (
                REVISE_ANSWER_CONTAINER_INSTRUCTION
            ),
            REVISE_RESULT_INDEPENDENT_ANSWER_CONTAINER: (
                REVISE_INDEPENDENT_ANSWER_CONTAINER_INSTRUCTION
            ),
        }[revise_result_mode]
    else:
        raise ValueError("worker action verb is unsupported")
    target = node_at(state, action.target)
    goal_id = (
        target.goal_id
        if isinstance(target, CandidateNode)
        else target.id
    )
    if root_result_mode == ROOT_RESULT_VERIFIER_READY and goal_id == 0:
        instruction += ROOT_RESULT_VERIFIER_READY_INSTRUCTION
    return _workspace_request(
        state,
        action,
        instruction,
        (
            REVISE_ANSWER_CONTAINER_SCHEMA
            if action.verb is Verb.REVISE
            and revise_result_mode
            in {
                REVISE_RESULT_ANSWER_CONTAINER,
                REVISE_RESULT_INDEPENDENT_ANSWER_CONTAINER,
            }
            else WORKSPACE_CANDIDATE_SCHEMA
        ),
        child_context_mode=child_context_mode,
    )


def normalize_json_text(value: Any) -> str:
    """Preserve arbitrary JSON model output as deterministic workspace text."""
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return "null"
    return _canonical_json(value)


def decode_children(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Decode only the bounded container, never mathematical semantics."""
    if not isinstance(payload, Mapping):
        raise ValueError("DECOMPOSE payload must be an object")
    raw = payload.get("children")
    if type(raw) is not list or not 2 <= len(raw) <= 5:
        raise ValueError("DECOMPOSE needs 2 to 5 child items")
    children: list[str] = []
    for item in raw:
        if isinstance(item, Mapping):
            preferred = item.get("task", item.get("text"))
            text = (
                normalize_json_text(preferred)
                if preferred is not None
                else normalize_json_text(dict(item))
            )
        else:
            text = normalize_json_text(item)
        if len(text) > 4_000:
            raise ValueError("DECOMPOSE child exceeds the text bound")
        children.append(text or "[empty generated child]")
    return tuple(children)


def decode_candidate(payload: Mapping[str, Any]) -> tuple[str, str]:
    """Decode answer content without deciding whether it is correct."""
    if not isinstance(payload, Mapping):
        raise ValueError("candidate payload must be an object")
    candidate: Mapping[str, Any] = payload
    if "answer" not in candidate:
        nested = candidate.get("candidate")
        if isinstance(nested, Mapping):
            candidate = nested
        else:
            candidates = candidate.get("candidates")
            if type(candidates) is list:
                for item in reversed(candidates):
                    if isinstance(item, Mapping) and "answer" in item:
                        candidate = item
                        break
    if "answer" in candidate:
        reasoning = normalize_json_text(candidate.get("reasoning", ""))
        answer = normalize_json_text(candidate["answer"])
    else:
        reasoning = normalize_json_text(dict(payload))
        answer = reasoning
    if len(reasoning) > 20_000 or len(answer) > 4_000:
        raise ValueError("candidate content exceeds the workspace bound")
    return reasoning, answer


def decode_answer_candidate(payload: Mapping[str, Any]) -> tuple[str, str]:
    """Decode the strict answer-bearing container without judging its math."""
    if not isinstance(payload, Mapping) or "answer" not in payload:
        raise ValueError("candidate payload needs a top-level answer")
    raw_answer = payload["answer"]
    if not (
        isinstance(raw_answer, str)
        or type(raw_answer) is int
        or (type(raw_answer) is float and math.isfinite(raw_answer))
    ):
        raise ValueError("candidate answer must be a string or finite number")
    reasoning = normalize_json_text(payload.get("reasoning", ""))
    answer = normalize_json_text(raw_answer)
    if not answer:
        raise ValueError("candidate answer must be nonempty")
    if len(reasoning) > 20_000 or len(answer) > 4_000:
        raise ValueError("candidate content exceeds the workspace bound")
    return reasoning, answer


__all__ = [
    "ACTIVE_FRONTIER_OBSERVATION_SCHEMA_VERSION",
    "CHILD_CONTEXT_CONTRACT_VERSION",
    "CHILD_CONTEXT_MODES",
    "CHILD_CONTEXT_PRECEDING_SIBLINGS",
    "CHILD_CONTEXT_TARGET_ONLY",
    "DECOMPOSE_SCHEMA",
    "DIRECT_PROMPT_VERSION",
    "DIRECT_SCHEMA",
    "DIRECT_SYSTEM",
    "HYBRID_LENS_VERSION",
    "OBSERVATION_SCHEMA_VERSION",
    "ROOT_RESULT_CONTRACT_VERSION",
    "ROOT_RESULT_FREE",
    "ROOT_RESULT_MODES",
    "ROOT_RESULT_VERIFIER_READY",
    "ROOT_RESULT_VERIFIER_READY_INSTRUCTION",
    "REVISE_ANSWER_CONTAINER_INSTRUCTION",
    "REVISE_ANSWER_CONTAINER_SCHEMA",
    "REVISE_OPEN_OBJECT_INSTRUCTION",
    "REVISE_RESULT_ANSWER_CONTAINER",
    "REVISE_RESULT_CONTRACT_VERSION",
    "REVISE_RESULT_MODES",
    "REVISE_RESULT_OPEN_OBJECT",
    "TEACHER_PRESENTATION_VERSION",
    "WORKER_PROMPT_VERSION",
    "WORKER_PROMPT_ROOT_RESULT_VERSION",
    "WORKER_PROMPT_REVISE_ANSWER_VERSION",
    "WORKSPACE_CANDIDATE_SCHEMA",
    "WorkerRequest",
    "build_hybrid_lens",
    "build_observation",
    "build_worker_request",
    "decode_candidate",
    "decode_answer_candidate",
    "decode_children",
    "direct_request",
    "direct_user_prompt",
    "normalize_json_text",
    "validate_child_context_mode",
    "validate_root_result_mode",
    "validate_revise_result_mode",
]
