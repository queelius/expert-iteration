"""Full-workspace LLM teacher for the permissive action contract."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..llm_client import (
    LLMClient,
    LLMResponseError,
    SCHEMA_VALIDATION_STRICT_V1,
)
from .lenses import TEACHER_PRESENTATION_VERSION, build_observation
from .model import (
    ACTION_AVAILABILITY_ALL,
    Action,
    State,
    VERB_ORDER,
    Verb,
    protected_direct_candidate_id,
    validate_action_availability_mode,
)


TEACHER_OUTPUT_VERSION = "workspace-teacher-choice-v1"
TEACHER_PROMPT_VERSION = "workspace-global-planner-prompt-v1"
TEACHER_INDEX_OUTPUT_VERSION = "workspace-teacher-menu-index-v1"
TEACHER_INDEX_PROMPT_VERSION = "workspace-global-planner-index-prompt-v1.1"
TEACHER_ROUTING_INDEX_PROMPT_VERSION = (
    "workspace-revision-routing-index-prompt-v1"
)
TEACHER_PROMPT_GENERIC = "global-planner-v1"
TEACHER_PROMPT_REVISION_ROUTING = "revision-routing-v1"
TEACHER_PROMPT_MODES = frozenset({
    TEACHER_PROMPT_GENERIC,
    TEACHER_PROMPT_REVISION_ROUTING,
})
TEACHER_CHOICE_VERB_TARGET = "verb-target-choice-v1"
TEACHER_CHOICE_MENU_INDEX = "legal-menu-index-v1"
TEACHER_CHOICE_MODES = frozenset({
    TEACHER_CHOICE_VERB_TARGET,
    TEACHER_CHOICE_MENU_INDEX,
})
TEACHER_FALLBACK_VERSION = "baseline-preserving-teacher-fallback-v1"

_TEACHER_SYSTEM = (
    "You are the global planner for a frozen worker LLM operating on an "
    "external reasoning workspace. You see the full workspace; the worker "
    "will see a focused lens for the action you select. Choose exactly one "
    "listed legal action that is most likely to lead to a symbolically "
    "correct submitted root answer within the remaining turns. DECOMPOSE "
    "creates unverified child tasks. ANSWER asks the worker to answer one "
    "Goal. SYNTHESIZE combines completed child work for a Goal. REVISE "
    "improves a Candidate. SUBMIT invokes the symbolic checker and ends the "
    "episode. Intermediate content is unverified and gives no reward. Avoid "
    "unproductive repeated actions. Return only the selected verb and target."
)

_TEACHER_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "verb": {
            "type": "string",
            "enum": [verb.value for verb in VERB_ORDER],
        },
        "target": {"type": "integer"},
    },
    "required": ["verb", "target"],
    "additionalProperties": False,
}

_TEACHER_INDEX_SYSTEM = (
    "You are the global planner for a frozen worker LLM operating on an "
    "external reasoning workspace. You see the full workspace; the worker "
    "will see a focused lens for the action you select. Choose exactly one "
    "entry from the indexed legal action menu that is most likely to lead "
    "to a symbolically correct submitted root answer within the remaining "
    "turns. DECOMPOSE creates unverified child tasks. ANSWER asks the worker "
    "to answer one Goal. SYNTHESIZE combines completed child work for a "
    "Goal. REVISE improves a Candidate. SUBMIT invokes the symbolic checker "
    "and ends the episode. Intermediate content is unverified and gives no "
    "reward. Avoid unproductive repeated actions. Return only the selected "
    "action_index."
)

_TEACHER_ROUTING_INDEX_SYSTEM = (
    "You are the routing controller for a frozen worker LLM at a checkpoint "
    "with one protected direct Candidate. Choose exactly one entry from the "
    "indexed legal action menu. Inspect the root problem and the Candidate's "
    "reasoning and answer. REVISE asks the worker to solve the Candidate's "
    "Goal independently from the problem conditions, treating the current "
    "answer as untrusted; the resulting newest Candidate will be submitted "
    "next in this bounded comparison. SUBMIT immediately sends the selected "
    "Candidate to the symbolic checker. Prefer REVISE when the current answer "
    "or reasoning has any substantive gap, unchecked case, arithmetic risk, "
    "or mismatch with the question. Prefer SUBMIT only when the current result "
    "is clearly and completely supported. Choose DECOMPOSE only when neither "
    "submission nor one independent revision is an appropriate next step. "
    "Candidate protection and numeric id are provenance, not evidence of "
    "correctness. Return only the selected action_index."
)


@dataclass(frozen=True, slots=True)
class TeacherRequest:
    system: str
    user: str
    schema: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class TeacherDecision:
    turn: int
    proposed_action: Action | None
    selected_action: Action
    fallback: bool
    observation_json: str
    request: TeacherRequest
    payload_json: str | None

    def __post_init__(self) -> None:
        if type(self.turn) is not int or self.turn < 0:
            raise ValueError("teacher decision turn is invalid")
        if self.proposed_action is not None and not isinstance(
            self.proposed_action, Action
        ):
            raise ValueError("teacher proposed action is invalid")
        if not isinstance(self.selected_action, Action):
            raise ValueError("teacher selected action is invalid")
        if type(self.fallback) is not bool:
            raise ValueError("teacher fallback must be boolean")
        if self.fallback != (self.proposed_action != self.selected_action):
            raise ValueError("teacher fallback differs from selected action")
        if not isinstance(self.observation_json, str):
            raise ValueError("teacher observation must be canonical text")
        if not isinstance(self.request, TeacherRequest):
            raise ValueError("teacher request has an invalid type")
        if self.payload_json is not None and not isinstance(
            self.payload_json, str
        ):
            raise ValueError("teacher payload must be null or canonical text")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def build_teacher_request(
    state: State,
    *,
    max_depth: int,
    action_availability_mode: str = ACTION_AVAILABILITY_ALL,
    choice_mode: str = TEACHER_CHOICE_VERB_TARGET,
    prompt_mode: str = TEACHER_PROMPT_GENERIC,
) -> TeacherRequest:
    validate_action_availability_mode(action_availability_mode)
    if choice_mode not in TEACHER_CHOICE_MODES:
        raise ValueError("unknown teacher choice mode")
    if prompt_mode not in TEACHER_PROMPT_MODES:
        raise ValueError("unknown teacher prompt mode")
    if (
        prompt_mode == TEACHER_PROMPT_REVISION_ROUTING
        and choice_mode != TEACHER_CHOICE_MENU_INDEX
    ):
        raise ValueError("revision routing requires indexed teacher choice")
    observation = build_observation(
        state,
        max_depth=max_depth,
        action_availability_mode=action_availability_mode,
    )
    presentation = {
        "schema_version": TEACHER_PRESENTATION_VERSION,
        "observation": observation,
    }
    if choice_mode == TEACHER_CHOICE_MENU_INDEX:
        legal = observation["legal_actions"]
        if type(legal) is not list or not legal:
            raise ValueError("indexed teacher request requires legal actions")
        menu = [
            {"action_index": index, **item}
            for index, item in enumerate(legal)
        ]
        presentation["legal_action_menu"] = menu
        schema: Mapping[str, object] = {
            "type": "object",
            "properties": {
                "action_index": {
                    "type": "integer",
                    "enum": list(range(len(menu))),
                },
            },
            "required": ["action_index"],
            "additionalProperties": False,
        }
        system = (
            _TEACHER_ROUTING_INDEX_SYSTEM
            if prompt_mode == TEACHER_PROMPT_REVISION_ROUTING
            else _TEACHER_INDEX_SYSTEM
        )
        instruction = (
            "Select one action_index from legal_action_menu."
        )
    else:
        schema = _TEACHER_SCHEMA
        system = _TEACHER_SYSTEM
        instruction = (
            "Select exactly one object from observation.legal_actions."
        )
    return TeacherRequest(
        system=system,
        user=(
            f"Full workspace:\n{_canonical_json(presentation)}\n\n"
            f"{instruction}"
        ),
        schema=json.loads(_canonical_json(schema)),
    )


def decode_teacher_action(payload: Mapping[str, Any]) -> Action:
    if type(payload) is not dict or set(payload) != {"verb", "target"}:
        raise ValueError("teacher payload keys must be verb and target")
    try:
        verb = Verb(payload["verb"])
    except (TypeError, ValueError) as exc:
        raise ValueError("teacher verb is unknown") from exc
    target = payload["target"]
    if type(target) is not int or target < 0:
        raise ValueError("teacher target must be a node id")
    return Action(verb, target)


def decode_teacher_index(
    payload: Mapping[str, Any],
    legal_actions: tuple[Action, ...],
) -> Action:
    if type(payload) is not dict or set(payload) != {"action_index"}:
        raise ValueError("teacher index payload requires action_index")
    index = payload["action_index"]
    if type(index) is not int or not 0 <= index < len(legal_actions):
        raise ValueError("teacher action_index is outside the legal menu")
    return legal_actions[index]


def _fallback_action(
    state: State,
    legal_actions: tuple[Action, ...],
) -> Action:
    direct = Action(Verb.ANSWER, 0)
    if direct in legal_actions:
        return direct
    protected = protected_direct_candidate_id(state)
    if protected is not None:
        submit_protected = Action(Verb.SUBMIT, protected)
        if submit_protected in legal_actions:
            return submit_protected
    return legal_actions[0]


@dataclass(slots=True)
class LLMTeacherPolicy:
    """Strong behavior policy with explicit deterministic fallback."""

    client: LLMClient
    max_depth: int
    temperature: float = 0.0
    max_tokens: int = 64
    base_seed: int | None = None
    action_availability_mode: str = ACTION_AVAILABILITY_ALL
    choice_mode: str = TEACHER_CHOICE_VERB_TARGET
    prompt_mode: str = TEACHER_PROMPT_GENERIC
    decisions: list[TeacherDecision] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.client, LLMClient):
            raise TypeError("teacher client has an invalid type")
        if type(self.max_depth) is not int or self.max_depth <= 0:
            raise ValueError("teacher max_depth must be positive")
        if type(self.temperature) is not float or self.temperature < 0.0:
            raise ValueError("teacher temperature must be nonnegative float")
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("teacher max_tokens must be positive")
        if self.base_seed is not None and (
            type(self.base_seed) is not int or self.base_seed < 0
        ):
            raise ValueError("teacher base_seed must be null or nonnegative")
        validate_action_availability_mode(self.action_availability_mode)
        if self.choice_mode not in TEACHER_CHOICE_MODES:
            raise ValueError("unknown teacher choice mode")
        if self.prompt_mode not in TEACHER_PROMPT_MODES:
            raise ValueError("unknown teacher prompt mode")
        if (
            self.prompt_mode == TEACHER_PROMPT_REVISION_ROUTING
            and self.choice_mode != TEACHER_CHOICE_MENU_INDEX
        ):
            raise ValueError("revision routing requires indexed teacher choice")

    def choose(
        self,
        state: State,
        legal_actions: tuple[Action, ...],
    ) -> Action:
        if not legal_actions:
            raise ValueError("teacher received an empty legal menu")
        expected_observation = build_observation(
            state,
            max_depth=self.max_depth,
            action_availability_mode=self.action_availability_mode,
        )
        expected_legal = tuple(
            Action(Verb(item["verb"]), item["target"])
            for item in expected_observation["legal_actions"]
        )
        if legal_actions != expected_legal:
            raise ValueError("teacher legal menu differs from its observation")
        request = build_teacher_request(
            state,
            max_depth=self.max_depth,
            action_availability_mode=self.action_availability_mode,
            choice_mode=self.choice_mode,
            prompt_mode=self.prompt_mode,
        )
        observation_json = _canonical_json(expected_observation)
        proposed: Action | None = None
        payload_json: str | None = None
        try:
            payload = self.client.complete_json(
                request.system,
                request.user,
                request.schema,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=(
                    None
                    if self.base_seed is None
                    else self.base_seed + state.turn
                ),
                schema_validation_mode=SCHEMA_VALIDATION_STRICT_V1,
            )
            payload_json = _canonical_json(payload)
            proposed = (
                decode_teacher_action(payload)
                if self.choice_mode == TEACHER_CHOICE_VERB_TARGET
                else decode_teacher_index(payload, legal_actions)
            )
        except (LLMResponseError, TypeError, ValueError, OverflowError):
            proposed = None
            if self.choice_mode == TEACHER_CHOICE_VERB_TARGET:
                payload_json = None

        selected = (
            proposed
            if proposed in legal_actions
            else _fallback_action(state, legal_actions)
        )
        decision = TeacherDecision(
            turn=state.turn,
            proposed_action=proposed,
            selected_action=selected,
            fallback=proposed != selected,
            observation_json=observation_json,
            request=request,
            payload_json=payload_json,
        )
        self.decisions.append(decision)
        return selected


__all__ = [
    "TEACHER_CHOICE_MENU_INDEX",
    "TEACHER_CHOICE_MODES",
    "TEACHER_CHOICE_VERB_TARGET",
    "TEACHER_FALLBACK_VERSION",
    "TEACHER_INDEX_OUTPUT_VERSION",
    "TEACHER_INDEX_PROMPT_VERSION",
    "TEACHER_OUTPUT_VERSION",
    "TEACHER_PROMPT_VERSION",
    "TEACHER_PROMPT_GENERIC",
    "TEACHER_PROMPT_MODES",
    "TEACHER_PROMPT_REVISION_ROUTING",
    "TEACHER_ROUTING_INDEX_PROMPT_VERSION",
    "LLMTeacherPolicy",
    "TeacherDecision",
    "TeacherRequest",
    "build_teacher_request",
    "decode_teacher_action",
    "decode_teacher_index",
]
