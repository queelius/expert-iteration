"""Permissive worker transitions with symbolic terminal reward only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from ..llm_client import LLMClient, LLMResponseError
from ..verifier_adapter import RichResult, check_answer
from .lenses import (
    CHILD_CONTEXT_TARGET_ONLY,
    REVISE_RESULT_ANSWER_CONTAINER,
    REVISE_RESULT_OPEN_OBJECT,
    ROOT_RESULT_FREE,
    WorkerRequest,
    build_worker_request,
    decode_answer_candidate,
    decode_candidate,
    decode_children,
    validate_child_context_mode,
    validate_revise_result_mode,
    validate_root_result_mode,
)
from .model import (
    ACTION_AVAILABILITY_ALL,
    Action,
    ActionEvent,
    CandidateNode,
    EventOutcome,
    GoalNode,
    State,
    Verb,
    actions_for_mode,
    initial_state,
    node_at,
    validate_action_availability_mode,
)


WORKER_EXECUTION_CONTRACT_VERSION = "permissive-worker-execution-v1"
WORKER_EXECUTION_REDESIGN_CONTRACT_VERSION = (
    "permissive-worker-execution-v2"
)
SUBMIT_RECEIPT_VERSION = "symbolic-submit-receipt-v1"

AnswerVerifier = Callable[[str, str], RichResult]


@dataclass(frozen=True, slots=True)
class SubmitReceipt:
    accepted: bool
    observed: str
    reason_code: str | None
    schema_version: str = SUBMIT_RECEIPT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SUBMIT_RECEIPT_VERSION:
            raise ValueError("submit receipt schema differs from runtime")
        if type(self.accepted) is not bool:
            raise ValueError("submit receipt accepted must be boolean")
        if not isinstance(self.observed, str):
            raise ValueError("submit receipt observed must be text")
        if self.reason_code not in {None, "incorrect_answer"}:
            raise ValueError("submit receipt reason code is unsupported")
        if self.accepted != (self.reason_code is None):
            raise ValueError("submit receipt reason differs from acceptance")


@dataclass(frozen=True, slots=True)
class Transition:
    state: State
    action: Action
    next_state: State
    worker_request: WorkerRequest | None
    worker_payload_json: str | None
    submit_receipt: SubmitReceipt | None
    generation_error: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.state, State):
            raise ValueError("transition state has an invalid type")
        if not isinstance(self.action, Action):
            raise ValueError("transition action has an invalid type")
        if not isinstance(self.next_state, State):
            raise ValueError("transition next state has an invalid type")
        if self.next_state.turn != self.state.turn + 1:
            raise ValueError("transition must consume one learned turn")
        if self.next_state.history[:-1] != self.state.history:
            raise ValueError("transition changed prior state history")
        event = self.next_state.history[-1]
        if event.action != self.action or event.turn != self.state.turn:
            raise ValueError("transition event differs from its action")
        if self.action.verb is Verb.SUBMIT:
            if self.worker_request is not None or self.worker_payload_json is not None:
                raise ValueError("SUBMIT cannot contain a worker call")
            if not isinstance(self.submit_receipt, SubmitReceipt):
                raise ValueError("SUBMIT requires a symbolic receipt")
            expected = (
                EventOutcome.SUBMIT_ACCEPTED
                if self.submit_receipt.accepted
                else EventOutcome.SUBMIT_REJECTED
            )
            if event.outcome is not expected or self.generation_error:
                raise ValueError("SUBMIT transition outcome is inconsistent")
            return
        if self.submit_receipt is not None:
            raise ValueError("generative transition cannot contain a submit receipt")
        if not isinstance(self.worker_request, WorkerRequest):
            raise ValueError("generative transition requires its worker request")
        if self.generation_error:
            if self.worker_payload_json is not None:
                raise ValueError("generation error cannot retain a decoded payload")
            if event.outcome is not EventOutcome.GENERATION_ERROR:
                raise ValueError("generation error event is inconsistent")
        else:
            if not isinstance(self.worker_payload_json, str):
                raise ValueError("applied generation must retain its JSON payload")
            try:
                payload = json.loads(self.worker_payload_json)
            except json.JSONDecodeError as exc:
                raise ValueError("worker payload is not JSON") from exc
            if _canonical_json(payload) != self.worker_payload_json:
                raise ValueError("worker payload is not canonical JSON")
            if event.outcome is not EventOutcome.APPLIED:
                raise ValueError("applied generation event is inconsistent")

    @property
    def reward(self) -> int:
        return self.next_state.reward

    @property
    def terminal(self) -> bool:
        return self.next_state.terminal


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


@dataclass(frozen=True, slots=True)
class WorkspaceEnvironment:
    """One problem-bound MDP whose expected answer never enters state."""

    problem: str
    expected_answer: str
    worker: LLMClient
    turn_limit: int = 12
    max_depth: int = 3
    child_context_mode: str = CHILD_CONTEXT_TARGET_ONLY
    temperature: float = 0.0
    max_tokens: int = 700
    base_seed: int | None = None
    answer_verifier: AnswerVerifier = check_answer
    root_result_mode: str = ROOT_RESULT_FREE
    revise_result_mode: str = REVISE_RESULT_OPEN_OBJECT
    action_availability_mode: str = ACTION_AVAILABILITY_ALL

    def __post_init__(self) -> None:
        if not isinstance(self.problem, str) or not self.problem.strip():
            raise ValueError("workspace problem must be nonempty text")
        if not isinstance(self.expected_answer, str):
            raise ValueError("workspace expected answer must be text")
        if not isinstance(self.worker, LLMClient):
            raise TypeError("workspace worker has an invalid type")
        if type(self.turn_limit) is not int or self.turn_limit <= 0:
            raise ValueError("workspace turn_limit must be positive")
        if type(self.max_depth) is not int or self.max_depth <= 0:
            raise ValueError("workspace max_depth must be positive")
        validate_child_context_mode(self.child_context_mode)
        if type(self.temperature) is not float or self.temperature < 0.0:
            raise ValueError("workspace temperature must be nonnegative float")
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("workspace max_tokens must be positive")
        if self.base_seed is not None and (
            type(self.base_seed) is not int or self.base_seed < 0
        ):
            raise ValueError("workspace base_seed must be null or nonnegative")
        if not callable(self.answer_verifier):
            raise TypeError("workspace answer verifier must be callable")
        validate_root_result_mode(self.root_result_mode)
        validate_revise_result_mode(self.revise_result_mode)
        validate_action_availability_mode(self.action_availability_mode)

    def initial_state(self) -> State:
        return initial_state(self.problem, turn_limit=self.turn_limit)

    def legal_actions(self, state: State) -> tuple[Action, ...]:
        self._validate_state_binding(state)
        return actions_for_mode(
            state,
            max_depth=self.max_depth,
            mode=self.action_availability_mode,
        )

    def _validate_state_binding(self, state: State) -> None:
        if not isinstance(state, State):
            raise TypeError("workspace operation requires a State")
        root = node_at(state, 0)
        if not isinstance(root, GoalNode) or root.task != self.problem:
            raise ValueError("workspace state differs from its bound problem")
        if state.turn_limit != self.turn_limit:
            raise ValueError("workspace state differs from its bound horizon")

    def _next_state(
        self,
        state: State,
        event: ActionEvent,
        nodes: tuple[GoalNode | CandidateNode, ...],
    ) -> State:
        next_turn = state.turn + 1
        terminal = event.action.verb is Verb.SUBMIT or next_turn == self.turn_limit
        reward = int(event.outcome is EventOutcome.SUBMIT_ACCEPTED)
        return State(
            nodes=nodes,
            history=state.history + (event,),
            turn=next_turn,
            turn_limit=state.turn_limit,
            terminal=terminal,
            reward=reward,
        )

    def _submit(self, state: State, action: Action) -> Transition:
        target = node_at(state, action.target)
        if not isinstance(target, CandidateNode) or target.goal_id != 0:
            raise ValueError("SUBMIT target must be a root Candidate")
        result = self.answer_verifier(target.answer, self.expected_answer)
        if not isinstance(result, RichResult):
            raise TypeError("answer verifier returned an invalid result")
        receipt = SubmitReceipt(
            accepted=bool(result.success),
            observed=result.observed,
            reason_code=None if result.success else "incorrect_answer",
        )
        event = ActionEvent(
            turn=state.turn,
            action=action,
            outcome=(
                EventOutcome.SUBMIT_ACCEPTED
                if receipt.accepted
                else EventOutcome.SUBMIT_REJECTED
            ),
        )
        return Transition(
            state=state,
            action=action,
            next_state=self._next_state(state, event, state.nodes),
            worker_request=None,
            worker_payload_json=None,
            submit_receipt=receipt,
        )

    def _created_nodes(
        self,
        state: State,
        action: Action,
        payload: dict[str, object],
    ) -> tuple[GoalNode | CandidateNode, ...]:
        first_id = len(state.nodes)
        if action.verb is Verb.DECOMPOSE:
            return tuple(
                GoalNode(first_id + index, action.target, task, state.turn)
                for index, task in enumerate(decode_children(payload))
            )
        reasoning, answer = (
            decode_answer_candidate(payload)
            if action.verb is Verb.REVISE
            and self.revise_result_mode == REVISE_RESULT_ANSWER_CONTAINER
            else decode_candidate(payload)
        )
        if action.verb in {Verb.ANSWER, Verb.SYNTHESIZE}:
            goal_id = action.target
            revises = None
        elif action.verb is Verb.REVISE:
            target = node_at(state, action.target)
            if not isinstance(target, CandidateNode):
                raise ValueError("REVISE target must be a Candidate")
            goal_id = target.goal_id
            revises = target.id
        else:
            raise ValueError("unsupported generative action")
        return (
            CandidateNode(
                first_id,
                goal_id,
                reasoning,
                answer,
                state.turn,
                revises,
            ),
        )

    def step(self, state: State, action: Action) -> Transition:
        """Execute one legal action with no intermediate semantic rejection."""
        self._validate_state_binding(state)
        if not isinstance(action, Action):
            raise TypeError("workspace action has an invalid type")
        if action not in self.legal_actions(state):
            raise ValueError("workspace action is outside the legal menu")
        if action.verb is Verb.SUBMIT:
            return self._submit(state, action)

        request = build_worker_request(
            state,
            action,
            child_context_mode=self.child_context_mode,
            root_result_mode=self.root_result_mode,
            revise_result_mode=self.revise_result_mode,
        )
        effective_seed = (
            None if self.base_seed is None else self.base_seed + state.turn
        )
        try:
            payload = self.worker.complete_json(
                request.system,
                request.user,
                request.schema,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=effective_seed,
                schema_validation_mode=request.schema_validation_mode,
            )
            if type(payload) is not dict:
                raise ValueError("worker response must be an object")
            created = self._created_nodes(state, action, payload)
            payload_json = _canonical_json(payload)
        except (LLMResponseError, TypeError, ValueError, OverflowError):
            event = ActionEvent(
                state.turn,
                action,
                EventOutcome.GENERATION_ERROR,
            )
            return Transition(
                state=state,
                action=action,
                next_state=self._next_state(state, event, state.nodes),
                worker_request=request,
                worker_payload_json=None,
                submit_receipt=None,
                generation_error=True,
            )

        event = ActionEvent(
            state.turn,
            action,
            EventOutcome.APPLIED,
            tuple(node.id for node in created),
        )
        return Transition(
            state=state,
            action=action,
            next_state=self._next_state(state, event, state.nodes + created),
            worker_request=request,
            worker_payload_json=payload_json,
            submit_receipt=None,
        )


__all__ = [
    "SUBMIT_RECEIPT_VERSION",
    "WORKER_EXECUTION_CONTRACT_VERSION",
    "WORKER_EXECUTION_REDESIGN_CONTRACT_VERSION",
    "AnswerVerifier",
    "SubmitReceipt",
    "Transition",
    "WorkspaceEnvironment",
]
