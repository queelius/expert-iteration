"""Paired replay of versioned REVISE answer-container contracts."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..llm_client import OpenAICompatibleLLMClient
from ..verifier_adapter import check_answer
from .experiment import (
    EVALUATION_BINDING_VERSION,
    PROBLEM_SOURCE_VERSION,
    ModelIdentity,
    ProblemExample,
    _AttemptRecorder,
    _counter_delta,
    _counter_snapshot,
    _finalize_no_clobber,
    _validate_client_binding,
    _validate_output_paths,
    _write_record,
    evaluation_binding_sha256,
    problem_source_sha256,
)
from .lenses import (
    ACTIVE_FRONTIER_OBSERVATION_SCHEMA_VERSION,
    CHILD_CONTEXT_CONTRACT_VERSION,
    CHILD_CONTEXT_PRECEDING_SIBLINGS,
    HYBRID_LENS_VERSION,
    REVISE_ANSWER_CONTAINER_INSTRUCTION,
    REVISE_ANSWER_CONTAINER_SCHEMA,
    REVISE_OPEN_OBJECT_INSTRUCTION,
    REVISE_RESULT_ANSWER_CONTAINER,
    REVISE_RESULT_CONTRACT_VERSION,
    REVISE_RESULT_OPEN_OBJECT,
    ROOT_RESULT_CONTRACT_VERSION,
    ROOT_RESULT_VERIFIER_READY,
    WORKER_PROMPT_REVISE_ANSWER_VERSION,
    WORKER_PROMPT_ROOT_RESULT_VERSION,
    WORKSPACE_CANDIDATE_SCHEMA,
    WorkerRequest,
    build_observation,
    build_worker_request,
    normalize_json_text,
)
from .model import (
    ACTION_AVAILABILITY_ACTIVE_FRONTIER,
    ACTION_SCHEMA_VERSION,
    REWARD_CONTRACT_VERSION,
    STATE_SCHEMA_VERSION,
    TRANSITION_SCHEMA_VERSION,
    Action,
    CandidateNode,
    State,
    Verb,
    actions_for_mode,
    node_at,
)
from .paired_direct_teacher import (
    PAIRED_DIRECT_TEACHER_HEADER_VERSION,
    PAIRED_DIRECT_TEACHER_TRANSITION_VERSION,
    ROLE_TEACHER_WORKER,
)
from .paired_lens import _read_canonical_jsonl, file_sha256, state_from_observation
from .records import (
    DIAGNOSTIC_ATTEMPT_VERSION,
    canonical_json_sha256,
    transition_record,
)
from .runtime import (
    SUBMIT_RECEIPT_VERSION,
    WORKER_EXECUTION_REDESIGN_CONTRACT_VERSION,
    Transition,
    WorkspaceEnvironment,
)


PAIRED_REVISE_CONTRACT_VERSION = "workspace-paired-revise-contract-v1"
PAIRED_REVISE_HEADER_VERSION = "workspace-paired-revise-header-v1"
PAIRED_REVISE_DIAGNOSTIC_VERSION = (
    "workspace-paired-revise-diagnostic-header-v1"
)
PAIRED_REVISE_TRANSITION_VERSION = "workspace-paired-revise-transition-v1"
PAIRED_REVISE_CASE_VERSION = "workspace-paired-revise-case-v1"
PAIRED_REVISE_CASE_SUMMARY_VERSION = (
    "workspace-paired-revise-case-summary-v1"
)
PAIRED_REVISE_SUMMARY_VERSION = "workspace-paired-revise-summary-v1"
PAIRED_REVISE_SOURCE_VERSION = "teacher-selected-revise-source-v1"
PAIRED_REVISE_SOURCE_ACTION_VERSION = "e37-teacher-selected-revise-v1"
PAIRED_REVISE_BRANCH_ORDER_VERSION = "checkerboard-revise-state-order-v1"
PAIRED_REVISE_SEED_POLICY_VERSION = "source-episode-turn-offset-v1"
PAIRED_REVISE_CONTAINER_MEASURE_VERSION = (
    "top-level-nonempty-string-or-number-v1"
)
PAIRED_REVISE_COLLECTION_PURPOSE = "evaluation-only-v1"
PAIRED_REVISE_MODES = (
    REVISE_RESULT_OPEN_OBJECT,
    REVISE_RESULT_ANSWER_CONTAINER,
)
_ROLE_BY_MODE = {
    REVISE_RESULT_OPEN_OBJECT: "open_object_revise_worker",
    REVISE_RESULT_ANSWER_CONTAINER: "answer_container_revise_worker",
}


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _sha256(value: str, name: str) -> str:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase sha256")
    return value


def _request_record(request: WorkerRequest) -> dict[str, object]:
    return {
        "system": request.system,
        "user": request.user,
        "schema": dict(request.schema),
        "schema_validation_mode": request.schema_validation_mode,
    }


def _model_identity(record: object) -> ModelIdentity:
    if type(record) is not dict:
        raise ValueError("source worker identity is invalid")
    try:
        return ModelIdentity(
            provider=record["provider"],
            model=record["model"],
            revision=record["revision"],
            base_url=record["base_url"],
            guided_json=record["guided_json"],
            timeout_s=record["timeout_s"],
            parse_retries=record["parse_retries"],
            backoff_retries=record["backoff_retries"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("source worker identity is invalid") from exc


def _answer_container(payload: object) -> tuple[bool, str | None]:
    if type(payload) is not dict or "answer" not in payload:
        return False, None
    answer = payload["answer"]
    if not (
        isinstance(answer, str)
        or type(answer) is int
        or (type(answer) is float and answer == answer and abs(answer) != float("inf"))
    ):
        return False, None
    normalized = normalize_json_text(answer)
    if not normalized:
        return False, None
    return True, normalized


def _assert_treatment_isolated(
    open_object: WorkerRequest,
    answer_container: WorkerRequest,
) -> None:
    if (
        open_object.system != answer_container.system
        or open_object.schema_validation_mode
        != answer_container.schema_validation_mode
    ):
        raise RuntimeError("REVISE treatment changed the transport boundary")
    if open_object.schema != WORKSPACE_CANDIDATE_SCHEMA:
        raise RuntimeError("open-object REVISE schema changed")
    if answer_container.schema != REVISE_ANSWER_CONTAINER_SCHEMA:
        raise RuntimeError("answer-container REVISE schema changed")
    if open_object.user.count(REVISE_OPEN_OBJECT_INSTRUCTION) != 1:
        raise RuntimeError("open-object REVISE instruction is not isolated")
    if answer_container.user.count(REVISE_ANSWER_CONTAINER_INSTRUCTION) != 1:
        raise RuntimeError("answer-container REVISE instruction is not isolated")
    expected = open_object.user.replace(
        REVISE_OPEN_OBJECT_INSTRUCTION,
        REVISE_ANSWER_CONTAINER_INSTRUCTION,
        1,
    )
    if expected != answer_container.user:
        raise RuntimeError("REVISE branches differ outside the result contract")


@dataclass(frozen=True, slots=True)
class ReviseReplayCase:
    problem_id: str
    source_episode_index: int
    source_turn: int
    state: State
    action: Action
    source_worker_call_id: str
    source_request_sha256: str
    source_payload_sha256: str | None
    source_generation_error: bool
    source_answer_container: bool

    def __post_init__(self) -> None:
        _nonempty(self.problem_id, "revision problem id")
        if type(self.source_episode_index) is not int or self.source_episode_index < 0:
            raise ValueError("source episode index must be nonnegative")
        if type(self.source_turn) is not int or self.source_turn < 0:
            raise ValueError("source turn must be nonnegative")
        if not isinstance(self.state, State) or self.state.terminal:
            raise ValueError("revision source state must be nonterminal")
        if self.state.turn != self.source_turn:
            raise ValueError("revision source turn differs from state")
        if not isinstance(self.action, Action) or self.action.verb is not Verb.REVISE:
            raise ValueError("revision source action must be REVISE")
        target = node_at(self.state, self.action.target)
        if not isinstance(target, CandidateNode):
            raise ValueError("revision source target must be a Candidate")
        _nonempty(self.source_worker_call_id, "source worker call id")
        _sha256(self.source_request_sha256, "source request sha256")
        if self.source_payload_sha256 is not None:
            _sha256(self.source_payload_sha256, "source payload sha256")
        if type(self.source_generation_error) is not bool:
            raise ValueError("source generation error must be boolean")
        if type(self.source_answer_container) is not bool:
            raise ValueError("source answer container must be boolean")

    @property
    def target_is_root(self) -> bool:
        target = node_at(self.state, self.action.target)
        if not isinstance(target, CandidateNode):
            raise AssertionError("validated revision target changed type")
        return target.goal_id == 0

    def source_record(self, *, max_depth: int) -> dict[str, object]:
        return {
            "schema_version": PAIRED_REVISE_SOURCE_VERSION,
            "problem_id": self.problem_id,
            "source_episode_index": self.source_episode_index,
            "source_turn": self.source_turn,
            "source_observation": build_observation(
                self.state,
                max_depth=max_depth,
                action_availability_mode=ACTION_AVAILABILITY_ACTIVE_FRONTIER,
            ),
            "action": {
                "verb": self.action.verb.value,
                "target": self.action.target,
            },
            "target_is_root": self.target_is_root,
            "source_worker_call_id": self.source_worker_call_id,
            "source_request_sha256": self.source_request_sha256,
            "source_payload_sha256": self.source_payload_sha256,
            "source_generation_error": self.source_generation_error,
            "source_answer_container": self.source_answer_container,
        }


@dataclass(frozen=True, slots=True)
class PreparedReviseReplays:
    source_run_path: Path
    source_diagnostic_path: Path
    source_run_sha256: str
    source_diagnostic_sha256: str
    source_run_contract_sha256: str
    selection_sha256: str
    cases: tuple[ReviseReplayCase, ...]
    examples: tuple[ProblemExample, ...]
    worker: ModelIdentity
    worker_temperature: float
    worker_max_tokens: int
    worker_seed: int
    turn_limit: int
    max_depth: int

    def __post_init__(self) -> None:
        _sha256(self.source_run_sha256, "source run sha256")
        _sha256(self.source_diagnostic_sha256, "source diagnostic sha256")
        _sha256(self.source_run_contract_sha256, "source contract sha256")
        _sha256(self.selection_sha256, "revision selection sha256")
        if not self.cases or not self.examples:
            raise ValueError("prepared revision replay must be nonempty")
        if not isinstance(self.worker, ModelIdentity):
            raise ValueError("prepared worker identity is invalid")


def prepare_revise_replays(
    *,
    source_run_path: str | Path,
    source_diagnostic_path: str | Path,
    examples: tuple[ProblemExample, ...],
    expected_source_run_contract_sha256: str,
    max_depth: int,
) -> PreparedReviseReplays:
    """Strict-read E37 and seal every teacher-selected pre-REVISE state."""
    _sha256(
        expected_source_run_contract_sha256,
        "expected source run contract sha256",
    )
    if not examples:
        raise ValueError("revision replay preparation requires examples")
    run_path = Path(source_run_path)
    diagnostic_path = Path(source_diagnostic_path)
    records = _read_canonical_jsonl(run_path)
    diagnostics = _read_canonical_jsonl(diagnostic_path)
    header = records[0]
    if (
        header.get("record_type") != "run_header"
        or header.get("schema_version") != PAIRED_DIRECT_TEACHER_HEADER_VERSION
    ):
        raise ValueError("source run is not a paired direct-teacher run")
    if header.get("run_contract_sha256") != expected_source_run_contract_sha256:
        raise ValueError("source run contract differs from the expected digest")
    source = header.get("problem_source")
    if type(source) is not dict:
        raise ValueError("source problem binding is invalid")
    if source.get("sha256") != problem_source_sha256(examples):
        raise ValueError("source problem projection differs")
    if source.get("evaluation_binding_sha256") != evaluation_binding_sha256(
        examples
    ):
        raise ValueError("source hidden-label binding differs")
    if tuple(source.get("problem_ids", ())) != tuple(
        example.problem_id for example in examples
    ):
        raise ValueError("source problem order differs")
    settings = header.get("settings")
    contracts = header.get("contracts")
    models = header.get("models")
    if type(settings) is not dict or type(contracts) is not dict or type(models) is not dict:
        raise ValueError("source run header containers are invalid")
    if (
        settings.get("child_context_mode") != CHILD_CONTEXT_PRECEDING_SIBLINGS
        or settings.get("root_result_mode") != ROOT_RESULT_VERIFIER_READY
        or settings.get("action_availability_mode")
        != ACTION_AVAILABILITY_ACTIVE_FRONTIER
        or settings.get("max_depth") != max_depth
        or contracts.get("worker_prompt") != WORKER_PROMPT_ROOT_RESULT_VERSION
    ):
        raise ValueError("source run does not use the sealed E37 worker interface")
    worker = _model_identity(models.get("worker"))
    worker_temperature = settings.get("worker_temperature")
    worker_max_tokens = settings.get("worker_max_tokens")
    worker_seed = settings.get("worker_seed")
    turn_limit = settings.get("turn_limit")
    if (
        type(worker_temperature) is not float
        or type(worker_max_tokens) is not int
        or type(worker_seed) is not int
        or type(turn_limit) is not int
    ):
        raise ValueError("source worker settings are invalid")

    diagnostic_header = diagnostics[0]
    if (
        diagnostic_header.get("record_type") != "diagnostic_header"
        or diagnostic_header.get("run_contract_sha256")
        != expected_source_run_contract_sha256
    ):
        raise ValueError("source diagnostics differ from the source contract")
    attempts_by_call: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in diagnostics[1:]:
        if record.get("record_type") != "http_attempt":
            raise ValueError("source diagnostics contain an unknown record")
        attempt = record.get("attempt")
        if type(attempt) is not dict:
            raise ValueError("source diagnostic attempt is invalid")
        call_id = _nonempty(attempt.get("call_id"), "source diagnostic call id")
        attempts_by_call[call_id].append(record)

    examples_by_id = {example.problem_id: example for example in examples}
    cases: list[ReviseReplayCase] = []
    for record in records[1:]:
        if (
            record.get("schema_version") != PAIRED_DIRECT_TEACHER_TRANSITION_VERSION
            or record.get("action") is None
        ):
            continue
        raw_action = record["action"]
        if type(raw_action) is not dict or raw_action.get("verb") != Verb.REVISE.value:
            continue
        if (
            record.get("arm") != "teacher_continuation"
            or record.get("paired_phase") != "teacher_continuation"
            or record.get("decision_source") != "teacher"
            or record.get("behavior_cloning_eligible") is not False
        ):
            raise ValueError("source REVISE row has an invalid provenance")
        problem_id = _nonempty(record.get("problem_id"), "source problem id")
        if problem_id not in examples_by_id:
            raise ValueError("source REVISE problem is outside the source")
        episode_index = record.get("episode_index")
        source_turn = record.get("turn")
        if type(episode_index) is not int or type(source_turn) is not int:
            raise ValueError("source REVISE position is invalid")
        if examples[episode_index].problem_id != problem_id:
            raise ValueError("source REVISE episode binding differs")
        state = state_from_observation(
            record["observation"],
            max_depth=max_depth,
            action_availability_mode=ACTION_AVAILABILITY_ACTIVE_FRONTIER,
        )
        state_from_observation(
            record["next_observation"],
            max_depth=max_depth,
            action_availability_mode=ACTION_AVAILABILITY_ACTIVE_FRONTIER,
        )
        action = Action(Verb.REVISE, raw_action["target"])
        if action not in actions_for_mode(
            state,
            max_depth=max_depth,
            mode=ACTION_AVAILABILITY_ACTIVE_FRONTIER,
        ):
            raise ValueError("source REVISE action is not legal in its observation")
        expected_request = build_worker_request(
            state,
            action,
            child_context_mode=CHILD_CONTEXT_PRECEDING_SIBLINGS,
            root_result_mode=ROOT_RESULT_VERIFIER_READY,
            revise_result_mode=REVISE_RESULT_OPEN_OBJECT,
        )
        request_record = _request_record(expected_request)
        if record.get("worker_request") != request_record:
            raise ValueError("source REVISE request does not replay exactly")
        call_id = _nonempty(record.get("worker_call_id"), "source worker call id")
        call_attempts = attempts_by_call.get(call_id, [])
        if not call_attempts or any(
            attempt.get("role") != ROLE_TEACHER_WORKER
            for attempt in call_attempts
        ):
            raise ValueError("source REVISE call is not bound to worker diagnostics")
        generation_error = record.get("generation_error")
        payload = record.get("worker_payload")
        if type(generation_error) is not bool:
            raise ValueError("source REVISE generation status is invalid")
        if generation_error != (payload is None):
            raise ValueError("source REVISE payload differs from generation status")
        container, _ = _answer_container(payload)
        case = ReviseReplayCase(
            problem_id=problem_id,
            source_episode_index=episode_index,
            source_turn=source_turn,
            state=state,
            action=action,
            source_worker_call_id=call_id,
            source_request_sha256=canonical_json_sha256(request_record),
            source_payload_sha256=(
                None if payload is None else canonical_json_sha256(payload)
            ),
            source_generation_error=generation_error,
            source_answer_container=container,
        )
        cases.append(case)
    if not cases:
        raise ValueError("source run contains no teacher-selected REVISE states")
    case_keys = [
        (case.source_episode_index, case.source_turn) for case in cases
    ]
    if len(case_keys) != len(set(case_keys)):
        raise ValueError("source REVISE positions are repeated")
    selection_record = {
        "schema_version": PAIRED_REVISE_SOURCE_VERSION,
        "source_run_sha256": file_sha256(run_path),
        "source_diagnostic_sha256": file_sha256(diagnostic_path),
        "source_run_contract_sha256": expected_source_run_contract_sha256,
        "cases": [case.source_record(max_depth=max_depth) for case in cases],
    }
    return PreparedReviseReplays(
        source_run_path=run_path,
        source_diagnostic_path=diagnostic_path,
        source_run_sha256=selection_record["source_run_sha256"],
        source_diagnostic_sha256=selection_record["source_diagnostic_sha256"],
        source_run_contract_sha256=expected_source_run_contract_sha256,
        selection_sha256=canonical_json_sha256(selection_record),
        cases=tuple(cases),
        examples=examples,
        worker=worker,
        worker_temperature=worker_temperature,
        worker_max_tokens=worker_max_tokens,
        worker_seed=worker_seed,
        turn_limit=turn_limit,
        max_depth=max_depth,
    )


@dataclass(frozen=True, slots=True)
class PairedReviseContract:
    preregistration_id: str
    source_run_sha256: str
    source_diagnostic_sha256: str
    source_run_contract_sha256: str
    selection_sha256: str
    problem_source_sha256: str
    evaluation_binding_sha256: str
    case_keys: tuple[tuple[int, int], ...]
    problem_ids: tuple[str, ...]
    worker: ModelIdentity
    worker_temperature: float
    worker_max_tokens: int
    worker_seed: int
    turn_limit: int
    max_depth: int
    modes: tuple[str, str] = PAIRED_REVISE_MODES
    branch_order_version: str = PAIRED_REVISE_BRANCH_ORDER_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.preregistration_id, "preregistration id")
        for name in (
            "source_run_sha256",
            "source_diagnostic_sha256",
            "source_run_contract_sha256",
            "selection_sha256",
            "problem_source_sha256",
            "evaluation_binding_sha256",
        ):
            _sha256(getattr(self, name), name.replace("_", " "))
        if not self.case_keys or len(self.case_keys) != len(set(self.case_keys)):
            raise ValueError("revision case keys must be nonempty and unique")
        if len(self.problem_ids) != len(self.case_keys):
            raise ValueError("revision problem ids differ from case keys")
        if self.modes != PAIRED_REVISE_MODES:
            raise ValueError("revision modes differ from the contract")
        if self.branch_order_version != PAIRED_REVISE_BRANCH_ORDER_VERSION:
            raise ValueError("revision branch order differs from the contract")
        if not isinstance(self.worker, ModelIdentity):
            raise ValueError("revision worker identity is invalid")
        if type(self.worker_temperature) is not float or self.worker_temperature < 0:
            raise ValueError("revision temperature must be nonnegative")
        if type(self.worker_max_tokens) is not int or self.worker_max_tokens <= 0:
            raise ValueError("revision max tokens must be positive")
        if type(self.worker_seed) is not int or self.worker_seed < 0:
            raise ValueError("revision seed must be nonnegative")
        if type(self.turn_limit) is not int or self.turn_limit <= 0:
            raise ValueError("revision turn limit must be positive")
        if type(self.max_depth) is not int or self.max_depth <= 0:
            raise ValueError("revision max depth must be positive")

    def header(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema_version": PAIRED_REVISE_HEADER_VERSION,
            "record_type": "run_header",
            "run_contract_version": PAIRED_REVISE_CONTRACT_VERSION,
            "preregistration_id": self.preregistration_id,
            "source": {
                "schema_version": PAIRED_REVISE_SOURCE_VERSION,
                "source_run_sha256": self.source_run_sha256,
                "source_diagnostic_sha256": self.source_diagnostic_sha256,
                "source_run_contract_sha256": self.source_run_contract_sha256,
                "selection_sha256": self.selection_sha256,
                "problem_source_schema": PROBLEM_SOURCE_VERSION,
                "problem_source_sha256": self.problem_source_sha256,
                "evaluation_binding_version": EVALUATION_BINDING_VERSION,
                "evaluation_binding_sha256": self.evaluation_binding_sha256,
                "case_keys": [list(key) for key in self.case_keys],
                "problem_ids": list(self.problem_ids),
            },
            "contracts": {
                "state": STATE_SCHEMA_VERSION,
                "action": ACTION_SCHEMA_VERSION,
                "transition": TRANSITION_SCHEMA_VERSION,
                "transition_record": PAIRED_REVISE_TRANSITION_VERSION,
                "reward": REWARD_CONTRACT_VERSION,
                "worker_execution": WORKER_EXECUTION_REDESIGN_CONTRACT_VERSION,
                "worker_prompt_root_result": WORKER_PROMPT_ROOT_RESULT_VERSION,
                "worker_prompt_answer_container": (
                    WORKER_PROMPT_REVISE_ANSWER_VERSION
                ),
                "hybrid_lens": HYBRID_LENS_VERSION,
                "child_answer_context": CHILD_CONTEXT_CONTRACT_VERSION,
                "root_result": ROOT_RESULT_CONTRACT_VERSION,
                "revise_result": REVISE_RESULT_CONTRACT_VERSION,
                "observation": ACTIVE_FRONTIER_OBSERVATION_SCHEMA_VERSION,
                "action_availability": ACTION_AVAILABILITY_ACTIVE_FRONTIER,
                "submit_receipt": SUBMIT_RECEIPT_VERSION,
                "source_action": PAIRED_REVISE_SOURCE_ACTION_VERSION,
                "branch_order": self.branch_order_version,
                "seed_policy": PAIRED_REVISE_SEED_POLICY_VERSION,
                "container_measure": PAIRED_REVISE_CONTAINER_MEASURE_VERSION,
                "collection_purpose": PAIRED_REVISE_COLLECTION_PURPOSE,
            },
            "settings": {
                "case_count": len(self.case_keys),
                "turn_limit": self.turn_limit,
                "max_depth": self.max_depth,
                "child_context_mode": CHILD_CONTEXT_PRECEDING_SIBLINGS,
                "root_result_mode": ROOT_RESULT_VERIFIER_READY,
                "action_availability_mode": (
                    ACTION_AVAILABILITY_ACTIVE_FRONTIER
                ),
                "modes": list(self.modes),
                "worker_temperature": self.worker_temperature,
                "worker_max_tokens": self.worker_max_tokens,
                "worker_seed": self.worker_seed,
            },
            "models": {"worker": self.worker.to_record(), "teacher": None},
        }
        body["run_contract_sha256"] = canonical_json_sha256(body)
        return body

    @property
    def sha256(self) -> str:
        return str(self.header()["run_contract_sha256"])


def build_paired_revise_contract(
    *,
    preregistration_id: str,
    prepared: PreparedReviseReplays,
) -> PairedReviseContract:
    return PairedReviseContract(
        preregistration_id=preregistration_id,
        source_run_sha256=prepared.source_run_sha256,
        source_diagnostic_sha256=prepared.source_diagnostic_sha256,
        source_run_contract_sha256=prepared.source_run_contract_sha256,
        selection_sha256=prepared.selection_sha256,
        problem_source_sha256=problem_source_sha256(prepared.examples),
        evaluation_binding_sha256=evaluation_binding_sha256(prepared.examples),
        case_keys=tuple(
            (case.source_episode_index, case.source_turn)
            for case in prepared.cases
        ),
        problem_ids=tuple(case.problem_id for case in prepared.cases),
        worker=prepared.worker,
        worker_temperature=prepared.worker_temperature,
        worker_max_tokens=prepared.worker_max_tokens,
        worker_seed=prepared.worker_seed,
        turn_limit=prepared.turn_limit,
        max_depth=prepared.max_depth,
    )


@dataclass(frozen=True, slots=True)
class PairedReviseCollection:
    run_path: Path
    diagnostic_path: Path
    case_count: int
    transition_count: int
    answer_containers_by_mode: Mapping[str, int]
    root_symbolic_success_by_mode: Mapping[str, int]
    run_contract_sha256: str


def _branch_order(case_index: int) -> tuple[str, str]:
    if case_index % 2 == 0:
        return PAIRED_REVISE_MODES
    return tuple(reversed(PAIRED_REVISE_MODES))  # type: ignore[return-value]


def _paired_outcome(values: Mapping[str, bool]) -> str:
    open_object = values[REVISE_RESULT_OPEN_OBJECT]
    answer_container = values[REVISE_RESULT_ANSWER_CONTAINER]
    if open_object and answer_container:
        return "both"
    if open_object:
        return "open_object_only"
    if answer_container:
        return "answer_container_only"
    return "neither"


def _new_candidate(transition: Transition) -> CandidateNode | None:
    created = transition.next_state.history[-1].created_ids
    if len(created) != 1:
        return None
    node = transition.next_state.nodes[created[0]]
    return node if isinstance(node, CandidateNode) else None


def collect_paired_revise_run(
    *,
    run_path: str | Path,
    diagnostic_path: str | Path,
    contract: PairedReviseContract,
    prepared: PreparedReviseReplays,
    worker_client: OpenAICompatibleLLMClient,
) -> PairedReviseCollection:
    """Replay both REVISE contracts from every exact sealed source state."""
    if not isinstance(contract, PairedReviseContract):
        raise TypeError("paired revision collection requires its run contract")
    if not isinstance(prepared, PreparedReviseReplays):
        raise TypeError("paired revision collection requires prepared states")
    if type(worker_client) is not OpenAICompatibleLLMClient:
        raise TypeError("paired revision worker must be OpenAI-compatible")
    _validate_client_binding(contract.worker, worker_client, role="worker")
    expected = build_paired_revise_contract(
        preregistration_id=contract.preregistration_id,
        prepared=prepared,
    )
    if expected != contract:
        raise ValueError("paired revision contract differs from prepared states")

    final_run = Path(run_path)
    final_diagnostic = Path(diagnostic_path)
    partial_run, partial_diagnostic = _validate_output_paths(
        final_run,
        final_diagnostic,
    )
    diagnostic_header = {
        "schema_version": PAIRED_REVISE_DIAGNOSTIC_VERSION,
        "record_type": "diagnostic_header",
        "preregistration_id": contract.preregistration_id,
        "run_contract_sha256": contract.sha256,
        "attempt_schema_version": DIAGNOSTIC_ATTEMPT_VERSION,
        "roles": list(_ROLE_BY_MODE.values()),
    }
    examples = {example.problem_id: example for example in prepared.examples}
    worker_before = _counter_snapshot(worker_client)
    old_observer = worker_client.attempt_observer
    answer_containers = Counter({mode: 0 for mode in PAIRED_REVISE_MODES})
    generation_errors = Counter({mode: 0 for mode in PAIRED_REVISE_MODES})
    root_symbolic_success = Counter({mode: 0 for mode in PAIRED_REVISE_MODES})
    root_symbolic_attempts = Counter({mode: 0 for mode in PAIRED_REVISE_MODES})
    paired_outcomes: Counter[str] = Counter()
    transition_count = 0

    try:
        with partial_run.open(
            "x", encoding="ascii", newline="\n"
        ) as run_handle, partial_diagnostic.open(
            "x", encoding="ascii", newline="\n"
        ) as diagnostic_handle:
            _write_record(run_handle, contract.header())
            _write_record(diagnostic_handle, diagnostic_header, flush=True)
            recorder = _AttemptRecorder(
                diagnostic_handle,
                roles=tuple(_ROLE_BY_MODE.values()),
            )
            for case_index, case in enumerate(prepared.cases):
                example = examples[case.problem_id]
                source_target = node_at(case.state, case.action.target)
                if not isinstance(source_target, CandidateNode):
                    raise AssertionError("validated revision target changed type")
                source_target_symbolic = (
                    bool(check_answer(
                        source_target.answer,
                        example.expected_answer,
                    ).success)
                    if case.target_is_root
                    else None
                )
                _write_record(run_handle, {
                    "schema_version": PAIRED_REVISE_CASE_VERSION,
                    "record_type": "paired_case_source",
                    "preregistration_id": contract.preregistration_id,
                    "paired_case_index": case_index,
                    **case.source_record(max_depth=contract.max_depth),
                    "source_target_symbolic_success": source_target_symbolic,
                })
                environments = {
                    mode: WorkspaceEnvironment(
                        problem=example.problem,
                        expected_answer=example.expected_answer,
                        worker=worker_client,
                        turn_limit=contract.turn_limit,
                        max_depth=contract.max_depth,
                        child_context_mode=CHILD_CONTEXT_PRECEDING_SIBLINGS,
                        temperature=contract.worker_temperature,
                        max_tokens=contract.worker_max_tokens,
                        base_seed=(
                            contract.worker_seed
                            + case.source_episode_index * contract.turn_limit
                        ),
                        root_result_mode=ROOT_RESULT_VERIFIER_READY,
                        revise_result_mode=mode,
                        action_availability_mode=(
                            ACTION_AVAILABILITY_ACTIVE_FRONTIER
                        ),
                    )
                    for mode in PAIRED_REVISE_MODES
                }
                requests = {
                    mode: build_worker_request(
                        case.state,
                        case.action,
                        child_context_mode=CHILD_CONTEXT_PRECEDING_SIBLINGS,
                        root_result_mode=ROOT_RESULT_VERIFIER_READY,
                        revise_result_mode=mode,
                    )
                    for mode in PAIRED_REVISE_MODES
                }
                _assert_treatment_isolated(
                    requests[REVISE_RESULT_OPEN_OBJECT],
                    requests[REVISE_RESULT_ANSWER_CONTAINER],
                )
                order = _branch_order(case_index)
                positions = {mode: index for index, mode in enumerate(order)}
                case_containers: dict[str, bool] = {}
                case_generation_complete: dict[str, bool] = {}
                case_symbolic: dict[str, bool | None] = {}
                call_ids: dict[str, str] = {}

                for mode in order:
                    environment = environments[mode]
                    if case.action not in environment.legal_actions(case.state):
                        raise RuntimeError("sealed REVISE action became illegal")
                    role = _ROLE_BY_MODE[mode]
                    offset = recorder.count(role)
                    worker_client.attempt_observer = recorder.callback(role)
                    transition = environment.step(case.state, case.action)
                    if transition.worker_request != requests[mode]:
                        raise RuntimeError("paired REVISE request changed at execution")
                    new_ids = recorder.since(role, offset)
                    if len(new_ids) != 1:
                        raise RuntimeError("paired REVISE generation lacks one call id")
                    call_ids[mode] = new_ids[0]
                    payload = (
                        None
                        if transition.worker_payload_json is None
                        else json.loads(transition.worker_payload_json)
                    )
                    has_container, normalized_answer = _answer_container(payload)
                    generated = _new_candidate(transition)
                    if has_container != (normalized_answer is not None):
                        raise AssertionError("container projection is inconsistent")
                    if mode == REVISE_RESULT_ANSWER_CONTAINER and (
                        has_container != (generated is not None)
                    ):
                        raise RuntimeError(
                            "answer-container transition differs from its contract"
                        )
                    symbolic: bool | None = None
                    if case.target_is_root and generated is not None:
                        symbolic = bool(check_answer(
                            generated.answer,
                            example.expected_answer,
                        ).success)
                        root_symbolic_attempts[mode] += 1
                        root_symbolic_success[mode] += int(symbolic)
                    case_containers[mode] = has_container
                    case_generation_complete[mode] = not transition.generation_error
                    case_symbolic[mode] = symbolic
                    answer_containers[mode] += int(has_container)
                    generation_errors[mode] += int(transition.generation_error)
                    record = transition_record(
                        transition,
                        experiment_id=contract.preregistration_id,
                        arm=mode,
                        episode_index=case.source_episode_index,
                        problem_id=case.problem_id,
                        max_depth=contract.max_depth,
                        worker_call_id=new_ids[0],
                        teacher_call_id=None,
                        teacher_decision=None,
                        action_availability_mode=(
                            ACTION_AVAILABILITY_ACTIVE_FRONTIER
                        ),
                    )
                    record.update({
                        "schema_version": PAIRED_REVISE_TRANSITION_VERSION,
                        "paired_case_index": case_index,
                        "source_episode_index": case.source_episode_index,
                        "source_turn": case.source_turn,
                        "branch_position": positions[mode],
                        "answer_container": has_container,
                        "root_symbolic_success": symbolic,
                        "behavior_cloning_eligible": False,
                    })
                    _write_record(run_handle, record)
                    transition_count += 1

                outcome = _paired_outcome(case_containers)
                paired_outcomes[outcome] += 1
                _write_record(run_handle, {
                    "schema_version": PAIRED_REVISE_CASE_SUMMARY_VERSION,
                    "record_type": "paired_case_summary",
                    "preregistration_id": contract.preregistration_id,
                    "paired_case_index": case_index,
                    "problem_id": case.problem_id,
                    "source_episode_index": case.source_episode_index,
                    "source_turn": case.source_turn,
                    "target_is_root": case.target_is_root,
                    "source_target_symbolic_success": source_target_symbolic,
                    "answer_containers": case_containers,
                    "generation_complete": case_generation_complete,
                    "root_symbolic_success": case_symbolic,
                    "paired_outcome": outcome,
                    "worker_call_ids": call_ids,
                }, flush=True)

            run_summary = {
                "schema_version": PAIRED_REVISE_SUMMARY_VERSION,
                "record_type": "run_summary",
                "preregistration_id": contract.preregistration_id,
                "run_contract_sha256": contract.sha256,
                "case_count": len(prepared.cases),
                "transition_count": transition_count,
                "answer_containers_by_mode": dict(answer_containers),
                "generation_errors_by_mode": dict(generation_errors),
                "root_symbolic_attempts_by_mode": dict(root_symbolic_attempts),
                "root_symbolic_success_by_mode": dict(root_symbolic_success),
                "paired_outcomes": dict(sorted(paired_outcomes.items())),
                "usage": {"worker": _counter_delta(worker_before, worker_client)},
            }
            _write_record(run_handle, run_summary)
            run_handle.flush()
            diagnostic_handle.flush()
            os.fsync(run_handle.fileno())
            os.fsync(diagnostic_handle.fileno())
    finally:
        worker_client.attempt_observer = old_observer

    _finalize_no_clobber((
        (partial_diagnostic, final_diagnostic),
        (partial_run, final_run),
    ))
    return PairedReviseCollection(
        run_path=final_run,
        diagnostic_path=final_diagnostic,
        case_count=len(prepared.cases),
        transition_count=transition_count,
        answer_containers_by_mode=dict(answer_containers),
        root_symbolic_success_by_mode=dict(root_symbolic_success),
        run_contract_sha256=contract.sha256,
    )


__all__ = [
    "PAIRED_REVISE_BRANCH_ORDER_VERSION",
    "PAIRED_REVISE_CASE_SUMMARY_VERSION",
    "PAIRED_REVISE_CASE_VERSION",
    "PAIRED_REVISE_COLLECTION_PURPOSE",
    "PAIRED_REVISE_CONTAINER_MEASURE_VERSION",
    "PAIRED_REVISE_CONTRACT_VERSION",
    "PAIRED_REVISE_DIAGNOSTIC_VERSION",
    "PAIRED_REVISE_HEADER_VERSION",
    "PAIRED_REVISE_MODES",
    "PAIRED_REVISE_SEED_POLICY_VERSION",
    "PAIRED_REVISE_SOURCE_ACTION_VERSION",
    "PAIRED_REVISE_SOURCE_VERSION",
    "PAIRED_REVISE_SUMMARY_VERSION",
    "PAIRED_REVISE_TRANSITION_VERSION",
    "PairedReviseCollection",
    "PairedReviseContract",
    "PreparedReviseReplays",
    "ReviseReplayCase",
    "build_paired_revise_contract",
    "collect_paired_revise_run",
    "prepare_revise_replays",
]
