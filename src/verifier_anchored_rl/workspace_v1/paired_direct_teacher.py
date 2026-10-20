"""Exact shared-direct-Candidate comparison against teacher continuation."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..llm_client import OpenAICompatibleLLMClient
from .controller import (
    TEACHER_CHOICE_MENU_INDEX,
    TEACHER_CHOICE_MODES,
    TEACHER_CHOICE_VERB_TARGET,
    TEACHER_FALLBACK_VERSION,
    TEACHER_INDEX_OUTPUT_VERSION,
    TEACHER_INDEX_PROMPT_VERSION,
    TEACHER_OUTPUT_VERSION,
    TEACHER_PROMPT_VERSION,
    LLMTeacherPolicy,
    TeacherDecision,
)
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
    CHILD_CONTEXT_MODES,
    DIRECT_PROMPT_VERSION,
    HYBRID_LENS_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    REVISE_RESULT_ANSWER_CONTAINER,
    REVISE_RESULT_CONTRACT_VERSION,
    REVISE_RESULT_OPEN_OBJECT,
    ROOT_RESULT_CONTRACT_VERSION,
    ROOT_RESULT_FREE,
    ROOT_RESULT_MODES,
    TEACHER_PRESENTATION_VERSION,
    WORKER_PROMPT_ROOT_RESULT_VERSION,
    WORKER_PROMPT_REVISE_ANSWER_VERSION,
    WORKER_PROMPT_VERSION,
    direct_request,
    validate_revise_result_mode,
)
from .model import (
    ACTION_AVAILABILITY_ALL,
    ACTION_SCHEMA_VERSION,
    REWARD_CONTRACT_VERSION,
    STATE_SCHEMA_VERSION,
    TRANSITION_SCHEMA_VERSION,
    Action,
    CandidateNode,
    Verb,
    protected_direct_candidate_id,
    validate_action_availability_mode,
)
from .policies import INITIALIZATION_DIRECT_CANDIDATE_PREFIX
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


PAIRED_DIRECT_TEACHER_CONTRACT_VERSION = (
    "workspace-paired-direct-teacher-contract-v1"
)
PAIRED_DIRECT_TEACHER_REVISE_CONTRACT_VERSION = (
    "workspace-paired-direct-teacher-contract-v1.1"
)
PAIRED_DIRECT_TEACHER_HEADER_VERSION = (
    "workspace-paired-direct-teacher-header-v1"
)
PAIRED_DIRECT_TEACHER_REVISE_HEADER_VERSION = (
    "workspace-paired-direct-teacher-header-v1.1"
)
PAIRED_DIRECT_TEACHER_DIAGNOSTIC_VERSION = (
    "workspace-paired-direct-teacher-diagnostic-header-v1"
)
PAIRED_DIRECT_TEACHER_TRANSITION_VERSION = (
    "workspace-paired-direct-teacher-transition-v1"
)
PAIRED_DIRECT_TEACHER_CASE_SUMMARY_VERSION = (
    "workspace-paired-direct-teacher-case-summary-v1"
)
PAIRED_DIRECT_TEACHER_SUMMARY_VERSION = (
    "workspace-paired-direct-teacher-summary-v1"
)
PAIRED_DIRECT_TEACHER_SEED_POLICY_VERSION = "episode-turn-offset-v1"
PAIRED_DIRECT_TEACHER_PREFIX_VERSION = "shared-direct-candidate-once-v1"
PAIRED_DIRECT_TEACHER_DIRECT_VERSION = "submit-shared-candidate-v1"
PAIRED_DIRECT_TEACHER_CONTINUATION_VERSION = "llm-teacher-continuation-v1"
PAIRED_DIRECT_TEACHER_COLLECTION_PURPOSE = "evaluation-only-v1"

BRANCH_DIRECT = "direct_submit"
BRANCH_TEACHER = "teacher_continuation"
PAIRED_DIRECT_TEACHER_BRANCHES = (BRANCH_DIRECT, BRANCH_TEACHER)

ROLE_SHARED_PREFIX_WORKER = "shared_prefix_worker"
ROLE_TEACHER_WORKER = "teacher_continuation_worker"
ROLE_TEACHER_CONTROLLER = "teacher_controller"
PAIRED_DIRECT_TEACHER_ROLES = (
    ROLE_SHARED_PREFIX_WORKER,
    ROLE_TEACHER_WORKER,
    ROLE_TEACHER_CONTROLLER,
)

DECISION_SOURCE_PREFIX = "deterministic_prefix"
DECISION_SOURCE_DIRECT = "direct_control"
DECISION_SOURCE_TEACHER = "teacher"


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _sha256(value: str, name: str) -> str:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase sha256")
    return value


def _sampling(
    temperature: float,
    max_tokens: int,
    seed: int,
    role: str,
) -> None:
    if type(temperature) is not float or temperature < 0.0:
        raise ValueError(f"{role} temperature must be nonnegative")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError(f"{role} max tokens must be positive")
    if type(seed) is not int or seed < 0:
        raise ValueError(f"{role} seed must be nonnegative")


@dataclass(frozen=True, slots=True)
class PairedDirectTeacherContract:
    preregistration_id: str
    problem_source_sha256: str
    evaluation_binding_sha256: str
    problem_ids: tuple[str, ...]
    turn_limit: int
    max_depth: int
    child_context_mode: str
    root_result_mode: str
    action_availability_mode: str
    teacher_choice_mode: str
    worker: ModelIdentity
    worker_temperature: float
    worker_max_tokens: int
    worker_seed: int
    teacher: ModelIdentity
    teacher_temperature: float
    teacher_max_tokens: int
    teacher_seed: int
    revise_result_mode: str = REVISE_RESULT_OPEN_OBJECT

    def __post_init__(self) -> None:
        _nonempty(self.preregistration_id, "preregistration id")
        _sha256(self.problem_source_sha256, "problem source sha256")
        _sha256(self.evaluation_binding_sha256, "evaluation binding sha256")
        if not self.problem_ids or len(self.problem_ids) != len(
            set(self.problem_ids)
        ):
            raise ValueError("problem ids must be nonempty and unique")
        if type(self.turn_limit) is not int or self.turn_limit < 2:
            raise ValueError("paired direct-teacher turn limit must be at least two")
        if type(self.max_depth) is not int or self.max_depth <= 0:
            raise ValueError("paired direct-teacher max depth must be positive")
        if self.child_context_mode not in CHILD_CONTEXT_MODES:
            raise ValueError("unknown child context mode")
        if self.root_result_mode not in ROOT_RESULT_MODES:
            raise ValueError("unknown root result mode")
        validate_action_availability_mode(self.action_availability_mode)
        if self.teacher_choice_mode not in TEACHER_CHOICE_MODES:
            raise ValueError("unknown teacher choice mode")
        validate_revise_result_mode(self.revise_result_mode)
        if not isinstance(self.worker, ModelIdentity):
            raise ValueError("worker identity is invalid")
        if not isinstance(self.teacher, ModelIdentity):
            raise ValueError("teacher identity is invalid")
        _sampling(
            self.worker_temperature,
            self.worker_max_tokens,
            self.worker_seed,
            "worker",
        )
        _sampling(
            self.teacher_temperature,
            self.teacher_max_tokens,
            self.teacher_seed,
            "teacher",
        )

    def header(self) -> dict[str, object]:
        revised_contract = (
            self.revise_result_mode == REVISE_RESULT_ANSWER_CONTAINER
        )
        observation_version = (
            OBSERVATION_SCHEMA_VERSION
            if self.action_availability_mode == ACTION_AVAILABILITY_ALL
            else ACTIVE_FRONTIER_OBSERVATION_SCHEMA_VERSION
        )
        worker_prompt_version = (
            WORKER_PROMPT_VERSION
            if self.root_result_mode == ROOT_RESULT_FREE
            else WORKER_PROMPT_ROOT_RESULT_VERSION
        )
        teacher_prompt_version = (
            TEACHER_PROMPT_VERSION
            if self.teacher_choice_mode == TEACHER_CHOICE_VERB_TARGET
            else TEACHER_INDEX_PROMPT_VERSION
        )
        teacher_output_version = (
            TEACHER_OUTPUT_VERSION
            if self.teacher_choice_mode == TEACHER_CHOICE_VERB_TARGET
            else TEACHER_INDEX_OUTPUT_VERSION
        )
        body: dict[str, object] = {
            "schema_version": (
                PAIRED_DIRECT_TEACHER_REVISE_HEADER_VERSION
                if revised_contract
                else PAIRED_DIRECT_TEACHER_HEADER_VERSION
            ),
            "record_type": "run_header",
            "run_contract_version": (
                PAIRED_DIRECT_TEACHER_REVISE_CONTRACT_VERSION
                if revised_contract
                else PAIRED_DIRECT_TEACHER_CONTRACT_VERSION
            ),
            "preregistration_id": self.preregistration_id,
            "problem_source": {
                "schema_version": PROBLEM_SOURCE_VERSION,
                "sha256": self.problem_source_sha256,
                "evaluation_binding_version": EVALUATION_BINDING_VERSION,
                "evaluation_binding_sha256": self.evaluation_binding_sha256,
                "problem_ids": list(self.problem_ids),
            },
            "contracts": {
                "state": STATE_SCHEMA_VERSION,
                "action": ACTION_SCHEMA_VERSION,
                "transition": TRANSITION_SCHEMA_VERSION,
                "transition_record": (
                    PAIRED_DIRECT_TEACHER_TRANSITION_VERSION
                ),
                "reward": REWARD_CONTRACT_VERSION,
                "worker_execution": WORKER_EXECUTION_REDESIGN_CONTRACT_VERSION,
                "worker_prompt": worker_prompt_version,
                "direct_prompt": DIRECT_PROMPT_VERSION,
                "hybrid_lens": HYBRID_LENS_VERSION,
                "child_answer_context": CHILD_CONTEXT_CONTRACT_VERSION,
                "root_result": ROOT_RESULT_CONTRACT_VERSION,
                "initialization": INITIALIZATION_DIRECT_CANDIDATE_PREFIX,
                "shared_prefix": PAIRED_DIRECT_TEACHER_PREFIX_VERSION,
                "direct_branch": PAIRED_DIRECT_TEACHER_DIRECT_VERSION,
                "teacher_branch": PAIRED_DIRECT_TEACHER_CONTINUATION_VERSION,
                "action_availability": self.action_availability_mode,
                "observation": observation_version,
                "submit_receipt": SUBMIT_RECEIPT_VERSION,
                "teacher_prompt": teacher_prompt_version,
                "teacher_output": teacher_output_version,
                "teacher_fallback": TEACHER_FALLBACK_VERSION,
                "teacher_presentation": TEACHER_PRESENTATION_VERSION,
                "seed_policy": PAIRED_DIRECT_TEACHER_SEED_POLICY_VERSION,
                "collection_purpose": (
                    PAIRED_DIRECT_TEACHER_COLLECTION_PURPOSE
                ),
            },
            "settings": {
                "turn_limit": self.turn_limit,
                "max_depth": self.max_depth,
                "child_context_mode": self.child_context_mode,
                "root_result_mode": self.root_result_mode,
                "action_availability_mode": self.action_availability_mode,
                "teacher_choice_mode": self.teacher_choice_mode,
                "branches": list(PAIRED_DIRECT_TEACHER_BRANCHES),
                "worker_temperature": self.worker_temperature,
                "worker_max_tokens": self.worker_max_tokens,
                "worker_seed": self.worker_seed,
                "teacher_temperature": self.teacher_temperature,
                "teacher_max_tokens": self.teacher_max_tokens,
                "teacher_seed": self.teacher_seed,
            },
            "models": {
                "worker": self.worker.to_record(),
                "teacher": self.teacher.to_record(),
            },
        }
        if revised_contract:
            contracts = body["contracts"]
            settings = body["settings"]
            if type(contracts) is not dict or type(settings) is not dict:
                raise AssertionError("paired contract containers changed type")
            contracts["revise_result"] = REVISE_RESULT_CONTRACT_VERSION
            contracts["worker_prompt_revise"] = (
                WORKER_PROMPT_REVISE_ANSWER_VERSION
            )
            settings["revise_result_mode"] = self.revise_result_mode
        body["run_contract_sha256"] = canonical_json_sha256(body)
        return body

    @property
    def sha256(self) -> str:
        return str(self.header()["run_contract_sha256"])


def build_paired_direct_teacher_contract(
    *,
    preregistration_id: str,
    examples: tuple[ProblemExample, ...],
    turn_limit: int,
    max_depth: int,
    child_context_mode: str,
    root_result_mode: str,
    action_availability_mode: str,
    teacher_choice_mode: str,
    worker: ModelIdentity,
    worker_temperature: float,
    worker_max_tokens: int,
    worker_seed: int,
    teacher: ModelIdentity,
    teacher_temperature: float,
    teacher_max_tokens: int,
    teacher_seed: int,
    revise_result_mode: str = REVISE_RESULT_OPEN_OBJECT,
) -> PairedDirectTeacherContract:
    return PairedDirectTeacherContract(
        preregistration_id=preregistration_id,
        problem_source_sha256=problem_source_sha256(examples),
        evaluation_binding_sha256=evaluation_binding_sha256(examples),
        problem_ids=tuple(example.problem_id for example in examples),
        turn_limit=turn_limit,
        max_depth=max_depth,
        child_context_mode=child_context_mode,
        root_result_mode=root_result_mode,
        action_availability_mode=action_availability_mode,
        teacher_choice_mode=teacher_choice_mode,
        worker=worker,
        worker_temperature=worker_temperature,
        worker_max_tokens=worker_max_tokens,
        worker_seed=worker_seed,
        teacher=teacher,
        teacher_temperature=teacher_temperature,
        teacher_max_tokens=teacher_max_tokens,
        teacher_seed=teacher_seed,
        revise_result_mode=revise_result_mode,
    )


@dataclass(frozen=True, slots=True)
class PairedDirectTeacherCollection:
    run_path: Path
    diagnostic_path: Path
    case_count: int
    transition_count: int
    solved_by_branch: Mapping[str, int]
    run_contract_sha256: str


def _paired_outcome(direct_reward: int, teacher_reward: int) -> str:
    if direct_reward and teacher_reward:
        return "both"
    if direct_reward:
        return "direct_only"
    if teacher_reward:
        return "teacher_only"
    return "neither"


def _terminal_reason(state) -> str:
    if state.history[-1].action.verb is Verb.SUBMIT:
        return "submit"
    return "horizon"


def _paired_transition_record(
    transition: Transition,
    *,
    contract: PairedDirectTeacherContract,
    episode_index: int,
    problem_id: str,
    branch: str,
    phase: str,
    decision_source: str,
    worker_call_id: str | None,
    teacher_call_id: str | None,
    teacher_decision: TeacherDecision | None,
) -> dict[str, object]:
    record = transition_record(
        transition,
        experiment_id=contract.preregistration_id,
        arm=branch,
        episode_index=episode_index,
        problem_id=problem_id,
        max_depth=contract.max_depth,
        worker_call_id=worker_call_id,
        teacher_call_id=teacher_call_id,
        teacher_decision=teacher_decision,
        action_availability_mode=contract.action_availability_mode,
    )
    record.update({
        "schema_version": PAIRED_DIRECT_TEACHER_TRANSITION_VERSION,
        "paired_case_index": episode_index,
        "paired_phase": phase,
        "decision_source": decision_source,
        "behavior_cloning_eligible": False,
    })
    return record


def collect_paired_direct_teacher_run(
    *,
    run_path: str | Path,
    diagnostic_path: str | Path,
    contract: PairedDirectTeacherContract,
    examples: tuple[ProblemExample, ...],
    worker_client: OpenAICompatibleLLMClient,
    teacher_client: OpenAICompatibleLLMClient,
) -> PairedDirectTeacherCollection:
    """Share one direct generation, then submit or continue with a teacher."""
    if not isinstance(contract, PairedDirectTeacherContract):
        raise TypeError("paired collection requires its run contract")
    if type(worker_client) is not OpenAICompatibleLLMClient:
        raise TypeError("worker client must be OpenAI-compatible")
    if type(teacher_client) is not OpenAICompatibleLLMClient:
        raise TypeError("teacher client must be OpenAI-compatible")
    if worker_client is teacher_client:
        raise ValueError("worker and teacher require distinct clients")
    _validate_client_binding(contract.worker, worker_client, role="worker")
    _validate_client_binding(contract.teacher, teacher_client, role="teacher")
    if contract.problem_ids != tuple(
        example.problem_id for example in examples
    ):
        raise ValueError("run contract problem ids differ from source")
    if contract.problem_source_sha256 != problem_source_sha256(examples):
        raise ValueError("run contract differs from problem projection")
    if contract.evaluation_binding_sha256 != evaluation_binding_sha256(
        examples
    ):
        raise ValueError("run contract differs from hidden-label binding")

    expected_contract = build_paired_direct_teacher_contract(
        preregistration_id=contract.preregistration_id,
        examples=examples,
        turn_limit=contract.turn_limit,
        max_depth=contract.max_depth,
        child_context_mode=contract.child_context_mode,
        root_result_mode=contract.root_result_mode,
        action_availability_mode=contract.action_availability_mode,
        teacher_choice_mode=contract.teacher_choice_mode,
        worker=contract.worker,
        worker_temperature=contract.worker_temperature,
        worker_max_tokens=contract.worker_max_tokens,
        worker_seed=contract.worker_seed,
        teacher=contract.teacher,
        teacher_temperature=contract.teacher_temperature,
        teacher_max_tokens=contract.teacher_max_tokens,
        teacher_seed=contract.teacher_seed,
        revise_result_mode=contract.revise_result_mode,
    )
    if expected_contract != contract:
        raise ValueError("paired contract differs from selected source")

    final_run = Path(run_path)
    final_diagnostic = Path(diagnostic_path)
    partial_run, partial_diagnostic = _validate_output_paths(
        final_run,
        final_diagnostic,
    )
    diagnostic_header = {
        "schema_version": PAIRED_DIRECT_TEACHER_DIAGNOSTIC_VERSION,
        "record_type": "diagnostic_header",
        "preregistration_id": contract.preregistration_id,
        "run_contract_sha256": contract.sha256,
        "attempt_schema_version": DIAGNOSTIC_ATTEMPT_VERSION,
        "roles": list(PAIRED_DIRECT_TEACHER_ROLES),
    }
    worker_before = _counter_snapshot(worker_client)
    teacher_before = _counter_snapshot(teacher_client)
    old_worker_observer = worker_client.attempt_observer
    old_teacher_observer = teacher_client.attempt_observer
    solved = Counter({branch: 0 for branch in PAIRED_DIRECT_TEACHER_BRANCHES})
    paired_outcomes: Counter[str] = Counter()
    action_counts = {
        branch: Counter() for branch in PAIRED_DIRECT_TEACHER_BRANCHES
    }
    prefix_calls = 0
    teacher_calls = 0
    continuation_worker_calls = 0
    transition_count = 0
    teacher_fallbacks = 0
    continuation_generation_errors = 0

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
                roles=PAIRED_DIRECT_TEACHER_ROLES,
            )

            for episode_index, example in enumerate(examples):
                environment = WorkspaceEnvironment(
                    problem=example.problem,
                    expected_answer=example.expected_answer,
                    worker=worker_client,
                    turn_limit=contract.turn_limit,
                    max_depth=contract.max_depth,
                    child_context_mode=contract.child_context_mode,
                    temperature=contract.worker_temperature,
                    max_tokens=contract.worker_max_tokens,
                    base_seed=(
                        contract.worker_seed
                        + episode_index * contract.turn_limit
                    ),
                    root_result_mode=contract.root_result_mode,
                    revise_result_mode=contract.revise_result_mode,
                    action_availability_mode=(
                        contract.action_availability_mode
                    ),
                )
                initial = environment.initial_state()
                prefix_offset = recorder.count(ROLE_SHARED_PREFIX_WORKER)
                worker_client.attempt_observer = recorder.callback(
                    ROLE_SHARED_PREFIX_WORKER
                )
                prefix = environment.step(initial, Action(Verb.ANSWER, 0))
                prefix_ids = recorder.since(
                    ROLE_SHARED_PREFIX_WORKER,
                    prefix_offset,
                )
                if len(prefix_ids) != 1:
                    raise RuntimeError("shared prefix lacks exactly one call id")
                if prefix.worker_request != direct_request(example.problem):
                    raise RuntimeError("shared prefix differs from direct request")
                if prefix.generation_error:
                    raise RuntimeError(
                        "shared direct prefix generation failed; pair unavailable"
                    )
                prefix_calls += 1
                transition_count += 1
                _write_record(run_handle, _paired_transition_record(
                    prefix,
                    contract=contract,
                    episode_index=episode_index,
                    problem_id=example.problem_id,
                    branch="shared_prefix",
                    phase="shared_prefix",
                    decision_source=DECISION_SOURCE_PREFIX,
                    worker_call_id=prefix_ids[0],
                    teacher_call_id=None,
                    teacher_decision=None,
                ))
                shared_state = prefix.next_state
                candidate_id = protected_direct_candidate_id(shared_state)
                if candidate_id is None:
                    raise RuntimeError("shared prefix did not create its Candidate")
                candidate = shared_state.nodes[candidate_id]
                if not isinstance(candidate, CandidateNode):
                    raise AssertionError("protected node is not a Candidate")

                submit = Action(Verb.SUBMIT, candidate_id)
                direct_transition = environment.step(shared_state, submit)
                transition_count += 1
                action_counts[BRANCH_DIRECT][Verb.SUBMIT.value] += 1
                _write_record(run_handle, _paired_transition_record(
                    direct_transition,
                    contract=contract,
                    episode_index=episode_index,
                    problem_id=example.problem_id,
                    branch=BRANCH_DIRECT,
                    phase="direct_submission",
                    decision_source=DECISION_SOURCE_DIRECT,
                    worker_call_id=None,
                    teacher_call_id=None,
                    teacher_decision=None,
                ))

                policy = LLMTeacherPolicy(
                    client=teacher_client,
                    max_depth=contract.max_depth,
                    temperature=contract.teacher_temperature,
                    max_tokens=contract.teacher_max_tokens,
                    base_seed=(
                        contract.teacher_seed
                        + episode_index * contract.turn_limit
                    ),
                    action_availability_mode=(
                        contract.action_availability_mode
                    ),
                    choice_mode=contract.teacher_choice_mode,
                )
                teacher_state = shared_state
                case_teacher_call_ids: list[str] = []
                case_worker_call_ids: list[str] = []
                case_teacher_transitions = 0
                while not teacher_state.terminal:
                    legal = environment.legal_actions(teacher_state)
                    if not legal:
                        raise RuntimeError(
                            "nonterminal teacher branch has no legal action"
                        )
                    teacher_offset = recorder.count(ROLE_TEACHER_CONTROLLER)
                    teacher_client.attempt_observer = recorder.callback(
                        ROLE_TEACHER_CONTROLLER
                    )
                    action = policy.choose(teacher_state, legal)
                    new_teacher_ids = recorder.since(
                        ROLE_TEACHER_CONTROLLER,
                        teacher_offset,
                    )
                    if len(new_teacher_ids) != 1:
                        raise RuntimeError(
                            "teacher decision lacks exactly one call id"
                        )
                    decision = policy.decisions[-1]
                    if decision.selected_action != action:
                        raise RuntimeError("teacher decision/action mismatch")
                    teacher_call_id = new_teacher_ids[0]
                    teacher_calls += 1
                    teacher_fallbacks += int(decision.fallback)
                    case_teacher_call_ids.append(teacher_call_id)

                    worker_offset = recorder.count(ROLE_TEACHER_WORKER)
                    worker_client.attempt_observer = recorder.callback(
                        ROLE_TEACHER_WORKER
                    )
                    transition = environment.step(teacher_state, action)
                    new_worker_ids = recorder.since(
                        ROLE_TEACHER_WORKER,
                        worker_offset,
                    )
                    expected_worker_calls = int(
                        transition.worker_request is not None
                    )
                    if len(new_worker_ids) != expected_worker_calls:
                        raise RuntimeError(
                            "teacher worker calls differ from its transition"
                        )
                    worker_call_id = (
                        new_worker_ids[0] if new_worker_ids else None
                    )
                    if worker_call_id is not None:
                        continuation_worker_calls += 1
                        case_worker_call_ids.append(worker_call_id)
                    continuation_generation_errors += int(
                        transition.generation_error
                    )
                    action_counts[BRANCH_TEACHER][action.verb.value] += 1
                    transition_count += 1
                    case_teacher_transitions += 1
                    _write_record(run_handle, _paired_transition_record(
                        transition,
                        contract=contract,
                        episode_index=episode_index,
                        problem_id=example.problem_id,
                        branch=BRANCH_TEACHER,
                        phase="teacher_continuation",
                        decision_source=DECISION_SOURCE_TEACHER,
                        worker_call_id=worker_call_id,
                        teacher_call_id=teacher_call_id,
                        teacher_decision=decision,
                    ))
                    teacher_state = transition.next_state

                direct_reward = direct_transition.next_state.reward
                teacher_reward = teacher_state.reward
                solved[BRANCH_DIRECT] += direct_reward
                solved[BRANCH_TEACHER] += teacher_reward
                outcome = _paired_outcome(direct_reward, teacher_reward)
                paired_outcomes[outcome] += 1
                _write_record(run_handle, {
                    "schema_version": (
                        PAIRED_DIRECT_TEACHER_CASE_SUMMARY_VERSION
                    ),
                    "record_type": "paired_case_summary",
                    "preregistration_id": contract.preregistration_id,
                    "paired_case_index": episode_index,
                    "problem_id": example.problem_id,
                    "shared_prefix_worker_call_id": prefix_ids[0],
                    "shared_candidate_id": candidate_id,
                    "rewards": {
                        BRANCH_DIRECT: direct_reward,
                        BRANCH_TEACHER: teacher_reward,
                    },
                    "paired_outcome": outcome,
                    "teacher_terminal_reason": _terminal_reason(teacher_state),
                    "teacher_transition_count": case_teacher_transitions,
                    "teacher_call_ids": case_teacher_call_ids,
                    "teacher_worker_call_ids": case_worker_call_ids,
                }, flush=True)

            run_summary = {
                "schema_version": PAIRED_DIRECT_TEACHER_SUMMARY_VERSION,
                "record_type": "run_summary",
                "preregistration_id": contract.preregistration_id,
                "run_contract_sha256": contract.sha256,
                "case_count": len(examples),
                "transition_count": transition_count,
                "shared_prefix_call_count": prefix_calls,
                "teacher_call_count": teacher_calls,
                "teacher_worker_call_count": continuation_worker_calls,
                "solved_by_branch": dict(solved),
                "paired_outcomes": dict(sorted(paired_outcomes.items())),
                "teacher_fallback_count": teacher_fallbacks,
                "teacher_generation_error_count": (
                    continuation_generation_errors
                ),
                "action_counts": {
                    branch: dict(sorted(counts.items()))
                    for branch, counts in action_counts.items()
                },
                "usage": {
                    "worker": _counter_delta(worker_before, worker_client),
                    "teacher": _counter_delta(teacher_before, teacher_client),
                },
            }
            _write_record(run_handle, run_summary)
            run_handle.flush()
            diagnostic_handle.flush()
            os.fsync(run_handle.fileno())
            os.fsync(diagnostic_handle.fileno())
    finally:
        worker_client.attempt_observer = old_worker_observer
        teacher_client.attempt_observer = old_teacher_observer

    _finalize_no_clobber((
        (partial_diagnostic, final_diagnostic),
        (partial_run, final_run),
    ))
    return PairedDirectTeacherCollection(
        run_path=final_run,
        diagnostic_path=final_diagnostic,
        case_count=len(examples),
        transition_count=transition_count,
        solved_by_branch=dict(solved),
        run_contract_sha256=contract.sha256,
    )


__all__ = [
    "BRANCH_DIRECT",
    "BRANCH_TEACHER",
    "PAIRED_DIRECT_TEACHER_BRANCHES",
    "PAIRED_DIRECT_TEACHER_CASE_SUMMARY_VERSION",
    "PAIRED_DIRECT_TEACHER_COLLECTION_PURPOSE",
    "PAIRED_DIRECT_TEACHER_CONTRACT_VERSION",
    "PAIRED_DIRECT_TEACHER_CONTINUATION_VERSION",
    "PAIRED_DIRECT_TEACHER_DIAGNOSTIC_VERSION",
    "PAIRED_DIRECT_TEACHER_DIRECT_VERSION",
    "PAIRED_DIRECT_TEACHER_HEADER_VERSION",
    "PAIRED_DIRECT_TEACHER_PREFIX_VERSION",
    "PAIRED_DIRECT_TEACHER_REVISE_CONTRACT_VERSION",
    "PAIRED_DIRECT_TEACHER_REVISE_HEADER_VERSION",
    "PAIRED_DIRECT_TEACHER_ROLES",
    "PAIRED_DIRECT_TEACHER_SEED_POLICY_VERSION",
    "PAIRED_DIRECT_TEACHER_SUMMARY_VERSION",
    "PAIRED_DIRECT_TEACHER_TRANSITION_VERSION",
    "ROLE_SHARED_PREFIX_WORKER",
    "ROLE_TEACHER_CONTROLLER",
    "ROLE_TEACHER_WORKER",
    "PairedDirectTeacherCollection",
    "PairedDirectTeacherContract",
    "build_paired_direct_teacher_contract",
    "collect_paired_direct_teacher_run",
]
