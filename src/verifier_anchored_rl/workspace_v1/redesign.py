"""Versioned collection boundary for the baseline-anchored redesign."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

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
    ROOT_RESULT_CONTRACT_VERSION,
    ROOT_RESULT_FREE,
    ROOT_RESULT_MODES,
    TEACHER_PRESENTATION_VERSION,
    WORKER_PROMPT_ROOT_RESULT_VERSION,
    WORKER_PROMPT_VERSION,
)
from .model import (
    ACTION_AVAILABILITY_ALL,
    ACTION_SCHEMA_VERSION,
    REWARD_CONTRACT_VERSION,
    STATE_SCHEMA_VERSION,
    TRANSITION_SCHEMA_VERSION,
    Action,
    Verb,
    validate_action_availability_mode,
)
from .policies import (
    INITIALIZATION_DIRECT_CANDIDATE_PREFIX,
    INITIALIZATION_EMPTY_ROOT,
    INITIALIZATION_MODES,
    DirectPolicy,
    FixedDecompositionPolicy,
    WorkspacePolicy,
    run_baseline_anchored_episode,
    run_policy_episode,
)
from .records import (
    DIAGNOSTIC_ATTEMPT_VERSION,
    action_record,
    canonical_json_sha256,
    transition_record,
)
from .runtime import (
    SUBMIT_RECEIPT_VERSION,
    WORKER_EXECUTION_REDESIGN_CONTRACT_VERSION,
    Transition,
    WorkspaceEnvironment,
)


REDESIGN_RUN_CONTRACT_VERSION = "workspace-baseline-anchored-run-contract-v2"
REDESIGN_RUN_HEADER_VERSION = "workspace-redesign-run-header-v1"
REDESIGN_TRANSITION_RECORD_VERSION = "workspace-redesign-transition-record-v1"
REDESIGN_EPISODE_SUMMARY_VERSION = "workspace-redesign-episode-summary-v1"
REDESIGN_RUN_SUMMARY_VERSION = "workspace-redesign-run-summary-v1"
REDESIGN_DIAGNOSTIC_HEADER_VERSION = (
    "workspace-redesign-diagnostic-header-v1"
)
REDESIGN_SEED_POLICY_VERSION = "episode-turn-offset-v1"
REDESIGN_POLICY_NAMES = frozenset({
    "direct",
    "fixed_decomposition",
    "teacher",
})
DECISION_SOURCE_PREFIX = "deterministic_prefix"
DECISION_SOURCE_DIRECT = "direct_control"
DECISION_SOURCE_FIXED = "fixed_policy"
DECISION_SOURCE_TEACHER = "teacher"
DECISION_SOURCES = frozenset({
    DECISION_SOURCE_PREFIX,
    DECISION_SOURCE_DIRECT,
    DECISION_SOURCE_FIXED,
    DECISION_SOURCE_TEACHER,
})


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
class RedesignRunContract:
    preregistration_id: str
    arm: str
    policy_name: str
    problem_source_sha256: str
    evaluation_binding_sha256: str
    problem_ids: tuple[str, ...]
    turn_limit: int
    max_depth: int
    initialization_mode: str
    child_context_mode: str
    root_result_mode: str
    action_availability_mode: str
    teacher_choice_mode: str
    worker: ModelIdentity
    worker_temperature: float
    worker_max_tokens: int
    worker_seed: int
    teacher: ModelIdentity | None = None
    teacher_temperature: float | None = None
    teacher_max_tokens: int | None = None
    teacher_seed: int | None = None

    def __post_init__(self) -> None:
        _nonempty(self.preregistration_id, "preregistration id")
        _nonempty(self.arm, "arm")
        if self.policy_name not in REDESIGN_POLICY_NAMES:
            raise ValueError("unknown redesign policy")
        _sha256(self.problem_source_sha256, "problem source sha256")
        _sha256(self.evaluation_binding_sha256, "evaluation binding sha256")
        if not self.problem_ids or len(self.problem_ids) != len(
            set(self.problem_ids)
        ):
            raise ValueError("problem ids must be nonempty and unique")
        if type(self.turn_limit) is not int or self.turn_limit <= 0:
            raise ValueError("turn limit must be positive")
        if (
            self.initialization_mode == INITIALIZATION_DIRECT_CANDIDATE_PREFIX
            and self.turn_limit < 2
        ):
            raise ValueError("direct prefix requires at least two turns")
        if type(self.max_depth) is not int or self.max_depth <= 0:
            raise ValueError("max depth must be positive")
        if self.initialization_mode not in INITIALIZATION_MODES:
            raise ValueError("unknown initialization mode")
        if self.child_context_mode not in CHILD_CONTEXT_MODES:
            raise ValueError("unknown child context mode")
        if self.root_result_mode not in ROOT_RESULT_MODES:
            raise ValueError("unknown root result mode")
        validate_action_availability_mode(self.action_availability_mode)
        if self.teacher_choice_mode not in TEACHER_CHOICE_MODES:
            raise ValueError("unknown teacher choice mode")
        if not isinstance(self.worker, ModelIdentity):
            raise ValueError("worker identity is invalid")
        _sampling(
            self.worker_temperature,
            self.worker_max_tokens,
            self.worker_seed,
            "worker",
        )
        needs_teacher = self.policy_name == "teacher"
        if needs_teacher != isinstance(self.teacher, ModelIdentity):
            raise ValueError("teacher identity must match the policy")
        teacher_values = (
            self.teacher_temperature,
            self.teacher_max_tokens,
            self.teacher_seed,
        )
        if needs_teacher != all(value is not None for value in teacher_values):
            raise ValueError("teacher settings must match the policy")
        if needs_teacher:
            _sampling(
                self.teacher_temperature,
                self.teacher_max_tokens,
                self.teacher_seed,
                "teacher",
            )

    def header(self) -> dict[str, object]:
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
            "schema_version": REDESIGN_RUN_HEADER_VERSION,
            "record_type": "run_header",
            "run_contract_version": REDESIGN_RUN_CONTRACT_VERSION,
            "preregistration_id": self.preregistration_id,
            "arm": self.arm,
            "policy_name": self.policy_name,
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
                "transition_record": REDESIGN_TRANSITION_RECORD_VERSION,
                "reward": REWARD_CONTRACT_VERSION,
                "worker_execution": (
                    WORKER_EXECUTION_REDESIGN_CONTRACT_VERSION
                ),
                "worker_prompt": worker_prompt_version,
                "direct_prompt": DIRECT_PROMPT_VERSION,
                "hybrid_lens": HYBRID_LENS_VERSION,
                "child_answer_context": CHILD_CONTEXT_CONTRACT_VERSION,
                "root_result": ROOT_RESULT_CONTRACT_VERSION,
                "initialization": self.initialization_mode,
                "action_availability": self.action_availability_mode,
                "observation": observation_version,
                "submit_receipt": SUBMIT_RECEIPT_VERSION,
                "teacher_prompt": teacher_prompt_version,
                "teacher_output": teacher_output_version,
                "teacher_fallback": TEACHER_FALLBACK_VERSION,
                "teacher_presentation": TEACHER_PRESENTATION_VERSION,
                "seed_policy": REDESIGN_SEED_POLICY_VERSION,
            },
            "settings": {
                "turn_limit": self.turn_limit,
                "max_depth": self.max_depth,
                "initialization_mode": self.initialization_mode,
                "child_context_mode": self.child_context_mode,
                "root_result_mode": self.root_result_mode,
                "action_availability_mode": self.action_availability_mode,
                "teacher_choice_mode": self.teacher_choice_mode,
                "worker_temperature": self.worker_temperature,
                "worker_max_tokens": self.worker_max_tokens,
                "worker_seed": self.worker_seed,
                "teacher_temperature": self.teacher_temperature,
                "teacher_max_tokens": self.teacher_max_tokens,
                "teacher_seed": self.teacher_seed,
            },
            "models": {
                "worker": self.worker.to_record(),
                "teacher": (
                    None if self.teacher is None else self.teacher.to_record()
                ),
            },
        }
        body["run_contract_sha256"] = canonical_json_sha256(body)
        return body

    @property
    def sha256(self) -> str:
        return str(self.header()["run_contract_sha256"])


@dataclass(frozen=True, slots=True)
class RedesignCollectionResult:
    run_path: Path
    diagnostic_path: Path
    episode_count: int
    transition_count: int
    controller_decision_count: int
    solved_count: int
    run_contract_sha256: str


def _policy(
    contract: RedesignRunContract,
    teacher_client: OpenAICompatibleLLMClient | None,
    *,
    episode_index: int,
) -> WorkspacePolicy:
    if contract.policy_name == "direct":
        return DirectPolicy()
    if contract.policy_name == "fixed_decomposition":
        return FixedDecompositionPolicy()
    if teacher_client is None:
        raise ValueError("teacher policy requires a teacher client")
    if (
        contract.teacher_temperature is None
        or contract.teacher_max_tokens is None
        or contract.teacher_seed is None
    ):
        raise AssertionError("validated teacher settings disappeared")
    return LLMTeacherPolicy(
        client=teacher_client,
        max_depth=contract.max_depth,
        temperature=contract.teacher_temperature,
        max_tokens=contract.teacher_max_tokens,
        base_seed=(
            contract.teacher_seed + episode_index * contract.turn_limit
        ),
        action_availability_mode=contract.action_availability_mode,
        choice_mode=contract.teacher_choice_mode,
    )


def _decision_source(contract: RedesignRunContract) -> str:
    if contract.policy_name == "direct":
        return DECISION_SOURCE_DIRECT
    if contract.policy_name == "fixed_decomposition":
        return DECISION_SOURCE_FIXED
    return DECISION_SOURCE_TEACHER


def redesign_transition_record(
    transition: Transition,
    *,
    contract: RedesignRunContract,
    episode_index: int,
    problem_id: str,
    worker_call_id: str | None,
    teacher_call_id: str | None,
    teacher_decision: TeacherDecision | None,
    decision_source: str,
) -> dict[str, object]:
    if decision_source not in DECISION_SOURCES:
        raise ValueError("unknown decision source")
    if (decision_source == DECISION_SOURCE_TEACHER) != (
        teacher_decision is not None and teacher_call_id is not None
    ):
        raise ValueError("teacher provenance differs from decision source")
    if decision_source == DECISION_SOURCE_PREFIX and (
        transition.action != Action(Verb.ANSWER, 0)
        or teacher_decision is not None
        or teacher_call_id is not None
    ):
        raise ValueError("deterministic prefix provenance is invalid")
    record = transition_record(
        transition,
        experiment_id=contract.preregistration_id,
        arm=contract.arm,
        episode_index=episode_index,
        problem_id=problem_id,
        max_depth=contract.max_depth,
        worker_call_id=worker_call_id,
        teacher_call_id=teacher_call_id,
        teacher_decision=teacher_decision,
        action_availability_mode=contract.action_availability_mode,
    )
    record["schema_version"] = REDESIGN_TRANSITION_RECORD_VERSION
    record["decision_source"] = decision_source
    record["behavior_cloning_eligible"] = (
        decision_source == DECISION_SOURCE_TEACHER
        and teacher_decision is not None
        and not teacher_decision.fallback
    )
    return record


def collect_redesign_run(
    *,
    run_path: str | Path,
    diagnostic_path: str | Path,
    contract: RedesignRunContract,
    examples: tuple[ProblemExample, ...],
    worker_client: OpenAICompatibleLLMClient,
    teacher_client: OpenAICompatibleLLMClient | None = None,
) -> RedesignCollectionResult:
    """Collect one redesign arm and publish only after complete coverage."""
    if not isinstance(contract, RedesignRunContract):
        raise TypeError("collection requires a RedesignRunContract")
    if type(worker_client) is not OpenAICompatibleLLMClient:
        raise TypeError("worker client must be OpenAI-compatible")
    if (contract.policy_name == "teacher") != (teacher_client is not None):
        raise ValueError("teacher client must match the selected policy")
    if teacher_client is worker_client:
        raise ValueError("worker and teacher require distinct client objects")
    if teacher_client is not None and type(
        teacher_client
    ) is not OpenAICompatibleLLMClient:
        raise TypeError("teacher client must be OpenAI-compatible")
    _validate_client_binding(contract.worker, worker_client, role="worker")
    if teacher_client is not None:
        if contract.teacher is None:
            raise AssertionError("validated teacher identity disappeared")
        _validate_client_binding(
            contract.teacher, teacher_client, role="teacher"
        )
    if contract.problem_ids != tuple(
        example.problem_id for example in examples
    ):
        raise ValueError("run contract problem ids differ from source")
    if contract.problem_source_sha256 != problem_source_sha256(examples):
        raise ValueError("run contract differs from problem projection")
    if contract.evaluation_binding_sha256 != evaluation_binding_sha256(
        examples
    ):
        raise ValueError("run contract differs from evaluation binding")

    final_run = Path(run_path)
    final_diagnostic = Path(diagnostic_path)
    partial_run, partial_diagnostic = _validate_output_paths(
        final_run, final_diagnostic
    )
    diagnostic_header = {
        "schema_version": REDESIGN_DIAGNOSTIC_HEADER_VERSION,
        "record_type": "diagnostic_header",
        "preregistration_id": contract.preregistration_id,
        "arm": contract.arm,
        "run_contract_sha256": contract.sha256,
        "attempt_schema_version": DIAGNOSTIC_ATTEMPT_VERSION,
    }
    worker_before = _counter_snapshot(worker_client)
    teacher_before = (
        None if teacher_client is None else _counter_snapshot(teacher_client)
    )
    old_worker_observer = worker_client.attempt_observer
    old_teacher_observer = (
        None if teacher_client is None else teacher_client.attempt_observer
    )
    solved_count = 0
    transition_count = 0
    controller_decision_count = 0
    generation_errors = 0
    prefix_generation_errors = 0
    teacher_fallbacks = 0
    action_counts: Counter[str] = Counter()
    controller_action_counts: Counter[str] = Counter()

    try:
        with partial_run.open(
            "x", encoding="ascii", newline="\n"
        ) as run_handle, partial_diagnostic.open(
            "x", encoding="ascii", newline="\n"
        ) as diagnostic_handle:
            _write_record(run_handle, contract.header())
            _write_record(diagnostic_handle, diagnostic_header, flush=True)
            recorder = _AttemptRecorder(diagnostic_handle)
            worker_client.attempt_observer = recorder.callback("worker")
            if teacher_client is not None:
                teacher_client.attempt_observer = recorder.callback("teacher")

            for episode_index, example in enumerate(examples):
                worker_offset = recorder.count("worker")
                teacher_offset = recorder.count("teacher")
                policy = _policy(
                    contract,
                    teacher_client,
                    episode_index=episode_index,
                )
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
                    action_availability_mode=(
                        contract.action_availability_mode
                    ),
                )
                if (
                    contract.initialization_mode
                    == INITIALIZATION_DIRECT_CANDIDATE_PREFIX
                ):
                    anchored = run_baseline_anchored_episode(
                        environment, policy
                    )
                    transitions = anchored.transitions
                    prefix_count = 1
                else:
                    episode = run_policy_episode(environment, policy)
                    transitions = episode.transitions
                    prefix_count = 0

                worker_ids = recorder.since("worker", worker_offset)
                teacher_ids = recorder.since("teacher", teacher_offset)
                worker_transitions = tuple(
                    transition
                    for transition in transitions
                    if transition.worker_request is not None
                )
                if len(worker_ids) != len(worker_transitions):
                    raise RuntimeError(
                        "worker call ids differ from generative transitions"
                    )
                policy_transition_count = len(transitions) - prefix_count
                if contract.policy_name == "teacher":
                    if not isinstance(policy, LLMTeacherPolicy):
                        raise AssertionError("teacher policy changed type")
                    decisions = tuple(policy.decisions)
                    if len(decisions) != policy_transition_count:
                        raise RuntimeError(
                            "teacher decisions differ from policy transitions"
                        )
                    if len(teacher_ids) != policy_transition_count:
                        raise RuntimeError(
                            "teacher calls differ from policy transitions"
                        )
                else:
                    decisions = ()
                    if teacher_ids:
                        raise RuntimeError("control policy made teacher calls")

                worker_iterator = iter(worker_ids)
                for local_index, transition in enumerate(transitions):
                    worker_call_id = (
                        next(worker_iterator)
                        if transition.worker_request is not None
                        else None
                    )
                    if local_index < prefix_count:
                        decision = None
                        teacher_call_id = None
                        decision_source = DECISION_SOURCE_PREFIX
                    else:
                        policy_index = local_index - prefix_count
                        decision = (
                            decisions[policy_index] if decisions else None
                        )
                        teacher_call_id = (
                            teacher_ids[policy_index] if teacher_ids else None
                        )
                        decision_source = _decision_source(contract)
                        controller_decision_count += 1
                        controller_action_counts[
                            transition.action.verb.value
                        ] += 1
                    record = redesign_transition_record(
                        transition,
                        contract=contract,
                        episode_index=episode_index,
                        problem_id=example.problem_id,
                        worker_call_id=worker_call_id,
                        teacher_call_id=teacher_call_id,
                        teacher_decision=decision,
                        decision_source=decision_source,
                    )
                    _write_record(run_handle, record)
                    action_counts[transition.action.verb.value] += 1
                    generation_errors += int(transition.generation_error)
                    prefix_generation_errors += int(
                        decision_source == DECISION_SOURCE_PREFIX
                        and transition.generation_error
                    )
                    teacher_fallbacks += int(
                        decision is not None and decision.fallback
                    )
                    transition_count += 1
                try:
                    next(worker_iterator)
                except StopIteration:
                    pass
                else:
                    raise AssertionError("unused worker call id")

                final_state = transitions[-1].next_state
                solved_count += final_state.reward
                summary = {
                    "schema_version": REDESIGN_EPISODE_SUMMARY_VERSION,
                    "record_type": "episode_summary",
                    "preregistration_id": contract.preregistration_id,
                    "arm": contract.arm,
                    "episode_index": episode_index,
                    "problem_id": example.problem_id,
                    "turn_count": len(transitions),
                    "controller_decision_count": policy_transition_count,
                    "reward": final_state.reward,
                    "terminal_reason": (
                        "submit"
                        if final_state.history[-1].action.verb is Verb.SUBMIT
                        else "horizon"
                    ),
                    "prefix_worker_call_id": (
                        worker_ids[0] if prefix_count else None
                    ),
                    "worker_call_ids": list(worker_ids),
                    "teacher_call_ids": list(teacher_ids),
                }
                _write_record(run_handle, summary, flush=True)

            run_summary = {
                "schema_version": REDESIGN_RUN_SUMMARY_VERSION,
                "record_type": "run_summary",
                "preregistration_id": contract.preregistration_id,
                "arm": contract.arm,
                "run_contract_sha256": contract.sha256,
                "episode_count": len(examples),
                "transition_count": transition_count,
                "controller_decision_count": controller_decision_count,
                "solved_count": solved_count,
                "generation_error_count": generation_errors,
                "prefix_generation_error_count": prefix_generation_errors,
                "teacher_fallback_count": teacher_fallbacks,
                "action_counts": dict(sorted(action_counts.items())),
                "controller_action_counts": dict(
                    sorted(controller_action_counts.items())
                ),
                "usage": {
                    "worker": _counter_delta(worker_before, worker_client),
                    "teacher": (
                        None
                        if teacher_client is None or teacher_before is None
                        else _counter_delta(teacher_before, teacher_client)
                    ),
                },
            }
            _write_record(run_handle, run_summary)
            run_handle.flush()
            diagnostic_handle.flush()
            os.fsync(run_handle.fileno())
            os.fsync(diagnostic_handle.fileno())
    finally:
        worker_client.attempt_observer = old_worker_observer
        if teacher_client is not None:
            teacher_client.attempt_observer = old_teacher_observer

    _finalize_no_clobber((
        (partial_diagnostic, final_diagnostic),
        (partial_run, final_run),
    ))
    return RedesignCollectionResult(
        run_path=final_run,
        diagnostic_path=final_diagnostic,
        episode_count=len(examples),
        transition_count=transition_count,
        controller_decision_count=controller_decision_count,
        solved_count=solved_count,
        run_contract_sha256=contract.sha256,
    )


__all__ = [
    "DECISION_SOURCE_DIRECT",
    "DECISION_SOURCE_FIXED",
    "DECISION_SOURCE_PREFIX",
    "DECISION_SOURCE_TEACHER",
    "DECISION_SOURCES",
    "REDESIGN_DIAGNOSTIC_HEADER_VERSION",
    "REDESIGN_EPISODE_SUMMARY_VERSION",
    "REDESIGN_POLICY_NAMES",
    "REDESIGN_RUN_CONTRACT_VERSION",
    "REDESIGN_RUN_HEADER_VERSION",
    "REDESIGN_RUN_SUMMARY_VERSION",
    "REDESIGN_SEED_POLICY_VERSION",
    "REDESIGN_TRANSITION_RECORD_VERSION",
    "RedesignCollectionResult",
    "RedesignRunContract",
    "collect_redesign_run",
    "redesign_transition_record",
]
