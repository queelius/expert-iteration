"""Paired two-key prompt calibration following the sealed E43 census."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TextIO

from ..llm_client import OpenAICompatibleLLMClient, SCHEMA_VALIDATION_STRICT_V1
from .experiment import (
    ModelIdentity,
    _AttemptRecorder,
    _counter_delta,
    _counter_snapshot,
    _finalize_no_clobber,
    _validate_client_binding,
    _validate_output_paths,
)
from .records import canonical_json_bytes, canonical_json_sha256
from .target_value import (
    TARGET_ACTION_VERSION,
    TARGET_EVALUATION_BINDING_VERSION,
    TARGET_PROBLEM_SOURCE_VERSION,
    TARGET_REDUCER_VERSION,
    TARGET_SCHEMA_VERSION,
    TARGET_STATE_VERSION,
    TWO_KEY_PROMPT_VERSION,
    Candidate,
    TargetAction,
    TargetProblem,
    candidate_semantically_correct,
    reduce_branch_answers,
    replace_branch,
    target_evaluation_binding_sha256,
    target_problem_source_sha256,
    target_state_record,
    terminal_reward,
    two_key_branch_request,
    two_key_root_request,
)
from .target_value_experiment import (
    LogicalCall,
    _call_record,
    _call_worker,
    _logical_seed,
    _root_answer,
    file_sha256,
)


CALIBRATION_CONTRACT_VERSION = "workspace-two-key-calibration-contract-v1"
CALIBRATION_HEADER_VERSION = "workspace-two-key-calibration-header-v1"
CALIBRATION_TRANSITION_VERSION = "workspace-two-key-transition-v1"
CALIBRATION_CASE_VERSION = "workspace-two-key-case-v1"
CALIBRATION_SUMMARY_VERSION = "workspace-two-key-summary-v1"
CALIBRATION_DIAGNOSTIC_VERSION = "workspace-two-key-diagnostic-header-v1"
CALIBRATION_ANALYSIS_VERSION = "workspace-two-key-analysis-v1"
CALIBRATION_BASELINE_VERSION = "workspace-two-key-baseline-selection-v1"
CALIBRATION_SEED_VERSION = "workspace-target-value-seeds-v1"

CALL_KEYS = (
    "initial_root",
    "initial_branch_0",
    "initial_branch_1",
    "retry_root",
    "retry_branch_0",
    "retry_branch_1",
)


def _sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase sha256")


def _baseline_key(
    problem_id: str,
    phase: str,
    target: str,
    repetition: int | None,
) -> str:
    if phase == "initial" and repetition is None:
        return f"initial_{target}"
    if phase == "retry" and repetition == 0:
        return f"retry_{target}"
    raise ValueError("call is outside the calibration baseline")


@dataclass(frozen=True, slots=True)
class BaselineSelection:
    run_sha256: str
    run_contract_sha256: str
    problem_ids: tuple[str, ...]
    acceptance: Mapping[str, Mapping[str, bool]]
    selection_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.run_sha256, "baseline run sha256")
        _sha256(self.run_contract_sha256, "baseline contract sha256")
        _sha256(self.selection_sha256, "baseline selection sha256")


def prepare_e43_baseline(
    *,
    run_path: str | Path,
    expected_run_sha256: str,
    expected_run_contract_sha256: str,
    problems: tuple[TargetProblem, ...],
) -> BaselineSelection:
    """Extract the six matched E43 call outcomes for each calibration case."""
    _sha256(expected_run_sha256, "expected baseline run sha256")
    _sha256(
        expected_run_contract_sha256,
        "expected baseline contract sha256",
    )
    if file_sha256(run_path) != expected_run_sha256:
        raise ValueError("baseline run bytes differ from the expected digest")
    problem_ids = tuple(problem.problem_id for problem in problems)
    wanted = set(problem_ids)
    acceptance: dict[str, dict[str, bool]] = {
        problem_id: {} for problem_id in problem_ids
    }
    observed_contract: str | None = None
    with Path(run_path).open("r", encoding="ascii", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ValueError(
                    f"baseline line {line_number} lacks a trailing newline"
                )
            record = json.loads(line)
            if type(record) is not dict:
                raise ValueError("baseline record must be an object")
            if record.get("record_type") == "run_header":
                observed_contract = str(record.get("run_contract_sha256"))
            if record.get("record_type") != "logical_worker_call":
                continue
            problem_id = record.get("problem_id")
            if problem_id not in wanted:
                continue
            phase = str(record.get("phase"))
            target = str(record.get("target"))
            repetition = record.get("repetition")
            if not (
                (phase == "initial" and repetition is None)
                or (phase == "retry" and repetition == 0)
            ):
                continue
            key = _baseline_key(problem_id, phase, target, repetition)
            bucket = acceptance[problem_id]
            if key in bucket:
                raise ValueError("baseline selection has a duplicate call")
            error = record.get("generation_error")
            if type(error) is not bool:
                raise ValueError("baseline generation flag must be boolean")
            bucket[key] = not error
    if observed_contract != expected_run_contract_sha256:
        raise ValueError("baseline run contract differs from the expected digest")
    for problem_id, bucket in acceptance.items():
        if tuple(sorted(bucket)) != tuple(sorted(CALL_KEYS)):
            raise ValueError(
                f"baseline calls are incomplete for {problem_id}"
            )
    selection_record = {
        "schema_version": CALIBRATION_BASELINE_VERSION,
        "baseline_run_sha256": expected_run_sha256,
        "baseline_run_contract_sha256": expected_run_contract_sha256,
        "problem_ids": list(problem_ids),
        "acceptance": {
            problem_id: {
                key: acceptance[problem_id][key] for key in CALL_KEYS
            }
            for problem_id in problem_ids
        },
    }
    selection_sha256 = canonical_json_sha256(selection_record)
    return BaselineSelection(
        run_sha256=expected_run_sha256,
        run_contract_sha256=expected_run_contract_sha256,
        problem_ids=problem_ids,
        acceptance=acceptance,
        selection_sha256=selection_sha256,
    )


@dataclass(frozen=True, slots=True)
class PromptCalibrationContract:
    preregistration_id: str
    dataset_sha256: str
    problem_source_sha256: str
    evaluation_binding_sha256: str
    problem_ids: tuple[str, ...]
    baseline_run_sha256: str
    baseline_run_contract_sha256: str
    baseline_selection_sha256: str
    worker: ModelIdentity
    worker_temperature: float = 0.2
    worker_max_tokens: int = 256
    base_seed: int = 430_000

    def __post_init__(self) -> None:
        if not isinstance(self.preregistration_id, str) or not (
            self.preregistration_id.strip()
        ):
            raise ValueError("preregistration id must be nonempty")
        for name in (
            "dataset_sha256",
            "problem_source_sha256",
            "evaluation_binding_sha256",
            "baseline_run_sha256",
            "baseline_run_contract_sha256",
            "baseline_selection_sha256",
        ):
            _sha256(getattr(self, name), name)
        if len(self.problem_ids) != 12 or len(set(self.problem_ids)) != 12:
            raise ValueError("calibration requires 12 unique problems")
        if not isinstance(self.worker, ModelIdentity):
            raise ValueError("worker identity is invalid")

    def header(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema_version": CALIBRATION_HEADER_VERSION,
            "record_type": "run_header",
            "run_contract_version": CALIBRATION_CONTRACT_VERSION,
            "preregistration_id": self.preregistration_id,
            "problem_source": {
                "dataset_sha256": self.dataset_sha256,
                "projection_schema_version": TARGET_PROBLEM_SOURCE_VERSION,
                "projection_sha256": self.problem_source_sha256,
                "evaluation_binding_version": (
                    TARGET_EVALUATION_BINDING_VERSION
                ),
                "evaluation_binding_sha256": self.evaluation_binding_sha256,
                "problem_ids": list(self.problem_ids),
            },
            "baseline": {
                "run_sha256": self.baseline_run_sha256,
                "run_contract_sha256": self.baseline_run_contract_sha256,
                "selection_schema_version": CALIBRATION_BASELINE_VERSION,
                "selection_sha256": self.baseline_selection_sha256,
            },
            "contracts": {
                "state": TARGET_STATE_VERSION,
                "action": TARGET_ACTION_VERSION,
                "reducer": TARGET_REDUCER_VERSION,
                "candidate_schema": TARGET_SCHEMA_VERSION,
                "schema_validation": SCHEMA_VALIDATION_STRICT_V1,
                "prompt": TWO_KEY_PROMPT_VERSION,
                "seed_policy": CALIBRATION_SEED_VERSION,
                "analysis": CALIBRATION_ANALYSIS_VERSION,
            },
            "worker": self.worker.to_record(),
            "settings": {
                "temperature": self.worker_temperature,
                "max_tokens": self.worker_max_tokens,
                "base_seed": self.base_seed,
                "retry_repetition": 0,
            },
        }
        body["run_contract_sha256"] = canonical_json_sha256(body)
        return body

    @property
    def sha256(self) -> str:
        return str(self.header()["run_contract_sha256"])


def build_prompt_calibration_contract(
    *,
    preregistration_id: str,
    dataset_path: str | Path,
    problems: tuple[TargetProblem, ...],
    baseline: BaselineSelection,
    worker: ModelIdentity,
) -> PromptCalibrationContract:
    if baseline.problem_ids != tuple(problem.problem_id for problem in problems):
        raise ValueError("baseline problem ids differ from calibration source")
    return PromptCalibrationContract(
        preregistration_id=preregistration_id,
        dataset_sha256=file_sha256(dataset_path),
        problem_source_sha256=target_problem_source_sha256(problems),
        evaluation_binding_sha256=target_evaluation_binding_sha256(problems),
        problem_ids=baseline.problem_ids,
        baseline_run_sha256=baseline.run_sha256,
        baseline_run_contract_sha256=baseline.run_contract_sha256,
        baseline_selection_sha256=baseline.selection_sha256,
        worker=worker,
    )


@dataclass(frozen=True, slots=True)
class CalibrationCase:
    problem_id: str
    accepted: Mapping[str, bool]
    baseline_accepted: Mapping[str, bool]
    direct_reward: int
    structured_submit_reward: int
    retry_rewards: Mapping[TargetAction, int]
    initial_branch_correct: tuple[bool, bool]

    def __post_init__(self) -> None:
        if set(self.accepted) != set(CALL_KEYS):
            raise ValueError("calibration case acceptance is incomplete")
        if set(self.baseline_accepted) != set(CALL_KEYS):
            raise ValueError("calibration baseline acceptance is incomplete")
        if set(self.retry_rewards) != {
            TargetAction.RETRY_ROOT,
            TargetAction.RETRY_BRANCH_0,
            TargetAction.RETRY_BRANCH_1,
        }:
            raise ValueError("calibration retry rewards are incomplete")


def analyze_prompt_calibration(
    cases: tuple[CalibrationCase, ...],
) -> dict[str, object]:
    if len(cases) != 12:
        raise ValueError("calibration analysis requires 12 cases")
    accepted_counts = {
        key: sum(int(case.accepted[key]) for case in cases)
        for key in CALL_KEYS
    }
    baseline_counts = {
        key: sum(int(case.baseline_accepted[key]) for case in cases)
        for key in CALL_KEYS
    }
    new_initial_branches = (
        accepted_counts["initial_branch_0"]
        + accepted_counts["initial_branch_1"]
    )
    old_initial_branches = (
        baseline_counts["initial_branch_0"]
        + baseline_counts["initial_branch_1"]
    )
    matched_gain = (new_initial_branches - old_initial_branches) / 24
    root_accepted = (
        accepted_counts["initial_root"] + accepted_counts["retry_root"]
    )
    integration_passed = (
        accepted_counts["initial_branch_0"] >= 11
        and accepted_counts["initial_branch_1"] >= 11
        and accepted_counts["retry_branch_0"] >= 11
        and accepted_counts["retry_branch_1"] >= 11
        and root_accepted >= 20
        and matched_gain >= 0.80
    )
    submit_solved = sum(case.structured_submit_reward for case in cases)
    regime_passed = 3 <= submit_solved <= 10
    return {
        "schema_version": CALIBRATION_ANALYSIS_VERSION,
        "case_count": len(cases),
        "accepted_counts": accepted_counts,
        "baseline_accepted_counts": baseline_counts,
        "matched_initial_branch_acceptance_gain": matched_gain,
        "root_accepted_count": root_accepted,
        "terminal_solved": {
            "direct_initial": sum(case.direct_reward for case in cases),
            "structured_submit": submit_solved,
            "retry_root": sum(
                case.retry_rewards[TargetAction.RETRY_ROOT] for case in cases
            ),
            "retry_branch_0": sum(
                case.retry_rewards[TargetAction.RETRY_BRANCH_0]
                for case in cases
            ),
            "retry_branch_1": sum(
                case.retry_rewards[TargetAction.RETRY_BRANCH_1]
                for case in cases
            ),
        },
        "initial_branch_semantic_correct": {
            "branch_0": sum(
                int(case.initial_branch_correct[0]) for case in cases
            ),
            "branch_1": sum(
                int(case.initial_branch_correct[1]) for case in cases
            ),
        },
        "gates": {
            "integration_passed": integration_passed,
            "regime_passed": regime_passed,
            "all_passed": integration_passed and regime_passed,
        },
        "behavior_cloning_eligible": False,
    }


def _write_record(
    handle: TextIO,
    record: Mapping[str, object],
    *,
    flush: bool = False,
) -> None:
    encoded = canonical_json_bytes(dict(record))
    if b'"expected_answer"' in encoded or b'"solution_set"' in encoded:
        raise ValueError("published calibration record leaked a hidden field")
    handle.write(encoded.decode("ascii") + "\n")
    if flush:
        handle.flush()


def _accepted(outcome: LogicalCall) -> bool:
    return not outcome.generation_error


@dataclass(frozen=True, slots=True)
class PromptCalibrationResult:
    run_path: Path
    diagnostic_path: Path
    logical_call_count: int
    transition_count: int
    run_contract_sha256: str
    analysis: Mapping[str, object]


def collect_prompt_calibration_run(
    *,
    run_path: str | Path,
    diagnostic_path: str | Path,
    dataset_path: str | Path,
    contract: PromptCalibrationContract,
    baseline: BaselineSelection,
    problems: tuple[TargetProblem, ...],
    worker_client: OpenAICompatibleLLMClient,
) -> PromptCalibrationResult:
    """Collect the fixed 72-call E43R1 prompt calibration."""
    if type(worker_client) is not OpenAICompatibleLLMClient:
        raise TypeError("worker client must be OpenAI-compatible")
    _validate_client_binding(contract.worker, worker_client, role="worker")
    if file_sha256(dataset_path) != contract.dataset_sha256:
        raise ValueError("dataset bytes differ from the calibration contract")
    if target_problem_source_sha256(problems) != contract.problem_source_sha256:
        raise ValueError("problem projection differs from calibration contract")
    if target_evaluation_binding_sha256(
        problems
    ) != contract.evaluation_binding_sha256:
        raise ValueError("evaluation binding differs from calibration contract")
    if baseline.selection_sha256 != contract.baseline_selection_sha256:
        raise ValueError("baseline selection differs from calibration contract")
    if baseline.problem_ids != contract.problem_ids:
        raise ValueError("baseline ids differ from calibration contract")

    final_run = Path(run_path)
    final_diagnostic = Path(diagnostic_path)
    partial_run, partial_diagnostic = _validate_output_paths(
        final_run, final_diagnostic
    )
    worker_before = _counter_snapshot(worker_client)
    old_observer = worker_client.attempt_observer
    logical_calls = 0
    transitions = 0
    generation_errors = 0
    cases: list[CalibrationCase] = []
    try:
        with partial_run.open(
            "x", encoding="ascii", newline="\n"
        ) as run_handle, partial_diagnostic.open(
            "x", encoding="ascii", newline="\n"
        ) as diagnostic_handle:
            _write_record(run_handle, contract.header())
            _write_record(diagnostic_handle, {
                "schema_version": CALIBRATION_DIAGNOSTIC_VERSION,
                "record_type": "diagnostic_header",
                "preregistration_id": contract.preregistration_id,
                "run_contract_sha256": contract.sha256,
            }, flush=True)
            recorder = _AttemptRecorder(diagnostic_handle, roles=("worker",))
            worker_client.attempt_observer = recorder.callback("worker")

            for problem_index, problem in enumerate(problems):
                _write_record(run_handle, {
                    "schema_version": TARGET_STATE_VERSION,
                    "record_type": "problem_source_state",
                    "preregistration_id": contract.preregistration_id,
                    "problem_index": problem_index,
                    "problem_id": problem.problem_id,
                    "problem": problem.problem,
                    "question_type": problem.question_type,
                    "branches": [
                        branch.source_record() for branch in problem.branches
                    ],
                })
                calls: dict[str, LogicalCall] = {}

                request = two_key_root_request(problem)
                direct = _call_worker(
                    client=worker_client,
                    recorder=recorder,
                    request=request,
                    seed=_logical_seed(
                        contract, problem_index, "initial_root"
                    ),
                    contract=contract,
                )
                calls["initial_root"] = direct
                _write_record(run_handle, _call_record(
                    contract=contract,
                    problem=problem,
                    problem_index=problem_index,
                    phase="initial",
                    target="root",
                    repetition=None,
                    request=request,
                    outcome=direct,
                ))

                initial_branch_calls: list[LogicalCall] = []
                for branch_index in (0, 1):
                    request = two_key_branch_request(problem, branch_index)
                    outcome = _call_worker(
                        client=worker_client,
                        recorder=recorder,
                        request=request,
                        seed=_logical_seed(
                            contract,
                            problem_index,
                            "initial_branch",
                            branch_index=branch_index,
                        ),
                        contract=contract,
                    )
                    key = f"initial_branch_{branch_index}"
                    calls[key] = outcome
                    initial_branch_calls.append(outcome)
                    _write_record(run_handle, _call_record(
                        contract=contract,
                        problem=problem,
                        problem_index=problem_index,
                        phase="initial",
                        target=f"branch_{branch_index}",
                        repetition=None,
                        request=request,
                        outcome=outcome,
                    ))

                initial_candidates = (
                    initial_branch_calls[0].candidate,
                    initial_branch_calls[1].candidate,
                )
                initial_state = target_state_record(
                    problem, initial_candidates
                )
                initial_root = reduce_branch_answers(
                    initial_candidates, problem.question_type
                )
                structured_reward = terminal_reward(initial_root, problem)
                retry_rewards: dict[TargetAction, int] = {}

                for action in (
                    TargetAction.RETRY_ROOT,
                    TargetAction.RETRY_BRANCH_0,
                    TargetAction.RETRY_BRANCH_1,
                ):
                    if action is TargetAction.RETRY_ROOT:
                        target = "root"
                        request = two_key_root_request(problem)
                        seed = _logical_seed(
                            contract,
                            problem_index,
                            "retry_root",
                            repetition=0,
                        )
                    else:
                        branch_index = (
                            0
                            if action is TargetAction.RETRY_BRANCH_0
                            else 1
                        )
                        target = f"branch_{branch_index}"
                        request = two_key_branch_request(
                            problem,
                            branch_index,
                            current=initial_candidates[branch_index],
                        )
                        seed = _logical_seed(
                            contract,
                            problem_index,
                            "retry_branch",
                            branch_index=branch_index,
                            repetition=0,
                        )
                    outcome = _call_worker(
                        client=worker_client,
                        recorder=recorder,
                        request=request,
                        seed=seed,
                        contract=contract,
                    )
                    calls[action.value] = outcome
                    _write_record(run_handle, _call_record(
                        contract=contract,
                        problem=problem,
                        problem_index=problem_index,
                        phase="retry",
                        target=target,
                        repetition=0,
                        request=request,
                        outcome=outcome,
                    ))
                    if action is TargetAction.RETRY_ROOT:
                        resulting_root = _root_answer(outcome.candidate)
                    else:
                        replacement_index = (
                            0
                            if action is TargetAction.RETRY_BRANCH_0
                            else 1
                        )
                        resulting_candidates = replace_branch(
                            initial_candidates,
                            replacement_index,
                            outcome.candidate,
                        )
                        resulting_root = reduce_branch_answers(
                            resulting_candidates, problem.question_type
                        )
                    reward = terminal_reward(resulting_root, problem)
                    retry_rewards[action] = reward
                    _write_record(run_handle, {
                        "schema_version": CALIBRATION_TRANSITION_VERSION,
                        "record_type": "retry_transition",
                        "preregistration_id": contract.preregistration_id,
                        "problem_index": problem_index,
                        "problem_id": problem.problem_id,
                        "initial_state_sha256": canonical_json_sha256(
                            initial_state
                        ),
                        "action": action.value,
                        "target": target,
                        "repetition": 0,
                        "worker_call_id": outcome.call_id,
                        "resulting_root_candidate": resulting_root,
                        "reward": reward,
                    })
                    transitions += 1

                logical_calls += len(calls)
                generation_errors += sum(
                    int(call.generation_error) for call in calls.values()
                )
                accepted = {
                    key: _accepted(calls[key]) for key in CALL_KEYS
                }
                baseline_accepted = dict(
                    baseline.acceptance[problem.problem_id]
                )
                branch_correct = tuple(
                    candidate_semantically_correct(call.candidate, branch)
                    for call, branch in zip(
                        initial_branch_calls, problem.branches, strict=True
                    )
                )
                case = CalibrationCase(
                    problem_id=problem.problem_id,
                    accepted=accepted,
                    baseline_accepted=baseline_accepted,
                    direct_reward=terminal_reward(
                        _root_answer(direct.candidate), problem
                    ),
                    structured_submit_reward=structured_reward,
                    retry_rewards=retry_rewards,
                    initial_branch_correct=(
                        branch_correct[0], branch_correct[1]
                    ),
                )
                cases.append(case)
                _write_record(run_handle, {
                    "schema_version": CALIBRATION_CASE_VERSION,
                    "record_type": "case_summary",
                    "preregistration_id": contract.preregistration_id,
                    "problem_index": problem_index,
                    "problem_id": problem.problem_id,
                    "initial_state": initial_state,
                    "accepted": accepted,
                    "baseline_accepted": baseline_accepted,
                    "direct_reward": case.direct_reward,
                    "structured_submit_reward": structured_reward,
                    "retry_rewards": {
                        action.value: retry_rewards[action]
                        for action in retry_rewards
                    },
                    "initial_branch_correct": list(
                        case.initial_branch_correct
                    ),
                }, flush=True)

            analysis = analyze_prompt_calibration(tuple(cases))
            _write_record(run_handle, {
                "schema_version": CALIBRATION_SUMMARY_VERSION,
                "record_type": "run_summary",
                "preregistration_id": contract.preregistration_id,
                "run_contract_sha256": contract.sha256,
                "problem_count": len(problems),
                "logical_call_count": logical_calls,
                "transition_count": transitions,
                "generation_error_count": generation_errors,
                "usage": _counter_delta(worker_before, worker_client),
                "analysis": analysis,
            })
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
    return PromptCalibrationResult(
        run_path=final_run,
        diagnostic_path=final_diagnostic,
        logical_call_count=logical_calls,
        transition_count=transitions,
        run_contract_sha256=contract.sha256,
        analysis=analysis,
    )


__all__ = [
    "CALL_KEYS",
    "BaselineSelection",
    "CalibrationCase",
    "PromptCalibrationContract",
    "PromptCalibrationResult",
    "analyze_prompt_calibration",
    "build_prompt_calibration_contract",
    "collect_prompt_calibration_run",
    "prepare_e43_baseline",
]
