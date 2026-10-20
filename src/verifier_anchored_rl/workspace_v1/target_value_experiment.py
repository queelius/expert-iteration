"""Hosted collection and paired analysis for fixed-graph target value."""

from __future__ import annotations

import hashlib
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Mapping, TextIO

from ..llm_client import (
    LLMResponseError,
    OpenAICompatibleLLMClient,
    SCHEMA_VALIDATION_STRICT_V1,
)
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
    ACTION_ORDER,
    RETRY_ACTIONS,
    TARGET_ACTION_VERSION,
    TARGET_EVALUATION_BINDING_VERSION,
    TARGET_PROBLEM_SOURCE_VERSION,
    TARGET_REDUCER_VERSION,
    TARGET_ROOT_PROMPT_VERSION,
    TARGET_SCHEMA_VERSION,
    TARGET_STATE_VERSION,
    TARGET_WORKER_PROMPT_VERSION,
    Candidate,
    TargetAction,
    TargetProblem,
    WorkerRequest,
    branch_request,
    candidate_from_payload,
    candidate_semantically_correct,
    parse_integer_answer,
    reduce_branch_answers,
    replace_branch,
    request_sha256,
    root_request,
    target_evaluation_binding_sha256,
    target_problem_source_sha256,
    target_state_record,
    terminal_reward,
)


TARGET_RUN_CONTRACT_VERSION = "workspace-target-value-run-contract-v1"
TARGET_RUN_HEADER_VERSION = "workspace-target-value-run-header-v1"
TARGET_CALL_RECORD_VERSION = "workspace-target-value-call-v1"
TARGET_TRANSITION_RECORD_VERSION = "workspace-target-value-transition-v1"
TARGET_CASE_SUMMARY_VERSION = "workspace-target-value-case-summary-v1"
TARGET_RUN_SUMMARY_VERSION = "workspace-target-value-run-summary-v1"
TARGET_DIAGNOSTIC_HEADER_VERSION = (
    "workspace-target-value-diagnostic-header-v1"
)
TARGET_SEED_POLICY_VERSION = "workspace-target-value-seeds-v1"
TARGET_ANALYSIS_VERSION = "workspace-target-value-analysis-v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase sha256")


@dataclass(frozen=True, slots=True)
class TargetRunContract:
    preregistration_id: str
    dataset_sha256: str
    problem_source_sha256: str
    evaluation_binding_sha256: str
    problem_ids: tuple[str, ...]
    worker: ModelIdentity
    worker_temperature: float
    worker_max_tokens: int
    base_seed: int
    repetitions: int = 4
    discovery_repetitions: tuple[int, ...] = (0, 1)
    evaluation_repetitions: tuple[int, ...] = (2, 3)

    def __post_init__(self) -> None:
        if not isinstance(self.preregistration_id, str) or not (
            self.preregistration_id.strip()
        ):
            raise ValueError("preregistration id must be nonempty text")
        for name in (
            "dataset_sha256",
            "problem_source_sha256",
            "evaluation_binding_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if not self.problem_ids or len(self.problem_ids) != len(
            set(self.problem_ids)
        ):
            raise ValueError("problem ids must be nonempty and unique")
        if not isinstance(self.worker, ModelIdentity):
            raise ValueError("worker identity is invalid")
        if type(self.worker_temperature) is not float or (
            self.worker_temperature < 0.0
        ):
            raise ValueError("worker temperature must be nonnegative")
        if type(self.worker_max_tokens) is not int or self.worker_max_tokens <= 0:
            raise ValueError("worker max tokens must be positive")
        if type(self.base_seed) is not int or self.base_seed < 0:
            raise ValueError("base seed must be nonnegative")
        if type(self.repetitions) is not int or self.repetitions <= 0:
            raise ValueError("repetitions must be positive")
        all_repetitions = tuple(range(self.repetitions))
        combined = self.discovery_repetitions + self.evaluation_repetitions
        if tuple(sorted(combined)) != all_repetitions:
            raise ValueError("discovery and evaluation repetitions must partition")
        if not self.discovery_repetitions or not self.evaluation_repetitions:
            raise ValueError("both repetition splits must be nonempty")

    def header(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema_version": TARGET_RUN_HEADER_VERSION,
            "record_type": "run_header",
            "run_contract_version": TARGET_RUN_CONTRACT_VERSION,
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
            "contracts": {
                "state": TARGET_STATE_VERSION,
                "action": TARGET_ACTION_VERSION,
                "reducer": TARGET_REDUCER_VERSION,
                "branch_prompt": TARGET_WORKER_PROMPT_VERSION,
                "root_prompt": TARGET_ROOT_PROMPT_VERSION,
                "candidate_schema": TARGET_SCHEMA_VERSION,
                "schema_validation": SCHEMA_VALIDATION_STRICT_V1,
                "seed_policy": TARGET_SEED_POLICY_VERSION,
                "analysis": TARGET_ANALYSIS_VERSION,
            },
            "worker": self.worker.to_record(),
            "settings": {
                "temperature": self.worker_temperature,
                "max_tokens": self.worker_max_tokens,
                "base_seed": self.base_seed,
                "repetitions": self.repetitions,
                "discovery_repetitions": list(self.discovery_repetitions),
                "evaluation_repetitions": list(self.evaluation_repetitions),
            },
        }
        body["run_contract_sha256"] = canonical_json_sha256(body)
        return body

    @property
    def sha256(self) -> str:
        return str(self.header()["run_contract_sha256"])


def build_target_run_contract(
    *,
    preregistration_id: str,
    dataset_path: str | Path,
    problems: tuple[TargetProblem, ...],
    worker: ModelIdentity,
    worker_temperature: float = 0.2,
    worker_max_tokens: int = 256,
    base_seed: int = 430_000,
) -> TargetRunContract:
    return TargetRunContract(
        preregistration_id=preregistration_id,
        dataset_sha256=file_sha256(dataset_path),
        problem_source_sha256=target_problem_source_sha256(problems),
        evaluation_binding_sha256=target_evaluation_binding_sha256(problems),
        problem_ids=tuple(problem.problem_id for problem in problems),
        worker=worker,
        worker_temperature=worker_temperature,
        worker_max_tokens=worker_max_tokens,
        base_seed=base_seed,
    )


def _write_record(
    handle: TextIO,
    record: Mapping[str, object],
    *,
    flush: bool = False,
) -> None:
    encoded = canonical_json_bytes(dict(record))
    if b'"expected_answer"' in encoded or b'"solution_set"' in encoded:
        raise ValueError("published record leaked a hidden answer field")
    handle.write(encoded.decode("ascii") + "\n")
    if flush:
        handle.flush()


def _logical_seed(
    contract: TargetRunContract,
    problem_index: int,
    phase: str,
    *,
    branch_index: int | None = None,
    repetition: int | None = None,
) -> int:
    base = contract.base_seed + problem_index * 100
    if phase == "initial_root":
        return base
    if phase == "initial_branch":
        if branch_index not in {0, 1}:
            raise ValueError("initial branch seed requires a branch index")
        return base + 1 + branch_index
    if phase == "retry_root":
        if repetition is None:
            raise ValueError("root retry seed requires a repetition")
        return base + 10 + repetition
    if phase == "retry_branch":
        if branch_index not in {0, 1} or repetition is None:
            raise ValueError("branch retry seed requires branch and repetition")
        return base + 20 + branch_index * 10 + repetition
    raise ValueError("unknown logical call phase")


@dataclass(frozen=True, slots=True)
class LogicalCall:
    candidate: Candidate
    call_id: str
    generation_error: bool
    seed: int
    request_sha256: str


def _call_worker(
    *,
    client: OpenAICompatibleLLMClient,
    recorder: _AttemptRecorder,
    request: WorkerRequest,
    seed: int,
    contract: TargetRunContract,
) -> LogicalCall:
    offset = recorder.count("worker")
    generation_error = False
    try:
        payload = client.complete_json(
            request.system,
            request.user,
            request.schema,
            temperature=contract.worker_temperature,
            max_tokens=contract.worker_max_tokens,
            seed=seed,
            schema_validation_mode=SCHEMA_VALIDATION_STRICT_V1,
        )
        candidate = candidate_from_payload(payload)
    except (LLMResponseError, ValueError) as exc:
        generation_error = True
        candidate = Candidate(
            reasoning="",
            answer=None,
            usable=False,
            error_type=type(exc).__name__,
        )
    call_ids = recorder.since("worker", offset)
    if len(call_ids) != 1:
        raise RuntimeError("one logical worker call must produce one call id")
    return LogicalCall(
        candidate=candidate,
        call_id=call_ids[0],
        generation_error=generation_error,
        seed=seed,
        request_sha256=request_sha256(request),
    )


def _call_record(
    *,
    contract: TargetRunContract,
    problem: TargetProblem,
    problem_index: int,
    phase: str,
    target: str,
    repetition: int | None,
    request: WorkerRequest,
    outcome: LogicalCall,
) -> dict[str, object]:
    return {
        "schema_version": TARGET_CALL_RECORD_VERSION,
        "record_type": "logical_worker_call",
        "preregistration_id": contract.preregistration_id,
        "problem_index": problem_index,
        "problem_id": problem.problem_id,
        "phase": phase,
        "target": target,
        "repetition": repetition,
        "seed": outcome.seed,
        "request_sha256": outcome.request_sha256,
        "request": request.to_record(),
        "candidate": outcome.candidate.to_record(),
        "generation_error": outcome.generation_error,
        "worker_call_id": outcome.call_id,
    }


def _root_answer(candidate: Candidate) -> str:
    if not candidate.usable or candidate.answer is None:
        return ""
    return candidate.answer


def _majority_answer(candidates: tuple[Candidate, ...]) -> str:
    if not candidates:
        raise ValueError("majority requires candidates")
    normalized: list[str] = []
    originals: list[str] = []
    for candidate in candidates:
        answer = _root_answer(candidate)
        originals.append(answer)
        try:
            parsed = parse_integer_answer(candidate.answer)
        except ValueError:
            normalized.append(answer.strip())
        else:
            normalized.append("" if parsed is None else str(parsed))
    counts = Counter(normalized)
    best_count = max(counts.values())
    for key, original in zip(normalized, originals, strict=True):
        if counts[key] == best_count:
            return original
    raise AssertionError("majority tie break failed")


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    problem_id: str
    direct_initial_reward: int
    direct_majority3_reward: int
    action_rewards: Mapping[TargetAction, tuple[int, ...]]
    initial_branch_correct: tuple[bool, bool]

    def __post_init__(self) -> None:
        if set(self.action_rewards) != set(ACTION_ORDER):
            raise ValueError("case evaluation requires every action")
        lengths = {len(values) for values in self.action_rewards.values()}
        if len(lengths) != 1:
            raise ValueError("action rewards must have equal lengths")
        for values in self.action_rewards.values():
            if any(value not in {0, 1} for value in values):
                raise ValueError("action rewards must be binary")


def _action_mean(
    case: CaseEvaluation,
    action: TargetAction,
    repetitions: tuple[int, ...],
) -> float:
    rewards = case.action_rewards[action]
    return mean(rewards[index] for index in repetitions)


def _best_action(scores: Mapping[TargetAction, float]) -> TargetAction:
    best = max(scores.values())
    return next(action for action in ACTION_ORDER if scores[action] == best)


def _first_error_action(case: CaseEvaluation) -> TargetAction:
    if not case.initial_branch_correct[0]:
        return TargetAction.RETRY_BRANCH_0
    if not case.initial_branch_correct[1]:
        return TargetAction.RETRY_BRANCH_1
    return TargetAction.SUBMIT


def _bootstrap_interval(
    differences: tuple[float, ...],
    *,
    seed: int = 43_043,
    samples: int = 10_000,
) -> tuple[float, float]:
    if not differences:
        raise ValueError("bootstrap requires paired differences")
    rng = random.Random(seed)
    n = len(differences)
    estimates = sorted(
        mean(differences[rng.randrange(n)] for _ in range(n))
        for _ in range(samples)
    )
    lower = estimates[int(0.05 * samples)]
    upper = estimates[min(samples - 1, int(0.95 * samples))]
    return lower, upper


def analyze_target_value_cases(
    cases: tuple[CaseEvaluation, ...],
    contract: TargetRunContract,
) -> dict[str, object]:
    """Analyze discovery-selected target value on held-out repetitions."""
    if not cases:
        raise ValueError("target-value analysis requires cases")
    discovery = contract.discovery_repetitions
    evaluation = contract.evaluation_repetitions
    selected_actions: list[TargetAction] = []
    first_error_actions: list[TargetAction] = []
    selected_evaluation: list[float] = []
    first_error_evaluation: list[float] = []
    realized_oracle_evaluation: list[float] = []
    heterogeneous = 0
    positive_target_disagreements = 0

    for case in cases:
        discovery_scores = {
            action: _action_mean(case, action, discovery)
            for action in ACTION_ORDER
        }
        evaluation_scores = {
            action: _action_mean(case, action, evaluation)
            for action in ACTION_ORDER
        }
        selected = _best_action(discovery_scores)
        first_error = _first_error_action(case)
        selected_actions.append(selected)
        first_error_actions.append(first_error)
        selected_evaluation.append(evaluation_scores[selected])
        first_error_evaluation.append(evaluation_scores[first_error])
        realized_oracle_evaluation.append(max(evaluation_scores.values()))
        retry_scores = {evaluation_scores[action] for action in RETRY_ACTIONS}
        heterogeneous += int(len(retry_scores) > 1)
        positive_target_disagreements += int(
            selected is not first_error
            and evaluation_scores[selected] > evaluation_scores[first_error]
        )

    fixed_discovery = {
        action: mean(
            _action_mean(case, action, discovery) for case in cases
        )
        for action in ACTION_ORDER
    }
    fixed_evaluation = {
        action: mean(
            _action_mean(case, action, evaluation) for case in cases
        )
        for action in ACTION_ORDER
    }
    best_fixed = _best_action(fixed_discovery)
    best_fixed_evaluation = fixed_evaluation[best_fixed]
    value_evaluation = mean(selected_evaluation)
    first_error_score = mean(first_error_evaluation)
    paired_differences = tuple(
        selected - _action_mean(case, best_fixed, evaluation)
        for case, selected in zip(cases, selected_evaluation, strict=True)
    )
    ci_lower, ci_upper = _bootstrap_interval(paired_differences)
    submit_accuracy = fixed_evaluation[TargetAction.SUBMIT]
    allocation_difference = value_evaluation - best_fixed_evaluation
    first_error_difference = value_evaluation - first_error_score
    allocation_gate = (
        allocation_difference >= 0.08 and ci_lower > 0.0
    )
    novelty_gate = (
        first_error_difference >= 0.05
        and positive_target_disagreements >= 5
    )
    support_gate = (
        heterogeneous >= 10 and 0.20 <= submit_accuracy <= 0.85
    )

    return {
        "schema_version": TARGET_ANALYSIS_VERSION,
        "case_count": len(cases),
        "accuracies": {
            "direct_initial": mean(
                case.direct_initial_reward for case in cases
            ),
            "structured_submit": submit_accuracy,
            "direct_majority3": mean(
                case.direct_majority3_reward for case in cases
            ),
            "fixed_evaluation": {
                action.value: fixed_evaluation[action]
                for action in ACTION_ORDER
            },
            "best_global_fixed": best_fixed_evaluation,
            "crossfit_value": value_evaluation,
            "hidden_first_error": first_error_score,
            "realized_oracle": mean(realized_oracle_evaluation),
        },
        "selection": {
            "best_global_fixed_action": best_fixed.value,
            "best_global_fixed_discovery_scores": {
                action.value: fixed_discovery[action]
                for action in ACTION_ORDER
            },
            "crossfit_action_counts": dict(sorted(Counter(
                action.value for action in selected_actions
            ).items())),
            "hidden_first_error_action_counts": dict(sorted(Counter(
                action.value for action in first_error_actions
            ).items())),
        },
        "gates": {
            "allocation": {
                "difference": allocation_difference,
                "bootstrap_90_ci": [ci_lower, ci_upper],
                "threshold": 0.08,
                "passed": allocation_gate,
            },
            "novelty": {
                "difference_vs_hidden_first_error": first_error_difference,
                "positive_target_disagreements": (
                    positive_target_disagreements
                ),
                "passed": novelty_gate,
            },
            "support": {
                "heterogeneous_problem_count": heterogeneous,
                "structured_submit_accuracy": submit_accuracy,
                "passed": support_gate,
            },
            "all_passed": allocation_gate and novelty_gate and support_gate,
        },
        "behavior_cloning_eligible": False,
    }


@dataclass(frozen=True, slots=True)
class TargetCollectionResult:
    run_path: Path
    diagnostic_path: Path
    problem_count: int
    logical_call_count: int
    transition_count: int
    run_contract_sha256: str
    analysis: Mapping[str, object]


def collect_target_value_run(
    *,
    run_path: str | Path,
    diagnostic_path: str | Path,
    dataset_path: str | Path,
    contract: TargetRunContract,
    problems: tuple[TargetProblem, ...],
    worker_client: OpenAICompatibleLLMClient,
) -> TargetCollectionResult:
    """Collect all one-call target outcomes and publish paired artifacts."""
    if type(worker_client) is not OpenAICompatibleLLMClient:
        raise TypeError("worker client must be OpenAI-compatible")
    _validate_client_binding(contract.worker, worker_client, role="worker")
    if file_sha256(dataset_path) != contract.dataset_sha256:
        raise ValueError("dataset bytes differ from the run contract")
    if target_problem_source_sha256(problems) != contract.problem_source_sha256:
        raise ValueError("problem projection differs from the run contract")
    if target_evaluation_binding_sha256(
        problems
    ) != contract.evaluation_binding_sha256:
        raise ValueError("evaluation binding differs from the run contract")
    if tuple(problem.problem_id for problem in problems) != contract.problem_ids:
        raise ValueError("problem ids differ from the run contract")

    final_run = Path(run_path)
    final_diagnostic = Path(diagnostic_path)
    partial_run, partial_diagnostic = _validate_output_paths(
        final_run, final_diagnostic
    )
    diagnostic_header = {
        "schema_version": TARGET_DIAGNOSTIC_HEADER_VERSION,
        "record_type": "diagnostic_header",
        "preregistration_id": contract.preregistration_id,
        "run_contract_sha256": contract.sha256,
    }
    worker_before = _counter_snapshot(worker_client)
    old_observer = worker_client.attempt_observer
    logical_call_count = 0
    transition_count = 0
    generation_errors = 0
    cases: list[CaseEvaluation] = []

    try:
        with partial_run.open(
            "x", encoding="ascii", newline="\n"
        ) as run_handle, partial_diagnostic.open(
            "x", encoding="ascii", newline="\n"
        ) as diagnostic_handle:
            _write_record(run_handle, contract.header())
            _write_record(diagnostic_handle, diagnostic_header, flush=True)
            recorder = _AttemptRecorder(diagnostic_handle, roles=("worker",))
            worker_client.attempt_observer = recorder.callback("worker")

            for problem_index, problem in enumerate(problems):
                source_state = {
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
                }
                _write_record(run_handle, source_state)

                direct_request = root_request(problem)
                direct = _call_worker(
                    client=worker_client,
                    recorder=recorder,
                    request=direct_request,
                    seed=_logical_seed(
                        contract, problem_index, "initial_root"
                    ),
                    contract=contract,
                )
                _write_record(run_handle, _call_record(
                    contract=contract,
                    problem=problem,
                    problem_index=problem_index,
                    phase="initial",
                    target="root",
                    repetition=None,
                    request=direct_request,
                    outcome=direct,
                ))
                logical_call_count += 1
                generation_errors += int(direct.generation_error)

                branch_calls: list[LogicalCall] = []
                for branch_index in (0, 1):
                    request = branch_request(problem, branch_index)
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
                    branch_calls.append(outcome)
                    logical_call_count += 1
                    generation_errors += int(outcome.generation_error)

                initial_candidates = (
                    branch_calls[0].candidate,
                    branch_calls[1].candidate,
                )
                initial_state = target_state_record(
                    problem, initial_candidates
                )
                initial_root = reduce_branch_answers(
                    initial_candidates, problem.question_type
                )
                submit_reward = terminal_reward(initial_root, problem)
                rewards: dict[TargetAction, list[int]] = {
                    TargetAction.SUBMIT: [
                        submit_reward for _ in range(contract.repetitions)
                    ],
                    TargetAction.RETRY_ROOT: [],
                    TargetAction.RETRY_BRANCH_0: [],
                    TargetAction.RETRY_BRANCH_1: [],
                }
                root_retries: list[Candidate] = []

                for action in RETRY_ACTIONS:
                    for repetition in range(contract.repetitions):
                        if action is TargetAction.RETRY_ROOT:
                            request = root_request(problem)
                            seed = _logical_seed(
                                contract,
                                problem_index,
                                "retry_root",
                                repetition=repetition,
                            )
                            target = "root"
                        else:
                            branch_index = (
                                0
                                if action is TargetAction.RETRY_BRANCH_0
                                else 1
                            )
                            request = branch_request(
                                problem,
                                branch_index,
                                current=initial_candidates[branch_index],
                            )
                            seed = _logical_seed(
                                contract,
                                problem_index,
                                "retry_branch",
                                branch_index=branch_index,
                                repetition=repetition,
                            )
                            target = f"branch_{branch_index}"
                        outcome = _call_worker(
                            client=worker_client,
                            recorder=recorder,
                            request=request,
                            seed=seed,
                            contract=contract,
                        )
                        _write_record(run_handle, _call_record(
                            contract=contract,
                            problem=problem,
                            problem_index=problem_index,
                            phase="retry",
                            target=target,
                            repetition=repetition,
                            request=request,
                            outcome=outcome,
                        ))
                        logical_call_count += 1
                        generation_errors += int(outcome.generation_error)
                        if action is TargetAction.RETRY_ROOT:
                            resulting_root = _root_answer(outcome.candidate)
                            root_retries.append(outcome.candidate)
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
                        rewards[action].append(reward)
                        transition = {
                            "schema_version": TARGET_TRANSITION_RECORD_VERSION,
                            "record_type": "retry_transition",
                            "preregistration_id": contract.preregistration_id,
                            "problem_index": problem_index,
                            "problem_id": problem.problem_id,
                            "initial_state_sha256": canonical_json_sha256(
                                initial_state
                            ),
                            "action": action.value,
                            "target": target,
                            "repetition": repetition,
                            "worker_call_id": outcome.call_id,
                            "resulting_root_candidate": resulting_root,
                            "reward": reward,
                        }
                        _write_record(run_handle, transition)
                        transition_count += 1

                direct_majority = _majority_answer((
                    direct.candidate,
                    root_retries[0],
                    root_retries[1],
                ))
                branch_correct = tuple(
                    candidate_semantically_correct(candidate, branch)
                    for candidate, branch in zip(
                        initial_candidates, problem.branches, strict=True
                    )
                )
                if len(branch_correct) != 2:
                    raise AssertionError("branch diagnostics changed arity")
                case = CaseEvaluation(
                    problem_id=problem.problem_id,
                    direct_initial_reward=terminal_reward(
                        _root_answer(direct.candidate), problem
                    ),
                    direct_majority3_reward=terminal_reward(
                        direct_majority, problem
                    ),
                    action_rewards={
                        action: tuple(rewards[action]) for action in ACTION_ORDER
                    },
                    initial_branch_correct=(
                        branch_correct[0], branch_correct[1]
                    ),
                )
                cases.append(case)
                case_summary = {
                    "schema_version": TARGET_CASE_SUMMARY_VERSION,
                    "record_type": "case_summary",
                    "preregistration_id": contract.preregistration_id,
                    "problem_index": problem_index,
                    "problem_id": problem.problem_id,
                    "initial_state": initial_state,
                    "direct_initial_reward": case.direct_initial_reward,
                    "direct_majority3_reward": (
                        case.direct_majority3_reward
                    ),
                    "structured_submit_reward": submit_reward,
                    "action_rewards": {
                        action.value: list(case.action_rewards[action])
                        for action in ACTION_ORDER
                    },
                    "initial_branch_correct": list(
                        case.initial_branch_correct
                    ),
                    "hidden_first_error_action": _first_error_action(
                        case
                    ).value,
                }
                _write_record(run_handle, case_summary, flush=True)

            analysis = analyze_target_value_cases(tuple(cases), contract)
            run_summary = {
                "schema_version": TARGET_RUN_SUMMARY_VERSION,
                "record_type": "run_summary",
                "preregistration_id": contract.preregistration_id,
                "run_contract_sha256": contract.sha256,
                "problem_count": len(problems),
                "logical_call_count": logical_call_count,
                "transition_count": transition_count,
                "generation_error_count": generation_errors,
                "usage": _counter_delta(worker_before, worker_client),
                "analysis": analysis,
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
    return TargetCollectionResult(
        run_path=final_run,
        diagnostic_path=final_diagnostic,
        problem_count=len(problems),
        logical_call_count=logical_call_count,
        transition_count=transition_count,
        run_contract_sha256=contract.sha256,
        analysis=analysis,
    )


__all__ = [
    "TARGET_ANALYSIS_VERSION",
    "TARGET_RUN_CONTRACT_VERSION",
    "CaseEvaluation",
    "TargetCollectionResult",
    "TargetRunContract",
    "analyze_target_value_cases",
    "build_target_run_contract",
    "collect_target_value_run",
    "file_sha256",
]
