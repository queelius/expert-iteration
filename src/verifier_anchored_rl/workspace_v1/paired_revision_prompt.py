"""Paired REVISE prompt comparison on sealed shared-prefix states."""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..llm_client import OpenAICompatibleLLMClient
from ..verifier_adapter import check_answer
from .experiment import (
    EVALUATION_BINDING_VERSION,
    PROBLEM_SOURCE_VERSION,
    ModelIdentity,
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
from .fixed_revision import (
    FIXED_REVISION_ADEQUACY_VERSION,
    FIXED_REVISION_COLLECTION_PURPOSE,
    FIXED_REVISION_SEED_POLICY_VERSION,
    FIXED_REVISION_SOURCE_ACTION_VERSION,
    FIXED_REVISION_SOURCE_VERSION,
    PreparedFixedRevisions,
)
from .lenses import (
    ACTIVE_FRONTIER_OBSERVATION_SCHEMA_VERSION,
    CHILD_CONTEXT_CONTRACT_VERSION,
    CHILD_CONTEXT_PRECEDING_SIBLINGS,
    HYBRID_LENS_VERSION,
    REVISE_ANSWER_CONTAINER_INSTRUCTION,
    REVISE_ANSWER_CONTAINER_SCHEMA,
    REVISE_INDEPENDENT_ANSWER_CONTAINER_INSTRUCTION,
    REVISE_RESULT_ANSWER_CONTAINER,
    REVISE_RESULT_CONTRACT_VERSION,
    REVISE_RESULT_INDEPENDENT_ANSWER_CONTAINER,
    ROOT_RESULT_CONTRACT_VERSION,
    ROOT_RESULT_VERIFIER_READY,
    WORKER_PROMPT_REVISE_ANSWER_VERSION,
    WORKER_PROMPT_REVISE_INDEPENDENT_VERSION,
    WORKER_PROMPT_ROOT_RESULT_VERSION,
    build_worker_request,
)
from .model import (
    ACTION_AVAILABILITY_ACTIVE_FRONTIER,
    ACTION_SCHEMA_VERSION,
    REWARD_CONTRACT_VERSION,
    STATE_SCHEMA_VERSION,
    TRANSITION_SCHEMA_VERSION,
    CandidateNode,
    node_at,
)
from .paired_revise import (
    _answer_container,
    _new_candidate,
    _nonempty,
    _sha256,
)
from .records import (
    DIAGNOSTIC_ATTEMPT_VERSION,
    canonical_json_sha256,
    transition_record,
)
from .runtime import (
    SUBMIT_RECEIPT_VERSION,
    WORKER_EXECUTION_REDESIGN_CONTRACT_VERSION,
    WorkspaceEnvironment,
)


PAIRED_REVISION_PROMPT_CONTRACT_VERSION = (
    "workspace-paired-revision-prompt-contract-v1"
)
PAIRED_REVISION_PROMPT_HEADER_VERSION = (
    "workspace-paired-revision-prompt-header-v1"
)
PAIRED_REVISION_PROMPT_DIAGNOSTIC_VERSION = (
    "workspace-paired-revision-prompt-diagnostic-header-v1"
)
PAIRED_REVISION_PROMPT_TRANSITION_VERSION = (
    "workspace-paired-revision-prompt-transition-v1"
)
PAIRED_REVISION_PROMPT_CASE_VERSION = (
    "workspace-paired-revision-prompt-case-v1"
)
PAIRED_REVISION_PROMPT_SUMMARY_VERSION = (
    "workspace-paired-revision-prompt-summary-v1"
)
PAIRED_REVISION_PROMPT_BRANCH_ORDER_VERSION = (
    "checkerboard-revision-prompt-order-v1"
)
PAIRED_REVISION_PROMPT_COLLECTION_PURPOSE = (
    "selection-corpus-prompt-factor-evaluation-only-v1"
)
PAIRED_REVISION_PROMPT_MODES = (
    REVISE_RESULT_ANSWER_CONTAINER,
    REVISE_RESULT_INDEPENDENT_ANSWER_CONTAINER,
)
_ROLE_BY_MODE = {
    REVISE_RESULT_ANSWER_CONTAINER: "retained_revision_worker",
    REVISE_RESULT_INDEPENDENT_ANSWER_CONTAINER: (
        "independent_revision_worker"
    ),
}


def _assert_prompt_isolated(retained, independent) -> None:
    if (
        retained.system != independent.system
        or retained.schema != REVISE_ANSWER_CONTAINER_SCHEMA
        or independent.schema != REVISE_ANSWER_CONTAINER_SCHEMA
        or retained.schema_validation_mode != independent.schema_validation_mode
    ):
        raise RuntimeError("paired revision prompt changed another contract")
    if retained.user.count(REVISE_ANSWER_CONTAINER_INSTRUCTION) != 1:
        raise RuntimeError("retained revision instruction is not isolated")
    if (
        independent.user.count(
            REVISE_INDEPENDENT_ANSWER_CONTAINER_INSTRUCTION
        )
        != 1
    ):
        raise RuntimeError("independent revision instruction is not isolated")
    if retained.user.replace(
        REVISE_ANSWER_CONTAINER_INSTRUCTION,
        REVISE_INDEPENDENT_ANSWER_CONTAINER_INSTRUCTION,
        1,
    ) != independent.user:
        raise RuntimeError("paired requests differ outside the instruction")


@dataclass(frozen=True, slots=True)
class PairedRevisionPromptContract:
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
    modes: tuple[str, str] = PAIRED_REVISION_PROMPT_MODES
    branch_order_version: str = PAIRED_REVISION_PROMPT_BRANCH_ORDER_VERSION

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
            raise ValueError("paired prompt case keys must be nonempty and unique")
        if len(self.problem_ids) != len(self.case_keys):
            raise ValueError("paired prompt problem ids differ from case keys")
        if self.modes != PAIRED_REVISION_PROMPT_MODES:
            raise ValueError("paired revision prompt modes differ")
        if self.branch_order_version != PAIRED_REVISION_PROMPT_BRANCH_ORDER_VERSION:
            raise ValueError("paired revision branch order differs")
        if not isinstance(self.worker, ModelIdentity):
            raise ValueError("paired revision worker identity is invalid")
        if type(self.worker_temperature) is not float or self.worker_temperature < 0:
            raise ValueError("paired revision temperature must be nonnegative")
        if type(self.worker_max_tokens) is not int or self.worker_max_tokens <= 0:
            raise ValueError("paired revision max tokens must be positive")
        if type(self.worker_seed) is not int or self.worker_seed < 0:
            raise ValueError("paired revision seed must be nonnegative")
        if type(self.turn_limit) is not int or self.turn_limit <= 0:
            raise ValueError("paired revision turn limit must be positive")
        if type(self.max_depth) is not int or self.max_depth <= 0:
            raise ValueError("paired revision max depth must be positive")

    def header(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema_version": PAIRED_REVISION_PROMPT_HEADER_VERSION,
            "record_type": "run_header",
            "run_contract_version": PAIRED_REVISION_PROMPT_CONTRACT_VERSION,
            "preregistration_id": self.preregistration_id,
            "source": {
                "schema_version": FIXED_REVISION_SOURCE_VERSION,
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
                "transition_record": PAIRED_REVISION_PROMPT_TRANSITION_VERSION,
                "reward": REWARD_CONTRACT_VERSION,
                "worker_execution": WORKER_EXECUTION_REDESIGN_CONTRACT_VERSION,
                "worker_prompt_root_result": WORKER_PROMPT_ROOT_RESULT_VERSION,
                "retained_worker_prompt": WORKER_PROMPT_REVISE_ANSWER_VERSION,
                "independent_worker_prompt": (
                    WORKER_PROMPT_REVISE_INDEPENDENT_VERSION
                ),
                "hybrid_lens": HYBRID_LENS_VERSION,
                "child_answer_context": CHILD_CONTEXT_CONTRACT_VERSION,
                "root_result": ROOT_RESULT_CONTRACT_VERSION,
                "revise_result": REVISE_RESULT_CONTRACT_VERSION,
                "observation": ACTIVE_FRONTIER_OBSERVATION_SCHEMA_VERSION,
                "action_availability": ACTION_AVAILABILITY_ACTIVE_FRONTIER,
                "submit_receipt": SUBMIT_RECEIPT_VERSION,
                "source_action": FIXED_REVISION_SOURCE_ACTION_VERSION,
                "branch_order": self.branch_order_version,
                "seed_policy": FIXED_REVISION_SEED_POLICY_VERSION,
                "adequacy_measure": FIXED_REVISION_ADEQUACY_VERSION,
                "collection_purpose": (
                    PAIRED_REVISION_PROMPT_COLLECTION_PURPOSE
                ),
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


def build_paired_revision_prompt_contract(
    *,
    preregistration_id: str,
    prepared: PreparedFixedRevisions,
) -> PairedRevisionPromptContract:
    return PairedRevisionPromptContract(
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


def _branch_order(case_index: int) -> tuple[str, str]:
    if case_index % 2 == 0:
        return PAIRED_REVISION_PROMPT_MODES
    return tuple(reversed(PAIRED_REVISION_PROMPT_MODES))  # type: ignore[return-value]


def _correctness_cell(baseline: bool, revision: bool) -> str:
    if baseline and revision:
        return "both"
    if baseline:
        return "baseline_only"
    if revision:
        return "revision_only"
    return "neither"


def _paired_outcome(values: Mapping[str, bool | None]) -> str:
    retained = values[REVISE_RESULT_ANSWER_CONTAINER]
    independent = values[REVISE_RESULT_INDEPENDENT_ANSWER_CONTAINER]
    if retained is None or independent is None:
        return "unavailable"
    if retained and independent:
        return "both"
    if retained:
        return "retained_only"
    if independent:
        return "independent_only"
    return "neither"


@dataclass(frozen=True, slots=True)
class PairedRevisionPromptCollection:
    run_path: Path
    diagnostic_path: Path
    case_count: int
    transition_count: int
    answer_containers_by_mode: Mapping[str, int]
    correctness_cells_by_mode: Mapping[str, Mapping[str, int]]
    independent_corpus_adequate: bool
    run_contract_sha256: str


def collect_paired_revision_prompt_run(
    *,
    run_path: str | Path,
    diagnostic_path: str | Path,
    contract: PairedRevisionPromptContract,
    prepared: PreparedFixedRevisions,
    worker_client: OpenAICompatibleLLMClient,
) -> PairedRevisionPromptCollection:
    """Compare retained and independent REVISE instructions exactly."""
    if not isinstance(contract, PairedRevisionPromptContract):
        raise TypeError("paired prompt collection requires its run contract")
    if not isinstance(prepared, PreparedFixedRevisions):
        raise TypeError("paired prompt collection requires prepared states")
    if type(worker_client) is not OpenAICompatibleLLMClient:
        raise TypeError("paired prompt worker must be OpenAI-compatible")
    _validate_client_binding(contract.worker, worker_client, role="worker")
    expected = build_paired_revision_prompt_contract(
        preregistration_id=contract.preregistration_id,
        prepared=prepared,
    )
    if expected != contract:
        raise ValueError("paired prompt contract differs from prepared states")

    final_run = Path(run_path)
    final_diagnostic = Path(diagnostic_path)
    partial_run, partial_diagnostic = _validate_output_paths(
        final_run,
        final_diagnostic,
    )
    diagnostic_header = {
        "schema_version": PAIRED_REVISION_PROMPT_DIAGNOSTIC_VERSION,
        "record_type": "diagnostic_header",
        "preregistration_id": contract.preregistration_id,
        "run_contract_sha256": contract.sha256,
        "attempt_schema_version": DIAGNOSTIC_ATTEMPT_VERSION,
        "roles": list(_ROLE_BY_MODE.values()),
    }
    examples = {example.problem_id: example for example in prepared.examples}
    worker_before = _counter_snapshot(worker_client)
    old_observer = worker_client.attempt_observer
    answer_containers = Counter({mode: 0 for mode in contract.modes})
    generation_errors = Counter({mode: 0 for mode in contract.modes})
    revision_correct = Counter({mode: 0 for mode in contract.modes})
    correctness_cells = {
        mode: Counter() for mode in contract.modes
    }
    paired_outcomes: Counter[str] = Counter()
    baseline_correct = 0
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
                target = node_at(case.state, case.action.target)
                if not isinstance(target, CandidateNode):
                    raise AssertionError("validated target changed type")
                baseline_symbolic = bool(check_answer(
                    target.answer,
                    example.expected_answer,
                ).success)
                baseline_correct += int(baseline_symbolic)
                _write_record(run_handle, {
                    "schema_version": PAIRED_REVISION_PROMPT_CASE_VERSION,
                    "record_type": "paired_revision_prompt_source",
                    "preregistration_id": contract.preregistration_id,
                    "case_index": case_index,
                    **case.source_record(max_depth=contract.max_depth),
                    "baseline_symbolic_success": baseline_symbolic,
                })
                requests = {
                    mode: build_worker_request(
                        case.state,
                        case.action,
                        child_context_mode=CHILD_CONTEXT_PRECEDING_SIBLINGS,
                        root_result_mode=ROOT_RESULT_VERIFIER_READY,
                        revise_result_mode=mode,
                    )
                    for mode in contract.modes
                }
                _assert_prompt_isolated(
                    requests[REVISE_RESULT_ANSWER_CONTAINER],
                    requests[REVISE_RESULT_INDEPENDENT_ANSWER_CONTAINER],
                )
                order = _branch_order(case_index)
                positions = {mode: index for index, mode in enumerate(order)}
                case_symbolic: dict[str, bool | None] = {}
                case_containers: dict[str, bool] = {}
                call_ids: dict[str, str] = {}

                for mode in order:
                    environment = WorkspaceEnvironment(
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
                    if case.action not in environment.legal_actions(case.state):
                        raise RuntimeError("sealed paired REVISE became illegal")
                    role = _ROLE_BY_MODE[mode]
                    offset = recorder.count(role)
                    worker_client.attempt_observer = recorder.callback(role)
                    transition = environment.step(case.state, case.action)
                    if transition.worker_request != requests[mode]:
                        raise RuntimeError("paired prompt request changed")
                    new_ids = recorder.since(role, offset)
                    if len(new_ids) != 1:
                        raise RuntimeError("paired prompt lacks one logical call")
                    call_ids[mode] = new_ids[0]
                    payload = (
                        None
                        if transition.worker_payload_json is None
                        else json.loads(transition.worker_payload_json)
                    )
                    has_container, _ = _answer_container(payload)
                    generated = _new_candidate(transition)
                    if has_container != (generated is not None):
                        raise RuntimeError("paired prompt Candidate differs")
                    symbolic: bool | None = None
                    cell = "unavailable"
                    if generated is not None:
                        symbolic = bool(check_answer(
                            generated.answer,
                            example.expected_answer,
                        ).success)
                        cell = _correctness_cell(
                            baseline_symbolic,
                            symbolic,
                        )
                        revision_correct[mode] += int(symbolic)
                    correctness_cells[mode][cell] += 1
                    answer_containers[mode] += int(has_container)
                    generation_errors[mode] += int(transition.generation_error)
                    case_symbolic[mode] = symbolic
                    case_containers[mode] = has_container
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
                        "schema_version": (
                            PAIRED_REVISION_PROMPT_TRANSITION_VERSION
                        ),
                        "case_index": case_index,
                        "source_episode_index": case.source_episode_index,
                        "source_turn": case.source_turn,
                        "branch_position": positions[mode],
                        "answer_container": has_container,
                        "baseline_symbolic_success": baseline_symbolic,
                        "revision_symbolic_success": symbolic,
                        "correctness_cell": cell,
                        "behavior_cloning_eligible": False,
                    })
                    _write_record(run_handle, record)
                    transition_count += 1

                outcome = _paired_outcome(case_symbolic)
                paired_outcomes[outcome] += 1
                _write_record(run_handle, {
                    "schema_version": PAIRED_REVISION_PROMPT_CASE_VERSION,
                    "record_type": "paired_revision_prompt_case_summary",
                    "preregistration_id": contract.preregistration_id,
                    "case_index": case_index,
                    "problem_id": case.problem_id,
                    "source_episode_index": case.source_episode_index,
                    "source_turn": case.source_turn,
                    "baseline_symbolic_success": baseline_symbolic,
                    "answer_containers": case_containers,
                    "revision_symbolic_success": case_symbolic,
                    "paired_revision_outcome": outcome,
                    "worker_call_ids": call_ids,
                }, flush=True)

            independent_cells = correctness_cells[
                REVISE_RESULT_INDEPENDENT_ANSWER_CONTAINER
            ]
            independent_corpus_adequate = (
                answer_containers[
                    REVISE_RESULT_INDEPENDENT_ANSWER_CONTAINER
                ]
                >= 55
                and independent_cells["baseline_only"] >= 5
                and independent_cells["revision_only"] >= 5
            )
            run_summary = {
                "schema_version": PAIRED_REVISION_PROMPT_SUMMARY_VERSION,
                "record_type": "run_summary",
                "preregistration_id": contract.preregistration_id,
                "run_contract_sha256": contract.sha256,
                "case_count": len(prepared.cases),
                "transition_count": transition_count,
                "answer_containers_by_mode": dict(answer_containers),
                "generation_errors_by_mode": dict(generation_errors),
                "baseline_symbolic_success": baseline_correct,
                "revision_symbolic_success_by_mode": dict(revision_correct),
                "correctness_cells_by_mode": {
                    mode: dict(sorted(cells.items()))
                    for mode, cells in correctness_cells.items()
                },
                "paired_revision_outcomes": dict(
                    sorted(paired_outcomes.items())
                ),
                "independent_corpus_adequate": (
                    independent_corpus_adequate
                ),
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
    return PairedRevisionPromptCollection(
        run_path=final_run,
        diagnostic_path=final_diagnostic,
        case_count=len(prepared.cases),
        transition_count=transition_count,
        answer_containers_by_mode=dict(answer_containers),
        correctness_cells_by_mode={
            mode: dict(cells) for mode, cells in correctness_cells.items()
        },
        independent_corpus_adequate=independent_corpus_adequate,
        run_contract_sha256=contract.sha256,
    )


__all__ = [
    "PAIRED_REVISION_PROMPT_BRANCH_ORDER_VERSION",
    "PAIRED_REVISION_PROMPT_CASE_VERSION",
    "PAIRED_REVISION_PROMPT_COLLECTION_PURPOSE",
    "PAIRED_REVISION_PROMPT_CONTRACT_VERSION",
    "PAIRED_REVISION_PROMPT_DIAGNOSTIC_VERSION",
    "PAIRED_REVISION_PROMPT_HEADER_VERSION",
    "PAIRED_REVISION_PROMPT_MODES",
    "PAIRED_REVISION_PROMPT_SUMMARY_VERSION",
    "PAIRED_REVISION_PROMPT_TRANSITION_VERSION",
    "PairedRevisionPromptCollection",
    "PairedRevisionPromptContract",
    "build_paired_revision_prompt_contract",
    "collect_paired_revision_prompt_run",
]
