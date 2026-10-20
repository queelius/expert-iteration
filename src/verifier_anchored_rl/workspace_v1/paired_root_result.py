"""Exact shared-state experiment for the root result prompt contract."""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..llm_client import OpenAICompatibleLLMClient
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
    CHILD_CONTEXT_CONTRACT_VERSION,
    CHILD_CONTEXT_PRECEDING_SIBLINGS,
    DIRECT_PROMPT_VERSION,
    HYBRID_LENS_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    ROOT_RESULT_CONTRACT_VERSION,
    ROOT_RESULT_FREE,
    ROOT_RESULT_VERIFIER_READY,
    ROOT_RESULT_VERIFIER_READY_INSTRUCTION,
    WORKER_PROMPT_ROOT_RESULT_VERSION,
    WORKER_PROMPT_VERSION,
    WorkerRequest,
    build_observation,
    build_worker_request,
)
from .model import (
    ACTION_AVAILABILITY_ALL,
    ACTION_SCHEMA_VERSION,
    REWARD_CONTRACT_VERSION,
    STATE_SCHEMA_VERSION,
    TRANSITION_SCHEMA_VERSION,
    Action,
    CandidateNode,
    State,
    Verb,
    available_actions,
    child_goals,
    goal_candidates,
)
from .paired_lens import (
    _read_canonical_jsonl,
    file_sha256,
    state_from_observation,
)
from .records import (
    DIAGNOSTIC_ATTEMPT_VERSION,
    TRANSITION_RECORD_VERSION,
    canonical_json_sha256,
    transition_record,
)
from .runtime import (
    SUBMIT_RECEIPT_VERSION,
    WORKER_EXECUTION_REDESIGN_CONTRACT_VERSION,
    WorkspaceEnvironment,
)


PAIRED_ROOT_RESULT_CONTRACT_VERSION = "workspace-paired-root-result-contract-v1"
PAIRED_ROOT_RESULT_HEADER_VERSION = "workspace-paired-root-result-header-v1"
PAIRED_ROOT_RESULT_DIAGNOSTIC_VERSION = (
    "workspace-paired-root-result-diagnostic-header-v1"
)
PAIRED_ROOT_RESULT_PREFIX_VERSION = "workspace-root-finalization-source-v1"
PAIRED_ROOT_RESULT_CASE_VERSION = "workspace-paired-root-result-case-v1"
PAIRED_ROOT_RESULT_CASE_SUMMARY_VERSION = (
    "workspace-paired-root-result-case-summary-v1"
)
PAIRED_ROOT_RESULT_SUMMARY_VERSION = "workspace-paired-root-result-summary-v1"
PAIRED_ROOT_RESULT_BRANCH_ORDER_VERSION = "checkerboard-case-order-v1"
PAIRED_ROOT_RESULT_SEED_POLICY_VERSION = "source-episode-turn-offset-v1"
PAIRED_ROOT_RESULT_SOURCE_ACTION_VERSION = "root-synthesize-only-v1"
PAIRED_ROOT_RESULT_MODES = (
    ROOT_RESULT_FREE,
    ROOT_RESULT_VERIFIER_READY,
)
_ROLE_BY_MODE = {
    ROOT_RESULT_FREE: "free_root_result_worker",
    ROOT_RESULT_VERIFIER_READY: "verifier_ready_root_result_worker",
}
_SYNTHESIZE_ROOT = Action(Verb.SYNTHESIZE, 0)


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


def _assert_treatment_isolated(
    free: WorkerRequest,
    verifier_ready: WorkerRequest,
) -> None:
    if (
        free.system != verifier_ready.system
        or free.schema != verifier_ready.schema
        or free.schema_validation_mode != verifier_ready.schema_validation_mode
    ):
        raise RuntimeError("root result treatment changed the worker boundary")
    if verifier_ready.user.count(ROOT_RESULT_VERIFIER_READY_INSTRUCTION) != 1:
        raise RuntimeError("root result treatment suffix is missing or repeated")
    if (
        verifier_ready.user.replace(
            ROOT_RESULT_VERIFIER_READY_INSTRUCTION,
            "",
            1,
        )
        != free.user
    ):
        raise RuntimeError("root result branches differ outside the treatment")


@dataclass(frozen=True, slots=True)
class RootFinalizationCase:
    problem_id: str
    source_episode_index: int
    prefix_state: State
    source_synthesis_call_id: str
    source_synthesis_request_sha256: str

    def __post_init__(self) -> None:
        _nonempty(self.problem_id, "root finalization problem id")
        if (
            type(self.source_episode_index) is not int
            or self.source_episode_index < 0
        ):
            raise ValueError("source episode index must be nonnegative")
        if not isinstance(self.prefix_state, State) or self.prefix_state.terminal:
            raise ValueError("root finalization prefix must be nonterminal")
        _nonempty(self.source_synthesis_call_id, "source synthesis call id")
        _sha256(
            self.source_synthesis_request_sha256,
            "source synthesis request sha256",
        )

    def source_record(self, *, max_depth: int) -> dict[str, object]:
        return {
            "problem_id": self.problem_id,
            "source_episode_index": self.source_episode_index,
            "source_synthesis_call_id": self.source_synthesis_call_id,
            "source_synthesis_request_sha256": (
                self.source_synthesis_request_sha256
            ),
            "prefix_observation": build_observation(
                self.prefix_state,
                max_depth=max_depth,
            ),
        }


@dataclass(frozen=True, slots=True)
class PreparedRootFinalizations:
    source_run_path: Path
    source_run_sha256: str
    source_run_contract_sha256: str
    prefix_sha256: str
    cases: tuple[RootFinalizationCase, ...]
    examples: tuple[ProblemExample, ...]
    excluded_problem_ids: tuple[str, ...]
    max_depth: int

    def __post_init__(self) -> None:
        _sha256(self.source_run_sha256, "source run sha256")
        _sha256(self.source_run_contract_sha256, "source contract sha256")
        _sha256(self.prefix_sha256, "prefix sha256")
        if not self.cases or len(self.cases) != len(self.examples):
            raise ValueError("prepared root cases and examples differ")
        if tuple(case.problem_id for case in self.cases) != tuple(
            example.problem_id for example in self.examples
        ):
            raise ValueError("prepared root problem order differs")
        if type(self.max_depth) is not int or self.max_depth <= 0:
            raise ValueError("prepared root max depth must be positive")


def prepare_root_finalizations(
    *,
    source_run_path: str | Path,
    examples: tuple[ProblemExample, ...],
    expected_source_run_contract_sha256: str,
    max_depth: int,
) -> PreparedRootFinalizations:
    """Strict-read a fixed run and select pre-SYNTHESIZE root states."""
    _sha256(
        expected_source_run_contract_sha256,
        "expected source run contract sha256",
    )
    if not examples:
        raise ValueError("root finalization preparation requires examples")
    path = Path(source_run_path)
    records = _read_canonical_jsonl(path)
    header = records[0]
    if header.get("record_type") != "run_header":
        raise ValueError("source run lacks its header")
    if header.get("run_contract_sha256") != expected_source_run_contract_sha256:
        raise ValueError("source run contract differs from the expected digest")
    if header.get("policy_name") != "fixed_decomposition":
        raise ValueError("source run is not fixed decomposition")
    settings = header.get("settings")
    if (
        type(settings) is not dict
        or settings.get("child_context_mode")
        != CHILD_CONTEXT_PRECEDING_SIBLINGS
    ):
        raise ValueError("source run child context is not preceding siblings")
    source = header.get("problem_source")
    if type(source) is not dict:
        raise ValueError("source run problem binding is invalid")
    if source.get("sha256") != problem_source_sha256(examples):
        raise ValueError("source run problem projection differs")
    if source.get("evaluation_binding_sha256") != evaluation_binding_sha256(
        examples
    ):
        raise ValueError("source run hidden-label binding differs")
    expected_ids = tuple(example.problem_id for example in examples)
    if tuple(source.get("problem_ids", ())) != expected_ids:
        raise ValueError("source run problem order differs")

    transitions: dict[str, list[dict[str, object]]] = defaultdict(list)
    summaries: list[dict[str, object]] = []
    run_summaries: list[dict[str, object]] = []
    for record in records[1:]:
        if record.get("schema_version") == TRANSITION_RECORD_VERSION:
            transitions[str(record.get("problem_id"))].append(record)
        elif record.get("record_type") == "episode_summary":
            summaries.append(record)
        elif record.get("record_type") == "run_summary":
            run_summaries.append(record)
        else:
            raise ValueError("source run contains an unknown record")
    if len(run_summaries) != 1 or records[-1] is not run_summaries[0]:
        raise ValueError("source run summary is missing or misplaced")
    if tuple(summary.get("problem_id") for summary in summaries) != expected_ids:
        raise ValueError("source episode summaries differ from problem order")

    examples_by_id = {example.problem_id: example for example in examples}
    cases: list[RootFinalizationCase] = []
    selected_examples: list[ProblemExample] = []
    excluded: list[str] = []
    for episode_index, problem_id in enumerate(expected_ids):
        rows = transitions.get(problem_id, [])
        if not rows:
            raise ValueError("source episode has no transitions")
        previous_next: dict[str, object] | None = None
        for turn, row in enumerate(rows):
            if (
                row.get("episode_index") != episode_index
                or row.get("turn") != turn
            ):
                raise ValueError("source transition order differs")
            if previous_next is not None and row.get("observation") != previous_next:
                raise ValueError("source transition chain is broken")
            state_from_observation(row["observation"], max_depth=max_depth)
            state_from_observation(row["next_observation"], max_depth=max_depth)
            previous_next = row["next_observation"]
        summary = summaries[episode_index]
        if (
            previous_next is None
            or previous_next.get("reward") != summary.get("reward")
        ):
            raise ValueError("source episode summary differs from terminal state")

        synthesis_rows = [
            row
            for row in rows
            if row.get("action") == {"verb": "synthesize", "target": 0}
        ]
        if not synthesis_rows:
            excluded.append(problem_id)
            continue
        if len(synthesis_rows) != 1:
            raise ValueError("source episode has repeated root synthesis")
        synthesis = synthesis_rows[0]
        prefix = state_from_observation(
            synthesis["observation"],
            max_depth=max_depth,
        )
        if prefix.turn + 2 > prefix.turn_limit:
            raise ValueError("source prefix lacks synthesis and submit budget")
        root_children = child_goals(prefix, 0)
        if (
            not root_children
            or any(not goal_candidates(prefix, child.id) for child in root_children)
            or goal_candidates(prefix, 0)
        ):
            raise ValueError("source prefix is not ready for root synthesis")
        if _SYNTHESIZE_ROOT not in available_actions(prefix, max_depth=max_depth):
            raise ValueError("root synthesis is not legal in the source prefix")
        expected_request = build_worker_request(
            prefix,
            _SYNTHESIZE_ROOT,
            child_context_mode=CHILD_CONTEXT_PRECEDING_SIBLINGS,
            root_result_mode=ROOT_RESULT_FREE,
        )
        request_record = _request_record(expected_request)
        if synthesis.get("worker_request") != request_record:
            raise ValueError("source synthesis request does not replay exactly")
        call_id = _nonempty(
            synthesis.get("worker_call_id"),
            "source synthesis call id",
        )
        case = RootFinalizationCase(
            problem_id=problem_id,
            source_episode_index=episode_index,
            prefix_state=prefix,
            source_synthesis_call_id=call_id,
            source_synthesis_request_sha256=canonical_json_sha256(
                request_record
            ),
        )
        cases.append(case)
        selected_examples.append(examples_by_id[problem_id])

    source_digest = file_sha256(path)
    prefix_record = {
        "schema_version": PAIRED_ROOT_RESULT_PREFIX_VERSION,
        "source_run_sha256": source_digest,
        "source_run_contract_sha256": expected_source_run_contract_sha256,
        "cases": [case.source_record(max_depth=max_depth) for case in cases],
        "excluded_problem_ids": excluded,
    }
    return PreparedRootFinalizations(
        source_run_path=path,
        source_run_sha256=source_digest,
        source_run_contract_sha256=expected_source_run_contract_sha256,
        prefix_sha256=canonical_json_sha256(prefix_record),
        cases=tuple(cases),
        examples=tuple(selected_examples),
        excluded_problem_ids=tuple(excluded),
        max_depth=max_depth,
    )


@dataclass(frozen=True, slots=True)
class PairedRootResultContract:
    preregistration_id: str
    source_run_sha256: str
    source_run_contract_sha256: str
    prefix_sha256: str
    problem_source_sha256: str
    evaluation_binding_sha256: str
    problem_ids: tuple[str, ...]
    source_episode_indexes: tuple[int, ...]
    excluded_problem_ids: tuple[str, ...]
    turn_limit: int
    max_depth: int
    worker: ModelIdentity
    worker_temperature: float
    worker_max_tokens: int
    worker_seed: int
    branch_modes: tuple[str, str] = PAIRED_ROOT_RESULT_MODES
    branch_order_version: str = PAIRED_ROOT_RESULT_BRANCH_ORDER_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.preregistration_id, "preregistration id")
        for name in (
            "source_run_sha256",
            "source_run_contract_sha256",
            "prefix_sha256",
            "problem_source_sha256",
            "evaluation_binding_sha256",
        ):
            _sha256(getattr(self, name), name.replace("_", " "))
        if not self.problem_ids or len(self.problem_ids) != len(set(self.problem_ids)):
            raise ValueError("paired root problem ids must be nonempty and unique")
        if len(self.source_episode_indexes) != len(self.problem_ids) or any(
            type(index) is not int or index < 0
            for index in self.source_episode_indexes
        ):
            raise ValueError("source episode indexes differ from root cases")
        if self.branch_modes != PAIRED_ROOT_RESULT_MODES:
            raise ValueError("root result branch modes differ from the contract")
        if self.branch_order_version != PAIRED_ROOT_RESULT_BRANCH_ORDER_VERSION:
            raise ValueError("root result branch order version differs")
        if type(self.turn_limit) is not int or self.turn_limit <= 0:
            raise ValueError("paired root turn limit must be positive")
        if type(self.max_depth) is not int or self.max_depth <= 0:
            raise ValueError("paired root max depth must be positive")
        if not isinstance(self.worker, ModelIdentity):
            raise ValueError("paired root worker identity is invalid")
        if (
            type(self.worker_temperature) is not float
            or self.worker_temperature < 0.0
        ):
            raise ValueError("paired root temperature must be nonnegative")
        if type(self.worker_max_tokens) is not int or self.worker_max_tokens <= 0:
            raise ValueError("paired root max tokens must be positive")
        if type(self.worker_seed) is not int or self.worker_seed < 0:
            raise ValueError("paired root seed must be nonnegative")

    def header(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema_version": PAIRED_ROOT_RESULT_HEADER_VERSION,
            "record_type": "run_header",
            "run_contract_version": PAIRED_ROOT_RESULT_CONTRACT_VERSION,
            "preregistration_id": self.preregistration_id,
            "source": {
                "schema_version": PAIRED_ROOT_RESULT_PREFIX_VERSION,
                "source_run_sha256": self.source_run_sha256,
                "source_run_contract_sha256": self.source_run_contract_sha256,
                "prefix_sha256": self.prefix_sha256,
                "problem_source_schema": PROBLEM_SOURCE_VERSION,
                "problem_source_sha256": self.problem_source_sha256,
                "evaluation_binding_version": EVALUATION_BINDING_VERSION,
                "evaluation_binding_sha256": self.evaluation_binding_sha256,
                "problem_ids": list(self.problem_ids),
                "source_episode_indexes": list(self.source_episode_indexes),
                "excluded_problem_ids": list(self.excluded_problem_ids),
            },
            "contracts": {
                "state": STATE_SCHEMA_VERSION,
                "action": ACTION_SCHEMA_VERSION,
                "transition": TRANSITION_SCHEMA_VERSION,
                "transition_record": TRANSITION_RECORD_VERSION,
                "reward": REWARD_CONTRACT_VERSION,
                "worker_execution": WORKER_EXECUTION_REDESIGN_CONTRACT_VERSION,
                "worker_prompts": {
                    ROOT_RESULT_FREE: WORKER_PROMPT_VERSION,
                    ROOT_RESULT_VERIFIER_READY: (
                        WORKER_PROMPT_ROOT_RESULT_VERSION
                    ),
                },
                "direct_prompt": DIRECT_PROMPT_VERSION,
                "hybrid_lens": HYBRID_LENS_VERSION,
                "child_answer_context": CHILD_CONTEXT_CONTRACT_VERSION,
                "root_result": ROOT_RESULT_CONTRACT_VERSION,
                "observation": OBSERVATION_SCHEMA_VERSION,
                "action_availability": ACTION_AVAILABILITY_ALL,
                "submit_receipt": SUBMIT_RECEIPT_VERSION,
                "source_action": PAIRED_ROOT_RESULT_SOURCE_ACTION_VERSION,
                "branch_order": self.branch_order_version,
                "seed_policy": PAIRED_ROOT_RESULT_SEED_POLICY_VERSION,
            },
            "settings": {
                "turn_limit": self.turn_limit,
                "max_depth": self.max_depth,
                "child_context_mode": CHILD_CONTEXT_PRECEDING_SIBLINGS,
                "branch_modes": list(self.branch_modes),
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


def build_paired_root_result_contract(
    *,
    preregistration_id: str,
    prepared: PreparedRootFinalizations,
    worker: ModelIdentity,
    worker_temperature: float,
    worker_max_tokens: int,
    worker_seed: int,
) -> PairedRootResultContract:
    return PairedRootResultContract(
        preregistration_id=preregistration_id,
        source_run_sha256=prepared.source_run_sha256,
        source_run_contract_sha256=prepared.source_run_contract_sha256,
        prefix_sha256=prepared.prefix_sha256,
        problem_source_sha256=problem_source_sha256(prepared.examples),
        evaluation_binding_sha256=evaluation_binding_sha256(prepared.examples),
        problem_ids=tuple(case.problem_id for case in prepared.cases),
        source_episode_indexes=tuple(
            case.source_episode_index for case in prepared.cases
        ),
        excluded_problem_ids=prepared.excluded_problem_ids,
        turn_limit=prepared.cases[0].prefix_state.turn_limit,
        max_depth=prepared.max_depth,
        worker=worker,
        worker_temperature=worker_temperature,
        worker_max_tokens=worker_max_tokens,
        worker_seed=worker_seed,
    )


@dataclass(frozen=True, slots=True)
class PairedRootResultCollection:
    run_path: Path
    diagnostic_path: Path
    case_count: int
    transition_count: int
    solved_by_mode: Mapping[str, int]
    run_contract_sha256: str


def _branch_order(case_index: int) -> tuple[str, str]:
    if case_index % 2 == 0:
        return PAIRED_ROOT_RESULT_MODES
    return tuple(reversed(PAIRED_ROOT_RESULT_MODES))  # type: ignore[return-value]


def _paired_outcome(rewards: Mapping[str, int]) -> str:
    free = rewards[ROOT_RESULT_FREE]
    ready = rewards[ROOT_RESULT_VERIFIER_READY]
    if free and ready:
        return "both"
    if free:
        return "free_only"
    if ready:
        return "verifier_ready_only"
    return "neither"


def collect_paired_root_result_run(
    *,
    run_path: str | Path,
    diagnostic_path: str | Path,
    contract: PairedRootResultContract,
    prepared: PreparedRootFinalizations,
    worker_client: OpenAICompatibleLLMClient,
) -> PairedRootResultCollection:
    """Generate and submit both root prompt branches from each exact state."""
    if not isinstance(contract, PairedRootResultContract):
        raise TypeError("paired root collection requires its run contract")
    if not isinstance(prepared, PreparedRootFinalizations):
        raise TypeError("paired root collection requires prepared states")
    if type(worker_client) is not OpenAICompatibleLLMClient:
        raise TypeError("paired root worker must be OpenAI-compatible")
    _validate_client_binding(contract.worker, worker_client, role="worker")
    expected = build_paired_root_result_contract(
        preregistration_id=contract.preregistration_id,
        prepared=prepared,
        worker=contract.worker,
        worker_temperature=contract.worker_temperature,
        worker_max_tokens=contract.worker_max_tokens,
        worker_seed=contract.worker_seed,
    )
    if expected != contract:
        raise ValueError("paired root contract differs from prepared states")

    final_run = Path(run_path)
    final_diagnostic = Path(diagnostic_path)
    partial_run, partial_diagnostic = _validate_output_paths(
        final_run,
        final_diagnostic,
    )
    diagnostic_header = {
        "schema_version": PAIRED_ROOT_RESULT_DIAGNOSTIC_VERSION,
        "record_type": "diagnostic_header",
        "preregistration_id": contract.preregistration_id,
        "run_contract_sha256": contract.sha256,
        "attempt_schema_version": DIAGNOSTIC_ATTEMPT_VERSION,
        "roles": list(_ROLE_BY_MODE.values()),
    }
    worker_before = _counter_snapshot(worker_client)
    old_observer = worker_client.attempt_observer
    transition_count = 0
    solved = Counter({mode: 0 for mode in PAIRED_ROOT_RESULT_MODES})
    generation_errors = Counter({mode: 0 for mode in PAIRED_ROOT_RESULT_MODES})
    action_counts = {mode: Counter() for mode in PAIRED_ROOT_RESULT_MODES}
    paired_outcomes: Counter[str] = Counter()

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
            examples = {
                example.problem_id: example for example in prepared.examples
            }
            for case_index, case in enumerate(prepared.cases):
                example = examples[case.problem_id]
                _write_record(run_handle, {
                    "schema_version": PAIRED_ROOT_RESULT_CASE_VERSION,
                    "record_type": "paired_case_prefix",
                    "preregistration_id": contract.preregistration_id,
                    "paired_case_index": case_index,
                    **case.source_record(max_depth=contract.max_depth),
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
                        root_result_mode=mode,
                        action_availability_mode=ACTION_AVAILABILITY_ALL,
                    )
                    for mode in PAIRED_ROOT_RESULT_MODES
                }
                requests = {
                    mode: build_worker_request(
                        case.prefix_state,
                        _SYNTHESIZE_ROOT,
                        child_context_mode=CHILD_CONTEXT_PRECEDING_SIBLINGS,
                        root_result_mode=mode,
                    )
                    for mode in PAIRED_ROOT_RESULT_MODES
                }
                _assert_treatment_isolated(
                    requests[ROOT_RESULT_FREE],
                    requests[ROOT_RESULT_VERIFIER_READY],
                )
                states = {
                    mode: case.prefix_state for mode in PAIRED_ROOT_RESULT_MODES
                }
                call_ids = {mode: [] for mode in PAIRED_ROOT_RESULT_MODES}
                branch_transition_counts = {
                    mode: 0 for mode in PAIRED_ROOT_RESULT_MODES
                }
                case_generation_complete = {
                    mode: False for mode in PAIRED_ROOT_RESULT_MODES
                }
                order = _branch_order(case_index)
                branch_positions = {
                    mode: index for index, mode in enumerate(order)
                }

                for mode in order:
                    environment = environments[mode]
                    if _SYNTHESIZE_ROOT not in environment.legal_actions(
                        states[mode]
                    ):
                        raise RuntimeError("paired root synthesis became illegal")
                    role = _ROLE_BY_MODE[mode]
                    offset = recorder.count(role)
                    worker_client.attempt_observer = recorder.callback(role)
                    transition = environment.step(
                        states[mode],
                        _SYNTHESIZE_ROOT,
                    )
                    if transition.worker_request != requests[mode]:
                        raise RuntimeError("paired root request changed at execution")
                    new_ids = recorder.since(role, offset)
                    if len(new_ids) != 1:
                        raise RuntimeError("paired root generation lacks one call id")
                    call_ids[mode].append(new_ids[0])
                    record = transition_record(
                        transition,
                        experiment_id=contract.preregistration_id,
                        arm=mode,
                        episode_index=case_index,
                        problem_id=case.problem_id,
                        max_depth=contract.max_depth,
                        worker_call_id=new_ids[0],
                        teacher_call_id=None,
                        teacher_decision=None,
                    )
                    record.update({
                        "paired_case_index": case_index,
                        "paired_round_index": 0,
                        "paired_phase": "generation",
                        "branch_position": branch_positions[mode],
                    })
                    _write_record(run_handle, record)
                    states[mode] = transition.next_state
                    branch_transition_counts[mode] += 1
                    transition_count += 1
                    action_counts[mode][Verb.SYNTHESIZE.value] += 1
                    generation_errors[mode] += int(transition.generation_error)
                    case_generation_complete[mode] = (
                        not transition.generation_error
                    )

                for mode in PAIRED_ROOT_RESULT_MODES:
                    state = states[mode]
                    if state.history[-1].outcome.value != "applied":
                        continue
                    created_ids = state.history[-1].created_ids
                    if len(created_ids) != 1:
                        raise RuntimeError("root synthesis created an invalid node set")
                    candidate = state.nodes[created_ids[0]]
                    if not isinstance(candidate, CandidateNode) or candidate.goal_id != 0:
                        raise RuntimeError("root synthesis did not create a root Candidate")
                    submit = Action(Verb.SUBMIT, candidate.id)
                    role = _ROLE_BY_MODE[mode]
                    offset = recorder.count(role)
                    worker_client.attempt_observer = recorder.callback(role)
                    transition = environments[mode].step(state, submit)
                    if recorder.count(role) != offset:
                        raise RuntimeError("paired root SUBMIT called the worker")
                    record = transition_record(
                        transition,
                        experiment_id=contract.preregistration_id,
                        arm=mode,
                        episode_index=case_index,
                        problem_id=case.problem_id,
                        max_depth=contract.max_depth,
                        worker_call_id=None,
                        teacher_call_id=None,
                        teacher_decision=None,
                    )
                    record.update({
                        "paired_case_index": case_index,
                        "paired_round_index": 1,
                        "paired_phase": "submission",
                        "branch_position": branch_positions[mode],
                    })
                    _write_record(run_handle, record)
                    states[mode] = transition.next_state
                    branch_transition_counts[mode] += 1
                    transition_count += 1
                    action_counts[mode][Verb.SUBMIT.value] += 1

                rewards = {
                    mode: states[mode].reward for mode in PAIRED_ROOT_RESULT_MODES
                }
                for mode, reward in rewards.items():
                    solved[mode] += reward
                outcome = _paired_outcome(rewards)
                paired_outcomes[outcome] += 1
                _write_record(run_handle, {
                    "schema_version": (
                        PAIRED_ROOT_RESULT_CASE_SUMMARY_VERSION
                    ),
                    "record_type": "paired_case_summary",
                    "preregistration_id": contract.preregistration_id,
                    "paired_case_index": case_index,
                    "problem_id": case.problem_id,
                    "source_episode_index": case.source_episode_index,
                    "rewards": rewards,
                    "paired_outcome": outcome,
                    "generation_complete": {
                        mode: case_generation_complete[mode]
                        for mode in PAIRED_ROOT_RESULT_MODES
                    },
                    "transition_counts": branch_transition_counts,
                    "worker_call_ids": call_ids,
                }, flush=True)

            run_summary = {
                "schema_version": PAIRED_ROOT_RESULT_SUMMARY_VERSION,
                "record_type": "run_summary",
                "preregistration_id": contract.preregistration_id,
                "run_contract_sha256": contract.sha256,
                "case_count": len(prepared.cases),
                "transition_count": transition_count,
                "solved_by_mode": dict(solved),
                "paired_outcomes": dict(sorted(paired_outcomes.items())),
                "action_counts": {
                    mode: dict(sorted(counts.items()))
                    for mode, counts in action_counts.items()
                },
                "generation_error_counts": dict(generation_errors),
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
    return PairedRootResultCollection(
        run_path=final_run,
        diagnostic_path=final_diagnostic,
        case_count=len(prepared.cases),
        transition_count=transition_count,
        solved_by_mode=dict(solved),
        run_contract_sha256=contract.sha256,
    )


__all__ = [
    "PAIRED_ROOT_RESULT_BRANCH_ORDER_VERSION",
    "PAIRED_ROOT_RESULT_CASE_SUMMARY_VERSION",
    "PAIRED_ROOT_RESULT_CASE_VERSION",
    "PAIRED_ROOT_RESULT_CONTRACT_VERSION",
    "PAIRED_ROOT_RESULT_DIAGNOSTIC_VERSION",
    "PAIRED_ROOT_RESULT_HEADER_VERSION",
    "PAIRED_ROOT_RESULT_MODES",
    "PAIRED_ROOT_RESULT_PREFIX_VERSION",
    "PAIRED_ROOT_RESULT_SEED_POLICY_VERSION",
    "PAIRED_ROOT_RESULT_SOURCE_ACTION_VERSION",
    "PAIRED_ROOT_RESULT_SUMMARY_VERSION",
    "PairedRootResultCollection",
    "PairedRootResultContract",
    "PreparedRootFinalizations",
    "RootFinalizationCase",
    "build_paired_root_result_contract",
    "collect_paired_root_result_run",
    "prepare_root_finalizations",
]
