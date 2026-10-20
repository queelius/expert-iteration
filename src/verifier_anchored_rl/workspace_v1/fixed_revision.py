"""Fixed REVISE corpus from exact shared-direct-Candidate states."""

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
    REVISE_RESULT_ANSWER_CONTAINER,
    REVISE_RESULT_CONTRACT_VERSION,
    ROOT_RESULT_CONTRACT_VERSION,
    ROOT_RESULT_VERIFIER_READY,
    WORKER_PROMPT_REVISE_ANSWER_VERSION,
    WORKER_PROMPT_ROOT_RESULT_VERSION,
    build_observation,
    build_worker_request,
    direct_request,
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
    protected_direct_candidate_id,
)
from .paired_direct_teacher import (
    PAIRED_DIRECT_TEACHER_REVISE_HEADER_VERSION,
    PAIRED_DIRECT_TEACHER_TRANSITION_VERSION,
    ROLE_SHARED_PREFIX_WORKER,
)
from .paired_lens import (
    _read_canonical_jsonl,
    file_sha256,
    state_from_observation,
)
from .paired_revise import (
    _answer_container,
    _model_identity,
    _new_candidate,
    _nonempty,
    _request_record,
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


FIXED_REVISION_CONTRACT_VERSION = "workspace-fixed-revision-contract-v1"
FIXED_REVISION_HEADER_VERSION = "workspace-fixed-revision-header-v1"
FIXED_REVISION_DIAGNOSTIC_VERSION = (
    "workspace-fixed-revision-diagnostic-header-v1"
)
FIXED_REVISION_SOURCE_VERSION = "e39-shared-prefix-revision-source-v1"
FIXED_REVISION_SOURCE_ACTION_VERSION = "revise-protected-candidate-once-v1"
FIXED_REVISION_TRANSITION_VERSION = "workspace-fixed-revision-transition-v1"
FIXED_REVISION_CASE_VERSION = "workspace-fixed-revision-case-v1"
FIXED_REVISION_SUMMARY_VERSION = "workspace-fixed-revision-summary-v1"
FIXED_REVISION_SEED_POLICY_VERSION = "source-episode-turn-offset-v1"
FIXED_REVISION_ADEQUACY_VERSION = "bidirectional-discordance-5-v1"
FIXED_REVISION_COLLECTION_PURPOSE = "selection-corpus-evaluation-only-v1"
FIXED_REVISION_ROLE = "fixed_revision_worker"


@dataclass(frozen=True, slots=True)
class FixedRevisionCase:
    problem_id: str
    source_episode_index: int
    source_turn: int
    state: State
    action: Action
    source_worker_call_id: str
    source_request_sha256: str
    source_payload_sha256: str

    def __post_init__(self) -> None:
        _nonempty(self.problem_id, "fixed revision problem id")
        if type(self.source_episode_index) is not int or self.source_episode_index < 0:
            raise ValueError("source episode index must be nonnegative")
        if type(self.source_turn) is not int or self.source_turn < 0:
            raise ValueError("source turn must be nonnegative")
        if not isinstance(self.state, State) or self.state.terminal:
            raise ValueError("fixed revision source state must be nonterminal")
        if self.state.turn != self.source_turn:
            raise ValueError("fixed revision source turn differs from state")
        if not isinstance(self.action, Action) or self.action.verb is not Verb.REVISE:
            raise ValueError("fixed revision action must be REVISE")
        target = node_at(self.state, self.action.target)
        if not isinstance(target, CandidateNode) or target.goal_id != 0:
            raise ValueError("fixed revision target must be a root Candidate")
        if protected_direct_candidate_id(self.state) != self.action.target:
            raise ValueError("fixed revision must target the protected Candidate")
        _nonempty(self.source_worker_call_id, "source worker call id")
        _sha256(self.source_request_sha256, "source request sha256")
        _sha256(self.source_payload_sha256, "source payload sha256")

    def source_record(self, *, max_depth: int) -> dict[str, object]:
        return {
            "schema_version": FIXED_REVISION_SOURCE_VERSION,
            "problem_id": self.problem_id,
            "source_episode_index": self.source_episode_index,
            "source_turn": self.source_turn,
            "source_observation": build_observation(
                self.state,
                max_depth=max_depth,
                action_availability_mode=(
                    ACTION_AVAILABILITY_ACTIVE_FRONTIER
                ),
            ),
            "action": {
                "verb": self.action.verb.value,
                "target": self.action.target,
            },
            "source_worker_call_id": self.source_worker_call_id,
            "source_request_sha256": self.source_request_sha256,
            "source_payload_sha256": self.source_payload_sha256,
        }


@dataclass(frozen=True, slots=True)
class PreparedFixedRevisions:
    source_run_path: Path
    source_diagnostic_path: Path
    source_run_sha256: str
    source_diagnostic_sha256: str
    source_run_contract_sha256: str
    selection_sha256: str
    cases: tuple[FixedRevisionCase, ...]
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
        _sha256(self.selection_sha256, "fixed revision selection sha256")
        if not self.cases or len(self.cases) != len(self.examples):
            raise ValueError("fixed revision preparation must cover the source")
        if not isinstance(self.worker, ModelIdentity):
            raise ValueError("fixed revision worker identity is invalid")


def prepare_fixed_revisions(
    *,
    source_run_path: str | Path,
    source_diagnostic_path: str | Path,
    examples: tuple[ProblemExample, ...],
    expected_source_run_contract_sha256: str,
    max_depth: int,
) -> PreparedFixedRevisions:
    """Strict-read every exact E39 shared post-prefix state."""
    _sha256(
        expected_source_run_contract_sha256,
        "expected source run contract sha256",
    )
    if not examples:
        raise ValueError("fixed revision preparation requires examples")
    run_path = Path(source_run_path)
    diagnostic_path = Path(source_diagnostic_path)
    records = _read_canonical_jsonl(run_path)
    diagnostics = _read_canonical_jsonl(diagnostic_path)
    header = records[0]
    if (
        header.get("record_type") != "run_header"
        or header.get("schema_version")
        != PAIRED_DIRECT_TEACHER_REVISE_HEADER_VERSION
        or header.get("run_contract_sha256")
        != expected_source_run_contract_sha256
    ):
        raise ValueError("source run is not the expected E39 contract")
    source = header.get("problem_source")
    settings = header.get("settings")
    contracts = header.get("contracts")
    models = header.get("models")
    if not all(type(value) is dict for value in (source, settings, contracts, models)):
        raise ValueError("source header containers are invalid")
    assert isinstance(source, dict)
    assert isinstance(settings, dict)
    assert isinstance(contracts, dict)
    assert isinstance(models, dict)
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
    if (
        settings.get("child_context_mode")
        != CHILD_CONTEXT_PRECEDING_SIBLINGS
        or settings.get("root_result_mode") != ROOT_RESULT_VERIFIER_READY
        or settings.get("revise_result_mode")
        != REVISE_RESULT_ANSWER_CONTAINER
        or settings.get("action_availability_mode")
        != ACTION_AVAILABILITY_ACTIVE_FRONTIER
        or settings.get("max_depth") != max_depth
        or contracts.get("worker_prompt_revise")
        != WORKER_PROMPT_REVISE_ANSWER_VERSION
    ):
        raise ValueError("source run does not use the sealed E39 interface")
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
    prefix_records: dict[int, dict[str, object]] = {}
    direct_rewards: dict[int, int] = {}
    for record in records[1:]:
        if record.get("schema_version") != PAIRED_DIRECT_TEACHER_TRANSITION_VERSION:
            continue
        episode_index = record.get("episode_index")
        if type(episode_index) is not int:
            raise ValueError("source transition episode index is invalid")
        if record.get("arm") == "shared_prefix":
            if episode_index in prefix_records:
                raise ValueError("source repeats a shared prefix")
            prefix_records[episode_index] = record
        elif record.get("arm") == "direct_submit":
            reward = record.get("reward")
            if reward not in (0, 1) or episode_index in direct_rewards:
                raise ValueError("source direct reward is invalid")
            direct_rewards[episode_index] = reward
    if set(prefix_records) != set(range(len(examples))):
        raise ValueError("source does not contain one prefix per example")
    if set(direct_rewards) != set(range(len(examples))):
        raise ValueError("source does not contain one direct result per example")

    cases: list[FixedRevisionCase] = []
    for episode_index, example in enumerate(examples):
        record = prefix_records[episode_index]
        if (
            record.get("problem_id") != example.problem_id
            or record.get("paired_phase") != "shared_prefix"
            or record.get("decision_source") != "deterministic_prefix"
            or record.get("behavior_cloning_eligible") is not False
            or record.get("generation_error") is not False
            or record.get("action") != {"verb": "answer", "target": 0}
        ):
            raise ValueError("source shared prefix provenance is invalid")
        expected_direct = _request_record(direct_request(example.problem))
        if record.get("worker_request") != expected_direct:
            raise ValueError("source shared prefix request does not replay")
        payload = record.get("worker_payload")
        if type(payload) is not dict:
            raise ValueError("source shared prefix lacks a payload")
        state = state_from_observation(
            record.get("next_observation"),
            max_depth=max_depth,
            action_availability_mode=ACTION_AVAILABILITY_ACTIVE_FRONTIER,
        )
        candidate_id = protected_direct_candidate_id(state)
        if candidate_id is None:
            raise ValueError("source shared state lacks a protected Candidate")
        action = Action(Verb.REVISE, candidate_id)
        if action not in actions_for_mode(
            state,
            max_depth=max_depth,
            mode=ACTION_AVAILABILITY_ACTIVE_FRONTIER,
        ):
            raise ValueError("fixed revision action is not legal")
        target = node_at(state, candidate_id)
        if not isinstance(target, CandidateNode):
            raise AssertionError("protected node changed type")
        verifier_reward = int(check_answer(
            target.answer,
            example.expected_answer,
        ).success)
        if verifier_reward != direct_rewards[episode_index]:
            raise ValueError("source direct reward does not replay symbolically")
        call_id = _nonempty(record.get("worker_call_id"), "source worker call id")
        call_attempts = attempts_by_call.get(call_id, [])
        if not call_attempts or any(
            attempt.get("role") != ROLE_SHARED_PREFIX_WORKER
            for attempt in call_attempts
        ):
            raise ValueError("source prefix call lacks bound diagnostics")
        cases.append(FixedRevisionCase(
            problem_id=example.problem_id,
            source_episode_index=episode_index,
            source_turn=state.turn,
            state=state,
            action=action,
            source_worker_call_id=call_id,
            source_request_sha256=canonical_json_sha256(expected_direct),
            source_payload_sha256=canonical_json_sha256(payload),
        ))

    selection_record = {
        "schema_version": FIXED_REVISION_SOURCE_VERSION,
        "source_run_sha256": file_sha256(run_path),
        "source_diagnostic_sha256": file_sha256(diagnostic_path),
        "source_run_contract_sha256": expected_source_run_contract_sha256,
        "cases": [case.source_record(max_depth=max_depth) for case in cases],
    }
    return PreparedFixedRevisions(
        source_run_path=run_path,
        source_diagnostic_path=diagnostic_path,
        source_run_sha256=selection_record["source_run_sha256"],
        source_diagnostic_sha256=selection_record[
            "source_diagnostic_sha256"
        ],
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
class FixedRevisionContract:
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
            raise ValueError("fixed revision case keys must be nonempty and unique")
        if len(self.problem_ids) != len(self.case_keys):
            raise ValueError("fixed revision problem ids differ from case keys")
        if not isinstance(self.worker, ModelIdentity):
            raise ValueError("fixed revision worker identity is invalid")
        if type(self.worker_temperature) is not float or self.worker_temperature < 0:
            raise ValueError("fixed revision temperature must be nonnegative")
        if type(self.worker_max_tokens) is not int or self.worker_max_tokens <= 0:
            raise ValueError("fixed revision max tokens must be positive")
        if type(self.worker_seed) is not int or self.worker_seed < 0:
            raise ValueError("fixed revision seed must be nonnegative")
        if type(self.turn_limit) is not int or self.turn_limit <= 0:
            raise ValueError("fixed revision turn limit must be positive")
        if type(self.max_depth) is not int or self.max_depth <= 0:
            raise ValueError("fixed revision max depth must be positive")

    def header(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema_version": FIXED_REVISION_HEADER_VERSION,
            "record_type": "run_header",
            "run_contract_version": FIXED_REVISION_CONTRACT_VERSION,
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
                "transition_record": FIXED_REVISION_TRANSITION_VERSION,
                "reward": REWARD_CONTRACT_VERSION,
                "worker_execution": WORKER_EXECUTION_REDESIGN_CONTRACT_VERSION,
                "worker_prompt_root_result": WORKER_PROMPT_ROOT_RESULT_VERSION,
                "worker_prompt_revise": WORKER_PROMPT_REVISE_ANSWER_VERSION,
                "hybrid_lens": HYBRID_LENS_VERSION,
                "child_answer_context": CHILD_CONTEXT_CONTRACT_VERSION,
                "root_result": ROOT_RESULT_CONTRACT_VERSION,
                "revise_result": REVISE_RESULT_CONTRACT_VERSION,
                "observation": ACTIVE_FRONTIER_OBSERVATION_SCHEMA_VERSION,
                "action_availability": ACTION_AVAILABILITY_ACTIVE_FRONTIER,
                "submit_receipt": SUBMIT_RECEIPT_VERSION,
                "source_action": FIXED_REVISION_SOURCE_ACTION_VERSION,
                "seed_policy": FIXED_REVISION_SEED_POLICY_VERSION,
                "adequacy_measure": FIXED_REVISION_ADEQUACY_VERSION,
                "collection_purpose": FIXED_REVISION_COLLECTION_PURPOSE,
            },
            "settings": {
                "case_count": len(self.case_keys),
                "turn_limit": self.turn_limit,
                "max_depth": self.max_depth,
                "child_context_mode": CHILD_CONTEXT_PRECEDING_SIBLINGS,
                "root_result_mode": ROOT_RESULT_VERIFIER_READY,
                "revise_result_mode": REVISE_RESULT_ANSWER_CONTAINER,
                "action_availability_mode": (
                    ACTION_AVAILABILITY_ACTIVE_FRONTIER
                ),
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


def build_fixed_revision_contract(
    *,
    preregistration_id: str,
    prepared: PreparedFixedRevisions,
) -> FixedRevisionContract:
    return FixedRevisionContract(
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


def _correctness_cell(
    baseline: bool,
    revision: bool,
) -> str:
    if baseline and revision:
        return "both"
    if baseline:
        return "baseline_only"
    if revision:
        return "revision_only"
    return "neither"


@dataclass(frozen=True, slots=True)
class FixedRevisionCollection:
    run_path: Path
    diagnostic_path: Path
    case_count: int
    transition_count: int
    answer_container_count: int
    correctness_cells: Mapping[str, int]
    selection_corpus_adequate: bool
    run_contract_sha256: str


def collect_fixed_revision_run(
    *,
    run_path: str | Path,
    diagnostic_path: str | Path,
    contract: FixedRevisionContract,
    prepared: PreparedFixedRevisions,
    worker_client: OpenAICompatibleLLMClient,
) -> FixedRevisionCollection:
    """Apply one fixed protected-Candidate REVISE to every sealed state."""
    if not isinstance(contract, FixedRevisionContract):
        raise TypeError("fixed revision collection requires its run contract")
    if not isinstance(prepared, PreparedFixedRevisions):
        raise TypeError("fixed revision collection requires prepared states")
    if type(worker_client) is not OpenAICompatibleLLMClient:
        raise TypeError("fixed revision worker must be OpenAI-compatible")
    _validate_client_binding(contract.worker, worker_client, role="worker")
    expected = build_fixed_revision_contract(
        preregistration_id=contract.preregistration_id,
        prepared=prepared,
    )
    if expected != contract:
        raise ValueError("fixed revision contract differs from prepared states")

    final_run = Path(run_path)
    final_diagnostic = Path(diagnostic_path)
    partial_run, partial_diagnostic = _validate_output_paths(
        final_run,
        final_diagnostic,
    )
    diagnostic_header = {
        "schema_version": FIXED_REVISION_DIAGNOSTIC_VERSION,
        "record_type": "diagnostic_header",
        "preregistration_id": contract.preregistration_id,
        "run_contract_sha256": contract.sha256,
        "attempt_schema_version": DIAGNOSTIC_ATTEMPT_VERSION,
        "roles": [FIXED_REVISION_ROLE],
    }
    examples = {example.problem_id: example for example in prepared.examples}
    worker_before = _counter_snapshot(worker_client)
    old_observer = worker_client.attempt_observer
    answer_containers = 0
    generation_errors = 0
    correctness_cells: Counter[str] = Counter()
    baseline_correct = 0
    revision_correct = 0
    oracle_correct = 0
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
                roles=(FIXED_REVISION_ROLE,),
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
                _write_record(run_handle, {
                    "schema_version": FIXED_REVISION_CASE_VERSION,
                    "record_type": "fixed_revision_source",
                    "preregistration_id": contract.preregistration_id,
                    "case_index": case_index,
                    **case.source_record(max_depth=contract.max_depth),
                    "baseline_symbolic_success": baseline_symbolic,
                })
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
                    revise_result_mode=REVISE_RESULT_ANSWER_CONTAINER,
                    action_availability_mode=(
                        ACTION_AVAILABILITY_ACTIVE_FRONTIER
                    ),
                )
                expected_request = build_worker_request(
                    case.state,
                    case.action,
                    child_context_mode=CHILD_CONTEXT_PRECEDING_SIBLINGS,
                    root_result_mode=ROOT_RESULT_VERIFIER_READY,
                    revise_result_mode=REVISE_RESULT_ANSWER_CONTAINER,
                )
                if case.action not in environment.legal_actions(case.state):
                    raise RuntimeError("sealed fixed REVISE became illegal")
                offset = recorder.count(FIXED_REVISION_ROLE)
                worker_client.attempt_observer = recorder.callback(
                    FIXED_REVISION_ROLE
                )
                transition = environment.step(case.state, case.action)
                if transition.worker_request != expected_request:
                    raise RuntimeError("fixed REVISE request changed at execution")
                new_ids = recorder.since(FIXED_REVISION_ROLE, offset)
                if len(new_ids) != 1:
                    raise RuntimeError("fixed REVISE lacks one logical call id")
                payload = (
                    None
                    if transition.worker_payload_json is None
                    else json.loads(transition.worker_payload_json)
                )
                has_container, _ = _answer_container(payload)
                generated = _new_candidate(transition)
                if has_container != (generated is not None):
                    raise RuntimeError("answer container and Candidate differ")
                revision_symbolic: bool | None = None
                cell = "unavailable"
                if generated is not None:
                    revision_symbolic = bool(check_answer(
                        generated.answer,
                        example.expected_answer,
                    ).success)
                    cell = _correctness_cell(
                        baseline_symbolic,
                        revision_symbolic,
                    )
                    correctness_cells[cell] += 1
                    revision_correct += int(revision_symbolic)
                    oracle_correct += int(
                        baseline_symbolic or revision_symbolic
                    )
                else:
                    correctness_cells[cell] += 1
                baseline_correct += int(baseline_symbolic)
                answer_containers += int(has_container)
                generation_errors += int(transition.generation_error)
                record = transition_record(
                    transition,
                    experiment_id=contract.preregistration_id,
                    arm="fixed_revision",
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
                    "schema_version": FIXED_REVISION_TRANSITION_VERSION,
                    "case_index": case_index,
                    "source_episode_index": case.source_episode_index,
                    "source_turn": case.source_turn,
                    "answer_container": has_container,
                    "baseline_symbolic_success": baseline_symbolic,
                    "revision_symbolic_success": revision_symbolic,
                    "correctness_cell": cell,
                    "behavior_cloning_eligible": False,
                })
                _write_record(run_handle, record)
                transition_count += 1
                _write_record(run_handle, {
                    "schema_version": FIXED_REVISION_CASE_VERSION,
                    "record_type": "fixed_revision_case_summary",
                    "preregistration_id": contract.preregistration_id,
                    "case_index": case_index,
                    "problem_id": case.problem_id,
                    "source_episode_index": case.source_episode_index,
                    "source_turn": case.source_turn,
                    "answer_container": has_container,
                    "generation_complete": not transition.generation_error,
                    "baseline_symbolic_success": baseline_symbolic,
                    "revision_symbolic_success": revision_symbolic,
                    "correctness_cell": cell,
                    "worker_call_id": new_ids[0],
                }, flush=True)

            selection_corpus_adequate = (
                answer_containers >= 55
                and correctness_cells["baseline_only"] >= 5
                and correctness_cells["revision_only"] >= 5
            )
            run_summary = {
                "schema_version": FIXED_REVISION_SUMMARY_VERSION,
                "record_type": "run_summary",
                "preregistration_id": contract.preregistration_id,
                "run_contract_sha256": contract.sha256,
                "case_count": len(prepared.cases),
                "transition_count": transition_count,
                "answer_container_count": answer_containers,
                "generation_error_count": generation_errors,
                "correctness_cells": dict(sorted(correctness_cells.items())),
                "baseline_symbolic_success": baseline_correct,
                "revision_symbolic_success": revision_correct,
                "available_candidate_oracle_success": oracle_correct,
                "selection_corpus_adequate": selection_corpus_adequate,
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
    return FixedRevisionCollection(
        run_path=final_run,
        diagnostic_path=final_diagnostic,
        case_count=len(prepared.cases),
        transition_count=transition_count,
        answer_container_count=answer_containers,
        correctness_cells=dict(correctness_cells),
        selection_corpus_adequate=selection_corpus_adequate,
        run_contract_sha256=contract.sha256,
    )


__all__ = [
    "FIXED_REVISION_ADEQUACY_VERSION",
    "FIXED_REVISION_CASE_VERSION",
    "FIXED_REVISION_COLLECTION_PURPOSE",
    "FIXED_REVISION_CONTRACT_VERSION",
    "FIXED_REVISION_DIAGNOSTIC_VERSION",
    "FIXED_REVISION_HEADER_VERSION",
    "FIXED_REVISION_ROLE",
    "FIXED_REVISION_SEED_POLICY_VERSION",
    "FIXED_REVISION_SOURCE_ACTION_VERSION",
    "FIXED_REVISION_SOURCE_VERSION",
    "FIXED_REVISION_SUMMARY_VERSION",
    "FIXED_REVISION_TRANSITION_VERSION",
    "FixedRevisionCase",
    "FixedRevisionCollection",
    "FixedRevisionContract",
    "PreparedFixedRevisions",
    "build_fixed_revision_contract",
    "collect_fixed_revision_run",
    "prepare_fixed_revisions",
]
