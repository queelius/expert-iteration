"""Paired coordinator routing prompts with sealed revision replay."""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..llm_client import OpenAICompatibleLLMClient
from .controller import (
    TEACHER_CHOICE_MENU_INDEX,
    TEACHER_FALLBACK_VERSION,
    TEACHER_INDEX_OUTPUT_VERSION,
    TEACHER_INDEX_PROMPT_VERSION,
    TEACHER_PROMPT_GENERIC,
    TEACHER_PROMPT_MODES,
    TEACHER_PROMPT_REVISION_ROUTING,
    TEACHER_ROUTING_INDEX_PROMPT_VERSION,
    LLMTeacherPolicy,
    build_teacher_request,
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
from .fixed_revision import FIXED_REVISION_SOURCE_VERSION
from .lenses import (
    ACTIVE_FRONTIER_OBSERVATION_SCHEMA_VERSION,
    REVISE_RESULT_INDEPENDENT_ANSWER_CONTAINER,
    TEACHER_PRESENTATION_VERSION,
    build_observation,
    build_worker_request,
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
)
from .paired_lens import (
    _read_canonical_jsonl,
    file_sha256,
    state_from_observation,
)
from .paired_revise import _model_identity, _nonempty, _request_record, _sha256
from .paired_revision_prompt import (
    PAIRED_REVISION_PROMPT_HEADER_VERSION,
    PAIRED_REVISION_PROMPT_TRANSITION_VERSION,
)
from .records import (
    DIAGNOSTIC_ATTEMPT_VERSION,
    action_record,
    canonical_json_sha256,
    teacher_decision_record,
    transition_record,
)
from .runtime import SUBMIT_RECEIPT_VERSION, WorkspaceEnvironment


PAIRED_ROUTING_PROMPT_CONTRACT_VERSION = (
    "workspace-paired-routing-prompt-contract-v1"
)
PAIRED_ROUTING_PROMPT_HEADER_VERSION = (
    "workspace-paired-routing-prompt-header-v1"
)
PAIRED_ROUTING_PROMPT_DIAGNOSTIC_VERSION = (
    "workspace-paired-routing-prompt-diagnostic-header-v1"
)
PAIRED_ROUTING_PROMPT_DECISION_VERSION = (
    "workspace-paired-routing-prompt-decision-v1"
)
PAIRED_ROUTING_PROMPT_CASE_VERSION = (
    "workspace-paired-routing-prompt-case-v1"
)
PAIRED_ROUTING_PROMPT_SUMMARY_VERSION = (
    "workspace-paired-routing-prompt-summary-v1"
)
PAIRED_ROUTING_PROMPT_SOURCE_VERSION = "e41-independent-revision-replay-v1"
PAIRED_ROUTING_PROMPT_FIXED_CONTINUATION_VERSION = (
    "submit-baseline-or-sealed-latest-v1"
)
PAIRED_ROUTING_PROMPT_BRANCH_ORDER_VERSION = (
    "checkerboard-routing-prompt-order-v1"
)
PAIRED_ROUTING_PROMPT_COLLECTION_PURPOSE = "evaluation-only-v1"
PAIRED_ROUTING_PROMPT_MODES = (
    TEACHER_PROMPT_GENERIC,
    TEACHER_PROMPT_REVISION_ROUTING,
)
_ROLE_BY_MODE = {
    TEACHER_PROMPT_GENERIC: "generic_routing_controller",
    TEACHER_PROMPT_REVISION_ROUTING: "explicit_routing_controller",
}


@dataclass(frozen=True, slots=True)
class RoutingReplayCase:
    problem_id: str
    source_episode_index: int
    source_turn: int
    state: State
    revision_action: Action
    revised_state: State
    protected_candidate_id: int
    revised_candidate_id: int
    source_worker_call_id: str
    source_transition_sha256: str

    def __post_init__(self) -> None:
        _nonempty(self.problem_id, "routing problem id")
        if type(self.source_episode_index) is not int or self.source_episode_index < 0:
            raise ValueError("routing source episode must be nonnegative")
        if type(self.source_turn) is not int or self.source_turn < 0:
            raise ValueError("routing source turn must be nonnegative")
        if not isinstance(self.state, State) or self.state.terminal:
            raise ValueError("routing source State must be nonterminal")
        if self.state.turn != self.source_turn:
            raise ValueError("routing source turn differs from State")
        if not isinstance(self.revised_state, State) or self.revised_state.terminal:
            raise ValueError("routing revised State must be nonterminal")
        if self.revised_state.turn != self.state.turn + 1:
            raise ValueError("routing revision must consume one turn")
        if self.revision_action != Action(
            Verb.REVISE,
            self.protected_candidate_id,
        ):
            raise ValueError("routing revision action differs from protected id")
        if protected_direct_candidate_id(self.state) != self.protected_candidate_id:
            raise ValueError("routing protected Candidate differs from source")
        if protected_direct_candidate_id(self.revised_state) != self.protected_candidate_id:
            raise ValueError("routing revision changed protected Candidate")
        protected = node_at(self.state, self.protected_candidate_id)
        revised = node_at(self.revised_state, self.revised_candidate_id)
        if not isinstance(protected, CandidateNode) or protected.goal_id != 0:
            raise ValueError("routing protected node must be a root Candidate")
        if (
            not isinstance(revised, CandidateNode)
            or revised.goal_id != 0
            or revised.revises != self.protected_candidate_id
        ):
            raise ValueError("routing replay lacks a root revision")
        _nonempty(self.source_worker_call_id, "routing source worker call id")
        _sha256(self.source_transition_sha256, "source transition sha256")

    def source_record(self, *, max_depth: int) -> dict[str, object]:
        return {
            "schema_version": PAIRED_ROUTING_PROMPT_SOURCE_VERSION,
            "problem_id": self.problem_id,
            "source_episode_index": self.source_episode_index,
            "source_turn": self.source_turn,
            "source_observation": build_observation(
                self.state,
                max_depth=max_depth,
                action_availability_mode=ACTION_AVAILABILITY_ACTIVE_FRONTIER,
            ),
            "revision_action": action_record(self.revision_action),
            "revised_observation": build_observation(
                self.revised_state,
                max_depth=max_depth,
                action_availability_mode=ACTION_AVAILABILITY_ACTIVE_FRONTIER,
            ),
            "protected_candidate_id": self.protected_candidate_id,
            "revised_candidate_id": self.revised_candidate_id,
            "source_worker_call_id": self.source_worker_call_id,
            "source_transition_sha256": self.source_transition_sha256,
        }


@dataclass(frozen=True, slots=True)
class PreparedRoutingReplays:
    revision_run_path: Path
    revision_diagnostic_path: Path
    revision_run_sha256: str
    revision_diagnostic_sha256: str
    revision_run_contract_sha256: str
    teacher_source_run_path: Path
    teacher_source_run_sha256: str
    teacher_source_run_contract_sha256: str
    selection_sha256: str
    cases: tuple[RoutingReplayCase, ...]
    examples: tuple[ProblemExample, ...]
    replay_worker: ModelIdentity
    teacher: ModelIdentity
    teacher_temperature: float
    teacher_max_tokens: int
    teacher_seed: int
    turn_limit: int
    max_depth: int

    def __post_init__(self) -> None:
        for name in (
            "revision_run_sha256",
            "revision_diagnostic_sha256",
            "revision_run_contract_sha256",
            "teacher_source_run_sha256",
            "teacher_source_run_contract_sha256",
            "selection_sha256",
        ):
            _sha256(getattr(self, name), name.replace("_", " "))
        if not self.cases or len(self.cases) != len(self.examples):
            raise ValueError("routing replay preparation must cover source")
        if not isinstance(self.replay_worker, ModelIdentity):
            raise ValueError("routing replay worker identity is invalid")
        if not isinstance(self.teacher, ModelIdentity):
            raise ValueError("routing teacher identity is invalid")


def prepare_routing_replays(
    *,
    revision_run_path: str | Path,
    revision_diagnostic_path: str | Path,
    expected_revision_run_contract_sha256: str,
    teacher_source_run_path: str | Path,
    expected_teacher_source_run_sha256: str,
    expected_teacher_source_run_contract_sha256: str,
    examples: tuple[ProblemExample, ...],
    max_depth: int,
) -> PreparedRoutingReplays:
    """Seal E41 independent revisions and E39 coordinator settings."""
    for digest, name in (
        (
            expected_revision_run_contract_sha256,
            "expected revision contract sha256",
        ),
        (expected_teacher_source_run_sha256, "expected teacher source sha256"),
        (
            expected_teacher_source_run_contract_sha256,
            "expected teacher source contract sha256",
        ),
    ):
        _sha256(digest, name)
    if not examples:
        raise ValueError("routing replay preparation requires examples")
    revision_path = Path(revision_run_path)
    diagnostic_path = Path(revision_diagnostic_path)
    teacher_path = Path(teacher_source_run_path)
    records = _read_canonical_jsonl(revision_path)
    diagnostics = _read_canonical_jsonl(diagnostic_path)
    teacher_records = _read_canonical_jsonl(teacher_path)
    header = records[0]
    if (
        header.get("record_type") != "run_header"
        or header.get("schema_version") != PAIRED_REVISION_PROMPT_HEADER_VERSION
        or header.get("run_contract_sha256")
        != expected_revision_run_contract_sha256
    ):
        raise ValueError("revision source is not the expected paired prompt run")
    source = header.get("source")
    models = header.get("models")
    if type(source) is not dict or type(models) is not dict:
        raise ValueError("revision source header is invalid")
    if (
        source.get("problem_source_sha256") != problem_source_sha256(examples)
        or source.get("evaluation_binding_sha256")
        != evaluation_binding_sha256(examples)
        or tuple(source.get("problem_ids", ()))
        != tuple(example.problem_id for example in examples)
    ):
        raise ValueError("revision source problem binding differs")
    replay_worker = _model_identity(models.get("worker"))

    teacher_header = teacher_records[0]
    if (
        file_sha256(teacher_path) != expected_teacher_source_run_sha256
        or teacher_header.get("record_type") != "run_header"
        or teacher_header.get("schema_version")
        != PAIRED_DIRECT_TEACHER_REVISE_HEADER_VERSION
        or teacher_header.get("run_contract_sha256")
        != expected_teacher_source_run_contract_sha256
    ):
        raise ValueError("teacher source is not the expected E39 run")
    teacher_models = teacher_header.get("models")
    teacher_settings = teacher_header.get("settings")
    teacher_contracts = teacher_header.get("contracts")
    if not all(
        type(value) is dict
        for value in (teacher_models, teacher_settings, teacher_contracts)
    ):
        raise ValueError("teacher source header containers are invalid")
    assert isinstance(teacher_models, dict)
    assert isinstance(teacher_settings, dict)
    assert isinstance(teacher_contracts, dict)
    if (
        teacher_settings.get("teacher_choice_mode")
        != TEACHER_CHOICE_MENU_INDEX
        or teacher_settings.get("action_availability_mode")
        != ACTION_AVAILABILITY_ACTIVE_FRONTIER
        or teacher_settings.get("max_depth") != max_depth
        or teacher_contracts.get("teacher_prompt")
        != TEACHER_INDEX_PROMPT_VERSION
    ):
        raise ValueError("teacher source does not use the sealed indexed planner")
    teacher = _model_identity(teacher_models.get("teacher"))
    teacher_temperature = teacher_settings.get("teacher_temperature")
    teacher_max_tokens = teacher_settings.get("teacher_max_tokens")
    teacher_seed = teacher_settings.get("teacher_seed")
    turn_limit = teacher_settings.get("turn_limit")
    if (
        type(teacher_temperature) is not float
        or type(teacher_max_tokens) is not int
        or type(teacher_seed) is not int
        or type(turn_limit) is not int
    ):
        raise ValueError("teacher source sampling settings are invalid")

    diagnostic_header = diagnostics[0]
    if (
        diagnostic_header.get("record_type") != "diagnostic_header"
        or diagnostic_header.get("run_contract_sha256")
        != expected_revision_run_contract_sha256
    ):
        raise ValueError("revision diagnostics differ from source contract")
    attempts_by_call: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in diagnostics[1:]:
        if record.get("record_type") != "http_attempt":
            raise ValueError("revision diagnostics contain an unknown record")
        attempt = record.get("attempt")
        if type(attempt) is not dict:
            raise ValueError("revision diagnostic attempt is invalid")
        call_id = _nonempty(attempt.get("call_id"), "revision diagnostic call id")
        attempts_by_call[call_id].append(record)

    sources: dict[int, dict[str, object]] = {}
    independent: dict[int, dict[str, object]] = {}
    for record in records[1:]:
        case_index = record.get("case_index")
        if record.get("record_type") == "paired_revision_prompt_source":
            if type(case_index) is not int or case_index in sources:
                raise ValueError("revision source case index is invalid")
            sources[case_index] = record
        elif (
            record.get("schema_version")
            == PAIRED_REVISION_PROMPT_TRANSITION_VERSION
            and record.get("arm")
            == REVISE_RESULT_INDEPENDENT_ANSWER_CONTAINER
        ):
            if type(case_index) is not int or case_index in independent:
                raise ValueError("independent transition case index is invalid")
            independent[case_index] = record
    expected_indexes = set(range(len(examples)))
    if set(sources) != expected_indexes or set(independent) != expected_indexes:
        raise ValueError("revision source does not contain every case")

    cases: list[RoutingReplayCase] = []
    for case_index, example in enumerate(examples):
        source_record = sources[case_index]
        transition = independent[case_index]
        if (
            source_record.get("problem_id") != example.problem_id
            or transition.get("problem_id") != example.problem_id
            or transition.get("behavior_cloning_eligible") is not False
            or transition.get("generation_error") is not False
            or transition.get("answer_container") is not True
        ):
            raise ValueError("independent replay provenance is invalid")
        state = state_from_observation(
            source_record.get("source_observation"),
            max_depth=max_depth,
            action_availability_mode=ACTION_AVAILABILITY_ACTIVE_FRONTIER,
        )
        if transition.get("observation") != source_record.get("source_observation"):
            raise ValueError("independent replay source observation drifted")
        protected_id = protected_direct_candidate_id(state)
        if protected_id is None:
            raise ValueError("routing source lacks protected Candidate")
        revision_action = Action(Verb.REVISE, protected_id)
        if transition.get("action") != action_record(revision_action):
            raise ValueError("independent replay action differs")
        expected_request = build_worker_request(
            state,
            revision_action,
            child_context_mode="preceding-siblings-v1",
            root_result_mode="verifier-ready-root-answer-v1",
            revise_result_mode=REVISE_RESULT_INDEPENDENT_ANSWER_CONTAINER,
        )
        if transition.get("worker_request") != _request_record(expected_request):
            raise ValueError("independent replay request differs")
        revised_state = state_from_observation(
            transition.get("next_observation"),
            max_depth=max_depth,
            action_availability_mode=ACTION_AVAILABILITY_ACTIVE_FRONTIER,
        )
        created_ids = revised_state.history[-1].created_ids
        if len(created_ids) != 1:
            raise ValueError("independent replay lacks one created Candidate")
        revised_id = created_ids[0]
        if Action(Verb.SUBMIT, revised_id) not in actions_for_mode(
            revised_state,
            max_depth=max_depth,
            mode=ACTION_AVAILABILITY_ACTIVE_FRONTIER,
        ):
            raise ValueError("independent replay Candidate is not submittable")
        call_id = _nonempty(
            transition.get("worker_call_id"),
            "independent source worker call id",
        )
        call_attempts = attempts_by_call.get(call_id, [])
        if not call_attempts or any(
            attempt.get("role") != "independent_revision_worker"
            for attempt in call_attempts
        ):
            raise ValueError("independent replay lacks bound diagnostics")
        cases.append(RoutingReplayCase(
            problem_id=example.problem_id,
            source_episode_index=case_index,
            source_turn=state.turn,
            state=state,
            revision_action=revision_action,
            revised_state=revised_state,
            protected_candidate_id=protected_id,
            revised_candidate_id=revised_id,
            source_worker_call_id=call_id,
            source_transition_sha256=canonical_json_sha256(transition),
        ))

    selection_record = {
        "schema_version": PAIRED_ROUTING_PROMPT_SOURCE_VERSION,
        "revision_run_sha256": file_sha256(revision_path),
        "revision_diagnostic_sha256": file_sha256(diagnostic_path),
        "revision_run_contract_sha256": expected_revision_run_contract_sha256,
        "teacher_source_run_sha256": expected_teacher_source_run_sha256,
        "teacher_source_run_contract_sha256": (
            expected_teacher_source_run_contract_sha256
        ),
        "cases": [case.source_record(max_depth=max_depth) for case in cases],
    }
    return PreparedRoutingReplays(
        revision_run_path=revision_path,
        revision_diagnostic_path=diagnostic_path,
        revision_run_sha256=selection_record["revision_run_sha256"],
        revision_diagnostic_sha256=selection_record[
            "revision_diagnostic_sha256"
        ],
        revision_run_contract_sha256=expected_revision_run_contract_sha256,
        teacher_source_run_path=teacher_path,
        teacher_source_run_sha256=expected_teacher_source_run_sha256,
        teacher_source_run_contract_sha256=(
            expected_teacher_source_run_contract_sha256
        ),
        selection_sha256=canonical_json_sha256(selection_record),
        cases=tuple(cases),
        examples=examples,
        replay_worker=replay_worker,
        teacher=teacher,
        teacher_temperature=teacher_temperature,
        teacher_max_tokens=teacher_max_tokens,
        teacher_seed=teacher_seed,
        turn_limit=turn_limit,
        max_depth=max_depth,
    )


@dataclass(frozen=True, slots=True)
class PairedRoutingPromptContract:
    preregistration_id: str
    revision_run_sha256: str
    revision_diagnostic_sha256: str
    revision_run_contract_sha256: str
    teacher_source_run_sha256: str
    teacher_source_run_contract_sha256: str
    selection_sha256: str
    problem_source_sha256: str
    evaluation_binding_sha256: str
    case_keys: tuple[tuple[int, int], ...]
    problem_ids: tuple[str, ...]
    replay_worker: ModelIdentity
    teacher: ModelIdentity
    teacher_temperature: float
    teacher_max_tokens: int
    teacher_seed: int
    turn_limit: int
    max_depth: int
    modes: tuple[str, str] = PAIRED_ROUTING_PROMPT_MODES
    branch_order_version: str = PAIRED_ROUTING_PROMPT_BRANCH_ORDER_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.preregistration_id, "preregistration id")
        for name in (
            "revision_run_sha256",
            "revision_diagnostic_sha256",
            "revision_run_contract_sha256",
            "teacher_source_run_sha256",
            "teacher_source_run_contract_sha256",
            "selection_sha256",
            "problem_source_sha256",
            "evaluation_binding_sha256",
        ):
            _sha256(getattr(self, name), name.replace("_", " "))
        if not self.case_keys or len(self.case_keys) != len(set(self.case_keys)):
            raise ValueError("routing case keys must be nonempty and unique")
        if len(self.problem_ids) != len(self.case_keys):
            raise ValueError("routing problem ids differ from case keys")
        if self.modes != PAIRED_ROUTING_PROMPT_MODES:
            raise ValueError("routing prompt modes differ")
        if self.branch_order_version != PAIRED_ROUTING_PROMPT_BRANCH_ORDER_VERSION:
            raise ValueError("routing branch order differs")
        if not isinstance(self.replay_worker, ModelIdentity):
            raise ValueError("routing replay worker identity is invalid")
        if not isinstance(self.teacher, ModelIdentity):
            raise ValueError("routing teacher identity is invalid")
        if type(self.teacher_temperature) is not float or self.teacher_temperature < 0:
            raise ValueError("routing teacher temperature must be nonnegative")
        if type(self.teacher_max_tokens) is not int or self.teacher_max_tokens <= 0:
            raise ValueError("routing teacher max tokens must be positive")
        if type(self.teacher_seed) is not int or self.teacher_seed < 0:
            raise ValueError("routing teacher seed must be nonnegative")
        if type(self.turn_limit) is not int or self.turn_limit <= 0:
            raise ValueError("routing turn limit must be positive")
        if type(self.max_depth) is not int or self.max_depth <= 0:
            raise ValueError("routing max depth must be positive")

    def header(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema_version": PAIRED_ROUTING_PROMPT_HEADER_VERSION,
            "record_type": "run_header",
            "run_contract_version": PAIRED_ROUTING_PROMPT_CONTRACT_VERSION,
            "preregistration_id": self.preregistration_id,
            "source": {
                "schema_version": PAIRED_ROUTING_PROMPT_SOURCE_VERSION,
                "revision_run_sha256": self.revision_run_sha256,
                "revision_diagnostic_sha256": self.revision_diagnostic_sha256,
                "revision_run_contract_sha256": (
                    self.revision_run_contract_sha256
                ),
                "teacher_source_run_sha256": self.teacher_source_run_sha256,
                "teacher_source_run_contract_sha256": (
                    self.teacher_source_run_contract_sha256
                ),
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
                "reward": REWARD_CONTRACT_VERSION,
                "decision_record": PAIRED_ROUTING_PROMPT_DECISION_VERSION,
                "observation": ACTIVE_FRONTIER_OBSERVATION_SCHEMA_VERSION,
                "teacher_presentation": TEACHER_PRESENTATION_VERSION,
                "teacher_output": TEACHER_INDEX_OUTPUT_VERSION,
                "teacher_fallback": TEACHER_FALLBACK_VERSION,
                "generic_prompt": TEACHER_INDEX_PROMPT_VERSION,
                "routing_prompt": TEACHER_ROUTING_INDEX_PROMPT_VERSION,
                "fixed_continuation": (
                    PAIRED_ROUTING_PROMPT_FIXED_CONTINUATION_VERSION
                ),
                "submit_receipt": SUBMIT_RECEIPT_VERSION,
                "branch_order": self.branch_order_version,
                "collection_purpose": (
                    PAIRED_ROUTING_PROMPT_COLLECTION_PURPOSE
                ),
            },
            "settings": {
                "case_count": len(self.case_keys),
                "turn_limit": self.turn_limit,
                "max_depth": self.max_depth,
                "action_availability_mode": (
                    ACTION_AVAILABILITY_ACTIVE_FRONTIER
                ),
                "teacher_choice_mode": TEACHER_CHOICE_MENU_INDEX,
                "prompt_modes": list(self.modes),
                "teacher_temperature": self.teacher_temperature,
                "teacher_max_tokens": self.teacher_max_tokens,
                "teacher_seed": self.teacher_seed,
            },
            "models": {
                "teacher": self.teacher.to_record(),
                "replay_worker": self.replay_worker.to_record(),
            },
        }
        body["run_contract_sha256"] = canonical_json_sha256(body)
        return body

    @property
    def sha256(self) -> str:
        return str(self.header()["run_contract_sha256"])


def build_paired_routing_prompt_contract(
    *,
    preregistration_id: str,
    prepared: PreparedRoutingReplays,
) -> PairedRoutingPromptContract:
    return PairedRoutingPromptContract(
        preregistration_id=preregistration_id,
        revision_run_sha256=prepared.revision_run_sha256,
        revision_diagnostic_sha256=prepared.revision_diagnostic_sha256,
        revision_run_contract_sha256=prepared.revision_run_contract_sha256,
        teacher_source_run_sha256=prepared.teacher_source_run_sha256,
        teacher_source_run_contract_sha256=(
            prepared.teacher_source_run_contract_sha256
        ),
        selection_sha256=prepared.selection_sha256,
        problem_source_sha256=problem_source_sha256(prepared.examples),
        evaluation_binding_sha256=evaluation_binding_sha256(prepared.examples),
        case_keys=tuple(
            (case.source_episode_index, case.source_turn)
            for case in prepared.cases
        ),
        problem_ids=tuple(case.problem_id for case in prepared.cases),
        replay_worker=prepared.replay_worker,
        teacher=prepared.teacher,
        teacher_temperature=prepared.teacher_temperature,
        teacher_max_tokens=prepared.teacher_max_tokens,
        teacher_seed=prepared.teacher_seed,
        turn_limit=prepared.turn_limit,
        max_depth=prepared.max_depth,
    )


def _branch_order(case_index: int) -> tuple[str, str]:
    if case_index % 2 == 0:
        return PAIRED_ROUTING_PROMPT_MODES
    return tuple(reversed(PAIRED_ROUTING_PROMPT_MODES))  # type: ignore[return-value]


def _paired_outcome(values: Mapping[str, int]) -> str:
    generic = values[TEACHER_PROMPT_GENERIC]
    routing = values[TEACHER_PROMPT_REVISION_ROUTING]
    if generic and routing:
        return "both"
    if generic:
        return "generic_only"
    if routing:
        return "routing_only"
    return "neither"


@dataclass(frozen=True, slots=True)
class PairedRoutingPromptCollection:
    run_path: Path
    diagnostic_path: Path
    case_count: int
    decision_count: int
    solved_by_mode: Mapping[str, int]
    paired_outcomes: Mapping[str, int]
    run_contract_sha256: str


def collect_paired_routing_prompt_run(
    *,
    run_path: str | Path,
    diagnostic_path: str | Path,
    contract: PairedRoutingPromptContract,
    prepared: PreparedRoutingReplays,
    teacher_client: OpenAICompatibleLLMClient,
) -> PairedRoutingPromptCollection:
    """Pair generic and explicit routing prompts with fixed outcomes."""
    if not isinstance(contract, PairedRoutingPromptContract):
        raise TypeError("routing collection requires its run contract")
    if not isinstance(prepared, PreparedRoutingReplays):
        raise TypeError("routing collection requires prepared replays")
    if type(teacher_client) is not OpenAICompatibleLLMClient:
        raise TypeError("routing teacher must be OpenAI-compatible")
    _validate_client_binding(contract.teacher, teacher_client, role="teacher")
    expected = build_paired_routing_prompt_contract(
        preregistration_id=contract.preregistration_id,
        prepared=prepared,
    )
    if expected != contract:
        raise ValueError("routing contract differs from prepared replays")

    final_run = Path(run_path)
    final_diagnostic = Path(diagnostic_path)
    partial_run, partial_diagnostic = _validate_output_paths(
        final_run,
        final_diagnostic,
    )
    diagnostic_header = {
        "schema_version": PAIRED_ROUTING_PROMPT_DIAGNOSTIC_VERSION,
        "record_type": "diagnostic_header",
        "preregistration_id": contract.preregistration_id,
        "run_contract_sha256": contract.sha256,
        "attempt_schema_version": DIAGNOSTIC_ATTEMPT_VERSION,
        "roles": list(_ROLE_BY_MODE.values()),
    }
    examples = {example.problem_id: example for example in prepared.examples}
    teacher_before = _counter_snapshot(teacher_client)
    old_observer = teacher_client.attempt_observer
    solved = Counter({mode: 0 for mode in contract.modes})
    action_counts = {mode: Counter() for mode in contract.modes}
    route_counts = {mode: Counter() for mode in contract.modes}
    fallbacks = Counter({mode: 0 for mode in contract.modes})
    paired_outcomes: Counter[str] = Counter()
    decision_count = 0

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
                _write_record(run_handle, {
                    "schema_version": PAIRED_ROUTING_PROMPT_CASE_VERSION,
                    "record_type": "paired_routing_source",
                    "preregistration_id": contract.preregistration_id,
                    "case_index": case_index,
                    **case.source_record(max_depth=contract.max_depth),
                })
                environment = WorkspaceEnvironment(
                    problem=example.problem,
                    expected_answer=example.expected_answer,
                    worker=teacher_client,
                    turn_limit=contract.turn_limit,
                    max_depth=contract.max_depth,
                    root_result_mode="verifier-ready-root-answer-v1",
                    revise_result_mode=(
                        REVISE_RESULT_INDEPENDENT_ANSWER_CONTAINER
                    ),
                    action_availability_mode=(
                        ACTION_AVAILABILITY_ACTIVE_FRONTIER
                    ),
                )
                legal = environment.legal_actions(case.state)
                if legal != actions_for_mode(
                    case.state,
                    max_depth=contract.max_depth,
                    mode=ACTION_AVAILABILITY_ACTIVE_FRONTIER,
                ):
                    raise RuntimeError("routing legal menu changed")
                expected_requests = {
                    mode: build_teacher_request(
                        case.state,
                        max_depth=contract.max_depth,
                        action_availability_mode=(
                            ACTION_AVAILABILITY_ACTIVE_FRONTIER
                        ),
                        choice_mode=TEACHER_CHOICE_MENU_INDEX,
                        prompt_mode=mode,
                    )
                    for mode in contract.modes
                }
                generic_request = expected_requests[TEACHER_PROMPT_GENERIC]
                routing_request = expected_requests[
                    TEACHER_PROMPT_REVISION_ROUTING
                ]
                if (
                    generic_request.system == routing_request.system
                    or generic_request.user != routing_request.user
                    or generic_request.schema != routing_request.schema
                ):
                    raise RuntimeError("routing prompts are not isolated")
                order = _branch_order(case_index)
                positions = {mode: index for index, mode in enumerate(order)}
                case_rewards: dict[str, int] = {}
                call_ids: dict[str, str] = {}

                for mode in order:
                    policy = LLMTeacherPolicy(
                        client=teacher_client,
                        max_depth=contract.max_depth,
                        temperature=contract.teacher_temperature,
                        max_tokens=contract.teacher_max_tokens,
                        base_seed=(
                            contract.teacher_seed
                            + case.source_episode_index * contract.turn_limit
                        ),
                        action_availability_mode=(
                            ACTION_AVAILABILITY_ACTIVE_FRONTIER
                        ),
                        choice_mode=TEACHER_CHOICE_MENU_INDEX,
                        prompt_mode=mode,
                    )
                    role = _ROLE_BY_MODE[mode]
                    offset = recorder.count(role)
                    teacher_client.attempt_observer = recorder.callback(role)
                    action = policy.choose(case.state, legal)
                    new_ids = recorder.since(role, offset)
                    if len(new_ids) != 1:
                        raise RuntimeError("routing decision lacks one call id")
                    call_id = new_ids[0]
                    call_ids[mode] = call_id
                    decision = policy.decisions[-1]
                    if decision.selected_action != action:
                        raise RuntimeError("routing decision/action mismatch")
                    if decision.request != expected_requests[mode]:
                        raise RuntimeError("routing policy request drifted")
                    decision_count += 1
                    fallbacks[mode] += int(decision.fallback)
                    action_counts[mode][action.verb.value] += 1

                    terminal = None
                    route = "nonterminal_action"
                    if action == Action(Verb.SUBMIT, case.protected_candidate_id):
                        terminal = environment.step(case.state, action)
                        route = "submit_protected"
                    elif action == case.revision_action:
                        submit_latest = Action(
                            Verb.SUBMIT,
                            case.revised_candidate_id,
                        )
                        terminal = environment.step(
                            case.revised_state,
                            submit_latest,
                        )
                        route = "replay_revision_submit_latest"
                    reward = 0 if terminal is None else terminal.reward
                    solved[mode] += reward
                    route_counts[mode][route] += 1
                    case_rewards[mode] = reward
                    terminal_record = (
                        None
                        if terminal is None
                        else transition_record(
                            terminal,
                            experiment_id=contract.preregistration_id,
                            arm=mode,
                            episode_index=case.source_episode_index,
                            problem_id=case.problem_id,
                            max_depth=contract.max_depth,
                            worker_call_id=None,
                            teacher_call_id=call_id,
                            teacher_decision=decision,
                            action_availability_mode=(
                                ACTION_AVAILABILITY_ACTIVE_FRONTIER
                            ),
                        )
                    )
                    _write_record(run_handle, {
                        "schema_version": (
                            PAIRED_ROUTING_PROMPT_DECISION_VERSION
                        ),
                        "record_type": "routing_decision",
                        "preregistration_id": contract.preregistration_id,
                        "case_index": case_index,
                        "problem_id": case.problem_id,
                        "source_episode_index": case.source_episode_index,
                        "source_turn": case.source_turn,
                        "arm": mode,
                        "branch_position": positions[mode],
                        "teacher_call_id": call_id,
                        "teacher_decision": teacher_decision_record(decision),
                        "routing_action": action_record(action),
                        "fixed_continuation": route,
                        "terminal_submit_transition": terminal_record,
                        "reward": reward,
                        "behavior_cloning_eligible": False,
                    })

                outcome = _paired_outcome(case_rewards)
                paired_outcomes[outcome] += 1
                _write_record(run_handle, {
                    "schema_version": PAIRED_ROUTING_PROMPT_CASE_VERSION,
                    "record_type": "paired_routing_case_summary",
                    "preregistration_id": contract.preregistration_id,
                    "case_index": case_index,
                    "problem_id": case.problem_id,
                    "source_episode_index": case.source_episode_index,
                    "rewards": case_rewards,
                    "paired_outcome": outcome,
                    "teacher_call_ids": call_ids,
                }, flush=True)

            run_summary = {
                "schema_version": PAIRED_ROUTING_PROMPT_SUMMARY_VERSION,
                "record_type": "run_summary",
                "preregistration_id": contract.preregistration_id,
                "run_contract_sha256": contract.sha256,
                "case_count": len(prepared.cases),
                "decision_count": decision_count,
                "solved_by_mode": dict(solved),
                "paired_outcomes": dict(sorted(paired_outcomes.items())),
                "action_counts_by_mode": {
                    mode: dict(counts) for mode, counts in action_counts.items()
                },
                "route_counts_by_mode": {
                    mode: dict(counts) for mode, counts in route_counts.items()
                },
                "fallbacks_by_mode": dict(fallbacks),
                "usage": {"teacher": _counter_delta(teacher_before, teacher_client)},
            }
            _write_record(run_handle, run_summary)
            run_handle.flush()
            diagnostic_handle.flush()
            os.fsync(run_handle.fileno())
            os.fsync(diagnostic_handle.fileno())
    finally:
        teacher_client.attempt_observer = old_observer

    _finalize_no_clobber((
        (partial_diagnostic, final_diagnostic),
        (partial_run, final_run),
    ))
    return PairedRoutingPromptCollection(
        run_path=final_run,
        diagnostic_path=final_diagnostic,
        case_count=len(prepared.cases),
        decision_count=decision_count,
        solved_by_mode=dict(solved),
        paired_outcomes=dict(paired_outcomes),
        run_contract_sha256=contract.sha256,
    )


__all__ = [
    "PAIRED_ROUTING_PROMPT_BRANCH_ORDER_VERSION",
    "PAIRED_ROUTING_PROMPT_CASE_VERSION",
    "PAIRED_ROUTING_PROMPT_COLLECTION_PURPOSE",
    "PAIRED_ROUTING_PROMPT_CONTRACT_VERSION",
    "PAIRED_ROUTING_PROMPT_DECISION_VERSION",
    "PAIRED_ROUTING_PROMPT_DIAGNOSTIC_VERSION",
    "PAIRED_ROUTING_PROMPT_FIXED_CONTINUATION_VERSION",
    "PAIRED_ROUTING_PROMPT_HEADER_VERSION",
    "PAIRED_ROUTING_PROMPT_MODES",
    "PAIRED_ROUTING_PROMPT_SOURCE_VERSION",
    "PAIRED_ROUTING_PROMPT_SUMMARY_VERSION",
    "PairedRoutingPromptCollection",
    "PairedRoutingPromptContract",
    "PreparedRoutingReplays",
    "RoutingReplayCase",
    "build_paired_routing_prompt_contract",
    "collect_paired_routing_prompt_run",
    "prepare_routing_replays",
]
