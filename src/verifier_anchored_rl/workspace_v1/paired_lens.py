"""Shared-prefix paired experiments for workspace child context modes."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..llm_client import OpenAICompatibleLLMClient
from .controller import TEACHER_OUTPUT_VERSION, TEACHER_PROMPT_VERSION
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
    CHILD_CONTEXT_TARGET_ONLY,
    DIRECT_PROMPT_VERSION,
    HYBRID_LENS_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    TEACHER_PRESENTATION_VERSION,
    WORKER_PROMPT_VERSION,
    build_observation,
)
from .model import (
    ACTION_AVAILABILITY_ALL,
    ACTION_SCHEMA_VERSION,
    REWARD_CONTRACT_VERSION,
    STATE_SCHEMA_VERSION,
    TRANSITION_SCHEMA_VERSION,
    Action,
    ActionEvent,
    CandidateNode,
    EventOutcome,
    GoalNode,
    State,
    Verb,
    child_goals,
    goal_candidates,
    validate_action_availability_mode,
)
from .policies import FixedDecompositionPolicy
from .records import (
    DIAGNOSTIC_ATTEMPT_VERSION,
    canonical_json_bytes,
    canonical_json_sha256,
    transition_record,
)
from .runtime import (
    SUBMIT_RECEIPT_VERSION,
    WORKER_EXECUTION_CONTRACT_VERSION,
    WorkspaceEnvironment,
)


PAIRED_LENS_CONTRACT_VERSION = "workspace-paired-lens-contract-v1"
PAIRED_LENS_HEADER_VERSION = "workspace-paired-lens-header-v1"
PAIRED_LENS_DIAGNOSTIC_HEADER_VERSION = (
    "workspace-paired-lens-diagnostic-header-v1"
)
PAIRED_PREFIX_VERSION = "workspace-shared-prefix-source-v1"
PAIRED_CASE_VERSION = "workspace-paired-lens-case-v1"
PAIRED_CASE_SUMMARY_VERSION = "workspace-paired-lens-case-summary-v1"
PAIRED_RUN_SUMMARY_VERSION = "workspace-paired-lens-summary-v1"
PAIRED_BRANCH_ORDER_VERSION = "checkerboard-case-round-order-v1"
PAIRED_SEED_POLICY_VERSION = "source-episode-turn-offset-v1"
PAIRED_MODES = (
    CHILD_CONTEXT_TARGET_ONLY,
    CHILD_CONTEXT_PRECEDING_SIBLINGS,
)
_ROLE_BY_MODE = {
    CHILD_CONTEXT_TARGET_ONLY: "target_only_worker",
    CHILD_CONTEXT_PRECEDING_SIBLINGS: "preceding_siblings_worker",
}


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _sha256(value: str, name: str) -> str:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase sha256")
    return value


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_canonical_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith(b"\n") or line.endswith(b"\r\n"):
                raise ValueError(
                    f"source line {line_number} has invalid newline framing"
                )
            try:
                record = json.loads(line[:-1].decode("ascii"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"source line {line_number} is not canonical JSON"
                ) from exc
            if type(record) is not dict:
                raise ValueError(f"source line {line_number} is not an object")
            if canonical_json_bytes(record) != line[:-1]:
                raise ValueError(
                    f"source line {line_number} is not canonically encoded"
                )
            records.append(record)
    if not records:
        raise ValueError("source run is empty")
    return tuple(records)


def state_from_observation(
    observation: Mapping[str, object],
    *,
    max_depth: int,
    action_availability_mode: str = ACTION_AVAILABILITY_ALL,
) -> State:
    """Reconstruct and exact-check one canonical workspace observation."""
    if not isinstance(observation, Mapping):
        raise ValueError("source observation must be an object")
    validate_action_availability_mode(action_availability_mode)
    expected_observation_version = (
        OBSERVATION_SCHEMA_VERSION
        if action_availability_mode == ACTION_AVAILABILITY_ALL
        else ACTIVE_FRONTIER_OBSERVATION_SCHEMA_VERSION
    )
    if observation.get("schema_version") != expected_observation_version:
        raise ValueError("source observation version is unsupported")
    raw_nodes = observation.get("nodes")
    raw_history = observation.get("history")
    if type(raw_nodes) is not list or type(raw_history) is not list:
        raise ValueError("source observation containers are invalid")
    nodes: list[GoalNode | CandidateNode] = []
    for raw in raw_nodes:
        if type(raw) is not dict:
            raise ValueError("source node is not an object")
        if raw.get("kind") == "goal":
            node = GoalNode(
                id=raw["id"],
                parent_goal=raw["parent_goal"],
                task=raw["task"],
                created_turn=raw["created_turn"],
            )
        elif raw.get("kind") == "candidate":
            node = CandidateNode(
                id=raw["id"],
                goal_id=raw["goal_id"],
                reasoning=raw["reasoning"],
                answer=raw["answer"],
                created_turn=raw["created_turn"],
                revises=raw["revises"],
            )
        else:
            raise ValueError("source node kind is unsupported")
        nodes.append(node)
    history: list[ActionEvent] = []
    for raw in raw_history:
        if type(raw) is not dict or type(raw.get("action")) is not dict:
            raise ValueError("source history event is invalid")
        action = raw["action"]
        history.append(ActionEvent(
            turn=raw["turn"],
            action=Action(Verb(action["verb"]), action["target"]),
            outcome=EventOutcome(raw["outcome"]),
            created_ids=tuple(raw["created_ids"]),
        ))
    state = State(
        nodes=tuple(nodes),
        history=tuple(history),
        turn=observation["turn"],
        turn_limit=observation["turn_limit"],
        terminal=observation["terminal"],
        reward=observation["reward"],
    )
    if build_observation(
        state,
        max_depth=max_depth,
        action_availability_mode=action_availability_mode,
    ) != dict(observation):
        raise ValueError("source observation does not reconstruct exactly")
    return state


@dataclass(frozen=True, slots=True)
class PrefixCase:
    problem_id: str
    source_episode_index: int
    prefix_state: State
    source_decompose_call_id: str
    source_first_child_call_id: str

    def __post_init__(self) -> None:
        _nonempty(self.problem_id, "prefix problem id")
        if type(self.source_episode_index) is not int or self.source_episode_index < 0:
            raise ValueError("source episode index must be nonnegative")
        if not isinstance(self.prefix_state, State) or self.prefix_state.terminal:
            raise ValueError("prefix state must be nonterminal")
        _nonempty(self.source_decompose_call_id, "decompose call id")
        _nonempty(self.source_first_child_call_id, "first-child call id")

    def source_record(self, *, max_depth: int) -> dict[str, object]:
        return {
            "problem_id": self.problem_id,
            "source_episode_index": self.source_episode_index,
            "source_decompose_call_id": self.source_decompose_call_id,
            "source_first_child_call_id": self.source_first_child_call_id,
            "prefix_observation": build_observation(
                self.prefix_state,
                max_depth=max_depth,
            ),
        }


@dataclass(frozen=True, slots=True)
class PreparedPrefixes:
    source_run_path: Path
    source_run_sha256: str
    source_run_contract_sha256: str
    prefix_sha256: str
    cases: tuple[PrefixCase, ...]
    examples: tuple[ProblemExample, ...]
    excluded_problem_ids: tuple[str, ...]
    max_depth: int

    def __post_init__(self) -> None:
        _sha256(self.source_run_sha256, "source run sha256")
        _sha256(self.source_run_contract_sha256, "source contract sha256")
        _sha256(self.prefix_sha256, "prefix sha256")
        if not self.cases or len(self.cases) != len(self.examples):
            raise ValueError("prepared prefix cases and examples differ")
        if tuple(case.problem_id for case in self.cases) != tuple(
            example.problem_id for example in self.examples
        ):
            raise ValueError("prepared prefix problem order differs")


def prepare_paired_prefixes(
    *,
    source_run_path: str | Path,
    examples: tuple[ProblemExample, ...],
    expected_source_run_contract_sha256: str,
    max_depth: int,
) -> PreparedPrefixes:
    """Strict-read E32 and select only structural pre-treatment prefixes."""
    _sha256(
        expected_source_run_contract_sha256,
        "expected source run contract sha256",
    )
    if not examples:
        raise ValueError("paired prefix preparation requires examples")
    path = Path(source_run_path)
    records = _read_canonical_jsonl(path)
    header = records[0]
    if header.get("record_type") != "run_header":
        raise ValueError("source run lacks its header")
    if header.get("run_contract_sha256") != expected_source_run_contract_sha256:
        raise ValueError("source run contract differs from the expected digest")
    if header.get("policy_name") != "fixed_decomposition":
        raise ValueError("source run is not fixed decomposition")
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
        if record.get("schema_version") == "workspace-transition-record-v1":
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

    example_by_id = {example.problem_id: example for example in examples}
    cases: list[PrefixCase] = []
    selected_examples: list[ProblemExample] = []
    excluded: list[str] = []
    for expected_episode, problem_id in enumerate(expected_ids):
        rows = transitions.get(problem_id, [])
        if not rows:
            raise ValueError("source episode has no transitions")
        previous_next: dict[str, object] | None = None
        for turn, row in enumerate(rows):
            if row.get("episode_index") != expected_episode or row.get("turn") != turn:
                raise ValueError("source transition order differs")
            if previous_next is not None and row.get("observation") != previous_next:
                raise ValueError("source transition chain is broken")
            state_from_observation(row["observation"], max_depth=max_depth)
            state_from_observation(row["next_observation"], max_depth=max_depth)
            previous_next = row["next_observation"]
        summary = summaries[expected_episode]
        if previous_next is None or previous_next.get("reward") != summary.get("reward"):
            raise ValueError("source episode summary differs from terminal state")
        decompose = rows[0]
        if decompose.get("action") != {"verb": "decompose", "target": 0}:
            raise ValueError("source episode does not begin with root decomposition")
        if decompose.get("generation_error"):
            excluded.append(problem_id)
            continue
        if len(rows) < 2:
            raise ValueError("source episode lacks a first-child answer")
        first_answer = rows[1]
        decompose_next = state_from_observation(
            decompose["next_observation"],
            max_depth=max_depth,
        )
        children = child_goals(decompose_next, 0)
        if not children or first_answer.get("action") != {
            "verb": "answer",
            "target": children[0].id,
        }:
            raise ValueError("source episode first child differs from fixed policy")
        if first_answer.get("generation_error"):
            excluded.append(problem_id)
            continue
        prefix_state = state_from_observation(
            first_answer["next_observation"],
            max_depth=max_depth,
        )
        prefix_children = child_goals(prefix_state, 0)
        if not goal_candidates(prefix_state, prefix_children[0].id) or any(
            goal_candidates(prefix_state, child.id)
            for child in prefix_children[1:]
        ):
            raise ValueError("source prefix does not end after the first child")
        if goal_candidates(prefix_state, 0):
            raise ValueError("source prefix already contains a root candidate")
        decompose_call_id = decompose.get("worker_call_id")
        first_call_id = first_answer.get("worker_call_id")
        case = PrefixCase(
            problem_id=problem_id,
            source_episode_index=expected_episode,
            prefix_state=prefix_state,
            source_decompose_call_id=_nonempty(
                decompose_call_id,
                "source decompose call id",
            ),
            source_first_child_call_id=_nonempty(
                first_call_id,
                "source first-child call id",
            ),
        )
        cases.append(case)
        selected_examples.append(example_by_id[problem_id])

    source_run_digest = file_sha256(path)
    prefix_record = {
        "schema_version": PAIRED_PREFIX_VERSION,
        "source_run_sha256": source_run_digest,
        "source_run_contract_sha256": expected_source_run_contract_sha256,
        "cases": [case.source_record(max_depth=max_depth) for case in cases],
        "excluded_problem_ids": excluded,
    }
    return PreparedPrefixes(
        source_run_path=path,
        source_run_sha256=source_run_digest,
        source_run_contract_sha256=expected_source_run_contract_sha256,
        prefix_sha256=canonical_json_sha256(prefix_record),
        cases=tuple(cases),
        examples=tuple(selected_examples),
        excluded_problem_ids=tuple(excluded),
        max_depth=max_depth,
    )


@dataclass(frozen=True, slots=True)
class PairedLensContract:
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
    branch_modes: tuple[str, str] = PAIRED_MODES
    branch_order_version: str = PAIRED_BRANCH_ORDER_VERSION

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
            raise ValueError("paired problem ids must be nonempty and unique")
        if len(self.source_episode_indexes) != len(self.problem_ids) or any(
            type(index) is not int or index < 0
            for index in self.source_episode_indexes
        ):
            raise ValueError("source episode indexes differ from paired cases")
        if self.branch_modes != PAIRED_MODES:
            raise ValueError("paired branch modes differ from the causal contract")
        if self.branch_order_version != PAIRED_BRANCH_ORDER_VERSION:
            raise ValueError("paired branch order version differs")
        if type(self.turn_limit) is not int or self.turn_limit <= 0:
            raise ValueError("paired turn limit must be positive")
        if type(self.max_depth) is not int or self.max_depth <= 0:
            raise ValueError("paired max depth must be positive")
        if not isinstance(self.worker, ModelIdentity):
            raise ValueError("paired worker identity is invalid")
        if type(self.worker_temperature) is not float or self.worker_temperature < 0:
            raise ValueError("paired worker temperature must be nonnegative")
        if type(self.worker_max_tokens) is not int or self.worker_max_tokens <= 0:
            raise ValueError("paired worker max tokens must be positive")
        if type(self.worker_seed) is not int or self.worker_seed < 0:
            raise ValueError("paired worker seed must be nonnegative")

    def header(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema_version": PAIRED_LENS_HEADER_VERSION,
            "record_type": "run_header",
            "run_contract_version": PAIRED_LENS_CONTRACT_VERSION,
            "preregistration_id": self.preregistration_id,
            "source": {
                "schema_version": PAIRED_PREFIX_VERSION,
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
                "reward": REWARD_CONTRACT_VERSION,
                "worker_execution": WORKER_EXECUTION_CONTRACT_VERSION,
                "worker_prompt": WORKER_PROMPT_VERSION,
                "direct_prompt": DIRECT_PROMPT_VERSION,
                "hybrid_lens": HYBRID_LENS_VERSION,
                "child_answer_context": CHILD_CONTEXT_CONTRACT_VERSION,
                "observation": OBSERVATION_SCHEMA_VERSION,
                "submit_receipt": SUBMIT_RECEIPT_VERSION,
                "teacher_prompt": TEACHER_PROMPT_VERSION,
                "teacher_output": TEACHER_OUTPUT_VERSION,
                "teacher_presentation": TEACHER_PRESENTATION_VERSION,
                "branch_order": self.branch_order_version,
                "seed_policy": PAIRED_SEED_POLICY_VERSION,
            },
            "settings": {
                "turn_limit": self.turn_limit,
                "max_depth": self.max_depth,
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


def build_paired_contract(
    *,
    preregistration_id: str,
    prepared: PreparedPrefixes,
    worker: ModelIdentity,
    worker_temperature: float,
    worker_max_tokens: int,
    worker_seed: int,
) -> PairedLensContract:
    return PairedLensContract(
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
class PairedCollectionResult:
    run_path: Path
    diagnostic_path: Path
    case_count: int
    transition_count: int
    solved_by_mode: Mapping[str, int]
    run_contract_sha256: str


def _round_order(case_index: int, round_index: int) -> tuple[str, str]:
    if (case_index + round_index) % 2 == 0:
        return PAIRED_MODES
    return tuple(reversed(PAIRED_MODES))  # type: ignore[return-value]


def _paired_outcome(rewards: Mapping[str, int]) -> str:
    target = rewards[CHILD_CONTEXT_TARGET_ONLY]
    preceding = rewards[CHILD_CONTEXT_PRECEDING_SIBLINGS]
    if target and preceding:
        return "both"
    if target:
        return "target_only"
    if preceding:
        return "preceding_siblings"
    return "neither"


def collect_paired_lens_run(
    *,
    run_path: str | Path,
    diagnostic_path: str | Path,
    contract: PairedLensContract,
    prepared: PreparedPrefixes,
    worker_client: OpenAICompatibleLLMClient,
) -> PairedCollectionResult:
    """Execute both lens branches from every exact sealed prefix."""
    if not isinstance(contract, PairedLensContract):
        raise TypeError("paired collection requires its run contract")
    if not isinstance(prepared, PreparedPrefixes):
        raise TypeError("paired collection requires prepared prefixes")
    if type(worker_client) is not OpenAICompatibleLLMClient:
        raise TypeError("paired worker must be OpenAI-compatible")
    _validate_client_binding(contract.worker, worker_client, role="worker")
    expected = build_paired_contract(
        preregistration_id=contract.preregistration_id,
        prepared=prepared,
        worker=contract.worker,
        worker_temperature=contract.worker_temperature,
        worker_max_tokens=contract.worker_max_tokens,
        worker_seed=contract.worker_seed,
    )
    if expected != contract:
        raise ValueError("paired contract differs from prepared prefixes")

    final_run = Path(run_path)
    final_diagnostic = Path(diagnostic_path)
    partial_run, partial_diagnostic = _validate_output_paths(
        final_run,
        final_diagnostic,
    )
    header = contract.header()
    diagnostic_header = {
        "schema_version": PAIRED_LENS_DIAGNOSTIC_HEADER_VERSION,
        "record_type": "diagnostic_header",
        "preregistration_id": contract.preregistration_id,
        "run_contract_sha256": contract.sha256,
        "attempt_schema_version": DIAGNOSTIC_ATTEMPT_VERSION,
        "roles": list(_ROLE_BY_MODE.values()),
    }
    worker_before = _counter_snapshot(worker_client)
    old_observer = worker_client.attempt_observer
    transition_count = 0
    solved = Counter({mode: 0 for mode in PAIRED_MODES})
    action_counts = {mode: Counter() for mode in PAIRED_MODES}
    generation_errors = Counter({mode: 0 for mode in PAIRED_MODES})
    paired_outcomes: Counter[str] = Counter()

    try:
        with partial_run.open(
            "x", encoding="ascii", newline="\n"
        ) as run_handle, partial_diagnostic.open(
            "x", encoding="ascii", newline="\n"
        ) as diagnostic_handle:
            _write_record(run_handle, header)
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
                prefix_record = {
                    "schema_version": PAIRED_CASE_VERSION,
                    "record_type": "paired_case_prefix",
                    "preregistration_id": contract.preregistration_id,
                    "paired_case_index": case_index,
                    **case.source_record(max_depth=contract.max_depth),
                }
                _write_record(run_handle, prefix_record)
                environments = {
                    mode: WorkspaceEnvironment(
                        problem=example.problem,
                        expected_answer=example.expected_answer,
                        worker=worker_client,
                        turn_limit=contract.turn_limit,
                        max_depth=contract.max_depth,
                        child_context_mode=mode,
                        temperature=contract.worker_temperature,
                        max_tokens=contract.worker_max_tokens,
                        base_seed=(
                            contract.worker_seed
                            + case.source_episode_index * contract.turn_limit
                        ),
                    )
                    for mode in PAIRED_MODES
                }
                policies = {
                    mode: FixedDecompositionPolicy() for mode in PAIRED_MODES
                }
                states = {mode: case.prefix_state for mode in PAIRED_MODES}
                call_ids = {mode: [] for mode in PAIRED_MODES}
                branch_transitions = {mode: 0 for mode in PAIRED_MODES}
                round_index = 0
                while not all(state.terminal for state in states.values()):
                    planned: dict[str, Action] = {}
                    for mode in PAIRED_MODES:
                        state = states[mode]
                        if state.terminal:
                            continue
                        legal = environments[mode].legal_actions(state)
                        planned[mode] = policies[mode].choose(state, legal)
                    order = _round_order(case_index, round_index)
                    for branch_position, mode in enumerate(order):
                        if mode not in planned:
                            continue
                        role = _ROLE_BY_MODE[mode]
                        offset = recorder.count(role)
                        worker_client.attempt_observer = recorder.callback(role)
                        transition = environments[mode].step(
                            states[mode],
                            planned[mode],
                        )
                        new_ids = recorder.since(role, offset)
                        expected_calls = int(transition.worker_request is not None)
                        if len(new_ids) != expected_calls:
                            raise RuntimeError(
                                "paired worker calls differ from transitions"
                            )
                        worker_call_id = new_ids[0] if new_ids else None
                        if worker_call_id is not None:
                            call_ids[mode].append(worker_call_id)
                        record = transition_record(
                            transition,
                            experiment_id=contract.preregistration_id,
                            arm=mode,
                            episode_index=case_index,
                            problem_id=case.problem_id,
                            max_depth=contract.max_depth,
                            worker_call_id=worker_call_id,
                            teacher_call_id=None,
                            teacher_decision=None,
                        )
                        record["paired_case_index"] = case_index
                        record["paired_round_index"] = round_index
                        record["branch_position"] = branch_position
                        _write_record(run_handle, record)
                        states[mode] = transition.next_state
                        branch_transitions[mode] += 1
                        transition_count += 1
                        action_counts[mode][transition.action.verb.value] += 1
                        generation_errors[mode] += int(
                            transition.generation_error
                        )
                    round_index += 1
                    if round_index > contract.turn_limit:
                        raise RuntimeError("paired branch exceeded its horizon")

                rewards = {mode: states[mode].reward for mode in PAIRED_MODES}
                for mode, reward in rewards.items():
                    solved[mode] += reward
                outcome = _paired_outcome(rewards)
                paired_outcomes[outcome] += 1
                case_summary = {
                    "schema_version": PAIRED_CASE_SUMMARY_VERSION,
                    "record_type": "paired_case_summary",
                    "preregistration_id": contract.preregistration_id,
                    "paired_case_index": case_index,
                    "problem_id": case.problem_id,
                    "source_episode_index": case.source_episode_index,
                    "rewards": rewards,
                    "paired_outcome": outcome,
                    "transition_counts": branch_transitions,
                    "worker_call_ids": call_ids,
                }
                _write_record(run_handle, case_summary, flush=True)

            run_summary = {
                "schema_version": PAIRED_RUN_SUMMARY_VERSION,
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
    return PairedCollectionResult(
        run_path=final_run,
        diagnostic_path=final_diagnostic,
        case_count=len(prepared.cases),
        transition_count=transition_count,
        solved_by_mode=dict(solved),
        run_contract_sha256=contract.sha256,
    )


__all__ = [
    "PAIRED_BRANCH_ORDER_VERSION",
    "PAIRED_CASE_SUMMARY_VERSION",
    "PAIRED_CASE_VERSION",
    "PAIRED_LENS_CONTRACT_VERSION",
    "PAIRED_LENS_DIAGNOSTIC_HEADER_VERSION",
    "PAIRED_LENS_HEADER_VERSION",
    "PAIRED_MODES",
    "PAIRED_PREFIX_VERSION",
    "PAIRED_RUN_SUMMARY_VERSION",
    "PAIRED_SEED_POLICY_VERSION",
    "PairedCollectionResult",
    "PairedLensContract",
    "PrefixCase",
    "PreparedPrefixes",
    "build_paired_contract",
    "collect_paired_lens_run",
    "file_sha256",
    "prepare_paired_prefixes",
    "state_from_observation",
]
