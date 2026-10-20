"""Paired full-state and target-local lens experiment over sealed census states."""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Mapping

from ..llm_client import (
    OpenAICompatibleLLMClient,
    SCHEMA_VALIDATION_STRICT_V1,
    append_openai_compatible_parse_retry,
    build_openai_compatible_json_body,
    classify_openai_compatible_json_content,
    project_openai_compatible_response,
)
from .action_census import (
    ACTION_CENSUS_ACTION_VERSION,
    ACTION_CENSUS_REWARD_VERSION,
    ACTION_CENSUS_STATE_VERSION,
    ACTION_TARGETS,
    ANSWER_SCHEMA,
    BUILDER_MODE_ANCHORED_RANGE_LISTS,
    CensusProblem,
    CensusState,
    SemanticNode,
    _ACTION_SUFFIX,
    _ACTION_SYSTEM,
    _ACTION_USER_TEMPLATE,
    _candidate,
    _load_jsonl,
    _logical_call,
    _pairs_object,
    _reject_constant,
    _write_record,
    action_request,
    base_request,
    evaluation_binding_sha256,
    file_sha256,
    implementation_binding,
    load_census_problems,
    problem_source_sha256,
    prompt_bindings,
    request_sha256,
    terminal_reward,
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
from .records import (
    DIAGNOSTIC_ATTEMPT_VERSION,
    canonical_json_bytes,
    canonical_json_sha256,
)
from .target_value import Candidate, WorkerRequest, candidate_from_payload

ACTION_LENS_SOURCE_VERSION = "action-lens-sealed-state-source-v1"
ACTION_LENS_LOCAL_VIEW_VERSION = "action-census-target-local-lens-v1"
ACTION_LENS_FULL_VIEW_VERSION = "action-census-full-state-lens-v1"
ACTION_LENS_DIRECT_VIEW_VERSION = "action-census-problem-only-lens-v1"
ACTION_LENS_ORDER_VERSION = "nine-arm-williams-reversal-v1"
ACTION_LENS_SEED_VERSION = "common-nine-arm-block-seed-v1"
ACTION_LENS_CONTRACT_VERSION = "action-lens-factorial-contract-v1"
ACTION_LENS_HEADER_VERSION = "action-lens-factorial-header-v1"
ACTION_LENS_DIAGNOSTIC_VERSION = "action-lens-factorial-diagnostic-v1"
ACTION_LENS_SOURCE_RECORD_VERSION = "action-lens-source-state-v1"
ACTION_LENS_TRANSITION_VERSION = "action-lens-transition-v1"
ACTION_LENS_CASE_VERSION = "action-lens-case-summary-v1"
ACTION_LENS_SUMMARY_VERSION = "action-lens-run-summary-v1"
ACTION_LENS_ANALYSIS_VERSION = "action-lens-analysis-v1"
ACTION_LENS_PILOT_ANALYSIS_VERSION = "action-lens-pilot-analysis-v2"
PILOT_BOOTSTRAP_SAMPLES = 10_000
PILOT_BOOTSTRAP_SEEDS = (457_001, 457_002, 457_003)
PILOT_PERMUTATION_SAMPLES = 100_000
PILOT_PERMUTATION_SEED = 457_004

VIEW_PROBLEM_ONLY = "problem_only_v1"
VIEW_FULL_STATE = "full_state_v1"
VIEW_TARGET_LOCAL = "target_local_v1"
VIEW_MODES = (VIEW_PROBLEM_ONLY, VIEW_FULL_STATE, VIEW_TARGET_LOCAL)

DIRECT_ARM = "problem_only"
FULL_ARMS = tuple(f"full_{target}" for target in ACTION_TARGETS)
LOCAL_ARMS = tuple(f"local_{target}" for target in ACTION_TARGETS)
FACTORIAL_ARMS = (DIRECT_ARM, *FULL_ARMS, *LOCAL_ARMS)
CHILD_TARGETS = ACTION_TARGETS[1:]
EXPERIMENT_STAGES = frozenset({"integration_smoke", "pilot"})

_TRANSITION_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "preregistration_id",
        "case_index",
        "source_problem_index",
        "problem_id",
        "source_state_sha256",
        "repetition",
        "split",
        "arm",
        "view_mode",
        "target",
        "arm_position",
        "seed",
        "request_sha256",
        "request",
        "worker_call_id",
        "candidate",
        "generation_error",
        "error_type",
        "reward",
        "behavior_cloning_eligible",
    }
)
_CASE_SUMMARY_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "preregistration_id",
        "case_index",
        "source_problem_index",
        "problem_id",
        "source_state_sha256",
        "arm_orders",
        "rewards",
        "usable",
        "worker_call_ids",
        "behavior_cloning_eligible",
    }
)
_RUN_SUMMARY_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "preregistration_id",
        "run_contract_sha256",
        "case_count",
        "logical_call_count",
        "transition_count",
        "call_failure_counts",
        "unusable_output_counts",
        "usage",
        "analysis",
        "behavior_cloning_eligible",
    }
)
_DIAGNOSTIC_RECORD_FIELDS = frozenset(
    {"schema_version", "record_type", "sequence", "role", "attempt"}
)


def arm_view_target(arm: str) -> tuple[str, str | None]:
    if arm == DIRECT_ARM:
        return VIEW_PROBLEM_ONLY, None
    for prefix, mode in (("full_", VIEW_FULL_STATE), ("local_", VIEW_TARGET_LOCAL)):
        if arm.startswith(prefix):
            target = arm.removeprefix(prefix)
            if target in ACTION_TARGETS:
                return mode, target
    raise ValueError("action-lens arm is invalid")


def _sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase sha256")


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value.strip()


def _candidate_from_record(record: object) -> Candidate:
    if not isinstance(record, Mapping):
        raise ValueError("source root Candidate is invalid")
    try:
        return Candidate(
            reasoning=record.get("reasoning"),  # type: ignore[arg-type]
            answer=record.get("answer"),  # type: ignore[arg-type]
            usable=record.get("usable"),  # type: ignore[arg-type]
            error_type=record.get("error_type"),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("source root Candidate is invalid") from exc


def state_from_observation(observation: Mapping[str, object]) -> CensusState:
    nodes = observation.get("nodes")
    if type(nodes) is not list or len(nodes) != 4:
        raise ValueError("source observation requires root plus three nodes")
    root = nodes[0]
    if not isinstance(root, Mapping):
        raise ValueError("source root node is invalid")
    draft = _candidate_from_record(root.get("candidate"))
    semantic_nodes: list[SemanticNode] = []
    for expected_id, raw in zip(ACTION_TARGETS[1:], nodes[1:], strict=True):
        if not isinstance(raw, Mapping) or raw.get("id") != expected_id:
            raise ValueError("source semantic node order differs")
        title = raw.get("title")
        content = raw.get("content")
        if not isinstance(title, str) or not isinstance(content, str):
            raise ValueError("source semantic node text is invalid")
        semantic_nodes.append(SemanticNode(expected_id, title, content))
    problem_id = observation.get("problem_id")
    problem = observation.get("problem")
    if not isinstance(problem_id, str) or not isinstance(problem, str):
        raise ValueError("source observation problem is invalid")
    state = CensusState(
        problem_id=problem_id,
        problem=problem,
        draft=draft,
        nodes=(semantic_nodes[0], semantic_nodes[1], semantic_nodes[2]),
    )
    if state.observation() != dict(observation):
        raise ValueError("source census State does not replay exactly")
    return state


@dataclass(frozen=True, slots=True)
class ActionLensSourceCase:
    source_problem_index: int
    problem: CensusProblem
    state: CensusState
    source_state_sha256: str

    def __post_init__(self) -> None:
        if type(self.source_problem_index) is not int or self.source_problem_index < 0:
            raise ValueError("source problem index must be nonnegative")
        if self.problem.problem_id != self.state.problem_id:
            raise ValueError("source problem and State ids differ")
        if self.problem.problem != self.state.problem:
            raise ValueError("source problem and State text differ")
        _sha256(self.source_state_sha256, "source state sha256")
        if self.state.sha256 != self.source_state_sha256:
            raise ValueError("source State hash differs")

    def source_record(self) -> dict[str, object]:
        return {
            "source_problem_index": self.source_problem_index,
            "problem_id": self.problem.problem_id,
            "state_sha256": self.source_state_sha256,
            "observation": self.state.observation(),
        }


@dataclass(frozen=True, slots=True)
class PreparedActionLensSource:
    source_run_sha256: str
    source_diagnostic_sha256: str
    source_run_contract_sha256: str
    dataset_sha256: str
    selection_sha256: str
    cases: tuple[ActionLensSourceCase, ...]

    def __post_init__(self) -> None:
        for name in (
            "source_run_sha256",
            "source_diagnostic_sha256",
            "source_run_contract_sha256",
            "dataset_sha256",
            "selection_sha256",
        ):
            _sha256(getattr(self, name), name.replace("_", " "))
        if not self.cases:
            raise ValueError("action-lens source requires cases")
        indices = tuple(case.source_problem_index for case in self.cases)
        ids = tuple(case.problem.problem_id for case in self.cases)
        if len(indices) != len(set(indices)) or len(ids) != len(set(ids)):
            raise ValueError("action-lens source cases must be unique")


def _selection_record(
    *,
    source_run_sha256: str,
    source_diagnostic_sha256: str,
    source_run_contract_sha256: str,
    cases: tuple[ActionLensSourceCase, ...],
) -> dict[str, object]:
    return {
        "schema_version": ACTION_LENS_SOURCE_VERSION,
        "source_run_sha256": source_run_sha256,
        "source_diagnostic_sha256": source_diagnostic_sha256,
        "source_run_contract_sha256": source_run_contract_sha256,
        "cases": [case.source_record() for case in cases],
    }


def _load_source_projection(
    path: str | Path,
    name: str,
    *,
    retain_states: bool,
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    with Path(path).open("r", encoding="ascii", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            retain = line_number == 1 or (
                retain_states and '"record_type":"census_state"' in line
            )
            if not retain:
                continue
            if line.endswith("\r\n"):
                raise ValueError(f"{name} line {line_number} uses CRLF")
            if not line.endswith("\n"):
                raise ValueError(f"{name} line {line_number} lacks a newline")
            value = json.loads(
                line,
                object_pairs_hook=_pairs_object,
                parse_constant=_reject_constant,
            )
            if type(value) is not dict:
                raise ValueError(f"{name} line {line_number} is not an object")
            if line != canonical_json_bytes(value).decode("ascii") + "\n":
                raise ValueError(f"{name} line {line_number} is not canonical")
            records.append(value)
    if not records:
        raise ValueError(f"{name} is empty")
    return tuple(records)


def prepare_action_lens_source(
    *,
    source_run_path: str | Path,
    source_diagnostic_path: str | Path,
    expected_source_run_sha256: str,
    expected_source_diagnostic_sha256: str,
    expected_source_run_contract_sha256: str,
    dataset_path: str | Path,
    source_case_indices: tuple[int, ...],
) -> PreparedActionLensSource:
    for value, name in (
        (expected_source_run_sha256, "expected source run sha256"),
        (expected_source_diagnostic_sha256, "expected source diagnostic sha256"),
        (expected_source_run_contract_sha256, "expected source contract sha256"),
    ):
        _sha256(value, name)
    if (
        not source_case_indices
        or len(source_case_indices) != len(set(source_case_indices))
        or any(type(index) is not int or index < 0 for index in source_case_indices)
    ):
        raise ValueError("source case indices must be nonempty, unique integers")
    if file_sha256(source_run_path) != expected_source_run_sha256:
        raise ValueError("source run hash differs")
    if file_sha256(source_diagnostic_path) != expected_source_diagnostic_sha256:
        raise ValueError("source diagnostic hash differs")
    run_records = _load_source_projection(
        source_run_path,
        "action-lens source run",
        retain_states=True,
    )
    diagnostic_records = _load_source_projection(
        source_diagnostic_path,
        "action-lens source diagnostics",
        retain_states=False,
    )
    header = run_records[0]
    diagnostic_header = diagnostic_records[0]
    settings = header.get("settings")
    problem_source = header.get("problem_source")
    contracts = header.get("contracts")
    if (
        header.get("record_type") != "run_header"
        or header.get("run_contract_sha256") != expected_source_run_contract_sha256
        or not isinstance(settings, Mapping)
        or settings.get("builder_mode") != BUILDER_MODE_ANCHORED_RANGE_LISTS
        or not isinstance(problem_source, Mapping)
        or not isinstance(contracts, Mapping)
        or contracts.get("state") != ACTION_CENSUS_STATE_VERSION
        or contracts.get("action") != ACTION_CENSUS_ACTION_VERSION
        or contracts.get("reward") != ACTION_CENSUS_REWARD_VERSION
    ):
        raise ValueError("source action-census contract is incompatible")
    if (
        diagnostic_header.get("record_type") != "diagnostic_header"
        or diagnostic_header.get("run_contract_sha256")
        != expected_source_run_contract_sha256
    ):
        raise ValueError("source diagnostic header differs")
    source_ids = tuple(problem_source.get("problem_ids", ()))
    state_records = tuple(
        record for record in run_records if record.get("record_type") == "census_state"
    )
    if len(state_records) != len(source_ids):
        raise ValueError("source run does not contain every census State")
    if tuple(record.get("problem_index") for record in state_records) != tuple(
        range(len(source_ids))
    ):
        raise ValueError("source census State ordering differs")
    states_by_index: dict[int, Mapping[str, object]] = {}
    for record in state_records:
        source_index = record.get("problem_index")
        if (
            type(source_index) is not int
            or source_index < 0
            or source_index >= len(source_ids)
            or source_index in states_by_index
            or record.get("problem_id") != source_ids[source_index]
            or record.get("builder_usable") is not True
            or record.get("node_text_complete") is not True
            or record.get("node_text_distinct") is not True
        ):
            raise ValueError("source census State record is invalid")
        states_by_index[source_index] = record
    dataset_sha256 = file_sha256(dataset_path)
    all_problems = load_census_problems(dataset_path)
    problem_by_id = {problem.problem_id: problem for problem in all_problems}
    if len(problem_by_id) != len(all_problems) or any(
        problem_id not in problem_by_id for problem_id in source_ids
    ):
        raise ValueError("source census problem ids are absent or duplicated")
    source_problems = tuple(problem_by_id[problem_id] for problem_id in source_ids)
    if (
        problem_source.get("dataset_sha256") != dataset_sha256
        or problem_source.get("projection_sha256")
        != problem_source_sha256(source_problems)
        or problem_source.get("evaluation_binding_sha256")
        != evaluation_binding_sha256(source_problems)
    ):
        raise ValueError("source census dataset or hidden binding differs")
    selected: list[ActionLensSourceCase] = []
    for source_index in source_case_indices:
        if source_index not in states_by_index:
            raise ValueError("source case index is outside the sealed States")
        record = states_by_index[source_index]
        observation = record.get("observation")
        if not isinstance(observation, Mapping):
            raise ValueError("source census observation is invalid")
        state = state_from_observation(observation)
        state_sha256 = record.get("state_sha256")
        if not isinstance(state_sha256, str) or state.sha256 != state_sha256:
            raise ValueError("source census State hash differs")
        problem = problem_by_id.get(state.problem_id)
        if problem is None:
            raise ValueError("source census problem is absent from the dataset")
        selected.append(
            ActionLensSourceCase(
                source_problem_index=source_index,
                problem=problem,
                state=state,
                source_state_sha256=state_sha256,
            )
        )
    cases = tuple(selected)
    selection_sha256 = canonical_json_sha256(
        _selection_record(
            source_run_sha256=expected_source_run_sha256,
            source_diagnostic_sha256=expected_source_diagnostic_sha256,
            source_run_contract_sha256=expected_source_run_contract_sha256,
            cases=cases,
        )
    )
    return PreparedActionLensSource(
        source_run_sha256=expected_source_run_sha256,
        source_diagnostic_sha256=expected_source_diagnostic_sha256,
        source_run_contract_sha256=expected_source_run_contract_sha256,
        dataset_sha256=dataset_sha256,
        selection_sha256=selection_sha256,
        cases=cases,
    )


def target_local_observation(state: CensusState, target: str) -> dict[str, object]:
    if target not in ACTION_TARGETS:
        raise ValueError("target-local lens target is invalid")
    if target == "root":
        selected_node: dict[str, object] = {
            "id": "root",
            "kind": "root_candidate",
            "title": "Complete draft solution",
            "candidate": state.draft.to_record(),
        }
    else:
        node = state.nodes[ACTION_TARGETS.index(target) - 1]
        selected_node = {
            "id": node.node_id,
            "kind": "semantic_component",
            "title": node.title or "[empty generated title]",
            "content": node.content or "[empty generated content]",
        }
    return {
        "schema_version": ACTION_LENS_LOCAL_VIEW_VERSION,
        "problem_id": state.problem_id,
        "problem": state.problem,
        "selected_action": {"verb": "revise", "target": target},
        "selected_node": selected_node,
    }


def target_local_action_request(state: CensusState, target: str) -> WorkerRequest:
    observation = canonical_json_bytes(target_local_observation(state, target)).decode(
        "ascii"
    )
    return WorkerRequest(
        system=_ACTION_SYSTEM,
        user=_ACTION_USER_TEMPLATE.format(
            observation=observation,
            target=target,
            instruction=_ACTION_SUFFIX,
        ),
        schema=ANSWER_SCHEMA,
    )


def request_for_arm(case: ActionLensSourceCase, arm: str) -> WorkerRequest:
    view_mode, target = arm_view_target(arm)
    if view_mode == VIEW_PROBLEM_ONLY:
        return base_request(case.problem)
    if target is None:
        raise AssertionError("revision arm lost its target")
    if view_mode == VIEW_FULL_STATE:
        return action_request(case.state, target)
    return target_local_action_request(case.state, target)


def prompt_bindings_factorial() -> dict[str, str]:
    census_bindings = prompt_bindings(BUILDER_MODE_ANCHORED_RANGE_LISTS)
    return {
        "problem_only": census_bindings["base"],
        "full_state": census_bindings["action_worker"],
        "target_local": canonical_json_sha256(
            {
                "system": _ACTION_SYSTEM,
                "user_template": _ACTION_USER_TEMPLATE,
                "instruction": _ACTION_SUFFIX,
                "schema": ANSWER_SCHEMA,
                "local_view_version": ACTION_LENS_LOCAL_VIEW_VERSION,
            }
        ),
    }


def _williams_rows() -> tuple[tuple[str, ...], ...]:
    count = len(FACTORIAL_ARMS)
    first_indices = tuple(
        index // 2 if index % 2 == 0 else count - 1 - index // 2
        for index in range(count)
    )
    base_rows = tuple(
        tuple(FACTORIAL_ARMS[(value + shift) % count] for value in first_indices)
        for shift in range(count)
    )
    return (*base_rows, *(tuple(reversed(row)) for row in base_rows))


WILLIAMS_ROWS = _williams_rows()


def arm_order(source_problem_index: int, repetition: int) -> tuple[str, ...]:
    if type(source_problem_index) is not int or source_problem_index < 0:
        raise ValueError("source problem index must be nonnegative")
    if repetition not in range(4):
        raise ValueError("action-lens repetition is invalid")
    return WILLIAMS_ROWS[(source_problem_index * 4 + repetition) % len(WILLIAMS_ROWS)]


def order_schedule_record(
    cases: tuple[ActionLensSourceCase, ...],
) -> dict[str, object]:
    return {
        "schema_version": ACTION_LENS_ORDER_VERSION,
        "blocks": [
            {
                "source_problem_index": case.source_problem_index,
                "repetition": repetition,
                "order": list(arm_order(case.source_problem_index, repetition)),
            }
            for case in cases
            for repetition in range(4)
        ],
    }


@dataclass(frozen=True, slots=True)
class ActionLensContract:
    preregistration_id: str
    stage: str
    source_run_sha256: str
    source_diagnostic_sha256: str
    source_run_contract_sha256: str
    dataset_sha256: str
    selection_sha256: str
    problem_source_sha256: str
    evaluation_binding_sha256: str
    source_case_indices: tuple[int, ...]
    problem_ids: tuple[str, ...]
    order_schedule_sha256: str
    worker: ModelIdentity
    worker_temperature: float
    worker_max_tokens: int
    worker_seed: int
    repetitions: int = 4
    discovery_repetitions: tuple[int, ...] = (0, 1)
    evaluation_repetitions: tuple[int, ...] = (2, 3)

    def __post_init__(self) -> None:
        _nonempty(self.preregistration_id, "preregistration id")
        if self.stage not in EXPERIMENT_STAGES:
            raise ValueError("action-lens stage is invalid")
        for name in (
            "source_run_sha256",
            "source_diagnostic_sha256",
            "source_run_contract_sha256",
            "dataset_sha256",
            "selection_sha256",
            "problem_source_sha256",
            "evaluation_binding_sha256",
            "order_schedule_sha256",
        ):
            _sha256(getattr(self, name), name.replace("_", " "))
        if (
            not self.source_case_indices
            or len(self.source_case_indices) != len(set(self.source_case_indices))
            or len(self.problem_ids) != len(self.source_case_indices)
            or len(self.problem_ids) != len(set(self.problem_ids))
        ):
            raise ValueError("action-lens case identities are invalid")
        if not isinstance(self.worker, ModelIdentity):
            raise ValueError("action-lens worker identity is invalid")
        if type(self.worker_temperature) is not float or self.worker_temperature < 0:
            raise ValueError("action-lens temperature must be nonnegative")
        if type(self.worker_max_tokens) is not int or self.worker_max_tokens <= 0:
            raise ValueError("action-lens max tokens must be positive")
        if type(self.worker_seed) is not int or self.worker_seed < 0:
            raise ValueError("action-lens seed must be nonnegative")
        if (
            self.repetitions != 4
            or self.discovery_repetitions != (0, 1)
            or self.evaluation_repetitions != (2, 3)
        ):
            raise ValueError("action-lens experiment requires the registered 2+2 split")

    def header(self) -> dict[str, object]:
        analysis_version = (
            ACTION_LENS_PILOT_ANALYSIS_VERSION
            if self.stage == "pilot"
            else ACTION_LENS_ANALYSIS_VERSION
        )
        body: dict[str, object] = {
            "schema_version": ACTION_LENS_HEADER_VERSION,
            "record_type": "run_header",
            "run_contract_version": ACTION_LENS_CONTRACT_VERSION,
            "preregistration_id": self.preregistration_id,
            "stage": self.stage,
            "source": {
                "schema_version": ACTION_LENS_SOURCE_VERSION,
                "source_run_sha256": self.source_run_sha256,
                "source_diagnostic_sha256": self.source_diagnostic_sha256,
                "source_run_contract_sha256": self.source_run_contract_sha256,
                "dataset_sha256": self.dataset_sha256,
                "selection_sha256": self.selection_sha256,
                "problem_source_sha256": self.problem_source_sha256,
                "evaluation_binding_sha256": self.evaluation_binding_sha256,
                "source_case_indices": list(self.source_case_indices),
                "problem_ids": list(self.problem_ids),
            },
            "contracts": {
                "state": ACTION_CENSUS_STATE_VERSION,
                "action": ACTION_CENSUS_ACTION_VERSION,
                "reward": ACTION_CENSUS_REWARD_VERSION,
                "schema_validation": SCHEMA_VALIDATION_STRICT_V1,
                "problem_only_view": ACTION_LENS_DIRECT_VIEW_VERSION,
                "full_state_view": ACTION_LENS_FULL_VIEW_VERSION,
                "target_local_view": ACTION_LENS_LOCAL_VIEW_VERSION,
                "arm_order": ACTION_LENS_ORDER_VERSION,
                "seed_policy": ACTION_LENS_SEED_VERSION,
                "analysis": analysis_version,
            },
            "prompt_bindings": prompt_bindings_factorial(),
            "implementation_binding": implementation_binding(),
            "models": {"worker": self.worker.to_record(), "teacher": None},
            "settings": {
                "case_count": len(self.problem_ids),
                "arms": list(FACTORIAL_ARMS),
                "worker_temperature": self.worker_temperature,
                "worker_max_tokens": self.worker_max_tokens,
                "worker_seed": self.worker_seed,
                "repetitions": self.repetitions,
                "discovery_repetitions": list(self.discovery_repetitions),
                "evaluation_repetitions": list(self.evaluation_repetitions),
                "order_schedule_sha256": self.order_schedule_sha256,
                "generation_failure_reward": 0,
            },
        }
        if self.stage == "pilot":
            settings = body["settings"]
            assert isinstance(settings, dict)
            settings["pilot_analysis"] = {
                "independent_unit": "source_state",
                "child_targets": list(CHILD_TARGETS),
                "global_target_tie_order": list(CHILD_TARGETS),
                "state_target_tie_fallback": "global_fixed_child",
                "bootstrap_samples": PILOT_BOOTSTRAP_SAMPLES,
                "bootstrap_seeds": list(PILOT_BOOTSTRAP_SEEDS),
                "permutation_samples": PILOT_PERMUTATION_SAMPLES,
                "permutation_seed": PILOT_PERMUTATION_SEED,
            }
        body["run_contract_sha256"] = canonical_json_sha256(body)
        return body

    @property
    def sha256(self) -> str:
        return str(self.header()["run_contract_sha256"])


def build_action_lens_contract(
    *,
    preregistration_id: str,
    stage: str,
    prepared: PreparedActionLensSource,
    worker: ModelIdentity,
    worker_temperature: float,
    worker_max_tokens: int,
    worker_seed: int,
) -> ActionLensContract:
    problems = tuple(case.problem for case in prepared.cases)
    return ActionLensContract(
        preregistration_id=preregistration_id,
        stage=stage,
        source_run_sha256=prepared.source_run_sha256,
        source_diagnostic_sha256=prepared.source_diagnostic_sha256,
        source_run_contract_sha256=prepared.source_run_contract_sha256,
        dataset_sha256=prepared.dataset_sha256,
        selection_sha256=prepared.selection_sha256,
        problem_source_sha256=problem_source_sha256(problems),
        evaluation_binding_sha256=evaluation_binding_sha256(problems),
        source_case_indices=tuple(case.source_problem_index for case in prepared.cases),
        problem_ids=tuple(case.problem.problem_id for case in prepared.cases),
        order_schedule_sha256=canonical_json_sha256(
            order_schedule_record(prepared.cases)
        ),
        worker=worker,
        worker_temperature=worker_temperature,
        worker_max_tokens=worker_max_tokens,
        worker_seed=worker_seed,
    )


def assert_request_isolation(case: ActionLensSourceCase) -> None:
    full_state_text = canonical_json_bytes(case.state.observation()).decode("ascii")
    for target in ACTION_TARGETS:
        full = request_for_arm(case, f"full_{target}")
        expected_full = action_request(case.state, target)
        if full != expected_full or full_state_text not in full.user:
            raise RuntimeError("full-state lens request differs from E46")
        local = request_for_arm(case, f"local_{target}")
        if local.system != full.system or local.schema != full.schema:
            raise RuntimeError("local lens changed a non-view request contract")
        expected_view = target_local_observation(case.state, target)
        expected_local = target_local_action_request(case.state, target)
        if local != expected_local:
            raise RuntimeError("target-local request differs from its exact projection")
        prefix = "Workspace state:\n"
        marker = f"\n\nSelected target id: {target}\n\n{_ACTION_SUFFIX}"
        if not local.user.startswith(prefix) or not local.user.endswith(marker):
            raise RuntimeError("target-local request framing differs")
        encoded_view = local.user[len(prefix) : -len(marker)]
        try:
            observed_view = json.loads(
                encoded_view,
                object_pairs_hook=_pairs_object,
                parse_constant=_reject_constant,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("target-local projection is not canonical JSON") from exc
        if observed_view != expected_view or encoded_view != canonical_json_bytes(
            expected_view
        ).decode("ascii"):
            raise RuntimeError("target-local projection exposed extra state")
        selected_node = expected_view["selected_node"]
        if not isinstance(selected_node, Mapping):
            raise AssertionError("target-local selected node is invalid")
        expected_node_fields = (
            {"id", "kind", "title", "candidate"}
            if target == "root"
            else {"id", "kind", "title", "content"}
        )
        if set(selected_node) != expected_node_fields:
            raise RuntimeError("target-local projection has unexpected node fields")
    direct = request_for_arm(case, DIRECT_ARM)
    if direct != base_request(case.problem):
        raise RuntimeError("problem-only request differs from the direct baseline")
    if "Selected target id:" in direct.user or "Workspace state:" in direct.user:
        raise RuntimeError("problem-only request exposed state or target")


def _bootstrap_interval(
    values: tuple[float, ...],
    *,
    seed: int = 457_001,
    samples: int = 10_000,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires case values")
    rng = random.Random(seed)
    count = len(values)
    estimates = sorted(
        mean(values[rng.randrange(count)] for _ in range(count)) for _ in range(samples)
    )
    return estimates[int(samples * 0.05)], estimates[int(samples * 0.95)]


def _paired_cells(left: list[int], right: list[int]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for left_value, right_value in zip(left, right, strict=True):
        if left_value and right_value:
            counts["both"] += 1
        elif left_value:
            counts["left_only"] += 1
        elif right_value:
            counts["right_only"] += 1
        else:
            counts["neither"] += 1
    return {key: counts[key] for key in ("both", "left_only", "right_only", "neither")}


def _validate_case_results(
    case_results: tuple[Mapping[str, tuple[int, ...]], ...],
    contract: ActionLensContract,
) -> None:
    if len(case_results) != len(contract.problem_ids):
        raise ValueError("action-lens analysis case count differs")
    for result in case_results:
        if tuple(result) != FACTORIAL_ARMS or any(
            len(values) != contract.repetitions for values in result.values()
        ):
            raise ValueError("action-lens result grid is incomplete")
        if any(
            type(value) is not int or value not in {0, 1}
            for values in result.values()
            for value in values
        ):
            raise ValueError("action-lens rewards must be binary integers")


def _analyze_action_lens_integration(
    case_results: tuple[Mapping[str, tuple[int, ...]], ...],
    contract: ActionLensContract,
) -> dict[str, object]:
    _validate_case_results(case_results, contract)

    def rewards(arm: str, repetitions: tuple[int, ...]) -> list[int]:
        return [
            result[arm][repetition]
            for result in case_results
            for repetition in repetitions
        ]

    accuracy_all = {
        arm: mean(rewards(arm, tuple(range(contract.repetitions))))
        for arm in FACTORIAL_ARMS
    }
    accuracy_discovery = {
        arm: mean(rewards(arm, contract.discovery_repetitions))
        for arm in FACTORIAL_ARMS
    }
    accuracy_evaluation = {
        arm: mean(rewards(arm, contract.evaluation_repetitions))
        for arm in FACTORIAL_ARMS
    }

    child_local: list[int] = []
    child_full: list[int] = []
    child_local_discovery: list[int] = []
    child_full_discovery: list[int] = []
    local_win_targets: Counter[str] = Counter()
    case_child_differences: list[float] = []
    for result in case_results:
        local_values: list[int] = []
        full_values: list[int] = []
        for target in CHILD_TARGETS:
            for repetition in contract.evaluation_repetitions:
                local_value = result[f"local_{target}"][repetition]
                full_value = result[f"full_{target}"][repetition]
                local_values.append(local_value)
                full_values.append(full_value)
                child_local.append(local_value)
                child_full.append(full_value)
                if local_value > full_value:
                    local_win_targets[target] += 1
            for repetition in contract.discovery_repetitions:
                child_local_discovery.append(result[f"local_{target}"][repetition])
                child_full_discovery.append(result[f"full_{target}"][repetition])
        case_child_differences.append(mean(local_values) - mean(full_values))
    child_cells = _paired_cells(child_local, child_full)
    child_delta = mean(case_child_differences)

    target_order = ACTION_TARGETS

    def select_global(view: str) -> str:
        scores = {
            target: accuracy_discovery[f"{view}_{target}"] for target in target_order
        }
        best = max(scores.values())
        return next(target for target in target_order if scores[target] == best)

    selected_local_target = select_global("local")
    selected_full_target = select_global("full")
    selected_local = rewards(
        f"local_{selected_local_target}", contract.evaluation_repetitions
    )
    selected_full = rewards(
        f"full_{selected_full_target}", contract.evaluation_repetitions
    )
    direct_evaluation = rewards(DIRECT_ARM, contract.evaluation_repetitions)
    selected_local_direct_cells = _paired_cells(selected_local, direct_evaluation)
    selected_local_full_cells = _paired_cells(selected_local, selected_full)
    direct_case_differences = [
        mean(
            result[f"local_{selected_local_target}"][repetition]
            - result[DIRECT_ARM][repetition]
            for repetition in contract.evaluation_repetitions
        )
        for result in case_results
    ]

    stable_winners: Counter[str] = Counter()
    state_selected_values: list[float] = []
    for result in case_results:
        discovery_scores = {
            target: mean(
                result[f"local_{target}"][repetition]
                for repetition in contract.discovery_repetitions
            )
            for target in target_order
        }
        evaluation_scores = {
            target: mean(
                result[f"local_{target}"][repetition]
                for repetition in contract.evaluation_repetitions
            )
            for target in target_order
        }
        discovery_best = max(discovery_scores.values())
        discovery_unique = [
            target
            for target, value in discovery_scores.items()
            if value == discovery_best
        ]
        evaluation_best = max(evaluation_scores.values())
        evaluation_unique = [
            target
            for target, value in evaluation_scores.items()
            if value == evaluation_best
        ]
        if len(discovery_unique) == 1 and discovery_unique == evaluation_unique:
            stable_winners[discovery_unique[0]] += 1
        selected = next(
            target
            for target in target_order
            if discovery_scores[target] == discovery_best
        )
        state_selected_values.append(evaluation_scores[selected])
    best_fixed_local_evaluation = max(
        accuracy_evaluation[f"local_{target}"] for target in target_order
    )
    state_selected_evaluation = mean(state_selected_values)

    primary_lens_gate = {
        "net_local_wins": (child_cells["left_only"] - child_cells["right_only"]),
        "threshold": 6,
        "positive_case_count": sum(value > 0 for value in case_child_differences),
        "local_win_target_count": sum(
            value > 0 for value in local_win_targets.values()
        ),
        "discovery_delta": mean(child_local_discovery) - mean(child_full_discovery),
    }
    primary_lens_gate["passed"] = bool(
        primary_lens_gate["net_local_wins"] >= 6
        and primary_lens_gate["positive_case_count"] >= 3
        and primary_lens_gate["local_win_target_count"] >= 2
        and primary_lens_gate["discovery_delta"] > 0
    )
    direct_gate = {
        "net_local_wins": selected_local_direct_cells["left_only"]
        - selected_local_direct_cells["right_only"],
        "threshold": 3,
        "positive_case_count": sum(value > 0 for value in direct_case_differences),
    }
    direct_gate["passed"] = bool(
        direct_gate["net_local_wins"] >= 3 and direct_gate["positive_case_count"] >= 3
    )
    stable_target_count = sum(value > 0 for value in stable_winners.values())
    stable_node_present = any(stable_winners[target] for target in CHILD_TARGETS)
    target_gate = {
        "stable_winner_state_count": sum(stable_winners.values()),
        "stable_winner_target_count": stable_target_count,
        "stable_node_present": stable_node_present,
        "state_selected_minus_best_fixed": (
            state_selected_evaluation - best_fixed_local_evaluation
        ),
    }
    target_gate["passed"] = bool(
        target_gate["stable_winner_state_count"] >= 2
        and stable_target_count >= 2
        and stable_node_present
        and target_gate["state_selected_minus_best_fixed"] >= 1 / 12
    )

    return {
        "schema_version": ACTION_LENS_ANALYSIS_VERSION,
        "case_count": len(case_results),
        "accuracy_all": accuracy_all,
        "accuracy_discovery": accuracy_discovery,
        "accuracy_evaluation": accuracy_evaluation,
        "primary_child_lens": {
            "evaluation_delta": child_delta,
            "bootstrap_90_ci": list(_bootstrap_interval(tuple(case_child_differences))),
            "paired_cells": child_cells,
            "case_differences": case_child_differences,
            "local_win_counts_by_target": dict(local_win_targets),
            "gate": primary_lens_gate,
        },
        "direct_reference": {
            "selected_local_target": selected_local_target,
            "selected_full_target": selected_full_target,
            "selected_local_vs_direct": selected_local_direct_cells,
            "selected_local_vs_selected_full": selected_local_full_cells,
            "case_differences": direct_case_differences,
            "gate": direct_gate,
        },
        "target_value": {
            "stable_winner_counts": dict(stable_winners),
            "state_selected_evaluation": state_selected_evaluation,
            "best_fixed_local_evaluation": best_fixed_local_evaluation,
            "gate": target_gate,
        },
        "all_required_gates_passed": bool(
            primary_lens_gate["passed"]
            and direct_gate["passed"]
            and target_gate["passed"]
        ),
        "behavior_cloning_eligible": False,
    }


def _bootstrap_intervals(
    values: tuple[float, ...],
    *,
    seed: int,
) -> dict[str, list[float]]:
    if not values:
        raise ValueError("pilot bootstrap requires case values")
    rng = random.Random(seed)
    count = len(values)
    estimates = sorted(
        mean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(PILOT_BOOTSTRAP_SAMPLES)
    )

    def interval(lower: float, upper: float) -> list[float]:
        return [
            estimates[int(PILOT_BOOTSTRAP_SAMPLES * lower)],
            estimates[
                min(PILOT_BOOTSTRAP_SAMPLES - 1, int(PILOT_BOOTSTRAP_SAMPLES * upper))
            ],
        ]

    return {"90": interval(0.05, 0.95), "95": interval(0.025, 0.975)}


def _case_sign_flip_probability(values: tuple[float, ...]) -> float:
    if not values:
        raise ValueError("pilot sign flip requires case values")
    observed = sum(values)
    exceedances = 0
    assignment_count = 1 << len(values)
    for mask in range(assignment_count):
        statistic = sum(
            value if mask & (1 << index) else -value
            for index, value in enumerate(values)
        )
        exceedances += int(statistic >= observed - 1e-12)
    return exceedances / assignment_count


def _family_sensitivity(
    values: tuple[float, ...],
    families: tuple[str, ...],
) -> dict[str, object]:
    if len(values) != len(families) or any(
        not isinstance(family, str) or not family for family in families
    ):
        raise ValueError("pilot families differ from case effects")
    family_order = tuple(dict.fromkeys(families))
    effects = {
        family: mean(
            value
            for value, observed_family in zip(values, families, strict=True)
            if observed_family == family
        )
        for family in family_order
    }
    leave_one_family_out = {
        family: mean(
            value
            for value, observed_family in zip(values, families, strict=True)
            if observed_family != family
        )
        for family in family_order
        if any(observed_family != family for observed_family in families)
    }
    return {
        "family_effects": effects,
        "leave_one_family_out": leave_one_family_out,
    }


def _pilot_comparison(
    left_by_case: tuple[tuple[int, ...], ...],
    right_by_case: tuple[tuple[int, ...], ...],
    *,
    families: tuple[str, ...],
    bootstrap_seed: int,
) -> dict[str, object]:
    if (
        not left_by_case
        or len(left_by_case) != len(right_by_case)
        or len(left_by_case) != len(families)
        or any(
            len(left) != len(right) or not left
            for left, right in zip(left_by_case, right_by_case, strict=True)
        )
    ):
        raise ValueError("pilot comparison grid differs")
    left = [value for values in left_by_case for value in values]
    right = [value for values in right_by_case for value in values]
    case_differences = tuple(
        mean(left_values) - mean(right_values)
        for left_values, right_values in zip(left_by_case, right_by_case, strict=True)
    )
    paired = _paired_cells(left, right)
    family = _family_sensitivity(case_differences, families)
    return {
        "effect": mean(case_differences),
        "paired_cells": paired,
        "net_left_wins": paired["left_only"] - paired["right_only"],
        "case_differences": list(case_differences),
        "case_signs": {
            "positive": sum(value > 0 for value in case_differences),
            "negative": sum(value < 0 for value in case_differences),
            "tie": sum(value == 0 for value in case_differences),
        },
        "cluster_bootstrap_ci": _bootstrap_intervals(
            case_differences,
            seed=bootstrap_seed,
        ),
        "case_sign_flip_one_sided_reference_p": _case_sign_flip_probability(
            case_differences
        ),
        **family,
    }


def _double_center(
    matrix: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("interaction matrix is invalid")
    row_means = tuple(mean(row) for row in matrix)
    column_means = tuple(
        mean(row[column] for row in matrix) for column in range(len(matrix[0]))
    )
    grand_mean = mean(row_means)
    return tuple(
        tuple(
            value - row_means[row_index] - column_means[column_index] + grand_mean
            for column_index, value in enumerate(row)
        )
        for row_index, row in enumerate(matrix)
    )


def _interaction_repeatability(
    discovery: tuple[tuple[float, ...], ...],
    evaluation: tuple[tuple[float, ...], ...],
) -> dict[str, object]:
    if len(discovery) != len(evaluation):
        raise ValueError("interaction split matrices differ")
    centered_discovery = _double_center(discovery)
    centered_evaluation = _double_center(evaluation)
    observed = sum(
        discovery_value * evaluation_value
        for discovery_row, evaluation_row in zip(
            centered_discovery,
            centered_evaluation,
            strict=True,
        )
        for discovery_value, evaluation_value in zip(
            discovery_row,
            evaluation_row,
            strict=True,
        )
    )
    discovery_norm = sum(value * value for row in centered_discovery for value in row)
    evaluation_norm = sum(value * value for row in centered_evaluation for value in row)
    denominator = (discovery_norm * evaluation_norm) ** 0.5
    correlation = observed / denominator if denominator else 0.0
    rng = random.Random(PILOT_PERMUTATION_SEED)
    permutation = list(range(len(evaluation)))
    exceedances = 0
    for _ in range(PILOT_PERMUTATION_SAMPLES):
        rng.shuffle(permutation)
        statistic = sum(
            discovery_value * evaluation_value
            for case_index, discovery_row in enumerate(centered_discovery)
            for discovery_value, evaluation_value in zip(
                discovery_row,
                centered_evaluation[permutation[case_index]],
                strict=True,
            )
        )
        exceedances += int(statistic >= observed - 1e-12)
    return {
        "statistic": observed,
        "centered_correlation": correlation,
        "permutation_samples": PILOT_PERMUTATION_SAMPLES,
        "permutation_seed": PILOT_PERMUTATION_SEED,
        "case_row_permutation_one_sided_p": (exceedances + 1)
        / (PILOT_PERMUTATION_SAMPLES + 1),
        "positive": observed > 0,
    }


def _analyze_action_lens_pilot(
    case_results: tuple[Mapping[str, tuple[int, ...]], ...],
    contract: ActionLensContract,
    families: tuple[str, ...],
) -> dict[str, object]:
    _validate_case_results(case_results, contract)
    if len(families) != len(case_results):
        raise ValueError("pilot family count differs")
    discovery = contract.discovery_repetitions
    evaluation = contract.evaluation_repetitions

    def values(
        result: Mapping[str, tuple[int, ...]], arm: str, repetitions: tuple[int, ...]
    ) -> tuple[int, ...]:
        return tuple(result[arm][repetition] for repetition in repetitions)

    def accuracy(arm: str, repetitions: tuple[int, ...]) -> float:
        return mean(
            result[arm][repetition]
            for result in case_results
            for repetition in repetitions
        )

    accuracy_all = {
        arm: accuracy(arm, tuple(range(contract.repetitions))) for arm in FACTORIAL_ARMS
    }
    accuracy_discovery = {arm: accuracy(arm, discovery) for arm in FACTORIAL_ARMS}
    accuracy_evaluation = {arm: accuracy(arm, evaluation) for arm in FACTORIAL_ARMS}

    child_local_evaluation = tuple(
        tuple(
            result[f"local_{target}"][repetition]
            for target in CHILD_TARGETS
            for repetition in evaluation
        )
        for result in case_results
    )
    child_full_evaluation = tuple(
        tuple(
            result[f"full_{target}"][repetition]
            for target in CHILD_TARGETS
            for repetition in evaluation
        )
        for result in case_results
    )
    q1 = _pilot_comparison(
        child_local_evaluation,
        child_full_evaluation,
        families=families,
        bootstrap_seed=PILOT_BOOTSTRAP_SEEDS[0],
    )
    child_local_discovery = tuple(
        tuple(
            result[f"local_{target}"][repetition]
            for target in CHILD_TARGETS
            for repetition in discovery
        )
        for result in case_results
    )
    child_full_discovery = tuple(
        tuple(
            result[f"full_{target}"][repetition]
            for target in CHILD_TARGETS
            for repetition in discovery
        )
        for result in case_results
    )
    q1_discovery = _pilot_comparison(
        child_local_discovery,
        child_full_discovery,
        families=families,
        bootstrap_seed=PILOT_BOOTSTRAP_SEEDS[0],
    )
    per_target = {
        target: _pilot_comparison(
            tuple(
                values(result, f"local_{target}", evaluation) for result in case_results
            ),
            tuple(
                values(result, f"full_{target}", evaluation) for result in case_results
            ),
            families=families,
            bootstrap_seed=PILOT_BOOTSTRAP_SEEDS[0],
        )
        for target in CHILD_TARGETS
    }
    q1_screen = {
        "minimum_effect": 1 / 12,
        "discovery_effect": q1_discovery["effect"],
        "passed": bool(
            q1["effect"] >= 1 / 12
            and q1["cluster_bootstrap_ci"]["90"][0] > 0
            and q1_discovery["effect"] > 0
        ),
    }

    fixed_child_scores = {
        target: accuracy_discovery[f"local_{target}"] for target in CHILD_TARGETS
    }
    best_fixed_score = max(fixed_child_scores.values())
    fixed_child_target = next(
        target
        for target in CHILD_TARGETS
        if fixed_child_scores[target] == best_fixed_score
    )
    fixed_child_evaluation = tuple(
        values(result, f"local_{fixed_child_target}", evaluation)
        for result in case_results
    )
    direct_evaluation = tuple(
        values(result, DIRECT_ARM, evaluation) for result in case_results
    )
    local_root_evaluation = tuple(
        values(result, "local_root", evaluation) for result in case_results
    )
    fixed_full_evaluation = tuple(
        values(result, f"full_{fixed_child_target}", evaluation)
        for result in case_results
    )
    fixed_vs_direct = _pilot_comparison(
        fixed_child_evaluation,
        direct_evaluation,
        families=families,
        bootstrap_seed=PILOT_BOOTSTRAP_SEEDS[1],
    )
    fixed_vs_root = _pilot_comparison(
        fixed_child_evaluation,
        local_root_evaluation,
        families=families,
        bootstrap_seed=PILOT_BOOTSTRAP_SEEDS[1],
    )
    fixed_vs_same_target_full = _pilot_comparison(
        fixed_child_evaluation,
        fixed_full_evaluation,
        families=families,
        bootstrap_seed=PILOT_BOOTSTRAP_SEEDS[1],
    )
    fixed_vs_direct_screen = bool(
        fixed_vs_direct["effect"] >= 1 / 8
        and fixed_vs_direct["cluster_bootstrap_ci"]["90"][0] > 0
    )
    fixed_vs_root_screen = bool(
        fixed_vs_root["effect"] > 0
        and fixed_vs_root["cluster_bootstrap_ci"]["90"][0] > 0
    )

    discovery_matrix = tuple(
        tuple(
            mean(values(result, f"local_{target}", discovery))
            for target in CHILD_TARGETS
        )
        for result in case_results
    )
    evaluation_matrix = tuple(
        tuple(
            mean(values(result, f"local_{target}", evaluation))
            for target in CHILD_TARGETS
        )
        for result in case_results
    )
    state_targets: list[str] = []
    discovery_ties = 0
    stable_unique: Counter[str] = Counter()
    for discovery_row, evaluation_row in zip(
        discovery_matrix,
        evaluation_matrix,
        strict=True,
    ):
        discovery_best = max(discovery_row)
        discovery_winners = tuple(
            target
            for target, score in zip(CHILD_TARGETS, discovery_row, strict=True)
            if score == discovery_best
        )
        if len(discovery_winners) == 1:
            selected_target = discovery_winners[0]
        else:
            selected_target = fixed_child_target
            discovery_ties += 1
        state_targets.append(selected_target)
        evaluation_best = max(evaluation_row)
        evaluation_winners = tuple(
            target
            for target, score in zip(CHILD_TARGETS, evaluation_row, strict=True)
            if score == evaluation_best
        )
        if len(discovery_winners) == 1 and discovery_winners == evaluation_winners:
            stable_unique[discovery_winners[0]] += 1
    adaptive_evaluation = tuple(
        values(result, f"local_{target}", evaluation)
        for result, target in zip(case_results, state_targets, strict=True)
    )
    adaptive_vs_fixed = _pilot_comparison(
        adaptive_evaluation,
        fixed_child_evaluation,
        families=families,
        bootstrap_seed=PILOT_BOOTSTRAP_SEEDS[2],
    )
    interaction = _interaction_repeatability(discovery_matrix, evaluation_matrix)
    adaptive_screen = bool(
        adaptive_vs_fixed["effect"] >= 1 / 12
        and adaptive_vs_fixed["cluster_bootstrap_ci"]["90"][0] > 0
        and sum(target != fixed_child_target for target in state_targets) >= 3
    )
    repeatability_screen = bool(adaptive_screen and interaction["positive"])

    root_vs_full = _pilot_comparison(
        local_root_evaluation,
        tuple(values(result, "full_root", evaluation) for result in case_results),
        families=families,
        bootstrap_seed=PILOT_BOOTSTRAP_SEEDS[0],
    )
    root_vs_direct = _pilot_comparison(
        local_root_evaluation,
        direct_evaluation,
        families=families,
        bootstrap_seed=PILOT_BOOTSTRAP_SEEDS[1],
    )

    if q1_screen["passed"] and fixed_vs_direct_screen and fixed_vs_root_screen:
        if repeatability_screen:
            recommendation = "fresh_semantic_observability_audit"
        else:
            recommendation = "retain_fixed_child_policy_no_selector"
    elif fixed_vs_direct_screen and fixed_vs_root_screen:
        recommendation = "fixed_child_signal_without_broad_locality_effect"
    elif q1_screen["passed"]:
        recommendation = "locality_only_no_simpler_baseline_advantage"
    else:
        recommendation = "stop_current_tree_lens_branch"

    return {
        "schema_version": ACTION_LENS_PILOT_ANALYSIS_VERSION,
        "development_only": True,
        "independent_unit": "source_state",
        "case_count": len(case_results),
        "families": list(families),
        "accuracy_all": accuracy_all,
        "accuracy_discovery": accuracy_discovery,
        "accuracy_evaluation": accuracy_evaluation,
        "q1_child_local_vs_same_target_full": {
            **q1,
            "discovery_comparison": q1_discovery,
            "per_target": per_target,
            "screen": q1_screen,
        },
        "q2_fixed_child_policy": {
            "selected_target": fixed_child_target,
            "discovery_scores": fixed_child_scores,
            "versus_direct": fixed_vs_direct,
            "versus_local_root": fixed_vs_root,
            "versus_same_target_full": fixed_vs_same_target_full,
            "direct_screen_passed": fixed_vs_direct_screen,
            "tree_specific_screen_passed": bool(
                fixed_vs_direct_screen and fixed_vs_root_screen
            ),
            "scope": (
                "conditional one-worker-call comparison; state construction and "
                "end-to-end cost are excluded"
            ),
        },
        "q3_reward_probed_target_repeatability": {
            "claim_scope": (
                "outcome-probed repeatability only; this is not semantic observability"
            ),
            "global_fixed_target": fixed_child_target,
            "state_selected_targets": state_targets,
            "state_selection_counts": dict(Counter(state_targets)),
            "discovery_tie_fallback_count": discovery_ties,
            "states_differing_from_fixed": sum(
                target != fixed_child_target for target in state_targets
            ),
            "stable_unique_winner_counts": dict(stable_unique),
            "adaptive_vs_fixed": adaptive_vs_fixed,
            "cross_split_interaction": interaction,
            "adaptive_value_screen_passed": adaptive_screen,
            "repeatability_screen_passed": repeatability_screen,
        },
        "root_self_revision_diagnostics": {
            "local_root_vs_full_root": root_vs_full,
            "local_root_vs_direct": root_vs_direct,
        },
        "screen_results": {
            "q1_child_lens": q1_screen["passed"],
            "q2_fixed_child_vs_direct": fixed_vs_direct_screen,
            "q2_tree_specific_vs_root": bool(
                fixed_vs_direct_screen and fixed_vs_root_screen
            ),
            "q3_reward_probed_repeatability": repeatability_screen,
        },
        "recommended_next_step": recommendation,
        "inference_note": (
            "sign-flip probabilities are exchangeability references, not "
            "design-exact tests; confirmation requires fresh states and randomized "
            "condition-to-slot assignment"
        ),
        "behavior_cloning_eligible": False,
    }


def analyze_action_lens(
    case_results: tuple[Mapping[str, tuple[int, ...]], ...],
    contract: ActionLensContract,
    *,
    families: tuple[str, ...] | None = None,
) -> dict[str, object]:
    if contract.stage == "integration_smoke":
        return _analyze_action_lens_integration(case_results, contract)
    if families is None:
        raise ValueError("pilot action-lens analysis requires case families")
    return _analyze_action_lens_pilot(case_results, contract, families)


@dataclass(frozen=True, slots=True)
class ActionLensCollectionResult:
    run_path: Path
    diagnostic_path: Path
    case_count: int
    logical_call_count: int
    transition_count: int
    run_contract_sha256: str
    analysis: Mapping[str, object]


def _source_state_record(
    contract: ActionLensContract,
    case_index: int,
    case: ActionLensSourceCase,
) -> dict[str, object]:
    return {
        "schema_version": ACTION_LENS_SOURCE_RECORD_VERSION,
        "record_type": "source_state",
        "preregistration_id": contract.preregistration_id,
        "case_index": case_index,
        **case.source_record(),
    }


def _candidate_from_call_payload(
    payload: Mapping[str, object] | None,
    *,
    generation_error: bool,
    error_type: str | None,
) -> Candidate:
    if generation_error:
        return Candidate(
            reasoning="",
            answer=None,
            usable=False,
            error_type=error_type or "generation_error",
        )
    if payload is None:
        raise ValueError("successful action-lens call lacks a payload")
    try:
        candidate = candidate_from_payload(payload)
        if candidate.answer is None or not candidate.answer.strip():
            raise ValueError("candidate answer must be nonempty")
        return candidate
    except ValueError as exc:
        return Candidate(
            reasoning="",
            answer=None,
            usable=False,
            error_type=type(exc).__name__,
        )


def _expected_contract(
    contract: ActionLensContract,
    prepared: PreparedActionLensSource,
) -> ActionLensContract:
    return build_action_lens_contract(
        preregistration_id=contract.preregistration_id,
        stage=contract.stage,
        prepared=prepared,
        worker=contract.worker,
        worker_temperature=contract.worker_temperature,
        worker_max_tokens=contract.worker_max_tokens,
        worker_seed=contract.worker_seed,
    )


def collect_action_lens_run(
    *,
    run_path: str | Path,
    diagnostic_path: str | Path,
    contract: ActionLensContract,
    prepared: PreparedActionLensSource,
    worker_client: OpenAICompatibleLLMClient,
) -> ActionLensCollectionResult:
    if not isinstance(contract, ActionLensContract):
        raise TypeError("action-lens collection requires its contract")
    if not isinstance(prepared, PreparedActionLensSource):
        raise TypeError("action-lens collection requires its prepared source")
    if type(worker_client) is not OpenAICompatibleLLMClient:
        raise TypeError("action-lens worker must be OpenAI-compatible")
    _validate_client_binding(contract.worker, worker_client, role="worker")
    if _expected_contract(contract, prepared) != contract:
        raise ValueError("action-lens contract differs from its prepared source")
    for case in prepared.cases:
        assert_request_isolation(case)

    final_run = Path(run_path)
    final_diagnostic = Path(diagnostic_path)
    partial_run, partial_diagnostic = _validate_output_paths(
        final_run,
        final_diagnostic,
    )
    diagnostic_header = {
        "schema_version": ACTION_LENS_DIAGNOSTIC_VERSION,
        "record_type": "diagnostic_header",
        "preregistration_id": contract.preregistration_id,
        "run_contract_sha256": contract.sha256,
        "attempt_schema_version": DIAGNOSTIC_ATTEMPT_VERSION,
        "roles": list(FACTORIAL_ARMS),
    }
    worker_before = _counter_snapshot(worker_client)
    old_observer = worker_client.attempt_observer
    case_results: list[Mapping[str, tuple[int, ...]]] = []
    logical_call_count = 0
    transition_count = 0
    call_failures: Counter[str] = Counter({arm: 0 for arm in FACTORIAL_ARMS})
    unusable_outputs: Counter[str] = Counter({arm: 0 for arm in FACTORIAL_ARMS})

    try:
        with (
            partial_run.open("x", encoding="ascii", newline="\n") as run_handle,
            partial_diagnostic.open(
                "x", encoding="ascii", newline="\n"
            ) as diagnostic_handle,
        ):
            _write_record(run_handle, contract.header())
            _write_record(diagnostic_handle, diagnostic_header, flush=True)
            recorder = _AttemptRecorder(
                diagnostic_handle,
                roles=FACTORIAL_ARMS,
            )
            for case_index, case in enumerate(prepared.cases):
                _write_record(
                    run_handle,
                    _source_state_record(contract, case_index, case),
                )
                rewards: dict[str, list[int]] = {arm: [] for arm in FACTORIAL_ARMS}
                usable: dict[str, list[bool]] = {arm: [] for arm in FACTORIAL_ARMS}
                call_ids: dict[str, list[str]] = {arm: [] for arm in FACTORIAL_ARMS}
                orders: list[list[str]] = []
                requests = {arm: request_for_arm(case, arm) for arm in FACTORIAL_ARMS}
                for repetition in range(contract.repetitions):
                    order = arm_order(case.source_problem_index, repetition)
                    orders.append(list(order))
                    seed = (
                        contract.worker_seed
                        + case.source_problem_index * contract.repetitions
                        + repetition
                    )
                    for arm_position, arm in enumerate(order):
                        request = requests[arm]
                        call = _logical_call(
                            client=worker_client,
                            recorder=recorder,
                            role=arm,
                            request=request,
                            temperature=contract.worker_temperature,
                            max_tokens=contract.worker_max_tokens,
                            seed=seed,
                        )
                        candidate = _candidate(call)
                        reward = terminal_reward(candidate, case.problem)
                        view_mode, target = arm_view_target(arm)
                        transition = {
                            "schema_version": ACTION_LENS_TRANSITION_VERSION,
                            "record_type": "lens_transition",
                            "preregistration_id": contract.preregistration_id,
                            "case_index": case_index,
                            "source_problem_index": case.source_problem_index,
                            "problem_id": case.problem.problem_id,
                            "source_state_sha256": case.source_state_sha256,
                            "repetition": repetition,
                            "split": (
                                "discovery"
                                if repetition in contract.discovery_repetitions
                                else "evaluation"
                            ),
                            "arm": arm,
                            "view_mode": view_mode,
                            "target": target,
                            "arm_position": arm_position,
                            "seed": seed,
                            "request_sha256": request_sha256(request),
                            "request": request.to_record(),
                            "worker_call_id": call.call_id,
                            "candidate": candidate.to_record(),
                            "generation_error": call.generation_error,
                            "error_type": call.error_type,
                            "reward": reward,
                            "behavior_cloning_eligible": False,
                        }
                        _write_record(run_handle, transition)
                        rewards[arm].append(reward)
                        usable[arm].append(candidate.usable)
                        call_ids[arm].append(call.call_id)
                        call_failures[arm] += int(call.generation_error)
                        unusable_outputs[arm] += int(not candidate.usable)
                        logical_call_count += 1
                        transition_count += 1
                normalized_rewards = {
                    arm: tuple(rewards[arm]) for arm in FACTORIAL_ARMS
                }
                case_results.append(normalized_rewards)
                _write_record(
                    run_handle,
                    {
                        "schema_version": ACTION_LENS_CASE_VERSION,
                        "record_type": "case_summary",
                        "preregistration_id": contract.preregistration_id,
                        "case_index": case_index,
                        "source_problem_index": case.source_problem_index,
                        "problem_id": case.problem.problem_id,
                        "source_state_sha256": case.source_state_sha256,
                        "arm_orders": orders,
                        "rewards": {arm: rewards[arm] for arm in FACTORIAL_ARMS},
                        "usable": {arm: usable[arm] for arm in FACTORIAL_ARMS},
                        "worker_call_ids": call_ids,
                        "behavior_cloning_eligible": False,
                    },
                    flush=True,
                )
            analysis = analyze_action_lens(
                tuple(case_results),
                contract,
                families=tuple(case.problem.family for case in prepared.cases),
            )
            summary = {
                "schema_version": ACTION_LENS_SUMMARY_VERSION,
                "record_type": "run_summary",
                "preregistration_id": contract.preregistration_id,
                "run_contract_sha256": contract.sha256,
                "case_count": len(prepared.cases),
                "logical_call_count": logical_call_count,
                "transition_count": transition_count,
                "call_failure_counts": dict(call_failures),
                "unusable_output_counts": dict(unusable_outputs),
                "usage": {"worker": _counter_delta(worker_before, worker_client)},
                "analysis": analysis,
                "behavior_cloning_eligible": False,
            }
            _write_record(run_handle, summary)
            run_handle.flush()
            diagnostic_handle.flush()
            os.fsync(run_handle.fileno())
            os.fsync(diagnostic_handle.fileno())
    finally:
        worker_client.attempt_observer = old_observer

    audit_action_lens_artifacts(
        run_path=partial_run,
        diagnostic_path=partial_diagnostic,
        contract=contract,
        prepared=prepared,
    )
    _finalize_no_clobber(
        (
            (partial_diagnostic, final_diagnostic),
            (partial_run, final_run),
        )
    )
    return ActionLensCollectionResult(
        run_path=final_run,
        diagnostic_path=final_diagnostic,
        case_count=len(prepared.cases),
        logical_call_count=logical_call_count,
        transition_count=transition_count,
        run_contract_sha256=contract.sha256,
        analysis=analysis,
    )


def audit_action_lens_artifacts(
    *,
    run_path: str | Path,
    diagnostic_path: str | Path,
    contract: ActionLensContract,
    prepared: PreparedActionLensSource,
) -> dict[str, object]:
    if _expected_contract(contract, prepared) != contract:
        raise ValueError("action-lens audit contract differs")
    run_records = _load_jsonl(run_path, "action-lens run")
    diagnostic_records = _load_jsonl(diagnostic_path, "action-lens diagnostics")
    if run_records[0] != contract.header():
        raise ValueError("action-lens run header differs")
    expected_diagnostic_header = {
        "schema_version": ACTION_LENS_DIAGNOSTIC_VERSION,
        "record_type": "diagnostic_header",
        "preregistration_id": contract.preregistration_id,
        "run_contract_sha256": contract.sha256,
        "attempt_schema_version": DIAGNOSTIC_ATTEMPT_VERSION,
        "roles": list(FACTORIAL_ARMS),
    }
    if diagnostic_records[0] != expected_diagnostic_header:
        raise ValueError("action-lens diagnostic header differs")

    sources = [
        record for record in run_records if record.get("record_type") == "source_state"
    ]
    transitions = [
        record
        for record in run_records
        if record.get("record_type") == "lens_transition"
    ]
    summaries = [
        record for record in run_records if record.get("record_type") == "case_summary"
    ]
    run_summaries = [
        record for record in run_records if record.get("record_type") == "run_summary"
    ]
    expected_calls = len(prepared.cases) * contract.repetitions * len(FACTORIAL_ARMS)
    if (
        len(sources) != len(prepared.cases)
        or len(transitions) != expected_calls
        or len(summaries) != len(prepared.cases)
        or len(run_summaries) != 1
        or len(run_records) != 2 + 2 * len(prepared.cases) + expected_calls
    ):
        raise ValueError("action-lens run record counts differ")
    expected_layout: list[tuple[str, int | None, int | None, str | None]] = []
    for case_index, case in enumerate(prepared.cases):
        expected_layout.append(("source_state", case_index, None, None))
        for repetition in range(contract.repetitions):
            expected_layout.extend(
                ("lens_transition", case_index, repetition, arm)
                for arm in arm_order(case.source_problem_index, repetition)
            )
        expected_layout.append(("case_summary", case_index, None, None))
    expected_layout.append(("run_summary", None, None, None))
    observed_layout = [
        (
            str(record.get("record_type")),
            (
                int(record["case_index"])
                if type(record.get("case_index")) is int
                else None
            ),
            (
                int(record["repetition"])
                if type(record.get("repetition")) is int
                else None
            ),
            str(record["arm"]) if isinstance(record.get("arm"), str) else None,
        )
        for record in run_records[1:]
    ]
    if observed_layout != expected_layout:
        raise ValueError("action-lens run record order differs")
    for case_index, (case, source) in enumerate(
        zip(prepared.cases, sources, strict=True)
    ):
        if source != _source_state_record(contract, case_index, case):
            raise ValueError("action-lens source State record differs")
        assert_request_isolation(case)

    transition_by_key: dict[tuple[int, int, str], Mapping[str, object]] = {}
    canonical_call_ids: list[str] = []
    case_results: list[dict[str, list[int]]] = [
        {arm: [] for arm in FACTORIAL_ARMS} for _ in prepared.cases
    ]
    case_usable: list[dict[str, list[bool]]] = [
        {arm: [] for arm in FACTORIAL_ARMS} for _ in prepared.cases
    ]
    reconstructed_failures: Counter[str] = Counter({arm: 0 for arm in FACTORIAL_ARMS})
    reconstructed_unusable: Counter[str] = Counter({arm: 0 for arm in FACTORIAL_ARMS})
    for transition in transitions:
        case_index = transition.get("case_index")
        repetition = transition.get("repetition")
        arm = transition.get("arm")
        if (
            type(case_index) is not int
            or case_index not in range(len(prepared.cases))
            or type(repetition) is not int
            or repetition not in range(contract.repetitions)
            or arm not in FACTORIAL_ARMS
        ):
            raise ValueError("action-lens transition key is invalid")
        key = (case_index, repetition, str(arm))
        if key in transition_by_key:
            raise ValueError("action-lens transition key is duplicated")
        transition_by_key[key] = transition
        case = prepared.cases[case_index]
        order = arm_order(case.source_problem_index, repetition)
        request = request_for_arm(case, str(arm))
        view_mode, target = arm_view_target(str(arm))
        seed = contract.worker_seed + case.source_problem_index * 4 + repetition
        request_record = request.to_record()
        candidate_record = transition.get("candidate")
        if not isinstance(candidate_record, Mapping):
            raise ValueError("action-lens transition Candidate is invalid")
        try:
            candidate = Candidate(**dict(candidate_record))
        except (TypeError, ValueError) as exc:
            raise ValueError("action-lens transition Candidate is invalid") from exc
        reward = terminal_reward(candidate, case.problem)
        call_id = transition.get("worker_call_id")
        generation_error = transition.get("generation_error")
        expected_error_type = "LLMResponseError" if generation_error is True else None
        if (
            set(transition) != _TRANSITION_FIELDS
            or transition.get("schema_version") != ACTION_LENS_TRANSITION_VERSION
            or transition.get("record_type") != "lens_transition"
            or transition.get("preregistration_id") != contract.preregistration_id
            or transition.get("source_problem_index") != case.source_problem_index
            or transition.get("problem_id") != case.problem.problem_id
            or transition.get("source_state_sha256") != case.source_state_sha256
            or transition.get("split")
            != (
                "discovery"
                if repetition in contract.discovery_repetitions
                else "evaluation"
            )
            or transition.get("view_mode") != view_mode
            or transition.get("target") != target
            or transition.get("arm_position") != order.index(str(arm))
            or transition.get("seed") != seed
            or transition.get("request") != request_record
            or transition.get("request_sha256") != request_sha256(request)
            or type(generation_error) is not bool
            or transition.get("error_type") != expected_error_type
            or transition.get("reward") != reward
            or transition.get("behavior_cloning_eligible") is not False
            or not isinstance(call_id, str)
        ):
            raise ValueError("action-lens transition replay differs")
        canonical_call_ids.append(call_id)
        case_results[case_index][str(arm)].append(reward)
        case_usable[case_index][str(arm)].append(candidate.usable)
        assert isinstance(generation_error, bool)
        reconstructed_failures[str(arm)] += int(generation_error)
        reconstructed_unusable[str(arm)] += int(not candidate.usable)
    if len(transition_by_key) != expected_calls or len(canonical_call_ids) != len(
        set(canonical_call_ids)
    ):
        raise ValueError("action-lens transition grid or call ids differ")

    attempts = diagnostic_records[1:]
    if any(record.get("record_type") != "http_attempt" for record in attempts):
        raise ValueError("action-lens diagnostic record type differs")
    attempts_by_call: dict[str, list[tuple[str, Mapping[str, object]]]] = {}
    diagnostic_call_order: list[str] = []
    for sequence, record in enumerate(attempts):
        role = record.get("role")
        attempt = record.get("attempt")
        if (
            set(record) != _DIAGNOSTIC_RECORD_FIELDS
            or record.get("schema_version") != DIAGNOSTIC_ATTEMPT_VERSION
            or record.get("sequence") != sequence
            or role not in FACTORIAL_ARMS
            or not isinstance(attempt, Mapping)
            or not isinstance(attempt.get("call_id"), str)
        ):
            raise ValueError("action-lens diagnostic attempt is invalid")
        call_id = str(attempt["call_id"])
        attempts_by_call.setdefault(call_id, []).append((str(role), attempt))
        if not diagnostic_call_order or diagnostic_call_order[-1] != call_id:
            diagnostic_call_order.append(call_id)
    if diagnostic_call_order != canonical_call_ids or set(attempts_by_call) != set(
        canonical_call_ids
    ):
        raise ValueError("action-lens diagnostic call order differs")

    replayed_usage = Counter(
        {"generate_calls": 0, "prompt_tokens": 0, "generated_tokens": 0}
    )
    replayed_usage_by_arm: dict[str, Counter[str]] = {
        arm: Counter({"generate_calls": 0, "prompt_tokens": 0, "generated_tokens": 0})
        for arm in FACTORIAL_ARMS
    }
    replayed_attempt_counts: Counter[str] = Counter({arm: 0 for arm in FACTORIAL_ARMS})
    first_attempt_accepted: Counter[str] = Counter({arm: 0 for arm in FACTORIAL_ARMS})
    attempt_outcomes: dict[str, Counter[str]] = {
        arm: Counter() for arm in FACTORIAL_ARMS
    }
    observed_response_models: dict[str, set[str]] = {
        arm: set() for arm in FACTORIAL_ARMS
    }
    attempt_fields = {
        "call_id",
        "attempt_index",
        "parse_retry_index",
        "backoff_retry_index",
        "effective_seed",
        "request_body_json",
        "raw_response_body",
        "raw_content",
        "raw_usage_json",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "outcome",
        "disposition",
        "http_status",
        "error_type",
        "error_message",
    }
    for transition in transitions:
        arm = str(transition["arm"])
        call_id = str(transition["worker_call_id"])
        request_record = transition["request"]
        assert isinstance(request_record, Mapping)
        request = WorkerRequest(
            system=str(request_record["system"]),
            user=str(request_record["user"]),
            schema=request_record["schema"],  # type: ignore[arg-type]
        )
        seed = int(transition["seed"])
        expected_body = build_openai_compatible_json_body(
            model=contract.worker.model,
            system=request.system,
            user=request.user,
            schema=request.schema,
            temperature=contract.worker_temperature,
            max_tokens=contract.worker_max_tokens,
            seed=seed,
            guided_json=contract.worker.guided_json,
        )
        parse_retry_index = 0
        backoff_retry_index = 0
        terminal_payload: Mapping[str, object] | None = None
        terminal_disposition: str | None = None
        call_attempts = attempts_by_call[call_id]
        for attempt_index, (observed_role, attempt) in enumerate(call_attempts):
            replayed_attempt_counts[arm] += 1
            if observed_role != arm:
                raise ValueError("action-lens diagnostic role differs")
            if set(attempt) != attempt_fields:
                raise ValueError("action-lens diagnostic attempt fields differ")
            effective_seed = seed + parse_retry_index
            expected_body["seed"] = effective_seed
            request_body_json = attempt.get("request_body_json")
            if not isinstance(request_body_json, str):
                raise ValueError("action-lens attempt lacks request body")
            observed_body = json.loads(
                request_body_json,
                object_pairs_hook=_pairs_object,
                parse_constant=_reject_constant,
            )
            if (
                attempt.get("call_id") != call_id
                or attempt.get("attempt_index") != attempt_index
                or attempt.get("parse_retry_index") != parse_retry_index
                or attempt.get("backoff_retry_index") != backoff_retry_index
                or attempt.get("effective_seed") != effective_seed
                or observed_body != expected_body
                or request_body_json
                != canonical_json_bytes(observed_body).decode("ascii")
            ):
                raise ValueError("action-lens retry request differs")
            outcome = attempt.get("outcome")
            disposition = attempt.get("disposition")
            if not isinstance(outcome, str) or not isinstance(disposition, str):
                raise ValueError("action-lens attempt outcome is invalid")
            attempt_outcomes[arm][f"{outcome}:{disposition}"] += 1
            if attempt_index == 0 and outcome == "accepted" and disposition == "return":
                first_attempt_accepted[arm] += 1
            raw_content = attempt.get("raw_content")
            raw_response_body = attempt.get("raw_response_body")
            http_status = attempt.get("http_status")
            error_type = attempt.get("error_type")
            error_message = attempt.get("error_message")
            successful_response = outcome in {
                "provider_response_error",
                "parse_error",
                "schema_error",
                "accepted",
            }
            if successful_response:
                if type(http_status) is not int or not 200 <= http_status < 300:
                    raise ValueError("action-lens successful HTTP status differs")
                replayed_usage["generate_calls"] += 1
                replayed_usage_by_arm[arm]["generate_calls"] += 1
                for field_name, counter_name in (
                    ("prompt_tokens", "prompt_tokens"),
                    ("completion_tokens", "generated_tokens"),
                ):
                    value = attempt.get(field_name)
                    if value is not None and (type(value) is not int or value < 0):
                        raise ValueError("action-lens token count is invalid")
                    replayed_usage[counter_name] += value or 0
                    replayed_usage_by_arm[arm][counter_name] += value or 0
                total_tokens = attempt.get("total_tokens")
                if total_tokens is not None and (
                    type(total_tokens) is not int or total_tokens < 0
                ):
                    raise ValueError("action-lens total token count is invalid")
            elif outcome == "http_status_error":
                if type(http_status) is not int or 200 <= http_status < 300:
                    raise ValueError("action-lens HTTP error status differs")
            elif outcome == "transport_error":
                if http_status is not None:
                    raise ValueError("action-lens transport status differs")
            else:
                raise ValueError("action-lens attempt outcome is invalid")
            if outcome == "accepted":
                if error_type is not None or error_message is not None:
                    raise ValueError("accepted action-lens attempt has an error")
            elif (
                not isinstance(error_type, str)
                or not error_type
                or not isinstance(error_message, str)
                or not error_message
            ):
                raise ValueError("failed action-lens attempt lacks its error")
            if outcome == "http_status_error" and error_message != (
                f"HTTP status {http_status}"
            ):
                raise ValueError("action-lens HTTP error message differs")
            if outcome in {"parse_error", "schema_error"} and error_type != (
                "LLMResponseError"
            ):
                raise ValueError("action-lens content error type differs")

            response_payload: object | None = None
            if raw_response_body is not None:
                if not isinstance(raw_response_body, str):
                    raise ValueError("action-lens raw response is invalid")
                try:
                    response_payload = json.loads(
                        raw_response_body,
                        object_pairs_hook=_pairs_object,
                        parse_constant=_reject_constant,
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    response_payload = None
            if isinstance(response_payload, Mapping):
                response_model = response_payload.get("model")
                if isinstance(response_model, str) and response_model:
                    observed_response_models[arm].add(response_model)
            if successful_response and outcome != "provider_response_error":
                if not isinstance(response_payload, Mapping):
                    raise ValueError("action-lens provider response is not replayable")
                projection = project_openai_compatible_response(dict(response_payload))
                if projection.provider_error is not None or (
                    projection.raw_content,
                    projection.raw_usage_json,
                    projection.prompt_tokens,
                    projection.completion_tokens,
                    projection.total_tokens,
                ) != (
                    raw_content,
                    attempt.get("raw_usage_json"),
                    attempt.get("prompt_tokens"),
                    attempt.get("completion_tokens"),
                    attempt.get("total_tokens"),
                ):
                    raise ValueError("action-lens provider projection differs")
            elif outcome == "provider_response_error":
                if isinstance(response_payload, Mapping):
                    projection = project_openai_compatible_response(
                        dict(response_payload)
                    )
                    if projection.provider_error is None or (
                        projection.raw_content,
                        projection.raw_usage_json,
                        projection.prompt_tokens,
                        projection.completion_tokens,
                        projection.total_tokens,
                    ) != (
                        raw_content,
                        attempt.get("raw_usage_json"),
                        attempt.get("prompt_tokens"),
                        attempt.get("completion_tokens"),
                        attempt.get("total_tokens"),
                    ):
                        raise ValueError(
                            "action-lens provider error projection differs"
                        )
                elif any(
                    attempt.get(field) is not None
                    for field in (
                        "raw_content",
                        "raw_usage_json",
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                    )
                ):
                    raise ValueError("invalid provider JSON exposed projected fields")
            elif outcome in {"http_status_error", "transport_error"} and any(
                attempt.get(field) is not None
                for field in (
                    "raw_content",
                    "raw_usage_json",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                )
            ):
                raise ValueError("failed HTTP attempt exposed projected fields")
            if outcome in {"parse_error", "schema_error", "accepted"}:
                if not isinstance(raw_content, str):
                    raise ValueError("action-lens content outcome lacks text")
                classified, payload = classify_openai_compatible_json_content(
                    raw_content,
                    request.schema,
                    schema_validation_mode=SCHEMA_VALIDATION_STRICT_V1,
                )
                if classified != outcome:
                    raise ValueError("action-lens content outcome differs")
                if outcome == "accepted":
                    terminal_payload = payload
            if disposition == "backoff_retry":
                if (
                    outcome != "http_status_error"
                    or not (http_status == 429 or http_status >= 500)
                    or backoff_retry_index >= contract.worker.backoff_retries
                ):
                    raise ValueError("action-lens backoff retry is invalid")
                backoff_retry_index += 1
                continue
            if disposition == "parse_retry":
                if (
                    outcome
                    not in {
                        "transport_error",
                        "provider_response_error",
                        "parse_error",
                        "schema_error",
                    }
                    or parse_retry_index >= contract.worker.parse_retries
                ):
                    raise ValueError("action-lens parse retry is invalid")
                append_openai_compatible_parse_retry(
                    expected_body,
                    raw_content if isinstance(raw_content, str) else None,
                )
                parse_retry_index += 1
                continue
            if disposition not in {"return", "raise"}:
                raise ValueError("action-lens terminal retry disposition differs")
            if attempt_index != len(call_attempts) - 1:
                raise ValueError("action-lens terminal attempt is not last")
            if disposition == "return" and outcome != "accepted":
                raise ValueError("action-lens return attempt was not accepted")
            if disposition == "raise" and outcome == "accepted":
                raise ValueError("action-lens accepted attempt was raised")
            if (
                disposition == "raise"
                and outcome
                in {
                    "transport_error",
                    "provider_response_error",
                    "parse_error",
                    "schema_error",
                }
                and parse_retry_index < contract.worker.parse_retries
            ):
                raise ValueError("action-lens parse-retryable error stopped early")
            if (
                disposition == "raise"
                and outcome == "http_status_error"
                and (http_status == 429 or http_status >= 500)
                and backoff_retry_index < contract.worker.backoff_retries
            ):
                raise ValueError("action-lens retryable HTTP error stopped early")
            terminal_disposition = disposition
        if terminal_disposition is None:
            raise ValueError("action-lens call lacks a terminal attempt")
        generation_error = bool(transition["generation_error"])
        if generation_error != (terminal_disposition == "raise"):
            raise ValueError("action-lens generation failure differs from attempts")
        if not generation_error and terminal_payload is None:
            raise ValueError("successful action-lens call lacks a payload")
        replayed_candidate = _candidate_from_call_payload(
            terminal_payload,
            generation_error=generation_error,
            error_type=(
                str(transition["error_type"])
                if transition.get("error_type") is not None
                else None
            ),
        )
        if transition.get("candidate") != replayed_candidate.to_record():
            raise ValueError("action-lens Candidate differs from raw response")

    normalized_case_results = tuple(
        {arm: tuple(result[arm]) for arm in FACTORIAL_ARMS} for result in case_results
    )
    for case_index, (case, summary) in enumerate(
        zip(prepared.cases, summaries, strict=True)
    ):
        expected_orders = [
            list(arm_order(case.source_problem_index, repetition))
            for repetition in range(contract.repetitions)
        ]
        expected_call_ids = {
            arm: [
                str(transition_by_key[(case_index, repetition, arm)]["worker_call_id"])
                for repetition in range(contract.repetitions)
            ]
            for arm in FACTORIAL_ARMS
        }
        if (
            set(summary) != _CASE_SUMMARY_FIELDS
            or summary.get("schema_version") != ACTION_LENS_CASE_VERSION
            or summary.get("record_type") != "case_summary"
            or summary.get("preregistration_id") != contract.preregistration_id
            or summary.get("case_index") != case_index
            or summary.get("source_problem_index") != case.source_problem_index
            or summary.get("problem_id") != case.problem.problem_id
            or summary.get("source_state_sha256") != case.source_state_sha256
            or summary.get("arm_orders") != expected_orders
            or summary.get("rewards")
            != {
                arm: list(normalized_case_results[case_index][arm])
                for arm in FACTORIAL_ARMS
            }
            or summary.get("usable") != case_usable[case_index]
            or summary.get("worker_call_ids") != expected_call_ids
            or summary.get("behavior_cloning_eligible") is not False
        ):
            raise ValueError("action-lens case summary differs")
    replay_analysis = analyze_action_lens(
        normalized_case_results,
        contract,
        families=tuple(case.problem.family for case in prepared.cases),
    )
    run_summary = run_summaries[0]
    if (
        set(run_summary) != _RUN_SUMMARY_FIELDS
        or run_summary.get("schema_version") != ACTION_LENS_SUMMARY_VERSION
        or run_summary.get("record_type") != "run_summary"
        or run_summary.get("preregistration_id") != contract.preregistration_id
        or run_summary.get("run_contract_sha256") != contract.sha256
        or run_summary.get("case_count") != len(prepared.cases)
        or run_summary.get("logical_call_count") != expected_calls
        or run_summary.get("transition_count") != expected_calls
        or run_summary.get("call_failure_counts") != dict(reconstructed_failures)
        or run_summary.get("unusable_output_counts") != dict(reconstructed_unusable)
        or run_summary.get("usage") != {"worker": dict(replayed_usage)}
        or run_summary.get("analysis") != replay_analysis
        or run_summary.get("behavior_cloning_eligible") is not False
    ):
        raise ValueError("action-lens run summary differs")

    run_bytes = Path(run_path).read_bytes()
    diagnostic_bytes = Path(diagnostic_path).read_bytes()
    for forbidden in (
        b'"expected_answer"',
        b'"solution_set"',
        b'"params"',
        b'"Authorization"',
        b"Bearer ",
    ):
        if forbidden in run_bytes or forbidden in diagnostic_bytes:
            raise ValueError("action-lens artifact contains a forbidden field")
    return {
        "schema_version": "action-lens-artifact-audit-v1",
        "run_sha256": file_sha256(run_path),
        "diagnostic_sha256": file_sha256(diagnostic_path),
        "run_record_count": len(run_records),
        "diagnostic_record_count": len(diagnostic_records),
        "logical_call_count": expected_calls,
        "transition_count": expected_calls,
        "unique_state_count": len(prepared.cases),
        "call_id_binding_passed": True,
        "request_hash_replay_passed": True,
        "source_state_replay_passed": True,
        "reward_replay_passed": True,
        "prompt_isolation_passed": True,
        "common_seed_passed": True,
        "order_schedule_passed": True,
        "summary_replay_passed": True,
        "retry_transcript_replay_passed": True,
        "usage_replay_passed": True,
        "usage_by_arm": {
            arm: dict(usage) for arm, usage in replayed_usage_by_arm.items()
        },
        "forbidden_field_scan_passed": True,
        "replayed_attempt_counts": dict(replayed_attempt_counts),
        "first_attempt_accepted_counts": dict(first_attempt_accepted),
        "attempt_outcome_counts": {
            arm: dict(sorted(counts.items()))
            for arm, counts in attempt_outcomes.items()
        },
        "observed_response_models": {
            arm: sorted(models) for arm, models in observed_response_models.items()
        },
        "served_revision_attested": False,
        "served_revision_note": (
            "registered revisions are request-side metadata; the provider does "
            "not attest immutable served weight revisions"
        ),
    }


__all__ = [
    "ACTION_LENS_ANALYSIS_VERSION",
    "ACTION_LENS_CONTRACT_VERSION",
    "ACTION_LENS_LOCAL_VIEW_VERSION",
    "ACTION_LENS_PILOT_ANALYSIS_VERSION",
    "ActionLensCollectionResult",
    "ActionLensContract",
    "ActionLensSourceCase",
    "CHILD_TARGETS",
    "DIRECT_ARM",
    "FACTORIAL_ARMS",
    "FULL_ARMS",
    "LOCAL_ARMS",
    "PreparedActionLensSource",
    "WILLIAMS_ROWS",
    "analyze_action_lens",
    "arm_order",
    "arm_view_target",
    "assert_request_isolation",
    "audit_action_lens_artifacts",
    "build_action_lens_contract",
    "collect_action_lens_run",
    "order_schedule_record",
    "prepare_action_lens_source",
    "prompt_bindings_factorial",
    "request_for_arm",
    "state_from_observation",
    "target_local_action_request",
    "target_local_observation",
]
