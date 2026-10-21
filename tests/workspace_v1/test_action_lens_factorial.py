from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from verifier_anchored_rl.workspace_v1 import (
    action_lens_factorial as action_lens_factorial_module,
)
from verifier_anchored_rl.llm_client import (
    LLMResponseError,
    OpenAICompatibleAttempt,
    OpenAICompatibleLLMClient,
    build_openai_compatible_json_body,
)
from verifier_anchored_rl.workspace_v1.action_census import (
    ACTION_CENSUS_ACTION_VERSION,
    ACTION_CENSUS_REWARD_VERSION,
    ACTION_CENSUS_STATE_VERSION,
    ACTION_TARGETS,
    BUILDER_MODE_ANCHORED_RANGE_LISTS,
    CensusProblem,
    CensusState,
    SemanticNode,
    evaluation_binding_sha256,
    file_sha256,
    problem_source_sha256,
)
from verifier_anchored_rl.workspace_v1.action_lens_factorial import (
    ACTION_LENS_LOCAL_VIEW_VERSION,
    ACTION_LENS_PILOT_ANALYSIS_VERSION,
    CHILD_TARGETS,
    DIRECT_ARM,
    FACTORIAL_ARMS,
    PILOT_BOOTSTRAP_SAMPLES,
    PILOT_BOOTSTRAP_SEEDS,
    PILOT_PERMUTATION_SAMPLES,
    PILOT_PERMUTATION_SEED,
    WILLIAMS_ROWS,
    ActionLensContract,
    ActionLensSourceCase,
    PreparedActionLensSource,
    analyze_action_lens,
    arm_order,
    assert_request_isolation,
    audit_action_lens_artifacts,
    build_action_lens_contract,
    collect_action_lens_run,
    prepare_action_lens_source,
    request_for_arm,
    state_from_observation,
    target_local_observation,
)
from verifier_anchored_rl.workspace_v1.experiment import ModelIdentity
from verifier_anchored_rl.workspace_v1.records import canonical_json_bytes
from verifier_anchored_rl.workspace_v1.target_value import Candidate


def _problem() -> CensusProblem:
    return CensusProblem(
        problem_id="p0",
        problem="What is 1 + 2?",
        family="f-test",
        expected_answer="3",
    )


def _state() -> CensusState:
    problem = _problem()
    return CensusState(
        problem_id=problem.problem_id,
        problem=problem.problem,
        draft=Candidate(
            "Interpret the quantities. Add one and two. Return three.",
            "3",
        ),
        nodes=(
            SemanticNode("node_1", "recheck_model [S001]", "Interpret the quantities."),
            SemanticNode("node_2", "recompute_local [S002]", "Add one and two."),
            SemanticNode("node_3", "check_finalization [S003]", "Return three."),
        ),
    )


def _identity(*, parse_retries: int = 0) -> ModelIdentity:
    return ModelIdentity(
        provider="test",
        model="test/action-lens-worker",
        revision="revision",
        base_url="https://worker.example.test/v1",
        guided_json=False,
        timeout_s=1.0,
        parse_retries=parse_retries,
        backoff_retries=0,
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_bytes(
        b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    )


def _prepared() -> PreparedActionLensSource:
    problem = _problem()
    state = _state()
    case = ActionLensSourceCase(
        source_problem_index=0,
        problem=problem,
        state=state,
        source_state_sha256=state.sha256,
    )
    return PreparedActionLensSource(
        source_run_sha256="1" * 64,
        source_diagnostic_sha256="2" * 64,
        source_run_contract_sha256="3" * 64,
        dataset_sha256="4" * 64,
        selection_sha256="5" * 64,
        cases=(case,),
    )


def _contract(
    prepared: PreparedActionLensSource | None = None,
    *,
    stage: str = "integration_smoke",
) -> ActionLensContract:
    return build_action_lens_contract(
        preregistration_id="E47-test",
        stage=stage,
        prepared=prepared or _prepared(),
        worker=_identity(),
        worker_temperature=0.2,
        worker_max_tokens=384,
        worker_seed=100,
    )


def _prepared_case_count(count: int) -> PreparedActionLensSource:
    cases: list[ActionLensSourceCase] = []
    for index in range(count):
        problem = CensusProblem(
            problem_id=f"p{index}",
            problem=f"What is {index} + 1?",
            family=f"f{index % 2}",
            expected_answer=str(index + 1),
        )
        state = CensusState(
            problem_id=problem.problem_id,
            problem=problem.problem,
            draft=Candidate(f"Add one to {index}. Return {index + 1}.", str(index + 1)),
            nodes=(
                SemanticNode("node_1", "recheck_model [S001]", "Check the setup."),
                SemanticNode("node_2", "recompute_local [S002]", "Redo the sum."),
                SemanticNode(
                    "node_3",
                    "check_finalization [S003]",
                    "Check the reported value.",
                ),
            ),
        )
        cases.append(
            ActionLensSourceCase(
                source_problem_index=index,
                problem=problem,
                state=state,
                source_state_sha256=state.sha256,
            )
        )
    return PreparedActionLensSource(
        source_run_sha256="1" * 64,
        source_diagnostic_sha256="2" * 64,
        source_run_contract_sha256="3" * 64,
        dataset_sha256="4" * 64,
        selection_sha256="5" * 64,
        cases=tuple(cases),
    )


def test_source_replay_and_target_local_projection_are_exact(tmp_path: Path) -> None:
    problem = _problem()
    state = _state()
    dataset = tmp_path / "problems.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": problem.problem_id,
                "problem": problem.problem,
                "family": problem.family,
                "answer": problem.expected_answer,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    source_contract_sha256 = "a" * 64
    source_run = tmp_path / "source-run.jsonl"
    _write_jsonl(
        source_run,
        [
            {
                "record_type": "run_header",
                "run_contract_sha256": source_contract_sha256,
                "settings": {"builder_mode": BUILDER_MODE_ANCHORED_RANGE_LISTS},
                "problem_source": {
                    "problem_ids": [problem.problem_id],
                    "dataset_sha256": file_sha256(dataset),
                    "projection_sha256": problem_source_sha256((problem,)),
                    "evaluation_binding_sha256": evaluation_binding_sha256((problem,)),
                },
                "contracts": {
                    "state": ACTION_CENSUS_STATE_VERSION,
                    "action": ACTION_CENSUS_ACTION_VERSION,
                    "reward": ACTION_CENSUS_REWARD_VERSION,
                },
            },
            {
                "record_type": "census_state",
                "problem_index": 0,
                "problem_id": problem.problem_id,
                "builder_usable": True,
                "node_text_complete": True,
                "node_text_distinct": True,
                "state_sha256": state.sha256,
                "observation": state.observation(),
            },
        ],
    )
    source_diagnostic = tmp_path / "source-diagnostic.jsonl"
    _write_jsonl(
        source_diagnostic,
        [
            {
                "record_type": "diagnostic_header",
                "run_contract_sha256": source_contract_sha256,
            }
        ],
    )

    prepared = prepare_action_lens_source(
        source_run_path=source_run,
        source_diagnostic_path=source_diagnostic,
        expected_source_run_sha256=file_sha256(source_run),
        expected_source_diagnostic_sha256=file_sha256(source_diagnostic),
        expected_source_run_contract_sha256=source_contract_sha256,
        dataset_path=dataset,
        source_case_indices=(0,),
    )
    assert prepared.cases[0].state == state
    assert state_from_observation(state.observation()) == state

    local = target_local_observation(state, "node_2")
    assert local == {
        "schema_version": ACTION_LENS_LOCAL_VIEW_VERSION,
        "problem_id": problem.problem_id,
        "problem": problem.problem,
        "selected_action": {"verb": "revise", "target": "node_2"},
        "selected_node": {
            "id": "node_2",
            "kind": "semantic_component",
            "title": "recompute_local [S002]",
            "content": "Add one and two.",
        },
    }
    local_text = request_for_arm(prepared.cases[0], "local_node_2").user
    assert canonical_json_bytes(local).decode("ascii") in local_text
    assert "recheck_model [S001]" not in local_text
    assert "check_finalization [S003]" not in local_text
    assert state.draft.to_record()["reasoning"] not in local_text
    assert_request_isolation(prepared.cases[0])

    leaked = state.observation()
    leaked["expected_answer"] = "3"
    with pytest.raises(ValueError, match="does not replay exactly"):
        state_from_observation(leaked)

    dataset.write_text(
        json.dumps(
            {
                "id": problem.problem_id,
                "problem": problem.problem,
                "family": problem.family,
                "answer": "4",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="dataset or hidden binding differs"):
        prepare_action_lens_source(
            source_run_path=source_run,
            source_diagnostic_path=source_diagnostic,
            expected_source_run_sha256=file_sha256(source_run),
            expected_source_diagnostic_sha256=file_sha256(source_diagnostic),
            expected_source_run_contract_sha256=source_contract_sha256,
            dataset_path=dataset,
            source_case_indices=(0,),
        )


def test_williams_schedule_balances_positions_and_ordered_neighbors() -> None:
    assert len(FACTORIAL_ARMS) == 9
    assert len(WILLIAMS_ROWS) == 18
    assert len(set(WILLIAMS_ROWS)) == 18
    assert all(set(row) == set(FACTORIAL_ARMS) for row in WILLIAMS_ROWS)

    for arm in FACTORIAL_ARMS:
        positions = Counter(row.index(arm) for row in WILLIAMS_ROWS)
        assert positions == {position: 2 for position in range(len(FACTORIAL_ARMS))}

    ordered_neighbors = Counter(
        (left, right) for row in WILLIAMS_ROWS for left, right in zip(row, row[1:])
    )
    assert ordered_neighbors == Counter(
        {
            (left, right): 2
            for left in FACTORIAL_ARMS
            for right in FACTORIAL_ARMS
            if left != right
        }
    )
    assert tuple(arm_order(0, repetition) for repetition in range(4)) == (
        WILLIAMS_ROWS[0],
        WILLIAMS_ROWS[1],
        WILLIAMS_ROWS[2],
        WILLIAMS_ROWS[3],
    )


def test_analysis_retains_zero_reward_failures_in_every_denominator() -> None:
    result = {arm: (0, 0, 0, 0) for arm in FACTORIAL_ARMS}
    result[DIRECT_ARM] = (1, 0, 0, 0)
    analysis = analyze_action_lens((result,), _contract())

    assert analysis["accuracy_all"][DIRECT_ARM] == pytest.approx(0.25)
    assert analysis["accuracy_discovery"][DIRECT_ARM] == pytest.approx(0.5)
    assert analysis["accuracy_evaluation"][DIRECT_ARM] == 0
    assert analysis["primary_child_lens"]["paired_cells"] == {
        "both": 0,
        "left_only": 0,
        "right_only": 0,
        "neither": 6,
    }
    assert analysis["all_required_gates_passed"] is False
    assert analysis["behavior_cloning_eligible"] is False


def test_pilot_header_binds_the_registered_analysis_protocol() -> None:
    prepared = _prepared()
    pilot = _contract(prepared, stage="pilot")
    integration = _contract(prepared)
    header = pilot.header()

    assert header["contracts"]["analysis"] == ACTION_LENS_PILOT_ANALYSIS_VERSION
    assert header["settings"]["pilot_analysis"] == {
        "independent_unit": "source_state",
        "child_targets": list(CHILD_TARGETS),
        "global_target_tie_order": list(CHILD_TARGETS),
        "state_target_tie_fallback": "global_fixed_child",
        "bootstrap_samples": PILOT_BOOTSTRAP_SAMPLES,
        "bootstrap_seeds": list(PILOT_BOOTSTRAP_SEEDS),
        "permutation_samples": PILOT_PERMUTATION_SAMPLES,
        "permutation_seed": PILOT_PERMUTATION_SEED,
    }
    assert header["run_contract_sha256"] == pilot.sha256
    assert pilot.sha256 != integration.sha256
    assert "pilot_analysis" not in integration.header()["settings"]


def _aligned_pilot_results() -> tuple[dict[str, tuple[int, ...]], ...]:
    preferred_targets = (
        "node_1",
        "node_2",
        "node_3",
        "node_1",
        "node_2",
        "node_3",
        None,
    )
    results: list[dict[str, tuple[int, ...]]] = []
    for preferred_target in preferred_targets:
        result = {arm: (0, 0, 0, 0) for arm in FACTORIAL_ARMS}
        result["local_root"] = (1, 1, 0, 0)
        if preferred_target is not None:
            result[f"local_{preferred_target}"] = (1, 1, 1, 1)
        else:
            result["local_node_2"] = (0, 0, 1, 1)
        results.append(result)
    return tuple(results)


def test_pilot_selection_uses_discovery_only_and_keeps_screens_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_case_count(7)
    contract = _contract(prepared, stage="pilot")
    results = _aligned_pilot_results()
    families = tuple(case.problem.family for case in prepared.cases)
    monkeypatch.setattr(action_lens_factorial_module, "PILOT_BOOTSTRAP_SAMPLES", 400)
    monkeypatch.setattr(
        action_lens_factorial_module,
        "PILOT_PERMUTATION_SAMPLES",
        1_000,
    )

    analysis = analyze_action_lens(results, contract, families=families)
    fixed = analysis["q2_fixed_child_policy"]
    repeatability = analysis["q3_reward_probed_target_repeatability"]

    assert analysis["accuracy_discovery"]["local_root"] == 1
    assert fixed["selected_target"] == "node_1"
    assert fixed["selected_target"] in CHILD_TARGETS
    assert (
        analysis["accuracy_evaluation"]["local_node_2"]
        > analysis["accuracy_evaluation"]["local_node_1"]
    )
    assert repeatability["global_fixed_target"] == "node_1"
    assert repeatability["state_selected_targets"][-1] == "node_1"
    assert repeatability["discovery_tie_fallback_count"] == 1

    changed_nonselected_evaluation = tuple(
        {
            **result,
            "local_node_2": (
                result["local_node_2"][0],
                result["local_node_2"][1],
                0,
                0,
            ),
        }
        for result in results
    )
    changed = analyze_action_lens(
        changed_nonselected_evaluation,
        contract,
        families=families,
    )
    assert changed["q2_fixed_child_policy"] == fixed

    screens = analysis["screen_results"]
    assert screens["q1_child_lens"] is True
    assert screens["q2_fixed_child_vs_direct"] is False
    assert screens["q2_tree_specific_vs_root"] is False
    assert screens["q3_reward_probed_repeatability"] is True
    assert (
        screens["q1_child_lens"]
        == analysis["q1_child_local_vs_same_target_full"]["screen"]["passed"]
    )
    assert screens["q2_fixed_child_vs_direct"] == fixed["direct_screen_passed"]
    assert (
        screens["q3_reward_probed_repeatability"]
        == repeatability["repeatability_screen_passed"]
    )


def test_cross_split_interaction_is_seeded_and_detects_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        action_lens_factorial_module,
        "PILOT_PERMUTATION_SAMPLES",
        1_000,
    )
    aligned = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    first = action_lens_factorial_module._interaction_repeatability(
        aligned,
        aligned,
    )
    second = action_lens_factorial_module._interaction_repeatability(
        aligned,
        aligned,
    )

    assert first == second
    assert first["statistic"] > 0
    assert first["centered_correlation"] == pytest.approx(1.0)
    assert first["positive"] is True
    assert first["permutation_seed"] == PILOT_PERMUTATION_SEED
    assert first["permutation_samples"] == 1_000


def _scripted_complete_json(
    self: OpenAICompatibleLLMClient,
    system: str,
    user: str,
    schema: Mapping[str, Any],
    *,
    temperature: float = 1.0,
    max_tokens: int = 512,
    seed: int | None = None,
    schema_validation_mode: str,
) -> dict[str, Any]:
    del schema_validation_mode
    call_number = int(getattr(self, "_action_lens_test_calls", 0))
    setattr(self, "_action_lens_test_calls", call_number + 1)
    call_id = f"action-lens-test-{call_number}"
    request_body = build_openai_compatible_json_body(
        model=self.model,
        system=system,
        user=user,
        schema=schema,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        guided_json=self.guided_json,
    )
    request_body_json = canonical_json_bytes(request_body).decode("ascii")
    observer = self.attempt_observer
    assert observer is not None

    local_node_3 = (
        ACTION_LENS_LOCAL_VIEW_VERSION in user and "Selected target id: node_3" in user
    )
    if local_node_3 and seed == 100:
        observer(
            OpenAICompatibleAttempt(
                call_id=call_id,
                attempt_index=0,
                parse_retry_index=0,
                backoff_retry_index=0,
                effective_seed=seed,
                request_body_json=request_body_json,
                raw_response_body=None,
                raw_content=None,
                raw_usage_json=None,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                outcome="transport_error",
                disposition="raise",
                http_status=None,
                error_type="LLMResponseError",
                error_message="scripted transport failure",
            )
        )
        raise LLMResponseError("scripted transport failure")

    empty_answer = (
        ACTION_LENS_LOCAL_VIEW_VERSION not in user
        and "Selected target id: node_2" in user
    )
    payload: dict[str, Any] = {
        "reasoning": "scripted empty" if empty_answer else "One plus two is three.",
        "answer": "" if empty_answer else "3",
    }
    raw_content = canonical_json_bytes(payload).decode("ascii")
    raw_usage = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    raw_response = canonical_json_bytes(
        {
            "model": self.model,
            "choices": [{"message": {"content": raw_content}}],
            "usage": raw_usage,
        }
    ).decode("ascii")
    self.generate_calls += 1
    self.prompt_tokens += 10
    self.generated_tokens += 5
    observer(
        OpenAICompatibleAttempt(
            call_id=call_id,
            attempt_index=0,
            parse_retry_index=0,
            backoff_retry_index=0,
            effective_seed=seed,
            request_body_json=request_body_json,
            raw_response_body=raw_response,
            raw_content=raw_content,
            raw_usage_json=canonical_json_bytes(raw_usage).decode("ascii"),
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            outcome="accepted",
            disposition="return",
            http_status=200,
            error_type=None,
            error_message=None,
        )
    )
    return payload


def test_mocked_collection_and_audit_keep_generation_failures_as_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    contract = _contract(prepared)
    identity = contract.worker
    worker = OpenAICompatibleLLMClient(
        base_url=identity.base_url,
        model=identity.model,
        timeout_s=identity.timeout_s,
        max_retries=identity.parse_retries,
        max_backoff_retries=identity.backoff_retries,
    )
    monkeypatch.setattr(
        OpenAICompatibleLLMClient,
        "complete_json",
        _scripted_complete_json,
    )

    def deterministic_terminal_reward(
        candidate: Candidate,
        problem: CensusProblem,
    ) -> int:
        return int(
            candidate.usable
            and candidate.answer is not None
            and candidate.answer == problem.expected_answer
        )

    monkeypatch.setattr(
        action_lens_factorial_module,
        "terminal_reward",
        deterministic_terminal_reward,
    )
    run_path = tmp_path / "run.jsonl"
    diagnostic_path = tmp_path / "diagnostic.jsonl"

    collected = collect_action_lens_run(
        run_path=run_path,
        diagnostic_path=diagnostic_path,
        contract=contract,
        prepared=prepared,
        worker_client=worker,
    )
    assert collected.logical_call_count == 36
    assert collected.transition_count == 36

    records = [json.loads(line) for line in run_path.read_text().splitlines()]
    transitions = [
        record for record in records if record["record_type"] == "lens_transition"
    ]
    assert len(transitions) == 36
    assert {(record["arm"], record["repetition"]) for record in transitions} == {
        (arm, repetition) for arm in FACTORIAL_ARMS for repetition in range(4)
    }
    for repetition in range(4):
        assert {
            record["seed"]
            for record in transitions
            if record["repetition"] == repetition
        } == {contract.worker_seed + repetition}

    failure = next(
        record
        for record in transitions
        if record["arm"] == "local_node_3" and record["repetition"] == 0
    )
    assert failure["generation_error"] is True
    assert failure["candidate"]["usable"] is False
    assert failure["reward"] == 0
    empty_outputs = [record for record in transitions if record["arm"] == "full_node_2"]
    assert len(empty_outputs) == 4
    assert all(record["generation_error"] is False for record in empty_outputs)
    assert all(record["candidate"]["usable"] is False for record in empty_outputs)
    assert all(record["reward"] == 0 for record in empty_outputs)

    audit = audit_action_lens_artifacts(
        run_path=run_path,
        diagnostic_path=diagnostic_path,
        contract=contract,
        prepared=prepared,
    )
    assert audit["logical_call_count"] == 36
    assert audit["reward_replay_passed"] is True
    assert audit["prompt_isolation_passed"] is True
    assert audit["first_attempt_accepted_counts"]["local_node_3"] == 3
    assert audit["attempt_outcome_counts"]["local_node_3"] == {
        "accepted:return": 3,
        "transport_error:raise": 1,
    }
    assert audit["usage_by_arm"]["full_node_2"] == {
        "generate_calls": 4,
        "prompt_tokens": 40,
        "generated_tokens": 20,
    }
    assert audit["usage_by_arm"]["local_node_3"] == {
        "generate_calls": 3,
        "prompt_tokens": 30,
        "generated_tokens": 15,
    }

    diagnostic_records = [
        json.loads(line) for line in diagnostic_path.read_text().splitlines()
    ]
    tamper_index = 0

    def reject_tamper(
        run_records: list[dict[str, Any]],
        diagnostics: list[dict[str, Any]],
        match: str,
    ) -> None:
        nonlocal tamper_index
        tamper_index += 1
        tampered_run = tmp_path / f"tampered-{tamper_index}.jsonl"
        tampered_diagnostic = tmp_path / f"tampered-{tamper_index}.diagnostics.jsonl"
        _write_jsonl(tampered_run, run_records)
        _write_jsonl(tampered_diagnostic, diagnostics)
        with pytest.raises(ValueError, match=match):
            audit_action_lens_artifacts(
                run_path=tampered_run,
                diagnostic_path=tampered_diagnostic,
                contract=contract,
                prepared=prepared,
            )

    reordered_run = deepcopy(records)
    reordered_diagnostics = deepcopy(diagnostic_records)
    reordered_run[2], reordered_run[3] = reordered_run[3], reordered_run[2]
    reordered_diagnostics[1], reordered_diagnostics[2] = (
        reordered_diagnostics[2],
        reordered_diagnostics[1],
    )
    reordered_diagnostics[1]["sequence"] = 0
    reordered_diagnostics[2]["sequence"] = 1
    reject_tamper(reordered_run, reordered_diagnostics, "run record order differs")

    changed_response = deepcopy(diagnostic_records)
    accepted_attempt = next(
        record["attempt"]
        for record in changed_response[1:]
        if record["attempt"]["outcome"] == "accepted"
    )
    response_body = json.loads(accepted_attempt["raw_response_body"])
    response_body["choices"][0]["message"]["content"] = canonical_json_bytes(
        {"reasoning": "tampered", "answer": "4"}
    ).decode("ascii")
    accepted_attempt["raw_response_body"] = canonical_json_bytes(response_body).decode(
        "ascii"
    )
    reject_tamper(records, changed_response, "provider projection differs")

    extra_diagnostic = deepcopy(diagnostic_records)
    extra_diagnostic.append({"record_type": "unexpected"})
    reject_tamper(records, extra_diagnostic, "diagnostic record type differs")

    changed_error_type = deepcopy(records)
    successful_transition = next(
        record
        for record in changed_error_type
        if record.get("record_type") == "lens_transition"
        and record["generation_error"] is False
    )
    successful_transition["error_type"] = "LLMResponseError"
    reject_tamper(changed_error_type, diagnostic_records, "transition replay differs")

    extra_transition_field = deepcopy(records)
    transition_with_extra = next(
        record
        for record in extra_transition_field
        if record.get("record_type") == "lens_transition"
    )
    transition_with_extra["unexpected"] = True
    reject_tamper(
        extra_transition_field,
        diagnostic_records,
        "transition replay differs",
    )

    changed_diagnostic_schema = deepcopy(diagnostic_records)
    changed_diagnostic_schema[1]["schema_version"] = "wrong"
    reject_tamper(
        records,
        changed_diagnostic_schema,
        "diagnostic attempt is invalid",
    )

    changed_summary = deepcopy(records)
    case_summary = next(
        record
        for record in changed_summary
        if record.get("record_type") == "case_summary"
    )
    case_summary["schema_version"] = "wrong"
    reject_tamper(changed_summary, diagnostic_records, "case summary differs")
