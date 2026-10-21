from __future__ import annotations

import hashlib
import json

from verifier_anchored_rl.workspace_v1.target_prompt_calibration import (
    CALL_KEYS,
    CalibrationCase,
    analyze_prompt_calibration,
    prepare_e43_baseline,
)
from verifier_anchored_rl.workspace_v1.target_value import (
    BranchGoal,
    Candidate,
    TargetAction,
    TargetProblem,
    branch_request,
    two_key_branch_request,
    two_key_root_request,
)


def _problem(index: int) -> TargetProblem:
    return TargetProblem(
        problem_id=f"p-{index}",
        problem=f"Solve visible problem {index}.",
        family=("f1a", "f1b", "f2a", "f2b")[index % 4],
        question_type="sum_of_solutions",
        branches=(
            BranchGoal(0, "Solve branch zero.", 1),
            BranchGoal(1, "Solve branch one.", 2),
        ),
        expected_answer="3",
    )


def test_two_key_suffix_is_the_only_request_change_and_comes_last() -> None:
    problem = _problem(0)
    current = Candidate(reasoning="visible", answer="7")
    original = branch_request(problem, 0, current=current)
    repaired = two_key_branch_request(problem, 0, current=current)
    assert repaired.system == original.system
    assert repaired.schema == original.schema
    assert repaired.user.startswith(original.user)
    assert repaired.user.endswith(
        "For an invalid selected branch, use JSON null as the answer."
    )
    assert 'exactly {"reasoning":"brief derivation","answer":0}' in (
        repaired.user
    )
    assert "Output contract:" in two_key_root_request(problem).user


def test_baseline_selection_extracts_exact_six_calls(tmp_path) -> None:
    problems = tuple(_problem(index) for index in range(12))
    contract_sha = "a" * 64
    records: list[dict[str, object]] = [{
        "record_type": "run_header",
        "run_contract_sha256": contract_sha,
    }]
    for problem in problems:
        for key in CALL_KEYS:
            phase, target = key.split("_", maxsplit=1)
            records.append({
                "record_type": "logical_worker_call",
                "problem_id": problem.problem_id,
                "phase": phase,
                "target": target,
                "repetition": None if phase == "initial" else 0,
                "generation_error": key.startswith("initial_branch"),
            })
    path = tmp_path / "baseline.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="ascii",
    )
    run_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    baseline = prepare_e43_baseline(
        run_path=path,
        expected_run_sha256=run_sha,
        expected_run_contract_sha256=contract_sha,
        problems=problems,
    )
    assert baseline.problem_ids == tuple(problem.problem_id for problem in problems)
    assert all(
        not baseline.acceptance[problem.problem_id]["initial_branch_0"]
        for problem in problems
    )
    assert len(baseline.selection_sha256) == 64


def test_calibration_gates_pass_only_with_schema_and_regime() -> None:
    cases = tuple(
        CalibrationCase(
            problem_id=f"p-{index}",
            accepted={key: True for key in CALL_KEYS},
            baseline_accepted={key: not key.startswith("initial_branch") for key in CALL_KEYS},
            direct_reward=0,
            structured_submit_reward=int(index < 6),
            retry_rewards={
                TargetAction.RETRY_ROOT: 0,
                TargetAction.RETRY_BRANCH_0: int(index < 4),
                TargetAction.RETRY_BRANCH_1: int(index < 5),
            },
            initial_branch_correct=(True, True),
        )
        for index in range(12)
    )
    analysis = analyze_prompt_calibration(cases)
    assert analysis["accepted_counts"]["initial_branch_0"] == 12
    assert analysis["matched_initial_branch_acceptance_gain"] == 1.0
    assert analysis["terminal_solved"]["structured_submit"] == 6
    assert analysis["gates"] == {
        "integration_passed": True,
        "regime_passed": True,
        "all_passed": True,
    }


def test_calibration_regime_rejects_zero_structured_solutions() -> None:
    cases = tuple(
        CalibrationCase(
            problem_id=f"p-{index}",
            accepted={key: True for key in CALL_KEYS},
            baseline_accepted={key: False for key in CALL_KEYS},
            direct_reward=0,
            structured_submit_reward=0,
            retry_rewards={
                TargetAction.RETRY_ROOT: 0,
                TargetAction.RETRY_BRANCH_0: 0,
                TargetAction.RETRY_BRANCH_1: 0,
            },
            initial_branch_correct=(False, False),
        )
        for index in range(12)
    )
    analysis = analyze_prompt_calibration(cases)
    assert analysis["gates"]["integration_passed"]
    assert not analysis["gates"]["regime_passed"]
    assert not analysis["gates"]["all_passed"]
