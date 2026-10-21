from __future__ import annotations

import hashlib
import json

import pytest

from verifier_anchored_rl.workspace_v1.target_value import (
    BranchGoal,
    TargetProblem,
)
from verifier_anchored_rl.workspace_v1.tolerant_replay import (
    INITIAL_TARGETS,
    TolerantReplayCase,
    analyze_tolerant_replay,
    canonicalize_payload,
    prepare_tolerant_replay,
)


@pytest.mark.parametrize(
    ("raw", "answer", "disposition", "fields"),
    [
        (
            '{"answer":4,"x":9,"reasoning":"work"}',
            "4",
            "answer_key",
            ("answer",),
        ),
        ('{"x":-3,"reasoning":"work"}', "-3", "unique_scalar_field", ("x",)),
        (
            '{"solution":2,"value":2}',
            "2",
            "agreeing_scalar_fields",
            ("solution", "value"),
        ),
        (
            'prefix {"sum_of_solutions":7} suffix',
            "7",
            "unique_scalar_field",
            ("sum_of_solutions",),
        ),
        ('{"solution":null}', None, "unique_scalar_field", ("solution",)),
    ],
)
def test_canonicalizer_accepts_label_blind_scalar_fields(
    raw: str,
    answer: str | None,
    disposition: str,
    fields: tuple[str, ...],
) -> None:
    result = canonicalize_payload(raw)
    assert result.candidate.usable
    assert result.candidate.answer == answer
    assert result.disposition == disposition
    assert result.selected_fields == fields


@pytest.mark.parametrize(
    ("raw", "disposition"),
    [
        ('{"x":1,"solution":2}', "conflicting_values"),
        ('{"x":true}', "boolean_value"),
        ('{"x":[1]}', "nonscalar_value"),
        ('{"reasoning":"only"}', "no_value_field"),
        ("not json", "no_json_object"),
    ],
)
def test_canonicalizer_rejects_only_syntactic_ambiguity(
    raw: str,
    disposition: str,
) -> None:
    result = canonicalize_payload(raw)
    assert not result.candidate.usable
    assert result.disposition == disposition


def _problem(index: int) -> TargetProblem:
    return TargetProblem(
        problem_id=f"p-{index}",
        problem=f"Solve problem {index}.",
        family=("f1a", "f1b", "f2a", "f2b")[index % 4],
        question_type="sum_of_solutions",
        branches=(
            BranchGoal(0, "Solve branch zero.", 1),
            BranchGoal(1, "Solve branch one.", 2),
        ),
        expected_answer="3",
    )


def _write_jsonl(path, records: list[dict[str, object]]) -> str:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="ascii",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preparation_selects_attempt_zero_not_corrective_retry(tmp_path) -> None:
    problem = _problem(0)
    contract_sha = "a" * 64
    run_records: list[dict[str, object]] = [{
        "record_type": "run_header",
        "run_contract_sha256": contract_sha,
    }]
    diagnostic_records: list[dict[str, object]] = []
    for index, target in enumerate(INITIAL_TARGETS):
        call_id = f"call-{index}"
        run_records.append({
            "record_type": "logical_worker_call",
            "problem_id": problem.problem_id,
            "phase": "initial",
            "target": target,
            "repetition": None,
            "worker_call_id": call_id,
            "generation_error": True,
        })
        for attempt_index, value in ((1, 99), (0, index + 1)):
            diagnostic_records.append({
                "record_type": "http_attempt",
                "role": "worker",
                "attempt": {
                    "call_id": call_id,
                    "attempt_index": attempt_index,
                    "raw_content": json.dumps({"x": value}),
                },
            })
    run_path = tmp_path / "run.jsonl"
    diagnostic_path = tmp_path / "diagnostic.jsonl"
    run_sha = _write_jsonl(run_path, run_records)
    diagnostic_sha = _write_jsonl(diagnostic_path, diagnostic_records)
    prepared = prepare_tolerant_replay(
        e43_run_path=run_path,
        e43_diagnostic_path=diagnostic_path,
        expected_e43_run_sha256=run_sha,
        expected_e43_diagnostic_sha256=diagnostic_sha,
        expected_e43_run_contract_sha256=contract_sha,
        problems=(problem,),
    )
    assert canonicalize_payload(
        prepared.attempts[(problem.problem_id, "root")].raw_content
    ).candidate.answer == "1"


def test_analysis_gates_require_usable_and_semantic_regime() -> None:
    cases = tuple(
        TolerantReplayCase(
            problem_id=f"p-{index}",
            family=("f1a", "f1b", "f2a", "f2b")[index % 4],
            strict_accepted={target: False for target in INITIAL_TARGETS},
            usable={target: True for target in INITIAL_TARGETS},
            direct_reward=int(index < 30),
            structured_reward=int(index < 24),
            branch_correct=(index < 20, index < 22),
        )
        for index in range(60)
    )
    analysis = analyze_tolerant_replay(cases)
    assert analysis["tolerant_usable_counts"] == {
        "root": 60,
        "branch_0": 60,
        "branch_1": 60,
    }
    assert analysis["terminal_solved"] == {
        "direct_initial": 30,
        "structured_submit": 24,
    }
    assert analysis["gates"] == {
        "integration_passed": True,
        "regime_passed": True,
        "all_passed": True,
    }
