import json
from dataclasses import FrozenInstanceError

import pytest

from verifier_anchored_rl.reasoning_v2.casework_source import (
    CASEWORK_PROBLEM_ADAPTER_IMPLEMENTATION,
    CASEWORK_PROBLEM_SOURCE_CONTRACT,
    CaseworkProblemSource,
    CaseworkSourceError,
    project_casework_problem,
)
from verifier_anchored_rl.reasoning_v2.model import Goal, State
from verifier_anchored_rl.reasoning_v2.replay import InitialStateResolver


def row(*, problem_id="cw-test-1", answer="5", solution_set=None):
    return {
        "id": problem_id,
        "problem": "Solve for all real x: |x - 5| = 0. Report the largest.",
        "answer": answer,
        "solution_set": ["5"] if solution_set is None else solution_set,
        "symbolic": "Eq(Abs(x - 5), 0)",
        "question_type": "largest_solution",
        "family": "f1a",
        "n_cases": 2,
        "difficulty": {"tier": 1, "params": {"unused": 17}},
    }


def write_jsonl(path, records):
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
        newline="",
    )


def test_source_digest_and_state_ignore_answer_bearing_fields(tmp_path):
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    write_jsonl(first_path, [row(answer="5", solution_set=["5"])])
    write_jsonl(second_path, [row(answer="forged", solution_set=["999"])])

    first = CaseworkProblemSource.from_jsonl(first_path)
    second = CaseworkProblemSource.from_jsonl(second_path)

    assert first.problem_source_sha256 == second.problem_source_sha256
    assert first("cw-test-1") == second("cw-test-1")
    approved = first.approved_problem_record("cw-test-1")
    assert "answer" not in approved
    assert "solution_set" not in approved
    assert "difficulty" not in approved


def test_source_is_an_exact_initial_state_resolver(tmp_path):
    path = tmp_path / "problems.jsonl"
    write_jsonl(path, [row()])

    source = CaseworkProblemSource.from_jsonl(path)
    state = source("cw-test-1")

    assert isinstance(source, InitialStateResolver)
    assert source.problem_source_contract == CASEWORK_PROBLEM_SOURCE_CONTRACT
    assert (
        source.problem_adapter_implementation
        == CASEWORK_PROBLEM_ADAPTER_IMPLEMENTATION
    )
    assert source.problem_adapter_implementation == (
        f"{CaseworkProblemSource.__module__}."
        f"{CaseworkProblemSource.__qualname__}"
    )
    assert isinstance(state, State)
    assert state.turn == 0
    assert state.previous_outcome is None
    assert len(state.graph.nodes) == 1
    assert isinstance(state.graph.nodes[0], Goal)
    assert state.graph.nodes[0].obligation.equation == "Eq(Abs(x - 5), 0)"


def test_source_rejects_duplicate_problem_ids(tmp_path):
    path = tmp_path / "duplicates.jsonl"
    write_jsonl(path, [row(), row(answer="different")])

    with pytest.raises(CaseworkSourceError, match="ids must be unique"):
        CaseworkProblemSource.from_jsonl(path)


def test_source_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "duplicate-key.jsonl"
    path.write_bytes(
        b'{"id":"first","id":"second","problem":"p",'
        b'"family":"f1a","question_type":"largest_solution",'
        b'"symbolic":"Eq(x, 1)"}\n'
    )

    with pytest.raises(CaseworkSourceError, match="duplicate JSON key"):
        CaseworkProblemSource.from_jsonl(path)


def test_projection_uses_only_f1b_domain_metadata():
    value = row(problem_id="cw-domain")
    value.update({
        "family": "f1b",
        "question_type": "find_x",
        "difficulty": {
            "params": {
                "direction": "lt",
                "d": 12,
                "answer_x": -1,
            }
        },
    })

    projected = project_casework_problem(value)

    assert projected["domain_direction"] == "lt"
    assert projected["domain_boundary"] == "12"
    assert "answer_x" not in projected


def test_slice_has_its_own_ordered_source_digest(tmp_path):
    path = tmp_path / "slice.jsonl"
    write_jsonl(path, [row(problem_id="a"), row(problem_id="b")])
    source = CaseworkProblemSource.from_jsonl(path)

    first = source.select_slice(0, 1)
    second = source.select_slice(1, 1)

    assert first.problem_ids == ("a",)
    assert second.problem_ids == ("b",)
    assert first.problem_source_sha256 != second.problem_source_sha256
    with pytest.raises(FrozenInstanceError):
        first.problem_source_contract = "changed"


def test_jsonl_requires_a_final_lf_newline(tmp_path):
    path = tmp_path / "unterminated.jsonl"
    path.write_text(json.dumps(row()), encoding="utf-8", newline="")

    with pytest.raises(CaseworkSourceError, match="trailing newline"):
        CaseworkProblemSource.from_jsonl(path)
