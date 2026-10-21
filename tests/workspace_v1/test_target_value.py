from __future__ import annotations

import json

import pytest

from verifier_anchored_rl.workspace_v1.experiment import ModelIdentity
from verifier_anchored_rl.workspace_v1.target_value import (
    ACTION_ORDER,
    Candidate,
    TargetAction,
    branch_request,
    candidate_from_payload,
    candidate_semantically_correct,
    reduce_branch_answers,
    root_request,
    target_problem_from_record,
    target_problem_source_record,
    target_state_record,
)
from verifier_anchored_rl.workspace_v1.target_value_experiment import (
    CaseEvaluation,
    TargetRunContract,
    analyze_target_value_cases,
)


def _record(
    family: str,
    question_type: str,
    answer: str,
    params: dict[str, object],
) -> dict[str, object]:
    return {
        "id": f"problem-{family}",
        "problem": f"Visible {family} problem.",
        "answer": answer,
        "family": family,
        "question_type": question_type,
        "difficulty": {"params": params},
    }


@pytest.mark.parametrize(
    ("record", "branch_answers"),
    [
        (
            _record(
                "f1a",
                "sum_of_solutions",
                "6",
                {"a": 2, "b": -6, "c": 4, "m": 3, "k": 2},
            ),
            ("5", "1"),
        ),
        (
            _record(
                "f1b",
                "find_x",
                "5",
                {
                    "a": 2,
                    "b": -6,
                    "c": 4,
                    "m": 3,
                    "k": 2,
                    "d": 2,
                    "direction": "gt",
                },
            ),
            ("5", None),
        ),
        (
            _record(
                "f2a",
                "find_x",
                "3",
                {
                    "a1": 2,
                    "b1": 0,
                    "a2": 1,
                    "b2": 7,
                    "t": 0,
                    "v": 10,
                },
            ),
            (None, "3"),
        ),
        (
            _record(
                "f2b",
                "sum_of_solutions",
                "1",
                {
                    "a1": 2,
                    "b1": 14,
                    "a2": 1,
                    "b2": 7,
                    "t": 0,
                    "v": 10,
                },
            ),
            ("-2", "3"),
        ),
    ],
)
def test_generator_projection_reduces_hidden_branches_to_root(
    record: dict[str, object],
    branch_answers: tuple[str | None, str | None],
) -> None:
    problem = target_problem_from_record(record)
    assert tuple(
        None if branch.expected_answer is None else str(branch.expected_answer)
        for branch in problem.branches
    ) == branch_answers
    candidates = tuple(
        Candidate(reasoning="test", answer=answer)
        for answer in branch_answers
    )
    assert reduce_branch_answers(candidates, problem.question_type) == str(
        record["answer"]
    )


def test_wrong_algebra_is_accepted_as_a_candidate() -> None:
    candidate = candidate_from_payload({
        "reasoning": "I made an algebra mistake.",
        "answer": 999,
    })
    assert candidate.usable
    assert candidate.answer == "999"
    assert reduce_branch_answers(
        (candidate, Candidate(reasoning="", answer="1")),
        "sum_of_solutions",
    ) == "1000"


def test_dead_branch_null_is_distinct_from_generation_failure() -> None:
    dead = Candidate(reasoning="domain check", answer=None)
    failed = Candidate(
        reasoning="",
        answer=None,
        usable=False,
        error_type="LLMResponseError",
    )
    assert reduce_branch_answers(
        (dead, Candidate(reasoning="", answer="3")), "find_x"
    ) == "3"
    assert reduce_branch_answers(
        (failed, Candidate(reasoning="", answer="3")), "find_x"
    ) == ""


def test_visible_state_has_exact_verb_target_actions_and_no_labels() -> None:
    problem = target_problem_from_record(_record(
        "f1a",
        "sum_of_solutions",
        "6",
        {"a": 2, "b": -6, "c": 4, "m": 3, "k": 2},
    ))
    candidates = (
        Candidate(reasoning="positive", answer="5"),
        Candidate(reasoning="negative", answer="1"),
    )
    state = target_state_record(problem, candidates)
    assert [(action["verb"], action["target"]) for action in state["actions"]] == [
        ("submit", "root"),
        ("retry", "root"),
        ("retry", "branch_0"),
        ("retry", "branch_1"),
    ]
    published = json.dumps({
        "source": target_problem_source_record((problem,)),
        "state": state,
        "root_request": root_request(problem).to_record(),
        "branch_request": branch_request(problem, 0).to_record(),
    })
    assert "expected_answer" not in published
    assert "solution_set" not in published
    assert "difficulty" not in published
    assert "params" not in published
    assert '"family"' not in published


def test_retry_prompt_contains_only_visible_current_candidate() -> None:
    problem = target_problem_from_record(_record(
        "f1b",
        "find_x",
        "5",
        {
            "a": 2,
            "b": -6,
            "c": 4,
            "m": 3,
            "k": 2,
            "d": 2,
            "direction": "gt",
        },
    ))
    current = Candidate(reasoning="visible trace", answer="4")
    request = branch_request(problem, 0, current=current)
    assert "visible trace" in request.user
    assert '"answer": "4"' in request.user
    assert "expected_answer" not in request.user
    assert not candidate_semantically_correct(current, problem.branches[0])


def _contract(problem_ids: tuple[str, ...]) -> TargetRunContract:
    return TargetRunContract(
        preregistration_id="E43-test",
        dataset_sha256="0" * 64,
        problem_source_sha256="1" * 64,
        evaluation_binding_sha256="2" * 64,
        problem_ids=problem_ids,
        worker=ModelIdentity(
            provider="test",
            model="test/model",
            revision="revision",
            base_url="https://example.test/v1",
            guided_json=False,
            timeout_s=1.0,
            parse_retries=0,
            backoff_retries=0,
        ),
        worker_temperature=0.2,
        worker_max_tokens=256,
        base_seed=430_000,
    )


def test_analysis_uses_discovery_only_for_selection() -> None:
    case = CaseEvaluation(
        problem_id="p0",
        direct_initial_reward=0,
        direct_majority3_reward=1,
        action_rewards={
            TargetAction.SUBMIT: (0, 0, 0, 0),
            TargetAction.RETRY_ROOT: (0, 0, 0, 0),
            TargetAction.RETRY_BRANCH_0: (1, 1, 0, 0),
            TargetAction.RETRY_BRANCH_1: (0, 0, 1, 1),
        },
        initial_branch_correct=(False, True),
    )
    analysis = analyze_target_value_cases((case,), _contract(("p0",)))
    assert analysis["accuracies"]["crossfit_value"] == 0.0
    assert analysis["accuracies"]["realized_oracle"] == 1.0
    assert analysis["selection"]["crossfit_action_counts"] == {
        TargetAction.RETRY_BRANCH_0.value: 1
    }


def test_action_order_is_submit_then_root_then_branches() -> None:
    assert ACTION_ORDER == (
        TargetAction.SUBMIT,
        TargetAction.RETRY_ROOT,
        TargetAction.RETRY_BRANCH_0,
        TargetAction.RETRY_BRANCH_1,
    )
