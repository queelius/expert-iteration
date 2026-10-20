"""Fixed-graph target-selection environment for workspace experiments."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from ..verifier_adapter import check_answer
from .records import canonical_json_sha256


TARGET_STATE_VERSION = "workspace-target-state-v1"
TARGET_ACTION_VERSION = "workspace-target-action-v1"
TARGET_REDUCER_VERSION = "workspace-two-branch-reducer-v1"
TARGET_WORKER_PROMPT_VERSION = "workspace-target-worker-prompt-v1"
TARGET_ROOT_PROMPT_VERSION = "workspace-target-root-prompt-v1"
TARGET_SCHEMA_VERSION = "workspace-target-candidate-schema-v1"
TWO_KEY_PROMPT_VERSION = "workspace-target-two-key-suffix-v1"
TARGET_PROBLEM_SOURCE_VERSION = "workspace-target-problem-source-v1"
TARGET_EVALUATION_BINDING_VERSION = "workspace-target-hidden-binding-v1"


class TargetAction(str, Enum):
    """One additional-call action from a fixed initial workspace."""

    SUBMIT = "submit"
    RETRY_ROOT = "retry_root"
    RETRY_BRANCH_0 = "retry_branch_0"
    RETRY_BRANCH_1 = "retry_branch_1"


ACTION_ORDER = (
    TargetAction.SUBMIT,
    TargetAction.RETRY_ROOT,
    TargetAction.RETRY_BRANCH_0,
    TargetAction.RETRY_BRANCH_1,
)
RETRY_ACTIONS = ACTION_ORDER[1:]


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _integer_fraction(value: Fraction, name: str) -> int:
    if value.denominator != 1:
        raise ValueError(f"{name} must be an integer")
    return value.numerator


def _linear(coefficient: int, constant: int) -> str:
    if coefficient == 1:
        head = "x"
    elif coefficient == -1:
        head = "-x"
    else:
        head = f"{coefficient}x"
    if constant == 0:
        return head
    sign = "+" if constant > 0 else "-"
    return f"{head} {sign} {abs(constant)}"


@dataclass(frozen=True, slots=True)
class BranchGoal:
    """A visible branch task plus its evaluation-only scalar label."""

    index: int
    task: str
    expected_answer: int | None = field(repr=False)

    def __post_init__(self) -> None:
        if self.index not in {0, 1}:
            raise ValueError("branch index must be zero or one")
        _nonempty_text(self.task, "branch task")

    def source_record(self) -> dict[str, object]:
        return {"index": self.index, "task": self.task}


@dataclass(frozen=True, slots=True)
class TargetProblem:
    """A two-branch problem with labels kept outside visible state."""

    problem_id: str
    problem: str
    family: str
    question_type: str
    branches: tuple[BranchGoal, BranchGoal]
    expected_answer: str = field(repr=False)

    def __post_init__(self) -> None:
        _nonempty_text(self.problem_id, "problem id")
        _nonempty_text(self.problem, "problem")
        if self.family not in {"f1a", "f1b", "f2a", "f2b"}:
            raise ValueError("unsupported target-value family")
        if self.question_type not in {
            "sum_of_solutions",
            "largest_solution",
            "smallest_solution",
            "find_x",
        }:
            raise ValueError("unsupported question type")
        if tuple(branch.index for branch in self.branches) != (0, 1):
            raise ValueError("target problem requires ordered branches 0 and 1")
        if not isinstance(self.expected_answer, str):
            raise ValueError("expected answer must be text")

    def source_record(self) -> dict[str, object]:
        return {
            "id": self.problem_id,
            "problem": self.problem,
            "question_type": self.question_type,
            "branches": [branch.source_record() for branch in self.branches],
        }


def _absolute_branches(
    family: str,
    params: Mapping[str, object],
) -> tuple[BranchGoal, BranchGoal]:
    required = {"a", "b", "c", "m", "k"}
    if not required.issubset(params):
        raise ValueError("absolute-value parameters are incomplete")
    a = int(params["a"])
    b = int(params["b"])
    c = int(params["c"])
    m = int(params["m"])
    k = int(params["k"])
    positive = m + k
    negative = m - k
    condition = ""
    if family == "f1b":
        if "d" not in params or "direction" not in params:
            raise ValueError("conditioned absolute-value parameters are incomplete")
        d = int(params["d"])
        direction = params["direction"]
        if direction == "gt":
            condition = f" and enforce x > {d}"
            positive_expected = positive if positive > d else None
            negative_expected = negative if negative > d else None
        elif direction == "lt":
            condition = f" and enforce x < {d}"
            positive_expected = positive if positive < d else None
            negative_expected = negative if negative < d else None
        else:
            raise ValueError("unknown absolute-value condition")
    else:
        positive_expected = positive
        negative_expected = negative
    lhs = _linear(a, b)
    return (
        BranchGoal(
            0,
            (
                f"Solve the positive-sign branch {lhs} = {c}{condition}. "
                "Return its integer x, or null when the branch is invalid."
            ),
            positive_expected,
        ),
        BranchGoal(
            1,
            (
                f"Solve the negative-sign branch {lhs} = {-c}{condition}. "
                "Return its integer x, or null when the branch is invalid."
            ),
            negative_expected,
        ),
    )


def _piecewise_branches(
    params: Mapping[str, object],
) -> tuple[BranchGoal, BranchGoal]:
    required = {"a1", "b1", "a2", "b2", "t", "v"}
    if not required.issubset(params):
        raise ValueError("piecewise parameters are incomplete")
    a1 = int(params["a1"])
    b1 = int(params["b1"])
    a2 = int(params["a2"])
    b2 = int(params["b2"])
    t = int(params["t"])
    v = int(params["v"])
    lower = _integer_fraction(Fraction(v - b1, a1), "lower solution")
    upper = _integer_fraction(Fraction(v - b2, a2), "upper solution")
    lower_expected = lower if lower < t else None
    upper_expected = upper if upper >= t else None
    return (
        BranchGoal(
            0,
            (
                f"Solve the lower piece {_linear(a1, b1)} = {v} and enforce "
                f"x < {t}. Return its integer x, or null when the branch is "
                "invalid."
            ),
            lower_expected,
        ),
        BranchGoal(
            1,
            (
                f"Solve the upper piece {_linear(a2, b2)} = {v} and enforce "
                f"x >= {t}. Return its integer x, or null when the branch is "
                "invalid."
            ),
            upper_expected,
        ),
    )


def target_problem_from_record(record: Mapping[str, object]) -> TargetProblem:
    """Project one generator record into a fixed visible branch graph."""
    if not isinstance(record, Mapping):
        raise ValueError("problem row must be an object")
    try:
        problem_id = _nonempty_text(record["id"], "problem id")
        problem = _nonempty_text(record["problem"], "problem")
        family = _nonempty_text(record["family"], "family")
        question_type = _nonempty_text(
            record["question_type"], "question type"
        )
        expected_answer = str(record["answer"])
        difficulty = record["difficulty"]
    except KeyError as exc:
        raise ValueError(f"problem row is missing field {exc.args[0]!r}") from exc
    if not isinstance(difficulty, Mapping):
        raise ValueError("problem difficulty must be an object")
    params = difficulty.get("params")
    if not isinstance(params, Mapping):
        raise ValueError("problem parameters must be an object")
    if family in {"f1a", "f1b"}:
        branches = _absolute_branches(family, params)
    elif family in {"f2a", "f2b"}:
        branches = _piecewise_branches(params)
    else:
        raise ValueError("unsupported target-value family")
    target = TargetProblem(
        problem_id=problem_id,
        problem=problem,
        family=family,
        question_type=question_type,
        branches=branches,
        expected_answer=expected_answer,
    )
    reduced = reduce_branch_answers(
        tuple(
            Candidate(reasoning="", answer=(
                None if branch.expected_answer is None
                else str(branch.expected_answer)
            ))
            for branch in target.branches
        ),
        target.question_type,
    )
    if not check_answer(reduced, target.expected_answer).success:
        raise ValueError("derived branch labels do not reduce to the root label")
    return target


def _pairs_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value!r}")


def load_target_problems(path: str | Path) -> tuple[TargetProblem, ...]:
    """Load strict generator JSONL and derive two branches per problem."""
    rows: list[TargetProblem] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.endswith("\r\n"):
                raise ValueError(f"line {line_number} uses a CRLF newline")
            if not line.endswith("\n"):
                raise ValueError(f"line {line_number} lacks a trailing newline")
            text = line[:-1]
            if not text:
                raise ValueError(f"line {line_number} is blank")
            value = json.loads(
                text,
                object_pairs_hook=_pairs_object,
                parse_constant=_reject_constant,
            )
            if type(value) is not dict:
                raise ValueError(f"line {line_number} must be an object")
            rows.append(target_problem_from_record(value))
    if not rows:
        raise ValueError("target problem source must be nonempty")
    ids = tuple(row.problem_id for row in rows)
    if len(ids) != len(set(ids)):
        raise ValueError("target problem ids must be unique")
    return tuple(rows)


def target_problem_source_record(
    problems: tuple[TargetProblem, ...],
) -> dict[str, object]:
    if not problems:
        raise ValueError("target problem source must be nonempty")
    return {
        "schema_version": TARGET_PROBLEM_SOURCE_VERSION,
        "problems": [problem.source_record() for problem in problems],
    }


def target_problem_source_sha256(
    problems: tuple[TargetProblem, ...],
) -> str:
    return canonical_json_sha256(target_problem_source_record(problems))


def target_evaluation_binding_sha256(
    problems: tuple[TargetProblem, ...],
) -> str:
    private_record = {
        "schema_version": TARGET_EVALUATION_BINDING_VERSION,
        "labels": [
            {
                "id": problem.problem_id,
                "root": problem.expected_answer,
                "branches": [
                    branch.expected_answer for branch in problem.branches
                ],
            }
            for problem in problems
        ],
    }
    return canonical_json_sha256(private_record)


@dataclass(frozen=True, slots=True)
class Candidate:
    """A worker answer accepted at the JSON interface without algebra checks."""

    reasoning: str
    answer: str | None
    usable: bool = True
    error_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reasoning, str):
            raise ValueError("candidate reasoning must be text")
        if self.answer is not None and not isinstance(self.answer, str):
            raise ValueError("candidate answer must be text or null")
        if type(self.usable) is not bool:
            raise ValueError("candidate usable flag must be boolean")
        if self.usable and self.error_type is not None:
            raise ValueError("usable candidates cannot carry an error type")
        if not self.usable and not self.error_type:
            raise ValueError("unusable candidates require an error type")

    def to_record(self) -> dict[str, object]:
        return asdict(self)


def _normalize_answer(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is bool:
        raise ValueError("candidate answer cannot be boolean")
    if isinstance(value, str):
        return value.strip()
    if type(value) is int:
        return str(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("candidate answer must be finite")
        return str(value)
    raise ValueError("candidate answer must be text, number, or null")


def candidate_from_payload(payload: Mapping[str, object]) -> Candidate:
    """Accept a schema-valid worker payload without semantic rejection."""
    if not isinstance(payload, Mapping):
        raise ValueError("candidate payload must be an object")
    if "answer" not in payload:
        raise ValueError("candidate payload is missing answer")
    reasoning = payload.get("reasoning", "")
    if not isinstance(reasoning, str):
        raise ValueError("candidate reasoning must be text")
    return Candidate(
        reasoning=reasoning.strip(),
        answer=_normalize_answer(payload["answer"]),
    )


CANDIDATE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "answer": {"type": ["string", "number", "null"]},
    },
    "required": ["reasoning", "answer"],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    system: str
    user: str
    schema: Mapping[str, object]

    def to_record(self) -> dict[str, object]:
        return {
            "system": self.system,
            "user": self.user,
            "schema": dict(self.schema),
        }


_BRANCH_SYSTEM = (
    "You are a frozen math worker. Solve only the selected branch. Return a "
    "JSON object with concise reasoning and an answer. The answer must be an "
    "integer or JSON null when the selected branch is invalid."
)
_ROOT_SYSTEM = (
    "You are a frozen math worker. Solve the complete problem independently. "
    "Return a JSON object with concise reasoning and the requested scalar "
    "answer."
)


def branch_request(
    problem: TargetProblem,
    branch_index: int,
    *,
    current: Candidate | None = None,
) -> WorkerRequest:
    if branch_index not in {0, 1}:
        raise ValueError("branch index must be zero or one")
    branch = problem.branches[branch_index]
    parts = [
        f"Complete problem:\n{problem.problem}",
        f"Selected branch:\n{branch.task}",
    ]
    if current is not None:
        parts.append(
            "Current branch attempt (untrusted):\n"
            + json.dumps(current.to_record(), sort_keys=True)
        )
        parts.append(
            "Retry this branch independently and replace the current attempt."
        )
    return WorkerRequest(
        system=_BRANCH_SYSTEM,
        user="\n\n".join(parts),
        schema=CANDIDATE_SCHEMA,
    )


def root_request(problem: TargetProblem) -> WorkerRequest:
    return WorkerRequest(
        system=_ROOT_SYSTEM,
        user=f"Complete problem:\n{problem.problem}",
        schema=CANDIDATE_SCHEMA,
    )


_TWO_KEY_SUFFIX = (
    "Output contract:\n"
    'Return exactly {"reasoning":"brief derivation","answer":0} with no '
    "other keys. Replace 0 with the integer answer. For an invalid selected "
    "branch, use JSON null as the answer."
)


def with_two_key_output_contract(request: WorkerRequest) -> WorkerRequest:
    """Append the E43R1 output example after every other prompt element."""
    return WorkerRequest(
        system=request.system,
        user=f"{request.user}\n\n{_TWO_KEY_SUFFIX}",
        schema=request.schema,
    )


def two_key_branch_request(
    problem: TargetProblem,
    branch_index: int,
    *,
    current: Candidate | None = None,
) -> WorkerRequest:
    return with_two_key_output_contract(branch_request(
        problem, branch_index, current=current
    ))


def two_key_root_request(problem: TargetProblem) -> WorkerRequest:
    return with_two_key_output_contract(root_request(problem))


def parse_integer_answer(answer: str | None) -> int | None:
    """Parse the worker's scalar interface, with null as a dead branch."""
    if answer is None:
        return None
    text = answer.strip()
    if not text:
        raise ValueError("candidate answer is empty")
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("candidate answer is not a scalar number") from exc
    if not value.is_finite() or value != value.to_integral_value():
        raise ValueError("candidate answer is not a finite integer")
    return int(value)


def reduce_branch_answers(
    candidates: tuple[Candidate, Candidate],
    question_type: str,
) -> str:
    """Mechanically reduce visible branch answers into a root candidate."""
    if len(candidates) != 2:
        raise ValueError("the reducer requires exactly two candidates")
    if any(not candidate.usable for candidate in candidates):
        return ""
    try:
        values = [
            value
            for value in (
                parse_integer_answer(candidate.answer)
                for candidate in candidates
            )
            if value is not None
        ]
    except ValueError:
        return ""
    unique = sorted(set(values))
    if not unique:
        return ""
    if question_type == "sum_of_solutions":
        return str(sum(unique))
    if question_type == "largest_solution":
        return str(max(unique))
    if question_type == "smallest_solution":
        return str(min(unique))
    if question_type == "find_x" and len(unique) == 1:
        return str(unique[0])
    return ""


def terminal_reward(answer: str, problem: TargetProblem) -> int:
    """Return the only reward: the terminal symbolic verifier result."""
    return int(check_answer(answer, problem.expected_answer).success)


def candidate_semantically_correct(
    candidate: Candidate,
    branch: BranchGoal,
) -> bool:
    """Evaluation-only branch diagnostic, never a policy observation."""
    if not candidate.usable:
        return False
    try:
        actual = parse_integer_answer(candidate.answer)
    except ValueError:
        return False
    return actual == branch.expected_answer


def target_state_record(
    problem: TargetProblem,
    candidates: tuple[Candidate, Candidate],
) -> dict[str, object]:
    """Build the label-free state visible to any future selector."""
    root_candidate = reduce_branch_answers(candidates, problem.question_type)
    return {
        "schema_version": TARGET_STATE_VERSION,
        "problem_id": problem.problem_id,
        "problem": problem.problem,
        "question_type": problem.question_type,
        "nodes": [
            {
                "id": "root",
                "kind": "root",
                "candidate": root_candidate,
                "children": ["branch_0", "branch_1"],
            },
        ] + [
            {
                "id": f"branch_{branch.index}",
                "kind": "branch",
                "task": branch.task,
                "candidate": candidate.to_record(),
                "children": [],
            }
            for branch, candidate in zip(
                problem.branches, candidates, strict=True
            )
        ],
        "actions": [
            {
                "verb": "submit" if action is TargetAction.SUBMIT else "retry",
                "target": {
                    TargetAction.SUBMIT: "root",
                    TargetAction.RETRY_ROOT: "root",
                    TargetAction.RETRY_BRANCH_0: "branch_0",
                    TargetAction.RETRY_BRANCH_1: "branch_1",
                }[action],
                "action_id": action.value,
            }
            for action in ACTION_ORDER
        ],
    }


def replace_branch(
    candidates: tuple[Candidate, Candidate],
    branch_index: int,
    replacement: Candidate,
) -> tuple[Candidate, Candidate]:
    if branch_index not in {0, 1}:
        raise ValueError("branch index must be zero or one")
    values = list(candidates)
    values[branch_index] = replacement
    return values[0], values[1]


def request_sha256(request: WorkerRequest) -> str:
    return canonical_json_sha256(request.to_record())


__all__ = [
    "ACTION_ORDER",
    "CANDIDATE_SCHEMA",
    "RETRY_ACTIONS",
    "TARGET_ACTION_VERSION",
    "TARGET_EVALUATION_BINDING_VERSION",
    "TARGET_PROBLEM_SOURCE_VERSION",
    "TARGET_REDUCER_VERSION",
    "TARGET_ROOT_PROMPT_VERSION",
    "TARGET_SCHEMA_VERSION",
    "TARGET_STATE_VERSION",
    "TARGET_WORKER_PROMPT_VERSION",
    "TWO_KEY_PROMPT_VERSION",
    "BranchGoal",
    "Candidate",
    "TargetAction",
    "TargetProblem",
    "WorkerRequest",
    "branch_request",
    "candidate_from_payload",
    "candidate_semantically_correct",
    "load_target_problems",
    "parse_integer_answer",
    "reduce_branch_answers",
    "replace_branch",
    "request_sha256",
    "root_request",
    "target_evaluation_binding_sha256",
    "target_problem_from_record",
    "target_problem_source_record",
    "target_problem_source_sha256",
    "target_state_record",
    "terminal_reward",
    "two_key_branch_request",
    "two_key_root_request",
    "with_two_key_output_contract",
]
