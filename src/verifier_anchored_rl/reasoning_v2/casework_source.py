"""Trusted label-free casework problem source for reasoning v2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import canonical_json_bytes, canonical_json_sha256
from .math_adapter import CaseworkMathVerifier, build_casework_root_obligation
from .model import State, new_graph


CASEWORK_PROBLEM_SOURCE_CONTRACT = "casework-label-free-source-v2"
CASEWORK_PROBLEM_ADAPTER_IMPLEMENTATION = (
    "verifier_anchored_rl.reasoning_v2.casework_source.CaseworkProblemSource"
)

_SOURCE_RECORD_KEYS = frozenset({
    "schema_version",
    "problem_adapter_implementation",
    "problems",
})
_PROBLEM_RECORD_KEYS = frozenset({
    "id",
    "problem",
    "family",
    "question_type",
    "symbolic",
    "domain_direction",
    "domain_boundary",
})


class CaseworkSourceError(ValueError):
    """Raised when casework input cannot form a trusted v2 source."""


def _nonempty_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CaseworkSourceError(f"{path} must be nonempty text")
    return value


def _pairs_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CaseworkSourceError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise CaseworkSourceError(f"invalid JSON constant {value!r}")


@dataclass(frozen=True, slots=True)
class CaseworkProblem:
    """The only casework fields approved to cross the problem boundary."""

    problem_id: str
    problem: str
    family: str
    question_type: str
    symbolic: str
    domain_direction: str | None = None
    domain_boundary: str | None = None

    def __post_init__(self) -> None:
        for field in (
            "problem_id",
            "problem",
            "family",
            "question_type",
            "symbolic",
        ):
            _nonempty_text(getattr(self, field), f"casework.{field}")
        if self.family == "f1b":
            if self.domain_direction not in {"lt", "gt"}:
                raise CaseworkSourceError(
                    "casework f1b domain_direction must be lt or gt"
                )
            _nonempty_text(
                self.domain_boundary,
                "casework f1b domain_boundary",
            )
        elif (
            self.domain_direction is not None
            or self.domain_boundary is not None
        ):
            raise CaseworkSourceError(
                "only casework f1b may carry domain metadata"
            )

    @classmethod
    def from_dataset_record(cls, value: Mapping[str, Any]) -> CaseworkProblem:
        """Project one dataset row without reading answer-bearing fields."""
        if not isinstance(value, Mapping):
            raise CaseworkSourceError("casework row must be an object")
        try:
            family = _nonempty_text(value["family"], "casework.family")
            direction: str | None = None
            boundary: str | None = None
            if family == "f1b":
                difficulty = value["difficulty"]
                if not isinstance(difficulty, Mapping):
                    raise CaseworkSourceError(
                        "casework f1b difficulty must be an object"
                    )
                params = difficulty["params"]
                if not isinstance(params, Mapping):
                    raise CaseworkSourceError(
                        "casework f1b params must be an object"
                    )
                direction = _nonempty_text(
                    params["direction"],
                    "casework f1b direction",
                )
                raw_boundary = params["d"]
                if isinstance(raw_boundary, bool) or raw_boundary is None:
                    raise CaseworkSourceError(
                        "casework f1b boundary must be scalar text or number"
                    )
                if not isinstance(raw_boundary, (str, int, float)):
                    raise CaseworkSourceError(
                        "casework f1b boundary must be scalar text or number"
                    )
                boundary = str(raw_boundary)
            return cls(
                problem_id=_nonempty_text(value["id"], "casework.id"),
                problem=_nonempty_text(
                    value["problem"], "casework.problem"
                ),
                family=family,
                question_type=_nonempty_text(
                    value["question_type"],
                    "casework.question_type",
                ),
                symbolic=_nonempty_text(
                    value["symbolic"], "casework.symbolic"
                ),
                domain_direction=direction,
                domain_boundary=boundary,
            )
        except KeyError as exc:
            raise CaseworkSourceError(
                f"casework row is missing required field {exc.args[0]!r}"
            ) from exc

    def to_record(self) -> dict[str, object]:
        return {
            "id": self.problem_id,
            "problem": self.problem,
            "family": self.family,
            "question_type": self.question_type,
            "symbolic": self.symbolic,
            "domain_direction": self.domain_direction,
            "domain_boundary": self.domain_boundary,
        }

    def adapter_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "family": self.family,
            "question_type": self.question_type,
            "symbolic": self.symbolic,
        }
        if self.family == "f1b":
            record["difficulty"] = {
                "params": {
                    "direction": self.domain_direction,
                    "d": self.domain_boundary,
                }
            }
        return record


def project_casework_problem(value: Mapping[str, Any]) -> dict[str, object]:
    """Return the canonical approved projection of one dataset row."""
    return CaseworkProblem.from_dataset_record(value).to_record()


def _initial_state(problem: CaseworkProblem) -> State:
    adapter = CaseworkMathVerifier()
    obligation = build_casework_root_obligation(problem.adapter_record())
    graph = new_graph(
        problem.problem,
        obligation,
        generation_contract=adapter,
    )
    return State(graph=graph, turn=0, previous_outcome=None)


@dataclass(frozen=True, slots=True)
class CaseworkProblemSource:
    """Immutable resolver over an ordered, label-free problem projection."""

    _problems: tuple[CaseworkProblem, ...]
    _states: tuple[State, ...]
    problem_source_contract: str = CASEWORK_PROBLEM_SOURCE_CONTRACT
    problem_adapter_implementation: str = (
        CASEWORK_PROBLEM_ADAPTER_IMPLEMENTATION
    )

    def __post_init__(self) -> None:
        if self.problem_source_contract != CASEWORK_PROBLEM_SOURCE_CONTRACT:
            raise CaseworkSourceError("problem source contract is invalid")
        if (
            self.problem_adapter_implementation
            != CASEWORK_PROBLEM_ADAPTER_IMPLEMENTATION
        ):
            raise CaseworkSourceError(
                "problem adapter implementation is invalid"
            )
        if not self._problems:
            raise CaseworkSourceError("casework source cannot be empty")
        if len(self._problems) != len(self._states):
            raise CaseworkSourceError(
                "casework problems and initial states are misaligned"
            )
        ids = tuple(problem.problem_id for problem in self._problems)
        if len(ids) != len(set(ids)):
            raise CaseworkSourceError("casework problem ids must be unique")
        for problem, state in zip(
            self._problems, self._states, strict=True
        ):
            if not isinstance(problem, CaseworkProblem):
                raise CaseworkSourceError("casework problem has invalid type")
            if state != _initial_state(problem):
                raise CaseworkSourceError(
                    "casework initial state differs from its approved projection"
                )

    @classmethod
    def from_records(
        cls, records: Iterable[Mapping[str, Any]]
    ) -> CaseworkProblemSource:
        problems = tuple(
            CaseworkProblem.from_dataset_record(record)
            for record in records
        )
        states = tuple(_initial_state(problem) for problem in problems)
        return cls(problems, states)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> CaseworkProblemSource:
        """Load JSONL and retain only the canonical approved projection."""
        source_path = Path(path)
        problems: list[CaseworkProblem] = []
        try:
            with source_path.open("r", encoding="utf-8", newline="") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if line.endswith("\r\n"):
                        raise CaseworkSourceError(
                            f"line {line_number} uses a CRLF newline"
                        )
                    if not line.endswith("\n"):
                        raise CaseworkSourceError(
                            f"line {line_number} lacks a trailing newline"
                        )
                    text = line[:-1]
                    if not text:
                        raise CaseworkSourceError(
                            f"line {line_number} is blank"
                        )
                    try:
                        value = json.loads(
                            text,
                            object_pairs_hook=_pairs_object,
                            parse_constant=_reject_constant,
                        )
                    except (json.JSONDecodeError, TypeError) as exc:
                        raise CaseworkSourceError(
                            f"line {line_number} is invalid JSON"
                        ) from exc
                    if type(value) is not dict:
                        raise CaseworkSourceError(
                            f"line {line_number} must be a JSON object"
                        )
                    problems.append(
                        CaseworkProblem.from_dataset_record(value)
                    )
        except UnicodeDecodeError as exc:
            raise CaseworkSourceError(
                "casework JSONL must be valid UTF-8"
            ) from exc
        frozen_problems = tuple(problems)
        return cls(
            frozen_problems,
            tuple(_initial_state(problem) for problem in frozen_problems),
        )

    @property
    def problem_ids(self) -> tuple[str, ...]:
        return tuple(problem.problem_id for problem in self._problems)

    @property
    def problem_source_sha256(self) -> str:
        return canonical_json_sha256(self.to_record())

    def to_record(self) -> dict[str, object]:
        record = {
            "schema_version": self.problem_source_contract,
            "problem_adapter_implementation": (
                self.problem_adapter_implementation
            ),
            "problems": [problem.to_record() for problem in self._problems],
        }
        if set(record) != _SOURCE_RECORD_KEYS or any(
            set(problem) != _PROBLEM_RECORD_KEYS
            for problem in record["problems"]
        ):
            raise AssertionError("casework source projection keys drifted")
        canonical_json_bytes(record)
        return record

    def select_slice(
        self, start: int, count: int
    ) -> CaseworkProblemSource:
        if type(start) is not int or start < 0:
            raise CaseworkSourceError("source slice start must be nonnegative")
        if type(count) is not int or count <= 0:
            raise CaseworkSourceError("source slice count must be positive")
        stop = start + count
        if stop > len(self._problems):
            raise CaseworkSourceError("source slice exceeds available problems")
        return CaseworkProblemSource(
            self._problems[start:stop],
            self._states[start:stop],
        )

    def approved_problem_record(self, problem_id: str) -> dict[str, object]:
        for problem in self._problems:
            if problem.problem_id == problem_id:
                return json.loads(canonical_json_bytes(problem.to_record()))
        raise KeyError(problem_id)

    def __len__(self) -> int:
        return len(self._problems)

    def __call__(self, problem_id: str) -> State:
        if not isinstance(problem_id, str) or not problem_id.strip():
            raise KeyError(problem_id)
        for problem, state in zip(
            self._problems, self._states, strict=True
        ):
            if problem.problem_id == problem_id:
                return state
        raise KeyError(problem_id)


__all__ = [
    "CASEWORK_PROBLEM_ADAPTER_IMPLEMENTATION",
    "CASEWORK_PROBLEM_SOURCE_CONTRACT",
    "CaseworkProblem",
    "CaseworkProblemSource",
    "CaseworkSourceError",
    "project_casework_problem",
]
