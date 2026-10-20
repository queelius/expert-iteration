"""Offline tolerant-payload replay over sealed E43 first attempts."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, TextIO

from .records import canonical_json_bytes, canonical_json_sha256
from .target_value import (
    TARGET_EVALUATION_BINDING_VERSION,
    TARGET_PROBLEM_SOURCE_VERSION,
    TARGET_REDUCER_VERSION,
    Candidate,
    TargetProblem,
    candidate_semantically_correct,
    reduce_branch_answers,
    target_evaluation_binding_sha256,
    target_problem_source_sha256,
    terminal_reward,
)
from .target_value_experiment import file_sha256


TOLERANT_CANONICALIZER_VERSION = "workspace-tolerant-payload-canonicalizer-v1"
TOLERANT_SELECTION_VERSION = "workspace-e43-first-attempt-selection-v1"
TOLERANT_CONTRACT_VERSION = "workspace-tolerant-replay-contract-v1"
TOLERANT_HEADER_VERSION = "workspace-tolerant-replay-header-v1"
TOLERANT_RECORD_VERSION = "workspace-tolerant-canonicalization-v1"
TOLERANT_CASE_VERSION = "workspace-tolerant-case-v1"
TOLERANT_SUMMARY_VERSION = "workspace-tolerant-summary-v1"
TOLERANT_ANALYSIS_VERSION = "workspace-tolerant-analysis-v1"

INITIAL_TARGETS = ("root", "branch_0", "branch_1")
_METADATA_KEYS = frozenset({
    "reasoning",
    "reason",
    "explanation",
    "work",
    "steps",
    "analysis",
    "usable",
    "error_type",
})
_REASONING_KEYS = (
    "reasoning",
    "reason",
    "explanation",
    "work",
    "analysis",
)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _validate_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase sha256")


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value!r}")


def _parse_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty_response")
    try:
        value = json.loads(stripped, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError):
        match = _JSON_OBJECT_RE.search(stripped)
        if match is None:
            raise ValueError("no_json_object") from None
        try:
            value = json.loads(
                match.group(), parse_constant=_reject_constant
            )
        except (json.JSONDecodeError, ValueError):
            raise ValueError("invalid_json_object") from None
    if type(value) is not dict:
        raise ValueError("top_level_not_object")
    return value


def _normalize_scalar(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is bool:
        raise ValueError("boolean_value")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("empty_string_value")
        return text
    if type(value) is int:
        return str(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite_value")
        return str(value)
    raise ValueError("nonscalar_value")


@dataclass(frozen=True, slots=True)
class Canonicalization:
    candidate: Candidate
    disposition: str
    selected_fields: tuple[str, ...]


def canonicalize_payload(raw_content: str) -> Canonicalization:
    """Canonicalize JSON field structure without evaluating answer truth."""
    if not isinstance(raw_content, str):
        raise TypeError("raw content must be text")
    try:
        payload = _parse_object(raw_content)
    except ValueError as exc:
        code = str(exc)
        return Canonicalization(
            candidate=Candidate(
                reasoning="",
                answer=None,
                usable=False,
                error_type=code,
            ),
            disposition=code,
            selected_fields=(),
        )
    reasoning = next((
        value.strip()
        for key in _REASONING_KEYS
        if isinstance((value := payload.get(key)), str) and value.strip()
    ), "")

    if "answer" in payload:
        try:
            answer = _normalize_scalar(payload["answer"])
        except ValueError as exc:
            code = str(exc)
            return Canonicalization(
                Candidate("", None, usable=False, error_type=code),
                code,
                ("answer",),
            )
        return Canonicalization(
            Candidate(reasoning=reasoning, answer=answer),
            "answer_key",
            ("answer",),
        )

    fields = tuple(
        (key, value)
        for key, value in payload.items()
        if key not in _METADATA_KEYS
    )
    if not fields:
        return Canonicalization(
            Candidate("", None, usable=False, error_type="no_value_field"),
            "no_value_field",
            (),
        )
    normalized: list[tuple[str, str | None]] = []
    try:
        for key, value in fields:
            normalized.append((key, _normalize_scalar(value)))
    except ValueError as exc:
        code = str(exc)
        return Canonicalization(
            Candidate("", None, usable=False, error_type=code),
            code,
            tuple(key for key, _ in fields),
        )
    values = {value for _, value in normalized}
    if len(values) != 1:
        return Canonicalization(
            Candidate(
                "", None, usable=False, error_type="conflicting_values"
            ),
            "conflicting_values",
            tuple(key for key, _ in normalized),
        )
    disposition = (
        "unique_scalar_field"
        if len(normalized) == 1
        else "agreeing_scalar_fields"
    )
    return Canonicalization(
        Candidate(reasoning=reasoning, answer=normalized[0][1]),
        disposition,
        tuple(key for key, _ in normalized),
    )


@dataclass(frozen=True, slots=True)
class SelectedAttempt:
    problem_id: str
    target: str
    call_id: str
    attempt_index: int
    strict_accepted: bool
    raw_content: str = field(repr=False)
    raw_content_sha256: str

    def __post_init__(self) -> None:
        if self.target not in INITIAL_TARGETS:
            raise ValueError("selected attempt target is invalid")
        if self.attempt_index != 0:
            raise ValueError("selected attempt must have index zero")
        _validate_sha256(self.raw_content_sha256, "raw content sha256")

    def binding_record(self) -> dict[str, object]:
        return {
            "problem_id": self.problem_id,
            "target": self.target,
            "call_id": self.call_id,
            "attempt_index": self.attempt_index,
            "strict_accepted": self.strict_accepted,
            "raw_content_sha256": self.raw_content_sha256,
        }


@dataclass(frozen=True, slots=True)
class PreparedTolerantReplay:
    e43_run_sha256: str
    e43_diagnostic_sha256: str
    e43_run_contract_sha256: str
    problem_ids: tuple[str, ...]
    attempts: Mapping[tuple[str, str], SelectedAttempt]
    selected_attempts_sha256: str


def _load_jsonl(path: str | Path, name: str) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    with Path(path).open("r", encoding="ascii", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.endswith("\r\n"):
                raise ValueError(f"{name} line {line_number} uses CRLF")
            if not line.endswith("\n"):
                raise ValueError(
                    f"{name} line {line_number} lacks a trailing newline"
                )
            try:
                value = json.loads(line, parse_constant=_reject_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"{name} line {line_number} is invalid JSON"
                ) from exc
            if type(value) is not dict:
                raise ValueError(f"{name} line {line_number} is not an object")
            records.append(value)
    return tuple(records)


def prepare_tolerant_replay(
    *,
    e43_run_path: str | Path,
    e43_diagnostic_path: str | Path,
    expected_e43_run_sha256: str,
    expected_e43_diagnostic_sha256: str,
    expected_e43_run_contract_sha256: str,
    problems: tuple[TargetProblem, ...],
) -> PreparedTolerantReplay:
    """Bind each E43 initial call to its first raw HTTP response."""
    for value, name in (
        (expected_e43_run_sha256, "expected E43 run sha256"),
        (
            expected_e43_diagnostic_sha256,
            "expected E43 diagnostic sha256",
        ),
        (
            expected_e43_run_contract_sha256,
            "expected E43 run contract sha256",
        ),
    ):
        _validate_sha256(value, name)
    if file_sha256(e43_run_path) != expected_e43_run_sha256:
        raise ValueError("E43 run bytes differ from the expected digest")
    if file_sha256(e43_diagnostic_path) != expected_e43_diagnostic_sha256:
        raise ValueError("E43 diagnostic bytes differ from the expected digest")
    problem_ids = tuple(problem.problem_id for problem in problems)
    wanted = set(problem_ids)
    run_records = _load_jsonl(e43_run_path, "E43 run")
    observed_contract: str | None = None
    calls: dict[str, tuple[str, str, bool]] = {}
    for record in run_records:
        if record.get("record_type") == "run_header":
            observed_contract = str(record.get("run_contract_sha256"))
        if record.get("record_type") != "logical_worker_call":
            continue
        problem_id = record.get("problem_id")
        if problem_id not in wanted or record.get("phase") != "initial":
            continue
        target = record.get("target")
        if target not in INITIAL_TARGETS or record.get("repetition") is not None:
            continue
        call_id = record.get("worker_call_id")
        error = record.get("generation_error")
        if not isinstance(call_id, str) or type(error) is not bool:
            raise ValueError("E43 initial call record is malformed")
        if call_id in calls:
            raise ValueError("E43 initial call id is duplicated")
        calls[call_id] = (str(problem_id), str(target), not error)
    if observed_contract != expected_e43_run_contract_sha256:
        raise ValueError("E43 run contract differs from the expected digest")
    if len(calls) != len(problems) * len(INITIAL_TARGETS):
        raise ValueError("E43 initial call selection is incomplete")

    first_attempts: dict[str, dict[str, object]] = {}
    for record in _load_jsonl(e43_diagnostic_path, "E43 diagnostic"):
        if record.get("record_type") != "http_attempt":
            continue
        if record.get("role") != "worker":
            raise ValueError("E43 diagnostic attempt has an unexpected role")
        attempt = record.get("attempt")
        if not isinstance(attempt, dict):
            raise ValueError("E43 diagnostic attempt is malformed")
        call_id = attempt.get("call_id")
        if call_id not in calls:
            continue
        attempt_index = attempt.get("attempt_index")
        if type(attempt_index) is not int or attempt_index < 0:
            raise ValueError("E43 attempt index is invalid")
        prior = first_attempts.get(call_id)
        if prior is None or attempt_index < prior["attempt_index"]:
            first_attempts[call_id] = attempt
        elif attempt_index == prior["attempt_index"]:
            raise ValueError("E43 call has a duplicate attempt index")
    if set(first_attempts) != set(calls):
        raise ValueError("E43 initial calls lack first attempts")

    selected: dict[tuple[str, str], SelectedAttempt] = {}
    for call_id, (problem_id, target, strict_accepted) in calls.items():
        attempt = first_attempts[call_id]
        if attempt["attempt_index"] != 0:
            raise ValueError("E43 first attempt does not have index zero")
        raw_content = attempt.get("raw_content")
        if not isinstance(raw_content, str):
            raise ValueError("E43 first attempt lacks raw content")
        key = (problem_id, target)
        if key in selected:
            raise ValueError("E43 initial problem target is duplicated")
        selected[key] = SelectedAttempt(
            problem_id=problem_id,
            target=target,
            call_id=call_id,
            attempt_index=0,
            strict_accepted=strict_accepted,
            raw_content=raw_content,
            raw_content_sha256=canonical_json_sha256({
                "raw_content": raw_content
            }),
        )
    expected_keys = {
        (problem_id, target)
        for problem_id in problem_ids
        for target in INITIAL_TARGETS
    }
    if set(selected) != expected_keys:
        raise ValueError("E43 selected attempt keys are incomplete")
    selection_record = {
        "schema_version": TOLERANT_SELECTION_VERSION,
        "e43_run_sha256": expected_e43_run_sha256,
        "e43_diagnostic_sha256": expected_e43_diagnostic_sha256,
        "e43_run_contract_sha256": expected_e43_run_contract_sha256,
        "attempts": [
            selected[(problem_id, target)].binding_record()
            for problem_id in problem_ids
            for target in INITIAL_TARGETS
        ],
    }
    return PreparedTolerantReplay(
        e43_run_sha256=expected_e43_run_sha256,
        e43_diagnostic_sha256=expected_e43_diagnostic_sha256,
        e43_run_contract_sha256=expected_e43_run_contract_sha256,
        problem_ids=problem_ids,
        attempts=selected,
        selected_attempts_sha256=canonical_json_sha256(selection_record),
    )


@dataclass(frozen=True, slots=True)
class TolerantReplayContract:
    preregistration_id: str
    dataset_sha256: str
    problem_source_sha256: str
    evaluation_binding_sha256: str
    problem_ids: tuple[str, ...]
    e43_run_sha256: str
    e43_diagnostic_sha256: str
    e43_run_contract_sha256: str
    selected_attempts_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.preregistration_id, str) or not (
            self.preregistration_id.strip()
        ):
            raise ValueError("preregistration id must be nonempty")
        for name in (
            "dataset_sha256",
            "problem_source_sha256",
            "evaluation_binding_sha256",
            "e43_run_sha256",
            "e43_diagnostic_sha256",
            "e43_run_contract_sha256",
            "selected_attempts_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if len(self.problem_ids) != 60 or len(set(self.problem_ids)) != 60:
            raise ValueError("tolerant replay requires 60 unique problems")

    def header(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema_version": TOLERANT_HEADER_VERSION,
            "record_type": "run_header",
            "run_contract_version": TOLERANT_CONTRACT_VERSION,
            "preregistration_id": self.preregistration_id,
            "problem_source": {
                "dataset_sha256": self.dataset_sha256,
                "projection_schema_version": TARGET_PROBLEM_SOURCE_VERSION,
                "projection_sha256": self.problem_source_sha256,
                "evaluation_binding_version": (
                    TARGET_EVALUATION_BINDING_VERSION
                ),
                "evaluation_binding_sha256": self.evaluation_binding_sha256,
                "problem_ids": list(self.problem_ids),
            },
            "sealed_source": {
                "e43_run_sha256": self.e43_run_sha256,
                "e43_diagnostic_sha256": self.e43_diagnostic_sha256,
                "e43_run_contract_sha256": self.e43_run_contract_sha256,
                "selection_schema_version": TOLERANT_SELECTION_VERSION,
                "selected_attempts_sha256": self.selected_attempts_sha256,
            },
            "contracts": {
                "canonicalizer": TOLERANT_CANONICALIZER_VERSION,
                "reducer": TARGET_REDUCER_VERSION,
                "analysis": TOLERANT_ANALYSIS_VERSION,
            },
        }
        body["run_contract_sha256"] = canonical_json_sha256(body)
        return body

    @property
    def sha256(self) -> str:
        return str(self.header()["run_contract_sha256"])


def build_tolerant_replay_contract(
    *,
    preregistration_id: str,
    dataset_path: str | Path,
    problems: tuple[TargetProblem, ...],
    prepared: PreparedTolerantReplay,
) -> TolerantReplayContract:
    if prepared.problem_ids != tuple(problem.problem_id for problem in problems):
        raise ValueError("prepared ids differ from tolerant replay source")
    return TolerantReplayContract(
        preregistration_id=preregistration_id,
        dataset_sha256=file_sha256(dataset_path),
        problem_source_sha256=target_problem_source_sha256(problems),
        evaluation_binding_sha256=target_evaluation_binding_sha256(problems),
        problem_ids=prepared.problem_ids,
        e43_run_sha256=prepared.e43_run_sha256,
        e43_diagnostic_sha256=prepared.e43_diagnostic_sha256,
        e43_run_contract_sha256=prepared.e43_run_contract_sha256,
        selected_attempts_sha256=prepared.selected_attempts_sha256,
    )


@dataclass(frozen=True, slots=True)
class TolerantReplayCase:
    problem_id: str
    family: str
    strict_accepted: Mapping[str, bool]
    usable: Mapping[str, bool]
    direct_reward: int
    structured_reward: int
    branch_correct: tuple[bool, bool]


def analyze_tolerant_replay(
    cases: tuple[TolerantReplayCase, ...],
) -> dict[str, object]:
    if len(cases) != 60:
        raise ValueError("tolerant replay analysis requires 60 cases")
    strict_counts = {
        target: sum(int(case.strict_accepted[target]) for case in cases)
        for target in INITIAL_TARGETS
    }
    usable_counts = {
        target: sum(int(case.usable[target]) for case in cases)
        for target in INITIAL_TARGETS
    }
    branch_correct = {
        "branch_0": sum(int(case.branch_correct[0]) for case in cases),
        "branch_1": sum(int(case.branch_correct[1]) for case in cases),
    }
    direct = sum(case.direct_reward for case in cases)
    structured = sum(case.structured_reward for case in cases)
    paired = Counter(
        (
            "both"
            if case.direct_reward and case.structured_reward
            else "direct_only"
            if case.direct_reward
            else "structured_only"
            if case.structured_reward
            else "neither"
        )
        for case in cases
    )
    compensating = sum(
        int(case.structured_reward and not all(case.branch_correct))
        for case in cases
    )
    family_results: dict[str, dict[str, int]] = {}
    for family in sorted({case.family for case in cases}):
        subset = tuple(case for case in cases if case.family == family)
        family_results[family] = {
            "count": len(subset),
            "direct_solved": sum(case.direct_reward for case in subset),
            "structured_solved": sum(
                case.structured_reward for case in subset
            ),
        }
    integration_passed = all(
        usable_counts[target] >= 54 for target in INITIAL_TARGETS
    )
    regime_passed = (
        12 <= structured <= 51
        and branch_correct["branch_0"] >= 18
        and branch_correct["branch_1"] >= 18
    )
    return {
        "schema_version": TOLERANT_ANALYSIS_VERSION,
        "case_count": len(cases),
        "strict_accepted_counts": strict_counts,
        "tolerant_usable_counts": usable_counts,
        "terminal_solved": {
            "direct_initial": direct,
            "structured_submit": structured,
        },
        "paired_outcomes": dict(sorted(paired.items())),
        "branch_semantic_correct": branch_correct,
        "compensating_error_solved_count": compensating,
        "family_results": family_results,
        "gates": {
            "integration_passed": integration_passed,
            "regime_passed": regime_passed,
            "all_passed": integration_passed and regime_passed,
        },
        "behavior_cloning_eligible": False,
    }


def _write_record(handle: TextIO, record: Mapping[str, object]) -> None:
    encoded = canonical_json_bytes(dict(record))
    if b'"expected_answer"' in encoded or b'"solution_set"' in encoded:
        raise ValueError("published tolerant replay leaked a hidden field")
    handle.write(encoded.decode("ascii") + "\n")


def _partial_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.partial")


def collect_tolerant_replay(
    *,
    run_path: str | Path,
    dataset_path: str | Path,
    contract: TolerantReplayContract,
    prepared: PreparedTolerantReplay,
    problems: tuple[TargetProblem, ...],
) -> dict[str, object]:
    """Run and atomically publish the sealed offline replay."""
    if file_sha256(dataset_path) != contract.dataset_sha256:
        raise ValueError("dataset bytes differ from tolerant replay contract")
    if target_problem_source_sha256(problems) != contract.problem_source_sha256:
        raise ValueError("problem source differs from tolerant replay contract")
    if target_evaluation_binding_sha256(
        problems
    ) != contract.evaluation_binding_sha256:
        raise ValueError("evaluation binding differs from replay contract")
    if prepared.selected_attempts_sha256 != contract.selected_attempts_sha256:
        raise ValueError("selected attempts differ from replay contract")
    final_path = Path(run_path)
    partial_path = _partial_path(final_path)
    for path in (final_path, partial_path):
        if path.exists():
            raise FileExistsError(f"replay output already exists: {path}")
    if not final_path.parent.is_dir():
        raise FileNotFoundError("replay output directory does not exist")

    cases: list[TolerantReplayCase] = []
    dispositions: Counter[str] = Counter()
    selected_fields: Counter[str] = Counter()
    try:
        with partial_path.open("x", encoding="ascii", newline="\n") as handle:
            _write_record(handle, contract.header())
            for problem_index, problem in enumerate(problems):
                canonical: dict[str, Canonicalization] = {}
                strict: dict[str, bool] = {}
                for target in INITIAL_TARGETS:
                    attempt = prepared.attempts[(problem.problem_id, target)]
                    result = canonicalize_payload(attempt.raw_content)
                    canonical[target] = result
                    strict[target] = attempt.strict_accepted
                    dispositions[result.disposition] += 1
                    selected_fields.update(result.selected_fields)
                    _write_record(handle, {
                        "schema_version": TOLERANT_RECORD_VERSION,
                        "record_type": "canonicalization",
                        "preregistration_id": contract.preregistration_id,
                        "problem_index": problem_index,
                        "problem_id": problem.problem_id,
                        "target": target,
                        "source_call_id": attempt.call_id,
                        "source_attempt_index": attempt.attempt_index,
                        "source_raw_content_sha256": (
                            attempt.raw_content_sha256
                        ),
                        "strict_accepted": attempt.strict_accepted,
                        "disposition": result.disposition,
                        "selected_fields": list(result.selected_fields),
                        "candidate": result.candidate.to_record(),
                    })
                branch_candidates = (
                    canonical["branch_0"].candidate,
                    canonical["branch_1"].candidate,
                )
                reduced = reduce_branch_answers(
                    branch_candidates, problem.question_type
                )
                direct_reward = terminal_reward(
                    canonical["root"].candidate.answer or "", problem
                ) if canonical["root"].candidate.usable else 0
                structured_reward = terminal_reward(reduced, problem)
                branch_correct_tuple = tuple(
                    candidate_semantically_correct(candidate, branch)
                    for candidate, branch in zip(
                        branch_candidates, problem.branches, strict=True
                    )
                )
                case = TolerantReplayCase(
                    problem_id=problem.problem_id,
                    family=problem.family,
                    strict_accepted=strict,
                    usable={
                        target: canonical[target].candidate.usable
                        for target in INITIAL_TARGETS
                    },
                    direct_reward=direct_reward,
                    structured_reward=structured_reward,
                    branch_correct=(
                        branch_correct_tuple[0], branch_correct_tuple[1]
                    ),
                )
                cases.append(case)
                _write_record(handle, {
                    "schema_version": TOLERANT_CASE_VERSION,
                    "record_type": "case_summary",
                    "preregistration_id": contract.preregistration_id,
                    "problem_index": problem_index,
                    "problem_id": problem.problem_id,
                    "strict_accepted": strict,
                    "tolerant_usable": dict(case.usable),
                    "direct_reward": direct_reward,
                    "structured_reward": structured_reward,
                    "branch_correct": list(case.branch_correct),
                })
            analysis = analyze_tolerant_replay(tuple(cases))
            summary = {
                "schema_version": TOLERANT_SUMMARY_VERSION,
                "record_type": "run_summary",
                "preregistration_id": contract.preregistration_id,
                "run_contract_sha256": contract.sha256,
                "canonicalization_count": len(problems) * 3,
                "case_count": len(problems),
                "disposition_counts": dict(sorted(dispositions.items())),
                "selected_field_counts": dict(sorted(
                    selected_fields.items()
                )),
                "analysis": analysis,
            }
            _write_record(handle, summary)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(partial_path, final_path)
        partial_path.unlink()
    except BaseException:
        if final_path.exists() and partial_path.exists():
            left = final_path.stat()
            right = partial_path.stat()
            if left.st_dev == right.st_dev and left.st_ino == right.st_ino:
                final_path.unlink()
        raise
    return summary


__all__ = [
    "INITIAL_TARGETS",
    "TOLERANT_CANONICALIZER_VERSION",
    "Canonicalization",
    "PreparedTolerantReplay",
    "SelectedAttempt",
    "TolerantReplayCase",
    "TolerantReplayContract",
    "analyze_tolerant_replay",
    "build_tolerant_replay_contract",
    "canonicalize_payload",
    "collect_tolerant_replay",
    "prepare_tolerant_replay",
]
