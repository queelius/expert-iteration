"""Complete-answer action census over one frozen semantic workspace state."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from math import comb
from pathlib import Path
from statistics import mean
from typing import Mapping, TextIO

from ..llm_client import (
    LLMResponseError,
    OpenAICompatibleLLMClient,
    SCHEMA_VALIDATION_STRICT_V1,
    append_openai_compatible_parse_retry,
    build_openai_compatible_json_body,
    classify_openai_compatible_json_content,
    project_openai_compatible_response,
)
from ..verifier_adapter import check_answer
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

ACTION_CENSUS_PROBLEM_SOURCE_VERSION = "action-census-problem-source-v1"
ACTION_CENSUS_EVALUATION_BINDING_VERSION = "action-census-hidden-evaluation-binding-v1"
ACTION_CENSUS_STATE_VERSION = "action-census-semantic-state-v2"
ACTION_CENSUS_ACTION_VERSION = "action-census-revise-target-v1"
ACTION_CENSUS_REWARD_VERSION = "symbolic-terminal-binary-v1"
ACTION_CENSUS_BASE_PROMPT_VERSION = "action-census-base-answer-v1"
ACTION_CENSUS_BUILDER_PROMPT_VERSION = "action-census-three-node-builder-v1"
ACTION_CENSUS_ACTIONABLE_BUILDER_PROMPT_VERSION = (
    "action-census-actionable-check-builder-v1"
)
ACTION_CENSUS_ANCHORED_BUILDER_PROMPT_VERSION = (
    "action-census-anchored-check-builder-v1"
)
ACTION_CENSUS_ANCHORED_RANGE_BUILDER_PROMPT_VERSION = (
    "action-census-anchored-range-builder-v1"
)
ACTION_CENSUS_ANCHORED_RANGE_LIST_BUILDER_PROMPT_VERSION = (
    "action-census-anchored-range-list-builder-v1"
)
ACTION_CENSUS_WORKER_PROMPT_VERSION = "action-census-complete-revision-v1"
ACTION_CENSUS_ORDER_VERSION = "id-ranked-williams-action-order-v1"
ACTION_CENSUS_SEED_VERSION = "common-random-target-seeds-v1"
ACTION_CENSUS_CONTRACT_VERSION = "action-census-run-contract-v4"
ACTION_CENSUS_HEADER_VERSION = "action-census-run-header-v4"
ACTION_CENSUS_LEGACY_HEADER_VERSION = "action-census-run-header-v3"
ACTION_CENSUS_DIAGNOSTIC_VERSION = "action-census-diagnostic-header-v2"
ACTION_CENSUS_CALL_VERSION = "action-census-logical-call-v1"
ACTION_CENSUS_BASE_REPLAY_VERSION = "action-census-base-replay-v1"
ACTION_CENSUS_BASE_REPLAY_BINDING_VERSION = (
    "action-census-sealed-base-replay-binding-v1"
)
ACTION_CENSUS_ANCHORED_SPAN_PROJECTION_VERSION = (
    "action-census-reasoning-span-projection-v1"
)
ACTION_CENSUS_ANCHORED_CATALOG_BINDING_VERSION = (
    "action-census-reasoning-catalog-binding-v1"
)
ACTION_CENSUS_STATE_RECORD_VERSION = "action-census-state-record-v1"
ACTION_CENSUS_TRANSITION_VERSION = "action-census-transition-v1"
ACTION_CENSUS_CASE_VERSION = "action-census-case-summary-v1"
ACTION_CENSUS_SUMMARY_VERSION = "action-census-run-summary-v3"
ACTION_CENSUS_ANALYSIS_VERSION = "action-census-analysis-v3"
ACTION_CENSUS_IMPLEMENTATION_BINDING_VERSION = "action-census-python-package-tree-v1"

ACTION_TARGETS = ("root", "node_1", "node_2", "node_3")
POLICY_OPTIONS = ("submit", *ACTION_TARGETS)
WILLIAMS_ORDER_INDICES = (
    (0, 1, 3, 2),
    (1, 2, 0, 3),
    (2, 3, 1, 0),
    (3, 0, 2, 1),
)
EXPERIMENT_STAGES = frozenset({"integration_smoke", "pilot", "full"})
BUILDER_MODE_SEGMENT_DRAFT = "segment_draft_v1"
BUILDER_MODE_ACTIONABLE_CHECKS = "actionable_checks_v1"
BUILDER_MODE_ANCHORED_CHECKS = "anchored_checks_v1"
BUILDER_MODE_ANCHORED_RANGES = "anchored_ranges_v1"
BUILDER_MODE_ANCHORED_RANGE_LISTS = "anchored_range_lists_v1"
ANCHORED_BUILDER_MODES = (
    BUILDER_MODE_ANCHORED_CHECKS,
    BUILDER_MODE_ANCHORED_RANGES,
    BUILDER_MODE_ANCHORED_RANGE_LISTS,
)
BUILDER_MODES = (
    BUILDER_MODE_SEGMENT_DRAFT,
    BUILDER_MODE_ACTIONABLE_CHECKS,
    *ANCHORED_BUILDER_MODES,
)

ANCHORED_OPERATIONS = (
    "recheck_model",
    "recompute_local",
    "test_admissibility",
    "check_coverage",
    "check_finalization",
)
ANCHORED_OPERATION_TITLES = {
    "recheck_model": "Recheck the model or case setup",
    "recompute_local": "Recompute the selected local work",
    "test_admissibility": "Test the selected claim for admissibility",
    "check_coverage": "Check coverage around the selected work",
    "check_finalization": "Check the selected finalization step",
}

ANSWER_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "answer": {"type": ["string", "number"]},
    },
    "required": ["reasoning", "answer"],
    "additionalProperties": False,
}

SEMANTIC_NODE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["title", "content"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["nodes"],
    "additionalProperties": False,
}

ACTIONABLE_NODE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "object": {"type": "string"},
                },
                "required": ["operation", "object"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["nodes"],
    "additionalProperties": False,
}


def _anchored_node_schema(
    span_ids: tuple[str, ...],
    *,
    max_span_ids: int = 2,
) -> dict[str, object]:
    if not span_ids or len(span_ids) != len(set(span_ids)):
        raise ValueError("anchored builder span ids must be nonempty and unique")
    if type(max_span_ids) is not int or max_span_ids <= 0:
        raise ValueError("anchored builder maximum span ids must be positive")
    return {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": list(ANCHORED_OPERATIONS),
                        },
                        "span_ids": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": max_span_ids,
                            "items": {"type": "string", "enum": list(span_ids)},
                        },
                    },
                    "required": ["operation", "span_ids"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["nodes"],
        "additionalProperties": False,
    }


_BASE_SYSTEM = (
    "You are a frozen mathematical problem solver. Solve the complete problem "
    "carefully and return one complete answer."
)
_BASE_SUFFIX = (
    "Return exactly one JSON object with this shape: "
    '{"reasoning":"complete derivation","answer":"final answer"}. '
    "The top-level answer must be a nonempty string or number. Put only the "
    "requested final answer in answer. Return no wrapper or extra key."
)
_BUILDER_SYSTEM = (
    "You construct addressable semantic state for an external controller. "
    "Segment the displayed draft into exactly three useful components. Do not "
    "judge correctness, verify algebra, repair the solution, or choose an "
    "action. Preserve mistakes as part of the untrusted draft state."
)
_ACTION_SYSTEM = (
    "You are the frozen worker in an external reasoning workspace. Produce a "
    "complete revised answer to the root problem while paying special "
    "attention to the selected semantic target. Treat all displayed work as "
    "untrusted. Do not choose another action, invoke a verifier, assign reward, "
    "or claim that the workspace is correct."
)
_ACTION_SUFFIX = (
    "Independently solve the complete root problem, using the selected target "
    "only as an attention focus. Return exactly one JSON object with this "
    'shape: {"reasoning":"complete revised derivation","answer":"final '
    'answer"}. The top-level answer must be a nonempty string or number. '
    "Return no workspace, action, target, wrapper, or extra key."
)
_BASE_USER_TEMPLATE = "Complete problem:\n{problem}\n\n{instruction}"
_BUILDER_USER_TEMPLATE = "Visible problem and draft:\n{visible}\n\n{instruction}"
_BUILDER_INSTRUCTION = (
    "Return exactly three ordered semantic components. For each component, "
    "provide a short title and the corresponding draft content or summary. "
    "Return exactly this JSON shape with no extra keys: "
    '{"nodes":[{"title":"first","content":"first component"},'
    '{"title":"second","content":"second component"},'
    '{"title":"third","content":"third component"}]}'
)
_ACTIONABLE_BUILDER_SYSTEM = (
    "You construct addressable task state for an external controller. Using "
    "the complete problem and untrusted draft, define exactly three distinct, "
    "problem-specific subproblems or checks that a solver can execute while "
    "revising the complete root answer. Each node must say what to reason "
    "about and what object it operates on, not present a document section. "
    "State what must be checked without selecting which interpretation, case, "
    "expression, value, or conclusion is correct. Do not solve a check, "
    "compute a result, judge correctness, verify algebra, repair the draft, "
    "supply replacement work, a verdict, an answer-bearing hint, or a final "
    "answer, or choose an action."
)
_ACTIONABLE_BUILDER_INSTRUCTION = (
    "Return exactly three ordered, pairwise non-overlapping revision tasks. "
    "Choose the three most problem-specific operations whose independent "
    "execution could expose different failure modes in the displayed draft. "
    "Do not assign fixed roles to node positions. Do not state a resolved "
    "case, computed expression, calculated value, proposed replacement, "
    "correctness verdict, or answer. For each task, provide an imperative "
    "operation and a concrete object for that operation. Return "
    "exactly this JSON shape with no extra keys: "
    '{"nodes":[{"operation":"imperative operation","object":"specific '
    'object"},{"operation":"imperative operation","object":"specific '
    'object"},{"operation":"imperative operation","object":"specific '
    'object"}]}'
)
_ANCHORED_BUILDER_SYSTEM = (
    "You are a frozen source-span router for an external controller. Select "
    "exactly three disjoint groups from the displayed untrusted reasoning "
    "spans. A group contains one span or two consecutive spans. Assign each "
    "group one controlled audit operation. Return operation keys and span IDs "
    "only. Do not add, quote, paraphrase, summarize, solve, judge, correct, or "
    "replace mathematical content. The three selections should be "
    "complementary local revision checks, not one broad check that subsumes "
    "the other two."
)
_ANCHORED_BUILDER_INSTRUCTION = (
    "Choose exactly three pairwise-disjoint span groups. Within a two-span "
    "group the IDs must be consecutive. Use only these operations: "
    "recheck_model to re-derive a formulation or case setup; recompute_local "
    "to redo local algebra, arithmetic, or enumeration; test_admissibility to "
    "test branch assumptions, bounds, domains, or constraints; check_coverage "
    "to check omitted, duplicated, or incomplete cases; check_finalization to "
    "redo a requested count, extremum, selection, or output conversion. "
    "Operations may repeat. Return exactly this JSON shape with no extra keys: "
    '{"nodes":[{"operation":"recheck_model","span_ids":["S001"]},'
    '{"operation":"recompute_local","span_ids":["S002"]},'
    '{"operation":"check_finalization","span_ids":["S003"]}]}'
)
_ANCHORED_RANGE_BUILDER_SYSTEM = (
    "You are a frozen source-range router for an external controller. Select "
    "exactly three disjoint local ranges from the displayed untrusted "
    "reasoning spans. A range uses one span ID or two ordered endpoint IDs; "
    "two endpoints include every source span between them. Assign each range "
    "one controlled audit operation. Return operation keys and endpoint IDs "
    "only. Do not add, quote, paraphrase, summarize, solve, judge, correct, or "
    "replace mathematical content. The three selections should be "
    "complementary local revision checks, not one broad check that subsumes "
    "the other two."
)
_ANCHORED_RANGE_BUILDER_INSTRUCTION = (
    "Choose exactly three pairwise-disjoint local source ranges. Use one ID "
    "for one span, or two increasing IDs for the inclusive range from the "
    "first through the second. Choose the smallest range that supports each "
    "check. Use only these operations: recheck_model to re-derive a "
    "formulation or case setup; recompute_local to redo local algebra, "
    "arithmetic, or enumeration; test_admissibility to test branch "
    "assumptions, bounds, domains, or constraints; check_coverage to check "
    "omitted, duplicated, or incomplete cases; check_finalization to redo a "
    "requested count, extremum, selection, or output conversion. Operations "
    "may repeat. Return exactly this JSON shape with no extra keys: "
    '{"nodes":[{"operation":"recheck_model","span_ids":["S001"]},'
    '{"operation":"recompute_local","span_ids":["S002"]},'
    '{"operation":"check_finalization","span_ids":["S003"]}]}'
)
_ANCHORED_RANGE_LIST_BUILDER_SYSTEM = (
    "You are a frozen source-range router for an external controller. Select "
    "exactly three disjoint local ranges from the displayed untrusted "
    "reasoning spans. A range uses one or more strictly increasing source "
    "IDs; its first and last IDs are inclusive endpoints, and intermediate "
    "IDs may be included. Assign each range one controlled audit operation. "
    "Return operation keys and source IDs only. Do not add, quote, paraphrase, "
    "summarize, solve, judge, correct, or replace mathematical content. The "
    "three selections should be complementary local revision checks, not one "
    "broad check that subsumes the other two."
)
_ANCHORED_RANGE_LIST_BUILDER_INSTRUCTION = (
    "Choose exactly three pairwise-disjoint local source ranges. Use one ID "
    "for one span, or a strictly increasing ID list for the inclusive range "
    "from the first through the last ID. Intermediate IDs are allowed and do "
    "not change the outer range boundaries. Choose the smallest range that "
    "supports each check. Use only these operations: recheck_model to "
    "re-derive a formulation or case setup; recompute_local to redo local "
    "algebra, arithmetic, or enumeration; test_admissibility to test branch "
    "assumptions, bounds, domains, or constraints; check_coverage to check "
    "omitted, duplicated, or incomplete cases; check_finalization to redo a "
    "requested count, extremum, selection, or output conversion. Operations "
    "may repeat. Return exactly this JSON shape with no extra keys: "
    '{"nodes":[{"operation":"recheck_model","span_ids":["S001"]},'
    '{"operation":"recompute_local","span_ids":["S002","S003"]},'
    '{"operation":"check_finalization","span_ids":["S004"]}]}'
)
_ACTION_USER_TEMPLATE = (
    "Workspace state:\n{observation}\n\n"
    "Selected target id: {target}\n\n{instruction}"
)


def _builder_prompt(mode: str) -> tuple[str, str, str, str]:
    if mode == BUILDER_MODE_SEGMENT_DRAFT:
        return (
            ACTION_CENSUS_BUILDER_PROMPT_VERSION,
            _BUILDER_SYSTEM,
            _BUILDER_USER_TEMPLATE,
            _BUILDER_INSTRUCTION,
        )
    if mode == BUILDER_MODE_ACTIONABLE_CHECKS:
        return (
            ACTION_CENSUS_ACTIONABLE_BUILDER_PROMPT_VERSION,
            _ACTIONABLE_BUILDER_SYSTEM,
            _BUILDER_USER_TEMPLATE,
            _ACTIONABLE_BUILDER_INSTRUCTION,
        )
    if mode == BUILDER_MODE_ANCHORED_CHECKS:
        return (
            ACTION_CENSUS_ANCHORED_BUILDER_PROMPT_VERSION,
            _ANCHORED_BUILDER_SYSTEM,
            _BUILDER_USER_TEMPLATE,
            _ANCHORED_BUILDER_INSTRUCTION,
        )
    if mode == BUILDER_MODE_ANCHORED_RANGES:
        return (
            ACTION_CENSUS_ANCHORED_RANGE_BUILDER_PROMPT_VERSION,
            _ANCHORED_RANGE_BUILDER_SYSTEM,
            _BUILDER_USER_TEMPLATE,
            _ANCHORED_RANGE_BUILDER_INSTRUCTION,
        )
    if mode == BUILDER_MODE_ANCHORED_RANGE_LISTS:
        return (
            ACTION_CENSUS_ANCHORED_RANGE_LIST_BUILDER_PROMPT_VERSION,
            _ANCHORED_RANGE_LIST_BUILDER_SYSTEM,
            _BUILDER_USER_TEMPLATE,
            _ANCHORED_RANGE_LIST_BUILDER_INSTRUCTION,
        )
    raise ValueError("state-builder mode is invalid")


def _builder_schema(
    mode: str,
    span_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    _builder_prompt(mode)
    if mode == BUILDER_MODE_ACTIONABLE_CHECKS:
        return ACTIONABLE_NODE_SCHEMA
    if mode in ANCHORED_BUILDER_MODES:
        effective_span_ids = span_ids or ("<frozen-span-id>",)
        max_span_ids = (
            len(effective_span_ids) if mode == BUILDER_MODE_ANCHORED_RANGE_LISTS else 2
        )
        return _anchored_node_schema(
            effective_span_ids,
            max_span_ids=max_span_ids,
        )
    return SEMANTIC_NODE_SCHEMA


def _pairs_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value!r}")


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value.strip()


def _sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase sha256")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_binding() -> dict[str, object]:
    package_root = Path(__file__).resolve().parents[1]
    files = [
        {
            "path": path.relative_to(package_root).as_posix(),
            "sha256": file_sha256(path),
        }
        for path in sorted(package_root.rglob("*.py"))
    ]

    def distribution_version(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "not-installed"

    return {
        "schema_version": ACTION_CENSUS_IMPLEMENTATION_BINDING_VERSION,
        "package_tree_sha256": canonical_json_sha256({"files": files}),
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "httpx_version": distribution_version("httpx"),
        "sympy_version": distribution_version("sympy"),
    }


@dataclass(frozen=True, slots=True)
class CensusProblem:
    problem_id: str
    problem: str
    family: str
    expected_answer: str = field(repr=False)

    def __post_init__(self) -> None:
        _nonempty(self.problem_id, "problem id")
        _nonempty(self.problem, "problem")
        _nonempty(self.family, "family")
        if not isinstance(self.expected_answer, str):
            raise ValueError("expected answer must be text")

    def source_record(self) -> dict[str, str]:
        return {"id": self.problem_id, "problem": self.problem}


def load_census_problems(path: str | Path) -> tuple[CensusProblem, ...]:
    rows: list[CensusProblem] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.endswith("\r\n"):
                raise ValueError(f"line {line_number} uses CRLF")
            if not line.endswith("\n"):
                raise ValueError(f"line {line_number} lacks a trailing newline")
            if not line[:-1]:
                raise ValueError(f"line {line_number} is blank")
            value = json.loads(
                line,
                object_pairs_hook=_pairs_object,
                parse_constant=_reject_constant,
            )
            if type(value) is not dict:
                raise ValueError(f"line {line_number} is not an object")
            try:
                rows.append(
                    CensusProblem(
                        problem_id=_nonempty(value["id"], "problem id"),
                        problem=_nonempty(value["problem"], "problem"),
                        family=_nonempty(value["family"], "family"),
                        expected_answer=str(value["answer"]),
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"line {line_number} lacks field {exc.args[0]!r}"
                ) from exc
    if not rows:
        raise ValueError("action census problem source must be nonempty")
    ids = tuple(row.problem_id for row in rows)
    if len(ids) != len(set(ids)):
        raise ValueError("action census problem ids must be unique")
    return tuple(rows)


def problem_source_record(
    problems: tuple[CensusProblem, ...],
) -> dict[str, object]:
    if not problems:
        raise ValueError("problem source must be nonempty")
    return {
        "schema_version": ACTION_CENSUS_PROBLEM_SOURCE_VERSION,
        "problems": [problem.source_record() for problem in problems],
    }


def problem_source_sha256(problems: tuple[CensusProblem, ...]) -> str:
    return canonical_json_sha256(problem_source_record(problems))


def evaluation_binding_sha256(problems: tuple[CensusProblem, ...]) -> str:
    return canonical_json_sha256(
        {
            "schema_version": ACTION_CENSUS_EVALUATION_BINDING_VERSION,
            "labels": [
                {
                    "id": problem.problem_id,
                    "answer": problem.expected_answer,
                    "family": problem.family,
                }
                for problem in problems
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class BaseReplayBinding:
    source_run_sha256: str
    source_diagnostic_sha256: str
    source_run_contract_sha256: str
    projection_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "source_run_sha256",
            "source_diagnostic_sha256",
            "source_run_contract_sha256",
            "projection_sha256",
        ):
            _sha256(getattr(self, name), name)

    def to_record(self) -> dict[str, str]:
        return {
            "schema_version": ACTION_CENSUS_BASE_REPLAY_BINDING_VERSION,
            "source_run_sha256": self.source_run_sha256,
            "source_diagnostic_sha256": self.source_diagnostic_sha256,
            "source_run_contract_sha256": self.source_run_contract_sha256,
            "projection_sha256": self.projection_sha256,
        }


@dataclass(frozen=True, slots=True)
class BaseReplayCase:
    problem_id: str
    problem_index: int
    source_call_id: str
    source_record_sha256: str
    seed: int
    request_sha256: str
    candidate: Candidate
    generation_error: bool
    error_type: str | None

    def __post_init__(self) -> None:
        _nonempty(self.problem_id, "base replay problem id")
        _nonempty(self.source_call_id, "base replay call id")
        if type(self.problem_index) is not int or self.problem_index < 0:
            raise ValueError("base replay problem index must be nonnegative")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("base replay seed must be nonnegative")
        _sha256(self.source_record_sha256, "base replay record sha256")
        _sha256(self.request_sha256, "base replay request sha256")
        if not isinstance(self.candidate, Candidate):
            raise ValueError("base replay Candidate is invalid")
        if type(self.generation_error) is not bool:
            raise ValueError("base replay generation flag is invalid")
        if self.generation_error or self.error_type is not None:
            raise ValueError("base replay requires a successful source call")

    def projection_record(self) -> dict[str, object]:
        return {
            "problem_id": self.problem_id,
            "problem_index": self.problem_index,
            "source_call_id": self.source_call_id,
            "source_record_sha256": self.source_record_sha256,
            "seed": self.seed,
            "request_sha256": self.request_sha256,
            "candidate": self.candidate.to_record(),
            "candidate_sha256": canonical_json_sha256(self.candidate.to_record()),
            "generation_error": self.generation_error,
            "error_type": self.error_type,
        }


@dataclass(frozen=True, slots=True)
class PreparedBaseReplay:
    binding: BaseReplayBinding
    cases: tuple[BaseReplayCase, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.binding, BaseReplayBinding) or not self.cases:
            raise ValueError("prepared base replay is invalid")
        ids = tuple(case.problem_id for case in self.cases)
        if len(ids) != len(set(ids)):
            raise ValueError("base replay problem ids must be unique")
        if tuple(case.problem_index for case in self.cases) != tuple(
            range(len(self.cases))
        ):
            raise ValueError("base replay problem indices are not contiguous")
        projection_sha256 = canonical_json_sha256(
            {
                "schema_version": ACTION_CENSUS_BASE_REPLAY_BINDING_VERSION,
                "cases": [case.projection_record() for case in self.cases],
            }
        )
        if projection_sha256 != self.binding.projection_sha256:
            raise ValueError("base replay cases differ from their binding")

    def case(self, problem_id: str) -> BaseReplayCase:
        matches = tuple(case for case in self.cases if case.problem_id == problem_id)
        if len(matches) != 1:
            raise ValueError("base replay problem lookup is not unique")
        return matches[0]


@dataclass(frozen=True, slots=True)
class DraftSpan:
    span_id: str
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        _nonempty(self.span_id, "draft span id")
        if type(self.start) is not int or type(self.end) is not int:
            raise ValueError("draft span offsets must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("draft span offsets are invalid")
        _nonempty(self.text, "draft span text")

    def to_record(self) -> dict[str, object]:
        return {
            "id": self.span_id,
            "start": self.start,
            "end": self.end,
            "sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest(),
            "text": self.text,
        }


def anchored_draft_spans(draft: Candidate) -> tuple[DraftSpan, ...]:
    reasoning = draft.reasoning
    bounds: list[tuple[int, int]] = []

    def append_trimmed(start: int, end: int) -> None:
        while start < end and reasoning[start].isspace():
            start += 1
        while end > start and reasoning[end - 1].isspace():
            end -= 1
        if start < end:
            bounds.append((start, end))

    cursor = 0
    for boundary in re.finditer(r"(?<=[.!?])(?:[ \t]+|\n+)|\n+", reasoning):
        append_trimmed(cursor, boundary.start())
        cursor = boundary.end()
    append_trimmed(cursor, len(reasoning))
    if len(bounds) < 3:
        raise ValueError("anchored builder requires at least three reasoning spans")
    return tuple(
        DraftSpan(
            span_id=f"S{index:03d}",
            start=start,
            end=end,
            text=reasoning[start:end],
        )
        for index, (start, end) in enumerate(bounds, start=1)
    )


def anchored_span_catalog_record(draft: Candidate) -> dict[str, object]:
    return {
        "schema_version": ACTION_CENSUS_ANCHORED_SPAN_PROJECTION_VERSION,
        "reasoning_sha256": hashlib.sha256(draft.reasoning.encode("utf-8")).hexdigest(),
        "spans": [span.to_record() for span in anchored_draft_spans(draft)],
    }


def anchored_catalog_binding_sha256(
    problems: tuple[CensusProblem, ...],
    base_replay: PreparedBaseReplay,
) -> str:
    if tuple(problem.problem_id for problem in problems) != tuple(
        case.problem_id for case in base_replay.cases
    ):
        raise ValueError("anchored catalog problems differ from base replay")
    return canonical_json_sha256(
        {
            "schema_version": ACTION_CENSUS_ANCHORED_CATALOG_BINDING_VERSION,
            "cases": [
                {
                    "problem_id": problem.problem_id,
                    "candidate_sha256": canonical_json_sha256(
                        replay.candidate.to_record()
                    ),
                    "catalog": anchored_span_catalog_record(replay.candidate),
                }
                for problem, replay in zip(
                    problems,
                    base_replay.cases,
                    strict=True,
                )
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class SemanticNode:
    node_id: str
    title: str
    content: str

    def __post_init__(self) -> None:
        if self.node_id not in ACTION_TARGETS[1:]:
            raise ValueError("semantic node id is invalid")
        if not isinstance(self.title, str) or not isinstance(self.content, str):
            raise ValueError("semantic node fields must be text")
        if len(self.title) > 500 or len(self.content) > 4_000:
            raise ValueError("semantic node exceeds its text bound")

    def to_record(self) -> dict[str, object]:
        return {
            "id": self.node_id,
            "kind": "semantic_component",
            "parent_id": "root",
            "title": self.title or "[empty generated title]",
            "content": self.content or "[empty generated content]",
            "children": [],
        }


@dataclass(frozen=True, slots=True)
class CensusState:
    problem_id: str
    problem: str
    draft: Candidate
    nodes: tuple[SemanticNode, SemanticNode, SemanticNode]

    def __post_init__(self) -> None:
        _nonempty(self.problem_id, "state problem id")
        _nonempty(self.problem, "state problem")
        if not isinstance(self.draft, Candidate):
            raise ValueError("state draft must be a Candidate")
        if tuple(node.node_id for node in self.nodes) != ACTION_TARGETS[1:]:
            raise ValueError("state requires ordered semantic nodes")

    def observation(self) -> dict[str, object]:
        return {
            "schema_version": ACTION_CENSUS_STATE_VERSION,
            "problem_id": self.problem_id,
            "problem": self.problem,
            "phase": "one_call_revision",
            "remaining_worker_calls": 1,
            "nodes": [
                {
                    "id": "root",
                    "kind": "root_candidate",
                    "parent_id": None,
                    "title": "Complete draft solution",
                    "candidate": self.draft.to_record(),
                    "children": list(ACTION_TARGETS[1:]),
                },
                *(node.to_record() for node in self.nodes),
            ],
            "legal_actions": [
                {"verb": "submit", "target": "root"},
                *({"verb": "revise", "target": target} for target in ACTION_TARGETS),
            ],
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.observation())


def base_request(problem: CensusProblem) -> WorkerRequest:
    return WorkerRequest(
        system=_BASE_SYSTEM,
        user=_BASE_USER_TEMPLATE.format(
            problem=problem.problem,
            instruction=_BASE_SUFFIX,
        ),
        schema=ANSWER_SCHEMA,
    )


def state_builder_request(
    problem: CensusProblem,
    draft: Candidate,
    mode: str = BUILDER_MODE_SEGMENT_DRAFT,
) -> WorkerRequest:
    _, system, user_template, instruction = _builder_prompt(mode)
    if mode in ANCHORED_BUILDER_MODES:
        spans = anchored_draft_spans(draft)
        visible = {
            "problem_id": problem.problem_id,
            "problem": problem.problem,
            "span_projection_version": (ACTION_CENSUS_ANCHORED_SPAN_PROJECTION_VERSION),
            "frozen_reasoning_spans": [span.to_record() for span in spans],
        }
        schema = _builder_schema(mode, tuple(span.span_id for span in spans))
    else:
        visible = {
            "problem": problem.problem,
            "untrusted_draft": draft.to_record(),
        }
        schema = _builder_schema(mode)
    return WorkerRequest(
        system=system,
        user=user_template.format(
            visible=canonical_json_bytes(visible).decode("ascii"),
            instruction=instruction,
        ),
        schema=schema,
    )


def decode_semantic_nodes(
    payload: Mapping[str, object],
    mode: str = BUILDER_MODE_SEGMENT_DRAFT,
    *,
    problem: CensusProblem | None = None,
    draft: Candidate | None = None,
) -> tuple[SemanticNode, SemanticNode, SemanticNode]:
    _builder_schema(mode)
    raw = payload.get("nodes") if isinstance(payload, Mapping) else None
    if type(raw) is not list or len(raw) != 3:
        raise ValueError("state builder requires exactly three nodes")
    if mode in ANCHORED_BUILDER_MODES:
        if problem is None or draft is None:
            raise ValueError("anchored builder decoding requires problem and draft")
        spans = anchored_draft_spans(draft)
        span_by_id = {span.span_id: span for span in spans}
        position_by_id = {span.span_id: position for position, span in enumerate(spans)}
        selections: list[tuple[int, str, tuple[str, ...], str]] = []
        selected_span_ids: set[str] = set()
        for item in raw:
            if not isinstance(item, Mapping):
                raise ValueError("state builder node must be an object")
            if set(item) != {"operation", "span_ids"}:
                raise ValueError("anchored builder node fields are invalid")
            operation = item.get("operation")
            span_ids = item.get("span_ids")
            if operation not in ANCHORED_OPERATIONS or type(span_ids) is not list:
                raise ValueError("anchored builder selection is invalid")
            valid_group_size = (
                len(span_ids) >= 1
                if mode == BUILDER_MODE_ANCHORED_RANGE_LISTS
                else len(span_ids) in (1, 2)
            )
            if not valid_group_size or any(
                not isinstance(span_id, str) for span_id in span_ids
            ):
                raise ValueError("anchored builder span group is invalid")
            group = tuple(span_ids)
            if len(group) != len(set(group)):
                raise ValueError("anchored builder span group contains duplicates")
            if any(span_id not in span_by_id for span_id in group):
                raise ValueError("anchored builder selected an unknown span")
            positions = tuple(position_by_id[span_id] for span_id in group)
            if mode == BUILDER_MODE_ANCHORED_CHECKS and positions != tuple(
                range(positions[0], positions[0] + len(group))
            ):
                raise ValueError("anchored builder span group must be consecutive")
            if (
                mode
                in (
                    BUILDER_MODE_ANCHORED_RANGES,
                    BUILDER_MODE_ANCHORED_RANGE_LISTS,
                )
                and len(positions) >= 2
                and any(
                    later <= earlier for earlier, later in zip(positions, positions[1:])
                )
            ):
                raise ValueError("anchored range ids must be increasing")
            covered_span_ids = tuple(
                spans[position].span_id
                for position in range(positions[0], positions[-1] + 1)
            )
            if selected_span_ids.intersection(covered_span_ids):
                raise ValueError("anchored builder span selections must be disjoint")
            selected_span_ids.update(covered_span_ids)
            first = span_by_id[group[0]]
            last = span_by_id[group[-1]]
            selections.append(
                (
                    first.start,
                    operation,
                    group,
                    draft.reasoning[first.start : last.end],
                )
            )
        selections.sort(key=lambda selection: selection[0])
        anchored_nodes = tuple(
            SemanticNode(
                f"node_{index}",
                f"{ANCHORED_OPERATION_TITLES[operation]} "
                + (
                    f"[{span_ids[0]}-{span_ids[-1]}]"
                    if mode
                    in (
                        BUILDER_MODE_ANCHORED_RANGES,
                        BUILDER_MODE_ANCHORED_RANGE_LISTS,
                    )
                    and len(span_ids) >= 2
                    else f"[{','.join(span_ids)}]"
                ),
                content,
            )
            for index, (_, operation, span_ids, content) in enumerate(
                selections,
                start=1,
            )
        )
        return anchored_nodes  # type: ignore[return-value]
    nodes: list[SemanticNode] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, Mapping):
            raise ValueError("state builder node must be an object")
        if mode == BUILDER_MODE_ACTIONABLE_CHECKS:
            title = item.get("operation")
            content = item.get("object")
        else:
            title = item.get("title")
            content = item.get("content")
        if not isinstance(title, str) or not isinstance(content, str):
            raise ValueError("state builder node fields must be text")
        nodes.append(SemanticNode(f"node_{index}", title.strip(), content.strip()))
    return (nodes[0], nodes[1], nodes[2])


def fallback_semantic_nodes() -> tuple[SemanticNode, SemanticNode, SemanticNode]:
    return tuple(  # type: ignore[return-value]
        SemanticNode(
            f"node_{index}",
            "Unavailable semantic component",
            "No schema-valid state-builder output was available.",
        )
        for index in range(1, 4)
    )


def semantic_nodes_distinct(
    nodes: tuple[SemanticNode, SemanticNode, SemanticNode],
    mode: str,
) -> bool:
    _builder_prompt(mode)
    if mode == BUILDER_MODE_ACTIONABLE_CHECKS or mode in ANCHORED_BUILDER_MODES:
        normalized = {
            (
                " ".join(node.title.casefold().split()),
                " ".join(node.content.casefold().split()),
            )
            for node in nodes
        }
    else:
        normalized = {(node.content,) for node in nodes}
    return len(normalized) == len(nodes)


def action_request(state: CensusState, target: str) -> WorkerRequest:
    if target not in ACTION_TARGETS:
        raise ValueError("action target is invalid")
    observation = canonical_json_bytes(state.observation()).decode("ascii")
    return WorkerRequest(
        system=_ACTION_SYSTEM,
        user=_ACTION_USER_TEMPLATE.format(
            observation=observation,
            target=target,
            instruction=_ACTION_SUFFIX,
        ),
        schema=ANSWER_SCHEMA,
    )


def assert_action_requests_isolated(
    requests: Mapping[str, WorkerRequest],
) -> None:
    if tuple(requests) != ACTION_TARGETS:
        raise ValueError("isolated requests require every ordered target")
    normalized: list[str] = []
    first = requests[ACTION_TARGETS[0]]
    for target in ACTION_TARGETS:
        request = requests[target]
        if request.system != first.system or request.schema != first.schema:
            raise RuntimeError("action requests changed a non-target contract")
        marker = f"Selected target id: {target}"
        if request.user.count(marker) != 1:
            raise RuntimeError("action request target marker is not unique")
        normalized.append(request.user.replace(marker, "Selected target id: <TARGET>"))
    if len(set(normalized)) != 1:
        raise RuntimeError("action requests differ outside selected target")


def prompt_bindings(
    builder_mode: str = BUILDER_MODE_SEGMENT_DRAFT,
) -> dict[str, str]:
    """Bind exact static prompt text and schemas into every run contract."""
    _, builder_system, builder_user_template, builder_instruction = _builder_prompt(
        builder_mode
    )
    builder_projection: dict[str, object] = {
        "system": builder_system,
        "user_template": builder_user_template,
        "instruction": builder_instruction,
        "schema": _builder_schema(builder_mode),
    }
    if builder_mode in ANCHORED_BUILDER_MODES:
        builder_projection["span_projection_version"] = (
            ACTION_CENSUS_ANCHORED_SPAN_PROJECTION_VERSION
        )
    return {
        "base": canonical_json_sha256(
            {
                "system": _BASE_SYSTEM,
                "user_template": _BASE_USER_TEMPLATE,
                "instruction": _BASE_SUFFIX,
                "schema": ANSWER_SCHEMA,
            }
        ),
        "state_builder": canonical_json_sha256(builder_projection),
        "action_worker": canonical_json_sha256(
            {
                "system": _ACTION_SYSTEM,
                "user_template": _ACTION_USER_TEMPLATE,
                "instruction": _ACTION_SUFFIX,
                "schema": ANSWER_SCHEMA,
            }
        ),
    }


@dataclass(frozen=True, slots=True)
class ActionCensusContract:
    preregistration_id: str
    stage: str
    dataset_sha256: str
    problem_source_sha256: str
    evaluation_binding_sha256: str
    problem_ids: tuple[str, ...]
    worker: ModelIdentity
    state_builder: ModelIdentity
    worker_temperature: float
    worker_max_tokens: int
    worker_seed: int
    builder_temperature: float
    builder_max_tokens: int
    builder_seed: int
    action_order_seed: int
    builder_mode: str = BUILDER_MODE_SEGMENT_DRAFT
    base_replay: BaseReplayBinding | None = None
    anchored_catalog_sha256: str | None = None
    repetitions: int = 4
    discovery_repetitions: tuple[int, ...] = (0, 1)
    evaluation_repetitions: tuple[int, ...] = (2, 3)

    def __post_init__(self) -> None:
        _nonempty(self.preregistration_id, "preregistration id")
        if self.stage not in EXPERIMENT_STAGES:
            raise ValueError("action census stage is invalid")
        _builder_prompt(self.builder_mode)
        if self.base_replay is not None and not isinstance(
            self.base_replay,
            BaseReplayBinding,
        ):
            raise ValueError("base replay binding is invalid")
        if self.builder_mode in ANCHORED_BUILDER_MODES:
            if self.base_replay is None or self.anchored_catalog_sha256 is None:
                raise ValueError("anchored builder requires a sealed catalog binding")
            _sha256(self.anchored_catalog_sha256, "anchored catalog sha256")
        elif self.anchored_catalog_sha256 is not None:
            raise ValueError("anchored catalog binding requires anchored builder mode")
        for name in (
            "dataset_sha256",
            "problem_source_sha256",
            "evaluation_binding_sha256",
        ):
            _sha256(getattr(self, name), name)
        if not self.problem_ids or len(self.problem_ids) != len(set(self.problem_ids)):
            raise ValueError("problem ids must be nonempty and unique")
        if not isinstance(self.worker, ModelIdentity) or not isinstance(
            self.state_builder, ModelIdentity
        ):
            raise ValueError("action census model identities are invalid")
        for name in ("worker_temperature", "builder_temperature"):
            value = getattr(self, name)
            if type(value) is not float or value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        for name in ("worker_max_tokens", "builder_max_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("worker_seed", "builder_seed", "action_order_seed"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if (
            self.repetitions != 4
            or self.discovery_repetitions != (0, 1)
            or self.evaluation_repetitions != (2, 3)
        ):
            raise ValueError("action census requires the registered 2+2 split")
        combined = self.discovery_repetitions + self.evaluation_repetitions
        if tuple(sorted(combined)) != tuple(range(self.repetitions)):
            raise ValueError("repetition splits must partition all repetitions")
        if not self.discovery_repetitions or not self.evaluation_repetitions:
            raise ValueError("both repetition splits must be nonempty")

    def header(self) -> dict[str, object]:
        builder_prompt_version, _, _, _ = _builder_prompt(self.builder_mode)
        body: dict[str, object] = {
            "schema_version": ACTION_CENSUS_HEADER_VERSION,
            "record_type": "run_header",
            "run_contract_version": ACTION_CENSUS_CONTRACT_VERSION,
            "preregistration_id": self.preregistration_id,
            "stage": self.stage,
            "problem_source": {
                "dataset_sha256": self.dataset_sha256,
                "projection_schema_version": ACTION_CENSUS_PROBLEM_SOURCE_VERSION,
                "projection_sha256": self.problem_source_sha256,
                "evaluation_binding_version": (
                    ACTION_CENSUS_EVALUATION_BINDING_VERSION
                ),
                "evaluation_binding_sha256": self.evaluation_binding_sha256,
                "problem_ids": list(self.problem_ids),
            },
            "contracts": {
                "state": ACTION_CENSUS_STATE_VERSION,
                "action": ACTION_CENSUS_ACTION_VERSION,
                "reward": ACTION_CENSUS_REWARD_VERSION,
                "base_prompt": ACTION_CENSUS_BASE_PROMPT_VERSION,
                "state_builder_prompt": builder_prompt_version,
                "action_worker_prompt": ACTION_CENSUS_WORKER_PROMPT_VERSION,
                "schema_validation": SCHEMA_VALIDATION_STRICT_V1,
                "action_order": ACTION_CENSUS_ORDER_VERSION,
                "seed_policy": ACTION_CENSUS_SEED_VERSION,
                "analysis": ACTION_CENSUS_ANALYSIS_VERSION,
            },
            "prompt_bindings": prompt_bindings(self.builder_mode),
            "implementation_binding": implementation_binding(),
            "models": {
                "worker": self.worker.to_record(),
                "state_builder": self.state_builder.to_record(),
            },
            "settings": {
                "case_count": len(self.problem_ids),
                "worker_temperature": self.worker_temperature,
                "worker_max_tokens": self.worker_max_tokens,
                "worker_seed": self.worker_seed,
                "builder_temperature": self.builder_temperature,
                "builder_max_tokens": self.builder_max_tokens,
                "builder_seed": self.builder_seed,
                "builder_mode": self.builder_mode,
                "action_order_seed": self.action_order_seed,
                "repetitions": self.repetitions,
                "discovery_repetitions": list(self.discovery_repetitions),
                "evaluation_repetitions": list(self.evaluation_repetitions),
                "hosted_revision_targets": list(ACTION_TARGETS),
                "zero_call_control": {
                    "verb": "submit",
                    "target": "root",
                    "value": "frozen_base_draft_terminal_reward",
                },
                "base_mode": (
                    "sealed_replay" if self.base_replay is not None else "hosted"
                ),
            },
        }
        if self.base_replay is not None:
            body["base_replay"] = self.base_replay.to_record()
        if self.anchored_catalog_sha256 is not None:
            body["anchored_span_catalog"] = {
                "schema_version": ACTION_CENSUS_ANCHORED_CATALOG_BINDING_VERSION,
                "sha256": self.anchored_catalog_sha256,
            }
        body["run_contract_sha256"] = canonical_json_sha256(body)
        return body

    @property
    def sha256(self) -> str:
        return str(self.header()["run_contract_sha256"])


def build_action_census_contract(
    *,
    preregistration_id: str,
    stage: str,
    dataset_path: str | Path,
    problems: tuple[CensusProblem, ...],
    worker: ModelIdentity,
    state_builder: ModelIdentity,
    worker_temperature: float = 0.2,
    worker_max_tokens: int = 384,
    worker_seed: int = 440_000,
    builder_temperature: float = 0.0,
    builder_max_tokens: int = 384,
    builder_seed: int = 441_000,
    action_order_seed: int = 442_000,
    builder_mode: str = BUILDER_MODE_SEGMENT_DRAFT,
    base_replay: PreparedBaseReplay | None = None,
) -> ActionCensusContract:
    anchored_catalog_sha256 = None
    if builder_mode in ANCHORED_BUILDER_MODES:
        if base_replay is None:
            raise ValueError("anchored builder requires sealed base replay")
        anchored_catalog_sha256 = anchored_catalog_binding_sha256(
            problems,
            base_replay,
        )
    return ActionCensusContract(
        preregistration_id=preregistration_id,
        stage=stage,
        dataset_sha256=file_sha256(dataset_path),
        problem_source_sha256=problem_source_sha256(problems),
        evaluation_binding_sha256=evaluation_binding_sha256(problems),
        problem_ids=tuple(problem.problem_id for problem in problems),
        worker=worker,
        state_builder=state_builder,
        worker_temperature=worker_temperature,
        worker_max_tokens=worker_max_tokens,
        worker_seed=worker_seed,
        builder_temperature=builder_temperature,
        builder_max_tokens=builder_max_tokens,
        builder_seed=builder_seed,
        action_order_seed=action_order_seed,
        builder_mode=builder_mode,
        base_replay=None if base_replay is None else base_replay.binding,
        anchored_catalog_sha256=anchored_catalog_sha256,
    )


def _validate_anchored_catalog_binding(
    contract: ActionCensusContract,
    problems: tuple[CensusProblem, ...],
    base_replay: PreparedBaseReplay | None,
) -> None:
    if contract.builder_mode not in ANCHORED_BUILDER_MODES:
        return
    if base_replay is None:
        raise ValueError("anchored builder requires sealed base replay")
    if (
        anchored_catalog_binding_sha256(problems, base_replay)
        != contract.anchored_catalog_sha256
    ):
        raise ValueError("anchored span catalog differs from action census contract")


@dataclass(frozen=True, slots=True)
class LogicalCall:
    call_id: str
    seed: int
    request_sha256: str
    payload: Mapping[str, object] | None
    generation_error: bool
    error_type: str | None


def request_sha256(request: WorkerRequest) -> str:
    return canonical_json_sha256(request.to_record())


def _logical_call(
    *,
    client: OpenAICompatibleLLMClient,
    recorder: _AttemptRecorder,
    role: str,
    request: WorkerRequest,
    temperature: float,
    max_tokens: int,
    seed: int,
) -> LogicalCall:
    offset = recorder.count(role)
    client.attempt_observer = recorder.callback(role)
    error_type: str | None = None
    payload: Mapping[str, object] | None
    try:
        payload = client.complete_json(
            request.system,
            request.user,
            request.schema,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            schema_validation_mode=SCHEMA_VALIDATION_STRICT_V1,
        )
        if type(payload) is not dict:
            raise ValueError("model payload must be an object")
    except (LLMResponseError, TypeError, ValueError, OverflowError) as exc:
        payload = None
        error_type = type(exc).__name__
    call_ids = recorder.since(role, offset)
    if len(call_ids) != 1:
        raise RuntimeError("one logical call must produce exactly one call id")
    return LogicalCall(
        call_id=call_ids[0],
        seed=seed,
        request_sha256=request_sha256(request),
        payload=payload,
        generation_error=payload is None,
        error_type=error_type,
    )


def _candidate(call: LogicalCall) -> Candidate:
    if call.payload is None:
        return Candidate(
            reasoning="",
            answer=None,
            usable=False,
            error_type=call.error_type or "generation_error",
        )
    try:
        candidate = candidate_from_payload(call.payload)
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


def terminal_reward(candidate: Candidate, problem: CensusProblem) -> int:
    answer = candidate.answer if candidate.usable and candidate.answer else ""
    return int(check_answer(answer, problem.expected_answer).success)


@dataclass(frozen=True, slots=True)
class CensusCase:
    problem_id: str
    family: str
    base_reward: int
    base_usable: bool
    builder_usable: bool
    node_text_complete: bool
    node_text_distinct: bool
    action_rewards: Mapping[str, tuple[int, ...]]
    action_usable: Mapping[str, tuple[bool, ...]]

    def __post_init__(self) -> None:
        if tuple(self.action_rewards) != ACTION_TARGETS:
            raise ValueError("case rewards require every ordered action")
        if tuple(self.action_usable) != ACTION_TARGETS:
            raise ValueError("case usability requires every ordered action")
        lengths = {len(values) for values in self.action_rewards.values()}
        lengths.update(len(values) for values in self.action_usable.values())
        if len(lengths) != 1:
            raise ValueError("case action repetitions must have equal lengths")


def _action_mean(
    case: CensusCase,
    target: str,
    repetitions: tuple[int, ...],
) -> float:
    return mean(case.action_rewards[target][index] for index in repetitions)


def _best_target(scores: Mapping[str, float]) -> str:
    best = max(scores.values())
    return next(target for target in ACTION_TARGETS if scores[target] == best)


def _unique_best_target(scores: Mapping[str, float]) -> str | None:
    best = max(scores.values())
    winners = tuple(target for target in ACTION_TARGETS if scores[target] == best)
    return winners[0] if len(winners) == 1 else None


def _best_policy_option(scores: Mapping[str, float]) -> str:
    best = max(scores.values())
    return next(option for option in POLICY_OPTIONS if scores[option] == best)


def _bootstrap_interval(
    differences: tuple[float, ...],
    *,
    seed: int = 444_444,
    samples: int = 10_000,
) -> tuple[float, float]:
    if not differences:
        raise ValueError("bootstrap requires paired differences")
    rng = random.Random(seed)
    count = len(differences)
    estimates = sorted(
        mean(differences[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return (
        estimates[int(0.05 * samples)],
        estimates[min(samples - 1, int(0.95 * samples))],
    )


def _paired_sign_summary(differences: tuple[float, ...]) -> dict[str, object]:
    positive = sum(value > 0.0 for value in differences)
    negative = sum(value < 0.0 for value in differences)
    non_ties = positive + negative
    if non_ties == 0:
        p_value = 1.0
    else:
        p_value = sum(
            comb(non_ties, successes) for successes in range(positive, non_ties + 1)
        ) / (2**non_ties)
    return {
        "positive": positive,
        "negative": negative,
        "ties": len(differences) - non_ties,
        "one_sided_exact_p": p_value,
    }


def analyze_action_census(
    cases: tuple[CensusCase, ...],
    contract: ActionCensusContract,
) -> dict[str, object]:
    if not cases:
        raise ValueError("action census analysis requires cases")
    discovery = contract.discovery_repetitions
    evaluation = contract.evaluation_repetitions
    fixed_discovery = {
        target: mean(_action_mean(case, target, discovery) for case in cases)
        for target in ACTION_TARGETS
    }
    fixed_evaluation = {
        target: mean(_action_mean(case, target, evaluation) for case in cases)
        for target in ACTION_TARGETS
    }
    best_fixed = _best_target(fixed_discovery)
    base_accuracy = mean(case.base_reward for case in cases)
    protected_fixed_discovery = {
        "submit": base_accuracy,
        **fixed_discovery,
    }
    protected_fixed_evaluation = {
        "submit": base_accuracy,
        **fixed_evaluation,
    }
    best_protected_fixed = _best_policy_option(protected_fixed_discovery)
    selected_targets: list[str] = []
    selected_evaluation: list[float] = []
    protected_selected_options: list[str] = []
    protected_selected_evaluation: list[float] = []
    oracle_evaluation: list[float] = []
    protected_oracle_evaluation: list[float] = []
    heterogeneous = 0
    protected_heterogeneous = 0
    stable_heterogeneous = 0
    protected_stable_heterogeneous = 0
    stable_root_over_target = 0
    stable_target_over_root = 0
    unique_wins: Counter[str] = Counter({target: 0 for target in ACTION_TARGETS})
    root_over_target = 0
    target_over_root = 0
    paired_by_target: dict[str, Counter[str]] = {
        target: Counter(
            {
                "both": 0,
                "root_only": 0,
                "target_only": 0,
                "neither": 0,
            }
        )
        for target in ACTION_TARGETS[1:]
    }
    paired_base_action: dict[str, Counter[str]] = {
        target: Counter(
            {
                "both": 0,
                "base_only": 0,
                "action_only": 0,
                "neither": 0,
            }
        )
        for target in ACTION_TARGETS
    }
    stable_recovery = Counter({target: 0 for target in ACTION_TARGETS})
    stable_regression = Counter({target: 0 for target in ACTION_TARGETS})
    stable_recovery_states = 0
    stable_regression_states = 0
    agreement = Counter({target: 0 for target in ACTION_TARGETS})
    protected_agreement = Counter({option: 0 for option in POLICY_OPTIONS})
    protected_unique_wins = Counter({option: 0 for option in POLICY_OPTIONS})
    stable_strict_winners = Counter({target: 0 for target in ACTION_TARGETS})
    clean_targeted_recovery_states = 0

    for case in cases:
        discovery_scores = {
            target: _action_mean(case, target, discovery) for target in ACTION_TARGETS
        }
        evaluation_scores = {
            target: _action_mean(case, target, evaluation) for target in ACTION_TARGETS
        }
        protected_discovery_scores = {
            "submit": float(case.base_reward),
            **discovery_scores,
        }
        protected_evaluation_scores = {
            "submit": float(case.base_reward),
            **evaluation_scores,
        }
        discovery_strict_winner = _unique_best_target(discovery_scores)
        evaluation_strict_winner = _unique_best_target(evaluation_scores)
        stable_strict_winner = (
            discovery_strict_winner
            if discovery_strict_winner is not None
            and discovery_strict_winner == evaluation_strict_winner
            else None
        )
        if stable_strict_winner is not None:
            stable_strict_winners[stable_strict_winner] += 1
            clean_targeted_recovery_states += int(
                stable_strict_winner != "root"
                and case.base_reward == 0
                and case.base_usable
                and all(all(case.action_usable[target]) for target in ACTION_TARGETS)
                and case.action_rewards["root"] == (0, 0, 0, 0)
                and case.action_rewards[stable_strict_winner] == (1, 1, 1, 1)
            )
        selected = _best_target(discovery_scores)
        protected_selected = _best_policy_option(protected_discovery_scores)
        selected_targets.append(selected)
        protected_selected_options.append(protected_selected)
        selected_evaluation.append(evaluation_scores[selected])
        protected_selected_evaluation.append(
            protected_evaluation_scores[protected_selected]
        )
        oracle_evaluation.append(max(evaluation_scores.values()))
        protected_oracle_evaluation.append(max(protected_evaluation_scores.values()))
        heterogeneous += int(len(set(evaluation_scores.values())) > 1)
        protected_heterogeneous += int(
            len(set(protected_evaluation_scores.values())) > 1
        )
        discovery_vector = tuple(discovery_scores[target] for target in ACTION_TARGETS)
        evaluation_vector = tuple(
            evaluation_scores[target] for target in ACTION_TARGETS
        )
        protected_discovery_vector = tuple(
            protected_discovery_scores[option] for option in POLICY_OPTIONS
        )
        protected_evaluation_vector = tuple(
            protected_evaluation_scores[option] for option in POLICY_OPTIONS
        )
        vector_stable = discovery_vector == evaluation_vector
        protected_vector_stable = (
            protected_discovery_vector == protected_evaluation_vector
        )
        vector_heterogeneous = len(set(evaluation_vector)) > 1
        protected_vector_heterogeneous = len(set(protected_evaluation_vector)) > 1
        stable_heterogeneous += int(vector_stable and vector_heterogeneous)
        protected_stable_heterogeneous += int(
            protected_vector_stable and protected_vector_heterogeneous
        )
        for target in ACTION_TARGETS:
            agreement[target] += int(
                discovery_scores[target] == evaluation_scores[target]
            )
            stable_recovery[target] += int(
                case.base_reward == 0
                and discovery_scores[target] == 1.0
                and evaluation_scores[target] == 1.0
            )
            stable_regression[target] += int(
                case.base_reward == 1
                and discovery_scores[target] == 0.0
                and evaluation_scores[target] == 0.0
            )
            if evaluation_scores[target] == 1.0 and all(
                evaluation_scores[other] == 0.0
                for other in ACTION_TARGETS
                if other != target
            ):
                unique_wins[target] += 1
            for index in evaluation:
                action_reward = case.action_rewards[target][index]
                cell = (
                    "both"
                    if case.base_reward and action_reward
                    else (
                        "base_only"
                        if case.base_reward
                        else "action_only" if action_reward else "neither"
                    )
                )
                paired_base_action[target][cell] += 1
        stable_recovery_states += int(
            case.base_reward == 0
            and any(
                discovery_scores[target] == 1.0 and evaluation_scores[target] == 1.0
                for target in ACTION_TARGETS
            )
        )
        stable_regression_states += int(
            case.base_reward == 1
            and any(
                discovery_scores[target] == 0.0 and evaluation_scores[target] == 0.0
                for target in ACTION_TARGETS
            )
        )
        for option in POLICY_OPTIONS:
            protected_agreement[option] += int(
                protected_discovery_scores[option]
                == protected_evaluation_scores[option]
            )
            if protected_evaluation_scores[option] == 1.0 and all(
                protected_evaluation_scores[other] == 0.0
                for other in POLICY_OPTIONS
                if other != option
            ):
                protected_unique_wins[option] += 1
        root_score = evaluation_scores["root"]
        node_scores = [evaluation_scores[target] for target in ACTION_TARGETS[1:]]
        root_over_target += int(root_score == 1.0 and min(node_scores) == 0.0)
        target_over_root += int(root_score == 0.0 and max(node_scores) == 1.0)
        stable_root_over_target += int(
            vector_stable and root_score == 1.0 and min(node_scores) == 0.0
        )
        stable_target_over_root += int(
            vector_stable and root_score == 0.0 and max(node_scores) == 1.0
        )
        for target in ACTION_TARGETS[1:]:
            for index in evaluation:
                root_reward = case.action_rewards["root"][index]
                node_reward = case.action_rewards[target][index]
                cell = (
                    "both"
                    if root_reward and node_reward
                    else (
                        "root_only"
                        if root_reward
                        else "target_only" if node_reward else "neither"
                    )
                )
                paired_by_target[target][cell] += 1

    best_fixed_evaluation = fixed_evaluation[best_fixed]
    best_protected_fixed_evaluation = protected_fixed_evaluation[best_protected_fixed]
    repeatability_evaluation = mean(selected_evaluation)
    protected_repeatability_evaluation = mean(protected_selected_evaluation)
    differences = tuple(
        selected - _action_mean(case, best_fixed, evaluation)
        for case, selected in zip(cases, selected_evaluation, strict=True)
    )
    protected_differences = tuple(
        selected
        - (
            float(case.base_reward)
            if best_protected_fixed == "submit"
            else _action_mean(case, best_protected_fixed, evaluation)
        )
        for case, selected in zip(cases, protected_selected_evaluation, strict=True)
    )
    ci_lower, ci_upper = _bootstrap_interval(differences)
    protected_ci_lower, protected_ci_upper = _bootstrap_interval(protected_differences)
    family_results: dict[str, dict[str, object]] = {}
    for family in sorted({case.family for case in cases}):
        subset = tuple(case for case in cases if case.family == family)
        family_results[family] = {
            "count": len(subset),
            "base_solved": sum(case.base_reward for case in subset),
            "fixed_evaluation": {
                target: mean(_action_mean(case, target, evaluation) for case in subset)
                for target in ACTION_TARGETS
            },
        }
    return {
        "schema_version": ACTION_CENSUS_ANALYSIS_VERSION,
        "case_count": len(cases),
        "base_draft_accuracy": base_accuracy,
        "base_usable_count": sum(int(case.base_usable) for case in cases),
        "builder_usable_count": sum(int(case.builder_usable) for case in cases),
        "action_total_usable_counts": {
            target: sum(
                int(value) for case in cases for value in case.action_usable[target]
            )
            for target in ACTION_TARGETS
        },
        "action_evaluation_usable_counts": {
            target: sum(
                int(case.action_usable[target][index])
                for case in cases
                for index in evaluation
            )
            for target in ACTION_TARGETS
        },
        "fixed_revision_discovery": fixed_discovery,
        "fixed_revision_evaluation": fixed_evaluation,
        "best_fixed_revision_action": best_fixed,
        "best_fixed_revision_evaluation": best_fixed_evaluation,
        "protected_fixed_discovery": protected_fixed_discovery,
        "protected_fixed_evaluation": protected_fixed_evaluation,
        "best_protected_fixed_policy": best_protected_fixed,
        "best_protected_fixed_evaluation": best_protected_fixed_evaluation,
        "node_text_complete_count": sum(int(case.node_text_complete) for case in cases),
        "node_text_distinct_count": sum(int(case.node_text_distinct) for case in cases),
        "revision_repeatability_selected_action_counts": dict(
            sorted(Counter(selected_targets).items())
        ),
        "revision_repeatability_selected_evaluation": repeatability_evaluation,
        "revision_repeatability_minus_best_fixed_revision": (
            repeatability_evaluation - best_fixed_evaluation
        ),
        "revision_repeatability_minus_best_fixed_revision_bootstrap_90_ci": [
            ci_lower,
            ci_upper,
        ],
        "revision_repeatability_paired_sign": _paired_sign_summary(differences),
        "protected_repeatability_selected_action_counts": dict(
            sorted(Counter(protected_selected_options).items())
        ),
        "protected_repeatability_selected_evaluation": (
            protected_repeatability_evaluation
        ),
        "protected_repeatability_minus_best_protected_fixed": (
            protected_repeatability_evaluation - best_protected_fixed_evaluation
        ),
        "protected_repeatability_minus_best_protected_fixed_bootstrap_90_ci": [
            protected_ci_lower,
            protected_ci_upper,
        ],
        "protected_repeatability_paired_sign": _paired_sign_summary(
            protected_differences
        ),
        "repeatability_estimand": (
            "within-state cross-replicate outcome repeatability oracle, not a "
            "state-to-action policy or a held-out semantic selector"
        ),
        "descriptive_realized_revision_oracle": mean(oracle_evaluation),
        "descriptive_realized_protected_oracle": mean(protected_oracle_evaluation),
        "heterogeneous_revision_state_count": heterogeneous,
        "protected_heterogeneous_state_count": protected_heterogeneous,
        "stable_heterogeneous_revision_state_count": stable_heterogeneous,
        "protected_stable_heterogeneous_state_count": (protected_stable_heterogeneous),
        "stable_root_over_target_state_count": stable_root_over_target,
        "stable_target_over_root_state_count": stable_target_over_root,
        "revision_unique_win_counts": dict(unique_wins),
        "stable_strict_winner_counts": dict(stable_strict_winners),
        "stable_strict_winner_state_count": sum(stable_strict_winners.values()),
        "stable_strict_winner_target_count": sum(
            count > 0 for count in stable_strict_winners.values()
        ),
        "clean_targeted_recovery_state_count": clean_targeted_recovery_states,
        "protected_unique_win_counts": dict(protected_unique_wins),
        "root_over_target_state_count": root_over_target,
        "target_over_root_state_count": target_over_root,
        "stable_recovery_counts": dict(stable_recovery),
        "stable_regression_counts": dict(stable_regression),
        "stable_recovery_state_count": stable_recovery_states,
        "stable_regression_state_count": stable_regression_states,
        "revision_discovery_evaluation_agreement_counts": dict(agreement),
        "protected_discovery_evaluation_agreement_counts": dict(protected_agreement),
        "paired_root_target_evaluation_replicates": {
            target: dict(sorted(counts.items()))
            for target, counts in paired_by_target.items()
        },
        "paired_base_action_evaluation_replicates": {
            target: dict(sorted(counts.items()))
            for target, counts in paired_base_action.items()
        },
        "family_results": family_results,
        "behavior_cloning_eligible": False,
    }


def _write_record(
    handle: TextIO,
    record: Mapping[str, object],
    *,
    flush: bool = False,
) -> None:
    encoded = canonical_json_bytes(dict(record))
    for forbidden in (b'"expected_answer"', b'"solution_set"', b'"params"'):
        if forbidden in encoded:
            raise ValueError("published action census record leaked a hidden field")
    handle.write(encoded.decode("ascii") + "\n")
    if flush:
        handle.flush()


@dataclass(frozen=True, slots=True)
class ActionCensusResult:
    run_path: Path
    diagnostic_path: Path
    case_count: int
    logical_call_count: int
    transition_count: int
    run_contract_sha256: str
    analysis: Mapping[str, object]


def _load_jsonl(path: str | Path, name: str) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    with Path(path).open("r", encoding="ascii", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
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
    return tuple(records)


def prepare_base_replay(
    *,
    source_run_path: str | Path,
    source_diagnostic_path: str | Path,
    expected_source_run_sha256: str,
    expected_source_diagnostic_sha256: str,
    expected_source_run_contract_sha256: str,
    problems: tuple[CensusProblem, ...],
    worker: ModelIdentity,
    worker_temperature: float,
    worker_max_tokens: int,
    worker_seed: int,
) -> PreparedBaseReplay:
    """Bind successful base Candidates from a sealed action-census run."""
    _sha256(expected_source_run_sha256, "expected base source run sha256")
    _sha256(
        expected_source_diagnostic_sha256,
        "expected base source diagnostic sha256",
    )
    _sha256(
        expected_source_run_contract_sha256,
        "expected base source contract sha256",
    )
    if not problems:
        raise ValueError("base replay requires problems")
    records = _load_jsonl(source_run_path, "base replay source")
    diagnostic_records = _load_jsonl(
        source_diagnostic_path,
        "base replay source diagnostics",
    )
    if file_sha256(source_run_path) != expected_source_run_sha256:
        raise ValueError("base replay source run hash differs")
    if file_sha256(source_diagnostic_path) != expected_source_diagnostic_sha256:
        raise ValueError("base replay source diagnostic hash differs")
    header = records[0]
    source = header.get("problem_source")
    settings = header.get("settings")
    models = header.get("models")
    bindings = header.get("prompt_bindings")
    if (
        header.get("record_type") != "run_header"
        or header.get("schema_version")
        not in {ACTION_CENSUS_LEGACY_HEADER_VERSION, ACTION_CENSUS_HEADER_VERSION}
        or header.get("run_contract_sha256") != expected_source_run_contract_sha256
        or not isinstance(source, Mapping)
        or not isinstance(settings, Mapping)
        or not isinstance(models, Mapping)
        or not isinstance(bindings, Mapping)
    ):
        raise ValueError("base replay source header differs")
    diagnostic_header = diagnostic_records[0]
    if (
        diagnostic_header.get("record_type") != "diagnostic_header"
        or diagnostic_header.get("run_contract_sha256")
        != expected_source_run_contract_sha256
    ):
        raise ValueError("base replay diagnostic header differs")
    if (
        tuple(source.get("problem_ids", ()))
        != tuple(problem.problem_id for problem in problems)
        or source.get("projection_sha256") != problem_source_sha256(problems)
        or source.get("evaluation_binding_sha256")
        != evaluation_binding_sha256(problems)
        or models.get("worker") != worker.to_record()
        or bindings.get("base") != prompt_bindings(BUILDER_MODE_SEGMENT_DRAFT)["base"]
        or settings.get("worker_temperature") != worker_temperature
        or settings.get("worker_max_tokens") != worker_max_tokens
        or settings.get("worker_seed") != worker_seed
    ):
        raise ValueError("base replay source contract is incompatible")

    base_rows = tuple(
        record
        for record in records
        if record.get("record_type") == "logical_call"
        and record.get("phase") == "base_draft"
    )
    if len(base_rows) != len(problems):
        raise ValueError("base replay source does not cover every problem")
    replay_cases: list[BaseReplayCase] = []
    for problem_index, problem in enumerate(problems):
        matches = tuple(
            record
            for record in base_rows
            if record.get("problem_id") == problem.problem_id
        )
        if len(matches) != 1:
            raise ValueError("base replay source problem is not unique")
        record = matches[0]
        expected_request = base_request(problem)
        candidate_record = record.get("candidate")
        if (
            record.get("problem_index") != problem_index
            or record.get("role") != "base_worker"
            or record.get("seed") != worker_seed + problem_index * 100
            or record.get("request") != expected_request.to_record()
            or record.get("request_sha256") != request_sha256(expected_request)
            or not isinstance(candidate_record, Mapping)
            or record.get("generation_error") is not False
            or record.get("error_type") is not None
            or not isinstance(record.get("call_id"), str)
        ):
            raise ValueError("base replay source call differs")
        try:
            candidate = Candidate(**dict(candidate_record))
        except (TypeError, ValueError) as exc:
            raise ValueError("base replay source Candidate is invalid") from exc
        if not candidate.usable:
            raise ValueError("base replay source Candidate must be usable")
        source_call_id = str(record["call_id"])
        source_attempts = tuple(
            diagnostic
            for diagnostic in diagnostic_records[1:]
            if isinstance(diagnostic.get("attempt"), Mapping)
            and diagnostic["attempt"].get("call_id") == source_call_id
        )
        if not source_attempts:
            raise ValueError("base replay source call lacks diagnostics")
        last_attempt = source_attempts[-1]["attempt"]
        assert isinstance(last_attempt, Mapping)
        if (
            last_attempt.get("outcome") != "accepted"
            or last_attempt.get("disposition") != "return"
        ):
            raise ValueError("base replay source call did not succeed")
        replay_cases.append(
            BaseReplayCase(
                problem_id=problem.problem_id,
                problem_index=problem_index,
                source_call_id=source_call_id,
                source_record_sha256=canonical_json_sha256(record),
                seed=worker_seed + problem_index * 100,
                request_sha256=request_sha256(expected_request),
                candidate=candidate,
                generation_error=False,
                error_type=None,
            )
        )
    projection_sha256 = canonical_json_sha256(
        {
            "schema_version": ACTION_CENSUS_BASE_REPLAY_BINDING_VERSION,
            "cases": [case.projection_record() for case in replay_cases],
        }
    )
    return PreparedBaseReplay(
        binding=BaseReplayBinding(
            source_run_sha256=expected_source_run_sha256,
            source_diagnostic_sha256=expected_source_diagnostic_sha256,
            source_run_contract_sha256=expected_source_run_contract_sha256,
            projection_sha256=projection_sha256,
        ),
        cases=tuple(replay_cases),
    )


def audit_action_census_artifacts(
    *,
    run_path: str | Path,
    diagnostic_path: str | Path,
    contract: ActionCensusContract,
    problems: tuple[CensusProblem, ...],
    base_replay: PreparedBaseReplay | None = None,
) -> dict[str, object]:
    """Replay sealed records and fail on any call, state, or reward mismatch."""
    run_records = _load_jsonl(run_path, "action census run")
    diagnostic_records = _load_jsonl(diagnostic_path, "action census diagnostic")
    expected_case_count = len(problems)
    expected_transition_count = (
        expected_case_count * len(ACTION_TARGETS) * (contract.repetitions)
    )
    hosted_base = base_replay is None
    expected_call_count = expected_case_count * (
        int(hosted_base) + 1 + len(ACTION_TARGETS) * contract.repetitions
    )
    expected_run_count = 2 + expected_case_count * (
        4 + len(ACTION_TARGETS) * contract.repetitions
    )
    if (
        tuple(problem.problem_id for problem in problems) != contract.problem_ids
        or problem_source_sha256(problems) != contract.problem_source_sha256
        or evaluation_binding_sha256(problems) != contract.evaluation_binding_sha256
    ):
        raise ValueError("action census audit inputs differ from contract")
    if (base_replay is None) != (contract.base_replay is None):
        raise ValueError("action census audit base replay presence differs")
    if base_replay is not None and (
        base_replay.binding != contract.base_replay
        or tuple(case.problem_id for case in base_replay.cases) != contract.problem_ids
    ):
        raise ValueError("action census audit base replay differs")
    _validate_anchored_catalog_binding(contract, problems, base_replay)
    if len(run_records) != expected_run_count:
        raise ValueError("action census run record count differs")
    if not run_records or run_records[0] != contract.header():
        raise ValueError("action census run header differs from contract")
    expected_diagnostic_header = {
        "schema_version": ACTION_CENSUS_DIAGNOSTIC_VERSION,
        "record_type": "diagnostic_header",
        "preregistration_id": contract.preregistration_id,
        "run_contract_sha256": contract.sha256,
        "roles": (
            ["base_worker", "state_builder", "action_worker"]
            if hosted_base
            else ["state_builder", "action_worker"]
        ),
    }
    if not diagnostic_records or diagnostic_records[0] != expected_diagnostic_header:
        raise ValueError("action census diagnostic header differs")

    expected_record_types = ["run_header"]
    for _ in problems:
        expected_record_types.extend(
            (
                "logical_call" if hosted_base else "base_replay",
                "logical_call",
                "census_state",
                *(
                    "action_transition"
                    for _ in range(len(ACTION_TARGETS) * contract.repetitions)
                ),
                "case_summary",
            )
        )
    expected_record_types.append("run_summary")
    if [record.get("record_type") for record in run_records] != (expected_record_types):
        raise ValueError("action census canonical record order differs")
    expected_schemas = {
        "logical_call": ACTION_CENSUS_CALL_VERSION,
        "base_replay": ACTION_CENSUS_BASE_REPLAY_VERSION,
        "census_state": ACTION_CENSUS_STATE_RECORD_VERSION,
        "action_transition": ACTION_CENSUS_TRANSITION_VERSION,
        "case_summary": ACTION_CENSUS_CASE_VERSION,
        "run_summary": ACTION_CENSUS_SUMMARY_VERSION,
    }
    expected_fields = {
        "logical_call": {
            "schema_version",
            "record_type",
            "preregistration_id",
            "problem_index",
            "problem_id",
            "phase",
            "role",
            "seed",
            "request_sha256",
            "request",
            "candidate",
            "generation_error",
            "error_type",
            "call_id",
        },
        "base_replay": {
            "schema_version",
            "record_type",
            "preregistration_id",
            "problem_index",
            "problem_id",
            "source_call_id",
            "source_record_sha256",
            "seed",
            "request_sha256",
            "request",
            "candidate",
            "candidate_sha256",
            "generation_error",
            "error_type",
        },
        "census_state": {
            "schema_version",
            "record_type",
            "preregistration_id",
            "problem_index",
            "problem_id",
            "state_sha256",
            "builder_usable",
            "builder_error_type",
            "node_text_complete",
            "node_text_distinct",
            "observation",
        },
        "action_transition": {
            "schema_version",
            "record_type",
            "preregistration_id",
            "problem_index",
            "problem_id",
            "state_sha256",
            "action",
            "repetition",
            "order_index",
            "seed",
            "request_sha256",
            "request",
            "candidate",
            "generation_error",
            "error_type",
            "worker_call_id",
            "reward",
        },
        "case_summary": {
            "schema_version",
            "record_type",
            "preregistration_id",
            "problem_index",
            "problem_id",
            "state_sha256",
            "base_reward",
            "base_usable",
            "builder_usable",
            "node_text_complete",
            "node_text_distinct",
            "action_orders",
            "action_rewards",
            "action_usable",
        },
        "run_summary": {
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
        },
    }
    for record in run_records[1:]:
        record_type = record.get("record_type")
        if (
            record.get("schema_version") != expected_schemas.get(record_type)
            or record.get("preregistration_id") != contract.preregistration_id
            or set(record) != expected_fields.get(record_type)
        ):
            raise ValueError("action census canonical record metadata differs")
    for sequence, record in enumerate(diagnostic_records[1:]):
        if (
            set(record)
            != {"schema_version", "record_type", "sequence", "role", "attempt"}
            or record.get("schema_version") != DIAGNOSTIC_ATTEMPT_VERSION
            or record.get("record_type") != "http_attempt"
            or record.get("sequence") != sequence
        ):
            raise ValueError("action census diagnostic sequence differs")

    logical = tuple(
        record for record in run_records if record.get("record_type") == "logical_call"
    )
    base_replays = tuple(
        record for record in run_records if record.get("record_type") == "base_replay"
    )
    transitions = tuple(
        record
        for record in run_records
        if record.get("record_type") == "action_transition"
    )
    states = tuple(
        record for record in run_records if record.get("record_type") == "census_state"
    )
    summaries = tuple(
        record for record in run_records if record.get("record_type") == "case_summary"
    )
    run_summaries = tuple(
        record for record in run_records if record.get("record_type") == "run_summary"
    )
    if (
        len(logical) != (1 + int(hosted_base)) * expected_case_count
        or len(base_replays) != (0 if hosted_base else expected_case_count)
        or len(transitions) != expected_transition_count
        or len(states) != expected_case_count
        or len(summaries) != expected_case_count
        or len(run_summaries) != 1
    ):
        raise ValueError("action census record strata differ")

    canonical_call_id_values = [record.get("call_id") for record in logical] + [
        record.get("worker_call_id") for record in transitions
    ]
    if any(not isinstance(call_id, str) for call_id in canonical_call_id_values):
        raise ValueError("action census canonical call id is invalid")
    canonical_call_ids = set(canonical_call_id_values)
    if len(canonical_call_ids) != expected_call_count:
        raise ValueError("action census canonical call ids are duplicated")

    attempts_by_call: dict[str, list[tuple[str, Mapping[str, object]]]] = {}
    for record in diagnostic_records:
        if record.get("record_type") != "http_attempt":
            continue
        role = record.get("role")
        attempt = record.get("attempt")
        if not isinstance(role, str) or not isinstance(attempt, Mapping):
            raise ValueError("action census diagnostic attempt is malformed")
        call_id = attempt.get("call_id")
        if not isinstance(call_id, str):
            raise ValueError("action census diagnostic call id is invalid")
        attempts_by_call.setdefault(call_id, []).append((role, attempt))
    if (
        set(attempts_by_call) != canonical_call_ids
        or len(attempts_by_call) != expected_call_count
    ):
        raise ValueError("action census diagnostic call coverage differs")
    canonical_call_order = [
        (
            record.get("call_id")
            if record.get("record_type") == "logical_call"
            else record.get("worker_call_id")
        )
        for record in run_records
        if record.get("record_type") in {"logical_call", "action_transition"}
    ]
    diagnostic_call_order: list[str] = []
    for record in diagnostic_records[1:]:
        attempt = record.get("attempt")
        assert isinstance(attempt, Mapping)
        call_id = attempt.get("call_id")
        assert isinstance(call_id, str)
        if not diagnostic_call_order or diagnostic_call_order[-1] != call_id:
            diagnostic_call_order.append(call_id)
    if diagnostic_call_order != canonical_call_order:
        raise ValueError("action census diagnostic call order differs")

    replayed_usage = {
        role: Counter(
            {
                "generate_calls": 0,
                "prompt_tokens": 0,
                "generated_tokens": 0,
            }
        )
        for role in ("base_worker", "state_builder", "action_worker")
    }
    observed_response_models: dict[str, set[str]] = {
        role: set() for role in ("base_worker", "state_builder", "action_worker")
    }
    replayed_attempt_counts = Counter(
        {role: 0 for role in ("base_worker", "state_builder", "action_worker")}
    )
    first_attempt_accepted_counts = Counter(
        {role: 0 for role in ("base_worker", "state_builder", "action_worker")}
    )
    attempt_outcome_counts: dict[str, Counter[str]] = {
        role: Counter() for role in ("base_worker", "state_builder", "action_worker")
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

    def replay_wire_call(
        *,
        call_id: str,
        role: str,
        request: WorkerRequest,
        identity: ModelIdentity,
        temperature: float,
        max_tokens: int,
        seed: int,
        generation_error: bool,
    ) -> Mapping[str, object] | None:
        attempts = attempts_by_call[call_id]
        if any(observed_role != role for observed_role, _ in attempts):
            raise ValueError("action census diagnostic role differs")
        expected_body = build_openai_compatible_json_body(
            model=identity.model,
            system=request.system,
            user=request.user,
            schema=request.schema,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            guided_json=identity.guided_json,
        )
        parse_retry_index = 0
        backoff_retry_index = 0
        terminal_payload: Mapping[str, object] | None = None
        terminal_disposition: str | None = None
        for attempt_index, (_, attempt) in enumerate(attempts):
            replayed_attempt_counts[role] += 1
            if set(attempt) != attempt_fields:
                raise ValueError("action census diagnostic attempt fields differ")
            effective_seed = seed + parse_retry_index
            if (
                attempt.get("call_id") != call_id
                or attempt.get("attempt_index") != attempt_index
                or attempt.get("parse_retry_index") != parse_retry_index
                or attempt.get("backoff_retry_index") != backoff_retry_index
                or attempt.get("effective_seed") != effective_seed
            ):
                raise ValueError("action census diagnostic retry indices differ")
            expected_body["seed"] = effective_seed
            request_body_json = attempt.get("request_body_json")
            if not isinstance(request_body_json, str):
                raise ValueError("action census attempt lacks request body")
            observed_body = json.loads(
                request_body_json,
                object_pairs_hook=_pairs_object,
                parse_constant=_reject_constant,
            )
            if (
                observed_body != expected_body
                or request_body_json
                != canonical_json_bytes(observed_body).decode("ascii")
            ):
                raise ValueError("action census diagnostic request body differs")

            outcome = attempt.get("outcome")
            disposition = attempt.get("disposition")
            if isinstance(outcome, str) and isinstance(disposition, str):
                attempt_outcome_counts[role][f"{outcome}:{disposition}"] += 1
                if (
                    attempt_index == 0
                    and outcome == "accepted"
                    and disposition == "return"
                ):
                    first_attempt_accepted_counts[role] += 1
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
                    raise ValueError("action census successful HTTP status differs")
                replayed_usage[role]["generate_calls"] += 1
                for field_name, counter_name in (
                    ("prompt_tokens", "prompt_tokens"),
                    ("completion_tokens", "generated_tokens"),
                ):
                    value = attempt.get(field_name)
                    if value is not None and (type(value) is not int or value < 0):
                        raise ValueError("action census token count is invalid")
                    replayed_usage[role][counter_name] += value or 0
                total_tokens = attempt.get("total_tokens")
                if total_tokens is not None and (
                    type(total_tokens) is not int or total_tokens < 0
                ):
                    raise ValueError("action census total token count is invalid")
            elif outcome == "http_status_error":
                if type(http_status) is not int or 200 <= http_status < 300:
                    raise ValueError("action census HTTP error status differs")
            elif outcome == "transport_error":
                if http_status is not None:
                    raise ValueError("action census transport status differs")
            else:
                raise ValueError("action census attempt outcome is invalid")
            if outcome == "accepted":
                if error_type is not None or error_message is not None:
                    raise ValueError("accepted action census attempt has an error")
            elif (
                not isinstance(error_type, str)
                or not error_type
                or not isinstance(error_message, str)
                or not error_message
            ):
                raise ValueError("failed action census attempt lacks its error")
            if outcome == "http_status_error" and error_message != (
                f"HTTP status {http_status}"
            ):
                raise ValueError("action census HTTP error message differs")
            if outcome in {"parse_error", "schema_error"} and error_type != (
                "LLMResponseError"
            ):
                raise ValueError("action census content error type differs")

            response_payload: object | None = None
            if raw_response_body is not None:
                if not isinstance(raw_response_body, str):
                    raise ValueError("action census raw response is invalid")
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
                    observed_response_models[role].add(response_model)
            if successful_response and outcome != "provider_response_error":
                if not isinstance(response_payload, Mapping):
                    raise ValueError(
                        "action census provider response is not replayable"
                    )
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
                    raise ValueError("action census provider projection differs")
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
                            "action census provider error projection differs"
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
                    raise ValueError("action census content outcome lacks text")
                classified, payload = classify_openai_compatible_json_content(
                    raw_content,
                    request.schema,
                    schema_validation_mode=SCHEMA_VALIDATION_STRICT_V1,
                )
                if classified != outcome:
                    raise ValueError("action census content outcome differs")
                if outcome == "accepted":
                    terminal_payload = payload

            if disposition == "backoff_retry":
                if (
                    outcome != "http_status_error"
                    or not (http_status == 429 or http_status >= 500)
                    or backoff_retry_index >= identity.backoff_retries
                ):
                    raise ValueError("action census backoff retry is invalid")
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
                    or parse_retry_index >= identity.parse_retries
                ):
                    raise ValueError("action census parse retry is invalid")
                append_openai_compatible_parse_retry(
                    expected_body,
                    raw_content if isinstance(raw_content, str) else None,
                )
                parse_retry_index += 1
                continue
            if disposition not in {"return", "raise"}:
                raise ValueError("action census terminal disposition is invalid")
            if attempt_index != len(attempts) - 1:
                raise ValueError("action census terminal attempt is not last")
            if disposition == "return" and outcome != "accepted":
                raise ValueError("action census return attempt was not accepted")
            if disposition == "raise" and outcome == "accepted":
                raise ValueError("action census accepted attempt was raised")
            if (
                disposition == "raise"
                and outcome
                in {
                    "transport_error",
                    "provider_response_error",
                    "parse_error",
                    "schema_error",
                }
                and parse_retry_index < identity.parse_retries
            ):
                raise ValueError("action census parse-retryable error stopped early")
            if (
                disposition == "raise"
                and outcome == "http_status_error"
                and (http_status == 429 or http_status >= 500)
                and backoff_retry_index < identity.backoff_retries
            ):
                raise ValueError("action census retryable HTTP error stopped early")
            terminal_disposition = disposition

        if terminal_disposition is None:
            raise ValueError("action census call lacks a terminal attempt")
        if generation_error != (terminal_disposition == "raise"):
            raise ValueError("action census generation error differs from attempts")
        if generation_error:
            return None
        if terminal_payload is None:
            raise ValueError("successful action census call lacks a payload")
        return terminal_payload

    for record in (*logical, *base_replays, *transitions):
        request = record.get("request")
        if not isinstance(request, Mapping):
            raise ValueError("action census record lacks a request")
        if canonical_json_sha256(dict(request)) != record.get("request_sha256"):
            raise ValueError("action census request hash differs")

    problem_by_id = {problem.problem_id: problem for problem in problems}
    logical_by_key: dict[tuple[str, str], Mapping[str, object]] = {}
    for record in logical:
        key = (str(record.get("problem_id")), str(record.get("phase")))
        if key in logical_by_key:
            raise ValueError("action census logical call key is duplicated")
        logical_by_key[key] = record
    base_replay_by_id = {
        str(record.get("problem_id")): record for record in base_replays
    }
    if len(base_replay_by_id) != len(base_replays):
        raise ValueError("action census base replay records are duplicated")
    transition_by_key: dict[tuple[str, str, int], Mapping[str, object]] = {}
    for record in transitions:
        action = record.get("action")
        if not isinstance(action, Mapping):
            raise ValueError("action census transition action is malformed")
        repetition = record.get("repetition")
        if type(repetition) is not int:
            raise ValueError("action census transition repetition is invalid")
        key = (
            str(record.get("problem_id")),
            str(action.get("target")),
            repetition,
        )
        if key in transition_by_key:
            raise ValueError("action census transition key is duplicated")
        transition_by_key[key] = record
    state_by_id = {str(record.get("problem_id")): record for record in states}
    summary_by_id = {str(record.get("problem_id")): record for record in summaries}
    if (
        len(state_by_id) != expected_case_count
        or len(summary_by_id) != expected_case_count
    ):
        raise ValueError("action census problem records are duplicated")

    reconstructed_states: dict[str, CensusState] = {}
    reconstructed_base: dict[str, Candidate] = {}
    reconstructed_builder_flags: dict[str, tuple[bool, bool, bool]] = {}
    reconstructed_candidates: dict[tuple[str, str, int], Candidate] = {}
    reconstructed_call_failures = Counter(
        {
            "base_worker": 0,
            "state_builder": 0,
            "action_worker": 0,
        }
    )
    reconstructed_unusable_outputs = Counter(
        {
            "base_worker": 0,
            "state_builder": 0,
            "action_worker": 0,
        }
    )
    for problem_index, problem in enumerate(problems):
        expected_base_request = base_request(problem)
        expected_base_seed = contract.worker_seed + problem_index * 100
        if hosted_base:
            base_row = logical_by_key.get((problem.problem_id, "base_draft"))
            if base_row is None:
                raise ValueError("action census base call is missing")
            if (
                base_row.get("problem_index") != problem_index
                or base_row.get("role") != "base_worker"
                or base_row.get("seed") != expected_base_seed
                or base_row.get("request") != expected_base_request.to_record()
            ):
                raise ValueError("action census base call contract differs")
            base_call_id = base_row.get("call_id")
            if not isinstance(base_call_id, str):
                raise ValueError("action census base call id is invalid")
            base_generation_error = base_row.get("generation_error")
            if type(base_generation_error) is not bool:
                raise ValueError("action census base error flag is invalid")
            base_error_type = "LLMResponseError" if base_generation_error else None
            if base_row.get("error_type") != base_error_type:
                raise ValueError("action census base error type differs")
            base_payload = replay_wire_call(
                call_id=base_call_id,
                role="base_worker",
                request=expected_base_request,
                identity=contract.worker,
                temperature=contract.worker_temperature,
                max_tokens=contract.worker_max_tokens,
                seed=expected_base_seed,
                generation_error=base_generation_error,
            )
            base_candidate = _candidate(
                LogicalCall(
                    call_id=base_call_id,
                    seed=expected_base_seed,
                    request_sha256=request_sha256(expected_base_request),
                    payload=base_payload,
                    generation_error=base_generation_error,
                    error_type=base_error_type,
                )
            )
            if base_row.get("candidate") != base_candidate.to_record():
                raise ValueError("action census base Candidate differs from raw call")
            reconstructed_call_failures["base_worker"] += int(base_generation_error)
        else:
            assert base_replay is not None
            replay = base_replay.case(problem.problem_id)
            base_row = base_replay_by_id.get(problem.problem_id)
            if base_row != _base_replay_record(
                contract=contract,
                problem=problem,
                replay=replay,
            ):
                raise ValueError("action census base replay record differs")
            base_candidate = replay.candidate
        reconstructed_base[problem.problem_id] = base_candidate
        reconstructed_unusable_outputs["base_worker"] += int(not base_candidate.usable)

        builder_row = logical_by_key.get((problem.problem_id, "state_construction"))
        if builder_row is None:
            raise ValueError("action census builder call is missing")
        expected_builder_request = state_builder_request(
            problem,
            base_candidate,
            contract.builder_mode,
        )
        expected_builder_seed = contract.builder_seed + problem_index
        if (
            builder_row.get("problem_index") != problem_index
            or builder_row.get("role") != "state_builder"
            or builder_row.get("seed") != expected_builder_seed
            or builder_row.get("request") != expected_builder_request.to_record()
        ):
            raise ValueError("action census builder call contract differs")
        builder_call_id = builder_row.get("call_id")
        if not isinstance(builder_call_id, str):
            raise ValueError("action census builder call id is invalid")
        builder_generation_error = builder_row.get("generation_error")
        if type(builder_generation_error) is not bool:
            raise ValueError("action census builder error flag is invalid")
        builder_call_error_type = (
            "LLMResponseError" if builder_generation_error else None
        )
        if (
            builder_row.get("error_type") != builder_call_error_type
            or builder_row.get("candidate") is not None
        ):
            raise ValueError("action census builder call fields differ")
        builder_payload = replay_wire_call(
            call_id=builder_call_id,
            role="state_builder",
            request=expected_builder_request,
            identity=contract.state_builder,
            temperature=contract.builder_temperature,
            max_tokens=contract.builder_max_tokens,
            seed=expected_builder_seed,
            generation_error=builder_generation_error,
        )
        builder_usable = False
        builder_error_type = builder_call_error_type
        if builder_payload is None:
            nodes = fallback_semantic_nodes()
        else:
            try:
                nodes = decode_semantic_nodes(
                    builder_payload,
                    contract.builder_mode,
                    problem=problem,
                    draft=base_candidate,
                )
                builder_usable = True
            except ValueError as exc:
                builder_error_type = type(exc).__name__
                nodes = fallback_semantic_nodes()
        reconstructed_call_failures["state_builder"] += int(builder_generation_error)
        reconstructed_unusable_outputs["state_builder"] += int(not builder_usable)
        state = CensusState(
            problem_id=problem.problem_id,
            problem=problem.problem,
            draft=base_candidate,
            nodes=nodes,
        )
        complete = builder_usable and all(
            bool(node.title and node.content) for node in state.nodes
        )
        distinct = builder_usable and semantic_nodes_distinct(
            state.nodes,
            contract.builder_mode,
        )
        reconstructed_states[problem.problem_id] = state
        reconstructed_builder_flags[problem.problem_id] = (
            builder_usable,
            complete,
            distinct,
        )
        state_row = state_by_id.get(problem.problem_id)
        if state_row is None or (
            state_row.get("problem_index") != problem_index
            or state_row.get("observation") != state.observation()
            or state_row.get("state_sha256") != state.sha256
            or state_row.get("builder_usable") != builder_usable
            or state_row.get("builder_error_type") != builder_error_type
            or state_row.get("node_text_complete") != complete
            or state_row.get("node_text_distinct") != distinct
        ):
            raise ValueError("action census state differs from raw builder call")

        for repetition in range(contract.repetitions):
            expected_order = _action_order(contract, problem_index, repetition)
            expected_seed = (
                contract.worker_seed
                + 50_000
                + problem_index * contract.repetitions
                + repetition
            )
            for order_index, target in enumerate(expected_order):
                transition = transition_by_key.get(
                    (problem.problem_id, target, repetition)
                )
                if transition is None:
                    raise ValueError("action census transition is missing")
                expected_request = action_request(state, target)
                if (
                    transition.get("problem_index") != problem_index
                    or transition.get("order_index") != order_index
                    or transition.get("seed") != expected_seed
                    or transition.get("request") != expected_request.to_record()
                    or transition.get("state_sha256") != state.sha256
                ):
                    raise ValueError("action census transition contract differs")
                action_call_id = transition.get("worker_call_id")
                generation_error = transition.get("generation_error")
                if (
                    not isinstance(action_call_id, str)
                    or type(generation_error) is not bool
                ):
                    raise ValueError("action census action call fields are invalid")
                action_error_type = "LLMResponseError" if generation_error else None
                if transition.get("error_type") != action_error_type:
                    raise ValueError("action census action error type differs")
                payload = replay_wire_call(
                    call_id=action_call_id,
                    role="action_worker",
                    request=expected_request,
                    identity=contract.worker,
                    temperature=contract.worker_temperature,
                    max_tokens=contract.worker_max_tokens,
                    seed=expected_seed,
                    generation_error=generation_error,
                )
                candidate = _candidate(
                    LogicalCall(
                        call_id=action_call_id,
                        seed=expected_seed,
                        request_sha256=request_sha256(expected_request),
                        payload=payload,
                        generation_error=generation_error,
                        error_type=action_error_type,
                    )
                )
                if transition.get("candidate") != candidate.to_record():
                    raise ValueError("action census Candidate differs from raw call")
                reconstructed_candidates[(problem.problem_id, target, repetition)] = (
                    candidate
                )
                reconstructed_call_failures["action_worker"] += int(generation_error)
                reconstructed_unusable_outputs["action_worker"] += int(
                    not candidate.usable
                )
    state_hash_by_id: dict[str, str] = {}
    for record in states:
        problem_id = str(record.get("problem_id"))
        observation = record.get("observation")
        if problem_id not in problem_by_id or not isinstance(observation, Mapping):
            raise ValueError("action census state record is invalid")
        state_hash = canonical_json_sha256(dict(observation))
        if state_hash != record.get("state_sha256"):
            raise ValueError("action census state hash differs")
        state_hash_by_id[problem_id] = state_hash

    reward_rows: dict[tuple[str, str, int], int] = {}
    request_normal_forms: dict[str, set[str]] = {
        problem.problem_id: set() for problem in problems
    }
    seed_sets: dict[tuple[str, int], set[int]] = {}
    for record in transitions:
        problem_id = str(record.get("problem_id"))
        action = record.get("action")
        candidate_record = record.get("candidate")
        repetition = record.get("repetition")
        request = record.get("request")
        if (
            problem_id not in problem_by_id
            or not isinstance(action, Mapping)
            or not isinstance(candidate_record, Mapping)
            or type(repetition) is not int
            or not isinstance(request, Mapping)
        ):
            raise ValueError("action census transition is malformed")
        target = action.get("target")
        if target not in ACTION_TARGETS or action.get("verb") != "revise":
            raise ValueError("action census transition action is invalid")
        if record.get("state_sha256") != state_hash_by_id[problem_id]:
            raise ValueError("action census transition state hash differs")
        try:
            candidate = Candidate(**dict(candidate_record))
        except (TypeError, ValueError) as exc:
            raise ValueError("action census transition Candidate is invalid") from exc
        reward = terminal_reward(candidate, problem_by_id[problem_id])
        if record.get("reward") != reward:
            raise ValueError("action census transition reward differs")
        key = (problem_id, str(target), repetition)
        if key in reward_rows:
            raise ValueError("action census transition key is duplicated")
        if reconstructed_candidates.get(key) != candidate:
            raise ValueError("action census transition Candidate replay differs")
        reward_rows[key] = reward
        user = request.get("user")
        if not isinstance(user, str):
            raise ValueError("action census transition request lacks user text")
        marker = f"Selected target id: {target}"
        if user.count(marker) != 1:
            raise ValueError("action census target marker differs")
        request_normal_forms[problem_id].add(
            user.replace(marker, "Selected target id: <TARGET>")
        )
        seed = record.get("seed")
        if type(seed) is not int:
            raise ValueError("action census transition seed is invalid")
        seed_sets.setdefault((problem_id, repetition), set()).add(seed)
    if len(reward_rows) != expected_transition_count:
        raise ValueError("action census transition grid is incomplete")
    if any(len(forms) != 1 for forms in request_normal_forms.values()):
        raise ValueError("action census prompts differ outside target")
    if any(len(seeds) != 1 for seeds in seed_sets.values()):
        raise ValueError("action census targets do not share repetition seeds")

    replay_cases: list[CensusCase] = []
    for record in summaries:
        problem_id = str(record.get("problem_id"))
        problem = problem_by_id.get(problem_id)
        rewards = record.get("action_rewards")
        usable = record.get("action_usable")
        if (
            problem is None
            or not isinstance(rewards, Mapping)
            or not isinstance(usable, Mapping)
        ):
            raise ValueError("action census case summary is malformed")
        expected_rewards = {
            target: tuple(
                reward_rows[(problem_id, target, repetition)]
                for repetition in range(contract.repetitions)
            )
            for target in ACTION_TARGETS
        }
        if {
            target: tuple(rewards.get(target, ())) for target in ACTION_TARGETS
        } != expected_rewards:
            raise ValueError("action census case rewards differ from transitions")
        expected_usable = {
            target: tuple(
                reconstructed_candidates[(problem_id, target, repetition)].usable
                for repetition in range(contract.repetitions)
            )
            for target in ACTION_TARGETS
        }
        observed_usable = {
            target: tuple(usable.get(target, ())) for target in ACTION_TARGETS
        }
        if observed_usable != expected_usable:
            raise ValueError("action census case usability differs from calls")
        base_candidate = reconstructed_base[problem_id]
        expected_base_reward = terminal_reward(base_candidate, problem)
        builder_usable, node_complete, node_distinct = reconstructed_builder_flags[
            problem_id
        ]
        problem_index = contract.problem_ids.index(problem_id)
        expected_orders = [
            list(_action_order(contract, problem_index, repetition))
            for repetition in range(contract.repetitions)
        ]
        if (
            record.get("problem_index") != problem_index
            or record.get("base_reward") != expected_base_reward
            or record.get("base_usable") != base_candidate.usable
            or record.get("builder_usable") != builder_usable
            or record.get("node_text_complete") != node_complete
            or record.get("node_text_distinct") != node_distinct
            or record.get("action_orders") != expected_orders
            or record.get("state_sha256") != reconstructed_states[problem_id].sha256
        ):
            raise ValueError("action census case summary replay differs")
        replay_cases.append(
            CensusCase(
                problem_id=problem_id,
                family=problem.family,
                base_reward=expected_base_reward,
                base_usable=base_candidate.usable,
                builder_usable=builder_usable,
                node_text_complete=node_complete,
                node_text_distinct=node_distinct,
                action_rewards=expected_rewards,
                action_usable=expected_usable,
            )
        )
    replay_analysis = analyze_action_census(tuple(replay_cases), contract)
    run_summary = run_summaries[0]
    if run_summary.get("analysis") != replay_analysis:
        raise ValueError("action census replay analysis differs")
    replayed_worker_usage = {
        key: replayed_usage["base_worker"][key] + replayed_usage["action_worker"][key]
        for key in ("generate_calls", "prompt_tokens", "generated_tokens")
    }
    replayed_builder_usage = {
        key: replayed_usage["state_builder"][key]
        for key in ("generate_calls", "prompt_tokens", "generated_tokens")
    }
    if (
        run_summary.get("run_contract_sha256") != contract.sha256
        or run_summary.get("case_count") != expected_case_count
        or run_summary.get("logical_call_count") != expected_call_count
        or run_summary.get("transition_count") != expected_transition_count
        or run_summary.get("call_failure_counts") != dict(reconstructed_call_failures)
        or run_summary.get("unusable_output_counts")
        != dict(reconstructed_unusable_outputs)
        or run_summary.get("usage")
        != {
            "worker": replayed_worker_usage,
            "state_builder": replayed_builder_usage,
        }
    ):
        raise ValueError("action census summary counts differ")

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
            raise ValueError("action census artifact contains a forbidden field")
    return {
        "schema_version": "action-census-artifact-replay-audit-v2",
        "run_sha256": file_sha256(run_path),
        "diagnostic_sha256": file_sha256(diagnostic_path),
        "run_record_count": len(run_records),
        "diagnostic_record_count": len(diagnostic_records),
        "logical_call_count": expected_call_count,
        "transition_count": expected_transition_count,
        "unique_state_count": len(state_hash_by_id),
        "call_id_binding_passed": True,
        "base_mode": "hosted" if hosted_base else "sealed_replay",
        "base_replay_passed": True,
        "request_hash_replay_passed": True,
        "state_hash_replay_passed": True,
        "reward_replay_passed": True,
        "prompt_isolation_passed": True,
        "common_seed_passed": True,
        "summary_replay_passed": True,
        "retry_transcript_replay_passed": True,
        "usage_replay_passed": True,
        "forbidden_field_scan_passed": True,
        "attempt_counts": dict(replayed_attempt_counts),
        "first_attempt_accepted_counts": dict(first_attempt_accepted_counts),
        "attempt_outcome_counts": {
            role: dict(sorted(counts.items()))
            for role, counts in attempt_outcome_counts.items()
        },
        "observed_response_models": {
            role: sorted(models) for role, models in observed_response_models.items()
        },
        "served_revision_attested": False,
        "served_revision_note": (
            "registered revisions are request-side metadata; the providers do "
            "not attest immutable served weight revisions"
        ),
    }


def _call_record(
    *,
    contract: ActionCensusContract,
    problem_index: int,
    problem_id: str,
    phase: str,
    role: str,
    request: WorkerRequest,
    call: LogicalCall,
    candidate: Candidate | None = None,
) -> dict[str, object]:
    return {
        "schema_version": ACTION_CENSUS_CALL_VERSION,
        "record_type": "logical_call",
        "preregistration_id": contract.preregistration_id,
        "problem_index": problem_index,
        "problem_id": problem_id,
        "phase": phase,
        "role": role,
        "seed": call.seed,
        "request_sha256": call.request_sha256,
        "request": request.to_record(),
        "candidate": None if candidate is None else candidate.to_record(),
        "generation_error": call.generation_error,
        "error_type": call.error_type,
        "call_id": call.call_id,
    }


def _base_replay_record(
    *,
    contract: ActionCensusContract,
    problem: CensusProblem,
    replay: BaseReplayCase,
) -> dict[str, object]:
    request = base_request(problem)
    return {
        "schema_version": ACTION_CENSUS_BASE_REPLAY_VERSION,
        "record_type": "base_replay",
        "preregistration_id": contract.preregistration_id,
        "problem_index": replay.problem_index,
        "problem_id": replay.problem_id,
        "source_call_id": replay.source_call_id,
        "source_record_sha256": replay.source_record_sha256,
        "seed": replay.seed,
        "request_sha256": replay.request_sha256,
        "request": request.to_record(),
        "candidate": replay.candidate.to_record(),
        "candidate_sha256": canonical_json_sha256(replay.candidate.to_record()),
        "generation_error": replay.generation_error,
        "error_type": replay.error_type,
    }


def _action_order(
    contract: ActionCensusContract,
    index: int,
    repetition: int,
) -> tuple[str, ...]:
    """Use an ID-ranked, position-balanced schedule independent of outcomes."""
    ranked_ids = sorted(
        contract.problem_ids,
        key=lambda problem_id: hashlib.sha256(
            f"{contract.action_order_seed}:{problem_id}".encode("utf-8")
        ).hexdigest(),
    )
    problem_id = contract.problem_ids[index]
    rank = ranked_ids.index(problem_id)
    row = WILLIAMS_ORDER_INDICES[repetition]
    return tuple(
        ACTION_TARGETS[(target_index + rank) % len(ACTION_TARGETS)]
        for target_index in row
    )


def collect_action_census_run(
    *,
    run_path: str | Path,
    diagnostic_path: str | Path,
    dataset_path: str | Path,
    contract: ActionCensusContract,
    problems: tuple[CensusProblem, ...],
    worker_client: OpenAICompatibleLLMClient,
    state_builder_client: OpenAICompatibleLLMClient,
    base_replay: PreparedBaseReplay | None = None,
) -> ActionCensusResult:
    """Collect shared-prefix complete-answer outcomes and publish atomically."""
    if (
        type(worker_client) is not OpenAICompatibleLLMClient
        or type(state_builder_client) is not OpenAICompatibleLLMClient
    ):
        raise TypeError("action census requires OpenAI-compatible clients")
    if worker_client is state_builder_client:
        raise ValueError("worker and state builder require distinct clients")
    _validate_client_binding(contract.worker, worker_client, role="worker")
    _validate_client_binding(
        contract.state_builder, state_builder_client, role="state builder"
    )
    if file_sha256(dataset_path) != contract.dataset_sha256:
        raise ValueError("dataset bytes differ from action census contract")
    if problem_source_sha256(problems) != contract.problem_source_sha256:
        raise ValueError("problem source differs from action census contract")
    if evaluation_binding_sha256(problems) != contract.evaluation_binding_sha256:
        raise ValueError("evaluation binding differs from action census contract")
    if tuple(problem.problem_id for problem in problems) != contract.problem_ids:
        raise ValueError("problem ids differ from action census contract")
    if (base_replay is None) != (contract.base_replay is None):
        raise ValueError("base replay presence differs from action census contract")
    if base_replay is not None and (
        base_replay.binding != contract.base_replay
        or tuple(case.problem_id for case in base_replay.cases) != contract.problem_ids
    ):
        raise ValueError("base replay differs from action census contract")
    _validate_anchored_catalog_binding(contract, problems, base_replay)

    final_run = Path(run_path)
    final_diagnostic = Path(diagnostic_path)
    partial_run, partial_diagnostic = _validate_output_paths(
        final_run, final_diagnostic
    )
    worker_before = _counter_snapshot(worker_client)
    builder_before = _counter_snapshot(state_builder_client)
    old_worker_observer = worker_client.attempt_observer
    old_builder_observer = state_builder_client.attempt_observer
    cases: list[CensusCase] = []
    logical_call_count = 0
    transition_count = 0
    call_failures: Counter[str] = Counter()
    unusable_outputs: Counter[str] = Counter()

    diagnostic_roles = (
        ("base_worker", "state_builder", "action_worker")
        if base_replay is None
        else ("state_builder", "action_worker")
    )
    diagnostic_header = {
        "schema_version": ACTION_CENSUS_DIAGNOSTIC_VERSION,
        "record_type": "diagnostic_header",
        "preregistration_id": contract.preregistration_id,
        "run_contract_sha256": contract.sha256,
        "roles": list(diagnostic_roles),
    }
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
                roles=diagnostic_roles,
            )

            for problem_index, problem in enumerate(problems):
                if base_replay is None:
                    direct_request = base_request(problem)
                    direct_call = _logical_call(
                        client=worker_client,
                        recorder=recorder,
                        role="base_worker",
                        request=direct_request,
                        temperature=contract.worker_temperature,
                        max_tokens=contract.worker_max_tokens,
                        seed=contract.worker_seed + problem_index * 100,
                    )
                    direct_candidate = _candidate(direct_call)
                    _write_record(
                        run_handle,
                        _call_record(
                            contract=contract,
                            problem_index=problem_index,
                            problem_id=problem.problem_id,
                            phase="base_draft",
                            role="base_worker",
                            request=direct_request,
                            call=direct_call,
                            candidate=direct_candidate,
                        ),
                    )
                    logical_call_count += 1
                    call_failures["base_worker"] += int(direct_call.generation_error)
                else:
                    replay = base_replay.case(problem.problem_id)
                    if replay.problem_index != problem_index:
                        raise ValueError("base replay problem order differs")
                    direct_candidate = replay.candidate
                    _write_record(
                        run_handle,
                        _base_replay_record(
                            contract=contract,
                            problem=problem,
                            replay=replay,
                        ),
                    )
                    call_failures["base_worker"] += 0
                unusable_outputs["base_worker"] += int(not direct_candidate.usable)

                builder_request = state_builder_request(
                    problem,
                    direct_candidate,
                    contract.builder_mode,
                )
                builder_call = _logical_call(
                    client=state_builder_client,
                    recorder=recorder,
                    role="state_builder",
                    request=builder_request,
                    temperature=contract.builder_temperature,
                    max_tokens=contract.builder_max_tokens,
                    seed=contract.builder_seed + problem_index,
                )
                builder_usable = False
                builder_error_type: str | None = builder_call.error_type
                if builder_call.payload is None:
                    nodes = fallback_semantic_nodes()
                else:
                    try:
                        nodes = decode_semantic_nodes(
                            builder_call.payload,
                            contract.builder_mode,
                            problem=problem,
                            draft=direct_candidate,
                        )
                        builder_usable = True
                    except ValueError as exc:
                        builder_error_type = type(exc).__name__
                        nodes = fallback_semantic_nodes()
                _write_record(
                    run_handle,
                    _call_record(
                        contract=contract,
                        problem_index=problem_index,
                        problem_id=problem.problem_id,
                        phase="state_construction",
                        role="state_builder",
                        request=builder_request,
                        call=builder_call,
                    ),
                )
                logical_call_count += 1
                call_failures["state_builder"] += int(builder_call.generation_error)
                unusable_outputs["state_builder"] += int(not builder_usable)

                state = CensusState(
                    problem_id=problem.problem_id,
                    problem=problem.problem,
                    draft=direct_candidate,
                    nodes=nodes,
                )
                node_text_complete = builder_usable and all(
                    bool(node.title and node.content) for node in state.nodes
                )
                node_text_distinct = builder_usable and semantic_nodes_distinct(
                    state.nodes,
                    contract.builder_mode,
                )
                requests = {
                    target: action_request(state, target) for target in ACTION_TARGETS
                }
                assert_action_requests_isolated(requests)
                _write_record(
                    run_handle,
                    {
                        "schema_version": ACTION_CENSUS_STATE_RECORD_VERSION,
                        "record_type": "census_state",
                        "preregistration_id": contract.preregistration_id,
                        "problem_index": problem_index,
                        "problem_id": problem.problem_id,
                        "state_sha256": state.sha256,
                        "builder_usable": builder_usable,
                        "builder_error_type": builder_error_type,
                        "node_text_complete": node_text_complete,
                        "node_text_distinct": node_text_distinct,
                        "observation": state.observation(),
                    },
                )

                rewards: dict[str, list[int]] = {
                    target: [] for target in ACTION_TARGETS
                }
                usable: dict[str, list[bool]] = {
                    target: [] for target in ACTION_TARGETS
                }
                orders: list[list[str]] = []
                pending_transitions: list[
                    tuple[
                        int,
                        int,
                        str,
                        int,
                        WorkerRequest,
                        LogicalCall,
                        Candidate,
                    ]
                ] = []
                for repetition in range(contract.repetitions):
                    order = _action_order(contract, problem_index, repetition)
                    orders.append(list(order))
                    common_seed = (
                        contract.worker_seed
                        + 50_000
                        + problem_index * contract.repetitions
                        + repetition
                    )
                    for order_index, target in enumerate(order):
                        request = requests[target]
                        call = _logical_call(
                            client=worker_client,
                            recorder=recorder,
                            role="action_worker",
                            request=request,
                            temperature=contract.worker_temperature,
                            max_tokens=contract.worker_max_tokens,
                            seed=common_seed,
                        )
                        candidate = _candidate(call)
                        call_failures["action_worker"] += int(call.generation_error)
                        unusable_outputs["action_worker"] += int(not candidate.usable)
                        pending_transitions.append(
                            (
                                repetition,
                                order_index,
                                target,
                                common_seed,
                                request,
                                call,
                                candidate,
                            )
                        )
                        logical_call_count += 1

                for (
                    repetition,
                    order_index,
                    target,
                    common_seed,
                    request,
                    call,
                    candidate,
                ) in pending_transitions:
                    reward = terminal_reward(candidate, problem)
                    rewards[target].append(reward)
                    usable[target].append(candidate.usable)
                    _write_record(
                        run_handle,
                        {
                            "schema_version": ACTION_CENSUS_TRANSITION_VERSION,
                            "record_type": "action_transition",
                            "preregistration_id": contract.preregistration_id,
                            "problem_index": problem_index,
                            "problem_id": problem.problem_id,
                            "state_sha256": state.sha256,
                            "action": {"verb": "revise", "target": target},
                            "repetition": repetition,
                            "order_index": order_index,
                            "seed": common_seed,
                            "request_sha256": call.request_sha256,
                            "request": request.to_record(),
                            "candidate": candidate.to_record(),
                            "generation_error": call.generation_error,
                            "error_type": call.error_type,
                            "worker_call_id": call.call_id,
                            "reward": reward,
                        },
                    )
                    transition_count += 1

                case = CensusCase(
                    problem_id=problem.problem_id,
                    family=problem.family,
                    base_reward=terminal_reward(direct_candidate, problem),
                    base_usable=direct_candidate.usable,
                    builder_usable=builder_usable,
                    node_text_complete=node_text_complete,
                    node_text_distinct=node_text_distinct,
                    action_rewards={
                        target: tuple(rewards[target]) for target in ACTION_TARGETS
                    },
                    action_usable={
                        target: tuple(usable[target]) for target in ACTION_TARGETS
                    },
                )
                cases.append(case)
                _write_record(
                    run_handle,
                    {
                        "schema_version": ACTION_CENSUS_CASE_VERSION,
                        "record_type": "case_summary",
                        "preregistration_id": contract.preregistration_id,
                        "problem_index": problem_index,
                        "problem_id": problem.problem_id,
                        "state_sha256": state.sha256,
                        "base_reward": case.base_reward,
                        "base_usable": case.base_usable,
                        "builder_usable": builder_usable,
                        "node_text_complete": node_text_complete,
                        "node_text_distinct": node_text_distinct,
                        "action_orders": orders,
                        "action_rewards": {
                            target: list(case.action_rewards[target])
                            for target in ACTION_TARGETS
                        },
                        "action_usable": {
                            target: list(case.action_usable[target])
                            for target in ACTION_TARGETS
                        },
                    },
                    flush=True,
                )

            analysis = analyze_action_census(tuple(cases), contract)
            summary = {
                "schema_version": ACTION_CENSUS_SUMMARY_VERSION,
                "record_type": "run_summary",
                "preregistration_id": contract.preregistration_id,
                "run_contract_sha256": contract.sha256,
                "case_count": len(cases),
                "logical_call_count": logical_call_count,
                "transition_count": transition_count,
                "call_failure_counts": dict(call_failures),
                "unusable_output_counts": dict(unusable_outputs),
                "usage": {
                    "worker": _counter_delta(worker_before, worker_client),
                    "state_builder": _counter_delta(
                        builder_before, state_builder_client
                    ),
                },
                "analysis": analysis,
            }
            _write_record(run_handle, summary)
            run_handle.flush()
            diagnostic_handle.flush()
            os.fsync(run_handle.fileno())
            os.fsync(diagnostic_handle.fileno())
    finally:
        worker_client.attempt_observer = old_worker_observer
        state_builder_client.attempt_observer = old_builder_observer

    audit_action_census_artifacts(
        run_path=partial_run,
        diagnostic_path=partial_diagnostic,
        contract=contract,
        problems=problems,
        base_replay=base_replay,
    )
    _finalize_no_clobber(
        (
            (partial_diagnostic, final_diagnostic),
            (partial_run, final_run),
        )
    )
    return ActionCensusResult(
        run_path=final_run,
        diagnostic_path=final_diagnostic,
        case_count=len(cases),
        logical_call_count=logical_call_count,
        transition_count=transition_count,
        run_contract_sha256=contract.sha256,
        analysis=analysis,
    )


__all__ = [
    "ACTION_TARGETS",
    "ANCHORED_OPERATIONS",
    "ANSWER_SCHEMA",
    "BUILDER_MODE_ACTIONABLE_CHECKS",
    "BUILDER_MODE_ANCHORED_CHECKS",
    "BUILDER_MODE_ANCHORED_RANGE_LISTS",
    "BUILDER_MODE_ANCHORED_RANGES",
    "BUILDER_MODE_SEGMENT_DRAFT",
    "BUILDER_MODES",
    "BaseReplayBinding",
    "BaseReplayCase",
    "SEMANTIC_NODE_SCHEMA",
    "ActionCensusContract",
    "ActionCensusResult",
    "CensusCase",
    "CensusProblem",
    "CensusState",
    "PreparedBaseReplay",
    "DraftSpan",
    "SemanticNode",
    "action_request",
    "analyze_action_census",
    "anchored_catalog_binding_sha256",
    "anchored_draft_spans",
    "anchored_span_catalog_record",
    "assert_action_requests_isolated",
    "audit_action_census_artifacts",
    "base_request",
    "build_action_census_contract",
    "collect_action_census_run",
    "decode_semantic_nodes",
    "evaluation_binding_sha256",
    "file_sha256",
    "load_census_problems",
    "problem_source_sha256",
    "prepare_base_replay",
    "state_builder_request",
    "terminal_reward",
]
