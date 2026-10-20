"""Reproducible publication boundary for workspace v1 experiments."""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping, TextIO
from urllib.parse import urlsplit

from ..llm_client import (
    OpenAICompatibleAttempt,
    OpenAICompatibleLLMClient,
)
from .controller import (
    TEACHER_OUTPUT_VERSION,
    TEACHER_PROMPT_VERSION,
    LLMTeacherPolicy,
    TeacherDecision,
)
from .lenses import (
    CHILD_CONTEXT_CONTRACT_VERSION,
    CHILD_CONTEXT_MODES,
    DIRECT_PROMPT_VERSION,
    HYBRID_LENS_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    TEACHER_PRESENTATION_VERSION,
    WORKER_PROMPT_VERSION,
)
from .model import (
    ACTION_SCHEMA_VERSION,
    REWARD_CONTRACT_VERSION,
    STATE_SCHEMA_VERSION,
    TRANSITION_SCHEMA_VERSION,
)
from .policies import (
    DirectPolicy,
    FixedDecompositionPolicy,
    WorkspacePolicy,
    run_policy_episode,
)
from .records import (
    DIAGNOSTIC_ATTEMPT_VERSION,
    DIAGNOSTIC_HEADER_VERSION,
    EPISODE_SUMMARY_VERSION,
    RUN_HEADER_VERSION,
    RUN_SUMMARY_VERSION,
    canonical_json_bytes,
    canonical_json_sha256,
    transition_record,
)
from .runtime import (
    SUBMIT_RECEIPT_VERSION,
    WORKER_EXECUTION_CONTRACT_VERSION,
    WorkspaceEnvironment,
)


PROBLEM_SOURCE_VERSION = "workspace-label-free-problem-source-v1"
EVALUATION_BINDING_VERSION = "workspace-hidden-label-binding-v1"
RUN_CONTRACT_VERSION = "workspace-hosted-run-contract-v1.1"
SEED_POLICY_VERSION = "episode-turn-offset-v1"
POLICY_NAMES = frozenset({"direct", "fixed_decomposition", "teacher"})


def _pairs_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value!r}")


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


@dataclass(frozen=True, slots=True)
class ProblemExample:
    """One problem plus a label that remains outside policy state."""

    problem_id: str
    problem: str
    expected_answer: str = field(repr=False)

    def __post_init__(self) -> None:
        _nonempty_text(self.problem_id, "problem id")
        _nonempty_text(self.problem, "problem")
        if not isinstance(self.expected_answer, str):
            raise ValueError("expected answer must be text")

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
        *,
        default_id: str | None = None,
    ) -> ProblemExample:
        if not isinstance(record, Mapping):
            raise ValueError("problem row must be an object")
        try:
            return cls(
                _nonempty_text(record.get("id", default_id), "problem id"),
                _nonempty_text(record["problem"], "problem"),
                str(record["answer"]),
            )
        except KeyError as exc:
            raise ValueError(
                f"problem row is missing field {exc.args[0]!r}"
            ) from exc

    def source_record(self) -> dict[str, str]:
        return {"id": self.problem_id, "problem": self.problem}


def load_problem_examples(
    path: str | Path,
    *,
    start_index: int,
    episode_count: int,
) -> tuple[ProblemExample, ...]:
    """Load one consecutive dataset slice with strict JSONL framing."""
    if type(start_index) is not int or start_index < 0:
        raise ValueError("start index must be nonnegative")
    if type(episode_count) is not int or episode_count <= 0:
        raise ValueError("episode count must be positive")
    rows: list[ProblemExample] = []
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
            rows.append(ProblemExample.from_record(
                value,
                default_id=f"dataset-row-{line_number - 1:06d}",
            ))
    selected = tuple(rows[start_index:start_index + episode_count])
    if len(selected) != episode_count:
        raise ValueError("selected problem slice exceeds the dataset")
    ids = tuple(example.problem_id for example in selected)
    if len(ids) != len(set(ids)):
        raise ValueError("selected problem ids must be unique")
    return selected


def problem_source_record(
    examples: tuple[ProblemExample, ...],
) -> dict[str, object]:
    if not examples or any(
        not isinstance(example, ProblemExample) for example in examples
    ):
        raise ValueError("problem source requires ProblemExample values")
    return {
        "schema_version": PROBLEM_SOURCE_VERSION,
        "problems": [example.source_record() for example in examples],
    }


def problem_source_sha256(examples: tuple[ProblemExample, ...]) -> str:
    return canonical_json_sha256(problem_source_record(examples))


def evaluation_binding_sha256(
    examples: tuple[ProblemExample, ...],
) -> str:
    """Bind the hidden labels without placing them in a run artifact."""
    private_record = {
        "schema_version": EVALUATION_BINDING_VERSION,
        "labels": [
            {
                "id": example.problem_id,
                "answer": example.expected_answer,
            }
            for example in examples
        ],
    }
    return canonical_json_sha256(private_record)


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    provider: str
    model: str
    revision: str
    base_url: str
    guided_json: bool
    timeout_s: float
    parse_retries: int
    backoff_retries: int

    def __post_init__(self) -> None:
        for name in ("provider", "model", "revision", "base_url"):
            _nonempty_text(getattr(self, name), name.replace("_", " "))
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("model base URL must be absolute HTTP(S)")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("model base URL cannot contain credentials or query")
        if type(self.guided_json) is not bool:
            raise ValueError("guided_json must be boolean")
        if type(self.timeout_s) is not float or self.timeout_s <= 0.0:
            raise ValueError("timeout must be a positive float")
        for name in ("parse_retries", "backoff_retries"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be nonnegative")

    def to_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RunContract:
    preregistration_id: str
    arm: str
    policy_name: str
    problem_source_sha256: str
    evaluation_binding_sha256: str
    problem_ids: tuple[str, ...]
    turn_limit: int
    max_depth: int
    child_context_mode: str
    worker: ModelIdentity
    worker_temperature: float
    worker_max_tokens: int
    worker_seed: int
    teacher: ModelIdentity | None = None
    teacher_temperature: float | None = None
    teacher_max_tokens: int | None = None
    teacher_seed: int | None = None

    def __post_init__(self) -> None:
        _nonempty_text(self.preregistration_id, "preregistration id")
        _nonempty_text(self.arm, "arm")
        if self.policy_name not in POLICY_NAMES:
            raise ValueError("unknown workspace policy")
        for name in (
            "problem_source_sha256",
            "evaluation_binding_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be a lowercase sha256")
        if not self.problem_ids or len(self.problem_ids) != len(
            set(self.problem_ids)
        ):
            raise ValueError("problem ids must be nonempty and unique")
        if type(self.turn_limit) is not int or self.turn_limit <= 0:
            raise ValueError("turn limit must be positive")
        if type(self.max_depth) is not int or self.max_depth <= 0:
            raise ValueError("max depth must be positive")
        if self.child_context_mode not in CHILD_CONTEXT_MODES:
            raise ValueError("unknown child context mode")
        if not isinstance(self.worker, ModelIdentity):
            raise ValueError("worker identity is invalid")
        if type(self.worker_temperature) is not float or self.worker_temperature < 0:
            raise ValueError("worker temperature must be nonnegative")
        if type(self.worker_max_tokens) is not int or self.worker_max_tokens <= 0:
            raise ValueError("worker max tokens must be positive")
        if type(self.worker_seed) is not int or self.worker_seed < 0:
            raise ValueError("worker seed must be nonnegative")
        needs_teacher = self.policy_name == "teacher"
        if needs_teacher != (self.teacher is not None):
            raise ValueError("teacher identity must match the policy")
        teacher_values = (
            self.teacher_temperature,
            self.teacher_max_tokens,
            self.teacher_seed,
        )
        if needs_teacher != all(value is not None for value in teacher_values):
            raise ValueError("teacher sampling settings must match the policy")
        if self.teacher_temperature is not None and (
            type(self.teacher_temperature) is not float
            or self.teacher_temperature < 0
        ):
            raise ValueError("teacher temperature must be nonnegative")
        if self.teacher_max_tokens is not None and (
            type(self.teacher_max_tokens) is not int
            or self.teacher_max_tokens <= 0
        ):
            raise ValueError("teacher max tokens must be positive")
        if self.teacher_seed is not None and (
            type(self.teacher_seed) is not int or self.teacher_seed < 0
        ):
            raise ValueError("teacher seed must be nonnegative")

    def header(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema_version": RUN_HEADER_VERSION,
            "record_type": "run_header",
            "run_contract_version": RUN_CONTRACT_VERSION,
            "preregistration_id": self.preregistration_id,
            "arm": self.arm,
            "policy_name": self.policy_name,
            "problem_source": {
                "schema_version": PROBLEM_SOURCE_VERSION,
                "sha256": self.problem_source_sha256,
                "evaluation_binding_version": EVALUATION_BINDING_VERSION,
                "evaluation_binding_sha256": self.evaluation_binding_sha256,
                "problem_ids": list(self.problem_ids),
            },
            "contracts": {
                "state": STATE_SCHEMA_VERSION,
                "action": ACTION_SCHEMA_VERSION,
                "transition": TRANSITION_SCHEMA_VERSION,
                "reward": REWARD_CONTRACT_VERSION,
                "worker_execution": WORKER_EXECUTION_CONTRACT_VERSION,
                "worker_prompt": WORKER_PROMPT_VERSION,
                "direct_prompt": DIRECT_PROMPT_VERSION,
                "hybrid_lens": HYBRID_LENS_VERSION,
                "child_answer_context": CHILD_CONTEXT_CONTRACT_VERSION,
                "observation": OBSERVATION_SCHEMA_VERSION,
                "submit_receipt": SUBMIT_RECEIPT_VERSION,
                "teacher_prompt": TEACHER_PROMPT_VERSION,
                "teacher_output": TEACHER_OUTPUT_VERSION,
                "teacher_presentation": TEACHER_PRESENTATION_VERSION,
                "seed_policy": SEED_POLICY_VERSION,
            },
            "settings": {
                "turn_limit": self.turn_limit,
                "max_depth": self.max_depth,
                "child_context_mode": self.child_context_mode,
                "worker_temperature": self.worker_temperature,
                "worker_max_tokens": self.worker_max_tokens,
                "worker_seed": self.worker_seed,
                "teacher_temperature": self.teacher_temperature,
                "teacher_max_tokens": self.teacher_max_tokens,
                "teacher_seed": self.teacher_seed,
            },
            "models": {
                "worker": self.worker.to_record(),
                "teacher": (
                    None if self.teacher is None else self.teacher.to_record()
                ),
            },
        }
        body["run_contract_sha256"] = canonical_json_sha256(body)
        return body

    @property
    def sha256(self) -> str:
        return str(self.header()["run_contract_sha256"])


@dataclass(frozen=True, slots=True)
class CollectionResult:
    run_path: Path
    diagnostic_path: Path
    episode_count: int
    transition_count: int
    solved_count: int
    run_contract_sha256: str


class _AttemptRecorder:
    def __init__(
        self,
        handle: TextIO,
        *,
        roles: tuple[str, ...] = ("worker", "teacher"),
    ) -> None:
        if not roles or len(roles) != len(set(roles)) or any(
            not isinstance(role, str) or not role.strip() for role in roles
        ):
            raise ValueError("diagnostic roles must be nonempty and unique")
        self._handle = handle
        self._call_ids = {role: [] for role in roles}
        self._seen = {role: set() for role in roles}
        self._sequence = 0

    def callback(
        self, role: str
    ) -> Callable[[OpenAICompatibleAttempt], None]:
        if role not in self._call_ids:
            raise ValueError("unknown diagnostic role")

        def observe(attempt: OpenAICompatibleAttempt) -> None:
            if not isinstance(attempt, OpenAICompatibleAttempt):
                raise TypeError("diagnostic observer received an invalid attempt")
            if attempt.call_id not in self._seen[role]:
                self._seen[role].add(attempt.call_id)
                self._call_ids[role].append(attempt.call_id)
            record = {
                "schema_version": DIAGNOSTIC_ATTEMPT_VERSION,
                "record_type": "http_attempt",
                "sequence": self._sequence,
                "role": role,
                "attempt": asdict(attempt),
            }
            self._sequence += 1
            _write_record(self._handle, record, flush=True)

        return observe

    def count(self, role: str) -> int:
        return len(self._call_ids[role])

    def since(self, role: str, offset: int) -> tuple[str, ...]:
        return tuple(self._call_ids[role][offset:])


def _write_record(
    handle: TextIO,
    record: Mapping[str, object],
    *,
    flush: bool = False,
) -> None:
    encoded = canonical_json_bytes(dict(record))
    if b"expected_answer" in encoded:
        raise ValueError("published record leaked the hidden answer field")
    handle.write(encoded.decode("ascii") + "\n")
    if flush:
        handle.flush()


def _partial_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.partial")


def _remove_owned_link(partial: Path, final: Path) -> None:
    try:
        left = partial.stat()
        right = final.stat()
    except FileNotFoundError:
        return
    if left.st_dev == right.st_dev and left.st_ino == right.st_ino:
        final.unlink()


def _finalize_no_clobber(
    pairs: tuple[tuple[Path, Path], ...],
) -> None:
    linked: list[tuple[Path, Path]] = []
    try:
        for partial, final in pairs:
            os.link(partial, final)
            linked.append((partial, final))
    except BaseException:
        for partial, final in reversed(linked):
            _remove_owned_link(partial, final)
        raise
    for partial, _ in pairs:
        partial.unlink()


def _validate_output_paths(
    run_path: Path,
    diagnostic_path: Path,
) -> tuple[Path, Path]:
    partial_run = _partial_path(run_path)
    partial_diagnostic = _partial_path(diagnostic_path)
    paths = (run_path, diagnostic_path, partial_run, partial_diagnostic)
    if len({path.resolve(strict=False) for path in paths}) != len(paths):
        raise ValueError("run, diagnostic, and partial paths must be distinct")
    for path in paths:
        if path.exists():
            raise FileExistsError(f"experiment output already exists: {path}")
        if not path.parent.is_dir():
            raise FileNotFoundError(
                f"experiment output directory does not exist: {path.parent}"
            )
    return partial_run, partial_diagnostic


def _policy_for_episode(
    contract: RunContract,
    teacher_client: OpenAICompatibleLLMClient | None,
    *,
    episode_index: int,
) -> WorkspacePolicy:
    if contract.policy_name == "direct":
        return DirectPolicy()
    if contract.policy_name == "fixed_decomposition":
        return FixedDecompositionPolicy()
    if teacher_client is None:
        raise ValueError("teacher policy requires a teacher client")
    if (
        contract.teacher_temperature is None
        or contract.teacher_max_tokens is None
        or contract.teacher_seed is None
    ):
        raise AssertionError("validated teacher settings disappeared")
    return LLMTeacherPolicy(
        client=teacher_client,
        max_depth=contract.max_depth,
        temperature=contract.teacher_temperature,
        max_tokens=contract.teacher_max_tokens,
        base_seed=(
            contract.teacher_seed + episode_index * contract.turn_limit
        ),
    )


def _counter_snapshot(
    client: OpenAICompatibleLLMClient,
) -> tuple[int, int, int]:
    return (
        client.generate_calls,
        client.prompt_tokens,
        client.generated_tokens,
    )


def _validate_client_binding(
    identity: ModelIdentity,
    client: OpenAICompatibleLLMClient,
    *,
    role: str,
) -> None:
    actual = {
        "base_url": client.base_url.rstrip("/"),
        "model": client.model,
        "guided_json": client.guided_json,
        "timeout_s": client.timeout_s,
        "parse_retries": client.max_retries,
        "backoff_retries": client.max_backoff_retries,
    }
    expected = {
        "base_url": identity.base_url.rstrip("/"),
        "model": identity.model,
        "guided_json": identity.guided_json,
        "timeout_s": identity.timeout_s,
        "parse_retries": identity.parse_retries,
        "backoff_retries": identity.backoff_retries,
    }
    if actual != expected:
        raise ValueError(f"{role} client differs from its model identity")


def _counter_delta(
    before: tuple[int, int, int],
    client: OpenAICompatibleLLMClient,
) -> dict[str, int]:
    after = _counter_snapshot(client)
    values = tuple(right - left for left, right in zip(before, after))
    if any(value < 0 for value in values):
        raise RuntimeError("client counters decreased during collection")
    return {
        "generate_calls": values[0],
        "prompt_tokens": values[1],
        "generated_tokens": values[2],
    }


def collect_workspace_run(
    *,
    run_path: str | Path,
    diagnostic_path: str | Path,
    contract: RunContract,
    examples: tuple[ProblemExample, ...],
    worker_client: OpenAICompatibleLLMClient,
    teacher_client: OpenAICompatibleLLMClient | None = None,
) -> CollectionResult:
    """Collect one sequential arm and publish only complete paired artifacts."""
    if not isinstance(contract, RunContract):
        raise TypeError("collection requires a RunContract")
    if type(worker_client) is not OpenAICompatibleLLMClient:
        raise TypeError("worker client must be OpenAI-compatible")
    if (contract.policy_name == "teacher") != (teacher_client is not None):
        raise ValueError("teacher client must match the selected policy")
    if teacher_client is worker_client:
        raise ValueError("worker and teacher require distinct client objects")
    if teacher_client is not None and type(
        teacher_client
    ) is not OpenAICompatibleLLMClient:
        raise TypeError("teacher client must be OpenAI-compatible")
    _validate_client_binding(contract.worker, worker_client, role="worker")
    if teacher_client is not None:
        if contract.teacher is None:
            raise AssertionError("validated teacher identity disappeared")
        _validate_client_binding(
            contract.teacher, teacher_client, role="teacher"
        )
    if contract.problem_ids != tuple(
        example.problem_id for example in examples
    ):
        raise ValueError("run contract problem ids differ from the source")
    if contract.problem_source_sha256 != problem_source_sha256(examples):
        raise ValueError("run contract differs from the problem projection")
    if contract.evaluation_binding_sha256 != evaluation_binding_sha256(
        examples
    ):
        raise ValueError("run contract differs from the evaluation binding")

    final_run = Path(run_path)
    final_diagnostic = Path(diagnostic_path)
    partial_run, partial_diagnostic = _validate_output_paths(
        final_run, final_diagnostic
    )
    header = contract.header()
    diagnostic_header = {
        "schema_version": DIAGNOSTIC_HEADER_VERSION,
        "record_type": "diagnostic_header",
        "preregistration_id": contract.preregistration_id,
        "arm": contract.arm,
        "run_contract_sha256": contract.sha256,
        "attempt_schema_version": DIAGNOSTIC_ATTEMPT_VERSION,
    }
    worker_before = _counter_snapshot(worker_client)
    teacher_before = (
        None if teacher_client is None else _counter_snapshot(teacher_client)
    )
    old_worker_observer = worker_client.attempt_observer
    old_teacher_observer = (
        None if teacher_client is None else teacher_client.attempt_observer
    )
    solved_count = 0
    transition_count = 0
    generation_errors = 0
    teacher_fallbacks = 0
    action_counts: Counter[str] = Counter()

    try:
        with partial_run.open(
            "x", encoding="ascii", newline="\n"
        ) as run_handle, partial_diagnostic.open(
            "x", encoding="ascii", newline="\n"
        ) as diagnostic_handle:
            _write_record(run_handle, header)
            _write_record(diagnostic_handle, diagnostic_header, flush=True)
            recorder = _AttemptRecorder(diagnostic_handle)
            worker_client.attempt_observer = recorder.callback("worker")
            if teacher_client is not None:
                teacher_client.attempt_observer = recorder.callback("teacher")

            for episode_index, example in enumerate(examples):
                worker_offset = recorder.count("worker")
                teacher_offset = recorder.count("teacher")
                policy = _policy_for_episode(
                    contract,
                    teacher_client,
                    episode_index=episode_index,
                )
                environment = WorkspaceEnvironment(
                    problem=example.problem,
                    expected_answer=example.expected_answer,
                    worker=worker_client,
                    turn_limit=contract.turn_limit,
                    max_depth=contract.max_depth,
                    child_context_mode=contract.child_context_mode,
                    temperature=contract.worker_temperature,
                    max_tokens=contract.worker_max_tokens,
                    base_seed=(
                        contract.worker_seed
                        + episode_index * contract.turn_limit
                    ),
                )
                episode = run_policy_episode(environment, policy)
                worker_ids = recorder.since("worker", worker_offset)
                teacher_ids = recorder.since("teacher", teacher_offset)
                worker_transitions = tuple(
                    transition
                    for transition in episode.transitions
                    if transition.worker_request is not None
                )
                if len(worker_ids) != len(worker_transitions):
                    raise RuntimeError(
                        "worker call ids differ from generative transitions"
                    )
                if contract.policy_name == "teacher":
                    if not isinstance(policy, LLMTeacherPolicy):
                        raise AssertionError("teacher policy changed type")
                    decisions: tuple[TeacherDecision | None, ...] = tuple(
                        policy.decisions
                    )
                    if len(decisions) != len(episode.transitions):
                        raise RuntimeError(
                            "teacher decisions differ from transitions"
                        )
                    if len(teacher_ids) != len(episode.transitions):
                        raise RuntimeError(
                            "teacher call ids differ from transitions"
                        )
                else:
                    decisions = (None,) * len(episode.transitions)
                    if teacher_ids:
                        raise RuntimeError("non-teacher policy made teacher calls")

                worker_iterator = iter(worker_ids)
                for local_index, (transition, decision) in enumerate(
                    zip(episode.transitions, decisions, strict=True)
                ):
                    worker_call_id = (
                        next(worker_iterator)
                        if transition.worker_request is not None
                        else None
                    )
                    teacher_call_id = (
                        teacher_ids[local_index] if teacher_ids else None
                    )
                    record = transition_record(
                        transition,
                        experiment_id=contract.preregistration_id,
                        arm=contract.arm,
                        episode_index=episode_index,
                        problem_id=example.problem_id,
                        max_depth=contract.max_depth,
                        worker_call_id=worker_call_id,
                        teacher_call_id=teacher_call_id,
                        teacher_decision=decision,
                    )
                    _write_record(run_handle, record)
                    action_counts[transition.action.verb.value] += 1
                    generation_errors += int(transition.generation_error)
                    teacher_fallbacks += int(
                        decision is not None and decision.fallback
                    )
                    transition_count += 1
                try:
                    next(worker_iterator)
                except StopIteration:
                    pass
                else:
                    raise AssertionError("unused worker call id")

                solved_count += episode.reward
                episode_summary = {
                    "schema_version": EPISODE_SUMMARY_VERSION,
                    "record_type": "episode_summary",
                    "preregistration_id": contract.preregistration_id,
                    "arm": contract.arm,
                    "episode_index": episode_index,
                    "problem_id": example.problem_id,
                    "turn_count": len(episode.transitions),
                    "reward": episode.reward,
                    "terminal_reason": (
                        "submit"
                        if episode.final_state.history[-1].action.verb.value
                        == "submit"
                        else "horizon"
                    ),
                    "worker_call_ids": list(worker_ids),
                    "teacher_call_ids": list(teacher_ids),
                }
                _write_record(run_handle, episode_summary, flush=True)

            run_summary = {
                "schema_version": RUN_SUMMARY_VERSION,
                "record_type": "run_summary",
                "preregistration_id": contract.preregistration_id,
                "arm": contract.arm,
                "run_contract_sha256": contract.sha256,
                "episode_count": len(examples),
                "transition_count": transition_count,
                "solved_count": solved_count,
                "generation_error_count": generation_errors,
                "teacher_fallback_count": teacher_fallbacks,
                "action_counts": dict(sorted(action_counts.items())),
                "usage": {
                    "worker": _counter_delta(worker_before, worker_client),
                    "teacher": (
                        None
                        if teacher_client is None or teacher_before is None
                        else _counter_delta(teacher_before, teacher_client)
                    ),
                },
            }
            _write_record(run_handle, run_summary)
            run_handle.flush()
            diagnostic_handle.flush()
            os.fsync(run_handle.fileno())
            os.fsync(diagnostic_handle.fileno())
    finally:
        worker_client.attempt_observer = old_worker_observer
        if teacher_client is not None:
            teacher_client.attempt_observer = old_teacher_observer

    _finalize_no_clobber((
        (partial_diagnostic, final_diagnostic),
        (partial_run, final_run),
    ))
    return CollectionResult(
        run_path=final_run,
        diagnostic_path=final_diagnostic,
        episode_count=len(examples),
        transition_count=transition_count,
        solved_count=solved_count,
        run_contract_sha256=contract.sha256,
    )


__all__ = [
    "EVALUATION_BINDING_VERSION",
    "POLICY_NAMES",
    "PROBLEM_SOURCE_VERSION",
    "RUN_CONTRACT_VERSION",
    "SEED_POLICY_VERSION",
    "CollectionResult",
    "ModelIdentity",
    "ProblemExample",
    "RunContract",
    "collect_workspace_run",
    "evaluation_binding_sha256",
    "load_problem_examples",
    "problem_source_record",
    "problem_source_sha256",
]
