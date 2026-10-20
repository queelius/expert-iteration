"""Production orchestration for hosted reasoning v2 teacher collection."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from ..llm_client import OpenAICompatibleLLMClient
from .casework_source import CaseworkProblemSource
from .contracts import CollectionContract
from .controller import TypedLLMController
from .diagnostics import DiagnosticSidecarWriter, read_diagnostic_sidecar
from .lenses import TypedPolicyDraftGenerator
from .records import (
    CanonicalRunWriter,
    build_run_header,
    build_transition_record,
)
from .replay import ReplayedRun, replay_canonical_run
from .runtime import BoundEnvironment, EpisodeRuntime


_PREREGISTRATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class HostedExecutionAuthorization:
    """Explicit acknowledgement that a hosted run is already preregistered."""

    preregistration_id: str
    expected_problem_source_sha256: str
    expected_environment_contract_sha256: str
    expected_collection_contract_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.preregistration_id, str)
            or _PREREGISTRATION_ID.fullmatch(self.preregistration_id) is None
        ):
            raise ValueError(
                "preregistration_id must be 1 to 128 safe identifier characters"
            )
        for field in (
            "expected_problem_source_sha256",
            "expected_environment_contract_sha256",
            "expected_collection_contract_sha256",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{field} must be a sha256 digest")

    def validate_bindings(
        self,
        *,
        problem_source_sha256: str,
        environment_contract_sha256: str,
        collection_contract_sha256: str,
    ) -> None:
        actual = {
            "problem source": problem_source_sha256,
            "environment contract": environment_contract_sha256,
            "collection contract": collection_contract_sha256,
        }
        expected = {
            "problem source": self.expected_problem_source_sha256,
            "environment contract": (
                self.expected_environment_contract_sha256
            ),
            "collection contract": self.expected_collection_contract_sha256,
        }
        for label, digest in actual.items():
            if digest != expected[label]:
                raise ValueError(
                    f"{label} digest differs from hosted authorization"
                )


@dataclass(frozen=True, slots=True)
class CollectedRun:
    """Artifacts returned only after exact diagnostic and symbolic replay."""

    run_path: Path
    diagnostic_path: Path
    replayed_run: ReplayedRun
    episode_count: int
    decision_count: int
    solved_count: int

    def __post_init__(self) -> None:
        if self.episode_count != len(self.replayed_run.episodes):
            raise ValueError("collected episode count differs from replay")
        if self.decision_count != len(self.replayed_run.decisions):
            raise ValueError("collected decision count differs from replay")
        if not 0 <= self.solved_count <= self.episode_count:
            raise ValueError("collected solved count is invalid")


def _validate_production_bindings(
    *,
    environment: BoundEnvironment,
    collection_contract: CollectionContract,
    problem_source: CaseworkProblemSource,
    controller: TypedLLMController,
    generator: TypedPolicyDraftGenerator,
) -> tuple[OpenAICompatibleLLMClient, OpenAICompatibleLLMClient]:
    if not isinstance(environment, BoundEnvironment):
        raise TypeError("collection environment has an invalid type")
    if not isinstance(collection_contract, CollectionContract):
        raise TypeError("collection contract has an invalid type")
    if type(problem_source) is not CaseworkProblemSource:
        raise TypeError("collection needs the exact casework problem source")
    if type(controller) is not TypedLLMController:
        raise TypeError("collection needs the exact typed LLM controller")
    if type(generator) is not TypedPolicyDraftGenerator:
        raise TypeError("collection needs the exact typed policy generator")
    controller_client = controller.client
    generator_client = generator.client
    if type(controller_client) is not OpenAICompatibleLLMClient:
        raise TypeError("controller client must be OpenAI-compatible")
    if type(generator_client) is not OpenAICompatibleLLMClient:
        raise TypeError("generator client must be OpenAI-compatible")
    if controller_client is generator_client:
        raise ValueError("controller and generator need distinct clients")
    expected_source = {
        "problem_source_contract": problem_source.problem_source_contract,
        "problem_source_sha256": problem_source.problem_source_sha256,
        "problem_adapter_implementation": (
            problem_source.problem_adapter_implementation
        ),
    }
    for field, expected in expected_source.items():
        if getattr(collection_contract, field) != expected:
            raise ValueError(
                f"collection {field} differs from the problem source"
            )
    if collection_contract.behavior_source != "teacher":
        raise ValueError("hosted teacher collection needs teacher behavior")
    if controller.environment != environment:
        raise ValueError("controller differs from the collection environment")
    if generator.environment != environment:
        raise ValueError("generator differs from the collection environment")
    return generator_client, controller_client


def _partial_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.partial")


def _remove_link_if_owned(partial: Path, final: Path) -> None:
    try:
        partial_stat = partial.stat()
        final_stat = final.stat()
    except FileNotFoundError:
        return
    if (
        partial_stat.st_dev == final_stat.st_dev
        and partial_stat.st_ino == final_stat.st_ino
    ):
        final.unlink()


def _finalize_no_clobber(
    pairs: tuple[tuple[Path, Path], ...]
) -> None:
    linked: list[tuple[Path, Path]] = []
    try:
        for partial, final in pairs:
            os.link(partial, final)
            linked.append((partial, final))
    except BaseException:
        for partial, final in reversed(linked):
            _remove_link_if_owned(partial, final)
        raise
    for partial, _ in pairs:
        partial.unlink()


def collect_and_strict_replay(
    *,
    run_path: str | Path,
    diagnostic_path: str | Path,
    environment: BoundEnvironment,
    collection_contract: CollectionContract,
    problem_source: CaseworkProblemSource,
    controller: TypedLLMController,
    generator: TypedPolicyDraftGenerator,
    authorization: HostedExecutionAuthorization,
) -> CollectedRun:
    """Collect every source problem and accept only strict replayable output.

    Constructing ``authorization`` does not preregister an experiment. The
    caller must do that in the experiment ledger before allowing this function
    to issue hosted requests.
    """
    if type(authorization) is not HostedExecutionAuthorization:
        raise TypeError("hosted collection needs explicit authorization")
    generator_client, controller_client = _validate_production_bindings(
        environment=environment,
        collection_contract=collection_contract,
        problem_source=problem_source,
        controller=controller,
        generator=generator,
    )
    authorization.validate_bindings(
        problem_source_sha256=problem_source.problem_source_sha256,
        environment_contract_sha256=environment.sha256,
        collection_contract_sha256=collection_contract.sha256,
    )
    output_path = Path(run_path)
    sidecar_path = Path(diagnostic_path)
    partial_output_path = _partial_path(output_path)
    partial_sidecar_path = _partial_path(sidecar_path)
    all_paths = (
        output_path,
        sidecar_path,
        partial_output_path,
        partial_sidecar_path,
    )
    if len({path.resolve(strict=False) for path in all_paths}) != len(all_paths):
        raise ValueError("run, diagnostic, and partial paths must be distinct")
    for path, label in (
        (output_path, "run"),
        (sidecar_path, "diagnostic"),
        (partial_output_path, "partial run"),
        (partial_sidecar_path, "partial diagnostic"),
    ):
        if path.exists():
            raise FileExistsError(f"{label} output already exists: {path}")
        if not path.parent.exists() or not path.parent.is_dir():
            raise FileNotFoundError(
                f"{label} output directory does not exist: {path.parent}"
            )

    header = build_run_header(environment, collection_contract)
    collected_decisions = 0
    collected_solved = 0
    with DiagnosticSidecarWriter(
        partial_sidecar_path,
        environment=environment,
        collection_contract=collection_contract,
        generator_client=generator_client,
        controller_client=controller_client,
    ) as diagnostic_writer, CanonicalRunWriter(
        partial_output_path,
        header,
        environment=environment,
        collection_contract=collection_contract,
    ) as run_writer:
        for episode_index, problem_id in enumerate(
            problem_source.problem_ids
        ):
            episode_key = (
                f"{authorization.preregistration_id}:"
                f"{episode_index:06d}:{problem_id}"
            )
            runtime = EpisodeRuntime(
                environment=environment,
                collection_contract=collection_contract,
                generator=generator,
                state=problem_source(problem_id),
                controller=controller,
                episode_key=episode_key,
                decision_recorder=diagnostic_writer,
            )
            while True:
                runtime, transition = runtime.step()
                if transition.reward not in {0, 1}:
                    raise RuntimeError(
                        "collection observed a non-symbolic reward"
                    )
                record = build_transition_record(
                    transition,
                    environment=environment,
                    collection_contract=collection_contract,
                    episode_key=episode_key,
                    problem_id=problem_id,
                )
                run_writer.append(record, transition=transition)
                collected_decisions += 1
                if transition.terminal:
                    collected_solved += transition.reward
                    break

    diagnostic_index = read_diagnostic_sidecar(
        partial_sidecar_path,
        environment_contract_sha256=environment.sha256,
        collection_contract_sha256=collection_contract.sha256,
        generator_transport_sha256=(
            collection_contract.generator_transport_sha256
        ),
        controller_transport_sha256=(
            collection_contract.controller_transport_sha256
        ),
    )
    replayed = replay_canonical_run(
        partial_output_path,
        environment=environment,
        initial_state_resolver=problem_source,
        collection_contract=collection_contract,
        diagnostic_index=diagnostic_index,
    )
    if tuple(
        episode.problem_id for episode in replayed.episodes
    ) != problem_source.problem_ids:
        raise RuntimeError("strict replay episode order differs from source")
    if len(replayed.decisions) != collected_decisions:
        raise RuntimeError("strict replay decision count differs from collection")
    replayed_solved = sum(
        episode.decisions[-1].transition.reward
        for episode in replayed.episodes
    )
    if replayed_solved != collected_solved:
        raise RuntimeError("strict replay reward differs from collection")
    _finalize_no_clobber((
        (partial_sidecar_path, sidecar_path),
        (partial_output_path, output_path),
    ))
    return CollectedRun(
        run_path=output_path,
        diagnostic_path=sidecar_path,
        replayed_run=replayed,
        episode_count=len(replayed.episodes),
        decision_count=len(replayed.decisions),
        solved_count=replayed_solved,
    )


__all__ = [
    "CollectedRun",
    "HostedExecutionAuthorization",
    "collect_and_strict_replay",
]
