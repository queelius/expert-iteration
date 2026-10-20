"""Strict episode-sequential symbolic replay for canonical v2 runs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

from .contracts import CollectionContract, EnvironmentContract
from .diagnostics import (
    DiagnosticSidecarIndex,
    DiagnosticValidationError,
    require_hosted_bc_diagnostic_coverage,
    validate_diagnostic_header,
)
from .model import (
    Action,
    GoalObjective,
    GoalObligation,
    RejectionReason,
    State,
    new_graph,
)
from .observation import observation_sha256
from .provenance import DecisionProvenance
from .records import (
    RUN_HEADER_VERSION,
    TRANSITION_RECORD_VERSION,
    BehaviorMetadata,
    DiagnosticRef,
    RecordValidationError,
    deserialize_action,
    deserialize_learned_effect,
    deserialize_outcome,
    parse_canonical_json_line,
    validate_run_header,
    validate_transition_record,
)
from .runtime import BoundEnvironment
from .transition import (
    MacroTransition,
    MacroTransitionError,
    PolicyDraft,
    PolicyGenerationError,
    PolicyPayloadError,
)


class ReplayValidationError(RecordValidationError):
    """Raised when a canonical run cannot be replayed exactly."""


_REPLAY_TRUST_SEAL = object()


@runtime_checkable
class InitialStateResolver(Protocol):
    """Trusted problem source bound to the collection header."""

    problem_source_contract: str
    problem_source_sha256: str
    problem_adapter_implementation: str

    def __call__(self, problem_id: str) -> State:
        ...


@dataclass(frozen=True, slots=True)
class ReplayedDecision:
    """One persisted decision paired with its exact symbolic replay."""

    transition: MacroTransition
    behavior: BehaviorMetadata
    diagnostic_ref: DiagnosticRef | None


@dataclass(frozen=True, slots=True)
class ReplayedEpisode:
    """One complete episode, in learned-turn order."""

    episode_key: str
    problem_id: str
    decisions: tuple[ReplayedDecision, ...]

    @property
    def terminal(self) -> bool:
        return bool(self.decisions and self.decisions[-1].transition.terminal)


@dataclass(frozen=True, slots=True)
class ReplayedRun:
    """A header-bound run containing only complete replayed episodes."""

    environment_contract: EnvironmentContract
    collection_contract: CollectionContract
    episodes: tuple[ReplayedEpisode, ...]
    _trust_seal: object = field(
        init=False, repr=False, compare=False, default=None
    )
    _trusted_binding: object = field(
        init=False, repr=False, compare=False, default=None
    )

    def validate_trusted(self) -> None:
        """Require construction by the strict canonical replay boundary."""
        binding = (
            self.environment_contract,
            self.collection_contract,
            self.episodes,
        )
        if (
            self._trust_seal is not _REPLAY_TRUST_SEAL
            or self._trusted_binding != binding
        ):
            raise ReplayValidationError(
                "replayed run was not produced by strict canonical replay"
            )

    @property
    def decisions(self) -> tuple[ReplayedDecision, ...]:
        return tuple(
            decision
            for episode in self.episodes
            for decision in episode.decisions
        )


@dataclass(slots=True)
class _EpisodeProgress:
    episode_key: str
    problem_id: str
    state: State
    decisions: list[ReplayedDecision]
    terminal: bool = False


class _PersistedEffectGenerator:
    """Reproduce a recorded typed effect without calling a model."""

    def __init__(
        self,
        draft: PolicyDraft | None,
        rejection: RejectionReason | None,
    ) -> None:
        self._draft = draft
        self._rejection = rejection
        self.called = False

    def generate(self, state: State, action: Action) -> PolicyDraft:
        if self.called:
            raise ReplayValidationError(
                "persisted effect generator was called more than once"
            )
        self.called = True
        if self._rejection is RejectionReason.GENERATION_ERROR:
            raise PolicyGenerationError("persisted generation failure")
        if self._rejection is RejectionReason.PAYLOAD_ERROR:
            raise PolicyPayloadError("persisted payload failure")
        if self._draft is None:
            raise ReplayValidationError(
                "persisted applied effect has no typed draft"
            )
        return self._draft


def _reconstruct_clean_state(
    observation: dict[str, object], environment: BoundEnvironment
) -> State:
    if observation["turn"] != 0:
        raise ReplayValidationError("an episode must begin at turn zero")
    if observation["previous_outcome"] is not None:
        raise ReplayValidationError(
            "a turn-zero observation cannot have a previous outcome"
        )

    goals = observation["goals"]
    evidence = observation["evidence"]
    claims = observation["claims"]
    if type(goals) is not list or len(goals) != 1:
        raise ReplayValidationError(
            "a clean initial graph must contain exactly one Goal"
        )
    if type(evidence) is not list or type(claims) is not list:
        raise ReplayValidationError("initial graph records are invalid")
    if claims:
        raise ReplayValidationError(
            "a clean initial graph cannot contain Claims"
        )

    root = goals[0]
    if (
        root["id"] != 0
        or observation["root_goal_id"] != 0
        or root["parent_goal"] is not None
        or root["assumptions"] != []
        or root["connective"] is not None
        or root["status"] != "open"
        or root["committed_claim"] is not None
    ):
        raise ReplayValidationError("turn-zero root is not a clean open Goal")

    givens: list[str] = []
    for index, item in enumerate(evidence, start=1):
        if (
            item["id"] != index
            or item["scope_goal"] != 0
            or item["premises"] != []
            or item["source"] != "trusted_given"
            or item["committable"] is not False
        ):
            raise ReplayValidationError(
                "turn-zero Evidence is not a clean trusted given"
            )
        givens.append(item["statement"])

    obligation_record = root["obligation"]
    try:
        obligation = GoalObligation(
            variable=obligation_record["variable"],
            equation=obligation_record["equation"],
            domain=obligation_record["domain"],
            objective=GoalObjective(obligation_record["objective"]),
        )
        graph = new_graph(
            root["statement"],
            obligation,
            tuple(givens),
            generation_contract=environment.generation_contract,
        )
        state = State(graph=graph, turn=0, previous_outcome=None)
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayValidationError(
            f"clean initial graph reconstruction failed: {exc}"
        ) from exc
    if environment.observe(state) != observation:
        raise ReplayValidationError(
            "turn-zero observation differs from its clean graph reconstruction"
        )
    return state


def _trusted_initial_state(
    problem_id: str,
    observation: dict[str, object],
    *,
    environment: BoundEnvironment,
    resolver: InitialStateResolver,
) -> State:
    try:
        trusted = resolver(problem_id)
    except KeyError as exc:
        raise ReplayValidationError(
            f"problem_id {problem_id!r} is absent from the trusted resolver"
        ) from exc
    if not isinstance(trusted, State):
        raise ReplayValidationError(
            "trusted initial-state resolver returned an invalid type"
        )
    if trusted.turn != 0 or trusted.previous_outcome is not None:
        raise ReplayValidationError(
            "trusted initial state must begin at turn zero without an outcome"
        )
    trusted_observation = environment.observe(trusted)
    canonical = _reconstruct_clean_state(trusted_observation, environment)
    if canonical != trusted:
        raise ReplayValidationError(
            "trusted initial state is not a canonical clean graph"
        )
    if trusted_observation != observation:
        raise ReplayValidationError(
            "turn-zero observation differs from the trusted problem state"
        )
    return trusted


def _validate_resolver_binding(
    resolver: InitialStateResolver,
    collection_contract: CollectionContract,
) -> None:
    expected = {
        "problem_source_contract": (
            collection_contract.problem_source_contract
        ),
        "problem_source_sha256": collection_contract.problem_source_sha256,
        "problem_adapter_implementation": (
            collection_contract.problem_adapter_implementation
        ),
    }
    for field, value in expected.items():
        if getattr(resolver, field) != value:
            raise ReplayValidationError(
                f"initial-state resolver {field} differs from the header"
            )


def _validate_diagnostic_binding(
    diagnostics: DiagnosticSidecarIndex | None,
    *,
    environment: BoundEnvironment,
    collection_contract: CollectionContract,
) -> None:
    if diagnostics is None:
        return
    try:
        validate_diagnostic_header(
            diagnostics.header,
            environment_contract_sha256=environment.sha256,
            collection_contract_sha256=collection_contract.sha256,
            generator_transport_sha256=(
                collection_contract.generator_transport_sha256
            ),
            controller_transport_sha256=(
                collection_contract.controller_transport_sha256
            ),
        )
    except DiagnosticValidationError as exc:
        raise ReplayValidationError(
            f"diagnostic sidecar differs from the run: {exc}"
        ) from exc


def _validate_decision_diagnostic(
    record: dict[str, object],
    behavior: BehaviorMetadata,
    diagnostic_ref: DiagnosticRef | None,
    *,
    environment: BoundEnvironment,
    collection_contract: CollectionContract,
    diagnostics: DiagnosticSidecarIndex | None,
    expected_policy_draft: PolicyDraft | None,
    expected_rejection_reason: RejectionReason | None,
) -> None:
    hosted = (
        collection_contract.controller_transport.base_url is not None
        or collection_contract.generator_transport.base_url is not None
    )
    if diagnostic_ref is None:
        if hosted:
            raise ReplayValidationError(
                "hosted transition needs a diagnostic decision receipt"
            )
        if behavior.bc_eligible:
            raise ReplayValidationError(
                "BC-eligible teacher decision needs a diagnostic receipt"
            )
        return
    if diagnostics is None:
        raise ReplayValidationError(
            "diagnostic reference cannot be trusted without its sidecar"
        )
    try:
        diagnostics.validate_decision_reference(
            diagnostic_ref,
            collection_contract_sha256=collection_contract.sha256,
            episode_key=record["episode_key"],
            turn=record["turn"],
            observation_sha256=observation_sha256(
                record["observation"],
                environment.generation_contract,
                environment.verifier,
                environment_contract_sha256=environment.sha256,
            ),
            selected_action=deserialize_action(
                record["action"], "transition.action"
            ),
            source=behavior.source.value,
            fallback=behavior.fallback,
            bc_eligible=behavior.bc_eligible,
            environment=environment,
            collection_contract=collection_contract,
            observation=record["observation"],
            expected_policy_draft=expected_policy_draft,
            expected_rejection_reason=expected_rejection_reason,
            require_bc_trust=behavior.bc_eligible,
        )
        if behavior.bc_eligible:
            if hosted:
                require_hosted_bc_diagnostic_coverage(diagnostics.header)
    except DiagnosticValidationError as exc:
        raise ReplayValidationError(
            f"decision diagnostic is not replay-trusted: {exc}"
        ) from exc


def _replay_record(
    record: dict[str, object],
    *,
    environment: BoundEnvironment,
    collection_contract: CollectionContract,
    diagnostics: DiagnosticSidecarIndex | None,
    state: State,
) -> tuple[ReplayedDecision, State]:
    if record["observation"] != environment.observe(state):
        raise ReplayValidationError(
            "current observation does not match the replayed episode state"
        )
    action = deserialize_action(record["action"], "transition.action")
    outcome = deserialize_outcome(record["outcome"], "transition.outcome")
    draft = deserialize_learned_effect(
        record["learned_effect"], action, outcome
    )
    behavior = BehaviorMetadata.from_record(record["behavior"])
    if behavior.source.value != collection_contract.behavior_source:
        raise ReplayValidationError(
            "transition behavior source differs from the collection header"
        )
    if behavior.fallback:
        first_legal = deserialize_action(
            record["observation"]["legal_pairs"][0],
            "transition.observation.legal_pairs[0]",
        )
        if action != first_legal:
            raise ReplayValidationError(
                "teacher fallback action differs from the pinned first legal pair"
            )
    diagnostic_ref = (
        None
        if record["diagnostic_ref"] is None
        else DiagnosticRef.from_record(record["diagnostic_ref"])
    )
    _validate_decision_diagnostic(
        record,
        behavior,
        diagnostic_ref,
        environment=environment,
        collection_contract=collection_contract,
        diagnostics=diagnostics,
        expected_policy_draft=draft,
        expected_rejection_reason=outcome.reason,
    )
    generator = _PersistedEffectGenerator(draft, outcome.reason)
    transition = environment.step(state, action, generator)
    if not generator.called:
        raise ReplayValidationError("persisted effect generator was not called")
    transition = replace(
        transition,
        collection_contract_sha256=collection_contract.sha256,
        decision_provenance=DecisionProvenance(
            source=behavior.source,
            bc_eligible=behavior.bc_eligible,
            fallback=behavior.fallback,
            diagnostic_record_id=(
                None if diagnostic_ref is None else diagnostic_ref.record_id
            ),
            episode_key=(
                None if diagnostic_ref is None else record["episode_key"]
            ),
            observation_sha256=(
                None
                if diagnostic_ref is None
                else observation_sha256(
                    record["observation"],
                    environment.generation_contract,
                    environment.verifier,
                    environment_contract_sha256=environment.sha256,
                )
            ),
        ),
    )
    validate_transition_record(
        record,
        environment=environment,
        collection_contract=collection_contract,
        expected_transition=transition,
    )
    return (
        ReplayedDecision(
            transition=transition,
            behavior=behavior,
            diagnostic_ref=diagnostic_ref,
        ),
        transition.next_state,
    )


def replay_canonical_run_lines(
    lines: Iterable[bytes],
    *,
    environment: BoundEnvironment,
    initial_state_resolver: InitialStateResolver,
    collection_contract: CollectionContract | None = None,
    diagnostic_index: DiagnosticSidecarIndex | None = None,
) -> ReplayedRun:
    """Parse and replay a complete header-first canonical JSONL run."""
    if not isinstance(environment, BoundEnvironment):
        raise TypeError("replay environment has an invalid type")
    if not isinstance(initial_state_resolver, InitialStateResolver):
        raise TypeError("initial_state_resolver has an invalid type")
    if collection_contract is not None and not isinstance(
        collection_contract, CollectionContract
    ):
        raise TypeError("replay collection contract has an invalid type")
    if diagnostic_index is not None and not isinstance(
        diagnostic_index, DiagnosticSidecarIndex
    ):
        raise TypeError("diagnostic_index has an invalid type")

    iterator = iter(lines)
    try:
        first_line = next(iterator)
    except StopIteration as exc:
        raise ReplayValidationError("canonical run is missing its header") from exc
    try:
        header = parse_canonical_json_line(first_line)
        if header.get("schema_version") != RUN_HEADER_VERSION:
            raise ReplayValidationError(
                "canonical run must begin with a v2 run header"
            )
        environment_contract, parsed_collection = validate_run_header(
            header,
            environment=environment,
            collection_contract=collection_contract,
        )
    except (RecordValidationError, TypeError, ValueError) as exc:
        raise ReplayValidationError(f"line 1: {exc}") from exc
    _validate_resolver_binding(initial_state_resolver, parsed_collection)
    _validate_diagnostic_binding(
        diagnostic_index,
        environment=environment,
        collection_contract=parsed_collection,
    )

    progress_by_key: dict[str, _EpisodeProgress] = {}
    episode_order: list[str] = []
    row_count = 0
    for line_number, line in enumerate(iterator, start=2):
        try:
            record = parse_canonical_json_line(line)
            schema = record.get("schema_version")
            if schema == RUN_HEADER_VERSION:
                raise ReplayValidationError(
                    "canonical run cannot contain a second header"
                )
            if schema != TRANSITION_RECORD_VERSION:
                raise ReplayValidationError(
                    "canonical run contains a mixed or unknown record schema"
                )
            validate_transition_record(
                record,
                environment=environment,
                collection_contract=parsed_collection,
            )

            episode_key = record["episode_key"]
            progress = progress_by_key.get(episode_key)
            if progress is None:
                if record["turn"] != 0:
                    raise ReplayValidationError(
                        "an episode must begin at turn zero"
                    )
                _validate_resolver_binding(
                    initial_state_resolver, parsed_collection
                )
                state = _trusted_initial_state(
                    record["problem_id"],
                    record["observation"],
                    environment=environment,
                    resolver=initial_state_resolver,
                )
                progress = _EpisodeProgress(
                    episode_key=episode_key,
                    problem_id=record["problem_id"],
                    state=state,
                    decisions=[],
                )
                progress_by_key[episode_key] = progress
                episode_order.append(episode_key)
            else:
                if progress.terminal:
                    raise ReplayValidationError(
                        "episode has a transition after its terminal row"
                    )
                if record["problem_id"] != progress.problem_id:
                    raise ReplayValidationError(
                        "episode problem_id changed between turns"
                    )
                if record["turn"] != progress.state.turn:
                    raise ReplayValidationError(
                        "episode turn is not contiguous"
                    )

            decision, next_state = _replay_record(
                record,
                environment=environment,
                collection_contract=parsed_collection,
                diagnostics=diagnostic_index,
                state=progress.state,
            )
            progress.decisions.append(decision)
            progress.state = next_state
            progress.terminal = decision.transition.terminal
            row_count += 1
        except (
            MacroTransitionError,
            RecordValidationError,
            TypeError,
            ValueError,
        ) as exc:
            raise ReplayValidationError(
                f"line {line_number}: {exc}"
            ) from exc

    if row_count == 0:
        raise ReplayValidationError("canonical run contains no transitions")
    unfinished = [
        episode_key
        for episode_key in episode_order
        if not progress_by_key[episode_key].terminal
    ]
    if unfinished:
        raise ReplayValidationError(
            f"canonical run ends with truncated episodes {unfinished!r}"
        )

    episodes = tuple(
        ReplayedEpisode(
            episode_key=progress_by_key[key].episode_key,
            problem_id=progress_by_key[key].problem_id,
            decisions=tuple(progress_by_key[key].decisions),
        )
        for key in episode_order
    )
    binding = (environment_contract, parsed_collection, episodes)
    replayed = ReplayedRun(
        environment_contract=environment_contract,
        collection_contract=parsed_collection,
        episodes=episodes,
    )
    object.__setattr__(replayed, "_trust_seal", _REPLAY_TRUST_SEAL)
    object.__setattr__(replayed, "_trusted_binding", binding)
    return replayed


def replay_canonical_run(
    path: str | Path,
    *,
    environment: BoundEnvironment,
    initial_state_resolver: InitialStateResolver,
    collection_contract: CollectionContract | None = None,
    diagnostic_index: DiagnosticSidecarIndex | None = None,
) -> ReplayedRun:
    """Read and replay one complete canonical JSONL run from disk."""
    with Path(path).open("rb") as handle:
        return replay_canonical_run_lines(
            handle,
            environment=environment,
            initial_state_resolver=initial_state_resolver,
            collection_contract=collection_contract,
            diagnostic_index=diagnostic_index,
        )


__all__ = [
    "InitialStateResolver",
    "ReplayValidationError",
    "ReplayedDecision",
    "ReplayedEpisode",
    "ReplayedRun",
    "replay_canonical_run",
    "replay_canonical_run_lines",
]
