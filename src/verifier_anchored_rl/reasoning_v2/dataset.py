"""Trusted replay conversion for the Phase B v2 training dataset."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

from .contracts import canonical_json_bytes, canonical_json_sha256
from .features import (
    BoundFeatureExtractor,
    StructuredFeatureBatch,
    feature_contract_record,
)
from .model import Action
from .observation import serialize_action
from .records import BehaviorMetadata, DiagnosticRef
from .replay import ReplayedRun
from .runtime import BoundEnvironment
from .transition import REWARD_CONTRACT_VERSION


DATASET_SCHEMA_VERSION = "phase-b-v2"
DATASET_EXAMPLE_VERSION = "phase-b-example-v2"
GROUPED_SPLIT_VERSION = "phase-b-grouped-split-v2"
TRAINING_TARGET_VERSION = "monte-carlo-return-v1"


class DatasetValidationError(ValueError):
    """Raised when trusted replay cannot form an exact Phase B dataset."""


def _digest(value: object, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise DatasetValidationError(f"{path} must be a SHA-256 digest")
    return value


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{path} must be nonempty text")
    return value


def _gamma(value: object) -> float:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or value < 0.0
        or value > 1.0
        or (value == 0.0 and math.copysign(1.0, value) < 0.0)
    ):
        raise DatasetValidationError("gamma must be a float in [0, 1]")
    return value


def _observation_record(value: bytes, path: str) -> dict[str, object]:
    if type(value) is not bytes:
        raise DatasetValidationError(f"{path} must be immutable canonical bytes")
    try:
        parsed = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"{path} is not canonical JSON") from exc
    if type(parsed) is not dict or canonical_json_bytes(parsed) != value:
        raise DatasetValidationError(f"{path} is not an exact canonical object")
    return parsed


def _serialize_features(batch: StructuredFeatureBatch) -> dict[str, object]:
    return {
        "contract_version": batch.contract_version,
        "environment_contract_sha256": batch.environment_contract_sha256,
        "global_serialization": batch.global_serialization,
        "target_ids": list(batch.target_ids),
        "target_serializations": list(batch.target_serializations),
        "structural_scalar_names": list(batch.structural_scalar_names),
        "structural_scalars": list(batch.structural_scalars),
        "policy_verb_order": list(batch.policy_verb_order),
        "legal_pairs": [
            {"verb": verb, "target": target}
            for verb, target in batch.legal_pairs
        ],
        "legal_pair_indices": [list(pair) for pair in batch.legal_pair_indices],
        "legal_pair_mask": [list(row) for row in batch.legal_pair_mask],
    }


@dataclass(frozen=True, slots=True)
class PhaseBExample:
    """One immutable, contract-bound controller training example."""

    environment_contract_sha256: str
    collection_contract_sha256: str
    episode_key: str
    problem_id: str
    turn: int
    observation_bytes: bytes
    next_observation_bytes: bytes
    current_features: StructuredFeatureBatch
    next_features: StructuredFeatureBatch
    action: Action
    chosen_verb_index: int
    chosen_target_index: int
    behavior: BehaviorMetadata
    reward: int
    terminal: bool
    diagnostic_ref: DiagnosticRef | None
    gamma: float
    mc_return: float
    schema_version: str = DATASET_EXAMPLE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATASET_EXAMPLE_VERSION:
            raise DatasetValidationError("example schema differs from runtime")
        _digest(self.environment_contract_sha256, "example environment digest")
        _digest(self.collection_contract_sha256, "example collection digest")
        _text(self.episode_key, "example episode_key")
        _text(self.problem_id, "example problem_id")
        if type(self.turn) is not int or self.turn < 0:
            raise DatasetValidationError("example turn is invalid")
        observation = _observation_record(
            self.observation_bytes, "example observation"
        )
        next_observation = _observation_record(
            self.next_observation_bytes, "example next_observation"
        )
        if type(self.current_features) is not StructuredFeatureBatch or type(
            self.next_features
        ) is not StructuredFeatureBatch:
            raise DatasetValidationError("example features have an invalid type")
        if (
            self.current_features.global_serialization.encode("ascii")
            != self.observation_bytes
            or self.next_features.global_serialization.encode("ascii")
            != self.next_observation_bytes
        ):
            raise DatasetValidationError("feature serialization changes observation")
        for features in (self.current_features, self.next_features):
            if features.environment_contract_sha256 != self.environment_contract_sha256:
                raise DatasetValidationError("feature environment digest differs")
        if observation.get("turn") != self.turn or next_observation.get(
            "turn"
        ) != self.turn + 1:
            raise DatasetValidationError("example observations are not contiguous")
        if not isinstance(self.action, Action):
            raise DatasetValidationError("example action has an invalid type")
        if type(self.chosen_verb_index) is not int or type(
            self.chosen_target_index
        ) is not int:
            raise DatasetValidationError("chosen action indices must be integers")
        try:
            indexed_action = (
                self.current_features.policy_verb_order[self.chosen_verb_index],
                self.current_features.target_ids[self.chosen_target_index],
            )
        except IndexError as exc:
            raise DatasetValidationError("chosen action index is out of range") from exc
        if indexed_action != (self.action.verb.value, self.action.target):
            raise DatasetValidationError("chosen indices change the action")
        if not self.current_features.legal_pair_mask[
            self.chosen_verb_index
        ][self.chosen_target_index]:
            raise DatasetValidationError("chosen action is masked")
        if not isinstance(self.behavior, BehaviorMetadata):
            raise DatasetValidationError("example behavior has an invalid type")
        if type(self.reward) is not int or self.reward not in {0, 1}:
            raise DatasetValidationError("example reward is invalid")
        if type(self.terminal) is not bool:
            raise DatasetValidationError("example terminal flag is invalid")
        if bool(next_observation.get("legal_pairs")) is self.terminal:
            raise DatasetValidationError("terminal flag differs from next legal mask")
        if self.diagnostic_ref is not None and not isinstance(
            self.diagnostic_ref, DiagnosticRef
        ):
            raise DatasetValidationError("diagnostic reference has an invalid type")
        _gamma(self.gamma)
        if (
            type(self.mc_return) is not float
            or not math.isfinite(self.mc_return)
            or self.mc_return < 0.0
        ):
            raise DatasetValidationError("Monte Carlo return is invalid")

    @property
    def bc_eligible(self) -> bool:
        """Behavior-cloning eligibility, derived only from provenance."""
        return self.behavior.bc_eligible

    @property
    def observation(self) -> dict[str, object]:
        return _observation_record(self.observation_bytes, "example observation")

    @property
    def next_observation(self) -> dict[str, object]:
        return _observation_record(
            self.next_observation_bytes, "example next_observation"
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "environment_contract_sha256": self.environment_contract_sha256,
            "collection_contract_sha256": self.collection_contract_sha256,
            "episode_key": self.episode_key,
            "problem_id": self.problem_id,
            "turn": self.turn,
            "observation": self.observation,
            "next_observation": self.next_observation,
            "current_features": _serialize_features(self.current_features),
            "next_features": _serialize_features(self.next_features),
            "action": serialize_action(self.action),
            "chosen_verb_index": self.chosen_verb_index,
            "chosen_target_index": self.chosen_target_index,
            "behavior": self.behavior.to_record(),
            "reward": self.reward,
            "terminal": self.terminal,
            "diagnostic_ref": (
                None if self.diagnostic_ref is None else self.diagnostic_ref.to_record()
            ),
            "gamma": self.gamma,
            "mc_return": self.mc_return,
        }


_TRUST_SEAL = object()
_SPLIT_TRUST_SEAL = object()


@dataclass(frozen=True, slots=True)
class PhaseBDataset:
    """A replay-gated, canonical Phase B dataset."""

    environment_contract_sha256: str
    collection_contract_sha256: str
    feature_contract_sha256: str
    gamma: float
    examples: tuple[PhaseBExample, ...]
    schema: str = DATASET_SCHEMA_VERSION
    _trust_seal: object = field(
        init=False, repr=False, compare=False, default=None
    )
    _trusted_record_sha256: str | None = field(
        init=False, repr=False, compare=False, default=None
    )

    def __post_init__(self) -> None:
        if self.schema != DATASET_SCHEMA_VERSION:
            raise DatasetValidationError("dataset schema differs from runtime")
        _digest(self.environment_contract_sha256, "dataset environment digest")
        _digest(self.collection_contract_sha256, "dataset collection digest")
        _digest(self.feature_contract_sha256, "dataset feature digest")
        _gamma(self.gamma)
        if type(self.examples) is not tuple or not self.examples:
            raise DatasetValidationError("dataset examples must be a nonempty tuple")
        previous_by_episode: dict[str, PhaseBExample] = {}
        closed: set[str] = set()
        episode_order: list[str] = []
        for example in self.examples:
            if not isinstance(example, PhaseBExample):
                raise DatasetValidationError("dataset contains an invalid example")
            if (
                example.environment_contract_sha256
                != self.environment_contract_sha256
                or example.collection_contract_sha256
                != self.collection_contract_sha256
                or example.gamma != self.gamma
            ):
                raise DatasetValidationError("example contract differs from dataset")
            previous = previous_by_episode.get(example.episode_key)
            if previous is None:
                if example.turn != 0 or example.episode_key in closed:
                    raise DatasetValidationError("episode order is incomplete")
                episode_order.append(example.episode_key)
            else:
                if previous.terminal or example.turn != previous.turn + 1:
                    raise DatasetValidationError("episode examples are not contiguous")
                if previous.next_observation_bytes != example.observation_bytes:
                    raise DatasetValidationError("episode observation chain differs")
                if previous.problem_id != example.problem_id:
                    raise DatasetValidationError("episode problem_id changed")
            previous_by_episode[example.episode_key] = example
            if example.terminal:
                closed.add(example.episode_key)
        if set(previous_by_episode) != closed:
            raise DatasetValidationError("dataset contains a truncated episode")
        for episode_key in episode_order:
            episode = [item for item in self.examples if item.episode_key == episode_key]
            future = 0.0
            for example in reversed(episode):
                expected = float(example.reward) + self.gamma * future
                if example.mc_return != expected:
                    raise DatasetValidationError("Monte Carlo return differs")
                future = expected

    def validate_trusted(self) -> None:
        if (
            self._trust_seal is not _TRUST_SEAL
            or self._trusted_record_sha256
            != canonical_json_sha256(self.to_record())
        ):
            raise DatasetValidationError("dataset was not produced from trusted replay")

    def to_record(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "environment_contract_sha256": self.environment_contract_sha256,
            "collection_contract_sha256": self.collection_contract_sha256,
            "feature_contract_sha256": self.feature_contract_sha256,
            "reward_contract": REWARD_CONTRACT_VERSION,
            "training_target": {
                "target_type": TRAINING_TARGET_VERSION,
                "gamma": self.gamma,
            },
            "examples": [example.to_record() for example in self.examples],
        }

    @property
    def dataset_sha256(self) -> str:
        self.validate_trusted()
        assert self._trusted_record_sha256 is not None
        return self._trusted_record_sha256

    def checkpoint_contract(self, split: GroupedSplit) -> dict[str, object]:
        """Return the exact checkpoint data binding after split validation."""
        self.validate_trusted()
        if type(split) is not GroupedSplit:
            raise TypeError("checkpoint contract needs an exact GroupedSplit")
        split.validate_trusted(self)
        return {
            "schema": self.schema,
            "dataset_sha256": self.dataset_sha256,
            "split_sha256": split.split_sha256,
        }


@dataclass(frozen=True, slots=True)
class GroupedSplit:
    """An exact problem-grouped train/validation assignment."""

    dataset_sha256: str
    validation_fraction: float
    seed: int
    assignments: tuple[tuple[str, str], ...]
    train_examples: tuple[PhaseBExample, ...]
    validation_examples: tuple[PhaseBExample, ...]
    schema: str = GROUPED_SPLIT_VERSION
    _trust_seal: object = field(
        init=False, repr=False, compare=False, default=None
    )
    _trusted_record_sha256: str | None = field(
        init=False, repr=False, compare=False, default=None
    )

    def __post_init__(self) -> None:
        _digest(self.dataset_sha256, "split dataset digest")
        if self.schema != GROUPED_SPLIT_VERSION:
            raise DatasetValidationError("split schema differs from runtime")
        if (
            type(self.validation_fraction) is not float
            or not math.isfinite(self.validation_fraction)
            or not 0.0 < self.validation_fraction < 1.0
        ):
            raise DatasetValidationError("validation_fraction must be in (0, 1)")
        if type(self.seed) is not int or self.seed < 0:
            raise DatasetValidationError("split seed is invalid")
        if not self.train_examples or not self.validation_examples:
            raise DatasetValidationError("split needs nonempty train and validation sets")
        assignment = dict(self.assignments)
        if len(assignment) != len(self.assignments) or set(assignment.values()) != {
            "train",
            "validation",
        }:
            raise DatasetValidationError("split assignments are invalid")
        train_problems = {item.problem_id for item in self.train_examples}
        validation_problems = {item.problem_id for item in self.validation_examples}
        if train_problems & validation_problems:
            raise DatasetValidationError("a problem crosses split groups")
        if any(assignment[item.problem_id] != "train" for item in self.train_examples):
            raise DatasetValidationError("train example assignment differs")
        if any(
            assignment[item.problem_id] != "validation"
            for item in self.validation_examples
        ):
            raise DatasetValidationError("validation example assignment differs")

    def to_record(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "dataset_sha256": self.dataset_sha256,
            "validation_fraction": self.validation_fraction,
            "seed": self.seed,
            "assignments": [
                {"problem_id": problem_id, "split": split}
                for problem_id, split in self.assignments
            ],
        }

    @property
    def train_problem_ids(self) -> tuple[str, ...]:
        return tuple(
            problem_id
            for problem_id, split in self.assignments
            if split == "train"
        )

    @property
    def validation_problem_ids(self) -> tuple[str, ...]:
        return tuple(
            problem_id
            for problem_id, split in self.assignments
            if split == "validation"
        )

    def validate_trusted(self, dataset: PhaseBDataset) -> None:
        if (
            self._trust_seal is not _SPLIT_TRUST_SEAL
            or self._trusted_record_sha256
            != canonical_json_sha256(self.to_record())
        ):
            raise DatasetValidationError(
                "split was not produced by the grouped split boundary"
            )
        dataset.validate_trusted()
        if self.dataset_sha256 != dataset.dataset_sha256:
            raise DatasetValidationError("split belongs to a different dataset")
        assignment = dict(self.assignments)
        problem_ids = {example.problem_id for example in dataset.examples}
        if set(assignment) != problem_ids:
            raise DatasetValidationError("split does not assign every problem")
        expected_train = tuple(
            example
            for example in dataset.examples
            if assignment[example.problem_id] == "train"
        )
        expected_validation = tuple(
            example
            for example in dataset.examples
            if assignment[example.problem_id] == "validation"
        )
        if (
            self.train_examples != expected_train
            or self.validation_examples != expected_validation
        ):
            raise DatasetValidationError("split example membership differs")

    @property
    def split_sha256(self) -> str:
        if (
            self._trust_seal is not _SPLIT_TRUST_SEAL
            or self._trusted_record_sha256
            != canonical_json_sha256(self.to_record())
        ):
            raise DatasetValidationError(
                "split was not produced by the grouped split boundary"
            )
        assert self._trusted_record_sha256 is not None
        return self._trusted_record_sha256


def _validate_replayed_run(run: ReplayedRun, environment: BoundEnvironment) -> None:
    if type(run) is not ReplayedRun:
        raise TypeError("conversion accepts only an exact ReplayedRun")
    if type(environment) is not BoundEnvironment:
        raise TypeError("conversion needs an exact BoundEnvironment")
    run.validate_trusted()
    environment.validate_binding()
    if run.environment_contract != environment.contract:
        raise DatasetValidationError("replayed environment contract differs")
    if not run.episodes:
        raise DatasetValidationError("replayed run contains no episodes")
    seen: set[str] = set()
    for episode in run.episodes:
        if episode.episode_key in seen or not episode.decisions:
            raise DatasetValidationError("replayed episode order is invalid")
        seen.add(episode.episode_key)
        for index, decision in enumerate(episode.decisions):
            transition = decision.transition
            environment.validate_transition(transition)
            if transition.state.turn != index:
                raise DatasetValidationError("replayed episode turn is not contiguous")
            if index and (
                episode.decisions[index - 1].transition.next_state
                != transition.state
            ):
                raise DatasetValidationError("replayed episode state chain differs")
            if transition.terminal != (index == len(episode.decisions) - 1):
                raise DatasetValidationError("replayed episode is not complete")
            if transition.collection_contract_sha256 != run.collection_contract.sha256:
                raise DatasetValidationError("transition collection contract differs")
            provenance = transition.decision_provenance
            if provenance is None or (
                provenance.source is not decision.behavior.source
                or provenance.fallback is not decision.behavior.fallback
                or provenance.bc_eligible is not decision.behavior.bc_eligible
            ):
                raise DatasetValidationError("decision provenance differs from replay")
            diagnostic_id = (
                None if decision.diagnostic_ref is None else decision.diagnostic_ref.record_id
            )
            if provenance.diagnostic_record_id != diagnostic_id:
                raise DatasetValidationError("decision diagnostic differs from replay")


def convert_replayed_run(
    run: ReplayedRun,
    environment: BoundEnvironment,
    gamma: float,
) -> PhaseBDataset:
    """Convert only exact, trusted symbolic replay into Phase B examples."""
    _gamma(gamma)
    _validate_replayed_run(run, environment)
    extractor = BoundFeatureExtractor(environment)
    examples: list[PhaseBExample] = []
    for episode in run.episodes:
        returns: list[float] = [0.0] * len(episode.decisions)
        future = 0.0
        for index in range(len(episode.decisions) - 1, -1, -1):
            future = float(episode.decisions[index].transition.reward) + gamma * future
            returns[index] = future
        for decision, mc_return in zip(episode.decisions, returns, strict=True):
            transition = decision.transition
            observation_bytes = canonical_json_bytes(environment.observe(transition.state))
            next_observation_bytes = canonical_json_bytes(
                environment.observe(transition.next_state)
            )
            current_features = extractor.build(json.loads(observation_bytes))
            next_features = extractor.build(json.loads(next_observation_bytes))
            try:
                verb_index = current_features.policy_verb_order.index(
                    transition.action.verb.value
                )
                target_index = current_features.target_ids.index(
                    transition.action.target
                )
            except ValueError as exc:
                raise DatasetValidationError("replayed action is absent from features") from exc
            examples.append(
                PhaseBExample(
                    environment_contract_sha256=environment.sha256,
                    collection_contract_sha256=run.collection_contract.sha256,
                    episode_key=episode.episode_key,
                    problem_id=episode.problem_id,
                    turn=transition.state.turn,
                    observation_bytes=observation_bytes,
                    next_observation_bytes=next_observation_bytes,
                    current_features=current_features,
                    next_features=next_features,
                    action=transition.action,
                    chosen_verb_index=verb_index,
                    chosen_target_index=target_index,
                    behavior=decision.behavior,
                    reward=transition.reward,
                    terminal=transition.terminal,
                    diagnostic_ref=decision.diagnostic_ref,
                    gamma=gamma,
                    mc_return=mc_return,
                )
            )
    candidate = PhaseBDataset(
        environment_contract_sha256=environment.sha256,
        collection_contract_sha256=run.collection_contract.sha256,
        feature_contract_sha256=canonical_json_sha256(feature_contract_record()),
        gamma=gamma,
        examples=tuple(examples),
    )
    object.__setattr__(candidate, "_trust_seal", _TRUST_SEAL)
    object.__setattr__(
        candidate,
        "_trusted_record_sha256",
        canonical_json_sha256(candidate.to_record()),
    )
    return candidate


def build_grouped_split(
    dataset: PhaseBDataset,
    validation_fraction: float,
    seed: int,
) -> GroupedSplit:
    """Assign complete problem groups by a deterministic seeded hash order."""
    if type(dataset) is not PhaseBDataset:
        raise TypeError("split needs an exact PhaseBDataset")
    dataset.validate_trusted()
    if (
        type(validation_fraction) is not float
        or not math.isfinite(validation_fraction)
        or not 0.0 < validation_fraction < 1.0
    ):
        raise DatasetValidationError("validation_fraction must be in (0, 1)")
    if type(seed) is not int or seed < 0:
        raise DatasetValidationError("split seed is invalid")
    problem_ids = sorted({example.problem_id for example in dataset.examples})
    if len(problem_ids) < 2:
        raise DatasetValidationError("grouped split needs at least two problems")
    ordered = sorted(
        problem_ids,
        key=lambda problem_id: (
            hashlib.sha256(
                f"{seed}\0{problem_id}".encode("utf-8")
            ).digest(),
            problem_id,
        ),
    )
    validation_count = min(
        len(problem_ids) - 1,
        max(1, int(math.floor(len(problem_ids) * validation_fraction + 0.5))),
    )
    validation = frozenset(ordered[:validation_count])
    assignments = tuple(
        (problem_id, "validation" if problem_id in validation else "train")
        for problem_id in problem_ids
    )
    candidate = GroupedSplit(
        dataset_sha256=dataset.dataset_sha256,
        validation_fraction=validation_fraction,
        seed=seed,
        assignments=assignments,
        train_examples=tuple(
            example for example in dataset.examples if example.problem_id not in validation
        ),
        validation_examples=tuple(
            example for example in dataset.examples if example.problem_id in validation
        ),
    )
    object.__setattr__(candidate, "_trust_seal", _SPLIT_TRUST_SEAL)
    object.__setattr__(
        candidate,
        "_trusted_record_sha256",
        canonical_json_sha256(candidate.to_record()),
    )
    return candidate


__all__ = [
    "DATASET_EXAMPLE_VERSION",
    "DATASET_SCHEMA_VERSION",
    "GROUPED_SPLIT_VERSION",
    "DatasetValidationError",
    "GroupedSplit",
    "PhaseBDataset",
    "PhaseBExample",
    "build_grouped_split",
    "convert_replayed_run",
]
