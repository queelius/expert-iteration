"""Exact structured feature input for the v2 controller.

This module stops at deterministic serialization and scalar extraction. Text
encoders and learned heads live outside the reasoning contract. Status,
ownership, and provenance are read only from typed observation fields.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .contracts import (
    CANONICAL_JSON_VERSION,
    canonical_json_bytes,
    canonical_json_sha256,
)
from .menu import POLICY_VERB_ORDER
from .model import ClaimStatus, GoalStatus
from .observation import (
    CONTROLLER_OBSERVATION_VERSION,
    ObservationValidationError,
    validate_observation_with_contract,
)
from .runtime import BoundEnvironment
from .verifier import GenerationContract, VerifierReceiptContract


FEATURE_CONTRACT_VERSION = "phase-b-features-v2"
TYPED_NODE_CONTEXT_VERSION = "typed-node-context-v2"
STRUCTURAL_SCALAR_FIELDS = (
    "turn_fraction",
    "remaining_turn_fraction",
    "log_node_count",
    "verified_claim_fraction",
    "solved_goal_fraction",
    "pruned_node_fraction",
)
STRUCTURAL_FEATURE_DIM = len(STRUCTURAL_SCALAR_FIELDS)

_FEATURE_CONTRACT_KEYS = frozenset(
    {
        "version",
        "global_serializer",
        "node_serializer",
        "structural_scalars",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class FeatureValidationError(ValueError):
    """Raised when observation or feature metadata is not exactly v2."""


def feature_contract_record() -> dict[str, object]:
    """Return a fresh JSON record for the one supported feature contract."""
    return {
        "version": FEATURE_CONTRACT_VERSION,
        "global_serializer": CANONICAL_JSON_VERSION,
        "node_serializer": TYPED_NODE_CONTEXT_VERSION,
        "structural_scalars": list(STRUCTURAL_SCALAR_FIELDS),
    }


def feature_contract_for_schema(
    observation_schema: str,
) -> dict[str, object]:
    """Return v2 feature metadata only for the exact observation schema."""
    if observation_schema != CONTROLLER_OBSERVATION_VERSION:
        raise FeatureValidationError(
            f"unsupported feature observation schema {observation_schema!r}"
        )
    return feature_contract_record()


def validate_feature_contract(contract: object) -> None:
    """Reject missing, unknown, or dimension-preserving contract drift."""
    if type(contract) is not dict:
        raise FeatureValidationError("feature contract must be a JSON object")
    try:
        canonical_json_bytes(contract)
    except ValueError as exc:
        raise FeatureValidationError(
            f"feature contract is not canonical JSON: {exc}"
        ) from exc
    if set(contract) != _FEATURE_CONTRACT_KEYS:
        raise FeatureValidationError(
            "feature contract keys must be exactly "
            f"{sorted(_FEATURE_CONTRACT_KEYS)!r}"
        )
    expected = feature_contract_record()
    for field in (
        "version",
        "global_serializer",
        "node_serializer",
        "structural_scalars",
    ):
        if contract[field] != expected[field]:
            raise FeatureValidationError(
                f"feature contract {field} differs from runtime"
            )


@dataclass(frozen=True, slots=True)
class StructuredFeatureBatch:
    """Immutable serializations and indices consumed by one controller step."""

    contract_version: str
    environment_contract_sha256: str
    global_serialization: str
    target_ids: tuple[int, ...]
    target_serializations: tuple[str, ...]
    structural_scalar_names: tuple[str, ...]
    structural_scalars: tuple[float, ...]
    policy_verb_order: tuple[str, ...]
    legal_pairs: tuple[tuple[str, int], ...]
    legal_pair_indices: tuple[tuple[int, int], ...]
    legal_pair_mask: tuple[tuple[bool, ...], ...]

    def __post_init__(self) -> None:
        if self.contract_version != FEATURE_CONTRACT_VERSION:
            raise ValueError("feature batch contract version differs from runtime")
        if _SHA256_RE.fullmatch(self.environment_contract_sha256) is None:
            raise ValueError("feature batch environment digest is invalid")
        if self.structural_scalar_names != STRUCTURAL_SCALAR_FIELDS:
            raise ValueError("feature batch scalar order differs from runtime")
        if len(self.structural_scalars) != STRUCTURAL_FEATURE_DIM:
            raise ValueError("feature batch scalar dimension differs from runtime")
        expected_verbs = tuple(verb.value for verb in POLICY_VERB_ORDER)
        if self.policy_verb_order != expected_verbs:
            raise ValueError("feature batch policy verb order differs from runtime")
        if len(self.target_ids) != len(self.target_serializations):
            raise ValueError("feature batch target serializations are misaligned")
        if self.target_ids != tuple(sorted(set(self.target_ids))):
            raise ValueError("feature batch targets must be unique and sorted")
        shape = (len(expected_verbs), len(self.target_ids))
        if len(self.legal_pair_mask) != shape[0] or any(
            len(row) != shape[1] for row in self.legal_pair_mask
        ):
            raise ValueError("feature batch legal pair mask has the wrong shape")
        if len(self.legal_pairs) != len(self.legal_pair_indices):
            raise ValueError("feature batch legal pair indices are misaligned")
        for pair, (verb_index, target_index) in zip(
            self.legal_pairs, self.legal_pair_indices, strict=True
        ):
            if not (0 <= verb_index < shape[0]) or not (
                0 <= target_index < shape[1]
            ):
                raise ValueError("feature batch legal pair index is out of range")
            if pair != (
                expected_verbs[verb_index], self.target_ids[target_index]
            ):
                raise ValueError("feature batch legal pair index changes the action")
            if not self.legal_pair_mask[verb_index][target_index]:
                raise ValueError("feature batch legal pair is absent from its mask")
        if sum(sum(row) for row in self.legal_pair_mask) != len(
            self.legal_pairs
        ):
            raise ValueError("feature batch mask contains an unlisted legal pair")


@dataclass(frozen=True, slots=True)
class BoundFeatureExtractor:
    """Live feature boundary with no per-decision semantic overrides."""

    environment: BoundEnvironment
    _contract_bytes: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.environment, BoundEnvironment):
            raise TypeError("feature environment has an invalid type")
        self.environment.validate_binding()
        contract = feature_contract_record()
        validate_feature_contract(contract)
        object.__setattr__(
            self, "_contract_bytes", canonical_json_bytes(contract)
        )

    @property
    def environment_contract_sha256(self) -> str:
        return self.environment.sha256

    @property
    def feature_contract(self) -> dict[str, object]:
        return json.loads(self._contract_bytes)

    @property
    def feature_contract_sha256(self) -> str:
        return canonical_json_sha256(self.feature_contract)

    def build(self, observation: object) -> StructuredFeatureBatch:
        self.environment.validate_binding()
        return build_structured_features(
            observation,
            self.environment.generation_contract,
            self.environment.verifier,
            environment_contract_sha256=self.environment.sha256,
            turn_limit=self.environment.contract.turn_limit,
            feature_contract=self.feature_contract,
        )


def _validated_snapshot(
    observation: object,
    generation_contract: GenerationContract,
    verifier_contract: VerifierReceiptContract,
    *,
    environment_contract_sha256: str,
    turn_limit: int,
) -> tuple[dict[str, Any], bytes]:
    if not isinstance(environment_contract_sha256, str) or (
        _SHA256_RE.fullmatch(environment_contract_sha256) is None
    ):
        raise FeatureValidationError(
            "environment contract digest must be 64 lowercase hex characters"
        )
    if type(turn_limit) is not int or turn_limit <= 0:
        raise FeatureValidationError("turn_limit must be a positive integer")
    try:
        encoded = canonical_json_bytes(observation)
        snapshot = json.loads(encoded)
        validate_observation_with_contract(
            snapshot,
            generation_contract,
            verifier_contract,
            environment_contract_sha256=environment_contract_sha256,
            turn_limit=turn_limit,
        )
    except (ObservationValidationError, TypeError, ValueError) as exc:
        raise FeatureValidationError(
            f"invalid controller observation: {exc}"
        ) from exc
    if snapshot["schema_version"] != CONTROLLER_OBSERVATION_VERSION:
        raise FeatureValidationError("observation schema differs from feature input")
    return snapshot, encoded


def _goal_ancestry(
    goals: dict[int, dict[str, Any]], target_id: int
) -> list[dict[str, Any]]:
    ancestry: list[dict[str, Any]] = []
    current: int | None = target_id
    while current is not None:
        goal = goals[current]
        ancestry.append(goal)
        current = goal["parent_goal"]
    ancestry.reverse()
    return ancestry


def _target_context(
    observation: dict[str, Any], target_id: int
) -> dict[str, object]:
    goals = {goal["id"]: goal for goal in observation["goals"]}
    goal = goals[target_id]
    ancestry = _goal_ancestry(goals, target_id)
    ancestry_ids = {item["id"] for item in ancestry}
    legal_verbs = [
        verb.value
        for verb in POLICY_VERB_ORDER
        if {"verb": verb.value, "target": target_id}
        in observation["legal_pairs"]
    ]
    previous = observation["previous_outcome"]
    if previous is not None and previous["action"]["target"] != target_id:
        previous = None
    return {
        "schema_version": TYPED_NODE_CONTEXT_VERSION,
        "target_kind": "goal",
        "target_id": target_id,
        "owner_goal_id": target_id,
        "target_record": goal,
        "goal_ancestry": ancestry,
        "children": [
            item
            for item in observation["goals"]
            if item["parent_goal"] == target_id
        ],
        "provenance": {
            "assumptions": list(goal["assumptions"]),
            "scoped_evidence": [
                item
                for item in observation["evidence"]
                if item["scope_goal"] in ancestry_ids
            ],
            "owned_claims": [
                item
                for item in observation["claims"]
                if item["owner_goal"] == target_id
            ],
        },
        "legal_verbs": legal_verbs,
        "previous_outcome": previous,
    }


def _structural_scalars(observation: dict[str, Any]) -> tuple[float, ...]:
    goals = observation["goals"]
    evidence = observation["evidence"]
    claims = observation["claims"]
    compact_claims = [
        goal["committed_claim"]
        for goal in goals
        if goal["committed_claim"] is not None
    ]
    claim_count = len(claims) + len(compact_claims)
    node_count = len(goals) + len(evidence) + claim_count
    verified_claim_count = len(compact_claims) + sum(
        claim["status"] == ClaimStatus.VERIFIED.value for claim in claims
    )
    solved_goal_count = sum(
        goal["status"] == GoalStatus.SOLVED.value for goal in goals
    )
    pruned_node_count = sum(
        goal["status"] == GoalStatus.PRUNED.value for goal in goals
    ) + sum(claim["status"] == ClaimStatus.PRUNED.value for claim in claims)
    turn = observation["turn"]
    turn_limit = observation["turn_limit"]
    return (
        turn / turn_limit,
        (turn_limit - turn) / turn_limit,
        math.log1p(node_count) / 4.0,
        verified_claim_count / max(1, claim_count),
        solved_goal_count / max(1, len(goals)),
        pruned_node_count / max(1, node_count),
    )


def build_structured_features(
    observation: object,
    generation_contract: GenerationContract,
    verifier_contract: VerifierReceiptContract,
    *,
    environment_contract_sha256: str,
    turn_limit: int,
    feature_contract: object,
) -> StructuredFeatureBatch:
    """Validate and serialize one exact stable-decision controller input."""
    validate_feature_contract(feature_contract)
    snapshot, global_bytes = _validated_snapshot(
        observation,
        generation_contract,
        verifier_contract,
        environment_contract_sha256=environment_contract_sha256,
        turn_limit=turn_limit,
    )

    legal_pairs = tuple(
        (pair["verb"], pair["target"]) for pair in snapshot["legal_pairs"]
    )
    target_ids = tuple(sorted({target for _, target in legal_pairs}))
    target_indexes = {
        target_id: index for index, target_id in enumerate(target_ids)
    }
    verb_order = tuple(verb.value for verb in POLICY_VERB_ORDER)
    verb_indexes = {verb: index for index, verb in enumerate(verb_order)}
    pair_indices = tuple(
        (verb_indexes[verb], target_indexes[target])
        for verb, target in legal_pairs
    )
    mask = [
        [False for _ in target_ids]
        for _ in POLICY_VERB_ORDER
    ]
    for verb_index, target_index in pair_indices:
        mask[verb_index][target_index] = True

    target_serializations = tuple(
        canonical_json_bytes(_target_context(snapshot, target_id)).decode("ascii")
        for target_id in target_ids
    )
    return StructuredFeatureBatch(
        contract_version=FEATURE_CONTRACT_VERSION,
        environment_contract_sha256=environment_contract_sha256,
        global_serialization=global_bytes.decode("ascii"),
        target_ids=target_ids,
        target_serializations=target_serializations,
        structural_scalar_names=STRUCTURAL_SCALAR_FIELDS,
        structural_scalars=_structural_scalars(snapshot),
        policy_verb_order=verb_order,
        legal_pairs=legal_pairs,
        legal_pair_indices=pair_indices,
        legal_pair_mask=tuple(tuple(row) for row in mask),
    )


__all__ = [
    "FEATURE_CONTRACT_VERSION",
    "BoundFeatureExtractor",
    "FeatureValidationError",
    "STRUCTURAL_FEATURE_DIM",
    "STRUCTURAL_SCALAR_FIELDS",
    "StructuredFeatureBatch",
    "TYPED_NODE_CONTEXT_VERSION",
    "build_structured_features",
    "feature_contract_for_schema",
    "feature_contract_record",
    "validate_feature_contract",
]
