"""Exact manifest-bound checkpoints for the reasoning v2 controller.

This module validates checkpoint dictionaries after a caller performs safe
deserialization. It intentionally does not import torch. The caller supplies
one narrow validator that checks the opaque state dictionary against the exact
model shape contract derived from the manifest.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .closure import CLOSURE_EVENTS_VERSION, CLOSURE_PRIORITY, CLOSURE_TURN_COST
from .contracts import (
    CANONICAL_JSON_VERSION,
    CollectionContract,
    EnvironmentContract,
    ModelIdentity,
    ModelTransportContract,
    canonical_json_bytes,
    canonical_json_sha256,
    validate_collection_binding,
)
from .features import (
    FEATURE_CONTRACT_VERSION,
    STRUCTURAL_FEATURE_DIM,
    FeatureValidationError,
    feature_contract_record,
    validate_feature_contract,
)
from .menu import (
    ACTION_SEMANTICS_VERSION,
    ACTION_TARGET_KINDS,
    POLICY_VERB_ORDER,
    VERB_ORDER,
)
from .observation import CONTROLLER_OBSERVATION_VERSION
from .records import RUN_HEADER_VERSION, TRANSITION_RECORD_VERSION
from .transition import REWARD_CONTRACT_VERSION, TRANSITION_SEMANTICS_VERSION


CHECKPOINT_SCHEMA_VERSION = "phase-b-controller-checkpoint-v2"
MANIFEST_SCHEMA_VERSION = "phase-b-contract-manifest-v2"
TRANSITION_OUTCOME_SCHEMA_VERSION = "transition-outcome-v2"
SYMBOLIC_RECEIPT_SCHEMA_VERSION = "symbolic-check-receipt-v2"
CONTROLLER_ARCHITECTURE_VERSION = "factored-controller-v2"
ACTING_SCORE_VERSION = "behavior-cloning-probability-v2"
TRAINING_TARGET_VERSION = "monte-carlo-return-v1"
DATASET_SCHEMA_VERSION = "phase-b-v2"
DECISION_RULE = "joint"
LEARNED_CONTROLLER_IMPLEMENTATION = (
    "verifier_anchored_rl.reasoning_v2.learned_controller."
    "FactoredLearnedController"
)
LEARNED_CONTROLLER_OUTPUT_CONTRACT = "factored-controller-choice-v2"
CHECKPOINT_CONTROLLER_PROVIDER = "checkpoint"
LEARNED_CONTROLLER_MAX_TOKENS = 64

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ENCODER_TYPES = frozenset({"hash", "frozen_embedding", "zero_text"})

_CHECKPOINT_KEYS = frozenset({
    "checkpoint_schema",
    "state_dict",
    "contract_manifest",
    "contract_sha256",
})
_MANIFEST_KEYS = frozenset({
    "manifest_schema",
    "observation_contract",
    "transition_contract",
    "action_contract",
    "environment_contract",
    "environment_contract_sha256",
    "training_collection_contract",
    "training_collection_contract_sha256",
    "feature_contract",
    "encoder_contract",
    "policy_contract",
    "training_target_contract",
    "dataset_contract",
})
_OBSERVATION_KEYS = frozenset({
    "schema",
    "canonical_serializer",
    "outcome_schema",
    "receipt_schema",
})
_TRANSITION_KEYS = frozenset({
    "run_header_schema",
    "record_schema",
    "semantics",
    "closure_schema",
    "closure_priority",
    "closure_turn_cost",
    "closure_bc_eligible",
    "previous_outcome_source",
})
_ACTION_KEYS = frozenset({
    "semantics",
    "algebra_verb_order",
    "policy_verb_order",
    "target_kinds",
})
_ENCODER_KEYS = frozenset({
    "type",
    "implementation",
    "algorithm_contract",
    "model",
    "model_digest",
    "dimension",
})
_POLICY_KEYS = frozenset({
    "architecture",
    "model_config",
    "feature_dim",
    "state_dict_sha256",
    "decision_rule",
    "acting_score",
})
_MODEL_CONFIG_KEYS = frozenset({
    "hidden_dim",
    "repr_dim",
    "node_repr_dim",
    "verb_embed_dim",
})
_TRAINING_TARGET_KEYS = frozenset({"target_type", "gamma", "reward_contract"})
_DATASET_KEYS = frozenset({"schema", "dataset_sha256", "split_sha256"})


class CheckpointValidationError(ValueError):
    """Raised when a v2 checkpoint is not exactly contract-bound."""


@dataclass(frozen=True, slots=True)
class ModelShapeContract:
    """Shape-determining values passed to the state-dictionary validator."""

    architecture: str
    feature_dim: int
    policy_verb_count: int
    hidden_dim: int
    repr_dim: int
    node_repr_dim: int
    verb_embed_dim: int


ModelStateValidator = Callable[[Mapping[str, object], ModelShapeContract], None]
ModelStateDigester = Callable[[Mapping[str, object]], str]


@dataclass(frozen=True, slots=True)
class ValidatedCheckpoint:
    """Trusted bindings returned after complete metadata and shape checks."""

    state_dict: Mapping[str, object]
    state_dict_sha256: str
    contract_sha256: str
    environment_contract: EnvironmentContract
    training_collection_contract: CollectionContract
    live_collection_contract: CollectionContract
    model_shape: ModelShapeContract


def _exact_object(
    value: object, keys: frozenset[str], path: str
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise CheckpointValidationError(
            f"{path} keys must be exactly {sorted(keys)!r}"
        )
    return value


def _sha256(value: object, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CheckpointValidationError(f"{path} must be a sha256 digest")
    return value


def _positive_integer(value: object, path: str) -> int:
    if type(value) is not int or value <= 0:
        raise CheckpointValidationError(f"{path} must be a positive integer")
    return value


def _optional_text(value: object, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CheckpointValidationError(f"{path} must be null or nonempty text")
    return value


def _snapshot_json(value: object, path: str) -> dict[str, Any]:
    try:
        encoded = canonical_json_bytes(value)
        snapshot = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise CheckpointValidationError(
            f"{path} is not canonical JSON: {exc}"
        ) from exc
    if type(snapshot) is not dict:
        raise CheckpointValidationError(f"{path} must be a JSON object")
    return snapshot


def _fixed_observation_contract() -> dict[str, object]:
    return {
        "schema": CONTROLLER_OBSERVATION_VERSION,
        "canonical_serializer": CANONICAL_JSON_VERSION,
        "outcome_schema": TRANSITION_OUTCOME_SCHEMA_VERSION,
        "receipt_schema": SYMBOLIC_RECEIPT_SCHEMA_VERSION,
    }


def _fixed_transition_contract() -> dict[str, object]:
    return {
        "run_header_schema": RUN_HEADER_VERSION,
        "record_schema": TRANSITION_RECORD_VERSION,
        "semantics": TRANSITION_SEMANTICS_VERSION,
        "closure_schema": CLOSURE_EVENTS_VERSION,
        "closure_priority": [verb.value for verb in CLOSURE_PRIORITY],
        "closure_turn_cost": CLOSURE_TURN_COST,
        "closure_bc_eligible": False,
        "previous_outcome_source": "learned_action",
    }


def _fixed_action_contract() -> dict[str, object]:
    return {
        "semantics": ACTION_SEMANTICS_VERSION,
        "algebra_verb_order": [verb.value for verb in VERB_ORDER],
        "policy_verb_order": [verb.value for verb in POLICY_VERB_ORDER],
        "target_kinds": {
            verb.value: list(ACTION_TARGET_KINDS[verb.value])
            for verb in VERB_ORDER
        },
    }


def _validate_encoder(value: object, path: str) -> dict[str, Any]:
    record = _exact_object(value, _ENCODER_KEYS, path)
    encoder_type = record["type"]
    if encoder_type not in _ENCODER_TYPES:
        raise CheckpointValidationError(f"{path}.type is unsupported")
    for field in ("implementation", "algorithm_contract"):
        if _optional_text(record[field], f"{path}.{field}") is None:
            raise CheckpointValidationError(
                f"{path}.{field} must be nonempty text"
            )
    model = _optional_text(record["model"], f"{path}.model")
    model_digest = _optional_text(
        record["model_digest"], f"{path}.model_digest"
    )
    if model_digest is not None:
        _sha256(model_digest, f"{path}.model_digest")
    _positive_integer(record["dimension"], f"{path}.dimension")
    if encoder_type == "frozen_embedding":
        if model is None or model_digest is None:
            raise CheckpointValidationError(
                f"{path} frozen_embedding requires model and model_digest"
            )
    elif model is not None or model_digest is not None:
        raise CheckpointValidationError(
            f"{path} {encoder_type} cannot carry model identity"
        )
    return record


def _validate_model_config(value: object, path: str) -> dict[str, Any]:
    record = _exact_object(value, _MODEL_CONFIG_KEYS, path)
    for field in sorted(_MODEL_CONFIG_KEYS):
        _positive_integer(record[field], f"{path}.{field}")
    return record


def _validate_training_target(value: object) -> dict[str, Any]:
    record = _exact_object(
        value, _TRAINING_TARGET_KEYS, "manifest.training_target_contract"
    )
    if record["target_type"] != TRAINING_TARGET_VERSION:
        raise CheckpointValidationError("training target type differs from runtime")
    gamma = record["gamma"]
    if (
        type(gamma) is not float
        or not math.isfinite(gamma)
        or gamma < 0.0
        or gamma > 1.0
    ):
        raise CheckpointValidationError("training target gamma is invalid")
    if record["reward_contract"] != REWARD_CONTRACT_VERSION:
        raise CheckpointValidationError(
            "training reward contract differs from runtime"
        )
    return record


def _validate_dataset(value: object) -> dict[str, Any]:
    record = _exact_object(value, _DATASET_KEYS, "manifest.dataset_contract")
    if record["schema"] != DATASET_SCHEMA_VERSION:
        raise CheckpointValidationError("dataset schema differs from runtime")
    _sha256(record["dataset_sha256"], "dataset dataset_sha256")
    _sha256(record["split_sha256"], "dataset split_sha256")
    return record


def _parse_manifest(
    value: object,
    *,
    environment_contract: EnvironmentContract,
    encoder_contract: object,
) -> tuple[
    dict[str, Any],
    EnvironmentContract,
    CollectionContract,
    ModelShapeContract,
    str,
]:
    if not isinstance(environment_contract, EnvironmentContract):
        raise TypeError("live environment_contract has an invalid type")

    manifest = _snapshot_json(value, "contract_manifest")
    manifest = _exact_object(manifest, _MANIFEST_KEYS, "contract_manifest")
    if manifest["manifest_schema"] != MANIFEST_SCHEMA_VERSION:
        raise CheckpointValidationError("manifest schema differs from runtime")

    fixed_contracts = (
        (
            "observation_contract",
            _OBSERVATION_KEYS,
            _fixed_observation_contract(),
        ),
        (
            "transition_contract",
            _TRANSITION_KEYS,
            _fixed_transition_contract(),
        ),
        ("action_contract", _ACTION_KEYS, _fixed_action_contract()),
    )
    for field, keys, expected in fixed_contracts:
        actual = _exact_object(manifest[field], keys, f"manifest.{field}")
        if actual != expected:
            raise CheckpointValidationError(
                f"manifest {field} differs from runtime"
            )

    try:
        saved_environment = EnvironmentContract.from_record(
            manifest["environment_contract"]
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointValidationError(
            f"manifest environment contract is invalid: {exc}"
        ) from exc
    environment_digest = _sha256(
        manifest["environment_contract_sha256"],
        "manifest environment_contract_sha256",
    )
    if environment_digest != saved_environment.sha256:
        raise CheckpointValidationError("environment contract digest mismatch")
    if (
        saved_environment.to_record() != environment_contract.to_record()
        or environment_digest != environment_contract.sha256
    ):
        raise CheckpointValidationError(
            "live environment differs from checkpoint"
        )

    try:
        training_collection = CollectionContract.from_record(
            manifest["training_collection_contract"]
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointValidationError(
            f"manifest training collection contract is invalid: {exc}"
        ) from exc
    collection_digest = _sha256(
        manifest["training_collection_contract_sha256"],
        "manifest training_collection_contract_sha256",
    )
    if collection_digest != training_collection.sha256:
        raise CheckpointValidationError(
            "training collection contract digest mismatch"
        )
    if training_collection.behavior_source != "teacher":
        raise CheckpointValidationError(
            "training collection must identify teacher behavior"
        )
    try:
        validate_collection_binding(saved_environment, training_collection)
    except (TypeError, ValueError) as exc:
        raise CheckpointValidationError(
            f"environment and collection contracts are incompatible: {exc}"
        ) from exc

    try:
        validate_feature_contract(manifest["feature_contract"])
    except FeatureValidationError as exc:
        raise CheckpointValidationError(
            f"manifest feature contract is invalid: {exc}"
        ) from exc

    saved_encoder = _validate_encoder(
        manifest["encoder_contract"], "manifest.encoder_contract"
    )
    live_encoder = _validate_encoder(
        _snapshot_json(encoder_contract, "live encoder_contract"),
        "live encoder_contract",
    )
    if saved_encoder != live_encoder:
        raise CheckpointValidationError(
            "live encoder differs from checkpoint"
        )

    policy = _exact_object(
        manifest["policy_contract"], _POLICY_KEYS, "manifest.policy_contract"
    )
    if policy["architecture"] != CONTROLLER_ARCHITECTURE_VERSION:
        raise CheckpointValidationError("policy architecture differs from runtime")
    model_config = _validate_model_config(
        policy["model_config"], "manifest.policy_contract.model_config"
    )
    feature_dim = _positive_integer(
        policy["feature_dim"], "manifest.policy_contract.feature_dim"
    )
    state_dict_sha256 = _sha256(
        policy["state_dict_sha256"],
        "manifest.policy_contract.state_dict_sha256",
    )
    expected_feature_dim = saved_encoder["dimension"] + STRUCTURAL_FEATURE_DIM
    if feature_dim != expected_feature_dim:
        raise CheckpointValidationError(
            "policy feature_dim differs from encoder plus structural features"
        )
    if policy["decision_rule"] != DECISION_RULE:
        raise CheckpointValidationError("policy decision rule differs from runtime")
    if policy["acting_score"] != ACTING_SCORE_VERSION:
        raise CheckpointValidationError("policy acting score differs from runtime")

    _validate_training_target(manifest["training_target_contract"])
    _validate_dataset(manifest["dataset_contract"])
    shape = ModelShapeContract(
        architecture=policy["architecture"],
        feature_dim=feature_dim,
        policy_verb_count=len(POLICY_VERB_ORDER),
        hidden_dim=model_config["hidden_dim"],
        repr_dim=model_config["repr_dim"],
        node_repr_dim=model_config["node_repr_dim"],
        verb_embed_dim=model_config["verb_embed_dim"],
    )
    return (
        manifest,
        saved_environment,
        training_collection,
        shape,
        state_dict_sha256,
    )


def validate_deployment_collection(
    training_collection_contract: CollectionContract,
    live_collection_contract: CollectionContract,
    *,
    environment_contract: EnvironmentContract,
    checkpoint_sha256: str,
) -> None:
    """Require a checkpoint-backed live policy and compatible generator."""
    if not isinstance(training_collection_contract, CollectionContract):
        raise TypeError("training_collection_contract has an invalid type")
    if not isinstance(live_collection_contract, CollectionContract):
        raise TypeError("live_collection_contract has an invalid type")
    if not isinstance(environment_contract, EnvironmentContract):
        raise TypeError("environment_contract has an invalid type")
    if training_collection_contract.behavior_source != "teacher":
        raise CheckpointValidationError(
            "training collection must identify teacher behavior"
        )
    checkpoint_digest = _sha256(checkpoint_sha256, "checkpoint_sha256")
    try:
        validate_collection_binding(
            environment_contract, live_collection_contract
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointValidationError(
            f"live collection is incompatible with the environment: {exc}"
        ) from exc

    generator_fields = (
        "generator",
        "generator_implementation",
        "generator_transport",
        "generation_seed",
    )
    for field in generator_fields:
        if getattr(live_collection_contract, field) != getattr(
            training_collection_contract, field
        ):
            raise CheckpointValidationError(
                f"live collection {field} differs from training"
            )

    expected_identity = ModelIdentity(
        CHECKPOINT_CONTROLLER_PROVIDER,
        CONTROLLER_ARCHITECTURE_VERSION,
        checkpoint_digest,
    )
    expected_transport = ModelTransportContract.local(
        LEARNED_CONTROLLER_IMPLEMENTATION
    )
    expected_controller_fields: tuple[tuple[str, object], ...] = (
        ("controller", expected_identity),
        ("controller_implementation", LEARNED_CONTROLLER_IMPLEMENTATION),
        ("controller_transport", expected_transport),
        ("controller_seed", None),
        ("controller_temperature", 0.0),
        ("controller_max_tokens", LEARNED_CONTROLLER_MAX_TOKENS),
        ("controller_seed_policy", "none-v1"),
        (
            "controller_output_contract",
            LEARNED_CONTROLLER_OUTPUT_CONTRACT,
        ),
        ("behavior_source", "learned_policy"),
        ("controller_fallback_policy", "forbidden-v1"),
        ("policy_input_contract", FEATURE_CONTRACT_VERSION),
    )
    for field, expected in expected_controller_fields:
        if getattr(live_collection_contract, field) != expected:
            raise CheckpointValidationError(
                f"live collection {field} is not checkpoint-compatible"
            )


def build_contract_manifest(
    *,
    state_dict: Mapping[str, object],
    model_state_digester: ModelStateDigester,
    environment_contract: EnvironmentContract,
    training_collection_contract: CollectionContract,
    encoder_contract: object,
    model_config: object,
    feature_dim: int,
    training_target_contract: object,
    dataset_contract: object,
) -> dict[str, object]:
    """Build and validate the one exact v2 checkpoint manifest."""
    if not isinstance(environment_contract, EnvironmentContract):
        raise TypeError("environment_contract has an invalid type")
    if not isinstance(training_collection_contract, CollectionContract):
        raise TypeError("training_collection_contract has an invalid type")
    encoder = _snapshot_json(encoder_contract, "encoder_contract")
    config = _snapshot_json(model_config, "model_config")
    training = _snapshot_json(training_target_contract, "training_target_contract")
    dataset = _snapshot_json(dataset_contract, "dataset_contract")
    state_dict_sha256 = _digest_model_state(
        state_dict, model_state_digester
    )
    manifest: dict[str, object] = {
        "manifest_schema": MANIFEST_SCHEMA_VERSION,
        "observation_contract": _fixed_observation_contract(),
        "transition_contract": _fixed_transition_contract(),
        "action_contract": _fixed_action_contract(),
        "environment_contract": environment_contract.to_record(),
        "environment_contract_sha256": environment_contract.sha256,
        "training_collection_contract": (
            training_collection_contract.to_record()
        ),
        "training_collection_contract_sha256": (
            training_collection_contract.sha256
        ),
        "feature_contract": feature_contract_record(),
        "encoder_contract": encoder,
        "policy_contract": {
            "architecture": CONTROLLER_ARCHITECTURE_VERSION,
            "model_config": config,
            "feature_dim": feature_dim,
            "state_dict_sha256": state_dict_sha256,
            "decision_rule": DECISION_RULE,
            "acting_score": ACTING_SCORE_VERSION,
        },
        "training_target_contract": training,
        "dataset_contract": dataset,
    }
    snapshot, _, _, _, _ = _parse_manifest(
        manifest,
        environment_contract=environment_contract,
        encoder_contract=encoder,
    )
    return snapshot


def validate_checkpoint(
    checkpoint: object,
    *,
    environment_contract: EnvironmentContract,
    live_collection_contract: CollectionContract,
    encoder_contract: object,
    model_state_validator: ModelStateValidator,
    model_state_digester: ModelStateDigester,
) -> ValidatedCheckpoint:
    """Validate one already-deserialized v2 checkpoint without defaults."""
    record = _exact_object(checkpoint, _CHECKPOINT_KEYS, "checkpoint")
    if record["checkpoint_schema"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointValidationError("checkpoint schema differs from v2")
    digest = _sha256(record["contract_sha256"], "checkpoint contract_sha256")
    try:
        actual_digest = canonical_json_sha256(record["contract_manifest"])
    except (TypeError, ValueError) as exc:
        raise CheckpointValidationError(
            f"checkpoint manifest is not canonical JSON: {exc}"
        ) from exc
    if digest != actual_digest:
        raise CheckpointValidationError("checkpoint manifest digest mismatch")

    (
        _,
        saved_environment,
        training_collection,
        shape,
        expected_state_digest,
    ) = _parse_manifest(
        record["contract_manifest"],
        environment_contract=environment_contract,
        encoder_contract=encoder_contract,
    )
    validate_deployment_collection(
        training_collection,
        live_collection_contract,
        environment_contract=saved_environment,
        checkpoint_sha256=digest,
    )
    state_dict = _validate_model_state(
        record["state_dict"], shape, model_state_validator
    )
    actual_state_digest = _digest_model_state(
        state_dict, model_state_digester
    )
    if actual_state_digest != expected_state_digest:
        raise CheckpointValidationError("checkpoint state_dict digest mismatch")
    return ValidatedCheckpoint(
        state_dict=state_dict,
        state_dict_sha256=expected_state_digest,
        contract_sha256=digest,
        environment_contract=saved_environment,
        training_collection_contract=training_collection,
        live_collection_contract=live_collection_contract,
        model_shape=shape,
    )


def _validate_model_state(
    value: object,
    shape: ModelShapeContract,
    model_state_validator: ModelStateValidator,
) -> Mapping[str, object]:
    state_dict = _validate_state_mapping(value)
    if not callable(model_state_validator):
        raise TypeError("model_state_validator must be callable")
    try:
        model_state_validator(state_dict, shape)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise CheckpointValidationError(
            f"model state validation failed: {exc}"
        ) from exc
    return state_dict


def _validate_state_mapping(value: object) -> Mapping[str, object]:
    state_dict = value
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise CheckpointValidationError(
            "checkpoint state_dict must be a nonempty mapping"
        )
    if any(not isinstance(key, str) or not key for key in state_dict):
        raise CheckpointValidationError(
            "checkpoint state_dict keys must be nonempty text"
        )
    return state_dict


def _digest_model_state(
    value: object,
    model_state_digester: ModelStateDigester,
) -> str:
    state_dict = _validate_state_mapping(value)
    if not callable(model_state_digester):
        raise TypeError("model_state_digester must be callable")
    try:
        digest = model_state_digester(state_dict)
        repeated_digest = model_state_digester(state_dict)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise CheckpointValidationError(
            f"model state digest failed: {exc}"
        ) from exc
    digest = _sha256(digest, "model state digest")
    repeated_digest = _sha256(
        repeated_digest, "repeated model state digest"
    )
    if digest != repeated_digest:
        raise CheckpointValidationError(
            "model_state_digester is not deterministic"
        )
    return digest


def build_checkpoint(
    *,
    state_dict: Mapping[str, object],
    environment_contract: EnvironmentContract,
    training_collection_contract: CollectionContract,
    encoder_contract: object,
    model_config: object,
    feature_dim: int,
    training_target_contract: object,
    dataset_contract: object,
    model_state_validator: ModelStateValidator,
    model_state_digester: ModelStateDigester,
) -> dict[str, object]:
    """Build a v2 checkpoint and validate its model state before returning."""
    manifest = build_contract_manifest(
        environment_contract=environment_contract,
        training_collection_contract=training_collection_contract,
        state_dict=state_dict,
        model_state_digester=model_state_digester,
        encoder_contract=encoder_contract,
        model_config=model_config,
        feature_dim=feature_dim,
        training_target_contract=training_target_contract,
        dataset_contract=dataset_contract,
    )
    checkpoint: dict[str, object] = {
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        "state_dict": state_dict,
        "contract_manifest": manifest,
        "contract_sha256": canonical_json_sha256(manifest),
    }
    _, _, _, shape, _ = _parse_manifest(
        manifest,
        environment_contract=environment_contract,
        encoder_contract=encoder_contract,
    )
    _validate_model_state(state_dict, shape, model_state_validator)
    return checkpoint


__all__ = [
    "ACTING_SCORE_VERSION",
    "CHECKPOINT_CONTROLLER_PROVIDER",
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointValidationError",
    "CONTROLLER_ARCHITECTURE_VERSION",
    "DATASET_SCHEMA_VERSION",
    "DECISION_RULE",
    "LEARNED_CONTROLLER_IMPLEMENTATION",
    "LEARNED_CONTROLLER_MAX_TOKENS",
    "LEARNED_CONTROLLER_OUTPUT_CONTRACT",
    "MANIFEST_SCHEMA_VERSION",
    "ModelShapeContract",
    "ModelStateDigester",
    "ModelStateValidator",
    "SYMBOLIC_RECEIPT_SCHEMA_VERSION",
    "TRAINING_TARGET_VERSION",
    "TRANSITION_OUTCOME_SCHEMA_VERSION",
    "ValidatedCheckpoint",
    "build_checkpoint",
    "build_contract_manifest",
    "validate_checkpoint",
    "validate_deployment_collection",
]
