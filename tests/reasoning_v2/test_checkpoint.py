from copy import deepcopy
from dataclasses import dataclass, replace

import pytest

from verifier_anchored_rl.reasoning_v2.checkpoint import (
    ACTING_SCORE_VERSION,
    CHECKPOINT_CONTROLLER_PROVIDER,
    CHECKPOINT_SCHEMA_VERSION,
    CONTROLLER_ARCHITECTURE_VERSION,
    DATASET_SCHEMA_VERSION,
    DECISION_RULE,
    LEARNED_CONTROLLER_IMPLEMENTATION,
    LEARNED_CONTROLLER_MAX_TOKENS,
    LEARNED_CONTROLLER_OUTPUT_CONTRACT,
    MANIFEST_SCHEMA_VERSION,
    TRAINING_TARGET_VERSION,
    CheckpointValidationError,
    ModelShapeContract,
    build_checkpoint,
    build_contract_manifest as build_contract_manifest_artifact,
    validate_checkpoint as validate_checkpoint_artifact,
)
from verifier_anchored_rl.reasoning_v2.closure import (
    CLOSURE_EVENTS_VERSION,
    CLOSURE_PRIORITY,
    CLOSURE_TURN_COST,
)
from verifier_anchored_rl.reasoning_v2.contracts import (
    CANONICAL_JSON_VERSION,
    CollectionContract,
    ModelIdentity,
    ModelTransportContract,
    build_environment_contract,
    canonical_json_sha256,
)
from verifier_anchored_rl.reasoning_v2.features import (
    FEATURE_CONTRACT_VERSION,
    STRUCTURAL_FEATURE_DIM,
    feature_contract_record,
)
from verifier_anchored_rl.reasoning_v2.lenses import (
    GENERATION_SCHEMA_VERSION,
    LENS_CONTRACT_VERSION,
)
from verifier_anchored_rl.reasoning_v2.math_adapter import CaseworkMathVerifier
from verifier_anchored_rl.reasoning_v2.menu import (
    ACTION_SEMANTICS_VERSION,
    ACTION_TARGET_KINDS,
    POLICY_VERB_ORDER,
    VERB_ORDER,
)
from verifier_anchored_rl.reasoning_v2.observation import (
    CONTROLLER_OBSERVATION_VERSION,
)
from verifier_anchored_rl.reasoning_v2.records import (
    RUN_HEADER_VERSION,
    TRANSITION_RECORD_VERSION,
)
from verifier_anchored_rl.reasoning_v2.transition import (
    REWARD_CONTRACT_VERSION,
    TRANSITION_SEMANTICS_VERSION,
)


ENCODER = {
    "type": "frozen_embedding",
    "implementation": "tests.FrozenEncoder",
    "algorithm_contract": "frozen-embedding-vector-v1",
    "model": "test-encoder",
    "model_digest": "d" * 64,
    "dimension": 8,
}
MODEL_CONFIG = {
    "hidden_dim": 16,
    "repr_dim": 8,
    "node_repr_dim": 6,
    "verb_embed_dim": 4,
}
FEATURE_DIM = ENCODER["dimension"] + STRUCTURAL_FEATURE_DIM
TRAINING_TARGET = {
    "target_type": TRAINING_TARGET_VERSION,
    "gamma": 1.0,
    "reward_contract": REWARD_CONTRACT_VERSION,
}
DATASET = {
    "schema": DATASET_SCHEMA_VERSION,
    "dataset_sha256": "a" * 64,
    "split_sha256": "b" * 64,
}


@dataclass(frozen=True)
class FakeTensor:
    shape: tuple[int, ...]
    values: tuple[float, ...] = ()


def environment_contract():
    adapter = CaseworkMathVerifier()
    return build_environment_contract(
        generation_contract=adapter,
        verifier=adapter,
        aggregate_builder=adapter,
        lens_contract=LENS_CONTRACT_VERSION,
        generation_schema=GENERATION_SCHEMA_VERSION,
        generation_temperature=0.0,
        generation_max_tokens=96,
        generation_seed_policy="none-v1",
        turn_limit=8,
    )


def training_collection_contract():
    generator_implementation = "tests.StaticGenerator"
    controller_implementation = "tests.StaticTeacher"
    return CollectionContract(
        generator=ModelIdentity("test", "generator", "r1"),
        controller=ModelIdentity("test", "controller", "r1"),
        controller_implementation=controller_implementation,
        controller_transport=ModelTransportContract.local(
            controller_implementation
        ),
        controller_seed=None,
        behavior_source="teacher",
        controller_fallback_policy="first-legal-pair-v1",
        generator_implementation=generator_implementation,
        generator_transport=ModelTransportContract.local(
            generator_implementation
        ),
        generation_seed=None,
        policy_input_contract="teacher-presentation-v2",
        problem_source_contract="checkpoint-test-problems-v1",
        problem_source_sha256="c" * 64,
        problem_adapter_implementation="tests.CheckpointProblemAdapter",
    )


def deployment_collection_contract(
    checkpoint_sha256, training_collection, *, problem_digest="e" * 64
):
    return CollectionContract(
        generator=training_collection.generator,
        controller=ModelIdentity(
            CHECKPOINT_CONTROLLER_PROVIDER,
            CONTROLLER_ARCHITECTURE_VERSION,
            checkpoint_sha256,
        ),
        controller_implementation=LEARNED_CONTROLLER_IMPLEMENTATION,
        controller_transport=ModelTransportContract.local(
            LEARNED_CONTROLLER_IMPLEMENTATION
        ),
        controller_seed=None,
        behavior_source="learned_policy",
        controller_fallback_policy="forbidden-v1",
        generator_implementation=(
            training_collection.generator_implementation
        ),
        generator_transport=training_collection.generator_transport,
        generation_seed=training_collection.generation_seed,
        policy_input_contract=FEATURE_CONTRACT_VERSION,
        problem_source_contract="checkpoint-eval-problems-v1",
        problem_source_sha256=problem_digest,
        problem_adapter_implementation="tests.EvalProblemAdapter",
        controller_temperature=0.0,
        controller_max_tokens=LEARNED_CONTROLLER_MAX_TOKENS,
        controller_seed_policy="none-v1",
        controller_output_contract=LEARNED_CONTROLLER_OUTPUT_CONTRACT,
    )


def state_dict(*, feature_dim=FEATURE_DIM):
    return {
        "global.weight": FakeTensor(
            (MODEL_CONFIG["hidden_dim"], feature_dim), (1.0,)
        ),
        "verb.weight": FakeTensor(
            (len(POLICY_VERB_ORDER), MODEL_CONFIG["repr_dim"]),
            (2.0,),
        ),
        "verb.embedding": FakeTensor(
            (len(POLICY_VERB_ORDER), MODEL_CONFIG["verb_embed_dim"]),
            (3.0,),
        ),
    }


def shape_validator(saved_state, shape):
    if saved_state["global.weight"].shape != (
        shape.hidden_dim,
        shape.feature_dim,
    ):
        raise ValueError("global adapter shape mismatch")
    if saved_state["verb.weight"].shape != (
        shape.policy_verb_count,
        shape.repr_dim,
    ):
        raise ValueError("verb head shape mismatch")
    if saved_state["verb.embedding"].shape != (
        shape.policy_verb_count,
        shape.verb_embed_dim,
    ):
        raise ValueError("verb embedding shape mismatch")


def state_digester(saved_state):
    return canonical_json_sha256({
        key: {
            "shape": list(tensor.shape),
            "values": list(tensor.values),
        }
        for key, tensor in sorted(saved_state.items())
    })


def build_contract_manifest(
    *, state_dict_value=None, model_state_digester=state_digester, **kwargs
):
    return build_contract_manifest_artifact(
        state_dict=(state_dict() if state_dict_value is None else state_dict_value),
        model_state_digester=model_state_digester,
        **kwargs,
    )


def make_checkpoint(*, env=None, collection=None, saved_state=None):
    env = environment_contract() if env is None else env
    collection = (
        training_collection_contract()
        if collection is None
        else collection
    )
    checkpoint = build_checkpoint(
        state_dict=state_dict() if saved_state is None else saved_state,
        environment_contract=env,
        training_collection_contract=collection,
        encoder_contract=ENCODER,
        model_config=MODEL_CONFIG,
        feature_dim=FEATURE_DIM,
        training_target_contract=TRAINING_TARGET,
        dataset_contract=DATASET,
        model_state_validator=shape_validator,
        model_state_digester=state_digester,
    )
    return checkpoint, env, collection


def validate_checkpoint(
    checkpoint,
    *,
    environment_contract,
    training_collection_contract=None,
    encoder_contract,
    model_state_validator,
    model_state_digester=state_digester,
    live_collection_contract=None,
    collection_contract=None,
):
    if training_collection_contract is None:
        training_collection_contract = collection_contract
    elif collection_contract is not None:
        raise AssertionError("test helper received two training collections")
    live = (
        deployment_collection_contract(
            checkpoint.get("contract_sha256", "0" * 64),
            training_collection_contract,
        )
        if live_collection_contract is None
        else live_collection_contract
    )
    return validate_checkpoint_artifact(
        checkpoint,
        environment_contract=environment_contract,
        live_collection_contract=live,
        encoder_contract=encoder_contract,
        model_state_validator=model_state_validator,
        model_state_digester=model_state_digester,
    )


def redigest(checkpoint):
    checkpoint["contract_sha256"] = canonical_json_sha256(
        checkpoint["contract_manifest"]
    )


def test_checkpoint_has_one_exact_manifest_and_canonical_digest():
    checkpoint, env, collection = make_checkpoint()
    manifest = checkpoint["contract_manifest"]

    assert set(checkpoint) == {
        "checkpoint_schema",
        "state_dict",
        "contract_manifest",
        "contract_sha256",
    }
    assert checkpoint["checkpoint_schema"] == CHECKPOINT_SCHEMA_VERSION
    assert checkpoint["contract_sha256"] == canonical_json_sha256(manifest)
    assert set(manifest) == {
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
    }
    assert manifest["manifest_schema"] == MANIFEST_SCHEMA_VERSION
    assert manifest["environment_contract"] == env.to_record()
    assert manifest["environment_contract_sha256"] == env.sha256
    assert manifest["training_collection_contract"] == collection.to_record()
    assert (
        manifest["training_collection_contract_sha256"]
        == collection.sha256
    )


def test_manifest_fixed_contracts_come_from_live_runtime_constants():
    checkpoint, _, _ = make_checkpoint()
    manifest = checkpoint["contract_manifest"]

    assert manifest["observation_contract"] == {
        "schema": CONTROLLER_OBSERVATION_VERSION,
        "canonical_serializer": CANONICAL_JSON_VERSION,
        "outcome_schema": "transition-outcome-v2",
        "receipt_schema": "symbolic-check-receipt-v2",
    }
    assert manifest["transition_contract"] == {
        "run_header_schema": RUN_HEADER_VERSION,
        "record_schema": TRANSITION_RECORD_VERSION,
        "semantics": TRANSITION_SEMANTICS_VERSION,
        "closure_schema": CLOSURE_EVENTS_VERSION,
        "closure_priority": [verb.value for verb in CLOSURE_PRIORITY],
        "closure_turn_cost": CLOSURE_TURN_COST,
        "closure_bc_eligible": False,
        "previous_outcome_source": "learned_action",
    }
    assert manifest["action_contract"] == {
        "semantics": ACTION_SEMANTICS_VERSION,
        "algebra_verb_order": [verb.value for verb in VERB_ORDER],
        "policy_verb_order": [verb.value for verb in POLICY_VERB_ORDER],
        "target_kinds": {
            verb.value: list(ACTION_TARGET_KINDS[verb.value])
            for verb in VERB_ORDER
        },
    }
    assert manifest["feature_contract"] == feature_contract_record()


def test_policy_encoder_training_and_dataset_fields_are_all_explicit():
    checkpoint, _, _ = make_checkpoint()
    manifest = checkpoint["contract_manifest"]

    assert manifest["encoder_contract"] == ENCODER
    assert manifest["policy_contract"] == {
        "architecture": CONTROLLER_ARCHITECTURE_VERSION,
        "model_config": MODEL_CONFIG,
        "feature_dim": FEATURE_DIM,
        "state_dict_sha256": state_digester(checkpoint["state_dict"]),
        "decision_rule": DECISION_RULE,
        "acting_score": ACTING_SCORE_VERSION,
    }
    assert manifest["training_target_contract"] == TRAINING_TARGET
    assert manifest["dataset_contract"] == DATASET


def test_validation_returns_typed_bindings_and_exact_shape_context():
    checkpoint, env, collection = make_checkpoint()
    calls = []

    def validator(saved_state, shape):
        calls.append((saved_state, shape))
        shape_validator(saved_state, shape)

    result = validate_checkpoint(
        checkpoint,
        environment_contract=env,
        training_collection_contract=collection,
        encoder_contract=ENCODER,
        model_state_validator=validator,
    )

    assert result.environment_contract == env
    assert result.training_collection_contract == collection
    assert result.live_collection_contract.behavior_source == "learned_policy"
    assert (
        result.live_collection_contract.problem_source_sha256
        != collection.problem_source_sha256
    )
    assert result.contract_sha256 == checkpoint["contract_sha256"]
    assert result.state_dict_sha256 == state_digester(
        checkpoint["state_dict"]
    )
    assert calls == [(checkpoint["state_dict"], result.model_shape)]
    assert result.model_shape == ModelShapeContract(
        architecture=CONTROLLER_ARCHITECTURE_VERSION,
        feature_dim=FEATURE_DIM,
        policy_verb_count=len(POLICY_VERB_ORDER),
        hidden_dim=MODEL_CONFIG["hidden_dim"],
        repr_dim=MODEL_CONFIG["repr_dim"],
        node_repr_dim=MODEL_CONFIG["node_repr_dim"],
        verb_embed_dim=MODEL_CONFIG["verb_embed_dim"],
    )


@pytest.mark.parametrize("operation", ["missing", "unknown"])
def test_checkpoint_rejects_missing_and_unknown_top_level_fields(operation):
    checkpoint, env, collection = make_checkpoint()
    if operation == "missing":
        checkpoint.pop("contract_sha256")
    else:
        checkpoint["feature_dim"] = FEATURE_DIM

    with pytest.raises(CheckpointValidationError, match="keys must be exactly"):
        validate_checkpoint(
            checkpoint,
            environment_contract=env,
            collection_contract=collection,
            encoder_contract=ENCODER,
            model_state_validator=shape_validator,
        )


@pytest.mark.parametrize("schema", ["legacy", "phase-b-controller-checkpoint-v1"])
def test_checkpoint_never_auto_upgrades_legacy_schema(schema):
    checkpoint, env, collection = make_checkpoint()
    checkpoint["checkpoint_schema"] = schema

    with pytest.raises(CheckpointValidationError, match="differs from v2"):
        validate_checkpoint(
            checkpoint,
            environment_contract=env,
            collection_contract=collection,
            encoder_contract=ENCODER,
            model_state_validator=shape_validator,
        )


def test_manifest_tampering_is_rejected_before_nested_validation():
    checkpoint, env, collection = make_checkpoint()
    checkpoint["contract_manifest"]["policy_contract"]["decision_rule"] = (
        "sequential"
    )

    with pytest.raises(CheckpointValidationError, match="manifest digest mismatch"):
        validate_checkpoint(
            checkpoint,
            environment_contract=env,
            collection_contract=collection,
            encoder_contract=ENCODER,
            model_state_validator=shape_validator,
        )


@pytest.mark.parametrize("operation", ["missing", "unknown"])
def test_manifest_rejects_missing_and_unknown_fields_even_with_new_digest(
    operation,
):
    checkpoint, env, collection = make_checkpoint()
    manifest = checkpoint["contract_manifest"]
    if operation == "missing":
        manifest.pop("dataset_contract")
    else:
        manifest["legacy_metadata"] = {}
    redigest(checkpoint)

    with pytest.raises(CheckpointValidationError, match="keys must be exactly"):
        validate_checkpoint(
            checkpoint,
            environment_contract=env,
            collection_contract=collection,
            encoder_contract=ENCODER,
            model_state_validator=shape_validator,
        )


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("observation_contract", "tree-render-v1"),
        ("transition_contract", "reasoning-macro-transition-v3"),
        ("action_contract", "goal-commit-v1"),
    ],
)
def test_runtime_contract_changes_are_rejected_even_when_redigested(
    field, replacement
):
    checkpoint, env, collection = make_checkpoint()
    manifest = checkpoint["contract_manifest"]
    nested_field = {
        "observation_contract": "schema",
        "transition_contract": "semantics",
        "action_contract": "semantics",
    }[field]
    manifest[field][nested_field] = replacement
    redigest(checkpoint)

    with pytest.raises(CheckpointValidationError, match=f"{field} differs"):
        validate_checkpoint(
            checkpoint,
            environment_contract=env,
            collection_contract=collection,
            encoder_contract=ENCODER,
            model_state_validator=shape_validator,
        )


def test_dimension_preserving_feature_contract_change_is_rejected():
    checkpoint, env, collection = make_checkpoint()
    checkpoint["contract_manifest"]["feature_contract"]["node_serializer"] = (
        "changed-but-same-dimension"
    )
    redigest(checkpoint)

    with pytest.raises(CheckpointValidationError, match="feature contract"):
        validate_checkpoint(
            checkpoint,
            environment_contract=env,
            collection_contract=collection,
            encoder_contract=ENCODER,
            model_state_validator=shape_validator,
        )


def test_nested_environment_digest_is_independently_checked():
    checkpoint, env, collection = make_checkpoint()
    checkpoint["contract_manifest"]["environment_contract"][
        "generation_temperature"
    ] = 0.25
    redigest(checkpoint)

    with pytest.raises(CheckpointValidationError, match="environment contract digest"):
        validate_checkpoint(
            checkpoint,
            environment_contract=env,
            collection_contract=collection,
            encoder_contract=ENCODER,
            model_state_validator=shape_validator,
        )


def test_nested_collection_digest_is_independently_checked():
    checkpoint, env, collection = make_checkpoint()
    checkpoint["contract_manifest"]["training_collection_contract"][
        "problem_source_sha256"
    ] = "d" * 64
    redigest(checkpoint)

    with pytest.raises(CheckpointValidationError, match="collection contract digest"):
        validate_checkpoint(
            checkpoint,
            environment_contract=env,
            collection_contract=collection,
            encoder_contract=ENCODER,
            model_state_validator=shape_validator,
        )


def test_live_environment_mismatch_is_rejected():
    checkpoint, env, collection = make_checkpoint()
    shifted = replace(env, generation_temperature=0.25)

    with pytest.raises(CheckpointValidationError, match="live environment"):
        validate_checkpoint(
            checkpoint,
            environment_contract=shifted,
            collection_contract=collection,
            encoder_contract=ENCODER,
            model_state_validator=shape_validator,
        )


def test_live_collection_mismatch_is_rejected():
    checkpoint, env, collection = make_checkpoint()
    shifted = deployment_collection_contract(
        checkpoint["contract_sha256"], collection
    )
    shifted = replace(
        shifted,
        generator=ModelIdentity("test", "other-generator", "r1"),
    )

    with pytest.raises(CheckpointValidationError, match="generator differs"):
        validate_checkpoint(
            checkpoint,
            environment_contract=env,
            training_collection_contract=collection,
            live_collection_contract=shifted,
            encoder_contract=ENCODER,
            model_state_validator=shape_validator,
        )


@pytest.mark.parametrize(
    "changes,match",
    [
        (
            {"generator_implementation": "tests.OtherGenerator"},
            "generator_implementation differs",
        ),
        (
            {
                "generator_transport": ModelTransportContract.local(
                    "tests.OtherGeneratorTransport"
                )
            },
            "generator_transport differs",
        ),
        ({"generation_seed": 7}, "incompatible with the environment"),
        (
            {
                "controller_transport": ModelTransportContract.local(
                    "tests.OtherLearnedController"
                )
            },
            "controller_transport",
        ),
        (
            {
                "behavior_source": "teacher",
                "controller_fallback_policy": "first-legal-pair-v1",
            },
            "behavior_source",
        ),
    ],
)
def test_live_collection_rejects_generator_or_controller_contract_drift(
    changes, match
):
    checkpoint, env, training = make_checkpoint()
    live = deployment_collection_contract(
        checkpoint["contract_sha256"], training
    )
    live = replace(live, **changes)

    with pytest.raises(CheckpointValidationError, match=match):
        validate_checkpoint_artifact(
            checkpoint,
            environment_contract=env,
            live_collection_contract=live,
            encoder_contract=ENCODER,
            model_state_validator=shape_validator,
            model_state_digester=state_digester,
        )


@pytest.mark.parametrize(
    "field,value,match",
    [
        (
            "controller",
            ModelIdentity(
                CHECKPOINT_CONTROLLER_PROVIDER,
                CONTROLLER_ARCHITECTURE_VERSION,
                "f" * 64,
            ),
            "controller is not checkpoint-compatible",
        ),
        (
            "controller_implementation",
            "tests.OtherLearnedController",
            "controller_implementation",
        ),
        (
            "controller_output_contract",
            "other-controller-choice-v2",
            "controller_output_contract",
        ),
        (
            "policy_input_contract",
            "dimension-preserving-change-v2",
            "policy_input_contract",
        ),
    ],
)
def test_live_controller_contract_requires_exact_checkpoint_policy(
    field, value, match
):
    checkpoint, env, training = make_checkpoint()
    live = deployment_collection_contract(
        checkpoint["contract_sha256"], training
    )
    live = replace(live, **{field: value})

    with pytest.raises(CheckpointValidationError, match=match):
        validate_checkpoint_artifact(
            checkpoint,
            environment_contract=env,
            live_collection_contract=live,
            encoder_contract=ENCODER,
            model_state_validator=shape_validator,
            model_state_digester=state_digester,
        )


def test_teacher_collection_cannot_be_used_as_live_deployment():
    checkpoint, env, training = make_checkpoint()

    with pytest.raises(CheckpointValidationError, match="controller"):
        validate_checkpoint_artifact(
            checkpoint,
            environment_contract=env,
            live_collection_contract=training,
            encoder_contract=ENCODER,
            model_state_validator=shape_validator,
            model_state_digester=state_digester,
        )


def test_different_trusted_eval_problem_source_is_compatible():
    checkpoint, env, training = make_checkpoint()
    first = deployment_collection_contract(
        checkpoint["contract_sha256"], training, problem_digest="e" * 64
    )
    second = replace(
        first,
        problem_source_contract="other-eval-problems-v1",
        problem_source_sha256="f" * 64,
        problem_adapter_implementation="tests.OtherEvalProblemAdapter",
    )

    for live in (first, second):
        validate_checkpoint_artifact(
            checkpoint,
            environment_contract=env,
            live_collection_contract=live,
            encoder_contract=ENCODER,
            model_state_validator=shape_validator,
            model_state_digester=state_digester,
        )


def test_cross_contract_seed_incompatibility_is_rejected_at_build():
    env = environment_contract()
    collection = replace(training_collection_contract(), generation_seed=7)

    with pytest.raises(CheckpointValidationError, match="incompatible"):
        build_contract_manifest(
            environment_contract=env,
            training_collection_contract=collection,
            encoder_contract=ENCODER,
            model_config=MODEL_CONFIG,
            feature_dim=FEATURE_DIM,
            training_target_contract=TRAINING_TARGET,
            dataset_contract=DATASET,
        )


def test_live_encoder_mismatch_is_rejected():
    checkpoint, env, collection = make_checkpoint()
    changed_encoder = {**ENCODER, "model_digest": "e" * 64}

    with pytest.raises(CheckpointValidationError, match="live encoder"):
        validate_checkpoint(
            checkpoint,
            environment_contract=env,
            collection_contract=collection,
            encoder_contract=changed_encoder,
            model_state_validator=shape_validator,
        )


@pytest.mark.parametrize("field", ["implementation", "algorithm_contract"])
def test_live_encoder_implementation_and_algorithm_are_exact(field):
    checkpoint, env, collection = make_checkpoint()
    changed_encoder = {**ENCODER, field: f"changed-{field}-v2"}

    with pytest.raises(CheckpointValidationError, match="live encoder"):
        validate_checkpoint(
            checkpoint,
            environment_contract=env,
            collection_contract=collection,
            encoder_contract=changed_encoder,
            model_state_validator=shape_validator,
        )


def test_nonnull_encoder_model_digest_must_be_sha256():
    env = environment_contract()
    collection = training_collection_contract()
    with pytest.raises(CheckpointValidationError, match="model_digest"):
        build_contract_manifest(
            environment_contract=env,
            training_collection_contract=collection,
            encoder_contract={**ENCODER, "model_digest": "not-a-sha256"},
            model_config=MODEL_CONFIG,
            feature_dim=FEATURE_DIM,
            training_target_contract=TRAINING_TARGET,
            dataset_contract=DATASET,
        )


@pytest.mark.parametrize(
    "encoder,match",
    [
        (
            {**ENCODER, "model_digest": None},
            "requires model and model_digest",
        ),
        (
            {
                "type": "hash",
                "implementation": "tests.HashEncoder",
                "algorithm_contract": "signed-blake2b-features-v1",
                "model": "untracked-model",
                "model_digest": None,
                "dimension": 8,
            },
            "cannot carry model identity",
        ),
        (
            {
                "type": "implicit-default",
                "implementation": "tests.UnknownEncoder",
                "algorithm_contract": "unknown-v1",
                "model": None,
                "model_digest": None,
                "dimension": 8,
            },
            "unsupported",
        ),
    ],
)
def test_encoder_contract_rejects_incomplete_or_unknown_identity(encoder, match):
    env = environment_contract()
    collection = training_collection_contract()
    with pytest.raises(CheckpointValidationError, match=match):
        build_contract_manifest(
            environment_contract=env,
            training_collection_contract=collection,
            encoder_contract=encoder,
            model_config=MODEL_CONFIG,
            feature_dim=FEATURE_DIM,
            training_target_contract=TRAINING_TARGET,
            dataset_contract=DATASET,
        )


def test_feature_dimension_must_include_exact_structural_dimension():
    env = environment_contract()
    collection = training_collection_contract()
    with pytest.raises(CheckpointValidationError, match="feature_dim differs"):
        build_contract_manifest(
            environment_contract=env,
            training_collection_contract=collection,
            encoder_contract=ENCODER,
            model_config=MODEL_CONFIG,
            feature_dim=ENCODER["dimension"],
            training_target_contract=TRAINING_TARGET,
            dataset_contract=DATASET,
        )


@pytest.mark.parametrize(
    "section,operation",
    [
        ("model_config", "missing"),
        ("model_config", "unknown"),
        ("training_target_contract", "missing"),
        ("training_target_contract", "unknown"),
        ("dataset_contract", "missing"),
        ("dataset_contract", "unknown"),
    ],
)
def test_nested_contracts_reject_missing_and_unknown_fields(section, operation):
    checkpoint, env, collection = make_checkpoint()
    manifest = checkpoint["contract_manifest"]
    record = (
        manifest["policy_contract"]["model_config"]
        if section == "model_config"
        else manifest[section]
    )
    if operation == "missing":
        record.pop(next(iter(record)))
    else:
        record["default"] = 0
    redigest(checkpoint)

    with pytest.raises(CheckpointValidationError, match="keys must be exactly"):
        validate_checkpoint(
            checkpoint,
            environment_contract=env,
            collection_contract=collection,
            encoder_contract=ENCODER,
            model_state_validator=shape_validator,
        )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("target_type", "td-target-v1", "target type"),
        ("gamma", 2.0, "gamma"),
        ("reward_contract", "learned-reward-v1", "reward contract"),
    ],
)
def test_training_contract_rejects_semantic_changes(field, value, match):
    checkpoint, env, collection = make_checkpoint()
    checkpoint["contract_manifest"]["training_target_contract"][field] = value
    redigest(checkpoint)

    with pytest.raises(CheckpointValidationError, match=match):
        validate_checkpoint(
            checkpoint,
            environment_contract=env,
            collection_contract=collection,
            encoder_contract=ENCODER,
            model_state_validator=shape_validator,
        )


@pytest.mark.parametrize("field", ["dataset_sha256", "split_sha256"])
def test_dataset_contract_requires_exact_digests(field):
    checkpoint, env, collection = make_checkpoint()
    checkpoint["contract_manifest"]["dataset_contract"][field] = "not-a-digest"
    redigest(checkpoint)

    with pytest.raises(CheckpointValidationError, match=field):
        validate_checkpoint(
            checkpoint,
            environment_contract=env,
            collection_contract=collection,
            encoder_contract=ENCODER,
            model_state_validator=shape_validator,
        )


def test_model_state_shape_mismatch_is_rejected():
    checkpoint, env, collection = make_checkpoint()
    checkpoint["state_dict"] = state_dict(feature_dim=FEATURE_DIM + 1)

    with pytest.raises(CheckpointValidationError, match="global adapter shape"):
        validate_checkpoint(
            checkpoint,
            environment_contract=env,
            collection_contract=collection,
            encoder_contract=ENCODER,
            model_state_validator=shape_validator,
        )


def test_verb_embedding_dimension_is_shape_validated():
    checkpoint, env, collection = make_checkpoint()
    checkpoint["contract_manifest"]["policy_contract"]["model_config"][
        "verb_embed_dim"
    ] += 1
    redigest(checkpoint)

    with pytest.raises(CheckpointValidationError, match="verb embedding shape"):
        validate_checkpoint(
            checkpoint,
            environment_contract=env,
            collection_contract=collection,
            encoder_contract=ENCODER,
            model_state_validator=shape_validator,
        )


def test_same_shape_state_value_swap_is_rejected_by_manifest_digest():
    checkpoint, env, collection = make_checkpoint()
    original = checkpoint["state_dict"]["global.weight"]
    checkpoint["state_dict"] = {
        **checkpoint["state_dict"],
        "global.weight": FakeTensor(original.shape, (999.0,)),
    }

    with pytest.raises(CheckpointValidationError, match="state_dict digest"):
        validate_checkpoint(
            checkpoint,
            environment_contract=env,
            collection_contract=collection,
            encoder_contract=ENCODER,
            model_state_validator=shape_validator,
        )


def test_model_state_digester_is_required_and_must_be_deterministic():
    checkpoint, env, collection = make_checkpoint()
    live = deployment_collection_contract(
        checkpoint["contract_sha256"], collection
    )

    with pytest.raises(TypeError, match="digester must be callable"):
        validate_checkpoint_artifact(
            checkpoint,
            environment_contract=env,
            live_collection_contract=live,
            encoder_contract=ENCODER,
            model_state_validator=shape_validator,
            model_state_digester=None,
        )

    calls = 0

    def changing_digester(saved_state):
        nonlocal calls
        calls += 1
        return ("a" if calls % 2 else "b") * 64

    with pytest.raises(CheckpointValidationError, match="not deterministic"):
        validate_checkpoint_artifact(
            checkpoint,
            environment_contract=env,
            live_collection_contract=live,
            encoder_contract=ENCODER,
            model_state_validator=shape_validator,
            model_state_digester=changing_digester,
        )


@pytest.mark.parametrize("saved_state", [{}, {1: FakeTensor((1,))}])
def test_state_dictionary_must_be_nonempty_with_text_keys(saved_state):
    checkpoint, env, collection = make_checkpoint()
    checkpoint["state_dict"] = saved_state

    with pytest.raises(CheckpointValidationError, match="state_dict"):
        validate_checkpoint(
            checkpoint,
            environment_contract=env,
            collection_contract=collection,
            encoder_contract=ENCODER,
            model_state_validator=shape_validator,
        )


def test_model_state_validator_is_required():
    checkpoint, env, collection = make_checkpoint()
    with pytest.raises(TypeError, match="must be callable"):
        validate_checkpoint(
            checkpoint,
            environment_contract=env,
            collection_contract=collection,
            encoder_contract=ENCODER,
            model_state_validator=None,
        )


def test_manifest_inputs_are_snapshotted_and_key_order_does_not_change_digest():
    env = environment_contract()
    collection = training_collection_contract()
    encoder = deepcopy(ENCODER)
    manifest = build_contract_manifest(
        environment_contract=env,
        training_collection_contract=collection,
        encoder_contract=encoder,
        model_config=MODEL_CONFIG,
        feature_dim=FEATURE_DIM,
        training_target_contract=TRAINING_TARGET,
        dataset_contract=DATASET,
    )
    original_digest = canonical_json_sha256(manifest)
    encoder["model_digest"] = "mutated-after-build"
    reversed_manifest = {
        key: manifest[key] for key in reversed(tuple(manifest))
    }

    assert manifest["encoder_contract"] == ENCODER
    assert canonical_json_sha256(reversed_manifest) == original_digest


def test_noncanonical_numeric_manifest_value_is_rejected():
    checkpoint, env, collection = make_checkpoint()
    checkpoint["contract_manifest"]["training_target_contract"]["gamma"] = -0.0

    with pytest.raises(CheckpointValidationError, match="invalid JSON number"):
        validate_checkpoint(
            checkpoint,
            environment_contract=env,
            collection_contract=collection,
            encoder_contract=ENCODER,
            model_state_validator=shape_validator,
        )
