from __future__ import annotations

from dataclasses import replace

import pytest

import verifier_anchored_rl.reasoning_v2.learned_controller as learned
from verifier_anchored_rl.reasoning_v2.actions import ResultDraft
from verifier_anchored_rl.reasoning_v2.checkpoint import (
    CHECKPOINT_CONTROLLER_PROVIDER,
    CONTROLLER_ARCHITECTURE_VERSION,
    DATASET_SCHEMA_VERSION,
    LEARNED_CONTROLLER_IMPLEMENTATION,
    LEARNED_CONTROLLER_MAX_TOKENS,
    LEARNED_CONTROLLER_OUTPUT_CONTRACT,
    TRAINING_TARGET_VERSION,
    CheckpointValidationError,
    ModelShapeContract,
    build_checkpoint,
)
from verifier_anchored_rl.reasoning_v2.contracts import (
    CollectionContract,
    ModelIdentity,
    ModelTransportContract,
    build_environment_contract,
)
from verifier_anchored_rl.reasoning_v2.features import (
    FEATURE_CONTRACT_VERSION,
    STRUCTURAL_FEATURE_DIM,
    BoundFeatureExtractor,
)
from verifier_anchored_rl.reasoning_v2.lenses import (
    GENERATION_SCHEMA_VERSION,
    LENS_CONTRACT_VERSION,
)
from verifier_anchored_rl.reasoning_v2.math_adapter import (
    CaseworkMathVerifier,
)
from verifier_anchored_rl.reasoning_v2.model import Action, Verb
from verifier_anchored_rl.reasoning_v2.model import (
    GoalObjective,
    GoalObligation,
    State,
    new_graph,
)
from verifier_anchored_rl.reasoning_v2.provenance import BehaviorSource
from verifier_anchored_rl.reasoning_v2.records import (
    build_run_header,
    build_transition_record,
    canonical_record_bytes,
)
from verifier_anchored_rl.reasoning_v2.replay import (
    replay_canonical_run_lines,
)
from verifier_anchored_rl.reasoning_v2.runtime import (
    BoundEnvironment,
    EpisodeRuntime,
)
from verifier_anchored_rl.reasoning_v2.transition import (
    REWARD_CONTRACT_VERSION,
)


requires_torch = pytest.mark.skipif(
    not learned._TORCH_AVAILABLE,
    reason="PyTorch is not installed",
)

ENCODER_DIM = 8
MODEL_CONFIG = {
    "hidden_dim": 12,
    "repr_dim": 7,
    "node_repr_dim": 6,
    "verb_embed_dim": 5,
}
FEATURE_DIM = ENCODER_DIM + STRUCTURAL_FEATURE_DIM


def environment() -> BoundEnvironment:
    adapter = CaseworkMathVerifier()
    contract = build_environment_contract(
        generation_contract=adapter,
        verifier=adapter,
        aggregate_builder=adapter,
        lens_contract=LENS_CONTRACT_VERSION,
        generation_schema=GENERATION_SCHEMA_VERSION,
        generation_temperature=0.0,
        generation_max_tokens=64,
        generation_seed_policy="none-v1",
        turn_limit=8,
    )
    return BoundEnvironment(contract, adapter, adapter, adapter)


def training_collection() -> CollectionContract:
    generator_implementation = "tests.StaticGenerator"
    teacher_implementation = "tests.StaticTeacher"
    return CollectionContract(
        generator=ModelIdentity("test", "generator", "r1"),
        controller=ModelIdentity("test", "teacher", "r1"),
        controller_implementation=teacher_implementation,
        controller_transport=ModelTransportContract.local(
            teacher_implementation
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
        problem_source_contract="learned-controller-train-problems-v1",
        problem_source_sha256="a" * 64,
        problem_adapter_implementation="tests.TrainingProblemAdapter",
    )


def deployment_collection(
    checkpoint_sha256: str, training: CollectionContract
) -> CollectionContract:
    return CollectionContract(
        generator=training.generator,
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
        generator_implementation=training.generator_implementation,
        generator_transport=training.generator_transport,
        generation_seed=training.generation_seed,
        policy_input_contract=FEATURE_CONTRACT_VERSION,
        problem_source_contract="learned-controller-eval-problems-v1",
        problem_source_sha256="b" * 64,
        problem_adapter_implementation="tests.EvalProblemAdapter",
        controller_temperature=0.0,
        controller_max_tokens=LEARNED_CONTROLLER_MAX_TOKENS,
        controller_seed_policy="none-v1",
        controller_output_contract=LEARNED_CONTROLLER_OUTPUT_CONTRACT,
    )


def _model():
    learned.torch.manual_seed(7)
    return learned.FactoredControllerNetwork(
        FEATURE_DIM,
        **MODEL_CONFIG,
    )


def checkpoint_fixture(encoder=None, model=None):
    env = environment()
    if encoder is None:
        encoder = learned.DeterministicHashEncoder(ENCODER_DIM)
    training = training_collection()
    if model is None:
        model = _model()
    checkpoint = build_checkpoint(
        state_dict=model.state_dict(),
        environment_contract=env.contract,
        training_collection_contract=training,
        encoder_contract=encoder.contract_record(),
        model_config=MODEL_CONFIG,
        feature_dim=FEATURE_DIM,
        training_target_contract={
            "target_type": TRAINING_TARGET_VERSION,
            "gamma": 1.0,
            "reward_contract": REWARD_CONTRACT_VERSION,
        },
        dataset_contract={
            "schema": DATASET_SCHEMA_VERSION,
            "dataset_sha256": "c" * 64,
            "split_sha256": "d" * 64,
        },
        model_state_validator=learned.validate_torch_state_dict,
        model_state_digester=learned.torch_state_dict_sha256,
    )
    live = deployment_collection(checkpoint["contract_sha256"], training)
    return checkpoint, env, encoder, live


class MutableFrozenEncoder:
    dimension = ENCODER_DIM

    def __init__(self):
        self.actual_digest = "e" * 64

    def contract_record(self):
        return {
            "type": "frozen_embedding",
            "implementation": (
                f"{type(self).__module__}.{type(self).__qualname__}"
            ),
            "algorithm_contract": "test-frozen-embedding-v1",
            "model": "test/model",
            "model_digest": "e" * 64,
            "dimension": self.dimension,
        }

    def state_sha256(self):
        return self.actual_digest

    def encode_many(self, texts):
        return tuple((0.0,) * self.dimension for _ in texts)


class StaticResultGenerator:
    generation_retry_budget = 0

    def __init__(self, env, collection):
        self.environment_contract_sha256 = env.sha256
        self.generator_identity = collection.generator
        self.generator_implementation = collection.generator_implementation
        self.generator_transport_contract = collection.generator_transport
        self.generator_transport_sha256 = (
            collection.generator_transport_sha256
        )
        self.generation_seed = collection.generation_seed
        self.lens_contract = env.contract.lens_contract
        self.generation_schema = env.contract.generation_schema
        self.generation_temperature = env.contract.generation_temperature
        self.generation_max_tokens = env.contract.generation_max_tokens
        self.generation_seed_policy = env.contract.generation_seed_policy

    def generate(self, state, action):
        assert action == Action(Verb.PROPOSE_RESULT, 0)
        return ResultDraft(GoalObjective.SUM, "2")


def _goal(
    node_id: int,
    *,
    parent: int | None,
    objective: str,
    statement: str,
    equation: str,
    domain: str = "Reals",
    connective: str | None = None,
) -> dict[str, object]:
    return {
        "id": node_id,
        "parent_goal": parent,
        "statement": statement,
        "obligation": {
            "variable": "x",
            "equation": equation,
            "domain": domain,
            "objective": objective,
        },
        "assumptions": [],
        "connective": connective,
        "status": "open",
        "committed_claim": None,
    }


def observation(env: BoundEnvironment) -> dict[str, object]:
    return {
        "schema_version": "controller-observation-v2",
        "environment_contract_sha256": env.sha256,
        "root_goal_id": 0,
        "turn": 2,
        "turn_limit": env.contract.turn_limit,
        "previous_outcome": None,
        "goals": [
            _goal(
                0,
                parent=None,
                objective="sum",
                statement="solve the two exhaustive branches",
                equation="Eq(x, 2)",
                connective="exhaustive_case",
            ),
            _goal(
                1,
                parent=0,
                objective="branch",
                statement="same local branch",
                equation="Eq(x, 2)",
                domain="x < 0",
            ),
            _goal(
                2,
                parent=0,
                objective="branch",
                statement="same local branch",
                equation="Eq(x, 2)",
                domain="x >= 0",
            ),
        ],
        "evidence": [
            {
                "id": 3,
                "scope_goal": 1,
                "statement": "left owner evidence",
                "premises": [],
                "source": "trusted_given",
                "committable": False,
            },
            {
                "id": 4,
                "scope_goal": 2,
                "statement": "right owner evidence",
                "premises": [],
                "source": "trusted_given",
                "committable": False,
            },
        ],
        "claims": [],
        "legal_pairs": [
            {"verb": "propose_result", "target": 0},
            {"verb": "propose_solution", "target": 1},
            {"verb": "propose_solution", "target": 2},
            {"verb": "propose_refutation", "target": 1},
            {"verb": "propose_refutation", "target": 2},
        ],
    }


@requires_torch
def test_hash_and_zero_encoder_records_match_checkpoint_shape():
    hashed = learned.DeterministicHashEncoder(11)
    zero = learned.ZeroTextEncoder(11)

    assert learned.encoder_contract_record(hashed) == {
        "type": "hash",
        "implementation": (
            "verifier_anchored_rl.reasoning_v2.learned_controller."
            "DeterministicHashEncoder"
        ),
        "algorithm_contract": learned.HASH_ENCODER_ALGORITHM,
        "model": None,
        "model_digest": None,
        "dimension": 11,
    }
    assert learned.encoder_contract_record(zero)["type"] == "zero_text"
    assert zero.encode_many(("first", "second")) == (
        (0.0,) * 11,
        (0.0,) * 11,
    )
    assert hashed.encode_many(("same",)) == hashed.encode_many(("same",))
    assert hashed.encode_many(("left",)) != hashed.encode_many(("right",))


@requires_torch
def test_encoded_features_align_mask_and_keep_owner_columns_distinct():
    env = environment()
    features = BoundFeatureExtractor(env).build(observation(env))
    hashed = learned.encode_structured_features(
        features, learned.DeterministicHashEncoder(ENCODER_DIM)
    )
    zero = learned.encode_structured_features(
        features, learned.ZeroTextEncoder(ENCODER_DIM)
    )

    assert tuple(hashed.global_features.shape) == (1, FEATURE_DIM)
    assert tuple(hashed.node_features.shape) == (1, 3, FEATURE_DIM)
    assert tuple(hashed.legal_pair_mask.shape) == (1, 4, 3)
    assert hashed.legal_pair_mask[0].tolist() == [
        list(row) for row in features.legal_pair_mask
    ]
    assert features.target_ids == (0, 1, 2)
    assert not learned.torch.equal(
        hashed.node_features[0, 1, :ENCODER_DIM],
        hashed.node_features[0, 2, :ENCODER_DIM],
    )
    assert learned.torch.equal(
        zero.node_features[0, 1], zero.node_features[0, 2]
    )
    assert learned.torch.equal(
        hashed.global_features[0, -STRUCTURAL_FEATURE_DIM:],
        hashed.node_features[0, 1, -STRUCTURAL_FEATURE_DIM:],
    )


@requires_torch
def test_global_joint_argmax_differs_from_sequential_greedy():
    env = environment()
    features = BoundFeatureExtractor(env).build(observation(env))
    verb_probability = learned.torch.tensor(
        [[0.0, 0.45, 0.55, 0.0]], dtype=learned.torch.float32
    )
    pointer_probability = learned.torch.zeros(
        1, 4, 3, dtype=learned.torch.float32
    )
    pointer_probability[0, 1, 0] = 1.0
    pointer_probability[0, 2, 1] = 0.51
    pointer_probability[0, 2, 2] = 0.49
    joint = verb_probability.unsqueeze(-1) * pointer_probability

    sequential_verb = int(verb_probability[0].argmax().item())
    sequential_target = int(
        pointer_probability[0, sequential_verb].argmax().item()
    )
    action = learned.select_joint_legal_action(features, joint)

    assert (sequential_verb, sequential_target) == (2, 1)
    assert action == Action(Verb.PROPOSE_RESULT, 0)


@requires_torch
def test_joint_selection_rejects_probability_outside_feature_mask():
    env = environment()
    features = BoundFeatureExtractor(env).build(observation(env))
    joint = learned.torch.zeros(1, 4, 3)
    joint[0, 2, 2] = 1.0

    assert learned.select_joint_legal_action(features, joint) == Action(
        Verb.PROPOSE_SOLUTION, 2
    )
    joint[0, 0, 0] = 0.1
    with pytest.raises(
        learned.ControllerIntegrationError,
        match="outside the legal mask",
    ):
        learned.select_joint_legal_action(features, joint)


@requires_torch
def test_network_has_exact_four_verb_pointer_q_and_embedding_shapes():
    model = _model()
    shape = ModelShapeContract(
        architecture=CONTROLLER_ARCHITECTURE_VERSION,
        feature_dim=FEATURE_DIM,
        policy_verb_count=4,
        **MODEL_CONFIG,
    )

    learned.validate_torch_state_dict(model.state_dict(), shape)
    assert tuple(model.verb_embedding.weight.shape) == (
        4,
        MODEL_CONFIG["verb_embed_dim"],
    )
    global_features = learned.torch.zeros(1, FEATURE_DIM)
    node_features = learned.torch.zeros(1, 3, FEATURE_DIM)
    pointer, q_values = model.pair_scores(global_features, node_features)
    assert tuple(pointer.shape) == (1, 4, 3)
    assert tuple(q_values.shape) == (1, 4, 3)
    unexpected = dict(model.state_dict())
    unexpected["extra.weight"] = learned.torch.zeros(1)
    with pytest.raises(ValueError, match="keys must be exactly"):
        learned.validate_torch_state_dict(unexpected, shape)


@requires_torch
def test_checkpoint_load_binds_exact_learned_controller_contract():
    checkpoint, env, encoder, live = checkpoint_fixture()
    controller = learned.FactoredLearnedController(
        checkpoint,
        environment=env,
        live_collection_contract=live,
        encoder=encoder,
    )

    choice = controller.choose(observation(env))

    assert choice.action in {
        Action(Verb.PROPOSE_RESULT, 0),
        Action(Verb.PROPOSE_SOLUTION, 1),
        Action(Verb.PROPOSE_SOLUTION, 2),
        Action(Verb.PROPOSE_REFUTATION, 1),
        Action(Verb.PROPOSE_REFUTATION, 2),
    }
    assert controller.controller_identity == ModelIdentity(
        CHECKPOINT_CONTROLLER_PROVIDER,
        CONTROLLER_ARCHITECTURE_VERSION,
        checkpoint["contract_sha256"],
    )
    assert controller.controller_implementation == (
        LEARNED_CONTROLLER_IMPLEMENTATION
    )
    assert controller.controller_transport_contract == (
        ModelTransportContract.local(LEARNED_CONTROLLER_IMPLEMENTATION)
    )
    assert controller.behavior_source is BehaviorSource.LEARNED_POLICY
    assert controller.controller_fallback_policy == "forbidden-v1"
    assert controller.controller_seed is None
    assert controller.policy_input_contract == FEATURE_CONTRACT_VERSION


@requires_torch
def test_checkpoint_controller_runs_through_runtime_record_and_replay():
    model = _model()
    with learned.torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.verb_head.bias[1] = 10.0
    checkpoint, env, encoder, live = checkpoint_fixture(model=model)
    controller = learned.FactoredLearnedController(
        checkpoint,
        environment=env,
        live_collection_contract=live,
        encoder=encoder,
    )
    graph = new_graph(
        "solve",
        GoalObligation("x", "Eq(x, 2)", "Reals", GoalObjective.SUM),
        generation_contract=env.generation_contract,
    )
    initial = State(graph, turn=0)
    episode = EpisodeRuntime(
        environment=env,
        collection_contract=live,
        generator=StaticResultGenerator(env, live),
        state=initial,
        controller=controller,
    )

    _, transition = episode.step()

    assert transition.terminal
    assert transition.reward == 1
    assert transition.decision_provenance.source is (
        BehaviorSource.LEARNED_POLICY
    )
    assert not transition.decision_provenance.bc_eligible
    record = build_transition_record(
        transition,
        environment=env,
        collection_contract=live,
        episode_key="episode:learned",
        problem_id="problem:learned",
    )
    header = build_run_header(env, live)

    class Resolver:
        problem_source_contract = live.problem_source_contract
        problem_source_sha256 = live.problem_source_sha256
        problem_adapter_implementation = (
            live.problem_adapter_implementation
        )

        def __call__(self, problem_id):
            if problem_id != "problem:learned":
                raise KeyError(problem_id)
            return initial

    replayed = replay_canonical_run_lines(
        (
            canonical_record_bytes(
                header,
                environment=env,
                collection_contract=live,
            ) + b"\n",
            canonical_record_bytes(
                record,
                environment=env,
                collection_contract=live,
            ) + b"\n",
        ),
        environment=env,
        initial_state_resolver=Resolver(),
        collection_contract=live,
    )
    assert replayed.episodes[0].decisions[0].transition == transition


@requires_torch
@pytest.mark.parametrize(
    "change,match",
    [
        (
            "collection",
            "policy_input_contract is not checkpoint-compatible",
        ),
        ("encoder", "live encoder differs from checkpoint"),
    ],
)
def test_checkpoint_load_rejects_live_contract_mismatch(change, match):
    checkpoint, env, encoder, live = checkpoint_fixture()
    if change == "collection":
        live = replace(live, policy_input_contract="other-features-v1")
    else:
        encoder = learned.ZeroTextEncoder(ENCODER_DIM)

    with pytest.raises(CheckpointValidationError, match=match):
        learned.FactoredLearnedController(
            checkpoint,
            environment=env,
            live_collection_contract=live,
            encoder=encoder,
        )


@requires_torch
def test_same_shape_state_tamper_is_rejected_before_load():
    checkpoint, env, encoder, live = checkpoint_fixture()
    tampered_state = {
        key: value.detach().clone()
        for key, value in checkpoint["state_dict"].items()
    }
    tensor = tampered_state["global_adapter.1.weight"]
    tensor.reshape(-1)[0] += 0.25
    tampered = {**checkpoint, "state_dict": tampered_state}

    assert tuple(tensor.shape) == tuple(
        checkpoint["state_dict"]["global_adapter.1.weight"].shape
    )
    assert learned.torch_state_dict_sha256(tampered_state) != (
        learned.torch_state_dict_sha256(checkpoint["state_dict"])
    )
    with pytest.raises(
        CheckpointValidationError, match="state_dict digest mismatch"
    ):
        learned.FactoredLearnedController(
            tampered,
            environment=env,
            live_collection_contract=live,
            encoder=encoder,
        )


@requires_torch
def test_loaded_controller_rejects_in_place_model_weight_drift():
    checkpoint, env, encoder, live = checkpoint_fixture()
    controller = learned.FactoredLearnedController(
        checkpoint,
        environment=env,
        live_collection_contract=live,
        encoder=encoder,
    )
    with learned.torch.no_grad():
        controller._model.verb_head.bias[0].add_(0.25)

    with pytest.raises(ValueError, match="model state differs"):
        controller.choose(observation(env))


@requires_torch
def test_loaded_controller_rejects_frozen_encoder_state_drift():
    encoder = MutableFrozenEncoder()
    checkpoint, env, _, live = checkpoint_fixture(encoder)
    controller = learned.FactoredLearnedController(
        checkpoint,
        environment=env,
        live_collection_contract=live,
        encoder=encoder,
    )
    encoder.actual_digest = "f" * 64

    with pytest.raises(
        learned.EncoderValidationError,
        match="live state differs",
    ):
        controller.choose(observation(env))


def test_torch_absence_fails_at_the_optional_controller_boundary(monkeypatch):
    monkeypatch.setattr(learned, "_TORCH_AVAILABLE", False)

    with pytest.raises(ImportError, match="PyTorch is required"):
        learned.FactoredControllerNetwork(
            FEATURE_DIM,
            **MODEL_CONFIG,
        )
