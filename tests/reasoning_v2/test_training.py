from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import pytest

import verifier_anchored_rl.reasoning_v2.learned_controller as learned
from verifier_anchored_rl.llm_client import (
    OpenAICompatibleAttempt,
    build_openai_compatible_json_body,
)
from verifier_anchored_rl.reasoning_v2.actions import (
    DecompositionDraft,
    ResultDraft,
    SolutionDraft,
)
from verifier_anchored_rl.reasoning_v2.checkpoint import (
    CHECKPOINT_CONTROLLER_PROVIDER,
    CONTROLLER_ARCHITECTURE_VERSION,
    LEARNED_CONTROLLER_IMPLEMENTATION,
    LEARNED_CONTROLLER_MAX_TOKENS,
    LEARNED_CONTROLLER_OUTPUT_CONTRACT,
)
from verifier_anchored_rl.reasoning_v2.contracts import (
    CollectionContract,
    ModelIdentity,
    ModelTransportContract,
    canonical_json_bytes,
    build_environment_contract,
)
from verifier_anchored_rl.reasoning_v2.controller import (
    POLICY_INPUT_CONTRACT_VERSION,
    TypedLLMController,
    build_controller_request,
)
from verifier_anchored_rl.reasoning_v2.dataset import (
    build_grouped_split,
    convert_replayed_run,
)
from verifier_anchored_rl.reasoning_v2.diagnostics import (
    build_attempt_record,
    build_call_record,
    build_decision_receipt,
    build_diagnostic_header,
    read_diagnostic_sidecar,
)
from verifier_anchored_rl.reasoning_v2.features import (
    FEATURE_CONTRACT_VERSION,
    STRUCTURAL_FEATURE_DIM,
)
from verifier_anchored_rl.reasoning_v2.lenses import (
    GENERATION_SCHEMA_VERSION,
    LENS_CONTRACT_VERSION,
    TypedPolicyDraftGenerator,
    build_policy_lens_request,
)
from verifier_anchored_rl.reasoning_v2.math_adapter import CaseworkMathVerifier
from verifier_anchored_rl.reasoning_v2.model import (
    Action,
    Connective,
    GoalObjective,
    GoalObligation,
    State,
    Verb,
    new_graph,
)
from verifier_anchored_rl.reasoning_v2.observation import observation_sha256
from verifier_anchored_rl.reasoning_v2.provenance import (
    BehaviorSource,
    DecisionProvenance,
)
from verifier_anchored_rl.reasoning_v2.records import (
    BehaviorMetadata,
    DiagnosticRef,
    build_run_header,
    build_transition_record,
)
from verifier_anchored_rl.reasoning_v2.replay import replay_canonical_run_lines
from verifier_anchored_rl.reasoning_v2.runtime import BoundEnvironment
from verifier_anchored_rl.reasoning_v2.training import (
    TrainingValidationError,
    WarmStartConfig,
    build_phase_b_warm_start_checkpoint,
    collate_phase_b_training_batch,
    compute_warm_start_loss,
    train_phase_b_warm_start,
)


requires_torch = pytest.mark.skipif(
    not learned._TORCH_AVAILABLE,
    reason="PyTorch is not installed",
)

ENCODER_DIM = 6
MODEL_CONFIG = {
    "hidden_dim": 10,
    "repr_dim": 7,
    "node_repr_dim": 6,
    "verb_embed_dim": 5,
}
FEATURE_DIM = ENCODER_DIM + STRUCTURAL_FEATURE_DIM
_OPENAI_CLIENT = (
    "verifier_anchored_rl.llm_client.OpenAICompatibleLLMClient"
)
_GENERATOR_IMPLEMENTATION = (
    f"{TypedPolicyDraftGenerator.__module__}."
    f"{TypedPolicyDraftGenerator.__qualname__}"
)
_CONTROLLER_IMPLEMENTATION = (
    f"{TypedLLMController.__module__}.{TypedLLMController.__qualname__}"
)


class FixedGenerator:
    def __init__(self, draft):
        self.draft = draft

    def generate(self, state, action):
        return self.draft


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
        turn_limit=2,
    )
    return BoundEnvironment(contract, adapter, adapter, adapter)


def hosted_transport(model: str, base_url: str) -> ModelTransportContract:
    return ModelTransportContract(
        implementation=_OPENAI_CLIENT,
        base_url=base_url,
        model=model,
        timeout_s=30.0,
        parse_retry_budget=0,
        backoff_retry_budget=0,
        guided_json=False,
        response_format_contract="openai-json-object-v1",
        parse_retry_contract="assistant-echo-seed-offset-v1",
        backoff_contract="retry-after-exponential-jitter-v1",
    )


def training_collection() -> CollectionContract:
    return CollectionContract(
        generator=ModelIdentity("test", "generator", "r1"),
        controller=ModelIdentity("test", "teacher", "r1"),
        controller_implementation=_CONTROLLER_IMPLEMENTATION,
        controller_transport=hosted_transport(
            "teacher", "https://controller.example/v1"
        ),
        controller_seed=None,
        controller_temperature=0.0,
        controller_max_tokens=64,
        controller_seed_policy="none-v1",
        behavior_source="teacher",
        controller_fallback_policy="first-legal-pair-v1",
        generator_implementation=_GENERATOR_IMPLEMENTATION,
        generator_transport=hosted_transport(
            "generator", "https://generator.example/v1"
        ),
        generation_seed=None,
        policy_input_contract=POLICY_INPUT_CONTRACT_VERSION,
        problem_source_contract="training-test-problems-v1",
        problem_source_sha256="2" * 64,
        problem_adapter_implementation="tests.TrainingProblemAdapter",
    )


def initial_state(env: BoundEnvironment, problem_id: str) -> State:
    if problem_id == "casework":
        equation = "Eq((x - 2)*(x - 3), 0)"
        statement = "sum both roots"
    else:
        value = "2" if problem_id == "direct-two" else "3"
        equation = f"Eq(x, {value})"
        statement = f"solve {problem_id}"
    return State(
        new_graph(
            statement,
            GoalObligation(
                "x", equation, "Reals", GoalObjective.SUM
            ),
            generation_contract=env.generation_contract,
        ),
        turn=0,
    )


@dataclass(frozen=True)
class PendingDecision:
    episode_key: str
    problem_id: str
    transition: object
    behavior: BehaviorMetadata
    generator_payload: dict[str, object]


def pending_decisions(env: BoundEnvironment) -> tuple[PendingDecision, ...]:
    state = initial_state(env, "casework")
    decomposition = DecompositionDraft(
        Connective.EXHAUSTIVE_CASE,
        ("case 1", "case 2"),
        (
            GoalObligation(
                "x",
                "Eq((x - 2)*(x - 3), 0)",
                "x < 5/2",
                GoalObjective.BRANCH,
            ),
            GoalObligation(
                "x",
                "Eq((x - 2)*(x - 3), 0)",
                "x >= 5/2",
                GoalObjective.BRANCH,
            ),
        ),
    )
    first = env.step(
        state,
        Action(Verb.DECOMPOSE, 0),
        FixedGenerator(decomposition),
    )
    second = env.step(
        first.next_state,
        Action(Verb.PROPOSE_SOLUTION, 1),
        FixedGenerator(SolutionDraft("x", "2")),
    )
    result = [
        PendingDecision(
            "episode-casework",
            "casework",
            first,
            BehaviorMetadata(BehaviorSource.TEACHER, fallback=True),
            {"domains": ["x < 5/2", "x >= 5/2"]},
        ),
        PendingDecision(
            "episode-casework",
            "casework",
            second,
            BehaviorMetadata(BehaviorSource.TEACHER, fallback=False),
            {"value": "2"},
        ),
    ]
    for problem_id, value in (("direct-two", "2"), ("direct-three", "3")):
        transition = env.step(
            initial_state(env, problem_id),
            Action(Verb.PROPOSE_RESULT, 0),
            FixedGenerator(ResultDraft(GoalObjective.SUM, value)),
        )
        result.append(PendingDecision(
            f"episode-{problem_id}",
            problem_id,
            transition,
            BehaviorMetadata(BehaviorSource.TEACHER, fallback=False),
            {"value": value},
        ))
    return tuple(result)


def accepted_call_records(
    *,
    role: str,
    call_id: str,
    request_body: dict[str, object],
    payload: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    raw_content = canonical_json_bytes(payload).decode("ascii")
    raw_response = canonical_json_bytes({
        "choices": [{"message": {"content": raw_content}}]
    }).decode("ascii")
    attempt = build_attempt_record(
        OpenAICompatibleAttempt(
            call_id=call_id,
            attempt_index=0,
            parse_retry_index=0,
            backoff_retry_index=0,
            effective_seed=None,
            request_body_json=canonical_json_bytes(request_body).decode(
                "ascii"
            ),
            raw_response_body=raw_response,
            raw_content=raw_content,
            raw_usage_json=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            outcome="accepted",
            disposition="return",
            http_status=200,
            error_type=None,
            error_message=None,
        ),
        role=role,
    )
    return attempt, build_call_record(role, call_id, [attempt])


def failed_controller_call_records(
    *, call_id: str, request_body: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    raw_content = '{"value":"not an action"}'
    raw_response = canonical_json_bytes({
        "choices": [{"message": {"content": raw_content}}]
    }).decode("ascii")
    attempt = build_attempt_record(
        OpenAICompatibleAttempt(
            call_id=call_id,
            attempt_index=0,
            parse_retry_index=0,
            backoff_retry_index=0,
            effective_seed=None,
            request_body_json=canonical_json_bytes(request_body).decode(
                "ascii"
            ),
            raw_response_body=raw_response,
            raw_content=raw_content,
            raw_usage_json=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            outcome="schema_error",
            disposition="raise",
            http_status=200,
            error_type="LLMResponseError",
            error_message="missing required action fields",
        ),
        role="controller",
    )
    return attempt, build_call_record("controller", call_id, [attempt])


@dataclass(frozen=True)
class Resolver:
    states: dict[str, State]
    problem_source_contract: str = "training-test-problems-v1"
    problem_source_sha256: str = "2" * 64
    problem_adapter_implementation: str = "tests.TrainingProblemAdapter"

    def __call__(self, problem_id: str) -> State:
        return self.states[problem_id]


def trusted_dataset(tmp_path):
    env = environment()
    collection = training_collection()
    diagnostics_records = [build_diagnostic_header(
        environment_contract_sha256=env.sha256,
        collection_contract_sha256=collection.sha256,
        generator_transport_sha256=collection.generator_transport_sha256,
        controller_transport_sha256=collection.controller_transport_sha256,
        controller_decision_receipts=True,
        controller_attempts=True,
    )]
    run_records = []
    next_id = 1

    def uid():
        nonlocal next_id
        value = f"{next_id:032x}"
        next_id += 1
        return value

    for pending in pending_decisions(env):
        transition = pending.transition
        observation = env.observe(transition.state)
        action = transition.action
        generator_request = build_policy_lens_request(observation, action)
        generator_body = build_openai_compatible_json_body(
            model=collection.generator_transport.model,
            system=generator_request.system,
            user=generator_request.user,
            schema=generator_request.schema,
            temperature=env.contract.generation_temperature,
            max_tokens=env.contract.generation_max_tokens,
            seed=None,
            guided_json=False,
        )
        generator_attempt, generator_call = accepted_call_records(
            role="generator",
            call_id=uid(),
            request_body=generator_body,
            payload=pending.generator_payload,
        )
        controller_request = build_controller_request(observation, env)
        controller_body = build_openai_compatible_json_body(
            model=collection.controller_transport.model,
            system=controller_request.system,
            user=controller_request.user,
            schema=controller_request.schema,
            temperature=collection.controller_temperature,
            max_tokens=collection.controller_max_tokens,
            seed=None,
            guided_json=False,
        )
        if pending.behavior.fallback:
            controller_attempt, controller_call = (
                failed_controller_call_records(
                    call_id=uid(), request_body=controller_body
                )
            )
        else:
            controller_attempt, controller_call = accepted_call_records(
                role="controller",
                call_id=uid(),
                request_body=controller_body,
                payload={"verb": action.verb.value, "target": action.target},
            )
        diagnostics_records.extend((controller_attempt, controller_call))
        diagnostics_records.extend((generator_attempt, generator_call))
        receipt_id = uid()
        observation_digest = observation_sha256(
            observation,
            env.generation_contract,
            env.verifier,
            environment_contract_sha256=env.sha256,
        )
        receipt = build_decision_receipt(
            record_id=receipt_id,
            collection_contract_sha256=collection.sha256,
            episode_key=pending.episode_key,
            turn=transition.state.turn,
            observation_sha256=observation_digest,
            source="teacher",
            proposed_action=(
                None if pending.behavior.fallback else action
            ),
            selected_action=action,
            fallback=pending.behavior.fallback,
            controller_call_ref=DiagnosticRef(controller_call["record_id"]),
            generator_call_ref=DiagnosticRef(generator_call["record_id"]),
        )
        diagnostics_records.append(receipt)
        bound = replace(
            transition,
            collection_contract_sha256=collection.sha256,
            decision_provenance=DecisionProvenance(
                source=BehaviorSource.TEACHER,
                bc_eligible=pending.behavior.bc_eligible,
                fallback=pending.behavior.fallback,
                diagnostic_record_id=receipt_id,
                episode_key=pending.episode_key,
                observation_sha256=observation_digest,
            ),
        )
        run_records.append(build_transition_record(
            bound,
            environment=env,
            collection_contract=collection,
            episode_key=pending.episode_key,
            problem_id=pending.problem_id,
        ))

    diagnostic_path = tmp_path / "training-diagnostics.jsonl"
    diagnostic_path.write_bytes(b"".join(
        canonical_json_bytes(record) + b"\n"
        for record in diagnostics_records
    ))
    diagnostics = read_diagnostic_sidecar(
        diagnostic_path,
        environment_contract_sha256=env.sha256,
        collection_contract_sha256=collection.sha256,
        generator_transport_sha256=collection.generator_transport_sha256,
        controller_transport_sha256=collection.controller_transport_sha256,
    )
    lines = [canonical_json_bytes(build_run_header(env, collection)) + b"\n"]
    lines.extend(canonical_json_bytes(record) + b"\n" for record in run_records)
    resolver = Resolver({
        problem_id: initial_state(env, problem_id)
        for problem_id in ("casework", "direct-two", "direct-three")
    })
    replayed = replay_canonical_run_lines(
        lines,
        environment=env,
        initial_state_resolver=resolver,
        collection_contract=collection,
        diagnostic_index=diagnostics,
    )
    dataset = convert_replayed_run(replayed, env, 1.0)
    split = None
    for seed in range(20):
        candidate = build_grouped_split(dataset, 0.34, seed)
        if "casework" in candidate.train_problem_ids:
            split = candidate
            break
    assert split is not None
    return env, collection, dataset, split


@pytest.fixture(scope="module")
def training_data(tmp_path_factory):
    return trusted_dataset(tmp_path_factory.mktemp("training-v2"))


def model(seed=7):
    with learned.torch.random.fork_rng(devices=[]):
        learned.torch.default_generator.manual_seed(seed)
        return learned.FactoredControllerNetwork(
            FEATURE_DIM, **MODEL_CONFIG
        )


@requires_torch
def test_variable_target_collation_masks_and_provenance_losses(training_data):
    _, _, dataset, _ = training_data
    encoder = learned.DeterministicHashEncoder(ENCODER_DIM)
    batch = collate_phase_b_training_batch(dataset, (0, 1), encoder)

    assert batch.target_counts == (1, 3)
    assert batch.node_features.shape == (2, 3, FEATURE_DIM)
    assert not batch.legal_pair_mask[0, :, 1:].any()
    assert not batch.node_features[0, 1:].any()
    assert batch.bc_label_mask.tolist() == [False, True]

    network = model()
    mixed = compute_warm_start_loss(network, batch)
    labeled = compute_warm_start_loss(
        network,
        collate_phase_b_training_batch(dataset, (1,), encoder),
    )
    assert mixed.bc_label_count == 1
    assert mixed.critic_example_count == 2
    assert mixed.behavior_cloning.item() == pytest.approx(
        labeled.behavior_cloning.item()
    )
    assert not batch.legal_pair_mask[1, 0].any()
    before = mixed.behavior_cloning.item()
    with learned.torch.no_grad():
        network.verb_head.bias[0] += 10_000.0
    after = compute_warm_start_loss(network, batch)
    assert after.behavior_cloning.item() == pytest.approx(before)


@requires_torch
def test_fallback_only_batch_has_q_loss_and_no_policy_optimizer_update(
    training_data,
):
    _, _, dataset, _ = training_data
    encoder = learned.DeterministicHashEncoder(ENCODER_DIM)
    labeled = collate_phase_b_training_batch(dataset, (1,), encoder)
    fallback = collate_phase_b_training_batch(dataset, (0,), encoder)
    network = model()
    optimizer = learned.torch.optim.AdamW(
        network.parameters(), lr=1e-2, weight_decay=0.2
    )
    optimizer.zero_grad(set_to_none=True)
    compute_warm_start_loss(network, labeled).total.backward()
    optimizer.step()
    policy_names = tuple(
        name
        for name, _ in network.named_parameters()
        if name.startswith(("verb_head.", "pointer_head."))
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in network.named_parameters()
        if name in policy_names
    }

    optimizer.zero_grad(set_to_none=True)
    loss = compute_warm_start_loss(network, fallback)
    assert loss.bc_label_count == 0
    assert loss.behavior_cloning.item() == 0.0
    assert loss.q_regression.item() >= 0.0
    loss.total.backward()
    for name, parameter in network.named_parameters():
        if name in policy_names:
            assert parameter.grad is None
    optimizer.step()
    for name, parameter in network.named_parameters():
        if name in policy_names:
            assert learned.torch.equal(parameter, before[name])


@requires_torch
def test_public_batch_rejects_mask_and_return_tampering(training_data):
    _, _, dataset, _ = training_data
    batch = collate_phase_b_training_batch(
        dataset, (0, 1), learned.ZeroTextEncoder(ENCODER_DIM)
    )
    mask = batch.legal_pair_mask.clone()
    mask[0, batch.chosen_verb_indices[0], batch.chosen_target_indices[0]] = False
    with pytest.raises(TrainingValidationError, match="chosen action"):
        replace(batch, legal_pair_mask=mask)

    returns = batch.monte_carlo_returns.clone()
    returns[0] = float("nan")
    with pytest.raises(TrainingValidationError, match="nonfinite"):
        replace(batch, monte_carlo_returns=returns)


class CountingHashEncoder:
    dimension = ENCODER_DIM

    def __init__(self):
        self.calls = 0
        self._delegate = learned.DeterministicHashEncoder(self.dimension)

    def contract_record(self):
        return {
            "type": "hash",
            "implementation": (
                f"{type(self).__module__}.{type(self).__qualname__}"
            ),
            "algorithm_contract": learned.HASH_ENCODER_ALGORITHM,
            "model": None,
            "model_digest": None,
            "dimension": self.dimension,
        }

    def state_sha256(self):
        return hashlib.sha256(
            canonical_json_bytes(self.contract_record())
        ).hexdigest()

    def encode_many(self, texts):
        self.calls += 1
        return self._delegate.encode_many(texts)


@requires_torch
def test_training_is_deterministic_uses_every_label_and_caches_encoder(
    training_data,
):
    _, _, dataset, split = training_data
    config = WarmStartConfig(
        seed=11,
        epochs=2,
        batch_size=1,
        learning_rate=2e-3,
    )
    first_encoder = CountingHashEncoder()
    first = train_phase_b_warm_start(
        dataset,
        split,
        first_encoder,
        model_config=MODEL_CONFIG,
        config=config,
    )
    second = train_phase_b_warm_start(
        dataset,
        split,
        learned.DeterministicHashEncoder(ENCODER_DIM),
        model_config=MODEL_CONFIG,
        config=config,
    )
    repeated = train_phase_b_warm_start(
        dataset,
        split,
        learned.DeterministicHashEncoder(ENCODER_DIM),
        model_config=MODEL_CONFIG,
        config=config,
    )

    eligible = sum(example.bc_eligible for example in split.train_examples)
    assert eligible > 0
    assert any(
        example.bc_eligible and example.mc_return == 0.0
        for example in split.train_examples
    )
    assert first_encoder.calls == len(split.train_examples)
    assert second.state_dict_sha256 == repeated.state_dict_sha256
    assert second.metrics == repeated.metrics
    assert first.metrics.bc_labeled_examples == eligible
    assert first.metrics.bc_label_exposures == eligible * config.epochs
    assert first.metrics.critic_examples == (
        len(split.train_examples) * config.epochs
    )
    different_seed = train_phase_b_warm_start(
        dataset,
        split,
        learned.DeterministicHashEncoder(ENCODER_DIM),
        model_config=MODEL_CONFIG,
        config=replace(config, seed=12),
    )
    assert different_seed.state_dict_sha256 != second.state_dict_sha256


def deployment_collection(checkpoint_sha256, training):
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
        controller_temperature=0.0,
        controller_max_tokens=LEARNED_CONTROLLER_MAX_TOKENS,
        controller_seed_policy="none-v1",
        controller_output_contract=LEARNED_CONTROLLER_OUTPUT_CONTRACT,
        behavior_source="learned_policy",
        controller_fallback_policy="forbidden-v1",
        generator_implementation=training.generator_implementation,
        generator_transport=training.generator_transport,
        generation_seed=training.generation_seed,
        policy_input_contract=FEATURE_CONTRACT_VERSION,
        problem_source_contract="training-test-eval-problems-v1",
        problem_source_sha256="3" * 64,
        problem_adapter_implementation="tests.EvalProblemAdapter",
    )


@requires_torch
def test_checkpoint_is_exactly_loadable_and_rejects_post_training_mutation(
    training_data,
):
    env, collection, dataset, split = training_data
    encoder = learned.DeterministicHashEncoder(ENCODER_DIM)
    result = train_phase_b_warm_start(
        dataset,
        split,
        encoder,
        model_config=MODEL_CONFIG,
        config=WarmStartConfig(seed=5, epochs=1, batch_size=2),
    )
    checkpoint = build_phase_b_warm_start_checkpoint(
        result,
        dataset=dataset,
        split=split,
        environment=env,
        training_collection_contract=collection,
    )
    live = deployment_collection(checkpoint["contract_sha256"], collection)
    controller = learned.FactoredLearnedController(
        checkpoint,
        environment=env,
        live_collection_contract=live,
        encoder=encoder,
    )
    assert controller.checkpoint_state_dict_sha256 == (
        result.state_dict_sha256
    )
    assert checkpoint["contract_manifest"]["dataset_contract"] == (
        dataset.checkpoint_contract(split)
    )

    with learned.torch.no_grad():
        next(result.model.parameters()).add_(1.0)
    with pytest.raises(TrainingValidationError, match="changed"):
        build_phase_b_warm_start_checkpoint(
            result,
            dataset=dataset,
            split=split,
            environment=env,
            training_collection_contract=collection,
        )
