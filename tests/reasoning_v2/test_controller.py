from dataclasses import replace

import pytest

from verifier_anchored_rl.llm_client import (
    LLMResponseError,
    OpenAICompatibleLLMClient,
    SCHEMA_VALIDATION_LEGACY_COERCE_V1,
    SCHEMA_VALIDATION_STRICT_V1,
)
from verifier_anchored_rl.reasoning_v2.controller import (
    CONTROLLER_OUTPUT_CONTRACT_VERSION,
    LEGACY_CONTROLLER_OUTPUT_CONTRACT_VERSION,
    POLICY_INPUT_CONTRACT_VERSION,
    ControllerPayloadError,
    TypedLLMController,
    build_controller_request,
    decode_controller_payload,
    schema_validation_mode_for_controller_output,
)
from verifier_anchored_rl.reasoning_v2.contracts import (
    CollectionContract,
    ModelIdentity,
    ModelTransportContract,
    build_environment_contract,
    canonical_json_bytes,
)
from verifier_anchored_rl.reasoning_v2.lenses import (
    GENERATION_SCHEMA_VERSION,
    LENS_CONTRACT_VERSION,
)
from verifier_anchored_rl.reasoning_v2.math_adapter import (
    CaseworkMathVerifier,
)
from verifier_anchored_rl.reasoning_v2.model import (
    Action,
    GoalObjective,
    GoalObligation,
    State,
    Verb,
    new_graph,
)
from verifier_anchored_rl.reasoning_v2.provenance import BehaviorSource
from verifier_anchored_rl.reasoning_v2.runtime import (
    BoundEnvironment,
    EpisodeRuntime,
)
from verifier_anchored_rl.reasoning_v2.actions import ResultDraft


CONTROLLER_ID = ModelIdentity("test", "teacher", "r1")
GENERATOR_ID = ModelIdentity("test", "generator", "r1")


class CaptureClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete_json(
        self,
        system,
        user,
        schema,
        *,
        temperature=1.0,
        max_tokens=512,
        seed=None,
        schema_validation_mode=None,
    ):
        self.calls.append({
            "system": system,
            "user": user,
            "schema": schema,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "schema_validation_mode": schema_validation_mode,
        })
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class StaticBoundGenerator:
    generator_identity = GENERATOR_ID
    generator_implementation = f"{__name__}.StaticBoundGenerator"
    generator_transport_contract = ModelTransportContract.local(
        generator_implementation
    )
    generator_transport_sha256 = generator_transport_contract.sha256
    generation_seed = None
    generation_retry_budget = 0
    lens_contract = LENS_CONTRACT_VERSION
    generation_schema = GENERATION_SCHEMA_VERSION
    generation_seed_policy = "none-v1"

    def __init__(self, environment, draft):
        self.environment = environment
        self.draft = draft

    @property
    def environment_contract_sha256(self):
        return self.environment.sha256

    @property
    def generation_temperature(self):
        return self.environment.contract.generation_temperature

    @property
    def generation_max_tokens(self):
        return self.environment.contract.generation_max_tokens

    def generate(self, state, action):
        return self.draft


def environment():
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
        turn_limit=6,
    )
    return BoundEnvironment(contract, adapter, adapter, adapter)


def state(env, *, turn=0):
    graph = new_graph(
        "solve",
        GoalObligation(
            "x", "Eq(x, 5)", "Reals", GoalObjective.SUM
        ),
        generation_contract=env.generation_contract,
    )
    return State(graph, turn=turn)


def collection(controller):
    return CollectionContract(
        generator=GENERATOR_ID,
        controller=CONTROLLER_ID,
        controller_implementation=controller.controller_implementation,
        controller_transport=controller.controller_transport_contract,
        controller_seed=controller.controller_seed,
        behavior_source="teacher",
        controller_fallback_policy="first-legal-pair-v1",
        generator_implementation=(
            StaticBoundGenerator.generator_implementation
        ),
        generator_transport=(
            StaticBoundGenerator.generator_transport_contract
        ),
        generation_seed=None,
        policy_input_contract=controller.policy_input_contract,
        problem_source_contract="test-problems-v1",
        problem_source_sha256="3" * 64,
        problem_adapter_implementation="tests.ControllerProblemAdapter",
        controller_temperature=controller.controller_temperature,
        controller_max_tokens=controller.controller_max_tokens,
        controller_seed_policy=controller.controller_seed_policy,
        controller_output_contract=controller.controller_output_contract,
    )


def test_controller_request_is_canonical_and_uses_exact_wire_contract():
    env = environment()
    observation = env.observe(state(env))

    request = build_controller_request(observation, env)

    presentation = canonical_json_bytes({
        "schema_version": POLICY_INPUT_CONTRACT_VERSION,
        "observation": observation,
    }).decode("ascii")
    assert presentation in request.user
    assert request.schema["additionalProperties"] is False
    assert request.schema["properties"]["verb"]["enum"] == [
        "decompose",
        "propose_result",
        "propose_solution",
        "propose_refutation",
    ]
    request.schema["properties"]["verb"]["enum"].clear()
    fresh = build_controller_request(observation, env)
    assert fresh.schema["properties"]["verb"]["enum"] == [
        "decompose",
        "propose_result",
        "propose_solution",
        "propose_refutation",
    ]


def test_controller_output_versions_bind_validation_behavior():
    assert schema_validation_mode_for_controller_output(
        LEGACY_CONTROLLER_OUTPUT_CONTRACT_VERSION
    ) == SCHEMA_VALIDATION_LEGACY_COERCE_V1
    assert schema_validation_mode_for_controller_output(
        CONTROLLER_OUTPUT_CONTRACT_VERSION
    ) == SCHEMA_VALIDATION_STRICT_V1
    with pytest.raises(ValueError, match="unsupported"):
        schema_validation_mode_for_controller_output("unknown")

    env = environment()
    client = CaptureClient({"verb": "propose_result", "target": 0})
    legacy = TypedLLMController(
        env,
        client,
        CONTROLLER_ID,
        output_contract=LEGACY_CONTROLLER_OUTPUT_CONTRACT_VERSION,
    )
    legacy.choose(env.observe(state(env)))
    assert (
        client.calls[0]["schema_validation_mode"]
        == SCHEMA_VALIDATION_LEGACY_COERCE_V1
    )


def test_typed_controller_selects_action_and_binds_runtime_distribution():
    env = environment()
    client = CaptureClient({"verb": "propose_result", "target": 0})
    controller = TypedLLMController(
        env,
        client,
        CONTROLLER_ID,
        temperature=0.25,
        max_tokens=48,
    )
    initial = state(env)
    episode = EpisodeRuntime(
        env,
        collection(controller),
        StaticBoundGenerator(
            env, ResultDraft(GoalObjective.SUM, "5")
        ),
        initial,
        controller,
    )

    _, transition = episode.step()

    assert transition.action == Action(Verb.PROPOSE_RESULT, 0)
    assert transition.decision_provenance.source is BehaviorSource.TEACHER
    assert client.calls[0]["temperature"] == 0.25
    assert client.calls[0]["max_tokens"] == 48
    assert client.calls[0]["seed"] is None
    assert (
        client.calls[0]["schema_validation_mode"]
        == SCHEMA_VALIDATION_STRICT_V1
    )
    assert controller.policy_input_contract == POLICY_INPUT_CONTRACT_VERSION
    assert (
        controller.controller_output_contract
        == CONTROLLER_OUTPUT_CONTRACT_VERSION
    )


def test_controller_offsets_seed_by_turn_and_rejects_contract_drift():
    env = environment()
    client = CaptureClient({"verb": "propose_result", "target": 0})
    controller = TypedLLMController(
        env,
        client,
        CONTROLLER_ID,
        seed=17,
        seed_policy="turn-and-retry-offset-v1",
    )

    choice = controller.choose(env.observe(state(env, turn=2)))

    assert choice.action == Action(Verb.PROPOSE_RESULT, 0)
    assert client.calls[0]["seed"] == 19
    with pytest.raises(ValueError, match="controller_temperature"):
        EpisodeRuntime(
            env,
            replace(collection(controller), controller_temperature=0.5),
            StaticBoundGenerator(
                env, ResultDraft(GoalObjective.SUM, "5")
            ),
            state(env),
            controller,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"verb": "derive", "target": 0},
        {"verb": "propose_result", "target": True},
        {"verb": "propose_result", "target": -1},
        {"verb": "propose_result", "target": 0, "extra": None},
    ],
)
def test_controller_payload_decoder_rejects_nonexact_choices(payload):
    with pytest.raises(ControllerPayloadError):
        decode_controller_payload(payload)


def test_payload_or_transport_failure_defers_fallback_to_runtime():
    env = environment()
    malformed = TypedLLMController(
        env,
        CaptureClient({"verb": "propose_result", "target": "zero"}),
        CONTROLLER_ID,
    )
    failed = TypedLLMController(
        env,
        CaptureClient(LLMResponseError("failed")),
        CONTROLLER_ID,
    )
    observation = env.observe(state(env))

    assert malformed.choose(observation).action is None
    assert failed.choose(observation).action is None


def test_hosted_controller_requires_exact_model_and_revision():
    env = environment()
    client = OpenAICompatibleLLMClient(
        base_url="https://provider.example/v1",
        model="teacher",
        api_key="secret",
        max_retries=0,
        max_backoff_retries=0,
    )

    with pytest.raises(ValueError, match="immutable revision"):
        TypedLLMController(
            env, client, ModelIdentity("test", "teacher", None)
        )
    with pytest.raises(ValueError, match="configured client"):
        TypedLLMController(
            env, client, ModelIdentity("test", "other", "r1")
        )
