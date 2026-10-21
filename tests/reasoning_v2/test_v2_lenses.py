import hashlib

import pytest

from verifier_anchored_rl.llm_client import (
    LLMResponseError,
    MockLLMClient,
    OpenAICompatibleLLMClient,
    SCHEMA_VALIDATION_LEGACY_COERCE_V1,
    SCHEMA_VALIDATION_STRICT_V1,
)
from verifier_anchored_rl.reasoning_v2 import (
    Action,
    BehaviorSource,
    BoundEnvironment,
    CollectionContract,
    Connective,
    ControllerChoice,
    DecompositionDraft,
    EpisodeRuntime,
    GoalObjective,
    GoalObligation,
    ModelIdentity,
    ModelTransportContract,
    Ok,
    OutcomeCode,
    PolicyPayloadError,
    RefutationDraft,
    RefutationMode,
    RejectionReason,
    ResultDraft,
    SolutionDraft,
    State,
    TypedPolicyDraftGenerator,
    Verb,
    build_environment_contract,
    build_policy_lens_request,
    decode_policy_payload,
    decompose,
    new_graph,
    canonical_json_bytes,
    llm_transport_contract,
    render_policy_lens_view,
)
from verifier_anchored_rl.reasoning_v2.lenses import (
    GENERATION_SCHEMA_VERSION,
    LEGACY_GENERATION_SCHEMA_VERSION,
    LEGACY_LENS_CONTRACT_VERSION,
    LENS_CONTRACT_VERSION,
    schema_validation_mode_for_generation_schema,
)
from verifier_anchored_rl.reasoning_v2.controller import (
    CONTROLLER_OUTPUT_CONTRACT_VERSION,
)
from verifier_anchored_rl.reasoning_v2.math_adapter import (
    CaseworkMathVerifier,
)


GENERATOR_ID = ModelIdentity("test", "frozen-small-model", "r1")
CONTROLLER_ID = ModelIdentity("test", "teacher", "r1")


class ActionController:
    controller_identity = CONTROLLER_ID
    controller_implementation = f"{__name__}.ActionController"
    controller_transport_contract = ModelTransportContract.local(
        controller_implementation
    )
    controller_transport_sha256 = controller_transport_contract.sha256
    controller_seed = None
    controller_temperature = 0.0
    controller_max_tokens = 64
    controller_seed_policy = "none-v1"
    controller_output_contract = CONTROLLER_OUTPUT_CONTRACT_VERSION
    behavior_source = BehaviorSource.TEACHER
    policy_input_contract = "teacher-presentation-v2"

    def __init__(self, env, *actions):
        self.env = env
        self.actions = list(actions)

    @property
    def environment_contract_sha256(self):
        return self.env.sha256

    def choose(self, observation):
        return ControllerChoice(self.actions.pop(0))


def environment(
    *,
    seed_policy="none-v1",
    lens_contract=LENS_CONTRACT_VERSION,
    generation_schema=GENERATION_SCHEMA_VERSION,
):
    adapter = CaseworkMathVerifier()
    contract = build_environment_contract(
        generation_contract=adapter,
        verifier=adapter,
        aggregate_builder=adapter,
        lens_contract=lens_contract,
        generation_schema=generation_schema,
        generation_temperature=0.0,
        generation_max_tokens=64,
        generation_seed_policy=seed_policy,
        turn_limit=6,
    )
    return BoundEnvironment(contract, adapter, adapter, adapter)


def collection_for(generator):
    return CollectionContract(
        generator=GENERATOR_ID,
        controller=CONTROLLER_ID,
        controller_implementation=ActionController.controller_implementation,
        controller_transport=ActionController.controller_transport_contract,
        controller_seed=None,
        behavior_source=BehaviorSource.TEACHER.value,
        controller_fallback_policy="first-legal-pair-v1",
        generator_implementation=generator.generator_implementation,
        generator_transport=generator.generator_transport_contract,
        generation_seed=generator.generation_seed,
        policy_input_contract="teacher-presentation-v2",
        problem_source_contract="test-problems-v1",
        problem_source_sha256="2" * 64,
        problem_adapter_implementation="tests.TestProblemAdapter",
    )


def graph(env, objective=GoalObjective.SUM, *, equation="Eq(x, 5)", domain="Reals"):
    return new_graph(
        "solve the equation",
        GoalObligation("x", equation, domain, objective),
        generation_contract=env.generation_contract,
    )


def observation(env, objective=GoalObjective.SUM, **kwargs):
    return env.observe(State(graph(env, objective, **kwargs), turn=0))


def test_minimal_wire_payloads_inject_all_trusted_fields():
    env = environment()
    sum_observation = observation(env)

    decomposition = decode_policy_payload(
        sum_observation,
        Action(Verb.DECOMPOSE, 0),
        {"domains": ["x < 0", "x >= 0"], "ignored_reasoning": "extra"},
    )
    assert decomposition == DecompositionDraft(
        Connective.EXHAUSTIVE_CASE,
        ("case 1", "case 2"),
        (
            GoalObligation(
                "x", "Eq(x, 5)", "x < 0", GoalObjective.BRANCH
            ),
            GoalObligation(
                "x", "Eq(x, 5)", "x >= 0", GoalObjective.BRANCH
            ),
        ),
    )
    assert decode_policy_payload(
        sum_observation,
        Action(Verb.PROPOSE_RESULT, 0),
        {"value": 5},
    ) == ResultDraft(GoalObjective.SUM, "5")

    identity = observation(env, GoalObjective.IDENTITY)
    assert decode_policy_payload(
        identity,
        Action(Verb.PROPOSE_SOLUTION, 0),
        {"value": 5.0},
    ) == SolutionDraft("x", "5.0")

    branch = observation(
        env,
        GoalObjective.BRANCH,
        equation="Eq(x, -1)",
        domain="x > 0",
    )
    assert decode_policy_payload(
        branch,
        Action(Verb.PROPOSE_REFUTATION, 0),
        {"candidate": -1},
    ) == RefutationDraft(
        RefutationMode.UNIQUE_CANDIDATE_OUTSIDE_DOMAIN,
        "x",
        "-1",
    )


@pytest.mark.parametrize(
    ("action", "payload"),
    [
        (Action(Verb.PROPOSE_RESULT, 0), {"value": True}),
        (Action(Verb.PROPOSE_RESULT, 0), {}),
        (Action(Verb.DECOMPOSE, 0), {"domains": ["Reals"]}),
        (Action(Verb.DECOMPOSE, 0), {"domains": ["x < 0", ""]}),
        (Action(Verb.DECOMPOSE, 0), {"domains": "x < 0"}),
    ],
)
def test_wire_payload_rejects_ambiguous_shapes(action, payload):
    env = environment()
    with pytest.raises(PolicyPayloadError):
        decode_policy_payload(observation(env), action, payload)


def test_grounded_local_view_hides_a_sibling_branch():
    env = environment()
    initial = graph(
        env,
        equation="Eq((x - 2)*(x - 3), 0)",
    )
    result = decompose(
        initial,
        0,
        DecompositionDraft(
            Connective.EXHAUSTIVE_CASE,
            ("low secret", "high secret"),
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
        ),
        env.generation_contract,
    )
    assert isinstance(result, Ok)
    state_observation = env.observe(State(result.graph, turn=1))
    view = render_policy_lens_view(
        state_observation,
        Action(Verb.PROPOSE_SOLUTION, 1),
    )

    assert [goal["id"] for goal in view["goals"]] == [0, 1]
    rendered = str(view)
    assert "low secret" in rendered
    assert "high secret" not in rendered
    assert "answer" not in rendered
    assert "solution_set" not in rendered


def test_lens_request_is_deterministic_and_uses_one_minimal_schema():
    env = environment()
    state_observation = observation(env)
    action = Action(Verb.PROPOSE_RESULT, 0)

    first = build_policy_lens_request(state_observation, action)
    second = build_policy_lens_request(state_observation, action)

    assert first == second
    assert first.schema["required"] == ["value"]
    assert set(first.schema["properties"]) == {"value"}
    assert "Verb: PROPOSE_RESULT" in first.user

    first.schema["required"].clear()
    fresh = build_policy_lens_request(state_observation, action)
    assert fresh.schema["required"] == ["value"]


def test_generation_schema_versions_bind_validation_behavior():
    assert schema_validation_mode_for_generation_schema(
        LEGACY_GENERATION_SCHEMA_VERSION
    ) == SCHEMA_VALIDATION_LEGACY_COERCE_V1
    assert schema_validation_mode_for_generation_schema(
        GENERATION_SCHEMA_VERSION
    ) == SCHEMA_VALIDATION_STRICT_V1
    with pytest.raises(ValueError, match="unsupported"):
        schema_validation_mode_for_generation_schema("unknown")


def test_legacy_generation_schema_replays_coercion_but_current_is_strict():
    action = Action(Verb.PROPOSE_RESULT, 0)
    script = {("PROPOSE_RESULT",): {"value": 5}}

    legacy = environment(
        generation_schema=LEGACY_GENERATION_SCHEMA_VERSION
    )
    legacy_transition = legacy.step(
        State(graph(legacy), turn=0),
        action,
        TypedPolicyDraftGenerator(
            legacy, MockLLMClient(script), GENERATOR_ID
        ),
    )
    assert legacy_transition.outcome.code is OutcomeCode.APPLIED

    current = environment()
    current_transition = current.step(
        State(graph(current), turn=0),
        action,
        TypedPolicyDraftGenerator(
            current, MockLLMClient(script), GENERATOR_ID
        ),
    )
    assert current_transition.outcome.reason is RejectionReason.PAYLOAD_ERROR


def test_legacy_lens_request_retains_exact_v2_bytes():
    env = environment(
        lens_contract=LEGACY_LENS_CONTRACT_VERSION,
        generation_schema=LEGACY_GENERATION_SCHEMA_VERSION,
    )
    state_observation = observation(env)
    request = build_policy_lens_request(
        state_observation,
        Action(Verb.PROPOSE_RESULT, 0),
        lens_contract=LEGACY_LENS_CONTRACT_VERSION,
    )

    assert hashlib.sha256(request.system.encode("ascii")).hexdigest() == (
        "83244131e873f6d31cc955b7bc30af598e18c9a93dcf9a489f3052543049e11e"
    )
    assert hashlib.sha256(request.user.encode("ascii")).hexdigest() == (
        "c94871e91c04ceedef4fb1013fdf5136e4c1ff69788983ad361a91552db880cd"
    )
    assert '"schema_version":"typed-claim-lens-v2"' in request.user
    assert 'Return {"value":"symbolic value"}.' in request.user


@pytest.mark.parametrize(
    ("objective", "equation", "domain", "action", "expected_sha256"),
    [
        (
            GoalObjective.SUM,
            "Eq(x, 5)",
            "Reals",
            Action(Verb.DECOMPOSE, 0),
            "eb5c0ec738d7b0a98a8ad8b4efebb902"
            "f8b2bb292c1cf4b6c73ab4ee68ecb742",
        ),
        (
            GoalObjective.IDENTITY,
            "Eq(x, 5)",
            "Reals",
            Action(Verb.PROPOSE_SOLUTION, 0),
            "293d7e871c55c1b9fd6c2ac71bbd110"
            "e99494c057ab85c332116ceb632d530ff",
        ),
        (
            GoalObjective.BRANCH,
            "Eq(x, -1)",
            "x > 0",
            Action(Verb.PROPOSE_REFUTATION, 0),
            "e23ed06f57d4eb3e20b08d5bc97f844"
            "9243bf146b5431e68ca6fa40e6523a7f2",
        ),
    ],
)
def test_other_legacy_lens_verbs_retain_exact_v2_bytes(
    objective, equation, domain, action, expected_sha256
):
    env = environment(
        lens_contract=LEGACY_LENS_CONTRACT_VERSION,
        generation_schema=LEGACY_GENERATION_SCHEMA_VERSION,
    )
    request = build_policy_lens_request(
        observation(
            env,
            objective,
            equation=equation,
            domain=domain,
        ),
        action,
        lens_contract=LEGACY_LENS_CONTRACT_VERSION,
    )

    assert hashlib.sha256(request.user.encode("ascii")).hexdigest() == (
        expected_sha256
    )


def test_aligned_decompose_prompt_exposes_only_ascii_domain_grammar():
    env = environment()
    request = build_policy_lens_request(
        observation(
            env,
            equation="Eq(Abs(x), 1)",
        ),
        Action(Verb.DECOMPOSE, 0),
    )

    assert request.user.isascii()
    assert '"domain 1"' not in request.user
    assert '"domain 2"' not in request.user
    assert "The complete domain grammar is" in request.user
    assert "followed by one of <, <=, >, >=" in request.user
    assert "another < or <=" in request.user
    assert "never use interval notation" in request.user
    assert "infinity, inf, oo, or Unicode symbols" in request.user
    assert "pairwise disjoint" in request.user
    assert "their union must be exactly Reals" in request.user
    assert "include the boundary in exactly one adjacent domain" in request.user
    assert "Return no equation text, objective labels" in request.user
    assert "Return no equations, variables" not in request.user


@pytest.mark.parametrize(
    ("objective", "domain", "action", "field"),
    [
        (
            GoalObjective.SUM,
            "Reals",
            Action(Verb.PROPOSE_RESULT, 0),
            "value",
        ),
        (
            GoalObjective.IDENTITY,
            "Reals",
            Action(Verb.PROPOSE_SOLUTION, 0),
            "value",
        ),
        (
            GoalObjective.BRANCH,
            "x > 0",
            Action(Verb.PROPOSE_REFUTATION, 0),
            "candidate",
        ),
    ],
)
def test_aligned_scalar_prompts_expose_parser_grammar_without_placeholder(
    objective, domain, action, field
):
    env = environment()
    request = build_policy_lens_request(
        observation(
            env,
            objective,
            equation="Eq(x, -1)",
            domain=domain,
        ),
        action,
    )

    assert request.user.isascii()
    assert "symbolic value" not in request.user
    assert f"exactly one field named {field}" in request.user
    assert "must evaluate to one finite real number" in request.user
    assert "may contain only digits, spaces" in request.user
    assert "Do not use **, //" in request.user
    assert "Do not copy the lens view" in request.user


def test_typed_generator_selects_legacy_prompt_from_environment():
    env = environment(lens_contract=LEGACY_LENS_CONTRACT_VERSION)
    client = MockLLMClient({("PROPOSE_RESULT",): {"value": "5"}})
    generator = TypedPolicyDraftGenerator(env, client, GENERATOR_ID)

    draft = generator.generate(
        State(graph(env), turn=0),
        Action(Verb.PROPOSE_RESULT, 0),
    )

    assert draft == ResultDraft(GoalObjective.SUM, "5")
    assert generator.lens_contract == LEGACY_LENS_CONTRACT_VERSION
    assert 'Return {"value":"symbolic value"}.' in client.calls[0][1]


def test_lens_builder_and_typed_generator_reject_unknown_versions():
    env = environment()
    state_observation = observation(env)
    with pytest.raises(ValueError, match="lens contract is unsupported"):
        build_policy_lens_request(
            state_observation,
            Action(Verb.PROPOSE_RESULT, 0),
            lens_contract="typed-claim-lens-unknown",
        )

    unknown = environment(lens_contract="typed-claim-lens-unknown")
    with pytest.raises(
        ValueError, match="environment lens contract is unsupported"
    ):
        TypedPolicyDraftGenerator(
            unknown,
            MockLLMClient({}),
            GENERATOR_ID,
        )


def test_typed_generator_maps_transport_and_payload_failures_separately():
    env = environment()
    state = State(graph(env), turn=0)
    action = Action(Verb.PROPOSE_RESULT, 0)

    missing = TypedPolicyDraftGenerator(
        env,
        MockLLMClient({}),
        GENERATOR_ID,
    )
    generation_failure = env.step(state, action, missing)
    assert (
        generation_failure.outcome.reason
        is RejectionReason.GENERATION_ERROR
    )

    malformed = TypedPolicyDraftGenerator(
        env,
        MockLLMClient({("PROPOSE_RESULT",): {}}),
        GENERATOR_ID,
    )
    payload_failure = env.step(state, action, malformed)
    assert payload_failure.outcome.reason is RejectionReason.PAYLOAD_ERROR


def test_unexpected_client_bug_aborts_collection():
    class BrokenClient:
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
            assert schema_validation_mode == SCHEMA_VALIDATION_STRICT_V1
            raise AssertionError("client implementation bug")

    env = environment()
    generator = TypedPolicyDraftGenerator(
        env,
        BrokenClient(),
        GENERATOR_ID,
    )

    with pytest.raises(AssertionError, match="implementation bug"):
        env.step(
            State(graph(env), turn=0),
            Action(Verb.PROPOSE_RESULT, 0),
            generator,
        )


def test_episode_runtime_binds_generator_and_solves_through_the_lens():
    env = environment()
    client = MockLLMClient({
        ("PROPOSE_RESULT",): {"value": "5", "reasoning": "ignored"}
    })
    generator = TypedPolicyDraftGenerator(env, client, GENERATOR_ID)
    collection = collection_for(generator)
    episode = EpisodeRuntime(
        environment=env,
        collection_contract=collection,
        generator=generator,
        state=State(graph(env), turn=0),
    )

    next_episode, transition = episode.step(
        Action(Verb.PROPOSE_RESULT, 0)
    )

    assert transition.outcome.code is OutcomeCode.APPLIED
    assert transition.reward == 1
    assert transition.terminal
    assert next_episode.state == transition.next_state
    assert len(client.calls) == 1


def test_episode_runtime_rejects_a_generator_from_another_environment():
    env = environment()
    other = environment(seed_policy="turn-and-retry-offset-v1")
    generator = TypedPolicyDraftGenerator(
        other,
        MockLLMClient({}),
        GENERATOR_ID,
        seed=7,
    )
    collection = collection_for(generator)

    with pytest.raises(
        ValueError,
        match="seed|environment_contract_sha256",
    ):
        EpisodeRuntime(
            environment=env,
            collection_contract=collection,
            generator=generator,
            state=State(graph(env), turn=0),
        )


def test_turn_and_retry_seed_contract_forwards_effective_turn_seed():
    class CaptureClient:
        def __init__(self):
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
            self.calls.append(
                (temperature, max_tokens, seed, schema_validation_mode)
            )
            value = "6" if len(self.calls) == 1 else "5"
            return {"value": value}

    env = environment(seed_policy="turn-and-retry-offset-v1")
    client = CaptureClient()
    generator = TypedPolicyDraftGenerator(
        env,
        client,
        GENERATOR_ID,
        seed=7,
    )
    episode = EpisodeRuntime(
        env,
        collection_for(generator),
        generator,
        State(graph(env), turn=0),
        ActionController(
            env,
            Action(Verb.PROPOSE_RESULT, 0),
            Action(Verb.PROPOSE_RESULT, 0),
        ),
    )

    episode, first = episode.step()
    _, second = episode.step()

    assert not first.terminal
    assert second.terminal
    assert client.calls == [
        (0.0, 64, 7, SCHEMA_VALIDATION_STRICT_V1),
        (0.0, 64, 8, SCHEMA_VALIDATION_STRICT_V1),
    ]
    assert first.collection_contract_sha256 == (
        collection_for(generator).sha256
    )


def test_configured_client_model_must_match_declared_identity():
    class ModelClient:
        model = "actual-model"

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
            return {"value": "5"}

    with pytest.raises(ValueError, match="configured client"):
        TypedPolicyDraftGenerator(
            environment(),
            ModelClient(),
            GENERATOR_ID,
        )


def test_hosted_transport_contract_is_exact_and_secret_free():
    client = OpenAICompatibleLLMClient(
        base_url="https://router.huggingface.co/v1/",
        model=GENERATOR_ID.model,
        api_key="hf_secret_token",
        timeout_s=45.0,
        max_retries=2,
        max_backoff_retries=4,
        guided_json=True,
    )
    contract = llm_transport_contract(client)

    assert contract.base_url == "https://router.huggingface.co/v1"
    assert contract.model == GENERATOR_ID.model
    assert contract.parse_retry_budget == 2
    assert contract.backoff_retry_budget == 4
    assert contract.guided_json is True
    assert b"hf_secret_token" not in canonical_json_bytes(
        contract.to_record()
    )

    changed = OpenAICompatibleLLMClient(
        base_url="https://router.huggingface.co/v1",
        model=GENERATOR_ID.model,
        api_key="different_secret",
        timeout_s=45.0,
        max_retries=2,
        max_backoff_retries=4,
        guided_json=False,
    )
    assert llm_transport_contract(changed).sha256 != contract.sha256


def test_hosted_generator_requires_revision_and_no_extra_headers():
    no_revision = ModelIdentity("huggingface", GENERATOR_ID.model, None)
    client = OpenAICompatibleLLMClient(
        base_url="https://router.huggingface.co/v1",
        model=GENERATOR_ID.model,
        api_key="secret",
    )

    with pytest.raises(ValueError, match="immutable revision"):
        TypedPolicyDraftGenerator(
            environment(),
            client,
            no_revision,
        )

    with pytest.raises(ValueError, match="extra headers"):
        llm_transport_contract(
            OpenAICompatibleLLMClient(
                base_url="https://router.huggingface.co/v1",
                model=GENERATOR_ID.model,
                extra_headers={"X-Route": "mutable"},
            )
        )


def test_raw_schema_failure_is_payload_error_not_transport_error():
    class RawFailureClient:
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
            raise LLMResponseError(
                "missing value",
                raw_response="{}",
            )

    env = environment()
    transition = env.step(
        State(graph(env), turn=0),
        Action(Verb.PROPOSE_RESULT, 0),
        TypedPolicyDraftGenerator(
            env,
            RawFailureClient(),
            GENERATOR_ID,
        ),
    )

    assert transition.outcome.reason is RejectionReason.PAYLOAD_ERROR


def test_local_view_drops_a_previous_outcome_for_a_hidden_sibling():
    class Drafts:
        def __init__(self, *items):
            self.items = list(items)

        def generate(self, state, action):
            return self.items.pop(0)

    env = environment()
    initial = graph(
        env,
        equation="Eq((x - 2)*(x - 3), 0)",
    )
    drafts = Drafts(
        DecompositionDraft(
            Connective.EXHAUSTIVE_CASE,
            ("low", "high"),
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
        ),
        SolutionDraft("x", "2"),
    )
    split = env.step(
        State(initial, turn=0),
        Action(Verb.DECOMPOSE, 0),
        drafts,
    )
    solved_low = env.step(
        split.next_state,
        Action(Verb.PROPOSE_SOLUTION, 1),
        drafts,
    )
    view = render_policy_lens_view(
        env.observe(solved_low.next_state),
        Action(Verb.PROPOSE_SOLUTION, 2),
    )

    assert [goal["id"] for goal in view["goals"]] == [0, 2]
    assert view["previous_outcome"] is None
