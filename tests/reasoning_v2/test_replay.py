from copy import deepcopy
from dataclasses import dataclass, replace

import pytest

from verifier_anchored_rl.llm_client import (
    OpenAICompatibleAttempt,
    build_openai_compatible_json_body,
)
from verifier_anchored_rl.reasoning_v2 import (
    Action,
    BoundEnvironment,
    CollectionContract,
    Connective,
    DecompositionDraft,
    DiagnosticRef,
    GoalObjective,
    GoalObligation,
    ModelIdentity,
    PolicyGenerationError,
    RefutationDraft,
    RefutationMode,
    RejectionReason,
    ResultDraft,
    SolutionDraft,
    State,
    Verb,
    build_environment_contract,
    build_run_header,
    build_transition_record,
    canonical_json_bytes,
    new_graph,
    observation_sha256,
)
from verifier_anchored_rl.reasoning_v2.contracts import ModelTransportContract
from verifier_anchored_rl.reasoning_v2.controller import (
    POLICY_INPUT_CONTRACT_VERSION,
    build_controller_request,
)
from verifier_anchored_rl.reasoning_v2.diagnostics import (
    build_attempt_record,
    build_call_record,
    build_decision_receipt,
    build_diagnostic_header,
    read_diagnostic_sidecar,
)
from verifier_anchored_rl.reasoning_v2.lenses import (
    GENERATION_SCHEMA_VERSION,
    LENS_CONTRACT_VERSION,
    build_policy_lens_request,
)
from verifier_anchored_rl.reasoning_v2.math_adapter import CaseworkMathVerifier
from verifier_anchored_rl.reasoning_v2.replay import (
    ReplayValidationError,
    replay_canonical_run,
    replay_canonical_run_lines,
)
from verifier_anchored_rl.reasoning_v2.provenance import (
    BehaviorSource as ProvenanceBehaviorSource,
    DecisionProvenance,
)


class QueueGenerator:
    def __init__(self, *drafts):
        self.drafts = list(drafts)

    def generate(self, state, action):
        return self.drafts.pop(0)


class FailingGenerator:
    def generate(self, state, action):
        raise PolicyGenerationError("provider failed")


@dataclass(frozen=True)
class TrustedProblemResolver:
    states: dict[str, State]
    problem_source_contract: str = "test-problems-v1"
    problem_source_sha256: str = "2" * 64
    problem_adapter_implementation: str = "tests.TestProblemAdapter"

    def __call__(self, problem_id):
        return self.states[problem_id]


def environment(*, turn_limit=2):
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
        turn_limit=turn_limit,
    )
    return BoundEnvironment(contract, adapter, adapter, adapter)


def hosted_transport(model):
    return ModelTransportContract(
        implementation=(
            "verifier_anchored_rl.llm_client.OpenAICompatibleLLMClient"
        ),
        base_url="https://provider.example/v1",
        model=model,
        timeout_s=30.0,
        parse_retry_budget=0,
        backoff_retry_budget=0,
        guided_json=False,
        response_format_contract="openai-json-object-v1",
        parse_retry_contract="assistant-echo-seed-offset-v1",
        backoff_contract="retry-after-exponential-jitter-v1",
    )


def collection(*, behavior_source="learned_policy", hosted=False):
    return CollectionContract(
        generator=ModelIdentity("test", "generator", "r1"),
        controller=ModelIdentity("test", "controller", "r1"),
        controller_implementation=(
            "verifier_anchored_rl.reasoning_v2.controller.TypedLLMController"
            if hosted
            else "tests.TeacherController"
        ),
        controller_transport=(
            hosted_transport("controller")
            if hosted
            else ModelTransportContract.local("tests.TeacherController")
        ),
        controller_seed=0,
        controller_temperature=0.0,
        controller_max_tokens=64,
        controller_seed_policy="turn-and-retry-offset-v1",
        behavior_source=behavior_source,
        controller_fallback_policy=(
            "first-legal-pair-v1"
            if behavior_source == "teacher"
            else "forbidden-v1"
        ),
        generator_implementation=(
            "verifier_anchored_rl.reasoning_v2.lenses."
            "TypedPolicyDraftGenerator"
            if hosted
            else "tests.QueueGenerator"
        ),
        generator_transport=(
            hosted_transport("generator")
            if hosted
            else ModelTransportContract.local("tests.QueueGenerator")
        ),
        generation_seed=None,
        policy_input_contract=POLICY_INPUT_CONTRACT_VERSION,
        problem_source_contract="test-problems-v1",
        problem_source_sha256="2" * 64,
        problem_adapter_implementation="tests.TestProblemAdapter",
    )


def graph(env, value):
    return new_graph(
        f"solve for {value}",
        GoalObligation(
            "x", f"Eq(x, {value})", "Reals", GoalObjective.SUM
        ),
        ("trusted premise",),
        generation_contract=env.generation_contract,
    )


def trusted_states(env, values):
    return TrustedProblemResolver({
        problem_id: State(graph(env, value), turn=0)
        for problem_id, value in values.items()
    })


def step(env, run_contract, state, action, draft):
    transition = env.step(state, action, QueueGenerator(draft))
    return replace(
        transition,
        collection_contract_sha256=run_contract.sha256,
        decision_provenance=DecisionProvenance(
            source=ProvenanceBehaviorSource(
                run_contract.behavior_source
            ),
            bc_eligible=(run_contract.behavior_source == "teacher"),
            fallback=False,
            diagnostic_record_id=None,
        ),
    )


def record(env, run_contract, transition, episode_key, problem_id):
    return build_transition_record(
        transition,
        environment=env,
        collection_contract=run_contract,
        episode_key=episode_key,
        problem_id=problem_id,
    )


def with_diagnostic_provenance(env, transition, episode_key, record_id):
    return replace(
        transition,
        decision_provenance=replace(
            transition.decision_provenance,
            diagnostic_record_id=record_id,
            episode_key=episode_key,
            observation_sha256=observation_sha256(
                env.observe(transition.state),
                env.generation_contract,
                env.verifier,
                environment_contract_sha256=env.sha256,
            ),
        ),
    )


def line(value):
    return canonical_json_bytes(value) + b"\n"


def decision_sidecar(
    env,
    run_contract,
    transition,
    *,
    receipt_episode_key,
    path,
    controller_attempts=False,
    controller_payload=None,
    generator_payload=None,
):
    observation = env.observe(transition.state)
    receipt_id = "a" * 32
    generator_call_id = "b" * 32
    generator_request = build_policy_lens_request(
        observation, transition.action
    )
    generator_seed = (
        None
        if env.contract.generation_seed_policy == "none-v1"
        else run_contract.generation_seed + transition.state.turn
    )
    generator_body = build_openai_compatible_json_body(
        model=run_contract.generator_transport.model,
        system=generator_request.system,
        user=generator_request.user,
        schema=generator_request.schema,
        temperature=env.contract.generation_temperature,
        max_tokens=env.contract.generation_max_tokens,
        seed=generator_seed,
        guided_json=run_contract.generator_transport.guided_json,
    )
    raw_generator_content = canonical_json_bytes(
        {"value": "5"} if generator_payload is None else generator_payload
    ).decode("ascii")
    raw_generator_response = canonical_json_bytes({
        "choices": [{
            "message": {"content": raw_generator_content}
        }]
    }).decode("ascii")
    attempt = build_attempt_record(
        OpenAICompatibleAttempt(
            call_id=generator_call_id,
            attempt_index=0,
            parse_retry_index=0,
            backoff_retry_index=0,
            effective_seed=None,
            request_body_json=canonical_json_bytes(generator_body).decode(
                "ascii"
            ),
            raw_response_body=raw_generator_response,
            raw_content=raw_generator_content,
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
        role="generator",
    )
    call = build_call_record(
        "generator", generator_call_id, [attempt]
    )
    controller_records = []
    controller_ref = None
    if controller_attempts:
        controller_call_id = "c" * 32
        controller_seed = run_contract.controller_seed + transition.state.turn
        controller_request = build_controller_request(observation, env)
        controller_body = build_openai_compatible_json_body(
            model=run_contract.controller_transport.model,
            system=controller_request.system,
            user=controller_request.user,
            schema=controller_request.schema,
            temperature=run_contract.controller_temperature,
            max_tokens=run_contract.controller_max_tokens,
            seed=controller_seed,
            guided_json=run_contract.controller_transport.guided_json,
        )
        raw_controller_content = canonical_json_bytes(
            {"value": "not an action"}
            if controller_payload is None
            else controller_payload
        ).decode("ascii")
        raw_controller_response = canonical_json_bytes({
            "choices": [{
                "message": {"content": raw_controller_content}
            }]
        }).decode("ascii")
        controller_attempt = build_attempt_record(
            OpenAICompatibleAttempt(
                call_id=controller_call_id,
                attempt_index=0,
                parse_retry_index=0,
                backoff_retry_index=0,
                effective_seed=controller_seed,
                request_body_json=canonical_json_bytes(
                    controller_body
                ).decode("ascii"),
                raw_response_body=raw_controller_response,
                raw_content=raw_controller_content,
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
            role="controller",
        )
        controller_call = build_call_record(
            "controller", controller_call_id, [controller_attempt]
        )
        controller_records = [controller_attempt, controller_call]
        controller_ref = DiagnosticRef(controller_call["record_id"])
    provenance = transition.decision_provenance
    receipt = build_decision_receipt(
        record_id=receipt_id,
        collection_contract_sha256=run_contract.sha256,
        episode_key=receipt_episode_key,
        turn=transition.state.turn,
        observation_sha256=observation_sha256(
            observation,
            env.generation_contract,
            env.verifier,
            environment_contract_sha256=env.sha256,
        ),
        source=provenance.source.value,
        proposed_action=transition.action,
        selected_action=transition.action,
        fallback=provenance.fallback,
        controller_call_ref=controller_ref,
        generator_call_ref=DiagnosticRef(call["record_id"]),
    )
    header = build_diagnostic_header(
        environment_contract_sha256=env.sha256,
        collection_contract_sha256=run_contract.sha256,
        generator_transport_sha256=(
            run_contract.generator_transport_sha256
        ),
        controller_transport_sha256=(
            run_contract.controller_transport_sha256
        ),
        controller_decision_receipts=True,
        controller_attempts=controller_attempts,
    )
    path.write_bytes(
        b"".join(
            line(item)
            for item in (
                header,
                *controller_records,
                attempt,
                call,
                receipt,
            )
        )
    )
    return read_diagnostic_sidecar(
        path,
        environment_contract_sha256=env.sha256,
        collection_contract_sha256=run_contract.sha256,
        generator_transport_sha256=run_contract.generator_transport_sha256,
        controller_transport_sha256=(
            run_contract.controller_transport_sha256
        ),
    ), receipt_id


def two_turn_episode(env, run_contract, key, problem_id, value):
    first = step(
        env,
        run_contract,
        State(graph(env, value), turn=0),
        Action(Verb.PROPOSE_RESULT, 0),
        ResultDraft(GoalObjective.SUM, str(value + 1)),
    )
    second = step(
        env,
        run_contract,
        first.next_state,
        Action(Verb.PROPOSE_RESULT, 0),
        ResultDraft(GoalObjective.SUM, str(value)),
    )
    return (
        record(env, run_contract, first, key, problem_id),
        record(env, run_contract, second, key, problem_id),
    )


def test_strict_replay_supports_interleaved_complete_episodes(tmp_path):
    env = environment()
    run_contract = collection()
    first_a, second_a = two_turn_episode(
        env, run_contract, "episode:a", "problem-a", 5
    )
    first_b, second_b = two_turn_episode(
        env, run_contract, "episode:b", "problem-b", 7
    )
    lines = [
        line(build_run_header(env, run_contract)),
        line(first_a),
        line(first_b),
        line(second_a),
        line(second_b),
    ]

    replayed = replay_canonical_run_lines(
        lines,
        environment=env,
        initial_state_resolver=trusted_states(
            env, {"problem-a": 5, "problem-b": 7}
        ),
    )

    assert [episode.episode_key for episode in replayed.episodes] == [
        "episode:a",
        "episode:b",
    ]
    assert [len(episode.decisions) for episode in replayed.episodes] == [2, 2]
    assert all(episode.terminal for episode in replayed.episodes)
    assert all(
        decision.transition.collection_contract_sha256
        == run_contract.sha256
        for decision in replayed.decisions
    )
    terminal_a = replayed.episodes[0].decisions[-1].transition
    assert terminal_a.next_state.graph.nodes[-1].statement == "result(sum) = 5"
    assert terminal_a.next_state.graph.nodes[-1].verification is not None

    path = tmp_path / "run.jsonl"
    path.write_bytes(b"".join(lines))
    from_path = replay_canonical_run(
        path,
        environment=env,
        initial_state_resolver=trusted_states(
            env, {"problem-a": 5, "problem-b": 7}
        ),
        collection_contract=run_contract,
    )
    assert from_path == replayed


def test_strict_replay_retains_effect_hidden_by_terminal_aggregate_compaction():
    env = environment(turn_limit=3)
    run_contract = collection()
    initial = new_graph(
        "sum roots",
        GoalObligation(
            "x",
            "Eq((x - 2)*(x - 3), 0)",
            "Reals",
            GoalObjective.SUM,
        ),
        generation_contract=env.generation_contract,
    )
    decomposition = DecompositionDraft(
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
    )
    first = step(
        env,
        run_contract,
        State(initial, turn=0),
        Action(Verb.DECOMPOSE, 0),
        decomposition,
    )
    second = step(
        env,
        run_contract,
        first.next_state,
        Action(Verb.PROPOSE_SOLUTION, 1),
        SolutionDraft("x", "2"),
    )
    third = step(
        env,
        run_contract,
        second.next_state,
        Action(Verb.PROPOSE_SOLUTION, 2),
        SolutionDraft("x", "3"),
    )
    records = [
        record(env, run_contract, item, "episode:aggregate", "problem-sum")
        for item in (first, second, third)
    ]
    assert records[-1]["next_observation"]["claims"] == []

    replayed = replay_canonical_run_lines(
        [line(build_run_header(env, run_contract)), *(line(x) for x in records)],
        environment=env,
        initial_state_resolver=TrustedProblemResolver({
            "problem-sum": State(initial, turn=0)
        }),
        collection_contract=run_contract,
    )

    final = replayed.episodes[0].decisions[-1].transition
    assert final.terminal
    assert len(final.next_state.graph.nodes) == 6
    assert final.next_state.graph.nodes[4].statement == "x = 3"
    assert final.next_state.graph.nodes[5].kind.value == "aggregate"


@pytest.mark.parametrize("failure", ["generation", "algebra"])
def test_strict_replay_reproduces_rejected_effects_without_a_model(failure):
    env = environment(turn_limit=1)
    run_contract = collection()
    state = State(graph(env, 5), turn=0)
    action = Action(Verb.PROPOSE_RESULT, 0)
    generator = (
        FailingGenerator()
        if failure == "generation"
        else QueueGenerator(ResultDraft(GoalObjective.SUM, "not symbolic"))
    )
    transition = env.step(state, action, generator)
    transition = replace(
        transition,
        collection_contract_sha256=run_contract.sha256,
        decision_provenance=DecisionProvenance(
            source=ProvenanceBehaviorSource(
                run_contract.behavior_source
            ),
            bc_eligible=(run_contract.behavior_source == "teacher"),
            fallback=False,
            diagnostic_record_id=None,
        ),
    )
    persisted = record(
        env, run_contract, transition, "episode:failure", "problem-failure"
    )

    replayed = replay_canonical_run_lines(
        [line(build_run_header(env, run_contract)), line(persisted)],
        environment=env,
        initial_state_resolver=TrustedProblemResolver({
            "problem-failure": state
        }),
        collection_contract=run_contract,
    )

    expected_reason = (
        RejectionReason.GENERATION_ERROR
        if failure == "generation"
        else RejectionReason.ALGEBRA_REJECTED
    )
    decision = replayed.episodes[0].decisions[0]
    assert decision.transition.outcome.reason is expected_reason
    assert decision.transition.terminal


@pytest.mark.parametrize("case", ["headerless", "second_header", "mixed"])
def test_strict_replay_rejects_headerless_and_mixed_inputs(case):
    env = environment()
    run_contract = collection()
    first, second = two_turn_episode(
        env, run_contract, "episode:a", "problem-a", 5
    )
    header = build_run_header(env, run_contract)
    if case == "headerless":
        lines = [line(first), line(second)]
    elif case == "second_header":
        lines = [line(header), line(first), line(header), line(second)]
    else:
        mixed = deepcopy(first)
        mixed["schema_version"] = "legacy-controller-transition-v1"
        lines = [line(header), line(mixed)]

    with pytest.raises(ReplayValidationError):
        replay_canonical_run_lines(
            lines,
            environment=env,
            initial_state_resolver=trusted_states(env, {"problem-a": 5}),
            collection_contract=run_contract,
        )


def test_strict_replay_rejects_mid_episode_and_post_terminal_rows():
    env = environment()
    run_contract = collection()
    first, second = two_turn_episode(
        env, run_contract, "episode:a", "problem-a", 5
    )
    header = line(build_run_header(env, run_contract))

    with pytest.raises(ReplayValidationError, match="turn zero"):
        replay_canonical_run_lines(
            [header, line(second)],
            environment=env,
            initial_state_resolver=trusted_states(env, {"problem-a": 5}),
            collection_contract=run_contract,
        )
    with pytest.raises(ReplayValidationError, match="after its terminal"):
        replay_canonical_run_lines(
            [header, line(first), line(second), line(second)],
            environment=env,
            initial_state_resolver=trusted_states(env, {"problem-a": 5}),
            collection_contract=run_contract,
        )


@pytest.mark.parametrize("field", ["learned_effect", "closure_events"])
def test_strict_replay_rejects_forged_effects_and_closure(field):
    env = environment()
    run_contract = collection()
    first, second = two_turn_episode(
        env, run_contract, "episode:a", "problem-a", 5
    )
    forged = deepcopy(second)
    if field == "learned_effect":
        forged["learned_effect"]["payload"]["value"] = "4"
    else:
        forged["closure_events"] = []

    with pytest.raises(ReplayValidationError, match="line 3"):
        replay_canonical_run_lines(
            [line(build_run_header(env, run_contract)), line(first), line(forged)],
            environment=env,
            initial_state_resolver=trusted_states(env, {"problem-a": 5}),
            collection_contract=run_contract,
        )


def test_strict_replay_rejects_broken_episode_chain():
    env = environment()
    run_contract = collection()
    first_a, _ = two_turn_episode(
        env, run_contract, "episode:a", "problem-a", 5
    )
    _, second_b = two_turn_episode(
        env, run_contract, "episode:b", "problem-b", 7
    )
    broken = deepcopy(second_b)
    broken["episode_key"] = "episode:a"
    broken["problem_id"] = "problem-a"

    with pytest.raises(ReplayValidationError, match="current observation"):
        replay_canonical_run_lines(
            [
                line(build_run_header(env, run_contract)),
                line(first_a),
                line(broken),
            ],
            environment=env,
            initial_state_resolver=trusted_states(env, {"problem-a": 5}),
            collection_contract=run_contract,
        )


def test_strict_replay_rejects_behavior_source_forged_against_header():
    env = environment(turn_limit=1)
    run_contract = collection()
    transition = step(
        env,
        run_contract,
        State(graph(env, 5), turn=0),
        Action(Verb.PROPOSE_RESULT, 0),
        ResultDraft(GoalObjective.SUM, "5"),
    )
    forged = record(
        env, run_contract, transition, "episode:a", "problem-a"
    )
    forged["behavior"] = {
        "source": "teacher",
        "bc_eligible": True,
        "fallback": False,
    }

    with pytest.raises(ReplayValidationError, match="behavior source"):
        replay_canonical_run_lines(
            [line(build_run_header(env, run_contract)), line(forged)],
            environment=env,
            initial_state_resolver=trusted_states(env, {"problem-a": 5}),
            collection_contract=run_contract,
        )


def test_strict_replay_enforces_the_pinned_teacher_fallback_action():
    env = environment(turn_limit=1)
    run_contract = collection(behavior_source="teacher")
    transition = step(
        env,
        run_contract,
        State(graph(env, 5), turn=0),
        Action(Verb.PROPOSE_RESULT, 0),
        ResultDraft(GoalObjective.SUM, "5"),
    )
    forged = record(
        env, run_contract, transition, "episode:a", "problem-a"
    )
    assert forged["action"] != forged["observation"]["legal_pairs"][0]
    forged["behavior"] = {
        "source": "teacher",
        "bc_eligible": False,
        "fallback": True,
    }

    with pytest.raises(ReplayValidationError, match="first legal pair"):
        replay_canonical_run_lines(
            [line(build_run_header(env, run_contract)), line(forged)],
            environment=env,
            initial_state_resolver=trusted_states(env, {"problem-a": 5}),
            collection_contract=run_contract,
        )


def test_strict_replay_rejects_a_nonclean_turn_zero_graph():
    env = environment(turn_limit=1)
    run_contract = collection()
    initial = new_graph(
        "split roots",
        GoalObligation(
            "x", "Eq((x - 2)*(x - 3), 0)", "Reals", GoalObjective.SUM
        ),
        generation_contract=env.generation_contract,
    )
    draft = DecompositionDraft(
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
    )
    decomposed = env.step(
        State(initial, turn=0),
        Action(Verb.DECOMPOSE, 0),
        QueueGenerator(draft),
    ).next_state.graph
    transition = step(
        env,
        run_contract,
        State(decomposed, turn=0),
        Action(Verb.PROPOSE_SOLUTION, 1),
        SolutionDraft("x", "2"),
    )
    persisted = record(
        env, run_contract, transition, "episode:dirty", "problem-dirty"
    )

    with pytest.raises(ReplayValidationError, match="trusted problem state"):
        replay_canonical_run_lines(
            [line(build_run_header(env, run_contract)), line(persisted)],
            environment=env,
            initial_state_resolver=TrustedProblemResolver({
                "problem-dirty": State(initial, turn=0)
            }),
            collection_contract=run_contract,
        )


def test_strict_replay_keeps_teacher_bc_untrusted_without_a_receipt():
    env = environment(turn_limit=1)
    run_contract = collection(behavior_source="teacher")
    transition = step(
        env,
        run_contract,
        State(graph(env, 5), turn=0),
        Action(Verb.PROPOSE_RESULT, 0),
        ResultDraft(GoalObjective.SUM, "5"),
    )
    persisted = record(
        env, run_contract, transition, "episode:teacher", "problem-a"
    )

    with pytest.raises(ReplayValidationError, match="diagnostic receipt"):
        replay_canonical_run_lines(
            [line(build_run_header(env, run_contract)), line(persisted)],
            environment=env,
            initial_state_resolver=trusted_states(env, {"problem-a": 5}),
        )


def test_strict_replay_rejects_a_diagnostic_ref_without_a_sidecar():
    env = environment(turn_limit=1)
    run_contract = collection()
    transition = step(
        env,
        run_contract,
        State(graph(env, 5), turn=0),
        Action(Verb.PROPOSE_RESULT, 0),
        ResultDraft(GoalObjective.SUM, "5"),
    )
    transition = with_diagnostic_provenance(
        env, transition, "episode:learned", "a" * 32
    )
    persisted = record(
        env, run_contract, transition, "episode:learned", "problem-a"
    )

    with pytest.raises(ReplayValidationError, match="without its sidecar"):
        replay_canonical_run_lines(
            [line(build_run_header(env, run_contract)), line(persisted)],
            environment=env,
            initial_state_resolver=trusted_states(env, {"problem-a": 5}),
        )


def test_strict_replay_requires_a_receipt_for_hosted_learned_policy():
    env = environment(turn_limit=1)
    run_contract = collection(hosted=True)
    transition = step(
        env,
        run_contract,
        State(graph(env, 5), turn=0),
        Action(Verb.PROPOSE_RESULT, 0),
        ResultDraft(GoalObjective.SUM, "5"),
    )
    persisted = record(
        env, run_contract, transition, "episode:learned", "problem-a"
    )

    with pytest.raises(ReplayValidationError, match="hosted transition"):
        replay_canonical_run_lines(
            [line(build_run_header(env, run_contract)), line(persisted)],
            environment=env,
            initial_state_resolver=trusted_states(env, {"problem-a": 5}),
        )


def test_strict_replay_exact_checks_a_non_bc_decision_receipt(tmp_path):
    env = environment(turn_limit=1)
    run_contract = collection(hosted=True)
    transition = step(
        env,
        run_contract,
        State(graph(env, 5), turn=0),
        Action(Verb.PROPOSE_RESULT, 0),
        ResultDraft(GoalObjective.SUM, "5"),
    )
    diagnostics, receipt_id = decision_sidecar(
        env,
        run_contract,
        transition,
        receipt_episode_key="episode:other",
        path=tmp_path / "mismatched-decision.jsonl",
    )
    transition = with_diagnostic_provenance(
        env, transition, "episode:learned", receipt_id
    )
    persisted = record(
        env, run_contract, transition, "episode:learned", "problem-a"
    )

    with pytest.raises(ReplayValidationError, match="episode_key differs"):
        replay_canonical_run_lines(
            [line(build_run_header(env, run_contract)), line(persisted)],
            environment=env,
            initial_state_resolver=trusted_states(env, {"problem-a": 5}),
            diagnostic_index=diagnostics,
        )


def test_strict_replay_accepts_exact_hosted_learned_policy_receipt(tmp_path):
    env = environment(turn_limit=1)
    run_contract = collection(hosted=True)
    transition = step(
        env,
        run_contract,
        State(graph(env, 5), turn=0),
        Action(Verb.PROPOSE_RESULT, 0),
        ResultDraft(GoalObjective.SUM, "5"),
    )
    diagnostics, receipt_id = decision_sidecar(
        env,
        run_contract,
        transition,
        receipt_episode_key="episode:learned",
        path=tmp_path / "linked-learned-policy.jsonl",
    )
    transition = with_diagnostic_provenance(
        env, transition, "episode:learned", receipt_id
    )
    persisted = record(
        env, run_contract, transition, "episode:learned", "problem-a"
    )

    replayed = replay_canonical_run_lines(
        [line(build_run_header(env, run_contract)), line(persisted)],
        environment=env,
        initial_state_resolver=trusted_states(env, {"problem-a": 5}),
        diagnostic_index=diagnostics,
    )

    assert not replayed.decisions[0].behavior.bc_eligible
    assert replayed.episodes[0].terminal


def test_strict_replay_rejects_generator_payload_effect_mismatch(tmp_path):
    env = environment(turn_limit=1)
    run_contract = collection(hosted=True)
    transition = step(
        env,
        run_contract,
        State(graph(env, 5), turn=0),
        Action(Verb.PROPOSE_RESULT, 0),
        ResultDraft(GoalObjective.SUM, "5"),
    )
    diagnostics, receipt_id = decision_sidecar(
        env,
        run_contract,
        transition,
        receipt_episode_key="episode:learned",
        path=tmp_path / "generator-effect-mismatch.jsonl",
        generator_payload={"value": "6"},
    )
    transition = with_diagnostic_provenance(
        env, transition, "episode:learned", receipt_id
    )
    persisted = record(
        env, run_contract, transition, "episode:learned", "problem-a"
    )

    with pytest.raises(ReplayValidationError, match="raw payload differs"):
        replay_canonical_run_lines(
            [line(build_run_header(env, run_contract)), line(persisted)],
            environment=env,
            initial_state_resolver=trusted_states(env, {"problem-a": 5}),
            diagnostic_index=diagnostics,
        )


def test_strict_replay_links_raw_generator_payload_to_canonical_result(
    tmp_path,
):
    env = environment(turn_limit=1)
    run_contract = collection(hosted=True)
    initial = State(graph(env, "1/2"), turn=0)
    transition = step(
        env,
        run_contract,
        initial,
        Action(Verb.PROPOSE_RESULT, 0),
        ResultDraft(GoalObjective.SUM, "2/4"),
    )
    assert transition.policy_draft == ResultDraft(
        GoalObjective.SUM, "1/2"
    )
    diagnostics, receipt_id = decision_sidecar(
        env,
        run_contract,
        transition,
        receipt_episode_key="episode:canonical-result",
        path=tmp_path / "canonical-result.jsonl",
        generator_payload={"value": "2/4"},
    )
    transition = with_diagnostic_provenance(
        env, transition, "episode:canonical-result", receipt_id
    )
    persisted = record(
        env,
        run_contract,
        transition,
        "episode:canonical-result",
        "problem-a",
    )

    replayed = replay_canonical_run_lines(
        [line(build_run_header(env, run_contract)), line(persisted)],
        environment=env,
        initial_state_resolver=TrustedProblemResolver({
            "problem-a": initial
        }),
        diagnostic_index=diagnostics,
    )

    assert replayed.decisions[0].transition.policy_draft == ResultDraft(
        GoalObjective.SUM, "1/2"
    )


@pytest.mark.parametrize(
    ("objective", "domain", "verb", "raw_draft", "raw_payload", "canonical"),
    [
        (
            GoalObjective.IDENTITY,
            "Reals",
            Verb.PROPOSE_SOLUTION,
            SolutionDraft("x", "2/4"),
            {"value": "2/4"},
            SolutionDraft("x", "1/2"),
        ),
        (
            GoalObjective.BRANCH,
            "x < 0",
            Verb.PROPOSE_REFUTATION,
            RefutationDraft(
                RefutationMode.UNIQUE_CANDIDATE_OUTSIDE_DOMAIN,
                "x",
                "2/4",
            ),
            {"candidate": "2/4"},
            RefutationDraft(
                RefutationMode.UNIQUE_CANDIDATE_OUTSIDE_DOMAIN,
                "x",
                "1/2",
            ),
        ),
    ],
)
def test_strict_replay_canonicalizes_other_proposal_drafts(
    tmp_path,
    objective,
    domain,
    verb,
    raw_draft,
    raw_payload,
    canonical,
):
    env = environment(turn_limit=1)
    run_contract = collection(hosted=True)
    initial = State(
        new_graph(
            "solve",
            GoalObligation("x", "Eq(x, 1/2)", domain, objective),
            generation_contract=env.generation_contract,
        ),
        turn=0,
    )
    transition = step(
        env,
        run_contract,
        initial,
        Action(verb, 0),
        raw_draft,
    )
    assert transition.policy_draft == canonical
    diagnostics, receipt_id = decision_sidecar(
        env,
        run_contract,
        transition,
        receipt_episode_key=f"episode:canonical-{verb.value}",
        path=tmp_path / f"canonical-{verb.value}.jsonl",
        generator_payload=raw_payload,
    )
    transition = with_diagnostic_provenance(
        env,
        transition,
        f"episode:canonical-{verb.value}",
        receipt_id,
    )
    persisted = record(
        env,
        run_contract,
        transition,
        f"episode:canonical-{verb.value}",
        "problem-a",
    )

    replayed = replay_canonical_run_lines(
        [line(build_run_header(env, run_contract)), line(persisted)],
        environment=env,
        initial_state_resolver=TrustedProblemResolver({
            "problem-a": initial
        }),
        diagnostic_index=diagnostics,
    )

    assert replayed.decisions[0].transition.policy_draft == canonical


def test_strict_replay_links_raw_decomposition_to_canonical_obligations(
    tmp_path,
):
    env = environment(turn_limit=1)
    run_contract = collection(hosted=True)
    initial = State(graph(env, 5), turn=0)
    raw_draft = DecompositionDraft(
        Connective.EXHAUSTIVE_CASE,
        ("case 1", "case 2"),
        (
            GoalObligation(
                "x", "Eq(x, 5)", "x < 10/2", GoalObjective.BRANCH
            ),
            GoalObligation(
                "x", "Eq(x, 5)", "x >= 10/2", GoalObjective.BRANCH
            ),
        ),
    )
    transition = step(
        env,
        run_contract,
        initial,
        Action(Verb.DECOMPOSE, 0),
        raw_draft,
    )
    assert transition.policy_draft != raw_draft
    diagnostics, receipt_id = decision_sidecar(
        env,
        run_contract,
        transition,
        receipt_episode_key="episode:canonical-decomposition",
        path=tmp_path / "canonical-decomposition.jsonl",
        generator_payload={
            "domains": ["x < 10/2", "x >= 10/2"]
        },
    )
    transition = with_diagnostic_provenance(
        env, transition, "episode:canonical-decomposition", receipt_id
    )
    persisted = record(
        env,
        run_contract,
        transition,
        "episode:canonical-decomposition",
        "problem-a",
    )

    replayed = replay_canonical_run_lines(
        [line(build_run_header(env, run_contract)), line(persisted)],
        environment=env,
        initial_state_resolver=TrustedProblemResolver({
            "problem-a": initial
        }),
        diagnostic_index=diagnostics,
    )

    assert (
        replayed.decisions[0].transition.policy_draft
        == transition.policy_draft
    )


def test_strict_replay_links_typed_payload_to_algebra_rejection(tmp_path):
    env = environment(turn_limit=1)
    run_contract = collection(hosted=True)
    initial = State(graph(env, 5), turn=0)
    raw_draft = ResultDraft(GoalObjective.SUM, "not symbolic")
    transition = step(
        env,
        run_contract,
        initial,
        Action(Verb.PROPOSE_RESULT, 0),
        raw_draft,
    )
    assert transition.outcome.reason is RejectionReason.ALGEBRA_REJECTED
    assert transition.policy_draft == raw_draft
    diagnostics, receipt_id = decision_sidecar(
        env,
        run_contract,
        transition,
        receipt_episode_key="episode:algebra-rejected",
        path=tmp_path / "algebra-rejected.jsonl",
        generator_payload={"value": "not symbolic"},
    )
    transition = with_diagnostic_provenance(
        env, transition, "episode:algebra-rejected", receipt_id
    )
    persisted = record(
        env,
        run_contract,
        transition,
        "episode:algebra-rejected",
        "problem-a",
    )

    replayed = replay_canonical_run_lines(
        [line(build_run_header(env, run_contract)), line(persisted)],
        environment=env,
        initial_state_resolver=TrustedProblemResolver({
            "problem-a": initial
        }),
        diagnostic_index=diagnostics,
    )

    assert (
        replayed.decisions[0].transition.outcome.reason
        is RejectionReason.ALGEBRA_REJECTED
    )


def test_strict_replay_cannot_promote_receipt_only_teacher_data_to_bc(
    tmp_path,
):
    env = environment(turn_limit=1)
    run_contract = collection(behavior_source="teacher", hosted=True)
    transition = step(
        env,
        run_contract,
        State(graph(env, 5), turn=0),
        Action(Verb.PROPOSE_RESULT, 0),
        ResultDraft(GoalObjective.SUM, "5"),
    )
    diagnostics, receipt_id = decision_sidecar(
        env,
        run_contract,
        transition,
        receipt_episode_key="episode:teacher",
        path=tmp_path / "receipt-only-teacher.jsonl",
    )
    transition = with_diagnostic_provenance(
        env, transition, "episode:teacher", receipt_id
    )
    persisted = record(
        env, run_contract, transition, "episode:teacher", "problem-a"
    )

    with pytest.raises(ReplayValidationError, match="fully linked model"):
        replay_canonical_run_lines(
            [line(build_run_header(env, run_contract)), line(persisted)],
            environment=env,
            initial_state_resolver=trusted_states(env, {"problem-a": 5}),
            diagnostic_index=diagnostics,
        )


def test_strict_replay_rejects_an_unlinked_accepted_controller_call(
    tmp_path,
):
    env = environment(turn_limit=1)
    run_contract = collection(behavior_source="teacher", hosted=True)
    transition = step(
        env,
        run_contract,
        State(graph(env, 5), turn=0),
        Action(Verb.PROPOSE_RESULT, 0),
        ResultDraft(GoalObjective.SUM, "5"),
    )
    diagnostics, receipt_id = decision_sidecar(
        env,
        run_contract,
        transition,
        receipt_episode_key="episode:teacher",
        path=tmp_path / "unlinked-controller.jsonl",
        controller_attempts=True,
        controller_payload={"verb": "decompose", "target": 0},
    )
    transition = with_diagnostic_provenance(
        env, transition, "episode:teacher", receipt_id
    )
    persisted = record(
        env, run_contract, transition, "episode:teacher", "problem-a"
    )

    with pytest.raises(ReplayValidationError, match="raw payload differs"):
        replay_canonical_run_lines(
            [line(build_run_header(env, run_contract)), line(persisted)],
            environment=env,
            initial_state_resolver=trusted_states(env, {"problem-a": 5}),
            diagnostic_index=diagnostics,
        )


def test_strict_replay_accepts_exactly_linked_hosted_bc(tmp_path):
    env = environment(turn_limit=1)
    run_contract = collection(behavior_source="teacher", hosted=True)
    transition = step(
        env,
        run_contract,
        State(graph(env, 5), turn=0),
        Action(Verb.PROPOSE_RESULT, 0),
        ResultDraft(GoalObjective.SUM, "5"),
    )
    diagnostics, receipt_id = decision_sidecar(
        env,
        run_contract,
        transition,
        receipt_episode_key="episode:teacher",
        path=tmp_path / "linked-controller.jsonl",
        controller_attempts=True,
        controller_payload={
            "verb": transition.action.verb.value,
            "target": transition.action.target,
        },
    )
    transition = with_diagnostic_provenance(
        env, transition, "episode:teacher", receipt_id
    )
    persisted = record(
        env, run_contract, transition, "episode:teacher", "problem-a"
    )

    replayed = replay_canonical_run_lines(
        [line(build_run_header(env, run_contract)), line(persisted)],
        environment=env,
        initial_state_resolver=trusted_states(env, {"problem-a": 5}),
        diagnostic_index=diagnostics,
    )

    assert replayed.decisions[0].behavior.bc_eligible
    assert replayed.episodes[0].terminal


@pytest.mark.parametrize("include_first", [False, True])
def test_strict_replay_rejects_empty_and_truncated_runs(include_first):
    env = environment()
    run_contract = collection()
    first, _ = two_turn_episode(
        env, run_contract, "episode:a", "problem-a", 5
    )
    lines = [line(build_run_header(env, run_contract))]
    if include_first:
        lines.append(line(first))
    expected = "truncated" if include_first else "no transitions"

    with pytest.raises(ReplayValidationError, match=expected):
        replay_canonical_run_lines(
            lines,
            environment=env,
            initial_state_resolver=trusted_states(env, {"problem-a": 5}),
            collection_contract=run_contract,
        )


def test_strict_replay_rejects_noncanonical_truncated_json_line():
    env = environment()
    run_contract = collection()

    with pytest.raises(ReplayValidationError, match="line 2"):
        replay_canonical_run_lines(
            [line(build_run_header(env, run_contract)), b'{"broken":true}'],
            environment=env,
            initial_state_resolver=trusted_states(env, {}),
            collection_contract=run_contract,
        )


def test_strict_replay_rejects_a_problem_relabelled_to_an_easier_obligation():
    env = environment(turn_limit=1)
    run_contract = collection()
    transition = step(
        env,
        run_contract,
        State(graph(env, 7), turn=0),
        Action(Verb.PROPOSE_RESULT, 0),
        ResultDraft(GoalObjective.SUM, "7"),
    )
    relabelled = record(
        env, run_contract, transition, "episode:forged", "problem-a"
    )

    with pytest.raises(ReplayValidationError, match="trusted problem state"):
        replay_canonical_run_lines(
            [line(build_run_header(env, run_contract)), line(relabelled)],
            environment=env,
            initial_state_resolver=trusted_states(env, {"problem-a": 5}),
        )


@pytest.mark.parametrize(
    "controller_changes",
    [
        {"controller_seed": 1},
        {"controller_temperature": 0.25},
        {"controller_max_tokens": 65},
        {
            "controller_seed": None,
            "controller_seed_policy": "none-v1",
        },
    ],
    ids=["seed", "temperature", "max-tokens", "seed-policy"],
)
def test_strict_replay_rejects_an_episode_spliced_under_another_collection(
    controller_changes,
):
    env = environment(turn_limit=1)
    first_contract = collection()
    second_contract = replace(first_contract, **controller_changes)
    transition = step(
        env,
        first_contract,
        State(graph(env, 5), turn=0),
        Action(Verb.PROPOSE_RESULT, 0),
        ResultDraft(GoalObjective.SUM, "5"),
    )
    spliced = record(
        env, first_contract, transition, "episode:spliced", "problem-a"
    )

    with pytest.raises(ReplayValidationError, match="collection digest"):
        replay_canonical_run_lines(
            [line(build_run_header(env, second_contract)), line(spliced)],
            environment=env,
            initial_state_resolver=trusted_states(env, {"problem-a": 5}),
        )


def test_strict_replay_rejects_a_problem_resolver_from_another_source():
    env = environment()
    run_contract = collection()
    resolver = replace(
        trusted_states(env, {}), problem_source_sha256="3" * 64
    )

    with pytest.raises(ReplayValidationError, match="differs from the header"):
        replay_canonical_run_lines(
            [line(build_run_header(env, run_contract))],
            environment=env,
            initial_state_resolver=resolver,
        )
