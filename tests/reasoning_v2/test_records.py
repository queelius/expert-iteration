from copy import deepcopy
from dataclasses import replace

import pytest

from verifier_anchored_rl.reasoning_v2 import (
    Action,
    BehaviorMetadata,
    BehaviorSource,
    BoundEnvironment,
    CanonicalRunWriter,
    CollectionContract,
    Connective,
    CONTROLLER_OUTPUT_CONTRACT_VERSION,
    ControllerChoice,
    DecompositionDraft,
    EpisodeRuntime,
    GoalObjective,
    GoalObligation,
    MAX_GENERATION_SEED,
    ModelIdentity,
    ModelTransportContract,
    RecordValidationError,
    ResultDraft,
    SolutionDraft,
    State,
    Verb,
    build_environment_contract,
    build_run_header,
    build_transition_record,
    canonical_json_bytes,
    canonical_json_sha256,
    canonical_record_bytes,
    new_graph,
    parse_canonical_json_line,
    validate_run_header,
    validate_transition_record,
)
from verifier_anchored_rl.reasoning_v2.lenses import (
    GENERATION_SCHEMA_VERSION,
    LENS_CONTRACT_VERSION,
)
from verifier_anchored_rl.reasoning_v2.math_adapter import (
    CaseworkMathVerifier,
)


GENERATOR_ID = ModelIdentity("test", "generator", "r1")
CONTROLLER_ID = ModelIdentity("test", "controller", "r1")
class QueueGenerator:
    generator_identity = GENERATOR_ID
    generator_implementation = (
        f"{__name__}.QueueGenerator"
    )
    generator_transport_contract = ModelTransportContract.local(
        generator_implementation
    )
    generator_transport_sha256 = generator_transport_contract.sha256
    generation_seed = None
    generation_retry_budget = 0
    lens_contract = LENS_CONTRACT_VERSION
    generation_schema = GENERATION_SCHEMA_VERSION
    generation_seed_policy = "none-v1"

    def __init__(self, env, *drafts):
        self.env = env
        self.drafts = list(drafts)

    @property
    def environment_contract_sha256(self):
        return self.env.sha256

    @property
    def generation_temperature(self):
        return self.env.contract.generation_temperature

    @property
    def generation_max_tokens(self):
        return self.env.contract.generation_max_tokens

    def generate(self, state, action):
        return self.drafts.pop(0)


class QueueController:
    controller_identity = CONTROLLER_ID
    controller_implementation = f"{__name__}.QueueController"
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


class RecordingDecisionRecorder:
    def __init__(self, env, expected_collection):
        self.environment_contract_sha256 = env.sha256
        self.collection_contract_sha256 = expected_collection.sha256
        self.calls = []

    def record_decision(self, **kwargs):
        self.calls.append(kwargs)
        return "1" * 32


def environment(turn_limit=6):
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


def collection():
    return CollectionContract(
        generator=GENERATOR_ID,
        controller=CONTROLLER_ID,
        controller_implementation=QueueController.controller_implementation,
        controller_transport=QueueController.controller_transport_contract,
        controller_seed=None,
        behavior_source=BehaviorSource.TEACHER.value,
        controller_fallback_policy="first-legal-pair-v1",
        generator_implementation=QueueGenerator.generator_implementation,
        generator_transport=QueueGenerator.generator_transport_contract,
        generation_seed=None,
        policy_input_contract="teacher-presentation-v2",
        problem_source_contract="test-problems-v1",
        problem_source_sha256="2" * 64,
        problem_adapter_implementation="tests.TestProblemAdapter",
    )


def graph(env, *, equation="Eq(x, 5)", objective=GoalObjective.SUM):
    return new_graph(
        "solve",
        GoalObligation("x", equation, "Reals", objective),
        generation_contract=env.generation_contract,
    )


def direct_transition(env, value="5"):
    episode = EpisodeRuntime(
        env,
        collection(),
        QueueGenerator(env, ResultDraft(GoalObjective.SUM, value)),
        State(graph(env), turn=0),
        QueueController(env, Action(Verb.PROPOSE_RESULT, 0)),
    )
    _, transition = episode.step()
    return transition


def record_for(env, transition, *, episode_key="case:0"):
    return build_transition_record(
        transition,
        environment=env,
        collection_contract=collection(),
        episode_key=episode_key,
        problem_id="problem-0",
    )


def test_run_header_round_trips_exact_contracts_and_digest():
    env = environment()
    expected_collection = collection()
    header = build_run_header(env, expected_collection)

    assert set(header) == {
        "schema_version",
        "environment_contract",
        "collection_contract",
        "environment_contract_sha256",
        "collection_contract_sha256",
    }
    parsed_environment, parsed_collection = validate_run_header(
        header,
        environment=env,
        collection_contract=expected_collection,
    )
    assert parsed_environment == env.contract
    assert parsed_collection == expected_collection
    assert canonical_record_bytes(header, environment=env) == (
        canonical_json_bytes(header)
    )


def test_run_header_rejects_contract_tampering_and_unknown_keys():
    env = environment()
    header = build_run_header(env, collection())
    tampered = deepcopy(header)
    tampered["environment_contract"]["generation_temperature"] = 0.5

    with pytest.raises(RecordValidationError, match="digest"):
        validate_run_header(tampered)

    unknown = deepcopy(header)
    unknown["extra"] = None
    with pytest.raises(RecordValidationError, match="keys"):
        validate_run_header(unknown)


def test_collection_contract_pins_controller_sampling_and_seed_range():
    env = environment()
    expected = collection()
    changed = replace(expected, controller_temperature=0.25)

    assert changed.sha256 != expected.sha256
    with pytest.raises(ValueError, match="none-v1 controller"):
        replace(expected, controller_seed=1)

    overflow = replace(
        expected,
        controller_seed_policy="turn-and-retry-offset-v1",
        controller_seed=MAX_GENERATION_SEED - env.contract.turn_limit + 2,
    )
    with pytest.raises(ValueError, match="effective controller seed"):
        build_run_header(env, overflow)


def test_transition_record_contains_exact_learned_effect_and_terminal_state():
    env = environment()
    transition = direct_transition(env)
    record = record_for(env, transition)

    assert set(record) == {
        "schema_version",
        "environment_contract_sha256",
        "collection_contract_sha256",
        "episode_key",
        "problem_id",
        "turn",
        "observation",
        "action",
        "learned_effect",
        "outcome",
        "closure_events",
        "reward",
        "next_observation",
        "terminal",
        "behavior",
        "diagnostic_ref",
    }
    assert record["learned_effect"] == {
        "kind": "result",
        "claim_id": 1,
        "payload": {"objective": "sum", "value": "5"},
        "premises": [],
    }
    assert record["terminal"] is True
    assert record["reward"] == 1
    assert record["next_observation"]["legal_pairs"] == []
    validate_transition_record(
        record,
        environment=env,
        collection_contract=collection(),
        expected_transition=transition,
    )


def test_terminal_aggregate_keeps_the_compacted_away_policy_effect():
    env = environment()
    parent = new_graph(
        "sum",
        GoalObligation(
            "x",
            "Eq((x - 2)*(x - 3), 0)",
            "Reals",
            GoalObjective.SUM,
        ),
        generation_contract=env.generation_contract,
    )
    generator = QueueGenerator(
        env,
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
        SolutionDraft("x", "3"),
    )
    episode = EpisodeRuntime(
        env,
        collection(),
        generator,
        State(parent, turn=0),
        QueueController(
            env,
            Action(Verb.DECOMPOSE, 0),
            Action(Verb.PROPOSE_SOLUTION, 1),
            Action(Verb.PROPOSE_SOLUTION, 2),
        ),
    )
    episode, first = episode.step()
    episode, second = episode.step()
    _, final = episode.step()
    record = record_for(env, final)

    assert final.terminal
    assert record["next_observation"]["claims"] == []
    assert record["learned_effect"] == {
        "kind": "solution",
        "claim_id": 4,
        "payload": {"variable": "x", "value": "3"},
        "premises": [],
    }
    validate_transition_record(
        record,
        environment=env,
        collection_contract=collection(),
        expected_transition=final,
    )

    forged = deepcopy(record)
    forged["learned_effect"]["payload"]["value"] = "4"
    with pytest.raises(RecordValidationError, match="learned_effect"):
        validate_transition_record(
            forged,
            environment=env,
            collection_contract=collection(),
            expected_transition=final,
        )


def test_canonical_writer_emits_header_first_and_exact_episode_chain(tmp_path):
    env = environment(turn_limit=3)
    episode = EpisodeRuntime(
        env,
        collection(),
        QueueGenerator(
            env,
            ResultDraft(GoalObjective.SUM, "6"),
            ResultDraft(GoalObjective.SUM, "5"),
        ),
        State(graph(env), turn=0),
        QueueController(
            env,
            Action(Verb.PROPOSE_RESULT, 0),
            Action(Verb.PROPOSE_RESULT, 0),
        ),
    )
    episode, first = episode.step()
    assert not first.terminal
    _, second = episode.step()
    first_record = record_for(env, first)
    second_record = record_for(env, second)
    header = build_run_header(env, collection())
    path = tmp_path / "run.jsonl"

    with CanonicalRunWriter(
        path,
        header,
        environment=env,
        collection_contract=collection(),
    ) as writer:
        writer.append(first_record, transition=first)
        writer.append(second_record, transition=second)

    lines = path.read_bytes().splitlines(keepends=True)
    assert len(lines) == 3
    assert parse_canonical_json_line(lines[0]) == header
    assert parse_canonical_json_line(lines[1]) == first_record
    assert parse_canonical_json_line(lines[2]) == second_record


def test_writer_rejects_mid_episode_start(tmp_path):
    env = environment(turn_limit=3)
    episode = EpisodeRuntime(
        env,
        collection(),
        QueueGenerator(
            env,
            ResultDraft(GoalObjective.SUM, "6"),
            ResultDraft(GoalObjective.SUM, "5"),
        ),
        State(graph(env), turn=0),
        QueueController(
            env,
            Action(Verb.PROPOSE_RESULT, 0),
            Action(Verb.PROPOSE_RESULT, 0),
        ),
    )
    episode, first = episode.step()
    _, second = episode.step()
    path = tmp_path / "mid.jsonl"

    with CanonicalRunWriter(
        path,
        build_run_header(env, collection()),
        environment=env,
        collection_contract=collection(),
    ) as writer:
        with pytest.raises(RecordValidationError, match="turn zero"):
            writer.append(record_for(env, second), transition=second)


def test_writer_chain_uses_the_persisted_snapshot(tmp_path):
    env = environment(turn_limit=3)
    episode = EpisodeRuntime(
        env,
        collection(),
        QueueGenerator(
            env,
            ResultDraft(GoalObjective.SUM, "6"),
            ResultDraft(GoalObjective.SUM, "5"),
        ),
        State(graph(env), turn=0),
        QueueController(
            env,
            Action(Verb.PROPOSE_RESULT, 0),
            Action(Verb.PROPOSE_RESULT, 0),
        ),
    )
    episode, first = episode.step()
    _, second = episode.step()
    first_record = record_for(env, first)
    second_record = record_for(env, second)
    path = tmp_path / "snapshot.jsonl"

    with CanonicalRunWriter(
        path,
        build_run_header(env, collection()),
        environment=env,
        collection_contract=collection(),
    ) as writer:
        writer.append(first_record, transition=first)
        first_record["next_observation"]["turn"] = 999
        writer.append(second_record, transition=second)

    persisted = path.read_bytes().splitlines(keepends=True)
    assert parse_canonical_json_line(persisted[1])["next_observation"][
        "turn"
    ] == 1


def test_bound_controller_owns_action_and_behavior_labels():
    env = environment()
    episode = EpisodeRuntime(
        env,
        collection(),
        QueueGenerator(
            env,
            ResultDraft(GoalObjective.SUM, "5"),
        ),
        State(graph(env), turn=0),
        QueueController(env, Action(Verb.PROPOSE_RESULT, 0)),
    )

    with pytest.raises(TypeError, match="selects its own action"):
        episode.step(Action(Verb.PROPOSE_RESULT, 0))


def test_bound_runtime_owns_composite_decision_receipt_creation():
    env = environment()
    expected_collection = collection()
    state = State(graph(env), turn=0)
    action = Action(Verb.PROPOSE_RESULT, 0)
    recorder = RecordingDecisionRecorder(env, expected_collection)
    episode = EpisodeRuntime(
        env,
        expected_collection,
        QueueGenerator(env, ResultDraft(GoalObjective.SUM, "5")),
        state,
        QueueController(env, action),
        episode_key="case:0",
        decision_recorder=recorder,
    )

    _, transition = episode.step()

    assert recorder.calls == [{
        "episode_key": "case:0",
        "turn": 0,
        "observation_sha256": canonical_json_sha256(env.observe(state)),
        "source": BehaviorSource.TEACHER,
        "proposed_action": action,
        "selected_action": action,
        "fallback": False,
    }]
    assert transition.decision_provenance.diagnostic_record_id == "1" * 32
    assert record_for(env, transition)["diagnostic_ref"] == {
        "schema_version": "controller-diagnostic-ref-v2",
        "record_id": "1" * 32,
    }
    with pytest.raises(RecordValidationError, match="episode_key"):
        record_for(env, transition, episode_key="case:other")

    forged = replace(
        transition,
        decision_provenance=replace(
            transition.decision_provenance,
            observation_sha256="f" * 64,
        ),
    )
    with pytest.raises(RecordValidationError, match="observation"):
        record_for(env, forged)


def test_decision_recorder_needs_episode_and_exact_contract_binding():
    env = environment()
    expected_collection = collection()
    state = State(graph(env), turn=0)
    controller = QueueController(env, Action(Verb.PROPOSE_RESULT, 0))
    recorder = RecordingDecisionRecorder(env, expected_collection)

    with pytest.raises(ValueError, match="episode_key"):
        EpisodeRuntime(
            env,
            expected_collection,
            QueueGenerator(env, ResultDraft(GoalObjective.SUM, "5")),
            state,
            controller,
            decision_recorder=recorder,
        )

    recorder.collection_contract_sha256 = "f" * 64
    with pytest.raises(ValueError, match="collection_contract_sha256"):
        EpisodeRuntime(
            env,
            expected_collection,
            QueueGenerator(env, ResultDraft(GoalObjective.SUM, "5")),
            state,
            controller,
            episode_key="case:0",
            decision_recorder=recorder,
        )


def test_teacher_fallback_is_runtime_owned_and_not_bc_eligible():
    env = environment()
    equation = "Eq((x - 2)*(x - 3), 0)"
    episode = EpisodeRuntime(
        env,
        collection(),
        QueueGenerator(
            env,
            DecompositionDraft(
                Connective.EXHAUSTIVE_CASE,
                ("low", "high"),
                (
                    GoalObligation(
                        "x",
                        equation,
                        "x < 5/2",
                        GoalObjective.BRANCH,
                    ),
                    GoalObligation(
                        "x",
                        equation,
                        "x >= 5/2",
                        GoalObjective.BRANCH,
                    ),
                ),
            ),
        ),
        State(
            graph(env, equation=equation),
            turn=0,
        ),
        QueueController(env, None),
    )

    _, transition = episode.step()
    record = record_for(env, transition)

    assert transition.action == Action(Verb.DECOMPOSE, 0)
    assert record["behavior"] == {
        "source": "teacher",
        "bc_eligible": False,
        "fallback": True,
    }


def test_semantic_only_transition_is_not_collectible():
    env = environment()
    transition = env.step(
        State(graph(env), turn=0),
        Action(Verb.PROPOSE_RESULT, 0),
        QueueGenerator(
            env,
            ResultDraft(GoalObjective.SUM, "5"),
        ),
    )

    with pytest.raises(RecordValidationError, match="collection runtime"):
        record_for(env, transition)


@pytest.mark.parametrize(
    "line",
    [
        b'{"a":1, "b":2}\n',
        b'{"a":1,"a":2}\n',
        b'{"a":NaN}\n',
        b'{}',
        b"\xef\xbb\xbf{}\n",
    ],
)
def test_canonical_jsonl_parser_rejects_noncanonical_lines(line):
    with pytest.raises(RecordValidationError):
        parse_canonical_json_line(line)


def test_behavior_metadata_has_one_mechanical_bc_rule():
    assert BehaviorMetadata(
        BehaviorSource.TEACHER, False
    ).to_record()["bc_eligible"]
    with pytest.raises(ValueError, match="fallback"):
        BehaviorMetadata(BehaviorSource.LEARNED_POLICY, True)
