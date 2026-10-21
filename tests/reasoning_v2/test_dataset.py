from dataclasses import dataclass, replace

import pytest

from verifier_anchored_rl.reasoning_v2.contracts import (
    CollectionContract,
    ModelIdentity,
    ModelTransportContract,
    build_environment_contract,
    canonical_json_bytes,
)
from verifier_anchored_rl.reasoning_v2.dataset import (
    DATASET_SCHEMA_VERSION,
    DatasetValidationError,
    PhaseBDataset,
    build_grouped_split,
    convert_replayed_run,
)
from verifier_anchored_rl.reasoning_v2.lenses import (
    GENERATION_SCHEMA_VERSION,
    LENS_CONTRACT_VERSION,
)
from verifier_anchored_rl.reasoning_v2.math_adapter import CaseworkMathVerifier
from verifier_anchored_rl.reasoning_v2.model import (
    Action,
    GoalObjective,
    GoalObligation,
    State,
    Verb,
    new_graph,
)
from verifier_anchored_rl.reasoning_v2.provenance import (
    BehaviorSource,
    DecisionProvenance,
)
from verifier_anchored_rl.reasoning_v2.records import (
    BehaviorMetadata,
    build_run_header,
    build_transition_record,
)
from verifier_anchored_rl.reasoning_v2.replay import (
    ReplayedDecision,
    ReplayedEpisode,
    ReplayedRun,
    replay_canonical_run_lines,
)
from verifier_anchored_rl.reasoning_v2.runtime import BoundEnvironment
from verifier_anchored_rl.reasoning_v2.transition import PolicyGenerationError
from verifier_anchored_rl.reasoning_v2.actions import ResultDraft


class FailingGenerator:
    def generate(self, state, action):
        raise PolicyGenerationError("provider failed")


class FixedGenerator:
    def __init__(self, draft):
        self.draft = draft

    def generate(self, state, action):
        return self.draft


def environment(turn_limit=2):
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
        generator=ModelIdentity("test", "generator", "r1"),
        controller=ModelIdentity("test", "controller", "r1"),
        controller_implementation="tests.LearnedController",
        controller_transport=ModelTransportContract.local(
            "tests.LearnedController"
        ),
        controller_seed=0,
        controller_temperature=0.0,
        controller_max_tokens=64,
        controller_seed_policy="turn-and-retry-offset-v1",
        behavior_source="learned_policy",
        controller_fallback_policy="forbidden-v1",
        generator_implementation="tests.FixedGenerator",
        generator_transport=ModelTransportContract.local(
            "tests.FixedGenerator"
        ),
        generation_seed=None,
        policy_input_contract="typed-controller-presentation-v2",
        problem_source_contract="test-problems-v1",
        problem_source_sha256="2" * 64,
        problem_adapter_implementation="tests.TestProblemAdapter",
    )


def initial_state(env, problem_id):
    value = "2" if problem_id == "p1" else "3"
    graph = new_graph(
        f"solve {problem_id}",
        GoalObligation(
            "x", f"Eq(x, {value})", "Reals", GoalObjective.SUM
        ),
        (f"trusted {problem_id}",),
        generation_contract=env.generation_contract,
    )
    return State(graph, turn=0)


def replayed_episode(env, run_contract, episode_key, problem_id):
    behavior = BehaviorMetadata(BehaviorSource.LEARNED_POLICY, fallback=False)
    state = initial_state(env, problem_id)
    action = Action(Verb.PROPOSE_RESULT, 0)
    first = env.step(state, action, FailingGenerator())
    value = "2" if problem_id == "p1" else "3"
    second = env.step(
        first.next_state,
        action,
        FixedGenerator(ResultDraft(GoalObjective.SUM, value)),
    )
    decisions = []
    for transition in (first, second):
        bound = replace(
            transition,
            collection_contract_sha256=run_contract.sha256,
            decision_provenance=DecisionProvenance(
                source=BehaviorSource.LEARNED_POLICY,
                bc_eligible=False,
                fallback=False,
                diagnostic_record_id=None,
            ),
        )
        decisions.append(ReplayedDecision(bound, behavior, None))
    return ReplayedEpisode(episode_key, problem_id, tuple(decisions))


@dataclass(frozen=True)
class Resolver:
    states: dict[str, State]
    problem_source_contract: str = "test-problems-v1"
    problem_source_sha256: str = "2" * 64
    problem_adapter_implementation: str = "tests.TestProblemAdapter"

    def __call__(self, problem_id):
        return self.states[problem_id]


def replayed_run(env, problem_ids=("p1", "p2")):
    run_contract = collection()
    episodes = tuple(
        replayed_episode(env, run_contract, f"episode-{index}", problem_id)
        for index, problem_id in enumerate(problem_ids)
    )
    lines = [canonical_json_bytes(build_run_header(env, run_contract)) + b"\n"]
    for episode in episodes:
        for decision in episode.decisions:
            record = build_transition_record(
                decision.transition,
                environment=env,
                collection_contract=run_contract,
                episode_key=episode.episode_key,
                problem_id=episode.problem_id,
            )
            lines.append(canonical_json_bytes(record) + b"\n")
    resolver = Resolver({
        problem_id: initial_state(env, problem_id)
        for problem_id in set(problem_ids)
    })
    return replay_canonical_run_lines(
        lines,
        environment=env,
        initial_state_resolver=resolver,
        collection_contract=run_contract,
    )


def test_conversion_computes_returns_and_retains_all_provenance_rows():
    env = environment()
    dataset = convert_replayed_run(replayed_run(env), env, 0.5)

    assert dataset.schema == DATASET_SCHEMA_VERSION
    assert [item.reward for item in dataset.examples[:2]] == [0, 1]
    assert [item.mc_return for item in dataset.examples[:2]] == [0.5, 1.0]
    assert not any(item.bc_eligible for item in dataset.examples)
    assert len(dataset.examples) == 4
    assert not dataset.examples[0].behavior.bc_eligible
    assert dataset.examples[0].reward == 0

    teacher_failure = replace(
        dataset.examples[0],
        behavior=BehaviorMetadata(BehaviorSource.TEACHER, fallback=False),
    )
    assert teacher_failure.bc_eligible
    assert teacher_failure.reward == 0


def test_features_masks_and_action_indices_remain_exactly_aligned():
    env = environment()
    example = convert_replayed_run(replayed_run(env), env, 1.0).examples[0]

    assert example.current_features.policy_verb_order[
        example.chosen_verb_index
    ] == example.action.verb.value
    assert example.current_features.target_ids[
        example.chosen_target_index
    ] == example.action.target
    assert example.current_features.legal_pair_mask[
        example.chosen_verb_index
    ][example.chosen_target_index]
    assert example.to_record()["observation"]["legal_pairs"] == [
        {"verb": verb, "target": target}
        for verb, target in example.current_features.legal_pairs
    ]
    assert example.to_record()["next_features"]["legal_pair_mask"] == [
        list(row) for row in example.next_features.legal_pair_mask
    ]


def test_observation_snapshots_are_immutable_and_digest_is_value_sensitive():
    env = environment()
    dataset = convert_replayed_run(replayed_run(env), env, 1.0)
    before = dataset.dataset_sha256
    snapshot = dataset.examples[0].observation
    snapshot["goals"][0]["statement"] = "mutated"

    assert dataset.examples[0].observation["goals"][0]["statement"] == "solve p1"
    assert dataset.dataset_sha256 == before

    changed = convert_replayed_run(replayed_run(env, ("p2", "p1")), env, 1.0)
    assert changed.dataset_sha256 != before


def test_conversion_rejects_raw_contract_drift_and_incomplete_order():
    env = environment()
    run = replayed_run(env)
    with pytest.raises(TypeError, match="ReplayedRun"):
        convert_replayed_run({"records": []}, env, 1.0)

    other = environment(turn_limit=3)
    with pytest.raises(DatasetValidationError, match="environment contract"):
        convert_replayed_run(run, other, 1.0)

    truncated = replace(
        run,
        episodes=(replace(run.episodes[0], decisions=run.episodes[0].decisions[:1]),),
    )
    with pytest.raises(Exception, match="strict canonical replay"):
        convert_replayed_run(truncated, env, 1.0)

    reversed_rows = replace(
        run,
        episodes=(replace(run.episodes[0], decisions=tuple(reversed(run.episodes[0].decisions))),),
    )
    with pytest.raises(Exception, match="strict canonical replay"):
        convert_replayed_run(reversed_rows, env, 1.0)


@pytest.mark.parametrize("gamma", [0, 1, -0.0, 1.01, float("nan")])
def test_gamma_contract_rejects_nonexact_values(gamma):
    env = environment()
    with pytest.raises(DatasetValidationError, match="gamma"):
        convert_replayed_run(replayed_run(env), env, gamma)


def test_grouped_split_is_deterministic_disjoint_and_digest_bound():
    env = environment()
    dataset = convert_replayed_run(
        replayed_run(env, ("p1", "p2", "p1", "p2")), env, 1.0
    )
    first = build_grouped_split(dataset, 0.5, 7)
    second = build_grouped_split(dataset, 0.5, 7)

    assert first.split_sha256 == second.split_sha256
    assert first.assignments == second.assignments
    train = {item.problem_id for item in first.train_examples}
    validation = {item.problem_id for item in first.validation_examples}
    assert train
    assert validation
    assert train.isdisjoint(validation)
    assert len(first.train_examples) + len(first.validation_examples) == len(
        dataset.examples
    )

    different_seed = build_grouped_split(dataset, 0.5, 8)
    assert different_seed.split_sha256 != first.split_sha256
    assert dataset.checkpoint_contract(first) == {
        "schema": "phase-b-v2",
        "dataset_sha256": dataset.dataset_sha256,
        "split_sha256": first.split_sha256,
    }

    copied = replace(first, seed=9)
    with pytest.raises(DatasetValidationError, match="grouped split"):
        dataset.checkpoint_contract(copied)


def test_direct_dataset_construction_is_not_replay_trusted():
    env = environment()
    trusted = convert_replayed_run(replayed_run(env), env, 1.0)
    forged = PhaseBDataset(
        environment_contract_sha256=trusted.environment_contract_sha256,
        collection_contract_sha256=trusted.collection_contract_sha256,
        feature_contract_sha256=trusted.feature_contract_sha256,
        gamma=trusted.gamma,
        examples=trusted.examples,
    )

    with pytest.raises(DatasetValidationError, match="trusted replay"):
        forged.validate_trusted()
    with pytest.raises(DatasetValidationError, match="trusted replay"):
        build_grouped_split(forged, 0.5, 0)
