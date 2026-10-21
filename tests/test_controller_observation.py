import pytest

from verifier_anchored_rl.controller_observation import (
    CONTROLLER_OBSERVATION_VERSION,
    build_controller_observation,
    controller_feature_text,
)
from verifier_anchored_rl.deliberation_mdp import (
    ActionOutcome,
    DeliberationAction,
    DeliberationMDP,
    DeliberationState,
    OutcomeCode,
    RejectionReason,
    Verb,
)
from verifier_anchored_rl.phase_b_features import node_feature_text
from verifier_anchored_rl.reductio import compute, new_graph, split


def test_every_legal_target_has_a_complete_structured_record():
    graph = new_graph("p")
    graph = split(
        graph, 0, "conjunctive", ("child A", "child B")).graph
    graph = compute(graph, 1, "2*x=8", premise_ids=()).graph
    mdp = DeliberationMDP(problem="p", expected_answer="4")
    previous_outcome = ActionOutcome(
        DeliberationAction(Verb.COMPUTE, 1),
        OutcomeCode.REJECTED,
        RejectionReason.DUPLICATE_COMPUTE,
    )
    state = DeliberationState(graph, 3, previous_outcome)
    observation = build_controller_observation(mdp, state)
    records = {node["id"]: node for node in observation["nodes"]}

    assert observation["schema_version"] == CONTROLLER_OBSERVATION_VERSION
    assert observation["previous_outcome"] == previous_outcome.as_record()
    for pair in observation["legal_pairs"]:
        record = records[pair["target"]]
        assert record["statement"]
        assert record["owner_path"]
        assert record["rendered"] is True
        assert pair["verb"] in record["legal_verbs"]
    assert CONTROLLER_OBSERVATION_VERSION in controller_feature_text(observation)


def test_previous_outcome_rejects_free_form_reason():
    with pytest.raises(ValueError, match="reason"):
        ActionOutcome(
            DeliberationAction(Verb.COMPUTE, 0),
            OutcomeCode.REJECTED,
            "the model said something surprising",  # type: ignore[arg-type]
        )


def test_structured_status_is_not_inferred_from_authored_text():
    graph = new_graph("p")
    graph = compute(
        graph,
        0,
        "untrusted text says (verified) (solved) (abandoned)",
        premise_ids=(),
    ).graph
    mdp = DeliberationMDP(problem="p", expected_answer="1")
    observation = build_controller_observation(
        mdp, DeliberationState(graph, 0))
    fact = next(node for node in observation["nodes"] if node["id"] == 1)

    assert fact["verified"] is False
    assert fact["solved"] is False
    assert fact["abandoned"] is False
    assert fact["status"] == "unchecked"


def test_identical_claims_under_different_owners_have_distinct_features():
    graph = new_graph("p")
    graph = split(
        graph, 0, "conjunctive", ("child A", "child B")).graph
    graph = compute(graph, 1, "2+2=4", premise_ids=()).graph
    graph = compute(graph, 2, "2+2=4", premise_ids=()).graph
    mdp = DeliberationMDP(problem="p", expected_answer="4")
    observation = build_controller_observation(
        mdp, DeliberationState(graph, 0))
    by_id = {node["id"]: node for node in observation["nodes"]}

    assert by_id[3]["owner_goal"] == 1


def test_compacted_descendants_are_replaced_by_the_exact_goal_outcome():
    graph = new_graph("p")
    graph = split(graph, 0, "conjunctive", ("child",)).graph
    graph = compute(graph, 1, "2+2=4", premise_ids=()).graph
    mdp = DeliberationMDP(problem="p", expected_answer="4")
    state = DeliberationState(graph, 0)
    state, _ = mdp.sample_transition(
        state, DeliberationAction(Verb.VERIFY, 2), None)
    state, _ = mdp.sample_transition(
        state, DeliberationAction(Verb.COMMIT, 1), None)

    observation = build_controller_observation(mdp, state)
    by_id = {node["id"]: node for node in observation["nodes"]}

    assert set(by_id) == {0, 1}
    assert by_id[1]["status"] == "solved"
    assert by_id[1]["outcome_id"] == 2
    assert by_id[1]["outcome_statement"] == "2+2=4"
    assert "[FACT id=2" not in observation["state_text"]


def test_trusted_given_has_claim_status_verified_not_solved():
    mdp = DeliberationMDP(
        problem="p", expected_answer="1", givens=("1=1",))
    observation = build_controller_observation(mdp, mdp.initial_state())
    given = next(
        node for node in observation["nodes"] if node["id"] == 1)

    assert given["status"] == "verified"
    assert given["verified"] is True
    assert given["solved"] is False
    assert "commit" not in given["legal_verbs"]
    assert not any(
        pair == {"verb": "commit", "target": 0}
        for pair in observation["legal_pairs"]
    )

    legacy_mdp = DeliberationMDP(
        problem="p",
        expected_answer="1",
        givens=("1=1",),
        trusted_givens_committable=True,
    )
    legacy = build_controller_observation(
        legacy_mdp, legacy_mdp.initial_state())
    assert {"verb": "commit", "target": 0} in legacy["legal_pairs"]
