from __future__ import annotations

from copy import deepcopy
from inspect import signature

import pytest

from verifier_anchored_rl.reasoning_v2 import (
    Action,
    AggregateDraft,
    AggregatePayload,
    ClaimKind,
    Connective,
    DecompositionDraft,
    DerivationDraft,
    GoalObjective,
    GoalObligation,
    MockRule,
    Ok,
    OutcomeCode,
    Reducer,
    RefutationMode,
    RefutationPayload,
    Relation,
    ResultDraft,
    ResultPayload,
    SolutionDraft,
    SolutionPayload,
    State,
    TableDrivenVerifier,
    TransitionOutcome,
    Verb,
    aggregate as _aggregate,
    commit_claim,
    decompose as _decompose,
    derive as _derive,
    get,
    new_graph as _new_graph,
    propose_solution as _propose_solution,
    propose_result as _propose_result,
    prune,
    verify_claim,
)
from verifier_anchored_rl.reasoning_v2.model import DerivationPayload
from verifier_anchored_rl.reasoning_v2.observation import (
    CONTROLLER_OBSERVATION_VERSION,
    ObservationContext,
    ObservationValidationError,
    build_observation,
    canonical_observation_bytes,
    observation_sha256,
    serialize_payload,
    validate_observation,
)


CONTRACT_DIGEST = "a" * 64
CONTEXT = ObservationContext(
    turn_limit=16,
    environment_contract_sha256=CONTRACT_DIGEST,
)
GENERATION = TableDrivenVerifier(())


def new_graph(problem, obligation, givens=()):
    return _new_graph(
        problem,
        obligation,
        givens,
        generation_contract=GENERATION,
    )


def decompose(graph, goal_id, draft):
    return _decompose(graph, goal_id, draft, GENERATION)


def propose_solution(graph, goal_id, draft):
    return _propose_solution(graph, goal_id, draft, GENERATION)


def propose_result(graph, goal_id, draft):
    return _propose_result(graph, goal_id, draft, GENERATION)


def derive(graph, goal_id, draft):
    return _derive(graph, goal_id, draft, GENERATION)


def aggregate(graph, goal_id, draft):
    return _aggregate(graph, goal_id, draft, GENERATION)


def obligation(
    objective: GoalObjective = GoalObjective.IDENTITY,
    *,
    equation: str = "x = 5",
    domain: str = "real",
) -> GoalObligation:
    return GoalObligation("x", equation, domain, objective)


def graph_of(result):
    assert isinstance(result, Ok), getattr(result, "message", result)
    return result.graph


def verifier(*rules: MockRule) -> TableDrivenVerifier:
    return TableDrivenVerifier(tuple(rules))


def claim_graph(*, derivation: bool = False):
    graph = new_graph("solve x", obligation(), ("x is real",))
    if derivation:
        return graph_of(
            derive(graph, 0, DerivationDraft("x", Relation.EQ, "5", (1,)))
        )
    return graph_of(
        propose_solution(graph, 0, SolutionDraft("x", "5"))
    )


def aggregate_graph():
    graph = new_graph("sum branches", obligation(GoalObjective.SUM))
    graph = graph_of(
        decompose(
            graph,
            0,
            DecompositionDraft(
                Connective.EXHAUSTIVE_CASE,
                ("only branch",),
                (obligation(GoalObjective.BRANCH, equation="x = 2"),),
            ),
        )
    )
    graph = graph_of(
        propose_solution(graph, 1, SolutionDraft("x", "2"))
    )
    child_claim = len(graph.nodes) - 1
    checks = verifier(MockRule(ClaimKind.SOLUTION, "x = 2", True, True))
    graph = graph_of(verify_claim(graph, child_claim, checks))
    graph = graph_of(commit_claim(graph, child_claim))
    graph = graph_of(
        aggregate(
            graph,
            0,
            AggregateDraft(Reducer.SUM, (child_claim,), "2"),
        )
    )
    return graph, child_claim, len(graph.nodes) - 1


def test_initial_observation_has_exact_trusted_records_and_canonical_menu():
    graph = new_graph("solve x", obligation(), ("x is real",))

    observation = build_observation(State(graph, turn=0), CONTEXT)

    assert set(observation) == {
        "schema_version",
        "environment_contract_sha256",
        "root_goal_id",
        "turn",
        "turn_limit",
        "previous_outcome",
        "goals",
        "evidence",
        "claims",
        "legal_pairs",
    }
    assert observation["schema_version"] == CONTROLLER_OBSERVATION_VERSION
    assert observation["goals"] == [
        {
            "id": 0,
            "parent_goal": None,
            "statement": "solve x",
            "obligation": {
                "variable": "x",
                "equation": "x = 5",
                "domain": "real",
                "objective": "identity",
            },
            "assumptions": [],
            "connective": None,
            "status": "open",
            "committed_claim": None,
        }
    ]
    assert observation["evidence"] == [
        {
            "id": 1,
            "scope_goal": 0,
            "statement": "x is real",
            "premises": [],
            "source": "trusted_given",
            "committable": False,
        }
    ]
    assert observation["legal_pairs"] == [
        {"verb": "decompose", "target": 0},
        {"verb": "propose_solution", "target": 0},
    ]
    validate_observation(observation)


def test_payload_serialization_is_exact_and_uses_only_locked_enums():
    assert serialize_payload(SolutionPayload("x", "5")) == {
        "variable": "x",
        "value": "5",
    }
    assert serialize_payload(ResultPayload(GoalObjective.SUM, "8")) == {
        "objective": "sum",
        "value": "8",
    }
    assert serialize_payload(DerivationPayload("x", Relation.GE, "0")) == {
        "relation": "ge",
        "lhs": "x",
        "rhs": "0",
    }
    assert serialize_payload(AggregatePayload(Reducer.MAX, "8")) == {
        "reducer": "max",
        "result": "8",
    }
    assert serialize_payload(
        RefutationPayload(
            RefutationMode.UNIQUE_CANDIDATE_OUTSIDE_DOMAIN,
            "x",
            "-5",
        )
    ) == {
        "mode": "unique_candidate_outside_domain",
        "variable": "x",
        "candidate": "-5",
    }


def test_solved_child_is_compact_but_remains_a_valid_aggregate_premise():
    graph, child_claim, aggregate_claim = aggregate_graph()

    observation = build_observation(State(graph, turn=5), CONTEXT)

    child = next(goal for goal in observation["goals"] if goal["id"] == 1)
    assert child["status"] == "solved"
    compact = child["committed_claim"]
    assert compact["id"] == child_claim
    assert compact["kind"] == "solution"
    assert compact["payload"] == {"variable": "x", "value": "2"}
    assert compact["statement"] == "x = 2"
    assert compact["verifier_result"]["accepted"] is True
    assert compact["discharge_result"]["accepted"] is True
    assert [claim["id"] for claim in observation["claims"]] == [aggregate_claim]
    assert observation["claims"][0]["premises"] == [child_claim]
    assert observation["legal_pairs"] == []
    validate_observation(observation)


def test_fully_committed_decomposed_branch_keeps_its_policy_menu():
    graph = new_graph(
        "branch",
        obligation(GoalObjective.BRANCH, equation="x = 2"),
    )
    graph = graph_of(
        decompose(
            graph,
            0,
            DecompositionDraft(
                Connective.EXHAUSTIVE_CASE,
                ("only branch",),
                (obligation(GoalObjective.BRANCH, equation="x = 2"),),
            ),
        )
    )
    graph = graph_of(
        propose_solution(graph, 1, SolutionDraft("x", "2"))
    )
    child_claim = len(graph.nodes) - 1
    graph = graph_of(
        verify_claim(
            graph,
            child_claim,
            verifier(MockRule(ClaimKind.SOLUTION, "x = 2", True, True)),
        )
    )
    graph = graph_of(commit_claim(graph, child_claim))

    observation = build_observation(State(graph, turn=2), CONTEXT)

    assert observation["legal_pairs"] == [
        {"verb": "propose_solution", "target": 0},
        {"verb": "propose_refutation", "target": 0},
    ]
    validate_observation(observation)


def test_direct_result_can_close_a_previously_decomposed_root():
    graph = new_graph("sum", obligation(GoalObjective.SUM))
    graph = graph_of(
        decompose(
            graph,
            0,
            DecompositionDraft(
                Connective.EXHAUSTIVE_CASE,
                ("case",),
                (obligation(GoalObjective.BRANCH),),
            ),
        )
    )
    graph = graph_of(
        propose_result(graph, 0, ResultDraft(GoalObjective.SUM, "5"))
    )
    claim_id = len(graph.nodes) - 1
    graph = graph_of(
        verify_claim(
            graph,
            claim_id,
            verifier(
                MockRule(ClaimKind.RESULT, "result(sum) = 5", True, True)
            ),
        )
    )
    graph = graph_of(commit_claim(graph, claim_id))

    observation = build_observation(State(graph, turn=1), CONTEXT)

    assert [goal["id"] for goal in observation["goals"]] == [0]
    assert observation["goals"][0]["connective"] is None
    assert observation["goals"][0]["committed_claim"]["kind"] == "result"
    assert observation["legal_pairs"] == []
    validate_observation(observation)


def test_root_commit_keeps_compact_outcome_and_previous_action_reference():
    graph = graph_of(
        propose_solution(
            new_graph("solve x", obligation()),
            0,
            SolutionDraft("x", "5"),
        )
    )
    claim_id = len(graph.nodes) - 1
    graph = graph_of(
        verify_claim(
            graph,
            claim_id,
            verifier(MockRule(ClaimKind.SOLUTION, "x = 5", True, True)),
        )
    )
    graph = graph_of(commit_claim(graph, claim_id))
    outcome = TransitionOutcome(
        action=Action(Verb.COMMIT, claim_id),
        code=OutcomeCode.APPLIED,
    )

    observation = build_observation(
        State(graph, turn=3, previous_outcome=outcome), CONTEXT
    )

    assert observation["claims"] == []
    assert observation["goals"][0]["committed_claim"]["id"] == claim_id
    assert observation["previous_outcome"] == {
        "action": {"verb": "commit", "target": claim_id},
        "code": "applied",
        "reason": None,
        "verifier_context": None,
    }
    assert observation["legal_pairs"] == []
    validate_observation(observation)
    impossible_verify = deepcopy(observation)
    impossible_verify["previous_outcome"]["action"]["verb"] = "verify"
    verified = get(graph, claim_id).verification.context
    impossible_verify["previous_outcome"]["verifier_context"] = {
        "contract_id": verified.contract_id,
        "purpose": "validity",
        "claim_kind": "solution",
        "owner_goal": 0,
        "context_digest": verified.context_digest,
    }
    with pytest.raises(ObservationValidationError, match="full active Claim"):
        validate_observation(impossible_verify)




def test_previous_outcome_comes_only_from_state_and_keeps_verifier_context():
    graph = claim_graph()
    claim_id = len(graph.nodes) - 1
    graph = graph_of(
        verify_claim(
            graph,
            claim_id,
            verifier(MockRule(ClaimKind.SOLUTION, "x = 5", False, False)),
        )
    )
    claim = get(graph, claim_id)
    outcome = TransitionOutcome(
        action=Action(Verb.VERIFY, claim_id),
        code=OutcomeCode.APPLIED,
        verifier_context=claim.verification.context,
    )

    observation = build_observation(
        State(graph, turn=1, previous_outcome=outcome), CONTEXT
    )

    assert tuple(signature(build_observation).parameters) == ("state", "context")
    assert observation["previous_outcome"]["verifier_context"] == {
        "contract_id": "table-driven-claim-verifier-v1",
        "purpose": "validity",
        "claim_kind": "solution",
        "owner_goal": 0,
        "context_digest": claim.verification.context.context_digest,
    }
    assert observation["claims"][0]["status"] == "rejected"
    assert observation["claims"][0]["discharge_status"] == "unchecked"

    bad_discharge = deepcopy(observation)
    bad_discharge["claims"][0]["discharge_status"] = "rejected"
    with pytest.raises(ObservationValidationError, match="discharge unchecked"):
        validate_observation(bad_discharge)

    assert {"verb": "verify", "target": claim_id} not in observation[
        "legal_pairs"
    ]


def test_effective_closure_omits_descendants_and_pruned_claim_keeps_audit():
    graph = new_graph("sum branches", obligation(GoalObjective.SUM))
    graph = graph_of(
        decompose(
            graph,
            0,
            DecompositionDraft(
                Connective.CONJUNCTIVE,
                ("child",),
                (obligation(GoalObjective.BRANCH),),
            ),
        )
    )
    graph = graph_of(
        propose_solution(graph, 1, SolutionDraft("x", "5"))
    )
    descendant_claim = len(graph.nodes) - 1
    graph = graph_of(prune(graph, 1))

    observation = build_observation(State(graph, turn=3), CONTEXT)

    assert [(goal["id"], goal["status"]) for goal in observation["goals"]] == [
        (0, "open"),
        (1, "pruned"),
    ]
    assert descendant_claim not in {claim["id"] for claim in observation["claims"]}
    assert descendant_claim not in {
        pair["target"] for pair in observation["legal_pairs"]
    }

    active_graph = claim_graph()
    active_claim = len(active_graph.nodes) - 1
    active_graph = graph_of(
        verify_claim(
            active_graph,
            active_claim,
            verifier(MockRule(ClaimKind.SOLUTION, "x = 5", True, False)),
        )
    )
    active_graph = graph_of(prune(active_graph, active_claim))
    pruned = build_observation(State(active_graph, turn=2), CONTEXT)
    claim_record = pruned["claims"][0]
    assert claim_record["status"] == "pruned"
    assert claim_record["verifier_result"]["accepted"] is True
    assert claim_record["discharge_status"] == "rejected"
    assert active_claim not in {pair["target"] for pair in pruned["legal_pairs"]}


def test_strict_validator_rejects_unknown_fields_enums_spoofs_and_bad_refs():
    observation = build_observation(State(claim_graph(derivation=True), 0), CONTEXT)

    unknown = deepcopy(observation)
    unknown["goals"][0]["depth"] = 0
    with pytest.raises(ObservationValidationError, match="keys must be exactly"):
        validate_observation(unknown)

    bad_relation = deepcopy(observation)
    bad_relation["claims"][0]["payload"]["relation"] = "ne"
    with pytest.raises(ObservationValidationError, match="relation"):
        validate_observation(bad_relation)

    spoofed = deepcopy(observation)
    spoofed["claims"][0]["statement"] = "x = 999"
    with pytest.raises(ObservationValidationError, match="typed payload"):
        validate_observation(spoofed)

    bad_owner = deepcopy(observation)
    bad_owner["claims"][0]["owner_goal"] = 99
    with pytest.raises(ObservationValidationError, match="owner_goal"):
        validate_observation(bad_owner)

    evidence_target = deepcopy(observation)
    evidence_target["legal_pairs"] = [{"verb": "verify", "target": 1}]
    with pytest.raises(ObservationValidationError, match="must target a Claim"):
        validate_observation(evidence_target)

    missing_legal = build_observation(
        State(new_graph("solve x", obligation()), 0), CONTEXT
    )
    missing_legal["legal_pairs"].pop()
    with pytest.raises(ObservationValidationError, match="canonical menu"):
        validate_observation(missing_legal)


def test_validator_rejects_nonaggregate_sibling_branch_premise():
    graph = new_graph("sum", obligation(GoalObjective.SUM))
    graph = graph_of(
        decompose(
            graph,
            0,
            DecompositionDraft(
                Connective.CONJUNCTIVE,
                ("left", "right"),
                (
                    obligation(GoalObjective.BRANCH, equation="x = 2"),
                    obligation(GoalObjective.BRANCH, equation="x = 3"),
                ),
            ),
        )
    )
    graph = graph_of(
        propose_solution(graph, 1, SolutionDraft("x", "2"))
    )
    sibling_claim = len(graph.nodes) - 1
    graph = graph_of(
        derive(graph, 2, DerivationDraft("1", Relation.EQ, "1", ()))
    )
    observation = build_observation(State(graph, turn=1), CONTEXT)
    observation["claims"][1]["premises"] = [sibling_claim]

    with pytest.raises(ObservationValidationError, match="owner ancestry"):
        validate_observation(observation)


def test_validator_rejects_tampered_status_order_and_aggregate_connective():
    graph = claim_graph()
    claim_id = len(graph.nodes) - 1
    observation = build_observation(State(graph, 0), CONTEXT)

    mismatched = deepcopy(observation)
    mismatched["claims"][0]["status"] = "verified"
    with pytest.raises(ObservationValidationError, match="accepted verifier"):
        validate_observation(mismatched)

    premised = deepcopy(observation)
    premised["claims"][0]["premises"] = [1]
    with pytest.raises(ObservationValidationError, match="must not have premises"):
        validate_observation(premised)

    unsorted = build_observation(
        State(new_graph("sum solutions", obligation(GoalObjective.SUM)), 0),
        CONTEXT,
    )
    unsorted["legal_pairs"] = list(reversed(unsorted["legal_pairs"]))
    with pytest.raises(ObservationValidationError, match="fixed verb"):
        validate_observation(unsorted)

    aggregate, _, _ = aggregate_graph()
    bad_connective = build_observation(State(aggregate, 5), CONTEXT)
    bad_connective["goals"][0]["connective"] = "conjunctive"
    with pytest.raises(ObservationValidationError, match="exhaustive child"):
        validate_observation(bad_connective)


def test_canonical_bytes_and_digest_ignore_json_object_insertion_order():
    observation = build_observation(State(claim_graph(), 0), CONTEXT)
    reordered = dict(reversed(list(deepcopy(observation).items())))
    reordered["claims"][0]["payload"] = dict(
        reversed(list(reordered["claims"][0]["payload"].items()))
    )

    assert canonical_observation_bytes(
        reordered,
        GENERATION,
        GENERATION,
        environment_contract_sha256=CONTRACT_DIGEST,
    ) == canonical_observation_bytes(
        observation,
        GENERATION,
        GENERATION,
        environment_contract_sha256=CONTRACT_DIGEST,
    )
    assert observation_sha256(
        reordered,
        GENERATION,
        GENERATION,
        environment_contract_sha256=CONTRACT_DIGEST,
    ) == observation_sha256(
        observation,
        GENERATION,
        GENERATION,
        environment_contract_sha256=CONTRACT_DIGEST,
    )
    assert len(
        observation_sha256(
            observation,
            GENERATION,
            GENERATION,
            environment_contract_sha256=CONTRACT_DIGEST,
        )
    ) == 64


def test_applied_verify_context_is_an_exact_receipt_for_the_full_claim():
    graph = new_graph("sum branches", obligation(GoalObjective.SUM))
    graph = graph_of(
        decompose(
            graph,
            0,
            DecompositionDraft(
                Connective.EXHAUSTIVE_CASE,
                ("child",),
                (obligation(GoalObjective.BRANCH),),
            ),
        )
    )
    graph = graph_of(propose_solution(graph, 1, SolutionDraft("x", "5")))
    claim_id = len(graph.nodes) - 1
    graph = graph_of(
        verify_claim(
            graph,
            claim_id,
            verifier(MockRule(ClaimKind.SOLUTION, "x = 5", False, False)),
        )
    )
    claim = get(graph, claim_id)
    outcome = TransitionOutcome(
        action=Action(Verb.VERIFY, claim_id),
        code=OutcomeCode.APPLIED,
        verifier_context=claim.verification.context,
    )
    observation = build_observation(
        State(graph, turn=2, previous_outcome=outcome), CONTEXT
    )

    mutations = (
        ("purpose", "owner_discharge", "must be validity"),
        ("claim_kind", "derivation", "target Claim"),
        ("owner_goal", 0, "target Claim owner"),
        ("contract_id", "tampered-contract", "verifier result"),
        ("context_digest", "b" * 64, "verifier result"),
    )
    for field, value, message in mutations:
        tampered = deepcopy(observation)
        tampered["previous_outcome"]["verifier_context"][field] = value
        with pytest.raises(ObservationValidationError, match=message):
            validate_observation(tampered)

    missing = deepcopy(observation)
    missing["previous_outcome"]["verifier_context"] = None
    with pytest.raises(ObservationValidationError, match="required"):
        validate_observation(missing)

    non_verify = deepcopy(observation)
    non_verify["previous_outcome"]["action"] = {
        "verb": "derive",
        "target": 0,
    }
    with pytest.raises(ObservationValidationError, match="applied VERIFY"):
        validate_observation(non_verify)


@pytest.mark.parametrize(
    ("turn_limit", "digest"),
    [(0, CONTRACT_DIGEST), (True, CONTRACT_DIGEST), (16, "A" * 64)],
)
def test_observation_context_rejects_noncanonical_static_values(
    turn_limit, digest
):
    with pytest.raises(ValueError):
        ObservationContext(turn_limit, digest)
