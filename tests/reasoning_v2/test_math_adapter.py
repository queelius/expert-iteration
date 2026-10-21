import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("sympy")

from verifier_anchored_rl.reasoning_v2 import (
    AggregateDraft,
    AggregatePayload,
    CheckPurpose,
    Claim,
    ClaimKind,
    Connective,
    DecompositionDraft,
    DerivationPayload,
    Err,
    Goal,
    GoalObjective,
    GoalObligation,
    Graph,
    Ok,
    ObservationContext,
    Reducer,
    RefutationMode,
    RefutationPayload,
    Relation,
    ResultDraft,
    ResultPayload,
    SolutionDraft,
    SolutionPayload,
    State,
    aggregate,
    build_observation,
    commit_claim,
    decompose,
    get,
    new_graph,
    propose_refutation,
    propose_result,
    propose_solution,
    validate_graph,
    validate_observation,
    validate_observation_with_contract,
    verify_claim,
)
from verifier_anchored_rl.reasoning_v2.actions import RefutationDraft
from verifier_anchored_rl.reasoning_v2.math_adapter import (
    AGGREGATE_CONTRACT_ID,
    AGGREGATE_CONTRACT_ID_V11,
    CANONICALIZATION_CONTRACT_ID,
    CANONICALIZATION_CONTRACT_ID_V11,
    CONTRACT_ID,
    CONTRACT_ID_V11,
    CORRECTED_DOMAIN_GRAMMAR_PROFILE,
    DOMAIN_GRAMMAR_IMPLEMENTATION_ID,
    DISCHARGE_REJECTION_CODES,
    LEGACY_DOMAIN_GRAMMAR_PROFILE,
    PAYLOAD_SEMANTICS_ID,
    VALIDITY_REJECTION_CODES,
    CaseworkMathVerifier,
    DomainGrammarProfile,
    build_casework_root_obligation,
    canonicalize_goal_obligation_v11,
    canonicalize_aggregate_payload,
    canonicalize_claim_payload,
    canonicalize_derivation_payload,
    canonicalize_goal_obligation,
    canonicalize_refutation_payload,
    canonicalize_result_payload,
    canonicalize_solution_payload,
    math_verifier_context,
    validate_decomposition_coverage,
    validate_decomposition_coverage_v11,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFIER = CaseworkMathVerifier()
VERIFIER_V11 = CaseworkMathVerifier.corrected_v11()


def obligation(
    objective: GoalObjective,
    equation: str,
    domain: str = "Reals",
) -> GoalObligation:
    return GoalObligation("x", equation, domain, objective)


def graph_of(result) -> Graph:
    assert isinstance(result, Ok), getattr(result, "message", result)
    validate_graph(result.graph)
    return result.graph


def append_claim(
    graph: Graph,
    kind: ClaimKind,
    payload,
    *,
    owner_goal: int = 0,
    premises: tuple[int, ...] = (),
) -> tuple[Graph, Claim]:
    claim = Claim(
        id=len(graph.nodes),
        owner_goal=owner_goal,
        kind=kind,
        payload=payload,
        premises=premises,
    )
    graph = replace(graph, nodes=graph.nodes + (claim,))
    validate_graph(graph)
    return graph, claim


def decision_for(owner: GoalObligation, kind: ClaimKind, payload):
    graph = new_graph(
        "opaque problem text", owner, generation_contract=VERIFIER
    )
    graph, claim = append_claim(graph, kind, payload)
    return VERIFIER.verify(graph, claim)


def branch(equation: str, domain: str) -> GoalObligation:
    return obligation(GoalObjective.BRANCH, equation, domain)


def aggregate_graph(
    parent: GoalObligation,
    children: tuple[GoalObligation, ...],
    outcomes: tuple[tuple[str, str], ...],
    reducer: Reducer,
    result: str,
    verifier=VERIFIER,
) -> tuple[Graph, int, tuple[int, ...]]:
    graph = new_graph(
        "opaque problem text", parent, generation_contract=verifier
    )
    graph = graph_of(
        decompose(
            graph,
            0,
            DecompositionDraft(
                Connective.EXHAUSTIVE_CASE,
                tuple(f"case {index}" for index in range(len(children))),
                children,
            ),
            verifier,
        )
    )
    claim_ids = []
    for child_id, (kind, value) in enumerate(outcomes, start=1):
        if kind == "solution":
            graph = graph_of(
                propose_solution(
                    graph, child_id, SolutionDraft("x", value), verifier
                )
            )
        else:
            graph = graph_of(
                propose_refutation(
                    graph,
                    child_id,
                    RefutationDraft(
                        RefutationMode.UNIQUE_CANDIDATE_OUTSIDE_DOMAIN,
                        "x",
                        value,
                    ),
                    verifier,
                )
            )
        claim_id = len(graph.nodes) - 1
        graph = graph_of(verify_claim(graph, claim_id, verifier))
        checked = get(graph, claim_id)
        assert isinstance(checked, Claim)
        assert checked.verification is not None
        assert checked.verification.accepted
        assert checked.discharge is not None
        assert checked.discharge.accepted
        graph = graph_of(commit_claim(graph, claim_id))
        claim_ids.append(claim_id)
    graph = graph_of(
        aggregate(
            graph,
            0,
            AggregateDraft(reducer, tuple(claim_ids), result),
            verifier,
        )
    )
    return graph, len(graph.nodes) - 1, tuple(claim_ids)


class NoLabelAccess(dict):
    def __getitem__(self, key):
        if key in {"answer", "solution_set"}:
            raise AssertionError(f"builder read forbidden label {key}")
        return super().__getitem__(key)


def test_corrected_profile_has_new_contract_ids_and_unchanged_payload_rules():
    assert type(VERIFIER_V11) is CaseworkMathVerifier
    assert type(VERIFIER_V11).__module__ == type(VERIFIER).__module__
    assert type(VERIFIER_V11).__qualname__ == type(VERIFIER).__qualname__
    assert VERIFIER.contract_id == CONTRACT_ID == "casework-claim-verifier-v1"
    assert (
        VERIFIER.generation_contract_id
        == VERIFIER.obligation_contract_id
        == CANONICALIZATION_CONTRACT_ID
        == "casework-goal-obligation-v1"
    )
    assert (
        VERIFIER.aggregate_contract_id
        == AGGREGATE_CONTRACT_ID
        == "casework-exact-aggregate-v1"
    )
    assert VERIFIER_V11.contract_id == CONTRACT_ID_V11
    assert (
        VERIFIER_V11.generation_contract_id
        == VERIFIER_V11.obligation_contract_id
        == CANONICALIZATION_CONTRACT_ID_V11
        == "casework-goal-obligation-v1.1"
    )
    assert (
        VERIFIER_V11.aggregate_contract_id
        == AGGREGATE_CONTRACT_ID_V11
        == "casework-exact-aggregate-v1.1"
    )
    assert VERIFIER.payload_semantics_id == PAYLOAD_SEMANTICS_ID
    assert VERIFIER_V11.payload_semantics_id == PAYLOAD_SEMANTICS_ID
    assert VERIFIER_V11.validity_rejection_codes == VALIDITY_REJECTION_CODES
    assert VERIFIER_V11.discharge_rejection_codes == DISCHARGE_REJECTION_CODES
    assert VERIFIER.domain_grammar_profile is LEGACY_DOMAIN_GRAMMAR_PROFILE
    assert (
        VERIFIER_V11.domain_grammar_profile
        is CORRECTED_DOMAIN_GRAMMAR_PROFILE
    )
    assert (
        VERIFIER.domain_grammar_implementation_id
        == VERIFIER_V11.domain_grammar_implementation_id
        == DOMAIN_GRAMMAR_IMPLEMENTATION_ID
    )
    with pytest.raises(ValueError, match="unsupported domain grammar profile"):
        CaseworkMathVerifier(
            domain_grammar_profile=DomainGrammarProfile(
                profile_id=CORRECTED_DOMAIN_GRAMMAR_PROFILE.profile_id,
                bounded_comparator_pattern=(
                    LEGACY_DOMAIN_GRAMMAR_PROFILE.bounded_comparator_pattern
                ),
            )
        )
    with pytest.raises(ValueError, match="contract ids are mixed"):
        CaseworkMathVerifier(contract_id=CONTRACT_ID_V11)


def test_upper_inclusive_bounded_domain_is_versioned_not_silently_changed():
    bounded = branch("Eq(x, 1)", "0 < x <= 2")

    with pytest.raises(ValueError, match="malformed_scalar"):
        canonicalize_goal_obligation(bounded)
    with pytest.raises(ValueError, match="malformed_scalar"):
        VERIFIER.canonicalize_obligation(bounded)

    assert canonicalize_goal_obligation_v11(bounded) == bounded
    assert VERIFIER_V11.canonicalize_obligation(bounded) == bounded


def test_root_obligation_uses_only_trusted_casework_metadata():
    record = NoLabelAccess({
        "family": "f1b",
        "question_type": "find_x",
        "symbolic": "Eq( Abs( 5*x - 30 ), 35/1 )",
        "answer": "-1",
        "solution_set": ["-1"],
        "difficulty": {"params": {"direction": "lt", "d": "24/2"}},
    })

    built = build_casework_root_obligation(record)

    assert built == obligation(
        GoalObjective.IDENTITY,
        "Eq(Abs(5*x - 30), 35)",
        "x < 12",
    )
    changed = dict(record)
    changed["answer"] = "999"
    changed["solution_set"] = ["999"]
    assert build_casework_root_obligation(changed) == built


@pytest.mark.parametrize("objective", [GoalObjective.BRANCH, GoalObjective.IDENTITY])
def test_singleton_solution_verifies_and_discharges(objective):
    decision = decision_for(
        obligation(objective, "Eq(2*x, 4)"),
        ClaimKind.SOLUTION,
        SolutionPayload("x", "2"),
    )

    assert decision.verification.accepted
    assert decision.discharge.accepted


def test_solution_validity_is_distinct_from_owner_discharge():
    nonunique = decision_for(
        obligation(GoalObjective.BRANCH, "Eq(Abs(x), 1)"),
        ClaimKind.SOLUTION,
        SolutionPayload("x", "1"),
    )
    aggregate_owner = decision_for(
        obligation(GoalObjective.SUM, "Eq(x, 1)"),
        ClaimKind.SOLUTION,
        SolutionPayload("x", "1"),
    )

    assert nonunique.verification.accepted
    assert not nonunique.discharge.accepted
    assert nonunique.discharge.reason_code == "not_unique_owner_solution"
    assert aggregate_owner.verification.accepted
    assert not aggregate_owner.discharge.accepted
    assert aggregate_owner.discharge.reason_code == "owner_not_branch"


@pytest.mark.parametrize(
    ("objective", "equation", "value"),
    [
        (GoalObjective.SUM, "Eq((x - 2)*(x - 3), 0)", "5"),
        (GoalObjective.MAX, "Eq((x - 2)*(x - 3), 0)", "3"),
        (GoalObjective.MIN, "Eq((x - 2)*(x - 3), 0)", "2"),
        (GoalObjective.IDENTITY, "Eq(2*x, 14)", "7"),
    ],
)
def test_direct_result_recomputes_and_discharges_full_objective(
    objective, equation, value
):
    decision = decision_for(
        obligation(objective, equation),
        ClaimKind.RESULT,
        ResultPayload(objective, value),
    )

    assert decision.verification.accepted
    assert decision.discharge is not None
    assert decision.discharge.accepted


def test_wrong_direct_result_fails_validity_without_discharge():
    decision = decision_for(
        obligation(GoalObjective.SUM, "Eq((x - 2)*(x - 3), 0)"),
        ClaimKind.RESULT,
        ResultPayload(GoalObjective.SUM, "6"),
    )

    assert not decision.verification.accepted
    assert decision.verification.reason_code == "result_value_mismatch"
    assert decision.discharge is None


def test_wrong_or_out_of_domain_solution_fails_closed_with_bounded_output():
    decision = decision_for(
        branch("Eq(x, -1)", "x > 0"),
        ClaimKind.SOLUTION,
        SolutionPayload("x", "-1"),
    )

    assert not decision.verification.accepted
    assert decision.verification.observed == "candidate_not_feasible"
    assert decision.discharge is None


@pytest.mark.parametrize(
    ("payload", "accepted"),
    [
        (DerivationPayload("2 + 3", Relation.EQ, "5"), True),
        (DerivationPayload("2", Relation.LT, "3"), True),
        (DerivationPayload("2 + 3", Relation.EQ, "6"), False),
        (DerivationPayload("x + 1", Relation.EQ, "2"), False),
    ],
)
def test_derivation_is_ground_arithmetic_and_never_discharges(payload, accepted):
    decision = decision_for(
        obligation(GoalObjective.IDENTITY, "Eq(x, 0)"),
        ClaimKind.DERIVATION,
        payload,
    )

    assert decision.verification.accepted is accepted
    if accepted:
        assert decision.discharge is not None
        assert not decision.discharge.accepted
        assert decision.discharge.reason_code == "derivation_not_discharging"
    else:
        assert decision.discharge is None


def test_derivation_requires_current_verified_premises():
    graph = new_graph(
        "opaque problem text",
        obligation(GoalObjective.IDENTITY, "Eq(x, 1)"),
        generation_contract=VERIFIER,
    )
    graph, premise = append_claim(
        graph,
        ClaimKind.DERIVATION,
        DerivationPayload("1", Relation.EQ, "1"),
    )
    graph, dependent = append_claim(
        graph,
        ClaimKind.DERIVATION,
        DerivationPayload("2", Relation.EQ, "2"),
        premises=(premise.id,),
    )

    unchecked = VERIFIER.verify(graph, dependent)
    assert unchecked.verification.reason_code == "premise_not_verified"

    graph = graph_of(verify_claim(graph, premise.id, VERIFIER))
    dependent = get(graph, dependent.id)
    assert isinstance(dependent, Claim)
    current = VERIFIER.verify(graph, dependent)
    assert current.verification.accepted
    assert current.discharge is not None
    assert not current.discharge.accepted

    checked_premise = get(graph, premise.id)
    assert isinstance(checked_premise, Claim)
    graph = replace(
        graph,
        nodes=graph.nodes[:premise.id]
        + (replace(checked_premise, pruned=True),)
        + graph.nodes[premise.id + 1 :],
    )
    stale = VERIFIER.verify(graph, dependent)
    assert stale.verification.reason_code == "premise_not_active"


def test_legacy_receipt_is_not_current_under_corrected_profile():
    graph = new_graph(
        "opaque problem text",
        obligation(GoalObjective.IDENTITY, "Eq(x, 1)"),
        generation_contract=VERIFIER,
    )
    graph, premise = append_claim(
        graph,
        ClaimKind.DERIVATION,
        DerivationPayload("1", Relation.EQ, "1"),
    )
    graph = graph_of(verify_claim(graph, premise.id, VERIFIER))
    checked = get(graph, premise.id)
    assert isinstance(checked, Claim)
    assert checked.verification is not None
    assert checked.verification.context.contract_id == CONTRACT_ID

    graph, dependent = append_claim(
        graph,
        ClaimKind.DERIVATION,
        DerivationPayload("2", Relation.EQ, "2"),
        premises=(premise.id,),
    )

    assert VERIFIER.verify(graph, dependent).verification.accepted
    current = VERIFIER_V11.verify(graph, dependent)
    assert not current.verification.accepted
    assert current.verification.reason_code == "premise_not_verified"
    assert current.verification.context.contract_id == CONTRACT_ID_V11


def test_generation_contract_canonicalizes_payloads_before_storage():
    parent = obligation(GoalObjective.SUM, "Eq((x - 2)*(x - 3), 0)")
    graph = new_graph(
        "sum roots", parent, generation_contract=VERIFIER
    )
    graph = graph_of(
        propose_result(
            graph,
            0,
            ResultDraft(GoalObjective.SUM, "10/2"),
            VERIFIER,
        )
    )
    claim = get(graph, len(graph.nodes) - 1)

    assert isinstance(claim, Claim)
    assert claim.payload == ResultPayload(GoalObjective.SUM, "5")
    assert canonicalize_result_payload(
        ResultPayload(GoalObjective.SUM, "2 + 3")
    ) == ResultPayload(GoalObjective.SUM, "5")


def test_environment_bound_observation_validation_rejects_symbolic_tampering():
    graph = new_graph(
        "sum roots",
        obligation(GoalObjective.SUM, "Eq((x - 2)*(x - 3), 0)"),
        generation_contract=VERIFIER,
    )
    graph = graph_of(
        propose_result(
            graph,
            0,
            ResultDraft(GoalObjective.SUM, "2 + 3"),
            VERIFIER,
        )
    )
    observation = build_observation(
        State(graph, turn=0), ObservationContext(16, "a" * 64)
    )
    validate_observation_with_contract(
        observation,
        VERIFIER,
        VERIFIER,
        environment_contract_sha256="a" * 64,
    )

    bad_obligation = deepcopy(observation)
    bad_obligation["goals"][0]["obligation"]["equation"] = "not symbolic"
    validate_observation(bad_obligation)
    with pytest.raises(ValueError, match="generation contract"):
        validate_observation_with_contract(
            bad_obligation,
            VERIFIER,
            VERIFIER,
            environment_contract_sha256="a" * 64,
        )

    bad_payload = deepcopy(observation)
    bad_payload["claims"][0]["payload"]["value"] = "2 + 3"
    bad_payload["claims"][0]["statement"] = "result(sum) = 2 + 3"
    validate_observation(bad_payload)
    with pytest.raises(ValueError, match="symbolically canonical"):
        validate_observation_with_contract(
            bad_payload,
            VERIFIER,
            VERIFIER,
            environment_contract_sha256="a" * 64,
        )

    rejected_graph = new_graph(
        "sum roots",
        obligation(GoalObjective.SUM, "Eq((x - 2)*(x - 3), 0)"),
        generation_contract=VERIFIER,
    )
    rejected_graph = graph_of(
        propose_result(
            rejected_graph,
            0,
            ResultDraft(GoalObjective.SUM, "6"),
            VERIFIER,
        )
    )
    rejected_claim = len(rejected_graph.nodes) - 1
    rejected_graph = graph_of(
        verify_claim(rejected_graph, rejected_claim, VERIFIER)
    )
    rejected = build_observation(
        State(rejected_graph, turn=1), ObservationContext(16, "a" * 64)
    )

    bad_reason = deepcopy(rejected)
    bad_reason["claims"][0]["verifier_result"]["reason_code"] = (
        "provider supplied text"
    )
    validate_observation(bad_reason)
    with pytest.raises(ValueError, match="not allowed"):
        validate_observation_with_contract(
            bad_reason,
            VERIFIER,
            VERIFIER,
            environment_contract_sha256="a" * 64,
        )

    bad_contract = deepcopy(rejected)
    bad_contract["claims"][0]["verifier_result"]["contract_id"] = "other"
    validate_observation(bad_contract)
    with pytest.raises(ValueError, match="pinned verifier"):
        validate_observation_with_contract(
            bad_contract,
            VERIFIER,
            VERIFIER,
            environment_contract_sha256="a" * 64,
        )


def test_unique_out_of_domain_refutation_verifies_and_discharges_branch():
    decision = decision_for(
        branch("Eq(2*x, -2)", "x > 0"),
        ClaimKind.REFUTATION,
        RefutationPayload(
            RefutationMode.UNIQUE_CANDIDATE_OUTSIDE_DOMAIN, "x", "-1"
        ),
    )

    assert decision.verification.accepted
    assert decision.discharge.accepted


@pytest.mark.parametrize(
    ("owner", "candidate", "reason"),
    [
        (branch("Eq(x, -1)", "x >= -1"), "-1", "candidate_not_outside_domain"),
        (
            branch("Eq(x**2, 1)", "x > 2"),
            "1",
            "candidate_not_unique_raw_solution",
        ),
    ],
)
def test_refutation_rejects_boundary_and_nonunique_certificates(
    owner, candidate, reason
):
    decision = decision_for(
        owner,
        ClaimKind.REFUTATION,
        RefutationPayload(
            RefutationMode.UNIQUE_CANDIDATE_OUTSIDE_DOMAIN,
            "x",
            candidate,
        ),
    )

    assert not decision.verification.accepted
    assert decision.verification.reason_code == reason
    assert decision.discharge is None


def test_coverage_requires_exact_disjoint_domains_and_local_equivalence():
    parent = obligation(GoalObjective.SUM, "Eq(Abs(x), 1)")
    valid = (
        branch("Eq(x, -1)", "x < 0"),
        branch("Eq(x, 1)", "x >= 0"),
    )
    gap = (
        branch("Eq(x, -1)", "x < 0"),
        branch("Eq(x, 1)", "x > 0"),
    )
    overlap = (
        branch("Eq(x, -1)", "x <= 0"),
        branch("Eq(x, 1)", "x >= 0"),
    )
    wrong_equation = (
        branch("Eq(x, -1)", "x < 0"),
        branch("Eq(x, 2)", "x >= 0"),
    )

    legacy = validate_decomposition_coverage(parent, valid)
    assert legacy.accepted
    assert legacy.context_digest == (
        "6d567c746a4e4ab689a56b3980731c5c44202474bba8ba86268e6f596f18d48a"
    )
    assert validate_decomposition_coverage(
        parent, gap
    ).reason_code == "decomposition_domain_not_exhaustive"
    assert validate_decomposition_coverage(
        parent, overlap
    ).reason_code == "decomposition_domain_overlap"
    assert validate_decomposition_coverage(
        parent, wrong_equation
    ).reason_code == "child_equation_not_equivalent"


def test_coverage_supports_bounded_intervals_for_restricted_root_domain():
    parent = obligation(
        GoalObjective.IDENTITY,
        "Eq(Abs(5*x - 30), 35)",
        "x < 12",
    )
    children = (
        branch("Eq(5*x - 30, 35)", "x < -1"),
        branch("Eq(5*x - 30, -35)", "-1 <= x < 12"),
    )

    assert validate_decomposition_coverage(parent, children).accepted


def test_corrected_coverage_accepts_upper_inclusive_bounded_child():
    equation = "Eq((x + 1)*(x - 1)*(x - 3), 0)"
    parent = obligation(GoalObjective.SUM, equation)
    children = (
        branch(equation, "x <= 0"),
        branch(equation, "0 < x <= 2"),
        branch(equation, "x > 2"),
    )

    legacy = validate_decomposition_coverage(parent, children)
    corrected = validate_decomposition_coverage_v11(parent, children)

    assert not legacy.accepted
    assert legacy.reason_code == "malformed_scalar"
    assert corrected.accepted
    assert corrected.reason_code is None
    assert corrected.context_digest != legacy.context_digest
    assert VERIFIER_V11.validate_decomposition(
        parent, Connective.EXHAUSTIVE_CASE, children
    ) == tuple(
        VERIFIER_V11.canonicalize_obligation(child) for child in children
    )
    with pytest.raises(ValueError, match="malformed_scalar"):
        VERIFIER.validate_decomposition(
            parent, Connective.EXHAUSTIVE_CASE, children
        )


def test_corrected_profile_aggregates_through_upper_inclusive_branch():
    equation = "Eq((x + 1)*(x - 1)*(x - 3), 0)"
    graph, aggregate_id, _ = aggregate_graph(
        obligation(GoalObjective.SUM, equation),
        (
            branch(equation, "x <= 0"),
            branch(equation, "0 < x <= 2"),
            branch(equation, "x > 2"),
        ),
        (("solution", "-1"), ("solution", "1"), ("solution", "3")),
        Reducer.SUM,
        "3",
        verifier=VERIFIER_V11,
    )
    owner = get(graph, 0)
    aggregate_claim = get(graph, aggregate_id)

    assert isinstance(owner, Goal)
    assert VERIFIER_V11.build_aggregate_payload(
        graph, owner
    ) == AggregatePayload(Reducer.SUM, "3")
    assert isinstance(aggregate_claim, Claim)
    decision = VERIFIER_V11.verify(graph, aggregate_claim)
    assert decision.verification.accepted
    assert decision.discharge is not None
    assert decision.discharge.accepted


def test_sum_aggregate_recomputes_exact_committed_child_outcomes():
    graph, aggregate_id, _ = aggregate_graph(
        obligation(GoalObjective.SUM, "Eq((x - 2)*(x - 3), 0)"),
        (
            branch("Eq(x, 2)", "x < 5/2"),
            branch("Eq(x, 3)", "x >= 5/2"),
        ),
        (("solution", "2"), ("solution", "3")),
        Reducer.SUM,
        "5",
    )
    claim = get(graph, aggregate_id)
    assert isinstance(claim, Claim)

    decision = VERIFIER.verify(graph, claim)

    assert decision.verification.accepted
    assert decision.discharge.accepted


def test_identity_aggregate_accepts_one_solution_and_one_refutation():
    graph, aggregate_id, _ = aggregate_graph(
        obligation(GoalObjective.IDENTITY, "Eq(Abs(x), 1)", "x > 0"),
        (
            branch("Eq(x, -1)", "0 < x < 1"),
            branch("Eq(x, 1)", "x >= 1"),
        ),
        (("refutation", "-1"), ("solution", "1")),
        Reducer.IDENTITY,
        "1",
    )
    claim = get(graph, aggregate_id)
    assert isinstance(claim, Claim)

    decision = VERIFIER.verify(graph, claim)

    assert decision.verification.accepted
    assert decision.discharge.accepted


def test_overlapping_decomposition_is_rejected_before_children_are_added():
    graph = new_graph(
        "opaque problem text",
        obligation(GoalObjective.SUM, "Eq((x - 2)*(x - 3), 0)"),
        generation_contract=VERIFIER,
    )
    result = decompose(
        graph,
        0,
        DecompositionDraft(
            Connective.EXHAUSTIVE_CASE,
            ("left", "right"),
            (
                branch("Eq(x, 2)", "x <= 5/2"),
                branch("Eq(x, 3)", "x >= 5/2"),
            ),
        ),
        VERIFIER,
    )

    assert isinstance(result, Err)
    assert result.message == (
        "decomposition rejected: decomposition_domain_overlap"
    )
    assert len(graph.nodes) == 1


def test_aggregate_rejects_wrong_result_and_reducer():
    graph, aggregate_id, _ = aggregate_graph(
        obligation(GoalObjective.SUM, "Eq((x - 2)*(x - 3), 0)"),
        (
            branch("Eq(x, 2)", "x < 5/2"),
            branch("Eq(x, 3)", "x >= 5/2"),
        ),
        (("solution", "2"), ("solution", "3")),
        Reducer.SUM,
        "6",
    )
    claim = get(graph, aggregate_id)
    assert isinstance(claim, Claim)
    wrong_result = VERIFIER.verify(graph, claim)
    assert wrong_result.verification.reason_code == "aggregate_result_mismatch"

    wrong_reducer_claim = replace(
        claim, payload=AggregatePayload(Reducer.MAX, "5")
    )
    wrong_reducer_graph = replace(
        graph,
        nodes=graph.nodes[:aggregate_id]
        + (wrong_reducer_claim,)
        + graph.nodes[aggregate_id + 1 :],
    )
    wrong_reducer = VERIFIER.verify(wrong_reducer_graph, wrong_reducer_claim)
    assert wrong_reducer.verification.reason_code == "reducer_objective_mismatch"


def test_context_is_stable_across_irrelevant_decomposition_but_binds_aggregate_state():
    graph = new_graph(
        "opaque problem text",
        obligation(GoalObjective.IDENTITY, "Eq(x, 1)"),
        generation_contract=VERIFIER,
    )
    graph = graph_of(
        propose_solution(graph, 0, SolutionDraft("x", "1"), VERIFIER)
    )
    solution_id = len(graph.nodes) - 1
    graph = graph_of(verify_claim(graph, solution_id, VERIFIER))
    solution = get(graph, solution_id)
    assert isinstance(solution, Claim)
    before = solution.verification.context
    graph = graph_of(
        decompose(
            graph,
            0,
            DecompositionDraft(
                Connective.EXHAUSTIVE_CASE,
                ("left", "right"),
                (
                    branch("Eq(x, 1)", "x < 1"),
                    branch("Eq(x, 1)", "x >= 1"),
                ),
            ),
            VERIFIER,
        )
    )
    solution = get(graph, solution_id)
    assert isinstance(solution, Claim)
    assert before == math_verifier_context(
        graph, solution, CheckPurpose.VALIDITY
    )

    aggregate_state, aggregate_id, _ = aggregate_graph(
        obligation(GoalObjective.SUM, "Eq((x - 2)*(x - 3), 0)"),
        (
            branch("Eq(x, 2)", "x < 5/2"),
            branch("Eq(x, 3)", "x >= 5/2"),
        ),
        (("solution", "2"), ("solution", "3")),
        Reducer.SUM,
        "5",
    )
    aggregate_claim = get(aggregate_state, aggregate_id)
    assert isinstance(aggregate_claim, Claim)
    context = math_verifier_context(
        aggregate_state, aggregate_claim, CheckPurpose.VALIDITY
    )
    assert context == math_verifier_context(
        aggregate_state, aggregate_claim, CheckPurpose.VALIDITY
    )
    assert context.context_digest != math_verifier_context(
        aggregate_state, aggregate_claim, CheckPurpose.OWNER_DISCHARGE
    ).context_digest

    child = get(aggregate_state, 1)
    assert isinstance(child, Goal)
    changed_state = replace(
        aggregate_state,
        nodes=aggregate_state.nodes[:1]
        + (replace(child, assumptions=("changed",)),)
        + aggregate_state.nodes[2:],
    )
    changed = math_verifier_context(
        changed_state, aggregate_claim, CheckPurpose.VALIDITY
    )
    assert changed.context_digest != context.context_digest
    stale = VERIFIER.verify(changed_state, aggregate_claim)
    assert stale.verification.reason_code == "premise_not_verified"


def casework_children(record, parent):
    params = record["difficulty"]["params"]
    family = record["family"]
    if family in {"f1a", "f1b"}:
        a, b, c = params["a"], params["b"], params["c"]
        lo, hi, middle = (
            params["m"] - params["k"],
            params["m"] + params["k"],
            params["m"],
        )
        negative = f"Eq({a}*x + ({b}), {-c})"
        positive = f"Eq({a}*x + ({b}), {c})"
        if family == "f1a":
            return (
                branch(negative, f"x < {middle}"),
                branch(positive, f"x >= {middle}"),
            )
        if params["direction"] == "gt":
            return (
                branch(negative, f"{params['d']} < x < {hi}"),
                branch(positive, f"x >= {hi}"),
            )
        return (
            branch(positive, f"x < {lo}"),
            branch(negative, f"{lo} <= x < {params['d']}"),
        )
    if family in {"f2a", "f2b"}:
        a1, b1 = params["a1"], params["b1"]
        a2, b2 = params["a2"], params["b2"]
        t, value = params["t"], params["v"]
        return (
            branch(f"Eq({a1}*x + ({b1}), {value})", f"x < {t}"),
            branch(f"Eq({a2}*x + ({b2}), {value})", f"x >= {t}"),
        )
    r1, r2 = params["r1"], params["r2"]
    split = f"({r1} + {r2})/2"
    return (
        branch(f"Eq(x, {r1})", f"x < {split}"),
        branch(f"Eq(x, {r2})", f"x >= {split}"),
    )


def test_all_casework_v1_records_build_and_have_exact_atomic_case_coverage():
    paths = (
        REPO_ROOT / "datasets/casework-v1/train.jsonl",
        REPO_ROOT / "datasets/casework-v1/test.jsonl",
    )
    seen = 0
    for path in paths:
        for line in path.read_text().splitlines():
            record = json.loads(line)
            parent = build_casework_root_obligation(NoLabelAccess(record))
            coverage = validate_decomposition_coverage(
                parent, casework_children(record, parent)
            )
            assert coverage.accepted, (record["id"], coverage.reason_code)
            seen += 1
    assert seen == 300
