from dataclasses import replace

from verifier_anchored_rl.reasoning_v2 import (
    AggregateDraft,
    Claim,
    ClaimKind,
    ClaimStatus,
    Connective,
    DecompositionDraft,
    DerivationDraft,
    DischargeStatus,
    Err,
    Goal,
    GoalObjective,
    GoalObligation,
    MockRule,
    Ok,
    Reducer,
    RefutationDraft,
    RefutationMode,
    Relation,
    ResultDraft,
    SolutionDraft,
    TableDrivenVerifier,
    aggregate as _aggregate,
    commit_claim,
    decompose as _decompose,
    derive as _derive,
    direct_child_goals,
    get,
    is_active,
    is_committable,
    is_effectively_pruned,
    is_solved,
    new_graph as _new_graph,
    propose_refutation as _propose_refutation,
    propose_result as _propose_result,
    propose_solution as _propose_solution,
    prune,
    validate_graph,
    verify_claim,
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


def derive(graph, goal_id, draft):
    return _derive(graph, goal_id, draft, GENERATION)


def aggregate(graph, goal_id, draft):
    return _aggregate(graph, goal_id, draft, GENERATION)


def propose_refutation(graph, goal_id, draft):
    return _propose_refutation(graph, goal_id, draft, GENERATION)


def propose_result(graph, goal_id, draft):
    return _propose_result(graph, goal_id, draft, GENERATION)


def obligation(
    objective: GoalObjective = GoalObjective.IDENTITY,
    *,
    equation: str = "x = 5",
    domain: str = "real",
) -> GoalObligation:
    return GoalObligation("x", equation, domain, objective)


def graph_of(result) -> object:
    assert isinstance(result, Ok), getattr(result, "message", result)
    validate_graph(result.graph)
    return result.graph


def verifier(*rules: MockRule) -> TableDrivenVerifier:
    return TableDrivenVerifier(tuple(rules))


def test_explicit_commit_preserves_older_committable_claim():
    graph = new_graph("solve x", obligation(), ("trusted premise",))
    graph = graph_of(propose_solution(graph, 0, SolutionDraft("x", "5")))
    first_id = len(graph.nodes) - 1
    graph = graph_of(propose_solution(graph, 0, SolutionDraft("x", "6")))
    second_id = len(graph.nodes) - 1
    checks = verifier(
        MockRule(ClaimKind.SOLUTION, "x = 5", True, True),
        MockRule(ClaimKind.SOLUTION, "x = 6", True, True),
    )
    graph = graph_of(verify_claim(graph, first_id, checks))
    graph = graph_of(verify_claim(graph, second_id, checks))

    assert is_committable(graph, first_id)
    assert is_committable(graph, second_id)
    graph = graph_of(commit_claim(graph, first_id))

    assert get(graph, 0).committed_claim == first_id
    assert is_solved(graph, 0)
    assert isinstance(
        propose_solution(graph, 0, SolutionDraft("x", "7")), Err
    )


def test_evidence_is_never_a_claim_target_or_commit_outcome():
    graph = new_graph("p", obligation(), ("x is real",))

    assert isinstance(commit_claim(graph, 1), Err)
    assert isinstance(verify_claim(graph, 1, verifier()), Err)
    assert isinstance(prune(graph, 1), Err)


def test_verify_makes_validity_and_discharge_terminal_together():
    graph = new_graph("p", obligation())
    graph = graph_of(
        derive(graph, 0, DerivationDraft("2 + 3", Relation.EQ, "5"))
    )
    claim_id = len(graph.nodes) - 1
    graph = graph_of(
        verify_claim(
            graph,
            claim_id,
            verifier(
                MockRule(
                    ClaimKind.DERIVATION,
                    "2 + 3 = 5",
                    True,
                    False,
                )
            ),
        )
    )
    claim = get(graph, claim_id)

    assert claim.status is ClaimStatus.VERIFIED
    assert claim.discharge_status is DischargeStatus.REJECTED
    assert isinstance(commit_claim(graph, claim_id), Err)
    assert isinstance(verify_claim(graph, claim_id, verifier()), Err)


def test_failed_verify_leaves_discharge_unchecked_and_cannot_repeat():
    graph = new_graph("p", obligation())
    graph = graph_of(propose_solution(graph, 0, SolutionDraft("x", "9")))
    claim_id = len(graph.nodes) - 1
    graph = graph_of(
        verify_claim(
            graph,
            claim_id,
            verifier(MockRule(ClaimKind.SOLUTION, "x = 9", False, False)),
        )
    )
    claim = get(graph, claim_id)

    assert claim.status is ClaimStatus.REJECTED
    assert claim.discharge is None
    assert claim.discharge_status is DischargeStatus.UNCHECKED
    assert isinstance(verify_claim(graph, claim_id, verifier()), Err)


def test_exact_duplicate_solution_claim_is_rejected():
    graph = new_graph("p", obligation())
    draft = SolutionDraft("x", "5")
    graph = graph_of(propose_solution(graph, 0, draft))

    duplicate = propose_solution(graph, 0, draft)
    assert isinstance(duplicate, Err)
    assert duplicate.message == "duplicate claim"


def test_direct_result_is_typed_and_owner_scoped():
    graph = new_graph("sum", obligation(GoalObjective.SUM))
    graph = graph_of(
        propose_result(graph, 0, ResultDraft(GoalObjective.SUM, "2 + 3"))
    )
    claim = get(graph, len(graph.nodes) - 1)

    assert isinstance(claim, Claim)
    assert claim.kind is ClaimKind.RESULT
    assert claim.statement == "result(sum) = 2 + 3"
    mismatch = propose_result(
        graph, 0, ResultDraft(GoalObjective.MAX, "5")
    )
    assert isinstance(mismatch, Err)
    assert "objective differs" in mismatch.message


def test_decompose_requires_trusted_branch_obligations_and_is_one_shot():
    graph = new_graph("sum cases", obligation(GoalObjective.SUM))
    draft = DecompositionDraft(
        connective=Connective.EXHAUSTIVE_CASE,
        child_statements=("positive branch", "negative branch"),
        child_obligations=(
            obligation(GoalObjective.BRANCH, equation="x = 2"),
            obligation(GoalObjective.BRANCH, equation="x = -2"),
        ),
    )
    graph = graph_of(decompose(graph, 0, draft))
    children = direct_child_goals(graph, 0)

    assert [child.parent_goal for child in children] == [0, 0]
    assert [child.obligation.objective for child in children] == [
        GoalObjective.BRANCH,
        GoalObjective.BRANCH,
    ]
    assert isinstance(decompose(graph, 0, draft), Err)
    bad = DecompositionDraft(
        Connective.CONJUNCTIVE,
        ("child",),
        (obligation(GoalObjective.SUM),),
    )
    assert isinstance(decompose(new_graph("p", obligation()), 0, bad), Err)


def test_malformed_nested_draft_fields_fail_closed():
    graph = new_graph("p", obligation())
    malformed_decomposition = DecompositionDraft(
        Connective.CONJUNCTIVE,
        ([],),  # type: ignore[arg-type]
        (obligation(GoalObjective.BRANCH),),
    )
    malformed_derivation = DerivationDraft(
        "1",
        Relation.EQ,
        "1",
        ([],),  # type: ignore[arg-type]
    )

    decomposition = decompose(graph, 0, malformed_decomposition)
    derivation = derive(graph, 0, malformed_derivation)

    assert isinstance(decomposition, Err)
    assert "statements" in decomposition.message
    assert isinstance(derivation, Err)
    assert "premises" in derivation.message


def test_nonaggregate_claim_rejects_descendant_premise_before_child_prune():
    graph = new_graph("sum", obligation(GoalObjective.SUM))
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
    child_claim = len(graph.nodes) - 1

    dependent = derive(
        graph,
        0,
        DerivationDraft("1", Relation.EQ, "1", premises=(child_claim,)),
    )
    assert isinstance(dependent, Err)
    assert "owner ancestry" in dependent.message
    assert isinstance(prune(graph, 1), Ok)


def test_active_dependent_prevents_pruning_its_same_owner_premise():
    graph = new_graph("p", obligation())
    graph = graph_of(propose_solution(graph, 0, SolutionDraft("x", "5")))
    premise = len(graph.nodes) - 1
    graph = graph_of(
        derive(
            graph,
            0,
            DerivationDraft("1", Relation.EQ, "1", premises=(premise,)),
        )
    )

    result = prune(graph, premise)
    assert isinstance(result, Err)
    assert "premise" in result.message


def test_prune_is_root_safe_and_cascades_inactivity_without_deleting_history():
    graph = new_graph("p", obligation(GoalObjective.SUM))
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
    graph = graph_of(propose_solution(graph, 1, SolutionDraft("x", "5")))
    claim_id = len(graph.nodes) - 1
    node_count = len(graph.nodes)
    graph = graph_of(prune(graph, 1))

    assert len(graph.nodes) == node_count
    assert is_effectively_pruned(graph, claim_id)
    assert not is_active(graph, claim_id)
    assert isinstance(prune(graph, 0), Err)
    assert isinstance(prune(graph, claim_id), Err)


def test_aggregate_consumes_only_exact_committed_child_claim_premises():
    graph = new_graph("sum branches", obligation(GoalObjective.SUM))
    graph = graph_of(
        decompose(
            graph,
            0,
            DecompositionDraft(
                Connective.EXHAUSTIVE_CASE,
                ("branch one", "branch two"),
                (
                    obligation(GoalObjective.BRANCH, equation="x = 2"),
                    obligation(GoalObjective.BRANCH, equation="x = 3"),
                ),
            ),
        )
    )
    checks = verifier(
        MockRule(ClaimKind.SOLUTION, "x = 2", True, True),
        MockRule(ClaimKind.SOLUTION, "x = 3", True, True),
        MockRule(
            ClaimKind.AGGREGATE,
            "sum(committed child outcomes) = 5",
            True,
            True,
        ),
    )
    child_claims = []
    for child_id, value in ((1, "2"), (2, "3")):
        graph = graph_of(
            propose_solution(graph, child_id, SolutionDraft("x", value))
        )
        claim_id = len(graph.nodes) - 1
        child_claims.append(claim_id)
        graph = graph_of(verify_claim(graph, claim_id, checks))
        graph = graph_of(commit_claim(graph, claim_id))

    conjunctive = replace(
        graph,
        nodes=(
            replace(graph.nodes[0], connective=Connective.CONJUNCTIVE),
            *graph.nodes[1:],
        ),
    )
    invalid = aggregate(
        conjunctive,
        0,
        AggregateDraft(Reducer.SUM, tuple(child_claims), "5"),
    )
    assert isinstance(invalid, Err)
    assert "exhaustive-case" in invalid.message

    assert isinstance(
        aggregate(
            graph,
            0,
            AggregateDraft(Reducer.SUM, tuple(reversed(child_claims)), "5"),
        ),
        Err,
    )
    graph = graph_of(
        aggregate(
            graph, 0, AggregateDraft(Reducer.SUM, tuple(child_claims), "5")
        )
    )
    aggregate_id = len(graph.nodes) - 1
    aggregate_claim = get(graph, aggregate_id)
    assert isinstance(aggregate_claim, Claim)
    assert aggregate_claim.premises == tuple(child_claims)
    assert not hasattr(aggregate_claim.payload, "child_claims")
    graph = graph_of(verify_claim(graph, aggregate_id, checks))
    graph = graph_of(commit_claim(graph, aggregate_id))
    assert get(graph, 0).committed_claim == aggregate_id


def test_refutation_has_one_typed_mode_and_owner_scoped_variable():
    graph = new_graph("branch", obligation(GoalObjective.BRANCH, domain="x > 0"))
    graph = graph_of(
        propose_refutation(
            graph,
            0,
            RefutationDraft(
                RefutationMode.UNIQUE_CANDIDATE_OUTSIDE_DOMAIN,
                "x",
                "-5",
            ),
        )
    )
    assert get(graph, len(graph.nodes) - 1).statement == (
        "x = -5 is outside the owner domain"
    )
    wrong_variable = RefutationDraft(
        RefutationMode.UNIQUE_CANDIDATE_OUTSIDE_DOMAIN, "y", "-5"
    )
    assert isinstance(propose_refutation(graph, 0, wrong_variable), Err)
