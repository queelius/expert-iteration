from dataclasses import FrozenInstanceError, replace

import pytest

from verifier_anchored_rl.reasoning_v2 import (
    Action,
    AggregatePayload,
    CheckPurpose,
    Claim,
    ClaimKind,
    Connective,
    DerivationPayload,
    Evidence,
    Goal,
    GoalObjective,
    GoalObligation,
    Graph,
    GraphValidationError,
    OutcomeCode,
    Reducer,
    RefutationMode,
    RefutationPayload,
    RejectionReason,
    Relation,
    ResultPayload,
    SolutionPayload,
    State,
    SymbolicCheck,
    TableDrivenVerifier,
    TransitionOutcome,
    Verb,
    VerifierContext,
    VerificationDecision,
    claim_statement,
    is_active,
    new_graph as _new_graph,
    validate_graph,
)


GENERATION = TableDrivenVerifier(())


def new_graph(problem, obligation, givens=()):
    return _new_graph(
        problem,
        obligation,
        givens,
        generation_contract=GENERATION,
    )


def obligation(
    objective: GoalObjective = GoalObjective.IDENTITY,
    *,
    equation: str = "x = 5",
    domain: str = "real",
) -> GoalObligation:
    return GoalObligation(
        variable="x",
        equation=equation,
        domain=domain,
        objective=objective,
    )


def test_new_graph_has_trusted_obligation_and_noncommittable_evidence():
    root_obligation = obligation(GoalObjective.MAX, equation="abs(x) = 5")
    graph = new_graph("find the largest x", root_obligation, ("x >= 0",))

    root, given = graph.nodes
    assert isinstance(root, Goal)
    assert root.obligation == root_obligation
    assert isinstance(given, Evidence)
    assert given.source == "trusted_given"
    assert given.committable is False
    assert is_active(graph, given.id)
    assert "answer" not in GoalObligation.__dataclass_fields__
    assert "solution_set" not in GoalObligation.__dataclass_fields__
    validate_graph(graph)


def test_claim_statement_is_derived_from_authoritative_typed_payload():
    payloads = [
        (ResultPayload(GoalObjective.SUM, "8"), "result(sum) = 8"),
        (SolutionPayload("x", "5"), "x = 5"),
        (DerivationPayload("2 + 3", Relation.EQ, "5"), "2 + 3 = 5"),
        (
            AggregatePayload(Reducer.SUM, "8"),
            "sum(committed child outcomes) = 8",
        ),
        (
            RefutationPayload(
                RefutationMode.UNIQUE_CANDIDATE_OUTSIDE_DOMAIN, "x", "-5"
            ),
            "x = -5 is outside the owner domain",
        ),
    ]

    for payload, expected in payloads:
        assert claim_statement(payload) == expected
        fields = Claim.__dataclass_fields__
        assert "statement" not in fields


def test_graph_and_nodes_are_frozen():
    graph = new_graph("p", obligation())

    with pytest.raises(FrozenInstanceError):
        graph.root_id = 2
    with pytest.raises(FrozenInstanceError):
        graph.nodes[0].statement = "changed"


def _check(purpose: CheckPurpose, *, accepted: bool) -> SymbolicCheck:
    return SymbolicCheck(
        context=VerifierContext(
            contract_id="test-v1",
            purpose=purpose,
            claim_kind=ClaimKind.SOLUTION,
            owner_goal=0,
            context_digest=f"digest-{purpose.value}",
        ),
        accepted=accepted,
        observed="x = 5",
        reason_code=None if accepted else "rejected",
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda graph: replace(
                graph, nodes=(graph.nodes[0], Evidence(3, 0, "given"))
            ),
            "dense",
        ),
        (
            lambda graph: replace(
                graph,
                nodes=(replace(graph.nodes[0], pruned=True),),
            ),
            "root goal cannot be pruned",
        ),
        (
            lambda graph: replace(
                graph,
                nodes=graph.nodes
                + (
                    Claim(
                        id=1,
                        owner_goal=0,
                        kind=ClaimKind.SOLUTION,
                        payload=DerivationPayload("2 + 3", Relation.EQ, "5"),
                    ),
                ),
            ),
            "payload",
        ),
        (
            lambda graph: replace(
                graph,
                nodes=graph.nodes
                + (
                    Claim(
                        id=1,
                        owner_goal=0,
                        kind=ClaimKind.SOLUTION,
                        payload=SolutionPayload("y", "5"),
                    ),
                ),
            ),
            "variable differs",
        ),
        (
            lambda graph: replace(
                graph,
                nodes=graph.nodes
                + (
                    Claim(
                        id=1,
                        owner_goal=0,
                        kind=ClaimKind.SOLUTION,
                        payload=SolutionPayload("x", "5"),
                        verification=_check(CheckPurpose.VALIDITY, accepted=True),
                    ),
                ),
            ),
            "accepted validity",
        ),
        (
            lambda graph: replace(
                graph,
                nodes=graph.nodes
                + (
                    Claim(
                        id=1,
                        owner_goal=0,
                        kind=ClaimKind.SOLUTION,
                        payload=SolutionPayload("x", "5"),
                        verification=_check(CheckPurpose.VALIDITY, accepted=False),
                        discharge=_check(
                            CheckPurpose.OWNER_DISCHARGE, accepted=True
                        ),
                    ),
                ),
            ),
            "rejected validity",
        ),
    ],
)
def test_validate_graph_rejects_invariant_violations(mutate, message):
    graph = new_graph("p", obligation())

    with pytest.raises(GraphValidationError, match=message):
        validate_graph(mutate(graph))


def test_validate_graph_rejects_nonaggregate_descendant_premise():
    graph = new_graph("p", obligation())
    root = replace(graph.nodes[0], connective=Connective.CONJUNCTIVE)
    child = Goal(
        id=1,
        parent_goal=0,
        statement="child",
        obligation=obligation(GoalObjective.BRANCH),
    )
    child_claim = Claim(
        id=2,
        owner_goal=1,
        kind=ClaimKind.SOLUTION,
        payload=SolutionPayload("x", "5"),
    )
    root_claim = Claim(
        id=3,
        owner_goal=0,
        kind=ClaimKind.DERIVATION,
        payload=DerivationPayload("1", Relation.EQ, "1"),
        premises=(2,),
    )

    with pytest.raises(GraphValidationError, match="owner ancestry"):
        validate_graph(
            replace(graph, nodes=(root, child, child_claim, root_claim))
        )


def test_state_owns_a_bounded_previous_outcome():
    graph = new_graph("p", obligation())
    action = Action(Verb.PROPOSE_SOLUTION, 0)
    outcome = TransitionOutcome(
        action=action,
        code=OutcomeCode.REJECTED,
        reason=RejectionReason.GENERATION_ERROR,
    )

    state = State(graph=graph, turn=1, previous_outcome=outcome)
    assert state.previous_outcome == outcome
    with pytest.raises(ValueError, match="requires"):
        TransitionOutcome(action, OutcomeCode.REJECTED)
    with pytest.raises(ValueError, match="cannot"):
        TransitionOutcome(
            action,
            OutcomeCode.APPLIED,
            RejectionReason.STALE_ACTION,
        )


def test_transition_outcome_context_is_exactly_for_applied_verify():
    verify = Action(Verb.VERIFY, 1)
    context = VerifierContext(
        contract_id="test-v1",
        purpose=CheckPurpose.VALIDITY,
        claim_kind=ClaimKind.SOLUTION,
        owner_goal=0,
        context_digest="digest-validity",
    )

    with pytest.raises(ValueError, match="required exactly"):
        TransitionOutcome(verify, OutcomeCode.APPLIED)
    with pytest.raises(ValueError, match="required exactly"):
        TransitionOutcome(
            Action(Verb.COMMIT, 1),
            OutcomeCode.APPLIED,
            verifier_context=context,
        )

    assert (
        TransitionOutcome(
            verify,
            OutcomeCode.APPLIED,
            verifier_context=context,
        ).verifier_context
        == context
    )


def test_validity_and_discharge_receipts_require_one_verifier_contract():
    validity = _check(CheckPurpose.VALIDITY, accepted=True)
    discharge = _check(CheckPurpose.OWNER_DISCHARGE, accepted=True)
    discharge = replace(
        discharge,
        context=replace(discharge.context, contract_id="other-contract"),
    )

    with pytest.raises(ValueError, match="same verifier contract"):
        VerificationDecision(validity, discharge)

    graph = new_graph("p", obligation())
    claim = Claim(
        id=1,
        owner_goal=0,
        kind=ClaimKind.SOLUTION,
        payload=SolutionPayload("x", "5"),
        verification=validity,
        discharge=discharge,
    )
    with pytest.raises(GraphValidationError, match="contracts differ"):
        validate_graph(replace(graph, nodes=graph.nodes + (claim,)))


@pytest.mark.parametrize(
    "build",
    [
        lambda graph: replace(
            graph,
            nodes=(replace(graph.nodes[0], assumptions=["mutable"]),),
        ),
        lambda graph: replace(
            graph,
            nodes=graph.nodes
            + (Evidence(id=1, scope_goal=0, statement="given", premises=[]),),
        ),
        lambda graph: replace(graph, root_id=True),
    ],
)
def test_validate_graph_rejects_mutable_or_ambiguous_structural_fields(build):
    with pytest.raises(GraphValidationError):
        validate_graph(build(new_graph("p", obligation())))


def test_validate_graph_requires_connective_and_typed_branch_children():
    graph = new_graph("p", obligation())
    root = graph.nodes[0]

    with pytest.raises(GraphValidationError, match="connective and children"):
        validate_graph(
            replace(
                graph,
                nodes=(replace(root, connective=Connective.CONJUNCTIVE),),
            )
        )

    invalid_child = Goal(
        id=1,
        parent_goal=0,
        statement="child",
        obligation=obligation(GoalObjective.SUM),
    )
    with pytest.raises(GraphValidationError, match="objective must be branch"):
        validate_graph(
            replace(
                graph,
                nodes=(
                    replace(root, connective=Connective.EXHAUSTIVE_CASE),
                    invalid_child,
                ),
            )
        )


def test_action_outcome_and_state_reject_runtime_enum_or_id_spoofing():
    graph = new_graph("p", obligation())

    with pytest.raises(ValueError, match="verb"):
        Action("verify", 0)
    with pytest.raises(ValueError, match="target"):
        Action(Verb.VERIFY, True)
    with pytest.raises(ValueError, match="code"):
        TransitionOutcome(Action(Verb.VERIFY, 0), "applied")
    with pytest.raises(ValueError, match="turn"):
        State(graph, True)
