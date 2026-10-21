import pytest

from verifier_anchored_rl.reasoning_v2 import (
    Action,
    ClaimKind,
    Connective,
    DecompositionDraft,
    GoalObjective,
    GoalObligation,
    MockRule,
    Ok,
    ResultDraft,
    SolutionDraft,
    State,
    TableDrivenVerifier,
    Verb,
    commit_claim,
    decompose as _decompose,
    new_graph as _new_graph,
    propose_result as _propose_result,
    propose_solution as _propose_solution,
    verify_claim,
)
from verifier_anchored_rl.reasoning_v2.menu import (
    ACTION_SEMANTICS_VERSION,
    ACTION_TARGET_KINDS,
    ACTION_VERB_ORDER,
    POLICY_VERB_ORDER,
    VERB_ORDER,
    available_actions,
    requires_internal_closure,
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


def obligation(
    objective: GoalObjective = GoalObjective.IDENTITY,
    *,
    equation: str = "x = 5",
) -> GoalObligation:
    return GoalObligation("x", equation, "Reals", objective)


def graph_of(result):
    assert isinstance(result, Ok), getattr(result, "message", result)
    return result.graph


def state_of(graph, turn: int = 0) -> State:
    return State(graph=graph, turn=turn)


def pairs(state: State, turn_limit: int = 16) -> set[tuple[Verb, int]]:
    return {
        (action.verb, action.target)
        for action in available_actions(state, turn_limit)
    }


def decompose_two(graph):
    return graph_of(
        decompose(
            graph,
            0,
            DecompositionDraft(
                Connective.EXHAUSTIVE_CASE,
                ("left", "right"),
                (
                    obligation(GoalObjective.BRANCH, equation="x = 2"),
                    obligation(GoalObjective.BRANCH, equation="x = 3"),
                ),
            ),
        )
    )


def solve_child(graph, child_id: int, value: str):
    graph = graph_of(
        propose_solution(graph, child_id, SolutionDraft("x", value))
    )
    claim_id = len(graph.nodes) - 1
    verifier = TableDrivenVerifier(
        (
            MockRule(
                ClaimKind.SOLUTION,
                f"x = {value}",
                verification_accepted=True,
                discharge_accepted=True,
            ),
        )
    )
    graph = graph_of(verify_claim(graph, claim_id, verifier))
    return graph_of(commit_claim(graph, claim_id)), claim_id


def test_action_contract_constants_are_exact_and_immutable():
    expected = (
        "decompose",
        "propose_result",
        "propose_solution",
        "derive",
        "aggregate",
        "propose_refutation",
        "verify",
        "commit",
        "prune",
    )
    assert ACTION_SEMANTICS_VERSION == "claim-closure-v2"
    assert ACTION_VERB_ORDER == expected
    assert tuple(verb.value for verb in VERB_ORDER) == expected
    assert POLICY_VERB_ORDER == (
        Verb.DECOMPOSE,
        Verb.PROPOSE_RESULT,
        Verb.PROPOSE_SOLUTION,
        Verb.PROPOSE_REFUTATION,
    )
    assert tuple(ACTION_TARGET_KINDS) == expected
    assert ACTION_TARGET_KINDS["propose_result"] == ("goal",)
    assert ACTION_TARGET_KINDS["verify"] == ("claim",)
    assert ACTION_TARGET_KINDS["prune"] == ("goal", "claim")
    with pytest.raises(TypeError):
        ACTION_TARGET_KINDS["verify"] = ("goal",)


def test_policy_menu_uses_only_public_objective_and_structure():
    identity = new_graph("identity", obligation())
    assert available_actions(state_of(identity), 16) == (
        Action(Verb.DECOMPOSE, 0),
        Action(Verb.PROPOSE_SOLUTION, 0),
    )

    aggregate = new_graph("sum", obligation(GoalObjective.SUM))
    assert available_actions(state_of(aggregate), 16) == (
        Action(Verb.DECOMPOSE, 0),
        Action(Verb.PROPOSE_RESULT, 0),
    )

    branch = new_graph("branch", obligation(GoalObjective.BRANCH))
    assert available_actions(state_of(branch), 16) == (
        Action(Verb.PROPOSE_SOLUTION, 0),
        Action(Verb.PROPOSE_REFUTATION, 0),
    )

    forbidden = {Verb.DERIVE, Verb.AGGREGATE, Verb.VERIFY, Verb.COMMIT, Verb.PRUNE}
    assert all(
        action.verb not in forbidden
        for graph in (identity, aggregate, branch)
        for action in available_actions(state_of(graph), 16)
    )


def test_pending_claims_are_internal_closure_not_policy_decisions():
    graph = new_graph("p", obligation())
    graph = graph_of(propose_solution(graph, 0, SolutionDraft("x", "6")))

    assert available_actions(state_of(graph), 16) == ()
    assert requires_internal_closure(state_of(graph))

    rejected = TableDrivenVerifier(
        (MockRule(ClaimKind.SOLUTION, "x = 6", False, False),)
    )
    graph = graph_of(verify_claim(graph, 1, rejected))
    assert not requires_internal_closure(state_of(graph))
    assert pairs(state_of(graph)) == {
        (Verb.DECOMPOSE, 0),
        (Verb.PROPOSE_SOLUTION, 0),
    }

    graph = graph_of(propose_solution(graph, 0, SolutionDraft("x", "5")))
    accepted_id = len(graph.nodes) - 1
    accepted = TableDrivenVerifier(
        (MockRule(ClaimKind.SOLUTION, "x = 5", True, True),)
    )
    graph = graph_of(verify_claim(graph, accepted_id, accepted))
    assert available_actions(state_of(graph), 16) == ()
    assert requires_internal_closure(state_of(graph))


def test_decomposition_exposes_only_result_fallback_and_atomic_case_actions():
    graph = decompose_two(new_graph("sum", obligation(GoalObjective.SUM)))

    assert pairs(state_of(graph)) == {
        (Verb.PROPOSE_RESULT, 0),
        (Verb.PROPOSE_SOLUTION, 1),
        (Verb.PROPOSE_REFUTATION, 1),
        (Verb.PROPOSE_SOLUTION, 2),
        (Verb.PROPOSE_REFUTATION, 2),
    }


def test_fully_committed_children_require_internal_aggregate_closure():
    graph = decompose_two(new_graph("sum", obligation(GoalObjective.SUM)))
    graph, _ = solve_child(graph, 1, "2")
    assert not requires_internal_closure(state_of(graph))

    graph, _ = solve_child(graph, 2, "3")
    assert requires_internal_closure(state_of(graph))
    assert available_actions(state_of(graph), 16) == ()


def test_direct_result_is_an_alternative_to_decomposition():
    graph = new_graph("sum", obligation(GoalObjective.SUM))
    graph = graph_of(
        propose_result(graph, 0, ResultDraft(GoalObjective.SUM, "5"))
    )

    assert requires_internal_closure(state_of(graph))
    assert available_actions(state_of(graph), 16) == ()


def test_menu_enforces_horizon_and_root_terminal_state():
    graph = new_graph("p", obligation())
    assert available_actions(state_of(graph, turn=4), 4) == ()
    assert available_actions(state_of(graph, turn=5), 4) == ()

    graph = graph_of(propose_solution(graph, 0, SolutionDraft("x", "5")))
    verifier = TableDrivenVerifier(
        (MockRule(ClaimKind.SOLUTION, "x = 5", True, True),)
    )
    graph = graph_of(verify_claim(graph, 1, verifier))
    graph = graph_of(commit_claim(graph, 1))
    assert available_actions(state_of(graph), 16) == ()


@pytest.mark.parametrize("turn_limit", [-1, True, 1.5, None])
def test_menu_rejects_invalid_horizons(turn_limit):
    with pytest.raises(ValueError, match="turn_limit"):
        available_actions(state_of(new_graph("p", obligation())), turn_limit)


def test_menu_rejects_non_state_inputs():
    with pytest.raises(TypeError, match="State"):
        available_actions(new_graph("p", obligation()), 16)

