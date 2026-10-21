from dataclasses import replace

import pytest

from verifier_anchored_rl.reasoning_v2 import (
    Action,
    AggregatePayload,
    AggregatePayloadBuilder,
    Claim,
    ClaimKind,
    Connective,
    DecompositionDraft,
    DerivationDraft,
    Goal,
    GoalObjective,
    GoalObligation,
    GoalStatus,
    MacroTransitionError,
    MockRule,
    Ok,
    OutcomeCode,
    ObservationContext,
    PolicyGenerationError,
    PolicyPayloadError,
    Reducer,
    RefutationDraft,
    RefutationMode,
    RejectionReason,
    Relation,
    ResultDraft,
    SolutionDraft,
    SolutionPayload,
    State,
    TableDrivenVerifier,
    Verb,
    aggregate_ready,
    build_observation,
    commit_claim,
    decompose,
    derive,
    execute_learned_action,
    get,
    macro_step,
    new_graph,
    propose_solution,
    requires_internal_closure,
    run_closure,
    validate_macro_transition,
    validate_observation_with_contract,
    verify_claim,
)
from verifier_anchored_rl.reasoning_v2.closure import ClosureExecutionError
from verifier_anchored_rl.reasoning_v2.math_adapter import CaseworkMathVerifier


ENVIRONMENT_DIGEST = "b" * 64


class StaticGenerator:
    def __init__(self, value=None, *, error=None):
        self.value = value
        self.error = error
        self.calls = []

    def generate(self, state, action):
        self.calls.append((state.turn, action))
        if self.error is not None:
            raise self.error
        return self.value


class QueueGenerator:
    def __init__(self, *values):
        self.values = list(values)
        self.calls = []

    def generate(self, state, action):
        self.calls.append((state.turn, action))
        return self.values.pop(0)


class CountingVerifier:
    def __init__(self, delegate):
        self.delegate = delegate
        self.calls = []

    @property
    def contract_id(self):
        return self.delegate.contract_id

    @property
    def validity_rejection_codes(self):
        return self.delegate.validity_rejection_codes

    @property
    def discharge_rejection_codes(self):
        return self.delegate.discharge_rejection_codes

    def verify(self, graph, claim):
        self.calls.append(claim.id)
        return self.delegate.verify(graph, claim)


class StaticAggregateBuilder:
    aggregate_contract_id = "static-aggregate-v1"

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def build_aggregate_payload(self, graph, goal):
        self.calls.append(goal.id)
        return self.payload


def obligation(
    objective=GoalObjective.IDENTITY,
    *,
    equation="x = 5",
    domain="Reals",
):
    return GoalObligation("x", equation, domain, objective)


def graph_of(result):
    assert isinstance(result, Ok), getattr(result, "message", result)
    return result.graph


def event_tuples(transition):
    return tuple(
        (event.outcome.action.verb, event.outcome.action.target, event.claim_id)
        for event in transition.closure_events
    )


def test_direct_result_closes_on_final_turn_before_horizon_failure():
    verifier = TableDrivenVerifier((
        MockRule(ClaimKind.RESULT, "result(sum) = 5", True, True),
    ))
    graph = new_graph(
        "sum the solutions",
        obligation(GoalObjective.SUM),
        generation_contract=verifier,
    )
    generator = StaticGenerator(ResultDraft(GoalObjective.SUM, "5"))

    transition = macro_step(
        State(graph, turn=0),
        Action(Verb.PROPOSE_RESULT, 0),
        turn_limit=1,
        environment_contract_sha256=ENVIRONMENT_DIGEST,
        generator=generator,
        generation_contract=verifier,
        verifier=verifier,
    )

    assert transition.outcome.code is OutcomeCode.APPLIED
    assert event_tuples(transition) == (
        (Verb.VERIFY, 1, 1),
        (Verb.COMMIT, 1, 1),
    )
    assert transition.reward == 1
    assert transition.terminal
    assert transition.next_state.turn == 1
    assert transition.next_state.previous_outcome == transition.outcome
    assert transition.next_state.previous_outcome != (
        transition.closure_events[-1].outcome
    )
    root = get(transition.next_state.graph, 0)
    claim = get(transition.next_state.graph, 1)
    assert isinstance(root, Goal) and root.status is GoalStatus.SOLVED
    assert isinstance(claim, Claim) and claim.verification is not None
    assert (
        transition.closure_events[0].outcome.verifier_context
        == claim.verification.context
    )
    assert len(generator.calls) == 1


@pytest.mark.parametrize(
    ("turn_limit", "expected_reward", "expected_terminal"),
    [(2, 0, False), (1, 0, True)],
)
def test_symbolically_rejected_result_only_fails_at_horizon(
    turn_limit, expected_reward, expected_terminal
):
    verifier = TableDrivenVerifier((
        MockRule(ClaimKind.RESULT, "result(sum) = 6", False),
    ))
    graph = new_graph(
        "sum the solutions",
        obligation(GoalObjective.SUM),
        generation_contract=verifier,
    )

    transition = macro_step(
        State(graph, turn=0),
        Action(Verb.PROPOSE_RESULT, 0),
        turn_limit=turn_limit,
        environment_contract_sha256=ENVIRONMENT_DIGEST,
        generator=StaticGenerator(ResultDraft(GoalObjective.SUM, "6")),
        generation_contract=verifier,
        verifier=verifier,
    )

    assert transition.outcome.code is OutcomeCode.APPLIED
    assert event_tuples(transition) == ((Verb.VERIFY, 1, 1),)
    assert transition.reward == expected_reward
    assert transition.terminal is expected_terminal
    root = get(transition.next_state.graph, 0)
    claim = get(transition.next_state.graph, 1)
    assert isinstance(root, Goal) and root.status is GoalStatus.OPEN
    assert isinstance(claim, Claim) and claim.status.value == "rejected"
    assert not requires_internal_closure(transition.next_state)


@pytest.mark.parametrize(
    ("action", "generator", "reason", "expected_calls"),
    [
        (
            Action(Verb.PROPOSE_SOLUTION, 0),
            StaticGenerator(error=PolicyGenerationError("provider failed")),
            RejectionReason.GENERATION_ERROR,
            1,
        ),
        (
            Action(Verb.PROPOSE_SOLUTION, 0),
            StaticGenerator(error=PolicyPayloadError("bad schema")),
            RejectionReason.PAYLOAD_ERROR,
            1,
        ),
        (
            Action(Verb.PROPOSE_SOLUTION, 0),
            StaticGenerator(ResultDraft(GoalObjective.SUM, "5")),
            RejectionReason.PAYLOAD_ERROR,
            1,
        ),
        (
            Action(Verb.PROPOSE_SOLUTION, 0),
            StaticGenerator(SolutionDraft("y", "5")),
            RejectionReason.ALGEBRA_REJECTED,
            1,
        ),
    ],
)
def test_rejected_learned_actions_consume_one_turn_without_closure(
    action, generator, reason, expected_calls
):
    contract = TableDrivenVerifier(())
    graph = new_graph(
        "solve",
        obligation(),
        generation_contract=contract,
    )

    transition = macro_step(
        State(graph, turn=0),
        action,
        turn_limit=2,
        environment_contract_sha256=ENVIRONMENT_DIGEST,
        generator=generator,
        generation_contract=contract,
        verifier=contract,
    )

    assert transition.outcome.code is OutcomeCode.REJECTED
    assert transition.outcome.reason is reason
    assert transition.next_state.graph == graph
    assert transition.next_state.turn == 1
    assert transition.closure_events == ()
    assert transition.reward == 0
    assert not transition.terminal
    assert len(generator.calls) == expected_calls


def test_stale_action_aborts_before_generation_or_canonical_transition():
    contract = TableDrivenVerifier(())
    graph = new_graph(
        "solve",
        obligation(),
        generation_contract=contract,
    )
    generator = StaticGenerator(SolutionDraft("x", "5"))

    with pytest.raises(MacroTransitionError, match="current learned menu"):
        macro_step(
            State(graph, turn=0),
            Action(Verb.PROPOSE_SOLUTION, 99),
            turn_limit=2,
            environment_contract_sha256=ENVIRONMENT_DIGEST,
            generator=generator,
            generation_contract=contract,
            verifier=contract,
        )
    assert generator.calls == []


def test_unexpected_generator_bug_aborts_instead_of_becoming_data():
    contract = TableDrivenVerifier(())
    graph = new_graph(
        "solve",
        obligation(),
        generation_contract=contract,
    )

    with pytest.raises(AssertionError, match="implementation bug"):
        macro_step(
            State(graph, turn=0),
            Action(Verb.PROPOSE_SOLUTION, 0),
            turn_limit=2,
            environment_contract_sha256=ENVIRONMENT_DIGEST,
            generator=StaticGenerator(
                error=AssertionError("implementation bug")
            ),
            generation_contract=contract,
            verifier=contract,
        )


def test_public_learned_executor_enforces_the_stable_legal_menu():
    contract = TableDrivenVerifier(())
    graph = new_graph(
        "sum",
        obligation(GoalObjective.SUM),
        generation_contract=contract,
    )

    with pytest.raises(MacroTransitionError, match="current learned menu"):
        execute_learned_action(
            State(graph, turn=0),
            Action(Verb.PROPOSE_SOLUTION, 0),
            turn_limit=2,
            generator=StaticGenerator(SolutionDraft("x", "5")),
            generation_contract=contract,
        )


def test_triggering_claim_verifies_before_older_pending_claims():
    generation = TableDrivenVerifier(())
    rules = TableDrivenVerifier((
        MockRule(ClaimKind.SOLUTION, "x = 5", True, True),
        MockRule(ClaimKind.DERIVATION, "1 = 1", True, False),
    ))
    verifier = CountingVerifier(rules)
    graph = new_graph(
        "solve",
        obligation(),
        generation_contract=generation,
    )
    graph = graph_of(
        propose_solution(graph, 0, SolutionDraft("x", "5"), generation)
    )
    graph = graph_of(
        derive(
            graph,
            0,
            DerivationDraft("1", Relation.EQ, "1"),
            generation,
        )
    )

    closed = run_closure(
        graph,
        triggering_claim_id=2,
        verifier=verifier,
        generation_contract=generation,
    )

    assert verifier.calls == [2, 1]
    assert tuple(
        (event.outcome.action.verb, event.claim_id)
        for event in closed.events
    ) == (
        (Verb.VERIFY, 2),
        (Verb.VERIFY, 1),
        (Verb.COMMIT, 1),
    )
    root = get(closed.graph, 0)
    assert isinstance(root, Goal) and root.status is GoalStatus.SOLVED


def test_bad_deterministic_aggregate_aborts_instead_of_retrying():
    generation = TableDrivenVerifier(())
    verifier = TableDrivenVerifier((
        MockRule(ClaimKind.SOLUTION, "x = 2", True, True),
        MockRule(ClaimKind.SOLUTION, "x = 3", True, True),
        MockRule(
            ClaimKind.AGGREGATE,
            "sum(committed child outcomes) = 5",
            False,
        ),
    ))
    graph = new_graph(
        "sum",
        obligation(GoalObjective.SUM),
        generation_contract=generation,
    )
    graph = graph_of(
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
            generation,
        )
    )
    for child_id, value in ((1, "2"), (2, "3")):
        graph = graph_of(
            propose_solution(
                graph,
                child_id,
                SolutionDraft("x", value),
                generation,
            )
        )
        claim_id = len(graph.nodes) - 1
        graph = graph_of(verify_claim(graph, claim_id, verifier))
        graph = graph_of(commit_claim(graph, claim_id))
    builder = StaticAggregateBuilder(AggregatePayload(Reducer.SUM, "5"))

    with pytest.raises(
        ClosureExecutionError,
        match="deterministic Aggregate failed symbolic discharge",
    ):
        run_closure(
            graph,
            triggering_claim_id=None,
            verifier=verifier,
            generation_contract=generation,
            aggregate_builder=builder,
        )
    assert builder.calls == [0]


def test_closure_rejects_a_preverified_committable_input():
    first = TableDrivenVerifier(
        (MockRule(ClaimKind.SOLUTION, "x = 5", True, True),),
        contract_id="verifier-a",
    )
    second = TableDrivenVerifier((), contract_id="verifier-b")
    graph = new_graph(
        "solve",
        obligation(),
        generation_contract=first,
    )
    graph = graph_of(
        propose_solution(graph, 0, SolutionDraft("x", "5"), first)
    )
    graph = graph_of(verify_claim(graph, 1, first))

    with pytest.raises(
        ClosureExecutionError,
        match="preverified committable Claim requires transition replay",
    ):
        run_closure(
            graph,
            triggering_claim_id=None,
            verifier=second,
            generation_contract=second,
        )


def test_casework_chain_builds_aggregate_without_extra_generator_call():
    pytest.importorskip("sympy")
    contract = CaseworkMathVerifier()
    assert isinstance(contract, AggregatePayloadBuilder)
    parent = obligation(
        GoalObjective.SUM,
        equation="Eq((x - 2)*(x - 3), 0)",
    )
    graph = new_graph("sum", parent, generation_contract=contract)
    generator = QueueGenerator(
        DecompositionDraft(
            Connective.EXHAUSTIVE_CASE,
            ("low case", "high case"),
            (
                obligation(
                    GoalObjective.BRANCH,
                    equation="Eq((x - 2)*(x - 3), 0)",
                    domain="x < 5/2",
                ),
                obligation(
                    GoalObjective.BRANCH,
                    equation="Eq((x - 2)*(x - 3), 0)",
                    domain="x >= 5/2",
                ),
            ),
        ),
        SolutionDraft("x", "2"),
        SolutionDraft("x", "3"),
    )

    first = macro_step(
        State(graph, turn=0),
        Action(Verb.DECOMPOSE, 0),
        turn_limit=6,
        environment_contract_sha256=ENVIRONMENT_DIGEST,
        generator=generator,
        generation_contract=contract,
        verifier=contract,
        aggregate_builder=contract,
    )
    assert first.closure_events == ()
    second = macro_step(
        first.next_state,
        Action(Verb.PROPOSE_SOLUTION, 1),
        turn_limit=6,
        environment_contract_sha256=ENVIRONMENT_DIGEST,
        generator=generator,
        generation_contract=contract,
        verifier=contract,
        aggregate_builder=contract,
    )
    assert event_tuples(second) == (
        (Verb.VERIFY, 3, 3),
        (Verb.COMMIT, 3, 3),
    )
    third = macro_step(
        second.next_state,
        Action(Verb.PROPOSE_SOLUTION, 2),
        turn_limit=6,
        environment_contract_sha256=ENVIRONMENT_DIGEST,
        generator=generator,
        generation_contract=contract,
        verifier=contract,
        aggregate_builder=contract,
    )

    assert event_tuples(third) == (
        (Verb.VERIFY, 4, 4),
        (Verb.COMMIT, 4, 4),
        (Verb.AGGREGATE, 0, 5),
        (Verb.VERIFY, 5, 5),
        (Verb.COMMIT, 5, 5),
    )
    assert third.reward == 1
    assert third.terminal
    assert len(generator.calls) == 3
    aggregate_claim = get(third.next_state.graph, 5)
    assert isinstance(aggregate_claim, Claim)
    assert aggregate_claim.kind is ClaimKind.AGGREGATE
    assert aggregate_claim.premises == (3, 4)
    assert aggregate_claim.payload == AggregatePayload(Reducer.SUM, "5")
    root = get(third.next_state.graph, 0)
    assert isinstance(root, Goal)
    assert not aggregate_ready(third.next_state.graph, root)
    assert not requires_internal_closure(third.next_state)
    observation = build_observation(
        third.next_state,
        ObservationContext(6, ENVIRONMENT_DIGEST),
    )
    validate_observation_with_contract(
        observation,
        contract,
        contract,
        environment_contract_sha256=ENVIRONMENT_DIGEST,
    )


def test_casework_rejects_a_conjunctive_case_partition():
    pytest.importorskip("sympy")
    contract = CaseworkMathVerifier()
    graph = new_graph(
        "sum",
        obligation(
            GoalObjective.SUM,
            equation="Eq((x - 2)*(x - 3), 0)",
        ),
        generation_contract=contract,
    )
    generator = StaticGenerator(
        DecompositionDraft(
            Connective.CONJUNCTIVE,
            ("low case", "high case"),
            (
                obligation(
                    GoalObjective.BRANCH,
                    equation="Eq((x - 2)*(x - 3), 0)",
                    domain="x < 5/2",
                ),
                obligation(
                    GoalObjective.BRANCH,
                    equation="Eq((x - 2)*(x - 3), 0)",
                    domain="x >= 5/2",
                ),
            ),
        )
    )

    transition = macro_step(
        State(graph, turn=0),
        Action(Verb.DECOMPOSE, 0),
        turn_limit=4,
        environment_contract_sha256=ENVIRONMENT_DIGEST,
        generator=generator,
        generation_contract=contract,
        verifier=contract,
        aggregate_builder=contract,
    )

    assert transition.outcome.code is OutcomeCode.REJECTED
    assert transition.outcome.reason is RejectionReason.ALGEBRA_REJECTED
    assert transition.next_state.graph == graph


def test_casework_aggregate_omits_a_symbolically_refuted_branch():
    pytest.importorskip("sympy")
    contract = CaseworkMathVerifier()
    graph = new_graph(
        "sum",
        obligation(GoalObjective.SUM, equation="Eq(x, 2)"),
        generation_contract=contract,
    )
    generator = QueueGenerator(
        DecompositionDraft(
            Connective.EXHAUSTIVE_CASE,
            ("negative case", "nonnegative case"),
            (
                obligation(
                    GoalObjective.BRANCH,
                    equation="Eq(x, 2)",
                    domain="x < 0",
                ),
                obligation(
                    GoalObjective.BRANCH,
                    equation="Eq(x, 2)",
                    domain="x >= 0",
                ),
            ),
        ),
        RefutationDraft(
            RefutationMode.UNIQUE_CANDIDATE_OUTSIDE_DOMAIN,
            "x",
            "2",
        ),
        SolutionDraft("x", "2"),
    )

    first = macro_step(
        State(graph, turn=0),
        Action(Verb.DECOMPOSE, 0),
        turn_limit=6,
        environment_contract_sha256=ENVIRONMENT_DIGEST,
        generator=generator,
        generation_contract=contract,
        verifier=contract,
        aggregate_builder=contract,
    )
    second = macro_step(
        first.next_state,
        Action(Verb.PROPOSE_REFUTATION, 1),
        turn_limit=6,
        environment_contract_sha256=ENVIRONMENT_DIGEST,
        generator=generator,
        generation_contract=contract,
        verifier=contract,
        aggregate_builder=contract,
    )
    third = macro_step(
        second.next_state,
        Action(Verb.PROPOSE_SOLUTION, 2),
        turn_limit=6,
        environment_contract_sha256=ENVIRONMENT_DIGEST,
        generator=generator,
        generation_contract=contract,
        verifier=contract,
        aggregate_builder=contract,
    )

    assert third.reward == 1
    assert third.terminal
    aggregate_claim = get(third.next_state.graph, 5)
    assert isinstance(aggregate_claim, Claim)
    assert aggregate_claim.payload == AggregatePayload(Reducer.SUM, "2")
    assert aggregate_claim.premises == (3, 4)
    assert len(generator.calls) == 3


def test_macro_transition_rejects_tampered_reward_and_previous_outcome():
    verifier = TableDrivenVerifier((
        MockRule(ClaimKind.SOLUTION, "x = 5", True, True),
    ))
    graph = new_graph(
        "solve",
        obligation(),
        generation_contract=verifier,
    )
    transition = macro_step(
        State(graph, turn=0),
        Action(Verb.PROPOSE_SOLUTION, 0),
        turn_limit=2,
        environment_contract_sha256=ENVIRONMENT_DIGEST,
        generator=StaticGenerator(SolutionDraft("x", "5")),
        generation_contract=verifier,
        verifier=verifier,
    )
    validate_macro_transition(
        transition,
        environment_contract_sha256=ENVIRONMENT_DIGEST,
        generation_contract=verifier,
        verifier=verifier,
    )

    with pytest.raises(ValueError, match="reward predicate"):
        replace(transition, reward=0)
    wrong_next = replace(
        transition.next_state,
        previous_outcome=transition.closure_events[-1].outcome,
    )
    with pytest.raises(ValueError, match="learned action outcome"):
        replace(transition, next_state=wrong_next)

    missing_events = replace(transition, closure_events=())
    with pytest.raises(MacroTransitionError, match="closure events differ"):
        validate_macro_transition(
            missing_events,
            environment_contract_sha256=ENVIRONMENT_DIGEST,
            generation_contract=verifier,
            verifier=verifier,
        )

    claim = get(transition.next_state.graph, 1)
    assert isinstance(claim, Claim)
    forged_claim = replace(claim, payload=SolutionPayload("x", "6"))
    forged_graph = replace(
        transition.next_state.graph,
        nodes=(transition.next_state.graph.nodes[0], forged_claim),
    )
    forged = replace(
        transition,
        next_state=replace(transition.next_state, graph=forged_graph),
    )
    with pytest.raises(MacroTransitionError, match="next graph differs"):
        validate_macro_transition(
            forged,
            environment_contract_sha256=ENVIRONMENT_DIGEST,
            generation_contract=verifier,
            verifier=verifier,
        )
