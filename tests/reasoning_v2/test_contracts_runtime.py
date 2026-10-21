from dataclasses import replace

import pytest

from verifier_anchored_rl.reasoning_v2 import (
    Action,
    BoundEnvironment,
    GoalObjective,
    GoalObligation,
    MacroTransitionError,
    ObservationContext,
    OutcomeCode,
    RejectionReason,
    ResultDraft,
    SolutionDraft,
    State,
    TransitionOutcome,
    Verb,
    build_environment_contract,
    build_observation,
    new_graph,
    validate_observation_with_contract,
    verifier_reason_codes_sha256,
)
from verifier_anchored_rl.reasoning_v2.lenses import (
    GENERATION_SCHEMA_VERSION,
    LENS_CONTRACT_VERSION,
)
from verifier_anchored_rl.reasoning_v2.math_adapter import (
    CaseworkMathVerifier,
)


class StaticGenerator:
    def __init__(self, draft):
        self.draft = draft

    def generate(self, state, action):
        return self.draft


def environment(*, temperature=0.0, turn_limit=4):
    adapter = CaseworkMathVerifier()
    contract = build_environment_contract(
        generation_contract=adapter,
        verifier=adapter,
        aggregate_builder=adapter,
        lens_contract=LENS_CONTRACT_VERSION,
        generation_schema=GENERATION_SCHEMA_VERSION,
        generation_temperature=temperature,
        generation_max_tokens=96,
        generation_seed_policy="none-v1",
        turn_limit=turn_limit,
    )
    return BoundEnvironment(contract, adapter, adapter, adapter)


def root(env, objective=GoalObjective.SUM):
    return new_graph(
        "solve x",
        GoalObligation("x", "Eq(x, 5)", "Reals", objective),
        generation_contract=env.generation_contract,
    )


def test_environment_digest_is_stable_and_sensitive_to_one_field():
    first = environment()
    second = environment()
    changed = environment(temperature=0.25)

    assert first.contract.to_record() == second.contract.to_record()
    assert first.sha256 == second.sha256
    assert first.sha256 != changed.sha256
    assert len(first.sha256) == 64


def test_reason_code_digest_ignores_frozenset_construction_order():
    class ReceiptContract:
        contract_id = "receipt-v1"
        validity_rejection_codes = frozenset(("z", "a"))
        discharge_rejection_codes = frozenset(("q", "b"))

    class ReorderedReceiptContract:
        contract_id = "receipt-v1"
        validity_rejection_codes = frozenset(("a", "z"))
        discharge_rejection_codes = frozenset(("b", "q"))

    assert verifier_reason_codes_sha256(
        ReceiptContract()
    ) == verifier_reason_codes_sha256(ReorderedReceiptContract())


def test_bound_environment_rejects_declared_implementation_drift():
    original = environment()
    drifted = replace(
        original.contract,
        verifier_implementation="different.Verifier",
    )

    with pytest.raises(ValueError, match="verifier_implementation"):
        BoundEnvironment(
            drifted,
            original.generation_contract,
            original.verifier,
            original.aggregate_builder,
        )


def test_environment_rejects_mixed_casework_semantic_profiles():
    legacy = CaseworkMathVerifier()
    corrected = CaseworkMathVerifier.corrected_v11()

    with pytest.raises(ValueError, match="mixed semantic profiles"):
        build_environment_contract(
            generation_contract=legacy,
            verifier=corrected,
            aggregate_builder=corrected,
            lens_contract=LENS_CONTRACT_VERSION,
            generation_schema=GENERATION_SCHEMA_VERSION,
            generation_temperature=0.0,
            generation_max_tokens=96,
            generation_seed_policy="none-v1",
            turn_limit=4,
        )

    legacy_environment = environment()
    with pytest.raises(ValueError, match="mixed semantic profiles"):
        BoundEnvironment(
            legacy_environment.contract,
            legacy,
            corrected,
            legacy,
        )


def test_observation_cannot_claim_an_arbitrary_environment_digest():
    env = environment()
    state = State(root(env), turn=0)
    observation = build_observation(
        state,
        ObservationContext(
            env.contract.turn_limit,
            "f" * 64,
        ),
    )

    with pytest.raises(ValueError, match="bound environment"):
        validate_observation_with_contract(
            observation,
            env.generation_contract,
            env.verifier,
            environment_contract_sha256=env.sha256,
        )


def test_macro_transition_is_bound_to_one_environment_digest():
    env = environment()
    transition = env.step(
        State(root(env), turn=0),
        Action(Verb.PROPOSE_RESULT, 0),
        StaticGenerator(ResultDraft(GoalObjective.SUM, "5")),
    )
    other = environment(temperature=0.5)

    assert transition.environment_contract_sha256 == env.sha256
    with pytest.raises(MacroTransitionError, match="different environment"):
        other.validate_transition(transition)


def test_algebra_rejection_retains_and_replays_its_typed_draft():
    env = environment()
    state = State(root(env, GoalObjective.IDENTITY), turn=0)
    transition = env.step(
        state,
        Action(Verb.PROPOSE_SOLUTION, 0),
        StaticGenerator(SolutionDraft("y", "5")),
    )

    assert transition.outcome.reason is RejectionReason.ALGEBRA_REJECTED
    assert transition.policy_draft == SolutionDraft("y", "5")

    forged = replace(transition, policy_draft=SolutionDraft("x", "5"))
    with pytest.raises(
        MacroTransitionError,
        match="algebra-rejected draft is accepted",
    ):
        env.validate_transition(forged)


def test_generation_failure_has_no_forged_typed_effect():
    env = environment()
    transition = env.step(
        State(root(env), turn=0),
        Action(Verb.PROPOSE_RESULT, 0),
        StaticGenerator(ResultDraft(GoalObjective.SUM, "not symbolic")),
    )

    assert transition.outcome.reason is RejectionReason.ALGEBRA_REJECTED
    with pytest.raises(ValueError, match="cannot retain a draft"):
        replace(
            transition,
            outcome=replace(
                transition.outcome,
                reason=RejectionReason.GENERATION_ERROR,
            ),
        )


def test_bound_environment_rechecks_mutated_collaborator_identity():
    env = environment()
    object.__setattr__(env.verifier, "contract_id", "drifted-verifier")

    with pytest.raises(ValueError, match="verifier_contract"):
        env.observe(State(root(env), turn=0))


def test_bound_environment_rejects_observation_horizon_drift():
    env = environment(turn_limit=4)
    observation = env.observe(State(root(env), turn=0))
    observation["turn_limit"] = 5

    with pytest.raises(ValueError, match="bound environment"):
        validate_observation_with_contract(
            observation,
            env.generation_contract,
            env.verifier,
            environment_contract_sha256=env.sha256,
            turn_limit=env.contract.turn_limit,
        )


def test_bound_environment_rejects_macro_horizon_drift():
    env = environment(turn_limit=4)
    transition = env.step(
        State(root(env), turn=0),
        Action(Verb.PROPOSE_RESULT, 0),
        StaticGenerator(ResultDraft(GoalObjective.SUM, "6")),
    )
    forged = replace(transition, turn_limit=5)

    with pytest.raises(MacroTransitionError, match="horizon"):
        env.validate_transition(forged)


def test_stale_action_cannot_form_a_canonical_macro_transition():
    env = environment()
    transition = env.step(
        State(root(env, GoalObjective.IDENTITY), turn=0),
        Action(Verb.PROPOSE_SOLUTION, 0),
        StaticGenerator(SolutionDraft("y", "5")),
    )
    stale = TransitionOutcome(
        transition.action,
        OutcomeCode.REJECTED,
        RejectionReason.STALE_ACTION,
    )
    forged = replace(
        transition,
        policy_draft=None,
        outcome=stale,
        next_state=replace(
            transition.next_state,
            previous_outcome=stale,
        ),
    )

    with pytest.raises(MacroTransitionError, match="stale actions"):
        env.validate_transition(forged)
