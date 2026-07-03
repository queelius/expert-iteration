import pytest

pytest.importorskip("sympy")

from verifier_anchored_rl.verifier_adapter import (
    MathVerifier, decide, run_verifier, check_answer,
)
from verifier_anchored_rl.reductio.verify import Verifier, VerifierResult


def test_adapter_satisfies_reductio_protocol():
    assert isinstance(MathVerifier(), Verifier)


def test_check_with_expected_none_decides_truth():
    v = MathVerifier()
    assert v.check("2 + 2 = 4", "check", None).success is True
    assert v.check("5 = 7", "check", None).success is False


def test_check_never_returns_vacuous_success():
    v = MathVerifier()
    # A bare expression is not a decidable claim: no vacuous parse-only True.
    assert v.check("x + 1", "check", None).success is False
    assert v.check("@@garbage@@", "check", None).success is False


def test_check_returns_reductio_result_type():
    r = MathVerifier().check("1 + 1 = 2", "check", None)
    assert isinstance(r, VerifierResult)
    assert not hasattr(r, "reason")     # the core result carries no reason field


def test_check_with_expected_dispatches_legacy_ops():
    v = MathVerifier()
    assert v.check("7*6", "evaluate", "42").success is True
    assert v.check("x**2 - 9", "factor", "(x-3)*(x+3)").success is True
    assert v.check("2*x + 5 = 13", "solve", "4").success is True
    assert v.check("2*(x+1)", "simplify", "2*x + 2").success is True
    assert v.check("7*6", "check_arith", "43").success is False


def test_check_unknown_verb_with_expected_falls_back_to_check_arith():
    assert MathVerifier().check("7*6", "check", "42").success is True


def test_decide_handles_equations_relations_and_garbage():
    assert decide("3*4 = 12").success is True
    assert decide("3*4 = 13").success is False
    assert decide("2 < 3").success is True
    assert decide("3 < 2").success is False
    assert decide("not math at all !!").success is False


def test_check_answer_ported():
    assert check_answer("42", "42.0").success is True
    assert check_answer("YES.", "yes").success is True
    assert check_answer("", "0").success is False       # empty never succeeds
    assert check_answer("41", "42").success is False


# ---------------------------------------------------------------------------
# solve-claim checking (spec 2026-07-03-solve-claim-compute)
# ---------------------------------------------------------------------------

def test_solve_claim_true_claim_passes_by_substitution():
    from verifier_anchored_rl.verifier_adapter import check_solve_claim
    assert check_solve_claim("x = 5", "4x - 16 = 4").success is True
    assert check_solve_claim("x = 5", "Solve equation 4x - 16 = 4").success is True


def test_solve_claim_false_claim_fails():
    from verifier_anchored_rl.verifier_adapter import check_solve_claim
    assert check_solve_claim("x = 3", "4x - 16 = 4").success is False


def test_solve_claim_side_condition_parenthetical_is_ignored():
    from verifier_anchored_rl.verifier_adapter import check_solve_claim
    # The domain condition is the controller's job (abandon); substitution
    # only checks the equation.
    assert check_solve_claim("x = 5", "case 4x - 16 = 4 (x >= 3/2)").success is True


def test_solve_claim_symbol_absent_from_goal_fails():
    from verifier_anchored_rl.verifier_adapter import check_solve_claim
    assert check_solve_claim("y = 5", "4x - 16 = 4").success is False


def test_solve_claim_tautological_goal_fails():
    from verifier_anchored_rl.verifier_adapter import check_solve_claim
    assert check_solve_claim("x = 5", "x = x").success is False
    assert check_solve_claim("x = 5", "4 = 4").success is False


def test_solve_claim_non_equation_goal_falls_back_to_decide():
    from verifier_anchored_rl.verifier_adapter import check_solve_claim
    # No equation in the goal text: decide("x = 5") fails closed (unknown).
    assert check_solve_claim("x = 5", "prove the problem statement").success is False
    # A ground claim still decides on its own truth regardless of context.
    assert check_solve_claim("2 + 2 = 4", "prose goal").success is True


def test_math_verifier_routes_solve_check_verb():
    v = MathVerifier()
    assert v.check("x = 5", "solve_check", "4x - 16 = 4").success is True
    assert v.check("x = 3", "solve_check", "4x - 16 = 4").success is False
    # default verb path unchanged: unknowns fail closed
    assert v.check("x = 5", "check", None).success is False


def test_parse_solve_claim_shapes():
    from verifier_anchored_rl.verifier_adapter import parse_solve_claim
    assert parse_solve_claim("x = 5") is not None
    assert parse_solve_claim("  y = -3/2 ") is not None
    assert parse_solve_claim("(4 + 16)/4 = 5") is None     # ground equation
    assert parse_solve_claim("2x = 8") is None             # not a lone symbol
    assert parse_solve_claim("x = y") is None              # non-numeric value
