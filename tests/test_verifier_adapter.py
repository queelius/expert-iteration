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
