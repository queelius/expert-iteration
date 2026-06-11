"""Tests for the SymPy-backed verifier.

These do not test SymPy itself; they test that the verifier dispatches
correctly, returns ``VerifierResult`` with the expected fields, and
fails closed (returns ``success=False``) on malformed input rather than
raising.
"""

from __future__ import annotations

import pytest

sympy = pytest.importorskip("sympy")  # skip module if sympy is unavailable

from verifier_anchored_rl.verifier import check_answer, run_verifier


# -----------------------------------------------------------------------------
# check_arith
# -----------------------------------------------------------------------------


def test_check_arith_true():
    result = run_verifier("check_arith", "7 * 6", "42")
    assert result.success
    assert "42" in result.observed


def test_check_arith_false():
    result = run_verifier("check_arith", "7 * 6", "43")
    assert not result.success
    assert result.reason


def test_check_arith_algebraic_equivalence():
    result = run_verifier("check_arith", "(x + 1)**2", "x**2 + 2*x + 1")
    assert result.success


def test_check_arith_malformed_input_fails_closed():
    result = run_verifier("check_arith", "this is not math", "42")
    assert not result.success
    assert result.reason


# -----------------------------------------------------------------------------
# simplify
# -----------------------------------------------------------------------------


def test_simplify_true():
    result = run_verifier("simplify", "(x**2 - 1) / (x - 1)", "x + 1")
    assert result.success


def test_simplify_false():
    result = run_verifier("simplify", "x + 1", "x + 2")
    assert not result.success


# -----------------------------------------------------------------------------
# factor
# -----------------------------------------------------------------------------


def test_factor_true():
    result = run_verifier("factor", "x**2 - 1", "(x - 1)*(x + 1)")
    assert result.success


def test_factor_false():
    result = run_verifier("factor", "x**2 - 1", "(x - 2)*(x + 1)")
    assert not result.success


# -----------------------------------------------------------------------------
# solve
# -----------------------------------------------------------------------------


def test_solve_linear():
    result = run_verifier("solve", "2*x + 4 = 0", "-2")
    assert result.success


def test_solve_quadratic_with_two_roots():
    result = run_verifier("solve", "x**2 - 4 = 0", "[-2, 2]")
    assert result.success


def test_solve_wrong_root():
    result = run_verifier("solve", "x**2 - 4 = 0", "3")
    assert not result.success


# -----------------------------------------------------------------------------
# evaluate
# -----------------------------------------------------------------------------


def test_evaluate_correct_numeric():
    result = run_verifier("evaluate", "sqrt(16) + 1", "5")
    assert result.success


def test_evaluate_wrong_numeric():
    result = run_verifier("evaluate", "sqrt(16) + 1", "6")
    assert not result.success


def test_evaluate_without_expected_just_reports_value():
    result = run_verifier("evaluate", "sqrt(16) + 1", None)
    assert result.success
    assert result.observed  # value present even without an expected claim


# -----------------------------------------------------------------------------
# Missing expected for binary verifier
# -----------------------------------------------------------------------------


def test_check_arith_without_expected_fails_closed():
    result = run_verifier("check_arith", "7 * 6", None)
    assert not result.success
    assert "expected required" in (result.reason or "")


# -----------------------------------------------------------------------------
# check_answer (terminal commit)
# -----------------------------------------------------------------------------


def test_check_answer_numeric_equality():
    assert check_answer("42", "42").success
    assert check_answer("42.0", "42").success
    assert not check_answer("41", "42").success


def test_check_answer_algebraic_equality():
    assert check_answer("(x + 1)**2", "x**2 + 2*x + 1").success


def test_check_answer_string_fallback():
    # When sympify fails (non-numeric, non-algebraic), fall back to
    # normalized string equality (lowercase, strip trailing punct).
    assert check_answer("Yes", "yes.").success
    assert not check_answer("Yes", "no").success


def test_check_answer_handles_whitespace_and_punctuation():
    assert check_answer("  42 . ", "42").success
    assert check_answer("42!", "42").success


def test_check_answer_empty_committed_never_succeeds():
    # "No answer" must not vacuously equal an empty expected label
    # (a data-entry slip); the empty committed answer is always a failure.
    assert not check_answer("", "").success
    assert not check_answer(None, "").success
    assert not check_answer("   ", "").success
    # "." normalizes to empty (trailing-punct strip) and must also fail.
    assert not check_answer(".", "").success
    assert not check_answer("", "42").success


def test_run_verifier_unknown_verb_is_failed_call_not_keyerror():
    # An out-of-enum verifier verb must take the failed-verifier branch
    # (logged, -0.05 reward path), not escape as KeyError into the MDP's
    # cheaper masked-action recovery.
    result = run_verifier("check", "7 * 6", "42")
    assert not result.success
    assert "unknown verifier verb" in (result.reason or "")
