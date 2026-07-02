import pytest


def test_adapter_satisfies_protocol():
    pytest.importorskip("sympy")
    from verifier_anchored_rl.reductio.adapters.sympy_math import SymPyVerifier
    from verifier_anchored_rl.reductio.verify import Verifier
    assert isinstance(SymPyVerifier(), Verifier)


def test_equal_expressions_verify():
    pytest.importorskip("sympy")
    from verifier_anchored_rl.reductio.adapters.sympy_math import SymPyVerifier
    v = SymPyVerifier()
    assert v.check("2*(x+1)", "simplify", "2*x+2").success


def test_unequal_expressions_do_not_verify():
    pytest.importorskip("sympy")
    from verifier_anchored_rl.reductio.adapters.sympy_math import SymPyVerifier
    v = SymPyVerifier()
    assert not v.check("x+1", "simplify", "x+2").success


def test_equation_equality():
    pytest.importorskip("sympy")
    from verifier_anchored_rl.reductio.adapters.sympy_math import SymPyVerifier
    v = SymPyVerifier()
    assert v.check("Eq(7*6, 42)", "check", "True").success or \
           v.check("7*6", "evaluate", "42").success


def test_parse_error_is_not_an_exception():
    pytest.importorskip("sympy")
    from verifier_anchored_rl.reductio.adapters.sympy_math import SymPyVerifier
    v = SymPyVerifier()
    r = v.check("@@bad@@", "check", "x")
    assert r.success is False
