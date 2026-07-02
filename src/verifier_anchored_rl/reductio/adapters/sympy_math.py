"""SymPy implementation of the Verifier protocol. The only module importing sympy."""

from __future__ import annotations

from dataclasses import dataclass

import sympy

from ..verify import VerifierResult


def _to_expr(text: str):
    expr = sympy.sympify(text)
    if isinstance(expr, sympy.Equality):
        return expr.lhs - expr.rhs
    # sympy.sympify('True') may return Python bool True rather than sympy.true;
    # handle both forms so that the Eq-vs-True path works correctly.
    if expr is sympy.true or expr is True:
        return sympy.Integer(0)
    if expr is sympy.false or expr is False:
        return sympy.Integer(1)
    return expr


@dataclass(frozen=True)
class SymPyVerifier:
    def check(self, statement: str, verb: str, expected: str | None) -> VerifierResult:
        try:
            lhs = _to_expr(statement)
            if expected is None:
                return VerifierResult(success=True, observed=str(lhs))
            rhs = _to_expr(expected)
            success = sympy.simplify(lhs - rhs) == 0
            return VerifierResult(success=bool(success), observed=str(sympy.sympify(statement)))
        except (sympy.SympifyError, TypeError, SyntaxError, AttributeError):
            return VerifierResult(success=False, observed="<parse-error>")
