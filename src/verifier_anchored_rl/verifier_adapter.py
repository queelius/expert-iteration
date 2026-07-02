"""Symbolic verifier for the verifier-anchored MDP.

Five verifier verbs are exposed for ``INVOKE_VERIFIER`` actions, plus
one terminal-answer check used by ``COMMIT_ANSWER``. Each function takes
string inputs (LLM-generated content) and returns a bool; all exceptions
are caught and converted to ``False`` so the MDP transition never blows
up on malformed verifier input.

The verifier is the only source of reward in the framework (see
paper/sections/method.tex Section 4.6, the verifier-anchored-reward-only
principle). Its coverage is therefore load-bearing: a claim that is
true-in-fact but rejected by the verifier produces a wrong-direction
training signal, which is the realistic threat the companion paper
section 4 calls out. SymPy covers most of GSM8K's arithmetic and a
useful fraction of MATH levels 1 through 3; harder cases will need
domain-specific extensions, but those are out of scope for the smoke
test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Literal

# Lazy import so unit tests that do not exercise the verifier do not
# pay the sympy import cost.
def _sympy():
    import sympy  # type: ignore[import-not-found]

    return sympy


VerifierVerb = Literal[
    "check_arith", "solve", "simplify", "factor", "evaluate"
]


@dataclass
class RichResult:
    """Outcome of a verifier call.

    ``success`` is the only field the reward function reads.
    ``observed`` is the symbolic form the verifier computed; it lands
    in the ``VerifierCall.output`` slot in the state for traceability
    and downstream encoding.
    ``reason`` is a short diagnostic string used when ``success`` is
    False; it is for logs and debugging, not for the agent.
    """

    success: bool
    observed: str
    reason: str | None = None


def run_verifier(verb: VerifierVerb, input_expr: str, expected: str | None) -> RichResult:
    """Dispatch one verifier call.

    ``input_expr`` is the LLM-generated expression to verify. For binary
    verbs (``check_arith``, ``simplify``, ``factor``, ``solve``) the
    ``expected`` argument is required: the claim is that the symbolic
    form of ``input_expr`` agrees with ``expected`` after the given
    operation. For ``evaluate`` the ``expected`` is the value the
    agent claims the expression evaluates to (numeric or symbolic);
    leave it as ``None`` to ask the verifier to produce the value
    without committing the agent to a specific number.
    """
    handler: Callable[[str, str | None], RichResult] | None = _VERIFIERS.get(verb)
    if handler is None:
        # An out-of-enum verifier verb is a failed verifier call, not a
        # KeyError that escapes to the MDP's masked-action recovery: the
        # two paths carry different rewards (-0.05 logged failure vs -0.01
        # unlogged retry), and a typo'd verb must not be cheaper than an
        # honest failed check.
        return RichResult(
            success=False,
            observed="",
            reason=f"unknown verifier verb {verb!r}",
        )
    try:
        return handler(input_expr, expected)
    except Exception as e:
        return RichResult(
            success=False,
            observed="",
            reason=f"verifier raised {type(e).__name__}: {e!s}",
        )


# -----------------------------------------------------------------------------
# Individual verifiers
# -----------------------------------------------------------------------------


def _check_arith(input_expr: str, expected: str | None) -> RichResult:
    """Both sides parse as SymPy expressions and ``simplify(lhs - rhs) == 0``."""
    if expected is None:
        return RichResult(success=False, observed="", reason="expected required")
    sp = _sympy()
    lhs = sp.sympify(input_expr)
    rhs = sp.sympify(expected)
    diff = sp.simplify(lhs - rhs)
    success = diff == 0
    return RichResult(
        success=bool(success),
        observed=str(sp.simplify(lhs)),
        reason=None if success else f"simplify(lhs - rhs) = {diff!s}",
    )


def _check_simplify(input_expr: str, expected: str | None) -> RichResult:
    """``simplify(input_expr)`` equals ``expected`` (also simplified)."""
    if expected is None:
        return RichResult(success=False, observed="", reason="expected required")
    sp = _sympy()
    simplified_in = sp.simplify(sp.sympify(input_expr))
    simplified_exp = sp.simplify(sp.sympify(expected))
    success = sp.simplify(simplified_in - simplified_exp) == 0
    return RichResult(
        success=bool(success),
        observed=str(simplified_in),
        reason=None if success else f"simplified({input_expr!r}) = {simplified_in!s}",
    )


def _check_factor(input_expr: str, expected: str | None) -> RichResult:
    """``factor(input_expr)`` equals ``expected`` (after simplification)."""
    if expected is None:
        return RichResult(success=False, observed="", reason="expected required")
    sp = _sympy()
    factored = sp.factor(sp.sympify(input_expr))
    expected_sym = sp.sympify(expected)
    success = sp.simplify(factored - expected_sym) == 0
    return RichResult(
        success=bool(success),
        observed=str(factored),
        reason=None if success else f"factor({input_expr!r}) = {factored!s}",
    )


def _check_solve(input_expr: str, expected: str | None) -> RichResult:
    """``solve(input_expr)`` (an equation in one variable) yields ``expected``.

    ``input_expr`` should be of the form ``"lhs = rhs"`` or a SymPy
    Eq() expression. ``expected`` is the claimed solution (a single
    value or a Python list literal of values).
    """
    if expected is None:
        return RichResult(success=False, observed="", reason="expected required")
    sp = _sympy()
    eq = _parse_equation(input_expr)
    free = eq.free_symbols
    if not free:
        return RichResult(
            success=False, observed="", reason="no free symbol in equation"
        )
    var = sorted(free, key=str)[0]
    solutions = sp.solve(eq, var)
    observed_str = str(solutions)
    claimed = _parse_solution_list(expected)
    if not isinstance(solutions, list):
        solutions = [solutions]
    sol_set = {sp.simplify(s) for s in solutions}
    claim_set = {sp.simplify(s) for s in claimed}
    success = sol_set == claim_set
    return RichResult(
        success=bool(success),
        observed=observed_str,
        reason=None if success else f"solve gave {sol_set}, claimed {claim_set}",
    )


def _check_evaluate(input_expr: str, expected: str | None) -> RichResult:
    """``input_expr`` evaluates to ``expected`` (no free variables allowed)."""
    sp = _sympy()
    value = sp.N(sp.sympify(input_expr))
    if expected is None:
        return RichResult(success=True, observed=str(value), reason=None)
    claimed = sp.N(sp.sympify(expected))
    diff = abs(complex(value - claimed))
    success = diff < 1e-9
    return RichResult(
        success=bool(success),
        observed=str(value),
        reason=None if success else f"|value - expected| = {diff}",
    )


_VERIFIERS: dict[VerifierVerb, Callable[[str, str | None], RichResult]] = {
    "check_arith": _check_arith,
    "simplify": _check_simplify,
    "factor": _check_factor,
    "solve": _check_solve,
    "evaluate": _check_evaluate,
}


# -----------------------------------------------------------------------------
# Terminal answer check (COMMIT_ANSWER)
# -----------------------------------------------------------------------------


def check_answer(committed: str, expected: str) -> RichResult:
    """Check the committed final answer against the ground-truth label.

    Strategy: parse both with SymPy and check ``simplify(committed -
    expected) == 0``. If sympify fails (for instance because the answer
    is a non-numeric string like ``"yes"``), fall back to normalized
    string equality (strip whitespace, lowercase, drop trailing
    punctuation). This dual path covers most of GSM8K (numeric answers)
    and a useful fraction of MATH levels 1 through 3 (algebraic
    expressions plus the occasional categorical answer).
    """
    expected_normalized = _normalize_answer(expected)
    committed_normalized = _normalize_answer(committed)

    # An empty committed answer is never a success, even against an empty
    # expected label (a data-entry slip): "no answer" must not satisfy the
    # one component that anchors all reward. Without this, a committed "."
    # (which strips to empty) would vacuously equal an empty expected
    # string below.
    if not committed_normalized:
        return RichResult(
            success=False,
            observed="",
            reason="empty committed answer after normalization",
        )

    sp = _sympy()
    try:
        c = sp.sympify(committed_normalized)
        e = sp.sympify(expected_normalized)
        diff = sp.simplify(c - e)
        if diff == 0:
            return RichResult(success=True, observed=str(sp.simplify(c)))
    except Exception:
        pass

    if committed_normalized == expected_normalized:
        return RichResult(success=True, observed=committed_normalized)

    return RichResult(
        success=False,
        observed=committed_normalized,
        reason=f"committed {committed_normalized!r} != expected {expected_normalized!r}",
    )


_TRAILING_PUNCT = re.compile(r"[.?!,;:]+$")


def _normalize_answer(text: str | None) -> str:
    """Normalize an answer string for equality comparison.

    Defensively accepts ``None`` (returns empty string) so that the
    verifier never crashes on a missing answer. Callers that need to
    distinguish "no answer" from "wrong answer" should check the
    input before passing it here.
    """
    if text is None:
        return ""
    text = text.strip()
    text = _TRAILING_PUNCT.sub("", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _parse_equation(text: str):
    """Parse 'lhs = rhs' or a SymPy Eq expression into a SymPy Eq."""
    sp = _sympy()
    if "=" in text and "Eq" not in text:
        left, right = text.split("=", 1)
        return sp.Eq(sp.sympify(left.strip()), sp.sympify(right.strip()))
    return sp.sympify(text)


def _parse_solution_list(text: str):
    """Parse '[1, 2, 3]' or '1' or '1, 2, 3' into a list of SymPy values."""
    sp = _sympy()
    s = text.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    items = [item.strip() for item in s.split(",") if item.strip()]
    return [sp.sympify(item) for item in items]


# -----------------------------------------------------------------------------
# Statement decision (expected=None) and the reductio protocol adapter
# -----------------------------------------------------------------------------

from .reductio.verify import VerifierResult as CoreVerifierResult


def decide(statement: str) -> RichResult:
    """Decide the truth of a self-contained claim, requiring no expected value.

    An equation ("lhs = rhs" or Eq(...)) is decided by simplify(lhs - rhs)
    == 0. A ground relational claim ("2 < 3") is decided by simplification
    to a boolean. Anything else (a bare expression, free symbols left over,
    unparsable text) is NOT a decidable claim and returns success=False:
    this function never returns a vacuous parse-only success.
    """
    sp = _sympy()
    try:
        obj = _parse_equation(statement)
    except Exception as e:
        return RichResult(success=False, observed="",
                          reason=f"parse error: {type(e).__name__}: {e!s}")
    try:
        if isinstance(obj, sp.Equality):
            diff = sp.simplify(obj.lhs - obj.rhs)
            success = diff == 0
            return RichResult(success=bool(success), observed=str(diff),
                              reason=None if success else f"lhs - rhs = {diff!s}")
        simplified = sp.simplify(obj)
        if simplified is sp.true or simplified is True:
            return RichResult(success=True, observed="True")
        if simplified is sp.false or simplified is False:
            return RichResult(success=False, observed="False",
                              reason="claim simplifies to False")
        return RichResult(success=False, observed=str(simplified),
                          reason="not a decidable claim")
    except Exception as e:
        return RichResult(success=False, observed="",
                          reason=f"decide raised {type(e).__name__}: {e!s}")


_VALID_VERBS = frozenset(_VERIFIERS)


@dataclass(frozen=True)
class MathVerifier:
    """The reductio Verifier for the math domain.

    check(statement, verb, expected):
    * expected is None  -> decide the statement itself (never vacuous).
    * expected is given -> dispatch the legacy SymPy operation named by
      verb; an unknown verb falls back to check_arith (statement equals
      expected after simplification).
    The policy never supplies expected in the Deliberation MDP; the
    None path is the only one VERIFY actions use.
    """

    def check(self, statement: str, verb: str, expected: str | None) -> CoreVerifierResult:
        if expected is None:
            rich = decide(statement)
        else:
            op = verb if verb in _VALID_VERBS else "check_arith"
            rich = run_verifier(op, statement, expected)  # type: ignore[arg-type]
        return CoreVerifierResult(success=rich.success, observed=rich.observed)


__all__ = ["MathVerifier", "RichResult", "VerifierVerb", "check_answer",
           "decide", "run_verifier"]
