"""Casework problem generator: a structure-exercising synthetic dataset.

Each problem is built in reverse: sample the solution structure first (roots,
cases, which branches are dead), then render a natural-language statement
around it. Ground truth is therefore correct by construction, and every
record is independently re-solved with ``sympy.solveset`` before it is
written, so a constructor bug raises instead of shipping a wrong label.

The point of the dataset is that case analysis is REQUIRED for correctness:
the scalar answer (a sum, a largest solution, a surviving root) cannot be
known without exploring every case, and some cases are dead ends that must
be pruned. That is what exercises the deliberation MDP's structural verbs
(``split``, ``abandon``), which collapse on linearly-solvable GSM8K.

See docs/superpowers/specs/2026-07-02-casework-problem-generator.md.

Families (v1):
  f1a  |a*x + b| = c, both roots valid; ask sum or largest solution.
  f1b  |a*x + b| = c with a side condition x>d or x<d; one root dies.
  f2a  piecewise f(x)=v with one domain-dead branch; ask the surviving x.
  f2b  piecewise f(x)=v with both branches live; ask the sum of solutions.
  f3   x^2 + p*x + q = 0 with distinct integer roots; ask largest/smallest.

Families (v2, for the E12 difficulty calibration):
  f4   |x - a| + |x - b| = c: three regions, the middle one dead; ask
       largest/smallest (sum leaks a+b without casework, so it is banned).
  f6   |x + b| = c*x + d: the classic extraneous-solution trap; the signed
       candidate that violates its own branch domain must be discarded.
  f7   tiered-pricing word problems, solved in reverse ("the bill was C;
       how many units?"): a piecewise-linear inverse with one tier dead by
       monotonicity; E12 round-2 candidate targeting word-problem
       semantics rather than algebra.
  f8   integer counting under interval + divisibility constraints ("how
       many integers x in [lo, hi] satisfy ..."): ground truth by
       exhaustive enumeration (cross-checked against the constructor's
       closed form); E12 round-2 candidate targeting boundary/off-by-one
       discipline.

All constructors take a ``scale`` knob multiplying the sampling ranges
(magnitude difficulty) while integer-cleanliness stays construction-exact.

Determinism: a single ``random.Random(seed)`` drives all sampling in a fixed
order, so ``--seed`` fully determines the output (within one environment;
provenance carries the git SHA and sympy version that would differ across
environments).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Callable

import sympy as sp

GENERATOR_VERSION = "1.0.0"
ID_PREFIX = "cw1"

x = sp.Symbol("x", real=True)


# --------------------------------------------------------------------------
# rendering helpers (natural-language statements, no "case" wording)
# --------------------------------------------------------------------------

def _linear(coef: int, const: int) -> str:
    """Render ``coef*x + const`` as human algebra: '2x - 3', 'x', '-x + 4'."""
    if coef == 1:
        head = "x"
    elif coef == -1:
        head = "-x"
    else:
        head = f"{coef}x"
    if const == 0:
        return head
    sign = "+" if const > 0 else "-"
    return f"{head} {sign} {abs(const)}"


def _signed(coef: int, var: str) -> str:
    """Render a trailing signed term like ' + 5x' or ' - x' for equations."""
    if coef == 0:
        return ""
    sign = "+" if coef > 0 else "-"
    mag = abs(coef)
    body = var if mag == 1 else f"{mag}{var}"
    return f" {sign} {body}"


# --------------------------------------------------------------------------
# the family intermediate: everything needed to render, verify, and record
# --------------------------------------------------------------------------

@dataclass
class Problem:
    problem: str                       # natural-language statement
    answer: str                        # single scalar, as a string
    symbolic: str                      # human-readable canonical equation
    question_type: str                 # sum_of_solutions | largest_solution | ...
    family: str
    n_cases: int
    n_dead: int
    params: dict[str, int]
    tier: int
    # verification plan: solve each (equation, domain) and union the roots.
    solve_plan: list[tuple[sp.Eq, sp.Set]] = field(default_factory=list)
    # counting families instead carry an enumeration spec; verification
    # brute-forces it and cross-checks the constructor's closed form.
    count_spec: dict[str, Any] | None = None


# --------------------------------------------------------------------------
# question-type reductions: solution set -> scalar answer
# --------------------------------------------------------------------------

_REDUCERS: dict[str, Callable[[list[sp.Integer]], sp.Integer]] = {
    "sum_of_solutions": lambda s: sum(s, sp.Integer(0)),
    "largest_solution": max,
    "smallest_solution": min,
    "find_x": lambda s: s[0],
}


def _reduce(question_type: str, sols: list[sp.Integer]) -> sp.Integer:
    if question_type == "find_x" and len(sols) != 1:
        raise ValueError(f"find_x expects one surviving solution, got {sols}")
    return _REDUCERS[question_type](sorted(sols))


# --------------------------------------------------------------------------
# family constructors (reverse generation)
# --------------------------------------------------------------------------

def gen_f1a(rng: Random, scale: int = 1) -> Problem:
    """|a*x + b| = c, both roots live. Roots are m+k and m-k."""
    a = rng.randint(1, 5)
    m = rng.randint(-9 * scale, 9 * scale)
    k = rng.randint(1, 9 * scale)
    b, c = -a * m, a * k
    qtype = rng.choice(["sum_of_solutions", "largest_solution"])
    inside = _linear(a, b)
    verb = ("the sum of all solutions for x"
            if qtype == "sum_of_solutions"
            else "the largest value of x that satisfies the equation")
    problem = (f"Solve for all real values of x: |{inside}| = {c}. "
               f"Report {verb}.")
    eq = sp.Eq(sp.Abs(a * x + b), c)
    return Problem(
        problem=problem,
        answer="",  # filled by verification
        symbolic=str(eq),
        question_type=qtype,
        family="f1a", n_cases=2, n_dead=0,
        params={"a": a, "m": m, "k": k, "b": b, "c": c, "scale": scale},
        tier=1,
        solve_plan=[(eq, sp.S.Reals)],
    )


def gen_f1b(rng: Random, scale: int = 1) -> Problem:
    """|a*x + b| = c with x>d or x<d; the condition kills one root."""
    a = rng.randint(1, 5)
    m = rng.randint(-9 * scale, 9 * scale)
    k = rng.randint(1, 9 * scale)
    b, c = -a * m, a * k
    lo, hi = m - k, m + k
    d = rng.randint(lo + 1, hi - 1)          # strictly between the two roots
    direction = rng.choice(["gt", "lt"])
    if direction == "gt":
        domain = sp.Interval.open(d, sp.oo)   # x > d keeps the larger root hi
        cond = f"x > {d}"
    else:
        domain = sp.Interval.open(-sp.oo, d)  # x < d keeps the smaller root lo
        cond = f"x < {d}"
    inside = _linear(a, b)
    problem = (f"Solve for x given that {cond}: |{inside}| = {c}. "
               f"Report the value of x.")
    eq = sp.Eq(sp.Abs(a * x + b), c)
    return Problem(
        problem=problem,
        answer="",
        symbolic=str(eq),
        question_type="find_x",
        family="f1b", n_cases=2, n_dead=1,
        params={"a": a, "m": m, "k": k, "b": b, "c": c, "d": d,
                "direction": direction, "scale": scale},
        tier=2,
        solve_plan=[(eq, domain)],
    )


def _piecewise(a1: int, b1: int, a2: int, b2: int, t: int) -> tuple[str, str]:
    """Return (statement fragment, symbolic str) for the piecewise function."""
    lower = _linear(a1, b1)
    upper = _linear(a2, b2)
    frag = (f"A function is defined by f(x) = {lower} when x < {t}, "
            f"and f(x) = {upper} when x >= {t}.")
    piece = sp.Piecewise((a1 * x + b1, x < t), (a2 * x + b2, x >= t))
    return frag, str(piece)


def gen_f2a(rng: Random, scale: int = 1) -> Problem:
    """Piecewise f(x)=v; one branch's candidate lands outside its domain."""
    t = rng.randint(-5 * scale, 5 * scale)
    v = rng.randint(-20 * scale, 20 * scale)
    a1 = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
    a2 = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
    dead_lower = rng.random() < 0.5
    if dead_lower:
        # Live branch: upper (x >= t), solution xUp >= t.
        x_up = rng.randint(t, t + 6 * scale)
        b2 = v - a2 * x_up
        # Dead branch: lower (x < t), candidate xLoCand >= t violates x < t.
        x_lo_cand = rng.randint(t, t + 6 * scale)
        b1 = v - a1 * x_lo_cand
        answer_x = x_up
        params_extra = {"x_up": x_up, "x_lo_cand": x_lo_cand}
    else:
        # Live branch: lower (x < t), solution xLo < t.
        x_lo = rng.randint(t - 6 * scale, t - 1)
        b1 = v - a1 * x_lo
        # Dead branch: upper (x >= t), candidate xUpCand < t violates x >= t.
        x_up_cand = rng.randint(t - 6 * scale, t - 1)
        b2 = v - a2 * x_up_cand
        answer_x = x_lo
        params_extra = {"x_lo": x_lo, "x_up_cand": x_up_cand}
    frag, sym = _piecewise(a1, b1, a2, b2, t)
    problem = f"{frag} Find the value of x for which f(x) = {v}."
    lower_eq = sp.Eq(a1 * x + b1, v)
    upper_eq = sp.Eq(a2 * x + b2, v)
    return Problem(
        problem=problem,
        answer="",
        symbolic=f"Eq({sym}, {v})",
        question_type="find_x",
        family="f2a", n_cases=2, n_dead=1,
        params={"a1": a1, "b1": b1, "a2": a2, "b2": b2, "t": t, "v": v,
                "answer_x": answer_x, "scale": scale, **params_extra},
        tier=2,
        solve_plan=[(lower_eq, sp.Interval.open(-sp.oo, t)),
                    (upper_eq, sp.Interval(t, sp.oo))],
    )


def gen_f2b(rng: Random, scale: int = 1) -> Problem:
    """Piecewise f(x)=v; both branches yield an in-domain solution."""
    t = rng.randint(-5 * scale, 5 * scale)
    v = rng.randint(-20 * scale, 20 * scale)
    a1 = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
    a2 = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
    x_lo = rng.randint(t - 6 * scale, t - 1)  # in x < t
    x_up = rng.randint(t, t + 6 * scale)      # in x >= t
    b1 = v - a1 * x_lo
    b2 = v - a2 * x_up
    frag, sym = _piecewise(a1, b1, a2, b2, t)
    problem = (f"{frag} Report the sum of all values of x for which "
               f"f(x) = {v}.")
    lower_eq = sp.Eq(a1 * x + b1, v)
    upper_eq = sp.Eq(a2 * x + b2, v)
    return Problem(
        problem=problem,
        answer="",
        symbolic=f"Eq({sym}, {v})",
        question_type="sum_of_solutions",
        family="f2b", n_cases=2, n_dead=0,
        params={"a1": a1, "b1": b1, "a2": a2, "b2": b2, "t": t, "v": v,
                "x_lo": x_lo, "x_up": x_up, "scale": scale},
        tier=2,
        solve_plan=[(lower_eq, sp.Interval.open(-sp.oo, t)),
                    (upper_eq, sp.Interval(t, sp.oo))],
    )


def gen_f3(rng: Random, scale: int = 1) -> Problem:
    """x^2 + p*x + q = 0 with distinct integer roots r1 < r2."""
    r1 = rng.randint(-9 * scale, 9 * scale)
    r2 = rng.randint(-9 * scale, 9 * scale)
    while r2 == r1:
        r2 = rng.randint(-9 * scale, 9 * scale)
    r1, r2 = sorted((r1, r2))
    p, q = -(r1 + r2), r1 * r2
    qtype = rng.choice(["largest_solution", "smallest_solution"])
    which = "largest" if qtype == "largest_solution" else "smallest"
    body = f"x^2{_signed(p, 'x')}{(' + ' + str(q)) if q > 0 else (' - ' + str(abs(q))) if q < 0 else ''} = 0"
    problem = (f"Solve the equation {body}. "
               f"Report the {which} value of x.")
    eq = sp.Eq(x**2 + p * x + q, 0)
    return Problem(
        problem=problem,
        answer="",
        symbolic=str(eq),
        question_type=qtype,
        family="f3", n_cases=2, n_dead=0,
        params={"r1": r1, "r2": r2, "p": p, "q": q, "scale": scale},
        tier=1,
        solve_plan=[(eq, sp.S.Reals)],
    )


def gen_f4(rng: Random, scale: int = 1) -> Problem:
    """|x - a| + |x - b| = c with c > b - a: three regions, middle dead.

    Solutions are (a+b-c)/2 (left region) and (a+b+c)/2 (right region);
    both are integers because c - (b - a) is sampled even. The sum leaks
    a + b without any casework, so only largest/smallest are asked.
    """
    a = rng.randint(-9 * scale, 9 * scale)
    g = rng.randint(1, 8 * scale)
    b = a + g
    c = g + 2 * rng.randint(1, 8 * scale)     # > b - a, matching parity
    qtype = rng.choice(["largest_solution", "smallest_solution"])
    which = "largest" if qtype == "largest_solution" else "smallest"
    lhs = f"|{_linear(1, -a)}| + |{_linear(1, -b)}|"
    problem = (f"Solve for all real values of x: {lhs} = {c}. "
               f"Report the {which} value of x that satisfies the equation.")
    eq = sp.Eq(sp.Abs(x - a) + sp.Abs(x - b), c)
    return Problem(
        problem=problem,
        answer="",
        symbolic=str(eq),
        question_type=qtype,
        family="f4", n_cases=3, n_dead=1,
        params={"a": a, "b": b, "c": c, "scale": scale},
        tier=3,
        solve_plan=[
            (sp.Eq(a + b - 2 * x, c), sp.Interval.open(-sp.oo, a)),
            (sp.Eq(sp.Integer(b - a), c), sp.Interval(a, b)),
            (sp.Eq(2 * x - (a + b), c), sp.Interval.open(b, sp.oo)),
        ],
    )


def gen_f6(rng: Random, scale: int = 1) -> Problem:
    """|x + b| = c*x + d with c >= 2: the extraneous-solution trap.

    Reverse construction: pick the vertex v = -b and a live solution
    x_live < v on the shallow negative arm, then pass the steep RHS line
    (slope c > 1) through it. Then RHS(v) = (c+1)(v - x_live) > 0, so the
    line starts above the positive arm at the vertex and, being steeper,
    never crosses it: the positive-branch candidate always violates its
    own branch domain and must be discarded. Exactly one valid solution,
    one extraneous candidate, provably, with no rejection sampling.
    """
    c = rng.randint(2, 4)
    v = rng.randint(-9 * scale, 9 * scale)
    x_live = v - rng.randint(1, 10 * scale)       # strictly left of the vertex
    b = -v
    d = -(x_live + b) - c * x_live                # line through the live point
    inside = _linear(1, b)
    rhs = _linear(c, d)
    problem = (f"Solve for x: |{inside}| = {rhs}. Report the value of x.")
    eq = sp.Eq(sp.Abs(x + b), c * x + d)
    return Problem(
        problem=problem,
        answer="",
        symbolic=str(eq),
        question_type="find_x",
        family="f6", n_cases=2, n_dead=1,
        params={"b": b, "c": c, "d": d, "v": v, "answer_x": x_live,
                "scale": scale},
        tier=3,
        solve_plan=[
            (sp.Eq(x + b, c * x + d), sp.Interval(v, sp.oo)),
            (sp.Eq(-(x + b), c * x + d), sp.Interval.open(-sp.oo, v)),
        ],
    )


_F7_SKINS = (
    ("A courier charges {r1} dollars per kilogram for the first {T} "
     "kilograms of a shipment and {r2} dollars for each kilogram after "
     "that. Shipping one package cost {C} dollars in total. How many "
     "kilograms did the package weigh?"),
    ("An electricity plan bills {r1} dollars per kilowatt-hour for the "
     "first {T} kilowatt-hours each month and {r2} dollars for each "
     "kilowatt-hour beyond that. One month's bill came to {C} dollars. "
     "How many kilowatt-hours were used that month?"),
    ("A taxi service charges {r1} dollars per mile for the first {T} "
     "miles of a trip and {r2} dollars for each mile after that. A trip "
     "cost {C} dollars. How many miles long was the trip?"),
    ("A data plan costs {r1} dollars per gigabyte for the first {T} "
     "gigabytes and {r2} dollars for each additional gigabyte. Last "
     "month the plan cost {C} dollars. How many gigabytes were used?"),
)


def gen_f7(rng: Random, scale: int = 1) -> Problem:
    """Tiered-pricing inverse problem: given the bill, recover the usage.

    The tariff is piecewise linear and strictly increasing, so exactly one
    tier contains the true usage; the other tier's candidate violates its
    own tier condition (dead by monotonicity). Boundary usage (exactly T
    units) is sampled deliberately: tier boundaries are where models slip.
    """
    r1 = rng.randint(2, 9)
    r2 = rng.randint(2, 9)
    while r2 == r1:
        r2 = rng.randint(2, 9)
    t = rng.randint(5, 5 + 15 * scale)
    if rng.random() < 0.5:
        w = rng.randint(1, t)                 # tier 1, includes the boundary
        cost = r1 * w
    else:
        w = t + rng.randint(1, 12 * scale)    # tier 2
        cost = r1 * t + r2 * (w - t)
    skin = rng.randrange(len(_F7_SKINS))
    problem = _F7_SKINS[skin].format(r1=r1, r2=r2, T=t, C=cost)
    piece = sp.Piecewise((r1 * x, x <= t), (r1 * t + r2 * (x - t), x > t))
    return Problem(
        problem=problem,
        answer="",
        symbolic=f"Eq({piece}, {cost})",
        question_type="find_x",
        family="f7", n_cases=2, n_dead=1,
        params={"r1": r1, "r2": r2, "t": t, "cost": cost, "answer_x": w,
                "skin": skin, "scale": scale},
        tier=4,
        solve_plan=[
            (sp.Eq(r1 * x, cost), sp.Interval(1, t)),
            (sp.Eq(r1 * t + r2 * (x - t), cost), sp.Interval.open(t, sp.oo)),
        ],
    )


def _f8_values(spec: dict[str, Any]) -> list[int]:
    """Exhaustive enumeration: the ground truth for counting problems."""
    lo, hi = spec["lo"], spec["hi"]
    a, b, m = spec["a"], spec["b"], spec["m"]
    out = []
    for v in range(lo, hi + 1):
        inside = abs(v - a) <= b
        if spec["form"] == "inside":
            ok = inside and v % m != 0
        else:                                   # outside
            ok = abs(v - a) >= b and v % m == 0
        if ok:
            out.append(v)
    return out


def _count_multiples(m: int, lo: int, hi: int) -> int:
    """Multiples of m in [lo, hi] (0 if empty); floor division is exact
    for negative bounds too."""
    if hi < lo:
        return 0
    return hi // m - (lo - 1) // m


def gen_f8(rng: Random, scale: int = 1) -> Problem:
    """Count integers in [lo, hi] under an absolute-value band plus a
    divisibility constraint.

    The answer is computed two independent ways: a closed form here
    (interval intersection and multiple-counting) and brute-force
    enumeration in verify_and_fill; a mismatch raises. n_cases/n_dead are
    advisory for counting families (the casework is region-splitting, not
    an AND-OR branch count).
    """
    for _ in range(200):
        form = "inside" if rng.random() < 0.5 else "outside"
        a = rng.randint(-9 * scale, 9 * scale)
        b = rng.randint(3, 12 * scale)
        m = rng.randint(2, 5)
        lo = a - b - rng.randint(-3, 8 * scale)
        hi = a + b + rng.randint(-3, 8 * scale)
        if hi < lo:
            continue
        if form == "inside":
            il, ih = max(lo, a - b), min(hi, a + b)
            if ih < il:
                count = 0
            else:
                count = (ih - il + 1) - _count_multiples(m, il, ih)
        else:
            count = (_count_multiples(m, lo, min(hi, a - b))
                     + _count_multiples(m, max(lo, a + b), hi))
        if 1 <= count <= 60:
            break
    else:
        raise RuntimeError("f8: no admissible parameters in 200 draws")
    band = f"|{_linear(1, -a)}|"
    if form == "inside":
        cond = f"{band} <= {b}, and x is not divisible by {m}"
    else:
        cond = f"{band} >= {b}, and x is divisible by {m}"
    problem = (f"How many integers x satisfy {lo} <= x <= {hi}, "
               f"{cond}? Report the count.")
    op = "<=" if form == "inside" else ">="
    divop = "!=" if form == "inside" else "=="
    return Problem(
        problem=problem,
        answer="",
        symbolic=(f"Count({lo} <= x <= {hi}, Abs(x - ({a})) {op} {b}, "
                  f"x % {m} {divop} 0)"),
        question_type="count",
        family="f8", n_cases=2, n_dead=0,
        params={"form": form, "lo": lo, "hi": hi, "a": a, "b": b, "m": m,
                "claimed_count": count, "scale": scale},
        tier=4,
        count_spec={"form": form, "lo": lo, "hi": hi, "a": a, "b": b, "m": m},
    )


FAMILIES: dict[str, Callable[..., Problem]] = {
    "f1a": gen_f1a, "f1b": gen_f1b,
    "f2a": gen_f2a, "f2b": gen_f2b,
    "f3": gen_f3,
    "f4": gen_f4, "f6": gen_f6,
    "f7": gen_f7, "f8": gen_f8,
}

# casework-v1 shipped with these five; v2 candidates are calibrated by E12.
V1_FAMILIES = ("f1a", "f1b", "f2a", "f2b", "f3")


# --------------------------------------------------------------------------
# independent verification: re-solve, reduce, and cross-check invariants
# --------------------------------------------------------------------------

def _solve(plan: list[tuple[sp.Eq, sp.Set]]) -> list[sp.Integer]:
    """Union the integer roots of each (equation, domain) in the plan."""
    roots: set[sp.Integer] = set()
    for eq, domain in plan:
        sols = sp.solveset(eq, x, domain=domain)
        if sols == sp.S.EmptySet:
            continue                       # a pruned (dead) branch: no roots
        if not isinstance(sols, sp.FiniteSet):
            raise ValueError(f"non-finite solution set for {eq} on {domain}: "
                             f"{sols}")
        for s in sols:
            if not (s.is_real and s.is_integer):
                raise ValueError(f"non-integer solution {s} for {eq}")
            roots.add(sp.Integer(s))
    return sorted(roots)


def verify_and_fill(p: Problem) -> Problem:
    """Independently re-derive the answer and cross-check structural counts.

    This is the correctness gate: the answer field is populated from a fresh
    ``solveset`` (or, for counting families, brute-force enumeration), not
    from the constructor's bookkeeping, so a mis-placed coefficient is
    caught here rather than shipped as a wrong label.
    """
    if p.count_spec is not None:
        values = _f8_values(p.count_spec)
        if len(values) != p.params["claimed_count"]:
            raise ValueError(
                f"{p.family}: closed-form count {p.params['claimed_count']} "
                f"!= enumerated {len(values)} for {p.symbolic}")
        if "case" in p.problem.lower():
            raise ValueError(f"{p.family}: statement leaks the word 'case'")
        p.answer = str(len(values))
        p._solution_set = [str(v) for v in values]  # type: ignore[attr-defined]
        return p
    sols = _solve(p.solve_plan)
    if not sols:
        raise ValueError(f"{p.family}: no valid solutions for {p.symbolic}")
    if p.n_cases - len(sols) != p.n_dead:
        raise ValueError(
            f"{p.family}: n_cases({p.n_cases}) - valid({len(sols)}) "
            f"!= n_dead({p.n_dead}) for {p.symbolic}")
    answer = _reduce(p.question_type, sols)
    if not answer.is_integer:
        raise ValueError(f"{p.family}: non-integer answer {answer}")
    if "case" in p.problem.lower():
        raise ValueError(f"{p.family}: statement leaks the word 'case'")
    p.answer = str(answer)
    p._solution_set = [str(s) for s in sols]   # type: ignore[attr-defined]
    return p


# --------------------------------------------------------------------------
# provenance + record assembly
# --------------------------------------------------------------------------

def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(Path(__file__).resolve().parent))
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _content_id(symbolic: str, question_type: str) -> str:
    h = hashlib.sha256(f"{symbolic}|{question_type}".encode()).hexdigest()
    return f"{ID_PREFIX}-{h[:12]}"


def _record(p: Problem, *, seed: int, index_in_seed: int, split: str | None,
            git_sha: str | None, sympy_version: str) -> dict[str, Any]:
    return {
        "id": _content_id(p.symbolic, p.question_type),
        "problem": p.problem,
        "answer": p.answer,
        "symbolic": p.symbolic,
        "solution_set": p._solution_set,       # type: ignore[attr-defined]
        "question_type": p.question_type,
        "family": p.family,
        "n_cases": p.n_cases,
        "n_dead": p.n_dead,
        "difficulty": {"tier": p.tier, "params": p.params},
        "provenance": {
            "generator": "generate_casework.py",
            "generator_version": GENERATOR_VERSION,
            "git_sha": git_sha,
            "sympy_version": sympy_version,
            "seed": seed,
            "index_in_seed": index_in_seed,
            "split": split,
        },
    }


# --------------------------------------------------------------------------
# generation driver
# --------------------------------------------------------------------------

def generate(n: int, seed: int, families: list[str], *,
             split: str | None = None,
             exclude_ids: frozenset[str] = frozenset(),
             scale: int = 1) -> list[dict[str, Any]]:
    """Generate ``n`` records, round-robin across ``families``.

    Records are deduplicated by content id within this call and against
    ``exclude_ids`` (the cross-split collision guard). Sampling continues
    until ``n`` distinct problems are produced or the attempt budget is hit.
    """
    rng = Random(seed)
    git_sha = _git_sha()
    sympy_version = sp.__version__
    seen: set[str] = set(exclude_ids)
    out: list[dict[str, Any]] = []
    idx = 0
    attempts = 0
    budget = max(2000, n * 200)
    while len(out) < n:
        attempts += 1
        if attempts > budget:
            raise RuntimeError(
                f"exhausted {budget} attempts with only {len(out)}/{n} "
                f"distinct problems; widen the parameter ranges")
        fam = families[len(out) % len(families)]
        p = verify_and_fill(FAMILIES[fam](rng, scale))
        rid = _content_id(p.symbolic, p.question_type)
        if rid in seen:
            continue
        seen.add(rid)
        out.append(_record(p, seed=seed, index_in_seed=idx, split=split,
                           git_sha=git_sha, sympy_version=sympy_version))
        idx += 1
    return out


def _load_ids(path: Path) -> frozenset[str]:
    if not path.exists():
        return frozenset()
    return frozenset(json.loads(line)["id"]
                     for line in path.read_text().splitlines() if line.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--families", default=",".join(V1_FAMILIES),
                    help="comma-separated subset of " + ",".join(FAMILIES) +
                         " (default: the v1 five, so the shipped dataset's "
                         "regeneration command stays byte-faithful)")
    ap.add_argument("--scale", type=int, default=1,
                    help="multiplies sampling ranges (magnitude difficulty)")
    ap.add_argument("--split", default=None,
                    help="label recorded in provenance (e.g. train, test)")
    ap.add_argument("--exclude", type=Path, default=None,
                    help="JSONL whose ids are excluded (cross-split guard)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in families if f not in FAMILIES]
    if unknown:
        raise SystemExit(f"unknown families: {unknown}")
    exclude = _load_ids(args.exclude) if args.exclude else frozenset()

    records = generate(args.n, args.seed, families,
                       split=args.split, exclude_ids=exclude,
                       scale=args.scale)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    by_fam: dict[str, int] = {}
    for r in records:
        by_fam[r["family"]] = by_fam.get(r["family"], 0) + 1
    print(f"wrote {len(records)} problems to {args.out}  by family: {by_fam}")


if __name__ == "__main__":
    main()
