"""Tests for the casework problem generator.

The correctness test deliberately recomputes each answer from the record's
stored parameters using closed-form root formulas, NOT by calling the
generator's own solveset path, so it is an independent check on the
generator rather than a tautology.
"""

import json

import pytest

pytest.importorskip("sympy")

import experiments.generate_casework as g

ALL = list(g.FAMILIES)


def _expected_answer(rec: dict) -> int:
    """Closed-form ground truth from the record's params, independent of the
    generator's solving code."""
    p = rec["difficulty"]["params"]
    fam, qt = rec["family"], rec["question_type"]
    if fam == "f1a":
        m, k = p["m"], p["k"]
        roots = [m - k, m + k]
        return sum(roots) if qt == "sum_of_solutions" else max(roots)
    if fam == "f1b":
        m, k, d = p["m"], p["k"], p["d"]
        roots = [m - k, m + k]
        survivors = ([r for r in roots if r > d] if p["direction"] == "gt"
                     else [r for r in roots if r < d])
        assert len(survivors) == 1
        return survivors[0]
    if fam == "f2a":
        return p["answer_x"]
    if fam == "f2b":
        return p["x_lo"] + p["x_up"]
    if fam == "f3":
        roots = sorted((p["r1"], p["r2"]))
        return roots[-1] if qt == "largest_solution" else roots[0]
    if fam == "f4":
        a, b, c = p["a"], p["b"], p["c"]
        return ((a + b + c) // 2 if qt == "largest_solution"
                else (a + b - c) // 2)
    if fam == "f6":
        # Recompute from the equation, not the recorded answer: the live
        # root solves -(x + b) = c*x + d, i.e. x = -(b + d)/(c + 1).
        num, den = -(p["b"] + p["d"]), p["c"] + 1
        assert num % den == 0
        return num // den
    if fam == "f7":
        # Recompute the bill from the recorded usage through the tariff.
        w, t, r1, r2, cost = (p["answer_x"], p["t"], p["r1"], p["r2"],
                              p["cost"])
        assert (r1 * w if w <= t else r1 * t + r2 * (w - t)) == cost
        return w
    if fam == "f8":
        # Brute-force from params, independently of the generator's code.
        good = 0
        for v in range(p["lo"], p["hi"] + 1):
            if p["form"] == "inside":
                good += abs(v - p["a"]) <= p["b"] and v % p["m"] != 0
            else:
                good += abs(v - p["a"]) >= p["b"] and v % p["m"] == 0
        return good
    raise AssertionError(f"unknown family {fam}")


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------

def test_same_seed_is_byte_identical():
    a = g.generate(50, seed=0, families=ALL)
    b = g.generate(50, seed=0, families=ALL)
    assert [json.dumps(r, sort_keys=True) for r in a] == \
           [json.dumps(r, sort_keys=True) for r in b]


def test_different_seed_differs():
    a = {r["id"] for r in g.generate(50, seed=0, families=ALL)}
    b = {r["id"] for r in g.generate(50, seed=1, families=ALL)}
    assert a != b


# --------------------------------------------------------------------------
# answer correctness (independent recomputation)
# --------------------------------------------------------------------------

def test_answers_match_closed_form_across_families():
    recs = g.generate(200, seed=7, families=ALL)
    assert {r["family"] for r in recs} == set(ALL)
    for r in recs:
        assert int(r["answer"]) == _expected_answer(r), r


def test_answers_are_integers():
    for r in g.generate(200, seed=3, families=ALL):
        int(r["answer"])                       # raises if not integer-clean
        for s in r["solution_set"]:
            int(s)


# --------------------------------------------------------------------------
# structure guarantees
# --------------------------------------------------------------------------

def test_dead_case_families_have_a_dead_branch():
    for r in g.generate(160, seed=0, families=["f1b", "f2a", "f4", "f6", "f7"]):
        assert r["n_dead"] >= 1
        assert len(r["solution_set"]) == r["n_cases"] - r["n_dead"]


def test_counting_family_enumerates_and_counts_consistently():
    for r in g.generate(40, seed=5, families=["f8"]):
        assert r["question_type"] == "count"
        assert int(r["answer"]) == len(r["solution_set"]) >= 1
        # every listed value actually satisfies the stated predicate
        p = r["difficulty"]["params"]
        for s in r["solution_set"]:
            v = int(s)
            assert p["lo"] <= v <= p["hi"]
            if p["form"] == "inside":
                assert abs(v - p["a"]) <= p["b"] and v % p["m"] != 0
            else:
                assert abs(v - p["a"]) >= p["b"] and v % p["m"] == 0


def test_scale_widens_parameter_magnitudes_and_stays_correct():
    small = g.generate(60, seed=2, families=["f1a", "f4", "f6"], scale=1)
    big = g.generate(60, seed=2, families=["f1a", "f4", "f6"], scale=10)
    mag = lambda recs: max(abs(int(r["answer"])) for r in recs)
    assert mag(big) > mag(small)              # the knob actually turns
    for r in big:                             # correctness holds at scale
        assert int(r["answer"]) == _expected_answer(r)


def test_live_case_families_have_no_dead_branch():
    for r in g.generate(160, seed=0, families=["f1a", "f2b", "f3"]):
        assert r["n_dead"] == 0
        assert len(r["solution_set"]) == r["n_cases"]


def test_statements_never_say_case():
    for r in g.generate(200, seed=0, families=ALL):
        assert "case" not in r["problem"].lower()


# --------------------------------------------------------------------------
# id stability, uniqueness, cross-split disjointness
# --------------------------------------------------------------------------

def test_ids_are_stable_across_regeneration():
    a = [r["id"] for r in g.generate(40, seed=0, families=ALL)]
    b = [r["id"] for r in g.generate(40, seed=0, families=ALL)]
    assert a == b


def test_ids_unique_within_a_batch():
    ids = [r["id"] for r in g.generate(200, seed=0, families=ALL)]
    assert len(ids) == len(set(ids))


def test_exclude_ids_guarantees_cross_split_disjointness():
    train = g.generate(120, seed=0, families=ALL)
    train_ids = frozenset(r["id"] for r in train)
    test = g.generate(60, seed=1, families=ALL, exclude_ids=train_ids)
    assert train_ids.isdisjoint({r["id"] for r in test})
    assert len(test) == 60


# --------------------------------------------------------------------------
# record schema + provenance
# --------------------------------------------------------------------------

def test_record_has_full_schema_and_provenance():
    r = g.generate(1, seed=0, families=["f1a"])[0]
    for key in ("id", "problem", "answer", "symbolic", "solution_set",
                "question_type", "family", "n_cases", "n_dead",
                "difficulty", "provenance"):
        assert key in r, key
    for key in ("generator", "generator_version", "git_sha", "sympy_version",
                "seed", "index_in_seed", "split"):
        assert key in r["provenance"], key
    assert r["difficulty"]["tier"] in (1, 2)
    assert isinstance(r["difficulty"]["params"], dict)


def test_split_label_recorded():
    r = g.generate(1, seed=0, families=["f3"], split="train")[0]
    assert r["provenance"]["split"] == "train"


# --------------------------------------------------------------------------
# the verification gate actually gates
# --------------------------------------------------------------------------

def test_verify_rejects_wrong_structural_counts():
    import sympy as sp
    p = g.gen_f1a(g.Random(0))
    p.n_dead = 1                               # lie: f1a has no dead branch
    with pytest.raises(ValueError):
        g.verify_and_fill(p)


def test_verify_rejects_statement_that_leaks_case():
    p = g.gen_f3(g.Random(0))
    p.problem = p.problem + " (consider each case)"
    with pytest.raises(ValueError):
        g.verify_and_fill(p)
