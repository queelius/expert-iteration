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
    for r in g.generate(160, seed=0, families=["f1b", "f2a"]):
        assert r["n_dead"] >= 1
        assert len(r["solution_set"]) == r["n_cases"] - r["n_dead"]


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
