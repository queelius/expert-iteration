"""The abs-value case analysis of design.md section 3.1, end to end.

Solve |x-3| + |x+2| = 7 by exhaustive case split. Exercises: SPLIT with
direct children, per-case hypothesis discharge, a derived contradiction
followed by ABANDON of the dead-end case, subgoal COMMIT as bookkeeping,
and the root COMMIT from a verified root hypothesis.
"""

from verifier_anchored_rl.reductio import (
    new_graph, split, form_hypothesis, compute, verify, commit, abandon,
    get, solved, is_abandoned, MockVerifier, scope,
)


def _discharge(g, goal_id, claim):
    """form -> verify -> commit a goal via a verified hypothesis."""
    g = form_hypothesis(g, goal_id, claim).graph
    hyp = get(g, goal_id).candidates[-1]
    g = verify(g, hyp, "check", claim, MockVerifier(((claim, True),))).graph
    return commit(g, goal_id, hyp).graph


def test_absolute_value_case_analysis_solves_root():
    g = new_graph("solutions of |x-3|+|x+2|=7")
    g = split(g, 0, "exhaustive_case", (
        "solutions with x<-2",
        "solutions with -2<=x<3",
        "solutions with x>=3",
    )).graph
    case_a, case_b, case_c = get(g, 0).children

    # case x < -2: 1-2x = 7 leads to x = -3, verified.
    g = _discharge(g, case_a, "x=-3")
    # case -2 <= x < 3: 5 = 7 is a contradiction; the case is a dead end.
    g = compute(g, case_b, "5=7", premise_ids=()).graph
    fact_id = len(g.nodes) - 1
    assert fact_id in set(scope(g, case_b).node_ids)
    g = abandon(g, case_b).graph
    # case x >= 3: 2x-1 = 7 leads to x = 4, verified.
    g = _discharge(g, case_c, "x=4")

    assert solved(g, case_a) and solved(g, case_c)
    assert is_abandoned(g, case_b) and not solved(g, case_b)

    # Root closes from its own verified hypothesis; children were scaffolding.
    g = _discharge(g, 0, "x=-3 or x=4")
    assert solved(g, 0)


def test_root_commit_does_not_require_all_children_closed():
    # Same structure, but the policy skips two cases entirely and still
    # closes the root from a verified hypothesis: structure never gates.
    g = new_graph("solutions of |x-3|+|x+2|=7")
    g = split(g, 0, "exhaustive_case", ("case A", "case B", "case C")).graph
    g = _discharge(g, 0, "x=-3 or x=4")
    assert solved(g, 0)
