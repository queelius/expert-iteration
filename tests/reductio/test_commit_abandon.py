from verifier_anchored_rl.reductio.graph import new_graph, get, solved, is_abandoned
from verifier_anchored_rl.reductio.actions import (
    split, form_hypothesis, compute, verify, commit, abandon, Ok, Err,
)
from verifier_anchored_rl.reductio.verify import MockVerifier


def _solved_leaf_goal():
    # root goal "x=2"; a verified derived fact "x=2".
    g = new_graph("x=2")
    g = compute(g, "x=2", premise_ids=()).graph              # node 1
    g = verify(g, 1, "check", "x=2", MockVerifier((("x=2", True),))).graph  # node 2
    return g


def test_commit_goal_to_matching_verified_fact():
    g = _solved_leaf_goal()
    assert not solved(g, 0)
    r = commit(g, 0, 1)
    assert isinstance(r, Ok)
    assert solved(r.graph, 0)


def test_commit_rejects_unverified_fact():
    g = new_graph("x=2")
    g = compute(g, "x=2", premise_ids=()).graph              # node 1, not verified
    r = commit(g, 0, 1)
    assert isinstance(r, Err)


def test_commit_accepts_verified_fact_v1_enforces_truth_not_relevance():
    # v1 deliberately does not enforce statement relevance: a verified/holding fact
    # may discharge a goal. Relevance (does it actually answer the goal) is the
    # agent's claim, with a structural entailment check deferred.
    g = new_graph("x=2")
    g = compute(g, "y=9", premise_ids=()).graph              # unrelated derived fact, node 1
    g = verify(g, 1, "check", "y=9", MockVerifier((("y=9", True),))).graph
    r = commit(g, 0, 1)
    assert isinstance(r, Ok)
    assert solved(r.graph, 0)


def test_commit_rejects_goal_witness():
    # With the Method layer gone, a Goal is never a commit witness.
    g = new_graph("g")
    g = split(g, 0, "conjunctive", ("a",)).graph             # node 1, child Goal
    r = commit(g, 0, 1)
    assert isinstance(r, Err)


def test_commit_parent_while_children_open_or_abandoned():
    # Subgoals are optional scaffolding (new-design.md section 3.3): the root
    # can be committed from its own verified hypothesis regardless of children.
    g = new_graph("g")
    g = split(g, 0, "conjunctive", ("a", "b")).graph         # children 1, 2
    g = abandon(g, 2).graph
    g = form_hypothesis(g, 0, "the answer").graph            # node 3
    hyp = get(g, 0).candidates[0]
    g = verify(g, hyp, "check", "the answer",
               MockVerifier((("the answer", True),))).graph
    r = commit(g, 0, hyp)
    assert isinstance(r, Ok)
    assert solved(r.graph, 0)


def test_abandon_marks_and_blocks_solved():
    g = _solved_leaf_goal()
    g = commit(g, 0, 1).graph
    assert solved(g, 0)
    g2 = abandon(g, 0).graph
    assert is_abandoned(g2, 0)
    assert not solved(g2, 0)
