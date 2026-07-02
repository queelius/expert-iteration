from verifier_anchored_rl.reductio.graph import new_graph
from verifier_anchored_rl.reductio.actions import split, form_hypothesis, compute
from verifier_anchored_rl.reductio.lens import scope


def test_scope_includes_focus_descendants_and_ancestors():
    g = new_graph("root")
    g = split(g, 0, "conjunctive", ("a", "b")).graph     # children: 1 (a), 2 (b)
    g = split(g, 1, "conjunctive", ("a1",)).graph        # under goal 1: child 3 (a1)
    sub = scope(g, 1)                                     # focus the "a" subgoal
    ids = set(sub.node_ids)
    assert sub.focus == 1
    assert 1 in ids                       # focus
    assert 3 in ids                       # descendant subgoal
    assert 0 in ids                       # ancestor path to root
    assert 2 not in ids                   # sibling subgoal "b" is out of scope


def test_scope_includes_candidate_hypotheses_of_focus():
    g = new_graph("root")
    g = split(g, 0, "conjunctive", ("a", "b")).graph     # children 1, 2
    g = form_hypothesis(g, 1, "claim for a").graph       # node 3, candidate of 1
    ids = set(scope(g, 1).node_ids)
    assert 3 in ids


def test_scope_of_root_is_whole_forest_backbone():
    g = new_graph("root")
    g = split(g, 0, "conjunctive", ("a",)).graph
    sub = scope(g, 0)
    assert set(sub.node_ids) == {0, 1}


def test_scope_reaches_a_goals_attached_facts():
    g = new_graph("root")
    g = split(g, 0, "conjunctive", ("a", "b")).graph       # children 1, 2
    g = compute(g, 1, "fact for a", premise_ids=()).graph  # node 3, attached to goal 1
    ids = set(scope(g, 1).node_ids)
    assert 3 in ids                       # the attached fact is in scope
    assert 3 not in set(scope(g, 2).node_ids)   # not in the sibling's scope
