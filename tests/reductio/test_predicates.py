from verifier_anchored_rl.reductio.graph import (
    Graph, Goal, Fact, Hypothesis, VerifierCall,
    is_abandoned, is_verified, holds, solved,
)


def _graph(nodes, root_id=0, abandoned=frozenset(), solved_by=()):
    return Graph(nodes=tuple(nodes), root_id=root_id,
                 abandoned=frozenset(abandoned), solved_by=tuple(solved_by))


def test_given_fact_holds_and_is_solved():
    g = _graph([Goal(0, "p"), Fact(1, "a=1", "given")])
    assert holds(g, 1)
    assert solved(g, 1)


def test_derived_fact_holds_only_when_verified():
    g = _graph([Goal(0, "p"), Fact(1, "x=2", "derived")])
    assert not holds(g, 1)
    g2 = _graph([
        Goal(0, "p"),
        Fact(1, "x=2", "derived"),
        VerifierCall(2, target=1, verb="check", expected="x=2", success=True, observed="x=2"),
    ])
    assert is_verified(g2, 1)
    assert holds(g2, 1)


def test_goal_solved_only_when_committed_to_a_discharging_witness():
    nodes = [
        Goal(0, "x=2"),
        Fact(1, "x=2", "derived"),
        VerifierCall(2, target=1, verb="check", expected="x=2", success=True, observed="x=2"),
    ]
    g = _graph(nodes)
    assert not solved(g, 0)
    g_committed = _graph(nodes, solved_by=((0, 1),))
    assert solved(g_committed, 0)


def test_hypothesis_witness_must_be_candidate_and_verified():
    nodes = [
        Goal(0, "g", candidates=(1,)),
        Hypothesis(1, "answer"),
        VerifierCall(2, target=1, verb="check", expected=None, success=True, observed="answer"),
    ]
    g = _graph(nodes, solved_by=((0, 1),))
    assert solved(g, 0)
    # same witness, but not a candidate of the goal: not solved
    nodes_nc = [
        Goal(0, "g"),
        Hypothesis(1, "answer"),
        VerifierCall(2, target=1, verb="check", expected=None, success=True, observed="answer"),
    ]
    g_nc = _graph(nodes_nc, solved_by=((0, 1),))
    assert not solved(g_nc, 0)


def test_children_do_not_gate_or_grant_parent_solved():
    # Parent with both children solved is NOT solved without its own commit
    # (structure is scaffolding, never control flow: design.md section 3.3).
    nodes = [
        Goal(0, "g", children=(1, 2), connective="conjunctive"),
        Goal(1, "g1"), Goal(2, "g2"),
        Fact(3, "g1", "given"), Fact(4, "g2", "given"),
    ]
    g = _graph(nodes, solved_by=((1, 3), (2, 4)))
    assert solved(g, 1) and solved(g, 2)
    assert not solved(g, 0)
    # Conversely: parent committed to its own verified witness IS solved
    # even though a child is open and another abandoned.
    nodes2 = [
        Goal(0, "g", children=(1, 2), connective="conjunctive", candidates=(5,)),
        Goal(1, "g1"), Goal(2, "g2"),
        Fact(3, "g1", "given"), Fact(4, "g2", "given"),
        Hypothesis(5, "the answer"),
        VerifierCall(6, target=5, verb="check", expected=None, success=True, observed="the answer"),
    ]
    g2 = _graph(nodes2, abandoned={2}, solved_by=((0, 5),))
    assert solved(g2, 0)


def test_abandoned_blocks_solved():
    nodes = [Goal(0, "x=2"), Fact(1, "x=2", "given")]
    g = _graph(nodes, abandoned={0}, solved_by=((0, 1),))
    assert is_abandoned(g, 0)
    assert not solved(g, 0)
