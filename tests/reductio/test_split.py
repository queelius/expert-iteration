from verifier_anchored_rl.reductio.graph import new_graph, get
from verifier_anchored_rl.reductio.actions import split, Ok, Err


def test_conjunctive_split_attaches_children_directly():
    g = new_graph("solve g")
    r = split(g, 0, "conjunctive", ("g1", "g2"))
    assert isinstance(r, Ok)
    g2 = r.graph
    root = get(g2, 0)
    assert root.connective == "conjunctive"
    assert [get(g2, c).statement for c in root.children] == ["g1", "g2"]


def test_exhaustive_case_split_does_not_reify_exhaustiveness_goal():
    g = new_graph("solve g")
    r = split(g, 0, "exhaustive_case", ("case A", "case B"))
    root = get(r.graph, 0)
    assert root.connective == "exhaustive_case"
    statements = [get(r.graph, c).statement for c in root.children]
    assert statements == ["case A", "case B"]


def test_second_split_same_connective_appends_children():
    g = new_graph("g")
    g = split(g, 0, "conjunctive", ("a",)).graph
    g = split(g, 0, "conjunctive", ("b",)).graph
    root = get(g, 0)
    assert [get(g, c).statement for c in root.children] == ["a", "b"]


def test_second_split_conflicting_connective_is_err():
    g = new_graph("g")
    g = split(g, 0, "conjunctive", ("a",)).graph
    r = split(g, 0, "exhaustive_case", ("b",))
    assert isinstance(r, Err)
    assert "conjunctive" in r.message


def test_split_inherits_assumptions():
    import dataclasses
    g = new_graph("g")
    root = dataclasses.replace(get(g, 0), assumptions=("h1", "h2"))
    g = dataclasses.replace(g, nodes=(root,) + g.nodes[1:])
    g = split(g, 0, "conjunctive", ("g1",)).graph
    child = get(g, get(g, 0).children[0])
    assert child.assumptions == ("h1", "h2")


def test_split_rejects_unknown_goal():
    g = new_graph("g")
    r = split(g, 99, "conjunctive", ("g1",))
    assert isinstance(r, Err)
    assert "99" in r.message


def test_split_rejects_non_goal_target():
    g = new_graph("g", givens=("a=1",))
    r = split(g, 1, "conjunctive", ("g1",))  # node 1 is a Fact
    assert isinstance(r, Err)


def test_split_rejects_bad_connective_or_empty_children():
    g = new_graph("g")
    assert isinstance(split(g, 0, "nonsense", ("g1",)), Err)
    assert isinstance(split(g, 0, "conjunctive", ()), Err)
