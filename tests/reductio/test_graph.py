from verifier_anchored_rl.reductio.graph import Goal, Fact, new_graph, get


def test_new_graph_seeds_root_goal_and_givens():
    g = new_graph("prove P", givens=("a=1", "b=2"))
    root = get(g, g.root_id)
    assert isinstance(root, Goal)
    assert root.statement == "prove P"
    facts = [n for n in g.nodes if isinstance(n, Fact)]
    assert [f.statement for f in facts] == ["a=1", "b=2"]
    assert all(f.source == "given" for f in facts)
    assert root.facts == (1, 2)      # givens attach to the root goal


def test_graph_is_a_value_equal_and_hashable():
    g1 = new_graph("p", givens=("a=1",))
    g2 = new_graph("p", givens=("a=1",))
    assert g1 == g2
    assert hash(g1) == hash(g2)
    assert len({g1, g2}) == 1


def test_node_id_equals_index():
    g = new_graph("p", givens=("a=1", "b=2"))
    for i, n in enumerate(g.nodes):
        assert n.id == i


def test_goal_defaults_have_no_children_and_no_connective():
    g = new_graph("p")
    root = get(g, 0)
    assert root.children == ()
    assert root.candidates == ()
    assert root.facts == ()
    assert root.connective is None
