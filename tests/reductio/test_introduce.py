from verifier_anchored_rl.reductio.graph import new_graph, get, Hypothesis, Fact
from verifier_anchored_rl.reductio.actions import form_hypothesis, compute, Ok, Err


def test_form_hypothesis_adds_candidate():
    g = new_graph("g")
    r = form_hypothesis(g, 0, "answer is 42")
    assert isinstance(r, Ok)
    root = get(r.graph, 0)
    assert len(root.candidates) == 1
    h = get(r.graph, root.candidates[0])
    assert isinstance(h, Hypothesis)
    assert h.statement == "answer is 42"


def test_form_hypothesis_rejects_non_goal():
    g = new_graph("g", givens=("a=1",))
    assert isinstance(form_hypothesis(g, 1, "x"), Err)


def test_compute_adds_derived_fact_with_premises():
    g = new_graph("g", givens=("a=2", "b=3"))
    r = compute(g, 0, "a*b=6", premise_ids=(1, 2))
    assert isinstance(r, Ok)
    facts = [n for n in r.graph.nodes if isinstance(n, Fact) and n.source == "derived"]
    assert len(facts) == 1
    assert facts[0].statement == "a*b=6"
    assert facts[0].premises == (1, 2)
    root = get(r.graph, 0)
    assert facts[0].id in root.facts          # attached to the target goal


def test_compute_rejects_bad_goal_or_premise():
    g = new_graph("g", givens=("a=1",))
    assert isinstance(compute(g, 1, "x=1", premise_ids=()), Err)    # node 1 is a Fact, not a Goal
    assert isinstance(compute(g, 99, "x=1", premise_ids=()), Err)   # missing goal
    assert isinstance(compute(g, 0, "x=1", premise_ids=(0,)), Err)  # premise 0 is a Goal
    assert isinstance(compute(g, 0, "x=1", premise_ids=(99,)), Err) # missing premise
