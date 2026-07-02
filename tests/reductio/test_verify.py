from verifier_anchored_rl.reductio.graph import new_graph, get, VerifierCall, is_verified
from verifier_anchored_rl.reductio.actions import split, compute, verify, Ok, Err
from verifier_anchored_rl.reductio.verify import MockVerifier


def test_verify_appends_call_and_marks_verified_on_success():
    g = new_graph("g", givens=("a=2", "b=3"))
    g = compute(g, 0, "a*b=6", premise_ids=(1, 2)).graph   # node 3, derived fact
    v = MockVerifier(table=(("a*b=6", True),))
    r = verify(g, 3, "check", "a*b=6", v)
    assert isinstance(r, Ok)
    calls = [n for n in r.graph.nodes if isinstance(n, VerifierCall)]
    assert len(calls) == 1 and calls[0].success
    assert is_verified(r.graph, 3)


def test_verify_failure_records_call_but_not_verified():
    g = new_graph("g", givens=("a=2",))
    g = compute(g, 0, "a=3", premise_ids=(1,)).graph       # node 2
    v = MockVerifier()                                  # nothing succeeds
    r = verify(g, 2, "check", "a=3", v)
    assert isinstance(r, Ok)
    calls = [n for n in r.graph.nodes if isinstance(n, VerifierCall) and n.target == 2]
    assert len(calls) == 1 and calls[0].success is False
    assert not is_verified(r.graph, 2)


def test_verify_is_idempotent():
    g = new_graph("g", givens=("a=2", "b=3"))
    g = compute(g, 0, "a*b=6", premise_ids=(1, 2)).graph
    v = MockVerifier(table=(("a*b=6", True),))
    g1 = verify(g, 3, "check", "a*b=6", v).graph
    g2 = verify(g1, 3, "check", "a*b=6", v).graph
    assert g1 == g2


def test_verify_rejects_node_without_statement():
    g = new_graph("g")
    g = compute(g, 0, "x=1", premise_ids=()).graph         # node 1, a Fact: fine
    g = split(g, 0, "conjunctive", ("g1",)).graph       # node 2, a child Goal
    # VerifierCall nodes carry no statement; create one via verify, then target it.
    v = MockVerifier(table=(("x=1", True),))
    g = verify(g, 1, "check", "x=1", v).graph           # node 3, a VerifierCall
    r = verify(g, 3, "check", None, v)                  # verifying a VerifierCall
    assert isinstance(r, Err)
