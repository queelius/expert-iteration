from verifier_anchored_rl.reductio.graph import new_graph
from verifier_anchored_rl.reductio.actions import split
from verifier_anchored_rl.reductio.lens import scope
from verifier_anchored_rl.reductio.serialize import render, parse


def test_render_is_deterministic_and_neutral():
    g = new_graph("root")
    g = split(g, 0, "conjunctive", ("a",)).graph
    text = render(scope(g, 0))
    assert text.splitlines()[0] == "focus=0"
    assert "[GOAL id=0] root connective=conjunctive" in text
    assert "[GOAL id=1] a" in text
    assert "METHOD" not in text
    # neutral: no prompt or menu words
    assert "prompt" not in text.lower() and "choose" not in text.lower()


def test_parse_round_trips_node_lines():
    g = new_graph("root")
    g = split(g, 0, "conjunctive", ("a",)).graph
    text = render(scope(g, 0))
    rows = parse(text)
    assert ("GOAL", 0, "root connective=conjunctive") in rows
    assert ("GOAL", 1, "a") in rows
