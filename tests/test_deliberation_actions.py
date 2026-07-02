from verifier_anchored_rl.deliberation_mdp import (
    DeliberationAction, DeliberationState, Verb, available_actions, _witnesses,
)
from verifier_anchored_rl.reductio import (
    new_graph, split, form_hypothesis, compute, verify, commit, abandon,
    get, MockVerifier,
)


def _acts(state, **kw):
    return {(a.verb, a.target) for a in available_actions(state, **kw)}


def test_state_and_action_are_hashable_values():
    s1 = DeliberationState(new_graph("p"), 0)
    s2 = DeliberationState(new_graph("p"), 0)
    assert s1 == s2 and hash(s1) == hash(s2)
    a1 = DeliberationAction(Verb.SPLIT, 0)
    a2 = DeliberationAction(Verb.SPLIT, 0)
    assert a1 == a2 and len({a1, a2}) == 1


def test_fresh_root_offers_structure_and_propose_verbs():
    s = DeliberationState(new_graph("p"), 0)
    acts = _acts(s)
    assert (Verb.SPLIT, 0) in acts
    assert (Verb.FORM_HYPOTHESIS, 0) in acts
    assert (Verb.COMPUTE, 0) in acts
    assert (Verb.ABANDON, 0) in acts
    assert (Verb.COMMIT, 0) not in acts          # no witness yet
    assert (Verb.VERIFY, 0) not in acts          # goals are not VERIFY targets


def test_verify_targets_are_unverified_hypotheses_and_derived_facts():
    g = new_graph("p", givens=("a=1",))          # given fact, id 1
    g = form_hypothesis(g, 0, "claim").graph     # hypothesis, id 2
    g = compute(g, 0, "2+2=4", premise_ids=()).graph   # derived fact, id 3
    acts = _acts(DeliberationState(g, 0))
    assert (Verb.VERIFY, 2) in acts
    assert (Verb.VERIFY, 3) in acts
    assert (Verb.VERIFY, 1) not in acts          # given facts need no VERIFY
    g2 = verify(g, 2, "check", "claim", MockVerifier((("claim", True),))).graph
    acts2 = _acts(DeliberationState(g2, 0))
    assert (Verb.VERIFY, 2) not in acts2         # already verified


def test_commit_available_only_with_discharging_witness_and_open_goal():
    g = new_graph("p")
    g = form_hypothesis(g, 0, "ans").graph       # hyp id 1
    s = DeliberationState(g, 0)
    assert (Verb.COMMIT, 0) not in _acts(s)      # unverified candidate
    g = verify(g, 1, "check", "ans", MockVerifier((("ans", True),))).graph
    assert (Verb.COMMIT, 0) in _acts(DeliberationState(g, 0))
    g = commit(g, 0, 1).graph                    # goal closes
    acts = _acts(DeliberationState(g, 0))
    assert (Verb.COMMIT, 0) not in acts          # closed goals accept nothing
    assert (Verb.SPLIT, 0) not in acts


def test_witnesses_ascending_most_recent_last():
    g = new_graph("p", givens=("g1",))                    # holding fact, id 1
    g = form_hypothesis(g, 0, "h").graph                  # hyp id 2
    g = verify(g, 2, "check", "h", MockVerifier((("h", True),))).graph
    root = get(g, 0)
    assert _witnesses(g, root) == (1, 2)


def test_abandoned_nodes_offer_nothing():
    g = new_graph("p")
    g = split(g, 0, "conjunctive", ("a",)).graph          # child goal id 1
    g = abandon(g, 1).graph
    acts = _acts(DeliberationState(g, 0))
    assert not any(t == 1 for _, t in acts)


def test_depth_limit_masks_everything():
    s = DeliberationState(new_graph("p"), 5)
    assert available_actions(s, depth_limit=5) == ()
    assert _acts(s, depth_limit=6)               # one turn left: non-empty
