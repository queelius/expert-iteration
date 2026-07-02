import numpy as np
import pytest

pytest.importorskip("sympy")

from verifier_anchored_rl.deliberation_mdp import (
    DeliberationAction, DeliberationMDP, DeliberationState, Verb,
    MASKED_ACTION_REWARD, TERMINAL_FAILURE_REWARD, TERMINAL_SUCCESS_REWARD,
    _verification_is_rewardable,
)
from verifier_anchored_rl.reductio import (
    compute, form_hypothesis, get, is_verified, solved, verify, MockVerifier,
)


def _mdp(**kw):
    defaults = dict(problem="what is 6*7?", expected_answer="42", depth_limit=8)
    defaults.update(kw)
    return DeliberationMDP(**defaults)


def _rng():
    return np.random.default_rng(0)


def test_generative_contract_surface():
    mdp = _mdp()
    s = mdp.initial_state()
    assert isinstance(s, DeliberationState) and s.turn == 0
    assert mdp.depth_limit == 8
    assert not mdp.is_terminal(s)
    assert mdp.actions(s)


def test_verify_decides_target_statement_with_no_llm_and_no_expected():
    mdp = _mdp()
    s = mdp.initial_state()
    g = compute(s.graph, 0, "6*7 = 42", premise_ids=()).graph   # fact id 1
    s = DeliberationState(g, 1)
    s2, r = mdp.sample_transition(s, DeliberationAction(Verb.VERIFY, 1), _rng())
    assert is_verified(s2.graph, 1)
    assert r == 0.0                        # v1 intermediate coefficient is zero
    assert s2.turn == 2


def test_verify_false_statement_records_failure():
    mdp = _mdp()
    s = mdp.initial_state()
    g = compute(s.graph, 0, "6*7 = 43", premise_ids=()).graph
    s = DeliberationState(g, 1)
    s2, r = mdp.sample_transition(s, DeliberationAction(Verb.VERIFY, 1), _rng())
    assert not is_verified(s2.graph, 1)
    assert r == 0.0


def test_root_commit_correct_answer_is_terminal_plus_one():
    mdp = _mdp(expected_answer="6*7 = 42")
    s = mdp.initial_state()
    g = compute(s.graph, 0, "6*7 = 42", premise_ids=()).graph          # fact 1
    g = verify(g, 1, "check", None, mdp.verifier).graph                 # verified
    s = DeliberationState(g, 2)
    s2, r = mdp.sample_transition(s, DeliberationAction(Verb.COMMIT, 0), _rng())
    assert r == TERMINAL_SUCCESS_REWARD
    assert mdp.is_terminal(s2)
    assert solved(s2.graph, 0)


def test_root_commit_wrong_answer_is_terminal_minus_one():
    mdp = _mdp(expected_answer="43")
    s = mdp.initial_state()
    g = compute(s.graph, 0, "6*7 = 42", premise_ids=()).graph
    g = verify(g, 1, "check", None, mdp.verifier).graph
    s = DeliberationState(g, 2)
    s2, r = mdp.sample_transition(s, DeliberationAction(Verb.COMMIT, 0), _rng())
    assert r == TERMINAL_FAILURE_REWARD
    assert mdp.is_terminal(s2)


def test_commit_uses_most_recent_witness_statement():
    # Two witnesses discharge the root; the most recent (highest id) wins.
    mdp = _mdp(expected_answer="42")
    s = mdp.initial_state()
    g = compute(s.graph, 0, "6*7 = 41", premise_ids=()).graph  # fact 1: wrong,
    g = form_hypothesis(g, 0, "6*7 = 42").graph                # hyp 2: right
    g = verify(g, 2, "check", None, mdp.verifier).graph
    # fact 1 is derived but unverified: it does not hold, so it is not a
    # witness; hyp 2 is. Commit compares "6*7 = 42" against the label.
    s = DeliberationState(g, 3)
    s2, r = mdp.sample_transition(s, DeliberationAction(Verb.COMMIT, 0), _rng())
    assert r == TERMINAL_FAILURE_REWARD    # "6*7 = 42" != "42" as an answer
    # (The witness statement is the whole equation; the label is the value.
    #  This pins the comparison semantics: check_answer sees the statement.)


def test_root_abandon_terminal_reward_zero():
    mdp = _mdp()
    s = mdp.initial_state()
    s2, r = mdp.sample_transition(s, DeliberationAction(Verb.ABANDON, 0), _rng())
    assert r == 0.0
    assert mdp.is_terminal(s2)


def test_subgoal_commit_and_abandon_are_reward_free_bookkeeping():
    from verifier_anchored_rl.reductio import split
    mdp = _mdp()
    s = mdp.initial_state()
    g = split(s.graph, 0, "conjunctive", ("a", "b")).graph      # goals 1, 2
    g = form_hypothesis(g, 1, "6*7 = 42").graph                 # hyp 3
    g = verify(g, 3, "check", None, mdp.verifier).graph
    s = DeliberationState(g, 3)
    s2, r = mdp.sample_transition(s, DeliberationAction(Verb.COMMIT, 1), _rng())
    assert r == 0.0 and not mdp.is_terminal(s2) and solved(s2.graph, 1)
    s3, r = mdp.sample_transition(s2, DeliberationAction(Verb.ABANDON, 2), _rng())
    assert r == 0.0 and not mdp.is_terminal(s3)


def test_stale_action_maps_to_masked_fallback():
    mdp = _mdp()
    s = mdp.initial_state()
    stale = DeliberationAction(Verb.COMMIT, 0)      # no witness: not available
    s2, r = mdp.sample_transition(s, stale, _rng())
    assert r == MASKED_ACTION_REWARD
    assert s2.turn == s.turn + 1
    assert s2.graph == s.graph                       # nothing changed


def test_depth_exhaustion_is_terminal_with_no_verdict():
    mdp = _mdp(depth_limit=1)
    s = mdp.initial_state()
    s2, r = mdp.sample_transition(s, DeliberationAction(Verb.FORM_HYPOTHESIS, 0), _rng())
    # Task 6 wires the LLM path; before that, LLM verbs mask. Either way the
    # step must not fabricate a terminal verdict:
    assert mdp.is_terminal(s2)                       # turn 1 >= depth_limit 1
    assert r in (0.0, MASKED_ACTION_REWARD)
    assert r != TERMINAL_FAILURE_REWARD              # changed from legacy


def test_gate_predicates():
    mdp = _mdp()
    s = mdp.initial_state()
    g = compute(s.graph, 0, "6*7 = 42", premise_ids=()).graph   # fact 1
    assert _verification_is_rewardable(g, 1, "6*7 = 42")
    assert not _verification_is_rewardable(g, 1, "42 = 42")     # tautology
    gv = verify(g, 1, "check", None, mdp.verifier).graph
    assert not _verification_is_rewardable(gv, 1, "6*7 = 42")   # dedup
