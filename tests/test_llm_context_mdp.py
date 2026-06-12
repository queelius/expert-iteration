"""Tests for the Phase 3 LLM context MDP stub.

The stub does not call any LLM. These tests check the structured-state
invariants: identifier uniqueness, depth computation (companion paper
Definition 2), serializer round-trip, action-availability predicate
(method.tex Table 1), depth-limit hard mask (companion paper section
4.3), and the stub MDP's ``NotImplementedError`` contract.
"""

from __future__ import annotations

import pytest

from verifier_anchored_rl.llm_context_mdp import (
    Fact,
    Hypothesis,
    LLMContextMDP,
    MathAction,
    MathState,
    Subgoal,
    Verb,
    VerifierCall,
    available_actions,
    parse_state,
    serialize_state,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def pythagorean_state() -> MathState:
    """State partway through the Pythagorean decomposition example.

    Matches the motivating example in paper-recursive/sections/
    introduction.tex: ``a^2 + b^2 = c^2`` decomposes into a right-angle
    geometric sub-claim and an arithmetic sub-claim.
    """
    return MathState(
        problem="Prove a^2 + b^2 = c^2 for a right triangle with legs a, b and hypotenuse c.",
        facts=(
            Fact(id="fact_1", statement="Triangle ABC has a right angle at C.", source="given"),
        ),
        open_subgoals=(
            Subgoal(id="sg_1", statement="Establish a^2 + b^2 = c^2."),
            Subgoal(id="sg_2", statement="Right angle gives perpendicular legs.", parent_subgoal="sg_1"),
        ),
        hypotheses=(
            Hypothesis(id="hyp_1", statement="Construct altitude from C to AB.", parent_subgoal="sg_2"),
            Hypothesis(id="hyp_2", statement="Altitude creates two similar triangles.", parent_hypothesis="hyp_1"),
            Hypothesis(id="hyp_3", statement="Similarity gives proportional sides.", parent_hypothesis="hyp_2"),
        ),
        verifier_log=(
            VerifierCall(
                id="vcall_1",
                verb="check_arith",
                input="a^2 / c + b^2 / c",
                output="(a^2 + b^2) / c",
                success=True,
                target_id="hyp_3",
            ),
        ),
        turn=7,
    )


# -----------------------------------------------------------------------------
# Identifier and depth invariants
# -----------------------------------------------------------------------------


def test_all_ids_collects_every_identifier(pythagorean_state):
    ids = pythagorean_state.all_ids()
    assert "fact_1" in ids
    assert "sg_1" in ids and "sg_2" in ids
    assert "hyp_1" in ids and "hyp_2" in ids and "hyp_3" in ids
    assert "vcall_1" in ids


def test_hypothesis_depth_is_parent_chain_length(pythagorean_state):
    # hyp_1: attached to a subgoal (no parent_hypothesis) -> depth 1.
    # hyp_2: parent_hypothesis=hyp_1 -> depth 2.
    # hyp_3: parent_hypothesis=hyp_2 -> depth 3.
    assert pythagorean_state.hypothesis_depth("hyp_1") == 1
    assert pythagorean_state.hypothesis_depth("hyp_2") == 2
    assert pythagorean_state.hypothesis_depth("hyp_3") == 3


def test_hypothesis_depth_handles_missing_id(pythagorean_state):
    assert pythagorean_state.hypothesis_depth("nope") == 0


def test_lookup_helpers_return_records(pythagorean_state):
    assert pythagorean_state.fact_by_id("fact_1").statement.startswith("Triangle")
    assert pythagorean_state.subgoal_by_id("sg_2").parent_subgoal == "sg_1"
    assert pythagorean_state.hypothesis_by_id("hyp_2").parent_hypothesis == "hyp_1"


def test_terminal_iff_committed_answer_set(pythagorean_state):
    assert not pythagorean_state.is_terminal()
    committed = MathState(
        problem=pythagorean_state.problem, committed_answer="QED."
    )
    assert committed.is_terminal()


# -----------------------------------------------------------------------------
# Serializer / parser round trip
# -----------------------------------------------------------------------------


def test_serialize_contains_canonical_markers(pythagorean_state):
    text = serialize_state(pythagorean_state)
    assert "[PROBLEM]" in text and "[/PROBLEM]" in text
    assert "[FACT id=fact_1 source=given]" in text
    assert "[OPEN_SUBGOAL id=sg_1 parent=none depth=0]" in text
    assert "[OPEN_SUBGOAL id=sg_2 parent=sg_1 depth=1]" in text
    assert "[HYPOTHESIS id=hyp_1 parent=sg_2 depth=1]" in text
    assert "[HYPOTHESIS id=hyp_2 parent=hyp_1 depth=2]" in text
    assert "[HYPOTHESIS id=hyp_3 parent=hyp_2 depth=3]" in text
    assert "[VERIFIER_CALL id=vcall_1 target=hyp_3 verb=check_arith success=true]" in text
    assert text.endswith("[STATE_END]")


def test_serialize_parse_round_trip(pythagorean_state):
    text = serialize_state(pythagorean_state)
    recovered = parse_state(text)
    assert recovered == pythagorean_state


def test_serialize_parse_round_trip_minimal():
    state = MathState(problem="Compute 2 + 2.")
    text = serialize_state(state)
    recovered = parse_state(text)
    assert recovered == state


def test_serialize_parse_round_trip_with_committed_answer():
    state = MathState(
        problem="Compute 2 + 2.",
        facts=(Fact(id="fact_1", statement="2 + 2 = 4", source="verified"),),
        committed_answer="4",
        turn=3,
    )
    text = serialize_state(state)
    recovered = parse_state(text)
    assert recovered == state


# -----------------------------------------------------------------------------
# Action validation
# -----------------------------------------------------------------------------


def test_validate_target_passes_for_existing_id(pythagorean_state):
    action = MathAction(verb=Verb.DECOMPOSE, target_id="hyp_2")
    assert action.validate_target(pythagorean_state)


def test_validate_target_fails_for_missing_id(pythagorean_state):
    action = MathAction(verb=Verb.DECOMPOSE, target_id="hyp_999")
    assert not action.validate_target(pythagorean_state)


def test_validate_target_passes_when_target_is_none(pythagorean_state):
    action = MathAction(verb=Verb.COMPUTE_SUBEXPR, target_id=None)
    assert action.validate_target(pythagorean_state)


# -----------------------------------------------------------------------------
# Availability predicate (Table 1 in paper/sections/method.tex)
# -----------------------------------------------------------------------------


def test_available_actions_initial_state_excludes_hypothesis_verbs():
    # Initial state with one given fact: no hypotheses, no subgoals.
    state = MathState(
        problem="Compute 2 + 2.",
        facts=(Fact(id="fact_1", statement="2 + 2", source="given"),),
    )
    available = available_actions(state)
    assert Verb.COMPUTE_SUBEXPR in available
    assert Verb.FORM_HYPOTHESIS in available
    assert Verb.CHECK_CASE not in available  # no hypotheses yet
    assert Verb.BACKTRACK not in available
    assert Verb.COMMIT_ANSWER not in available  # no answer-bearing claim yet


def test_available_actions_with_hypothesis_unlocks_check_and_verify():
    state = MathState(
        problem="P",
        open_subgoals=(Subgoal(id="sg_1", statement="G"),),
        hypotheses=(Hypothesis(id="hyp_1", statement="H", parent_subgoal="sg_1"),),
    )
    available = available_actions(state)
    assert Verb.CHECK_CASE in available
    assert Verb.INVOKE_VERIFIER in available
    assert Verb.BACKTRACK in available
    assert Verb.COMMIT_ANSWER in available  # top-level hypothesis counts as answer-bearing


def test_depth_limit_masks_recursive_verbs_at_limit():
    # Chain of three hypotheses; depth_limit=3 should mask DECOMPOSE,
    # FORM_HYPOTHESIS, CHECK_CASE at hyp_3 (which has depth=3).
    state = MathState(
        problem="P",
        open_subgoals=(Subgoal(id="sg_1", statement="G"),),
        hypotheses=(
            Hypothesis(id="hyp_1", statement="H1", parent_subgoal="sg_1"),
            Hypothesis(id="hyp_2", statement="H2", parent_hypothesis="hyp_1"),
            Hypothesis(id="hyp_3", statement="H3", parent_hypothesis="hyp_2"),
        ),
    )
    available_capped = set(available_actions(state, depth_limit=3))
    available_uncapped = set(available_actions(state, depth_limit=None))

    # Companion paper section 4.3: the depth limit masks the three
    # recursive verbs. Non-recursive verbs (COMPUTE_SUBEXPR, INVOKE_
    # VERIFIER, BACKTRACK, COMMIT_ANSWER) remain available.
    assert Verb.DECOMPOSE not in available_capped
    assert Verb.FORM_HYPOTHESIS not in available_capped
    assert Verb.CHECK_CASE not in available_capped
    assert Verb.DECOMPOSE in available_uncapped
    assert Verb.FORM_HYPOTHESIS in available_uncapped
    assert Verb.CHECK_CASE in available_uncapped

    # The non-recursive verbs are unaffected by the mask.
    for verb in (Verb.COMPUTE_SUBEXPR, Verb.INVOKE_VERIFIER, Verb.BACKTRACK, Verb.COMMIT_ANSWER):
        assert verb in available_capped
        assert verb in available_uncapped


def test_depth_limit_does_not_mask_below_limit():
    state = MathState(
        problem="P",
        open_subgoals=(Subgoal(id="sg_1", statement="G"),),
        hypotheses=(
            Hypothesis(id="hyp_1", statement="H1", parent_subgoal="sg_1"),
            Hypothesis(id="hyp_2", statement="H2", parent_hypothesis="hyp_1"),
        ),
    )
    available = set(available_actions(state, depth_limit=3))
    assert Verb.DECOMPOSE in available
    assert Verb.FORM_HYPOTHESIS in available
    assert Verb.CHECK_CASE in available


# -----------------------------------------------------------------------------
# LLMContextMDP construction and protocol conformance
# -----------------------------------------------------------------------------


def test_initial_state_packages_given_facts():
    mdp = LLMContextMDP(problem_statement="Compute 2 + 2.")
    state = mdp.initial_state(given_facts=[Fact(id="fact_1", statement="2 + 2", source="given")])
    assert state.problem == "Compute 2 + 2."
    assert state.facts[0].id == "fact_1"
    assert state.turn == 0
    assert not state.is_terminal()
    # Initial state seeds a root subgoal so the first action has something to bind to.
    assert any(g.id == "sg_root" for g in state.open_subgoals)


def test_actions_returns_math_actions_with_none_targets():
    mdp = LLMContextMDP(problem_statement="P")
    state = mdp.initial_state()
    actions = mdp.actions(state)
    assert all(isinstance(a, MathAction) for a in actions)
    assert all(a.target_id is None and a.content is None for a in actions)


def test_sample_transition_requires_llm_client():
    mdp = LLMContextMDP(problem_statement="P")
    state = mdp.initial_state()
    action = MathAction(verb=Verb.COMMIT_ANSWER)
    with pytest.raises(RuntimeError, match="no llm_client configured"):
        mdp.sample_transition(state, action, rng=None)


def test_llm_context_mdp_satisfies_generative_mdp_protocol():
    from verifier_anchored_rl.mdp import GenerativeMDP

    mdp = LLMContextMDP(problem_statement="P")
    # PEP 544 runtime_checkable: GenerativeMDP advertises actions(),
    # sample_transition(), and is_terminal(). LLMContextMDP implements
    # all three; the protocol check confirms downstream code (MCTS)
    # can typecheck against this MDP.
    assert isinstance(mdp, GenerativeMDP)


# -----------------------------------------------------------------------------
# End-to-end transition tests with MockLLMClient
# -----------------------------------------------------------------------------


from verifier_anchored_rl.llm_client import MockLLMClient


def _mdp(script, *, expected_answer=None, depth_limit=3):
    """Helper: build an MDP with a scripted client."""
    client = MockLLMClient(script=script)
    return LLMContextMDP(
        problem_statement="Compute 7 * 6.",
        expected_answer=expected_answer,
        llm_client=client,
        depth_limit=depth_limit,
    )


def test_decompose_spawns_child_subgoals():
    mdp = _mdp(script={
        ("DECOMPOSE",): {"target_id": "sg_root", "children": ["Identify operands.", "Multiply."]}
    })
    state = mdp.initial_state()
    next_state, reward = mdp.sample_transition(state, MathAction(verb=Verb.DECOMPOSE))
    assert reward == 0.0
    assert next_state.turn == state.turn + 1
    child_statements = {g.statement for g in next_state.open_subgoals if g.parent_subgoal == "sg_root"}
    assert child_statements == {"Identify operands.", "Multiply."}


def test_compute_subexpr_adds_derived_fact():
    mdp = _mdp(script={
        ("COMPUTE_SUBEXPR",): {"expression": "7 * 6", "result": "42", "derivation": "by arithmetic"}
    })
    state = mdp.initial_state()
    next_state, reward = mdp.sample_transition(state, MathAction(verb=Verb.COMPUTE_SUBEXPR))
    assert reward == 0.0
    derived = [f for f in next_state.facts if f.source == "derived"]
    assert len(derived) == 1
    assert "42" in derived[0].statement
    assert derived[0].derivation == "by arithmetic"


def test_form_hypothesis_adds_hypothesis_under_subgoal():
    mdp = _mdp(script={
        ("FORM_HYPOTHESIS",): {"parent_subgoal_id": "sg_root", "statement": "7 * 6 = 42"}
    })
    state = mdp.initial_state()
    next_state, reward = mdp.sample_transition(state, MathAction(verb=Verb.FORM_HYPOTHESIS))
    assert reward == 0.0
    assert len(next_state.hypotheses) == 1
    assert next_state.hypotheses[0].parent_subgoal == "sg_root"
    assert next_state.hypotheses[0].statement == "7 * 6 = 42"


def test_check_case_spawns_case_child_hypotheses():
    base = MathState(
        problem="P",
        open_subgoals=(Subgoal(id="sg_root", statement="root"),),
        hypotheses=(Hypothesis(id="hyp_1", statement="x is positive", parent_subgoal="sg_root"),),
    )
    client = MockLLMClient(script={
        ("CHECK_CASE",): {"target_id": "hyp_1", "cases": ["x > 0", "x = 0"]}
    })
    mdp = LLMContextMDP(problem_statement="P", llm_client=client)
    next_state, reward = mdp.sample_transition(base, MathAction(verb=Verb.CHECK_CASE))
    assert reward == 0.0
    children = [h for h in next_state.hypotheses if h.parent_hypothesis == "hyp_1"]
    assert len(children) == 2
    assert {c.statement for c in children} == {"x > 0", "x = 0"}


def test_invoke_verifier_success_promotes_to_fact_with_positive_reward():
    base = MathState(
        problem="P",
        hypotheses=(Hypothesis(id="hyp_1", statement="7 * 6 = 42"),),
    )
    client = MockLLMClient(script={
        ("INVOKE_VERIFIER",): {
            "target_id": "hyp_1",
            "verifier_verb": "check_arith",
            "input_expression": "7 * 6",
            "expected": "42",
        }
    })
    mdp = LLMContextMDP(problem_statement="P", llm_client=client)
    next_state, reward = mdp.sample_transition(base, MathAction(verb=Verb.INVOKE_VERIFIER))
    assert reward > 0
    assert len(next_state.verifier_log) == 1
    assert next_state.verifier_log[0].success is True
    assert any(f.source == "verified" for f in next_state.facts)


def test_invoke_verifier_failure_does_not_promote_and_yields_negative_reward():
    base = MathState(
        problem="P",
        hypotheses=(Hypothesis(id="hyp_1", statement="7 * 6 = 43"),),
    )
    client = MockLLMClient(script={
        ("INVOKE_VERIFIER",): {
            "target_id": "hyp_1",
            "verifier_verb": "check_arith",
            "input_expression": "7 * 6",
            "expected": "43",
        }
    })
    mdp = LLMContextMDP(problem_statement="P", llm_client=client)
    next_state, reward = mdp.sample_transition(base, MathAction(verb=Verb.INVOKE_VERIFIER))
    assert reward < 0
    assert len(next_state.verifier_log) == 1
    assert next_state.verifier_log[0].success is False
    assert not any(f.source == "verified" for f in next_state.facts)


def test_backtrack_moves_subgoal_to_closed_and_abandoned():
    base = MathState(
        problem="P",
        open_subgoals=(Subgoal(id="sg_root", statement="root"), Subgoal(id="sg_1", statement="other")),
    )
    client = MockLLMClient(script={
        ("BACKTRACK",): {"target_id": "sg_1", "reason": "unproductive"}
    })
    mdp = LLMContextMDP(problem_statement="P", llm_client=client)
    next_state, reward = mdp.sample_transition(base, MathAction(verb=Verb.BACKTRACK))
    assert reward == 0.0
    assert "sg_1" in next_state.abandoned
    assert all(g.id != "sg_1" for g in next_state.open_subgoals)
    assert any(g.id == "sg_1" for g in next_state.closed_subgoals)


def test_commit_answer_correct_returns_terminal_positive_reward():
    mdp = _mdp(
        script={("COMMIT_ANSWER",): {"answer": "42"}},
        expected_answer="42",
    )
    state = mdp.initial_state()
    next_state, reward = mdp.sample_transition(state, MathAction(verb=Verb.COMMIT_ANSWER))
    assert reward == 1.0
    assert next_state.committed_answer == "42"
    assert next_state.is_terminal()
    assert mdp.is_terminal(next_state)


def test_commit_answer_wrong_returns_terminal_negative_reward():
    mdp = _mdp(
        script={("COMMIT_ANSWER",): {"answer": "41"}},
        expected_answer="42",
    )
    state = mdp.initial_state()
    next_state, reward = mdp.sample_transition(state, MathAction(verb=Verb.COMMIT_ANSWER))
    assert reward == -1.0
    assert next_state.committed_answer == "41"
    assert next_state.is_terminal()


def test_commit_answer_without_label_returns_zero_reward():
    mdp = _mdp(script={("COMMIT_ANSWER",): {"answer": "42"}}, expected_answer=None)
    state = mdp.initial_state()
    next_state, reward = mdp.sample_transition(state, MathAction(verb=Verb.COMMIT_ANSWER))
    assert reward == 0.0
    assert next_state.committed_answer == "42"
    assert next_state.is_terminal()


def test_commit_answer_with_none_answer_is_masked_not_terminal():
    """LLM occasionally returns COMMIT_ANSWER with a null/missing answer
    field. The commit must not happen: the action is masked (small shaped
    penalty, episode continues), the same recovery every other verb gets
    for a malformed continuation. Terminating with -1 would punish
    transient JSON noise as a verifier rejection, and the old "<none>"
    sentinel collided with serialize_state's placeholder for None,
    breaking the parse round-trip.
    """
    mdp = _mdp(
        script={("COMMIT_ANSWER",): {"answer": None}},
        expected_answer="42",
    )
    state = mdp.initial_state()
    next_state, reward = mdp.sample_transition(state, MathAction(verb=Verb.COMMIT_ANSWER))
    assert reward == -0.01
    assert not next_state.is_terminal()
    assert next_state.committed_answer is None
    assert next_state.turn == state.turn + 1
    # The round-trip invariant holds: no sentinel aliasing.
    from verifier_anchored_rl import parse_state, serialize_state

    assert parse_state(serialize_state(next_state)).committed_answer is None


def test_commit_answer_with_empty_string_answer_is_masked():
    """Empty/whitespace answers also count as no-commit (masked action)."""
    mdp = _mdp(
        script={("COMMIT_ANSWER",): {"answer": "   "}},
        expected_answer="42",
    )
    state = mdp.initial_state()
    next_state, reward = mdp.sample_transition(state, MathAction(verb=Verb.COMMIT_ANSWER))
    assert reward == -0.01
    assert not next_state.is_terminal()
    assert next_state.committed_answer is None


def test_commit_answer_numeric_answer_is_coerced_and_scored():
    """A JSON number answer (the way small JSON-mode models often emit
    numeric answers) is coerced to its canonical string and scored
    normally: {"answer": 42} with expected "42" earns the success reward.
    """
    mdp = _mdp(
        script={("COMMIT_ANSWER",): {"answer": 42}},
        expected_answer="42",
    )
    state = mdp.initial_state()
    next_state, reward = mdp.sample_transition(state, MathAction(verb=Verb.COMMIT_ANSWER))
    assert reward == 1.0
    assert next_state.committed_answer == "42"
    assert next_state.is_terminal()


def test_commit_answer_integral_float_answer_is_coerced():
    """42.0 coerces to "42", not "42.0"."""
    mdp = _mdp(
        script={("COMMIT_ANSWER",): {"answer": 42.0}},
        expected_answer="42",
    )
    state = mdp.initial_state()
    next_state, reward = mdp.sample_transition(state, MathAction(verb=Verb.COMMIT_ANSWER))
    assert reward == 1.0
    assert next_state.committed_answer == "42"


def test_decompose_with_string_children_is_masked_not_per_character():
    """A string in the children slot must not iterate per character into
    junk subgoals; it is a malformed continuation -> masked action."""
    mdp = _mdp(
        script={
            ("DECOMPOSE",): {"target_id": "sg_root", "children": "split into cases"},
        },
    )
    state = mdp.initial_state()
    avail = available_actions(state, depth_limit=mdp.depth_limit)
    assert Verb.DECOMPOSE in avail
    next_state, reward = mdp.sample_transition(state, MathAction(verb=Verb.DECOMPOSE))
    assert reward == -0.01
    assert len(next_state.open_subgoals) == len(state.open_subgoals)


def test_form_hypothesis_with_null_statement_is_masked():
    """statement=None must not produce a Hypothesis serializing as the
    literal 'None'; it is a malformed continuation -> masked action."""
    mdp = _mdp(script={("FORM_HYPOTHESIS",): {"statement": None}})
    state = mdp.initial_state()
    next_state, reward = mdp.sample_transition(
        state, MathAction(verb=Verb.FORM_HYPOTHESIS)
    )
    assert reward == -0.01
    assert len(next_state.hypotheses) == 0


def test_compute_subexpr_numeric_fields_are_coerced():
    """Numeric expression/result/derivation coerce to strings so the Fact
    serializes cleanly (a non-string derivation used to crash
    serialize_state's quoting downstream, outside the masked-action net)."""
    from verifier_anchored_rl import serialize_state

    mdp = _mdp(
        script={
            ("COMPUTE_SUBEXPR",): {
                "expression": "7*6",
                "result": 42,
                "derivation": 42,
            },
        },
    )
    state = mdp.initial_state()
    next_state, reward = mdp.sample_transition(
        state, MathAction(verb=Verb.COMPUTE_SUBEXPR)
    )
    assert reward == 0.0
    assert next_state.facts[-1].statement == "7*6 = 42"
    assert next_state.facts[-1].derivation == "42"
    serialize_state(next_state)  # must not raise


def test_masked_action_when_llm_returns_unparseable_response():
    from verifier_anchored_rl.llm_client import LLMResponseError

    class FlakyClient:
        def complete_json(self, *a, **kw):
            raise LLMResponseError("bad json")

    mdp = LLMContextMDP(problem_statement="P", llm_client=FlakyClient())
    state = mdp.initial_state()
    next_state, reward = mdp.sample_transition(state, MathAction(verb=Verb.COMMIT_ANSWER))
    assert reward < 0
    assert next_state.turn == state.turn + 1
    assert next_state.committed_answer is None


def test_masked_action_when_target_id_does_not_exist():
    base = MathState(problem="P", open_subgoals=(Subgoal(id="sg_root", statement="g"),))
    client = MockLLMClient(script={
        ("DECOMPOSE",): {"target_id": "sg_nope", "children": ["a", "b"]}
    })
    mdp = LLMContextMDP(problem_statement="P", llm_client=client)
    next_state, reward = mdp.sample_transition(base, MathAction(verb=Verb.DECOMPOSE))
    assert reward < 0
    assert next_state.turn == base.turn + 1
    assert next_state.open_subgoals == base.open_subgoals  # state unchanged except turn


def test_full_three_step_trajectory_with_commit():
    """form hypothesis -> verify it -> commit answer.

    Demonstrates the loop closes end-to-end with reward signals at the
    expected places.
    """
    script = {
        ("FORM_HYPOTHESIS",): {"parent_subgoal_id": "sg_root", "statement": "7 * 6 = 42"},
        ("INVOKE_VERIFIER",): {
            "target_id": "hyp_1",
            "verifier_verb": "check_arith",
            "input_expression": "7 * 6",
            "expected": "42",
        },
        ("COMMIT_ANSWER",): {"answer": "42"},
    }
    mdp = _mdp(script=script, expected_answer="42")
    state = mdp.initial_state()

    state, r1 = mdp.sample_transition(state, MathAction(verb=Verb.FORM_HYPOTHESIS))
    assert r1 == 0.0
    assert len(state.hypotheses) == 1

    state, r2 = mdp.sample_transition(state, MathAction(verb=Verb.INVOKE_VERIFIER))
    assert r2 > 0
    assert any(f.source == "verified" for f in state.facts)

    state, r3 = mdp.sample_transition(state, MathAction(verb=Verb.COMMIT_ANSWER))
    assert r3 == 1.0
    assert state.is_terminal()


def test_max_turns_cap_makes_state_terminal_via_mdp():
    mdp = LLMContextMDP(problem_statement="P", max_turns=2, llm_client=MockLLMClient())
    state = MathState(problem="P", turn=2)
    # Cap is checked by the MDP, not by MathState.is_terminal alone.
    assert mdp.is_terminal(state)
    assert not state.is_terminal()  # MathState alone is not terminal


# -----------------------------------------------------------------------------
# EchoLLMClient end-to-end smoke
# -----------------------------------------------------------------------------


def test_echo_client_smoke_walks_through_each_verb():
    """The echo client should at least not crash on every verb.

    EchoLLMClient produces minimum-information responses for each
    schema. This is a smoke test: it confirms the prompt assembly +
    parse + dispatch pipeline runs end-to-end for every verb without
    needing a real model.
    """
    from verifier_anchored_rl.llm_client import EchoLLMClient

    mdp = LLMContextMDP(
        problem_statement="P",
        llm_client=EchoLLMClient(),
        expected_answer="something",
    )
    state = mdp.initial_state()
    # Echo will return empty target_ids that won't validate; the MDP
    # should mask each such action without raising.
    for verb in Verb:
        if verb in available_actions(state, depth_limit=mdp.depth_limit):
            try:
                state, _ = mdp.sample_transition(state, MathAction(verb=verb))
            except Exception as e:
                pytest.fail(f"verb {verb} raised: {e}")


def test_reward_constants_are_public():
    """Downstream packages (rewrite-rl) import these instead of copying
    magic numbers; the private names remain as aliases."""
    from verifier_anchored_rl import (
        INTERMEDIATE_VERIFIER_REWARD,
        MASKED_ACTION_REWARD,
        TERMINAL_FAILURE_REWARD,
        TERMINAL_SUCCESS_REWARD,
    )
    from verifier_anchored_rl import llm_context_mdp as m

    assert TERMINAL_SUCCESS_REWARD == 1.0
    assert TERMINAL_FAILURE_REWARD == -1.0
    assert INTERMEDIATE_VERIFIER_REWARD == 0.05
    assert MASKED_ACTION_REWARD == -0.01
    assert m._TERMINAL_SUCCESS_REWARD is TERMINAL_SUCCESS_REWARD
    assert m._INTERMEDIATE_VERIFIER_REWARD is INTERMEDIATE_VERIFIER_REWARD
