"""Tests for schema-constrained decoding helpers.

The high-value, model-free guard is schema parseability: lm-format-enforcer
must be able to parse each verb's JSON schema. The mixed-type enum that broke
the closed-loop run (``["sg_root", null]``) is rejected during parsing, so the
guard drives a representative JSON instance through the parser, mirroring the
per-token path the TokenEnforcer uses, with no model and no GPU.
"""

from __future__ import annotations

import pytest


def test_assert_schema_parseable_accepts_string_enum():
    pytest.importorskip("lmformatenforcer")
    from verifier_anchored_rl.constrained_json import assert_schema_parseable

    schema = {
        "type": "object",
        "properties": {
            "target_id": {"type": "string", "enum": ["sg_root", "none"]},
            "statement": {"type": "string"},
        },
        "required": ["statement"],
    }
    # A valid instance must parse without raising.
    assert_schema_parseable(schema, '{"target_id": "none", "statement": "x"}')


def test_assert_schema_parseable_rejects_mixed_null_enum():
    pytest.importorskip("lmformatenforcer")
    from verifier_anchored_rl.constrained_json import assert_schema_parseable

    # This is the exact construct that masked FORM_HYPOTHESIS in the run:
    # lm-format-enforcer cannot parse an enum mixing a string and null.
    bad = {
        "type": "object",
        "properties": {
            "parent_subgoal_id": {
                "type": ["string", "null"],
                "enum": ["sg_root", None],
            },
            "statement": {"type": "string"},
        },
        "required": ["statement"],
    }
    with pytest.raises(Exception):
        assert_schema_parseable(bad, '{"parent_subgoal_id": null, "statement": "x"}')


# A representative schema-valid JSON instance per verb, exercising the enum and
# optional fields (notably the ones that used a mixed string/null type).
_VERB_SAMPLES = {
    "DECOMPOSE": '{"target_id": "sg_root", "children": ["a", "b"]}',
    "COMPUTE_SUBEXPR": '{"expression": "7*6", "result": "42"}',
    "FORM_HYPOTHESIS": '{"parent_subgoal_id": "sg_root", "statement": "x"}',
    "CHECK_CASE": '{"target_id": "hyp_0", "cases": ["a", "b"]}',
    "INVOKE_VERIFIER": '{"target_id": "hyp_0", "verifier_verb": "check_arith", "input_expression": "7*6", "expected": "42"}',
    "BACKTRACK": '{"target_id": "sg_root", "reason": "x"}',
    "COMMIT_ANSWER": '{"answer": "42"}',
}


def _populated_state():
    from verifier_anchored_rl.llm_client import EchoLLMClient
    from verifier_anchored_rl.llm_context_mdp import (
        Hypothesis, LLMContextMDP, _with,
    )

    mdp = LLMContextMDP(problem_statement="Compute 7*6", expected_answer="42",
                        llm_client=EchoLLMClient(), depth_limit=3, max_turns=8)
    state = mdp.initial_state()  # has sg_root
    return _with(state, hypotheses=(
        Hypothesis(id="hyp_0", statement="The product is 42.",
                   parent_subgoal="sg_root"),))


@pytest.mark.parametrize("verb_name", list(_VERB_SAMPLES))
def test_every_verb_schema_is_constraint_parseable(verb_name):
    pytest.importorskip("lmformatenforcer")
    from verifier_anchored_rl.constrained_json import assert_schema_parseable
    from verifier_anchored_rl.llm_context_mdp import Verb, _schema_for

    state = _populated_state()
    schema = _schema_for(Verb[verb_name], state)
    # Raises if lm-format-enforcer cannot parse this verb's schema (the bug
    # that masked the closed-loop run). All verbs must pass.
    assert_schema_parseable(schema, _VERB_SAMPLES[verb_name])

