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


# A representative schema-valid JSON instance per LLM-calling verb (spec
# section 3: VERIFY/COMMIT/ABANDON never call the LLM, so only these three
# have a schema_for() to check).
_VERB_SAMPLES = {
    "SPLIT": '{"connective": "conjunctive", "children": ["a", "b"]}',
    "FORM_HYPOTHESIS": '{"statement": "x"}',
    "COMPUTE": '{"statement": "7*6 = 42", "premises": [1]}',
}


def _populated_state():
    from verifier_anchored_rl.deliberation_mdp import DeliberationState
    from verifier_anchored_rl.reductio import form_hypothesis, new_graph

    graph = new_graph("Compute 7*6")
    result = form_hypothesis(graph, 0, "The product is 42.")
    return DeliberationState(result.graph, 0)  # hypothesis id=1 (a live premise)


@pytest.mark.parametrize("verb_name", list(_VERB_SAMPLES))
def test_every_verb_schema_is_constraint_parseable(verb_name):
    pytest.importorskip("lmformatenforcer")
    from verifier_anchored_rl.constrained_json import assert_schema_parseable
    from verifier_anchored_rl.deliberation_mdp import Verb
    from verifier_anchored_rl.lenses import schema_for

    state = _populated_state()
    schema = schema_for(Verb[verb_name], state.graph.root_id, state)
    # Raises if lm-format-enforcer cannot parse this verb's schema (the bug
    # that masked the closed-loop run). All verbs must pass.
    assert_schema_parseable(schema, _VERB_SAMPLES[verb_name])
