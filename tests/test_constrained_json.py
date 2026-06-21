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
