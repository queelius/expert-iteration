from __future__ import annotations

import numpy as np
import pytest

from verifier_anchored_rl.phase_b_features import (
    STRUCTURAL_SCALAR_FIELDS,
    feature_contract_for_schema,
    CachedTextFeatures,
    HashingTextFeatures,
    ZeroTextFeatures,
    node_feature_text,
    structural_scalar_features,
)


def test_hashing_features_are_deterministic_normalized_and_text_sensitive():
    encoder = HashingTextFeatures(64)
    first = encoder.encode("[GOAL] solve x + 1 = 3")
    second = encoder.encode("[GOAL] solve x + 1 = 3")
    other = encoder.encode("[FACT] 1 + 1 = 2 verified")
    assert np.array_equal(first, second)
    assert np.linalg.norm(first) == pytest.approx(1.0)
    assert not np.array_equal(first, other)


def test_node_encoding_includes_status_when_text_is_empty():
    encoder = HashingTextFeatures(64)
    base = {
        "kind": "verifiable",
        "depth": -1,
        "parent": None,
        "rendered": False,
        "verified": False,
        "verify_failed": False,
        "solved": False,
        "abandoned": False,
        "text": "",
    }
    assert np.linalg.norm(encoder.encode_node(base)) == pytest.approx(1.0)


def test_structural_scalars_include_remaining_budget():
    row = {
        "turn": 4,
        "turn_limit": 16,
        "nodes": [
            {
                "verified": True,
                "solved": False,
                "abandoned": False,
            }
        ],
    }
    values = structural_scalar_features(row)
    assert values.shape == (6,)
    assert values[0] == pytest.approx(0.25)
    assert values[1] == pytest.approx(0.75)


def test_cached_text_features_round_trip(tmp_path):
    text = "State: test"
    node = {
        "kind": "goal",
        "depth": 0,
        "parent": None,
        "rendered": True,
        "verified": False,
        "verify_failed": False,
        "solved": False,
        "abandoned": False,
        "text": "test",
    }
    texts = [text, node_feature_text(node)]
    keys = [CachedTextFeatures.key(value) for value in texts]
    embeddings = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    path = tmp_path / "embeddings.npz"
    np.savez_compressed(
        path,
        model=np.asarray("fake"),
        model_digest=np.asarray("abc"),
        keys=np.asarray(keys),
        embeddings=embeddings,
    )
    encoder = CachedTextFeatures(str(path))
    assert encoder.model == "fake"
    assert encoder.dim == 2
    assert np.array_equal(encoder.encode(text), embeddings[0])
    assert np.array_equal(encoder.encode_node(node), embeddings[1])


def test_zero_text_features_are_shape_matched_and_content_free():
    encoder = ZeroTextFeatures(17)
    first = encoder.encode("one state")
    second = encoder.encode("entirely different text")

    assert first.shape == (17,)
    assert first.dtype == np.float32
    assert np.array_equal(first, np.zeros(17, dtype=np.float32))
    assert np.array_equal(first, second)
def test_feature_contract_pins_serializers_and_scalar_order():
    contract = feature_contract_for_schema("controller-observation-v1.1")

    assert contract["global_serializer"] == "controller-feature-json-v1.1"
    assert contract["node_serializer"] == "node-feature-text-v1.1"
    assert contract["structural_scalars"] == list(STRUCTURAL_SCALAR_FIELDS)

