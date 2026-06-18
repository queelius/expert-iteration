"""Tests for the StateActor bridge and the PrefillCache."""

from __future__ import annotations

import numpy as np
import pytest

from tests.fakes import FakeEncoder


def test_prefill_cache_encodes_once_per_distinct_prefix():
    pytest.importorskip("torch")
    from verifier_anchored_rl.planner_actor import PrefillCache

    enc = FakeEncoder(embed_dim=8)
    cache = PrefillCache(capacity=16)

    a1 = cache.get_or_encode("state A", enc)
    a2 = cache.get_or_encode("state A", enc)
    b1 = cache.get_or_encode("state B", enc)

    assert enc.prefill_calls == 2  # "state A" encoded once, "state B" once
    assert a1 is a2  # cache hit returns the same StatePrefill object
    assert len(cache) == 2
    assert not np.allclose(a1.hidden.numpy(), b1.hidden.numpy())


def test_prefill_cache_lru_evicts_oldest():
    pytest.importorskip("torch")
    from verifier_anchored_rl.planner_actor import PrefillCache

    enc = FakeEncoder(embed_dim=8)
    cache = PrefillCache(capacity=2)
    cache.get_or_encode("a", enc)
    cache.get_or_encode("b", enc)
    cache.get_or_encode("a", enc)  # touch "a" so "b" is the oldest
    cache.get_or_encode("c", enc)  # evicts "b"
    assert len(cache) == 2
    cache.get_or_encode("b", enc)  # must re-encode "b"
    assert enc.prefill_calls == 4  # a, b, c, b-again
