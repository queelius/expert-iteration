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


def _toy_mdp():
    from verifier_anchored_rl.llm_client import EchoLLMClient
    from verifier_anchored_rl.llm_context_mdp import LLMContextMDP

    return LLMContextMDP(problem_statement="2+2", expected_answer="4",
                         llm_client=EchoLLMClient(), depth_limit=3, max_turns=4)


def test_state_actor_prior_sums_to_one_over_actions():
    pytest.importorskip("torch")
    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.planner_actor import StateActor

    enc = FakeEncoder(embed_dim=16)
    actor = HiddenStateActor(embed_dim=16)
    sa = StateActor(enc, actor, depth_limit=3)

    mdp = _toy_mdp()
    state = mdp.initial_state()
    actions = list(mdp.actions(state))
    prior = sa.prior(state, actions)

    assert len(prior) == len(actions)
    assert all(p >= 0.0 for p in prior)
    assert abs(sum(prior) - 1.0) < 1e-5


def test_state_actor_leaf_value_is_max_target_q_over_available():
    pytest.importorskip("torch")
    import torch
    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.heads import VERB_NAME_TO_INDEX
    from verifier_anchored_rl.llm_context_mdp import available_actions
    from verifier_anchored_rl.planner_actor import StateActor

    enc = FakeEncoder(embed_dim=16)
    actor = HiddenStateActor(embed_dim=16)
    sa = StateActor(enc, actor, depth_limit=3)

    mdp = _toy_mdp()
    state = mdp.initial_state()
    avail = list(available_actions(state, depth_limit=3))

    h = sa.embed(state).unsqueeze(0)
    with torch.no_grad():
        qt = actor.target_q(h)[0]
    expected = max(float(qt[VERB_NAME_TO_INDEX[v.value]]) for v in avail)
    assert abs(sa.leaf_value(state) - expected) < 1e-6


def test_state_actor_embed_uses_cache_once_per_state():
    pytest.importorskip("torch")
    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.planner_actor import PrefillCache, StateActor

    enc = FakeEncoder(embed_dim=16)
    actor = HiddenStateActor(embed_dim=16)
    cache = PrefillCache()
    sa = StateActor(enc, actor, cache, depth_limit=3)

    mdp = _toy_mdp()
    state = mdp.initial_state()
    actions = list(mdp.actions(state))

    sa.embed(state)
    sa.prior(state, actions)
    sa.leaf_value(state)
    assert enc.prefill_calls == 1  # all three reads share one prefill


class _CastingFakeEncoder(FakeEncoder):
    """FakeEncoder that returns the hidden state in a chosen dtype/device,
    mimicking the real SharedForwardEncoder (bf16 on CUDA) instead of the
    default fp32-on-CPU fake. Used to reproduce the GPU-box device/dtype
    mismatch on a machine without changing any other behavior."""

    def __init__(self, embed_dim: int, *, dtype, device: str = "cpu"):
        super().__init__(embed_dim=embed_dim)
        self._dtype = dtype
        self._device = device

    def encode(self, prefix_text: str):
        prefill = super().encode(prefix_text)
        prefill.hidden = prefill.hidden.to(device=self._device, dtype=self._dtype)
        return prefill


def test_state_actor_reconciles_bf16_hidden_with_fp32_heads():
    """The real encoder emits a bf16 hidden state; the fp32 heads must still
    consume it. Reproduces the dtype half of the GPU-box crash on CPU."""
    pytest.importorskip("torch")
    import torch
    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.planner_actor import StateActor

    enc = _CastingFakeEncoder(embed_dim=16, dtype=torch.bfloat16)
    actor = HiddenStateActor(embed_dim=16)  # fp32 on CPU, like the harness
    sa = StateActor(enc, actor, depth_limit=3)

    mdp = _toy_mdp()
    state = mdp.initial_state()
    actions = list(mdp.actions(state))

    prior = sa.prior(state, actions)
    assert abs(sum(prior) - 1.0) < 1e-5
    assert np.isfinite(sa.leaf_value(state))


@pytest.mark.skipif(
    "not __import__('torch').cuda.is_available()",
    reason="needs a CUDA device",
)
def test_state_actor_reconciles_cuda_hidden_with_cpu_heads():
    """The real encoder emits a CUDA bf16 hidden state while the heads stay on
    CPU; the bridge must reconcile both device and dtype. This is the exact
    configuration that crashed the first GPU run."""
    pytest.importorskip("torch")
    import torch
    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.planner_actor import StateActor

    enc = _CastingFakeEncoder(embed_dim=16, dtype=torch.bfloat16, device="cuda")
    actor = HiddenStateActor(embed_dim=16)  # fp32 on CPU, like the harness
    sa = StateActor(enc, actor, depth_limit=3)

    mdp = _toy_mdp()
    state = mdp.initial_state()
    actions = list(mdp.actions(state))

    prior = sa.prior(state, actions)
    assert abs(sum(prior) - 1.0) < 1e-5
    assert np.isfinite(sa.leaf_value(state))


def test_state_actor_prior_is_aligned_to_action_order():
    pytest.importorskip("torch")
    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.planner_actor import StateActor

    enc = FakeEncoder(embed_dim=16)
    actor = HiddenStateActor(embed_dim=16)
    sa = StateActor(enc, actor, depth_limit=3)

    mdp = _toy_mdp()
    state = mdp.initial_state()
    actions = list(mdp.actions(state))

    prior_fwd = sa.prior(state, actions)
    prior_rev = sa.prior(state, list(reversed(actions)))
    # The masked softmax is over the same verb set either way, so reversing
    # the action list must reverse the returned probabilities exactly.
    assert prior_rev == list(reversed(prior_fwd))
