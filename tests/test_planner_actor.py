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


# ---------------------------------------------------------------------------
# Bug 1: PrefillCache.get_or_encode must be thread-safe
# ---------------------------------------------------------------------------

def test_prefill_cache_get_or_encode_is_thread_safe():
    """Hammer get_or_encode from several threads on a small-capacity cache
    with many distinct keys. Asserts no exception and that every returned value
    matches a direct encode result (no value corruption, no KeyError under
    concurrent eviction)."""
    import threading

    from verifier_anchored_rl.planner_actor import PrefillCache

    DIM = 4
    N_THREADS = 8
    N_KEYS = 40
    CAPACITY = 5  # small enough to provoke concurrent eviction

    class _SimpleEncoder:
        """Fake encoder returning a deterministic list-based 'hidden' (no torch
        needed) so the test runs without GPU and without importorskip."""

        embed_dim = DIM

        def encode(self, prefix_text: str):
            # Return an object whose 'hidden' is a tuple derived from the text.
            class _FakePrefill:
                pass
            p = _FakePrefill()
            p.hidden = tuple(hash(prefix_text + str(i)) % 100 for i in range(DIM))
            return p

    enc = _SimpleEncoder()
    keys = [f"key_{i}" for i in range(N_KEYS)]

    # Ground truth: encode each key independently to know the expected value.
    expected = {k: enc.encode(k).hidden for k in keys}

    cache = PrefillCache(capacity=CAPACITY)
    errors: list[Exception] = []
    results: dict[str, tuple] = {}
    lock = threading.Lock()

    def worker(thread_keys):
        try:
            for k in thread_keys:
                val = cache.get_or_encode(k, enc)
                with lock:
                    results[k] = val.hidden
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append(e)

    chunk = N_KEYS // N_THREADS
    threads = [
        threading.Thread(target=worker, args=(keys[i * chunk:(i + 1) * chunk],))
        for i in range(N_THREADS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread(s) raised exceptions: {errors}"
    for k, v in results.items():
        assert v == expected[k], f"Value mismatch for {k}: got {v}, expected {expected[k]}"


def test_embed_drops_kv_cache_and_offloads_hidden_to_cpu():
    """The heads need only h_T, not the encoder's KV cache. Retaining one KV
    cache per embedded state exhausts VRAM at high search budgets (CUDA OOM at
    B=64). embed must drop the KV cache and keep the cached embedding small."""
    pytest.importorskip("torch")
    import torch
    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.llm_client import EchoLLMClient
    from verifier_anchored_rl.llm_context_mdp import LLMContextMDP, _state_prefix
    from verifier_anchored_rl.planner_actor import PrefillCache, StateActor
    from verifier_anchored_rl.shared_forward import StatePrefill

    class KVHoldingEncoder:
        embed_dim = 8

        def __init__(self):
            self.calls = 0

        def encode(self, prefix_text):
            self.calls += 1
            return StatePrefill(
                hidden=torch.zeros(8), cache=["BIG_KV_CACHE"],
                n_prefix_tokens=3, prefix_ids=torch.zeros(1, 3))

    enc = KVHoldingEncoder()
    cache = PrefillCache()
    sa = StateActor(enc, HiddenStateActor(embed_dim=8), cache, depth_limit=3)
    mdp = LLMContextMDP(problem_statement="2+2", expected_answer="4",
                        llm_client=EchoLLMClient(), depth_limit=3, max_turns=4)
    state = mdp.initial_state()

    h = sa.embed(state)
    assert tuple(h.shape) == (8,)
    # The cached StatePrefill must no longer pin the KV cache or prefix ids.
    stored = cache.get_or_encode(_state_prefix(state), enc)  # cache hit
    assert stored.cache is None and stored.prefix_ids is None
    assert enc.calls == 1  # the hit did not re-encode
