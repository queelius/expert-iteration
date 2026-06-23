"""TextEmbedder adapts an EmbeddingClient to the StateActor's Encoder protocol."""

from __future__ import annotations

import pytest


class _FakeEmbeddingClient:
    """Returns a fixed-dim vector; records calls."""
    def __init__(self, dim):
        self.dim = dim
        self.calls = 0
    def embed(self, text):
        self.calls += 1
        # deterministic, content-dependent, values in [0,1)
        return [((hash(text) >> i) & 255) / 255.0 for i in range(self.dim)]


def test_text_embedder_is_an_encoder_and_returns_hidden_vector():
    pytest.importorskip("torch")
    import torch
    from verifier_anchored_rl.embedder import TextEmbedder

    client = _FakeEmbeddingClient(dim=32)
    emb = TextEmbedder(client, embed_dim=32)
    assert emb.embed_dim == 32
    prefill = emb.encode("State:\nsome serialized state")
    assert isinstance(prefill.hidden, torch.Tensor)
    assert tuple(prefill.hidden.shape) == (32,)
    assert prefill.hidden.dtype == torch.float32
    assert prefill.cache is None
    assert client.calls == 1


def test_text_embedder_drops_into_state_actor():
    pytest.importorskip("torch")
    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.planner_actor import PrefillCache, StateActor
    from verifier_anchored_rl.llm_client import EchoLLMClient
    from verifier_anchored_rl.llm_context_mdp import LLMContextMDP
    from verifier_anchored_rl.embedder import TextEmbedder

    emb = TextEmbedder(_FakeEmbeddingClient(dim=16), embed_dim=16)
    actor = HiddenStateActor(embed_dim=16)
    sa = StateActor(emb, actor, PrefillCache(), depth_limit=3)
    mdp = LLMContextMDP(problem_statement="2+2", expected_answer="4",
                        llm_client=EchoLLMClient(), depth_limit=3, max_turns=4)
    state = mdp.initial_state()
    actions = list(mdp.actions(state))
    prior = sa.prior(state, actions)
    assert abs(sum(prior) - 1.0) < 1e-5


def test_text_embedder_raises_on_dim_mismatch():
    pytest.importorskip("torch")
    from verifier_anchored_rl.embedder import TextEmbedder
    emb = TextEmbedder(_FakeEmbeddingClient(dim=8), embed_dim=16)  # client gives 8, configured 16
    with pytest.raises(ValueError):
        emb.encode("x")
