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
    from verifier_anchored_rl.deliberation_mdp import DeliberationMDP
    from verifier_anchored_rl.embedder import TextEmbedder

    emb = TextEmbedder(_FakeEmbeddingClient(dim=16), embed_dim=16)
    actor = HiddenStateActor(embed_dim=16)
    sa = StateActor(emb, actor, PrefillCache(), depth_limit=3)
    mdp = DeliberationMDP(problem="2+2", expected_answer="4", depth_limit=3)
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


# ---------------------------------------------------------------------------
# Bug 2: prefill_tokens must use true token count when client has a tokenizer
# ---------------------------------------------------------------------------

class _TokenizerFakeClient:
    """EmbeddingClient that exposes a .tokenizer callable returning a known
    number of token ids regardless of the text, so the test can be exact."""

    def __init__(self, dim: int, n_tokens: int):
        self.dim = dim
        self._n_tokens = n_tokens

    def tokenizer(self, text, **kwargs):
        return {"input_ids": list(range(self._n_tokens))}

    def embed(self, text):
        return [0.0] * self.dim


class _NoTokenizerClient:
    """EmbeddingClient without a tokenizer attribute."""

    def __init__(self, dim: int):
        self.dim = dim

    def embed(self, text):
        return [0.0] * self.dim


def test_text_embedder_prefill_tokens_uses_tokenizer_when_available():
    pytest.importorskip("torch")
    from verifier_anchored_rl.embedder import TextEmbedder

    n_tok = 7
    client = _TokenizerFakeClient(dim=4, n_tokens=n_tok)
    emb = TextEmbedder(client, embed_dim=4)
    emb.encode("hello world")
    assert emb.prefill_tokens == n_tok


def test_text_embedder_prefill_tokens_falls_back_to_word_count():
    pytest.importorskip("torch")
    from verifier_anchored_rl.embedder import TextEmbedder

    text = "three word phrase"
    client = _NoTokenizerClient(dim=4)
    emb = TextEmbedder(client, embed_dim=4)
    emb.encode(text)
    assert emb.prefill_tokens == len(text.split())  # 3
