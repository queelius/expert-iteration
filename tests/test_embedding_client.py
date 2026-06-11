"""Tests for the embedding client abstraction.

The HTTP backends are not exercised here (they require a live Ollama or
OpenAI endpoint). The mock backend is the focus: deterministic seed-based
unit vectors that the Q/policy heads can train against without an LLM.
"""

from __future__ import annotations

import pytest

import numpy as np

from verifier_anchored_rl.embedding_client import (
    EmbeddingClient,
    MockEmbeddingClient,
    OllamaEmbeddingClient,
    OpenAICompatibleEmbeddingClient,
)


def test_mock_protocol_conformance():
    """The mock satisfies the runtime_checkable Protocol."""
    client = MockEmbeddingClient()
    assert isinstance(client, EmbeddingClient)


def test_mock_returns_correct_dimension():
    client = MockEmbeddingClient(embedding_dim=2048)
    v = client.embed("Compute 7 * 6.")
    assert len(v) == 2048


def test_mock_returns_unit_vectors():
    client = MockEmbeddingClient(embedding_dim=64)
    v = client.embed("anything")
    norm = float(np.linalg.norm(v))
    assert abs(norm - 1.0) < 1e-5


def test_mock_is_deterministic_per_input():
    a = MockEmbeddingClient(embedding_dim=32)
    b = MockEmbeddingClient(embedding_dim=32)
    v1 = a.embed("hello")
    v2 = b.embed("hello")
    assert v1 == v2  # different instances, same hash-derived seed


def test_mock_is_different_for_different_inputs():
    client = MockEmbeddingClient(embedding_dim=64)
    v1 = client.embed("hello")
    v2 = client.embed("world")
    assert v1 != v2


def test_mock_salt_changes_output():
    a = MockEmbeddingClient(embedding_dim=32, salt=0)
    b = MockEmbeddingClient(embedding_dim=32, salt=1)
    v1 = a.embed("same text")
    v2 = b.embed("same text")
    assert v1 != v2


def test_mock_records_calls():
    client = MockEmbeddingClient()
    client.embed("first")
    client.embed("second")
    assert client.calls == ["first", "second"]


def test_ollama_client_constructor_does_not_require_httpx():
    """Constructing the HTTP clients does not actually hit the network."""
    OllamaEmbeddingClient(base_url="http://localhost:11434/api/embed", model="m")
    OpenAICompatibleEmbeddingClient(base_url="http://localhost:8000/v1", model="m")


def test_ollama_client_satisfies_protocol():
    client = OllamaEmbeddingClient(base_url="http://localhost:11434/api/embed", model="m")
    assert isinstance(client, EmbeddingClient)


def test_openai_compatible_satisfies_protocol():
    client = OpenAICompatibleEmbeddingClient(base_url="http://localhost:8000/v1", model="m")
    assert isinstance(client, EmbeddingClient)


# -----------------------------------------------------------------------------
# Endpoint-shape chooser
# -----------------------------------------------------------------------------


def test_make_embedding_client_routes_by_url_shape():
    from verifier_anchored_rl.embedding_client import make_embedding_client

    ollama = make_embedding_client("http://host:11434/api/embed", "m")
    assert isinstance(ollama, OllamaEmbeddingClient)

    openai_style = make_embedding_client("http://host:8000/v1/embeddings", "m")
    assert isinstance(openai_style, OpenAICompatibleEmbeddingClient)
    # The OpenAI client appends /embeddings itself; the chooser strips it.
    assert not openai_style.base_url.endswith("/embeddings")

    openai_base = make_embedding_client("http://host:8000/v1", "m")
    assert isinstance(openai_base, OpenAICompatibleEmbeddingClient)


# -----------------------------------------------------------------------------
# EmbeddingCache (positive cache, negative cache, counters)
# -----------------------------------------------------------------------------


class _FlakyClient:
    """Raises EmbeddingError for texts in `fail_for`, embeds otherwise."""

    def __init__(self, fail_for=()):
        self.fail_for = set(fail_for)
        self.inner = MockEmbeddingClient(embedding_dim=8)
        self.embed_calls = 0

    def embed(self, text):
        from verifier_anchored_rl.embedding_client import EmbeddingError

        self.embed_calls += 1
        if text in self.fail_for:
            raise EmbeddingError(f"scripted failure for {text!r}")
        return self.inner.embed(text)


def test_embedding_cache_positive_hit_does_not_refetch():
    from verifier_anchored_rl.trained_prior import EmbeddingCache

    client = _FlakyClient()
    cache = EmbeddingCache(client)
    first = cache.get("state-a")
    second = cache.get("state-a")
    assert first == second
    assert client.embed_calls == 1
    assert cache.fetch_count == 1
    assert cache.hit_count == 1


def test_embedding_cache_negative_caches_failures():
    """A failed fetch is recorded once and never retried: every sibling
    expansion of the same state sees the same uniform fallback instead of
    a uniform/trained mix, and a downed endpoint cannot stall every
    expansion with a fresh network timeout."""
    from verifier_anchored_rl.trained_prior import EmbeddingCache

    client = _FlakyClient(fail_for={"bad-state"})
    cache = EmbeddingCache(client)
    assert cache.get("bad-state") is None
    assert cache.get("bad-state") is None
    assert cache.get("bad-state") is None
    assert client.embed_calls == 1
    assert cache.failure_count == 1
    assert cache.negative_hit_count == 2
    assert cache.fetch_count == 0


def test_trained_prior_uniform_fallback_is_consistent_across_calls():
    """With the embedding permanently failed, every prior_fn call for that
    state returns exactly uniform (no partial trained mixing)."""
    torch = pytest.importorskip("torch")
    from verifier_anchored_rl.heads import TrainableActor
    from verifier_anchored_rl.llm_context_mdp import MathAction, Verb
    from verifier_anchored_rl.trained_prior import EmbeddingCache, make_trained_prior
    from verifier_anchored_rl import LLMContextMDP
    from verifier_anchored_rl.llm_client import MockLLMClient

    class _AlwaysFail:
        def embed(self, text):
            from verifier_anchored_rl.embedding_client import EmbeddingError

            raise EmbeddingError("down")

    cache = EmbeddingCache(_AlwaysFail())
    model = TrainableActor(embed_dim=8, hidden_dim=16, repr_dim=8)
    prior_fn = make_trained_prior(model, cache, smoothing=0.02)

    mdp = LLMContextMDP(problem_statement="P", llm_client=MockLLMClient())
    state = mdp.initial_state()
    actions = [MathAction(verb=Verb.DECOMPOSE), MathAction(verb=Verb.FORM_HYPOTHESIS)]

    for _ in range(3):
        priors = prior_fn(state, actions)
        assert priors == [0.5, 0.5]
    assert cache.failure_count == 1  # negative-cached after the first attempt


def test_trained_prior_raises_on_embedder_checkpoint_dim_mismatch():
    """A checkpoint trained on one embedder must fail loudly (with the
    cause named) when served embeddings of a different dimension, not as
    an opaque matmul shape error mid-search."""
    torch = pytest.importorskip("torch")
    from verifier_anchored_rl.heads import TrainableActor
    from verifier_anchored_rl.llm_context_mdp import MathAction, Verb
    from verifier_anchored_rl.trained_prior import EmbeddingCache, make_trained_prior
    from verifier_anchored_rl import LLMContextMDP
    from verifier_anchored_rl.llm_client import MockLLMClient

    cache = EmbeddingCache(MockEmbeddingClient(embedding_dim=768))
    model = TrainableActor(embed_dim=2048, hidden_dim=16, repr_dim=8)
    prior_fn = make_trained_prior(model, cache)

    mdp = LLMContextMDP(problem_statement="P", llm_client=MockLLMClient())
    state = mdp.initial_state()
    actions = [MathAction(verb=Verb.DECOMPOSE), MathAction(verb=Verb.FORM_HYPOTHESIS)]
    with pytest.raises(ValueError, match="does not match the .*embed_dim"):
        prior_fn(state, actions)
