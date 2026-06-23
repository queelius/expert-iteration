"""Decoupled state embedder: adapts an EmbeddingClient (text -> vector) to the
StateActor's Encoder protocol, so the heads can use a dedicated embedding model
instead of the generator LLM's hidden state. This frees generation to be served
by vLLM (which does not expose hidden states). The legacy LLM h_T path
(SharedForwardEncoder) remains a separate Encoder implementation, selectable as
a baseline; a future GraphEmbedder (GNN over the reasoning graph) will implement
the same protocol.
"""

from __future__ import annotations

from typing import Any


class TextEmbedder:
    """Encoder backed by a text EmbeddingClient. ``encode(text)`` returns a
    StatePrefill whose ``hidden`` is the embedding vector (fp32, on CPU) and
    whose KV cache is None (there is no generation cache on this path)."""

    def __init__(self, embedding_client: Any, embed_dim: int):
        self._client = embedding_client
        self._embed_dim = int(embed_dim)
        # FLOP-accounting parity with the encoder interface.
        self.prefill_calls = 0
        self.prefill_tokens = 0
        self.gen_calls = 0
        self.gen_tokens = 0

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def encode(self, prefix_text: str):
        import torch

        from .shared_forward import StatePrefill

        vec = self._client.embed(prefix_text)
        if len(vec) != self._embed_dim:
            raise ValueError(
                f"TextEmbedder: embedding dim {len(vec)} != configured "
                f"{self._embed_dim} (embedder/heads mismatch)"
            )
        self.prefill_calls += 1
        self.prefill_tokens += max(1, len(prefix_text.split()))
        return StatePrefill(
            hidden=torch.tensor(vec, dtype=torch.float32),
            cache=None, n_prefix_tokens=0, prefix_ids=None,
        )


__all__ = ["TextEmbedder"]
