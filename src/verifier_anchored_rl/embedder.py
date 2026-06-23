"""Decoupled state embedder: adapts an EmbeddingClient (text -> vector) to the
StateActor's Encoder protocol, so the heads can use a dedicated embedding model
instead of the generator LLM's hidden state. This frees generation to be served
by vLLM (which does not expose hidden states). The legacy LLM h_T path
(SharedForwardEncoder) remains a separate Encoder implementation, selectable as
a baseline; a future GraphEmbedder (GNN over the reasoning graph) will implement
the same protocol.
"""

from __future__ import annotations

import threading
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
        self._counter_lock = threading.Lock()

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
        # Use true token count when the client exposes a tokenizer, so this
        # counter has the same unit as SharedForwardEncoder.prefill_tokens (both
        # feed IterationStats.prefill_tokens). Fall back to whitespace word count
        # when no tokenizer is available (e.g. HTTP-only embedding clients).
        tokenizer = getattr(self._client, "tokenizer", None)
        if tokenizer is not None:
            try:
                n_tok = len(tokenizer(prefix_text)["input_ids"])
            except Exception:  # noqa: BLE001
                n_tok = max(1, len(prefix_text.split()))
        else:
            n_tok = max(1, len(prefix_text.split()))
        with self._counter_lock:
            self.prefill_calls += 1
            self.prefill_tokens += n_tok
        return StatePrefill(
            hidden=torch.tensor(vec, dtype=torch.float32),
            cache=None, n_prefix_tokens=0, prefix_ids=None,
        )


__all__ = ["TextEmbedder"]
