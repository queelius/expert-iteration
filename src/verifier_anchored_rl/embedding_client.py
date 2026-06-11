"""Embedding client abstraction.

The framework's Q-head and policy head take a fixed-length embedding of
the serialized state as input. That embedding comes from the same frozen
LLM that serves as the verb-instantiator, but accessed through a
different endpoint: ``/api/embed`` on Ollama, ``/v1/embeddings`` on
OpenAI-compatible servers, or any other backend that maps a string to a
vector.

This module mirrors the ``llm_client`` design: a Protocol that the
trainable heads program against, plus mock and HTTP implementations.

Three implementations:

* ``MockEmbeddingClient`` returns deterministic seed-based unit vectors.
  No network, no model. Used in tests and for offline development of
  the Q/policy heads before the real LLM is available.

* ``OllamaEmbeddingClient`` is an HTTP client for Ollama's ``/api/embed``
  endpoint. Ollama auto-normalizes the embedding to L2 norm 1.0.

* ``OpenAICompatibleEmbeddingClient`` is an HTTP client for any server
  that implements OpenAI's ``/v1/embeddings`` endpoint (vLLM, Tabby,
  OpenAI, etc.). Output is one vector per input string.

The Protocol method is ``embed(text) -> list[float]``. Batched embedding
is intentionally not part of the Protocol: most consumers fetch one
embedding per state, and batching is a backend optimization left to
specific implementations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingClient(Protocol):
    """Interface every embedding backend must satisfy."""

    def embed(self, text: str) -> list[float]:
        ...


class EmbeddingError(RuntimeError):
    """Raised when the backend cannot produce an embedding."""

    def __init__(self, message: str, *, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause


# -----------------------------------------------------------------------------
# Mock (deterministic seed-based)
# -----------------------------------------------------------------------------


@dataclass
class MockEmbeddingClient:
    """Returns a deterministic seed-based unit vector for each input string.

    The seed is derived from ``hash(text) ^ self.salt``. Same input
    always yields the same vector. Useful for testing head training
    without depending on a real LLM.
    """

    embedding_dim: int = 2048
    salt: int = 0
    calls: list[str] = field(default_factory=list)

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        # Stable seed across runs from text + salt; Python hash() is
        # randomized per-process by default, so we use a stable digest.
        import hashlib
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "big") ^ self.salt
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.embedding_dim).astype(np.float32)
        # L2 normalize so the mock behaves like Ollama's normalized output.
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


# -----------------------------------------------------------------------------
# Ollama HTTP backend
# -----------------------------------------------------------------------------


@dataclass
class OllamaEmbeddingClient:
    """HTTP client for Ollama's ``/api/embed`` endpoint.

    Example
    -------
    >>> client = OllamaEmbeddingClient(
    ...     base_url="http://192.168.0.204:11434/api/embed",
    ...     model="qwen2.5:3b",
    ... )
    >>> v = client.embed("Compute 7 * 6.")
    >>> len(v)  # qwen2.5:3b hidden dim
    2048
    """

    base_url: str
    model: str
    timeout_s: float = 30.0
    extra_headers: dict[str, str] = field(default_factory=dict)

    def embed(self, text: str) -> list[float]:
        import httpx  # lazy import; mock client works without httpx

        body = {"model": self.model, "input": text}
        headers = {"Content-Type": "application/json", **self.extra_headers}

        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                response = client.post(self.base_url, headers=headers, json=body)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            # json.JSONDecodeError is not an httpx.HTTPError: a 200 with a
            # non-JSON body (proxy error page, truncated stream) must still
            # surface as the typed EmbeddingError the Protocol promises.
            raise EmbeddingError(
                f"OllamaEmbeddingClient: HTTP/parse error at {self.base_url}",
                cause=e,
            )

        embeddings = payload.get("embeddings", [])
        if not embeddings:
            raise EmbeddingError(
                f"OllamaEmbeddingClient: empty embeddings response: {payload}"
            )
        return embeddings[0]


# -----------------------------------------------------------------------------
# OpenAI-compatible embeddings endpoint
# -----------------------------------------------------------------------------


@dataclass
class OpenAICompatibleEmbeddingClient:
    """HTTP client for any server with the OpenAI ``/v1/embeddings`` API.

    vLLM (with an embedding-capable model loaded), Tabby, and the
    commercial OpenAI endpoint all share this shape::

        POST /v1/embeddings
        {"model": "...", "input": "..."}
        -> {"data": [{"embedding": [...], "index": 0}]}
    """

    base_url: str
    model: str
    api_key: str | None = None
    timeout_s: float = 30.0
    extra_headers: dict[str, str] = field(default_factory=dict)

    def embed(self, text: str) -> list[float]:
        import httpx

        url = self.base_url.rstrip("/") + "/embeddings"
        body = {"model": self.model, "input": text}
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                response = client.post(url, headers=headers, json=body)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            raise EmbeddingError(
                f"OpenAICompatibleEmbeddingClient: HTTP/parse error at {url}",
                cause=e,
            )

        data = payload.get("data", [])
        if not data or "embedding" not in data[0]:
            raise EmbeddingError(
                f"OpenAICompatibleEmbeddingClient: unexpected payload: {payload}"
            )
        return data[0]["embedding"]


# -----------------------------------------------------------------------------
# Endpoint-shape chooser
# -----------------------------------------------------------------------------


def make_embedding_client(
    url: str,
    model: str,
    *,
    timeout_s: float = 30.0,
    api_key: str | None = None,
) -> EmbeddingClient:
    """Choose the HTTP embedding client matching the endpoint URL shape.

    URLs containing ``/v1`` (the OpenAI-compatible convention used by
    vLLM, Tabby, and OpenAI itself) route to
    ``OpenAICompatibleEmbeddingClient``; everything else is treated as
    Ollama's ``/api/embed``. This is what lets experiment scripts accept
    a single ``--embed-url`` flag and work against either backend instead
    of silently failing on the response-shape mismatch.
    """
    normalized = url.rstrip("/")
    if "/v1" in normalized:
        # OpenAICompatibleEmbeddingClient appends /embeddings itself.
        if normalized.endswith("/embeddings"):
            normalized = normalized[: -len("/embeddings")]
        return OpenAICompatibleEmbeddingClient(
            base_url=normalized,
            model=model,
            timeout_s=timeout_s,
            api_key=api_key,
        )
    return OllamaEmbeddingClient(
        base_url=normalized,
        model=model,
        timeout_s=timeout_s,
    )


__all__ = [
    "EmbeddingClient",
    "EmbeddingError",
    "MockEmbeddingClient",
    "OllamaEmbeddingClient",
    "OpenAICompatibleEmbeddingClient",
    "make_embedding_client",
]
