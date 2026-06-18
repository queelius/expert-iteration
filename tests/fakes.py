"""Shared CPU test doubles for the outer-loop tests.

``FakeEncoder`` implements the ``planner_actor.Encoder`` protocol with a
deterministic hidden state derived from a hash of the prefix text, so the
whole collection-and-train loop is testable with no GPU and no model. It
also provides ``instantiate_json`` (echo-valued, schema-conformant) so it
can stand in for the SharedForwardEncoder on the MDP's generation path,
and the same FLOP counters the real encoder maintains.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

from verifier_anchored_rl.shared_forward import StatePrefill


class FakeEncoder:
    def __init__(self, embed_dim: int = 32):
        self._embed_dim = embed_dim
        self.prefill_calls = 0
        self.prefill_tokens = 0
        self.gen_calls = 0
        self.gen_tokens = 0

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def encode(self, prefix_text: str) -> StatePrefill:
        import torch

        digest = hashlib.sha256(prefix_text.encode("utf-8")).digest()[:8]
        seed = int.from_bytes(digest, "little")
        vec = np.random.default_rng(seed).standard_normal(self._embed_dim)
        hidden = torch.from_numpy(vec.astype("float32"))
        n_tok = max(1, len(prefix_text.split()))
        self.prefill_calls += 1
        self.prefill_tokens += n_tok
        return StatePrefill(hidden=hidden, cache=None, n_prefix_tokens=n_tok,
                            prefix_ids=None)

    def instantiate_json(
        self, prefill: StatePrefill, suffix_text: str, schema: Mapping[str, Any],
        *, temperature: float = 1.0, max_tokens: int = 192,
        seed: int | None = None, reuse_cache: bool = True,
    ) -> tuple[dict[str, Any], int]:
        from verifier_anchored_rl.llm_client import _echo_value

        self.gen_calls += 1
        self.gen_tokens += 1
        props = schema.get("properties", {})
        return {k: _echo_value(spec) for k, spec in props.items()}, 1
