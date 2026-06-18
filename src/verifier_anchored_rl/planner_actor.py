"""Bridge between the trainable heads and the MCTS planner over MathState.

method.tex 3.7: the outer loop runs MCTS with the online policy head as the
PUCT prior and the target-Q head as the bootstrap leaf value, and feeds the
same per-state embedding embed(s) = h_T to the heads. This module wires
those three reads (embed / prior / leaf value) over the LLM context MDP's
MathState, depending only on a minimal encoder Protocol so a CPU fake can
drive the whole loop with no GPU.

MCTS interleaves prior(s), value_fn(s'), and sample_transition(s, a) across
states in no clean per-node order, so encode-once is made emergent: the
PrefillCache, keyed by the serialized state prefix, ensures a node is
prefilled exactly once and reused by the heads and (when the MDP runs the
encoder generation path) by every verb generation at that node. This is the
"one forward per decision" property (method.tex 3.4).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Protocol


class Encoder(Protocol):
    """Minimal encoder contract the heads bridge depends on."""

    @property
    def embed_dim(self) -> int: ...

    def encode(self, prefix_text: str) -> Any: ...  # returns a StatePrefill


class PrefillCache:
    """LRU cache from a state-prefix string to its StatePrefill.

    The cache makes encode-once emergent regardless of the order in which
    MCTS asks for a state's prior, leaf value, or generation: the first
    request for a prefix encodes it; the rest are cache hits.
    """

    def __init__(self, capacity: int = 512):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._store: "OrderedDict[str, Any]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._store)

    def get_or_encode(self, prefix_text: str, encoder: Encoder) -> Any:
        if prefix_text in self._store:
            self._store.move_to_end(prefix_text)
            return self._store[prefix_text]
        prefill = encoder.encode(prefix_text)
        self._store[prefix_text] = prefill
        self._store.move_to_end(prefix_text)
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)  # FIFO/LRU eviction of the oldest
        return prefill


__all__ = ["Encoder", "PrefillCache"]
