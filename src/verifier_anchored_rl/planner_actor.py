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
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)  # FIFO/LRU eviction of the oldest
        return prefill

    def clear(self) -> None:
        """Empty the cache in place. Callers sharing this instance (e.g. an
        LLMContextMDP and a StateActor holding the same cache) keep their
        reference and see the cleared cache."""
        self._store.clear()


class StateActor:
    """Heads-over-MathState bridge: embed(s), prior(s), leaf_value(s).

    ``prior`` reads the ONLINE policy head (the PUCT prior); ``leaf_value``
    reads the TARGET-Q head (the bootstrap leaf), matching method.tex 3.7.
    Both run under no_grad: they drive search, not training.
    """

    def __init__(self, encoder: Encoder, actor: Any, cache: PrefillCache | None = None,
                 *, depth_limit: int = 3):
        self.encoder = encoder
        self.actor = actor
        self.cache = cache if cache is not None else PrefillCache()
        self.depth_limit = depth_limit

    def embed(self, state: Any) -> Any:
        from .llm_context_mdp import _state_prefix

        return self.cache.get_or_encode(_state_prefix(state), self.encoder).hidden

    def _head_input(self, state: Any) -> Any:
        """``embed(s)`` reconciled to the heads' device and dtype.

        The frozen encoder runs on the GPU in bf16, so ``embed(s) = h_T`` is a
        bf16 CUDA tensor, whereas the small trainable heads live wherever their
        parameters do (fp32 on CPU in the harness). Feeding the raw hidden
        state straight into the heads is a device/dtype mismatch; cast it to
        the heads' own (device, dtype) so the bridge works regardless of where
        either side lives."""
        h = self.embed(state).unsqueeze(0)
        ref = next(self.actor.parameters(), None)
        if ref is not None:
            h = h.to(device=ref.device, dtype=ref.dtype)
        return h

    def prior(self, state: Any, actions: list[Any]) -> list[float]:
        import torch

        from .heads import NUM_VERBS, VERB_NAME_TO_INDEX

        idxs = [VERB_NAME_TO_INDEX[a.verb.value] for a in actions]
        with torch.no_grad():
            h = self._head_input(state)
            mask = torch.zeros(1, NUM_VERBS, dtype=torch.bool, device=h.device)
            for j in idxs:
                mask[0, j] = True
            pi = self.actor.pi(h, mask)  # masked softmax, 0 off the mask
        return [float(pi[0, j]) for j in idxs]

    def leaf_value(self, state: Any) -> float:
        import torch

        from .heads import VERB_NAME_TO_INDEX
        from .llm_context_mdp import available_actions

        avail = list(available_actions(state, depth_limit=self.depth_limit))
        if not avail:
            return 0.0  # defensive: MCTS only calls value_fn at non-terminal leaves
        with torch.no_grad():
            h = self._head_input(state)
            qt = self.actor.target_q(h)[0]
        return max(float(qt[VERB_NAME_TO_INDEX[v.value]]) for v in avail)


__all__ = ["Encoder", "PrefillCache", "StateActor"]
