"""Prioritized replay buffer for the approximate-Q-learning training loop.

method.tex Section 3.7 (``Replay buffer``): collected transitions are
stored keyed by the chance-node tuple, sampled with priority proportional
to the TD error ``|Q_psi(s, a) - Q_MCTS(s, a)|`` (prioritized-DQN style),
and evicted first-in-first-out under a capacity cap. Stale entries do not
break correctness because ``Q_MCTS`` and ``pi_MCTS`` are stored with the
sample rather than recomputed from the current heads.

A sample here is per-state (one frozen-LLM hidden state ``h`` plus the
all-verb target vectors), matching the multi-head architecture: one ``h``
yields all seven ``Q_psi(s, a)`` and all seven policy logits in one pass,
so the per-edge tuples of the paper are carried together as masked
vectors. Per-state priority scalarizes the per-edge TD errors by their
mean over the verbs MCTS actually estimated.

The buffer is torch-only (the [hf]/[policy] workflows that train the heads
require torch); it stores whatever device the added tensors are on and
returns stacked batches, leaving device placement to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .heads import NUM_VERBS, _TORCH_AVAILABLE, _require_torch

if _TORCH_AVAILABLE:
    import torch


@dataclass
class ReplaySample:
    """One per-state chance-node record.

    All vectors are length ``NUM_VERBS``. ``hidden`` is ``embed(s) = h_T``
    (the frozen-LLM state embedding; stored so training never re-runs the
    LLM, which is frozen). ``available_mask`` marks legal verbs;
    ``q_target`` / ``q_target_mask`` are the MCTS chance-node estimates and
    the verbs MCTS visited; ``pi_target`` is the visit-count distribution
    (sums to 1 over available verbs).
    """

    hidden: Any  # torch.Tensor [d]
    available_mask: Any  # bool [NUM_VERBS]
    q_target: Any  # float [NUM_VERBS]
    q_target_mask: Any  # bool [NUM_VERBS]
    pi_target: Any  # float [NUM_VERBS]


class ReplayBuffer:
    """Capacity-capped, FIFO-evicting, prioritized replay buffer."""

    def __init__(self, capacity: int, *, alpha: float = 0.6, eps: float = 1e-3):
        _require_torch()
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.alpha = alpha  # priority exponent (0 = uniform, 1 = full priority)
        self.eps = eps  # so zero-TD samples keep a nonzero sampling chance
        self._samples: list[ReplaySample] = []
        self._priorities = np.zeros(capacity, dtype=np.float64)
        self._pos = 0  # next write index (ring buffer)
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, sample: ReplaySample, priority: float | None = None) -> None:
        """Add a sample. New samples default to the current max priority so
        they are sampled at least once before their TD error is known
        (standard prioritized-replay practice)."""
        if priority is None:
            priority = float(self._priorities[: self._size].max()) if self._size else 1.0
        if len(self._samples) < self.capacity:
            self._samples.append(sample)
        else:
            self._samples[self._pos] = sample  # FIFO overwrite of oldest
        self._priorities[self._pos] = max(priority, self.eps)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[dict[str, Any], np.ndarray]:
        """Draw a prioritized minibatch. Returns ``(batch, indices)`` where
        ``batch`` is a dict of stacked tensors and ``indices`` are buffer
        positions for a later ``update_priorities`` call."""
        if self._size == 0:
            raise ValueError("cannot sample from an empty buffer")
        n = min(batch_size, self._size)
        scaled = self._priorities[: self._size] ** self.alpha
        probs = scaled / scaled.sum()
        indices = rng.choice(self._size, size=n, replace=False, p=probs)

        chosen = [self._samples[i] for i in indices]
        batch = {
            "hidden": torch.stack([s.hidden for s in chosen]),
            "available_mask": torch.stack([s.available_mask for s in chosen]),
            "q_target": torch.stack([s.q_target for s in chosen]),
            "q_target_mask": torch.stack([s.q_target_mask for s in chosen]),
            "pi_target": torch.stack([s.pi_target for s in chosen]),
        }
        return batch, indices

    def update_priorities(
        self, indices: np.ndarray, priorities: np.ndarray
    ) -> None:
        """Set sampling priority (typically the per-sample mean TD error)
        for the given buffer positions, floored at ``eps``."""
        for i, p in zip(indices, priorities):
            self._priorities[int(i)] = max(float(p), self.eps)


def sample_from_record(record: dict[str, Any]) -> ReplaySample:
    """Build a ``ReplaySample`` from a distillation JSONL record (the schema
    ``collect_distillation`` writes), using its ``embedding`` as ``hidden``.

    Reuses ``heads.encode_batch`` on the single record so the verb-index
    mapping, target renormalization, and masking match training exactly.
    """
    _require_torch()
    from .heads import encode_batch

    batch = encode_batch([record])
    return ReplaySample(
        hidden=batch["embedding"][0],
        available_mask=batch["available_mask"][0],
        q_target=batch["q_target"][0],
        q_target_mask=batch["q_target_mask"][0],
        pi_target=batch["pi_target"][0],
    )


__all__ = ["ReplayBuffer", "ReplaySample", "sample_from_record"]
