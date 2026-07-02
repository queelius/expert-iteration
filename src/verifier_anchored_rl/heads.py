"""Trainable Q-head and policy head over the frozen-LLM embedding.

The framework's central learned object is an approximate action-value
function ``Q_psi(s, a)`` over the six-verb action set, plus a policy
``pi_theta(a | s)`` used as the PUCT prior during search and as the
fast-mode acting policy at deployment (paper section 4.5). Both heads
read the same fixed-length state representation produced by the frozen
LLM's hidden state at the ``[STATE_END]`` token.

This module implements the parametric heads in PyTorch. They are
deliberately small (~1M parameters total for a 2048-d input) because
the heavy lifting is done by the frozen LLM upstream; the heads only
have to map the embedding to verb-categorical outputs.

Architecture (depth-3 MLP with a shared trunk):

    embedding (2048-d)
        |
        v
    LayerNorm
        |
        v
    Linear(2048 -> 512), GELU, Dropout
        |
        v
    Linear(512 -> 256), GELU
        |
        v
        +------------+
        |            |
        v            v
    PolicyHead    QHead
    (256 -> NUM_VERBS)    (256 -> NUM_VERBS)
        |            |
        logits      q_values

Policy logits are masked to the available action set before softmax.
The Q-head is trained on the chance-node mean values ``Qhat_MCTS(s, a)``
that MCTS produces; only verbs that MCTS expanded contribute to the
loss, with the rest masked out.

Loss functions distill from MCTS targets:

* ``policy_cross_entropy(model, batch)``: ``-sum_a pi_MCTS(a) log
  pi_theta(a)``, summed only over the available action set (since both
  distributions are zero on unavailable actions by construction).
* ``q_mse(model, batch)``: mean-squared error on the verbs where the
  batch supplies a target (the ``q_target_mask`` indicates which).

Both losses operate on a batch dict produced by ``encode_batch`` (see
``experiments/train_heads.py`` for the call site).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - tests skip if torch missing
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from .deliberation_mdp import Verb


# -----------------------------------------------------------------------------
# Verb ordering for tensor layout
# -----------------------------------------------------------------------------


# The position in this list is the index used along the categorical
# dimension of every tensor in this module. Do not reorder.
VERB_ORDER: list[Verb] = list(Verb)
NUM_VERBS: int = len(VERB_ORDER)
VERB_NAME_TO_INDEX: dict[str, int] = {v.value: i for i, v in enumerate(VERB_ORDER)}


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the trainable heads. "
            "Install with: pip install torch"
        )


# -----------------------------------------------------------------------------
# Architecture
# -----------------------------------------------------------------------------


if _TORCH_AVAILABLE:

    class StateEncoder(nn.Module):
        """Shared trunk: 2048-d embedding -> ``repr_dim`` state representation.

        LayerNorm at the input stabilizes against the L2-normalized
        Ollama embeddings (whose component magnitudes scale with 1/sqrt(d)).
        Dropout in the first hidden layer regularizes the small dataset.
        """

        def __init__(
            self,
            embed_dim: int = 2048,
            hidden_dim: int = 512,
            repr_dim: int = 256,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.embed_dim = embed_dim
            self.repr_dim = repr_dim
            self.norm = nn.LayerNorm(embed_dim)
            self.fc1 = nn.Linear(embed_dim, hidden_dim)
            self.act1 = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_dim, repr_dim)
            self.act2 = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.norm(x)
            x = self.fc1(x)
            x = self.act1(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.act2(x)
            return x

    class PolicyHead(nn.Module):
        """``pi_theta`` logits: ``repr_dim -> NUM_VERBS``."""

        def __init__(self, repr_dim: int = 256, num_verbs: int = NUM_VERBS):
            super().__init__()
            self.linear = nn.Linear(repr_dim, num_verbs)

        def forward(self, repr_: torch.Tensor) -> torch.Tensor:
            return self.linear(repr_)

    class QHead(nn.Module):
        """``Q_psi(s, *)`` scalars: ``repr_dim -> NUM_VERBS``.

        Output is unbounded; we do not apply tanh / sigmoid because the
        targets are chance-node means which can be near 1.0 at terminal
        commits and slightly negative on shaped verifier failures. A
        small final bias initialization keeps the head near zero on
        random embeddings.
        """

        def __init__(self, repr_dim: int = 256, num_verbs: int = NUM_VERBS):
            super().__init__()
            self.linear = nn.Linear(repr_dim, num_verbs)
            nn.init.zeros_(self.linear.bias)
            nn.init.normal_(self.linear.weight, std=0.01)

        def forward(self, repr_: torch.Tensor) -> torch.Tensor:
            return self.linear(repr_)

    class TrainableActor(nn.Module):
        """Encoder + policy head + Q-head, with mask-aware inference helpers."""

        def __init__(
            self,
            embed_dim: int = 2048,
            hidden_dim: int = 512,
            repr_dim: int = 256,
            dropout: float = 0.1,
            num_verbs: int = NUM_VERBS,
        ):
            super().__init__()
            self.encoder = StateEncoder(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                repr_dim=repr_dim,
                dropout=dropout,
            )
            self.policy_head = PolicyHead(repr_dim=repr_dim, num_verbs=num_verbs)
            self.q_head = QHead(repr_dim=repr_dim, num_verbs=num_verbs)
            self.num_verbs = num_verbs
            # Recorded so consumers (e.g. the trained-prior factory) can
            # validate that incoming embeddings match the dimension this
            # actor was trained on, instead of failing as an opaque shape
            # error mid-search when the embedder changes.
            self.embed_dim = embed_dim

        def forward(
            self, embedding: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            repr_ = self.encoder(embedding)
            policy_logits = self.policy_head(repr_)
            q_values = self.q_head(repr_)
            return policy_logits, q_values

        def pi(
            self,
            embedding: torch.Tensor,
            available_mask: torch.Tensor,
        ) -> torch.Tensor:
            """``pi_theta(. | s)`` masked-softmax over the available verbs.

            Unavailable positions in the returned distribution are exactly
            zero. ``embedding`` has shape ``(batch, embed_dim)`` and
            ``available_mask`` has shape ``(batch, NUM_VERBS)`` of dtype
            bool (or anything castable to bool).
            """
            logits, _ = self.forward(embedding)
            available_mask = available_mask.bool()
            if (~available_mask).all(dim=-1).any():
                # softmax over an all--inf row is NaN, and NaN silently
                # poisons every downstream argmax/agreement metric (numpy
                # argmax of an all-NaN row returns 0). Fail loud instead.
                raise ValueError(
                    "TrainableActor.pi: a batch row has no available verbs"
                )
            masked = logits.masked_fill(~available_mask, float("-inf"))
            return F.softmax(masked, dim=-1)

        def q(
            self,
            embedding: torch.Tensor,
        ) -> torch.Tensor:
            """``Q_psi(s, *)`` over all verbs; caller masks if needed."""
            _, q_values = self.forward(embedding)
            return q_values

else:
    # Stubs so module import does not fail when torch is missing.
    StateEncoder = None  # type: ignore[assignment]
    PolicyHead = None  # type: ignore[assignment]
    QHead = None  # type: ignore[assignment]
    TrainableActor = None  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# Loss functions
# -----------------------------------------------------------------------------


def policy_cross_entropy(
    model: Any,
    embedding: Any,
    pi_target: Any,
    available_mask: Any,
) -> Any:
    """Cross-entropy distillation loss from ``pi_MCTS`` to ``pi_theta``.

    Both distributions are over the seven categorical verbs. The
    expression is

        L_policy = - sum_{batch} sum_{a in avail(s)} pi_MCTS(a) log pi_theta(a)

    averaged over the batch. ``pi_target`` should sum to 1 along the
    verb dimension for entries where ``available_mask`` is True; the
    unavailable positions are not constrained (they will contribute 0
    to the loss because the target is 0 there and the masked log-softmax
    sends -inf positions to 0 after exp).
    """
    _require_torch()
    logits, _ = model(embedding)
    available_mask = available_mask.bool()
    masked_logits = logits.masked_fill(~available_mask, float("-inf"))
    log_pi = F.log_softmax(masked_logits, dim=-1)
    # `0 * -inf` would be NaN, so we replace log_pi at masked positions
    # with 0 before multiplying. Target is 0 there anyway, so this is
    # numerically clean.
    log_pi = torch.where(available_mask, log_pi, torch.zeros_like(log_pi))
    per_sample = -(pi_target * log_pi).sum(dim=-1)
    return per_sample.mean()


def q_mse(
    model: Any,
    embedding: Any,
    q_target: Any,
    q_target_mask: Any,
) -> Any:
    """Masked MSE between ``Q_psi(s, a)`` and the MCTS chance-node mean.

    Only positions where ``q_target_mask`` is True contribute to the
    loss. Per sample, we average over the contributing positions so
    that samples with more available verbs do not dominate.
    """
    _require_torch()
    _, q_pred = model(embedding)
    mask = q_target_mask.bool().float()
    sq = (q_pred - q_target) ** 2
    masked_sq = sq * mask
    num_contrib = mask.sum(dim=-1).clamp(min=1.0)
    per_sample = masked_sq.sum(dim=-1) / num_contrib
    return per_sample.mean()


# -----------------------------------------------------------------------------
# JSONL record -> tensor conversion
# -----------------------------------------------------------------------------


def encode_batch(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pack a list of distillation records into a batch dict of tensors.

    Each record is expected to have the schema produced by
    ``experiments/collect_distillation.py``:

    * ``embedding``: list[float] of length 2048
    * ``available_verbs``: list[str], values from ``Verb`` names
    * ``pi_mcts``: dict[str, float] (may not sum to 1 if MCTS missed
      some available verbs; we re-normalize)
    * ``q_mcts``: dict[str, float] over verbs MCTS gave estimates for

    Returns a dict with tensors:

    * ``embedding``: float32, shape ``(B, embed_dim)``
    * ``pi_target``: float32, shape ``(B, NUM_VERBS)``, rows sum to 1
    * ``q_target``: float32, shape ``(B, NUM_VERBS)``, 0 where missing
    * ``available_mask``: bool, shape ``(B, NUM_VERBS)``
    * ``q_target_mask``: bool, shape ``(B, NUM_VERBS)`` (verbs MCTS estimated)
    """
    _require_torch()
    B = len(records)
    if B == 0:
        raise ValueError("encode_batch: empty records list")
    # Fail loud on records whose embedding fetch failed at collection time
    # (collect_distillation writes embedding: null on fetch failure).
    # Without this, np.asarray(None, float32) is a 0-d NaN that broadcast-
    # fills the row, the loss goes NaN, and the trained checkpoint is
    # silently garbage. Callers should filter null-embedding records and
    # report the count before batching.
    for i, rec in enumerate(records):
        if not rec.get("embedding"):
            raise ValueError(
                f"encode_batch: record {i} "
                f"(problem={rec.get('problem_key')!r}, seed={rec.get('seed')!r}, "
                f"turn={rec.get('turn')!r}) has no embedding; filter "
                f"null-embedding records before batching"
            )
    embed_dim = len(records[0]["embedding"])

    embedding = np.zeros((B, embed_dim), dtype=np.float32)
    pi_target = np.zeros((B, NUM_VERBS), dtype=np.float32)
    q_target = np.zeros((B, NUM_VERBS), dtype=np.float32)
    available_mask = np.zeros((B, NUM_VERBS), dtype=bool)
    q_target_mask = np.zeros((B, NUM_VERBS), dtype=bool)

    for i, rec in enumerate(records):
        embedding[i] = np.asarray(rec["embedding"], dtype=np.float32)

        for verb_name in rec.get("available_verbs", []):
            idx = VERB_NAME_TO_INDEX.get(verb_name)
            if idx is not None:
                available_mask[i, idx] = True

        pi_dict = rec.get("pi_mcts", {})
        for verb_name, prob in pi_dict.items():
            idx = VERB_NAME_TO_INDEX.get(verb_name)
            if idx is not None:
                pi_target[i, idx] = float(prob)
        # Re-normalize over available positions (defensive).
        row_sum = pi_target[i].sum()
        if row_sum > 0:
            pi_target[i] /= row_sum

        q_dict = rec.get("q_mcts", {})
        for verb_name, value in q_dict.items():
            idx = VERB_NAME_TO_INDEX.get(verb_name)
            if idx is not None:
                q_target[i, idx] = float(value)
                q_target_mask[i, idx] = True

    return {
        "embedding": torch.from_numpy(embedding),
        "pi_target": torch.from_numpy(pi_target),
        "q_target": torch.from_numpy(q_target),
        "available_mask": torch.from_numpy(available_mask),
        "q_target_mask": torch.from_numpy(q_target_mask),
    }


__all__ = [
    "NUM_VERBS",
    "PolicyHead",
    "QHead",
    "StateEncoder",
    "TrainableActor",
    "VERB_ORDER",
    "VERB_NAME_TO_INDEX",
    "encode_batch",
    "policy_cross_entropy",
    "q_mse",
]
