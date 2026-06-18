"""The benchmark-study architecture: frozen-LLM hidden state -> adapter ->
shared multi-head Q and policy, with a target-Q network.

This is the faithful implementation of paper/sections/method.tex
Section 3.4 (``Architecture: Shared LLM and Trainable Heads``), the model
the pre-registered study trains. It is distinct from the pilot's
``heads.TrainableActor`` in three ways that matter:

1. **It consumes the base model's hidden state at its native dimension**
   (1536 for Qwen2.5-1.5B, 2048 for the 3B, 3584 for 7B), not a fixed
   2048-d pooled vector from a separate embedding endpoint. The hidden
   state is ``embed(s) = h_T``, the hidden state at the ``[STATE_END]``
   sentinel position (method.tex Eq. embed), produced in-process by the
   same frozen model that generates verb content
   (``transformers_client``). Encoder and generator are the same weights,
   so representation drift is bounded by construction.

2. **One adapter is shared by both heads** (method.tex Eqs. adapter,
   qhead, policy). The adapter ``phi: R^d -> R^m`` (m=256, GELU)
   compresses the hidden state; the Q-head and policy head each add one
   hidden layer on top of ``phi(h)``. Sharing the adapter is what lets a
   single frozen-LLM forward feed both heads.

3. **It carries a target-Q network** (method.tex Eq. bellmantarget): a
   periodically-synced frozen copy of the adapter + Q-head used to form
   Bellman targets ``y = r + gamma * max_a' Q_target(s', a')``. The
   policy head is excluded from the target copy (it trains directly
   toward the MCTS visit-count distribution, not by bootstrapping). The
   pilot did pure one-step distillation and had no target network; the
   benchmark study's headline is approximate Q-learning, which needs it.

The module is pure torch over hidden-state tensors of shape ``(B, d)``;
the frozen-model forward that produces ``h`` lives in
``transformers_client`` and is GPU-bound, so this module is unit-testable
on CPU with no model. Trainable parameter count is well under the five
million the paper flags (Section 3.4, ``Phase 1 freezing rationale``):
for d=2048, m=256, hidden=256, it is about 0.7M.
"""

from __future__ import annotations

from typing import Any

from .heads import NUM_VERBS, _TORCH_AVAILABLE, _require_torch

if _TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Adapter(nn.Module):
        """``phi(h) = W2 sigma(W1 h + b1) + b2`` (method.tex Eq. adapter).

        A LayerNorm on the hidden-state input is included by default
        (not in the equation, but it stabilizes training across the wide
        magnitude range of un-normalized LLM hidden states; toggle off
        with ``layernorm=False`` to match the bare equation).
        """

        def __init__(
            self,
            embed_dim: int,
            hidden_dim: int = 512,
            repr_dim: int = 256,
            layernorm: bool = True,
        ):
            super().__init__()
            self.norm = nn.LayerNorm(embed_dim) if layernorm else nn.Identity()
            self.fc1 = nn.Linear(embed_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, repr_dim)

        def forward(self, h: torch.Tensor) -> torch.Tensor:
            return self.fc2(F.gelu(self.fc1(self.norm(h))))

    class _VerbHead(nn.Module):
        """One hidden layer on ``phi(h)`` to ``NUM_VERBS`` outputs:
        ``W sigma(U phi(h)) + b`` (method.tex Eqs. qhead, policy)."""

        def __init__(self, repr_dim: int = 256, hidden_dim: int = 256,
                     num_verbs: int = NUM_VERBS):
            super().__init__()
            self.u = nn.Linear(repr_dim, hidden_dim)
            self.w = nn.Linear(hidden_dim, num_verbs)

        def forward(self, repr_: torch.Tensor) -> torch.Tensor:
            return self.w(F.gelu(self.u(repr_)))

    class HiddenStateActor(nn.Module):
        """Adapter + shared multi-head Q and policy + target-Q network.

        ``forward(h)`` returns ``(q_values, policy_logits)``, each of
        shape ``(B, NUM_VERBS)``, from a single shared adapter pass.
        """

        def __init__(
            self,
            embed_dim: int,
            adapter_hidden: int = 512,
            repr_dim: int = 256,
            head_hidden: int = 256,
            layernorm: bool = True,
            num_verbs: int = NUM_VERBS,
        ):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_verbs = num_verbs
            self.adapter = Adapter(embed_dim, adapter_hidden, repr_dim, layernorm)
            self.q_head = _VerbHead(repr_dim, head_hidden, num_verbs)
            self.policy_head = _VerbHead(repr_dim, head_hidden, num_verbs)

            # Target copy of (adapter + Q-head) for Bellman targets. Built
            # as independent modules and kept in eval()/no-grad; the policy
            # head is deliberately not mirrored.
            self._target_adapter = Adapter(
                embed_dim, adapter_hidden, repr_dim, layernorm
            )
            self._target_q_head = _VerbHead(repr_dim, head_hidden, num_verbs)
            for p in self._target_adapter.parameters():
                p.requires_grad_(False)
            for p in self._target_q_head.parameters():
                p.requires_grad_(False)
            self.sync_target()

        # -- online forward -------------------------------------------------

        def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            repr_ = self.adapter(h)
            return self.q_head(repr_), self.policy_head(repr_)

        def q(self, h: torch.Tensor) -> torch.Tensor:
            """``Q_psi(s, *)`` over all verbs; caller masks if needed."""
            return self.q_head(self.adapter(h))

        def pi(self, h: torch.Tensor, available_mask: torch.Tensor) -> torch.Tensor:
            """``pi_theta(. | s)`` masked-softmax over available verbs
            (method.tex Eq. policy). Unavailable positions are exactly 0.
            Raises on an all-masked row (softmax over all -inf is NaN, which
            silently poisons downstream argmax)."""
            logits = self.policy_head(self.adapter(h))
            available_mask = available_mask.bool()
            if (~available_mask).all(dim=-1).any():
                raise ValueError(
                    "HiddenStateActor.pi: a batch row has no available verbs"
                )
            masked = logits.masked_fill(~available_mask, float("-inf"))
            return F.softmax(masked, dim=-1)

        # -- target network (method.tex Eq. bellmantarget) ------------------

        @torch.no_grad()
        def target_q(self, h: torch.Tensor) -> torch.Tensor:
            """``Q_target(s, *)`` from the frozen target copy (no grad)."""
            return self._target_q_head(self._target_adapter(h))

        @torch.no_grad()
        def sync_target(self) -> None:
            """Hard-copy the online adapter + Q-head into the target copy."""
            self._target_adapter.load_state_dict(self.adapter.state_dict())
            self._target_q_head.load_state_dict(self.q_head.state_dict())

        # -- parameter accounting ------------------------------------------

        def trainable_parameter_count(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

else:  # pragma: no cover
    Adapter = None  # type: ignore[assignment]
    HiddenStateActor = None  # type: ignore[assignment]


def make_actor(embed_dim: int, **kwargs: Any):
    """Construct a ``HiddenStateActor``, raising a clear error if torch
    is absent (the package imports without torch; this entry point does
    not)."""
    _require_torch()
    return HiddenStateActor(embed_dim=embed_dim, **kwargs)


__all__ = ["Adapter", "HiddenStateActor", "make_actor"]
