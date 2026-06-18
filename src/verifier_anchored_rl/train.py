"""Approximate-Q-learning training step and loop (method.tex Section 3.7).

This is the learnable core of the benchmark study's headline algorithm:
approximate Q-learning with MCTS-supplied targets, alternated with policy
distillation (expert iteration). It is what makes the study ``approximate
Q-learning`` rather than the pilot's one-shot distillation. Three pieces
the pilot lacked, all here:

* the **combined objective** ``L_total = L_Q + lambda * L_pi`` (Eqs.
  lossq, losspi, losstotal): masked MSE of ``Q_psi`` against the MCTS
  chance-node estimate ``Q_MCTS`` (the paper's stated novelty: the target
  is ``Q_MCTS``, not a single-sample bootstrapped Bellman target), plus
  cross-entropy of the masked policy head against the visit-count
  distribution ``pi_MCTS``;
* the **prioritized replay buffer** (``replay.ReplayBuffer``), sampled by
  TD-error magnitude and updated after each step;
* the **target-Q network sync** every ``N`` optimizer steps
  (``HiddenStateActor.sync_target``), whose role is as the bootstrapped
  leaf evaluator inside MCTS during the next collection round.

The losses operate on the actor's explicit ``q(h)`` and
``policy_logits(h)`` accessors (not the forward tuple, whose order is an
implementation detail), so they are unambiguous about which head is which.
The outer collection-and-train loop (MCTS-driven, GPU-bound) wires this
core to the encoder and the planner; this module is the CPU-testable part:
losses, the optimizer step, priority computation, and the sync schedule.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .heads import _TORCH_AVAILABLE, _require_torch

if _TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F


def q_loss(actor: Any, hidden: Any, q_target: Any, q_target_mask: Any) -> Any:
    """Masked MSE of ``Q_psi(s, *)`` against ``Q_MCTS`` (Eq. lossq).

    Only verbs MCTS estimated (``q_target_mask`` True) contribute; per
    sample the squared errors are averaged over the contributing verbs so
    states with more visited verbs do not dominate, then averaged over the
    batch.
    """
    q_pred = actor.q(hidden)
    mask = q_target_mask.bool().float()
    sq = (q_pred - q_target) ** 2 * mask
    num = mask.sum(dim=-1).clamp(min=1.0)
    return (sq.sum(dim=-1) / num).mean()


def policy_loss(
    actor: Any, hidden: Any, pi_target: Any, available_mask: Any
) -> Any:
    """Masked cross-entropy from ``pi_MCTS`` to ``pi_theta`` (Eq. losspi).

    Masked log-softmax over available verbs; ``0 * -inf`` at unavailable
    positions is zeroed before the product (the target is 0 there anyway).
    """
    logits = actor.policy_logits(hidden)
    available = available_mask.bool()
    masked_logits = logits.masked_fill(~available, float("-inf"))
    log_pi = F.log_softmax(masked_logits, dim=-1)
    log_pi = torch.where(available, log_pi, torch.zeros_like(log_pi))
    return -(pi_target * log_pi).sum(dim=-1).mean()


def td_errors(
    actor: Any, hidden: Any, q_target: Any, q_target_mask: Any
) -> np.ndarray:
    """Per-sample priority signal: mean ``|Q_psi(s, a) - Q_MCTS(s, a)|`` over
    the verbs MCTS estimated (Eq. lossq's residual). Detached, on CPU."""
    with torch.no_grad():
        q_pred = actor.q(hidden)
        mask = q_target_mask.bool().float()
        abs_err = (q_pred - q_target).abs() * mask
        num = mask.sum(dim=-1).clamp(min=1.0)
        per_sample = abs_err.sum(dim=-1) / num
    return per_sample.cpu().numpy()


def train_step(
    actor: Any,
    optimizer: Any,
    batch: dict[str, Any],
    *,
    lambda_pi: float = 1.0,
) -> tuple[dict[str, float], np.ndarray]:
    """One optimizer step on ``L_Q + lambda * L_pi`` (Eq. losstotal).

    Returns ``(losses, td)`` where ``losses`` has float ``q``, ``pi``,
    ``total`` and ``td`` is the per-sample TD error for a priority update.
    """
    _require_torch()
    actor.train()
    optimizer.zero_grad()
    q_l = q_loss(actor, batch["hidden"], batch["q_target"], batch["q_target_mask"])
    pi_l = policy_loss(
        actor, batch["hidden"], batch["pi_target"], batch["available_mask"]
    )
    total = q_l + lambda_pi * pi_l
    total.backward()
    optimizer.step()
    td = td_errors(actor, batch["hidden"], batch["q_target"], batch["q_target_mask"])
    return (
        {"q": float(q_l.item()), "pi": float(pi_l.item()), "total": float(total.item())},
        td,
    )


class Trainer:
    """Runs minibatch steps from a replay buffer with periodic target sync.

    ``sync_interval`` is the ``N`` of Algorithm 1 line 14: every ``N``
    optimizer steps the target-Q network is hard-synced to the online
    network (method.tex flags ``N`` on the order of one thousand steps;
    faster sync destabilizes the leaf-value estimate, slower wastes
    signal).
    """

    def __init__(
        self,
        actor: Any,
        optimizer: Any,
        buffer: Any,
        *,
        sync_interval: int = 1000,
        lambda_pi: float = 1.0,
    ):
        _require_torch()
        self.actor = actor
        self.optimizer = optimizer
        self.buffer = buffer
        self.sync_interval = sync_interval
        self.lambda_pi = lambda_pi
        self.global_step = 0

    def step(self, batch_size: int, rng: np.random.Generator) -> dict[str, float]:
        batch, indices = self.buffer.sample(batch_size, rng)
        losses, td = train_step(
            self.actor, self.optimizer, batch, lambda_pi=self.lambda_pi
        )
        self.buffer.update_priorities(indices, td)
        self.global_step += 1
        if self.global_step % self.sync_interval == 0:
            self.actor.sync_target()
        return losses

    def run(
        self, num_steps: int, batch_size: int, rng: np.random.Generator
    ) -> list[dict[str, float]]:
        return [self.step(batch_size, rng) for _ in range(num_steps)]


__all__ = ["Trainer", "policy_loss", "q_loss", "td_errors", "train_step"]
