"""Factored verb, pointer, and Q heads for Phase B.

The legacy ``heads`` module is verb-only.  This module implements the design
contract over a variable node set without changing that historical path.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by minimal installs
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from .heads import NUM_VERBS


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for the Phase B heads")


if _TORCH_AVAILABLE:

    class FactoredController(nn.Module):
        """Small controller over global and per-node frozen features.

        Input shapes are ``global_features: [B, D]`` and
        ``node_features: [B, N, D]``.  The pair mask is ``[B, V, N]`` and is
        the sole legality source for both factored policy stages.
        """

        def __init__(
            self,
            feature_dim: int,
            *,
            hidden_dim: int = 128,
            repr_dim: int = 64,
            node_repr_dim: int = 64,
            verb_embed_dim: int = 16,
            num_verbs: int = NUM_VERBS,
        ):
            super().__init__()
            self.feature_dim = feature_dim
            self.num_verbs = num_verbs
            self.global_adapter = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, repr_dim),
                nn.GELU(),
            )
            self.node_adapter = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, node_repr_dim),
                nn.GELU(),
            )
            self.verb_head = nn.Linear(repr_dim, num_verbs)
            self.verb_embedding = nn.Embedding(num_verbs, verb_embed_dim)
            pair_dim = repr_dim + node_repr_dim + verb_embed_dim
            self.pointer_head = nn.Sequential(
                nn.Linear(pair_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.q_head = nn.Sequential(
                nn.Linear(pair_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

        def _represent(
            self, global_features: torch.Tensor, node_features: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return (
                self.global_adapter(global_features),
                self.node_adapter(node_features),
            )

        def verb_logits(self, global_features: torch.Tensor) -> torch.Tensor:
            return self.verb_head(self.global_adapter(global_features))

        def pair_scores(
            self, global_features: torch.Tensor, node_features: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Return pointer logits and Q values with shape ``[B, V, N]``."""
            global_repr, node_repr = self._represent(global_features, node_features)
            batch, nodes, _ = node_repr.shape
            verbs = self.verb_embedding.weight.shape[0]
            global_grid = global_repr[:, None, None, :].expand(-1, verbs, nodes, -1)
            node_grid = node_repr[:, None, :, :].expand(-1, verbs, -1, -1)
            verb_grid = self.verb_embedding.weight[None, :, None, :].expand(
                batch, -1, nodes, -1
            )
            pair_repr = torch.cat((global_grid, node_grid, verb_grid), dim=-1)
            return (
                self.pointer_head(pair_repr).squeeze(-1),
                self.q_head(pair_repr).squeeze(-1),
            )

        def masked_policy(
            self,
            global_features: torch.Tensor,
            node_features: torch.Tensor,
            legal_pair_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Return verb probabilities and conditional pointer probabilities."""
            legal_pair_mask = legal_pair_mask.bool()
            legal_verbs = legal_pair_mask.any(dim=-1)
            if (~legal_verbs).all(dim=-1).any():
                raise ValueError("a batch row has no legal action pairs")
            verb_logits = self.verb_logits(global_features)
            verb_prob = F.softmax(
                verb_logits.masked_fill(~legal_verbs, float("-inf")), dim=-1
            )
            pointer_logits, _ = self.pair_scores(global_features, node_features)
            # Rows for illegal verbs have no valid pointer distribution.  Zero
            # them after softmax rather than creating all-negative-infinity NaNs.
            safe_mask = legal_pair_mask.clone()
            safe_mask[~legal_verbs] = True
            pointer_prob = F.softmax(
                pointer_logits.masked_fill(~safe_mask, float("-inf")), dim=-1
            )
            pointer_prob = torch.where(
                legal_verbs.unsqueeze(-1), pointer_prob, torch.zeros_like(pointer_prob)
            )
            return verb_prob, pointer_prob

        def masked_joint_policy(
            self,
            global_features: torch.Tensor,
            node_features: torch.Tensor,
            legal_pair_mask: torch.Tensor,
        ) -> torch.Tensor:
            """Return the normalized joint probability of every legal pair."""
            verb_prob, pointer_prob = self.masked_policy(
                global_features, node_features, legal_pair_mask)
            joint = verb_prob.unsqueeze(-1) * pointer_prob
            # The factorization is normalized by construction. Keep illegal
            # pairs exactly zero so global argmax is always legal.
            return joint.masked_fill(~legal_pair_mask.bool(), 0.0)

else:
    FactoredController = None  # type: ignore[assignment,misc]


def behavior_cloning_loss(
    model: Any,
    global_features: Any,
    node_features: Any,
    legal_pair_mask: Any,
    chosen_verbs: Any,
    chosen_nodes: Any,
) -> tuple[Any, Any, Any]:
    """Factored cross-entropy for the demonstrated legal pair."""
    _require_torch()
    legal_pair_mask = legal_pair_mask.bool()
    legal_verbs = legal_pair_mask.any(dim=-1)
    verb_logits = model.verb_logits(global_features).masked_fill(
        ~legal_verbs, float("-inf")
    )
    verb_loss = F.cross_entropy(verb_logits, chosen_verbs)

    pointer_logits, _ = model.pair_scores(global_features, node_features)
    batch_index = torch.arange(chosen_verbs.shape[0], device=chosen_verbs.device)
    selected_logits = pointer_logits[batch_index, chosen_verbs]
    selected_mask = legal_pair_mask[batch_index, chosen_verbs]
    pointer_loss = F.cross_entropy(
        selected_logits.masked_fill(~selected_mask, float("-inf")), chosen_nodes
    )
    return verb_loss + pointer_loss, verb_loss, pointer_loss


def selected_q_loss(
    model: Any,
    global_features: Any,
    node_features: Any,
    chosen_verbs: Any,
    chosen_nodes: Any,
    returns: Any,
) -> Any:
    """MSE of the demonstrated pair against its offline Monte Carlo return."""
    _require_torch()
    _, q_values = model.pair_scores(global_features, node_features)
    batch_index = torch.arange(chosen_verbs.shape[0], device=chosen_verbs.device)
    selected = q_values[batch_index, chosen_verbs, chosen_nodes]
    return F.mse_loss(selected, returns)


__all__ = [
    "FactoredController",
    "behavior_cloning_loss",
    "selected_q_loss",
]
