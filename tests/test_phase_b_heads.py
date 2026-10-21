from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from verifier_anchored_rl.heads import NUM_VERBS
from verifier_anchored_rl.phase_b_heads import (
    FactoredController,
    behavior_cloning_loss,
    selected_q_loss,
)


def _batch():
    global_features = torch.randn(2, 12)
    node_features = torch.randn(2, 4, 12)
    legal = torch.zeros(2, NUM_VERBS, 4, dtype=torch.bool)
    legal[0, 0, 0] = True
    legal[0, 2, 1] = True
    legal[1, 2, 0] = True
    legal[1, 2, 3] = True
    legal[1, 5, 2] = True
    return global_features, node_features, legal


def test_factored_policy_masks_verbs_and_targets():
    model = FactoredController(12, hidden_dim=16, repr_dim=8, node_repr_dim=8)
    global_features, node_features, legal = _batch()
    verb_prob, pointer_prob = model.masked_policy(
        global_features, node_features, legal
    )
    assert verb_prob.shape == (2, NUM_VERBS)
    assert pointer_prob.shape == (2, NUM_VERBS, 4)
    assert torch.allclose(verb_prob.sum(-1), torch.ones(2))
    assert verb_prob[0, 1] == 0
    assert pointer_prob[1, 2, 1] == 0
    assert torch.allclose(pointer_prob[1, 2].sum(), torch.tensor(1.0))
    assert pointer_prob[0, 1].sum() == 0


def test_factored_losses_are_finite_and_reach_every_head():
    model = FactoredController(12, hidden_dim=16, repr_dim=8, node_repr_dim=8)
    global_features, node_features, legal = _batch()
    chosen_verbs = torch.tensor([0, 2])
    chosen_nodes = torch.tensor([0, 3])
    returns = torch.tensor([1.0, -0.2])
    bc, verb, pointer = behavior_cloning_loss(
        model,
        global_features,
        node_features,
        legal,
        chosen_verbs,
        chosen_nodes,
    )
    q = selected_q_loss(
        model,
        global_features,
        node_features,
        chosen_verbs,
        chosen_nodes,
        returns,
    )
    total = bc + q
    total.backward()
    assert torch.isfinite(torch.stack((bc, verb, pointer, q))).all()
    assert model.verb_head.weight.grad is not None
    assert model.pointer_head[-1].weight.grad is not None
    assert model.q_head[-1].weight.grad is not None


def test_factored_policy_rejects_empty_action_menu():
    model = FactoredController(12, hidden_dim=16, repr_dim=8, node_repr_dim=8)
    global_features = torch.randn(1, 12)
    node_features = torch.randn(1, 2, 12)
    legal = torch.zeros(1, NUM_VERBS, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="no legal action"):
        model.masked_policy(global_features, node_features, legal)


def test_joint_policy_is_normalized_and_zero_on_illegal_pairs():
    model = FactoredController(12, hidden_dim=16, repr_dim=8, node_repr_dim=8)
    global_features, node_features, legal = _batch()
    joint = model.masked_joint_policy(
        global_features, node_features, legal)

    assert joint.shape == legal.shape
    assert torch.allclose(joint.sum(dim=(1, 2)), torch.ones(2))
    assert torch.all(joint[~legal] == 0)
    verb_prob, pointer_prob = model.masked_policy(
        global_features, node_features, legal)
    assert torch.allclose(joint, verb_prob.unsqueeze(-1) * pointer_prob)
