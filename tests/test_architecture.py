"""Tests for the benchmark-study architecture (HiddenStateActor).

Pure-torch over hidden-state tensors, so these run on CPU with no model.
They verify the method.tex Section 3.4 properties by construction:
shared adapter, masked policy, Q-head gradient routing (only the acted
row of W_Q plus the shared body), the target-Q network (sync, staleness,
policy exclusion), native-dimension flexibility, and the sub-5M
trainable-parameter claim.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verifier_anchored_rl.architecture import HiddenStateActor, make_actor
from verifier_anchored_rl.heads import NUM_VERBS


def test_forward_shapes_and_native_dimension():
    for d in (1536, 2048, 3584):
        actor = HiddenStateActor(embed_dim=d)
        h = torch.randn(4, d)
        q, logits = actor(h)
        assert q.shape == (4, NUM_VERBS)
        assert logits.shape == (4, NUM_VERBS)
        assert actor.embed_dim == d


def test_masked_policy_zeros_unavailable_and_sums_to_one():
    actor = HiddenStateActor(embed_dim=64)
    h = torch.randn(3, 64)
    mask = torch.tensor(
        [[True, True, False, False, False, False]] * 3, dtype=torch.bool
    )
    pi = actor.pi(h, mask)
    assert torch.allclose(pi.sum(dim=-1), torch.ones(3), atol=1e-6)
    assert torch.all(pi[:, 2:] == 0.0)
    assert torch.all(pi[:, :2] > 0.0)


def test_pi_raises_on_all_masked_row():
    actor = HiddenStateActor(embed_dim=64)
    h = torch.randn(1, 64)
    mask = torch.zeros(1, NUM_VERBS, dtype=torch.bool)
    with pytest.raises(ValueError, match="no available verbs"):
        actor.pi(h, mask)


def test_adapter_is_shared_by_both_heads():
    actor = HiddenStateActor(embed_dim=64)
    h = torch.randn(2, 64)
    # A loss touching only the policy head still flows gradient into the
    # shared adapter (the property that lets one forward feed both heads).
    actor.zero_grad()
    _, logits = actor(h)
    logits.sum().backward()
    assert actor.adapter.fc1.weight.grad is not None
    assert actor.adapter.fc1.weight.grad.abs().sum() > 0
    # And a Q-only loss likewise reaches the same adapter parameters.
    actor.zero_grad()
    actor.q(h).sum().backward()
    assert actor.adapter.fc1.weight.grad.abs().sum() > 0


def test_q_head_gradient_routes_to_acted_row_only():
    # method.tex 3.4: backprop on (s, a_k) updates row k of W_Q plus the
    # shared body; the other rows of W_Q get zero gradient from that sample.
    actor = HiddenStateActor(embed_dim=64)
    h = torch.randn(1, 64)
    k = 3
    actor.zero_grad()
    q = actor.q(h)
    (q[0, k] ** 2).backward()
    grad = actor.q_head.w.weight.grad  # (NUM_VERBS, head_hidden)
    assert grad[k].abs().sum() > 0
    other = [j for j in range(NUM_VERBS) if j != k]
    assert grad[other].abs().sum() == 0
    # The shared adapter still receives gradient (the "shared body").
    assert actor.adapter.fc1.weight.grad.abs().sum() > 0


def test_target_starts_synced():
    actor = HiddenStateActor(embed_dim=64)
    h = torch.randn(5, 64)
    assert torch.allclose(actor.q(h), actor.target_q(h), atol=1e-6)


def test_target_is_stale_until_synced():
    actor = HiddenStateActor(embed_dim=64)
    h = torch.randn(5, 64)
    opt = torch.optim.SGD(actor.parameters(), lr=1.0)
    opt.zero_grad()
    actor.q(h).pow(2).sum().backward()
    opt.step()  # online Q moves
    assert not torch.allclose(actor.q(h), actor.target_q(h), atol=1e-4)
    actor.sync_target()
    assert torch.allclose(actor.q(h), actor.target_q(h), atol=1e-6)


def test_target_excludes_policy_head_and_is_frozen():
    actor = HiddenStateActor(embed_dim=64)
    # The target copy mirrors adapter + Q-head, not the policy head.
    assert hasattr(actor, "_target_adapter")
    assert hasattr(actor, "_target_q_head")
    assert not hasattr(actor, "_target_policy_head")
    # Target params carry no gradient.
    for p in actor._target_adapter.parameters():
        assert not p.requires_grad
    for p in actor._target_q_head.parameters():
        assert not p.requires_grad


def test_target_q_has_no_grad():
    actor = HiddenStateActor(embed_dim=64)
    h = torch.randn(2, 64)
    out = actor.target_q(h)
    assert not out.requires_grad


def test_trainable_param_count_under_five_million():
    actor = HiddenStateActor(embed_dim=2048)
    n = actor.trainable_parameter_count()
    assert n < 5_000_000, n
    # Target params are excluded from the trainable count.
    total = sum(p.numel() for p in actor.parameters())
    assert total > n  # the frozen target copy adds non-trainable params


def test_make_actor_entry_point():
    actor = make_actor(embed_dim=1536, repr_dim=128)
    assert isinstance(actor, HiddenStateActor)
    assert actor.embed_dim == 1536
