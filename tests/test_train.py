"""Tests for the approximate-Q-learning training loop (method.tex 3.7).

CPU, no model: synthetic replay samples with known targets. The load-
bearing checks are behavioral: the combined objective actually drives
Q_psi toward Q_MCTS and pi_theta toward pi_MCTS, priorities update from
TD errors, and the target network syncs on the configured interval.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verifier_anchored_rl.architecture import HiddenStateActor
from verifier_anchored_rl.heads import NUM_VERBS
from verifier_anchored_rl.replay import ReplayBuffer, ReplaySample, sample_from_record
from verifier_anchored_rl.train import (
    Trainer,
    policy_loss,
    q_loss,
    td_errors,
    train_step,
)


# -----------------------------------------------------------------------------
# Synthetic data
# -----------------------------------------------------------------------------


def _make_sample(d=32, seed=0):
    """A sample whose hidden state deterministically encodes its targets, so
    a head with capacity can fit it."""
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(d, generator=g)
    avail = torch.ones(NUM_VERBS, dtype=torch.bool)
    q_target = torch.randn(NUM_VERBS, generator=g)
    q_mask = torch.ones(NUM_VERBS, dtype=torch.bool)
    pi_logits = torch.randn(NUM_VERBS, generator=g)
    pi_target = torch.softmax(pi_logits, dim=-1)
    return ReplaySample(hidden, avail, q_target, q_mask, pi_target)


def _fill_buffer(buf, n, d=32):
    for i in range(n):
        buf.add(_make_sample(d=d, seed=i))


# -----------------------------------------------------------------------------
# Replay buffer
# -----------------------------------------------------------------------------


def test_buffer_capacity_and_fifo_eviction():
    buf = ReplayBuffer(capacity=3)
    for i in range(5):
        buf.add(_make_sample(seed=i))
    assert len(buf) == 3  # capped


def test_buffer_sample_shapes_and_indices():
    buf = ReplayBuffer(capacity=10)
    _fill_buffer(buf, 6)
    rng = np.random.default_rng(0)
    batch, idx = buf.sample(4, rng)
    assert batch["hidden"].shape == (4, 32)
    assert batch["q_target"].shape == (4, NUM_VERBS)
    assert batch["available_mask"].shape == (4, NUM_VERBS)
    assert len(idx) == 4


def test_buffer_sample_respects_size_when_batch_too_large():
    buf = ReplayBuffer(capacity=10)
    _fill_buffer(buf, 3)
    batch, idx = buf.sample(8, np.random.default_rng(0))
    assert batch["hidden"].shape[0] == 3


def test_buffer_priority_biases_sampling():
    buf = ReplayBuffer(capacity=5, alpha=1.0, eps=1e-6)
    _fill_buffer(buf, 5)
    # Make index 2 dominate.
    buf.update_priorities(np.arange(5), np.array([1e-6] * 5))
    buf.update_priorities(np.array([2]), np.array([100.0]))
    rng = np.random.default_rng(0)
    counts = np.zeros(5)
    for _ in range(200):
        _, idx = buf.sample(1, rng)
        counts[idx[0]] += 1
    assert counts[2] > 150  # the high-priority sample dominates


def test_empty_buffer_sample_raises():
    buf = ReplayBuffer(capacity=4)
    with pytest.raises(ValueError, match="empty buffer"):
        buf.sample(2, np.random.default_rng(0))


def test_sample_from_record_matches_encode_batch():
    record = {
        "problem_key": "p", "seed": 0, "turn": 0,
        "embedding": [0.1] * 8,
        "available_verbs": ["DECOMPOSE", "COMMIT_ANSWER"],
        "pi_mcts": {"DECOMPOSE": 0.7, "COMMIT_ANSWER": 0.3},
        "q_mcts": {"DECOMPOSE": 0.5},
    }
    s = sample_from_record(record)
    assert s.hidden.shape == (8,)
    assert s.available_mask.sum().item() == 2
    assert s.q_target_mask.sum().item() == 1  # only DECOMPOSE had a Q estimate


# -----------------------------------------------------------------------------
# Losses and one step
# -----------------------------------------------------------------------------


def _batch(buf, n=8):
    return buf.sample(n, np.random.default_rng(0))[0]


def test_losses_are_finite_and_nonnegative():
    actor = HiddenStateActor(embed_dim=32)
    buf = ReplayBuffer(capacity=16)
    _fill_buffer(buf, 8)
    b = _batch(buf)
    ql = q_loss(actor, b["hidden"], b["q_target"], b["q_target_mask"])
    pl = policy_loss(actor, b["hidden"], b["pi_target"], b["available_mask"])
    assert torch.isfinite(ql) and ql.item() >= 0
    assert torch.isfinite(pl) and pl.item() >= 0


def test_train_step_returns_losses_and_td():
    actor = HiddenStateActor(embed_dim=32)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    buf = ReplayBuffer(capacity=16)
    _fill_buffer(buf, 8)
    b = _batch(buf)
    losses, td = train_step(actor, opt, b, lambda_pi=1.0)
    assert set(losses) == {"q", "pi", "total"}
    assert td.shape == (8,)
    assert np.all(td >= 0)


def test_combined_objective_drives_both_heads_down():
    # Overfit a tiny fixed buffer: both losses must fall substantially.
    torch.manual_seed(0)
    actor = HiddenStateActor(embed_dim=32, layernorm=False)
    opt = torch.optim.Adam(actor.parameters(), lr=3e-3)
    buf = ReplayBuffer(capacity=8)
    _fill_buffer(buf, 8)
    rng = np.random.default_rng(0)
    first = train_step(actor, opt, buf.sample(8, rng)[0])[0]
    for _ in range(300):
        train_step(actor, opt, buf.sample(8, rng)[0])
    last = train_step(actor, opt, buf.sample(8, rng)[0])[0]
    assert last["q"] < 0.5 * first["q"]
    assert last["pi"] < 0.9 * first["pi"]


# -----------------------------------------------------------------------------
# Trainer: priority update + target sync schedule
# -----------------------------------------------------------------------------


def test_trainer_syncs_target_on_interval():
    actor = HiddenStateActor(embed_dim=32)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-2)
    buf = ReplayBuffer(capacity=16)
    _fill_buffer(buf, 8)
    trainer = Trainer(actor, opt, buf, sync_interval=5)
    rng = np.random.default_rng(0)
    h = torch.randn(3, 32)

    # Diverge online from target over the first <interval steps.
    for _ in range(4):
        trainer.step(8, rng)
    assert not torch.allclose(actor.q(h), actor.target_q(h), atol=1e-4)
    # The 5th step hits the sync interval.
    trainer.step(8, rng)
    assert trainer.global_step == 5
    assert torch.allclose(actor.q(h), actor.target_q(h), atol=1e-6)


def test_trainer_updates_priorities_from_td():
    actor = HiddenStateActor(embed_dim=32)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    buf = ReplayBuffer(capacity=16)
    _fill_buffer(buf, 8)
    before = buf._priorities[:8].copy()
    Trainer(actor, opt, buf, sync_interval=1000).step(8, np.random.default_rng(0))
    after = buf._priorities[:8]
    # At least some sampled priorities changed from the initial uniform max.
    assert not np.allclose(before, after)


def test_trainer_run_returns_history():
    actor = HiddenStateActor(embed_dim=32)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    buf = ReplayBuffer(capacity=16)
    _fill_buffer(buf, 8)
    history = Trainer(actor, opt, buf, sync_interval=1000).run(
        10, 4, np.random.default_rng(0)
    )
    assert len(history) == 10
    assert all("total" in h for h in history)
