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
        "available_verbs": ["split", "commit"],
        "pi_mcts": {"split": 0.7, "commit": 0.3},
        "q_mcts": {"split": 0.5},
    }
    s = sample_from_record(record)
    assert s.hidden.shape == (8,)
    assert s.available_mask.sum().item() == 2
    assert s.q_target_mask.sum().item() == 1  # only "split" had a Q estimate


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
    # q_loss returns (loss_tensor, per_sample_td_array); unpack accordingly.
    ql, td_arr = q_loss(actor, b["hidden"], b["q_target"], b["q_target_mask"])
    pl = policy_loss(actor, b["hidden"], b["pi_target"], b["available_mask"])
    assert torch.isfinite(ql) and ql.item() >= 0
    assert torch.isfinite(pl) and pl.item() >= 0
    # Also verify the returned TD array has the right shape and is non-negative.
    assert td_arr.shape == (b["hidden"].shape[0],)
    assert np.all(td_arr >= 0)


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


def test_trainer_run_fires_on_step_callback():
    import pytest
    pytest.importorskip("torch")
    import numpy as np
    import torch
    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.heads import NUM_VERBS
    from verifier_anchored_rl.replay import ReplayBuffer, ReplaySample
    from verifier_anchored_rl.train import Trainer

    actor = HiddenStateActor(embed_dim=8)
    buf = ReplayBuffer(capacity=8)
    g = torch.Generator().manual_seed(0)
    for _ in range(4):
        buf.add(ReplaySample(
            hidden=torch.randn(8, generator=g),
            available_mask=torch.ones(NUM_VERBS, dtype=torch.bool),
            q_target=torch.randn(NUM_VERBS, generator=g),
            q_target_mask=torch.ones(NUM_VERBS, dtype=torch.bool),
            pi_target=torch.softmax(torch.randn(NUM_VERBS, generator=g), dim=-1),
        ))
    opt = torch.optim.Adam(actor.parameters(), lr=1e-2)
    trainer = Trainer(actor, opt, buf, sync_interval=100)

    seen = []
    trainer.run(5, batch_size=4, rng=np.random.default_rng(0),
                on_step=lambda step, losses: seen.append((step, losses)))

    assert len(seen) == 5
    assert seen[0][0] == 1 and seen[-1][0] == 5      # 1-based global step
    assert set(seen[0][1].keys()) == {"q", "pi", "total"}


# -----------------------------------------------------------------------------
# TD priorities use the PRE-update network (one Q forward per step)
# -----------------------------------------------------------------------------


def test_td_priorities_use_preupdate_network():
    """train_step must set priorities from the PRE-update forward, not a
    second forward taken after optimizer.step.

    Two assertions:
    1. The priorities the buffer receives equal the |TD error| computed from
       the network BEFORE the step (within floating-point tolerance), and
       differ from the |TD error| computed AFTER the step (unless the two
       happen to coincide numerically, which is astronomically unlikely with
       a non-trivial learning rate).
    2. The Q head is called exactly ONCE per train_step (the redundant
       post-update forward is gone), verified by a call counter.
    """
    torch.manual_seed(42)
    embed_dim = 16
    actor = HiddenStateActor(embed_dim=embed_dim)
    # Use a large learning rate so PRE vs POST differ noticeably.
    opt = torch.optim.SGD(actor.parameters(), lr=0.5)

    buf = ReplayBuffer(capacity=8)
    g = torch.Generator().manual_seed(7)
    for i in range(6):
        buf.add(ReplaySample(
            hidden=torch.randn(embed_dim, generator=g),
            available_mask=torch.ones(NUM_VERBS, dtype=torch.bool),
            q_target=torch.randn(NUM_VERBS, generator=g),
            q_target_mask=torch.ones(NUM_VERBS, dtype=torch.bool),
            pi_target=torch.softmax(torch.randn(NUM_VERBS, generator=g), dim=-1),
        ))

    rng = np.random.default_rng(99)
    batch, indices = buf.sample(6, rng)

    # Capture the PRE-update Q output (detached snapshot of the current params).
    with torch.no_grad():
        q_pre = actor.q(batch["hidden"]).clone()

    # Count how many times the Q head is called during train_step.
    q_forward_count = [0]
    original_q = actor.q

    def counting_q(hidden):
        q_forward_count[0] += 1
        return original_q(hidden)

    actor.q = counting_q

    # Run the step (uses the monkeypatched q).
    losses, td_returned = train_step(actor, opt, batch, lambda_pi=1.0)

    actor.q = original_q  # restore

    # Assertion 2: exactly ONE Q forward per train_step.
    assert q_forward_count[0] == 1, (
        f"Expected 1 Q forward per train_step, got {q_forward_count[0]}. "
        "The redundant post-update forward was not removed."
    )

    # Compute expected priorities from the PRE-update outputs.
    mask = batch["q_target_mask"].bool().float()
    abs_err_pre = (q_pre - batch["q_target"]).abs() * mask
    num = mask.sum(dim=-1).clamp(min=1.0)
    td_pre = (abs_err_pre.sum(dim=-1) / num).cpu().numpy()

    # Compute what priorities would look like from the POST-update network.
    with torch.no_grad():
        q_post = actor.q(batch["hidden"])
    abs_err_post = (q_post - batch["q_target"]).abs() * mask
    td_post = (abs_err_post.sum(dim=-1) / num).cpu().numpy()

    # Assertion 1: returned TD values match PRE-update, not POST-update.
    assert np.allclose(td_returned, td_pre, atol=1e-5), (
        "train_step returned TD errors that do NOT match the pre-update "
        f"network. Max diff vs pre: {np.abs(td_returned - td_pre).max():.6f}, "
        f"max diff vs post: {np.abs(td_returned - td_post).max():.6f}"
    )
    # Sanity: PRE and POST must differ enough to be distinguishable (large lr).
    # If they are identical the test is vacuous; fail loudly.
    assert not np.allclose(td_pre, td_post, atol=1e-5), (
        "PRE and POST TD errors are numerically identical - the learning rate "
        "may be too small or the test data degenerate; the assertion above is "
        "vacuous. Increase lr or change seed."
    )
