"""Tests for the trainable Q-head and policy head.

These tests verify the structural properties of the heads (correct
shapes, gradient flow, mask handling, loss invariants) on synthetic data.
End-to-end training against the JSONL dataset is in
``experiments/train_heads.py``.

The tests skip cleanly when torch is not installed (it is an optional
dependency under the ``[llm]`` extras).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np

from verifier_anchored_rl.heads import (
    NUM_VERBS,
    VERB_NAME_TO_INDEX,
    VERB_ORDER,
    TrainableActor,
    encode_batch,
    policy_cross_entropy,
    q_mse,
)
from verifier_anchored_rl.deliberation_mdp import Verb


# -----------------------------------------------------------------------------
# Architecture invariants
# -----------------------------------------------------------------------------


def test_verb_order_matches_enum_definition_order():
    expected = [
        Verb.SPLIT,
        Verb.FORM_HYPOTHESIS,
        Verb.COMPUTE,
        Verb.VERIFY,
        Verb.COMMIT,
        Verb.ABANDON,
    ]
    assert VERB_ORDER == expected
    assert NUM_VERBS == 6


def test_actor_forward_returns_correct_shapes():
    model = TrainableActor(embed_dim=128, hidden_dim=64, repr_dim=32)
    batch = 5
    x = torch.randn(batch, 128)
    logits, q = model(x)
    assert logits.shape == (batch, NUM_VERBS)
    assert q.shape == (batch, NUM_VERBS)


def test_actor_parameter_count_is_roughly_1m_at_default_dims():
    model = TrainableActor()  # defaults: 2048 -> 512 -> 256
    total = sum(p.numel() for p in model.parameters())
    # Embed-dim Linear dominates: 2048 * 512 + 512 + ...
    assert 1_000_000 < total < 1_300_000, f"unexpected param count: {total}"


def test_actor_pi_returns_distribution_over_available():
    model = TrainableActor(embed_dim=32, hidden_dim=32, repr_dim=16)
    x = torch.randn(2, 32)
    mask = torch.tensor(
        [[True, True, False, False, True, False],
         [False, True, True, False, False, True]],
        dtype=torch.bool,
    )
    pi = model.pi(x, mask)
    # Rows sum to 1 (over available positions).
    assert torch.allclose(pi.sum(dim=-1), torch.ones(2), atol=1e-5)
    # Unavailable positions are exactly 0.
    assert (pi[~mask] == 0).all()
    # Available positions are strictly positive.
    assert (pi[mask] > 0).all()


def test_actor_q_returns_unmasked_values():
    model = TrainableActor(embed_dim=32, hidden_dim=32, repr_dim=16)
    x = torch.randn(3, 32)
    q = model.q(x)
    assert q.shape == (3, NUM_VERBS)
    # No nan / inf.
    assert torch.isfinite(q).all()


# -----------------------------------------------------------------------------
# Loss functions
# -----------------------------------------------------------------------------


def test_policy_cross_entropy_is_zero_for_perfectly_matched_distributions():
    """If pi_theta exactly matches pi_target on a single available verb,
    cross-entropy is 0 (since target is a one-hot)."""
    model = TrainableActor(embed_dim=8, hidden_dim=8, repr_dim=4)
    # One sample, only verb 0 available, target is one-hot on verb 0.
    x = torch.randn(1, 8)
    pi_target = torch.zeros(1, NUM_VERBS)
    pi_target[0, 0] = 1.0
    mask = torch.zeros(1, NUM_VERBS, dtype=torch.bool)
    mask[0, 0] = True

    # With only one available verb, the masked softmax assigns prob 1.0
    # to that verb regardless of logits, so log(prob) = 0 and CE = 0.
    loss = policy_cross_entropy(model, x, pi_target, mask)
    assert abs(float(loss.item())) < 1e-5


def test_policy_cross_entropy_is_finite_with_two_available_verbs():
    model = TrainableActor(embed_dim=16, hidden_dim=16, repr_dim=8)
    x = torch.randn(4, 16)
    mask = torch.zeros(4, NUM_VERBS, dtype=torch.bool)
    mask[:, 0] = True
    mask[:, 1] = True
    pi_target = torch.zeros(4, NUM_VERBS)
    pi_target[:, 0] = 0.7
    pi_target[:, 1] = 0.3
    loss = policy_cross_entropy(model, x, pi_target, mask)
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0


def test_q_mse_only_counts_target_positions():
    model = TrainableActor(embed_dim=16, hidden_dim=16, repr_dim=8)
    x = torch.randn(3, 16)
    q_target = torch.zeros(3, NUM_VERBS)
    q_target[:, 0] = 1.0
    mask = torch.zeros(3, NUM_VERBS, dtype=torch.bool)
    mask[:, 0] = True
    loss = q_mse(model, x, q_target, mask)
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0
    # Loss has gradients with respect to model parameters.
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters())


def test_policy_loss_drives_logits_toward_target_with_optimization():
    """Sanity check: a few Adam steps on a single sample should reduce policy loss."""
    torch.manual_seed(0)
    model = TrainableActor(embed_dim=16, hidden_dim=16, repr_dim=8, dropout=0.0)
    x = torch.randn(1, 16)
    mask = torch.zeros(1, NUM_VERBS, dtype=torch.bool)
    mask[:, 0] = True
    mask[:, 1] = True
    mask[:, 2] = True
    pi_target = torch.zeros(1, NUM_VERBS)
    pi_target[0, 0] = 0.7
    pi_target[0, 1] = 0.2
    pi_target[0, 2] = 0.1

    initial = float(policy_cross_entropy(model, x, pi_target, mask).item())
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(200):
        opt.zero_grad()
        loss = policy_cross_entropy(model, x, pi_target, mask)
        loss.backward()
        opt.step()
    final = float(policy_cross_entropy(model, x, pi_target, mask).item())
    assert final < initial - 0.1, f"loss did not drop: {initial} -> {final}"


def test_q_loss_drives_q_values_toward_target_with_optimization():
    torch.manual_seed(0)
    model = TrainableActor(embed_dim=16, hidden_dim=16, repr_dim=8, dropout=0.0)
    x = torch.randn(1, 16)
    mask = torch.zeros(1, NUM_VERBS, dtype=torch.bool)
    mask[:, 0] = True
    q_target = torch.zeros(1, NUM_VERBS)
    q_target[0, 0] = 0.75

    initial = float(q_mse(model, x, q_target, mask).item())
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(200):
        opt.zero_grad()
        loss = q_mse(model, x, q_target, mask)
        loss.backward()
        opt.step()
    final = float(q_mse(model, x, q_target, mask).item())
    assert final < 1e-3, f"Q loss did not drop to ~0: {final}"

    # And Q(s, 0) should now be close to 0.75.
    q_pred = model.q(x)
    assert abs(float(q_pred[0, 0]) - 0.75) < 5e-2


# -----------------------------------------------------------------------------
# Batch encoding
# -----------------------------------------------------------------------------


def test_encode_batch_handles_minimal_record():
    rec = {
        "embedding": [0.0] * 32,
        "available_verbs": ["form_hypothesis", "split"],
        "pi_mcts": {"form_hypothesis": 0.6, "split": 0.4},
        "q_mcts": {"form_hypothesis": 0.1, "split": 0.05},
    }
    batch = encode_batch([rec])

    assert batch["embedding"].shape == (1, 32)
    assert batch["pi_target"].shape == (1, NUM_VERBS)
    assert batch["q_target"].shape == (1, NUM_VERBS)
    assert batch["available_mask"].shape == (1, NUM_VERBS)
    assert batch["q_target_mask"].shape == (1, NUM_VERBS)

    fh = VERB_NAME_TO_INDEX["form_hypothesis"]
    dc = VERB_NAME_TO_INDEX["split"]
    # pi_target rows sum to 1 over available positions.
    assert abs(float(batch["pi_target"][0].sum()) - 1.0) < 1e-5
    assert batch["pi_target"][0, fh].item() == pytest.approx(0.6)
    assert batch["pi_target"][0, dc].item() == pytest.approx(0.4)
    # available_mask is True at FH and DC, False elsewhere.
    assert batch["available_mask"][0, fh] and batch["available_mask"][0, dc]
    assert batch["available_mask"][0].sum().item() == 2
    # q_target_mask matches q_mcts keys.
    assert batch["q_target_mask"][0, fh] and batch["q_target_mask"][0, dc]
    assert batch["q_target_mask"][0].sum().item() == 2
    # q values match.
    assert batch["q_target"][0, fh].item() == pytest.approx(0.1)
    assert batch["q_target"][0, dc].item() == pytest.approx(0.05)


def test_encode_batch_handles_pi_mcts_missing_some_available_verbs():
    """If MCTS visited only a subset of available verbs, the missing ones
    get pi_target=0 (and re-normalization handles the rest)."""
    rec = {
        "embedding": [0.0] * 16,
        "available_verbs": ["form_hypothesis", "split", "abandon"],
        "pi_mcts": {"form_hypothesis": 0.7, "split": 0.3},  # abandon missing
        "q_mcts": {"form_hypothesis": 0.2, "split": 0.1},
    }
    batch = encode_batch([rec])
    bt = VERB_NAME_TO_INDEX["abandon"]
    # abandon in mask but with 0 prob target.
    assert batch["available_mask"][0, bt]
    assert batch["pi_target"][0, bt].item() == 0.0
    # Row still sums to 1.
    assert abs(float(batch["pi_target"][0].sum()) - 1.0) < 1e-5
    # BACKTRACK NOT in q_target_mask (no q_mcts entry).
    assert not batch["q_target_mask"][0, bt]


def test_encode_batch_handles_multiple_records():
    recs = [
        {
            "embedding": [float(i)] * 8,
            "available_verbs": ["form_hypothesis"],
            "pi_mcts": {"form_hypothesis": 1.0},
            "q_mcts": {"form_hypothesis": float(i) / 10},
        }
        for i in range(4)
    ]
    batch = encode_batch(recs)
    assert batch["embedding"].shape == (4, 8)
    # Verify per-record embedding values.
    for i in range(4):
        assert batch["embedding"][i, 0].item() == pytest.approx(float(i))


def test_encode_batch_rejects_empty_input():
    with pytest.raises(ValueError):
        encode_batch([])


def test_encode_batch_rejects_null_embedding_records():
    """collect_distillation writes embedding: null on fetch failure;
    np.asarray(None) would silently broadcast NaN into the whole row and
    poison training, so encode_batch must fail loud instead."""
    records = [
        {
            "problem_key": "p", "seed": 0, "turn": 0,
            "embedding": None,
            "available_verbs": ["split"],
            "pi_mcts": {"split": 1.0},
            "q_mcts": {"split": 0.5},
        }
    ]
    with pytest.raises(ValueError, match="no embedding"):
        encode_batch(records)


def test_pi_raises_on_all_masked_row():
    """softmax over an all--inf row is NaN; pi must fail loud rather than
    hand NaN to downstream argmax (which silently returns index 0)."""
    model = TrainableActor(embed_dim=8, hidden_dim=16, repr_dim=8)
    emb = torch.zeros((1, 8))
    mask = torch.zeros((1, NUM_VERBS), dtype=torch.bool)
    with pytest.raises(ValueError, match="no available verbs"):
        model.pi(emb, mask)
