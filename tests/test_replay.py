"""Tests for the prioritized replay buffer."""


def test_replay_buffer_state_dict_round_trips():
    import pytest
    pytest.importorskip("torch")
    import numpy as np
    import torch
    from verifier_anchored_rl.replay import ReplayBuffer, ReplaySample

    def _sample(seed):
        g = torch.Generator().manual_seed(seed)
        return ReplaySample(
            hidden=torch.randn(8, generator=g),
            available_mask=torch.ones(7, dtype=torch.bool),
            q_target=torch.randn(7, generator=g),
            q_target_mask=torch.ones(7, dtype=torch.bool),
            pi_target=torch.softmax(torch.randn(7, generator=g), dim=-1),
        )

    buf = ReplayBuffer(capacity=4)
    for i in range(3):
        buf.add(_sample(i), priority=float(i + 1))

    state = buf.state_dict()

    restored = ReplayBuffer(capacity=4)
    restored.load_state_dict(state)

    assert len(restored) == len(buf)
    assert restored._pos == buf._pos
    assert np.allclose(restored._priorities, buf._priorities)
    for a, b in zip(restored._samples, buf._samples):
        assert torch.equal(a.hidden, b.hidden)
        assert torch.equal(a.q_target, b.q_target)
        assert torch.equal(a.pi_target, b.pi_target)
