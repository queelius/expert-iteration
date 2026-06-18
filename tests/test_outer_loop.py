"""End-to-end test that the outer loop CLOSES (not merely runs).

Uses FakeEncoder for the heads' embeddings and EchoLLMClient for the MDP's
generation, so the whole expert-iteration loop runs on CPU with no model.
"""

from __future__ import annotations

import numpy as np
import pytest


def _build(embed_dim=16):
    from tests.fakes import FakeEncoder
    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.llm_client import EchoLLMClient
    from verifier_anchored_rl.llm_context_mdp import LLMContextMDP
    from verifier_anchored_rl.planner_actor import PrefillCache, StateActor
    from verifier_anchored_rl.replay import ReplayBuffer
    from verifier_anchored_rl.train import Trainer
    import torch

    enc = FakeEncoder(embed_dim=embed_dim)
    actor = HiddenStateActor(embed_dim=embed_dim)
    sa = StateActor(enc, actor, PrefillCache(), depth_limit=3)
    buf = ReplayBuffer(capacity=64)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-2)
    trainer = Trainer(actor, opt, buf, sync_interval=3)

    def mdp_factory(problem):
        return LLMContextMDP(problem_statement=problem[0],
                             expected_answer=problem[1],
                             llm_client=EchoLLMClient(), depth_limit=3, max_turns=4)

    return enc, actor, sa, buf, trainer, mdp_factory


def test_outer_loop_runs_and_populates_stats():
    pytest.importorskip("torch")
    from verifier_anchored_rl.outer_loop import run_outer_loop

    enc, actor, sa, buf, trainer, mdp_factory = _build()
    problems = [("2+2", "4")]
    stats = run_outer_loop(
        problems, mdp_factory, sa, trainer, buf,
        n_iterations=3, train_steps_per_iter=20, batch_size=8,
        min_buffer_size=4, num_simulations=8, c_puct=1.5, discount=1.0,
        tau=1.0, max_turns=4, rng=np.random.default_rng(0), encoder=enc)

    assert len(stats) == 3
    assert len(buf) > 0
    assert stats[-1].buffer_size == len(buf)
    assert stats[-1].prefill_calls > 0           # encoder was exercised
    assert any(s.losses for s in stats)          # training happened
    assert stats[-1].n_decisions >= 1
    assert stats[-1].n_simulations == stats[-1].n_decisions * 8
    assert trainer.global_step >= 3              # target sync fired at N=3


def test_outer_loop_closes_prior_shifts_search():
    """After training, the policy prior is no longer uniform and it changes
    the MCTS visit distribution (search -> distill -> better prior ->
    different search)."""
    pytest.importorskip("torch")
    from verifier_anchored_rl.mcts import MCTSConfig, mcts_search, visit_count_policy
    from verifier_anchored_rl.outer_loop import run_outer_loop

    enc, actor, sa, buf, trainer, mdp_factory = _build()
    problems = [("2+2", "4")]
    run_outer_loop(
        problems, mdp_factory, sa, trainer, buf,
        n_iterations=4, train_steps_per_iter=40, batch_size=8,
        min_buffer_size=4, num_simulations=8, c_puct=1.5, discount=1.0,
        tau=1.0, max_turns=4, rng=np.random.default_rng(0), encoder=enc)

    mdp = mdp_factory(problems[0])
    s0 = mdp.initial_state()
    acts = list(mdp.actions(s0))
    prior = sa.prior(s0, acts)
    assert max(prior) - min(prior) > 1e-3        # diverged from uniform

    cfg_u = MCTSConfig(num_simulations=24, c_puct=2.0, discount=1.0)
    cfg_t = MCTSConfig(num_simulations=24, c_puct=2.0, discount=1.0,
                       prior_fn=sa.prior)
    ru = mcts_search(mdp, s0, cfg_u, np.random.default_rng(1))
    rt = mcts_search(mdp, s0, cfg_t, np.random.default_rng(1))
    vu = {a.verb.value: round(v, 6) for a, v in visit_count_policy(ru).items()}
    vt = {a.verb.value: round(v, 6) for a, v in visit_count_policy(rt).items()}
    assert vu != vt                              # the learned prior steers search
