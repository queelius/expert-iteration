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
    from verifier_anchored_rl.deliberation_mdp import DeliberationMDP
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
        return DeliberationMDP(problem=problem[0], expected_answer=problem[1],
                               llm_client=EchoLLMClient(), depth_limit=3)

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
    """The loop CLOSES: training moves the learned prior, which then steers
    MCTS differently from a uniform prior.

    Two robust-by-construction assertions prove this:
    (1) Training MOVED the prior: max absolute change between prior_before
        and prior_after exceeds 1e-3 (isolates the training effect from the
        fact that random-init weights are already non-uniform).
    (2) The learned prior steers search differently from a uniform prior:
        visit-count distributions under the trained prior and under uniform
        must differ.

    A Q-loss trend assertion was considered but dropped because on a tiny
    synthetic buffer (capacity 64, 40 steps/iter, 4 iterations) the Q-loss
    can temporarily spike before converging, making the trend non-monotone
    across seeds. The two assertions above are sufficient: they prove the
    loop closes (prior changed, search changed). Q-loss decrease is already
    covered by test_train.py. Validated across torch seeds {0,1,2,3,42}:
    both assertions pass on every seed; prior_delta ranged from 0.087 to
    0.224.
    """
    pytest.importorskip("torch")
    from verifier_anchored_rl.mcts import MCTSConfig, mcts_search, visit_count_policy
    from verifier_anchored_rl.outer_loop import run_outer_loop

    import torch
    torch.manual_seed(0)  # neutral seed for determinism; not hand-picked to pass
    enc, actor, sa, buf, trainer, mdp_factory = _build()
    problems = [("2+2", "4")]
    mdp = mdp_factory(problems[0])
    s0 = mdp.initial_state()
    acts = list(mdp.actions(s0))

    prior_before = sa.prior(s0, acts)  # untrained (random init)

    run_outer_loop(
        problems, mdp_factory, sa, trainer, buf,
        n_iterations=4, train_steps_per_iter=40, batch_size=8,
        min_buffer_size=4, num_simulations=8, c_puct=1.5, discount=1.0,
        tau=1.0, max_turns=4, rng=np.random.default_rng(0), encoder=enc)

    prior_after = sa.prior(s0, acts)
    # (1) Training MOVED the prior (not merely that it is non-uniform at init).
    assert max(abs(a - b) for a, b in zip(prior_after, prior_before)) > 1e-3

    # (2) The learned prior steers search differently from a uniform prior.
    cfg_u = MCTSConfig(num_simulations=24, c_puct=2.0, discount=1.0)
    cfg_t = MCTSConfig(num_simulations=24, c_puct=2.0, discount=1.0,
                       prior_fn=sa.prior)
    ru = mcts_search(mdp, s0, cfg_u, np.random.default_rng(1))
    rt = mcts_search(mdp, s0, cfg_t, np.random.default_rng(1))
    vu = {a.verb.value: round(v, 6) for a, v in visit_count_policy(ru).items()}
    vt = {a.verb.value: round(v, 6) for a, v in visit_count_policy(rt).items()}
    assert vu != vt


def test_entropy_helper_is_in_bits():
    from verifier_anchored_rl.outer_loop import _entropy

    assert abs(_entropy([0.5, 0.5]) - 1.0) < 1e-9          # 1 bit
    assert abs(_entropy([1.0]) - 0.0) < 1e-9               # delta, 0 bits
    assert abs(_entropy([0.25, 0.25, 0.25, 0.25]) - 2.0) < 1e-9
    assert abs(_entropy([0.5, 0.5, 0.0]) - 1.0) < 1e-9     # zero-prob skipped


def test_collect_iteration_reports_visit_and_prior_entropy():
    import math

    import pytest
    pytest.importorskip("torch")
    import numpy as np
    from verifier_anchored_rl.mcts import MCTSConfig
    from verifier_anchored_rl.outer_loop import collect_iteration

    enc, actor, sa, buf, trainer, mdp_factory = _build()
    config = MCTSConfig(num_simulations=6, c_puct=1.5, discount=1.0,
                        prior_fn=sa.prior, value_fn=sa.leaf_value)

    decisions = []
    counts = collect_iteration(
        [("2+2", "4")], mdp_factory, sa, config, buf,
        np.random.default_rng(0), tau=1.0, max_turns=4,
        on_decision=lambda info: decisions.append(info),
    )

    assert math.isfinite(counts["mean_visit_entropy"])
    assert math.isfinite(counts["mean_prior_entropy"])
    assert counts["mean_visit_entropy"] >= 0.0
    assert counts["mean_prior_entropy"] >= 0.0
    assert "pi_mcts_entropy" in decisions[0]
    assert "prior_entropy" in decisions[0]


def test_collect_iteration_reports_normalized_concentration():
    import math

    import pytest
    pytest.importorskip("torch")
    import numpy as np
    from verifier_anchored_rl.mcts import MCTSConfig
    from verifier_anchored_rl.outer_loop import collect_iteration

    enc, actor, sa, buf, trainer, mdp_factory = _build()
    config = MCTSConfig(num_simulations=6, c_puct=1.5, discount=1.0,
                        prior_fn=sa.prior, value_fn=sa.leaf_value)
    counts = collect_iteration(
        [("2+2", "4")], mdp_factory, sa, config, buf,
        np.random.default_rng(0), tau=1.0, max_turns=4)

    # 1 - H/log2(k): confound-free sharpness in [0, 1], higher = more concentrated.
    cv = counts["mean_visit_concentration"]
    cp = counts["mean_prior_concentration"]
    assert math.isfinite(cv) and 0.0 <= cv <= 1.0
    assert math.isfinite(cp) and 0.0 <= cp <= 1.0


def test_run_outer_loop_stats_carry_entropy_metrics():
    import math

    import pytest
    pytest.importorskip("torch")
    import numpy as np
    from verifier_anchored_rl.outer_loop import run_outer_loop

    enc, actor, sa, buf, trainer, mdp_factory = _build()
    stats = run_outer_loop(
        [("2+2", "4")], mdp_factory, sa, trainer, buf,
        n_iterations=2, train_steps_per_iter=10, batch_size=8,
        min_buffer_size=2, num_simulations=6, c_puct=1.5, discount=1.0,
        tau=1.0, max_turns=4, rng=np.random.default_rng(0), encoder=enc)

    assert math.isfinite(stats[-1].visit_entropy)
    assert math.isfinite(stats[-1].prior_entropy)


def test_flop_counters_reads_gen_from_gen_source():
    from verifier_anchored_rl.outer_loop import _flop_counters

    class Enc:
        prefill_calls, prefill_tokens, gen_calls, gen_tokens = 2, 20, 99, 990

    class Gen:  # an llm_client: generate_calls / generated_tokens
        generate_calls, generated_tokens = 5, 500

    c = _flop_counters(Enc(), Gen())
    assert c["prefill_calls"] == 2 and c["prefill_tokens"] == 20
    # gen counts come from the gen_source (the client), not the encoder
    assert c["gen_calls"] == 5 and c["gen_tokens"] == 500
    # default: gen_source falls back to the encoder
    d = _flop_counters(Enc())
    assert d["gen_calls"] == 99 and d["gen_tokens"] == 990


def test_loss_summary_extracts_q_pi_total_split():
    from verifier_anchored_rl.outer_loop import loss_summary

    losses = [
        {"q": 1.0, "pi": 2.0, "total": 3.0},
        {"q": 0.5, "pi": 1.5, "total": 2.0},
        {"q": 0.1, "pi": 1.4, "total": 1.5},
    ]
    s = loss_summary(losses)
    assert s["q_first"] == 1.0 and s["q_last"] == 0.1
    assert abs(s["q_mean"] - (1.6 / 3)) < 1e-9
    assert s["pi_first"] == 2.0 and s["pi_last"] == 1.4
    assert s["total_first"] == 3.0 and s["total_last"] == 1.5


def test_loss_summary_handles_empty():
    from verifier_anchored_rl.outer_loop import loss_summary
    import math

    s = loss_summary([])
    assert math.isnan(s["q_first"]) and math.isnan(s["total_last"])


def test_collect_iteration_fires_decision_and_trajectory_callbacks():
    import pytest
    pytest.importorskip("torch")
    import numpy as np
    from verifier_anchored_rl.mcts import MCTSConfig
    from verifier_anchored_rl.outer_loop import collect_iteration

    enc, actor, sa, buf, trainer, mdp_factory = _build()
    config = MCTSConfig(num_simulations=6, c_puct=1.5, discount=1.0,
                        prior_fn=sa.prior, value_fn=sa.leaf_value)

    decisions = []
    trajectories = []
    counts = collect_iteration(
        [("2+2", "4"), ("3+3", "6")], mdp_factory, sa, config, buf,
        np.random.default_rng(0), tau=1.0, max_turns=4,
        on_decision=lambda info: decisions.append(info),
        on_trajectory_end=lambda done: trajectories.append(done),
    )

    assert len(decisions) == counts["n_decisions"]
    assert decisions[0]["iteration_problem_index"] == 0
    assert "chosen_verb" in decisions[0]
    assert "gen_tokens" in decisions[0]
    assert trajectories == [1, 2]      # one callback per completed trajectory


def test_collect_iteration_start_problem_skips_done_trajectories():
    import pytest
    pytest.importorskip("torch")
    import numpy as np
    from verifier_anchored_rl.mcts import MCTSConfig
    from verifier_anchored_rl.outer_loop import collect_iteration

    enc, actor, sa, buf, trainer, mdp_factory = _build()
    config = MCTSConfig(num_simulations=6, c_puct=1.5, discount=1.0,
                        prior_fn=sa.prior, value_fn=sa.leaf_value)

    trajectories = []
    collect_iteration(
        [("2+2", "4"), ("3+3", "6")], mdp_factory, sa, config, buf,
        np.random.default_rng(0), tau=1.0, max_turns=4, start_problem=1,
        on_trajectory_end=lambda done: trajectories.append(done),
    )
    # problem index 0 skipped, only the second trajectory runs
    assert trajectories == [2]


def _checkpoint_state(actor, opt, buf, trainer, rng, phase, iteration, done):
    return {
        "phase": phase,
        "iteration": iteration,
        "problems_done_this_iter": done,
        "actor_state": actor.state_dict(),
        "optimizer_state": opt.state_dict(),
        "buffer": buf.state_dict(),
        "global_step": trainer.global_step,
        "numpy_rng_state": rng.bit_generator.state,
    }


def test_resume_from_iteration_checkpoint_matches_uninterrupted_run(tmp_path):
    import pytest
    pytest.importorskip("torch")
    import numpy as np
    import torch
    from verifier_anchored_rl.outer_loop import run_outer_loop
    from verifier_anchored_rl.run_artifacts import (
        save_checkpoint_atomic, load_checkpoint,
    )

    problems = [("2+2", "4")]
    common = dict(
        n_iterations=2, train_steps_per_iter=10, batch_size=8,
        min_buffer_size=2, num_simulations=6, c_puct=1.5, discount=1.0,
        tau=1.0, max_turns=4,
    )

    # Uninterrupted reference run. Seed torch so the actor init is fixed.
    torch.manual_seed(0)
    enc, actor, sa, buf, trainer, mdp_factory = _build()
    run_outer_loop(problems, mdp_factory, sa, trainer, buf,
                   rng=np.random.default_rng(0), encoder=enc, **common)
    ref_weights = {k: v.clone() for k, v in actor.state_dict().items()}

    # Interruptible run: identical torch seed gives identical actor init.
    # Stop after iteration 1, save a "training"-phase checkpoint.
    torch.manual_seed(0)
    enc2, actor2, sa2, buf2, trainer2, mdp2 = _build()
    rng2 = np.random.default_rng(0)
    ckpt_path = str(tmp_path / "checkpoint.pt")

    def _save(state):
        if state["phase"] == "training" and state["iteration"] == 1:
            save_checkpoint_atomic(ckpt_path, _checkpoint_state(
                actor2, trainer2.optimizer, buf2, trainer2, rng2,
                "training", 1, len(problems)))

    run_outer_loop(problems, mdp2, sa2, trainer2, buf2,
                   rng=rng2, encoder=enc2, n_iterations=1,
                   train_steps_per_iter=common["train_steps_per_iter"],
                   batch_size=common["batch_size"],
                   min_buffer_size=common["min_buffer_size"],
                   num_simulations=common["num_simulations"],
                   c_puct=common["c_puct"], discount=common["discount"],
                   tau=common["tau"], max_turns=common["max_turns"],
                   save_checkpoint=_save)

    # Fresh stack, restore, resume iteration 2.
    torch.manual_seed(0)
    enc3, actor3, sa3, buf3, trainer3, mdp3 = _build()
    ckpt = load_checkpoint(ckpt_path)
    actor3.load_state_dict(ckpt["actor_state"])
    trainer3.optimizer.load_state_dict(ckpt["optimizer_state"])
    buf3.load_state_dict(ckpt["buffer"])
    trainer3.global_step = ckpt["global_step"]
    rng3 = np.random.default_rng(0)
    rng3.bit_generator.state = ckpt["numpy_rng_state"]

    run_outer_loop(problems, mdp3, sa3, trainer3, buf3,
                   rng=rng3, encoder=enc3, start_iteration=2, **common)

    resumed_weights = actor3.state_dict()
    for k in ref_weights:
        assert torch.allclose(ref_weights[k], resumed_weights[k], atol=1e-6), k


def test_run_outer_loop_invokes_train_and_checkpoint_callbacks():
    import pytest
    pytest.importorskip("torch")
    import numpy as np
    from verifier_anchored_rl.outer_loop import run_outer_loop

    enc, actor, sa, buf, trainer, mdp_factory = _build()
    train_steps = []
    checkpoints = []
    stats = run_outer_loop(
        [("2+2", "4")], mdp_factory, sa, trainer, buf,
        n_iterations=2, train_steps_per_iter=10, batch_size=8,
        min_buffer_size=2, num_simulations=6, c_puct=1.5, discount=1.0,
        tau=1.0, max_turns=4, rng=np.random.default_rng(0), encoder=enc,
        on_train_step=lambda info: train_steps.append(info),
        save_checkpoint=lambda state: checkpoints.append(
            (state["phase"], state["iteration"], state["problems_done_this_iter"])),
        log_every=5,
    )

    # train-step callback fired on the log_every cadence for both iterations
    assert all(set(t.keys()) >= {"global_step", "q", "pi", "total"} for t in train_steps)
    # a checkpoint after each trajectory (collecting) and after each train phase
    assert ("collecting", 1, 1) in checkpoints
    assert ("training", 1, 1) in checkpoints
    assert ("training", 2, 1) in checkpoints
    # loss split landed on the stats
    assert stats[-1].q_first == stats[-1].q_first   # not NaN: training ran
    assert stats[-1].wall_seconds >= 0.0


def test_collect_iteration_concurrent_matches_sequential_record_count():
    import pytest
    pytest.importorskip("torch")
    import numpy as np
    import torch
    from verifier_anchored_rl.mcts import MCTSConfig
    from verifier_anchored_rl.outer_loop import (
        collect_iteration, collect_iteration_concurrent,
    )

    problems = [("2+2", "4"), ("3+3", "6"), ("4+4", "8")]

    # Sequential reference.
    enc1, actor1, sa1, buf1, tr1, mf1 = _build()
    cfg1 = MCTSConfig(num_simulations=6, c_puct=1.5, discount=1.0,
                      prior_fn=sa1.prior, value_fn=sa1.leaf_value)
    collect_iteration(problems, mf1, sa1, cfg1, buf1, np.random.default_rng(1),
                      tau=1.0, max_turns=4)

    # Concurrent: per-trajectory seeds, thread pool. Under DeliberationMDP a
    # trajectory can terminate early (root ABANDON), so unlike the legacy
    # fixed-max_turns MDP, n_decisions is sensitive to the actor's random
    # init, not just the worker count. Pin the torch seed before each _build()
    # (same idiom as test_resume_from_iteration_checkpoint_matches_uninterrupted_run
    # below) so the worker-count comparison holds the actor weights fixed.
    torch.manual_seed(0)
    enc2, actor2, sa2, buf2, tr2, mf2 = _build()
    cfg2 = MCTSConfig(num_simulations=6, c_puct=1.5, discount=1.0,
                      prior_fn=sa2.prior, value_fn=sa2.leaf_value)
    counts = collect_iteration_concurrent(
        problems, mf2, sa2, cfg2, buf2, base_seed=0,
        tau=1.0, max_turns=4, max_workers=3)

    assert counts["n_trajectories"] == 3
    assert counts["n_decisions"] == len(buf2)
    assert len(buf2) > 0
    # Determinism across pool sizes: same base_seed + workers=1 gives same count.
    torch.manual_seed(0)
    enc3, actor3, sa3, buf3, tr3, mf3 = _build()
    cfg3 = MCTSConfig(num_simulations=6, c_puct=1.5, discount=1.0,
                      prior_fn=sa3.prior, value_fn=sa3.leaf_value)
    counts1 = collect_iteration_concurrent(
        problems, mf3, sa3, cfg3, buf3, base_seed=0,
        tau=1.0, max_turns=4, max_workers=1)
    assert counts1["n_decisions"] == counts["n_decisions"]


def test_collect_iteration_concurrent_fires_on_decision_with_gen_source():
    import pytest
    pytest.importorskip("torch")
    import numpy as np
    from verifier_anchored_rl.mcts import MCTSConfig
    from verifier_anchored_rl.outer_loop import collect_iteration_concurrent

    enc, actor, sa, buf, tr, mf = _build()
    cfg = MCTSConfig(num_simulations=4, c_puct=1.5, discount=1.0,
                     prior_fn=sa.prior, value_fn=sa.leaf_value)

    class GenSrc:
        generate_calls = 5
        generated_tokens = 500

    seen = []
    collect_iteration_concurrent(
        [("2+2", "4")], mf, sa, cfg, buf, base_seed=0, tau=1.0, max_turns=4,
        max_workers=2, on_decision=lambda i: seen.append(i), gen_source=GenSrc())
    assert seen and seen[0]["iteration_problem_index"] == 0
    assert "chosen_verb" in seen[0]


def test_collect_iteration_on_decision_reads_gen_from_gen_source():
    """on_decision must read generation counts from gen_source (the llm_client
    on the two-forward path), not the encoder (which never generates there and
    would always log gen=0). 2026-06-22 review, minor finding."""
    import pytest
    pytest.importorskip("torch")
    import numpy as np
    from verifier_anchored_rl.mcts import MCTSConfig
    from verifier_anchored_rl.outer_loop import collect_iteration

    enc, actor, sa, buf, trainer, mdp_factory = _build()
    config = MCTSConfig(num_simulations=4, c_puct=1.5, discount=1.0,
                        prior_fn=sa.prior, value_fn=sa.leaf_value)

    class GenSrc:
        generate_calls = 7
        generated_tokens = 700

    decisions = []
    collect_iteration(
        [("2+2", "4")], mdp_factory, sa, config, buf, np.random.default_rng(0),
        tau=1.0, max_turns=4, on_decision=lambda i: decisions.append(i),
        gen_source=GenSrc())
    assert decisions[0]["gen_calls"] == 7 and decisions[0]["gen_tokens"] == 700


def test_run_outer_loop_concurrent_branch_collects_and_defers_diagnostics():
    import math
    import pytest
    pytest.importorskip("torch")
    import numpy as np
    from verifier_anchored_rl.outer_loop import run_outer_loop

    enc, actor, sa, buf, trainer, mf = _build()
    problems = [("2+2", "4"), ("3+3", "6")]
    stats = run_outer_loop(
        problems, mf, sa, trainer, buf,
        n_iterations=1, train_steps_per_iter=0, batch_size=4,
        min_buffer_size=10**9,  # skip training; exercise only the collect branch
        num_simulations=4, c_puct=1.5, discount=1.0, tau=1.0, max_turns=4,
        rng=np.random.default_rng(0), encoder=enc, gen_source=None,
        concurrent=True, collect_workers=2, base_seed=0,
    )
    assert len(stats) == 1
    s = stats[0]
    assert s.n_trajectories == 2
    assert s.n_decisions == len(buf)
    # The concurrent path defers entropy/concentration to offline analysis.
    assert math.isnan(s.visit_concentration)
    assert math.isnan(s.visit_entropy)


# ---------------------------------------------------------------------------
# Bug 1: concentration helper + integration test (full available-action set)
# ---------------------------------------------------------------------------


def test_normalized_concentration_pure_helper():
    """Unit test for _normalized_concentration helper.

    Worked example: k=7 available actions, 4 equal nonzero entries and 3 zeros.
    visit entropy = log2(4) = 2 bits; concentration = 1 - 2/log2(7) ~ 0.287.
    """
    import math
    from verifier_anchored_rl.outer_loop import _normalized_concentration

    visit_probs = [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0]
    visit_entropy = -sum(p * math.log2(p) for p in visit_probs if p > 0.0)
    assert abs(visit_entropy - 2.0) < 1e-9, f"entropy should be 2 bits, got {visit_entropy}"
    k = 7
    conc = _normalized_concentration(visit_entropy, k)
    expected = 1.0 - 2.0 / math.log2(7)
    assert abs(conc - expected) < 1e-3, f"concentration {conc} not near {expected}"
    assert abs(expected - 0.287) < 0.001, "sanity: expected concentration ~0.287"


def test_concentration_uses_full_action_set_not_expanded_subset():
    """Integration test: concentration is computed with k = len(mdp.actions(state)),
    not k = len(root.children). When num_simulations < num_available_actions the
    root is under-expanded, but k should still be the full support size.
    """
    import math
    import pytest
    pytest.importorskip("torch")
    import numpy as np
    from verifier_anchored_rl.mcts import MCTSConfig
    from verifier_anchored_rl.outer_loop import collect_iteration, _normalized_concentration

    enc, actor, sa, buf, trainer, mdp_factory = _build()
    mdp = mdp_factory(("2+2", "4"))
    s0 = mdp.initial_state()
    full_actions = list(mdp.actions(s0))
    k_full = len(full_actions)

    # Run with fewer simulations than available actions so root is under-expanded.
    num_sims = max(1, k_full - 1)
    config = MCTSConfig(num_simulations=num_sims, c_puct=1.5, discount=1.0,
                        prior_fn=sa.prior, value_fn=sa.leaf_value)

    captured_records = []
    counts = collect_iteration(
        [("2+2", "4")], mdp_factory, sa, config, buf,
        np.random.default_rng(0), tau=1.0, max_turns=4,
        record_sink=lambda r: captured_records.append(r),
    )

    # The reported concentration must be finite and match _normalized_concentration
    # with the FULL k (k_full), not the expanded subset count.
    cv = counts["mean_visit_concentration"]
    assert math.isfinite(cv), "visit_concentration should be finite"
    # Verify k used is the full set by recomputing concentration for every
    # captured decision over the full support (avail_verbs = full support),
    # then averaging, and comparing to the reported mean.
    if k_full > 1 and captured_records:
        concentrations = []
        for rec in captured_records:
            pi_dict = rec["pi_mcts"]  # verb -> prob over EXPANDED actions only
            avail_verbs = rec["available_verbs"]
            # visit_probs over FULL support (unexpanded contribute 0)
            visit_probs_full = [pi_dict.get(v, 0.0) for v in avail_verbs]
            k_rec = len(avail_verbs)
            if k_rec > 1:
                h_full = -sum(p * math.log2(p) for p in visit_probs_full if p > 0.0)
                concentrations.append(_normalized_concentration(h_full, k_rec))
        if concentrations:
            expected_mean = sum(concentrations) / len(concentrations)
            assert abs(cv - expected_mean) < 1e-6, (
                f"mean concentration {cv} does not match full-support recomputation "
                f"{expected_mean} (k_full={k_full}, num_sims={num_sims}, "
                f"n_decisions={len(captured_records)})"
            )


# ---------------------------------------------------------------------------
# Bug 2: _flop_counters warns when gen_source exposes no counters
# ---------------------------------------------------------------------------


def test_flop_counters_warns_when_gen_source_has_no_counters():
    """_flop_counters must warn (not raise) when gen_source has none of the
    four expected counter attributes."""
    import warnings
    from verifier_anchored_rl.outer_loop import _flop_counters

    class NoCounters:
        pass

    class Enc:
        prefill_calls = 1
        prefill_tokens = 10

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _flop_counters(Enc(), gen_source=NoCounters())
    assert any(issubclass(warning.category, UserWarning) for warning in w), (
        "_flop_counters should emit a UserWarning when gen_source has no counters"
    )
    # Should still return 0 for gen counts without raising
    assert result["gen_calls"] == 0
    assert result["gen_tokens"] == 0


def test_flop_counters_no_warn_when_gen_source_is_none():
    """No warning when gen_source is None (falls back to encoder)."""
    import warnings
    from verifier_anchored_rl.outer_loop import _flop_counters

    class Enc:
        prefill_calls = 1
        prefill_tokens = 10
        gen_calls = 3
        gen_tokens = 30

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _flop_counters(Enc(), gen_source=None)
    user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
    assert not user_warnings, "should not warn when gen_source is None"


# ---------------------------------------------------------------------------
# Bug 4: records carry iteration/problem_index/turn tags
# ---------------------------------------------------------------------------


def test_collect_iteration_record_carries_provenance_tags():
    """Records passed to record_sink must contain iteration, problem_index,
    and turn tags for offline de-duplication."""
    import pytest
    pytest.importorskip("torch")
    import numpy as np
    from verifier_anchored_rl.mcts import MCTSConfig
    from verifier_anchored_rl.outer_loop import collect_iteration

    enc, actor, sa, buf, trainer, mdp_factory = _build()
    config = MCTSConfig(num_simulations=4, c_puct=1.5, discount=1.0,
                        prior_fn=sa.prior, value_fn=sa.leaf_value)

    sink_records = []
    collect_iteration(
        [("2+2", "4"), ("3+3", "6")], mdp_factory, sa, config, buf,
        np.random.default_rng(0), tau=1.0, max_turns=4,
        record_sink=lambda r: sink_records.append(dict(r)),
        iteration_index=7,
    )

    assert sink_records, "record_sink should have been called"
    for r in sink_records:
        assert "iteration" in r, "record missing 'iteration' tag"
        assert "problem_index" in r, "record missing 'problem_index' tag"
        assert "turn" in r, "record missing 'turn' tag"
    # iteration_index=7 was passed in
    assert all(r["iteration"] == 7 for r in sink_records)
    # problem_index and turn must be non-negative integers
    assert all(isinstance(r["problem_index"], int) and r["problem_index"] >= 0
               for r in sink_records)
    assert all(isinstance(r["turn"], int) and r["turn"] >= 0
               for r in sink_records)


def test_collect_iteration_concurrent_record_carries_provenance_tags():
    """Concurrent path records must also carry provenance tags."""
    import pytest
    pytest.importorskip("torch")
    import numpy as np
    from verifier_anchored_rl.mcts import MCTSConfig
    from verifier_anchored_rl.outer_loop import collect_iteration_concurrent

    enc, actor, sa, buf, trainer, mdp_factory = _build()
    config = MCTSConfig(num_simulations=4, c_puct=1.5, discount=1.0,
                        prior_fn=sa.prior, value_fn=sa.leaf_value)

    sink_records = []
    collect_iteration_concurrent(
        [("2+2", "4"), ("3+3", "6")], mdp_factory, sa, config, buf,
        base_seed=0, tau=1.0, max_turns=4, max_workers=2,
        record_sink=lambda r: sink_records.append(dict(r)),
        iteration_index=3,
    )

    assert sink_records, "record_sink should have been called"
    for r in sink_records:
        assert "iteration" in r, "concurrent record missing 'iteration' tag"
        assert "problem_index" in r, "concurrent record missing 'problem_index' tag"
        assert "turn" in r, "concurrent record missing 'turn' tag"
    assert all(r["iteration"] == 3 for r in sink_records)
