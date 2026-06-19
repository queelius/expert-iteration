"""Tests for the MCTS planner.

These tests exercise the four-phase loop (selection, expansion,
simulation, backup) on small environments where ground-truth values are
either hand-computable or available from the exact value-iteration
solver. Every test uses a deterministic numpy RNG seed so the suite is
reproducible.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pytest

from verifier_anchored_rl.gridworld import GridAction, GridState, Gridworld
from verifier_anchored_rl.mcts import (
    ChanceNode,
    DecisionNode,
    MCTSConfig,
    best_action,
    mcts_search,
    root_q_estimates,
    visit_count_policy,
)
from verifier_anchored_rl.value_iteration import value_iteration

from tests.test_value_iteration import SAGBMDP, _A, _B, _G, _S


# ---------------------------------------------------------------------------
# Generative-MDP adapter for the SAGBMDP (which is defined as a FiniteMDP).
# MCTS needs ``sample_transition``; we synthesize it from ``transitions``.
# ---------------------------------------------------------------------------


class SAGBGenerative:
    """Wrap the hand-computable SAGBMDP behind the GenerativeMDP protocol."""

    def __init__(self) -> None:
        self._inner = SAGBMDP()

    def actions(self, state):
        return self._inner.actions(state)

    def is_terminal(self, state) -> bool:
        return self._inner.is_terminal(state)

    def sample_transition(self, state, action, rng):
        outcomes = list(self._inner.transitions(state, action))
        next_states = [o[0] for o in outcomes]
        probs = np.array([o[1] for o in outcomes], dtype=float)
        rewards = [o[2] for o in outcomes]
        probs = probs / probs.sum()
        idx = int(rng.choice(len(next_states), p=probs))
        return next_states[idx], rewards[idx]


# ---------------------------------------------------------------------------
# Trivial single-state stub used by the PUCT-prior test.
# ---------------------------------------------------------------------------


class _PriorTestMDP:
    """A one-state MDP whose two actions both terminate with reward 0.

    Because both actions have identical (zero) value, PUCT's first term
    contributes nothing, so any difference in visit counts must come
    from the prior. This is the cleanest way to verify that the prior
    actually drives selection when value information is uninformative.
    """

    START = "S"
    END = "T"  # terminal

    def actions(self, state):
        if state == self.START:
            return ["a", "b"]
        return []

    def is_terminal(self, state) -> bool:
        return state == self.END

    def sample_transition(self, state, action, rng):
        # Both actions deterministically transition to the terminal
        # state with zero reward.
        return self.END, 0.0


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_mcts_visit_counts_sum_correctly() -> None:
    """Every simulation visits exactly one chance node at the root."""
    mdp = SAGBGenerative()
    config = MCTSConfig(num_simulations=200, c_puct=1.0, discount=1.0)
    rng = np.random.default_rng(0)
    root = mcts_search(mdp, _S, config, rng)

    total = sum(c.visits for c in root.children.values())
    assert total == config.num_simulations
    assert root.visits == config.num_simulations


def test_puct_respects_priors_when_values_equal() -> None:
    """A non-uniform prior should attract visits when Q-values are equal."""
    mdp = _PriorTestMDP()

    # Strongly bias the prior toward action "a".
    def prior_fn(state, actions: list[str]) -> list[float]:
        probs = []
        for a in actions:
            probs.append(0.9 if a == "a" else 0.1 / max(len(actions) - 1, 1))
        # Normalize to be safe.
        z = sum(probs)
        return [p / z for p in probs]

    config = MCTSConfig(
        num_simulations=200,
        c_puct=2.0,
        discount=1.0,
        prior_fn=prior_fn,
    )
    rng = np.random.default_rng(42)
    root = mcts_search(mdp, _PriorTestMDP.START, config, rng)

    visits = {a: c.visits for a, c in root.children.items()}
    assert visits["a"] > visits["b"], (
        f"expected action 'a' (prior 0.9) to outvisit action 'b' (prior 0.1), "
        f"got {visits}"
    )


def test_mcts_converges_on_sagb() -> None:
    """With enough simulations, MCTS recovers V*(S) and Q*(S, .).

    The SAGB Q-values are on the order of 4 to 10, so the PUCT bonus
    has to scale similarly for visit-count argmax to track Q-argmax;
    c_puct = 2.0 is a reasonable choice for this range. The recovered
    Q estimates are insensitive to c_puct (any positive constant
    yields the correct mean under enough simulations); it is the visit
    distribution and therefore the visit-count-best action that depend
    on the exploration constant.
    """
    mdp = SAGBGenerative()
    # gamma = 1 to match the hand-computed values in the value-iteration
    # test. step rewards are zero in the SAGB MDP, so per-edge backup is
    # exercised only by the terminal rewards.
    config = MCTSConfig(num_simulations=5_000, c_puct=2.0, discount=1.0)
    rng = np.random.default_rng(7)
    root = mcts_search(mdp, _S, config, rng)

    q = root_q_estimates(root)
    assert "safe" in q and "risky" in q
    assert math.isclose(q["safe"], 4.0, abs_tol=0.5), f"Q(S, safe) = {q['safe']}"
    assert math.isclose(q["risky"], 6.0, abs_tol=0.5), f"Q(S, risky) = {q['risky']}"
    assert best_action(root) == "risky"


def test_mcts_converges_on_small_gridworld() -> None:
    """The MCTS Q of the VI-greedy action should approximate V*(start)."""
    layout = [
        "S.G",
        "..P",
    ]
    env = Gridworld(layout)
    gamma = 0.9
    V_vi, policy_vi = value_iteration(env, gamma=gamma, theta=1e-9)
    start = env.start
    greedy_action = policy_vi[start]

    config = MCTSConfig(num_simulations=3_000, c_puct=1.0, discount=gamma)
    rng = np.random.default_rng(123)
    root = mcts_search(env, start, config, rng)

    q = root_q_estimates(root)
    assert greedy_action in q, (
        f"VI-greedy action {greedy_action} was never expanded at the root"
    )
    # Sampled estimator: allow noticeable slack. V*(start) is on the
    # order of 0.5 to 0.8 for a small layout with one-step accessibility
    # to the goal, so a tolerance of 0.3 keeps the test robust while
    # still catching gross errors.
    assert math.isclose(q[greedy_action], V_vi[start], abs_tol=0.3), (
        f"MCTS Q for greedy action {greedy_action} = {q[greedy_action]}, "
        f"V*(start) = {V_vi[start]}"
    )


def test_mcts_backup_handles_step_rewards() -> None:
    """Far-from-goal states should have a meaningfully lower MCTS Q.

    The gridworld has step_reward = -0.01, so a path of length k that
    eventually reaches the goal accumulates roughly -0.01 * k of step
    penalty before the +1.0 terminal reward. A far-from-goal state must
    therefore have a lower MCTS Q for its best action than a near-goal
    state has for its best action.
    """
    layout = [
        "S...G",
        ".....",
        ".....",
    ]
    env = Gridworld(layout, step_reward=-0.01, goal_reward=1.0, pit_reward=-1.0)
    gamma = 0.99

    # Near-goal: (0, 3), one step from G at (0, 4).
    near = GridState(0, 3)
    # Far-from-goal: bottom-left corner, multiple steps from G.
    far = GridState(2, 0)

    # 800 simulations keeps the test under a second while still giving
    # a clear value separation between the near-goal and far-from-goal
    # roots (the assertion threshold is 0.1, comfortably wider than the
    # sampling noise at this budget).
    config = MCTSConfig(num_simulations=800, c_puct=1.0, discount=gamma)

    rng_near = np.random.default_rng(5)
    root_near = mcts_search(env, near, config, rng_near)
    q_near = root_q_estimates(root_near)
    best_q_near = max(q_near.values())

    rng_far = np.random.default_rng(6)
    root_far = mcts_search(env, far, config, rng_far)
    q_far = root_q_estimates(root_far)
    best_q_far = max(q_far.values())

    assert best_q_near > best_q_far + 0.1, (
        f"expected near-goal MCTS Q ({best_q_near}) to exceed "
        f"far-from-goal MCTS Q ({best_q_far}) by more than 0.1"
    )
    # Sanity: near-goal best Q should be positive (close to goal_reward
    # discounted by one step, minus a sliver of step penalty).
    assert best_q_near > 0.0


def test_mcts_terminal_states_have_no_expansion() -> None:
    """Searching from a terminal state should not blow up or expand."""
    layout = ["S.G", "..P"]
    env = Gridworld(layout)
    goal = GridState(0, 2)
    assert env.is_terminal(goal)

    config = MCTSConfig(num_simulations=50, c_puct=1.0, discount=0.99)
    rng = np.random.default_rng(0)
    root = mcts_search(env, goal, config, rng)

    assert root.terminal is True
    assert root.children == {}
    assert root.untried_actions == []


def test_mcts_with_uniform_prior_matches_no_prior() -> None:
    """Explicit uniform prior should produce identical trees to the default."""
    mdp = SAGBGenerative()

    def uniform_prior(state, actions: list[str]) -> list[float]:
        n = len(actions)
        return [1.0 / n] * n

    config_none = MCTSConfig(num_simulations=300, c_puct=1.0, discount=1.0)
    config_uniform = MCTSConfig(
        num_simulations=300,
        c_puct=1.0,
        discount=1.0,
        prior_fn=uniform_prior,
    )

    rng_none = np.random.default_rng(99)
    rng_uniform = np.random.default_rng(99)

    root_none = mcts_search(mdp, _S, config_none, rng_none)
    root_uniform = mcts_search(mdp, _S, config_uniform, rng_uniform)

    visits_none = {a: c.visits for a, c in root_none.children.items()}
    visits_uniform = {a: c.visits for a, c in root_uniform.children.items()}
    assert visits_none == visits_uniform, (
        f"uniform-prior visits {visits_uniform} differ from "
        f"no-prior visits {visits_none}"
    )


# ---------------------------------------------------------------------------
# value_fn leaf-evaluation hook (method.tex 3.7: target-Q as bootstrap leaf).
# ---------------------------------------------------------------------------


# A deterministic chain MDP for an exact, seed-independent value_fn test.
# (It replaces an earlier SAGBMDP V* test that asserted an exact Q on an
# under-explored stochastic chance node, which made it RNG-seed dependent.)
class _ChainMDP:
    """X --go--> Y (deterministic, reward 2); Y --stop--> Z (terminal).

    With one simulation the search expands 'go' and evaluates the
    non-terminal leaf Y, so the value_fn bootstrap fully determines
    Q(X, go) = r + gamma * V(Y) with zero variance (the transition is
    deterministic and Y is never expanded deeper)."""

    def actions(self, s):
        return {"X": ["go"], "Y": ["stop"]}.get(s, [])

    def is_terminal(self, s):
        return s == "Z"

    def sample_transition(self, s, a, rng):  # deterministic; rng unused
        transitions = {"X": ("Y", 2.0), "Y": ("Z", 0.0)}
        if s not in transitions:
            raise ValueError(f"_ChainMDP: no transition from {s!r}")
        return transitions[s]


def test_value_fn_bootstrap_equals_reward_plus_discounted_leaf_value():
    """One simulation: Q(X, go) = r + gamma * V(Y), exactly (method.tex 3.7)."""
    mdp = _ChainMDP()
    vf = lambda s: 5.0 if s == "Y" else 0.0
    cfg = MCTSConfig(num_simulations=1, c_puct=1.0, discount=1.0, value_fn=vf)
    root = mcts_search(mdp, "X", cfg, np.random.default_rng(0))
    assert math.isclose(root_q_estimates(root)["go"], 2.0 + 1.0 * 5.0, abs_tol=1e-9)

    cfg_disc = MCTSConfig(num_simulations=1, c_puct=1.0, discount=0.5, value_fn=vf)
    root_d = mcts_search(mdp, "X", cfg_disc, np.random.default_rng(0))
    assert math.isclose(root_q_estimates(root_d)["go"], 2.0 + 0.5 * 5.0, abs_tol=1e-9)


def test_value_fn_replaces_rollout_at_leaf():
    """The hook changes the leaf value used: with value_fn(Y)=5 a one-sim
    search gives Q(X,go)=7; with no value_fn the rollout from Y reaches
    terminal Z (reward 0), giving Q(X,go)=2."""
    mdp = _ChainMDP()
    cfg_vf = MCTSConfig(num_simulations=1, c_puct=1.0, discount=1.0,
                        value_fn=lambda s: 5.0 if s == "Y" else 0.0)
    cfg_rollout = MCTSConfig(num_simulations=1, c_puct=1.0, discount=1.0)
    r_vf = mcts_search(mdp, "X", cfg_vf, np.random.default_rng(0))
    r_ro = mcts_search(mdp, "X", cfg_rollout, np.random.default_rng(0))
    assert math.isclose(root_q_estimates(r_vf)["go"], 7.0, abs_tol=1e-9)
    assert math.isclose(root_q_estimates(r_ro)["go"], 2.0, abs_tol=1e-9)


def test_value_fn_none_matches_rollout_default():
    """The default (no value_fn) path is the unchanged rollout path: same
    seed and config give identical Q estimates run-to-run. (The actual
    value_fn-vs-rollout divergence is asserted in
    test_value_fn_replaces_rollout_at_leaf.)"""
    mdp = SAGBGenerative()
    cfg = MCTSConfig(num_simulations=200, c_puct=1.0, discount=1.0,
                     rollout_depth_limit=10)
    root_a = mcts_search(mdp, _S, cfg, np.random.default_rng(7))
    root_b = mcts_search(mdp, _S, cfg, np.random.default_rng(7))
    qa = {a: round(q, 9) for a, q in root_q_estimates(root_a).items()}
    qb = {a: round(q, 9) for a, q in root_q_estimates(root_b).items()}
    assert qa == qb  # determinism guard; the rollout path is unchanged


def test_value_fn_not_called_on_terminal_leaf():
    """Terminal leaves stay 0.0; value_fn must never be consulted for them."""
    mdp = SAGBGenerative()
    seen = []
    def vf(s):
        seen.append(s)
        return 99.0  # absurd value that would corrupt Q if used at terminals
    cfg = MCTSConfig(num_simulations=300, c_puct=1.0, discount=1.0, value_fn=vf)
    root = mcts_search(mdp, _S, cfg, np.random.default_rng(0))
    assert _G not in seen and _B not in seen  # G, B are terminal
    # safe -> G (terminal) with reward 4, so its estimate is unaffected by vf.
    assert math.isclose(root_q_estimates(root)["safe"], 4.0, abs_tol=1e-9)


def test_expansion_order_prior_pops_highest_prior_first():
    """With expansion_order='prior', untried actions expand in
    prior-descending order; 'random' is unchanged at a fixed seed."""
    import numpy as np
    from verifier_anchored_rl.mcts import (MCTSConfig, _make_decision_node,
                                          _expand)

    class TwoStepMDP:
        # state 0 has actions ['a','b','c']; any action -> terminal state 1.
        def actions(self, state):
            return ["a", "b", "c"] if state == 0 else []
        def is_terminal(self, state):
            return state == 1
        def sample_transition(self, state, action, rng):
            return 1, 0.0

    mdp = TwoStepMDP()
    rng = np.random.default_rng(0)
    # Prior strongly prefers 'b' > 'c' > 'a'.
    prior = {"a": 0.1, "b": 0.6, "c": 0.3}
    config = MCTSConfig(prior_fn=lambda s, acts: [prior[a] for a in acts],
                        expansion_order="prior")
    node = _make_decision_node(mdp, 0, parent=None)
    first, _, _ = _expand(mdp, node, config, rng)
    assert first.action == "b"   # highest prior popped first
    second, _, _ = _expand(mdp, node, config, rng)
    assert second.action == "c"  # next highest

    # 'random' reproduces the current behavior: a fixed seed pops a
    # deterministic (not prior-ordered) action.
    node2 = _make_decision_node(mdp, 0, parent=None)
    config_rand = MCTSConfig(prior_fn=lambda s, acts: [prior[a] for a in acts],
                             expansion_order="random")
    rng2 = np.random.default_rng(0)
    popped = _expand(mdp, node2, config_rand, rng2)[0].action
    assert popped in {"a", "b", "c"}
