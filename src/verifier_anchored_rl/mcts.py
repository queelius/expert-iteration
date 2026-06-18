"""Monte Carlo Tree Search planner.

Implements the four-phase MCTS loop from design.md (section 8) with
explicit decision/chance node bifurcation.

The pedagogical framing is the one set out in design.md (section 1): the
central object is an agent interacting with an environment and learning
from sampled reward-bearing transitions. MCTS is not the central method.
It is a sample-efficient policy improvement operator and a way to
generate stronger Bellman targets from sampled transitions.

Each chance node stores the empirical action-value estimate

    Qhat_MCTS(s, a) = W(s, a) / N(s, a)

obtained by averaging the discounted returns of all simulations that
passed through the (s, a) edge. The root's chance-node visit counts
yield the visit-count policy

    pi_MCTS(a | s) = N(s, a)^(1 / tau) / sum_b N(s, b)^(1 / tau)

which can be used directly for action selection or as a distillation
target for a separate policy model.

Selection uses PUCT:

    PUCT(s, a) = Q(s, a) + c_puct * P(a | s) * sqrt(N(s)) / (1 + N(s, a))

The prior P defaults to uniform. When ``config.c_puct`` is zero (or the
prior is missing) the planner falls back to the UCB1 form
``Q + sqrt(2 * ln(N(s)) / N(s, a))``.

Backups follow the per-edge form in design.md (section 8.4): given the
path of (chance_node, observed_reward) pairs collected during selection
plus expansion, and a leaf value G obtained from the rollout, every
chance node receives the return *from its own position*, not a single
terminal scalar broadcast to all ancestors. This matters whenever step
rewards are nonzero (the gridworld has step_reward = -0.01).

Only the ``GenerativeMDP`` protocol is required. MCTS samples
transitions instead of enumerating them; that is the whole point of
swapping value iteration / expectimax for MCTS in the first place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Generic, Hashable, TypeVar

import numpy as np

from .mdp import GenerativeMDP

S = TypeVar("S", bound=Hashable)
A = TypeVar("A")


# ---------------------------------------------------------------------------
# Tree node classes.
# ---------------------------------------------------------------------------


@dataclass
class ChanceNode(Generic[S, A]):
    """A (state, action) edge in the MCTS tree.

    The chance node owns the empirical action-value estimate

        mean_value = total_value / visits

    which is the running average of sampled returns observed below this
    edge. Its ``children`` list contains the next-state DecisionNodes
    that have actually been sampled (one per distinct ``s'``); the
    transition itself is stochastic, so the same edge may have several
    distinct children once enough simulations have run through it.

    Note: ``children`` is kept as a list (not a dict keyed by state) to
    avoid imposing a hashing convention on ``S`` for child lookup. The
    planner walks this list with equality comparison when finding an
    existing child for a sampled next state.
    """

    parent: "DecisionNode[S, A]"
    action: A
    children: list["DecisionNode[S, A]"] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    prior: float = 1.0  # Uniform default; overwritten by config.prior_fn.

    @property
    def mean_value(self) -> float:
        """Return Qhat_MCTS(s, a) = total_value / visits (0 if unvisited)."""
        return self.total_value / self.visits if self.visits > 0 else 0.0


@dataclass
class DecisionNode(Generic[S, A]):
    """A state at which the agent chooses an action.

    ``visits`` is N(s); summing the chance-node visit counts over all
    children of a fully-expanded decision node would give the same
    number, since every simulation that reaches s descends through
    exactly one chance edge.
    """

    state: S
    parent: ChanceNode[S, A] | None
    children: dict[A, ChanceNode[S, A]] = field(default_factory=dict)
    visits: int = 0
    terminal: bool = False
    untried_actions: list[A] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------


# A prior function takes a state and the action list and returns a list
# of probabilities (one per action, in the same order) summing to 1.
PriorFn = Callable[[object, list[object]], list[float]]

# A rollout policy takes (state, action_list, rng) and returns one action.
RolloutPolicy = Callable[[object, list[object], object], object]

# A leaf-value estimator V(s): used in place of a rollout to bootstrap the
# value of a non-terminal expanded leaf. In expert iteration this is
# max_a' Q_target(s', a') (method.tex Eq. bellmantarget); MCTS thus becomes
# AlphaZero-style (value net at the leaf) rather than pure rollout.
ValueFn = Callable[[object], float]


@dataclass
class MCTSConfig(Generic[S, A]):
    """Hyperparameters for ``mcts_search``.

    Parameters
    ----------
    num_simulations:
        Number of selection / expansion / rollout / backup loops.
    c_puct:
        PUCT exploration constant. Setting this to 0 (with no prior)
        causes the planner to fall back to the UCB1 selection form.
    discount:
        Discount factor used for backups and rollout returns.
    rollout_depth_limit:
        Maximum number of rollout steps after an expansion. If the
        rollout has not reached a terminal by this depth, the leaf
        value is the accumulated discounted reward so far (no
        bootstrap; treat as the actual sample so the estimator stays
        well-defined).
    prior_fn:
        Optional learned prior. Signature ``prior(state, action_list)``
        must return a list of probabilities summing to 1. If ``None``,
        the uniform prior is used.
    rollout_policy:
        Optional rollout policy. Signature
        ``rollout(state, action_list, rng)`` returns one action. If
        ``None``, a uniform random action is sampled.
    value_fn:
        Optional leaf-value estimator ``value(state) -> float``. When
        provided, a non-terminal expanded leaf is scored by ``value_fn``
        instead of a rollout (terminal leaves remain 0). ``None`` keeps
        the rollout behavior unchanged.
    """

    num_simulations: int = 100
    c_puct: float = 1.0
    discount: float = 0.99
    rollout_depth_limit: int = 100
    prior_fn: PriorFn | None = None
    rollout_policy: RolloutPolicy | None = None
    value_fn: ValueFn | None = None


# ---------------------------------------------------------------------------
# Core algorithm.
# ---------------------------------------------------------------------------


def _uniform_prior(actions: list[A]) -> list[float]:
    n = len(actions)
    if n == 0:
        return []
    p = 1.0 / n
    return [p] * n


def _make_decision_node(
    mdp: GenerativeMDP[S, A],
    state: S,
    parent: ChanceNode[S, A] | None,
) -> DecisionNode[S, A]:
    """Construct a DecisionNode and initialize its untried-action list."""
    terminal = mdp.is_terminal(state)
    untried: list[A] = [] if terminal else list(mdp.actions(state))
    return DecisionNode(
        state=state,
        parent=parent,
        children={},
        visits=0,
        terminal=terminal,
        untried_actions=untried,
    )


def _puct_score(
    chance: ChanceNode[S, A],
    parent_visits: int,
    c_puct: float,
) -> float:
    """PUCT exploration score, with a UCB1 fallback when c_puct is 0.

    Formulas (parent_visits is N(s), child_visits is N(s, a)):

        PUCT(s, a) = Q(s, a) + c_puct * P(a | s) * sqrt(N(s)) / (1 + N(s, a))
        UCB1(s, a) = Q(s, a) + sqrt(2 * ln(N(s)) / N(s, a))

    For unvisited children under PUCT the second term is c_puct * P *
    sqrt(N(s)); under UCB1 we return +inf so the unvisited child is
    selected first.
    """
    q = chance.mean_value
    n_sa = chance.visits

    if c_puct > 0.0:
        bonus = c_puct * chance.prior * math.sqrt(parent_visits) / (1.0 + n_sa)
        return q + bonus

    # UCB1 fallback (used when c_puct == 0).
    if n_sa == 0:
        return float("inf")
    if parent_visits <= 0:
        return q
    return q + math.sqrt(2.0 * math.log(max(parent_visits, 1)) / n_sa)


def _select_action(
    decision: DecisionNode[S, A],
    c_puct: float,
    rng: np.random.Generator,
) -> A:
    """Pick an action at a fully-expanded DecisionNode using PUCT."""
    best_score = -float("inf")
    best_actions: list[A] = []
    for action, chance in decision.children.items():
        score = _puct_score(chance, decision.visits, c_puct)
        if score > best_score:
            best_score = score
            best_actions = [action]
        elif score == best_score:
            best_actions.append(action)
    # Deterministic tie-break by sampling: this keeps two runs with the
    # same RNG seed identical even when several actions tie in score.
    if len(best_actions) == 1:
        return best_actions[0]
    idx = int(rng.integers(len(best_actions)))
    return best_actions[idx]


def _expand(
    mdp: GenerativeMDP[S, A],
    decision: DecisionNode[S, A],
    config: MCTSConfig[S, A],
    rng: np.random.Generator,
) -> tuple[ChanceNode[S, A], DecisionNode[S, A], float]:
    """Pop one untried action at ``decision`` and sample one transition.

    Returns the freshly created chance node, the resulting decision
    node (terminal or otherwise), and the immediate reward observed for
    the sampled transition.
    """
    # Pop an untried action. Random selection avoids any positional bias
    # that would otherwise correlate the expansion order with the action
    # enumeration order of the underlying MDP.
    idx = int(rng.integers(len(decision.untried_actions)))
    action = decision.untried_actions.pop(idx)

    # Compute the prior over the full action list available at this state.
    # We need this both to set the prior on the new chance node and so
    # that priors for siblings (computed at their own expansion) are
    # consistent. Using the original action ordering keeps priors aligned
    # between calls.
    all_actions = list(mdp.actions(decision.state))
    if config.prior_fn is not None:
        priors = list(config.prior_fn(decision.state, all_actions))
    else:
        priors = _uniform_prior(all_actions)
    prior_for_action = 1.0
    for a, p in zip(all_actions, priors):
        if a == action:
            prior_for_action = float(p)
            break

    chance = ChanceNode(
        parent=decision,
        action=action,
        children=[],
        visits=0,
        total_value=0.0,
        prior=prior_for_action,
    )
    decision.children[action] = chance

    next_state, reward = mdp.sample_transition(decision.state, action, rng)
    child_decision = _make_decision_node(mdp, next_state, parent=chance)
    chance.children.append(child_decision)

    return chance, child_decision, reward


def _rollout(
    mdp: GenerativeMDP[S, A],
    state: S,
    config: MCTSConfig[S, A],
    rng: np.random.Generator,
) -> float:
    """Simulate forward and return the discounted accumulated reward.

    The rollout starts from ``state`` (which is the next state observed
    after the freshly-expanded chance edge). It returns

        G = r_0 + gamma * r_1 + gamma^2 * r_2 + ...

    where r_0 is the reward of the first transition *inside the rollout*
    (the expansion's own reward is folded in during backup, not here).

    If a terminal state is reached, accumulation stops (and there is no
    bootstrap, consistent with the terminal-value-zero convention used
    elsewhere in the package). If the depth limit is hit first, the
    accumulated G so far is returned as the leaf value (again no
    bootstrap; we treat the truncated sample as the actual observation).
    """
    if mdp.is_terminal(state):
        return 0.0

    gamma = config.discount
    G = 0.0
    discount = 1.0
    current = state

    for _ in range(config.rollout_depth_limit):
        actions = list(mdp.actions(current))
        if not actions:
            break
        if config.rollout_policy is not None:
            action = config.rollout_policy(current, actions, rng)
        else:
            idx = int(rng.integers(len(actions)))
            action = actions[idx]
        next_state, reward = mdp.sample_transition(current, action, rng)
        G += discount * reward
        discount *= gamma
        if mdp.is_terminal(next_state):
            break
        current = next_state

    return G


def _backup(
    path: list[tuple[ChanceNode[S, A], float]],
    leaf_value: float,
    discount: float,
) -> None:
    """Back up the per-edge discounted return along ``path``.

    ``path`` is a list of (chance_node, observed_reward) pairs in the
    order encountered during descent: the leftmost entry is the chance
    edge taken at the root, the rightmost entry is the freshly-expanded
    chance edge whose reward is the immediate reward of the sampled
    transition into the rollout's starting state.

    Per design.md (section 8.4) the backup applies

        G = leaf_value
        for chance, reward in reversed(path):
            G = reward + gamma * G
            chance.visits += 1
            chance.total_value += G
            chance.parent.visits += 1

    so each ancestor receives the return *from its own position*, not a
    single broadcast scalar. This is the form that handles nonzero
    step rewards correctly.
    """
    G = leaf_value
    for chance, reward in reversed(path):
        G = reward + discount * G
        chance.visits += 1
        chance.total_value += G
        chance.parent.visits += 1


def mcts_search(
    mdp: GenerativeMDP[S, A],
    root_state: S,
    config: MCTSConfig[S, A],
    rng: np.random.Generator,
) -> DecisionNode[S, A]:
    """Run MCTS for ``config.num_simulations`` and return the root.

    The four phases (selection, expansion, simulation, backup) are run
    once per simulation. After ``num_simulations`` iterations the root
    holds the visit counts and Q estimates from which downstream code
    can extract ``pi_MCTS`` and ``Qhat_MCTS`` (see ``visit_count_policy``
    and ``root_q_estimates``).
    """
    root = _make_decision_node(mdp, root_state, parent=None)

    # A terminal root is a no-op: no actions to try, no rollouts to do.
    # We bump root.visits once to signal the function was actually
    # called, but no children are ever created.
    if root.terminal:
        root.visits += 0  # explicit no-op for readability
        return root

    for _ in range(config.num_simulations):
        path: list[tuple[ChanceNode[S, A], float]] = []
        node = root

        # ---- Selection -----------------------------------------------
        # Walk down through fully-expanded, non-terminal decision nodes.
        while (not node.terminal) and (not node.untried_actions):
            action = _select_action(node, config.c_puct, rng)
            chance = node.children[action]
            # Sample the next state from the chance edge. Re-use an
            # existing DecisionNode if we have seen this s' before for
            # this edge; otherwise create a fresh one.
            next_state, reward = mdp.sample_transition(node.state, chance.action, rng)
            child = None
            for existing in chance.children:
                if existing.state == next_state:
                    child = existing
                    break
            if child is None:
                child = _make_decision_node(mdp, next_state, parent=chance)
                chance.children.append(child)
            path.append((chance, reward))
            node = child

        # ---- Expansion + Simulation ----------------------------------
        if node.terminal:
            # Path already ends at a terminal: the rollout return is 0.
            # Backup whatever rewards were collected during selection.
            # Note: the backup increments visit counts for every chance
            # node on the path plus the parent decision node of each
            # such chance node, so the root.visits accounting is handled
            # there (the leftmost chance's parent is the root).
            _backup(path, leaf_value=0.0, discount=config.discount)
            # If the selection loop took zero steps (root itself was
            # terminal, which is caught above) there would be no chance
            # nodes to back up through; that case is already handled by
            # the early return above. Otherwise the root visit was
            # incremented by the backup.
            continue

        # Otherwise expand one untried action and rollout from the
        # sampled next state.
        chance, child_decision, expansion_reward = _expand(mdp, node, config, rng)
        path.append((chance, expansion_reward))

        if child_decision.terminal:
            leaf_value = 0.0
        elif config.value_fn is not None:
            # Bootstrap V(s') from the learned estimator (method.tex 3.7),
            # in place of a rollout. The per-edge backup still folds in the
            # expansion reward, so Q(s,a) -> r + gamma * V(s').
            leaf_value = config.value_fn(child_decision.state)
        else:
            leaf_value = _rollout(mdp, child_decision.state, config, rng)

        _backup(path, leaf_value=leaf_value, discount=config.discount)

    return root


# ---------------------------------------------------------------------------
# Helpers for reading out the search result.
# ---------------------------------------------------------------------------


def visit_count_policy(
    root: DecisionNode[S, A],
    temperature: float = 1.0,
) -> dict[A, float]:
    """Compute pi_MCTS(a | s) from the root's chance-node visit counts.

    ``pi_MCTS(a | s) = N(s, a)^(1 / tau) / sum_b N(s, b)^(1 / tau)``.

    With ``temperature -> 0`` this becomes argmax; with ``temperature == 1``
    it is the visit-count distribution; with larger temperatures it
    smooths toward uniform. A child with zero visits contributes zero
    mass except in the degenerate case where every child is unvisited,
    in which case a uniform distribution is returned.
    """
    if not root.children:
        return {}

    if temperature <= 0.0:
        # Argmax with deterministic tie-break by Python dict insertion order.
        best_a = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
        return {a: (1.0 if a == best_a else 0.0) for a in root.children}

    counts = {a: chance.visits for a, chance in root.children.items()}
    total = sum(counts.values())
    if total == 0:
        # Degenerate fallback: uniform over the expanded actions.
        n = len(counts)
        return {a: 1.0 / n for a in counts}

    inv_tau = 1.0 / temperature
    weighted = {a: (n ** inv_tau) for a, n in counts.items()}
    z = sum(weighted.values())
    return {a: w / z for a, w in weighted.items()}


def root_q_estimates(root: DecisionNode[S, A]) -> dict[A, float]:
    """Return Qhat_MCTS(s, a) = W(s, a) / N(s, a) at the root."""
    return {a: chance.mean_value for a, chance in root.children.items()}


def best_action(root: DecisionNode[S, A]) -> A:
    """Return argmax over actions of the root's chance-node visit count.

    Ties are broken in dict-iteration order (the order ``max`` returns
    the first qualifying entry).
    """
    if not root.children:
        raise ValueError("root has no expanded children; cannot pick an action")
    return max(root.children.items(), key=lambda kv: kv[1].visits)[0]


__all__ = [
    "ChanceNode",
    "DecisionNode",
    "MCTSConfig",
    "mcts_search",
    "visit_count_policy",
    "root_q_estimates",
    "best_action",
]
