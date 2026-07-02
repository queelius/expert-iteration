"""The Deliberation MDP: (verb, target) actions over the reductio forest.

State is a reductio.Graph plus a turn counter. Three verbs call the frozen
LLM through a lens (SPLIT, FORM_HYPOTHESIS, COMPUTE); three never do
(VERIFY, COMMIT, ABANDON), which is what closes the self-verification hole:
the policy can neither author what gets checked nor what it is checked
against. All legality is computed in available_actions, once.

See docs/superpowers/specs/2026-07-02-deliberation-mdp-integration-design.md
and design.md sections 3-8.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .reductio import (
    Fact, Goal, Graph, Hypothesis, NodeId,
    get, holds, is_abandoned, is_verified, new_graph, solved,
)
from .reductio import abandon as _r_abandon
from .reductio import commit as _r_commit
from .reductio import verify as _r_verify


class Verb(Enum):
    SPLIT = "split"
    FORM_HYPOTHESIS = "form_hypothesis"
    COMPUTE = "compute"
    VERIFY = "verify"
    COMMIT = "commit"
    ABANDON = "abandon"


@dataclass(frozen=True)
class DeliberationState:
    graph: Graph
    turn: int


@dataclass(frozen=True)
class DeliberationAction:
    verb: Verb
    target: NodeId


def _is_open_goal(graph: Graph, node_id: NodeId) -> bool:
    n = get(graph, node_id)
    return (isinstance(n, Goal)
            and not solved(graph, node_id)
            and not is_abandoned(graph, node_id))


def _witnesses(graph: Graph, goal: Goal) -> tuple[NodeId, ...]:
    """Discharging witnesses of a goal, ascending by id (most recent last)."""
    out = [wid for wid in goal.candidates if is_verified(graph, wid)]
    out += [wid for wid in goal.facts if holds(graph, wid)]
    return tuple(sorted(out))


def available_actions(state: DeliberationState,
                      depth_limit: int | None = None
                      ) -> tuple[DeliberationAction, ...]:
    """Every legal (verb, target) pair, computed in one place.

    With a depth_limit, an exhausted state offers nothing (the episode is
    terminal). Legality per verb follows spec section 3.
    """
    if depth_limit is not None and state.turn >= depth_limit:
        return ()
    g = state.graph
    acts: list[DeliberationAction] = []
    for node in g.nodes:
        nid = node.id
        if isinstance(node, Goal):
            if not _is_open_goal(g, nid):
                continue
            acts.append(DeliberationAction(Verb.SPLIT, nid))
            acts.append(DeliberationAction(Verb.FORM_HYPOTHESIS, nid))
            acts.append(DeliberationAction(Verb.COMPUTE, nid))
            if _witnesses(g, node):
                acts.append(DeliberationAction(Verb.COMMIT, nid))
            acts.append(DeliberationAction(Verb.ABANDON, nid))
        elif isinstance(node, Hypothesis):
            if is_abandoned(g, nid):
                continue
            if not is_verified(g, nid):
                acts.append(DeliberationAction(Verb.VERIFY, nid))
                acts.append(DeliberationAction(Verb.ABANDON, nid))
        elif isinstance(node, Fact):
            if node.source == "derived" and not is_verified(g, nid):
                acts.append(DeliberationAction(Verb.VERIFY, nid))
    return tuple(acts)


# -----------------------------------------------------------------------------
# Reward conventions (design.md section 6; spec section 7)
# -----------------------------------------------------------------------------

TERMINAL_SUCCESS_REWARD = 1.0    # root COMMIT, verifier accepts
TERMINAL_FAILURE_REWARD = -1.0   # root COMMIT, verifier rejects
MASKED_ACTION_REWARD = -0.01     # LLM failure / stale action; turn advances
INTERMEDIATE_VERIFIER_REWARD = 0.0     # v1: terminal-only reward
MAX_INTERMEDIATE_VERIFIER_REWARD = 0.2  # cap if the channel is ever enabled

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful mathematical assistant. Follow the instructions "
    "exactly and return only the requested JSON object."
)


def _verification_is_rewardable(graph_before: Graph, target: NodeId,
                                statement: str) -> bool:
    """Anti-farming gates, kept as predicates (defense in depth).

    Rewardable means: the target was not already verified (dedup), the
    statement is not a tautology (identical sides after whitespace
    normalization), and the episode's rewarded-verification cap is not
    exhausted (count of already-verified nodes as the proxy). The v1
    reward coefficient is zero, so these gates guard a channel that is
    off; enabling it is a config change, not a code change.
    """
    if is_verified(graph_before, target):
        return False
    if "=" in statement:
        lhs, rhs = statement.split("=", 1)
        if lhs.strip() == rhs.strip():
            return False
    if INTERMEDIATE_VERIFIER_REWARD > 0:
        already = sum(1 for n in graph_before.nodes
                      if isinstance(n, (Fact, Hypothesis))
                      and is_verified(graph_before, n.id))
        if already * INTERMEDIATE_VERIFIER_REWARD >= MAX_INTERMEDIATE_VERIFIER_REWARD:
            return False
    return True


class DeliberationMDP:
    """GenerativeMDP over DeliberationState; see module docstring."""

    def __init__(self, problem: str, expected_answer: str,
                 givens: tuple[str, ...] = (), depth_limit: int = 16,
                 llm_client=None, encoder=None, verifier=None,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        from .verifier_adapter import MathVerifier
        self.problem = problem
        self.expected_answer = expected_answer
        self.givens = tuple(givens)
        self.depth_limit = depth_limit
        self.llm_client = llm_client
        self.encoder = encoder
        self.verifier = verifier if verifier is not None else MathVerifier()
        self.system_prompt = system_prompt

    # -- environment contract -------------------------------------------------

    def initial_state(self) -> DeliberationState:
        return DeliberationState(new_graph(self.problem, self.givens), 0)

    def actions(self, state: DeliberationState) -> tuple[DeliberationAction, ...]:
        return available_actions(state, depth_limit=self.depth_limit)

    def is_terminal(self, state: DeliberationState) -> bool:
        g = state.graph
        return (solved(g, g.root_id)
                or is_abandoned(g, g.root_id)
                or state.turn >= self.depth_limit)

    def sample_transition(self, state: DeliberationState,
                          action: DeliberationAction, rng
                          ) -> tuple[DeliberationState, float]:
        if action not in set(self.actions(state)):
            return self._masked(state)      # stale action from a search tree
        if action.verb is Verb.VERIFY:
            return self._do_verify(state, action.target)
        if action.verb is Verb.COMMIT:
            return self._do_commit(state, action.target)
        if action.verb is Verb.ABANDON:
            return self._do_abandon(state, action.target)
        return self._do_llm_verb(state, action, rng)     # Task 6

    # -- the three no-LLM verbs ----------------------------------------------

    def _masked(self, state: DeliberationState
                ) -> tuple[DeliberationState, float]:
        return (DeliberationState(state.graph, state.turn + 1),
                MASKED_ACTION_REWARD)

    def _do_verify(self, state: DeliberationState, target: NodeId
                   ) -> tuple[DeliberationState, float]:
        statement = get(state.graph, target).statement
        rewardable = _verification_is_rewardable(state.graph, target, statement)
        result = _r_verify(state.graph, target, "check", None, self.verifier)
        g = result.graph if hasattr(result, "graph") else state.graph
        reward = 0.0
        if (rewardable and INTERMEDIATE_VERIFIER_REWARD > 0
                and is_verified(g, target)):
            reward = INTERMEDIATE_VERIFIER_REWARD
        return DeliberationState(g, state.turn + 1), reward

    def _do_commit(self, state: DeliberationState, target: NodeId
                   ) -> tuple[DeliberationState, float]:
        from .verifier_adapter import check_answer
        goal = get(state.graph, target)
        witness = _witnesses(state.graph, goal)[-1]     # most recent
        result = _r_commit(state.graph, target, witness)
        if not hasattr(result, "graph"):                # cannot happen if avail
            return self._masked(state)
        g = result.graph
        reward = 0.0
        if target == g.root_id:
            verdict = check_answer(get(g, witness).statement,
                                   self.expected_answer)
            reward = (TERMINAL_SUCCESS_REWARD if verdict.success
                      else TERMINAL_FAILURE_REWARD)
        return DeliberationState(g, state.turn + 1), reward

    def _do_abandon(self, state: DeliberationState, target: NodeId
                    ) -> tuple[DeliberationState, float]:
        result = _r_abandon(state.graph, target)
        g = result.graph if hasattr(result, "graph") else state.graph
        return DeliberationState(g, state.turn + 1), 0.0

    def _do_llm_verb(self, state, action, rng):
        # Wired in Task 6; until then LLM verbs take the masked fallback.
        return self._masked(state)
