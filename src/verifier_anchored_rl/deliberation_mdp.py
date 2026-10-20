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
    VerifierCall, get, holds, is_abandoned, is_verified, new_graph, solved,
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
class DeliberationAction:
    verb: Verb
    target: NodeId


class OutcomeCode(Enum):
    APPLIED = "applied"
    REJECTED = "rejected"


class RejectionReason(Enum):
    COMMIT_REJECTED = "commit_rejected"
    DUPLICATE_COMPUTE = "duplicate_compute"
    GENERATION_ERROR = "generation_error"
    PAYLOAD_APPLY_ERROR = "payload_apply_error"
    REDUCTIO_REJECTED = "reductio_rejected"
    REJECTED = "rejected"
    STALE_ACTION = "stale_action"


ACTION_SEMANTICS_VERSION = "goal-commit-v1"
TRANSITION_SEMANTICS_VERSION = "deliberation-transition-v1.1"
APPLY_SEMANTICS_VERSION = "legacy-coercion-v1"
PREVIOUS_OUTCOME_SCHEMA = "action-outcome-v1"
VERIFIER_CONTRACT_VERSION = "math-verifier-v1"
ACTION_TARGET_KINDS = {
    "split": ("goal",),
    "form_hypothesis": ("goal",),
    "compute": ("goal",),
    "verify": ("fact", "hypothesis"),
    "commit": ("goal",),
    "abandon": ("goal", "hypothesis"),
}


@dataclass(frozen=True)
class ActionOutcome:
    action: DeliberationAction
    code: OutcomeCode
    reason: RejectionReason | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, DeliberationAction):
            raise ValueError("outcome action must be a DeliberationAction")
        if not isinstance(self.code, OutcomeCode):
            raise ValueError("outcome code must be an OutcomeCode")
        if self.reason is not None and not isinstance(
                self.reason, RejectionReason):
            raise ValueError("outcome reason must be a RejectionReason")
        if self.code is OutcomeCode.APPLIED and self.reason is not None:
            raise ValueError("an applied outcome cannot have a rejection reason")
        if self.code is OutcomeCode.REJECTED and self.reason is None:
            raise ValueError("a rejected outcome requires a bounded reason")

    def as_record(self) -> dict[str, str | None]:
        return {
            "action": f"{self.action.verb.value}:{self.action.target}",
            "code": self.code.value,
            "reason": self.reason.value if self.reason is not None else None,
        }


@dataclass(frozen=True)
class DeliberationState:
    graph: Graph
    turn: int
    previous_outcome: ActionOutcome | None = None


def _is_open_goal(graph: Graph, node_id: NodeId) -> bool:
    n = get(graph, node_id)
    return (isinstance(n, Goal)
            and not solved(graph, node_id)
            and not is_abandoned(graph, node_id))


def _parent_goal_id(graph: Graph, node_id: NodeId) -> NodeId | None:
    """The structural owner of a goal, hypothesis, or fact."""
    for node in graph.nodes:
        if not isinstance(node, Goal):
            continue
        if node_id in node.children + node.candidates + node.facts:
            return node.id
    return None


def _has_closed_ancestor(graph: Graph, node_id: NodeId) -> bool:
    """Whether any owning goal above ``node_id`` is solved or abandoned."""
    parent = _parent_goal_id(graph, node_id)
    while parent is not None:
        if solved(graph, parent) or is_abandoned(graph, parent):
            return True
        parent = _parent_goal_id(graph, parent)
    return False


def _verification_rejected(
    graph: Graph,
    node_id: NodeId,
    verb: str = "check",
    expected: str | None = None,
) -> bool:
    """Whether the immutable claim has already failed deterministic VERIFY."""
    return any(
        isinstance(node, VerifierCall)
        and node.target == node_id
        and node.verb == verb
        and node.expected == expected
        and not node.success
        for node in graph.nodes
    )


def _owning_goal_statement(graph: Graph, fact_id: NodeId) -> str | None:
    """The statement of the goal whose facts tuple contains fact_id."""
    for n in graph.nodes:
        if isinstance(n, Goal) and fact_id in n.facts:
            return n.statement
    return None


def _verification_context(
    graph: Graph, node_id: NodeId, solve_claims: bool
) -> tuple[str, str | None]:
    """Return the deterministic symbolic route for a claim."""
    node = get(graph, node_id)
    if solve_claims and isinstance(node, Fact) and node.source == "derived":
        from .verifier_adapter import parse_solve_claim

        if parse_solve_claim(node.statement) is not None:
            owner = _owning_goal_statement(graph, node_id)
            if owner is not None:
                return "solve_check", owner
    return "check", None


def _witnesses(
    graph: Graph,
    goal: Goal,
    trusted_givens_committable: bool = False,
) -> tuple[NodeId, ...]:
    """Discharging, non-abandoned witnesses of a goal, ascending by id (most recent last)."""
    out = [wid for wid in goal.candidates
           if is_verified(graph, wid) and not is_abandoned(graph, wid)]
    out += [
        wid
        for wid in goal.facts
        if (
            isinstance(get(graph, wid), Fact)
            and (
                trusted_givens_committable
                or get(graph, wid).source != "given"
            )
            and holds(graph, wid)
            and not is_abandoned(graph, wid)
        )
    ]
    return tuple(sorted(out))


def available_actions(state: DeliberationState,
                      depth_limit: int | None = None, *,
                      reject_repeated_verify: bool = True,
                      active_subtrees_only: bool = True,
                      solve_claims: bool = False,
                      trusted_givens_committable: bool = False,
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
        if active_subtrees_only and _has_closed_ancestor(g, nid):
            continue
        if isinstance(node, Goal):
            if not _is_open_goal(g, nid):
                continue
            # SPLIT names a goal's full case set in one call; a goal that
            # already has children is not re-splittable (re-splitting only
            # appends duplicate cases and lets a controller loop on the root
            # forever). Solve the children, or COMPUTE/COMMIT this goal.
            if not node.children:
                acts.append(DeliberationAction(Verb.SPLIT, nid))
            acts.append(DeliberationAction(Verb.FORM_HYPOTHESIS, nid))
            acts.append(DeliberationAction(Verb.COMPUTE, nid))
            if _witnesses(g, node, trusted_givens_committable):
                acts.append(DeliberationAction(Verb.COMMIT, nid))
            acts.append(DeliberationAction(Verb.ABANDON, nid))
        elif isinstance(node, Hypothesis):
            if is_abandoned(g, nid):
                continue
            if not is_verified(g, nid):
                verify_verb, verify_expected = _verification_context(
                    g, nid, solve_claims)
                if not (reject_repeated_verify
                        and _verification_rejected(
                            g, nid, verify_verb, verify_expected)):
                    acts.append(DeliberationAction(Verb.VERIFY, nid))
                acts.append(DeliberationAction(Verb.ABANDON, nid))
        elif isinstance(node, Fact):
            if node.source == "derived" and not is_verified(g, nid):
                verify_verb, verify_expected = _verification_context(
                    g, nid, solve_claims)
                if not (reject_repeated_verify
                        and _verification_rejected(
                            g, nid, verify_verb, verify_expected)):
                    acts.append(DeliberationAction(Verb.VERIFY, nid))
    return tuple(acts)


# -----------------------------------------------------------------------------
# Reward conventions (design.md section 6; spec section 7)
# -----------------------------------------------------------------------------

TERMINAL_SUCCESS_REWARD = 1.0    # root COMMIT, verifier accepts
TERMINAL_FAILURE_REWARD = -1.0   # root COMMIT, verifier rejects
MASKED_ACTION_REWARD = 0.0       # live contract: verifier-only reward
LEGACY_MASKED_ACTION_REWARD = -0.01
SYMBOLIC_ONLY_REWARD = "symbolic-only-v1"
LEGACY_MASK_PENALTY_REWARD = "legacy-mask-penalty-v1"
INTERMEDIATE_VERIFIER_REWARD = 0.0     # v1: terminal-only reward
MAX_INTERMEDIATE_VERIFIER_REWARD = 0.2  # cap if the channel is ever enabled

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful mathematical assistant. Follow the instructions "
    "exactly and return only the requested JSON object."
)


def _terminal_verdict(witness_statement: str, label: str) -> bool:
    """Terminal check: the witness statement, or the value it claims.

    Witnesses that can pass VERIFY are typically equations ("6*7 = 42");
    dataset labels are typically bare values ("42"). The statement is
    checked against the label directly first; if it is an equation, the
    value it claims (the right-hand side) is checked as well. A syntactic
    identity ("42 = 42") claims nothing and must be rejected. The extraction is
    a deterministic string operation; the equation's own truth was already
    the verifier's job at VERIFY time. Honesty note (spec section 5): the
    broader class of contentless-but-true equations ("42 + 0 = 42") is the
    answer-verification bar itself, the same bar the legacy free-text
    commit had; closing it needs relevance checking, deferred in v1.
    """
    from .verifier_adapter import check_answer
    if check_answer(witness_statement, label).success:
        return True
    if "=" in witness_statement and "Eq" not in witness_statement:
        lhs, rhs = witness_statement.rsplit("=", 1)
        if lhs.strip() == rhs.strip():
            return False    # a syntactic identity claims nothing
        return check_answer(rhs, label).success
    return False


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
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 solve_claims: bool = False,
                 reject_repeated_verify: bool = True,
                 active_subtrees_only: bool = True,
                 render_committed_outcomes: bool = True,
                 generation_temperature: float = 0.0,
                 reward_contract: str = SYMBOLIC_ONLY_REWARD,
                 trusted_givens_committable: bool = False):
        from .verifier_adapter import MathVerifier
        self.problem = problem
        self.expected_answer = expected_answer
        self.givens = tuple(givens)
        self.depth_limit = depth_limit
        self.llm_client = llm_client
        self.encoder = encoder
        self.verifier = verifier if verifier is not None else MathVerifier()
        self.system_prompt = system_prompt
        # Solve-claim COMPUTE (spec 2026-07-03): off by default; the E13
        # control arm requires byte-identical behavior when False.
        self.solve_claims = solve_claims
        self.trusted_givens_committable = trusted_givens_committable
        # Independently switchable v1.1 repairs. Keeping the switches explicit
        # lets the experiment ledger isolate one legality manipulation at a
        # time while new runs use the coherent defaults.
        self.reject_repeated_verify = reject_repeated_verify
        self.active_subtrees_only = active_subtrees_only
        self.render_committed_outcomes = render_committed_outcomes
        self.generation_temperature = float(generation_temperature)
        if reward_contract not in {
            SYMBOLIC_ONLY_REWARD, LEGACY_MASK_PENALTY_REWARD
        }:
            raise ValueError(f"unknown reward contract {reward_contract!r}")
        self.reward_contract = reward_contract
        self.masked_action_reward = (
            LEGACY_MASKED_ACTION_REWARD
            if reward_contract == LEGACY_MASK_PENALTY_REWARD
            else MASKED_ACTION_REWARD
        )
        # Observational diagnostics only: reset and populated on every
        # transition so trajectory logs retain raw generator output and the
        # exact reason for a masked no-op (handoff task 28).
        self.last_transition_debug: dict | None = None

    # -- environment contract -------------------------------------------------

    def initial_state(self) -> DeliberationState:
        return DeliberationState(new_graph(self.problem, self.givens), 0)

    def actions(self, state: DeliberationState) -> tuple[DeliberationAction, ...]:
        return available_actions(
            state,
            depth_limit=self.depth_limit,
            reject_repeated_verify=self.reject_repeated_verify,
            active_subtrees_only=self.active_subtrees_only,
            solve_claims=self.solve_claims,
            trusted_givens_committable=(
                self.trusted_givens_committable
            ),
        )

    def contract_record(self) -> dict:
        """Exact transition settings that data and checkpoints must bind."""
        verifier_type = type(self.verifier)
        return {
            "action_semantics": ACTION_SEMANTICS_VERSION,
            "action_target_kinds": {
                verb: list(kinds)
                for verb, kinds in ACTION_TARGET_KINDS.items()
            },
            "transition_semantics": TRANSITION_SEMANTICS_VERSION,
            "apply_semantics": APPLY_SEMANTICS_VERSION,
            "previous_outcome_schema": PREVIOUS_OUTCOME_SCHEMA,
            "verifier_contract": VERIFIER_CONTRACT_VERSION,
            "verifier_implementation": (
                f"{verifier_type.__module__}.{verifier_type.__qualname__}"
            ),
            "reward_contract": self.reward_contract,
            "depth_limit": self.depth_limit,
            "solve_claims": self.solve_claims,
            "trusted_givens_committable": (
                self.trusted_givens_committable
            ),
            "reject_repeated_verify": self.reject_repeated_verify,
            "active_subtrees_only": self.active_subtrees_only,
            "render_committed_outcomes": self.render_committed_outcomes,
            "generation_temperature": self.generation_temperature,
        }

    def is_terminal(self, state: DeliberationState) -> bool:
        g = state.graph
        return (solved(g, g.root_id)
                or is_abandoned(g, g.root_id)
                or state.turn >= self.depth_limit)

    def sample_transition(self, state: DeliberationState,
                          action: DeliberationAction, rng
                          ) -> tuple[DeliberationState, float]:
        self.last_transition_debug = {"generator_payload": None,
                                      "mask_reason": None}
        if action not in set(self.actions(state)):
            next_state, reward = self._masked(state, "stale_action")
        elif action.verb is Verb.VERIFY:
            next_state, reward = self._do_verify(state, action.target)
        elif action.verb is Verb.COMMIT:
            next_state, reward = self._do_commit(state, action.target)
        elif action.verb is Verb.ABANDON:
            next_state, reward = self._do_abandon(state, action.target)
        else:
            next_state, reward = self._do_llm_verb(
                state, action, rng)     # Task 6

        raw_reason = self.last_transition_debug.get("mask_reason")
        if raw_reason is not None:
            reason = RejectionReason(raw_reason or "rejected")
            outcome = ActionOutcome(
                action, OutcomeCode.REJECTED, reason)
        else:
            outcome = ActionOutcome(action, OutcomeCode.APPLIED)
        return (
            DeliberationState(
                next_state.graph, next_state.turn, outcome),
            reward,
        )

    # -- the three no-LLM verbs ----------------------------------------------

    def _masked(self, state: DeliberationState, reason: str = "rejected"
                ) -> tuple[DeliberationState, float]:
        if self.last_transition_debug is None:
            self.last_transition_debug = {"generator_payload": None,
                                          "mask_reason": reason}
        else:
            self.last_transition_debug["mask_reason"] = reason
        return (DeliberationState(state.graph, state.turn + 1),
                self.masked_action_reward)

    def _do_verify(self, state: DeliberationState, target: NodeId
                   ) -> tuple[DeliberationState, float]:
        node = get(state.graph, target)
        statement = node.statement
        rewardable = _verification_is_rewardable(state.graph, target, statement)
        verb_name, context = _verification_context(
            state.graph, target, self.solve_claims)
        result = _r_verify(state.graph, target, verb_name, context,
                           self.verifier)
        g = result.graph if hasattr(result, "graph") else state.graph
        reward = 0.0
        if (rewardable and INTERMEDIATE_VERIFIER_REWARD > 0
                and is_verified(g, target)):
            reward = INTERMEDIATE_VERIFIER_REWARD
        return DeliberationState(g, state.turn + 1), reward

    def _do_commit(self, state: DeliberationState, target: NodeId
                   ) -> tuple[DeliberationState, float]:
        goal = get(state.graph, target)
        witness = _witnesses(
            state.graph,
            goal,
            self.trusted_givens_committable,
        )[-1]     # most recent
        result = _r_commit(state.graph, target, witness)
        if not hasattr(result, "graph"):                # cannot happen if avail
            return self._masked(state, "commit_rejected")
        g = result.graph
        reward = 0.0
        if target == g.root_id:
            success = _terminal_verdict(get(g, witness).statement,
                                        self.expected_answer)
            reward = (TERMINAL_SUCCESS_REWARD if success
                      else TERMINAL_FAILURE_REWARD)
        return DeliberationState(g, state.turn + 1), reward

    def _do_abandon(self, state: DeliberationState, target: NodeId
                    ) -> tuple[DeliberationState, float]:
        result = _r_abandon(state.graph, target)
        g = result.graph if hasattr(result, "graph") else state.graph
        return DeliberationState(g, state.turn + 1), 0.0

    # -- the three LLM verbs ---------------------------------------------------

    def _generate(self, state: DeliberationState, verb: Verb,
                  target: NodeId, rng) -> dict:
        """One constrained generation from view ++ suffix, either backend."""
        from .lenses import render_view, schema_for, suffix
        view = render_view(
            state.graph,
            target,
            include_committed_outcomes=self.render_committed_outcomes,
        )
        tail = suffix(verb, target, state, solve_claims=self.solve_claims)
        schema = schema_for(verb, target, state)
        seed = int(rng.integers(2 ** 31 - 1)) if rng is not None else None
        if self.encoder is not None:
            prefill = self.encoder.encode(view)
            payload, _n_tokens = self.encoder.instantiate_json(
                prefill,
                tail,
                schema,
                seed=seed,
                temperature=self.generation_temperature,
            )
            return payload
        if self.llm_client is None:
            raise RuntimeError(
                "DeliberationMDP needs an llm_client or an encoder for "
                f"{verb.name}")
        return self.llm_client.complete_json(
            self.system_prompt,
            view + tail,
            schema,
            seed=seed,
            temperature=self.generation_temperature,
        )

    @staticmethod
    def _coerce_child(c) -> str:
        """A SPLIT child as a string, from whatever shape the model sent.

        Small models wrap children in objects ({"text": ...},
        {"goal_id": 1, "sentence": ...}); the content is the longest
        string value. Bare values pass through str(). (E14 boundary fix.)
        """
        if isinstance(c, dict):
            strings = [v for v in c.values() if isinstance(v, str)]
            return max(strings, key=len) if strings else ""
        return str(c)

    @staticmethod
    def _coerce_connective(raw) -> str:
        """Normalize connective variants ('Exhaustive case', 'cases',
        'and') onto the two canonical values; unknown text falls through
        unchanged so reductio's own validation stays the arbiter."""
        text = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
        if text.startswith("exhaustive") or "case" in text:
            return "exhaustive_case"
        if text.startswith("conj") or text in ("and", "all"):
            return "conjunctive"
        return str(raw)

    def _do_llm_verb(self, state: DeliberationState,
                     action: DeliberationAction, rng
                     ) -> tuple[DeliberationState, float]:
        from .lenses import sanitize
        from .llm_client import LLMResponseError
        from .reductio import compute as _r_compute
        from .reductio import form_hypothesis as _r_form
        from .reductio import split as _r_split
        try:
            payload = self._generate(state, action.verb, action.target, rng)
            self.last_transition_debug["generator_payload"] = payload
        except (LLMResponseError, RuntimeError) as exc:
            self.last_transition_debug["generator_error"] = repr(exc)
            cause = getattr(exc, "cause", None)
            if cause is not None:
                self.last_transition_debug["generator_error_cause"] = repr(cause)
            raw = getattr(exc, "raw_response", None)
            if raw is not None:
                self.last_transition_debug["generator_error_raw"] = raw
            return self._masked(state, "generation_error")
        try:
            if action.verb is Verb.SPLIT:
                children = tuple(
                    s for s in (sanitize(self._coerce_child(c))
                                for c in payload["children"])
                    if s.strip())
                result = _r_split(state.graph, action.target,
                                  self._coerce_connective(payload["connective"]),
                                  children)
            elif action.verb is Verb.FORM_HYPOTHESIS:
                result = _r_form(state.graph, action.target,
                                 sanitize(str(payload["statement"])))
            else:  # Verb.COMPUTE
                stmt = sanitize(str(payload["statement"]))
                # Anti-loop guard: a COMPUTE that reproduces a fact already
                # attached to the SAME goal adds nothing and, unchecked, lets
                # a controller emit the same fact every turn until the depth
                # limit. Scoped per-goal on purpose: restating a case's value
                # at the parent (the aggregation step) is legitimate.
                tgt = get(state.graph, action.target)
                existing = getattr(tgt, "facts", ())
                if any(get(state.graph, fid).statement == stmt
                       for fid in existing):
                    return self._masked(state, "duplicate_compute")
                # Coerce premises leniently: ints or int-shaped strings that
                # name LIVE nodes survive; everything else is dropped rather
                # than failing the whole action (E14 boundary fix).
                live = {n.id for n in state.graph.nodes}
                premises = []
                for p in payload.get("premises", ()) or ():
                    try:
                        pid = int(p)
                    except (TypeError, ValueError):
                        continue
                    if pid in live:
                        premises.append(pid)
                result = _r_compute(state.graph, action.target, stmt,
                                    tuple(premises))
        except (KeyError, TypeError, ValueError) as exc:
            self.last_transition_debug["apply_error"] = repr(exc)
            return self._masked(state, "payload_apply_error")
        if not hasattr(result, "graph"):        # reductio Err
            self.last_transition_debug["reductio_error"] = repr(result)
            return self._masked(state, "reductio_rejected")
        return DeliberationState(result.graph, state.turn + 1), 0.0
