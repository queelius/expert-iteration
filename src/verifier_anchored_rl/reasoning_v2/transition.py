"""One-controller-turn macro transitions for the v2 reasoning MDP."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

from .actions import (
    DecompositionDraft,
    Err,
    RefutationDraft,
    ResultDraft,
    SolutionDraft,
    decompose,
    propose_refutation,
    propose_result,
    propose_solution,
)
from .closure import (
    AggregatePayloadBuilder,
    ClosureEvent,
    ClosureResult,
    ClosureVerifier,
    run_closure,
)
from .menu import POLICY_VERB_ORDER, available_actions, requires_internal_closure
from .model import (
    Action,
    Claim,
    ClaimKind,
    ClaimStatus,
    Connective,
    Goal,
    GoalStatus,
    Graph,
    NodeId,
    OutcomeCode,
    RejectionReason,
    RefutationPayload,
    ResultPayload,
    SolutionPayload,
    State,
    TransitionOutcome,
    Verb,
    direct_child_goals,
    get,
    is_active,
    is_solved,
    validate_graph,
)
from .provenance import DecisionProvenance
from .verifier import GenerationContract


TRANSITION_SEMANTICS_VERSION = "reasoning-macro-transition-v2"
REWARD_CONTRACT_VERSION = "symbolic-only-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

PolicyDraft: TypeAlias = (
    DecompositionDraft | ResultDraft | SolutionDraft | RefutationDraft
)


class MacroTransitionError(RuntimeError):
    """Raised when a macro step violates the trusted environment contract."""


class PolicyGenerationError(RuntimeError):
    """A provider or generation transport failed before typed payload output."""


class PolicyPayloadError(ValueError):
    """Generated output could not be converted to the required typed draft."""


@runtime_checkable
class PolicyDraftGenerator(Protocol):
    """Produce one typed draft for a legal learned action."""

    def generate(self, state: State, action: Action) -> PolicyDraft:
        ...


_DRAFT_TYPES: dict[Verb, type[PolicyDraft]] = {
    Verb.DECOMPOSE: DecompositionDraft,
    Verb.PROPOSE_RESULT: ResultDraft,
    Verb.PROPOSE_SOLUTION: SolutionDraft,
    Verb.PROPOSE_REFUTATION: RefutationDraft,
}

_CLAIM_KINDS = {
    Verb.PROPOSE_RESULT: ClaimKind.RESULT,
    Verb.PROPOSE_SOLUTION: ClaimKind.SOLUTION,
    Verb.PROPOSE_REFUTATION: ClaimKind.REFUTATION,
}


@dataclass(frozen=True)
class LearnedEffect:
    graph: Graph
    outcome: TransitionOutcome
    triggering_claim_id: NodeId | None
    policy_draft: PolicyDraft | None

    def __post_init__(self) -> None:
        if not isinstance(self.graph, Graph):
            raise ValueError("learned effect graph has an invalid type")
        validate_graph(self.graph)
        if not isinstance(self.outcome, TransitionOutcome):
            raise ValueError("learned effect outcome has an invalid type")
        if self.outcome.action.verb not in POLICY_VERB_ORDER:
            raise ValueError("learned effect action is not a policy verb")
        expected_draft = _DRAFT_TYPES[self.outcome.action.verb]
        if self.triggering_claim_id is not None and (
            type(self.triggering_claim_id) is not int
            or self.triggering_claim_id < 0
        ):
            raise ValueError("learned effect trigger has an invalid node id")
        if self.outcome.code is OutcomeCode.REJECTED:
            if self.triggering_claim_id is not None:
                raise ValueError("rejected learned effects cannot trigger closure")
            if self.outcome.reason is RejectionReason.ALGEBRA_REJECTED:
                if type(self.policy_draft) is not expected_draft:
                    raise ValueError(
                        "algebra rejection requires its exact typed draft"
                    )
            elif self.policy_draft is not None:
                raise ValueError(
                    "non-algebra rejection cannot retain a typed draft"
                )
            return
        if type(self.policy_draft) is not expected_draft:
            raise ValueError("applied learned effect needs its typed draft")
        if self.outcome.action.verb is Verb.DECOMPOSE:
            if self.triggering_claim_id is not None:
                raise ValueError("DECOMPOSE cannot create a triggering Claim")
            return
        if self.triggering_claim_id is None:
            raise ValueError("applied proposal must create a triggering Claim")
        claim = get(self.graph, self.triggering_claim_id)
        expected_kind = _CLAIM_KINDS[self.outcome.action.verb]
        if (
            not isinstance(claim, Claim)
            or claim.kind is not expected_kind
            or claim.owner_goal != self.outcome.action.target
            or claim.status is not ClaimStatus.UNCHECKED
            or not is_active(self.graph, claim.id)
        ):
            raise ValueError("learned effect trigger is not the created Claim")


def _rejected_effect(
    graph: Graph,
    action: Action,
    reason: RejectionReason,
    *,
    policy_draft: PolicyDraft | None = None,
) -> LearnedEffect:
    return LearnedEffect(
        graph=graph,
        outcome=TransitionOutcome(
            action=action,
            code=OutcomeCode.REJECTED,
            reason=reason,
        ),
        triggering_claim_id=None,
        policy_draft=policy_draft,
    )


def _apply_draft(
    graph: Graph,
    action: Action,
    draft: PolicyDraft,
    generation_contract: GenerationContract,
):
    if action.verb is Verb.DECOMPOSE:
        return decompose(graph, action.target, draft, generation_contract)
    if action.verb is Verb.PROPOSE_RESULT:
        return propose_result(graph, action.target, draft, generation_contract)
    if action.verb is Verb.PROPOSE_SOLUTION:
        return propose_solution(graph, action.target, draft, generation_contract)
    if action.verb is Verb.PROPOSE_REFUTATION:
        return propose_refutation(
            graph, action.target, draft, generation_contract
        )
    raise MacroTransitionError("unsupported learned action verb")


def _infer_applied_draft(
    before: Graph,
    after: Graph,
    action: Action,
) -> tuple[PolicyDraft, NodeId | None]:
    if action.verb is Verb.DECOMPOSE:
        target = get(after, action.target)
        if not isinstance(target, Goal) or not isinstance(
            target.connective, Connective
        ):
            raise MacroTransitionError(
                "applied DECOMPOSE has no stored connective"
            )
        children = direct_child_goals(after, target.id)
        if not children:
            raise MacroTransitionError(
                "applied DECOMPOSE has no stored child Goals"
            )
        return (
            DecompositionDraft(
                target.connective,
                tuple(child.statement for child in children),
                tuple(child.obligation for child in children),
            ),
            None,
        )

    trigger = len(before.nodes)
    try:
        claim = get(after, trigger)
    except KeyError as exc:
        raise MacroTransitionError(
            "applied proposal has no created Claim"
        ) from exc
    if not isinstance(claim, Claim) or claim.owner_goal != action.target:
        raise MacroTransitionError(
            "applied proposal created the wrong Claim"
        )
    payload = claim.payload
    if action.verb is Verb.PROPOSE_RESULT and isinstance(
        payload, ResultPayload
    ):
        return ResultDraft(payload.objective, payload.value), trigger
    if action.verb is Verb.PROPOSE_SOLUTION and isinstance(
        payload, SolutionPayload
    ):
        return SolutionDraft(payload.variable, payload.value), trigger
    if action.verb is Verb.PROPOSE_REFUTATION and isinstance(
        payload, RefutationPayload
    ):
        return (
            RefutationDraft(
                payload.mode,
                payload.variable,
                payload.candidate,
            ),
            trigger,
        )
    raise MacroTransitionError(
        "created Claim payload differs from the learned action"
    )


def execute_learned_action(
    state: State,
    action: Action,
    *,
    turn_limit: int,
    generator: PolicyDraftGenerator,
    generation_contract: GenerationContract,
) -> LearnedEffect:
    """Generate and apply one legal learned action without running closure."""
    if not isinstance(state, State) or not isinstance(action, Action):
        raise TypeError("execute_learned_action needs a State and Action")
    if action.verb not in POLICY_VERB_ORDER:
        raise MacroTransitionError("action is not a learned policy verb")
    if type(turn_limit) is not int or turn_limit <= 0:
        raise ValueError("turn_limit must be a positive integer")
    if action not in available_actions(state, turn_limit):
        raise MacroTransitionError("action is not in the current learned menu")
    if not isinstance(generator, PolicyDraftGenerator):
        raise TypeError("generator has an invalid type")
    if not isinstance(generation_contract, GenerationContract):
        raise TypeError("generation_contract has an invalid type")

    try:
        draft = generator.generate(state, action)
    except PolicyPayloadError:
        return _rejected_effect(
            state.graph, action, RejectionReason.PAYLOAD_ERROR
        )
    except PolicyGenerationError:
        return _rejected_effect(
            state.graph, action, RejectionReason.GENERATION_ERROR
        )

    expected_type = _DRAFT_TYPES[action.verb]
    if type(draft) is not expected_type:
        return _rejected_effect(
            state.graph, action, RejectionReason.PAYLOAD_ERROR
        )
    result = _apply_draft(state.graph, action, draft, generation_contract)
    if isinstance(result, Err):
        return _rejected_effect(
            state.graph,
            action,
            RejectionReason.ALGEBRA_REJECTED,
            policy_draft=draft,
        )

    updated = result.graph
    canonical_draft, trigger = _infer_applied_draft(
        state.graph, updated, action
    )
    if action.verb is Verb.DECOMPOSE and len(updated.nodes) <= len(
        state.graph.nodes
    ):
        raise MacroTransitionError("DECOMPOSE did not allocate child Goals")
    if action.verb is not Verb.DECOMPOSE and (
        len(updated.nodes) != len(state.graph.nodes) + 1
    ):
        raise MacroTransitionError(
            "learned proposal did not allocate exactly one Claim"
        )
    return LearnedEffect(
        graph=updated,
        outcome=TransitionOutcome(action, OutcomeCode.APPLIED),
        triggering_claim_id=trigger,
        policy_draft=canonical_draft,
    )


@dataclass(frozen=True)
class MacroTransition:
    state: State
    action: Action
    policy_draft: PolicyDraft | None
    outcome: TransitionOutcome
    closure_events: tuple[ClosureEvent, ...]
    reward: int
    next_state: State
    terminal: bool
    turn_limit: int
    environment_contract_sha256: str
    collection_contract_sha256: str | None = None
    decision_provenance: DecisionProvenance | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, State):
            raise ValueError("macro transition state has an invalid type")
        if not isinstance(self.action, Action):
            raise ValueError("macro transition action has an invalid type")
        if self.action.verb not in POLICY_VERB_ORDER:
            raise ValueError("macro transition action is not learned")
        if not isinstance(self.outcome, TransitionOutcome):
            raise ValueError("macro transition outcome has an invalid type")
        if self.outcome.action != self.action:
            raise ValueError("macro transition outcome action differs")
        expected_draft = _DRAFT_TYPES[self.action.verb]
        if self.outcome.code is OutcomeCode.APPLIED:
            if type(self.policy_draft) is not expected_draft:
                raise ValueError("applied transition needs its typed draft")
        elif self.outcome.reason is RejectionReason.ALGEBRA_REJECTED:
            if type(self.policy_draft) is not expected_draft:
                raise ValueError("algebra rejection needs its typed draft")
        elif self.policy_draft is not None:
            raise ValueError(
                "generation and payload rejection cannot retain a draft"
            )
        if not isinstance(self.next_state, State):
            raise ValueError("macro transition next state has an invalid type")
        if type(self.turn_limit) is not int or self.turn_limit <= 0:
            raise ValueError("macro transition turn limit is invalid")
        if (
            not isinstance(self.environment_contract_sha256, str)
            or _SHA256_RE.fullmatch(
                self.environment_contract_sha256
            ) is None
        ):
            raise ValueError("macro transition environment digest is invalid")
        if self.collection_contract_sha256 is not None and (
            not isinstance(self.collection_contract_sha256, str)
            or _SHA256_RE.fullmatch(self.collection_contract_sha256) is None
        ):
            raise ValueError("macro transition collection digest is invalid")
        if self.decision_provenance is not None and not isinstance(
            self.decision_provenance, DecisionProvenance
        ):
            raise ValueError("macro transition decision provenance is invalid")
        if (self.collection_contract_sha256 is None) != (
            self.decision_provenance is None
        ):
            raise ValueError(
                "collection digest and decision provenance must appear together"
            )
        if self.state.turn >= self.turn_limit:
            raise ValueError("macro transition starts after the horizon")
        root = get(self.state.graph, self.state.graph.root_id)
        if not isinstance(root, Goal) or root.status is not GoalStatus.OPEN:
            raise ValueError("macro transition must start from an open root")
        if requires_internal_closure(self.state):
            raise ValueError("macro transition must start at a stable boundary")
        if self.state.previous_outcome is not None and (
            self.state.previous_outcome.action.verb not in POLICY_VERB_ORDER
        ):
            raise ValueError("state previous outcome is not a learned action")

        legal = available_actions(self.state, self.turn_limit)
        if self.action not in legal:
            raise ValueError("macro transition action is outside the legal menu")
        if type(self.closure_events) is not tuple or any(
            not isinstance(event, ClosureEvent)
            for event in self.closure_events
        ):
            raise ValueError("macro transition closure events are invalid")
        if tuple(event.sequence for event in self.closure_events) != tuple(
            range(len(self.closure_events))
        ):
            raise ValueError("macro transition closure sequence is invalid")
        if self.outcome.code is OutcomeCode.REJECTED:
            if self.closure_events:
                raise ValueError("rejected learned action cannot have closure")
            if self.next_state.graph != self.state.graph:
                raise ValueError("rejected learned action changed the graph")

        if self.outcome.code is OutcomeCode.APPLIED:
            if self.next_state.graph == self.state.graph:
                raise ValueError("applied learned action did not change the graph")
            ClosureResult(self.next_state.graph, self.closure_events)
        if self.next_state.turn != self.state.turn + 1:
            raise ValueError("macro transition must consume one learned turn")
        if self.next_state.previous_outcome != self.outcome:
            raise ValueError("next state must retain the learned action outcome")
        if requires_internal_closure(self.next_state):
            raise ValueError("macro transition next state is not stable")
        if type(self.reward) is not int or self.reward not in {0, 1}:
            raise ValueError("macro transition reward is invalid")
        if type(self.terminal) is not bool:
            raise ValueError("macro transition terminal flag is invalid")

        root_solved = is_solved(
            self.next_state.graph, self.next_state.graph.root_id
        )
        expected_terminal = (
            root_solved or self.next_state.turn >= self.turn_limit
        )
        expected_reward = 1 if root_solved else 0
        if self.terminal is not expected_terminal:
            raise ValueError("macro transition terminal predicate disagrees")
        if self.reward != expected_reward:
            raise ValueError("macro transition reward predicate disagrees")
        next_legal = available_actions(self.next_state, self.turn_limit)
        if self.terminal and next_legal:
            raise ValueError("terminal macro transition has legal actions")
        if not self.terminal and not next_legal:
            raise ValueError("nonterminal macro transition has no legal action")


def validate_macro_transition(
    transition: MacroTransition,
    *,
    environment_contract_sha256: str,
    turn_limit: int | None = None,
    generation_contract: GenerationContract,
    verifier: ClosureVerifier,
    aggregate_builder: AggregatePayloadBuilder | None = None,
) -> None:
    """Replay an applied effect and closure before trusting its endpoint."""
    if not isinstance(transition, MacroTransition):
        raise TypeError("validate_macro_transition needs a MacroTransition")
    if (
        not isinstance(environment_contract_sha256, str)
        or _SHA256_RE.fullmatch(environment_contract_sha256) is None
    ):
        raise TypeError("environment contract digest has an invalid type")
    if (
        transition.environment_contract_sha256
        != environment_contract_sha256
    ):
        raise MacroTransitionError(
            "macro transition belongs to a different environment"
        )
    if turn_limit is not None and transition.turn_limit != turn_limit:
        raise MacroTransitionError(
            "macro transition horizon differs from the environment"
        )
    if not isinstance(generation_contract, GenerationContract):
        raise TypeError("generation_contract has an invalid type")
    if not isinstance(verifier, ClosureVerifier):
        raise TypeError("verifier has an invalid type")
    if aggregate_builder is not None and not isinstance(
        aggregate_builder, AggregatePayloadBuilder
    ):
        raise TypeError("aggregate_builder has an invalid type")
    if transition.outcome.code is OutcomeCode.REJECTED:
        if transition.outcome.reason is RejectionReason.STALE_ACTION:
            raise MacroTransitionError(
                "stale actions cannot form canonical macro transitions"
            )
        if transition.outcome.reason is RejectionReason.ALGEBRA_REJECTED:
            draft = transition.policy_draft
            if draft is None:
                raise MacroTransitionError(
                    "algebra rejection has no typed draft"
                )
            result = _apply_draft(
                transition.state.graph,
                transition.action,
                draft,
                generation_contract,
            )
            if not isinstance(result, Err):
                raise MacroTransitionError(
                    "algebra-rejected draft is accepted during replay"
                )
        return

    try:
        draft = transition.policy_draft
        if draft is None:
            raise MacroTransitionError("applied transition has no typed draft")
        result = _apply_draft(
            transition.state.graph,
            transition.action,
            draft,
            generation_contract,
        )
    except MacroTransitionError:
        raise
    except Exception as exc:
        raise MacroTransitionError(
            f"learned effect reconstruction failed: {exc}"
        ) from exc
    if isinstance(result, Err):
        raise MacroTransitionError(
            f"learned effect replay was rejected: {result.message}"
        )
    canonical_draft, trigger = _infer_applied_draft(
        transition.state.graph, result.graph, transition.action
    )
    if canonical_draft != draft:
        raise MacroTransitionError(
            "stored learned effect is not symbolically canonical"
        )
    try:
        LearnedEffect(
            graph=result.graph,
            outcome=transition.outcome,
            triggering_claim_id=trigger,
            policy_draft=draft,
        )
        closed = run_closure(
            result.graph,
            triggering_claim_id=trigger,
            verifier=verifier,
            generation_contract=generation_contract,
            aggregate_builder=aggregate_builder,
        )
    except Exception as exc:
        raise MacroTransitionError(
            f"deterministic closure replay failed: {exc}"
        ) from exc
    if closed.graph != transition.next_state.graph:
        raise MacroTransitionError(
            "macro next graph differs from deterministic replay"
        )
    if closed.events != transition.closure_events:
        raise MacroTransitionError(
            "macro closure events differ from deterministic replay"
        )


def macro_step(
    state: State,
    action: Action,
    *,
    turn_limit: int,
    environment_contract_sha256: str,
    generator: PolicyDraftGenerator,
    generation_contract: GenerationContract,
    verifier: ClosureVerifier,
    aggregate_builder: AggregatePayloadBuilder | None = None,
) -> MacroTransition:
    """Advance one learned turn, close symbolically, then assign reward."""
    if not isinstance(state, State) or not isinstance(action, Action):
        raise TypeError("macro_step needs a State and Action")
    if type(turn_limit) is not int or turn_limit <= 0:
        raise ValueError("turn_limit must be a positive integer")
    if (
        not isinstance(environment_contract_sha256, str)
        or _SHA256_RE.fullmatch(environment_contract_sha256) is None
    ):
        raise ValueError("environment contract digest is invalid")
    if action.verb not in POLICY_VERB_ORDER:
        raise MacroTransitionError("macro action is not a learned policy verb")
    if not isinstance(generator, PolicyDraftGenerator):
        raise TypeError("generator has an invalid type")
    if not isinstance(generation_contract, GenerationContract):
        raise TypeError("generation_contract has an invalid type")
    if not isinstance(verifier, ClosureVerifier):
        raise TypeError("verifier has an invalid type")
    if aggregate_builder is not None and not isinstance(
        aggregate_builder, AggregatePayloadBuilder
    ):
        raise TypeError("aggregate_builder has an invalid type")
    if state.turn >= turn_limit:
        raise MacroTransitionError("cannot step at or beyond the horizon")
    root = get(state.graph, state.graph.root_id)
    if not isinstance(root, Goal) or root.status is not GoalStatus.OPEN:
        raise MacroTransitionError("cannot step a terminal root")
    if requires_internal_closure(state):
        raise MacroTransitionError("cannot step a closure-pending state")
    if state.previous_outcome is not None and (
        state.previous_outcome.action.verb not in POLICY_VERB_ORDER
    ):
        raise MacroTransitionError(
            "stable policy state carries an internal previous outcome"
        )

    if action not in available_actions(state, turn_limit):
        raise MacroTransitionError(
            "macro action is not in the current learned menu"
        )
    effect = execute_learned_action(
        state,
        action,
        turn_limit=turn_limit,
        generator=generator,
        generation_contract=generation_contract,
    )

    if effect.outcome.code is OutcomeCode.APPLIED:
        closed = run_closure(
            effect.graph,
            triggering_claim_id=effect.triggering_claim_id,
            verifier=verifier,
            generation_contract=generation_contract,
            aggregate_builder=aggregate_builder,
        )
        next_graph = closed.graph
        closure_events = closed.events
    else:
        next_graph = effect.graph
        closure_events = ()

    next_turn = state.turn + 1
    root_solved = is_solved(next_graph, next_graph.root_id)
    terminal = root_solved or next_turn >= turn_limit
    reward = 1 if root_solved else 0
    next_state = State(
        graph=next_graph,
        turn=next_turn,
        previous_outcome=effect.outcome,
    )
    transition = MacroTransition(
        state=state,
        action=action,
        policy_draft=effect.policy_draft,
        outcome=effect.outcome,
        closure_events=closure_events,
        reward=reward,
        next_state=next_state,
        terminal=terminal,
        turn_limit=turn_limit,
        environment_contract_sha256=environment_contract_sha256,
    )
    validate_macro_transition(
        transition,
        environment_contract_sha256=environment_contract_sha256,
        turn_limit=turn_limit,
        generation_contract=generation_contract,
        verifier=verifier,
        aggregate_builder=aggregate_builder,
    )
    return transition


__all__ = [
    "LearnedEffect",
    "MacroTransition",
    "MacroTransitionError",
    "PolicyDraft",
    "PolicyDraftGenerator",
    "PolicyGenerationError",
    "PolicyPayloadError",
    "REWARD_CONTRACT_VERSION",
    "TRANSITION_SEMANTICS_VERSION",
    "execute_learned_action",
    "macro_step",
    "validate_macro_transition",
]
