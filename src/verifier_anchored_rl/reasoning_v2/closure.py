"""Deterministic symbolic closure for the v2 reasoning graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .actions import (
    AggregateDraft,
    Err,
    aggregate,
    commit_claim,
    verify_claim,
)
from .menu import aggregate_ready, requires_internal_closure
from .model import (
    Action,
    AggregatePayload,
    Claim,
    ClaimKind,
    ClaimStatus,
    DischargeStatus,
    Goal,
    Graph,
    NodeId,
    OutcomeCode,
    State,
    TransitionOutcome,
    Verb,
    VerifierContext,
    direct_child_goals,
    get,
    is_active,
    is_committable,
    is_solved,
    validate_graph,
)
from .verifier import (
    ClaimVerifier,
    GenerationContract,
    VerifierReceiptContract,
)


CLOSURE_SEMANTICS_VERSION = "deterministic-claim-closure-v2"
CLOSURE_EVENTS_VERSION = "deterministic-closure-events-v2"
CLOSURE_PRIORITY: tuple[Verb, ...] = (
    Verb.VERIFY,
    Verb.COMMIT,
    Verb.AGGREGATE,
)
CLOSURE_TURN_COST = 0


class ClosureExecutionError(RuntimeError):
    """Raised when a trusted closure operation cannot be completed."""


@runtime_checkable
class ClosureVerifier(
    ClaimVerifier, VerifierReceiptContract, Protocol
):
    """Verifier behavior plus its pinned receipt identity and reason sets."""


@runtime_checkable
class AggregatePayloadBuilder(Protocol):
    """Build a reducer payload from committed exhaustive child outcomes."""

    aggregate_contract_id: str

    def build_aggregate_payload(
        self, graph: Graph, goal: Goal
    ) -> AggregatePayload:
        ...


@dataclass(frozen=True)
class ClosureEvent:
    sequence: int
    claim_id: NodeId
    outcome: TransitionOutcome

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("closure sequence must be nonnegative")
        if type(self.claim_id) is not int or self.claim_id < 0:
            raise ValueError("closure claim id must be nonnegative")
        if not isinstance(self.outcome, TransitionOutcome):
            raise ValueError("closure outcome has an invalid type")
        action = self.outcome.action
        if action.verb not in CLOSURE_PRIORITY:
            raise ValueError("closure event action is not internal")
        if self.outcome.code is not OutcomeCode.APPLIED:
            raise ValueError("closure event outcome must be applied")
        if action.verb is Verb.AGGREGATE:
            if self.claim_id <= action.target:
                raise ValueError(
                    "aggregate event must identify its newly allocated Claim"
                )
        elif self.claim_id != action.target:
            raise ValueError(
                "verify and commit events must identify their target Claim"
            )


def _validate_closure_trace(
    graph: Graph, events: tuple[ClosureEvent, ...]
) -> None:
    seen: set[tuple[Verb, NodeId]] = set()
    created_aggregates: list[NodeId] = []
    phase = Verb.VERIFY
    pending_aggregate: NodeId | None = None

    for event in events:
        action = event.outcome.action
        key = (action.verb, event.claim_id)
        if key in seen:
            raise ValueError("closure trace repeats an immutable operation")
        seen.add(key)
        try:
            claim = get(graph, event.claim_id)
        except KeyError as exc:
            raise ValueError("closure event Claim does not exist") from exc
        if not isinstance(claim, Claim):
            raise ValueError("closure event claim_id is not a Claim")

        if pending_aggregate is not None:
            if not (
                action.verb is Verb.VERIFY
                and action.target == pending_aggregate
                and event.claim_id == pending_aggregate
            ):
                raise ValueError(
                    "new Aggregate must be the next verified Claim"
                )
            pending_aggregate = None
        elif action.verb is Verb.VERIFY and phase is Verb.COMMIT:
            raise ValueError("VERIFY cannot follow COMMIT before AGGREGATE")

        if action.verb is Verb.VERIFY:
            if claim.verification is None:
                raise ValueError("VERIFY event has no stored validity receipt")
            if (
                event.outcome.verifier_context
                != claim.verification.context
            ):
                raise ValueError(
                    "VERIFY event context differs from the stored receipt"
                )
            phase = Verb.VERIFY
        elif action.verb is Verb.COMMIT:
            owner = get(graph, claim.owner_goal)
            if not isinstance(owner, Goal) or owner.committed_claim != claim.id:
                raise ValueError("COMMIT event is not reflected by its owner")
            phase = Verb.COMMIT
        else:
            if (
                claim.kind is not ClaimKind.AGGREGATE
                or claim.owner_goal != action.target
            ):
                raise ValueError(
                    "AGGREGATE event does not identify its created Claim"
                )
            created_aggregates.append(claim.id)
            pending_aggregate = claim.id
            phase = Verb.VERIFY

    if pending_aggregate is not None:
        raise ValueError("closure trace ends before Aggregate verification")
    if created_aggregates:
        expected = tuple(
            range(created_aggregates[0], len(graph.nodes))
        )
        if tuple(created_aggregates) != expected:
            raise ValueError(
                "closure-created Aggregate ids are not the final dense suffix"
            )
        if any(
            (Verb.COMMIT, claim_id) not in seen
            for claim_id in created_aggregates
        ):
            raise ValueError("closure-created Aggregate was not committed")


@dataclass(frozen=True)
class ClosureResult:
    graph: Graph
    events: tuple[ClosureEvent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.graph, Graph):
            raise ValueError("closure result graph has an invalid type")
        validate_graph(self.graph)
        if type(self.events) is not tuple or any(
            not isinstance(event, ClosureEvent) for event in self.events
        ):
            raise ValueError("closure events must be a tuple of ClosureEvent")
        if tuple(event.sequence for event in self.events) != tuple(
            range(len(self.events))
        ):
            raise ValueError("closure event sequences must be contiguous")
        _validate_closure_trace(self.graph, self.events)
        if requires_internal_closure(State(self.graph, turn=0)):
            raise ValueError("closure result is not at a stable boundary")


def _event(
    events: list[ClosureEvent],
    action: Action,
    claim_id: NodeId,
    *,
    verifier_context: VerifierContext | None = None,
) -> None:
    events.append(
        ClosureEvent(
            sequence=len(events),
            claim_id=claim_id,
            outcome=TransitionOutcome(
                action=action,
                code=OutcomeCode.APPLIED,
                verifier_context=verifier_context,
            ),
        )
    )


def _active_unchecked_claims(graph: Graph) -> tuple[NodeId, ...]:
    return tuple(
        node.id
        for node in graph.nodes
        if isinstance(node, Claim)
        and node.status is ClaimStatus.UNCHECKED
        and is_active(graph, node.id)
    )


def _ready_aggregate_goals(graph: Graph) -> tuple[Goal, ...]:
    return tuple(
        node
        for node in graph.nodes
        if isinstance(node, Goal) and aggregate_ready(graph, node)
    )


def _require_trigger(graph: Graph, claim_id: NodeId) -> None:
    try:
        node = get(graph, claim_id)
    except KeyError as exc:
        raise ClosureExecutionError(
            f"triggering Claim {claim_id} does not exist"
        ) from exc
    if (
        not isinstance(node, Claim)
        or node.status is not ClaimStatus.UNCHECKED
        or not is_active(graph, node.id)
    ):
        raise ClosureExecutionError(
            "triggering claim must be active and unchecked"
        )


def _exact_child_claims(graph: Graph, goal: Goal) -> tuple[NodeId, ...]:
    claim_ids: list[NodeId] = []
    for child in direct_child_goals(graph, goal.id):
        if child.committed_claim is None:
            raise ClosureExecutionError(
                f"aggregate-ready Goal {goal.id} has an open child"
            )
        claim_ids.append(child.committed_claim)
    return tuple(claim_ids)


def _bind_receipts_to_verifier(
    claim: Claim, verifier: ClosureVerifier
) -> None:
    verification = claim.verification
    if verification is None:
        raise ClosureExecutionError("VERIFY stored no validity receipt")
    if verification.context.contract_id != verifier.contract_id:
        raise ClosureExecutionError(
            "validity receipt differs from the pinned verifier"
        )
    if (
        not verification.accepted
        and verification.reason_code not in verifier.validity_rejection_codes
    ):
        raise ClosureExecutionError(
            "validity receipt has an unbounded rejection code"
        )
    discharge = claim.discharge
    if discharge is None:
        return
    if discharge.context.contract_id != verifier.contract_id:
        raise ClosureExecutionError(
            "discharge receipt differs from the pinned verifier"
        )
    if (
        not discharge.accepted
        and discharge.reason_code not in verifier.discharge_rejection_codes
    ):
        raise ClosureExecutionError(
            "discharge receipt has an unbounded rejection code"
        )


def run_closure(
    graph: Graph,
    *,
    triggering_claim_id: NodeId | None,
    verifier: ClosureVerifier,
    generation_contract: GenerationContract,
    aggregate_builder: AggregatePayloadBuilder | None = None,
) -> ClosureResult:
    """Run VERIFY, COMMIT, and ready AGGREGATE to a stable boundary."""
    if not isinstance(graph, Graph):
        raise TypeError("run_closure needs a Graph")
    validate_graph(graph)
    if not isinstance(verifier, ClosureVerifier):
        raise TypeError("verifier has an invalid type")
    if not isinstance(generation_contract, GenerationContract):
        raise TypeError("generation_contract has an invalid type")
    if triggering_claim_id is not None:
        if type(triggering_claim_id) is not int or triggering_claim_id < 0:
            raise TypeError("triggering_claim_id must be a node id or None")
        _require_trigger(graph, triggering_claim_id)
    if any(
        isinstance(node, Claim) and is_committable(graph, node.id)
        for node in graph.nodes
    ):
        raise ClosureExecutionError(
            "preverified committable Claim requires transition replay"
        )

    current = graph
    trigger = triggering_claim_id
    generated_aggregates: set[NodeId] = set()
    events: list[ClosureEvent] = []

    while True:
        unchecked = _active_unchecked_claims(current)
        if trigger is not None:
            if trigger not in unchecked:
                raise ClosureExecutionError(
                    "triggering Claim disappeared before verification"
                )
            unchecked = (trigger,) + tuple(
                claim_id for claim_id in unchecked if claim_id != trigger
            )

        for claim_id in unchecked:
            result = verify_claim(current, claim_id, verifier)
            if isinstance(result, Err):
                raise ClosureExecutionError(
                    f"VERIFY({claim_id}) failed: {result.message}"
                )
            current = result.graph
            checked = get(current, claim_id)
            if not isinstance(checked, Claim) or checked.verification is None:
                raise ClosureExecutionError(
                    f"VERIFY({claim_id}) did not store a validity receipt"
                )
            _bind_receipts_to_verifier(checked, verifier)
            _event(
                events,
                Action(Verb.VERIFY, claim_id),
                claim_id,
                verifier_context=checked.verification.context,
            )
            if claim_id in generated_aggregates and (
                checked.status is not ClaimStatus.VERIFIED
                or checked.discharge_status is not DischargeStatus.ACCEPTED
            ):
                raise ClosureExecutionError(
                    "deterministic Aggregate failed symbolic discharge"
                )
        trigger = None

        committable = tuple(
            node.id
            for node in current.nodes
            if isinstance(node, Claim) and is_committable(current, node.id)
        )
        for claim_id in committable:
            if not is_committable(current, claim_id):
                continue
            result = commit_claim(current, claim_id)
            if isinstance(result, Err):
                raise ClosureExecutionError(
                    f"COMMIT({claim_id}) failed: {result.message}"
                )
            current = result.graph
            _event(
                events,
                Action(Verb.COMMIT, claim_id),
                claim_id,
            )
            if is_solved(current, current.root_id):
                return ClosureResult(current, tuple(events))

        ready = _ready_aggregate_goals(current)
        if not ready:
            return ClosureResult(current, tuple(events))
        goal = ready[0]
        if aggregate_builder is None or not isinstance(
            aggregate_builder, AggregatePayloadBuilder
        ):
            raise ClosureExecutionError(
                "ready Aggregate requires a deterministic payload builder"
            )
        try:
            payload = aggregate_builder.build_aggregate_payload(current, goal)
        except Exception as exc:
            raise ClosureExecutionError(
                f"Aggregate payload construction failed: {exc}"
            ) from exc
        if type(payload) is not AggregatePayload:
            raise ClosureExecutionError(
                "aggregate payload builder returned an invalid type"
            )
        child_claims = _exact_child_claims(current, goal)
        before = len(current.nodes)
        result = aggregate(
            current,
            goal.id,
            AggregateDraft(payload.reducer, child_claims, payload.result),
            generation_contract,
        )
        if isinstance(result, Err):
            raise ClosureExecutionError(
                f"AGGREGATE({goal.id}) failed: {result.message}"
            )
        current = result.graph
        if len(current.nodes) != before + 1:
            raise ClosureExecutionError(
                "AGGREGATE did not allocate exactly one Claim"
            )
        aggregate_id = before
        created = get(current, aggregate_id)
        if (
            not isinstance(created, Claim)
            or created.kind is not ClaimKind.AGGREGATE
            or created.owner_goal != goal.id
        ):
            raise ClosureExecutionError(
                "AGGREGATE allocated the wrong Claim"
            )
        _event(
            events,
            Action(Verb.AGGREGATE, goal.id),
            aggregate_id,
        )
        generated_aggregates.add(aggregate_id)
        trigger = aggregate_id


__all__ = [
    "AggregatePayloadBuilder",
    "CLOSURE_EVENTS_VERSION",
    "CLOSURE_PRIORITY",
    "CLOSURE_SEMANTICS_VERSION",
    "CLOSURE_TURN_COST",
    "ClosureEvent",
    "ClosureExecutionError",
    "ClosureResult",
    "ClosureVerifier",
    "run_closure",
]
