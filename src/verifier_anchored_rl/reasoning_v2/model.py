"""Typed immutable state for the claim-targeted v2 reasoning contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Callable, TypeAlias

if TYPE_CHECKING:
    from .verifier import GenerationContract


NodeId: TypeAlias = int


class Connective(str, Enum):
    CONJUNCTIVE = "conjunctive"
    EXHAUSTIVE_CASE = "exhaustive_case"


class ClaimKind(str, Enum):
    SOLUTION = "solution"
    RESULT = "result"
    DERIVATION = "derivation"
    AGGREGATE = "aggregate"
    REFUTATION = "refutation"


class GoalStatus(str, Enum):
    OPEN = "open"
    SOLVED = "solved"
    PRUNED = "pruned"


class ClaimStatus(str, Enum):
    UNCHECKED = "unchecked"
    VERIFIED = "verified"
    REJECTED = "rejected"
    PRUNED = "pruned"


class DischargeStatus(str, Enum):
    UNCHECKED = "unchecked"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class CheckPurpose(str, Enum):
    VALIDITY = "validity"
    OWNER_DISCHARGE = "owner_discharge"


class Verb(str, Enum):
    DECOMPOSE = "decompose"
    PROPOSE_RESULT = "propose_result"
    PROPOSE_SOLUTION = "propose_solution"
    DERIVE = "derive"
    AGGREGATE = "aggregate"
    PROPOSE_REFUTATION = "propose_refutation"
    VERIFY = "verify"
    COMMIT = "commit"
    PRUNE = "prune"


class GoalObjective(str, Enum):
    BRANCH = "branch"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    IDENTITY = "identity"


class Relation(str, Enum):
    EQ = "eq"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"


class Reducer(str, Enum):
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    IDENTITY = "identity"


class RefutationMode(str, Enum):
    UNIQUE_CANDIDATE_OUTSIDE_DOMAIN = "unique_candidate_outside_domain"


@dataclass(frozen=True)
class GoalObligation:
    """Trusted canonical inputs for symbolic owner-discharge checks.

    The envelope intentionally has no answer or solution-set field.
    Domain adapters interpret the canonical strings outside this stdlib core.
    """

    variable: str
    equation: str
    domain: str
    objective: GoalObjective


@dataclass(frozen=True)
class SolutionPayload:
    variable: str
    value: str


@dataclass(frozen=True)
class ResultPayload:
    objective: GoalObjective
    value: str


@dataclass(frozen=True)
class DerivationPayload:
    lhs: str
    relation: Relation
    rhs: str


@dataclass(frozen=True)
class AggregatePayload:
    reducer: Reducer
    result: str


@dataclass(frozen=True)
class RefutationPayload:
    mode: RefutationMode
    variable: str
    candidate: str


ClaimPayload: TypeAlias = (
    SolutionPayload
    | ResultPayload
    | DerivationPayload
    | AggregatePayload
    | RefutationPayload
)


@dataclass(frozen=True)
class VerifierContext:
    contract_id: str
    purpose: CheckPurpose
    claim_kind: ClaimKind
    owner_goal: NodeId
    context_digest: str


@dataclass(frozen=True)
class SymbolicCheck:
    context: VerifierContext
    accepted: bool
    observed: str
    reason_code: str | None = None


@dataclass(frozen=True)
class Goal:
    id: NodeId
    parent_goal: NodeId | None
    statement: str
    obligation: GoalObligation
    assumptions: tuple[str, ...] = ()
    connective: Connective | None = None
    committed_claim: NodeId | None = None
    pruned: bool = False

    @property
    def status(self) -> GoalStatus:
        if self.pruned:
            return GoalStatus.PRUNED
        if self.committed_claim is not None:
            return GoalStatus.SOLVED
        return GoalStatus.OPEN


@dataclass(frozen=True)
class Evidence:
    id: NodeId
    scope_goal: NodeId
    statement: str
    premises: tuple[NodeId, ...] = ()
    source: str = "trusted_given"

    @property
    def committable(self) -> bool:
        return False


@dataclass(frozen=True)
class Claim:
    id: NodeId
    owner_goal: NodeId
    kind: ClaimKind
    payload: ClaimPayload
    premises: tuple[NodeId, ...] = ()
    verification: SymbolicCheck | None = None
    discharge: SymbolicCheck | None = None
    pruned: bool = False

    @property
    def statement(self) -> str:
        return claim_statement(self.payload)

    @property
    def status(self) -> ClaimStatus:
        if self.pruned:
            return ClaimStatus.PRUNED
        if self.verification is None:
            return ClaimStatus.UNCHECKED
        if self.verification.accepted:
            return ClaimStatus.VERIFIED
        return ClaimStatus.REJECTED

    @property
    def discharge_status(self) -> DischargeStatus:
        if self.discharge is None:
            return DischargeStatus.UNCHECKED
        if self.discharge.accepted:
            return DischargeStatus.ACCEPTED
        return DischargeStatus.REJECTED


Node: TypeAlias = Goal | Evidence | Claim


@dataclass(frozen=True)
class Graph:
    nodes: tuple[Node, ...]
    root_id: NodeId


@dataclass(frozen=True)
class Action:
    verb: Verb
    target: NodeId

    def __post_init__(self) -> None:
        if not isinstance(self.verb, Verb):
            raise ValueError("action verb has an invalid type")
        if type(self.target) is not int or self.target < 0:
            raise ValueError("action target must be a nonnegative node id")


class OutcomeCode(str, Enum):
    APPLIED = "applied"
    REJECTED = "rejected"


class RejectionReason(str, Enum):
    STALE_ACTION = "stale_action"
    GENERATION_ERROR = "generation_error"
    PAYLOAD_ERROR = "payload_error"
    ALGEBRA_REJECTED = "algebra_rejected"


@dataclass(frozen=True)
class TransitionOutcome:
    action: Action
    code: OutcomeCode
    reason: RejectionReason | None = None
    verifier_context: VerifierContext | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, Action):
            raise ValueError("transition outcome action has an invalid type")
        if not isinstance(self.code, OutcomeCode):
            raise ValueError("transition outcome code has an invalid type")
        if self.reason is not None and not isinstance(
            self.reason, RejectionReason
        ):
            raise ValueError("transition rejection reason has an invalid type")
        if self.verifier_context is not None and not isinstance(
            self.verifier_context, VerifierContext
        ):
            raise ValueError("transition verifier context has an invalid type")
        if self.code is OutcomeCode.APPLIED and self.reason is not None:
            raise ValueError("an applied outcome cannot have a rejection reason")
        if self.code is OutcomeCode.REJECTED and self.reason is None:
            raise ValueError("a rejected outcome requires a rejection reason")
        has_context = self.verifier_context is not None
        is_applied_verify = (
            self.code is OutcomeCode.APPLIED
            and self.action.verb is Verb.VERIFY
        )
        if has_context != is_applied_verify:
            raise ValueError(
                "verifier context is required exactly for applied VERIFY"
            )
        if has_context and (
            self.verifier_context is None
            or self.verifier_context.purpose is not CheckPurpose.VALIDITY
        ):
            raise ValueError("transition verifier context must be validity")


@dataclass(frozen=True)
class State:
    graph: Graph
    turn: int
    previous_outcome: TransitionOutcome | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.graph, Graph):
            raise ValueError("state graph has an invalid type")
        validate_graph(self.graph)
        if type(self.turn) is not int or self.turn < 0:
            raise ValueError("turn must be nonnegative")
        if self.previous_outcome is not None and not isinstance(
            self.previous_outcome, TransitionOutcome
        ):
            raise ValueError("state previous outcome has an invalid type")


class GraphValidationError(ValueError):
    """Raised when a Graph violates the v2 semantic contract."""


_PAYLOAD_BY_KIND: dict[ClaimKind, type[ClaimPayload]] = {
    ClaimKind.SOLUTION: SolutionPayload,
    ClaimKind.RESULT: ResultPayload,
    ClaimKind.DERIVATION: DerivationPayload,
    ClaimKind.AGGREGATE: AggregatePayload,
    ClaimKind.REFUTATION: RefutationPayload,
}


def get(graph: Graph, node_id: NodeId) -> Node:
    if type(node_id) is not int or not 0 <= node_id < len(graph.nodes):
        raise KeyError(f"no node with id {node_id!r}")
    return graph.nodes[node_id]


def _append(
    graph: Graph, make_node: Callable[[NodeId], Node]
) -> tuple[Graph, NodeId]:
    node_id = len(graph.nodes)
    node = make_node(node_id)
    if node.id != node_id:
        raise GraphValidationError("appended node id is not dense")
    return replace(graph, nodes=graph.nodes + (node,)), node_id


def _replace(graph: Graph, node: Node) -> Graph:
    if not 0 <= node.id < len(graph.nodes):
        raise GraphValidationError(f"cannot replace missing node {node.id}")
    nodes = graph.nodes[:node.id] + (node,) + graph.nodes[node.id + 1 :]
    return replace(graph, nodes=nodes)


def _require_text(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise GraphValidationError(f"{field} must be nonempty text")


def _require_bool(value: bool, field: str) -> None:
    if type(value) is not bool:
        raise GraphValidationError(f"{field} must be a bool")


def _require_node_id(value: NodeId, field: str) -> None:
    if type(value) is not int or value < 0:
        raise GraphValidationError(f"{field} must be a nonnegative node id")


def _require_tuple(value: tuple[object, ...], field: str) -> None:
    if not isinstance(value, tuple):
        raise GraphValidationError(f"{field} must be a tuple")


def _validate_obligation(obligation: GoalObligation) -> None:
    if not isinstance(obligation, GoalObligation):
        raise GraphValidationError("goal obligation has an invalid type")
    _require_text(obligation.variable, "goal obligation variable")
    _require_text(obligation.equation, "goal obligation equation")
    _require_text(obligation.domain, "goal obligation domain")
    if not isinstance(obligation.objective, GoalObjective):
        raise GraphValidationError("goal obligation objective has an invalid type")


def claim_statement(payload: ClaimPayload) -> str:
    """Derive presentation text without introducing a second authored field."""
    if isinstance(payload, SolutionPayload):
        return f"{payload.variable} = {payload.value}"
    if isinstance(payload, ResultPayload):
        return f"result({payload.objective.value}) = {payload.value}"
    if isinstance(payload, DerivationPayload):
        symbols = {
            Relation.EQ: "=",
            Relation.LT: "<",
            Relation.LE: "<=",
            Relation.GT: ">",
            Relation.GE: ">=",
        }
        return f"{payload.lhs} {symbols[payload.relation]} {payload.rhs}"
    if isinstance(payload, AggregatePayload):
        return (
            f"{payload.reducer.value}(committed child outcomes) = "
            f"{payload.result}"
        )
    if isinstance(payload, RefutationPayload):
        return (
            f"{payload.variable} = {payload.candidate} is outside the owner domain"
        )
    raise TypeError(f"unsupported claim payload {type(payload).__name__}")


def _validate_check(check: SymbolicCheck, claim: Claim, purpose: CheckPurpose) -> None:
    if not isinstance(check, SymbolicCheck):
        raise GraphValidationError(f"claim {claim.id} check has an invalid type")
    context = check.context
    if not isinstance(context, VerifierContext):
        raise GraphValidationError(
            f"claim {claim.id} verifier context has an invalid type"
        )
    if context.purpose is not purpose:
        raise GraphValidationError(f"claim {claim.id} has the wrong check purpose")
    if context.claim_kind is not claim.kind or context.owner_goal != claim.owner_goal:
        raise GraphValidationError(f"claim {claim.id} has a mismatched verifier context")
    _require_text(context.contract_id, "verifier contract id")
    _require_text(context.context_digest, "verifier context digest")
    _require_bool(check.accepted, "symbolic check accepted")
    _require_text(check.observed, "symbolic check observed")
    if check.accepted and check.reason_code is not None:
        raise GraphValidationError("an accepted symbolic check cannot have a reason code")
    if not check.accepted:
        _require_text(check.reason_code or "", "rejected check reason code")


def _validate_payload(graph: Graph, claim: Claim) -> None:
    if not isinstance(claim.kind, ClaimKind):
        raise GraphValidationError("claim kind has an invalid type")
    expected_type = _PAYLOAD_BY_KIND[claim.kind]
    if not isinstance(claim.payload, expected_type):
        raise GraphValidationError(
            f"claim {claim.id} kind {claim.kind.value} has payload "
            f"{type(claim.payload).__name__}"
        )
    payload = claim.payload
    owner = get(graph, claim.owner_goal)
    if not isinstance(owner, Goal):
        raise GraphValidationError(f"claim {claim.id} owner is not a Goal")
    if isinstance(payload, SolutionPayload):
        _require_text(payload.variable, "solution variable")
        _require_text(payload.value, "solution value")
        if payload.variable != owner.obligation.variable:
            raise GraphValidationError(
                "solution variable differs from the owner obligation"
            )
    elif isinstance(payload, ResultPayload):
        _require_text(payload.value, "result value")
        if not isinstance(payload.objective, GoalObjective):
            raise GraphValidationError("result objective has an invalid type")
        if payload.objective is GoalObjective.BRANCH:
            raise GraphValidationError("branch goals cannot own result claims")
        if payload.objective is not owner.obligation.objective:
            raise GraphValidationError(
                "result objective differs from the owner obligation"
            )
    elif isinstance(payload, DerivationPayload):
        _require_text(payload.lhs, "derivation left side")
        _require_text(payload.rhs, "derivation right side")
        if not isinstance(payload.relation, Relation):
            raise GraphValidationError("derivation relation has an invalid type")
    elif isinstance(payload, AggregatePayload):
        _require_text(payload.result, "aggregate result")
        if not isinstance(payload.reducer, Reducer):
            raise GraphValidationError("aggregate reducer has an invalid type")
        expected_reducer = {
            GoalObjective.SUM: Reducer.SUM,
            GoalObjective.MAX: Reducer.MAX,
            GoalObjective.MIN: Reducer.MIN,
            GoalObjective.IDENTITY: Reducer.IDENTITY,
        }.get(owner.obligation.objective)
        if payload.reducer is not expected_reducer:
            raise GraphValidationError(
                "aggregate reducer differs from the owner objective"
            )
        owner_children = direct_child_goals(graph, claim.owner_goal)
        if not owner_children:
            raise GraphValidationError("aggregate owner has no child goals")
        expected = tuple(child.committed_claim for child in owner_children)
        if any(item is None for item in expected) or claim.premises != expected:
            raise GraphValidationError(
                "aggregate premises must be the exact committed child outcomes"
            )
    elif isinstance(payload, RefutationPayload):
        if not isinstance(payload.mode, RefutationMode):
            raise GraphValidationError("refutation mode has an invalid type")
        _require_text(payload.variable, "refutation variable")
        _require_text(payload.candidate, "refutation candidate")
        if payload.variable != owner.obligation.variable:
            raise GraphValidationError(
                "refutation variable differs from the owner obligation"
            )


def validate_graph(graph: Graph) -> None:
    """Raise GraphValidationError unless all structural invariants hold."""
    if not isinstance(graph, Graph):
        raise GraphValidationError("graph has an invalid type")
    if not isinstance(graph.nodes, tuple) or not graph.nodes:
        raise GraphValidationError("graph needs a nonempty node tuple")
    if type(graph.root_id) is not int or not 0 <= graph.root_id < len(graph.nodes):
        raise GraphValidationError("graph root id is missing")
    root = graph.nodes[graph.root_id]
    if not isinstance(root, Goal) or root.parent_goal is not None:
        raise GraphValidationError("graph root must be a parentless Goal")
    if root.pruned:
        raise GraphValidationError("root goal cannot be pruned")

    for index, node in enumerate(graph.nodes):
        if type(node.id) is not int or node.id != index:
            raise GraphValidationError("node ids must be dense tuple positions")
        if isinstance(node, Goal):
            _require_text(node.statement, "goal statement")
            _validate_obligation(node.obligation)
            _require_tuple(node.assumptions, "goal assumptions")
            _require_bool(node.pruned, "goal pruned")
            if node.connective is not None and not isinstance(
                node.connective, Connective
            ):
                raise GraphValidationError("goal connective has an invalid type")
            for assumption in node.assumptions:
                _require_text(assumption, "goal assumption")
            if node.id == graph.root_id:
                if node.parent_goal is not None:
                    raise GraphValidationError("root goal cannot have a parent")
            else:
                if node.parent_goal is None:
                    raise GraphValidationError(
                        f"goal {node.id} needs an earlier parent goal"
                    )
                _require_node_id(node.parent_goal, f"goal {node.id} parent")
                if node.parent_goal >= node.id:
                    raise GraphValidationError(
                        f"goal {node.id} needs an earlier parent goal"
                    )
                if not isinstance(get(graph, node.parent_goal), Goal):
                    raise GraphValidationError(f"goal {node.id} parent is not a Goal")
            if node.pruned and node.committed_claim is not None:
                raise GraphValidationError("a pruned goal cannot retain a commit")
            if node.committed_claim is not None:
                _require_node_id(
                    node.committed_claim, f"goal {node.id} committed claim"
                )
        elif isinstance(node, Evidence):
            _require_text(node.statement, "evidence statement")
            _require_tuple(node.premises, "evidence premises")
            if node.source != "trusted_given":
                raise GraphValidationError("Evidence source must be trusted_given")
            _require_node_id(node.scope_goal, f"evidence {node.id} scope")
            if node.scope_goal >= node.id or not isinstance(
                get(graph, node.scope_goal), Goal
            ):
                raise GraphValidationError(
                    f"evidence {node.id} needs an earlier scope goal"
                )
        elif isinstance(node, Claim):
            _require_node_id(node.owner_goal, f"claim {node.id} owner")
            _require_tuple(node.premises, "claim premises")
            _require_bool(node.pruned, "claim pruned")
            if node.owner_goal >= node.id or not isinstance(
                get(graph, node.owner_goal), Goal
            ):
                raise GraphValidationError(
                    f"claim {node.id} needs an earlier owner goal"
                )
            _validate_payload(graph, node)
            _require_text(node.statement, "derived claim statement")
            if node.verification is not None:
                _validate_check(node.verification, node, CheckPurpose.VALIDITY)
            if node.discharge is not None:
                if node.verification is None:
                    raise GraphValidationError(
                        "owner discharge requires a claim verification result"
                    )
                _validate_check(node.discharge, node, CheckPurpose.OWNER_DISCHARGE)
                if not node.verification.accepted:
                    raise GraphValidationError(
                        "rejected validity cannot have an owner discharge result"
                    )
                if (
                    node.verification.context.contract_id
                    != node.discharge.context.contract_id
                ):
                    raise GraphValidationError(
                        "validity and discharge verifier contracts differ"
                    )
            if (
                node.verification is not None
                and node.verification.accepted
                and node.discharge is None
            ):
                raise GraphValidationError(
                    "accepted validity requires an owner discharge result"
                )
        else:
            raise GraphValidationError(f"unsupported node type {type(node).__name__}")

        premises = getattr(node, "premises", ())
        if len(premises) != len(set(premises)):
            raise GraphValidationError(f"node {node.id} has duplicate premises")
        for premise_id in premises:
            _require_node_id(premise_id, f"node {node.id} premise")
            if premise_id >= node.id:
                raise GraphValidationError(
                    f"node {node.id} premise {premise_id} is not earlier"
                )
            if not isinstance(get(graph, premise_id), (Evidence, Claim)):
                raise GraphValidationError(
                    f"node {node.id} premise {premise_id} is not Evidence or Claim"
                )
        if isinstance(node, Claim) and node.kind in {
            ClaimKind.SOLUTION,
            ClaimKind.RESULT,
            ClaimKind.REFUTATION,
        } and node.premises:
            raise GraphValidationError(
                f"{node.kind.value} claim premises must be empty"
            )

    for node in graph.nodes:
        if not isinstance(node, Claim):
            continue
        if node.kind is ClaimKind.AGGREGATE:
            owner = get(graph, node.owner_goal)
            children = direct_child_goals(graph, node.owner_goal)
            if (
                not isinstance(owner, Goal)
                or owner.connective is not Connective.EXHAUSTIVE_CASE
                or not children
            ):
                raise GraphValidationError(
                    f"aggregate claim {node.id} needs exhaustive child goals"
                )
            expected_claims = tuple(child.committed_claim for child in children)
            if any(claim_id is None for claim_id in expected_claims):
                raise GraphValidationError(
                    f"aggregate claim {node.id} needs committed child goals"
                )
            if node.premises != expected_claims:
                raise GraphValidationError(
                    f"aggregate claim {node.id} premises differ from "
                    "committed child outcomes"
                )
            continue
        ancestry = {
            goal.id for goal in goal_ancestry(graph, node.owner_goal)
        }
        for premise_id in node.premises:
            premise = get(graph, premise_id)
            scope_goal = (
                premise.scope_goal
                if isinstance(premise, Evidence)
                else premise.owner_goal
            )
            if scope_goal not in ancestry:
                raise GraphValidationError(
                    f"claim {node.id} premise {premise_id} is outside "
                    "its owner ancestry"
                )
            if is_active(graph, node.id) and not is_active(graph, premise_id):
                raise GraphValidationError(
                    f"active claim {node.id} premise {premise_id} is not active"
                )

    for node in graph.nodes:
        if not isinstance(node, Goal):
            continue
        children = direct_child_goals(graph, node.id)
        if bool(children) != (node.connective is not None):
            raise GraphValidationError(
                f"goal {node.id} connective and children disagree"
            )
        for child in children:
            if child.obligation.objective is not GoalObjective.BRANCH:
                raise GraphValidationError(
                    f"child goal {child.id} objective must be branch"
                )
            if child.obligation.variable != node.obligation.variable:
                raise GraphValidationError(
                    f"child goal {child.id} variable differs from its parent"
                )
        if node.committed_claim is None:
            continue
        committed = get(graph, node.committed_claim)
        if not isinstance(committed, Claim) or committed.owner_goal != node.id:
            raise GraphValidationError(
                f"goal {node.id} committed claim is not owner-scoped"
            )
        if (
            committed.status is not ClaimStatus.VERIFIED
            or committed.discharge_status is not DischargeStatus.ACCEPTED
        ):
            raise GraphValidationError(
                f"goal {node.id} committed claim is not verified and discharging"
            )


def new_graph(
    problem: str,
    obligation: GoalObligation,
    givens: tuple[str, ...] = (),
    *,
    generation_contract: GenerationContract,
) -> Graph:
    from .verifier import GenerationContract

    _require_text(problem, "problem")
    if not isinstance(generation_contract, GenerationContract):
        raise ValueError("generation_contract has an invalid type")
    try:
        canonical_obligation = generation_contract.canonicalize_obligation(
            obligation
        )
    except Exception as exc:
        raise ValueError(f"root obligation rejected: {exc}") from exc
    if not isinstance(canonical_obligation, GoalObligation):
        raise ValueError("root obligation canonicalizer returned an invalid type")
    _validate_obligation(canonical_obligation)
    root = Goal(
        id=0,
        parent_goal=None,
        statement=problem,
        obligation=canonical_obligation,
    )
    nodes: list[Node] = [root]
    for statement in givens:
        _require_text(statement, "given")
        nodes.append(
            Evidence(id=len(nodes), scope_goal=root.id, statement=statement)
        )
    graph = Graph(nodes=tuple(nodes), root_id=root.id)
    validate_graph(graph)
    return graph


def direct_child_goals(graph: Graph, goal_id: NodeId) -> tuple[Goal, ...]:
    goal = get(graph, goal_id)
    if not isinstance(goal, Goal):
        raise TypeError(f"node {goal_id} is not a Goal")
    return tuple(
        node
        for node in graph.nodes
        if isinstance(node, Goal) and node.parent_goal == goal_id
    )


def owned_claims(graph: Graph, goal_id: NodeId) -> tuple[Claim, ...]:
    goal = get(graph, goal_id)
    if not isinstance(goal, Goal):
        raise TypeError(f"node {goal_id} is not a Goal")
    return tuple(
        node
        for node in graph.nodes
        if isinstance(node, Claim) and node.owner_goal == goal_id
    )


def goal_ancestry(graph: Graph, goal_id: NodeId) -> tuple[Goal, ...]:
    node = get(graph, goal_id)
    if not isinstance(node, Goal):
        raise TypeError(f"node {goal_id} is not a Goal")
    ancestry = []
    current = node
    while True:
        ancestry.append(current)
        if current.parent_goal is None:
            break
        parent = get(graph, current.parent_goal)
        if not isinstance(parent, Goal):
            raise GraphValidationError(f"goal {current.id} parent is not a Goal")
        current = parent
    return tuple(reversed(ancestry))


def owner_goal_id(graph: Graph, node_id: NodeId) -> NodeId:
    node = get(graph, node_id)
    if isinstance(node, Goal):
        return node.id
    if isinstance(node, Evidence):
        return node.scope_goal
    return node.owner_goal


def is_effectively_pruned(graph: Graph, node_id: NodeId) -> bool:
    node = get(graph, node_id)
    if isinstance(node, Claim) and node.pruned:
        return True
    owner = owner_goal_id(graph, node_id)
    return any(goal.pruned for goal in goal_ancestry(graph, owner))


def is_active(graph: Graph, node_id: NodeId) -> bool:
    node = get(graph, node_id)
    if isinstance(node, Claim) and node.pruned:
        return False
    owner = owner_goal_id(graph, node_id)
    return all(
        goal.status is GoalStatus.OPEN for goal in goal_ancestry(graph, owner)
    )


def is_solved(graph: Graph, goal_id: NodeId) -> bool:
    node = get(graph, goal_id)
    return isinstance(node, Goal) and node.status is GoalStatus.SOLVED


def is_committable(graph: Graph, claim_id: NodeId) -> bool:
    node = get(graph, claim_id)
    return (
        isinstance(node, Claim)
        and node.status is ClaimStatus.VERIFIED
        and node.discharge_status is DischargeStatus.ACCEPTED
        and is_active(graph, claim_id)
    )


__all__ = [
    "Action",
    "AggregatePayload",
    "CheckPurpose",
    "Claim",
    "ClaimKind",
    "ClaimPayload",
    "ClaimStatus",
    "Connective",
    "DerivationPayload",
    "DischargeStatus",
    "Evidence",
    "Goal",
    "GoalObjective",
    "GoalObligation",
    "GoalStatus",
    "Graph",
    "GraphValidationError",
    "Node",
    "NodeId",
    "OutcomeCode",
    "RefutationPayload",
    "RefutationMode",
    "ResultPayload",
    "Reducer",
    "Relation",
    "RejectionReason",
    "SolutionPayload",
    "State",
    "SymbolicCheck",
    "TransitionOutcome",
    "Verb",
    "VerifierContext",
    "claim_statement",
    "direct_child_goals",
    "get",
    "goal_ancestry",
    "is_active",
    "is_committable",
    "is_effectively_pruned",
    "is_solved",
    "new_graph",
    "owned_claims",
    "owner_goal_id",
    "validate_graph",
]
