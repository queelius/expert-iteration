"""Pure claim-targeted action algebra for the v2 reasoning graph."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .model import (
    AggregatePayload,
    Claim,
    ClaimKind,
    ClaimStatus,
    Connective,
    DerivationPayload,
    Evidence,
    Goal,
    GoalObjective,
    GoalObligation,
    GoalStatus,
    Graph,
    GraphValidationError,
    NodeId,
    Reducer,
    RefutationPayload,
    RefutationMode,
    Relation,
    ResultPayload,
    SolutionPayload,
    _append,
    _replace,
    direct_child_goals,
    get,
    goal_ancestry,
    is_active,
    is_committable,
    owned_claims,
    validate_graph,
)
from .verifier import ClaimVerifier, GenerationContract, VerificationDecision


@dataclass(frozen=True)
class Ok:
    graph: Graph


@dataclass(frozen=True)
class Err:
    message: str


Result = Ok | Err


_PAYLOAD_TYPES = {
    ClaimKind.SOLUTION: SolutionPayload,
    ClaimKind.RESULT: ResultPayload,
    ClaimKind.DERIVATION: DerivationPayload,
    ClaimKind.AGGREGATE: AggregatePayload,
    ClaimKind.REFUTATION: RefutationPayload,
}


@dataclass(frozen=True)
class DecompositionDraft:
    connective: Connective
    child_statements: tuple[str, ...]
    child_obligations: tuple[GoalObligation, ...]


@dataclass(frozen=True)
class SolutionDraft:
    variable: str
    value: str


@dataclass(frozen=True)
class ResultDraft:
    objective: GoalObjective
    value: str


@dataclass(frozen=True)
class DerivationDraft:
    lhs: str
    relation: Relation
    rhs: str
    premises: tuple[NodeId, ...] = ()


@dataclass(frozen=True)
class AggregateDraft:
    reducer: Reducer
    child_claims: tuple[NodeId, ...]
    result: str


@dataclass(frozen=True)
class RefutationDraft:
    mode: RefutationMode
    variable: str
    candidate: str


def _input_error(graph: Graph) -> Err | None:
    try:
        validate_graph(graph)
    except (GraphValidationError, KeyError, TypeError, AttributeError) as exc:
        return Err(f"invalid input graph: {exc}")
    return None


def _finish(graph: Graph) -> Result:
    try:
        validate_graph(graph)
    except (GraphValidationError, KeyError, TypeError, AttributeError) as exc:
        return Err(f"action would violate graph invariants: {exc}")
    return Ok(graph)


def _open_goal(graph: Graph, goal_id: NodeId) -> Goal | Err:
    try:
        node = get(graph, goal_id)
    except KeyError:
        return Err(f"no node with id {goal_id}")
    if not isinstance(node, Goal):
        return Err(f"node {goal_id} is not a Goal")
    if node.status is not GoalStatus.OPEN or not is_active(graph, goal_id):
        return Err(f"goal {goal_id} is not active and open")
    return node


def _claim_premises_error(
    graph: Graph,
    claim_id: NodeId,
    owner_ancestry: frozenset[NodeId] | None,
) -> str | None:
    try:
        node = get(graph, claim_id)
    except KeyError:
        return f"no premise with id {claim_id}"
    if not isinstance(node, (Evidence, Claim)):
        return f"premise {claim_id} is not Evidence or Claim"
    if owner_ancestry is None:
        return None
    scope_goal = (
        node.scope_goal if isinstance(node, Evidence) else node.owner_goal
    )
    if scope_goal not in owner_ancestry:
        return f"premise {claim_id} is outside the owner ancestry"
    if not is_active(graph, claim_id):
        return f"premise {claim_id} is not active"
    return None


def _add_claim(
    graph: Graph,
    goal_id: NodeId,
    kind: ClaimKind,
    payload,
    premises: tuple[NodeId, ...],
    generation_contract: GenerationContract,
) -> Result:
    invalid = _input_error(graph)
    if invalid is not None:
        return invalid
    goal = _open_goal(graph, goal_id)
    if isinstance(goal, Err):
        return goal
    if not isinstance(generation_contract, GenerationContract):
        return Err("generation_contract has an invalid type")
    try:
        canonical_payload = generation_contract.canonicalize_payload(
            goal.obligation, kind, payload
        )
    except Exception as exc:
        return Err(f"claim payload rejected: {exc}")
    expected_payload = _PAYLOAD_TYPES.get(kind)
    if expected_payload is None or not isinstance(
        canonical_payload, expected_payload
    ):
        return Err("claim payload canonicalizer returned an invalid type")
    payload = canonical_payload
    if type(premises) is not tuple or any(
        type(premise_id) is not int or premise_id < 0
        for premise_id in premises
    ):
        return Err("claim premises must be a tuple of nonnegative node ids")
    if len(premises) != len(set(premises)):
        return Err("claim premises must be unique")
    owner_ancestry = (
        None
        if kind is ClaimKind.AGGREGATE
        else frozenset(item.id for item in goal_ancestry(graph, goal.id))
    )
    for premise_id in premises:
        message = _claim_premises_error(
            graph, premise_id, owner_ancestry
        )
        if message is not None:
            return Err(message)
    if any(
        claim.kind is kind
        and claim.payload == payload
        and claim.premises == premises
        for claim in owned_claims(graph, goal.id)
    ):
        return Err("duplicate claim")
    try:
        updated, _ = _append(
            graph,
            lambda node_id: Claim(
                id=node_id,
                owner_goal=goal.id,
                kind=kind,
                payload=payload,
                premises=premises,
            ),
        )
    except (GraphValidationError, KeyError, TypeError, ValueError) as exc:
        return Err(f"cannot create claim: {exc}")
    return _finish(updated)


def decompose(
    graph: Graph,
    goal_id: NodeId,
    draft: DecompositionDraft,
    generation_contract: GenerationContract,
) -> Result:
    invalid = _input_error(graph)
    if invalid is not None:
        return invalid
    goal = _open_goal(graph, goal_id)
    if isinstance(goal, Err):
        return goal
    if not isinstance(draft, DecompositionDraft):
        return Err("decompose needs a DecompositionDraft")
    if not isinstance(draft.connective, Connective):
        return Err("decomposition connective has an invalid type")
    if direct_child_goals(graph, goal_id):
        return Err(f"goal {goal_id} is already decomposed")
    if type(draft.child_statements) is not tuple or any(
        not isinstance(statement, str) or not statement.strip()
        for statement in draft.child_statements
    ):
        return Err("decomposition child statements must be nonempty strings")
    if type(draft.child_obligations) is not tuple or any(
        not isinstance(obligation, GoalObligation)
        for obligation in draft.child_obligations
    ):
        return Err("decomposition child obligations have an invalid type")
    if not draft.child_statements:
        return Err("decompose needs at least one child")
    if len(draft.child_statements) != len(draft.child_obligations):
        return Err("every child statement needs one trusted obligation")
    if len(set(draft.child_statements)) != len(draft.child_statements):
        return Err("decomposition child statements must be unique")
    if not isinstance(generation_contract, GenerationContract):
        return Err("generation_contract has an invalid type")
    try:
        canonical_children = generation_contract.validate_decomposition(
            goal.obligation, draft.connective, draft.child_obligations
        )
    except Exception as exc:
        return Err(f"decomposition rejected: {exc}")
    if type(canonical_children) is not tuple or len(
        canonical_children
    ) != len(draft.child_statements):
        return Err("decomposition validator returned an invalid child tuple")
    if any(
        not isinstance(child, GoalObligation) for child in canonical_children
    ):
        return Err("decomposition validator returned an invalid child type")

    updated = graph
    try:
        for statement, obligation in zip(
            draft.child_statements, canonical_children, strict=True
        ):
            if obligation.objective is not GoalObjective.BRANCH:
                return Err("decomposition child objective must be branch")
            updated, _ = _append(
                updated,
                lambda node_id, text=statement, item=obligation: Goal(
                    id=node_id,
                    parent_goal=goal.id,
                    statement=text,
                    obligation=item,
                    assumptions=goal.assumptions,
                ),
            )
        updated = _replace(
            updated, replace(get(updated, goal.id), connective=draft.connective)
        )
    except (AttributeError, GraphValidationError, KeyError, TypeError) as exc:
        return Err(f"cannot decompose goal: {exc}")
    return _finish(updated)


def propose_solution(
    graph: Graph,
    goal_id: NodeId,
    draft: SolutionDraft,
    generation_contract: GenerationContract,
) -> Result:
    if not isinstance(draft, SolutionDraft):
        return Err("propose_solution needs a SolutionDraft")
    invalid = _input_error(graph)
    if invalid is not None:
        return invalid
    return _add_claim(
        graph,
        goal_id,
        ClaimKind.SOLUTION,
        SolutionPayload(variable=draft.variable, value=draft.value),
        (),
        generation_contract,
    )


def propose_result(
    graph: Graph,
    goal_id: NodeId,
    draft: ResultDraft,
    generation_contract: GenerationContract,
) -> Result:
    if not isinstance(draft, ResultDraft):
        return Err("propose_result needs a ResultDraft")
    return _add_claim(
        graph,
        goal_id,
        ClaimKind.RESULT,
        ResultPayload(objective=draft.objective, value=draft.value),
        (),
        generation_contract,
    )


def derive(
    graph: Graph,
    goal_id: NodeId,
    draft: DerivationDraft,
    generation_contract: GenerationContract,
) -> Result:
    if not isinstance(draft, DerivationDraft):
        return Err("derive needs a DerivationDraft")
    return _add_claim(
        graph,
        goal_id,
        ClaimKind.DERIVATION,
        DerivationPayload(lhs=draft.lhs, relation=draft.relation, rhs=draft.rhs),
        draft.premises,
        generation_contract,
    )


_REDUCER_BY_OBJECTIVE = {
    GoalObjective.SUM: Reducer.SUM,
    GoalObjective.MAX: Reducer.MAX,
    GoalObjective.MIN: Reducer.MIN,
    GoalObjective.IDENTITY: Reducer.IDENTITY,
}


def aggregate(
    graph: Graph,
    goal_id: NodeId,
    draft: AggregateDraft,
    generation_contract: GenerationContract,
) -> Result:
    if not isinstance(draft, AggregateDraft):
        return Err("aggregate needs an AggregateDraft")
    invalid = _input_error(graph)
    if invalid is not None:
        return invalid
    goal = _open_goal(graph, goal_id)
    if isinstance(goal, Err):
        return goal
    if goal.connective is not Connective.EXHAUSTIVE_CASE:
        return Err("aggregate needs an exhaustive-case decomposition")
    expected_reducer = _REDUCER_BY_OBJECTIVE.get(goal.obligation.objective)
    if expected_reducer is None or draft.reducer is not expected_reducer:
        return Err("aggregate reducer differs from the owner objective")
    children = direct_child_goals(graph, goal_id)
    if not children:
        return Err("aggregate needs decomposed child goals")
    expected_claims = tuple(child.committed_claim for child in children)
    if any(claim_id is None for claim_id in expected_claims):
        return Err("aggregate needs every child goal committed")
    if draft.child_claims != expected_claims:
        return Err("aggregate must consume the exact committed child claims")
    return _add_claim(
        graph,
        goal_id,
        ClaimKind.AGGREGATE,
        AggregatePayload(
            reducer=draft.reducer,
            result=draft.result,
        ),
        draft.child_claims,
        generation_contract,
    )


def propose_refutation(
    graph: Graph,
    goal_id: NodeId,
    draft: RefutationDraft,
    generation_contract: GenerationContract,
) -> Result:
    if not isinstance(draft, RefutationDraft):
        return Err("propose_refutation needs a RefutationDraft")
    invalid = _input_error(graph)
    if invalid is not None:
        return invalid
    return _add_claim(
        graph,
        goal_id,
        ClaimKind.REFUTATION,
        RefutationPayload(
            mode=draft.mode,
            variable=draft.variable,
            candidate=draft.candidate,
        ),
        (),
        generation_contract,
    )


def verify_claim(
    graph: Graph, claim_id: NodeId, verifier: ClaimVerifier
) -> Result:
    invalid = _input_error(graph)
    if invalid is not None:
        return invalid
    try:
        node = get(graph, claim_id)
    except KeyError:
        return Err(f"no node with id {claim_id}")
    if not isinstance(node, Claim):
        return Err(f"node {claim_id} is not a Claim")
    if node.status is not ClaimStatus.UNCHECKED or not is_active(graph, claim_id):
        return Err(f"claim {claim_id} is not active and unchecked")
    try:
        decision = verifier.verify(graph, node)
    except Exception as exc:
        return Err(f"claim verifier raised {type(exc).__name__}: {exc}")
    if not isinstance(decision, VerificationDecision):
        return Err("claim verifier returned an invalid decision type")
    updated = _replace(
        graph,
        replace(
            node,
            verification=decision.verification,
            discharge=decision.discharge,
        ),
    )
    return _finish(updated)


def commit_claim(graph: Graph, claim_id: NodeId) -> Result:
    invalid = _input_error(graph)
    if invalid is not None:
        return invalid
    try:
        node = get(graph, claim_id)
    except KeyError:
        return Err(f"no node with id {claim_id}")
    if not isinstance(node, Claim):
        return Err(f"node {claim_id} is not a Claim")
    if not is_committable(graph, claim_id):
        return Err(f"claim {claim_id} is not verified and owner-discharging")
    owner = get(graph, node.owner_goal)
    if not isinstance(owner, Goal):
        return Err(f"claim {claim_id} owner is not a Goal")
    return _finish(
        _replace(graph, replace(owner, committed_claim=claim_id))
    )


def prune(graph: Graph, target_id: NodeId) -> Result:
    invalid = _input_error(graph)
    if invalid is not None:
        return invalid
    if target_id == graph.root_id:
        return Err("root goal cannot be pruned")
    try:
        node = get(graph, target_id)
    except KeyError:
        return Err(f"no node with id {target_id}")
    if isinstance(node, Evidence):
        return Err("Evidence cannot be pruned")
    if not isinstance(node, (Goal, Claim)):
        return Err(f"node {target_id} is not a Goal or Claim")
    if not is_active(graph, target_id):
        return Err(f"node {target_id} is not active")
    if isinstance(node, Goal) and node.status is not GoalStatus.OPEN:
        return Err(f"goal {target_id} is not open")
    if isinstance(node, Claim) and node.pruned:
        return Err(f"claim {target_id} is already pruned")
    return _finish(_replace(graph, replace(node, pruned=True)))


__all__ = [
    "AggregateDraft",
    "DecompositionDraft",
    "DerivationDraft",
    "Err",
    "Ok",
    "RefutationDraft",
    "ResultDraft",
    "Result",
    "SolutionDraft",
    "aggregate",
    "commit_claim",
    "decompose",
    "derive",
    "propose_refutation",
    "propose_result",
    "propose_solution",
    "prune",
    "verify_claim",
]
