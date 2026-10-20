"""Canonical controller observations for the typed v2 reasoning graph."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from .menu import (
    ACTION_TARGET_KINDS,
    POLICY_VERB_ORDER,
    VERB_ORDER,
    available_actions,
)

from .model import (
    Action,
    AggregatePayload,
    CheckPurpose,
    Claim,
    ClaimKind,
    ClaimStatus,
    Connective,
    DerivationPayload,
    DischargeStatus,
    Evidence,
    Goal,
    GoalObjective,
    GoalObligation,
    GoalStatus,
    Graph,
    OutcomeCode,
    Reducer,
    RefutationMode,
    RefutationPayload,
    RejectionReason,
    Relation,
    ResultPayload,
    SolutionPayload,
    State,
    SymbolicCheck,
    TransitionOutcome,
    Verb,
    VerifierContext,
    claim_statement,
    get,
    goal_ancestry,
    validate_graph,
)
from .verifier import GenerationContract, VerifierReceiptContract


CONTROLLER_OBSERVATION_VERSION = "controller-observation-v2"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "environment_contract_sha256",
        "root_goal_id",
        "turn",
        "turn_limit",
        "previous_outcome",
        "goals",
        "evidence",
        "claims",
        "legal_pairs",
    }
)
_GOAL_KEYS = frozenset(
    {
        "id",
        "parent_goal",
        "statement",
        "obligation",
        "assumptions",
        "connective",
        "status",
        "committed_claim",
    }
)
_OBLIGATION_KEYS = frozenset(
    {"variable", "equation", "domain", "objective"}
)
_EVIDENCE_KEYS = frozenset(
    {"id", "scope_goal", "statement", "premises", "source", "committable"}
)
_CLAIM_KEYS = frozenset(
    {
        "id",
        "owner_goal",
        "kind",
        "payload",
        "premises",
        "statement",
        "status",
        "verifier_result",
        "discharge_result",
        "discharge_status",
    }
)
_COMPACT_CLAIM_KEYS = frozenset(
    {
        "id",
        "kind",
        "payload",
        "statement",
        "verifier_result",
        "discharge_result",
    }
)
_VERIFIER_RESULT_KEYS = frozenset(
    {"contract_id", "context_digest", "accepted", "reason_code"}
)
_ACTION_KEYS = frozenset({"verb", "target"})
_OUTCOME_KEYS = frozenset(
    {"action", "code", "reason", "verifier_context"}
)
_VERIFIER_CONTEXT_KEYS = frozenset(
    {
        "contract_id",
        "purpose",
        "claim_kind",
        "owner_goal",
        "context_digest",
    }
)

_VERB_RANK = {verb.value: index for index, verb in enumerate(VERB_ORDER)}
_GOAL_VERBS = frozenset(
    verb
    for verb in VERB_ORDER
    if ACTION_TARGET_KINDS[verb.value] == ("goal",)
)
_CLAIM_VERBS = frozenset(
    verb
    for verb in VERB_ORDER
    if ACTION_TARGET_KINDS[verb.value] == ("claim",)
)
_OBJECTIVE_REDUCERS = {
    GoalObjective.SUM.value: Reducer.SUM.value,
    GoalObjective.MAX.value: Reducer.MAX.value,
    GoalObjective.MIN.value: Reducer.MIN.value,
    GoalObjective.IDENTITY.value: Reducer.IDENTITY.value,
}


class ObservationValidationError(ValueError):
    """Raised when a v2 controller observation violates its contract."""


@dataclass(frozen=True)
class ObservationContext:
    """Episode-static values required to build an observation."""

    turn_limit: int
    environment_contract_sha256: str

    def __post_init__(self) -> None:
        if type(self.turn_limit) is not int or self.turn_limit <= 0:
            raise ValueError("turn_limit must be a positive integer")
        if not _is_sha256(self.environment_contract_sha256):
            raise ValueError(
                "environment_contract_sha256 must be 64 lowercase hex characters"
            )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _fail(path: str, message: str) -> None:
    raise ObservationValidationError(f"{path}: {message}")


def _object(value: object, path: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(path, "must be an object")
    return value


def _array(value: object, path: str) -> list[Any]:
    if type(value) is not list:
        _fail(path, "must be an array")
    return value


def _exact_keys(
    value: object, expected: frozenset[str], path: str
) -> dict[str, Any]:
    record = _object(value, path)
    if set(record) != expected:
        _fail(path, f"keys must be exactly {sorted(expected)!r}")
    return record


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be nonempty text")
    return value


def _integer(value: object, path: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(path, f"must be an integer greater than or equal to {minimum}")
    return value


def _boolean(value: object, path: str) -> bool:
    if type(value) is not bool:
        _fail(path, "must be a boolean")
    return value


def _enum(value: object, members: type[Any], path: str) -> str:
    choices = {member.value for member in members}
    if not isinstance(value, str) or value not in choices:
        _fail(path, f"must be one of {sorted(choices)!r}")
    return value


def _digest(value: object, path: str) -> str:
    if not _is_sha256(value):
        _fail(path, "must be 64 lowercase hex characters")
    return value


def serialize_payload(payload: object) -> dict[str, Any]:
    """Return the exact kind-indexed JSON payload for one Claim payload."""
    if isinstance(payload, SolutionPayload):
        return {"variable": payload.variable, "value": payload.value}
    if isinstance(payload, ResultPayload):
        return {
            "objective": payload.objective.value,
            "value": payload.value,
        }
    if isinstance(payload, DerivationPayload):
        return {
            "relation": payload.relation.value,
            "lhs": payload.lhs,
            "rhs": payload.rhs,
        }
    if isinstance(payload, AggregatePayload):
        return {"reducer": payload.reducer.value, "result": payload.result}
    if isinstance(payload, RefutationPayload):
        return {
            "mode": payload.mode.value,
            "variable": payload.variable,
            "candidate": payload.candidate,
        }
    raise TypeError(f"unsupported Claim payload {type(payload).__name__}")


def _payload_statement(
    value: object, kind: str, path: str
) -> tuple[dict[str, Any], str]:
    payload = _object(value, path)
    if kind == ClaimKind.SOLUTION.value:
        payload = _exact_keys(
            payload, frozenset({"variable", "value"}), path
        )
        variable = _text(payload["variable"], f"{path}.variable")
        item = _text(payload["value"], f"{path}.value")
        return payload, f"{variable} = {item}"
    if kind == ClaimKind.RESULT.value:
        payload = _exact_keys(
            payload, frozenset({"objective", "value"}), path
        )
        objective = _enum(
            payload["objective"], GoalObjective, f"{path}.objective"
        )
        if objective == GoalObjective.BRANCH.value:
            _fail(f"{path}.objective", "cannot be branch")
        item = _text(payload["value"], f"{path}.value")
        return payload, f"result({objective}) = {item}"
    if kind == ClaimKind.DERIVATION.value:
        payload = _exact_keys(
            payload, frozenset({"relation", "lhs", "rhs"}), path
        )
        relation = _enum(payload["relation"], Relation, f"{path}.relation")
        lhs = _text(payload["lhs"], f"{path}.lhs")
        rhs = _text(payload["rhs"], f"{path}.rhs")
        symbols = {"eq": "=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
        return payload, f"{lhs} {symbols[relation]} {rhs}"
    if kind == ClaimKind.AGGREGATE.value:
        payload = _exact_keys(
            payload, frozenset({"reducer", "result"}), path
        )
        reducer = _enum(payload["reducer"], Reducer, f"{path}.reducer")
        result = _text(payload["result"], f"{path}.result")
        return payload, f"{reducer}(committed child outcomes) = {result}"
    if kind == ClaimKind.REFUTATION.value:
        payload = _exact_keys(
            payload, frozenset({"mode", "variable", "candidate"}), path
        )
        _enum(payload["mode"], RefutationMode, f"{path}.mode")
        variable = _text(payload["variable"], f"{path}.variable")
        candidate = _text(payload["candidate"], f"{path}.candidate")
        return payload, f"{variable} = {candidate} is outside the owner domain"
    _fail(path, f"unsupported claim kind {kind!r}")


def _typed_obligation(record: dict[str, Any]) -> GoalObligation:
    return GoalObligation(
        variable=record["variable"],
        equation=record["equation"],
        domain=record["domain"],
        objective=GoalObjective(record["objective"]),
    )


def _typed_payload(record: dict[str, Any], kind: ClaimKind):
    if kind is ClaimKind.RESULT:
        return ResultPayload(
            GoalObjective(record["objective"]), record["value"]
        )
    if kind is ClaimKind.SOLUTION:
        return SolutionPayload(record["variable"], record["value"])
    if kind is ClaimKind.DERIVATION:
        return DerivationPayload(
            record["lhs"], Relation(record["relation"]), record["rhs"]
        )
    if kind is ClaimKind.AGGREGATE:
        return AggregatePayload(
            Reducer(record["reducer"]), record["result"]
        )
    if kind is ClaimKind.REFUTATION:
        return RefutationPayload(
            RefutationMode(record["mode"]),
            record["variable"],
            record["candidate"],
        )
    raise AssertionError(f"unsupported Claim kind {kind.value}")


def _serialize_obligation(goal: Goal) -> dict[str, Any]:
    obligation = goal.obligation
    return {
        "variable": obligation.variable,
        "equation": obligation.equation,
        "domain": obligation.domain,
        "objective": obligation.objective.value,
    }


def serialize_obligation(obligation: GoalObligation) -> dict[str, Any]:
    if not isinstance(obligation, GoalObligation):
        raise TypeError("serialize_obligation needs a GoalObligation")
    return {
        "variable": obligation.variable,
        "equation": obligation.equation,
        "domain": obligation.domain,
        "objective": obligation.objective.value,
    }


def serialize_action(action: Action) -> dict[str, Any]:
    if not isinstance(action, Action):
        raise TypeError("serialize_action needs an Action")
    return {"verb": action.verb.value, "target": action.target}


def _serialize_check(check: SymbolicCheck | None) -> dict[str, Any] | None:
    if check is None:
        return None
    return {
        "contract_id": check.context.contract_id,
        "context_digest": check.context.context_digest,
        "accepted": check.accepted,
        "reason_code": check.reason_code,
    }


def _serialize_context(context: VerifierContext) -> dict[str, Any]:
    return {
        "contract_id": context.contract_id,
        "purpose": context.purpose.value,
        "claim_kind": context.claim_kind.value,
        "owner_goal": context.owner_goal,
        "context_digest": context.context_digest,
    }


def serialize_verifier_context(
    context: VerifierContext,
) -> dict[str, Any]:
    if not isinstance(context, VerifierContext):
        raise TypeError(
            "serialize_verifier_context needs a VerifierContext"
        )
    return _serialize_context(context)


def _serialize_outcome(outcome: TransitionOutcome | None) -> dict[str, Any] | None:
    if outcome is None:
        return None
    return {
        "action": {
            "verb": outcome.action.verb.value,
            "target": outcome.action.target,
        },
        "code": outcome.code.value,
        "reason": outcome.reason.value if outcome.reason is not None else None,
        "verifier_context": (
            _serialize_context(outcome.verifier_context)
            if outcome.verifier_context is not None
            else None
        ),
    }


def serialize_outcome(
    outcome: TransitionOutcome | None,
) -> dict[str, Any] | None:
    if outcome is not None and not isinstance(outcome, TransitionOutcome):
        raise TypeError("serialize_outcome needs a TransitionOutcome or None")
    return _serialize_outcome(outcome)


def _goal_is_visible(graph: Graph, goal: Goal) -> bool:
    return all(
        ancestor.status is GoalStatus.OPEN
        for ancestor in goal_ancestry(graph, goal.id)[:-1]
    )


def _owner_is_active(graph: Graph, goal_id: int) -> bool:
    return all(
        ancestor.status is GoalStatus.OPEN
        for ancestor in goal_ancestry(graph, goal_id)
    )


def _compact_claim(claim: Claim) -> dict[str, Any]:
    return {
        "id": claim.id,
        "kind": claim.kind.value,
        "payload": serialize_payload(claim.payload),
        "statement": claim.statement,
        "verifier_result": _serialize_check(claim.verification),
        "discharge_result": _serialize_check(claim.discharge),
    }


def _goal_record(graph: Graph, goal: Goal) -> dict[str, Any]:
    compact = None
    if goal.status is GoalStatus.SOLVED:
        committed = get(graph, goal.committed_claim)
        if not isinstance(committed, Claim):
            raise ObservationValidationError(
                f"goal {goal.id} committed outcome is not a Claim"
            )
        compact = _compact_claim(committed)
    return {
        "id": goal.id,
        "parent_goal": goal.parent_goal,
        "statement": goal.statement,
        "obligation": _serialize_obligation(goal),
        "assumptions": list(goal.assumptions),
        "connective": (
            goal.connective.value
            if goal.status is GoalStatus.OPEN and goal.connective is not None
            else None
        ),
        "status": goal.status.value,
        "committed_claim": compact,
    }


def _evidence_record(node: Evidence) -> dict[str, Any]:
    return {
        "id": node.id,
        "scope_goal": node.scope_goal,
        "statement": node.statement,
        "premises": list(node.premises),
        "source": node.source,
        "committable": False,
    }


def _claim_record(node: Claim) -> dict[str, Any]:
    return {
        "id": node.id,
        "owner_goal": node.owner_goal,
        "kind": node.kind.value,
        "payload": serialize_payload(node.payload),
        "premises": list(node.premises),
        "statement": claim_statement(node.payload),
        "status": node.status.value,
        "verifier_result": _serialize_check(node.verification),
        "discharge_result": _serialize_check(node.discharge),
        "discharge_status": node.discharge_status.value,
    }


def build_observation(
    state: State,
    context: ObservationContext,
) -> dict[str, Any]:
    """Build and validate the sole canonical v2 controller observation."""
    if not isinstance(state, State):
        raise TypeError("state must be a State")
    if not isinstance(context, ObservationContext):
        raise TypeError("context must be an ObservationContext")
    if type(state.turn) is not int or state.turn < 0:
        raise ObservationValidationError("state turn must be nonnegative")
    if state.turn > context.turn_limit:
        raise ObservationValidationError("state turn exceeds turn_limit")
    try:
        validate_graph(state.graph)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ObservationValidationError(f"invalid graph: {exc}") from exc

    actions = available_actions(state, context.turn_limit)
    root = get(state.graph, state.graph.root_id)
    if not isinstance(root, Goal):
        raise ObservationValidationError("graph root is not a Goal")
    if root.status is not GoalStatus.OPEN and actions:
        raise ObservationValidationError("closed-root observations have no legal actions")

    goals = [
        _goal_record(state.graph, node)
        for node in state.graph.nodes
        if isinstance(node, Goal) and _goal_is_visible(state.graph, node)
    ]
    evidence = [
        _evidence_record(node)
        for node in state.graph.nodes
        if isinstance(node, Evidence)
        and _owner_is_active(state.graph, node.scope_goal)
    ]
    claims = [
        _claim_record(node)
        for node in state.graph.nodes
        if isinstance(node, Claim)
        and _owner_is_active(state.graph, node.owner_goal)
    ]
    legal_pairs = [
        {"verb": action.verb.value, "target": action.target}
        for action in sorted(
            actions, key=lambda item: (_VERB_RANK[item.verb.value], item.target)
        )
    ]
    observation = {
        "schema_version": CONTROLLER_OBSERVATION_VERSION,
        "environment_contract_sha256": context.environment_contract_sha256,
        "root_goal_id": state.graph.root_id,
        "turn": state.turn,
        "turn_limit": context.turn_limit,
        "previous_outcome": _serialize_outcome(state.previous_outcome),
        "goals": goals,
        "evidence": evidence,
        "claims": claims,
        "legal_pairs": legal_pairs,
    }
    validate_observation(observation)
    return observation


def _validate_obligation_record(value: object, path: str) -> dict[str, Any]:
    record = _exact_keys(value, _OBLIGATION_KEYS, path)
    _text(record["variable"], f"{path}.variable")
    _text(record["equation"], f"{path}.equation")
    _text(record["domain"], f"{path}.domain")
    _enum(record["objective"], GoalObjective, f"{path}.objective")
    return record


def _validate_verifier_result(value: object, path: str) -> dict[str, Any]:
    record = _exact_keys(value, _VERIFIER_RESULT_KEYS, path)
    _text(record["contract_id"], f"{path}.contract_id")
    _digest(record["context_digest"], f"{path}.context_digest")
    accepted = _boolean(record["accepted"], f"{path}.accepted")
    reason = record["reason_code"]
    if accepted:
        if reason is not None:
            _fail(f"{path}.reason_code", "must be null for an accepted result")
    else:
        _text(reason, f"{path}.reason_code")
    return record


def _validate_context_record(value: object, path: str) -> dict[str, Any]:
    record = _exact_keys(value, _VERIFIER_CONTEXT_KEYS, path)
    _text(record["contract_id"], f"{path}.contract_id")
    _enum(record["purpose"], CheckPurpose, f"{path}.purpose")
    _enum(record["claim_kind"], ClaimKind, f"{path}.claim_kind")
    _integer(record["owner_goal"], f"{path}.owner_goal")
    _digest(record["context_digest"], f"{path}.context_digest")
    return record


def _validate_id_array(value: object, path: str) -> list[int]:
    items = _array(value, path)
    result = [_integer(item, f"{path}[{index}]") for index, item in enumerate(items)]
    if len(result) != len(set(result)):
        _fail(path, "must not contain duplicate ids")
    return result


def _validate_sorted_ids(records: list[dict[str, Any]], path: str) -> None:
    ids = [record["id"] for record in records]
    if ids != sorted(ids):
        _fail(path, "records must be sorted by id")


def _validate_claim_status(
    status: str,
    verifier_result: dict[str, Any] | None,
    discharge_result: dict[str, Any] | None,
    discharge_status: str,
    path: str,
) -> None:
    if verifier_result is not None and discharge_result is not None and (
        verifier_result["contract_id"] != discharge_result["contract_id"]
    ):
        _fail(path, "validity and discharge verifier contracts differ")
    if status == ClaimStatus.UNCHECKED.value:
        if (
            verifier_result is not None
            or discharge_result is not None
            or discharge_status != DischargeStatus.UNCHECKED.value
        ):
            _fail(path, "unchecked Claim has a terminal verifier result")
        return
    if status == ClaimStatus.VERIFIED.value:
        if verifier_result is None or not verifier_result["accepted"]:
            _fail(path, "verified Claim requires an accepted verifier result")
        if discharge_result is None:
            _fail(path, "verified Claim requires a discharge result")
        expected = (
            DischargeStatus.ACCEPTED.value
            if discharge_result["accepted"]
            else DischargeStatus.REJECTED.value
        )
        if discharge_status != expected:
            _fail(path, "verified Claim discharge result and status differ")
        return
    if status == ClaimStatus.REJECTED.value:
        if verifier_result is None or verifier_result["accepted"]:
            _fail(path, "rejected Claim requires a rejected verifier result")
        if (
            discharge_result is not None
            or discharge_status != DischargeStatus.UNCHECKED.value
        ):
            _fail(path, "rejected Claim must leave owner discharge unchecked")
        return
    if verifier_result is None:
        if (
            discharge_result is not None
            or discharge_status != DischargeStatus.UNCHECKED.value
        ):
            _fail(path, "unchecked pruned Claim has terminal discharge")
    elif verifier_result["accepted"]:
        if discharge_result is None:
            _fail(path, "verified pruned Claim needs terminal discharge")
        expected = (
            DischargeStatus.ACCEPTED.value
            if discharge_result["accepted"]
            else DischargeStatus.REJECTED.value
        )
        if discharge_status != expected:
            _fail(path, "verified pruned Claim discharge result and status differ")
    elif (
        discharge_result is not None
        or discharge_status != DischargeStatus.UNCHECKED.value
    ):
        _fail(path, "rejected pruned Claim must leave owner discharge unchecked")


def _validate_payload_owner(
    payload: dict[str, Any], kind: str, owner: dict[str, Any], path: str
) -> None:
    obligation = owner["obligation"]
    if kind in {ClaimKind.SOLUTION.value, ClaimKind.REFUTATION.value}:
        if payload["variable"] != obligation["variable"]:
            _fail(path, "variable differs from the owner obligation")
    if kind == ClaimKind.RESULT.value:
        if payload["objective"] != obligation["objective"]:
            _fail(path, "objective differs from the owner obligation")
    if kind == ClaimKind.AGGREGATE.value:
        expected = _OBJECTIVE_REDUCERS.get(obligation["objective"])
        if payload["reducer"] != expected:
            _fail(path, "reducer differs from the owner objective")


def _target_kind(
    target: int,
    goals: dict[int, dict[str, Any]],
    evidence: dict[int, dict[str, Any]],
    claims: dict[int, dict[str, Any]],
    compact_claims: dict[int, dict[str, Any]],
    path: str,
) -> str:
    matches = [
        name
        for name, records in (
            ("goal", goals),
            ("evidence", evidence),
            ("claim", claims),
            ("compact_claim", compact_claims),
        )
        if target in records
    ]
    if len(matches) != 1:
        _fail(path, "target must reference exactly one serialized node")
    return matches[0]


def _validate_action_target(
    verb: str,
    target: int,
    goals: dict[int, dict[str, Any]],
    evidence: dict[int, dict[str, Any]],
    claims: dict[int, dict[str, Any]],
    compact_claims: dict[int, dict[str, Any]],
    path: str,
    *,
    require_active: bool,
) -> str:
    kind = _target_kind(
        target, goals, evidence, claims, compact_claims, f"{path}.target"
    )
    if verb in {item.value for item in _GOAL_VERBS}:
        if kind != "goal":
            _fail(path, f"{verb} must target a Goal")
        if require_active and goals[target]["status"] != GoalStatus.OPEN.value:
            _fail(path, f"{verb} must target an active open Goal")
    elif verb in {item.value for item in _CLAIM_VERBS}:
        if kind not in {"claim", "compact_claim"}:
            _fail(path, f"{verb} must target a Claim")
        if require_active and kind != "claim":
            _fail(path, f"{verb} must target an active Claim")
    elif verb == Verb.PRUNE.value:
        if kind not in {"goal", "claim", "compact_claim"}:
            _fail(path, "prune must target a Goal or Claim")
        if require_active and (
            kind == "compact_claim"
            or (kind == "goal" and goals[target]["status"] != GoalStatus.OPEN.value)
            or (kind == "claim" and claims[target]["status"] == ClaimStatus.PRUNED.value)
        ):
            _fail(path, "prune must target an active Goal or Claim")
    return kind


def _expected_legal_pairs(
    goals: dict[int, dict[str, Any]],
    claims: dict[int, dict[str, Any]],
    root_id: int,
    turn: int,
    turn_limit: int,
) -> list[dict[str, Any]]:
    if turn >= turn_limit or goals[root_id]["status"] != GoalStatus.OPEN.value:
        return []
    children_by_parent: dict[int, list[dict[str, Any]]] = {}
    for goal in goals.values():
        parent = goal["parent_goal"]
        if parent is not None:
            children_by_parent.setdefault(parent, []).append(goal)

    closure_pending = any(
        claim["status"] == ClaimStatus.UNCHECKED.value
        or (
            claim["status"] == ClaimStatus.VERIFIED.value
            and claim["discharge_status"]
            == DischargeStatus.ACCEPTED.value
        )
        for claim in claims.values()
        if claim["status"] != ClaimStatus.PRUNED.value
    ) or any(
        goal["status"] == GoalStatus.OPEN.value
        and goal["obligation"]["objective"] in _OBJECTIVE_REDUCERS
        and goal["connective"] == Connective.EXHAUSTIVE_CASE.value
        and bool(children_by_parent.get(goal_id))
        and all(
            child["status"] == GoalStatus.SOLVED.value
            for child in children_by_parent[goal_id]
        )
        for goal_id, goal in goals.items()
    )
    if closure_pending:
        return []

    expected: list[dict[str, Any]] = []
    for verb in POLICY_VERB_ORDER:
        for target in sorted(goals):
            goal = goals[target]
            if goal["status"] != GoalStatus.OPEN.value:
                continue
            children = children_by_parent.get(target, [])
            objective = goal["obligation"]["objective"]
            allowed = (
                (
                    verb is Verb.DECOMPOSE
                    and objective in _OBJECTIVE_REDUCERS
                    and not children
                )
                or (
                    verb is Verb.PROPOSE_RESULT
                    and objective
                    in {
                        GoalObjective.SUM.value,
                        GoalObjective.MAX.value,
                        GoalObjective.MIN.value,
                    }
                )
                or (
                    verb is Verb.PROPOSE_SOLUTION
                    and objective
                    in {
                        GoalObjective.BRANCH.value,
                        GoalObjective.IDENTITY.value,
                    }
                )
                or (
                    verb is Verb.PROPOSE_REFUTATION
                    and objective == GoalObjective.BRANCH.value
                )
            )
            if allowed:
                expected.append({"verb": verb.value, "target": target})
    return expected


def validate_observation(observation: object) -> None:
    """Validate one decoded v2 observation, including semantic references."""
    record = _exact_keys(observation, _TOP_LEVEL_KEYS, "observation")
    if record["schema_version"] != CONTROLLER_OBSERVATION_VERSION:
        _fail("observation.schema_version", "has an unsupported value")
    _digest(
        record["environment_contract_sha256"],
        "observation.environment_contract_sha256",
    )
    root_id = _integer(record["root_goal_id"], "observation.root_goal_id")
    turn = _integer(record["turn"], "observation.turn")
    turn_limit = _integer(
        record["turn_limit"], "observation.turn_limit", minimum=1
    )
    if turn > turn_limit:
        _fail("observation.turn", "cannot exceed turn_limit")

    goal_records: list[dict[str, Any]] = []
    compact_claims: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(_array(record["goals"], "observation.goals")):
        path = f"observation.goals[{index}]"
        goal = _exact_keys(item, _GOAL_KEYS, path)
        goal_id = _integer(goal["id"], f"{path}.id")
        parent = goal["parent_goal"]
        if parent is not None:
            _integer(parent, f"{path}.parent_goal")
        _text(goal["statement"], f"{path}.statement")
        _validate_obligation_record(goal["obligation"], f"{path}.obligation")
        assumptions = _array(goal["assumptions"], f"{path}.assumptions")
        for assumption_index, assumption in enumerate(assumptions):
            _text(assumption, f"{path}.assumptions[{assumption_index}]")
        connective = goal["connective"]
        if connective is not None:
            _enum(connective, Connective, f"{path}.connective")
        status = _enum(goal["status"], GoalStatus, f"{path}.status")
        committed = goal["committed_claim"]
        if status == GoalStatus.SOLVED.value:
            compact_path = f"{path}.committed_claim"
            compact = _exact_keys(committed, _COMPACT_CLAIM_KEYS, compact_path)
            compact_id = _integer(compact["id"], f"{compact_path}.id")
            kind = _enum(compact["kind"], ClaimKind, f"{compact_path}.kind")
            payload, statement = _payload_statement(
                compact["payload"], kind, f"{compact_path}.payload"
            )
            if compact["statement"] != statement:
                _fail(f"{compact_path}.statement", "differs from the typed payload")
            verifier_result = _validate_verifier_result(
                compact["verifier_result"],
                f"{compact_path}.verifier_result",
            )
            discharge_result = _validate_verifier_result(
                compact["discharge_result"],
                f"{compact_path}.discharge_result",
            )
            _validate_claim_status(
                ClaimStatus.VERIFIED.value,
                verifier_result,
                discharge_result,
                DischargeStatus.ACCEPTED.value,
                compact_path,
            )
            if compact_id <= goal_id:
                _fail(f"{compact_path}.id", "must be later than its owner Goal")
            _validate_payload_owner(payload, kind, goal, compact_path)
            if compact_id in compact_claims:
                _fail(f"{compact_path}.id", "duplicates another compact Claim")
            compact_claims[compact_id] = compact
        elif committed is not None:
            _fail(f"{path}.committed_claim", "must be null unless Goal is solved")
        goal_records.append(goal)
    _validate_sorted_ids(goal_records, "observation.goals")
    goals = {goal["id"]: goal for goal in goal_records}
    if len(goals) != len(goal_records):
        _fail("observation.goals", "contains duplicate ids")
    if root_id not in goals:
        _fail("observation.root_goal_id", "does not reference a Goal")
    if root_id != 0 or goals[root_id]["parent_goal"] is not None:
        _fail("observation.root_goal_id", "must identify parentless Goal 0")
    if goals[root_id]["status"] == GoalStatus.PRUNED.value:
        _fail("observation.goals", "root Goal cannot be pruned")
    for index, goal in enumerate(goal_records):
        goal_id = goal["id"]
        parent = goal["parent_goal"]
        if goal_id == root_id:
            if parent is not None:
                _fail(f"observation.goals[{index}].parent_goal", "must be null")
            continue
        if parent not in goals or parent >= goal_id:
            _fail(
                f"observation.goals[{index}].parent_goal",
                "must reference an earlier serialized Goal",
            )
        if goals[parent]["status"] != GoalStatus.OPEN.value:
            _fail(
                f"observation.goals[{index}].parent_goal",
                "cannot place a record below a closed Goal",
            )
    children_by_parent: dict[int, list[dict[str, Any]]] = {}
    for goal in goal_records:
        if goal["parent_goal"] is not None:
            children_by_parent.setdefault(goal["parent_goal"], []).append(goal)
    for index, goal in enumerate(goal_records):
        children = children_by_parent.get(goal["id"], [])
        if bool(children) != (goal["connective"] is not None):
            _fail(
                f"observation.goals[{index}].connective",
                "connective and direct children disagree",
            )
        for child in children:
            if child["obligation"]["objective"] != GoalObjective.BRANCH.value:
                _fail(
                    f"observation.goals[{index}]",
                    "direct child objective must be branch",
                )
            if (
                child["obligation"]["variable"]
                != goal["obligation"]["variable"]
            ):
                _fail(
                    f"observation.goals[{index}]",
                    "direct child variable differs from its parent",
                )

    evidence_records: list[dict[str, Any]] = []
    for index, item in enumerate(
        _array(record["evidence"], "observation.evidence")
    ):
        path = f"observation.evidence[{index}]"
        node = _exact_keys(item, _EVIDENCE_KEYS, path)
        node_id = _integer(node["id"], f"{path}.id")
        scope_goal = _integer(node["scope_goal"], f"{path}.scope_goal")
        if scope_goal not in goals or scope_goal >= node_id:
            _fail(f"{path}.scope_goal", "must reference an earlier Goal")
        if goals[scope_goal]["status"] != GoalStatus.OPEN.value:
            _fail(f"{path}.scope_goal", "must reference an active Goal")
        _text(node["statement"], f"{path}.statement")
        _validate_id_array(node["premises"], f"{path}.premises")
        if node["source"] != "trusted_given":
            _fail(f"{path}.source", "must be trusted_given")
        if _boolean(node["committable"], f"{path}.committable"):
            _fail(f"{path}.committable", "must be false")
        evidence_records.append(node)
    _validate_sorted_ids(evidence_records, "observation.evidence")
    evidence = {node["id"]: node for node in evidence_records}
    if len(evidence) != len(evidence_records):
        _fail("observation.evidence", "contains duplicate ids")

    claim_records: list[dict[str, Any]] = []
    for index, item in enumerate(_array(record["claims"], "observation.claims")):
        path = f"observation.claims[{index}]"
        claim = _exact_keys(item, _CLAIM_KEYS, path)
        claim_id = _integer(claim["id"], f"{path}.id")
        owner_goal = _integer(claim["owner_goal"], f"{path}.owner_goal")
        if owner_goal not in goals or owner_goal >= claim_id:
            _fail(f"{path}.owner_goal", "must reference an earlier Goal")
        if goals[owner_goal]["status"] != GoalStatus.OPEN.value:
            _fail(f"{path}.owner_goal", "must reference an active Goal")
        kind = _enum(claim["kind"], ClaimKind, f"{path}.kind")
        payload, statement = _payload_statement(
            claim["payload"], kind, f"{path}.payload"
        )
        _validate_payload_owner(payload, kind, goals[owner_goal], path)
        premises = _validate_id_array(claim["premises"], f"{path}.premises")
        if kind in {
            ClaimKind.RESULT.value,
            ClaimKind.SOLUTION.value,
            ClaimKind.REFUTATION.value,
        } and premises:
            _fail(
                f"{path}.premises",
                f"{kind} claims must not have premises",
            )
        if kind == ClaimKind.AGGREGATE.value:
            owner = goals[owner_goal]
            children = children_by_parent.get(owner_goal, [])
            if (
                owner["connective"] != Connective.EXHAUSTIVE_CASE.value
                or not children
            ):
                _fail(path, "aggregate needs exhaustive child goals")
            if any(
                child["status"] != GoalStatus.SOLVED.value for child in children
            ):
                _fail(path, "aggregate needs committed child goals")
            expected_premises = [
                child["committed_claim"]["id"] for child in children
            ]
            if premises != expected_premises:
                _fail(
                    f"{path}.premises",
                    "differ from the committed child outcomes",
                )
        if claim["statement"] != statement:
            _fail(f"{path}.statement", "differs from the typed payload")
        status = _enum(claim["status"], ClaimStatus, f"{path}.status")
        result = claim["verifier_result"]
        verified = (
            _validate_verifier_result(result, f"{path}.verifier_result")
            if result is not None
            else None
        )
        discharge_record = claim["discharge_result"]
        discharge_result = (
            _validate_verifier_result(
                discharge_record, f"{path}.discharge_result"
            )
            if discharge_record is not None
            else None
        )
        discharge = _enum(
            claim["discharge_status"],
            DischargeStatus,
            f"{path}.discharge_status",
        )
        _validate_claim_status(
            status, verified, discharge_result, discharge, path
        )
        claim_records.append(claim)
    _validate_sorted_ids(claim_records, "observation.claims")
    claims = {claim["id"]: claim for claim in claim_records}
    if len(claims) != len(claim_records):
        _fail("observation.claims", "contains duplicate ids")

    all_ids = set(goals) | set(evidence) | set(claims) | set(compact_claims)
    expected_count = len(goals) + len(evidence) + len(claims) + len(compact_claims)
    if len(all_ids) != expected_count:
        _fail("observation", "node ids are not globally unique")
    premise_ids = set(evidence) | set(claims) | set(compact_claims)
    for collection_name, nodes in (
        ("evidence", evidence_records),
        ("claims", claim_records),
    ):
        for index, node in enumerate(nodes):
            for premise_index, premise in enumerate(node["premises"]):
                if premise not in premise_ids or premise >= node["id"]:
                    _fail(
                        f"observation.{collection_name}[{index}].premises[{premise_index}]",
                        "must reference an earlier Evidence or Claim",
                    )

    compact_claim_owners = {
        goal["committed_claim"]["id"]: goal["id"]
        for goal in goal_records
        if goal["committed_claim"] is not None
    }
    for index, claim in enumerate(claim_records):
        if claim["kind"] == ClaimKind.AGGREGATE.value:
            continue
        ancestry: set[int] = set()
        goal_id: int | None = claim["owner_goal"]
        while goal_id is not None:
            ancestry.add(goal_id)
            goal_id = goals[goal_id]["parent_goal"]
        for premise_index, premise in enumerate(claim["premises"]):
            if premise in evidence:
                scope_goal = evidence[premise]["scope_goal"]
            elif premise in claims:
                scope_goal = claims[premise]["owner_goal"]
            else:
                scope_goal = compact_claim_owners[premise]
            premise_path = (
                f"observation.claims[{index}].premises[{premise_index}]"
            )
            if scope_goal not in ancestry:
                _fail(premise_path, "is outside its owner ancestry")
            if claim["status"] != ClaimStatus.PRUNED.value and (
                premise in compact_claims
                or (
                    premise in claims
                    and claims[premise]["status"] == ClaimStatus.PRUNED.value
                )
            ):
                _fail(premise_path, "must reference an active premise")

    legal_records: list[dict[str, Any]] = []
    legal_keys: list[tuple[int, int]] = []
    for index, item in enumerate(
        _array(record["legal_pairs"], "observation.legal_pairs")
    ):
        path = f"observation.legal_pairs[{index}]"
        pair = _exact_keys(item, _ACTION_KEYS, path)
        verb = _enum(pair["verb"], Verb, f"{path}.verb")
        target = _integer(pair["target"], f"{path}.target")
        kind = _validate_action_target(
            verb,
            target,
            goals,
            evidence,
            claims,
            compact_claims,
            path,
            require_active=True,
        )
        if verb == Verb.DECOMPOSE.value and any(
            goal["parent_goal"] == target for goal in goal_records
        ):
            _fail(path, "decompose target is already decomposed")
        if verb == Verb.AGGREGATE.value:
            children = [
                goal for goal in goal_records if goal["parent_goal"] == target
            ]
            if not children or any(
                child["status"] != GoalStatus.SOLVED.value for child in children
            ):
                _fail(path, "aggregate requires every direct child Goal solved")
            if goals[target]["connective"] != Connective.EXHAUSTIVE_CASE.value:
                _fail(
                    path, "aggregate target must use exhaustive_case connective"
                )
            objective = goals[target]["obligation"]["objective"]
            if objective not in _OBJECTIVE_REDUCERS:
                _fail(path, "aggregate target has no objective reducer")
        if verb == Verb.VERIFY.value and claims[target]["status"] != ClaimStatus.UNCHECKED.value:
            _fail(path, "verify target must be unchecked")
        if verb == Verb.COMMIT.value and not (
            claims[target]["status"] == ClaimStatus.VERIFIED.value
            and claims[target]["discharge_status"] == DischargeStatus.ACCEPTED.value
        ):
            _fail(path, "commit target must be verified and owner-discharging")
        if verb == Verb.PRUNE.value and kind == "goal" and target == root_id:
            _fail(path, "prune cannot target the root Goal")
        legal_records.append(pair)
        legal_keys.append((_VERB_RANK[verb], target))
    if len(legal_keys) != len(set(legal_keys)):
        _fail("observation.legal_pairs", "contains duplicate actions")
    if legal_keys != sorted(legal_keys):
        _fail("observation.legal_pairs", "must follow fixed verb and target order")
    if turn == turn_limit and legal_records:
        _fail("observation.legal_pairs", "must be empty at the turn limit")
    if goals[root_id]["status"] != GoalStatus.OPEN.value and legal_records:
        _fail("observation.legal_pairs", "must be empty after the root closes")
    expected_legal = _expected_legal_pairs(
        goals, claims, root_id, turn, turn_limit
    )
    if legal_records != expected_legal:
        _fail("observation.legal_pairs", "does not match the canonical menu")

    previous = record["previous_outcome"]
    if previous is not None:
        outcome = _exact_keys(previous, _OUTCOME_KEYS, "observation.previous_outcome")
        action = _exact_keys(
            outcome["action"],
            _ACTION_KEYS,
            "observation.previous_outcome.action",
        )
        verb = _enum(
            action["verb"], Verb, "observation.previous_outcome.action.verb"
        )
        target = _integer(
            action["target"], "observation.previous_outcome.action.target"
        )
        if verb in {item.value for item in POLICY_VERB_ORDER}:
            target_kind = "goal"
            if target in evidence or target in claims or target in compact_claims:
                _fail(
                    "observation.previous_outcome.action.target",
                    "learned action target has a non-Goal record",
                )
        else:
            target_kind = _validate_action_target(
                verb,
                target,
                goals,
                evidence,
                claims,
                compact_claims,
                "observation.previous_outcome.action",
                require_active=False,
            )
        code = _enum(outcome["code"], OutcomeCode, "observation.previous_outcome.code")
        reason = outcome["reason"]
        if code == OutcomeCode.APPLIED.value:
            if reason is not None:
                _fail(
                    "observation.previous_outcome.reason",
                    "must be null for an applied outcome",
                )
        else:
            _enum(
                reason,
                RejectionReason,
                "observation.previous_outcome.reason",
            )
        verifier_context = outcome["verifier_context"]
        if verifier_context is None:
            if verb == Verb.VERIFY.value and code == OutcomeCode.APPLIED.value:
                _fail(
                    "observation.previous_outcome.verifier_context",
                    "is required for an applied VERIFY",
                )
        else:
            context = _validate_context_record(
                verifier_context,
                "observation.previous_outcome.verifier_context",
            )
            if verb != Verb.VERIFY.value or code != OutcomeCode.APPLIED.value:
                _fail(
                    "observation.previous_outcome.verifier_context",
                    "is only valid for an applied VERIFY",
                )
            if context["purpose"] != CheckPurpose.VALIDITY.value:
                _fail(
                    "observation.previous_outcome.verifier_context.purpose",
                    "must be validity",
                )
            if (
                target_kind != "claim"
                or claims[target]["status"] == ClaimStatus.PRUNED.value
            ):
                _fail(
                    "observation.previous_outcome.action.target",
                    "applied VERIFY must target a full active Claim record",
                )
            target_claim = claims[target]
            if context["claim_kind"] != target_claim["kind"]:
                _fail(
                    "observation.previous_outcome.verifier_context.claim_kind",
                    "differs from the target Claim",
                )
            owner = context["owner_goal"]
            if owner not in goals:
                _fail(
                    "observation.previous_outcome.verifier_context.owner_goal",
                    "does not reference a Goal",
                )
            if owner != target_claim["owner_goal"]:
                _fail(
                    "observation.previous_outcome.verifier_context.owner_goal",
                    "differs from the target Claim owner",
                )
            verifier_result = target_claim["verifier_result"]
            if verifier_result is None:
                _fail(
                    "observation.previous_outcome.verifier_context",
                    "requires the target Claim verifier result",
                )
            for field in ("contract_id", "context_digest"):
                if context[field] != verifier_result[field]:
                    _fail(
                        f"observation.previous_outcome.verifier_context.{field}",
                        "differs from the target Claim verifier result",
                    )


def canonical_observation_bytes(
    observation: object,
    generation_contract: GenerationContract,
    verifier_contract: VerifierReceiptContract,
    *,
    environment_contract_sha256: str,
) -> bytes:
    """Return canonical JSON bytes after environment-bound validation."""
    validate_observation_with_contract(
        observation,
        generation_contract,
        verifier_contract,
        environment_contract_sha256=environment_contract_sha256,
    )
    return json.dumps(
        observation,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def validate_observation_with_contract(
    observation: object,
    generation_contract: GenerationContract,
    verifier_contract: VerifierReceiptContract,
    *,
    environment_contract_sha256: str,
    turn_limit: int | None = None,
) -> None:
    """Validate canonical symbols and bounded receipts against pinned contracts."""
    validate_observation(observation)
    if not _is_sha256(environment_contract_sha256):
        raise TypeError("environment contract digest has an invalid type")
    if (
        not isinstance(observation, dict)
        or observation.get("environment_contract_sha256")
        != environment_contract_sha256
    ):
        _fail(
            "observation.environment_contract_sha256",
            "differs from the bound environment contract",
        )
    if turn_limit is not None and observation.get("turn_limit") != turn_limit:
        _fail(
            "observation.turn_limit",
            "differs from the bound environment contract",
        )
    if not isinstance(generation_contract, GenerationContract):
        raise TypeError("generation_contract has an invalid type")
    if not isinstance(verifier_contract, VerifierReceiptContract):
        raise TypeError("verifier_contract has an invalid type")
    if not isinstance(verifier_contract.contract_id, str) or not (
        verifier_contract.contract_id.strip()
    ):
        raise TypeError("verifier contract id has an invalid value")
    for name in (
        "validity_rejection_codes",
        "discharge_rejection_codes",
    ):
        codes = getattr(verifier_contract, name)
        if type(codes) is not frozenset or any(
            not isinstance(code, str) or not code.strip() for code in codes
        ):
            raise TypeError(f"verifier {name} have an invalid value")
    if not isinstance(observation, dict):
        raise AssertionError("validated observation must be an object")

    goals = {goal["id"]: goal for goal in observation["goals"]}
    goal_indexes = {
        goal["id"]: index for index, goal in enumerate(observation["goals"])
    }
    obligations = {
        goal_id: _typed_obligation(goal["obligation"])
        for goal_id, goal in goals.items()
    }
    for index, goal in enumerate(observation["goals"]):
        obligation = obligations[goal["id"]]
        try:
            canonical = generation_contract.canonicalize_obligation(obligation)
        except Exception as exc:
            _fail(
                f"observation.goals[{index}].obligation",
                f"is rejected by the generation contract: {exc}",
            )
        if canonical != obligation:
            _fail(
                f"observation.goals[{index}].obligation",
                "is not symbolically canonical",
            )

    def check_payload(
        claim: dict[str, Any], owner_goal: int, path: str
    ) -> None:
        kind = ClaimKind(claim["kind"])
        payload = _typed_payload(claim["payload"], kind)
        try:
            canonical = generation_contract.canonicalize_payload(
                obligations[owner_goal], kind, payload
            )
        except Exception as exc:
            _fail(
                f"{path}.payload",
                f"is rejected by the generation contract: {exc}",
            )
        if canonical != payload:
            _fail(f"{path}.payload", "is not symbolically canonical")

    for index, goal in enumerate(observation["goals"]):
        compact = goal["committed_claim"]
        if compact is not None:
            check_payload(
                compact,
                goal["id"],
                f"observation.goals[{index}].committed_claim",
            )
    for index, claim in enumerate(observation["claims"]):
        check_payload(
            claim, claim["owner_goal"], f"observation.claims[{index}]"
        )

    def check_receipts(claim: dict[str, Any], path: str) -> None:
        for field, codes in (
            (
                "verifier_result",
                verifier_contract.validity_rejection_codes,
            ),
            (
                "discharge_result",
                verifier_contract.discharge_rejection_codes,
            ),
        ):
            receipt = claim[field]
            if receipt is None:
                continue
            if receipt["contract_id"] != verifier_contract.contract_id:
                _fail(
                    f"{path}.{field}.contract_id",
                    "differs from the pinned verifier contract",
                )
            if not receipt["accepted"] and receipt["reason_code"] not in codes:
                _fail(
                    f"{path}.{field}.reason_code",
                    "is not allowed by the pinned verifier contract",
                )

    for index, goal in enumerate(observation["goals"]):
        compact = goal["committed_claim"]
        if compact is not None:
            check_receipts(
                compact, f"observation.goals[{index}].committed_claim"
            )
    for index, claim in enumerate(observation["claims"]):
        check_receipts(claim, f"observation.claims[{index}]")

    children_by_parent: dict[int, list[dict[str, Any]]] = {}
    for goal in observation["goals"]:
        parent = goal["parent_goal"]
        if parent is not None:
            children_by_parent.setdefault(parent, []).append(goal)
    for parent_id, children in children_by_parent.items():
        child_obligations = tuple(
            obligations[child["id"]] for child in children
        )
        parent_record = observation["goals"][goal_indexes[parent_id]]
        connective = parent_record["connective"]
        if connective is None:
            _fail(
                f"observation.goals[{goal_indexes[parent_id]}].connective",
                "decomposed Goal needs a connective",
            )
        try:
            canonical = generation_contract.validate_decomposition(
                obligations[parent_id],
                Connective(connective),
                child_obligations,
            )
        except Exception as exc:
            _fail(
                f"observation.goals[{goal_indexes[parent_id]}].connective",
                f"decomposition is rejected by the generation contract: {exc}",
            )
        if canonical != child_obligations:
            _fail(
                f"observation.goals[{goal_indexes[parent_id]}].connective",
                "child obligations are not symbolically canonical",
            )


def observation_sha256(
    observation: object,
    generation_contract: GenerationContract,
    verifier_contract: VerifierReceiptContract,
    *,
    environment_contract_sha256: str,
) -> str:
    """Return the SHA-256 digest of canonical observation bytes."""
    return hashlib.sha256(
        canonical_observation_bytes(
            observation,
            generation_contract,
            verifier_contract,
            environment_contract_sha256=environment_contract_sha256,
        )
    ).hexdigest()


__all__ = [
    "CONTROLLER_OBSERVATION_VERSION",
    "ObservationContext",
    "ObservationValidationError",
    "build_observation",
    "canonical_observation_bytes",
    "observation_sha256",
    "serialize_action",
    "serialize_obligation",
    "serialize_outcome",
    "serialize_payload",
    "serialize_verifier_context",
    "validate_observation",
    "validate_observation_with_contract",
]
