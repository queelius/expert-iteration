"""Lossless, versioned controller observations built directly from Graph."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from .deliberation_mdp import (
    _has_closed_ancestor,
    _is_open_goal,
    _parent_goal_id,
    _verification_context,
    _verification_rejected,
    _witnesses,
)
from .lenses import global_prefix
from .reductio import (
    Fact,
    Goal,
    Hypothesis,
    VerifierCall,
    get,
    is_abandoned,
    is_verified,
    solved,
)


CONTROLLER_OBSERVATION_VERSION = "controller-observation-v1.1"



def _owner_path(graph, node_id: int) -> tuple[int, ...]:
    path = []
    current: int | None = node_id
    while current is not None:
        path.append(current)
        current = _parent_goal_id(graph, current)
    return tuple(reversed(path))


def build_controller_observation(
    mdp: Any,
    state: Any,
) -> dict[str, Any]:
    """Return one canonical observation shared by collection and inference."""
    graph = state.graph
    actions = list(mdp.actions(state))
    legal_by_target: dict[int, list[str]] = defaultdict(list)
    legal_pairs = []
    for action in actions:
        legal_by_target[action.target].append(action.verb.value)
        legal_pairs.append({
            "verb": action.verb.value,
            "target": action.target,
        })

    calls_by_target: dict[int, list[VerifierCall]] = defaultdict(list)
    verifier_calls = []
    for node in graph.nodes:
        if not isinstance(node, VerifierCall):
            continue
        if (_has_closed_ancestor(graph, node.target)
                and node.target not in legal_by_target):
            continue
        calls_by_target[node.target].append(node)
        verifier_calls.append({
            "id": node.id,
            "target": node.target,
            "verb": node.verb,
            "expected": node.expected,
            "success": node.success,
            "observed": node.observed,
        })

    nodes = []
    for node in graph.nodes:
        if isinstance(node, VerifierCall):
            continue
        if (_has_closed_ancestor(graph, node.id)
                and node.id not in legal_by_target):
            continue
        parent = _parent_goal_id(graph, node.id)
        path = _owner_path(graph, node.id)
        owner_goal = node.id if isinstance(node, Goal) else parent
        abandoned = is_abandoned(graph, node.id)
        verified = isinstance(node, (Fact, Hypothesis)) and (
            is_verified(graph, node.id)
            or (isinstance(node, Fact) and node.source == "given")
        )
        verify_verb, verify_expected = _verification_context(
            graph, node.id, mdp.solve_claims)
        rejected = _verification_rejected(
            graph, node.id, verify_verb, verify_expected)
        node_solved = isinstance(node, Goal) and solved(graph, node.id)
        evidence_id = None
        outcome_id = None
        outcome_statement = None
        if isinstance(node, Goal):
            witnesses = _witnesses(
                graph, node, mdp.trusted_givens_committable)
            evidence_id = witnesses[-1] if witnesses else None
            outcome_id = dict(graph.solved_by).get(node.id)
            if outcome_id is not None:
                outcome_statement = get(graph, outcome_id).statement

        if isinstance(node, Goal):
            if abandoned:
                status = "pruned"
            elif node_solved:
                status = "solved"
            else:
                status = "open"
        elif abandoned:
            status = "pruned"
        elif verified:
            status = "verified"
        elif rejected:
            status = "rejected"
        else:
            status = "unchecked"

        record = {
            "id": node.id,
            "kind": type(node).__name__.lower(),
            "parent": parent,
            "owner_goal": owner_goal,
            "owner_path": list(path),
            "owner_path_text": [
                get(graph, item).statement
                for item in path
                if isinstance(get(graph, item), Goal)
            ],
            "depth": len(path) - 1,
            "statement": node.statement,
            "text": node.statement,
            "rendered": True,
            "active": (
                _is_open_goal(graph, node.id)
                if isinstance(node, Goal)
                else not abandoned
            ),
            "status": status,
            "verified": verified,
            "verify_failed": rejected,
            "solved": node_solved,
            "abandoned": abandoned,
            "evidence_id": evidence_id,
            "outcome_id": outcome_id,
            "outcome_statement": outcome_statement,
            "legal_verbs": legal_by_target.get(node.id, []),
            "verification_attempts": len(calls_by_target.get(node.id, [])),
            "last_verifier_observed": (
                calls_by_target[node.id][-1].observed
                if calls_by_target.get(node.id) else None
            ),
            "source": node.source if isinstance(node, Fact) else None,
            "premises": list(node.premises) if isinstance(node, Fact) else [],
            "assumptions": (
                list(node.assumptions) if isinstance(node, Goal) else []
            ),
            "connective": node.connective if isinstance(node, Goal) else None,
        }
        nodes.append(record)

    state_text = global_prefix(
        state,
        mdp.depth_limit,
        include_committed_outcomes=mdp.render_committed_outcomes,
    )
    return {
        "schema_version": CONTROLLER_OBSERVATION_VERSION,
        "problem": mdp.problem,
        "root_id": graph.root_id,
        "turn": state.turn,
        "turn_limit": mdp.depth_limit,
        "mdp_contract": mdp.contract_record(),
        "previous_outcome": (
            state.previous_outcome.as_record()
            if state.previous_outcome is not None
            else None
        ),
        "state_text": state_text,
        "nodes": nodes,
        "legal_pairs": legal_pairs,
        "verifier_calls": verifier_calls,
    }



def controller_feature_text(observation: dict[str, Any]) -> str:
    """Stable serialization for a frozen text encoder."""
    payload = dict(observation)
    payload.pop("state_text", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


__all__ = [
    "CONTROLLER_OBSERVATION_VERSION",
    "build_controller_observation",
    "controller_feature_text",
]
