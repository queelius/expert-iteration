"""Phase B trajectory schema and conversion utilities.

Rung-1 logs are optimized for auditability.  Phase B needs a stricter,
versioned row per decision so encoders and factored heads can consume the same
contract without reinterpreting the raw JSONL on every run.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .controller_observation import (
    CONTROLLER_OBSERVATION_VERSION,
    controller_feature_text,
)
from .deliberation_mdp import Verb
from .deliberation_mdp import (
    ACTION_TARGET_KINDS,
    ACTION_SEMANTICS_VERSION,
    APPLY_SEMANTICS_VERSION,
    LEGACY_MASK_PENALTY_REWARD,
    PREVIOUS_OUTCOME_SCHEMA,
    RejectionReason,
    VERIFIER_CONTRACT_VERSION,
)


BC_SCHEMA_VERSION = "phase-b-v1"
BC_SCHEMA_VERSION_V1_1 = "phase-b-v1.1"
SERIALIZATION_SCHEMA = "tree-render-v1"

_NODE_LINE = re.compile(
    r"^(?P<indent> *)\[(?P<kind>[A-Z_]+) id=(?P<id>\d+)"
    r"(?: parent=(?P<parent>\d+))?\] (?P<text>.*)$"
)
_EVIDENCE = re.compile(r"\(evidence: node (?P<id>\d+) verified\)")
_TURN = re.compile(r"\[TURN (?P<turn>\d+)(?:/(?P<limit>\d+))?\]$")
_VERBS = frozenset(verb.value for verb in Verb)
_VERB_ORDER = [verb.value for verb in Verb]
_OUTCOME_REASONS = frozenset(reason.value for reason in RejectionReason)


def parse_action_key(key: str) -> dict[str, Any]:
    """Parse the stable ``verb:integer-target`` wire key."""
    try:
        verb, target_text = key.rsplit(":", 1)
        target = int(target_text)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"invalid action key {key!r}") from exc
    if verb not in _VERBS:
        raise ValueError(f"unknown verb in action key {key!r}")
    return {"verb": verb, "target": target}


def validate_action_outcome(
    value: dict[str, Any] | None, *, allow_none: bool
) -> None:
    """Validate the bounded serialized outcome before replay accepts it."""
    if value is None:
        if allow_none:
            return
        raise ValueError("current action outcome is missing")
    if set(value) != {"action", "code", "reason"}:
        raise ValueError("action outcome has an invalid shape")
    parse_action_key(value["action"])
    if value["code"] == "applied":
        if value["reason"] is not None:
            raise ValueError("applied action outcome has a rejection reason")
    elif value["code"] == "rejected":
        if value["reason"] not in _OUTCOME_REASONS:
            raise ValueError("action outcome has an unknown rejection reason")
    else:
        raise ValueError("action outcome has an unknown code")


def _legacy_mdp_contract(
    meta: dict[str, Any], turn_limit: int | None
) -> dict[str, Any]:
    """Best recoverable contract for archived tree-render trajectories."""
    return {
        "action_semantics": ACTION_SEMANTICS_VERSION,
        "action_target_kinds": {
            verb: list(kinds)
            for verb, kinds in ACTION_TARGET_KINDS.items()
        },
        "transition_semantics": "deliberation-transition-v1-archive",
        "apply_semantics": APPLY_SEMANTICS_VERSION,
        "previous_outcome_schema": "runner-feedback-v0",
        "verifier_contract": VERIFIER_CONTRACT_VERSION,
        "verifier_implementation": (
            "verifier_anchored_rl.verifier_adapter.MathVerifier"
        ),
        "reward_contract": LEGACY_MASK_PENALTY_REWARD,
        "depth_limit": turn_limit,
        "solve_claims": meta.get("solve_claims") == "on",
        "trusted_givens_committable": True,
        "reject_repeated_verify": False,
        "active_subtrees_only": False,
        "render_committed_outcomes": False,
        "generation_temperature": 1.0,
    }


def parse_state_render(state_render: str) -> tuple[list[dict[str, Any]], dict[str, int | None]]:
    """Extract typed node features and the explicit turn budget.

    ``text`` intentionally retains status annotations.  They are observable
    controller features, while the parallel booleans make the same information
    available to non-text encoders without brittle suffix parsing.
    """
    nodes: list[dict[str, Any]] = []
    turn_info: dict[str, int | None] | None = None
    for line in state_render.splitlines():
        marker = _NODE_LINE.match(line)
        if marker:
            text = marker.group("text")
            evidence = _EVIDENCE.search(text)
            nodes.append(
                {
                    "id": int(marker.group("id")),
                    "kind": marker.group("kind").lower(),
                    "parent": (
                        int(marker.group("parent"))
                        if marker.group("parent") is not None
                        else None
                    ),
                    "depth": len(marker.group("indent")) // 2,
                    "text": text,
                    "rendered": True,
                    "verified": "(verified)" in text,
                    "verify_failed": "(verify failed" in text,
                    "solved": "(solved)" in text,
                    "abandoned": "(abandoned)" in text,
                    "evidence_id": int(evidence.group("id")) if evidence else None,
                }
            )
            continue
        turn = _TURN.match(line)
        if turn:
            turn_info = {
                "turn": int(turn.group("turn")),
                "limit": int(turn.group("limit")) if turn.group("limit") else None,
            }
    if not nodes:
        raise ValueError("state render contains no node markers")
    if turn_info is None:
        raise ValueError("state render contains no turn marker")
    ids = [node["id"] for node in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("state render contains duplicate node ids")
    return nodes, turn_info


def complete_nodes_for_legal_pairs(
    nodes: list[dict[str, Any]], legal_pairs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Add explicit placeholders for legal targets hidden by compaction."""
    complete = list(nodes)
    node_ids = {node["id"] for node in complete}
    for target in sorted({pair["target"] for pair in legal_pairs} - node_ids):
        target_verbs = {
            pair["verb"] for pair in legal_pairs if pair["target"] == target
        }
        complete.append(
            {
                "id": target,
                "kind": (
                    "verifiable"
                    if target_verbs == {Verb.VERIFY.value}
                    else "unrendered"
                ),
                "parent": None,
                "depth": -1,
                "text": "",
                "rendered": False,
                "verified": False,
                "verify_failed": False,
                "solved": False,
                "abandoned": False,
                "evidence_id": None,
            }
        )
    return complete


def _read_run(path: Path) -> tuple[dict[int, list[dict[str, Any]]], dict[int, dict[str, Any]]]:
    decisions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    summaries: dict[int, dict[str, Any]] = {}
    with path.open() as fh:
        for line_number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            episode = int(record["episode"])
            if record.get("summary"):
                if episode in summaries:
                    raise ValueError(f"{path}:{line_number}: duplicate episode summary")
                summaries[episode] = record
            else:
                decisions[episode].append(record)
    missing = set(decisions) - set(summaries)
    if missing:
        raise ValueError(f"{path}: episodes without summaries: {sorted(missing)}")
    return decisions, summaries


def convert_runs(paths: Iterable[Path], *, gamma: float = 0.99) -> list[dict[str, Any]]:
    """Convert Rung-1 logs to versioned decision rows.

    Returns are computed within each source episode.  A chosen action selected
    by the controller's fallback is retained for value learning and audit, but
    ``bc_eligible`` is false because it is not a teacher label.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    rows: list[dict[str, Any]] = []
    for input_path in paths:
        path = Path(input_path)
        decisions, summaries = _read_run(path)
        for episode in sorted(decisions):
            episode_rows = sorted(decisions[episode], key=lambda row: int(row["turn"]))
            summary = summaries[episode]
            meta = summary.get("meta", {})
            problem_index = int(meta.get("problem_index", episode))
            running_return = 0.0
            mc_returns: list[float] = []
            for decision in reversed(episode_rows):
                running_return = float(decision["reward"]) + gamma * running_return
                mc_returns.append(running_return)
            mc_returns.reverse()
            won = float(summary["total_reward"]) > 0.0
            for decision, mc_return in zip(episode_rows, mc_returns, strict=True):
                legal_pairs = [parse_action_key(key) for key in decision["legal"]]
                observation = decision.get("controller_observation")
                if observation is not None:
                    if observation.get("schema_version") != \
                            CONTROLLER_OBSERVATION_VERSION:
                        raise ValueError(
                            f"{path}: unsupported controller observation "
                            f"{observation.get('schema_version')!r}")
                    observed_legal = observation["legal_pairs"]
                    if observed_legal != legal_pairs:
                        raise ValueError(
                            f"{path}: logged menu differs from structured "
                            "controller observation")
                    nodes = observation["nodes"]
                    mdp_contract = observation.get("mdp_contract")
                    if not isinstance(mdp_contract, dict):
                        raise ValueError(
                            f"{path}: structured observation has no "
                            "MDP contract")
                    if (
                        mdp_contract.get("action_semantics")
                        != ACTION_SEMANTICS_VERSION
                    ):
                        raise ValueError(
                            f"{path}: unsupported action semantics "
                            f"{mdp_contract.get('action_semantics')!r}")
                    if (
                        mdp_contract.get("previous_outcome_schema")
                        != PREVIOUS_OUTCOME_SCHEMA
                    ):
                        raise ValueError(
                            f"{path}: unsupported outcome schema "
                            f"{mdp_contract.get('previous_outcome_schema')!r}")
                    validate_action_outcome(
                        observation.get("previous_outcome"),
                        allow_none=True,
                    )
                    turn_info = {
                        "turn": int(observation["turn"]),
                        "limit": observation["turn_limit"],
                    }
                    state_text = controller_feature_text(observation)
                    schema_version = BC_SCHEMA_VERSION_V1_1
                    serialization_schema = CONTROLLER_OBSERVATION_VERSION
                    previous_outcome_schema = PREVIOUS_OUTCOME_SCHEMA
                else:
                    nodes, turn_info = parse_state_render(
                        decision["state_render"])
                    # Preserve historical hidden targets as placeholders only
                    # for archive reproducibility.
                    nodes = complete_nodes_for_legal_pairs(nodes, legal_pairs)
                    state_text = decision["state_render"]
                    schema_version = BC_SCHEMA_VERSION
                    serialization_schema = SERIALIZATION_SCHEMA
                    mdp_contract = _legacy_mdp_contract(
                        meta, turn_info["limit"])
                    previous_outcome_schema = "runner-feedback-v0"
                chosen = parse_action_key(decision["chosen"])
                if chosen not in legal_pairs:
                    raise ValueError(
                        f"{path}: episode {episode} turn {decision['turn']}: "
                        "chosen pair is not legal"
                    )
                if turn_info["turn"] != int(decision["turn"]):
                    raise ValueError(
                        f"{path}: episode {episode}: turn field/render mismatch"
                    )
                fallback = bool(decision.get("fallback", False))
                outcome = decision.get("outcome")
                if observation is not None:
                    validate_action_outcome(outcome, allow_none=False)
                    if parse_action_key(outcome["action"]) != chosen:
                        raise ValueError(
                            f"{path}: action outcome differs from chosen pair")
                rows.append(
                    {
                        "schema_version": schema_version,
                        "serialization_schema": serialization_schema,
                        "action_semantics": ACTION_SEMANTICS_VERSION,
                        "verb_order": _VERB_ORDER,
                        "mdp_contract": mdp_contract,
                        "previous_outcome_schema": previous_outcome_schema,
                        "source_run": path.name,
                        "episode_key": f"{path.stem}:{episode}",
                        "episode": episode,
                        "problem_index": problem_index,
                        "problem": decision["problem"],
                        "turn": int(decision["turn"]),
                        "turn_limit": turn_info["limit"],
                        "state_text": state_text,
                        "nodes": nodes,
                        "legal_pairs": legal_pairs,
                        "gamma": gamma,
                        "chosen": chosen,
                        "outcome": outcome,
                        "reward": float(decision["reward"]),
                        "mc_return": mc_return,
                        "masked": bool(decision.get("masked", False)),
                        "terminal": bool(decision.get("terminal", False)),
                        "fallback": fallback,
                        "bc_eligible": not fallback,
                        "episode_return": float(summary["total_reward"]),
                        "episode_won": won,
                        "episode_turns": int(summary["turns"]),
                        "meta": meta,
                    }
                )
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> int:
    """Write converted rows and return the number written."""
    count = 0
    with Path(path).open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def dataset_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    episodes = {row["episode_key"] for row in materialized}
    winning_episodes = {
        row["episode_key"] for row in materialized if row["episode_won"]
    }
    return {
        "decisions": len(materialized),
        "episodes": len(episodes),
        "winning_episodes": len(winning_episodes),
        "problem_indices": len({row["problem_index"] for row in materialized}),
        "bc_eligible": sum(bool(row["bc_eligible"]) for row in materialized),
        "winning_bc_eligible": sum(
            bool(row["bc_eligible"] and row["episode_won"])
            for row in materialized
        ),
        "masked": sum(bool(row["masked"]) for row in materialized),
        "fallback": sum(bool(row["fallback"]) for row in materialized),
        "rows_with_unrendered_targets": sum(
            any(not node["rendered"] for node in row["nodes"])
            for row in materialized
        ),
        "chosen_unrendered_targets": sum(
            any(
                node["id"] == row["chosen"]["target"] and not node["rendered"]
                for node in row["nodes"]
            )
            for row in materialized
        ),
    }


__all__ = [
    "BC_SCHEMA_VERSION",
    "BC_SCHEMA_VERSION_V1_1",
    "SERIALIZATION_SCHEMA",
    "convert_runs",
    "complete_nodes_for_legal_pairs",
    "dataset_summary",
    "parse_action_key",
    "parse_state_render",
    "validate_action_outcome",
    "write_jsonl",
]
