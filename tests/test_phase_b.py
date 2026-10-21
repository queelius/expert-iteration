from __future__ import annotations

import json

import pytest

from verifier_anchored_rl.controller_observation import (
    CONTROLLER_OBSERVATION_VERSION,
    build_controller_observation,
)
from verifier_anchored_rl.deliberation_mdp import DeliberationMDP
from verifier_anchored_rl.deliberation_mdp import (
    LEGACY_MASK_PENALTY_REWARD,
)
from verifier_anchored_rl.phase_b import (
    BC_SCHEMA_VERSION,
    BC_SCHEMA_VERSION_V1_1,
    convert_runs,
    dataset_summary,
    parse_action_key,
    parse_state_render,
    validate_action_outcome,
)

from verifier_anchored_rl.reductio import compute

STATE = """State:
Problem: p
focus=0
[GOAL id=0] p connective=exhaustive_case
  [GOAL id=1 parent=0] case (evidence: node 2 verified)
    [FACT id=2 parent=1] 1 + 1 = 2 (verified)
[TURN 3/16]"""


def test_parse_state_render_preserves_pointer_features():
    nodes, turn = parse_state_render(STATE)
    assert turn == {"turn": 3, "limit": 16}
    assert nodes[0]["parent"] is None
    assert nodes[1]["depth"] == 1
    assert nodes[1]["evidence_id"] == 2
    assert nodes[2]["kind"] == "fact"
    assert nodes[2]["verified"] is True


def test_parse_action_key_rejects_unknown_or_noninteger_keys():
    assert parse_action_key("verify:2") == {"verb": "verify", "target": 2}
    with pytest.raises(ValueError, match="unknown verb"):
        parse_action_key("judge:2")
    with pytest.raises(ValueError, match="invalid action"):
        parse_action_key("verify:node-2")


def test_convert_runs_attaches_returns_and_excludes_fallback_from_bc(tmp_path):
    path = tmp_path / "run.jsonl"
    records = [
        {
            "episode": 7,
            "problem": "p",
            "turn": 3,
            "state_render": STATE,
            "legal": ["compute:0", "verify:2"],
            "chosen": "verify:2",
            "reward": -0.01,
            "masked": True,
            "terminal": False,
            "fallback": False,
        },
        {
            "episode": 7,
            "problem": "p",
            "turn": 4,
            "state_render": STATE.replace("[TURN 3/16]", "[TURN 4/16]"),
            "legal": ["commit:1", "abandon:0"],
            "chosen": "abandon:0",
            "reward": 1.0,
            "masked": False,
            "terminal": True,
            "fallback": True,
        },
        {
            "episode": 7,
            "summary": True,
            "total_reward": 0.99,
            "turns": 5,
            "llm_calls": 2,
            "meta": {"problem_index": 12},
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    rows = convert_runs([path])

    assert [row["mc_return"] for row in rows] == pytest.approx([0.98, 1.0])
    assert [row["gamma"] for row in rows] == [0.99, 0.99]
    assert rows[0]["bc_eligible"] is True
    assert rows[1]["bc_eligible"] is False
    assert rows[0]["schema_version"] == BC_SCHEMA_VERSION
    assert (
        rows[0]["mdp_contract"]["reward_contract"]
        == LEGACY_MASK_PENALTY_REWARD
    )
    assert rows[0]["problem_index"] == 12
    assert dataset_summary(rows) == {
        "decisions": 2,
        "episodes": 1,
        "winning_episodes": 1,
        "problem_indices": 1,
        "bc_eligible": 1,
        "winning_bc_eligible": 1,
        "masked": 1,
        "fallback": 1,
        "rows_with_unrendered_targets": 0,
        "chosen_unrendered_targets": 0,
    }


def test_convert_runs_rejects_chosen_pair_outside_menu(tmp_path):
    path = tmp_path / "bad.jsonl"
    decision = {
        "episode": 0,
        "problem": "p",
        "turn": 3,
        "state_render": STATE,
        "legal": ["compute:0"],
        "chosen": "verify:2",
        "reward": 0.0,
    }
    summary = {
        "episode": 0,
        "summary": True,
        "total_reward": 0.0,
        "turns": 4,
        "meta": {},
    }
    path.write_text(json.dumps(decision) + "\n" + json.dumps(summary) + "\n")
    with pytest.raises(ValueError, match="chosen pair is not legal"):
        convert_runs([path])


def test_convert_runs_keeps_unrendered_legal_target_as_placeholder(tmp_path):
    path = tmp_path / "compacted.jsonl"
    decision = {
        "episode": 0,
        "problem": "p",
        "turn": 3,
        "state_render": STATE,
        "legal": ["compute:0", "verify:9"],
        "chosen": "verify:9",
        "reward": 0.0,
        "fallback": False,
    }
    summary = {
        "episode": 0,
        "summary": True,
        "total_reward": 0.0,
        "turns": 4,
        "meta": {},
    }
    path.write_text(json.dumps(decision) + "\n" + json.dumps(summary) + "\n")

    rows = convert_runs([path])

    placeholder = next(node for node in rows[0]["nodes"] if node["id"] == 9)
    assert placeholder["rendered"] is False
    assert placeholder["kind"] == "verifiable"
    assert dataset_summary(rows)["chosen_unrendered_targets"] == 1


def test_convert_runs_uses_structured_observation_without_text_reparsing(tmp_path):
    mdp = DeliberationMDP(problem="p", expected_answer="4")
    state = mdp.initial_state()
    graph = compute(
        state.graph,
        0,
        "authored (verified) text",
        premise_ids=(),
    ).graph
    state = type(state)(graph, state.turn)
    observation = build_controller_observation(mdp, state)
    legal = [
        f"{pair['verb']}:{pair['target']}"
        for pair in observation["legal_pairs"]
    ]
    decision = {
        "episode": 0,
        "problem": "p",
        "turn": 0,
        "state_render": observation["state_text"],
        "controller_observation": observation,
        "legal": legal,
        "chosen": "verify:1",
        "outcome": {
            "action": "verify:1",
            "code": "applied",
            "reason": None,
        },
        "reward": 0.0,
        "fallback": False,
    }
    summary = {
        "episode": 0,
        "summary": True,
        "total_reward": 0.0,
        "turns": 1,
        "meta": {},
    }
    path = tmp_path / "structured.jsonl"
    path.write_text(
        json.dumps(decision) + "\n" + json.dumps(summary) + "\n")

    row = convert_runs([path])[0]
    fact = next(node for node in row["nodes"] if node["id"] == 1)

    assert row["schema_version"] == BC_SCHEMA_VERSION_V1_1
    assert row["serialization_schema"] == CONTROLLER_OBSERVATION_VERSION
    assert fact["verified"] is False
    assert all(node["rendered"] for node in row["nodes"])
    assert row["mdp_contract"] == mdp.contract_record()
    assert row["outcome"]["action"] == "verify:1"


def test_training_schema_is_inferred_and_cannot_be_user_stamped():
    from experiments.run_phase_b_bc import validate_observation_schema

    legacy = [{"serialization_schema": "tree-render-v1"}]
    assert validate_observation_schema(legacy, None) == "tree-render-v1"
    with pytest.raises(ValueError, match="requested schema"):
        validate_observation_schema(
            legacy, CONTROLLER_OBSERVATION_VERSION)
    with pytest.raises(ValueError, match="mixed or missing"):
        validate_observation_schema(
            [
                {"serialization_schema": "tree-render-v1"},
                {
                    "serialization_schema":
                    CONTROLLER_OBSERVATION_VERSION
                },
            ],
            None,
        )
    with pytest.raises(ValueError, match="mixed or missing"):
        validate_observation_schema([{}], None)


def test_training_contracts_are_inferred_not_hardcoded(tmp_path):
    from experiments.run_phase_b_bc import validate_training_contracts

    path = tmp_path / "run.jsonl"
    records = [
        {
            "episode": 0,
            "problem": "p",
            "turn": 3,
            "state_render": STATE,
            "legal": ["verify:2"],
            "chosen": "verify:2",
            "reward": 0.0,
            "fallback": False,
        },
        {
            "episode": 0,
            "summary": True,
            "total_reward": 0.0,
            "turns": 4,
            "meta": {},
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records))
    rows = convert_runs([path], gamma=1.0)

    contract = validate_training_contracts(rows)

    assert contract["action_semantics"] == "goal-commit-v1"
    assert contract["gamma"] == 1.0
    assert contract["mdp_contracts"] == [rows[0]["mdp_contract"]]
    rows[0]["action_semantics"] = "claim-commit-v2"
    with pytest.raises(ValueError, match="action semantics"):
        validate_training_contracts(rows)


def test_replay_rejects_unbounded_action_outcome():
    with pytest.raises(ValueError, match="unknown rejection reason"):
        validate_action_outcome(
            {
                "action": "compute:0",
                "code": "rejected",
                "reason": "free-form model feedback",
            },
            allow_none=False,
        )
