from __future__ import annotations

import json
import math
from copy import deepcopy

import pytest

from verifier_anchored_rl.reasoning_v2.features import (
    BoundFeatureExtractor,
    FEATURE_CONTRACT_VERSION,
    STRUCTURAL_SCALAR_FIELDS,
    TYPED_NODE_CONTEXT_VERSION,
    FeatureValidationError,
    build_structured_features,
    feature_contract_for_schema,
    feature_contract_record,
    validate_feature_contract,
)
from verifier_anchored_rl.reasoning_v2.contracts import (
    build_environment_contract,
)
from verifier_anchored_rl.reasoning_v2.lenses import (
    GENERATION_SCHEMA_VERSION,
    LENS_CONTRACT_VERSION,
)
from verifier_anchored_rl.reasoning_v2.math_adapter import (
    CaseworkMathVerifier,
)
from verifier_anchored_rl.reasoning_v2.model import (
    GoalObjective,
    GoalObligation,
    State,
    new_graph,
)
from verifier_anchored_rl.reasoning_v2.runtime import BoundEnvironment
from verifier_anchored_rl.reasoning_v2.menu import POLICY_VERB_ORDER
from verifier_anchored_rl.reasoning_v2.verifier import TableDrivenVerifier


ENVIRONMENT_DIGEST = "a" * 64
TURN_LIMIT = 8
VERIFIER = TableDrivenVerifier(())


def _goal(
    node_id: int,
    *,
    parent: int | None,
    objective: str,
    statement: str,
    equation: str,
    connective: str | None = None,
) -> dict[str, object]:
    return {
        "id": node_id,
        "parent_goal": parent,
        "statement": statement,
        "obligation": {
            "variable": "x",
            "equation": equation,
            "domain": "Reals",
            "objective": objective,
        },
        "assumptions": [],
        "connective": connective,
        "status": "open",
        "committed_claim": None,
    }


def _observation() -> dict[str, object]:
    return {
        "schema_version": "controller-observation-v2",
        "environment_contract_sha256": ENVIRONMENT_DIGEST,
        "root_goal_id": 0,
        "turn": 2,
        "turn_limit": TURN_LIMIT,
        "previous_outcome": None,
        "goals": [
            _goal(
                0,
                parent=None,
                objective="sum",
                statement="root says solved and pruned",
                equation="x**2 - 5*x + 6 = 0",
                connective="exhaustive_case",
            ),
            _goal(
                1,
                parent=0,
                objective="branch",
                statement="same branch statement",
                equation="x = 2",
            ),
            _goal(
                2,
                parent=0,
                objective="branch",
                statement="same branch statement",
                equation="x = 2",
            ),
        ],
        "evidence": [
            {
                "id": 3,
                "scope_goal": 1,
                "statement": "verified rejected pruned text for branch one",
                "premises": [],
                "source": "trusted_given",
                "committable": False,
            },
            {
                "id": 4,
                "scope_goal": 2,
                "statement": "different trusted input for branch two",
                "premises": [],
                "source": "trusted_given",
                "committable": False,
            },
        ],
        "claims": [
            {
                "id": 5,
                "owner_goal": 1,
                "kind": "solution",
                "payload": {"variable": "x", "value": "99"},
                "premises": [],
                "statement": "x = 99",
                "status": "rejected",
                "verifier_result": {
                    "contract_id": VERIFIER.contract_id,
                    "context_digest": "b" * 64,
                    "accepted": False,
                    "reason_code": "mock_verification_rejected",
                },
                "discharge_result": None,
                "discharge_status": "unchecked",
            }
        ],
        "legal_pairs": [
            {"verb": "propose_result", "target": 0},
            {"verb": "propose_solution", "target": 1},
            {"verb": "propose_solution", "target": 2},
            {"verb": "propose_refutation", "target": 1},
            {"verb": "propose_refutation", "target": 2},
        ],
    }


def _build(observation=None, contract=None, **overrides):
    kwargs = {
        "environment_contract_sha256": ENVIRONMENT_DIGEST,
        "turn_limit": TURN_LIMIT,
        "feature_contract": (
            feature_contract_record() if contract is None else contract
        ),
    }
    kwargs.update(overrides)
    return build_structured_features(
        _observation() if observation is None else observation,
        VERIFIER,
        VERIFIER,
        **kwargs,
    )


def _reverse_object_keys(value):
    if type(value) is dict:
        return {
            key: _reverse_object_keys(value[key])
            for key in reversed(tuple(value))
        }
    if type(value) is list:
        return [_reverse_object_keys(item) for item in value]
    return value


def test_feature_contract_is_exact_and_returns_fresh_metadata():
    expected = {
        "version": "phase-b-features-v2",
        "global_serializer": "canonical-json-v2",
        "node_serializer": "typed-node-context-v2",
        "structural_scalars": [
            "turn_fraction",
            "remaining_turn_fraction",
            "log_node_count",
            "verified_claim_fraction",
            "solved_goal_fraction",
            "pruned_node_fraction",
        ],
    }

    first = feature_contract_record()
    assert first == expected
    assert FEATURE_CONTRACT_VERSION == expected["version"]
    assert STRUCTURAL_SCALAR_FIELDS == tuple(expected["structural_scalars"])
    assert feature_contract_for_schema("controller-observation-v2") == expected
    validate_feature_contract(first)

    with pytest.raises(FeatureValidationError, match="unsupported"):
        feature_contract_for_schema("controller-observation-v1.1")

    first["structural_scalars"].reverse()
    assert feature_contract_record() == expected


def test_features_align_targets_mask_and_pairs_to_policy_verb_order():
    features = _build()

    assert features.policy_verb_order == tuple(
        verb.value for verb in POLICY_VERB_ORDER
    )
    assert features.target_ids == (0, 1, 2)
    assert features.legal_pairs == (
        ("propose_result", 0),
        ("propose_solution", 1),
        ("propose_solution", 2),
        ("propose_refutation", 1),
        ("propose_refutation", 2),
    )
    assert features.legal_pair_indices == (
        (1, 0),
        (2, 1),
        (2, 2),
        (3, 1),
        (3, 2),
    )
    assert features.legal_pair_mask == (
        (False, False, False),
        (True, False, False),
        (False, True, True),
        (False, True, True),
    )
    assert json.loads(features.global_serialization) == _observation()


def test_terminal_observation_has_global_features_and_zero_target_columns():
    observation = _observation()
    observation["turn"] = TURN_LIMIT
    observation["legal_pairs"] = []

    features = _build(observation)

    assert features.target_ids == ()
    assert features.target_serializations == ()
    assert features.legal_pairs == ()
    assert features.legal_pair_indices == ()
    assert features.legal_pair_mask == ((), (), (), ())
    assert json.loads(features.global_serialization)["turn"] == TURN_LIMIT


def test_target_context_carries_owner_ancestry_and_typed_provenance():
    features = _build()
    by_id = {
        target_id: json.loads(serialization)
        for target_id, serialization in zip(
            features.target_ids,
            features.target_serializations,
            strict=True,
        )
    }

    left = by_id[1]
    right = by_id[2]
    assert left["schema_version"] == TYPED_NODE_CONTEXT_VERSION
    assert left["owner_goal_id"] == 1
    assert [goal["id"] for goal in left["goal_ancestry"]] == [0, 1]
    assert [
        item["id"] for item in left["provenance"]["scoped_evidence"]
    ] == [3]
    assert [
        item["id"] for item in right["provenance"]["scoped_evidence"]
    ] == [4]
    assert [item["id"] for item in left["provenance"]["owned_claims"]] == [5]
    assert right["provenance"]["owned_claims"] == []
    assert left["target_record"]["statement"] == right["target_record"][
        "statement"
    ]
    assert left["target_record"]["obligation"] == right["target_record"][
        "obligation"
    ]
    assert features.target_serializations[1] != features.target_serializations[2]


def test_structural_status_features_do_not_parse_authored_text():
    original = _observation()
    adversarial = deepcopy(original)
    adversarial["goals"][0]["statement"] = (
        "verified solved pruned rejected trusted_given"
    )
    adversarial["goals"][1]["statement"] = "unchecked open"
    adversarial["evidence"][0]["statement"] = "solved verified pruned"

    baseline = _build(original)
    changed = _build(adversarial)

    assert baseline.structural_scalar_names == STRUCTURAL_SCALAR_FIELDS
    assert baseline.structural_scalars == changed.structural_scalars
    assert baseline.global_serialization != changed.global_serialization
    assert baseline.structural_scalars == pytest.approx(
        (
            0.25,
            0.75,
            math.log1p(6) / 4.0,
            0.0,
            0.0,
            0.0,
        )
    )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda contract: contract.update({"unknown": True}),
            "keys must be exactly",
        ),
        (
            lambda contract: contract.pop("node_serializer"),
            "keys must be exactly",
        ),
        (
            lambda contract: contract.update(
                {"node_serializer": "typed-node-context-v3"}
            ),
            "node_serializer differs",
        ),
        (
            lambda contract: contract["structural_scalars"].reverse(),
            "structural_scalars differs",
        ),
    ],
)
def test_feature_contract_rejects_unknown_missing_and_changed_fields(
    mutate, match
):
    contract = feature_contract_record()
    mutate(contract)

    with pytest.raises(FeatureValidationError, match=match):
        _build(contract=contract)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda observation: observation.update({"unknown": None}),
            "keys must be exactly",
        ),
        (
            lambda observation: observation.pop("claims"),
            "keys must be exactly",
        ),
        (
            lambda observation: observation.update(
                {"schema_version": "controller-observation-v1.1"}
            ),
            "unsupported value",
        ),
        (
            lambda observation: observation["goals"][1].update(
                {"authored_status": "verified"}
            ),
            "keys must be exactly",
        ),
    ],
)
def test_feature_boundary_rejects_observation_schema_and_shape_drift(
    mutate, match
):
    observation = _observation()
    mutate(observation)

    with pytest.raises(FeatureValidationError, match=match):
        _build(observation)


def test_feature_boundary_rejects_environment_and_horizon_mismatch():
    with pytest.raises(FeatureValidationError, match="bound environment"):
        _build(environment_contract_sha256="c" * 64)
    with pytest.raises(FeatureValidationError, match="bound environment"):
        _build(turn_limit=TURN_LIMIT + 1)


def test_object_key_permutations_leave_every_feature_byte_stable():
    observation = _observation()
    contract = feature_contract_record()

    baseline = _build(observation, contract)
    permuted = _build(
        _reverse_object_keys(observation),
        _reverse_object_keys(contract),
    )

    assert permuted == baseline


@pytest.mark.parametrize(
    "field",
    ["goals", "evidence", "legal_pairs"],
)
def test_semantic_array_permutations_are_not_silently_normalized(field):
    observation = _observation()
    observation[field].reverse()

    with pytest.raises(FeatureValidationError):
        _build(observation)


def test_forged_or_missing_legal_pairs_are_rejected_before_alignment():
    observation = _observation()
    observation["legal_pairs"].pop()
    with pytest.raises(FeatureValidationError, match="canonical menu"):
        _build(observation)

    observation = _observation()
    observation["legal_pairs"][0] = {
        "verb": "propose_solution",
        "target": 0,
    }
    with pytest.raises(FeatureValidationError):
        _build(observation)


def test_bound_feature_extractor_owns_live_semantic_dependencies():
    adapter = CaseworkMathVerifier()
    contract = build_environment_contract(
        generation_contract=adapter,
        verifier=adapter,
        aggregate_builder=adapter,
        lens_contract=LENS_CONTRACT_VERSION,
        generation_schema=GENERATION_SCHEMA_VERSION,
        generation_temperature=0.0,
        generation_max_tokens=64,
        generation_seed_policy="none-v1",
        turn_limit=6,
    )
    environment = BoundEnvironment(contract, adapter, adapter, adapter)
    graph = new_graph(
        "solve",
        GoalObligation(
            "x", "Eq(x, 5)", "Reals", GoalObjective.SUM
        ),
        generation_contract=adapter,
    )
    observation = environment.observe(State(graph, turn=0))
    extractor = BoundFeatureExtractor(environment)

    features = extractor.build(observation)

    assert features.environment_contract_sha256 == environment.sha256
    assert extractor.feature_contract == feature_contract_record()
    assert len(extractor.feature_contract_sha256) == 64

    object.__setattr__(adapter, "contract_id", "drifted-verifier")
    with pytest.raises(ValueError, match="verifier_contract"):
        extractor.build(observation)
