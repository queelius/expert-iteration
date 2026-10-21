from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from verifier_anchored_rl.controller_observation import (
    CONTROLLER_OBSERVATION_VERSION,
)
from verifier_anchored_rl.deliberation_mdp import (
    DeliberationMDP, DeliberationState, Verb,
)
from verifier_anchored_rl.heads import VERB_NAME_TO_INDEX
from verifier_anchored_rl.phase_b_heads import FactoredController
from verifier_anchored_rl.phase_b_features import ZeroTextFeatures
from verifier_anchored_rl.phase_b_features import feature_contract_for_schema
from verifier_anchored_rl.phase_b_policy import (
    LearnedController,
    OllamaBatchTextFeatures,
    load_learned_controller,
)
from verifier_anchored_rl.reductio import new_graph, split


class FakeEncoder:
    dim = 8

    def encode_many(self, texts):
        return [np.full(self.dim, index + 1, dtype=np.float32) for index, _ in enumerate(texts)]


def test_learned_controller_always_returns_a_legal_pair():
    mdp = DeliberationMDP(problem="p", expected_answer="1", llm_client=None)
    state = mdp.initial_state()
    model = FactoredController(14, hidden_dim=16, repr_dim=8, node_repr_dim=8)
    controller = LearnedController(model=model, encoder=FakeEncoder())
    action = controller.choose(mdp, state)
    assert action in mdp.actions(state)
    assert controller.last_choice_fallback is False
    assert controller.last_policy_record is not None


class FixedPolicyModel:
    def eval(self):
        return self

    def masked_policy(self, global_features, node_features, legal):
        _, verbs, nodes = legal.shape
        verb_prob = torch.zeros(1, verbs)
        pointer_prob = torch.zeros(1, verbs, nodes)
        split_index = VERB_NAME_TO_INDEX["split"]
        compute_index = VERB_NAME_TO_INDEX["compute"]
        verb_prob[0, split_index] = 0.6
        verb_prob[0, compute_index] = 0.4

        split_targets = torch.nonzero(legal[0, split_index]).flatten()
        pointer_prob[0, split_index, split_targets[0]] = 0.55
        pointer_prob[0, split_index, split_targets[1]] = 0.45

        compute_targets = torch.nonzero(legal[0, compute_index]).flatten()
        pointer_prob[0, compute_index, compute_targets[0]] = 0.95
        remainder = 0.05 / (len(compute_targets) - 1)
        pointer_prob[0, compute_index, compute_targets[1:]] = remainder
        return verb_prob, pointer_prob


def test_joint_greedy_can_differ_from_sequential_greedy():
    graph = new_graph("p")
    graph = split(
        graph, 0, "conjunctive", ("child A", "child B")).graph
    state = DeliberationState(graph, 0)
    mdp = DeliberationMDP(problem="p", expected_answer="1")

    joint = LearnedController(
        model=FixedPolicyModel(), encoder=FakeEncoder(), decision_rule="joint"
    ).choose(mdp, state)
    sequential = LearnedController(
        model=FixedPolicyModel(),
        encoder=FakeEncoder(),
        decision_rule="sequential",
    ).choose(mdp, state)

    assert joint.verb is Verb.COMPUTE
    assert joint.target == 0
    assert sequential.verb is Verb.SPLIT
    assert sequential.target == 1


def _write_controller_checkpoint(path, *, encoder, **metadata):
    model = FactoredController(
        14, hidden_dim=16, repr_dim=8, node_repr_dim=8)
    checkpoint = {
        "state_dict": model.state_dict(),
        "feature_dim": 14,
        "model_config": {
            "hidden_dim": 16,
            "repr_dim": 8,
            "node_repr_dim": 8,
        },
        "encoder": encoder,
        **metadata,
    }
    torch.save(checkpoint, path)


def test_checkpoint_contract_defaults_legacy_metadata_and_rejects_overrides(
    tmp_path,
):
    path = tmp_path / "legacy.pt"
    _write_controller_checkpoint(
        path,
        encoder={"type": "zero_text_ablation", "dimension": 8},
    )

    controller = load_learned_controller(
        str(path), ZeroTextFeatures(8))
    assert controller.action_semantics == "goal-commit-v1"

    assert controller.decision_rule == "sequential"
    assert controller.observation_schema == "tree-render-v1"
    with pytest.raises(ValueError, match="decision rule differs"):
        load_learned_controller(
            str(path), ZeroTextFeatures(8), decision_rule="joint")
    with pytest.raises(ValueError, match="observation schema differs"):
        load_learned_controller(
            str(path),
            ZeroTextFeatures(8),
            observation_schema=CONTROLLER_OBSERVATION_VERSION,
        )
    with pytest.raises(ValueError, match="feature dimension mismatch"):
        load_learned_controller(str(path), ZeroTextFeatures(9))


def test_checkpoint_contract_requires_the_pinned_encoder_digest(tmp_path):
    path = tmp_path / "frozen.pt"
    _write_controller_checkpoint(
        path,
        encoder={
            "type": "frozen_embedding_cache",
            "model": "nomic",
            "model_digest": "expected",
            "dimension": 8,
        },
        decision_rule="joint",
        observation_schema=CONTROLLER_OBSERVATION_VERSION,
    )

    class FrozenEncoder:
        dim = 8
        contract_type = "frozen_embedding_cache"
        model = "nomic"
        model_digest = "wrong"

    with pytest.raises(ValueError, match="model_digest mismatch"):
        load_learned_controller(str(path), FrozenEncoder())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("verb_order", ["wrong"], "verb order differs"),
        (
            "action_semantics",
            "claim-commit-v2",
            "unsupported action semantics",
        ),
    ],
)
def test_checkpoint_contract_rejects_action_algebra_mismatch(
    tmp_path, field, value, message,
):
    path = tmp_path / f"{field}.pt"
    metadata = {
        "decision_rule": "sequential",
        "observation_schema": "tree-render-v1",
        field: value,
    }
    _write_controller_checkpoint(
        path,
        encoder={"type": "zero_text_ablation", "dimension": 8},
        **metadata,
    )

    with pytest.raises(ValueError, match=message):
        load_learned_controller(str(path), ZeroTextFeatures(8))


def test_checkpoint_rejects_dimension_preserving_feature_contract_change(
    tmp_path,
):
    path = tmp_path / "feature-contract.pt"
    feature_contract = feature_contract_for_schema("tree-render-v1")
    feature_contract["node_serializer"] = "silently-changed"
    _write_controller_checkpoint(
        path,
        encoder={"type": "zero_text_ablation", "dimension": 8},
        feature_contract=feature_contract,
    )

    with pytest.raises(ValueError, match="feature serialization"):
        load_learned_controller(str(path), ZeroTextFeatures(8))


def test_live_mdp_contract_requires_exact_semantics_or_declared_shift():
    mdp = DeliberationMDP(problem="p", expected_answer="1")
    controller = LearnedController(
        model=FixedPolicyModel(),
        encoder=FakeEncoder(),
        training_mdp_contracts=(mdp.contract_record(),),
    )

    assert controller.validate_mdp_contract(mdp) is True
    shifted = DeliberationMDP(
        problem="p",
        expected_answer="1",
        generation_temperature=0.5,
    )
    with pytest.raises(ValueError, match="distribution differs"):
        controller.validate_mdp_contract(shifted)
    assert controller.validate_mdp_contract(
        shifted, allow_distribution_shift=True) is False

    semantic_mismatch = DeliberationMDP(
        problem="p",
        expected_answer="1",
        solve_claims=True,
    )
    with pytest.raises(ValueError, match="transition semantics"):
        controller.validate_mdp_contract(
            semantic_mismatch, allow_distribution_shift=True)


def test_ollama_encoder_pins_digest_reported_by_server():
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "models": [{
                    "name": "nomic:latest",
                    "digest": "served-digest",
                }]
            }

    class Client:
        def get(self, url):
            assert url.endswith("/api/tags")
            return Response()

    encoder = OllamaBatchTextFeatures(
        model="nomic:latest",
        model_digest="served-digest",
        dim=8,
    )
    encoder.client = Client()

    assert encoder.pin_served_model_digest() == "served-digest"

    encoder.model_digest = "incorrect"
    with pytest.raises(ValueError, match="differs from the Ollama server"):
        encoder.pin_served_model_digest()


def test_checkpoint_rejects_tampered_contract_manifest(tmp_path):
    path = tmp_path / "tampered-manifest.pt"
    _write_controller_checkpoint(
        path,
        encoder={"type": "zero_text_ablation", "dimension": 8},
        contract_manifest={"action_semantics": "goal-commit-v1"},
        contract_sha256="not-the-manifest-digest",
    )

    with pytest.raises(ValueError, match="manifest digest"):
        load_learned_controller(str(path), ZeroTextFeatures(8))
