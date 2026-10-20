"""Greedy B=0 learned controller over live deliberation states."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .controller_observation import (
    CONTROLLER_OBSERVATION_VERSION,
    build_controller_observation,
    controller_feature_text,
)
from .deliberation_mdp import (
    ACTION_SEMANTICS_VERSION,
    PREVIOUS_OUTCOME_SCHEMA,
)
from .heads import NUM_VERBS, VERB_NAME_TO_INDEX, VERB_ORDER
from .phase_b import complete_nodes_for_legal_pairs, parse_state_render
from .phase_b_features import (
    STRUCTURAL_FEATURE_DIM,
    feature_contract_for_schema,
    node_feature_text,
    structural_scalar_features,
)
from .phase_b_heads import FactoredController


class OllamaBatchTextFeatures:
    """Frozen Ollama embeddings with an in-process content cache."""
    contract_type = "frozen_embedding_cache"

    def __init__(
        self,
        *,
        model: str = "nomic-embed-text:latest",
        base_url: str = "http://localhost:11434",
        dim: int = 768,
        batch_size: int = 32,
        timeout: float = 120.0,
        model_digest: str | None = None,
    ):
        import httpx

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dim = dim
        self.model_digest = model_digest
        self.batch_size = batch_size
        self.client = httpx.Client(timeout=timeout)
        self._cache: dict[str, np.ndarray] = {}

    def pin_served_model_digest(self) -> str:
        """Resolve and verify the digest reported by the Ollama server."""
        response = self.client.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        matches = [
            item
            for item in response.json().get("models", ())
            if item.get("name") == self.model
            or item.get("model") == self.model
        ]
        if len(matches) != 1 or not matches[0].get("digest"):
            raise ValueError(
                f"cannot resolve one digest for Ollama model {self.model!r}")
        served_digest = str(matches[0]["digest"])
        if (
            self.model_digest is not None
            and self.model_digest != served_digest
        ):
            raise ValueError(
                "configured model digest differs from the Ollama server")
        self.model_digest = served_digest
        return served_digest

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def encode_many(self, texts: list[str]) -> list[np.ndarray]:
        missing: dict[str, str] = {}
        for text in texts:
            key = self._key(text)
            if key not in self._cache:
                missing[key] = text
        items = list(missing.items())
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            response = self.client.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model,
                    "input": [text for _, text in batch],
                    "truncate": False,
                },
            )
            response.raise_for_status()
            embeddings = np.asarray(response.json()["embeddings"], dtype=np.float32)
            if embeddings.shape != (len(batch), self.dim):
                raise ValueError(
                    f"embedding shape {embeddings.shape} != {(len(batch), self.dim)}"
                )
            for (key, _), vector in zip(batch, embeddings, strict=True):
                self._cache[key] = vector
        return [self._cache[self._key(text)] for text in texts]


@dataclass
class LearnedController:
    model: Any
    encoder: Any
    depth_limit: int = 16
    decision_rule: str = "joint"
    observation_schema: str = CONTROLLER_OBSERVATION_VERSION
    action_semantics: str = ACTION_SEMANTICS_VERSION
    feature_contract: dict[str, Any] = field(default_factory=dict)
    training_mdp_contracts: tuple[dict[str, Any], ...] = ()
    training_target_contract: dict[str, Any] = field(default_factory=dict)
    contract_sha256: str | None = None
    last_choice_fallback: bool = False
    last_raw_response: object = None
    last_policy_record: dict[str, Any] | None = field(default=None, init=False)
    last_controller_observation: dict[str, Any] | None = field(
        default=None, init=False)

    def validate_mdp_contract(
        self,
        mdp: Any,
        *,
        allow_distribution_shift: bool = False,
        allow_unbound_legacy: bool = False,
    ) -> bool:
        """Return exact-match status, rejecting silent semantic drift."""
        runtime = mdp.contract_record()
        if runtime["action_semantics"] != self.action_semantics:
            raise ValueError("runtime action semantics differ from checkpoint")
        if not self.training_mdp_contracts:
            if (
                allow_unbound_legacy
                and self.observation_schema == "tree-render-v1"
            ):
                return False
            raise ValueError(
                "checkpoint has no bound training MDP contract; "
                "legacy use requires an explicit override")

        distribution_keys = {
            "generation_temperature",
            "render_committed_outcomes",
        }
        semantic_runtime = {
            key: value for key, value in runtime.items()
            if key not in distribution_keys
        }
        semantic_matches = [
            contract
            for contract in self.training_mdp_contracts
            if {
                key: value for key, value in contract.items()
                if key not in distribution_keys
            } == semantic_runtime
        ]
        if not semantic_matches:
            raise ValueError(
                "runtime transition semantics differ from checkpoint data")
        if runtime in semantic_matches:
            return True
        if allow_distribution_shift:
            return False
        raise ValueError(
            "runtime generator/lens distribution differs from checkpoint "
            "data; declare the shift explicitly")

    def choose(self, mdp: Any, state: Any, *, rng=None) -> Any:
        import torch

        actions = list(mdp.actions(state))
        if not actions:
            raise ValueError("no legal actions; state is terminal")
        observation = build_controller_observation(mdp, state)
        self.last_controller_observation = observation
        legal_pairs = observation["legal_pairs"]
        if self.observation_schema == CONTROLLER_OBSERVATION_VERSION:
            state_text = controller_feature_text(observation)
            nodes = observation["nodes"]
        elif self.observation_schema == "tree-render-v1":
            state_text = observation["state_text"]
            nodes, _ = parse_state_render(state_text)
            nodes = complete_nodes_for_legal_pairs(nodes, legal_pairs)
        else:
            raise ValueError(
                f"unsupported observation schema {self.observation_schema!r}")
        node_index = {node["id"]: index for index, node in enumerate(nodes)}
        row = {
            "turn": state.turn,
            "turn_limit": mdp.depth_limit,
            "nodes": nodes,
        }
        scalars = structural_scalar_features(row)
        texts = [state_text] + [node_feature_text(node) for node in nodes]
        vectors = self.encoder.encode_many(texts)
        global_features = torch.from_numpy(
            np.concatenate((vectors[0], scalars)).astype(np.float32)
        ).unsqueeze(0)
        node_features = torch.from_numpy(
            np.stack(
                [np.concatenate((vector, scalars)) for vector in vectors[1:]]
            ).astype(np.float32)
        ).unsqueeze(0)
        legal = torch.zeros(1, NUM_VERBS, len(nodes), dtype=torch.bool)
        by_pair = {}
        for action in actions:
            verb_index = VERB_NAME_TO_INDEX[action.verb.value]
            target_index = node_index[action.target]
            legal[0, verb_index, target_index] = True
            by_pair[(verb_index, target_index)] = action
        self.model.eval()
        with torch.no_grad():
            verb_prob, pointer_prob = self.model.masked_policy(
                global_features, node_features, legal
            )
        joint_prob = verb_prob.unsqueeze(-1) * pointer_prob
        if self.decision_rule == "joint":
            flat_index = int(joint_prob[0].reshape(-1).argmax())
            verb_index, target_index = divmod(flat_index, len(nodes))
        elif self.decision_rule == "sequential":
            verb_index = int(verb_prob[0].argmax())
            target_index = int(pointer_prob[0, verb_index].argmax())
        else:
            raise ValueError(
                f"unknown decision rule {self.decision_rule!r}; "
                "expected 'joint' or 'sequential'")
        action = by_pair[(verb_index, target_index)]
        self.last_choice_fallback = False
        self.last_raw_response = None
        self.last_policy_record = {
            "chosen_verb_probability": float(verb_prob[0, verb_index]),
            "chosen_target_probability": float(
                pointer_prob[0, verb_index, target_index]
            ),
            "chosen_joint_probability": float(
                joint_prob[0, verb_index, target_index]
            ),
            "decision_rule": self.decision_rule,
            "verb_probabilities": {
                VERB_ORDER[index].value: float(probability)
                for index, probability in enumerate(verb_prob[0])
                if float(probability) > 0.0
            },
            "target_rendered": bool(nodes[target_index]["rendered"]),
        }
        return action


def load_learned_controller(
    checkpoint_path: str, encoder: Any, *, depth_limit: int = 16,
    decision_rule: str | None = None,
    observation_schema: str | None = None,
) -> LearnedController:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected_feature_dim = int(encoder.dim) + STRUCTURAL_FEATURE_DIM
    if int(checkpoint["feature_dim"]) != expected_feature_dim:
        raise ValueError(
            "encoder/checkpoint feature dimension mismatch: "
            f"{expected_feature_dim} != {checkpoint['feature_dim']}"
        )
    expected_encoder = checkpoint.get("encoder", {})
    for key, actual in (
        ("type", getattr(encoder, "contract_type", None)),
        ("model", getattr(encoder, "model", None)),
        ("model_digest", getattr(encoder, "model_digest", None)),
        ("dimension", getattr(encoder, "dim", None)),
    ):
        expected = expected_encoder.get(key)
        if expected is not None and actual != expected:
            raise ValueError(
                f"encoder/checkpoint {key} mismatch: {actual!r} != "
                f"{expected!r}"
            )

    saved_decision_rule = checkpoint.get("decision_rule", "sequential")
    if saved_decision_rule not in {"joint", "sequential"}:
        raise ValueError(
            f"unsupported checkpoint decision rule {saved_decision_rule!r}")
    if decision_rule is not None and decision_rule != saved_decision_rule:
        raise ValueError(
            "decision rule differs from checkpoint contract: "
            f"{decision_rule!r} != {saved_decision_rule!r}"
        )
    current_verb_order = [verb.value for verb in VERB_ORDER]
    saved_verb_order = checkpoint.get("verb_order", current_verb_order)
    if saved_verb_order != current_verb_order:
        raise ValueError(
            f"verb order differs from runtime: {saved_verb_order!r} != "
            f"{current_verb_order!r}")
    action_semantics = checkpoint.get(
        "action_semantics", ACTION_SEMANTICS_VERSION)
    if action_semantics != ACTION_SEMANTICS_VERSION:
        raise ValueError(
            f"unsupported action semantics {action_semantics!r}")

    saved_schema = checkpoint.get("observation_schema", "tree-render-v1")
    if saved_schema not in {
        "tree-render-v1", CONTROLLER_OBSERVATION_VERSION
    }:
        raise ValueError(
            f"unsupported checkpoint observation schema {saved_schema!r}")
    if observation_schema is not None and observation_schema != saved_schema:
        raise ValueError(
            "observation schema differs from checkpoint contract: "
            f"{observation_schema!r} != {saved_schema!r}"
        )
    runtime_feature_contract = feature_contract_for_schema(saved_schema)
    saved_feature_contract = checkpoint.get(
        "feature_contract", runtime_feature_contract)
    if saved_feature_contract != runtime_feature_contract:
        raise ValueError(
            "feature serialization contract differs from runtime")
    training_mdp_contracts = tuple(
        checkpoint.get("training_mdp_contracts", ()))
    for contract in training_mdp_contracts:
        if contract.get("action_semantics") != action_semantics:
            raise ValueError(
                "checkpoint MDP contract has different action semantics")
    training_target_contract = checkpoint.get(
        "training_target_contract", {})
    saved_outcome_schema = checkpoint.get(
        "previous_outcome_schema",
        (
            "runner-feedback-v0"
            if saved_schema == "tree-render-v1"
            else PREVIOUS_OUTCOME_SCHEMA
        ),
    )
    expected_outcome_schema = (
        "runner-feedback-v0"
        if saved_schema == "tree-render-v1"
        else PREVIOUS_OUTCOME_SCHEMA
    )
    if saved_outcome_schema != expected_outcome_schema:
        raise ValueError(
            "previous-outcome schema differs from observation contract")
    config = checkpoint["model_config"]
    contract_manifest = checkpoint.get("contract_manifest")
    contract_sha256 = checkpoint.get("contract_sha256")
    if contract_manifest is not None:
        actual_contract_sha256 = hashlib.sha256(
            json.dumps(
                contract_manifest,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if contract_sha256 != actual_contract_sha256:
            raise ValueError("checkpoint contract manifest digest mismatch")
        for field, actual in (
            ("observation_schema", saved_schema),
            ("previous_outcome_schema", saved_outcome_schema),
            ("action_semantics", action_semantics),
            ("verb_order", saved_verb_order),
            ("feature_contract", saved_feature_contract),
            ("encoder", expected_encoder),
            ("training_mdp_contracts", list(training_mdp_contracts)),
            ("training_target_contract", training_target_contract),
            (
                "policy_contract",
                {
                    "decision_rule": saved_decision_rule,
                    "acting_score": "behavior-cloning-probability-v1",
                    "model_config": config,
                },
            ),
        ):
            if contract_manifest.get(field) != actual:
                raise ValueError(
                    f"checkpoint manifest {field} differs from metadata")

    model = FactoredController(checkpoint["feature_dim"], **config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return LearnedController(
        model=model,
        encoder=encoder,
        depth_limit=depth_limit,
        decision_rule=saved_decision_rule,
        observation_schema=saved_schema,
        action_semantics=action_semantics,
        feature_contract=saved_feature_contract,
        training_mdp_contracts=training_mdp_contracts,
        training_target_contract=training_target_contract,
        contract_sha256=contract_sha256,
    )


__all__ = [
    "LearnedController",
    "OllamaBatchTextFeatures",
    "load_learned_controller",
]
