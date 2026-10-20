"""Deterministic lexical features for the first Phase B pipeline diagnostic.

These features are deliberately cheap.  They validate the converter, masks,
factored heads, losses, and grouped evaluation before paying for frozen-LLM
hidden states.  They are not the encoder result claimed by the system design.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import numpy as np

from .controller_observation import CONTROLLER_OBSERVATION_VERSION


_TOKEN = re.compile(r"[A-Za-z_]+|\d+(?:\.\d+)?|[^\sA-Za-z_0-9]")
FEATURE_CONTRACT_VERSION = "phase-b-features-v1.1"
STRUCTURAL_SCALAR_FIELDS = (
    "turn_fraction",
    "remaining_turn_fraction",
    "log_node_count",
    "verified_node_fraction",
    "solved_goal_fraction",
    "pruned_node_fraction",
)
STRUCTURAL_FEATURE_DIM = len(STRUCTURAL_SCALAR_FIELDS)


def feature_contract_for_schema(observation_schema: str) -> dict[str, Any]:
    """Version every dimension-preserving feature transform."""
    if observation_schema == "tree-render-v1":
        global_serializer = "tree-render-v1-bytes"
        node_serializer = "node-feature-text-v1"
    elif observation_schema == CONTROLLER_OBSERVATION_VERSION:
        global_serializer = "controller-feature-json-v1.1"
        node_serializer = "node-feature-text-v1.1"
    else:
        raise ValueError(
            f"unsupported feature observation schema {observation_schema!r}")
    return {
        "version": FEATURE_CONTRACT_VERSION,
        "global_serializer": global_serializer,
        "node_serializer": node_serializer,
        "structural_scalars": list(STRUCTURAL_SCALAR_FIELDS),
    }


class HashingTextFeatures:
    """Signed feature hashing over lowercased unigrams and bigrams."""
    contract_type = "hash"

    def __init__(self, dim: int = 512):
        if dim < 8:
            raise ValueError("feature dimension must be at least 8")
        self.dim = int(dim)

    def encode(self, text: str) -> np.ndarray:
        tokens = [token.casefold() for token in _TOKEN.findall(text)]
        features = tokens + [
            f"{left}::{right}" for left, right in zip(tokens, tokens[1:])
        ]
        vector = np.zeros(self.dim, dtype=np.float32)
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self.dim
            vector[index] += 1.0 if value & (1 << 63) else -1.0
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector

    def encode_node(self, node: dict[str, Any]) -> np.ndarray:
        return self.encode(node_feature_text(node))


def node_feature_text(node: dict[str, Any]) -> str:
    """Stable structured context passed to either text feature encoder."""
    if "owner_path_text" not in node:
        # Exact historical phase-b-v1 text for E18/E19 reproducibility.
        legacy_status = " ".join(
            name
            for name in ("verified", "verify_failed", "solved", "abandoned")
            if node[name]
        )
        return (
            f"kind_{node['kind']} depth_{node['depth']} "
            f"parent_{'root' if node['parent'] is None else 'child'} "
            f"rendered_{node['rendered']} {legacy_status} {node['text']}"
        )

    status = " ".join(
        name
        for name in ("verified", "verify_failed", "solved", "abandoned")
        if node.get(name, False)
    )
    owner_path = " > ".join(node.get("owner_path_text", ()))
    premises = ",".join(str(item) for item in node.get("premises", ()))
    outcome = node.get("outcome_statement") or ""
    structural = (
        f"kind_{node['kind']} depth_{node['depth']} "
        f"parent_{'root' if node['parent'] is None else 'child'} "
        f"owner_{node.get('owner_goal')} "
        f"rendered_{node['rendered']} active_{node.get('active', True)} "
        f"source_{node.get('source')} status_{node.get('status')} {status}"
    )
    context = (
        f"owner_path {owner_path} premises {premises} "
        f"outcome {outcome} verifier {node.get('last_verifier_observed')}"
    )
    return f"{structural} {context} {node['text']}"


class CachedTextFeatures:
    """Read-only content-addressed frozen embeddings from an ``.npz`` file."""
    contract_type = "frozen_embedding_cache"

    def __init__(self, path: str):
        archive = np.load(path, allow_pickle=False)
        self.model = str(archive["model"].item())
        self.model_digest = str(archive["model_digest"].item())
        keys = archive["keys"].tolist()
        matrix = archive["embeddings"].astype(np.float32, copy=False)
        if matrix.ndim != 2 or len(keys) != matrix.shape[0]:
            raise ValueError("invalid frozen embedding cache shape")
        self.dim = int(matrix.shape[1])
        self._vectors = dict(zip(keys, matrix, strict=True))

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def encode(self, text: str) -> np.ndarray:
        key = self.key(text)
        try:
            return self._vectors[key]
        except KeyError as exc:
            raise KeyError(f"embedding cache misses text digest {key}") from exc

    def encode_node(self, node: dict[str, Any]) -> np.ndarray:
        return self.encode(node_feature_text(node))


class ZeroTextFeatures:
    """Shape-matched text ablation that retains only structural scalars."""
    contract_type = "zero_text_ablation"

    def __init__(self, dim: int = 768):
        if dim < 1:
            raise ValueError("feature dimension must be positive")
        self.dim = int(dim)

    def encode(self, text: str) -> np.ndarray:
        return np.zeros(self.dim, dtype=np.float32)

    def encode_node(self, node: dict[str, Any]) -> np.ndarray:
        return np.zeros(self.dim, dtype=np.float32)


def structural_scalar_features(row: dict[str, Any]) -> np.ndarray:
    """Small normalized state scalars appended to the hashed global vector."""
    limit = row["turn_limit"] or max(1, row["turn"] + 1)
    nodes = row["nodes"]
    return np.asarray(
        [
            row["turn"] / limit,
            (limit - row["turn"]) / limit,
            math.log1p(len(nodes)) / 4.0,
            sum(node["verified"] for node in nodes) / max(1, len(nodes)),
            sum(node["solved"] for node in nodes) / max(1, len(nodes)),
            sum(node["abandoned"] for node in nodes) / max(1, len(nodes)),
        ],
        dtype=np.float32,
    )


__all__ = [
    "CachedTextFeatures",
    "FEATURE_CONTRACT_VERSION",
    "HashingTextFeatures",
    "STRUCTURAL_FEATURE_DIM",
    "STRUCTURAL_SCALAR_FIELDS",
    "ZeroTextFeatures",
    "feature_contract_for_schema",
    "node_feature_text",
    "structural_scalar_features",
]
