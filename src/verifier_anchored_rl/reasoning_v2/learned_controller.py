"""Checkpoint-bound learned controller for the reasoning v2 policy.

The controller consumes only ``BoundFeatureExtractor`` output. Text encoders
are frozen adapters with exact JSON contracts, while the trainable network is
the four-verb factored policy and Q architecture pinned by ``checkpoint.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in minimal installations
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from .checkpoint import (
    CHECKPOINT_CONTROLLER_PROVIDER,
    CONTROLLER_ARCHITECTURE_VERSION,
    LEARNED_CONTROLLER_IMPLEMENTATION,
    LEARNED_CONTROLLER_MAX_TOKENS,
    LEARNED_CONTROLLER_OUTPUT_CONTRACT,
    ModelShapeContract,
    validate_checkpoint,
)
from .contracts import (
    CollectionContract,
    ModelIdentity,
    ModelTransportContract,
    canonical_json_bytes,
    implementation_name,
)
from .features import (
    FEATURE_CONTRACT_VERSION,
    STRUCTURAL_FEATURE_DIM,
    BoundFeatureExtractor,
    StructuredFeatureBatch,
)
from .menu import POLICY_VERB_ORDER
from .model import Action, Verb
from .provenance import BehaviorSource, ControllerChoice
from .runtime import BoundEnvironment


HASH_ENCODER_ALGORITHM = "sha256-counter-float32-v1"
ZERO_TEXT_ENCODER_ALGORITHM = "all-zero-float32-v1"
TORCH_STATE_DIGEST_VERSION = b"phase-b-torch-state-dict-sha256-v1\0"

_ENCODER_CONTRACT_KEYS = frozenset({
    "type",
    "implementation",
    "algorithm_contract",
    "model",
    "model_digest",
    "dimension",
})
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_POLICY_VERB_COUNT = 4


class EncoderValidationError(ValueError):
    """Raised when an encoder does not match its exact frozen contract."""


class ControllerIntegrationError(RuntimeError):
    """Raised when model output and the legal feature mask disagree."""


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the reasoning v2 learned controller"
        )


@runtime_checkable
class ControllerTextEncoder(Protocol):
    """Exact frozen text encoder used by the v2 controller."""

    dimension: int

    def contract_record(self) -> dict[str, object]:
        ...

    def state_sha256(self) -> str:
        """Digest the live algorithm state or frozen model weights."""
        ...

    def encode_many(
        self, texts: Sequence[str]
    ) -> Sequence[Sequence[float]]:
        ...


def _positive_dimension(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise EncoderValidationError(
            "encoder dimension must be a positive integer"
        )
    return value


def encoder_contract_record(
    encoder: ControllerTextEncoder,
) -> dict[str, object]:
    """Return and validate a detached exact checkpoint encoder record."""
    if not isinstance(encoder, ControllerTextEncoder):
        raise EncoderValidationError(
            "encoder must implement the exact controller encoder protocol"
        )
    if not callable(encoder.contract_record) or not callable(
        encoder.encode_many
    ):
        raise EncoderValidationError("encoder methods must be callable")
    try:
        encoded = canonical_json_bytes(encoder.contract_record())
        record = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise EncoderValidationError(
            f"encoder contract is not canonical JSON: {exc}"
        ) from exc
    if type(record) is not dict or set(record) != _ENCODER_CONTRACT_KEYS:
        raise EncoderValidationError(
            "encoder contract keys must be exactly "
            f"{sorted(_ENCODER_CONTRACT_KEYS)!r}"
        )
    if record["type"] not in {"hash", "frozen_embedding", "zero_text"}:
        raise EncoderValidationError("encoder type is unsupported")
    for field in ("implementation", "algorithm_contract"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise EncoderValidationError(
                f"encoder {field} must be nonempty text"
            )
    if record["implementation"] != implementation_name(encoder):
        raise EncoderValidationError(
            "encoder implementation differs from its concrete adapter"
        )
    dimension = _positive_dimension(record["dimension"])
    if encoder.dimension != dimension:
        raise EncoderValidationError(
            "encoder dimension differs from its contract record"
        )
    model = record["model"]
    model_digest = record["model_digest"]
    if record["type"] == "frozen_embedding":
        if not isinstance(model, str) or not model.strip():
            raise EncoderValidationError(
                "frozen embedding encoder needs a model"
            )
        if (
            not isinstance(model_digest, str)
            or _SHA256_RE.fullmatch(model_digest) is None
        ):
            raise EncoderValidationError(
                "frozen embedding encoder needs a model digest"
            )
    elif model is not None or model_digest is not None:
        raise EncoderValidationError(
            "hash and zero-text encoders cannot carry model identity"
        )
    try:
        state_digest = encoder.state_sha256()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise EncoderValidationError(
            f"encoder state digest failed: {exc}"
        ) from exc
    if not isinstance(state_digest, str) or _SHA256_RE.fullmatch(
        state_digest
    ) is None:
        raise EncoderValidationError(
            "encoder state digest must be 64 lowercase hex characters"
        )
    expected_state_digest = (
        model_digest
        if record["type"] == "frozen_embedding"
        else hashlib.sha256(canonical_json_bytes(record)).hexdigest()
    )
    if state_digest != expected_state_digest:
        raise EncoderValidationError(
            "encoder live state differs from its contract"
        )
    return record


def _validate_texts(texts: Sequence[str]) -> tuple[str, ...]:
    if isinstance(texts, (str, bytes)):
        raise TypeError("encoder input must be a sequence of strings")
    snapshot = tuple(texts)
    if any(not isinstance(text, str) for text in snapshot):
        raise TypeError("encoder input must contain only strings")
    return snapshot


@dataclass(frozen=True, slots=True)
class DeterministicHashEncoder:
    """Content-sensitive SHA-256 expansion with no learned parameters."""

    dimension: int

    def __post_init__(self) -> None:
        _positive_dimension(self.dimension)

    def contract_record(self) -> dict[str, object]:
        return {
            "type": "hash",
            "implementation": implementation_name(self),
            "algorithm_contract": HASH_ENCODER_ALGORITHM,
            "model": None,
            "model_digest": None,
            "dimension": self.dimension,
        }

    def state_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self.contract_record())
        ).hexdigest()

    def encode_many(
        self, texts: Sequence[str]
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(self._encode(text) for text in _validate_texts(texts))

    def _encode(self, text: str) -> tuple[float, ...]:
        raw = text.encode("utf-8")
        prefix = (
            HASH_ENCODER_ALGORITHM.encode("ascii")
            + len(raw).to_bytes(8, "big")
            + raw
        )
        values: list[float] = []
        counter = 0
        while len(values) < self.dimension:
            block = hashlib.sha256(
                prefix + counter.to_bytes(8, "big")
            ).digest()
            for offset in range(0, len(block), 4):
                integer = int.from_bytes(
                    block[offset : offset + 4], "big"
                )
                values.append(integer / 2147483648.0 - 1.0)
                if len(values) == self.dimension:
                    break
            counter += 1
        return tuple(values)


@dataclass(frozen=True, slots=True)
class ZeroTextEncoder:
    """Content-free encoder used for the registered structural comparator."""

    dimension: int

    def __post_init__(self) -> None:
        _positive_dimension(self.dimension)

    def contract_record(self) -> dict[str, object]:
        return {
            "type": "zero_text",
            "implementation": implementation_name(self),
            "algorithm_contract": ZERO_TEXT_ENCODER_ALGORITHM,
            "model": None,
            "model_digest": None,
            "dimension": self.dimension,
        }

    def state_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self.contract_record())
        ).hexdigest()

    def encode_many(
        self, texts: Sequence[str]
    ) -> tuple[tuple[float, ...], ...]:
        snapshot = _validate_texts(texts)
        zero = (0.0,) * self.dimension
        return tuple(zero for _ in snapshot)


@dataclass(frozen=True, slots=True)
class EncodedControllerBatch:
    """Torch tensors derived from one exact structured feature batch."""

    source: StructuredFeatureBatch
    global_features: Any
    node_features: Any
    legal_pair_mask: Any


def _encoded_matrix(
    encoder: ControllerTextEncoder,
    texts: tuple[str, ...],
) -> Any:
    _require_torch()
    vectors = encoder.encode_many(texts)
    if len(vectors) != len(texts):
        raise EncoderValidationError(
            "encoder returned the wrong number of vectors"
        )
    rows: list[tuple[float, ...]] = []
    for vector in vectors:
        try:
            row = tuple(vector)
        except TypeError as exc:
            raise EncoderValidationError(
                "encoder vector is not a finite numeric sequence"
            ) from exc
        if len(row) != encoder.dimension:
            raise EncoderValidationError(
                "encoder returned a vector with the wrong dimension"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value))
            for value in row
        ):
            raise EncoderValidationError(
                "encoder vector is not a finite numeric sequence"
            )
        rows.append(tuple(float(value) for value in row))
    if not rows:
        raise EncoderValidationError("encoder received no global input")
    return torch.tensor(rows, dtype=torch.float32)


def encode_structured_features(
    features: StructuredFeatureBatch,
    encoder: ControllerTextEncoder,
) -> EncodedControllerBatch:
    """Encode global and target text, appending the same six scalars."""
    _require_torch()
    if not isinstance(features, StructuredFeatureBatch):
        raise TypeError(
            "learned controller input must be StructuredFeatureBatch"
        )
    record = encoder_contract_record(encoder)
    if features.contract_version != FEATURE_CONTRACT_VERSION:
        raise EncoderValidationError(
            "structured feature contract differs from the controller"
        )
    texts = (
        features.global_serialization,
        *features.target_serializations,
    )
    encoded = _encoded_matrix(encoder, texts)
    scalars = torch.tensor(
        features.structural_scalars, dtype=torch.float32
    )
    if scalars.shape != (STRUCTURAL_FEATURE_DIM,) or not torch.isfinite(
        scalars
    ).all():
        raise EncoderValidationError(
            "structured scalar vector is invalid"
        )
    scalar_rows = scalars.unsqueeze(0).expand(encoded.shape[0], -1)
    combined = torch.cat((encoded, scalar_rows), dim=-1)
    expected_dim = record["dimension"] + STRUCTURAL_FEATURE_DIM
    if combined.shape != (len(texts), expected_dim):
        raise EncoderValidationError(
            "combined controller features have the wrong shape"
        )
    mask = torch.tensor(features.legal_pair_mask, dtype=torch.bool).unsqueeze(0)
    expected_mask = (
        1,
        _POLICY_VERB_COUNT,
        len(features.target_ids),
    )
    if tuple(mask.shape) != expected_mask:
        raise ControllerIntegrationError(
            "legal pair mask differs from the fixed four-verb shape"
        )
    return EncodedControllerBatch(
        source=features,
        global_features=combined[0].unsqueeze(0),
        node_features=combined[1:].unsqueeze(0),
        legal_pair_mask=mask,
    )


if _TORCH_AVAILABLE:

    class FactoredControllerNetwork(nn.Module):
        """Four-verb policy and Q network over global and target features."""

        def __init__(
            self,
            feature_dim: int,
            *,
            hidden_dim: int,
            repr_dim: int,
            node_repr_dim: int,
            verb_embed_dim: int,
        ) -> None:
            _require_torch()
            super().__init__()
            for name, value in (
                ("feature_dim", feature_dim),
                ("hidden_dim", hidden_dim),
                ("repr_dim", repr_dim),
                ("node_repr_dim", node_repr_dim),
                ("verb_embed_dim", verb_embed_dim),
            ):
                if type(value) is not int or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
            self.feature_dim = feature_dim
            self.global_adapter = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, repr_dim),
                nn.GELU(),
            )
            self.node_adapter = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, node_repr_dim),
                nn.GELU(),
            )
            self.verb_head = nn.Linear(repr_dim, _POLICY_VERB_COUNT)
            self.verb_embedding = nn.Embedding(
                _POLICY_VERB_COUNT, verb_embed_dim
            )
            pair_dim = repr_dim + node_repr_dim + verb_embed_dim
            self.pointer_head = nn.Sequential(
                nn.Linear(pair_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.q_head = nn.Sequential(
                nn.Linear(pair_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

        def _validate_inputs(
            self,
            global_features: Any,
            node_features: Any,
            legal_pair_mask: Any | None = None,
        ) -> None:
            if global_features.ndim != 2 or global_features.shape[1] != (
                self.feature_dim
            ):
                raise ValueError("global feature tensor has the wrong shape")
            if node_features.ndim != 3 or node_features.shape[2] != (
                self.feature_dim
            ):
                raise ValueError("node feature tensor has the wrong shape")
            if global_features.shape[0] != node_features.shape[0]:
                raise ValueError("global and node batch dimensions differ")
            if not torch.isfinite(global_features).all() or not torch.isfinite(
                node_features
            ).all():
                raise ValueError("controller features must be finite")
            if legal_pair_mask is not None:
                expected = (
                    global_features.shape[0],
                    _POLICY_VERB_COUNT,
                    node_features.shape[1],
                )
                if tuple(legal_pair_mask.shape) != expected:
                    raise ValueError("legal pair mask has the wrong shape")
                if (~legal_pair_mask.bool().any(dim=(1, 2))).any():
                    raise ValueError("a batch row has no legal action pairs")

        def verb_logits(self, global_features: Any) -> Any:
            if global_features.ndim != 2 or global_features.shape[1] != (
                self.feature_dim
            ):
                raise ValueError("global feature tensor has the wrong shape")
            return self.verb_head(self.global_adapter(global_features))

        def pair_scores(
            self, global_features: Any, node_features: Any
        ) -> tuple[Any, Any]:
            """Return pointer logits and Q values with shape ``[B, 4, N]``."""
            self._validate_inputs(global_features, node_features)
            global_repr = self.global_adapter(global_features)
            node_repr = self.node_adapter(node_features)
            batch, nodes, _ = node_repr.shape
            global_grid = global_repr[:, None, None, :].expand(
                -1, _POLICY_VERB_COUNT, nodes, -1
            )
            node_grid = node_repr[:, None, :, :].expand(
                -1, _POLICY_VERB_COUNT, -1, -1
            )
            verb_grid = self.verb_embedding.weight[None, :, None, :].expand(
                batch, -1, nodes, -1
            )
            pair_repr = torch.cat(
                (global_grid, node_grid, verb_grid), dim=-1
            )
            pointer = self.pointer_head(pair_repr).squeeze(-1)
            q_values = self.q_head(pair_repr).squeeze(-1)
            if not torch.isfinite(pointer).all() or not torch.isfinite(
                q_values
            ).all():
                raise ControllerIntegrationError(
                    "pointer or Q head produced a nonfinite value"
                )
            return pointer, q_values

        def masked_policy(
            self,
            global_features: Any,
            node_features: Any,
            legal_pair_mask: Any,
        ) -> tuple[Any, Any]:
            """Return verb and conditional pointer BC probabilities."""
            mask = legal_pair_mask.bool()
            self._validate_inputs(global_features, node_features, mask)
            legal_verbs = mask.any(dim=-1)
            verb_logits = self.verb_logits(global_features)
            verb_prob = F.softmax(
                verb_logits.masked_fill(~legal_verbs, float("-inf")),
                dim=-1,
            )
            pointer_logits, _ = self.pair_scores(
                global_features, node_features
            )
            safe_mask = mask.clone()
            safe_mask[~legal_verbs] = True
            pointer_prob = F.softmax(
                pointer_logits.masked_fill(~safe_mask, float("-inf")),
                dim=-1,
            )
            pointer_prob = torch.where(
                legal_verbs.unsqueeze(-1),
                pointer_prob,
                torch.zeros_like(pointer_prob),
            )
            if not torch.isfinite(verb_prob).all() or not torch.isfinite(
                pointer_prob
            ).all():
                raise ControllerIntegrationError(
                    "policy head produced a nonfinite probability"
                )
            return verb_prob, pointer_prob

        def masked_joint_policy(
            self,
            global_features: Any,
            node_features: Any,
            legal_pair_mask: Any,
        ) -> Any:
            """Return the normalized BC probability of every legal pair."""
            mask = legal_pair_mask.bool()
            verb_prob, pointer_prob = self.masked_policy(
                global_features, node_features, mask
            )
            joint = (verb_prob.unsqueeze(-1) * pointer_prob).masked_fill(
                ~mask, 0.0
            )
            if not torch.allclose(
                joint.sum(dim=(1, 2)),
                torch.ones(joint.shape[0], device=joint.device),
                atol=1e-6,
                rtol=1e-6,
            ):
                raise ControllerIntegrationError(
                    "joint legal-pair policy is not normalized"
                )
            return joint

else:

    class FactoredControllerNetwork:  # pragma: no cover - import-only path
        def __init__(self, *args: object, **kwargs: object) -> None:
            _require_torch()


def _expected_state_shapes(
    shape: ModelShapeContract,
) -> dict[str, tuple[int, ...]]:
    if shape.architecture != CONTROLLER_ARCHITECTURE_VERSION:
        raise ValueError("model architecture differs from the runtime")
    if shape.policy_verb_count != _POLICY_VERB_COUNT or len(
        POLICY_VERB_ORDER
    ) != _POLICY_VERB_COUNT:
        raise ValueError("controller must have exactly four policy verbs")
    pair_dim = shape.repr_dim + shape.node_repr_dim + shape.verb_embed_dim
    return {
        "global_adapter.0.weight": (shape.feature_dim,),
        "global_adapter.0.bias": (shape.feature_dim,),
        "global_adapter.1.weight": (shape.hidden_dim, shape.feature_dim),
        "global_adapter.1.bias": (shape.hidden_dim,),
        "global_adapter.3.weight": (shape.repr_dim, shape.hidden_dim),
        "global_adapter.3.bias": (shape.repr_dim,),
        "node_adapter.0.weight": (shape.feature_dim,),
        "node_adapter.0.bias": (shape.feature_dim,),
        "node_adapter.1.weight": (shape.node_repr_dim, shape.feature_dim),
        "node_adapter.1.bias": (shape.node_repr_dim,),
        "verb_head.weight": (shape.policy_verb_count, shape.repr_dim),
        "verb_head.bias": (shape.policy_verb_count,),
        "verb_embedding.weight": (
            shape.policy_verb_count,
            shape.verb_embed_dim,
        ),
        "pointer_head.0.weight": (shape.hidden_dim, pair_dim),
        "pointer_head.0.bias": (shape.hidden_dim,),
        "pointer_head.2.weight": (1, shape.hidden_dim),
        "pointer_head.2.bias": (1,),
        "q_head.0.weight": (shape.hidden_dim, pair_dim),
        "q_head.0.bias": (shape.hidden_dim,),
        "q_head.2.weight": (1, shape.hidden_dim),
        "q_head.2.bias": (1,),
    }


def _validate_torch_tensor(value: object, key: str) -> Any:
    _require_torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"model state {key!r} is not a torch tensor")
    if value.layout != torch.strided or value.is_quantized:
        raise ValueError(f"model state {key!r} must be a dense tensor")
    if value.dtype != torch.float32:
        raise ValueError(f"model state {key!r} must use float32")
    if not torch.isfinite(value.detach()).all().item():
        raise ValueError(f"model state {key!r} must contain finite values")
    return value


def validate_torch_state_dict(
    state_dict: Mapping[str, object], shape: ModelShapeContract
) -> None:
    """Validate exact keys, float32 tensors, and all architecture shapes."""
    _require_torch()
    if not isinstance(state_dict, Mapping):
        raise TypeError("model state must be a mapping")
    expected = _expected_state_shapes(shape)
    if set(state_dict) != set(expected):
        raise ValueError(
            "model state keys must be exactly "
            f"{sorted(expected)!r}"
        )
    for key in sorted(expected):
        tensor = _validate_torch_tensor(state_dict[key], key)
        if tuple(tensor.shape) != expected[key]:
            raise ValueError(
                f"model state {key!r} shape {tuple(tensor.shape)!r} "
                f"does not equal {expected[key]!r}"
            )


def torch_state_dict_sha256(state_dict: Mapping[str, object]) -> str:
    """Hash sorted tensor keys, shapes, dtypes, and canonical weight bytes."""
    _require_torch()
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("model state must be a nonempty mapping")
    if any(not isinstance(key, str) or not key for key in state_dict):
        raise ValueError("model state keys must be nonempty strings")
    digest = hashlib.sha256()
    digest.update(TORCH_STATE_DIGEST_VERSION)
    for key in sorted(state_dict):
        tensor = _validate_torch_tensor(state_dict[key], key)
        key_bytes = key.encode("utf-8")
        digest.update(struct.pack(">Q", len(key_bytes)))
        digest.update(key_bytes)
        dtype_bytes = b"float32-le"
        digest.update(struct.pack(">Q", len(dtype_bytes)))
        digest.update(dtype_bytes)
        digest.update(struct.pack(">Q", tensor.ndim))
        for dimension in tensor.shape:
            digest.update(struct.pack(">Q", dimension))
        array = tensor.detach().cpu().contiguous().numpy().astype(
            "<f4", copy=False
        )
        raw = array.tobytes(order="C")
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def select_joint_legal_action(
    features: StructuredFeatureBatch, joint_probability: Any
) -> Action:
    """Take the global legal-pair argmax, rejecting any mask violation."""
    _require_torch()
    if not isinstance(features, StructuredFeatureBatch):
        raise TypeError("joint selection needs StructuredFeatureBatch")
    expected_shape = (
        1,
        _POLICY_VERB_COUNT,
        len(features.target_ids),
    )
    if not isinstance(joint_probability, torch.Tensor) or tuple(
        joint_probability.shape
    ) != expected_shape:
        raise ControllerIntegrationError(
            "joint policy tensor has the wrong shape"
        )
    if not torch.isfinite(joint_probability).all():
        raise ControllerIntegrationError(
            "joint policy contains a nonfinite value"
        )
    if (joint_probability < 0.0).any():
        raise ControllerIntegrationError(
            "joint policy contains a negative probability"
        )
    mask = torch.tensor(
        features.legal_pair_mask,
        dtype=torch.bool,
        device=joint_probability.device,
    ).unsqueeze(0)
    if not mask.any():
        raise ControllerIntegrationError("controller state has no legal pair")
    if (joint_probability.masked_select(~mask) != 0.0).any():
        raise ControllerIntegrationError(
            "joint policy assigned mass outside the legal mask"
        )
    legal_mass = joint_probability.masked_select(mask).sum()
    if not legal_mass.item() > 0.0:
        raise ControllerIntegrationError(
            "joint policy assigned no mass to legal pairs"
        )
    masked = joint_probability.masked_fill(~mask, float("-inf"))
    flat_index = int(masked[0].reshape(-1).argmax().item())
    target_count = len(features.target_ids)
    verb_index, target_index = divmod(flat_index, target_count)
    if not mask[0, verb_index, target_index].item():
        raise ControllerIntegrationError(
            "global joint argmax selected an out-of-mask pair"
        )
    indexed = {
        pair_index: pair
        for pair_index, pair in zip(
            features.legal_pair_indices,
            features.legal_pairs,
            strict=True,
        )
    }
    pair = indexed.get((verb_index, target_index))
    if pair is None:
        raise ControllerIntegrationError(
            "legal mask and legal-pair index table disagree"
        )
    verb_name, target_id = pair
    if (
        features.policy_verb_order[verb_index] != verb_name
        or features.target_ids[target_index] != target_id
    ):
        raise ControllerIntegrationError(
            "selected action differs from its feature columns"
        )
    return Action(Verb(verb_name), target_id)


class FactoredLearnedController:
    """Exact checkpoint-backed implementation of the v2 bound controller."""

    controller_implementation = LEARNED_CONTROLLER_IMPLEMENTATION
    controller_transport_contract = ModelTransportContract.local(
        LEARNED_CONTROLLER_IMPLEMENTATION
    )
    controller_transport_sha256 = controller_transport_contract.sha256
    controller_seed = None
    controller_temperature = 0.0
    controller_max_tokens = LEARNED_CONTROLLER_MAX_TOKENS
    controller_seed_policy = "none-v1"
    controller_output_contract = LEARNED_CONTROLLER_OUTPUT_CONTRACT
    controller_fallback_policy = "forbidden-v1"
    behavior_source = BehaviorSource.LEARNED_POLICY
    policy_input_contract = FEATURE_CONTRACT_VERSION

    def __init__(
        self,
        checkpoint: object,
        *,
        environment: BoundEnvironment,
        live_collection_contract: CollectionContract,
        encoder: ControllerTextEncoder,
    ) -> None:
        _require_torch()
        if not isinstance(environment, BoundEnvironment):
            raise TypeError("learned controller needs a BoundEnvironment")
        if not isinstance(live_collection_contract, CollectionContract):
            raise TypeError(
                "learned controller needs a live CollectionContract"
            )
        environment.validate_binding()
        encoder_record = encoder_contract_record(encoder)
        validated = validate_checkpoint(
            checkpoint,
            environment_contract=environment.contract,
            live_collection_contract=live_collection_contract,
            encoder_contract=encoder_record,
            model_state_validator=validate_torch_state_dict,
            model_state_digester=torch_state_dict_sha256,
        )
        if validated.model_shape.feature_dim != (
            encoder.dimension + STRUCTURAL_FEATURE_DIM
        ):
            raise ValueError(
                "checkpoint feature dimension differs from encoder plus "
                "six structural scalars"
            )
        snapshot = {
            key: _validate_torch_tensor(value, key)
            .detach()
            .cpu()
            .clone()
            for key, value in validated.state_dict.items()
        }
        validate_torch_state_dict(snapshot, validated.model_shape)
        if torch_state_dict_sha256(snapshot) != validated.state_dict_sha256:
            raise ValueError(
                "model state changed between checkpoint validation and load"
            )
        shape = validated.model_shape
        model = FactoredControllerNetwork(
            shape.feature_dim,
            hidden_dim=shape.hidden_dim,
            repr_dim=shape.repr_dim,
            node_repr_dim=shape.node_repr_dim,
            verb_embed_dim=shape.verb_embed_dim,
        )
        model.load_state_dict(snapshot, strict=True)
        model.eval()

        expected_identity = ModelIdentity(
            CHECKPOINT_CONTROLLER_PROVIDER,
            CONTROLLER_ARCHITECTURE_VERSION,
            validated.contract_sha256,
        )
        if live_collection_contract.controller != expected_identity:
            raise ValueError(
                "live controller identity differs from the checkpoint"
            )
        self.controller_identity = expected_identity
        self.environment_contract_sha256 = environment.sha256
        self._model = model
        self._environment = environment
        self._features = BoundFeatureExtractor(environment)
        self._encoder = encoder
        self._encoder_contract_bytes = canonical_json_bytes(encoder_record)
        self._collection = live_collection_contract
        self._collection_contract_sha256 = live_collection_contract.sha256
        self.checkpoint_contract_sha256 = validated.contract_sha256
        self.checkpoint_state_dict_sha256 = validated.state_dict_sha256

    @property
    def encoder_contract(self) -> dict[str, object]:
        return json.loads(self._encoder_contract_bytes)

    def _validate_live_binding(self) -> None:
        self._environment.validate_binding()
        if self._features.environment is not self._environment:
            raise ValueError("feature extractor environment changed in place")
        if self._collection.sha256 != self._collection_contract_sha256:
            raise ValueError("live collection contract changed in place")
        current_encoder = canonical_json_bytes(
            encoder_contract_record(self._encoder)
        )
        if current_encoder != self._encoder_contract_bytes:
            raise ValueError("live encoder contract changed in place")
        if self.environment_contract_sha256 != self._environment.sha256:
            raise ValueError("learned controller environment binding drifted")
        if torch_state_dict_sha256(self._model.state_dict()) != (
            self.checkpoint_state_dict_sha256
        ):
            raise ValueError(
                "learned controller model state differs from its checkpoint"
            )

    def choose(self, observation: Mapping[str, object]) -> ControllerChoice:
        """Select the global maximum BC probability over all legal pairs."""
        self._validate_live_binding()
        features = self._features.build(observation)
        encoded = encode_structured_features(features, self._encoder)
        if not features.legal_pairs:
            raise ControllerIntegrationError(
                "learned controller received a state with no legal action"
            )
        self._model.eval()
        with torch.no_grad():
            joint_probability = self._model.masked_joint_policy(
                encoded.global_features,
                encoded.node_features,
                encoded.legal_pair_mask,
            )
        action = select_joint_legal_action(features, joint_probability)
        return ControllerChoice(action)


def load_factored_learned_controller(
    checkpoint: object,
    *,
    environment: BoundEnvironment,
    live_collection_contract: CollectionContract,
    encoder: ControllerTextEncoder,
) -> FactoredLearnedController:
    """Validate and load one already safely deserialized v2 checkpoint."""
    return FactoredLearnedController(
        checkpoint,
        environment=environment,
        live_collection_contract=live_collection_contract,
        encoder=encoder,
    )


__all__ = [
    "ControllerIntegrationError",
    "ControllerTextEncoder",
    "DeterministicHashEncoder",
    "EncodedControllerBatch",
    "EncoderValidationError",
    "FactoredControllerNetwork",
    "FactoredLearnedController",
    "HASH_ENCODER_ALGORITHM",
    "TORCH_STATE_DIGEST_VERSION",
    "ZERO_TEXT_ENCODER_ALGORITHM",
    "ZeroTextEncoder",
    "encode_structured_features",
    "encoder_contract_record",
    "load_factored_learned_controller",
    "select_joint_legal_action",
    "torch_state_dict_sha256",
    "validate_torch_state_dict",
]
