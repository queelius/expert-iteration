"""Deterministic Phase B behavior-cloning and Monte Carlo warm-start.

This module implements only rung 2. Policy heads imitate every explicitly
provenance-eligible teacher action, while the Q head regresses the selected
action to the stored full-episode Monte Carlo return on every training row.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

try:
    import torch
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - import-only path
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from .checkpoint import (
    CONTROLLER_ARCHITECTURE_VERSION,
    ModelShapeContract,
    TRAINING_TARGET_VERSION,
    build_checkpoint,
)
from .contracts import (
    CollectionContract,
    canonical_json_bytes,
)
from .dataset import GroupedSplit, PhaseBDataset
from .features import STRUCTURAL_FEATURE_DIM
from .learned_controller import (
    ControllerTextEncoder,
    FactoredControllerNetwork,
    encode_structured_features,
    encoder_contract_record,
    torch_state_dict_sha256,
    validate_torch_state_dict,
)
from .runtime import BoundEnvironment
from .transition import REWARD_CONTRACT_VERSION


WARM_START_ALGORITHM_VERSION = "phase-b-bc-mc-warm-start-v2"
_MODEL_CONFIG_KEYS = frozenset({
    "hidden_dim",
    "repr_dim",
    "node_repr_dim",
    "verb_embed_dim",
})


class TrainingValidationError(ValueError):
    """Raised when trusted data or a warm-start input is inconsistent."""


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for Phase B warm-start")


@dataclass(frozen=True, slots=True)
class WarmStartConfig:
    """Optimization controls with no policy-label selection knob."""

    seed: int = 0
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    behavior_cloning_weight: float = 1.0
    q_regression_weight: float = 1.0

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be a nonnegative int63")
        if type(self.epochs) is not int or self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        for name in (
            "learning_rate",
            "behavior_cloning_weight",
            "q_regression_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, (int, float))
            or not math.isfinite(float(self.weight_decay))
            or float(self.weight_decay) < 0.0
        ):
            raise ValueError("weight_decay must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class PhaseBTrainingBatch:
    """One padded batch over a variable number of legal target nodes."""

    example_indices: tuple[int, ...]
    target_counts: tuple[int, ...]
    global_features: Any
    node_features: Any
    legal_pair_mask: Any
    chosen_verb_indices: Any
    chosen_target_indices: Any
    bc_label_mask: Any
    monte_carlo_returns: Any

    def __post_init__(self) -> None:
        self.validate()

    @property
    def size(self) -> int:
        return len(self.example_indices)

    def validate(self) -> None:
        """Reject forged, mutated, padded, or nonfinite tensor layouts."""
        _require_torch()
        batch_size = len(self.example_indices)
        if (
            batch_size == 0
            or len(self.example_indices) != len(set(self.example_indices))
            or any(type(index) is not int or index < 0 for index in self.example_indices)
        ):
            raise TrainingValidationError("batch example indices are invalid")
        if (
            type(self.target_counts) is not tuple
            or len(self.target_counts) != batch_size
            or any(type(count) is not int or count <= 0 for count in self.target_counts)
        ):
            raise TrainingValidationError("batch target counts are invalid")
        tensors = (
            self.global_features,
            self.node_features,
            self.legal_pair_mask,
            self.chosen_verb_indices,
            self.chosen_target_indices,
            self.bc_label_mask,
            self.monte_carlo_returns,
        )
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            raise TrainingValidationError("batch fields must be Torch tensors")
        if any(value.device.type != "cpu" for value in tensors):
            raise TrainingValidationError("warm-start batches must remain on CPU")
        if (
            self.global_features.dtype != torch.float32
            or self.node_features.dtype != torch.float32
            or self.monte_carlo_returns.dtype != torch.float32
        ):
            raise TrainingValidationError("batch floating tensors must use float32")
        if (
            self.legal_pair_mask.dtype != torch.bool
            or self.bc_label_mask.dtype != torch.bool
            or self.chosen_verb_indices.dtype != torch.long
            or self.chosen_target_indices.dtype != torch.long
        ):
            raise TrainingValidationError("batch mask or label dtype is invalid")
        if self.global_features.ndim != 2 or self.global_features.shape[0] != batch_size:
            raise TrainingValidationError("batch global feature shape is invalid")
        feature_dim = self.global_features.shape[1]
        max_targets = max(self.target_counts)
        if tuple(self.node_features.shape) != (
            batch_size,
            max_targets,
            feature_dim,
        ):
            raise TrainingValidationError("batch node feature shape is invalid")
        if tuple(self.legal_pair_mask.shape) != (batch_size, 4, max_targets):
            raise TrainingValidationError("batch legal mask shape is invalid")
        for value in (
            self.chosen_verb_indices,
            self.chosen_target_indices,
            self.bc_label_mask,
            self.monte_carlo_returns,
        ):
            if tuple(value.shape) != (batch_size,):
                raise TrainingValidationError("batch label shape is invalid")
        if not torch.isfinite(self.global_features).all().item() or not torch.isfinite(
            self.node_features
        ).all().item() or not torch.isfinite(self.monte_carlo_returns).all().item():
            raise TrainingValidationError("batch contains nonfinite values")
        if ((self.chosen_verb_indices < 0) | (self.chosen_verb_indices >= 4)).any().item():
            raise TrainingValidationError("batch verb label is out of range")
        rows = torch.arange(batch_size, device="cpu")
        for row, target_count in enumerate(self.target_counts):
            target_index = int(self.chosen_target_indices[row].item())
            if not 0 <= target_index < target_count:
                raise TrainingValidationError("batch target label is out of range")
            if self.legal_pair_mask[row, :, target_count:].any().item():
                raise TrainingValidationError("batch padded target is marked legal")
            if self.node_features[row, target_count:].count_nonzero().item():
                raise TrainingValidationError("batch node padding must be zero")
        if (~self.legal_pair_mask.any(dim=(1, 2))).any().item():
            raise TrainingValidationError("batch row has no legal action")
        if not self.legal_pair_mask[
            rows, self.chosen_verb_indices, self.chosen_target_indices
        ].all().item():
            raise TrainingValidationError("batch chosen action is masked")


@dataclass(frozen=True, slots=True)
class WarmStartLoss:
    """Differentiable rung-2 losses and the explicit BC label count."""

    total: Any
    behavior_cloning: Any
    verb: Any
    pointer: Any
    q_regression: Any
    bc_label_count: int
    critic_example_count: int


@dataclass(frozen=True, slots=True)
class WarmStartMetrics:
    """Deterministic update accounting returned by one training run."""

    algorithm: str
    training_examples: int
    bc_labeled_examples: int
    bc_label_exposures: int
    critic_examples: int
    updates: int
    total_losses: tuple[float, ...]
    behavior_cloning_losses: tuple[float, ...]
    q_regression_losses: tuple[float, ...]


_RESULT_TRUST_SEAL = object()


@dataclass(frozen=True, slots=True)
class WarmStartResult:
    """Trained model plus exact inputs needed to create its checkpoint."""

    model: Any
    encoder: ControllerTextEncoder
    encoder_contract: Mapping[str, object]
    model_config: Mapping[str, int]
    feature_dim: int
    gamma: float
    environment_contract_sha256: str
    collection_contract_sha256: str
    dataset_sha256: str
    split_sha256: str
    state_dict_sha256: str
    config: WarmStartConfig
    metrics: WarmStartMetrics
    _trust_seal: object = field(
        init=False, repr=False, compare=False, default=None
    )
    _trusted_binding: object = field(
        init=False, repr=False, compare=False, default=None
    )

    def validate_trusted(self) -> None:
        binding = _warm_start_result_binding(self)
        if (
            self._trust_seal is not _RESULT_TRUST_SEAL
            or self._trusted_binding != binding
        ):
            raise TrainingValidationError(
                "warm-start result was not produced by the trainer"
            )
        if torch_state_dict_sha256(self.model.state_dict()) != (
            self.state_dict_sha256
        ):
            raise TrainingValidationError(
                "trained model changed after warm-start completion"
            )


def _warm_start_result_binding(result: WarmStartResult) -> tuple[object, ...]:
    return (
        canonical_json_bytes(result.encoder_contract),
        canonical_json_bytes(result.model_config),
        result.feature_dim,
        result.gamma,
        result.environment_contract_sha256,
        result.collection_contract_sha256,
        result.dataset_sha256,
        result.split_sha256,
        result.state_dict_sha256,
        result.config,
        result.metrics,
    )


def _validated_model_config(value: object) -> dict[str, int]:
    try:
        snapshot = json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError) as exc:
        raise TrainingValidationError(
            f"model_config is not canonical JSON: {exc}"
        ) from exc
    if type(snapshot) is not dict or set(snapshot) != _MODEL_CONFIG_KEYS:
        raise TrainingValidationError(
            "model_config keys must be exactly "
            f"{sorted(_MODEL_CONFIG_KEYS)!r}"
        )
    for field in sorted(_MODEL_CONFIG_KEYS):
        if type(snapshot[field]) is not int or snapshot[field] <= 0:
            raise TrainingValidationError(
                f"model_config {field} must be a positive integer"
            )
    return snapshot


def _validated_dataset(
    dataset: PhaseBDataset,
    split: GroupedSplit | None = None,
) -> None:
    if type(dataset) is not PhaseBDataset:
        raise TypeError("training requires an exact PhaseBDataset")
    dataset.validate_trusted()
    if split is not None:
        if type(split) is not GroupedSplit:
            raise TypeError("training requires an exact GroupedSplit")
        dataset.checkpoint_contract(split)


def _validated_indices(
    dataset: PhaseBDataset, example_indices: Sequence[int]
) -> tuple[int, ...]:
    if isinstance(example_indices, (str, bytes)):
        raise TypeError("example_indices must be a sequence of integers")
    indices = tuple(example_indices)
    if not indices:
        raise TrainingValidationError("a training batch cannot be empty")
    if any(
        type(index) is not int or not 0 <= index < len(dataset.examples)
        for index in indices
    ):
        raise TrainingValidationError("training batch index is out of range")
    if len(indices) != len(set(indices)):
        raise TrainingValidationError(
            "training batch cannot repeat an example"
        )
    return indices


def _collate_trusted_indices(
    dataset: PhaseBDataset,
    indices: tuple[int, ...],
    encoder: ControllerTextEncoder,
    *,
    encoded_cache: Mapping[int, tuple[Any, Any, Any]] | None = None,
) -> PhaseBTrainingBatch:
    _require_torch()
    encoded_rows = []
    examples = []
    for index in indices:
        example = dataset.examples[index]
        if encoded_cache is None:
            encoded = encode_structured_features(
                example.current_features, encoder
            )
            tensors = (
                encoded.global_features.detach().to(
                    device="cpu", dtype=torch.float32
                ),
                encoded.node_features.detach().to(
                    device="cpu", dtype=torch.float32
                ),
                encoded.legal_pair_mask.detach().to(
                    device="cpu", dtype=torch.bool
                ),
            )
        else:
            try:
                tensors = encoded_cache[index]
            except KeyError as exc:
                raise TrainingValidationError(
                    "encoded cache is missing a training example"
                ) from exc
        encoded_rows.append(tensors)
        examples.append(example)

    feature_dim = encoded_rows[0][0].shape[1]
    target_counts = tuple(
        row[1].shape[1] for row in encoded_rows
    )
    max_targets = max(target_counts)
    batch_size = len(indices)
    global_features = torch.cat(
        tuple(row[0] for row in encoded_rows), dim=0
    )
    node_features = torch.zeros(
        (batch_size, max_targets, feature_dim),
        dtype=torch.float32,
        device="cpu",
    )
    legal_pair_mask = torch.zeros(
        (batch_size, 4, max_targets),
        dtype=torch.bool,
        device="cpu",
    )
    for row_index, (encoded, target_count) in enumerate(
        zip(encoded_rows, target_counts, strict=True)
    ):
        global_row, node_row, mask_row = encoded
        if global_row.shape != (1, feature_dim):
            raise TrainingValidationError(
                "encoded global feature dimensions differ within a batch"
            )
        if node_row.shape != (
            1,
            target_count,
            feature_dim,
        ):
            raise TrainingValidationError(
                "encoded node feature dimensions differ within a batch"
            )
        node_features[row_index, :target_count] = node_row[0]
        legal_pair_mask[row_index, :, :target_count] = (
            mask_row[0]
        )

    chosen_verbs = torch.tensor(
        [example.chosen_verb_index for example in examples],
        dtype=torch.long,
        device="cpu",
    )
    chosen_targets = torch.tensor(
        [example.chosen_target_index for example in examples],
        dtype=torch.long,
        device="cpu",
    )
    bc_label_mask = torch.tensor(
        [example.bc_eligible for example in examples],
        dtype=torch.bool,
        device="cpu",
    )
    returns = torch.tensor(
        [example.mc_return for example in examples],
        dtype=torch.float32,
        device="cpu",
    )
    batch_rows = torch.arange(batch_size, device="cpu")
    if not legal_pair_mask[
        batch_rows, chosen_verbs, chosen_targets
    ].all().item():
        raise TrainingValidationError(
            "a selected training action is outside its legal mask"
        )
    if not torch.isfinite(returns).all().item():
        raise TrainingValidationError(
            "Monte Carlo training returns must be finite"
        )
    return PhaseBTrainingBatch(
        example_indices=indices,
        target_counts=target_counts,
        global_features=global_features,
        node_features=node_features,
        legal_pair_mask=legal_pair_mask,
        chosen_verb_indices=chosen_verbs,
        chosen_target_indices=chosen_targets,
        bc_label_mask=bc_label_mask,
        monte_carlo_returns=returns,
    )


def collate_phase_b_training_batch(
    dataset: PhaseBDataset,
    example_indices: Sequence[int],
    encoder: ControllerTextEncoder,
) -> PhaseBTrainingBatch:
    """Encode and pad only examples owned by one replay-trusted dataset."""
    _validated_dataset(dataset)
    indices = _validated_indices(dataset, example_indices)
    return _collate_trusted_indices(dataset, indices, encoder)


def _build_encoded_training_cache(
    dataset: PhaseBDataset,
    indices: tuple[int, ...],
    encoder: ControllerTextEncoder,
) -> dict[int, tuple[Any, Any, Any]]:
    """Encode each frozen feature row exactly once before replay updates."""
    cache: dict[int, tuple[Any, Any, Any]] = {}
    for index in indices:
        encoded = encode_structured_features(
            dataset.examples[index].current_features, encoder
        )
        cache[index] = (
            encoded.global_features.detach().to(
                device="cpu", dtype=torch.float32
            ).clone(),
            encoded.node_features.detach().to(
                device="cpu", dtype=torch.float32
            ).clone(),
            encoded.legal_pair_mask.detach().to(
                device="cpu", dtype=torch.bool
            ).clone(),
        )
    return cache


def compute_warm_start_loss(
    model: FactoredControllerNetwork,
    batch: PhaseBTrainingBatch,
    *,
    behavior_cloning_weight: float = 1.0,
    q_regression_weight: float = 1.0,
) -> WarmStartLoss:
    """Compute masked factored BC and selected-pair Monte Carlo losses."""
    _require_torch()
    if not isinstance(model, FactoredControllerNetwork):
        raise TypeError("warm-start loss needs FactoredControllerNetwork")
    if not isinstance(batch, PhaseBTrainingBatch):
        raise TypeError("warm-start loss needs PhaseBTrainingBatch")
    batch.validate()
    for name, value in (
        ("behavior_cloning_weight", behavior_cloning_weight),
        ("q_regression_weight", q_regression_weight),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{name} must be finite and positive")

    mask = batch.legal_pair_mask.bool()
    legal_verbs = mask.any(dim=-1)
    raw_verb_logits = model.verb_logits(batch.global_features)
    verb_logits = raw_verb_logits.masked_fill(
        ~legal_verbs, float("-inf")
    )
    pointer_logits, q_values = model.pair_scores(
        batch.global_features, batch.node_features
    )
    rows = torch.arange(batch.size, device=batch.global_features.device)
    labels = batch.bc_label_mask.bool()
    label_count = int(labels.sum().item())
    if label_count:
        verb_loss = F.cross_entropy(
            verb_logits[labels], batch.chosen_verb_indices[labels]
        )
        labeled_rows = rows[labels]
        labeled_verbs = batch.chosen_verb_indices[labels]
        selected_pointer = pointer_logits[labeled_rows, labeled_verbs]
        selected_pointer_mask = mask[labeled_rows, labeled_verbs]
        pointer_loss = F.cross_entropy(
            selected_pointer.masked_fill(
                ~selected_pointer_mask, float("-inf")
            ),
            batch.chosen_target_indices[labels],
        )
    else:
        zero = q_values.new_zeros(())
        verb_loss = zero
        pointer_loss = zero
    bc_loss = verb_loss + pointer_loss

    selected_q = q_values[
        rows,
        batch.chosen_verb_indices,
        batch.chosen_target_indices,
    ]
    q_loss = F.mse_loss(selected_q, batch.monte_carlo_returns)
    total = (
        float(behavior_cloning_weight) * bc_loss
        + float(q_regression_weight) * q_loss
    )
    if not torch.isfinite(total).item():
        raise TrainingValidationError("warm-start loss is nonfinite")
    return WarmStartLoss(
        total=total,
        behavior_cloning=bc_loss,
        verb=verb_loss,
        pointer=pointer_loss,
        q_regression=q_loss,
        bc_label_count=label_count,
        critic_example_count=batch.size,
    )


def _training_example_indices(
    dataset: PhaseBDataset, split: GroupedSplit
) -> tuple[int, ...]:
    by_key = {
        (example.episode_key, example.turn): (index, example)
        for index, example in enumerate(dataset.examples)
    }
    indices: list[int] = []
    for example in split.train_examples:
        resolved = by_key.get((example.episode_key, example.turn))
        if resolved is None or resolved[1] != example:
            raise TrainingValidationError(
                "training split contains an example outside its dataset"
            )
        indices.append(resolved[0])
    if not indices:
        raise TrainingValidationError("training split contains no examples")
    if len(indices) != len(set(indices)):
        raise TrainingValidationError("training split repeats an example")
    return tuple(indices)


def train_phase_b_warm_start(
    dataset: PhaseBDataset,
    split: GroupedSplit,
    encoder: ControllerTextEncoder,
    *,
    model_config: object,
    config: WarmStartConfig = WarmStartConfig(),
) -> WarmStartResult:
    """Train rung 2 on all train rows and all eligible teacher labels."""
    _require_torch()
    _validated_dataset(dataset, split)
    if not isinstance(config, WarmStartConfig):
        raise TypeError("config must be WarmStartConfig")
    model_shape = _validated_model_config(model_config)
    encoder_record = encoder_contract_record(encoder)
    feature_dim = encoder.dimension + STRUCTURAL_FEATURE_DIM
    training_indices = _training_example_indices(dataset, split)
    bc_labeled_examples = sum(
        dataset.examples[index].bc_eligible for index in training_indices
    )
    if bc_labeled_examples == 0:
        raise TrainingValidationError(
            "warm-start training split has no provenance-eligible BC labels"
        )

    total_losses: list[float] = []
    bc_losses: list[float] = []
    q_losses: list[float] = []
    critic_examples = 0
    bc_label_exposures = 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    prior_dtype = torch.get_default_dtype()
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
    )
    try:
        torch.set_default_dtype(torch.float32)
        torch.use_deterministic_algorithms(True)
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(config.seed)
            encoded_cache = _build_encoded_training_cache(
                dataset, training_indices, encoder
            )
            model = FactoredControllerNetwork(
                feature_dim,
                **model_shape,
            ).to(device="cpu", dtype=torch.float32)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(config.learning_rate),
                weight_decay=float(config.weight_decay),
            )
            model.train()
            for _ in range(config.epochs):
                permutation = torch.randperm(
                    len(training_indices),
                    generator=generator,
                    device="cpu",
                ).tolist()
                ordered = tuple(
                    training_indices[index] for index in permutation
                )
                for start in range(0, len(ordered), config.batch_size):
                    batch_indices = ordered[
                        start : start + config.batch_size
                    ]
                    batch = _collate_trusted_indices(
                        dataset,
                        batch_indices,
                        encoder,
                        encoded_cache=encoded_cache,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss = compute_warm_start_loss(
                        model,
                        batch,
                        behavior_cloning_weight=(
                            config.behavior_cloning_weight
                        ),
                        q_regression_weight=config.q_regression_weight,
                    )
                    loss.total.backward()
                    optimizer.step()
                    total_losses.append(
                        float(loss.total.detach().item())
                    )
                    bc_losses.append(
                        float(loss.behavior_cloning.detach().item())
                    )
                    q_losses.append(
                        float(loss.q_regression.detach().item())
                    )
                    bc_label_exposures += loss.bc_label_count
                    critic_examples += loss.critic_example_count
            model.eval()
    finally:
        torch.use_deterministic_algorithms(
            prior_deterministic, warn_only=prior_warn_only
        )
        torch.set_default_dtype(prior_dtype)

    if encoder_contract_record(encoder) != encoder_record:
        raise TrainingValidationError(
            "encoder contract or frozen state changed during training"
        )
    expected_critic_examples = (
        len(training_indices) * config.epochs
    )
    if critic_examples != expected_critic_examples:
        raise AssertionError("critic did not consume every training example")
    if bc_label_exposures != bc_labeled_examples * config.epochs:
        raise AssertionError(
            "policy heads did not consume every eligible teacher label"
        )
    shape = ModelShapeContract(
        architecture=CONTROLLER_ARCHITECTURE_VERSION,
        feature_dim=feature_dim,
        policy_verb_count=4,
        hidden_dim=model_shape["hidden_dim"],
        repr_dim=model_shape["repr_dim"],
        node_repr_dim=model_shape["node_repr_dim"],
        verb_embed_dim=model_shape["verb_embed_dim"],
    )
    validate_torch_state_dict(model.state_dict(), shape)
    state_dict_sha256 = torch_state_dict_sha256(model.state_dict())
    checkpoint_contract = dataset.checkpoint_contract(split)
    metrics = WarmStartMetrics(
        algorithm=WARM_START_ALGORITHM_VERSION,
        training_examples=len(training_indices),
        bc_labeled_examples=bc_labeled_examples,
        bc_label_exposures=bc_label_exposures,
        critic_examples=critic_examples,
        updates=len(total_losses),
        total_losses=tuple(total_losses),
        behavior_cloning_losses=tuple(bc_losses),
        q_regression_losses=tuple(q_losses),
    )
    result = WarmStartResult(
        model=model,
        encoder=encoder,
        encoder_contract=encoder_record,
        model_config=model_shape,
        feature_dim=feature_dim,
        gamma=dataset.gamma,
        environment_contract_sha256=(
            dataset.environment_contract_sha256
        ),
        collection_contract_sha256=(
            dataset.collection_contract_sha256
        ),
        dataset_sha256=dataset.dataset_sha256,
        split_sha256=checkpoint_contract["split_sha256"],
        state_dict_sha256=state_dict_sha256,
        config=config,
        metrics=metrics,
    )
    object.__setattr__(result, "_trust_seal", _RESULT_TRUST_SEAL)
    object.__setattr__(
        result, "_trusted_binding", _warm_start_result_binding(result)
    )
    result.validate_trusted()
    return result


def build_phase_b_warm_start_checkpoint(
    result: WarmStartResult,
    *,
    dataset: PhaseBDataset,
    split: GroupedSplit,
    environment: BoundEnvironment,
    training_collection_contract: CollectionContract,
) -> dict[str, object]:
    """Bind trained weights to exact data, encoder, and MDP contracts."""
    _require_torch()
    if not isinstance(result, WarmStartResult):
        raise TypeError("checkpoint creation needs WarmStartResult")
    result.validate_trusted()
    if not isinstance(result.model, FactoredControllerNetwork):
        raise TrainingValidationError(
            "warm-start result model has an invalid type"
        )
    _validated_dataset(dataset, split)
    if not isinstance(environment, BoundEnvironment):
        raise TypeError("checkpoint environment must be BoundEnvironment")
    if not isinstance(training_collection_contract, CollectionContract):
        raise TypeError(
            "training_collection_contract must be CollectionContract"
        )
    environment.validate_binding()
    if environment.sha256 != result.environment_contract_sha256 or (
        dataset.environment_contract_sha256 != environment.sha256
    ):
        raise TrainingValidationError(
            "training result environment differs from checkpoint environment"
        )
    if training_collection_contract.sha256 != (
        result.collection_contract_sha256
    ) or dataset.collection_contract_sha256 != (
        training_collection_contract.sha256
    ):
        raise TrainingValidationError(
            "training result collection differs from checkpoint collection"
        )
    dataset_contract = dataset.checkpoint_contract(split)
    if (
        dataset_contract["dataset_sha256"] != result.dataset_sha256
        or dataset_contract["split_sha256"] != result.split_sha256
    ):
        raise TrainingValidationError(
            "training result data binding differs from checkpoint data"
        )
    if dataset.gamma != result.gamma:
        raise TrainingValidationError(
            "training result gamma differs from its trusted dataset"
        )
    current_encoder = encoder_contract_record(result.encoder)
    if current_encoder != result.encoder_contract:
        raise TrainingValidationError(
            "training encoder changed before checkpoint creation"
        )
    state_dict = {
        key: value.detach().cpu().clone()
        for key, value in result.model.state_dict().items()
    }
    if torch_state_dict_sha256(state_dict) != result.state_dict_sha256:
        raise TrainingValidationError(
            "trained model changed during checkpoint snapshot"
        )
    return build_checkpoint(
        state_dict=state_dict,
        environment_contract=environment.contract,
        training_collection_contract=training_collection_contract,
        encoder_contract=current_encoder,
        model_config=result.model_config,
        feature_dim=result.feature_dim,
        training_target_contract={
            "target_type": TRAINING_TARGET_VERSION,
            "gamma": float(result.gamma),
            "reward_contract": REWARD_CONTRACT_VERSION,
        },
        dataset_contract=dataset_contract,
        model_state_validator=validate_torch_state_dict,
        model_state_digester=torch_state_dict_sha256,
    )


__all__ = [
    "PhaseBTrainingBatch",
    "TrainingValidationError",
    "WARM_START_ALGORITHM_VERSION",
    "WarmStartConfig",
    "WarmStartLoss",
    "WarmStartMetrics",
    "WarmStartResult",
    "build_phase_b_warm_start_checkpoint",
    "collate_phase_b_training_batch",
    "compute_warm_start_loss",
    "train_phase_b_warm_start",
]
