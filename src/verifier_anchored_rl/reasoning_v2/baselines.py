"""Content-free exact legal-menu comparator for Phase B v2."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from .contracts import canonical_json_bytes
from .dataset import (
    DatasetValidationError,
    GroupedSplit,
    PhaseBDataset,
    PhaseBExample,
)


BASELINE_SCHEMA_VERSION = "exact-legal-menu-lookup-v2"
TIE_BREAK_VERSION = (
    "local-count-then-global-legal-pair-count-then-"
    "canonical-menu-order-v2"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

ActionKey = tuple[str, int]
MenuSignature = tuple[ActionKey, ...]


class BaselineValidationError(ValueError):
    """Raised when trusted data cannot support the exact comparator."""


def _digest(value: object, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise BaselineValidationError(f"{path} must be a SHA-256 digest")
    return value


def _count(value: object, path: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if type(value) is not int or value < minimum:
        qualifier = "positive" if positive else "nonnegative"
        raise BaselineValidationError(
            f"{path} must be a {qualifier} integer"
        )
    return value


def _rate(value: object, path: str) -> float:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or value < 0.0
        or value > 1.0
    ):
        raise BaselineValidationError(f"{path} must be a finite rate")
    return value


@dataclass(frozen=True, slots=True)
class LegalMenuLookupResult:
    """Immutable canonical summary of one grouped-split comparison."""

    dataset_sha256: str
    split_sha256: str
    training_label_count: int
    validation_label_count: int
    training_menu_signature_count: int
    correct_count: int
    accuracy: float
    seen_menu_label_count: int
    seen_menu_coverage: float
    seen_menu_correct_count: int
    seen_menu_accuracy: float | None
    uniform_expected_accuracy: float
    schema: str = BASELINE_SCHEMA_VERSION
    tie_break: str = TIE_BREAK_VERSION

    def __post_init__(self) -> None:
        if self.schema != BASELINE_SCHEMA_VERSION:
            raise BaselineValidationError(
                "baseline result schema differs from runtime"
            )
        if self.tie_break != TIE_BREAK_VERSION:
            raise BaselineValidationError(
                "baseline tie-break contract differs from runtime"
            )
        _digest(self.dataset_sha256, "baseline dataset digest")
        _digest(self.split_sha256, "baseline split digest")
        training_labels = _count(
            self.training_label_count,
            "baseline training_label_count",
            positive=True,
        )
        validation_labels = _count(
            self.validation_label_count,
            "baseline validation_label_count",
            positive=True,
        )
        signatures = _count(
            self.training_menu_signature_count,
            "baseline training_menu_signature_count",
            positive=True,
        )
        correct = _count(self.correct_count, "baseline correct_count")
        seen = _count(
            self.seen_menu_label_count,
            "baseline seen_menu_label_count",
        )
        seen_correct = _count(
            self.seen_menu_correct_count,
            "baseline seen_menu_correct_count",
        )
        if signatures > training_labels:
            raise BaselineValidationError(
                "training menu signatures exceed training labels"
            )
        if correct > validation_labels or seen > validation_labels:
            raise BaselineValidationError(
                "baseline result counts exceed validation labels"
            )
        if seen_correct > seen or seen_correct > correct:
            raise BaselineValidationError(
                "seen-menu correct count exceeds its containing count"
            )
        accuracy = _rate(self.accuracy, "baseline accuracy")
        coverage = _rate(
            self.seen_menu_coverage,
            "baseline seen_menu_coverage",
        )
        uniform = _rate(
            self.uniform_expected_accuracy,
            "baseline uniform_expected_accuracy",
        )
        if accuracy != correct / validation_labels:
            raise BaselineValidationError(
                "baseline accuracy differs from its counts"
            )
        if coverage != seen / validation_labels:
            raise BaselineValidationError(
                "seen-menu coverage differs from its counts"
            )
        if uniform <= 0.0:
            raise BaselineValidationError(
                "uniform expected accuracy must be positive"
            )
        if seen == 0:
            if self.seen_menu_accuracy is not None:
                raise BaselineValidationError(
                    "unseen-only evaluation needs null seen-menu accuracy"
                )
        else:
            seen_accuracy = _rate(
                self.seen_menu_accuracy,
                "baseline seen_menu_accuracy",
            )
            if seen_accuracy != seen_correct / seen:
                raise BaselineValidationError(
                    "seen-menu accuracy differs from its counts"
                )

    def to_record(self) -> dict[str, object]:
        """Return the one exact JSON representation of this result."""
        return {
            "schema": self.schema,
            "tie_break": self.tie_break,
            "dataset_sha256": self.dataset_sha256,
            "split_sha256": self.split_sha256,
            "training_label_count": self.training_label_count,
            "validation_label_count": self.validation_label_count,
            "training_menu_signature_count": (
                self.training_menu_signature_count
            ),
            "correct_count": self.correct_count,
            "accuracy": self.accuracy,
            "seen_menu_label_count": self.seen_menu_label_count,
            "seen_menu_coverage": self.seen_menu_coverage,
            "seen_menu_correct_count": self.seen_menu_correct_count,
            "seen_menu_accuracy": self.seen_menu_accuracy,
            "uniform_expected_accuracy": self.uniform_expected_accuracy,
        }

    @property
    def canonical_bytes(self) -> bytes:
        """Return stable canonical JSON bytes for logging and comparison."""
        return canonical_json_bytes(self.to_record())


def _trusted_binding(
    dataset: PhaseBDataset,
    split: GroupedSplit,
) -> tuple[str, str]:
    if type(dataset) is not PhaseBDataset:
        raise TypeError("baseline requires an exact PhaseBDataset")
    if type(split) is not GroupedSplit:
        raise TypeError("baseline requires an exact GroupedSplit")
    try:
        contract = dataset.checkpoint_contract(split)
    except DatasetValidationError as exc:
        raise BaselineValidationError(
            f"baseline data binding is not trusted: {exc}"
        ) from exc
    return (
        _digest(contract["dataset_sha256"], "baseline dataset digest"),
        _digest(contract["split_sha256"], "baseline split digest"),
    )


def _menu(example: PhaseBExample) -> MenuSignature:
    return tuple(example.current_features.legal_pairs)


def _chosen(example: PhaseBExample) -> ActionKey:
    return (example.action.verb.value, example.action.target)


def _predict(
    menu: MenuSignature,
    local_counts: Counter[ActionKey],
    global_counts: Counter[ActionKey],
) -> ActionKey:
    if not menu:
        raise BaselineValidationError(
            "a provenance-eligible example has an empty legal menu"
        )
    return max(
        enumerate(menu),
        key=lambda indexed: (
            local_counts[indexed[1]],
            global_counts[indexed[1]],
            -indexed[0],
        ),
    )[1]


def legal_menu_lookup_baseline(
    dataset: PhaseBDataset,
    split: GroupedSplit,
) -> LegalMenuLookupResult:
    """Evaluate the exact ordered-menu lookup on one trusted grouped split."""
    dataset_digest, split_digest = _trusted_binding(dataset, split)

    menu_counts: dict[
        MenuSignature, Counter[ActionKey]
    ] = defaultdict(Counter)
    global_counts: Counter[ActionKey] = Counter()
    training_label_count = 0
    for example in split.train_examples:
        if not example.bc_eligible:
            continue
        menu = _menu(example)
        chosen = _chosen(example)
        if chosen not in menu:
            raise BaselineValidationError(
                "a training label is absent from its exact legal menu"
            )
        menu_counts[menu][chosen] += 1
        global_counts[chosen] += 1
        training_label_count += 1
    if training_label_count == 0:
        raise BaselineValidationError(
            "training split has no provenance-eligible labels"
        )

    validation_label_count = 0
    correct_count = 0
    seen_menu_label_count = 0
    seen_menu_correct_count = 0
    uniform_total = 0.0
    for example in split.validation_examples:
        if not example.bc_eligible:
            continue
        menu = _menu(example)
        chosen = _chosen(example)
        if chosen not in menu:
            raise BaselineValidationError(
                "a validation label is absent from its exact legal menu"
            )
        seen = menu in menu_counts
        predicted = _predict(
            menu,
            menu_counts.get(menu, Counter()),
            global_counts,
        )
        correct = predicted == chosen
        validation_label_count += 1
        correct_count += int(correct)
        seen_menu_label_count += int(seen)
        seen_menu_correct_count += int(seen and correct)
        uniform_total += 1.0 / len(menu)
    if validation_label_count == 0:
        raise BaselineValidationError(
            "validation split has no provenance-eligible labels"
        )

    if _trusted_binding(dataset, split) != (dataset_digest, split_digest):
        raise BaselineValidationError(
            "baseline data binding changed during evaluation"
        )
    return LegalMenuLookupResult(
        dataset_sha256=dataset_digest,
        split_sha256=split_digest,
        training_label_count=training_label_count,
        validation_label_count=validation_label_count,
        training_menu_signature_count=len(menu_counts),
        correct_count=correct_count,
        accuracy=correct_count / validation_label_count,
        seen_menu_label_count=seen_menu_label_count,
        seen_menu_coverage=(
            seen_menu_label_count / validation_label_count
        ),
        seen_menu_correct_count=seen_menu_correct_count,
        seen_menu_accuracy=(
            None
            if seen_menu_label_count == 0
            else seen_menu_correct_count / seen_menu_label_count
        ),
        uniform_expected_accuracy=uniform_total / validation_label_count,
    )


__all__ = [
    "BASELINE_SCHEMA_VERSION",
    "BaselineValidationError",
    "LegalMenuLookupResult",
    "TIE_BREAK_VERSION",
    "legal_menu_lookup_baseline",
]
