"""Controller choices and immutable collection provenance."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .model import Action


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class BehaviorSource(str, Enum):
    TEACHER = "teacher"
    LEARNED_POLICY = "learned_policy"


@dataclass(frozen=True, slots=True)
class ControllerChoice:
    """One action selected by a bound controller adapter."""

    action: Action | None

    def __post_init__(self) -> None:
        if self.action is not None and not isinstance(self.action, Action):
            raise ValueError("controller choice action has an invalid type")


@dataclass(frozen=True, slots=True)
class DecisionProvenance:
    """Behavior labels derived from a bound controller choice."""

    source: BehaviorSource
    bc_eligible: bool
    fallback: bool
    diagnostic_record_id: str | None
    episode_key: str | None = None
    observation_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, BehaviorSource):
            raise ValueError("decision source has an invalid type")
        if type(self.bc_eligible) is not bool or type(self.fallback) is not bool:
            raise ValueError("decision provenance flags must be boolean")
        expected_bc = (
            self.source is BehaviorSource.TEACHER and not self.fallback
        )
        if self.bc_eligible is not expected_bc:
            raise ValueError(
                "bc_eligible must mean a non-fallback teacher decision"
            )
        if (
            self.source is BehaviorSource.LEARNED_POLICY
            and self.fallback
        ):
            raise ValueError("learned-policy behavior cannot be a fallback")
        if self.diagnostic_record_id is not None and (
            not isinstance(self.diagnostic_record_id, str)
            or not self.diagnostic_record_id.strip()
        ):
            raise ValueError(
                "decision diagnostic record id must be null or nonempty"
            )
        bound_fields = (self.episode_key, self.observation_sha256)
        if self.diagnostic_record_id is None:
            if any(value is not None for value in bound_fields):
                raise ValueError(
                    "diagnostic-null provenance cannot bind a receipt context"
                )
        else:
            if (
                not isinstance(self.episode_key, str)
                or not self.episode_key.strip()
            ):
                raise ValueError(
                    "diagnostic provenance needs a nonempty episode_key"
                )
            if (
                not isinstance(self.observation_sha256, str)
                or _SHA256_RE.fullmatch(self.observation_sha256) is None
            ):
                raise ValueError(
                    "diagnostic provenance needs an observation digest"
                )

    @classmethod
    def from_choice(
        cls,
        source: BehaviorSource,
        choice: ControllerChoice,
        *,
        fallback: bool,
        diagnostic_record_id: str | None = None,
        episode_key: str | None = None,
        observation_sha256: str | None = None,
    ) -> DecisionProvenance:
        if not isinstance(choice, ControllerChoice):
            raise TypeError("decision provenance needs a ControllerChoice")
        return cls(
            source=source,
            bc_eligible=(
                source is BehaviorSource.TEACHER and not fallback
            ),
            fallback=fallback,
            diagnostic_record_id=diagnostic_record_id,
            episode_key=episode_key,
            observation_sha256=observation_sha256,
        )


__all__ = [
    "BehaviorSource",
    "ControllerChoice",
    "DecisionProvenance",
]
