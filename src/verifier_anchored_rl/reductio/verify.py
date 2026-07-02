"""The Verifier protocol and a mock implementation for tests.

No domain library is imported here. Real verifiers live in reductio.adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class VerifierResult:
    success: bool
    observed: str


@runtime_checkable
class Verifier(Protocol):
    def check(self, statement: str, verb: str, expected: str | None) -> VerifierResult:
        ...


@dataclass(frozen=True)
class MockVerifier:
    table: tuple[tuple[str, bool], ...] = ()

    def check(self, statement: str, verb: str, expected: str | None) -> VerifierResult:
        success = any(s == statement and ok for s, ok in self.table)
        return VerifierResult(success=success, observed=statement)
