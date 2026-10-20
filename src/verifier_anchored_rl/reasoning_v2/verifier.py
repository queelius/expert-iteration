"""Claim-kind verifier boundary and deterministic test implementation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

from .model import (
    AggregatePayload,
    CheckPurpose,
    Claim,
    ClaimKind,
    ClaimPayload,
    Connective,
    DerivationPayload,
    GoalObligation,
    Graph,
    RefutationPayload,
    ResultPayload,
    SolutionPayload,
    SymbolicCheck,
    VerifierContext,
    get,
)


@dataclass(frozen=True)
class VerificationDecision:
    verification: SymbolicCheck
    discharge: SymbolicCheck | None

    def __post_init__(self) -> None:
        if not isinstance(self.verification, SymbolicCheck):
            raise ValueError("verification decision has an invalid validity check")
        if self.discharge is not None and not isinstance(
            self.discharge, SymbolicCheck
        ):
            raise ValueError("verification decision has an invalid discharge check")
        if self.verification.accepted and self.discharge is None:
            raise ValueError("accepted validity requires a discharge check")
        if not self.verification.accepted and self.discharge is not None:
            raise ValueError("rejected validity cannot have a discharge check")
        if self.discharge is not None and (
            self.verification.context.contract_id
            != self.discharge.context.contract_id
        ):
            raise ValueError(
                "validity and discharge must use the same verifier contract"
            )


@runtime_checkable
class ClaimVerifier(Protocol):
    def verify(self, graph: Graph, claim: Claim) -> VerificationDecision:
        """Check immutable claim validity and owner-goal discharge."""
        ...


@runtime_checkable
class GenerationContract(Protocol):
    """Canonicalize generated structures before they enter immutable state."""

    generation_contract_id: str
    obligation_contract_id: str
    payload_semantics_id: str

    def canonicalize_obligation(
        self, obligation: GoalObligation
    ) -> GoalObligation:
        ...

    def canonicalize_payload(
        self,
        owner: GoalObligation,
        kind: ClaimKind,
        payload: ClaimPayload,
    ) -> ClaimPayload:
        ...

    def validate_decomposition(
        self,
        parent: GoalObligation,
        connective: Connective,
        children: tuple[GoalObligation, ...],
    ) -> tuple[GoalObligation, ...]:
        ...


@runtime_checkable
class VerifierReceiptContract(Protocol):
    """Pin verifier identity and bounded rejection codes for replay."""

    contract_id: str
    validity_rejection_codes: frozenset[str]
    discharge_rejection_codes: frozenset[str]


@dataclass(frozen=True)
class MockRule:
    kind: ClaimKind
    statement: str
    verification_accepted: bool
    discharge_accepted: bool = False
    owner_obligation: GoalObligation | None = None
    observed: str | None = None

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError("mock verifier rule statement must be nonempty")
        if self.owner_obligation is not None and not isinstance(
            self.owner_obligation, GoalObligation
        ):
            raise ValueError("mock verifier owner obligation has an invalid type")
        if self.discharge_accepted and not self.verification_accepted:
            raise ValueError("discharge cannot pass when verification fails")


def verifier_context(
    graph: Graph,
    claim: Claim,
    purpose: CheckPurpose,
    contract_id: str,
) -> VerifierContext:
    """Return a stable context identity derived only from immutable state."""
    owner = get(graph, claim.owner_goal)
    material = "\x1f".join((
        contract_id,
        purpose.value,
        claim.kind.value,
        str(claim.owner_goal),
        repr(owner.obligation),
        claim.statement,
        repr(claim.payload),
        repr(claim.premises),
    ))
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return VerifierContext(
        contract_id=contract_id,
        purpose=purpose,
        claim_kind=claim.kind,
        owner_goal=claim.owner_goal,
        context_digest=digest,
    )


@dataclass(frozen=True)
class TableDrivenVerifier:
    """Exact rule table for action-algebra and invariant tests."""

    rules: tuple[MockRule, ...]
    contract_id: str = "table-driven-claim-verifier-v1"
    generation_contract_id: ClassVar[str] = "table-driven-generation-v1"
    obligation_contract_id: ClassVar[str] = "table-driven-obligation-v1"
    payload_semantics_id: ClassVar[str] = "flat-claim-payload-v2"
    validity_rejection_codes: ClassVar[frozenset[str]] = frozenset(
        {"no_matching_rule", "mock_verification_rejected"}
    )
    discharge_rejection_codes: ClassVar[frozenset[str]] = frozenset(
        {"mock_discharge_rejected"}
    )

    def __post_init__(self) -> None:
        if not self.contract_id.strip():
            raise ValueError("verifier contract id must be nonempty")
        keys = [
            (rule.kind, rule.statement, rule.owner_obligation)
            for rule in self.rules
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("mock verifier rules contain duplicate keys")

    def _rule(self, graph: Graph, claim: Claim) -> MockRule | None:
        owner = get(graph, claim.owner_goal)
        exact = [
            rule
            for rule in self.rules
            if rule.kind is claim.kind
            and rule.statement == claim.statement
            and rule.owner_obligation == owner.obligation
        ]
        if exact:
            return exact[0]
        generic = [
            rule
            for rule in self.rules
            if rule.kind is claim.kind
            and rule.statement == claim.statement
            and rule.owner_obligation is None
        ]
        return generic[0] if generic else None

    def canonicalize_obligation(
        self, obligation: GoalObligation
    ) -> GoalObligation:
        if not isinstance(obligation, GoalObligation):
            raise ValueError("obligation has an invalid type")
        return obligation

    def canonicalize_payload(
        self,
        owner: GoalObligation,
        kind: ClaimKind,
        payload: ClaimPayload,
    ) -> ClaimPayload:
        if not isinstance(owner, GoalObligation):
            raise ValueError("payload owner has an invalid type")
        expected = {
            ClaimKind.SOLUTION: SolutionPayload,
            ClaimKind.RESULT: ResultPayload,
            ClaimKind.DERIVATION: DerivationPayload,
            ClaimKind.AGGREGATE: AggregatePayload,
            ClaimKind.REFUTATION: RefutationPayload,
        }
        if not isinstance(kind, ClaimKind) or not isinstance(
            payload, expected[kind]
        ):
            raise ValueError("payload kind has an invalid type")
        return payload

    def validate_decomposition(
        self,
        parent: GoalObligation,
        connective: Connective,
        children: tuple[GoalObligation, ...],
    ) -> tuple[GoalObligation, ...]:
        if not isinstance(parent, GoalObligation):
            raise ValueError("decomposition parent has an invalid type")
        if not isinstance(connective, Connective):
            raise ValueError("decomposition connective has an invalid type")
        if type(children) is not tuple or any(
            not isinstance(child, GoalObligation) for child in children
        ):
            raise ValueError("decomposition children have an invalid type")
        return children

    def verify(self, graph: Graph, claim: Claim) -> VerificationDecision:
        validity_context = verifier_context(
            graph, claim, CheckPurpose.VALIDITY, self.contract_id)
        rule = self._rule(graph, claim)
        observed = claim.statement if rule is None else (
            rule.observed if rule.observed is not None else claim.statement
        )
        if rule is None:
            return VerificationDecision(
                verification=SymbolicCheck(
                    context=validity_context,
                    accepted=False,
                    observed=observed,
                    reason_code="no_matching_rule",
                ),
                discharge=None,
            )
        if not rule.verification_accepted:
            return VerificationDecision(
                verification=SymbolicCheck(
                    context=validity_context,
                    accepted=False,
                    observed=observed,
                    reason_code="mock_verification_rejected",
                ),
                discharge=None,
            )
        verification = SymbolicCheck(
            context=validity_context,
            accepted=True,
            observed=observed,
        )
        discharge_context = verifier_context(
            graph, claim, CheckPurpose.OWNER_DISCHARGE, self.contract_id)
        discharge = SymbolicCheck(
            context=discharge_context,
            accepted=rule.discharge_accepted,
            observed=observed,
            reason_code=(
                None if rule.discharge_accepted else "mock_discharge_rejected"
            ),
        )
        return VerificationDecision(
            verification=verification,
            discharge=discharge,
        )


__all__ = [
    "ClaimVerifier",
    "GenerationContract",
    "MockRule",
    "TableDrivenVerifier",
    "VerifierReceiptContract",
    "VerificationDecision",
    "verifier_context",
]
