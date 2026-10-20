"""Fail-closed SymPy adapter for the casework-v1 claim contract.

The stdlib-only reasoning core does not import this module. SymPy is loaded
lazily so importing :mod:`verifier_anchored_rl.reasoning_v2.model` keeps the
core dependency boundary intact.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from .model import (
    AggregatePayload,
    CheckPurpose,
    Claim,
    ClaimKind,
    ClaimPayload,
    Connective,
    DerivationPayload,
    Evidence,
    Goal,
    GoalObjective,
    GoalObligation,
    Graph,
    Reducer,
    RefutationMode,
    RefutationPayload,
    Relation,
    ResultPayload,
    SolutionPayload,
    SymbolicCheck,
    VerifierContext,
    direct_child_goals,
    get,
    goal_ancestry,
    is_active,
)
from .verifier import VerificationDecision


CONTRACT_ID = "casework-claim-verifier-v1"
CANONICALIZATION_CONTRACT_ID = "casework-goal-obligation-v1"
PAYLOAD_SEMANTICS_ID = "flat-claim-payload-v2"
AGGREGATE_CONTRACT_ID = "casework-exact-aggregate-v1"

CONTRACT_ID_V11 = "casework-claim-verifier-v1.1"
CANONICALIZATION_CONTRACT_ID_V11 = "casework-goal-obligation-v1.1"
AGGREGATE_CONTRACT_ID_V11 = "casework-exact-aggregate-v1.1"
DOMAIN_GRAMMAR_IMPLEMENTATION_ID = "casework-domain-parser-profiled-regex-v1"


@dataclass(frozen=True)
class DomainGrammarProfile:
    """Immutable selector for exact bounded-domain parsing semantics."""

    profile_id: str
    bounded_comparator_pattern: str
    implementation_id: str = DOMAIN_GRAMMAR_IMPLEMENTATION_ID

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("domain grammar profile id must be nonempty")
        if self.bounded_comparator_pattern not in {"<|<=", "<=|<"}:
            raise ValueError("unsupported bounded comparator pattern")
        if (
            not isinstance(self.implementation_id, str)
            or not self.implementation_id.strip()
        ):
            raise ValueError("domain grammar implementation id must be nonempty")


LEGACY_DOMAIN_GRAMMAR_PROFILE = DomainGrammarProfile(
    profile_id="casework-domain-grammar-v1",
    bounded_comparator_pattern="<|<=",
)
CORRECTED_DOMAIN_GRAMMAR_PROFILE = DomainGrammarProfile(
    profile_id="casework-domain-grammar-v1.1",
    bounded_comparator_pattern="<=|<",
)

VALIDITY_REJECTION_CODES = frozenset(
    {
        "aggregate_has_no_live_solution",
        "aggregate_requires_exhaustive_cases",
        "aggregate_result_mismatch",
        "candidate_not_feasible",
        "candidate_not_outside_domain",
        "candidate_not_unique_raw_solution",
        "child_not_verified",
        "committed_children_mismatch",
        "identity_requires_one_solution",
        "malformed_claim",
        "malformed_equation",
        "malformed_scalar",
        "malformed_variable",
        "owner_not_empty",
        "owner_not_goal",
        "payload_kind_mismatch",
        "premise_not_active",
        "premise_not_trusted",
        "premise_not_verified",
        "reducer_objective_mismatch",
        "relation_rejected",
        "result_objective_mismatch",
        "result_objective_unsupported",
        "result_value_mismatch",
        "uncommitted_child",
        "unsupported_child_claim",
        "unsupported_domain",
        "unsupported_equation",
        "unsupported_refutation_mode",
        "unsupported_solution_set",
        "variable_mismatch",
    }
)

DISCHARGE_REJECTION_CODES = frozenset(
    {
        "child_equation_not_equivalent",
        "child_not_atomic",
        "child_objective_not_branch",
        "decomposition_domain_not_exhaustive",
        "decomposition_domain_overlap",
        "decomposition_needs_multiple_children",
        "decomposition_not_exhaustive",
        "decomposition_overlap",
        "decomposition_rejected",
        "decomposition_variable_mismatch",
        "derivation_not_discharging",
        "malformed_decomposition",
        "not_unique_owner_solution",
        "owner_not_branch",
        "owner_reduction_mismatch",
    }
)


def _sympy():
    import sympy  # type: ignore[import-not-found]

    return sympy


class _AdapterError(ValueError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True)
class CoverageDecision:
    accepted: bool
    context_digest: str
    reason_code: str | None = None


@dataclass(frozen=True)
class _KindDecision:
    verification_accepted: bool
    verification_reason: str | None
    discharge_accepted: bool | None
    discharge_reason: str | None

    def __post_init__(self) -> None:
        if self.verification_accepted == (self.discharge_accepted is None):
            raise ValueError("validity and discharge decision are inconsistent")


_VARIABLE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_SCALAR = re.compile(r"^[0-9+\-*/().\s]+$")
_EQUATION_CHARS = re.compile(r"^[A-Za-z0-9_+\-*/().,<>=\s]+$")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


_OBJECTIVES = {
    ("f1a", "sum_of_solutions"): GoalObjective.SUM,
    ("f1a", "largest_solution"): GoalObjective.MAX,
    ("f1b", "find_x"): GoalObjective.IDENTITY,
    ("f2a", "find_x"): GoalObjective.IDENTITY,
    ("f2b", "sum_of_solutions"): GoalObjective.SUM,
    ("f3", "largest_solution"): GoalObjective.MAX,
    ("f3", "smallest_solution"): GoalObjective.MIN,
}


def _build_casework_root_obligation(
    record: Mapping[str, Any],
    profile: DomainGrammarProfile,
) -> GoalObligation:
    try:
        family = record["family"]
        question_type = record["question_type"]
        equation = record["symbolic"]
    except (KeyError, TypeError) as exc:
        raise ValueError("missing_casework_metadata") from exc

    objective = _OBJECTIVES.get((family, question_type))
    if objective is None:
        raise ValueError("unsupported_casework_contract")
    if not isinstance(equation, str) or not equation.strip():
        raise ValueError("malformed_equation")

    domain = "Reals"
    if family == "f1b":
        try:
            params = record["difficulty"]["params"]
            direction = params["direction"]
            boundary = str(params["d"])
        except (KeyError, TypeError) as exc:
            raise ValueError("missing_casework_domain") from exc
        try:
            _parse_scalar(boundary)
        except _AdapterError as exc:
            raise ValueError(exc.reason_code) from exc
        symbols = {"lt": "<", "gt": ">"}
        if direction not in symbols:
            raise ValueError("unsupported_casework_domain")
        domain = f"x {symbols[direction]} {boundary}"

    obligation = GoalObligation(
        variable="x",
        equation=equation,
        domain=domain,
        objective=objective,
    )
    try:
        return _canonicalize_goal_obligation(obligation, profile)
    except _AdapterError as exc:
        raise ValueError(exc.reason_code) from exc


def build_casework_root_obligation(
    record: Mapping[str, Any],
) -> GoalObligation:
    """Build a trusted root obligation under the exact legacy profile."""
    return _build_casework_root_obligation(
        record, LEGACY_DOMAIN_GRAMMAR_PROFILE
    )


def build_casework_root_obligation_v11(
    record: Mapping[str, Any],
) -> GoalObligation:
    """Build a trusted root obligation under the corrected v1.1 profile."""
    return _build_casework_root_obligation(
        record, CORRECTED_DOMAIN_GRAMMAR_PROFILE
    )


def _parse_scalar(text: str):
    if (
        not isinstance(text, str)
        or not text.strip()
        or len(text) > 128
        or _SCALAR.fullmatch(text) is None
        or "**" in text
        or "//" in text
    ):
        raise _AdapterError("malformed_scalar")
    sp = _sympy()
    try:
        value = sp.sympify(text, rational=True)
    except Exception as exc:
        raise _AdapterError("malformed_scalar") from exc
    if (
        getattr(value, "free_symbols", None)
        or value.is_number is not True
        or value.is_real is not True
        or value.is_finite is not True
    ):
        raise _AdapterError("malformed_scalar")
    return sp.simplify(value)


def _parse_domain(
    variable: str,
    text: str,
    profile: DomainGrammarProfile,
):
    if not _VARIABLE.fullmatch(variable):
        raise _AdapterError("malformed_variable")
    if not isinstance(text, str):
        raise _AdapterError("unsupported_domain")
    sp = _sympy()
    if text.strip() == "Reals":
        return sp.S.Reals
    comparator = profile.bounded_comparator_pattern
    chain = re.fullmatch(
        rf"\s*(.*?)\s*({comparator})\s*{re.escape(variable)}"
        rf"\s*({comparator})\s*(.*?)\s*",
        text,
    )
    if chain is not None:
        lower = _parse_scalar(chain.group(1))
        upper = _parse_scalar(chain.group(4))
        return sp.Interval(
            lower,
            upper,
            left_open=chain.group(2) == "<",
            right_open=chain.group(3) == "<",
        )
    reverse = re.fullmatch(
        rf"\s*(.*?)\s*(<=|>=|<|>)\s*{re.escape(variable)}\s*", text
    )
    if reverse is not None:
        boundary = _parse_scalar(reverse.group(1))
        relation = reverse.group(2)
        if relation == "<":
            return sp.Interval.open(boundary, sp.oo)
        if relation == "<=":
            return sp.Interval(boundary, sp.oo)
        if relation == ">":
            return sp.Interval.open(-sp.oo, boundary)
        return sp.Interval(-sp.oo, boundary)
    match = re.fullmatch(
        rf"\s*{re.escape(variable)}\s*(<=|>=|<|>)\s*(.*?)\s*", text
    )
    if match is None:
        raise _AdapterError("unsupported_domain")
    boundary = _parse_scalar(match.group(2))
    relation = match.group(1)
    if relation == "<":
        return sp.Interval.open(-sp.oo, boundary)
    if relation == "<=":
        return sp.Interval(-sp.oo, boundary)
    if relation == ">":
        return sp.Interval.open(boundary, sp.oo)
    return sp.Interval(boundary, sp.oo)


def _parse_equation(variable: str, text: str):
    if (
        not _VARIABLE.fullmatch(variable)
        or not isinstance(text, str)
        or not text.strip()
        or len(text) > 2_000
        or _EQUATION_CHARS.fullmatch(text) is None
    ):
        raise _AdapterError("malformed_equation")
    allowed_names = {variable, "Eq", "Abs", "Piecewise", "True"}
    if set(_IDENTIFIER.findall(text)) - allowed_names:
        raise _AdapterError("unsupported_equation")
    sp = _sympy()
    symbol = sp.Symbol(variable, real=True)
    locals_ = {
        variable: symbol,
        "Eq": sp.Eq,
        "Abs": sp.Abs,
        "Piecewise": sp.Piecewise,
        "True": sp.true,
    }
    try:
        equation = sp.sympify(text, locals=locals_)
    except Exception as exc:
        raise _AdapterError("malformed_equation") from exc
    if not isinstance(equation, sp.Equality):
        raise _AdapterError("malformed_equation")
    if equation.free_symbols - {symbol}:
        raise _AdapterError("unsupported_equation")
    return symbol, equation


def _parse_obligation(
    obligation: GoalObligation,
    profile: DomainGrammarProfile,
):
    symbol, equation = _parse_equation(
        obligation.variable, obligation.equation
    )
    domain = _parse_domain(
        obligation.variable, obligation.domain, profile
    )
    return symbol, equation, domain


def _serialize_domain(variable: str, domain: Any) -> str:
    sp = _sympy()
    if domain == sp.S.Reals:
        return "Reals"
    if not isinstance(domain, sp.Interval):
        raise _AdapterError("unsupported_domain")
    if domain.start is sp.S.NegativeInfinity:
        relation = "<" if domain.right_open else "<="
        return f"{variable} {relation} {sp.sstr(domain.end)}"
    if domain.end is sp.S.Infinity:
        relation = ">" if domain.left_open else ">="
        return f"{variable} {relation} {sp.sstr(domain.start)}"
    left = "<" if domain.left_open else "<="
    right = "<" if domain.right_open else "<="
    return (
        f"{sp.sstr(domain.start)} {left} {variable} "
        f"{right} {sp.sstr(domain.end)}"
    )


def _canonicalize_goal_obligation(
    obligation: GoalObligation,
    profile: DomainGrammarProfile,
) -> GoalObligation:
    if not isinstance(obligation, GoalObligation):
        raise ValueError("invalid_obligation_type")
    if not isinstance(obligation.variable, str) or not _VARIABLE.fullmatch(
        obligation.variable
    ):
        raise ValueError("malformed_variable")
    if not isinstance(obligation.objective, GoalObjective):
        raise ValueError("invalid_objective")
    _, equation, domain = _parse_obligation(obligation, profile)
    return GoalObligation(
        variable=obligation.variable,
        equation=_sympy().sstr(equation),
        domain=_serialize_domain(obligation.variable, domain),
        objective=obligation.objective,
    )


def canonicalize_goal_obligation(
    obligation: GoalObligation,
) -> GoalObligation:
    """Canonicalize an obligation under the exact legacy profile."""
    return _canonicalize_goal_obligation(
        obligation, LEGACY_DOMAIN_GRAMMAR_PROFILE
    )


def canonicalize_goal_obligation_v11(
    obligation: GoalObligation,
) -> GoalObligation:
    """Canonicalize an obligation under the corrected v1.1 profile."""
    return _canonicalize_goal_obligation(
        obligation, CORRECTED_DOMAIN_GRAMMAR_PROFILE
    )


def canonicalize_solution_payload(payload: SolutionPayload) -> SolutionPayload:
    """Return the canonical variable and ground scalar solution payload."""
    if not isinstance(payload, SolutionPayload):
        raise ValueError("invalid_solution_payload_type")
    if not isinstance(payload.variable, str) or not _VARIABLE.fullmatch(
        payload.variable
    ):
        raise ValueError("malformed_variable")
    return SolutionPayload(
        payload.variable, _sympy().sstr(_parse_scalar(payload.value))
    )


def canonicalize_result_payload(payload: ResultPayload) -> ResultPayload:
    """Return a canonical direct objective-result payload."""
    if not isinstance(payload, ResultPayload):
        raise ValueError("invalid_result_payload_type")
    if (
        not isinstance(payload.objective, GoalObjective)
        or payload.objective is GoalObjective.BRANCH
    ):
        raise ValueError("invalid_result_objective")
    return ResultPayload(
        payload.objective, _sympy().sstr(_parse_scalar(payload.value))
    )


def canonicalize_derivation_payload(
    payload: DerivationPayload,
) -> DerivationPayload:
    """Canonicalize the adapter's intentionally ground-only derivation."""
    if not isinstance(payload, DerivationPayload):
        raise ValueError("invalid_derivation_payload_type")
    if not isinstance(payload.relation, Relation):
        raise ValueError("invalid_relation")
    sp = _sympy()
    return DerivationPayload(
        sp.sstr(_parse_scalar(payload.lhs)),
        payload.relation,
        sp.sstr(_parse_scalar(payload.rhs)),
    )


def canonicalize_aggregate_payload(
    payload: AggregatePayload,
) -> AggregatePayload:
    """Return a canonical reducer and ground scalar aggregate payload."""
    if not isinstance(payload, AggregatePayload):
        raise ValueError("invalid_aggregate_payload_type")
    if not isinstance(payload.reducer, Reducer):
        raise ValueError("invalid_reducer")
    return AggregatePayload(
        payload.reducer, _sympy().sstr(_parse_scalar(payload.result))
    )


def canonicalize_refutation_payload(
    payload: RefutationPayload,
) -> RefutationPayload:
    """Return a canonical typed out-of-domain refutation payload."""
    if not isinstance(payload, RefutationPayload):
        raise ValueError("invalid_refutation_payload_type")
    if not isinstance(payload.mode, RefutationMode):
        raise ValueError("invalid_refutation_mode")
    if not isinstance(payload.variable, str) or not _VARIABLE.fullmatch(
        payload.variable
    ):
        raise ValueError("malformed_variable")
    return RefutationPayload(
        payload.mode,
        payload.variable,
        _sympy().sstr(_parse_scalar(payload.candidate)),
    )


def canonicalize_claim_payload(payload: ClaimPayload) -> ClaimPayload:
    """Dispatch strict canonicalization by authoritative payload type."""
    if isinstance(payload, SolutionPayload):
        return canonicalize_solution_payload(payload)
    if isinstance(payload, ResultPayload):
        return canonicalize_result_payload(payload)
    if isinstance(payload, DerivationPayload):
        return canonicalize_derivation_payload(payload)
    if isinstance(payload, AggregatePayload):
        return canonicalize_aggregate_payload(payload)
    if isinstance(payload, RefutationPayload):
        return canonicalize_refutation_payload(payload)
    raise ValueError("unsupported_claim_payload_type")


def _solve_values(
    obligation: GoalObligation,
    profile: DomainGrammarProfile,
    *,
    reals: bool = False,
    domain_override: Any | None = None,
):
    sp = _sympy()
    symbol, equation, domain = _parse_obligation(obligation, profile)
    solve_domain = (
        domain_override
        if domain_override is not None
        else sp.S.Reals if reals else domain
    )
    try:
        solutions = sp.solveset(
            equation.lhs - equation.rhs, symbol, domain=solve_domain
        )
    except Exception as exc:
        raise _AdapterError("unsupported_solution_set") from exc
    if solutions is sp.S.EmptySet:
        return frozenset()
    if not isinstance(solutions, sp.FiniteSet):
        raise _AdapterError("unsupported_solution_set")
    values = []
    for value in solutions:
        value = sp.simplify(value)
        if value.is_real is not True or value.is_finite is not True:
            raise _AdapterError("unsupported_solution_set")
        values.append(value)
    return frozenset(values)


def _same_scalar(left: Any, right: Any) -> bool:
    sp = _sympy()
    try:
        return bool(sp.simplify(left - right) == 0)
    except Exception:
        return False


def _contains(values: frozenset[Any], candidate: Any) -> bool:
    return any(_same_scalar(value, candidate) for value in values)


def _reduce(values: tuple[Any, ...], reducer: Reducer):
    if not values:
        raise _AdapterError("aggregate_has_no_live_solution")
    sp = _sympy()
    if reducer is Reducer.SUM:
        return sp.simplify(sp.Add(*values))
    if reducer is Reducer.MAX:
        return sp.simplify(sp.Max(*values))
    if reducer is Reducer.MIN:
        return sp.simplify(sp.Min(*values))
    if len(values) != 1:
        raise _AdapterError("identity_requires_one_solution")
    return sp.simplify(values[0])


def _obligation_material(obligation: GoalObligation) -> dict[str, str]:
    return {
        "variable": obligation.variable,
        "equation": obligation.equation,
        "domain": obligation.domain,
        "objective": obligation.objective.value,
    }


def _payload_material(payload: Any) -> dict[str, Any]:
    if isinstance(payload, SolutionPayload):
        return {"type": "solution", "variable": payload.variable,
                "value": payload.value}
    if isinstance(payload, ResultPayload):
        return {"type": "result", "objective": payload.objective.value,
                "value": payload.value}
    if isinstance(payload, DerivationPayload):
        return {"type": "derivation", "lhs": payload.lhs,
                "relation": payload.relation.value, "rhs": payload.rhs}
    if isinstance(payload, AggregatePayload):
        return {"type": "aggregate", "reducer": payload.reducer.value,
                "result": payload.result}
    if isinstance(payload, RefutationPayload):
        return {"type": "refutation", "mode": payload.mode.value,
                "variable": payload.variable, "candidate": payload.candidate}
    return {"type": type(payload).__name__}


def _check_material(check: SymbolicCheck | None) -> dict[str, Any] | None:
    if check is None:
        return None
    return {
        "accepted": check.accepted,
        "contract_id": check.context.contract_id,
        "purpose": check.context.purpose.value,
        "context_digest": check.context.context_digest,
    }


def _premise_active_for_context(
    graph: Graph, current_claim: Claim | None, premise: Evidence | Claim
) -> bool:
    if is_active(graph, premise.id):
        return True
    if current_claim is None:
        return False
    owner = get(graph, current_claim.owner_goal)
    if not isinstance(owner, Goal) or owner.committed_claim != current_claim.id:
        return False
    ancestry = {goal.id for goal in goal_ancestry(graph, current_claim.owner_goal)}
    if isinstance(premise, Evidence):
        return premise.scope_goal in ancestry
    return premise.owner_goal in ancestry and not premise.pruned


def _premise_material(
    graph: Graph,
    premise_id: int,
    current_claim: Claim | None = None,
) -> dict[str, Any]:
    premise = get(graph, premise_id)
    if isinstance(premise, Evidence):
        return {
            "id": premise.id,
            "type": "evidence",
            "active": _premise_active_for_context(
                graph, current_claim, premise
            ),
            "scope_goal": premise.scope_goal,
            "source": premise.source,
            "statement": premise.statement,
        }
    if isinstance(premise, Claim):
        return {
            "id": premise.id,
            "type": "claim",
            "owner_goal": premise.owner_goal,
            "kind": premise.kind.value,
            "payload": _payload_material(premise.payload),
            "active": _premise_active_for_context(
                graph, current_claim, premise
            ),
            "pruned": premise.pruned,
            "verification": _check_material(premise.verification),
            "discharge": _check_material(premise.discharge),
        }
    return {"id": premise_id, "type": type(premise).__name__}


def _child_material(graph: Graph, child: Goal) -> dict[str, Any]:
    committed = None
    if child.committed_claim is not None:
        committed = _premise_material(graph, child.committed_claim)
    return {
        "id": child.id,
        "obligation": _obligation_material(child.obligation),
        "assumptions": list(child.assumptions),
        "pruned": child.pruned,
        "committed_claim": committed,
    }


def math_verifier_context(
    graph: Graph,
    claim: Claim,
    purpose: CheckPurpose,
    contract_id: str = CONTRACT_ID,
) -> VerifierContext:
    """Bind a check to trusted obligations and relevant committed state."""
    owner = get(graph, claim.owner_goal)
    if not isinstance(owner, Goal):
        raise ValueError("claim owner is not a goal")
    material = {
        "contract_id": contract_id,
        "purpose": purpose.value,
        "claim_id": claim.id,
        "claim_kind": claim.kind.value,
        "owner_goal": claim.owner_goal,
        "owner_obligation": _obligation_material(owner.obligation),
        "owner_assumptions": list(owner.assumptions),
        "owner_connective": (
            owner.connective.value
            if claim.kind is ClaimKind.AGGREGATE and owner.connective is not None
            else None
        ),
        "payload": _payload_material(claim.payload),
        "premises": [
            _premise_material(graph, premise_id, claim)
            for premise_id in claim.premises
        ],
        "children": (
            [_child_material(graph, child) for child in direct_child_goals(
                graph, owner.id
            )]
            if claim.kind is ClaimKind.AGGREGATE
            else []
        ),
    }
    encoded = json.dumps(
        material, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return VerifierContext(
        contract_id=contract_id,
        purpose=purpose,
        claim_kind=claim.kind,
        owner_goal=claim.owner_goal,
        context_digest=hashlib.sha256(encoded).hexdigest(),
    )


def _coverage_digest(
    parent: GoalObligation,
    children: tuple[GoalObligation, ...],
    contract_id: str,
    profile: DomainGrammarProfile,
) -> str:
    material = {
        "contract_id": contract_id,
        "purpose": "decomposition_coverage",
        "parent": _obligation_material(parent),
        "children": [_obligation_material(child) for child in children],
    }
    if (
        profile != LEGACY_DOMAIN_GRAMMAR_PROFILE
        or contract_id != CONTRACT_ID
    ):
        material["domain_grammar_implementation"] = profile.implementation_id
        material["domain_grammar_profile"] = profile.profile_id
    encoded = json.dumps(
        material, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_decomposition_coverage(
    parent: GoalObligation,
    children: tuple[GoalObligation, ...],
    profile: DomainGrammarProfile,
    contract_id: str,
) -> CoverageDecision:
    digest = _coverage_digest(parent, children, contract_id, profile)
    try:
        if len(children) < 2:
            raise _AdapterError("decomposition_needs_multiple_children")
        if any(child.variable != parent.variable for child in children):
            raise _AdapterError("decomposition_variable_mismatch")
        if any(child.objective is not GoalObjective.BRANCH for child in children):
            raise _AdapterError("child_objective_not_branch")
        _, _, parent_domain = _parse_obligation(parent, profile)
        child_domains = tuple(
            _parse_obligation(child, profile)[2] for child in children
        )
        sp = _sympy()
        for index, domain in enumerate(child_domains):
            for other in child_domains[index + 1 :]:
                if sp.Intersection(domain, other) != sp.S.EmptySet:
                    raise _AdapterError("decomposition_domain_overlap")
        if sp.Union(*child_domains) != parent_domain:
            raise _AdapterError("decomposition_domain_not_exhaustive")

        parent_values = _solve_values(parent, profile)
        child_values = tuple(
            _solve_values(child, profile) for child in children
        )
        if any(len(values) > 1 for values in child_values):
            raise _AdapterError("child_not_atomic")
        for child, domain, values in zip(
            children, child_domains, child_values, strict=True
        ):
            parent_branch_values = _solve_values(
                parent, profile, domain_override=domain
            )
            if values != parent_branch_values:
                raise _AdapterError("child_equation_not_equivalent")
        seen: frozenset[Any] = frozenset()
        for values in child_values:
            if seen.intersection(values):
                raise _AdapterError("decomposition_overlap")
            seen = seen.union(values)
        if seen != parent_values:
            raise _AdapterError("decomposition_not_exhaustive")
    except _AdapterError as exc:
        return CoverageDecision(False, digest, exc.reason_code)
    except Exception:
        return CoverageDecision(False, digest, "malformed_decomposition")
    return CoverageDecision(True, digest)


def validate_decomposition_coverage(
    parent: GoalObligation,
    children: tuple[GoalObligation, ...],
) -> CoverageDecision:
    """Check coverage under the exact legacy parser and contract identity."""
    return _validate_decomposition_coverage(
        parent,
        children,
        LEGACY_DOMAIN_GRAMMAR_PROFILE,
        CONTRACT_ID,
    )


def validate_decomposition_coverage_v11(
    parent: GoalObligation,
    children: tuple[GoalObligation, ...],
) -> CoverageDecision:
    """Check coverage under the corrected v1.1 parser and contract identity."""
    return _validate_decomposition_coverage(
        parent,
        children,
        CORRECTED_DOMAIN_GRAMMAR_PROFILE,
        CONTRACT_ID_V11,
    )


_REDUCER_OBJECTIVE = {
    Reducer.SUM: GoalObjective.SUM,
    Reducer.MAX: GoalObjective.MAX,
    Reducer.MIN: GoalObjective.MIN,
    Reducer.IDENTITY: GoalObjective.IDENTITY,
}
_OBJECTIVE_REDUCER = {
    objective: reducer for reducer, objective in _REDUCER_OBJECTIVE.items()
}


def _accepted_check_is_current(
    graph: Graph,
    claim: Claim,
    purpose: CheckPurpose,
    contract_id: str,
) -> bool:
    check = claim.verification if purpose is CheckPurpose.VALIDITY \
        else claim.discharge
    if check is None or not check.accepted:
        return False
    return check.context == math_verifier_context(
        graph, claim, purpose, contract_id
    )


def _check_claim_premises(
    graph: Graph, claim: Claim, contract_id: str
) -> None:
    ancestry = {goal.id for goal in goal_ancestry(graph, claim.owner_goal)}
    committed_children = (
        {
            child.committed_claim
            for child in direct_child_goals(graph, claim.owner_goal)
            if child.committed_claim is not None
        }
        if claim.kind is ClaimKind.AGGREGATE
        else set()
    )
    for premise_id in claim.premises:
        premise = get(graph, premise_id)
        if isinstance(premise, Evidence):
            if premise.source != "trusted_given" or premise.scope_goal not in ancestry:
                raise _AdapterError("premise_not_trusted")
        elif isinstance(premise, Claim):
            committed_child = premise.id in committed_children
            if (
                not committed_child
                and (premise.owner_goal not in ancestry or not is_active(
                    graph, premise.id
                ))
            ):
                raise _AdapterError("premise_not_active")
            if not _accepted_check_is_current(
                graph, premise, CheckPurpose.VALIDITY, contract_id
            ):
                raise _AdapterError("premise_not_verified")
        else:
            raise _AdapterError("premise_not_verified")


def _solution_decision(
    graph: Graph,
    claim: Claim,
    profile: DomainGrammarProfile,
) -> _KindDecision:
    if not isinstance(claim.payload, SolutionPayload):
        raise _AdapterError("payload_kind_mismatch")
    owner = get(graph, claim.owner_goal)
    if not isinstance(owner, Goal):
        raise _AdapterError("owner_not_goal")
    payload = claim.payload
    if payload.variable != owner.obligation.variable:
        raise _AdapterError("variable_mismatch")
    candidate = _parse_scalar(payload.value)
    values = _solve_values(owner.obligation, profile)
    if not _contains(values, candidate):
        raise _AdapterError("candidate_not_feasible")
    if owner.obligation.objective not in (
        GoalObjective.BRANCH,
        GoalObjective.IDENTITY,
    ):
        return _KindDecision(True, None, False, "owner_not_branch")
    if len(values) != 1:
        return _KindDecision(True, None, False, "not_unique_owner_solution")
    return _KindDecision(True, None, True, None)


def _result_decision(
    graph: Graph,
    claim: Claim,
    profile: DomainGrammarProfile,
) -> _KindDecision:
    if not isinstance(claim.payload, ResultPayload):
        raise _AdapterError("payload_kind_mismatch")
    owner = get(graph, claim.owner_goal)
    if not isinstance(owner, Goal):
        raise _AdapterError("owner_not_goal")
    payload = claim.payload
    if (
        payload.objective is GoalObjective.BRANCH
        or payload.objective is not owner.obligation.objective
    ):
        raise _AdapterError("result_objective_mismatch")
    reducer = _OBJECTIVE_REDUCER.get(payload.objective)
    if reducer is None:
        raise _AdapterError("result_objective_unsupported")
    expected = _reduce(
        tuple(_solve_values(owner.obligation, profile)), reducer
    )
    candidate = _parse_scalar(payload.value)
    if not _same_scalar(candidate, expected):
        raise _AdapterError("result_value_mismatch")
    return _KindDecision(True, None, True, None)


def _derivation_decision(claim: Claim) -> _KindDecision:
    """Check this adapter's sound but incomplete derivation subset.

    Casework-v1 admits ground rational arithmetic only. It does not claim to
    implement the core contract's broader symbolic-entailment capability.
    """
    if not isinstance(claim.payload, DerivationPayload):
        raise _AdapterError("payload_kind_mismatch")
    left = _parse_scalar(claim.payload.lhs)
    right = _parse_scalar(claim.payload.rhs)
    sp = _sympy()
    relation = claim.payload.relation
    if relation is Relation.EQ:
        accepted = _same_scalar(left, right)
    else:
        constructors = {
            Relation.LT: sp.Lt,
            Relation.LE: sp.Le,
            Relation.GT: sp.Gt,
            Relation.GE: sp.Ge,
        }
        accepted = sp.simplify(constructors[relation](left, right)) is sp.true
    if not accepted:
        raise _AdapterError("relation_rejected")
    return _KindDecision(True, None, False, "derivation_not_discharging")


def _aggregate_decision(
    graph: Graph,
    claim: Claim,
    contract_id: str,
    profile: DomainGrammarProfile,
) -> _KindDecision:
    if not isinstance(claim.payload, AggregatePayload):
        raise _AdapterError("payload_kind_mismatch")
    owner = get(graph, claim.owner_goal)
    if not isinstance(owner, Goal):
        raise _AdapterError("owner_not_goal")
    if owner.connective is not Connective.EXHAUSTIVE_CASE:
        raise _AdapterError("aggregate_requires_exhaustive_cases")
    children = direct_child_goals(graph, owner.id)
    expected = tuple(child.committed_claim for child in children)
    if not children or any(item is None for item in expected):
        raise _AdapterError("uncommitted_child")
    if claim.premises != expected:
        raise _AdapterError("committed_children_mismatch")
    payload = claim.payload
    if _REDUCER_OBJECTIVE[payload.reducer] is not owner.obligation.objective:
        raise _AdapterError("reducer_objective_mismatch")

    live_values = []
    for child, child_claim_id in zip(children, claim.premises, strict=True):
        child_claim = get(graph, child_claim_id)
        if not isinstance(child_claim, Claim) or child_claim.owner_goal != child.id:
            raise _AdapterError("committed_children_mismatch")
        if not _accepted_check_is_current(
            graph, child_claim, CheckPurpose.VALIDITY, contract_id
        ) or not _accepted_check_is_current(
            graph, child_claim, CheckPurpose.OWNER_DISCHARGE, contract_id
        ):
            raise _AdapterError("child_not_verified")
        if isinstance(child_claim.payload, SolutionPayload):
            live_values.append(_parse_scalar(child_claim.payload.value))
        elif not isinstance(child_claim.payload, RefutationPayload):
            raise _AdapterError("unsupported_child_claim")

    result = _parse_scalar(payload.result)
    reduced_children = _reduce(tuple(live_values), payload.reducer)
    if not _same_scalar(result, reduced_children):
        raise _AdapterError("aggregate_result_mismatch")

    coverage = _validate_decomposition_coverage(
        owner.obligation,
        tuple(child.obligation for child in children),
        profile,
        contract_id,
    )
    if not coverage.accepted:
        return _KindDecision(
            True, None, False, coverage.reason_code or "decomposition_rejected"
        )
    owner_values = tuple(_solve_values(owner.obligation, profile))
    owner_reduction = _reduce(owner_values, payload.reducer)
    if not _same_scalar(result, owner_reduction):
        return _KindDecision(True, None, False, "owner_reduction_mismatch")
    return _KindDecision(True, None, True, None)


def _refutation_decision(
    graph: Graph,
    claim: Claim,
    profile: DomainGrammarProfile,
) -> _KindDecision:
    if not isinstance(claim.payload, RefutationPayload):
        raise _AdapterError("payload_kind_mismatch")
    owner = get(graph, claim.owner_goal)
    if not isinstance(owner, Goal):
        raise _AdapterError("owner_not_goal")
    payload = claim.payload
    if payload.mode is not RefutationMode.UNIQUE_CANDIDATE_OUTSIDE_DOMAIN:
        raise _AdapterError("unsupported_refutation_mode")
    if payload.variable != owner.obligation.variable:
        raise _AdapterError("variable_mismatch")
    candidate = _parse_scalar(payload.candidate)
    raw_values = _solve_values(owner.obligation, profile, reals=True)
    if len(raw_values) != 1 or not _contains(raw_values, candidate):
        raise _AdapterError("candidate_not_unique_raw_solution")
    _, _, domain = _parse_obligation(owner.obligation, profile)
    membership = _sympy().simplify(domain.contains(candidate))
    if membership is not _sympy().false:
        raise _AdapterError("candidate_not_outside_domain")
    if _solve_values(owner.obligation, profile):
        raise _AdapterError("owner_not_empty")
    if owner.obligation.objective is not GoalObjective.BRANCH:
        return _KindDecision(True, None, False, "owner_not_branch")
    return _KindDecision(True, None, True, None)


@dataclass(frozen=True)
class CaseworkMathVerifier:
    """Claim-kind verifier for the casework-v1 obligation grammar."""

    contract_id: str = CONTRACT_ID
    generation_contract_id: str = CANONICALIZATION_CONTRACT_ID
    obligation_contract_id: str = CANONICALIZATION_CONTRACT_ID
    aggregate_contract_id: str = AGGREGATE_CONTRACT_ID
    domain_grammar_profile: DomainGrammarProfile = (
        LEGACY_DOMAIN_GRAMMAR_PROFILE
    )
    payload_semantics_id: ClassVar[str] = PAYLOAD_SEMANTICS_ID
    domain_grammar_implementation_id: ClassVar[str] = (
        DOMAIN_GRAMMAR_IMPLEMENTATION_ID
    )
    validity_rejection_codes: ClassVar[frozenset[str]] = (
        VALIDITY_REJECTION_CODES
    )
    discharge_rejection_codes: ClassVar[frozenset[str]] = (
        DISCHARGE_REJECTION_CODES
    )

    def __post_init__(self) -> None:
        expected_by_profile = {
            LEGACY_DOMAIN_GRAMMAR_PROFILE: (
                CONTRACT_ID,
                CANONICALIZATION_CONTRACT_ID,
                CANONICALIZATION_CONTRACT_ID,
                AGGREGATE_CONTRACT_ID,
            ),
            CORRECTED_DOMAIN_GRAMMAR_PROFILE: (
                CONTRACT_ID_V11,
                CANONICALIZATION_CONTRACT_ID_V11,
                CANONICALIZATION_CONTRACT_ID_V11,
                AGGREGATE_CONTRACT_ID_V11,
            ),
        }
        expected = expected_by_profile.get(self.domain_grammar_profile)
        if expected is None:
            raise ValueError("unsupported domain grammar profile")
        actual = (
            self.contract_id,
            self.generation_contract_id,
            self.obligation_contract_id,
            self.aggregate_contract_id,
        )
        if actual != expected:
            raise ValueError("casework semantic profile contract ids are mixed")

    @classmethod
    def corrected_v11(cls) -> CaseworkMathVerifier:
        """Construct the corrected v1.1 profile under the same adapter type."""
        return cls(
            contract_id=CONTRACT_ID_V11,
            generation_contract_id=CANONICALIZATION_CONTRACT_ID_V11,
            obligation_contract_id=CANONICALIZATION_CONTRACT_ID_V11,
            aggregate_contract_id=AGGREGATE_CONTRACT_ID_V11,
            domain_grammar_profile=CORRECTED_DOMAIN_GRAMMAR_PROFILE,
        )

    @property
    def domain_grammar_profile_id(self) -> str:
        return self.domain_grammar_profile.profile_id

    @property
    def semantic_profile_id(self) -> str:
        """Expose parser semantics for cross-collaborator environment checks."""
        return self.domain_grammar_profile_id

    def canonicalize_obligation(
        self, obligation: GoalObligation
    ) -> GoalObligation:
        """Return a canonical obligation or raise a bounded ValueError."""
        return _canonicalize_goal_obligation(
            obligation, self.domain_grammar_profile
        )

    def canonicalize_payload(
        self,
        owner: GoalObligation,
        kind: ClaimKind,
        payload: ClaimPayload,
    ) -> ClaimPayload:
        """Canonicalize a typed payload against its owner obligation."""
        canonical_owner = self.canonicalize_obligation(owner)
        if not isinstance(kind, ClaimKind):
            raise ValueError("invalid_claim_kind")
        expected_types = {
            ClaimKind.SOLUTION: SolutionPayload,
            ClaimKind.RESULT: ResultPayload,
            ClaimKind.DERIVATION: DerivationPayload,
            ClaimKind.AGGREGATE: AggregatePayload,
            ClaimKind.REFUTATION: RefutationPayload,
        }
        if not isinstance(payload, expected_types[kind]):
            raise ValueError("payload_kind_mismatch")
        canonical = canonicalize_claim_payload(payload)
        if isinstance(canonical, (SolutionPayload, RefutationPayload)):
            if canonical.variable != canonical_owner.variable:
                raise ValueError("variable_mismatch")
        if isinstance(canonical, ResultPayload):
            if canonical.objective is not canonical_owner.objective:
                raise ValueError("result_objective_mismatch")
        if isinstance(canonical, AggregatePayload):
            expected_reducer = _OBJECTIVE_REDUCER.get(
                canonical_owner.objective
            )
            if canonical.reducer is not expected_reducer:
                raise ValueError("reducer_objective_mismatch")
        return canonical

    def validate_decomposition(
        self,
        parent: GoalObligation,
        connective: Connective,
        children: tuple[GoalObligation, ...],
    ) -> tuple[GoalObligation, ...]:
        """Return canonical children only when they exactly partition parent."""
        canonical_parent = self.canonicalize_obligation(parent)
        if connective is not Connective.EXHAUSTIVE_CASE:
            raise ValueError("decomposition_requires_exhaustive_case")
        if type(children) is not tuple:
            raise ValueError("decomposition_children_not_tuple")
        canonical_children = tuple(
            self.canonicalize_obligation(child) for child in children
        )
        decision = _validate_decomposition_coverage(
            canonical_parent,
            canonical_children,
            self.domain_grammar_profile,
            self.contract_id,
        )
        if not decision.accepted:
            raise ValueError(
                decision.reason_code or "decomposition_rejected"
            )
        return canonical_children

    def build_aggregate_payload(
        self, graph: Graph, goal: Goal
    ) -> AggregatePayload:
        """Reduce exact committed child outcomes without model inference."""
        if not isinstance(graph, Graph) or not isinstance(goal, Goal):
            raise ValueError("invalid_aggregate_builder_input")
        try:
            stored_goal = get(graph, goal.id)
        except KeyError as exc:
            raise ValueError("aggregate_owner_missing") from exc
        if stored_goal != goal or goal.committed_claim is not None:
            raise ValueError("aggregate_owner_not_open")
        if (
            goal.connective is not Connective.EXHAUSTIVE_CASE
            or not is_active(graph, goal.id)
        ):
            raise ValueError("aggregate_requires_exhaustive_cases")
        reducer = _OBJECTIVE_REDUCER.get(goal.obligation.objective)
        if reducer is None:
            raise ValueError("reducer_objective_mismatch")
        children = direct_child_goals(graph, goal.id)
        if not children:
            raise ValueError("uncommitted_child")
        coverage = _validate_decomposition_coverage(
            goal.obligation,
            tuple(child.obligation for child in children),
            self.domain_grammar_profile,
            self.contract_id,
        )
        if not coverage.accepted:
            raise ValueError(
                coverage.reason_code or "decomposition_rejected"
            )

        live_values = []
        for child in children:
            if child.committed_claim is None:
                raise ValueError("uncommitted_child")
            child_claim = get(graph, child.committed_claim)
            if (
                not isinstance(child_claim, Claim)
                or child_claim.owner_goal != child.id
            ):
                raise ValueError("committed_children_mismatch")
            if not _accepted_check_is_current(
                graph, child_claim, CheckPurpose.VALIDITY, self.contract_id
            ) or not _accepted_check_is_current(
                graph,
                child_claim,
                CheckPurpose.OWNER_DISCHARGE,
                self.contract_id,
            ):
                raise ValueError("child_not_verified")
            if isinstance(child_claim.payload, SolutionPayload):
                live_values.append(_parse_scalar(child_claim.payload.value))
            elif not isinstance(child_claim.payload, RefutationPayload):
                raise ValueError("unsupported_child_claim")

        reduced = _reduce(tuple(live_values), reducer)
        return canonicalize_aggregate_payload(
            AggregatePayload(reducer, _sympy().sstr(reduced))
        )

    def verify(self, graph: Graph, claim: Claim) -> VerificationDecision:
        validity_context = math_verifier_context(
            graph, claim, CheckPurpose.VALIDITY, self.contract_id
        )
        try:
            _check_claim_premises(graph, claim, self.contract_id)
            handlers = {
                ClaimKind.SOLUTION: lambda: _solution_decision(
                    graph, claim, self.domain_grammar_profile
                ),
                ClaimKind.RESULT: lambda: _result_decision(
                    graph, claim, self.domain_grammar_profile
                ),
                ClaimKind.DERIVATION: lambda: _derivation_decision(claim),
                ClaimKind.AGGREGATE: lambda: _aggregate_decision(
                    graph,
                    claim,
                    self.contract_id,
                    self.domain_grammar_profile,
                ),
                ClaimKind.REFUTATION: lambda: _refutation_decision(
                    graph, claim, self.domain_grammar_profile
                ),
            }
            decision = handlers[claim.kind]()
        except _AdapterError as exc:
            decision = _KindDecision(
                False, exc.reason_code, None, None
            )
        except Exception:
            decision = _KindDecision(
                False, "malformed_claim", None, None
            )

        verification = SymbolicCheck(
            context=validity_context,
            accepted=decision.verification_accepted,
            observed=(
                "accepted" if decision.verification_accepted
                else decision.verification_reason or "rejected"
            ),
            reason_code=decision.verification_reason,
        )
        if not verification.accepted:
            return VerificationDecision(verification, None)
        assert decision.discharge_accepted is not None
        discharge_context = math_verifier_context(
            graph, claim, CheckPurpose.OWNER_DISCHARGE, self.contract_id
        )
        discharge = SymbolicCheck(
            context=discharge_context,
            accepted=decision.discharge_accepted,
            observed=(
                "accepted" if decision.discharge_accepted
                else decision.discharge_reason or "rejected"
            ),
            reason_code=decision.discharge_reason,
        )
        return VerificationDecision(verification, discharge)


__all__ = [
    "AGGREGATE_CONTRACT_ID",
    "AGGREGATE_CONTRACT_ID_V11",
    "CANONICALIZATION_CONTRACT_ID",
    "CANONICALIZATION_CONTRACT_ID_V11",
    "CORRECTED_DOMAIN_GRAMMAR_PROFILE",
    "CONTRACT_ID",
    "CONTRACT_ID_V11",
    "DOMAIN_GRAMMAR_IMPLEMENTATION_ID",
    "PAYLOAD_SEMANTICS_ID",
    "DISCHARGE_REJECTION_CODES",
    "CaseworkMathVerifier",
    "CoverageDecision",
    "DomainGrammarProfile",
    "LEGACY_DOMAIN_GRAMMAR_PROFILE",
    "build_casework_root_obligation",
    "build_casework_root_obligation_v11",
    "canonicalize_aggregate_payload",
    "canonicalize_claim_payload",
    "canonicalize_derivation_payload",
    "canonicalize_goal_obligation",
    "canonicalize_goal_obligation_v11",
    "canonicalize_refutation_payload",
    "canonicalize_result_payload",
    "canonicalize_solution_payload",
    "math_verifier_context",
    "VALIDITY_REJECTION_CODES",
    "validate_decomposition_coverage",
    "validate_decomposition_coverage_v11",
]
