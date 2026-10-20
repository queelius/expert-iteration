"""Typed, grounded-local generator lenses for reasoning v2."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from ..llm_client import (
    LLMClient,
    LLMResponseError,
    OpenAICompatibleLLMClient,
    SCHEMA_VALIDATION_LEGACY_COERCE_V1,
    SCHEMA_VALIDATION_STRICT_V1,
)
from .actions import (
    DecompositionDraft,
    RefutationDraft,
    ResultDraft,
    SolutionDraft,
)
from .contracts import (
    MAX_GENERATION_SEED,
    MODEL_TRANSPORT_CONTRACT_VERSION,
    ModelIdentity,
    ModelTransportContract,
    canonical_json_bytes,
    canonical_json_sha256,
    implementation_name,
)
from .model import (
    Action,
    Connective,
    GoalObjective,
    GoalObligation,
    RefutationMode,
    State,
    Verb,
)
from .runtime import BoundEnvironment
from .transition import (
    PolicyDraft,
    PolicyGenerationError,
    PolicyPayloadError,
)


LEGACY_LENS_CONTRACT_VERSION = "typed-claim-lens-v2"
LENS_CONTRACT_VERSION = "typed-claim-lens-v2.1"
SUPPORTED_LENS_CONTRACT_VERSIONS = frozenset({
    LEGACY_LENS_CONTRACT_VERSION,
    LENS_CONTRACT_VERSION,
})
LEGACY_GENERATION_SCHEMA_VERSION = "typed-policy-generation-v2"
GENERATION_SCHEMA_VERSION = "typed-policy-generation-v2.1"
SUPPORTED_GENERATION_SCHEMA_VERSIONS = frozenset({
    LEGACY_GENERATION_SCHEMA_VERSION,
    GENERATION_SCHEMA_VERSION,
})
GENERATION_SEED_POLICIES = frozenset({
    "none-v1",
    "turn-and-retry-offset-v1",
})
LLM_TRANSPORT_CONTRACT_VERSION = MODEL_TRANSPORT_CONTRACT_VERSION
MIN_DECOMPOSITION_CHILDREN = 2
MAX_DECOMPOSITION_CHILDREN = 6


def schema_validation_mode_for_generation_schema(version: str) -> str:
    modes = {
        LEGACY_GENERATION_SCHEMA_VERSION: (
            SCHEMA_VALIDATION_LEGACY_COERCE_V1
        ),
        GENERATION_SCHEMA_VERSION: SCHEMA_VALIDATION_STRICT_V1,
    }
    try:
        return modes[version]
    except (KeyError, TypeError) as exc:
        raise ValueError("generation schema version is unsupported") from exc

_SYSTEM_PROMPT = (
    "You are the frozen content generator in a verifier-anchored reasoning "
    "environment. Fill only the requested typed fields. Do not choose an "
    "action, verify a claim, assign reward, or include prose outside JSON."
)

_SCHEMAS: dict[Verb, dict[str, object]] = {
    Verb.DECOMPOSE: {
        "type": "object",
        "properties": {
            "domains": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": MIN_DECOMPOSITION_CHILDREN,
                "maxItems": MAX_DECOMPOSITION_CHILDREN,
            }
        },
        "required": ["domains"],
    },
    Verb.PROPOSE_RESULT: {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    },
    Verb.PROPOSE_SOLUTION: {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    },
    Verb.PROPOSE_REFUTATION: {
        "type": "object",
        "properties": {"candidate": {"type": "string"}},
        "required": ["candidate"],
    },
}


@dataclass(frozen=True, slots=True)
class LensRequest:
    system: str
    user: str
    schema: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.system.strip() or not self.user.strip():
            raise ValueError("lens request prompts must be nonempty")
        if not isinstance(self.schema, Mapping):
            raise ValueError("lens request schema has an invalid type")


def _goal_map(observation: Mapping[str, object]) -> dict[int, dict[str, Any]]:
    goals = observation.get("goals")
    if type(goals) is not list:
        raise PolicyPayloadError("observation goals have an invalid type")
    return {goal["id"]: goal for goal in goals}


def render_policy_lens_view(
    observation: Mapping[str, object],
    action: Action,
    *,
    lens_contract: str = LENS_CONTRACT_VERSION,
) -> dict[str, object]:
    """Project one canonical observation to its grounded-local target view."""
    if not isinstance(action, Action):
        raise TypeError("lens action has an invalid type")
    if lens_contract not in SUPPORTED_LENS_CONTRACT_VERSIONS:
        raise ValueError("lens contract is unsupported")
    if action.verb not in _SCHEMAS:
        raise ValueError("lens supports only learned generator verbs")
    goals = _goal_map(observation)
    if action.target not in goals:
        raise ValueError("lens target is not a visible Goal")
    legal_pairs = observation.get("legal_pairs")
    pair = {"verb": action.verb.value, "target": action.target}
    if type(legal_pairs) is not list or pair not in legal_pairs:
        raise ValueError("lens action is not in the canonical learned menu")

    ancestry: list[int] = []
    cursor: int | None = action.target
    while cursor is not None:
        ancestry.append(cursor)
        cursor = goals[cursor]["parent_goal"]
    ancestry.reverse()

    descendants = {action.target}
    changed = True
    while changed:
        changed = False
        for goal_id, goal in goals.items():
            if goal["parent_goal"] in descendants and goal_id not in descendants:
                descendants.add(goal_id)
                changed = True
    included_goals = set(ancestry) | descendants
    root_id = observation["root_goal_id"]
    root = goals[root_id]

    previous_outcome = observation["previous_outcome"]
    if (
        previous_outcome is not None
        and previous_outcome["action"]["target"] not in included_goals
    ):
        previous_outcome = None

    return {
        "schema_version": lens_contract,
        "environment_contract_sha256": observation[
            "environment_contract_sha256"
        ],
        "turn": observation["turn"],
        "turn_limit": observation["turn_limit"],
        "previous_outcome": previous_outcome,
        "action": pair,
        "problem_statement": root["statement"],
        "goals": [
            goal for goal_id, goal in goals.items()
            if goal_id in included_goals
        ],
        "evidence": [
            item for item in observation["evidence"]
            if item["scope_goal"] in included_goals
        ],
        "claims": [
            item for item in observation["claims"]
            if item["owner_goal"] in included_goals
        ],
    }


def _legacy_instruction(
    action: Action, target: Mapping[str, object]
) -> str:
    obligation = target["obligation"]
    if action.verb is Verb.DECOMPOSE:
        return (
            "Return {\"domains\":[\"domain 1\",\"domain 2\"]}. "
            "Use 2 to 6 pairwise-disjoint symbolic domains whose union is "
            f"exactly {obligation['domain']}. Each domain must contain at "
            "most one in-domain solution of the focus equation."
        )
    if action.verb is Verb.PROPOSE_RESULT:
        return (
            "Return {\"value\":\"symbolic value\"}. The environment "
            f"will bind it to objective {obligation['objective']}."
        )
    if action.verb is Verb.PROPOSE_SOLUTION:
        return (
            "Return {\"value\":\"symbolic value\"}. The environment "
            f"will bind it to variable {obligation['variable']}."
        )
    return (
        "Return {\"candidate\":\"symbolic value\"}. The environment "
        "will test the fixed unique-candidate-outside-domain refutation."
    )


_SCALAR_INSTRUCTION = (
    "Encode it as a nonempty JSON string of at most 128 characters. The "
    "string must evaluate to one finite real number and may contain only "
    "digits, spaces, the ASCII operators +, -, *, /, parentheses, and "
    "decimal points. Do not use **, //, a variable, a name, an equation, "
    "a set, an interval, infinity, inf, oo, Unicode symbols, or words from "
    "this instruction."
)


def _aligned_instruction(
    action: Action, target: Mapping[str, object]
) -> str:
    obligation = target["obligation"]
    variable = obligation["variable"]
    if action.verb is Verb.DECOMPOSE:
        return (
            "Return a JSON object with exactly one field named domains. Set "
            "domains to an array of 2 to 6 ASCII domain strings. In a domain "
            f"string, use the target variable {variable}. The complete "
            "domain grammar is: the exact word Reals; the target variable "
            "followed by one of <, <=, >, >= and a finite scalar; or a "
            "finite scalar followed by < or <=, the target variable, another "
            "< or <=, and a finite scalar. A finite scalar must evaluate to "
            "one finite real number and may contain only digits, spaces, +, "
            "-, *, /, parentheses, and decimal points, with no ** or //. "
            "Inside each domain string, never use interval notation, square "
            "brackets, set braces, infinity, inf, oo, or Unicode symbols. "
            "The domains must be pairwise disjoint, their union must be "
            f"exactly {obligation['domain']}, and each must contain at most "
            "one in-domain solution of the target equation. At each shared "
            "boundary, include the boundary in exactly one adjacent domain. "
            "Compute every boundary scalar from the target equation. Do not "
            "copy the lens view. Return no equation text, objective labels, "
            "statements, reasoning, or other fields."
        )
    if action.verb is Verb.PROPOSE_RESULT:
        return (
            "Return a JSON object with exactly one field named value. Solve "
            "the target Goal equation over its target domain, apply the "
            f"target objective {obligation['objective']} to the complete "
            "in-domain solution set, and put the exact result in value. "
            f"{_SCALAR_INSTRUCTION} Do not copy the lens view or return any "
            "other field."
        )
    if action.verb is Verb.PROPOSE_SOLUTION:
        return (
            "Return a JSON object with exactly one field named value. Solve "
            "the target Goal equation over its target domain and put its "
            f"unique in-domain solution for {variable} in value. "
            f"{_SCALAR_INSTRUCTION} Do not copy the lens view or return any "
            "other field."
        )
    return (
        "Return a JSON object with exactly one field named candidate. Solve "
        "the target Goal equation over all real numbers, without restricting "
        "it to the target domain. The equation must have exactly one real "
        "solution, and that candidate must lie outside the target domain. Put "
        f"that exact candidate for {variable} in candidate. "
        f"{_SCALAR_INSTRUCTION} Do not copy the lens view or return any "
        "other field."
    )


def _instruction(
    action: Action,
    target: Mapping[str, object],
    lens_contract: str,
) -> str:
    if lens_contract == LEGACY_LENS_CONTRACT_VERSION:
        return _legacy_instruction(action, target)
    if lens_contract == LENS_CONTRACT_VERSION:
        return _aligned_instruction(action, target)
    raise ValueError("lens contract is unsupported")


def build_policy_lens_request(
    observation: Mapping[str, object],
    action: Action,
    *,
    lens_contract: str = LENS_CONTRACT_VERSION,
) -> LensRequest:
    view = render_policy_lens_view(
        observation, action, lens_contract=lens_contract
    )
    target = next(
        goal for goal in view["goals"] if goal["id"] == action.target
    )
    rendered = canonical_json_bytes(view).decode("ascii")
    user = (
        f"Verb: {action.verb.name}\n"
        f"Target: {action.target}\n"
        f"Typed lens view:\n{rendered}\n\n"
        f"{_instruction(action, target, lens_contract)}\n"
        "Return only one JSON object."
    )
    return LensRequest(
        system=_SYSTEM_PROMPT,
        user=user,
        schema=json.loads(canonical_json_bytes(_SCHEMAS[action.verb])),
    )


def _scalar(value: object, field: str) -> str:
    if isinstance(value, bool) or value is None:
        raise PolicyPayloadError(f"{field} must be a string or number")
    if isinstance(value, str):
        text = value.strip()
    elif type(value) in {int, float}:
        if isinstance(value, float) and not math.isfinite(value):
            raise PolicyPayloadError(f"{field} must be finite")
        text = str(value)
    else:
        raise PolicyPayloadError(f"{field} must be a string or number")
    if not text:
        raise PolicyPayloadError(f"{field} must be nonempty")
    return text


def llm_transport_contract(client: LLMClient) -> ModelTransportContract:
    """Return a secret-free contract for the configured transport adapter."""
    if not isinstance(client, LLMClient):
        raise TypeError("client has an invalid type")
    implementation = implementation_name(client)
    if not isinstance(client, OpenAICompatibleLLMClient):
        return ModelTransportContract.local(implementation)
    if client.extra_headers:
        raise ValueError(
            "v2 hosted transport forbids uncontracted extra headers"
        )
    base_url = client.base_url.rstrip("/")
    return ModelTransportContract(
        implementation=implementation,
        base_url=base_url,
        model=client.model,
        timeout_s=float(client.timeout_s),
        parse_retry_budget=client.max_retries,
        backoff_retry_budget=client.max_backoff_retries,
        guided_json=client.guided_json,
        response_format_contract="openai-json-object-v1",
        parse_retry_contract="assistant-echo-seed-offset-v1",
        backoff_contract="retry-after-exponential-jitter-v1",
    )


def llm_transport_sha256(client: LLMClient) -> str:
    return llm_transport_contract(client).sha256


def decode_policy_payload(
    observation: Mapping[str, object],
    action: Action,
    payload: Mapping[str, object],
) -> PolicyDraft:
    """Convert a minimal wire response into the exact internal Draft union."""
    if not isinstance(action, Action) or action.verb not in _SCHEMAS:
        raise PolicyPayloadError("payload action is not a generator verb")
    if type(payload) is not dict:
        raise PolicyPayloadError("generator payload must be an object")
    goals = _goal_map(observation)
    try:
        target = goals[action.target]
    except KeyError as exc:
        raise PolicyPayloadError("generator target is not a visible Goal") from exc
    obligation = target["obligation"]

    if action.verb is Verb.DECOMPOSE:
        domains = payload.get("domains")
        if type(domains) is not list or not (
            MIN_DECOMPOSITION_CHILDREN
            <= len(domains)
            <= MAX_DECOMPOSITION_CHILDREN
        ):
            raise PolicyPayloadError(
                "domains must contain 2 to 6 symbolic strings"
            )
        if any(not isinstance(domain, str) or not domain.strip() for domain in domains):
            raise PolicyPayloadError("every child domain must be nonempty text")
        children = tuple(
            GoalObligation(
                variable=obligation["variable"],
                equation=obligation["equation"],
                domain=domain.strip(),
                objective=GoalObjective.BRANCH,
            )
            for domain in domains
        )
        return DecompositionDraft(
            connective=Connective.EXHAUSTIVE_CASE,
            child_statements=tuple(
                f"case {index}" for index in range(1, len(children) + 1)
            ),
            child_obligations=children,
        )
    if action.verb is Verb.PROPOSE_RESULT:
        return ResultDraft(
            objective=GoalObjective(obligation["objective"]),
            value=_scalar(payload.get("value"), "value"),
        )
    if action.verb is Verb.PROPOSE_SOLUTION:
        return SolutionDraft(
            variable=obligation["variable"],
            value=_scalar(payload.get("value"), "value"),
        )
    return RefutationDraft(
        mode=RefutationMode.UNIQUE_CANDIDATE_OUTSIDE_DOMAIN,
        variable=obligation["variable"],
        candidate=_scalar(payload.get("candidate"), "candidate"),
    )


@dataclass(frozen=True, slots=True)
class TypedPolicyDraftGenerator:
    """LLM-backed generator bound to one canonical environment and model."""

    environment: BoundEnvironment
    client: LLMClient
    generator_identity: ModelIdentity
    seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.environment, BoundEnvironment):
            raise TypeError("generator environment has an invalid type")
        if not isinstance(self.client, LLMClient):
            raise TypeError("generator client has an invalid type")
        if not isinstance(self.generator_identity, ModelIdentity):
            raise TypeError("generator identity has an invalid type")
        if (
            self.environment.contract.lens_contract
            not in SUPPORTED_LENS_CONTRACT_VERSIONS
        ):
            raise ValueError("environment lens contract is unsupported")
        schema_validation_mode_for_generation_schema(
            self.environment.contract.generation_schema
        )
        configured_model = getattr(self.client, "model", None)
        if (
            configured_model is not None
            and configured_model != self.generator_identity.model
        ):
            raise ValueError(
                "generator identity model differs from the configured client"
            )
        if (
            isinstance(self.client, OpenAICompatibleLLMClient)
            and self.generator_identity.revision is None
        ):
            raise ValueError(
                "hosted generator identity requires an immutable revision"
            )
        policy = self.environment.contract.generation_seed_policy
        if policy not in GENERATION_SEED_POLICIES:
            raise ValueError("environment generation seed policy is unknown")
        if policy == "none-v1" and self.seed is not None:
            raise ValueError("none-v1 generation cannot carry a seed")
        if policy == "turn-and-retry-offset-v1" and (
            type(self.seed) is not int
            or not 0 <= self.seed <= MAX_GENERATION_SEED
        ):
            raise ValueError(
                "turn-and-retry-offset-v1 generation requires an int31 seed"
            )

    @property
    def environment_contract_sha256(self) -> str:
        return self.environment.sha256

    @property
    def generator_implementation(self) -> str:
        return implementation_name(self)

    @property
    def generator_transport_sha256(self) -> str:
        return llm_transport_sha256(self.client)

    @property
    def generator_transport_contract(self) -> ModelTransportContract:
        return llm_transport_contract(self.client)

    @property
    def generation_seed(self) -> int | None:
        return self.seed

    @property
    def generation_retry_budget(self) -> int:
        value = getattr(self.client, "max_retries", 0)
        if type(value) is not int or value < 0:
            raise ValueError("client max_retries has an invalid type")
        return value

    @property
    def lens_contract(self) -> str:
        return self.environment.contract.lens_contract

    @property
    def generation_schema(self) -> str:
        return self.environment.contract.generation_schema

    @property
    def generation_temperature(self) -> float:
        return self.environment.contract.generation_temperature

    @property
    def generation_max_tokens(self) -> int:
        return self.environment.contract.generation_max_tokens

    @property
    def generation_seed_policy(self) -> str:
        return self.environment.contract.generation_seed_policy

    def generate(self, state: State, action: Action) -> PolicyDraft:
        observation = self.environment.observe(state)
        request = build_policy_lens_request(
            observation,
            action,
            lens_contract=self.environment.contract.lens_contract,
        )
        seed = (
            None
            if self.generation_seed_policy == "none-v1"
            else self.seed + state.turn
        )
        try:
            payload = self.client.complete_json(
                request.system,
                request.user,
                request.schema,
                temperature=self.generation_temperature,
                max_tokens=self.generation_max_tokens,
                seed=seed,
                schema_validation_mode=(
                    schema_validation_mode_for_generation_schema(
                        self.generation_schema
                    )
                ),
            )
        except LLMResponseError as exc:
            if exc.raw_response is not None:
                raise PolicyPayloadError(str(exc)) from exc
            raise PolicyGenerationError(str(exc)) from exc
        return decode_policy_payload(observation, action, payload)


__all__ = [
    "GENERATION_SCHEMA_VERSION",
    "GENERATION_SEED_POLICIES",
    "LEGACY_GENERATION_SCHEMA_VERSION",
    "LEGACY_LENS_CONTRACT_VERSION",
    "LENS_CONTRACT_VERSION",
    "LLM_TRANSPORT_CONTRACT_VERSION",
    "LensRequest",
    "MAX_DECOMPOSITION_CHILDREN",
    "MIN_DECOMPOSITION_CHILDREN",
    "SUPPORTED_GENERATION_SCHEMA_VERSIONS",
    "SUPPORTED_LENS_CONTRACT_VERSIONS",
    "TypedPolicyDraftGenerator",
    "build_policy_lens_request",
    "decode_policy_payload",
    "llm_transport_contract",
    "llm_transport_sha256",
    "render_policy_lens_view",
    "schema_validation_mode_for_generation_schema",
]
