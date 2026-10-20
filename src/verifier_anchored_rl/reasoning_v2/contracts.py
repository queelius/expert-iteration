"""Immutable semantic contracts and canonical JSON for reasoning v2."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlsplit

from .closure import (
    CLOSURE_EVENTS_VERSION,
    CLOSURE_PRIORITY,
    CLOSURE_SEMANTICS_VERSION,
    CLOSURE_TURN_COST,
)
from .menu import (
    ACTION_SEMANTICS_VERSION,
    ACTION_TARGET_KINDS,
    POLICY_VERB_ORDER,
    VERB_ORDER,
)
from .observation import CONTROLLER_OBSERVATION_VERSION
from .transition import REWARD_CONTRACT_VERSION, TRANSITION_SEMANTICS_VERSION


CANONICAL_JSON_VERSION = "canonical-json-v2"
ENVIRONMENT_CONTRACT_VERSION = "reasoning-environment-contract-v2"
COLLECTION_CONTRACT_VERSION = "controller-collection-contract-v2"
PAYLOAD_SEMANTICS_VERSION = "flat-claim-payload-v2"
MAX_GENERATION_SEED = 2**31 - 1
MODEL_TRANSPORT_CONTRACT_VERSION = "model-transport-contract-v2"
GENERATION_SEED_POLICY_IDS = frozenset({
    "none-v1",
    "turn-and-retry-offset-v1",
})
CONTROLLER_SEED_POLICY_IDS = GENERATION_SEED_POLICY_IDS


def _validate_json_value(value: object, path: str = "$") -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value) or (
            value == 0.0 and math.copysign(1.0, value) < 0.0
        ):
            raise ValueError(f"{path} contains an invalid JSON number")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string object key")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise ValueError(f"{path} contains non-JSON type {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON value with the one v2 canonical encoding."""
    _validate_json_value(value)
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not canonical JSON: {exc}") from exc
    return encoded.encode("ascii")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def implementation_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def verifier_reason_codes_record(verifier: object) -> dict[str, object]:
    contract_id = getattr(verifier, "contract_id", None)
    validity = getattr(verifier, "validity_rejection_codes", None)
    discharge = getattr(verifier, "discharge_rejection_codes", None)
    if not isinstance(contract_id, str) or not contract_id.strip():
        raise TypeError("verifier contract_id must be nonempty text")
    for name, codes in (("validity", validity), ("discharge", discharge)):
        if type(codes) is not frozenset or any(
            not isinstance(code, str) or not code.strip() for code in codes
        ):
            raise TypeError(f"verifier {name} rejection codes are invalid")
    return {
        "schema_version": "verifier-reason-codes-v2",
        "verifier_contract": contract_id,
        "validity_rejection_codes": sorted(validity),
        "discharge_rejection_codes": sorted(discharge),
    }


def verifier_reason_codes_sha256(verifier: object) -> str:
    return canonical_json_sha256(verifier_reason_codes_record(verifier))


_ENVIRONMENT_KEYS = frozenset({
    "schema_version",
    "canonical_serializer",
    "observation_schema",
    "action_semantics",
    "algebra_verb_order",
    "policy_verb_order",
    "target_kinds",
    "transition_semantics",
    "closure_semantics",
    "closure_events_schema",
    "closure_priority",
    "closure_turn_cost",
    "payload_semantics",
    "obligation_contract",
    "generation_contract",
    "generation_contract_implementation",
    "verifier_contract",
    "verifier_implementation",
    "verifier_reason_codes_sha256",
    "aggregate_contract",
    "aggregate_implementation",
    "reward_contract",
    "lens_contract",
    "generation_schema",
    "generation_temperature",
    "generation_max_tokens",
    "generation_seed_policy",
    "turn_limit",
})


@dataclass(frozen=True, slots=True)
class EnvironmentContract:
    """Exact episode-static semantics that determine v2 transitions."""

    schema_version: str
    canonical_serializer: str
    observation_schema: str
    action_semantics: str
    algebra_verb_order: tuple[str, ...]
    policy_verb_order: tuple[str, ...]
    target_kinds: tuple[tuple[str, tuple[str, ...]], ...]
    transition_semantics: str
    closure_semantics: str
    closure_events_schema: str
    closure_priority: tuple[str, ...]
    closure_turn_cost: int
    payload_semantics: str
    obligation_contract: str
    generation_contract: str
    generation_contract_implementation: str
    verifier_contract: str
    verifier_implementation: str
    verifier_reason_codes_sha256: str
    aggregate_contract: str
    aggregate_implementation: str
    reward_contract: str
    lens_contract: str
    generation_schema: str
    generation_temperature: float
    generation_max_tokens: int
    generation_seed_policy: str
    turn_limit: int

    def __post_init__(self) -> None:
        fixed = {
            "schema_version": ENVIRONMENT_CONTRACT_VERSION,
            "canonical_serializer": CANONICAL_JSON_VERSION,
            "observation_schema": CONTROLLER_OBSERVATION_VERSION,
            "action_semantics": ACTION_SEMANTICS_VERSION,
            "algebra_verb_order": tuple(verb.value for verb in VERB_ORDER),
            "policy_verb_order": tuple(
                verb.value for verb in POLICY_VERB_ORDER
            ),
            "target_kinds": tuple(
                (verb.value, tuple(ACTION_TARGET_KINDS[verb.value]))
                for verb in VERB_ORDER
            ),
            "transition_semantics": TRANSITION_SEMANTICS_VERSION,
            "closure_semantics": CLOSURE_SEMANTICS_VERSION,
            "closure_events_schema": CLOSURE_EVENTS_VERSION,
            "closure_priority": tuple(
                verb.value for verb in CLOSURE_PRIORITY
            ),
            "closure_turn_cost": CLOSURE_TURN_COST,
            "payload_semantics": PAYLOAD_SEMANTICS_VERSION,
            "reward_contract": REWARD_CONTRACT_VERSION,
        }
        for field, expected in fixed.items():
            if getattr(self, field) != expected:
                raise ValueError(f"environment {field} differs from runtime")
        for field in (
            "obligation_contract",
            "generation_contract",
            "generation_contract_implementation",
            "verifier_contract",
            "verifier_implementation",
            "aggregate_contract",
            "aggregate_implementation",
            "lens_contract",
            "generation_schema",
            "generation_seed_policy",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"environment {field} must be nonempty text")
        if self.generation_seed_policy not in GENERATION_SEED_POLICY_IDS:
            raise ValueError("environment generation seed policy is unknown")
        digest = self.verifier_reason_codes_sha256
        if not isinstance(digest, str) or len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise ValueError("verifier reason-code digest is invalid")
        temperature = self.generation_temperature
        if type(temperature) is not float or not math.isfinite(temperature):
            raise ValueError("generation temperature must be a finite float")
        if temperature < 0.0 or (
            temperature == 0.0 and math.copysign(1.0, temperature) < 0.0
        ):
            raise ValueError("generation temperature must be nonnegative")
        if type(self.turn_limit) is not int or self.turn_limit <= 0:
            raise ValueError("turn_limit must be a positive integer")
        if (
            type(self.generation_max_tokens) is not int
            or self.generation_max_tokens <= 0
        ):
            raise ValueError(
                "generation_max_tokens must be a positive integer"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "canonical_serializer": self.canonical_serializer,
            "observation_schema": self.observation_schema,
            "action_semantics": self.action_semantics,
            "algebra_verb_order": list(self.algebra_verb_order),
            "policy_verb_order": list(self.policy_verb_order),
            "target_kinds": {
                verb: list(kinds) for verb, kinds in self.target_kinds
            },
            "transition_semantics": self.transition_semantics,
            "closure_semantics": self.closure_semantics,
            "closure_events_schema": self.closure_events_schema,
            "closure_priority": list(self.closure_priority),
            "closure_turn_cost": self.closure_turn_cost,
            "payload_semantics": self.payload_semantics,
            "obligation_contract": self.obligation_contract,
            "generation_contract": self.generation_contract,
            "generation_contract_implementation": (
                self.generation_contract_implementation
            ),
            "verifier_contract": self.verifier_contract,
            "verifier_implementation": self.verifier_implementation,
            "verifier_reason_codes_sha256": (
                self.verifier_reason_codes_sha256
            ),
            "aggregate_contract": self.aggregate_contract,
            "aggregate_implementation": self.aggregate_implementation,
            "reward_contract": self.reward_contract,
            "lens_contract": self.lens_contract,
            "generation_schema": self.generation_schema,
            "generation_temperature": self.generation_temperature,
            "generation_max_tokens": self.generation_max_tokens,
            "generation_seed_policy": self.generation_seed_policy,
            "turn_limit": self.turn_limit,
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.to_record())

    @classmethod
    def from_record(cls, value: object) -> EnvironmentContract:
        if type(value) is not dict or set(value) != _ENVIRONMENT_KEYS:
            raise ValueError("environment contract has invalid keys")
        record: Mapping[str, Any] = value
        target_kinds = record["target_kinds"]
        if type(target_kinds) is not dict:
            raise ValueError("environment target_kinds must be an object")
        if set(target_kinds) != {verb.value for verb in VERB_ORDER}:
            raise ValueError("environment target_kinds has invalid keys")
        try:
            return cls(
                schema_version=record["schema_version"],
                canonical_serializer=record["canonical_serializer"],
                observation_schema=record["observation_schema"],
                action_semantics=record["action_semantics"],
                algebra_verb_order=tuple(record["algebra_verb_order"]),
                policy_verb_order=tuple(record["policy_verb_order"]),
                target_kinds=tuple(
                    (verb.value, tuple(target_kinds[verb.value]))
                    for verb in VERB_ORDER
                ),
                transition_semantics=record["transition_semantics"],
                closure_semantics=record["closure_semantics"],
                closure_events_schema=record["closure_events_schema"],
                closure_priority=tuple(record["closure_priority"]),
                closure_turn_cost=record["closure_turn_cost"],
                payload_semantics=record["payload_semantics"],
                obligation_contract=record["obligation_contract"],
                generation_contract=record["generation_contract"],
                generation_contract_implementation=record[
                    "generation_contract_implementation"
                ],
                verifier_contract=record["verifier_contract"],
                verifier_implementation=record["verifier_implementation"],
                verifier_reason_codes_sha256=record[
                    "verifier_reason_codes_sha256"
                ],
                aggregate_contract=record["aggregate_contract"],
                aggregate_implementation=record["aggregate_implementation"],
                reward_contract=record["reward_contract"],
                lens_contract=record["lens_contract"],
                generation_schema=record["generation_schema"],
                generation_temperature=record["generation_temperature"],
                generation_max_tokens=record["generation_max_tokens"],
                generation_seed_policy=record["generation_seed_policy"],
                turn_limit=record["turn_limit"],
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("environment contract has invalid values") from exc


def validate_semantic_profile_consistency(
    generation_contract: object,
    verifier: object,
    aggregate_builder: object,
) -> None:
    """Reject only partial or mixed profiles across collaborators.

    Environment identity remains bound by versioned collaborator contract IDs.
    This consistency guard does not add profile data to that identity.
    """
    collaborators = (
        generation_contract,
        verifier,
        aggregate_builder,
    )
    profile_ids = tuple(
        getattr(collaborator, "semantic_profile_id", None)
        for collaborator in collaborators
    )
    for profile_id in profile_ids:
        if profile_id is not None and (
            not isinstance(profile_id, str) or not profile_id.strip()
        ):
            raise ValueError("semantic profile id must be null or nonempty")
    declared = tuple(
        profile_id for profile_id in profile_ids if profile_id is not None
    )
    if not declared:
        return
    if len(declared) != len(collaborators):
        raise ValueError(
            "semantic profile must be declared by every environment collaborator"
        )
    if len(set(declared)) != 1:
        raise ValueError("environment collaborators use mixed semantic profiles")


def build_environment_contract(
    *,
    generation_contract: object,
    verifier: object,
    aggregate_builder: object,
    lens_contract: str,
    generation_schema: str,
    generation_temperature: float,
    generation_max_tokens: int,
    generation_seed_policy: str,
    turn_limit: int,
) -> EnvironmentContract:
    validate_semantic_profile_consistency(
        generation_contract, verifier, aggregate_builder
    )
    return EnvironmentContract(
        schema_version=ENVIRONMENT_CONTRACT_VERSION,
        canonical_serializer=CANONICAL_JSON_VERSION,
        observation_schema=CONTROLLER_OBSERVATION_VERSION,
        action_semantics=ACTION_SEMANTICS_VERSION,
        algebra_verb_order=tuple(verb.value for verb in VERB_ORDER),
        policy_verb_order=tuple(verb.value for verb in POLICY_VERB_ORDER),
        target_kinds=tuple(
            (verb.value, tuple(ACTION_TARGET_KINDS[verb.value]))
            for verb in VERB_ORDER
        ),
        transition_semantics=TRANSITION_SEMANTICS_VERSION,
        closure_semantics=CLOSURE_SEMANTICS_VERSION,
        closure_events_schema=CLOSURE_EVENTS_VERSION,
        closure_priority=tuple(verb.value for verb in CLOSURE_PRIORITY),
        closure_turn_cost=CLOSURE_TURN_COST,
        payload_semantics=PAYLOAD_SEMANTICS_VERSION,
        obligation_contract=getattr(
            generation_contract, "obligation_contract_id", None
        ),
        generation_contract=getattr(
            generation_contract, "generation_contract_id", None
        ),
        generation_contract_implementation=implementation_name(
            generation_contract
        ),
        verifier_contract=getattr(verifier, "contract_id", None),
        verifier_implementation=implementation_name(verifier),
        verifier_reason_codes_sha256=verifier_reason_codes_sha256(verifier),
        aggregate_contract=getattr(
            aggregate_builder, "aggregate_contract_id", None
        ),
        aggregate_implementation=implementation_name(aggregate_builder),
        reward_contract=REWARD_CONTRACT_VERSION,
        lens_contract=lens_contract,
        generation_schema=generation_schema,
        generation_temperature=float(generation_temperature),
        generation_max_tokens=generation_max_tokens,
        generation_seed_policy=generation_seed_policy,
        turn_limit=turn_limit,
    )


_IDENTITY_KEYS = frozenset({"provider", "model", "revision"})


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    provider: str
    model: str
    revision: str | None

    def __post_init__(self) -> None:
        for field in ("provider", "model"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"model identity {field} must be nonempty")
        if self.revision is not None and (
            not isinstance(self.revision, str) or not self.revision.strip()
        ):
            raise ValueError("model identity revision must be null or nonempty")

    def to_record(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "revision": self.revision,
        }

    @classmethod
    def from_record(cls, value: object) -> ModelIdentity:
        if type(value) is not dict or set(value) != _IDENTITY_KEYS:
            raise ValueError("model identity has invalid keys")
        return cls(value["provider"], value["model"], value["revision"])


_TRANSPORT_KEYS = frozenset({
    "schema_version",
    "implementation",
    "base_url",
    "model",
    "timeout_s",
    "parse_retry_budget",
    "backoff_retry_budget",
    "guided_json",
    "response_format_contract",
    "parse_retry_contract",
    "backoff_contract",
    "semantic_header_names",
})


@dataclass(frozen=True, slots=True)
class ModelTransportContract:
    """Exact secret-free configuration of one model transport adapter."""

    implementation: str
    base_url: str | None
    model: str | None
    timeout_s: float | None
    parse_retry_budget: int
    backoff_retry_budget: int
    guided_json: bool
    response_format_contract: str
    parse_retry_contract: str
    backoff_contract: str
    semantic_header_names: tuple[str, ...] = ()
    schema_version: str = MODEL_TRANSPORT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_TRANSPORT_CONTRACT_VERSION:
            raise ValueError("model transport schema differs from runtime")
        for field in (
            "implementation",
            "response_format_contract",
            "parse_retry_contract",
            "backoff_contract",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"model transport {field} must be nonempty")
        if self.base_url is None:
            if self.timeout_s is not None:
                raise ValueError(
                    "local model transport cannot carry an HTTP timeout"
                )
        else:
            if not isinstance(self.base_url, str) or not self.base_url.strip():
                raise ValueError("model transport base_url is invalid")
            parsed = urlsplit(self.base_url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
                or self.base_url.endswith("/")
            ):
                raise ValueError(
                    "model transport base_url must be normalized and secret-free"
                )
            if (
                type(self.timeout_s) is not float
                or not math.isfinite(self.timeout_s)
                or self.timeout_s <= 0.0
            ):
                raise ValueError("model transport timeout_s is invalid")
        if self.model is not None and (
            not isinstance(self.model, str) or not self.model.strip()
        ):
            raise ValueError("model transport model is invalid")
        for field in ("parse_retry_budget", "backoff_retry_budget"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"model transport {field} must be nonnegative"
                )
        if type(self.guided_json) is not bool:
            raise ValueError("model transport guided_json must be boolean")
        if type(self.semantic_header_names) is not tuple or any(
            not isinstance(name, str) or not name.strip()
            for name in self.semantic_header_names
        ):
            raise ValueError("model transport header names are invalid")
        if self.semantic_header_names != tuple(
            sorted(set(self.semantic_header_names))
        ):
            raise ValueError(
                "model transport header names must be sorted and unique"
            )
        if self.semantic_header_names:
            raise ValueError(
                "semantic transport headers are not supported by v2"
            )

    @classmethod
    def local(cls, implementation: str) -> ModelTransportContract:
        return cls(
            implementation=implementation,
            base_url=None,
            model=None,
            timeout_s=None,
            parse_retry_budget=0,
            backoff_retry_budget=0,
            guided_json=False,
            response_format_contract="direct-json-object-v1",
            parse_retry_contract="none-v1",
            backoff_contract="none-v1",
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "implementation": self.implementation,
            "base_url": self.base_url,
            "model": self.model,
            "timeout_s": self.timeout_s,
            "parse_retry_budget": self.parse_retry_budget,
            "backoff_retry_budget": self.backoff_retry_budget,
            "guided_json": self.guided_json,
            "response_format_contract": self.response_format_contract,
            "parse_retry_contract": self.parse_retry_contract,
            "backoff_contract": self.backoff_contract,
            "semantic_header_names": list(self.semantic_header_names),
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.to_record())

    @classmethod
    def from_record(cls, value: object) -> ModelTransportContract:
        if type(value) is not dict or set(value) != _TRANSPORT_KEYS:
            raise ValueError("model transport contract has invalid keys")
        if type(value["semantic_header_names"]) is not list:
            raise ValueError(
                "model transport semantic_header_names must be an array"
            )
        try:
            return cls(
                schema_version=value["schema_version"],
                implementation=value["implementation"],
                base_url=value["base_url"],
                model=value["model"],
                timeout_s=value["timeout_s"],
                parse_retry_budget=value["parse_retry_budget"],
                backoff_retry_budget=value["backoff_retry_budget"],
                guided_json=value["guided_json"],
                response_format_contract=value[
                    "response_format_contract"
                ],
                parse_retry_contract=value["parse_retry_contract"],
                backoff_contract=value["backoff_contract"],
                semantic_header_names=tuple(value["semantic_header_names"]),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "model transport contract has invalid values"
            ) from exc


_COLLECTION_KEYS = frozenset({
    "schema_version",
    "generator",
    "controller",
    "controller_implementation",
    "controller_transport",
    "controller_seed",
    "controller_temperature",
    "controller_max_tokens",
    "controller_seed_policy",
    "controller_output_contract",
    "behavior_source",
    "controller_fallback_policy",
    "generator_implementation",
    "generator_transport",
    "generation_seed",
    "policy_input_contract",
    "problem_source_contract",
    "problem_source_sha256",
    "problem_adapter_implementation",
})


@dataclass(frozen=True, slots=True)
class CollectionContract:
    generator: ModelIdentity
    controller: ModelIdentity
    controller_implementation: str
    controller_transport: ModelTransportContract
    controller_seed: int | None
    behavior_source: str
    controller_fallback_policy: str
    generator_implementation: str
    generator_transport: ModelTransportContract
    generation_seed: int | None
    policy_input_contract: str
    problem_source_contract: str
    problem_source_sha256: str
    problem_adapter_implementation: str
    controller_temperature: float = 0.0
    controller_max_tokens: int = 64
    controller_seed_policy: str = "none-v1"
    controller_output_contract: str = "controller-choice-wire-v2.1"
    schema_version: str = COLLECTION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != COLLECTION_CONTRACT_VERSION:
            raise ValueError("collection contract schema differs from runtime")
        if not isinstance(self.generator, ModelIdentity) or not isinstance(
            self.controller, ModelIdentity
        ):
            raise ValueError("collection model identities are invalid")
        if (
            not isinstance(self.controller_implementation, str)
            or not self.controller_implementation.strip()
        ):
            raise ValueError("controller_implementation must be nonempty text")
        if not isinstance(
            self.controller_transport, ModelTransportContract
        ):
            raise ValueError("controller transport contract is invalid")
        if (
            self.controller_transport.model is not None
            and self.controller_transport.model != self.controller.model
        ):
            raise ValueError(
                "controller identity model differs from its transport"
            )
        if (
            self.controller_transport.base_url is not None
            and self.controller.revision is None
        ):
            raise ValueError(
                "hosted controller identity requires an immutable revision"
            )
        temperature = self.controller_temperature
        if type(temperature) is not float or not math.isfinite(temperature):
            raise ValueError("controller_temperature must be a finite float")
        if temperature < 0.0 or (
            temperature == 0.0 and math.copysign(1.0, temperature) < 0.0
        ):
            raise ValueError("controller_temperature must be nonnegative")
        if (
            type(self.controller_max_tokens) is not int
            or self.controller_max_tokens <= 0
        ):
            raise ValueError("controller_max_tokens must be positive")
        if self.controller_seed_policy not in CONTROLLER_SEED_POLICY_IDS:
            raise ValueError("controller_seed_policy is unknown")
        if (
            not isinstance(self.controller_output_contract, str)
            or not self.controller_output_contract.strip()
        ):
            raise ValueError(
                "controller_output_contract must be nonempty text"
            )
        if self.controller_seed is not None and (
            type(self.controller_seed) is not int
            or not 0 <= self.controller_seed <= MAX_GENERATION_SEED
        ):
            raise ValueError(
                "controller_seed is outside the signed int31 range"
            )
        if self.controller_seed_policy == "none-v1":
            if self.controller_seed is not None:
                raise ValueError("none-v1 controller cannot carry a seed")
        elif type(self.controller_seed) is not int:
            raise ValueError("offset controller seed policy needs a seed")
        if self.behavior_source not in {"teacher", "learned_policy"}:
            raise ValueError("collection behavior_source is invalid")
        expected_fallback = (
            "first-legal-pair-v1"
            if self.behavior_source == "teacher"
            else "forbidden-v1"
        )
        if self.controller_fallback_policy != expected_fallback:
            raise ValueError(
                "controller fallback policy differs from behavior source"
            )
        if (
            not isinstance(self.generator_implementation, str)
            or not self.generator_implementation.strip()
        ):
            raise ValueError("generator_implementation must be nonempty text")
        if not isinstance(
            self.generator_transport, ModelTransportContract
        ):
            raise ValueError("generator transport contract is invalid")
        if (
            self.generator_transport.model is not None
            and self.generator_transport.model != self.generator.model
        ):
            raise ValueError(
                "generator identity model differs from its transport"
            )
        if (
            self.generator_transport.base_url is not None
            and self.generator.revision is None
        ):
            raise ValueError(
                "hosted generator identity requires an immutable revision"
            )
        if self.generation_seed is not None and (
            type(self.generation_seed) is not int
            or not 0 <= self.generation_seed <= MAX_GENERATION_SEED
        ):
            raise ValueError(
                "generation_seed is outside the signed int31 range"
            )
        if not isinstance(self.policy_input_contract, str) or not (
            self.policy_input_contract.strip()
        ):
            raise ValueError("policy_input_contract must be nonempty text")
        for field in (
            "problem_source_contract",
            "problem_adapter_implementation",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be nonempty text")
        source_digest = self.problem_source_sha256
        if not isinstance(source_digest, str) or len(
            source_digest
        ) != 64 or any(
            char not in "0123456789abcdef" for char in source_digest
        ):
            raise ValueError("problem source digest is invalid")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generator": self.generator.to_record(),
            "controller": self.controller.to_record(),
            "controller_implementation": self.controller_implementation,
            "controller_transport": self.controller_transport.to_record(),
            "controller_seed": self.controller_seed,
            "controller_temperature": self.controller_temperature,
            "controller_max_tokens": self.controller_max_tokens,
            "controller_seed_policy": self.controller_seed_policy,
            "controller_output_contract": self.controller_output_contract,
            "behavior_source": self.behavior_source,
            "controller_fallback_policy": self.controller_fallback_policy,
            "generator_implementation": self.generator_implementation,
            "generator_transport": self.generator_transport.to_record(),
            "generation_seed": self.generation_seed,
            "policy_input_contract": self.policy_input_contract,
            "problem_source_contract": self.problem_source_contract,
            "problem_source_sha256": self.problem_source_sha256,
            "problem_adapter_implementation": (
                self.problem_adapter_implementation
            ),
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.to_record())

    @property
    def controller_transport_sha256(self) -> str:
        return self.controller_transport.sha256

    @property
    def generator_transport_sha256(self) -> str:
        return self.generator_transport.sha256

    @classmethod
    def from_record(cls, value: object) -> CollectionContract:
        if type(value) is not dict or set(value) != _COLLECTION_KEYS:
            raise ValueError("collection contract has invalid keys")
        return cls(
            schema_version=value["schema_version"],
            generator=ModelIdentity.from_record(value["generator"]),
            controller=ModelIdentity.from_record(value["controller"]),
            controller_implementation=value["controller_implementation"],
            controller_transport=ModelTransportContract.from_record(
                value["controller_transport"]
            ),
            controller_seed=value["controller_seed"],
            controller_temperature=value["controller_temperature"],
            controller_max_tokens=value["controller_max_tokens"],
            controller_seed_policy=value["controller_seed_policy"],
            controller_output_contract=value["controller_output_contract"],
            behavior_source=value["behavior_source"],
            controller_fallback_policy=value[
                "controller_fallback_policy"
            ],
            generator_implementation=value["generator_implementation"],
            generator_transport=ModelTransportContract.from_record(
                value["generator_transport"]
            ),
            generation_seed=value["generation_seed"],
            policy_input_contract=value["policy_input_contract"],
            problem_source_contract=value["problem_source_contract"],
            problem_source_sha256=value["problem_source_sha256"],
            problem_adapter_implementation=value[
                "problem_adapter_implementation"
            ],
        )


def validate_collection_binding(
    environment: EnvironmentContract,
    collection: CollectionContract,
) -> None:
    """Validate cross-contract settings that determine live collection."""
    if not isinstance(environment, EnvironmentContract):
        raise TypeError("environment binding has an invalid type")
    if not isinstance(collection, CollectionContract):
        raise TypeError("collection binding has an invalid type")
    if collection.policy_input_contract == environment.observation_schema:
        raise ValueError(
            "policy_input_contract cannot masquerade as observation schema"
        )
    if environment.generation_seed_policy == "none-v1":
        if collection.generation_seed is not None:
            raise ValueError("none-v1 collection cannot carry a seed")
    else:
        seed = collection.generation_seed
        if type(seed) is not int:
            raise ValueError("offset seed policy needs a collection seed")
        maximum = (
            seed
            + environment.turn_limit
            - 1
            + collection.generator_transport.parse_retry_budget
        )
        if maximum > MAX_GENERATION_SEED:
            raise ValueError("effective generation seed exceeds int31")
    if collection.controller_seed_policy == "none-v1":
        if collection.controller_seed is not None:
            raise ValueError("none-v1 controller cannot carry a seed")
    else:
        controller_seed = collection.controller_seed
        if type(controller_seed) is not int:
            raise ValueError("offset controller seed policy needs a seed")
        controller_maximum = (
            controller_seed
            + environment.turn_limit
            - 1
            + collection.controller_transport.parse_retry_budget
        )
        if controller_maximum > MAX_GENERATION_SEED:
            raise ValueError("effective controller seed exceeds int31")


__all__ = [
    "CANONICAL_JSON_VERSION",
    "CONTROLLER_SEED_POLICY_IDS",
    "COLLECTION_CONTRACT_VERSION",
    "CollectionContract",
    "ENVIRONMENT_CONTRACT_VERSION",
    "EnvironmentContract",
    "GENERATION_SEED_POLICY_IDS",
    "MAX_GENERATION_SEED",
    "MODEL_TRANSPORT_CONTRACT_VERSION",
    "ModelIdentity",
    "ModelTransportContract",
    "PAYLOAD_SEMANTICS_VERSION",
    "build_environment_contract",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "implementation_name",
    "validate_collection_binding",
    "validate_semantic_profile_consistency",
    "verifier_reason_codes_record",
    "verifier_reason_codes_sha256",
]
