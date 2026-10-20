"""Contract-bound LLM teacher controller for reasoning v2."""

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
from .contracts import (
    CONTROLLER_SEED_POLICY_IDS,
    MAX_GENERATION_SEED,
    ModelIdentity,
    ModelTransportContract,
    canonical_json_bytes,
    implementation_name,
)
from .lenses import llm_transport_contract, llm_transport_sha256
from .menu import POLICY_VERB_ORDER
from .model import Action, Verb
from .observation import validate_observation_with_contract
from .provenance import BehaviorSource, ControllerChoice
from .runtime import BoundEnvironment


POLICY_INPUT_CONTRACT_VERSION = "typed-controller-presentation-v2"
LEGACY_CONTROLLER_OUTPUT_CONTRACT_VERSION = "controller-choice-wire-v2"
CONTROLLER_OUTPUT_CONTRACT_VERSION = "controller-choice-wire-v2.1"
SUPPORTED_CONTROLLER_OUTPUT_CONTRACT_VERSIONS = frozenset({
    LEGACY_CONTROLLER_OUTPUT_CONTRACT_VERSION,
    CONTROLLER_OUTPUT_CONTRACT_VERSION,
})


def schema_validation_mode_for_controller_output(version: str) -> str:
    modes = {
        LEGACY_CONTROLLER_OUTPUT_CONTRACT_VERSION: (
            SCHEMA_VALIDATION_LEGACY_COERCE_V1
        ),
        CONTROLLER_OUTPUT_CONTRACT_VERSION: SCHEMA_VALIDATION_STRICT_V1,
    }
    try:
        return modes[version]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "controller output contract is unsupported"
        ) from exc

_SYSTEM_PROMPT = (
    "You are the teacher controller in a verifier-anchored reasoning "
    "environment. Choose exactly one listed legal action. DECOMPOSE "
    "partitions a goal into exhaustive cases. PROPOSE_RESULT supplies the "
    "goal objective value. PROPOSE_SOLUTION supplies a solution in the "
    "goal domain. PROPOSE_REFUTATION supplies the fixed outside-domain "
    "refutation candidate. Do not generate claim content, verify work, "
    "assign reward, or invent an unlisted pair. Return only JSON."
)

_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "verb": {
            "type": "string",
            "enum": [verb.value for verb in POLICY_VERB_ORDER],
        },
        "target": {"type": "integer"},
    },
    "required": ["verb", "target"],
    "additionalProperties": False,
}


class ControllerPayloadError(ValueError):
    """Raised when a model reply is not one exact typed controller choice."""


@dataclass(frozen=True, slots=True)
class ControllerRequest:
    system: str
    user: str
    schema: Mapping[str, object]


def build_controller_request(
    observation: Mapping[str, object],
    environment: BoundEnvironment,
) -> ControllerRequest:
    """Build the deterministic teacher presentation for one stable state."""
    if not isinstance(environment, BoundEnvironment):
        raise TypeError("controller environment has an invalid type")
    validate_observation_with_contract(
        observation,
        environment.generation_contract,
        environment.verifier,
        environment_contract_sha256=environment.sha256,
        turn_limit=environment.contract.turn_limit,
    )
    presentation = {
        "schema_version": POLICY_INPUT_CONTRACT_VERSION,
        "observation": observation,
    }
    rendered = canonical_json_bytes(presentation).decode("ascii")
    user = (
        f"Controller state:\n{rendered}\n\n"
        "Select one object from observation.legal_pairs. Produce exactly "
        "{\"verb\":\"<legal verb>\",\"target\":<legal integer id>}."
    )
    return ControllerRequest(
        system=_SYSTEM_PROMPT,
        user=user,
        schema=json.loads(canonical_json_bytes(_OUTPUT_SCHEMA)),
    )


def decode_controller_payload(payload: Mapping[str, Any]) -> Action:
    """Decode an exact wire reply without performing runtime fallback."""
    if type(payload) is not dict or set(payload) != {"verb", "target"}:
        raise ControllerPayloadError(
            "controller payload keys must be exactly verb and target"
        )
    verb_value = payload["verb"]
    target = payload["target"]
    if not isinstance(verb_value, str):
        raise ControllerPayloadError("controller verb must be text")
    try:
        verb = Verb(verb_value)
    except ValueError as exc:
        raise ControllerPayloadError("controller verb is unknown") from exc
    if verb not in POLICY_VERB_ORDER:
        raise ControllerPayloadError("controller verb is not learned")
    if type(target) is not int or target < 0:
        raise ControllerPayloadError(
            "controller target must be a nonnegative integer"
        )
    return Action(verb, target)


@dataclass(frozen=True, slots=True)
class TypedLLMController:
    """Teacher action selector bound to one environment and transport."""

    environment: BoundEnvironment
    client: LLMClient
    controller_identity: ModelIdentity
    temperature: float = 0.0
    max_tokens: int = 64
    seed: int | None = None
    seed_policy: str = "none-v1"
    output_contract: str = CONTROLLER_OUTPUT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.environment, BoundEnvironment):
            raise TypeError("controller environment has an invalid type")
        if not isinstance(self.client, LLMClient):
            raise TypeError("controller client has an invalid type")
        if not isinstance(self.controller_identity, ModelIdentity):
            raise TypeError("controller identity has an invalid type")
        configured_model = getattr(self.client, "model", None)
        if (
            configured_model is not None
            and configured_model != self.controller_identity.model
        ):
            raise ValueError(
                "controller identity model differs from the configured client"
            )
        if (
            isinstance(self.client, OpenAICompatibleLLMClient)
            and self.controller_identity.revision is None
        ):
            raise ValueError(
                "hosted controller identity requires an immutable revision"
            )
        if type(self.temperature) is not float or not math.isfinite(
            self.temperature
        ):
            raise ValueError("controller temperature must be a finite float")
        if self.temperature < 0.0 or (
            self.temperature == 0.0
            and math.copysign(1.0, self.temperature) < 0.0
        ):
            raise ValueError("controller temperature must be nonnegative")
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("controller max_tokens must be positive")
        if self.seed_policy not in CONTROLLER_SEED_POLICY_IDS:
            raise ValueError("controller seed policy is unknown")
        if self.seed_policy == "none-v1":
            if self.seed is not None:
                raise ValueError("none-v1 controller cannot carry a seed")
        elif (
            type(self.seed) is not int
            or not 0 <= self.seed <= MAX_GENERATION_SEED
        ):
            raise ValueError("offset controller policy requires an int31 seed")
        schema_validation_mode_for_controller_output(self.output_contract)
        # Validate secret-free transport settings during construction.
        llm_transport_contract(self.client)

    @property
    def environment_contract_sha256(self) -> str:
        return self.environment.sha256

    @property
    def controller_implementation(self) -> str:
        return implementation_name(self)

    @property
    def controller_transport_contract(self) -> ModelTransportContract:
        return llm_transport_contract(self.client)

    @property
    def controller_transport_sha256(self) -> str:
        return llm_transport_sha256(self.client)

    @property
    def controller_seed(self) -> int | None:
        return self.seed

    @property
    def controller_temperature(self) -> float:
        return self.temperature

    @property
    def controller_max_tokens(self) -> int:
        return self.max_tokens

    @property
    def controller_seed_policy(self) -> str:
        return self.seed_policy

    @property
    def controller_output_contract(self) -> str:
        return self.output_contract

    @property
    def behavior_source(self) -> BehaviorSource:
        return BehaviorSource.TEACHER

    @property
    def policy_input_contract(self) -> str:
        return POLICY_INPUT_CONTRACT_VERSION

    def choose(
        self,
        observation: Mapping[str, object],
    ) -> ControllerChoice:
        request = build_controller_request(observation, self.environment)
        turn = observation["turn"]
        if type(turn) is not int:
            raise ValueError("controller observation turn is invalid")
        effective_seed = (
            None
            if self.seed_policy == "none-v1"
            else self.seed + turn
        )
        if (
            effective_seed is not None
            and effective_seed > MAX_GENERATION_SEED
        ):
            raise ValueError("effective controller seed exceeds int31")
        try:
            payload = self.client.complete_json(
                request.system,
                request.user,
                request.schema,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=effective_seed,
                schema_validation_mode=(
                    schema_validation_mode_for_controller_output(
                        self.controller_output_contract
                    )
                ),
            )
            action = decode_controller_payload(payload)
        except (ControllerPayloadError, LLMResponseError):
            action = None
        return ControllerChoice(action)


__all__ = [
    "CONTROLLER_OUTPUT_CONTRACT_VERSION",
    "LEGACY_CONTROLLER_OUTPUT_CONTRACT_VERSION",
    "POLICY_INPUT_CONTRACT_VERSION",
    "SUPPORTED_CONTROLLER_OUTPUT_CONTRACT_VERSIONS",
    "ControllerPayloadError",
    "ControllerRequest",
    "TypedLLMController",
    "build_controller_request",
    "decode_controller_payload",
    "schema_validation_mode_for_controller_output",
]
