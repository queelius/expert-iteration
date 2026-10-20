"""Immutable environment and episode bindings for reasoning v2."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Protocol, runtime_checkable

from .closure import AggregatePayloadBuilder, ClosureVerifier
from .contracts import (
    CollectionContract,
    EnvironmentContract,
    MAX_GENERATION_SEED,
    ModelIdentity,
    ModelTransportContract,
    canonical_json_sha256,
    implementation_name,
    verifier_reason_codes_sha256,
    validate_collection_binding,
    validate_semantic_profile_consistency,
)
from .model import Action, State, Verb
from .observation import (
    ObservationContext,
    build_observation,
    validate_observation_with_contract,
)
from .transition import (
    MacroTransition,
    PolicyDraftGenerator,
    macro_step,
    validate_macro_transition,
)
from .provenance import (
    BehaviorSource,
    ControllerChoice,
    DecisionProvenance,
)
from .verifier import GenerationContract


@runtime_checkable
class BoundPolicyDraftGenerator(PolicyDraftGenerator, Protocol):
    """A generator whose model and transition distribution are explicit."""

    environment_contract_sha256: str
    generator_identity: ModelIdentity
    lens_contract: str
    generation_schema: str
    generation_temperature: float
    generation_max_tokens: int
    generation_seed_policy: str
    generator_implementation: str
    generator_transport_sha256: str
    generator_transport_contract: ModelTransportContract
    generation_seed: int | None
    generation_retry_budget: int


@runtime_checkable
class BoundController(Protocol):
    """A controller whose identity and decision distribution are explicit."""

    environment_contract_sha256: str
    controller_identity: ModelIdentity
    controller_implementation: str
    controller_transport_sha256: str
    controller_transport_contract: ModelTransportContract
    controller_seed: int | None
    controller_temperature: float
    controller_max_tokens: int
    controller_seed_policy: str
    controller_output_contract: str
    behavior_source: BehaviorSource
    policy_input_contract: str

    def choose(
        self,
        observation: Mapping[str, object],
    ) -> ControllerChoice:
        ...


@runtime_checkable
class BoundDecisionRecorder(Protocol):
    """A collection-bound writer for composite decision receipts."""

    environment_contract_sha256: str
    collection_contract_sha256: str

    def record_decision(
        self,
        *,
        episode_key: str,
        turn: int,
        observation_sha256: str,
        source: BehaviorSource,
        proposed_action: Action | None,
        selected_action: Action,
        fallback: bool,
    ) -> str:
        ...


@dataclass(frozen=True, slots=True)
class BoundEnvironment:
    """One immutable set of semantic dependencies for an episode."""

    contract: EnvironmentContract
    generation_contract: GenerationContract
    verifier: ClosureVerifier
    aggregate_builder: AggregatePayloadBuilder

    def __post_init__(self) -> None:
        self._validate_binding()

    def _validate_binding(self) -> None:
        """Recheck identities so mutable collaborators cannot drift in place."""
        if not isinstance(self.contract, EnvironmentContract):
            raise TypeError("environment contract has an invalid type")
        if not isinstance(self.generation_contract, GenerationContract):
            raise TypeError("generation contract has an invalid type")
        if not isinstance(self.verifier, ClosureVerifier):
            raise TypeError("verifier has an invalid type")
        if not isinstance(self.aggregate_builder, AggregatePayloadBuilder):
            raise TypeError("aggregate builder has an invalid type")
        validate_semantic_profile_consistency(
            self.generation_contract,
            self.verifier,
            self.aggregate_builder,
        )

        expected = {
            "obligation_contract": self.generation_contract.obligation_contract_id,
            "generation_contract": self.generation_contract.generation_contract_id,
            "generation_contract_implementation": implementation_name(
                self.generation_contract
            ),
            "verifier_contract": self.verifier.contract_id,
            "verifier_implementation": implementation_name(self.verifier),
            "verifier_reason_codes_sha256": verifier_reason_codes_sha256(
                self.verifier
            ),
            "aggregate_contract": self.aggregate_builder.aggregate_contract_id,
            "aggregate_implementation": implementation_name(
                self.aggregate_builder
            ),
        }
        if (
            self.generation_contract.payload_semantics_id
            != self.contract.payload_semantics
        ):
            raise ValueError(
                "generation payload semantics differ from the environment"
            )
        for field, value in expected.items():
            if getattr(self.contract, field) != value:
                raise ValueError(
                    f"bound {field} differs from the environment contract"
                )

    @property
    def sha256(self) -> str:
        return self.contract.sha256

    def validate_binding(self) -> None:
        """Revalidate the immutable semantic collaborators in place."""
        self._validate_binding()

    def observe(self, state: State) -> dict[str, object]:
        self._validate_binding()
        observation = build_observation(
            state,
            ObservationContext(
                turn_limit=self.contract.turn_limit,
                environment_contract_sha256=self.sha256,
            ),
        )
        validate_observation_with_contract(
            observation,
            self.generation_contract,
            self.verifier,
            environment_contract_sha256=self.sha256,
            turn_limit=self.contract.turn_limit,
        )
        return observation

    def step(
        self,
        state: State,
        action: Action,
        generator: PolicyDraftGenerator,
    ) -> MacroTransition:
        self._validate_binding()
        transition = macro_step(
            state,
            action,
            turn_limit=self.contract.turn_limit,
            environment_contract_sha256=self.sha256,
            generator=generator,
            generation_contract=self.generation_contract,
            verifier=self.verifier,
            aggregate_builder=self.aggregate_builder,
        )
        self.validate_transition(transition)
        return transition

    def validate_transition(self, transition: MacroTransition) -> None:
        self._validate_binding()
        validate_macro_transition(
            transition,
            environment_contract_sha256=self.sha256,
            turn_limit=self.contract.turn_limit,
            generation_contract=self.generation_contract,
            verifier=self.verifier,
            aggregate_builder=self.aggregate_builder,
        )
        self.observe(transition.state)
        self.observe(transition.next_state)


@dataclass(frozen=True, slots=True)
class EpisodeRuntime:
    """Collection/live stepping with no per-turn semantic overrides."""

    environment: BoundEnvironment
    collection_contract: CollectionContract
    generator: BoundPolicyDraftGenerator
    state: State
    controller: BoundController | None = None
    episode_key: str | None = None
    decision_recorder: BoundDecisionRecorder | None = None

    def __post_init__(self) -> None:
        self._validate_binding()

    def _validate_binding(self) -> None:
        if not isinstance(self.environment, BoundEnvironment):
            raise TypeError("episode environment has an invalid type")
        if not isinstance(self.collection_contract, CollectionContract):
            raise TypeError("episode collection contract has an invalid type")
        if not isinstance(self.generator, BoundPolicyDraftGenerator):
            raise TypeError("episode generator is not contract-bound")
        if not isinstance(self.state, State):
            raise TypeError("episode state has an invalid type")
        if self.episode_key is not None and (
            not isinstance(self.episode_key, str)
            or not self.episode_key.strip()
        ):
            raise ValueError("episode_key must be null or nonempty text")
        contract = self.environment.contract
        validate_collection_binding(contract, self.collection_contract)
        checks = {
            "environment_contract_sha256": self.environment.sha256,
            "generator_identity": self.collection_contract.generator,
            "generator_implementation": (
                self.collection_contract.generator_implementation
            ),
            "generator_transport_sha256": (
                self.collection_contract.generator_transport_sha256
            ),
            "generator_transport_contract": (
                self.collection_contract.generator_transport
            ),
            "generation_seed": self.collection_contract.generation_seed,
            "lens_contract": contract.lens_contract,
            "generation_schema": contract.generation_schema,
            "generation_temperature": contract.generation_temperature,
            "generation_max_tokens": contract.generation_max_tokens,
            "generation_seed_policy": contract.generation_seed_policy,
        }
        for field, expected in checks.items():
            if getattr(self.generator, field) != expected:
                raise ValueError(
                    f"episode generator {field} differs from its contracts"
                )
        retry_budget = self.generator.generation_retry_budget
        if type(retry_budget) is not int or retry_budget < 0:
            raise ValueError("episode generator retry budget is invalid")
        if contract.generation_seed_policy == "none-v1":
            if self.collection_contract.generation_seed is not None:
                raise ValueError("none-v1 collection cannot carry a seed")
        else:
            seed = self.collection_contract.generation_seed
            if type(seed) is not int:
                raise ValueError("offset seed policy needs a collection seed")
            maximum = seed + contract.turn_limit - 1 + retry_budget
            if maximum > MAX_GENERATION_SEED:
                raise ValueError("effective generation seed exceeds int31")
        if self.state.turn > contract.turn_limit:
            raise ValueError("episode state is beyond the bound horizon")
        if self.controller is not None:
            if not isinstance(self.controller, BoundController):
                raise TypeError("episode controller is not contract-bound")
            controller_checks = {
                "environment_contract_sha256": self.environment.sha256,
                "controller_identity": self.collection_contract.controller,
                "controller_implementation": (
                    self.collection_contract.controller_implementation
                ),
                "controller_transport_sha256": (
                    self.collection_contract.controller_transport_sha256
                ),
                "controller_transport_contract": (
                    self.collection_contract.controller_transport
                ),
                "controller_seed": self.collection_contract.controller_seed,
                "controller_temperature": (
                    self.collection_contract.controller_temperature
                ),
                "controller_max_tokens": (
                    self.collection_contract.controller_max_tokens
                ),
                "controller_seed_policy": (
                    self.collection_contract.controller_seed_policy
                ),
                "controller_output_contract": (
                    self.collection_contract.controller_output_contract
                ),
                "behavior_source": BehaviorSource(
                    self.collection_contract.behavior_source
                ),
                "policy_input_contract": (
                    self.collection_contract.policy_input_contract
                ),
            }
            for field, expected in controller_checks.items():
                if getattr(self.controller, field) != expected:
                    raise ValueError(
                        f"episode controller {field} differs from its contracts"
                    )
        elif self.decision_recorder is not None:
            raise ValueError(
                "a decision recorder requires a bound controller"
            )
        if self.decision_recorder is not None:
            if self.episode_key is None:
                raise ValueError(
                    "a decision recorder requires a bound episode_key"
                )
            if not isinstance(self.decision_recorder, BoundDecisionRecorder):
                raise TypeError("episode decision recorder is not contract-bound")
            recorder_checks = {
                "environment_contract_sha256": self.environment.sha256,
                "collection_contract_sha256": self.collection_contract.sha256,
            }
            for field, expected in recorder_checks.items():
                if getattr(self.decision_recorder, field) != expected:
                    raise ValueError(
                        f"episode decision recorder {field} differs from its contracts"
                    )
        self.environment.observe(self.state)

    def step(
        self, action: Action | None = None
    ) -> tuple[EpisodeRuntime, MacroTransition]:
        self._validate_binding()
        provenance = None
        if self.controller is None:
            if not isinstance(action, Action):
                raise TypeError(
                    "an unbound-controller episode needs an explicit action"
                )
        else:
            if action is not None:
                raise TypeError(
                    "a collection-bound episode selects its own action"
                )
            observation = self.environment.observe(self.state)
            choice = self.controller.choose(observation)
            if not isinstance(choice, ControllerChoice):
                raise TypeError(
                    "bound controller returned an invalid choice"
                )
            legal = tuple(
                Action(Verb(pair["verb"]), pair["target"])
                for pair in observation["legal_pairs"]
            )
            action = choice.action
            fallback = False
            if action not in legal:
                if (
                    self.controller.behavior_source
                    is BehaviorSource.LEARNED_POLICY
                ):
                    raise ValueError(
                        "learned controller returned no legal action"
                    )
                if (
                    self.collection_contract.controller_fallback_policy
                    != "first-legal-pair-v1"
                    or not legal
                ):
                    raise ValueError("teacher fallback is unavailable")
                action = legal[0]
                fallback = True
        if action is None:
            raise AssertionError("episode action must be bound")
        transition = self.environment.step(
            self.state, action, self.generator
        )
        if self.controller is not None:
            diagnostic_record_id = None
            decision_observation_sha256 = None
            if self.decision_recorder is not None:
                if self.episode_key is None:
                    raise AssertionError("recorded episode key must be bound")
                decision_observation_sha256 = canonical_json_sha256(
                    observation
                )
                diagnostic_record_id = self.decision_recorder.record_decision(
                    episode_key=self.episode_key,
                    turn=self.state.turn,
                    observation_sha256=decision_observation_sha256,
                    source=self.controller.behavior_source,
                    proposed_action=choice.action,
                    selected_action=action,
                    fallback=fallback,
                )
            provenance = DecisionProvenance.from_choice(
                self.controller.behavior_source,
                choice,
                fallback=fallback,
                diagnostic_record_id=diagnostic_record_id,
                episode_key=(
                    self.episode_key
                    if diagnostic_record_id is not None
                    else None
                ),
                observation_sha256=decision_observation_sha256,
            )
            transition = replace(
                transition,
                collection_contract_sha256=(
                    self.collection_contract.sha256
                ),
                decision_provenance=provenance,
            )
        return (
            EpisodeRuntime(
                environment=self.environment,
                collection_contract=self.collection_contract,
                generator=self.generator,
                state=transition.next_state,
                controller=self.controller,
                episode_key=self.episode_key,
                decision_recorder=self.decision_recorder,
            ),
            transition,
        )


__all__ = [
    "BoundEnvironment",
    "BoundController",
    "BoundDecisionRecorder",
    "BoundPolicyDraftGenerator",
    "EpisodeRuntime",
]
