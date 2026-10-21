import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import verifier_anchored_rl.reasoning_v2.diagnostics as diagnostics_module

from verifier_anchored_rl.llm_client import (
    LLMResponseError,
    OpenAICompatibleLLMClient,
    SCHEMA_VALIDATION_STRICT_V1,
)
from verifier_anchored_rl.reasoning_v2.contracts import (
    CollectionContract,
    ModelIdentity,
    ModelTransportContract,
    build_environment_contract,
    canonical_json_bytes,
)
from verifier_anchored_rl.reasoning_v2.controller import (
    POLICY_INPUT_CONTRACT_VERSION,
    TypedLLMController,
    build_controller_request,
)
from verifier_anchored_rl.reasoning_v2.diagnostics import (
    CONTROLLER_ATTEMPT_COVERAGE,
    CONTROLLER_RECEIPT_COVERAGE,
    DECISION_RECEIPT_VERSION,
    DIAGNOSTIC_COVERAGE_UNAVAILABLE,
    DiagnosticSidecarIndex,
    DiagnosticSidecarWriter,
    DiagnosticValidationError,
    read_diagnostic_sidecar,
    require_bc_ready_decision_receipt,
    require_hosted_bc_diagnostic_coverage,
)
from verifier_anchored_rl.reasoning_v2.lenses import (
    GENERATION_SCHEMA_VERSION,
    LEGACY_LENS_CONTRACT_VERSION,
    LENS_CONTRACT_VERSION,
    TypedPolicyDraftGenerator,
    build_policy_lens_request,
    decode_policy_payload,
    llm_transport_contract,
)
from verifier_anchored_rl.reasoning_v2.math_adapter import (
    CaseworkMathVerifier,
)
from verifier_anchored_rl.reasoning_v2.model import (
    Action,
    GoalObjective,
    GoalObligation,
    RejectionReason,
    State,
    Verb,
    new_graph,
)
from verifier_anchored_rl.reasoning_v2.observation import observation_sha256
from verifier_anchored_rl.reasoning_v2.provenance import BehaviorSource
from verifier_anchored_rl.reasoning_v2.records import DiagnosticRef
from verifier_anchored_rl.reasoning_v2.runtime import (
    BoundEnvironment,
    EpisodeRuntime,
)


GENERATOR_ID = ModelIdentity("test", "model-m", "r1")
CONTROLLER_ID = ModelIdentity("test", "teacher", "r1")
CONTROLLER_IMPLEMENTATION = (
    f"{TypedLLMController.__module__}.{TypedLLMController.__qualname__}"
)
CONTROLLER_TRANSPORT = ModelTransportContract.local(
    CONTROLLER_IMPLEMENTATION
)


def environment(*, lens_contract=LENS_CONTRACT_VERSION):
    adapter = CaseworkMathVerifier()
    contract = build_environment_contract(
        generation_contract=adapter,
        verifier=adapter,
        aggregate_builder=adapter,
        lens_contract=lens_contract,
        generation_schema=GENERATION_SCHEMA_VERSION,
        generation_temperature=0.0,
        generation_max_tokens=64,
        generation_seed_policy="turn-and-retry-offset-v1",
        turn_limit=6,
    )
    return BoundEnvironment(contract, adapter, adapter, adapter)


def configured_client(*, max_retries=1, max_backoff_retries=1):
    return OpenAICompatibleLLMClient(
        base_url="https://provider.example/v1",
        model="model-m",
        api_key="SUPERSECRET",
        max_retries=max_retries,
        max_backoff_retries=max_backoff_retries,
    )


def configured_controller_client(*, max_retries=1):
    return OpenAICompatibleLLMClient(
        base_url="https://controller.example/v1",
        model="teacher",
        api_key="CONTROLLERSECRET",
        max_retries=max_retries,
        max_backoff_retries=1,
    )


def collection_for(env, client, controller_client=None):
    generator = TypedPolicyDraftGenerator(
        env,
        client,
        GENERATOR_ID,
        seed=11,
    )
    return CollectionContract(
        generator=GENERATOR_ID,
        controller=CONTROLLER_ID,
        controller_implementation=CONTROLLER_IMPLEMENTATION,
        controller_transport=(
            CONTROLLER_TRANSPORT
            if controller_client is None
            else llm_transport_contract(controller_client)
        ),
        controller_seed=None if controller_client is None else 19,
        behavior_source="teacher",
        controller_fallback_policy="first-legal-pair-v1",
        generator_implementation=generator.generator_implementation,
        generator_transport=generator.generator_transport_contract,
        generation_seed=generator.generation_seed,
        policy_input_contract=POLICY_INPUT_CONTRACT_VERSION,
        problem_source_contract="test-problems-v1",
        problem_source_sha256="1" * 64,
        problem_adapter_implementation="tests.DiagnosticProblemAdapter",
        controller_temperature=0.0,
        controller_max_tokens=64,
        controller_seed_policy=(
            "none-v1"
            if controller_client is None
            else "turn-and-retry-offset-v1"
        ),
    )


class _Response:
    def __init__(self, status, payload=None, *, body=None, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = (
            body
            if body is not None
            else json.dumps(payload, separators=(",", ":"))
        )

    def raise_for_status(self):
        if self.status_code < 400:
            return
        import httpx

        request = httpx.Request("POST", "https://provider.example/v1")
        response = httpx.Response(
            self.status_code,
            request=request,
            headers=self.headers,
            text=self.text,
        )
        raise httpx.HTTPStatusError(
            "provider status",
            request=request,
            response=response,
        )

    def json(self):
        return self._payload


def install_responses(monkeypatch, responses):
    import httpx

    queue = list(responses)
    calls = []
    lock = threading.Lock()

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            with lock:
                calls.append({
                    "url": url,
                    "headers": dict(headers or {}),
                    "body": json,
                })
                response = queue.pop(0)
                if isinstance(response, BaseException):
                    raise response
                return response

    monkeypatch.setattr(httpx, "Client", _Client)
    monkeypatch.setattr(
        "verifier_anchored_rl.llm_client.time.sleep", lambda delay: None
    )
    monkeypatch.setattr(
        "verifier_anchored_rl.llm_client.random.uniform",
        lambda low, high: 0.0,
    )
    return calls


def schema():
    return {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }


def canonical_decision(env, action):
    graph = new_graph(
        "solve",
        GoalObligation(
            "x", "Eq(x, 5)", "Reals", GoalObjective.SUM
        ),
        generation_contract=env.generation_contract,
    )
    observation = env.observe(State(graph, turn=0))
    digest = observation_sha256(
        observation,
        env.generation_contract,
        env.verifier,
        environment_contract_sha256=env.sha256,
    )
    return observation, digest


def hosted_decision_trace(
    monkeypatch,
    path,
    *,
    controller_content=None,
    generator_content='{"value":"5"}',
    lens_contract=LENS_CONTRACT_VERSION,
):
    action = Action(Verb.PROPOSE_RESULT, 0)
    content = controller_content or (
        '{"verb":"propose_result","target":0}'
    )
    install_responses(monkeypatch, [
        _Response(200, {
            "choices": [{"message": {"content": content}}],
        }),
        _Response(200, {
            "choices": [{"message": {"content": generator_content}}],
        }),
    ])
    env = environment(lens_contract=lens_contract)
    generator_client = configured_client(max_retries=0)
    controller_client = configured_controller_client(max_retries=0)
    collection = collection_for(env, generator_client, controller_client)
    observation, digest = canonical_decision(env, action)
    controller_request = build_controller_request(observation, env)
    generator_request = build_policy_lens_request(
        observation, action, lens_contract=lens_contract
    )
    with DiagnosticSidecarWriter(
        path,
        environment=env,
        collection_contract=collection,
        generator_client=generator_client,
        controller_client=controller_client,
    ) as writer:
        controller_client.complete_json(
            controller_request.system,
            controller_request.user,
            controller_request.schema,
            temperature=0.0,
            max_tokens=64,
            seed=19,
        )
        generator_client.complete_json(
            generator_request.system,
            generator_request.user,
            generator_request.schema,
            temperature=0.0,
            max_tokens=64,
            seed=11,
        )
        reference = DiagnosticRef(writer.record_decision(
            episode_key="problem:hosted",
            turn=0,
            observation_sha256=digest,
            source=BehaviorSource.TEACHER,
            proposed_action=action,
            selected_action=action,
            fallback=False,
        ))
    return env, collection, observation, digest, action, reference


def hosted_generator_retry_trace(
    monkeypatch,
    path,
    *,
    first_generator_response,
):
    action = Action(Verb.PROPOSE_RESULT, 0)
    install_responses(monkeypatch, [
        _Response(200, {
            "choices": [{"message": {"content": (
                '{"verb":"propose_result","target":0}'
            )}}],
        }),
        first_generator_response,
        _Response(200, {
            "choices": [{"message": {"content": '{"value":"5"}'}}],
        }),
    ])
    env = environment()
    generator_client = configured_client(
        max_retries=1, max_backoff_retries=1
    )
    controller_client = configured_controller_client(max_retries=0)
    collection = collection_for(env, generator_client, controller_client)
    observation, digest = canonical_decision(env, action)
    controller_request = build_controller_request(observation, env)
    generator_request = build_policy_lens_request(observation, action)
    with DiagnosticSidecarWriter(
        path,
        environment=env,
        collection_contract=collection,
        generator_client=generator_client,
        controller_client=controller_client,
    ) as writer:
        controller_client.complete_json(
            controller_request.system,
            controller_request.user,
            controller_request.schema,
            temperature=0.0,
            max_tokens=64,
            seed=19,
        )
        generator_client.complete_json(
            generator_request.system,
            generator_request.user,
            generator_request.schema,
            temperature=0.0,
            max_tokens=64,
            seed=11,
        )
        reference = DiagnosticRef(writer.record_decision(
            episode_key="problem:hosted",
            turn=0,
            observation_sha256=digest,
            source=BehaviorSource.TEACHER,
            proposed_action=action,
            selected_action=action,
            fallback=False,
        ))
    return env, collection, observation, digest, action, reference


def validate_hosted_reference(
    index, reference, env, collection, observation, digest, action
):
    return index.validate_decision_reference(
        reference,
        environment=env,
        collection_contract=collection,
        observation=observation,
        collection_contract_sha256=collection.sha256,
        episode_key="problem:hosted",
        turn=0,
        observation_sha256=digest,
        selected_action=action,
        source="teacher",
        fallback=False,
        bc_eligible=True,
        expected_policy_draft=decode_policy_payload(
            observation, action, {"value": "5"}
        ),
        expected_rejection_reason=None,
        require_bc_trust=True,
    )


def test_diagnostic_replay_uses_role_bound_strict_wire_versions(
    monkeypatch, tmp_path
):
    path = tmp_path / "strict-wire-replay.jsonl"
    trace = hosted_decision_trace(monkeypatch, path)
    original = diagnostics_module.classify_openai_compatible_json_content
    observed = []

    def capture_mode(raw_content, schema, *, schema_validation_mode):
        observed.append(schema_validation_mode)
        return original(
            raw_content,
            schema,
            schema_validation_mode=schema_validation_mode,
        )

    monkeypatch.setattr(
        diagnostics_module,
        "classify_openai_compatible_json_content",
        capture_mode,
    )
    env, collection, observation, digest, action, reference = trace
    index = read_diagnostic_sidecar(path)

    validate_hosted_reference(
        index,
        reference,
        env,
        collection,
        observation,
        digest,
        action,
    )

    assert observed == [
        SCHEMA_VALIDATION_STRICT_V1,
        SCHEMA_VALIDATION_STRICT_V1,
    ]


def test_diagnostics_reconstruct_legacy_lens_request(monkeypatch, tmp_path):
    path = tmp_path / "legacy-lens-diagnostics.jsonl"
    (
        env,
        collection,
        observation,
        digest,
        action,
        reference,
    ) = hosted_decision_trace(
        monkeypatch,
        path,
        lens_contract=LEGACY_LENS_CONTRACT_VERSION,
    )

    index = read_diagnostic_sidecar(path)
    assessment = validate_hosted_reference(
        index,
        reference,
        env,
        collection,
        observation,
        digest,
        action,
    )

    assert assessment.sufficient_for_bc
    assert assessment.generator_request_bound
    generator_diagnostic = assessment.resolved.generator_diagnostic
    assert generator_diagnostic is not None
    generator_attempt = generator_diagnostic.attempt_records[0]
    assert (
        'Return {"value":"symbolic value"}.'
        in generator_attempt["request_body"]["messages"][1]["content"]
    )


def hosted_generator_failure_trace(monkeypatch, path, *, payload_failure):
    controller_content = '{"verb":"propose_result","target":0}'
    generator_response = (
        _Response(200, {
            "choices": [{"message": {"content": "not json"}}],
        })
        if payload_failure
        else _Response(500, {"error": "down"})
    )
    install_responses(monkeypatch, [
        _Response(200, {
            "choices": [{"message": {"content": controller_content}}],
        }),
        generator_response,
    ])
    env = environment()
    generator_client = configured_client(
        max_retries=0, max_backoff_retries=0
    )
    controller_client = configured_controller_client(max_retries=0)
    collection = collection_for(env, generator_client, controller_client)
    action = Action(Verb.PROPOSE_RESULT, 0)
    observation, digest = canonical_decision(env, action)
    controller_request = build_controller_request(observation, env)
    generator_request = build_policy_lens_request(observation, action)
    with DiagnosticSidecarWriter(
        path,
        environment=env,
        collection_contract=collection,
        generator_client=generator_client,
        controller_client=controller_client,
    ) as writer:
        controller_client.complete_json(
            controller_request.system,
            controller_request.user,
            controller_request.schema,
            temperature=0.0,
            max_tokens=64,
            seed=19,
        )
        with pytest.raises(LLMResponseError):
            generator_client.complete_json(
                generator_request.system,
                generator_request.user,
                generator_request.schema,
                temperature=0.0,
                max_tokens=64,
                seed=11,
            )
        reference = DiagnosticRef(writer.record_decision(
            episode_key="problem:hosted-failure",
            turn=0,
            observation_sha256=digest,
            source=BehaviorSource.TEACHER,
            proposed_action=action,
            selected_action=action,
            fallback=False,
        ))
    return env, collection, observation, digest, action, reference


def test_sidecar_records_backoff_schema_retry_usage_seed_and_resolves(
    monkeypatch, tmp_path
):
    responses = [
        _Response(
            429,
            {"error": "busy"},
            body='{"error":"busy"}',
            headers={"retry-after": "0"},
        ),
        _Response(200, {
            "choices": [{"message": {"content": '{"value":[]}'}}],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
            },
        }),
        _Response(200, {
            "choices": [{"message": {"content": '{"value":"5"}'}}],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 3,
                "total_tokens": 11,
            },
        }),
    ]
    calls = install_responses(monkeypatch, responses)
    env = environment()
    client = configured_client()
    collection = collection_for(env, client)
    path = tmp_path / "diagnostics.jsonl"

    with DiagnosticSidecarWriter(
        path,
        environment=env,
        collection_contract=collection,
        generator_client=client,
    ) as writer:
        result = client.complete_json(
            "system", "user", schema(), temperature=0.0, max_tokens=64,
            seed=11,
        )
        reference = writer.take_diagnostic_ref()
        with pytest.raises(RuntimeError, match="no completed"):
            writer.take_diagnostic_ref()

    assert result == {"value": "5"}
    assert len(calls) == 3
    assert calls[0]["headers"]["Authorization"] == "Bearer SUPERSECRET"
    assert b"SUPERSECRET" not in path.read_bytes()
    assert b"Authorization" not in path.read_bytes()

    index = read_diagnostic_sidecar(
        path,
        environment_contract_sha256=env.sha256,
        collection_contract_sha256=collection.sha256,
        generator_transport_sha256=collection.generator_transport_sha256,
    )
    assert (
        index.header["coverage"]["controller_attempts"]
        == DIAGNOSTIC_COVERAGE_UNAVAILABLE
    )
    resolved = index.resolve(reference)
    attempts = resolved.attempt_records
    assert [item["outcome"] for item in attempts] == [
        "http_status_error", "schema_error", "accepted"
    ]
    assert [item["disposition"] for item in attempts] == [
        "backoff_retry", "parse_retry", "return"
    ]
    assert [item["effective_seed"] for item in attempts] == [11, 11, 12]
    assert attempts[0]["raw_response_body"] == '{"error":"busy"}'
    assert attempts[1]["raw_content"] == '{"value":[]}'
    assert attempts[1]["usage"]["prompt_tokens"] == 5
    assert attempts[2]["usage"]["raw"]["total_tokens"] == 11
    assert len(attempts[2]["request_body"]["messages"]) == 4
    assert resolved.call_record["final_outcome"] == "transport_accepted"


def test_decision_receipt_composes_generator_trace_and_gates_hosted_bc(
    monkeypatch, tmp_path
):
    install_responses(monkeypatch, [
        _Response(200, {
            "choices": [{"message": {"content": '{"value":"5"}'}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        })
    ])
    env = environment()
    client = configured_client(max_retries=0)
    collection = collection_for(env, client)
    path = tmp_path / "receipt-diagnostics.jsonl"
    action = Action(Verb.PROPOSE_RESULT, 0)
    observation, observation_digest = canonical_decision(env, action)
    request = build_policy_lens_request(observation, action)

    with DiagnosticSidecarWriter(
        path,
        environment=env,
        collection_contract=collection,
        generator_client=client,
    ) as writer:
        client.complete_json(
            request.system,
            request.user,
            request.schema,
            temperature=0.0,
            max_tokens=64,
            seed=11,
        )
        receipt_ref = DiagnosticRef(writer.record_decision(
            episode_key="problem:0",
            turn=0,
            observation_sha256=observation_digest,
            source=BehaviorSource.TEACHER,
            proposed_action=action,
            selected_action=action,
            fallback=False,
        ))

    index = read_diagnostic_sidecar(path)
    assert (
        index.header["coverage"]["controller_decision_receipts"]
        == CONTROLLER_RECEIPT_COVERAGE
    )
    resolved = index.resolve_decision_receipt(receipt_ref)
    assert resolved.receipt_record["collection_contract_sha256"] == (
        collection.sha256
    )
    assert resolved.receipt_record["selected_action"] == {
        "verb": "propose_result", "target": 0
    }
    assert resolved.generator_diagnostic.call_record["role"] == "generator"
    expected_draft = decode_policy_payload(
        observation, action, {"value": "5"}
    )
    assessment = index.validate_decision_reference(
        receipt_ref,
        environment=env,
        collection_contract=collection,
        observation=observation,
        collection_contract_sha256=collection.sha256,
        episode_key="problem:0",
        turn=0,
        observation_sha256=observation_digest,
        selected_action=action,
        source="teacher",
        fallback=False,
        bc_eligible=True,
        expected_policy_draft=expected_draft,
        expected_rejection_reason=None,
    )
    assert assessment.receipt_coverage
    assert not assessment.controller_attempt_coverage
    assert not assessment.controller_call_resolved
    assert assessment.generator_request_bound
    assert assessment.generator_effect_bound
    assert not assessment.sufficient_for_bc
    with pytest.raises(DiagnosticValidationError, match="fully linked"):
        index.validate_decision_reference(
            receipt_ref,
            environment=env,
            collection_contract=collection,
            observation=observation,
            collection_contract_sha256=collection.sha256,
            episode_key="problem:0",
            turn=0,
            observation_sha256=observation_digest,
            selected_action=action,
            source="teacher",
            fallback=False,
            bc_eligible=True,
            expected_policy_draft=expected_draft,
            expected_rejection_reason=None,
            require_bc_trust=True,
        )
    with pytest.raises(DiagnosticValidationError, match="observation digest"):
        index.validate_decision_reference(
            receipt_ref,
            environment=env,
            collection_contract=collection,
            observation=observation,
            collection_contract_sha256=collection.sha256,
            episode_key="problem:0",
            turn=0,
            observation_sha256="3" * 64,
            selected_action=action,
            source="teacher",
            fallback=False,
            bc_eligible=True,
            expected_policy_draft=expected_draft,
            expected_rejection_reason=None,
        )
    with pytest.raises(DiagnosticValidationError, match="raw controller"):
        require_bc_ready_decision_receipt(resolved.receipt_record)
    with pytest.raises(DiagnosticValidationError, match="hosted BC"):
        require_hosted_bc_diagnostic_coverage(index.header)


def test_parse_failure_still_has_a_resolvable_terminal_call(
    monkeypatch, tmp_path
):
    install_responses(monkeypatch, [
        _Response(200, {
            "choices": [{"message": {"content": "not json"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        })
    ])
    env = environment()
    client = configured_client(max_retries=0)
    collection = collection_for(env, client)
    path = tmp_path / "failed-diagnostics.jsonl"

    with DiagnosticSidecarWriter(
        path,
        environment=env,
        collection_contract=collection,
        generator_client=client,
    ) as writer:
        with pytest.raises(LLMResponseError):
            client.complete_json(
                "system", "user", schema(), temperature=0.0,
                max_tokens=64, seed=11,
            )
        reference = writer.take_diagnostic_ref()

    resolved = read_diagnostic_sidecar(path).resolve(reference)
    assert resolved.call_record["final_outcome"] == "transport_error"
    assert resolved.call_record["final_attempt_outcome"] == "parse_error"
    assert resolved.attempt_records[0]["disposition"] == "raise"
    assert resolved.attempt_records[0]["raw_content"] == "not json"


def test_hosted_teacher_receipt_consumes_both_roles_and_is_bc_trusted(
    monkeypatch, tmp_path
):
    install_responses(monkeypatch, [
        _Response(200, {
            "choices": [{
                "message": {
                    "content": (
                        '{"verb":"propose_result","target":0}'
                    )
                }
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }),
        _Response(200, {
            "choices": [{"message": {"content": '{"value":"5"}'}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }),
    ])
    env = environment()
    generator_client = configured_client(max_retries=0)
    controller_client = configured_controller_client(max_retries=0)
    collection = collection_for(env, generator_client, controller_client)
    path = tmp_path / "hosted-receipt-diagnostics.jsonl"
    action = Action(Verb.PROPOSE_RESULT, 0)
    observation, observation_digest = canonical_decision(env, action)
    controller_request = build_controller_request(observation, env)
    generator_request = build_policy_lens_request(observation, action)

    with DiagnosticSidecarWriter(
        path,
        environment=env,
        collection_contract=collection,
        generator_client=generator_client,
        controller_client=controller_client,
    ) as writer:
        controller_client.complete_json(
            controller_request.system,
            controller_request.user,
            controller_request.schema,
            temperature=0.0, max_tokens=64, seed=19,
        )
        generator_client.complete_json(
            generator_request.system,
            generator_request.user,
            generator_request.schema,
            temperature=0.0, max_tokens=64, seed=11,
        )
        receipt_ref = DiagnosticRef(writer.record_decision(
            episode_key="problem:hosted",
            turn=0,
            observation_sha256=observation_digest,
            source=BehaviorSource.TEACHER,
            proposed_action=action,
            selected_action=action,
            fallback=False,
        ))

    assert generator_client.attempt_observer is None
    assert controller_client.attempt_observer is None
    index = read_diagnostic_sidecar(
        path,
        controller_transport_sha256=(
            collection.controller_transport_sha256
        ),
    )
    assert (
        index.header["coverage"]["controller_attempts"]
        == CONTROLLER_ATTEMPT_COVERAGE
    )
    require_hosted_bc_diagnostic_coverage(index.header)
    assessment = index.validate_decision_reference(
        receipt_ref,
        environment=env,
        collection_contract=collection,
        observation=observation,
        collection_contract_sha256=collection.sha256,
        episode_key="problem:hosted",
        turn=0,
        observation_sha256=observation_digest,
        selected_action=action,
        source="teacher",
        fallback=False,
        bc_eligible=True,
        expected_policy_draft=decode_policy_payload(
            observation, action, {"value": "5"}
        ),
        expected_rejection_reason=None,
        require_bc_trust=True,
    )
    assert assessment.sufficient_for_bc
    assert assessment.generator_request_bound
    assert assessment.generator_effect_bound
    assert assessment.controller_request_bound
    assert assessment.controller_payload_bound
    with pytest.raises(DiagnosticValidationError, match="receipt alone"):
        require_bc_ready_decision_receipt(
            assessment.resolved.receipt_record
        )
    assert assessment.resolved.controller_diagnostic is not None
    assert (
        assessment.resolved.controller_diagnostic.call_record["role"]
        == "controller"
    )


def test_episode_runtime_records_one_fully_linked_hosted_decision(
    monkeypatch, tmp_path
):
    install_responses(monkeypatch, [
        _Response(200, {
            "choices": [{"message": {"content": (
                '{"verb":"propose_result","target":0}'
            )}}],
        }),
        _Response(200, {
            "choices": [{"message": {"content": '{"value":"5"}'}}],
        }),
    ])
    env = environment()
    generator_client = configured_client(max_retries=0)
    controller_client = configured_controller_client(max_retries=0)
    collection = collection_for(env, generator_client, controller_client)
    generator = TypedPolicyDraftGenerator(
        env, generator_client, GENERATOR_ID, seed=11
    )
    controller = TypedLLMController(
        env,
        controller_client,
        CONTROLLER_ID,
        seed=19,
        seed_policy="turn-and-retry-offset-v1",
    )
    graph = new_graph(
        "solve",
        GoalObligation("x", "Eq(x, 5)", "Reals", GoalObjective.SUM),
        generation_contract=env.generation_contract,
    )
    initial = State(graph, turn=0)
    observation = env.observe(initial)
    action = Action(Verb.PROPOSE_RESULT, 0)
    path = tmp_path / "runtime-diagnostics.jsonl"

    with DiagnosticSidecarWriter(
        path,
        environment=env,
        collection_contract=collection,
        generator_client=generator_client,
        controller_client=controller_client,
    ) as writer:
        runtime = EpisodeRuntime(
            environment=env,
            collection_contract=collection,
            generator=generator,
            state=initial,
            controller=controller,
            episode_key="problem:runtime",
            decision_recorder=writer,
        )
        _, transition = runtime.step()

    provenance = transition.decision_provenance
    assert provenance is not None
    assert provenance.source is BehaviorSource.TEACHER
    assert provenance.episode_key == "problem:runtime"
    reference = DiagnosticRef(provenance.diagnostic_record_id)
    index = read_diagnostic_sidecar(path)
    assessment = index.validate_decision_reference(
        reference,
        environment=env,
        collection_contract=collection,
        observation=observation,
        collection_contract_sha256=collection.sha256,
        episode_key="problem:runtime",
        turn=0,
        observation_sha256=provenance.observation_sha256,
        selected_action=action,
        source="teacher",
        fallback=False,
        bc_eligible=True,
        expected_policy_draft=decode_policy_payload(
            observation, action, {"value": "5"}
        ),
        expected_rejection_reason=None,
        require_bc_trust=True,
    )
    assert assessment.sufficient_for_bc


@pytest.mark.parametrize(
    "drift",
    ["model", "temperature", "max_tokens", "seed", "prompt"],
)
def test_hosted_bc_rejects_contract_and_prompt_drift(
    monkeypatch, tmp_path, drift
):
    path = tmp_path / f"hosted-{drift}-drift.jsonl"
    (
        env,
        collection,
        observation,
        digest,
        action,
        reference,
    ) = hosted_decision_trace(monkeypatch, path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    attempt = next(
        record
        for record in records
        if record.get("role") == "controller"
        and "request_body" in record
    )
    request = attempt["request_body"]
    if drift == "model":
        request["model"] = "different-teacher"
    elif drift == "temperature":
        request["temperature"] = 0.5
    elif drift == "max_tokens":
        request["max_tokens"] = 65
    elif drift == "seed":
        request["seed"] = 20
        attempt["effective_seed"] = 20
    else:
        request["messages"][0]["content"] += " tampered"
    path.write_bytes(
        b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    )

    index = read_diagnostic_sidecar(path)
    with pytest.raises(DiagnosticValidationError, match="request body"):
        validate_hosted_reference(
            index,
            reference,
            env,
            collection,
            observation,
            digest,
            action,
        )


def test_hosted_bc_rejects_raw_controller_action_mismatch(
    monkeypatch, tmp_path
):
    path = tmp_path / "hosted-action-mismatch.jsonl"
    (
        env,
        collection,
        observation,
        digest,
        action,
        reference,
    ) = hosted_decision_trace(
        monkeypatch,
        path,
        controller_content='{"verb":"decompose","target":0}',
    )

    index = read_diagnostic_sidecar(path)
    with pytest.raises(DiagnosticValidationError, match="raw payload"):
        validate_hosted_reference(
            index,
            reference,
            env,
            collection,
            observation,
            digest,
            action,
        )


def test_hosted_bc_rejects_raw_generator_effect_mismatch(
    monkeypatch, tmp_path
):
    path = tmp_path / "hosted-generator-mismatch.jsonl"
    (
        env,
        collection,
        observation,
        digest,
        action,
        reference,
    ) = hosted_decision_trace(
        monkeypatch, path, generator_content='{"value":"999"}'
    )

    index = read_diagnostic_sidecar(path)
    with pytest.raises(DiagnosticValidationError, match="learned effect"):
        validate_hosted_reference(
            index,
            reference,
            env,
            collection,
            observation,
            digest,
            action,
        )


@pytest.mark.parametrize(
    ("payload_failure", "expected_reason"),
    [
        (True, RejectionReason.PAYLOAD_ERROR),
        (False, RejectionReason.GENERATION_ERROR),
    ],
)
def test_hosted_generator_failure_category_is_exactly_linked(
    monkeypatch, tmp_path, payload_failure, expected_reason
):
    path = tmp_path / f"hosted-{expected_reason.value}.jsonl"
    (
        env,
        collection,
        observation,
        digest,
        action,
        reference,
    ) = hosted_generator_failure_trace(
        monkeypatch, path, payload_failure=payload_failure
    )
    index = read_diagnostic_sidecar(path)
    assessment = index.validate_decision_reference(
        reference,
        environment=env,
        collection_contract=collection,
        observation=observation,
        collection_contract_sha256=collection.sha256,
        episode_key="problem:hosted-failure",
        turn=0,
        observation_sha256=digest,
        selected_action=action,
        source="teacher",
        fallback=False,
        bc_eligible=True,
        expected_policy_draft=None,
        expected_rejection_reason=expected_reason,
        require_bc_trust=True,
    )
    assert assessment.generator_effect_bound
    assert assessment.sufficient_for_bc

    wrong_reason = (
        RejectionReason.GENERATION_ERROR
        if payload_failure
        else RejectionReason.PAYLOAD_ERROR
    )
    with pytest.raises(DiagnosticValidationError, match="learned effect"):
        index.validate_decision_reference(
            reference,
            environment=env,
            collection_contract=collection,
            observation=observation,
            collection_contract_sha256=collection.sha256,
            episode_key="problem:hosted-failure",
            turn=0,
            observation_sha256=digest,
            selected_action=action,
            source="teacher",
            fallback=False,
            bc_eligible=True,
            expected_policy_draft=None,
            expected_rejection_reason=wrong_reason,
        )


@pytest.mark.parametrize("projection", ["content", "usage"])
def test_hosted_bc_derives_projections_from_raw_response_body(
    monkeypatch, tmp_path, projection
):
    path = tmp_path / f"hosted-forged-{projection}.jsonl"
    (
        env,
        collection,
        observation,
        digest,
        action,
        reference,
    ) = hosted_decision_trace(monkeypatch, path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    attempt = next(
        record
        for record in records
        if record.get("role") == "controller"
        and "request_body" in record
    )
    if projection == "content":
        attempt["raw_content"] = '{"verb":"decompose","target":0}'
    else:
        attempt["usage"]["prompt_tokens"] = 999
    path.write_bytes(
        b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    )

    index = read_diagnostic_sidecar(path)
    with pytest.raises(DiagnosticValidationError, match="response envelope"):
        validate_hosted_reference(
            index,
            reference,
            env,
            collection,
            observation,
            digest,
            action,
        )


def test_hosted_bc_rejects_a_content_outcome_relabel(
    monkeypatch, tmp_path
):
    path = tmp_path / "hosted-relabelled-generator.jsonl"
    (
        env,
        collection,
        observation,
        digest,
        action,
        reference,
    ) = hosted_decision_trace(monkeypatch, path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    attempt = next(
        record
        for record in records
        if record.get("role") == "generator"
        and "request_body" in record
    )
    call = next(
        record
        for record in records
        if record.get("role") == "generator"
        and "final_outcome" in record
    )
    attempt["outcome"] = "parse_error"
    attempt["disposition"] = "raise"
    attempt["error"] = {
        "type": "LLMResponseError",
        "message": "forged parse failure",
    }
    call["final_outcome"] = "transport_error"
    call["final_attempt_outcome"] = "parse_error"
    path.write_bytes(
        b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    )

    index = read_diagnostic_sidecar(path)
    with pytest.raises(DiagnosticValidationError, match="raw response"):
        validate_hosted_reference(
            index,
            reference,
            env,
            collection,
            observation,
            digest,
            action,
        )


@pytest.mark.parametrize(
    ("failure_kind", "expected_outcome"),
    [
        ("parse", "parse_error"),
        ("schema", "schema_error"),
        ("provider", "provider_response_error"),
        ("transport", "transport_error"),
        ("429", "http_status_error"),
        ("500", "http_status_error"),
    ],
)
def test_offline_validation_rejects_a_retryable_failure_forged_terminal(
    monkeypatch, tmp_path, failure_kind, expected_outcome
):
    if failure_kind == "parse":
        first_response = _Response(200, {
            "choices": [{"message": {"content": "not json"}}],
        })
    elif failure_kind == "schema":
        first_response = _Response(200, {
            "choices": [{"message": {"content": '{"value":[]}'}}],
        })
    elif failure_kind == "provider":
        first_response = _Response(200, {})
    elif failure_kind == "transport":
        import httpx

        first_response = httpx.ConnectError(
            "connection failed",
            request=httpx.Request(
                "POST", "https://provider.example/v1/chat/completions"
            ),
        )
    else:
        first_response = _Response(
            int(failure_kind),
            {"error": "retryable"},
            body='{"error":"retryable"}',
            headers={"retry-after": "0"},
        )

    path = tmp_path / f"forged-terminal-{failure_kind}.jsonl"
    (
        env,
        collection,
        observation,
        digest,
        action,
        reference,
    ) = hosted_generator_retry_trace(
        monkeypatch,
        path,
        first_generator_response=first_response,
    )
    records = [json.loads(line) for line in path.read_text().splitlines()]
    first_attempt = next(
        record
        for record in records
        if record.get("role") == "generator"
        and record.get("attempt_index") == 0
    )
    assert first_attempt["outcome"] == expected_outcome
    first_attempt["disposition"] = "raise"
    call = next(
        record
        for record in records
        if record.get("role") == "generator"
        and "final_outcome" in record
    )
    call["attempt_record_ids"] = [first_attempt["record_id"]]
    call["final_outcome"] = "transport_error"
    call["final_attempt_outcome"] = expected_outcome
    records = [
        record
        for record in records
        if not (
            record.get("role") == "generator"
            and record.get("attempt_index", 0) > 0
        )
    ]
    path.write_bytes(
        b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    )

    index = read_diagnostic_sidecar(path)
    with pytest.raises(
        DiagnosticValidationError, match="live retry policy"
    ):
        validate_hosted_reference(
            index,
            reference,
            env,
            collection,
            observation,
            digest,
            action,
        )


def test_offline_validation_rejects_backoff_for_nonretryable_status(
    monkeypatch, tmp_path
):
    path = tmp_path / "forged-backoff-400.jsonl"
    (
        env,
        collection,
        observation,
        digest,
        action,
        reference,
    ) = hosted_generator_retry_trace(
        monkeypatch,
        path,
        first_generator_response=_Response(
            500,
            {"error": "server"},
            body='{"error":"server"}',
            headers={"retry-after": "0"},
        ),
    )
    records = [json.loads(line) for line in path.read_text().splitlines()]
    first_attempt = next(
        record
        for record in records
        if record.get("role") == "generator"
        and record.get("attempt_index") == 0
    )
    first_attempt["http_status"] = 400
    path.write_bytes(
        b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    )

    index = read_diagnostic_sidecar(path)
    with pytest.raises(
        DiagnosticValidationError, match="live retry policy"
    ):
        validate_hosted_reference(
            index,
            reference,
            env,
            collection,
            observation,
            digest,
            action,
        )


def test_fallback_receipt_cannot_relabel_a_legal_controller_action(
    monkeypatch, tmp_path
):
    proposed = Action(Verb.PROPOSE_RESULT, 0)
    selected = Action(Verb.DECOMPOSE, 0)
    domains = ["x <= 0", "x > 0"]
    install_responses(monkeypatch, [
        _Response(200, {
            "choices": [{"message": {"content": (
                '{"verb":"propose_result","target":0}'
            )}}],
        }),
        _Response(200, {
            "choices": [{"message": {"content": (
                '{"domains":["x <= 0","x > 0"]}'
            )}}],
        }),
    ])
    env = environment()
    generator_client = configured_client(max_retries=0)
    controller_client = configured_controller_client(max_retries=0)
    collection = collection_for(env, generator_client, controller_client)
    observation, digest = canonical_decision(env, selected)
    controller_request = build_controller_request(observation, env)
    generator_request = build_policy_lens_request(observation, selected)
    path = tmp_path / "legal-action-relabelled-fallback.jsonl"

    with DiagnosticSidecarWriter(
        path,
        environment=env,
        collection_contract=collection,
        generator_client=generator_client,
        controller_client=controller_client,
    ) as writer:
        controller_client.complete_json(
            controller_request.system,
            controller_request.user,
            controller_request.schema,
            temperature=0.0,
            max_tokens=64,
            seed=19,
        )
        generator_client.complete_json(
            generator_request.system,
            generator_request.user,
            generator_request.schema,
            temperature=0.0,
            max_tokens=64,
            seed=11,
        )
        reference = DiagnosticRef(writer.record_decision(
            episode_key="problem:fallback",
            turn=0,
            observation_sha256=digest,
            source=BehaviorSource.TEACHER,
            proposed_action=proposed,
            selected_action=selected,
            fallback=True,
        ))

    index = read_diagnostic_sidecar(path)
    with pytest.raises(
        DiagnosticValidationError, match="relabels a legal proposed action"
    ):
        index.validate_decision_reference(
            reference,
            environment=env,
            collection_contract=collection,
            observation=observation,
            collection_contract_sha256=collection.sha256,
            episode_key="problem:fallback",
            turn=0,
            observation_sha256=digest,
            selected_action=selected,
            source="teacher",
            fallback=True,
            bc_eligible=False,
            expected_policy_draft=decode_policy_payload(
                observation, selected, {"domains": domains}
            ),
            expected_rejection_reason=None,
        )


def test_reader_rejects_call_reuse_across_decision_receipts(
    monkeypatch, tmp_path
):
    path = tmp_path / "reused-call-receipts.jsonl"
    hosted_decision_trace(monkeypatch, path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    duplicate = dict(next(
        record
        for record in records
        if record.get("schema_version") == DECISION_RECEIPT_VERSION
    ))
    duplicate["record_id"] = "f" * 32
    path.write_bytes(
        b"".join(
            canonical_json_bytes(record) + b"\n"
            for record in (*records, duplicate)
        )
    )

    with pytest.raises(DiagnosticValidationError, match="call is reused"):
        read_diagnostic_sidecar(path)


def test_reader_rejects_unannounced_controller_records(
    monkeypatch, tmp_path
):
    path = tmp_path / "unannounced-controller.jsonl"
    hosted_decision_trace(monkeypatch, path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["coverage"]["controller_attempts"] = (
        DIAGNOSTIC_COVERAGE_UNAVAILABLE
    )
    path.write_bytes(
        b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    )

    with pytest.raises(DiagnosticValidationError, match="unannounced"):
        read_diagnostic_sidecar(path)


def test_atomic_recorder_rejects_a_missing_hosted_controller_call(
    monkeypatch, tmp_path
):
    install_responses(monkeypatch, [
        _Response(200, {
            "choices": [{"message": {"content": '{"value":"5"}'}}],
        })
    ])
    env = environment()
    generator_client = configured_client(max_retries=0)
    controller_client = configured_controller_client(max_retries=0)
    collection = collection_for(env, generator_client, controller_client)
    action = Action(Verb.PROPOSE_RESULT, 0)

    with DiagnosticSidecarWriter(
        tmp_path / "missing-controller.jsonl",
        environment=env,
        collection_contract=collection,
        generator_client=generator_client,
        controller_client=controller_client,
    ) as writer:
        generator_client.complete_json(
            "system", "user", schema(), temperature=0.0, max_tokens=64,
            seed=11,
        )
        with pytest.raises(DiagnosticValidationError, match="controller calls"):
            writer.record_decision(
                episode_key="problem:missing",
                turn=0,
                observation_sha256="5" * 64,
                source=BehaviorSource.TEACHER,
                proposed_action=action,
                selected_action=action,
                fallback=False,
            )


def test_nonfallback_receipt_rejects_a_failed_controller_call(
    monkeypatch, tmp_path
):
    install_responses(monkeypatch, [
        _Response(200, {
            "choices": [{"message": {"content": "not json"}}],
        }),
        _Response(200, {
            "choices": [{"message": {"content": '{"value":"5"}'}}],
        }),
    ])
    env = environment()
    generator_client = configured_client(max_retries=0)
    controller_client = configured_controller_client(max_retries=0)
    collection = collection_for(env, generator_client, controller_client)
    action = Action(Verb.PROPOSE_RESULT, 0)

    with DiagnosticSidecarWriter(
        tmp_path / "failed-controller.jsonl",
        environment=env,
        collection_contract=collection,
        generator_client=generator_client,
        controller_client=controller_client,
    ) as writer:
        with pytest.raises(LLMResponseError):
            controller_client.complete_json(
                "system", "user", schema(), temperature=0.0,
                max_tokens=64, seed=19,
            )
        generator_client.complete_json(
            "system", "user", schema(), temperature=0.0, max_tokens=64,
            seed=11,
        )
        with pytest.raises(DiagnosticValidationError, match="not accepted"):
            writer.record_decision(
                episode_key="problem:failed",
                turn=0,
                observation_sha256="6" * 64,
                source=BehaviorSource.TEACHER,
                proposed_action=action,
                selected_action=action,
                fallback=False,
            )


def test_observer_failure_aborts_without_model_retry(monkeypatch):
    calls = install_responses(monkeypatch, [
        _Response(200, {
            "choices": [{"message": {"content": '{"value":"5"}'}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        })
    ])
    client = configured_client(max_retries=4)

    def fail_observer(event):
        raise RuntimeError("diagnostic write failed")

    client.attempt_observer = fail_observer
    with pytest.raises(RuntimeError, match="diagnostic write failed"):
        client.complete_json(
            "system", "user", schema(), temperature=0.0, max_tokens=64,
            seed=11,
        )
    assert len(calls) == 1


def test_context_local_references_do_not_cross_concurrent_calls(
    monkeypatch, tmp_path
):
    import httpx

    class _ConcurrentClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            user = json["messages"][1]["content"]
            return _Response(200, {
                "choices": [{
                    "message": {"content": f'{{"value":"{user}"}}'}
                }],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            })

    monkeypatch.setattr(httpx, "Client", _ConcurrentClient)
    env = environment()
    client = configured_client(max_retries=0)
    collection = collection_for(env, client)
    path = tmp_path / "concurrent-diagnostics.jsonl"
    writer = DiagnosticSidecarWriter(
        path,
        environment=env,
        collection_contract=collection,
        generator_client=client,
    )

    def invoke(value):
        result = client.complete_json(
            "system", value, schema(), temperature=0.0, max_tokens=64,
            seed=11,
        )
        return result, writer.take_diagnostic_ref()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(invoke, "first")
        second = pool.submit(invoke, "second")
        results = [first.result(), second.result()]
    writer.close()

    assert {item[0]["value"] for item in results} == {"first", "second"}
    references = [item[1] for item in results]
    assert references[0] != references[1]
    index = read_diagnostic_sidecar(path)
    assert {
        index.resolve(reference).attempt_records[0]["raw_content"]
        for reference in references
    } == {'{"value":"first"}', '{"value":"second"}'}


def test_back_to_back_calls_queue_references_without_losing_attempts(
    monkeypatch, tmp_path
):
    install_responses(monkeypatch, [
        _Response(200, {
            "choices": [{"message": {"content": '{"value":"1"}'}}],
        }),
        _Response(200, {
            "choices": [{"message": {"content": '{"value":"2"}'}}],
        }),
    ])
    env = environment()
    client = configured_client(max_retries=0)
    collection = collection_for(env, client)
    path = tmp_path / "queued-diagnostics.jsonl"
    with DiagnosticSidecarWriter(
        path,
        environment=env,
        collection_contract=collection,
        generator_client=client,
    ) as writer:
        for value in ("first", "second"):
            client.complete_json(
                "system", value, schema(), temperature=0.0, max_tokens=64,
                seed=11,
            )
        references = [
            writer.take_diagnostic_ref(),
            writer.take_diagnostic_ref(),
        ]

    index = read_diagnostic_sidecar(path)
    assert len(index.calls) == 2
    assert all(index.resolve(reference) for reference in references)


def test_strict_reader_rejects_an_incomplete_call(monkeypatch, tmp_path):
    install_responses(monkeypatch, [
        _Response(200, {
            "choices": [{"message": {"content": '{"value":"5"}'}}],
        })
    ])
    env = environment()
    client = configured_client(max_retries=0)
    collection = collection_for(env, client)
    path = tmp_path / "complete.jsonl"
    with DiagnosticSidecarWriter(
        path,
        environment=env,
        collection_contract=collection,
        generator_client=client,
    ) as writer:
        client.complete_json(
            "system", "user", schema(), temperature=0.0, max_tokens=64,
            seed=11,
        )
        writer.take_diagnostic_ref()

    incomplete = tmp_path / "incomplete.jsonl"
    incomplete.write_bytes(b"".join(path.read_bytes().splitlines(keepends=True)[:-1]))
    with pytest.raises(DiagnosticValidationError, match="incomplete"):
        read_diagnostic_sidecar(incomplete)


def test_index_trust_origin_is_reader_only_and_snapshots_are_detached(
    monkeypatch, tmp_path
):
    with pytest.raises(TypeError):
        DiagnosticSidecarIndex(
            header={}, calls={}, attempts={}, decision_receipts={}
        )

    install_responses(monkeypatch, [
        _Response(200, {
            "choices": [{"message": {"content": '{"value":"5"}'}}],
        })
    ])
    env = environment()
    client = configured_client(max_retries=0)
    collection = collection_for(env, client)
    path = tmp_path / "detached-index.jsonl"
    with DiagnosticSidecarWriter(
        path,
        environment=env,
        collection_contract=collection,
        generator_client=client,
    ) as writer:
        client.complete_json(
            "system", "user", schema(), temperature=0.0, max_tokens=64,
            seed=11,
        )
        reference = writer.take_diagnostic_ref()

    index = read_diagnostic_sidecar(path)
    header = index.header
    header["coverage"]["generator_attempts"] = "forged"
    calls = index.calls
    calls.clear()

    assert index.header["coverage"]["generator_attempts"] != "forged"
    assert index.resolve(reference).call_record["role"] == "generator"
