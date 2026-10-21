import json
import threading

import pytest

import experiments.collect_reasoning_v2 as collection_cli
import verifier_anchored_rl.reasoning_v2.collection as collection_module
from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient
from verifier_anchored_rl.reasoning_v2.casework_source import (
    CaseworkProblemSource,
)
from verifier_anchored_rl.reasoning_v2.collection import (
    HostedExecutionAuthorization,
    collect_and_strict_replay,
)
from verifier_anchored_rl.reasoning_v2.contracts import (
    CollectionContract,
    ModelIdentity,
    build_environment_contract,
)
from verifier_anchored_rl.reasoning_v2.controller import (
    CONTROLLER_OUTPUT_CONTRACT_VERSION,
    POLICY_INPUT_CONTRACT_VERSION,
    TypedLLMController,
)
from verifier_anchored_rl.reasoning_v2.diagnostics import (
    read_diagnostic_sidecar,
)
from verifier_anchored_rl.reasoning_v2.lenses import (
    GENERATION_SCHEMA_VERSION,
    LENS_CONTRACT_VERSION,
    TypedPolicyDraftGenerator,
    llm_transport_contract,
)
from verifier_anchored_rl.reasoning_v2.math_adapter import (
    CaseworkMathVerifier,
)
from verifier_anchored_rl.reasoning_v2.runtime import BoundEnvironment


class _Response:
    status_code = 200
    headers = {}

    def __init__(self, payload):
        self._payload = payload
        self.text = json.dumps(payload, separators=(",", ":"))

    def raise_for_status(self):
        return None

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
                return queue.pop(0)

    monkeypatch.setattr(httpx, "Client", _Client)
    return calls


def problem_row(problem_id="cw-collect-1"):
    return {
        "id": problem_id,
        "problem": "Solve |x - 5| = 0 and report the largest solution.",
        "answer": "5",
        "solution_set": ["5"],
        "symbolic": "Eq(Abs(x - 5), 0)",
        "question_type": "largest_solution",
        "family": "f1a",
        "difficulty": {"tier": 1, "params": {}},
    }


def source(tmp_path):
    path = tmp_path / "problems.jsonl"
    path.write_text(
        json.dumps(problem_row(), separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="",
    )
    return CaseworkProblemSource.from_jsonl(path)


def bindings(problem_source, *, turn_limit=2):
    adapter = CaseworkMathVerifier()
    environment = BoundEnvironment(
        build_environment_contract(
            generation_contract=adapter,
            verifier=adapter,
            aggregate_builder=adapter,
            lens_contract=LENS_CONTRACT_VERSION,
            generation_schema=GENERATION_SCHEMA_VERSION,
            generation_temperature=0.0,
            generation_max_tokens=64,
            generation_seed_policy="turn-and-retry-offset-v1",
            turn_limit=turn_limit,
        ),
        adapter,
        adapter,
        adapter,
    )
    controller_client = OpenAICompatibleLLMClient(
        base_url="https://controller.example/v1",
        model="teacher-model",
        api_key="CONTROLLER_SECRET",
        timeout_s=30.0,
        max_retries=0,
        max_backoff_retries=0,
    )
    generator_client = OpenAICompatibleLLMClient(
        base_url="https://generator.example/v1",
        model="generator-model",
        api_key="GENERATOR_SECRET",
        timeout_s=30.0,
        max_retries=0,
        max_backoff_retries=0,
    )
    controller_identity = ModelIdentity(
        "test", "teacher-model", "teacher-revision-1"
    )
    generator_identity = ModelIdentity(
        "test", "generator-model", "generator-revision-1"
    )
    controller = TypedLLMController(
        environment,
        controller_client,
        controller_identity,
        temperature=0.0,
        max_tokens=64,
        seed=19,
        seed_policy="turn-and-retry-offset-v1",
    )
    generator = TypedPolicyDraftGenerator(
        environment,
        generator_client,
        generator_identity,
        seed=11,
    )
    collection = CollectionContract(
        generator=generator_identity,
        controller=controller_identity,
        controller_implementation=controller.controller_implementation,
        controller_transport=llm_transport_contract(controller_client),
        controller_seed=19,
        controller_temperature=0.0,
        controller_max_tokens=64,
        controller_seed_policy="turn-and-retry-offset-v1",
        controller_output_contract=CONTROLLER_OUTPUT_CONTRACT_VERSION,
        behavior_source="teacher",
        controller_fallback_policy="first-legal-pair-v1",
        generator_implementation=generator.generator_implementation,
        generator_transport=llm_transport_contract(generator_client),
        generation_seed=11,
        policy_input_contract=POLICY_INPUT_CONTRACT_VERSION,
        problem_source_contract=problem_source.problem_source_contract,
        problem_source_sha256=problem_source.problem_source_sha256,
        problem_adapter_implementation=(
            problem_source.problem_adapter_implementation
        ),
    )
    return environment, collection, controller, generator


def authorization_for(
    problem_source, environment, collection, preregistration_id="E-test"
):
    return HostedExecutionAuthorization(
        preregistration_id,
        problem_source.problem_source_sha256,
        environment.sha256,
        collection.sha256,
    )


def cli_contract_args(problem_path):
    return [
        "--problems", str(problem_path),
        "--start-index", "0",
        "--episode-count", "1",
        "--turn-limit", "2",
        "--controller-base-url", "https://controller.example/v1",
        "--controller-provider", "test",
        "--controller-model", "teacher-model",
        "--controller-revision", "teacher-revision-1",
        "--controller-temperature", "0.0",
        "--controller-max-tokens", "64",
        "--controller-seed", "19",
        "--controller-timeout-s", "30.0",
        "--controller-parse-retries", "0",
        "--controller-backoff-retries", "0",
        "--no-controller-guided-json",
        "--generator-base-url", "https://generator.example/v1",
        "--generator-provider", "test",
        "--generator-model", "generator-model",
        "--generator-revision", "generator-revision-1",
        "--generator-temperature", "0.0",
        "--generator-max-tokens", "64",
        "--generator-seed", "11",
        "--generator-timeout-s", "30.0",
        "--generator-parse-retries", "0",
        "--generator-backoff-retries", "0",
        "--no-generator-guided-json",
    ]


def test_mocked_hosted_collection_writes_and_strict_replays_one_episode(
    monkeypatch, tmp_path
):
    calls = install_responses(monkeypatch, [
        _Response({
            "choices": [{
                "message": {
                    "content": (
                        '{"verb":"propose_result","target":0}'
                    )
                }
            }]
        }),
        _Response({
            "choices": [{"message": {"content": '{"value":"5"}'}}]
        }),
    ])
    problem_source = source(tmp_path)
    environment, collection, controller, generator = bindings(problem_source)
    run_path = tmp_path / "run.jsonl"
    diagnostic_path = tmp_path / "diagnostics.jsonl"

    result = collect_and_strict_replay(
        run_path=run_path,
        diagnostic_path=diagnostic_path,
        environment=environment,
        collection_contract=collection,
        problem_source=problem_source,
        controller=controller,
        generator=generator,
        authorization=authorization_for(
            problem_source,
            environment,
            collection,
            "E-test-collection",
        ),
    )

    assert result.episode_count == 1
    assert result.decision_count == 1
    assert result.solved_count == 1
    decision = result.replayed_run.decisions[0]
    assert decision.transition.reward == 1
    assert decision.transition.terminal is True
    assert decision.behavior.bc_eligible is True
    assert decision.diagnostic_ref is not None
    assert [call["url"] for call in calls] == [
        "https://controller.example/v1/chat/completions",
        "https://generator.example/v1/chat/completions",
    ]
    assert calls[0]["headers"]["Authorization"] == (
        "Bearer CONTROLLER_SECRET"
    )
    assert calls[1]["headers"]["Authorization"] == (
        "Bearer GENERATOR_SECRET"
    )
    assert "CONTROLLER_SECRET" not in diagnostic_path.read_text("ascii")
    assert "GENERATOR_SECRET" not in diagnostic_path.read_text("ascii")
    assert not (tmp_path / "run.jsonl.partial").exists()
    assert not (tmp_path / "diagnostics.jsonl.partial").exists()
    index = read_diagnostic_sidecar(
        diagnostic_path,
        environment_contract_sha256=environment.sha256,
        collection_contract_sha256=collection.sha256,
    )
    assert index.header["coverage"]["controller_attempts"] == (
        "openai-compatible-http-attempts-v2"
    )


def test_multi_episode_run_finalizes_only_after_full_ordered_replay(
    monkeypatch, tmp_path
):
    responses = []
    for _ in range(2):
        responses.extend([
            _Response({
                "choices": [{
                    "message": {
                        "content": (
                            '{"verb":"propose_result","target":0}'
                        )
                    }
                }]
            }),
            _Response({
                "choices": [{
                    "message": {"content": '{"value":"5"}'}
                }]
            }),
        ])
    install_responses(monkeypatch, responses)
    problem_path = tmp_path / "two-problems.jsonl"
    problem_path.write_text(
        "".join(
            json.dumps(problem_row(problem_id), separators=(",", ":")) + "\n"
            for problem_id in ("cw-first", "cw-second")
        ),
        encoding="utf-8",
        newline="",
    )
    problem_source = CaseworkProblemSource.from_jsonl(problem_path)
    environment, collection, controller, generator = bindings(problem_source)
    run_path = tmp_path / "multi-run.jsonl"
    diagnostic_path = tmp_path / "multi-diagnostics.jsonl"

    result = collect_and_strict_replay(
        run_path=run_path,
        diagnostic_path=diagnostic_path,
        environment=environment,
        collection_contract=collection,
        problem_source=problem_source,
        controller=controller,
        generator=generator,
        authorization=authorization_for(
            problem_source, environment, collection
        ),
    )

    assert result.episode_count == 2
    assert tuple(
        episode.problem_id for episode in result.replayed_run.episodes
    ) == ("cw-first", "cw-second")
    assert run_path.exists()
    assert diagnostic_path.exists()
    assert not (tmp_path / "multi-run.jsonl.partial").exists()
    assert not (tmp_path / "multi-diagnostics.jsonl.partial").exists()


def test_interrupted_prefix_remains_partial_and_never_appears_final(
    monkeypatch, tmp_path
):
    install_responses(monkeypatch, [
        _Response({
            "choices": [{
                "message": {
                    "content": '{"verb":"propose_result","target":0}'
                }
            }]
        }),
        _Response({
            "choices": [{"message": {"content": '{"value":"5"}'}}]
        }),
    ])
    problem_path = tmp_path / "interrupt-problems.jsonl"
    problem_path.write_text(
        "".join(
            json.dumps(problem_row(problem_id), separators=(",", ":")) + "\n"
            for problem_id in ("cw-first", "cw-second")
        ),
        encoding="utf-8",
        newline="",
    )
    problem_source = CaseworkProblemSource.from_jsonl(problem_path)
    environment, collection, controller, generator = bindings(problem_source)
    original_step = collection_module.EpisodeRuntime.step
    step_count = 0

    def interrupt_second_episode(runtime, action=None):
        nonlocal step_count
        step_count += 1
        if step_count == 2:
            raise KeyboardInterrupt("test interruption")
        return original_step(runtime, action)

    monkeypatch.setattr(
        collection_module.EpisodeRuntime,
        "step",
        interrupt_second_episode,
    )
    run_path = tmp_path / "interrupted-run.jsonl"
    diagnostic_path = tmp_path / "interrupted-diagnostics.jsonl"

    with pytest.raises(KeyboardInterrupt, match="test interruption"):
        collect_and_strict_replay(
            run_path=run_path,
            diagnostic_path=diagnostic_path,
            environment=environment,
            collection_contract=collection,
            problem_source=problem_source,
            controller=controller,
            generator=generator,
            authorization=authorization_for(
                problem_source, environment, collection
            ),
        )

    assert not run_path.exists()
    assert not diagnostic_path.exists()
    assert (tmp_path / "interrupted-run.jsonl.partial").exists()
    assert (tmp_path / "interrupted-diagnostics.jsonl.partial").exists()


def test_authorization_digest_mismatch_precedes_files_and_http(
    monkeypatch, tmp_path
):
    calls = install_responses(monkeypatch, [])
    problem_source = source(tmp_path)
    environment, collection, controller, generator = bindings(problem_source)
    authorization = HostedExecutionAuthorization(
        "E-wrong-source",
        "f" * 64,
        environment.sha256,
        collection.sha256,
    )
    run_path = tmp_path / "mismatch-run.jsonl"
    diagnostic_path = tmp_path / "mismatch-diagnostics.jsonl"

    with pytest.raises(ValueError, match="problem source digest"):
        collect_and_strict_replay(
            run_path=run_path,
            diagnostic_path=diagnostic_path,
            environment=environment,
            collection_contract=collection,
            problem_source=problem_source,
            controller=controller,
            generator=generator,
            authorization=authorization,
        )

    assert calls == []
    assert not run_path.exists()
    assert not diagnostic_path.exists()
    assert not (tmp_path / "mismatch-run.jsonl.partial").exists()
    assert not (tmp_path / "mismatch-diagnostics.jsonl.partial").exists()


def test_no_clobber_finalization_rolls_back_first_link(
    monkeypatch, tmp_path
):
    install_responses(monkeypatch, [
        _Response({
            "choices": [{
                "message": {
                    "content": '{"verb":"propose_result","target":0}'
                }
            }]
        }),
        _Response({
            "choices": [{"message": {"content": '{"value":"5"}'}}]
        }),
    ])
    problem_source = source(tmp_path)
    environment, collection, controller, generator = bindings(problem_source)
    real_link = collection_module.os.link
    link_count = 0

    def fail_second_link(source_path, final_path):
        nonlocal link_count
        link_count += 1
        if link_count == 2:
            raise FileExistsError("simulated no-clobber race")
        return real_link(source_path, final_path)

    monkeypatch.setattr(collection_module.os, "link", fail_second_link)
    run_path = tmp_path / "race-run.jsonl"
    diagnostic_path = tmp_path / "race-diagnostics.jsonl"

    with pytest.raises(FileExistsError, match="no-clobber race"):
        collect_and_strict_replay(
            run_path=run_path,
            diagnostic_path=diagnostic_path,
            environment=environment,
            collection_contract=collection,
            problem_source=problem_source,
            controller=controller,
            generator=generator,
            authorization=authorization_for(
                problem_source, environment, collection
            ),
        )

    assert not run_path.exists()
    assert not diagnostic_path.exists()
    assert (tmp_path / "race-run.jsonl.partial").exists()
    assert (tmp_path / "race-diagnostics.jsonl.partial").exists()


def test_controller_failure_falls_back_and_replays_with_raw_receipt(
    monkeypatch, tmp_path
):
    install_responses(monkeypatch, [
        _Response({
            "choices": [{"message": {"content": "not-json"}}]
        }),
        _Response({
            "choices": [{
                "message": {
                    "content": '{"domains":["x < 5","x >= 5"]}'
                }
            }]
        }),
    ])
    problem_source = source(tmp_path)
    environment, collection, controller, generator = bindings(
        problem_source, turn_limit=1
    )

    result = collect_and_strict_replay(
        run_path=tmp_path / "fallback-run.jsonl",
        diagnostic_path=tmp_path / "fallback-diagnostics.jsonl",
        environment=environment,
        collection_contract=collection,
        problem_source=problem_source,
        controller=controller,
        generator=generator,
        authorization=authorization_for(
            problem_source, environment, collection
        ),
    )

    decision = result.replayed_run.decisions[0]
    assert decision.behavior.fallback is True
    assert decision.behavior.bc_eligible is False
    assert decision.transition.action.verb.value == "decompose"
    assert decision.transition.reward == 0


def test_generator_payload_failure_is_persisted_and_strict_replayed(
    monkeypatch, tmp_path
):
    install_responses(monkeypatch, [
        _Response({
            "choices": [{
                "message": {
                    "content": '{"verb":"propose_result","target":0}'
                }
            }]
        }),
        _Response({
            "choices": [{"message": {"content": "not-json"}}]
        }),
    ])
    problem_source = source(tmp_path)
    environment, collection, controller, generator = bindings(
        problem_source, turn_limit=1
    )

    result = collect_and_strict_replay(
        run_path=tmp_path / "failure-run.jsonl",
        diagnostic_path=tmp_path / "failure-diagnostics.jsonl",
        environment=environment,
        collection_contract=collection,
        problem_source=problem_source,
        controller=controller,
        generator=generator,
        authorization=authorization_for(
            problem_source, environment, collection
        ),
    )

    decision = result.replayed_run.decisions[0]
    assert decision.behavior.fallback is False
    assert decision.behavior.bc_eligible is True
    assert decision.transition.outcome.reason.value == "payload_error"
    assert decision.transition.policy_draft is None
    assert decision.transition.reward == 0


def test_collection_requires_explicit_hosted_authorization(tmp_path):
    problem_source = source(tmp_path)
    environment, collection, controller, generator = bindings(problem_source)

    with pytest.raises(TypeError, match="explicit authorization"):
        collect_and_strict_replay(
            run_path=tmp_path / "run.jsonl",
            diagnostic_path=tmp_path / "diagnostics.jsonl",
            environment=environment,
            collection_contract=collection,
            problem_source=problem_source,
            controller=controller,
            generator=generator,
            authorization=None,
        )


def test_collection_rejects_an_authorization_subclass(tmp_path):
    problem_source = source(tmp_path)
    environment, collection, controller, generator = bindings(problem_source)

    class BypassAuthorization(HostedExecutionAuthorization):
        def validate_bindings(self, **kwargs):
            return None

    authorization = BypassAuthorization(
        "E-test-subclass",
        "0" * 64,
        "0" * 64,
        "0" * 64,
    )
    with pytest.raises(TypeError, match="explicit authorization"):
        collect_and_strict_replay(
            run_path=tmp_path / "subclass-run.jsonl",
            diagnostic_path=tmp_path / "subclass-diagnostics.jsonl",
            environment=environment,
            collection_contract=collection,
            problem_source=problem_source,
            controller=controller,
            generator=generator,
            authorization=authorization,
        )


def test_authorization_rejects_unstable_identifier_text():
    with pytest.raises(ValueError, match="safe identifier"):
        HostedExecutionAuthorization(
            "not preregistered / unsafe",
            "1" * 64,
            "2" * 64,
            "3" * 64,
        )


def test_cli_hosted_gate_is_explicit():
    parser = collection_cli._parser()

    with pytest.raises(SystemExit) as error:
        collection_cli._hosted_authorization(
            parser,
            execute_hosted=False,
            preregistration_id="E23",
            expected_problem_source_sha256="1" * 64,
            expected_environment_contract_sha256="2" * 64,
            expected_collection_contract_sha256="3" * 64,
        )

    assert error.value.code == 2
    assert collection_cli._hosted_authorization(
        parser,
        execute_hosted=True,
        preregistration_id="E23",
        expected_problem_source_sha256="1" * 64,
        expected_environment_contract_sha256="2" * 64,
        expected_collection_contract_sha256="3" * 64,
    ) == HostedExecutionAuthorization(
        "E23", "1" * 64, "2" * 64, "3" * 64
    )


def test_cli_accepts_secrets_only_by_environment_variable_name(
    monkeypatch,
):
    parser = collection_cli._parser()
    destinations = {action.dest for action in parser._actions}
    assert "controller_api_key" not in destinations
    assert "generator_api_key" not in destinations
    assert "controller_api_key_env" in destinations
    assert "generator_api_key_env" in destinations

    monkeypatch.setenv("TEST_CONTROLLER_TOKEN", "secret-from-env")
    assert collection_cli._secret_from_environment(
        parser, "TEST_CONTROLLER_TOKEN", "controller"
    ) == "secret-from-env"


def test_cli_print_contracts_reads_no_keys_and_makes_no_http_calls(
    monkeypatch, tmp_path, capsys
):
    calls = install_responses(monkeypatch, [])
    problem_path = tmp_path / "prepare-problems.jsonl"
    problem_path.write_text(
        json.dumps(problem_row(), separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="",
    )
    monkeypatch.delenv("UNUSED_CONTROLLER_TOKEN", raising=False)
    monkeypatch.delenv("UNUSED_GENERATOR_TOKEN", raising=False)

    exit_code = collection_cli.main([
        "--print-contracts",
        *cli_contract_args(problem_path),
    ])

    assert exit_code == 0
    assert calls == []
    summary = json.loads(capsys.readouterr().out)
    prepared_source = CaseworkProblemSource.from_jsonl(problem_path)
    prepared_environment, prepared_collection, _, _ = bindings(
        prepared_source
    )
    assert summary == {
        "schema_version": "reasoning-v2-contract-preparation-v1",
        "problem_source_sha256": prepared_source.problem_source_sha256,
        "environment_contract_sha256": prepared_environment.sha256,
        "collection_contract_sha256": prepared_collection.sha256,
        "episode_count": 1,
        "problem_ids": ["cw-collect-1"],
    }


@pytest.mark.parametrize(
    ("mismatch", "expected_client_constructions"),
    (("source", 0), ("environment", 2), ("collection", 2)),
)
def test_cli_digest_mismatch_precedes_keys_and_http(
    monkeypatch,
    tmp_path,
    mismatch,
    expected_client_constructions,
):
    calls = install_responses(monkeypatch, [])
    problem_path = tmp_path / "mismatch-problems.jsonl"
    problem_path.write_text(
        json.dumps(problem_row(), separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="",
    )
    prepared_source = CaseworkProblemSource.from_jsonl(problem_path)
    prepared_environment, prepared_collection, _, _ = bindings(
        prepared_source
    )
    expected = {
        "source": prepared_source.problem_source_sha256,
        "environment": prepared_environment.sha256,
        "collection": prepared_collection.sha256,
    }
    expected[mismatch] = "f" * 64
    client_constructions = 0
    real_client = collection_cli._client

    def count_client_construction(**kwargs):
        nonlocal client_constructions
        client_constructions += 1
        return real_client(**kwargs)

    monkeypatch.setattr(
        collection_cli, "_client", count_client_construction
    )
    monkeypatch.delenv("MISSING_CONTROLLER_TOKEN", raising=False)
    monkeypatch.delenv("MISSING_GENERATOR_TOKEN", raising=False)

    with pytest.raises(SystemExit) as error:
        collection_cli.main([
            "--execute-hosted",
            "--preregistration-id", "E-mismatch-test",
            "--expected-problem-source-sha256", expected["source"],
            "--expected-environment-contract-sha256", expected["environment"],
            "--expected-collection-contract-sha256", expected["collection"],
            "--controller-api-key-env", "MISSING_CONTROLLER_TOKEN",
            "--generator-api-key-env", "MISSING_GENERATOR_TOKEN",
            "--run-output", str(tmp_path / "mismatch-run.jsonl"),
            "--diagnostic-output", str(
                tmp_path / "mismatch-diagnostics.jsonl"
            ),
            *cli_contract_args(problem_path),
        ])

    assert error.value.code == 2
    assert calls == []
    assert client_constructions == expected_client_constructions
    assert not (tmp_path / "mismatch-run.jsonl").exists()
    assert not (tmp_path / "mismatch-diagnostics.jsonl").exists()


def test_cli_counter_projection_is_role_ready():
    client = OpenAICompatibleLLMClient(
        base_url="https://provider.example/v1",
        model="model",
    )
    client.generate_calls = 3
    client.prompt_tokens = 17
    client.generated_tokens = 9

    assert collection_cli._client_counters(client) == {
        "generate_calls": 3,
        "prompt_tokens": 17,
        "generated_tokens": 9,
    }


def test_cli_composes_mocked_collection_and_reports_role_counters(
    monkeypatch, tmp_path, capsys
):
    install_responses(monkeypatch, [
        _Response({
            "choices": [{
                "message": {
                    "content": (
                        '{"verb":"propose_result","target":0}'
                    )
                }
            }]
        }),
        _Response({
            "choices": [{"message": {"content": '{"value":"5"}'}}]
        }),
    ])
    problem_path = tmp_path / "cli-problems.jsonl"
    problem_path.write_text(
        json.dumps(problem_row(), separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="",
    )
    run_path = tmp_path / "cli-run.jsonl"
    diagnostic_path = tmp_path / "cli-diagnostics.jsonl"
    prepared_source = CaseworkProblemSource.from_jsonl(problem_path)
    prepared_environment, prepared_collection, _, _ = bindings(
        prepared_source
    )
    monkeypatch.setenv("TEST_CONTROLLER_TOKEN", "controller-secret")
    monkeypatch.setenv("TEST_GENERATOR_TOKEN", "generator-secret")

    exit_code = collection_cli.main([
        "--execute-hosted",
        "--preregistration-id", "E-cli-test",
        "--expected-problem-source-sha256",
        prepared_source.problem_source_sha256,
        "--expected-environment-contract-sha256",
        prepared_environment.sha256,
        "--expected-collection-contract-sha256",
        prepared_collection.sha256,
        "--problems", str(problem_path),
        "--start-index", "0",
        "--episode-count", "1",
        "--run-output", str(run_path),
        "--diagnostic-output", str(diagnostic_path),
        "--turn-limit", "2",
        "--controller-base-url", "https://controller.example/v1",
        "--controller-provider", "test",
        "--controller-model", "teacher-model",
        "--controller-revision", "teacher-revision-1",
        "--controller-api-key-env", "TEST_CONTROLLER_TOKEN",
        "--controller-temperature", "0.0",
        "--controller-max-tokens", "64",
        "--controller-seed", "19",
        "--controller-timeout-s", "30.0",
        "--controller-parse-retries", "0",
        "--controller-backoff-retries", "0",
        "--no-controller-guided-json",
        "--generator-base-url", "https://generator.example/v1",
        "--generator-provider", "test",
        "--generator-model", "generator-model",
        "--generator-revision", "generator-revision-1",
        "--generator-api-key-env", "TEST_GENERATOR_TOKEN",
        "--generator-temperature", "0.0",
        "--generator-max-tokens", "64",
        "--generator-seed", "11",
        "--generator-timeout-s", "30.0",
        "--generator-parse-retries", "0",
        "--generator-backoff-retries", "0",
        "--no-generator-guided-json",
    ])

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["episode_count"] == 1
    assert summary["decision_count"] == 1
    assert summary["solved_count"] == 1
    assert summary["client_counters"] == {
        "controller": {
            "generate_calls": 1,
            "prompt_tokens": 0,
            "generated_tokens": 0,
        },
        "generator": {
            "generate_calls": 1,
            "prompt_tokens": 0,
            "generated_tokens": 0,
        },
    }
    assert run_path.exists()
    assert diagnostic_path.exists()
