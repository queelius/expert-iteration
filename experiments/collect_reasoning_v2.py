"""Collect a preregistered hosted reasoning v2 teacher run."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

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
    canonical_json_bytes,
)
from verifier_anchored_rl.reasoning_v2.controller import (
    CONTROLLER_OUTPUT_CONTRACT_VERSION,
    POLICY_INPUT_CONTRACT_VERSION,
    TypedLLMController,
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


_SHA256 = re.compile(r"[0-9a-f]{64}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect one exact v2 teacher run, persist raw role-bound model "
            "diagnostics, and require strict symbolic replay before success."
        ),
        epilog=(
            "Hosted execution is intentionally gated. Before running, add the "
            "experiment manipulation, measure, and reading to "
            "docs/superpowers/research/EXPERIMENTS.md. This command does not "
            "create or amend that preregistration. API keys are read only from "
            "the named environment variables and are never written to artifacts. "
            "The diagnostic sidecar is authoritative for raw attempts and usage; "
            "the final stdout summary also reports role-separated client counters."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--execute-hosted",
        action="store_true",
        help="Acknowledge that this command will spend hosted inference credit.",
    )
    mode.add_argument(
        "--print-contracts",
        action="store_true",
        help=(
            "Resolve and print source/environment/collection digests without "
            "reading API keys or issuing HTTP requests."
        ),
    )
    parser.add_argument(
        "--preregistration-id",
        help="Existing experiment-ledger identifier required for execution.",
    )
    parser.add_argument("--expected-problem-source-sha256")
    parser.add_argument("--expected-environment-contract-sha256")
    parser.add_argument("--expected-collection-contract-sha256")
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument(
        "--start-index",
        type=int,
        required=True,
        help="Zero-based start in source JSONL order, as preregistered.",
    )
    parser.add_argument(
        "--episode-count",
        type=int,
        required=True,
        help="Exact number of consecutive source rows, as preregistered.",
    )
    parser.add_argument("--run-output", type=Path)
    parser.add_argument("--diagnostic-output", type=Path)
    parser.add_argument("--turn-limit", type=int, required=True)

    for role in ("controller", "generator"):
        parser.add_argument(f"--{role}-base-url", required=True)
        parser.add_argument(f"--{role}-provider", required=True)
        parser.add_argument(f"--{role}-model", required=True)
        parser.add_argument(
            f"--{role}-revision",
            required=True,
            help=(
                f"Asserted immutable {role} model revision. Use a provider "
                "commit or immutable deployment identifier."
            ),
        )
        parser.add_argument(
            f"--{role}-api-key-env",
            help=(
                f"Environment variable containing the {role} bearer token; "
                "required only with --execute-hosted."
            ),
        )
        parser.add_argument(f"--{role}-temperature", type=float, required=True)
        parser.add_argument(f"--{role}-max-tokens", type=int, required=True)
        parser.add_argument(f"--{role}-seed", type=int, required=True)
        parser.add_argument(f"--{role}-timeout-s", type=float, required=True)
        parser.add_argument(
            f"--{role}-parse-retries", type=int, required=True
        )
        parser.add_argument(
            f"--{role}-backoff-retries", type=int, required=True
        )
        parser.add_argument(
            f"--{role}-guided-json",
            action=argparse.BooleanOptionalAction,
            required=True,
            help=(
                f"Explicitly enable or disable provider guided JSON for {role}."
            ),
        )
    return parser


def _secret_from_environment(
    parser: argparse.ArgumentParser, variable: str | None, role: str
) -> str:
    if variable is None or not variable.strip():
        parser.error(f"--{role}-api-key-env is required for hosted execution")
    value = os.environ.get(variable)
    if value is None or not value.strip():
        parser.error(
            f"{role} API key environment variable {variable!r} is unset"
        )
    return value


def _hosted_authorization(
    parser: argparse.ArgumentParser,
    *,
    execute_hosted: bool,
    preregistration_id: str | None,
    expected_problem_source_sha256: str | None,
    expected_environment_contract_sha256: str | None,
    expected_collection_contract_sha256: str | None,
) -> HostedExecutionAuthorization:
    if not execute_hosted:
        parser.error(
            "hosted calls are disabled without the explicit --execute-hosted gate"
        )
    if preregistration_id is None or not preregistration_id.strip():
        parser.error("--preregistration-id is required for hosted execution")
    required = {
        "--expected-problem-source-sha256": (
            expected_problem_source_sha256
        ),
        "--expected-environment-contract-sha256": (
            expected_environment_contract_sha256
        ),
        "--expected-collection-contract-sha256": (
            expected_collection_contract_sha256
        ),
    }
    for flag, value in required.items():
        if value is None or _SHA256.fullmatch(value) is None:
            parser.error(f"{flag} requires a lowercase sha256 digest")
    return HostedExecutionAuthorization(
        preregistration_id,
        expected_problem_source_sha256,
        expected_environment_contract_sha256,
        expected_collection_contract_sha256,
    )


def _client(
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    timeout_s: float,
    parse_retries: int,
    backoff_retries: int,
    guided_json: bool,
) -> OpenAICompatibleLLMClient:
    return OpenAICompatibleLLMClient(
        base_url=base_url.rstrip("/"),
        model=model,
        api_key=api_key,
        timeout_s=timeout_s,
        max_retries=parse_retries,
        max_backoff_retries=backoff_retries,
        guided_json=guided_json,
    )


def _client_counters(
    client: OpenAICompatibleLLMClient,
) -> dict[str, int]:
    return {
        "generate_calls": client.generate_calls,
        "prompt_tokens": client.prompt_tokens,
        "generated_tokens": client.generated_tokens,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    full_source = CaseworkProblemSource.from_jsonl(args.problems)
    problem_source = full_source.select_slice(
        args.start_index, args.episode_count
    )
    if args.execute_hosted:
        expected_source = args.expected_problem_source_sha256
        if expected_source is None or _SHA256.fullmatch(expected_source) is None:
            parser.error(
                "--expected-problem-source-sha256 requires a lowercase "
                "sha256 digest"
            )
        if expected_source != problem_source.problem_source_sha256:
            parser.error(
                "selected problem source differs from "
                "--expected-problem-source-sha256"
            )
    adapter = CaseworkMathVerifier()
    environment_contract = build_environment_contract(
        generation_contract=adapter,
        verifier=adapter,
        aggregate_builder=adapter,
        lens_contract=LENS_CONTRACT_VERSION,
        generation_schema=GENERATION_SCHEMA_VERSION,
        generation_temperature=args.generator_temperature,
        generation_max_tokens=args.generator_max_tokens,
        generation_seed_policy="turn-and-retry-offset-v1",
        turn_limit=args.turn_limit,
    )
    environment = BoundEnvironment(
        environment_contract,
        adapter,
        adapter,
        adapter,
    )

    controller_client = _client(
        base_url=args.controller_base_url,
        model=args.controller_model,
        api_key=None,
        timeout_s=args.controller_timeout_s,
        parse_retries=args.controller_parse_retries,
        backoff_retries=args.controller_backoff_retries,
        guided_json=args.controller_guided_json,
    )
    generator_client = _client(
        base_url=args.generator_base_url,
        model=args.generator_model,
        api_key=None,
        timeout_s=args.generator_timeout_s,
        parse_retries=args.generator_parse_retries,
        backoff_retries=args.generator_backoff_retries,
        guided_json=args.generator_guided_json,
    )
    if controller_client is generator_client:
        raise AssertionError("role clients must be distinct objects")

    controller_identity = ModelIdentity(
        args.controller_provider,
        args.controller_model,
        args.controller_revision,
    )
    generator_identity = ModelIdentity(
        args.generator_provider,
        args.generator_model,
        args.generator_revision,
    )
    controller = TypedLLMController(
        environment=environment,
        client=controller_client,
        controller_identity=controller_identity,
        temperature=args.controller_temperature,
        max_tokens=args.controller_max_tokens,
        seed=args.controller_seed,
        seed_policy="turn-and-retry-offset-v1",
        output_contract=CONTROLLER_OUTPUT_CONTRACT_VERSION,
    )
    generator = TypedPolicyDraftGenerator(
        environment=environment,
        client=generator_client,
        generator_identity=generator_identity,
        seed=args.generator_seed,
    )
    collection_contract = CollectionContract(
        generator=generator_identity,
        controller=controller_identity,
        controller_implementation=controller.controller_implementation,
        controller_transport=llm_transport_contract(controller_client),
        controller_seed=args.controller_seed,
        controller_temperature=args.controller_temperature,
        controller_max_tokens=args.controller_max_tokens,
        controller_seed_policy="turn-and-retry-offset-v1",
        controller_output_contract=CONTROLLER_OUTPUT_CONTRACT_VERSION,
        behavior_source="teacher",
        controller_fallback_policy="first-legal-pair-v1",
        generator_implementation=generator.generator_implementation,
        generator_transport=llm_transport_contract(generator_client),
        generation_seed=args.generator_seed,
        policy_input_contract=POLICY_INPUT_CONTRACT_VERSION,
        problem_source_contract=problem_source.problem_source_contract,
        problem_source_sha256=problem_source.problem_source_sha256,
        problem_adapter_implementation=(
            problem_source.problem_adapter_implementation
        ),
    )
    contract_summary = {
        "schema_version": "reasoning-v2-contract-preparation-v1",
        "problem_source_sha256": problem_source.problem_source_sha256,
        "environment_contract_sha256": environment.sha256,
        "collection_contract_sha256": collection_contract.sha256,
        "episode_count": len(problem_source),
        "problem_ids": list(problem_source.problem_ids),
    }
    if args.print_contracts:
        print(canonical_json_bytes(contract_summary).decode("ascii"))
        return 0

    authorization = _hosted_authorization(
        parser,
        execute_hosted=args.execute_hosted,
        preregistration_id=args.preregistration_id,
        expected_problem_source_sha256=(
            args.expected_problem_source_sha256
        ),
        expected_environment_contract_sha256=(
            args.expected_environment_contract_sha256
        ),
        expected_collection_contract_sha256=(
            args.expected_collection_contract_sha256
        ),
    )
    try:
        authorization.validate_bindings(
            problem_source_sha256=problem_source.problem_source_sha256,
            environment_contract_sha256=environment.sha256,
            collection_contract_sha256=collection_contract.sha256,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.run_output is None:
        parser.error("--run-output is required for hosted execution")
    if args.diagnostic_output is None:
        parser.error("--diagnostic-output is required for hosted execution")
    controller_client.api_key = _secret_from_environment(
        parser, args.controller_api_key_env, "controller"
    )
    generator_client.api_key = _secret_from_environment(
        parser, args.generator_api_key_env, "generator"
    )
    result = collect_and_strict_replay(
        run_path=args.run_output,
        diagnostic_path=args.diagnostic_output,
        environment=environment,
        collection_contract=collection_contract,
        problem_source=problem_source,
        controller=controller,
        generator=generator,
        authorization=authorization,
    )
    summary = {
        "schema_version": "reasoning-v2-collection-summary-v1",
        "preregistration_id": args.preregistration_id,
        "environment_contract_sha256": environment.sha256,
        "collection_contract_sha256": collection_contract.sha256,
        "problem_source_sha256": problem_source.problem_source_sha256,
        "episode_count": result.episode_count,
        "decision_count": result.decision_count,
        "solved_count": result.solved_count,
        "client_counters": {
            "controller": _client_counters(controller_client),
            "generator": _client_counters(generator_client),
        },
        "run_path": str(result.run_path),
        "diagnostic_path": str(result.diagnostic_path),
    }
    print(canonical_json_bytes(summary).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
