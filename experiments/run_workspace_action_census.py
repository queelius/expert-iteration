"""Prepare or execute the complete-answer semantic action census."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient
from verifier_anchored_rl.workspace_v1.action_census import (
    BUILDER_MODES,
    audit_action_census_artifacts,
    build_action_census_contract,
    collect_action_census_run,
    load_census_problems,
    prepare_base_replay,
)
from verifier_anchored_rl.workspace_v1.experiment import ModelIdentity
from verifier_anchored_rl.workspace_v1.records import canonical_json_bytes

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _add_role_arguments(parser: argparse.ArgumentParser, role: str) -> None:
    parser.add_argument(f"--{role}-provider", required=True)
    parser.add_argument(f"--{role}-model", required=True)
    parser.add_argument(f"--{role}-revision", required=True)
    parser.add_argument(f"--{role}-base-url", required=True)
    parser.add_argument(f"--{role}-api-key-env")
    parser.add_argument(
        f"--{role}-guided-json",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
    parser.add_argument(f"--{role}-timeout-s", type=float, required=True)
    parser.add_argument(f"--{role}-parse-retries", type=int, required=True)
    parser.add_argument(f"--{role}-backoff-retries", type=int, required=True)
    parser.add_argument(f"--{role}-temperature", type=float, required=True)
    parser.add_argument(f"--{role}-max-tokens", type=int, required=True)
    parser.add_argument(f"--{role}-seed", type=int, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-contracts", action="store_true")
    mode.add_argument("--execute-hosted", action="store_true")
    parser.add_argument("--preregistration-id", required=True)
    parser.add_argument(
        "--stage",
        choices=("integration_smoke", "pilot", "full"),
        required=True,
    )
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--case-count", type=int, required=True)
    parser.add_argument("--action-order-seed", type=int, required=True)
    parser.add_argument(
        "--builder-mode",
        choices=BUILDER_MODES,
        default=BUILDER_MODES[0],
    )
    parser.add_argument("--run-output", type=Path)
    parser.add_argument("--diagnostic-output", type=Path)
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--expected-problem-source-sha256")
    parser.add_argument("--expected-evaluation-binding-sha256")
    parser.add_argument("--expected-run-contract-sha256")
    parser.add_argument("--base-replay-run", type=Path)
    parser.add_argument("--base-replay-diagnostic", type=Path)
    parser.add_argument("--base-replay-run-sha256")
    parser.add_argument("--base-replay-diagnostic-sha256")
    parser.add_argument("--base-replay-contract-sha256")
    parser.add_argument("--expected-base-replay-projection-sha256")
    _add_role_arguments(parser, "worker")
    _add_role_arguments(parser, "builder")
    return parser


def _identity(args: argparse.Namespace, role: str) -> ModelIdentity:
    return ModelIdentity(
        provider=getattr(args, f"{role}_provider"),
        model=getattr(args, f"{role}_model"),
        revision=getattr(args, f"{role}_revision"),
        base_url=getattr(args, f"{role}_base_url"),
        guided_json=getattr(args, f"{role}_guided_json"),
        timeout_s=getattr(args, f"{role}_timeout_s"),
        parse_retries=getattr(args, f"{role}_parse_retries"),
        backoff_retries=getattr(args, f"{role}_backoff_retries"),
    )


def _client(identity: ModelIdentity, api_key: str) -> OpenAICompatibleLLMClient:
    return OpenAICompatibleLLMClient(
        base_url=identity.base_url.rstrip("/"),
        model=identity.model,
        api_key=api_key,
        timeout_s=identity.timeout_s,
        max_retries=identity.parse_retries,
        max_backoff_retries=identity.backoff_retries,
        guided_json=identity.guided_json,
    )


def _secret(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    role: str,
) -> str:
    variable = getattr(args, f"{role}_api_key_env")
    if not isinstance(variable, str) or not variable.strip():
        parser.error(f"--{role}-api-key-env is required for execution")
    value = os.environ.get(variable)
    if value is None or not value.strip():
        parser.error(f"{role} API key environment variable is unset")
    return value


def _require_digest(
    parser: argparse.ArgumentParser,
    flag: str,
    supplied: str | None,
    actual: str,
) -> None:
    if supplied is None or _SHA256.fullmatch(supplied) is None:
        parser.error(f"{flag} requires a lowercase sha256")
    if supplied != actual:
        parser.error(f"{flag} differs from the prepared contract")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.start_index < 0 or args.case_count <= 0:
        parser.error("start index and case count must select a positive slice")
    try:
        all_problems = load_census_problems(args.problems)
        problems = all_problems[args.start_index : args.start_index + args.case_count]
        if len(problems) != args.case_count:
            raise ValueError("selected action census slice exceeds the source")
        worker = _identity(args, "worker")
        builder = _identity(args, "builder")
        base_replay_args = (
            args.base_replay_run,
            args.base_replay_diagnostic,
            args.base_replay_run_sha256,
            args.base_replay_diagnostic_sha256,
            args.base_replay_contract_sha256,
        )
        if any(value is not None for value in base_replay_args) and any(
            value is None for value in base_replay_args
        ):
            raise ValueError("base replay arguments must be supplied together")
        if (
            args.base_replay_run is None
            and args.expected_base_replay_projection_sha256 is not None
        ):
            raise ValueError(
                "base replay projection requires the complete replay source"
            )
        base_replay = None
        if args.base_replay_run is not None:
            assert args.base_replay_diagnostic is not None
            assert args.base_replay_run_sha256 is not None
            assert args.base_replay_diagnostic_sha256 is not None
            assert args.base_replay_contract_sha256 is not None
            base_replay = prepare_base_replay(
                source_run_path=args.base_replay_run,
                source_diagnostic_path=args.base_replay_diagnostic,
                expected_source_run_sha256=args.base_replay_run_sha256,
                expected_source_diagnostic_sha256=(args.base_replay_diagnostic_sha256),
                expected_source_run_contract_sha256=(args.base_replay_contract_sha256),
                problems=problems,
                worker=worker,
                worker_temperature=args.worker_temperature,
                worker_max_tokens=args.worker_max_tokens,
                worker_seed=args.worker_seed,
            )
        contract = build_action_census_contract(
            preregistration_id=args.preregistration_id,
            stage=args.stage,
            dataset_path=args.problems,
            problems=problems,
            worker=worker,
            state_builder=builder,
            worker_temperature=args.worker_temperature,
            worker_max_tokens=args.worker_max_tokens,
            worker_seed=args.worker_seed,
            builder_temperature=args.builder_temperature,
            builder_max_tokens=args.builder_max_tokens,
            builder_seed=args.builder_seed,
            action_order_seed=args.action_order_seed,
            builder_mode=args.builder_mode,
            base_replay=base_replay,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))

    contract_header = contract.header()
    preparation = {
        "schema_version": "action-census-preparation-v2",
        "preregistration_id": contract.preregistration_id,
        "stage": contract.stage,
        "dataset_sha256": contract.dataset_sha256,
        "problem_source_sha256": contract.problem_source_sha256,
        "evaluation_binding_sha256": contract.evaluation_binding_sha256,
        "case_count": len(problems),
        "problem_ids": list(contract.problem_ids),
        "run_contract_sha256": contract.sha256,
        "prompt_bindings": contract_header["prompt_bindings"],
        "implementation_binding": contract_header["implementation_binding"],
        "settings": contract_header["settings"],
        "models": contract_header["models"],
        "anchored_span_catalog": contract_header.get("anchored_span_catalog"),
        "base_replay": (
            None if base_replay is None else base_replay.binding.to_record()
        ),
    }
    if args.print_contracts:
        print(canonical_json_bytes(preparation).decode("ascii"))
        return 0

    for flag, supplied, actual in (
        (
            "--expected-dataset-sha256",
            args.expected_dataset_sha256,
            contract.dataset_sha256,
        ),
        (
            "--expected-problem-source-sha256",
            args.expected_problem_source_sha256,
            contract.problem_source_sha256,
        ),
        (
            "--expected-evaluation-binding-sha256",
            args.expected_evaluation_binding_sha256,
            contract.evaluation_binding_sha256,
        ),
        (
            "--expected-run-contract-sha256",
            args.expected_run_contract_sha256,
            contract.sha256,
        ),
    ):
        _require_digest(parser, flag, supplied, actual)
    if base_replay is not None:
        _require_digest(
            parser,
            "--expected-base-replay-projection-sha256",
            args.expected_base_replay_projection_sha256,
            base_replay.binding.projection_sha256,
        )
    if args.run_output is None or args.diagnostic_output is None:
        parser.error("run and diagnostic output paths are required for execution")
    worker_client = _client(worker, _secret(parser, args, "worker"))
    builder_client = _client(builder, _secret(parser, args, "builder"))
    result = collect_action_census_run(
        run_path=args.run_output,
        diagnostic_path=args.diagnostic_output,
        dataset_path=args.problems,
        contract=contract,
        problems=problems,
        worker_client=worker_client,
        state_builder_client=builder_client,
        base_replay=base_replay,
    )
    audit = audit_action_census_artifacts(
        run_path=result.run_path,
        diagnostic_path=result.diagnostic_path,
        contract=contract,
        problems=problems,
        base_replay=base_replay,
    )
    print(
        canonical_json_bytes(
            {
                "schema_version": "action-census-execution-v3",
                "preregistration_id": contract.preregistration_id,
                "run_contract_sha256": result.run_contract_sha256,
                "case_count": result.case_count,
                "logical_call_count": result.logical_call_count,
                "transition_count": result.transition_count,
                "analysis": result.analysis,
                "prepublication_and_sealed_artifact_audit": audit,
                "run_path": str(result.run_path),
                "diagnostic_path": str(result.diagnostic_path),
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
