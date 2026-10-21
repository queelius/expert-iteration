"""Prepare or execute one shared-prefix paired workspace lens run."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient
from verifier_anchored_rl.workspace_v1.experiment import (
    ModelIdentity,
    load_problem_examples,
)
from verifier_anchored_rl.workspace_v1.paired_lens import (
    build_paired_contract,
    collect_paired_lens_run,
    prepare_paired_prefixes,
)
from verifier_anchored_rl.workspace_v1.records import canonical_json_bytes


_SHA256 = re.compile(r"[0-9a-f]{64}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an exact shared-prefix paired lens contract or execute "
            "it after preregistration. Preparation makes no network call."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-contracts", action="store_true")
    mode.add_argument("--execute-hosted", action="store_true")
    parser.add_argument("--preregistration-id", required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-run-contract-sha256", required=True)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--episode-count", type=int, required=True)
    parser.add_argument("--max-depth", type=int, required=True)
    parser.add_argument("--run-output", type=Path)
    parser.add_argument("--diagnostic-output", type=Path)
    parser.add_argument("--expected-source-run-sha256")
    parser.add_argument("--expected-prefix-sha256")
    parser.add_argument("--expected-problem-source-sha256")
    parser.add_argument("--expected-evaluation-binding-sha256")
    parser.add_argument("--expected-run-contract-sha256")
    parser.add_argument("--worker-provider", required=True)
    parser.add_argument("--worker-model", required=True)
    parser.add_argument("--worker-revision", required=True)
    parser.add_argument("--worker-base-url", required=True)
    parser.add_argument("--worker-api-key-env")
    parser.add_argument(
        "--worker-guided-json",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
    parser.add_argument("--worker-timeout-s", type=float, required=True)
    parser.add_argument("--worker-parse-retries", type=int, required=True)
    parser.add_argument("--worker-backoff-retries", type=int, required=True)
    parser.add_argument("--worker-temperature", type=float, required=True)
    parser.add_argument("--worker-max-tokens", type=int, required=True)
    parser.add_argument("--worker-seed", type=int, required=True)
    return parser


def _identity(args: argparse.Namespace) -> ModelIdentity:
    return ModelIdentity(
        provider=args.worker_provider,
        model=args.worker_model,
        revision=args.worker_revision,
        base_url=args.worker_base_url,
        guided_json=args.worker_guided_json,
        timeout_s=args.worker_timeout_s,
        parse_retries=args.worker_parse_retries,
        backoff_retries=args.worker_backoff_retries,
    )


def _require_digest(
    parser: argparse.ArgumentParser,
    flag: str,
    supplied: str | None,
    actual: str,
) -> None:
    if supplied is None or _SHA256.fullmatch(supplied) is None:
        parser.error(f"{flag} requires a lowercase sha256 digest")
    if supplied != actual:
        parser.error(f"{flag} differs from the prepared contract")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        examples = load_problem_examples(
            args.problems,
            start_index=args.start_index,
            episode_count=args.episode_count,
        )
        prepared = prepare_paired_prefixes(
            source_run_path=args.source_run,
            examples=examples,
            expected_source_run_contract_sha256=(
                args.source_run_contract_sha256
            ),
            max_depth=args.max_depth,
        )
        identity = _identity(args)
        contract = build_paired_contract(
            preregistration_id=args.preregistration_id,
            prepared=prepared,
            worker=identity,
            worker_temperature=args.worker_temperature,
            worker_max_tokens=args.worker_max_tokens,
            worker_seed=args.worker_seed,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))

    preparation = {
        "schema_version": "workspace-paired-lens-preparation-v1",
        "preregistration_id": contract.preregistration_id,
        "case_count": len(prepared.cases),
        "problem_ids": list(contract.problem_ids),
        "excluded_problem_ids": list(contract.excluded_problem_ids),
        "source_run_sha256": contract.source_run_sha256,
        "source_run_contract_sha256": contract.source_run_contract_sha256,
        "prefix_sha256": contract.prefix_sha256,
        "problem_source_sha256": contract.problem_source_sha256,
        "evaluation_binding_sha256": contract.evaluation_binding_sha256,
        "run_contract_sha256": contract.sha256,
    }
    if args.print_contracts:
        print(canonical_json_bytes(preparation).decode("ascii"))
        return 0

    for flag, supplied, actual in (
        (
            "--expected-source-run-sha256",
            args.expected_source_run_sha256,
            contract.source_run_sha256,
        ),
        (
            "--expected-prefix-sha256",
            args.expected_prefix_sha256,
            contract.prefix_sha256,
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
    if args.run_output is None or args.diagnostic_output is None:
        parser.error(
            "--run-output and --diagnostic-output are required for execution"
        )
    variable = args.worker_api_key_env
    if variable is None or not variable.strip():
        parser.error("--worker-api-key-env is required for execution")
    api_key = os.environ.get(variable)
    if api_key is None or not api_key.strip():
        parser.error("worker API key environment variable is unset")
    worker = OpenAICompatibleLLMClient(
        base_url=identity.base_url.rstrip("/"),
        model=identity.model,
        api_key=api_key,
        timeout_s=identity.timeout_s,
        max_retries=identity.parse_retries,
        max_backoff_retries=identity.backoff_retries,
        guided_json=identity.guided_json,
    )
    result = collect_paired_lens_run(
        run_path=args.run_output,
        diagnostic_path=args.diagnostic_output,
        contract=contract,
        prepared=prepared,
        worker_client=worker,
    )
    summary = {
        "schema_version": "workspace-paired-lens-execution-summary-v1",
        "preregistration_id": contract.preregistration_id,
        "run_contract_sha256": result.run_contract_sha256,
        "case_count": result.case_count,
        "transition_count": result.transition_count,
        "solved_by_mode": dict(result.solved_by_mode),
        "run_path": str(result.run_path),
        "diagnostic_path": str(result.diagnostic_path),
    }
    print(canonical_json_bytes(summary).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
