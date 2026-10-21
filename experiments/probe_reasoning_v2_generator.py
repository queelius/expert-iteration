"""Run a preregistered generator-only probe over a sealed v2 run."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient
from verifier_anchored_rl.reasoning_v2.casework_source import (
    CaseworkProblemSource,
)
from verifier_anchored_rl.reasoning_v2.contracts import canonical_json_bytes
from verifier_anchored_rl.reasoning_v2.microprobe import (
    GeneratorMicroprobeAuthorization,
    prepare_generator_microprobe,
    run_generator_microprobe,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a sealed v2 run, hold every recorded state, action, and "
            "first seed fixed, and call only the pinned generator under one "
            "explicit replacement lens contract."
        ),
        epilog=(
            "The output is a counterfactual diagnostic artifact, never a "
            "training run or reward claim. Hosted execution requires a prior "
            "experiment-ledger registration and exact expected digests."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--print-contract",
        action="store_true",
        help="Strict-replay and print the probe contract without key lookup or HTTP.",
    )
    mode.add_argument(
        "--execute-hosted",
        action="store_true",
        help="Acknowledge that all sealed rows will spend hosted generator credit.",
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-diagnostic", type=Path, required=True)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--episode-count", type=int, required=True)
    parser.add_argument("--probe-lens-contract", required=True)
    parser.add_argument("--expected-decision-count", type=int, required=True)
    parser.add_argument("--preregistration-id")
    parser.add_argument("--expected-source-run-sha256")
    parser.add_argument("--expected-source-diagnostic-sha256")
    parser.add_argument("--expected-probe-contract-sha256")
    parser.add_argument("--generator-api-key-env")
    parser.add_argument("--output", type=Path)
    return parser


def _required_digest(
    parser: argparse.ArgumentParser, flag: str, value: str | None
) -> str:
    if value is None or _SHA256.fullmatch(value) is None:
        parser.error(f"{flag} requires a lowercase sha256 digest")
    return value


def _secret(
    parser: argparse.ArgumentParser, variable: str | None
) -> str:
    if variable is None or not variable.strip():
        parser.error("--generator-api-key-env is required for hosted execution")
    value = os.environ.get(variable)
    if value is None or not value.strip():
        parser.error(f"generator API key environment variable {variable!r} is unset")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    full_source = CaseworkProblemSource.from_jsonl(args.problems)
    problem_source = full_source.select_slice(
        args.start_index, args.episode_count
    )
    prepared = prepare_generator_microprobe(
        run_path=args.source_run,
        diagnostic_path=args.source_diagnostic,
        problem_source=problem_source,
        probe_lens_contract=args.probe_lens_contract,
    )
    if len(prepared.cases) != args.expected_decision_count:
        parser.error(
            "strict-replayed decision count differs from "
            "--expected-decision-count"
        )
    summary = {
        "schema_version": "generator-microprobe-preparation-v1",
        "probe_contract": prepared.contract.to_record(),
        "probe_contract_sha256": prepared.contract.sha256,
        "case_manifest_sha256": prepared.contract.case_manifest_sha256,
        "source_environment_contract_sha256": (
            prepared.source_environment.sha256
        ),
        "probe_environment_contract_sha256": (
            prepared.probe_environment.sha256
        ),
        "decision_count": len(prepared.cases),
    }
    if args.print_contract:
        print(canonical_json_bytes(summary).decode("ascii"))
        return 0

    if args.preregistration_id is None:
        parser.error("--preregistration-id is required for hosted execution")
    authorization = GeneratorMicroprobeAuthorization(
        args.preregistration_id,
        _required_digest(
            parser,
            "--expected-source-run-sha256",
            args.expected_source_run_sha256,
        ),
        _required_digest(
            parser,
            "--expected-source-diagnostic-sha256",
            args.expected_source_diagnostic_sha256,
        ),
        _required_digest(
            parser,
            "--expected-probe-contract-sha256",
            args.expected_probe_contract_sha256,
        ),
    )
    try:
        authorization.validate(prepared.contract)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output is None:
        parser.error("--output is required for hosted execution")
    output = args.output
    partial = output.with_name(f"{output.name}.partial")
    if output.exists() or partial.exists():
        parser.error("microprobe output or partial output already exists")
    if not output.parent.exists() or not output.parent.is_dir():
        parser.error("microprobe output directory does not exist")

    transport = prepared.contract.generator_transport
    api_key = _secret(parser, args.generator_api_key_env)
    client = OpenAICompatibleLLMClient(
        base_url=transport.base_url,
        model=transport.model,
        api_key=api_key,
        timeout_s=transport.timeout_s,
        max_retries=transport.parse_retry_budget,
        max_backoff_retries=transport.backoff_retry_budget,
        guided_json=transport.guided_json,
    )
    result = run_generator_microprobe(
        output_path=output,
        prepared=prepared,
        generator_client=client,
        authorization=authorization,
    )
    print(canonical_json_bytes({
        "schema_version": "generator-microprobe-completion-v1",
        "output": str(result.output_path),
        "probe_contract_sha256": result.contract_sha256,
        "case_count": result.case_count,
        "wire_usable_count": result.wire_usable_count,
        "decoded_count": result.decoded_count,
        "symbolic_applied_count": result.symbolic_applied_count,
        "generator_generate_calls": client.generate_calls,
        "generator_prompt_tokens": client.prompt_tokens,
        "generator_generated_tokens": client.generated_tokens,
    }).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
