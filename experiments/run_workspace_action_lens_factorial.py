"""Prepare or execute the sealed-state action-lens factorial experiment."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path

from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient
from verifier_anchored_rl.workspace_v1.action_lens_factorial import (
    audit_action_lens_artifacts,
    build_action_lens_contract,
    collect_action_lens_run,
    prepare_action_lens_source,
)
from verifier_anchored_rl.workspace_v1.experiment import ModelIdentity
from verifier_anchored_rl.workspace_v1.records import canonical_json_bytes

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _case_indices(value: str) -> tuple[int, ...]:
    parts = value.split(",")
    try:
        indices = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "source case indices must be comma-separated integers"
        ) from exc
    if (
        not indices
        or any(not part or index < 0 for part, index in zip(parts, indices))
        or len(indices) != len(set(indices))
    ):
        raise argparse.ArgumentTypeError(
            "source case indices must be nonnegative and unique"
        )
    return indices


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-contracts", action="store_true")
    mode.add_argument("--execute-hosted", action="store_true")
    parser.add_argument("--preregistration-id", required=True)
    parser.add_argument(
        "--stage",
        choices=("integration_smoke", "pilot"),
        required=True,
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-diagnostic", type=Path, required=True)
    parser.add_argument("--source-run-sha256", required=True)
    parser.add_argument("--source-diagnostic-sha256", required=True)
    parser.add_argument("--source-run-contract-sha256", required=True)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument(
        "--source-case-indices",
        type=_case_indices,
        required=True,
        help="Comma-separated zero-based indices into the sealed source run.",
    )
    parser.add_argument("--run-output", type=Path)
    parser.add_argument("--diagnostic-output", type=Path)
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--expected-selection-sha256")
    parser.add_argument("--expected-problem-source-sha256")
    parser.add_argument("--expected-evaluation-binding-sha256")
    parser.add_argument("--expected-order-schedule-sha256")
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
        parser.error(f"{flag} requires a lowercase sha256")
    if supplied != actual:
        parser.error(f"{flag} differs from the prepared contract")


def _worker_secret(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> str:
    variable = args.worker_api_key_env
    if not isinstance(variable, str) or not variable.strip():
        parser.error("--worker-api-key-env is required for execution")
    value = os.environ.get(variable)
    if value is None or not value.strip():
        parser.error("worker API key environment variable is unset")
    return value


def _operational_summary(path: Path) -> dict[str, object]:
    last_line = ""
    with path.open("r", encoding="ascii", newline="") as handle:
        for line in handle:
            if line.strip():
                last_line = line
    record = json.loads(last_line)
    if (
        not isinstance(record, Mapping)
        or record.get("record_type") != "run_summary"
        or not isinstance(record.get("call_failure_counts"), Mapping)
        or not isinstance(record.get("unusable_output_counts"), Mapping)
        or not isinstance(record.get("usage"), Mapping)
    ):
        raise RuntimeError("audited action-lens run lacks its operational summary")
    return {
        "call_failure_counts": record["call_failure_counts"],
        "unusable_output_counts": record["unusable_output_counts"],
        "usage": record["usage"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        prepared = prepare_action_lens_source(
            source_run_path=args.source_run,
            source_diagnostic_path=args.source_diagnostic,
            expected_source_run_sha256=args.source_run_sha256,
            expected_source_diagnostic_sha256=args.source_diagnostic_sha256,
            expected_source_run_contract_sha256=(args.source_run_contract_sha256),
            dataset_path=args.problems,
            source_case_indices=args.source_case_indices,
        )
        worker = _identity(args)
        contract = build_action_lens_contract(
            preregistration_id=args.preregistration_id,
            stage=args.stage,
            prepared=prepared,
            worker=worker,
            worker_temperature=args.worker_temperature,
            worker_max_tokens=args.worker_max_tokens,
            worker_seed=args.worker_seed,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))

    header = contract.header()
    source = header["source"]
    preparation = {
        "schema_version": "action-lens-factorial-preparation-v1",
        "preregistration_id": contract.preregistration_id,
        "stage": contract.stage,
        "case_count": len(prepared.cases),
        "problem_ids": list(contract.problem_ids),
        "source": source,
        "dataset_sha256": contract.dataset_sha256,
        "selection_sha256": contract.selection_sha256,
        "problem_source_sha256": contract.problem_source_sha256,
        "evaluation_binding_sha256": contract.evaluation_binding_sha256,
        "order_schedule_sha256": contract.order_schedule_sha256,
        "prompt_bindings": header["prompt_bindings"],
        "implementation_binding": header["implementation_binding"],
        "settings": header["settings"],
        "models": header["models"],
        "run_contract_sha256": contract.sha256,
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
            "--expected-selection-sha256",
            args.expected_selection_sha256,
            contract.selection_sha256,
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
            "--expected-order-schedule-sha256",
            args.expected_order_schedule_sha256,
            contract.order_schedule_sha256,
        ),
        (
            "--expected-run-contract-sha256",
            args.expected_run_contract_sha256,
            contract.sha256,
        ),
    ):
        _require_digest(parser, flag, supplied, actual)
    if args.run_output is None or args.diagnostic_output is None:
        parser.error("--run-output and --diagnostic-output are required for execution")

    worker_client = OpenAICompatibleLLMClient(
        base_url=worker.base_url.rstrip("/"),
        model=worker.model,
        api_key=_worker_secret(parser, args),
        timeout_s=worker.timeout_s,
        max_retries=worker.parse_retries,
        max_backoff_retries=worker.backoff_retries,
        guided_json=worker.guided_json,
    )
    result = collect_action_lens_run(
        run_path=args.run_output,
        diagnostic_path=args.diagnostic_output,
        contract=contract,
        prepared=prepared,
        worker_client=worker_client,
    )
    audit = audit_action_lens_artifacts(
        run_path=result.run_path,
        diagnostic_path=result.diagnostic_path,
        contract=contract,
        prepared=prepared,
    )
    operational = _operational_summary(result.run_path)
    summary: dict[str, object] = {
        "schema_version": "action-lens-factorial-execution-v1",
        "preregistration_id": contract.preregistration_id,
        "stage": contract.stage,
        "run_contract_sha256": result.run_contract_sha256,
        "case_count": result.case_count,
        "logical_call_count": result.logical_call_count,
        "transition_count": result.transition_count,
        **operational,
        "prepublication_and_sealed_artifact_audit": audit,
        "run_path": str(result.run_path),
        "diagnostic_path": str(result.diagnostic_path),
    }
    if contract.stage == "pilot":
        summary["analysis"] = result.analysis
    else:
        summary["analysis_withheld"] = "reward-blind integration stage"
    print(canonical_json_bytes(summary).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
