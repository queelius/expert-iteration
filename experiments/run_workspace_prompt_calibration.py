"""Prepare or execute the E43R1 two-key prompt calibration."""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from pathlib import Path

from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient
from verifier_anchored_rl.workspace_v1.experiment import ModelIdentity
from verifier_anchored_rl.workspace_v1.records import canonical_json_bytes
from verifier_anchored_rl.workspace_v1.target_prompt_calibration import (
    build_prompt_calibration_contract,
    collect_prompt_calibration_run,
    prepare_e43_baseline,
)
from verifier_anchored_rl.workspace_v1.target_value import load_target_problems


_SHA256 = re.compile(r"[0-9a-f]{64}")
_WORKER = ModelIdentity(
    provider="huggingface-inference-providers:featherless-ai",
    model="Qwen/Qwen2.5-3B-Instruct:featherless-ai",
    revision="aa8e72537993ba99e69dfaafa59ed015b17504d1",
    base_url="https://router.huggingface.co/v1",
    guided_json=False,
    timeout_s=120.0,
    parse_retries=1,
    backoff_retries=6,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-contracts", action="store_true")
    mode.add_argument("--execute-hosted", action="store_true")
    parser.add_argument("--preregistration-id", required=True)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--baseline-run-sha256", required=True)
    parser.add_argument("--baseline-run-contract-sha256", required=True)
    parser.add_argument("--run-output", type=Path)
    parser.add_argument("--diagnostic-output", type=Path)
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--expected-problem-source-sha256")
    parser.add_argument("--expected-evaluation-binding-sha256")
    parser.add_argument("--expected-baseline-selection-sha256")
    parser.add_argument("--expected-run-contract-sha256")
    parser.add_argument("--worker-api-key-env")
    return parser


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
        problems = load_target_problems(args.problems)[:12]
        family_counts = Counter(problem.family for problem in problems)
        if family_counts != Counter({
            "f1a": 3,
            "f1b": 3,
            "f2a": 3,
            "f2b": 3,
        }):
            raise ValueError("E43R1 requires three problems from each family")
        baseline = prepare_e43_baseline(
            run_path=args.baseline_run,
            expected_run_sha256=args.baseline_run_sha256,
            expected_run_contract_sha256=(
                args.baseline_run_contract_sha256
            ),
            problems=problems,
        )
        contract = build_prompt_calibration_contract(
            preregistration_id=args.preregistration_id,
            dataset_path=args.problems,
            problems=problems,
            baseline=baseline,
            worker=_WORKER,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))

    preparation = {
        "schema_version": "workspace-two-key-calibration-preparation-v1",
        "preregistration_id": contract.preregistration_id,
        "problem_count": len(problems),
        "family_counts": dict(sorted(family_counts.items())),
        "dataset_sha256": contract.dataset_sha256,
        "problem_source_sha256": contract.problem_source_sha256,
        "evaluation_binding_sha256": contract.evaluation_binding_sha256,
        "baseline_run_sha256": contract.baseline_run_sha256,
        "baseline_selection_sha256": contract.baseline_selection_sha256,
        "worker": contract.worker.to_record(),
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
            "--expected-baseline-selection-sha256",
            args.expected_baseline_selection_sha256,
            contract.baseline_selection_sha256,
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
        base_url=contract.worker.base_url.rstrip("/"),
        model=contract.worker.model,
        api_key=api_key,
        timeout_s=contract.worker.timeout_s,
        max_retries=contract.worker.parse_retries,
        max_backoff_retries=contract.worker.backoff_retries,
        guided_json=contract.worker.guided_json,
    )
    result = collect_prompt_calibration_run(
        run_path=args.run_output,
        diagnostic_path=args.diagnostic_output,
        dataset_path=args.problems,
        contract=contract,
        baseline=baseline,
        problems=problems,
        worker_client=worker,
    )
    summary = {
        "schema_version": "workspace-two-key-calibration-execution-v1",
        "preregistration_id": contract.preregistration_id,
        "logical_call_count": result.logical_call_count,
        "transition_count": result.transition_count,
        "run_contract_sha256": result.run_contract_sha256,
        "analysis": result.analysis,
        "run_path": str(result.run_path),
        "diagnostic_path": str(result.diagnostic_path),
    }
    print(canonical_json_bytes(summary).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
