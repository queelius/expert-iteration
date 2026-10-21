"""Prepare or execute the sealed E43R2 tolerant first-attempt replay."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from verifier_anchored_rl.workspace_v1.records import canonical_json_bytes
from verifier_anchored_rl.workspace_v1.target_value import load_target_problems
from verifier_anchored_rl.workspace_v1.tolerant_replay import (
    build_tolerant_replay_contract,
    collect_tolerant_replay,
    prepare_tolerant_replay,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-contracts", action="store_true")
    mode.add_argument("--execute-replay", action="store_true")
    parser.add_argument("--preregistration-id", required=True)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--e43-run", type=Path, required=True)
    parser.add_argument("--e43-diagnostics", type=Path, required=True)
    parser.add_argument("--e43-run-sha256", required=True)
    parser.add_argument("--e43-diagnostic-sha256", required=True)
    parser.add_argument("--e43-run-contract-sha256", required=True)
    parser.add_argument("--run-output", type=Path)
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--expected-problem-source-sha256")
    parser.add_argument("--expected-evaluation-binding-sha256")
    parser.add_argument("--expected-selected-attempts-sha256")
    parser.add_argument("--expected-run-contract-sha256")
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
        problems = load_target_problems(args.problems)
        if len(problems) != 60:
            raise ValueError("E43R2 requires exactly 60 problems")
        prepared = prepare_tolerant_replay(
            e43_run_path=args.e43_run,
            e43_diagnostic_path=args.e43_diagnostics,
            expected_e43_run_sha256=args.e43_run_sha256,
            expected_e43_diagnostic_sha256=args.e43_diagnostic_sha256,
            expected_e43_run_contract_sha256=(
                args.e43_run_contract_sha256
            ),
            problems=problems,
        )
        contract = build_tolerant_replay_contract(
            preregistration_id=args.preregistration_id,
            dataset_path=args.problems,
            problems=problems,
            prepared=prepared,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))

    preparation = {
        "schema_version": "workspace-tolerant-replay-preparation-v1",
        "preregistration_id": contract.preregistration_id,
        "problem_count": len(problems),
        "dataset_sha256": contract.dataset_sha256,
        "problem_source_sha256": contract.problem_source_sha256,
        "evaluation_binding_sha256": contract.evaluation_binding_sha256,
        "e43_run_sha256": contract.e43_run_sha256,
        "e43_diagnostic_sha256": contract.e43_diagnostic_sha256,
        "selected_attempts_sha256": contract.selected_attempts_sha256,
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
            "--expected-selected-attempts-sha256",
            args.expected_selected_attempts_sha256,
            contract.selected_attempts_sha256,
        ),
        (
            "--expected-run-contract-sha256",
            args.expected_run_contract_sha256,
            contract.sha256,
        ),
    ):
        _require_digest(parser, flag, supplied, actual)
    if args.run_output is None:
        parser.error("--run-output is required for execution")
    summary = collect_tolerant_replay(
        run_path=args.run_output,
        dataset_path=args.problems,
        contract=contract,
        prepared=prepared,
        problems=problems,
    )
    output = {
        "schema_version": "workspace-tolerant-replay-execution-v1",
        "preregistration_id": contract.preregistration_id,
        "run_contract_sha256": contract.sha256,
        "run_path": str(args.run_output),
        "summary": summary,
    }
    print(canonical_json_bytes(output).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
