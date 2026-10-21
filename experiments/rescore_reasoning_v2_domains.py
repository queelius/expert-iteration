"""Run a preregistered deterministic domain-parser rescore."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from verifier_anchored_rl.reasoning_v2.casework_source import (
    CaseworkProblemSource,
)
from verifier_anchored_rl.reasoning_v2.contracts import canonical_json_bytes
from verifier_anchored_rl.reasoning_v2.domain_rescore import (
    DomainRescoreAuthorization,
    prepare_domain_rescore,
    run_domain_rescore,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate sealed E23 and E24 inputs, recover the 33 exact "
            "DECOMPOSE drafts, and compare the legacy and corrected domain "
            "parser profiles without any model call."
        ),
        epilog=(
            "The output is a deterministic adapter diagnostic, never a "
            "trajectory, reward record, model evaluation, or training row."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--print-contract",
        action="store_true",
        help=(
            "Strict-replay and print all frozen bindings without writing "
            "the rescore artifact."
        ),
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Execute only after every printed binding is preregistered.",
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-diagnostic", type=Path, required=True)
    parser.add_argument("--e24-artifact", type=Path, required=True)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--episode-count", type=int, required=True)
    parser.add_argument("--expected-source-decision-count", type=int, required=True)
    parser.add_argument("--expected-decompose-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preregistration-id")
    parser.add_argument("--expected-source-run-sha256")
    parser.add_argument("--expected-source-diagnostic-sha256")
    parser.add_argument("--expected-source-environment-sha256")
    parser.add_argument("--expected-source-collection-sha256")
    parser.add_argument("--expected-source-problem-sha256")
    parser.add_argument("--expected-e24-artifact-sha256")
    parser.add_argument("--expected-e24-probe-contract-sha256")
    parser.add_argument("--expected-e24-probe-environment-sha256")
    parser.add_argument("--expected-corrected-environment-sha256")
    parser.add_argument("--expected-e24-case-manifest-sha256")
    parser.add_argument("--expected-rescore-case-manifest-sha256")
    parser.add_argument("--expected-legacy-profile-sha256")
    parser.add_argument("--expected-corrected-profile-sha256")
    parser.add_argument("--expected-legacy-expectation-sha256")
    parser.add_argument("--expected-corrected-expectation-sha256")
    parser.add_argument("--expected-rescore-contract-sha256")
    return parser


def _required_digest(
    parser: argparse.ArgumentParser, flag: str, value: str | None
) -> str:
    if value is None or _SHA256.fullmatch(value) is None:
        parser.error(f"{flag} requires a lowercase sha256 digest")
    return value


def _preparation_record(prepared) -> dict[str, object]:
    contract = prepared.contract
    return {
        "schema_version": "domain-parser-rescore-preparation-v1",
        "rescore_contract": contract.to_record(),
        "rescore_contract_sha256": contract.sha256,
        "source_run_sha256": contract.source_run_sha256,
        "source_diagnostic_sha256": contract.source_diagnostic_sha256,
        "source_environment_contract_sha256": (
            contract.source_environment_contract_sha256
        ),
        "source_collection_contract_sha256": (
            contract.source_collection_contract_sha256
        ),
        "source_problem_source_sha256": (
            contract.source_problem_source_sha256
        ),
        "e24_artifact_sha256": contract.e24_artifact_sha256,
        "e24_probe_contract_sha256": contract.e24_probe_contract_sha256,
        "e24_probe_environment_contract_sha256": (
            contract.e24_probe_environment_contract_sha256
        ),
        "corrected_environment_contract_sha256": (
            contract.corrected_environment_contract_sha256
        ),
        "e24_case_manifest_sha256": contract.e24_case_manifest_sha256,
        "rescore_case_manifest_sha256": (
            contract.rescore_case_manifest_sha256
        ),
        "legacy_profile_sha256": contract.legacy_profile_sha256,
        "corrected_profile_sha256": contract.corrected_profile_sha256,
        "legacy_expectation_sha256": contract.legacy_expectation_sha256,
        "corrected_expectation_sha256": (
            contract.corrected_expectation_sha256
        ),
        "source_decision_count": contract.source_decision_count,
        "decompose_case_count": contract.case_count,
        "planned_output_path": contract.planned_output_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    full_source = CaseworkProblemSource.from_jsonl(args.problems)
    problem_source = full_source.select_slice(
        args.start_index, args.episode_count
    )
    prepared = prepare_domain_rescore(
        source_run_path=args.source_run,
        source_diagnostic_path=args.source_diagnostic,
        e24_artifact_path=args.e24_artifact,
        problem_source=problem_source,
        planned_output_path=args.output,
    )
    contract = prepared.contract
    if contract.source_decision_count != args.expected_source_decision_count:
        parser.error(
            "strict-replayed decision count differs from "
            "--expected-source-decision-count"
        )
    if contract.case_count != args.expected_decompose_count:
        parser.error(
            "selected DECOMPOSE count differs from "
            "--expected-decompose-count"
        )
    preparation = _preparation_record(prepared)
    if args.print_contract:
        print(canonical_json_bytes(preparation).decode("ascii"))
        return 0

    if args.preregistration_id is None or not args.preregistration_id.strip():
        parser.error("--preregistration-id is required for execution")
    authorization = DomainRescoreAuthorization(
        preregistration_id=args.preregistration_id,
        expected_source_run_sha256=_required_digest(
            parser,
            "--expected-source-run-sha256",
            args.expected_source_run_sha256,
        ),
        expected_source_diagnostic_sha256=_required_digest(
            parser,
            "--expected-source-diagnostic-sha256",
            args.expected_source_diagnostic_sha256,
        ),
        expected_source_environment_contract_sha256=_required_digest(
            parser,
            "--expected-source-environment-sha256",
            args.expected_source_environment_sha256,
        ),
        expected_source_collection_contract_sha256=_required_digest(
            parser,
            "--expected-source-collection-sha256",
            args.expected_source_collection_sha256,
        ),
        expected_source_problem_source_sha256=_required_digest(
            parser,
            "--expected-source-problem-sha256",
            args.expected_source_problem_sha256,
        ),
        expected_e24_artifact_sha256=_required_digest(
            parser,
            "--expected-e24-artifact-sha256",
            args.expected_e24_artifact_sha256,
        ),
        expected_e24_probe_contract_sha256=_required_digest(
            parser,
            "--expected-e24-probe-contract-sha256",
            args.expected_e24_probe_contract_sha256,
        ),
        expected_e24_probe_environment_contract_sha256=_required_digest(
            parser,
            "--expected-e24-probe-environment-sha256",
            args.expected_e24_probe_environment_sha256,
        ),
        expected_corrected_environment_contract_sha256=_required_digest(
            parser,
            "--expected-corrected-environment-sha256",
            args.expected_corrected_environment_sha256,
        ),
        expected_e24_case_manifest_sha256=_required_digest(
            parser,
            "--expected-e24-case-manifest-sha256",
            args.expected_e24_case_manifest_sha256,
        ),
        expected_rescore_case_manifest_sha256=_required_digest(
            parser,
            "--expected-rescore-case-manifest-sha256",
            args.expected_rescore_case_manifest_sha256,
        ),
        expected_legacy_profile_sha256=_required_digest(
            parser,
            "--expected-legacy-profile-sha256",
            args.expected_legacy_profile_sha256,
        ),
        expected_corrected_profile_sha256=_required_digest(
            parser,
            "--expected-corrected-profile-sha256",
            args.expected_corrected_profile_sha256,
        ),
        expected_legacy_expectation_sha256=_required_digest(
            parser,
            "--expected-legacy-expectation-sha256",
            args.expected_legacy_expectation_sha256,
        ),
        expected_corrected_expectation_sha256=_required_digest(
            parser,
            "--expected-corrected-expectation-sha256",
            args.expected_corrected_expectation_sha256,
        ),
        expected_rescore_contract_sha256=_required_digest(
            parser,
            "--expected-rescore-contract-sha256",
            args.expected_rescore_contract_sha256,
        ),
        expected_source_decision_count=args.expected_source_decision_count,
        expected_case_count=args.expected_decompose_count,
        expected_output_path=args.output.as_posix(),
    )
    try:
        authorization.validate(contract)
    except ValueError as exc:
        parser.error(str(exc))
    result = run_domain_rescore(
        output_path=args.output,
        prepared=prepared,
        authorization=authorization,
    )
    print(canonical_json_bytes({
        "schema_version": "domain-parser-rescore-completion-v1",
        "output": str(result.output_path),
        "rescore_contract_sha256": result.contract_sha256,
        "case_count": result.case_count,
        "legacy_applied_count": result.legacy_applied_count,
        "corrected_applied_count": result.corrected_applied_count,
    }).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
