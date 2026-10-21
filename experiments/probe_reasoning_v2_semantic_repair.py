"""Run a preregistered one-shot decomposition semantic-repair probe."""

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
from verifier_anchored_rl.reasoning_v2.semantic_repair import (
    SemanticRepairProbeAuthorization,
    prepare_semantic_repair_probe,
    run_semantic_repair_probe,
    semantic_repair_feedback_map_record,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate sealed E23-E25 artifacts, recover the 33 rejected "
            "E24 DECOMPOSE requests, append one bounded E25 reason-specific "
            "feedback object, and call only the pinned generator once per row."
        ),
        epilog=(
            "This is a cross-version generator diagnostic. It creates no "
            "controller decision, trajectory, closure, reward, or training row."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--print-contract",
        action="store_true",
        help=(
            "Strict-replay and print every binding without key lookup, HTTP, "
            "or artifact creation."
        ),
    )
    mode.add_argument(
        "--execute-hosted",
        action="store_true",
        help="Execute only after the printed contract is preregistered.",
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-diagnostic", type=Path, required=True)
    parser.add_argument("--e24-artifact", type=Path, required=True)
    parser.add_argument("--e25-artifact", type=Path, required=True)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--episode-count", type=int, required=True)
    parser.add_argument(
        "--expected-source-decision-count", type=int, required=True
    )
    parser.add_argument("--expected-decompose-count", type=int, required=True)
    parser.add_argument("--expected-problem-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preregistration-id")
    parser.add_argument("--expected-source-run-sha256")
    parser.add_argument("--expected-source-diagnostic-sha256")
    parser.add_argument("--expected-source-environment-sha256")
    parser.add_argument("--expected-request-environment-sha256")
    parser.add_argument("--expected-source-collection-sha256")
    parser.add_argument("--expected-source-problem-sha256")
    parser.add_argument("--expected-e24-artifact-sha256")
    parser.add_argument("--expected-e24-probe-contract-sha256")
    parser.add_argument("--expected-e24-case-manifest-sha256")
    parser.add_argument("--expected-e25-artifact-sha256")
    parser.add_argument("--expected-e25-rescore-contract-sha256")
    parser.add_argument("--expected-e25-rescore-manifest-sha256")
    parser.add_argument("--expected-e25-result-manifest-sha256")
    parser.add_argument("--expected-scoring-environment-sha256")
    parser.add_argument("--expected-corrected-profile-sha256")
    parser.add_argument("--expected-feedback-map-sha256")
    parser.add_argument("--expected-case-manifest-sha256")
    parser.add_argument("--expected-base-request-manifest-sha256")
    parser.add_argument("--expected-repair-request-manifest-sha256")
    parser.add_argument("--expected-generator-sha256")
    parser.add_argument("--expected-generator-transport-sha256")
    parser.add_argument("--expected-generation-settings-sha256")
    parser.add_argument("--expected-contract-sha256")
    parser.add_argument("--generator-api-key-env")
    return parser


def _required_digest(
    parser: argparse.ArgumentParser, flag: str, value: str | None
) -> str:
    if value is None or _SHA256.fullmatch(value) is None:
        parser.error(f"{flag} requires a lowercase sha256 digest")
    return value


def _secret(parser: argparse.ArgumentParser, variable: str | None) -> str:
    if variable is None or not variable.strip():
        parser.error("--generator-api-key-env is required for hosted execution")
    value = os.environ.get(variable)
    if value is None or not value.strip():
        parser.error(
            f"generator API key environment variable {variable!r} is unset"
        )
    return value


def _preparation_record(prepared) -> dict[str, object]:
    contract = prepared.contract
    return {
        "schema_version": "decomposition-semantic-repair-preparation-v1",
        "repair_contract": contract.to_record(),
        "repair_contract_sha256": contract.sha256,
        "source_run_sha256": contract.source_run_sha256,
        "source_diagnostic_sha256": contract.source_diagnostic_sha256,
        "source_environment_contract_sha256": (
            contract.source_environment_contract_sha256
        ),
        "request_environment_contract_sha256": (
            contract.request_environment_contract_sha256
        ),
        "source_collection_contract_sha256": (
            contract.source_collection_contract_sha256
        ),
        "source_problem_source_sha256": (
            contract.source_problem_source_sha256
        ),
        "e24_artifact_sha256": contract.e24_artifact_sha256,
        "e24_probe_contract_sha256": contract.e24_probe_contract_sha256,
        "e24_case_manifest_sha256": contract.e24_case_manifest_sha256,
        "e25_artifact_sha256": contract.e25_artifact_sha256,
        "e25_rescore_contract_sha256": (
            contract.e25_rescore_contract_sha256
        ),
        "e25_rescore_case_manifest_sha256": (
            contract.e25_rescore_case_manifest_sha256
        ),
        "e25_result_manifest_sha256": contract.e25_result_manifest_sha256,
        "scoring_environment_contract_sha256": (
            contract.scoring_environment_contract_sha256
        ),
        "corrected_profile_sha256": contract.corrected_profile_sha256,
        "feedback_map": semantic_repair_feedback_map_record(),
        "feedback_map_sha256": contract.feedback_map_sha256,
        "case_manifest_sha256": contract.case_manifest_sha256,
        "base_request_manifest_sha256": (
            contract.base_request_manifest_sha256
        ),
        "repair_request_manifest_sha256": (
            contract.repair_request_manifest_sha256
        ),
        "generator_sha256": contract.generator_sha256,
        "generator_transport_sha256": contract.generator_transport_sha256,
        "generation_settings_sha256": contract.generation_settings_sha256,
        "source_decision_count": contract.source_decision_count,
        "decompose_case_count": contract.case_count,
        "problem_count": contract.problem_count,
        "planned_output_path": contract.planned_output_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    full_source = CaseworkProblemSource.from_jsonl(args.problems)
    problem_source = full_source.select_slice(
        args.start_index, args.episode_count
    )
    prepared = prepare_semantic_repair_probe(
        source_run_path=args.source_run,
        source_diagnostic_path=args.source_diagnostic,
        e24_artifact_path=args.e24_artifact,
        e25_artifact_path=args.e25_artifact,
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
            "selected DECOMPOSE count differs from --expected-decompose-count"
        )
    if contract.problem_count != args.expected_problem_count:
        parser.error(
            "distinct problem count differs from --expected-problem-count"
        )
    preparation = _preparation_record(prepared)
    if args.print_contract:
        print(canonical_json_bytes(preparation).decode("ascii"))
        return 0

    if args.preregistration_id is None or not args.preregistration_id.strip():
        parser.error("--preregistration-id is required for hosted execution")
    authorization = SemanticRepairProbeAuthorization(
        preregistration_id=args.preregistration_id,
        expected_source_run_sha256=_required_digest(
            parser, "--expected-source-run-sha256", args.expected_source_run_sha256
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
        expected_request_environment_contract_sha256=_required_digest(
            parser,
            "--expected-request-environment-sha256",
            args.expected_request_environment_sha256,
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
        expected_e24_case_manifest_sha256=_required_digest(
            parser,
            "--expected-e24-case-manifest-sha256",
            args.expected_e24_case_manifest_sha256,
        ),
        expected_e25_artifact_sha256=_required_digest(
            parser,
            "--expected-e25-artifact-sha256",
            args.expected_e25_artifact_sha256,
        ),
        expected_e25_rescore_contract_sha256=_required_digest(
            parser,
            "--expected-e25-rescore-contract-sha256",
            args.expected_e25_rescore_contract_sha256,
        ),
        expected_e25_rescore_case_manifest_sha256=_required_digest(
            parser,
            "--expected-e25-rescore-manifest-sha256",
            args.expected_e25_rescore_manifest_sha256,
        ),
        expected_e25_result_manifest_sha256=_required_digest(
            parser,
            "--expected-e25-result-manifest-sha256",
            args.expected_e25_result_manifest_sha256,
        ),
        expected_scoring_environment_contract_sha256=_required_digest(
            parser,
            "--expected-scoring-environment-sha256",
            args.expected_scoring_environment_sha256,
        ),
        expected_corrected_profile_sha256=_required_digest(
            parser,
            "--expected-corrected-profile-sha256",
            args.expected_corrected_profile_sha256,
        ),
        expected_feedback_map_sha256=_required_digest(
            parser,
            "--expected-feedback-map-sha256",
            args.expected_feedback_map_sha256,
        ),
        expected_case_manifest_sha256=_required_digest(
            parser,
            "--expected-case-manifest-sha256",
            args.expected_case_manifest_sha256,
        ),
        expected_base_request_manifest_sha256=_required_digest(
            parser,
            "--expected-base-request-manifest-sha256",
            args.expected_base_request_manifest_sha256,
        ),
        expected_repair_request_manifest_sha256=_required_digest(
            parser,
            "--expected-repair-request-manifest-sha256",
            args.expected_repair_request_manifest_sha256,
        ),
        expected_generator_sha256=_required_digest(
            parser,
            "--expected-generator-sha256",
            args.expected_generator_sha256,
        ),
        expected_generator_transport_sha256=_required_digest(
            parser,
            "--expected-generator-transport-sha256",
            args.expected_generator_transport_sha256,
        ),
        expected_generation_settings_sha256=_required_digest(
            parser,
            "--expected-generation-settings-sha256",
            args.expected_generation_settings_sha256,
        ),
        expected_contract_sha256=_required_digest(
            parser,
            "--expected-contract-sha256",
            args.expected_contract_sha256,
        ),
        expected_source_decision_count=args.expected_source_decision_count,
        expected_case_count=args.expected_decompose_count,
        expected_problem_count=args.expected_problem_count,
        expected_output_path=args.output.as_posix(),
    )
    try:
        authorization.validate(contract)
    except ValueError as exc:
        parser.error(str(exc))
    partial = args.output.with_name(f"{args.output.name}.partial")
    if args.output.exists() or partial.exists():
        parser.error("semantic-repair output or partial output already exists")
    if not args.output.parent.exists() or not args.output.parent.is_dir():
        parser.error("semantic-repair output directory does not exist")

    transport = contract.generator_transport
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
    result = run_semantic_repair_probe(
        output_path=args.output,
        prepared=prepared,
        generator_client=client,
        authorization=authorization,
    )
    print(canonical_json_bytes({
        "schema_version": "decomposition-semantic-repair-completion-v1",
        "output": str(result.output_path),
        "repair_contract_sha256": result.contract_sha256,
        "case_count": result.case_count,
        "wire_usable_count": result.wire_usable_count,
        "decoded_count": result.decoded_count,
        "symbolic_applied_count": result.symbolic_applied_count,
        "applied_problem_count": result.applied_problem_count,
        "generator_generate_calls": client.generate_calls,
        "generator_prompt_tokens": client.prompt_tokens,
        "generator_generated_tokens": client.generated_tokens,
    }).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
