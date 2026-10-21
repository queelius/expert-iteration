"""Run the preregistered E27 key-aligned semantic-repair probe."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient
from verifier_anchored_rl.reasoning_v2.casework_source import CaseworkProblemSource
from verifier_anchored_rl.reasoning_v2.contracts import canonical_json_bytes
from verifier_anchored_rl.reasoning_v2.key_aligned_repair import (
    KeyAlignedRepairProbeAuthorization,
    key_aligned_repair_feedback_map_record,
    prepare_key_aligned_repair_probe,
    run_key_aligned_repair_probe,
)
from verifier_anchored_rl.reasoning_v2.semantic_repair import (
    SemanticRepairProbeAuthorization,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_E26_ARTIFACT_SHA256 = (
    "7290e10733c1195edcea42684881afa4681f4c03f08e7aa7cf317a386623b1bd"
)
_E27_DIGEST_FIELDS = (
    "expected_source_environment_contract_sha256",
    "expected_request_environment_contract_sha256",
    "expected_scoring_environment_contract_sha256",
    "expected_corrected_profile_sha256",
    "expected_e26_artifact_sha256",
    "expected_e26_repair_contract_sha256",
    "expected_e26_case_manifest_sha256",
    "expected_e26_result_manifest_sha256",
    "expected_e26_summary_sha256",
    "expected_feedback_map_sha256",
    "expected_case_manifest_sha256",
    "expected_base_request_manifest_sha256",
    "expected_e26_repair_request_manifest_sha256",
    "expected_key_aligned_request_manifest_sha256",
    "expected_generator_sha256",
    "expected_generator_transport_sha256",
    "expected_generation_settings_sha256",
    "expected_contract_sha256",
)


def _sealed_e26_authorization() -> SemanticRepairProbeAuthorization:
    return SemanticRepairProbeAuthorization(
        preregistration_id="E26-v2-decompose-repair33",
        expected_source_run_sha256=(
            "5efea19bd29619aba71d6d147135aababd7ded474230918b7d77b72d8b7368f6"
        ),
        expected_source_diagnostic_sha256=(
            "0418043eb3abb50d86ad03c01946cc83602e76c8bd2c91f41defb87d3af6e369"
        ),
        expected_source_environment_contract_sha256=(
            "9316c93bc2cdf9b2ce507a989882af69669f8bf886e1f2ad6ce802c77dfdcfa7"
        ),
        expected_request_environment_contract_sha256=(
            "664deedf1caa44efff16daac1888c2e3966aa37e9a2631a62036643a9e681c71"
        ),
        expected_source_collection_contract_sha256=(
            "627a013c0aa4cc7c73e99f944209c1760a8b69b9f616030f8c8e8cf18cc321de"
        ),
        expected_source_problem_source_sha256=(
            "f3be145c1e536a98d4207c54ef85b5e0e516bce777c6ccd3d2a1f846c8881426"
        ),
        expected_e24_artifact_sha256=(
            "507c3b818ebee6c160bec021b84c60290f4af1a6fdcae725d2fa4efae74a5efc"
        ),
        expected_e24_probe_contract_sha256=(
            "3ef15fcccc0c3ce4fc109ae07ef3ad20ef29cb7677146b9fdce8f329352b019c"
        ),
        expected_e24_case_manifest_sha256=(
            "629205cc6f0047a19b81f31750c1fa0c74df1e30f2a142b026d1bdfcf100049d"
        ),
        expected_e25_artifact_sha256=(
            "f2be14e62cd04d680b454545cd253bfdb05c8ed5e404212b5b90bfcb7c8b1307"
        ),
        expected_e25_rescore_contract_sha256=(
            "ec307795e8dc39ae70515ca5036f5442f5784b72d40ed269fa3bb654ccbd842a"
        ),
        expected_e25_rescore_case_manifest_sha256=(
            "8955f15991ac60d9d99ec85399dcb573a178ab44cf4ddbdd07921e20d7f9fcb8"
        ),
        expected_e25_result_manifest_sha256=(
            "fb2919c2a043a4a1e241f149716cd176a23e16a2d898a55f7b9e5c71faeddabd"
        ),
        expected_scoring_environment_contract_sha256=(
            "3396ce7464c8bf4a8cd60c5aa26172c02dc4ed4589ca7eb9e2bcfd906119ee54"
        ),
        expected_corrected_profile_sha256=(
            "42ee4bf1f481dd92bebac990b8fc0c95d280a4a5c1054b0b9dc4d5070f6581c8"
        ),
        expected_feedback_map_sha256=(
            "56b646c26c48f6c1f5ba675aa322f49bfb4919df6c0ceace9ca08d9f36c2ded7"
        ),
        expected_case_manifest_sha256=(
            "4377ceda24ff9731c5d28f3b4ff3678b513f7f016c1b7471d158356e0df982c1"
        ),
        expected_base_request_manifest_sha256=(
            "ddc854aef8431f78550315f45a99633e67bb32e8d77bef06d6e21b612167286b"
        ),
        expected_repair_request_manifest_sha256=(
            "0cc914d1c3ec1c685b419a76ce3af603296e10b2f07da742d08c3f3096795b30"
        ),
        expected_generator_sha256=(
            "5fc23203d49417008fb63f5356f2a8e86cc249e77cbc3fac5b6a262858af8e58"
        ),
        expected_generator_transport_sha256=(
            "b0e2806c348fcde14bf2bbc3c48656eb7547f3b7142afc93f99ba52c40896f2a"
        ),
        expected_generation_settings_sha256=(
            "9059e1a6bbfb1422344ea9a6c9b9c5b85879f9fc0fe56de6ed2a4e1f846cd463"
        ),
        expected_contract_sha256=(
            "69643e2c01b103b6c1bda3ec296a1519806bcdce92286981bf3cd812deb72110"
        ),
        expected_source_decision_count=66,
        expected_case_count=33,
        expected_problem_count=5,
        expected_output_path=(
            "experiments/runs/2026-07-13-E26-v2-decompose-repair33.jsonl"
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the sealed E26 artifact, substitute only its feedback "
            "key and schema version, and call the pinned generator once per case."
        ),
        epilog=(
            "This is a cross-version wire diagnostic. It creates no controller "
            "decision, trajectory, closure, reward, or training row."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-contract", action="store_true")
    mode.add_argument("--execute-hosted", action="store_true")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-diagnostic", type=Path, required=True)
    parser.add_argument("--e24-artifact", type=Path, required=True)
    parser.add_argument("--e25-artifact", type=Path, required=True)
    parser.add_argument("--e26-artifact", type=Path, required=True)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--episode-count", type=int, required=True)
    parser.add_argument("--expected-source-decision-count", type=int, required=True)
    parser.add_argument("--expected-decompose-count", type=int, required=True)
    parser.add_argument("--expected-problem-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preregistration-id")
    for field_name in _E27_DIGEST_FIELDS:
        parser.add_argument(
            "--" + field_name.removeprefix("expected_").replace("_", "-")
        )
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
        parser.error(f"generator API key environment variable {variable!r} is unset")
    return value


def _preparation_record(prepared) -> dict[str, object]:
    contract = prepared.contract
    return {
        "schema_version": "decomposition-key-aligned-repair-preparation-v1",
        "repair_contract": contract.to_record(),
        "repair_contract_sha256": contract.sha256,
        "source_e26_repair_contract": prepared.source_probe.contract.to_record(),
        "source_e26_repair_contract_sha256": contract.e26_repair_contract_sha256,
        "source_e26_summary": dict(prepared.source_summary),
        "source_e26_summary_sha256": contract.e26_summary_sha256,
        "e26_artifact_sha256": contract.e26_artifact_sha256,
        "e26_result_manifest_sha256": contract.e26_result_manifest_sha256,
        "source_environment_contract_sha256": (
            contract.source_environment_contract_sha256
        ),
        "request_environment_contract_sha256": (
            contract.request_environment_contract_sha256
        ),
        "scoring_environment_contract_sha256": (
            contract.scoring_environment_contract_sha256
        ),
        "corrected_profile_sha256": contract.corrected_profile_sha256,
        "feedback_map": key_aligned_repair_feedback_map_record(),
        "feedback_map_sha256": contract.feedback_map_sha256,
        "case_manifest_sha256": contract.case_manifest_sha256,
        "base_request_manifest_sha256": contract.base_request_manifest_sha256,
        "e26_repair_request_manifest_sha256": (
            contract.e26_repair_request_manifest_sha256
        ),
        "key_aligned_request_manifest_sha256": (
            contract.key_aligned_request_manifest_sha256
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
    problem_source = CaseworkProblemSource.from_jsonl(args.problems).select_slice(
        args.start_index, args.episode_count
    )
    prepared = prepare_key_aligned_repair_probe(
        source_run_path=args.source_run,
        source_diagnostic_path=args.source_diagnostic,
        e24_artifact_path=args.e24_artifact,
        e25_artifact_path=args.e25_artifact,
        e26_artifact_path=args.e26_artifact,
        e26_authorization=_sealed_e26_authorization(),
        expected_e26_artifact_sha256=_E26_ARTIFACT_SHA256,
        problem_source=problem_source,
        planned_output_path=args.output,
    )
    contract = prepared.contract
    if contract.source_decision_count != args.expected_source_decision_count:
        parser.error("strict-replayed decision count differs from expected count")
    if contract.case_count != args.expected_decompose_count:
        parser.error("selected DECOMPOSE count differs from expected count")
    if contract.problem_count != args.expected_problem_count:
        parser.error("distinct problem count differs from expected count")
    preparation = _preparation_record(prepared)
    if args.print_contract:
        print(canonical_json_bytes(preparation).decode("ascii"))
        return 0
    if args.preregistration_id is None or not args.preregistration_id.strip():
        parser.error("--preregistration-id is required for hosted execution")
    digest_values = {}
    for field_name in _E27_DIGEST_FIELDS:
        argument_name = field_name.removeprefix("expected_")
        flag = "--" + argument_name.replace("_", "-")
        digest_values[field_name] = _required_digest(
            parser, flag, getattr(args, argument_name)
        )
    authorization = KeyAlignedRepairProbeAuthorization(
        preregistration_id=args.preregistration_id,
        **digest_values,
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
        parser.error("key-aligned output or partial output already exists")
    if not args.output.parent.exists() or not args.output.parent.is_dir():
        parser.error("key-aligned output directory does not exist")
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
    result = run_key_aligned_repair_probe(
        output_path=args.output,
        prepared=prepared,
        generator_client=client,
        authorization=authorization,
    )
    print(
        canonical_json_bytes(
            {
                "schema_version": "decomposition-key-aligned-repair-completion-v1",
                "output": str(result.output_path),
                "repair_contract_sha256": result.contract_sha256,
                "case_count": result.case_count,
                "wire_usable_count": result.wire_usable_count,
                "typed_decoded_count": result.decoded_count,
                "symbolic_applied_count": result.symbolic_applied_count,
                "applied_problem_count": result.applied_problem_count,
                "generator_generate_calls": client.generate_calls,
                "generator_prompt_tokens": client.prompt_tokens,
                "generator_generated_tokens": client.generated_tokens,
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
