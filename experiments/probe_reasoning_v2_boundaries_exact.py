"""Run the preregistered E29 exact-schema boundary replication."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

if __package__:
    from experiments.probe_reasoning_v2_boundaries import (
        _E26_ARTIFACT_SHA256,
        _E27_ARTIFACT_SHA256,
        _sealed_e26_authorization,
        _sealed_e27_authorization,
    )
else:
    from probe_reasoning_v2_boundaries import (
        _E26_ARTIFACT_SHA256,
        _E27_ARTIFACT_SHA256,
        _sealed_e26_authorization,
        _sealed_e27_authorization,
    )
from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient
from verifier_anchored_rl.reasoning_v2.boundary_exact_probe import (
    ExactBoundaryProbeAuthorization,
    exact_boundary_retry_policy_record,
    exact_boundary_schema_validation_record,
    prepare_exact_boundary_probe,
    run_exact_boundary_probe,
    validate_exact_boundary_oracle_cases,
)
from verifier_anchored_rl.reasoning_v2.boundary_probe import (
    BoundaryDecompositionProbeAuthorization,
)
from verifier_anchored_rl.reasoning_v2.casework_source import (
    CaseworkProblemSource,
)
from verifier_anchored_rl.reasoning_v2.contracts import canonical_json_bytes


_SHA256 = re.compile(r"[0-9a-f]{64}")
_E28_ARTIFACT_SHA256 = (
    "f42e36bc426169ca2edb25237c26443e40eea7952b10242d25590e45bb96f404"
)
_E29_DIGEST_FIELDS = tuple(
    name for name in ExactBoundaryProbeAuthorization.__dataclass_fields__
    if name.startswith("expected_") and name.endswith("sha256")
)


def _sealed_e28_authorization() -> BoundaryDecompositionProbeAuthorization:
    return BoundaryDecompositionProbeAuthorization(
        preregistration_id="E28-v2-boundary-decompose33",
        expected_source_environment_contract_sha256=(
            "9316c93bc2cdf9b2ce507a989882af69669f8bf886e1f2ad6ce802c77dfdcfa7"
        ),
        expected_request_environment_contract_sha256=(
            "664deedf1caa44efff16daac1888c2e3966aa37e9a2631a62036643a9e681c71"
        ),
        expected_scoring_environment_contract_sha256=(
            "3396ce7464c8bf4a8cd60c5aa26172c02dc4ed4589ca7eb9e2bcfd906119ee54"
        ),
        expected_corrected_profile_sha256=(
            "42ee4bf1f481dd92bebac990b8fc0c95d280a4a5c1054b0b9dc4d5070f6581c8"
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
        expected_e27_artifact_sha256=_E27_ARTIFACT_SHA256,
        expected_e27_repair_contract_sha256=(
            "e3c4fc625c1cb79e2217a2fb86d8603a3b9955e9bac1fd2e59cb47fada078696"
        ),
        expected_e27_case_manifest_sha256=(
            "5a6b78190e0a54a39ad32e23e6570cdb93c8adc14c52efc3f8696574e2999977"
        ),
        expected_e27_summary_sha256=(
            "e936a50d74cbfa67559d248e1c12ef3db6021408bbf333d8ae9ebc350429726b"
        ),
        expected_boundary_instruction_sha256=(
            "6004c2f9324b15321228201d96bdbdc421052811aa70064a33d27baf629b47ec"
        ),
        expected_boundary_payload_schema_sha256=(
            "58743116168367a400b9a15af8d336caef850f9e60be62e2fa01782d4e9c31e9"
        ),
        expected_boundary_constructor_sha256=(
            "6c9873799b9b1dd54eb11c523823e2cac0fb57cdc46a78c2366915f60aa20dfb"
        ),
        expected_boundary_oracle_manifest_sha256=(
            "b4af1720ed561f3c65fa44f3092e8aada05130121151aa91fe24ecbf1e5fe3d3"
        ),
        expected_case_manifest_sha256=(
            "c8b89f7f25a07d3519404e4e30e7d477bf7be6a8f108e4386e22a9ce26ea5052"
        ),
        expected_base_request_manifest_sha256=(
            "ddc854aef8431f78550315f45a99633e67bb32e8d77bef06d6e21b612167286b"
        ),
        expected_boundary_request_manifest_sha256=(
            "d78bb4976c3f0b8098a87260abc1d96bed4c599cb89a4b1b4f668558aa436c3d"
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
            "6dbfd0e04b0c702ad1cc5342868d932a97cb56c904bb98bc7fb1dc45e993ca06"
        ),
        expected_source_decision_count=66,
        expected_case_count=33,
        expected_problem_count=5,
        expected_output_path=(
            "experiments/runs/2026-07-13-E28-v2-boundary-decompose33.jsonl"
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the invalidated E28 control, preserve its 33 initial "
            "requests, and run the exact no-coercion boundary replication."
        ),
        epilog=(
            "This diagnostic creates no controller decision, trajectory, "
            "closure, reward, or training row."
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
    parser.add_argument("--e27-artifact", type=Path, required=True)
    parser.add_argument("--e28-artifact", type=Path, required=True)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--episode-count", type=int, required=True)
    parser.add_argument("--expected-source-decision-count", type=int, required=True)
    parser.add_argument("--expected-decompose-count", type=int, required=True)
    parser.add_argument("--expected-problem-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preregistration-id")
    for field_name in _E29_DIGEST_FIELDS:
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
        parser.error(
            f"generator API key environment variable {variable!r} is unset"
        )
    return value


def _preparation_record(prepared) -> dict[str, object]:
    contract = prepared.contract
    return {
        "schema_version": "decomposition-boundary-exact-preparation-v1",
        "exact_boundary_contract": contract.to_record(),
        "exact_boundary_contract_sha256": contract.sha256,
        "source_e28_contract": prepared.source_e28_probe.contract.to_record(),
        "source_e28_contract_sha256": (
            contract.invalidated_e28_probe_contract_sha256
        ),
        "invalidated_e28_raw_audit": (
            prepared.invalidated_e28_raw_audit.to_record()
        ),
        "invalidated_e28_raw_audit_sha256": (
            contract.invalidated_e28_raw_audit_sha256
        ),
        "invalidated_e28_raw_controls": [
            control.to_record()
            for control in prepared.invalidated_e28_raw_controls
        ],
        "invalidated_e28_raw_control_manifest_sha256": (
            contract.invalidated_e28_raw_control_manifest_sha256
        ),
        "schema_validation": exact_boundary_schema_validation_record(),
        "schema_validation_contract_sha256": (
            contract.schema_validation_contract_sha256
        ),
        "retry_policy": exact_boundary_retry_policy_record(),
        "retry_policy_contract_sha256": (
            contract.retry_policy_contract_sha256
        ),
        "boundary_oracle_manifest": validate_exact_boundary_oracle_cases(
            prepared
        ),
        "boundary_oracle_manifest_sha256": (
            contract.boundary_oracle_manifest_sha256
        ),
        "case_manifest_sha256": contract.case_manifest_sha256,
        "base_request_manifest_sha256": contract.base_request_manifest_sha256,
        "boundary_request_manifest_sha256": (
            contract.boundary_request_manifest_sha256
        ),
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
    prepared = prepare_exact_boundary_probe(
        source_run_path=args.source_run,
        source_diagnostic_path=args.source_diagnostic,
        e24_artifact_path=args.e24_artifact,
        e25_artifact_path=args.e25_artifact,
        e26_artifact_path=args.e26_artifact,
        e27_artifact_path=args.e27_artifact,
        e28_artifact_path=args.e28_artifact,
        e26_authorization=_sealed_e26_authorization(),
        e27_authorization=_sealed_e27_authorization(),
        e28_authorization=_sealed_e28_authorization(),
        expected_e26_artifact_sha256=_E26_ARTIFACT_SHA256,
        expected_e27_artifact_sha256=_E27_ARTIFACT_SHA256,
        expected_e28_artifact_sha256=_E28_ARTIFACT_SHA256,
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
    if args.print_contract:
        print(canonical_json_bytes(_preparation_record(prepared)).decode("ascii"))
        return 0
    if args.preregistration_id is None or not args.preregistration_id.strip():
        parser.error("--preregistration-id is required for hosted execution")
    digest_values = {}
    for field_name in _E29_DIGEST_FIELDS:
        argument_name = field_name.removeprefix("expected_")
        flag = "--" + argument_name.replace("_", "-")
        digest_values[field_name] = _required_digest(
            parser, flag, getattr(args, argument_name)
        )
    authorization = ExactBoundaryProbeAuthorization(
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
        parser.error("exact boundary output or partial output already exists")
    if not args.output.parent.exists() or not args.output.parent.is_dir():
        parser.error("exact boundary output directory does not exist")
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
    result = run_exact_boundary_probe(
        output_path=args.output,
        prepared=prepared,
        generator_client=client,
        authorization=authorization,
    )
    print(canonical_json_bytes({
        "schema_version": "decomposition-boundary-exact-completion-v1",
        "output": str(result.output_path),
        "exact_boundary_contract_sha256": result.contract_sha256,
        "case_count": result.case_count,
        "wire_usable_count": result.wire_usable_count,
        "typed_decoded_count": result.decoded_count,
        "symbolic_applied_count": result.symbolic_applied_count,
        "applied_problem_count": result.applied_problem_count,
        "generator_generate_calls": client.generate_calls,
        "generator_prompt_tokens": client.prompt_tokens,
        "generator_generated_tokens": client.generated_tokens,
    }).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
