"""Prepare or execute a shared-direct-prefix teacher comparison."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient
from verifier_anchored_rl.workspace_v1.controller import TEACHER_CHOICE_MODES
from verifier_anchored_rl.workspace_v1.experiment import (
    ModelIdentity,
    load_problem_examples,
)
from verifier_anchored_rl.workspace_v1.lenses import (
    CHILD_CONTEXT_MODES,
    REVISE_RESULT_MODES,
    REVISE_RESULT_OPEN_OBJECT,
    ROOT_RESULT_MODES,
)
from verifier_anchored_rl.workspace_v1.model import ACTION_AVAILABILITY_MODES
from verifier_anchored_rl.workspace_v1.paired_direct_teacher import (
    build_paired_direct_teacher_contract,
    collect_paired_direct_teacher_run,
)
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
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an exact shared-direct-Candidate comparison or execute "
            "it after preregistration. Preparation makes no network call."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-contracts", action="store_true")
    mode.add_argument("--execute-hosted", action="store_true")
    parser.add_argument("--preregistration-id", required=True)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--episode-count", type=int, required=True)
    parser.add_argument("--turn-limit", type=int, required=True)
    parser.add_argument("--max-depth", type=int, required=True)
    parser.add_argument(
        "--child-context-mode",
        choices=tuple(sorted(CHILD_CONTEXT_MODES)),
        required=True,
    )
    parser.add_argument(
        "--root-result-mode",
        choices=tuple(sorted(ROOT_RESULT_MODES)),
        required=True,
    )
    parser.add_argument(
        "--action-availability-mode",
        choices=tuple(sorted(ACTION_AVAILABILITY_MODES)),
        required=True,
    )
    parser.add_argument(
        "--teacher-choice-mode",
        choices=tuple(sorted(TEACHER_CHOICE_MODES)),
        required=True,
    )
    parser.add_argument(
        "--revise-result-mode",
        choices=tuple(sorted(REVISE_RESULT_MODES)),
        default=REVISE_RESULT_OPEN_OBJECT,
    )
    parser.add_argument("--run-output", type=Path)
    parser.add_argument("--diagnostic-output", type=Path)
    parser.add_argument("--expected-problem-source-sha256")
    parser.add_argument("--expected-evaluation-binding-sha256")
    parser.add_argument("--expected-run-contract-sha256")
    _add_role_arguments(parser, "worker")
    _add_role_arguments(parser, "teacher")
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


def _secret(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    role: str,
) -> str:
    variable = getattr(args, f"{role}_api_key_env")
    if variable is None or not variable.strip():
        parser.error(f"--{role}-api-key-env is required for execution")
    value = os.environ.get(variable)
    if value is None or not value.strip():
        parser.error(f"{role} API key environment variable is unset")
    return value


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


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        examples = load_problem_examples(
            args.problems,
            start_index=args.start_index,
            episode_count=args.episode_count,
        )
        worker_identity = _identity(args, "worker")
        teacher_identity = _identity(args, "teacher")
        contract = build_paired_direct_teacher_contract(
            preregistration_id=args.preregistration_id,
            examples=examples,
            turn_limit=args.turn_limit,
            max_depth=args.max_depth,
            child_context_mode=args.child_context_mode,
            root_result_mode=args.root_result_mode,
            action_availability_mode=args.action_availability_mode,
            teacher_choice_mode=args.teacher_choice_mode,
            worker=worker_identity,
            worker_temperature=args.worker_temperature,
            worker_max_tokens=args.worker_max_tokens,
            worker_seed=args.worker_seed,
            teacher=teacher_identity,
            teacher_temperature=args.teacher_temperature,
            teacher_max_tokens=args.teacher_max_tokens,
            teacher_seed=args.teacher_seed,
            revise_result_mode=args.revise_result_mode,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))

    preparation = {
        "schema_version": (
            "workspace-paired-direct-teacher-preparation-v1.1"
            if contract.revise_result_mode != REVISE_RESULT_OPEN_OBJECT
            else "workspace-paired-direct-teacher-preparation-v1"
        ),
        "preregistration_id": contract.preregistration_id,
        "case_count": len(examples),
        "problem_ids": list(contract.problem_ids),
        "problem_source_sha256": contract.problem_source_sha256,
        "evaluation_binding_sha256": contract.evaluation_binding_sha256,
        "run_contract_sha256": contract.sha256,
        "settings": contract.header()["settings"],
    }
    if args.print_contracts:
        print(canonical_json_bytes(preparation).decode("ascii"))
        return 0

    for flag, supplied, actual in (
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
    worker = _client(
        worker_identity,
        _secret(parser, args, "worker"),
    )
    teacher = _client(
        teacher_identity,
        _secret(parser, args, "teacher"),
    )
    result = collect_paired_direct_teacher_run(
        run_path=args.run_output,
        diagnostic_path=args.diagnostic_output,
        contract=contract,
        examples=examples,
        worker_client=worker,
        teacher_client=teacher,
    )
    summary = {
        "schema_version": (
            "workspace-paired-direct-teacher-execution-v1.1"
            if contract.revise_result_mode != REVISE_RESULT_OPEN_OBJECT
            else "workspace-paired-direct-teacher-execution-v1"
        ),
        "preregistration_id": contract.preregistration_id,
        "run_contract_sha256": result.run_contract_sha256,
        "case_count": result.case_count,
        "transition_count": result.transition_count,
        "solved_by_branch": dict(result.solved_by_branch),
        "run_path": str(result.run_path),
        "diagnostic_path": str(result.diagnostic_path),
    }
    print(canonical_json_bytes(summary).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
