"""Run a trained Phase B controller greedily at B=0 with a frozen generator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from verifier_anchored_rl.controller_observation import (
    CONTROLLER_OBSERVATION_VERSION,
)
from verifier_anchored_rl.deliberation_mdp import (
    DeliberationMDP,
    LEGACY_MASK_PENALTY_REWARD,
    SYMBOLIC_ONLY_REWARD,
)
from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient
from verifier_anchored_rl.llm_controller import run_episode
from verifier_anchored_rl.phase_b_policy import (
    OllamaBatchTextFeatures,
    load_learned_controller,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--model", default="qwen2.5:1.5b-instruct")
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--embedding-base-url", default="http://localhost:11434")
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest")
    parser.add_argument("--embedding-dim", type=int, default=768)
    parser.add_argument("--embedding-model-digest")
    parser.add_argument("--depth-limit", type=int, default=16)
    parser.add_argument(
        "--decision-rule", choices=("joint", "sequential"), default=None)
    parser.add_argument(
        "--observation-schema",
        choices=("tree-render-v1", CONTROLLER_OBSERVATION_VERSION),
        default=None,
    )
    parser.add_argument(
        "--reject-repeated-verify",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--active-subtrees-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--render-committed-outcomes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--generation-temperature", type=float, default=1.0)
    parser.add_argument(
        "--solve-claims",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--trusted-givens-committable",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--reward-contract",
        choices=(SYMBOLIC_ONLY_REWARD, LEGACY_MASK_PENALTY_REWARD),
        default=SYMBOLIC_ONLY_REWARD,
    )
    parser.add_argument(
        "--allow-distribution-shift",
        action="store_true",
        help="Allow only generator temperature/lens differences.",
    )
    parser.add_argument(
        "--allow-unbound-legacy-checkpoint",
        action="store_true",
        help="Required for old tree-render checkpoints without MDP metadata.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-problems", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    problems = [
        json.loads(line)
        for line in args.problems.read_text().splitlines()
        if line.strip()
    ]
    indexed = list(enumerate(problems))[args.start_index :]
    if args.max_problems is not None:
        indexed = indexed[: args.max_problems]

    api_key = (
        os.environ.get(args.api_key_env)
        if args.api_key_env is not None
        else "ollama"
    )
    if not api_key:
        raise SystemExit(f"need a key in ${args.api_key_env}")
    generator = OpenAICompatibleLLMClient(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
    )
    encoder = OllamaBatchTextFeatures(
        model=args.embedding_model,
        base_url=args.embedding_base_url,
        dim=args.embedding_dim,
        model_digest=args.embedding_model_digest,
    )
    served_embedding_digest = encoder.pin_served_model_digest()
    controller = load_learned_controller(
        str(args.checkpoint),
        encoder,
        depth_limit=args.depth_limit,
        decision_rule=args.decision_rule,
        observation_schema=args.observation_schema,
    )
    wins = 0
    with args.out.open("w") as fh:
        for problem_index, problem in indexed:
            records: list[dict] = []
            rng = np.random.default_rng(args.seed + problem_index)
            mdp = DeliberationMDP(
                problem=problem["problem"],
                expected_answer=problem["answer"],
                depth_limit=args.depth_limit,
                llm_client=generator,
                solve_claims=args.solve_claims,
                reject_repeated_verify=args.reject_repeated_verify,
                active_subtrees_only=args.active_subtrees_only,
                render_committed_outcomes=args.render_committed_outcomes,
                generation_temperature=args.generation_temperature,
                reward_contract=args.reward_contract,
                trusted_givens_committable=(
                    args.trusted_givens_committable
                ),
            )
            contract_match = controller.validate_mdp_contract(
                mdp,
                allow_distribution_shift=args.allow_distribution_shift,
                allow_unbound_legacy=(
                    args.allow_unbound_legacy_checkpoint
                ),
            )
            summary = run_episode(
                mdp,
                controller,
                rng,
                log_fn=records.append,
                episode_id=problem_index,
                meta={
                    "checkpoint": str(args.checkpoint),
                    "generator_model": args.model,
                    "embedding_model": args.embedding_model,
                    "seed": args.seed,
                    "problem_index": problem_index,
                    "budget": 0,
                    "decision_rule": controller.decision_rule,
                    "observation_schema": controller.observation_schema,
                    "action_semantics": controller.action_semantics,
                    "contract_sha256": controller.contract_sha256,
                    "embedding_model_digest": served_embedding_digest,
                    "reject_repeated_verify": args.reject_repeated_verify,
                    "active_subtrees_only": args.active_subtrees_only,
                    "render_committed_outcomes": args.render_committed_outcomes,
                    "generation_temperature": args.generation_temperature,
                    "solve_claims": args.solve_claims,
                    "reward_contract": args.reward_contract,
                    "trusted_givens_committable": (
                        args.trusted_givens_committable
                    ),
                    "contract_match": contract_match,
                    "declared_distribution_shift": (
                        args.allow_distribution_shift
                    ),
                    "unbound_legacy_override": (
                        args.allow_unbound_legacy_checkpoint
                    ),
                },
            )
            for record in records:
                fh.write(json.dumps(record) + "\n")
            fh.flush()
            wins += summary.reward > 0.0
            print(
                f"[{problem_index}] reward={summary.reward:+.2f} "
                f"turns={summary.turns} problem={problem['problem'][:60]!r}",
                flush=True,
            )
    print(
        f"episodes={len(indexed)} wins={wins} "
        f"prompt_tokens={generator.prompt_tokens} "
        f"completion_tokens={generator.generated_tokens}",
        flush=True,
    )


if __name__ == "__main__":
    main()
