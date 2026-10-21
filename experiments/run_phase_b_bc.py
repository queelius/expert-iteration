"""Run the first grouped-holdout Phase B behavior-cloning diagnostic.

The policy loss uses successful, non-fallback teacher decisions.  The critic
uses every training transition and its offline Monte Carlo return.  This first
run uses deterministic hashed lexical features to validate the full learning
path before the frozen-LLM encoder is introduced as a separate manipulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verifier_anchored_rl.heads import NUM_VERBS, VERB_NAME_TO_INDEX, VERB_ORDER
from verifier_anchored_rl.deliberation_mdp import (
    LEGACY_MASKED_ACTION_REWARD,
    LEGACY_MASK_PENALTY_REWARD,
    MASKED_ACTION_REWARD,
    SYMBOLIC_ONLY_REWARD,
)
from verifier_anchored_rl.controller_observation import (
    CONTROLLER_OBSERVATION_VERSION,
)
from verifier_anchored_rl.phase_b_baselines import legal_menu_lookup_baseline
from verifier_anchored_rl.phase_b_features import (
    CachedTextFeatures,
    HashingTextFeatures,
    ZeroTextFeatures,
    STRUCTURAL_FEATURE_DIM,
    feature_contract_for_schema,
    structural_scalar_features,
)
from verifier_anchored_rl.phase_b_heads import (
    FactoredController,
    behavior_cloning_loss,
    selected_q_loss,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def validate_observation_schema(
    rows: list[dict[str, Any]], requested: str | None
) -> str:
    """Infer one dataset schema and reject user-stamped mismatches."""
    schemas = {row.get("serialization_schema") for row in rows}
    if None in schemas or len(schemas) != 1:
        raise ValueError(f"dataset has mixed or missing schemas: {schemas!r}")
    dataset_schema = next(iter(schemas))
    if dataset_schema not in {
        "tree-render-v1", CONTROLLER_OBSERVATION_VERSION
    }:
        raise ValueError(f"unsupported dataset schema {dataset_schema!r}")
    if requested is not None and requested != dataset_schema:
        raise ValueError(
            f"requested schema {requested!r} != dataset {dataset_schema!r}")
    return dataset_schema


def validate_training_contracts(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reject missing or falsely unified action and transition contracts."""
    semantics = {row.get("action_semantics") for row in rows}
    if None in semantics or len(semantics) != 1:
        raise ValueError(
            f"dataset has mixed or missing action semantics: {semantics!r}")
    action_semantics = next(iter(semantics))

    verb_orders = {
        tuple(row.get("verb_order", ()))
        for row in rows
    }
    if len(verb_orders) != 1 or not next(iter(verb_orders)):
        raise ValueError("dataset has mixed or missing verb order")
    verb_order = list(next(iter(verb_orders)))
    runtime_order = [verb.value for verb in VERB_ORDER]
    if verb_order != runtime_order:
        raise ValueError(
            f"dataset verb order {verb_order!r} != runtime {runtime_order!r}")

    outcome_schemas = {
        row.get("previous_outcome_schema") for row in rows
    }
    if None in outcome_schemas or len(outcome_schemas) != 1:
        raise ValueError(
            "dataset has mixed or missing previous-outcome schemas")

    contract_by_json: dict[str, dict[str, Any]] = {}
    for row in rows:
        contract = row.get("mdp_contract")
        if not isinstance(contract, dict):
            raise ValueError("dataset row has no MDP contract")
        if contract.get("action_semantics") != action_semantics:
            raise ValueError("MDP contract action semantics differ from row")
        if (
            contract.get("previous_outcome_schema")
            != row["previous_outcome_schema"]
        ):
            raise ValueError(
                "MDP contract outcome schema differs from row")
        if row["masked"]:
            expected_mask_reward = {
                LEGACY_MASK_PENALTY_REWARD: LEGACY_MASKED_ACTION_REWARD,
                SYMBOLIC_ONLY_REWARD: MASKED_ACTION_REWARD,
            }.get(contract.get("reward_contract"))
            if expected_mask_reward is None:
                raise ValueError("dataset has an unknown reward contract")
            if float(row["reward"]) != expected_mask_reward:
                raise ValueError(
                    "masked reward differs from declared reward contract")
        key = json.dumps(contract, sort_keys=True, separators=(",", ":"))
        contract_by_json[key] = contract

    gammas = {float(row["gamma"]) for row in rows}
    if len(gammas) != 1:
        raise ValueError(f"dataset has mixed gammas: {sorted(gammas)!r}")

    return {
        "action_semantics": action_semantics,
        "verb_order": verb_order,
        "previous_outcome_schema": next(iter(outcome_schemas)),
        "mdp_contracts": [
            contract_by_json[key] for key in sorted(contract_by_json)
        ],
        "gamma": next(iter(gammas)),
    }


def encode_row(row: dict[str, Any], encoder: Any) -> dict[str, Any]:
    node_index = {node["id"]: index for index, node in enumerate(row["nodes"])}
    feature_dim = encoder.dim + STRUCTURAL_FEATURE_DIM
    global_features = np.concatenate(
        (encoder.encode(row["state_text"]), structural_scalar_features(row))
    ).astype(np.float32)
    node_features = np.zeros((len(row["nodes"]), feature_dim), dtype=np.float32)
    state_scalars = structural_scalar_features(row)
    for index, node in enumerate(row["nodes"]):
        node_features[index] = np.concatenate(
            (encoder.encode_node(node), state_scalars)
        )
    legal = np.zeros((NUM_VERBS, len(row["nodes"])), dtype=np.bool_)
    for pair in row["legal_pairs"]:
        legal[VERB_NAME_TO_INDEX[pair["verb"]], node_index[pair["target"]]] = True
    chosen = row["chosen"]
    chosen_node = node_index[chosen["target"]]
    chosen_rendered = row["nodes"][chosen_node]["rendered"]
    return {
        "row": row,
        "global": global_features,
        "nodes": node_features,
        "legal": legal,
        "chosen_verb": VERB_NAME_TO_INDEX[chosen["verb"]],
        "chosen_node": chosen_node,
        "return": float(row["mc_return"]),
        "policy_label": bool(row["bc_eligible"] and row["episode_won"]),
        "eval_label": bool(row["bc_eligible"]),
        "chosen_rendered": bool(chosen_rendered),
    }


def collate(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    batch = len(samples)
    feature_dim = samples[0]["global"].shape[0]
    max_nodes = max(sample["nodes"].shape[0] for sample in samples)
    global_features = torch.zeros(batch, feature_dim)
    node_features = torch.zeros(batch, max_nodes, feature_dim)
    legal = torch.zeros(batch, NUM_VERBS, max_nodes, dtype=torch.bool)
    chosen_verbs = torch.zeros(batch, dtype=torch.long)
    chosen_nodes = torch.zeros(batch, dtype=torch.long)
    returns = torch.zeros(batch)
    policy_labels = torch.zeros(batch, dtype=torch.bool)
    for index, sample in enumerate(samples):
        count = sample["nodes"].shape[0]
        global_features[index] = torch.from_numpy(sample["global"])
        node_features[index, :count] = torch.from_numpy(sample["nodes"])
        legal[index, :, :count] = torch.from_numpy(sample["legal"])
        chosen_verbs[index] = sample["chosen_verb"]
        chosen_nodes[index] = sample["chosen_node"]
        returns[index] = sample["return"]
        policy_labels[index] = sample["policy_label"]
    return {
        "global": global_features,
        "nodes": node_features,
        "legal": legal,
        "chosen_verbs": chosen_verbs,
        "chosen_nodes": chosen_nodes,
        "returns": returns,
        "policy_labels": policy_labels,
    }


def metrics(
    model: FactoredController,
    samples: list[dict[str, Any]],
    decision_rule: str = "joint",
) -> dict[str, Any]:
    eligible = [sample for sample in samples if sample["eval_label"]]
    batch = collate(eligible)
    model.eval()
    with torch.no_grad():
        verb_prob, pointer_prob = model.masked_policy(
            batch["global"], batch["nodes"], batch["legal"]
        )
        row_index = torch.arange(len(eligible))
        if decision_rule == "joint":
            joint = verb_prob.unsqueeze(-1) * pointer_prob
            flat = joint.reshape(len(eligible), -1).argmax(dim=-1)
            node_count = pointer_prob.shape[-1]
            predicted_verbs = torch.div(
                flat, node_count, rounding_mode="floor")
            predicted_nodes = flat % node_count
        elif decision_rule == "sequential":
            predicted_verbs = verb_prob.argmax(dim=-1)
            predicted_nodes = pointer_prob[
                row_index, predicted_verbs].argmax(dim=-1)
        else:
            raise ValueError(f"unknown decision rule {decision_rule!r}")
        gold_pointer_nodes = pointer_prob[
            row_index, batch["chosen_verbs"]
        ].argmax(dim=-1)
    verb_correct = predicted_verbs == batch["chosen_verbs"]
    pointer_correct = gold_pointer_nodes == batch["chosen_nodes"]
    joint_correct = verb_correct & (predicted_nodes == batch["chosen_nodes"])
    rendered = torch.tensor([sample["chosen_rendered"] for sample in eligible])

    def rate(mask: torch.Tensor, correct: torch.Tensor) -> float | None:
        count = int(mask.sum())
        return float(correct[mask].float().mean()) if count else None

    return {
        "decision_rule": decision_rule,
        "decisions": len(eligible),
        "verb_accuracy": float(verb_correct.float().mean()),
        "pointer_accuracy_given_gold_verb": float(pointer_correct.float().mean()),
        "joint_accuracy": float(joint_correct.float().mean()),
        "rendered_decisions": int(rendered.sum()),
        "rendered_joint_accuracy": rate(rendered, joint_correct),
        "winning_decisions": sum(sample["row"]["episode_won"] for sample in eligible),
        "winning_joint_accuracy": rate(
            torch.tensor([sample["row"]["episode_won"] for sample in eligible]),
            joint_correct,
        ),
    }


def menu_frequency_baseline(
    train_samples: list[dict[str, Any]], eval_samples: list[dict[str, Any]]
) -> dict[str, float]:
    counts = Counter(
        sample["chosen_verb"] for sample in train_samples if sample["policy_label"]
    )
    correct = 0
    chance = 0.0
    eligible = [sample for sample in eval_samples if sample["eval_label"]]
    for sample in eligible:
        legal = sample["legal"]
        legal_verbs = [verb for verb in range(NUM_VERBS) if legal[verb].any()]
        predicted_verb = max(legal_verbs, key=lambda verb: (counts[verb], -verb))
        predicted_node = int(np.flatnonzero(legal[predicted_verb])[0])
        correct += (
            predicted_verb == sample["chosen_verb"]
            and predicted_node == sample["chosen_node"]
        )
        chance += 1.0 / float(legal.sum())
    return {
        "menu_frequency_joint_accuracy": correct / len(eligible),
        "uniform_legal_pair_expected_accuracy": chance / len(eligible),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = load_jsonl(args.dataset)
    args.observation_schema = validate_observation_schema(
        rows, args.observation_schema)
    training_contract = validate_training_contracts(rows)
    if args.encoder == "cache":
        if args.embedding_cache is None:
            raise ValueError("--encoder cache requires --embedding-cache")
        encoder = CachedTextFeatures(str(args.embedding_cache))
    elif args.encoder == "zero":
        encoder = ZeroTextFeatures(args.zero_dim)
    else:
        encoder = HashingTextFeatures(args.hash_dim)
    samples = [encode_row(row, encoder) for row in rows]
    train_samples = [
        sample for sample in samples if sample["row"]["problem_index"] < args.holdout_start
    ]
    eval_samples = [
        sample for sample in samples if sample["row"]["problem_index"] >= args.holdout_start
    ]
    if not train_samples or not eval_samples:
        raise ValueError("grouped split produced an empty partition")

    feature_dim = encoder.dim + STRUCTURAL_FEATURE_DIM
    model = FactoredController(
        feature_dim,
        hidden_dim=args.hidden_dim,
        repr_dim=args.repr_dim,
        node_repr_dim=args.node_repr_dim,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    history = []
    pre = metrics(model, eval_samples, args.decision_rule)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_samples)))
        rng.shuffle(order)
        totals = Counter()
        steps = 0
        for start in range(0, len(order), args.batch_size):
            selected = [train_samples[index] for index in order[start : start + args.batch_size]]
            batch = collate(selected)
            optimizer.zero_grad()
            q_loss = selected_q_loss(
                model,
                batch["global"],
                batch["nodes"],
                batch["chosen_verbs"],
                batch["chosen_nodes"],
                batch["returns"],
            )
            policy_mask = batch["policy_labels"]
            if policy_mask.any():
                bc_loss, verb_loss, pointer_loss = behavior_cloning_loss(
                    model,
                    batch["global"][policy_mask],
                    batch["nodes"][policy_mask],
                    batch["legal"][policy_mask],
                    batch["chosen_verbs"][policy_mask],
                    batch["chosen_nodes"][policy_mask],
                )
            else:
                bc_loss = q_loss.new_zeros(())
                verb_loss = q_loss.new_zeros(())
                pointer_loss = q_loss.new_zeros(())
            loss = bc_loss + args.q_weight * q_loss
            loss.backward()
            optimizer.step()
            totals.update(
                total=float(loss.detach()),
                bc=float(bc_loss.detach()),
                verb=float(verb_loss.detach()),
                pointer=float(pointer_loss.detach()),
                q=float(q_loss.detach()),
            )
            steps += 1
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            history.append(
                {"epoch": epoch, **{key: value / steps for key, value in totals.items()}}
            )

    if isinstance(encoder, ZeroTextFeatures):
        encoder_record = {
            "type": "zero_text_ablation",
            "dimension": encoder.dim,
        }
    elif isinstance(encoder, CachedTextFeatures):
        encoder_record = {
            "type": "frozen_embedding_cache",
            "model": encoder.model,
            "model_digest": encoder.model_digest,
            "dimension": encoder.dim,
        }
    else:
        encoder_record = {"type": "hash", "dimension": encoder.dim}

    feature_contract = feature_contract_for_schema(args.observation_schema)
    training_target_contract = {
        "target_type": "monte-carlo-return-v1",
        "gamma": training_contract["gamma"],
        "reward_contracts": sorted({
            contract["reward_contract"]
            for contract in training_contract["mdp_contracts"]
        }),
        "acting_policy": "behavior-cloning-heads-v1",
    }
    model_contract = {
        "hidden_dim": args.hidden_dim,
        "repr_dim": args.repr_dim,
        "node_repr_dim": args.node_repr_dim,
    }
    contract_manifest = {
        "observation_schema": args.observation_schema,
        "previous_outcome_schema": training_contract[
            "previous_outcome_schema"],
        "action_semantics": training_contract["action_semantics"],
        "verb_order": training_contract["verb_order"],
        "training_mdp_contracts": training_contract["mdp_contracts"],
        "feature_contract": feature_contract,
        "encoder": encoder_record,
        "training_target_contract": training_target_contract,
        "policy_contract": {
            "decision_rule": args.decision_rule,
            "acting_score": "behavior-cloning-probability-v1",
            "model_config": model_contract,
        },
    }
    contract_sha256 = hashlib.sha256(
        json.dumps(
            contract_manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()

    result = {
        "experiment": args.experiment,
        "observation_schema": args.observation_schema,
        "action_semantics": training_contract["action_semantics"],
        "verb_order": training_contract["verb_order"],
        "previous_outcome_schema": (
            training_contract["previous_outcome_schema"]
        ),
        "training_mdp_contracts": training_contract["mdp_contracts"],
        "feature_contract": feature_contract,
        "training_target_contract": training_target_contract,
        "contract_manifest": contract_manifest,
        "contract_sha256": contract_sha256,
        "encoder": encoder_record,
        "seed": args.seed,
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(
            args.dataset.read_bytes()).hexdigest(),
        "source_runs": sorted({row["source_run"] for row in rows}),
        "split": {
            "train_problem_indices": sorted(
                {sample["row"]["problem_index"] for sample in train_samples}
            ),
            "holdout_problem_indices": sorted(
                {sample["row"]["problem_index"] for sample in eval_samples}
            ),
            "train_decisions": len(train_samples),
            "train_policy_labels": sum(sample["policy_label"] for sample in train_samples),
            "holdout_decisions": len(eval_samples),
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "q_weight": args.q_weight,
            "zero_dim": args.zero_dim,
            "hash_dim": args.hash_dim,
            "hidden_dim": args.hidden_dim,
            "repr_dim": args.repr_dim,
            "node_repr_dim": args.node_repr_dim,
            "decision_rule": args.decision_rule,
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "baseline": {
            **menu_frequency_baseline(train_samples, eval_samples),
            "legal_menu_lookup": legal_menu_lookup_baseline(
                train_samples, eval_samples),
        },
        "pre_train": pre,
        "post_train": metrics(model, eval_samples, args.decision_rule),
        "history": history,
    }
    if args.checkpoint is not None:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "feature_dim": feature_dim,
                "model_config": model_contract,
                "encoder": result["encoder"],
                "experiment": args.experiment,
                "seed": args.seed,
                "decision_rule": args.decision_rule,
                "observation_schema": args.observation_schema,
                "action_semantics": result["action_semantics"],
                "verb_order": result["verb_order"],
                "previous_outcome_schema": result[
                    "previous_outcome_schema"],
                "training_mdp_contracts": result[
                    "training_mdp_contracts"],
                "feature_contract": result["feature_contract"],
                "training_target_contract": result[
                    "training_target_contract"],
                "contract_manifest": result["contract_manifest"],
                "contract_sha256": result["contract_sha256"],
            },
            args.checkpoint,
        )
        result["checkpoint"] = str(args.checkpoint)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experiment", default="E18")
    parser.add_argument(
        "--encoder", choices=("hash", "cache", "zero"), default="hash")
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--holdout-start", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--q-weight", type=float, default=0.25)
    parser.add_argument("--zero-dim", type=int, default=768)
    parser.add_argument("--hash-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--repr-dim", type=int, default=64)
    parser.add_argument("--node-repr-dim", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--decision-rule", choices=("joint", "sequential"), default="joint")
    parser.add_argument(
        "--observation-schema",
        choices=("tree-render-v1", CONTROLLER_OBSERVATION_VERSION),
        default=None,
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = train(args)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
