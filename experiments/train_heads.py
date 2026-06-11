"""Distill the trainable Q-head and policy head from an MCTS dataset.

Loads the JSONL records produced by ``experiments/collect_distillation.py``
and runs full-batch Adam to fit:

    L = policy_cross_entropy + q_loss_weight * q_mse

The dataset is small enough (~12 records initially) that full-batch is
trivially affordable and minibatching is overkill. The training step
walks both heads jointly through the shared encoder.

Outputs:

* Loss curve (policy_loss, q_loss, total) printed to stdout per epoch.
* Per-record agreement of pi_theta argmax with MCTS-recommended action.
* A checkpoint dict containing the model state at the final epoch.
* (Optional) JSON dump of trained pi_theta and Q_psi at every record,
  for offline comparison against the MCTS targets.

Usage::

    python experiments/train_heads.py \\
        --dataset /tmp/distillation_v1.jsonl \\
        --epochs 200 \\
        --lr 1e-3 \\
        --checkpoint /tmp/heads_v1.pt

The script exits 1 if torch is not installed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

try:
    import torch
except ImportError:
    print("torch is required: pip install torch", file=sys.stderr)
    sys.exit(1)

from verifier_anchored_rl.heads import (
    NUM_VERBS,
    VERB_ORDER,
    TrainableActor,
    encode_batch,
    policy_cross_entropy,
    q_mse,
)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def report_dataset(records: list[dict[str, Any]]) -> None:
    print(f"Loaded {len(records)} records", flush=True)
    by_problem: dict[str, int] = {}
    embed_dims = set()
    correct = 0
    pi_concentrated = 0
    for r in records:
        by_problem[r["problem_key"]] = by_problem.get(r["problem_key"], 0) + 1
        if r.get("embedding"):
            embed_dims.add(len(r["embedding"]))
        max_pi = max(r["pi_mcts"].values()) if r["pi_mcts"] else 0
        if max_pi >= 0.4:
            pi_concentrated += 1
        # "correct" here means MCTS recommended action matches the
        # eventual scripted-policy verb, which is informative but not
        # ground truth.
        rec_verb = r.get("recommended_action")
        if rec_verb and rec_verb in {
            "FORM_HYPOTHESIS", "INVOKE_VERIFIER", "COMMIT_ANSWER",
        }:
            correct += 1
    print("  per problem:", by_problem, flush=True)
    print(f"  embedding dims: {embed_dims}", flush=True)
    print(
        f"  records with sharp pi_MCTS (max>=0.4): {pi_concentrated}/{len(records)}",
        flush=True,
    )
    print(
        f"  records where MCTS preferred FH/IV/CA: {correct}/{len(records)}",
        flush=True,
    )


# -----------------------------------------------------------------------------
# Per-record evaluation
# -----------------------------------------------------------------------------


def evaluate(
    model: TrainableActor,
    batch: dict[str, torch.Tensor],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Report per-record agreement of trained pi_theta argmax with MCTS argmax."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        embedding = batch["embedding"]
        available_mask = batch["available_mask"]
        pi_theta = model.pi(embedding, available_mask)
        # pi_MCTS argmax over available positions
        pi_target = batch["pi_target"]
    model.train(was_training)

    pi_theta_np = pi_theta.cpu().numpy()
    pi_target_np = pi_target.cpu().numpy()

    n = len(records)
    n_agree_argmax = 0
    rows = []
    for i, rec in enumerate(records):
        target_argmax = int(pi_target_np[i].argmax())
        theta_argmax = int(pi_theta_np[i].argmax())
        agree = target_argmax == theta_argmax
        if agree:
            n_agree_argmax += 1
        rows.append(
            {
                "problem_key": rec["problem_key"],
                "turn": rec["turn"],
                "mcts_recommended": rec.get("recommended_action"),
                "mcts_argmax_idx": target_argmax,
                "mcts_argmax_verb": VERB_ORDER[target_argmax].value if target_argmax < NUM_VERBS else None,
                "theta_argmax_verb": VERB_ORDER[theta_argmax].value if theta_argmax < NUM_VERBS else None,
                "argmax_agreement": agree,
            }
        )

    return {
        "n_records": n,
        "n_agree_argmax": n_agree_argmax,
        "argmax_agreement_pct": round(100.0 * n_agree_argmax / max(1, n), 1),
        "rows": rows,
    }


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


def train(
    model: TrainableActor,
    batch: dict[str, torch.Tensor],
    *,
    epochs: int,
    lr: float,
    q_loss_weight: float,
    log_every: int,
    seed: int,
) -> dict[str, list[float]]:
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list[float]] = {
        "policy_loss": [],
        "q_loss": [],
        "total_loss": [],
    }

    embedding = batch["embedding"]
    pi_target = batch["pi_target"]
    q_target = batch["q_target"]
    available_mask = batch["available_mask"]
    q_target_mask = batch["q_target_mask"]

    print()
    print(f"{'epoch':>6}  {'policy_loss':>11}  {'q_loss':>11}  {'total':>11}", flush=True)
    print("-" * 47, flush=True)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        p_loss = policy_cross_entropy(model, embedding, pi_target, available_mask)
        q_loss = q_mse(model, embedding, q_target, q_target_mask)
        total = p_loss + q_loss_weight * q_loss
        total.backward()
        optimizer.step()

        history["policy_loss"].append(float(p_loss.item()))
        history["q_loss"].append(float(q_loss.item()))
        history["total_loss"].append(float(total.item()))

        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            print(
                f"{epoch:>6}  {p_loss.item():>11.4f}  {q_loss.item():>11.6f}  "
                f"{total.item():>11.4f}",
                flush=True,
            )

    return history


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="/tmp/distillation_v1.jsonl")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--q-loss-weight",
        type=float,
        default=1.0,
        help="weight of the Q-MSE loss in the total objective",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="print loss line every N epochs (in addition to first and last)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--repr-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0,
                   help="zero by default since this is overfitting to a tiny dataset")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="optional path to save the trained model state",
    )
    p.add_argument(
        "--evaluation-json",
        default=None,
        help="optional path to dump per-record argmax agreement as JSON",
    )
    args = p.parse_args(argv)

    print("=== Distillation training ===", flush=True)
    print(f"dataset: {args.dataset}", flush=True)
    print(
        f"hyperparams: epochs={args.epochs}, lr={args.lr}, q_weight={args.q_loss_weight}, "
        f"hidden_dim={args.hidden_dim}, repr_dim={args.repr_dim}, dropout={args.dropout}",
        flush=True,
    )
    print(flush=True)

    records = load_jsonl(args.dataset)
    report_dataset(records)

    # Drop records whose embedding fetch failed at collection time
    # (written as embedding: null). encode_batch raises on them; training
    # should proceed on the remaining good records, loudly.
    usable = [r for r in records if r.get("embedding")]
    dropped = len(records) - len(usable)
    if dropped:
        print(
            f"\nDropped {dropped} record(s) with missing embeddings "
            f"(embed-fetch failures at collection time); "
            f"training on {len(usable)} records.",
            flush=True,
        )
    if not usable:
        print("No usable records: every embedding is missing.", file=sys.stderr)
        return 1
    records = usable
    embed_dim = len(records[0]["embedding"])

    batch = encode_batch(records)
    print(f"\nBatch tensors:", flush=True)
    for k, v in batch.items():
        print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}", flush=True)

    torch.manual_seed(args.seed)
    model = TrainableActor(
        embed_dim=embed_dim,
        hidden_dim=args.hidden_dim,
        repr_dim=args.repr_dim,
        dropout=args.dropout,
    )
    print(
        f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters",
        flush=True,
    )

    print("\n=== Pre-training evaluation ===", flush=True)
    pre = evaluate(model, batch, records)
    print(
        f"pi_theta argmax agreement with pi_MCTS: "
        f"{pre['n_agree_argmax']}/{pre['n_records']} ({pre['argmax_agreement_pct']}%)",
        flush=True,
    )

    started = time.time()
    history = train(
        model,
        batch,
        epochs=args.epochs,
        lr=args.lr,
        q_loss_weight=args.q_loss_weight,
        log_every=args.log_every,
        seed=args.seed,
    )
    duration = time.time() - started

    print(flush=True)
    print(f"=== Training complete in {duration:.2f}s ===", flush=True)
    print(f"final policy_loss: {history['policy_loss'][-1]:.4f}", flush=True)
    print(f"final q_loss:      {history['q_loss'][-1]:.6f}", flush=True)
    print(f"final total:       {history['total_loss'][-1]:.4f}", flush=True)

    print("\n=== Post-training evaluation ===", flush=True)
    post = evaluate(model, batch, records)
    print(
        f"pi_theta argmax agreement with pi_MCTS: "
        f"{post['n_agree_argmax']}/{post['n_records']} ({post['argmax_agreement_pct']}%)",
        flush=True,
    )

    print("\nPer-record:", flush=True)
    for row in post["rows"]:
        marker = "✓" if row["argmax_agreement"] else "✗"
        print(
            f"  {marker}  {row['problem_key']:<18} turn={row['turn']}  "
            f"MCTS={row['mcts_argmax_verb']:<20}  pi_theta={row['theta_argmax_verb']}",
            flush=True,
        )

    if args.checkpoint:
        ckpt = {
            "model_state": model.state_dict(),
            "embed_dim": embed_dim,
            "hidden_dim": args.hidden_dim,
            "repr_dim": args.repr_dim,
            "dropout": args.dropout,
            "history": history,
            "pre_eval": pre,
            "post_eval": post,
            "args": vars(args),
        }
        torch.save(ckpt, args.checkpoint)
        print(f"\nWrote checkpoint to {args.checkpoint}", flush=True)

    if args.evaluation_json:
        with open(args.evaluation_json, "w") as f:
            json.dump({"pre": pre, "post": post, "history": history}, f, indent=2)
        print(f"Wrote evaluation JSON to {args.evaluation_json}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
