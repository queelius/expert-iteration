"""End-to-end demo of the Section 3.7 training loop on collected records.

CPU-only (uses the stored embeddings; the frozen LLM is never re-run), so
it runs anywhere without a GPU. Loads a distillation JSONL into the
prioritized replay buffer via ``sample_from_record`` (the exact
``collect_distillation`` schema), constructs a ``HiddenStateActor`` at the
records' embedding dimension, and runs the ``Trainer`` (combined
``L_Q + lambda * L_pi`` with periodic target sync). Reports the loss curve.

On the pilot's v2 records this reproduces the pilot's training floor (the
policy cross-entropy settles near the target-entropy floor that the pilot
measured at ~1.45), validating that the new architecture + training loop
match the pilot's behavior on the pilot's own data while adding the replay
buffer and target-Q sync the pilot lacked.

Usage::

    python experiments/run_train_demo.py \\
        --data experiments/data/pilot/distillation_v2.jsonl --steps 400
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="experiments/data/pilot/distillation_v2.jsonl")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-pi", type=float, default=1.0)
    p.add_argument("--sync-interval", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    try:
        import torch
    except ImportError:
        print("torch is required: pip install -e .[hf]", file=sys.stderr)
        return 1

    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.replay import ReplayBuffer, sample_from_record
    from verifier_anchored_rl.train import Trainer

    records = [json.loads(l) for l in open(args.data) if l.strip()]
    usable = [r for r in records if r.get("embedding")]
    if not usable:
        print("no records with embeddings", file=sys.stderr)
        return 1
    d = len(usable[0]["embedding"])

    buf = ReplayBuffer(capacity=len(usable))
    for r in usable:
        buf.add(sample_from_record(r))

    torch.manual_seed(args.seed)
    actor = HiddenStateActor(embed_dim=d)
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    trainer = Trainer(
        actor, opt, buf,
        sync_interval=args.sync_interval, lambda_pi=args.lambda_pi,
    )
    rng = np.random.default_rng(args.seed)

    print(f"=== Section 3.7 training-loop demo ===", flush=True)
    print(f"data: {args.data}", flush=True)
    print(f"{len(buf)} records, embed_dim={d}, "
          f"actor trainable params={actor.trainable_parameter_count():,}", flush=True)
    print(f"lambda_pi={args.lambda_pi}, sync_interval={args.sync_interval}, "
          f"batch={args.batch_size}, lr={args.lr}", flush=True)
    print(f"\n{'step':>6} {'q_loss':>10} {'pi_loss':>10} {'total':>10}", flush=True)
    log_every = max(1, args.steps // 10)
    for i in range(1, args.steps + 1):
        losses = trainer.step(args.batch_size, rng)
        if i == 1 or i % log_every == 0:
            print(f"{i:>6} {losses['q']:>10.4f} {losses['pi']:>10.4f} "
                  f"{losses['total']:>10.4f}", flush=True)
    print(f"\ntarget synced {trainer.global_step // args.sync_interval} time(s) "
          f"(every {args.sync_interval} steps)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
