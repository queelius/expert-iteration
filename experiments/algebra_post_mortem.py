"""Side-by-side post-mortem of v3 vs v4 on the algebra problem family.

After v4 caused a 60% -> 27% argmax regression on algebra_toy_ho relative
to v3, this script asks: did the underlying training data change, or did
the model just under-fit algebra because gsm8k crowded it out?

Steps:

1. Compare per-record pi_MCTS for ALGEBRA-FAMILY (algebra_toy_1) records
   that appear in BOTH v3-train and v4-train. If the MCTS targets look
   similar across rounds, the regression is a fitting/dilution problem,
   not a data-collection problem.

   NOTE (post-v4b correction): v3-train used the v2-heads prior and
   v4-train the v3-heads prior, so this comparison confounds prior change
   with LLM-sampling noise. The clean same-prior comparison is v4-train
   vs v4b-train (both v3-heads prior, matched seeds): L1 = 0.553, 7/15
   argmax agreement on algebra_toy_1. The instability is nearly as large
   under fully matched conditions; the LLM transitions are unseeded, so
   collection runs are not reproducible even at matched MCTS seeds.

2. Run v3-heads and v4-heads side-by-side on the algebra_toy_ho records
   from v3-holdout. Show per-record pi_theta predictions and where they
   disagree.

3. Report what fraction of v4-train was algebra (14%) vs v3-train (25%)
   to make the dilution explicit.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verifier_anchored_rl.heads import NUM_VERBS, VERB_ORDER, encode_batch
from verifier_anchored_rl.trained_prior import load_actor_from_checkpoint


def load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def entropy(d):
    return -sum(p * math.log2(p) for p in d.values() if p > 0)


def step1_compare_training_targets():
    print("=" * 90)
    print("STEP 1: Same problem (algebra_toy_1), same turn -> v3 vs v4 pi_MCTS")
    print("=" * 90)
    v3 = [r for r in load_jsonl("/tmp/distillation_v3_train.jsonl")
          if r["problem_key"] == "algebra_toy_1"]
    v4 = [r for r in load_jsonl("/tmp/distillation_v4_train.jsonl")
          if r["problem_key"] == "algebra_toy_1"]

    # Index by (seed, turn)
    def index(recs):
        return {(r["seed"], r["turn"]): r for r in recs}
    v3i, v4i = index(v3), index(v4)
    common = sorted(set(v3i) & set(v4i))

    print(f"\nalgebra_toy_1: v3-train has {len(v3)} records, v4-train has {len(v4)} records")
    print(f"common (seed, turn) pairs: {len(common)}\n")
    if not common:
        print("No common (seed, turn) pairs (disjoint --seeds or different "
              "trajectory lengths); skipping step 1 comparison.")
        return

    print(f"{'seed,turn':>10}  {'v3 max':>7} {'v4 max':>7}  {'v3 H':>6} {'v4 H':>6}  {'L1(v3,v4)':>10}  v3_argmax->v4_argmax")
    print("-" * 90)
    l1_total = 0.0
    argmax_agree = 0
    for k in common:
        r3, r4 = v3i[k], v4i[k]
        p3, p4 = r3["pi_mcts"], r4["pi_mcts"]
        verbs = set(p3) | set(p4)
        l1 = sum(abs(p3.get(v, 0) - p4.get(v, 0)) for v in verbs)
        l1_total += l1
        am3 = max(p3.items(), key=lambda kv: kv[1])[0]
        am4 = max(p4.items(), key=lambda kv: kv[1])[0]
        if am3 == am4:
            argmax_agree += 1
        mark = "" if am3 == am4 else " <-- shifted"
        print(
            f"{str(k):>10}  {max(p3.values()):>7.3f} {max(p4.values()):>7.3f}  "
            f"{entropy(p3):>6.2f} {entropy(p4):>6.2f}  {l1:>10.3f}  "
            f"{am3} -> {am4}{mark}"
        )
    print()
    print(f"mean L1(v3-train algebra, v4-train algebra) = {l1_total/len(common):.3f}")
    print(f"argmax agreement across rounds = {argmax_agree}/{len(common)} ({100*argmax_agree/len(common):.0f}%)")
    print()
    print("Interpretation: if mean L1 is small (<0.3) and argmax agreement is high,")
    print("the algebra MCTS targets did NOT change between rounds. That isolates")
    print("the regression to the model-fitting side (dilution / under-fitting),")
    print("not the data-collection side.")


def step2_head_predictions_on_holdout_algebra():
    print()
    print("=" * 90)
    print("STEP 2: v3-heads vs v4-heads on algebra_toy_ho records from v3-holdout")
    print("=" * 90)
    recs = [r for r in load_jsonl("/tmp/distillation_v3_holdout.jsonl")
            if r["problem_key"] == "algebra_toy_ho" and r.get("embedding")]
    print(f"\nN = {len(recs)} algebra records in v3-holdout\n")

    model_v3, _ = load_actor_from_checkpoint("/tmp/heads_v3.pt")
    model_v4, _ = load_actor_from_checkpoint("/tmp/heads_v4.pt")

    batch = encode_batch(recs)
    with torch.no_grad():
        pi3 = model_v3.pi(batch["embedding"], batch["available_mask"]).cpu().numpy()
        pi4 = model_v4.pi(batch["embedding"], batch["available_mask"]).cpu().numpy()

    print(f"{'seed,turn':>10}  {'mcts argmax':>16}  {'v3 argmax':>12}  {'v4 argmax':>12}  v3 L1   v4 L1")
    print("-" * 95)
    for i, rec in enumerate(recs):
        pi_mcts = rec["pi_mcts"]
        verbs = list(pi_mcts.keys())
        # Build dicts over all 7 verbs
        pi3_dict = {VERB_ORDER[j].value: float(pi3[i, j]) for j in range(NUM_VERBS)
                    if VERB_ORDER[j].value in pi_mcts}
        pi4_dict = {VERB_ORDER[j].value: float(pi4[i, j]) for j in range(NUM_VERBS)
                    if VERB_ORDER[j].value in pi_mcts}
        # Renormalize over available verbs
        s3 = sum(pi3_dict.values()) or 1.0
        s4 = sum(pi4_dict.values()) or 1.0
        pi3_dict = {k: v/s3 for k, v in pi3_dict.items()}
        pi4_dict = {k: v/s4 for k, v in pi4_dict.items()}

        am_mcts = max(pi_mcts.items(), key=lambda kv: kv[1])[0]
        am_v3 = max(pi3_dict.items(), key=lambda kv: kv[1])[0]
        am_v4 = max(pi4_dict.items(), key=lambda kv: kv[1])[0]
        l1_v3 = sum(abs(pi3_dict.get(v, 0) - pi_mcts.get(v, 0)) for v in verbs)
        l1_v4 = sum(abs(pi4_dict.get(v, 0) - pi_mcts.get(v, 0)) for v in verbs)
        mark_v3 = "OK" if am_v3 == am_mcts else "MISS"
        mark_v4 = "OK" if am_v4 == am_mcts else "MISS"
        print(
            f"({rec['seed']},{rec['turn']})    {am_mcts:>16}  {am_v3:>10} {mark_v3:>2}  "
            f"{am_v4:>10} {mark_v4:>2}  {l1_v3:>.3f}  {l1_v4:>.3f}"
        )


def step3_class_balance():
    print()
    print("=" * 90)
    print("STEP 3: Class balance across rounds")
    print("=" * 90)
    for path, label in [
        ("/tmp/distillation_v2.jsonl", "v2"),
        ("/tmp/distillation_v3_train.jsonl", "v3-train"),
        ("/tmp/distillation_v4_train.jsonl", "v4-train"),
    ]:
        recs = load_jsonl(path)
        by_family = defaultdict(int)
        for r in recs:
            pk = r["problem_key"]
            if pk.startswith("multiply"):
                family = "arithmetic"
            elif pk.startswith("gsm8k"):
                family = "gsm8k (word)"
            elif pk.startswith("algebra"):
                family = "algebra"
            elif pk.startswith("factor"):
                family = "factor"
            else:
                family = "other"
            by_family[family] += 1
        total = sum(by_family.values())
        print(f"\n{label} ({total} records):")
        for fam, n in sorted(by_family.items(), key=lambda kv: -kv[1]):
            print(f"  {fam:<14} {n:>3}  ({100*n/total:>4.1f}%)")


def main():
    step1_compare_training_targets()
    step2_head_predictions_on_holdout_algebra()
    step3_class_balance()


if __name__ == "__main__":
    main()
