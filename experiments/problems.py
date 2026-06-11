"""Canonical toy-problem registry shared by the experiment scripts.

Before this module existed, ``collect_distillation.py``,
``run_llm_mcts_with_prior.py``, and ``run_llm_mcts.py`` each carried a
hand-copied PROBLEMS dict, and they had already diverged (the comparison
script could not see the held-out or v4b problems). Every script now
imports from here, so adding a problem or fixing an expected answer is a
one-file change and train/eval cannot silently disagree on ground truth.

Each problem maps a key to ``{"problem": <statement>, "expected":
<canonical answer string>}``. ``PROBLEM_GROUPS`` names the per-round
training compositions used by the pilot study plus the held-out
evaluation set; group membership is part of the experimental record
(the paper reports per-round composition), so groups are append-only:
define a new group for a new round rather than mutating an old one.

``SCRIPTED_VERB_SEQUENCE`` / ``scripted_verb`` is the fixed three-step
trajectory policy (hypothesize, verify, commit) that data collection
walks; it determines WHICH states MCTS is asked about, so collection
and any script reasoning about those states must share it.
"""

from __future__ import annotations

from typing import Sequence

from verifier_anchored_rl import Verb


PROBLEMS: dict[str, dict[str, str]] = {
    # --- core training set (used for v1, v2, v3 head training) ---
    "multiply_small": {
        "problem": "Compute 7 * 6.",
        "expected": "42",
    },
    "gsm8k_toy_1": {
        "problem": (
            "Alice has 3 boxes of apples. Each box contains 12 apples. "
            "She gives away 7 apples. How many apples does she have left?"
        ),
        "expected": "29",
    },
    "algebra_toy_1": {
        "problem": "Solve for x: 2x + 5 = 13.",
        "expected": "4",
    },
    "factor": {
        "problem": "Factor the expression x^2 - 9.",
        "expected": "(x - 3)*(x + 3)",
    },
    # --- extra gsm8k variants for v4 (target the weakest holdout generalization) ---
    "gsm8k_toy_2": {
        "problem": (
            "Sarah has 5 baskets of strawberries. Each basket contains "
            "8 strawberries. She eats 6 strawberries. How many strawberries "
            "does she have left?"
        ),
        "expected": "34",
    },
    "gsm8k_toy_3": {
        "problem": (
            "Tom buys 6 packs of trading cards. Each pack contains 10 cards. "
            "He gives away 12 cards. How many cards does he have left?"
        ),
        "expected": "48",
    },
    "gsm8k_toy_4": {
        "problem": (
            "A library has 7 shelves of books. Each shelf holds 9 books. "
            "11 books are checked out. How many books remain on the shelves?"
        ),
        "expected": "52",
    },
    # --- v4b additions: rebalance after v4's class-imbalance regression ---
    "algebra_toy_2": {
        "problem": "Solve for x: 4x - 3 = 9.",
        "expected": "3",
    },
    "algebra_toy_3": {
        "problem": "Solve for x: 7x + 2 = 16.",
        "expected": "2",
    },
    "factor_2": {
        "problem": "Factor the expression x^2 - 16.",
        "expected": "(x - 4)*(x + 4)",
    },
    # --- held-out set (generalization eval only; same shape, fresh content) ---
    "multiply_small_ho": {
        "problem": "Compute 9 * 8.",
        "expected": "72",
    },
    "gsm8k_toy_ho": {
        "problem": (
            "Bob has 4 bags of marbles. Each bag contains 15 marbles. "
            "He loses 9 marbles. How many marbles does he have left?"
        ),
        "expected": "51",
    },
    "algebra_toy_ho": {
        "problem": "Solve for y: 3y - 7 = 8.",
        "expected": "5",
    },
    "factor_ho": {
        "problem": "Factor the expression x^2 + 5x + 6.",
        "expected": "(x + 2)*(x + 3)",
    },
}


PROBLEM_GROUPS: dict[str, list[str]] = {
    # 'train' = the legacy 4 used in v1/v2/v3; preserved so old runs
    # remain reproducible.
    "train": [
        "multiply_small",
        "gsm8k_toy_1",
        "algebra_toy_1",
        "factor",
    ],
    # 'train_v4' = legacy 4 + 3 extra gsm8k variants, used by the v4
    # round to address the weak word-problem generalization observed in
    # v3 heads on gsm8k_toy_ho.
    "train_v4": [
        "multiply_small",
        "gsm8k_toy_1",
        "algebra_toy_1",
        "factor",
        "gsm8k_toy_2",
        "gsm8k_toy_3",
        "gsm8k_toy_4",
    ],
    # 'train_v4b' = train_v4 + 2 algebra variants + 1 factor variant,
    # rebalanced after v4's algebra regression. Composition:
    # 10% arithmetic / 40% gsm8k / 30% algebra / 20% factor.
    "train_v4b": [
        "multiply_small",
        "gsm8k_toy_1",
        "algebra_toy_1",
        "factor",
        "gsm8k_toy_2",
        "gsm8k_toy_3",
        "gsm8k_toy_4",
        "algebra_toy_2",
        "algebra_toy_3",
        "factor_2",
    ],
    "holdout": [
        "multiply_small_ho",
        "gsm8k_toy_ho",
        "algebra_toy_ho",
        "factor_ho",
    ],
}


SCRIPTED_VERB_SEQUENCE: Sequence[Verb] = (
    Verb.FORM_HYPOTHESIS,
    Verb.INVOKE_VERIFIER,
    Verb.COMMIT_ANSWER,
)


def scripted_verb(turn: int, available: Sequence[Verb]) -> Verb:
    """Fixed three-step trajectory policy: hypothesize, verify, commit.

    Falls back to the first available verb when the scripted choice is
    unavailable at the current turn.
    """
    if turn < len(SCRIPTED_VERB_SEQUENCE):
        candidate = SCRIPTED_VERB_SEQUENCE[turn]
        if candidate in available:
            return candidate
    return available[0]


def resolve_problem_selection(requested: Sequence[str]) -> list[str]:
    """Expand a mix of problem keys and group aliases into a deduplicated
    ordered list of problem keys. ``"all"`` expands to every problem."""
    groups = {**PROBLEM_GROUPS, "all": list(PROBLEMS)}
    expanded: list[str] = []
    for item in requested:
        expanded.extend(groups.get(item, [item]))
    return list(dict.fromkeys(expanded))


__all__ = [
    "PROBLEMS",
    "PROBLEM_GROUPS",
    "SCRIPTED_VERB_SEQUENCE",
    "resolve_problem_selection",
    "scripted_verb",
]
