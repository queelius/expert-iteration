"""Content-free baselines for Phase B action imitation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def legal_menu_lookup_baseline(
    train_samples: list[dict[str, Any]], eval_samples: list[dict[str, Any]]
) -> dict[str, Any]:
    """Predict from the exact legal menu, without state or node content."""

    def action_key(pair: dict[str, Any]) -> str:
        return f"{pair['verb']}:{pair['target']}"

    def signature(sample: dict[str, Any]) -> tuple[str, ...]:
        return tuple(action_key(pair) for pair in sample["row"]["legal_pairs"])

    menu_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    training_labels = 0
    for sample in train_samples:
        if not sample["policy_label"]:
            continue
        chosen = action_key(sample["row"]["chosen"])
        menu_counts[signature(sample)][chosen] += 1
        global_counts[chosen] += 1
        training_labels += 1

    records = []
    for sample in eval_samples:
        if not sample["eval_label"]:
            continue
        legal = list(signature(sample))
        menu = tuple(legal)
        seen = menu in menu_counts
        local_counts = menu_counts.get(menu, Counter())
        predicted = max(
            legal,
            key=lambda action: (
                local_counts[action], global_counts[action], action),
        )
        records.append(
            {
                "correct": predicted == action_key(sample["row"]["chosen"]),
                "seen": seen,
                "winning": bool(sample["row"]["episode_won"]),
            }
        )

    def summarize(selected: list[dict[str, bool]]) -> dict[str, Any]:
        decisions = len(selected)
        seen_rows = [record for record in selected if record["seen"]]
        correct = sum(record["correct"] for record in selected)
        seen_correct = sum(record["correct"] for record in seen_rows)
        return {
            "decisions": decisions,
            "correct": correct,
            "accuracy": correct / decisions if decisions else None,
            "seen_menu_decisions": len(seen_rows),
            "seen_menu_coverage": (
                len(seen_rows) / decisions if decisions else None
            ),
            "seen_menu_correct": seen_correct,
            "seen_menu_accuracy": (
                seen_correct / len(seen_rows) if seen_rows else None
            ),
        }

    return {
        "tie_break": "menu_count_then_global_count_then_lexical",
        "training_labels": training_labels,
        "training_menu_signatures": len(menu_counts),
        "all": summarize(records),
        "winning": summarize([record for record in records if record["winning"]]),
        "losing": summarize([
            record for record in records if not record["winning"]
        ]),
    }


