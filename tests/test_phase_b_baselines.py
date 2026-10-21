from verifier_anchored_rl.phase_b_baselines import legal_menu_lookup_baseline


def _pair(key):
    verb, target = key.split(":")
    return {"verb": verb, "target": int(target)}


def _sample(menu, chosen, *, policy_label=False, eval_label=False, won=False):
    return {
        "policy_label": policy_label,
        "eval_label": eval_label,
        "row": {
            "legal_pairs": [_pair(key) for key in menu],
            "chosen": _pair(chosen),
            "episode_won": won,
        },
    }


def test_legal_menu_lookup_uses_seen_signature_then_global_legal_fallback():
    train = [
        _sample(
            ["compute:0", "verify:1"],
            "verify:1",
            policy_label=True,
        ),
        _sample(
            ["compute:0", "verify:1"],
            "verify:1",
            policy_label=True,
        ),
        _sample(
            ["compute:0", "abandon:0"],
            "compute:0",
            policy_label=True,
        ),
    ]
    evaluation = [
        _sample(
            ["compute:0", "verify:1"],
            "verify:1",
            eval_label=True,
            won=True,
        ),
        _sample(
            ["split:0", "compute:0"],
            "compute:0",
            eval_label=True,
        ),
    ]

    result = legal_menu_lookup_baseline(train, evaluation)

    assert result["training_labels"] == 3
    assert result["all"]["correct"] == 2
    assert result["tie_break"] == (
        "menu_count_then_global_count_then_lexical")


def test_legal_menu_lookup_uses_global_counts_to_break_local_ties():
    menu = ["compute:0", "verify:1"]
    train = [
        _sample(menu, "verify:1", policy_label=True),
        _sample(menu, "compute:0", policy_label=True),
        _sample(
            ["verify:1", "abandon:0"],
            "verify:1",
            policy_label=True,
        ),
    ]
    evaluation = [
        _sample(menu, "verify:1", eval_label=True),
    ]

    result = legal_menu_lookup_baseline(train, evaluation)

    assert result["all"]["correct"] == 1
    assert result["all"]["seen_menu_correct"] == 1
