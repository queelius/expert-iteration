from __future__ import annotations


def test_load_gsm8k_parses_gold_after_hashes(tmp_path):
    from experiments.benchmarks.loaders import load_gsm8k
    fixture = tmp_path / "gsm8k_test.jsonl"
    import json
    fixture.write_text("\n".join(json.dumps(r) for r in [
        {"question": "2+2?", "answer": "two plus two\n#### 4"},
        {"question": "3+3?", "answer": "#### 6"},
    ]) + "\n")
    items = load_gsm8k("test", source=str(fixture))
    assert len(items) == 2
    assert items[0]["gold"] == "4" and items[0]["problem"] == "2+2?"
    assert items[0]["problem_id"] == "gsm8k-test-0"


def test_gsm8k_correct_uses_verifier_and_normalizes():
    from experiments.benchmarks.loaders import gsm8k_correct
    assert gsm8k_correct("72", "72") is True
    assert gsm8k_correct("1,000", "1000") is True   # comma stripped
    assert gsm8k_correct("$72", "72") is True        # currency stripped
    assert gsm8k_correct("73", "72") is False


def test_registry_lists_gsm8k():
    from experiments.benchmarks.loaders import registry
    reg = registry()
    assert "gsm8k" in reg and reg["gsm8k"]["answer_style"] == "final-number"
