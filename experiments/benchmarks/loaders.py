# experiments/benchmarks/loaders.py
"""Benchmark dataset registry and the GSM8K loader + verifier-anchored metric.

`load_gsm8k` reads from a cached JSONL `source` (one {"question","answer"} per
line, GSM8K's native format, gold after "####"). The live download via the
optional `datasets` library is a thin wrapper that writes that JSONL cache;
tests run against a fixture and never hit the network. `gsm8k_correct` scores a
predicted answer against the gold via the SymPy verifier (verifier_anchored_rl),
after stripping commas and currency symbols.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_REGISTRY_PATH = Path(__file__).with_name("registry.json")


def registry() -> dict:
    return json.loads(_REGISTRY_PATH.read_text())


def _gold_from_answer(answer: str) -> str:
    # GSM8K marks the final answer after "####".
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer.strip()


def load_gsm8k(split: str, *, source: str | None = None) -> list[dict]:
    """Return [{problem_id, problem, gold}] for the split. `source` is a JSONL
    cache (question/answer per line); if None, the default cache path under the
    benchmarks dir is used. Raises FileNotFoundError if the cache is absent
    (run the download wrapper first)."""
    path = Path(source) if source else _REGISTRY_PATH.with_name(f"cache/gsm8k_{split}.jsonl")
    if not path.exists():
        raise FileNotFoundError(
            f"GSM8K cache not found at {path}. Provide source= or pre-download "
            f"(pip install -e .[bench]; then the download wrapper writes it).")
    items = []
    for i, line in enumerate(p for p in path.read_text().splitlines() if p.strip()):
        rec = json.loads(line)
        items.append({"problem_id": f"gsm8k-{split}-{i}",
                      "problem": rec["question"],
                      "gold": _gold_from_answer(rec["answer"])})
    return items


_NUM_CLEAN = re.compile(r"[,$%\s]")


def gsm8k_correct(pred: str, gold: str) -> bool:
    """Verifier-anchored exact match: strip commas/currency/whitespace, then
    use the SymPy-backed terminal verifier (numeric or normalized-string)."""
    from verifier_anchored_rl.verifier import check_answer
    pred_c = _NUM_CLEAN.sub("", pred or "")
    gold_c = _NUM_CLEAN.sub("", gold or "")
    return bool(check_answer(pred_c, gold_c).success)


__all__ = ["registry", "load_gsm8k", "gsm8k_correct"]
