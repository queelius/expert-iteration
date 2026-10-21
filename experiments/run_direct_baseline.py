"""Direct-prompt baseline: the same model, no harness.

Asks the model to solve each problem in one shot with free-form
step-by-step reasoning and a final numeric answer, scored with the SAME
check_answer used by the Deliberation MDP's terminal verdict. This is
the honesty baseline for the Rung-1 runs: it measures what the model
does on its own, so the deck can report the accuracy cost of the
harness alongside its data-collection benefit.

    python experiments/run_direct_baseline.py \
        --model gpt-4.1-mini \
        --problems experiments/problems/gsm8k-train-500.jsonl \
        --workers 8 --out baseline.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from verifier_anchored_rl.llm_client import (
    LLMResponseError, OpenAICompatibleLLMClient,
)
from verifier_anchored_rl.verifier_adapter import check_answer
from verifier_anchored_rl.workspace_v1.lenses import (
    DIRECT_SCHEMA,
    DIRECT_SYSTEM,
    direct_user_prompt,
)

SYSTEM = DIRECT_SYSTEM

# Types deliberately unconstrained: the baseline measures MATH ability,
# and small models return schema-spirited but type-loose JSON (reasoning
# as a list of steps, answer as a bare number). A {"type": "string"}
# constraint made a correct Qwen-1.5B answer fail validation three times
# in a row. _solve_one coerces with str().
SCHEMA = DIRECT_SCHEMA


def _solve_one(i: int, p: dict, client: OpenAICompatibleLLMClient) -> dict:
    user = direct_user_prompt(p["problem"])
    answer = None
    try:
        resp = client.complete_json(SYSTEM, user, SCHEMA,
                                    temperature=0.0, max_tokens=700)
        answer = str(resp.get("answer", "")).strip()
    except LLMResponseError:
        pass  # scored as wrong below; an unusable reply is a miss
    result = check_answer(answer, p["answer"]) if answer else None
    return {
        "index": i,
        "problem": p["problem"],
        "expected": p["answer"],
        "answer": answer,
        "correct": bool(result and result.success),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--problems", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("baseline.jsonl"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-problems", type=int, default=None)
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env) or None
    if not api_key:
        raise SystemExit(f"need a key in ${args.api_key_env}")
    client = OpenAICompatibleLLMClient(base_url=args.base_url,
                                       model=args.model, api_key=api_key)

    problems = [json.loads(l)
                for l in args.problems.read_text().splitlines() if l.strip()]
    if args.max_problems is not None:
        problems = problems[: args.max_problems]

    lock = threading.Lock()
    correct = 0
    with args.out.open("w") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_solve_one, i, p, client)
                   for i, p in enumerate(problems)]
        for fut in as_completed(futures):
            rec = fut.result()
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            correct += rec["correct"]

    n = len(problems)
    print(f"direct baseline: {correct}/{n} = {correct/n:.1%} "
          f"(model={args.model}) "
          f"prompt_tokens={client.prompt_tokens} "
          f"completion_tokens={client.generated_tokens}")


if __name__ == "__main__":
    main()
