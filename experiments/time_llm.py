"""Time a single model on the verb-instantiator and embedding paths.

Usage::

    python experiments/time_llm.py \\
        --base-url http://192.168.0.204:11434/v1 \\
        --embed-url http://192.168.0.204:11434/api/embed \\
        --model qwen2.5:3b \\
        --rounds 5

The script:

1. Does ONE warmup call against each endpoint (chat-completions and
   embed) with the target model. Result discarded. This loads the
   model into VRAM if it is not already there.
2. Does ``--rounds`` timed verb-instantiator calls on the canonical
   ``FORM_HYPOTHESIS`` prompt (the one we already verified works).
3. Does ``--rounds`` timed embedding calls on the same serialized
   state.
4. Reports mean / median / min / max wall-clock per call plus token
   counts.

Run sequentially for each model. Do not run two models in parallel
since they would compete for VRAM and the timing would be useless.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import httpx

from verifier_anchored_rl.llm_context_mdp import (
    MathState,
    Subgoal,
    Verb,
    _prompt_for,
    _SYSTEM_PROMPT,
)


CANONICAL_STATE = MathState(
    problem="Compute 7 * 6.",
    open_subgoals=(Subgoal(id="sg_root", statement="Produce the final answer."),),
)


def _chat_body(model: str, user: str, temperature: float, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": f"{_SYSTEM_PROMPT}\n\nReply with a single JSON object only, no prose.",
            },
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }


def time_chat(
    base_url: str,
    model: str,
    *,
    rounds: int,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
) -> dict:
    """Time verb-instantiator (chat completions, JSON-mode)."""
    user = _prompt_for(Verb.FORM_HYPOTHESIS, CANONICAL_STATE)
    body = _chat_body(model, user, temperature, max_tokens)

    print(f"  chat warmup ({model})...", flush=True)
    with httpx.Client(timeout=timeout_s) as client:
        warmup_started = time.time()
        try:
            r = client.post(base_url.rstrip("/") + "/chat/completions", json=body)
            r.raise_for_status()
            payload = r.json()
            warmup_duration = time.time() - warmup_started
            warmup_completion_tokens = payload["usage"]["completion_tokens"]
            print(
                f"    warmup: {warmup_duration:.2f}s, "
                f"completion_tokens={warmup_completion_tokens}",
                flush=True,
            )
        except Exception as e:
            return {"error": f"chat warmup failed: {type(e).__name__}: {e}"}

        durations = []
        completion_tokens = []
        prompt_tokens = []
        for i in range(rounds):
            started = time.time()
            try:
                r = client.post(base_url.rstrip("/") + "/chat/completions", json=body)
                r.raise_for_status()
                payload = r.json()
                durations.append(time.time() - started)
                completion_tokens.append(payload["usage"]["completion_tokens"])
                prompt_tokens.append(payload["usage"]["prompt_tokens"])
                print(
                    f"    round {i+1}: {durations[-1]:.2f}s, "
                    f"completion_tokens={completion_tokens[-1]}",
                    flush=True,
                )
            except Exception as e:
                return {
                    "error": f"chat round {i+1} failed: {type(e).__name__}: {e}",
                    "partial_durations": durations,
                }

    return {
        "warmup_duration_s": round(warmup_duration, 3),
        "warmup_completion_tokens": warmup_completion_tokens,
        "rounds": rounds,
        "durations_s": [round(d, 3) for d in durations],
        "mean_s": round(statistics.mean(durations), 3),
        "median_s": round(statistics.median(durations), 3),
        "min_s": round(min(durations), 3),
        "max_s": round(max(durations), 3),
        "stdev_s": round(statistics.stdev(durations), 3) if rounds > 1 else 0.0,
        "completion_tokens": completion_tokens,
        "mean_completion_tokens": round(statistics.mean(completion_tokens), 1),
        "prompt_tokens": prompt_tokens[0] if prompt_tokens else None,
        "tokens_per_second_mean": (
            round(statistics.mean(completion_tokens) / statistics.mean(durations), 1)
            if durations
            else None
        ),
    }


def time_embed(
    embed_url: str,
    model: str,
    *,
    rounds: int,
    timeout_s: float,
) -> dict:
    """Time embedding endpoint on the canonical serialized state."""
    from verifier_anchored_rl.llm_context_mdp import serialize_state

    text = serialize_state(CANONICAL_STATE)

    print(f"  embed warmup ({model})...", flush=True)
    with httpx.Client(timeout=timeout_s) as client:
        warmup_started = time.time()
        try:
            r = client.post(embed_url, json={"model": model, "input": text})
            r.raise_for_status()
            payload = r.json()
            warmup_duration = time.time() - warmup_started
            dim = len(payload["embeddings"][0]) if payload.get("embeddings") else None
            print(f"    warmup: {warmup_duration:.2f}s, dim={dim}", flush=True)
        except Exception as e:
            return {"error": f"embed warmup failed: {type(e).__name__}: {e}"}

        durations = []
        for i in range(rounds):
            started = time.time()
            try:
                r = client.post(embed_url, json={"model": model, "input": text})
                r.raise_for_status()
                durations.append(time.time() - started)
                print(f"    round {i+1}: {durations[-1]:.3f}s", flush=True)
            except Exception as e:
                return {
                    "error": f"embed round {i+1} failed: {type(e).__name__}: {e}",
                    "partial_durations": durations,
                }

    return {
        "warmup_duration_s": round(warmup_duration, 3),
        "embedding_dim": dim,
        "rounds": rounds,
        "durations_s": [round(d, 3) for d in durations],
        "mean_s": round(statistics.mean(durations), 3),
        "median_s": round(statistics.median(durations), 3),
        "min_s": round(min(durations), 3),
        "max_s": round(max(durations), 3),
        "stdev_s": round(statistics.stdev(durations), 3) if rounds > 1 else 0.0,
        "input_chars": len(text),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://192.168.0.204:11434/v1")
    p.add_argument("--embed-url", default="http://192.168.0.204:11434/api/embed")
    p.add_argument("--model", required=True, help="Ollama model tag (e.g. qwen2.5:3b)")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--skip-embed", action="store_true")
    p.add_argument("--skip-chat", action="store_true")
    args = p.parse_args(argv)

    print(f"\n========== timing: {args.model} ==========\n", flush=True)
    result = {"model": args.model}

    if not args.skip_chat:
        print("CHAT (verb-instantiator)", flush=True)
        result["chat"] = time_chat(
            args.base_url,
            args.model,
            rounds=args.rounds,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout,
        )

    if not args.skip_embed:
        print("\nEMBED", flush=True)
        result["embed"] = time_embed(
            args.embed_url,
            args.model,
            rounds=args.rounds,
            timeout_s=args.timeout,
        )

    print("\n--- summary ---", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
