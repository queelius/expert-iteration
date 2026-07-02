"""Task 0 spike (throwaway): confirm the vLLM guided-JSON request shape and
measure that concurrent requests batch and lift GPU utilization, BEFORE wiring
vLLM into the harness. Resolves the one external-API unknown the later tasks
depend on (the exact field name) and validates the throughput premise.

This script assumes a vLLM OpenAI-compatible server is ALREADY running. Launch
it (in the isolated env, with the GPU free) like so:

    conda run -n vllm-server python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-3B-Instruct --port 8000 --max-model-len 4096

Then run this probe from the project env:

    python experiments/spike_vllm.py --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-3B-Instruct --concurrency 64 --requests 128

Findings (the confirmed guided-JSON field, the saturating concurrency, and the
throughput delta) are printed and appended to
experiments/vllm_setup/spike_findings.log for provenance. Tasks 2 and 4 use the
confirmed field name (the harness currently assumes top-level "guided_json").
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

# Two representative verb schemas mirroring src/verifier_anchored_rl/
# lenses.py::schema_for: a plain two-string object (COMPUTE) and one
# carrying an enum (the harder case for constrained decoding). The spike
# only needs realistic shapes, not the live state-dependent enums.
SCHEMA_SIMPLE = {
    "type": "object",
    "properties": {
        "expression": {"type": "string"},
        "result": {"type": "string"},
    },
    "required": ["expression", "result"],
}
SCHEMA_ENUM = {
    "type": "object",
    "properties": {
        "target_id": {"type": "string", "enum": ["sg_0", "sg_1", "none"]},
        "statement": {"type": "string"},
    },
    "required": ["target_id", "statement"],
}

SYSTEM = "You are a careful math reasoning module. Respond with JSON only."
USER = ("Compute one sub-expression toward solving 12 * (3 + 4). "
        "Return an object with string fields 'expression' and 'result'.")
USER_ENUM = ("Form one hypothesis about solving 12 * (3 + 4). Return an object "
             "with 'target_id' (one of sg_0, sg_1, none) and a 'statement' string.")


def _candidate_bodies(model: str, schema: dict, messages: list) -> dict:
    """The three request shapes a vLLM server might honor for schema-constrained
    JSON. Returns {name: body}. Task 0 step 3 picks whichever returns valid JSON."""
    base = {"model": model, "messages": messages, "temperature": 0.7,
            "max_tokens": 256}
    return {
        "guided_json (top-level)": {**base, "guided_json": schema},
        "response_format json_schema": {
            **base,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "verb", "schema": schema},
            },
        },
        "extra_body guided_json": {**base, "extra_body": {"guided_json": schema}},
    }


def _post(client: httpx.Client, base_url: str, body: dict) -> tuple[bool, str]:
    """POST one chat completion. Returns (ok, content-or-error)."""
    try:
        r = client.post(f"{base_url}/chat/completions", json=body, timeout=120.0)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return True, content
    except Exception as e:  # noqa: BLE001 - spike: report any failure verbatim
        return False, f"{type(e).__name__}: {e}"


def _valid_against(content: str, schema: dict) -> bool:
    """Is content strict JSON with all required keys present and the enum (if
    any) respected? Lightweight check, no jsonschema dependency."""
    try:
        obj = json.loads(content)
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    for key in schema.get("required", []):
        if key not in obj:
            return False
    for key, spec in schema.get("properties", {}).items():
        if key in obj and "enum" in spec and obj[key] not in spec["enum"]:
            return False
    return True


def probe_guided_json(base_url: str, model: str, log) -> str | None:
    """Try each candidate request shape against both schemas; return the field
    name that yields strictly-valid JSON for both, or None."""
    log("--- Step 3: confirm guided-JSON field ---")
    winner = None
    for name in ("guided_json (top-level)", "response_format json_schema",
                 "extra_body guided_json"):
        ok_both = True
        details = []
        with httpx.Client() as client:
            for schema, messages in (
                (SCHEMA_SIMPLE, [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": USER}]),
                (SCHEMA_ENUM, [{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": USER_ENUM}]),
            ):
                body = _candidate_bodies(model, schema, messages)[name]
                ok, content = _post(client, base_url, body)
                valid = ok and _valid_against(content, schema)
                ok_both = ok_both and valid
                details.append(f"valid={valid} ({content[:80]!r})")
        log(f"  [{name}] {'  |  '.join(details)}")
        if ok_both and winner is None:
            winner = name
    log(f"  => confirmed field: {winner!r}")
    return winner


def _sample_gpu() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader"], text=True).strip()
        return out
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi error: {e}"


def measure(base_url: str, model: str, field: str | None, n: int,
            concurrency: int, log) -> dict:
    """Fire n requests at the given concurrency; sample GPU util mid-burst.
    Returns timing + validity + peak GPU util."""
    body_field = field if field and "extra_body" not in field else None
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER}]
    bodies = _candidate_bodies(model, SCHEMA_SIMPLE, messages)
    # Use the confirmed shape; fall back to plain (no constraint) if none.
    body = bodies.get(field) if field else {
        "model": model, "messages": messages, "temperature": 0.7,
        "max_tokens": 256}

    gpu_samples: list[str] = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            gpu_samples.append(_sample_gpu())
            time.sleep(0.25)

    st = threading.Thread(target=sampler, daemon=True)
    st.start()

    results: list[bool] = []
    t0 = time.monotonic()
    with httpx.Client() as client:
        def one(_i):
            ok, content = _post(client, base_url, dict(body))
            return ok and _valid_against(content, SCHEMA_SIMPLE)
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            results = list(ex.map(one, range(n)))
    elapsed = time.monotonic() - t0
    stop.set()
    st.join(timeout=1.0)

    def _util(s: str) -> int:
        try:
            return int(s.split("%")[0].strip())
        except Exception:
            return -1

    utils = [_util(s) for s in gpu_samples if "%" in s]
    peak = max(utils) if utils else -1
    valid = sum(1 for r in results if r)
    return {
        "concurrency": concurrency, "requests": n, "elapsed_s": round(elapsed, 2),
        "req_per_s": round(n / elapsed, 2) if elapsed > 0 else 0.0,
        "valid_frac": round(valid / n, 3) if n else 0.0,
        "peak_gpu_util": peak, "n_gpu_samples": len(utils),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--requests", type=int, default=128)
    p.add_argument("--concurrency", type=int, default=64)
    args = p.parse_args(argv)

    log_path = Path(__file__).resolve().parent / "vllm_setup" / "spike_findings.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "a")

    def log(msg: str) -> None:
        line = msg if msg.endswith("\n") else msg + "\n"
        fh.write(line)
        fh.flush()
        sys.stdout.write(line)
        sys.stdout.flush()

    log(f"\n=== vLLM spike {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
        f"model={args.model} url={args.base_url} ===")

    # Reachability check.
    try:
        with httpx.Client() as c:
            c.get(f"{args.base_url}/models", timeout=10.0).raise_for_status()
        log("  server reachable")
    except Exception as e:  # noqa: BLE001
        log(f"  ERROR: server not reachable at {args.base_url}: {e}")
        log("  Launch it first (see module docstring) on a free GPU.")
        return 2

    field = probe_guided_json(args.base_url, args.model, log)

    log("--- Step 4: throughput / GPU-util, sequential vs concurrent ---")
    seq = measure(args.base_url, args.model, field, n=min(16, args.requests),
                  concurrency=1, log=log)
    log(f"  sequential (c=1): {seq}")
    conc = measure(args.base_url, args.model, field, n=args.requests,
                   concurrency=args.concurrency, log=log)
    log(f"  concurrent (c={args.concurrency}): {conc}")

    speedup = (conc["req_per_s"] / seq["req_per_s"]) if seq["req_per_s"] else 0.0
    log("=== FINDINGS ===")
    log(f"  guided-JSON field: {field!r}")
    log(f"  throughput: {seq['req_per_s']} -> {conc['req_per_s']} req/s "
        f"({speedup:.1f}x)")
    log(f"  peak GPU util: seq {seq['peak_gpu_util']}% -> "
        f"conc {conc['peak_gpu_util']}%")
    log(f"  valid JSON frac: seq {seq['valid_frac']} conc {conc['valid_frac']}")
    fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
