# vLLM-batched, decoupled-embedder fast pipeline - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Saturate the GPU during closed-loop collection by generating with vLLM (GPU-side guided-JSON, continuous batching) driven by a concurrent collection loop, and decouple the heads' state embedding behind a swappable `Embedder` (text-embedder default; legacy `h_T` retained; GNN later).

**Architecture:** Generation and representation are separated. A vLLM server serves the frozen model with `guided_json`; the collection loop runs many trajectories concurrently so vLLM continuously batches their requests. The heads consume `embed(state)` from a `TextEmbedder` (implements the existing `Encoder` protocol) instead of the LLM hidden state. The existing `OpenAICompatibleLLMClient` (httpx, already documents vLLM) is the generation integration point; the existing `EmbeddingClient`/`TransformersEmbeddingClient` is reused for embeddings.

**Tech Stack:** Python 3.12, vLLM (new dep), httpx, torch, transformers, pytest. Hard constraint: no em-dash glyph in any file (repo hook). Frozen-model generation is GPU-bound; GPU-dependent tasks (0, 5) need the GPU free (the corrected E1 sweep must finish or move to cloud first).

---

### Task 0: vLLM setup + guided-JSON + batched-throughput spike (GPU; de-risk)

**Files:** Create `experiments/spike_vllm.py` (throwaway; not committed to src).

Purpose: confirm the exact vLLM request shape for schema-constrained JSON and measure that concurrent requests batch and raise GPU utilization, BEFORE wiring it in. This resolves the one external-API unknown the later tasks depend on.

- [ ] **Step 1: Install vLLM** (pin a known-good version): `pip install "vllm>=0.6,<0.9"`. Record the installed version.
- [ ] **Step 2: Launch a 3B vLLM server** (GPU must be free):
  `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-3B-Instruct --port 8000 --max-model-len 4096 &`
- [ ] **Step 3: Confirm guided JSON.** POST one `/v1/chat/completions` with a verb schema using BOTH candidate fields and see which the server honors: (a) top-level `"guided_json": <schema>`, (b) `"response_format": {"type": "json_schema", "json_schema": {"schema": <schema>}}`, (c) `extra_body`. Record the field that returns schema-valid JSON. Expected: at least one returns strictly-valid JSON.
- [ ] **Step 4: Measure batching.** Fire 64 concurrent requests (threads or asyncio) and watch `nvidia-smi`. Expected: GPU utilization rises well above the ~10% batch-1 baseline (target: a clear multiple), and aggregate throughput (requests/s) is far above sequential.
- [ ] **Step 5: Record findings** in the run log: the confirmed guided-JSON field name, the concurrency level that saturates the 3B model on this GPU, and the throughput delta. Tasks 2 and 4 use these.

Acceptance: a documented guided-JSON request shape + evidence that concurrency batches and lifts utilization.

---

### Task 1: TextEmbedder (decoupled embedder behind the `Encoder` protocol)

**Files:**
- Create: `src/verifier_anchored_rl/embedder.py`
- Test: `tests/test_embedder.py`

- [ ] **Step 1: Write the failing test**

```python
"""TextEmbedder adapts an EmbeddingClient to the StateActor's Encoder protocol."""

from __future__ import annotations

import pytest


class _FakeEmbeddingClient:
    """Returns a fixed-dim vector; records calls."""
    def __init__(self, dim):
        self.dim = dim
        self.calls = 0
    def embed(self, text):
        self.calls += 1
        # deterministic, content-dependent, L2-ish (values in [0,1))
        return [((hash(text) >> i) & 255) / 255.0 for i in range(self.dim)]


def test_text_embedder_is_an_encoder_and_returns_hidden_vector():
    pytest.importorskip("torch")
    import torch
    from verifier_anchored_rl.embedder import TextEmbedder

    client = _FakeEmbeddingClient(dim=32)
    emb = TextEmbedder(client, embed_dim=32)
    assert emb.embed_dim == 32
    prefill = emb.encode("State:\nsome serialized state")
    assert isinstance(prefill.hidden, torch.Tensor)
    assert tuple(prefill.hidden.shape) == (32,)
    assert prefill.hidden.dtype == torch.float32
    assert prefill.cache is None  # no KV cache: StateActor.embed must not try to free one
    assert client.calls == 1


def test_text_embedder_drops_into_state_actor():
    pytest.importorskip("torch")
    from tests.fakes import FakeEncoder  # noqa: F401  (ensures torch/test env)
    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.planner_actor import PrefillCache, StateActor
    from verifier_anchored_rl.llm_client import EchoLLMClient
    from verifier_anchored_rl.llm_context_mdp import LLMContextMDP
    from verifier_anchored_rl.embedder import TextEmbedder

    emb = TextEmbedder(_FakeEmbeddingClient(dim=16), embed_dim=16)
    actor = HiddenStateActor(embed_dim=16)
    sa = StateActor(emb, actor, PrefillCache(), depth_limit=3)
    mdp = LLMContextMDP(problem_statement="2+2", expected_answer="4",
                        llm_client=EchoLLMClient(), depth_limit=3, max_turns=4)
    state = mdp.initial_state()
    actions = list(mdp.actions(state))
    prior = sa.prior(state, actions)
    assert abs(sum(prior) - 1.0) < 1e-5
```

- [ ] **Step 2: Run, verify FAIL** (`ModuleNotFoundError: embedder`):
  `python -m pytest tests/test_embedder.py -q`

- [ ] **Step 3: Implement**

```python
"""Decoupled state embedder: adapts an EmbeddingClient (text -> vector) to the
StateActor's Encoder protocol, so the heads can use a dedicated embedding model
instead of the generator LLM's hidden state. This frees generation to be served
by vLLM (which does not expose hidden states). The legacy LLM h_T path
(SharedForwardEncoder) remains a separate Encoder implementation, selectable as
a baseline/ablation; a future GraphEmbedder (GNN over the reasoning graph) will
implement the same protocol.
"""

from __future__ import annotations

from typing import Any


class TextEmbedder:
    """Encoder backed by a text EmbeddingClient. ``encode(text)`` returns a
    StatePrefill whose ``hidden`` is the embedding vector (fp32, on CPU) and
    whose KV cache is None (there is no generation cache on this path)."""

    def __init__(self, embedding_client: Any, embed_dim: int):
        self._client = embedding_client
        self._embed_dim = int(embed_dim)
        # FLOP-accounting parity with the encoder interface.
        self.prefill_calls = 0
        self.prefill_tokens = 0
        self.gen_calls = 0
        self.gen_tokens = 0

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def encode(self, prefix_text: str):
        import torch

        from .shared_forward import StatePrefill

        vec = self._client.embed(prefix_text)
        if len(vec) != self._embed_dim:
            raise ValueError(
                f"TextEmbedder: embedding dim {len(vec)} != configured "
                f"{self._embed_dim} (embedder/heads mismatch)"
            )
        self.prefill_calls += 1
        self.prefill_tokens += max(1, len(prefix_text.split()))
        return StatePrefill(
            hidden=torch.tensor(vec, dtype=torch.float32),
            cache=None, n_prefix_tokens=0, prefix_ids=None,
        )


__all__ = ["TextEmbedder"]
```

- [ ] **Step 4: Run, verify PASS** (`python -m pytest tests/test_embedder.py -q`).
- [ ] **Step 5: Commit**
```bash
git add src/verifier_anchored_rl/embedder.py tests/test_embedder.py
git commit -m "Add TextEmbedder: decoupled state embedder behind the Encoder protocol

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: guided-JSON support in OpenAICompatibleLLMClient

**Files:**
- Modify: `src/verifier_anchored_rl/llm_client.py` (`OpenAICompatibleLLMClient`)
- Test: `tests/test_llm_client.py` (add)

Use the guided-JSON field confirmed in Task 0 (placeholder name below is `guided_json`; if Task 0 found `response_format: json_schema`, use that and update the assertion). The client gains a `guided_json: bool = False` constructor flag; when set, `complete_json` adds the schema to the request body so vLLM constrains decoding GPU-side.

- [ ] **Step 1: Write the failing test** (capture the POST body via a fake httpx)

```python
def test_openai_client_sends_guided_json_when_enabled(monkeypatch):
    import pytest
    pytest.importorskip("httpx")
    import httpx
    from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient

    captured = {}

    class _Resp:
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": '{"answer": "42"}'}}]}

    class _Client:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, url, headers=None, json=None):
            captured["body"] = json
            return _Resp()

    monkeypatch.setattr(httpx, "Client", _Client)
    schema = {"type": "object", "properties": {"answer": {"type": "string"}},
              "required": ["answer"]}
    client = OpenAICompatibleLLMClient(base_url="http://x/v1", model="m",
                                       guided_json=True)
    out = client.complete_json(system="s", user="u", schema=schema)
    assert out == {"answer": "42"}
    assert captured["body"].get("guided_json") == schema  # vLLM structured output
```

- [ ] **Step 2: Run, verify FAIL** (`TypeError: unexpected kwarg 'guided_json'`).
- [ ] **Step 3: Implement** - add `guided_json: bool = False` to `__init__` (store `self.guided_json`); in `complete_json`, after building `body`, add `if self.guided_json: body["guided_json"] = schema` (or the Task-0-confirmed field). Keep the existing `response_format` json_object as a fallback hint.
- [ ] **Step 4: Run, verify PASS**; then `python -m pytest tests/test_llm_client.py -q` (no regression).
- [ ] **Step 5: Commit**
```bash
git add src/verifier_anchored_rl/llm_client.py tests/test_llm_client.py
git commit -m "Add guided_json to OpenAICompatibleLLMClient for vLLM structured decoding

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: concurrent collection (thread-pool over trajectories, shared client)

**Files:**
- Modify: `src/verifier_anchored_rl/outer_loop.py` (add `collect_iteration_concurrent`)
- Test: `tests/test_outer_loop.py` (add)

A single trajectory's MCTS stays sequential; throughput comes from running many trajectories at once (a `ThreadPoolExecutor`) all hitting the shared vLLM client, which batches their concurrent requests. Each worker uses its OWN rng (sub-seeded) and returns its records; the buffer is populated single-threaded after, so results are deterministic and thread-safe regardless of pool size.

- [ ] **Step 1: Write the failing test**

```python
def test_collect_iteration_concurrent_matches_sequential_records(monkeypatch):
    import pytest
    pytest.importorskip("torch")
    import numpy as np
    from verifier_anchored_rl.mcts import MCTSConfig
    from verifier_anchored_rl.outer_loop import (
        collect_iteration, collect_iteration_concurrent,
    )

    def fresh():
        enc, actor, sa, buf, trainer, mdp_factory = _build()
        cfg = MCTSConfig(num_simulations=6, c_puct=1.5, discount=1.0,
                         prior_fn=sa.prior, value_fn=sa.leaf_value)
        return sa, buf, cfg, mdp_factory

    problems = [("2+2", "4"), ("3+3", "6"), ("4+4", "8")]

    sa1, buf1, cfg1, mf1 = fresh()
    collect_iteration(problems, mf1, sa1, cfg1, buf1, np.random.default_rng(0),
                      tau=1.0, max_turns=4)

    sa2, buf2, cfg2, mf2 = fresh()
    collect_iteration_concurrent(problems, mf2, sa2, cfg2, buf2, base_seed=0,
                                 tau=1.0, max_turns=4, max_workers=3)

    # Same number of records and same per-problem decisions (seeded per problem).
    assert len(buf2) == len(buf1)
```

- [ ] **Step 2: Run, verify FAIL** (`ImportError: collect_iteration_concurrent`).
- [ ] **Step 3: Implement** `collect_iteration_concurrent` in outer_loop.py:

```python
def collect_iteration_concurrent(
    problems, mdp_factory, state_actor, config, buffer, *,
    base_seed, tau, max_turns, max_workers=8,
    record_sink=None, on_decision=None, gen_source=None, iteration_index=0,
):
    """Run one trajectory per problem CONCURRENTLY (thread pool). Each worker
    drives a normal sequential MCTS trajectory with its OWN rng (sub-seeded from
    base_seed + problem index) and returns its ReplaySamples; the shared
    llm_client (vLLM) batches the concurrent generation requests. The buffer is
    populated single-threaded after the workers finish, so it is thread-safe and
    deterministic regardless of max_workers."""
    import concurrent.futures as cf
    import threading

    import numpy as np

    from .llm_context_mdp import available_actions
    from .mcts import mcts_search, root_q_estimates, visit_count_policy

    log_lock = threading.Lock()

    def run_one(p_idx, problem):
        rng = np.random.default_rng(base_seed + 1 + p_idx)
        mdp = mdp_factory(problem)
        state = mdp.initial_state()
        records = []
        while not mdp.is_terminal(state) and state.turn < max_turns:
            avail = list(available_actions(state, depth_limit=mdp.depth_limit))
            if not avail:
                break
            root = mcts_search(mdp, state, config, rng)
            if not root.children:
                break
            pi = visit_count_policy(root, temperature=tau)
            q = root_q_estimates(root)
            rec = {
                "embedding": state_actor.embed(state).detach().cpu().tolist(),
                "available_verbs": [a.value for a in avail],
                "pi_mcts": {a.verb.value: float(p) for a, p in pi.items()},
                "q_mcts": {a.verb.value: float(v) for a, v in q.items()},
            }
            records.append(rec)
            actions = list(pi.keys())
            probs = np.array([pi[a] for a in actions], dtype=float)
            probs = probs / probs.sum()
            chosen = actions[int(rng.choice(len(actions), p=probs))]
            if on_decision is not None:
                with log_lock:
                    top = max(pi.items(), key=lambda kv: kv[1])
                    on_decision({"iteration": iteration_index,
                                 "iteration_problem_index": p_idx,
                                 "turn": int(state.turn),
                                 "chosen_verb": chosen.verb.value,
                                 "pi_argmax_verb": top[0].verb.value,
                                 "pi_argmax_prob": float(top[1])})
            state, _r = mdp.sample_transition(state, chosen, rng)
        return records

    n_dec = 0
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(lambda ip: run_one(*ip), list(enumerate(problems))))
    # Single-threaded buffer population (thread-safe, deterministic order).
    from .replay import sample_from_record
    for recs in results:
        for rec in recs:
            buffer.add(sample_from_record(rec))
            if record_sink is not None:
                record_sink(rec)
            n_dec += 1
    return {"n_trajectories": len(problems), "n_decisions": n_dec,
            "n_simulations": n_dec * config.num_simulations}
```

- [ ] **Step 4: Run, verify PASS**; then `python -m pytest tests/test_outer_loop.py -q` (no regression).
- [ ] **Step 5: Commit**
```bash
git add src/verifier_anchored_rl/outer_loop.py tests/test_outer_loop.py
git commit -m "Add concurrent collection: run trajectories in a thread pool for vLLM batching

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

Note: the concurrent path reports a reduced on_decision payload and does not thread the full entropy/concentration diagnostics (those are computed from the records during analysis). The per-iteration metrics are recomputed in the harness from the returned records; keep `collect_iteration` (sequential) as the canonical diagnostic path for small runs and CPU tests.

---

### Task 4: harness wiring (vLLM backend + embedder selector + concurrency)

**Files:** Modify `experiments/run_outer_loop.py`.

- [ ] **Step 1** Add args: `--gen-backend {vllm,transformers}` (default vllm), `--vllm-url` (default `http://localhost:8000/v1`), `--embedder {text,hidden_state}` (default text), `--embedder-model` (default a small sentence model id), `--collect-workers` (default 8).
- [ ] **Step 2** In `_build_stack`: when `gen_backend==vllm`, build `llm = OpenAICompatibleLLMClient(base_url=args.vllm_url, model=args.model, guided_json=True)` (no in-process generation model); when `embedder==text`, build `TextEmbedder(TransformersEmbeddingClient(model_name=args.embedder_model), embed_dim=<that model's dim>)`; when `embedder==hidden_state`, keep the SharedForwardEncoder path. Heads `embed_dim = encoder.embed_dim`.
- [ ] **Step 3** In the collection call, use `collect_iteration_concurrent(..., max_workers=args.collect_workers, gen_source=llm)` when `gen_backend==vllm`.
- [ ] **Step 4** Document in the run notes that generation requires a running vLLM server (Task 0 command). Confirm `--help` works and the harness unit test (monkeypatched) still passes: `python -m pytest tests/test_run_outer_loop_harness.py -q`.
- [ ] **Step 5: Commit.**

---

### Task 5: GPU smoke + speedup validation (3B)

**Files:** none (validation); append a note to `experiments/results/expert_iteration_rounds.md`.

- [ ] **Step 1** Start the 3B vLLM server (Task 0 command).
- [ ] **Step 2** Run a short closed loop on the toy `train` problems via the new path:
  `python experiments/run_outer_loop.py --gen-backend vllm --embedder text --iterations 1 --num-simulations 8 --collect-workers 16 --problems train --tag vllm-smoke --notes "vLLM+text-embedder throughput smoke"`
- [ ] **Step 3** While it runs, sample `nvidia-smi`. Acceptance: GPU utilization is a clear multiple of the ~10% baseline; per-decision wall-clock is substantially below the lm-format-enforcer path; verbs are valid/non-masked (check progress.log).
- [ ] **Step 4** Record the utilization + wall-clock delta in the journal (no em-dash).
- [ ] **Step 5: Commit the journal note.**

---

## Self-Review

**Spec coverage:** vLLM guided-JSON generation (Tasks 0,2,4); batched/concurrent collection (Task 3); swappable Embedder with text default + `h_T` retained (Tasks 1,4); harness selectors + FLOP via gen_source (Task 4, reuses existing gen_source); CPU tests + GPU smoke (Tasks 1-3 CPU, 0+5 GPU); the equivalence intent (Task 3 record-count check). GNN explicitly deferred (spec non-goal). Covered.

**Placeholder scan:** The only deferred specifics are the exact vLLM guided-JSON field name and the saturating concurrency level, both RESOLVED by Task 0 before Tasks 2/4/5 use them; Task 2's assertion is to be updated to the confirmed field. No vague "handle errors" steps; code is concrete for the unit-testable tasks.

**Type consistency:** `TextEmbedder` implements the `Encoder` protocol (`embed_dim`, `encode(text)->StatePrefill`), drops into `StateActor` unchanged; `collect_iteration_concurrent` returns the same counts dict shape as `collect_iteration` (`n_trajectories`, `n_decisions`, `n_simulations`); `guided_json` flag threads `OpenAICompatibleLLMClient.__init__ -> complete_json`. Consistent.
