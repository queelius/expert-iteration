"""Smoke test for the in-process Transformers backend on the GPU box.

Validates, against real weights, the four properties that motivated the
migration off the HTTP serving stack (see the infra post-mortem in
experiments/results/expert_iteration_rounds.md):

1. The frozen model loads once and serves BOTH roles (generator via
   ``TransformersLLMClient``, encoder via
   ``TransformersEmbeddingClient.share_model_with``), restoring the
   pilot's one-model-two-roles design point with a single copy in VRAM.
2. Seeded sampling is reproducible in-process: the same prompt at the
   same seed and temperature > 0 yields the identical continuation
   (the property the HTTP stack could not guarantee under load).
3. Hidden-state embeddings come straight from the model with a
   documented pooling rule: dimension equals the model's hidden size,
   L2 norm 1, deterministic.
4. Forward truth is countable: the client reports generate calls and
   generated tokens, the accounting the pre-registered compute axis
   needs and an HTTP API hides.

Then runs a tiny MCTS (default B=3) over the LLM context MDP end to end.

Run ON the GPU box (not over HTTP):

    python experiments/run_transformers_smoke.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct --num-simulations 3
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from verifier_anchored_rl import (
    LLMContextMDP,
    MCTSConfig,
    TransformersEmbeddingClient,
    TransformersLLMClient,
    mcts_search,
    serialize_state,
)
from verifier_anchored_rl.llm_context_mdp import (
    _SYSTEM_PROMPT,
    Verb,
    _prompt_for,
    _schema_for,
)

try:
    from problems import PROBLEMS
except ImportError:
    from experiments.problems import PROBLEMS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--problem", default="multiply_small",
                   choices=list(PROBLEMS.keys()))
    p.add_argument("--num-simulations", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=192)
    args = p.parse_args(argv)

    import torch

    print("=== Transformers in-process smoke ===", flush=True)
    print(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}", flush=True)

    t0 = time.time()
    llm = TransformersLLMClient(args.model, max_retries=1)
    embedder = TransformersEmbeddingClient.share_model_with(llm)
    print(f"loaded {args.model} in {time.time() - t0:.1f}s "
          f"(hidden_size={llm.model.config.hidden_size}, "
          f"dtype={next(llm.model.parameters()).dtype}, "
          f"device={llm.model.device})", flush=True)
    if torch.cuda.is_available():
        print(f"VRAM allocated: {torch.cuda.memory_allocated() / 2**30:.2f} GiB",
              flush=True)

    problem = PROBLEMS[args.problem]
    mdp = LLMContextMDP(
        problem_statement=problem["problem"],
        expected_answer=problem["expected"],
        llm_client=llm,
        depth_limit=3,
        max_turns=8,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    state = mdp.initial_state()

    # -- Property 2: matched-seed reproducibility of a single call --
    print("\n-- seeded-generation reproducibility (temp "
          f"{args.temperature}, seed 42) --", flush=True)
    schema = _schema_for(Verb.COMMIT_ANSWER, state)
    user = _prompt_for(Verb.COMMIT_ANSWER, state)
    outs = []
    for i in range(3):
        t0 = time.time()
        r = llm.complete_json(system=_SYSTEM_PROMPT, user=user, schema=schema,
                              temperature=args.temperature,
                              max_tokens=64, seed=42)
        outs.append(r)
        print(f"  call {i}: {r}  ({time.time() - t0:.1f}s)", flush=True)
    identical = all(o == outs[0] for o in outs)
    print(f"  => matched-seed identical: {identical}", flush=True)

    # -- Property 3: hidden-state embedding --
    print("\n-- hidden-state embedding (masked mean pool, L2) --", flush=True)
    serialized = serialize_state(state)
    t0 = time.time()
    emb = embedder.embed(serialized)
    norm = float(np.linalg.norm(emb))
    emb2 = embedder.embed(serialized)
    print(f"  dim={len(emb)}, L2={norm:.4f}, "
          f"deterministic={np.allclose(emb, emb2)}, "
          f"latency={time.time() - t0:.2f}s (first of two)", flush=True)

    # -- End to end: tiny MCTS with countable forwards --
    print(f"\n-- MCTS B={args.num_simulations} on {args.problem!r} --",
          flush=True)
    calls_before = llm.generate_calls
    tokens_before = llm.generated_tokens
    config = MCTSConfig(num_simulations=args.num_simulations, c_puct=1.0,
                        discount=0.99, rollout_depth_limit=0,
                        prior_fn=None, rollout_policy=None)
    t0 = time.time()
    root = mcts_search(mdp, state, config, np.random.default_rng(0))
    dur = time.time() - t0
    visits = {a.verb.value: c.visits for a, c in root.children.items()}
    print(f"  visits={visits}  ({dur:.1f}s)", flush=True)
    print(f"  forward truth: {llm.generate_calls - calls_before} generate "
          f"calls, {llm.generated_tokens - tokens_before} new tokens "
          f"({(llm.generated_tokens - tokens_before) / dur:.1f} tok/s incl. "
          f"search overhead)", flush=True)

    print("\nSMOKE", "PASS" if identical and norm > 0.99 else "CHECK OUTPUT",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
