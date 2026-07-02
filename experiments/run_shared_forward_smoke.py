"""Verify the shared single-forward encoder against a real model (frodo).

The cache-reuse mechanics (method.tex 3.4 ``one forward, not two``) are
correct only if, against real attention:

1. the hidden state h_T from the shared prefill equals h_T from a
   standalone forward over the same prefix (the heads see the same
   embedding either way), and
2. greedy verb generation from the cached prefill is token-identical to
   greedy generation from a fresh full (prefix+suffix) prefill (the
   generator produces the same content; KV-cache reuse is mathematically
   exact under greedy decoding).

If either fails, the cache plumbing is wrong and the one-forward claim is
unsupported. The script also reports the prefill-token saving of doing one
state prefill plus N short verb suffixes versus N full re-prefills.

Run ON the GPU box:

    python experiments/run_shared_forward_smoke.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import sys

from verifier_anchored_rl import (
    DeliberationMDP,
    TransformersLLMClient,
    available_actions,
)
from verifier_anchored_rl.deliberation_mdp import DEFAULT_SYSTEM_PROMPT
from verifier_anchored_rl.lenses import render_view, suffix
from verifier_anchored_rl.shared_forward import SharedForwardEncoder

try:
    from problems import PROBLEMS
except ImportError:
    from experiments.problems import PROBLEMS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--problem", default="algebra_toy_1", choices=list(PROBLEMS.keys()))
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--dtype", default=None,
                   help="float32 makes cache-reuse vs fresh-prefill bit-stable; "
                        "default (bf16 on cuda) leaves a last-token numerical floor")
    args = p.parse_args(argv)

    import torch

    print(f"=== shared-forward encoder smoke (dtype={args.dtype or 'default'}) ===",
          flush=True)
    llm = TransformersLLMClient(args.model, dtype=args.dtype)
    enc = SharedForwardEncoder.from_llm_client(llm)
    print(f"loaded {args.model} (hidden_size={enc.embed_dim})", flush=True)

    problem = PROBLEMS[args.problem]
    mdp = DeliberationMDP(
        problem=problem["problem"],
        expected_answer=problem["expected"],
        llm_client=llm,
        depth_limit=3,
    )
    state = mdp.initial_state()

    # The shared prefix is the system preamble + the target-scoped view,
    # ending where the verb suffix picks up (mirrors
    # DeliberationMDP._generate: render_view(...) + suffix(...)). The first
    # legal action's target anchors both the h_T equivalence check and the
    # generation-equivalence check below.
    action = next(iter(available_actions(state, depth_limit=mdp.depth_limit)))
    verb, target = action.verb, action.target
    prefix_text = DEFAULT_SYSTEM_PROMPT + "\n\n" + render_view(state.graph, target)

    # -- Property 1: h_T equivalence --
    print("\n-- h_T equivalence (shared prefill vs standalone forward) --", flush=True)
    prefill = enc.encode(prefix_text)
    ids = enc.tokenizer(prefix_text, return_tensors="pt").input_ids.to(llm.model.device)
    with torch.no_grad():
        standalone = llm.model(ids, output_hidden_states=True).hidden_states[-1][0, -1]
    max_abs = float((prefill.hidden.float() - standalone.float()).abs().max())
    print(f"  max|h_T_shared - h_T_standalone| = {max_abs:.2e}  "
          f"(dim={prefill.hidden.shape[0]}, n_prefix={prefill.n_prefix_tokens})",
          flush=True)
    h_ok = max_abs < 1e-3

    # -- Property 2: greedy generation equivalence on the first real verb --
    # Compared at the RAW TOKEN level (decoded text), not parsed JSON: the
    # cache-reuse property is about the generated tokens being identical,
    # independent of whether the small model's output is schema-valid.
    suffix_text = suffix(verb, target, state)
    print(f"\n-- greedy generation equivalence (verb={verb.value}) --", flush=True)

    cached_text, _ = enc.generate_text(
        prefill, suffix_text, temperature=0.0, max_tokens=args.max_tokens
    )

    full_ids = enc.tokenizer(
        prefix_text + suffix_text, return_tensors="pt"
    ).input_ids.to(llm.model.device)
    with torch.no_grad():
        full_gen = llm.model.generate(
            full_ids, max_new_tokens=args.max_tokens, do_sample=False,
            temperature=None, top_p=None, top_k=None,
            pad_token_id=enc.tokenizer.eos_token_id,
        )
    fresh_text = enc.tokenizer.decode(
        full_gen[0][full_ids.shape[1]:], skip_special_tokens=True
    )
    gen_ok = cached_text == fresh_text
    print(f"  cached-prefill : {cached_text!r}", flush=True)
    print(f"  fresh-prefill  : {fresh_text!r}", flush=True)
    print(f"  => token-identical: {gen_ok}", flush=True)

    # -- FLOP saving: 1 state prefill + N suffixes vs N full re-prefills --
    n_verbs = len([a for a in available_actions(state, depth_limit=mdp.depth_limit)
                   if a.target == target])
    n_prefix = prefill.n_prefix_tokens
    suffix_tok = enc.tokenizer(
        suffix_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.shape[1]
    shared_prefill_tokens = n_prefix + n_verbs * suffix_tok
    naive_prefill_tokens = n_verbs * (n_prefix + suffix_tok)
    print(f"\n-- prefill-token cost for {n_verbs} verbs at this node --", flush=True)
    print(f"  shared (1 state prefill + {n_verbs} suffixes): {shared_prefill_tokens} tok",
          flush=True)
    print(f"  naive  ({n_verbs} full re-prefills):           {naive_prefill_tokens} tok",
          flush=True)
    print(f"  saving: {100 * (1 - shared_prefill_tokens / naive_prefill_tokens):.0f}%",
          flush=True)

    print("\nSMOKE", "PASS" if (h_ok and gen_ok) else "CHECK OUTPUT", flush=True)
    return 0 if (h_ok and gen_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
