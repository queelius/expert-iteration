"""Shared single-forward encoder: one prefill of the state yields both the
state embedding for the heads and the KV cache for verb generation.

This realizes method.tex Section 3.4's ``one forward, not two``. The
verb-instantiator generates content ``conditioned on a small verb-specific
prompt prefix`` as a continuation from the ``[STATE_END]`` sentinel, so the
same forward that produces ``embed(s) = h_T`` also seeds generation. The
pilot's HTTP path could not do this: it ran one forward for the embedding
(the Ollama embed endpoint) and a second, independent forward that
re-encoded the entire state for generation. Collapsing those to a single
state prefill is the architecture's efficiency core and the precondition
for honest per-decision FLOP accounting (you cannot claim ``one forward per
decision`` until generation actually reuses the encoder's forward).

Shape of one decision node:

* ``encode(prefix_text)`` prefills the shared state span once and returns
  ``h_T`` (last-layer hidden at the final prefix token, the sentinel) plus
  the prefill's KV cache. The Q and policy heads score all verbs from
  ``h_T`` with no further LLM forward (the multi-head ``Q`` returns all
  seven values in one head pass).
* ``instantiate_json(prefill, verb_suffix, ...)`` extends the cached
  prefix with the short per-verb suffix and generates the action content,
  so the expensive state prefill is paid once even though the heads and
  the generator both consume it.

The module is generic over the text spans (a ``prefix_text`` and a
``suffix_text``); the faithful spans for the LLM context MDP come from
``llm_context_mdp._state_prefix`` / ``_verb_suffix``. The frozen-model
forward is GPU-bound, so correctness of the cache reuse (hidden-state and
greedy-generation equivalence against a fresh full prefill) is verified by
``experiments/run_shared_forward_smoke.py`` on the GPU box; the bookkeeping
and parse contract are unit-tested on CPU with fakes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from .llm_client import (
    LLMResponseError,
    _parse_json_lenient,
    _validate_against_schema,
)


@dataclass
class StatePrefill:
    """The result of prefilling the shared state span once.

    ``hidden`` is ``embed(s) = h_T`` (the last-layer hidden state at the
    final prefix token, the ``[STATE_END]`` sentinel), a 1-D tensor of
    length ``d``. ``cache`` is the prefill's KV cache; ``n_prefix_tokens``
    is the prefill length, for FLOP accounting.
    """

    hidden: Any  # torch.Tensor [d]
    cache: Any  # past_key_values
    n_prefix_tokens: int
    prefix_ids: Any = None  # torch.Tensor [1, n_prefix], for full-seq generate


class SharedForwardEncoder:
    """Prefill-once, reuse-for-generation wrapper over an in-process model.

    Construct from a model + tokenizer, or share the weights already loaded
    by a ``TransformersLLMClient`` via ``from_llm_client`` (the pilot's
    one-frozen-model design: the encoder and the generator are the same
    weights, so representation drift is bounded by construction).
    """

    def __init__(self, model: Any, tokenizer: Any, device: str | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or str(getattr(model, "device", "cpu"))
        # FLOP-accounting counters (token counts are the implementation-
        # independent proxy for decoder forwards).
        self.prefill_calls = 0
        self.prefill_tokens = 0
        self.gen_calls = 0
        self.gen_tokens = 0

    @classmethod
    def from_llm_client(cls, llm_client: Any) -> "SharedForwardEncoder":
        return cls(model=llm_client.model, tokenizer=llm_client.tokenizer)

    @property
    def embed_dim(self) -> int:
        return int(self.model.config.hidden_size)

    # -- the shared prefill --------------------------------------------------

    def encode(self, prefix_text: str) -> StatePrefill:
        """One forward over the shared state span. Returns ``h_T`` and the
        reusable KV cache."""
        import torch

        prefix_ids = self.tokenizer(
            prefix_text, return_tensors="pt"
        ).input_ids.to(self.model.device)
        with torch.no_grad():
            out = self.model(
                prefix_ids, use_cache=True, output_hidden_states=True
            )
        hidden = out.hidden_states[-1][0, -1]  # h_T at the sentinel
        self.prefill_calls += 1
        self.prefill_tokens += int(prefix_ids.shape[1])
        return StatePrefill(
            hidden=hidden,
            cache=out.past_key_values,
            n_prefix_tokens=int(prefix_ids.shape[1]),
            prefix_ids=prefix_ids,
        )

    # -- verb generation from the cached prefill -----------------------------

    def generate_text(
        self,
        prefill: StatePrefill,
        suffix_text: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 192,
        seed: int | None = None,
        reuse_cache: bool = True,
    ) -> tuple[str, int]:
        """Generate the verb content by extending the cached state prefill
        with ``suffix_text``. Returns ``(raw_decoded_text, n_generated)``.

        This is the cache-reuse generation in isolation, without JSON
        parsing, so the cache-equivalence property (raw tokens identical
        to a fresh full prefill under greedy decoding) can be checked
        directly. With ``reuse_cache=True`` (default) the prefill's cache
        is deep-copied first, so the same ``StatePrefill`` can seed several
        verbs (or retries); ``False`` consumes it in place, copy-free.
        """
        import torch

        suffix_ids = self.tokenizer(
            suffix_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.model.device)

        # Pass the FULL sequence (prefix + suffix) plus the prefix cache:
        # generate detects that the cache covers the first n_prefix
        # positions and recomputes only the suffix onward, while the
        # full-length input lets it build a correct attention mask and
        # position ids. Passing the suffix alone makes generate infer a
        # wrong mask (the prefix is not attended to) and the continuation
        # degenerates. Deep-copy the cache so the prefill stays reusable.
        full_ids = torch.cat([prefill.prefix_ids.to(suffix_ids.device), suffix_ids], dim=1)
        cache = copy.deepcopy(prefill.cache) if reuse_cache else prefill.cache

        if seed is not None:
            torch.manual_seed(seed)
        do_sample = temperature > 0
        attention_mask = torch.ones_like(full_ids)

        with torch.no_grad():
            output = self.model.generate(
                full_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0][full_ids.shape[1] :]
        self.gen_calls += 1
        n_gen = int(new_tokens.shape[0])
        self.gen_tokens += n_gen
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text, n_gen

    def instantiate_json(
        self,
        prefill: StatePrefill,
        suffix_text: str,
        schema: Mapping[str, Any],
        *,
        temperature: float = 1.0,
        max_tokens: int = 192,
        seed: int | None = None,
        reuse_cache: bool = True,
    ) -> tuple[dict[str, Any], int]:
        """Generate verb content from the cached prefill and parse it as a
        schema-conformant JSON object. Returns ``(parsed_json, n_gen)``.

        Raises ``LLMResponseError`` on malformed or schema-violating output
        (the MDP's masked-action recovery handles this upstream, exactly as
        for the HTTP client)."""
        text, n_gen = self.generate_text(
            prefill,
            suffix_text,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            reuse_cache=reuse_cache,
        )
        parsed = _parse_json_lenient(text)
        _validate_against_schema(parsed, schema)
        return parsed, n_gen

    def stats(self) -> str:
        return (
            f"{self.prefill_calls} prefills ({self.prefill_tokens} tok), "
            f"{self.gen_calls} generations ({self.gen_tokens} tok)"
        )


__all__ = ["SharedForwardEncoder", "StatePrefill"]
