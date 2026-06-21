"""Schema-constrained decoding for verb instantiation.

A small model cannot reliably emit schema-valid JSON for the verb instantiator:
left unconstrained it produces prose, an invented bracket DSL, or repeated
special tokens, all of which fail schema validation and collapse the transition
to a masked no-op. Constraining decoding to the verb's JSON schema removes that
failure mode at the source: every generated token is restricted to those that
keep the output a valid prefix of a schema-conformant JSON object.

We use lm-format-enforcer's core API (``JsonSchemaParser`` + ``TokenEnforcer``)
and deliberately do NOT import ``lmformatenforcer.integrations.transformers``:
that module fails on transformers 5.x because it imports
``PreTrainedTokenizerBase`` from ``transformers.tokenization_utils``, a path that
moved. The few lines that helper would run (build the regular-token list, wrap a
decode function, assemble ``TokenEnforcerTokenizerData``) are replicated here
against the stable tokenizer surface (``encode``/``decode``/``all_special_ids``/
``eos_token_id``/``len``), so this module depends only on the parts of
lm-format-enforcer that import cleanly.

The public entry point ``build_prefix_allowed_tokens_fn(tokenizer, schema)``
returns a callable for ``model.generate(..., prefix_allowed_tokens_fn=...)``.
``assert_schema_parseable(schema, sample_json)`` is a model-free guard used in
tests to catch schemas lm-format-enforcer cannot parse (e.g. mixed-type enums).
"""

from __future__ import annotations

import functools
from typing import Any, Callable

# Building the regular-token list walks the whole vocabulary, which is the
# expensive part; cache it per tokenizer object so repeated verb generations
# reuse it. Keyed by id(tokenizer): the harness holds one tokenizer for the run.
_TOKENIZER_DATA_CACHE: dict[int, Any] = {}


def _build_regular_tokens_list(tokenizer: Any, vocab_size: int):
    """Replicates lm-format-enforcer's transformers helper of the same name,
    without importing its (transformers-5.x-broken) integration module."""
    token_0 = tokenizer.encode("0")[-1]
    special = set(tokenizer.all_special_ids)
    regular = []
    for token_idx in range(vocab_size):
        if token_idx in special:
            continue
        # Prepend token "0" and drop its first character: this surfaces a
        # leading space for word-start tokens, which lm-format-enforcer needs.
        decoded_after_0 = tokenizer.decode([token_0, token_idx])[1:]
        decoded_regular = tokenizer.decode([token_idx])
        is_word_start = len(decoded_after_0) > len(decoded_regular)
        regular.append((token_idx, decoded_after_0, is_word_start))
    return regular


def _decode(tokenizer: Any, tokens: list[int]) -> str:
    return tokenizer.decode(tokens).rstrip("�")


def _tokenizer_data(tokenizer: Any):
    key = id(tokenizer)
    cached = _TOKENIZER_DATA_CACHE.get(key)
    if cached is None:
        from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData

        vocab_size = len(tokenizer)
        regular = _build_regular_tokens_list(tokenizer, vocab_size)
        cached = TokenEnforcerTokenizerData(
            regular,
            functools.partial(_decode, tokenizer),
            tokenizer.eos_token_id,
            False,  # use_bitmask: return explicit token-id lists
            vocab_size,
        )
        _TOKENIZER_DATA_CACHE[key] = cached
    return cached


def build_prefix_allowed_tokens_fn(
    tokenizer: Any, schema: dict[str, Any]
) -> Callable[[int, Any], list[int]]:
    """Return a ``prefix_allowed_tokens_fn`` enforcing ``schema`` during
    ``model.generate``. The callable maps ``(batch_id, input_ids_tensor)`` to
    the list of token ids that keep the decoded suffix a valid JSON prefix of
    a schema-conformant object."""
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.tokenenforcer import TokenEnforcer

    enforcer = TokenEnforcer(_tokenizer_data(tokenizer), JsonSchemaParser(schema))

    def prefix_allowed_tokens_fn(batch_id: int, input_ids: Any) -> list[int]:
        return enforcer.get_allowed_tokens(input_ids.tolist()).allowed_tokens

    return prefix_allowed_tokens_fn


def assert_schema_parseable(schema: dict[str, Any], sample_json: str):
    """Drive ``sample_json`` character by character through the schema parser,
    raising if lm-format-enforcer cannot parse ``schema`` (e.g. an unsupported
    mixed-type enum). Mirrors the per-token path ``TokenEnforcer`` walks, on
    CPU with no tokenizer and no model. Returns the final parser state."""
    from lmformatenforcer import JsonSchemaParser

    parser = JsonSchemaParser(schema)
    for ch in sample_json:
        parser = parser.add_character(ch)
    return parser


__all__ = ["assert_schema_parseable", "build_prefix_allowed_tokens_fn"]
