"""Tests for the shared single-forward encoder.

Two layers: (1) the prompt split is byte-exact so the HTTP/mock paths are
unchanged and the shared prefix ends at the [STATE_END] sentinel; (2) the
encoder's bookkeeping and parse contract, with fakes mirroring the HF
generate contract (generate returns input + continuation). The numerical
correctness of cache reuse (hidden-state and greedy-generation equivalence
against a fresh full prefill) needs a real model and is verified by
experiments/run_shared_forward_smoke.py on the GPU box.
"""

from __future__ import annotations

import pytest

from verifier_anchored_rl.llm_context_mdp import (
    Hypothesis,
    MathState,
    Subgoal,
    Verb,
    _prompt_for,
    _state_prefix,
    _verb_suffix,
    available_actions,
    serialize_state,
)


# -----------------------------------------------------------------------------
# Prompt split (no torch)
# -----------------------------------------------------------------------------


def _rich_state() -> MathState:
    return MathState(
        problem="Solve for x: 2x + 5 = 13.",
        open_subgoals=(Subgoal(id="sg_root", statement="Produce the final answer."),),
        hypotheses=(Hypothesis(id="hyp_1", statement="x = 4.", parent_subgoal="sg_root"),),
        turn=1,
    )


def test_split_is_byte_identical_to_prompt_for():
    state = _rich_state()
    for verb in available_actions(state):
        assert _state_prefix(state) + _verb_suffix(verb, state) == _prompt_for(verb, state)


def test_state_prefix_ends_at_sentinel():
    state = _rich_state()
    prefix = _state_prefix(state)
    # The shared prefix ends exactly at the serialized state's sentinel, so
    # h_T (the last prefix token) is the [STATE_END] position.
    assert prefix.endswith("[STATE_END]")
    assert serialize_state(state) in prefix


def test_state_prefix_is_verb_independent():
    state = _rich_state()
    verbs = list(available_actions(state))
    prefixes = {_state_prefix(state) for _ in verbs}  # prefix ignores verb
    assert len(prefixes) == 1
    # The suffix, by contrast, varies with the verb.
    suffixes = {_verb_suffix(v, state) for v in verbs}
    assert len(suffixes) == len(verbs)


# -----------------------------------------------------------------------------
# Encoder bookkeeping + parse contract (fakes; torch for tensors)
# -----------------------------------------------------------------------------

torch = pytest.importorskip("torch")

from verifier_anchored_rl.shared_forward import SharedForwardEncoder, StatePrefill
from verifier_anchored_rl.llm_client import LLMResponseError


class FakeConfig:
    hidden_size = 8


class FakeTokenizerOut:
    def __init__(self, ids):
        self.input_ids = ids


class FakeTokenizer:
    eos_token_id = 0

    def __init__(self, continuation_text):
        self.continuation_text = continuation_text
        self.decode_calls = 0

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        # token count proportional to text length, deterministic
        n = max(1, len(text) // 5)
        return FakeTokenizerOut(torch.arange(1, n + 1).unsqueeze(0))

    def decode(self, token_ids, skip_special_tokens=True):
        self.decode_calls += 1
        return self.continuation_text


class FakeModelOutput:
    def __init__(self, hidden, cache):
        # hidden_states: tuple of layers; take [-1], shape (1, seq, d)
        self.hidden_states = (hidden,)
        self.past_key_values = cache


class FakeModel:
    device = torch.device("cpu")
    config = FakeConfig()

    def __init__(self, continuation_len=5):
        self.continuation_len = continuation_len
        self.forward_calls = 0
        self.generate_calls = 0

    def __call__(self, input_ids, use_cache=False, output_hidden_states=False):
        self.forward_calls += 1
        seq = input_ids.shape[1]
        hidden = torch.randn(1, seq, self.config.hidden_size)
        cache = {"len": seq}  # opaque fake cache
        return FakeModelOutput(hidden, cache)

    def generate(self, input_ids, attention_mask=None, past_key_values=None,
                 **kwargs):
        # HF contract: returns input_ids ++ continuation.
        self.generate_calls += 1
        cont = torch.full((1, self.continuation_len), 9, dtype=torch.long)
        return torch.cat([input_ids, cont], dim=1)


SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}},
          "required": ["answer"]}


def _encoder(continuation='{"answer": "42"}'):
    return SharedForwardEncoder(
        model=FakeModel(), tokenizer=FakeTokenizer(continuation)
    )


def test_encode_returns_h_T_and_counts_prefill():
    enc = _encoder()
    prefill = enc.encode("State:\n...[STATE_END]")
    assert isinstance(prefill, StatePrefill)
    assert prefill.hidden.shape == (8,)  # h_T is 1-D of hidden_size
    assert enc.prefill_calls == 1
    assert enc.prefill_tokens == prefill.n_prefix_tokens > 0
    assert enc.embed_dim == 8


def test_instantiate_parses_and_counts_generation():
    enc = _encoder('{"answer": "42"}')
    prefill = enc.encode("State:\n...[STATE_END]")
    parsed, n_gen = enc.instantiate_json(prefill, "\n\nVerb: COMMIT_ANSWER", SCHEMA)
    assert parsed == {"answer": "42"}
    assert n_gen == 5  # the fake continuation length
    assert enc.gen_calls == 1
    assert enc.gen_tokens == 5


def test_one_prefill_serves_multiple_verbs():
    # The efficiency claim: prefill once, generate several verbs from it.
    enc = _encoder('{"answer": "x"}')
    prefill = enc.encode("State:\n...[STATE_END]")
    for _ in range(3):
        enc.instantiate_json(prefill, "\n\nVerb: COMMIT_ANSWER", SCHEMA)
    assert enc.prefill_calls == 1          # state prefilled ONCE
    assert enc.gen_calls == 3              # three verb generations
    assert enc.model.forward_calls == 1    # only the prefill hit the model.__call__


def test_generate_text_returns_raw_decoded_text():
    # generate_text is the cache-reuse generation without JSON parsing, so
    # the cache-equivalence smoke can compare raw tokens directly.
    enc = _encoder("arbitrary non-json continuation")
    prefill = enc.encode("s[STATE_END]")
    text, n_gen = enc.generate_text(prefill, "suffix", temperature=0.0)
    assert text == "arbitrary non-json continuation"
    assert n_gen == 5
    assert enc.gen_calls == 1


def test_numeric_answer_coerced_by_shared_validator():
    enc = _encoder('{"answer": 42}')
    prefill = enc.encode("s[STATE_END]")
    parsed, _ = enc.instantiate_json(prefill, "suffix", SCHEMA)
    assert parsed == {"answer": "42"}


def test_malformed_generation_raises():
    enc = _encoder("not json at all")
    prefill = enc.encode("s[STATE_END]")
    with pytest.raises(LLMResponseError):
        enc.instantiate_json(prefill, "suffix", SCHEMA)


def test_reuse_cache_false_does_not_deepcopy():
    # With reuse_cache=False the prefill cache is consumed in place; the
    # fake cache is opaque, so we only assert the call path runs and counts.
    enc = _encoder('{"answer": "x"}')
    prefill = enc.encode("s[STATE_END]")
    enc.instantiate_json(prefill, "suffix", SCHEMA, reuse_cache=False)
    assert enc.gen_calls == 1
