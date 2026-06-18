"""Tests for the in-process Transformers backends.

The clients accept injected (model, tokenizer) components, so these
tests exercise the real client logic (prompt assembly, seeding, retry,
pooling, normalization, accounting) with lightweight fakes: no model
downloads, no transformers import, CPU-only torch tensors.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verifier_anchored_rl.llm_client import LLMClient, LLMResponseError
from verifier_anchored_rl.embedding_client import EmbeddingClient
from verifier_anchored_rl.transformers_client import (
    TransformersEmbeddingClient,
    TransformersLLMClient,
)


# -----------------------------------------------------------------------------
# Fakes implementing the minimal HF surface the clients touch
# -----------------------------------------------------------------------------


class FakeConfig:
    name_or_path = "fake/test-model"


class FakeTokenizer:
    """Scripted tokenizer: decode() returns queued strings in order."""

    eos_token_id = 0

    def __init__(self, scripted_responses):
        self.scripted = list(scripted_responses)
        self.decode_calls = 0
        self.chat_template_calls = []  # records the messages of each call

    def apply_chat_template(self, messages, add_generation_prompt=True,
                            return_tensors="pt", return_dict=False):
        self.chat_template_calls.append([dict(m) for m in messages])
        ids = torch.arange(1, 6).unsqueeze(0)  # (1, 5) prompt ids
        if return_dict:
            # transformers 5.x returns a BatchEncoding (dict-like).
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        return ids

    def decode(self, token_ids, skip_special_tokens=True):
        text = self.scripted[min(self.decode_calls, len(self.scripted) - 1)]
        self.decode_calls += 1
        return text

    def __call__(self, text, return_tensors="pt", truncation=True,
                 max_length=8192):
        n = 3
        return {
            "input_ids": torch.arange(1, n + 1).unsqueeze(0),
            "attention_mask": torch.ones(1, n, dtype=torch.long),
        }


class FakeOutputs:
    def __init__(self, hidden):
        self.hidden_states = (hidden,)


class FakeModel:
    device = torch.device("cpu")
    config = FakeConfig()

    def __init__(self, continuation_len=4, hidden=None):
        self.continuation_len = continuation_len
        self.hidden = hidden
        self.generate_kwargs = []

    def eval(self):
        # torch nn.Module's train/eval mode method (mimicking the HF
        # interface), NOT Python's eval() builtin; nothing is executed.
        return self

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        # The client calls generate(**encoded, ...), so input_ids and
        # attention_mask arrive as keywords (transformers 5.x contract).
        self.generate_kwargs.append(kwargs)
        continuation = torch.full((1, self.continuation_len), 9, dtype=torch.long)
        return torch.cat([input_ids, continuation], dim=1)

    def __call__(self, input_ids=None, attention_mask=None,
                 output_hidden_states=False):
        assert output_hidden_states
        return FakeOutputs(self.hidden)


def make_llm(scripted, **kwargs):
    return TransformersLLMClient(
        model=FakeModel(), tokenizer=FakeTokenizer(scripted), **kwargs
    )


# -----------------------------------------------------------------------------
# LLM client
# -----------------------------------------------------------------------------

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}},
          "required": ["answer"]}


def test_satisfies_llm_protocol():
    client = make_llm(['{"answer": "42"}'])
    assert isinstance(client, LLMClient)


def test_complete_json_happy_path_and_accounting():
    client = make_llm(['{"answer": "42"}'])
    out = client.complete_json(system="s", user="u", schema=SCHEMA)
    assert out == {"answer": "42"}
    assert client.generate_calls == 1
    assert client.generated_tokens == 4  # the fake continuation length


def test_retry_appends_failed_response_and_correction():
    client = make_llm(["this is not json", '{"answer": "42"}'], max_retries=1)
    out = client.complete_json(system="s", user="u", schema=SCHEMA)
    assert out == {"answer": "42"}
    assert client.generate_calls == 2
    # Second prompt assembly must include the failed assistant reply and
    # the correction message (the HTTP client never showed the model its
    # own failed response; this client does).
    second_call = client.tokenizer.chat_template_calls[1]
    roles = [m["role"] for m in second_call]
    assert roles == ["system", "user", "assistant", "user"]
    assert second_call[2]["content"] == "this is not json"
    assert "could not be parsed" in second_call[3]["content"]


def test_exhausted_retries_raise():
    client = make_llm(["garbage", "more garbage"], max_retries=1)
    with pytest.raises(LLMResponseError, match="no parseable response"):
        client.complete_json(system="s", user="u", schema=SCHEMA)


def test_seed_plumbing_with_per_attempt_bump(monkeypatch):
    seeds = []
    monkeypatch.setattr(torch, "manual_seed", lambda s: seeds.append(int(s)))
    client = make_llm(["garbage", '{"answer": "x"}'], max_retries=1)
    client.complete_json(system="s", user="u", schema=SCHEMA, seed=7)
    assert seeds == [7, 8]


def test_no_seed_means_no_manual_seed(monkeypatch):
    seeds = []
    monkeypatch.setattr(torch, "manual_seed", lambda s: seeds.append(int(s)))
    client = make_llm(['{"answer": "x"}'])
    client.complete_json(system="s", user="u", schema=SCHEMA)
    assert seeds == []


def test_temperature_zero_is_greedy():
    client = make_llm(['{"answer": "x"}'])
    client.complete_json(system="s", user="u", schema=SCHEMA, temperature=0.0)
    kwargs = client.model.generate_kwargs[0]
    assert kwargs["do_sample"] is False
    assert kwargs["temperature"] is None


def test_numeric_answer_coerced_by_shared_validator():
    # The client reuses _validate_against_schema, so the number-to-string
    # coercion from the reward-channel fixes applies here too.
    client = make_llm(['{"answer": 42}'])
    out = client.complete_json(system="s", user="u", schema=SCHEMA)
    assert out == {"answer": "42"}


# -----------------------------------------------------------------------------
# Embedding client
# -----------------------------------------------------------------------------


def make_embedder(hidden, **kwargs):
    return TransformersEmbeddingClient(
        model=FakeModel(hidden=hidden), tokenizer=FakeTokenizer([]), **kwargs
    )


def test_satisfies_embedding_protocol():
    hidden = torch.zeros(1, 3, 4)
    assert isinstance(make_embedder(hidden), EmbeddingClient)


def test_mean_pooling_and_normalization():
    # rows: [1,0,0,0], [0,1,0,0], [0,0,1,0]; masked mean = [1/3,1/3,1/3,0]
    hidden = torch.eye(3, 4).unsqueeze(0)
    emb = torch.tensor(make_embedder(hidden).embed("text"))
    expected = torch.tensor([1 / 3, 1 / 3, 1 / 3, 0.0])
    expected = expected / expected.norm()
    assert torch.allclose(emb, expected, atol=1e-6)
    assert abs(emb.norm().item() - 1.0) < 1e-6


def test_last_token_pooling():
    hidden = torch.eye(3, 4).unsqueeze(0)
    emb = torch.tensor(
        make_embedder(hidden, pooling="last_token").embed("text")
    )
    expected = torch.tensor([0.0, 0.0, 1.0, 0.0])  # row at index sum(mask)-1
    assert torch.allclose(emb, expected, atol=1e-6)


def test_normalize_false_returns_raw_pooled():
    hidden = torch.full((1, 3, 4), 2.0)
    emb = make_embedder(hidden, normalize=False).embed("text")
    assert emb == pytest.approx([2.0, 2.0, 2.0, 2.0])


def test_share_model_with_uses_same_weights_object():
    llm = make_llm(['{"answer": "x"}'])
    embedder = TransformersEmbeddingClient.share_model_with(llm)
    assert embedder.model is llm.model
    assert embedder.tokenizer is llm.tokenizer
