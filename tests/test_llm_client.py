"""Tests for OpenAICompatibleLLMClient (guided_json support and FLOP counters)."""


def test_openai_client_sends_guided_json_when_enabled(monkeypatch):
    import pytest
    pytest.importorskip("httpx")
    import httpx
    from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient

    captured = {}

    class _Resp:
        def raise_for_status(self):
            pass
        def json(self):
            return {"choices": [{"message": {"content": '{"answer": "42"}'}}]}

    class _Client:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
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
    assert captured["body"].get("guided_json") == schema


def test_openai_client_omits_guided_json_when_disabled(monkeypatch):
    import pytest
    pytest.importorskip("httpx")
    import httpx
    from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient

    captured = {}

    class _Resp:
        def raise_for_status(self):
            pass
        def json(self):
            return {"choices": [{"message": {"content": '{"answer": "42"}'}}]}

    class _Client:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def post(self, url, headers=None, json=None):
            captured["body"] = json
            return _Resp()

    monkeypatch.setattr(httpx, "Client", _Client)
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    client = OpenAICompatibleLLMClient(base_url="http://x/v1", model="m")  # default: no guided_json
    client.complete_json(system="s", user="u", schema=schema)
    assert "guided_json" not in captured["body"]


def test_openai_client_counts_generate_calls_and_tokens(monkeypatch):
    """generate_calls increments per call; generated_tokens accumulates completion_tokens."""
    import pytest
    pytest.importorskip("httpx")
    import httpx
    from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient

    class _Resp:
        def raise_for_status(self):
            pass
        def json(self):
            return {
                "choices": [{"message": {"content": '{"answer": "42"}'}}],
                "usage": {"completion_tokens": 7},
            }

    class _Client:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def post(self, url, headers=None, json=None):
            return _Resp()

    monkeypatch.setattr(httpx, "Client", _Client)
    schema = {"type": "object", "properties": {"answer": {"type": "string"}},
              "required": ["answer"]}
    client = OpenAICompatibleLLMClient(base_url="http://x/v1", model="m")
    client.complete_json(system="s", user="u", schema=schema)
    client.complete_json(system="s", user="u", schema=schema)
    assert client.generate_calls == 2
    assert client.generated_tokens == 14


def test_openai_client_gen_counters_default_zero_and_tolerate_missing_usage(monkeypatch):
    """Fresh client starts at zero; a call with no usage key leaves generated_tokens at 0."""
    import pytest
    pytest.importorskip("httpx")
    import httpx
    from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient

    # Check fresh-client defaults before any HTTP interaction.
    client = OpenAICompatibleLLMClient(base_url="http://x/v1", model="m")
    assert client.generate_calls == 0
    assert client.generated_tokens == 0

    class _RespNoUsage:
        def raise_for_status(self):
            pass
        def json(self):
            return {"choices": [{"message": {"content": '{"answer": "42"}'}}]}

    class _Client:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def post(self, url, headers=None, json=None):
            return _RespNoUsage()

    monkeypatch.setattr(httpx, "Client", _Client)
    schema = {"type": "object", "properties": {"answer": {"type": "string"}},
              "required": ["answer"]}
    client.complete_json(system="s", user="u", schema=schema)
    assert client.generate_calls == 1
    assert client.generated_tokens == 0
