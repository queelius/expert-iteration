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


def test_http_client_backs_off_on_429_then_succeeds(monkeypatch):
    """A 429 gets patience (backoff + unchanged messages), not a corrective
    prompt; the call succeeds once the server stops rate-limiting."""
    import httpx

    from verifier_anchored_rl import llm_client as mod

    sleeps: list[float] = []
    monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))

    calls = {"n": 0}

    class _Resp:
        def __init__(self, status):
            self.status_code = status
            self.headers = {}

        def raise_for_status(self):
            if self.status_code != 200:
                raise httpx.HTTPStatusError(
                    "rate limited", request=httpx.Request("POST", "http://x"),
                    response=self)

        def json(self):
            return {"choices": [{"message": {"content": '{"action": "ok"}'}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3}}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers=None, json=None):
            calls["n"] += 1
            _FakeClient.last_body = json
            return _Resp(429 if calls["n"] <= 2 else 200)

    monkeypatch.setattr(httpx, "Client", _FakeClient)

    client = mod.OpenAICompatibleLLMClient(base_url="http://x/v1", model="m")
    result = client.complete_json("sys", "user", {"properties": {"action": {}}})
    assert result == {"action": "ok"}
    assert calls["n"] == 3
    assert len(sleeps) == 2                       # backed off twice
    assert len(_FakeClient.last_body["messages"]) == 2   # no corrective prompt
    assert client.prompt_tokens == 7 and client.generated_tokens == 3


def test_http_client_repr_never_contains_the_api_key():
    from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient
    secret = "sk-proj-SUPERSECRET123"
    client = OpenAICompatibleLLMClient(base_url="http://x/v1", model="m",
                                       api_key=secret)
    assert secret not in repr(client)
    assert secret not in str(client)


def test_parse_retry_echoes_failed_reply_before_correction(monkeypatch):
    """The corrective retry says "your previous response could not be
    parsed", so that response must be in the transcript as an assistant
    message; without it the model is scolded about a reply it cannot see."""
    import pytest
    pytest.importorskip("httpx")
    import httpx
    from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient

    requests = []
    replies = ["not json at all", '{"answer": "42"}']

    class _Resp:
        def __init__(self, content):
            self._content = content
        def raise_for_status(self):
            pass
        def json(self):
            return {"choices": [{"message": {"content": self._content}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    class _Client:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def post(self, url, headers=None, json=None):
            requests.append([dict(m) for m in json["messages"]])
            return _Resp(replies[len(requests) - 1])

    monkeypatch.setattr(httpx, "Client", _Client)
    schema = {"type": "object", "properties": {"answer": {"type": "string"}},
              "required": ["answer"]}
    client = OpenAICompatibleLLMClient(base_url="http://x/v1", model="m")
    out = client.complete_json(system="s", user="u", schema=schema)
    assert out == {"answer": "42"}
    second = requests[1]
    roles = [m["role"] for m in second]
    assert roles == ["system", "user", "assistant", "user"]
    assert second[2]["content"] == "not json at all"
    assert "could not be parsed" in second[3]["content"]
