"""Tests for OpenAICompatibleLLMClient (guided_json support and FLOP counters)."""


def test_provider_usage_requires_exact_nonnegative_integers():
    from verifier_anchored_rl.llm_client import (
        project_openai_compatible_response,
    )

    for invalid in (1.5, "1", True, -1):
        projection = project_openai_compatible_response({
            "choices": [{"message": {"content": '{"answer": "42"}'}}],
            "usage": {"completion_tokens": invalid},
        })
        assert projection.provider_error is not None
        assert projection.completion_tokens is None

    projection = project_openai_compatible_response({
        "choices": [{"message": {"content": '{"answer": "42"}'}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 1},
    })
    assert projection.provider_error is None
    assert projection.prompt_tokens == 0
    assert projection.completion_tokens == 1


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
    assert client.generate_calls == 2
    assert client.prompt_tokens == 2
    assert client.generated_tokens == 2


def test_malformed_billable_responses_are_counted_and_preserved(monkeypatch):
    import pytest
    pytest.importorskip("httpx")
    import httpx
    from verifier_anchored_rl.llm_client import (
        LLMResponseError, OpenAICompatibleLLMClient,
    )

    replies = iter(("first malformed", "second malformed"))

    class _Resp:
        def __init__(self, content):
            self._content = content

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{"message": {"content": self._content}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            }

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            return _Resp(next(replies))

    monkeypatch.setattr(httpx, "Client", _Client)
    client = OpenAICompatibleLLMClient(base_url="http://x/v1", model="m")
    with pytest.raises(LLMResponseError) as caught:
        client.complete_json("s", "u", {"properties": {"answer": {}}})

    assert client.generate_calls == 2
    assert client.prompt_tokens == 10
    assert client.generated_tokens == 6
    assert caught.value.raw_response == "second malformed"


def test_schema_validation_modes_preserve_legacy_numeric_string_coercion():
    from verifier_anchored_rl.llm_client import (
        SCHEMA_VALIDATION_LEGACY_COERCE_V1,
        classify_openai_compatible_json_content,
        decode_openai_compatible_json_content,
    )

    schema = {
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["values"],
    }
    content = '{"values":[-5,1.25,"03"]}'

    assert decode_openai_compatible_json_content(content, schema) == {
        "values": ["-5", "1.25", "03"]
    }
    assert classify_openai_compatible_json_content(
        content,
        schema,
        schema_validation_mode=SCHEMA_VALIDATION_LEGACY_COERCE_V1,
    ) == ("accepted", {"values": ["-5", "1.25", "03"]})


def test_strict_schema_validation_rejects_numeric_string_items_without_mutation():
    import pytest

    from verifier_anchored_rl import llm_client as mod

    schema = {
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["values"],
    }
    for values in ([1], [1.5], ["1", 2, "3"]):
        source = {"values": list(values)}
        before = {"values": list(values)}
        with pytest.raises(mod.LLMResponseError, match="expected string"):
            mod._validate_against_schema(
                source,
                schema,
                schema_validation_mode=mod.SCHEMA_VALIDATION_STRICT_V1,
            )
        assert source == before
        content = mod._json_snapshot(before)
        assert mod.classify_openai_compatible_json_content(
            content,
            schema,
            schema_validation_mode=mod.SCHEMA_VALIDATION_STRICT_V1,
        ) == ("schema_error", None)
        with pytest.raises(mod.LLMResponseError, match="expected string"):
            mod.decode_openai_compatible_json_content(
                content,
                schema,
                schema_validation_mode=mod.SCHEMA_VALIDATION_STRICT_V1,
            )


def test_strict_schema_validation_preserves_exact_string_lexemes():
    from verifier_anchored_rl.llm_client import (
        SCHEMA_VALIDATION_STRICT_V1,
        classify_openai_compatible_json_content,
        decode_openai_compatible_json_content,
    )

    schema = {
        "type": "object",
        "properties": {
            "boundaries": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["boundaries"],
    }
    content = '{"boundaries":["-5.0","4/2"," 2 "]}'
    expected = {"boundaries": ["-5.0", "4/2", " 2 "]}

    assert decode_openai_compatible_json_content(
        content,
        schema,
        schema_validation_mode=SCHEMA_VALIDATION_STRICT_V1,
    ) == expected
    assert classify_openai_compatible_json_content(
        content,
        schema,
        schema_validation_mode=SCHEMA_VALIDATION_STRICT_V1,
    ) == ("accepted", expected)


def test_strict_schema_validation_rejects_non_json_and_nonfinite_numbers():
    import pytest

    from verifier_anchored_rl import llm_client as mod

    schema = {
        "type": "object",
        "properties": {"value": {"type": "number"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    for literal in ("NaN", "Infinity", "-Infinity"):
        content = f'{{"value":{literal}}}'
        assert mod.classify_openai_compatible_json_content(
            content,
            schema,
            schema_validation_mode=mod.SCHEMA_VALIDATION_STRICT_V1,
        ) == ("parse_error", None)
        with pytest.raises(mod.LLMResponseError, match="JSON"):
            mod.decode_openai_compatible_json_content(
                content,
                schema,
                schema_validation_mode=mod.SCHEMA_VALIDATION_STRICT_V1,
            )

    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(mod.LLMResponseError, match="expected number"):
            mod._validate_against_schema(
                {"value": value},
                schema,
                schema_validation_mode=mod.SCHEMA_VALIDATION_STRICT_V1,
            )


def test_strict_schema_subset_rejects_shape_cardinality_enum_and_bool_traps():
    from copy import deepcopy

    import pytest

    from verifier_anchored_rl import llm_client as mod

    schema = {
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "items": {"type": "string", "enum": ["a", "b"]},
                "minItems": 1,
                "maxItems": 2,
            },
            "rank": {"type": "integer"},
            "enabled": {"type": "boolean"},
        },
        "required": ["values", "rank", "enabled"],
        "additionalProperties": False,
    }
    valid = {"values": ["a", "b"], "rank": 1, "enabled": True}
    valid_before = deepcopy(valid)
    mod._validate_against_schema(
        valid,
        schema,
        schema_validation_mode=mod.SCHEMA_VALIDATION_STRICT_V1,
    )
    assert valid == valid_before

    invalid_values = [
        {"values": ["a"], "rank": 1, "enabled": True, "extra": "x"},
        {"values": ["a"], "rank": 1},
        {"values": [], "rank": 1, "enabled": True},
        {"values": ["a", "b", "a"], "rank": 1, "enabled": True},
        {"values": ["a", 1], "rank": 1, "enabled": True},
        {"values": ["a", "c"], "rank": 1, "enabled": True},
        {"values": ["a"], "rank": True, "enabled": True},
        {"values": ["a"], "rank": 1, "enabled": 1},
        {"values": ("a",), "rank": 1, "enabled": True},
    ]
    for value in invalid_values:
        before = deepcopy(value)
        with pytest.raises(mod.LLMResponseError):
            mod._validate_against_schema(
                value,
                schema,
                schema_validation_mode=mod.SCHEMA_VALIDATION_STRICT_V1,
            )
        assert value == before


def test_test_clients_cannot_bypass_explicit_strict_validation():
    import pytest

    from verifier_anchored_rl.llm_client import (
        EchoLLMClient,
        LLMResponseError,
        MockLLMClient,
        SCHEMA_VALIDATION_STRICT_V1,
    )

    schema = {
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            }
        },
        "required": ["values"],
        "additionalProperties": False,
    }
    mock = MockLLMClient(
        script={("u",): {"values": [1]}},
        key_fn=lambda *_: ("u",),
    )

    with pytest.raises(LLMResponseError, match="expected string"):
        mock.complete_json(
            "s",
            "u",
            schema,
            schema_validation_mode=SCHEMA_VALIDATION_STRICT_V1,
        )
    with pytest.raises(LLMResponseError, match="minItems"):
        EchoLLMClient().complete_json(
            "s",
            "u",
            schema,
            schema_validation_mode=SCHEMA_VALIDATION_STRICT_V1,
        )


def test_mock_client_reproduces_legacy_numeric_string_coercion():
    from verifier_anchored_rl.llm_client import (
        MockLLMClient,
        SCHEMA_VALIDATION_LEGACY_COERCE_V1,
    )

    schema = {
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["values"],
    }
    scripted = {"values": [-5, 1.25, "03"]}
    client = MockLLMClient(
        script={("u",): scripted},
        key_fn=lambda *_: ("u",),
    )

    assert client.complete_json(
        "s",
        "u",
        schema,
        schema_validation_mode=SCHEMA_VALIDATION_LEGACY_COERCE_V1,
    ) == {"values": ["-5", "1.25", "03"]}
    assert scripted == {"values": [-5, 1.25, "03"]}


def test_strict_schema_error_retries_with_raw_assistant_echo(monkeypatch):
    import httpx

    from verifier_anchored_rl.llm_client import (
        SCHEMA_VALIDATION_STRICT_V1,
        OpenAICompatibleLLMClient,
    )

    raw_replies = (
        '{"boundaries":[-5.0,1]}',
        '{"boundaries":["-5.0","1"]}',
    )
    requests = []
    attempts = []

    class _Response:
        status_code = 200
        headers = {}

        def __init__(self, content):
            self._payload = {
                "choices": [{"message": {"content": content}}],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            }
            self.text = mod_json(self._payload)

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            requests.append(json)
            return _Response(raw_replies[len(requests) - 1])

    def mod_json(value):
        import json

        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    monkeypatch.setattr(httpx, "Client", _Client)
    schema = {
        "type": "object",
        "properties": {
            "boundaries": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["boundaries"],
    }
    client = OpenAICompatibleLLMClient(
        base_url="http://x/v1",
        model="m",
        attempt_observer=attempts.append,
    )

    result = client.complete_json(
        "system",
        "user",
        schema,
        seed=7,
        schema_validation_mode=SCHEMA_VALIDATION_STRICT_V1,
    )

    assert result == {"boundaries": ["-5.0", "1"]}
    assert [attempt.outcome for attempt in attempts] == [
        "schema_error", "accepted"
    ]
    assert [attempt.disposition for attempt in attempts] == [
        "parse_retry", "return"
    ]
    assert [attempt.effective_seed for attempt in attempts] == [7, 8]
    assert [message["role"] for message in requests[1]["messages"]] == [
        "system", "user", "assistant", "user"
    ]
    assert requests[1]["messages"][2]["content"] == raw_replies[0]
    assert "could not be parsed" in requests[1]["messages"][3]["content"]
