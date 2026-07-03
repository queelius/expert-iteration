"""LLM client abstraction.

The framework calls a frozen base LLM as a stochastic transition oracle:
given a serialized state and a chosen verb, the LLM produces a
verb-instantiated continuation (a target identifier and free-form
content) that is consumed by the next-state assembler. This module
defines the abstraction the rest of the package programs against.

Three implementations are provided:

* ``MockLLMClient`` is a scripted dispatcher used in tests. It returns
  whatever the test author tells it to, indexed by a fingerprint of the
  request. No network, no model, deterministic.

* ``OpenAICompatibleLLMClient`` is an HTTP client that talks to any
  server that speaks the OpenAI chat-completions API with JSON-mode
  response formatting. vLLM, Ollama, llama.cpp's server, and a handful
  of other inference servers all qualify, so the same client code
  works against any of them by changing one URL.

* ``EchoLLMClient`` returns a deterministic minimal response for each
  schema (the lexicographically first allowed enum value or an empty
  string for free fields). Useful for end-to-end MDP smoke tests
  without any real model.

The ``LLMClient`` Protocol is the surface the rest of the package
imports. PEP 544 structural typing means new backends can be added
without changing the Protocol.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """The interface every backend must satisfy.

    Implementations produce a JSON object that conforms to the provided
    ``schema`` (a JSON-Schema-like dict). Implementations may rely on
    the underlying server's JSON-mode support or apply a parse-and-retry
    loop on top of free-form text; either is acceptable. The contract
    is that the return value is a Python ``dict`` whose top-level keys
    match the schema's ``properties`` keys.

    Implementations should raise ``LLMResponseError`` when the
    continuation cannot be coerced into a schema-conformant dict after
    any retries the implementation chooses to make. The MDP's
    transition assembler treats this as the same failure mode that the
    companion paper section 4.4 calls "invalid-id from the
    verb-instantiator": the action is masked and control returns to
    the search.
    """

    def complete_json(
        self,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> dict[str, Any]:
        ...


class LLMResponseError(RuntimeError):
    """Raised when the backend cannot produce a schema-conformant response.

    The MDP transition layer catches this and treats it as a masked
    action. Callers that want to surface the underlying server error
    can read ``self.cause``.
    """

    def __init__(self, message: str, *, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause


# -----------------------------------------------------------------------------
# Mock client (tests)
# -----------------------------------------------------------------------------


ScriptKey = tuple[str, ...]
"""Key into a MockLLMClient script.

The default fingerprint is ``(verb_name,)`` extracted from the user
message. Tests that need finer dispatch can pass a custom
``key_fn`` to fingerprint on more of the request.
"""


@dataclass
class MockLLMClient:
    """Deterministic scripted client for tests.

    The script is a mapping from a fingerprint of the request to a
    response dict. The default fingerprint extracts the verb name from
    the user message (which prompt templates include verbatim). Tests
    that need richer dispatch can pass a ``key_fn`` that receives the
    full request and returns a hashable key.

    Example
    -------
    >>> client = MockLLMClient(script={
    ...     ("SPLIT",): {"connective": "conjunctive", "children": ["a", "b"]},
    ...     ("FORM_HYPOTHESIS",): {"statement": "x = 4"},
    ... })
    """

    script: dict[ScriptKey, dict[str, Any]] = field(default_factory=dict)
    key_fn: Callable[[str, str, Mapping[str, Any]], ScriptKey] | None = None
    calls: list[tuple[str, str, Mapping[str, Any]]] = field(default_factory=list)

    def complete_json(
        self,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> dict[str, Any]:
        self.calls.append((system, user, dict(schema)))
        key = (
            self.key_fn(system, user, schema)
            if self.key_fn is not None
            else self._default_key(user)
        )
        if key not in self.script:
            raise LLMResponseError(
                f"MockLLMClient has no scripted response for key {key!r}. "
                f"Configured keys: {sorted(self.script)}."
            )
        return dict(self.script[key])

    @staticmethod
    def _default_key(user_message: str) -> ScriptKey:
        m = re.search(r"Verb:\s*([A-Z_]+)", user_message)
        if m is None:
            return (user_message[:40],)
        return (m.group(1),)


# -----------------------------------------------------------------------------
# Echo client (zero-config smoke tests)
# -----------------------------------------------------------------------------


@dataclass
class EchoLLMClient:
    """A minimal client that returns the first valid choice for each field.

    For each property in the schema, returns:
    * the first enum value if the property has an ``enum`` list;
    * an empty string for ``"type": "string"``;
    * ``None`` for ``"type": ["string", "null"]``;
    * the first item if the property is an ``array`` of strings;
    * 0 for numeric types.

    Useful for sanity-checking that the prompt assembly and parse
    pipeline run end to end before any real model is wired in.
    """

    def complete_json(
        self,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> dict[str, Any]:
        properties = schema.get("properties", {})
        result: dict[str, Any] = {}
        for key, spec in properties.items():
            result[key] = _echo_value(spec)
        return result


def _echo_value(spec: Mapping[str, Any]) -> Any:
    if "enum" in spec and spec["enum"]:
        return spec["enum"][0]
    t = spec.get("type")
    if isinstance(t, list):
        if "null" in t:
            return None
        t = t[0]
    if t == "string":
        return ""
    if t == "array":
        item = spec.get("items", {})
        return [_echo_value(item)] if "enum" in item else []
    if t in {"integer", "number"}:
        return 0
    if t == "boolean":
        return False
    return None


# -----------------------------------------------------------------------------
# OpenAI-compatible HTTP client (vLLM, Ollama, llama.cpp server, etc.)
# -----------------------------------------------------------------------------


@dataclass
class OpenAICompatibleLLMClient:
    """HTTP client for any server with the OpenAI chat-completions API.

    The most common backends, and the URL each one uses by default:

    * **Ollama** (the easiest setup): runs on the GPU box with
      ``ollama serve`` and ``ollama pull qwen2.5-math:1.5b``;
      base URL ``http://gpu-host:11434/v1``, no API key required.
    * **vLLM** (the highest throughput): runs on the GPU box with
      ``python -m vllm.entrypoints.openai.api_server --model
      Qwen/Qwen2.5-Math-1.5B-Instruct --host 0.0.0.0 --port 8000``;
      base URL ``http://gpu-host:8000/v1``, no API key required by
      default.
    * **llama.cpp's server**: similar, base URL
      ``http://gpu-host:8080/v1``.

    All three accept the same request shape, so the client works
    against any of them by setting ``base_url`` and ``model`` once.

    For JSON-mode the client sets ``response_format={"type":
    "json_object"}`` (the widely-supported OpenAI variant). If the
    backend supports the stricter ``json_schema`` format, the schema
    is also surfaced via the system message so the model has the
    field list and constraints to follow. A best-effort retry
    (default: one) re-asks the model when the response is not parseable
    JSON or fails minimal schema validation.
    """

    base_url: str
    model: str
    api_key: str | None = None
    timeout_s: float = 60.0
    max_retries: int = 1
    # Rate-limit/server-error backoff budget, separate from the parse-retry
    # budget above: a 429 needs patience, not a corrective prompt. Verified
    # failure mode (2026-07-02): 8 concurrent workers against api.openai.com
    # turned sustained 429s into silent controller fallbacks and ruined a
    # 500-episode collection run.
    max_backoff_retries: int = 6
    extra_headers: dict[str, str] = field(default_factory=dict)
    guided_json: bool = False
    # FLOP counters: mirroring TransformersLLMClient attribute names so that
    # outer_loop._flop_counters can read them uniformly across backends.
    generate_calls: int = field(default=0, compare=False, repr=False)
    generated_tokens: int = field(default=0, compare=False, repr=False)
    # Billing counter: prompt-side tokens as reported by the server's usage
    # payload. Not part of the FLOP-counter contract; used for cost accounting.
    prompt_tokens: int = field(default=0, compare=False, repr=False)
    # Lock guards counter mutations: complete_json is called concurrently from
    # many collection-thread workers in collect_iteration_concurrent.
    _counter_lock: threading.Lock = field(
        default_factory=threading.Lock, compare=False, repr=False
    )

    def complete_json(
        self,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> dict[str, Any]:
        import httpx  # lazy import so the package works without httpx for tests

        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Note: we deliberately do **not** dump the JSON Schema dict into
        # the system message. Small instruct-tuned LLMs reproduce JSON
        # Schema structure verbatim (returning ``{"properties": {...},
        # "required": [...], "type": "object"}`` instead of values
        # matching the schema). The user-message body is responsible for
        # describing the expected JSON shape in natural language, with a
        # one-shot example. We use ``response_format`` as a soft hint
        # only; minimal validation runs after parsing.
        system_with_jsonhint = (
            f"{system}\n\nReply with a single JSON object only, no prose."
        )

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_with_jsonhint},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        if self.guided_json:
            body["guided_json"] = dict(schema)

        last_err: Exception | None = None
        parse_attempts = 0
        backoff_attempts = 0
        while True:
            if seed is not None:
                # Per-attempt bump: with a fixed seed a malformed response
                # would be resampled identically on retry; offsetting by the
                # parse-attempt index keeps the call reproducible while letting
                # the retry actually draw a fresh continuation. Ollama and vLLM
                # both honor request-level "seed" on /v1/chat/completions.
                body["seed"] = seed + parse_attempts
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    response = client.post(url, headers=headers, json=body)
                response.raise_for_status()
                payload = response.json()
                content = payload["choices"][0]["message"]["content"]
                parsed = _parse_json_lenient(content)
                _validate_against_schema(parsed, schema)
                usage = payload.get("usage") or {}
                completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                new_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                with self._counter_lock:
                    self.generate_calls += 1
                    self.generated_tokens += completion_tokens
                    self.prompt_tokens += new_prompt_tokens
                return parsed
            except httpx.HTTPStatusError as e:
                # Rate limits and transient server errors get patience, not a
                # corrective prompt: honor Retry-After when present, else
                # exponential backoff with jitter.
                status = e.response.status_code
                if (status == 429 or status >= 500) \
                        and backoff_attempts < self.max_backoff_retries:
                    retry_after = e.response.headers.get("retry-after")
                    try:
                        delay = float(retry_after)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        delay = min(2.0 ** backoff_attempts, 30.0)
                    time.sleep(delay + random.uniform(0.0, 0.5))
                    backoff_attempts += 1
                    continue
                last_err = e
                break
            except (httpx.HTTPError, KeyError, ValueError, LLMResponseError) as e:
                last_err = e
                if parse_attempts < self.max_retries:
                    parse_attempts += 1
                    body["messages"].append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response could not be parsed. "
                                "Return a single valid JSON object matching the schema. "
                                "Return JSON only."
                            ),
                        }
                    )
                    continue
                break

        raise LLMResponseError(
            f"OpenAICompatibleLLMClient: no usable response after "
            f"{parse_attempts + 1} parse attempt(s) and {backoff_attempts} "
            f"backoff retr(ies) at {url} (model={self.model})",
            cause=last_err,
        )


def _schema_hint_for_prompt(schema: Mapping[str, Any]) -> str:
    """Compact human-readable rendering of the schema for the system prompt."""
    return json.dumps(schema, indent=2, sort_keys=True)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_lenient(text: str) -> dict[str, Any]:
    """Parse a JSON object out of a response that may include extra prose."""
    text = text.strip()
    if not text:
        raise LLMResponseError("Empty response.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(text)
        if match is None:
            raise LLMResponseError(f"No JSON object found in response: {text[:120]!r}")
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            raise LLMResponseError(
                f"Failed to parse JSON: {e.msg} (input: {text[:120]!r})"
            ) from e


def _validate_against_schema(value: Any, schema: Mapping[str, Any]) -> None:
    """Minimal schema validation: required fields, enums, and declared types.

    Type enforcement carries one pragmatic coercion: a JSON number in a
    ``"type": "string"`` field is coerced to its canonical string form
    (``42`` -> ``"42"``, ``42.0`` -> ``"42"``), because small JSON-mode
    models routinely emit numeric answers as numbers. All other type
    mismatches raise ``LLMResponseError`` so the caller's retry loop and
    the MDP's masked-action recovery engage, instead of a malformed value
    flowing into state assembly (where a string in an array slot iterates
    per character, or a number reaches code expecting ``str``).

    ``None`` in an optional field is treated as the field being absent.
    ``None`` in a required field that does not declare ``"null"`` among
    its types is an error. Coercions are applied to ``value`` in place.
    """
    if not isinstance(value, dict):
        raise LLMResponseError(f"Expected JSON object, got {type(value).__name__}.")
    required = set(schema.get("required", []))
    for req in required:
        if req not in value:
            raise LLMResponseError(f"Missing required field {req!r}.")
    for prop, spec in schema.get("properties", {}).items():
        if prop not in value:
            continue
        v = value[prop]
        if v is None:
            if prop in required and not _type_allows_null(spec):
                raise LLMResponseError(f"Required field {prop!r} is null.")
            continue
        if "enum" in spec and v not in spec["enum"]:
            raise LLMResponseError(
                f"Field {prop!r} value {v!r} not in allowed enum {spec['enum']!r}."
            )
        value[prop] = _check_type(prop, v, spec)


def _type_allows_null(spec: Mapping[str, Any]) -> bool:
    t = spec.get("type")
    return isinstance(t, list) and "null" in t


def _number_to_string(v: int | float) -> str:
    """Canonical string form of a JSON number: integral floats drop the .0."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _check_type(prop: str, v: Any, spec: Mapping[str, Any]) -> Any:
    """Enforce ``spec``'s declared type on ``v``, returning the (possibly
    coerced) value. Raises ``LLMResponseError`` on mismatch."""
    t = spec.get("type")
    if t is None:
        return v
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        if not non_null:
            return v
        t = non_null[0]

    if t == "string":
        if isinstance(v, bool):
            raise LLMResponseError(f"Field {prop!r} expected string, got bool.")
        if isinstance(v, (int, float)):
            return _number_to_string(v)
        if not isinstance(v, str):
            raise LLMResponseError(
                f"Field {prop!r} expected string, got {type(v).__name__}."
            )
        return v
    if t == "array":
        if not isinstance(v, list):
            raise LLMResponseError(
                f"Field {prop!r} expected array, got {type(v).__name__}."
            )
        max_items = spec.get("maxItems")
        if max_items is not None and len(v) > max_items:
            raise LLMResponseError(
                f"Field {prop!r} has {len(v)} items, maxItems is {max_items}."
            )
        min_items = spec.get("minItems")
        if min_items is not None and len(v) < min_items:
            raise LLMResponseError(
                f"Field {prop!r} has {len(v)} items, minItems is {min_items}."
            )
        items_spec = spec.get("items")
        if items_spec is not None:
            for i, item in enumerate(v):
                v[i] = _check_type(f"{prop}[{i}]", item, items_spec)
        return v
    if t == "object":
        if not isinstance(v, dict):
            raise LLMResponseError(
                f"Field {prop!r} expected object, got {type(v).__name__}."
            )
        return v
    if t in ("number", "integer"):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise LLMResponseError(
                f"Field {prop!r} expected {t}, got {type(v).__name__}."
            )
        return v
    if t == "boolean":
        if not isinstance(v, bool):
            raise LLMResponseError(
                f"Field {prop!r} expected boolean, got {type(v).__name__}."
            )
        return v
    return v


__all__ = [
    "EchoLLMClient",
    "LLMClient",
    "LLMResponseError",
    "MockLLMClient",
    "OpenAICompatibleLLMClient",
]
