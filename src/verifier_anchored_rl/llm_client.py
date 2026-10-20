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

from copy import deepcopy
import json
import math
import random
import re
import threading
import time
import uuid
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
        schema_validation_mode: str = (
            "json-schema-numeric-string-coercion-v1"
        ),
    ) -> dict[str, Any]:
        ...


class LLMResponseError(RuntimeError):
    """Raised when the backend cannot produce a schema-conformant response.

    The MDP transition layer catches this and treats it as a masked
    action. Callers that want to surface the underlying server error
    can read ``self.cause``.
    """

    def __init__(self, message: str, *, cause: Exception | None = None,
                 raw_response: str | None = None):
        super().__init__(message)
        self.cause = cause
        # Kept out of the exception text so logs opt into potentially long
        # provider output explicitly.
        self.raw_response = raw_response


@dataclass(frozen=True, slots=True)
class OpenAICompatibleAttempt:
    """Secret-free trace event for one OpenAI-compatible HTTP POST.

    ``request_body_json`` contains only the JSON request body. HTTP headers,
    bearer tokens, and the endpoint URL are deliberately absent. The JSON
    snapshots keep the event immutable even though the client mutates its
    retry transcript after a failed parse.
    """

    call_id: str
    attempt_index: int
    parse_retry_index: int
    backoff_retry_index: int
    effective_seed: int | None
    request_body_json: str
    raw_response_body: str | None
    raw_content: str | None
    raw_usage_json: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    outcome: str
    disposition: str
    http_status: int | None
    error_type: str | None
    error_message: str | None


@dataclass(frozen=True, slots=True)
class OpenAICompatibleResponseProjection:
    """Deterministic fields projected from one provider response object."""

    raw_content: str | None
    raw_usage_json: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    provider_error: Exception | None


def _json_snapshot(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


OPENAI_JSON_RETRY_INSTRUCTION = (
    "Your previous response could not be parsed. Return a single valid "
    "JSON object matching the schema. Return JSON only."
)

SCHEMA_VALIDATION_LEGACY_COERCE_V1 = (
    "json-schema-numeric-string-coercion-v1"
)
SCHEMA_VALIDATION_STRICT_V1 = "json-schema-strict-types-v1"
SCHEMA_VALIDATION_MODES = frozenset({
    SCHEMA_VALIDATION_LEGACY_COERCE_V1,
    SCHEMA_VALIDATION_STRICT_V1,
})


def _require_schema_validation_mode(mode: str) -> None:
    if mode not in SCHEMA_VALIDATION_MODES:
        raise ValueError(f"unsupported schema validation mode {mode!r}")


def build_openai_compatible_json_body(
    *,
    model: str,
    system: str,
    user: str,
    schema: Mapping[str, Any],
    temperature: float,
    max_tokens: int,
    seed: int | None,
    guided_json: bool,
) -> dict[str, Any]:
    """Build the exact secret-free JSON body used for the first POST."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"{system}\n\nReply with a single JSON object only, "
                    "no prose."
                ),
            },
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if guided_json:
        body["guided_json"] = dict(schema)
    if seed is not None:
        body["seed"] = seed
    return body


def append_openai_compatible_parse_retry(
    body: dict[str, Any], raw_content: str | None
) -> None:
    """Apply the exact deterministic transcript update for a parse retry."""
    messages = body.get("messages")
    if type(messages) is not list:
        raise TypeError("OpenAI-compatible body messages must be a list")
    if raw_content:
        messages.append({"role": "assistant", "content": raw_content})
    messages.append({
        "role": "user",
        "content": OPENAI_JSON_RETRY_INSTRUCTION,
    })


def _response_text(response: Any) -> str | None:
    try:
        value = response.text
    except Exception:
        return None
    return value if isinstance(value, str) else None


def _token_count(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise ValueError(f"usage.{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"usage.{field_name} must be nonnegative")
    return value


def project_openai_compatible_response(
    payload: object,
) -> OpenAICompatibleResponseProjection:
    """Project content and usage exactly as the live client does."""
    if type(payload) is not dict:
        raise ValueError("provider response must be an object")
    content: str | None = None
    provider_error: Exception | None = None
    try:
        candidate = payload["choices"][0]["message"]["content"]
        if not isinstance(candidate, str):
            raise ValueError("provider content must be text")
        content = candidate
    except (KeyError, IndexError, TypeError, ValueError) as error:
        provider_error = error

    raw_usage_json: str | None = None
    prompt_count: int | None = None
    completion_count: int | None = None
    total_count: int | None = None
    try:
        raw_usage = payload.get("usage")
        if raw_usage is not None:
            raw_usage_json = _json_snapshot(raw_usage)
            if type(raw_usage) is not dict:
                raise ValueError("provider usage must be an object")
            prompt_count = _token_count(
                raw_usage.get("prompt_tokens"), "prompt_tokens"
            )
            completion_count = _token_count(
                raw_usage.get("completion_tokens"),
                "completion_tokens",
            )
            total_count = _token_count(
                raw_usage.get("total_tokens"), "total_tokens"
            )
    except (TypeError, ValueError, OverflowError) as error:
        if provider_error is None:
            provider_error = error
    return OpenAICompatibleResponseProjection(
        raw_content=content,
        raw_usage_json=raw_usage_json,
        prompt_tokens=prompt_count,
        completion_tokens=completion_count,
        total_tokens=total_count,
        provider_error=provider_error,
    )


def _attempt_error_message(error: Exception, status: int | None) -> str:
    if status is not None:
        return f"HTTP status {status}"
    if isinstance(error, LLMResponseError):
        return str(error)
    return type(error).__name__


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
        schema_validation_mode: str = SCHEMA_VALIDATION_LEGACY_COERCE_V1,
    ) -> dict[str, Any]:
        _require_schema_validation_mode(schema_validation_mode)
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
        result = deepcopy(self.script[key])
        raw_response = json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            _validate_against_schema(
                result,
                schema,
                schema_validation_mode=schema_validation_mode,
            )
        except LLMResponseError as exc:
            raise LLMResponseError(
                str(exc),
                raw_response=raw_response,
            ) from exc
        return result

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
        schema_validation_mode: str = SCHEMA_VALIDATION_LEGACY_COERCE_V1,
    ) -> dict[str, Any]:
        _require_schema_validation_mode(schema_validation_mode)
        properties = schema.get("properties", {})
        result: dict[str, Any] = {}
        for key, spec in properties.items():
            result[key] = _echo_value(spec)
        if schema_validation_mode == SCHEMA_VALIDATION_STRICT_V1:
            _validate_against_schema(
                result,
                schema,
                schema_validation_mode=schema_validation_mode,
            )
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
    # repr=False: the bearer token must never appear in reprs, tracebacks,
    # or logs that format this object.
    api_key: str | None = field(default=None, repr=False)
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
    attempt_observer: Callable[[OpenAICompatibleAttempt], None] | None = field(
        default=None, compare=False, repr=False
    )
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
        schema_validation_mode: str = SCHEMA_VALIDATION_LEGACY_COERCE_V1,
    ) -> dict[str, Any]:
        _require_schema_validation_mode(schema_validation_mode)
        import httpx  # lazy import so the package works without httpx for tests

        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # The schema stays out of the system prompt because small instruct
        # models often reproduce schema structure instead of filling values.
        body = build_openai_compatible_json_body(
            model=self.model,
            system=system,
            user=user,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            guided_json=self.guided_json,
        )

        call_id = uuid.uuid4().hex
        last_err: Exception | None = None
        parse_attempts = 0
        backoff_attempts = 0
        attempt_index = 0
        raw_responses: list[str] = []

        def emit_attempt(
            *,
            current_attempt: int,
            effective_seed: int | None,
            request_body_json: str,
            raw_response_body: str | None,
            raw_content: str | None,
            raw_usage_json: str | None,
            prompt_count: int | None,
            completion_count: int | None,
            total_count: int | None,
            outcome: str,
            disposition: str,
            http_status: int | None,
            error: Exception | None,
        ) -> None:
            observer = self.attempt_observer
            if observer is None:
                return
            observer(OpenAICompatibleAttempt(
                call_id=call_id,
                attempt_index=current_attempt,
                parse_retry_index=parse_attempts,
                backoff_retry_index=backoff_attempts,
                effective_seed=effective_seed,
                request_body_json=request_body_json,
                raw_response_body=raw_response_body,
                raw_content=raw_content,
                raw_usage_json=raw_usage_json,
                prompt_tokens=prompt_count,
                completion_tokens=completion_count,
                total_tokens=total_count,
                outcome=outcome,
                disposition=disposition,
                http_status=http_status,
                error_type=None if error is None else type(error).__name__,
                error_message=(
                    None
                    if error is None
                    else _attempt_error_message(error, http_status)
                ),
            ))

        while True:
            content: str | None = None
            effective_seed = None if seed is None else seed + parse_attempts
            if effective_seed is not None:
                # Backoff retries retain the same seed. Parse retries bump it so
                # a malformed deterministic continuation can be resampled.
                body["seed"] = effective_seed
            request_body_json = _json_snapshot(body)
            current_attempt = attempt_index
            attempt_index += 1
            response_body: str | None = None
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    response = client.post(url, headers=headers, json=body)
                response_body = _response_text(response)
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                response_body = _response_text(error.response)
                retry = (
                    (status == 429 or status >= 500)
                    and backoff_attempts < self.max_backoff_retries
                )
                emit_attempt(
                    current_attempt=current_attempt,
                    effective_seed=effective_seed,
                    request_body_json=request_body_json,
                    raw_response_body=response_body,
                    raw_content=None,
                    raw_usage_json=None,
                    prompt_count=None,
                    completion_count=None,
                    total_count=None,
                    outcome="http_status_error",
                    disposition="backoff_retry" if retry else "raise",
                    http_status=status,
                    error=error,
                )
                if retry:
                    retry_after = error.response.headers.get("retry-after")
                    try:
                        delay = float(retry_after)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        delay = min(2.0 ** backoff_attempts, 30.0)
                    time.sleep(delay + random.uniform(0.0, 0.5))
                    backoff_attempts += 1
                    continue
                last_err = error
                break
            except httpx.HTTPError as error:
                retry = parse_attempts < self.max_retries
                emit_attempt(
                    current_attempt=current_attempt,
                    effective_seed=effective_seed,
                    request_body_json=request_body_json,
                    raw_response_body=response_body,
                    raw_content=None,
                    raw_usage_json=None,
                    prompt_count=None,
                    completion_count=None,
                    total_count=None,
                    outcome="transport_error",
                    disposition="parse_retry" if retry else "raise",
                    http_status=None,
                    error=error,
                )
                last_err = error
                if retry:
                    parse_attempts += 1
                    append_openai_compatible_parse_retry(body, None)
                    continue
                break

            status_value = getattr(response, "status_code", None)
            http_status = (
                status_value if type(status_value) is int else None
            )
            with self._counter_lock:
                self.generate_calls += 1

            raw_usage_json: str | None = None
            prompt_count: int | None = None
            completion_count: int | None = None
            total_count: int | None = None
            try:
                payload = response.json()
                if type(payload) is not dict:
                    raise ValueError("provider response must be an object")
            except (TypeError, ValueError) as error:
                retry = parse_attempts < self.max_retries
                emit_attempt(
                    current_attempt=current_attempt,
                    effective_seed=effective_seed,
                    request_body_json=request_body_json,
                    raw_response_body=response_body,
                    raw_content=None,
                    raw_usage_json=None,
                    prompt_count=None,
                    completion_count=None,
                    total_count=None,
                    outcome="provider_response_error",
                    disposition="parse_retry" if retry else "raise",
                    http_status=http_status,
                    error=error,
                )
                last_err = error
                if retry:
                    parse_attempts += 1
                    append_openai_compatible_parse_retry(body, None)
                    continue
                break

            projection = project_openai_compatible_response(payload)
            content = projection.raw_content
            raw_usage_json = projection.raw_usage_json
            prompt_count = projection.prompt_tokens
            completion_count = projection.completion_tokens
            total_count = projection.total_tokens
            provider_error = projection.provider_error

            with self._counter_lock:
                self.generated_tokens += completion_count or 0
                self.prompt_tokens += prompt_count or 0
            if content is not None:
                raw_responses.append(content)

            if provider_error is not None:
                retry = parse_attempts < self.max_retries
                emit_attempt(
                    current_attempt=current_attempt,
                    effective_seed=effective_seed,
                    request_body_json=request_body_json,
                    raw_response_body=response_body,
                    raw_content=content,
                    raw_usage_json=raw_usage_json,
                    prompt_count=prompt_count,
                    completion_count=completion_count,
                    total_count=total_count,
                    outcome="provider_response_error",
                    disposition="parse_retry" if retry else "raise",
                    http_status=http_status,
                    error=provider_error,
                )
                last_err = provider_error
                if retry:
                    parse_attempts += 1
                    append_openai_compatible_parse_retry(body, content)
                    continue
                break

            try:
                parsed = _parse_json_lenient(
                    content,
                    schema_validation_mode=schema_validation_mode,
                )
            except LLMResponseError as error:
                retry = parse_attempts < self.max_retries
                emit_attempt(
                    current_attempt=current_attempt,
                    effective_seed=effective_seed,
                    request_body_json=request_body_json,
                    raw_response_body=response_body,
                    raw_content=content,
                    raw_usage_json=raw_usage_json,
                    prompt_count=prompt_count,
                    completion_count=completion_count,
                    total_count=total_count,
                    outcome="parse_error",
                    disposition="parse_retry" if retry else "raise",
                    http_status=http_status,
                    error=error,
                )
                last_err = error
                if retry:
                    parse_attempts += 1
                    append_openai_compatible_parse_retry(body, content)
                    continue
                break

            try:
                _validate_against_schema(
                    parsed,
                    schema,
                    schema_validation_mode=schema_validation_mode,
                )
            except LLMResponseError as error:
                retry = parse_attempts < self.max_retries
                emit_attempt(
                    current_attempt=current_attempt,
                    effective_seed=effective_seed,
                    request_body_json=request_body_json,
                    raw_response_body=response_body,
                    raw_content=content,
                    raw_usage_json=raw_usage_json,
                    prompt_count=prompt_count,
                    completion_count=completion_count,
                    total_count=total_count,
                    outcome="schema_error",
                    disposition="parse_retry" if retry else "raise",
                    http_status=http_status,
                    error=error,
                )
                last_err = error
                if retry:
                    parse_attempts += 1
                    append_openai_compatible_parse_retry(body, content)
                    continue
                break

            emit_attempt(
                current_attempt=current_attempt,
                effective_seed=effective_seed,
                request_body_json=request_body_json,
                raw_response_body=response_body,
                raw_content=content,
                raw_usage_json=raw_usage_json,
                prompt_count=prompt_count,
                completion_count=completion_count,
                total_count=total_count,
                outcome="accepted",
                disposition="return",
                http_status=http_status,
                error=None,
            )
            return parsed

        raise LLMResponseError(
            f"OpenAICompatibleLLMClient: no usable response after "
            f"{parse_attempts + 1} parse attempt(s) and {backoff_attempts} "
            f"backoff retr(ies) at {url} (model={self.model})",
            raw_response=raw_responses[-1] if raw_responses else None,
            cause=last_err,
        )


def _schema_hint_for_prompt(schema: Mapping[str, Any]) -> str:
    """Compact human-readable rendering of the schema for the system prompt."""
    return json.dumps(schema, indent=2, sort_keys=True)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"non-JSON numeric constant {value!r}")


def _parse_json_lenient(
    text: str,
    *,
    schema_validation_mode: str = SCHEMA_VALIDATION_LEGACY_COERCE_V1,
) -> dict[str, Any]:
    """Parse a JSON object out of a response that may include extra prose."""
    _require_schema_validation_mode(schema_validation_mode)
    load_options: dict[str, object] = {}
    if schema_validation_mode == SCHEMA_VALIDATION_STRICT_V1:
        load_options["parse_constant"] = _reject_non_json_constant
    text = text.strip()
    if not text:
        raise LLMResponseError("Empty response.")
    try:
        return json.loads(text, **load_options)
    except (json.JSONDecodeError, ValueError):
        match = _JSON_OBJECT_RE.search(text)
        if match is None:
            raise LLMResponseError(f"No JSON object found in response: {text[:120]!r}")
        try:
            return json.loads(match.group(), **load_options)
        except (json.JSONDecodeError, ValueError) as e:
            reason = e.msg if isinstance(e, json.JSONDecodeError) else str(e)
            raise LLMResponseError(
                f"Failed to parse JSON: {reason} (input: {text[:120]!r})"
            ) from e


def _validate_against_schema(
    value: Any,
    schema: Mapping[str, Any],
    *,
    schema_validation_mode: str = SCHEMA_VALIDATION_LEGACY_COERCE_V1,
) -> None:
    """Validate the supported schema subset under the selected mode.

    Legacy type enforcement carries one pragmatic coercion: a JSON number in a
    ``"type": "string"`` field is coerced to its canonical string form
    (``42`` -> ``"42"``, ``42.0`` -> ``"42"``), because small JSON-mode
    models routinely emit numeric answers as numbers. Strict validation uses
    exact JSON types, enforces the supported object and array constraints, and
    never mutates the parsed value. Type mismatches raise ``LLMResponseError``
    so the caller's retry loop and masked-action recovery can engage.

    In legacy mode, ``None`` in an optional field is treated as absent and
    coercions are applied to ``value`` in place. Strict mode follows the
    declared nullability exactly.
    """
    _require_schema_validation_mode(schema_validation_mode)
    if schema_validation_mode == SCHEMA_VALIDATION_STRICT_V1:
        _validate_strict_schema_value("$", value, schema)
        return
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
        value[prop] = _check_type(
            prop,
            v,
            spec,
            schema_validation_mode=schema_validation_mode,
        )


def _strict_type_matches(value: Any, declared_type: str) -> bool:
    if declared_type == "null":
        return value is None
    if declared_type == "object":
        return type(value) is dict
    if declared_type == "array":
        return type(value) is list
    if declared_type == "string":
        return type(value) is str
    if declared_type == "integer":
        return type(value) is int
    if declared_type == "number":
        return type(value) is int or (
            type(value) is float and math.isfinite(value)
        )
    if declared_type == "boolean":
        return type(value) is bool
    return False


def _validate_strict_schema_value(
    path: str,
    value: Any,
    schema: Mapping[str, Any],
) -> None:
    declared = schema.get("type")
    if declared is None:
        declared_types: tuple[str, ...] = ()
    elif isinstance(declared, str):
        declared_types = (declared,)
    elif type(declared) is list and all(
        isinstance(item, str) for item in declared
    ):
        declared_types = tuple(declared)
    else:
        raise LLMResponseError(f"Field {path!r} has an invalid schema type.")

    matched_type: str | None = None
    if declared_types:
        matched_type = next(
            (
                item
                for item in declared_types
                if _strict_type_matches(value, item)
            ),
            None,
        )
        if matched_type is None:
            expected = " or ".join(declared_types)
            raise LLMResponseError(
                f"Field {path!r} expected {expected}, "
                f"got {type(value).__name__}."
            )

    enum = schema.get("enum")
    if enum is not None and not any(
        type(value) is type(candidate) and value == candidate
        for candidate in enum
    ):
        raise LLMResponseError(
            f"Field {path!r} value {value!r} not in allowed enum {enum!r}."
        )

    if matched_type == "object" or (
        matched_type is None and "properties" in schema
    ):
        if type(value) is not dict:
            raise LLMResponseError(
                f"Field {path!r} expected object, got {type(value).__name__}."
            )
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        missing = required - set(value)
        if missing:
            field = sorted(missing)[0]
            raise LLMResponseError(f"Missing required field {field!r} at {path}.")
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                field = sorted(extra)[0]
                raise LLMResponseError(
                    f"Unexpected field {field!r} at {path}."
                )
        for prop, prop_schema in properties.items():
            if prop in value:
                _validate_strict_schema_value(
                    f"{path}.{prop}", value[prop], prop_schema
                )
        return

    if matched_type == "array":
        min_items = schema.get("minItems")
        if min_items is not None and len(value) < min_items:
            raise LLMResponseError(
                f"Field {path!r} has {len(value)} items, "
                f"minItems is {min_items}."
            )
        max_items = schema.get("maxItems")
        if max_items is not None and len(value) > max_items:
            raise LLMResponseError(
                f"Field {path!r} has {len(value)} items, "
                f"maxItems is {max_items}."
            )
        items_schema = schema.get("items")
        if items_schema is not None:
            for index, item in enumerate(value):
                _validate_strict_schema_value(
                    f"{path}[{index}]", item, items_schema
                )


def decode_openai_compatible_json_content(
    content: str,
    schema: Mapping[str, Any],
    *,
    schema_validation_mode: str = SCHEMA_VALIDATION_LEGACY_COERCE_V1,
) -> dict[str, Any]:
    """Parse and validate raw content exactly as ``complete_json`` does."""
    parsed = _parse_json_lenient(
        content,
        schema_validation_mode=schema_validation_mode,
    )
    _validate_against_schema(
        parsed,
        schema,
        schema_validation_mode=schema_validation_mode,
    )
    return parsed


def classify_openai_compatible_json_content(
    content: str,
    schema: Mapping[str, Any],
    *,
    schema_validation_mode: str = SCHEMA_VALIDATION_LEGACY_COERCE_V1,
) -> tuple[str, dict[str, Any] | None]:
    """Return the live content outcome and decoded value without retrying."""
    try:
        parsed = _parse_json_lenient(
            content,
            schema_validation_mode=schema_validation_mode,
        )
    except LLMResponseError:
        return "parse_error", None
    try:
        _validate_against_schema(
            parsed,
            schema,
            schema_validation_mode=schema_validation_mode,
        )
    except LLMResponseError:
        return "schema_error", None
    return "accepted", parsed


def _type_allows_null(spec: Mapping[str, Any]) -> bool:
    t = spec.get("type")
    return isinstance(t, list) and "null" in t


def _number_to_string(v: int | float) -> str:
    """Canonical string form of a JSON number: integral floats drop the .0."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _check_type(
    prop: str,
    v: Any,
    spec: Mapping[str, Any],
    *,
    schema_validation_mode: str = SCHEMA_VALIDATION_LEGACY_COERCE_V1,
) -> Any:
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
            if schema_validation_mode == SCHEMA_VALIDATION_STRICT_V1:
                raise LLMResponseError(
                    f"Field {prop!r} expected string, got {type(v).__name__}."
                )
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
                v[i] = _check_type(
                    f"{prop}[{i}]",
                    item,
                    items_spec,
                    schema_validation_mode=schema_validation_mode,
                )
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
    "OPENAI_JSON_RETRY_INSTRUCTION",
    "SCHEMA_VALIDATION_LEGACY_COERCE_V1",
    "SCHEMA_VALIDATION_MODES",
    "SCHEMA_VALIDATION_STRICT_V1",
    "EchoLLMClient",
    "LLMClient",
    "LLMResponseError",
    "MockLLMClient",
    "OpenAICompatibleAttempt",
    "OpenAICompatibleLLMClient",
    "OpenAICompatibleResponseProjection",
    "append_openai_compatible_parse_retry",
    "build_openai_compatible_json_body",
    "classify_openai_compatible_json_content",
    "decode_openai_compatible_json_content",
    "project_openai_compatible_response",
]
