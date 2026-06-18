"""In-process HuggingFace Transformers backends for the LLM and embedding
protocols.

Motivation (see experiments/results/expert_iteration_rounds.md, infra
post-mortem of 2026-06-12): the pilot ran against an Ollama HTTP server,
which (a) stopped serving embeddings from generative models after a
version update, (b) cannot expose hidden states at all, which the
pre-registered study's adapter architecture requires, (c) hides FLOP
truth behind an API, and (d) owns the sampling RNG, which made
matched-seed collections irreproducible. Running the frozen model
in-process removes all four problems: hidden states are directly
available, forwards are countable, and sampling is seeded by us.

Two classes, implementing the package's existing protocols so nothing
else changes:

* ``TransformersLLMClient`` implements ``LLMClient.complete_json``:
  chat-template prompt assembly, seeded sampling
  (``torch.manual_seed(seed + attempt)``, mirroring the HTTP client's
  per-attempt bump), greedy when ``temperature == 0``, and the same
  lenient-JSON-parse plus schema-validation plus retry loop as
  ``OpenAICompatibleLLMClient`` (the helpers are imported, not
  reimplemented).

* ``TransformersEmbeddingClient`` implements ``EmbeddingClient.embed``:
  a documented pooling rule over the model's last hidden layer
  (attention-masked mean by default, ``last_token`` optional),
  L2-normalized. Unlike the Ollama endpoint, the pooling rule here is
  explicit and ours. The embedder can share the generator's loaded
  weights (``share_model_with``), restoring the pilot's
  one-frozen-model-both-roles design point without a second copy in
  VRAM.

Heavy imports (``transformers``) happen lazily at load time, and both
classes accept pre-built ``model``/``tokenizer`` components, which is
how the unit tests exercise the logic with fakes and no downloads.

Install with ``pip install -e .[hf]`` (torch is required by the same
workflows that require the heads).
"""

from __future__ import annotations

from typing import Any, Mapping

from .llm_client import (
    LLMResponseError,
    _parse_json_lenient,
    _validate_against_schema,
)


def _default_device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_causal_lm(model_name: str, device: str | None, dtype: str | None):
    """Load tokenizer + causal LM once. Returns (model, tokenizer, device)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or _default_device()
    torch_dtype = (
        getattr(torch, dtype)
        if dtype
        else (torch.bfloat16 if device == "cuda" else torch.float32)
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()
    return model, tokenizer, device


class TransformersLLMClient:
    """LLMClient backend running a causal LM in-process.

    Parameters mirror the HTTP client where they overlap. The frozen
    model is loaded once at construction (or injected via ``model=`` and
    ``tokenizer=`` for tests and for sharing an already-loaded model).
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        model: Any = None,
        tokenizer: Any = None,
        device: str | None = None,
        dtype: str | None = None,
        max_retries: int = 1,
    ):
        if model is not None and tokenizer is not None:
            self.model, self.tokenizer = model, tokenizer
            self.device = device or str(getattr(model, "device", "cpu"))
        elif model_name is not None:
            self.model, self.tokenizer, self.device = _load_causal_lm(
                model_name, device, dtype
            )
        else:
            raise ValueError(
                "TransformersLLMClient needs model_name or (model, tokenizer)"
            )
        self.model_name = model_name or getattr(
            getattr(self.model, "config", None), "name_or_path", "injected"
        )
        self.max_retries = max_retries
        # Countable forward truth for the compute-accounting axis:
        # generated tokens are a faithful proxy for decoder forwards.
        self.generate_calls = 0
        self.generated_tokens = 0

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
        import torch

        messages = [
            {
                "role": "system",
                "content": f"{system}\n\nReply with a single JSON object only, no prose.",
            },
            {"role": "user", "content": user},
        ]

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            # transformers 5.x returns a BatchEncoding (input_ids +
            # attention_mask), not a bare tensor; pass both to generate so
            # the mask is correct under padding.
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
            prompt_len = encoded["input_ids"].shape[1]

            if seed is not None:
                # Per-attempt bump mirrors OpenAICompatibleLLMClient: a
                # fixed seed would resample a malformed response
                # identically on retry.
                torch.manual_seed(seed + attempt)

            do_sample = temperature > 0
            with torch.no_grad():
                output = self.model.generate(
                    **encoded,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = output[0][prompt_len:]
            self.generate_calls += 1
            self.generated_tokens += int(new_tokens.shape[0])
            content = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            try:
                parsed = _parse_json_lenient(content)
                _validate_against_schema(parsed, schema)
                return parsed
            except LLMResponseError as e:
                last_err = e
                if attempt < self.max_retries:
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
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

        raise LLMResponseError(
            f"TransformersLLMClient: no parseable response after "
            f"{self.max_retries + 1} attempts (model={self.model_name})",
            cause=last_err,
        )


class TransformersEmbeddingClient:
    """EmbeddingClient backend pooling a model's last hidden layer.

    The pooling rule is explicit (the Ollama endpoint never documented
    its own): attention-masked mean over the final hidden layer by
    default, or the final position's hidden state with
    ``pooling="last_token"``; L2-normalized either way. Output dimension
    is the model's hidden size.

    Construct from a model name, from injected components, or from an
    existing ``TransformersLLMClient`` via ``share_model_with`` so the
    generator and the encoder are the same frozen weights (the pilot's
    design point, one copy in VRAM).
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        model: Any = None,
        tokenizer: Any = None,
        device: str | None = None,
        dtype: str | None = None,
        pooling: str = "mean",
        normalize: bool = True,
        max_length: int = 8192,
    ):
        if pooling not in ("mean", "last_token"):
            raise ValueError(f"unknown pooling {pooling!r}")
        if model is not None and tokenizer is not None:
            self.model, self.tokenizer = model, tokenizer
        elif model_name is not None:
            self.model, self.tokenizer, _ = _load_causal_lm(model_name, device, dtype)
        else:
            raise ValueError(
                "TransformersEmbeddingClient needs model_name or (model, tokenizer)"
            )
        self.pooling = pooling
        self.normalize = normalize
        self.max_length = max_length

    @classmethod
    def share_model_with(
        cls, llm_client: TransformersLLMClient, *, pooling: str = "mean"
    ) -> "TransformersEmbeddingClient":
        return cls(
            model=llm_client.model, tokenizer=llm_client.tokenizer, pooling=pooling
        )

    def embed(self, text: str) -> list[float]:
        import torch

        batch = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        batch = {k: v.to(self.model.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.model(**batch, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0]  # (seq, hidden)
        mask = batch["attention_mask"][0].unsqueeze(-1)  # (seq, 1)

        if self.pooling == "mean":
            pooled = (hidden * mask).sum(dim=0) / mask.sum().clamp(min=1)
        else:  # last_token
            last_index = int(batch["attention_mask"][0].sum().item()) - 1
            pooled = hidden[last_index]

        pooled = pooled.float()
        if self.normalize:
            norm = pooled.norm()
            if float(norm) > 0:
                pooled = pooled / norm
        return pooled.cpu().tolist()


__all__ = [
    "TransformersEmbeddingClient",
    "TransformersLLMClient",
]
