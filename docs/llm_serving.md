# Serving an LLM for the verifier-anchored RL framework

The framework calls a frozen base LLM as a stochastic transition oracle
through `verifier_anchored_rl.llm_client.OpenAICompatibleLLMClient`.
This client speaks the OpenAI chat-completions API with JSON-mode
response formatting, which means any server that implements that API
will work: vLLM, Ollama, llama.cpp's server, LM Studio, plus the
commercial OpenAI/Anthropic endpoints if you ever want them.

This document covers the recommended local-network setup: run an
inference server on a GPU box, hit it from your development machine
over LAN.

## Architecture

```
+-------------------+         HTTP            +-------------------+
| development box   |  ---- /v1/chat/        |   GPU box (LAN)   |
| (CPU only)        |        completions ---->| Ollama or vLLM    |
| - LLMContextMDP   |                         | - Qwen2.5-Math-1.5B
| - MCTS + training |  <---- JSON response -- | - bf16/fp16        |
+-------------------+                         +-------------------+
```

The framework code runs on the dev box without a GPU. Only the LLM
server needs the GPU. The single configuration knob is the base URL
of the server, which is passed to `OpenAICompatibleLLMClient`.

## Option A: Ollama (easiest)

Ollama is the simplest setup. It pulls quantized GGUF models, runs
them through llama.cpp under the hood, and serves an OpenAI-compatible
API.

### On the GPU box

```bash
# Install (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Or macOS via Homebrew
brew install ollama

# Start the server (default port 11434, binds to localhost by default).
# Set OLLAMA_HOST to expose on all interfaces so the LAN dev box can reach it.
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# In a second terminal, pull the model you want to use.
ollama pull qwen2.5-math:1.5b
# Or other small math models:
#   ollama pull qwen2.5:1.5b-instruct
#   ollama pull qwen2.5:7b-instruct
#   ollama pull deepseek-math:7b
```

### On the dev box

```python
from verifier_anchored_rl import OpenAICompatibleLLMClient, LLMContextMDP

client = OpenAICompatibleLLMClient(
    base_url="http://gpu-box.local:11434/v1",  # or http://192.168.x.y:11434/v1
    model="qwen2.5-math:1.5b",
)

mdp = LLMContextMDP(
    problem_statement="...",
    expected_answer="...",
    llm_client=client,
)
```

Ollama auto-pulls the model on first request if it is not already cached.

### Notes

- Ollama defaults to 4-bit quantization. For better quality, pull a
  higher-precision tag (e.g. `qwen2.5-math:1.5b-fp16`) if the GPU has
  the memory.
- Ollama's throughput is good for development (10–50 tokens/sec on
  consumer GPUs) but not the highest available. For training runs,
  vLLM is faster.
- JSON-mode is supported via the `response_format` parameter; the
  framework client already sets this.

## Option B: vLLM (highest throughput)

vLLM uses PagedAttention and continuous batching for substantially
higher throughput than llama.cpp/Ollama, especially with concurrent
requests. Recommended when you start running training rollouts.

### On the GPU box

```bash
pip install vllm

# Serve the model. --host 0.0.0.0 exposes on LAN.
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --guided-decoding-backend outlines
```

The `--guided-decoding-backend outlines` flag enables grammar-constrained
JSON generation, which is the most robust way to get schema-valid output.

### On the dev box

```python
from verifier_anchored_rl import OpenAICompatibleLLMClient, LLMContextMDP

client = OpenAICompatibleLLMClient(
    base_url="http://gpu-box.local:8000/v1",
    model="Qwen/Qwen2.5-Math-1.5B-Instruct",
    timeout_s=120.0,
)
```

vLLM uses the full HuggingFace model path as the `model` field.

### Notes

- vLLM expects unquantized or AWQ/GPTQ-quantized weights, not GGUFs.
  Use HuggingFace model IDs.
- For models that do not fit in single-GPU VRAM, set
  `--tensor-parallel-size N` to shard across N GPUs.
- vLLM supports OpenAI-compatible `response_format={"type":
  "json_object"}` natively when the guided-decoding backend is
  enabled.

## Option C: llama.cpp server

For deployments where Ollama's wrapper is unwanted but vLLM is too
heavy, llama.cpp's built-in HTTP server works:

```bash
# On the GPU box
./llama-server -m /path/to/model.gguf --host 0.0.0.0 --port 8080 -c 8192
```

```python
client = OpenAICompatibleLLMClient(
    base_url="http://gpu-box.local:8080/v1",
    model="local",  # llama.cpp ignores the model name in single-model mode
)
```

## Choosing a model for the smoke test

The Tier-2 smoke test asks: does the framework's loop close on a small
benchmark slice? Optimize for low engineering friction over headline
accuracy.

| Model | Params | VRAM (bf16) | Notes |
|---|---|---|---|
| `Qwen2.5-Math-1.5B-Instruct` | 1.5 B | ~3 GB | Math-tuned, JSON-friendly. **Recommended.** |
| `Qwen2.5-1.5B-Instruct` | 1.5 B | ~3 GB | General-purpose, slightly weaker on math. |
| `SmolLM2-1.7B-Instruct` | 1.7 B | ~3.5 GB | Smallest viable. Less reliable JSON. |
| `Qwen2.5-Math-7B-Instruct` | 7 B | ~14 GB | If you have a 24 GB+ GPU. Headline-quality math. |
| `Phi-3.5-mini-instruct` | 3.8 B | ~7.5 GB | Microsoft's small reasoner. Good JSON support. |

Use bf16 on Ampere+ (RTX 3000 series and later) for best speed-quality
tradeoff. Use fp16 on older cards or set Ollama's quantization tag to
`q4_K_M` if memory-constrained.

## Reachability checks

Before running the framework, sanity-check the server is reachable and
producing JSON:

```bash
# From the dev box, a one-shot request:
curl -s http://gpu-box.local:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen2.5-math:1.5b",
        "messages": [
            {"role": "system", "content": "Reply in JSON."},
            {"role": "user", "content": "Return {\"ping\": \"pong\"} only."}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0
    }' | python -m json.tool
```

If the response includes `"content": "{\"ping\": \"pong\"}"` (or
similar), the loop is ready.

## Firewall and routing

- Ollama and vLLM bind to `127.0.0.1` by default. Use
  `OLLAMA_HOST=0.0.0.0:11434` or `--host 0.0.0.0` to expose on all
  interfaces.
- If the GPU box has a host-level firewall (ufw, firewalld, Windows
  Defender), open the chosen port to the dev box's IP only. Do not
  expose to the internet.
- Hostname resolution: if mDNS works on your network, use
  `gpu-box.local`. Otherwise use the LAN IP directly. SSH tunnels
  (`ssh -L 11434:localhost:11434 gpu-box`) are a viable fallback if
  direct exposure is awkward.

## Cost accounting hooks

The framework paper's pre-registered experimental plan calls for
FLOP-matched evaluation. The LLM server should log per-request token
counts and timings; both vLLM and Ollama expose these via the API
response's `usage` field. The `OpenAICompatibleLLMClient` can be
extended to capture them on every call for reproducible compute
accounting at a later stage.
