# vLLM-batched, decoupled-embedder fast pipeline (design)

Date: 2026-06-22. Status: approved, pending spec review.

## Motivation

The closed-loop runs hold the GPU at roughly 10 percent utilization. Two
independent causes: (1) generation is batch-1 and sequential (one decoding
sequence cannot saturate the GPU), and (2) the schema constraint
(lm-format-enforcer) computes the allowed-token set in Python every decoded
token, stalling the GPU on CPU work. For the AAAI timeline we need many runs at
3B and beyond; throughput is the gating engineering work. The fix is to batch
generation and move the constraint onto the GPU. Because the high-throughput
generation backend (vLLM) does not expose the LLM hidden state, the heads' state
embedding must be decoupled from the generator, which is also desirable on its
own (it enables embedder ablations, including a future GNN over the reasoning
graph).

## Goal

A decoupled, batched, fast collection-and-train pipeline:
- generation via vLLM with GPU-side guided-JSON decoding and continuous
  batching, fed by a collection loop that runs many trajectories concurrently so
  the GPU stays saturated;
- state embedding behind a swappable `Embedder` interface, default a dedicated
  text embedder, with the legacy LLM `h_T` retained as a selectable backend and
  a GNN over the reasoning graph as a planned future backend.

Success criterion: on a 3B model, GPU utilization rises from ~10 percent to a
high fraction during collection, and per-decision wall-clock drops substantially
versus the lm-format-enforcer batch-1 path.

## Non-goals

- The full benchmark study (separate effort; this enables it).
- The GNN embedder (future ablation; the interface accommodates it, we do not
  build it now).
- Multi-GPU / distributed serving.
- Changing the learning algorithm, the MDP semantics, or the reward (the
  2026-06-22 reward fixes stand).

## Architecture

Generation and representation are fully decoupled.

### 1. Generation: vLLM, batched, guided JSON

Use vLLM with structured output (`guided_json` from the verb's JSON schema), so
every generated token is constrained on the GPU with negligible per-token
overhead, and concurrent requests are continuously batched. Two viable
mechanisms; the plan picks one:
- vLLM offline `LLM` class: gather all pending verb generations across the
  in-flight trajectories into one `llm.generate(list, SamplingParams(guided_...))`
  call per step (in-process, no server, maximal batching). Recommended.
- vLLM OpenAI-compatible server + concurrent client requests: reuse the existing
  `OpenAICompatibleLLMClient` (already documents vLLM), extended to pass
  `guided_json` and issue requests concurrently. Simpler integration, async
  plumbing.

Either way the contract is a batched generation call: a list of (prompt, schema)
requests in, a list of schema-valid JSON objects out. The masked-action recovery
remains a backstop, but guided JSON should make it rare.

### 2. Concurrent / batched collection

The throughput win requires concurrency, because a single trajectory's MCTS is
sequential. The collection loop runs many problems' trajectories at once and
gathers the generation requests they produce at each step into a single batched
vLLM call ("batched MCTS"): all in-flight trees advance in lockstep, each step's
pending verb-instantiation (and leaf) generations are batched together, results
are distributed back to the trees. This is the core implementation challenge and
is detailed in the plan. The MCTS math per tree is unchanged; only the execution
is batched.

### 3. Embedding: swappable `Embedder`

A single interface, `embed(state) -> vector` (and a batched
`embed_batch(states) -> vectors`), with backends:
- `TextEmbedder` (default for viability): a dedicated sentence/text embedding
  model over the state serialization. Model id is configurable (small by
  default, may be larger); batched; runs in-process (cheap) or as a separate
  service. Output dim feeds the heads.
- `LLMHiddenStateEmbedder` (retained): the current `h_T` path (last-layer hidden
  at the sentinel), kept as a baseline/ablation. Requires an in-process forward
  of the base model (so it is the slower, coupled option, used when comparing).
- `GraphEmbedder` (future, not built now): a GNN over the reasoning graph
  (typed nodes for problem/subgoals/hypotheses/facts/verifier-calls, typed
  edges for decomposition/parent/verification), node features = type embedding
  plus a frozen text embedding of each artifact plus status scalars, message
  passing, pooled to a vector. Permutation-invariant over artifacts; trainable
  with the heads. The interface is designed so this is a drop-in.

The heads (`architecture.py`) consume `embed(state)` exactly as today; only the
embedding source and its dimension change. Heads are retrained per embedder
(embeddings are not comparable across backends), which the closed loop already
does from scratch.

### 4. Harness integration

The run harness gains: a generation backend selector (vLLM vs the existing
in-process/HTTP clients for CPU tests) and an embedder selector
(`text` | `hidden_state`, with model id). Resumability, per-iteration metrics,
the sharpening/concentration diagnostics, and FLOP accounting all carry over;
FLOP accounting now reads generation token counts from vLLM usage and embedding
forwards from the embedder. The "one forward per decision" property is
deliberately retired in favor of decoupled, batched throughput (noted in the
paper as a variant; the FLOP axis still reports honest generation + embedding
costs).

## Data flow (per collection step, batched)

1. For each in-flight trajectory, MCTS requests the verb generations it needs at
   the current frontier (and any leaf evaluations).
2. The loop gathers all such (prompt, schema) requests across trajectories into
   one batched vLLM `guided_json` call.
3. States needing embeddings are gathered into one batched `embed_batch` call.
4. Results are distributed back; each tree advances; completed trajectories add
   their ReplaySamples; the rest continue.

## Error handling

- A vLLM request that still fails to parse (rare under guided JSON) falls back to
  the existing masked-action recovery.
- vLLM server/process errors: surfaced and retried; the resumable checkpointing
  means a crash loses at most one trajectory boundary.
- Embedder failures are fatal (no silent fallback) since embeddings are
  load-bearing for the heads.

## Testing

- CPU, no GPU: the `Embedder` interface (a fake/text embedder returning fixed
  vectors), the batched-generation contract against a fake batched client, and
  the batched-collection control flow (many trajectories, gathered requests,
  correct result distribution) with `EchoLLMClient`-style fakes. The MCTS math is
  already covered.
- GPU smoke (3B): a vLLM-backed run that asserts (a) verbs are valid/non-masked,
  (b) GPU utilization is high during collection, (c) per-decision wall-clock is
  substantially below the lm-format-enforcer path. This gates adoption.
- Equivalence: with the same seed and a single in-flight trajectory + batch size
  1, the batched path produces the same trajectories as the sequential path
  (modulo the documented compute-order floor), so batching is a pure throughput
  change.

## Scope for the first build

vLLM generation (guided JSON) + concurrent/batched collection + the default
`TextEmbedder`, validated on 3B for the utilization/throughput win. Keep
`LLMHiddenStateEmbedder` (`h_T`) selectable as the baseline. Defer the GNN.

## Risks

- New dependency (vLLM) and its install/footprint; pin a known-good version.
- Batched MCTS correctness: the gather/distribute must preserve each tree's
  semantics; the equivalence test guards this.
- Decoupling retires the one-forward FLOP story; reframe in the paper.
- Embedder dim/coupling: a config/checkpoint dim mismatch must fail loudly
  (as the pilot's embedder/checkpoint guard already does).
