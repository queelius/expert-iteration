# Outer collection-and-train loop: design spec

**Date:** 2026-06-18
**Status:** approved (design); pending implementation plan
**Scope:** `method.tex` §3.7 Algorithm 1, the outer loop that wires the three
CPU-verified modules (`architecture.HiddenStateActor`, `train.Trainer`,
`replay.ReplayBuffer`) to the planner (`mcts`) and the in-process encoder
(`shared_forward.SharedForwardEncoder`).

## Framing (RL-first, per CLAUDE.md §1)

The headline algorithm is **approximate Q-learning** (`method.tex` §3.7):
the parametric action-value head `Q_ψ(s,a)` is regressed onto MCTS
chance-node estimates `Q_MCTS(s,a)`, and the target network `Q_θ̄` supplies
bootstrap leaf values inside search. **MCTS is the policy-improvement
operator**, not the central object: each round it turns the current heads
into stronger training targets (`π_MCTS`, `Q_MCTS`), which are distilled
back into the heads, which then drive better search. This spec describes
the loop that closes that cycle. It does not introduce any new learning
objective. `train.py` already implements `L_total = L_Q + λ·L_π`
(Eqs. lossq/losspi/losstotal).

## Goal

A CPU-testable orchestrator that runs Algorithm 1 end to end:

```
Initialize Q_ψ, π_θ, target Q_θ̄ ← Q_ψ
for iteration = 1, 2, …:
    # collect (new orchestration)
    for each train problem:
        drive a trajectory; at each visited non-terminal node:
            run MCTS with prior = π_θ and leaf value = max_{a'} Q_θ̄(s', a')
            read π_MCTS(·|s), Q_MCTS(s,·), embed(s)
            build a ReplaySample → add to buffer
    # train (exists: train.Trainer)
    run M minibatch steps of L_Q + λ·L_π; sync target every N steps
    # record (new, thin)
    emit IterationStats: loss curve, FLOP counters, buffer size
```

The collect step and the per-iteration record are the only genuinely new
orchestration; the optimizer step, priority update, and target sync already
exist in `train.Trainer`.

## Decisions taken (2026-06-18)

1. **Leaf value:** add the `target-Q` leaf-evaluation hook to MCTS **now**
   (faithful to Algorithm 1 line 3 / the journal). Additive: the default
   preserves today's rollout behavior.
2. **Encoder coupling:** the loop depends on a **pluggable collection
   backend**. The real backend is the `SharedForwardEncoder` (native-dim
   `h_T` feeds the heads; cache-reuse generation); a CPU `FakeEncoder`
   makes the whole loop end-to-end testable with no GPU.
3. **FLOP accounting:** include a **thin** per-iteration layer now (read
   the encoder's existing token counters; no new hot-path instrumentation).

## Notation (defined before use, per CLAUDE.md §12)

- `s`: a `MathState`; `a`: a `MathAction` (one of `NUM_VERBS` verbs).
- `embed(s) = h_T`: last-layer hidden state at the `[STATE_END]` sentinel
  (`method.tex` Eq. embed), produced by `SharedForwardEncoder.encode`.
- `avail(s)`: legal verbs at `s` (`available_actions`).
- `π_θ(a|s)`: masked-softmax of the **online** policy head over `avail(s)`.
- `Q_ψ(s,a)`: the **online** action-value head; `Q_θ̄(s,a)`: the
  periodically-synced **target** copy (`HiddenStateActor.target_q`).
- `π_MCTS(a|s) = N(s,a)^{1/τ} / Σ_b N(s,b)^{1/τ}`: visit-count
  distribution at temperature `τ` (`mcts.visit_count_policy`).
- `Q_MCTS(s,a)`: chance-node mean value (`mcts.root_q_estimates`).
- `V(s') = max_{a' ∈ avail(s')} Q_θ̄(s', a')`: the bootstrap leaf value.

## Component 1: `value_fn` hook in MCTS (edit `mcts.py`)

Add a type alias `ValueFn = Callable[[S], float]` and a field to
`MCTSConfig`:

```python
value_fn: ValueFn | None = None
```

In `mcts_search`, at the expansion/simulation step, replace the leaf-value
computation for a **non-terminal** expanded child:

```python
if child_decision.terminal:
    leaf_value = 0.0
elif config.value_fn is not None:
    leaf_value = config.value_fn(child_decision.state)   # bootstrap V(s')
else:
    leaf_value = _rollout(mdp, child_decision.state, config, rng)  # today
```

Properties:
- `value_fn is None` gives byte-identical behavior to today. The §10
  triangulation tests (value iteration vs expectimax vs MCTS on the
  S/A/G/B MDP) are **untouched**.
- Terminal leaves remain `0.0` (the package's terminal-value-zero
  convention); `value_fn` is consulted only at non-terminal leaves.
- The per-edge discounted backup (`_backup`) is unchanged: the leaf value
  enters as `G ← leaf_value`, then `G ← r + γ·G` up the path.

## Component 2: `StateActor` bridge + `PrefillCache` (new `planner_actor.py`)

`StateActor` wraps `(encoder, HiddenStateActor, PrefillCache)` and exposes
exactly what the planner hooks and the loop need over `MathState`:

```python
class StateActor:
    def embed(self, state) -> Tensor            # h_T (via PrefillCache)
    def prior(self, state, actions) -> list[float]   # online π_θ, masked, sums to 1
    def leaf_value(self, state) -> float        # max over avail of target_q
```

- `prior` builds the PUCT prior from the **online** policy head; ordering
  matches the `actions` list MCTS passes (so `_expand`'s prior alignment
  holds).
- `leaf_value` masks `target_q(embed(s))` to `avail(s)` and takes the max.
  Used as `MCTSConfig.value_fn`.
- Depends on a minimal **encoder Protocol** (PEP 544), not on
  `SharedForwardEncoder` concretely:

```python
class Encoder(Protocol):
    @property
    def embed_dim(self) -> int: ...
    def encode(self, prefix_text: str) -> StatePrefill: ...
```

A CPU `FakeEncoder` (deterministic vector from a hash of the prefix text,
configurable `embed_dim`) implements this for tests.

**`PrefillCache`: the one-forward-per-node mechanism.** MCTS interleaves
`prior(s)`, `value_fn(s')`, and `sample_transition(s,a)` across states in
no clean per-node order, so encode-once is made *emergent*: an
LRU-bounded cache keyed by the serialized state **prefix** maps to a
`StatePrefill`. Both the heads path (`StateActor.embed`) and the MDP's
generation path consult this one cache, so a node is prefilled once and
reused for the heads **and** every verb generation at it. This is what
earns "one forward per decision" plus the measured 53% prefill saving
honestly. Cache correctness of generation reuse itself is already
GPU-smoke-verified (`run_shared_forward_smoke.py`); here we unit-test only
the routing / cache-hit behavior with `FakeEncoder`.

## Component 3: additive encoder path in the MDP (edit `llm_context_mdp.py`)

`LLMContextMDP` gains optional `encoder` and `prefill_cache` fields. In
`sample_transition`:

```python
if self.encoder is not None:
    prefill = self.prefill_cache.get_or_encode(_state_prefix(state), self.encoder)
    response, _ = self.encoder.instantiate_json(
        prefill, _verb_suffix(verb, state), schema,
        temperature=self.temperature, max_tokens=self.max_tokens, seed=sampling_seed,
    )
else:
    response = self.llm_client.complete_json(...)   # unchanged
```

This is **additive**: when `encoder is None` (Mock/Echo/HTTP clients, all
tests not exercising the encoder), the existing `complete_json` path is
byte-for-byte unchanged. The `LLMResponseError` to `_masked_action`
recovery wraps both paths identically.

## Component 4: the loop (new `outer_loop.py`)

```python
@dataclass
class IterationStats:
    iteration: int
    losses: list[dict[str, float]]       # per train step: q, pi, total
    prefill_calls: int; prefill_tokens: int
    gen_calls: int;     gen_tokens: int
    n_trajectories: int; n_decisions: int; n_simulations: int
    buffer_size: int

def collect_iteration(problems, mdp_factory, state_actor, config, buffer, rng,
                      *, tau: float, max_turns: int) -> dict[str, int]: ...

def run_outer_loop(problems, mdp_factory, state_actor, trainer, buffer, *,
                   n_iterations, train_steps_per_iter, batch_size,
                   min_buffer_size, tau, rng) -> list[IterationStats]: ...
```

- `mdp_factory(problem) -> LLMContextMDP` constructs the per-problem env
  (carrying the encoder + shared `prefill_cache` for the real run, or a
  `MockLLMClient` for tests).
- `config` is an `MCTSConfig` with `prior_fn = state_actor.prior` and
  `value_fn = state_actor.leaf_value`.
- **Trajectory control (judgment call):** advance each trajectory by
  **sampling from `π_MCTS` at temperature `τ`** (the annealed `τ` of
  §3.7), and record `π_MCTS` at that same `τ` as the distillation target.
  Sampling (vs greedy argmax-visits) diversifies the buffer's state
  distribution, which matters most early. AlphaZero's sample-early /
  argmax-late split is deferred; v1 uses one `τ`.
- Per visited non-terminal node with `avail(s) ≠ ∅`: run `mcts_search`,
  read `π_MCTS`/`Q_MCTS`, `embed(s)`, build a `ReplaySample` (reuse
  `heads.encode_batch` schema via `replay.sample_from_record`, or build the
  `ReplaySample` directly from the readout; implementation detail),
  `buffer.add(...)`.
- After collection, train only if `len(buffer) ≥ min_buffer_size` (warmup
  gate); otherwise skip training this iteration and keep collecting. Then
  `trainer.run(train_steps_per_iter, batch_size, rng)` (target sync fires
  inside on its `N`-step schedule).
- FLOP counters: snapshot `encoder.{prefill,gen}_{calls,tokens}` before and
  after collection; the delta goes into `IterationStats`.

## Component 5: experiment script (new `experiments/run_outer_loop.py`)

Frodo-bound. Builds the real `SharedForwardEncoder` +
`TransformersLLMClient` (sharing one frozen model via `from_llm_client`),
a `HiddenStateActor` at the model's native `embed_dim`, the shared
`PrefillCache`, runs a few outer iterations on the toy `train` problem
group, and appends an entry to `experiments/results/expert_iteration_rounds.md`
reporting the loss curve, FLOP counters, and the across-iteration shift in
the search visit distribution.

## Testing strategy (the §16 "make the core unmistakably correct" ethos)

- **`test_mcts.py` (edit):**
  - `value_fn=None` reproduces current rollout behavior (regression guard).
  - With `value_fn` set, a non-terminal leaf uses it; terminal leaf stays 0.
  - **Anchor:** on the §10.2 S/A/G/B MDP, `value_fn = V*` lets a tiny-budget
    search reproduce the hand-computed `Q*(S,safe)=4`, `Q*(S,risky)=6`.
- **`test_planner_actor.py` (new):** `prior` sums to 1 over `avail(s)` and
  is 0 elsewhere; `leaf_value` equals `max` of `target_q` over `avail(s)`;
  `PrefillCache` calls `encoder.encode` exactly once per distinct state
  prefix across repeated `embed`/generation requests (encode-once).
- **`test_outer_loop.py` (new), the load-bearing test:** with `FakeEncoder`
  + `MockLLMClient`, prove the **loop closes**, not merely "runs":
  1. runs `n_iterations` without error;
  2. buffer grows, then caps at capacity;
  3. combined loss trends down across the run;
  4. **the learned policy prior diverges from uniform and shifts the MCTS
     visit distribution across iterations** (search → distill → better
     prior → different search: the expert-iteration cycle actually
     closing);
  5. target sync fires on its `N`-step schedule;
  6. `IterationStats` FLOP/loss/buffer fields are populated.

## Error handling & determinism

- Masked-action failures (`LLMResponseError`, bad target id) keep the
  existing `-0.01` + `turn+1` recovery, for both generation paths.
- Trajectories that never commit hit `max_turns`, becoming terminal-failure
  (already handled by `is_terminal`).
- One seeded `np.random.Generator` threaded throughout; per-call LLM
  sampling seed via the existing `seed_llm` path.
- `PrefillCache` is LRU-bounded to cap memory across a long collection.

## Non-goals (this deliverable)

- No new learning objective (uses `train.py` as-is).
- No `prompt_strategy="focused"` ablation (still `NotImplementedError`).
- No AlphaZero sample-early/argmax-late schedule (one `τ`).
- No hyperparameter sweep; starting values from §3.7 (`γ=1`,
  `c_PUCT∈[1.0,1.5]`, `τ` annealed near 1, `N`~1000 steps).
- No change to the classical-MDP scaffold beyond the additive `value_fn`.

## Module-change summary

| File | New? | Change |
|------|------|--------|
| `mcts.py` | edit | `ValueFn` alias; `value_fn` in `MCTSConfig`; leaf-eval branch |
| `llm_context_mdp.py` | edit | optional `encoder`/`prefill_cache` generation path (additive) |
| `planner_actor.py` | new | `StateActor`, `Encoder` Protocol, `PrefillCache` |
| `outer_loop.py` | new | `collect_iteration`, `run_outer_loop`, `IterationStats` |
| `experiments/run_outer_loop.py` | new | frodo run + journal entry |
| `tests/test_mcts.py` | edit | `value_fn` additive + `V*` anchor |
| `tests/test_planner_actor.py` | new | prior/leaf/cache properties |
| `tests/test_outer_loop.py` | new | end-to-end "loop closes" with fakes |
