# Outer collection-and-train loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the expert-iteration cycle (`method.tex` §3.7 Algorithm 1): MCTS collection with the current heads as priors and the target-Q as bootstrap leaf value, then replay-buffered approximate-Q-learning + policy distillation, repeated for N iterations.

**Architecture:** A `value_fn` leaf-evaluation hook is added to the existing MCTS (additive; default preserves rollout behavior). A `StateActor` bridge wraps `(encoder, HiddenStateActor, PrefillCache)` and supplies the planner's prior and leaf-value hooks plus the heads' input embedding over `MathState`. An `outer_loop` module drives trajectories, reads `pi_MCTS`/`Q_MCTS`/`embed(s)` per node into the `ReplayBuffer`, runs the existing `Trainer`, and records per-iteration FLOP/loss stats. The whole loop is CPU-testable via a `FakeEncoder`; the real run uses `SharedForwardEncoder` on the GPU box.

**Tech Stack:** Python 3.11+, numpy, pytest, torch (heads only). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-18-outer-loop-design.md`

## Global Constraints

- Package dir is `src/verifier_anchored_rl/`; tests in `tests/`; experiments in `experiments/` (CLAUDE.md).
- Two MDP protocols stay separate; use `typing.Protocol` (structural), never ABCs (CLAUDE.md "layered interfaces").
- The classical-MDP scaffold is the validation anchor: the §10 triangulation tests must keep passing unchanged. The `value_fn` change is additive (`value_fn=None` is byte-identical to today).
- RL-first framing (CLAUDE.md §1): MCTS is the policy-improvement operator, not the central object. Comments/docstrings must not regress to MCTS-first.
- Style (CLAUDE.md §12): expanded sums before `E[...]`; loop variable names match the math symbols; define notation where first used.
- **No em-dash glyph (U+2014) in any file** (a soul plugin hook rejects the write). Use commas, colons, periods, or parentheses.
- Every git commit message ends with these two trailer lines verbatim:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01MGnZr92THoybtHaa9vfk2d
  ```
- Work happens on the `outer-loop` branch (already created). Run the full suite (`pytest -q`) green before the final task's commit.
- Verb-index layout is `heads.VERB_NAME_TO_INDEX` (position in `heads.VERB_ORDER = list(Verb)`); `heads.NUM_VERBS == len(Verb)`. Never hardcode verb indices.

---

### Task 1: `value_fn` leaf-evaluation hook in MCTS

**Files:**
- Modify: `src/verifier_anchored_rl/mcts.py` (alias near line 122-125; `MCTSConfig` near line 157-162; leaf branch near line 452-457)
- Test: `tests/test_mcts.py`

**Interfaces:**
- Produces: `mcts.ValueFn = Callable[[object], float]`; new field `MCTSConfig.value_fn: ValueFn | None = None`. When set, a non-terminal expanded leaf is evaluated by `value_fn(state)` instead of `_rollout`; terminal leaves stay `0.0`.
- Consumes: existing `mcts_search`, `root_q_estimates`; the `SAGBGenerative` wrapper and `_S,_A,_G,_B` already imported at the top of `tests/test_mcts.py`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mcts.py`:

```python
# ---------------------------------------------------------------------------
# value_fn leaf-evaluation hook (method.tex 3.7: target-Q as bootstrap leaf).
# ---------------------------------------------------------------------------

# V* for the SAGBMDP, hand-computed in test_value_iteration:
#   V*(S)=6, V*(A)=10, V*(G)=0, V*(B)=0.
_VSTAR = {_S: 6.0, _A: 10.0, _G: 0.0, _B: 0.0}


def test_value_fn_none_matches_rollout_default():
    """value_fn=None must leave behavior byte-identical to the rollout path."""
    mdp = SAGBGenerative()
    cfg = MCTSConfig(num_simulations=200, c_puct=1.0, discount=1.0,
                     rollout_depth_limit=10)
    root_a = mcts_search(mdp, _S, cfg, np.random.default_rng(7))
    root_b = mcts_search(mdp, _S, cfg, np.random.default_rng(7))
    qa = {a: round(q, 9) for a, q in root_q_estimates(root_a).items()}
    qb = {a: round(q, 9) for a, q in root_q_estimates(root_b).items()}
    assert qa == qb  # determinism guard; the rollout path is unchanged


def test_value_fn_bootstraps_nonterminal_leaf_to_handcomputed_q():
    """With value_fn = V*, the chance-node estimates reproduce the
    hand-computed Q*(S, safe)=4 and Q*(S, risky)=6 (method.tex Eq. qmcts)."""
    mdp = SAGBGenerative()
    cfg = MCTSConfig(num_simulations=600, c_puct=1.0, discount=1.0,
                     value_fn=lambda s: _VSTAR[s])
    root = mcts_search(mdp, _S, cfg, np.random.default_rng(0))
    q = {a: v for a, v in root_q_estimates(root).items()}
    assert math.isclose(q["safe"], 4.0, abs_tol=0.4)
    assert math.isclose(q["risky"], 6.0, abs_tol=0.6)


def test_value_fn_not_called_on_terminal_leaf():
    """Terminal leaves stay 0.0; value_fn must never be consulted for them."""
    mdp = SAGBGenerative()
    seen = []
    def vf(s):
        seen.append(s)
        return 99.0  # absurd value that would corrupt Q if used at terminals
    cfg = MCTSConfig(num_simulations=300, c_puct=1.0, discount=1.0, value_fn=vf)
    root = mcts_search(mdp, _S, cfg, np.random.default_rng(0))
    assert _G not in seen and _B not in seen  # G, B are terminal
    # safe -> G (terminal) with reward 4, so its estimate is unaffected by vf.
    assert math.isclose(root_q_estimates(root)["safe"], 4.0, abs_tol=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mcts.py::test_value_fn_bootstraps_nonterminal_leaf_to_handcomputed_q tests/test_mcts.py::test_value_fn_not_called_on_terminal_leaf -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'value_fn'`.

- [ ] **Step 3: Add the alias and config field**

In `src/verifier_anchored_rl/mcts.py`, after the `RolloutPolicy` alias (line 125), add:

```python
# A leaf-value estimator V(s): used in place of a rollout to bootstrap the
# value of a non-terminal expanded leaf. In expert iteration this is
# max_a' Q_target(s', a') (method.tex Eq. bellmantarget); MCTS thus becomes
# AlphaZero-style (value net at the leaf) rather than pure rollout.
ValueFn = Callable[[object], float]
```

In `MCTSConfig` (after `rollout_policy` on line 162), add the field and document it in the docstring's Parameters list:

```python
    value_fn: ValueFn | None = None
```

Add to the `MCTSConfig` docstring, after the `rollout_policy` entry:

```
    value_fn:
        Optional leaf-value estimator ``value(state) -> float``. When
        provided, a non-terminal expanded leaf is scored by ``value_fn``
        instead of a rollout (terminal leaves remain 0). ``None`` keeps
        the rollout behavior unchanged.
```

- [ ] **Step 4: Wire the leaf branch in `mcts_search`**

In `src/verifier_anchored_rl/mcts.py`, replace the leaf-value block (currently lines 452-455):

```python
        if child_decision.terminal:
            leaf_value = 0.0
        else:
            leaf_value = _rollout(mdp, child_decision.state, config, rng)
```

with:

```python
        if child_decision.terminal:
            leaf_value = 0.0
        elif config.value_fn is not None:
            # Bootstrap V(s') from the learned estimator (method.tex 3.7),
            # in place of a rollout. The per-edge backup still folds in the
            # expansion reward, so Q(s,a) -> r + gamma * V(s').
            leaf_value = config.value_fn(child_decision.state)
        else:
            leaf_value = _rollout(mdp, child_decision.state, config, rng)
```

- [ ] **Step 5: Run the new tests and the triangulation regression**

Run: `pytest tests/test_mcts.py -q`
Expected: PASS (new tests pass; all prior MCTS tests, including the §10 triangulation, still pass).

- [ ] **Step 6: Commit**

```bash
git add src/verifier_anchored_rl/mcts.py tests/test_mcts.py
git commit -F- <<'EOF'
Add value_fn leaf-evaluation hook to MCTS (method.tex 3.7)

Additive: value_fn=None preserves the rollout path byte-identically, so
the section-10 triangulation tests are untouched. With value_fn set, a
non-terminal expanded leaf is bootstrapped by V(s') instead of a rollout,
making search AlphaZero-style. Anchored on the SAGBMDP: value_fn=V*
reproduces the hand-computed Q*(S,safe)=4, Q*(S,risky)=6.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MGnZr92THoybtHaa9vfk2d
EOF
```

---

### Task 2: `PrefillCache`, `Encoder` protocol, and the `FakeEncoder` test double

**Files:**
- Create: `src/verifier_anchored_rl/planner_actor.py` (cache + protocol only this task)
- Create: `tests/fakes.py` (shared test doubles)
- Test: `tests/test_planner_actor.py` (cache behavior)

**Interfaces:**
- Produces: `planner_actor.Encoder` (Protocol: `embed_dim` property, `encode(prefix_text) -> StatePrefill`); `planner_actor.PrefillCache(capacity=512)` with `get_or_encode(prefix_text, encoder) -> StatePrefill` (LRU, encode-once per distinct prefix) and `__len__`.
- Produces: `tests.fakes.FakeEncoder(embed_dim=32)` implementing the `Encoder` protocol plus `instantiate_json(...)` (for Task 4's MDP routing test) and FLOP counters (`prefill_calls`, `prefill_tokens`, `gen_calls`, `gen_tokens`).
- Consumes: `shared_forward.StatePrefill`.

- [ ] **Step 1: Write the shared `FakeEncoder` double**

Create `tests/fakes.py`:

```python
"""Shared CPU test doubles for the outer-loop tests.

``FakeEncoder`` implements the ``planner_actor.Encoder`` protocol with a
deterministic hidden state derived from a hash of the prefix text, so the
whole collection-and-train loop is testable with no GPU and no model. It
also provides ``instantiate_json`` (echo-valued, schema-conformant) so it
can stand in for the SharedForwardEncoder on the MDP's generation path,
and the same FLOP counters the real encoder maintains.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

from verifier_anchored_rl.shared_forward import StatePrefill


class FakeEncoder:
    def __init__(self, embed_dim: int = 32):
        self._embed_dim = embed_dim
        self.prefill_calls = 0
        self.prefill_tokens = 0
        self.gen_calls = 0
        self.gen_tokens = 0

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def encode(self, prefix_text: str) -> StatePrefill:
        import torch

        digest = hashlib.sha256(prefix_text.encode("utf-8")).digest()[:8]
        seed = int.from_bytes(digest, "little")
        vec = np.random.default_rng(seed).standard_normal(self._embed_dim)
        hidden = torch.from_numpy(vec.astype("float32"))
        n_tok = max(1, len(prefix_text.split()))
        self.prefill_calls += 1
        self.prefill_tokens += n_tok
        return StatePrefill(hidden=hidden, cache=None, n_prefix_tokens=n_tok,
                            prefix_ids=None)

    def instantiate_json(
        self, prefill: StatePrefill, suffix_text: str, schema: Mapping[str, Any],
        *, temperature: float = 1.0, max_tokens: int = 192,
        seed: int | None = None, reuse_cache: bool = True,
    ) -> tuple[dict[str, Any], int]:
        from verifier_anchored_rl.llm_client import _echo_value

        self.gen_calls += 1
        self.gen_tokens += 1
        props = schema.get("properties", {})
        return {k: _echo_value(spec) for k, spec in props.items()}, 1
```

- [ ] **Step 2: Write the failing cache test**

Create `tests/test_planner_actor.py`:

```python
"""Tests for the StateActor bridge and the PrefillCache."""

from __future__ import annotations

import numpy as np
import pytest

from tests.fakes import FakeEncoder


def test_prefill_cache_encodes_once_per_distinct_prefix():
    pytest.importorskip("torch")
    from verifier_anchored_rl.planner_actor import PrefillCache

    enc = FakeEncoder(embed_dim=8)
    cache = PrefillCache(capacity=16)

    a1 = cache.get_or_encode("state A", enc)
    a2 = cache.get_or_encode("state A", enc)
    b1 = cache.get_or_encode("state B", enc)

    assert enc.prefill_calls == 2  # "state A" encoded once, "state B" once
    assert a1 is a2  # cache hit returns the same StatePrefill object
    assert len(cache) == 2
    assert not np.allclose(a1.hidden.numpy(), b1.hidden.numpy())


def test_prefill_cache_lru_evicts_oldest():
    pytest.importorskip("torch")
    from verifier_anchored_rl.planner_actor import PrefillCache

    enc = FakeEncoder(embed_dim=8)
    cache = PrefillCache(capacity=2)
    cache.get_or_encode("a", enc)
    cache.get_or_encode("b", enc)
    cache.get_or_encode("a", enc)  # touch "a" so "b" is the oldest
    cache.get_or_encode("c", enc)  # evicts "b"
    assert len(cache) == 2
    cache.get_or_encode("b", enc)  # must re-encode "b"
    assert enc.prefill_calls == 4  # a, b, c, b-again
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_planner_actor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'verifier_anchored_rl.planner_actor'`.

- [ ] **Step 4: Create `planner_actor.py` with the protocol and cache**

Create `src/verifier_anchored_rl/planner_actor.py`:

```python
"""Bridge between the trainable heads and the MCTS planner over MathState.

method.tex 3.7: the outer loop runs MCTS with the online policy head as the
PUCT prior and the target-Q head as the bootstrap leaf value, and feeds the
same per-state embedding embed(s) = h_T to the heads. This module wires
those three reads (embed / prior / leaf value) over the LLM context MDP's
MathState, depending only on a minimal encoder Protocol so a CPU fake can
drive the whole loop with no GPU.

MCTS interleaves prior(s), value_fn(s'), and sample_transition(s, a) across
states in no clean per-node order, so encode-once is made emergent: the
PrefillCache, keyed by the serialized state prefix, ensures a node is
prefilled exactly once and reused by the heads and (when the MDP runs the
encoder generation path) by every verb generation at that node. This is the
"one forward per decision" property (method.tex 3.4).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Protocol


class Encoder(Protocol):
    """Minimal encoder contract the heads bridge depends on."""

    @property
    def embed_dim(self) -> int: ...

    def encode(self, prefix_text: str) -> Any: ...  # returns a StatePrefill


class PrefillCache:
    """LRU cache from a state-prefix string to its StatePrefill.

    The cache makes encode-once emergent regardless of the order in which
    MCTS asks for a state's prior, leaf value, or generation: the first
    request for a prefix encodes it; the rest are cache hits.
    """

    def __init__(self, capacity: int = 512):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._store: "OrderedDict[str, Any]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._store)

    def get_or_encode(self, prefix_text: str, encoder: Encoder) -> Any:
        if prefix_text in self._store:
            self._store.move_to_end(prefix_text)
            return self._store[prefix_text]
        prefill = encoder.encode(prefix_text)
        self._store[prefix_text] = prefill
        self._store.move_to_end(prefix_text)
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)  # FIFO/LRU eviction of the oldest
        return prefill


__all__ = ["Encoder", "PrefillCache"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_planner_actor.py -v`
Expected: PASS (both cache tests).

- [ ] **Step 6: Commit**

```bash
git add src/verifier_anchored_rl/planner_actor.py tests/fakes.py tests/test_planner_actor.py
git commit -F- <<'EOF'
Add PrefillCache + Encoder protocol + FakeEncoder double

PrefillCache (LRU, keyed by state-prefix) makes "encode once per node"
emergent across MCTS's interleaved prior/value/generation calls. The
Encoder protocol keeps the heads bridge GPU-free in tests; FakeEncoder is
the deterministic CPU double the whole outer loop is tested against.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MGnZr92THoybtHaa9vfk2d
EOF
```

---

### Task 3: `StateActor` (embed / prior / leaf_value)

**Files:**
- Modify: `src/verifier_anchored_rl/planner_actor.py` (add `StateActor`)
- Test: `tests/test_planner_actor.py` (add StateActor tests)

**Interfaces:**
- Produces: `planner_actor.StateActor(encoder, actor, cache=None, *, depth_limit=3)` with:
  - `embed(state) -> torch.Tensor` shape `(embed_dim,)` (via the cache);
  - `prior(state, actions) -> list[float]` aligned to `actions`, summing to 1 over them (online policy head, masked softmax); usable as `MCTSConfig.prior_fn`;
  - `leaf_value(state) -> float` = max over `available_actions(state)` of the target-Q head; usable as `MCTSConfig.value_fn`.
- Consumes: `architecture.HiddenStateActor` (`pi`, `target_q`); `heads.VERB_NAME_TO_INDEX`, `heads.NUM_VERBS`; `llm_context_mdp._state_prefix`, `available_actions`; Task 2's `PrefillCache`.

- [ ] **Step 1: Write the failing StateActor tests**

Append to `tests/test_planner_actor.py`:

```python
def _toy_mdp():
    from verifier_anchored_rl.llm_client import EchoLLMClient
    from verifier_anchored_rl.llm_context_mdp import LLMContextMDP

    return LLMContextMDP(problem_statement="2+2", expected_answer="4",
                         llm_client=EchoLLMClient(), depth_limit=3, max_turns=4)


def test_state_actor_prior_sums_to_one_over_actions():
    pytest.importorskip("torch")
    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.planner_actor import StateActor

    enc = FakeEncoder(embed_dim=16)
    actor = HiddenStateActor(embed_dim=16)
    sa = StateActor(enc, actor, depth_limit=3)

    mdp = _toy_mdp()
    state = mdp.initial_state()
    actions = list(mdp.actions(state))
    prior = sa.prior(state, actions)

    assert len(prior) == len(actions)
    assert all(p >= 0.0 for p in prior)
    assert abs(sum(prior) - 1.0) < 1e-5


def test_state_actor_leaf_value_is_max_target_q_over_available():
    pytest.importorskip("torch")
    import torch
    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.heads import VERB_NAME_TO_INDEX
    from verifier_anchored_rl.llm_context_mdp import available_actions
    from verifier_anchored_rl.planner_actor import StateActor

    enc = FakeEncoder(embed_dim=16)
    actor = HiddenStateActor(embed_dim=16)
    sa = StateActor(enc, actor, depth_limit=3)

    mdp = _toy_mdp()
    state = mdp.initial_state()
    avail = list(available_actions(state, depth_limit=3))

    h = sa.embed(state).unsqueeze(0)
    with torch.no_grad():
        qt = actor.target_q(h)[0]
    expected = max(float(qt[VERB_NAME_TO_INDEX[v.value]]) for v in avail)
    assert abs(sa.leaf_value(state) - expected) < 1e-6


def test_state_actor_embed_uses_cache_once_per_state():
    pytest.importorskip("torch")
    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.planner_actor import PrefillCache, StateActor

    enc = FakeEncoder(embed_dim=16)
    actor = HiddenStateActor(embed_dim=16)
    cache = PrefillCache()
    sa = StateActor(enc, actor, cache, depth_limit=3)

    mdp = _toy_mdp()
    state = mdp.initial_state()
    actions = list(mdp.actions(state))

    sa.embed(state)
    sa.prior(state, actions)
    sa.leaf_value(state)
    assert enc.prefill_calls == 1  # all three reads share one prefill
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_planner_actor.py -k state_actor -v`
Expected: FAIL with `ImportError: cannot import name 'StateActor'`.

- [ ] **Step 3: Add `StateActor` to `planner_actor.py`**

In `src/verifier_anchored_rl/planner_actor.py`, add (and extend `__all__`):

```python
class StateActor:
    """Heads-over-MathState bridge: embed(s), prior(s), leaf_value(s).

    ``prior`` reads the ONLINE policy head (the PUCT prior); ``leaf_value``
    reads the TARGET-Q head (the bootstrap leaf), matching method.tex 3.7.
    Both run under no_grad: they drive search, not training.
    """

    def __init__(self, encoder: Encoder, actor: Any, cache: PrefillCache | None = None,
                 *, depth_limit: int = 3):
        self.encoder = encoder
        self.actor = actor
        self.cache = cache if cache is not None else PrefillCache()
        self.depth_limit = depth_limit

    def embed(self, state: Any) -> Any:
        from .llm_context_mdp import _state_prefix

        return self.cache.get_or_encode(_state_prefix(state), self.encoder).hidden

    def prior(self, state: Any, actions: list[Any]) -> list[float]:
        import torch

        from .heads import NUM_VERBS, VERB_NAME_TO_INDEX

        idxs = [VERB_NAME_TO_INDEX[a.verb.value] for a in actions]
        h = self.embed(state).unsqueeze(0)
        mask = torch.zeros(1, NUM_VERBS, dtype=torch.bool)
        for j in idxs:
            mask[0, j] = True
        with torch.no_grad():
            pi = self.actor.pi(h, mask)  # masked softmax, 0 off the mask
        return [float(pi[0, j]) for j in idxs]

    def leaf_value(self, state: Any) -> float:
        import torch

        from .heads import VERB_NAME_TO_INDEX
        from .llm_context_mdp import available_actions

        avail = list(available_actions(state, depth_limit=self.depth_limit))
        if not avail:
            return 0.0  # defensive: MCTS only calls value_fn at non-terminal leaves
        h = self.embed(state).unsqueeze(0)
        with torch.no_grad():
            qt = self.actor.target_q(h)[0]
        return max(float(qt[VERB_NAME_TO_INDEX[v.value]]) for v in avail)
```

Update the export list:

```python
__all__ = ["Encoder", "PrefillCache", "StateActor"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_planner_actor.py -v`
Expected: PASS (cache tests + the three StateActor tests).

- [ ] **Step 5: Commit**

```bash
git add src/verifier_anchored_rl/planner_actor.py tests/test_planner_actor.py
git commit -F- <<'EOF'
Add StateActor heads-over-MathState bridge

Supplies the MCTS prior (online policy head, masked softmax) and leaf value
(target-Q head, max over available verbs) plus the heads' input embedding,
all over the LLM context MDP's MathState and all sharing one prefill per
node via the PrefillCache. This is what mcts_search consumes as prior_fn
and value_fn in the outer loop.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MGnZr92THoybtHaa9vfk2d
EOF
```

---

### Task 4: Additive encoder generation path in `LLMContextMDP`

**Files:**
- Modify: `src/verifier_anchored_rl/llm_context_mdp.py` (`LLMContextMDP` fields near line 980; `sample_transition` None-guard near line 1032 and generation call near line 1062-1072)
- Test: `tests/test_llm_context_mdp.py` (add routing tests)

**Interfaces:**
- Produces: `LLMContextMDP(..., encoder=None, prefill_cache=None)`. When `encoder is not None`, `sample_transition` generates via `encoder.instantiate_json(prefill, _verb_suffix(verb, state), schema, ...)`, taking the prefill from `prefill_cache.get_or_encode(_state_prefix(state), encoder)` when a cache is given. When `encoder is None`, the `llm_client.complete_json` path is unchanged.
- Consumes: `_state_prefix`, `_verb_suffix` (already in the module); Task 2's `FakeEncoder` (has `instantiate_json`) and `PrefillCache`.

- [ ] **Step 1: Write the failing routing tests**

Append to `tests/test_llm_context_mdp.py`:

```python
def test_sample_transition_uses_encoder_when_provided():
    """With an encoder set, generation goes through encoder.instantiate_json
    and the llm_client is not consulted."""
    pytest.importorskip("torch")
    from tests.fakes import FakeEncoder
    from verifier_anchored_rl.llm_client import MockLLMClient
    from verifier_anchored_rl.llm_context_mdp import LLMContextMDP, MathAction, Verb
    from verifier_anchored_rl.planner_actor import PrefillCache

    enc = FakeEncoder(embed_dim=8)
    spy = MockLLMClient(script={})  # any call would raise (empty script)
    mdp = LLMContextMDP(problem_statement="2+2", expected_answer="4",
                        llm_client=spy, encoder=enc, prefill_cache=PrefillCache(),
                        depth_limit=3, max_turns=4)
    state = mdp.initial_state()
    next_state, reward = mdp.sample_transition(
        state, MathAction(verb=Verb.COMMIT_ANSWER), np.random.default_rng(0))

    assert enc.gen_calls == 1          # encoder generated
    assert spy.calls == []             # llm_client untouched
    assert next_state.turn == state.turn + 1


def test_sample_transition_uses_llm_client_when_no_encoder():
    """Without an encoder, the existing llm_client path is unchanged."""
    from verifier_anchored_rl.llm_client import EchoLLMClient
    from verifier_anchored_rl.llm_context_mdp import LLMContextMDP, MathAction, Verb

    mdp = LLMContextMDP(problem_statement="2+2", expected_answer="4",
                        llm_client=EchoLLMClient(), depth_limit=3, max_turns=4)
    state = mdp.initial_state()
    next_state, _ = mdp.sample_transition(
        state, MathAction(verb=Verb.COMMIT_ANSWER), np.random.default_rng(0))
    assert next_state.turn == state.turn + 1
```

Confirm `import numpy as np` and `import pytest` are present at the top of `tests/test_llm_context_mdp.py`; add whichever is missing.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_context_mdp.py::test_sample_transition_uses_encoder_when_provided -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'encoder'`.

- [ ] **Step 3: Add the fields**

In `src/verifier_anchored_rl/llm_context_mdp.py`, in the `LLMContextMDP` dataclass field block, after `seed_llm: bool = True` (line 980), add:

```python
    # Optional in-process encoder path (method.tex 3.4 "one forward, not
    # two"). When set, sample_transition generates verb content by reusing
    # the state prefill rather than calling llm_client.complete_json. The
    # shared prefill_cache lets the heads and generation reuse one prefill
    # per node. When None, the llm_client path is used unchanged.
    encoder: Any = field(default=None, repr=False)
    prefill_cache: Any = field(default=None, repr=False)
```

- [ ] **Step 4: Update the None-guard and the generation call**

In `sample_transition`, replace the `llm_client is None` guard (lines 1032-1037):

```python
        if self.llm_client is None:
            raise RuntimeError(
                "LLMContextMDP.sample_transition: no llm_client configured. "
                "Pass a MockLLMClient, EchoLLMClient, or "
                "OpenAICompatibleLLMClient at construction."
            )
```

with:

```python
        if self.encoder is None and self.llm_client is None:
            raise RuntimeError(
                "LLMContextMDP.sample_transition: no generation backend. "
                "Pass an encoder, or a MockLLMClient/EchoLLMClient/"
                "OpenAICompatibleLLMClient, at construction."
            )
```

Then replace the generation `try` block (lines 1062-1072):

```python
        try:
            response = self.llm_client.complete_json(
                system=_SYSTEM_PROMPT,
                user=user,
                schema=schema,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=sampling_seed,
            )
        except LLMResponseError:
            return self._masked_action(state)
```

with:

```python
        try:
            if self.encoder is not None:
                prefix = _state_prefix(state)
                prefill = (
                    self.prefill_cache.get_or_encode(prefix, self.encoder)
                    if self.prefill_cache is not None
                    else self.encoder.encode(prefix)
                )
                response, _ = self.encoder.instantiate_json(
                    prefill,
                    _verb_suffix(verb, state),
                    schema,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=sampling_seed,
                )
            else:
                response = self.llm_client.complete_json(
                    system=_SYSTEM_PROMPT,
                    user=user,
                    schema=schema,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=sampling_seed,
                )
        except LLMResponseError:
            return self._masked_action(state)
```

Note: `user = _prompt_for(verb, state)` (line 1048) is still computed above this block and is used only by the `llm_client` branch; leave it in place (the encoder branch builds its own prefix/suffix).

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_llm_context_mdp.py -q`
Expected: PASS (the two new routing tests plus all existing MDP tests, which use the unchanged `llm_client` path).

- [ ] **Step 6: Commit**

```bash
git add src/verifier_anchored_rl/llm_context_mdp.py tests/test_llm_context_mdp.py
git commit -F- <<'EOF'
Add additive encoder generation path to LLMContextMDP

When an encoder is provided, sample_transition generates verb content via
encoder.instantiate_json reusing the state prefill (one forward per node,
method.tex 3.4); the shared prefill_cache lets heads and generation share
that prefill. When encoder is None the llm_client.complete_json path is
byte-for-byte unchanged, so Mock/Echo/HTTP and all existing tests are
unaffected.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MGnZr92THoybtHaa9vfk2d
EOF
```

---

### Task 5: The outer loop (`collect_iteration`, `run_outer_loop`, `IterationStats`)

**Files:**
- Create: `src/verifier_anchored_rl/outer_loop.py`
- Modify: `src/verifier_anchored_rl/__init__.py` (export the new public names)
- Test: `tests/test_outer_loop.py`

**Interfaces:**
- Produces:
  - `outer_loop.IterationStats` (dataclass): `iteration:int`, `losses:list[dict[str,float]]`, `prefill_calls:int`, `prefill_tokens:int`, `gen_calls:int`, `gen_tokens:int`, `n_trajectories:int`, `n_decisions:int`, `n_simulations:int`, `buffer_size:int`.
  - `outer_loop.collect_iteration(problems, mdp_factory, state_actor, config, buffer, rng, *, tau, max_turns, record_sink=None) -> dict[str,int]` returning `{"n_trajectories","n_decisions","n_simulations"}`.
  - `outer_loop.run_outer_loop(problems, mdp_factory, state_actor, trainer, buffer, *, n_iterations, train_steps_per_iter, batch_size, min_buffer_size, num_simulations, c_puct, discount, tau, max_turns, rng, encoder=None, record_sink=None) -> list[IterationStats]`.
- Consumes: `mcts.MCTSConfig`, `mcts.mcts_search`, `mcts.visit_count_policy`, `mcts.root_q_estimates`; `replay.sample_from_record`; `train.Trainer`; `StateActor.prior`/`.leaf_value`/`.embed`; `available_actions`; the `mdp_factory(problem)` callable where `problem` is a `(problem_statement, expected_answer)` tuple.

- [ ] **Step 1: Write the failing end-to-end test**

Create `tests/test_outer_loop.py`:

```python
"""End-to-end test that the outer loop CLOSES (not merely runs).

Uses FakeEncoder for the heads' embeddings and EchoLLMClient for the MDP's
generation, so the whole expert-iteration loop runs on CPU with no model.
"""

from __future__ import annotations

import numpy as np
import pytest


def _build(embed_dim=16):
    from tests.fakes import FakeEncoder
    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.llm_client import EchoLLMClient
    from verifier_anchored_rl.llm_context_mdp import LLMContextMDP
    from verifier_anchored_rl.planner_actor import PrefillCache, StateActor
    from verifier_anchored_rl.replay import ReplayBuffer
    from verifier_anchored_rl.train import Trainer
    import torch

    enc = FakeEncoder(embed_dim=embed_dim)
    actor = HiddenStateActor(embed_dim=embed_dim)
    sa = StateActor(enc, actor, PrefillCache(), depth_limit=3)
    buf = ReplayBuffer(capacity=64)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-2)
    trainer = Trainer(actor, opt, buf, sync_interval=3)

    def mdp_factory(problem):
        return LLMContextMDP(problem_statement=problem[0],
                             expected_answer=problem[1],
                             llm_client=EchoLLMClient(), depth_limit=3, max_turns=4)

    return enc, actor, sa, buf, trainer, mdp_factory


def test_outer_loop_runs_and_populates_stats():
    pytest.importorskip("torch")
    from verifier_anchored_rl.outer_loop import run_outer_loop

    enc, actor, sa, buf, trainer, mdp_factory = _build()
    problems = [("2+2", "4")]
    stats = run_outer_loop(
        problems, mdp_factory, sa, trainer, buf,
        n_iterations=3, train_steps_per_iter=20, batch_size=8,
        min_buffer_size=4, num_simulations=8, c_puct=1.5, discount=1.0,
        tau=1.0, max_turns=4, rng=np.random.default_rng(0), encoder=enc)

    assert len(stats) == 3
    assert len(buf) > 0
    assert stats[-1].buffer_size == len(buf)
    assert stats[-1].prefill_calls > 0           # encoder was exercised
    assert any(s.losses for s in stats)          # training happened
    assert stats[-1].n_decisions >= 1
    assert stats[-1].n_simulations == stats[-1].n_decisions * 8
    assert trainer.global_step >= 3              # target sync fired at N=3


def test_outer_loop_closes_prior_shifts_search():
    """After training, the policy prior is no longer uniform and it changes
    the MCTS visit distribution (search -> distill -> better prior ->
    different search)."""
    pytest.importorskip("torch")
    from verifier_anchored_rl.mcts import MCTSConfig, mcts_search, visit_count_policy
    from verifier_anchored_rl.outer_loop import run_outer_loop

    enc, actor, sa, buf, trainer, mdp_factory = _build()
    problems = [("2+2", "4")]
    run_outer_loop(
        problems, mdp_factory, sa, trainer, buf,
        n_iterations=4, train_steps_per_iter=40, batch_size=8,
        min_buffer_size=4, num_simulations=8, c_puct=1.5, discount=1.0,
        tau=1.0, max_turns=4, rng=np.random.default_rng(0), encoder=enc)

    mdp = mdp_factory(problems[0])
    s0 = mdp.initial_state()
    acts = list(mdp.actions(s0))
    prior = sa.prior(s0, acts)
    assert max(prior) - min(prior) > 1e-3        # diverged from uniform

    cfg_u = MCTSConfig(num_simulations=24, c_puct=2.0, discount=1.0)
    cfg_t = MCTSConfig(num_simulations=24, c_puct=2.0, discount=1.0,
                       prior_fn=sa.prior)
    ru = mcts_search(mdp, s0, cfg_u, np.random.default_rng(1))
    rt = mcts_search(mdp, s0, cfg_t, np.random.default_rng(1))
    vu = {a.verb.value: round(v, 6) for a, v in visit_count_policy(ru).items()}
    vt = {a.verb.value: round(v, 6) for a, v in visit_count_policy(rt).items()}
    assert vu != vt                              # the learned prior steers search
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_outer_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'verifier_anchored_rl.outer_loop'`.

- [ ] **Step 3: Create `outer_loop.py`**

Create `src/verifier_anchored_rl/outer_loop.py`:

```python
"""The outer collection-and-train loop (method.tex 3.7, Algorithm 1).

Approximate Q-learning with MCTS-supplied targets, alternated with policy
distillation. MCTS is the policy-improvement operator: each iteration it
turns the current heads (online policy as prior, target-Q as bootstrap leaf
value) into stronger targets pi_MCTS and Q_MCTS, which the Trainer distills
back into the heads, which then drive better search next iteration.

This module is the orchestration only. The loss/optimizer step, priority
update, and target sync live in train.Trainer; the prior/leaf reads live in
planner_actor.StateActor; the leaf hook lives in mcts. The loop is CPU-
testable end to end with a FakeEncoder; the real run uses SharedForwardEncoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .mcts import MCTSConfig, mcts_search, root_q_estimates, visit_count_policy
from .replay import sample_from_record


@dataclass
class IterationStats:
    """Per-iteration record: the loss curve, FLOP counters, and counts."""

    iteration: int
    losses: list[dict[str, float]] = field(default_factory=list)
    prefill_calls: int = 0
    prefill_tokens: int = 0
    gen_calls: int = 0
    gen_tokens: int = 0
    n_trajectories: int = 0
    n_decisions: int = 0
    n_simulations: int = 0
    buffer_size: int = 0


def _flop_counters(encoder: Any) -> dict[str, int]:
    """Read the encoder's token/call counters (zeros if absent)."""
    return {
        "prefill_calls": int(getattr(encoder, "prefill_calls", 0)),
        "prefill_tokens": int(getattr(encoder, "prefill_tokens", 0)),
        "gen_calls": int(getattr(encoder, "gen_calls", 0)),
        "gen_tokens": int(getattr(encoder, "gen_tokens", 0)),
    }


def collect_iteration(
    problems: list[Any],
    mdp_factory: Callable[[Any], Any],
    state_actor: Any,
    config: MCTSConfig,
    buffer: Any,
    rng: np.random.Generator,
    *,
    tau: float,
    max_turns: int,
    record_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, int]:
    """Drive one trajectory per problem, adding one ReplaySample per node.

    At each visited non-terminal node: run MCTS (config carries the heads as
    prior_fn and value_fn), read pi_MCTS at temperature ``tau`` and Q_MCTS,
    embed the state, build a distillation record, and add it to the buffer.
    The trajectory is advanced by SAMPLING the next verb from pi_MCTS (vs a
    greedy argmax) so the buffer's state distribution stays diverse.
    """
    from .llm_context_mdp import available_actions

    n_traj = 0
    n_dec = 0
    for problem in problems:
        mdp = mdp_factory(problem)
        state = mdp.initial_state()
        n_traj += 1
        while not mdp.is_terminal(state) and state.turn < max_turns:
            avail = list(available_actions(state, depth_limit=mdp.depth_limit))
            if not avail:
                break
            root = mcts_search(mdp, state, config, rng)
            if not root.children:
                break
            pi = visit_count_policy(root, temperature=tau)  # {action: prob}
            q = root_q_estimates(root)                       # {action: q}
            record = {
                "embedding": state_actor.embed(state).detach().cpu().tolist(),
                "available_verbs": [a.verb.value for a in avail],
                "pi_mcts": {a.verb.value: float(p) for a, p in pi.items()},
                "q_mcts": {a.verb.value: float(v) for a, v in q.items()},
            }
            buffer.add(sample_from_record(record))
            if record_sink is not None:
                record_sink(record)
            n_dec += 1

            # Advance by sampling the next verb from pi_MCTS (method.tex 3.7
            # visit-count distribution at temperature tau).
            actions = list(pi.keys())
            probs = np.array([pi[a] for a in actions], dtype=float)
            probs = probs / probs.sum()
            action = actions[int(rng.choice(len(actions), p=probs))]
            state, _reward = mdp.sample_transition(state, action, rng)

    return {
        "n_trajectories": n_traj,
        "n_decisions": n_dec,
        "n_simulations": n_dec * config.num_simulations,
    }


def run_outer_loop(
    problems: list[Any],
    mdp_factory: Callable[[Any], Any],
    state_actor: Any,
    trainer: Any,
    buffer: Any,
    *,
    n_iterations: int,
    train_steps_per_iter: int,
    batch_size: int,
    min_buffer_size: int,
    num_simulations: int,
    c_puct: float,
    discount: float,
    tau: float,
    max_turns: int,
    rng: np.random.Generator,
    encoder: Any = None,
    record_sink: Callable[[dict[str, Any]], None] | None = None,
) -> list[IterationStats]:
    """Run Algorithm 1 for ``n_iterations``: collect, then train M steps.

    The MCTS prior is the online policy head and the leaf value is the
    target-Q head, both via ``state_actor``. Training is gated behind a
    ``min_buffer_size`` warmup. Target sync fires inside ``trainer`` on its
    own N-step schedule.
    """
    stats: list[IterationStats] = []
    for it in range(1, n_iterations + 1):
        config = MCTSConfig(
            num_simulations=num_simulations,
            c_puct=c_puct,
            discount=discount,
            prior_fn=state_actor.prior,
            value_fn=state_actor.leaf_value,
        )
        before = _flop_counters(encoder)
        counts = collect_iteration(
            problems, mdp_factory, state_actor, config, buffer, rng,
            tau=tau, max_turns=max_turns, record_sink=record_sink,
        )
        after = _flop_counters(encoder)

        losses: list[dict[str, float]] = []
        if len(buffer) >= min_buffer_size:
            losses = trainer.run(train_steps_per_iter, batch_size, rng)

        stats.append(IterationStats(
            iteration=it,
            losses=losses,
            prefill_calls=after["prefill_calls"] - before["prefill_calls"],
            prefill_tokens=after["prefill_tokens"] - before["prefill_tokens"],
            gen_calls=after["gen_calls"] - before["gen_calls"],
            gen_tokens=after["gen_tokens"] - before["gen_tokens"],
            n_trajectories=counts["n_trajectories"],
            n_decisions=counts["n_decisions"],
            n_simulations=counts["n_simulations"],
            buffer_size=len(buffer),
        ))
    return stats


__all__ = ["IterationStats", "collect_iteration", "run_outer_loop"]
```

- [ ] **Step 4: Export the new public names**

In `src/verifier_anchored_rl/__init__.py`, add an import block (next to the existing `from .mcts import ...` block near line 66) and extend `__all__`:

```python
from .outer_loop import IterationStats, collect_iteration, run_outer_loop
from .planner_actor import Encoder, PrefillCache, StateActor
```

Add these names to the `__all__` list:

```python
    "Encoder",
    "IterationStats",
    "PrefillCache",
    "StateActor",
    "collect_iteration",
    "run_outer_loop",
```

- [ ] **Step 5: Run the new tests**

Run: `pytest tests/test_outer_loop.py -v`
Expected: PASS (both the runs-and-populates-stats test and the loop-closes test).

- [ ] **Step 6: Run the full suite**

Run: `pytest -q`
Expected: PASS (all prior tests plus the new outer-loop, planner-actor, mcts, and mdp tests; warning count unchanged from baseline).

- [ ] **Step 7: Commit**

```bash
git add src/verifier_anchored_rl/outer_loop.py src/verifier_anchored_rl/__init__.py tests/test_outer_loop.py
git commit -F- <<'EOF'
Add outer collection-and-train loop (method.tex 3.7 Algorithm 1)

collect_iteration drives one trajectory per problem, runs MCTS at each node
(online policy as prior, target-Q as bootstrap leaf), reads pi_MCTS/Q_MCTS/
embed(s) into the replay buffer, and steps by sampling from pi_MCTS.
run_outer_loop alternates collection with Trainer.run(M) under a warmup
gate and records per-iteration FLOP/loss/count stats. The load-bearing
test asserts the loop CLOSES: after training the learned prior diverges
from uniform and shifts the MCTS visit distribution.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MGnZr92THoybtHaa9vfk2d
EOF
```

---

### Task 6: Frodo experiment script + journal entry

**Files:**
- Create: `experiments/run_outer_loop.py`
- Modify: `experiments/results/expert_iteration_rounds.md` (append a dated section)

**Interfaces:**
- Consumes: `SharedForwardEncoder.from_llm_client`, `TransformersLLMClient`, `HiddenStateActor`, `StateActor`, `PrefillCache`, `ReplayBuffer`, `Trainer`, `run_outer_loop`; the toy `train` problem group from `experiments/problems.py` (`PROBLEM_GROUPS`, `PROBLEMS`, `resolve_problem_selection`).
- Produces: a runnable script (GPU-bound; not unit-tested) and a journal entry recording the loss curve, FLOP counters, and the across-iteration visit-distribution shift.

- [ ] **Step 1: Write the experiment script**

Create `experiments/run_outer_loop.py`:

```python
"""Run the outer collection-and-train loop on toy problems (method.tex 3.7).

This is the GPU-bound integration of the three CPU-verified modules
(architecture/train/replay) with the planner (mcts) and the in-process
encoder (shared_forward). It runs a few outer iterations on the toy `train`
problem group, sharing one frozen model between generation and the heads'
embeddings, and prints the per-iteration loss curve and FLOP counters.

Usage::

    python experiments/run_outer_loop.py \\
        --model Qwen/Qwen2.5-Math-1.5B-Instruct \\
        --iterations 3 --num-simulations 16 --train-steps 200 \\
        --batch-size 16 --max-turns 8
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--num-simulations", type=int, default=16)
    p.add_argument("--train-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--min-buffer", type=int, default=16)
    p.add_argument("--capacity", type=int, default=2048)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--discount", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--sync-interval", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--depth-limit", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--problems", nargs="+", default=["train"])
    args = p.parse_args(argv)

    import torch

    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.llm_context_mdp import LLMContextMDP
    from verifier_anchored_rl.outer_loop import run_outer_loop
    from verifier_anchored_rl.planner_actor import PrefillCache, StateActor
    from verifier_anchored_rl.replay import ReplayBuffer
    from verifier_anchored_rl.shared_forward import SharedForwardEncoder
    from verifier_anchored_rl.train import Trainer
    from verifier_anchored_rl.transformers_client import TransformersLLMClient

    try:
        from problems import PROBLEMS, resolve_problem_selection
    except ImportError:
        from experiments.problems import PROBLEMS, resolve_problem_selection

    keys = resolve_problem_selection(args.problems)
    problems = [(PROBLEMS[k]["problem"], PROBLEMS[k]["expected"]) for k in keys]
    print(f"=== Outer loop: {len(problems)} problems, model={args.model} ===",
          flush=True)

    llm_client = TransformersLLMClient(model_name=args.model)
    encoder = SharedForwardEncoder.from_llm_client(llm_client)
    cache = PrefillCache(capacity=4096)
    actor = HiddenStateActor(embed_dim=encoder.embed_dim)
    state_actor = StateActor(encoder, actor, cache, depth_limit=args.depth_limit)
    buffer = ReplayBuffer(capacity=args.capacity)
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    trainer = Trainer(actor, opt, buffer, sync_interval=args.sync_interval)

    def mdp_factory(problem):
        return LLMContextMDP(
            problem_statement=problem[0], expected_answer=problem[1],
            llm_client=llm_client, encoder=encoder, prefill_cache=cache,
            depth_limit=args.depth_limit, max_turns=args.max_turns)

    stats = run_outer_loop(
        problems, mdp_factory, state_actor, trainer, buffer,
        n_iterations=args.iterations, train_steps_per_iter=args.train_steps,
        batch_size=args.batch_size, min_buffer_size=args.min_buffer,
        num_simulations=args.num_simulations, c_puct=args.c_puct,
        discount=args.discount, tau=args.tau, max_turns=args.max_turns,
        rng=np.random.default_rng(args.seed), encoder=encoder)

    print("\n=== Per-iteration summary ===", flush=True)
    for s in stats:
        first = s.losses[0]["total"] if s.losses else float("nan")
        last = s.losses[-1]["total"] if s.losses else float("nan")
        print(
            f"iter {s.iteration}: decisions={s.n_decisions} "
            f"sims={s.n_simulations} buffer={s.buffer_size} "
            f"prefill={s.prefill_calls} ({s.prefill_tokens} tok) "
            f"gen={s.gen_calls} ({s.gen_tokens} tok) "
            f"L_total {first:.4f} -> {last:.4f}",
            flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify the script imports and shows help (no GPU needed)**

Run: `python experiments/run_outer_loop.py --help`
Expected: argparse help text prints, exit 0 (confirms imports resolve; the actual run is GPU-bound and done on frodo separately).

- [ ] **Step 3: Append the journal entry**

Append to `experiments/results/expert_iteration_rounds.md` (do not use the em-dash glyph anywhere in the text):

```markdown

## Outer loop: collect -> train -> re-prior (2026-06-18)

The outer collection-and-train loop (method.tex 3.7, Algorithm 1) wiring
the three CPU-verified modules (architecture/train/replay) to the planner
(mcts) and the in-process encoder (shared_forward). What it adds over the
pilot's one-shot offline distillation:

* a `value_fn` leaf-evaluation hook in MCTS (additive; value_fn=None keeps
  the rollout path, so the section-10 triangulation tests are untouched).
  The leaf value is max_a' Q_target(s', a'), the bootstrap of approximate
  Q-learning, so search is now AlphaZero-style rather than pure rollout.
* a StateActor bridge (planner_actor.py) that supplies the MCTS prior
  (online policy head) and leaf value (target-Q head) plus embed(s) over
  MathState, all sharing one prefill per node via a state-prefix-keyed
  PrefillCache (the "one forward per decision" property).
* an additive encoder generation path in LLMContextMDP (one forward per
  node via encoder.instantiate_json; the llm_client path is unchanged when
  no encoder is passed).
* the loop itself (outer_loop.py): collect_iteration drives one trajectory
  per problem, samples the next verb from pi_MCTS at temperature tau, and
  adds one ReplaySample per node; run_outer_loop alternates collection with
  Trainer.run(M) under a warmup gate and records per-iteration FLOP/loss
  counts in IterationStats.

CPU-tested end to end with FakeEncoder + EchoLLMClient: the load-bearing
test asserts the loop CLOSES (after training the learned prior diverges
from uniform and shifts the MCTS visit distribution), not merely that it
runs. Full suite green.

Frodo run: `python experiments/run_outer_loop.py` (record numbers here
after the first GPU run).
```

- [ ] **Step 4: Run the full suite once more**

Run: `pytest -q`
Expected: PASS (unchanged count plus the new tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/run_outer_loop.py experiments/results/expert_iteration_rounds.md
git commit -F- <<'EOF'
Add outer-loop experiment script + journal entry

experiments/run_outer_loop.py wires the real SharedForwardEncoder +
TransformersLLMClient (one shared frozen model) to the outer loop and runs
a few iterations on the toy train problems, printing the loss curve and
FLOP counters. Journal entry documents the deliverable; GPU numbers to be
filled in after the first frodo run.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MGnZr92THoybtHaa9vfk2d
EOF
```

---

## Notes for the implementer

- `available_actions(state, depth_limit=...)` returns a list of `Verb`; `mdp.actions(state)` returns a list of `MathAction` (each `.verb` is a `Verb`). The MCTS prior receives `MathAction`s; the leaf value enumerates `Verb`s. `heads.VERB_NAME_TO_INDEX[verb.value]` maps either to the tensor column.
- `visit_count_policy(root, temperature=tau)` and `root_q_estimates(root)` both return `dict[MathAction, float]` over the root's visited children only (unvisited actions are absent), which is exactly what `encode_batch`/`sample_from_record` expect (they re-normalize and mask).
- `replay.sample_from_record` re-tensorizes `embedding` via `encode_batch`, so passing a Python list (from `.tolist()`) is correct and keeps a single code path for verb-index mapping, target normalization, and masking (DRY). The `record_sink` hook lets a caller also stream those same records to JSONL for checkpointing for free.
- Torch-dependent tests must start with `pytest.importorskip("torch")` (match the existing `tests/test_train.py` pattern).
