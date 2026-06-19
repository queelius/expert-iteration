# Observable and Resumable Outer-Loop Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Instrument the outer collection-and-train loop with periodic flushed logging, per-trajectory checkpoint and resume, the L_Q/L_pi loss split, and a unique run directory with full provenance metadata, without changing the learning algorithm.

**Architecture:** Additive callbacks on the tested core (`outer_loop.py`, `train.py`, `replay.py`) keep every existing signature and CPU path byte-identical. A new pure-Python `run_artifacts.py` module owns run-directory creation, metadata, JSONL append, and atomic checkpoint save/load (unit-testable on CPU). The harness `experiments/run_outer_loop.py` wires the callbacks to those writers and adds `--resume`.

**Tech Stack:** Python 3.12, torch (CPU-testable with the `FakeEncoder`), numpy, pytest. Persistence is plain JSON / JSONL plus `torch.save` checkpoint dicts, matching the repo's existing experiment style.

**Hard constraint:** Do NOT write the em-dash glyph in any file in this repo. A soul plugin hook rejects writes containing it. Use commas, colons, periods, or parentheses.

---

### Task 1: ReplayBuffer serialization

**Files:**
- Modify: `src/verifier_anchored_rl/replay.py` (add two methods to `ReplayBuffer`)
- Test: `tests/test_replay.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_replay.py`:

```python
def test_replay_buffer_state_dict_round_trips():
    import pytest
    pytest.importorskip("torch")
    import numpy as np
    import torch
    from verifier_anchored_rl.replay import ReplayBuffer, ReplaySample

    def _sample(seed):
        g = torch.Generator().manual_seed(seed)
        return ReplaySample(
            hidden=torch.randn(8, generator=g),
            available_mask=torch.ones(7, dtype=torch.bool),
            q_target=torch.randn(7, generator=g),
            q_target_mask=torch.ones(7, dtype=torch.bool),
            pi_target=torch.softmax(torch.randn(7, generator=g), dim=-1),
        )

    buf = ReplayBuffer(capacity=4)
    for i in range(3):
        buf.add(_sample(i), priority=float(i + 1))

    state = buf.state_dict()

    restored = ReplayBuffer(capacity=4)
    restored.load_state_dict(state)

    assert len(restored) == len(buf)
    assert restored._pos == buf._pos
    assert np.allclose(restored._priorities, buf._priorities)
    for a, b in zip(restored._samples, buf._samples):
        assert torch.equal(a.hidden, b.hidden)
        assert torch.equal(a.q_target, b.q_target)
        assert torch.equal(a.pi_target, b.pi_target)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_replay.py::test_replay_buffer_state_dict_round_trips -v`
Expected: FAIL with `AttributeError: 'ReplayBuffer' object has no attribute 'state_dict'`

- [ ] **Step 3: Write minimal implementation**

In `src/verifier_anchored_rl/replay.py`, add these two methods to `ReplayBuffer` (after `update_priorities`):

```python
    def state_dict(self) -> dict[str, Any]:
        """Serializable snapshot of the buffer (samples, priorities, ring
        position, size). Tensors are kept as-is so ``torch.save`` handles
        them; the priority array is a numpy array."""
        return {
            "samples": [vars(s) for s in self._samples],
            "priorities": self._priorities.copy(),
            "pos": self._pos,
            "size": self._size,
            "capacity": self.capacity,
            "alpha": self.alpha,
            "eps": self.eps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a snapshot produced by ``state_dict``. Capacity must
        match the buffer this is loaded into."""
        if state["capacity"] != self.capacity:
            raise ValueError(
                f"capacity mismatch: checkpoint {state['capacity']} "
                f"vs buffer {self.capacity}"
            )
        self.alpha = state["alpha"]
        self.eps = state["eps"]
        self._samples = [ReplaySample(**d) for d in state["samples"]]
        self._priorities = np.asarray(state["priorities"], dtype=np.float64).copy()
        self._pos = int(state["pos"])
        self._size = int(state["size"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_replay.py::test_replay_buffer_state_dict_round_trips -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verifier_anchored_rl/replay.py tests/test_replay.py
git commit -m "Add ReplayBuffer state_dict/load_state_dict for checkpointing

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Trainer per-step callback

**Files:**
- Modify: `src/verifier_anchored_rl/train.py` (`Trainer.run`)
- Test: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_train.py` (reuse whatever buffer/actor helper the file already has; if none, build inline as below):

```python
def test_trainer_run_fires_on_step_callback():
    import pytest
    pytest.importorskip("torch")
    import numpy as np
    import torch
    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.replay import ReplayBuffer, ReplaySample
    from verifier_anchored_rl.train import Trainer

    actor = HiddenStateActor(embed_dim=8)
    buf = ReplayBuffer(capacity=8)
    g = torch.Generator().manual_seed(0)
    for _ in range(4):
        buf.add(ReplaySample(
            hidden=torch.randn(8, generator=g),
            available_mask=torch.ones(7, dtype=torch.bool),
            q_target=torch.randn(7, generator=g),
            q_target_mask=torch.ones(7, dtype=torch.bool),
            pi_target=torch.softmax(torch.randn(7, generator=g), dim=-1),
        ))
    opt = torch.optim.Adam(actor.parameters(), lr=1e-2)
    trainer = Trainer(actor, opt, buf, sync_interval=100)

    seen = []
    trainer.run(5, batch_size=4, rng=np.random.default_rng(0),
                on_step=lambda step, losses: seen.append((step, losses)))

    assert len(seen) == 5
    assert seen[0][0] == 1 and seen[-1][0] == 5      # 1-based global step
    assert set(seen[0][1].keys()) == {"q", "pi", "total"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train.py::test_trainer_run_fires_on_step_callback -v`
Expected: FAIL with `TypeError: run() got an unexpected keyword argument 'on_step'`

- [ ] **Step 3: Write minimal implementation**

In `src/verifier_anchored_rl/train.py`, replace `Trainer.step` and `Trainer.run` with:

```python
    def step(self, batch_size: int, rng: np.random.Generator) -> dict[str, float]:
        batch, indices = self.buffer.sample(batch_size, rng)
        losses, td = train_step(
            self.actor, self.optimizer, batch, lambda_pi=self.lambda_pi
        )
        self.buffer.update_priorities(indices, td)
        self.global_step += 1
        if self.global_step % self.sync_interval == 0:
            self.actor.sync_target()
        return losses

    def run(
        self,
        num_steps: int,
        batch_size: int,
        rng: np.random.Generator,
        on_step: Any = None,
    ) -> list[dict[str, float]]:
        out = []
        for _ in range(num_steps):
            losses = self.step(batch_size, rng)
            out.append(losses)
            if on_step is not None:
                on_step(self.global_step, losses)
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_train.py::test_trainer_run_fires_on_step_callback -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verifier_anchored_rl/train.py tests/test_train.py
git commit -m "Add optional on_step callback to Trainer.run

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: IterationStats loss split

**Files:**
- Modify: `src/verifier_anchored_rl/outer_loop.py` (`IterationStats` and a new `loss_summary` helper)
- Test: `tests/test_outer_loop.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_outer_loop.py`:

```python
def test_loss_summary_extracts_q_pi_total_split():
    from verifier_anchored_rl.outer_loop import loss_summary

    losses = [
        {"q": 1.0, "pi": 2.0, "total": 3.0},
        {"q": 0.5, "pi": 1.5, "total": 2.0},
        {"q": 0.1, "pi": 1.4, "total": 1.5},
    ]
    s = loss_summary(losses)
    assert s["q_first"] == 1.0 and s["q_last"] == 0.1
    assert abs(s["q_mean"] - (1.6 / 3)) < 1e-9
    assert s["pi_first"] == 2.0 and s["pi_last"] == 1.4
    assert s["total_first"] == 3.0 and s["total_last"] == 1.5


def test_loss_summary_handles_empty():
    from verifier_anchored_rl.outer_loop import loss_summary
    import math

    s = loss_summary([])
    assert math.isnan(s["q_first"]) and math.isnan(s["total_last"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_outer_loop.py::test_loss_summary_extracts_q_pi_total_split -v`
Expected: FAIL with `ImportError: cannot import name 'loss_summary'`

- [ ] **Step 3: Write minimal implementation**

In `src/verifier_anchored_rl/outer_loop.py`, add the helper near the top (after the imports) and extend `IterationStats`:

```python
def loss_summary(losses: list[dict[str, float]]) -> dict[str, float]:
    """First, last, and mean of each loss component over a list of per-step
    loss dicts. All NaN if the list is empty (training did not run)."""
    nan = float("nan")
    if not losses:
        return {
            "q_first": nan, "q_last": nan, "q_mean": nan,
            "pi_first": nan, "pi_last": nan, "pi_mean": nan,
            "total_first": nan, "total_last": nan, "total_mean": nan,
        }

    def col(key):
        return [d[key] for d in losses]

    out = {}
    for key in ("q", "pi", "total"):
        vals = col(key)
        out[f"{key}_first"] = vals[0]
        out[f"{key}_last"] = vals[-1]
        out[f"{key}_mean"] = sum(vals) / len(vals)
    return out
```

Add these fields to `IterationStats` (after `buffer_size`):

```python
    wall_seconds: float = 0.0
    q_first: float = float("nan")
    q_last: float = float("nan")
    pi_first: float = float("nan")
    pi_last: float = float("nan")
    total_first: float = float("nan")
    total_last: float = float("nan")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_outer_loop.py::test_loss_summary_extracts_q_pi_total_split tests/test_outer_loop.py::test_loss_summary_handles_empty -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/verifier_anchored_rl/outer_loop.py tests/test_outer_loop.py
git commit -m "Add loss_summary helper and loss-split fields to IterationStats

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: collect_iteration callbacks and resume offset

**Files:**
- Modify: `src/verifier_anchored_rl/outer_loop.py` (`collect_iteration`)
- Test: `tests/test_outer_loop.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_outer_loop.py` (uses the existing `_build` helper):

```python
def test_collect_iteration_fires_decision_and_trajectory_callbacks():
    import pytest
    pytest.importorskip("torch")
    import numpy as np
    from verifier_anchored_rl.mcts import MCTSConfig
    from verifier_anchored_rl.outer_loop import collect_iteration

    enc, actor, sa, buf, trainer, mdp_factory = _build()
    config = MCTSConfig(num_simulations=6, c_puct=1.5, discount=1.0,
                        prior_fn=sa.prior, value_fn=sa.leaf_value)

    decisions = []
    trajectories = []
    counts = collect_iteration(
        [("2+2", "4"), ("3+3", "6")], mdp_factory, sa, config, buf,
        np.random.default_rng(0), tau=1.0, max_turns=4,
        on_decision=lambda info: decisions.append(info),
        on_trajectory_end=lambda done: trajectories.append(done),
    )

    assert len(decisions) == counts["n_decisions"]
    assert decisions[0]["iteration_problem_index"] == 0
    assert "chosen_verb" in decisions[0]
    assert "gen_tokens" in decisions[0]
    assert trajectories == [1, 2]      # one callback per completed trajectory


def test_collect_iteration_start_problem_skips_done_trajectories():
    import pytest
    pytest.importorskip("torch")
    import numpy as np
    from verifier_anchored_rl.mcts import MCTSConfig
    from verifier_anchored_rl.outer_loop import collect_iteration

    enc, actor, sa, buf, trainer, mdp_factory = _build()
    config = MCTSConfig(num_simulations=6, c_puct=1.5, discount=1.0,
                        prior_fn=sa.prior, value_fn=sa.leaf_value)

    trajectories = []
    collect_iteration(
        [("2+2", "4"), ("3+3", "6")], mdp_factory, sa, config, buf,
        np.random.default_rng(0), tau=1.0, max_turns=4, start_problem=1,
        on_trajectory_end=lambda done: trajectories.append(done),
    )
    # problem index 0 skipped, only the second trajectory runs
    assert trajectories == [2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_outer_loop.py::test_collect_iteration_fires_decision_and_trajectory_callbacks -v`
Expected: FAIL with `TypeError: collect_iteration() got an unexpected keyword argument 'on_decision'`

- [ ] **Step 3: Write minimal implementation**

In `src/verifier_anchored_rl/outer_loop.py`, replace `collect_iteration` with the version below. Changes: new `on_decision`, `on_trajectory_end`, `start_problem`, and an `iteration_index` keyword for labeling; per-decision callback with encoder counters; per-trajectory callback. The collection math is unchanged.

```python
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
    on_decision: Callable[[dict[str, Any]], None] | None = None,
    on_trajectory_end: Callable[[int], None] | None = None,
    start_problem: int = 0,
    iteration_index: int = 0,
) -> dict[str, int]:
    """Drive one trajectory per problem, adding one ReplaySample per node.

    See module docstring. ``start_problem`` skips the first k problems (for
    resume); ``on_decision`` fires per decision with a labeled info dict;
    ``on_trajectory_end(done)`` fires after each completed trajectory with the
    1-based count of trajectories finished this iteration.
    """
    from .llm_context_mdp import available_actions

    n_traj = 0
    n_dec = 0
    enc = getattr(state_actor, "encoder", None)
    for p_idx, problem in enumerate(problems):
        if p_idx < start_problem:
            continue
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
            pi = visit_count_policy(root, temperature=tau)
            q = root_q_estimates(root)
            record = {
                "embedding": state_actor.embed(state).detach().cpu().tolist(),
                "available_verbs": [a.value for a in avail],
                "pi_mcts": {a.verb.value: float(p) for a, p in pi.items()},
                "q_mcts": {a.verb.value: float(v) for a, v in q.items()},
            }
            buffer.add(sample_from_record(record))
            if record_sink is not None:
                record_sink(record)
            n_dec += 1

            actions = list(pi.keys())
            probs = np.array([pi[a] for a in actions], dtype=float)
            probs = probs / probs.sum()
            chosen = actions[int(rng.choice(len(actions), p=probs))]

            if on_decision is not None:
                top = max(pi.items(), key=lambda kv: kv[1])
                on_decision({
                    "iteration": iteration_index,
                    "iteration_problem_index": p_idx,
                    "turn": int(state.turn),
                    "chosen_verb": chosen.verb.value,
                    "pi_argmax_verb": top[0].verb.value,
                    "pi_argmax_prob": float(top[1]),
                    "q_chosen": float(q.get(chosen, float("nan"))),
                    "prefill_calls": int(getattr(enc, "prefill_calls", 0)),
                    "prefill_tokens": int(getattr(enc, "prefill_tokens", 0)),
                    "gen_calls": int(getattr(enc, "gen_calls", 0)),
                    "gen_tokens": int(getattr(enc, "gen_tokens", 0)),
                })

            state, _reward = mdp.sample_transition(state, chosen, rng)

        if on_trajectory_end is not None:
            on_trajectory_end(n_traj)

    return {
        "n_trajectories": n_traj,
        "n_decisions": n_dec,
        "n_simulations": n_dec * config.num_simulations,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_outer_loop.py::test_collect_iteration_fires_decision_and_trajectory_callbacks tests/test_outer_loop.py::test_collect_iteration_start_problem_skips_done_trajectories -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the existing outer-loop tests to confirm no regression**

Run: `python -m pytest tests/test_outer_loop.py -v`
Expected: PASS (all, including `test_outer_loop_runs_and_populates_stats` and `test_outer_loop_closes_prior_shifts_search`)

- [ ] **Step 6: Commit**

```bash
git add src/verifier_anchored_rl/outer_loop.py tests/test_outer_loop.py
git commit -m "Add decision/trajectory callbacks and start_problem to collect_iteration

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: run_outer_loop callbacks, checkpoint hook, and resume

**Files:**
- Modify: `src/verifier_anchored_rl/outer_loop.py` (`run_outer_loop`)
- Test: `tests/test_outer_loop.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_outer_loop.py`:

```python
def test_run_outer_loop_invokes_train_and_checkpoint_callbacks():
    import pytest
    pytest.importorskip("torch")
    import numpy as np
    from verifier_anchored_rl.outer_loop import run_outer_loop

    enc, actor, sa, buf, trainer, mdp_factory = _build()
    train_steps = []
    checkpoints = []
    stats = run_outer_loop(
        [("2+2", "4")], mdp_factory, sa, trainer, buf,
        n_iterations=2, train_steps_per_iter=10, batch_size=8,
        min_buffer_size=2, num_simulations=6, c_puct=1.5, discount=1.0,
        tau=1.0, max_turns=4, rng=np.random.default_rng(0), encoder=enc,
        on_train_step=lambda info: train_steps.append(info),
        save_checkpoint=lambda state: checkpoints.append(
            (state["phase"], state["iteration"], state["problems_done_this_iter"])),
        log_every=5,
    )

    # train-step callback fired on the log_every cadence for both iterations
    assert all(set(t.keys()) >= {"global_step", "q", "pi", "total"} for t in train_steps)
    # a checkpoint after each trajectory (collecting) and after each train phase
    assert ("collecting", 1, 1) in checkpoints
    assert ("training", 1, 1) in checkpoints
    assert ("training", 2, 1) in checkpoints
    # loss split landed on the stats
    assert stats[-1].q_first == stats[-1].q_first   # not NaN: training ran
    assert stats[-1].wall_seconds >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_outer_loop.py::test_run_outer_loop_invokes_train_and_checkpoint_callbacks -v`
Expected: FAIL with `TypeError: run_outer_loop() got an unexpected keyword argument 'on_train_step'`

- [ ] **Step 3: Write minimal implementation**

In `src/verifier_anchored_rl/outer_loop.py`, add `import time` at the top with the other imports, then replace `run_outer_loop` with:

```python
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
    on_decision: Callable[[dict[str, Any]], None] | None = None,
    on_train_step: Callable[[dict[str, Any]], None] | None = None,
    on_iteration_end: Callable[["IterationStats"], None] | None = None,
    save_checkpoint: Callable[[dict[str, Any]], None] | None = None,
    log_every: int = 25,
    start_iteration: int = 1,
    start_problem: int = 0,
) -> list[IterationStats]:
    """Run Algorithm 1 for ``n_iterations``: collect, then train M steps.

    Observability and resume hooks are all optional and default-off, so the
    original behavior is unchanged when none are passed. ``start_iteration`` /
    ``start_problem`` resume a partially completed run (1-based iteration index;
    ``start_problem`` only applies to ``start_iteration``).
    """
    stats: list[IterationStats] = []
    for it in range(start_iteration, n_iterations + 1):
        t0 = time.monotonic()
        config = MCTSConfig(
            num_simulations=num_simulations,
            c_puct=c_puct,
            discount=discount,
            prior_fn=state_actor.prior,
            value_fn=state_actor.leaf_value,
        )
        before = _flop_counters(encoder)
        state_actor.cache.clear()

        def _on_trajectory_end(done: int) -> None:
            if save_checkpoint is not None:
                save_checkpoint({
                    "phase": "collecting",
                    "iteration": it,
                    "problems_done_this_iter": done,
                })

        counts = collect_iteration(
            problems, mdp_factory, state_actor, config, buffer, rng,
            tau=tau, max_turns=max_turns, record_sink=record_sink,
            on_decision=on_decision, on_trajectory_end=_on_trajectory_end,
            start_problem=(start_problem if it == start_iteration else 0),
            iteration_index=it,
        )
        after = _flop_counters(encoder)

        losses: list[dict[str, float]] = []
        if len(buffer) >= min_buffer_size:
            def _on_step(global_step: int, step_losses: dict[str, float]) -> None:
                if on_train_step is not None and (
                    global_step % log_every == 0 or global_step == 1
                ):
                    on_train_step({"global_step": global_step, **step_losses})
            losses = trainer.run(
                train_steps_per_iter, batch_size, rng, on_step=_on_step
            )

        if save_checkpoint is not None:
            save_checkpoint({
                "phase": "training",
                "iteration": it,
                "problems_done_this_iter": len(problems),
            })

        summary = loss_summary(losses)
        s = IterationStats(
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
            wall_seconds=time.monotonic() - t0,
            q_first=summary["q_first"], q_last=summary["q_last"],
            pi_first=summary["pi_first"], pi_last=summary["pi_last"],
            total_first=summary["total_first"], total_last=summary["total_last"],
        )
        stats.append(s)
        if on_iteration_end is not None:
            on_iteration_end(s)
    return stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_outer_loop.py::test_run_outer_loop_invokes_train_and_checkpoint_callbacks -v`
Expected: PASS

- [ ] **Step 5: Run the full outer-loop and train test files to confirm no regression**

Run: `python -m pytest tests/test_outer_loop.py tests/test_train.py -v`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add src/verifier_anchored_rl/outer_loop.py tests/test_outer_loop.py
git commit -m "Add observability and resume hooks to run_outer_loop

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: run_artifacts module (run dir, metadata, JSONL, atomic checkpoint)

**Files:**
- Create: `src/verifier_anchored_rl/run_artifacts.py`
- Test: `tests/test_run_artifacts.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_artifacts.py`:

```python
"""Tests for run-directory artifacts: metadata, JSONL append, atomic
checkpoint save/load. Pure filesystem, no GPU."""

from __future__ import annotations

import json

import pytest


def test_create_run_dir_writes_metadata(tmp_path):
    from verifier_anchored_rl.run_artifacts import create_run_dir

    run_dir = create_run_dir(
        root=str(tmp_path), tag="smoke",
        metadata={"model": "fake", "seed": 0},
        timestamp="20260619T000000Z",
    )
    meta = json.loads((tmp_path / run_dir.name / "metadata.json").read_text())
    assert meta["model"] == "fake"
    assert meta["status"] == "running"
    assert "20260619T000000Z" in run_dir.name
    assert "smoke" in run_dir.name


def test_append_jsonl_appends_one_line_per_call(tmp_path):
    from verifier_anchored_rl.run_artifacts import append_jsonl

    path = tmp_path / "metrics.jsonl"
    append_jsonl(str(path), {"iteration": 1})
    append_jsonl(str(path), {"iteration": 2})
    lines = path.read_text().strip().splitlines()
    assert [json.loads(x)["iteration"] for x in lines] == [1, 2]


def test_atomic_save_load_checkpoint_round_trips(tmp_path):
    pytest.importorskip("torch")
    import torch
    from verifier_anchored_rl.run_artifacts import (
        save_checkpoint_atomic, load_checkpoint,
    )

    path = tmp_path / "checkpoint.pt"
    save_checkpoint_atomic(str(path), {"x": torch.tensor([1.0, 2.0]), "step": 7})
    assert not (tmp_path / "checkpoint.pt.tmp").exists()   # tmp cleaned up
    loaded = load_checkpoint(str(path))
    assert loaded["step"] == 7
    assert torch.equal(loaded["x"], torch.tensor([1.0, 2.0]))


def test_update_status_rewrites_metadata(tmp_path):
    from verifier_anchored_rl.run_artifacts import create_run_dir, update_status

    run_dir = create_run_dir(
        root=str(tmp_path), tag=None, metadata={"model": "fake"},
        timestamp="20260619T000000Z",
    )
    update_status(run_dir, "completed")
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["status"] == "completed"
    assert "end_time" in meta
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_artifacts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'verifier_anchored_rl.run_artifacts'`

- [ ] **Step 3: Write minimal implementation**

Create `src/verifier_anchored_rl/run_artifacts.py`:

```python
"""Run-directory artifacts for long experiment runs: a unique provenance
directory, metadata, append-only JSONL, and an atomic checkpoint save/load.

Pure filesystem and JSON plus ``torch.save``; no model and no GPU, so it is
unit-testable on any machine. The harness (``experiments/run_outer_loop.py``)
composes these with the outer-loop callbacks to make a run observable and
resumable. Motivated by the 14.7 hour first GPU run that left only three
summary lines (see experiments/results/expert_iteration_rounds.md)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    """Compact UTC timestamp like ``20260619T134501Z`` for run-dir names."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def create_run_dir(
    *,
    root: str,
    tag: str | None,
    metadata: dict[str, Any],
    timestamp: str | None = None,
) -> Path:
    """Create ``<root>/<timestamp>[-<tag>]/`` and write ``metadata.json`` with
    status ``running`` and a ``start_time``. Returns the run directory path."""
    ts = timestamp or utc_timestamp()
    name = f"{ts}-{tag}" if tag else ts
    run_dir = Path(root) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata)
    meta.setdefault("start_time", datetime.now(timezone.utc).isoformat())
    meta["status"] = "running"
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    return run_dir


def update_status(run_dir: Path, status: str) -> None:
    """Rewrite ``metadata.json`` with a new status and an ``end_time``."""
    path = Path(run_dir) / "metadata.json"
    meta = json.loads(path.read_text())
    meta["status"] = status
    meta["end_time"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(meta, indent=2, default=str))


def append_jsonl(path: str, obj: dict[str, Any]) -> None:
    """Append one JSON object as a line. Flushed so a crash keeps prior lines."""
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")
        f.flush()


def save_checkpoint_atomic(path: str, state: dict[str, Any]) -> None:
    """Write a checkpoint to ``path.tmp`` then ``os.replace`` it into place, so
    a crash mid-write never leaves a torn checkpoint at ``path``."""
    import torch

    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> dict[str, Any]:
    """Load a checkpoint dict (CPU map location). Not ``weights_only``: the
    checkpoint carries optimizer and buffer state, not only tensors."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


__all__ = [
    "append_jsonl",
    "create_run_dir",
    "load_checkpoint",
    "save_checkpoint_atomic",
    "update_status",
    "utc_timestamp",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_run_artifacts.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/verifier_anchored_rl/run_artifacts.py tests/test_run_artifacts.py
git commit -m "Add run_artifacts: run dir, metadata, JSONL, atomic checkpoint

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Load-bearing resume-equivalence test

**Files:**
- Test: `tests/test_outer_loop.py`

This test proves the resume contract end to end on CPU: an uninterrupted
2-iteration run and a run that stops after iteration 1, checkpoints, and
resumes from that checkpoint produce identical final actor weights and
identical buffer contents. It uses `run_artifacts` for save/load and
`ReplayBuffer.state_dict` from Task 1.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_outer_loop.py`:

```python
def _checkpoint_state(actor, opt, buf, trainer, rng, phase, iteration, done):
    return {
        "phase": phase,
        "iteration": iteration,
        "problems_done_this_iter": done,
        "actor_state": actor.state_dict(),
        "optimizer_state": opt.state_dict(),
        "buffer": buf.state_dict(),
        "global_step": trainer.global_step,
        "numpy_rng_state": rng.bit_generator.state,
    }


def test_resume_from_iteration_checkpoint_matches_uninterrupted_run(tmp_path):
    import pytest
    pytest.importorskip("torch")
    import numpy as np
    import torch
    from verifier_anchored_rl.outer_loop import run_outer_loop
    from verifier_anchored_rl.run_artifacts import (
        save_checkpoint_atomic, load_checkpoint,
    )

    problems = [("2+2", "4")]
    common = dict(
        n_iterations=2, train_steps_per_iter=10, batch_size=8,
        min_buffer_size=2, num_simulations=6, c_puct=1.5, discount=1.0,
        tau=1.0, max_turns=4,
    )

    # Uninterrupted reference run.
    enc, actor, sa, buf, trainer, mdp_factory = _build()
    run_outer_loop(problems, mdp_factory, sa, trainer, buf,
                   rng=np.random.default_rng(0), encoder=enc, **common)
    ref_weights = {k: v.clone() for k, v in actor.state_dict().items()}

    # Interruptible run: stop after iteration 1, save a "training"-phase
    # checkpoint, then build a fresh stack and resume from it.
    enc2, actor2, sa2, buf2, trainer2, mdp2 = _build()
    rng2 = np.random.default_rng(0)
    ckpt_path = str(tmp_path / "checkpoint.pt")

    def _save(state):
        if state["phase"] == "training" and state["iteration"] == 1:
            save_checkpoint_atomic(ckpt_path, _checkpoint_state(
                actor2, trainer2.optimizer, buf2, trainer2, rng2,
                "training", 1, len(problems)))

    run_outer_loop(problems, mdp2, sa2, trainer2, buf2,
                   rng=rng2, encoder=enc2, n_iterations=1,
                   train_steps_per_iter=common["train_steps_per_iter"],
                   batch_size=common["batch_size"],
                   min_buffer_size=common["min_buffer_size"],
                   num_simulations=common["num_simulations"],
                   c_puct=common["c_puct"], discount=common["discount"],
                   tau=common["tau"], max_turns=common["max_turns"],
                   save_checkpoint=_save)

    # Fresh stack, restore, resume iteration 2.
    enc3, actor3, sa3, buf3, trainer3, mdp3 = _build()
    ckpt = load_checkpoint(ckpt_path)
    actor3.load_state_dict(ckpt["actor_state"])
    trainer3.optimizer.load_state_dict(ckpt["optimizer_state"])
    buf3.load_state_dict(ckpt["buffer"])
    trainer3.global_step = ckpt["global_step"]
    rng3 = np.random.default_rng(0)
    rng3.bit_generator.state = ckpt["numpy_rng_state"]

    run_outer_loop(problems, mdp3, sa3, trainer3, buf3,
                   rng=rng3, encoder=enc3, start_iteration=2, **common)

    resumed_weights = actor3.state_dict()
    for k in ref_weights:
        assert torch.allclose(ref_weights[k], resumed_weights[k], atol=1e-6), k
```

- [ ] **Step 2: Run test to verify it fails or passes, and understand why**

Run: `python -m pytest tests/test_outer_loop.py::test_resume_from_iteration_checkpoint_matches_uninterrupted_run -v`
Expected: PASS if Tasks 1, 5, 6 are complete. If it FAILS on weight mismatch, the cause is RNG or buffer state not being captured at exactly the post-training point; verify the checkpoint is saved on the `("training", 1)` callback (after `trainer.run`) and that `rng2.bit_generator.state` is read at save time (it is captured by reference inside `_checkpoint_state`, so it reflects the RNG at the moment `_save` runs, which is correct).

- [ ] **Step 3: If failing, fix the captured-state timing (no production change expected)**

If the mismatch persists, the most likely cause is that `_save` reads `trainer2.global_step` or `buf2` after further mutation. Confirm `save_checkpoint` fires synchronously inside `run_outer_loop` immediately after `trainer.run` returns and before the next iteration, which Task 5 guarantees. No production code change should be needed; if one is, it belongs in Task 5 and that task's tests must be updated.

- [ ] **Step 4: Commit**

```bash
git add tests/test_outer_loop.py
git commit -m "Prove resume-from-checkpoint matches the uninterrupted run

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Rewrite the harness to use run dir, logging, checkpoint, and resume

**Files:**
- Modify: `experiments/run_outer_loop.py`
- Test: `tests/test_run_outer_loop_harness.py` (light test of the run-dir wiring with a monkeypatched loop)

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_outer_loop_harness.py`:

```python
"""Light harness test: a real run dir with metadata, progress log, and a
metrics line is produced. The outer loop itself is monkeypatched to a stub so
this stays CPU-only and fast (the real loop is covered in test_outer_loop)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def test_harness_builds_run_dir_and_writes_metrics(tmp_path, monkeypatch):
    import experiments.run_outer_loop as harness
    from verifier_anchored_rl.outer_loop import IterationStats

    # Stub the heavy pieces: model client, encoder, and the loop.
    monkeypatch.setattr(harness, "_build_stack", lambda args: ("ENC", "SA", "TR", "BUF", "ACTOR", "OPT"))

    def fake_loop(*a, **kw):
        kw["on_iteration_end"](IterationStats(iteration=1, n_decisions=3,
                                              n_simulations=18, buffer_size=3,
                                              total_first=2.0, total_last=1.5))
        kw["save_checkpoint"]({"phase": "training", "iteration": 1,
                               "problems_done_this_iter": 1})
        return [IterationStats(iteration=1, n_decisions=3, n_simulations=18,
                               buffer_size=3, total_first=2.0, total_last=1.5)]

    monkeypatch.setattr(harness, "run_outer_loop", fake_loop)
    monkeypatch.setattr(harness, "_save_checkpoint_for", lambda *a, **k: (lambda state: None))

    run_root = tmp_path / "runs"
    rc = harness.main([
        "--iterations", "1", "--problems", "multiply_small",
        "--run-root", str(run_root), "--tag", "unit",
    ])
    assert rc == 0
    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    rd = run_dirs[0]
    assert (rd / "metadata.json").exists()
    assert (rd / "metrics.jsonl").exists()
    meta = json.loads((rd / "metadata.json").read_text())
    assert meta["status"] == "completed"
    metrics = (rd / "metrics.jsonl").read_text().strip().splitlines()
    assert json.loads(metrics[0])["iteration"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_outer_loop_harness.py -v`
Expected: FAIL (the harness has no `_build_stack`, `_save_checkpoint_for`, `--run-root`, `--tag`, or `--resume`, and `main` does not build a run dir).

- [ ] **Step 3: Write the implementation**

Rewrite `experiments/run_outer_loop.py` as below. It factors the model/stack construction into `_build_stack` (so the test can stub it), builds a run dir, gathers metadata, wires a flushed logger and the JSONL writers and the checkpoint saver, and supports `--resume`.

```python
"""Run the outer collection-and-train loop on toy problems (method.tex 3.7),
with full observability and resume. See
docs/superpowers/specs/2026-06-19-observable-resumable-outer-loop-design.md.

Each run gets a unique directory under --run-root with metadata.json,
progress.log (flushed, tee'd to stdout), records.jsonl, metrics.jsonl, a
per-trajectory checkpoint.pt, and a summary.md rewritten each iteration.
--resume <run-dir> continues a crashed run from its checkpoint."""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np

from verifier_anchored_rl.outer_loop import run_outer_loop
from verifier_anchored_rl.run_artifacts import (
    append_jsonl,
    create_run_dir,
    load_checkpoint,
    save_checkpoint_atomic,
    update_status,
)


def _git_sha() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip())
        return sha, dirty
    except Exception:
        return "nogit", False


def _gpu_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {"gpu": torch.cuda.get_device_name(0),
                    "cuda": torch.version.cuda}
    except Exception:
        pass
    return {"gpu": "cpu", "cuda": None}


def _build_stack(args):
    """Construct (encoder, state_actor, trainer, buffer, actor, optimizer).
    Heavy: loads the frozen model. Factored out so tests can stub it."""
    import torch

    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.llm_context_mdp import LLMContextMDP
    from verifier_anchored_rl.planner_actor import PrefillCache, StateActor
    from verifier_anchored_rl.replay import ReplayBuffer
    from verifier_anchored_rl.shared_forward import SharedForwardEncoder
    from verifier_anchored_rl.train import Trainer
    from verifier_anchored_rl.transformers_client import TransformersLLMClient

    llm_client = TransformersLLMClient(model_name=args.model)
    encoder = SharedForwardEncoder.from_llm_client(llm_client)
    cache = PrefillCache(capacity=4096)
    actor = HiddenStateActor(embed_dim=encoder.embed_dim)
    state_actor = StateActor(encoder, actor, cache, depth_limit=args.depth_limit)
    buffer = ReplayBuffer(capacity=args.capacity)
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    trainer = Trainer(actor, opt, buffer, sync_interval=args.sync_interval)
    return encoder, state_actor, trainer, buffer, actor, opt, llm_client, cache


def _save_checkpoint_for(run_dir, actor, opt, buffer, trainer, rng):
    """Return a save_checkpoint(state) closure that assembles the full
    checkpoint dict from the live objects and writes it atomically."""
    ckpt_path = str(Path(run_dir) / "checkpoint.pt")

    def _save(state: dict) -> None:
        full = {
            **state,
            "actor_state": actor.state_dict(),
            "optimizer_state": opt.state_dict(),
            "buffer": buffer.state_dict(),
            "global_step": trainer.global_step,
            "numpy_rng_state": rng.bit_generator.state,
        }
        save_checkpoint_atomic(ckpt_path, full)

    return _save


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
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--problems", nargs="+", default=["train"])
    p.add_argument("--run-root", default="experiments/results/runs")
    p.add_argument("--tag", default=None)
    p.add_argument("--resume", default=None,
                   help="path to an existing run dir to resume")
    args = p.parse_args(argv)

    try:
        from problems import PROBLEMS, resolve_problem_selection
    except ImportError:
        from experiments.problems import PROBLEMS, resolve_problem_selection

    keys = resolve_problem_selection(args.problems)
    problems = [(PROBLEMS[k]["problem"], PROBLEMS[k]["expected"]) for k in keys]

    (encoder, state_actor, trainer, buffer, actor, opt,
     _llm, _cache) = _build_stack(args)
    rng = np.random.default_rng(args.seed)

    start_iteration, start_problem = 1, 0
    if args.resume:
        run_dir = Path(args.resume)
        ckpt = load_checkpoint(str(run_dir / "checkpoint.pt"))
        actor.load_state_dict(ckpt["actor_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        buffer.load_state_dict(ckpt["buffer"])
        trainer.global_step = ckpt["global_step"]
        rng.bit_generator.state = ckpt["numpy_rng_state"]
        if ckpt["phase"] == "collecting":
            start_iteration = ckpt["iteration"]
            start_problem = ckpt["problems_done_this_iter"]
        else:  # training phase done for ckpt["iteration"]
            start_iteration = ckpt["iteration"] + 1
            start_problem = 0
    else:
        meta = {
            "args": vars(args), "hostname": socket.gethostname(),
            "python": platform.python_version(), **_gpu_info(),
            "model": args.model, "seed": args.seed,
        }
        sha, dirty = _git_sha()
        meta["git_sha"], meta["git_dirty"] = sha, dirty
        try:
            import torch, transformers
            meta["torch"], meta["transformers"] = torch.__version__, transformers.__version__
            meta["dtype"] = str(getattr(encoder.model, "dtype", "unknown"))
            meta["device"] = str(getattr(encoder.model, "device", "unknown"))
        except Exception:
            pass
        run_dir = create_run_dir(root=args.run_root, tag=args.tag, metadata=meta)

    log_path = run_dir / "progress.log"
    records_path = str(run_dir / "records.jsonl")
    metrics_path = str(run_dir / "metrics.jsonl")
    summary_path = run_dir / "summary.md"
    summaries = []

    def log(msg: str) -> None:
        line = msg if msg.endswith("\n") else msg + "\n"
        with open(log_path, "a") as f:
            f.write(line)
            f.flush()
        sys.stdout.write(line)
        sys.stdout.flush()

    def on_decision(info):
        log(f"iter {info['iteration']} prob {info['iteration_problem_index']} "
            f"turn {info['turn']}: verb={info['chosen_verb']} "
            f"pi_argmax={info['pi_argmax_verb']}({info['pi_argmax_prob']:.2f}) "
            f"gen={info['gen_calls']} ({info['gen_tokens']} tok) "
            f"prefill={info['prefill_calls']} ({info['prefill_tokens']} tok)")

    def on_train_step(info):
        log(f"  train step {info['global_step']}: "
            f"L_Q {info['q']:.4f} L_pi {info['pi']:.4f} L_total {info['total']:.4f}")

    def on_iteration_end(s):
        row = {
            "iteration": s.iteration, "decisions": s.n_decisions,
            "sims": s.n_simulations, "buffer": s.buffer_size,
            "prefill_calls": s.prefill_calls, "prefill_tokens": s.prefill_tokens,
            "gen_calls": s.gen_calls, "gen_tokens": s.gen_tokens,
            "L_Q_first": s.q_first, "L_Q_last": s.q_last,
            "L_pi_first": s.pi_first, "L_pi_last": s.pi_last,
            "L_total_first": s.total_first, "L_total_last": s.total_last,
            "wall_seconds": s.wall_seconds,
        }
        append_jsonl(metrics_path, row)
        summaries.append(row)
        lines = ["# Run summary", "", "| iter | decisions | buffer | gen tok | L_Q first->last | L_total first->last |",
                 "|---|---:|---:|---:|---|---|"]
        for r in summaries:
            lines.append(
                f"| {r['iteration']} | {r['decisions']} | {r['buffer']} | "
                f"{r['gen_tokens']} | {r['L_Q_first']:.4f} to {r['L_Q_last']:.4f} | "
                f"{r['L_total_first']:.4f} to {r['L_total_last']:.4f} |")
        summary_path.write_text("\n".join(lines) + "\n")
        log(f"=== iter {s.iteration} done in {s.wall_seconds:.0f}s: "
            f"L_total {s.total_first:.4f} to {s.total_last:.4f} ===")

    save_ckpt = _save_checkpoint_for(run_dir, actor, opt, buffer, trainer, rng)

    log(f"=== Outer loop: {len(problems)} problems, model={args.model}, "
        f"run_dir={run_dir} ===")
    try:
        run_outer_loop(
            problems, lambda problem: _mdp_factory(problem, args, encoder, state_actor),
            state_actor, trainer, buffer,
            n_iterations=args.iterations, train_steps_per_iter=args.train_steps,
            batch_size=args.batch_size, min_buffer_size=args.min_buffer,
            num_simulations=args.num_simulations, c_puct=args.c_puct,
            discount=args.discount, tau=args.tau, max_turns=args.max_turns,
            rng=rng, encoder=encoder,
            record_sink=lambda rec: append_jsonl(records_path, rec),
            on_decision=on_decision, on_train_step=on_train_step,
            on_iteration_end=on_iteration_end, save_checkpoint=save_ckpt,
            log_every=args.log_every,
            start_iteration=start_iteration, start_problem=start_problem)
    except BaseException:
        update_status(run_dir, "failed")
        raise
    update_status(run_dir, "completed")
    return 0


def _mdp_factory(problem, args, encoder, state_actor):
    from verifier_anchored_rl.llm_context_mdp import LLMContextMDP
    from verifier_anchored_rl.transformers_client import TransformersLLMClient
    # The MDP shares the encoder's model and the same prefill cache as the
    # StateActor (one frozen model, one forward per decision).
    return LLMContextMDP(
        problem_statement=problem[0], expected_answer=problem[1],
        llm_client=_shared_llm(encoder), encoder=encoder,
        prefill_cache=state_actor.cache,
        depth_limit=args.depth_limit, max_turns=args.max_turns)


def _shared_llm(encoder):
    from verifier_anchored_rl.transformers_client import TransformersLLMClient
    return TransformersLLMClient(model=encoder.model, tokenizer=encoder.tokenizer)


if __name__ == "__main__":
    sys.exit(main())
```

Note: the test stubs `_build_stack` to return a 6-tuple, but the real
`_build_stack` returns 8 values. Update the test's stub to return the same
arity the code unpacks, OR (preferred) have the test stub return the full
8-tuple `("ENC","SA","TR","BUF","ACTOR","OPT","LLM","CACHE")`. Fix the test
stub accordingly before running (this is the one place the test and code
must agree on arity).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_run_outer_loop_harness.py -v`
Expected: PASS. If it fails on `_mdp_factory` or stack arity, align the stub's return arity with `_build_stack` and ensure `run_outer_loop` is the monkeypatched stub (so no real model loads).

- [ ] **Step 5: Confirm the help text still works without loading a model**

Run: `python experiments/run_outer_loop.py --help`
Expected: usage text including `--run-root`, `--tag`, `--resume`, `--log-every`.

- [ ] **Step 6: Commit**

```bash
git add experiments/run_outer_loop.py tests/test_run_outer_loop_harness.py
git commit -m "Rewrite outer-loop harness: run dir, logging, checkpoint, resume

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Full suite green, then short real-GPU smoke and resume validation

**Files:**
- None (validation only); then append a short note to `experiments/results/expert_iteration_rounds.md`

- [ ] **Step 1: Run the entire test suite**

Run: `python -m pytest -q`
Expected: all pass (the prior 193 plus the new tests). No warnings beyond the pre-existing one in `test_heads.py`.

- [ ] **Step 2: Short real-GPU smoke (about 10 to 20 minutes)**

Run:
```bash
python experiments/run_outer_loop.py --iterations 1 --num-simulations 8 \
    --train-steps 50 --problems multiply_small --tag smoke
```
While it runs, in another shell confirm `progress.log` is growing with
per-decision lines and that `checkpoint.pt` appears after the first
trajectory:
```bash
ls -la experiments/results/runs/*-smoke/
tail -f experiments/results/runs/*-smoke/progress.log
```
Expected: per-decision lines stream during collection; `metrics.jsonl` gets a
line with non-NaN `L_Q_first`/`L_Q_last`; `metadata.json` ends with status
`completed`; `summary.md` has one row.

- [ ] **Step 3: Validate resume by interrupting and restarting**

Run a 2-iteration smoke, interrupt it with Ctrl-C after the first
iteration's checkpoint appears (status will be `failed`), then resume:
```bash
python experiments/run_outer_loop.py --iterations 2 --num-simulations 8 \
    --train-steps 50 --problems multiply_small --tag resume
# (Ctrl-C after iter 1 checkpoint, then:)
python experiments/run_outer_loop.py --resume experiments/results/runs/<the-resume-dir>
```
Expected: the resumed process loads the checkpoint, logs that it starts at
iteration 2 (no re-collection of iteration 1), and finishes with status
`completed`. `metrics.jsonl` ends with two iteration rows.

- [ ] **Step 4: Record the validation in the journal**

Append a short dated note to `experiments/results/expert_iteration_rounds.md`
recording that the instrumented and resumable harness was validated on frodo:
the run-dir artifacts produced, the smoke timing, and confirmation that
resume continues from the checkpoint. Do NOT use the em-dash glyph.

- [ ] **Step 5: Commit**

```bash
git add experiments/results/expert_iteration_rounds.md
git commit -m "Validate instrumented resumable harness on frodo (GPU smoke + resume)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Periodic flushed output: Task 8 `log()` plus `on_decision`/`on_train_step`. Covered.
- records.jsonl streaming: Task 8 `record_sink`, using the existing hook. Covered.
- metrics.jsonl with loss split: Tasks 3, 5, 8. Covered.
- metadata.json with provenance: Task 6 `create_run_dir`, Task 8 metadata assembly. Covered.
- Atomic per-trajectory + post-training checkpoint: Tasks 5 (`save_checkpoint` cadence), 6 (`save_checkpoint_atomic`), 8 (closure). Covered.
- Resume contract: Tasks 4 (`start_problem`), 5 (`start_iteration`), 8 (`--resume`), 7 (equivalence proof). Covered.
- summary.md rewritten each iteration: Task 8 `on_iteration_end`. Covered.
- L_Q/L_pi split: Tasks 2, 3, 5. Covered.
- Testing (CPU TDD then GPU smoke): Tasks 1 to 8 are CPU TDD; Task 9 is the GPU smoke and resume validation. Covered.
- Non-goals respected: no algorithm change (losses/MCTS/heads untouched); plain JSONL, no W&B/trackio.

**Placeholder scan:** No TBD/TODO. Every code step shows full code. Task 8 explicitly flags the one stub-arity agreement point rather than leaving it implicit.

**Type/name consistency:** `state_dict`/`load_state_dict` (Task 1) used by Tasks 7, 8. `loss_summary` (Task 3) used by Task 5. `on_step` (Task 2) used by Task 5. `on_decision`/`on_trajectory_end`/`start_problem`/`iteration_index` (Task 4) used by Task 5. `save_checkpoint_atomic`/`load_checkpoint`/`create_run_dir`/`append_jsonl`/`update_status` (Task 6) used by Task 8. Checkpoint dict keys (`phase`, `iteration`, `problems_done_this_iter`, `actor_state`, `optimizer_state`, `buffer`, `global_step`, `numpy_rng_state`) are identical across Tasks 5, 7, 8. Consistent.
