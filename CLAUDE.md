# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

This repo contains a methodology paper at `paper/`, a design document at `design.md` (source of truth for what to build), and a Python implementation under `src/verifier_anchored_rl/`. The package is named `verifier_anchored_rl` to align with the framework paper's title and the §1 RL-first directive; `design.md` §3 still mentions the older `mcts_expert_iteration` name in its layout description, which is a historical artifact that should be read as the now-renamed package.

When implementing, follow the repository layout in `design.md` §3 (with the package directory renamed to `src/verifier_anchored_rl/`): `tests/`, `experiments/`, `docs/`. Do not invent a different layout.

## Common commands (after implementation exists)

The design specifies Python 3.11+ with `numpy` and `pytest` as the only core dependencies. `matplotlib` is optional for plots; `torch` is explicitly forbidden until tabular expert iteration works end-to-end (`design.md` §3).

```bash
# Install in editable mode with dev deps
pip install -e ".[dev]"

# Run the full test suite
pytest

# Run a single test file
pytest tests/test_value_iteration.py

# Run a single test by node id
pytest tests/test_mcts.py::test_puct_respects_priors_when_values_equal

# Run experiments (each is a standalone script)
python experiments/run_value_iteration.py
python experiments/run_mcts.py
python experiments/run_expert_iteration.py
```

There is no lint configuration prescribed by the design. If introducing one, prefer `ruff` over a heavier stack to keep the dependency footprint small.

## Architecture: the pedagogical spine

The project teaches an explicit conceptual chain, and the code should make every link visible. The chain (from `design.md` §0):

```
utility maximization
  → deterministic search
  → stochastic search / expectimax
  → MDPs and Bellman equations
  → MCTS as sampled expectimax / online planning
  → MCTS expert iteration as approximate policy iteration / RL
  → LLM deliberation as an induced context MDP
```

**Critical ordering note** (`design.md` §1): the design was revised to teach **RL first, MCTS second**. MCTS is framed as a sample-efficient policy-improvement operator that produces stronger training targets, not as the central object. When writing docs, comments, paper sections, or the blog article, do not regress to MCTS-first framing.

**Subagent-briefing discipline**: when dispatching subagents for any paper drafting or restructuring work, restate the §1 RL-first directive at the top of the subagent brief verbatim. The directive is a layered correction over older MCTS-first content in `design.md`, and a subagent reading the bulk of the design document statistically pulls toward MCTS-first framing unless the correction is surfaced explicitly. Verified failure mode (resolved 2026-05-23): an entire paper draft regressed to MCTS-first framing because the directive was not in the subagent briefs, even though it was in this file. Treat this as a hard checklist item: every paper-related subagent brief gets the §1 RL-first directive at the top, every time.

## Architecture: layered interfaces

Two MDP protocols separate concerns by what each algorithm needs:

- `FiniteMDP[S, A]`: exposes `transitions(s, a) → [(s', prob, reward)]`. Required by **value iteration** and **expectimax**, which sum over all next states.
- `GenerativeMDP[S, A]`: exposes `sample_transition(s, a, rng) → (s', reward)`. Required by **MCTS**, which samples rather than enumerates.

The Gridworld environment implements **both**. This is intentional: the same environment supports exact planners (for ground-truth comparison) and the sampling-based planner that will later generalize to the LLM-context setting where `P(s'|s,a)` cannot be enumerated.

Use `Protocol` (PEP 544 structural subtyping), not ABCs. This matches the design's `from typing import Protocol` snippet in §4.1 and keeps `Gridworld` and any later `ContextEnvironment` decoupled.

## Architecture: MCTS with explicit chance nodes

`design.md` §8.1 mandates **separate `DecisionNode` and `ChanceNode` types**. Decision nodes hold state and per-action children; chance nodes hold `(s, a)` statistics (`visits`, `total_value`, `mean_value`, `prior`) and a list of sampled successor decision nodes.

Why this matters: the MCTS chance-node mean value is the empirical estimator of

```
Q(s,a) = Σ_s' P(s'|s,a) [R(s,a,s') + γ V(s')]
```

Collapsing chance and decision nodes (common in deterministic-game MCTS tutorials) hides this estimator and breaks the analogy to Bellman that the article is supposed to teach.

**Backup must accumulate per-edge rewards**, not a single terminal scalar. The pattern from §8.4:

```python
G = leaf_value
for chance, reward in reversed(path):
    G = reward + gamma * G
    update chance with G
```

A backup that propagates one fixed return to every ancestor is wrong for discounted MDPs with intermediate rewards (the gridworld has step rewards of -0.01).

PUCT formula (`design.md` §8.2): `Q(s,a) + c_puct · P0(a|s) · sqrt(N(s)) / (1 + N(s,a))`. The prior may be uniform initially; in expert iteration it becomes the learned policy.

## Architecture: expert iteration as policy distillation

Loop from `design.md` §9.3:

```
search → improved visit distribution → distill → better policy prior → better search
```

`pi_MCTS(a|s) = N(s,a)^(1/τ) / Σ_b N(s,b)^(1/τ)` is the distillation target. Train `p_θ(a|s)` toward it with cross-entropy.

**Start with a tabular softmax policy.** §9.1 is explicit: do not introduce a neural network until the tabular version trains and improves measurably. The pedagogical value comes from seeing the loop close in the simplest possible setting.

The later sections (`design.md` §17–19) extend this to:
- **Online learning during deployment** (§17): user feedback as reward, per-user adapter tiers, channel-decomposed reward (correctness vs satisfaction).
- **Direct Q-learning from chance-node estimates** (§18): `Q_ψ(C,a) ≈ Q̂_MCTS(C,a)`, with the policy prior derived by softmax over Q. Recommended over a separate policy network in the LLM-context setting.
- **Inference-time compute budget B** (§19): `B=0` uses the learned model directly, `B>0` runs MCTS for that many simulations. Persistent search trees enable "think harder" semantics.

These are design extensions, not Phase 1 work.

## Phase discipline (do not skip ahead)

`design.md` §2 partitions the work into phases that must be respected:

- **Phase 1**: Gridworld + `FiniteMDP`/`GenerativeMDP` + value iteration + expectimax + MCTS + tests. **Do not implement MCTS until value iteration and expectimax tests pass** (§15).
- **Phase 2**: Tabular policy prior + PUCT-guided MCTS + visit-count distillation + training loop.
- **Phase 3**: LLM context MDP. **Relaxed 2026-05-27 (user instruction to begin Tier 2 smoke test):** LLM calls are now permitted. The schema, action space, structured-marker serializer, and `LLMContextMDP.sample_transition` are implemented and tested with `MockLLMClient`, `EchoLLMClient`, and `OpenAICompatibleLLMClient` (HTTP to vLLM/Ollama/llama.cpp servers). The verifier (`verifier.py`) wraps SymPy. LLM-extension deps are optional (`pip install -e .[llm]`).

The first version exists to make "the formal core unmistakably correct" (§16). Speed, polish, and breadth come later.

## Validation: hand-computable ground truth

Tests should anchor on the tiny MDP from `design.md` §10.2 (the S/A/G/B example) where the optimal values are derived by hand:

```
V*(A) = 10
Q*(S, safe)  = 4
Q*(S, risky) = 0.8·10 + 0.2·(-10) = 6
V*(S) = 6
```

Value iteration must reproduce these exactly. Expectimax at sufficient depth must match. MCTS must converge toward them within a configurable simulation budget. This triangulation is the central correctness signal. If any of the three disagrees with the hand calculation on this MDP, the bug is in that planner, not in the math.

Gridworld property tests (`design.md` §10.1): transition probabilities sum to 1 for every `(s, a)`; walls and boundaries keep the agent in place; terminal states have `V = 0`.

## Style requirements (load-bearing for the article)

`design.md` §12 imposes a specific pedagogical style that comments, docstrings, and the eventual blog article must honor:

1. Write expanded sums (`Σ_s' P(s'|s,a) [R(s,a,s') + γV(s')]`) before introducing `E[...]` expectation notation.
2. Define `P(s'|s,a)`, `R(s,a,s')`, `π(a|s)`, and `γ` explicitly before using them.
3. Any shorthand (e.g., `G_t`, `R_t`) must be defined where it first appears. No unexplained notation in comments.
4. Keep code aligned with the math: the loop variable names in `value_iteration` and `expectimax_value` should match the symbols in the corresponding equations (`next_state`, `prob`, `reward`).
5. Show hand-computable examples before general algorithms in docs.

This is not aesthetic preference. The project's deliverable includes `docs/blog_article.md`, and the article's argument requires the math, code, and prose to share a vocabulary. Compact notation introduced too early defeats the pedagogy.

## Non-goals (design.md §16)

Do not introduce, even when tempting:

- neural networks beyond the small Q/policy heads the framework specifies (no general-purpose ML stack creep)
- ~~LLM API calls (Phase 3 is design-only)~~ **Relaxed 2026-05-27.** LLM calls are permitted through `LLMContextMDP` against an OpenAI-compatible HTTP endpoint (Ollama or vLLM on a LAN GPU box). The classical-MDP scaffold remains the validation anchor; LLM-extension code does not regress the gridworld/expectimax/MCTS tests.
- adversarial games (Go/chess/checkers); minimax; multi-agent
- POMDP belief states
- reward-model training (the verifier is the only reward source)
- a web UI

The single-player stochastic Gridworld is chosen because it generalizes cleanly to the LLM context MDP without adversarial complexity. Resist scope creep into game-playing.

## Deliverables checklist (§14)

A "done" Phase 1+2 implementation produces:

1. Working `verifier_anchored_rl` package (editable-installable).
2. Gridworld environment implementing both protocols.
3. Value iteration, expectimax, chance-node MCTS, tabular policy, expert-iteration training loop.
4. Tests for each component, including the hand-computable triangulation in §10.
5. `experiments/run_*.py` scripts that produce reproducible numbers/plots.
6. `docs/blog_article.md` drafted to the §11 outline, honoring §12 style.
7. `README.md` with run instructions.

When marking work complete, verify against this list and against §10's correctness criteria, not just "tests pass."
