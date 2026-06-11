# verifier-anchored-rl

Approximate $Q$-learning over verifiable context MDPs, with Monte Carlo Tree
Search as a sample-efficient policy improvement operator and a test-time
compute knob. The project develops the chain that begins with the Bellman
equation, moves through tabular and approximate $Q$-learning, exploration,
and MCTS, and finally generalizes to LLM deliberation over a verifiable
context MDP. Two methodology papers accompany the implementation: the
framework paper at `paper/` and the companion paper on bounded structural
recursion at `paper-recursive/`.

This repository ships the classical-MDP scaffold (Phase 1 plus the MCTS
planner from Phase 2 of the design document). It implements:

- a `FiniteMDP` / `GenerativeMDP` protocol pair (PEP 544 structural typing),
- a small stochastic Gridworld environment with noisy moves and terminal
  goal / pit rewards,
- an exact value iteration solver,
- a depth-limited expectimax planner with optional memoization,
- a chance-node MCTS planner with PUCT selection and per-edge reward backup,
- pytest coverage tying the four planners together, including the
  hand-computable S/A/G/B triangulation invariant.

The expert iteration training loop and the LLM-context extension are
scheduled for later phases; see `design.md` for the roadmap.

## Where things live

| Path | Purpose |
|------|---------|
| `design.md` | Authoritative design specification (Phase 1, 2, 3, and the LLM-context MDP target) |
| `paper/` | Methodology paper: Verifier-Anchored Reinforcement Learning for Small Language Model Deliberation |
| `paper-recursive/` | Companion paper: Bounded Structural Recursion in Verifier-Anchored Reinforcement Learning |
| `src/verifier_anchored_rl/` | The Python package |
| `tests/` | Pytest suite |
| `experiments/` | Standalone scripts that exercise the package end to end |

## Install

The package uses a standard PEP 621 layout with a `src/` directory. Editable
install from the repository root:

```bash
pip install -e ".[dev]"
```

Python 3.11 or newer is required. The only runtime dependency is `numpy`;
`pytest` is the only dev dependency. No PyTorch, no LLM libraries, and no
network access are needed for the classical-MDP scaffold.

## Run the tests

```bash
pytest -v
```

The suite covers the Gridworld dynamics (transition probabilities sum to
one, wall and boundary stay-in-place, terminal-state convention, reward
attribution), the value-iteration solver (the hand-computable S/A/G/B
example from `design.md` section 10.2, plus Gridworld sanity checks), the
expectimax planner (agreement with value iteration on both the toy MDP and
a small Gridworld), and the MCTS planner (visit-count properties, PUCT
prior respect, convergence toward value iteration's $Q$ on S/A/G/B and
small gridworlds, and per-edge backup correctness).

## Run the experiments

```bash
python experiments/run_value_iteration.py
python experiments/run_mcts.py
```

The first script builds the canonical example layout from `design.md`
(section 5.1), runs value iteration with `gamma = 0.99`, and prints the
optimal value table and the greedy policy as ASCII grids. The second runs
MCTS from the start state with 2000 simulations and compares its
recommended action and visit-count policy against value iteration.

## What is not implemented yet

The expert iteration training loop and policy distillation, the LLM-context
MDP extension, and any neural-network or LLM-backed components are
scheduled for later dispatches. The `design.md` file is the source of truth
for what each phase contains.
