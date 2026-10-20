# Project Map

## Current Reader Route

Start with the [overview](research-overview.md), then the
[findings](findings.md), [reproduction instructions](reproduction.md), and
[tutorial](tutorial.md). The [readiness record](release-readiness.md) tracks
verification and exclusions.

## Code and Evidence

| Location in the full source repository | Purpose |
| --- | --- |
| `src/verifier_anchored_rl/gridworld.py` | Small stochastic environment |
| `value_iteration.py`, `expectimax.py`, `mcts.py` in that package | Exact and search-based planning |
| `expert_iteration.py`, `policies.py`, `eval.py` in that package | Tabular teaching loop |
| `reductio/`, `deliberation_mdp.py` | Historical structured reasoning forest and operations |
| `phase_b*.py` | Historical imitation controller |
| `reasoning_v2/` | Later typed contracts, replay, and training infrastructure |
| `workspace_v1/` | Permissive workspace and later intervention experiments |
| `experiments/runs/` | Original run records and diagnostics |
| `analysis/` | Selected archive claims, hash pins, and offline table reconstruction |
| `tests/` | Unit, integration, and archive checks |

These generations coexist for historical reproduction. They are not all
parts of one finished production system.

## Historical Documents

The full repository retains `design.md`, `HANDOFF.md`, and the dated documents
under `docs/superpowers/`. They describe the state and ambitions at their
respective dates, not the archive's current critical path.

`docs/superpowers/research/EXPERIMENTS.md` is the detailed experiment ledger.
`docs/archive/design-2025-pedagogical.md` explains the earlier classical
learning/search idea. `research-output/` holds prior-art investigations and
follow-up proposals. `paper/` and `docs/archive/paper-v1/` contain unfinished
or superseded manuscript work, not peer-reviewed publications.

The compact release packet omits most historical working documents and
diagnostic sidecars. It is sufficient for its declared metric reconstruction
and tutorial, not for every deeper historical replay.

## Separate Follow-Up Projects

Names below identify separate local projects discussed during this work.
Their current results and public URLs have not been audited for this archive.
No sibling checkout is required to reproduce the selected findings.

| Project | Question discussed in the handoffs |
| --- | --- |
| `llm-intervention-value` | When could interventions help, and what can a controller observe? |
| `worker-combiner-benchmark` | Can a model use workers' subproblem answers effectively? |
| `intermediate-synthesis-audit` | How does intermediate synthesis affect use of supplied work? |
| `structured-decomposition-benchmark` | When does decomposition and combining supplied answers help? |

Do not treat proposed follow-ups as completed results or attribute those
projects' findings to this repository.
