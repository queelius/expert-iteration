# CLAUDE.md

Current work (2026-10-20): prepare a pedagogical research archive, not new
controller experiments. Read `README.md`, `docs/release-readiness.md`, and
`docs/superpowers/plans/2026-10-20-public-research-archive.md`.
The research description below is historical; its invariants still apply.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A verifier-anchored RL system: a small external controller network (verb head,
pointer head, Q head; a few million parameters) is trained to orchestrate a
**frozen** LLM by issuing typed `(verb, target)` actions over a structured,
nested reasoning forest, with a symbolic verifier as the **only** reward
source. This is explicitly NOT an o1/DeepSeek-R1-style system: RL never
touches the LLM's weights.

**`design.md` (repo root) is the source of truth.** It is self-contained
(state, actions/lenses, controller, reward, training, evaluation, prior-art
positioning). Historical documents live in `docs/archive/`:
`design-2025-pedagogical.md` (the original MCTS-pedagogy design plus its
LLM specialization; still the detailed reference for the classical
Gridworld/value-iteration/expectimax/MCTS core) and `paper-v1/` (the
superseded 48-page paper).

## Framing directive (do not regress)

**RL first, MCTS second.** The central object is the agent (policy + Q heads)
learning over an induced context MDP whose only reward source is the symbolic
verifier. The training spine is off-policy value-based Q-learning with n-step
or Monte Carlo returns; MCTS is a strictly additive accelerator (better
behavior, better targets) and the test-time compute knob, never the spine
(`design.md` section 0 and section 7).

**Subagent-briefing discipline:** every paper-drafting or doc-restructuring
subagent brief must restate this directive verbatim at the top. Verified
failure mode (2026-05-23): an entire paper draft regressed to MCTS-first
framing because the directive was not in the briefs. Hard checklist item.

## Common commands

```bash
pip install -e ".[dev]"        # editable install, pytest
pip install -e ".[llm]"        # + httpx, sympy (LLM-backed paths, verifier)
pytest                          # full suite
pytest tests/reductio/ -q       # the vendored state core (48 tests)
pytest tests/test_mcts.py::test_puct_respects_priors_when_values_equal  # one test
cd paper && latexmk -pdf main.tex   # build the paper scaffold
pandoc design.md -o design.pdf --pdf-engine=xelatex -V colorlinks=true  # design PDF
```

Python 3.11+; core deps are `numpy` + `pytest` only. No lint config; prefer
`ruff` if introducing one.

## Architecture map

- **`src/verifier_anchored_rl/reductio/`**: the vendored state core (from the
  now-archived github.com/queelius/reductio, slimmed during roll-in). An
  immutable reasoning forest (`Goal`/`Hypothesis`/`Fact`/`VerifierCall`
  frozen dataclasses) plus a pure verb algebra (`split`, `form_hypothesis`,
  `compute`, `verify`, `commit`, `abandon`: total functions
  `Graph -> Ok | Err`). Invariants the tests pin and any change must
  preserve: nodes are frozen with no reward/value fields; core modules import
  stdlib only (sympy lives in `adapters/sympy_math.py`; boundary test
  enforces this); `solved`/`verified`/`holds` are derived predicates that
  never gate anything except `commit`'s verifier-vouched witness check;
  structure never gates (a parent commits regardless of children; the
  AND/OR connective is non-gating metadata); there is NO Method node kind.
- **Classical core** (`gridworld.py`, `value_iteration.py`, `expectimax.py`,
  `mcts.py`, `mdp.py`, ...): the validation anchor. Value iteration,
  expectimax, and MCTS triangulate on hand-computable MDPs; the LLM system
  must never regress these tests. Details in the archived pedagogical design.
- **`deliberation_mdp.py` + `lenses.py` + `verifier_adapter.py` +
  `llm_controller.py`**: the LLM-context MDP over the reductio core.
  Six `(verb, target)` verbs; `VERIFY`/`COMMIT`/`ABANDON` never call the
  LLM (the policy can neither author what is checked nor what it is
  checked against); statements are sanitized against marker injection;
  terminal-only verifier reward. `llm_controller.py` is the Rung-1
  LLM-as-controller runner whose JSONL logs feed Phase-B behavior cloning.
- **`paper/`**: fresh LaTeX scaffold (article + natbib + latexmk). STUB
  discipline: `\stub{...}` marks content awaiting the integration or
  experimental runs; never present stubbed content as complete, and never
  resurrect numbers from the archived pilot.

## Reward invariant (load-bearing)

The symbolic verifier is the only reward source: terminal +1/-1 at root
COMMIT, optional gated and capped intermediate reward through VERIFY, zero
everywhere else. No reward model, no learned process reward, no LLM-judge
reward, ever. Aesthetic signals may enter only as search priors or pruning
heuristics, never as Bellman targets.

## Style requirements (pedagogical, load-bearing for the paper)

1. Expanded sums before expectation notation; define `P(s'|s,a)`,
   `R(s,a,s')`, `pi(a|s)`, `gamma` before use.
2. Any shorthand (`G_t`, `r_T`) defined where it first appears.
3. Code aligned with the math (loop variables named `next_state`, `prob`,
   `reward`).
4. Hand-computable examples before general algorithms.
5. Do not write the em-dash glyph in files (a hook rejects it); use commas,
   colons, parentheses, or hyphens.

## Honesty discipline

Present the `(verb, target)` lens system as the method, with explicit
markers separating what the code realizes today from what is designed but
unbuilt. Do not claim as built what is still spec. Report archived pilot
results nowhere.

## Non-goals

- No reward model, no learned PRM/ORM/critic in the reward path.
- No macro solve-it-all verb bypassing per-step search and verification.
- No adversarial games, minimax, or multi-agent scope creep.
- No POMDP belief states; no web UI.
- No regression of the classical anchors; no MCTS-first reframing anywhere.
