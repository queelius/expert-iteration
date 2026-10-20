# Agent instructions (Codex and others)

Current task (2026-10-20): archive preparation. Start with `README.md`,
`docs/release-readiness.md`, and the public research archive implementation
plan under `docs/superpowers/plans/`. Do not restart the historical Phase B
critical path below. Preserve the scientific and repository rules.

Read, in this order:
1. `HANDOFF.md`: full project state, results, and next steps (2026-07-09).
2. `CLAUDE.md`: repo rules; the content applies to ANY agent, not just
   Claude (framing directive, reward invariant, style rules, non-goals).
3. `design.md`: the system specification (source of truth).
4. `docs/superpowers/research/EXPERIMENTS.md`: the experiment ledger
   (E1-E17); every empirical claim traces to it and to
   `experiments/runs/*.jsonl`.

Non-negotiable rules (duplicated from those files because they bite):
- Never write the em-dash glyph in any file; a commit hook rejects it.
  Use commas, colons, parentheses, or hyphens.
- `git add` explicit paths only; never `git add -A`/`-u`/`.` (the
  working tree carries unrelated uncommitted edits).
- The symbolic verifier is the ONLY reward source. No reward model,
  learned PRM/ORM/critic, or LLM-judge, ever. VERIFY/COMMIT/ABANDON
  never call the LLM.
- Framing: RL first, MCTS second. Any paper/doc work restates this
  directive; MCTS is a strictly additive accelerator, never the spine.
- Experiments: one manipulation each, pre-register the measure and its
  reading in the ledger BEFORE running, log everything.
- Outward artifacts (paper, deck, emails) carry forward-facing claims
  only; debugging history stays in the ledger.
- Tests: `pytest -q` must stay green (324 passed, 6 skipped as of
  2026-07-09). Python 3.11+, PYTHONPATH=src for the experiment scripts.

Current critical path: Phase B (behavior-cloning warm-start + off-policy
Q-learning for the controller heads); see HANDOFF.md section 8.
