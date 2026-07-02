# Archived experiments

These data, results, and scripts predate the 2026-06 action-space redesign (the
move to the `(verb, target)` lens model and the reduced verb set). They reflect
the older verb-only / seven-verb pilot and are kept for the record only. Do not
treat any number here as current: the paper no longer leans on them. New runs
against the redesigned action space live under `experiments/`, not here.

- `data/`    : pilot distillation data, trained heads, and budget-sweep / eval outputs.
- `results/` : pilot run summaries and the correctness audit.
- `*.py`     : pilot-era scripts built against the deleted `llm_context_mdp.py`
  monolith (`MathState`/`MathAction`, `LLMContextMDP`). They do not run
  as-is against the current `DeliberationMDP` stack (no mechanical
  import re-point exists: `MathAction(verb=...)` has no `target`,
  `state.committed_answer` and `state.hypotheses` have no
  `DeliberationState` equivalent) and are frozen here rather than ported.
  `run_outer_loop.py`, `run_shared_forward_smoke.py`,
  `run_transformers_smoke.py`, and `run_llm_controller.py` (Rung-1) cover
  their ground against the new stack.
