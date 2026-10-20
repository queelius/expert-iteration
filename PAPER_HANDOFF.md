# Handoff: update the paper from bilbo (latest insights and results)

> Historical handoff, not an active instruction. As of 2026-10-20, this
> repository is being prepared as a research archive. The manuscript remains
> unfinished. Start with `README.md` and `docs/release-readiness.md`.

**Date:** 2026-06-19. **From:** frodo (GPU box, running the experiment). **To:**
bilbo (your local Claude Code session, owning the paper).

## TL;DR

Your job is to keep `paper/` current with our latest insights and results. The
closed-loop implementation writeup is already drafted and pushed (pilot
subsection 5.6 plus a light abstract touch). The one big result that is NOT in
yet is the headline closed-loop run, which is executing on frodo right now; its
numbers are pending. When that run finishes, frodo will append the numbers to
the running journal and push, and your task is to fold them into the paper.

## CRITICAL framing rule (do not skip, restate it to any subagent)

From `CLAUDE.md`, verbatim, the load-bearing directive for all paper work:

> Critical ordering note (design.md section 1): the design was revised to teach
> RL first, MCTS second. MCTS is framed as a sample-efficient policy-improvement
> operator that produces stronger training targets, not as the central object.
> When writing docs, comments, paper sections, or the blog article, do not
> regress to MCTS-first framing.

And the subagent discipline, also verbatim:

> Subagent-briefing discipline: when dispatching subagents for any paper
> drafting or restructuring work, restate the section 1 RL-first directive at
> the top of the subagent brief verbatim. ... every paper-related subagent brief
> gets the section 1 RL-first directive at the top, every time.

A subagent that reads the bulk of design.md statistically pulls toward
MCTS-first unless this correction is surfaced explicitly. Put it at the top of
every paper subagent brief.

## Second hard rule: no em-dash glyph

A repo hook rejects any write containing the em-dash glyph (the long dash). Use
commas, colons, periods, or parentheses. This applies to every file, including
this one and commit messages.

## What is already done and pushed (do not redo)

Origin `main` is at the tip that includes all of this (run `git log --oneline`
to confirm). Relevant commits:

- `paper: document the closed-loop implementation and the in-progress run` adds
  pilot subsection 5.6 "Closing the Loop End-to-End (Run In Progress)"
  (`paper/sections/pilot_results.tex`) and a light abstract touch
  (`paper/sections/abstract.tex`). It covers: the in-process frozen-model
  backend and faithful h_T embedding, the shared single-forward encoder
  (one-forward-per-decision, bit-identical h_T, about 53 percent prefill
  saving), the full Section 3.7 training loop (combined L_Q + lambda L_pi
  against Q_MCTS, prioritized replay, target-Q sync; L_Q down about 10x on the
  pilot's own v2 data with the L_pi floor matching the pilot's 1.4512), and the
  first closed-loop GPU run's per-decision compute accounting (about 27
  generations / 9k generated tokens per decision vs about 1 prefill of 114
  tokens, generation dominating about 79 to 1). It also threads the new harness
  into the third Lesson, the Caveats, and Reproducibility.
- Section 5.6 is explicit that the headline run is pending. The build is clean
  (50 pages, all cross-references resolve, no em-dash).

So the closed-loop infrastructure story is told. Do not rewrite 5.6 from
scratch; extend and finalize it.

## The source of truth for results

`experiments/results/expert_iteration_rounds.md` is the running journal and the
authority for every number. Read it before editing the paper. The closed-loop
sections at the end (the in-process backend, the shared-forward encoder, the
Section 3.7 training loop, the outer loop, the first GPU run, and the
observable/resumable harness validation) are what 5.6 summarizes. The pending
headline numbers will be appended there by frodo when the run completes.

## The pending experiment (the most important one)

Frodo is running the first fully instrumented closed-loop run:

- config: 5 iterations, num_simulations 8, train_steps 200 per iteration, the 4
  toy training problems (multiply_small, gsm8k_toy_1, algebra_toy_1, factor),
  substrate Qwen2.5-Math-1.5B-Instruct (an instruct-and-math variant chosen for
  feasibility, NOT the base-only substrate the pre-registered study mandates;
  keep that caveat in the paper), tau 1.0, c_puct 1.5, discount 1.0,
  sync_interval 200, seed 0.
- what it will deliver, per iteration, in the run's metrics.jsonl: separated
  L_Q and L_pi (first and last), L_total, the visit-distribution entropy and
  the learned-prior entropy (the sharpening diagnostic, in bits), the FLOP
  counters, decisions, buffer size, wall time. Early reading: iteration 1 shows
  both entropies at about 2.0 bits (log2 of 4 verbs), the unsharpened baseline
  we expect to fall in later iterations.
- the two signals to report: the across-iteration L_Q trend (approximate
  Q-learning) and the across-iteration sharpening (prior and visit entropy
  decreasing). These are the signals the first 14.7 hour run could not surface,
  which is why the harness now persists them.

When frodo appends the final numbers to the journal and pushes, replace the
"pending" wording in two places:

1. `paper/sections/pilot_results.tex`, the last paragraph of subsection 5.6
   (currently begins "The first end-to-end run of this fully instrumented loop
   is executing as we write"). Replace with the result: the L_Q trend and the
   per-iteration entropy decrease, ideally as a small per-iteration table
   (iteration, L_Q first to last, L_pi, visit entropy, prior entropy). Frame it
   RL-first: the loop is approximate Q-learning, MCTS supplies the targets.
2. `paper/sections/abstract.tex`, the clause "a fully instrumented run that
   captures the across-iteration Q-loss trend and the prior-sharpening signal
   is currently executing, with those learning-signal results pending."
   Replace "currently executing, with those learning-signal results pending"
   with the headline finding.

If the run is still pending when you work, do the independent tasks below and
leave the two pending spots untouched.

## Tasks you can start now (independent of the pending run)

1. Pull and read: `git pull` (rebase is the configured default on frodo; set it
   on bilbo too with `git config pull.rebase true`). Read 5.6, the abstract,
   and the journal.
2. Optional method cross-links: in `paper/sections/method.tex`, Sections 3.4
   (architecture) and 3.7 (training loop) are now implemented, not only
   specified. A one-line forward reference to 5.6 in each is reasonable, but do
   not restructure the method section.
3. Optional discussion: consider whether the discussion or conclusion should
   note the two methodological contributions the closed-loop work surfaced: the
   one-forward-per-decision FLOP accounting (generation dominates about 79 to 1)
   and the observability-and-resumability harness as research practice (a long
   run must persist per-iteration signals and checkpoints, learned the hard way
   from a 14.7 hour run that yielded three summary lines). Keep it brief and
   RL-first.
4. Consistency pass: confirm nothing you add reintroduces MCTS-first framing,
   and that 5.6 and the abstract agree with the rest of the paper.
5. Build check before pushing: `cd paper && latexmk -pdf -interaction=nonstopmode
   -halt-on-error main.tex`, then confirm zero errors and no undefined
   references in `paper/main.log`. Run-output and LaTeX build artifacts are
   gitignored.

## Coordination protocol (avoid the divergence we just fixed)

- frodo owns the experiment, the journal (`experiments/results/`), and code.
  frodo will NOT edit `paper/` going forward.
- bilbo owns `paper/`. Edit only paper files (and method/discussion as above) so
  the two sessions never touch the same files.
- Always `git pull --rebase` before starting and `git push` after committing.
  Both boxes are active (frodo runs the experiment, you write the paper), so
  sync often.

## Map of the work

- Paper: `paper/main.tex` and `paper/sections/*.tex`. The new content is in
  `paper/sections/pilot_results.tex` (subsection 5.6) and
  `paper/sections/abstract.tex`.
- Results authority: `experiments/results/expert_iteration_rounds.md`.
- Design and plan for the harness work: `docs/superpowers/specs/2026-06-19-
  observable-resumable-outer-loop-design.md` and `docs/superpowers/plans/2026-
  06-19-observable-resumable-outer-loop.md`.
- The running experiment's outputs live on frodo under
  `experiments/results/runs/20260619T193634Z-headline/` (gitignored, not
  visible from bilbo; numbers arrive via the journal).


## UPDATE (2026-06-21): the pending experiment completed (and a fix landed first)

The "currently executing, results pending" framing in section 5.6 and the
abstract is now superseded. Two things happened:

1. The first closed-loop runs were found to be running on masked no-op
   transitions (verb instantiation failed schema validation on the small
   model), so they could not sharpen by construction. A fix landed:
   chat-template plus JSON-schema-constrained decoding (lm-format-enforcer),
   plus a FORM_HYPOTHESIS schema fix. See the journal section "Verb
   instantiation was masking out" (2026-06-21) for the full diagnosis, the fix,
   and the GPU validation (0 of 4 to 4 of 4 non-masked; verifier reward firing).

2. A 5-iteration closed-loop run on the fixed path completed (73 minutes, B=8,
   Qwen2.5-Math-1.5B-Instruct, git 0dbf75a). Honest result, with numbers in the
   journal:
   * the loop is now functional (non-degenerate signals throughout);
   * one round of sharpening is clear (iteration 1 to 2: prior entropy 2.628 to
     2.163 bits, visit concentration 0.024 to 0.060, L_pi floor 1.798 to 1.532);
   * sustained monotone sharpening does NOT occur at B=8 (the prior oscillates
     and partly de-sharpens after iteration 2; net change is in the sharpening
     direction but noisy and non-monotone), reproducing the pilot's
     target-instability finding and the B-sweep "budget is the lever" conclusion
     in the closed online loop.

Paper actions for the bilbo session (replace the pending framing honestly):
* Section 5.6 last paragraph: replace "is executing as we write ... result is
  not yet in" with the completed result above. Frame it as: the fix made the
  loop functional and it demonstrably closes for one round, while sustained
  round-over-round sharpening is budget-limited at B=8, reinforcing (not
  contradicting) the pilot's target-instability story and the budget-as-lever
  argument. Keep RL-first framing. Consider a small per-iteration table.
* Section 5.6 should also gain a short note on the verb-instantiation failure
  and the constrained-decoding fix: it is a substantive, honest engineering
  finding (a small instruct model cannot emit schema-valid verb JSON without
  constrained decoding) that the benchmark study inherits as a requirement.
* Abstract: replace "currently executing, with those results pending" with the
  one-round-sharpening-but-budget-limited result, keeping the substrate caveat
  (instruct/math 1.5B, not the base-only pre-registered substrate).

The journal (experiments/results/expert_iteration_rounds.md) is the numbers
authority. Pull before editing (frodo pushed through git 0dbf75a plus the
journal update).
