# Targeted Report Revision: Research Handoff

Date: 2026-08-06

Status: idea for a later experiment. This is not the written plan for a model
run, permission to spend money, or an instruction to change the experiment
that the separate Codex session is preparing now.

Destination: `structured-decomposition-benchmark`. This document is retained
in `expert-iteration` to preserve the research path that led to the proposal.

## Short version

The proposed follow-up asks:

> Given a fixed decomposition and three naturally produced subtask reports,
> can we identify which one report, if any, should receive one additional LLM
> revision so that the final answer is correct more often?

Run the oracle experiment before training a controller. Here, "oracle" means
that we try every choice and then look back and select the one that worked
best. Try every permitted revision target from the same saved information. If
even the best target chosen with
hindsight does not beat strong fixed and compute-matched baselines, stop. If
the best target is almost always the same, use a fixed rule. Train a small
target selector only if the best target varies across problems, the advantage
is stable across repeated generations, and the information available before
the choice helps predict the best target.

This follow-up belongs in `structured-decomposition-benchmark` because that
repository already owns the task generator, fixed decomposition, worker
reports, synthesis conditions, exact graders, data boundaries, and split
policy. It should not be implemented as another isolated experiment in
`expert-iteration`.

The separate Codex session is currently preparing a larger decomposition
experiment. It will generate many new original problems. Each problem will be
split into three smaller problems in a predefined, known-correct way. It will
compare direct solving with several ways of supplying the three smaller
answers. This document must not change or delay that experiment.

In this document, "compute-matched direct prompting" means spending about the
same number of model calls or tokens on the complete original problem instead
of spending them on decomposition or report revision.

## Why this question arose

The original `expert-iteration` design used a tree-shaped reasoning state and
a learned `(verb, target)` policy. A frozen LLM generated content from a local
view of the selected target. The controller was intended eventually to use RL,
with MCTS as an optional accelerator.

The initial controller results did not justify that full design:

- E19 used frozen Nomic embeddings for the complete state and each node. Its
  small factored verb and target network reached 64.0 percent exact held-out
  teacher-action agreement.
- A rule using only the exact legal-action menu reached 64.89 percent. The
  learned network therefore did not establish semantic generalization.
- When text embeddings were replaced with zero vectors in E22, exact action
  agreement fell to 49.33 percent. Text helped target selection, but the
  experiment could not separate mathematical meaning from identifiers,
  rendering details, status words, and workflow phase.
- In E20, direct Qwen 1.5B solved 8 of 20 held-out problems. The E19 controller
  with the same worker solved 0 of 20.
- Later experiments found useful fixed interventions, including a fixed
  revision workflow, while an explicit GPT-4.1-mini router did not beat that
  fixed workflow.

These results did not prove that learned routing is impossible. They showed
that the original test asked too many questions at once:

- Can any of the available choices improve the answer?
- Is the best choice different for different problems?
- Can we predict that difference from the information available at the time?
- Does the embedding preserve the useful information?
- Can a small network learn the mapping?
- Does a multi-step policy improve final correctness?

The proposed follow-up separates those questions. It fixes the intervention
verb and initially learns nothing. It first measures whether choosing a target
has any stable value.

## Connection to the decomposition benchmark

The current `structured-decomposition-benchmark` study asks whether fixed
one-level decomposition and subtask reports improve final correctness over
direct and compute-matched direct prompting. It also studies whether the final
model follows correct and incorrect reports.

The completed two-root report-use check found:

- With advisory wording, Qwen matched the answer implied by the reports on 2
  of 10 calls.
- With binding wording, Qwen matched the answer implied by the reports on all
  10 calls.
- Seven report tuples implied the true root answer. Under binding wording,
  Qwen answered all seven correctly.
- Three report tuples deliberately implied a wrong root answer. Under binding
  wording, Qwen followed all three and answered all three incorrectly.

This is evidence that Qwen can combine the reports when clearly required to
do so. The earlier synthesis failures often came from ignoring the reports and
solving the complete problem again. The result uses only two independent root
problems, so it is a mechanism check, not a population accuracy estimate.

The targeted-revision proposal is a natural follow-up to the benchmark's
one-report error work:

- Existing question: what happens when one known report is wrong or repaired?
- Proposed question: from the visible reports alone, can a system decide which
  report is worth revising?

The new question should not be added to the experiment now being prepared.
That experiment should first establish whether decomposition and report use
help on many newly generated original problems.

## Proposed plain-language research question

Primary question:

> After three workers answer three fixed subtasks, can choosing the right one
> to reconsider improve the final answer more than always reconsidering the
> same subtask, choosing randomly, doing no revision, or spending comparable
> compute on the complete problem?

A shorter version is:

> Can we learn which worker report should be checked again?

The question concerns final correctness, not agreement with a teacher and not
whether the controller copies a preferred workflow.

## Questions to answer in order

Answer these four questions one at a time.

### Step 1: Could choosing a report help at all?

From the same saved reports, try every allowed choice. This produces a
complete table showing what happened after every choice. Compare the best
choice for each original problem with
no revision, every fixed target, random targeting, and compute-matched direct
prompting.

This is the oracle test. It asks whether choosing a report could help if we
somehow always chose the best one.

### Step 2: Does the advantage repeat?

An LLM completion can succeed or fail because of sampling variation. Repeat
each action under registered seeds. Determine whether a target that looks best
on some repetitions also performs well on separate repetitions.

If the apparent oracle improvement disappears on separate repetitions, it is a
best-of-N sampling effect rather than predictable target choice.

### Step 3: Can we predict the useful choice?

Only after Steps 1 and 2 succeed, test whether information available before
the choice predicts which report will help. Begin with simple comparisons before
using text embeddings:

- report position;
- report length and valid-format status;
- fixed structural fields;
- parsed report features that are genuinely model-visible;
- a strong fixed rule; and
- an LLM asked directly to choose a report.

The trusted answer, report correctness, future revision, and final reward are
never controller inputs.

### Step 4: Can a small network make a useful choice?

Only after the first three steps succeed, train a small network that scores
the possible report choices. Evaluate it on original problems and structural
patterns that were not used for training or design.

The controller must improve final correctness or the accuracy-cost frontier.
Teacher-action agreement alone is not success.

## Recommended first decision point

Make the decision after the three initial workers have produced their natural
reports but before final synthesis. The saved information is:

```text
original problem
fixed one-level decomposition
natural report 1
natural report 2
natural report 3
```

This is cleaner than first generating a root answer and then deciding what to
revise. It avoids an extra root-generation step, reduces cost, and makes the
intervention easy to define.

The trusted data may also contain exact answers and correctness labels, but
those fields are not shown to the controller.

## Initial choices

Use one fixed intervention verb: reconsider a report.

```text
KEEP_REPORTS
REVISE_REPORT_1
REVISE_REPORT_2
REVISE_REPORT_3
```

For a revision action, the worker receives only:

```text
its original local subtask
its original report
a fixed instruction to check the work and return a revised report
```

The worker does not receive sibling reports, the root answer, the trusted
subtask answer, a correctness label, or verifier feedback that would not be
available in the intended system.

After the selected action, every branch uses the same synthesis model, prompt,
and decoding policy. The synthesizer receives the original problem and the
resulting three-report tuple. Exact task code grades the final answer.

`KEEP_REPORTS` costs less than a revision action. Report both accuracy and
cost. Do not disguise this difference. The compute-matched direct condition
from the decomposition benchmark remains necessary to determine whether a
structured revision is better than simply spending another call on the full
problem.

Do not add root revision, multiple revision verbs, recursion, MCTS, or a
multi-step policy to the first screen. Those changes create different actions
and different information budgets.

## Two ways to calculate the oracle result

Report two different upper bounds in plain language.

### Most optimistic hindsight result

For each root and repetition, select a successful action if any tested action
succeeded. This is the most optimistic reading. It includes luck from model
sampling.

### Result that must repeat

Use some repetitions to estimate which action is best for a root, then
evaluate that selected action on different repetitions. This is a more useful
upper bound for a learnable selector.

Do not choose and score a choice using the same single noisy outcome without
labeling the result as optimistic. Taking a maximum over noisy samples makes
the oracle look better than it is.

## Required comparisons

At minimum, report:

- no revision;
- always revise each named report position;
- random report revision;
- the best fixed policy selected without evaluation leakage;
- the realized hindsight oracle;
- the repeated-action oracle;
- direct prompting;
- compute-matched direct prompting; and
- the fixed decomposition and natural-report conditions from the current
  decomposition study.

After the oracle shows a clear possible improvement, add:

- a simple feature-based target selector;
- an LLM target selector; and
- the small embedding-based target selector.

All comparisons use original problems as the independent units. Repetitions
and action branches from one root are correlated observations, not additional
independent problems.

## Synthesis instruction is an important boundary

The advisory and binding report instructions define different systems.

- Advisory use is the practical setting. The synthesizer may check or reject
  reports, but it may also ignore useful reports and repeat the entire task.
- Binding use proves that the report channel can control synthesis, but it
  also forces the model to follow wrong reports.

The current experiment on newly generated original problems should determine
how these instructions behave before the targeted-revision plan is finalized.
Do not select a synthesis
instruction by looking at targeted-revision outcomes.

The preferred primary setting is advisory report use if it produces a
measurable and usable report channel. Binding use can serve as a mechanism
condition or optimistic upper bound. If only binding use creates revision
possible improvement, the result concerns forced report following, not a generally useful
controller.

## Reusing the existing data infrastructure

The finite-controller rendezvous family already supplies:

- exactly three independent subtasks;
- safe root and local child records;
- a fixed lossless decomposition;
- deterministic generation;
- two independent subtask solution paths;
- two independent root solution paths;
- an exact report combiner;
- exact final grading;
- natural and controlled report machinery;
- stable content identifiers;
- safe and trusted data separation; and
- a calibrated `layered` difficulty where the direct model is neither perfect
  nor hopeless in open development data.

Reuse this infrastructure at a pinned source commit. Do not create another
task generator in `expert-iteration`.

The existing data is not large enough to measure the improvement reliably or
to train a controller:

- The direct calibration roots do not contain the required natural report
  action matrix.
- The completed report-use study has only two independent roots.
- No sealed evaluation set currently exists.

The two existing development problems are useful for implementing and testing
the alternative choices. They are not enough to measure the oracle advantage.

## Data collection record

For each newly generated original problem:

1. Generate the safe original problem and its fixed three-child
   decomposition.
2. Run one initial worker call per child and retain every natural response.
3. Parse each response without outcome-based retry or deletion.
4. Save exactly what the controller would be allowed to see before choosing.
5. Create every choice listed in the written experiment plan from exactly
   that saved information.
6. For each revision branch, run the same revision prompt on the selected
   local report.
7. Run the same synthesis procedure on every resulting report tuple.
8. Grade final answers with exact code.
9. Retain raw responses, parser failures, timeouts, token counts, latency, and
   cost.
10. Repeat according to a seed and repetition plan fixed before execution.

Every action variant, report order, and repetition for one root must remain in
the same data split.

Model-visible records should contain only the original problem,
decomposition, initial reports, and permitted structural fields. Trusted
records should separately contain:

- exact subtask answers;
- whether each initial and revised report is correct;
- the root answer implied by each report tuple;
- the exact original root answer;
- final correctness for every action branch;
- solver and combiner receipts; and
- action and repetition identifiers.

No natural worker failure may be removed because it makes the action matrix
uninteresting. No root may be retained because one target happened to be
especially helpful.

## Stopping rules

Stop before controller training when any of the following holds:

1. The most optimistic hindsight result does not improve meaningfully over no
   revision and strong fixed baselines.
2. The selected action loses that improvement on separate
   repetitions.
3. One fixed choice accounts for almost all of the repeatable improvement.
4. The gain does not survive comparison with compute-matched direct
   prompting.
5. Report revisions rarely affect synthesis because the final model ignores
   the reports.
6. Useful action differences exist only under forced use of reports, and that
   condition is not the intended practical system.
7. The likely recoverable gain is too small to justify the cost and
   complexity of a learned selector.

The exact minimum worthwhile effect, number of roots, repetitions, uncertainty
calculation, and spending cap must be registered before model inference.

## Small controller, only after the oracle test succeeds

The first learned controller should select only the target. The intervention
verb remains fixed.

For each candidate report, it receives:

- an embedding of the original problem and all information visible before the
  choice;
- an embedding of that report's local subtask and current answer;
- simple structural features such as position or role, if those are truly
  visible; and
- an indicator for the keep-reports action when that action is modeled.

The same small network scores each candidate. Its score means:

> Predicted chance that the exact final answer will be correct after taking
> this action and running the fixed synthesizer.

The controller chooses the candidate with the highest predicted value. It
does not solve a subtask, revise a report, or synthesize the answer. The LLM
still performs those operations.

The first representation comparison should include:

1. structure and legal actions only;
2. simple lexical features;
3. one pooled embedding of all visible information;
4. an embedding of the complete problem plus separate embeddings of each
   subtask and report; and
5. explicit tree communication only after the simpler versions are
   understood.

Node identifiers and report order must be randomized or otherwise audited so
the network cannot learn target position as a shortcut. Out-of-distribution
evaluation requires held-out semantic templates or structural regimes, not a
random row split.

Training should use observed final outcomes or action-value estimates, not
the actions chosen by an LLM teacher. The purpose is to predict useful
interventions, not imitate a workflow.

## Relation to `(verb, target)` control

This proposal temporarily removes the difficult joint `(verb, target)`
decision. It does not claim that the larger set of choices is permanently
useless.

The staged test is:

1. fix one verb and test target choice;
2. establish stable target-specific value;
3. test whether the visible information predicts that value; and
4. only then consider multiple verbs or multi-step control.

If target choice fails under one fixed intervention, a larger joint action
space is unlikely to rescue the approach. If it succeeds, later work can score
each legal `(verb, target)` pair directly rather than first choosing a verb
and then a target.

## How the current decomposition results change the decision

Use the results on newly generated original problems to decide what to do:

- If decomposition and reports do not beat compute-matched direct prompting,
  retire the targeted-revision line for that model and task family.
- If exact reports help but natural reports do not, the practical bottleneck
  is worker quality. A revision oracle may be relevant, but only if revising a
  natural report often repairs it.
- If reports help only when their use is forced, report adherence is the main
  mechanism. Treat targeted revision under forced use as a limited mechanism
  study.
- If natural reports help on average but uneven report quality causes
  failures, the targeted-revision oracle is strongly motivated.
- If natural reports are already almost always correct, there may be too
  little room for revision to help. Use a harder difficulty chosen in a
  written plan rather than
  selecting failures after the fact.

## Novelty boundary

The following are not new claims:

- decomposition can help language models;
- worker reports can be wrong;
- repairing wrong information can improve an answer;
- text embeddings can feed a small policy; and
- hindsight action selection is an upper bound.

A possible contribution would be the disciplined combination of:

- a complete table showing the result of revising every target from identical
  saved reports;
- exact final-answer grading;
- a clear separation between whether useful choices exist, whether their
  advantage repeats, whether the needed information is visible, and whether a
  network can learn the choice;
- strong fixed, LLM, and compute-matched controls;
- natural rather than postselected report errors; and
- evaluation on new problem types of whether the visible meaning predicts
  which revision will help.

Prior-art review is required before making a novelty claim. The oracle result
should decide whether that review and controller implementation are worth the
effort.

## Instructions for the receiving session

1. Do not modify or enlarge the experiment currently being prepared. That
   experiment will test fixed three-part decompositions on many newly
   generated original problems.
2. Record this as a proposed or queued follow-up question using the next
   repository-assigned research-question identifier.
3. Preserve natural worker reports and stable original-problem, subtask,
   prompt, and response identifiers in the current study so every later
   revision choice can start from the same saved reports.
4. Finish and analyze the current study before writing the targeted-revision
   experiment plan.
5. If those results support the follow-up, first write and run a plan that
   tests only the oracle. Do not include controller training in the same
   experiment.
6. Do not start model calls, paid work, or sealed generation without the
   repository's normal exact plan, provenance, and approval process.
7. Keep all final correctness labels anchored to exact task code.

## Decisions for the later written plan

The receiving session must decide, before inference:

- the worker and synthesizer model;
- the fixed model revision and decoding settings;
- the synthesis instruction justified by the current decomposition results;
- the exact revision prompt;
- whether `KEEP_REPORTS` is one of the choices or only a cost comparison;
- the number of newly generated original problems and repetitions;
- the exact calculations and uncertainty method;
- the minimum worthwhile oracle improvement;
- the compute-matched direct comparison;
- which familiar and new structural patterns belong to training and
  evaluation; and
- the exact cost ceiling and stopping behavior.

These decisions should not be made from targeted-revision outcomes.

## Recommended immediate disposition

The separate `structured-decomposition-benchmark` session is currently
preparing the larger experiment on newly generated original problems. Let
that work continue.

For now:

- save this handoff in both repositories;
- mark targeted report revision as a possible later experiment;
- do not implement a controller in `expert-iteration`; and
- revisit this proposal after the current decomposition results are
  available.
