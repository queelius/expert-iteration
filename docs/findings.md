# Findings and Lessons

These findings concern particular models, prompts, and saved problems. They
are not three independent confirmations of one proposed method.

The [generated tables](../analysis/tables.md) and
[input manifest](../analysis/claims.json) let readers check the reported
counts. The archive analysis recounts record-level outcomes. It does not
repeat inference or independently certify every historical verifier decision.

## 1. Copying a Teacher Did Not Produce a Useful Transferred Controller

**Question:** Would a small network trained to copy a teacher's decisions
successfully coordinate a different worker?

In E20, direct Qwen2.5-1.5B prompting answered **8 of 20** problems correctly.
The trained imitation controller with that worker solved **0 of 20**.
The reconstructed comparison uses the same original problems in both arms.

The controller had learned from another generator's trajectories. This is a
small, single-seed transfer test, not evidence that small controllers cannot
work. It also is not a test of the planned off-policy RL system: the fitted
Q head did not choose actions and no off-policy Q-learning update had run.

**Lesson:** Before collecting more teacher data, check whether the copied
policy can handle the states produced by the worker it will actually control.

Reproduction: table `e20`, from direct-answer records and per-turn controller
rewards checked against episode summaries.

## 2. A Better Routing Prompt Still Trailed a Fixed Revision

**Question:** Did the coordinator choose useful additional work better than a
simple fixed workflow?

E41 and E42 reused **60 original problems** and shared saved candidate answers.
The worker was Qwen2.5-3B; the coordinator was GPT-4.1-mini.

| Condition | Correct answers or candidates |
| --- | ---: |
| Saved initial answer | 43/60 |
| Fixed revision with the retained prompt | 49/60 |
| Fixed revision with an independent-solution prompt | 55/60 |
| Generic coordinator routing | 46/60 |
| Explicit coordinator routing | 51/60 |

The revised routing instructions helped in this comparison. They still missed
four of the correct candidates available from always taking the independent
revision.

E42 replayed E41 worker outputs, rather than making fresh worker calls. The
fixed workflow revised every problem; the explicit router chose revision on
19 problems and also incurred coordinator calls. Thus this is **not an
equal-cost comparison**. E41 measures candidate correctness, not an executed
terminal submission. These development comparisons do not establish fresh
end-to-end efficacy or efficiency.

**Lesson:** Compare a coordinator against the strongest simple workflow you
can specify, and separately account for extra calls, tokens, and latency.
An accuracy advantage over direct prompting alone is not enough.

Reproduction: `e41_revision`, `e41_baseline`, `e42_routing`, and `e42_fixed`.

## 3. Focused Context Helped, but Target Selection Was Not Established

**Question:** Does showing a worker only the selected subproblem's context
help, and can earlier trials identify which subproblem to choose next?

E47 used **12 source states**, with repeated calls for several targets and
context choices. On the held evaluation repetitions, child-local context
produced **33 correct outcomes in 72 paired observations**, compared with
**24 in 72** for the same targets with full context.

Those are not 72 independent problems. They are repeated observations within
12 states, drawn from a development screen.

A fixed child target selected using the earlier discovery repetitions tied
direct problem-only prompting: **10/24 versus 10/24**, across the same 12
states. Choosing a target separately for each state using those earlier
outcomes reached **11/24**, versus the fixed child's **10/24**. That did not
meet the original repeatability and improvement requirements.

The original grouped uncertainty and stopping rules remain in the experiment
ledger. These reconstructed tables are descriptive; they introduce no new
significance claim. Discovery and evaluation used different repetitions of
the same states, not new independently sampled problems.

**Lesson:** Removing distracting context can help without showing that a
state-aware controller can reliably choose a useful action. Test those claims
separately. This result does not establish a decomposition advantage.

Reproduction: `e47_locality`, `e47_fixed`, and `e47_adaptive`.

## What We Are Not Claiming

We did not establish that learned orchestration is generally ineffective, that
small networks cannot coordinate larger ones, or that the original expert
iteration idea fails. We also did not complete the proposed study of RL
sample efficiency with search assistance.

Quarantined experiments, partial runs, and confounded larger-model probes are
not included in these tables. Selection and exclusions are documented in
[release readiness](release-readiness.md), not silently discarded.
