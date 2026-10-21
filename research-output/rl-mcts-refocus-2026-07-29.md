# RL and MCTS Refocus: Prior Art and a Tractable Research Question

Date: 2026-07-29

## Executive Decision

`expert-iteration` should not return to the broad claim that a small learned
controller can orchestrate a frozen LLM. That claim is now directly occupied,
and this repository has already shown that a controller cannot recover value
that the action space does not contain or that the observable state does not
identify.

There is, however, a narrower and more defensible RL-first question:

> When can the value found by expensive search over a stochastic frozen-LLM
> harness be learned by a cheap observable-state policy, and when should that
> policy defer to search?

The contribution would be a measurement and diagnosis of search
amortizability, not a new claim that MCTS can generate training traces.
MCTS is an additive data-collection and fallback mechanism. The learned value
or policy model remains the central object.

This looks tractable for one researcher if it begins with one exact-verifiable
domain, a small established action space, and explicit gates before policy
training. It is plausibly an empirical or workshop paper. A main-conference
claim would probably require replication across at least two materially
different domains and a result that is more informative than "distillation
works."

## Why the Broad Versions Are Occupied

Several nearby claims now have direct prior art:

| Broad claim | Direct collision |
| --- | --- |
| A small external policy controls a frozen LLM harness | [Learning to Control LLM Agent Harnesses with Offline Reinforcement Learning](https://arxiv.org/abs/2607.05458) and [A Control System, a Dataset, and a Recipe for Making Frozen LLM Agents Learn a Domain](https://arxiv.org/abs/2607.25415) |
| A learned high-level planner selects reasoning actions for a frozen executor | [PG-HAP](https://aclanthology.org/2026.findings-acl.2024.pdf), [RLoT](https://arxiv.org/abs/2505.14140), [ACTS](https://arxiv.org/abs/2606.03965), and [CoRefine](https://arxiv.org/abs/2602.08948) |
| MCTS finds workflows that are distilled into a compact policy | [SQLConductor](https://arxiv.org/abs/2606.23537) |
| Search traces train a better reasoner or policy | [TS-LLM](https://arxiv.org/abs/2309.17179), [ReST-MCTS*](https://arxiv.org/abs/2406.03816), [AlphaLLM-CPL](https://arxiv.org/abs/2410.06508), [TreeRL](https://arxiv.org/abs/2506.11902), and [InfoTree](https://arxiv.org/abs/2605.05262) |
| An optimizer discovers prompts, workflows, or harness code | [AFlow](https://arxiv.org/abs/2410.10762), [ADAS](https://arxiv.org/abs/2408.08435), [GEPA](https://arxiv.org/abs/2507.19457), [Meta-Harness](https://arxiv.org/abs/2603.28052), and [AutoTTS](https://arxiv.org/abs/2605.08083) |
| A router selectively allocates inference-time computation | [Adaptive Test-Time Compute Allocation](https://arxiv.org/abs/2604.14853), [Select-then-Solve](https://arxiv.org/abs/2604.06753), and [Budget-Aware Tree Search](https://arxiv.org/abs/2603.12634) |

Therefore, neither "MCTS plus RL" nor "search then distill" is a sufficient
novelty claim. The open question must concern a failure mode or scientific
property that those pipelines do not isolate.

## Where the Current Excitement Is

### 1. Tree-structured RL data, not just tree search at inference

Recent work uses branching rollouts to obtain more informative comparisons,
better credit assignment, or more diverse training batches. Examples include
[Tree-GRPO](https://arxiv.org/abs/2509.21240) and
[InfoTree](https://arxiv.org/abs/2605.05262). This is attractive because the
tree exposes counterfactual decisions that a single successful trajectory
does not.

The expensive version trains or fine-tunes an LLM. That is not a good
side-project target. The tractable version asks whether the same tree
information can supervise a small external value or policy model.

### 2. Automatic optimization of the agent harness

Systems such as [GEPA](https://arxiv.org/abs/2507.19457),
[Meta-Harness](https://arxiv.org/abs/2603.28052), and
[AutoTTS](https://arxiv.org/abs/2605.08083) search prompts, programs, or
controller logic rather than merely sample more answers. This area is active,
but it moves quickly and makes causal attribution difficult. A new general
harness optimizer would be a poor target for a small secondary project.

### 3. Adaptive inference-time computation

Another active line learns when to spend more calls or tokens. It includes
selective reasoning, search budgets, and early stopping. This is practically
important but already crowded. A generic "learn when to search" contribution
would be weak unless grounded in a sharper account of what makes search value
predictable.

### 4. Search amortization

Search can produce strong decisions, but repeatedly running it is expensive.
Distilling those decisions is an old and active idea, from
[Expert Iteration](https://arxiv.org/abs/1705.08439) through
[Searchformer](https://arxiv.org/abs/2402.14083),
[Stream of Search](https://arxiv.org/abs/2404.03683), and SQLConductor.

The less studied issue is how much apparent search value survives:

1. repeated stochastic execution of the selected action,
2. reconstruction from only the state available to the policy,
3. a fresh search batch,
4. deployment on held-out tasks and fresh random seeds.

That sequence is the most promising opening for this repository.

## The Recommended Research Question

### Main question

> Under stochastic frozen-LLM transitions and exact terminal verification,
> which MCTS-derived supervision targets represent repeatable action value,
> and how much of that value can an observable-state policy retain on fresh
> rollouts?

### Why this is not merely another distillation paper

MCTS may select an action because one sampled continuation happened to
succeed. A controller trained on that choice is asked to reproduce the
decision without access to the fortunate future sample. There are four
different quantities that are often conflated:

1. **Realized search outcome:** the reward of the best continuation that
   search happened to observe.
2. **Expected action value:** the average reward from repeatedly taking an
   action at the same state and resampling the future.
3. **Observable-state value:** the best value attainable by a policy that
   only sees the recorded controller state.
4. **Deployed policy value:** the value retained by the learned controller on
   held-out tasks and new random seeds.

The gaps between these quantities locate the actual bottleneck:

- A large gap from realized outcome to expected action value means search
  mostly found lucky continuations.
- A large gap from expected action value to observable-state value means the
  state hides information needed to choose well.
- A large gap from observable-state value to deployed policy value means the
  model, targets, or training procedure failed to learn the available rule.

[SQLConductor](https://arxiv.org/abs/2606.23537) already recognizes unstable
success and uses a Beta-smoothed workflow stability estimate. The defensible
new contribution is therefore not "stability-aware distillation." It is a
systematic decomposition based on balanced, repeated same-state action
rollouts, independent target-estimation batches, and fresh-seed deployment.

The closest related work on stochastic search uncertainty includes
[PlanU](https://arxiv.org/abs/2510.18442) and
[Stochastic-Power-UCT](https://arxiv.org/abs/2406.02235). These papers raise
the bar for claiming a new tree-search estimator. The proposed paper should
instead treat the estimators as experimental conditions and make
amortizability the object of study.

[BiPACE](https://arxiv.org/abs/2606.25556) studies state-action credit mismatch
and estimates local counterfactual advantages. [Causal Agent Replay](https://arxiv.org/abs/2606.08275)
re-executes stochastic agent trajectories under interventions for failure
attribution. These are close enough that the paper must not claim the first
action-specific counterfactual credit method or the first use of stochastic
re-execution. Its distinct target is the validity and deployability of
search-derived supervision.

## Stochastic Search Is the Central Technical Issue

If action \(a\) at state \(s\) calls a frozen LLM, its result is sampled from
\(P(s' \mid s,a)\). Different sampled completions are outcomes of the same
action, not separate deterministic actions available to the deployed
controller.

A mean backup estimates the expected value of taking that action once. A max
backup, or a label that becomes positive whenever any descendant succeeds,
estimates the value of a favorable realization found through repeated
sampling. The latter grows with the search budget and may not be realizable by
a controller that executes the selected action only once.

A max or coverage target is valid only when the deployed macro-action itself
includes repeated attempts and a reliable selector. Otherwise, it gives the
controller a clairvoyant training target. UCB optimism may still be useful for
exploration, but it must not silently become the policy target.

This issue also strengthens the required baselines.
[AB-MCTS](https://arxiv.org/abs/2503.04412) already compares repeated sampling,
sequential refinement, standard MCTS, and adaptive MCTS under matched
generation-call budgets. A new study must go beyond a calls-only comparison.
It should separately report generated tokens, input tokens, verifier or tool
calls, latency, and monetary cost.

## Formal Measurements

Let:

- \(V_{\mathrm{fixed}}\) be the value of the best fixed workflow.
- \(V_{\mathrm{BoN}}\) be compute-matched best-of-N or reranking value.
- \(V_{\mathrm{search}}\) be full MCTS value on fresh evaluation rollouts.
- \(V_{\mathrm{policy}}\) be the learned controller's fresh-rollout value.
- \(V_{\mathrm{policy+search}}\) be the value of a controller that defers to
  search when uncertain.

Define:

\[
H_{\mathrm{search}} =
V_{\mathrm{search}} -
\max(V_{\mathrm{fixed}}, V_{\mathrm{BoN}})
\]

as search headroom, and:

\[
G_{\mathrm{amort}} =
V_{\mathrm{search}} - V_{\mathrm{policy}}
\]

as the amortization gap.

When search headroom is positive, the fraction retained by the policy is:

\[
R_{\mathrm{retained}} =
\frac{
V_{\mathrm{policy}} -
\max(V_{\mathrm{fixed}}, V_{\mathrm{BoN}})
}{
V_{\mathrm{search}} -
\max(V_{\mathrm{fixed}}, V_{\mathrm{BoN}})
}
\]

These values must be estimated on fresh seeds. Computing them from the same
trajectories that generated the MCTS targets would overstate both search
headroom and retention.

## Go/No-Go Gates Before Controller Training

The old project trained controllers before establishing that the policy had
something useful and observable to learn. The new sequence should reverse
that order.

### Gate A: Search headroom

Under the same call or token budget, MCTS must beat:

- the strongest fixed workflow,
- best-of-N or self-consistency,
- simple breadth-first or reranking strategies.

If it does not, stop. A learned controller cannot amortize an advantage that
search itself does not possess.

### Gate B: Repeatable action differences

At replayable states, run every legal action multiple times with independently
sampled continuations. The best action must vary across states, and the
ranking must reproduce in a second independent batch.

If one action or workflow dominates, use it as a fixed policy. If action
rankings do not reproduce, search labels are too noisy to be useful policy
targets.

### Gate C: State observability

The full semantic state must predict repeated action-value differences beyond:

- workflow phase,
- legal-action menu,
- action history,
- topology,
- node identifiers and ordering,
- remaining budget.

Compare full state against those structural features, shuffled semantic
content, and a capable prompted LLM controller. If semantics add no held-out
predictive value, redesign the state or abandon adaptive control.

### Gate D: Cheap learnability

Only after Gates A through C pass should a small controller be trained. It
must outperform the best structural lookup or fixed policy on held-out tasks.
Otherwise, the available relationship may be too sample-inefficient or
complex for the intended controller.

## Minimum Credible Experiment

### Domain

Begin with one exact-verifiable domain. Small code repair is a plausible
candidate because tests provide exact terminal reward, partial artifacts are
meaningful, and alternative repair workflows can be replayed. Before adopting
it, confirm that states can be serialized and replayed without hidden tool or
environment differences.

A second domain should only be added after the protocol works. It is likely
necessary for a strong general claim, especially if the first domain produces
a negative result.

### Action space

Use three to five domain-grounded macro-actions with clear operational
semantics and a horizon of roughly three to five decisions. Prefer an action
space from an established agent framework or a published study.

Do not begin with:

- an open-ended universal verb ontology,
- arbitrary tree-node targeting,
- actions that differ only in prompt wording,
- actions whose effects cannot be replayed,
- a menu where common sense already implies one dominant ordering.

Every action should have a plausible state in which it helps and another in
which it wastes budget or causes harm. A short pilot must demonstrate those
bidirectional tradeoffs empirically.

### Data collection

1. Run MCTS to collect candidate states and search statistics.
2. Freeze a stratified set of replayable states.
3. From every state, execute every legal action with balanced repeated
   continuations.
4. Repeat target estimation with an independent random-seed batch.
5. Hold out tasks, not merely trajectories, for deployment evaluation.

Adaptive MCTS samples are selection-biased. Balanced resampling at the frozen
states is essential for estimating expected \(Q(s,a)\).

The tree should represent LLM sampling as chance outcomes under an action, or
attach repeated outcome samples to the same action edge. It should not treat
each sampled completion as though the controller could knowingly select it in
advance.

### Supervision targets

Train identical controller architectures against:

- the single MCTS-selected action,
- normalized MCTS visit counts,
- mean repeated \(Q(s,a)\),
- a conservative lower-confidence value,
- SQLConductor-style workflow stability,
- no-semantic structural lookup.

This is an ablation over targets, not a proposal for a new MCTS algorithm.

### Deployment conditions

Compare:

- direct frozen-LLM execution,
- best fixed workflow,
- compute-matched best-of-N,
- capable prompted LLM controller,
- full MCTS,
- learned controller,
- learned controller with uncertainty-triggered MCTS fallback.

Report reward, calls, tokens, wall-clock time, verifier invocations, and
variance across seeds. Evaluate both the original frozen executor and at least
one executor shift if the budget permits.

## Ranked Project Options

| Rank | Project | Novelty outlook | Tractability | Decision |
| --- | --- | --- | --- | --- |
| 1 | Search-amortizability audit with repeated same-state action values | Medium, if the oracle-gap decomposition is central and evaluated beyond one narrow setting | High | Pursue as the first screen |
| 2 | Audit whether a learned high-level planner uses semantic state rather than phase/menu structure | Medium | Very high | Strong fallback; coordinate with `llm-intervention-value` to avoid duplicate framing |
| 3 | Tree-based informative rollout collection for an external value model | Medium-low because InfoTree, Tree-GRPO, and SQLConductor are close | Medium | Consider only after a positive audit |
| 4 | New general MCTS-to-policy orchestration algorithm | Low | Medium | Reject |
| 5 | General automatic harness or workflow evolution | Low in a fast-moving field | Low for a secondary project | Reject |
| 6 | Full RL fine-tuning of a reasoning LLM | Potentially high impact but heavily occupied | Low for one researcher without substantial compute | Reject for this project |

## What a Positive and Negative Result Would Mean

### Positive

Search has reproducible headroom, semantic state predicts the action-value
differences, and a cheap controller retains a substantial part of the gain.
The paper can characterize which supervision targets and uncertainty
conditions preserve that value.

### Search helps, but policy does not

The result isolates an amortization or observability barrier. This is useful
if the repeated-action protocol rules out lucky search outcomes and the
failure reproduces across domains or executor scales.

### Search itself does not help

The action space is inadequate, the search evaluator is inadequate, or the
baseline already captures the useful workflow. Stop before policy training.
This is a valid kill decision but probably not a standalone paper from one
domain.

### Fixed policy matches search

There is no useful routing problem in that setting. The simplest deployment
is the fixed workflow. This directly mirrors the lesson from E40 and E41 and
must be treated as a scientific result rather than a controller-training
failure.

## Relationship to This Repository

`expert-iteration` can remain the implementation testbed for:

- verifier-anchored terminal reward,
- replayable controller states,
- RL policy and value heads,
- MCTS collection and optional deployment fallback.

The forest state and unrestricted `(verb, target)` interface should not be
presumed correct. The first study should use a smaller, domain-grounded,
empirically screened interface. If that succeeds, later work can test whether
the tree representation adds value.

The existing Phase B behavior-cloning implementation is infrastructure, not
the immediate scientific priority. The next artifact should be a
pre-registration for Gates A through C. No paid controller training should
start until those gates pass.

## Novelty Claim to Use Carefully

A safe provisional claim is:

> Prior search-to-policy work has used search traces and stability heuristics
> to train controllers. We study a different question: how apparent search
> gain decomposes into lucky realized outcomes, repeatable same-state expected
> action value, information recoverable from the controller's observable
> state, and value retained by a deployed policy on fresh stochastic
> rollouts.

This remains provisional until the protocol and final paper search are
complete. Avoid claiming the first stochastic MCTS distillation method, the
first stability-aware controller, or the first learned controller for a
frozen LLM.

## Primary Sources

### Direct controller and planner precedents

- [Learning to Control LLM Agent Harnesses with Offline Reinforcement Learning](https://arxiv.org/abs/2607.05458)
- [A Control System, a Dataset, and a Recipe for Making Frozen LLM Agents Learn a Domain](https://arxiv.org/abs/2607.25415)
- [Context Assembly as the Controlled Variable](https://arxiv.org/abs/2607.25408)
- [PG-HAP](https://aclanthology.org/2026.findings-acl.2024.pdf)
- [RLoT](https://arxiv.org/abs/2505.14140)
- [ACTS](https://arxiv.org/abs/2606.03965)
- [CoRefine](https://arxiv.org/abs/2602.08948)

### Search, RL, and distillation

- [Expert Iteration](https://arxiv.org/abs/1705.08439)
- [SAVE: Combining Q-Learning and Search with Amortized Value Estimates](https://arxiv.org/abs/1912.02807)
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601)
- [RAP](https://arxiv.org/abs/2305.14992)
- [LATS](https://arxiv.org/abs/2310.04406)
- [TS-LLM](https://arxiv.org/abs/2309.17179)
- [ReST-MCTS*](https://arxiv.org/abs/2406.03816)
- [Searchformer](https://arxiv.org/abs/2402.14083)
- [Stream of Search](https://arxiv.org/abs/2404.03683)
- [SQLConductor](https://arxiv.org/abs/2606.23537)
- [TreeRL](https://arxiv.org/abs/2506.11902)
- [Tree-GRPO](https://arxiv.org/abs/2509.21240)
- [InfoTree](https://arxiv.org/abs/2605.05262)
- [AB-MCTS](https://arxiv.org/abs/2503.04412)

### Limits and required baselines

- [When is Tree Search Useful for LLM Planning? It Depends on the Discriminator](https://arxiv.org/abs/2402.10890)
- [Limits of PRM-Guided Tree Search](https://arxiv.org/abs/2510.20272)
- [Lessons of Developing Process Reward Models](https://arxiv.org/abs/2501.07301)
- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171)
- [s1: Simple Test-Time Scaling](https://arxiv.org/abs/2501.19393)
- [PlanU](https://arxiv.org/abs/2510.18442)
- [Stochastic-Power-UCT](https://arxiv.org/abs/2406.02235)
- [BiPACE](https://arxiv.org/abs/2606.25556)
- [Causal Agent Replay](https://arxiv.org/abs/2606.08275)

### Workflow and harness optimization

- [DSPy](https://arxiv.org/abs/2310.03714)
- [AFlow](https://arxiv.org/abs/2410.10762)
- [ADAS](https://arxiv.org/abs/2408.08435)
- [GEPA](https://arxiv.org/abs/2507.19457)
- [Meta-Harness](https://arxiv.org/abs/2603.28052)
- [AutoTTS](https://arxiv.org/abs/2605.08083)
