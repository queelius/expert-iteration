---
title: "Verifier-Anchored RL: A Learned Controller Orchestrating a Frozen LLM over a Structured Reasoning State"
author: "Alexander Towell"
date: "2026-06-30"
geometry: margin=1in
fontsize: 11pt
---

# 0. What this document is

Archive notice (2026-10-20): this is a historical design specification, not
a statement that all proposed work was completed. See
`docs/research-overview.md` for the implemented/tested/proposed distinction
and `docs/release-readiness.md` for current archive preparation.

This is the self-contained design of the system: a small external controller
network that orchestrates a frozen large language model (LLM) by issuing typed
operations over a highly structured, nested reasoning state, with a symbolic
verifier as the only source of reward. Section 2 develops the small amount of
reinforcement-learning (RL) theory the rest of the document relies on, so no
outside reading is needed.

The framing discipline is RL first, search second. The central object is an
agent (a policy and a Q function) learning over an induced context Markov
decision process (MDP) whose only reward source is a symbolic verifier. Monte
Carlo Tree Search (MCTS) is a sample-efficient policy-improvement operator and a
test-time compute knob (Sections 7.4 and 8), not the central object. The core
training story stands on plain RL.

## 0.1 Contract status after the Phase B audit

Sections 3 and 4 retain v1 examples for historical continuity. Where their
older details conflict with this paragraph, the audited v2 contract in
`docs/superpowers/specs/2026-07-13-state-action-contract-v2.md` governs.

The corrections are structural. Controller state is a versioned graph-derived
record shared by teacher collection, replay, and live inference. Trusted givens
are non-committable Evidence. A generated Claim has an owner goal, a typed
symbolic contract, explicit premises and payload, terminal verification status
`unchecked -> verified | rejected`, and a separate symbolic owner-discharge
status. Active and compact Claims retain both receipts, bound to one pinned
verifier contract and bounded reason-code sets. The casework policy learns only
DECOMPOSE, direct PROPOSE_RESULT,
PROPOSE_SOLUTION, and PROPOSE_REFUTATION over objective-appropriate Goals.
VERIFY, COMMIT, and ready AGGREGATE run as deterministic closure after one
learned decision, consume no extra controller turn, and create no BC example.
Each closure event names the exact Claim it verifies, creates, or commits. A
solved compacted goal retains that exact committed outcome. DERIVE and root-safe
PRUNE remain generic typed operations but are absent from the initial casework
policy menu. Greedy deployment maximizes the factored joint probability over
all learned legal pairs.

The live v2 macro transition replays the learned graph delta and deterministic
closure under the pinned contracts before an accepted receipt or reward is
trusted. Casework decomposition is always exhaustive-case decomposition. Under
`symbolic-only-v1`, verifier-backed root closure receives +1 and every other
transition, including horizon expiry, receives 0.

The bounded previous action outcome is part of immutable MDP state, not a
runner-local prompt side channel. The implemented collection boundary binds
the exact environment semantics, model identities, secret-free transport
semantics, retry and seed policies, controller behavior source and fallback
rule, policy-input and output contracts, and trusted problem source. A bound
controller owns action selection. Explicit action stepping is a semantic test
path and cannot produce canonical collection provenance.

Canonical rows carry both environment and collection digests and the exact
typed learned effect. Strict replay starts from a contract-bound trusted
problem resolver, never from row-authored givens or obligations, and reproduces
every learned effect and deterministic closure without an LLM. Hosted model
calls are retained in a role-bound append-only diagnostic sidecar. A teacher
label is eligible for behavior cloning only when replay links the exact
controller request and decoded choice, generator request and decoded effect,
retry transcript, seeds, and receipt to the canonical row.

The v2 checkpoint manifest binds the full training environment and collection
provenance, feature and encoder algorithms, target and decoding contracts,
dataset and split digests, every shape-changing model field, and a
value-sensitive state-dictionary digest. Loading separately validates the
checkpoint-backed learned controller and compatible frozen generator used at
deployment. A dimension-preserving serializer change, same-shape weight swap,
or live transition-semantic mismatch is a load error.

The operational Phase B path preserves those bindings end to end. The
casework source discards answer-bearing fields and digests the exact ordered
selected projection. Collection has a no-network preparation mode that
resolves source, environment, and collection digests before preregistration.
Hosted execution requires those expected digests, retains incomplete work only
under explicit `.partial` paths, and exposes final artifacts only after raw-call
diagnostics, exact symbolic replay, and full ordered source coverage agree.
Strict replay seals the only input accepted by the v2 dataset converter. The
converter stores current and next canonical observations, their structured
features and legal masks, action indices, behavior provenance, symbolic reward,
terminal flag, and full-episode Monte Carlo return. A deterministic grouped
problem split is sealed against that dataset digest. The rung-2 trainer applies
factored behavior cloning to every provenance-eligible teacher decision and
selected-action Q regression to every training transition, then creates the
exact manifest-bound checkpoint. The required content-free comparator uses only
the exact ordered legal menu and never success-filters teacher labels.

The generator request contract is independently versioned. The current lens is
`typed-claim-lens-v2.1`; exact `typed-claim-lens-v2` request bytes remain
available for legacy replay, including reconstruction of all 66 E23 generator
requests. E23 passed a five-problem hosted integration gate, but its artifact is
not training data. A separator-only diagnostic kept interval topology and
symbolic scoring in trusted code while requiring exact JSON string boundaries.
Under the fixed one-retry protocol, 8/33 logical cases ended typed and all 25
schema retries exhausted. The string-valued boundary wire is therefore not
viable for this generator. This is GenerationContract evidence, not controller
performance, reward evidence, a trajectory, or training data; no boundary path
is integrated.

These gates establish an operational hosted boundary, but the representation
gate remains uncleared. No usable v2 training collection or v2 checkpoint
exists. Current canonical generator and teacher wire contracts are
`typed-policy-generation-v2.1` and `controller-choice-wire-v2.1`. Both bind
nonmutating strict validation, and live plus diagnostic replay paths derive the
mode from those role-local ids. The prior `typed-policy-generation-v2` and
`controller-choice-wire-v2` ids select legacy coercion only for exact replay.
The low-level client default remains legacy for compatibility; canonical paths
always pass the bound mode. Outer environment and collection record shapes and
historical hashes remain unchanged. The next representation design is a
separately versioned JSON-number boundary payload that preserves raw decimal
lexemes and converts them to exact rationals in trusted code without
binary-float conversion. It requires contract tests, proof obligations, and
preregistration before hosted execution. Controller exploration,
target-network Bellman updates, and gated Q-to-policy distillation remain the
next RL implementation stage after a training collection clears that gate.
The full repository suite has 931 passing tests and 1 skipped CUDA-only test.

These corrections do not change the reward invariant or the framing: the
symbolic verifier remains the only reward source, RL remains the spine, and
MCTS remains strictly additive.

## 0.2 Current permissive-workspace experimental path

As of 2026-07-17, the immediate empirical path is the separately versioned
contract in
`docs/superpowers/specs/2026-07-14-permissive-workspace-v1.md`. This is a new
mechanism test, not a retroactive explanation of the earlier controller
results. The question is whether a focused external workspace can make a
frozen small worker more useful than its exact direct-prompt path before a
learned controller is trained.

The state contains natural-language Goal nodes, unverified Candidate nodes,
complete action history, and a finite turn budget. The action is still a typed
pair `(verb, target)`, with `DECOMPOSE`, `ANSWER`, `SYNTHESIZE`, `REVISE`, and
`SUBMIT`. A controller receives the full canonical workspace and exact legal
menu. A frozen worker receives an action-specific hybrid lens containing the
root problem, selected target, ancestry, relevant local work, exact prior
attempts, and remaining budget. Generated children and candidates are admitted
as unverified content after bounded container checks. Admission supplies no
reward and makes no truth claim. `SUBMIT` is root-only, calls no LLM, terminates
the episode, and receives binary reward only from the symbolic answer checker.

Child `ANSWER` has an explicit run-bound context mode. `target-only-v1` keeps
the local target view. `preceding-siblings-v1` adds only completed earlier
sibling Goals and Candidates under the immediate parent's ancestry record.
Incomplete and later work remains absent, and initial root `ANSWER` remains the
exact direct request.

Direct recoverability is a contract invariant: initial `ANSWER(root)` uses the
same request as the raw direct baseline. E31 therefore tests the mechanism in
the smallest useful comparison, exact direct prompting versus a deterministic
decompose-answer-synthesize trace with the same 7B worker. The fixed trace uses
more inference, so it can establish only whether focused scaffolding can buy
performance, not compute efficiency or controller value. E35 introduced a
strong LLM teacher with direct and fixed policies retained as controls.
Successful teacher trajectories would provide a behavior-cloning warm start
for the RL controller; symbolic terminal reward would continue to drive policy
and Q learning. MCTS remains an optional later accelerator.

This path supersedes E30 as the execution priority but does not erase the v2
implementation or its sealed artifacts. E31-E33 first screened the mechanism:
the exact direct 7B path solved 17/20, fixed target-only decomposition solved
9/20, and fixed preceding-sibling decomposition solved 12/20 with substantially
more inference. Provider drift made that cross-run lens comparison causally
inconclusive. E34 therefore branched both lens modes from the exact same 18
sealed DECOMPOSE plus first-child prefixes and alternated hosted branch order.
Preceding-sibling context solved 12/18 versus target-only's 9/18, with eight
both, four preceding-only, one target-only, and five neither. This confirms a
positive causal context mechanism on the reused development screen and retains
`preceding-siblings-v1` for the first permissive-workspace teacher experiment.
E35 then ran contemporaneous policies on all 60 project-held-out casework-v1
test problems. Exact direct solved 59/60, fixed decomposition solved 27/60,
and the GPT-4.1-mini teacher solved 31/60. Teacher/direct paired cells were 31
both, zero teacher-only, 28 direct-only, and one neither. Teacher/fixed cells
were 24 both, seven teacher-only, three fixed-only, and 26 neither. No E35
trajectory is eligible for behavior cloning.

E35 changes the next state design. The teacher selected initial root ANSWER on
12 problems and solved all 12, but selected root DECOMPOSE on 48 and solved
only 19. Direct solved 47 of those same 48. Asking a controller with no
worker-specific transition experience to choose whether to obtain the direct
baseline before it has seen any worker output is therefore the wrong decision
boundary. The next version should generate the exact direct Candidate as a
shared deterministic prefix, then let the controller inspect that Candidate
and choose SUBMIT, REVISE, or structured work. This makes the workspace an
improvement layer over the baseline rather than merely leaving the baseline as
one legal option.

Three other E35 findings remain separate versioned manipulations. Root
SYNTHESIZE, later root ANSWER, and root REVISE need the same verifier-ready
answer-field contract as the direct lens; 11 rejected teacher submissions were
the exact correct scalar wrapped only as `x = n`. Action availability should
expose active work instead of every answered Goal and stale Candidate, because
the legal menu grew to 27 choices. The teacher wire should return one menu
index, which trusted code maps to the internal `(verb, target)` pair, so an
invalid cross-pairing cannot be formed. These changes require separate tests
and experiments. The current 7B worker is also at 59/60 on casework-v1 test, so
that source cannot measure a positive teacher delta.

The local baseline-anchored workspace v2 implements those four factors as
independent run-bound modes. Under
`direct-candidate-prefix-v1`, exact direct `ANSWER(root)` runs before the first
controller decision and is logged as deterministic initialization rather than
a behavior-cloning label. `verifier-ready-root-answer-v1` changes only later
root-owned worker instructions. `active-frontier-v1` masks answered and stale
work while keeping the direct prefix Candidate submittable. The active
observation exposes that protected Candidate id explicitly. Finally,
`legal-menu-index-v1` maps one returned menu index to an internal legal action.
If teacher decoding fails, the bound fallback chooses exact direct ANSWER when
legal, then protected direct SUBMIT when legal, then the first ordered action.

The redesigned collector binds every mode, model identity, source digest,
hidden-label digest, sampling setting, prompt contract, and provenance role.
Prefix transitions cannot enter behavior cloning. Preparation is zero-network
and hosted execution requires the prepared digests.

E36 tested only the root result factor by forking 59 exact sealed E35
pre-SYNTHESIZE states. `verifier-ready-root-answer-v1` solved 49/59 versus
27/59 for `workspace-answer-free-v1`: 27 both, 22 verifier-ready-only, zero
legacy-only, and 10 neither. The exact one-sided discordant-pair probability is
`1 / 2^22`. All calls completed on their first attempt, the exact legacy branch
reproduced E35 reward on 59/59 states, and the only within-pair request change
was the registered instruction. This establishes a positive causal root-prompt
mechanism on the reused sealed source, so the verifier-ready contract is
retained. It does not show that decomposition or a teacher beats direct
prompting. The general redesigned collector still runs one arm at a time;
the separate paired root-result collector exists for that prompt factor.

E37 added the bound shared-direct-Candidate branch collector and evaluated the
complete redesigned teacher continuation with Qwen2.5-3B. Direct submission and
GPT-4.1-mini continuation both solved 43/60 reused development cases: 43 both,
zero discordances, and 17 neither. Every teacher path ultimately submitted the
protected direct Candidate, so anchoring prevented regression but recovered no
failure. The teacher selected additional work on seven of 17 direct failures
and four of 43 direct successes, making the generated Candidate a useful
post-hoc routing signal but not improving reward.

E38 isolated worker-side REVISE generation on those exact 14 teacher-selected
states. `workspace-revise-answer-container-v1` requires a top-level nonempty
string or finite numeric `answer`, permits optional unverified `reasoning`, and
rejects other top-level fields. This is a container contract, not a truth or
algebra gate. It produced 14/14 answer-bearing revised Candidates versus 0/14
for the exact legacy open-object control, with all 14 pairs treatment-only and
one-sided exact probability `1 / 2^14`. The treatment completed every call on
its first attempt; the control admitted five lens echoes without answers and
exhausted nine calls after a retry.

Container repair is not yet teacher efficacy. Across 10 unique root episodes
in this selected subset, the source Candidate and treatment revision were both
symbolically correct on three, only the revision was correct on one, only the
source was correct on zero, and neither was correct on six. No controller chose
or submitted these revisions. The next causal comparison must keep the E37
state, action menu, lenses, worker, GPT-4.1-mini controller, indexed controller
prompt, seeds, and source fixed while changing only the retained REVISE
contract. If terminal reward remains flat, compare coordinator candidate-
selection prompts on exact post-revision states, then compare coordinator model
capacity separately. E37 and E38 are evaluation-only and supply no behavior-
cloning rows.

E39 performed that retained-contract end-to-end comparison. Exact shared
direct submission and GPT-4.1-mini continuation each solved 43/60, with no
discordant terminal rewards. Unlike E37, every selected REVISE produced an
answer-bearing Candidate. In the 12 revised cases, the baseline and revision
were both symbolically correct on four, revision-only on three, baseline-only
on zero, and neither on five. Both Candidates remained legal SUBMIT targets at
every post-revision decision, but the generic planner prompt chose protected
Candidate 1 all 12 times. The available-Candidate oracle was therefore 46/60
while the realized teacher remained 43/60. This is a diagnostic upper bound,
not an alternate reward signal.

The next isolated factor is coordinator prompting on the exact sealed E39
post-revision states. One arm must replay the current global planner prompt and
one must use a versioned candidate-comparison prompt, with the coordinator
model, state, legal menu, Candidates, seeds, transport, and local symbolic
SUBMIT fixed. Worker generation is absent. Coordinator capacity follows as a
separate model-only factor. E39 is evaluation-only and supplies no behavior-
cloning rows.

E40 tested whether those 12 selected cases were broad enough for that prompt
factor by applying one fixed retained-contract REVISE to every exact E39
post-prefix State. All 60 revisions were usable. The baseline and revision
were both symbolically correct on 43 cases, revision-only on six, baseline-only
on zero, and neither on 11. Latest revision and the available-Candidate oracle
were therefore 49/60 versus 43/60 for protected baseline. This is promising
worker self-revision evidence on a reused development source, but E40 executed
no SUBMIT and no coordinator and is not a terminal-reward result.

The proposed candidate-comparison prompt experiment would be uninformative on
this corpus because always-latest already equals its oracle and no case tests
preservation. The next isolated generation factor replaces only the retained
REVISE instruction with an independent-solution instruction while keeping the
answer-container schema, worker, state, action, lens, seeds, source, and
verifier fixed. Coordinator prompting resumes only if the resulting sealed
corpus contains both baseline-only and revision-only cases. E40 is evaluation-
only and supplies no behavior-cloning rows.

E41 paired that independent instruction against the retained instruction on
the same exact 60 States. Retained revisions were symbolically correct on
49/60 and independent revisions on 55/60: 49 both, six independent-only, zero
retained-only, and five neither. The one-sided exact probability is `1 / 2^6`.
Against protected baseline, independent revision was correct on all 43 baseline
successes plus 12 baseline failures. It therefore again supplied no baseline-
only preservation case, but it established a positive worker-prompt factor.

The next isolated coordinator factor moves one decision earlier. From each
exact prefix State, generic planning and an explicit routing prompt choose
between the same legal actions. SUBMIT terminates on protected baseline.
REVISE replays the exact sealed E41 independent transition and follows a fixed
latest-Candidate SUBMIT, with no new worker generation. This isolates prompt
control of REVISE-versus-SUBMIT routing while retaining local symbolic terminal
reward as the sole outcome. E41 is evaluation-only and supplies no behavior-
cloning rows.

E42 performed that paired coordinator comparison. The generic planner prompt
realized 46/60 verifier-scored terminal successes; the explicit routing prompt
realized 51/60, with 46 both, five routing-only, zero generic-only, and nine
neither. The one-sided exact probability is `1 / 2^5`. Explicit routing chose
REVISE on 19 cases versus eight for generic, preserved all 43 protected-
baseline successes, and recovered 8/12 cases whose independent revision alone
was correct. Four correct sealed revisions remained unused, so routing reached
51/60 against a 55/60 fixed action-outcome oracle.

This establishes a positive coordinator-prompt mechanism with symbolic
terminal reward, but worker transitions were replayed rather than generated
live. The next system experiment combines the already isolated independent
worker prompt and explicit routing prompt in a live shared-prefix comparison
against exact direct submission, preferably on a fresh harder verifier-covered
source. Only after that live result should coordinator model capacity be
changed. E42 is evaluation-only and supplies no behavior-cloning rows.

---

# 1. Thesis, and what this is not

## 1.1 The thesis in one paragraph

We train a small external controller network (on the order of a few million
parameters: a verb head, a pointer head, and a Q head) that orchestrates a
frozen LLM. The controller never edits the LLM; the LLM is a frozen
forward-pass component that generates content and, optionally, encodes state.
Each action is a pair `(verb, target)`, a lens: it projects a small, focused
slice of the structured state into a prompting template, lets the frozen LLM
fill in content under a constrained schema, and writes the result back into the
neighborhood of one node. The state is not a flat growing transcript of tokens;
it is a structured, nested reasoning forest of goals, subgoals, hypotheses, and
verified facts. A symbolic verifier (SymPy on a canonicalized answer, a unit-test
runner, a proof checker) is the only thing that produces reward.

## 1.2 The contrast that motivates this document

A reader can mistake this for an o1-style or DeepSeek-R1-style reasoning model.
It is not, and the difference is the point. o1 and R1 apply RL to the LLM's own
weights so it emits a better chain of thought, token by token. Here the LLM is
frozen and the RL operates over a tiny external action space of typed operations
on a symbolic state.

| Axis | o1 / DeepSeek-R1 | This system |
| --- | --- | --- |
| What RL optimizes | the LLM's own weights | a small external controller; the LLM is frozen |
| Action granularity | the next token of a free-form chain of thought | one typed `(verb, target)` operation |
| State | a flat, growing context of tokens | a structured, nested reasoning forest |
| What the LLM sees per step | the entire running trace | a lens: one node's grounded-local slice |
| Reward source | learned reward model, preference, or verifiable reward over the trace | a symbolic verifier only: terminal verdict on the committed answer (plus optional gated `VERIFY` shaping, Section 6) |
| Cost of "thinking" at deploy | a large model emitting many tokens | a frozen small generator plus a tiny controller; optional search |

"Prompt an LLM to reason step by step" is exactly what this system does not do.
The LLM is a content-generation oracle that an external policy steers over a
symbolic state, and correctness is decided by a program, not by a model's
opinion.

## 1.3 Why the structure earns its place

A small model reasons poorly across a long, flat context but well over a short,
on-topic one. The lens exploits this: when the controller acts on a subgoal, the
LLM sees that subgoal's small neighborhood, not the entire history. The nested
state is the substrate that makes such focusing possible, and the verifier is
what keeps the learning signal honest.

---

# 2. Reinforcement-learning foundations

This section develops the minimum RL the rest of the document uses; its notation
recurs in Sections 4 to 7. A reader fluent in MDPs and Q-learning can skim it.

**MDP.** A discounted MDP is $(S, A, P, R, \gamma)$: states $S$; actions
$A(s)$ available in state $s$; transition probability $P(s' \mid s, a)$; immediate
reward $R(s, a, s')$; discount $\gamma \in [0, 1)$. The Markov assumption is that
the next state depends only on the current state and action, not the full
history. Here $s$ is the reasoning forest (Section 3), $a$ is a `(verb, target)`
pair (Section 4), $P$ is induced by the frozen LLM, and $R$ comes from the
verifier (Section 6). The state space here is discrete but far too large to
enumerate or tabulate; the expanded sums below need only discreteness.

**Bellman optimality.** The optimal action value satisfies, written first as an
explicit sum over next states:

$$
Q^\star(s, a) \;=\; \sum_{s'} P(s' \mid s, a)\,\big[\,R(s, a, s') + \gamma \max_{a'} Q^\star(s', a')\,\big].
$$

Take $a$ in $s$, average over next states using $P$, collect the immediate
reward, add the discounted best continuation. The greedy policy is
$\arg\max_a Q^\star(s, a)$; a stochastic one is
$\operatorname{softmax}_a(Q(s, a) / \tau)$ at temperature $\tau$.

A hand-computable instance, introduced here so the sum has something concrete to
bind to: four states, a start $S$ and terminals $G$, $A$, $B$. From $S$ there are
two actions. Action *safe* moves to $G$ with probability $1$ and reward $4$.
Action *risky* moves to $A$ with probability $0.8$ and reward $+10$, or to $B$
with probability $0.2$ and reward $-10$. Terminal states have value $0$, so the
explicit sum gives $Q^\star(S, \text{safe}) = 1.0\,(4 + \gamma \cdot 0) = 4$ and
$Q^\star(S, \text{risky}) = 0.8\,(10 + \gamma \cdot 0) + 0.2\,(-10 + \gamma \cdot 0) = 6$:
so $V^\star(S) = 6$ and the greedy policy takes the risky action. The value of an
action is its probability-weighted average outcome, not its best or worst case.
This tiny MDP doubles as ground truth for the classical validation anchors of
Section 10.

**Q-learning from samples.** $P$ cannot be enumerated, since it integrates over
the LLM's continuations, so we sample a transition $(s, a, r, s')$ and form the
target $y = r + \gamma \max_{a'} Q(s', a')$, then move $Q(s, a)$ toward $y$. If
$s'$ is terminal (in this system: a root COMMIT has ended the episode, so
$A(s')$ is empty), the continuation value is defined to be zero and the target is
$y = r$; the terminal flag is the structural condition of Section 4.2 (a COMMIT
whose target is the root), not a learned predicate. Q-learning is **off-policy**:
it learns the optimal $Q$ from experience generated by any adequately-covering
behavior policy (random, epsilon-greedy, search-guided). That
convergence-to-optimal statement is a tabular theorem (every state-action pair
visited infinitely often, decaying step sizes); combining function approximation,
bootstrapping, and off-policy data is the classic "deadly triad," so in the
approximate form it is a motivation, not a guarantee. The approximate form
replaces the table with $Q_\theta$ (the adapter and heads of Section 5), trained
to reduce $(Q_\theta(s, a) - y)^2$ using a replay buffer and a
periodically-synced target network $Q_{\text{target}}$, the standard stabilizers
for exactly this regime.

**Returns.** A trajectory interleaves states, actions, and rewards,

$$
s_0, a_0, r_0,\; s_1, a_1, r_1,\; \dots,\; s_T, a_T, r_T,
$$

ending with the terminal reward $r_T$, which in this system is the verifier's
verdict at the root COMMIT, or $0$ if the step budget expires with no root commit
(Section 6). The intermediate $r_t$ are all $0$ in the first version (Section 6).
The return from step $t$ is the discounted sum of rewards from $t$ to the end,

$$
G_t \;=\; r_t + \gamma\, r_{t+1} + \cdots + \gamma^{T-t} r_T \;=\; \sum_{k=t}^{T} \gamma^{k-t} r_k.
$$

A one-step target ($y = r_t + \gamma \max_{a'} Q(s_{t+1}, a')$) bootstraps after a
single transition and is fragile when reward is sparse and only terminal;
regressing $Q(s_t, a_t)$ onto a longer n-step or Monte Carlo return $G_t$ is a
remedy that needs no search (Section 7.2). One caveat, developed in Section 7.2:
the max inside the one-step target is what makes Q-learning off-policy, and a
longer uncorrected return gives that property up in proportion to its length,
because every reward beyond the first follows the behavior policy rather than the
greedy one. MCTS (Section 7.4) is itself a search procedure that can supply even
better targets, but it is optional. Methods that sum over $s'$ (value iteration,
expectimax) need an enumerable model; samplers (Q-learning, and MCTS, which is
sampled search) need only to draw $(s', r)$. This MDP is generative-only.

---

# 3. The structured state

## 3.1 The forest

The state is an immutable reasoning forest; each action returns a new state.
Nodes are of a few kinds:

- **Goals and subgoals**: things to be solved. The original problem is the root
  goal; subgoals are optional decompositions.
- **Hypotheses**: tentative, unverified claims attached to a goal.
- **Facts**: claims with provenance `given`, `derived`, or `verified`.
- **Verifier-call records**: input, output, and success flag of each call to the
  symbolic oracle, retained as part of the state.

Edges record parent and child relationships and, optionally, a connective
(Section 3.3). The verifier sits behind a pluggable protocol, so the core is
domain-agnostic: a domain supplies a concrete verifier and a node vocabulary.

For example, solving $|x - 3| + |x + 2| = 7$ by case analysis induces a forest
like the following (node kinds in capitals, verbs on the edges):

```text
g0  GOAL  solve |x-3| + |x+2| = 7                              (root)
|
+- h0  HYPOTHESIS  square both sides to remove the absolute values
|  \- ABANDON h0                (walked away: degree-4 blowup; no verdict claimed)
|
|  SPLIT : exhaustive case analysis on the sign changes at x = -2 and x = 3
|
+- sg1  SUBGOAL  case x < -2 :   -(x-3) - (x+2) = 7
|  |
|  +- f1  FACT (derived)        1 - 2x = 7
|  +- h1  HYPOTHESIS            x = -3
|  +- VERIFY h1 -> f2  FACT (verified)   x = -3 solves it and x < -2
|  \- COMMIT sg1 (solution := h1)        x = -3
|
+- sg2  SUBGOAL  case -2 <= x < 3 :   -(x-3) + (x+2) = 7
|  |
|  +- f3  FACT (derived)        5 = 7      (contradiction)
|  +- VERIFY f3                 f3 promoted to (verified): the case has no solution
|  \- COMMIT sg2 (solution := f3)        no solution in this case
|
+- sg3  SUBGOAL  case x >= 3 :   (x-3) + (x+2) = 7
|  |
|  +- h2  HYPOTHESIS            x = 4
|  +- VERIFY h2 -> f4  FACT (verified)   x = 4 solves it and x >= 3
|  \- COMMIT sg3 (solution := h2)        x = 4
|
+- h3  HYPOTHESIS (root)        x in { -3, 4 }
\- COMMIT g0 (solution := h3)   final answer x = -3 or x = 4    (verifier-gated, terminal)
```

Three things this makes concrete. The verified-versus-tentative split is
visible: $x = -3$ is a hypothesis until `VERIFY` promotes it, and the derived
fact `f1` stays tentative because nothing ever verifies it. Every case closes
with a `COMMIT`: `sg1` and `sg3` with solutions, and `sg2` with the verified
outcome "no solution in this case." A case that provably has no solutions is a
*solved* subgoal; its refutation is a result the parent consumes (the root
hypothesis $x \in \{-3, 4\}$ is licensed by all three verdicts, including
`sg2`'s). And `ABANDON` appears exactly once, on `h0`: the policy walked away
from an unpromising line of attack, claiming no verdict. It is never used for
cases resolved by contradiction. Only the root `COMMIT` is verifier-gated and
terminal.

## 3.2 The one load-bearing distinction

The single distinction that must exist is verified facts versus tentative
hypotheses. Without it the controller cannot tell "I proved this" from "I think
this," and `VERIFY` (Section 4) loses its meaning. Precisely: the line is
verifier-verified versus everything else. `given` facts sit on the trusted side
by fiat (they restate the problem, not an LLM claim). A `derived` fact, despite
the name, sits on the tentative side: it is an unverified LLM computation, and it
is exactly the kind of node `VERIFY` exists to promote. Only provenance
`verified` is on the proved side, and only a verifier call puts it there.

Predicates such as `solved` and `verified` are derived from the structure:
information the policy and lens may read, never a control-flow constraint. The
only hard gate in the system is the verifier on the committed root answer;
everything else is scaffolding the policy may use or ignore.

## 3.3 Subgoals are optional scaffolding, not proof obligations

Decomposition exists to give the lens something to focus on, not to impose a
control flow. The policy may split a goal, work some children, ignore others,
abandon a branch, or commit the root directly from a single hypothesis. Nothing
forces it to discharge every child of a conjunctive node. Subgoal-solving is
instrumental, never required for reward: the LLM can produce a correct final
answer even when not every subgoal was solved explicitly.

So the AND/OR connective on a decomposition edge is optional, non-gating
metadata. Under final-answer verification (GSM8K, the MATH dataset), AND/OR
aggregation is not load-bearing for correctness; it is at most a hint to the
policy and a cue to the lens. It would become load-bearing only if we verified
the completeness of the reasoning (exhaustive case analysis, or proof obligations
in a system like Lean), a deliberately deferred and larger setting. Whether a
typed connective with aggregation, and a checked exhaustiveness obligation, helps
is an ablation, not a v1 commitment.

## 3.4 The math instantiation, and the Markov bet

The first target is verifiable math. The math state instantiates the core with
`facts` (given, derived, verified), open and closed subgoals, `hypotheses`, and a
`verifier_log`, all carrying stable identifiers so actions can reference nodes
safely. The verifier is SymPy. The terminal check on a committed answer is one
deterministic call: no learned judge, no reward model, no preference data.

The Markov bet, stated honestly: the state is a compressed approximation of the
true reasoning history, since a lossy parser maps LLM text into nodes. When a
learned value looks wrong, suspect the parser before the network.

---

# 4. The action space: verbs as lenses

## 4.1 An action is a lens

An action is `a = (verb, target)`, where `target` is an identifier into the
state. The verb is the kind of move; the target is the node it operates on; the
frozen LLM, where the verb calls for it, supplies free-form content. Each verb is
fully specified by five fields:

- `avail(s, t)`: a state-conditioned legality mask.
- `schema(s, t)`: a JSON schema constraining the LLM output.
- `view(s, t)`: the lens get, the projection of $s$ rendered into the LLM context.
- `apply(s, t, o)`: the lens put, the localized update to target $t$ and its
  neighborhood given LLM output $o$.
- a reward, zero for almost every verb (Section 6).

The transition samples $o$ from the frozen LLM conditioned on
$\text{view}(s, t)$ plus a verb suffix, then sets $s' = \text{apply}(s, t, o)$.
Since $P(s' \mid s, a)$ integrates over LLM continuations, the MDP is generative:
we sample transitions, we do not enumerate them.

## 4.2 The verb set: six core, one optional

```text
SPLIT            structure : expand a goal into children
FORM_HYPOTHESIS  propose   : add a tentative, unverified claim
COMPUTE_SUBEXPR  propose   : add a derived fact (a computation step)
VERIFY           assess    : call the symbolic oracle; sole intermediate-reward channel
COMMIT           resolve   : close a goal with an outcome (terminal at the root)
ABANDON          walk away : drop an unproductive branch; no outcome claimed
CRITIQUE         assess    : optional, reward-free LLM self-assessment
```

The set covers the five operation families that the reasoning-systems literature
collapses to (Appendix A). The propose/expand family contributes three verbs:
`SPLIT` for expansion, and `FORM_HYPOTHESIS` and `COMPUTE_SUBEXPR` for the two
epistemic flavors of proposing. Each remaining family contributes one verb
(`VERIFY`, `COMMIT`, `ABANDON`), with `CRITIQUE`, the critique/refine family's
verb, optional: six core verbs plus one optional. Per-verb rationale:

- **`SPLIT`** folds conjunctive decomposition ("solve all children") and
  disjunctive or exhaustive case analysis ("cover the parent"): the same move
  (expand a goal into children), differing only in the AND/OR semantics of the
  children, which is recorded as optional non-gating metadata (Section 3.3).
- **`FORM_HYPOTHESIS`** and **`COMPUTE_SUBEXPR`** stay distinct because the role
  of the node they write differs (a free-standing conjecture versus a computation
  step), though both start on the tentative side of the Section 3.2 line, and
  `VERIFY`'s reward gate depends on that line: a derived fact must never be
  treated as verified without a verifier route.
- **`VERIFY`** invokes the symbolic oracle and, on success, promotes the target
  to a verified fact. The verifier is the sole reward *source* and reaches the
  agent through exactly two verbs: `VERIFY` is the only channel for intermediate
  reward (gated and capped, and zero-coefficient in the terminal-only first
  version, Section 6), and root `COMMIT` carries the terminal verdict. `VERIFY`
  is kept strictly distinct from `CRITIQUE`.
- **`COMMIT`** closes a goal with an explicit existing outcome. In v2 its
  target is a verified claim, and `avail(s, t)` requires that claim to
  discharge its open owner goal. No free-form answer text enters at COMMIT and
  no witness is selected silently. The claim's owner determines which goal
  closes. COMMIT is recursive but terminal only when that owner is the root, so
  there is no separate final-answer verb. Subgoal commits are reward-free
  bookkeeping. A verified refutation closes its case as solved because its
  result is needed by the parent. COMMIT makes no LLM call.
- **`ABANDON` (v1) / `PRUNE` (v2)** walks away from an unproductive branch:
  no outcome is claimed and no verdict is recorded. V2 makes this operation
  explicitly irreversible, cascades inactivity through descendants, and
  disallows it at the root. A branch resolved by contradiction is closed with
  COMMIT, not PRUNE. The operation remains emittable because at inference
  budget $B = 0$ (Section 8) the deployed model runs no search and must be able
  to express "this branch is unproductive, move on."
- **No `SUMMARIZE` verb**: compressing a closed subtree to its statement and
  verdict is a deterministic rendering property of the lens (Section 4.3).
- **`CRITIQUE`** is an optional, reward-free LLM self-assessment of a hypothesis
  against its goal; whether it earns its tokens is an ablation.

## 4.3 The lens (view) and the write-back (apply)

In v2, the environment contract selects the exact versioned generator request
builder. `typed-claim-lens-v2.1` is current, while
`typed-claim-lens-v2` is retained byte-exact for legacy replay. Changing lens
bytes therefore changes the environment contract rather than silently changing
the transition distribution.

The default `view`, when the target is a subgoal, is grounded-local: the problem
statement, the ancestor path from root to target, and the target's own subtree
(its facts, hypotheses, open children). Siblings and unrelated branches are
hidden; verbs that need more override the default. Closed subtrees render
compactly as their statement plus verdict, which is the lens property that
replaces a `SUMMARIZE` verb.

The `apply` write-back is localized: the parsed output mutates the target and its
neighborhood (add a hypothesis under the goal, add a derived fact, mark a node
verified, record a critique), producing the next immutable state. Because both
read and write are scoped to the target's neighborhood, acting on a subgoal is
identical to stepping a smaller MDP rooted there. Locality, modularity, and
recursion are the same property, derived from the action interface rather than
asserted.

## 4.4 The factored action and why the target is promoted

If the controller chose only a verb and let the LLM sample the target, credit
would blur: a value over verbs averages over every target the LLM might pick, so
the agent would never learn which node to act on. Promoting the target into the
chosen action fixes this. The policy factors as

$$\pi(\text{verb} \mid s)\,\cdot\,\pi(\text{target} \mid s, \text{verb}),$$

a categorical verb head and a pointer head over the live identifiers in the
state. The effective action set is `verbs x valid-targets`, which grows with the
tree, which is why the target side is a pointer over live identifiers, not a
fixed-length vector.

Invalid pairs are masked before the policy normalizes. Legality is pair-level
and defined by `avail`: pair `(verb, target)` is legal in $s$ iff
`avail(s, target)` admits the verb. The mask is type-aware per verb: `COMMIT`
points only at verified claims that discharge an open owner, `VERIFY` at
unchecked claims, and `DECOMPOSE` at open goals. Verb-level legality is
*derived*, never separately specified: a verb is legal in $s$ iff some live
target admits it (the OR over targets), so the verb head masks exactly the
verbs with no legal target and the pointer head never normalizes over an empty
set. $A(s)$ is nonempty in every non-terminal state by construction: the root
goal is open and at least one proposal verb is legal there. A COMMIT is
terminal only when its target claim is owned by the root; otherwise episodes
end at the step-budget terminal of Section 6.

---

# 5. The controller network and the encoder ladder

## 5.1 Global value, local generation

One split organizes everything: the learned heads value on the global state,
while the frozen LLM generates content from a local view. Whether an action helps
depends on the whole problem, so the policy and Q function embed the entire
forest; but generation is short and on-topic, so the LLM sees only the verb's
scoped view. The full state remains the Markov state; the view is a projection
for generation, never the state itself.

## 5.2 Encoder signature and heads

To score a variable, growing set of targets, the pointer head needs a feature
vector per live node, not a single global vector. The encoder therefore emits a
pair: a global readout, and one vector per node.

```text
encoder(s) -> (g, {h_k})           g = global readout, h_k = per-node vector

g    -> adapter MLP -> z
z    -> verb head    -> pi(verb | s)            (K logits; verbs with no legal target masked, Section 4.4)
(z, h_k, verb) -> pointer head -> pi(target | s, verb)  (over live nodes, masked)
(z, h_k, verb) -> Q head       -> Q(s, verb, target)    (scalar per legal pair)
```

A target Q network (Section 2) supplies the bootstrap target. $Q_{\text{target}}$
is a periodically-synced frozen snapshot of every trainable parameter on the Q
path: the adapter and the Q head (the frozen LLM encoder is shared, never
copied). The scope matters: the adapter trains too, and a snapshot of the Q head
alone would still chase a moving adapter. The LLM is always frozen; only the
adapter and the three heads ever train (together under five million parameters).
A Bellman update on $(s, (\text{verb}, \text{target}), r, s')$ touches the scored
pair and shared body; other actions see zero gradient on that example.

## 5.3 The encoder ladder (a pluggable axis)

Three encoders, in increasing structural fidelity; the heads and loss are
unchanged across them, so the choice is an experiment, not a commitment.

1. **LLM marker-token hidden states (v1 default).** Serialize the forest in
   root-to-children (pre-order) order with marker tokens
   (`[PROBLEM] ... [NODE id=sg_7 type=subgoal] <statement> ... [STATE_END]`) and
   run one forward pass; a transformer yields a hidden vector at every position,
   so all reads come from that single pass. The v1 reading rule: each node's
   vector $h_k$ is the mean over the node's own statement tokens, from its marker
   through the end of its own statement, excluding descendant spans (under
   pre-order, a node's own span ends where its first child's marker begins); the
   global readout $g$ is the hidden state at `[STATE_END]`, which attends over the
   entire serialization. Marker-position-only reading and pooling over the full
   subtree span are ablation arms, not the default. Caveat: causal attention
   means a node's own tokens see ancestors and earlier siblings but not
   descendants, so a parent's vector does not reflect its children's progress;
   the global readout carries that information, and option 3 removes the
   limitation entirely. Deployment note: reading hidden states at chosen
   positions requires a raw transformers-style forward pass with hidden-state
   output; throughput engines behind completion or embedding APIs (vLLM- and
   TGI-style servers) do not readily expose per-token hidden states. In v1 the
   encoder is the same frozen checkpoint as the generator, co-hosted in one
   process serving both roles. Each controller step therefore costs one
   full-state encoding forward (prefill only, no decoding) in addition to the
   generation pass over the lens view; both count toward Section 9's forward-pass
   and FLOP totals.

2. **Text-embedding endpoint.** A decoupled sentence embedder over the serialized
   state gives a clean separation from generation and is compatible with standard
   embedding endpoints. It returns one pooled vector, so per-node vectors require
   one call per node, which is the cost of the cleaner separation.

3. **Semantic-feature graph neural network (the principled upgrade).** Because the
   state is genuinely a graph, a graph neural network (GNN) over it is the natural
   encoder. The essential design point: a topology-only GNN knows parent/child and
   connective edges but nothing about the mathematics, so each node's input
   features are initialized from a text embedding of its statement; typed edges
   (parent/child, fact/hypothesis, AND/OR) carry the relations; message passing
   then yields per-node and pooled-global vectors that are both semantic and
   structural, and bidirectional, which removes the causal-visibility limitation
   of option 1. Because nodes are immutable, their text embeddings are computed
   once and cached; only the cheap GNN re-runs as the forest grows, which is what
   makes the per-decision cost of this rung far below any LLM forward pass.

---

# 6. Reward: the verifier-anchor invariant

The symbolic verifier is the only reward source. The literature supports this:
DeepSeek-R1 rejects learned process-reward models (PRMs) on reward-hacking and
ill-defined-step-boundary grounds, CRITIC shows self-critique without external
grounding is unreliable, and the most reliable systems (AlphaProof,
AlphaGeometry, rStar-Math, R1, Tulu-3) are hard-verifier-anchored.

- Most verbs return reward 0 (`SPLIT`, `FORM_HYPOTHESIS`, `COMPUTE_SUBEXPR`,
  subgoal `COMMIT`, `ABANDON`, `CRITIQUE`). Lens compression, being a rendering
  property of the view rather than an action (Section 4.3), carries no reward by
  construction.
- `VERIFY` carries the design's only intermediate-reward channel: a gated, capped
  positive reward on a genuinely new promotion to a verified fact (anti-farming
  gates: per-episode cumulative cap, dedup per target, reject tautologies). In v1
  this channel's coefficient is 0: every intermediate reward, including
  `VERIFY`'s, is 0.
- Root `COMMIT` returns the terminal verdict, $+1$ correct, $-1$ incorrect.
- A masked action (an illegal or failed operation: a stale target from a
  search tree, a rejected duplicate `COMPUTE`, an unusable generation)
  advances the turn and receives reward 0. The spent turn and finite horizon
  carry the opportunity cost without adding a non-verifier Bellman target.
  Archived v1 trajectories used a $-0.01$ housekeeping penalty; exact archive
  reproduction must request the explicit `legacy-mask-penalty-v1` contract.
  The live `symbolic-only-v1` contract and all v2 work keep reward zero.

The first version uses terminal-only reward:

$$
r_T =
\begin{cases}
+1.0 & \text{verifier judges the committed answer correct (root COMMIT)}\\
\ \ 0.0 & \text{step budget } T_{\max} \text{ exhausted with no root commit}\\
-1.0 & \text{committed answer incorrect (root COMMIT)}
\end{cases}
$$

with all intermediate rewards 0. $T_{\max}$ is a maximum number of controller
actions per episode; it bounds the trajectory, not the forest depth. Budget
expiry is treated as a genuine terminal ($y = r = 0$), and the remaining budget
is part of the state, serialized in the state header so the encoder sees it (the
immutable forest already implicitly encodes elapsed steps, since every action
leaves a record, but the explicit feature makes the Markov property manifest
rather than inferred).

The gated, capped `VERIFY` channel is the pre-specified fallback, and the only
sanctioned one: its magnitude, per-episode cap, and activation criterion (a
pre-registered stall test, for example learned-controller accuracy at $B = 0$
failing to improve over $K$ consecutive evaluation checkpoints) are fixed in the
pre-registered experiment plan (Section 9.1) before any run enables it; they are
not tuned mid-experiment. "More signal" always means more verifier-derived
shaping, never a reward model and never an LLM-judge reward. The moment reward
comes from a model's opinion rather than the verifier's check, the project has
changed. Aesthetic signals (reasoning-quality classifiers, stylistic critics) may
enter only as MCTS priors or pruning heuristics, never as Bellman targets.

---

# 7. Training: value-based RL as the spine

## 7.1 The core: off-policy value-based Q-learning

The core is the approximate Q-learning of Section 2 over sampled transitions
$(s, a, r, s')$ with $a = (\text{verb}, \text{target})$. The target is

$$
y \;=\; r \;+\; \gamma \,\max_{(\text{verb}', \text{target}')\in A(s')} Q_{\text{target}}(s', \text{verb}', \text{target}'),
$$

with the convention that the max term is zero when $s'$ is terminal (root
COMMIT, or budget expiry under Section 6's terminal convention), so $y = r$ on
exactly the transitions that carry the verifier's verdict; the n-step and Monte
Carlo returns of Section 7.2 use the same convention at the episode end. We
update the Q head and shared body to reduce
$(Q(s, \text{verb}, \text{target}) - y)^2$.

**Roles: the heads act, Q criticizes.** The factored verb and pointer heads
(Section 4.4) are the acting policy at every stage and at deployment: sample
verb from $\pi(\text{verb} \mid s)$, then target from
$\pi(\text{target} \mid s, \text{verb})$, with explicit exploration (temperature,
or epsilon-greedy over legal pairs). $Q$ is the critic. The heads are refreshed
toward the improved policy by distillation: the factored joint
$\pi(\text{verb} \mid s)\,\pi(\text{target} \mid s, \text{verb})$ is trained by
cross-entropy toward $\operatorname{softmax}_{a \in A(s)}(Q(s, a)/\tau)$ over the
legal-pair set (toward $\pi_{\text{MCTS}}$ when search is on, Section 7.4). The
distillation is gated and annealed: it begins only after $Q$'s regression error
on the warm-start data falls below a threshold (or after a fixed warm-up), so a
cold $Q$ never overwrites the behavior-cloned heads (Section 7.3). A learned $Q$
remains the natural object: it folds the expensive expectation over LLM
transitions into a cheap action value, so action selection at deployment samples
no LLM transition to look ahead.

Off-policy is load-bearing: with the one-step max target above, the same replay
buffer can be fed by random exploration, an LLM-controller (Section 7.3), or
MCTS, with no change to the update. The qualifier matters: the longer returns of
Section 7.2 weaken exactly this property, and Section 7.2 states the bias and the
convention that manages it.

**Replay rows and encode cost.** Computing the max over $A(s')$ needs the
per-node vectors and legality mask for $s'$, which under the v1 encoder would
cost one full LLM forward pass per replay row per update. Because the v1 encoder
is frozen, this is avoided by construction: each replay row stores the raw states
$s$ and $s'$ plus the encoder outputs $(g, \{h_k\})$ for both and the mask
$A(s')$, computed once at collection time and never stale, so replay training
touches no LLM forward pass at all (gradients flow only through the adapter and
heads). The text-embedding rung caches the same way; the GNN rung is trained, so
cached GNN outputs go stale and that rung re-encodes from the stored raw state
(its frozen per-node text embeddings remain cacheable). Every replay row carries
a (serialization-schema version, encoder id) tag; rows whose tags mismatch the
current configuration are dropped or re-derived from the stored raw state, never
silently consumed. This is load-bearing because the buffer is long-lived by
design: rung-1 teacher trajectories (Section 7.3) are replayed at rung 3,
potentially across encoder and serialization changes.

## 7.2 Sparse reward: n-step and Monte Carlo returns

A one-step bootstrap is fragile under terminal-only reward. The search-free
remedy is to regress $Q(s_t, a_t)$ toward an n-step or full Monte Carlo return
$G_t$ from the verifier terminal (Section 2).

The price must be stated rather than hidden: every reward beyond the first
inside such a return was produced by the behavior policy $\mu$ that generated the
trajectory, so an uncorrected long return estimates $Q^\mu$ (take $a_t$, then
follow $\mu$), not $Q^\star$. Since $Q^\mu \le Q^\star$ for every $\mu$, returns
from a strong behavior policy are tight, useful lower bounds on $Q^\star$, while
returns from weak or random behavior are severely pessimistic. The design
therefore commits to a convention per data source:

- **Near-on-policy data** (the controller's own recent behavior): uncorrected
  short-horizon n-step, $n \in [3, 5]$, the standard deep-Q convention; the bias
  is small because the behavior is close to greedy-by-Q.
- **Genuinely off-policy data** (teacher trajectories, stale replay): either cut
  the return at the first action that disagrees with the current greedy policy
  and bootstrap there (tree-backup / Retrace-style truncation), or use the full
  return deliberately as a warm-start labeled for what it is: regression toward
  $Q^{\text{teacher}}$, valuable precisely because the teacher is strong and
  $Q^{\text{teacher}}$ is a tight lower bound on $Q^\star$.
- **Schedule**: long returns while teacher data dominates the buffer, shortening
  toward the one-step target as the controller's own behavior takes over,
  recovering the full off-policy property of Section 7.1 in the limit. Exact $n$
  and the switch-over are fixed in the pre-registered experiment plan
  (Section 9.1).

## 7.3 The three-rung controller-training ladder

Learning a policy over a variable, growing `(verb, target)` space under sparse
reward is the highest-risk part of the design. A ladder defangs it by never
asking the network to learn from scratch.

Two LLM roles are distinct. The *generator* is the small frozen LLM that
produces content under the lens; it is the deployed model and the small-model
thesis is about it. The *controller-teacher* is an LLM used only at training
time to pick `(verb, target)` decisions; it has no deployment footprint and may
be much larger, even a frontier model.

**Generator model class.** The deployed generator is a small open-weights model,
on the order of 1 to 4 billion parameters. Instruction tuning is permitted, and
in practice needed, because every verb constrains the generator's output with a
JSON schema (Section 4.1). What is excluded is reasoning-specific post-training:
no o1/R1-style long-chain-of-thought RL and no R1-distilled checkpoints. The
exclusion is load-bearing for the contrast of Section 1.2: with a
reasoning-tuned generator, deliberation would happen inside each verb's
generation as internal chain of thought rather than in the orchestrated state,
and gains could no longer be attributed to the controller and the structure. (A
pure base model driven by constrained decoding is the attribution-maximal
variant, recorded as an ablation.) The exact checkpoint is delegated to the
pre-registered experiment plan (Section 9.1); the class constraint (small,
open-weights, instruction-tuned, non-reasoning) is a design commitment.
Hosted 7B to 32B models are capacity and interface diagnostics only unless a
separately pre-registered design revision expands the deployed generator class.

The ladder:

1. **LLM-as-controller (no training).** A controller-teacher picks the next
   `(verb, target)`, the small frozen generator fills in content under the lens,
   and the verifier checks. This validates the whole interface end to end before
   any network is trained, and its trajectories are the bootstrap data. To reduce
   the train-versus-deploy mismatch, the teacher orchestrates the small generator
   during collection, so trajectories reflect the content the deployed generator
   can actually produce: the teacher supplies decision quality, the small
   generator supplies realistic content. Stated honestly, this removes the
   generator-content mismatch but not the controller-induced state-visitation
   shift: the teacher's `(verb, target)` choices visit states the learned
   controller will not, so the rung-2 clone inherits the classic imitation
   covariate shift. Rung 3 mitigates this by consuming the controller's own
   off-policy behavior; if the cloned heads degrade off the teacher's
   distribution before rung 3 closes the gap, the named remedy is a DAgger-style
   pass (roll out the warm-started controller, query the teacher for
   `(verb, target)` labels on the states the controller actually visits, continue
   cloning on the aggregated data). The teacher remains a training-time labeler
   only; no reward is derived from it.

2. **Behavior-cloning warm-start.** Train the verb and pointer heads to imitate
   the teacher's chosen `(verb, target)` pairs, by cross-entropy on the factored
   heads over the encoder outputs $(g, \{h_k\})$. In the same rung, warm-start
   the critic: regress the Q head on Monte Carlo returns $G_t$ computed offline
   from the teacher trajectories (the Section 7.2 machinery applied to the
   teacher buffer, labeled as regression toward $Q^{\text{teacher}}$). The heads
   start from "acts like the teacher" and the critic starts from "values like the
   teacher's returns" rather than from noise, which is what survives the
   sparse-reward cold start.

3. **RL fine-tune of the controller heads.** Act from the behavior-cloned heads
   with exploration (temperature or epsilon over legal pairs), update $Q$
   off-policy on the returns of Section 7.2 under its stated convention
   (trace-cut or warm-start-labeled full returns on teacher data, short-horizon
   n-step on the controller's own behavior), and distill the heads toward
   softmax-over-Q only on the gated, annealed schedule of Section 7.1, consuming
   both the teacher's trajectories and the controller's own behavior.

Per-stage roles, in one view:

| Stage | Who acts | What trains |
| --- | --- | --- |
| Rung 1 (collect) | controller-teacher | nothing (data only) |
| Rung 2 (warm-start) | offline | heads by BC; Q by MC regression on teacher returns |
| Rung 3 (RL) | cloned heads + exploration | Q (off-policy targets); heads (gated distillation) |
| Deploy $B=0$ | heads (greedy or low temperature) | nothing |
| Deploy $B>0$ | MCTS with heads as prior, Q at leaves | optionally distill from search |

This is a distillation, but a different one from o1 or R1. Standard distillation
compresses a big model's weights or chain-of-thought traces into a smaller model.
Here the big model's weights are never touched and its chain of thought is never
the product; we distill its policy over typed operations into a tiny external
controller that drives a small frozen generator. Generation stays small and
cheap; only the decision-making is borrowed, and only at training time.

## 7.4 MCTS as an additive add-on (not the spine)

MCTS is an optional accelerator that changes nothing about the core algorithm.
Briefly: MCTS estimates action values by sampled search, building a tree of
decision nodes (a state) and chance nodes (a state-action pair $(s, a)$ holding a
visit count $N(s, a)$ and a mean value). Each simulation selects actions by PUCT
(predictor upper confidence bounds applied to trees),

$$
Q(s, a) + c_{\text{puct}}\,P_0(a \mid s)\,\frac{\sqrt{N(s)}}{1 + N(s, a)},
$$

where the prior $P_0$ is the learned policy (the factored heads of Section 7.1),
$N(s) = \sum_b N(s, b)$ is the decision node's total visit count, and
$c_{\text{puct}} > 0$ is an exploration constant (a pinned hyperparameter of the
B3/B4 baselines, Section 9.1); expands a node; samples a transition and evaluates
the leaf; and backs up the discounted return. The visit distribution
$\pi_{\text{MCTS}}(a \mid s) = N(s, a)^{1/\tau_{\text{visit}}} / \sum_b N(s, b)^{1/\tau_{\text{visit}}}$
(at visit-count temperature $\tau_{\text{visit}}$, distinct from the acting
temperature $\tau$) and the chance-node mean $\hat{Q}_{\text{MCTS}}(s, a)$ are
the improved policy and value.

Because the core is off-policy, MCTS contributes in two purely additive ways:
better behavior (search explores more intelligently than epsilon-greedy, adding
higher-quality rows to the same buffer; those rows train with the one-step target
or the chance-node regression below, both valid off-policy) and better targets
($\hat{Q}_{\text{MCTS}}$ beats a one-step bootstrap as a regression target for
$Q$, and $\pi_{\text{MCTS}}$ beats softmax-over-Q as a distillation target for
the heads). Distillation from visit counts is exactly this add-on: the "MCTS-on"
variant of the same losses, an ablation rather than a requirement. Keeping MCTS
off-policy and additive is what makes "MCTS is optional" literally true. The
on-policy policy-gradient alternative is recorded in Appendix B.

---

# 8. Inference-time compute knob

Once the controller is trained, the system trades latency for search via a budget
$B \in \{0, 1, 2, \dots\}$:

- $B = 0$: act directly from the verb and pointer heads (greedy or low
  temperature; $Q$ available for tie-breaking). Per-step cost under the v1
  encoder: one decode-free frozen-LLM prefill over the serialized state
  (producing $(g, \{h_k\})$), negligible head FLOPs, and one lens-scoped
  generation for verbs that produce content. The encoder pass is a full LLM
  forward pass and is counted as such in Section 9; per-decision cost far below
  any LLM forward pass is claimed only under the cached-embedding GNN encoder
  (Section 5.3, option 3).
- $B > 0$: run MCTS for $B$ simulations from the current state, initialized from
  the learned heads as prior and $Q$ at the leaves, and act on the improved visit
  distribution.
- Persistent search trees let a user "think harder" by continuing the search
  without restarting.

The budget is explicit and the compute is accounted for (Section 9).

---

# 9. Evaluation

The headline result is a compute-versus-accuracy frontier on verifiable-math
benchmarks, escalating in difficulty: GSM8K, then MATH (evaluated on the standard
MATH-500 subset), then AIME-style competition problems (for example AIME 2024/25,
OlympiadBench) once the MATH frontier is non-trivial. The baseline grid is
pre-registered in the experiment plan (Section 9.1) and not adjusted post hoc:

```text
B0.  Deployed generator (same checkpoint), greedy decode      raw capability floor
B1.  Same generator, best-of-N, self-consistency majority
     vote over canonicalized final answers; N swept to
     trace its own compute-accuracy frontier                  cheap deliberation
B1v. Same generator, best-of-N, verifier-checked selection
     (gold-free substitute-back check per candidate); run
     only where the check needs no gold answer                strong deliberation
LCs. Small generator as its own controller: handed the
     full serialized state, picks each (verb, target)
     zero-shot; no learned heads                              deployable orchestration baseline
LCt. Rung-1 teacher at inference: the controller-teacher
     picks (verb, target); the small generator fills
     content                                                  orchestration ceiling
B2.  Learned controller, B=0 (no search)                      fast mode
B3.  Learned controller, B=small MCTS                         light search
B4.  Learned controller, B=large MCTS                         heavy search
B5.  Larger reasoning model, greedy decode                    capability ceiling
```

B0, B1, B1v, LCs, and LCt use the identical generator checkpoint as B2 to B4, so
any gain over them is attributable to orchestration and training rather than to
the generator's prior post-training; B5 is the only row using a different
(larger, reasoning-tuned) model.

The compute axis is committed in advance and reported three ways together: total
inference FLOPs per problem, total LLM forward passes per problem, and wall-clock
at fixed hardware. Accounting is itemized per row: (i) all frozen-LLM forward
passes, prefill and decode, including encoder passes under the v1 encoder;
(ii) controller head and adapter FLOPs (roughly $10^7$ per decision, negligible);
(iii) embedding-endpoint or GNN FLOPs where those encoders are used;
(iv) verifier CPU time, reported separately. Controller forward passes count for
every row: LCs adds one generator pass per action choice, LCt adds one teacher
pass per action choice at the teacher's FLOPs. Comparisons are
frontier-versus-frontier (B1's $N$ is swept, not matched to a single row).
Training-time teacher calls have no deployment footprint and are excluded from
inference frontiers, reported once as training cost. The risk to guard against
is a compute leak: reporting only the small model's FLOPs while spending many
tokens elsewhere.

The honest win condition names its baseline. Against LCt the claim is
distillation, not compute: the teacher's decision quality compressed into a
roughly five-million-parameter controller that retains most of LCt's accuracy
(beating LCt on compute is automatic and is not claimed as a result). Against
LCs both axes are live: the learned controller should match or exceed LCs's
accuracy while replacing a full prefill-and-decode LLM call per action choice
with, under the v1 encoder, a decode-free prefill plus negligible head FLOPs
(the saving is the decision-decode tokens and independence of decision quality
from controller-LLM size; per-decision cost far below any LLM pass is claimed
only under the GNN encoder). A sanity-check ablation (random Q head, uniform
policy, fixed MCTS) confirms that training, not the bare scaffold, does the
work.

**Splits and contamination.** Training and evaluation problems are disjoint by
construction: rung-1 teacher trajectories, the rung-2 cloning set, and every
problem that ever enters the replay buffer are drawn only from each benchmark's
training split (GSM8K train; MATH train, excluding MATH-500); all reported
numbers come from held-out problems no rung of the controller has seen. The
frozen generator's pretraining has almost certainly included GSM8K and MATH;
that contamination cannot be controlled, so it is handled two ways. First, every
row runs the same frozen generator, so results are read as deltas over B0:
pretraining contamination inflates the floor and the treatment rows alike, and
the delta isolates what the controller and structure add. Second, the escalation
includes at least one plausibly-uncontaminated set: problems released after the
generator's training cutoff (the most recent AIME year) or synthetic problems
generated and answer-checked by the verifier itself. Exact problem lists, split
manifests, and seeds are fixed in the pre-registered experiment plan.

## 9.1 Pinned versus delegated quantities

Every load-bearing quantity is either pinned here or delegated to a single
pre-registered experiment plan, `experiments/PLAN.md`, which this document
commissions. That file is the registration referred to above: it is committed
before the first training run and not adjusted post hoc.

Pinned at design level:

- Terminal reward $+1 / 0 / -1$; v1 reward is terminal-only (`VERIFY` shaping
  off, Section 6).
- Discount $\gamma = 0.99$ (episodic, terminal-dominated reward; $\gamma$ exists
  to weakly prefer shorter episodes).
- Controller: under five million trainable parameters (Section 5.2).
- Generator class: small open-weights, instruction-tuned, non-reasoning, roughly
  1 to 4 billion parameters (Section 7.3); exact checkpoint named in the plan.
  Hosted 7B to 32B models remain capacity and interface diagnostics unless the
  design and experiment plan are revised before collection.
- Teacher default: a 70B-class open-weights model, chosen for reproducibility; a
  frontier API teacher is a sanctioned variant provided all rung-1 trajectories
  are archived, since an API model may change or disappear.
- Benchmarks and the baseline grid above.

Delegated to `experiments/PLAN.md` (each fixed there before any run):

- Episode step budget $T_{\max}$ (Section 6).
- Acting temperature $\tau$ (Sections 2, 7.1) and visit-count temperature
  $\tau_{\text{visit}}$ (Section 7.4); epsilon schedules.
- $n$ for the n-step return, the trace-cut rule, and the return-length schedule
  (Section 7.2).
- `VERIFY` shaping, if ever activated: magnitude, per-episode cap, activation
  criterion (Section 6).
- Replay capacity, batch size, target-network sync period, optimizer, learning
  rates; the distillation gate and anneal schedule (Section 7.1).
- $c_{\text{puct}}$ and concrete simulation budgets for B3 and B4 (for example
  $B_3 = 16$, $B_4 = 128$), fixed before the frontier is measured.

---

# 10. Non-goals and honesty discipline

- No reward model and no learned process-reward; the verifier is the sole reward
  source.
- No LLM-judge reward; `CRITIQUE` is reward-free information.
- No macro "solve it all" verb that bypasses per-step search and verification. A
  verb may write a localized multi-node chain (a step-by-step expansion under a
  subgoal), still atomic at the MDP level; what is excluded is a single verb that
  solves the whole problem.
- No regression of the classical correctness anchors: a small stochastic
  Gridworld with value iteration, expectimax, and MCTS triangulated against
  hand-computed values (in the style of Section 2's example) is a standing
  validation requirement, a non-regression gate the LLM system must never break.
- No search-first reframing. The central object is the agent (policy and $Q$)
  learning over the induced context MDP; MCTS is the policy-improvement operator
  and the compute knob.

Honesty boundary: this `(verb, target)` lens system is the designed method. Any
presentation of this work marks explicitly which designed components (the
pointer head, scoped lenses, the reduced verb set, optional `CRITIQUE`) are
realized in the evaluated artifact and which are not; nothing still specified is
claimed as built. Style: expanded sums before expectation notation, symbols
defined at first use, hand-computable examples before general algorithms.

---

# Appendix A. Prior-art grounding for the verb set

Across roughly thirty reasoning systems (Tree of Thoughts, Graph of Thoughts,
RAP, LATS, ReST-MCTS\*, rStar and rStar-Math, MCTSr, TS-LLM, Self-Refine,
Reflexion, CRITIC, ReAct, Least-to-Most, Decomposed Prompting, Plan-and-Solve,
ADaPT, Program-of-Thought and PAL, AlphaProof, AlphaGeometry, DeepSeek-R1,
Tulu-3, STaR and V-STaR, Stream of Search and Guided Stream of Search (GSoS),
Searchformer, Self-Backtracking, ReVISE, SCoRe, ASTRO), the operation vocabulary
collapses to five families: propose/expand, verify/evaluate, critique/refine,
commit/close, prune/backtrack. The verb set of Section 4 covers each family:
propose/expand contributes `SPLIT` (expand) plus `FORM_HYPOTHESIS` and
`COMPUTE_SUBEXPR` (propose, split hypothesis-versus-fact), and the four
remaining families contribute one verb each, with `CRITIQUE` optional.

Key data points behind the design:

- **Minimal verifier-anchored systems are tiny.** AlphaGeometry has two verbs
  (construct, deduce); AlphaProof is a single tactic space. This validates
  aggressive minimization.
- **Reward source is the load-bearing axis.** DeepSeek-R1 rejects learned PRMs;
  CRITIC shows external grounding is what makes critique reliable. Strong support
  for the verifier-only reward of Section 6.
- **Backtracking is search-control by default.** Every MCTS system puts it in
  selection; it becomes a learned, emittable action only in the
  internalized-search cluster (Self-Backtracking, ReVISE, SCoRe, ASTRO, GSoS),
  which is exactly the $B = 0$ deployment case that justifies `ABANDON` as a verb.
- **Closest structural analogue: TS-LLM** (PUCT MCTS, a learned value, and policy
  distillation, that is, generalized policy iteration). It differs principally in
  reward source: it trains an outcome reward model, whereas this project uses a
  symbolic verifier as the sole, untrained reward. That difference is the main
  novelty.
- **GSoS operation-level MDP** (each search operation is one discrete action) is
  the closest analogue to the `(verb, target)` action MDP and supports
  operation-level over token-level credit assignment.

---

# Appendix B. The policy-gradient alternative

For completeness, the on-policy alternative to the value-based core of Section 7:
sample trajectories under the factored policy, compute returns from the verifier
terminal, and update by policy gradient with a learned value baseline (REINFORCE,
PPO, or GRPO-style). The recent strong results in verifiable-reward reasoning are
on this side, but at the token level over whole traces, which is the o1/R1 recipe
this system contrasts against. Policy gradient is also on-policy, so it turns MCTS
from an additive add-on into a fork (importance weighting, or a switch to
distillation). For a small discrete action space with a natural learned $Q$, and
to keep MCTS additive, the value-based core is preferred; policy gradient is the
recorded alternative, not the default.

---

# Appendix C. Positioning against prior art (novelty)

A full assessment (about twenty systems checked against abstracts, with
per-work differentiation sentences) lives in
`docs/superpowers/research/2026-07-02-related-work-and-novelty.md`. The
condensed verdict follows. Appendix A grounds the verb set; this appendix
addresses publishable novelty, a different question.

**The four named threats.**

1. **TS-LLM** (Feng et al. 2023, arXiv:2309.17179): AlphaZero-style
   generalized policy iteration for LLM reasoning (PUCT MCTS, learned value,
   policy distillation), trained against a learned outcome reward model. A
   reviewer can frame this design as "TS-LLM with the reward model swapped for
   SymPy" unless the external-tiny-controller, pointer-action, and
   frozen-generator elements are foregrounded as the contribution.
2. **The combination argument.** ArCHer (arXiv:2402.19446) establishes
   off-policy value-based multi-turn RL for LLM agents; GSoS
   (arXiv:2410.02992) establishes the operation-level action MDP; RLVR
   establishes verifier reward. Each ingredient is published, so a skeptical
   reviewer will call the combination engineering rather than a new idea.
3. **Active Context Curation via RL** (Li et al. 2026, arXiv:2604.11462),
   concurrent work: a lightweight RL policy curates what a frozen executor
   model sees. Closest existing work on the lens axis; must be cited as
   concurrent and differentiated (their curator is a 7B model doing context
   pruning; our lens is a byproduct of typed reasoning operations chosen by an
   under-5M verifier-anchored Q controller).
4. **DialogXpert** (arXiv:2505.17795, 2025): a frozen LLM proposes candidate
   actions per dialogue turn and a compact external Q-network over frozen
   text embeddings, TD-trained, selects among them. The nearest neighbor on
   the controller shape itself (value-based external control of a frozen
   LLM). Differentiation: its actions are open-domain dialogue moves, its
   reward is task and emotion feedback, and it has no typed operations, no
   pointer over a symbolic state, no verifier, and no lens. Related cluster:
   LLM priors folded into value-based RL (LINVIT arXiv:2402.16181; Yan et
   al. arXiv:2410.07927; LLM-guided Q-learning arXiv:2405.03341).

**What survives as novel: the conjunction, claimed as such.** No single prior
work has: (a) a tiny external controller that is neither the LLM nor an
LLM-sized critic; (b) the factored verb head plus pointer head over the live
node identifiers of a growing reasoning forest; (c) off-policy value-based
Q-learning as the training spine with MCTS strictly optional and additive;
(d) a symbolic verifier as the sole reward with no learned value, PRM, ORM,
or critic anywhere in the reward path; (e) the lens property, one action
selecting both the operation and the scoped context of the frozen generator.
Honest caveat: each element individually has a precedent (pointer networks,
RLPrompt/TEMPERA's small-policy-over-frozen-LM, DialogXpert's external
Q-network over a frozen LLM, ArCHer's value-based RL, GSoS's operation MDP,
RLVR's verifier reward, theorem-proving GNN state encoders, DQfD-style
teacher bootstrapping). The paper must claim the conjunction and the
small-controller compute-frontier result, not a new RL algorithm, a new
encoder idea, or a new bootstrapping idea. The verification addendum in the
research memo (Part 3) records the full evidence for this narrowing.
