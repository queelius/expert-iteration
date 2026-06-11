# Logic Checker Report

**Paper**: "Bounded Structural Recursion in Verifier-Anchored Reinforcement Learning"
**Date**: 2026-05-27
**Focus**: Definition 1, Definition 2, productive-vs-unproductive distinction, hard-mask vs soft-penalty argument.

## Summary

The paper's formal apparatus is light but the definitions are well-formed. The argument that hard-mask depth limits preserve the verifier-anchored-reward-only principle holds, with one significant logical gap in how Definition 1 quantifies over trajectories versus subtrees. No critical issues.

## Definition 1 (Productive Recursion): well-formed but ambiguous

**Quoted text** (method.tex, lines 101-108):
> "A recursive subtree rooted at hypothesis $h$ is *productive* if there exists a finite trajectory from $h$ to a leaf that terminates in a successful \textsc{InvokeVerifier} call on a descendant hypothesis or in a \textsc{CommitAnswer} that the terminal verifier accepts. The subtree is *unproductive* otherwise."

**Finding (Major)**: The existential quantifier "there exists a finite trajectory" has an ambiguity. A subtree may admit:
- a trajectory that reaches a verifier-accepting leaf (so the subtree is productive by Definition 1), but
- the policy may never actually take that trajectory in practice.

Under Definition 1, *any* subtree with a verifier-reachable descendant is productive, even if no realized trajectory uses that path. This makes the definition vacuously satisfiable for almost any subtree where at least one InvokeVerifier on any descendant could succeed. The discussion section (line 23-30) addresses this implicitly when it discusses "marginal cases", but the definition itself does not distinguish *reachability under some policy* from *reachability under the trained policy*.

**Suggested remediation**: Strengthen Definition 1 to specify the quantification:

> "A recursive subtree rooted at hypothesis $h$ is *productive under policy $\pi$* if, with positive probability under the trajectory distribution induced by $\pi$, the trajectory from $h$ reaches a leaf that terminates in a successful InvokeVerifier..."

Or, alternatively, define two notions: *syntactically productive* (the existential as currently written) and *operationally productive* (under the policy distribution). The paper uses "productive" in both senses across different sections; making this explicit would clarify.

**Severity**: Major. Not venue-blocking, but a reviewer who looks closely will ask exactly this question, and the answer is not in the text.

## Definition 2 (Tree Depth): well-formed

**Quoted text** (method.tex, lines 145-154):
> "The *tree depth* of an entity (hypothesis, subgoal, or verifier call) is the length of the longest parent-pointer chain reachable from that entity by following parent_hypothesis (or parent_subgoal, or follow_up_to) repeatedly until a None field is reached."

**Finding**: Definition 2 is sound. The base case (root with parent = None has depth zero) is correctly identified. The definition is consistent with the depth markers in the serialization template (method.tex, lines 302-317).

**Minor issue**: The serialization example in lines 304-309 has an apparent inconsistency. Line 305 has `[OPEN_SUBGOAL id=sg_1 parent=none depth=0]` and line 306 has `[OPEN_SUBGOAL id=sg_2 parent=sg_1 depth=1]`. Then line 307 has `[HYPOTHESIS id=hyp_1 parent=sg_2 depth=1]`. By Definition 2, hyp_1's parent chain is hyp_1 -> sg_2 -> sg_1 -> None, so the chain length is 3, depth should be 2 (with sg_1 at depth 0, sg_2 at depth 1, hyp_1 at depth 2). The example shows hyp_1 at depth=1.

The resolution depends on whether parent chains *cross entity types* (subgoal to hypothesis) or only chase same-type pointers (hypothesis only follows parent_hypothesis). The definition mentions following "parent_hypothesis (or parent_subgoal, or follow_up_to)" as if all three were equivalent, but the serialization treats hypothesis depth as counting only along parent_hypothesis. This is a definitional inconsistency.

**Suggested remediation**: Either clarify in Definition 2 that depth is computed per-entity-type (depth of a hypothesis follows parent_hypothesis only, depth of a subgoal follows parent_subgoal only), or change the example to be consistent with cross-type chains.

**Severity**: Minor, but the example is the load-bearing illustration of the entire serialization argument, so fixing it matters for clarity.

## The hard-mask vs soft-penalty argument

**Quoted text** (method.tex, lines 190-204):
> "A soft-penalty alternative... was considered and rejected for two reasons. First, soft penalties interact with the verifier-anchored reward principle in ways we want to avoid: the framework paper reserves Bellman targets for verifier signals, and a depth penalty is exactly the kind of aesthetically motivated additive reward that the framework paper warns against. Mixing aesthetic and verifier rewards would muddy the Bellman targets that train $\Qpsi$. Second, a soft penalty makes the policy learnable but the state space exploration unbounded: an MCTS simulation could still descend arbitrarily deep, paying the penalty at each step but accruing no other information."

**Finding (Minor)**: The first argument is sound on framework-paper consistency grounds. The second argument has a subtle gap. A soft penalty *can* bound exploration depth if combined with discounting: a negative reward at each recursive step, accumulated with discount $\gamma$, eventually makes the expected return from further recursion negative, and a value-maximizing policy would stop. The paper's claim "an MCTS simulation could still descend arbitrarily deep" is true only of the *simulation*, not of the *optimal policy* under the modified reward. The paper conflates these.

**Suggested remediation**: Reframe the second argument to distinguish behavioral disincentive (under the learned policy) from exploration bound (in MCTS rollouts). The hard mask's distinguishing property is that it bounds simulation depth at the MDP level, before MCTS even has a chance to descend. This is a more precise statement.

**Severity**: Minor. Not venue-blocking.

## Consistency of "productive" across sections

**Finding (Minor)**: The term "productive" is used in three distinct ways across the paper:
1. **Definitional sense** (method.tex Definition 1): syntactic existence of a verifier-reaching trajectory.
2. **Operational sense** (introduction, line 13-21): trajectories that "deliver leaf-level verifier signal" under the policy.
3. **Regime sense** (experimental plan, hypothesis H1): the depth-limit value that empirically yields the productive trajectory distribution.

These are related but not identical. The paper would benefit from a one-sentence clarification (probably in §4.2 after Definition 1) that distinguishes the three.

**Severity**: Minor.

## Cross-paper logical coherence

**Finding**: The companion paper inherits the verifier-anchored-reward-only principle from the framework paper. The argument that soft penalties violate this principle is consistent with framework paper §6 (verifier-anchored reward). No conflict detected.

## Verifier-reach claim and the MCTS interaction

**Quoted text** (method.tex, lines 428-442):
> "A subtle question: when MCTS backs up a verifier reward from a leaf hypothesis hyp_3 (which is a child of hyp_1), does the reward flow through the chance-node parents (the InvokeVerifier action that produced the verifier call) or through the structural parents (the parent_hypothesis link)? In our design, only the former."

**Finding**: This is a well-stated subtlety. The implication is correct: MCTS backups follow the search tree's decision-and-chance-node structure, not the implicit hypothesis tree. The state representation (via serialization) gives $\Qpsi$ access to the hypothesis tree implicitly, but the MCTS backup operator does not propagate reward along the parent_hypothesis links.

The logical consistency check: this means a verifier-anchored reward at a deep leaf hypothesis enters $\Qmcts$ only via the decision-step trajectory from the root of the MCTS tree, not via the structural ancestors of the verified hypothesis. This is consistent with treating MCTS as a Bellman-equation simulator over the (state, action) trajectory, where the state happens to contain implicit-tree information.

## Findings summary

| ID | Issue | Severity | Venue-blocking? |
|----|-------|----------|-----------------|
| L1 | Definition 1's existential quantifier ambiguity (syntactic vs operational productivity) | Major | No |
| L2 | Serialization example depth inconsistency (cross-type vs same-type parent chains) | Minor | No |
| L3 | Hard-mask vs soft-penalty argument conflates simulation depth and policy depth | Minor | No |
| L4 | "Productive" used in three distinct senses without explicit disambiguation | Minor | No |

No critical issues. The formal definitions are well-formed enough to support the paper's claims, but Definition 1's quantifier ambiguity should be addressed before submission.
