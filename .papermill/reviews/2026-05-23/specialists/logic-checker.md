# Logic Checker Report

Scope: proofs, definitions, equations, logical inference chains, claim support.

## Summary

The paper is mathematically clean for a methodology paper. The notational scaffolding is consistent (the custom commands in main.tex preamble enforce uniform notation for $Q_\psi$, $Q^\ast$, $V^\ast$, $\widehat{Q}_\mathrm{MCTS}$, $\pi_\mathrm{MCTS}$, $p_\theta$). Equations are well-numbered, cross-referenced, and consistent across sections. One Proposition (chance-node convergence, §4.5) and one main argumentative structure (the Goodhart asymmetry argument, §4.6 and §6.3) are the load-bearing logical artifacts. Both are correctly stated as design-intent rather than as fully formal theorems, which is appropriate for a methodology paper but invites tightening in a few places.

## Findings

### MAJOR

**M1. Proposition 1 (chance-node convergence) understates its own conditions.**
- Location: §4.5, prop:mcts-convergence, p. ~15
- Quoted text: "Fix a state-action pair $(s, a)$ visited during MCTS, and let $N(s, a) \to \infty$. If the leaf evaluation $\widehat{V}$ is unbiased for $V^{\ast}$ (or, more generally, if its bias is bounded uniformly over the leaves visited under $(s, a)$ by a constant $b$), and if the selection rule continues to sample every descendant infinitely often, then $W(s, a) / N(s, a) \to Q^{\ast}(s, a)$ (up to bias $b$)."
- Problem: The proposition is stated cleanly but the conditions are weaker than needed for the stated conclusion. Specifically:
  - Unbiasedness of the leaf evaluator $\widehat V$ for $V^\ast$ does not by itself guarantee convergence of $\Qmcts$ to $Q^\ast$; it also needs PUCT to converge to the optimal action distribution at internal nodes, which is the substantive part of UCT-style analyses.
  - The phrase "samples every descendant infinitely often" is not the same as "samples every descendant from every state infinitely often"; the latter is what UCT's logarithmic regret bound gives.
  - The bias bound clause "up to bias $b$" is informal but not pinned down. If the leaf bias is $b$, the chance-node mean converges to $Q^\ast(s,a) + \gamma^{L_{\min}} b$ in expectation, where $L_{\min}$ is the minimum depth at which $\widehat V$ is invoked under that chance node. The simple "(up to bias $b$)" misleadingly suggests the bias does not depend on depth.
- The paper's own admission ("the proposition states the design intent rather than a fully formal guarantee") covers this, but the prose should be sharper. Either:
  - (a) Restate as a remark rather than a proposition; the sentence after the proposition is already calling it design intent. A formal proposition that the prose then waves away is awkward.
  - (b) Tighten the statement to be technically defensible: replace with a citation-anchored claim like "Under the standard UCT regret conditions of Kocsis-Szepesvari, adapted to the PUCT prior of Equation (\ref{eq:puct}), the chance-node mean $W(s,a)/N(s,a)$ is a strongly consistent estimator of $Q^*(s,a)$ when the leaf evaluator is consistent."
- Suggestion: Replace Proposition 1 with a Remark, citing the kocsis2006bandit result as the operative template (which the prose already does). This avoids the impression of a formal theorem that the paper does not prove.

**M2. The Goodhart argument in §6.3 conflates two senses of "the signal IS the objective."**
- Location: §4.6 ("The Goodhart argument") and §6.3 in full
- Quoted text (§4.6): "Verifier signals do not have this property because the signal IS the objective: the SymPy equality call is not a proxy for correctness, it is the operational definition of correctness for the problem class."
- Quoted text (§6.3): "When SymPy reports equality between a canonicalized answer and the ground truth, SymPy's verdict is correctness for the problem class. ... In each case, the signal and the objective are the same object, so optimizing against the signal cannot introduce a Goodhart gap."
- Problem: The claim "the signal IS the objective" is too strong. There are two distinct senses:
  - (a) "The signal is the operational definition for the problem class" (this is true: a benchmark is defined by its verifier).
  - (b) "The signal cannot be Goodharted" (this requires more: it requires that there is no slip between what the benchmark designer intended and what the verifier actually accepts).
  - Sense (a) does not entail sense (b). A buggy regex extractor, an over-aggressive SymPy canonicalization (which the threats-to-validity discussion in §5.7 explicitly flags), or a Lean tactic that produces a closed proof by abusing decidable instances all let the agent satisfy the verifier without satisfying the underlying intuition the verifier was meant to capture.
- The paper's threats-to-validity subsection (§5.7) acknowledges this for SymPy specifically ("a too-aggressive canonicalization can collapse distinct symbolic answers into a single equivalence class"). So the threat is known; the issue is just that §4.6 and §6.3 overclaim. The Discussion's §6.3 even concedes the point at the end ("The agent may still produce outputs the verifier accepts but a human evaluator would reject, but those cases are the verifier's responsibility, not the optimizer's"), but only after the strong claim has been made.
- Suggestion: Tighten the strong claim. Replace "cannot introduce a Goodhart gap" with "does not introduce the kind of Goodhart gap that learned-classifier signals do, because the gap, when it exists, is in the specification (the verifier's coverage) rather than in the optimizer's behavior." Or: distinguish "intrinsic Goodhart" (proxy vs true objective drift; ruled out by verifier-anchored signals) from "extrinsic Goodhart" (specification vs intent gap; still present and addressed by strengthening the verifier).

### MINOR

**m1. Equation (\ref{eq:bellman}) presentation.**
- Location: §2.2
- The expansion of the inner Bellman update into three steps ("(i) enumerate, (ii) collect, (iii) add discounted future") is pedagogically good but uses the phrase "for each transition" where "for each next state" would be more precise. A transition is $(s, a, s')$; the sum is over $s'$ alone. Minor wording polish, no logic error.

**m2. Approximate $Q$ loss (eq:approx-q-loss) vs training-loop $Q$ loss (eq:lossq) inconsistency.**
- Location: §2.4 vs §4.7
- §2.4 Equation (5) writes the loss as $\frac{1}{2}\mathbb{E}_{(s,a,r,s')\sim D}[(\Qpsi(s,a) - y)^2]$ with the $\tfrac{1}{2}$ factor.
- §4.7 Equation (\ref{eq:lossq}) writes it as $\sum_{(s,a)\in B}(\Qpsi(s,a) - \Qmcts(s,a))^2$ with no $\tfrac{1}{2}$.
- Both forms are correct (the factor is conventional, absorbed by the learning rate). But the inconsistency between background and method should be smoothed: either add $\tfrac{1}{2}$ to (\ref{eq:lossq}) or drop it from (5). Trivial fix.

**m3. Target $y$ definition (\ref{eq:bellmantarget}) appears unused in the loss equation it is supposed to feed.**
- Location: §4.3 ("Target $Q$ network") on p. ~11, then §4.7 loss equation (\ref{eq:lossq})
- The Bellman target $y = r + \gamma \max_{a' \in \mathrm{avail}(s')} Q_{\bar\psi}(s', a')$ is defined at (\ref{eq:bellmantarget}) but the actual $Q$ loss in (\ref{eq:lossq}) uses $\Qmcts(s,a)$ as the target, not $y$. This is intentional (the whole point of the framework is that MCTS supplies the target), but the prose around (\ref{eq:bellmantarget}) gives the reader the impression that this single-sample form will appear in training. The transition is smoothed in §4.7 ("what is novel is the use of MCTS chance-node estimates $\Qmcts(s, a)$ as the target in place of single-sample Bellman targets $r + \gamma \max_{a'} Q_{\bar\psi}(s', a')$"), but the §4.3 definition is left as if it were the operative target.
- Suggestion: Either (a) move the single-sample Bellman target equation (\ref{eq:bellmantarget}) to §4.7 where the contrast with $\Qmcts$ is made, or (b) add a sentence after (\ref{eq:bellmantarget}) in §4.3 that says "This single-sample target is replaced by the MCTS chance-node estimate in §4.7 (Equation \ref{eq:lossq}); the target network architecture is still relevant because $Q_{\bar\psi}$ continues to evaluate leaf states inside MCTS during simulation."

**m4. The action availability table (page ~9) defines `\textsc{CommitAnswer}` as available "when at least one answer-bearing claim exists" but this predicate is not formalized.**
- Location: §4.2, table after Definition 2
- "answer-bearing claim" is not defined. Is it any \texttt{Fact} or \texttt{Hypothesis}? Only \texttt{Fact}s with \texttt{source = "verified"}? Some subset of \texttt{Hypothesis} fields? This matters for the availability predicate's machine-checkability.
- Suggestion: Define answer-bearing as e.g. "exists $f \in F$ with $f.\texttt{source} \in \{\texttt{verified}, \texttt{derived}\}$ or exists $h \in H$." Pick the operational form and write it down.

**m5. The PUCT formula in (\ref{eq:puct}) uses $\Qpsi(s, a)$ but the prose immediately below says the exploitation term is "the chance node's empirical mean once visited and the head estimate $\Qpsi(s, a)$ otherwise."**
- Location: §4.5 PUCT selection, page ~14
- The equation should display $\widehat Q(s,a)$ (or similar overloaded notation) rather than $\Qpsi(s,a)$ to match the verbal explanation that the term is the empirical mean once visited. The background section's PUCT formula (\ref{eq:puct-bg}) gets this right by writing $\widehat Q(s,a)$ explicitly. The method section's PUCT formula should match.
- Suggestion: Replace $\Qpsi(s, a)$ in (\ref{eq:puct}) with $\widehat Q(s, a)$, with an explanatory sentence that $\widehat Q$ defaults to $\Qpsi$ when $N(s,a) = 0$ and to the chance-node empirical mean $W(s,a)/N(s,a)$ otherwise.

**m6. Reward Equation (\ref{eq:rewardterminal}) does not specify what happens at $r_T$ when the agent reaches $d_{\max}$ on a problem whose verifier would say `correct` if it could see the (uncommitted) committed_answer slot.**
- Location: §4.6
- The middle case "if depth limit reached without commit" returns 0. If the agent has built up a correct hypothesis but never fires CommitAnswer, does it really get zero reward? The phrasing "without commit" presumably refers to never executing CommitAnswer, so yes, zero. But in real implementation, this creates a subtle bug surface: the agent learning to satisfy this reward shape might never commit on hard problems, accepting 0 instead of risking -1. The training-loop dynamics here are worth a sentence of discussion: is this risk-aversion expected? Is it actually exploration-positive (the agent learns to commit only when confident)?
- Suggestion: Add one sentence acknowledging that 0 for unfired-commit and -1 for wrong-commit can create commit-aversion, and that this is monitored as part of the sanity-check ablations (or, if not, why not).

**m7. The fact-vs-hypothesis distinction is described as "load-bearing" but its operational consequence in the loss equations is not visible.**
- Location: §4.1 "The fact-versus-hypothesis distinction" paragraph
- The text correctly explains why the distinction matters epistemically. But there is no place in §4.7's loss functions or §4.5's MCTS dynamics where the distinction enters the math. It enters only via the availability predicate (\textsc{InvokeVerifier} available when a hypothesis or derived fact exists) and via the serialization template (separate marker tags). This is reasonable, but the prose claims more structural load-bearingness than the equations support. A reader expecting the distinction to enter the reward function or the value head will not find it there.
- Suggestion: Add one sentence: "The Fact/Hypothesis distinction enters the parametric components only through (a) the availability predicate in Section 4.2 and (b) the serialization template's marker tags in Section 4.3; the heads and the loss functions remain agnostic to the distinction beyond what the encoder picks up from the serialization."

### SUGGESTIONS (non-issues, but worth considering)

- The "Markov bet" paragraph in §4.1 is good. Could be cross-referenced explicitly from §2.7 ("Partial observability and the Markov bet"), since those two paragraphs are saying the same thing at different abstraction levels.
- The per-edge backup equation (\ref{eq:backup}) is correctly stated as a reverse accumulation. The pseudo-code style mixes `for` with `:=` notation; a `\State`-style line break would be cleaner. Stylistic only.
- §4.5's discussion of "Collapsing decision and chance nodes" is excellent but is buried at the end of a single dense paragraph. Worth splitting into a dedicated paragraph with its own emphasis: this is one of the load-bearing technical claims of the paper.

## Cross-section consistency

- Notation is consistent. $\Qmcts$, $\pimcts$, $\Qpsi$, $Q_{\bar\psi}$ all used identically across background and method.
- Equation numbering is clean.
- Cross-references (eq:bellman from method, eq:approx-q-loss from method) all resolve.
- One forward-reference convention: §4.6 forward-refers to §6.3, with explicit phrase "we develop this position in Section 6.3." Good practice for splitting a long argument across two locations.

## Verdict from logic-checker

The paper's logical structure is sound. Two MAJOR items (Proposition 1's tightness and the Goodhart claim's overstatement) should be addressed before any venue submission. Seven MINOR items are wording or cross-reference polish. No critical correctness errors.
