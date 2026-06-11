# Prose Auditor Report

**Paper**: "Bounded Structural Recursion in Verifier-Anchored Reinforcement Learning"
**Date**: 2026-05-27
**Focus**: narrative arc, definition introduction order, Pythagorean motivating example, implicit-vs-explicit hierarchy position, five-contribution list, banned phrases, notation consistency.

## Summary

The writing is strong throughout. The narrative arc is clear: the framework paper introduces recursion as a forward-compatibility hook; this paper takes that activation seriously and develops the design dimension. The Pythagorean-theorem motivating example anchors the abstract claim well. The implicit-vs-explicit hierarchy position in the discussion is a load-bearing methodological contribution that justifies the paper's existence.

Three minor prose issues, no critical ones.

## Banned-phrase check

**Findings**: Clean.
- No em-dash characters (U+2014) detected.
- No LaTeX three-hyphen sequences detected.
- The banned verb form (the L-word) is not used.
- The banned phrase about cross-disciplinary positioning is not used.

The paper respects the banned-phrase constraints.

## Narrative arc

The arc is:
1. **Abstract**: motivation (framework paper introduces recursion hooks, defers them), thesis (two modes: productive and unproductive), three design principles (depth limit, identifier targeting, flat serialization), pre-registered ablation.
2. **Introduction**: same arc expanded, with Pythagorean motivating example anchoring the productive-vs-unproductive distinction.
3. **Background**: notation recap, structured context MDP, hierarchical RL, tree encoders.
4. **Method**: the core development. Six subsections covering each design principle in turn.
5. **Experimental plan**: pre-registered ablation, hypotheses, secondary measurements, threats, reporting.
6. **Related work**: positioning against four threads.
7. **Discussion**: limitations, extensions, when-it-pays-off, threats, implicit-vs-explicit position.
8. **Conclusion**: brief restatement.

**Finding**: The arc is well-constructed. The five-contribution list in the introduction maps cleanly to method subsections (verified in novelty-assessor report). The discussion's implicit-vs-explicit hierarchy position is a strong closing methodological argument.

## Pythagorean motivating example

**Quoted text** (introduction.tex, lines 20-37):
> "Consider a triangle with vertices $A$, $B$, $C$ and side lengths $a$, $b$, $c$, and the claim that $a^2 + b^2 = c^2$. A monolithic verifier asked to confirm this claim must perform two qualitatively different checks at once... A decomposed approach instead splits the parent claim into two sub-claims..."

**Finding**: Excellent motivating example. Concrete, immediately legible, and the decomposition is intuitive without requiring domain expertise. The example bridges the abstract framework (state schema, verifier reach) and a familiar mathematical claim.

**Minor concern**: The example assumes the reader accepts that the Pythagorean theorem genuinely has two qualitatively different sub-checks (geometric and arithmetic). For a reader who thinks of the Pythagorean theorem as a single claim about distances in Euclidean space, the decomposition feels slightly artificial. A second example (perhaps from competition math: "prove that the sum of the first n cubes equals (n(n+1)/2)^2" decomposed into an induction base case and a step case) would broaden the appeal. Not necessary; one good example is sufficient.

## Definition introduction order

**Finding**: Definition 1 (productive recursion) appears at method.tex line 101, after a substantial discussion of the two failure modes (lines 55-99). This is the right order: motivation first, then formal definition. Definition 2 (tree depth) appears at line 145, immediately when needed for the depth-limit discussion. Both definitions are placed correctly.

**Minor concern**: Definition 1 uses the term "leaf" without prior definition. In the implicit-tree structure, a leaf is a hypothesis with no children entries. This is implicit in the schema, but a one-line clarification ("by 'leaf' we mean an entity with no children entries in the implicit tree") would help.

## Implicit-vs-explicit hierarchy position

**Quoted text** (discussion.tex, lines 184-234):
> "The hierarchical reinforcement learning literature has spent two decades architecting explicit hierarchy... Each of these designs makes hierarchy a property of the policy or value architecture: separate networks, separate parameter sets, separate optimization signals at each level. The framework paper adopts the opposite methodological stance, and the present companion paper develops it in depth..."

**Finding**: This is the paper's strongest argumentative passage. The position is clear, the trade-off is honest (implicit hierarchy gives up the inductive bias that explicit machinery provides, in exchange for letting LLMs' pretraining knowledge carry more of the work), and the conclusion is precisely the position the entire paper is in service of: "keep the architecture as simple as possible... and let hierarchy emerge from the state representation rather than from the policy architecture."

**Minor concern**: The passage is placed in the discussion section. Some readers will not reach the discussion before forming an overall impression. Consider moving (or duplicating in shorter form) the core argument to the introduction's position paragraph (introduction.tex lines 67-92). The introduction's position paragraph currently restates the framework, but the implicit-vs-explicit methodological commitment is what most differentiates this paper from prior work. Surfacing it earlier would let a quick reader see the methodological stake before reaching §6.5.

## Five-contribution list

**Finding**: The five-contribution list (introduction.tex lines 96-164) is well-constructed:
- Each contribution is described in 5-8 lines of prose.
- Each contribution cites the specific method subsection where it is developed.
- The order is logical: definitional (Contribution 1) to mechanism (Contribution 2) to action-space implication (Contribution 3) to encoder implication (Contribution 4) to empirical test (Contribution 5).

**Minor concern**: Contribution 5 (the pre-registered ablation) is qualitatively different from Contributions 1-4 (which are technical/conceptual). Some venues would prefer to see Contribution 5 framed as "an empirical commitment to testing Contributions 1-4" rather than as a parallel contribution. Stylistic preference; not a defect.

## Notation consistency

**Finding**: Custom commands defined in main.tex (lines 58-77) are used consistently throughout the section files:
- $\Qpsi$, $\Qstar$, $\Vstar$, $\Qmcts$, $\pimcts$, $\ptheta$
- $\Aset$, $\Sset$, $\Mset$
- $\E$, $\R$
- $\depthlim$ (the recursive-paper addition)
- $\dmax$ (defined but checked: it appears in the experimental plan as $d_{\max}$ in the table caption; this is fine because $\dmax$ is defined as `D_{\max}` and rendered correctly)

No notation inconsistency detected.

**Minor concern**: The verb names use \textsc (small caps) in some places but appear as plain `Decompose` (without \textsc) inside the listing block (method.tex line 251 example). The listing block uses typewriter font, so this is consistent with the LaTeX style, but a reader might briefly wonder. Not worth changing.

## Sentence-level prose issues

Spot-checking. Most prose is clean. A few minor issues:

**Issue 1** (introduction.tex line 11): "Approximate Q-learning over structured context MDPs offers a principled way to teach small language models to deliberate." The opening sentence is correct but slightly anodyne. A more vivid opening might say what the principled way *is* (or *isn't*).

**Issue 2** (method.tex line 76): "may split naturally into 'the antiderivative is $\tfrac{x^2}{2}$' and 'substitution gives $\tfrac{1}{2} - 0$,'" - the substitution example assumes the reader can connect "the antiderivative is x^2/2" to the integral "the integral over [0,1] equals 1/2." This is fine for a mathematically literate reader but the chain (antiderivative to evaluation at endpoints) is slightly compressed.

**Issue 3** (related_work.tex line 184): "Our position relative to this line is methodological rather than competitive." Good sentence; clear positioning.

**Issue 4** (discussion.tex line 224-227): "The commitment is, then, methodological: keep the architecture as simple as possible..." This is the closing summary of §6.5. It is the right summary and well-placed.

## Length and pacing

The paper is 23 pages. This is appropriate for a methodology-and-position companion paper. The framework paper is 37 pages; the companion's shorter length reflects its narrower scope. The method section (~10 pages by LaTeX line count) is the longest, which is correct: the method is the substantive contribution.

**Finding**: Length is appropriate. The paper does not redo framework content (verified by reading the manuscript: framework content is referenced descriptively rather than restated). The 23-page length is consistent with a companion paper.

## Findings summary

| ID | Issue | Severity | Venue-blocking? |
|----|-------|----------|-----------------|
| P1 | Implicit-vs-explicit hierarchy position is in discussion, should surface earlier (intro or related work summary) | Minor | No |
| P2 | "Leaf" used in Definition 1 without prior definition | Minor | No |
| P3 | Pythagorean example might benefit from a second illustration (induction example) | Suggestion | No |
| P4 | Opening sentence of introduction is anodyne | Suggestion | No |

No critical prose issues. The paper's writing is among the cleaner manuscripts I've reviewed in this review pass. The Pythagorean motivating example and the implicit-vs-explicit hierarchy position are both load-bearing rhetorical anchors, and both succeed.
