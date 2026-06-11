# Prose Auditor Report

Scope: writing quality, narrative structure, notation consistency, communication effectiveness.

## Summary

The prose is generally strong: high-density technical writing with consistent voice, careful notation, and effective use of paragraph-level signposting. The recent RL-first reframe has been applied uniformly; no residual MCTS-first language survives. The paper reads as a methodology paper rather than a results paper, which is the right register. Several specific places benefit from prose tightening, and a small number of paragraphs would benefit from restructuring into shorter units. The narrative arc is sound.

## Findings

### MAJOR

**M1. §4.5's "MCTS as a Policy Improvement Operator" opens with a single 350-word paragraph that contains three distinct moves.**
- Location: §4.5, first paragraph (lines 543 to 577 in method.tex)
- Quoted opening: "The parametric Q_psi and p_theta from approximate Q-learning are useful, but on a fresh state s they encode only the average behavior accumulated across the training distribution; they have not yet allocated any compute specifically to s. MCTS produces locally improved estimates by sampling the environment more intelligently than the behavior policy would..."
- Problem: This single paragraph (a) motivates MCTS as a local-compute mechanism, (b) sets up the chance-node bifurcation argument, and (c) explains why neither collapsed-node nor model-based-MCTS variants are usable. Each of these moves deserves its own paragraph. The current density makes the section's most important technical argument (the chance-node bifurcation rationale) easy to miss.
- Suggestion: Split the paragraph into three. Paragraph 1: motivation (why MCTS at all). Paragraph 2: the chance-node bifurcation argument (why decision/chance nodes are split). Paragraph 3: the position between collapsed-node and model-based extremes. This is the section's intellectual climax; it should be paced for emphasis.

**M2. §4.1's "Markov bet" paragraph and §2.7's "Partial observability and the Markov bet" paragraph cover the same idea twice.**
- Location: §2.7 (last paragraph of background) and §4.1 (Markov bet paragraph)
- Problem: Both passages discuss whether the structured context can be treated as a Markov state and what to do when the projection is lossy. The arguments are not contradictory, but the redundancy weakens the prose: a reader encountering the §4.1 passage will recognize the §2.7 framing and wonder why both exist.
- Suggestion: Either (a) merge the §2.7 passage into §4.1 (since the LLM-context-MDP framing is the concrete instance that §2.7 introduces abstractly), or (b) keep both and add an explicit cross-reference: "Following the framing of §2.7, the structured-context approximation here makes the same bet at the operational level."

**M3. Contribution C5's phrasing buries the most important word.**
- Location: §1, contribution C5
- Quoted text: "A pre-registered evaluation plan (Section 5) that defines the baseline grid, the compute axis, and the headline compute-versus-accuracy frontier figure before any training begins"
- Problem: "Pre-registered" is the load-bearing word here, but it appears as the modifier of "evaluation plan," at the start of the sentence, with no further emphasis. The actual novelty (committing the falsification criterion in advance) is mentioned second. A reader skimming the contribution list will see "evaluation plan" and assume this is the obligatory experiments section.
- Suggestion: Lead with the novelty: "A falsifiable pre-registration of the evaluation plan (Section 5): the baseline grid, the compute axis, the headline compute-versus-accuracy frontier figure, and the conditions under which the framework's headline claim is falsified are all committed before any training begins."

### MINOR

**m1. The abstract is one long paragraph; the journal/conference convention varies but a two-sentence opening hook would help.**
- Location: abstract.tex
- The abstract is dense and information-rich (good) but reads as a continuous block. Three sentences in particular do narrative work that would be more effective if separated: the opening setting-out sentence, the "we separate verifier-anchored from learned-aesthetic signals" sentence (which is the conceptual high point), and the closing "empirical results are deferred to a follow-up study" sentence.
- The single-paragraph form is a stylistic choice that is defensible for arxiv. For workshop or conference submission, breaking it into one to three sentences of setup, three to five sentences of method summary, and one sentence of scope-statement would be more conventional.
- Suggestion: Leave the abstract structure as-is for the current draft; revisit when a target venue is selected.

**m2. §4.6 ("Reward Structure") uses "two familiar failure modes" as its opening framing, then describes (a) sparse verifier reward and (b) Goodhart-able learned signals.**
- Location: §4.6, opening paragraph
- Problem: "Familiar failure modes" is slightly misleading. The sparse-reward issue is a well-known practical problem; the learned-signal Goodhart issue is a structural argument the paper is itself developing. Calling them both "familiar" undersells the second.
- Suggestion: Replace "two familiar failure modes" with "two failure modes that operate against each other in standard reward designs." The contrast between operational sparsity and structural Goodhart is what makes the paper's middle-path choice interesting.

**m3. §4.5's "fully model-based MCTS that enumerates P(s' | s, a) exactly is also unavailable to us" is correct but the phrase "is also unavailable" is too soft.**
- Location: §4.5, in the dense opening paragraph (covered by M1)
- Problem: It is not just that fully model-based MCTS is "unavailable"; it is that enumeration is in principle impossible because the LLM-induced transition has infinitely many continuations. The current phrasing reads as a practical limitation rather than a fundamental one.
- Suggestion: Replace "is also unavailable to us" with "is in principle infeasible because the LLM-induced transition has effectively infinite continuation cardinality."

**m4. §4.7's algorithm block (Algorithm 1) does not explicitly indicate which lines are inner vs outer loops.**
- Location: §4.7, alg:outer-loop
- The pseudocode uses standard algorithmic style with `\For{iteration}` and `\For{minibatch step}`. The outer loop runs across "iteration"; the inner loop runs across "minibatch step." This is correct but the visual structure does not make the nesting obvious because both `\For` blocks contain similar verbs (Collect, Compute, Add, Sample, Step, Sync).
- Suggestion: Add a line comment after the inner `\EndFor` saying "% end inner training loop", and similarly after the outer `\EndFor`. Or restructure the algorithm so the outer loop's role (trajectory collection + target update) is visually separated from the inner loop's role (gradient steps).

**m5. The phrase "frozen LLM" appears throughout in three slightly different forms: "frozen LLM", "frozen base language model", "frozen base model".**
- Location: throughout
- Problem: The variations are not wrong but they add cognitive load. A consistent term ("frozen base LLM") used uniformly would reduce noise.
- Suggestion: Standardize on "frozen base LLM" (the term used in the contribution list and abstract); rewrite §4.3's variants to match.

**m6. §2.5 ("Exploration and Behavior Policies") lists five common behavior policies in a single sentence that is structured as a serial list (uniform, greedy, epsilon-greedy, softmax, search-guided).**
- Location: §2.5
- Quoted text: "Common behavior policies include the uniform random policy mu(a|s) = 1/|A(s)| (maximally exploratory, useful as a warm start); the greedy policy that always picks argmax_a Q_psi(s, a) (exploits current knowledge but can collapse onto a suboptimal action under biased Q_psi); the epsilon-greedy policy that flips between random and greedy with probability epsilon (typically annealed over training); the softmax / Boltzmann policy mu(a|s) prop exp(Q_psi(s, a)/tau) with temperature tau; and the search-guided policy mu(a|s) = pi_MCTS(a|s) produced by an online procedure such as MCTS..."
- Problem: 200-word run-on sentence in a section meant to be didactic. The semicolon-separated list is hard to parse.
- Suggestion: Convert to an itemized list (the section already uses itemize elsewhere). Each policy gets its own bullet; the comparison reads cleanly.

**m7. §4.4 ("Architecture") uses inline LaTeX terms like \texttt{[STATE\_END]} in math contexts; the typesetting works but reads cluttered.**
- Location: §4.3, §4.4, §4.5
- Problem: The mix of typewriter font tokens (the LLM sentinel) with math notation (h_T) in the same equation makes the equation harder to scan visually.
- Suggestion: Consider defining a custom \stateend command that typesets the sentinel consistently and is shorter to read. Stylistic only.

**m8. §6.3's title "Why Verifier-Anchored Signals Are Different" is good; the section body starts with "The asymmetry between verifier-anchored reward and learned-aesthetic reward is the most consequential design choice in this paper."**
- Location: §6.3
- Problem: The opening sentence is appropriately bold but tags itself as "the most consequential design choice" which sets a high bar that the rest of the section should clear. The section clears the bar in argument but the closing paragraph ("we acknowledge that this is a soft commitment rather than a theorem") partially walks the claim back. The result is a slight whiplash.
- Suggestion: Either (a) tone down the opening to "The asymmetry between verifier-anchored reward and learned-aesthetic reward is one of the most consequential design choices in this paper", or (b) tighten the closing paragraph so it acknowledges the soft commitment without retreating from the structural claim. The current version reads as if the section is hedging at the end.

### SUGGESTIONS

- §6.1 ("Limitations") opens with "The framework's reach is bounded by its reliance on a non-LLM verifier." Good lead. Consider adding one sentence at the very start of §6 that previews the discussion's overall arc: "We discuss limitations, extensions, the load-bearing position on reward-design (§6.3), threats to the empirical story, and the connection to LLM deliberation beyond math." A discussion-section overview helps the reader navigate.
- §6.4 ("Threats to the Empirical Story") substantially reprises §5.7. The split is intentional (operational vs forward-looking framing), but the §6.4 prose could be tightened by explicitly saying "we restate the operational concerns of §5.7 here as items for monitoring during implementation, with brief positioning for each as a forward-looking caveat." The current text re-explains each concern fully, which feels repetitive.
- §3 of the conclusion is good. The "expected experimental headline" sentence ("matches a substantially larger reasoning model at matched inference compute") is appropriately conditional and ties the paper to the eventual empirical study.
- §4.4's "Phase 1 freezing rationale" paragraph is mostly repeated in §4.7's "Phase 1 backprop scope" paragraph. Same content, different framing. Consider consolidating into one passage.
- §4.5's chance-node-vs-decision-node terminology is introduced cleanly with the bullet list. The diagram in figures/architecture.tex does not show the search tree; a second figure depicting the alternating decision/chance node structure would help readers who learn visually. This is a SUGGESTION rather than a finding because the prose alone is adequate.
- The introduction's "Position" paragraph contains the sentence "The reframing relative to prior MCTS-centric work is deliberate." Good. The subsequent sentences explain the practical consequences clearly. The "This shift in emphasis has practical consequences" sentence is well-placed.

## Notation consistency

- Q_psi, Q*, V*, Q_MCTS, pi_MCTS, p_theta are used consistently across all sections.
- Available action set A(s) is consistently written; the "avail" predicate appears in three forms (avail(s, a), avail(s), \mathrm{avail}(s)). Stylistic but worth unifying.
- The state symbol is mostly s, occasionally C (in §2.7 and the LLM-as-context-MDP discussion). This is intentional (C for context, s for the abstract MDP state), but the transition at the start of §4.1 from C back to s is worth a brief sentence: "We use s for the structured context throughout §4 to align with the standard MDP notation of §2."
- The sentinel \texttt{[STATE\_END]} is used consistently.
- The verb names \textsc{Decompose}, \textsc{ComputeSubexpr}, etc., are consistent.

## Narrative arc

- Introduction sets the problem (test-time compute on verifiable tasks; gap between small and large models).
- Background develops the RL vocabulary in a sensible order.
- Method specifies the framework.
- Experimental plan pre-registers the evaluation.
- Related work positions the contribution.
- Discussion deepens limitations and the verifier-vs-aesthetic argument.
- Conclusion wraps up.

The arc is clean. The companion paper on bounded structural recursion is referenced once in §4.1 with a single forward-pointing sentence, which is the right level of acknowledgment for a sibling submission.

## Verdict from prose-auditor

The prose is at a publishable level for a 2026 methodology paper. Three MAJOR items (M1 paragraph split, M2 Markov-bet redundancy, M3 C5 phrasing) and eight MINOR items would tighten the writing. The narrative arc is sound and the RL-first framing is uniformly applied. No critical writing issues.
