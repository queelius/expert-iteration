# Citation Verifier Report

Scope: each citation exists, is accurately summarized, and is relevant.

## Summary

The bibliography has 28 entries, all cited at least once in the body. No orphan bib entries. No undefined citation references in the bibtex log. Most citations are accurate. Four entries are marked TODO in the bibliography header for verification; three of these have been addressed inline (with explanatory notes), and one (AlphaProof blog reference) is acknowledged as awaiting a formal publication.

The most consequential missing citations are documented in the literature-context packet rather than this report. This report focuses on accuracy of existing citations.

## Per-citation verification

### Verified accurate

- **sutton2018reinforcement.** 2nd edition, MIT Press 2018. Correct.
- **kocsis2006bandit.** Bandit-based Monte Carlo planning, ECML 2006. Correct.
- **coulom2006efficient.** Efficient selectivity and backup operators in Monte Carlo tree search, ICCG 2006. Correct.
- **browne2012survey.** A survey of MCTS methods, IEEE TCIAIG 2012. Correct.
- **silver2016alphago.** Nature 2016, AlphaGo paper. Correct.
- **silver2017alphagozero.** Nature 2017, AlphaGo Zero. Correct.
- **silver2018alphazero.** Science 2018, AlphaZero. Correct.
- **schrittwieser2020muzero.** Nature 2020, MuZero. Correct.
- **anthony2017thinking.** NeurIPS 2017, expert iteration / Hex paper. Correct.
- **mnih2015human.** Nature 2015, DQN. Correct.
- **hu2022lora.** ICLR 2022, LoRA. Correct.
- **lightman2024verify.** ICLR 2024, "Let's Verify Step by Step." Correct.
- **cobbe2021gsm8k.** arXiv 2110.14168. Correct.
- **hendrycks2021math.** arXiv 2103.03874. Correct.
- **li2022alphacode.** Science 2022. Correct.
- **wei2022cot.** NeurIPS 2022. Correct.
- **snell2024scaling.** arXiv 2408.03314. Correct.
- **zelikman2022star.** NeurIPS 2022 STaR. Correct.
- **hosseini2024vstar.** arXiv 2402.06457. Correct.
- **feng2023alphazerollm.** arXiv 2309.17179. Correct.
- **xie2024monte.** arXiv 2405.00451. Correct.
- **willard2023outlines.** arXiv 2307.09702. Correct.

### Needs verification / has caveats

- **deepmind2024alphaproof.** Blog reference. The paper's note in the bibliography acknowledges this is the only available citation as of 2026-Q2. Status: appropriate given the constraint, but flag for replacement when a formal paper appears. The framework paper's claim that AlphaProof uses "progressive widening with LLM-proposed candidates" (in §4.2) is plausible but not citable against the blog content; the paper should soften this to "appears to use" or "is consistent with descriptions from."
  - Suggestion: Change §4.2's "This is the pattern in AlphaProof \citep{deepmind2024alphaproof} and AlphaCode \citep{li2022alphacode}" to "This is the pattern reported for AlphaProof \citep{deepmind2024alphaproof} and used in AlphaCode \citep{li2022alphacode}." Acknowledges the source-quality difference.

- **openai2024o1.** OpenAI o1 system card. URL points to the openai.com/o1/ landing page rather than to the system card itself. The system card was published in 2024 with a different URL; the current entry will not lead a reader to the actual document.
  - Suggestion: Update the URL to the actual system card location (https://cdn.openai.com/o1-system-card-20240917.pdf or the current authoritative URL).

- **yang2024qwen25.** Qwen2.5 Technical Report, arXiv:2412.15115. Verified the arXiv ID is correct.

- **dubey2024llama3.** "The Llama 3 Herd of Models", arXiv:2407.21783. Verified the arXiv ID is correct. The bibliographic title is slightly informal (the official title is "The Llama 3 Herd of Models" which matches).

- **allal2024smollm2.** SmolLM2 Hugging Face technical report. The URL points to the model page rather than to a paper. A more formal citation may exist by 2026; if not, the current Hugging Face link is appropriate.

- **guo2025deepseekr1.** arXiv:2501.12948. Verified the arXiv ID is correct.

### Accuracy of in-text citation context

I spot-checked the in-text use of each citation against its actual content. Findings:

- **lightman2024verify in §6.2** is summarized as: "demonstrated that process reward models trained on per-step labels for MATH solutions outperform outcome-only supervision on the same benchmark." Accurate.
- **anthony2017thinking in §6.1** is summarized as: "The expert-iteration formulation of Anthony et al. states the underlying pattern abstractly: a fast apprentice policy is improved by an expensive expert operator (search), and the apprentice is then trained to imitate the expert's behavior." Accurate.
- **kocsis2006bandit and coulom2006efficient in §2.6** are presented as the originators of MCTS and UCT. Accurate.
- **silver2017alphagozero in multiple places** is cited for the AlphaGo Zero result and for the PUCT formula. Both uses are accurate.
- **snell2024scaling in §6.3** is summarized as: "established empirical scaling laws relating test-time compute to accuracy on verifiable benchmarks." Accurate.
- **guo2025deepseekr1 in §6.2** is summarized as: "report large-scale reinforcement learning against rule-based verifiers for math and code without a learned reward model, deliberately avoiding the Goodhart attack surface that attends preference-trained critics." Accurate.
- **mnih2015human in §2.4 and §4.3** is cited for the target-network technique. Accurate.
- **wei2022cot in §6.3** is cited for chain-of-thought. Accurate.
- **hu2022lora in §4.3 and §6.1** is cited for LoRA. Accurate.

### Coverage gaps (missing citations that should be added)

These are documented in the literature-context packet but listed here for completeness:

1. **rStar-Math (Guan et al. 2025)** is the most consequential missing citation. Should be added to §6.4 ("MCTS for LLM reasoning") with a contrastive sentence.
2. **Math-Shepherd (Wang et al. 2024)** would strengthen §6.2 ("Verifier-Anchored LLM Reasoning") as a precedent for per-step value estimation via rollouts.
3. **OmegaPRM (Luo et al. 2024)** as additional precedent for the MCTS-based per-step Q-estimation pattern.
4. **TS-LLM (Wan et al. 2024)** for §6.4 as another AlphaZero-style LLM search system.
5. **Tree of Thoughts (Yao et al. 2023)** as a precursor to the AlphaZero-LLM wave, worth a sentence in §6.4.
6. **ReST / ReSTEM (Singh et al. 2023, 2024)** for §6.5 ("Self-Improvement") as the closest non-search expert iteration in LLM literature.
7. **Brown et al. 2024 "Large Language Monkeys"** for §6.3 ("Test-Time Compute Scaling") as best-of-N coverage scaling.
8. **Gao et al. 2023 ("Scaling Laws for Reward Model Overoptimization")** for §4.6 and §6.3 as empirical support for the Goodhart claim against learned rewards.
9. **Skalse et al. 2022 ("Defining and Characterizing Reward Hacking")** alternative or additional citation for the Goodhart claim.

### Bibliography quality issues

- The bibliography is consistently styled (BibTeX `plainnat` format with `[numbers, sort&compress]` natbib option), no formatting inconsistencies.
- Author lists are appropriately abbreviated with "and others" for longer team papers (silver2016, li2022, schrittwieser2020, mnih2015, guo2025, yang2024, dubey2024, allal2024). This is correct natbib convention.
- Three "TODO" comments survive in the bib file header noting that AlphaProof, OpenAI o1, and base model citations need verification before submission. Two of these (Qwen2.5, Llama3) have been verified above; one (AlphaProof) is acknowledged as awaiting a formal publication.
- Suggestion: Update the bib header TODO comments to reflect verification status. Specifically, mark yang2024qwen25 and dubey2024llama3 as verified; keep the AlphaProof and OpenAI o1 TODOs as ongoing.

### Self-citation / companion paper

- The framework paper references "the companion paper at paper-recursive/" in §4.1 without a bibliographic citation. This is appropriate for a sibling submission that does not yet have a public preprint URL. When both papers are posted to arXiv, the companion paper should be cited formally; for the present draft, the path-based mention is acceptable.

## Verdict from citation-verifier

The existing bibliography is accurate and consistently formatted. Two minor URL/identifier updates (openai2024o1 URL; consider updating dubey2024llama3 if a published version of the Llama 3 report has appeared). Nine recommended additional citations cover the literature-context gaps. The bibliography is in adequate shape for an arXiv preprint and approaching acceptable shape for venue submission once the additions are made.
