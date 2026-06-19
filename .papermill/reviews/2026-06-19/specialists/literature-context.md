# Literature Context Packet (merged from two scouts)

**Date**: 2026-06-19
**Paper**: Verifier-Anchored Reinforcement Learning for Small Language Model Deliberation
**Scouts**: literature-scout-broad, literature-scout-targeted

## RL-first framing (both scouts independently agree)

The paper is consistently RL-first. Approximate Q-learning is the core; MCTS is
subordinated as a policy-improvement operator that supplies training targets and a
test-time-compute knob. No scout found MCTS-first drift in the prose. One latent
risk both scouts note: the paper's sharpest structural differentiators (chance-node
bifurcation, per-edge backup) are MCTS-internal mechanics, so a reviewer skimming the
contrast paragraphs could misread the lineage. The prose mitigates this by subordinating
those mechanics to "honest Bellman-expectation estimator for the Q-target." It holds, but
it is doing work. Several recommended missing citations (VerifierQ especially) reinforce
the RL-first framing because they are Q-learning-over-LLM-MDP works rather than tree-search
works.

## Closest / competing works (deduplicated, ranked by review impact)

### Critical-tier omissions (each undercuts a specific load-bearing novelty claim)

1. **Stochastic MuZero** (Antonoglou, Schrittwieser, Ozair, Hubert, Silver, ICLR 2022,
   "Planning in Stochastic Environments with a Learned Model", openreview X6D9bAHhBQ1).
   NOT CITED. Introduced decision/chance-node (afterstate) bifurcation for stochastic MCTS,
   exactly the mechanism the paper presents as its own structural contribution (method.tex
   ~547-583, related_work.tex ~134-142, background.tex ~266-274), contrasting only against
   "deterministic-game tutorials." The paper cites plain MuZero (schrittwieser2020muzero)
   but not the stochastic variant that owns the chance-node mechanism. Genuine residual
   novelty: applying chance-node bifurcation to the non-enumerable LLM-induced transition.

2. **VerifierQ** (Qi, Tang, Zhu, arXiv 2410.08048; AI-for-Math Workshop at ICML 2025).
   NOT CITED. Closest prior work on the paper's CORE axis: offline Q-learning (IQL/CQL,
   modified Bellman update) over an LLM MDP for test-time compute, framing generator-verifier
   as actor-critic. The paper differentiates extensively from MCTS-centric work (rStar-Math)
   but never from the one prior Q-learning-centric system. Different framing (verifier-as-
   Q-function vs the paper's multi-head Q over a structured context state with verifier-as-action),
   so the overlap is addressable by cite-and-contrast, not fatal.

### Major-tier omissions

3. **Tree of Thoughts** (Yao et al., NeurIPS 2023, arXiv 2305.10601). NOT CITED. Canonical
   deliberate-search-over-thoughts; a paper titled "...Deliberation" with an MCTS-for-LLM
   related-work subsection conspicuously omits it.

4. **RAP / Reasoning via Planning** (Hao et al., EMNLP 2023, arXiv 2305.14992). NOT CITED.
   LLM as world model, MCTS over explicit LLM-maintained state; direct conceptual ancestor of
   the "LLM-induced context MDP with structured state" framing.

5. **ReST-MCTS\*** (Zhang et al., NeurIPS 2024, arXiv 2406.03816). NOT CITED. Per-step
   value-target distillation from process-reward-guided tree search; the correct comparison for
   the paper's "finer-grained per-step distillation vs STaR/V-STaR" claim (related_work 4.5
   contrasts only STaR/V-STaR).

6. **Math-Shepherd** (Wang et al., ACL 2024, arXiv 2312.08935). NOT CITED. Canonical automatic
   step-level process reward via MC estimation; the chance-node Q-estimate is conceptually the
   same MC estimator. Standard reference any verifier-anchored-reward LLM paper is expected to carry.

7. **Skalse et al. 2022** (NeurIPS, "Defining and Characterizing Reward Hacking", arXiv 2209.13085)
   and **Gao et al. 2023** (ICML, "Scaling Laws for Reward Model Overoptimization", arXiv 2210.10760).
   NOT CITED. Foundational basis for the intrinsic-Goodhart argument (method.tex ~815-831,
   discussion.tex ~113-160), currently asserted without citation.

8. **Brown et al. 2024** (Large Language Monkeys / repeated sampling, arXiv 2407.21787). NOT CITED.
   The B1 best-of-N FLOP-matched baseline (experimental_plan.tex ~120-144) instantiates exactly
   this; canonical repeated-sampling inference-scaling reference.

### Minor-tier omissions

9. **AlphaLLM** (Tian et al., NeurIPS 2024, arXiv 2404.12253). Option-level (coarse-action)
   search-then-distill with critics; relevant to the "coarse verbs vs token/step decoding" bet.
10. **OmegaPRM** (Luo et al. 2024, arXiv 2406.06592). Automated process supervision via MCTS;
    completes the Math-Shepherd / ReST-MCTS* trio.
11. **ReST-EM / "Beyond Human Data"** (Singh et al., TMLR 2024, arXiv 2312.06585). EM-style sibling
    of the cited STaR/V-STaR.
12. **ARTIST / ReCall / ReSearch** (2025, ARTIST arXiv 2505.01441). Tool/verifier-call as a learnable
    RL action for multi-turn math reasoning; closest prior to the "InvokeVerifier as first-class action"
    claim. The bare idea is no longer novel; the integration (verifier-call inside chance-node MCTS over
    a typed state) is. Likely concurrent / post-drafting.

## Venue / dating updates for cited entries

- **deepmind2024alphaproof**: replace the 2024 DeepMind blog post with the peer-reviewed Nature 2025
  paper (Hubert, Mehta, Sartran et al., "Olympiad-level formal mathematical reasoning with reinforcement
  learning", DOI 10.1038/s41586-025-09833-y). The "silver-medal" framing (IMO 2024) is dated; gold-medal
  IMO 2025 systems now exist.
- **guan2025rstarmath** -> ICML 2025 (oral).
- **feng2023alphazerollm** (TS-LLM) -> ICML 2024.
- **snell2024scaling** -> ICLR 2025.
- **hendrycks2021math** -> NeurIPS 2021 Datasets and Benchmarks track.
- **hosseini2024vstar** (V-STaR) -> COLM 2024.

## State of the art (compute-vs-accuracy framing)

- Small-model verifiable math: rStar-Math (7B) is the live SOTA datapoint (MATH 90.0%, AIME 53.3%,
  surpassing o1-preview). The paper's deferral of the headline benchmark claim is well-judged.
- Process supervision: OmegaPRM, Math-Shepherd, ReST-MCTS* define the automatic verifier-anchored
  credit-assignment frontier.
- Test-time compute scaling: Snell et al. 2024 (cited) is the canonical FLOP-vs-accuracy reference and
  is correctly used as the compute-axis basis.

## Positioning-statement accuracy

- "closest prior published system is rStar-Math": accurate on the MCTS axis, incomplete on the
  Q-learning axis (VerifierQ) and the structured-state axis (RAP). Scope to "MCTS-for-LLM systems."
- "AlphaProof silver-medal": out of date (July 2024). Update citation and soften phrasing.
- Pre-registration as a methodology contribution: accurate as a rare ML practice, but should be situated
  against the ML reproducibility / pre-registration literature rather than asserted.

## Scout self-critique / confidence

- High confidence that all listed works exist, are uncited, and are relevant; venue facts confirmed from
  multiple sources.
- Medium confidence that the missing-citation list is complete (the field publishes weekly; the paper's
  idiosyncratic vocabulary, verb set / context MDP / chance node, may hide a still-closer competitor).
- Adjacent fields possibly under-surveyed: classical options / hierarchical RL (the coarse-verb action set
  is essentially an options abstraction; Sutton-Precup-Singh options could be a foundational cite),
  constrained-decoding / tool-use agent literature.
