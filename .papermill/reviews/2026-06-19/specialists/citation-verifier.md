# Citation Verification Report

## Summary
All 29 cited papers are real; author lists and years correct; no phantom citations. Issues: four
venue-update issues, one blog-to-Nature upgrade, one overstated bib header comment, one claim-citation
mismatch, formatting inconsistencies, and eight missing references (two critical). Citations checked: 29.
Issues: 1 critical, 8 major (incl. missing refs), several minor.

## Critical
- **C-1. deepmind2024alphaproof**: cited as a 2024 DeepMind BLOG POST. Peer-reviewed paper now exists:
  Hubert, Mehta, Sartran et al., "Olympiad-level formal mathematical reasoning with reinforcement learning,"
  Nature 2025, DOI 10.1038/s41586-025-09833-y. Also the "silver-medal" (IMO 2024) framing is dated (gold-medal
  IMO 2025 systems exist). Fix: replace the @misc entry with the Nature article; update the two in-text uses.

## Major (existing citations)
- **C-2. guan2025rstarmath** -> ICML 2025 (oral); change to @inproceedings.
- **C-3. feng2023alphazerollm** (TS-LLM) -> ICML 2024; change to @inproceedings.
- **C-4. snell2024scaling** -> ICLR 2025; change to @inproceedings.
- **C-5. hendrycks2021math** -> NeurIPS 2021 Datasets and Benchmarks track; change to @inproceedings.
- **C-6. hosseini2024vstar** (V-STaR) -> COLM 2024; change to @inproceedings.
- **C-7. Claim-citation mismatch**: introduction.tex cites lightman2024verify for "not subject to reward
  hacking by a learned critic." Lightman et al. compare PRM vs ORM on MATH accuracy; they do NOT establish
  immunity to reward hacking. Correct citations: Skalse et al. 2022 + Gao et al. 2023. Fix the citation.

## Major (missing references)
- **C-8 (Critical-tier). VerifierQ** (Qi, Tang, Zhu, arXiv 2410.08048; AI-for-Math Workshop @ ICML 2025).
  Closest prior on the Q-learning-over-LLM-MDP-for-test-time-compute axis (IQL/CQL, modified Bellman). Must
  appear in sec:rw-verifiers with explicit comparison.
- **C-9 (Critical-tier). Stochastic MuZero** (Antonoglou et al., ICLR 2022 spotlight). Owns the chance-node/
  afterstate mechanism the paper claims as a structural contribution (sec:method-mcts). Must cite + distinguish.
- **C-10. Tree of Thoughts** (Yao et al., NeurIPS 2023). Canonical deliberation-via-search; conspicuous omission
  in sec:rw-mcts-llm.
- **C-11. RAP / Reasoning via Planning** (Hao et al., EMNLP 2023). LLM-as-world-model MCTS-over-state; ancestor
  of the context-MDP framing.
- **C-12. ReST-MCTS\*** (Zhang et al., NeurIPS 2024). Per-step value-target distillation from search; direct
  competitor for the per-step-distillation claim.
- **C-13. Math-Shepherd** (Wang et al., ACL 2024, arXiv 2312.08935). Canonical automatic step-level process
  reward; belongs in sec:rw-verifiers.
- **C-14. Skalse et al. 2022** (NeurIPS, arXiv 2209.13085) and **Gao et al. 2023** (ICML). Foundational basis
  for the Goodhart argument; also the correct citation for C-7.
- **C-15. Brown et al. 2024** (Large Language Monkeys, arXiv 2407.21787). Canonical repeated-sampling / best-of-N
  reference; the pre-registered B1 baseline instantiates it.

## Minor
- AlphaLLM (Tian et al. 2024, arXiv 2404.12253; note the brief's id 2405.09118 was wrong), OmegaPRM (Luo 2024,
  arXiv 2406.06592), ReST-EM/"Beyond Human Data" (Singh et al., TMLR 2024, arXiv 2312.06585), ARTIST (2025,
  arXiv 2505.01441): nice-to-have additions for completeness.
- wei2022cot: @article with journal=NeurIPS; should be @inproceedings/booktitle.
- cobbe2021gsm8k, willard2023outlines, xie2024monte, guo2025deepseekr1: arXiv-only; @article+arXiv-journal is
  conventional; recheck before camera-ready.
- Bib top-comment "All entries verified for submission with urldate fields where applicable" is overstated
  (five venue upgrades available; most entries have no urldate). Revise the comment.
- No DOI fields for Nature/Science entries (silver2016/2017/2018, schrittwieser2020, mnih2015, li2022); add DOIs.

## Verified accurate (spot-checked)
sutton2018reinforcement, kocsis2006bandit (and the rem:mcts-consistency hedging is accurate: UCT does not directly
apply to PUCT-with-priors, and the paper says so), silver2016/2017/2018alpha*, schrittwieser2020muzero,
anthony2017thinking, lightman2024verify (ICLR 2024; venue correct, claim-match issue separate), mnih2015human,
hu2022lora, zelikman2022star, openai2024o1, browne2012survey, coulom2006efficient, yang2024qwen25, dubey2024llama3,
allal2024smollm2.
