# Citation Verifier Report

**Paper**: "Bounded Structural Recursion in Verifier-Anchored Reinforcement Learning"
**Date**: 2026-05-27
**Focus**: bibliography entries (especially the hierarchical-RL and tree-encoder additions), TODO-marked entries, uncited entries, missing critical citations.

## Summary

The bibliography is 36 entries. 28 are cited; 8 are uncited (a non-trivial fraction). Seven entries are marked `TODO` in the bib file, indicating the author needs to verify citation metadata before submission. The most consequential gap is the absence of rStar-Math (Guan et al., 2025), which the framework paper's review identified as a venue-blocking comparison and which the companion paper does not address.

## Uncited entries (8)

Diff between cited keys and defined keys shows these `references.bib` entries are not cited anywhere in the manuscript:

| Key | Entry | Status |
|-----|-------|--------|
| browne2012survey | MCTS survey (Browne et al., 2012) | Standard reference; should be cited in background §3.1 alongside Kocsis/Coulom |
| guo2025deepseekr1 | DeepSeek-R1 | Should be cited in §5.5 self-improvement, or removed |
| hu2022lora | LoRA | Mentioned in bib comments as "framework-paper subset" but the companion does not exercise adapter mechanics; can be removed |
| openai2024o1 | OpenAI o1 system card | Should be cited in §1 or §5.5 as test-time-compute scaling exemplar; or removed |
| silver2017alphagozero | AlphaGo Zero | Cited in background.tex line 35 via `\citep{silver2017alphagozero,silver2018alphazero}` - **wait, let me re-check** |
| silver2018alphazero | AlphaZero | Same |
| snell2024scaling | Test-time compute scaling | Should be cited in §1 or §discussion; or removed |
| wei2022cot | Chain-of-thought prompting | Standard reference; should be cited in §1 motivation or removed |

**Re-checking**: My diff said silver2017alphagozero and silver2018alphazero were uncited but background.tex line 33-35 has `\citep{anthony2017thinking, silver2017alphagozero, silver2018alphazero}`. Let me verify.

Looking again at background.tex lines 33-35:
> "MCTS appears in that system as an optional online policy improvement operator over $\Qpsi$, in the expert-iteration sense \citep{anthony2017thinking, silver2017alphagozero, silver2018alphazero}"

So silver2017alphagozero and silver2018alphazero ARE cited; my diff was bad because of how `\citep{a, b, c}` flattens. Re-running the diff with proper multi-key parsing:

The actually-uncited entries are: browne2012survey, guo2025deepseekr1, hu2022lora, openai2024o1, snell2024scaling, wei2022cot. Six uncited entries.

**Finding (Minor)**: Six uncited bibliography entries inflate the bibliography without supporting argument. Some are inherited from the framework paper. Recommendation: either cite them (cot at intro motivation; o1 and snell as test-time-compute exemplars; browne in background MCTS recap; deepseekr1 in self-improvement related work) or remove them.

**Severity**: Minor. Does not affect correctness; affects polish.

## TODO-marked entries (7)

The `references.bib` file has explicit TODO comments on these entries:

1. **bacon2017option** (lines 113-114): "Verify exact venue and page numbers for the Bacon et al. option-critic paper. AAAI 2017 is the commonly cited version."
   - Verified: Bacon, Harb, Precup, "The Option-Critic Architecture", AAAI 2017, pages 1726-1734. The bib entry's metadata matches. **TODO can be removed.**

2. **velickovic2018gat** (lines 150-151): "confirm exact citation; the Graph Attention Networks paper is widely cited as Velickovic et al. 2018 ICLR."
   - Verified: ICLR 2018. Authors Velickovic, Cucurull, Casanova, Romero, Lio, Bengio. Bib entry matches. **TODO can be removed.**

3. **deepmind2024alphaproof** (lines 202-204): "AlphaProof has been announced (IMO 2024 silver medal) but the formal paper is forthcoming. Use the blog/announcement reference until a formal citation is available."
   - As of 2026-05, AlphaProof has appeared in venue publications but the citation remains a blog post in many papers. If the author has access to a more formal citation (perhaps Nature or a workshop paper), update; otherwise the blog citation is acceptable.

4. **yang2023leandojo** (lines 213-214): "LeanDojo, NeurIPS 2023 Datasets and Benchmarks; verify citation."
   - Verified: NeurIPS 2023 Datasets and Benchmarks Track. The bib entry matches. **TODO can be removed.**

5. **openai2024o1** (lines 232-233): "OpenAI o1 system card; verify exact arXiv/report citation."
   - The o1 system card is a technical report from OpenAI. The bib entry has the official URL. **TODO can be removed**, though as noted, the entry is currently uncited.

6. **yang2024qwen25**, **dubey2024llama3**, **allal2024smollm2** (lines 277-298): "verify exact citation for each base model release."
   - Qwen2.5 technical report: arXiv:2412.15115. Matches.
   - Llama 3 herd: arXiv:2407.21783. Matches.
   - SmolLM2: Hugging Face model card; the bib entry references the model card URL. Acceptable.
   - **All three TODOs can be removed.**

## Missing citations identified by literature scout

### Venue-blocking gap: rStar-Math

**Guan, Zhang, Liu et al., 2025**, "rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking", arXiv:2501.04519.

Not in the bibliography. Must be added. This is the most directly comparable contemporary system in the same problem space (small-LLM math reasoning, MCTS-driven, verifier-anchored).

Recommended bib entry:

```bibtex
@article{guan2025rstarmath,
  title={{rStar-Math}: Small {LLMs} Can Master Math Reasoning with Self-Evolved Deep Thinking},
  author={Guan, Xinyu and Zhang, Li Lyna and Liu, Yifei and Shang, Ning and Sun, Youran and Zhu, Yi and Yang, Fan and Yang, Mao},
  journal={arXiv preprint arXiv:2501.04519},
  year={2025}
}
```

Should be cited in:
- §5.3 (MCTS for LLM Reasoning) as the most direct contemporary comparison.
- Possibly also §5.2 (Tree Encoders) as an example of unstructured-state-with-MCTS (the contrast the companion paper draws).

### Major gap: HyperTree Proof Search

**Lample et al., 2022**, "HyperTree Proof Search for Neural Theorem Proving", NeurIPS 2022.

Not in the bibliography. The closest analog for the bounded-depth recursive subgoal tree mechanism. Should be cited in §5.4 (Formal Proof Systems).

Recommended bib entry:

```bibtex
@inproceedings{lample2022hypertree,
  title={{HyperTree} Proof Search for Neural Theorem Proving},
  author={Lample, Guillaume and Lacroix, Timoth{\'e}e and Lachaux, Marie-Anne and Rodriguez, Aurelien and Hayat, Amaury and Lavril, Thibaut and Ebner, Gabriel and Martinet, Xavier},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```

### Minor gaps (would strengthen specific arguments)

1. **Shiv and Quirk, 2019**, "Novel positional encodings to enable tree-based transformers", NeurIPS 2019. Would support the serialization argument in §4.5.

2. **Hao et al., 2023**, "Reasoning with Language Model is Planning with World Model" (RAP), EMNLP 2023. Would belong in §5.3 alongside other MCTS-for-LLM work.

3. **Su et al., 2024**, "RoFormer: Enhanced Transformer with Rotary Position Embedding" (RoPE original; published 2021 arXiv, 2024 in Neurocomputing). Would support the long-context-attention argument in §4.5.

4. **Press et al., 2022**, "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (ALiBi). Alternative to RoPE for the long-context argument.

## Bibliography sanity check

- 36 entries total.
- 30 currently cited (after correcting the diff).
- 6 uncited (browne2012survey, guo2025deepseekr1, hu2022lora, openai2024o1, snell2024scaling, wei2022cot).
- 7 TODO-marked (most can be cleared after metadata verification).

## Cross-paper consistency

The brief notes that the bibliography inherits from the framework paper. Some inherited entries (hu2022lora) appear inapplicable to the companion's content scope. Recommend pruning. Other inherited entries (cot, snell) would be useful if cited.

## Findings summary

| ID | Issue | Severity | Venue-blocking? |
|----|-------|----------|-----------------|
| C1 | Missing rStar-Math citation (Guan et al., 2025) | Major | Yes |
| C2 | Missing HyperTree Proof Search citation (Lample et al., 2022) | Major | Yes |
| C3 | 6 uncited bibliography entries (browne, guo, hu, openai, snell, wei) | Minor | No |
| C4 | 7 TODO-marked entries need metadata verification (most can be cleared) | Minor | No |
| C5 | Missing supporting citations for serialization argument (Shiv 2019, RoPE/ALiBi, GraphCodeBERT) | Minor | No |
| C6 | Missing RAP (Hao 2023) in MCTS-for-LLM discussion | Minor | No |

The dominant findings are C1 and C2. The other findings are polish issues that should be addressed before submission but are not venue-blocking on their own.
