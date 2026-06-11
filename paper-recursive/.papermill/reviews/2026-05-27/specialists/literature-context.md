# Literature Context Packet (merged scouts)

**Paper under review**: "Bounded Structural Recursion in Verifier-Anchored Reinforcement Learning: When and How Deliberation Should Decompose Itself" (Towell, 2026)

**Scope**: companion paper to a framework paper on verifier-anchored approximate Q-learning over LLM-induced context MDPs. The companion develops one design dimension (bounded structural recursion via implicit parent-pointer references in the state) in depth.

## 1. Hierarchical Reinforcement Learning

The literature most directly addressed by the paper.

### 1.1 Classical foundations

- **Options framework** (Sutton, Precup, Singh, 1999, Artificial Intelligence). Temporally extended actions as triples (initiation set, internal policy, termination probability). Semi-Markov decision process formulation. Cited as `sutton1999between`. The paper correctly identifies that options are temporally extended sub-policies, whereas its seven verbs are atomic; the recursive subtree is the option analog, planned over by MCTS as policy improvement operator.
- **MAXQ** (Dietterich, 2000, JAIR). Value-function decomposition along an explicit task hierarchy: V_pi(i, s) = V_pi(a, s) + C_pi(i, s, a). Cited as `dietterich2000hierarchical`. The paper acknowledges MAXQ as an analytical lens but does not impose the decomposition architecturally.
- **Feudal RL** (Dayan and Hinton, 1993; Vezhnevets et al., 2017, ICML). Manager-worker decomposition at different timescales. Cited as `vezhnevets2017feudal`. The paper notes that its manager and worker share the same frozen LLM.
- **Option-critic** (Bacon, Harb, Precup, 2017, AAAI). Learns options end-to-end from reward. Cited as `bacon2017option`.
- **HIRO / data-efficient hierarchical RL** (Nachum et al., 2018, NeurIPS). Off-policy goal-conditioned hierarchical RL. Cited as `nachum2018data`.

### 1.2 Notable omissions from this thread

- **Hierarchical Abstract Machines (HAM)** (Parr and Russell, 1998): an early third pillar of HRL alongside options and MAXQ. Not cited; arguably acceptable in a companion paper.
- **STRAW / strategic attentive writer** (Vezhnevets et al., 2016): macro-action learning. Not cited.
- **AlphaProof and DSP (Draft-Sketch-Prove)** (Jiang et al., 2023, "Draft, Sketch, and Prove"): explicit hierarchical decomposition in theorem proving. Not cited; would directly address the formal-proof connection.
- **HRL for LLMs**: there has been a small but growing body of work using hierarchical RL with LLM substrates (e.g., Carta et al., 2024 on hierarchical RL with language). Largely absent.

## 2. Tree-structured and recursive neural encoders

The position paper argues for flat-token serialization as a substitute for tree encoders.

- **Tree-LSTM** (Tai, Socher, Manning, 2015, ACL). Tree-structured LSTM for semantic representations. Cited as `tai2015improved`.
- **Recursive neural networks** (Socher et al., 2013, EMNLP, sentiment treebank). Cited as `socher2013recursive`.
- **Graph Attention Networks (GAT)** (Velickovic et al., 2018, ICLR). Cited as `velickovic2018gat`.
- **Set Transformer** (Lee et al., 2019, ICML). Cited as `lee2019set`.

### 2.1 Notable omissions

- **Tree Transformer / TreeBERT / Transformer-XL with tree positional encodings**: there is a substantial body of work using transformers with explicit tree-aware positional encodings (Shiv and Quirk, 2019, "Novel positional encodings to enable tree-based transformers"). The paper's argument about positional anchors and depth markers would benefit from engagement with this line.
- **Code representation with structural attention**: GraphCodeBERT (Guo et al., 2020), CodeT5+ with abstract syntax trees. Directly addresses the claim about LLMs handling nested structure in code via attention.
- **Long-context transformers and RoPE/ALiBi positional encoding**: the paper's argument that 32k/128k contexts make distance-based attention dilution a remote concern would benefit from citing the actual long-context machinery (Press et al. 2022 ALiBi, Su et al. 2024 RoPE).

## 3. MCTS for LLM reasoning

The paper positions against this line.

- **Tree of Thoughts (ToT)** (Yao et al., 2023, NeurIPS). Tree search over reasoning steps. Cited as `yao2023tree`.
- **AlphaZero-like tree search for LLMs** (Feng et al., 2023). Cited as `feng2023alphazerollm`.
- **MCTS with iterative preference learning** (Xie et al., 2024). Cited as `xie2024monte`.
- **STaR** (Zelikman et al., 2022, NeurIPS). Self-Taught Reasoner. Cited as `zelikman2022star`.

### 3.1 Critical omission: rStar-Math

The framework paper's review explicitly flagged rStar-Math (Guan, Zhang, Liu et al., 2025, arXiv:2501.04519, "rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking") as a venue-blocking comparison gap. rStar-Math uses token-level MCTS for math reasoning with a process reward model and a self-evolved deep thinking loop. The companion paper **does not cite rStar-Math** at all, even though:

- rStar-Math operates in the same problem space (small-LLM math reasoning).
- rStar-Math uses MCTS with reward signal in the same way the companion paper does.
- rStar-Math is the most direct contemporary comparison to the framework paper's setup.
- The distinguishing claim of the companion paper (structural decomposition at the hypothesis level, not at the token level) is in direct contrast with rStar-Math's token-level MCTS.

The framework paper has already resolved this gap. The companion paper should at minimum cite rStar-Math in the related work section (§5.3) and explicitly contrast its approach (structured-state recursion at the hypothesis level) against rStar-Math (unstructured-state MCTS at the token level). This is the most important missing citation.

### 3.2 Other notable omissions

- **Reasoning via Planning (RAP)** (Hao et al., 2023, EMNLP): "Reasoning with Language Model is Planning with World Model". Uses MCTS over reasoning steps with an LLM as world model. A natural comparison.
- **ReST and ReST-MCTS*** (Gulcehre et al., 2023; Zhang et al., 2024): self-improvement with verifier-anchored search. Directly contemporary.
- **PROVER, Curriculum learning for theorem provers**: the formal-proof connection would benefit from a wider survey.

## 4. Formal proof systems with recursive decomposition

- **AlphaProof** (DeepMind, 2024 blog post). Cited as `deepmind2024alphaproof`.
- **LeanDojo** (Yang et al., 2023, NeurIPS Datasets and Benchmarks). Cited as `yang2023leandojo`.

### 4.1 Notable omissions

- **HyperTree Proof Search** (Lample et al., 2022, NeurIPS): explicit hypertree-structured proof search with subgoal decomposition. **This is the most directly analogous prior work** for the companion paper's central technical contribution (recursive subgoal trees with bounded depth). Its omission is a significant gap.
- **Thor / Magnushammer / Lean copilot**: contemporary tactic-search systems with structural decomposition.
- **GPT-f, PACT** (Polu and Sutskever, 2020; Han et al., 2022): formal math with LLMs.

## 5. Position-paper-style contributions in adjacent areas

The companion paper is a position-and-methodology paper without empirical results. Adjacent precedents:

- The framework paper itself, of which this is a companion, is a position-and-methodology paper.
- "Pre-registered" empirical-design papers are common in psychology and cognitive science, less so in ML; the paper's commitment to pre-registration is a strength.

## 6. Direct comparisons relevant to the paper's central claims

The paper makes four substantive claims. For each, the closest prior work:

### Claim 1: Productive vs unproductive recursion as a verifier-reach property

Original. No directly analogous prior framing. Potentially related: "process reward models" (Lightman et al., 2024, cited) provide leaf-level signal; AlphaProof's tactic search has implicit leaf signal via Lean's type checker. No prior work names the operational distinction this way.

### Claim 2: Hard-mask depth limit as structural prior

Closest precedent: **HyperTree Proof Search** (Lample et al., 2022) uses bounded-depth hypertree expansion. **rStar-Math** uses bounded MCTS depth. The companion paper's argument that hard masks (rather than soft penalties) preserve verifier-anchored reward purity is a more principled framing than either precedent provides explicitly.

### Claim 3: Identifier-based action targeting at fixed verb cardinality

This is the most original technical contribution. The closest analog is hierarchical RL's "parameterized actions" (Masson et al., 2016), but those typically use continuous parameters rather than discrete identifier dereferencing. The claim is sufficiently novel.

### Claim 4: Flat serialization as a sufficient substitute for tree encoders

The argument is plausible but under-supported. The cited tree-encoder works are 2013-2019 vintage. The paper would benefit from engaging with more recent work on positional encodings in long-context transformers, structural attention in code models, and explicit tree-positional encodings (Shiv and Quirk, 2019). The qualitative argument from "LLMs handle JSON / code / HTML" is good but should be operationalized with a citation or two.

## 7. Venue and prior-art assessment

The paper is well-positioned for a reasoning/test-time-compute workshop or a methodology track at NeurIPS/ICML/ICLR. The methodological contribution is sound; the empirical contribution is pre-registered, so the central evaluative question for a reviewer is whether the methodological development is novel and well-grounded.

Key venue-blocking concerns from a literature standpoint:
1. **Missing rStar-Math citation** (venue-blocking): direct contemporary comparison absent.
2. **Missing HyperTree Proof Search citation** (major): the most analogous prior work on bounded-depth recursive subgoal trees.
3. **Tree-encoder literature dated to pre-2019** (minor): the case for "flat serialization suffices" would be strengthened by engagement with recent work.
