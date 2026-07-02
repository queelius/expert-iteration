# Related-work research memo: four links + prior-art novelty assessment

Date: 2026-07-02. Inputs: the four links the user supplied, plus a targeted
prior-art search against the claimed contribution of `new-design.md`.
Produced by two research agents (link analysis; novelty assessment), verified
against abstracts. This memo feeds the paper's related-work section.

## Part 1: the four links, and how each bears on the design

### 1. OREO: Offline RL for LLM Multi-Step Reasoning (arXiv:2412.16145)

Wang et al.; ACL 2025 Findings. Jointly trains a policy (the LLM itself,
token-level) and a separate value network via a soft Bellman objective,
offline, with terminal-only rule-based reward (GSM8K/MATH/ALFWorld). Beats
SFT and DPO baselines (e.g. Qwen-1.5B GSM8K 77.3 vs 73.5 SFT); the learned
value also guides step-level beam search at inference (+17.9 percent relative
on MATH-500 at beam width 7).

Salience: the closest technical analogue among the four. It is empirical
evidence for exactly the two bets our Section 7 makes: value-based/Bellman
training beats policy-gradient/preference methods under sparse terminal
reward, and a learned value usefully drives test-time search. But it sits on
the wrong side of the Section 1.2 contrast: OREO's policy IS the LLM's
weights, token-level, flat token state; no external controller, no structured
forest, no typed actions. Simultaneously our best "same math" ally and the
cleanest "different target" foil.

Use: cite in Section 7.1 and Appendix B (value-based beats DPO/PG under
sparse terminal reward); Section 7.4/8 (value-guided search as compute knob);
Section 1.2 (nearest value-based sibling that still trains the LLM).

### 2. GSM8K-RLVR repo (github.com/Mohammadjafari80/GSM8K-RLVR)

GRPO + LoRA fine-tuning of Qwen2.5-Math-1.5B and Qwen2.5-0.5B on GSM8K with
rule-based reward (final-answer correctness + a #### format check).
Math-1.5B 70.7 to 77.3; 0.5B 32.6 to 46.1.

Salience: a small, reproducible instance of exactly the token-level RLVR
baseline our evaluation grid (Section 9, rows B0/B1/B5) is read against. Its
unguarded format-adherence bonus is a useful negative example against Section
6's gated, capped shaping discipline.

Use: Section 9 reference point for token-level RLVR-on-GSM8K; not method
support.

### 3. Wolfe, "Online RL" (substack)

Survey of online vs offline RL for LLM alignment: PPO, GRPO/REINFORCE, DPO
and descendants; names RLVR as reward from rules-based verifiers. Everything
surveyed is token-level RL on the LLM's own weights.

Salience: supplies the standard on-policy/off-policy vocabulary Appendix B
uses; sharpens Section 7.1's "off-policy is load-bearing" claim by contrast
(PPO/GRPO need live on-policy samples; our Q-learning buffer can mix random,
teacher, and MCTS experience with no change to the update). By omission it
confirms the mainstream backdrop Section 1.2 differentiates against.

Use: background only (terminology, landscape framing).

### 4. Wolfe, "Demystifying Reasoning Models" (substack)

Survey of o1/o3/R1-style reasoning models. Defines RLVR (answer matching,
test execution, format checks). Documents that DeepSeek-R1-Zero rejected
neural reward models because they "may suffer from reward hacking in
large-scale RL."

Salience: secondary corroboration for Section 6's central citation (R1
rejects learned PRMs on reward-hacking grounds), nearly verbatim to our
phrasing. Its RLVR description IS our table's o1/R1 column. One nuance worth
making explicit if cited: the post notes RLVR usually needs human-provided
ground-truth answer keys, whereas our SymPy verifier computes ground truth
symbolically, which slightly sharpens our positioning.

Use: Section 1.2 (reader-facing RLVR/o1/R1 explainer); Section 6
(reward-hacking corroboration).

### Cross-cutting synthesis

All four confirm the design sits inside the RLVR moment (verifier reward, no
learned reward model, math as proving ground) while confirming that the
surveyed literature applies RL to the LLM's own weights at token granularity
over a flat trace, which is precisely what Section 1.2's contrast table
separates us from. OREO is the pivot: strongest evidence for the value-based
bet, made while training the generator itself. None of the four has an
external controller, a typed (verb, target) action space, or a structured
non-token state.

## Part 2: prior-art novelty assessment

Claim under test: a tiny (under 5M param) EXTERNAL controller (verb head +
pointer head over live node ids + Q head), trained by off-policy value-based
Q-learning, orchestrating a FROZEN small LLM via typed (verb, target) lens
actions over a nested reasoning forest, with a symbolic verifier as the ONLY
reward; MCTS optional and additive.

### Closest prior work (ranked by threat)

1. **TS-LLM** (Feng et al. 2023, arXiv:2309.17179). AlphaZero-like: PUCT MCTS
   + learned value + learned outcome reward model (ORM) + policy distillation
   into the LLM. Highest risk: a reviewer can frame us as "TS-LLM with the ORM
   swapped for SymPy" unless the external-tiny-controller + pointer-action +
   frozen-generator conjunction is foregrounded.
2. **GSoS** (Moon et al. 2024, arXiv:2410.02992). Operation-level MDP over
   tree-search operations, distilled INTO the generating model via SFT. Shares
   the operation-level action idea; no external controller, no Q-learning, no
   verifier-only reward.
3. **ArCHer** (Zhou et al., ICML 2024, arXiv:2402.19446). Hierarchical
   off-policy value-based RL for LLM agents; trains the LLM's token policy.
   Dangerous as the "off-policy value-based RL for LLMs" precedent a reviewer
   combines with GSoS.
4. **Agent Q** (Putta et al. 2024, arXiv:2408.07199). MCTS + self-critique
   intermediate reward + off-policy DPO; trains the LLM; uses exactly the
   LLM-judge reward we forbid.
5. **OREO** (arXiv:2412.16145). See Part 1: value-based, sparse reward, math;
   trains the LLM's own policy.
6. **RAP** (Hao et al., EMNLP 2023, arXiv:2305.14992). LLM as world model +
   MCTS; pure prompting, no training.
7. **LATS** (Zhou et al., ICML 2024, arXiv:2310.04406). MCTS with LLM-as-value
   via prompting; no trained network.
8. **rStar-Math** (Guan et al. 2025, arXiv:2501.04519). Small LM + MCTS +
   learned process preference model (PPM); trains the SLM.
9. **ReST-MCTS*** (Zhang et al., NeurIPS 2024, arXiv:2406.03816). Process
   rewards inferred from outcomes train a PRM + self-training; learned PRM is
   central (we forbid it).
10. **RLPrompt** (Deng et al., EMNLP 2022, arXiv:2205.12548). Direct ancestor
    of "small external policy + frozen LM + value-flavored RL", but action =
    prompt tokens, single turn, no structured state.
11. **TEMPERA** (Zhang et al., ICLR 2023, arXiv:2211.11890). RL prompt
    editing over a frozen LLM with discrete typed edit actions; single-turn
    classification, no reasoning forest, no Q object.
12. **PNLC / Planning without Search** (2025, arXiv:2505.18098). Offline
    goal-conditioned value critic guiding a frozen LLM; value only, no
    external action policy, learned (not symbolic) value.
13. **Active Context Curation via RL** (Li et al. 2026, arXiv:2604.11462).
    Lightweight RL policy curates a frozen executor's context. Closest on the
    lens axis and near-simultaneous; curator is 7B, action = pruning, not
    typed reasoning operations. Must be cited as concurrent work.
14. **UniR / Universal Reasoner** (Kim et al., ICML 2026, arXiv:2505.19075).
    External module + frozen LLM via logit addition, verifiable-reward
    supervised guidance; token/logit level, no operation MDP, no RL.

Supporting/mechanism precedents to cite briefly: Searchformer
(arXiv:2402.14083, internalize-search precedent for B=0), V-STaR
(arXiv:2402.06457, learned verifier as contrast), SWEET-RL
(arXiv:2503.15478), ROVER (arXiv:2509.24981, value-based RLVR training the
LLM), DSPy (arXiv:2310.03714, optimize-orchestration lineage), Pointer
Networks (Vinyals et al. 2015), Options framework (Sutton/Precup/Singh 1999),
RL for branch-and-bound node selection (arXiv:2310.00112).

### Verdict

**Novelty risk.** No single paper anticipates the full conjunction. Three
real threats: (a) TS-LLM read as "same loop, different reward source"; (b)
the obvious ArCHer + GSoS + RLVR combination argument ("each ingredient is
published, the combination is engineering"); (c) Active Context Curation as
concurrent work on the lens axis.

**What survives as novel: the conjunction, claimed as such.**
- A tiny external controller that is neither the LLM nor an LLM-sized critic
  (vs ArCHer/OREO/Agent Q/rStar training the LLM; PNLC's critic; the 7B
  curator; UniR's logit module).
- The factored verb head + pointer head over live node identifiers of a
  growing reasoning forest (no precedent in LLM-reasoning RL).
- Off-policy value-based Q-learning as the spine with MCTS strictly
  optional/additive (LLM-reasoning RL is otherwise policy-gradient/DPO or
  SFT distillation; the value-based exceptions train the LLM).
- Symbolic verifier as the sole reward, no learned value/PRM/ORM/critic
  anywhere in the reward path (vs TS-LLM, ReST-MCTS*, rStar-Math, V-STaR,
  PNLC, SWEET-RL).
- The lens property: one action selects the operation AND scopes the frozen
  generator's context, with local write-back.

Honest caveat: the conjunction is defensible but thin per-element; a reviewer
asking "which single mechanism is new?" finds a precedent for each. Frame the
contribution as the architecture-conjunction plus the small-controller
compute-frontier claim, not as a new RL algorithm or a new action idea.

**Positioning one-liners (must-cite differentiations).**
- TS-LLM: they fine-tune the LLM against a learned ORM with MCTS as the
  spine; our controller is an external under-5M network against a symbolic
  verifier with the LLM frozen and MCTS optional.
- GSoS: shares the operation-level MDP but distills into the generator via
  SFT; we train an external value-based controller and never touch the
  generator.
- ArCHer: establishes off-policy value-based multi-turn RL for LLM agents but
  optimizes the token policy; we keep the LLM frozen.
- OREO: offline value-based RL training the LLM's policy and a token-level
  value; our value scores typed (verb, target) operations with a frozen
  generator.
- Agent Q: MCTS + DPO + self-critique reward; we forbid LLM-judge reward.
- RLPrompt/TEMPERA: small external RL policy over a frozen LM, but
  single-turn prompt tokens/edits; ours is multi-step typed operations with a
  pointer head over an evolving forest.
- PNLC: learned value critic only; we add a discrete external action policy
  and anchor value on a symbolic verifier.
- Active Context Curation (concurrent): dedicated 7B context-pruning policy;
  our lens is a byproduct of typed operations chosen by an under-5M
  verifier-anchored Q controller.
- ReST-MCTS*/rStar-Math: learned process reward/preference models; we take
  the harder line that only a symbolic verifier produces reward.
- UniR: token-logit-level external module with supervised guidance; ours is
  operation-level off-policy Q-learning.
