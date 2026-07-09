# Handoff: state of the work as of 2026-07-09

(Replaces the stale 2026-06-18 handoff, which described the pre-redesign
outer loop.) Written for resuming this project in another agent/tool.
Read this file, then `CLAUDE.md` (rules), `design.md` (system spec,
source of truth), and `docs/superpowers/research/EXPERIMENTS.md` (the
experiment ledger, E1-E17, the single most important record). Branch:
`action-space-redesign`. Test suite: 324 passed, 6 skipped (optional
deps). Everything described below is committed.

## 1. What this project is

A verifier-anchored RL system: a tiny EXTERNAL controller network (verb
head + pointer head + Q head, a few million params, NOT YET TRAINED)
learns to orchestrate a FROZEN small LLM by issuing typed (verb, target)
actions over an immutable reasoning forest. A symbolic verifier (SymPy)
is the ONLY reward source: terminal +1/-1 at root COMMIT, plus a fixed
-0.01 housekeeping constant for masked (illegal/failed) actions; no
learned reward model, PRM, ORM, critic, or LLM-judge, ever. Explicitly
NOT o1/R1: RL never touches the LLM's weights.

Framing directive (must be restated verbatim in any doc/paper subagent
brief): RL first, MCTS second. The training spine is off-policy
value-based Q-learning with n-step or MC returns; MCTS is a strictly
additive accelerator and the test-time compute knob, never the spine.

Six verbs: SPLIT, FORM_HYPOTHESIS, COMPUTE (LLM-calling, lens-scoped);
VERIFY, COMMIT, ABANDON (no LLM call: the policy can neither author what
is checked nor what it is checked against).

Three-rung training ladder: (1) LLM-as-controller (a teacher LLM picks
actions; trajectories logged as behavior-cloning data), (2) BC
warm-start of the heads, (3) off-policy Q-learning fine-tune. Rung 1 is
done (results below); rungs 2-3 ("Phase B") are the critical path.

## 2. Hard rules (violating these has bitten us)

- NEVER write the em-dash glyph in any file: a global hook rejects the
  write. Use commas/colons/parens/hyphens.
- `git add` explicit file paths only; never `-A`/`-u`/`.` (the working
  tree carries unrelated modified slide decks and an old spec edit that
  must not be swept into commits).
- The reward invariant above is load-bearing. Aesthetic signals may
  enter only as search priors, never as Bellman targets.
- Controlled-experiment discipline: one manipulation per experiment,
  outcome measure and its reading pre-registered in the ledger BEFORE
  the run, all trajectories and raw replies logged. 20-episode probes
  cannot distinguish deltas under ~15 points; replicate before
  concluding.
- Lab-notebook vs paper: debugging arcs and bug-fix war stories stay in
  the ledger; outward artifacts (paper, deck, emails) carry only
  forward-facing claims. The user feels strongly about this.
- Paper STUB discipline: `\stub{...}` marks unfinished content; never
  present stubbed content as complete; never resurrect archived pilot
  numbers. The GSM8K pilot (E1) is QUARANTINED (see 4.2).
- Style rules for the paper: define every symbol at first use; expanded
  sums before expectation notation; hand-computable example before
  general algorithm.

## 3. Infrastructure facts

- OpenAI: key in `$OPENAI_API_KEY` (from `~/.bashrc`), project
  "metafunctor", prepaid balance topped to $10 on 07-02; roughly $2.5-3
  remains. gpt-4.1-mini is the workhorse teacher (gpt-4.1 measured to
  add nothing at ~5x the price).
- Local Ollama (this machine, CPU): `qwen2.5:1.5b-instruct` and
  `qwen2.5:3b-instruct` pulled; OpenAI-compatible endpoint at
  `http://localhost:11434/v1` (pass any non-empty dummy key env var).
  ~40 min per 20-episode probe; ~2 h per 240-problem baseline.
- GPU box at 192.168.0.204 (often in use; a 7B-class generator would
  live there when free).
- Long CPU jobs: launch with `setsid nohup ... & disown` (foreground
  shells get killed at ~10 min in the previous harness).

## 4. Results (ledger entries E1-E17; run files in experiments/runs/)

### 4.1 Direct-prompt baselines (experiments/run_direct_baseline.py)

| model | GSM8K-500 | casework-v1 (240) | f7 | f8 |
|---|---|---|---|---|
| gpt-4.1-mini | 94.4% | 100% | 98% | 100% |
| Qwen2.5-1.5B (local) | - | 39.2% | 5% | 5% |
| Qwen2.5-3B (local) | - | 68.8% | - | - |

Qwen-1.5B per family: f3 75, f1a 40, f1b 35, f2a 31, f2b 15 percent;
dead-case problems harder (33% vs 43%); ~14% unusable replies.

**The decisive comparison**: trained controller + frozen Qwen generator
must beat that generator's own direct number (39.2% for 1.5B, 68.8% for
3B) on casework-v1. That is the success criterion of the whole project.

### 4.2 Harness (LLM-as-controller, Rung 1) results

- mini as BOTH teacher and generator on casework-v1 probes: plateau
  **20-25%** (best 5/20). Teacher-side remedies exhausted, all null
  under pre-registration: prompt engineering (E6-E7), a 5x-price teacher
  (E8, E9b), completion budget 64->192 (E11), substitution-verified
  solve-claims (E13; built and tested but OFF by default,
  `--solve-claims`).
- Structural verbs fire iff the distribution demands them: 0.5% of
  decisions on GSM8K vs 12-23% on casework, SPLIT in 20/20 episodes
  (E3). This is why the casework dataset exists.
- GSM8K 500-episode run (E1, 70.8%): **QUARANTINED**. Collected under a
  superseded flat state rendering; the numbers characterize the old
  interface and the trajectories are excluded from BC (not
  re-renderable). GSM8K/MATH re-enter later as eval targets only.
- The current observation interface (part of the method, described in
  paper method.tex): tree render with parent= tags, commit-readiness
  annotation "(evidence: node N verified)", verify-failure markers,
  rejection feedback to the controller, TURN k/N budget.

### 4.3 Design-true regime (API teacher + local Qwen generator): HALTED

Four probes, all 0/20, each fixing the previous proximate cause:
- E14: strict wire schemas masked 55-63% of typed-verb output ->
  loosened schemas + apply-time coercion (committed, tested).
- E15: coercion worked (SPLIT mask 63%->8%) but 1.5B CONTENT is junk.
- E16: 3B fixes per-case content (cases solve end to end; teacher
  fallbacks 8%) but the memoryless generator re-emits its rejected
  root-aggregation statement (COMPUTE mask 82%).
- E17: duplicate-aware COMPUTE suffix (lists existing facts, "must be
  different") did NOT break the loop (83% mask); failures diversified.

Stop-rule verdict (in the ledger): blocked on GENERATOR QUALITY under
the typed protocol, not on any single interface fix. Resume only with a
7B-class generator (GPU box), and first do task #28: log raw generator
payloads + mask reasons into trajectories so the next diagnosis is one
run deep.

### 4.4 Datasets

- `datasets/casework-v1/`: 240 train / 60 test, five families, balanced,
  content-hash ids, per-record provenance (generator version, git SHA,
  sympy version, seed), dataset card (HF-style + datasheet), sha256
  checksums, CC-BY-4.0. Labels verified by construction AND independent
  re-derivation (solveset or exhaustive enumeration).
- `experiments/generate_casework.py`: 9 families total (f4, f6, f7, f8
  calibrated but held out of v1), `--scale` knob, `--exclude`
  cross-split guard. casework-v2 = reweighting existing families once
  the target generator is settled; no new generator work needed.

## 5. Code map (src/verifier_anchored_rl/)

- `reductio/`: vendored immutable forest core + pure verb algebra.
  Invariants: frozen nodes, no reward/value fields, stdlib-only core,
  structure never gates (connective is metadata), commit requires a
  verifier-vouched witness.
- `deliberation_mdp.py`: the context MDP. Guards added this campaign:
  SPLIT only on childless goals; per-goal duplicate-COMPUTE masking;
  apply-time payload coercion (object children, connective normalizing,
  premise int-coercion); `solve_claims=False` flag (E13, not adopted).
- `lenses.py`: tree render, evidence annotation, verify-failure markers,
  TURN k/N, loose wire schemas, duplicate-aware COMPUTE suffix (E17),
  solve-claim teaching when enabled.
- `llm_controller.py`: CONTROLLER_SYSTEM + CONTROLLER_SYSTEM_STRUCTURED
  (casework playbook); action-string normalization; verify-preferring
  safe fallback (never ends an episode); raw reply logged on fallback;
  rejection-feedback threading. Decision records carry: state_render,
  legal, chosen, reward, masked, fallback (+raw_response), terminal.
- `llm_client.py`: OpenAI-compatible; 429/5xx backoff; parse-retry
  echoes the failed assistant reply; api_key repr=False; token counters.
- `verifier_adapter.py`: MathVerifier decide + solve_check
  (substitution); check_answer (terminal: RHS fallback, tautology
  rejection).
- Torch components awaiting Phase B (all tested): `heads.py`,
  `embedder.py`, `planner_actor.py`, `shared_forward.py`, `train.py`,
  `replay.py`.

Runner: `experiments/run_llm_controller.py`. Flags: `--backend
{echo,http,openai}`; mixed `--controller-backend/-model/-base-url`;
`--controller-prompt {default,structured}`; `--controller-max-tokens`;
`--solve-claims {on,off}`; `--start-index` (resume, keeps true episode
ids); `--max-problems`; `--workers`; `--token-budget`; `--seed`.
Run with PYTHONPATH=src (or `pip install -e .`).

## 6. Behavior-cloning data available for Phase B

USABLE (tree-render interface, mini-as-both, casework), in
experiments/runs/: `2026-07-02-rung1-casework-probe20-v6-treeview.jsonl`,
`2026-07-03-rung1-casework-probe20-v7.jsonl`,
`2026-07-03-rung1-casework-E11-gpt41-maxtok192.jsonl`,
`2026-07-03-rung1-casework-E13-solveclaims-on.jsonl`: ~80 episodes,
~1,200 decisions, ~14 winning episodes. The design-true runs (E14-E17)
add ~1,250 decisions with zero wins (useful as negatives and
masked-action labels). QUARANTINED: everything GSM8K (flat render) and
casework probes 1-5 (pre-tree-render).

Cheap way to grow winning data: mini-as-both on the remaining 220
casework train problems (~$2, expect ~45-55 wins) with the current
interface; budget allows it.

## 7. Publication artifacts

- Paper `paper/main.tex` (19 pp, builds with latexmk): full scaffold +
  written Preliminary Findings section; TikZ figures (simple
  architecture, episode timeline, training ladder in the body; detailed
  architecture in Appendix A). A multi-agent review was applied (2
  Criticals + ~16 Majors fixed); reports in
  `.papermill/reviews/2026-07-03/`. Casework is described as "prepared
  for public release" (the repo is private; do not claim published).
- Deck `docs/slides/2026-07-02-design-overview.pdf` (11 pp): reviewed,
  every number verified against run logs, GSM8K removed, no war
  stories; sent (or ready to send) to coauthors.
- Experiment ledger: `docs/superpowers/research/EXPERIMENTS.md`.
- Specs under `docs/superpowers/specs/`: casework generator, solve-claim
  COMPUTE, deliberation-MDP integration.

## 8. Next steps, in priority order

1. **Phase B** (the critical path; pure local work, no API spend):
   a. Define the BC record schema BEFORE converting: per decision, the
      rendered state, the legal (verb,target) menu, the chosen pair;
      per-node features for the pointer head.
   b. Converter: trajectory JSONL -> training dataset (usable runs of
      section 6; wins as positives; masked/illegal teacher intents as
      negatives).
   c. Encoder rung 1 per design.md section 5 (frozen-LLM hidden states
      at marker positions), or the cheaper text-embedding rung to start.
   d. BC warm-start of verb+pointer heads; eval by action-match rate on
      held-out teacher decisions, then live casework episodes at B=0
      against the 39.2% (1.5B) target.
   e. Off-policy Q-learning fine-tune (uncorrected n-step/MC returns
      estimate Q^mu of the behavior policy; stated openly in
      training.tex).
2. Optional: the ~$2 mini-as-both collection (section 6) to roughly
   triple the winning-trajectory count before BC.
3. Design-true regime: dormant until a 7B generator is available; then
   task #28 first (payload/mask-reason logging), then ONE probe.
4. casework-v2 mixing and GSM8K/MATH evals: after Phase B shows life.

## 9. Common commands

```bash
pytest -q                                   # 324 passed, 6 skipped
PYTHONPATH=src python3 experiments/run_llm_controller.py --backend echo \
  --problems datasets/casework-v1/train.jsonl --max-problems 3 --out /tmp/smoke.jsonl
PYTHONPATH=src python3 experiments/run_direct_baseline.py \
  --model qwen2.5:3b-instruct --base-url http://localhost:11434/v1 \
  --api-key-env DUMMY --problems datasets/casework-v1/train.jsonl \
  --out /tmp/base.jsonl                     # DUMMY = any set env var
python3 experiments/generate_casework.py --n 40 --seed 0 --families f7 --out /tmp/f7.jsonl
cd paper && latexmk -pdf main.tex           # 19 pp
cd docs/slides && latexmk -pdf 2026-07-02-design-overview.tex
```
