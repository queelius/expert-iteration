# Correctness Audit: verifier-anchored-RL

Date (UTC): 2026-06-23T05:49:08Z
Scope: full-codebase correctness audit of the verifier-anchored-RL pipeline (branch feature/vllm-fast-pipeline).
Method: per-dimension adversarial review. A finding is CONFIRMED only when at least 2 of 3 independent verifiers agreed it is a real bug, with a code-grounded trace.

---

## 1. Executive summary

Confirmed findings by severity:

- Critical: 1
- High: 3
- Medium: 3
- Low: 6
- Total confirmed: 13

The single most important thing to fix before running more experiments:

The verifier intermediate reward can be farmed with content-free tautologies (`src/verifier_anchored_rl/llm_context_mdp.py:1208-1250`). An agent earns +0.05 and a `source="verified"` fact on every turn by invoking the verifier on something like `input_expression="1"`, `expected="1"`, because the checked proposition is never tied to the target hypothesis or the problem. This is the only confirmed defect that biases the core training signal: MCTS backs up a "spam the verifier" branch as high value, so any sample-efficiency or accuracy gain attributed to verifier-anchored intermediate shaping is confounded, and the paper's explicit Goodhart-immunity claim (method.tex 4.6) is false as implemented. Fix this before collecting any new training data.

Important context on the other high-severity items: none of them have corrupted a previously reported run (the recorded runs used the transformers backend and the SharedForwardEncoder true-token path), but each silently invalidates a headline number the moment the default/vLLM/text-embedder path is exercised. They must be fixed before the next batch since the next batch is expected to use exactly those default paths.

---

## 2. Confirmed findings (severity-ranked)

### Critical / High

#### [CRITICAL] Verifier intermediate reward farmable with tautologies unrelated to the target
- File: `src/verifier_anchored_rl/llm_context_mdp.py:1208-1250` (also `_promote_to_fact` at 1431-1449)
- Bug: `_handle_invoke_verifier` reads `input_expression` and `expected` straight from the LLM continuation and grants `_INTERMEDIATE_VERIFIER_REWARD` (+0.05) plus a `source="verified"` fact whenever `result.success and expected is not None`. The only gate is `if target_id not in state.all_ids(): return self._masked_action(state)`. Neither the handler nor `_promote_to_fact` ever compares the checked expression to `state.hypothesis_by_id(target_id).statement` or to the problem. `run_verifier("check_arith","1","1")` succeeds (simplify(1-1)==0), as do "x"/"x" and "0"/"0". There is no dedup on target_id and no cumulative cap on verifier reward (the depth mask never touches INVOKE_VERIFIER), so reward is farmable up to max_turns. Reproduced end-to-end: five successive calls each returned reward=0.05 and added a verified fact while the hypothesis statement ("whatever"/"the answer is 999") was never inspected.
- Why it matters: corrupts the central training signal. With gamma=0.99 these +0.05 signals barely discount, so MCTS backs up a progress-decoupled "verifier-spam" branch as high value, biasing the learned Q/policy toward reward hacking. A tautology-promoted `source="verified"` fact also satisfies `has_answer_bearing_claim`, unlocking COMMIT_ANSWER. This directly contradicts the paper's claim (method.tex 4.6 / lines 751-779) that the verifier-anchored signal "closes the intrinsic Goodhart gap by construction." The checked proposition is attacker-chosen, so the construction guarantee is violated.
- Severity nuance: all three verifiers rated this high (not critical) on the grounds that terminal rewards are +/-1.0 and per-episode farming is bounded by max_turns (~0.35 with default 8 turns), so it biases rather than dominates. It is ranked critical here because it is the one confirmed defect that attacks the core reward source the entire project rests on and falsifies a specific paper claim.
- Suggested fix: tie the rewarded/promoted claim to the target, require `input_expression`/`expected` to be derived from or string-match the target hypothesis/fact statement before granting reward and promoting; and/or reward only the FIRST successful verification of a given target_id (dedup on target_id in `verifier_log`); and/or reject syntactically trivial tautologies (lhs == rhs).

#### [HIGH] vLLM/OpenAI generation backend exposes no FLOP counters, so gen_calls/gen_tokens silently become 0
- File: `src/verifier_anchored_rl/llm_client.py:215-334`; consumed at `src/verifier_anchored_rl/outer_loop.py:98-99, 205-208, 284-288`
- Bug: `OpenAICompatibleLLMClient` (selected by `--gen-backend vllm`, the default and the whole purpose of this branch) defines none of `generate_calls`, `generated_tokens`, `gen_calls`, `gen_tokens`. The consumers read these via `getattr(gen, "generate_calls", getattr(gen, "gen_calls", 0))`, which falls through to literal 0. `complete_json` discards `payload["usage"]` (completion_tokens is available but ignored). By contrast `TransformersLLMClient` increments the counters (transformers_client.py:182-183), which is why existing transformers-backend metrics show non-zero gen_calls.
- Why it matters: compute/FLOP accounting is a reported axis (IterationStats gen_calls/gen_tokens, per-decision FLOP logs, design.md compute-accounting metric). On the branch's headline backend every generation FLOP column becomes 0 with no warning, so any compute-vs-accuracy figure derived from a vLLM run reports prefill-only as the total. No recorded run is corrupted yet (all used the transformers backend), but the next vLLM run is.
- Suggested fix: add `generate_calls=0`/`generated_tokens=0` to the client and increment them in `complete_json` (parse `usage.completion_tokens`, +=1 per call), under a `threading.Lock` since `complete_json` runs from many concurrent workers. Alternatively, have `_flop_counters` warn/raise when `gen_source` exposes none of the four counters, so a silent zero is impossible.
- Note: this defect was independently surfaced in both the metrics_and_signal and concurrency_and_determinism dimensions (same root cause). The two confirmed entries are consolidated here.

#### [HIGH] Fresh runs are not reproducible from --seed: torch global RNG (actor weight init) is never seeded
- File: `experiments/run_outer_loop.py:120-215` (main; `_build_stack` at 165-166, numpy seed at 167)
- Bug: `main()` builds the stack (which constructs `HiddenStateActor` with nn.Linear/nn.LayerNorm default init drawing from torch's global RNG) BEFORE `rng = np.random.default_rng(args.seed)`, and never calls `torch.manual_seed`. The only `torch.manual_seed` calls in the package are per-generation sampling seeds (transformers_client.py:166, shared_forward.py:156), not actor construction. The resume test (tests/test_outer_loop.py:315,323,345) explicitly calls `torch.manual_seed(0)` before each build precisely so weights match, acknowledging init is otherwise non-deterministic; the production harness omits this. Sibling scripts (train_heads.py:284, run_train_demo.py:63) already do it correctly.
- Why it matters: metadata.json records seed and args, implying reproducibility, but two fresh same-seed runs draw different initial weights, hence different distillation dynamics, loss curves, sharpening metrics, and holdout success rates. In `run_budget_sweep` the same `--seed` is passed to every budget, so each budget gets a different random init, confounding the cross-budget comparison the sweep exists to produce. Resume is unaffected (actor_state is restored verbatim), so this is a fresh-run / cross-run defect only.
- Suggested fix: in `main()`, before `_build_stack`, call `torch.manual_seed(args.seed)` (and `torch.cuda.manual_seed_all(args.seed)` if CUDA), and record the torch seed in metadata. A single `seed_everything` helper covering numpy/python/torch is preferable.

#### [HIGH] evaluate_holdout.py hardcodes the legacy hidden_state encoder, inconsistent with the default --embedder text training path
- File: `experiments/evaluate_holdout.py:66-77` (`_build_eval_stack`), 111-113 (`load_state_dict`)
- Bug: `_build_eval_stack` always builds `SharedForwardEncoder.from_llm_client(...)` and sizes the actor with `embed_dim=encoder.embed_dim`, with no `--embedder`/`--gen-backend` selector and no read of metadata.json. Training defaults to `--embedder text` with `--embedder-model Qwen2.5-0.5B-Instruct` (hidden_size 896), so the saved actor has embed_dim 896. The eval default `--model` is Qwen2.5-Math-1.5B-Instruct (hidden_size 1536), so `encoder.embed_dim=1536`. `actor.load_state_dict` (strict) raises a shape mismatch on every default-trained run. Even if a user forces dims to match, the eval feeds generation-model h_T into heads trained on TextEmbedder vectors, an out-of-distribution embedding distribution yielding meaningless success rates.
- Why it matters: the holdout success-rate table is the experiment's reported in-loop generalization metric (B=0 and B>0). For any text-embedder run it either crashes loudly on load (common default case) or, if dims coincide, silently produces invalid success rates.
- Suggested fix: have `evaluate_holdout.py` read `run_dir/metadata.json` args (embedder, embedder_model, gen_backend, model) and rebuild the SAME stack used in training (TextEmbedder via the embedding client when embedder==text), so embed_dim and the embedding distribution match the checkpoint. Fail fast with a clear message if the requested model/embedder disagrees with metadata.

### Medium / Low

#### [MEDIUM] Concentration normalizer k counts EXPANDED children, not available verbs
- File: `src/verifier_anchored_rl/outer_loop.py:167-189`
- Bug: `k = len(actions) = len(pi.keys()) = len(root.children)`, i.e. the number of MCTS-expanded actions, not the number of available verbs (`len(avail)`). Because the root forces one expansion per simulation (mcts.py:443,479), `len(root.children) = min(num_simulations, len(avail))`. When `num_simulations <= len(avail)`, unexpanded verbs (zero visits, the strongest concentration signal) are dropped from both the entropy sum and from `k=log2(k)`. Worked check: 7 verbs, B=4 yields code concentration 0.0 vs intended 0.288. The same true sharpness gives different concentration at different B, contaminating budget-vs-sharpening comparisons.
- Why it matters: latent. Every reported run uses B>=8 with at most 7 verbs, so the root is fully expanded and `k=len(avail)` in practice; no reported number is affected. Triggers only for a future B<7 cell or an added 8th verb (the B=8 margin over 7 verbs is exactly one). It is a diagnostic metric, not a training target.
- Suggested fix: compute `k = len(avail)` (already available at line 148) and entropy over the full available support (unexpanded verbs as probability 0, already skipped by `_entropy` but must be counted in `log2(len(avail))`). Request the prior over the full available set. Keep the k>1 exclusion.

#### [LOW] Concurrent collection docstring overclaims determinism for record values under real vLLM
- File: `src/verifier_anchored_rl/outer_loop.py:233-238` (claim); honest caveat at `src/verifier_anchored_rl/llm_context_mdp.py:1063-1073`
- Bug: the docstring asserts the result is "deterministic regardless of max_workers." True for record count/order and per-worker RNG streams, but NOT for record values/counts under a real vLLM server: vLLM's sampled tokens depend on batch composition even at a fixed per-request seed, and max_workers changes which requests are co-batched, which can change continuations, next states, trajectory length (n_decisions), and recorded embeddings/targets.
- Why it matters: documentation-level. Does not corrupt any single run (each run is internally consistent), and the claim holds under the deterministic mock used in tests. A reader sweeping `--collect-workers` could mistake batch-composition noise for a real effect.
- Suggested fix: narrow the docstring to "insertion order and per-worker RNG streams are deterministic regardless of max_workers; record values and counts are deterministic only to the extent the generation backend is (deterministic mock is bit-reproducible; real vLLM is reproducible only up to batch-composition/compute-level nondeterminism)." Pin `--collect-workers` in any compared run.

#### [LOW] Leaf evaluator uses target-Q net, but the paper specifies the online Q-head
- File: `src/verifier_anchored_rl/planner_actor.py:132-144`
- Bug: MCTS leaf bootstrap `V_hat(s') = max_a Q(s',a)` is computed from `actor.target_q` during collection, whereas method.tex (Simulation, lines 565-571) specifies `max_a Q_psi(s,a)` from the online Q-head. The code's own train.py docstring (17-19) documents the target net as "the bootstrapped leaf evaluator inside MCTS," i.e. the codebase has internally redefined the intent.
- Why it matters: spec divergence, not a training-data defect. The Q-loss still regresses the online head against stored Q_MCTS chance-node means. The only effect is that leaf values (and the Q_MCTS targets depending on them) come from a one-iteration-stale evaluator, a standard DQN-style stabilization. One verifier voted not-a-bug for exactly this reason; it is retained as a low-severity paper/implementation mismatch to reconcile.
- Suggested fix: decide intent and make code and paper agree. Either add a sentence to method.tex Section 3.7 noting the leaf uses Q_target, or change `leaf_value` to call `self.actor.q(h)`.

#### [LOW] TD priorities computed after the optimizer step, not at sample time
- File: `src/verifier_anchored_rl/train.py:99-113`
- Bug: `train_step` runs `total.backward(); optimizer.step()` then `td = td_errors(...)`, so the priority reflects the post-update network, whereas method.tex (883) describes the priority "at the time of last visit" (pre-update). The recompute is also a redundant extra forward (q_loss already computed q_pred).
- Why it matters: negligible. The priority only orders future sampling; it never weights a loss, alters a target/gradient, or feeds a reported metric. One verifier voted not-a-bug. Retained as a micro spec-fidelity / efficiency nit.
- Suggested fix: capture td_errors before backward/step (or reuse the q_pred from q_loss), which also removes the redundant forward.

#### [LOW] prefill_tokens uses whitespace word count for the text embedder but real token ids for the shared-forward encoder (mixed units)
- File: `src/verifier_anchored_rl/embedder.py:44-45` vs `src/verifier_anchored_rl/shared_forward.py:109`
- Bug: `TextEmbedder` counts `prefill_tokens += max(1, len(prefix_text.split()))` (whitespace words) while `SharedForwardEncoder` counts `prefill_tokens += int(prefix_ids.shape[1])` (true tokens). Both feed the same `IterationStats.prefill_tokens` via `_flop_counters`, so the field carries different units depending on `--embedder`, and word count systematically undercounts true tokens for math/structured text. The text default path is now `--embedder text`, so default runs use the word-count proxy.
- Why it matters: reporting only; prefill_calls is exact in both paths, and the metric never feeds training/reward/sharpening. All committed reported runs used the SharedForwardEncoder true-token path, so no published number is wrong; it biases future cross-backend compute comparisons.
- Suggested fix: count prefill_tokens from the embedding client's tokenizer (length of encoded ids) so the unit is consistent across encoders. The realistic text backend already tokenizes inside `embed()` (transformers_client.py:262), so the true count is available and simply discarded.

#### [LOW] PrefillCache.get_or_encode and TextEmbedder counters mutated by concurrent workers without a lock
- File: `src/verifier_anchored_rl/planner_actor.py:50-58` (get_or_encode); `src/verifier_anchored_rl/embedder.py:44-45` (counters)
- Bug: in the concurrent path all workers share one PrefillCache and one encoder. `get_or_encode` does a non-atomic check-then-mutate on an OrderedDict (move_to_end on hit; setitem + popitem on miss eviction) with no lock, and `encode()` releases the GIL during the forward/HTTP call. Under contention at cache capacity an unlucky interleave can raise KeyError during eviction, and the non-atomic `+=` counter increments can lose updates.
- Why it matters: cannot produce a wrong embedding (values are a pure function of the text, so a racing overwrite writes an identical value), so training records are not corrupted. Realistic outcomes: (a) a rare, loud, resumable crash during eviction at high search budgets (>4096 distinct prefixes/iteration); (b) a small undercount of the prefill FLOP metric.
- Suggested fix: guard the `get_or_encode` body and the counter increments with a `threading.Lock` (cheap relative to the GPU/HTTP forward). At minimum lock the eviction to avoid the crash.

#### [LOW] Concurrent path can leave partial duplicate records in records.jsonl after a crash during the merge loop
- File: `src/verifier_anchored_rl/outer_loop.py:292-306` (merge loop), 375-388 and 410-415 (checkpoint placement)
- Bug: the concurrent (vLLM, default) path never checkpoints inside `collect_iteration_concurrent`; the only per-iteration checkpoint is `phase="training"` at the end (410-415). The single-threaded merge loop appends this iteration's records to records.jsonl (flushed per line). A crash inside the merge loop leaves the last checkpoint at iteration it-1; resume reruns iteration it in full (concurrent ignores start_problem) and re-appends all of its records. Records carry no (iteration, problem, turn) tag, so duplicates cannot be de-duplicated offline.
- Why it matters: training data is NOT corrupted (buffer restored from the it-1 checkpoint, which predates the partial merge). Only records.jsonl (offline pi_mcts analysis) gets partial duplicate lines, biasing any offline distribution computed over it. Training/holdout numbers are unaffected.
- Suggested fix: tag each record with (iteration, problem_index, turn) for offline dedup, or rewrite records.jsonl per-iteration to a temp file with os.replace at iteration end, or truncate records.jsonl back to the last completed iteration on resume.

#### [LOW] holdout_eval.json written non-atomically; a torn write loses all accumulated cells on resume
- File: `experiments/evaluate_holdout.py:104-109` (read), 135 and 147 (write)
- Bug: progress is persisted via `out_path.write_text(json.dumps(...))` (non-atomic), unlike `run_artifacts.save_checkpoint_atomic` (temp + os.replace). The resume read `json.loads(out_path.read_text())` has no try/except, so a truncated file (from a crash mid-write or OOM/preemption during a long per-cell eval) raises and aborts the eval, discarding all previously evaluated cells.
- Why it matters: no training-data corruption (cells recompute deterministically), but it defeats the module's advertised idempotent/resumable contract and forces a full re-eval. Narrow window (single write call).
- Suggested fix: reuse an atomic write (`.tmp` + os.replace) and wrap the resume read in try/except that falls back to a fresh dict (or a .bak) on JSONDecodeError.

#### [LOW] metadata.json rewritten non-atomically on resume and update_status; a torn write breaks resume reads
- File: `experiments/run_outer_loop.py:187-196` (resume rewrite); `src/verifier_anchored_rl/run_artifacts.py:44-50` (update_status), 40 (create_run_dir)
- Bug: metadata.json is updated via plain `write_text`, not the atomic temp+replace used for checkpoints. The resume read at run_outer_loop.py:188 has no guard, so a torn metadata aborts resume; the sweep's `_status()` catches the exception and returns None, routing the run to resume-if-checkpoint-exists.
- Why it matters: metadata.json is provenance only (training state lives in the atomically-written checkpoint.pt, loaded before the metadata read), so a torn file cannot corrupt training or bias numbers. Worst case is a confusing crash on the recovery path or a sweep status mis-classification (which still routes to resume-then-crash, never silent wrong results). Narrow window.
- Suggested fix: add an atomic metadata writer (`.tmp` + os.replace) reused in create_run_dir, update_status, and the resume rewrite; guard the resume read with try/except to tolerate a torn file.

---

## 3. What was verified sound (the assurances)

Per-dimension, the following were independently checked and found correct. These are the parts you can trust.

- reward_and_verifier: terminal reward sign/magnitude (+1.0 / -1.0) and termination handling are correct; failed verifier check yields -0.05 with no promotion; evaluate-with-None is neutral 0.0; MASKED_ACTION_REWARD=-0.01 applied on unparseable JSON / nonexistent target / handler ValueError; the budget-cap override converts only an uncommitted timeout to failure and cannot clobber a genuine commit; empty-answer guard prevents vacuous matches; verifier dispatch fails closed (unknown verb / exceptions / free variables all map to success=False); SymPy equivalence checks handle algebraic/trig equivalence and the '^'->power coercion.

- mcts_core: chance-node mean estimator, per-edge discounted backup, PUCT/UCB1 selection, visit-count invariants (root.visits == num_simulations; N(s) == sum_a N(s,a)), prior computation and normalization, visit_count_policy temperature/argmax/uniform fallbacks, and the terminal-leaf-before-value_fn ordering are all correct and empirically validated. No correctness defects that would bias reported numbers.

- mdp_semantics: rng-determinism of sample_transition (seed derived solely from the passed rng, drawn unconditionally before the try block); frozen-dataclass state isolation (no input mutation, safe MCTS re-sampling, structural dedup); terminal detection and budget-exhaustion override; masked-action routing without double turn-bumps; per-verb schemas including the FORM_HYPOTHESIS "none" sentinel and empty-enum fallbacks; depth_limit masking; collision-free id generation; robust serialize/parse round-trip.

- learning_and_training: Q-loss is masked MSE of online Q_psi against stored Q_MCTS means (no grad leak into targets); policy loss is masked cross-entropy with the 0*-inf hazard zeroed; combined L_Q + lambda*L_pi; target-Q net has 0 trainable params and is hard-synced (policy head excluded), no base-model grad path; sync cadence matches Algorithm 1; prioritized-replay math (priority^alpha, eps floor, FIFO eviction, state_dict round-trip) is sound; root_q_estimates never emits a spurious masked 0 target.

- metrics_and_signal: `_entropy` is correct Shannon entropy in bits; the k>1 guard is correct; concentration/raw-entropy denominators are distinct and correct; visit-vs-prior comparison shares support and k (apples-to-apples); the prior diagnostic reuses the cached prefill and does not double-count FLOPs; diagnostics consume no rng; FLOP delta logic (before/after wrap, cache.clear forcing accurate recount) is correct; concurrent path leaves entropy/concentration as NaN cleanly.

- encoder_actor_embedder: prior() uses the online policy head (action-order aligned, sums to 1); leaf_value reads the synced target copy; `_head_input` reconciles device+dtype (bf16-CUDA hidden -> fp32-CPU heads); embed() KV-cache drop + CPU offload is idempotent and value-stable; PrefillCache keying by full canonical serialization has no wrong-embedding collisions; h_T sentinel lands on the final ']' of [STATE_END]; dim validation present; within each pipeline the stored-embedding text matches the inference text.

- concurrency_and_determinism: per-worker sub-seeding keyed by problem index (reproducible, independent of max_workers and of the main rng); buffer populated single-threaded after join in deterministic order; concurrent record shape/semantics match the sequential path; read-only shared heads under no_grad during collection; embed() is a pure function of text (cache race causes at most a redundant identical re-encode); the StateActor in-place mutation branch is never taken on the production concurrent path (gated to vLLM => text embedder => CPU fp32, cache None); sequential rng-threading across resume is correct.

- resumability_and_artifacts: checkpoint atomicity is sound (temp + os.replace) and the saved state is complete (actor incl. target submodules, optimizer, buffer, global_step, numpy rng); non-concurrent resume off-by-one is correct (completed problem skipped not re-run, rng matches); training-phase checkpoint precedes on_iteration_end so metrics rows are not double-written; metrics.jsonl/summary.md append-and-rebuild on resume is correct; sweep skip/resume/fresh decision is correct.

---

## 4. Rejected candidates worth a human glance

These were refuted by the verifiers (only 1 of 3 votes), but a few are worth a quick second look before the next batch:

- COMMIT_ANSWER unlocked by tautology-promoted verified facts (llm_context_mdp.py): largely a downstream consequence of the confirmed critical reward-farming bug. Fixing the critical finding (tie the verified fact to the target / reject tautologies) also closes this. Worth confirming the fix removes the COMMIT unlock.
- Intermediate verifier reward magnitude 0.05 vs spec 0.1 (method.tex 4.6, design.md:1848): not a correctness bug, but a real spec/code value mismatch. Worth reconciling the constant with the paper when touching the reward code anyway.
- Prioritized replay has no importance-sampling (beta) correction: refuted as not breaking training, but it is a known PER bias. A human may want to decide whether to add IS weights or document the omission.
- Two divergent state-embedding text conventions (`serialize_state` vs `"State:\n"+serialize_state`) across pipelines: refuted because each pipeline is internally consistent, but a deploy-time cross-pipeline checkpoint reuse could mismatch. Worth a guard if checkpoints are ever shared across the StateActor and trained_prior paths.
- Latent concurrent in-place mutation race in StateActor.embed if ever driven by a CUDA/cache-bearing encoder: currently blocked only by the harness guard (vLLM => text embedder). Worth a defensive assert so a future config change cannot silently re-enable the race.

---

## 5. Recommended fix order before the next experiment batch

1. Critical reward-farming (llm_context_mdp.py:1208-1250). Fix first; it is the only confirmed defect that corrupts the core training signal and falsifies a paper claim. Re-run a small collection to confirm tautology verification earns nothing and no longer unlocks COMMIT.
2. vLLM gen FLOP counters (llm_client.py:215-334). The next batch uses the vLLM backend; without this every compute number is silently 0. Add counters + lock, or make `_flop_counters` fail loud on missing counters.
3. torch seed in run_outer_loop main() (run_outer_loop.py before 165-166). Add `torch.manual_seed(args.seed)` before `_build_stack` and record it, so fresh-run and cross-budget comparisons are controlled.
4. evaluate_holdout stack reconstruction (evaluate_holdout.py:66-77). Read metadata.json and rebuild the training-time embedder/encoder, or the holdout table crashes/invalidates for every default text-embedder run.
5. Concentration normalizer k=len(avail) (outer_loop.py:167-189). Do this before any B<7 cell or before adding an 8th verb; harmless to fix now.
6. Low-severity hardening (batch together): atomic writes for holdout_eval.json and metadata.json with guarded reads; locks on PrefillCache/encoder counters; record tagging or per-iteration temp-file rewrite for records.jsonl; consistent prefill_tokens units; reconcile leaf-evaluator and TD-priority text/code with the paper; narrow the concurrent-determinism docstring.
