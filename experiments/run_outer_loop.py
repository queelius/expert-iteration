"""Run the outer collection-and-train loop on toy problems (method.tex 3.7),
with full observability and resume. See
docs/superpowers/specs/2026-06-19-observable-resumable-outer-loop-design.md.

Each run gets a unique directory under --run-root with metadata.json,
progress.log (flushed, tee'd to stdout), records.jsonl, metrics.jsonl, a
per-trajectory checkpoint.pt, and a summary.md rewritten each iteration.
--resume <run-dir> continues a crashed run from its checkpoint."""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np

from verifier_anchored_rl.outer_loop import run_outer_loop
from verifier_anchored_rl.run_artifacts import (
    append_jsonl,
    create_run_dir,
    load_checkpoint,
    save_checkpoint_atomic,
    seed_everything,
    update_status,
    write_json_atomic,
)


def _git_sha() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip())
        return sha, dirty
    except Exception:
        return "nogit", False


def _gpu_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {"gpu": torch.cuda.get_device_name(0),
                    "cuda": torch.version.cuda}
    except Exception:
        pass
    return {"gpu": "cpu", "cuda": None}


def _build_stack(args):
    """Construct the run stack. Heavy: loads the frozen model(s). Factored out
    so tests can stub it. Branches on two selectors: --gen-backend (vllm or
    transformers) for generation and --embedder (text or hidden_state) for the
    StateActor's encoder. Returns (encoder, state_actor, trainer, buffer, actor,
    optimizer, llm_client, cache)."""
    import torch

    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.planner_actor import PrefillCache, StateActor
    from verifier_anchored_rl.replay import ReplayBuffer
    from verifier_anchored_rl.train import Trainer

    # Generation backend.
    if args.gen_backend == "vllm":
        from verifier_anchored_rl.llm_client import OpenAICompatibleLLMClient
        llm_client = OpenAICompatibleLLMClient(
            base_url=args.vllm_url, model=args.model, guided_json=True)
    else:
        from verifier_anchored_rl.transformers_client import TransformersLLMClient
        llm_client = TransformersLLMClient(model_name=args.model)

    # State embedder (the StateActor's encoder).
    if args.embedder == "text":
        from verifier_anchored_rl.embedder import TextEmbedder
        from verifier_anchored_rl.transformers_client import TransformersEmbeddingClient
        emb_client = TransformersEmbeddingClient(model_name=args.embedder_model)
        embed_dim = int(emb_client.model.config.hidden_size)
        encoder = TextEmbedder(emb_client, embed_dim=embed_dim)
    else:  # hidden_state: legacy h_T path; needs an in-process generation model
        from verifier_anchored_rl.shared_forward import SharedForwardEncoder
        if args.gen_backend != "transformers":
            raise SystemExit(
                "embedder=hidden_state requires gen-backend=transformers "
                "(the h_T encoder needs the in-process generation model; "
                "vLLM does not expose hidden states). Use --embedder text with vLLM.")
        encoder = SharedForwardEncoder.from_llm_client(llm_client)

    cache = PrefillCache(capacity=4096)
    actor = HiddenStateActor(embed_dim=encoder.embed_dim)
    state_actor = StateActor(encoder, actor, cache, depth_limit=args.depth_limit)
    buffer = ReplayBuffer(capacity=args.capacity)
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    trainer = Trainer(actor, opt, buffer, sync_interval=args.sync_interval)
    return encoder, state_actor, trainer, buffer, actor, opt, llm_client, cache


def _save_checkpoint_for(run_dir, actor, opt, buffer, trainer, rng):
    """Return a save_checkpoint(state) closure that assembles the full
    checkpoint dict from the live objects and writes it atomically."""
    ckpt_path = str(Path(run_dir) / "checkpoint.pt")

    def _save(state: dict) -> None:
        full = {
            **state,
            "actor_state": actor.state_dict(),
            "optimizer_state": opt.state_dict(),
            "buffer": buffer.state_dict(),
            "global_step": trainer.global_step,
            "numpy_rng_state": rng.bit_generator.state,
        }
        save_checkpoint_atomic(ckpt_path, full)

    return _save


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--num-simulations", type=int, default=16)
    p.add_argument("--train-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--min-buffer", type=int, default=16)
    p.add_argument("--capacity", type=int, default=2048)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--discount", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--sync-interval", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--depth-limit", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--problems", nargs="+", default=["train"])
    p.add_argument("--run-root", default="experiments/results/runs")
    p.add_argument("--tag", default=None)
    p.add_argument("--notes", default="",
                   help="motivation/notes for this run, recorded in metadata.json")
    p.add_argument("--resume", default=None,
                   help="path to an existing run dir to resume")
    p.add_argument("--gen-backend", choices=["vllm", "transformers"], default="vllm",
                   help="generation backend: vllm (OpenAI-compatible server, guided_json, batched) or transformers (in-process)")
    p.add_argument("--vllm-url", default="http://localhost:8000/v1",
                   help="base URL of the vLLM OpenAI-compatible server (gen-backend=vllm)")
    p.add_argument("--embedder", choices=["text", "hidden_state"], default="text",
                   help="state embedder: text (dedicated TransformersEmbeddingClient) or hidden_state (legacy SharedForwardEncoder h_T)")
    p.add_argument("--embedder-model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="model id for the text embedder (gen decoupled from embedding)")
    p.add_argument("--collect-workers", type=int, default=8,
                   help="concurrent trajectory workers (vLLM batching); used when gen-backend=vllm")
    args = p.parse_args(argv)

    try:
        from problems import PROBLEMS, resolve_problem_selection
    except ImportError:
        from experiments.problems import PROBLEMS, resolve_problem_selection

    keys = resolve_problem_selection(args.problems)
    problems = [(PROBLEMS[k]["problem"], PROBLEMS[k]["expected"]) for k in keys]

    # Seed all RNGs before constructing the stack so that weight inits,
    # dropout masks, and any other torch random draws are reproducible from
    # --seed alone. numpy rng is re-seeded below via default_rng for the
    # trajectory rng (which is checkpointed separately).
    seed_everything(args.seed)

    (encoder, state_actor, trainer, buffer, actor, opt,
     llm, cache) = _build_stack(args)
    rng = np.random.default_rng(args.seed)

    start_iteration, start_problem = 1, 0
    if args.resume:
        run_dir = Path(args.resume)
        ckpt = load_checkpoint(str(run_dir / "checkpoint.pt"))
        actor.load_state_dict(ckpt["actor_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        buffer.load_state_dict(ckpt["buffer"])
        trainer.global_step = ckpt["global_step"]
        rng.bit_generator.state = ckpt["numpy_rng_state"]
        if ckpt["phase"] == "collecting":
            start_iteration = ckpt["iteration"]
            start_problem = ckpt["problems_done_this_iter"]
        else:
            start_iteration = ckpt["iteration"] + 1
            start_problem = 0
        # Mark the run live again and record that a resume happened (the dir
        # was left "failed" by the interrupted process). Status flips to
        # "completed" at the end as usual.
        meta_path = run_dir / "metadata.json"
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            # Torn write from a previous crash: fall back to a minimal dict
            # so the resume can proceed. Provenance fields are lost but
            # training state is intact (checkpoint.pt is atomic).
            meta = {"status": "running"}
        meta["status"] = "running"
        meta["resume_count"] = int(meta.get("resume_count", 0)) + 1
        meta["resumed_from"] = {
            "phase": ckpt["phase"], "iteration": ckpt["iteration"],
            "global_step": ckpt["global_step"],
            "start_iteration": start_iteration, "start_problem": start_problem,
        }
        write_json_atomic(meta_path, meta)
    else:
        meta = {
            "notes": args.notes,
            "args": vars(args), "hostname": socket.gethostname(),
            "python": platform.python_version(), **_gpu_info(),
            "model": args.model, "seed": args.seed,
            "torch_seed": args.seed,
        }
        sha, dirty = _git_sha()
        meta["git_sha"], meta["git_dirty"] = sha, dirty
        try:
            import torch
            import transformers
            meta["torch"] = torch.__version__
            meta["transformers"] = transformers.__version__
            meta["dtype"] = str(getattr(encoder.model, "dtype", "unknown"))
            meta["device"] = str(getattr(encoder.model, "device", "unknown"))
        except Exception:
            pass
        run_dir = create_run_dir(root=args.run_root, tag=args.tag, metadata=meta)

    log_path = run_dir / "progress.log"
    records_path = str(run_dir / "records.jsonl")
    metrics_path = str(run_dir / "metrics.jsonl")
    summary_path = run_dir / "summary.md"
    # On resume, re-seed the rendered summary from the iterations already
    # recorded in metrics.jsonl (which is append-only and complete) so the
    # rewritten summary.md keeps the pre-resume rows rather than starting blank.
    summaries = []
    if args.resume and Path(metrics_path).exists():
        with open(metrics_path) as f:
            summaries = [json.loads(line) for line in f if line.strip()]

    def log(msg: str) -> None:
        line = msg if msg.endswith("\n") else msg + "\n"
        with open(log_path, "a") as f:
            f.write(line)
            f.flush()
        sys.stdout.write(line)
        sys.stdout.flush()

    def mdp_factory(problem):
        from verifier_anchored_rl.deliberation_mdp import DeliberationMDP
        # Generate verbs via the constrained chat path (llm_client.complete_json
        # with schema-constrained decoding), NOT the encoder's raw-completion
        # path, which a small instruct model cannot drive (it masked every verb,
        # collapsing transitions to no-ops). The encoder is still used by the
        # StateActor for embed(s) = h_T. This is two forwards per decision-node
        # (embed + generate); folding generation back into the encoder's shared
        # prefill (one-forward) is a tracked follow-up.
        # DeliberationMDP folds the old max_turns cap into depth_limit
        # (is_terminal fires at state.turn >= depth_limit); the outer loop's
        # own max_turns argument (below) remains a separate rollout-level cap.
        return DeliberationMDP(
            problem=problem[0], expected_answer=problem[1],
            llm_client=llm, depth_limit=args.depth_limit)

    def on_decision(info):
        log(f"iter {info['iteration']} prob {info['iteration_problem_index']} "
            f"turn {info['turn']}: verb={info['chosen_verb']} "
            f"pi_argmax={info.get('pi_argmax_verb','?')}({info.get('pi_argmax_prob', float('nan')):.2f}) "
            f"H_visit={info.get('pi_mcts_entropy', float('nan')):.2f} "
            f"H_prior={info.get('prior_entropy', float('nan')):.2f} "
            f"gen={info.get('gen_calls',0)} ({info.get('gen_tokens',0)} tok) "
            f"prefill={info.get('prefill_calls',0)} ({info.get('prefill_tokens',0)} tok)")

    def on_train_step(info):
        log(f"  train step {info['global_step']}: "
            f"L_Q {info['q']:.4f} L_pi {info['pi']:.4f} L_total {info['total']:.4f}")

    def on_iteration_end(s):
        row = {
            "iteration": s.iteration, "decisions": s.n_decisions,
            "sims": s.n_simulations, "buffer": s.buffer_size,
            "prefill_calls": s.prefill_calls, "prefill_tokens": s.prefill_tokens,
            "gen_calls": s.gen_calls, "gen_tokens": s.gen_tokens,
            "L_Q_first": s.q_first, "L_Q_last": s.q_last,
            "L_pi_first": s.pi_first, "L_pi_last": s.pi_last,
            "L_total_first": s.total_first, "L_total_last": s.total_last,
            "visit_entropy": s.visit_entropy, "prior_entropy": s.prior_entropy,
            "visit_concentration": s.visit_concentration,
            "prior_concentration": s.prior_concentration,
            "wall_seconds": s.wall_seconds,
        }
        append_jsonl(metrics_path, row)
        summaries.append(row)
        lines = ["# Run summary", "",
                 "| iter | decisions | buffer | gen tok | L_Q first to last "
                 "| L_pi last | visit conc | prior conc |",
                 "|---|---:|---:|---:|---|---:|---:|---:|"]
        for r in summaries:
            lines.append(
                f"| {r['iteration']} | {r['decisions']} | {r['buffer']} | "
                f"{r['gen_tokens']} | {r['L_Q_first']:.4f} to {r['L_Q_last']:.4f} | "
                f"{r.get('L_pi_last', float('nan')):.4f} | "
                f"{r.get('visit_concentration', float('nan')):.3f} | "
                f"{r.get('prior_concentration', float('nan')):.3f} |")
        summary_path.write_text("\n".join(lines) + "\n")
        log(f"=== iter {s.iteration} done in {s.wall_seconds:.0f}s: "
            f"L_total {s.total_first:.4f} to {s.total_last:.4f} ===")

    save_ckpt = _save_checkpoint_for(run_dir, actor, opt, buffer, trainer, rng)

    log(f"=== Outer loop: {len(problems)} problems, model={args.model}, "
        f"run_dir={run_dir} ===")
    try:
        run_outer_loop(
            problems, mdp_factory, state_actor, trainer, buffer,
            n_iterations=args.iterations, train_steps_per_iter=args.train_steps,
            batch_size=args.batch_size, min_buffer_size=args.min_buffer,
            num_simulations=args.num_simulations, c_puct=args.c_puct,
            discount=args.discount, tau=args.tau, max_turns=args.max_turns,
            rng=rng, encoder=encoder, gen_source=llm,
            concurrent=(args.gen_backend == "vllm"),
            collect_workers=args.collect_workers, base_seed=args.seed,
            record_sink=lambda rec: append_jsonl(records_path, rec),
            on_decision=on_decision, on_train_step=on_train_step,
            on_iteration_end=on_iteration_end, save_checkpoint=save_ckpt,
            log_every=args.log_every,
            start_iteration=start_iteration, start_problem=start_problem)
    except BaseException:
        update_status(run_dir, "failed")
        raise
    update_status(run_dir, "completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
