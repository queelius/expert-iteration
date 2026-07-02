"""Shared utilities for using a trained TrainableActor as a PUCT prior.

Used by both ``experiments/archive/run_llm_mcts_with_prior.py`` (which compares
uniform vs trained priors on a fixed initial state) and
``experiments/archive/collect_distillation.py`` (which uses the trained prior to
drive search during a fresh distillation round in expert iteration).

The two pieces:

1. ``EmbeddingCache``: caches embeddings (and failures) per unique
   serialized-state string, on top of any ``EmbeddingClient`` backend
   (Ollama, OpenAI-compatible, mock). MCTS revisits states across
   simulations and across turns in a trajectory, so the cache pays back
   quickly.

2. ``make_trained_prior``: returns a ``PriorFn`` (the signature
   ``prior_fn(state, actions) -> list[float]``) that the MCTS planner
   accepts. The factory closes over a model and an EmbeddingCache, so a
   single trained actor can be reused across many MCTS searches without
   re-loading weights. The computed policy distribution is cached per
   (state, available-verb-set) pair: the planner calls ``prior_fn`` once
   per expansion, so a decision node with k actions triggers k calls for
   the same state, and only the first pays the forward pass.

Label smoothing prevents any action from getting a zero prior, which
would make PUCT permanently ignore that edge. The default 0.02 matches
typical AlphaZero-style smoothing.

This module needs torch (to run the actor) but not httpx: the network
layer lives behind the ``EmbeddingClient`` protocol, so torch-only
offline use (loading a checkpoint to inspect or evaluate it) works
without the [llm] extra installed.
"""

from __future__ import annotations

import time
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - heads tests will catch
    torch = None  # type: ignore

from .embedding_client import EmbeddingClient, EmbeddingError
from .heads import NUM_VERBS, VERB_NAME_TO_INDEX, TrainableActor
from .lenses import global_prefix


# -----------------------------------------------------------------------------
# Checkpoint loading
# -----------------------------------------------------------------------------


def load_actor_from_checkpoint(path: str) -> tuple[TrainableActor, dict[str, Any]]:
    """Load a TrainableActor and return (model, full_checkpoint_dict).

    Uses ``weights_only=True`` (PyTorch's safe loader): the checkpoint
    contains only tensors and standard-library types (dict / list / int /
    float / str / argparse Namespace vars), so the safe loader is
    sufficient. This protects against arbitrary code execution if the
    checkpoint file is tampered with.
    """
    if torch is None:
        raise ImportError("torch is required to load a TrainableActor checkpoint")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = TrainableActor(
        embed_dim=ckpt["embed_dim"],
        hidden_dim=ckpt["hidden_dim"],
        repr_dim=ckpt["repr_dim"],
        dropout=ckpt.get("dropout", 0.0),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


# -----------------------------------------------------------------------------
# Embedding cache
# -----------------------------------------------------------------------------


class EmbeddingCache:
    """Caches embeddings (and failures) per unique serialized-state string.

    Wraps an ``EmbeddingClient`` so any backend can drive the trained
    prior; the typed ``EmbeddingError`` is the only failure mode caught
    (a non-conforming client raising something else is a programming
    error and propagates).

    Failures are negative-cached for the lifetime of the cache: the
    first failed fetch for a state records ``None``, so (a) a downed or
    misconfigured endpoint cannot stall every subsequent expansion with
    a full network timeout, and (b) all sibling expansions of the same
    decision node see the same uniform fallback, keeping sibling priors
    a coherent distribution instead of a uniform/trained mix that does
    not sum to 1.

    Counters: ``fetch_count`` (successful fetches), ``hit_count``
    (positive cache hits), ``failure_count`` (fetch attempts that raised
    ``EmbeddingError``), ``negative_hit_count`` (hits on negatively
    cached states), ``fetch_wallclock_s`` (time spent in all fetch
    attempts, including failed ones).
    """

    def __init__(self, client: EmbeddingClient):
        self.client = client
        self.cache: dict[str, list[float] | None] = {}
        self.fetch_count = 0
        self.hit_count = 0
        self.failure_count = 0
        self.negative_hit_count = 0
        self.fetch_wallclock_s = 0.0

    def get(self, serialized_state: str) -> list[float] | None:
        if serialized_state in self.cache:
            cached = self.cache[serialized_state]
            if cached is None:
                self.negative_hit_count += 1
            else:
                self.hit_count += 1
            return cached
        t0 = time.time()
        try:
            emb = self.client.embed(serialized_state)
        except EmbeddingError as e:
            self.fetch_wallclock_s += time.time() - t0
            self.failure_count += 1
            self.cache[serialized_state] = None
            print(
                f"  ! embedding fetch failed (negative-cached for this state): {e}",
                flush=True,
            )
            return None
        self.fetch_wallclock_s += time.time() - t0
        self.fetch_count += 1
        self.cache[serialized_state] = emb
        return emb

    def stats(self) -> str:
        """One-line human-readable counter summary for run reports."""
        return (
            f"{self.fetch_count} fetches ({self.fetch_wallclock_s:.1f}s), "
            f"{self.hit_count} hits, {self.failure_count} failures, "
            f"{self.negative_hit_count} negative hits"
        )


# -----------------------------------------------------------------------------
# Trained-prior factory
# -----------------------------------------------------------------------------


def make_trained_prior(
    model: TrainableActor,
    embedding_cache: EmbeddingCache,
    *,
    smoothing: float = 0.02,
):
    """Return a PriorFn that uses pi_theta(s) restricted to ``actions``.

    The prior is computed by:

      1. Serialize the state.
      2. Fetch the embedding (cached, failures negative-cached).
      3. Run ``model.pi(emb, mask)`` with a mask over the verbs in
         ``actions`` (cached per (state, verb-set): the planner calls
         prior_fn once per expansion, so a node with k actions makes k
         calls for the same state, and only the first pays the forward).
      4. Reorder pi to match the ``actions`` list order (MCTS asks for
         one probability per action in the order it was given).

    If the embedding fetch fails, or none of the action verbs are known
    to the head's verb vocabulary, fall back to uniform.

    ``smoothing`` adds an epsilon to every action's probability and
    re-normalizes; this keeps any verb from getting a zero prior, which
    prevents PUCT from never exploring those edges. The default 0.02
    matches typical AlphaZero-style label smoothing.
    """
    if torch is None:
        raise ImportError("torch is required to use a trained prior")

    # pi depends on the state AND on which verbs are available, so the
    # cache key includes both. In DeliberationMDP the available set is a
    # deterministic function of the state, making the verb-set component
    # redundant there, but other GenerativeMDPs may pass action subsets.
    pi_cache: dict[tuple[str, frozenset[str]], Any] = {}

    def prior_fn(state, actions: list) -> list[float]:
        n = len(actions)
        if n == 0:
            return []
        serialized = global_prefix(state)
        verb_names = frozenset(
            action.verb.value
            for action in actions
            if action.verb.value in VERB_NAME_TO_INDEX
        )
        if not verb_names:
            # No action's verb is in the head's vocabulary; the masked
            # softmax would have no support. Uniform is the only honest
            # prior here.
            return [1.0 / n] * n

        cache_key = (serialized, verb_names)
        pi = pi_cache.get(cache_key)
        if pi is None:
            emb = embedding_cache.get(serialized)
            if emb is None:
                return [1.0 / n] * n
            expected = getattr(model, "embed_dim", None)
            if expected is not None and len(emb) != expected:
                # Embedder/checkpoint mismatch (e.g. a 2048-d checkpoint
                # served by a 768-d embedding model). Fail with the cause,
                # not an opaque matmul shape error mid-search.
                raise ValueError(
                    f"embedding dimension {len(emb)} does not match the "
                    f"checkpoint's embed_dim {expected}; the trained prior "
                    f"must use the same embedding model its heads were "
                    f"trained on"
                )
            with torch.no_grad():
                x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
                mask = torch.zeros((1, NUM_VERBS), dtype=torch.bool)
                for name in verb_names:
                    mask[0, VERB_NAME_TO_INDEX[name]] = True
                pi = model.pi(x, mask)[0].numpy()
            pi_cache[cache_key] = pi

        # ``actions`` is over (verb, target) pairs, but pi is verb-indexed:
        # several actions can share a verb (e.g. ABANDON legal on every open
        # node). Splitting the verb's mass evenly across its legal targets
        # before smoothing/renormalizing keeps a verb's aggregate mass equal
        # to the head's trained per-verb marginal instead of scaling with
        # its target count. Same pattern as StateActor.prior.
        idxs = [VERB_NAME_TO_INDEX.get(action.verb.value) for action in actions]
        verb_counts: dict[int, int] = {}
        for idx in idxs:
            if idx is not None:
                verb_counts[idx] = verb_counts.get(idx, 0) + 1
        probs = []
        for idx in idxs:
            probs.append(float(pi[idx]) / verb_counts[idx] if idx is not None else 0.0)

        smoothed = [p + smoothing for p in probs]
        total = sum(smoothed)
        if total <= 0:
            return [1.0 / n] * n
        return [p / total for p in smoothed]

    return prior_fn


__all__ = [
    "EmbeddingCache",
    "load_actor_from_checkpoint",
    "make_trained_prior",
]
