"""CPU test of the held-out rollout logic (Echo MDP + FakeEncoder, no GPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _stack():
    from tests.fakes import FakeEncoder
    from verifier_anchored_rl.architecture import HiddenStateActor
    from verifier_anchored_rl.planner_actor import PrefillCache, StateActor

    enc = FakeEncoder(embed_dim=16)
    actor = HiddenStateActor(embed_dim=16)
    return StateActor(enc, actor, PrefillCache(), depth_limit=3)


def _mdp():
    from verifier_anchored_rl.llm_client import EchoLLMClient
    from verifier_anchored_rl.llm_context_mdp import LLMContextMDP

    return LLMContextMDP(problem_statement="Compute 7*6", expected_answer="42",
                         llm_client=EchoLLMClient(), depth_limit=3, max_turns=4)


@pytest.mark.parametrize("budget", [0, 2])
def test_rollout_terminates_and_reports_success_bool(budget):
    pytest.importorskip("torch")
    import numpy as np
    from experiments.evaluate_holdout import rollout

    res = rollout(_mdp(), _stack(), budget=budget, c_puct=1.5, discount=1.0,
                  max_turns=4, rng=np.random.default_rng(0))
    assert set(res) == {"committed", "success", "turns", "total_reward"}
    assert isinstance(res["success"], bool)
    assert 0 <= res["turns"] <= 4


# ---------------------------------------------------------------------------
# Helpers for _build_eval_stack tests: lightweight fakes that expose
# the attrs the function under test reads (.embed_dim, .model.config.hidden_size)
# without loading any GPU model.
# ---------------------------------------------------------------------------

FAKE_HIDDEN_SIZE = 256


def _fake_emb_client():
    """Fake TransformersEmbeddingClient with .model.config.hidden_size."""
    cfg = MagicMock()
    cfg.hidden_size = FAKE_HIDDEN_SIZE
    model = MagicMock()
    model.config = cfg
    client = MagicMock()
    client.model = model
    return client


def _fake_encoder(embed_dim=FAKE_HIDDEN_SIZE):
    """Fake encoder with .embed_dim."""
    enc = MagicMock()
    enc.embed_dim = embed_dim
    return enc


def _fake_actor():
    act = MagicMock()
    return act


def _fake_state_actor():
    return MagicMock()


def _fake_prefill_cache():
    return MagicMock()


def _patch_all(monkeypatch, *, emb_client=None, encoder=None,
               text_embedder_cls=None, hidden_state_encoder_cls=None,
               actor_cls=None, state_actor_cls=None, prefill_cache_cls=None,
               llm_client_cls=None, openai_llm_cls=None):
    """Patch all heavy constructors that _build_eval_stack may call."""
    if emb_client is None:
        emb_client = _fake_emb_client()
    if encoder is None:
        encoder = _fake_encoder()

    # TextEmbedder: constructor receives (emb_client, embed_dim=...)
    te_instance = _fake_encoder(embed_dim=FAKE_HIDDEN_SIZE)
    if text_embedder_cls is None:
        text_embedder_cls = MagicMock(return_value=te_instance)

    # SharedForwardEncoder.from_llm_client: classmethod
    sfe_instance = _fake_encoder(embed_dim=512)
    sfe_cls = MagicMock()
    sfe_cls.from_llm_client = MagicMock(return_value=sfe_instance)
    if hidden_state_encoder_cls is None:
        hidden_state_encoder_cls = sfe_cls

    # TransformersEmbeddingClient
    tec_cls = MagicMock(return_value=emb_client)

    # TransformersLLMClient
    tllm_instance = MagicMock()
    tllm_cls = MagicMock(return_value=tllm_instance)

    # OpenAICompatibleLLMClient
    ollm_instance = MagicMock()
    ollm_cls = MagicMock(return_value=ollm_instance)

    # HiddenStateActor
    actor_instance = _fake_actor()
    act_cls = MagicMock(return_value=actor_instance)

    # StateActor
    sa_instance = _fake_state_actor()
    sa_cls = MagicMock(return_value=sa_instance)

    # PrefillCache
    pc_instance = _fake_prefill_cache()
    pc_cls = MagicMock(return_value=pc_instance)

    monkeypatch.setattr(
        "verifier_anchored_rl.shared_forward.SharedForwardEncoder",
        hidden_state_encoder_cls,
    )
    monkeypatch.setattr(
        "verifier_anchored_rl.embedder.TextEmbedder",
        text_embedder_cls,
    )
    monkeypatch.setattr(
        "verifier_anchored_rl.transformers_client.TransformersEmbeddingClient",
        tec_cls,
    )
    monkeypatch.setattr(
        "verifier_anchored_rl.transformers_client.TransformersLLMClient",
        tllm_cls,
    )
    monkeypatch.setattr(
        "verifier_anchored_rl.llm_client.OpenAICompatibleLLMClient",
        ollm_cls,
    )
    monkeypatch.setattr(
        "verifier_anchored_rl.architecture.HiddenStateActor",
        act_cls,
    )
    monkeypatch.setattr(
        "verifier_anchored_rl.planner_actor.StateActor",
        sa_cls,
    )
    monkeypatch.setattr(
        "verifier_anchored_rl.planner_actor.PrefillCache",
        pc_cls,
    )

    return {
        "text_embedder_cls": text_embedder_cls,
        "hidden_state_encoder_cls": hidden_state_encoder_cls,
        "tec_cls": tec_cls,
        "tllm_cls": tllm_cls,
        "ollm_cls": ollm_cls,
        "actor_cls": act_cls,
        "state_actor_cls": sa_cls,
        "prefill_cache_cls": pc_cls,
        "te_instance": te_instance,
        "sfe_instance": sfe_instance,
        "emb_client": emb_client,
    }


# ---------------------------------------------------------------------------
# Bug 1 tests: _build_eval_stack reads metadata and rebuilds the right stack
# ---------------------------------------------------------------------------

def test_eval_stack_text_embedder_from_metadata(tmp_path, monkeypatch):
    """New-style run with embedder=text and gen_backend=vllm: TextEmbedder path."""
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(json.dumps({
        "args": {
            "embedder": "text",
            "embedder_model": "m-small",
            "gen_backend": "vllm",
            "vllm_url": "http://x/v1",
            "model": "m-3b",
        }
    }))

    mocks = _patch_all(monkeypatch)

    # Must import AFTER patching so the module sees the patched names.
    # Re-import to get a fresh reference if the module is already imported.
    import importlib
    import experiments.evaluate_holdout as mod
    importlib.reload(mod)

    llm, actor, state_actor = mod._build_eval_stack(run_dir, depth_limit=3)

    # TextEmbedder must have been constructed (not SharedForwardEncoder).
    mocks["text_embedder_cls"].assert_called_once()
    mocks["hidden_state_encoder_cls"].from_llm_client.assert_not_called()
    # OpenAICompatibleLLMClient used (vllm backend).
    mocks["ollm_cls"].assert_called_once()
    mocks["tllm_cls"].assert_not_called()
    # Actor embed_dim comes from the embedding client's hidden_size.
    mocks["actor_cls"].assert_called_once()
    call_kwargs = mocks["actor_cls"].call_args
    # embed_dim should equal FAKE_HIDDEN_SIZE (from emb_client.model.config.hidden_size)
    assert call_kwargs.kwargs.get("embed_dim") == FAKE_HIDDEN_SIZE or \
           (call_kwargs.args and call_kwargs.args[0] == FAKE_HIDDEN_SIZE)


def test_eval_stack_legacy_metadata_defaults_to_hidden_state(tmp_path, monkeypatch):
    """Old checkpoint with only 'model' in args: falls back to hidden_state/transformers."""
    run_dir = tmp_path / "run_legacy"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(json.dumps({
        "args": {"model": "m-1.5b"}
    }))

    mocks = _patch_all(monkeypatch)

    import importlib
    import experiments.evaluate_holdout as mod
    importlib.reload(mod)

    llm, actor, state_actor = mod._build_eval_stack(run_dir, depth_limit=3)

    # SharedForwardEncoder path (legacy hidden_state).
    mocks["hidden_state_encoder_cls"].from_llm_client.assert_called_once()
    mocks["text_embedder_cls"].assert_not_called()
    # TransformersLLMClient (legacy transformers backend).
    mocks["tllm_cls"].assert_called_once()
    mocks["ollm_cls"].assert_not_called()


def test_eval_stack_missing_metadata_fails_fast(tmp_path, monkeypatch):
    """No metadata.json -> SystemExit with a clear message."""
    run_dir = tmp_path / "run_no_meta"
    run_dir.mkdir()
    # Deliberately do NOT write metadata.json.

    _patch_all(monkeypatch)

    import importlib
    import experiments.evaluate_holdout as mod
    importlib.reload(mod)

    with pytest.raises(SystemExit):
        mod._build_eval_stack(run_dir, depth_limit=3)


# ---------------------------------------------------------------------------
# Bug 2 test: torn holdout_eval.json on resume falls back gracefully
# ---------------------------------------------------------------------------

def _load_results_fn():
    """Import the factored-out _load_results helper from evaluate_holdout."""
    import importlib
    import experiments.evaluate_holdout as mod
    importlib.reload(mod)
    return mod._load_results


def test_holdout_eval_resume_tolerates_torn_json(tmp_path):
    """A truncated holdout_eval.json on disk does not abort the run."""
    run_dir = tmp_path / "run_torn"
    run_dir.mkdir()
    out_path = run_dir / "holdout_eval.json"
    # Write deliberately truncated JSON (simulates a crash mid-write).
    out_path.write_text('{"checkpoint": "/ck", "per_cell": {"B0/p/seed0": {"success"')

    _load_results = _load_results_fn()

    # Should NOT raise; must return a fresh results dict.
    result = _load_results(out_path, force=False, run_dir=run_dir, model="any-model")
    assert "per_cell" in result
    # Torn file should yield an empty per_cell (fresh start), not a partial one.
    assert result["per_cell"] == {}
