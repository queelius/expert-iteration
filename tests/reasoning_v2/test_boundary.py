import os
import subprocess
import sys
from pathlib import Path

from verifier_anchored_rl import reasoning_v2


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_public_api_exposes_the_v2_algebra():
    for name in (
        "GoalObligation",
        "GenerationContract",
        "Claim",
        "decompose",
        "ResultPayload",
        "ResultDraft",
        "propose_result",
        "validate_observation_with_contract",
        "propose_solution",
        "derive",
        "aggregate",
        "propose_refutation",
        "verify_claim",
        "commit_claim",
        "prune",
        "TableDrivenVerifier",
        "ClosureVerifier",
        "run_closure",
        "macro_step",
        "validate_macro_transition",
    ):
        assert hasattr(reasoning_v2, name), name


def test_generation_and_receipt_runtime_protocols_are_distinct():
    class ObligationOnly:
        def canonicalize_obligation(self, obligation):
            return obligation

    verifier = reasoning_v2.TableDrivenVerifier(())

    assert not isinstance(ObligationOnly(), reasoning_v2.GenerationContract)
    assert isinstance(verifier, reasoning_v2.GenerationContract)
    assert isinstance(verifier, reasoning_v2.VerifierReceiptContract)


def test_reasoning_v2_core_adds_no_heavy_imports():
    code = (
        "import importlib, sys\n"
        "importlib.import_module('verifier_anchored_rl')\n"
        "before = set(sys.modules)\n"
        "mods = ['verifier_anchored_rl.reasoning_v2',\n"
        "        'verifier_anchored_rl.reasoning_v2.model',\n"
        "        'verifier_anchored_rl.reasoning_v2.actions',\n"
        "        'verifier_anchored_rl.reasoning_v2.verifier']\n"
        "for module in mods: importlib.import_module(module)\n"
        "added = set(sys.modules) - before\n"
        "banned = {'sympy', 'torch', 'httpx', 'requests', 'numpy'}\n"
        "leaked = banned & added\n"
        "assert not leaked, f'reasoning_v2 import pulled in: {leaked}'\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=str(REPO_ROOT),
        env=env,
    )


def test_v2_nodes_have_no_reward_or_value_fields():
    for cls in (reasoning_v2.Goal, reasoning_v2.Evidence, reasoning_v2.Claim):
        fields = cls.__dataclass_fields__
        assert "reward" not in fields
        assert "value" not in fields
