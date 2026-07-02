import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_public_api_is_importable():
    import verifier_anchored_rl.reductio as reductio
    for name in ["new_graph", "split", "form_hypothesis", "compute",
                 "verify", "commit", "abandon", "scope", "render",
                 "Verifier", "Ok", "Err"]:
        assert hasattr(reductio, name), name


def test_reductio_modules_add_no_heavy_imports():
    # The parent package (verifier_anchored_rl/__init__.py) eagerly imports
    # numpy and the LLM client stack, and Python must execute it before any
    # subpackage import. So the boundary claim for the vendored core is a
    # DELTA claim: importing reductio's modules adds nothing banned beyond
    # what the host package already loaded. Run in a fresh interpreter so
    # the surrounding test session cannot mask a leak.
    code = (
        "import importlib, sys\n"
        "importlib.import_module('verifier_anchored_rl')\n"
        "before = set(sys.modules)\n"
        "mods = ['verifier_anchored_rl.reductio',\n"
        "        'verifier_anchored_rl.reductio.graph',\n"
        "        'verifier_anchored_rl.reductio.actions',\n"
        "        'verifier_anchored_rl.reductio.verify',\n"
        "        'verifier_anchored_rl.reductio.lens',\n"
        "        'verifier_anchored_rl.reductio.serialize']\n"
        "for m in mods: importlib.import_module(m)\n"
        "added = set(sys.modules) - before\n"
        "banned = {'sympy', 'torch', 'httpx', 'requests', 'numpy'}\n"
        "leaked = banned & added\n"
        "assert not leaked, f'reductio import pulled in: {leaked}'\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    subprocess.run([sys.executable, "-c", code], check=True,
                   cwd=str(REPO_ROOT), env=env)


def test_no_reward_or_value_field_on_nodes():
    from verifier_anchored_rl.reductio import graph
    for cls in (graph.Goal, graph.Hypothesis, graph.Fact, graph.VerifierCall):
        fields = cls.__dataclass_fields__.keys()
        assert "reward" not in fields and "value" not in fields


def test_method_layer_is_gone():
    from verifier_anchored_rl.reductio import graph
    assert not hasattr(graph, "Method")
