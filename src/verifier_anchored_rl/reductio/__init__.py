"""reductio: the immutable reasoning forest and verb algebra (state core).

Vendored from github.com/queelius/reductio and slimmed to the minimal
state model of new-design.md: no Method node layer; SPLIT attaches child
subgoals directly and records the connective as non-gating metadata.
"""

from .graph import (
    Graph, NodeId, Goal, Hypothesis, Fact, VerifierCall,
    new_graph, get, solved, is_verified, is_abandoned, holds,
)
from .actions import (
    Result, Ok, Err, split, form_hypothesis, compute, verify, commit, abandon,
)
from .verify import Verifier, VerifierResult, MockVerifier
from .lens import Subgraph, scope
from .serialize import render, parse

__all__ = [
    "Graph", "NodeId", "Goal", "Hypothesis", "Fact", "VerifierCall",
    "new_graph", "get", "solved", "is_verified", "is_abandoned", "holds",
    "Result", "Ok", "Err", "split", "form_hypothesis", "compute", "verify",
    "commit", "abandon", "Verifier", "VerifierResult", "MockVerifier",
    "Subgraph", "scope", "render", "parse",
]
