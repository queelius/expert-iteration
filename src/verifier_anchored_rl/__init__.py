"""verifier_anchored_rl: Verifier-anchored RL with MCTS expert iteration.

Phase 1 implements the classical-MDP scaffold: finite-MDP and generative-MDP
protocols, a stochastic Gridworld environment, an exact value-iteration
solver, a depth-limited expectimax planner, and an MCTS planner whose
chance-node Q estimates serve as sample-efficient Bellman targets.

Phase 2 adds the expert-iteration loop: a tabular softmax policy, a
training loop that distills MCTS-improved visit-count distributions back
into the policy, and evaluation utilities for fast mode (no search) and
search mode (MCTS at action selection).

Phase 3 adds the LLM context MDP in ``llm_context_mdp``:
``MathState``, ``MathAction``, the seven-verb ``Verb`` enum, the
structured-marker serializer, and a fully implemented ``LLMContextMDP``
whose ``sample_transition`` calls the LLM via the ``llm_client``
abstraction (``MockLLMClient`` for tests, ``EchoLLMClient`` for
zero-config smoke checks, ``OpenAICompatibleLLMClient`` for real
vLLM/Ollama/llama.cpp HTTP backends). The symbolic verifier lives in
``verifier`` (five INVOKE_VERIFIER sub-verbs plus ``check_answer`` for
COMMIT_ANSWER). LLM-extension dependencies (``httpx``, ``sympy``) are
optional and only required to exercise the LLM-backed path; install
with ``pip install -e .[llm]``.

See design.md (repository root) for the full specification and roadmap.
"""

from .eval import evaluate_policy, evaluate_with_search
from .expert_iteration import (
    ExpertIterationConfig,
    IterationResult,
    PolicyTarget,
    run_expert_iteration,
)
from .embedding_client import (
    EmbeddingClient,
    EmbeddingError,
    MockEmbeddingClient,
    OllamaEmbeddingClient,
    OpenAICompatibleEmbeddingClient,
)
from .llm_client import (
    EchoLLMClient,
    LLMClient,
    LLMResponseError,
    MockLLMClient,
    OpenAICompatibleLLMClient,
)
from .llm_context_mdp import (
    Fact,
    Hypothesis,
    LLMContextMDP,
    MathAction,
    MathState,
    Subgoal,
    Verb,
    VerifierCall,
    available_actions,
    parse_state,
    serialize_state,
)
from .mcts import (
    ChanceNode,
    DecisionNode,
    MCTSConfig,
    best_action,
    mcts_search,
    root_q_estimates,
    visit_count_policy,
)
from .policies import TabularPolicy, policy_as_prior
from .transformers_client import (
    TransformersEmbeddingClient,
    TransformersLLMClient,
)

__version__ = "0.1.0"

__all__ = [
    "ChanceNode",
    "DecisionNode",
    "EchoLLMClient",
    "EmbeddingClient",
    "EmbeddingError",
    "ExpertIterationConfig",
    "Fact",
    "Hypothesis",
    "IterationResult",
    "LLMClient",
    "LLMContextMDP",
    "LLMResponseError",
    "MCTSConfig",
    "MathAction",
    "MathState",
    "MockEmbeddingClient",
    "MockLLMClient",
    "OllamaEmbeddingClient",
    "OpenAICompatibleEmbeddingClient",
    "OpenAICompatibleLLMClient",
    "PolicyTarget",
    "Subgoal",
    "TabularPolicy",
    "TransformersEmbeddingClient",
    "TransformersLLMClient",
    "Verb",
    "VerifierCall",
    "available_actions",
    "best_action",
    "evaluate_policy",
    "evaluate_with_search",
    "mcts_search",
    "parse_state",
    "policy_as_prior",
    "root_q_estimates",
    "run_expert_iteration",
    "serialize_state",
    "visit_count_policy",
]
