"""verifier_anchored_rl: Verifier-anchored RL with MCTS expert iteration.

Phase 1 implements the classical-MDP scaffold: finite-MDP and generative-MDP
protocols, a stochastic Gridworld environment, an exact value-iteration
solver, a depth-limited expectimax planner, and an MCTS planner whose
chance-node Q estimates serve as sample-efficient Bellman targets.

Phase 2 adds the expert-iteration loop: a tabular softmax policy, a
training loop that distills MCTS-improved visit-count distributions back
into the policy, and evaluation utilities for fast mode (no search) and
search mode (MCTS at action selection).

Phase 3 adds the Deliberation MDP in ``deliberation_mdp``, over an
immutable AND-OR proof forest (``reductio``): ``DeliberationState``,
``DeliberationAction``, and a six-verb ``Verb`` enum (``SPLIT``,
``FORM_HYPOTHESIS``, ``COMPUTE``, ``VERIFY``, ``COMMIT``, ``ABANDON``).
``VERIFY``/``COMMIT``/``ABANDON`` never call the LLM, so the policy can
neither author what is checked nor what it is checked against; the other
three verbs generate content via the ``llm_client`` abstraction
(``MockLLMClient`` for tests, ``EchoLLMClient`` for zero-config smoke
checks, ``OpenAICompatibleLLMClient`` for real vLLM/Ollama/llama.cpp HTTP
backends). ``lenses`` renders scoped, sanitized views and verb suffixes
for each LLM call. The symbolic verifier lives in ``verifier_adapter``
(``MathVerifier`` / ``check_answer``). ``llm_controller`` is the Rung-1
LLM-as-controller runner, whose JSONL trajectory logs feed Phase-B
behavior cloning. LLM-extension dependencies (``httpx``, ``sympy``) are
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
from .deliberation_mdp import (
    INTERMEDIATE_VERIFIER_REWARD,
    MASKED_ACTION_REWARD,
    TERMINAL_FAILURE_REWARD,
    TERMINAL_SUCCESS_REWARD,
    DeliberationAction,
    DeliberationMDP,
    DeliberationState,
    Verb,
    available_actions,
)
from .lenses import global_prefix, render_view, sanitize
from .llm_controller import EpisodeSummary, LLMController, run_episode
from .verifier_adapter import MathVerifier, check_answer
from .mcts import (
    ChanceNode,
    DecisionNode,
    MCTSConfig,
    best_action,
    mcts_search,
    root_q_estimates,
    visit_count_policy,
)
from .outer_loop import IterationStats, collect_iteration, run_outer_loop
from .planner_actor import Encoder, PrefillCache, StateActor
from .policies import TabularPolicy, policy_as_prior
from .transformers_client import (
    TransformersEmbeddingClient,
    TransformersLLMClient,
)

__version__ = "0.1.0"

__all__ = [
    "ChanceNode",
    "DecisionNode",
    "DeliberationAction",
    "DeliberationMDP",
    "DeliberationState",
    "EchoLLMClient",
    "EmbeddingClient",
    "EmbeddingError",
    "Encoder",
    "EpisodeSummary",
    "ExpertIterationConfig",
    "INTERMEDIATE_VERIFIER_REWARD",
    "IterationResult",
    "IterationStats",
    "LLMClient",
    "LLMController",
    "LLMResponseError",
    "MASKED_ACTION_REWARD",
    "MCTSConfig",
    "MathVerifier",
    "MockEmbeddingClient",
    "MockLLMClient",
    "OllamaEmbeddingClient",
    "OpenAICompatibleEmbeddingClient",
    "OpenAICompatibleLLMClient",
    "PolicyTarget",
    "PrefillCache",
    "StateActor",
    "TERMINAL_FAILURE_REWARD",
    "TERMINAL_SUCCESS_REWARD",
    "TabularPolicy",
    "TransformersEmbeddingClient",
    "TransformersLLMClient",
    "Verb",
    "available_actions",
    "best_action",
    "check_answer",
    "collect_iteration",
    "evaluate_policy",
    "evaluate_with_search",
    "global_prefix",
    "mcts_search",
    "policy_as_prior",
    "render_view",
    "root_q_estimates",
    "run_episode",
    "run_expert_iteration",
    "run_outer_loop",
    "sanitize",
    "visit_count_policy",
]
