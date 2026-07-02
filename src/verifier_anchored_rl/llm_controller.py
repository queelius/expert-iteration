"""Rung 1: an LLM as the controller (no trained network).

A controller-teacher LLM sees the rendered global state plus the legal
action menu and picks one (verb, target). The small frozen generator
still fills all content inside the MDP; the teacher only decides. The
JSONL trajectory log produced by run_episode is the Phase-B behavior
cloning dataset (spec section 9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .deliberation_mdp import (
    DeliberationAction, DeliberationMDP, DeliberationState, Verb,
)
from .lenses import global_prefix
from .llm_client import LLMClient, LLMResponseError
from .reductio import get

CONTROLLER_SYSTEM = (
    "You are the controller of a mathematical deliberation. The state is a "
    "forest of goals, hypotheses, and facts; each turn you pick exactly one "
    "action from the legal menu. The episode ends when you commit the root "
    "goal (commit:0); you succeed only if the committed evidence claims the "
    "correct final answer.\n"
    "\n"
    "Verb semantics (actions are verb:node-id):\n"
    "- split: break a goal into subgoals (rarely needed for short problems)\n"
    "- form_hypothesis: add a tentative claim under a goal\n"
    "- compute: add ONE checkable equation as a derived fact under a goal, "
    'for example "6*7 = 42" (pure math, no prose)\n'
    "- verify: run the symbolic checker on that node's own statement; only "
    "verified nodes count as evidence\n"
    "- commit: close a goal using its most recent verified evidence; "
    "commit:0 ends the episode and is scored on the evidence's "
    "right-hand-side value\n"
    "- abandon: mark a dead-end node; abandon:0 gives up the episode\n"
    "\n"
    "Play efficiently: compute the equation whose right-hand side is the "
    "final answer, verify that fact, then commit the root as soon as "
    "commit:0 is on the menu. The checker only accepts GROUND equations "
    '(no unknowns): "(13 - 5)/2 = 4" verifies, "2*x = 8" does not. If a '
    "node shows (verify failed), reformulate with a new compute instead of "
    "retrying it. Never add a duplicate of an existing fact. Return only "
    "the requested JSON object."
)


def _menu(mdp: DeliberationMDP, state: DeliberationState
          ) -> list[tuple[str, DeliberationAction]]:
    out = []
    for a in mdp.actions(state):
        out.append((f"{a.verb.value}:{a.target}", a))
    return out


@dataclass
class LLMController:
    client: LLMClient
    system: str = CONTROLLER_SYSTEM
    last_choice_fallback: bool = False
    # Decision-making wants near-greedy sampling; the menu is small and the
    # cost of a bad pick is a wasted turn, so keep the temperature low.
    temperature: float = 0.2
    max_tokens: int = 64

    def choose(self, mdp: DeliberationMDP,
               state: DeliberationState) -> DeliberationAction:
        menu = _menu(mdp, state)
        if not menu:
            raise ValueError("no legal actions; state is terminal")
        by_key = dict(menu)
        lines = []
        for key, a in menu:
            stmt = get(state.graph, a.target).statement
            lines.append(f"  {key}  (target: {stmt})")
        user = (
            f"{global_prefix(state)}\n\n"
            f"Verb: CONTROLLER_CHOICE\n\n"
            f"Legal actions:\n" + "\n".join(lines) + "\n\n"
            'Produce a JSON object with this field:\n'
            '  - "action": exactly one of the menu keys above\n'
            "Return only the JSON object."
        )
        schema = {"type": "object",
                  "properties": {"action": {"type": "string",
                                            "enum": [k for k, _ in menu]}},
                  "required": ["action"]}
        try:
            resp = self.client.complete_json(
                self.system, user, schema,
                temperature=self.temperature, max_tokens=self.max_tokens)
            chosen = by_key.get(str(resp.get("action")))
        except LLMResponseError:
            chosen = None
        self.last_choice_fallback = chosen is None
        return chosen if chosen is not None else menu[0][1]


@dataclass(frozen=True)
class EpisodeSummary:
    reward: float
    turns: int
    llm_calls: int


def run_episode(mdp: DeliberationMDP, controller: LLMController, rng,
                log_fn: Callable[[dict], None] | None = None, *,
                episode_id: int = 0, meta: dict | None = None
                ) -> EpisodeSummary:
    state = mdp.initial_state()
    total = 0.0
    calls = 0
    while not mdp.is_terminal(state):
        legal = mdp.actions(state)
        if not legal:
            break
        action = controller.choose(mdp, state)
        calls += 1
        next_state, reward = mdp.sample_transition(state, action, rng)
        total += reward
        if log_fn is not None:
            log_fn({
                "episode": episode_id,
                "problem": mdp.problem,
                "turn": state.turn,
                "state_render": global_prefix(state),
                "legal": [f"{a.verb.value}:{a.target}" for a in legal],
                "chosen": f"{action.verb.value}:{action.target}",
                "reward": reward,
                "terminal": mdp.is_terminal(next_state),
                "fallback": controller.last_choice_fallback,
            })
        state = next_state
    if log_fn is not None:
        log_fn({
            "episode": episode_id,
            "summary": True,
            "total_reward": total,
            "turns": state.turn,
            "llm_calls": calls,
            "meta": meta or {},
        })
    return EpisodeSummary(reward=total, turns=state.turn, llm_calls=calls)
