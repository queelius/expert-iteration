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
    "You are the controller of a mathematical deliberation. Read the state "
    "and pick exactly one next action from the menu. Return only the "
    "requested JSON object."
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
            resp = self.client.complete_json(self.system, user, schema)
            chosen = by_key.get(str(resp.get("action")))
        except LLMResponseError:
            chosen = None
        return chosen if chosen is not None else menu[0][1]


@dataclass(frozen=True)
class EpisodeSummary:
    reward: float
    turns: int
    llm_calls: int


def run_episode(mdp: DeliberationMDP, controller: LLMController, rng,
                log_fn: Callable[[dict], None] | None = None
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
                "problem": mdp.problem,
                "turn": state.turn,
                "state_render": global_prefix(state),
                "legal": [f"{a.verb.value}:{a.target}" for a in legal],
                "chosen": f"{action.verb.value}:{action.target}",
                "reward": reward,
                "terminal": mdp.is_terminal(next_state),
            })
        state = next_state
    return EpisodeSummary(reward=total, turns=state.turn, llm_calls=calls)
