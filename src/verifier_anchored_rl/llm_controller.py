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

from .controller_observation import build_controller_observation
from .deliberation_mdp import (
    DeliberationAction, DeliberationMDP, DeliberationState,
    OutcomeCode, Verb,
)
from .llm_client import LLMClient, LLMResponseError

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
    "- commit: close a goal using its most recent verified evidence; a "
    "goal line showing (evidence: node N verified) is ready to commit and "
    "N is what the commit will use; commit:0 ends the episode and is "
    "scored on the evidence's right-hand-side value\n"
    "- abandon: mark a dead-end node; abandon:0 gives up the episode\n"
    "\n"
    "The state is drawn as a tree: children are indented under their "
    "parent and carry a parent= tag showing which goal owns them.\n"
    "\n"
    "Play efficiently: compute the equation whose right-hand side is the "
    "final answer, verify that fact, then commit the root as soon as "
    "commit:0 is on the menu. The checker only accepts GROUND equations "
    '(no unknowns): "(13 - 5)/2 = 4" verifies, "2*x = 8" does not. If a '
    "node shows (verify failed), reformulate with a new compute instead of "
    "retrying it. Never add a duplicate of an existing fact. Return only "
    "the requested JSON object."
)


# A variant that foregrounds the structural verbs. Selected with
# --controller-prompt structured. It teaches WHEN to decompose and prune
# without naming any problem family's algebra: the teacher decides
# structure, the generator does the math. Used for the casework contrast
# batch (docs/superpowers/specs/2026-07-02-casework-problem-generator.md),
# where split and abandon are optimal and the default prompt's "split is
# rarely needed" advice would be actively wrong.
CONTROLLER_SYSTEM_STRUCTURED = (
    "You are the controller of a mathematical deliberation. The state is a "
    "forest of goals, hypotheses, and facts; each turn you pick exactly one "
    "action from the legal menu. The episode ends when you commit the root "
    "goal (commit:0); you succeed only if the committed evidence claims the "
    "correct final answer.\n"
    "\n"
    "Verb semantics (actions are verb:node-id):\n"
    "- split: break a goal into an exhaustive set of cases (subgoals). Use "
    "this FIRST whenever a goal has more than one branch to consider, for "
    "example an equation that can be satisfied in two ways, or a function "
    "defined differently on different ranges. Solving only one branch and "
    "committing loses.\n"
    "- form_hypothesis: add a tentative claim under a goal\n"
    "- compute: add ONE checkable ground equation as a derived fact under a "
    'goal, for example "6*7 = 42" (pure math, no unknowns)\n'
    "- verify: run the symbolic checker on that node's statement; only "
    "verified nodes count as evidence\n"
    "- commit: close a goal using its most recent verified evidence\n"
    "- abandon: mark a dead-end node. Abandon a case whose candidate "
    "solution violates a stated condition of the problem (a range "
    "restriction, a domain, a sign); that branch contributes nothing.\n"
    "\n"
    "The state is drawn as a tree: children are indented under their "
    "parent and carry a parent= tag, so a fact's line tells you exactly "
    "which case goal it belongs to. A goal line showing (evidence: node N "
    "verified) is ready to commit, and node N is what the commit will "
    "use.\n"
    "\n"
    "Play the structure, then close it out. The winning line:\n"
    "1. Split the root ONCE into its cases (a goal that already has "
    "case-children is not offered split again).\n"
    "2. For each case (a child goal): compute its solution as a GROUND "
    'equation, for example the case "4x - 16 = 4" is solved by the ground '
    'fact "(4 + 16)/4 = 5"; verify that fact; then pick that case goal\'s '
    "commit action from the menu to CLOSE the case. A case with a "
    "(verified) fact is finished by committing it, never by computing "
    "more.\n"
    "3. Pick a case goal's abandon action when its solution violates a "
    "stated condition (a range restriction, a domain, a sign).\n"
    "4. Once every case is committed or abandoned, add ONE root-level "
    "compute (compute:0) that establishes the final answer from the case "
    'values: for a sum, "5 + 3 = 8"; for the largest, "max(5, 3) = 5"; for '
    'a lone surviving value, restate it with a fresh expression such as '
    '"max(6, 6) = 6". Verify it, then commit:0. Every episode must end in '
    "commit:0; never let it run out of turns.\n"
    "\n"
    "Efficiency rules. Your action MUST be copied verbatim from the legal "
    "menu. A compute whose statement duplicates a fact already on the SAME "
    "goal is REJECTED and the turn is wasted; if the fact you need already "
    "shows (verified), act on it: commit that case or move to the root "
    "aggregation. The checker only accepts GROUND equations (no unknowns): "
    '"(13 - 5)/2 = 4" verifies, "4x - 16 = 4" and "2*x = 8" do not. If a '
    "node shows (verify failed), reformulate with a new compute instead of "
    "retrying it. Return only the requested JSON object."
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
    last_raw_response: object = None
    last_controller_observation: dict | None = None
    # Decision-making wants near-greedy sampling; the menu is small and the
    # cost of a bad pick is a wasted turn, so keep the temperature low.
    temperature: float = 0.2
    max_tokens: int = 64

    def choose(self, mdp: DeliberationMDP, state: DeliberationState, *,
               rng=None) -> DeliberationAction:
        menu = _menu(mdp, state)
        if not menu:
            raise ValueError("no legal actions; state is terminal")
        observation = build_controller_observation(mdp, state)
        self.last_controller_observation = observation
        feedback = observation["previous_outcome"]
        nodes_by_id = {node["id"]: node for node in observation["nodes"]}
        by_key = dict(menu)
        lines = []
        for key, a in menu:
            stmt = nodes_by_id[a.target]["statement"]
            lines.append(f"  {key}  (target: {stmt})")
        feedback_block = ""
        if feedback is not None and feedback["code"] == "rejected":
            feedback_block = (
                f'\n\nYour previous action "{feedback["action"]}" was '
                f'REJECTED ({feedback["reason"]}). Do not repeat it; '
                "choose a different action."
            )
        state_text = observation["state_text"]
        user = (
            f"{state_text}"
            f"{feedback_block}\n\n"
            f"Verb: CONTROLLER_CHOICE\n\n"
            f"Legal actions:\n" + "\n".join(lines) + "\n\n"
            'Produce a JSON object with this field:\n'
            '  - "action": exactly one of the menu keys above\n'
            "Return only the JSON object."
        )
        # No enum in the wire schema, on purpose: the client VALIDATES
        # against the schema and retries on mismatch, so an enum turns
        # every illegal-action intent (for example commit on a case with
        # no verified witness) into two wasted corrective API calls and an
        # opaque parse error. A plain string lets the reply reach the menu
        # lookup below, where a wrong key is logged with its content and
        # handled by the fallback policy.
        schema = {"type": "object",
                  "properties": {"action": {"type": "string"}},
                  "required": ["action"]}
        try:
            resp = self.client.complete_json(
                self.system, user, schema,
                temperature=self.temperature, max_tokens=self.max_tokens)
            self.last_raw_response = resp
            # Normalize before lookup: whitespace and case slips must not
            # turn a well-intended key into a fallback.
            key = str(resp.get("action", "")).strip()
            chosen = by_key.get(key)
            if chosen is None:
                folded = {k.casefold(): a for k, a in menu}
                chosen = folded.get(key.casefold())
        except LLMResponseError as exc:
            self.last_raw_response = (
                exc.raw_response
                if exc.raw_response is not None
                else repr(exc)
            )
            chosen = None
        self.last_choice_fallback = chosen is None
        if chosen is not None:
            return chosen
        # Fallback policy for an unusable reply: prefer a legal VERIFY
        # (cheap, informative, never destructive), else a uniform-random
        # SAFE action; menu[0] only as the deterministic last resort.
        # menu[0] was the OLD default, and it silently manufactured
        # form_hypothesis floods in the casework probes. Irreversible root
        # actions are excluded from the random pool: a fallback must never
        # end the episode (abandon:0 forfeits; commit:0 gambles on -1). A
        # probe episode was observed losing exactly this way: perfect play
        # for six turns, then a random-fallback abandon:0.
        verifies = [a for k, a in menu if k.startswith("verify")]
        if verifies:
            return verifies[0]
        root = state.graph.root_id
        safe = [a for _, a in menu
                if not (a.target == root
                        and a.verb in (Verb.ABANDON, Verb.COMMIT))]
        if rng is not None and safe:
            return safe[int(rng.integers(len(safe)))]
        return (safe or [a for _, a in menu])[0]


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
        action = controller.choose(mdp, state, rng=rng)
        calls += 1
        next_state, reward = mdp.sample_transition(state, action, rng)
        total += reward
        chosen_key = f"{action.verb.value}:{action.target}"
        masked = (
            next_state.previous_outcome.code is OutcomeCode.REJECTED
        )
        transition_debug = getattr(mdp, "last_transition_debug", None)
        if log_fn is not None:
            rec = {
                "episode": episode_id,
                "problem": mdp.problem,
                "turn": state.turn,
                "state_render": controller.last_controller_observation["state_text"],
                "controller_observation": controller.last_controller_observation,
                "legal": [f"{a.verb.value}:{a.target}" for a in legal],
                "chosen": chosen_key,
                "reward": reward,
                "masked": masked,
                "terminal": mdp.is_terminal(next_state),
                "fallback": controller.last_choice_fallback,
                "outcome": next_state.previous_outcome.as_record(),
            }
            if transition_debug is not None:
                rec["transition_debug"] = transition_debug
            if controller.last_choice_fallback:
                # The raw reply is the only diagnostic for WHY a fallback
                # happened; discarding it made probe 4 undiagnosable.
                rec["raw_response"] = controller.last_raw_response
            policy_record = getattr(controller, "last_policy_record", None)
            if policy_record is not None:
                rec["policy"] = policy_record
            log_fn(rec)
        state = next_state
    if log_fn is not None:
        summary_meta = dict(meta or {})
        summary_meta.update({
            "observation_schema": (
                controller.last_controller_observation["schema_version"]
                if controller.last_controller_observation is not None
                else None
            ),
            "mdp_contract": mdp.contract_record(),
        })
        log_fn({
            "episode": episode_id,
            "summary": True,
            "total_reward": total,
            "turns": state.turn,
            "llm_calls": calls,
            "meta": summary_meta,
        })
    return EpisodeSummary(reward=total, turns=state.turn, llm_calls=calls)
