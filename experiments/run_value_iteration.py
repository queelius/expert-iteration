"""Smoke-test script: solve the example Gridworld with value iteration.

Builds the layout from design.md (section 5.1), runs synchronous value
iteration with gamma = 0.99, and prints the value table (as a numeric
grid) plus the greedy policy (as an arrow grid).

Run from the repository root::

    python experiments/run_value_iteration.py
"""

from __future__ import annotations

from verifier_anchored_rl.gridworld import GridAction, GridState, Gridworld
from verifier_anchored_rl.value_iteration import value_iteration

_ARROW = {
    GridAction.UP: "^",
    GridAction.DOWN: "v",
    GridAction.LEFT: "<",
    GridAction.RIGHT: ">",
}


def _format_values(env: Gridworld, V: dict[GridState, float]) -> str:
    """Render the value table as a grid of formatted floats."""
    lines = []
    for r in range(env.n_rows):
        row_cells = []
        for c in range(env.n_cols):
            state = GridState(r, c)
            kind = env.cell_kind(state)
            if kind == "#":
                row_cells.append(" ##### ")
            else:
                row_cells.append(f"{V[state]:+7.3f}")
        lines.append(" ".join(row_cells))
    return "\n".join(lines)


def _format_policy(
    env: Gridworld, policy: dict[GridState, GridAction]
) -> str:
    """Render the greedy policy as a grid of arrows (terminals shown)."""
    lines = []
    for r in range(env.n_rows):
        row_cells = []
        for c in range(env.n_cols):
            state = GridState(r, c)
            kind = env.cell_kind(state)
            if kind == "#":
                row_cells.append("#")
            elif kind == "G":
                row_cells.append("G")
            elif kind == "P":
                row_cells.append("P")
            else:
                action = policy.get(state)
                row_cells.append(_ARROW.get(action, "?") if action else "?")
        lines.append(" ".join(row_cells))
    return "\n".join(lines)


def main() -> None:
    layout = [
        "S..G",
        ".#.P",
        "....",
    ]
    env = Gridworld(layout, step_reward=-0.01, goal_reward=1.0, pit_reward=-1.0)

    print("Layout:")
    print(env.render_layout())
    print()

    V, policy = value_iteration(env, gamma=0.99, theta=1e-6, max_iters=10_000)

    print("Optimal state values V*(s):")
    print(_format_values(env, V))
    print()

    print("Greedy policy pi*(s):")
    print(_format_policy(env, policy))
    print()

    start = env.start
    print(f"V*(start={start}) = {V[start]:+.4f}")
    if start in policy:
        print(f"pi*(start) = {policy[start].name}")


if __name__ == "__main__":
    main()
