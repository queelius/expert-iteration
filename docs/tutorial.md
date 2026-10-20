# A Small Learning-and-Search Example

The original idea is easiest to see without a language model.

The agent moves through this grid:

```text
S..G
.#.P
....
```

`S` is the start, `G` is the goal, `P` is a pit, and `#` is a wall. Moves are
noisy, so choosing a direction does not guarantee moving that way.
The environment assigns +1 at the goal, -1 at the pit, and -0.01 on ordinary
steps. Later rewards are discounted by 0.99 per step. This classical example's
environment rewards are separate from the LLM experiments' symbolic reward.

The state is the agent's location. The actions are movement directions.
The policy stores preferences for those directions at each location.

## What Happens

1. Start with a policy that has no useful preferences.
2. Use MCTS to explore possible moves. PUCT uses the current policy's
   preferences while keeping less-tried choices in consideration.
3. Record how often search visits each move.
4. Train the policy to resemble those visit proportions.
5. Repeat with the updated policy.

This is expert iteration: search supplies improved training targets to a
policy. The project is RL first, MCTS second; this particular teaching loop
shows search-assisted policy learning, not the unfinished off-policy Q-learning
experiment proposed for the LLM controller.

## Run It

After installing the package as described in [reproduction](reproduction.md):

```bash
python experiments/run_expert_iteration.py \
  --iterations 2 --episodes 4 --simulations 8 \
  --evaluation-episodes 4 --max-steps 20 \
  --evaluation-max-steps 20 --seed 2026
```

The command runs two small learning rounds. The original, longer demonstration
is still available by omitting the budget arguments.

The output reports:

- **Mean return:** the average discounted reward from the sampled episodes.
- **Mean length:** how many moves those episodes took.
- **Training examples:** the recorded search decisions used for learning.
- **Fast mode:** the trained policy chooses moves without search.
- **Search mode:** the policy guides additional search at each move.
- **Optimal value:** value iteration's exact reference, up to its numerical
  stopping tolerance, for this small known environment.

A small sampled mean can lie above or below the exact expected value by
chance. Learning need not improve every round. This tiny demonstration
illustrates mechanics; it does not establish a sample-efficiency advantage,
and it says nothing by itself about LLM performance.

If comparing learning efficiency later, count the work spent generating
search targets as well as the training updates and evaluation calls.
