# What This Research Was About

## The Original Question

Can a learned policy help search spend its effort on better choices, and can
the choices found by search make that policy better?

A policy is a rule for choosing what to do next. Monte Carlo Tree Search
(MCTS) tries possible continuations before making a choice. PUCT is its rule
for balancing promising choices against choices it has not explored enough.
A policy supplies initial preferences to that rule. Search can then produce
training examples for the policy.

The repository's framing is **RL first, MCTS second**. Learning is the central
aim; search is an optional accelerator. Better search guidance need not mean
permanently pruning branches. It usually means spending fewer visits on them.
The [teaching example](tutorial.md) demonstrates the learning/search loop in a
small grid environment. It does not establish a new method or improved LLM
training efficiency.

## The Later Language-Model Experiments

The project then explored an external controller directing a frozen language
model. "Frozen" means that we did not train the worker model's weights.

The controller saw a tree-like workspace containing a main problem,
subproblems, and proposed answers. Its choices were pairs: an operation and
the node to apply it to. Operations included decomposition, answering,
revision, and submission, with different menus in different versions.
A "lens" was the selected portion of the workspace shown to the worker.

We tried using an LLM as the controller and copying its decisions into a small
neural network. We also compared fixed workflows and different prompts.
Symbolic checking supplied the recorded task reward. Proposed answers and
accepted workspace entries were not automatically verified facts.

## What Was and Was Not Tested

| Component | Evidence boundary |
| --- | --- |
| Tabular learning and MCTS | Implemented, tested, and runnable locally |
| Typed reasoning forests and permissive workspaces | Implemented, with several historical versions |
| LLM teacher control | Tested in small model-specific experiments |
| Small-controller behavior cloning | Trained and tested in the historical v1 path |
| Off-policy RL controlling the LLM workflow | Not completed in the reported studies |
| Full RL/search sample-efficiency advantage | Not established |
| General superiority of decomposition or learned orchestration | Not established |

Behavior cloning means learning to imitate recorded decisions. It is not the
same experiment as learning better decisions from task rewards. A failed
imitation-transfer test cannot settle whether a different reward-trained
controller would work.

## Why Preserve It?

This archive offers practical checks for similar research: compare against the
worker alone, compare against a strong fixed workflow, test whether generated
work is actually usable, and distinguish a useful context change from evidence
that a controller can learn when to make it.

The [findings](findings.md) are deliberately narrow. This archive is a
reproducible record and teaching resource, not a claim of algorithmic novelty.
The unfinished manuscripts remain historical drafts.
