# Maze follow-up: a small policy and explicit move constraints

Changing from Qwen's ASCII input to a small model reading the grid directly was not sufficient: with the original 256 training states, it still reached **0/10 goals**. More training examples from the same training layouts improved that to **5/10**. Explicitly excluding moves through walls or boundaries improved it to **7/10**. Breadth-first search still solved **10/10**.

These are exploratory results on the ten previously inspected synthetic 4x4 mazes. They are not real-world navigation validation, and no latency comparison is claimed.

## What changed

The policy reads three 4x4 one-hot channels: walls, agent position and goal position. A small neural network maps the 48 inputs through two 128-unit GELU layers to four action scores. It does not generate text, search a path or receive distances. It has no memory of previous actions.

We used the original 32 training wall layouts and compared two declared training budgets:

- **Original states:** the same 256 labeled states used by the old Qwen classifier experiment.
- **More task supervision:** all 4,992 distinct agent/goal pairs on those same 32 layouts. Breadth-first search generates the training targets, with equal target probability for every optimal move in a tie.

All other wall layouts remain disjoint from training. No test or episode layout is used to generate training examples. This larger training set changes both the number of examples and the coverage of goals within each training layout; it is not an equal-supervision comparison.

Both models use AdamW with learning rate 0.002, weight decay 0.01, seed 913 and 600 minibatch steps of size 256. Checkpoints at 100, 300 and 600 steps are ranked by unmasked goal success on the 16 tuning layouts, then by optimal-action accuracy on the 128 tuning states. Both selected checkpoint 600. Every checkpoint outcome is saved.

Each selected model is evaluated twice with the same weights. The **legal mask** sets impossible action scores to negative infinity using only visible walls and grid boundaries. It neither knows which legal move is optimal nor prevents a loop. Without a mask, illegal moves leave the agent in place. Every episode has the original start and a 20-move limit.

## Results

| Policy | Optimal state decisions / 128 | Legal state decisions / 128 | Goals reached / 10 | Episode actions | Illegal episode actions |
|---|---:|---:|---:|---:|---:|
| Original-state MLP | 88 | 104 | 0 | 200 | 100 |
| Original-state MLP + legal mask | 109 | 128 | 2 | 170 | 0 |
| More-supervision MLP | 110 | 119 | 5 | 126 | 78 |
| More-supervision MLP + legal mask | 117 | 128 | 7 | 99 | 0 |
| Breadth-first search | 128 | 128 | 10 | 59 | 0 |

The old frozen Qwen full and early policies both reached 0/10 goals, each making 115 illegal moves out of 200. This is historical context: Qwen reads ASCII rather than structured channels, and its architecture and training differ. The old saved records contain only each classifier's winning label and confidence, so we could not reconstruct a valid Qwen legal-mask experiment. It has not been tested here.

The more-supervision model with a legal mask still fails three mazes through legal loops. That illustrates why good single-action accuracy is insufficient: one repeated mistake can prevent an entire episode from finishing. There is no early-stopping experiment in this follow-up; its purpose is to first improve the underlying task policy.

## What this supports

For a fully observed small static maze, search remains the dependable solution. The learned policy improved when its training covered more situations and deterministic constraints blocked impossible actions. That supports investigating task-specific representations, richer supervision and explicit action constraints before spending effort on an early-exit gate.

It does not establish general navigation competence, Pac-Man play, visual perception, planning with moving enemies or generalization to larger mazes. Ten reused episodes and one training seed provide only a smoke test. The next meaningful test would freeze the policy and evaluate newly generated layouts and larger grids; a memoryless policy may still need a planning or state-tracking component.

## Evidence

- [Experiment code](../experiments/maze_smoke.py)
- [Protocol and executed source hash](../results/maze-smoke/protocol.json)
- [All results](../results/maze-smoke/result.json)
- [Training and checkpoint selection history](../results/maze-smoke/history.json)
- [Every episode action](../results/maze-smoke/episodes.json)
- [Every state decision](../results/maze-smoke/states.json)
- [Original-state weights](../results/maze-smoke/original_states.npz), [more-supervision weights](../results/maze-smoke/all_train_goal_pairs.npz)
- [Earlier Qwen maze study](maze-actions.md)

Run `python experiments/maze_smoke.py` to reproduce from the saved original maze protocol. It uses two CPU threads and does not load Qwen or modify the parent experiment.
